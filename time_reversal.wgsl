struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    microphones_amount: i32,
    i: i32,
    samples_per_microphone: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
    dt: f32,
};

@group(0) @binding(0)
var<storage,read_write> infoI32: InfoInt;

@group(0) @binding(1)
var<storage,read> infoF32: InfoFloat;

@group(0) @binding(2)
var<storage,read> c: array<f32>;

@group(0) @binding(3)
var<storage,read> microphone_zx: array<i32>;

@group(0) @binding(5)
var<storage,read_write> p_future: array<f32>;

@group(0) @binding(6)
var<storage,read_write> p_present: array<f32>;

@group(0) @binding(7)
var<storage,read_write> p_past: array<f32>;

@group(0) @binding(8)
var<storage,read_write> dp_1: array<f32>;

@group(0) @binding(10)
var<storage,read_write> dp_2: array<f32>;

@group(0) @binding(12)
var<storage,read_write> psi: array<f32>;

@group(0) @binding(14)
var<storage,read_write> phi: array<f32>;

@group(0) @binding(16)
var<storage,read> absorption: array<f32>;

@group(0) @binding(18)
var<storage,read> is_absorption: array<i32>;

@group(0) @binding(20)
var<storage,read> flipped_bscan: array<f32>;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    return x + z * infoI32.grid_size_x;
}

fn x_field_index(index: i32) -> i32 {
    return (infoI32.grid_size_z * infoI32.grid_size_x) + index;
}

fn microphone_x_index(index: i32) -> i32 {
    return infoI32.microphones_amount + index;
}

@compute
@workgroup_size(wsz, wsx)
fn forward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let grid_index: i32 = zx(z, x);

    if (z + 1 < infoI32.grid_size_z) {
        dp_1[grid_index] = (p_present[zx(z + 1, x)] - p_present[grid_index]) / infoF32.dz;
    }
    if (x + 1 < infoI32.grid_size_x) {
        dp_1[x_field_index(grid_index)] = (p_present[zx(z, x + 1)] - p_present[grid_index]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn backward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let grid_index: i32 = zx(z, x);
    let x_grid_index: i32 = x_field_index(grid_index);

    if (z - 1 >= 0) {
        dp_2[grid_index] = (dp_1[grid_index] - dp_1[zx(z - 1, x)]) / infoF32.dz;
    }
    if (x - 1 >= 0) {
        dp_2[x_grid_index] = (dp_1[x_grid_index] - dp_1[x_field_index(zx(z, x - 1))]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_forward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let grid_index: i32 = zx(z, x);
    let x_grid_index: i32 = x_field_index(grid_index);

    if (is_absorption[grid_index] == 1) {
        phi[grid_index] = absorption[grid_index] * phi[grid_index] + (absorption[grid_index] - 1) * dp_1[grid_index];
        dp_1[grid_index] += phi[grid_index];
    }
    if (is_absorption[x_grid_index] == 1) {
        phi[x_grid_index] = absorption[x_grid_index] * phi[x_grid_index] + (absorption[x_grid_index] - 1) * dp_1[x_grid_index];
        dp_1[x_grid_index] += phi[x_grid_index];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_backward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let grid_index: i32 = zx(z, x);
    let x_grid_index: i32 = x_field_index(grid_index);

    if (is_absorption[grid_index] == 1) {
        psi[grid_index] = absorption[grid_index] * psi[grid_index] + (absorption[grid_index] - 1) * dp_2[grid_index];
        dp_2[grid_index] += psi[grid_index];
    }
    if (is_absorption[x_grid_index] == 1) {
        psi[x_grid_index] = absorption[x_grid_index] * psi[x_grid_index] + (absorption[x_grid_index] - 1) * dp_2[x_grid_index];
        dp_2[x_grid_index] += psi[x_grid_index];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let grid_index: i32 = zx(z, x);
    let x_grid_index: i32 = x_field_index(grid_index);

    p_future[grid_index] = (c[grid_index] * c[grid_index]) * (dp_2[grid_index] + dp_2[x_grid_index]) * (infoF32.dt * infoF32.dt);

    p_future[grid_index] += ((2. * p_present[grid_index]) - p_past[grid_index]);

    for (var microphone_index: i32 = 0; microphone_index < infoI32.microphones_amount; microphone_index += 1)
    {
        if (z == microphone_zx[microphone_index] && x == microphone_zx[microphone_x_index(microphone_index)])
        {
            let bscan_index: i32 = microphone_index * infoI32.samples_per_microphone + infoI32.i;
            p_future[grid_index] += flipped_bscan[bscan_index];
        }
    }

    p_past[grid_index] = p_present[grid_index];
    p_present[grid_index] = p_future[grid_index];
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
