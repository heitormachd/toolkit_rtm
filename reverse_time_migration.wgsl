struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    sources_amount: i32,
    source_time: i32,
    i: i32,
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
var<storage,read> source: array<f32>;

@group(0) @binding(3)
var<storage,read> c: array<f32>;

@group(0) @binding(4)
var<storage,read_write> p_future: array<f32>;

@group(0) @binding(5)
var<storage,read_write> p_present: array<f32>;

@group(0) @binding(6)
var<storage,read_write> p_past: array<f32>;

@group(0) @binding(7)
var<storage,read_write> dp_1_z: array<f32>;

@group(0) @binding(8)
var<storage,read_write> dp_1_x: array<f32>;

@group(0) @binding(9)
var<storage,read_write> dp_2_z: array<f32>;

@group(0) @binding(10)
var<storage,read_write> dp_2_x: array<f32>;

@group(0) @binding(11)
var<storage,read_write> psi_z: array<f32>;

@group(0) @binding(12)
var<storage,read_write> psi_x: array<f32>;

@group(0) @binding(13)
var<storage,read_write> phi_z: array<f32>;

@group(0) @binding(14)
var<storage,read_write> phi_x: array<f32>;

@group(0) @binding(15)
var<storage,read> absorption_z: array<f32>;

@group(0) @binding(16)
var<storage,read> absorption_x: array<f32>;

@group(0) @binding(17)
var<storage,read> is_z_absorption: array<i32>;

@group(0) @binding(18)
var<storage,read> is_x_absorption: array<i32>;

@group(0) @binding(19)
var<storage,read_write> p_future_flipped_tr: array<f32>;

@group(0) @binding(20)
var<storage,read_write> p_present_flipped_tr: array<f32>;

@group(0) @binding(21)
var<storage,read_write> p_past_flipped_tr: array<f32>;

@group(0) @binding(22)
var<storage,read_write> dp_1_z_flipped_tr: array<f32>;

@group(0) @binding(23)
var<storage,read_write> dp_1_x_flipped_tr: array<f32>;

@group(0) @binding(24)
var<storage,read_write> dp_2_z_flipped_tr: array<f32>;

@group(0) @binding(25)
var<storage,read_write> dp_2_x_flipped_tr: array<f32>;

@group(0) @binding(26)
var<storage,read_write> psi_z_flipped_tr: array<f32>;

@group(0) @binding(27)
var<storage,read_write> psi_x_flipped_tr: array<f32>;

@group(0) @binding(28)
var<storage,read_write> phi_z_flipped_tr: array<f32>;

@group(0) @binding(29)
var<storage,read_write> phi_x_flipped_tr: array<f32>;

@group(0) @binding(30)
var<storage,read> absorption_z_flipped_tr: array<f32>;

@group(0) @binding(31)
var<storage,read> absorption_x_flipped_tr: array<f32>;

@group(0) @binding(32)
var<storage,read> is_z_absorption_flipped_tr: array<i32>;

@group(0) @binding(33)
var<storage,read> is_x_absorption_flipped_tr: array<i32>;

@group(0) @binding(34)
var<storage,read_write> v_z_present: array<f32>;

@group(0) @binding(35)
var<storage,read_write> v_x_present: array<f32>;

@group(0) @binding(36)
var<storage,read_write> v_z_present_flipped_tr: array<f32>;

@group(0) @binding(37)
var<storage,read_write> v_x_present_flipped_tr: array<f32>;

@group(0) @binding(38)
var<storage,read_write> rtm_poynting_image: array<f32>;

@group(0) @binding(39)
var<storage,read> source_zx: array<i32>;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn forward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (z + 1 < infoI32.grid_size_z) {
        dp_1_z[zx(z, x)] = (p_present[zx(z + 1, x)] - p_present[zx(z, x)]) / infoF32.dz;
    }
    if (x + 1 < infoI32.grid_size_x) {
        dp_1_x[zx(z, x)] = (p_present[zx(z, x + 1)] - p_present[zx(z, x)]) / infoF32.dx;
    }

    if (z + 1 < infoI32.grid_size_z) {
        dp_1_z_flipped_tr[zx(z, x)] = (p_present_flipped_tr[zx(z + 1, x)] - p_present_flipped_tr[zx(z, x)]) / infoF32.dz;
    }
    if (x + 1 < infoI32.grid_size_x) {
        dp_1_x_flipped_tr[zx(z, x)] = (p_present_flipped_tr[zx(z, x + 1)] - p_present_flipped_tr[zx(z, x)]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn backward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (z - 1 >= 0) {
        dp_2_z[zx(z, x)] = (dp_1_z[zx(z, x)] - dp_1_z[zx(z - 1, x)]) / infoF32.dz;
    }
    if (x - 1 >= 0) {
        dp_2_x[zx(z, x)] = (dp_1_x[zx(z, x)] - dp_1_x[zx(z, x - 1)]) / infoF32.dx;
    }

    if (z - 1 >= 0) {
        dp_2_z_flipped_tr[zx(z, x)] = (dp_1_z_flipped_tr[zx(z, x)] - dp_1_z_flipped_tr[zx(z - 1, x)]) / infoF32.dz;
    }
    if (x - 1 >= 0) {
        dp_2_x_flipped_tr[zx(z, x)] = (dp_1_x_flipped_tr[zx(z, x)] - dp_1_x_flipped_tr[zx(z, x - 1)]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_forward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        phi_z[zx(z, x)] = absorption_z[zx(z, x)] * phi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * dp_1_z[zx(z, x)];
        dp_1_z[zx(z, x)] += phi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        phi_x[zx(z, x)] = absorption_x[zx(z, x)] * phi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * dp_1_x[zx(z, x)];
        dp_1_x[zx(z, x)] += phi_x[zx(z, x)];
    }

    if (is_z_absorption_flipped_tr[zx(z, x)] == 1) {
        phi_z_flipped_tr[zx(z, x)] = absorption_z_flipped_tr[zx(z, x)] * phi_z_flipped_tr[zx(z, x)] + (absorption_z_flipped_tr[zx(z, x)] - 1) * dp_1_z_flipped_tr[zx(z, x)];
        dp_1_z_flipped_tr[zx(z, x)] += phi_z_flipped_tr[zx(z, x)];
    }
    if (is_x_absorption_flipped_tr[zx(z, x)] == 1) {
        phi_x_flipped_tr[zx(z, x)] = absorption_x_flipped_tr[zx(z, x)] * phi_x_flipped_tr[zx(z, x)] + (absorption_x_flipped_tr[zx(z, x)] - 1) * dp_1_x_flipped_tr[zx(z, x)];
        dp_1_x_flipped_tr[zx(z, x)] += phi_x_flipped_tr[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_backward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        psi_z[zx(z, x)] = absorption_z[zx(z, x)] * psi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * dp_2_z[zx(z, x)];
        dp_2_z[zx(z, x)] += psi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        psi_x[zx(z, x)] = absorption_x[zx(z, x)] * psi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * dp_2_x[zx(z, x)];
        dp_2_x[zx(z, x)] += psi_x[zx(z, x)];
    }

    if (is_z_absorption_flipped_tr[zx(z, x)] == 1) {
        psi_z_flipped_tr[zx(z, x)] = absorption_z_flipped_tr[zx(z, x)] * psi_z_flipped_tr[zx(z, x)] + (absorption_z_flipped_tr[zx(z, x)] - 1) * dp_2_z_flipped_tr[zx(z, x)];
        dp_2_z_flipped_tr[zx(z, x)] += psi_z_flipped_tr[zx(z, x)];
    }
    if (is_x_absorption_flipped_tr[zx(z, x)] == 1) {
        psi_x_flipped_tr[zx(z, x)] = absorption_x_flipped_tr[zx(z, x)] * psi_x_flipped_tr[zx(z, x)] + (absorption_x_flipped_tr[zx(z, x)] - 1) * dp_2_x_flipped_tr[zx(z, x)];
        dp_2_x_flipped_tr[zx(z, x)] += psi_x_flipped_tr[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim_flipped_tr(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_future_flipped_tr[zx(z, x)] = (c[zx(z, x)] * c[zx(z, x)]) * (dp_2_z_flipped_tr[zx(z, x)] + dp_2_x_flipped_tr[zx(z, x)]) * (infoF32.dt * infoF32.dt);

    p_future_flipped_tr[zx(z, x)] += ((2. * p_present_flipped_tr[zx(z, x)]) - p_past_flipped_tr[zx(z, x)]);

    p_past_flipped_tr[zx(z, x)] = p_present_flipped_tr[zx(z, x)];
    p_present_flipped_tr[zx(z, x)] = p_future_flipped_tr[zx(z, x)];
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_future[zx(z, x)] = (c[zx(z, x)] * c[zx(z, x)]) * (dp_2_z[zx(z, x)] + dp_2_x[zx(z, x)]) * (infoF32.dt * infoF32.dt);

    p_future[zx(z, x)] += ((2. * p_present[zx(z, x)]) - p_past[zx(z, x)]);

    for (var source_index: i32 = 0; source_index < infoI32.sources_amount; source_index = source_index + 1) {
        let source_z = source_zx[source_index];
        let source_x = source_zx[source_index + infoI32.sources_amount];

        if (z == source_z && x == source_x)
        {
            p_future[zx(z, x)] += source[source_index * infoI32.source_time + infoI32.i];
        }
    }

    p_past[zx(z, x)] = p_present[zx(z, x)];
    p_present[zx(z, x)] = p_future[zx(z, x)];
}

@compute
@workgroup_size(wsz, wsx)
fn update_velocity(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let idx = zx(z, x);

    if (idx == -1) { return; }

    v_z_present[idx] = v_z_present[idx] - infoF32.dt * dp_1_z[idx];
    v_x_present[idx] = v_x_present[idx] - infoF32.dt * dp_1_x[idx];

    v_z_present_flipped_tr[idx] = v_z_present_flipped_tr[idx] - infoF32.dt * dp_1_z_flipped_tr[idx];
    v_x_present_flipped_tr[idx] = v_x_present_flipped_tr[idx] - infoF32.dt * dp_1_x_flipped_tr[idx];
}

@compute
@workgroup_size(wsz, wsx)
fn update_rtm_image(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);
    let idx = zx(z, x);

    if (idx == -1) { return; }

    let pd = p_present[idx];

    let vdz = v_z_present[idx];

    let pu = p_present_flipped_tr[idx];

    let vuz = v_z_present_flipped_tr[idx];

    let sdz = -pd * vdz;
    let suz = -pu * vuz; 

    let pdpu = pd * pu;

    var ic = 0.0;
    if (sdz > 0.0 && suz < 0.0) {
        ic = 1.0;
    }
    rtm_poynting_image[idx] = rtm_poynting_image[idx] + (pdpu * ic);
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
