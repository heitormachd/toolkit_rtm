"""Run recorded-data TR/RTM: uv run python -m scripts.real_workflow."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, gausspulse, hilbert, resample_poly, sosfiltfilt

from acoustics_imaging.ReverseTimeMigration import ReverseTimeMigration
from acoustics_imaging.SyntheticReverseTimeMigration import _normalize_by_source_energy
from acoustics_imaging.TimeReversal import TimeReversal
from acoustics_imaging.InputTest import InputTest
from acoustics_imaging.paths import (
    ACUDE_DATA_DIR,
    PROJECT_ROOT,
    REAL_SIMULATION_OUTPUT_DIR,
)

# 'panther' ou 'acude'
dados = 'panther'
PANTHER_DATA_DIR = PROJECT_ROOT / 'panther_data'
PANTHER_ACQUISITIONS = (
    'immersion_fmc_rot_Xdeg.m2k',
    'meia_lua_fmc.m2k',
)
# Panther sets this to each acquisition's complete directory name, including .m2k.
OUTPUT_SUBFOLDER_NAME = None
ENABLE_POYNTING_VECTORS = True
GENERATE_VIDEO = False
ANIMATION_STEP = 300

# Scalar acoustic models: immersion water/steel and direct-contact specimen.
# The immersion time window covers the notebook's ROI down to 72 mm,
# excluding the later water-path multiples in the 100 us recording.
PANTHER_SETTINGS = {
    'immersion_fmc_rot_Xdeg.m2k': {
        'spacing': 0.05e-3, 'width': 64e-3, 'depth': 80e-3,
        'time_upsample': 2, 'end_time_us': 60.0, 'source_delay_us': 0.9,
        'imaging_stride': 8,
        'spatial_order': 8,
        # The notebook's holes are at 47–72 mm; omit the strong front-wall echo.
        'mute_before_us': 40.0,
    },
    'meia_lua_fmc.m2k': {
        'spacing': 0.1e-3, 'width': 180e-3, 'depth': 100e-3,
        'time_upsample': 1, 'end_time_us': 30.0, 'source_delay_us': 0.5,
        'imaging_stride': 4,
    },
}
PML_CELLS = 45  # Matches SimulationConfig; put the array inside the physical grid.


def prepare_panther(acquisition):
    input_test = InputTest()
    data = input_test.load_data_panther(PANTHER_DATA_DIR / acquisition)
    settings = PANTHER_SETTINGS[acquisition]
    spacing = np.float32(settings['spacing'])
    margin = PML_CELLS + 5
    nx = round(settings['width'] / float(spacing)) + 2 * margin
    nz = round(settings['depth'] / float(spacing)) + 2 * margin
    x = (np.arange(nx) - (nx - 1) / 2) * float(spacing)
    z = (np.arange(nz) - margin) * float(spacing)
    c = np.full((nz, nx), data.specimen_params.cl, dtype=np.float32)
    surface = None
    if acquisition == 'immersion_fmc_rot_Xdeg.m2k':
        # The acquisition's contact label is wrong; follow immersion_TFM.ipynb:
        # https://colab.research.google.com/drive/1vLUmfbmtpuQCr2o68tJ2GWtZq1v0cce_
        from surface.surface import Surface, SurfaceType
        data.inspection_params.type_insp = 'immersion'
        fitted = Surface(data, 0, keep_data_insp=False)
        fitted.fit(surf_type=SurfaceType.LINE_LS)
        surface = [float(fitted.surfaceparam.a), float(fitted.surfaceparam.b) * 1e-3]
        c[z[:, None] < surface[0] * x[None, :] + surface[1]] = (
            data.inspection_params.coupling_cl
        )

    dt = np.float32(input_test.dt / settings['time_upsample'])
    courant = float(c.max() * dt * np.sqrt(2) / spacing)
    spatial_order = settings.get('spatial_order', 2)
    if spatial_order == 8:
        courant *= 1225/1024 + 245/3072 + 49/5120 + 5/7168
    if courant >= 1:
        raise ValueError(f'Unstable Panther grid: Courant number {courant:.3f} >= 1.')
    probe_x = data.probe_params.elem_center[:, 0] * 1e-3
    microphone_x = np.rint((probe_x - x[0]) / spacing).astype(np.int32)
    microphone_z = np.full(input_test.microphones_amount, margin, dtype=np.int32)
    config = {
        'dt': dt, 'c': c, 'dz': spacing, 'dx': spacing,
        'grid_size_z': np.int32(nz), 'grid_size_x': np.int32(nx),
        'input_test': input_test, 'microphone_x': microphone_x,
        'microphone_z': microphone_z,
        # Correlate every 32 ns (15.625 MHz Nyquist for the 2–6 MHz product).
        'imaging_stride': settings['imaging_stride'],
        'spatial_order': spatial_order,
    }
    metadata = {
        'acquisition': acquisition, 'inspection_type': data.inspection_params.type_insp,
        'elements': int(input_test.microphones_amount),
        'pitch_m': float(input_test.microphones_distance),
        'recording_dt_s': float(input_test.dt), 'simulation_dt_s': float(dt),
        'spacing_m': float(spacing), 'grid_shape': [nz, nx],
        'specimen_speed_m_s': float(data.specimen_params.cl),
        'coupling_speed_m_s': float(data.inspection_params.coupling_cl),
        'surface_z_equals_a_x_plus_b_m': surface, 'courant': courant,
        'bandpass_hz': [2e6, 6e6], **settings,
        'spatial_order': spatial_order,
        'source_note': 'Nominal 5 MHz Gaussian; delay estimated from early diagonal pulses, not calibrated.',
    }
    return input_test, config, metadata, x, z


def prepare_emitter(input_test, config, settings, emitter):
    # select_fmc_emitter always starts from the original recording and pads its gate.
    input_test.select_fmc_emitter(emitter)
    recording_dt = float(input_test.dt)
    end = min(input_test.total_time, round(settings['end_time_us'] * 1e-6 / recording_dt))
    bscan = input_test.bscan[:, :end]
    sos = butter(4, [2e6, 6e6], btype='bandpass', fs=1 / recording_dt, output='sos')
    bscan = sosfiltfilt(sos, bscan, axis=1).astype(np.float32)
    upsample = settings['time_upsample']
    if upsample > 1:
        bscan = resample_poly(bscan, upsample, 1, axis=1).astype(np.float32)
    time = np.arange(bscan.shape[1]) * float(config['dt'])
    # Remove launch/cross-talk and the direct wave, with a 0.3 us taper.
    distance = abs(np.arange(input_test.microphones_amount) - emitter) * input_test.microphones_distance
    coupling_speed = float(config['c'][config['microphone_z'][emitter], config['microphone_x'][emitter]])
    mute_end = distance / coupling_speed + 2e-6
    mute_end = np.maximum(mute_end, settings.get('mute_before_us', 0.0) * 1e-6)
    taper = np.clip((time[None, :] - mute_end[:, None]) / 0.3e-6, 0, 1)
    bscan *= (0.5 - 0.5 * np.cos(np.pi * taper)).astype(np.float32)
    input_test.bscan = bscan
    input_test.total_time = np.int32(bscan.shape[1])
    config['total_time'] = input_test.total_time
    # Do not overwrite the shared synthetic source.npy (its sample interval differs).
    config['source'] = gausspulse(
        time - settings['source_delay_us'] * 1e-6, fc=5e6, bw=0.5
    ).astype(np.float32)


def run_panther(acquisition, emitters=None, prepare_only=False,
                use_poynting_vectors=ENABLE_POYNTING_VECTORS,
                generate_video=GENERATE_VIDEO, animation_step=ANIMATION_STEP,
                output_root=REAL_SIMULATION_OUTPUT_DIR):
    global OUTPUT_SUBFOLDER_NAME
    OUTPUT_SUBFOLDER_NAME = acquisition
    output_dir = Path(output_root) / OUTPUT_SUBFOLDER_NAME
    input_test, config, metadata, x, z = prepare_panther(acquisition)
    selected = list(range(input_test.microphones_amount)) if emitters is None else list(dict.fromkeys(emitters))
    if not selected or any(i < 0 or i >= input_test.microphones_amount for i in selected):
        raise ValueError(f'Emitters must be between 0 and {input_test.microphones_amount - 1}.')
    config['output_dir'] = output_dir
    metadata['emitters'] = selected
    metadata.update(use_poynting_vectors=use_poynting_vectors, generate_video=generate_video,
                    animation_step=animation_step, poynting_angle_degrees=120,
                    poynting_direction_storage='float16 unit vectors',
                    pressure_time='half step' if use_poynting_vectors else 'full step')
    print(json.dumps(metadata, indent=2), flush=True)
    if prepare_only:
        prepare_emitter(input_test, config, metadata, selected[0])
        print(f'Prepared B-scan {input_test.bscan.shape}; no simulation or files written.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    # One normalization factor for the acquisition preserves relative shot amplitudes.
    scale = float(np.max(np.abs(input_test.bscan_fmc)))
    if scale == 0 or not np.isfinite(scale):
        raise ValueError('The FMC recording has no finite, nonzero amplitude.')
    metadata['amplitude_scale'] = scale
    (output_dir / 'run_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.save(output_dir / 'velocity_model.npy', config['c'])
    np.save(output_dir / 'x_m.npy', x)
    np.save(output_dir / 'z_m.npy', z)
    stacked = np.zeros_like(config['c'])
    energy_sum = np.zeros_like(stacked)
    poynting_sum = np.zeros_like(stacked) if use_poynting_vectors else None
    for emitter in selected:
        print(f'{acquisition}: emitter {emitter}/{input_test.microphones_amount - 1}', flush=True)
        prepare_emitter(input_test, config, metadata, emitter)
        input_test.bscan /= scale
        np.save(output_dir / 'source.npy', config['source'])
        rtm_sim = ReverseTimeMigration(**config)
        result = rtm_sim.run(generate_video=generate_video, animation_step=animation_step,
                             use_poynting_vectors=use_poynting_vectors)
        if not np.all(np.isfinite(result)) or not np.any(result):
            raise RuntimeError(f'Emitter {emitter} produced a nonfinite or zero RTM image.')
        stacked += result
        energy_sum += rtm_sim.source_energy
        if use_poynting_vectors:
            poynting_sum += rtm_sim.poynting_image
        del rtm_sim

    folder = output_dir / 'ReverseTimeMigration'
    data_folder = folder / 'data'
    data_folder.mkdir(parents=True, exist_ok=True)
    np.save(data_folder / 'accumulated_product_fmc.npy', stacked)
    np.save(data_folder / 'accumulated_source_energy_fmc.npy', energy_sum)
    np.save(data_folder / 'fmc_transmitters.npy', np.asarray(selected, dtype=np.int32))
    standard_normalized = _normalize_by_source_energy(stacked, energy_sum)
    np.save(data_folder / 'accumulated_product_normalized_fmc.npy', standard_normalized)
    roi = (slice(PML_CELLS, -PML_CELLS), slice(PML_CELLS, -PML_CELLS))
    extent = [x[PML_CELLS] * 1e3, x[-PML_CELLS-1] * 1e3,
              z[-PML_CELLS-1] * 1e3, z[PML_CELLS] * 1e3]
    emitter_label = f'{len(selected)} emitter' + ('s' if len(selected) != 1 else '')
    images = [standard_normalized[roi]]
    titles = ['Normalized Standard RTM']
    if use_poynting_vectors:
        poynting_normalized = _normalize_by_source_energy(poynting_sum, energy_sum)
        np.save(data_folder / 'accumulated_product_poynting_fmc.npy', poynting_sum)
        np.save(data_folder / 'accumulated_product_poynting_normalized_fmc.npy', poynting_normalized)
        images.append(poynting_normalized[roi])
        titles.append('Normalized Poynting RTM')
    envelopes = [np.abs(hilbert(data, axis=0)) for data in images]
    # One reference preserves amplitude differences between imaging conditions.
    reference = max(float(data.max()) for data in envelopes) or 1.0
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 7), layout='constrained')
    for ax, envelope, title in zip(np.atleast_1d(axes), envelopes, titles):
        db = 20 * np.log10(np.maximum(envelope / reference, 1e-6))
        im = ax.imshow(db, extent=extent, cmap='inferno', vmin=-60, vmax=0,
                       interpolation='none', aspect='equal')
        ax.set(title=title, xlabel='x (mm)', ylabel='z (mm)')
    fig.suptitle(f'{acquisition}: {emitter_label}, {input_test.microphones_amount} receivers')
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(),
                 label='Envelope (dB relative to shared maximum)', shrink=0.8)
    fig.savefig(folder / 'fmc_comparison.png', dpi=180)
    plt.close(fig)
    envelope = np.abs(hilbert(stacked[roi], axis=0))
    db = 20 * np.log10(np.maximum(envelope / envelope.max(), 1e-6))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(db, extent=[x[PML_CELLS] * 1e3, x[-PML_CELLS-1] * 1e3,
                              z[-PML_CELLS-1] * 1e3, z[PML_CELLS] * 1e3],
                   vmin=-60, vmax=0, cmap='inferno', aspect='equal')
    surface = metadata['surface_z_equals_a_x_plus_b_m']
    if surface is not None:
        ax.plot(x[roi[1]] * 1e3, (surface[0] * x[roi[1]] + surface[1]) * 1e3, 'c--', linewidth=0.8)
    ax.set(xlabel='x (mm)', ylabel='z (mm)', title=f'{acquisition}: Standard RTM ({emitter_label})')
    fig.colorbar(im, ax=ax, label='Envelope (dB relative to maximum)')
    fig.savefig(folder / 'rtm_fmc.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved RTM stack to {folder}', flush=True)


def main():
    if dados == 'acude':
        input_test = InputTest()
        input_test.load_data_acude(ACUDE_DATA_DIR / 'azulPerpendicular1_Variables.mat', resampled=True)
        input_test.process_bscan()
        output_dir = REAL_SIMULATION_OUTPUT_DIR / OUTPUT_SUBFOLDER_NAME if OUTPUT_SUBFOLDER_NAME else REAL_SIMULATION_OUTPUT_DIR
        config = {
            'dt': input_test.dt, 'c': np.full((1000, 4500), 1500, dtype=np.float32),
            'dz': np.float32(0.01), 'dx': np.float32(0.01),
            'grid_size_z': np.int32(1000), 'grid_size_x': np.int32(4500),
            'total_time': input_test.total_time, 'output_dir': output_dir,
            'input_test': input_test, 'padding_zeros': np.int32(0),
        }
        TimeReversal(**config).run(generate_video=False, animation_step=15)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=PANTHER_ACQUISITIONS, default=PANTHER_ACQUISITIONS)
    parser.add_argument('--emitters', nargs='+', type=int, help='Zero-based emitter indices; default: all.')
    parser.add_argument('--prepare-only', action='store_true', help='Validate data/model/preprocessing without running WebGPU.')
    parser.add_argument('--poynting', action=argparse.BooleanOptionalAction, default=ENABLE_POYNTING_VECTORS)
    parser.add_argument('--generate-video', action=argparse.BooleanOptionalAction, default=GENERATE_VIDEO,
                        help='Save TR and RTM frames/videos for the last processed emitter.')
    parser.add_argument('--animation-step', type=int, default=ANIMATION_STEP)
    parser.add_argument('--output-root', type=Path, default=REAL_SIMULATION_OUTPUT_DIR,
                        help='Use a separate output root to preserve earlier reconstructions.')
    args = parser.parse_args()
    if args.animation_step < 1:
        parser.error('--animation-step must be positive.')
    for acquisition in args.datasets:
        run_panther(acquisition, args.emitters, args.prepare_only, args.poynting,
                    args.generate_video, args.animation_step, args.output_root)


if __name__ == '__main__':
    main()
