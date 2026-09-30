"""Compare four RTM boundary settings, standard and Poynting, on fixed full FMC."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts import run_td_rtm_fmc as base

ROOT = base.SYNTHETIC_SIMULATION_OUTPUT_DIR / 'td_rtm_boundary_comparison_20260918'
# True means a nonabsorbing pressure-release plane. Top always has CPML.
CASES = {
    'reflect_sides_bottom': (True, True),
    'reflect_bottom': (False, True),
    'reflect_sides': (True, False),
    'absorb_all': (False, False),
}
TITLES = ('Reflect sides + bottom', 'Reflect bottom only', 'Reflect sides only', 'Absorb all')
ROI = (slice(480, 801), slice(240, 561))  # z=24..40 mm, x=-8..8 mm
SHADER_SOURCE = (base.SHADERS_DIR / 'time_reversal.wgsl').read_text()


class BoundaryPropagation(base.TimeReversal):
    reflect_sides = True
    reflect_bottom = True

    def setup_gpu(self):
        if self.reflect_sides:
            self.is_x_absorption_int[:] = 0
        if self.reflect_bottom:
            self.is_z_absorption_int[-self.absorption_layer_size:] = 0
        super().setup_gpu()


def install_boundaries(root, case):
    sides, bottom = CASES[case]
    shader = SHADER_SOURCE
    planes = []
    if sides:
        shader = shader.replace('x >= 3 &&', 'x >= 4 &&')
        shader = shader.replace('x >= 4 && x + 3', 'x >= 5 && x + 3')
        shader = shader.replace('x + 4 < infoI32.grid_size_x', 'x + 4 < infoI32.grid_size_x - 1')
        shader = shader.replace('x + 3 < infoI32.grid_size_x', 'x + 3 < infoI32.grid_size_x - 1')
        planes.append('x == 0 || x == infoI32.grid_size_x - 1')
    if bottom:
        shader = shader.replace('z + 4 < infoI32.grid_size_z', 'z + 4 < infoI32.grid_size_z - 1')
        shader = shader.replace('z + 3 < infoI32.grid_size_z', 'z + 3 < infoI32.grid_size_z - 1')
        planes.append('z == infoI32.grid_size_z - 1')
    if planes:
        anchor = '    p_past[grid_index] = p_present[grid_index];'
        assert shader.count(anchor) == 1
        shader = shader.replace(anchor, '    if (' + ' || '.join(planes) + ') {\n'
                                '        p_future[grid_index] = 0.0;\n    }\n' + anchor)
    folder = root / 'shaders' / case
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'time_reversal.wgsl').write_text(shader)
    BoundaryPropagation.reflect_sides = sides
    BoundaryPropagation.reflect_bottom = bottom
    base.TR_MODULE.SHADERS_DIR = folder
    base.RTM_MODULE.TimeReversal = BoundaryPropagation


def count_shots(root):
    return sum(1 for _ in root.glob('*/*/shots/tx_*.npz'))


def run_shot(folder, case, tx, background, recording, source, status, root):
    started = time.monotonic()
    rz = np.full(64, base.PAD + 1, dtype=np.int32)
    rx = np.arange(85, 716, 10, dtype=np.int32)
    muted = np.array(recording[tx], dtype=np.float32, copy=True)
    if not np.isfinite(muted).all() or not np.any(muted != 0, axis=1).all():
        raise RuntimeError(f'Invalid input FMC shot {tx}.')
    distance = np.abs(rx.astype(float) - rx[tx]) * float(base.DX)
    stops = np.ceil(distance / (float(base.WATER) * float(base.DT))).astype(int) + 100
    base._apply_direct_arrival_mute(muted, stops, taper_samples=50)
    status.update(current_case=case, current_transmitter=tx, stage='migration', updated_unix=time.time())
    base.write_json(root / 'status.json', status)
    with TemporaryDirectory(prefix='work_', dir=folder) as temporary:
        cfg = base.config(background, base.make_input(muted, tx), rz, rx, Path(temporary))
        cfg['source'] = source
        rtm = base.ReverseTimeMigration(**cfg)
        standard = rtm.run(False, base.STEPS + 1, use_poynting_vectors=True)[base.PAD:].copy()
        poynting = rtm.poynting_image[base.PAD:].copy()
        energy = rtm.source_energy[base.PAD:].copy()
        del rtm
    for label, field in [('standard', standard), ('poynting', poynting), ('energy', energy)]:
        if not np.isfinite(field).all() or not np.any(field):
            raise RuntimeError(f'{case} TX{tx}: invalid {label}.')
    sides, bottom = CASES[case]
    for field in (standard, poynting, energy):
        if sides:
            assert np.all(field[:, 0] == 0) and np.all(field[:, -1] == 0)
        if bottom:
            assert np.all(field[-1] == 0)
    target = folder / 'shots' / f'tx_{tx:02d}.npz'
    temporary = target.with_suffix('.tmp')
    elapsed = time.monotonic() - started
    with temporary.open('wb') as handle:
        np.savez(handle, standard=standard, poynting=poynting, source_energy=energy,
                 transmitter=tx, elapsed_seconds=elapsed)
    temporary.replace(target)
    gc.collect()
    return elapsed


def stack_case(folder):
    standard = np.zeros((1041, 801), np.float32)
    poynting = np.zeros_like(standard)
    energy = np.zeros_like(standard)
    completed = []
    for path in sorted((folder / 'shots').glob('tx_*.npz')):
        with np.load(path) as saved:
            standard += saved['standard']
            poynting += saved['poynting']
            energy += saved['source_energy']
            completed.append(int(saved['transmitter']))
    np.save(folder / 'standard_raw.npy', standard)
    np.save(folder / 'poynting_raw.npy', poynting)
    np.save(folder / 'source_energy.npy', energy)
    np.save(folder / 'standard_normalized.npy', base._normalize_by_source_energy(standard, energy))
    np.save(folder / 'poynting_normalized.npy', base._normalize_by_source_energy(poynting, energy))
    np.save(folder / 'transmitters.npy', np.array(completed, np.int32))
    base.write_json(folder / 'status.json', dict(completed_transmitters=len(completed),
                    total_transmitters=64, state='completed' if len(completed) == 64 else 'partial'))
    return completed


def render_comparison(root, model, defect):
    """A common scale across all eight panels; ground truth is evaluation only."""
    loaded = {}
    for case in CASES:
        folder = root / model / case
        if (folder / 'transmitters.npy').exists():
            loaded[case] = (np.load(folder / 'standard_normalized.npy'),
                            np.load(folder / 'poynting_normalized.npy'),
                            np.load(folder / 'transmitters.npy'))
    values = [np.abs(data[ROI]).ravel() for pair in loaded.values() for data in pair[:2]]
    limit = float(np.percentile(np.concatenate(values), 99.5)) if values else 1.
    limit = limit or 1.
    distances = distance_transform_edt(~defect, sampling=(.05, .05))[ROI]
    near = distances <= .25
    background = distances >= 1.
    metrics = {}
    for crop in (True, False):
        fig, axes = plt.subplots(4, 2, figsize=(10, 15), layout='constrained')
        for row, (case, title) in enumerate(zip(CASES, TITLES)):
            for column, condition in enumerate(('standard', 'poynting')):
                ax = axes[row, column]
                if case not in loaded:
                    ax.set_title(f'{title} / {condition}: pending')
                    ax.axis('off')
                    continue
                field = loaded[case][column]
                count = len(loaded[case][2])
                plotted = ax.imshow(field, extent=(-20, 20, 52, 0), cmap='seismic',
                                    vmin=-limit, vmax=limit, interpolation='none')
                ax.set(title=f'{title}\n{condition.capitalize()} — {count}/64 TX',
                       xlabel='x (mm)', ylabel='z (mm)')
                if crop:
                    ax.set(xlim=(-8, 8), ylim=(40, 24))
                    ax.contour(np.linspace(-20,20,801), np.linspace(0,52,1041), defect,
                               levels=[.5], colors='black', linewidths=.4, linestyles='dashed')
                    data = field[ROI].astype(np.float64)
                    signal_rms = float(np.sqrt(np.mean(data[near]**2)))
                    background_rms = float(np.sqrt(np.mean(data[background]**2)))
                    metrics.setdefault(case, {})[condition] = dict(
                        transmitters=count, near_crack_rms=signal_rms, background_rms=background_rms,
                        near_to_background_db=float(20*np.log10(max(signal_rms, 1e-30) /
                                                               max(background_rms, 1e-30))))
        tx_sets = {tuple(pair[2].tolist()) for pair in loaded.values()}
        progress_note = ' | Partial: transmitter sets differ' if len(tx_sets) > 1 else ''
        fig.suptitle(f'{model}: fixed FMC, four migration boundary conditions\n'
                     f'Shared amplitude scale across all panels{progress_note}')
        if loaded:
            fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=.65, label='Illumination-normalized amplitude')
        fig.savefig(root / model / ('comparison_roi.png' if crop else 'comparison_full.png'), dpi=150)
        plt.close(fig)
    base.write_json(root / model / 'metrics.json', dict(
        roi_mm=dict(x=[-8,8], z=[24,40]), near_crack_distance_mm=.25,
        background_min_distance_mm=1., description='RMS contrast is descriptive, not proof of crack completeness.',
        panels=metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='TX32 of t_b1 in all four boundary cases; reusable by full run.')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--input-dir', type=Path, default=base.ROOT)
    args = parser.parse_args()
    root, inputs = args.output_dir.resolve(), args.input_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((inputs / 'status.json').read_text())
    if previous['state'] != 'completed' or previous['completed_transmitters'] != 320:
        raise RuntimeError('Input full FMC run is incomplete.')
    input_config = json.loads((inputs / 'config.json').read_text())
    for key, value in dict(steps=base.STEPS, imaging_stride=base.STRIDE, spatial_order=8,
                           dx_m=float(base.DX), dz_m=float(base.DZ), dt_s=float(base.DT),
                           top_padding_cells=base.PAD, receivers=64, transmitters_per_model=64).items():
        if input_config[key] != value:
            raise RuntimeError(f'Input configuration mismatch: {key}')
    source = np.load(inputs / 'source.npy')
    if source.shape != (base.STEPS,) or not np.isfinite(source).all():
        raise RuntimeError('Invalid original source waveform.')
    provenance = {}
    for model in base.MODELS:
        for name in ('fmc_raw.npy', 'velocity_migration.npy', 'defect_mask.npy', 'transmitters.npy'):
            path = inputs / model / name
            info = path.stat()
            provenance[str(path.relative_to(inputs))] = [info.st_size, info.st_mtime_ns]
    cfg = dict(input_dir=str(inputs), input_config=input_config, input_file_stats=provenance,
               source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
               cases={k: dict(reflect_sides=v[0], reflect_bottom=v[1]) for k,v in CASES.items()},
               source_and_receiver_boundaries_identical=True, top='45-cell CPML in all cases',
               fixed_acquisition='original FMC generated with reflecting sides and bottom',
               absorbing_edges='original 45-cell CPML inside the unchanged domain; no extra side/bottom padding',
               imaging_conditions=['standard','poynting'], poynting_angle_limit_degrees=120,
               standard_recomputed='Both conditions use half-time pressure in the same Poynting-enabled run.',
               total_shots=1280)
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError('Existing comparison configuration differs; choose a new directory.')
    base.write_json(config_path, cfg)
    (root / 'run.pid').write_text(str(os.getpid())+'\n')
    status = dict(state='running', mode='pilot' if args.pilot else 'full', pid=os.getpid(),
                  started_unix=time.time(), total_migrations=1280, completed_migrations=count_shots(root))
    times = []
    try:
        for model in (base.MODELS[:1] if args.pilot else base.MODELS):
            saved_background = np.load(inputs / model / 'velocity_migration.npy')
            defect = np.load(inputs / model / 'defect_mask.npy')
            if saved_background.shape != (1041,801) or not np.all(saved_background[defect] == base.ALUMINUM):
                raise RuntimeError('Migration background contains defects or wrong dimensions.')
            background = np.pad(saved_background, ((base.PAD,0),(0,0)), constant_values=base.WATER)
            recording = np.load(inputs / model / 'fmc_raw.npy', mmap_mode='r')
            if recording.shape != (64,64,base.STEPS):
                raise RuntimeError('Unexpected FMC shape.')
            np.testing.assert_array_equal(np.load(inputs / model / 'transmitters.npy'), np.arange(64))
            for tx in ([32] if args.pilot else range(64)):
                for case in CASES:
                    folder = root / model / case
                    (folder / 'shots').mkdir(parents=True, exist_ok=True)
                    install_boundaries(root, case)
                    status.update(current_model=model, current_case=case)
                    if not (folder/'shots'/f'tx_{tx:02d}.npz').exists():
                        print(f'RUN {model} {case} TX{tx}', flush=True)
                        times.append(run_shot(folder, case, tx, background, recording, source, status, root))
                    completed = stack_case(folder)
                    # Rendering after each shot makes pilot and partial stacks inspectable.
                    render_comparison(root, model, defect)
                    status.update(completed_migrations=count_shots(root), stage='shot_complete',
                                  current_case_completed=len(completed), updated_unix=time.time())
                    if times:
                        status.update(mean_migration_seconds=float(np.mean(times)),
                            estimated_remaining_hours=float(np.mean(times)*(1280-count_shots(root))/3600))
                    base.write_json(root/'status.json',status)
                    print(json.dumps(status),flush=True)
            if not args.pilot:
                for case in CASES:
                    completed = np.load(root / model / case / 'transmitters.npy')
                    np.testing.assert_array_equal(completed, np.arange(64))
            del recording
        status.update(state='pilot_complete' if args.pilot else 'completed', stage='finished')
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        status['updated_unix']=time.time()
        base.write_json(root/'status.json',status)


if __name__ == '__main__':
    main()
