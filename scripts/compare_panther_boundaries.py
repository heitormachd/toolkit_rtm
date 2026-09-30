"""Four-boundary full-FMC RTM comparison on the fixed Panther immersion recording."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

from scripts import real_workflow as workflow
from scripts import compare_td_rtm_boundaries as boundaries

PROJECT = workflow.PROJECT_ROOT
ROOT = workflow.REAL_SIMULATION_OUTPUT_DIR / 'panther_boundary_comparison_20260924'
ACQUISITION = 'immersion_fmc_rot_Xdeg.m2k'
BASELINE = workflow.REAL_SIMULATION_OUTPUT_DIR / 'timing_corrected_20260916' / ACQUISITION
FIELDS = ('standard', 'poynting', 'source_energy')
write_json = boundaries.base.write_json
normalize = workflow._normalize_by_source_energy


def prepare():
    inp, cfg, metadata, x, z = workflow.prepare_panther(ACQUISITION)
    # Freeze the same fitted surface/grid used by the corrected full-FMC result.
    for name, array in [('velocity_model.npy', cfg['c']), ('x_m.npy', x), ('z_m.npy', z)]:
        np.testing.assert_array_equal(array, np.load(BASELINE / name))
    scale = float(np.max(np.abs(inp.bscan_fmc)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid FMC amplitude scale.')
    metadata.update(amplitude_scale=scale, timing_correction=dict(
        effective_source_delay_us=0., data_padding_us=.9,
        simulation_duration_us=60.9, original_recording_window_us=60.))
    return inp, cfg, metadata, x, z


def prepare_shot(inp, cfg, metadata, tx):
    workflow.prepare_emitter(inp, cfg, metadata, tx)
    shift = round(.9e-6 / float(cfg['dt']))
    inp.bscan = np.pad(inp.bscan, ((0, 0), (shift, 0)))
    inp.bscan /= metadata['amplitude_scale']
    inp.total_time = np.int32(inp.bscan.shape[1])
    cfg['total_time'] = inp.total_time
    if inp.bscan.shape != (64, 15225) or not np.isfinite(inp.bscan).all():
        raise ValueError('Unexpected processed Panther recording.')


def provenance(metadata, x, z):
    paths = [PROJECT / 'scripts' / name for name in
             ('compare_panther_boundaries.py', 'compare_td_rtm_boundaries.py', 'real_workflow.py')]
    paths += list((PROJECT / 'acoustics_imaging').glob('*.py'))
    paths += [PROJECT / 'shaders/time_reversal.wgsl', BASELINE / 'velocity_model.npy']
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raw = workflow.PANTHER_DATA_DIR / ACQUISITION
    files = {str(p.relative_to(raw)): [p.stat().st_size, p.stat().st_mtime_ns]
             for p in sorted(raw.rglob('*')) if p.is_file()}
    return dict(acquisition=metadata, cases=boundaries.CASES, total_migrations=256,
                transmitters=64, receivers=64, imaging_conditions=['standard', 'poynting'],
                pressure_time='half step for both conditions', poynting_angle_degrees=120,
                top='CPML in every case', source_and_receiver_boundaries_identical=True,
                reflecting_planes_m=dict(left=float(x[0]), right=float(x[-1]), bottom=float(z[-1])),
                boundary_note='Pressure-release computational edges, not verified specimen walls. Same grid in every case.',
                normalization='Sum raw images divided by summed source energy; raw stacks also compared.',
                source_hashes=hashes, raw_file_stats=files)


def load_stack(folder, shape):
    sums = {key: np.zeros(shape, np.float32) for key in FIELDS}
    transmitters = []
    for path in sorted((folder / 'shots').glob('tx_*.npz')):
        with np.load(path) as data:
            tx = int(data['transmitter'])
            if path.stem != f'tx_{tx:02d}' or not 0 <= tx < 64:
                raise ValueError(f'Invalid checkpoint transmitter: {path}')
            for key in FIELDS:
                field = data[key]
                if field.shape != shape or not np.isfinite(field).all():
                    raise ValueError(f'Invalid checkpoint: {path} {key}')
                sums[key] += field
        transmitters.append(tx)
    return sums, transmitters


def save_stack(folder, sums, txs):
    folder.mkdir(parents=True, exist_ok=True)
    for key in ('standard', 'poynting'):
        np.save(folder / f'{key}_raw.npy', sums[key])
        np.save(folder / f'{key}_normalized.npy', normalize(sums[key], sums['source_energy']))
    np.save(folder / 'source_energy.npy', sums['source_energy'])
    np.save(folder / 'transmitters.npy', np.array(sorted(txs), np.int32))


def report(root, stacks, x, z):
    ref = np.load(PROJECT / 'outputs/analysis/panther_diagnostics/immersion_tfm_processed.npy').astype(float)
    diagnostic = json.loads((PROJECT / 'outputs/analysis/immersion_validation_20260916/diagnostics.json').read_text())
    peaks = diagnostic['image_metrics']['TFM, processed FMC (64 TX)']['matched_peaks']
    xs = np.linspace(-5, 25, 192, endpoint=False)
    zs = np.linspace(47, 72, 160, endpoint=False)
    xx, zz = np.meshgrid(xs, zs)
    coords = np.stack([zz, xx], axis=-1)
    signal = np.zeros(ref.shape, bool)
    for peak in peaks:
        signal |= (xx - peak['tfm_x_mm'])**2 + (zz - peak['tfm_z_mm'])**2 < 4
    panels = {}
    tx_sets = {case: sorted(txs) for case, (_, txs) in stacks.items()}
    matched = len({tuple(txs) for txs in tx_sets.values()}) == 1
    for form in ('raw', 'normalized'):
        images = {}
        for case, (sums, txs) in stacks.items():
            if not txs:
                continue
            for key in ('standard', 'poynting'):
                field = sums[key] if form == 'raw' else normalize(sums[key], sums['source_energy'])
                envelope = np.abs(hilbert(field, axis=0))
                img = RegularGridInterpolator((z * 1e3, x * 1e3), envelope)(coords)
                images[case, key] = img
                peak_value = max(float(img.max()), 1e-30)
                bg = max(float(np.sqrt(np.mean(img[~signal]**2))), 1e-30)
                near = max(float(np.sqrt(np.mean(img[signal]**2))), 1e-30)
                offsets = []
                for peak in peaks:
                    px, pz = peak['tfm_x_mm'], peak['tfm_z_mm']
                    selected = (abs(xx-px) < .95) & (abs(zz-pz) < .95)
                    i, j = np.unravel_index(np.argmax(np.where(selected, img, -1)), img.shape)
                    offsets.append(dict(tfm_x_mm=px, tfm_z_mm=pz,
                                        dx_mm=float(xx[i,j]-px), dz_mm=float(zz[i,j]-pz),
                                        peak_relative_roi_max=float(img[i,j]/peak_value)))
                panels.setdefault(case, {}).setdefault(key, {})[form] = dict(
                    transmitters=len(txs), envelope_cosine_to_tfm=float(np.sum(img*ref) /
                        max(float(np.linalg.norm(img)*np.linalg.norm(ref)), 1e-30)),
                    background_rms_relative_peak_db=float(20*np.log10(bg/peak_value)),
                    near_to_background_db=float(20*np.log10(near/bg)),
                    median_dx_mm=float(np.median([p['dx_mm'] for p in offsets])),
                    median_dz_mm=float(np.median([p['dz_mm'] for p in offsets])), peaks=offsets)
        shared = max([float(a.max()) for a in images.values()] or [1.]) or 1.
        for scaling in ('shared', 'own'):
            fig, axes = plt.subplots(4, 2, figsize=(10, 16), layout='constrained')
            im = None
            for row, (case, title) in enumerate(zip(boundaries.CASES, boundaries.TITLES)):
                for col, key in enumerate(('standard', 'poynting')):
                    ax = axes[row, col]
                    if (case, key) not in images:
                        ax.set_title(f'{title} / {key}: pending')
                        ax.axis('off')
                        continue
                    img = images[case, key]
                    level = shared if scaling == 'shared' else max(float(img.max()), 1e-30)
                    im = ax.imshow(20*np.log10(np.maximum(img/level, 1e-6)),
                                   extent=(-5-.078125, 25-.078125, 72-.078125, 47-.078125),
                                   cmap='inferno', vmin=-35, vmax=0, aspect='equal')
                    ax.set(title=f'{title} / {key}\n{len(tx_sets[case])}/64 TX', xlabel='x (mm)', ylabel='z (mm)')
            fig.suptitle(f'Panther immersion: {form} RTM, {scaling} ROI amplitude reference\n'
                         + ('Matched transmitter sets' if matched else 'PARTIAL: transmitter sets differ'))
            if im is not None:
                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, label='Envelope (dB)')
            fig.savefig(root / f'comparison_{form}_{scaling}.png', dpi=150)
            plt.close(fig)
    write_json(root / 'metrics.json', dict(panels=panels, transmitters=tx_sets, matched_transmitters=matched,
        roi_mm=dict(x=[-5,25], z=[47,72]), reference='Previously saved processed full-FMC TFM, no registration.',
        limitations='TFM is not independent ground truth. Background includes sidelobes/unlabelled indications. '
        'Peak searches are constrained to within 0.95 mm of TFM positions. Reflectors are computational edges. '
        'The fixed 40–60 us measured window excludes later paths. Compare raw and normalized images.'))


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state='preparing', pid=os.getpid(), total_migrations=256,
                  completed_migrations=len(list(root.glob('*/shots/tx_*.npz'))), started_unix=time.time())
    (root / 'run.pid').write_text(str(os.getpid())+'\n')
    try:
        if args.wait_for_pid:
            # Check process identity as well as PID so a reused PID cannot hold the queue.
            command = Path(f'/proc/{args.wait_for_pid}/cmdline')
            while command.exists() and b'compare_td_rtm_boundaries.py' in command.read_bytes():
                status.update(state='queued', waiting_for_pid=args.wait_for_pid, updated_unix=time.time())
                write_json(root / 'status.json', status)
                time.sleep(30)
        status.update(state='preparing', updated_unix=time.time())
        write_json(root / 'status.json', status)
        inp, cfg, metadata, x, z = prepare()
        configuration = json.loads(json.dumps(provenance(metadata, x, z)))
        config_path = root / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError('Comparison configuration changed; use a new output directory.')
        write_json(config_path, configuration)
        np.save(root / 'velocity_model.npy', cfg['c'])
        np.save(root / 'x_m.npy', x)
        np.save(root / 'z_m.npy', z)
        prepare_shot(inp, cfg, metadata, 32)
        np.save(root / 'source.npy', np.pad(cfg['source'], (0, int(inp.total_time)-len(cfg['source']))))
        status.update(grid_shape=list(cfg['c'].shape), steps=int(inp.total_time),
                      checkpoint_memory_gib=float(((int(inp.total_time)-1)//cfg['imaging_stride']+1)*cfg['c'].size*8/2**30))
        if args.prepare_only:
            status.update(state='prepared')
            return
        stacks = {case: load_stack(root / case, cfg['c'].shape) for case in boundaries.CASES}
        for case, (sums, txs) in stacks.items():
            save_stack(root / case, sums, txs)
        order = [32] if args.pilot else [32] + [tx for tx in range(64) if tx != 32]
        elapsed = []
        for tx in order:
            prepare_shot(inp, cfg, metadata, tx)
            for case in boundaries.CASES:
                sums, completed = stacks[case]
                if tx in completed:
                    continue
                folder = root / case
                (folder / 'shots').mkdir(parents=True, exist_ok=True)
                boundaries.install_boundaries(root, case)
                status.update(state='running', stage='migration', current_case=case,
                              current_transmitter=tx, updated_unix=time.time())
                write_json(root / 'status.json', status)
                print(f'RUN {case} TX{tx}', flush=True)
                started = time.monotonic()
                with TemporaryDirectory(prefix='work_', dir=folder) as temporary:
                    rtm = workflow.ReverseTimeMigration(**{**cfg, 'output_dir': Path(temporary)})
                    fields = dict(standard=rtm.run(False, 300, use_poynting_vectors=True),
                                  poynting=rtm.poynting_image, source_energy=rtm.source_energy)
                    for key, field in fields.items():
                        if not np.isfinite(field).all() or not np.any(field):
                            raise ValueError(f'Invalid {case} TX{tx}: {key}')
                        sides, bottom = boundaries.CASES[case]
                        if sides and (np.any(field[:,0]) or np.any(field[:,-1])):
                            raise ValueError('Reflecting side plane is nonzero.')
                        if bottom and np.any(field[-1]):
                            raise ValueError('Reflecting bottom plane is nonzero.')
                    target = folder / 'shots' / f'tx_{tx:02d}.npz'
                    temp = target.with_suffix('.tmp')
                    duration = time.monotonic()-started
                    with temp.open('wb') as handle:
                        np.savez(handle, **fields, transmitter=tx, elapsed_seconds=duration)
                    temp.replace(target)
                    for key in FIELDS:
                        sums[key] += fields[key]
                    del rtm, fields, field
                completed.append(tx)
                elapsed.append(duration)
                save_stack(folder, sums, completed)
                gc.collect()
                count = sum(len(txs) for _, txs in stacks.values())
                status.update(completed_migrations=count, stage='shot_complete', updated_unix=time.time(),
                              mean_migration_seconds=float(np.mean(elapsed)),
                              estimated_remaining_hours=float(np.mean(elapsed)*(256-count)/3600))
                write_json(root / 'status.json', status)
                print(json.dumps(status), flush=True)
            # Render balanced four-boundary stacks, including the initial TX32 pilot.
            report(root, stacks, x, z)
        if not args.pilot:
            for _, txs in stacks.values():
                np.testing.assert_array_equal(sorted(txs), np.arange(64))
        status.update(state='pilot_complete' if args.pilot else 'completed', stage='finished')
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        status['updated_unix'] = time.time()
        write_json(root / 'status.json', status)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--pilot', action='store_true', help='Only TX32; its checkpoints are reused by a full run.')
    parser.add_argument('--wait-for-pid', type=int, help='Wait for an existing synthetic boundary comparison to exit.')
    run(parser.parse_args())
