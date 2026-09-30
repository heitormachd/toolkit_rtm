"""Full 64 x 64 FMC/RTM of the five paper PNGs using known, crack-free surfaces."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import time
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from acoustics_imaging.InputTest import InputTest
from acoustics_imaging.ReverseTimeMigration import ReverseTimeMigration
from acoustics_imaging.SyntheticReverseTimeMigration import _normalize_by_source_energy
from acoustics_imaging.SyntheticTimeReversal import _apply_direct_arrival_mute
from acoustics_imaging.TimeReversal import TimeReversal
from acoustics_imaging.functions import convert_image_to_matrix
from acoustics_imaging.paths import MODELS_DIR, SHADERS_DIR, SYNTHETIC_SIMULATION_OUTPUT_DIR

MODELS = ('t_b1', 'y_b1', 'y_b2', 'y_b3', 'y_b4')
ROOT = SYNTHETIC_SIMULATION_OUTPUT_DIR / 'td_rtm_full_fmc_20260917'
DX = DZ = np.float32(50e-6)
DT = np.float32(4e-9)
STEPS = 15000
PAD = 50
STRIDE = 4
WATER, ALUMINUM = np.float32(1496), np.float32(6320)
TR_MODULE = importlib.import_module('acoustics_imaging.TimeReversal')
RTM_MODULE = importlib.import_module('acoustics_imaging.ReverseTimeMigration')


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


class TopAbsorbingPropagation(TimeReversal):
    def setup_gpu(self):
        self.is_x_absorption_int[:] = 0
        self.is_z_absorption_int[:] = 0
        self.is_z_absorption_int[:self.absorption_layer_size] = 1
        super().setup_gpu()


def install_boundaries(root):
    """Local shader copy; p=0 at physical side and bottom planes."""
    shader = (SHADERS_DIR / 'time_reversal.wgsl').read_text()
    # Use the existing second-order closure near reflecting planes, as in the
    # validated bottom-reflector experiment. Top remains the original CPML.
    shader = shader.replace('x >= 3 &&', 'x >= 4 &&')
    shader = shader.replace('x >= 4 && x + 3', 'x >= 5 && x + 3')
    shader = shader.replace('x + 4 < infoI32.grid_size_x', 'x + 4 < infoI32.grid_size_x - 1')
    shader = shader.replace('x + 3 < infoI32.grid_size_x', 'x + 3 < infoI32.grid_size_x - 1')
    shader = shader.replace('z + 4 < infoI32.grid_size_z', 'z + 4 < infoI32.grid_size_z - 1')
    shader = shader.replace('z + 3 < infoI32.grid_size_z', 'z + 3 < infoI32.grid_size_z - 1')
    anchor = '    p_past[grid_index] = p_present[grid_index];'
    assert shader.count(anchor) == 1
    shader = shader.replace(anchor, '''    if (x == 0 || x == infoI32.grid_size_x - 1 || z == infoI32.grid_size_z - 1) {
        p_future[grid_index] = 0.0;
    }
''' + anchor)
    folder = root / 'shaders'
    folder.mkdir(exist_ok=True)
    (folder / 'time_reversal.wgsl').write_text(shader)
    TR_MODULE.SHADERS_DIR = folder
    RTM_MODULE.TimeReversal = TopAbsorbingPropagation


def geometry(name):
    parsed, _, _, rz, rx = convert_image_to_matrix(MODELS_DIR / f'td_rtm_{name}.png')
    assert parsed.shape == (1041, 801)
    np.testing.assert_array_equal(rx, np.arange(85, 716, 10))
    np.testing.assert_array_equal(rz, np.ones(64))
    forward = np.where(parsed == 1500, WATER, ALUMINUM).astype(np.float32)
    x = np.linspace(-20., 20., 801)
    z = np.linspace(0., 52., 1041)
    surface = 14 + int(name[-1]) * np.sin(.05 * np.pi * x)
    background = np.where(z[:, None] <= surface, WATER, ALUMINUM).astype(np.float32)
    defect = forward != background
    assert defect.any() and not defect[z < 25].any()
    assert np.all(forward[defect] == WATER) and np.all(background[defect] == ALUMINUM)
    return (np.pad(forward, ((PAD, 0), (0, 0)), constant_values=WATER),
            np.pad(background, ((PAD, 0), (0, 0)), constant_values=WATER),
            rz + PAD, rx, defect)


def source_signal():
    t = np.arange(STEPS) * float(DT)
    duration = 2 / 5e6
    return np.where(t < duration, np.sin(2 * np.pi * 5e6 * t) *
                    np.sin(np.pi * t / duration)**2, 0).astype(np.float32)


def make_input(bscan, tx):
    inp = InputTest()
    inp.bscan = bscan
    inp.microphones_amount = np.int32(bscan.shape[0])
    inp.microphones_distance = np.float32(.0005)
    inp.total_time = np.int32(STEPS)
    inp.fmc_emitter = np.int32(tx)
    return inp


def config(c, inp, rz, rx, folder):
    return dict(c=c, dx=DX, dz=DZ, dt=DT, grid_size_z=np.int32(c.shape[0]),
                grid_size_x=np.int32(c.shape[1]), total_time=np.int32(STEPS),
                input_test=inp, microphone_z=rz, microphone_x=rx,
                spatial_order=8, output_dir=folder, imaging_stride=STRIDE)


def run_shot(folder, tx, forward, background, rz, rx, source, status, root):
    start = time.monotonic()
    raw = np.empty((64, STEPS), np.float32)
    with TemporaryDirectory(prefix='work_', dir=folder) as temporary:
        temporary = Path(temporary)
        status.update(current_transmitter=tx, stage='forward', updated_unix=time.time())
        write_json(root / 'status.json', status)
        inp = make_input(source[None, ::-1], 0)
        sim = TopAbsorbingPropagation(**config(forward, inp, rz[tx:tx+1], rx[tx:tx+1], temporary))
        def record(step, pressure):
            raw[:, step] = pressure[rz, rx]
        sim.run(False, STEPS + 1, wavefield_callback=record)
        del sim
        assert np.isfinite(raw).all() and np.any(raw)
        muted = raw.copy()
        distance = np.abs(rx.astype(float) - rx[tx]) * float(DX)
        stops = np.ceil(distance / (float(WATER) * float(DT))).astype(int) + 100
        _apply_direct_arrival_mute(muted, stops, taper_samples=50)
        status.update(stage='migration', updated_unix=time.time())
        write_json(root / 'status.json', status)
        inp = make_input(muted, tx)
        cfg = config(background, inp, rz, rx, temporary)
        cfg['source'] = source
        rtm = ReverseTimeMigration(**cfg)
        image = rtm.run(False, STEPS + 1, use_poynting_vectors=False)[PAD:].copy()
        energy = rtm.source_energy[PAD:].copy()
        del rtm
    if not np.isfinite(image).all() or not np.any(image) or not np.isfinite(energy).all():
        raise RuntimeError('Invalid RTM image or illumination.')
    recording_path = folder / 'fmc_raw.npy'
    if recording_path.exists():
        fmc = np.load(recording_path, mmap_mode='r+')
    else:
        fmc = np.lib.format.open_memmap(recording_path, mode='w+', dtype=np.float32,
                                       shape=(64, 64, STEPS))
    fmc[tx] = raw
    fmc.flush()
    del fmc
    # Atomically commit a shot only after its FMC traces have been flushed.
    target = folder / 'shots' / f'tx_{tx:02d}.npz'
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, image=image, source_energy=energy, transmitter=tx,
                 elapsed_seconds=time.monotonic()-start)
    temporary.replace(target)
    gc.collect()
    return time.monotonic()-start


def stack_model(folder, defect):
    image = np.zeros((1041, 801), np.float32)
    energy = np.zeros_like(image)
    completed = []
    for shot in sorted((folder / 'shots').glob('tx_*.npz')):
        with np.load(shot) as saved:
            image += saved['image']
            energy += saved['source_energy']
            completed.append(int(saved['transmitter']))
    normalized = _normalize_by_source_energy(image, energy)
    np.save(folder / 'rtm_raw.npy', image)
    np.save(folder / 'source_energy.npy', energy)
    np.save(folder / 'rtm_normalized.npy', normalized)
    np.save(folder / 'transmitters.npy', np.array(completed, np.int32))
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')
    limit = float(np.percentile(np.abs(normalized[480:800, 240:560]), 99.5)) or 1.
    for ax in axes:
        ax.imshow(normalized, extent=(-20, 20, 52, 0), cmap='seismic',
                  vmin=-limit, vmax=limit, interpolation='none')
        ax.set(xlabel='x (mm)', ylabel='z (mm)')
    axes[0].set_title(f'{folder.name}: {len(completed)}/64 TX')
    axes[1].set(xlim=(-8, 8), ylim=(40, 24), title='Defect region; dashed = true boundary')
    axes[1].contour(np.linspace(-20,20,801), np.linspace(0,52,1041), defect,
                    levels=[.5], colors='black', linewidths=.45, linestyles='dashed')
    fig.savefig(folder / 'rtm.png', dpi=160)
    plt.close(fig)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Run only t_b1 TX32; full run reuses this shot.')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(models=list(MODELS), dx_m=float(DX), dz_m=float(DZ), dt_s=float(DT),
               steps=STEPS, duration_us=60, spatial_order=8, imaging_stride=STRIDE,
               velocities_mps=[float(WATER),float(ALUMINUM)], top_padding_cells=PAD,
               top_cpml_cells=45, boundaries='pressure-release sides x=-20,+20 mm and bottom z=52 mm; top CPML',
               source='5 MHz two-cycle Hann-windowed sine; point elements',
               receivers=64, transmitters_per_model=64,
               migration_model='known analytic surface, all buried crack pixels replaced by aluminum; no TFM',
               normalization='sum of cross-correlations divided by summed source illumination, floor=1e-3',
               direct_mute='water travel time + 0.4 us, final 0.2 us sine-squared ramp',
               physics='toolkit constant-density scalar acoustics; no elastic mode conversion')
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError('Existing batch configuration differs; use a separate output directory.')
    write_json(config_path, cfg)
    install_boundaries(root)
    source = source_signal()
    np.save(root / 'source.npy', source)
    (root / 'run.pid').write_text(str(os.getpid())+'\n')
    status = dict(state='running', pid=os.getpid(), mode='pilot' if args.pilot else 'full',
                  started_unix=time.time(), total_transmitters=320)
    times=[]
    try:
        for name in (MODELS[:1] if args.pilot else MODELS):
            folder = root / name
            (folder / 'shots').mkdir(parents=True, exist_ok=True)
            forward, background, rz, rx, defect = geometry(name)
            np.save(folder / 'velocity_forward.npy', forward[PAD:])
            np.save(folder / 'velocity_migration.npy', background[PAD:])
            np.save(folder / 'defect_mask.npy', defect)
            status['current_model'] = name
            for tx in ([32] if args.pilot else range(64)):
                if not (folder/'shots'/f'tx_{tx:02d}.npz').exists():
                    print(f'RUN {name} TX {tx}/63', flush=True)
                    times.append(run_shot(folder, tx, forward, background, rz, rx, source, status, root))
                completed=stack_model(folder, defect)
                count=sum(len(list((root/m/'shots').glob('tx_*.npz'))) for m in MODELS)
                status.update(completed_transmitters=count, current_model_completed=len(completed),
                              stage='shot_complete', updated_unix=time.time())
                if times:
                    status.update(mean_shot_seconds=float(np.mean(times)),
                                  estimated_remaining_hours=float(np.mean(times)*(320-count)/3600))
                write_json(root/'status.json', status)
                print(json.dumps(status), flush=True)
            if not args.pilot:
                np.testing.assert_array_equal(completed, np.arange(64))
                recording = np.load(folder / 'fmc_raw.npy', mmap_mode='r')
                if not np.isfinite(recording).all() or not np.any(recording != 0, axis=2).all():
                    raise RuntimeError(f'{name}: incomplete or invalid full FMC recordings.')
                del recording
        status.update(state='pilot_complete' if args.pilot else 'completed', stage='finished')
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        status['updated_unix']=time.time()
        write_json(root/'status.json',status)


if __name__ == '__main__':
    main()
