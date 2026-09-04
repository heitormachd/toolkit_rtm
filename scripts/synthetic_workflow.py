import gc

import matplotlib.pyplot as plt
import numpy as np

from acoustics_imaging.SyntheticAcouSim import SyntheticAcouSim
from acoustics_imaging.SyntheticReverseTimeMigration import (
    SyntheticReverseTimeMigration,
    _normalize_by_source_energy,
)
from acoustics_imaging.SyntheticTimeReversal import SyntheticTimeReversal
from acoustics_imaging.functions import convert_image_to_matrix, load_source
from acoustics_imaging.paths import MODELS_DIR, SYNTHETIC_RTM_OUTPUT_DIR


IMAGE_PATH = MODELS_DIR / 'poynting_benchmark.png'
ENABLE_POYNTING_VECTORS = True
ENABLE_SPARSE_FMC = True
FMC_TRANSMITTER_STRIDE = 16
FORCE_RECEIVERS_TO_SURFACE = False
SURFACE_RECEIVER_Z = np.int32(1)
FORWARD_TOTAL_TIME = np.int32(2800)
TIME_REVERSAL_TOTAL_TIME = FORWARD_TOTAL_TIME
DIRECT_ARRIVAL_TAPER_SAMPLES = 100
MEDIUM_C = np.float32(1500)
DT = np.float32(3e-7)
DZ = np.float32(1.5e-3)
DX = np.float32(1.5e-3)


def _select_sparse_fmc_transmitters(receiver_count, stride):
    if receiver_count < 1:
        raise ValueError('FMC requires at least one receiver.')
    if stride < 1:
        raise ValueError('FMC_TRANSMITTER_STRIDE must be at least 1.')

    transmitter_indices = np.arange(0, receiver_count, stride, dtype=np.int32)
    if transmitter_indices[-1] != receiver_count - 1:
        transmitter_indices = np.append(
            transmitter_indices,
            np.int32(receiver_count - 1),
        )
    return transmitter_indices


def _direct_arrival_mute_samples(
    transmitter_z,
    transmitter_x,
    receiver_z,
    receiver_x,
    source_duration_samples,
):
    distance = np.hypot(
        (receiver_z - transmitter_z) * DZ,
        (receiver_x - transmitter_x) * DX,
    )
    travel_samples = np.ceil(distance / (MEDIUM_C * DT)).astype(np.int32)
    return travel_samples + source_duration_samples + DIRECT_ARRIVAL_TAPER_SAMPLES


def _save_fmc_results(standard_sum, source_energy_sum, poynting_sum, transmitters):
    standard_normalized = _normalize_by_source_energy(standard_sum, source_energy_sum)

    np.save(SYNTHETIC_RTM_OUTPUT_DIR / 'accumulated_product_fmc.npy', standard_sum)
    np.save(
        SYNTHETIC_RTM_OUTPUT_DIR / 'accumulated_source_energy_fmc.npy',
        source_energy_sum,
    )
    np.save(
        SYNTHETIC_RTM_OUTPUT_DIR / 'accumulated_product_normalized_fmc.npy',
        standard_normalized,
    )
    np.save(SYNTHETIC_RTM_OUTPUT_DIR / 'fmc_transmitters.npy', transmitters)

    images = [standard_normalized]
    titles = ['Normalized Standard RTM - sparse FMC']
    if poynting_sum is not None:
        poynting_normalized = _normalize_by_source_energy(
            poynting_sum,
            source_energy_sum,
        )
        np.save(
            SYNTHETIC_RTM_OUTPUT_DIR / 'accumulated_product_poynting_fmc.npy',
            poynting_sum,
        )
        np.save(
            SYNTHETIC_RTM_OUTPUT_DIR / 'accumulated_product_poynting_normalized_fmc.npy',
            poynting_normalized,
        )
        images.append(poynting_normalized)
        titles.append('Normalized Poynting RTM - sparse FMC')

    display_limit = np.percentile(
        np.abs(np.concatenate([image.ravel() for image in images])),
        99.5,
    )
    if display_limit == 0.0:
        display_limit = 1.0

    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(6 * len(images), 6),
        dpi=180,
        layout='constrained',
    )
    axes = np.atleast_1d(axes)
    for axis, image, title in zip(axes, images, titles):
        plotted = axis.imshow(
            image,
            cmap='seismic',
            vmin=-display_limit,
            vmax=display_limit,
            interpolation='none',
        )
        axis.set_title(title)
        axis.set_xlabel('x (pixel)')
        axis.set_ylabel('z (pixel)')
    fig.colorbar(plotted, ax=axes.tolist(), shrink=0.82, label='Normalized amplitude')
    fig.savefig(SYNTHETIC_RTM_OUTPUT_DIR / 'fmc_comparison.png')
    plt.close(fig)


def _run_sparse_fmc(base_config, receiver_z, receiver_x, source_id):
    transmitter_indices = _select_sparse_fmc_transmitters(
        len(receiver_z),
        FMC_TRANSMITTER_STRIDE,
    )
    source = load_source(source_id, total_time=int(FORWARD_TOTAL_TIME))
    nonzero_samples = np.flatnonzero(source)
    source_duration_samples = int(nonzero_samples[-1]) + 1 if len(nonzero_samples) else 0

    standard_sum = None
    source_energy_sum = None
    poynting_sum = None
    transmitter_positions = np.column_stack((
        receiver_z[transmitter_indices],
        receiver_x[transmitter_indices],
    )).astype(np.int32)

    print(
        f'Running sparse FMC with {len(transmitter_indices)} sequential transmitters '
        f'and {len(receiver_z)} receivers per shot.'
    )
    for shot_number, transmitter_index in enumerate(transmitter_indices):
        transmitter_z = np.int32(receiver_z[transmitter_index])
        transmitter_x = np.int32(receiver_x[transmitter_index])
        print(
            f'FMC shot {shot_number + 1}/{len(transmitter_indices)}: '
            f'transmitter z={int(transmitter_z)}, x={int(transmitter_x)}'
        )

        shot_config = base_config.copy()
        shot_config.update({
            'source_z': np.array([transmitter_z], dtype=np.int32),
            'source_x': np.array([transmitter_x], dtype=np.int32),
            'source_ids': np.array([source_id], dtype=np.int32),
            'emitter_index': shot_number,
        })

        acou_sim = SyntheticAcouSim(**shot_config)
        acou_sim.run(generate_video=False, animation_step=15)

        time_reversal_config = shot_config.copy()
        time_reversal_config['total_time'] = TIME_REVERSAL_TOTAL_TIME
        time_reversal_config['direct_arrival_mute_samples'] = (
            _direct_arrival_mute_samples(
                transmitter_z,
                transmitter_x,
                receiver_z,
                receiver_x,
                source_duration_samples,
            )
        )
        time_reversal_config['direct_arrival_taper_samples'] = (
            DIRECT_ARRIVAL_TAPER_SAMPLES
        )

        tr_sim = SyntheticTimeReversal(**time_reversal_config)
        tr_sim.run(generate_video=False, animation_step=15)
        rtm_sim = SyntheticReverseTimeMigration(**time_reversal_config)
        shot_result = rtm_sim.run(
            generate_video=False,
            animation_step=15,
            use_poynting_vectors=ENABLE_POYNTING_VECTORS,
        )

        if standard_sum is None:
            standard_sum = np.zeros_like(shot_result['standard_raw'])
            source_energy_sum = np.zeros_like(shot_result['source_energy'])
            if ENABLE_POYNTING_VECTORS:
                poynting_sum = np.zeros_like(shot_result['poynting_raw'])
        standard_sum += shot_result['standard_raw']
        source_energy_sum += shot_result['source_energy']
        if ENABLE_POYNTING_VECTORS:
            poynting_sum += shot_result['poynting_raw']

        del acou_sim, tr_sim, rtm_sim, shot_result
        gc.collect()

    _save_fmc_results(
        standard_sum,
        source_energy_sum,
        poynting_sum,
        transmitter_positions,
    )
    print(f'Sparse FMC results saved in {SYNTHETIC_RTM_OUTPUT_DIR}.')


def _run_single_shot(base_config, source_z, source_x, source_ids):
    simulation_config = base_config.copy()
    simulation_config.update({
        'source_z': source_z,
        'source_x': source_x,
        'source_ids': source_ids,
        'emitter_index': 0,
    })
    time_reversal_config = simulation_config.copy()
    time_reversal_config['total_time'] = TIME_REVERSAL_TOTAL_TIME

    print(f'Running one simulation with {len(source_z)} independent sources.')
    acou_sim = SyntheticAcouSim(**simulation_config)
    acou_sim.run(generate_video=True, animation_step=15)
    tr_sim = SyntheticTimeReversal(**time_reversal_config)
    tr_sim.run(generate_video=True, animation_step=15)
    rtm_sim = SyntheticReverseTimeMigration(**time_reversal_config)
    rtm_sim.run(
        generate_video=True,
        animation_step=15,
        use_poynting_vectors=ENABLE_POYNTING_VECTORS,
    )


def main():
    c, source_z, source_x, receiver_z, receiver_x, source_ids = (
        convert_image_to_matrix(IMAGE_PATH, return_source_ids=True)
    )
    if FORCE_RECEIVERS_TO_SURFACE:
        receiver_z = np.full_like(receiver_z, SURFACE_RECEIVER_Z)

    if len(source_ids) == 0:
        raise ValueError('The model must contain a source marker to select a waveform.')

    print('Sources parsed from bitmap:')
    for source_number, (z, x, source_id) in enumerate(
        zip(source_z, source_x, source_ids)
    ):
        print(
            f'  Source {source_number}: z={int(z)}, x={int(x)}, '
            f'source_id={int(source_id)}'
        )

    grid_size_z, grid_size_x = np.int32(c.shape)
    base_config = {
        'dt': DT,
        'c': c,
        'dz': DZ,
        'dx': DX,
        'grid_size_z': grid_size_z,
        'grid_size_x': grid_size_x,
        'total_time': FORWARD_TOTAL_TIME,
        'medium_c': MEDIUM_C,
        'microphones_amount': np.int32(len(receiver_z)),
        'microphone_z': receiver_z,
        'microphone_x': receiver_x,
    }

    if ENABLE_SPARSE_FMC:
        _run_sparse_fmc(base_config, receiver_z, receiver_x, np.int32(source_ids[0]))
    else:
        _run_single_shot(base_config, source_z, source_x, source_ids)


if __name__ == '__main__':
    main()
