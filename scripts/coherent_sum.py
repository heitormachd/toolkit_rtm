from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from acoustics_imaging.functions import convert_image_to_matrix
from acoustics_imaging.paths import (
    COHERENT_SUM_OUTPUT_DIR,
    MODELS_DIR,
    SYNTHETIC_ACOU_SIM_OUTPUT_DIR,
)


RECORDING_PATH = SYNTHETIC_ACOU_SIM_OUTPUT_DIR / 'microphones_recording.npy'
IMAGE_PATH = MODELS_DIR / '1r150px-15s1px.png'
OUTPUT_DIR = COHERENT_SUM_OUTPUT_DIR
SOURCE_COUNT = 15
SLICE_START = 1400
SLICE_LENGTH = 800
ANGLE_DECIMALS = 6
DT = 3e-7
DZ = 1.5e-3
MEDIUM_C = 1500.0
SCAN_ANGLE_MIN = -80.0
SCAN_ANGLE_MAX = 80.0
SCAN_ANGLE_COUNT = 321


def slice_microphones_recording():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recording = np.load(RECORDING_PATH)
    if recording.ndim != 2:
        raise ValueError(f'Expected a 2D recording array, got shape {recording.shape}.')

    required_samples = SLICE_START + (SOURCE_COUNT * SLICE_LENGTH)
    if recording.shape[1] < required_samples:
        raise ValueError(
            f'Recording has {recording.shape[1]} samples, but {required_samples} are needed.'
        )

    for source_id in range(SOURCE_COUNT):
        start = SLICE_START + (source_id * SLICE_LENGTH)
        stop = start + SLICE_LENGTH
        recording_slice = recording[:, start:stop].T

        np.save(OUTPUT_DIR / f'source{source_id:02d}_slice.npy', recording_slice)

        vmax = np.percentile(np.abs(recording_slice), 99.5)
        if vmax == 0:
            vmax = 1

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, layout='tight')
        im = ax.imshow(
            recording_slice.T,
            aspect='auto',
            cmap='seismic',
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest',
            extent=(start, stop - 1, recording_slice.shape[1] - 1, 0),
        )
        ax.set_xlabel('Sample')
        ax.set_ylabel('Channel')
        ax.set_title(f'Source {source_id:02d} recording slice [{start}:{stop})')
        fig.colorbar(im, ax=ax, label='Amplitude')
        fig.savefig(OUTPUT_DIR / f'source{source_id:02d}_slice.png')
        plt.close(fig)


def calculate_source_receiver_angles():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, source_z, source_x, receptor_z, receptor_x, source_ids = convert_image_to_matrix(
        IMAGE_PATH,
        return_source_ids=True,
    )

    expected_ids = set(range(SOURCE_COUNT))
    found_ids = set(source_ids.astype(int).tolist())
    if found_ids != expected_ids:
        missing_ids = sorted(expected_ids - found_ids)
        extra_ids = sorted(found_ids - expected_ids)
        raise ValueError(
            f'Expected source IDs 0 through {SOURCE_COUNT - 1}; '
            f'missing={missing_ids}, extra={extra_ids}.'
        )

    if len(source_ids) != len(found_ids):
        duplicate_ids = sorted(
            source_id
            for source_id in found_ids
            if np.count_nonzero(source_ids == source_id) > 1
        )
        raise ValueError(f'Expected one pixel per source ID; duplicates={duplicate_ids}.')

    order = np.argsort(source_ids)
    source_z = source_z[order].astype(np.float64)
    source_x = source_x[order].astype(np.float64)
    receptor_z = receptor_z.astype(np.float64)
    receptor_x = receptor_x.astype(np.float64)

    dz = receptor_z[np.newaxis, :] - source_z[:, np.newaxis]
    distance_to_left = source_x[:, np.newaxis] - receptor_x[np.newaxis, :]
    angles_by_source = np.degrees(np.arctan2(dz, distance_to_left)).astype(np.float32)

    np.save(OUTPUT_DIR / 'angles_by_source.npy', angles_by_source)
    for source_id, angles in enumerate(angles_by_source):
        np.save(OUTPUT_DIR / f'source{source_id:02d}_angles.npy', angles)


def attach_angles_and_plot():
    angles_by_source = np.load(OUTPUT_DIR / 'angles_by_source.npy')
    if angles_by_source.shape[0] != SOURCE_COUNT:
        raise ValueError(
            f'Expected {SOURCE_COUNT} source angle arrays, got shape {angles_by_source.shape}.'
        )

    summed_by_angle = {}

    for source_id in range(SOURCE_COUNT):
        recording_slice = np.load(OUTPUT_DIR / f'source{source_id:02d}_slice.npy')
        angles = np.load(OUTPUT_DIR / f'source{source_id:02d}_angles.npy')

        if recording_slice.ndim != 2:
            raise ValueError(
                f'Expected source{source_id:02d}_slice.npy to be 2D, got {recording_slice.shape}.'
            )
        if recording_slice.shape[1] != angles.shape[0]:
            raise ValueError(
                f'Source {source_id:02d} has {recording_slice.shape[1]} channels, '
                f'but {angles.shape[0]} receptor angles.'
            )

        recording_slice = recording_slice.astype(np.float64)
        recording_slice -= np.mean(recording_slice, axis=0, keepdims=True)

        samples = np.arange(recording_slice.shape[0], dtype=np.float64)
        center_channel = recording_slice.shape[1] // 2
        channel_positions = (np.arange(recording_slice.shape[1]) - center_channel) * DZ
        source_center_angle = float(angles[center_channel])
        scan_angles = np.linspace(SCAN_ANGLE_MIN, SCAN_ANGLE_MAX, SCAN_ANGLE_COUNT)
        coherent_amplitudes = []

        for scan_angle in scan_angles:
            physical_angle = np.deg2rad(source_center_angle + scan_angle)
            channel_delays = channel_positions * np.sin(physical_angle) / (MEDIUM_C * DT)
            coherent_sum = np.zeros(recording_slice.shape[0], dtype=np.float64)

            for channel_index, channel_delay in enumerate(channel_delays):
                coherent_sum += np.interp(
                    samples + channel_delay,
                    samples,
                    recording_slice[:, channel_index],
                    left=0,
                    right=0,
                )

            coherent_sum /= recording_slice.shape[1]
            coherent_amplitudes.append(np.max(np.abs(coherent_sum)))

        coherent_amplitudes = np.asarray(coherent_amplitudes, dtype=np.float32)
        max_amplitude = np.max(coherent_amplitudes)
        if max_amplitude > 0:
            coherent_amplitudes /= max_amplitude

        angle_amplitude = np.column_stack(
            (scan_angles.astype(np.float32), coherent_amplitudes)
        ).astype(np.float32)
        np.save(OUTPUT_DIR / f'source{source_id:02d}_angle_amplitude.npy', angle_amplitude)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150, layout='tight')
        ax.plot(
            angle_amplitude[:, 0],
            angle_amplitude[:, 1],
            marker='o',
            markersize=2,
            linewidth=1,
        )
        ax.set_xlabel('Angle error from source direction (degrees)')
        ax.set_ylabel('Normalized coherent amplitude')
        ax.set_title(f'Source {source_id:02d}: coherent angular response')
        ax.grid(True, alpha=0.3)
        fig.savefig(OUTPUT_DIR / f'source{source_id:02d}_angle_amplitude.png')
        plt.close(fig)

        for angle, amplitude in angle_amplitude:
            angle_key = round(float(angle), ANGLE_DECIMALS)
            summed_by_angle[angle_key] = summed_by_angle.get(angle_key, 0.0) + float(amplitude)

    summed_angle_amplitude = np.array(
        [[angle, summed_by_angle[angle]] for angle in sorted(summed_by_angle)],
        dtype=np.float32,
    )
    max_summed_amplitude = np.max(summed_angle_amplitude[:, 1])
    if max_summed_amplitude > 0:
        summed_angle_amplitude[:, 1] /= max_summed_amplitude
    np.save(OUTPUT_DIR / 'summed_angle_amplitude.npy', summed_angle_amplitude)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, layout='tight')
    ax.plot(
        summed_angle_amplitude[:, 0],
        summed_angle_amplitude[:, 1],
        marker='o',
        markersize=2,
        linewidth=1,
    )
    ax.set_xlabel('Angle error from source direction (degrees)')
    ax.set_ylabel('Normalized coherent amplitude')
    ax.set_title('Summed coherent angular response')
    ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / 'summed_angle_amplitude.png')
    plt.close(fig)


if __name__ == '__main__':
    slice_microphones_recording()
    calculate_source_receiver_angles()
    attach_angles_and_plot()
