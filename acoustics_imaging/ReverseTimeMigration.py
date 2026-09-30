from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt

from .InputTest import InputTest
from .SimulationConfig import SimulationConfig
from .TimeReversal import TimeReversal
from .SyntheticReverseTimeMigration import _normalize_by_source_energy
from .functions import save_rtm_image, create_video
from .paths import REAL_SIMULATION_OUTPUT_DIR, SOURCES_DIR


def _poynting_mask(source_direction, receiver_direction):
    # Unlike SyntheticRTM's forward replay, this receiver solver evolves in
    # reverse time already. Its p*v direction must NOT be negated again.
    source = source_direction.astype(np.float32, copy=False)
    receiver = receiver_direction.astype(np.float32, copy=False)
    norm_product = np.hypot(source[0], source[1]) * np.hypot(receiver[0], receiver[1])
    dot = source[0] * receiver[0] + source[1] * receiver[1]
    return (norm_product > 0) & (dot >= -0.5 * norm_product)


class ReverseTimeMigration(SimulationConfig):
    """Correlate forward source checkpoints with backward-propagated recordings.

    Replaying only the final two TR frames loses waves absorbed by the PML.
    Retain source pressure at imaging times instead; propagate the measured
    receiver signals backward using the same acoustic solver and velocity model.
    """

    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)
        self.simulation_config = simulation_config.copy()
        self.input_test: InputTest = simulation_config['input_test']
        self.emitter_index = int(self.input_test.fmc_emitter)
        self.total_time = int(self.input_test.total_time)
        self.simulation_config['total_time'] = self.total_time
        output_dir = Path(simulation_config.get('output_dir', REAL_SIMULATION_OUTPUT_DIR))
        self.folder = output_dir / 'ReverseTimeMigration'
        self.data_folder = self.folder / 'data'
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.frames_folder = self.folder / 'frames'
        self.frames_folder.mkdir(parents=True, exist_ok=True)
        self.tr_folder = output_dir / 'TimeReversal'
        self.imaging_stride = int(simulation_config.get('imaging_stride', 1))
        if self.imaging_stride < 1:
            raise ValueError('imaging_stride must be positive.')

        source = simulation_config.get('source')
        if source is None:
            source = np.load(SOURCES_DIR / 'source.npy')
        self.source = np.asarray(source, dtype=np.float32)[:self.total_time]
        if len(self.source) < self.total_time:
            self.source = np.pad(self.source, (0, self.total_time - len(self.source)))

    def run(self, generate_video: bool, animation_step: int, use_poynting_vectors=False):
        if animation_step < 1:
            raise ValueError('animation_step must be positive.')
        if generate_video:
            for frame in self.frames_folder.glob('frame_*.png'):
                if frame.stem.removeprefix('frame_').isdigit():
                    frame.unlink()
        receiver = TimeReversal(**self.simulation_config)
        self.source_z = receiver.microphone_z[self.emitter_index]
        self.source_x = receiver.microphone_x[self.emitter_index]
        checkpoint_count = (self.total_time - 1) // self.imaging_stride + 1
        checkpoints = np.empty((checkpoint_count, *self.grid_size_shape), dtype=np.float32)
        # Unit directions tolerate float16 storage; pressure remains float32.
        directions = (np.empty((checkpoint_count, 2, *self.grid_size_shape), dtype=np.float16)
                      if use_poynting_vectors else None)
        memory = checkpoints.nbytes + (directions.nbytes if directions is not None else 0)
        print(f'Forward source checkpoints: {memory / 2**30:.2f} GiB RAM', flush=True)

        def save_source(step, pressure):
            if step % self.imaging_stride == 0:
                checkpoints[step // self.imaging_stride] = pressure

        def save_source_poynting(step, pressure_half, direction):
            save_source(step, pressure_half)
            directions[step // self.imaging_stride] = direction

        # TimeReversal reverses its input. Reverse the source here so that this
        # first pass propagates the source pulse forward, with just one emitter.
        source_input = InputTest()
        source_input.bscan = self.source[None, ::-1]
        source_input.microphones_amount = np.int32(1)
        source_input.microphones_distance = self.input_test.microphones_distance
        source_input.total_time = np.int32(self.total_time)
        source_input.fmc_emitter = np.int32(0)
        # Keep source-pass diagnostics separate from the measured receiver TR.
        with TemporaryDirectory(prefix='source_', dir=self.folder) as tmp:
            source_config = {
                **self.simulation_config, 'input_test': source_input, 'output_dir': tmp,
                'microphone_z': np.array([self.source_z], dtype=np.int32),
                'microphone_x': np.array([self.source_x], dtype=np.int32),
            }
            forward = TimeReversal(**source_config)
            if use_poynting_vectors:
                forward.run(False, animation_step, poynting_callback=save_source_poynting,
                            poynting_stride=self.imaging_stride)
            else:
                forward.run(False, animation_step, wavefield_callback=save_source)
            del forward

        accumulated_product = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.source_energy = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.poynting_image = (np.zeros(self.grid_size_shape, dtype=np.float32)
                               if use_poynting_vectors else None)
        frame_index = 0

        def correlate(step, pressure, receiver_direction=None):
            nonlocal frame_index
            source_step = self.total_time - 1 - step
            if source_step % self.imaging_stride:
                return
            source_pressure = checkpoints[source_step // self.imaging_stride]
            product = source_pressure * pressure
            accumulated_product[:] += product * self.imaging_stride
            self.source_energy += source_pressure * source_pressure * self.imaging_stride
            if receiver_direction is not None:
                accepted = _poynting_mask(directions[source_step // self.imaging_stride], receiver_direction)
                self.poynting_image += product * accepted * self.imaging_stride
            # At most one imaging interval away from each requested video time.
            if generate_video and step // animation_step >= frame_index:
                if use_poynting_vectors:
                    roi = (slice(self.absorption_layer_size, -self.absorption_layer_size),) * 2
                    standard = _normalize_by_source_energy(accumulated_product, self.source_energy)[roi]
                    poynting = _normalize_by_source_energy(self.poynting_image, self.source_energy)[roi]
                    limit = np.percentile(np.abs(np.stack((standard, poynting))), 99.5) or 1.0
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes[0, 0].imshow(pressure[roi], cmap='viridis')
                    axes[0, 0].set_title('Receiver wavefield (backward)')
                    axes[1, 0].imshow(source_pressure[roi], cmap='viridis')
                    axes[1, 0].set_title('Source wavefield (forward)')
                    for ax, data, title in ((axes[0, 1], standard, 'Normalized Standard RTM'),
                                            (axes[1, 1], poynting, 'Normalized Poynting RTM')):
                        ax.imshow(data, cmap='seismic', vmin=-limit, vmax=limit)
                        ax.set_title(title)
                    fig.savefig(self.frames_folder / f'frame_{frame_index}.png')
                    plt.close(fig)
                else:
                    save_rtm_image(
                        upper_left=pressure, upper_right=product,
                        bottom_left=source_pressure, bottom_right=accumulated_product,
                        path=self.frames_folder / f'frame_{frame_index}.png',
                    )
                frame_index += 1

        if use_poynting_vectors:
            receiver.run(generate_video, animation_step, poynting_callback=correlate,
                         poynting_stride=self.imaging_stride,
                         poynting_offset=(self.total_time - 1) % self.imaging_stride)
        else:
            receiver.run(generate_video, animation_step, wavefield_callback=correlate)
        del checkpoints
        del directions
        if not np.all(np.isfinite(accumulated_product)):
            raise RuntimeError('RTM produced a nonfinite image; check the grid/time step and data.')
        np.save(self.data_folder / f'accumulated_product_{self.emitter_index}.npy', accumulated_product)
        np.save(self.data_folder / f'accumulated_source_energy_{self.emitter_index}.npy', self.source_energy)
        np.save(self.data_folder / f'accumulated_product_normalized_{self.emitter_index}.npy',
                _normalize_by_source_energy(accumulated_product, self.source_energy))
        if use_poynting_vectors:
            if not np.all(np.isfinite(self.poynting_image)):
                raise RuntimeError('Poynting RTM produced a nonfinite image.')
            np.save(self.data_folder / f'accumulated_product_poynting_{self.emitter_index}.npy', self.poynting_image)
            np.save(self.data_folder / f'accumulated_product_poynting_normalized_{self.emitter_index}.npy',
                    _normalize_by_source_energy(self.poynting_image, self.source_energy))
        if generate_video:
            create_video(path=self.frames_folder, output_path=self.folder / 'rtm.mp4')
        print('Reverse Time Migration finished.', flush=True)
        return accumulated_product
