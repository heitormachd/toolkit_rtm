import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acoustics_imaging.InputTest import InputTest
from acoustics_imaging.ReverseTimeMigration import ReverseTimeMigration, _poynting_mask
from acoustics_imaging.TimeReversal import TimeReversal, _poynting_direction
from acoustics_imaging.SyntheticTimeReversal import SyntheticTimeReversal
from acoustics_imaging.WebGpuHandler import read_shader_bindings
from acoustics_imaging.paths import SHADERS_DIR
from scripts.real_workflow import prepare_emitter


class RealWorkflowTest(unittest.TestCase):
    def test_spatial_order_stability_limit(self):
        data = self.make_input()
        data.select_fmc_emitter(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(dt=np.float32(40e-9), c=np.full((110, 110), 1500, dtype=np.float32),
                          dz=np.float32(1e-4), dx=np.float32(1e-4), grid_size_z=110,
                          grid_size_x=110, total_time=6, input_test=data, output_dir=tmp)
            with patch.object(TimeReversal, 'setup_gpu'):
                self.assertEqual(TimeReversal(**config).info_i32[-1], 2)
                with self.assertRaisesRegex(ValueError, 'Unstable time step'):
                    TimeReversal(**config, spatial_order=8)
                config['dt'] = np.float32(20e-9)
                self.assertEqual(TimeReversal(**config, spatial_order=8).info_i32[-1], 8)
                with self.assertRaisesRegex(ValueError, 'spatial_order must'):
                    TimeReversal(**config, spatial_order=4)

    def test_poynting_direction_is_collocated_and_amplitude_independent(self):
        pressure = np.ones((4, 5), dtype=np.float32)
        pressure[2, 2] = -1
        pressure[2, 3] = 0
        velocity = np.zeros((2, 4, 5), dtype=np.float32)
        velocity[0] = 1e-30
        direction = _poynting_direction(pressure, velocity)
        np.testing.assert_array_equal(direction[:, 1, 1], [1, 0])
        np.testing.assert_array_equal(direction[:, 2, 2], [-1, 0])
        np.testing.assert_array_equal(direction[:, 2, 3], [0, 0])
        np.testing.assert_array_equal(direction[:, 0, :], 0)
        np.testing.assert_array_equal(direction[:, :, 0], 0)

    def test_poynting_angle_and_backward_receiver_sign(self):
        angles = np.deg2rad([0, 90, 119.9, 120.1, 180])
        source = np.tile(np.array([[1], [0]], dtype=np.float32), (1, len(angles)))
        receiver = np.array([np.cos(angles), np.sin(angles)], dtype=np.float32)
        # Reflected waves converge in reverse time along the incident direction.
        np.testing.assert_array_equal(_poynting_mask(source.astype(np.float16), receiver),
                                      [True, True, True, False, False])
        np.testing.assert_array_equal(_poynting_mask(source * 0, receiver), False)

    def make_input(self):
        data = InputTest()
        data.dt = np.float32(8e-9)
        data.gate_start = np.float32(0.024)  # Three samples, in microseconds.
        data.microphones_amount = np.int32(2)
        data.microphones_distance = np.float32(0.5e-3)
        t = np.arange(1000) * float(data.dt)
        trace = np.sin(2 * np.pi * 5e6 * t).astype(np.float32)
        data.bscan_fmc = np.tile(trace[:, None, None], (1, 2, 2))
        return data

    def test_shared_tr_shader_bindings_for_real_and_synthetic(self):
        bindings = read_shader_bindings((SHADERS_DIR / 'time_reversal.wgsl').read_text().splitlines())
        names = {value[0] for group in bindings.values() for value in group.values()}
        data = self.make_input()
        data.select_fmc_emitter(0)
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(dt=data.dt, c=np.full((110, 110), 1500, dtype=np.float32),
                          dz=np.float32(1e-4), dx=np.float32(1e-4), grid_size_z=110,
                          grid_size_x=110, total_time=6, input_test=data, output_dir=tmp,
                          microphone_z=np.array([50, 50]), microphone_x=np.array([50, 55]),
                          microphones_amount=2, source_z=np.array([50]), source_x=np.array([50]),
                          medium_c=np.float32(1500))
            folder = Path(tmp) / 'SyntheticAcouSim'
            folder.mkdir()
            np.save(folder / 'microphones_recording.npy', np.zeros((2, 6), dtype=np.float32))
            for cls in (TimeReversal, SyntheticTimeReversal):
                with patch(f'{cls.__module__}.WebGpuHandler') as gpu, \
                        patch('acoustics_imaging.SyntheticTimeReversal.load_sources',
                              return_value=np.zeros((1, 6), dtype=np.float32)):
                    cls(**config)
                buffers = gpu.return_value.create_buffers.call_args.args[0]
                self.assertEqual(set(buffers), names)

    def test_gate_padding_and_repeatable_emitter_preprocessing(self):
        data = self.make_input()
        raw = data.bscan_fmc.copy()
        data.select_fmc_emitter(0)
        np.testing.assert_array_equal(data.bscan[:, :3], 0)
        self.assertEqual(data.bscan.shape, (2, 1003))
        config = {
            'dt': np.float32(4e-9), 'c': np.full((110, 110), 1483, dtype=np.float32),
            'microphone_z': np.array([50, 50]), 'microphone_x': np.array([50, 60]),
        }
        settings = {'end_time_us': 6.0, 'time_upsample': 2, 'source_delay_us': 0.9}
        prepare_emitter(data, config, settings, 0)
        first = data.bscan.copy()
        self.assertEqual(first.shape, (2, 1500))
        self.assertEqual(config['total_time'], 1500)
        self.assertEqual(config['source'].shape, (1500,))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertGreater(np.max(abs(first)), 0)
        np.testing.assert_array_equal(first[:, :500], 0)  # At least 2 us muted.
        prepare_emitter(data, config, settings, 1)
        prepare_emitter(data, config, settings, 0)
        np.testing.assert_array_equal(data.bscan, first)
        np.testing.assert_array_equal(data.bscan_fmc, raw)
        self.assertEqual(data.dt, np.float32(8e-9))  # Original gate's sampling stays intact.
        settings['mute_before_us'] = 4.0
        prepare_emitter(data, config, settings, 0)
        np.testing.assert_array_equal(data.bscan[:, :1000], 0)
        self.assertGreater(np.max(abs(data.bscan[:, 1100:])), 0)

    def test_real_rtm_correlates_matching_times_and_custom_source(self):
        data = self.make_input()
        data.select_fmc_emitter(1)
        data.total_time = np.int32(5)
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                'dt': data.dt, 'c': np.full((110, 110), 1500, dtype=np.float32),
                'dz': np.float32(0.1e-3), 'dx': np.float32(0.1e-3),
                'grid_size_z': np.int32(110), 'grid_size_x': np.int32(110),
                'total_time': data.total_time, 'input_test': data, 'output_dir': tmp,
                'microphone_z': np.array([50, 50]), 'microphone_x': np.array([50, 55]),
                'source': np.array([1, -1], dtype=np.float32), 'imaging_stride': 2,
            }
            with patch.object(TimeReversal, 'setup_gpu'):
                tr = TimeReversal(**config)
            self.assertEqual(int(np.load(tr.folder / 'emitter_x.npy')), 55)
            self.assertEqual(int(np.load(tr.folder / 'emitter_z.npy')), 50)
            calls = []

            def propagate(sim, generate_video, animation_step, wavefield_callback=None):
                calls.append(sim)
                for step in range(5):
                    value = step if len(calls) == 1 else step + 1
                    wavefield_callback(step, np.full((110, 110), value, dtype=np.float32))

            # No saved terminal TR frames: imaging must use the recordings.
            with patch.object(TimeReversal, 'setup_gpu'), patch.object(TimeReversal, 'run', propagate):
                rtm = ReverseTimeMigration(**config)
                result = rtm.run(False, 15)
            np.testing.assert_array_equal(rtm.source, [1, -1, 0, 0, 0])
            np.testing.assert_array_equal(calls[0].flipped_bscan[0], rtm.source)
            np.testing.assert_array_equal(calls[0].microphone_x, [55])
            np.testing.assert_array_equal(calls[0].microphone_z, [50])
            np.testing.assert_array_equal(calls[1].flipped_bscan, data.bscan[:, ::-1])
            # Forward times 0, 2, 4 pair with reverse times 4, 2, 0.
            np.testing.assert_array_equal(result, 2 * (0 * 5 + 2 * 3 + 4 * 1))
            np.testing.assert_array_equal(np.load(rtm.folder / 'data' / 'accumulated_product_1.npy'), result)

            self.assertEqual(list(rtm.folder.glob('*.npy')), [])

            data.total_time = np.int32(6)  # Reverse samples must have offset 1 for stride 2.
            calls.clear()

            def propagate_poynting(sim, generate_video, animation_step, poynting_callback=None,
                                   poynting_stride=1, poynting_offset=0):
                calls.append((generate_video, poynting_offset))
                direction = np.zeros((2, 110, 110), dtype=np.float32)
                direction[0] = 1
                if len(calls) == 2:
                    direction[0, :, 55:] = -1
                for step in range(6):
                    if step % poynting_stride == poynting_offset:
                        poynting_callback(step, np.full((110, 110), step + 1, dtype=np.float32), direction)

            with patch.object(TimeReversal, 'setup_gpu'), patch.object(TimeReversal, 'run', propagate_poynting), \
                    patch('acoustics_imaging.ReverseTimeMigration.create_video'), \
                    patch('acoustics_imaging.ReverseTimeMigration.plt.Figure.savefig'):
                rtm = ReverseTimeMigration(**config)
                result = rtm.run(True, 15, use_poynting_vectors=True)
            np.testing.assert_array_equal(result, 2 * (1 * 6 + 3 * 4 + 5 * 2))
            np.testing.assert_array_equal(rtm.source_energy, 2 * (1 + 9 + 25))
            np.testing.assert_array_equal(rtm.poynting_image[:, :55], result[:, :55])
            np.testing.assert_array_equal(rtm.poynting_image[:, 55:], 0)
            self.assertEqual(calls, [(False, 0), (True, 1)])
            self.assertTrue((rtm.folder / 'data' / 'accumulated_product_poynting_normalized_1.npy').exists())



if __name__ == '__main__':
    unittest.main()
