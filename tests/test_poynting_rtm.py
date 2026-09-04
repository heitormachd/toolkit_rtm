import math
import unittest
from pathlib import Path

import numpy as np

from acoustics_imaging.SyntheticReverseTimeMigration import _normalize_by_source_energy
from acoustics_imaging.SyntheticTimeReversal import _apply_direct_arrival_mute
from scripts.synthetic_workflow import (
    _direct_arrival_mute_samples,
    _select_sparse_fmc_transmitters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RTM_SHADER = PROJECT_ROOT / 'shaders' / 'reverse_time_migration.wgsl'
SYNTHETIC_RTM = PROJECT_ROOT / 'acoustics_imaging' / 'SyntheticReverseTimeMigration.py'
SYNTHETIC_TR = PROJECT_ROOT / 'acoustics_imaging' / 'SyntheticTimeReversal.py'
SYNTHETIC_WORKFLOW = PROJECT_ROOT / 'scripts' / 'synthetic_workflow.py'
SOURCE_WAVEFORM = PROJECT_ROOT / 'assets' / 'sources' / 'source.npy'


def passes_yoon_filter(source_vector, receiver_vector):
    source_scale = max(abs(component) for component in source_vector)
    receiver_scale = max(abs(component) for component in receiver_vector)
    if source_scale == 0.0 or receiver_scale == 0.0:
        return False

    source_direction = tuple(component / source_scale for component in source_vector)
    receiver_direction = tuple(component / receiver_scale for component in receiver_vector)
    dot_product = sum(
        source_component * receiver_component
        for source_component, receiver_component in zip(source_direction, receiver_direction)
    )
    denominator = math.hypot(*source_direction) * math.hypot(*receiver_direction)
    cos_theta = max(-1.0, min(1.0, dot_product / denominator))
    return cos_theta >= -0.5


class PoyntingRtmTest(unittest.TestCase):
    def test_yoon_120_degree_angle_filter(self):
        expected = {
            0: True,
            90: True,
            120: True,
            150: False,
            180: False,
        }

        for angle, should_be_kept in expected.items():
            with self.subTest(angle=angle):
                angle_radians = math.radians(angle)
                receiver_vector = (math.cos(angle_radians), math.sin(angle_radians))
                self.assertEqual(
                    should_be_kept,
                    passes_yoon_filter((1.0, 0.0), receiver_vector),
                )

    def test_low_amplitude_and_zero_vectors(self):
        self.assertTrue(passes_yoon_filter((1e-30, 0.0), (1e-30, 0.0)))
        self.assertFalse(passes_yoon_filter((0.0, 0.0), (1.0, 0.0)))

    def test_shader_uses_collocated_full_2d_vectors(self):
        shader = RTM_SHADER.read_text()

        self.assertIn('v_z_present[previous_z_idx]', shader)
        self.assertIn('v_x_present[previous_x_idx]', shader)
        self.assertIn('v_z_present_flipped_tr[previous_z_idx]', shader)
        self.assertIn('v_x_present_flipped_tr[previous_x_idx]', shader)
        self.assertIn('js_direction_x * jr_direction_x', shader)
        self.assertIn('js_direction_z * jr_direction_z', shader)
        self.assertIn('let jr_x = -pr * vr_x;', shader)
        self.assertIn('let jr_z = -pr * vr_z;', shader)
        self.assertIn('cos_theta >= -0.5', shader)
        self.assertNotIn('sdz > 0.0 && suz < 0.0', shader)

    def test_source_is_zero_mean_and_direct_arrival_mute_is_applied(self):
        source = np.load(SOURCE_WAVEFORM)

        self.assertLess(float(source.min()), 0.0)
        self.assertGreater(float(source.max()), 0.0)
        self.assertAlmostEqual(float(source.sum()), 0.0, places=5)

        bscan = np.ones((2, 1100), dtype=np.float32)
        _apply_direct_arrival_mute(bscan)
        self.assertTrue(np.all(bscan[:, :900] == 0.0))
        self.assertEqual(0.0, float(bscan[0, 900]))
        self.assertGreater(float(bscan[0, 950]), 0.0)
        self.assertLess(float(bscan[0, 950]), 1.0)
        self.assertAlmostEqual(1.0, float(bscan[0, 999]), places=6)
        self.assertTrue(np.all(bscan[:, 1000:] == 1.0))

    def test_direct_arrival_mute_accepts_one_cutoff_per_receiver(self):
        bscan = np.ones((3, 30), dtype=np.float32)

        _apply_direct_arrival_mute(
            bscan,
            mute_samples=np.array([5, 10, 20]),
            taper_samples=4,
        )

        self.assertTrue(np.all(bscan[0, :1] == 0.0))
        self.assertTrue(np.all(bscan[1, :6] == 0.0))
        self.assertTrue(np.all(bscan[2, :16] == 0.0))
        self.assertAlmostEqual(1.0, float(bscan[0, 4]), places=6)
        self.assertAlmostEqual(1.0, float(bscan[1, 9]), places=6)
        self.assertAlmostEqual(1.0, float(bscan[2, 19]), places=6)

    def test_sparse_fmc_geometry_and_offset_dependent_mute(self):
        transmitter_indices = _select_sparse_fmc_transmitters(551, 16)

        self.assertEqual(36, len(transmitter_indices))
        self.assertEqual(0, int(transmitter_indices[0]))
        self.assertEqual(550, int(transmitter_indices[-1]))
        self.assertTrue(np.all(np.diff(transmitter_indices) <= 16))

        receiver_z = np.array([60, 60], dtype=np.int32)
        receiver_x = np.array([75, 625], dtype=np.int32)
        mute_samples = _direct_arrival_mute_samples(
            60,
            75,
            receiver_z,
            receiver_x,
            source_duration_samples=103,
        )
        self.assertEqual(203, int(mute_samples[0]))
        self.assertEqual(2037, int(mute_samples[1]))

    def test_normalization_floors_low_illumination(self):
        image = np.ones((1, 4), dtype=np.float32)
        source_energy = np.array([[1.0, 1e-2, 1e-3, 1e-4]], dtype=np.float32)

        normalized = _normalize_by_source_energy(image, source_energy)

        self.assertGreater(float(normalized[0, 2]), 0.0)
        self.assertAlmostEqual(float(normalized[0, 2]), float(normalized[0, 3]), places=3)

    def test_workflow_exposes_poynting_toggle(self):
        workflow = SYNTHETIC_WORKFLOW.read_text()
        rtm = SYNTHETIC_RTM.read_text()

        self.assertIn('ENABLE_POYNTING_VECTORS = True', workflow)
        self.assertIn('ENABLE_SPARSE_FMC = True', workflow)
        self.assertIn('FMC_TRANSMITTER_STRIDE = 16', workflow)
        self.assertIn('TIME_REVERSAL_TOTAL_TIME = FORWARD_TOTAL_TIME', workflow)
        self.assertIn('use_poynting_vectors=ENABLE_POYNTING_VECTORS', workflow)
        self.assertIn("standard_sum += shot_result['standard_raw']", workflow)
        self.assertIn("source_energy_sum += shot_result['source_energy']", workflow)
        self.assertIn('if use_poynting_vectors:', rtm)
        self.assertIn('roi_slice = (slice(L, -L), slice(L, -L))', rtm)
        self.assertIn('accumulated_source_energy_', rtm)
        self.assertNotIn('scatter(roi_reflector', rtm)


if __name__ == '__main__':
    unittest.main()
