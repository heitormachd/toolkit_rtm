"""Check thesis metric definitions on analytically known image samples."""
import unittest

import numpy as np

from scripts.image_quality_metrics import THRESHOLD, api_measure, cnr_masks, measure


class ImageQualityMetricsTest(unittest.TestCase):
    def test_power_contrast_and_background_power_cnr(self):
        amplitude = np.array([[4., 4.], [1., 3.]])
        signal = np.array([[True, True], [False, False]])
        noise = ~signal
        result, _ = measure(amplitude, signal, noise, .1, .2, 1.)
        self.assertEqual(result['signal_mean_power'], 16.)
        self.assertEqual(result['noise_mean_power'], 5.)
        self.assertEqual(result['noise_std_power'], 4.)
        self.assertAlmostEqual(result['contrast'], 3.2)
        self.assertAlmostEqual(result['cnr'], 2.75)
        self.assertAlmostEqual(result['contrast_db'], 5.05149978319906)
        self.assertAlmostEqual(result['cnr_db'], 8.786653876605253)

    def test_db_values_preserve_ratios_below_one(self):
        amplitude = np.sqrt([[2., 2.], [1., 4.]])
        signal = np.array([[True, True], [False, False]])
        result, _ = measure(amplitude, signal, ~signal, 1., 1., 1.)
        self.assertAlmostEqual(result['contrast_db'], -.969100130080564)
        self.assertAlmostEqual(result['cnr_db'], -9.542425094393248)

    def test_zero_ratios_have_no_finite_db_value(self):
        signal = np.array([[True, True], [False, False]])
        with np.errstate(divide='raise', invalid='raise'):
            equal, _ = measure(np.array([[1., 3.], [1., 3.]]), signal, ~signal, 1., 1., 1.)
            zero, _ = measure(np.array([[0., 0.], [1., 3.]]), signal, ~signal, 1., 1., 1.)
        self.assertEqual(equal['contrast_db'], 0.)
        self.assertEqual(equal['cnr'], 0.)
        self.assertIsNone(equal['cnr_db'])
        self.assertEqual(zero['contrast'], 0.)
        self.assertIsNone(zero['contrast_db'])
        self.assertTrue(equal['metric_notes'])
        self.assertTrue(zero['metric_notes'])

    def test_global_minus_6db_includes_disconnected_regions_and_threshold(self):
        amplitude = np.array([[1., THRESHOLD, 0.], [0., THRESHOLD * .999, .8]])
        signal = np.zeros_like(amplitude, bool)
        signal[0, 0] = True
        result, mask = measure(amplitude, signal, ~signal, .2, .3, 2.)
        np.testing.assert_array_equal(mask, [[True, True, False], [False, False, True]])
        self.assertEqual(result['a_minus_6db_pixels'], 3)
        self.assertAlmostEqual(result['a_minus_6db_mm2'], .18)
        self.assertAlmostEqual(result['api'], .045)

    def test_cnr_and_api_do_not_depend_on_amplitude_scale(self):
        amplitude = np.array([[4., 4.], [1., 3.]])
        signal = np.array([[True, True], [False, False]])
        first, first_mask = measure(amplitude, signal, ~signal, 1., 1., 2.)
        second, second_mask = measure(amplitude * 100, signal, ~signal, 1., 1., 2.)
        for key in ('contrast', 'contrast_db', 'cnr', 'cnr_db', 'api'):
            self.assertAlmostEqual(first[key], second[key])
        np.testing.assert_array_equal(first_mask, second_mask)

    def test_reflector_window_excludes_stronger_external_artifact(self):
        amplitude = np.array([[100., 0.], [1., .8]])
        domain = np.array([[False, False], [True, True]])
        full, full_mask = api_measure(amplitude, 1., 1., 1.)
        roi, roi_mask = api_measure(amplitude, 1., 1., 1., domain)
        self.assertEqual(full['image_peak_amplitude'], 100.)
        self.assertEqual(roi['image_peak_amplitude'], 1.)
        self.assertEqual(roi['api'], 2.)
        np.testing.assert_array_equal(full_mask, [[True, False], [False, False]])
        np.testing.assert_array_equal(roi_mask, [[False, False], [True, True]])

    def test_zero_background_has_explicit_undefined_metrics(self):
        amplitude = np.array([[1., 0.], [0., 0.]])
        signal = amplitude > 0
        result, _ = measure(amplitude, signal, ~signal, 1., 1., 1.)
        self.assertIsNone(result['contrast'])
        self.assertIsNone(result['contrast_db'])
        self.assertIsNone(result['cnr'])
        self.assertIsNone(result['cnr_db'])
        self.assertEqual(len(result['metric_notes']), 2)

    def test_zero_image_and_overlapping_regions_are_rejected(self):
        signal = np.array([[True, False], [False, False]])
        with self.assertRaises(ValueError):
            measure(np.zeros((2, 2)), signal, ~signal, 1., 1., 1.)
        with self.assertRaises(ValueError):
            measure(np.ones((2, 2)), signal, np.ones((2, 2), bool), 1., 1., 1.)

    def test_circle_regions_preserve_guard_band_and_roi(self):
        coordinates = np.arange(-4., 5.)
        region = {'cnr_roi_mm': [-3, 3, -3, 3],
                  'signal': {'kind': 'circles', 'centers_mm': [[0, 0]], 'radius_mm': 1.},
                  'noise': {'kind': 'outside_circles', 'centers_mm': [[0, 0]], 'radius_mm': 2.}}
        signal, noise = cnr_masks(region, coordinates, coordinates)
        self.assertTrue(signal[4, 4])
        self.assertFalse(noise[4, 4])
        self.assertFalse(signal[4, 6])
        self.assertFalse(noise[4, 6])
        self.assertTrue(noise[4, 7])
        self.assertFalse(noise[4, 8])
        self.assertFalse(np.any(signal & noise))

    def test_local_noise_patches_pool_power_samples_and_exclude_other_pixels(self):
        coordinates = np.arange(9.)
        region = {'cnr_roi_mm': [0, 8, 0, 8],
                  'signal': {'kind': 'rectangles', 'bounds_mm': [[3, 4, 3, 4]]},
                  'noise': {'kind': 'rectangles', 'bounds_mm': [[0, 1, 0, 1], [7, 8, 7, 8]]}}
        signal, noise = cnr_masks(region, coordinates, coordinates)
        amplitude = np.full((9, 9), 1000.)
        amplitude[signal] = 4.
        amplitude[:2, :2] = 1.
        amplitude[7:, 7:] = 3.
        result, _ = measure(amplitude, signal, noise, 1., 1., 1.)
        self.assertEqual(result['noise_pixels'], 8)
        self.assertEqual(result['noise_mean_power'], 5.)
        self.assertEqual(result['noise_std_power'], 4.)
        self.assertAlmostEqual(result['contrast'], 3.2)
        self.assertAlmostEqual(result['cnr'], 2.75)


if __name__ == '__main__':
    unittest.main()
