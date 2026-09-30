import os
import tempfile
import unittest
from pathlib import Path

_MPLCONFIGDIR = tempfile.TemporaryDirectory()
os.environ.setdefault('MPLCONFIGDIR', _MPLCONFIGDIR.name)

import numpy as np
from matplotlib.image import imsave

from acoustics_imaging.functions import convert_image_to_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLUE = np.array([0, 0, 255], dtype=np.uint8)
CYAN = np.array([0, 255, 255], dtype=np.uint8)
YELLOW = np.array([255, 255, 0], dtype=np.uint8)


class ConvertImageToMatrixDASTest(unittest.TestCase):
    def write_bitmap(self, path, shape, receptor_points, source_point=(0, 0)):
        image = np.zeros((*shape, 3), dtype=np.uint8)
        image[:, :] = BLUE
        image[source_point] = YELLOW
        for receptor_point in receptor_points:
            image[receptor_point] = CYAN
        imsave(path, image)

    def parse_receptors(self, path):
        _, _, _, receptor_z, receptor_x = convert_image_to_matrix(str(path))
        return list(zip(receptor_z.astype(int).tolist(), receptor_x.astype(int).tolist()))

    def assert_contiguous_path(self, receptor_points):
        for current_point, next_point in zip(receptor_points, receptor_points[1:]):
            dz = abs(current_point[0] - next_point[0])
            dx = abs(current_point[1] - next_point[1])
            self.assertLessEqual(max(dz, dx), 1)

    def test_curved_receptor_line_is_ordered_by_path(self):
        receptor_points = [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 2),
            (7, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (7, 7),
            (6, 8),
            (5, 9),
            (4, 10),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'curved.png'
            self.write_bitmap(image_path, (12, 12), receptor_points)

            parsed_receptors = self.parse_receptors(image_path)

        self.assertEqual(receptor_points, parsed_receptors)
        self.assert_contiguous_path(parsed_receptors)

    def test_map_fixture_has_expected_source_and_receiver_array(self):
        _, source_z, source_x, receptor_z, receptor_x = convert_image_to_matrix(
            str(PROJECT_ROOT / 'assets' / 'models' / 'map.png')
        )

        self.assertEqual([(50, 350)], list(zip(source_z.astype(int), source_x.astype(int))))
        self.assertEqual((50, 297), (int(receptor_z[0]), int(receptor_x[0])))
        self.assertEqual((50, 402), (int(receptor_z[-1]), int(receptor_x[-1])))
        self.assertEqual(106, len(receptor_z))

        receptor_points = list(zip(receptor_z.astype(int), receptor_x.astype(int)))
        self.assert_contiguous_path(receptor_points)

    def test_poynting_benchmark_has_one_source_and_wide_receiver_array(self):
        c, source_z, source_x, receptor_z, receptor_x = convert_image_to_matrix(
            str(PROJECT_ROOT / 'assets' / 'models' / 'poynting_benchmark.png')
        )

        self.assertEqual([(60, 350)], list(zip(source_z.astype(int), source_x.astype(int))))
        self.assertEqual((60, 75), (int(receptor_z[0]), int(receptor_x[0])))
        self.assertEqual((60, 625), (int(receptor_z[-1]), int(receptor_x[-1])))
        self.assertEqual(551, len(receptor_z))
        self.assertTrue(np.any(c == np.float32(3200)))
        self.assertTrue(np.any(c == np.float32(0)))

        receptor_points = list(zip(receptor_z.astype(int), receptor_x.astype(int)))
        self.assert_contiguous_path(receptor_points)

    def test_horizontal_receptor_line_keeps_left_to_right_order(self):
        receptor_points = [(4, x) for x in range(2, 8)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'horizontal.png'
            self.write_bitmap(image_path, (8, 10), receptor_points)

            parsed_receptors = self.parse_receptors(image_path)

        self.assertEqual(receptor_points, parsed_receptors)

    def test_disconnected_receptor_components_raise_value_error(self):
        receptor_points = [(2, 2), (2, 3), (6, 6), (6, 7)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'disconnected.png'
            self.write_bitmap(image_path, (10, 10), receptor_points)

            with self.assertRaisesRegex(ValueError, 'connected DAS line'):
                convert_image_to_matrix(str(image_path))

    def test_td_rtm_maps_have_two_velocities_and_64_colocated_elements(self):
        expected_x = np.arange(85, 716, 10, dtype=np.int32)
        for name in ('t_b1', 'y_b1', 'y_b2', 'y_b3', 'y_b4'):
            with self.subTest(model=name):
                path = PROJECT_ROOT / 'assets' / 'models' / f'td_rtm_{name}.png'
                c, source_z, source_x, receptor_z, receptor_x, source_ids = (
                    convert_image_to_matrix(str(path), return_source_ids=True)
                )
                self.assertEqual(c.shape, (1041, 801))
                np.testing.assert_array_equal(np.unique(c), [1500, 6400])
                np.testing.assert_array_equal(source_x, expected_x)
                np.testing.assert_array_equal(receptor_x, expected_x)
                np.testing.assert_array_equal(source_z, np.ones(64, dtype=np.int32))
                np.testing.assert_array_equal(receptor_z, source_z)
                np.testing.assert_array_equal(source_ids, np.zeros(64, dtype=np.int32))

    def test_disconnected_colocated_markers_on_different_rows_still_raise(self):
        pixels = np.zeros((8, 10, 3), dtype=np.uint8)
        pixels[:] = BLUE
        pixels[1, 1] = 255
        pixels[5, 7] = 255
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'disconnected_white.png'
            imsave(path, pixels)
            with self.assertRaisesRegex(ValueError, 'connected DAS line'):
                convert_image_to_matrix(str(path))

    def test_branched_receptor_line_raises_value_error(self):
        receptor_points = [(2, 1), (2, 2), (2, 3), (1, 2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / 'branched.png'
            self.write_bitmap(image_path, (6, 6), receptor_points)

            with self.assertRaisesRegex(ValueError, 'branch|ambiguous'):
                convert_image_to_matrix(str(image_path))


if __name__ == '__main__':
    unittest.main()
