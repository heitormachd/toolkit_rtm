"""Check the four physical boundary choices without allocating a GPU device."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from acoustics_imaging.SimulationConfig import SimulationConfig
from scripts import compare_td_rtm_boundaries as comparison


class BoundaryComparisonTest(unittest.TestCase):
    def test_shader_and_absorption_choices_are_independent(self):
        original_shader_dir = comparison.base.TR_MODULE.SHADERS_DIR
        original_propagation = comparison.base.RTM_MODULE.TimeReversal
        original_flags = (comparison.BoundaryPropagation.reflect_sides,
                          comparison.BoundaryPropagation.reflect_bottom)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                original = (comparison.base.SHADERS_DIR / 'time_reversal.wgsl').read_text()
                for case, (sides, bottom) in comparison.CASES.items():
                    with self.subTest(case=case):
                        comparison.install_boundaries(root, case)
                        shader = (root / 'shaders' / case / 'time_reversal.wgsl').read_text()
                        self.assertEqual('x == 0 || x == infoI32.grid_size_x - 1' in shader, sides)
                        self.assertEqual('z == infoI32.grid_size_z - 1' in shader, bottom)
                        self.assertEqual('x + 4 < infoI32.grid_size_x - 1' in shader, sides)
                        self.assertEqual('z + 4 < infoI32.grid_size_z - 1' in shader, bottom)
                        if not sides and not bottom:
                            self.assertEqual(shader, original)
                        if sides or bottom:
                            self.assertLess(shader.index('p_future[grid_index] = 0.0;'),
                                            shader.index('p_past[grid_index] = p_present[grid_index];'))
                        sim = comparison.BoundaryPropagation.__new__(comparison.BoundaryPropagation)
                        SimulationConfig.__init__(sim, c=np.full((140,130),1496,np.float32),
                            dx=5e-5, dz=5e-5, dt=4e-9, grid_size_z=140, grid_size_x=130, total_time=1)
                        with patch.object(comparison.base.TimeReversal, 'setup_gpu'):
                            sim.setup_gpu()
                        # Top absorber is present in every mode. The center is never damped.
                        self.assertTrue(np.all(sim.is_z_absorption_int[:45]))
                        self.assertFalse(np.any(sim.is_z_absorption_int[45:-45]))
                        self.assertEqual(bool(np.any(sim.is_z_absorption_int[-45:])), not bottom)
                        self.assertEqual(bool(np.any(sim.is_x_absorption_int[:, :45])), not sides)
                        self.assertEqual(bool(np.any(sim.is_x_absorption_int[:, -45:])), not sides)
                        self.assertFalse(np.any(sim.is_x_absorption_int[:,45:-45]))
        finally:
            comparison.base.TR_MODULE.SHADERS_DIR = original_shader_dir
            comparison.base.RTM_MODULE.TimeReversal = original_propagation
            comparison.BoundaryPropagation.reflect_sides, comparison.BoundaryPropagation.reflect_bottom = original_flags


if __name__ == '__main__':
    unittest.main()
