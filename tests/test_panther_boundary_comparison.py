"""Timing and checkpoint integrity for the recorded-data boundary experiment."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import compare_panther_boundaries as comparison


class PantherComparisonTest(unittest.TestCase):
    def test_timing_padding_preserves_complete_trace_and_source(self):
        recording = np.arange(64 * 15000, dtype=np.float32).reshape(64, 15000)
        source = np.arange(15000, dtype=np.float32)
        inp = SimpleNamespace()
        cfg = {'dt': np.float32(4e-9)}

        def prepare(inp, cfg, metadata, tx):
            inp.bscan = recording.copy()
            cfg['source'] = source.copy()

        with patch.object(comparison.workflow, 'prepare_emitter', side_effect=prepare):
            for _ in range(2):
                comparison.prepare_shot(inp, cfg, {'amplitude_scale': 2.}, 32)
                self.assertEqual(int(cfg['total_time']), 15225)
                np.testing.assert_array_equal(inp.bscan[:, :225], 0)
                np.testing.assert_array_equal(inp.bscan[:, 225:], recording / 2)
                np.testing.assert_array_equal(cfg['source'], source)

    def test_resume_uses_only_committed_checkpoints_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'shots').mkdir()
            arrays = {key: np.ones((3, 4), np.float32) for key in comparison.FIELDS}
            np.savez(folder / 'shots/tx_32.npz', **arrays, transmitter=32)
            (folder / 'shots/tx_33.tmp').write_bytes(b'interrupted write')
            sums, txs = comparison.load_stack(folder, (3, 4))
            self.assertEqual(txs, [32])
            for key in comparison.FIELDS:
                np.testing.assert_array_equal(sums[key], arrays[key])
            arrays['standard'][0,0] = np.nan
            np.savez(folder / 'shots/tx_33.npz', **arrays, transmitter=33)
            with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                comparison.load_stack(folder, (3, 4))


if __name__ == '__main__':
    unittest.main()
