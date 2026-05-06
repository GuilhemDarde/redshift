import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from visual_band_inspection import bandwise_visual_metrics, select_visual_indices


class VisualBandInspectionTests(unittest.TestCase):
    def test_bandwise_metrics_are_finite_for_scaled_clone(self):
        source = np.zeros((2, 6, 4, 4), dtype=np.float32)
        source[:, :, 1:3, 1:3] = 1.0
        augmented = source * 1.1

        metrics = bandwise_visual_metrics(source, augmented)

        self.assertEqual(metrics["flux_ratio"].shape, (2, 6))
        self.assertEqual(metrics["relative_l1"].shape, (2, 6))
        self.assertTrue(np.isfinite(metrics["flux_ratio"]).all())
        self.assertTrue(np.isfinite(metrics["relative_l1"]).all())
        self.assertTrue(np.nanmedian(metrics["flux_ratio"]) > 1.0)

    def test_select_visual_indices_mixed_returns_requested_count(self):
        candidate_indices = np.arange(20)
        median_l1 = np.linspace(0.0, 1.0, 20)

        selected = select_visual_indices(candidate_indices, median_l1, max_examples=6, seed=42, strategy="mixed")

        self.assertEqual(len(selected), 6)
        self.assertTrue(set(selected).issubset(set(candidate_indices)))


if __name__ == "__main__":
    unittest.main()
