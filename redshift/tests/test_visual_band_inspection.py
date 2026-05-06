import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from argparse import Namespace

from visual_band_inspection import bandwise_visual_metrics, select_visual_indices, visual_acceptance_mask


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

    def test_visual_acceptance_rejects_large_core_band_flux_shift(self):
        row = {
            "median_relative_l1": 0.1,
            "negative_flux_fraction_aug": 0.0,
        }
        for band in ["u", "g", "r", "i", "z", "y"]:
            row[f"flux_ratio_{band}"] = 1.0
            row[f"relative_l1_{band}"] = 0.1
            row[f"corr_{band}"] = 0.99
        bad = dict(row)
        bad["flux_ratio_i"] = 0.7
        args = Namespace(
            core_bands="g,r,i,z",
            core_flux_ratio_min=0.9,
            core_flux_ratio_max=1.08,
            edge_flux_ratio_min=0.75,
            edge_flux_ratio_max=1.25,
            core_relative_l1_max=0.3,
            edge_relative_l1_max=0.5,
            core_corr_min=0.97,
            edge_corr_min=0.9,
            min_median_relative_l1=0.03,
            max_median_relative_l1=0.3,
            max_negative_fraction=0.45,
        )

        mask = visual_acceptance_mask([row, bad], args)

        np.testing.assert_array_equal(mask, np.array([True, False]))


if __name__ == "__main__":
    unittest.main()
