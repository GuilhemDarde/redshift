import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from analysis_utils import (
    aggregate_by_bins,
    compute_regression_metrics,
    load_metadata,
    save_metadata_npz,
    stripe82_mask,
    z_to_bin_indices,
)


class AnalysisUtilsTests(unittest.TestCase):
    def test_regression_metrics(self):
        z_true = np.array([0.0, 1.0, 2.0])
        z_pred = np.array([0.0, 1.2, 1.7])
        metrics = compute_regression_metrics(z_true, z_pred)
        self.assertEqual(metrics["n"], 3)
        self.assertIn("sigma_nmad", metrics)
        self.assertIn("outlier_rate", metrics)

    def test_z_bins(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        labels = z_to_bin_indices(np.array([-1.0, 0.2, 1.5, 4.0]), edges)
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 2]))

    def test_i_band_aggregation_20_bins(self):
        mag_i = np.linspace(18.0, 20.0, 21)
        z_true = np.linspace(0.1, 1.0, 21)
        z_pred = z_true + 0.01
        rows = aggregate_by_bins(mag_i, np.linspace(18.0, 20.0, 21), z_true, z_pred)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all("sigma_nmad" in row for row in rows))

    def test_stripe82_wraparound(self):
        ra = np.array([350.0, 10.0, 120.0])
        dec = np.array([0.0, 0.5, 0.0])
        mask = stripe82_mask(ra, dec, ra_min=300.0, ra_max=60.0, dec_min=-1.25, dec_max=1.25)
        np.testing.assert_array_equal(mask, np.array([True, True, False]))

    def test_mini_npz_metadata_load(self):
        metadata = {
            "ra": np.array([1.0, 2.0]),
            "dec": np.array([0.0, 0.1]),
            "z_true": np.array([0.2, 0.3]),
            "mag_i": np.array([19.0, 20.0]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mini_metadata.npz")
            save_metadata_npz(path, metadata)
            loaded = load_metadata(path)
        np.testing.assert_array_equal(loaded["z_true"], metadata["z_true"])


if __name__ == "__main__":
    unittest.main()
