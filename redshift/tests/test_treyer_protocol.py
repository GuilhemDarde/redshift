import os
import sys
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from analyze_treyer_figure7 import aggregate_treyer_bins, magnitude_support_rows
from data_loader import normalize_field_label, infer_field_labels


class TreyerProtocolTests(unittest.TestCase):
    def test_normalize_field_label_recognizes_cosmos_ud(self):
        self.assertEqual(normalize_field_label("COSMOS Ultra Deep"), "cosmos_ud")
        self.assertEqual(normalize_field_label("UDF"), "cosmos_ud")

    def test_infer_field_labels_from_boolean_column(self):
        info = np.zeros(3, dtype=[("cosmos_ud", "i4")])
        info["cosmos_ud"] = np.array([1, 0, 1])
        labels = infer_field_labels(info, ("cosmos_ud",), "x.npz")
        np.testing.assert_array_equal(labels, np.array(["cosmos_ud", "other", "cosmos_ud"]))

    def test_aggregate_treyer_bins_includes_bootstrap_columns(self):
        z_true = np.linspace(0.1, 1.0, 20)
        z_pred = z_true + 0.01
        rows = aggregate_treyer_bins(z_true, np.linspace(0.0, 1.2, 4), z_true, z_pred, n_bootstrap=3, seed=1)
        self.assertEqual(len(rows), 3)
        self.assertIn("mad", rows[0])
        self.assertIn("sigma_nmad_err", rows[0])

    def test_magnitude_support_rows_marks_low_support_bins(self):
        train_mag = np.array([18.1, 18.2, 18.3, 23.9])
        mag = np.array([18.15, 23.95])
        z_true = np.array([0.2, 0.8])
        z_pred = np.array([0.21, 0.9])
        rows = magnitude_support_rows(mag, train_mag, z_true, z_pred, np.array([18.0, 19.0, 24.0, 25.0]), quantile=0.5)
        by_name = {row["subset"]: row for row in rows}
        self.assertEqual(by_name["low_mag_support"]["n"], 1)
        self.assertEqual(by_name["normal_mag_support"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
