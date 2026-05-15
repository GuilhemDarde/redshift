import os
import sys
import tarfile
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from analyze_marie_cv_folds import build_marie_cv_concat
from analysis_utils import compute_marie_regular_cv_indices
from analyze_treyer_figure7 import aggregate_treyer_bins, magnitude_support_rows
from data_loader import field_label_from_filename, infer_field_labels, normalize_field_label


class TreyerProtocolTests(unittest.TestCase):
    def test_normalize_field_label_recognizes_cosmos_ud(self):
        self.assertEqual(normalize_field_label("COSMOS Ultra Deep"), "cosmos_ud")
        self.assertEqual(normalize_field_label("UDF"), "cosmos_ud")

    def test_field_label_from_filename_prefers_ud_suffix(self):
        self.assertEqual(field_label_from_filename("COSMOS_v11_uijk_0223_spec_UD.npz"), "cosmos_ud")
        self.assertEqual(field_label_from_filename("COSMOS_v11_uijk_0223_cos2020_D.npz"), "cosmos_deep")

    def test_infer_field_labels_from_boolean_column(self):
        info = np.zeros(3, dtype=[("cosmos_ud", "i4")])
        info["cosmos_ud"] = np.array([1, 0, 1])
        labels = infer_field_labels(info, ("cosmos_ud",), "x.npz")
        np.testing.assert_array_equal(labels, np.array(["cosmos_ud", "other", "cosmos_ud"]))

    def test_infer_field_labels_prefers_filename_over_survey_column(self):
        info = np.zeros(2, dtype=[("field", "U8")])
        info["field"] = np.array(["vuds", "c3r2"])
        labels = infer_field_labels(info, ("field",), "COSMOS_v11_uijk_0223_spec_UD.npz")
        np.testing.assert_array_equal(labels, np.array(["cosmos_ud", "cosmos_ud"]))

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

    def test_marie_regular_cv_split_uses_one_fold_as_eval(self):
        split = compute_marie_regular_cv_indices(10, n_folds=5, fold_id=2, seed=42)
        self.assertEqual(len(split["test"]), 2)
        self.assertEqual(split["test"].tolist(), split["val"].tolist())
        self.assertEqual(len(np.intersect1d(split["train"], split["test"])), 0)
        self.assertEqual(len(np.union1d(split["train"], split["test"])), 10)

    def _write_mini_marie_fold(self, base_dir, fold, offset=0):
        fold_dir = os.path.join(base_dir, str(fold))
        os.makedirs(fold_dir, exist_ok=True)
        data = {
            "z_true": np.array([0.2 + offset, 0.4 + offset]),
            "z_pred": np.array([0.21 + offset, 0.39 + offset]),
            "i": np.array([22.0 + fold, 23.0 + fold]),
            "field": np.array(["cosmos_ud", "cosmos_ud"]),
            "index": np.array([10 * fold, 10 * fold + 1]),
        }
        np.save(os.path.join(fold_dir, "epoch_50_val_results.npy"), data)
        np.save(os.path.join(fold_dir, "val_indices.npy"), np.array([10 * fold, 10 * fold + 1]))

    def test_build_marie_cv_concat_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "folds")
            self._write_mini_marie_fold(source, 0)
            self._write_mini_marie_fold(source, 1, offset=0.5)

            output = os.path.join(tmp, "out")
            predictions_path, metadata_path, fold_rows = build_marie_cv_concat(
                output_dir=output,
                folds=(0, 1),
                folds_dir=source,
                strict_unique_indices=True,
            )

            predictions = np.load(predictions_path, allow_pickle=False)
            metadata = np.load(metadata_path, allow_pickle=False)
            self.assertEqual(len(predictions["z_true"]), 4)
            self.assertEqual(len(metadata["mag_i"]), 4)
            self.assertEqual(len(fold_rows), 3)
            self.assertIn("metrics_by_fold.csv", os.listdir(output))

    def test_build_marie_cv_concat_from_tar_with_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "folds")
            self._write_mini_marie_fold(source, 0)
            self._write_mini_marie_fold(source, 1, offset=0.5)
            tar_path = os.path.join(tmp, "folds.tar")
            with tarfile.open(tar_path, "w") as tf:
                tf.add(source, arcname="marie_export")

            output = os.path.join(tmp, "out")
            predictions_path, metadata_path, _ = build_marie_cv_concat(
                output_dir=output,
                folds=(0, 1),
                folds_tar=tar_path,
            )

            predictions = np.load(predictions_path, allow_pickle=False)
            metadata = np.load(metadata_path, allow_pickle=False)
            self.assertEqual(predictions["val_index"].tolist(), [0, 1, 10, 11])
            self.assertEqual(metadata["fold_id"].tolist(), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
