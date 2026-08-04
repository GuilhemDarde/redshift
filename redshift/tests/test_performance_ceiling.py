import os
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from diagnose_performance_ceiling import (
    local_redshift_ambiguity,
    photometric_features_from_metadata,
    prediction_indices,
    run,
)


def mini_metadata():
    n = 12
    split = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 4)
    mag_i = np.array([20.0, 20.1, 20.2, 22.0, 22.2, 23.5, 21.0, 22.4, 23.8, 24.0, 24.1, 24.3])
    mag_g = mag_i + np.array([0.6, 0.62, 0.58, 1.0, 1.1, 1.2, 0.7, 0.8, 1.4, 1.5, 1.6, 1.7])
    mag_r = mag_i + np.array([0.25, 0.27, 0.24, 0.5, 0.55, 0.6, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9])
    mag_z = mag_i - np.array([0.15, 0.16, 0.14, 0.2, 0.25, 0.3, 0.16, 0.22, 0.3, 0.35, 0.4, 0.45])
    return {
        "z_true": np.array([0.10, 0.12, 0.11, 0.55, 0.58, 1.20, 0.20, 0.60, 1.25, 1.30, 1.80, 2.10]),
        "split": split,
        "mag_i": mag_i,
        "mag_u": mag_g + 0.4,
        "mag_g": mag_g,
        "mag_r": mag_r,
        "mag_z": mag_z,
        "mag_y": mag_z - 0.1,
        "ra": np.linspace(150.0, 151.0, n),
        "dec": np.linspace(2.0, 2.1, n),
    }


class PerformanceCeilingTests(unittest.TestCase):
    def test_prediction_indices_prefers_explicit_test_indices(self):
        metadata = mini_metadata()
        predictions = {
            "z_true": metadata["z_true"][[9, 10]],
            "z_pred": metadata["z_true"][[9, 10]],
            "test_indices": np.array([9, 10]),
        }
        np.testing.assert_array_equal(prediction_indices(predictions, metadata), np.array([9, 10]))

    def test_classic_photometric_features_are_expected_colors(self):
        metadata = mini_metadata()
        features, names = photometric_features_from_metadata(metadata, "classic_colors")
        self.assertEqual(names, ["mag_i", "g-r", "r-i", "i-z"])
        self.assertEqual(features.shape, (12, 4))
        np.testing.assert_allclose(features[:, 0], metadata["mag_i"])
        np.testing.assert_allclose(features[:, 1], metadata["mag_g"] - metadata["mag_r"])

    def test_local_redshift_ambiguity_returns_normalized_dispersion(self):
        z = np.array([0.1, 0.2, 1.0, 1.2])
        neighbors = np.array([[0, 1], [2, 3]])
        ambiguity = local_redshift_ambiguity(z, neighbors)
        self.assertGreater(ambiguity["local_z_nmad_norm"][0], 0.0)
        self.assertGreater(ambiguity["local_z_nmad_norm"][1], 0.0)
        self.assertLess(ambiguity["local_z_nmad_norm"][1], ambiguity["local_z_nmad"][1])

    def test_run_writes_core_outputs(self):
        metadata = mini_metadata()
        test_idx = np.where(metadata["split"] == "test")[0]
        z_true = metadata["z_true"][test_idx]
        z_pred = z_true + np.array([0.03, -0.02, 0.10, -0.12])
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = os.path.join(tmp, "metadata.npz")
            predictions_path = os.path.join(tmp, "predictions.npz")
            output_dir = os.path.join(tmp, "ceiling")
            np.savez(metadata_path, **metadata)
            np.savez(predictions_path, z_true=z_true, z_pred=z_pred, test_indices=test_idx)

            run(
                Namespace(
                    predictions=predictions_path,
                    metadata=metadata_path,
                    output_dir=output_dir,
                    feature_space="classic_colors",
                    k=2,
                    low_fraction=0.25,
                    curve_bins=3,
                    surface_bins=4,
                    min_cell_count=1,
                    max_scatter_points=100,
                    ambiguity_floor=1e-4,
                    seed=42,
                )
            )

            self.assertTrue(os.path.exists(os.path.join(output_dir, "ceiling_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "ceiling_by_support_radius.csv")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "ceiling_color_color_surfaces.png")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "ceiling_diagnostic_report.md")))


if __name__ == "__main__":
    unittest.main()
