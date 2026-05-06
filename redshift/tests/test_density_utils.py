import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from density_utils import compute_catalog_knn_density, compute_train_knn_density, low_density_mask, projected_radec_coordinates


class DensityUtilsTests(unittest.TestCase):
    def test_projected_radec_wraparound_is_finite(self):
        coords = projected_radec_coordinates(np.array([359.9, 0.1]), np.array([0.0, 0.0]), ra_ref=0.0, dec_ref=0.0)
        self.assertEqual(coords.shape, (2, 2))
        self.assertTrue(np.isfinite(coords).all())
        self.assertLess(abs(coords[0, 0]), 1.0)

    def test_knn_density_and_low_density_quantile(self):
        ra = np.array([0.0, 0.01, 0.02, 10.0, 20.0, 30.0])
        dec = np.zeros_like(ra)
        train_idx = np.arange(len(ra))

        density, radius = compute_train_knn_density(ra, dec, train_idx, k=1)
        mask, threshold = low_density_mask(density, train_idx, quantile=0.5)

        self.assertEqual(density.shape, ra.shape)
        self.assertEqual(radius.shape, ra.shape)
        self.assertTrue(np.isfinite(density).all())
        self.assertTrue(np.isfinite(threshold))
        self.assertTrue(mask[-1])
        self.assertFalse(mask[1])

    def test_catalog_density_leave_one_out(self):
        ra = np.array([0.0, 0.01, 0.02, 10.0])
        dec = np.zeros_like(ra)

        density, radius = compute_catalog_knn_density(ra, dec, k=1)

        self.assertEqual(density.shape, ra.shape)
        self.assertEqual(radius.shape, ra.shape)
        self.assertTrue(np.isfinite(density).all())
        self.assertGreater(density[1], density[-1])


if __name__ == "__main__":
    unittest.main()
