import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from config import CONFIG
from experiment_marie_augmented import load_synthetic_dataset
from photometric_validation import (
    acceptance_mask,
    calibrate_zero_points,
    image_magnitudes,
    photometric_features_from_conditions,
    photometric_features_from_magnitudes,
    residual_thresholds,
    residuals_from_features,
)


def make_images_from_fluxes(fluxes: np.ndarray) -> np.ndarray:
    images = np.zeros((len(fluxes), len(CONFIG.BAND_NAMES), 4, 4), dtype=np.float32)
    images[:, :, 1, 1] = fluxes
    return np.arcsinh(images) if CONFIG.ASINH_NORM else images


class PhotometricValidationTests(unittest.TestCase):
    def test_zero_point_calibration_and_magnitudes_are_finite(self):
        fluxes = np.array([[10.0, 8.0, 6.0, 4.0, 3.0, 2.0], [12.0, 9.0, 7.0, 5.0, 4.0, 3.0]])
        zero_point = np.full(6, 25.0)
        mags = zero_point[None, :] - 2.5 * np.log10(fluxes)
        images = make_images_from_fluxes(fluxes)

        calibrated = calibrate_zero_points(images, mags)
        recovered, measured_fluxes = image_magnitudes(images, calibrated)

        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.isfinite(recovered).all())
        self.assertTrue(np.isfinite(measured_fluxes).all())
        np.testing.assert_allclose(calibrated, zero_point, atol=1e-5)
        np.testing.assert_allclose(recovered, mags, atol=1e-5)

    def test_acceptance_filter_keeps_clone_and_rejects_bad_photometry(self):
        cond = np.array([
            [0.5, 0.0, 0.2, 0.1, -0.1, 0.0, 0.0],
            [0.5, 0.0, 0.2, 0.1, -0.1, 0.0, 0.0],
        ], dtype=np.float32)
        target = photometric_features_from_conditions(cond)
        good_mags = np.array([[22.4, 22.3, 22.1, 22.0, 22.1, 22.2]])
        observed_good = photometric_features_from_magnitudes(good_mags)
        real_residuals = residuals_from_features(observed_good, {k: v[:1] for k, v in target.items()})
        thresholds = residual_thresholds(real_residuals, quantile=0.95)

        observed = {key: np.array([target[key][0], target[key][1] + 5.0]) for key in target}
        residuals = residuals_from_features(observed, target)
        fluxes = np.ones((2, 6), dtype=np.float64)
        images = make_images_from_fluxes(fluxes)
        mask = acceptance_mask(residuals, thresholds, fluxes, images)

        np.testing.assert_array_equal(mask, np.array([True, False]))

    def test_synthetic_loader_filters_mode_without_touching_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.npz")
            np.savez(
                path,
                x=np.zeros((3, 6, 4, 4), dtype=np.float32),
                cond=np.zeros((3, 7), dtype=np.float32),
                mode=np.array(["i2i", "interp", "i2i"]),
            )

            ds = load_synthetic_dataset(path, mode_filter="i2i")

        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0][0].shape, (6, 4, 4))
        self.assertEqual(ds[0][1].shape, (7,))


if __name__ == "__main__":
    unittest.main()
