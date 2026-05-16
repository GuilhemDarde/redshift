import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from cfm_conditioning import build_cfm_condition, build_cfm_photo_targets, photometric_features_from_cfm_condition


class CFMConditioningTests(unittest.TestCase):
    def test_marie_mags_condition_uses_six_magnitudes_and_metadata(self):
        data = {
            "cond": np.zeros((2, 7), dtype=np.float32),
            "z_true": np.array([0.2, 0.7], dtype=np.float32),
            "mags_marie": np.array(
                [
                    [24.0, 23.0, 22.5, 22.0, 21.8, 21.7],
                    [25.0, 24.0, 23.5, 23.0, 22.8, 22.6],
                ],
                dtype=np.float32,
            ),
            "ebv": np.array([0.01, 0.02], dtype=np.float32),
            "re_norm": np.array([0.3, -0.1], dtype=np.float32),
            "n_norm": np.array([0.4, 0.5], dtype=np.float32),
        }

        cond = build_cfm_condition(data, schema="marie_mags")
        targets = build_cfm_photo_targets(data, schema="marie_mags")
        features = photometric_features_from_cfm_condition(cond, schema="marie_mags")

        self.assertEqual(cond.shape, (2, 10))
        np.testing.assert_allclose(cond[:, 0], data["z_true"])
        np.testing.assert_allclose(cond[:, 1:7], (data["mags_marie"] - 22.0) / 2.0)
        np.testing.assert_allclose(targets, data["mags_marie"])
        np.testing.assert_allclose(features["mag_i"], data["mags_marie"][:, 3])
        np.testing.assert_allclose(features["g_r"], data["mags_marie"][:, 1] - data["mags_marie"][:, 2])


if __name__ == "__main__":
    unittest.main()
