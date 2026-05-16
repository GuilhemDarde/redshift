import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from model import (
    ConditionalFlowMatching,
    OT_CFM_Physics_Wrapper,
    PhysicsInformedLoss,
    get_timestep_embedding,
)
from marie_treyer_exact import build_marie_treyer_model, marie_point_estimate, marie_z_centers, marie_z_edges


class SpyPhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, x_pred, cond):
        self.received = x_pred
        return x_pred.mean() * 0.0


def make_batch(batch_size=2):
    x = torch.randn(batch_size, 6, 64, 64)
    cond = torch.zeros(batch_size, 7)
    cond[:, 0] = torch.linspace(0.2, 1.0, batch_size)
    cond[:, 1] = 0.0
    return x, cond


class ModelSmokeTests(unittest.TestCase):
    def test_timestep_embedding_shape(self):
        t = torch.tensor([0.0, 500.0, 1000.0])
        emb = get_timestep_embedding(t, 128)
        self.assertEqual(emb.shape, (3, 128))
        self.assertTrue(torch.isfinite(emb).all())

    def test_cfm_forward_and_endpoint_shape(self):
        model = ConditionalFlowMatching()
        x, cond = make_batch()

        loss, x1_pred = model(x, cond, return_x1_pred=True)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(x1_pred.shape, x.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x1_pred).all())

    def test_cfm_marie_condition_and_multiband_loss_smoke(self):
        model = ConditionalFlowMatching(condition_dim=10)
        wrapper = OT_CFM_Physics_Wrapper(
            model,
            lambda_photo=0.1,
            lambda_color=0.01,
            mag_zp=torch.full((6,), 25.0),
        )
        x = torch.randn(2, 6, 64, 64)
        cond = torch.zeros(2, 10)
        cond[:, 0] = torch.tensor([0.3, 0.8])
        target_mags = torch.full((2, 6), 22.0)

        loss, metrics = wrapper(x, cond, target_mags=target_mags)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_photo", metrics)
        self.assertIn("loss_color", metrics)

    def test_physics_wrapper_uses_predicted_endpoint_not_target(self):
        base = ConditionalFlowMatching()
        wrapper = OT_CFM_Physics_Wrapper(base, lambda_photo=0.1)
        spy_loss = SpyPhysicsLoss()
        wrapper.photo_loss_fn = spy_loss
        x, cond = make_batch()

        loss, metrics = wrapper(x, cond)

        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(spy_loss.received)
        self.assertEqual(spy_loss.received.shape, x.shape)
        self.assertNotEqual(spy_loss.received.data_ptr(), x.data_ptr())
        self.assertEqual(set(metrics), {"loss_total", "loss_vf", "loss_photo"})

    def test_physics_loss_is_scalar_and_finite(self):
        x, cond = make_batch()
        loss = PhysicsInformedLoss()(x, cond)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_generation_smoke_shape(self):
        model = ConditionalFlowMatching()
        _, cond = make_batch(batch_size=1)

        generated = model.generate(cond, num_steps=2)

        self.assertEqual(generated.shape, (1, 6, 64, 64))
        self.assertTrue(torch.isfinite(generated).all())

    def test_image_to_image_augmentation_smoke_shape(self):
        model = ConditionalFlowMatching()
        x, cond = make_batch(batch_size=1)

        augmented = model.augment_image_to_image(x, cond, t0=0.5, noise_scale=0.01, num_steps=1)

        self.assertEqual(augmented.shape, x.shape)
        self.assertTrue(torch.isfinite(augmented).all())

    def test_marie_treyer_exact_forward_shape(self):
        model = build_marie_treyer_model(n_bins=12, mags_input_size=6)
        x = torch.randn(1, 6, 64, 64)
        ebv = torch.zeros(1)
        mags = torch.randn(1, 6)
        logits, z_reg = model(x, ebv, mags=mags)
        centers = torch.tensor(marie_z_centers(marie_z_edges(12)), dtype=torch.float32)
        z_pred = marie_point_estimate(logits, centers)

        self.assertEqual(logits.shape, (1, 12))
        self.assertEqual(z_reg.shape, (1, 1))
        self.assertEqual(z_pred.shape, (1,))
        self.assertTrue(torch.isfinite(z_pred).all())


if __name__ == "__main__":
    unittest.main()
