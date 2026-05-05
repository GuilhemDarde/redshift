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


if __name__ == "__main__":
    unittest.main()
