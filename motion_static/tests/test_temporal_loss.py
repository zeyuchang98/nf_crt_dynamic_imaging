"""Check reference coordinates, time units, boundaries, and branch gradients."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from motion_static.losses import compute_training_losses, intensity_temporal_l2
from motion_static.models.nested_hash_nf import NestedHashNF


class LinearIntensity(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(float(rate)))
        self.queries = []

    def forward(self, xyt):
        self.queries.append(xyt.detach().clone())
        return 2 * xyt[:, 0] - xyt[:, 1] + self.rate * xyt[:, 2]


class LinearMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.speed = nn.Parameter(torch.tensor(.2))

    def forward(self, xyt):
        return torch.stack((self.speed * xyt[:, 2], torch.zeros_like(xyt[:, 2])), dim=1)


class LinearNestedField(nn.Module):
    """Analytic fixture: I(r,t) = 2(x + speed*t) - y + rate*t."""
    def __init__(self):
        super().__init__()
        self.motion = LinearMotion()
        self.intensity = LinearIntensity(3.)

    def forward(self, xyt):
        reference_xyt = xyt.clone()
        reference_xyt[:, :2] += self.motion(xyt)
        return self.intensity(reference_xyt)


class TemporalLossTests(unittest.TestCase):
    def test_composed_objective_matches_analytic_values_and_branch_gradients(self):
        model = LinearNestedField()
        coordinates = torch.tensor([[.2, 0., .4]])
        config = dict(temporal_weight=.25, temporal_step=.1,
                      coordinate_bounds=[[-.5, .5], [-.5, .5], [0., 1.]])
        losses = compute_training_losses(model, coordinates, torch.tensor([.5]), config)
        # Predicted image = 1.76, residual = 1.26, temporal derivative = 3.
        self.assertAlmostEqual(losses["batch_mse"].item(), 1.5876, places=5)
        self.assertAlmostEqual(losses["temporal_l2"].item(), 9., places=4)
        self.assertAlmostEqual(losses["total_loss"].item(), 3.8376, places=4)
        losses["total_loss"].backward()
        # Motion receives image gradients only; intensity also receives 2*w*rate.
        self.assertAlmostEqual(model.motion.speed.grad.item(), 2.016, places=5)
        self.assertAlmostEqual(model.intensity.rate.grad.item(), 2.508, places=4)

    def test_default_objective_skips_temporal_queries(self):
        model = LinearNestedField()
        coordinates = torch.tensor([[.2, 0., .4]])
        with patch("motion_static.losses.intensity_temporal_l2", side_effect=AssertionError("disabled term ran")):
            losses = compute_training_losses(model, coordinates, torch.tensor([.5]), {})
        self.assertEqual(len(model.intensity.queries), 1)
        self.assertEqual(losses["temporal_l2"].item(), 0.)
        self.assertEqual(losses["batch_mse"].item(), losses["total_loss"].item())
        losses["total_loss"].backward()
        self.assertAlmostEqual(model.intensity.rate.grad.item(), 1.008, places=5)

    def model(self, rate):
        motion = nn.Linear(3, 2)
        return SimpleNamespace(intensity=LinearIntensity(rate), motion=motion)

    def test_static_intensity_is_zero_despite_time_dependent_motion(self):
        model = self.model(0.)
        xyt = torch.rand(40, 3)
        self.assertEqual(intensity_temporal_l2(model, xyt, .1, 1.).item(), 0.)

    def test_rate_uses_time_units_and_does_not_update_motion(self):
        model = self.model(3.)
        xyt = torch.tensor([[.2, .4, 10.], [.4, .1, 12.]], requires_grad=True)
        loss = intensity_temporal_l2(model, xyt, .5, 12.)
        self.assertAlmostEqual(loss.item(), 9., places=4)
        loss.backward()
        self.assertAlmostEqual(model.intensity.rate.grad.item(), 6., places=4)
        self.assertIsNone(xyt.grad)
        for parameter in model.motion.parameters():
            self.assertIsNone(parameter.grad)
        later, earlier = model.intensity.queries
        torch.testing.assert_close(later[:, :2], earlier[:, :2])
        torch.testing.assert_close(later[:, 2] - earlier[:, 2], torch.full((2,), .5))
        self.assertLessEqual(later[:, 2].max().item(), 12.)
        self.assertGreaterEqual(earlier[:, 2].min().item(), 10.)

    def test_hash_intensity_receives_gradients(self):
        torch.manual_seed(0)
        model = NestedHashNF(width=16, levels=2, min_resolution=4, max_resolution=8, log2_hash_size=9)
        loss = intensity_temporal_l2(model, torch.rand(64, 3) - .5, .05, .5)
        loss.backward()
        self.assertGreater(model.intensity.encoding.table.grad.abs().sum().item(), 0)
        for parameter in model.motion.parameters():
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
