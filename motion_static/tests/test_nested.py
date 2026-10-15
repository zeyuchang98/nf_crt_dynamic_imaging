"""Image-loss gradients must reach motion; reference anchoring must survive updates."""
import unittest

import numpy as np
import torch

from motion_static.models.nested_mlp import NestedMLP
from motion_static.evaluation.evaluate import motion_scores


class NestedTests(unittest.TestCase):
    def test_image_loss_trains_both_networks_and_keeps_reference_zero(self):
        torch.manual_seed(0)
        model = NestedMLP(intensity_width=16, motion_width=8)
        coordinates = torch.rand(50, 3) - .5
        target = coordinates[:, 0] + coordinates[:, 1].square()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = (model(coordinates) - target).square().mean()
        loss.backward()
        for component in (model.intensity, model.motion):
            self.assertGreater(component.net[-1].weight.grad.abs().sum().item(), 0)
        for direction in (0, 1):
            self.assertGreater(model.motion.net[-1].weight.grad[direction].abs().sum().item(), 0)
        optimizer.step()
        reference = coordinates.clone()
        reference[:, 2] = -.5
        torch.testing.assert_close(model.motion(reference), torch.zeros(50, 2), atol=0, rtol=0)

    def test_motion_gradient_matches_finite_difference(self):
        torch.manual_seed(7)
        model = NestedMLP(intensity_width=16, motion_width=8).double()
        coordinates = torch.tensor([[.14, -.18, .23]], dtype=torch.float64)
        value = model(coordinates).sum()
        value.backward()
        weight = model.motion.net[-1].weight
        expected = weight.grad[0, 0].item()
        eps = 1e-6
        with torch.no_grad():
            weight[0, 0] += eps
            plus = model(coordinates).item()
            weight[0, 0] -= 2 * eps
            minus = model(coordinates).item()
            weight[0, 0] += eps
        self.assertAlmostEqual(expected, (plus - minus) / (2 * eps), places=7)

    def test_endpoint_error_pixel_units_and_foreground_mask(self):
        truth = np.zeros((4, 4, 3, 2))
        prediction = np.ones_like(truth) * .1
        mask = np.zeros((4, 4, 3), dtype=bool)
        mask[1:3, 1:3] = True
        prediction[0, 0] = 10
        result = motion_scores(truth, prediction, mask, spacing=.1)
        self.assertAlmostEqual(result["foreground"]["endpoint_rmse_pixels"], np.sqrt(2))
        self.assertGreater(result["whole_fov"]["endpoint_rmse_pixels"], 10)


if __name__ == "__main__":
    unittest.main()
