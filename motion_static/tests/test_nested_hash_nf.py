"""Check warp gradients, zero reference motion, and hash queries outside the FOV."""
import unittest

import torch

from motion_static.models.hash_nf import HashGridEncoding
from motion_static.models.nested_hash_nf import NestedHashNF


def small_model():
    return NestedHashNF(width=16, levels=2, min_resolution=4, max_resolution=8,
                        log2_hash_size=9)


class NestedHashTests(unittest.TestCase):
    def test_image_loss_reaches_both_tables_and_both_motion_outputs(self):
        torch.manual_seed(0)
        model = small_model()
        coordinates = torch.rand(64, 3) - .5
        target = coordinates[:, 0] + coordinates[:, 1].square()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        # A zero final motion layer initially blocks its own table gradient;
        # after the first update that table must also learn through image loss.
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            (model(coordinates) - target).square().mean().backward()
            optimizer.step()
        for field in (model.intensity, model.motion.field):
            self.assertGreater(field.encoding.table.grad.abs().sum().item(), 0)
            self.assertGreater(field.mlp[-1].weight.grad.abs().sum().item(), 0)
        for direction in (0, 1):
            self.assertGreater(model.motion.field.mlp[-1].weight.grad[direction].abs().sum().item(), 0)
        reference = coordinates.clone()
        reference[:, 2] = -.5
        torch.testing.assert_close(model.motion(reference), torch.zeros(64, 2), atol=0, rtol=0)

    def test_composed_motion_gradient_matches_finite_difference(self):
        torch.manual_seed(7)
        model = small_model().double()
        coordinates = torch.tensor([[.113, -.217, .191]], dtype=torch.float64)
        with torch.no_grad():
            model.intensity.encoding.table.uniform_(-.2, .2)
        model(coordinates).sum().backward()
        weight = model.motion.field.mlp[-1].weight
        for direction in (0, 1):
            index = int(weight.grad[direction].abs().argmax())
            expected = weight.grad[direction, index].item()
            self.assertGreater(abs(expected), 1e-12)
            eps = 1e-5
            with torch.no_grad():
                weight[direction, index] += eps
                plus = model(coordinates).item()
                weight[direction, index] -= 2 * eps
                minus = model(coordinates).item()
                weight[direction, index] += eps
            actual = (plus - minus) / (2 * eps)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=.001, atol=1e-10)

    def test_outside_queries_keep_coordinate_gradients(self):
        torch.manual_seed(3)
        encoder = HashGridEncoding(levels=2, features=2, min_resolution=3,
                                   max_resolution=7, log2_hash_size=5,
                                   clamp_coordinates=False).double()
        point = torch.tensor([[-.13, .23, 1.17]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(encoder, (point,), eps=1e-6, atol=1e-7))
        encoder(point).sum().backward()
        self.assertGreater(abs(point.grad[0, 0].item()), 1e-8)
        self.assertGreater(abs(point.grad[0, 2].item()), 1e-8)

    def test_dense_boundary_is_continuous_when_outside_vertices_are_hashed(self):
        encoder = HashGridEncoding(levels=1, features=1, min_resolution=2,
                                   max_resolution=2, log2_hash_size=5,
                                   clamp_coordinates=False).double()
        with torch.no_grad():
            encoder.table[:, 0] = torch.arange(len(encoder.table), dtype=torch.float64)
        for boundary in (0., 1.):
            points = torch.tensor([[boundary - 1e-9, .23, .37],
                                   [boundary, .23, .37],
                                   [boundary + 1e-9, .23, .37]], dtype=torch.float64)
            values = encoder(points)
            torch.testing.assert_close(values, values[1:2].expand_as(values), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
