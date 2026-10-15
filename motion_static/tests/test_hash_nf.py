"""Numerical checks for hash interpolation and gradients."""
import unittest

import torch

from motion_static.models.hash_nf import HashGridEncoding, HashNeuralField


class HashGridTests(unittest.TestCase):
    def test_dense_affine_interpolation_and_boundary(self):
        encoder = HashGridEncoding(levels=1, features=1, min_resolution=2,
                                   max_resolution=2, log2_hash_size=5).double()
        # Dense indexing is x + side * (y + side * z).
        with torch.no_grad():
            for z in range(3):
                for y in range(3):
                    for x in range(3):
                        encoder.table[x + 3 * (y + 3 * z), 0] = (x + 2*y + 3*z) / 2
        points = torch.tensor([[0., 0., 0.], [1., 1., 1.],
                               [.23, .61, .89], [1., .2, .7]], dtype=torch.float64)
        torch.testing.assert_close(encoder(points)[:, 0], points @ torch.tensor([1., 2., 3.], dtype=torch.float64))

    def test_hash_collisions_preserve_constant_field_and_gradients(self):
        encoder = HashGridEncoding(levels=3, features=2, min_resolution=4,
                                   max_resolution=16, log2_hash_size=2).double()
        with torch.no_grad():
            encoder.table.fill_(2.5)
        points = torch.tensor([[.19, .43, .77], [1., 1., 1.]], dtype=torch.float64)
        values = encoder(points)
        torch.testing.assert_close(values, torch.full((2, 6), 2.5, dtype=torch.float64))
        values.sum().backward()
        self.assertTrue(torch.isfinite(encoder.table.grad).all())
        self.assertAlmostEqual(encoder.table.grad.sum().item(), 12.)

    def test_coordinate_gradient(self):
        encoder = HashGridEncoding(levels=2, features=2, min_resolution=3,
                                   max_resolution=7, log2_hash_size=4).double()
        points = torch.tensor([[.19, .43, .77]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(encoder, (points,), eps=1e-6, atol=1e-6))

    def test_both_hash_and_mlp_receive_gradients(self):
        torch.manual_seed(0)
        model = HashNeuralField(levels=2, log2_hash_size=4)
        points = torch.rand(32, 3) - .5
        model(points).square().mean().backward()
        self.assertGreater(model.encoding.table.grad.abs().sum().item(), 0)
        self.assertGreater(model.mlp[-1].weight.grad.abs().sum().item(), 0)



if __name__ == "__main__":
    unittest.main()
