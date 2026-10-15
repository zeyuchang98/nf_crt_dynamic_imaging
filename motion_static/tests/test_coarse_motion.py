"""Per-axis interpolation, gradients, branch isolation and checkpoint compatibility."""
import io
import unittest

import torch

from motion_static.models.hash_grid import HashGridEncoding
from motion_static.run import build_model, train
from motion_static.tests.test_low_rank import config


def coarse_config():
    settings = config(rank=2)
    settings["motion_encoding"] = dict(min_resolution=[2, 2, 3], max_resolution=[3, 3, 7])
    return settings


class CoarseMotionTests(unittest.TestCase):
    def test_rectangular_grid_reproduces_trilinear_function(self):
        encoder = HashGridEncoding(levels=1, features=1, min_resolution=[2, 3, 4],
                                   max_resolution=[2, 3, 4], log2_hash_size=7).double()

        def polynomial(points):
            x, y, t = points.unbind(-1)
            return x + 2*y + 3*t + 4*x*y + 5*y*t + 6*x*t + 7*x*y*t

        with torch.no_grad():
            for t in range(5):
                for y in range(4):
                    for x in range(3):
                        encoder.table[x + 3*(y + 4*t), 0] = polynomial(
                            torch.tensor([x/2, y/3, t/4], dtype=torch.float64))
        points = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., .21, .77],
                               [.13, .61, .83]], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(encoder(points)[:, 0], polynomial(points), atol=1e-12, rtol=0)
        # Clamp endpoint subgradients are backend-defined; test the analytic
        # derivative strictly inside the FOV and endpoint values above.
        interior = points[-1:].detach().requires_grad_()
        predicted = torch.autograd.grad(encoder(interior).sum(), interior)[0]
        expected = torch.autograd.grad(polynomial(interior).sum(), interior)[0]
        torch.testing.assert_close(predicted, expected, atol=1e-12, rtol=0)

    def test_scalar_and_equal_axis_grids_are_exactly_equivalent(self):
        for dimensions in (2, 3):
            for clamp in (False, True):
                settings = dict(dimensions=dimensions, levels=3, features=2, log2_hash_size=5,
                                clamp_coordinates=clamp)
                torch.manual_seed(19)
                scalar = HashGridEncoding(min_resolution=2, max_resolution=9, **settings).double()
                torch.manual_seed(19)
                axes = HashGridEncoding(min_resolution=[2]*dimensions,
                                        max_resolution=[9]*dimensions, **settings).double()
                points = torch.rand(37, dimensions, dtype=torch.float64) * 1.4 - .2
                points[:2] = torch.tensor([[0.]*dimensions, [1.]*dimensions])
                points.requires_grad_()
                torch.testing.assert_close(scalar(points), axes(points), atol=0, rtol=0)
                torch.testing.assert_close(torch.autograd.grad(scalar(points).sum(), points)[0],
                                           torch.autograd.grad(axes(points).sum(), points)[0], atol=0, rtol=0)
                self.assertEqual(scalar.resolutions.ndim, 1)
                torch.testing.assert_close(scalar.table, axes.table, atol=0, rtol=0)

    def test_anisotropic_hashed_coordinate_gradient(self):
        encoder = HashGridEncoding(levels=3, features=2, log2_hash_size=3,
                                   min_resolution=[2, 3, 4], max_resolution=[7, 9, 15]).double()
        points = torch.tensor([[.137, .421, .719]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(encoder, (points,), eps=1e-6, atol=1e-7))

    def test_anisotropic_outside_boundary_continuity(self):
        encoder = HashGridEncoding(dimensions=2, levels=1, features=1, log2_hash_size=5,
                                   min_resolution=[2, 3], max_resolution=[2, 3],
                                   clamp_coordinates=False).double()
        with torch.no_grad():
            encoder.table[:, 0] = torch.arange(len(encoder.table), dtype=torch.float64)
        for boundary in (0., 1.):
            points = torch.tensor([[boundary - 1e-9, .23], [boundary, .23],
                                   [boundary + 1e-9, .23]], dtype=torch.float64)
            values = encoder(points)
            torch.testing.assert_close(values, values[1:2].expand_as(values), atol=1e-7, rtol=0)

    def test_motion_override_preserves_intensity_initialization_and_time_grid(self):
        torch.manual_seed(0)
        original = build_model(config(rank=2))
        torch.manual_seed(0)
        coarse = build_model(coarse_config())
        for key, value in original.intensity.state_dict().items():
            torch.testing.assert_close(value, coarse.intensity.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(original.motion.field.encoding.resolutions,
                                   coarse.motion.field.encoding.resolutions[:, 2], atol=0, rtol=0)
        self.assertLess(coarse.motion.field.encoding.resolutions[-1, 0],
                        original.motion.field.encoding.resolutions[-1])

    def test_training_gradients_rank_anchor_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        settings = coarse_config()
        model = build_model(settings)
        points = torch.rand(64, 3) - .5
        target = points[:, 0] + points[:, 1].square() + points[:, 2]
        train(model, points.numpy(), target.numpy(), settings)
        for field in (model.intensity, model.motion.field):
            self.assertTrue(torch.isfinite(field.encoding.table.grad).all())
            self.assertGreater(field.encoding.table.grad.abs().sum().item(), 0)
        anchored = points.clone()
        anchored[:, 2] = settings["reference_time"]
        torch.testing.assert_close(model.motion(anchored), torch.zeros(64, 2), atol=0, rtol=0)
        query = points.clone().requires_grad_()
        gradient = torch.autograd.grad(model(query).sum(), query)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[:, :2].abs().sum().item(), 0)
        matrix = model.intensity.spatial_basis(points[:, :2]) @ model.intensity.temporal_coefficients(points[:, 2:]).T
        singular = torch.linalg.svdvals(matrix.detach().double())
        self.assertLess((singular[2:].norm() / singular.norm()).item(), 1e-6)
        stream = io.BytesIO()
        torch.save(dict(config=settings, state_dict=model.state_dict()), stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        restored = build_model(checkpoint["config"])
        restored.load_state_dict(checkpoint["state_dict"])
        torch.testing.assert_close(model(points), restored(points), atol=0, rtol=0)

    def test_invalid_axis_resolutions(self):
        for low, high in (([2, 3], [4, 5]), ([2, 0, 3], [4, 5, 6]),
                          ([2, 3.5, 4], [4, 5, 6]), ([2, 3, 4], [4, 2, 6])):
            with self.assertRaises(ValueError):
                HashGridEncoding(min_resolution=low, max_resolution=high)


if __name__ == "__main__":
    unittest.main()
