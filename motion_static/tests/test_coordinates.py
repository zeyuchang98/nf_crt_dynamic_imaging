"""Coordinate units must not change the represented image or break motion units."""
import io
import unittest

import torch

from motion_static.models.coordinates import CoordinateNormalizer
from motion_static.models.hash_nf import HashNeuralField
from motion_static.models.nested_mlp import NestedMLP
from motion_static.models.nested_hash_nf import NestedHashNF
from motion_static.models.nested_low_rank_hash_nf import NestedLowRankHashNF
from motion_static.run import build_model


BOUNDS = ((10., 14.), (-6., 2.), (20., 22.))
SMALL_GRID = dict(levels=2, min_resolution=4, max_resolution=8, log2_hash_size=9)


def make_model(kind, **kwargs):
    """Use nonzero motion so unit conversion is actually exercised."""
    torch.manual_seed(12)
    if kind == "hash_nf":
        model = HashNeuralField(width=16, **SMALL_GRID, **kwargs)
    elif kind == "nested_mlp":
        model = NestedMLP(intensity_width=16, motion_width=8, **kwargs)
    elif kind == "nested_hash_nf":
        model = NestedHashNF(width=16, **SMALL_GRID, **kwargs)
    else:
        model = NestedLowRankHashNF(width=16, temporal_width=8, **SMALL_GRID, **kwargs)
    if kind != "hash_nf":
        head = model.motion.net[-1] if kind == "nested_mlp" else model.motion.field.mlp[-1]
        with torch.no_grad():
            head.weight.normal_(std=0.05)
    return model.double()


class CoordinateTests(unittest.TestCase):
    def test_fixed_bounds_map_endpoints_and_keep_outside_queries(self):
        normalizer = CoordinateNormalizer(BOUNDS).double()
        points = torch.tensor([[10., -6., 20.], [12., -2., 21.],
                               [14., 2., 22.], [16., -10., 23.]], dtype=torch.float64,
                              requires_grad=True)
        expected = torch.tensor([[0., 0., 0.], [.5, .5, .5],
                                 [1., 1., 1.], [1.5, -.5, 1.5]], dtype=torch.float64)
        torch.testing.assert_close(normalizer(points), expected)
        torch.testing.assert_close(normalizer.to_signed(points), 2 * expected - 1)
        # Changing the other queries in a batch must not change this point.
        torch.testing.assert_close(normalizer(points[1:2]), expected[1:2])
        normalizer(points).sum().backward()
        torch.testing.assert_close(points.grad, (1 / normalizer.span).expand_as(points))

    def test_zero_width_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            CoordinateNormalizer(((0, 0), (0, 1), (0, 1)))

    def test_images_motion_and_coordinate_gradients_are_invariant_to_units(self):
        lower = torch.tensor([10., -6., 20.], dtype=torch.float64)
        span = torch.tensor([4., 8., 2.], dtype=torch.float64)
        torch.manual_seed(3)
        points = torch.rand(32, 3, dtype=torch.float64) * .8 - .4
        for kind in ("hash_nf", "nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf"):
            with self.subTest(model=kind):
                original = make_model(kind)
                physical = make_model(kind, coordinate_bounds=BOUNDS)
                x = points.clone().requires_grad_()
                physical_x = (lower + (points + .5) * span).requires_grad_()
                y, physical_y = original(x), physical(physical_x)
                torch.testing.assert_close(y, physical_y, rtol=1e-8, atol=1e-10)
                grad = torch.autograd.grad(y.sum(), x)[0]
                physical_grad = torch.autograd.grad(physical_y.sum(), physical_x)[0]
                torch.testing.assert_close(grad, physical_grad * span, rtol=1e-7, atol=1e-10)
                if kind != "hash_nf":
                    motion = original.motion(x)
                    self.assertGreater(motion.abs().max().item(), 0)
                    torch.testing.assert_close(physical.motion(physical_x), motion * span[:2],
                                               rtol=1e-8, atol=1e-10)
                    reference = physical_x.detach().clone()
                    reference[:, 2] = 20.
                    torch.testing.assert_close(physical.motion(reference), torch.zeros(32, 2).double(),
                                               rtol=0, atol=0)

    def test_custom_reference_time_uses_input_units(self):
        points = torch.tensor([[11., -3., 21.], [13., 0., 21.]], dtype=torch.float64)
        for kind in ("nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf"):
            with self.subTest(model=kind):
                model = make_model(kind, coordinate_bounds=BOUNDS, reference_time=21.)
                torch.testing.assert_close(model.motion(points), torch.zeros(2, 2).double(), rtol=0, atol=0)
                points_at_start = points.clone()
                points_at_start[:, 2] = 20.
                self.assertGreater(model.motion(points_at_start).abs().max().item(), 0)

    def test_checkpoint_config_restores_bounds_and_reference_time(self):
        points = torch.tensor([[11., -3., 20.5], [13., 0., 21.5]])
        for kind in ("hash_nf", "nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf"):
            with self.subTest(model=kind):
                config = dict(model=kind, coordinate_bounds=BOUNDS, reference_time=21.,
                              intensity_width=16, motion_width=8, head_width=16, encoding=SMALL_GRID,
                              rank=3, temporal_width=8)
                model = build_model(config)
                stream = io.BytesIO()
                torch.save(dict(config=config, state_dict=model.state_dict()), stream)
                stream.seek(0)
                checkpoint = torch.load(stream, weights_only=True)
                restored = build_model(checkpoint["config"])
                restored.load_state_dict(checkpoint["state_dict"], strict=True)
                torch.testing.assert_close(model(points), restored(points), rtol=0, atol=0)
                normalizer = restored.normalizer if kind == "hash_nf" else restored.intensity.normalizer
                torch.testing.assert_close(normalizer.lower, torch.tensor([10., -6., 20.]))
                torch.testing.assert_close(normalizer.span, torch.tensor([4., 8., 2.]))
                if kind != "hash_nf":
                    self.assertEqual(restored.motion.reference_time.item(), 21.)


if __name__ == "__main__":
    unittest.main()
