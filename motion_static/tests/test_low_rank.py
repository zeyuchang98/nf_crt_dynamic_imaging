"""Rank, spatial interpolation, physical gradients and shared training integration."""
import unittest

import numpy as np
import torch

from motion_static.models.hash_grid import HashGridEncoding
from motion_static.models.low_rank_intensity import LowRankIntensity
from motion_static.evaluation.evaluate import intensity_rank_report, predict_fields
from motion_static.phantoms.data import coordinate_grid, sample
from motion_static.run import build_model, train


def config(rank=3):
    return dict(model="nested_low_rank_hash_nf", phantom="complex", size=16,
                rank=rank, head_width=16, temporal_width=8, reference_time=-.5,
                encoding=dict(levels=2, min_resolution=3, max_resolution=7, log2_hash_size=5),
                hash_lr=.01, mlp_lr=.001, steps=3, batch_size=64)


class LowRankTests(unittest.TestCase):
    def test_spatial_dense_grid_interpolates_affine_function_at_boundaries(self):
        encoder = HashGridEncoding(dimensions=2, levels=1, features=1, min_resolution=2,
                                   max_resolution=2, log2_hash_size=4).double()
        with torch.no_grad():
            for y in range(3):
                for x in range(3):
                    encoder.table[x + 3*y, 0] = (x + 2*y) / 2
        points = torch.tensor([[0., 0.], [1., 1.], [.23, .61], [1., .2]], dtype=torch.float64)
        torch.testing.assert_close(encoder(points)[:, 0], points @ torch.tensor([1., 2.], dtype=torch.float64))

    def test_spatial_dense_to_hashed_boundary_is_continuous(self):
        encoder = HashGridEncoding(dimensions=2, levels=1, features=1, min_resolution=2,
                                   max_resolution=2, log2_hash_size=4, clamp_coordinates=False).double()
        with torch.no_grad():
            encoder.table[:, 0] = torch.arange(len(encoder.table), dtype=torch.float64)
        for boundary in (0., 1.):
            points = torch.tensor([[boundary - 1e-9, .23], [boundary, .23],
                                   [boundary + 1e-9, .23]], dtype=torch.float64)
            values = encoder(points)
            torch.testing.assert_close(values, values[1:2].expand_as(values), atol=1e-7, rtol=0)

    def test_total_rank_bound_including_outside_reference_coordinates(self):
        for rank in (2, 3):
            with self.subTest(rank=rank):
                torch.manual_seed(5)
                field = build_model(config(rank)).intensity.double()
                with torch.no_grad():
                    field.encoding.table.uniform_(-.2, .2)
                xy = torch.rand(23, 2, dtype=torch.float64) * 1.4 - .7
                times = torch.linspace(-.5, .5, 19, dtype=torch.float64)
                coordinates = torch.cat((xy[:, None, :].expand(-1, 19, -1),
                                         times[None, :, None].expand(23, -1, -1)), dim=-1)
                matrix = field(coordinates.reshape(-1, 3)).reshape(23, 19)
                singular = torch.linalg.svdvals(matrix)
                self.assertGreater(singular[0].item(), 1e-4)
                self.assertLess((torch.linalg.vector_norm(singular[rank:]) /
                                 torch.linalg.vector_norm(singular)).item(), 1e-13)

    def test_intensity_coordinate_gradients_inside_and_outside_image(self):
        torch.manual_seed(3)
        field = build_model(config()).intensity.double()
        with torch.no_grad():
            field.encoding.table.uniform_(-.2, .2)
        points = torch.tensor([[.137, -.219, .193], [-.63, .73, -.17]],
                              dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(field, (points,), eps=1e-6, atol=1e-7))
        gradient = torch.autograd.grad(field(points).sum(), points)[0]
        self.assertTrue((gradient.abs() > 1e-8).all())

    def test_nested_motion_gradients_match_finite_differences(self):
        torch.manual_seed(7)
        model = build_model(config()).double()
        with torch.no_grad():
            model.intensity.encoding.table.uniform_(-.2, .2)
        point = torch.tensor([[.113, -.217, .191]], dtype=torch.float64)
        model(point).sum().backward()
        weight = model.motion.field.mlp[-1].weight
        for direction in (0, 1):
            index = int(weight.grad[direction].abs().argmax())
            expected = weight.grad[direction, index].item()
            self.assertGreater(abs(expected), 1e-12)
            eps = 1e-5
            with torch.no_grad():
                weight[direction, index] += eps
                plus = model(point).item()
                weight[direction, index] -= 2 * eps
                minus = model(point).item()
                weight[direction, index] += eps
            self.assertAlmostEqual((plus - minus) / (2 * eps), expected, delta=abs(expected)*.001)

    def test_shared_trainer_updates_all_branches_and_keeps_anchor(self):
        torch.manual_seed(0)
        settings = config(rank=2)
        model = build_model(settings)
        coordinates = torch.rand(64, 3) - .5
        target = coordinates[:, 0] + coordinates[:, 1].square() + coordinates[:, 2]
        tracked = [model.intensity.encoding.table, model.intensity.mlp[-1].weight,
                   model.intensity.temporal[-1].weight, model.motion.field.encoding.table,
                   model.motion.field.mlp[-1].weight]
        before = [p.detach().clone() for p in tracked]
        train(model, coordinates.numpy(), target.numpy(), settings)
        for old, parameter in zip(before, tracked):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(old, parameter))
        reference = coordinates.clone()
        reference[:, 2] = settings["reference_time"]
        torch.testing.assert_close(model.motion(reference), torch.zeros(64, 2), atol=0, rtol=0)

    def test_saved_motion_only_prediction_freezes_learned_intensity_at_custom_t0(self):
        torch.manual_seed(1)
        settings = config(rank=2)
        settings["reference_time"] = .1
        model = build_model(settings)
        with torch.no_grad():
            model.motion.field.mlp[-1].weight.normal_(std=.2)
            model.intensity.encoding.table.uniform_(-.2, .2)
        data = sample("complex", 16, np.array([-.2, .3]))
        prediction = predict_fields(model, data, settings)
        frozen = prediction["frozen_reference_intensity"]
        np.testing.assert_array_equal(frozen[..., 0], frozen[..., 1])
        queries = coordinate_grid(data["x"], data["y"], data["t"])
        queries[:, :2] += prediction["motion"].reshape(-1, 2)
        queries[:, 2] = .1
        with torch.no_grad():
            expected = model.intensity(torch.tensor(queries, dtype=torch.float32)).numpy().reshape(16, 16, 2)
        np.testing.assert_allclose(prediction["motion_only_frozen_intensity"], expected, atol=1e-7)
        self.assertGreater(np.max(np.abs(expected - frozen)), 1e-8)
        report = intensity_rank_report({"test": prediction["reference_intensity"]}, rank=2)
        self.assertLess(report["splits"]["test"]["relative_frobenius_tail_after_rank"], 1e-7)

    def test_invalid_rank_is_rejected(self):
        for rank in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                LowRankIntensity(rank=rank)


if __name__ == "__main__":
    unittest.main()
