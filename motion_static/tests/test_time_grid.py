"""Optional 1D time grid and separate time-MLP rate: interpolation, rank, optimizer scope."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from motion_static.evaluation.render import load_run
from motion_static.models.hash_grid import HashGridEncoding
from motion_static.models.time_grid import TimeGridEncoding, TimeGridField
from motion_static.phantoms.data import coordinate_grid, sample
from motion_static.run import build_model, build_optimizer, execute, parse_configuration, train
from motion_static.tests.test_run_records import small_config

GRID = dict(levels=3, features=2, min_resolution=3, max_resolution=12)


def grid_config(**extra):
    config = small_config()
    config.update(temporal_encoding=dict(GRID), steps=3, **extra)
    return config


class TimeGridTests(unittest.TestCase):
    def test_levels_follow_hash_grid_rule_and_interpolate_linearly(self):
        encoder = TimeGridEncoding(levels=4, features=1, min_resolution=3, max_resolution=40).double()
        reference = HashGridEncoding(dimensions=2, levels=4, features=1, min_resolution=3, max_resolution=40)
        self.assertEqual(encoder.resolutions.tolist(), reference.resolutions.tolist())
        with torch.no_grad():
            encoder.table[:, 0] = torch.arange(len(encoder.table), dtype=torch.float64) ** 2
        times = torch.tensor([[0.], [.37], [.5], [1.]], dtype=torch.float64)
        output = encoder(times)
        for level, (resolution, offset) in enumerate(zip(encoder.resolutions, encoder.offsets)):
            nodes = (offset + torch.arange(resolution + 1, dtype=torch.float64)) ** 2
            grid = torch.linspace(0, 1, int(resolution) + 1, dtype=torch.float64)
            expected = torch.from_numpy(np.interp(times[:, 0].numpy(), grid.numpy(), nodes.numpy()))
            torch.testing.assert_close(output[:, level], expected)

    def test_a_node_only_changes_times_in_its_adjacent_cells(self):
        encoder = TimeGridEncoding(levels=1, features=1, min_resolution=10, max_resolution=10).double()
        times = torch.linspace(0, 1, 201, dtype=torch.float64)[:, None]
        before = encoder(times).detach()
        with torch.no_grad():
            encoder.table[5, 0] += 1.
        changed = (encoder(times).detach() - before).abs()[:, 0] > 0
        self.assertTrue(torch.all((times[changed, 0] > .4) & (times[changed, 0] < .6)))

    def test_grid_time_network_keeps_rank_bound_and_coordinate_gradients(self):
        torch.manual_seed(2)
        field = build_model(grid_config()).intensity.double()
        self.assertIsInstance(field.temporal, TimeGridField)
        with torch.no_grad():
            field.encoding.table.uniform_(-.2, .2)
            field.temporal.encoding.table.uniform_(-.2, .2)
        xy = torch.rand(23, 2, dtype=torch.float64) * 1.4 - .7
        times = torch.linspace(-.5, 2.5, 19, dtype=torch.float64)
        coordinates = torch.cat((xy[:, None, :].expand(-1, 19, -1),
                                 times[None, :, None].expand(23, -1, -1)), dim=-1)
        singular = torch.linalg.svdvals(field(coordinates.reshape(-1, 3)).reshape(23, 19))
        self.assertLess((torch.linalg.vector_norm(singular[2:]) / torch.linalg.vector_norm(singular)).item(), 1e-13)
        points = torch.tensor([[.11, -.21, .3]], dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(field(points).sum(), points)[0]
        self.assertTrue((gradient[:, :2].abs() > 1e-8).all())

    def test_default_groups_unchanged_and_new_groups_cover_learned_parameters(self):
        default = small_config()
        model = build_model(default)
        names = [group["name"] for group in build_optimizer(model, default).param_groups]
        self.assertEqual(names, ["intensity_hash", "intensity_mlp", "motion_hash", "motion_mlp",
                                 "intensity_time_mlp"])
        cases = {
            "grid": (grid_config(), ["intensity_hash", "intensity_mlp", "motion_hash", "motion_mlp",
                                     "intensity_time_grid", "intensity_time_mlp"]),
            "oracle grid": (grid_config(oracle_branch="motion", temporal_mlp_lr=.02),
                            ["intensity_hash", "intensity_mlp", "intensity_time_grid", "intensity_time_mlp"]),
            "mlp rate": (dict(small_config(), temporal_mlp_lr=.02),
                         ["intensity_hash", "intensity_mlp", "motion_hash", "motion_mlp", "intensity_time_mlp"]),
        }
        for label, (config, expected) in cases.items():
            with self.subTest(label):
                model = build_model(config)
                optimizer = build_optimizer(model, config)
                self.assertEqual([group["name"] for group in optimizer.param_groups], expected)
                rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
                self.assertEqual(rates["intensity_time_mlp"], config.get("temporal_mlp_lr", config["mlp_lr"]))
                self.assertEqual(rates["intensity_mlp"], config["mlp_lr"])
                if "intensity_time_grid" in rates:
                    self.assertEqual(rates["intensity_time_grid"], config["hash_lr"])
                optimized = [p for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimized), len({id(p) for p in optimized}))
                self.assertEqual({id(p) for p in optimized}, {id(p) for p in model.parameters()})

    def test_training_updates_time_grid_with_true_motion(self):
        torch.manual_seed(0)
        config = grid_config(oracle_branch="motion", batch_size=256)
        model = build_model(config)
        data = sample("complex", 16, np.linspace(-.5, 2.5, 9), (-.5, 2.5))
        coordinates = coordinate_grid(data["x"], data["y"], data["t"])
        tracked = [model.intensity.temporal.encoding.table, model.intensity.temporal.mlp[-1].weight,
                   model.intensity.encoding.table]
        before = [p.detach().clone() for p in tracked]
        with redirect_stdout(io.StringIO()):
            train(model, coordinates, data["truth"].reshape(-1), config)
        for old, parameter in zip(before, tracked):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertFalse(torch.equal(old, parameter))

    def test_configuration_rejects_invalid_time_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            template = json.loads((Path(__file__).resolve().parents[1] / "experiments/template.json").read_text())
            cases = [(dict(temporal_encoding=GRID), ["--model", "nested_hash_nf"]),
                     (dict(temporal_mlp_lr=.01), ["--model", "nested_hash_nf"]),
                     (dict(temporal_encoding=GRID, oracle_branch="intensity"), []),
                     (dict(temporal_encoding=dict(GRID, min_resolution=20)), []),
                     (dict(temporal_encoding=dict(GRID, extra=1)), []),
                     (dict(temporal_encoding=dict(GRID, levels=2.)), []),
                     (dict(temporal_mlp_lr=-.01), []),
                     (dict(temporal_mlp_lr=float("nan")), [])]
            for extra, arguments in cases:
                with self.subTest(extra=extra, arguments=arguments):
                    path.write_text(json.dumps(template | extra))
                    with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                            patch("sys.stderr", io.StringIO()):
                        parse_configuration(["--config", str(path), "--dry-run", *arguments])
            path.write_text(json.dumps(template | dict(temporal_encoding=GRID, temporal_mlp_lr=.01)))
            for model in ("nested_low_rank_hash_nf", "low_rank_hash_nf"):
                _, resolved, _ = parse_configuration(["--config", str(path), "--model", model, "--dry-run"])
                self.assertEqual(resolved["temporal_encoding"], GRID)

    def test_complete_run_reloads_and_records_time_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with redirect_stdout(io.StringIO()):
                execute(grid_config(oracle_branch="motion", temporal_mlp_lr=.02), out)
            model, _ = load_run(out)
            self.assertIsInstance(model.intensity.temporal, TimeGridField)
            with (out.parent / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["temporal_encoding"]), GRID)
            self.assertEqual(float(row["temporal_mlp_lr"]), .02)
            report = json.loads((out / "metrics.json").read_text())
            self.assertIn("intensity_rank", report)


if __name__ == "__main__":
    unittest.main()
