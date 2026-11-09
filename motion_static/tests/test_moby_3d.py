"""3D (x, y, z, t) support: coordinate rows, 4D hash grids, the 3D nested model, box files and a tiny run."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from motion_static.evaluation.evaluate import predict, predict_entry
from motion_static.evaluation.metrics import error_metrics
from motion_static.evaluation.moby_embedding_figures import load_model
from motion_static.models.hash_grid import HashGridEncoding
from motion_static.models.nested_low_rank_hash_nf import NestedLowRankHashNF
from motion_static.phantoms import moby
from motion_static.phantoms.data import CoordinateGrid, coordinate_grid
from motion_static.run import main, prepare_data
from motion_static.tests.test_moby_data import DATA, DT, FRAMES, SPACING, T0


SIZE, DEPTH = 16, 16
BOX_START = (5, 6, 7)


def write_box(path):
    """Box-file layout: p0 (x, y, z, frame), times on a 0.05 s grid, 4 phases, the box's own fov."""
    p0 = np.random.default_rng(0).random((SIZE, SIZE, DEPTH, FRAMES), dtype=np.float32) * 0.4
    fov = np.array([[0, SIZE * SPACING], [0, SIZE * SPACING], [1, 1 + DEPTH * SPACING]])
    with h5py.File(path, "w") as f:
        f["p0"] = p0
        f["time"] = (T0 + DT * np.arange(FRAMES)).astype(np.float32)
        f["resp_phase"] = np.arange(FRAMES) % 4
        f.attrs.update(fov=fov, spacing=np.array([SPACING] * 3), box_start=np.array(BOX_START))
    return p0


def config_3d(path, **changes):
    template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
    for key in ("size", "frames", "cycles"):
        template.pop(key)
    template.update(phantom=moby.NAME, data=dict(DATA, file=str(path)), rank=2, steps=3, batch_size=64,
                    threads=1, monitor_samples=32, log_interval=1, held_out_split="none",
                    encoding=dict(levels=2, features=2, log2_hash_size=10, min_resolution=4, max_resolution=8),
                    motion_encoding=dict(levels=2, features=2, log2_hash_size=10,
                                         min_resolution=[2, 2, 2, 4], max_resolution=[4, 4, 4, 8]))
    template.update(changes)
    return template


class CoordinateRowTests(unittest.TestCase):
    def test_grid_rows_equal_coordinate_grid_rows_in_2d_and_3d(self):
        x, y, z, t = np.linspace(0, 1, 5), np.linspace(2, 3, 4), np.linspace(-1, 0, 3), np.linspace(-36, -35, 6)
        indices = np.random.default_rng(0).integers(0, 5 * 4 * 3 * 6, 200)
        for axes in ((x, y, t), (x, y, t, z)):
            rows = torch.as_tensor(coordinate_grid(*axes), dtype=torch.float32)
            grid = CoordinateGrid(*axes)
            self.assertEqual(len(grid), len(rows))
            picked = indices[indices < len(rows)]
            self.assertTrue(torch.equal(grid[picked], rows[picked]))
            self.assertTrue(torch.equal(grid[torch.as_tensor(picked)], rows[picked]))

    def test_four_dimensional_dense_grid_gives_each_vertex_its_own_row(self):
        encoding = HashGridEncoding(levels=1, features=2, log2_hash_size=12, min_resolution=[2, 3, 2, 4],
                                    max_resolution=[2, 3, 2, 4], dimensions=4)
        self.assertTrue(bool(encoding.dense[0]))
        vertices = torch.cartesian_prod(*[torch.arange(n + 1) for n in (2, 3, 2, 4)])  # all 3*4*3*5 vertices
        rows = encoding._table_indices(vertices[:, None, None, :]).reshape(-1)
        self.assertEqual(len(torch.unique(rows)), len(vertices))
        self.assertEqual(int(rows.max()) + 1, len(vertices))
        self.assertEqual(encoding(torch.rand(7, 4)).shape, (7, 2))


class NestedModel3DTests(unittest.TestCase):
    def test_motion_has_three_components_and_is_zero_at_the_reference_time(self):
        bounds = [[0, 2.4], [0, 2.4], [1, 3.4], [-36, -35]]
        model = NestedLowRankHashNF(rank=2, width=16, temporal_width=8, coordinate_bounds=bounds,
                                    reference_time=-36, levels=2, log2_hash_size=10, min_resolution=4,
                                    max_resolution=8,
                                    motion_encoding=dict(min_resolution=[2, 2, 2, 4], max_resolution=[4, 4, 4, 8]))
        with torch.no_grad():  # give the motion a nonzero output
            model.motion.field.mlp[-1].weight.normal_()
        points = torch.rand(50, 4) * torch.tensor([2.4, 2.4, 2.4, 1.0]) + torch.tensor([0, 0, 1, -36])
        displacement = model.motion(points)
        self.assertEqual(displacement.shape, (50, 3))
        at_reference = points.clone()
        at_reference[:, -1] = -36
        self.assertTrue(torch.equal(model.motion(at_reference), torch.zeros(50, 3)))
        image = model(points)
        self.assertEqual(image.shape, (50,))
        # The image is the intensity at the displaced spatial point, time unchanged.
        moved = points.clone()
        moved[:, :3] += displacement
        torch.testing.assert_close(image, model.intensity(moved))
        image.sum().backward()
        self.assertGreater(float(model.motion.field.mlp[-1].weight.grad.abs().sum()), 0)


class BoxFileTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "box.h5"
        self.p0 = write_box(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def test_loader_gives_y_x_z_t_truth_and_four_coordinate_bounds(self):
        data, run = moby.resolve_settings(dict(DATA, file=str(self.path)))
        self.assertEqual(run["spatial_dimensions"], 3)
        self.assertEqual(run["depth"], DEPTH)
        self.assertEqual(len(run["coordinate_bounds"]), 4)
        np.testing.assert_allclose(run["coordinate_bounds"][2], [1, 1 + DEPTH * SPACING])
        self.assertEqual(data["box_start"], list(BOX_START))
        entry = moby.make_all_frames(data)["training"]
        frames = moby.window_frames(data)
        expected = np.transpose(self.p0[..., frames], (1, 0, 2, 3)) / np.float32(data["intensity_scale_value"])
        np.testing.assert_array_equal(entry["truth"], expected)
        np.testing.assert_allclose(entry["z"], 1 + (np.arange(DEPTH) + 0.5) * SPACING)
        self.assertEqual(entry["truth"].shape, (SIZE, SIZE, DEPTH, len(frames)))

    def test_tiny_3d_embedding_run_trains_saves_and_scores_the_grid(self):
        config = self.root / "run.json"
        config.write_text(json.dumps(config_3d(self.path)))
        out = self.root / "results" / "run"
        main(["--config", str(config), "--out", str(out)])
        saved = json.loads((out / "config.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(len(saved["coordinate_bounds"]), 4)
        self.assertEqual(saved["reference_time"], saved["coordinate_bounds"][3][0])
        self.assertTrue((out / "loss.csv").exists() and (out / "model.pt").exists())

        # The saved image metrics equal metrics of a direct prediction on coordinate_grid rows.
        training = prepare_data(saved)[0]["training"]
        model = load_model(out)
        direct = predict(model, coordinate_grid(training["x"], training["y"], training["t"], training["z"]))
        direct = direct.reshape(training["truth"].shape)
        np.testing.assert_array_equal(predict_entry(model, training), direct)
        metrics = json.loads((out / "metrics.json").read_text())["image"]["all_window_frames"]
        self.assertAlmostEqual(metrics["rrmse"], error_metrics(training["truth"], direct, data_range=1)["rrmse"], places=6)
        self.assertEqual(len(metrics["per_frame_rrmse"]), len(training["t"]))

    def test_3d_rejects_held_out_splits_and_wrong_resolution_lists(self):
        bad = [dict(held_out_split="midpoints"),
               dict(motion_encoding=dict(levels=2, features=2, log2_hash_size=10,
                                         min_resolution=[2, 2, 4], max_resolution=[4, 4, 8]))]
        for k, changes in enumerate(bad):
            config = self.root / f"bad{k}.json"
            config.write_text(json.dumps(config_3d(self.path, **changes)))
            with self.assertRaises(SystemExit):
                main(["--config", str(config), "--out", str(self.root / f"bad{k}")])


if __name__ == "__main__":
    unittest.main()
