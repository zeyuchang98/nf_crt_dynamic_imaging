"""Check the DCE-MOBY training window: layout, coordinates, split, scaling and run wiring."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.phantoms import moby
from motion_static.run import main, parse_configuration


SIZE, FRAMES, SPACING, T0, DT = 16, 40, 0.15, -36.0, 0.05


def write_slice(path):
    """Slice-file layout: p0 (x, y, frame), times on a 0.05 s grid, 4 phases."""
    p0 = np.random.default_rng(0).random((SIZE, SIZE, FRAMES), dtype=np.float32) * 0.4
    with h5py.File(path, "w") as f:
        f["p0"] = p0
        f["time"] = (T0 + DT * np.arange(FRAMES)).astype(np.float32)
        f["resp_phase"] = np.arange(FRAMES) % 4
        f.attrs.update(fov=np.array([[0, SIZE * SPACING], [0, SIZE * SPACING], [1, 2]]),
                       spacing=np.array([SPACING] * 3), z_index=7)
    return p0


DATA = dict(first_frame=10, cycles=3, frames_per_cycle=4)


class MobyDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "slice.h5"
        self.p0 = write_slice(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def test_settings_come_from_file_and_window(self):
        data, run = moby.resolve_settings(dict(DATA, file=str(self.path)))
        frames = np.arange(10, 23)  # 3 cycles of 4 frames, both ends included
        self.assertEqual(run["frames"], 7)  # even frames of the window
        self.assertEqual(run["size"], SIZE)
        np.testing.assert_allclose(run["coordinate_bounds"][:2], [[0, SIZE * SPACING]] * 2)
        np.testing.assert_allclose(run["coordinate_bounds"][2], [T0 + DT * frames[0], T0 + DT * frames[-1]], atol=1e-5)
        np.testing.assert_allclose(run["joint_period"], 4 * DT, atol=1e-5)
        self.assertEqual(data["intensity_scale_value"], float(self.p0.max()))
        self.assertEqual(data["z_index"], 7)
        self.assertEqual(len(data["sha256"]), 64)

    def test_rejects_window_past_end_and_odd_cycle(self):
        with self.assertRaisesRegex(ValueError, "beyond"):
            moby.resolve_settings(dict(DATA, file=str(self.path), first_frame=30))
        with self.assertRaisesRegex(ValueError, "even"):
            moby.resolve_settings(dict(DATA, file=str(self.path), frames_per_cycle=3))

    def test_pair_is_transposed_split_and_scaled(self):
        data, _ = moby.resolve_settings(dict(DATA, file=str(self.path), intensity_scale=0.5))
        pair = moby.make_pair(data)
        training, midpoint = pair["training"], pair["midpoint"]
        np.testing.assert_allclose(training["x"], (np.arange(SIZE) + 0.5) * SPACING)
        self.assertEqual(training["truth"].shape, (SIZE, SIZE, 7))
        self.assertEqual(midpoint["truth"].shape, (SIZE, SIZE, 6))
        iy, ix = 3, 11  # toy layout is (y, x, t); the slice file is (x, y, frame)
        np.testing.assert_allclose(training["truth"][iy, ix, 2], self.p0[ix, iy, 14] / 0.5, rtol=1e-6)
        np.testing.assert_allclose(midpoint["truth"][iy, ix, 0], self.p0[ix, iy, 11] / 0.5, rtol=1e-6)
        self.assertTrue(np.all(training["t"][:-1] < midpoint["t"]) and np.all(midpoint["t"] < training["t"][1:]))

    def write_config(self, **overrides):
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        template.update(phantom=moby.NAME, data=dict(DATA, file=str(self.path)), rank=2, steps=2,
                        batch_size=64, threads=1, monitor_samples=32, log_interval=1, **overrides)
        for key in ("size", "frames", "cycles"):
            template.pop(key)
        path = Path(self.folder.name) / "config.json"
        path.write_text(json.dumps(template))
        return path

    def test_configuration_uses_data_bounds_and_first_frame_reference(self):
        out = Path(self.folder.name) / "run"
        _, config, _ = parse_configuration(["--config", str(self.write_config()), "--out", str(out)])
        self.assertEqual(config["coordinate_bounds"][2][0], config["reference_time"])
        self.assertEqual((config["size"], config["frames"], config["cycles"]), (SIZE, 7, 3))
        with self.assertRaises(SystemExit):
            parse_configuration(["--config", str(self.write_config(output=dict(gifs=True, predictions=False))),
                                 "--out", str(out)])

    def test_tiny_run_writes_image_only_metrics(self):
        out = Path(self.folder.name) / "results" / "run"
        main(["--config", str(self.write_config()), "--out", str(out)])
        report = json.loads((out / "metrics.json").read_text())
        self.assertEqual(sorted(report["image"]), ["training_grid", "unseen_time_midpoints"])
        self.assertNotIn("motion", report)
        self.assertEqual(json.loads((out / "config.json").read_text())["status"], "complete")
        for name in ("loss.csv", "loss.png", "model.pt"):
            self.assertTrue((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
