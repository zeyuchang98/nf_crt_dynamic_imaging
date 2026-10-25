"""SS(r) baseline: exact recovery of low-rank data, linear held-out frames, outputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_static.evaluation import moby_svd_baseline as svd
from motion_static.phantoms import moby
from motion_static.run import main as run_main
from motion_static.tests.test_moby_data import DATA, write_slice


class SemiseparableBaselineTests(unittest.TestCase):
    def test_rank_two_data_with_linear_curves_is_recovered_everywhere(self):
        rng = np.random.default_rng(3)
        maps = rng.random((5, 4, 2))
        times = np.arange(0., 7.)
        # Two separable terms with constant and linear temporal curves.
        volume = lambda t: np.einsum("yxk,tk->yxt", maps, np.stack([1 + 0 * t, 0.5 * t], axis=1))
        training = dict(t=times, truth=volume(times))
        spatial, temporal = svd.semiseparable_factors(training)
        np.testing.assert_allclose(svd.reconstruct(spatial, temporal, times, times, 2, (5, 4)), training["truth"],
                                   atol=1e-12)
        held_out = np.array([0.5, 2.5, 5.5])
        # Linear temporal curves are reproduced exactly by linear interpolation.
        np.testing.assert_allclose(svd.reconstruct(spatial, temporal, times, held_out, 2, (5, 4)), volume(held_out),
                                   atol=1e-12)
        self.assertGreater(np.abs(svd.reconstruct(spatial, temporal, times, times, 1, (5, 4)) - training["truth"]).max(), 1e-3)
        with self.assertRaises(ValueError):
            svd.reconstruct(spatial, temporal, times, np.array([7.5]), 2, (5, 4))
        self.assertEqual(svd.parameter_count(20, 7, 2), 54)

    def test_tiny_window_writes_rank_table_and_plot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_slice(root / "slice.h5")
            template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
            template.update(phantom=moby.NAME, data=dict(DATA, file=str(root / "slice.h5")), rank=2, steps=2,
                            batch_size=64, threads=1, monitor_samples=32, log_interval=1)
            for key in ("size", "frames", "cycles"):
                template.pop(key)
            (root / "config.json").write_text(json.dumps(template))
            run = root / "results" / "run"
            run_main(["--config", str(root / "config.json"), "--out", str(run)])
            out = root / "svd"
            svd.main([str(run), "--out", str(out), "--full-reports", "7"])
            with (out / "svd_ranks.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["rank"]) for row in rows], list(range(1, 8)))  # 7 training frames
            training = [float(row["training_rrmse"]) for row in rows]
            self.assertTrue(all(a >= b - 1e-12 for a, b in zip(training, training[1:])))
            self.assertLess(training[-1], 1e-6)  # full rank reproduces every training frame
            self.assertEqual(int(rows[0]["parameters"]), 16 * 16 + 7)
            report = json.loads((out / "metrics_r7.json").read_text())
            self.assertEqual(sorted(report["image"]), ["test_midpoints", "training_grid", "validation_midpoints"])
            self.assertTrue((out / "rrmse_vs_parameters.png").exists())
            with self.assertRaises(SystemExit):
                svd.main([str(run), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
