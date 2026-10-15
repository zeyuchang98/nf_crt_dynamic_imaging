"""Monitoring must not change optimization, sampling, or the reference constraint."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from motion_static.run import build_model, train


class TrainingMonitorTests(unittest.TestCase):
    def test_monitoring_does_not_change_training_and_checkpoints_restore(self):
        config = dict(phantom="complex", model="nested_low_rank_hash_nf", rank=2,
                      head_width=8, temporal_width=8, steps=6, batch_size=8,
                      mlp_lr=.001, hash_lr=.01, log_interval=2, monitor_samples=16,
                      checkpoint_steps=[2, 6], coordinate_bounds=[[-.5, .5], [-.5, .5], [-.5, 2.5]],
                      reference_time=-.5,
                      encoding=dict(levels=2, features=2, min_resolution=4, max_resolution=8, log2_hash_size=8))
        rng = np.random.default_rng(7)
        x = rng.uniform([-.5, -.5, -.5], [.5, .5, 2.5], (32, 3)).astype(np.float32)
        y = x[:, 0] + .1 * x[:, 2]
        torch.manual_seed(9)
        baseline = build_model(config)
        monitored = build_model(config)
        monitored.load_state_dict(baseline.state_dict())
        torch.manual_seed(41)
        train(baseline, x, y, config)
        expected_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            torch.manual_seed(41)
            # Deliberately different validation targets would corrupt training
            # immediately if they leaked into gradients or batch selection.
            train(monitored, x, y, config, out=out, validation=(x, y + 100))
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(monitored.state_dict()[key], value, atol=0, rtol=0)
            with (out / "loss.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2, 4, 6])
            self.assertGreater(float(rows[-1]["fixed_validation_mse"]), 9000)
            with np.load(out / "monitor_samples.npz") as saved:
                indices = saved["training_indices"]
            with torch.no_grad():
                expected = (monitored(torch.tensor(x[indices])) - torch.tensor(y[indices])).double().square().mean().item()
            self.assertAlmostEqual(float(rows[-1]["fixed_training_mse"]), expected, places=12)
            self.assertEqual(float(rows[-1]["next_lr_intensity_hash"]), 0.)
            self.assertGreater((out / "loss.png").stat().st_size, 1000)
            self.assertFalse((out / "plots").exists())
            checkpoint = torch.load(out / "checkpoints" / "step_000006.pt", weights_only=True)
            restored = build_model(checkpoint["config"])
            restored.load_state_dict(checkpoint["state_dict"])
            torch.testing.assert_close(restored(torch.tensor(x)), monitored(torch.tensor(x)), atol=0, rtol=0)
            points = torch.tensor(x, requires_grad=True)
            gradient = torch.autograd.grad(restored(points).sum(), points)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient[:, :2].abs().sum().item(), 0)
            reference = torch.tensor(x)
            reference[:, 2] = -.5
            torch.testing.assert_close(restored.motion(reference), torch.zeros(32, 2), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
