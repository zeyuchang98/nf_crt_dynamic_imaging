"""Analytical and independent reference checks for image-sequence metrics."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from motion_static.evaluation.metrics import error_metrics, evaluate_sequence, save_metrics


class MetricTests(unittest.TestCase):
    def test_analytic_constant_error(self):
        truth = np.ones((9, 9, 9)) * 0.5
        result = evaluate_sequence(truth, truth + 0.1, data_range=1)
        self.assertAlmostEqual(result["mse"], 0.01)
        self.assertAlmostEqual(result["rmse"], 0.1)
        self.assertAlmostEqual(result["rrmse"], 0.2)
        self.assertAlmostEqual(result["psnr_db"], 20)
        self.assertAlmostEqual(result["ssim"], (2 * 0.5 * 0.6 + 0.0001) / (0.5**2 + 0.6**2 + 0.0001))

    def test_exact_and_zero_reference_json(self):
        truth = np.ones((7, 7, 7))
        result = evaluate_sequence(truth, truth, data_range=1)
        self.assertEqual(result["ssim"], 1)
        self.assertEqual(result["rrmse"], 0)
        self.assertTrue(np.isposinf(result["psnr_db"]))
        result["zero"] = error_metrics(np.zeros(4), np.ones(4), data_range=1)
        self.assertTrue(np.isnan(result["zero"]["rrmse"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            save_metrics(path, result)
            decoded = json.loads(path.read_text(), parse_constant=lambda s: self.fail(s))
            self.assertEqual(decoded["psnr_db"], "Infinity")
            self.assertEqual(decoded["zero"]["rrmse"], "undefined")

    def test_ssim_reference_and_negative_structure(self):
        truth = np.indices((9, 9, 9)).sum(axis=0) % 2
        prediction = 1.0 - truth
        result = evaluate_sequence(truth, prediction, data_range=1)
        expected = structural_similarity(truth.astype(float), prediction, data_range=1, win_size=7)
        self.assertAlmostEqual(result["ssim"], expected)
        self.assertLess(result["ssim"], 0)

    def test_roi_excludes_outside_errors_and_preserves_neighborhoods(self):
        truth = np.ones((15, 15, 9))
        prediction = truth.copy()
        prediction[:3] = 4
        result = evaluate_sequence(truth, prediction, data_range=1,
                                   roi=(slice(4, 12), slice(4, 12)))
        self.assertGreater(result["mse"], 0)
        self.assertEqual(result["roi"]["mse"], 0)
        self.assertEqual(result["roi"]["ssim"], 1)
        self.assertEqual(result["activity_curve"]["rrmse"], 0)

    def test_activity_and_points_with_nonlast_time_axis(self):
        truth = np.broadcast_to(np.arange(1, 10)[:, None, None], (9, 8, 8)).copy()
        mask = np.zeros_like(truth, dtype=bool)
        for k in range(9):
            mask[k, k % 8, 2] = True
        result = evaluate_sequence(truth, truth * 0.8, data_range=10, time_axis=0,
                                   activity_mask=mask, points={"point": (3, 2)})
        np.testing.assert_allclose(result["activity_curve"]["truth"], np.arange(1, 10))
        self.assertAlmostEqual(result["activity_curve"]["rrmse"], 0.2)
        self.assertAlmostEqual(result["point_curves"]["point"]["rrmse"], 0.2)
        self.assertAlmostEqual(result["rrmse"], 0.2)

    def test_volume_psnr_is_not_mean_frame_psnr(self):
        truth = np.ones((7, 7, 2))
        prediction = truth + np.array([0.01, 0.1])
        result = evaluate_sequence(truth, prediction, data_range=1)
        self.assertAlmostEqual(result["psnr_db"], -10 * np.log10((0.0001 + 0.01) / 2))
        self.assertNotAlmostEqual(result["psnr_db"], np.mean(result["per_frame_psnr_db"]))
        self.assertIsNone(result["ssim"])
        self.assertTrue(all(v is not None for v in result["per_frame_ssim"]))

    def test_four_dimensional_sequence_and_scale(self):
        truth = np.random.default_rng(0).uniform(0.2, 0.8, (7, 8, 7, 9))
        result = evaluate_sequence(truth, truth + 0.1, data_range=1)
        scaled = evaluate_sequence(truth * 10, (truth + 0.1) * 10, data_range=10)
        for key in ("rrmse", "ssim", "psnr_db"):
            self.assertAlmostEqual(result[key], scaled[key])
        self.assertEqual(len(result["per_frame_ssim"]), 9)

    def test_invalid_inputs(self):
        truth = np.ones((9, 9, 9))
        for kwargs in ({"data_range": 0}, {"data_range": 1, "ssim_win_size": 4},
                       {"data_range": 1, "activity_mask": np.zeros((9, 9), dtype=bool)},
                       {"data_range": 1, "roi": (slice(0, 20), slice(0, 9))},
                       {"data_range": 1, "points": {"bad": (-1, 2)}},
                       {"data_range": 1, "times": np.zeros(9)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_sequence(truth, truth, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_sequence(truth, truth[..., 0], data_range=1)
        truth[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_sequence(truth, truth, data_range=1)


if __name__ == "__main__":
    unittest.main()
