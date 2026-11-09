"""Analytical and independent reference checks for image-sequence metrics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from skimage.metrics import structural_similarity

from motion_static.evaluation import metrics
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

    def test_time_chunked_ssim_equals_whole_array_ssim(self):
        rng = np.random.default_rng(4)
        settings = dict(data_range=1, win_size=7, gaussian_weights=False, use_sample_covariance=True,
                        K1=0.01, K2=0.03, channel_axis=None)
        for shape in ((12, 10, 41), (9, 8, 7, 23)):  # 2D+t and 3D+t
            truth = rng.random(shape)
            prediction = truth + 0.2 * rng.standard_normal(shape)
            expected = structural_similarity(truth, prediction, **settings)
            frame = truth.size // shape[-1]
            # Chunks of 1 kept frame, uneven chunks with a short last one, and a single chunk.
            for chunk_voxels in (7 * frame, 13 * frame, 40 * frame, truth.size - 1):
                chunked = metrics._ssim(truth, prediction, 1, 7, chunk_voxels=chunk_voxels)
                self.assertAlmostEqual(chunked, expected, delta=1e-13 * abs(expected), msg=(shape, chunk_voxels))
            # At or below the threshold the whole-array call is used, bit for bit.
            self.assertEqual(metrics._ssim(truth, prediction, 1, 7, chunk_voxels=truth.size), expected)

    def test_large_sequences_record_the_chunked_ssim(self):
        truth = np.random.default_rng(5).random((10, 10, 30))
        prediction = truth + 0.05
        whole = evaluate_sequence(truth, prediction, data_range=1)
        self.assertNotIn("computation", whole["settings"]["ssim"])
        with patch.object(metrics, "SSIM_CHUNK_VOXELS", 900):
            chunked = evaluate_sequence(truth, prediction, data_range=1)
        self.assertIn("time chunks", chunked["settings"]["ssim"]["computation"])
        self.assertAlmostEqual(chunked["ssim"], whole["ssim"], delta=1e-13)
        self.assertEqual(chunked["per_frame_ssim"], whole["per_frame_ssim"])

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


def assert_reports_close(test, streamed, whole, path="report"):
    """Same keys and values; floats to 1e-12 relative. The streamed note in settings is extra."""
    if isinstance(whole, dict):
        keys = set(streamed) - ({"computation"} if path.endswith("settings") else set())
        test.assertEqual(keys, set(whole), path)
        for key in whole:
            assert_reports_close(test, streamed[key], whole[key], f"{path}.{key}")
    elif isinstance(whole, list):
        test.assertEqual(len(streamed), len(whole), path)
        for k, (a, b) in enumerate(zip(streamed, whole)):
            assert_reports_close(test, a, b, f"{path}[{k}]")
    elif isinstance(whole, float):
        test.assertTrue(np.isclose(streamed, whole, rtol=1e-12, atol=1e-15, equal_nan=True),
                        f"{path}: {streamed} != {whole}")
    else:
        test.assertEqual(streamed, whole, path)


class BlockEvaluationTests(unittest.TestCase):
    def check(self, truth, prediction, roi, block_frames):
        calls = []

        def predict_frames(frames):
            calls.append(frames)
            return prediction[..., frames]

        times = np.linspace(0, 1, truth.shape[-1])
        streamed = metrics.evaluate_sequence_in_blocks(truth, predict_frames, data_range=1.0, times=times,
                                                       roi=roi, block_frames=block_frames)
        whole = evaluate_sequence(truth, prediction, data_range=1.0, times=times, roi=roi, points={})
        assert_reports_close(self, streamed, whole)
        self.assertIn("computation", streamed["settings"])
        return calls

    def test_blocks_reproduce_the_whole_array_evaluation_in_3d(self):
        rng = np.random.default_rng(0)
        truth = rng.random((9, 8, 10, 23))
        prediction = truth + 0.05 * rng.normal(size=truth.shape)
        roi = (slice(1, 9), slice(0, 7), slice(2, 10))
        for block_frames in (1, 4, 23, 50):
            with self.subTest(block_frames=block_frames):
                calls = self.check(truth, prediction, roi, block_frames)
                if block_frames < 23:  # each call covers one block and its 3-frame halos only
                    self.assertTrue(all(c.stop - c.start <= block_frames + 6 for c in calls))

    def test_blocks_reproduce_2d_sequences_and_short_axes(self):
        rng = np.random.default_rng(1)
        truth = rng.random((16, 12, 30))
        self.check(truth, truth + 0.1 * rng.normal(size=truth.shape), (slice(2, 14), slice(1, 12)), 5)
        short = rng.random((9, 8, 5, 12))  # z = 5 is shorter than the window: SSIM is null
        streamed_calls = self.check(short, short * 0.9, None, 4)
        self.assertEqual(len(streamed_calls), 3)


if __name__ == "__main__":
    unittest.main()
