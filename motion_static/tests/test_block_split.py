"""Check temporal gaps, train-only interpolation, and masked metric isolation."""
import unittest

import numpy as np
from skimage.metrics import structural_similarity

from motion_static.splits import block_split, interpolate_training_frames
from motion_static.evaluation.masked_metrics import evaluate_masked_sequence


class BlockSplitTests(unittest.TestCase):
    def test_whole_blocks_and_short_gaps(self):
        split = block_split((64, 64, 64))
        train, test, blocks = (split[key] for key in ("train_mask", "test_mask", "test_blocks"))
        self.assertEqual(int(blocks.sum()), 410)
        self.assertEqual(int(test.sum()), 52480)
        np.testing.assert_array_equal(train, ~test)
        self.assertFalse(blocks[..., 0].any())
        self.assertFalse(blocks[..., -1].any())
        self.assertFalse((blocks[..., :-1] & blocks[..., 1:]).any())
        # Every voxel in a block has the same membership.
        grouped = test.reshape(8, 8, 8, 8, 32, 2)
        np.testing.assert_array_equal(grouped.all(axis=(1, 3, 5)), blocks)
        np.testing.assert_array_equal(grouped.any(axis=(1, 3, 5)), blocks)
        np.testing.assert_array_equal(test, block_split((64, 64, 64))["test_mask"])
        self.assertFalse(np.array_equal(test, block_split((64, 64, 64), seed=1)["test_mask"]))
        gaps = [np.diff(np.flatnonzero(observed)).max() for observed in train.reshape(-1, 64)]
        self.assertEqual(max(gaps), 3)

    def test_invalid_or_infeasible_split(self):
        for shape, fraction in (((63, 64, 64), .2), ((64, 64, 64), .5), ((64, 64, 4), .2)):
            with self.subTest(shape=shape, fraction=fraction), self.assertRaises(ValueError):
                block_split(shape, test_fraction=fraction)

    def test_interpolation_does_not_read_test_labels(self):
        times = np.linspace(0, 1, 16) ** 2
        truth = np.broadcast_to(2 + 3 * times, (16, 16, 16)).copy()
        train = block_split(truth.shape)["train_mask"]
        expected = interpolate_training_frames(truth, train, times)
        np.testing.assert_allclose(expected, truth, atol=1e-14)
        truth[~train] = 1000
        np.testing.assert_array_equal(interpolate_training_frames(truth, train, times), expected)


class MaskedMetricTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.random.default_rng(0).uniform(.1, .9, (16, 16, 3))
        self.mask = np.zeros_like(self.truth, dtype=bool)
        self.mask[4:12, 4:12, 1] = True

    def evaluate(self, prediction, **kwargs):
        return evaluate_masked_sequence(self.truth, prediction, self.mask, times=[0, 1, 2], **kwargs)

    def test_errors_outside_mask_are_excluded(self):
        prediction = self.truth.copy()
        prediction[~self.mask] += 2
        result = self.evaluate(prediction)
        self.assertEqual(result["rrmse"], 0)
        self.assertAlmostEqual(result["ssim"], 1, places=12)
        self.assertEqual(result["pixel_count"], 64)
        self.assertEqual(result["ssim_window_count"], 4)
        self.assertIsNone(result["per_frame_rrmse"][0])
        self.assertIsNone(result["per_frame_ssim"][2])

    def test_ssim_matches_independent_crop(self):
        prediction = self.truth.copy()
        prediction[self.mask] *= .8
        result = self.evaluate(prediction, points={"inside": (5, 5)})
        expected = structural_similarity(self.truth[4:12, 4:12, 1], prediction[4:12, 4:12, 1],
                                         data_range=1, win_size=7)
        self.assertAlmostEqual(result["ssim"], expected)
        self.assertAlmostEqual(result["rrmse"], .2)
        self.assertAlmostEqual(result["point_curves"]["inside"]["rrmse"], .2)
        self.assertEqual(result["point_curves"]["inside"]["evaluated"], [False, True, False])

    def test_roi_and_insufficient_windows(self):
        prediction = self.truth.copy()
        prediction[4] += 1
        result = self.evaluate(prediction, roi=(slice(5, 12), slice(4, 12)))
        self.assertGreater(result["rrmse"], 0)
        self.assertEqual(result["roi"]["rrmse"], 0)
        self.assertAlmostEqual(result["roi"]["ssim"], 1)
        self.assertEqual(result["roi"]["ssim_window_count"], 2)
        self.mask[:] = False
        self.mask[5, 5, 1] = True
        result = self.evaluate(prediction)
        self.assertIsNone(result["ssim"])
        self.assertEqual(result["rrmse"], 0)


if __name__ == "__main__":
    unittest.main()
