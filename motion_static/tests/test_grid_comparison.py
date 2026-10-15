"""Reject comparisons that mix an intensity grid change with other conditions."""
import copy
import json
from pathlib import Path
import unittest

from motion_static.evaluation.compare_training import validate_configs


class GridComparisonTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "experiments/template.json"
        self.baseline = json.loads(path.read_text())
        self.baseline.update(rank=2, steps=32000, motion_encoding=dict(min_resolution=[2, 2, 8], max_resolution=[32, 32, 128]),
                             coordinate_bounds=[[-.5, .5], [-.5, .5], [-.5, 2.5]],
                             reference_time=-.5, loss="MSE(predicted image, observed image)",
                             scheduler="cosine to zero")
        self.changed = copy.deepcopy(self.baseline)
        self.changed["encoding"]["max_resolution"] = 64

    def test_intensity_maximum_requires_explicit_comparison_mode(self):
        with self.assertRaises(ValueError):
            validate_configs([self.baseline, self.changed])
        validate_configs([self.baseline, self.changed], vary_intensity_grid=True)

    def test_unrelated_conditions_cannot_change_with_intensity_maximum(self):
        for key, value in (("rank", 3), ("frames", 127), ("steps", 8000), ("mlp_lr", .01)):
            with self.subTest(key=key):
                other = copy.deepcopy(self.changed)
                other[key] = value
                with self.assertRaises(ValueError):
                    validate_configs([self.baseline, other], vary_intensity_grid=True)

    def test_intensity_minimum_and_motion_grid_must_stay_fixed(self):
        for branch, key, value in (("encoding", "min_resolution", 4),
                                   ("motion_encoding", "max_resolution", [16, 16, 128])):
            with self.subTest(branch=branch):
                other = copy.deepcopy(self.changed)
                other[branch][key] = value
                with self.assertRaises(ValueError):
                    validate_configs([self.baseline, other], vary_intensity_grid=True)

    def test_both_grid_comparison_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_configs([self.baseline, self.changed], vary_motion_grid=True,
                             vary_intensity_grid=True)


if __name__ == "__main__":
    unittest.main()
