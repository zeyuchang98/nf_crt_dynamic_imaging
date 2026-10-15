"""Check brightness/motion separation, pixel units, and saved-time loading."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_static.evaluation.branch_metrics import (
    calibrate_geometry, motion_statistics, toy_geometry_template, true_geometry)
from motion_static.evaluation.diagnose_branches import load_midpoints
from motion_static.phantoms.complex import dynamics
from motion_static.phantoms.data import sample


class BranchDiagnosticTests(unittest.TestCase):
    def test_geometry_recovers_motion_without_confusing_brightness(self):
        times = np.array([-.41, -.12, .09, .27])
        data = sample("complex", 32, times)
        template = toy_geometry_template("complex", data["x"], data["y"])
        scores, stationary, moving = calibrate_geometry(data, "complex", template)
        translation, scale, gain, spot_gain = dynamics(times)
        np.testing.assert_allclose(moving["parameters"][:, :2], translation, atol=1e-6)
        np.testing.assert_allclose(moving["parameters"][:, 2:4], scale, atol=1e-6)
        np.testing.assert_allclose(stationary["parameters"][:, 4], gain, atol=1e-6)
        np.testing.assert_allclose(stationary["parameters"][:, 5], gain * spot_gain, atol=1e-6)
        self.assertLess(scores["stationary_max_shift_pixels"], 1e-5)

    def test_simple_template_and_nondefault_frame_count(self):
        data = sample("simple", 32, np.array([-.43, -.09, .17]))
        template = toy_geometry_template("simple", data["x"], data["y"])
        _, stationary, moving = calibrate_geometry(data, "simple", template)
        self.assertEqual(moving["parameters"].shape, (3, 5))
        translation, scale = true_geometry("simple", data["t"])
        np.testing.assert_allclose(moving["parameters"][:, :2], translation, atol=1e-6)
        np.testing.assert_allclose(moving["parameters"][:, 2:4], scale, atol=1e-6)
        np.testing.assert_allclose(stationary["parameters"][:, :2], 0, atol=1e-6)

    def test_motion_uses_component_pixel_spacing_and_foreground(self):
        truth = np.broadcast_to([.3, .8], (3, 4, 2, 2)).copy()
        prediction = truth + [.1, -.4]
        foreground = np.zeros((3, 4, 2), dtype=bool)
        foreground[1, 2, 0] = True
        prediction[~foreground] = 999
        scores = motion_statistics(truth, prediction, foreground, spacing=np.array([.1, .2]))
        np.testing.assert_allclose(scores["midpoint_foreground_component_rmse_pixels"], [1, 2])
        self.assertAlmostEqual(scores["midpoint_foreground_endpoint_rmse_pixels"], np.sqrt(5))
        self.assertAlmostEqual(scores["true_displacement_rms_pixels"], 5)
        self.assertIsNone(scores["per_frame_full_foreground_endpoint_rmse_pixels"][1])

    def test_zero_prediction_has_undefined_direction(self):
        truth = np.ones((2, 2, 3, 2))
        scores = motion_statistics(truth, np.zeros_like(truth), np.ones((2, 2, 3), dtype=bool), spacing=1)
        self.assertIsNone(scores["pooled_vector_cosine"])
        self.assertEqual(scores["predicted_displacement_rms_pixels"], 0)
        self.assertEqual(scores["midpoint_foreground_endpoint_rmse_pixels"], scores["true_displacement_rms_pixels"])

    def test_loader_uses_saved_midpoint_times(self):
        times = np.array([-.37, -.02, .21, .43])
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "arrays").mkdir()
            (run / "config.json").write_text(json.dumps(dict(phantom="complex", model="nested_mlp", size=32)))
            motion = np.ones((32, 32, len(times), 2))
            np.savez(run / "arrays/predictions.npz", midpoint_t=times, midpoint_motion=motion,
                     midpoint_image=np.zeros((32, 32, len(times))), training_t=np.linspace(-.5, .5, 9))
            _, data, prediction = load_midpoints(run)
            np.testing.assert_array_equal(data["t"], times)
            np.testing.assert_array_equal(prediction["motion"], motion)
            self.assertEqual(data["truth"].shape, (32, 32, 4))
            self.assertNotIn("t", prediction)


if __name__ == "__main__":
    unittest.main()
