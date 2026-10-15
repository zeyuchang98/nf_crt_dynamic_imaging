"""Motion GIF controls: freeze true intensity without freezing displacement."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from motion_static.evaluation.motion_visuals import frozen_true_motion
from motion_static.evaluation.render import save_gifs, unchanged_inference_source
from motion_static.phantoms.data import sample


class MotionVisualTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([-.41, -.12, .09, .27])
        self.data = sample("complex", 32, self.times)
        self.config = dict(phantom="complex", size=32, reference_time=.13)

    def test_true_motion_prediction_has_zero_error_but_truth_still_moves(self):
        original = {key: value.copy() for key, value in self.data.items()}
        truth, prediction = frozen_true_motion(self.data, self.data["u_true"], self.config)
        np.testing.assert_array_equal(truth, prediction)
        self.assertGreater(np.max(np.abs(truth[..., 0] - truth[..., 1])), .1)
        self.assertGreater(np.max(np.abs(truth - self.data["truth"])), .1)
        for key, value in original.items():
            np.testing.assert_array_equal(self.data[key], value)

    def test_zero_motion_gives_constant_true_reference_at_custom_t0(self):
        _, prediction = frozen_true_motion(self.data, np.zeros_like(self.data["u_true"]), self.config)
        reference = sample("complex", 32, np.array([self.config["reference_time"]]))["reference_intensity"]
        expected = np.repeat(reference, len(self.times), axis=-1)
        np.testing.assert_allclose(prediction, expected, atol=3e-8, rtol=1e-7)
        np.testing.assert_array_equal(prediction, np.repeat(prediction[..., :1], len(self.times), axis=-1))

    def test_motion_view_uses_frozen_truth_and_displacement_not_old_dynamic_images(self):
        predictions = dict(motion=self.data["u_true"].copy(),
                           learned_motion_true_intensity=np.full_like(self.data["truth"], 999.))
        with tempfile.TemporaryDirectory() as directory, patch('motion_static.evaluation.render.save_comparison') as save:
            save_gifs(Path(directory), self.data, predictions, config=self.config, views=['motion'])
            args, kwargs = save.call_args
        self.assertEqual(args[0].name, 'motion_effect.gif')
        np.testing.assert_array_equal(args[1], args[2])
        self.assertLess(args[2].max(), 2.)
        self.assertEqual(kwargs['labels'], ('True motion', 'Predicted motion', 'Absolute image error'))
        self.assertIn('fixed at t0', args[4])

    def test_render_source_check_allows_only_unused_evaluate_entrypoint_to_change(self):
        original = 'import numpy as np\ndef predict(x): return np.asarray(x)\ndef evaluate(): return 1\n'
        self.assertTrue(unchanged_inference_source(original.replace('return 1', 'return 2'), original))
        self.assertFalse(unchanged_inference_source(original.replace('np.asarray(x)', 'np.asarray(x) + 1'), original))
        self.assertFalse(unchanged_inference_source(original.replace('import numpy as np', 'import torch as np'), original))


if __name__ == '__main__':
    unittest.main()
