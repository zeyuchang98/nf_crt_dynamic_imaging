"""Check DCE-MOBY GIF frame selection and display orientation on small arrays."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from motion_static.evaluation.moby_gifs import fixed_phase_frames, respiratory_cycle_frames, save_animation


class MobyGifTests(unittest.TestCase):
    def setUp(self):
        # Four phases interlaced as in the real file, with one extra frame at phase 0.
        self.resp_phase = np.array([0, 1, 2, 3] * 3 + [0])

    def test_respiratory_cycle_is_consecutive_full_cycle(self):
        np.testing.assert_array_equal(respiratory_cycle_frames(self.resp_phase, 1), [4, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, "Cycle 3"):
            respiratory_cycle_frames(self.resp_phase, 3)

    def test_fixed_phase_takes_one_frame_per_cycle(self):
        np.testing.assert_array_equal(fixed_phase_frames(self.resp_phase, 0), [0, 4, 8, 12])
        np.testing.assert_array_equal(fixed_phase_frames(self.resp_phase, 3), [3, 7, 11])
        with self.assertRaisesRegex(ValueError, "phase 4"):
            fixed_phase_frames(self.resp_phase, 4)

    def test_gif_frames_and_x_right_y_up(self):
        images = np.zeros((6, 4, 2), dtype=np.float32)  # (x, y, frame)
        images[5, 3, 0] = 1.0  # largest x and largest y: should appear top right
        images[0, 0, 1] = 1.0  # distinct second frame (identical GIF frames are merged)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.gif"
            info = save_animation(path, images, ["a", "b"], "title", display_max=1.0, scale=1, still_index=0)
            self.assertEqual(info["frames"], 2)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 2)
            still = np.asarray(Image.open(path.with_suffix(".png")).convert("L"))
            image = still[52:52 + 4, :6]  # the image is pasted below the 52-pixel header
            self.assertEqual(np.argwhere(image == 255).tolist(), [[0, 5]])


if __name__ == "__main__":
    unittest.main()
