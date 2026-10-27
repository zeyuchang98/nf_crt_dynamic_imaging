"""DCE-MOBY branch GIFs: window frame order, orientation and outputs of a tiny run."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from motion_static.evaluation import moby_branch_gifs
from motion_static.phantoms import moby
from motion_static.run import main
from motion_static.tests.test_moby_data import DATA, write_slice


class MobyBranchGifTests(unittest.TestCase):
    def test_window_frames_merge_splits_in_time_order(self):
        entry = lambda t: dict(t=np.array(t, float), truth=np.ones((2, 2, len(t))) * np.array(t, float))
        truth, times, labels = moby_branch_gifs.window_frames(
            {"training": entry([0, 2, 4]), "validation": entry([1]), "test": entry([3])})
        np.testing.assert_array_equal(times, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(truth[0, 0], [0, 1, 2, 3, 4])
        self.assertEqual(labels.tolist(), ["training", "validation", "training", "test", "training"])

    def test_frame_ranges_start_at_the_reference_frame(self):
        select = moby_branch_gifs.selected_frames
        np.testing.assert_array_equal(select([], 4), [0, 1, 2, 3])
        np.testing.assert_array_equal(select(["0:10:5", "12:14"], 14), [0, 5, 12, 13])
        np.testing.assert_array_equal(select(["0:14401:50"], 14401), np.arange(0, 14401, 50))
        for ranges in (["1:5"], ["0:5", "3:8"], ["0:15"], ["0:5:0"], ["0"], ["5:2"]):
            with self.assertRaises(ValueError, msg=ranges):
                select(ranges, 14)

    def test_tiles_show_y_upward(self):
        panel = np.zeros((3, 2))
        panel[0, 1] = 1  # lowest y, highest x
        tile = moby_branch_gifs.to_rgb(panel, None, 0, 1)
        self.assertEqual(tile[-1, 1].tolist(), [255, 255, 255])
        self.assertEqual(tile[0, 1].tolist(), [0, 0, 0])

    def test_tiny_run_writes_three_views_over_all_window_frames(self):
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
            main(["--config", str(root / "config.json"), "--out", str(run)])
            moby_branch_gifs.main([str(run), "--scale", "1", "--threads", "1"])
            summary = json.loads((run / "visuals" / "branch_gifs.json").read_text())
            self.assertEqual(summary["frames"], 13)
            self.assertEqual(sorted(summary["gifs"]), ["image", "intensity", "motion"])
            for name in summary["gifs"]:
                with Image.open(run / "visuals" / f"window_{name}.gif") as gif:
                    self.assertEqual(gif.n_frames, 13)
            # The motion anchor makes the first frame's change exactly zero for every view.
            for values in summary["relative_change_from_first_frame"].values():
                self.assertEqual(values[0], 0.)

            # A frame subset shows the same frames and changes, with window frame captions.
            moby_branch_gifs.main([str(run), "--scale", "1", "--threads", "1",
                                   "--frames", "0:5:4", "8:13:4", "--name", "subset"])
            subset = json.loads((run / "visuals" / "subset_branch_gifs.json").read_text())
            self.assertEqual(subset["window_frames"], [0, 4, 8, 12])
            self.assertEqual(subset["frame_ranges"], ["0:5:4", "8:13:4"])
            np.testing.assert_allclose(subset["times"], np.array(summary["times"])[[0, 4, 8, 12]])
            for view, values in subset["relative_change_from_first_frame"].items():
                full = np.array(summary["relative_change_from_first_frame"][view])[[0, 4, 8, 12]]
                np.testing.assert_allclose(values, full, rtol=1e-5, atol=1e-7)
            for name in subset["gifs"]:
                with Image.open(run / "visuals" / f"subset_{name}.gif") as gif:
                    self.assertEqual(gif.n_frames, 4)
            with self.assertRaises(SystemExit):
                moby_branch_gifs.main([str(run), "--frames", "2:6", "--name", "late"])


if __name__ == "__main__":
    unittest.main()
