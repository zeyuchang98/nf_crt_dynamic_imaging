"""Truth | NF volume-render GIF of a 3D box run (needs the optional PyVista)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from motion_static.run import main
from motion_static.tests.test_moby_3d import config_3d, write_box

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None


@unittest.skipUnless(HAS_PYVISTA, "PyVista is an optional dependency")
class BoxGifTests(unittest.TestCase):
    def test_updated_volumes_render_like_fresh_panels(self):
        from motion_static.evaluation.moby_box_gif import VolumePanels
        rng = np.random.default_rng(0)
        axes = (np.arange(16) * 0.15, np.arange(12) * 0.15, np.arange(10) * 0.15)
        volumes = [rng.random((12, 16, 10)) * 0.05 for _ in range(3)]
        reused = VolumePanels(axes, (12, 16, 10), 0.05, 120)
        first = reused.render(volumes[0], volumes[1])
        second = reused.render(volumes[2], volumes[0])
        reused.close()
        fresh = VolumePanels(axes, (12, 16, 10), 0.05, 120)
        expected = fresh.render(volumes[2], volumes[0])
        fresh.close()
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(second, expected)

    def test_gif_has_one_frame_per_selected_window_frame(self):
        from motion_static.evaluation import moby_box_gif
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_box(root / "box.h5")
            (root / "run.json").write_text(json.dumps(config_3d(root / "box.h5")))
            run = root / "results" / "run"
            main(["--config", str(root / "run.json"), "--out", str(run)])
            moby_box_gif.main([str(run), "--frames", "0:13:3", "--size", "96"])
            with Image.open(run / "visuals" / "window_volume.gif") as gif:
                self.assertEqual(gif.n_frames, 5)  # window frames 0, 3, 6, 9, 12
                self.assertEqual(gif.size, (192, 96 + moby_box_gif.HEADER))
            record = json.loads((run / "visuals" / "window_volume_gif.json").read_text())
            self.assertEqual(record["window_frames"], [0, 3, 6, 9, 12])
            self.assertEqual(record["display_range"], [0, 0.05])
            self.assertTrue((run / "visuals" / "window_volume.png").exists())

            two_d = root / "two_d"
            two_d.mkdir()
            (two_d / "config.json").write_text(json.dumps({"phantom": "dce_moby"}))
            with self.assertRaises(SystemExit):  # slice runs have no volume to render
                moby_box_gif.main([str(two_d)])


if __name__ == "__main__":
    unittest.main()
