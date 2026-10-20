"""Check the DCE-MOBY z-slice extraction on small synthetic files with the real layout."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.phantoms.moby_slice import INDICATOR_TISSUES, extract


SHAPE = (5, 4, 3)  # (x, y, z)
FRAME_STEP, PHASE_STEP = 10.0, 0.05
GEOMETRY = dict(fov=np.array([[0, 0.75], [0, 0.6], [1.0, 1.45]]), spacing=np.array([0.15] * 3), spacing_units="mm")


def write_source(root, frames_per_phase, fov=None):
    """Phase p starts PHASE_STEP*p later, as in the real files, so frames interlace."""
    rng = np.random.default_rng(0)
    (root / "p0").mkdir()
    (root / "indicator").mkdir()
    sources = {}
    for phase, frames in enumerate(frames_per_phase):
        p0 = rng.random((*SHAPE, frames), dtype=np.float32)
        time = -36 + PHASE_STEP * phase + FRAME_STEP * np.arange(frames)
        with h5py.File(root / "p0" / f"p0_{phase:05d}.h5", "w") as f:
            f["p0"] = p0
            f["time"] = time
            attributes = dict(GEOMETRY, resp_phase=phase, wavelength=800)
            if fov is not None and phase == 1:
                attributes["fov"] = fov
            f["p0"].attrs.update(attributes)
        tissues = {tissue: rng.random(SHAPE, dtype=np.float32) for tissue in INDICATOR_TISSUES}
        with h5py.File(root / "indicator" / f"indicator_{phase:05d}.h5", "w") as f:
            for tissue, values in tissues.items():
                f[tissue] = values
                f[tissue].attrs.update(dict(GEOMETRY, phase=phase, dt=PHASE_STEP))
        sources[phase] = dict(p0=p0, time=time, **tissues)
    return sources


class MobySliceTests(unittest.TestCase):
    def test_frames_are_interlaced_by_time_and_match_source_plane(self):
        with tempfile.TemporaryDirectory() as folder:
            root, z = Path(folder), 1
            sources = write_source(root, frames_per_phase=[3, 2, 2])
            output = extract(root, z, root / "slice.h5")
            with h5py.File(output, "r") as f:
                time, phase, index = f["time"][:], f["resp_phase"][:], f["frame_in_phase"][:]
                np.testing.assert_array_equal(phase, [0, 1, 2, 0, 1, 2, 0])
                np.testing.assert_array_equal(index, [0, 0, 0, 1, 1, 1, 2])
                self.assertTrue(np.all(np.diff(time) > 0))
                for k in range(len(time)):
                    source = sources[phase[k]]
                    self.assertEqual(time[k], source["time"][index[k]])
                    np.testing.assert_array_equal(f["p0"][..., k], source["p0"][:, :, z, index[k]])
                np.testing.assert_array_equal(f["indicator/resp_phase"][:], [0, 1, 2])
                for tissue in INDICATOR_TISSUES:
                    for k in range(3):
                        np.testing.assert_array_equal(f["indicator"][tissue][..., k], sources[k][tissue][:, :, z])
                self.assertEqual(f.attrs["z_index"], z)
                np.testing.assert_array_equal(f.attrs["fov"], GEOMETRY["fov"])

    def test_phase_subset_and_no_indicator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_source(root, frames_per_phase=[3, 2, 2])
            output = extract(root, 0, root / "slice.h5", phases=[0, 2], with_indicator=False)
            with h5py.File(output, "r") as f:
                np.testing.assert_array_equal(f["resp_phase"][:], [0, 2, 0, 2, 0])
                self.assertNotIn("indicator", f)

    def test_rejects_inconsistent_geometry_and_bad_z(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_source(root, frames_per_phase=[2, 2], fov=GEOMETRY["fov"] + 1)
            with self.assertRaisesRegex(ValueError, "fov"):
                extract(root, 0, root / "slice.h5")
            with self.assertRaisesRegex(ValueError, "z_index"):
                extract(root, SHAPE[2], root / "other.h5", phases=[0])


if __name__ == "__main__":
    unittest.main()
