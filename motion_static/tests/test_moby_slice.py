"""Check the DCE-MOBY z-slice extraction on small synthetic files with the real layout."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.phantoms.moby_slice import INDICATOR_TISSUES, extract, extract_box, lesion_margins


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

    def test_box_holds_the_source_box_of_the_selected_sorted_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = write_source(root, frames_per_phase=[3, 2, 2])
            start, size = (1, 1, 1), (3, 2, 2)
            box = tuple(slice(a, a + n) for a, n in zip(start, size))
            output = extract_box(root, start, size, root / "box.h5", frames=(2, 6))
            with h5py.File(output, "r") as f:
                phase, index = f["resp_phase"][:], f["frame_in_phase"][:]
                # Sorted series: phases 0, 1, 2, 0, 1, 2, 0; frames 2-5 are its third to sixth.
                np.testing.assert_array_equal(phase, [2, 0, 1, 2])
                np.testing.assert_array_equal(index, [0, 1, 1, 1])
                self.assertEqual(f["p0"].shape, (*size, 4))
                for k in range(4):
                    source = sources[phase[k]]
                    self.assertEqual(f["time"][k], source["time"][index[k]])
                    np.testing.assert_array_equal(f["p0"][..., k], source["p0"][box + (index[k],)])
                for tissue in INDICATOR_TISSUES:
                    for k in range(3):
                        np.testing.assert_array_equal(f["indicator"][tissue][..., k], sources[k][tissue][box])
                # The fov is the box's own extent: voxel centers follow from it and the spacing.
                expected = GEOMETRY["fov"][:, :1] + np.array([start, np.add(start, size)]).T * 0.15
                np.testing.assert_allclose(f.attrs["fov"], expected)
                np.testing.assert_array_equal(f.attrs["box_start"], start)
                np.testing.assert_array_equal(f.attrs["frame_range"], [2, 6])
            with self.assertRaisesRegex(ValueError, "do not fit"):
                extract_box(root, (3, 0, 0), size, root / "outside.h5")
            with self.assertRaisesRegex(ValueError, "Frame range"):
                extract_box(root, start, size, root / "late.h5", frames=(5, 9))

    def test_lesion_margins_count_free_voxels_to_each_face(self):
        lesion = np.zeros((6, 5, 4, 2))
        lesion[2, 1, 1, 0] = 0.5
        lesion[3, 3, 1, 1] = 1.0  # another phase: the union spans x 2-3, y 1-3, z 1
        low, high = lesion_margins(lesion)
        np.testing.assert_array_equal(low, [2, 1, 1])
        np.testing.assert_array_equal(high, [2, 1, 2])


if __name__ == "__main__":
    unittest.main()
