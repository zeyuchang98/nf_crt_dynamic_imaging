"""Check the DCE-MOBY displacement extraction on a small synthetic source with the real layout."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.phantoms.moby import file_sha256
from motion_static.phantoms.moby_motion_slice import extract


NX, NY, NZ_SOURCE, PHASES = 5, 4, 7, 3
SPACING = np.array([0.15] * 3)
P0_FOV = np.array([[0, 0.75], [0, 0.6], [0.3, 0.75]])  # p0's z crop starts at source plane 0.3 / 0.15 = 2


def write_source(root, phase0_motion=0.0):
    """A displacement whose p0-layout value names its own position: ux = x + 10 p, uy = 100 y, uz = 1000 z.

    The source stores it as the real file does: axes (y, x, z, component, phase), components (uy, ux, uz).
    """
    x, y, z = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ_SOURCE), indexing="xy")  # (y, x, z) arrays
    motion = np.zeros((NY, NX, NZ_SOURCE, 3, PHASES), np.float32)
    for p in range(1, PHASES):
        motion[..., 0, p], motion[..., 1, p], motion[..., 2, p] = 100 * y, x + 10 * p, 1000 * z
    motion[0, 0, 2, 0, 0] = phase0_motion  # source plane 2 is p0 plane 0
    flag = (x + 2 * y + 3 * z)[..., None] % 3 - 1 + np.zeros(PHASES, np.int8)  # -1, 0, 1, also (y, x, z)
    (root / "p0").mkdir()
    with h5py.File(root / "p0" / "p0_00000.h5", "w") as f:
        f["p0"] = np.zeros((NX, NY, 3, 2), np.float32)
        f["p0"].attrs.update(fov=P0_FOV, spacing=SPACING)
    with h5py.File(root / "motion_vectors.h5", "w") as f:
        f["motion"] = motion
        f["motion"].attrs.update(dt=0.05, spacing=SPACING, units="mm")
        f["interpolated"] = flag.astype(np.int8)
    return flag


class MobyMotionSliceTests(unittest.TestCase):
    def test_displacement_is_in_p0_layout_at_the_cropped_planes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            flag = write_source(root)
            output = extract(root, [2, 0], root / "motion.h5")
            with h5py.File(output, "r") as f:
                u, interpolated = f["displacement"][...], f["interpolated"][...]
                np.testing.assert_array_equal(f["z_index"][:], [0, 2])
                np.testing.assert_array_equal(f["resp_phase"][:], np.arange(PHASES))
                self.assertEqual(f.attrs["z_offset"], 2)
                self.assertEqual(f.attrs["source_sha256"], file_sha256(root / "motion_vectors.h5"))
                self.assertEqual(f["displacement"].attrs["components"], "ux, uy, uz")
            self.assertEqual(u.shape, (NX, NY, 2, 3, PHASES))
            np.testing.assert_array_equal(u[..., 0], 0)
            for k, plane in enumerate((2, 4)):  # p0 planes 0 and 2 are source planes 2 and 4
                for p in range(1, PHASES):
                    for x in range(NX):
                        for y in range(NY):
                            np.testing.assert_array_equal(u[x, y, k, :, p], [x + 10 * p, 100 * y, 1000 * plane])
                np.testing.assert_array_equal(interpolated[:, :, k, :], np.swapaxes(flag[:, :, plane, :], 0, 1))

    def test_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_source(root, phase0_motion=0.5)
            with self.assertRaises(ValueError):  # the reference phase must have zero displacement
                extract(root, [0], root / "nonzero.h5")
            for z in ([3], [-1]):  # p0 has planes 0-2 only
                with self.assertRaises(ValueError):
                    extract(root, z, root / "outside.h5")
            (root / "exists.h5").touch()
            with self.assertRaises(FileExistsError):
                extract(root, [0], root / "exists.h5")


if __name__ == "__main__":
    unittest.main()
