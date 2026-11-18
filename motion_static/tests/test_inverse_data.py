"""Check the 2D CRT operator wrapper (caSRT) and the simulated measurements (2026-11-18 setup)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

from motion_static.inverse import crt_operator as crt
from motion_static.inverse import measurements
from motion_static.phantoms import moby


HAVE_CASRT = (crt.DEFAULT_CASRT / "caSRT" / "CRT.py").is_file()
SIZE, SPACING, DR = 24, 0.15, 0.15
FOV = SIZE * SPACING


def ring_radii(meta):
    """caSRT radii: arange(R - sqrt(2)/2 L + dr, R + sqrt(2)/2 L + dr/2, dr)."""
    first = meta["ring_radius_mm"] - np.sqrt(2) / 2 * meta["fov_mm"] + meta["ring_spacing_mm"]
    return first + meta["ring_spacing_mm"] * np.arange(meta["num_rings"])


def pixel_centers(size, pixel_mm):
    """caSRT pixel centers relative to the ring center."""
    return (np.arange(size) - (size // 2 - 1)) * pixel_mm


@unittest.skipUnless(HAVE_CASRT, "caSRT clone not found next to the repository")
class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radius = crt.circumscribed_radius(FOV)
        cls.matrix, cls.meta = crt.build_operator(SIZE, FOV, cls.radius, DR, 1.)

    def test_import_leaves_no_cupy_stub(self):
        crt.load_casrt_crt()
        for name in ("cupy", "cupyx"):
            if importlib.util.find_spec(name) is None:
                self.assertNotIn(name, sys.modules)

    def test_shape_and_rings(self):
        rings = len(np.arange(DR, np.sqrt(2) * FOV + 0.5 * DR, DR))  # R circumscribed: smallest radius 0
        self.assertEqual(self.meta["num_rings"], rings)
        self.assertEqual(self.matrix.shape, (360 * rings, SIZE ** 2))
        self.assertAlmostEqual(self.radius, np.sqrt(2) / 2 * FOV)

    def test_row_sums_are_arc_lengths_inside_the_image(self):
        # Rows weight each pixel by the arc length of the circle inside it, so A 1 is the
        # arc length inside the image square, up to quadrature points rounded at its edge.
        centers = pixel_centers(SIZE, SPACING)
        low, high = centers[0] - SPACING / 2, centers[-1] + SPACING / 2
        phi = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
        radii = ring_radii(self.meta)
        sums = np.asarray(self.matrix.sum(axis=1)).reshape(360, -1)
        for view in range(0, 360, 7):
            theta = np.deg2rad(view)
            x = self.radius * np.cos(theta) + radii[:, None] * np.cos(phi)
            y = self.radius * np.sin(theta) + radii[:, None] * np.sin(phi)
            inside = (x >= low) & (x < high) & (y >= low) & (y < high)
            arc = 2 * np.pi * radii * inside.mean(axis=1)
            np.testing.assert_allclose(sums[view], arc, atol=2 * SPACING)

    def test_single_pixel_lies_on_its_circles(self):
        # An asymmetric pixel (iy, ix) responds only on circles through its own square, so
        # a transposed column order would fail. Image layout is (y, x), reshaped x-fastest.
        iy, ix = 5, 17
        image = np.zeros((SIZE, SIZE))
        image[iy, ix] = 1.
        data = (self.matrix @ image.reshape(-1)).reshape(360, -1)
        centers = pixel_centers(SIZE, SPACING)
        radii = ring_radii(self.meta)
        seen = 0
        for view in range(360):
            theta = np.deg2rad(view)
            distance = np.hypot(self.radius * np.cos(theta) - centers[ix], self.radius * np.sin(theta) - centers[iy])
            hit = data[view] > 0
            seen += hit.any()
            self.assertTrue(np.all(np.abs(radii[hit] - distance) <= np.sqrt(2) / 2 * SPACING + 1e-9))
        self.assertGreaterEqual(seen, 0.95 * 360)

    def test_ring_center_convention(self):
        self.assertAlmostEqual(crt.ring_center(0., 248, 0.15), 18.525)
        self.assertAlmostEqual(crt.ring_center(0., 5, 1.), 1.5)

    def test_frame_views_follow_proxnf(self):
        np.testing.assert_array_equal(crt.frame_view_degrees(0)[0], [0, 1, 2, 3, 4, 180, 181, 182, 183, 184])
        np.testing.assert_array_equal(crt.frame_view_degrees(1250)[0], np.r_[130:135, 310:315])
        # applications/proximal_splitting: index (i * vpt + k + j * nV * vpt / cover) % (vpt * nV)
        vpt, n_v, cover = 5, 72, 2
        for i in range(n_v + 3):
            proxnf = [int((i % n_v * vpt + k + j * n_v * vpt / cover) % (vpt * n_v))
                      for j in range(cover) for k in range(vpt)]
            np.testing.assert_array_equal(crt.frame_view_degrees(i)[0], proxnf)

    def test_frame_rows(self):
        rows = crt.frame_rows([3., 183.], num_rings=4)
        np.testing.assert_array_equal(rows, [12, 13, 14, 15, 732, 733, 734, 735])
        with self.assertRaisesRegex(ValueError, "angle grid"):
            crt.frame_rows([0.5], num_rings=4)

    def test_save_load_round_trip_and_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "op.npz"
            crt.save_operator(path, self.matrix, self.meta)
            loaded, meta = crt.load_operator(path, expected=dict(size=SIZE, ring_spacing_mm=DR))
            self.assertEqual((loaded != self.matrix).nnz, 0)
            with self.assertRaisesRegex(ValueError, "ring_spacing_mm"):
                crt.load_operator(path, expected=dict(ring_spacing_mm=0.2))
            changed = self.matrix.copy()
            changed.data[0] += 1.
            np.savez(path, data=changed.data, indices=changed.indices, indptr=changed.indptr,
                     shape=np.asarray(changed.shape))
            with self.assertRaisesRegex(ValueError, "hash"):
                crt.load_operator(path)


def write_slice(path, frames=12):
    p0 = np.random.default_rng(1).random((SIZE, SIZE, frames), dtype=np.float32) * 0.4
    with h5py.File(path, "w") as f:
        f["p0"] = p0
        f["time"] = (-36. + 0.05 * np.arange(frames)).astype(np.float32)
        f.attrs.update(fov=np.array([[0, FOV], [0, FOV], [1, 2]]), spacing=np.array([SPACING] * 3), z_index=7)
    return p0


@unittest.skipUnless(HAVE_CASRT, "caSRT clone not found next to the repository")
class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.p0 = write_slice(root / "slice.h5")
        self.data = dict(file=str(root / "slice.h5"), first_frame=2, cycles=2, frames_per_cycle=4)
        self.matrix, meta = crt.build_operator(SIZE, FOV, crt.circumscribed_radius(FOV), DR, 1.)
        self.operator = root / "op.npz"
        self.meta = crt.save_operator(self.operator, self.matrix, meta)
        self.arrays, self.info = measurements.generate(self.data, self.operator, rnl=0.04, seed=3)

    def tearDown(self):
        self.folder.cleanup()

    def test_clean_data_use_each_frames_views_on_the_yx_image(self):
        frames = np.arange(2, 11)
        np.testing.assert_array_equal(self.arrays["frames"], frames)
        scale = float(self.p0.max())
        for k, frame in enumerate(frames):
            truth = self.p0[..., frame].T.astype(np.float64) / scale  # file (x, y) -> (y, x)
            rows = crt.frame_rows(crt.frame_view_degrees(frame)[0], self.meta["num_rings"])
            expected = (self.matrix[rows] @ truth.reshape(-1)).reshape(10, -1)
            np.testing.assert_allclose(self.arrays["clean"][k], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(self.info["data"]["intensity_scale_value"], scale)

    def test_noise_level_and_reproducibility(self):
        clean, noisy = self.arrays["clean"].astype(np.float64), self.arrays["noisy"].astype(np.float64)
        sigma = self.info["noise"]["sigma"]
        self.assertAlmostEqual(sigma, 0.04 * clean.max(), places=5)
        noise = noisy - clean
        self.assertAlmostEqual(noise.std() / sigma, 1., delta=0.05)
        self.assertLess(abs(noise.mean()), 4 * sigma / np.sqrt(noise.size))
        again, _ = measurements.generate(self.data, self.operator, rnl=0.04, seed=3)
        np.testing.assert_array_equal(again["noisy"], self.arrays["noisy"])
        other, _ = measurements.generate(self.data, self.operator, rnl=0.04, seed=4)
        self.assertFalse(np.array_equal(other["noisy"], self.arrays["noisy"]))

    def test_save_load_checks_hash(self):
        path = Path(self.folder.name) / "meas.npz"
        measurements.save_measurements(path, self.arrays, self.info)
        arrays, info = measurements.load_measurements(path)
        np.testing.assert_array_equal(arrays["noisy"], self.arrays["noisy"])
        self.assertEqual(info["operator"]["operator_sha256"], self.meta["operator_sha256"])
        json.loads(path.with_suffix(".json").read_text())
        np.savez(path, **dict(self.arrays, noisy=self.arrays["noisy"] + 1))
        with self.assertRaisesRegex(ValueError, "hash"):
            measurements.load_measurements(path)

    def test_rejects_operator_for_another_grid(self):
        matrix, meta = crt.build_operator(SIZE + 2, (SIZE + 2) * SPACING, 3., DR, 1.)
        other = Path(self.folder.name) / "other.npz"
        crt.save_operator(other, matrix, meta)
        with self.assertRaisesRegex(ValueError, "does not match"):
            measurements.generate(self.data, other, rnl=0.04, seed=3)


if __name__ == "__main__":
    unittest.main()
