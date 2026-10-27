"""Check coordinate order, exact temporal sampling, and physical warp conventions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motion_static.phantoms.data import coordinate_grid, get_phantom, make_pair, sample


class PhantomTests(unittest.TestCase):
    def test_flattening_matches_y_x_time(self):
        data = make_pair("simple", size=16, frames=9)["training"]
        coordinates = coordinate_grid(data["x"], data["y"], data["t"])
        iy, ix, it = 4, 9, 2
        flat = np.ravel_multi_index((iy, ix, it), data["truth"].shape)
        np.testing.assert_equal(coordinates[flat], [data["x"][ix], data["y"][iy], data["t"][it]])

    def test_temporal_split_and_reference(self):
        for name in ("simple", "complex"):
            with self.subTest(phantom=name):
                pair = make_pair(name, size=16, frames=9, cycles=1)
                fine = sample(name, 16, np.linspace(-.5, .5, 17))
                np.testing.assert_array_equal(fine["truth"][..., ::2], pair["training"]["truth"])
                np.testing.assert_array_equal(fine["truth"][..., 1::2], pair["midpoint"]["truth"])
                np.testing.assert_allclose(pair["training"]["u_true"][..., 0, :], 0, atol=1e-15)
                np.testing.assert_array_equal(pair["training"]["truth"][..., 0], pair["training"]["reference_intensity"][..., 0])

    def test_backward_warp_recovers_material_points(self):
        q = np.array([[-.055, .025], [.060, -.030]])
        times = np.array([-.25, .125])
        from motion_static.phantoms.complex import dynamics
        translation, scale, _, _ = dynamics(times)
        r = q * scale + translation
        coordinates = np.column_stack((r, times))
        np.testing.assert_allclose(r + get_phantom("complex").motion(coordinates), q, atol=1e-15)
        # Simple disk center at t=-.25 is y=+.15; the backward displacement is -.15.
        u = get_phantom("simple").motion(np.array([[0., .15, -.25]]))
        np.testing.assert_allclose(u, [[0, -.15]], atol=1e-15)

    def test_only_complex_reference_intensity_changes_brightness(self):
        simple = make_pair("simple", 16, 9)["training"]["reference_intensity"]
        np.testing.assert_equal(simple, np.broadcast_to(simple[..., :1], simple.shape))
        complex_data = make_pair("complex", 16, 9)["training"]["reference_intensity"]
        self.assertGreater(np.std(complex_data, axis=-1).max(), .01)

    def test_multicycle_extends_time_with_same_states_and_disjoint_midpoints(self):
        for name in ("simple", "complex"):
            pair = make_pair(name, size=16, frames=49, cycles=3)
            training, midpoint = pair["training"], pair["midpoint"]
            original = make_pair(name, size=16, frames=17, cycles=1)["training"]
            np.testing.assert_array_equal(training["t"][:17], original["t"])
            self.assertEqual(training["t"][-1], 2.5)
            self.assertEqual(len(np.intersect1d(training["t"], midpoint["t"])), 0)
            for key in ("truth", "reference_intensity", "u_true"):
                for cycle in range(3):
                    # Motion has an extra component axis after the time axis.
                    np.testing.assert_allclose(training[key][:, :, cycle*16:cycle*16+17],
                                               original[key], atol=1e-7, rtol=0)
            matrix = training["reference_intensity"].reshape(-1, 49).astype(float)
            singular = np.linalg.svd(matrix, compute_uv=False)
            self.assertLess(np.linalg.norm(singular[2:]) / np.linalg.norm(singular), 1e-7)

    def test_default_sampling_bounds_remain_explicit(self):
        with self.assertRaises(ValueError):
            sample("complex", 16, [1., 2.])
        data = sample("complex", 16, [1., 2.], time_bounds=(-.5, 2.5))
        self.assertEqual(data["truth"].shape, (16, 16, 2))
        for cycles in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                make_pair("complex", size=16, frames=9, cycles=cycles)

    def test_default_complex_baseline_has_379_frames_and_378_unseen_midpoints(self):
        pair = make_pair(size=16)
        training, midpoint = pair["training"], pair["midpoint"]
        self.assertEqual(training["truth"].shape, (16, 16, 379))
        self.assertEqual(midpoint["truth"].shape, (16, 16, 378))
        np.testing.assert_array_equal(training["t"][[0, -1]], [-.5, 2.5])
        np.testing.assert_allclose(np.diff(training["t"]), 1 / 126)
        np.testing.assert_array_equal(midpoint["t"], (training["t"][:-1] + training["t"][1:]) / 2)
        self.assertEqual(len(np.intersect1d(training["t"], midpoint["t"])), 0)
        expected = sample("complex", 16, training["t"], time_bounds=(-.5, 2.5))
        np.testing.assert_array_equal(training["truth"], expected["truth"])
        simple = make_pair("simple", size=16)["training"]
        self.assertEqual(simple["truth"].shape, (16, 16, 64))
        np.testing.assert_array_equal(simple["t"][[0, -1]], [-.5, .5])

    def test_reusable_template_uses_the_approved_sampling(self):
        directory = Path(__file__).resolve().parents[1] / "experiments"
        # Besides the template: the agreed whole-series DCE-MOBY input and its Vista job scripts.
        self.assertEqual(sorted(p.name for p in directory.iterdir()),
                         ["README.md", "dce_moby_full_series.json", "template.json", "vista"])
        config = json.loads((directory / "template.json").read_text())
        self.assertEqual((config["phantom"], config["size"], config["cycles"], config["frames"]),
                         ("complex", 64, 3, 379))

    def test_training_entry_uses_template_and_preserves_explicit_overrides(self):
        from motion_static.run import parse_configuration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = [(["--model", model], ("complex", 64, 379, 3)) for model in
                        ("hash_nf", "nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf")]
            commands += [(["--cycles", "1", "--frames", "64"], ("complex", 64, 64, 1)),
                         (["--phantom", "simple"], ("simple", 64, 64, 1))]
            config_path = Path(__file__).resolve().parents[1] / "experiments/template.json"
            config = json.loads(config_path.read_text())
            config.pop("frames")
            config.pop("cycles")
            custom = root / "custom.json"
            custom.write_text(json.dumps(config))
            commands.append((["--config", str(custom)], ("complex", 64, 379, 3)))
            for arguments, expected in commands:
                with self.subTest(arguments=arguments):
                    _, resolved, _ = parse_configuration([*arguments, "--out", str(root / "run")])
                    self.assertEqual(tuple(resolved[k] for k in ("phantom", "size", "frames", "cycles")), expected)
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
