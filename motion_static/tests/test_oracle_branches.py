"""Oracle diagnostics: true-branch formulas, initialization parity, optimization scope."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from motion_static.models.oracle_branches import TrueMotion, TrueReferenceIntensity
from motion_static.phantoms import complex as phantom
from motion_static.phantoms.data import coordinate_grid, sample
from motion_static.run import build_model, build_optimizer, execute, parse_configuration, train
from motion_static.evaluation.render import load_run
from motion_static.tests.test_run_records import small_config


def oracle_config(branch):
    config = small_config()
    config.update(oracle_branch=branch, steps=3)
    return config


class OracleBranchTests(unittest.TestCase):
    def test_torch_formulas_match_numpy_phantom_inside_and_outside_image(self):
        rng = np.random.default_rng(0)
        xyt = np.column_stack((rng.uniform(-.7, .7, (500, 2)), rng.uniform(-.5, 2.5, 500)))
        xyt[:3, :2] = [[0, 0], [-.055, .025], [.06, -.03]]  # Exact ellipse centers.
        points = torch.as_tensor(xyt, dtype=torch.float64)
        motion = TrueMotion()(points).numpy()
        np.testing.assert_allclose(motion, phantom.motion(xyt), rtol=0, atol=1e-12)
        for size in (16, 64):
            intensity = TrueReferenceIntensity(size)(points).numpy()
            np.testing.assert_allclose(intensity, phantom.reference_intensity(xyt, size), rtol=0, atol=1e-12)

    def test_true_motion_is_anchored_and_both_oracles_reproduce_observed_images(self):
        data = sample("complex", 16, np.linspace(-.5, 2.5, 7), (-.5, 2.5))
        xyt = torch.as_tensor(coordinate_grid(data["x"], data["y"], data["t"]), dtype=torch.float64)
        reference = xyt.clone()
        reference[:, 2] = -.5
        self.assertEqual(TrueMotion()(reference).abs().max().item(), 0.)
        warped = xyt.clone()
        warped[:, :2] += TrueMotion()(xyt)
        image = TrueReferenceIntensity(16)(warped).reshape(data["truth"].shape).numpy()
        np.testing.assert_allclose(image, data["truth"], rtol=0, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "anchored"):
            TrueMotion(reference_time=.1)

    def test_oracle_keeps_learned_branch_initialization_and_rng_state(self):
        for branch, learned in (("motion", "intensity"), ("intensity", "motion")):
            with self.subTest(branch=branch):
                torch.manual_seed(4)
                baseline = build_model(small_config())
                baseline_rng = torch.get_rng_state()
                torch.manual_seed(4)
                oracle = build_model(oracle_config(branch))
                self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
                expected = getattr(baseline, learned).state_dict()
                for name, value in getattr(oracle, learned).state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]), name)
                self.assertEqual(list(getattr(oracle, branch).parameters()), [])

    def test_optimizer_covers_only_and_all_learned_parameters(self):
        expected_groups = {
            "motion": ["intensity_hash", "intensity_mlp", "intensity_time_mlp"],
            "intensity": ["motion_hash", "motion_mlp"],
        }
        for branch, names in expected_groups.items():
            with self.subTest(branch=branch):
                config = oracle_config(branch)
                model = build_model(config)
                optimizer = build_optimizer(model, config)
                self.assertEqual([group["name"] for group in optimizer.param_groups], names)
                optimized = [p for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(optimized), len({id(p) for p in optimized}))
                self.assertEqual({id(p) for p in optimized}, {id(p) for p in model.parameters()})

    def test_training_updates_learned_branch_through_true_field_gradients(self):
        for branch in ("motion", "intensity"):
            with self.subTest(branch=branch):
                torch.manual_seed(0)
                config = oracle_config(branch)
                config["batch_size"] = 256
                model = build_model(config)
                data = sample("complex", 16, np.linspace(-.5, 2.5, 9), (-.5, 2.5))
                coordinates = coordinate_grid(data["x"], data["y"], data["t"])
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                with redirect_stdout(io.StringIO()):
                    train(model, coordinates, data["truth"].reshape(-1), config)
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    # Motion biases cancel in F(r,t) - F(r,t0) while both queries
                    # share ReLU patterns, as for the learned-intensity model.
                    if name.startswith("motion.") and name.endswith("bias"):
                        continue
                    self.assertFalse(torch.equal(before[name], parameter), name)
                points = torch.as_tensor(coordinates, dtype=torch.float32)
                if branch == "motion":
                    torch.testing.assert_close(model.motion(points), TrueMotion()(points))
                else:
                    reference = points.clone()
                    reference[:, 2] = -.5
                    self.assertEqual(model.motion(reference).abs().max().item(), 0.)

    def test_configuration_rejects_unsupported_oracle_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            template = json.loads((Path(__file__).resolve().parents[1] / "experiments/template.json").read_text())
            cases = [(dict(oracle_branch="image"), []),
                     (dict(oracle_branch="motion"), ["--model", "nested_hash_nf"]),
                     (dict(oracle_branch="motion"), ["--phantom", "simple"]),
                     (dict(oracle_branch="motion"), ["--split", "blocks"]),
                     (dict(oracle_branch="motion", reference_time=0.), []),
                     (dict(oracle_branch="intensity", temporal_weight=.1), [])]
            for extra, arguments in cases:
                with self.subTest(extra=extra, arguments=arguments):
                    path.write_text(json.dumps(template | extra))
                    with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                            patch("sys.stderr", io.StringIO()):
                        parse_configuration(["--config", str(path), "--dry-run", *arguments])
            path.write_text(json.dumps(template | dict(oracle_branch="motion", rank=2)))
            _, resolved, _ = parse_configuration(["--config", str(path), "--dry-run"])
            self.assertEqual(resolved["oracle_branch"], "motion")

    def test_complete_run_labels_oracle_and_reloads_saved_model(self):
        for branch in ("motion", "intensity"):
            with self.subTest(branch=branch), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                config = oracle_config(branch)
                with redirect_stdout(io.StringIO()):
                    execute(config, out)
                saved = json.loads((out / "config.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(saved["true_motion_used_for_training"], branch == "motion")
                self.assertEqual(saved["reference_intensity_labels_used_for_training"], branch == "intensity")
                report = json.loads((out / "metrics.json").read_text())
                if branch == "motion":
                    motion = report["motion"]["midpoint"]["foreground"]
                    self.assertLess(motion["endpoint_rmse_pixels"], 1e-4)
                    self.assertIn("intensity_rank", report)
                else:
                    self.assertLess(report["reference_intensity"]["unseen_time_midpoints"]["rrmse"], 1e-6)
                    self.assertNotIn("intensity_rank", report)
                with (out / "loss.csv").open() as stream:
                    header = stream.readline()
                self.assertNotIn(f"next_lr_{branch}_", header)
                model, _ = load_run(out)
                self.assertEqual(type(getattr(model, branch)).__name__,
                                 "TrueMotion" if branch == "motion" else "TrueReferenceIntensity")
                summary = (out.parent / "summary.csv").read_text().splitlines()
                self.assertIn("oracle_branch", summary[0].split(","))
                self.assertIn(f",{branch},", summary[1])


if __name__ == "__main__":
    unittest.main()
