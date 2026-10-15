"""Motion-free representation, optimizer coverage, records and rank selection."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from motion_static.baselines import baseline_configs, load_plan, prepare, select_best_rank
from motion_static.evaluation.evaluate import predict_fields
from motion_static.evaluation.block_evaluate import evaluate_blocks
from motion_static.evaluation.render import load_run
from motion_static.models.low_rank_intensity import LowRankIntensity
from motion_static.run import build_model, build_optimizer, execute, parse_configuration, prepare_data, train
from motion_static.run_records import summary_row
from motion_static.tests.test_run_records import record_initial_model, small_config


class BaselineTests(unittest.TestCase):
    def test_exact_five_run_scope_and_cli_resolution(self):
        configs = baseline_configs()
        self.assertEqual(len(configs), 5)
        self.assertEqual([c["rank"] for c in configs.values() if c["model"] == "low_rank_hash_nf"], [8, 16, 24])
        self.assertEqual(configs["low_rank_motion_r2"]["rank"], 2)
        for config in configs.values():
            self.assertEqual((config["steps"], config["frames"], config["cycles"], config["size"]),
                             (8000, 379, 3, 64))
            self.assertEqual(config["encoding"], dict(levels=8, features=2, log2_hash_size=15,
                                                     min_resolution=8, max_resolution=128))
            self.assertEqual(config["temporal_weight"], 0.)
            self.assertEqual(config["output"], dict(gifs=False, predictions=False))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            scratch = Path(directory) / "inputs"
            prepare(scratch)
            _, resolved = load_plan(scratch)
            self.assertNotIn("motion_encoding", resolved["low_rank_only_r8"])
            self.assertNotIn("reference_time", resolved["low_rank_only_r8"])
            path = scratch / "low_rank_only_r8.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "input changed"):
                load_plan(scratch)

    def test_approved_models_have_expected_branches_rank_gradients_and_parameter_groups(self):
        torch.set_num_threads(4)
        for name, config in baseline_configs().items():
            with self.subTest(run=name):
                torch.manual_seed(0)
                model = build_model(config).double()
                optimizer = build_optimizer(model, config)
                parameters = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(parameters), len(set(parameters)))
                self.assertEqual(set(parameters), {id(p) for p in model.parameters()})
                for group in optimizer.param_groups:
                    self.assertEqual(group["lr"], .01 if group["name"].endswith("_hash") else .001)
                points = torch.tensor([[.13, -.21, .19], [-.31, .27, 1.2]],
                                      dtype=torch.float64, requires_grad=True)
                prediction = model(points)
                self.assertEqual(tuple(prediction.shape), (2,))
                gradient = torch.autograd.grad(prediction.sum(), points)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient[:, :2].abs().sum().item(), 0.)
                if config["model"] == "low_rank_hash_nf":
                    self.assertIsInstance(model, LowRankIntensity)
                    self.assertFalse(hasattr(model, "motion"))
                    field = model
                else:
                    reference = points.detach().clone()
                    reference[:, 2] = -.5
                    torch.testing.assert_close(model.motion(reference), torch.zeros(2, 2).double(), atol=0, rtol=0)
                    field = model.intensity
                if hasattr(field, "rank"):
                    xy = torch.rand(37, 2, dtype=torch.float64) - .5
                    times = torch.linspace(-.5, 2.5, 35, dtype=torch.float64)
                    queries = torch.cat((xy[:, None].expand(-1, 35, -1),
                                         times[None, :, None].expand(37, -1, -1)), dim=-1)
                    with torch.no_grad():
                        singular = torch.linalg.svdvals(field(queries.reshape(-1, 3)).reshape(37, 35))
                    self.assertLess((torch.linalg.vector_norm(singular[field.rank:]) /
                                     torch.linalg.vector_norm(singular)).item(), 1e-12)

    def test_motion_free_training_updates_spatial_and_temporal_parameters_and_roundtrips(self):
        config = small_config()
        config.update(model="low_rank_hash_nf", rank=8, steps=3)
        torch.manual_seed(5)
        model = build_model(config)
        points = torch.rand(64, 3)
        points[:, :2] -= .5
        target = points[:, 0] + points[:, 2].square()
        tracked = [model.encoding.table, model.mlp[-1].weight, model.temporal[-1].weight]
        initial = [p.detach().clone() for p in tracked]
        with redirect_stdout(io.StringIO()):
            train(model, points.numpy(), target.numpy(), config)
        for before, parameter in zip(initial, tracked):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
            self.assertFalse(torch.equal(before, parameter))
        saved = io.BytesIO()
        torch.save(dict(config=config, state_dict=model.state_dict()), saved)
        saved.seek(0)
        checkpoint = torch.load(saved, weights_only=True)
        restored = build_model(checkpoint["config"])
        restored.load_state_dict(checkpoint["state_dict"])
        torch.testing.assert_close(restored(points), model(points), rtol=0, atol=0)

    def test_motion_free_records_do_not_report_fake_branch_metrics(self):
        config = small_config()
        config.update(model="low_rank_hash_nf", rank=8)
        config.pop("motion_encoding", None)
        config.pop("reference_time", None)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory) / "run"
            with patch("motion_static.run.train", side_effect=record_initial_model):
                execute(config, out)
            report = json.loads((out / "metrics.json").read_text())
            self.assertNotIn("reference_intensity", report)
            self.assertNotIn("motion", report)
            self.assertEqual(report["image_rank"]["rank_limit"], 8)
            model, saved_config = load_run(out)
            data, _ = prepare_data(saved_config)
            self.assertEqual(set(predict_fields(model, data["midpoint"], saved_config)), {"image"})
            row = summary_row(out, saved_config, report)
            self.assertEqual(row["rank"], 8)
            self.assertEqual(row["motion_levels"], "")
            self.assertEqual(row["intensity_rrmse"], "")
            self.assertNotEqual(row["image_rrmse"], "")

            block_config = copy.deepcopy(config)
            block_config.update(frames=16, evaluation_split="blocks",
                                block_split=dict(block_shape=[8, 8, 2], test_fraction=.2, seed=0))
            block_data, split = prepare_data(block_config)
            block_out = Path(directory) / "blocks"
            block_out.mkdir()
            block_report = evaluate_blocks(model, block_data["training"], split, block_config, block_out, 0.)
            self.assertNotIn("motion", block_report)
            self.assertNotIn("reference_intensity_full_grid", block_report)
            self.assertEqual(block_report["image_rank"]["rank_limit"], 8)

    def test_motion_free_temporal_penalty_is_rejected(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parse_configuration(["--model", "low_rank_hash_nf", "--rank", "8", "--temporal-weight", ".1"])

    def test_selection_uses_validation_not_full_grid_scores_and_checks_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for rank, validation, rrmse in ((8, .4, .01), (16, .2, .2), (24, .2, .1)):
                run = Path(directory) / f"r{rank}"
                run.mkdir()
                runs[f"low_rank_only_r{rank}"] = run
                (run / "loss.csv").write_text(f"step,fixed_validation_mse\n8000,{validation}\n")
                (run / "metrics.json").write_text(json.dumps({"image": {
                    "unseen_time_midpoints": dict(rrmse=rrmse, psnr_db=20, ssim=.9)}}))
                np.savez(run / "monitor_samples.npz", validation_indices=[3, 2, 1])
            selection = select_best_rank(runs)
            self.assertEqual(selection["selected_rank"], 16)
            np.savez(runs["low_rank_only_r24"] / "monitor_samples.npz", validation_indices=[1, 2, 3])
            with self.assertRaisesRegex(ValueError, "identical validation"):
                select_best_rank(runs)


if __name__ == "__main__":
    unittest.main()
