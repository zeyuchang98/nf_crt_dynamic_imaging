"""Plateau schedule: validation/test split, rate reductions, stop, kept weights, outputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from motion_static.evaluation import moby_split_eval
from motion_static.evaluation.evaluate import image_only_report
from motion_static.evaluation.metrics import save_metrics
from motion_static.phantoms import moby
from motion_static.run import build_model, main, parse_configuration, prepare_data, train
from motion_static.tests.test_moby_data import DATA, write_slice
from motion_static.training_monitor import fixed_mse


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class PlateauScheduleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.path = self.root / "slice.h5"
        write_slice(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def write_config(self, **overrides):
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        template.update(phantom=moby.NAME, data=dict(DATA, file=str(self.path)), rank=2, steps=6,
                        batch_size=64, threads=1, monitor_samples=32, log_interval=1, **overrides)
        for key in ("size", "frames", "cycles"):
            template.pop(key)
        path = self.root / "config.json"
        path.write_text(json.dumps(template))
        return path

    def test_midpoints_alternate_into_validation_and_test(self):
        data, _ = moby.resolve_settings(dict(DATA, file=str(self.path)))
        pair = moby.make_pair(data)
        split = moby.split_midpoints(pair)
        self.assertEqual(sorted(split), ["test", "training", "validation"])
        self.assertIs(split["training"], pair["training"])
        # Six midpoints: window frames 1, 3, ..., 11; validation 1, 5, 9 and test 3, 7, 11.
        np.testing.assert_array_equal(split["validation"]["t"], pair["midpoint"]["t"][[0, 2, 4]])
        np.testing.assert_array_equal(split["test"]["t"], pair["midpoint"]["t"][[1, 3, 5]])
        np.testing.assert_array_equal(split["test"]["truth"], pair["midpoint"]["truth"][..., [1, 3, 5]])
        self.assertFalse(set(split["validation"]["t"]) & set(split["test"]["t"]))

    def test_rates_drop_on_plateau_then_stop_and_keep_best_weights(self):
        config = dict(phantom="complex", model="nested_low_rank_hash_nf", rank=2,
                      head_width=8, temporal_width=8, steps=50, batch_size=8,
                      mlp_lr=.001, hash_lr=.01, log_interval=1, monitor_samples=16,
                      coordinate_bounds=[[-.5, .5], [-.5, .5], [-.5, 2.5]], reference_time=-.5,
                      encoding=dict(levels=2, features=2, min_resolution=4, max_resolution=8, log2_hash_size=8),
                      schedule="plateau",
                      # Only a 1000-fold drop would count as improvement, so every evaluation
                      # after the first is a plateau.
                      plateau=dict(factor=.5, patience=0, threshold=.999, max_reductions=2))
        rng = np.random.default_rng(7)
        x = rng.uniform([-.5, -.5, -.5], [.5, .5, 2.5], (32, 3)).astype(np.float32)
        y = x[:, 0] + .1 * x[:, 2]
        selection = (x[:16], y[:16] + .3)
        torch.manual_seed(9)
        model = build_model(config)
        with self.assertRaisesRegex(ValueError, "validation frames"):
            train(model, x, y, config)
        out = self.root / "run"
        out.mkdir()
        _, _, summary = train(model, x, y, config, out=out, validation=selection, selection=selection)
        self.assertEqual(summary["stopped_by"], "plateau")
        self.assertEqual(summary["rate_reduction_steps"], [2, 3])
        self.assertEqual(summary["stop_step"], 4)
        rows = read_rows(out / "loss.csv")
        self.assertEqual([int(row["step"]) for row in rows], [0, 1, 2, 3, 4])
        rates = [float(row["next_lr_intensity_hash"]) for row in rows]
        np.testing.assert_allclose(rates, [.01, .01, .005, .0025, .00125])
        logged = {int(row["step"]): float(row["validation_frames_mse"]) for row in rows[1:]}
        self.assertEqual(summary["best_step"], min(logged, key=logged.get))
        self.assertEqual(summary["best_validation_frames_mse"], min(logged.values()))
        kept = fixed_mse(model, torch.tensor(selection[0]), torch.tensor(selection[1]))
        self.assertEqual(kept, summary["best_validation_frames_mse"])

    def test_configuration_rules(self):
        out = str(self.root / "run")
        with self.assertRaises(SystemExit):  # the plateau schedule needs the alternate split
            parse_configuration(["--config", str(self.write_config(schedule="plateau")), "--out", out])
        with self.assertRaises(SystemExit):
            parse_configuration(["--config", str(self.write_config(plateau=dict(patience=2))), "--out", out])
        # Toy phantoms have no separate test frames yet, so both options are refused there.
        template = json.loads((Path(moby.__file__).parents[1] / "experiments/template.json").read_text())
        for overrides in (dict(schedule="plateau", held_out_split="alternate_validation_test"),
                          dict(held_out_split="alternate_validation_test")):
            toy = self.root / "toy.json"
            toy.write_text(json.dumps(dict(template, **overrides)))
            with self.assertRaises(SystemExit):
                parse_configuration(["--config", str(toy), "--out", out])
        _, config, _ = parse_configuration(["--config", str(self.write_config(
            schedule="plateau", held_out_split="alternate_validation_test", plateau=dict(patience=2))), "--out", out])
        self.assertEqual(config["plateau"], dict(factor=.5, patience=2, threshold=1e-3, max_reductions=4))
        _, default, _ = parse_configuration(["--config", str(self.write_config()), "--out", out])
        self.assertNotIn("schedule", default)
        self.assertNotIn("plateau", default)

    def test_tiny_plateau_run_scores_validation_and_test_frames(self):
        out = self.root / "results" / "run"
        config_path = self.write_config(schedule="plateau", held_out_split="alternate_validation_test",
                                        plateau=dict(patience=0, threshold=.999, max_reductions=0),
                                        output=dict(gifs=False, predictions=True))
        main(["--config", str(config_path), "--out", str(out)])
        config = json.loads((out / "config.json").read_text())
        report = json.loads((out / "metrics.json").read_text())
        self.assertEqual(config["status"], "complete")
        self.assertEqual(config["early_stopping"]["stop_step"], 2)
        self.assertEqual(report["early_stopping"], config["early_stopping"])
        self.assertEqual(sorted(report["image"]), ["test_midpoints", "training_grid", "validation_midpoints"])
        self.assertEqual(len(report["image"]["test_midpoints"]["times"]), 3)
        checkpoint = torch.load(out / "model.pt", weights_only=True)
        self.assertEqual(checkpoint["step"], config["early_stopping"]["best_step"])
        self.assertIn("validation_frames_mse", read_rows(out / "loss.csv")[0])
        with np.load(out / "predictions.npz") as saved:
            self.assertEqual(sorted(saved), ["test_image", "test_t", "training_image", "training_t",
                                             "validation_image", "validation_t", "x", "y"])
        summary = read_rows(out.parent / "summary.csv")
        self.assertEqual(summary[0]["evaluation_split"], "midpoints, test half")
        self.assertEqual(float(summary[0]["image_rrmse"]), report["image"]["test_midpoints"]["rrmse"])

    def test_saved_cosine_run_is_rescored_on_the_test_frames(self):
        out = self.root / "results" / "cosine"
        main(["--config", str(self.write_config()), "--out", str(out)])
        config = json.loads((out / "config.json").read_text())
        saved = json.loads((out / "metrics.json").read_text())
        checkpoint = torch.load(out / "model.pt", weights_only=True)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        # The reloaded model reproduces the saved midpoint metrics exactly.
        report, _ = image_only_report(model, prepare_data(config)[0], config, saved["training_seconds"])
        save_metrics(self.root / "rescored.json", report)
        self.assertEqual(json.loads((self.root / "rescored.json").read_text()), saved)
        moby_split_eval.main([str(out)])
        rescored = json.loads((out / "metrics_alternate_split.json").read_text())
        self.assertEqual(sorted(rescored["image"]), ["test_midpoints", "training_grid", "validation_midpoints"])
        self.assertEqual(rescored["image"]["training_grid"], saved["image"]["training_grid"])
        self.assertEqual(json.loads((out / "metrics.json").read_text()), saved)


if __name__ == "__main__":
    unittest.main()
