"""Scope guards and independent branch settings for the approved factorial sweep."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from motion_static.hash_sweep import approved_configs, load_manifest, prepare, validate_config
from motion_static.evaluation.hash_sweep import nondominated
from motion_static.run import build_model


class HashSweepTests(unittest.TestCase):
    def test_all_36_unique_combinations_have_the_same_24k_budget(self):
        configs = approved_configs()
        self.assertEqual(len(configs), 36)
        combinations = set()
        normalized = []
        for config in configs.values():
            self.assertEqual(config["steps"], 24000)
            self.assertEqual(config["checkpoint_steps"], [8000, 16000, 24000])
            self.assertEqual(config["motion_encoding"]["min_resolution"], [2, 2, 8])
            self.assertEqual(config["motion_encoding"]["max_resolution"][2], 128)
            self.assertEqual(config["encoding"]["min_resolution"], 8)
            motion, intensity = config["motion_encoding"], config["encoding"]
            combinations.add((motion["levels"], motion["max_resolution"][0],
                              intensity["levels"], intensity["max_resolution"]))
            canonical = copy.deepcopy(config)
            for field in ("motion_encoding", "encoding"):
                canonical[field].pop("levels")
                canonical[field].pop("max_resolution")
            normalized.append(canonical)
        self.assertEqual(len(combinations), 36)
        self.assertTrue(all(value == normalized[0] for value in normalized))

    def test_asymmetric_layer_counts_do_not_leak_between_branches(self):
        for name in ("mL4_max16_iL12_max64", "mL12_max32_iL4_max32"):
            settings = approved_configs()[name]
            model = build_model(settings)
            self.assertEqual(model.motion.field.encoding.resolutions.shape[0], settings["motion_encoding"]["levels"])
            self.assertEqual(model.intensity.encoding.resolutions.shape[0], settings["encoding"]["levels"])
            reference = torch.tensor([[.13, -.21, -.5]])
            self.assertEqual(model.motion(reference).abs().max().item(), 0.)

    def test_changed_loss_data_schedule_or_motion_time_is_rejected(self):
        expected = next(iter(approved_configs().values()))
        for key, value in (("steps", 32000), ("frames", 64), ("rank", 3),
                           ("temporal_weight", .001), ("reference_time", 0),
                           ("midpoint_labels_used_for_training", True)):
            changed = copy.deepcopy(expected)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed, expected)
        changed = copy.deepcopy(expected)
        changed["motion_encoding"]["max_resolution"][2] = 64
        with self.assertRaises(ValueError):
            validate_config(changed, expected)

    def test_missing_independent_motion_level_is_rejected(self):
        expected = next(iter(approved_configs().values()))
        changed = copy.deepcopy(expected)
        changed["motion_encoding"].pop("levels")
        with self.assertRaises(ValueError):
            validate_config(changed, expected)

    def test_prepared_config_modification_and_duplicate_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configs"
            prepare(path)
            manifest = load_manifest(path)
            self.assertEqual(len(manifest["run_ids"]), 36)
            config_path = path / f"{manifest['run_ids'][0]}.json"
            original = config_path.read_text()
            config_path.write_text(original + " ")
            with self.assertRaises(ValueError):
                load_manifest(path)
            config_path.write_text(original)
            manifest["run_ids"][1] = manifest["run_ids"][0]
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_nondominated_frontier_retains_tradeoffs_and_ties(self):
        def row(name, values):
            return dict(zip(("run_id", "image_rrmse_percent", "intensity_rrmse_percent", "motion_rmse_pixels"),
                            (name, *values)))
        rows = [row("image_best", (1, 3, 3)), row("branch_best", (2, 1, 1)),
                row("dominated", (3, 3, 3)), row("tie", (2, 1, 1))]
        self.assertEqual(nondominated(rows), ["image_best", "branch_best", "tie"])


if __name__ == "__main__":
    unittest.main()
