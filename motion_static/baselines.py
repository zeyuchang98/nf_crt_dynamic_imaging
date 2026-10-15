"""Prepare/run the approved five-run baseline comparison using the shared trainer.

Nested hash; low-rank + motion at R=2; motion-free low rank at R=8,16,24.
All runs use the complex 3-cycle/379-frame benchmark and an 8k step budget.
Generated inputs live in a scratch directory; experiments/ keeps its template.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .run import parse_configuration
from .run_records import read_last_losses, source_hashes, write_json

ROOT = Path(__file__).resolve().parents[1]
LOW_RANK_CANDIDATES = (8, 16, 24)
REQUIRED_RECORDS = (
    "config.json", "metrics.json", "loss.csv", "loss.png", "model.pt",
    "monitor_samples.npz", "source.zip", "run.log",
)


def baseline_configs():
    """Explicit approved settings; do not inherit future template budget changes."""
    base = json.loads((ROOT / "motion_static/experiments/template.json").read_text())
    base.update(
        phantom="complex", size=64, frames=379, cycles=3, seed=0, steps=8000,
        batch_size=4096, threads=4, head_width=64, temporal_width=32,
        mlp_lr=.001, hash_lr=.01, temporal_weight=0.,
        evaluation_split="midpoints", monitor_samples=65536, monitor_seed=2026,
        log_interval=500, checkpoint_steps=[], output=dict(gifs=False, predictions=False),
        coordinate_bounds=[[-.5, .5], [-.5, .5], [-.5, 2.5]], reference_time=-.5,
    )
    base["encoding"] = dict(
        levels=8, features=2, log2_hash_size=15, min_resolution=8, max_resolution=128,
    )
    base["motion_encoding"] = dict(base["encoding"])
    variants = [("nested_hash", "nested_hash_nf", None),
                ("low_rank_motion_r2", "nested_low_rank_hash_nf", 2)]
    variants += [(f"low_rank_only_r{rank}", "low_rank_hash_nf", rank)
                 for rank in LOW_RANK_CANDIDATES]
    configs = {}
    for name, model, rank in variants:
        config = copy.deepcopy(base)
        config["model"] = model
        if rank is None:
            config.pop("rank", None)
            config.pop("rank_definition", None)
        else:
            config["rank"] = rank
        if model == "low_rank_hash_nf":
            config.pop("motion_encoding")
            config.pop("reference_time")
        configs[name] = config
    return configs


def prepare(config_dir):
    """Save a reviewable plan and freeze the exact input/source content."""
    config_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, config in baseline_configs().items():
        path = config_dir / f"{name}.json"
        write_json(path, config)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(config_dir / "manifest.json", {
        "run_ids": list(hashes),
        "config_sha256": hashes,
        "source_sha256": source_hashes(),
        "selection": "Lowest final fixed_validation_mse; exact ties choose the smaller rank",
        "selection_scope": "R=8,16,24 motion-free models at step 8000; no early stopping",
        "validation_note": "Midpoints select rank and are not an independent test set after selection",
    })
    print(f"Prepared five baseline inputs: {config_dir}")


def load_plan(config_dir):
    manifest = json.loads((config_dir / "manifest.json").read_text())
    approved = baseline_configs()
    if manifest["run_ids"] != list(approved):
        raise ValueError("The prepared plan must contain exactly the five approved baselines")
    configs = {}
    for name, expected in approved.items():
        path = config_dir / f"{name}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["config_sha256"][name]:
            raise ValueError(f"Prepared input changed: {name}")
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Settings differ from the approved baseline: {name}")
        # Use the real CLI resolution/validation, including motion-free cleanup.
        _, config, _ = parse_configuration(["--config", str(path)])
        configs[name] = config
    return manifest, configs


def verify_run(run, expected, expected_source):
    """Reject incomplete/stale records instead of silently resuming or rerunning."""
    for name in REQUIRED_RECORDS:
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing baseline record: {path}")
    config = json.loads((run / "config.json").read_text())
    if config.get("status") != "complete":
        raise ValueError(f"Inspect incomplete run before continuing: {run}")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Run setting differs: {run.name}/{key}")
    if config["code_provenance"]["source_sha256"] != expected_source:
        raise ValueError(f"Run sources differ: {run}")
    losses = read_last_losses(run)
    if int(losses["step"]) != expected["steps"]:
        raise ValueError(f"Final monitoring step differs: {run}")
    report = json.loads((run / "metrics.json").read_text())
    if report["evaluation_split"] != "midpoints":
        raise ValueError(f"Wrong evaluation protocol: {run}")
    return losses, report


def select_best_rank(runs):
    """Select from final fixed validation MSE, never batch loss or branch truth."""
    candidates = []
    common_indices = None
    for rank in LOW_RANK_CANDIDATES:
        run = runs[f"low_rank_only_r{rank}"]
        with np.load(run / "monitor_samples.npz", allow_pickle=False) as samples:
            indices = samples["validation_indices"]
            if common_indices is None:
                common_indices = indices.copy()
            elif not np.array_equal(indices, common_indices):
                raise ValueError("Rank candidates must use identical validation monitoring points")
        losses = read_last_losses(run)
        mse = float(losses["fixed_validation_mse"])
        if not np.isfinite(mse):
            raise ValueError(f"Nonfinite selection score: {run}")
        report = json.loads((run / "metrics.json").read_text())
        image = report["image"]["unseen_time_midpoints"]
        candidates.append({
            "rank": rank, "run": run.name, "fixed_validation_mse": mse,
            "full_midpoint_rrmse": image["rrmse"], "full_midpoint_psnr_db": image["psnr_db"],
            "full_midpoint_ssim": image["ssim"],
        })
    best = min(candidates, key=lambda row: (row["fixed_validation_mse"], row["rank"]))
    return {
        "criterion": "Lowest final fixed midpoint validation image MSE; ties choose smaller rank",
        "selected_rank": best["rank"], "selected_run": best["run"], "candidates": candidates,
        "all_runs": {name: path.name for name, path in runs.items()},
        "scope": "Best among R=8,16,24 at 8k steps, seed 0; all candidates retained",
        "limitation": "Rank-selected midpoint scores are validation results, not an independent test",
    }


def run_queue(config_dir, results):
    manifest, configs = load_plan(config_dir)
    results.mkdir(parents=True, exist_ok=True)
    runs = {name: results / f"{config_dir.name}_{name}" for name in configs}
    with (config_dir / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, config in configs.items():
            subprocess.run(["git", "pull", "--ff-only"], cwd=ROOT, check=True)
            if source_hashes() != manifest["source_sha256"]:
                raise RuntimeError("Code changed after preparation; validate and prepare a new plan")
            out = runs[name]
            if not out.exists():
                command = [sys.executable, "-u", "-m", "motion_static.run", "--config",
                           str(config_dir / f"{name}.json"), "--out", str(out)]
                print(f"Starting {name}: {out}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            verify_run(out, config, manifest["source_sha256"])
        selection_path = results / f"{config_dir.name}_selection.json"
        write_json(selection_path, select_best_rank(runs))
        print(f"Completed all five baselines. Rank selection: {selection_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config-dir", type=Path, required=True,
                        help="Fresh scratch directory outside maintained experiments/ and results/")
    parser.add_argument("--out", type=Path, default=ROOT / "motion_static/results")
    args = parser.parse_args(argv)
    config_dir = args.config_dir.resolve()
    if any(config_dir.is_relative_to(path.resolve()) for path in
           (ROOT / "motion_static/experiments", ROOT / "motion_static/results", args.out)):
        parser.error("Keep generated inputs outside experiments/ and results/")
    if args.action == "prepare":
        prepare(config_dir)
    else:
        run_queue(config_dir, args.out.resolve())


if __name__ == "__main__":
    main()
