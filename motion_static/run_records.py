"""Persist run metadata, source, checkpoints, logs and the derived summary CSV.

This module records existing results. It does not define losses, train models or
compute metrics. Fixed-sample loss recording belongs to training_monitor.py.
"""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Replace a JSON record atomically so readers do not see a partial file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Tee:
    """Keep terminal output visible while recording the same text in run.log."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def read_report(run):
    """Read the compact record, or an explicitly supplied historical run."""
    run = Path(run)
    path = run / "metrics.json"
    if not path.exists():
        path = run / "summary.json"
    report = json.loads(path.read_text())
    report["config"] = json.loads((run / "config.json").read_text())
    return report


def initialize_run_metadata(config):
    """Add runtime descriptions in place, clearing status from any prior run."""
    config.pop("error", None)
    config.pop("finished_at", None)
    config.update({
        "status": "running",
        "started_at": utc_now(),
        "device": "cpu",
        "scheduler": "cosine to zero",
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "loss": "MSE(predicted image, observed image)",
        "true_motion_used_for_training": False,
        "reference_intensity_labels_used_for_training": False,
        "midpoint_labels_used_for_training": False,
        "joint_period": 1.,
        "time_bounds": config["coordinate_bounds"][2],
        "intervals_per_cycle": (config["frames"] - 1) / config["cycles"],
        "monitoring": "Fixed uniform samples without replacement; image MSE; no validation gradients",
        "monitor_lr_convention": "Learning rates after scheduler step, for the next optimizer update",
    })
    if config.get("temporal_weight", 0) > 0:
        config["loss"] += " + temporal_weight * intensity_temporal_l2"
    # Oracle diagnostics feed one true branch into the model during training.
    oracle = config.get("oracle_branch")
    if oracle is not None:
        config.update(
            true_motion_used_for_training=oracle == "motion",
            reference_intensity_labels_used_for_training=oracle == "intensity",
            oracle_note=f"Diagnostic only: the {oracle} branch is the phantom's ground truth, not learned",
        )


def snapshot_source(root, out):
    """Save actual Python sources (including local edits) and their hashes."""
    hashes = {}
    with ZipFile(out / "source.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root)
            if relative.parts[0] in ("archive", "results"):
                continue
            name = str(Path(root.name) / relative)
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_archive": "source.zip",
        "source_sha256": hashes,
    }


def source_hashes():
    """Fingerprint maintained Python sources before preparing or resuming a queue."""
    root = Path(__file__).resolve().parent
    hashes = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] not in ("archive", "results"):
            hashes[str(Path(root.name) / relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def save_checkpoint(out, model, optimizer, scheduler, config, step):
    """Save an explicitly requested intermediate step, including optimizer/RNG."""
    checkpoint_dir = Path(out) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = {
        "config": config,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, checkpoint_dir / f"step_{step:06d}.pt")


SUMMARY_FIELDS = (
    'run', 'status', 'phantom', 'model', 'oracle_branch', 'evaluation_split', 'size', 'cycles', 'frames',
    'rank', 'steps', 'seed', 'motion_levels', 'motion_min_resolution', 'motion_max_resolution',
    'intensity_levels', 'intensity_min_resolution', 'intensity_max_resolution',
    'temporal_width', 'temporal_encoding', 'temporal_mlp_lr', 'image_rrmse', 'image_psnr_db', 'image_ssim', 'intensity_rrmse',
    'intensity_psnr_db', 'intensity_ssim', 'motion_vector_rrmse',
    'motion_endpoint_rmse_pixels', 'motion_mean_endpoint_error_pixels',
    'fixed_training_mse', 'fixed_validation_mse', 'training_seconds', 'git_commit',
)


def summary_row(run, config, report=None):
    """Build one labeled CSV row; missing metrics stay blank, not zero.

    The two evaluation protocols select different image/intensity domains.
    Motion always uses the existing foreground displacement metrics.
    """
    # Configuration and provenance.
    row = {key: config.get(key, "") for key in SUMMARY_FIELDS}
    row["run"] = run.name
    row["status"] = config.get("status", "unknown")
    row["git_commit"] = config.get("code_provenance", {}).get("git_commit", "")
    if config["model"] not in ("low_rank_hash_nf", "nested_low_rank_hash_nf"):
        row["rank"] = row["temporal_width"] = ""
    if isinstance(row["temporal_encoding"], dict):
        row["temporal_encoding"] = json.dumps(row["temporal_encoding"], sort_keys=True)

    # Motion settings inherit from the intensity hash unless overridden.
    if config["model"] in ("nested_hash_nf", "low_rank_hash_nf", "nested_low_rank_hash_nf"):
        intensity_settings = config.get("encoding", {})
        motion_settings = {**intensity_settings, **config.get("motion_encoding", {})}
        branches = [("intensity", intensity_settings)]
        if config["model"] != "low_rank_hash_nf":
            branches.append(("motion", motion_settings))
        for branch, settings in branches:
            for key in ("levels", "min_resolution", "max_resolution"):
                value = settings.get(key, "")
                if isinstance(value, list):
                    value = json.dumps(value)  # Per-axis resolutions fit in one CSV cell.
                row[f"{branch}_{key}"] = value

    # Select saved metrics; do not recompute or normalize any values here.
    if report is not None:
        if report["evaluation_split"] == "blocks":
            image_scores = report["image"]["test_pixels"]
            intensity_scores = report.get("reference_intensity_full_grid", {})
            motion_scores = report.get("motion", {}).get("foreground", {})
        else:
            image_scores = report["image"]["unseen_time_midpoints"]
            intensity_scores = report.get("reference_intensity", {}).get("unseen_time_midpoints", {})
            motion_scores = report.get("motion", {}).get("midpoint", {}).get("foreground", {})

        for branch, scores in (("image", image_scores), ("intensity", intensity_scores)):
            for metric in ("rrmse", "psnr_db", "ssim"):
                row[f"{branch}_{metric}"] = scores.get(metric, "")
        for metric in ("vector_rrmse", "endpoint_rmse_pixels", "mean_endpoint_error_pixels"):
            row[f"motion_{metric}"] = motion_scores.get(metric, "")
        row["training_seconds"] = report["training_seconds"]

    # Even an incomplete run can have a useful final monitoring row.
    last_losses = read_last_losses(run)
    for key in ("fixed_training_mse", "fixed_validation_mse"):
        if last_losses:
            row[key] = last_losses.get(key, "")
    return row


def read_last_losses(run):
    """Read the final logged step without loading the full loss history."""
    path = run / "loss.csv"
    if not path.exists():
        return {}
    last_row = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            last_row = row
    return last_row


def update_summary(root):
    """Regenerate summary.csv with explicit protocol labels under a writer lock."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Lock the directory itself so no extra lock file is left in results/.
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rows = []
        for config_path in sorted(root.glob("*/config.json")):
            run = config_path.parent
            config = json.loads(config_path.read_text())
            report_path = run / "metrics.json"
            report = None
            if config.get("status") == "complete" and report_path.exists():
                report = json.loads(report_path.read_text())
            rows.append(summary_row(run, config, report))

        # Write the complete index before replacing the previous one.
        with tempfile.NamedTemporaryFile(mode="w", newline="", dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "summary.csv")
    finally:
        os.close(descriptor)
