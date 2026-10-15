"""Verify and compare the approved sweep using the existing metric definitions."""
import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from .compare_gifs import save_animation
from .compare_training import load_model
from .evaluate import intensity_rank_report
from .metrics import error_metrics
from .plots import pyplot, save_comparison
from ..hash_sweep import (CONFIG_DIR, RESULT_DIR, LEVELS, MOTION_MAX, INTENSITY_MAX,
                          artifact_files, load_manifest, sha256, validate_config, write_json)
from ..phantoms.data import coordinate_grid, make_pair
from ..training_monitor import fixed_mse

METRICS = (("image_rrmse_percent", "Image RRMSE (%)"),
           ("intensity_rrmse_percent", "Intensity RRMSE (%)"),
           ("motion_rmse_pixels", "Foreground motion RMSE (px)"))


def validate_result(run, expected, source_hashes, check_only=False):
    config = json.loads((run / "config.json").read_text())
    validate_config(config, expected)
    verification = run / "verification.json"
    if check_only:
        report = json.loads(verification.read_text())
        assert report["status"] == "passed"
        assert report["source_sha256"] == source_hashes
        for name, digest in report["artifact_sha256"].items():
            assert sha256(run / name) == digest, name
        return report
    torch.set_num_threads(config["threads"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["config"] == config
    model, checkpoint = load_model(run / "model.pt")
    assert checkpoint["config"] == config and checkpoint["step"] == config["steps"]
    for step in config["checkpoint_steps"]:
        saved = torch.load(run / "checkpoints" / f"step_{step:06d}.pt", weights_only=True, map_location="cpu")
        assert saved["step"] == step and saved["config"] == config
        assert saved["scheduler"]["T_max"] == config["steps"]
        if step == config["steps"]:
            assert all(torch.equal(value, saved["state_dict"][key])
                       for key, value in checkpoint["state_dict"].items())
    hashes = config["code_provenance"]["source_sha256"]
    assert hashes == source_hashes
    import hashlib
    with ZipFile(run / "source.zip") as archive:
        assert set(archive.namelist()) == set(hashes)
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    data = make_pair(config["phantom"], config["size"], config["frames"], config["cycles"])
    with np.load(run / "arrays/predictions.npz") as arrays:
        for split in data:
            np.testing.assert_array_equal(arrays[f"{split}_t"], data[split]["t"])
        coordinates = coordinate_grid(data["midpoint"]["x"], data["midpoint"]["y"], data["midpoint"]["t"])
        indices = np.random.default_rng(2026).choice(len(coordinates), 511, replace=False)
        query = torch.tensor(coordinates[indices], dtype=torch.float32, requires_grad=True)
        prediction = model(query)
        difference = float(np.max(np.abs(prediction.detach().numpy() - arrays["midpoint_image"].reshape(-1)[indices])))
        assert difference < 1e-6
        gradient = torch.autograd.grad(prediction.sum(), query)[0]
        assert torch.isfinite(gradient).all() and gradient[:, :2].abs().max() > 0
        truth = data["midpoint"]
        spacing = float(truth["x"][1] - truth["x"][0])
        error = (arrays["midpoint_motion"].astype(float) - truth["u_true"])[truth["foreground_mask"]]
        motion_rmse = float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1))) / spacing)
        np.testing.assert_allclose(motion_rmse, summary["motion"]["midpoint"]["foreground"]["endpoint_rmse_pixels"], rtol=1e-12)
        # True and predicted motion query the SAME time-varying intensity.
        save_comparison(run / "plots/motion_effect.gif", truth["truth"],
                        arrays["midpoint_learned_motion_true_intensity"], truth["t"],
                        "Motion comparison: both panels use the same true time-varying intensity",
                        labels=("True motion + true C(t)", "Pred motion + true C(t)", "Absolute error"))
    reference = coordinate_grid(data["training"]["x"], data["training"]["y"], [config["reference_time"]])
    with torch.no_grad():
        assert model.motion(torch.tensor(reference, dtype=torch.float32)).abs().max() == 0
    with (run / "loss.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected_steps = sorted({0, 1, config["steps"], *config["checkpoint_steps"],
                             *range(config["log_interval"], config["steps"] + 1, config["log_interval"])})
    assert [int(row["step"]) for row in rows] == expected_steps
    assert all(float(row["temporal_l2"]) == 0 and row["batch_mse"] == row["total_loss"] for row in rows[1:])
    for row in rows:
        assert all(np.isfinite(float(value)) for value in row.values() if value)
        for key, value in row.items():
            if key.startswith("next_lr_"):
                start_lr = config["hash_lr"] if key.endswith("_hash") else config["mlp_lr"]
                expected_lr = start_lr * (1 + np.cos(np.pi * int(row["step"]) / config["steps"])) / 2
                np.testing.assert_allclose(float(value), expected_lr, atol=1e-14, rtol=1e-9)
    assert all(float(rows[-1][key]) == 0 for key in rows[-1] if key.startswith("next_lr_"))
    monitoring_rng = np.random.default_rng(config["monitor_seed"])
    recomputed_losses = {}
    with np.load(run / "monitor_samples.npz") as monitoring:
        for split, name in (("training", "training"), ("midpoint", "validation")):
            entry = data[split]
            full = coordinate_grid(entry["x"], entry["y"], entry["t"])
            selected = monitoring_rng.choice(len(full), min(config["monitor_samples"], len(full)), replace=False)
            np.testing.assert_array_equal(selected, monitoring[f"{name}_indices"])
            loss = fixed_mse(model, torch.tensor(full[selected], dtype=torch.float32),
                             torch.tensor(entry["truth"].reshape(-1)[selected], dtype=torch.float32))
            np.testing.assert_allclose(loss, float(rows[-1][f"fixed_{name}_mse"]), rtol=1e-7, atol=1e-12)
            recomputed_losses[name] = loss
    rank_volumes = {}
    for name in ("image", "reference_intensity"):
        with np.load(run / "arrays" / f"{name}.npz") as arrays:
            for split, prefix, label in (("training", "", "training_grid"),
                                         ("midpoint", "midpoint_", "unseen_time_midpoints")):
                key = "truth" if name == "image" else name
                np.testing.assert_array_equal(arrays[prefix + "truth"], data[split][key])
                rrmse = error_metrics(arrays[prefix + "truth"], arrays[prefix + "prediction"], data_range=1)["rrmse"]
                np.testing.assert_allclose(rrmse, summary[name][label]["rrmse"], rtol=1e-12)
                if name == "reference_intensity":
                    rank_volumes[split] = arrays[prefix + "prediction"]
    ranks = intensity_rank_report(rank_volumes, config["rank"])
    assert summary["intensity_rank"]["rank_limit"] == 2
    assert all(v["relative_frobenius_tail_after_rank"] < 1e-6 for v in ranks["splits"].values())
    animations = {}
    for name in ("image", "reference_intensity", "motion_only", "motion_effect"):
        with Image.open(run / "plots" / f"{name}.gif") as animation:
            assert animation.n_frames == config["frames"] - 1
            assert animation.size == (864, 408) and animation.info["duration"] == 90
            distinct = set()
            for frame in np.linspace(0, animation.n_frames - 1, 11, dtype=int):
                animation.seek(int(frame))
                distinct.add(hashlib.sha256(animation.convert("RGB").crop((0, 76, 288, 364)).tobytes()).hexdigest())
            if name == "motion_effect":
                assert len(distinct) > 1
            animations[name] = dict(frames=animation.n_frames, sampled_truth_panels=len(distinct))
    for name in ("loss_curve.png", "loss_curve.svg"):
        assert (run / "plots" / name).stat().st_size > 0
    report = dict(status="passed", final_step=config["steps"], loss_rows=len(rows),
                  prediction_sample_max_error=difference, reference_anchor_max=0.,
                  reference_points=len(reference), finite_nonzero_coordinate_gradients=True,
                  fixed_losses_recomputed=recomputed_losses, cosine_schedule_verified=True,
                  image_intensity_and_motion_metrics_recomputed=True,
                  rank_relative_tails={k: v["relative_frobenius_tail_after_rank"] for k, v in ranks["splits"].items()},
                  gifs=animations, source_sha256=hashes,
                  artifact_sha256={str(p.relative_to(run)): sha256(p) for p in artifact_files(run)
                                   if p != verification})
    write_json(verification, report)
    print(f"Verified {run.name}: checkpoint, sources, rank, anchor, gradients, metrics, losses and GIFs", flush=True)
    return report


def nondominated(rows):
    """Minimize all three existing errors; ties remain on the frontier."""
    values = np.array([[row[key] for key, _ in METRICS] for row in rows])
    return [row["run_id"] for i, row in enumerate(rows)
            if not np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))]


def collect_rows(config_dir, group):
    manifest = load_manifest(config_dir)
    rows = []
    for name in manifest["run_ids"]:
        run = group / "runs" / name
        if not (run / "verification.json").exists():
            continue
        verification = json.loads((run / "verification.json").read_text())
        assert verification["status"] == "passed"
        for key in ("summary.json", "config.json", "loss.csv"):
            assert sha256(run / key) == verification["artifact_sha256"][key]
        summary = json.loads((run / "summary.json").read_text())
        config = summary["config"]
        validate_config(config, json.loads((config_dir / f"{name}.json").read_text()))
        with (run / "loss.csv").open() as stream:
            last = list(csv.DictReader(stream))[-1]
        rows.append(dict(run_id=name, motion_levels=config["motion_encoding"]["levels"],
                         motion_max=config["motion_encoding"]["max_resolution"][0],
                         intensity_levels=config["encoding"]["levels"], intensity_max=config["encoding"]["max_resolution"],
                         image_rrmse_percent=100 * summary["image"]["unseen_time_midpoints"]["rrmse"],
                         intensity_rrmse_percent=100 * summary["reference_intensity"]["unseen_time_midpoints"]["rrmse"],
                         motion_rmse_pixels=summary["motion"]["midpoint"]["foreground"]["endpoint_rmse_pixels"],
                         fixed_training_mse=float(last["fixed_training_mse"]),
                         fixed_validation_mse=float(last["fixed_validation_mse"]),
                         parameters=summary["parameters"], training_seconds=summary["training_seconds"]))
    return manifest, rows


def plot_results(group, rows, manifest):
    plt = pyplot(group)
    motion = [(level, maximum) for level in LEVELS for maximum in MOTION_MAX]
    intensity = [(level, maximum) for level in LEVELS for maximum in INTENSITY_MAX]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (key, label) in zip(axes, METRICS):
        values = np.full((6, 6), np.nan)
        for row in rows:
            i = motion.index((row["motion_levels"], row["motion_max"]))
            j = intensity.index((row["intensity_levels"], row["intensity_max"]))
            values[i, j] = row[key]
        view = ax.imshow(np.ma.masked_invalid(values), cmap="viridis_r")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                    fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1))
        ax.set(xticks=range(6), xticklabels=[f"L{l}/{r}" for l, r in intensity],
               yticks=range(6), yticklabels=[f"L{l}/{r}" for l, r in motion],
               xlabel="Intensity: levels / max resolution", ylabel="Motion: levels / spatial max", title=label)
        fig.colorbar(view, ax=ax, shrink=.7)
    budget = "SMOKE ONLY: 3 steps" if manifest["smoke_only"] else "24k steps"
    fig.suptitle(f"{budget}, seed 0: {len(rows)}/{len(manifest['run_ids'])} verified runs; lower is better; blank = no result")
    fig.savefig(group / "metric_heatmaps.png", dpi=170)
    fig.savefig(group / "metric_heatmaps.svg")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True, layout="constrained")
    for ax, (ml, mr) in zip(axes.flat, motion):
        for row in rows:
            if (row["motion_levels"], row["motion_max"]) != (ml, mr):
                continue
            with (group / "runs" / row["run_id"] / "loss.csv").open() as stream:
                history = list(csv.DictReader(stream))
            color = f"C{intensity.index((row['intensity_levels'], row['intensity_max']))}"
            for key, style in (("fixed_training_mse", "-"), ("fixed_validation_mse", "--")):
                ax.semilogy([int(r["step"]) for r in history], [float(r[key]) for r in history],
                            style, color=color, label=f"I L{row['intensity_levels']}/max{row['intensity_max']}" if style == "--" else None)
        ax.set(title=f"Motion L{ml}, spatial max {mr}", xlabel="Optimizer steps", ylabel="Image MSE")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle(f"{budget}; fixed samples: solid = training, dashed = held-out; no smoothing")
    fig.savefig(group / "loss_comparison.png", dpi=170)
    fig.savefig(group / "loss_comparison.svg")
    plt.close(fig)


def comparison_gifs(group, names):
    reports = []
    # Limit columns so captions and image errors remain readable.
    for start in range(0, len(names), 3):
        chosen = names[start:start + 3]
        destination = group / "comparisons" / f"group_{start // 3 + 1:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        runs = [group / "runs" / name for name in chosen]
        labels = [name.replace("_max", "/").replace("_i", " I").replace("m", "M", 1) for name in chosen]
        inputs = {str((run / item).relative_to(group)): sha256(run / item) for run in runs
                  for item in ("config.json", "arrays/image.npz", "arrays/reference_intensity.npz", "arrays/predictions.npz")}
        with np.load(runs[0] / "arrays/predictions.npz") as arrays:
            times = arrays["midpoint_t"]
        for run in runs[1:]:
            with np.load(run / "arrays/predictions.npz") as arrays:
                np.testing.assert_array_equal(times, arrays["midpoint_t"])
        report = dict(run_ids=chosen, source_sha256=inputs, animations={})
        for name, array_name, special, title in (
                ("image", "image", None, "Image fitting: shared times and display scales"),
                ("intensity", "reference_intensity", None, "Intensity at fixed reference coordinates"),
                ("motion", "image", "midpoint_learned_motion_true_intensity", "Motion: same true time-varying intensity in every panel")):
            truth, predictions = None, []
            for run in runs:
                with np.load(run / "arrays" / f"{array_name}.npz") as arrays:
                    if truth is None:
                        truth = arrays["midpoint_truth"]
                    else:
                        np.testing.assert_array_equal(truth, arrays["midpoint_truth"])
                    if special is None:
                        predictions.append(arrays["midpoint_prediction"])
                if special:
                    with np.load(run / "arrays/predictions.npz") as arrays:
                        predictions.append(arrays[special])
            path = destination / f"{name}.gif"
            report["animations"][name] = save_animation(path, truth, predictions, times, labels, title)
            with Image.open(path) as animation:
                assert animation.n_frames == len(times) and animation.size == ((len(chosen) + 1) * 240, 636)
        write_json(destination / "provenance.json", report)
        reports.append(dict(directory=str(destination.relative_to(group)), run_ids=chosen))
    return reports


def summarize(config_dir, group, make_gifs=False):
    manifest, rows = collect_rows(config_dir, group)
    if not rows:
        raise ValueError("No verified results available")
    full = len(rows) == len(manifest["run_ids"])
    if make_gifs and not full:
        raise ValueError("Final GIF selection requires the full sweep")
    best = {key: min(rows, key=lambda row: row[key])["run_id"] for key, _ in METRICS}
    frontier = nondominated(rows)
    baseline = "mL8_max32_iL8_max64"
    selected = list(dict.fromkeys([*([baseline] if baseline in {r['run_id'] for r in rows} else []),
                                   *best.values(), *frontier]))
    if manifest["smoke_only"]:
        # Validate multi-model layouts even when one tiny probe dominates another.
        selected = [row["run_id"] for row in rows]
    report = dict(status="complete" if full else "partial", smoke_only=manifest["smoke_only"],
                  completed=len(rows), total=len(manifest["run_ids"]), rows=rows,
                  per_metric_best=best, nondominated=frontier, gif_selection=selected,
                  note="Single-seed exploratory selection, not an independent test; no weighted score. Runtime is not a controlled benchmark.",
                  source_sha256={str((group / "runs" / row["run_id"] / "summary.json").relative_to(group)):
                                 sha256(group / "runs" / row["run_id"] / "summary.json") for row in rows},
                  analysis_sha256={"hash_sweep.py": sha256(Path(__file__)),
                                   "compare_gifs.py": sha256(Path(__file__).with_name("compare_gifs.py"))})
    if make_gifs:
        report["comparisons"] = comparison_gifs(group, selected)
    with (group / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(group / "comparison.json", report)
    plot_results(group, rows, manifest)
    lines = ["# Hash Grid Sweep: " + ("3-Step Smoke Test" if manifest["smoke_only"] else "24,000 Steps"), "",
             f"**{len(rows)}/{len(manifest['run_ids'])} runs verified; {report['status']}.** "
             + ("SMOKE TEST ONLY; these are not research results." if manifest["smoke_only"] else
                "All runs are newly trained. Previous 32k-run checkpoints are excluded."), "",
             "Motion levels 4/8/12, spatial max 16/32; intensity levels 4/8/12, max 32/64. "
             "Motion temporal endpoints remain 8/128, with intermediate scales changing with levels. "
             "Rank 2; image MSE only; unchanged phantom, anchor and evaluation definitions.", "",
             "[Numeric table](metrics.csv) · [Comparison data](comparison.json) · [Metric heatmaps](metric_heatmaps.png) "
             "· [Fixed loss comparison](loss_comparison.png)", "",
             "Per-metric best among verified runs: " + "; ".join(f"{label}: `{best[key]}`" for key, label in METRICS) + ".", "",
             "Nondominated tradeoffs: " + ", ".join(f"`{name}`" for name in frontier) + ".", "",
             "No weighted score is used. Midpoint data are used for hyperparameter comparison, so this is "
             "exploratory selection rather than an independent generalization test. Same seed does not guarantee "
             "identical initialization or batches when parameter shapes change. No real PACT claim is made.", "",
             "| Run (motion / intensity) | Image RRMSE (%) | Intensity RRMSE (%) | Motion RMSE (px) | Artifacts |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        name = row["run_id"]
        folder = f"runs/{name}"
        links = " · ".join(f"[{label}]({folder}/{path})" for label, path in (
            ("Metrics", "summary.json"), ("Image", "plots/image.gif"), ("Intensity", "plots/reference_intensity.gif"),
            ("Motion truth", "plots/motion_effect.gif"), ("Motion alone", "plots/motion_only.gif"),
            ("Loss", "plots/loss_curve.png"), ("Loss/LR CSV", "loss.csv"), ("Checks", "verification.json")))
        lines.append(f"| {name} | {row['image_rrmse_percent']:.4f} | {row['intensity_rrmse_percent']:.4f} | {row['motion_rmse_pixels']:.4f} | {links} |")
    if make_gifs:
        lines.extend(["", "## Shared-time comparisons", "",
                      "Reference configuration, per-metric best configurations and all nondominated configurations, "
                      "deduplicated into groups of at most three. Display/error scales are shared within each GIF. "
                      "Motion panels query the same true time-varying intensity; the truth panel moves.", ""])
        for entry in report["comparisons"]:
            directory = entry["directory"]
            lines.append("- " + ", ".join(entry["run_ids"]) + ": " + " · ".join(
                f"[{name}]({directory}/{name}.gif)" for name in ("image", "intensity", "motion")))
    (group / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(rows)}/{len(manifest['run_ids'])} verified runs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "summarize"))
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out", type=Path, default=RESULT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--gifs", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.config_dir)
    if args.action == "verify":
        if args.run_id not in manifest["run_ids"]:
            parser.error("--run-id must identify an approved configuration")
        expected = json.loads((args.config_dir / f"{args.run_id}.json").read_text())
        validate_result(args.out / "runs" / args.run_id, expected, manifest["source_sha256"])
    else:
        summarize(args.config_dir, args.out, args.gifs)


if __name__ == "__main__":
    main()
