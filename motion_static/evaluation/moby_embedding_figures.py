"""Embedding-study figures for a dce_moby window in the format of Lozenski et al. 2022.

Compares SS(r) (from evaluation.moby_svd_baseline) with saved neural-field runs of the
same window and protocol, on the scored frames of that protocol (all window frames for
the embedding protocol, the 37 test frames for the held-out protocol). Writes:

- fig_singular_values.png: singular values of the fitted frames, chosen ranks marked
  (paper Fig. 4)
- fig_frames.png: two scored frames, truth | SS(r) | NF (paper Fig. 5). Frames: the one
  closest to the window's first frame (t0 geometry) and the one farthest from it.
- fig_activity_curves.png: time curves at two pixels (paper Fig. 6): the tumor
  centroid at t0 and the body pixel with the largest temporal standard deviation.
- fig_nf_decomposition.png: the NF's own outputs at six times (in place of the paper's
  partition maps, Fig. 7): truth, J(q, t) - J(q, t0) and |d(r, t)|.
- table.csv / table.md / table.png: parameters, RRMSE, PSNR, SSIM (paper Table IV).

SS ranks follow the paper's matching rules: closest parameter count to the NF, smallest
rank reaching the NF's mean SSIM, smallest rank reaching its mean RRMSE (the lowest-RRMSE
rank when SS(r) never reaches it). The first NF run provides the images and curves; the
table reports all runs. Display ranges never enter the metrics.

python -m motion_static.evaluation.moby_embedding_figures --svd SVD_DIR --nf-runs RUN [RUN ...] --out NEW
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .evaluate import predict
from .moby_branch_gifs import DIVERGING, MAGNITUDE, branch_fields
from .moby_gifs import git_commit
from .moby_svd_baseline import PROTOCOLS, reconstruct, semiseparable_factors
from .plots import pyplot
from ..phantoms import moby
from ..phantoms.data import coordinate_grid
from ..run import build_model, prepare_data


TRUTH_COLOR, NF_COLOR = "#0b0b0b", "#eb6834"
SS_COLORS = ("#2a78d6", "#1baf7a", "#6250d6")


def choose_ranks(rows, scored, nf):
    """Paper rules: parameter-matched, SSIM-matched and RRMSE-matched ranks with reasons."""
    reasons = {}
    closest = min(rows, key=lambda row: abs(row["parameters"] - nf["parameters"]))
    reasons.setdefault(closest["rank"], []).append("closest parameter count")
    defined = [row for row in rows if row[f"{scored}_ssim"] is not None]
    if nf["ssim"] is not None and defined:  # SSIM is undefined when frames are too few for its window
        ssim = next((row for row in defined if row[f"{scored}_ssim"] >= nf["ssim"]), None)
        if ssim is None:
            ssim = max(defined, key=lambda row: row[f"{scored}_ssim"])
            reasons.setdefault(ssim["rank"], []).append("highest SSIM (never reaches the NF)")
        else:
            reasons.setdefault(ssim["rank"], []).append("first to reach the NF SSIM")
    rrmse = next((row for row in rows if row[f"{scored}_rrmse"] <= nf["rrmse"]), None)
    if rrmse is None:
        rrmse = min(rows, key=lambda row: row[f"{scored}_rrmse"])
        reasons.setdefault(rrmse["rank"], []).append("lowest RRMSE (never reaches the NF)")
    else:
        reasons.setdefault(rrmse["rank"], []).append("first to reach the NF RRMSE")
    return {rank: "; ".join(text) for rank, text in sorted(reasons.items())}


def nf_scores(runs, label):
    scores = []
    for run in runs:
        report = json.loads((run / "metrics.json").read_text())
        entry = report["image"][label]
        scores.append(dict(run=run.name, parameters=report["parameters"], rrmse=entry["rrmse"],
                           psnr_db=entry["psnr_db"], ssim=entry["ssim"]))
    mean, spread = {}, {}
    for key in ("rrmse", "psnr_db", "ssim"):
        values = [s[key] for s in scores]
        defined = None not in values
        mean[key] = float(np.mean(values)) if defined else None
        spread[key] = float(np.std(values, ddof=1)) if defined and len(values) > 1 else None
    return scores, dict(parameters=scores[0]["parameters"], **mean), spread


def number(value, pattern):
    return "n/a" if value is None else pattern.format(value)


def load_model(run):
    checkpoint = torch.load(run / "model.pt", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def tumor_centroid_pixel(config, x, y):
    """Nearest (iy, ix) pixel to the tumor centroid at the window's first frame."""
    first = config["data"]["first_frame"]
    with h5py.File(config["data"]["file"], "r") as f:
        phase = int(f["resp_phase"][first])
        tumor = f["indicator"]["tumor"][..., phase]  # (x, y)
    weights = np.transpose(tumor, (1, 0))  # (y, x), partial volumes
    if weights.sum() == 0:
        raise ValueError("No tumor in this slice at the window's first frame")
    iy = np.sum(np.arange(len(y))[:, None] * weights) / weights.sum()
    ix = np.sum(np.arange(len(x))[None, :] * weights) / weights.sum()
    return int(round(iy)), int(round(ix))


def show(ax, image, low, high, cmap="gray"):
    ax.imshow(np.flipud(image), cmap=cmap, vmin=low, vmax=high, interpolation="nearest")  # y upward
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--svd", type=Path, required=True, help="output of evaluation.moby_svd_baseline")
    parser.add_argument("--nf-runs", type=Path, nargs="+", required=True, help="neural-field runs (seeds)")
    parser.add_argument("--nf-label", default="NF (ours)")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--display-max", type=float, default=0.05, help="image range 0..max, raw p0 units")
    parser.add_argument("--change-max", type=float, default=0.01, help="intensity-change range, raw p0 units")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")

    summary = json.loads((args.svd / "svd_baseline.json").read_text())
    summary.setdefault("protocol", "heldout")  # results written before the embedding protocol existed
    protocol = PROTOCOLS[summary["protocol"]]
    scored, label = protocol["scored"]
    config = json.loads((args.nf_runs[0] / "config.json").read_text())
    for run in args.nf_runs:
        run_config = json.loads((run / "config.json").read_text())
        if run_config["data"]["sha256"] != summary["data"]["sha256"] or \
                run_config["data"]["first_frame"] != summary["data"]["first_frame"]:
            parser.error(f"{run.name} uses a different data window than the SS(r) results")
        if run_config.get("held_out_split", "midpoints") != protocol["held_out_split"]:
            parser.error(f"{run.name} does not use the {summary['protocol']} protocol's split")
    data, _ = prepare_data(config)
    entry, training = data[scored], data["training"]
    scale = config["data"]["intensity_scale_value"]

    scores, nf, spread = nf_scores(args.nf_runs, label)
    rows = summary["ranks"]
    ranks = choose_ranks(rows, scored, nf)
    spatial, temporal = semiseparable_factors(training)
    shape = training["truth"].shape
    ss = {rank: reconstruct(spatial, temporal, training["t"], entry["t"], rank, shape) for rank in ranks}
    model = load_model(args.nf_runs[0])
    nf_image = predict(model, coordinate_grid(entry["x"], entry["y"], entry["t"])).reshape(entry["truth"].shape)
    truth = entry["truth"]
    args.out.mkdir(parents=True)
    plt = pyplot(args.out)
    methods = [(f"SS({rank})", ss[rank], color) for rank, color in zip(ranks, SS_COLORS)] + \
              [(args.nf_label, nf_image, NF_COLOR)]

    # Paper Fig. 4: singular values with the chosen ranks.
    values = np.asarray(summary["singular_values"])
    fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
    ax.semilogy(np.arange(1, len(values) + 1), values, color=SS_COLORS[0], lw=2, marker="o", ms=2.5,
                label="Singular values")
    ax.scatter(list(ranks), values[np.array(list(ranks)) - 1], s=60, color=NF_COLOR, zorder=3,
               edgecolor="white", linewidth=1.5, label="Chosen ranks")
    for rank in ranks:
        ax.annotate(f"r={rank}", (rank, values[rank - 1]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Index i", ylabel=r"$\sigma_i$", title="Singular values of the fitted frames")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(args.out / "fig_singular_values.png", dpi=170)
    plt.close(fig)

    # Paper Fig. 5: the scored frame nearest to the window's first frame and the farthest one.
    window_x, window_y, window_t, window_truth = moby.read_window(config["data"])
    first = window_truth[..., 0]
    distance = np.linalg.norm(truth - first[..., None], axis=(0, 1))
    frames = [int(np.argmin(distance)), int(np.argmax(distance))]
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(2.3 * (len(methods) + 1), 5.0), layout="constrained")
    for row, k in enumerate(frames):
        panels = [("Object", truth)] + [(name, image) for name, image, _ in methods]
        for column, (name, volume) in enumerate(panels):
            show(axes[row, column], volume[..., k] * scale, 0, args.display_max)
            if row == 0:
                axes[row, column].set_title(name, fontsize=10)
        axes[row, 0].set_ylabel(f"t = {entry['t'][k]:.2f} s", fontsize=10)
    fig.suptitle(f"Scored frames ({protocol['title']}); display 0..{args.display_max:g} raw p0", fontsize=10)
    fig.savefig(args.out / "fig_frames.png", dpi=170)
    plt.close(fig)

    # Paper Fig. 6: two pixels, the tumor centroid and the most variable body pixel.
    body = (window_truth > 0).any(axis=2)
    variability = np.where(body, truth.std(axis=2), -1)
    points = [("Point 1: tumor centroid", tumor_centroid_pixel(config, window_x, window_y)),
              ("Point 2: largest temporal std", tuple(int(i) for i in np.unravel_index(np.argmax(variability), variability.shape)))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
    show(axes[0], first * scale, 0, args.display_max)
    for (name, (iy, ix)), marker in zip(points, ("o", "s")):
        axes[0].plot(ix, len(window_y) - 1 - iy, marker, ms=8, mfc=NF_COLOR, mec="white", mew=1.5)
        axes[0].annotate(name.split(":")[0], (ix, len(window_y) - 1 - iy), xytext=(8, -4),
                         textcoords="offset points", color="white", fontsize=9)
    axes[0].set_title("Window frame 0", fontsize=10)
    for ax, (name, (iy, ix)) in zip(axes[1:], points):
        ax.plot(entry["t"], truth[iy, ix] * scale, color=TRUTH_COLOR, lw=1.2, marker=".", ms=4, label="Object")
        for method, volume, color in methods:
            ax.plot(entry["t"], volume[iy, ix] * scale, color=color, lw=1.5, label=method)
        ax.set(title=name, xlabel="Time (s)", ylabel="Raw p0")
        ax.grid(alpha=.25)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(args.out / "fig_activity_curves.png", dpi=170)
    plt.close(fig)

    # In place of the paper's partition maps: the NF's own decomposition at six times.
    picks = np.round(np.linspace(0, len(window_t) - 1, 6)).astype(int)
    _, intensity, _, displacement = branch_fields(model, window_x, window_y, window_t[picks],
                                                  float(model.motion.reference_time))
    _, intensity_t0, _, _ = branch_fields(model, window_x, window_y, window_t[:1], float(model.motion.reference_time))
    change = (intensity - intensity_t0) * scale
    magnitude = np.linalg.norm(displacement, axis=-1)
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.8), layout="constrained")
    for column, k in enumerate(picks):
        show(axes[0, column], window_truth[..., k] * scale, 0, args.display_max)
        show(axes[1, column], change[..., column], -args.change_max, args.change_max, cmap=DIVERGING)
        image = axes[2, column].imshow(np.flipud(magnitude[..., column]), cmap=MAGNITUDE, vmin=0,
                                       vmax=float(magnitude.max()), interpolation="nearest")
        axes[2, column].set_xticks([])
        axes[2, column].set_yticks([])
        axes[0, column].set_title(f"t = {window_t[k]:.2f} s", fontsize=10)
    for row, text in enumerate(("Object", f"J(q,t) - J(q,t0)\n±{args.change_max:g} raw", "|d(r,t)| (mm)")):
        axes[row, 0].set_ylabel(text, fontsize=10)
    fig.colorbar(image, ax=axes[2, :], shrink=0.8, label="mm")
    fig.suptitle(f"{args.nf_label} ({args.nf_runs[0].name}): intensity change and displacement", fontsize=10)
    fig.savefig(args.out / "fig_nf_decomposition.png", dpi=170)
    plt.close(fig)

    # Paper Table IV.
    table = [dict(method=f"SS({rank})", parameters=rows[rank - 1]["parameters"],
                  rrmse=rows[rank - 1][f"{scored}_rrmse"], psnr_db=rows[rank - 1][f"{scored}_psnr_db"],
                  ssim=rows[rank - 1][f"{scored}_ssim"], note=reason) for rank, reason in ranks.items()]
    table.append(dict(method=args.nf_label, parameters=nf["parameters"], rrmse=nf["rrmse"], psnr_db=nf["psnr_db"],
                      ssim=nf["ssim"], note=f"mean of {len(scores)} runs"))
    with (args.out / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "parameters", "rrmse", "psnr_db", "ssim", "note"])
        writer.writeheader()
        writer.writerows(table + [dict(method=s["run"], parameters=s["parameters"], rrmse=s["rrmse"],
                                       psnr_db=s["psnr_db"], ssim=s["ssim"], note="single run") for s in scores])
    lines = ["| Method | Parameters | RRMSE | PSNR (dB) | SSIM | Rank choice |", "|---|---:|---:|---:|---:|---|"]
    cells = []
    for item in table:
        rrmse = f"{100 * item['rrmse']:.2f}%"
        psnr, ssim = number(item["psnr_db"], "{:.2f}"), number(item["ssim"], "{:.4f}")
        if item["method"] == args.nf_label and len(scores) > 1:  # mean ± sample standard deviation
            rrmse += f" ± {100 * spread['rrmse']:.2f}"
            psnr += number(spread["psnr_db"], " ± {:.2f}") if spread["psnr_db"] is not None else ""
            ssim += number(spread["ssim"], " ± {:.4f}") if spread["ssim"] is not None else ""
        cell = [item["method"], f"{item['parameters']:,}", rrmse, psnr, ssim, item["note"]]
        cells.append(cell)
        lines.append("| " + " | ".join(cell) + " |")
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.38 * len(cells)), layout="constrained")
    ax.axis("off")
    rendered = ax.table(cellText=cells, colLabels=["Method", "Parameters", "RRMSE", "PSNR (dB)", "SSIM", "Rank choice"],
                        loc="center", cellLoc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.4)
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("#c3c2b7")
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.savefig(args.out / "table.png", dpi=170)
    plt.close(fig)

    record = dict(protocol=summary["protocol"], scored_split=scored, scored_label=label,
                  scored_frames=len(entry["t"]), svd_results=str(args.svd), nf_runs=[run.name for run in args.nf_runs],
                  nf_label=args.nf_label, ranks={str(rank): reason for rank, reason in ranks.items()},
                  frame_rule="scored frames nearest to and farthest from window frame 0 (norm of the difference)",
                  frames=[dict(index=k, time=float(entry["t"][k])) for k in frames],
                  point_rule="tumor centroid at window frame 0; body pixel with the largest temporal std of the scored truth",
                  points=[dict(name=name, iy=iy, ix=ix) for name, (iy, ix) in points],
                  decomposition_times=[float(window_t[k]) for k in picks],
                  display=dict(image=[0, args.display_max], change=[-args.change_max, args.change_max]),
                  code_commit=git_commit(), table=table, nf_runs_scores=scores, nf_spread=spread)
    (args.out / "embedding_figures.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
