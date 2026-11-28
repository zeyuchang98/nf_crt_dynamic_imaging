"""Evaluation of 2D inverse reconstructions: image scores, single-pixel curves and lesion curves.

Every window frame of each run (model.pt) is rendered once and compared with the slice file's
truth, normalized by the run's intensity scale:
- Image scores as in the run's metrics.json (inverse/execute.score_all_frames): RRMSE, PSNR and
  the space-time SSIM over (y, x, t) with 7 x 7 x 7 windows and data range 1 (the truth's range
  after normalization), plus per-frame scores. Also the same space-time SSIM with ProxNF's code
  convention (applications/proximal_splitting/create_animations.py): data range = the
  reconstruction's own max - min over all frames, so it differs between methods.
- Single-pixel time-activity curves at the tumor core, the tumor shell and the artery. Each
  region's pixel is fixed in the image: fully inside the region (indicator 1) at every
  respiratory phase of the window's frames, and the deepest such pixel (largest distance to a
  pixel not fully inside, at the worst phase). Curve RRMSE = ||c - c_truth||_2 / ||c_truth||_2
  over all frames.
- Lesion activity curves (METRICS.md 3.6): partial-volume weighted means over each frame's own
  phase mask for the whole lesion, its shell and its core, and the LAC RRMSE.
Curves are saved and plotted in raw p0 units (normalized values times intensity_scale_value);
the RRMSEs do not depend on the scale. Writes inverse_eval.json, inverse_eval.npz,
point_curves.png, lesion_curves.png and per_frame_scores.png.

    python -m motion_static.evaluation.inverse_evaluate --runs RUN [RUN ...] --out NEW [--device mps]
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from .evaluate import bounding_roi
from .inverse_figures import STYLE, run_key
from .metrics import _ssim, evaluate_sequence_in_blocks, save_metrics
from .moby_embedding_figures import load_model
from .moby_gifs import git_commit
from .moby_lesion_curves import TRUTH, region_weights, scores as lac_scores
from .plots import pyplot
from ..phantoms import moby

# Point name -> indicator dataset of the slice file ("tumor" is the lesion's shell).
POINT_REGIONS = {"core": "tumor_core", "shell": "tumor", "artery": "artery"}
SSIM_WINDOW = 7
FALLBACK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6", "#c2416b")
ZOOM_SECONDS = 96.  # right-hand column of the curve figures: the first 96 s (wash-in)


def indicator_maps(path, names):
    """Per indicator, (y, x, phase) values; the file stores (x, y, phase)."""
    with h5py.File(path, "r") as f:
        return {name: np.transpose(f["indicator"][name][...], (1, 0, 2)) for name in names}


def deepest_inside_pixel(full):
    """The fixed pixel deepest inside a moving region.

    full: (y, x, phase) booleans, True where the pixel lies wholly inside the region. Depth at
    one phase is the Euclidean distance to the nearest pixel not wholly inside (1 next to the
    boundary); a pixel's depth is its smallest depth over the phases. Returns (row, col, depth,
    number of pixels inside at every phase); ties go to the first pixel in row-major order.
    """
    always = full.all(axis=-1)
    if not always.any():
        raise ValueError("No pixel lies wholly inside the region at every phase")
    depth = np.min([distance_transform_edt(full[..., p]) for p in range(full.shape[-1])], axis=0)
    depth = np.where(always, depth, 0.)
    row, col = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return int(row), int(col), float(depth[row, col]), int(always.sum())


def region_points(path, phases):
    """Per point name, the deepest fixed pixel over the given respiratory phases."""
    maps = indicator_maps(path, POINT_REGIONS.values())
    points = {}
    for name, indicator in POINT_REGIONS.items():
        # Indicators are means over 7 Gauss points (values k/7): wholly inside means 7/7.
        full = maps[indicator][..., np.unique(phases)] >= 1 - 1e-6
        row, col, depth, count = deepest_inside_pixel(full)
        points[name] = dict(indicator=indicator, row=row, col=col, depth_px=depth, pixels_inside_every_phase=count)
    return points


@torch.no_grad()
def render_frames(model, x, y, times, device, chunk=8):
    """(frames, y, x) float32 images, as InverseProblem.render: x-fastest pixels, POUnet by psi(t) C^T."""
    yy, xx = np.meshgrid(y, x, indexing="ij")
    xy = torch.as_tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=device)
    stamps = torch.as_tensor(np.asarray(times), dtype=torch.float32, device=device)
    images = np.empty((len(times), len(y), len(x)), dtype=np.float32)
    for start in range(0, len(times), chunk):
        part = stamps[start:start + chunk]
        if hasattr(model, "frames"):
            values = model.frames(part)
        else:
            points = torch.cat([xy.repeat(len(part), 1), part.repeat_interleave(len(xy))[:, None]], dim=1)
            values = model(points)
        images[start:start + len(part)] = values.reshape(len(part), len(y), len(x)).float().cpu().numpy()
    return images


def curve_rrmse(curve, truth):
    return float(np.linalg.norm(curve - truth) / np.linalg.norm(truth))


def lesion_curves(weights, phases, frame_values):
    """Per region, sum_r w(r, phase(t)) f(r, t) / sum_r w(r, phase(t)) for every frame.

    weights: region -> (y, x, phase); phases: each frame's phase; frame_values(picks) returns
    the frames `picks` as a (y, x, len(picks)) array.
    """
    curves = {region: np.zeros(len(phases)) for region in weights}
    for phase in np.unique(phases):
        picks = np.nonzero(phases == phase)[0]
        values = frame_values(picks).astype(np.float64)
        for region, volume in weights.items():
            w = volume[..., phase].astype(np.float64)
            curves[region][picks] = np.tensordot(w, values, axes=([0, 1], [0, 1])) / w.sum()
    return curves


def evaluate_reconstruction(truth, images, times, points, weights, phases, roi):
    """Scores of one reconstruction against the normalized truth.

    truth: (y, x, t); images: (t, y, x) of the same frames. Returns (report, curves) with the
    curves in normalized units.
    """
    recon = np.moveaxis(images, 0, -1)  # (y, x, t) view
    image = evaluate_sequence_in_blocks(truth, lambda frames: recon[..., frames], data_range=1, times=times, roi=roi)
    low, high = float(images.min()), float(images.max())
    proxnf_ssim = _ssim(truth, recon, high - low, SSIM_WINDOW)
    point_curves = {name: images[:, p["row"], p["col"]].astype(np.float64) for name, p in points.items()}
    lac = lesion_curves(weights, phases, lambda picks: recon[..., picks])
    report = dict(
        rrmse=image["rrmse"], psnr_db=image["psnr_db"], ssim=image["ssim"],
        ssim_proxnf_convention=proxnf_ssim, reconstruction_range=[low, high],
        mean_per_frame_ssim=float(np.mean(image["per_frame_ssim"])),
        roi=dict(rrmse=image["roi"]["rrmse"], psnr_db=image["roi"]["psnr_db"], ssim=image["roi"]["ssim"]),
        point_curve_rrmse={name: curve_rrmse(point_curves[name], truth[p["row"], p["col"]].astype(np.float64))
                           for name, p in points.items()})
    per_frame = dict(rrmse=np.asarray(image["per_frame_rrmse"]), ssim=np.asarray(image["per_frame_ssim"]))
    return report, dict(points=point_curves, lesion=lac, per_frame=per_frame)


def saved_check(run, report, per_frame):
    """Differences from the run's own metrics.json (rendered on its device during the run)."""
    path = run / "metrics.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())["image"]["all_window_frames"]
    return dict(rrmse_difference=report["rrmse"] - saved["rrmse"], ssim_difference=report["ssim"] - saved["ssim"],
                max_per_frame_rrmse_difference=float(np.max(np.abs(per_frame["rrmse"] - saved["per_frame_rrmse"]))),
                max_per_frame_ssim_difference=float(np.max(np.abs(per_frame["ssim"] - saved["per_frame_ssim"]))))


def styles_for(runs, labels):
    """Display label and colour per run folder name: inverse_figures.STYLE when known."""
    styles = {}
    for index, (run, label) in enumerate(zip(runs, labels)):
        known = STYLE.get(run_key(run))
        styles[run.name] = (label or (known[0] if known else run.name),
                            known[1] if known else FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
    return styles


def plot_curves(path, curves, times, rrmse, styles, title, ylabel):
    """Rows: regions; columns: the whole series and its first ZOOM_SECONDS. Truth on top, in black."""
    plt = pyplot(path.parent)
    fig, axes = plt.subplots(len(curves), 2, figsize=(13, 3.0 * len(curves)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[3, 1.3]), layout="constrained")
    zoom = times <= times[0] + ZOOM_SECONDS
    for row, (region, entry) in enumerate(curves.items()):
        for column, part in enumerate([slice(None), zoom]):
            ax = axes[row, column]
            for name, curve in entry.items():
                if name != TRUTH:
                    label, color = styles[name]
                    ax.plot(times[part], curve[part], color=color, lw=0.6, alpha=0.85,
                            label=f"{label}: RRMSE {100 * rrmse[region][name]:.1f}%")
            ax.plot(times[part], entry[TRUTH][part], color="black", lw=1.1, label="Truth")
            ax.grid(alpha=0.25)
            ax.set_title(f"{region}" + (f" (first {ZOOM_SECONDS:.0f} s)" if column else ""), fontsize=10)
            if row == len(curves) - 1:
                ax.set_xlabel("Time (s)")
            if column == 0:
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_frame(path, times, per_frame, styles, window=72):
    """Thin: every frame; thick: mean over `window` frames (72 = one sensor revolution, as
    inverse_figures), for display only, since the scores swing with each breath."""
    plt = pyplot(path.parent)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    kernel = np.ones(window) / window
    smooth_times = np.convolve(times, kernel, mode="valid")
    for name, scores in per_frame.items():
        label, color = styles[name]
        for ax, values in zip(axes, (100 * scores["rrmse"], scores["ssim"])):
            ax.plot(times, values, color=color, lw=0.3, alpha=0.3)
            ax.plot(smooth_times, np.convolve(values, kernel, mode="valid"), color=color, lw=1.4, label=label)
    axes[0].set_ylabel(f"Frame RRMSE (%)\nthin: per frame; thick: mean over {window}")
    axes[1].set_ylabel(f"Frame SSIM, data range 1\nthin: per frame; thick: mean over {window}")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, framealpha=0.9, ncol=2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="2D inverse run folders with model.pt")
    parser.add_argument("--labels", nargs="*", default=[], help="display labels for --runs, in order")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", default="cpu", help="device for rendering (cpu, mps, cuda)")
    parser.add_argument("--data-file", type=Path, help="local copy of the slice file, checked by its sha256")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")
    if len(args.labels) > len(args.runs):
        parser.error("More labels than runs")

    configs = {run.name: json.loads((run / "config.json").read_text()) for run in args.runs}
    if len(configs) != len(args.runs):
        parser.error("Run folder names must differ")
    data = dict(next(iter(configs.values()))["data"])
    window = ("sha256", "first_frame", "cycles", "frames_per_cycle", "intensity_scale_value")
    for name, config in configs.items():
        if "inverse" not in config or config.get("status") != "complete":
            parser.error(f"{name} is not a complete inverse run")
        if any(config["data"].get(key) != data.get(key) for key in window):
            parser.error(f"{name} uses a different data window or intensity scale")
    if args.data_file is not None:
        data["file"] = str(args.data_file.resolve())
    if moby.file_sha256(data["file"]) != data["sha256"]:
        parser.error(f"{data['file']} does not have the sha256 recorded by the runs")

    entry = moby.read_window_entry(data)
    truth, times = entry["truth"], entry["t"]
    scale = data["intensity_scale_value"]
    weights, phases = region_weights(data["file"], moby.window_frames(data))
    points = region_points(data["file"], phases)
    for p in points.values():
        p.update(x_mm=float(entry["x"][p["col"]]), y_mm=float(entry["y"][p["row"]]))
    roi = bounding_roi([truth])
    print("Points:", {name: (p["row"], p["col"], p["depth_px"]) for name, p in points.items()}, flush=True)

    point_curves = {name: {TRUTH: truth[p["row"], p["col"]].astype(np.float64) * scale} for name, p in points.items()}
    lac = {region: {TRUTH: curve * scale}
           for region, curve in lesion_curves(weights, phases, lambda picks: truth[..., picks]).items()}
    reports, per_frame = {}, {}
    device = torch.device(args.device)
    for run in args.runs:
        start = time.perf_counter()
        model = load_model(run).to(device)
        images = render_frames(model, entry["x"], entry["y"], times, device)
        rendered = time.perf_counter() - start
        del model
        report, curves = evaluate_reconstruction(truth, images, times, points, weights, phases, roi)
        del images
        report.update(run=str(run), model=configs[run.name]["model"], method=configs[run.name]["inverse"]["method"],
                      render_seconds=rendered, evaluation_seconds=time.perf_counter() - start - rendered,
                      check_against_saved_metrics=saved_check(run, report, curves["per_frame"]))
        reports[run.name], per_frame[run.name] = report, curves["per_frame"]
        for name, curve in curves["points"].items():
            point_curves[name][run.name] = curve * scale
        for region, curve in curves["lesion"].items():
            lac[region][run.name] = curve * scale
        print(f"{run.name}: RRMSE {report['rrmse']:.4%}, SSIM {report['ssim']:.4f} (ProxNF convention "
              f"{report['ssim_proxnf_convention']:.4f}), points {report['point_curve_rrmse']}, "
              f"render {rendered:.0f} s, check {report['check_against_saved_metrics']}", flush=True)

    lac_report = lac_scores(lac, times)
    for name, report in reports.items():
        report["lesion_lac_rrmse"] = {region: lac_report[region][name]["lac_rrmse"] for region in lac}
    point_rrmse = {region: {name: reports[name]["point_curve_rrmse"][region] for name in reports} for region in points}
    lac_rrmse = {region: {name: reports[name]["lesion_lac_rrmse"][region] for name in reports} for region in lac}

    args.out.mkdir(parents=True)
    styles = styles_for(args.runs, args.labels + [None] * (len(args.runs) - len(args.labels)))
    np.savez_compressed(args.out / "inverse_eval.npz", times=times,
                        **{f"point|{region}|{name}": c for region, entry in point_curves.items() for name, c in entry.items()},
                        **{f"lesion|{region}|{name}": c for region, entry in lac.items() for name, c in entry.items()},
                        **{f"per_frame_{score}|{name}": values[score] for name, values in per_frame.items()
                           for score in ("rrmse", "ssim")})
    plot_curves(args.out / "point_curves.png", point_curves, times, point_rrmse, styles,
                "Single-pixel time-activity curves (fixed pixels inside the region at every phase)", "raw p0")
    plot_curves(args.out / "lesion_curves.png", lac, times, lac_rrmse, styles,
                "Lesion activity curves: partial-volume weighted mean over each frame's own phase mask", "mean raw p0")
    plot_per_frame(args.out / "per_frame_scores.png", times, per_frame, styles)
    save_metrics(args.out / "inverse_eval.json", dict(
        definitions=dict(
            image="metrics.evaluate_sequence_in_blocks on all window frames, as inverse/execute.score_all_frames: "
                  "RRMSE, PSNR and space-time SSIM over (y, x, t), 7 x 7 x 7 uniform windows, sample covariance, "
                  "K1 0.01, K2 0.03, data range 1 (the normalized truth's range); roi = bounding box of positive truth",
            ssim_proxnf_convention="the same space-time SSIM with data range = the reconstruction's max - min over all "
                                   "frames, as applications/proximal_splitting/create_animations.py",
            point_curves="f(row, col, t) at a fixed pixel wholly inside the region (indicator 1) at every respiratory "
                         "phase of the window's frames, deepest inside it (largest smallest-over-phases distance to a "
                         "pixel not wholly inside); RRMSE = ||c - c_truth||_2 / ||c_truth||_2 over all frames",
            lesion_lac="METRICS.md 3.6: LAC(t) = sum_r w(r, phase(t)) f(r, t) / sum_r w(r, phase(t)), w the region's "
                       "partial volume at the frame's own phase; LAC RRMSE = ||LAC - LAC_truth||_2 / ||LAC_truth||_2",
            units="curves in raw p0 (normalized values times intensity_scale_value)"),
        points=points, lesion_regions={"whole lesion": "tumor + tumor_core", "shell": "tumor", "core": "tumor_core"},
        data=data, scored_frames=len(times), roi=[[s.start, s.stop] for s in roi], device=args.device,
        code_commit=git_commit(), labels={name: styles[name][0] for name in reports}, lesion_lac=lac_report,
        runs=reports))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
