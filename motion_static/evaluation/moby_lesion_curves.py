"""Lesion activity curves (LAC) of a dce_moby window: truth, SS(r) and neural-field runs.

As in Lozenski et al. 2024 (Table II), the LAC is the spatial mean of the image over the
lesion at each frame, and the LAC RRMSE compares a method's curve with the truth's:

    LAC(t) = sum_r w(r, phase(t)) f(r, t) / sum_r w(r, phase(t)),
    LAC RRMSE = ||LAC - LAC_truth||_2 / ||LAC_truth||_2 over the scored frames.

w is the region's partial volume in pixel r from the slice file's indicators, taken at
the frame's own respiratory phase, so the region follows the breathing motion. Three
regions: the whole lesion (tumor + tumor_core), its shell (tumor) and its core
(tumor_core); the two indicators are complementary partial volumes. Time 0 is the
injection, so frames with t < 0 give the pre-injection mean.

Embedding protocol only: every window frame is fitted and scored. SS(r) uses the top r
singular vectors of the (pixels x frames) matrix, computed from its frame Gram matrix to
keep memory low; their RRMSE and singular values are checked against the saved SS(r)
results before use. Writes lesion_curves.json (definition, checks, scores),
lesion_curves.npz (curves in raw p0 units, all scored frames) and lesion_curves.png (the
10-90% rise of the whole lesion, the first three breaths and three breaths at the peak).

python -m motion_static.evaluation.moby_lesion_curves --svd SVD_DIR --nf-runs RUN --ranks 17 21 --out NEW
`--data-file` reads a local copy of the slice file, which must have the recorded sha256.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg

from .evaluate import predict
from .metrics import save_metrics
from .moby_embedding_figures import NF_COLOR, SS_COLORS, TRUTH_COLOR, load_model
from .moby_gifs import git_commit
from .plots import pyplot
from ..phantoms import moby
from ..run import prepare_data


TRUTH = "Object"
# The two indicators are complementary partial volumes of the lesion (their sum is at most 1).
REGIONS = {"whole lesion": ("tumor", "tumor_core"), "shell": ("tumor",), "core": ("tumor_core",)}


def top_singular_factors(frames, rank):
    """Top-r factors of the uncentered (pixels x frames) matrix F, largest first.

    Same as the first r columns of moby_svd_baseline.semiseparable_factors up to signs and
    rounding: spatial maps u_i * s_i = F v_i and temporal curves v_i. The v_i are the
    eigenvectors of the frame Gram matrix F^T F, with eigenvalues s_i^2, so F is never
    decomposed in full. Also returns the s_i and ||F||_F^2 for the rank-r residuals.
    """
    count = frames.shape[1]
    gram = np.zeros((count, count))
    for start in range(0, frames.shape[0], 4096):
        block = frames[start:start + 4096].astype(np.float64)
        gram += block.T @ block
    values, temporal = scipy.linalg.eigh(gram, subset_by_index=[count - rank, count - 1])
    values, temporal = np.clip(values[::-1], 0, None), np.ascontiguousarray(temporal[:, ::-1])
    spatial = np.zeros((frames.shape[0], rank))
    for start in range(0, frames.shape[0], 4096):
        spatial[start:start + 4096] = frames[start:start + 4096].astype(np.float64) @ temporal
    return spatial, temporal, np.sqrt(values), float(np.trace(gram))


def lesion_curve(weights, values):
    """Partial-volume weighted mean over pixels: weights (pixels,), values (pixels x frames)."""
    return weights @ values / weights.sum()


def region_weights(path, frames):
    """Per region, (y, x, phase) partial volumes, and each window frame's phase index."""
    with h5py.File(path, "r") as f:
        phases = f["resp_phase"][frames[0]:frames[-1] + 1].astype(int)
        names = {name for region in REGIONS.values() for name in region}
        # The file stores (x, y, phase); the window's truth is (y, x, t).
        indicators = {name: np.transpose(f["indicator"][name][...], (1, 0, 2)).astype(np.float64) for name in names}
    return {region: sum(indicators[name] for name in names) for region, names in REGIONS.items()}, phases


def curves_for(training, weights, phases, spatial, temporal, ranks, models, scale):
    """LACs in raw p0 units, per region: truth, SS(r) for each rank and each neural field."""
    shape = training["truth"].shape
    frames = training["truth"].reshape(-1, shape[-1])
    ix, iy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))  # flat index iy * nx + ix
    x, y = training["x"][ix.ravel()], training["y"][iy.ravel()]
    flat = {region: volume.reshape(-1, volume.shape[-1]) for region, volume in weights.items()}
    curves = {region: {name: np.zeros(shape[-1]) for name in [TRUTH, *[f"SS({r})" for r in ranks], *models]}
              for region in weights}
    for phase in np.unique(phases):
        picks = np.nonzero(phases == phase)[0]
        support = np.nonzero(flat["whole lesion"][:, phase] > 0)[0]  # the shell and core lie inside
        coordinates = np.column_stack([np.repeat(x[support], len(picks)), np.repeat(y[support], len(picks)),
                                       np.tile(training["t"][picks], len(support))])
        fields = {name: predict(model, coordinates).reshape(len(support), len(picks)) for name, model in models.items()}
        truth = frames[support][:, picks].astype(np.float64)
        for region in weights:
            w = flat[region][support, phase]
            curves[region][TRUTH][picks] = lesion_curve(w, truth)
            # SS(r) image at these frames: spatial[:, :r] @ temporal[picks, :r].T, averaged without forming it.
            projected = lesion_curve(w, spatial[support])
            for r in ranks:
                curves[region][f"SS({r})"][picks] = temporal[picks, :r] @ projected[:r]
            for name, field in fields.items():
                curves[region][name][picks] = lesion_curve(w, field)
    return {region: {name: curve * scale for name, curve in entry.items()} for region, entry in curves.items()}


def scores(curves, times):
    """Per region: truth summary and, per method, LAC RRMSE and pre-injection mean (t < 0)."""
    before = times < 0
    report = {}
    for region, entry in curves.items():
        truth = entry[TRUTH]
        report[region] = dict(truth_maximum=float(truth.max()),
                              truth_pre_injection_mean=float(truth[before].mean()) if before.any() else None)
        for name, curve in entry.items():
            if name != TRUTH:
                report[region][name] = dict(
                    lac_rrmse=float(np.linalg.norm(curve - truth) / np.linalg.norm(truth)),
                    pre_injection_mean=float(curve[before].mean()) if before.any() else None)
    return report


def rise_window(curve, times, breath):
    """Frames of the 10-90% rise of a curve's breath means, from the pre-injection mean to the peak.

    Breath means remove the within-breath variation; the window covers whole breaths.
    """
    count = (len(curve) - 1) // breath
    means = curve[:count * breath].reshape(count, breath).mean(axis=1)
    base = curve[times < 0].mean() if np.any(times < 0) else means[0]
    first = int(np.argmax(means >= base + 0.1 * (means.max() - base)))
    last = int(np.argmax(means >= base + 0.9 * (means.max() - base)))
    return first * breath, (last + 1) * breath + 1


def plot(out, curves, times, report, breath):
    """Each region: the 10-90% rise of the whole lesion, the first three breaths and three breaths at the peak.

    The whole series is too long to show the breathing variation; its curves are in the npz.
    """
    plt = pyplot(out)
    span = 3 * breath + 1
    peak = int(np.argmax(curves["whole lesion"][TRUTH]))
    late = max(0, min(len(times) - span, peak - span // 2))
    first, last = rise_window(curves["whole lesion"][TRUTH], times, breath)
    windows = [(slice(first, last), f"10-90% rise of the whole lesion, frames {first}-{min(last, len(times)) - 1}"),
               (slice(0, span), f"frames 0-{min(span, len(times)) - 1}"),
               (slice(late, late + span), f"frames {late}-{min(late + span, len(times)) - 1}, around the peak")]
    methods = [name for name in curves["whole lesion"] if name != TRUTH]
    colors = dict(zip([name for name in methods if name.startswith("SS(")], SS_COLORS))
    fig, axes = plt.subplots(len(curves), 3, figsize=(13, 3.1 * len(curves)), layout="constrained", squeeze=False)
    for row, (region, entry) in enumerate(curves.items()):
        for column, (part, label) in enumerate(windows):
            ax = axes[row, column]
            for name in methods:
                ax.plot(times[part], entry[name][part], color=colors.get(name, NF_COLOR), lw=1.6,
                        label=f"{name}: LAC RRMSE {100 * report[region][name]['lac_rrmse']:.2f}%")
            # The truth goes on top, thin, so it stays visible over the methods.
            ax.plot(times[part], entry[TRUTH][part], color=TRUTH_COLOR, lw=0.9, label=TRUTH)
            ax.grid(alpha=.25)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if row == len(curves) - 1:
                ax.set_xlabel("Time (s)")
            if column == 0:
                ax.set_ylabel(f"{region}: mean raw p0")
                ax.legend(frameon=False, fontsize=8, loc="upper left")  # the rise leaves this corner empty
    fig.suptitle("Lesion activity curves: partial-volume weighted mean over each frame's own phase mask", fontsize=12)
    fig.savefig(out / "lesion_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--svd", type=Path, required=True, help="embedding output of evaluation.moby_svd_baseline")
    parser.add_argument("--nf-runs", type=Path, nargs="+", required=True, help="neural-field runs of the same window")
    parser.add_argument("--nf-labels", nargs="*", default=[], help="labels for --nf-runs (default: NF (ours), then run names)")
    parser.add_argument("--ranks", type=int, nargs="+", required=True, help="SS(r) ranks to compare")
    parser.add_argument("--data-file", type=Path, help="local copy of the slice file, checked by its sha256")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")
    if min(args.ranks) < 1:
        parser.error("Ranks must be positive")

    summary = json.loads((args.svd / "svd_baseline.json").read_text())
    if summary.get("protocol") != "embedding":
        parser.error("Lesion curves are defined here for the embedding protocol only")
    config = json.loads((args.nf_runs[0] / "config.json").read_text())
    for run in args.nf_runs:
        run_config = json.loads((run / "config.json").read_text())
        if run_config["data"]["sha256"] != summary["data"]["sha256"] or \
                run_config["data"]["first_frame"] != summary["data"]["first_frame"]:
            parser.error(f"{run.name} uses a different data window than the SS(r) results")
        if run_config.get("held_out_split") != "none":
            parser.error(f"{run.name} does not use the embedding protocol")
    if args.data_file is not None:
        if moby.file_sha256(args.data_file) != config["data"]["sha256"]:
            parser.error("--data-file does not have the sha256 recorded by the runs")
        config["data"] = dict(config["data"], file=str(args.data_file.resolve()))
    rows = {row["rank"]: row for row in summary["ranks"]}
    if max(args.ranks) > len(rows):
        parser.error(f"The SS(r) results stop at rank {len(rows)}")

    data, _ = prepare_data(config)
    training = data["training"]
    frames = training["truth"].reshape(-1, training["truth"].shape[-1])
    spatial, temporal, sigma, energy = top_singular_factors(frames, max(args.ranks))
    saved = np.asarray(summary["singular_values"][:len(sigma)])
    check = dict(max_relative_difference_singular_values=float(np.max(np.abs(sigma - saved) / saved)))
    for r in args.ranks:
        # Rank-r residual: ||F||^2 minus the r largest s_i^2, since the v_i are orthonormal.
        rebuilt = float(np.sqrt(max(0.0, 1 - np.sum(sigma[:r] ** 2) / energy)))
        check[f"SS({r})"] = dict(rrmse_rebuilt=rebuilt, rrmse_saved=rows[r]["training_rrmse"])
        if not np.isclose(rebuilt, rows[r]["training_rrmse"], rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"SS({r}) does not reproduce the saved SS(r) results: {check[f'SS({r})']}")

    labels = args.nf_labels + [run.name for run in args.nf_runs[len(args.nf_labels):]]
    if not args.nf_labels:
        labels[0] = "NF (ours)"
    models = {label: load_model(run) for label, run in zip(labels, args.nf_runs)}
    weights, phases = region_weights(config["data"]["file"], moby.window_frames(config["data"]))
    scale = config["data"]["intensity_scale_value"]
    curves = curves_for(training, weights, phases, spatial, temporal, args.ranks, models, scale)
    report = scores(curves, training["t"])

    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "lesion_curves.npz", times=training["t"],
                        **{f"{region}|{name}": curve for region, entry in curves.items() for name, curve in entry.items()})
    plot(args.out, curves, training["t"], report, config["data"]["frames_per_cycle"])
    save_metrics(args.out / "lesion_curves.json", dict(
        definition="LAC(t) = sum_r w(r, phase(t)) f(r, t) / sum_r w(r, phase(t)); w = partial volume of the region at "
                   "the frame's own respiratory phase; LAC RRMSE = ||LAC - LAC_truth||_2 / ||LAC_truth||_2",
        regions={region: " + ".join(names) for region, names in REGIONS.items()},
        units="raw p0 (normalized values times intensity_scale_value)", protocol="embedding",
        scored_frames=len(training["t"]), pre_injection="frames with t < 0 (time 0 is the injection)",
        data=dict(config["data"]), svd_results=str(args.svd),
        nf_runs={label: str(run) for label, run in zip(labels, args.nf_runs)},
        ranks=args.ranks, ss_check=check, code_commit=git_commit(), scores=report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
