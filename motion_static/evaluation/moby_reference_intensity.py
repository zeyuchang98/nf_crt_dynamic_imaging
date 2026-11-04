"""Reference intensity of dce_moby runs, scored at reference-configuration frames (METRICS.md Section 3.7).

When the true displacement on the scored grid G equals the displacement at the reference
time t0 (within a tolerance), the body is in t0's configuration and the observed frame is the
reference intensity: J(q, t) = I(q, t). The intensity branch J^(q, t), queried at fixed
coordinates without motion, is scored against these frames:

    T_ref = { scored t : max_{r in G} |u(r, p(t)) - u(r, p(t0))| <= tau },  tau = 1e-3 h,

where |.| is the length of the three-component difference and h the smallest grid spacing.
Scores: RRMSE and PSNR pooled over all selected samples, and the mean of per-frame spatial
SSIM, on the full grid and on the run's image ROI (normalized intensity, data range 1).

The core functions take any spatial dimension (space axes first, time or phase last), so
they apply to a 2D slice or a 3D volume. The command scores 2D embedding-protocol runs
against the displacement slice from phantoms/moby_motion_slice.py:

python -m motion_static.evaluation.moby_reference_intensity --displacement MOTION.h5 \
    --nf-runs RUN [RUN ...] --data-file SLICE.h5 --out NEW
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .evaluate import bounding_roi, predict
from .metrics import error_metrics, save_metrics, spatial_ssim
from .moby_embedding_figures import load_model
from .moby_gifs import git_commit
from ..phantoms import moby
from ..phantoms.data import coordinate_grid
from ..run import prepare_data


TOLERANCE_IN_SPACINGS = 1e-3


def configuration_deviation(displacement, reference_phase):
    """max over the grid of |u(r, p) - u(r, p_ref)| for each phase p.

    displacement: (*space, component, phase) array or HDF5 dataset, read one phase at a time.
    """
    reference = np.asarray(displacement[..., reference_phase], np.float64)
    return np.array([np.sqrt(np.sum((np.asarray(displacement[..., p], np.float64) - reference) ** 2, axis=-1)).max()
                     for p in range(displacement.shape[-1])])


def reference_frames(frame_phases, deviation, tolerance):
    """Indices of the frames whose phase is within `tolerance` of t0's configuration."""
    return np.flatnonzero(deviation[np.asarray(frame_phases)] <= tolerance)


def reference_intensity_scores(truth, intensity, roi, *, data_range=1.0, win_size=7):
    """J^ against the observed frames, both (*space, frame), on the full grid and on the ROI crop.

    roi: tuple of spatial slices. SSIM is the mean of per-frame spatial SSIM (None if any
    frame is too small for the window); RRMSE and PSNR pool all selected samples.
    """
    report = {}
    for region, crop in (("full_grid", ()), ("roi", tuple(roi))):
        observed, predicted = truth[crop], intensity[crop]
        scores = error_metrics(observed, predicted, data_range=data_range)
        frames = range(observed.shape[-1])
        ssim = [spatial_ssim(observed[..., k], predicted[..., k], data_range=data_range, win_size=win_size)
                for k in frames]
        scores["ssim_mean_of_frames"] = None if None in ssim else float(np.mean(ssim))
        scores["per_frame_ssim"] = ssim
        scores["per_frame_rrmse"] = [error_metrics(observed[..., k], predicted[..., k], data_range=data_range)["rrmse"]
                                     for k in frames]
        report[region] = scores
    return report


def reference_phase(config, frame_times, frame_phases):
    """Respiratory phase of the window frame at the motion reference time t0."""
    matches = np.flatnonzero(np.isclose(frame_times, config["reference_time"], rtol=0, atol=1e-4))
    if len(matches) != 1:
        raise ValueError(f"No single window frame at the reference time {config['reference_time']}")
    return int(frame_phases[matches[0]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--displacement", type=Path, required=True, help="output of phantoms.moby_motion_slice")
    parser.add_argument("--nf-runs", type=Path, nargs="+", required=True, help="embedding runs of the same window")
    parser.add_argument("--data-file", type=Path, help="local copy of the slice file, checked by its sha256")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")

    config = json.loads((args.nf_runs[0] / "config.json").read_text())
    for run in args.nf_runs:
        run_config = json.loads((run / "config.json").read_text())
        if run_config.get("held_out_split") != "none":
            parser.error(f"{run.name} does not use the embedding protocol")
        if run_config["data"]["sha256"] != config["data"]["sha256"] or \
                run_config["data"]["first_frame"] != config["data"]["first_frame"] or \
                run_config["reference_time"] != config["reference_time"]:
            parser.error(f"{run.name} uses a different data window or reference time")
    if args.data_file is not None:
        if moby.file_sha256(args.data_file) != config["data"]["sha256"]:
            parser.error("--data-file does not have the sha256 recorded by the runs")
        config["data"] = dict(config["data"], file=str(args.data_file.resolve()))

    with h5py.File(args.displacement, "r") as f:
        if f["z_index"][:].tolist() != [config["data"]["z_index"]]:
            parser.error(f"--displacement holds planes {f['z_index'][:].tolist()}, not the runs' z_index")
        spacing = np.asarray(f.attrs["spacing"], float)
        if not np.isclose(spacing[0], config["data"]["spacing_mm"]):
            parser.error("--displacement has another grid spacing than the runs")
        displacement = f["displacement"][:, :, 0]  # (x, y, component, phase) of the slice
        displacement_record = dict(file=str(args.displacement.resolve()), sha256=moby.file_sha256(args.displacement),
                                   source_sha256=f.attrs["source_sha256"], code_commit=f.attrs["code_commit"])
    window = moby.window_frames(config["data"])
    with h5py.File(config["data"]["file"], "r") as f:
        frame_phases = f["resp_phase"][window]
        frame_times = f["time"][window]
    p_ref = reference_phase(config, frame_times, frame_phases)
    tolerance = TOLERANCE_IN_SPACINGS * float(spacing.min())
    deviation = configuration_deviation(displacement, p_ref)
    frames = reference_frames(frame_phases, deviation, tolerance)

    data, _ = prepare_data(config)
    training = data["training"]  # every window frame (embedding protocol), (y, x, t), normalized
    roi = bounding_roi([training["truth"]])
    truth = training["truth"][..., frames]
    shape = truth.shape
    coordinates = coordinate_grid(training["x"], training["y"], training["t"][frames])
    scores = {}
    for run in args.nf_runs:
        intensity = predict(load_model(run).intensity, coordinates).reshape(shape)
        scores[run.name] = reference_intensity_scores(truth, intensity, roi)
        print(f"{run.name}: reference-intensity RRMSE {scores[run.name]['full_grid']['rrmse']:.4%} "
              f"on {len(frames)} frames", flush=True)

    args.out.mkdir(parents=True)
    save_metrics(args.out / "reference_intensity.json", dict(
        definition="T_ref = {scored t : max_r |u(r, p(t)) - u(r, p(t0))| <= tolerance}; at T_ref, J^(q, t) (intensity "
                   "branch at fixed coordinates, no motion) is compared with the observed frame I(q, t)",
        scores_definition="rrmse and psnr_db pooled over all selected samples; ssim_mean_of_frames = mean of "
                          "per-frame spatial SSIM (window 7, uniform, sample covariance, K1 0.01, K2 0.03)",
        units="normalized intensity (p0 / intensity_scale_value), data range 1",
        protocol="embedding", reference_time=config["reference_time"], reference_phase=p_ref,
        tolerance_mm=tolerance, tolerance_in_spacings=TOLERANCE_IN_SPACINGS,
        deviation_mm_per_phase=deviation.tolist(),
        qualifying_phases=np.flatnonzero(deviation <= tolerance).tolist(),
        frames=dict(count=len(frames), window_indices=frames.tolist(), times=training["t"][frames].tolist()),
        roi=[[s.start, s.stop] for s in roi], roi_axes="y, x (rows, columns)",
        data=dict(config["data"]), displacement=displacement_record,
        nf_runs={run.name: str(run) for run in args.nf_runs}, code_commit=git_commit(), scores=scores))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
