"""Settings check and run record of a reconstruction from simulated CRT measurements.

run.py dispatches here when the configuration has an "inverse" entry. Protocol E1 (agreed
2026-11-18): every window frame has data, and every frame is scored against the truth
after training; the truth is never used for optimization, scheduling or selection.
"""
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from ..evaluation.evaluate import _image_report, bounding_roi, image_splits
from ..evaluation.metrics import evaluate_sequence_in_blocks, save_metrics
from ..phantoms import moby
from ..run_records import Tee, initialize_run_metadata, snapshot_source, update_summary, utc_now, write_json
from .crt_operator import file_sha256
from .problem import check_measurements

METHODS = ("direct", "proximal")
REGULARIZERS = ("none", "squared_gradient", "tv")
POU_DEFAULTS = dict(partitions=10, width=140, depth=4)  # ProxNF's 2D code
# LSQR stop chosen by the user 2026-11-18 ("proxnf" stays selectable).
WARM_START_DEFAULTS = dict(frames_per_group=36, lsqr_atol="discrepancy", partition_steps=10_000)
CG_DEFAULTS = dict(max_iterations=100, tolerance=1e-6)  # ProxNF's coefficient_update
REPOSITORY = Path(__file__).resolve().parents[2]


def positive_int(value):
    return type(value) is int and value >= 1


def validate_inverse(config):
    """Check and complete config["inverse"] (and config["pou"]) in place; raise ValueError."""
    settings = config["inverse"]
    if config["phantom"] != moby.NAME or "file" not in config["data"] or config.get("spatial_dimensions", 2) != 2:
        raise ValueError("inverse runs need a 2D dce_moby slice file (data.file)")
    if config.get("held_out_split") != "none":
        raise ValueError("inverse runs use protocol E1: held_out_split none")
    if config.get("schedule", "cosine") != "cosine":
        raise ValueError("inverse runs use the cosine schedule (the plateau schedule would select on the truth)")
    if config.get("temporal_weight", 0.) or config.get("oracle_branch") is not None:
        raise ValueError("inverse runs do not support temporal_weight or oracle_branch")
    if config.get("output", {}).get("predictions", False):
        raise ValueError("output.predictions is not supported for inverse runs (scored frame block by block)")
    if settings.get("method") not in METHODS:
        raise ValueError(f"inverse.method must be one of {METHODS}")
    path = Path(settings.get("measurements", "")).expanduser()
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError(f"inverse.measurements not found: {path}")
    settings["measurements"] = str(path.resolve())
    metadata = json.loads(path.with_suffix(".json").read_text())
    check_measurements(metadata, config["data"])
    if settings["method"] == "direct":
        if not positive_int(settings.get("frames_per_step")) or settings["frames_per_step"] > config["frames"]:
            raise ValueError("inverse.frames_per_step must be a positive integer, at most the frame count")
    else:
        if not (positive_int(settings.get("outer_iterations")) and positive_int(settings.get("inner_steps"))):
            raise ValueError("proximal runs need positive integers inverse.outer_iterations and inner_steps")
        # `steps` counts the inner optimizer steps of all outer iterations (run summary).
        if config["steps"] != settings["outer_iterations"] * settings["inner_steps"]:
            raise ValueError("proximal runs need steps = outer_iterations * inner_steps")
    regularizer = settings.setdefault("regularizer", dict(type="none", weight=0.))
    if regularizer.get("type") not in REGULARIZERS:
        raise ValueError(f"inverse.regularizer.type must be one of {REGULARIZERS}")
    weight = regularizer.setdefault("weight", 0.)
    if not (np.isfinite(weight) and weight >= 0) or (regularizer["type"] == "none" and weight):
        raise ValueError("inverse.regularizer.weight must be finite, nonnegative and 0 without a regularizer")
    if regularizer["type"] == "tv" and not regularizer.get("epsilon", 0) > 0:
        raise ValueError("the tv regularizer needs a positive epsilon")
    count = settings.get("monitor_frames")
    if count is not None and not positive_int(count):
        raise ValueError("inverse.monitor_frames must be a positive integer or null")

    if config["model"] == "pou_net":
        pou = {**POU_DEFAULTS, **config.get("pou", {})}
        if not all(positive_int(pou[key]) for key in POU_DEFAULTS) or pou["partitions"] < 2:
            raise ValueError("pou needs positive integers width, depth and partitions >= 2")
        # One time unit per sensor revolution, as in ProxNF's code.
        t0, t1 = config["coordinate_bounds"][-1]
        frame_interval = (t1 - t0) / (config["frames"] - 1)
        pou["time_unit_s"] = 360. / metadata["schedule"]["rotation_deg"] * frame_interval
        pou["source_sha256"] = file_sha256(REPOSITORY / "nfCRT" / "pou.py")
        config["pou"] = pou
        for key in ("pou_lr", "coefficient_lr"):
            if not (isinstance(config.get(key), (int, float)) and np.isfinite(config[key]) and config[key] > 0):
                raise ValueError(f"pou_net needs a positive {key}")
        warm = settings.get("warm_start", WARM_START_DEFAULTS)
        if warm is not None:
            warm = {**WARM_START_DEFAULTS, **warm}
            warm["cover"] = 2 if metadata["schedule"]["opposite_groups"] else 1
            if not positive_int(warm["frames_per_group"]) or warm["frames_per_group"] > config["frames"]:
                raise ValueError("warm_start.frames_per_group must be a positive integer, at most the frame count")
            if not (warm["lsqr_atol"] in ("proxnf", "discrepancy") or isinstance(warm["lsqr_atol"], (int, float))):
                raise ValueError('warm_start.lsqr_atol must be "proxnf", "discrepancy" or a number')
        settings["warm_start"] = warm
    elif settings.get("warm_start") is not None:
        # Other models: no default; the group LSQR images are fitted for `fit_steps` (warm_start_field).
        warm = {**WARM_START_DEFAULTS, **settings["warm_start"]}
        warm.pop("partition_steps")
        warm["cover"] = 2 if metadata["schedule"]["opposite_groups"] else 1
        if not positive_int(warm["frames_per_group"]) or warm["frames_per_group"] > config["frames"]:
            raise ValueError("warm_start.frames_per_group must be a positive integer, at most the frame count")
        if not positive_int(warm.get("fit_steps")):
            raise ValueError("warm_start for this model needs a positive integer fit_steps")
        if not (warm["lsqr_atol"] in ("proxnf", "discrepancy") or isinstance(warm["lsqr_atol"], (int, float))):
            raise ValueError('warm_start.lsqr_atol must be "proxnf", "discrepancy" or a number')
        settings["warm_start"] = warm

    # ProxNF's exact coefficient update (POUnet, proximal only; the penalty must be quadratic).
    if config["model"] == "pou_net" and settings["method"] == "proximal":
        cg = settings.get("coefficient_cg", CG_DEFAULTS)
        if cg is not None:
            cg = {**CG_DEFAULTS, **cg}
            if not positive_int(cg["max_iterations"]) or not cg["tolerance"] > 0:
                raise ValueError("coefficient_cg needs a positive integer max_iterations and a positive tolerance")
            if regularizer["type"] == "tv":
                raise ValueError("coefficient_cg needs a quadratic penalty (none or squared_gradient)")
        settings["coefficient_cg"] = cg
    elif settings.get("coefficient_cg") is not None:
        raise ValueError("coefficient_cg is implemented for pou_net with method proximal only")


@torch.no_grad()
def score_all_frames(model, problem, chunk=8):
    """Image metrics of every frame and the all-frame data fidelity from one rendering pass.

    The image report is the one streamed_image_report gives (metrics.evaluate_sequence_in_blocks
    with the same ROI and label, equal to the whole-array evaluation up to rounding); the
    blocks' renderings also give mean(((P f - d) / sigma)^2), each frame counted once
    although SSIM halos render a few frames twice.
    """
    was_training = model.training
    model.eval()
    window = problem.window_entry()
    (split, label), = image_splits(window)
    entry = window[split]
    counted = np.zeros(len(problem), dtype=bool)
    misfit = [0.]

    def predict_frames(frames):
        part = np.arange(frames.start, frames.stop)
        pieces = []
        for start in range(0, len(part), chunk):
            positions = part[start:start + chunk]
            images = problem.render(model, positions)
            new = ~counted[positions]
            if new.any():
                index = positions[new]
                residual = (problem.predicted_data(images[torch.as_tensor(new, device=problem.device)], index)
                            - problem.data[torch.as_tensor(index, device=problem.device)])
                misfit[0] += (residual / problem.sigma).cpu().double().square().sum().item()
                counted[index] = True
            pieces.append(images.cpu().numpy())
        return np.moveaxis(np.concatenate(pieces).reshape(len(part), problem.size, problem.size), 0, -1)

    try:
        image = {label: evaluate_sequence_in_blocks(entry["truth"], predict_frames, data_range=1, times=entry["t"],
                                                    roi=bounding_roi([entry["truth"]]))}
    finally:
        model.train(was_training)
    if not counted.all():
        raise RuntimeError("Some frames were not scored")
    data_fidelity = misfit[0] / (len(problem) * problem.operator.rows_per_frame)
    return image, dict(data_fidelity=data_fidelity, truth_rrmse=image[label]["rrmse"])


def execute_inverse(config, out):
    """Reconstruct from the measurement file and save a run record like run.execute."""
    from ..run import build_model, cpu_state, torch_device
    from .direct import train_direct
    from .problem import InverseProblem
    from .proximal import train_proximal
    from .warm_start import warm_start_field, warm_start_pou

    root = Path(__file__).resolve().parents[1]
    initialize_run_metadata(config)
    proximal = config["inverse"]["method"] == "proximal"
    config["loss"] = ("J = 1/2 mean(((P_k f - d_k) / sigma)^2) + weight * R(f), "
                      + ("by proximal splitting (inverse/proximal.py)" if proximal
                         else "on random whole frames by direct backpropagation")
                      + "; the truth is used for monitoring and scoring only")
    out.mkdir(parents=True, exist_ok=False)
    with ((out / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)),
          redirect_stderr(Tee(sys.stderr, log))):
        try:
            config["code_provenance"] = snapshot_source(root, out)
            write_json(out / "config.json", config)
            settings = config["inverse"]
            budget = (f"{settings['outer_iterations']} outer x {settings['inner_steps']} inner steps" if proximal
                      else f"{settings['frames_per_step']} frames per step")
            print(f"Inverse ({settings['method']}) | model {config['model']} | frames {config['frames']} | "
                  f"{budget} | regularizer {settings['regularizer']}", flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])
            device = torch_device(config)
            problem = InverseProblem(config, device, adjoint=proximal)
            settings["problem"] = dict(frames=len(problem), rows_per_frame=problem.operator.rows_per_frame,
                                       distinct_view_sets=len(problem.operator.forward_blocks), sigma=problem.sigma,
                                       measurements_sha256=problem.metadata["npz_sha256"],
                                       operator_sha256=problem.metadata["operator"]["operator_sha256"])
            model = build_model(config).to(device)
            if settings.get("warm_start") is not None:
                start = (warm_start_pou(model, problem, settings["warm_start"]) if config["model"] == "pou_net"
                         else warm_start_field(model, problem, settings["warm_start"], config))
                settings["warm_start_result"] = start
                print(f"Warm start: {len(start['groups'])} groups, LSQR iterations "
                      f"{[group['iterations'] for group in start['groups']]}", flush=True)
            write_json(out / "config.json", config)

            _, seconds = (train_proximal if proximal else train_direct)(model, problem, config, out)
            torch.save(dict(config=config, step=config["steps"], state_dict=cpu_state(model)), out / "model.pt")

            # Every frame scored frame block by frame block on the truth already loaded, with the
            # final data fidelity from the same rendering pass.
            model.eval()
            from .proximal import release_device_memory
            release_device_memory(device)
            image, settings["final_all_frames"] = score_all_frames(model, problem)
            print(f"Final all-frame data fidelity {settings['final_all_frames']['data_fidelity']:.6g}", flush=True)
            window = problem.window_entry()
            report = _image_report(model, config, seconds, image, image_splits(window))
            save_metrics(out / "metrics.json", report)
            print(f"All-frame image RRMSE={report['image']['all_window_frames']['rrmse']:.4%}, "
                  f"SSIM={report['image']['all_window_frames']['ssim']}", flush=True)
            config.update(status="complete", finished_at=utc_now())
            write_json(out / "config.json", config)
            update_summary(out.parent)
            print(f"Saved results: {out}", flush=True)
        except BaseException as error:
            config.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
            write_json(out / "config.json", config)
            traceback.print_exc()
            raise
