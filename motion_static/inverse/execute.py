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

from ..evaluation.evaluate import evaluate_image_only
from ..phantoms import moby
from ..run_records import Tee, initialize_run_metadata, snapshot_source, update_summary, utc_now, write_json
from .crt_operator import file_sha256
from .problem import check_measurements

METHODS = ("direct",)
REGULARIZERS = ("none", "squared_gradient", "tv")
POU_DEFAULTS = dict(partitions=10, width=140, depth=4)  # ProxNF's 2D code
WARM_START_DEFAULTS = dict(frames_per_group=36, lsqr_atol="proxnf", partition_steps=10_000)
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
    if settings.get("method") not in METHODS:
        raise ValueError(f"inverse.method must be one of {METHODS}")
    path = Path(settings.get("measurements", "")).expanduser()
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError(f"inverse.measurements not found: {path}")
    settings["measurements"] = str(path.resolve())
    metadata = json.loads(path.with_suffix(".json").read_text())
    check_measurements(metadata, config["data"])
    if not positive_int(settings.get("frames_per_step")) or settings["frames_per_step"] > config["frames"]:
        raise ValueError("inverse.frames_per_step must be a positive integer, at most the frame count")
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
    elif "warm_start" in settings and settings["warm_start"] is not None:
        raise ValueError("warm_start is implemented for pou_net only")


def execute_inverse(config, out):
    """Reconstruct from the measurement file and save a run record like run.execute."""
    from ..run import build_model, cpu_state, torch_device
    from .direct import train_direct
    from .problem import InverseProblem
    from .warm_start import warm_start_pou

    root = Path(__file__).resolve().parents[1]
    initialize_run_metadata(config)
    config["loss"] = ("1/2 mean(((P_k f - d_k) / sigma)^2) over random whole frames + weight * R(f); "
                      "the truth is used for monitoring and scoring only")
    out.mkdir(parents=True, exist_ok=False)
    with ((out / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)),
          redirect_stderr(Tee(sys.stderr, log))):
        try:
            config["code_provenance"] = snapshot_source(root, out)
            write_json(out / "config.json", config)
            settings = config["inverse"]
            print(f"Inverse ({settings['method']}) | model {config['model']} | frames {config['frames']} | "
                  f"{settings['frames_per_step']} frames per step | regularizer {settings['regularizer']}", flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])
            device = torch_device(config)
            problem = InverseProblem(config, device)
            settings["problem"] = dict(frames=len(problem), rows_per_frame=problem.operator.rows_per_frame,
                                       distinct_view_sets=len(problem.operator.forward_blocks), sigma=problem.sigma,
                                       measurements_sha256=problem.metadata["npz_sha256"],
                                       operator_sha256=problem.metadata["operator"]["operator_sha256"])
            model = build_model(config).to(device)
            if config["model"] == "pou_net" and settings.get("warm_start") is not None:
                settings["warm_start_result"] = warm_start_pou(model, problem, settings["warm_start"])
                print(f"Warm start: {settings['warm_start_result']['groups']}", flush=True)
            write_json(out / "config.json", config)

            _, seconds = train_direct(model, problem, config, out)
            torch.save(dict(config=config, step=config["steps"], state_dict=cpu_state(model)), out / "model.pt")

            model.eval()
            settings["final_all_frames"] = problem.evaluate(model, np.arange(len(problem)))
            print(f"Final all-frame data fidelity {settings['final_all_frames']['data_fidelity']:.6g}", flush=True)
            evaluate_image_only(model, moby.make_all_frames(config["data"]), config, out, seconds)
            config.update(status="complete", finished_at=utc_now())
            write_json(out / "config.json", config)
            update_summary(out.parent)
            print(f"Saved results: {out}", flush=True)
        except BaseException as error:
            config.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
            write_json(out / "config.json", config)
            traceback.print_exc()
            raise
