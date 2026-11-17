"""One training entry point for image and nested fields. Run from the repository root."""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import re
import time

import numpy as np
import torch

from .losses import compute_training_losses
from .models.hash_nf import HashNeuralField
from .models.coordinates import DEFAULT_BOUNDS
from .models.nested_mlp import NestedMLP
from .models.nested_hash_nf import NestedHashNF
from .models.nested_low_rank_hash_nf import NestedLowRankHashNF
from .models.low_rank_intensity import LowRankIntensity
from .models.oracle_branches import PHANTOM_REFERENCE_TIME, TrueMotion, TrueReferenceIntensity
from .models.time_grid import TimeGridField
from .phantoms import moby, moby_source
from .phantoms.data import (DEFAULT_PHANTOM, DEFAULT_SAMPLING, CoordinateGrid, VoxelSubsetGrid, coordinate_rows,
                            make_pair, sample)
from .splits import block_split
from .evaluation.evaluate import (evaluate, evaluate_image_only, roi_from_mask, streamed_embedding_partials,
                                  streamed_embedding_report)
from .evaluation.metrics import save_metrics
from .evaluation.block_evaluate import evaluate_blocks
from .training_monitor import TrainingMonitor, fixed_indices, fixed_mse
from . import distributed
from .distributed import Context, world_size_from_environment
from .streaming import (InMemorySampler, StreamedSampler, StreamedTruth, frame_shards, groups_per_round,
                        scan_shard)
from .run_records import (
    Tee,
    initialize_run_metadata,
    save_checkpoint,
    save_resume_checkpoint,
    source_hashes,
    snapshot_source,
    update_summary,
    utc_now,
    write_json,
)

# Plateau schedule (ReduceLROnPlateau semantics): every log_interval steps the MSE on
# all validation frames is compared with its best value; after more than `patience`
# evaluations without a relative improvement of `threshold`, every rate is multiplied
# by `factor`. The run stops at the plateau after `max_reductions` reductions, or at
# `steps`, and keeps the weights with the lowest validation MSE.
PLATEAU_DEFAULTS = dict(factor=0.5, patience=3, threshold=1e-3, max_reductions=4)
HELD_OUT_SPLITS = ("midpoints", "alternate_validation_test", "none")


def torch_device(config):
    """The configured compute device; batches are still drawn with the CPU generator."""
    return torch.device(config.get("device", "cpu"))


def cpu_state(model):
    """State dict on the CPU, so saved weights load on any machine."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def build_model(config):
    # Older toy checkpoints omit bounds; all new runs save them explicitly.
    bounds = config.get("coordinate_bounds", DEFAULT_BOUNDS)
    reference_time = config.get("reference_time")
    if config["model"] == "hash_nf":
        return HashNeuralField(coordinate_bounds=bounds)
    if config["model"] == "nested_mlp":
        return NestedMLP(
            config.get("intensity_width", 128), config.get("motion_width", 32),
            coordinate_bounds=bounds, reference_time=reference_time,
        )
    if config["model"] == "nested_hash_nf":
        return NestedHashNF(
            width=config["head_width"], coordinate_bounds=bounds,
            reference_time=reference_time, **config["encoding"],
        )
    if config["model"] == "low_rank_hash_nf":
        # No motion module: spatial bases are evaluated at observation coordinates r.
        return LowRankIntensity(
            rank=config["rank"],
            width=config["head_width"],
            temporal_width=config["temporal_width"],
            coordinate_bounds=bounds,
            temporal_encoding=config.get("temporal_encoding"),
            **config["encoding"],
        )
    if config["model"] == "nested_low_rank_hash_nf":
        model = NestedLowRankHashNF(
            rank=config["rank"],
            width=config["head_width"],
            temporal_width=config["temporal_width"],
            coordinate_bounds=bounds,
            reference_time=reference_time,
            motion_encoding=config.get("motion_encoding"),
            temporal_encoding=config.get("temporal_encoding"),
            **config["encoding"],
        )
        return apply_oracle_branch(model, config)
    raise ValueError(f"Unknown model: {config['model']}")


def apply_oracle_branch(model, config):
    """Diagnostic only: replace one learned branch by the phantom's true field.

    Both branches are built first, so the learned branch's initialization and
    all later training batches match the run without an oracle branch.
    """
    branch = config.get("oracle_branch")
    if branch == "motion":
        model.motion = TrueMotion(reference_time=config["reference_time"])
    elif branch == "intensity":
        model.intensity = TrueReferenceIntensity(size=config["size"])
    elif branch is not None:
        raise ValueError(f"Unknown oracle branch: {branch}")
    return model


def build_optimizer(model, config):
    """Give hash tables and MLPs their configured learning rates.

    Keep group names and order stable: checkpoints and loss.csv use them.
    The low-rank temporal MLP is separate from its spatial hash and MLP.
    """
    model_name = config["model"]
    if model_name in ("hash_nf", "low_rank_hash_nf"):
        fields = [("image", model)]
    elif model_name in ("nested_hash_nf", "nested_low_rank_hash_nf"):
        # An oracle branch is fixed ground truth with nothing to optimize.
        oracle = config.get("oracle_branch")
        fields = []
        if oracle != "intensity":
            fields.append(("intensity", model.intensity))
        if oracle != "motion":
            fields.append(("motion", model.motion.field))
    else:
        return torch.optim.Adam([
            {"params": model.parameters(), "lr": config["mlp_lr"], "name": "mlp"},
        ])

    groups = []
    for branch_name, field in fields:
        groups.append({
            "params": field.encoding.parameters(),
            "lr": config["hash_lr"],
            "name": f"{branch_name}_hash",
        })
        groups.append({
            "params": field.mlp.parameters(),
            "lr": config["mlp_lr"],
            "name": f"{branch_name}_mlp",
        })
    learned_low_rank = config.get("oracle_branch") != "intensity"
    if model_name in ("low_rank_hash_nf", "nested_low_rank_hash_nf") and learned_low_rank:
        if model_name == "low_rank_hash_nf":
            temporal = model.temporal
            prefix = "image_time"
        else:
            temporal = model.intensity.temporal
            prefix = "intensity_time"
        # The time MLP may use its own rate; it defaults to the shared MLP rate.
        temporal_mlp_lr = config.get("temporal_mlp_lr", config["mlp_lr"])
        if isinstance(temporal, TimeGridField):
            # A time grid table is a feature table like a hash table.
            groups.append({
                "params": temporal.encoding.parameters(),
                "lr": config["hash_lr"],
                "name": f"{prefix}_grid",
            })
            temporal_parameters = temporal.mlp.parameters()
        else:
            temporal_parameters = temporal.parameters()
        groups.append({
            "params": temporal_parameters,
            "lr": temporal_mlp_lr,
            "name": f"{prefix}_mlp",
        })
    return torch.optim.Adam(groups)


class Segment:
    """When a job hands the run over to the next job (run.py --continue): at the first log
    step after `steps` optimizer steps of this job, or once `minutes` have passed since the
    process started (`started`, time.time()). The final step always completes the run."""

    def __init__(self, steps=None, minutes=None, started=None):
        self.steps = steps
        self.deadline = None if minutes is None else started + 60. * minutes
        self.stopped_at = None

    def due(self, step, first_step):
        by_steps = self.steps is not None and step - first_step + 1 >= self.steps
        return by_steps or (self.deadline is not None and time.time() >= self.deadline)


def train(model, coordinates, intensities, config, *, out=None, validation=None, selection=None,
          sampler=None, context=None, monitor_subset=None, resume=None, segment=None):
    """One step = one random batch and one joint optimizer update.

    Loss definitions live in losses.compute_training_losses. Validation is used
    only by the monitor, after updates, without gradients. Passing out=None skips
    file output. With the plateau schedule, `selection` (all validation frames)
    sets the learning rate, the stop and the kept weights; it never enters the loss.
    Return the logged (step, image MSE, temporal L2, total) history, elapsed
    seconds and the plateau summary (None for cosine); the model is updated in place.

    Streamed training (run.execute_streamed) passes a `sampler` instead of the pool, a
    distributed `context` (weighted gradients summed over processes, see streaming.py)
    and the monitor's precomputed fixed samples; only process 0 writes files or prints.
    `resume` (a resume checkpoint, run_records.save_resume_checkpoint) continues after its
    step with its optimizer and schedule; the model, sampler and random generators are
    restored by the caller. With a `segment`, training stops when it is due, saves a resume
    checkpoint and sets segment.stopped_at; the returned seconds include earlier segments.
    """
    device = torch_device(config)
    model.to(device)
    optimizer = build_optimizer(model, config)
    plateau = config.get("schedule") == "plateau"
    if plateau:
        if selection is None:
            raise ValueError("The plateau schedule needs validation frames")
        settings = config["plateau"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings["factor"], patience=settings["patience"],
            threshold=settings["threshold"], threshold_mode="rel")
        selection_coordinates = torch.as_tensor(selection[0], dtype=torch.float32, device=device)
        selection_values = torch.as_tensor(np.asarray(selection[1]).reshape(-1), dtype=torch.float32, device=device)
        best = dict(mse=float("inf"), step=None, state=None)
        reduction_steps = []
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])
    first_step, elapsed_offset = 1, 0.
    if resume is not None:
        if plateau:
            raise ValueError("Resuming is not supported with the plateau schedule")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        first_step, elapsed_offset = resume["step"] + 1, resume["training_seconds"]
    if segment is not None and not hasattr(sampler, "state"):
        raise ValueError("Segments need a streamed sampler, whose state can be saved")
    stopped_by = "step limit"
    if sampler is None:
        sampler = InMemorySampler(coordinates, intensities, config["batch_size"], device)
    context = context or Context()
    checkpoint_steps = set(config.get("checkpoint_steps", []))
    log_interval = config.get("log_interval", 500)
    history = []
    start = time.perf_counter()
    model.train()

    # Record the initial model on fixed samples, using a separate sampling RNG.
    monitor = None
    if out is not None and context.is_root:
        monitor = TrainingMonitor(out, coordinates, intensities, validation, config, optimizer,
                                  training=monitor_subset, resume=resume is not None)
        if resume is None:
            monitor.record(model, optimizer, 0, time.perf_counter() - start)

    for step in range(first_step, config["steps"] + 1):
        # 1. Sample observed space-time points, with replacement.
        batch_coordinates, batch_targets = sampler.batch(step)

        # 2. Compute named loss terms and the objective before the update.
        losses = compute_training_losses(model, batch_coordinates, batch_targets, config)
        total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")

        # 3. Update all model branches jointly, then set the next step's rates.
        # With several processes: g = sum_p w_p grad(loss_p), the same on every process.
        optimizer.zero_grad(set_to_none=True)
        if context.parallel:
            (context.weight * total_loss).backward()
            context.all_reduce_gradients(model)
        else:
            total_loss.backward()
        optimizer.step()
        selection_mse = None
        if not plateau:
            scheduler.step()
        elif step % log_interval == 0 or step == config["steps"]:
            # The plateau rates follow the validation MSE, evaluated after the update.
            selection_mse = fixed_mse(model, selection_coordinates, selection_values)
            if selection_mse < best["mse"]:
                best.update(mse=selection_mse, step=step, state=copy.deepcopy(model.state_dict()))
            previous_rates = [group["lr"] for group in optimizer.param_groups]
            scheduler.step(selection_mse)
            if any(group["lr"] < rate for group, rate in zip(optimizer.param_groups, previous_rates)):
                reduction_steps.append(step)

        # 4. Log pre-update batch losses and post-update fixed-sample MSE.
        should_log = (
            step == 1
            or step % log_interval == 0
            or step == config["steps"]
            or step in checkpoint_steps
        )
        if should_log:
            scalar_losses = {name: value.item() for name, value in losses.items()}
            if context.parallel:
                scalar_losses = context.weighted_sum(scalar_losses)  # the objective's batch estimate
                context.check_parameters(model)
            history.append((step, scalar_losses["batch_mse"],
                            scalar_losses["temporal_l2"], scalar_losses["total_loss"]))
            monitoring_text = ""
            if monitor is not None:
                logged = dict(scalar_losses)
                if selection_mse is not None:
                    logged["validation_frames_mse"] = selection_mse
                row = monitor.record(
                    model, optimizer, step, time.perf_counter() - start + elapsed_offset,
                    batch_losses=logged,
                )
                monitoring_text = (
                    f", fixed train {row['fixed_training_mse']:.6g}, "
                    f"held-out {row.get('fixed_validation_mse', float('nan')):.6g}"
                )
            if selection_mse is not None:
                monitoring_text += f", validation frames {selection_mse:.6g}"
            if context.is_root:
                print(
                    f"{config['phantom']}/{config['model']} step {step}: "
                    f"MSE {scalar_losses['batch_mse']:.6g}, "
                    f"temporal L2 {scalar_losses['temporal_l2']:.6g}, "
                    f"total {scalar_losses['total_loss']:.6g}{monitoring_text}",
                    flush=True,
                )

        if out is not None and context.is_root and step in checkpoint_steps:
            save_checkpoint(out, model, optimizer, scheduler, config, step)

        # End of this job's segment: save what the next job needs, on process 0.
        if (segment is not None and should_log and step < config["steps"]
                and context.decide(segment.due(step, first_step))):
            states = sampler.state()
            states = context.all_gather(states) if context.parallel else [states]
            elapsed = time.perf_counter() - start + elapsed_offset
            if out is not None and context.is_root:
                save_resume_checkpoint(out, model, optimizer, scheduler, config, step, states, elapsed)
            segment.stopped_at = step
            return history, elapsed, None

        # The reduction after the last allowed one marks the final plateau.
        if plateau and len(reduction_steps) > settings["max_reductions"]:
            stopped_by = "plateau"
            break

    if not plateau:
        return history, time.perf_counter() - start + elapsed_offset, None
    model.load_state_dict(best["state"])
    summary = dict(best_step=best["step"], best_validation_frames_mse=best["mse"], stop_step=step,
                   stopped_by=stopped_by, rate_reduction_steps=reduction_steps[:settings["max_reductions"]])
    print(f"Plateau schedule stopped at step {step} ({stopped_by}); kept step {best['step']} "
          f"with validation-frame MSE {best['mse']:.6g}", flush=True)
    return history, time.perf_counter() - start, summary


def parse_configuration(argv=None):
    """Resolve template + CLI overrides, validate, and choose a new output path.

    This function creates no files or models; --dry-run uses the same settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON template copy; defaults to experiments/template.json")
    parser.add_argument("--phantom", choices=("simple", "complex"), help="Explicitly override the template phantom")
    parser.add_argument("--model", choices=("hash_nf", "nested_mlp", "nested_hash_nf",
                                           "low_rank_hash_nf", "nested_low_rank_hash_nf"))
    parser.add_argument("--split", choices=("midpoints", "blocks"))
    parser.add_argument("--out", type=Path, help="New run directory; default is results/<UTC timestamp>_<name>")
    parser.add_argument("--name", help="Short run label for the automatic directory name")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, help="Total training frames, including endpoints")
    parser.add_argument("--cycles", type=int, help="Complete periods; extend time without speeding up dynamics")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", help="Optional intermediate checkpoints; default: none")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Total space-time rank for either low-rank model")
    parser.add_argument("--temporal-weight", type=float)
    parser.add_argument("--temporal-step", type=float)
    parser.add_argument("--gifs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without creating files or training")
    parser.add_argument("--distributed", action="store_true",
                        help="One of several processes (RANK, WORLD_SIZE, MASTER_ADDR/PORT in the environment); "
                             "source data and --out only")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Start the run in --out, or continue it if it is paused (source data only)")
    parser.add_argument("--segment-minutes", type=float,
                        help="With --continue: pause at the first log step after this many minutes")
    parser.add_argument("--segment-steps", type=int,
                        help="With --continue: pause at the first log step after this many steps")
    args = parser.parse_args(argv)

    # Load the template, then apply explicit command-line overrides.
    root = Path(__file__).resolve().parent
    path = args.config or root / "experiments/template.json"
    if not path.exists():
        parser.error(f"Configuration does not exist: {path}")
    config = json.loads(path.read_text())
    if args.phantom is not None and args.phantom != config.get("phantom"):
        config.update(phantom=args.phantom, **DEFAULT_SAMPLING[args.phantom])
    config.setdefault("phantom", DEFAULT_PHANTOM)
    if args.model is not None:
        config["model"] = args.model
    measured = config["phantom"] == moby.NAME
    if measured:
        # The data file and window fix the grid, frame count and coordinate bounds.
        if args.frames is not None or args.cycles is not None:
            parser.error("dce_moby takes its window from config['data'], not --frames/--cycles")
        # An extracted file (data.file) or the source phase files (data.source_root, streamed).
        resolve = moby_source.resolve_settings if "source_root" in config["data"] else moby.resolve_settings
        try:
            config["data"], window = resolve(config["data"])
        except (KeyError, OSError, ValueError) as error:
            parser.error(f"dce_moby data: {error}")
        config.update(window)
        if config.get("held_out_split") == "none":
            # The embedding protocol observes every window frame, not only the even ones.
            config["frames"] = len(moby.window_frames(config["data"]))
    elif config["phantom"] not in DEFAULT_SAMPLING:
        parser.error("Unknown phantom")
    else:
        for key, value in DEFAULT_SAMPLING[config["phantom"]].items():
            config.setdefault(key, value)
    for key in ("steps", "seed", "frames", "cycles", "checkpoint_steps", "rank",
                "temporal_weight", "temporal_step"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    # Validate the model, sampling budget and logging options.
    config["evaluation_split"] = args.split or config.get("evaluation_split", "midpoints")
    if config["evaluation_split"] not in ("midpoints", "blocks"):
        parser.error("evaluation_split must be midpoints or blocks")
    low_rank = config["model"] in ("low_rank_hash_nf", "nested_low_rank_hash_nf")
    nested = config["model"] in ("nested_mlp", "nested_hash_nf", "nested_low_rank_hash_nf")
    if args.rank is not None and not low_rank:
        parser.error("--rank requires low_rank_hash_nf or nested_low_rank_hash_nf")
    if low_rank and config["rank"] < 1:
        parser.error("rank must be positive")
    for key in ("steps", "batch_size", "threads", "size", "frames", "cycles"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if config["frames"] < 2 or config["size"] < 16:
        parser.error("frames must be at least 2 and size must be at least 16")
    config.setdefault("checkpoint_steps", [])
    if any(step < 1 or step > config["steps"] for step in config["checkpoint_steps"]):
        parser.error("Checkpoint steps must be within the training budget")
    for key, default in (("monitor_seed", 2026), ("monitor_samples", 65536), ("log_interval", 500)):
        config.setdefault(key, default)
    if config["monitor_samples"] < 1 or config["log_interval"] < 1:
        parser.error("monitor_samples and log_interval must be positive")

    # Large visual/array exports are opt-in; numeric losses are always recorded.
    options = {"gifs": False, "predictions": False, **config.get("output", {})}
    if args.gifs is not None:
        options["gifs"] = args.gifs
    if args.save_predictions is not None:
        options["predictions"] = args.save_predictions
    if set(options) != {"gifs", "predictions"} or any(type(value) is not bool for value in options.values()):
        parser.error("output accepts boolean gifs and predictions only")
    config["output"] = options

    # Physical coordinates span complete periods, with the first frame as reference.
    if measured:
        bounds = config["coordinate_bounds"]  # mm and seconds, from the data file
        if config["evaluation_split"] != "midpoints":
            parser.error("dce_moby supports the midpoints split only")
        if options["gifs"]:
            parser.error("GIFs are not implemented for dce_moby; use output.predictions")
    else:
        bounds = [[-.5, .5], [-.5, .5], [-.5, config["cycles"] - .5]]
        if "coordinate_bounds" in config and config["coordinate_bounds"] != bounds:
            parser.error("coordinate_bounds do not match the configured phantom domain")
    config["coordinate_bounds"] = bounds
    if nested and config.get("reference_time") is None:
        config["reference_time"] = bounds[-1][0]  # time is the last coordinate
    if config["model"] == "low_rank_hash_nf":
        config.pop("motion_encoding", None)
        config.pop("reference_time", None)

    # The optional existing temporal penalty uses physical time differences.
    weight = config.get("temporal_weight", 0.)
    if not np.isfinite(weight) or weight < 0:
        parser.error("temporal_weight must be finite and nonnegative")
    if weight > 0 and not nested:
        parser.error("Temporal intensity regularization requires a nested model")
    if args.temporal_step is not None and weight == 0:
        parser.error("--temporal-step requires positive temporal_weight")
    if weight > 0:
        delta = config.get("temporal_step", float(np.diff(np.linspace(*bounds[-1], config["frames"])).min()))
        if not np.isfinite(delta) or not 0 < delta <= config["cycles"]:
            parser.error("temporal_step must be positive and no larger than the time span")
        config.update(temporal_weight=weight, temporal_step=delta,
                      temporal_sampling="detached warped training coordinates; same q at both times",
                      temporal_penalty="mean squared finite-difference time derivative")

    # Optional time-network settings exist only for a learned low-rank intensity.
    time_grid = config.get("temporal_encoding")
    temporal_mlp_lr = config.get("temporal_mlp_lr")
    if time_grid is not None or temporal_mlp_lr is not None:
        if not low_rank or config.get("oracle_branch") == "intensity":
            parser.error("temporal_encoding and temporal_mlp_lr require a learned low-rank intensity")
    if time_grid is not None:
        keys = {"levels", "features", "min_resolution", "max_resolution"}
        if (not isinstance(time_grid, dict) or set(time_grid) != keys
                or any(type(time_grid[key]) is not int or time_grid[key] < 1 for key in keys)
                or time_grid["max_resolution"] < time_grid["min_resolution"]):
            parser.error("temporal_encoding needs positive integer levels, features, min_resolution <= max_resolution")
    if temporal_mlp_lr is not None and not (np.isfinite(temporal_mlp_lr) and temporal_mlp_lr > 0):
        parser.error("temporal_mlp_lr must be finite and positive")

    # The plateau schedule selects on validation frames, so it needs a held-out split
    # that keeps separate test frames; only the DCE-MOBY window provides one so far.
    held_out_split = config.get("held_out_split", "midpoints")
    if held_out_split not in HELD_OUT_SPLITS:
        parser.error(f"held_out_split must be one of {HELD_OUT_SPLITS}")
    if held_out_split != "midpoints" and not measured:
        parser.error(f"held_out_split {held_out_split} is implemented for dce_moby only")
    schedule = config.get("schedule", "cosine")
    if schedule not in ("cosine", "plateau"):
        parser.error("schedule must be cosine or plateau")
    if schedule == "plateau":
        if not measured:
            parser.error("The plateau schedule is implemented for dce_moby only")
        if held_out_split != "alternate_validation_test":
            parser.error("The plateau schedule requires held_out_split alternate_validation_test")
        settings = {**PLATEAU_DEFAULTS, **config.get("plateau", {})}
        if set(settings) != set(PLATEAU_DEFAULTS):
            parser.error(f"plateau accepts only {sorted(PLATEAU_DEFAULTS)}")
        if not (0 < settings["factor"] < 1 and settings["threshold"] >= 0
                and all(type(settings[key]) is int and settings[key] >= 0 for key in ("patience", "max_reductions"))):
            parser.error("plateau needs 0 < factor < 1, threshold >= 0 and nonnegative integer patience, max_reductions")
        config["plateau"] = settings
    elif "plateau" in config:
        parser.error("plateau settings require schedule plateau")

    # CPU by default. GPU runs move the model and data; they are not bitwise reproducible.
    device = config.get("device", "cpu")
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError):
        parser.error(f"Unknown device {device!r}")
    if device_type == "cuda" and not torch.cuda.is_available():
        parser.error("device cuda requested but CUDA is not available")
    if device_type == "mps" and not torch.backends.mps.is_available():
        parser.error("device mps requested but MPS is not available")
    if device_type not in ("cpu", "cuda", "mps"):
        parser.error("device must be cpu, cuda (optionally cuda:N) or mps")
    if device_type != "cpu" and config.get("oracle_branch") is not None:
        parser.error("Oracle branches evaluate the phantom on the CPU; use device cpu")

    # Oracle diagnostics replace one branch by the complex phantom's true field.
    oracle = config.get("oracle_branch")
    if oracle is not None:
        if oracle not in ("motion", "intensity"):
            parser.error("oracle_branch must be motion or intensity")
        if (config["model"] != "nested_low_rank_hash_nf" or config["phantom"] != "complex"
                or config["evaluation_split"] != "midpoints"):
            parser.error("oracle_branch requires nested_low_rank_hash_nf, the complex phantom and midpoints")
        if config["reference_time"] != PHANTOM_REFERENCE_TIME:
            parser.error("oracle_branch requires the phantom's motion reference time -0.5")
        if weight > 0:
            parser.error("oracle_branch does not support the temporal penalty")

    # 3D box data: the nested low-rank model, with the embedding protocol or the held-out
    # midpoints (even window frames observed, odd frames scored only).
    if config.get("spatial_dimensions", 2) == 3:
        if config["model"] != "nested_low_rank_hash_nf" or held_out_split not in ("none", "midpoints"):
            parser.error("3D data support the nested_low_rank_hash_nf model with held_out_split none or midpoints only")
        for name in ("encoding", "motion_encoding"):
            for key in ("min_resolution", "max_resolution"):
                value = config.get(name, {}).get(key)
                expected = 3 if name == "encoding" else 4  # space (x, y, z); space-time (x, y, z, t)
                if isinstance(value, list) and len(value) != expected:
                    parser.error(f"{name}.{key} needs {expected} entries for 3D data")

    # Streamed training reads the source h5 frame by frame, optionally over several processes
    # (streaming.py): contiguous frame shards, optional frame groups per round.
    streamed = measured and "source_root" in config["data"]
    world_size = world_size_from_environment() if args.distributed else 1
    if args.distributed and not streamed:
        parser.error("--distributed requires source data (data.source_root)")
    if args.distributed and args.out is None:
        parser.error("--distributed requires --out, the same on every process")
    if "frame_groups" in config and not streamed:
        parser.error("frame_groups require source data (data.source_root)")
    if "support_mask" in config and not streamed:
        parser.error("support_mask requires source data (data.source_root)")
    if type(config.get("support_mask", False)) is not bool:
        parser.error("support_mask must be true or false")
    if streamed:
        if held_out_split != "none":
            parser.error("Source data are streamed for held_out_split none only")
        if options["predictions"]:
            parser.error("output.predictions is not supported for streamed source data")
        if config["batch_size"] % world_size:
            parser.error(f"batch_size must be divisible by the {world_size} processes")
        try:
            shards = frame_shards(config["frames"], world_size)
        except ValueError as error:
            parser.error(str(error))
        config["data_parallel"] = dict(
            world_size=world_size, batch_per_process=config["batch_size"] // world_size,
            frame_shards=[[int(shard[0]), int(shard[-1]) + 1] for shard in shards],
            shard_weights=[len(shard) / config["frames"] for shard in shards],
            gradient="sum over processes of w_p * grad(batch mean loss of p), w_p = shard frames / frames")
        groups = config.get("frame_groups")
        if groups is not None:
            keys = ("frames_per_group", "steps_per_group")
            if (not isinstance(groups, dict) or set(groups) - {*keys, "groups_per_round", "rounds"}
                    or any(type(groups.get(key)) is not int or groups[key] < 1 for key in keys)):
                parser.error("frame_groups needs positive integers frames_per_group and steps_per_group")
            try:
                count = groups_per_round(shards, groups["frames_per_group"])
            except ValueError as error:
                parser.error(f"frame_groups: {error}")
            if count > 1 and config["steps"] % (count * groups["steps_per_group"]):
                parser.error(f"steps must be a whole number of rounds of {count} groups x "
                             f"{groups['steps_per_group']} steps")
            config["frame_groups"] = dict(
                frames_per_group=groups["frames_per_group"], steps_per_group=groups["steps_per_group"],
                groups_per_round=count, rounds=config["steps"] // (count * groups["steps_per_group"]) if count > 1 else 1)

    # Blocks are a separate protocol, with their own grid divisibility constraint.
    if config["evaluation_split"] == "blocks":
        config.setdefault("block_split", dict(block_shape=[8, 8, 2], test_fraction=.2, seed=0))
        shape = (config["size"], config["size"], config["frames"])
        if any(n % b for n, b in zip(shape, config["block_split"]["block_shape"])):
            parser.error("Block holdout requires each grid size divisible by its block size; choose explicit compatible settings")

    # Segments: a run continued over several jobs (run.py --continue, streamed data only).
    if (args.segment_minutes is not None or args.segment_steps is not None) and not args.continue_run:
        parser.error("--segment-minutes and --segment-steps require --continue")
    if args.segment_minutes is not None and not (np.isfinite(args.segment_minutes) and args.segment_minutes >= 0):
        parser.error("--segment-minutes must be finite and nonnegative")
    if args.segment_steps is not None and args.segment_steps < 1:
        parser.error("--segment-steps must be positive")
    if args.continue_run:
        if not streamed:
            parser.error("--continue requires source data (data.source_root)")
        if args.out is None:
            parser.error("--continue requires --out")
        if args.out.exists():
            return args, continued_configuration(parser, config, args.out), args.out
        return args, config, args.out

    # Never overwrite a run, including an incomplete one.
    name = args.name or f"{config['model']}_seed{config['seed']}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("name must contain only letters, digits, underscore, hyphen or period")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = args.out or root / "results" / f"{stamp}_{name}"
    if out.exists():
        parser.error("Output already exists; choose a new --out")
    return args, config, out


def first_difference(fresh, saved, path=""):
    """The first setting of `fresh` that `saved` lacks or holds differently, or None."""
    if isinstance(fresh, dict):
        if not isinstance(saved, dict):
            return path.rstrip(".") or "<all settings>"
        for key, value in fresh.items():
            if key not in saved:
                return f"{path}{key}"
            difference = first_difference(value, saved[key], f"{path}{key}.")
            if difference is not None:
                return difference
        return None
    return None if fresh == saved else path.rstrip(".")


def continued_configuration(parser, config, out):
    """The saved configuration of the run in `out`, checked against this invocation's settings.

    Every setting resolved from the given template and options (process count and shards
    included) must equal the saved one; the saved run adds what it found at run time (scale,
    source hashes, segments). A paused run also needs the same code as when it started.
    """
    path = out / "config.json"
    if not path.exists():
        parser.error(f"--continue: {out} exists but holds no config.json")
    saved = json.loads(path.read_text())
    status = saved.get("status")
    if status not in ("paused", "complete"):
        parser.error(f"--continue: the run in {out} is {status!r}; only a paused run can continue")
    difference = first_difference(json.loads(json.dumps(config)), saved)
    if difference is not None:
        parser.error(f"--continue: setting {difference!r} differs from the run saved in {out}")
    if status == "paused":
        if saved["code_provenance"]["source_sha256"] != source_hashes():
            parser.error(f"--continue: the code differs from the code that started the run in {out}")
        # The first segment's scan is reused, so the source files must be unchanged (no new hashing).
        with moby_source.open_source(saved["data"]) as source:
            if source_file_stats(source) != saved["data"]["source_files"]:
                parser.error("--continue: the source files changed (size or modification time) since the run started")
    return saved


def prepare_data(config):
    """Use saved settings for training or later inference, with no output files."""
    if config["phantom"] == moby.NAME:
        held_out_split = config.get("held_out_split", "midpoints")
        if held_out_split == "none":
            return moby.make_all_frames(config["data"]), None
        pair = moby.make_pair(config["data"])
        if held_out_split == "alternate_validation_test":
            pair = moby.split_midpoints(pair)
        return pair, None
    if config.get("evaluation_split", "midpoints") == "midpoints":
        pair = make_pair(config["phantom"], config["size"], config["frames"], config["cycles"])
        return pair, None
    times = np.linspace(-.5, config["cycles"] - .5, config["frames"])
    training = sample(config["phantom"], config["size"], times, (-.5, config["cycles"] - .5))
    split = block_split(training["truth"].shape, **config["block_split"])
    return {"training": training}, split


def execute(config, out):
    """Run one explicitly selected configuration and save a flat record."""
    root = Path(__file__).resolve().parent
    initialize_run_metadata(config)
    out.mkdir(parents=True, exist_ok=False)
    with (
        (out / "run.log").open("w") as log,
        redirect_stdout(Tee(sys.stdout, log)),
        redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            # Save provenance before data/model construction so failures are traceable.
            config["code_provenance"] = snapshot_source(root, out)
            write_json(out / "config.json", config)
            depth = f"x{config['depth']}" if "depth" in config else ""
            print(f"Phantom: {config['phantom']} | size: {config['size']}x{config['size']}{depth} | "
                  f"cycles: {config['cycles']} | training frames: {config['frames']} | time: {config['time_bounds']}", flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])

            # Build observed and held-out pools; held-out samples go only to the monitor.
            data, split = prepare_data(config)
            training = data["training"]
            # 3D grids give a CoordinateGrid: the (N, 4) array may not fit; rows are computed per batch.
            coordinates = coordinate_rows(training)
            intensities = training["truth"].reshape(-1)
            if split is None:
                # Test frames, when present, are never shown to training or monitoring.
                # The embedding protocol has no held-out frames to monitor.
                held_out = data.get("validation", data.get("midpoint"))
                validation = None if held_out is None else (
                    coordinate_rows(held_out),
                    held_out["truth"].reshape(-1),
                )
            else:
                config.update(split=split["settings"], held_out_labels_used_for_training=False)
                observed = split["train_mask"].reshape(-1)
                hidden = split["test_mask"].reshape(-1)
                validation = (coordinates[hidden], intensities[hidden])
                coordinates = coordinates[observed]
                intensities = intensities[observed]
                # Preserve the exact evaluation mask; monitoring indices refer to these selected pools.
                np.savez_compressed(
                    out / "split.npz",
                    train_mask=split["train_mask"], test_mask=split["test_mask"],
                    test_blocks=split["test_blocks"],
                    x=training["x"], y=training["y"], t=training["t"],
                )

            # Optimize, then retain the final weights for later inference.
            model = build_model(config)
            write_json(out / "config.json", config)
            selection = validation if config.get("schedule") == "plateau" else None
            _, seconds, stopping = train(model, coordinates, intensities, config, out=out,
                                         validation=validation, selection=selection)
            kept_step = config["steps"]
            if stopping is not None:
                config["early_stopping"] = stopping
                kept_step = stopping["best_step"]
                write_json(out / "config.json", config)
            torch.save(dict(config=config, step=kept_step, state_dict=cpu_state(model)), out / "model.pt")

            # Evaluate the complete configured grid, independently of export options.
            model.eval()
            if config["phantom"] == moby.NAME:
                # No true motion or reference intensity exists for measured data.
                evaluate_image_only(model, data, config, out, seconds)
            elif split is None:
                evaluate(model, data, config, out, seconds)
            else:
                evaluate_blocks(model, training, split, config, out, seconds)
            config.update(status="complete", finished_at=utc_now())
            write_json(out / "config.json", config)
            update_summary(out.parent)
            print(f"Saved results: {out}", flush=True)
        except BaseException as error:
            # Keep partial files and the original error; never label a failed run complete.
            config.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
            write_json(out / "config.json", config)
            traceback.print_exc()
            try:
                update_summary(out.parent)
            except Exception:
                traceback.print_exc()
            raise


def source_file_stats(source):
    """Size and modification time of every source file, checked again by a continued run."""
    return {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in source.paths}


def scan_source(source, window, shard, config, context, out):
    """Before training: hash the source files, scan this process's shard and combine.

    One pass over the shard finds the scale (global maximum; frames outside the window
    count too), the voxels positive in any window frame (ROI, support mask) and the
    monitor's fixed-sample values (the same indices as for the in-memory pool). With the
    support mask the monitor's pool (mask voxels x frames) is known only after this pass,
    so a second pass reads the monitor values. Process 0 saves scan.npz for later segments.
    Returns (positive voxels, raw monitor values, monitor indices).
    """
    data, frame_count = config["data"], len(window)
    voxels = source.shape[0] * source.shape[1] * source.shape[2]
    support = config.get("support_mask", False)
    read_raw = lambda k: source.read_raw(window[k])
    hashes = source.sha256(source.paths[context.rank::context.world_size])
    monitor_rng = np.random.default_rng(config["monitor_seed"])
    monitor_indices = (np.zeros(0, dtype=np.int64) if support else
                       fixed_indices(voxels * frame_count, config["monitor_samples"], monitor_rng))
    maximum, positive, raw_values = scan_shard(read_raw, shard, frame_count, monitor_indices)
    if data["intensity_scale"] == "global_maximum":
        # One fixed scale for the whole series (moby.py): frames outside the window count too.
        outside = np.setdiff1d(np.arange(source.frame_count), window)
        for frame in outside[context.rank::context.world_size]:
            maximum = max(maximum, float(source.read_raw(frame).max()))
    if context.parallel:
        hashes = {name: digest for part in context.all_gather(hashes) for name, digest in part.items()}
        maximum, positive = context.maximum(maximum), context.any(positive)
    if support:
        # The voxels with p0 > 0 in any window frame (streaming.py, "Support mask").
        mask_voxels = np.flatnonzero(positive.reshape(-1))
        monitor_indices = fixed_indices(len(mask_voxels) * frame_count, config["monitor_samples"], monitor_rng)
        _, _, raw_values = scan_shard(read_raw, shard, frame_count, monitor_indices, mask_voxels)
    if context.parallel:
        raw_values = context.sum_array(raw_values)  # each sample comes from one process
    if data["intensity_scale"] == "global_maximum":
        data["intensity_scale_value"] = maximum
        data["intensity_scale_scope"] = "maximum over every source frame of the region (whole series)"
    data["source_sha256"] = dict(sorted(hashes.items()))
    data["source_files"] = source_file_stats(source)
    if context.is_root:
        np.savez_compressed(out / "scan.npz", positive=positive, monitor_raw_values=raw_values,
                            monitor_indices=monitor_indices, axes="positive: y, x, z")
    return positive, raw_values, monitor_indices


def load_scan(out):
    """A continued run's scan: scan.npz of the first segment (continued_configuration has
    checked that every source file still has the size and modification time recorded then)."""
    with np.load(out / "scan.npz") as scan:
        return scan["positive"], scan["monitor_raw_values"], scan["monitor_indices"]


def execute_streamed(config, out, context, resume=False, segment=None):
    """Train and score on source data streamed from h5 (streaming.py), on one or more processes.

    Same record as execute() for the embedding protocol. Before training, scan_source finds
    the scale, the ROI and the monitor's fixed samples; process 0 writes every file. With a
    `segment` (run.py --continue) the job may stop early: it saves a resume checkpoint,
    marks the run paused and leaves the scoring to the job that finishes the run. `resume`
    continues a paused run from its checkpoint and scan.npz.
    """
    root = Path(__file__).resolve().parent
    if not resume:
        initialize_run_metadata(config)
        if context.is_root:
            out.mkdir(parents=True, exist_ok=False)
    config["status"] = "running"
    with ExitStack() as stack:
        if context.is_root:
            log = stack.enter_context((out / "run.log").open("a" if resume else "w"))
            stack.enter_context(redirect_stdout(Tee(sys.stdout, log)))
            stack.enter_context(redirect_stderr(Tee(sys.stderr, log)))
        try:
            if context.is_root:
                if not resume:
                    config["code_provenance"] = snapshot_source(root, out)
                config.setdefault("segments", []).append(
                    dict(job=os.environ.get("SLURM_JOB_ID"), processes=context.world_size, started_at=utc_now()))
                write_json(out / "config.json", config)
                print(f"Phantom: {config['phantom']} (source files, streamed) | box: {config['data']['box_start']} "
                      f"+ {config['data']['box_shape']} | frames: {config['frames']} | processes: "
                      f"{context.world_size} | time: {config['time_bounds']}"
                      + (" | continuing" if resume else ""), flush=True)
            torch.set_num_threads(config["threads"])
            torch.manual_seed(config["seed"])
            data, device = config["data"], torch_device(config)
            window = moby.window_frames(data)
            frame_count = len(window)
            shards = frame_shards(frame_count, context.world_size)
            shard = shards[context.rank]
            context.weight = len(shard) / frame_count  # w_p, this process's share of the points
            with moby_source.open_source(data) as source:
                spatial_shape = (source.shape[1], source.shape[0], source.shape[2])  # (y, x, z)
                voxels = int(np.prod(spatial_shape))
                times = source.times[window]
                if resume:
                    positive, raw_values, monitor_indices = load_scan(out)
                else:
                    positive, raw_values, monitor_indices = scan_source(source, window, shard, config, context, out)
                support = config.get("support_mask", False)
                mask_voxels = np.flatnonzero(positive.reshape(-1)) if support else None
                scale = data["intensity_scale_value"]
                roi = roi_from_mask(positive)
                config["training_data"] = ("streamed from the source h5 files: frame shards per process"
                                           + (", frame groups per round" if "frame_groups" in config else ""))
                config["data_parallel"]["batch_generators"] = (
                    "process 0: the torch default generator after torch.manual_seed(seed), as one process; "
                    "process p > 0: torch.Generator seeded with SeedSequence([seed, p]); "
                    "group partitions: numpy default_rng([seed, p])")
                if support:
                    config["support_mask_details"] = dict(
                        definition="voxels with p0 > 0 in any window frame (union over time)",
                        voxels=int(len(mask_voxels)), fraction=len(mask_voxels) / voxels, file="support_mask.npz",
                        training="batches drawn from the mask voxels' points only",
                        monitor="fixed samples drawn from the mask voxels' points",
                        evaluation="prediction 0 outside the mask, then every metric as without a mask")
                monitor_subset = None
                if context.is_root:
                    if support and not resume:
                        np.savez_compressed(out / "support_mask.npz", mask=positive, axes="y, x, z")
                    grid = (CoordinateGrid(source.x, source.y, times, source.z) if mask_voxels is None else
                            VoxelSubsetGrid(source.x, source.y, times, source.z, mask_voxels))
                    monitor_subset = (grid[monitor_indices],
                                      torch.as_tensor(raw_values / np.float32(scale), dtype=torch.float32),
                                      monitor_indices)
                    write_json(out / "config.json", config)

                # Optimize; every process starts from process 0's initial weights or the checkpoint.
                model = build_model(config)
                checkpoint = None
                if resume:
                    checkpoint = torch.load(out / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
                    model.load_state_dict(checkpoint["state_dict"])
                    model.to(device)
                elif context.parallel:
                    model.to(device)
                    context.broadcast_parameters(model)
                groups = config.get("frame_groups", {})
                generator = None  # process 0 draws from the default generator, as one process does
                if context.rank > 0:
                    generator = torch.Generator().manual_seed(
                        int(np.random.SeedSequence([config["seed"], context.rank]).generate_state(1)[0]))
                sampler = StreamedSampler(
                    lambda k: source.read(window[k], scale), shard, (source.x, source.y, source.z), times,
                    config["batch_size"] // context.world_size, groups.get("groups_per_round", 1),
                    groups.get("steps_per_group"), generator,
                    np.random.default_rng([config["seed"], context.rank]), device, voxels=mask_voxels)
                if resume:
                    sampler.restore(checkpoint["sampler_states"][context.rank])
                    torch.set_rng_state(checkpoint["torch_rng_state"])  # process 0's batch generator
                first_step = 1 if checkpoint is None else checkpoint["step"] + 1
                if context.is_root:
                    config["segments"][-1]["first_step"] = first_step
                _, seconds, _ = train(model, None, None, config, out=out if context.is_root else None,
                                      sampler=sampler, context=context, monitor_subset=monitor_subset,
                                      resume=checkpoint, segment=segment)
                sampler = None  # release the loaded frames before scoring
                if segment is not None and segment.stopped_at is not None:
                    if context.is_root:
                        config["segments"][-1].update(last_step=segment.stopped_at, finished_at=utc_now(),
                                                      stopped_by="segment limit")
                        config.update(status="paused", training_seconds_so_far=seconds)
                        write_json(out / "config.json", config)
                        update_summary(out.parent)
                        print(f"Paused after step {segment.stopped_at}; continue with --continue", flush=True)
                    return
                if context.is_root:
                    torch.save(dict(config=config, step=config["steps"], state_dict=cpu_state(model)),
                               out / "model.pt")

                # Score every window frame: each process its shard, then process 0 combines.
                model.eval()
                truth = StreamedTruth(lambda k: source.read(window[k], scale), spatial_shape, frame_count)
                entry = dict(x=source.x, y=source.y, z=source.z, t=times)
                partial = streamed_embedding_partials(model, entry, truth, (int(shard[0]), int(shard[-1]) + 1), roi,
                                                      voxels=mask_voxels)
                partials = context.all_gather(partial) if context.parallel else [partial]
            if context.is_root:
                report = streamed_embedding_report(model, config, seconds, partials, truth.shape, times, roi)
                save_metrics(out / "metrics.json", report)
                selected = report["image"]["all_window_frames"]
                print(f"All-frame (embedding) image RRMSE={selected['rrmse']:.4%}, SSIM={selected['ssim']}",
                      flush=True)
                config["segments"][-1].update(last_step=config["steps"], finished_at=utc_now(),
                                              stopped_by="step limit")
                config.pop("training_seconds_so_far", None)
                config.update(status="complete", finished_at=utc_now())
                write_json(out / "config.json", config)
                update_summary(out.parent)
                print(f"Saved results: {out}", flush=True)
        except BaseException as error:
            if context.is_root:
                config.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
                write_json(out / "config.json", config)
                traceback.print_exc()
                try:
                    update_summary(out.parent)
                except Exception:
                    traceback.print_exc()
            raise


def main(argv=None):
    started = time.time()  # a segment's time limit counts from here
    args, config, out = parse_configuration(argv)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return config
    if config["phantom"] == moby.NAME and "source_root" in config["data"]:
        if args.continue_run and config.get("status") == "complete":
            print(f"{out}: the run is already complete; nothing to continue", flush=True)
            return
        segment = None
        if args.segment_steps is not None or args.segment_minutes is not None:
            segment = Segment(args.segment_steps, args.segment_minutes, started)
        resume = args.continue_run and config.get("status") == "paused"
        context = distributed.initialize(config.get("device", "cpu")) if args.distributed else Context()
        try:
            context.barrier()  # every process has checked the settings before process 0 creates `out`
            execute_streamed(config, out, context, resume=resume, segment=segment)
        finally:
            distributed.finalize(context)
        return
    execute(config, out)


if __name__ == "__main__":
    main()
