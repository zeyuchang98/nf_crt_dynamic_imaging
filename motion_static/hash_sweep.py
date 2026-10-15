"""The approved 36-run hash-grid sweep; existing training mathematics is unchanged."""
import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "motion_static/experiments/complex/hash_sweep_24k"
RESULT_DIR = ROOT / "motion_static/results/complex/hash_sweep_24k"
BASELINE = ROOT / "motion_static/experiments/complex/low_rank_intensity_max64.json"
PROGRESS = ROOT / "motion_static/PROGRESS.md"
START = "<!-- hash-sweep-24k:start -->"
END = "<!-- hash-sweep-24k:end -->"
LEVELS = (4, 8, 12)
MOTION_MAX = (16, 32)
INTENSITY_MAX = (32, 64)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_configs(smoke=False):
    base = json.loads(BASELINE.read_text())
    base.update(steps=24000, checkpoint_steps=[8000, 16000, 24000],
                coordinate_bounds=[[-.5, .5], [-.5, .5], [-.5, 2.5]], reference_time=-.5)
    base.pop("motion_grid_note", None)
    result = {}
    for ml, mr, il, ir in itertools.product(LEVELS, MOTION_MAX, LEVELS, INTENSITY_MAX):
        config = copy.deepcopy(base)
        config["encoding"].update(levels=il, max_resolution=ir)
        # Motion inherits encoding keys unless explicitly overridden.
        config["motion_encoding"].update(levels=ml, max_resolution=[mr, mr, 128])
        if smoke:
            config.update(steps=3, checkpoint_steps=[3], frames=25, monitor_samples=256)
        result[f"mL{ml}_max{mr}_iL{il}_max{ir}"] = config
    return result


def validate_config(config, expected):
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unapproved configuration difference: {key}")
    # Saved configs may contain training provenance, but no new objective/split.
    for key, value in (("temporal_weight", 0.), ("loss", "MSE(predicted image, observed image)"),
                       ("scheduler", "cosine to zero"), ("device", "cpu")):
        if config.get(key, value) != value:
            raise ValueError(f"Unapproved training convention: {key}")
    if "split" in config or any(config.get(k, False) for k in (
            "true_motion_used_for_training", "reference_intensity_labels_used_for_training",
            "midpoint_labels_used_for_training")):
        raise ValueError("Sweep requires image-only supervision and midpoint evaluation")


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "motion_static").rglob("*.py"))
            if p.relative_to(ROOT / "motion_static").parts[0] not in ("archive", "results")}


def checked(command, **kwargs):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, **kwargs)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stderr}")
    return result.stdout.strip()


def synchronize(manifest=None):
    if checked(["git", "branch", "--show-current"]) != "motion-static":
        raise RuntimeError("The sweep is pinned to motion-static; stop after a branch change")
    checked(["git", "pull", "--ff-only"], timeout=120)
    if manifest is not None:
        if source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("Source changed since sweep preparation; stop before mixing implementations")


def prepare(config_dir, smoke=False):
    config_dir.mkdir(parents=True, exist_ok=False)
    configs = approved_configs(smoke)
    if smoke:
        names = ("mL4_max16_iL12_max64", "mL12_max32_iL4_max32")
        configs = {name: configs[name] for name in names}
    for name, config in configs.items():
        write_json(config_dir / f"{name}.json", config)
    manifest = dict(smoke_only=smoke, run_ids=list(configs),
                    config_sha256={name: sha256(config_dir / f"{name}.json") for name in configs},
                    source_sha256=source_hashes(),
                    comparison="Three midpoint errors, per-metric minima and nondominated configurations; no weighted score",
                    temporal_note="Motion temporal endpoints remain 8 and 128; intermediate scales vary with level count",
                    rng_note="Same seed; changed parameter shapes change RNG consumption and sampled batches")
    write_json(config_dir / "manifest.json", manifest)
    print(f"Prepared {len(configs)} {'smoke' if smoke else 'research'} configurations: {config_dir}")


def load_manifest(config_dir):
    manifest = json.loads((config_dir / "manifest.json").read_text())
    expected = approved_configs(manifest["smoke_only"])
    names = manifest["run_ids"]
    if len(names) != len(set(names)) or any(name not in expected for name in names):
        raise ValueError("Invalid or duplicate sweep configurations")
    if not manifest["smoke_only"] and set(names) != set(expected):
        raise ValueError("Research sweep must contain all 36 approved configurations")
    for name in names:
        path = config_dir / f"{name}.json"
        if sha256(path) != manifest["config_sha256"][name]:
            raise ValueError(f"Configuration modified after preparation: {name}")
        validate_config(json.loads(path.read_text()), expected[name])
    return manifest


def preflight(config_dir, out):
    import numpy as np
    import torch
    from .run import build_model
    manifest = load_manifest(config_dir)
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("Prepared source hashes do not match current implementation")
    torch.set_num_threads(4)
    results = {}
    for name in manifest["run_ids"]:
        config = json.loads((config_dir / f"{name}.json").read_text())
        torch.manual_seed(0)
        model = build_model(config).double()
        motion = model.motion.field.encoding.resolutions
        intensity = model.intensity.encoding.resolutions
        assert motion.shape == (config["motion_encoding"]["levels"], 3)
        assert intensity.shape == (config["encoding"]["levels"],)
        assert motion[0].tolist() == [2, 2, 8]
        assert motion[-1].tolist() == config["motion_encoding"]["max_resolution"]
        assert intensity[0] == 8 and intensity[-1] == config["encoding"]["max_resolution"]
        # An initialized model has very small hash features. Excite them only in
        # this disposable probe so finite-difference checks remain informative.
        with torch.no_grad():
            model.intensity.encoding.table.uniform_(-.1, .1)
        points = torch.tensor([[.113, -.217, .191], [-.613, .713, 1.219]], dtype=torch.float64,
                              requires_grad=True)
        assert torch.autograd.gradcheck(model.intensity, (points,), eps=1e-6, atol=1e-7)
        gradient = torch.autograd.grad(model(points).sum(), points)[0]
        assert torch.isfinite(gradient).all() and gradient[:, :2].abs().max() > 0
        reference = points.detach().clone()
        reference[:, 2] = -.5
        assert model.motion(reference).abs().max() == 0
        xy = torch.rand(23, 2, dtype=torch.float64) - .5
        times = torch.linspace(-.5, 2.5, 19, dtype=torch.float64)
        queries = torch.cat((xy[:, None].expand(-1, 19, -1),
                             times[None, :, None].expand(23, -1, -1)), dim=-1)
        with torch.no_grad():
            singular = torch.linalg.svdvals(model.intensity(queries.reshape(-1, 3)).reshape(23, 19))
        tail = (torch.linalg.vector_norm(singular[2:]) / torch.linalg.vector_norm(singular)).item()
        assert np.isfinite(tail) and tail < 1e-12
        results[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                             motion_resolutions=motion.tolist(), intensity_resolutions=intensity.tolist(),
                             intensity_rank_tail=tail, reference_motion_max=0., coordinate_gradcheck=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(python=platform.python_version(), torch=str(torch.__version__),
                        numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                        mps_available=torch.backends.mps.is_available(), device="cpu", threads=4,
                        status="passed", configurations=results))
    print(f"WITNESS: {len(results)} configurations passed endpoints, rank, anchor and coordinate gradients")


def update_progress(completed, total, status):
    content = PROGRESS.read_text()
    section = f"""{START}
## Hash Grid Sweep — 24k Steps, 36 Configurations (2026-10-14)

**Approved scope:** independently combine motion levels 4/8/12 and spatial max 16/32 with intensity levels 4/8/12 and max 32/64. Motion minima stay [2,2,8], temporal max stays 128, and intensity min stays 8. Intermediate temporal scales change with motion level count. All 36 runs start from seed 0 with a fresh 24,000-step cosine schedule; previous 32k checkpoints are excluded. Rank 2, the 64x64 three-cycle/379-frame phantom, image MSE, reference anchor, batch 4,096, four CPU threads, learning rates and evaluation conventions are unchanged.

**Status:** {status}. {completed}/{total} runs have passed saved-result verification. The serial queue preserves per-run configurations, source snapshots, checkpoints, full metrics, image/intensity/motion GIFs, numeric loss and learning-rate histories, and loss curves. Checks cover branch layer independence, grid endpoints, rank, reference anchor, gradients, reloaded predictions, fixed losses and metric recomputation. Completed runs are published in separate milestones; partial or failed output is preserved and never silently retrained.

[Configurations and commands](experiments/complex/hash_sweep_24k/README.md) · [Result table and artifact links](results/complex/hash_sweep_24k/README.md) · [Preflight checks](results/complex/hash_sweep_24k/preflight.json).

**Interpretation:** compare the three original midpoint errors separately, report each minimum and the nondominated tradeoffs, and inspect shared-time/shared-scale GIFs and loss curves. No weighted score is introduced. Parameter shapes, initialization and sampled batches can differ despite the same seed. This is single-seed toy hyperparameter exploration; choosing using midpoint errors does not provide an independent test of generalization or real PACT reconstruction.
{END}"""
    if START in content:
        before, rest = content.split(START, 1)
        _, after = rest.split(END, 1)
        content = before + section + after
    else:
        content = content.replace("## 9. Subsequent Discussion", section + "\n\n## 9. Subsequent Discussion", 1)
    content = content.replace("Last updated: 2026-10-13.", "Last updated: 2026-10-14.")
    PROGRESS.write_text(content)


def artifact_files(directory):
    return [p for p in sorted(directory.rglob("*")) if p.is_file()
            and not any(part.startswith(".") or part == "__pycache__"
                        for part in p.relative_to(directory).parts)
            and p.name not in ("queue_state.json", "queue.log", "launch.json")]


def publish(group, run_name, completed, total):
    """Publish only this sweep's stable artifacts, preserving unrelated work."""
    if checked(["git", "branch", "--show-current"]) != "motion-static":
        raise RuntimeError("Stop publication after a branch change")
    if checked(["git", "diff", "--cached", "--name-only"]):
        raise RuntimeError("Pre-existing staged changes: publication requires an empty index")
    status = "complete" if completed == total else "running"
    update_progress(completed, total, status)
    paths = artifact_files(group) + [PROGRESS]
    if any(p.stat().st_size >= 100 * 1024**2 for p in paths):
        raise RuntimeError("Artifact exceeds GitHub's per-file limit; preserve files and stop publication")
    checked(["git", "add", "--", *[str(p.relative_to(ROOT)) for p in paths]])
    staged = set(checked(["git", "diff", "--cached", "--name-only"]).splitlines())
    if not staged.issubset({str(p.relative_to(ROOT)) for p in paths}):
        raise RuntimeError("Unexpected staged paths; stop without changing the index")
    if staged:
        checked(["git", "commit", "-m", f"Record hash sweep {completed}/{total}: {run_name}"], timeout=120)
    checked(["git", "push"], timeout=600)
    head = checked(["git", "rev-parse", "HEAD"])
    remote = checked(["git", "config", "branch.motion-static.remote"])
    ref = checked(["git", "config", "branch.motion-static.merge"])
    if checked(["git", "ls-remote", remote, ref], timeout=120).split()[0] != head:
        raise RuntimeError("Remote HEAD does not match the published milestone")
    # Verify actual GitHub paths, not just a locally updated tracking ref.
    for relative in (group.relative_to(ROOT) / "README.md",
                     group.relative_to(ROOT) / "runs" / run_name / "summary.json"):
        actual = checked(["gh", "api", f"repos/zeyuchang98/nf_crt_dynamic_imaging/contents/{relative}?ref={head}",
                          "--jq", ".sha"], timeout=120)
        if actual != checked(["git", "rev-parse", f"HEAD:{relative}"]):
            raise RuntimeError(f"GitHub artifact verification failed: {relative}")
    print(f"Published and verified {head}", flush=True)


def run_queue(config_dir, group, publication=False):
    manifest = load_manifest(config_dir)
    if publication and (manifest["smoke_only"] or group.resolve() != RESULT_DIR):
        raise ValueError("Publication is scoped to the approved research result directory")
    if publication and checked(["git", "branch", "--show-current"]) != "motion-static":
        raise RuntimeError("Publication requires the approved motion-static branch")
    group.mkdir(parents=True, exist_ok=True)
    lock = (group / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (group / "logs").mkdir(exist_ok=True)
    (group / "runs").mkdir(exist_ok=True)
    state = dict(status="starting", pid=os.getpid(), total=len(manifest["run_ids"]), completed=[],
                 config_dir=str(config_dir.resolve()), publication_enabled=publication)

    def save_state():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(group / "queue_state.json", state)

    save_state()
    try:
        if not (group / "preflight.json").exists():
            preflight(config_dir, group / "preflight.json")
        if shutil.disk_usage(group).free < (1 if manifest["smoke_only"] else 10) * 1024**3:
            raise RuntimeError("Insufficient free disk space for preserved sweep artifacts")
        for name in manifest["run_ids"]:
            state.update(status="synchronizing", current_run=name)
            save_state()
            synchronize(manifest)
            # Revalidate configuration files after pulling, before every run.
            load_manifest(config_dir)
            run = group / "runs" / name
            expected = json.loads((config_dir / f"{name}.json").read_text())
            from .evaluation.hash_sweep import validate_result, summarize
            if run.exists():
                if not (run / "verification.json").exists():
                    raise RuntimeError(f"Partial run preserved at {run}; no automatic overwrite or restart")
                validate_result(run, expected, manifest["source_sha256"], check_only=True)
            else:
                state.update(status="training", training_log=str(group / "logs" / f"{name}.log"))
                save_state()
                command = [sys.executable, "-u", "-m", "motion_static.run", "--phantom", "complex",
                           "--model", "nested_low_rank_hash_nf", "--config", str(config_dir / f"{name}.json"),
                           "--out", str(run)]
                print(f"Starting {name}: {expected['steps']} steps", flush=True)
                with (group / "logs" / f"{name}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                state["status"] = "validating"
                save_state()
                # Isolate plotting/verification memory from the long-lived queue.
                checked([sys.executable, "-u", "-m", "motion_static.evaluation.hash_sweep", "verify",
                         "--config-dir", str(config_dir), "--out", str(group), "--run-id", name])
            state["completed"].append(name)
            state["status"] = "summarizing"
            save_state()
            summarize(config_dir, group, make_gifs=False)
            final = len(state["completed"]) == state["total"]
            if final:
                checked([sys.executable, "-u", "-m", "motion_static.evaluation.hash_sweep", "summarize",
                         "--config-dir", str(config_dir), "--out", str(group), "--gifs"])
            if publication:
                state["status"] = "publishing"
                save_state()
                publish(group, name, len(state["completed"]), state["total"])
        state["status"] = "complete"
        state.pop("current_run", None)
        save_state()
    except BaseException as error:
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        save_state()
        traceback.print_exc()
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run"))
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out", type=Path, default=RESULT_DIR)
    parser.add_argument("--smoke", action="store_true", help="Two asymmetric 3-step/25-frame probes only")
    parser.add_argument("--publish", action="store_true", help="Commit/push verified research milestones")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config_dir, args.smoke)
    elif args.action == "preflight":
        preflight(args.config_dir, args.out / "preflight.json")
    else:
        run_queue(args.config_dir, args.out, args.publish)


if __name__ == "__main__":
    main()
