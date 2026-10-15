"""Optional historical 36-setting plan, using flat run outputs and no publication.

Only template.json is maintained as an experiment input. Generate this explicitly
requested comparison into a separate scratch directory; importing does not run it.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys

from .run_records import write_json

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "motion_static/experiments/template.json"
LEVELS = (4, 8, 12)
MOTION_MAX = (16, 32)
INTENSITY_MAX = (32, 64)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_configs(smoke=False):
    base = json.loads(TEMPLATE.read_text())
    base.update(rank=2, size=64, frames=379, cycles=3, seed=0, steps=24000,
                batch_size=4096, threads=4, head_width=64, temporal_width=32,
                mlp_lr=.001, hash_lr=.01, monitor_samples=65536, monitor_seed=2026, log_interval=500)
    base["encoding"] = dict(levels=8, features=2, log2_hash_size=15, min_resolution=8, max_resolution=64)
    base["motion_encoding"] = dict(min_resolution=[2, 2, 8], max_resolution=[32, 32, 128])
    base["output"] = dict(gifs=False, predictions=False)
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


def run_queue(config_dir, results):
    """Execute only an explicitly prepared plan; failed/incomplete runs are preserved."""
    import fcntl
    from .evaluation.hash_sweep import verify_run
    manifest = load_manifest(config_dir)
    results.mkdir(parents=True, exist_ok=True)
    with (config_dir / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in manifest["run_ids"]:
            subprocess.run(["git", "pull", "--ff-only"], cwd=ROOT, check=True)
            if source_hashes() != manifest["source_sha256"]:
                raise RuntimeError("Source changed since preparation; prepare a new plan for a new implementation")
            expected = json.loads((config_dir / f"{name}.json").read_text())
            out = results / f"{config_dir.name}_{name}"
            if out.exists():
                verify_run(out, expected, manifest["source_sha256"])
                continue
            command = [sys.executable, "-u", "-m", "motion_static.run", "--config",
                       str(config_dir / f"{name}.json"), "--out", str(out)]
            subprocess.run(command, cwd=ROOT, check=True)
            verify_run(out, expected, manifest["source_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run"))
    parser.add_argument("--config-dir", type=Path, required=True,
                        help="Fresh scratch directory outside experiments/ and results/")
    parser.add_argument("--out", type=Path, default=ROOT / "motion_static/results")
    parser.add_argument("--smoke", action="store_true", help="Explicit two-setting validation plan only")
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    if any(config_dir.is_relative_to(path.resolve()) for path in
           (ROOT / "motion_static/experiments", ROOT / "motion_static/results", args.out)):
        parser.error("Keep generated queue inputs outside the maintained experiments/ and output directories")
    if args.action == "prepare":
        prepare(config_dir, args.smoke)
    elif args.action == "preflight":
        preflight(config_dir, config_dir / "preflight.json")
    else:
        run_queue(config_dir, args.out)


if __name__ == "__main__":
    main()
