# Run the 36-Configuration Hash Sweep on Another Machine

## Purpose and current handoff

This document transfers the **existing, approved 36-configuration experiment**.
The research goal is to separate motion from genuine intensity changes in a
representation intended for real PACT reconstruction. This experiment uses
image supervision on a toy phantom; it does not yet use acoustic measurements
or a PACT forward operator.

**Status checked on 2026-10-14:** 10/36 configurations finished 24,000 steps and
passed saved-result verification. The local queue stopped while pushing the
tenth result because GitHub DNS resolution failed. The failed push has since
been recovered. The local queue has not been restarted for this handoff.
No results were lost. The remaining 26 configurations have not started.

All ten completed configurations use **four motion levels**. Therefore, they
cannot establish whether motion should use 4, 8 or 12 levels. Among those ten:

| Current distinction | Motion levels / spatial max | Intensity levels / spatial max | Image RRMSE | Intensity RRMSE | Motion RMSE |
|---|---|---|---:|---:|---:|
| Lowest intensity and motion errors | 4 / 32 | 8 / 64 | 5.1053% | 28.5814% | 2.7565 px |
| Lowest image error | 4 / 32 | 4 / 32 | 4.7152% | 39.1963% | 3.7580 px |

These are provisional, single-seed results, not the winner of the full sweep.
The leading separation configuration still has blurred intensity and local
motion errors. See the [current result index](../../../results/complex/hash_sweep_24k/README.md)
for all ten metrics, GIFs and loss curves, and [PROGRESS.md](../../../PROGRESS.md)
for the preceding experiments.

## Fixed model, data and evaluation

The nested model is

\[
\hat I(r,t)=\phi_{\mathrm{intensity}}(r+\phi_{\mathrm{motion}}(r,t),t),
\qquad
\phi_{\mathrm{intensity}}(q,t)=\sum_{k=1}^{2}b_k(q)a_k(t).
\]

The spatial hash branch outputs two bases. The small time MLP outputs two
coefficients. Rank 2 includes the static intensity component; no additional
static residual or nonlinearity is applied after the inner product. Motion is
backward displacement and uses the existing reference anchor
`M(r,t) - M(r,t0)`, with `t0 = -0.5`.

| Swept setting | Motion | Intensity |
|---|---|---|
| Hash levels | 4, 8, 12 | 4, 8, 12 |
| Spatial maximum resolution | 16, 32 | 32, 64 |
| Spatial minimum resolution | Fixed at 2 | Fixed at 8 |
| Temporal hash endpoints | Fixed at 8 and 128 | No temporal hash |

The Cartesian product is `3 × 2 × 3 × 2 = 36`. Branch layer counts are
independent: both `motion_encoding.levels` and `encoding.levels` are explicit
in every JSON. Changing motion levels changes intermediate temporal scales,
even though the temporal endpoints stay fixed.

| Fixed setting | Value |
|---|---|
| Phantom | Existing complex phantom, 64 × 64 |
| Time domain | Three cycles, `[-0.5, 2.5]` |
| Training / unseen evaluation times | 379 frames / 378 midpoint frames |
| Steps / batch size / seed | 24,000 / 4,096 sampled points / 0 |
| Optimizer / scheduler | Existing Adam groups / cosine to zero at step 24,000 |
| Hash / MLP initial learning rate | 0.01 / 0.001 |
| Features per hash level / hash capacity | 2 / `2^15` |
| Spatial decoder width / temporal MLP width | 64 / 32 |
| Device / CPU threads | CPU / 4 |
| Checkpoints | 8,000, 16,000 and 24,000 steps |
| Fixed monitoring samples / seed | 65,536 / 2026 |
| Loss logging | Step 0, step 1, every 500 steps and final step |
| Optimized loss | Observed-image MSE only |

Keep the model, phantom, loss, sampling and metric definitions unchanged.
There is no motion/intensity ground-truth supervision and no Jacobian penalty.
Existing auxiliary evaluation diagnostics remain in the saved results.
Old 32k runs' 24k checkpoints do not replace newly trained 24k runs: their
learning-rate schedules differ.

Report the existing full-midpoint image and intensity relative L2 errors
(RRMSE), and foreground backward-displacement endpoint RMSE in pixels.
Compare each metric separately and report nondominated tradeoffs; do not
introduce a weighted score. Same seed does not guarantee identical batches
or initialization when parameter shapes change. Selection using midpoint
errors is exploratory, not an independent test of generalization.

## Repository and environment

Use a dedicated Linux or macOS checkout. The queue imports `fcntl`; native
Windows is not supported by this entry point. The existing trainer explicitly
uses **CPU**, even on a machine with CUDA or MPS. GPU adaptation is separate
implementation work and is not included in this handoff.

```bash
git clone --branch motion-static --single-branch https://github.com/zeyuchang98/nf_crt_dynamic_imaging.git
cd nf_crt_dynamic_imaging
git pull --ff-only
git status -sb
```

For an existing checkout, enter its repository root and run the last two
commands. Preserve any unrelated changes. Keep the branch named
`motion-static` tracking `origin/motion-static`; the queue checks this branch.
Use one machine as the active writer to avoid competing experiment outputs.

The frozen configuration manifest is [manifest.json](manifest.json), beside
this document. The preparation commit is
`de8e6ae2f4c7fae28a62d1ab694893d8fc7bb8e5`; all ten current runs match its
frozen Python source hashes. Each run records its own Git HEAD, which can
differ because later commits add results/documentation. The manifest's
Python source hashes are the actual implementation check.
Do not regenerate the research manifest or change source hashes to bypass a
mismatch. The training command snapshots the actual source used by every run.

Observed local environment:

| Component | Observed version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| scikit-image | 0.26.0 |
| Pillow | 11.3.0 |

Use an isolated environment with these versions and their dependencies. For
example, with Python 3.11 already installed:

```bash
python3.11 -m venv .venv-hash-sweep
source .venv-hash-sweep/bin/activate
python -m pip install torch==2.14.0 numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 scikit-image==0.26.0 Pillow==11.3.0
python -m pip check
```

These are the observed version targets, not a verified installation on the
receiving machine. In particular, availability of a matching PyTorch wheel
must be checked there. If installation fails, report the exact error rather
than silently changing the numerical environment. A different Python patch,
package build, platform or version must be recorded in the new machine's run
record and validated before treating results as comparable.

Git access is required before **every** run, even without `--publish`.
Automatic publication additionally needs `gh` authenticated with repository
access, Git write access, and configured Git commit identity. Use the
machine's normal credential setup; do not put credentials in this document
or experiment logs.

## Validate the receiving machine first

The commands below assume the repository root and the activated environment.
Keep thread limits consistent with the four-thread baseline. Use new smoke
directories if these paths already exist; preserve old smoke outputs.

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

git pull --ff-only
python -m unittest discover -s motion_static/tests -v

python -m motion_static.hash_sweep prepare --smoke --config-dir /tmp/nf_crt_hash_sweep_smoke_configs
python -u -m motion_static.hash_sweep run --config-dir /tmp/nf_crt_hash_sweep_smoke_configs --out /tmp/nf_crt_hash_sweep_smoke_results
```

Acceptance: the tests pass; the smoke queue reports `complete`, `2/2`; both
run verification reports pass; the comparison GIFs have two model columns;
the motion truth panel changes over time. The smoke uses only three steps
and 25 frames and must never appear in the research comparison. Its
`preflight.json` records actual package versions and device availability.
On the original machine, 70 tests, the 36-configuration numerical preflight
and two independent-branch smoke runs passed. The receiving machine has not
yet been tested.

## Option A: Run all 36 afresh on the new machine

This is the primary route for a complete set on one machine. A normal clone
already contains the ten completed runs under the default output directory.
Use a **new output directory** to train every configuration again while
preserving those results.

```bash
export SWEEP_CONFIG_DIR="$PWD/motion_static/experiments/complex/hash_sweep_24k"
export SWEEP_OUT="$PWD/motion_static/results/complex/hash_sweep_24k_machine2"
test ! -e "$SWEEP_OUT" || { echo "Output already exists; inspect it before proceeding."; exit 1; }

git pull --ff-only
python -m motion_static.hash_sweep preflight --config-dir "$SWEEP_CONFIG_DIR" --out "$SWEEP_OUT"
python -m pip freeze > "$SWEEP_OUT/environment.txt"
python -c 'import json, platform; from pathlib import Path; import os; Path(os.environ["SWEEP_OUT"], "machine.json").write_text(json.dumps({"system":platform.system(),"machine":platform.machine(),"python":platform.python_version()}, indent=2) + "\n")'

nohup python -u -m motion_static.hash_sweep run --config-dir "$SWEEP_CONFIG_DIR" --out "$SWEEP_OUT" > "$SWEEP_OUT/queue.log" 2>&1 < /dev/null &
echo $! > "$SWEEP_OUT/.launcher.pid"
```

`nohup` protects against terminal disconnection, not machine shutdown or
suspension. Run on the intended persistent machine or through its normal job
scheduler. The queue is serial; do not add parallel workers or change thread
counts as part of this comparison. The queue checks at least 10 GiB free;
reserve additional space for the checkout/history and saved results.

**Publication limitation:** the current `--publish` option only supports
the default `hash_sweep_24k` result directory. It rejects a custom output
directory, so the Option A command intentionally omits it. Training,
verification and final comparisons still run automatically, and Git pulls
still occur before every run. Publish each completed run and its updated
comparison files as a scoped milestone from the receiving machine, following
the project's standing commit/push rules. Exclude caches, the live queue
state/log, lock and PID file. Keep the numeric histories, predictions, source
snapshots, checkpoints, metrics and GIFs. Do not use `git add .` or include
unrelated changes. Do not claim GitHub is current until its push is verified.

If fully automatic publication to a separate output directory is required,
that is a small queue change to implement and validate before launch; this
documentation update does not change that behavior.

## Option B: Preserve the ten completed runs and finish the remaining 26

Use the default directory from the clone. Verify Git/`gh` access first, then:

```bash
export SWEEP_OUT="$PWD/motion_static/results/complex/hash_sweep_24k"
git pull --ff-only
gh auth status
python -m motion_static.hash_sweep preflight --out "$SWEEP_OUT/validation/machine2"
python -m pip freeze > "$SWEEP_OUT/validation/machine2/environment.txt"
nohup python -u -m motion_static.hash_sweep run --publish >> "$SWEEP_OUT/queue.log" 2>&1 < /dev/null &
echo $! > "$SWEEP_OUT/.launcher.pid"
```

The queue verifies and skips completed runs; it does not train them again.
It currently reprocesses comparisons/publication for skipped runs before
reaching the first missing configuration, so its live completed counter
rebuilds from zero during restart. Count verified run folders when reporting
the already completed total. The next missing configuration in this handoff
is `mL4_max32_iL12_max32`.

This route combines the old machine's ten runs with 26 new-machine runs.
Keep that provenance explicit; do not describe it as all 36 rerun on one
machine. Automatic publication updates the marked sweep section of
`motion_static/PROGRESS.md` and publishes each verified milestone.

## Inspect progress, recover, and collect results

With `SWEEP_OUT` set to the chosen output directory:

```bash
cat "$SWEEP_OUT/queue_state.json"
tail -n 30 "$SWEEP_OUT/queue.log"
cat "$SWEEP_OUT/metrics.csv"
```

The current per-run training log is named in `queue_state.json`. A growing
loss CSV confirms training progress; `summary.json` plus a passed
`verification.json` establishes a completed, verified run.

- **Network or publication failure:** preserve the log and outputs. Repair
  connectivity, run `git pull --ff-only`, and retry any pending `git push`
  when using automatic publication. Then rerun the same queue command with
  log append (`>>`). Do not rerun the fresh-output guard from Option A.
- **Partial run:** the queue intentionally stops if a run directory exists
  without `verification.json`. It does not automatically resume training
  from intermediate checkpoints. Inspect whether training, evaluation or
  verification failed. Preserve the failed output and actual errors; agree
  on recovery before moving/deleting output or restarting a full run.
- **Code/config hash mismatch or divergent Git history:** stop and resolve
  it without stash/reset, forced pulls, changed manifest hashes or scientific
  changes. A different implementation must not silently enter this sweep.
- **Duplicate launch:** the output-directory file lock rejects a second
  queue. Do not remove or bypass the lock while a queue is active.

Every formal run must retain:

- `config.json`, `source.zip`, `model.pt`, and all three checkpoints;
- raw prediction arrays and the existing full metric JSONs;
- `image.gif`, `reference_intensity.gif`, `motion_effect.gif`, and
  `motion_only.gif`, with still images;
- `loss.csv`, fixed monitoring sample indices, loss curves and per-group
  learning-rate history;
- a passed `verification.json` and the original training log.

The group contains the incremental `metrics.csv`, `comparison.json`, metric
heatmaps and grouped loss curves. After all 36 finish, it also generates
shared-time/shared-scale GIFs for the reference configuration, each metric's
best configuration and all nondominated configurations. There is no need to
start another training job to generate these comparisons.

Completion means **36/36 verified**, a queue state of `complete`, all final
comparison artifacts present, and publication verified separately. A small
image loss alone is not evidence that the branches separated correctly.

## Scope of this handoff

The source and configuration commands above were checked against the current
entry points. This handoff adds documentation and records existing evidence;
it does not change training code, launch remote jobs, adapt the model to GPU,
change the loss, add seeds, or expand beyond the approved 36 configurations.
