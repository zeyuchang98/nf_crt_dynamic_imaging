> Cleanup update: use a fresh output directory for a new approved sweep. The old result folders retain records only and cannot be resumed or revalidated without restoring their artifacts. The full-artifact `--publish` option documented below is historical; do not use it with the current compact Git retention policy. See the [current training guide](../../../README.md#output-retention).

# Joint Motion / Intensity Hash Sweep

The manifest in this directory is pinned to the historical source. Following the
approved phantom-default update, prepare a fresh manifest in a new configuration
directory before a new sweep; see the [current training guide](../../../README.md#run-a-fresh-experiment).
The 36 parameter combinations already use the standard complex/3-cycle/379-frame data.

Approved on 2026-10-14: **36 new runs, each with 24,000 optimizer steps**.

For transfer to a persistent machine, use [Run on another machine](RUN_ON_ANOTHER_MACHINE.md).
It distinguishes a fresh 36-run set from continuing the 26 unfinished runs,
and documents the CPU execution and custom-directory publication limitations.

| Setting | Motion branch | Low-rank intensity branch |
|---|---|---|
| Hash levels | 4, 8, 12 | 4, 8, 12 |
| Spatial minimum resolution | 2 | 8 |
| Spatial maximum resolution | 16, 32 | 32, 64 |
| Temporal hash endpoints | 8 to 128 | No temporal hash; existing small time MLP |

Both branch settings vary independently. A run named `mL4_max16_iL12_max64`
has four motion levels with spatial maximum 16 and twelve intensity levels with
spatial maximum 64. Each JSON explicitly sets both level counts to avoid
unintended inheritance from the intensity encoder. Changing motion levels also
changes the intermediate temporal scales; temporal endpoints remain fixed.

Rank 2, the complex 64x64 phantom, 3 cycles, 379 frames, seed 0, batch 4,096,
four CPU threads, features 2/level, hash capacity 2^15, head width 64, time MLP
width 32, hash/MLP learning rates 0.01/0.001, image MSE and the reference anchor
are unchanged. Each cosine schedule ends at 24k. Checkpoints are saved at
8k, 16k and 24k. Old 32k-run checkpoints are excluded from this comparison.

## Execution

Run from the repository root with the existing local `nf-crt` environment.
No packages, external data, GPU rental or environment rebuild are required.
The established trainer executes on CPU. The queue runs one experiment at a
time; it checks free disk space and pulls the current tracking branch with
`git pull --ff-only` before every run. A source-hash change, conflict,
validation failure or publication failure stops the queue without deleting
or replacing outputs. A file lock prevents duplicate queues for the same
output directory. Already verified runs can be checked and skipped after
interruption; an incomplete run requires inspection before any restart.

```bash
/Users/zeyuchang/miniconda3/envs/nf-crt/bin/python -m motion_static.hash_sweep preflight
/Users/zeyuchang/miniconda3/envs/nf-crt/bin/python -u -m motion_static.hash_sweep run --publish
```

The 36 committed JSONs and `manifest.json` are the frozen input. Do not edit
them during a sweep. The queue preserves actual Python sources in each run's
`source.zip` through the unchanged training entry point. Milestone publication
stages only the sweep results and its marked section of `PROGRESS.md`, checks
for an empty pre-existing staging area, then commits, pushes and verifies the
GitHub artifact hashes. Unrelated local changes are not included.

## Validation-only smoke invocation

Provider: existing local CPU, four threads, Python executable above. No
`.aris/compute` ledger exists for this environment; actual package versions
and availability are recorded in `preflight.json` instead of claiming a
cached environment match. The following uses two asymmetric branch settings,
three steps, 25 frames and 256 monitoring samples. It is **not a research run**.
The configuration preparation command must be run once into a new directory.

```bash
/Users/zeyuchang/miniconda3/envs/nf-crt/bin/python -m motion_static.hash_sweep prepare --smoke --config-dir /tmp/hash_sweep_24k_smoke_v2_configs
/Users/zeyuchang/miniconda3/envs/nf-crt/bin/python -u -m motion_static.hash_sweep run --config-dir /tmp/hash_sweep_24k_smoke_v2_configs --out /tmp/hash_sweep_24k_smoke_v2_results
```

The second command is the fresh-agent smoke invocation, after preparation.
It includes training, saved-result verification, numeric comparisons, heatmaps,
loss comparisons and shared-scale GIFs. Publication is disabled for smoke
outputs. A success is `queue_state.json` with status `complete`, two verified
runs and generated comparisons. Any failure is reported verbatim, without
improvising alternative commands or silently changing the experiment.

## Outputs and interpretation

Results are written to `motion_static/results/complex/hash_sweep_24k`:

- `queue_state.json`: live local execution state. The background launcher redirects
  console output to `queue.log`; a direct command prints it to the calling terminal
  or its chosen log destination instead.
- `runs/<id>/`: configurations, source snapshots, final/intermediate models,
  raw prediction arrays, full existing metrics, four branch/image GIFs,
  numeric losses and per-group learning rates, and loss curves.
- `runs/<id>/verification.json`: source/checkpoint consistency, rank-2 tails,
  exact reference anchor, finite coordinate gradients, sampled prediction
  reloads, recomputed main metrics and final fixed losses, monitoring indices,
  cosine schedule, GIF checks and stable artifact hashes.
- `logs/<id>.log`: original training and evaluation output.
- `metrics.csv`, `comparison.json`, `metric_heatmaps.png`,
  `loss_comparison.png`, `README.md`: incrementally refreshed verified results.
- `comparisons/`: after all runs finish, shared-time/shared-scale image,
  intensity and motion GIFs for the reference configuration, each metric's
  best configuration and all nondominated configurations, deduplicated and
  split into readable groups of at most three.

Metrics retain the existing definitions: global midpoint image/intensity
RRMSE, and foreground backward-displacement endpoint RMSE in pixels. Report
per-metric best settings and nondominated tradeoffs, **without a weighted
score**. No Jacobian penalty is added. Auxiliary diagnostics already generated
by the trainer remain in the full saved results.

The motion truth GIF queries the same true time-varying intensity with true
and predicted motion. It differs from `motion_only.gif`, which freezes learned
intensity at the reference time. Loss curves show unsmoothed fixed training
and held-out losses; held-out labels never enter optimization.

This is a single-seed hyperparameter comparison on a toy phantom. Changing
parameter shapes changes RNG consumption, so identical seeds do not imply
identical initialization or batches. Selecting settings using midpoint errors
does not provide an independent generalization test. Training time is recorded
but is not a controlled speed benchmark. Real PACT reconstruction remains the
target; this sweep does not establish real-object performance or a general
rank bound.
