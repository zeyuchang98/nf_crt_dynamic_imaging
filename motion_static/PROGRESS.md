# Motion and Intensity Separation: Current Progress

Last updated: 2026-10-14.

## Goal and current baseline

Represent real dynamic objects for PACT reconstruction, including both motion and
genuine brightness changes. Current experiments fit toy images directly; no PACT
acoustic forward operator or real measurements have been evaluated here.

The default data are the **complex 64x64 phantom, 3 complete cycles, 379 training
frames**, time [-0.5, 2.5], 126 intervals per cycle, and 378 unseen midpoint frames.
The phantom has an elliptical body, two spots, x/y translation, unequal x/y
expansion, and global/local brightness changes. The multicycle extension retains
these dynamics and doubles the original temporal sampling density.

The maintained input is [experiments/template.json](experiments/template.json).
It retains the previous generic configuration's rank 3, 8k steps and eight-level
8-128 hash settings. These are editable starting values, not a new best-setting
claim. R=2 remains the working research prototype; select it explicitly when
planning a run. Rank, budget and architecture changes require an agreed experiment.

## Representation and conventions

$$
q=r+\phi_{\mathrm{motion}}(r,t),\qquad
\hat I(r,t)=\phi_{\mathrm{intensity}}(q,t),\qquad
\phi_{\mathrm{intensity}}(q,t)=\sum_{k=1}^{R} b_k(q)a_k(t).
$$

The spatial hash produces R bases; a small temporal MLP produces R coefficients.
R is the total rank, including static intensity. There is no extra residual or
nonlinearity after the inner product. Motion is a backward displacement to the
reference frame, anchored by subtracting its output at t0=-0.5, so it is exactly
zero at the first frame. Reference intensity is evaluated at fixed q coordinates.

Only observed image values supervise training. True motion, reference intensity
and held-out labels are for evaluation/monitoring. A step is one optimizer update
on a sampled batch, normally 4,096 space-time points. The main objective remains
image MSE, without branch supervision or an added regularizer.

[METRICS.md](METRICS.md) contains formulas and implementation status. Motion accuracy
compares displacement vectors directly; applying motion to true intensity is for
visualization. Existing pixel endpoint RMSE, mean EPE, vector RRMSE and scalar
RRMSE/PSNR/SSIM remain implemented. Proposed dynamic-intensity RRMSE, physical/FOV
units and unified spatial-frame SSIM remain unimplemented. Historical midpoint
volume-SSIM and masked block spatial-SSIM are different protocols.

## Historical findings retained after artifact deletion

The following numbers summarize previously completed experiments. Old run folders,
configs, metrics files and loss curves were deleted at the user's request during
the current cleanup. These are historical conclusions, not newly reproduced
results or links to available models. Local and second-machine runs remain distinct.

### Nested models and low-rank intensity

Single cycle, 64 training frames, 63 unseen midpoints, seed 0, 8k steps:

| Model | Image RRMSE | Intensity RRMSE | Foreground motion endpoint RMSE |
|---|---:|---:|---:|
| Direct Hash NF | 4.03% | — | — |
| Nested MLP | 13.53% | 98.72% | 6.553 px |
| Nested Hash NF | 3.40% | 98.99% | 6.421 px |
| Low-rank R=2 | 5.16% | 46.65% | 4.775 px |
| Low-rank R=3 | 5.31% | 52.04% | 4.817 px |

The zero-motion baseline was 6.066 px. Original nested models fit images while
intensity retains geometry changes. Low rank improves branch errors but does not
establish correct separation. R=2 was slightly better in this single-seed comparison.

The user preferred the early R=2 intensity GIF (the fourth reviewed example).
That run used eight 8-128 levels in both encoders, two features per level, capacity
2^15 per level, head width 64, temporal width 32, Adam hash/MLP rates 0.01/0.001 and
an 8k cosine schedule. This visual preference does not verify the true temporal
waveform. The GIF and model are no longer retained in the current worktree.

### More cycles, denser frames and longer training

Three cycles, 379 training frames, seed 0, R=2:

| Independent budget | Image RRMSE | Intensity RRMSE | Motion RMSE | Fixed train / validation MSE |
|---|---:|---:|---:|---|
| 8k | 2.46% | 63.87% | 5.473 px | 4.02896e-6 / 4.96784e-6 |
| 32k | 1.88% | 68.66% | 5.484 px | 1.09396e-6 / 2.83704e-6 |

Each run had its own cosine schedule; the 32k run did not continue the 8k model.
Image fitting improved without recovering true branches. No sustained validation
loss increase was observed; the late plateau with decaying learning rate does not
prove complete convergence. Cycle count and density changed together, and the
normalized temporal grid changed relative to physical cycles, so their effects
were not isolated. Single- and multicycle aggregate errors cover different domains.

### Hash-grid tuning

At 32k steps, reducing motion spatial resolution from 8-128 to 2-32 while retaining
its time resolution improved motion RMSE from 5.484 to 4.339 px and intensity RRMSE
from 68.66% to 55.06%, while image RRMSE increased from 1.88% to 3.14%.

With this motion grid, the eight-level intensity max-64 run had the lowest image
and intensity RRMSE among max 128/64/32: 3.13% / 35.10%. Max 32 gave 3.60% / 42.04%
and motion RMSE 3.29 px. Lower maximum resolution was not uniformly better.

The 24k sweep varied motion levels 4/8/12 and max 16/32, and intensity levels
4/8/12 and max 32/64, for 36 combinations. All used the three-cycle data and R=2.
The local machine completed 10; the second machine completed all 36.

| Source and distinction | Motion levels/max | Intensity levels/max | Image RRMSE | Intensity RRMSE | Motion RMSE |
|---|---|---|---:|---:|---:|
| Local: lowest intensity and motion errors | 4/32 | 8/64 | 5.1053% | 28.5814% | 2.7565 px |
| Second machine: lowest image error | 12/32 | 8/64 | 2.5825% | 33.8151% | 3.4273 px |
| Second machine: lowest intensity/motion errors | 4/32 | 8/64 | 5.0667% | 28.8133% | 2.8088 px |

The local low-error intensity prediction was nearly static: center brightness
0.3315-0.3386 versus truth 0.21-0.35, and spot brightness 0.7879-0.8300 versus truth
0.4056-0.9744. Lower aggregate error alone does not establish better intensity
dynamics. Reliable separation remains unresolved. These are single-seed toy
comparisons, not real-object validation or an independent generalization test.

### Earlier auxiliary experiments

The simple translating disk also showed accurate image fitting with poor branch
recovery. Its 8k Nested MLP result had 2.55% image RRMSE, 97.71% intensity RRMSE and
7.874 px motion RMSE.

The separate 64-frame block-holdout study used 8x8x2 blocks and 20.02% held-out
pixels. Nested Hash NF had 11.09% test image RRMSE, 98.20% reference-intensity RRMSE
and 6.475 px motion RMSE; the zero-motion control was 5.974 px. This is missing-region
completion, not the unseen-frame midpoint protocol.

An earlier non-low-rank, single-cycle temporal-L2 experiment used weight 1e-4 and
time difference 1/63 at fixed, detached warped q. Image RRMSE changed 3.40% to
5.79%, intensity RRMSE 98.99% to 85.72%, and motion RMSE 6.421 to 5.925 px. Estimated
true brightness variation was suppressed. This loss is not adopted for the default
low-rank setup.

## Decisions and unresolved directions

- Preserve both geometry and true brightness dynamics; a low image loss alone is insufficient.
- R=2 is a useful toy prototype, not a proven rank bound for real PACT objects.
- No Jacobian training penalty was adopted; existing auxiliary evaluation diagnostics remain.
- New motion/intensity separation losses and alternative intensity coordinates remain discussion only.
- Proposed standardized metrics in `METRICS.md` require review before implementation.
- Future training must retain numeric losses, learning rates, fixed training/held-out monitoring and a loss curve.

## Current organization cleanup

The user approved removal of old experiment inputs and results, one reusable
template, flat run folders, retained monitoring indices and opt-in GIFs. Removed
50 old experiment files and 593 old result files (about 66.6 MiB of results).
Research conclusions above replace obsolete links; no old results were rerun.

Training now records config/environment/source/status, one complete metric JSON,
loss CSV/PNG, final model, monitoring indices, actual source ZIP and console log.
Optional exports are one prediction archive, selected intermediate checkpoints
and GIFs in `visuals/`. Block experiments additionally preserve their exact split.
`summary.csv` is derived from flat run records with explicit protocol labels.
A dry-run command previews settings without creating files or training.

The old sweep helper now requires explicitly generated scratch inputs and uses
flat runs. It no longer publishes results or creates plots/GIF comparisons
automatically. Shared models, phantom dynamics and numerical metric definitions
are retained. Unrelated `nfCRT/` and `applications/` edits are preserved.

**Validation completed:** all 79 unit tests passed. Default and opt-in exports
produce identical metrics; later rendering leaves the model, metrics, loss and
monitoring samples unchanged. Inference-only comparisons against the pre-cleanup
evaluation code reproduced every existing midpoint and block metric, including
ux/uy reports, exactly on small fixtures. The template preserves the previous
generic model's parameter layout and seeded initialization exactly. Dry-run and
failure recording checks passed; local documentation links resolve. Refreshed
`BASELINE.json` and confirmed unrelated files are byte-identical.

No research experiment or parameter search was launched. New integration fixtures
used initialized models without optimizer updates; existing numerical unit tests
retain their small synthetic optimizer checks. These checks establish software
consistency, not model quality on the new benchmark.

## Training readability refactor

`run.train()` now reads as sampling, objective calculation, optimizer update and
recording. Optimizer groups are built separately. `losses.compute_training_losses()`
assembles named scalar terms and their weighted total; the monitor receives names
instead of relying on tuple order. Run metadata, source snapshots and checkpoints
are in `run_records.py`, whose summary extraction explicitly separates midpoint
and block report fields. The README documents where to edit an agreed future loss.
Current formulas, weights, defaults, logging intervals and output schemas are unchanged.

**Validation completed:** 81 tests passed, including analytic objective/gradient
checks and verification that zero temporal weight skips that penalty's queries.
Compared with commit `208b8ffc4ec0c1b2232a0b174ee506ed822fe541`, seven four-step
synthetic checks covered all four model families and both zero/nonzero temporal
weight for nested models. Loss history, weights, RNG, optimizer/scheduler checkpoint
state, monitoring indices, console values and loss CSV values matched exactly
(elapsed time excluded). Sixteen summary-row comparisons also matched. Refreshed
`BASELINE.json`; results remain empty except for `.gitkeep`. No research experiment
or parameter search was launched, and no new scientific conclusion is implied.

## Approved baseline comparison: completed

The user approved local CPU runs with the earlier visually preferred 8-level
8-128 grids and 8k updates: nested hash, low-rank + motion R=2, and motion-free
low rank at R=8,16,24. All use the complex 64x64, 3-cycle/379-frame benchmark,
seed 0, batch 4096, Adam hash/MLP rates 0.01/0.001 and the existing cosine schedule.
The three motion-free candidates are ranked by final fixed midpoint validation
image MSE. Full midpoint image RRMSE/PSNR/SSIM are reported for all; these are
validation results after rank selection. No extra seeds or rank expansion is approved.

Added `low_rank_hash_nf` by directly reusing `LowRankIntensity` at observation
coordinates. The runner, optimizer and checkpoint loader now support it; its
metrics report full-image rank and image quality, without invented motion or
reference-intensity branch scores. Existing nested models and loss formulas are
unchanged. `baselines.py` prepares the exact five inputs in scratch storage,
checks code/config hashes and runs serially into flat result folders, retaining
all records and identifying the selected rank. GIFs remain opt-in.

**Preflight verified:** 87 tests passed, including all five actual model settings,
rank bounds, coordinate gradients, reference anchors where applicable, optimizer
parameter coverage, three-step synthetic motion-free updates, checkpoint reload,
image-only evaluation and validation-based selection. A fresh agent executed the
documented seeded runtime witness successfully. Existing local Python 3.11.16 /
PyTorch 2.14.0 environment is reused on CPU; no dependencies changed.

All five runs completed on 2026-10-14 local. The first used commit
`4c74bf9a527860e251b1027b42914a63e772cb67`; the remaining runs used `75343b6`,
which only updated this progress record. Every run's Python source hashes are
identical. Scratch inputs/queue log are under `/tmp/baselines_20261014_8k`;
resolved configurations, numerical results and curves are retained below.

### Complete midpoint image comparison

All rows use the same 378 midpoint frames, data range 1 and existing sequence
SSIM convention. RRMSE is displayed as a percentage. Training time includes
fixed monitoring and plotting, and excludes the final full-grid evaluation.

| Model and full metrics | Image RRMSE | PSNR (dB) | SSIM | Fixed validation MSE | Train (s) | Loss/LR curve |
|---|---:|---:|---:|---:|---:|---|
| [Nested hash](results/baselines_20261014_8k_nested_hash/metrics.json) | 1.1988% | 59.1210 | 0.999882 | 1.2356603e-06 | 281.23 | [PNG](results/baselines_20261014_8k_nested_hash/loss.png), [CSV](results/baselines_20261014_8k_nested_hash/loss.csv) |
| [Low rank + motion, R=2](results/baselines_20261014_8k_low_rank_motion_r2/metrics.json) | 2.5181% | 52.6742 | 0.999630 | 5.3106635e-06 | 277.42 | [PNG](results/baselines_20261014_8k_low_rank_motion_r2/loss.png), [CSV](results/baselines_20261014_8k_low_rank_motion_r2/loss.csv) |
| [Low rank only, R=8](results/baselines_20261014_8k_low_rank_only_r8/metrics.json) | 61.4310% | 24.9278 | 0.761793 | 0.0031772972 | 67.80 | [PNG](results/baselines_20261014_8k_low_rank_only_r8/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r8/loss.csv) |
| [Low rank only, R=16](results/baselines_20261014_8k_low_rank_only_r16/metrics.json) | 65.4166% | 24.3818 | 0.750756 | 0.0035703279 | 68.07 | [PNG](results/baselines_20261014_8k_low_rank_only_r16/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r16/loss.csv) |
| [Low rank only, R=24 (selected)](results/baselines_20261014_8k_low_rank_only_r24/metrics.json) | 59.7455% | 25.1695 | 0.772001 | 0.0029911024 | 67.66 | [PNG](results/baselines_20261014_8k_low_rank_only_r24/loss.png), [CSV](results/baselines_20261014_8k_low_rank_only_r24/loss.csv) |

### Branch comparison

Only the two nested models have independently evaluated motion and reference
intensity. Motion errors compare displacement vectors directly on the existing
true foreground mask; the zero-motion endpoint RMSE is 6.0701 px.

| Model | Reference intensity RRMSE | Motion vector RRMSE | Motion endpoint RMSE (px) |
|---|---:|---:|---:|
| Nested hash | 94.2346% | 101.7730% | 6.1777 |
| Low rank + motion, R=2 | 59.0838% | 85.9171% | 5.2153 |

**Selected motion-free baseline: R=24**, by the lowest final fixed validation MSE
among R=8,16,24. See the [selection record](results/baselines_20261014_8k_selection.json)
and [all-run summary](results/summary.csv). All candidate records and local models
are retained. Model weights remain local in each run's `model.pt`; lightweight
records (config, metrics, losses, curve, monitoring indices, source and log) are
published. GIFs were subsequently generated on request (links below); prediction
arrays and intermediate checkpoints remain absent.

Nested hash fits the image best but does not recover the true branches. Low-rank
+ motion reduces both branch errors while retaining substantial decomposition
error. Motion-free low rank fits poorly at this budget, even for the selected
R=24. These runs do not distinguish insufficient capacity from optimization or
training-budget limitations; they do not establish a fundamental limitation of
motion-free low-rank representations. There is no extra rank search or training.

**Post-run verification:** reloaded all five saved models and reproduced their
final fixed training and validation MSE exactly. All five runs use identical
monitoring indices. Checked the 8k final step, complete loss/LR rows, zero final
learning rates, eight-file core output layout, source archive hashes, motion anchors
and absence of fictitious branch metrics for motion-free models. Rank tails were
below 6.1e-8 for the trained low-rank predictions. Recomputed the rank selection
and matched the saved selection record. These checks validate saved results,
not convergence or physical separation.

### GIFs generated on request

Rendered all five saved models with the existing `evaluation.render` command;
no optimization, model/configuration changes or metric recomputation. GIFs remain
opt-in for future runs. Each animation uses all 378 unseen midpoint frames over
three cycles, 90 ms per frame and infinite looping.

| Model | Full image | Reference intensity | Motion effect | Motion only |
|---|---|---|---|---|
| Nested hash | [GIF](results/baselines_20261014_8k_nested_hash/visuals/image.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/reference_intensity.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_effect.gif) | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_only.gif) |
| Low rank + motion, R=2 | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/image.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/reference_intensity.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_effect.gif) | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_only.gif) |
| Low rank only, R=8 | [GIF](results/baselines_20261014_8k_low_rank_only_r8/visuals/image.gif) | N/A | N/A | N/A |
| Low rank only, R=16 | [GIF](results/baselines_20261014_8k_low_rank_only_r16/visuals/image.gif) | N/A | N/A | N/A |
| Low rank only, R=24 (selected) | [GIF](results/baselines_20261014_8k_low_rank_only_r24/visuals/image.gif) | N/A | N/A | N/A |

Full image compares truth, prediction and absolute error. Reference intensity is
queried at fixed coordinates, without applying motion. Motion effect applies true
and predicted motion to the same true time-varying intensity, so brightness may
still vary. Motion only freezes the learned intensity at the motion reference time:
the left panel is the fixed learned image, the middle applies predicted motion,
and the right shows absolute change, not error against ground truth. Motion-free
models have only the full-image view.

Truth/prediction display ranges are fixed at 0..1. Each GIF's error/change range
is fixed over time but scaled independently and printed above its right panel;
compare the numerical metrics rather than error-panel brightness across GIFs.

**Rendering verification:** all 11 GIFs decode to 378 frames at 864x408, with
90 ms duration and looping. All prediction panels vary over time; both motion-only
left panels remain identical across all frames as intended. Inspected labeled
frames for every GIF. SHA-256 checks confirmed all 42 original result artifacts
(including models, metrics and losses) and 49 unrelated local files are unchanged.
The new GIFs occupy 36.22 MiB and are published with this index.

## Motion maximum resolution 32 baseline (2026-10-15): completed

Approved one additional local CPU run of low-rank + motion R=2, changing only
`motion_encoding.max_resolution` from 128 to scalar 32 and enabling requested GIF
exports. The scalar setting applies to x, y and t; motion resolutions are
`[8, 9, 11, 14, 17, 21, 26, 32]`. Intensity remains eight levels from 8 to 128.
The 8k budget, seed 0, complex 64x64/3-cycle/379-frame benchmark, batch size,
optimizer, learning rates, loss, monitoring samples and evaluation protocol match
the original R=2 baseline. No loss or implementation changes are made.

The unchanged local `nf-crt` runtime is reused. Preflight checked exact setting
and source parity, identical intensity initialization, optimizer coverage,
coordinate gradients, motion reference anchoring and rank bounds. Three tiny
synthetic updates verified gradient flow through both branches. The initial
scratch check incorrectly demanded nonzero motion-table gradients before the
zero-initialized output head updated; the corrected check verifies the expected
initial zero and subsequent nonzero gradients. Model parameters: 224,372 versus
390,934 for max=128. Scratch inputs/checks are in `/tmp/baseline_motion32_20261015`.
The run completed all 8,000 updates using commit `fab901e1e02a71953a9f301393ff6df81408b937`.
Its Python source hashes match the original baseline exactly. The resolved
[configuration](results/baselines_20261015_8k_low_rank_motion_r2_m32/config.json),
[full metrics](results/baselines_20261015_8k_low_rank_motion_r2_m32/metrics.json),
[loss/LR curve](results/baselines_20261015_8k_low_rank_motion_r2_m32/loss.png),
[numeric losses](results/baselines_20261015_8k_low_rank_motion_r2_m32/loss.csv),
monitoring indices, source archive and run log are retained. Weights remain local
in `model.pt`, consistent with the other baselines.

All image/intensity scores below use the complete 378 midpoint frames. Motion
scores compare displacement vectors directly on the existing true foreground mask.

| Metric | Motion max 128 | Motion max 32 |
|---|---:|---:|
| Image RRMSE | 2.5181% | 5.3585% |
| Image PSNR (dB) | 52.6742 | 46.1147 |
| Image SSIM | 0.999630 | 0.998119 |
| Reference intensity RRMSE | 59.0838% | 52.0252% |
| Reference intensity PSNR (dB) | 26.7674 | 27.8725 |
| Reference intensity SSIM | 0.874101 | 0.880784 |
| Motion vector RRMSE | 85.9171% | 80.0716% |
| Motion endpoint RMSE (px) | 5.2153 | 4.8604 |
| Fixed training MSE | 4.0248556e-06 | 2.5598572e-05 |
| Fixed validation MSE | 5.3106635e-06 | 2.6125811e-05 |
| Parameters | 390,934 | 224,372 |
| Training time (s; excludes final evaluation/GIFs) | 277.42 | 264.65 |

At this budget, reducing motion resolution improves both branch errors but
worsens complete-image fit. Substantial branch errors remain. This single-seed
comparison changes spatial and temporal resolution together; it does not establish
an optimal grid or convergence. Monitoring samples and intensity initialization
match exactly, while different motion table sizes alter subsequent random-number
consumption, so equal seeds do not imply matched training batches.

Requested GIFs use the existing definitions and display conventions described above:
[full image](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/image.gif),
[reference intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/reference_intensity.gif),
[motion effect with true time-varying intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_effect.gif),
and [motion only with frozen learned intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_only.gif).
Motion-only still uses the learned intensity at t0; the discussed alternative
using frozen true intensity was not implemented in this configuration-only run.
Default GIF export remains disabled in the reusable template.

**Post-run verification:** the saved model reproduces final fixed training and
validation MSE exactly; monitoring indices match the max=128 run. Checked all
18 loss rows, finite values, cosine rates ending at zero, source archive hashes,
the zero-motion reference anchor and rank-2 tails below 3.7e-8. All four GIFs
decode to 378 frames at 864x408 and 90 ms per frame; inspected the labels and
sampled frames, and verified the frozen left panel in motion-only. The loss curve
was visually checked. SHA-256 checks preserve all 53 old result artifacts and
49 unrelated local files; all existing summary rows are unchanged.
