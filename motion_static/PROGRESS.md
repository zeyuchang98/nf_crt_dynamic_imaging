# Motion and Intensity Separation: Current Progress

Last updated: 2026-10-15.

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
- Oracle diagnostics (2026-10-15, below): with true motion the R=2 intensity still learns nearly static
  brightness, and with true intensity the motion branch still misses most displacement. The follow-up
  direction has not been chosen.

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
queried at fixed coordinates, without applying motion. As updated on 2026-10-15,
the linked motion-effect GIFs apply true and predicted motion to the same true
intensity frozen at t0; their right panel is absolute image error. The original
exports used time-varying true intensity. Motion only freezes the learned intensity at the motion reference time:
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
[motion effect with true intensity fixed at t0](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_effect.gif),
and [motion only with frozen learned intensity](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_only.gif).
Motion-only still uses the learned intensity at t0. The motion-effect link was
subsequently regenerated with frozen true intensity in the visualization update
below; that change was not part of this configuration-only training run.
Default GIF export remains disabled in the reusable template.

**Post-run verification:** the saved model reproduces final fixed training and
validation MSE exactly; monitoring indices match the max=128 run. Checked all
18 loss rows, finite values, cosine rates ending at zero, source archive hashes,
the zero-motion reference anchor and rank-2 tails below 3.7e-8. All four GIFs
decode to 378 frames at 864x408 and 90 ms per frame; inspected the labels and
sampled frames, and verified the frozen left panel in motion-only. The loss curve
was visually checked. SHA-256 checks preserve all 53 old result artifacts and
49 unrelated local files; all existing summary rows are unchanged.

## Motion max 32: 24k-step follow-up (2026-10-15): completed

Approved one additional run with the exact R=2, motion max=32 setup above,
changing only `steps` from 8,000 to 24,000. Training starts from the same seeded
initialization; the existing cosine scheduler now reaches zero at step 24,000.
This is a new run with a longer schedule, not a continuation of the finished 8k
checkpoint. All data, model, optimizer, loss, monitoring and evaluation settings
remain identical, including the four requested GIF views. The reusable template
is unchanged.

Preflight confirmed the input differs only in `steps`, all numerical source
hashes and the runtime match, and the complete initial model state is identical
to a fresh model under the 8k configuration. The model still has 224,372 parameters.
Scratch inputs and checks are in `/tmp/baseline_motion32_24k_20261015`.
Completed all 24,000 updates using commit `5774f44cfcbaf739fbffd9a2dccd0c582c699cd5`.
The saved [configuration](results/baselines_20261015_24k_low_rank_motion_r2_m32/config.json),
[full metrics](results/baselines_20261015_24k_low_rank_motion_r2_m32/metrics.json),
[loss/LR curve](results/baselines_20261015_24k_low_rank_motion_r2_m32/loss.png),
[numeric losses](results/baselines_20261015_24k_low_rank_motion_r2_m32/loss.csv),
monitoring indices, source archive and log are retained. The final `model.pt`
remains local, as for the other baselines.

### 8k versus 24k at motion max 32

Full-image and reference-intensity scores use all 378 midpoint frames; motion
scores compare displacement vectors directly on the same true foreground mask.

| Metric | 8k steps | 24k steps |
|---|---:|---:|
| Image RRMSE | 5.3585% | 3.7109% |
| Image PSNR (dB) | 46.1147 | 49.3060 |
| Image SSIM | 0.998119 | 0.999122 |
| Reference intensity RRMSE | 52.0252% | 59.3448% |
| Reference intensity PSNR (dB) | 27.8725 | 26.7291 |
| Reference intensity SSIM | 0.880784 | 0.855292 |
| Motion vector RRMSE | 80.0716% | 83.5368% |
| Motion endpoint RMSE (px) | 4.8604 | 5.0708 |
| Fixed training MSE | 2.5598572e-05 | 1.0711806e-05 |
| Fixed validation MSE | 2.6125811e-05 | 1.2019928e-05 |
| Training time (s; excludes final evaluation/GIFs) | 264.65 | 658.16 |

The [loss comparison](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/loss_comparison.png)
plots both numeric histories without smoothing. Longer training with the longer
cosine schedule reduces fixed validation MSE by about 54% and improves image
RRMSE/PSNR/SSIM, but both branch errors become worse. In this comparison, better
image fitting does not produce better motion/intensity separation. The late loss
curve flattens as the learning rate approaches zero; this does not prove full
convergence. These are two single-seed runs with different schedule horizons,
not an isolated test of extra updates at an unchanged learning-rate schedule.

All four requested animations are saved:
[full image](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/image.gif),
[reference intensity](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/reference_intensity.gif),
[motion effect with true intensity fixed at t0](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_effect.gif),
and [motion only with frozen learned intensity](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_only.gif).
At training completion the GIF definitions, clipping and individual fixed error
scales were unchanged. The linked motion-effect GIF was subsequently regenerated
with frozen true intensity in the visualization update below.

**Post-run verification:** reloading the final model reproduces fixed training
and validation MSE exactly. Monitoring indices match the 8k run. Checked the
24k final step, all 50 finite loss/LR rows, the cosine schedule ending at zero,
source archive hashes, zero reference motion and rank-2 tails below 3.6e-8.
Initial monitoring and the first pre-update batch loss match exactly. An extra
bitwise check of first post-update monitoring failed on absolute differences of
3.59e-13 (training) and 4.29e-12 (validation); the original values were preserved
and these checks pass at 1e-8 relative tolerance. This check did not alter any
training, metrics or saved loss values; its initial failure log is retained in
the scratch folder. All four GIFs decode to 378 frames at 864x408 and 90 ms, and
motion-only has the intended fixed left panel. Sampled GIF frames and both loss
plots were visually checked. All 65 old result artifacts, 49 unrelated local
files and prior summary rows are unchanged.

## Standard motion GIF: true intensity frozen at t0 (2026-10-15)

The user approved replacing the motion-effect visualization with a shared true
reference frozen at the model's motion reference time (t0=-0.5 for these runs):
left `I_true(r + u_true(r,t), t0)`, middle
`I_true(r + phi_motion(r,t), t0)`, right the absolute image difference.
Both motion fields retain their current observation time. Neither panel uses
the learned intensity network. The true reference is evaluated continuously at
the displaced coordinates, rather than warped from a rendered reference bitmap.

`evaluation/motion_visuals.py` implements the shared definition for the standard
renderer, optional training exports, branch diagnostics and multi-run GIFs.
This replaces the former time-varying true-intensity motion view. The separately
named `motion_only.gif` retains its learned-reference diagnostic definition.
The template still disables automatic GIF generation. Numerical evaluation,
including direct motion errors and time-varying branch-replacement metrics,
is unchanged; no model was trained or metric record recomputed.

Regenerated all four existing motion-effect GIFs from the saved models:

| Model | Updated motion GIF |
|---|---|
| Low rank + motion R=2, motion max 32, 24k | [GIF](results/baselines_20261015_24k_low_rank_motion_r2_m32/visuals/motion_effect.gif) |
| Low rank + motion R=2, motion max 32, 8k | [GIF](results/baselines_20261015_8k_low_rank_motion_r2_m32/visuals/motion_effect.gif) |
| Low rank + motion R=2, motion max 128, 8k | [GIF](results/baselines_20261014_8k_low_rank_motion_r2/visuals/motion_effect.gif) |
| Nested hash, 8k | [GIF](results/baselines_20261014_8k_nested_hash/visuals/motion_effect.gif) |

**Validation:** 91 tests passed. Controls verify zero image error when predicted
motion equals true motion, a time-constant image for zero predicted motion,
custom t0 handling and unchanged inputs. Checked every panel of all 378 frames
in each regenerated GIF against the frozen-reference formula: decoded grayscale
values match exactly. All four left-panel sequences are identical and move over
time. Dimensions (864x408), 90 ms duration, looping, fixed 0..1 image display and
per-GIF fixed error scales are preserved; labeled frames were visually inspected.
Source syntax comparisons confirm that midpoint/block numerical modules changed
only at the GIF export call. Rendering old checkpoints verifies unchanged model,
phantom and data-source hashes; the revised evaluation export call is accepted
only after comparing all remaining module syntax with the hash-verified source
archive. Changed inference functions still fail the provenance check.

Checksums preserve all 75 other result files (including models, configurations,
metrics, losses, source archives and other GIFs) and 37 unrelated working files.
Validation scripts/logs are in `/tmp/frozen_true_motion_20261015`. The updated
images visualize existing motion errors; this is not a new experiment or evidence
of improved model performance.

## Oracle-branch diagnostics A1/A2 (2026-10-15): completed

Approved two 8k runs that each replace one branch of the R=2, motion-max-32 baseline
with the complex phantom's true field and train only the other branch from image MSE:
**A1** uses true backward motion `u = (r - T(t))/S(t) - r` and learns the low-rank
intensity; **A2** uses the true reference intensity `C(q,t)` (exactly rank 2) and
learns the motion hash field. These runs use ground-truth branch labels and are
diagnostics, not reconstruction methods. All other settings match the baseline.

`models/oracle_branches.py` holds torch copies of the phantom formulas (A2 needs
coordinate gradients through `C`); `phantoms/complex.py` is unchanged. The option
`oracle_branch` is accepted only for `nested_low_rank_hash_nf`, complex, midpoints
and t0=-0.5; runs set `true_motion_used_for_training` or
`reference_intensity_labels_used_for_training` and a new `oracle_branch` summary
column. Both branches are built before replacement, so the learned branch's
initialization and every training batch equal the baseline's.

**Preflight:** 98 tests passed (7 new: torch/NumPy formulas agree to 1e-12, anchors,
both oracles reproduce observed images, initialization/RNG parity, optimizer scope,
gradient flow through the true intensity, CLI rejection, complete run and reload).
At full size, dry-run inputs differ from the baseline only in `oracle_branch`; RNG
state, learned-branch initialization and the first two batches matched exactly, and
the unchanged baseline path reproduced its saved step-1 batch MSE exactly.
Implementation commit `703aa29`; both runs used it. Scratch inputs and checks are in
the session scratch folder `oracle_20261015`.

| Metric (378 midpoints) | Baseline (joint) | A1: true motion | A2: true intensity |
|---|---:|---:|---:|
| Image RRMSE | 5.3585% | 17.7370% | 10.3651% |
| Image PSNR (dB) / SSIM | 46.1147 / 0.998119 | 35.7180 / 0.991679 | 40.3841 / 0.996908 |
| Reference intensity RRMSE | 52.0252% | 19.0426% | (true) |
| Reference intensity center / spot curve RRMSE | 79.45% / 22.37% | 17.65% / 21.76% | (true) |
| Motion vector RRMSE | 80.0716% | (true) | 64.6749% |
| Motion endpoint RMSE / mean (px) | 4.8604 / 4.1877 | (true) | 3.9258 / 3.0206 |
| Foreground nonpositive-Jacobian fraction | 31.39% | 0 | 15.82% |
| Fixed training / validation MSE | 2.5599e-05 / 2.6126e-05 | 2.7660e-04 / 2.5991e-04 | 9.6457e-05 / 9.0028e-05 |
| Trainable parameters / training seconds | 224,372 / 264.65 | 67,468 / 47.87 | 156,904 / 134.85 |

Records: A1 [config](results/oracle_20261015_8k_true_motion_r2/config.json),
[metrics](results/oracle_20261015_8k_true_motion_r2/metrics.json),
[loss curve](results/oracle_20261015_8k_true_motion_r2/loss.png),
[losses](results/oracle_20261015_8k_true_motion_r2/loss.csv),
GIFs: [image](results/oracle_20261015_8k_true_motion_r2/visuals/image.gif),
[reference intensity](results/oracle_20261015_8k_true_motion_r2/visuals/reference_intensity.gif).
A2 [config](results/oracle_20261015_8k_true_intensity_m32/config.json),
[metrics](results/oracle_20261015_8k_true_intensity_m32/metrics.json),
[loss curve](results/oracle_20261015_8k_true_intensity_m32/loss.png),
[losses](results/oracle_20261015_8k_true_intensity_m32/loss.csv),
GIFs: [image](results/oracle_20261015_8k_true_intensity_m32/visuals/image.gif),
[motion effect](results/oracle_20261015_8k_true_intensity_m32/visuals/motion_effect.gif).
Weights remain local in `model.pt`.

**A1, intensity branch:** even with exact motion, the image fit is worse than the
joint baseline, and the learned intensity is nearly static. Its reference-space
singular values are 94.81 and 0.54, versus 96.46 and 5.60 for the truth. At the
body center, learned intensity has standard deviation 0.0063 over time versus
0.0495 for truth. In a least-squares fit, span(a_1, a_2, 1) of the learned temporal
coefficients explains R^2=0.021 of g(t) and 0.063 of g(t)s(t) (baseline: 0.017 and
0.053). The reference-intensity GIF also shows vertical streaks in the body. At
this architecture and budget, the temporal coefficients do not learn the true
brightness waveforms, which oscillate 9 and up to 15 times over [-0.5, 2.5]. The
joint baseline's better image fit therefore relies on incorrect, folded motion.
These runs do not distinguish temporal-MLP capacity from optimization (learning
rate 0.001, 8k steps).

*Rank versus time (analysis of the saved A1 model, no training):* R=2 is not the
limit, because the truth is exactly rank 2 and even its best rank-1 approximation
has 5.80% RRMSE. A1's 19.04% reference-intensity error comes from the temporal
coefficients. Keeping the learned a(t) and fitting optimal spatial bases by least
squares only lowers it to 18.36%; keeping the learned b(q) and fitting optimal
time coefficients lowers it to 5.60%. Of the learned coefficients' variance,
78-82% lies below 1.5 cycles per period, whereas 99-100% of the true g(t) and
g(t)s(t) variance lies above it. This is consistent with a low-frequency bias of
the temporal MLP; capacity versus optimization remains untested.

**A2, motion branch:** even with exact intensity, the learned motion recovers only
part of the displacement and still folds 15.8% of the foreground. Per-frame
foreground endpoint RMSE correlates with true displacement magnitude (r=0.60);
frames whose true foreground RMS displacement exceeds 6 px average 4.19 px error
(234 frames), versus 2.16 px below 2 px (30 frames). The median ratio of error to
true displacement is 0.70. Flat-interior motion is unobservable, so part of the
endpoint error is unidentifiable; the 10.37% image error with true intensity shows
that edge correspondence is also wrong, mainly around spot1 and the body edge.
This is consistent with a limited capture range for large displacements of sharp
edges, but that cause has not been isolated.

The waveform R^2, center standard deviation and per-frame correlations above are
ad hoc analyses of the saved models and metrics, computed outside the pipeline;
they are not adopted metrics. Single seed, single configuration.

**Post-run verification:** reloading both models reproduces final fixed training
and validation MSE exactly; monitoring indices equal the baseline's. Checked all
18 finite loss/LR rows, cosine rates ending at zero, the optimizer groups (A1:
intensity hash/MLP/time MLP; A2: motion hash/MLP), zero motion at t0, the A1 rank-2
tail (3.9e-8), source archive hashes, and sampled GIF frames.

**Provenance note:** `run.py` changed, so `evaluation.render` now refuses earlier
checkpoints by design. Re-render them from their `source.zip` or the original
commit. For this analysis, the baseline was loaded with `build_model` plus its
`state_dict`; this reproduced its saved fixed validation MSE exactly.
