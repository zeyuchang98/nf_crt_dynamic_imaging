# Coarser Motion Spatial Hash Grid

The approved 32,000-step coarser-motion run is complete. Coarsening improves motion
and reference-intensity errors relative to the original grid, but worsens image
fitting. Accurate physical separation has not been recovered.

## Comparison design

- Same 64x64 complex phantom, three complete joint cycles, 379 training times and
  378 unseen midpoints over [-0.5, 2.5]. No known motion or intensity labels are
  used for optimization, and no regularization loss is added.
- Same seed 0, batch 4096, four CPU threads, Adam learning rates, and 32,000-step
  cosine schedule. Checkpoints are saved at 8k, 16k, 24k and 32k.
- Intensity: unchanged 2D hash grid, eight levels from 8 to 128, and R=2.
- Motion: x/y cell-count limits reduced from 8-128 to 2-32, making spatial cells
  approximately four times wider. Time remains 8-128 across the same eight levels.
- The motion hash features are still interpolated before the original MLP.
  This is not a displacement-control-point model or a strict bandwidth constraint.
- Total parameters: 390,934 (baseline) versus 241,942 (coarse). Motion parameters:
  323,466 versus 174,474. Smaller tables alter hash collisions and RNG consumption.
  The initial intensity parameters and fixed-sample losses are identical, but
  motion hidden weights and random training batches are not identical.

[Configuration](../../../experiments/complex/low_rank_coarse_motion.json) ·
[Original 32k baseline](../low_rank_multicycle/steps_32000/summary.json) ·
[Validation](verification.json) · [Test log](validation.log).

## Final results

All branch comparisons below use all 378 unseen midpoint frames. RRMSE is the
existing global relative L2 error; motion error is backward-displacement endpoint
RMSE over the true foreground. Ground truth is used only for evaluation.

| Metric | Original grid | Coarse motion grid |
|---|---:|---:|
| Training image RRMSE | 1.116% | 2.504% |
| Midpoint image RRMSE | 1.877% | 3.141% |
| Midpoint reference intensity RRMSE | 68.656% | 55.064% |
| Midpoint foreground motion RMSE | 5.484 px | 4.339 px |
| Fixed-sample training MSE, final step | 1.09396e-6 | 5.43553e-6 |
| Fixed-sample validation MSE, final step | 2.83704e-6 | 7.99252e-6 |
| Parameters | 390,934 | 241,942 |

The coarse model has approximately 38.1% fewer parameters. Motion error decreases
by approximately 20.9%, and intensity error by 19.8%, while midpoint image error
increases. The intensity-only images remain visibly blurred. The existing auxiliary
finite-difference nonpositive-Jacobian fraction is 29.85%, versus 32.21%; it is not
a training objective or a substitute for branch accuracy.

### Training trajectory

| Coarse-model checkpoint | Midpoint image RRMSE | Intensity RRMSE | Motion RMSE |
|---|---:|---:|---:|
| 8,000 | 5.427% | 52.514% | 4.134 px |
| 16,000 | 3.802% | 54.633% | 4.289 px |
| 24,000 | 3.285% | 54.276% | 4.339 px |
| 32,000 | 3.141% | 55.064% | 4.339 px |

Image fitting improves with longer training, but branch accuracy does not improve
along this trajectory. The 8k entry is an intermediate checkpoint of the 32k
schedule, not a separate 8k-budget run. No checkpoint is selected using test labels.

[Loss comparison](loss_comparison.png) · Branch trajectories (historical reference; artifact not retained) ·
[Full checkpoint metrics](checkpoint_metrics.json) · [Comparison provenance](comparison.json).

### Matched query cost

The original and coarse models take **25.61 ms** and **26.13 ms**, respectively,
for a matched 4096-query forward/backward batch on CPU with four threads. These
are medians of six alternating rounds of 25 repetitions, after ten warmup batches
per model. The coarse/original ratio is 1.020; this measurement shows no speedup.
The number of levels, eight-corner interpolation and MLP widths are unchanged,
so the smaller parameter table does not reduce all per-query work.

This excludes optimizer updates and the acquisition forward model and is not an
end-to-end PACT timing claim. The coarse training loop took 911.3 seconds; the
earlier baseline took 858.0 seconds under different concurrent CPU workloads,
so those full-run times are not a controlled speed comparison. See
[timing samples and scope](timing.json) and [timing log](timing.log).

## Saved views and validation

- Image prediction (historical reference; artifact not retained).
- Reference intensity: truth, prediction, absolute error (historical reference; artifact not retained).
- Motion effect against truth (historical reference; artifact not retained): true and learned
  motion both query the same true time-varying intensity; the third panel is error.
- Motion-only effect (historical reference; artifact not retained): freeze learned intensity
  at the reference time; the first panel is intentionally static and the third
  panel is change from that reference, not ground-truth error.
- [Loss and learning-rate curves](steps_32000/plots/loss_curve.png),
  [numeric history](steps_32000/loss.csv), and model checkpoint (historical reference; artifact not retained).

All 60 unit tests and a three-step end-to-end smoke run passed. The trained model
has exactly zero motion at all 4,096 reference-time spatial points. All 511 sampled
reloaded predictions exactly match the saved arrays. Coordinate gradients are finite
and spatial gradients are nonzero. Rank-2 tails are approximately 3.58e-8 in float32.
The final model and final intermediate checkpoint match exactly. Source hashes,
fixed monitoring indices, disjoint training/test times, MSE-only histories and
learning-rate endpoints have been checked.

## Reproduction

Run from the repository root using the existing `nf-crt` environment. Choose a
fresh output directory to rerun training; the runner refuses to overwrite results.

```bash
python -m motion_static.run --phantom complex --model nested_low_rank_hash_nf \
  --config motion_static/experiments/complex/low_rank_coarse_motion.json \
  --out motion_static/results/complex/low_rank_coarse_motion/steps_32000

python -m motion_static.evaluation.compare_training \
  motion_static/results/complex/low_rank_multicycle/steps_32000 \
  motion_static/results/complex/low_rank_coarse_motion/steps_32000 \
  --out motion_static/results/complex/low_rank_coarse_motion \
  --vary-motion-grid --labels original_grid coarse_grid

python -m motion_static.evaluation.benchmark_query_cost \
  motion_static/results/complex/low_rank_multicycle/steps_32000/model.pt \
  motion_static/results/complex/low_rank_coarse_motion/steps_32000/model.pt \
  --labels original_grid coarse_grid \
  --out motion_static/results/complex/low_rank_coarse_motion/timing.json
```

Run the timing comparison after all training and evaluation jobs finish. It uses
matched continuous queries and alternating measurement order, without optimizer
updates. It measures nested-model forward/backward cost, excluding Adam, data
loading, monitoring, plotting and the future PACT forward operator. Prior full-run
timings involved concurrent jobs and are not used to claim an end-to-end speedup.

## Scope

This is one seed on an image-supervised toy object. Any improvement here requires
further validation on real objects and PACT measurements. Coarser spatial features
can suppress useful local motion as well as unwanted variation; image quality and
both branch errors must be evaluated together. Existing metric definitions and
reference-time anchoring are retained.
