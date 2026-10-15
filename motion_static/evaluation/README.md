# Reading the evaluation code

For formulas, units, aggregation conventions, and the distinction between existing
metrics and proposals, see [METRICS.md](../METRICS.md).

Training lives in `../run.py`. These modules read predictions and evaluate them.

| Goal | Start here |
|---|---|
| Run the two-branch midpoint diagnosis | [diagnose_branches.py](diagnose_branches.py): `diagnose_run()` |
| Measure residual geometry in intensity | [branch_metrics.py](branch_metrics.py): `fit_intensity_geometry()` |
| Measure motion magnitude, direction, and error | [branch_metrics.py](branch_metrics.py): `motion_statistics()` |
| Generate the branch GIFs and curves | [branch_plots.py](branch_plots.py): `save_branch_plots()` |
| Compare the complex temporal-loss experiment | [compare_temporal.py](compare_temporal.py): `compare()` |
| Inspect the original image/ROI/curve metrics | [metrics.py](metrics.py) |
| Evaluate the original midpoint experiment | [evaluate.py](evaluate.py) |
| Evaluate held-out blocks | [block_evaluate.py](block_evaluate.py), [masked_metrics.py](masked_metrics.py) |
| Check the low-rank intensity matrix and create motion-only GIFs | [evaluate.py](evaluate.py): `intensity_rank_report()`, `predict_fields()` |
| Compare training budgets and checkpoint branch errors | [compare_training.py](compare_training.py) |
| Save loss and learning-rate curves during every training run | [../training_monitor.py](../training_monitor.py) |

`compare_training.py` reuses the existing full-grid image/reference RRMSE and
foreground motion/Jacobian calculations. It also saves `plots/motion_effect.gif`
for each compared run: true motion versus learned motion, with identical true
time-varying intensity in both panels. The true object moves in the first panel.
An optional previous one-cycle model is compared only at common first-cycle times
that are held out for every model. It is never extrapolated to later cycles.
Loss curves use fixed monitoring subsets; branch convergence uses complete grids.
Different cosine training horizons are explicitly labeled; the long run is not a
continuation of the short run's zero-learning-rate endpoint.

## Low-rank branch displays

The R=2 and R=3 runs use the same branch diagnosis command below. Pass
`results/complex/low_rank/rank_2` and `rank_3` as the run folders.

Each new nested run also saves two separate displays:

- `plots/reference_intensity.gif`: fixed reference coordinates, with time varying
  only through learned intensity. The first panel supplies toy ground truth for comparison.
- `plots/motion_only.gif`: freeze the learned intensity at the configured motion
  reference time, comparing `C_pred(r,t0)` with `C_pred(r+u_pred(r,t),t0)`.
  All time dependence comes from motion. The third panel shows change, not error.
  This view needs no ground-truth reference image and can be reused with real objects.

The existing `branch_diagnostics/.../motion_effect.gif` instead applies learned
motion to the **true, time-varying** intensity. It is a toy accuracy diagnostic,
not an isolated motion-only animation. Both uses retain the backward-warp convention.

`metrics/intensity_rank.json` stores the singular values of the **uncentered**
space-by-time matrix on a fixed reference grid and its relative Frobenius tail
after R components. Finite-precision predictions can have tiny nonzero tails.
The warped observation-space image sequence is not constrained to rank R.

## Run the midpoint branch diagnosis

Run from the repository root. Pass completed **nested-model experiment folders**:

```bash
python -m motion_static.evaluation.diagnose_branches \
  motion_static/results/complex/nested_mlp \
  motion_static/results/complex/nested_hash_nf \
  --out motion_static/results/complex/midpoint_branch_diagnostics
```

For one experiment, pass just one folder. It also supports the simple phantom:

```bash
python -m motion_static.evaluation.diagnose_branches \
  motion_static/results/simple/nested_mlp \
  --out motion_static/results/simple/midpoint_branch_diagnostics
```

The command reads `config.json` and `arrays/predictions.npz`; it never trains or
loads a model into an optimizer. It uses the saved midpoint times and configured
image size, so paths, grid resolution, and frame count are not fixed in the code.
Block experiments and direct image fields do not have the required midpoint
branch predictions and are rejected with a clear message.

The output contains one folder per input run, a `summary.json`, and a generated
`README.md`. Repeating the command refreshes those diagnostic artifacts. Use a
new `--out` to preserve a previous diagnosis. Source predictions and configuration
are hashed before and after the run; they are only read.

## What is measured

For the complex temporal-L2 experiment, run the same diagnosis and compare it
with the saved original diagnosis. No vector-arrow plots are generated.

```bash
python -m motion_static.evaluation.diagnose_branches \
  motion_static/results/complex/temporal_l2/nested_hash_nf \
  --out motion_static/results/complex/temporal_l2/branch_diagnostics
python -m motion_static.evaluation.compare_temporal \
  motion_static/results/complex/midpoint_branch_diagnostics/nested_hash_nf \
  motion_static/results/complex/temporal_l2/branch_diagnostics/nested_hash_nf \
  --out motion_static/results/complex/temporal_l2
```

The comparison includes image RRMSE/SSIM/ROI, motion error, residual intensity
geometry, and the two brightness-gain curves estimated by the known-shape fit.
Brightness estimates are diagnostic approximations; the fit residual is reported.

1. **Intensity alone:** inspect `C_pred(q,t)` on fixed coordinates. Fit translation,
   x/y size, and independent brightness gains to quantify remaining geometry changes.
2. **Motion alone:** inspect the saved `(ux,uy)` output, its magnitude, its direction
   agreement, and its error against true backward displacement.
3. **Component swaps:** compare the full image, motion disabled, predicted motion
   with true intensity, and true motion with predicted intensity.

The image comparisons score all saved midpoint pixels. Motion metrics use true
object foreground. Geometry shift and size are relative to the first evaluated
midpoint, which differs from the network's zero-motion reference time.

The geometry fit is **specific to known toy shapes**, not a general optical-flow
method. Complex uses two brightness gains; simple uses one. Calibration first fits
the brightness-changing stationary reference and the true moving object. The
saved fit residual and `geometry_fit.gif` expose mismatch when learned intensity
does not match the assumed shape.

## Reuse the calculations

Functions take arrays and coordinates explicitly; importing the modules does not
read result files, create output directories, or initialize plotting.

```python
from motion_static.evaluation.branch_metrics import motion_statistics

# Arrays: truth/prediction (y, x, t, 2), foreground (y, x, t).
scores = motion_statistics(truth, prediction, foreground, spacing=(dx, dy))
```

`fit_intensity_geometry(volume, x, y, render, initial, bounds)` accepts a callable
that renders a known shape. Parameters start with `[tx,ty,sx,sy]`, followed by
brightness parameters. Supply a different renderer and explicit fit bounds for
another known shape. `toy_geometry_template()` contains the current toy choices;
the complex geometry itself is shared with `phantoms/complex.py:reference_shapes()`.

`branch_plots.py` only draws from supplied arrays and metrics. Change figure
appearance there without changing the numerical diagnosis.
