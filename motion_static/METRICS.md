# Evaluation Metrics: Definitions and Implementation Status

Last reviewed: 2026-10-14. Implementation audit: local code at `9eec337`.

This is the central reference for metric definitions and their implementation
status. **Documenting a recommendation does not adopt a new evaluation protocol.**
Existing formulas, output files, and historical results are unchanged. Proposed
aggregation, units, and dynamic-intensity metrics below require agreement before
implementation or retrospective evaluation. See [PROGRESS.md](PROGRESS.md) for
experiment results and [evaluation/README.md](evaluation/README.md) for code usage.

## 1. What is evaluated

The two neural fields and their composed image are

$$
\hat{\boldsymbol d}(r,t)=\phi_{\mathrm{motion}}(r,t),\qquad
\hat J(q,t)=\phi_{\mathrm{intensity}}(q,t),
$$

$$
q=r+\hat{\boldsymbol d}(r,t),\qquad
\hat I(r,t)=\hat J\bigl(r+\hat{\boldsymbol d}(r,t),t\bigr).
$$

| Object | Prediction and reference | Interpretation |
|---|---|---|
| Full image | $\hat I(r,t)$ versus $I(r,t)$ | Accuracy of the complete reconstruction |
| Reference intensity | $\hat J(q,t)$ versus $J(q,t)$ at fixed reference coordinates | Accuracy of intensity independent of an applied motion warp |
| Motion | $\hat{\boldsymbol d}(r,t)$ versus $\boldsymbol d(r,t)$ | Accuracy of the backward displacement from current to reference coordinates |

The code names the displacement components `ux` and `uy`; they correspond to
$d_x$ and $d_y$ here. Displacement is not velocity or adjacent-frame optical flow.
Comparisons require the same coordinate system, reference time, and warp direction.
The current motion anchor is $\hat{\boldsymbol d}(r,t_0)=0$, with $t_0=-0.5$.

**Confirmed assessment rule, 2026-10-14:** quantitative motion accuracy compares
predicted and ground-truth displacement fields directly. Applying motion to the
true intensity is for visualizing its effect, not for measuring or ranking motion
accuracy through image error. The existing code and historical results still
contain predicted-motion/true-intensity image errors; those are retained as legacy
diagnostics, not adopted as motion-accuracy metrics. This decision does not change
the evaluator or remove saved results.

For low-rank intensity,

$$
\hat J(q,t)=\sum_{k=1}^{R}b_k(q)a_k(t).
$$

Evaluate the resulting intensity field, not individual bases or coefficients:
their factorization is not unique. Rank applies to the uncentered reference-space
intensity matrix, not the warped image sequence.

All accuracy metrics below require a corresponding reference. On real PACT data,
an image reference alone does not supply ground-truth motion or reference
intensity. No dense branch-accuracy claim can be made without those references.

## 2. Recommended metric set and current implementation

These roles are recommendations for discussion, not a new ranking rule.

| Metric | Full image | Reference intensity | Motion | Current implementation |
|---|---|---|---|---|
| RRMSE, lower is better | Primary | Primary | Joint vector RRMSE | Implemented and saved |
| PSNR, higher is better | Primary | Primary | Separate component maps; auxiliary | Implemented and saved for `ux` and `uy` as well as scalar images |
| SSIM, higher is better | Primary | Primary | Separate component maps; auxiliary | Implemented, but existing protocols use different temporal aggregation |
| Mean endpoint error (EPE), lower is better | Not applicable | Not applicable | Primary absolute displacement error | Implemented in pixels |
| Endpoint RMSE, lower is better | Not applicable | Not applicable | Retain for error sensitivity and historical comparison | Implemented in pixels |
| Motion errors in mm or percent of FOV | Not applicable | Not applicable | Proposed reporting units | No standard output or physical-calibration interface yet |
| Mean of per-frame spatial SSIM | Proposed summary | Proposed summary | Proposed component summaries | Per-frame scores exist; this is not the standard midpoint summary field |
| Dynamic RRMSE, lower is better | Not proposed here | Candidate diagnostic | Not proposed here | Not implemented in the evaluation pipeline |

Motion PSNR and SSIM compare signed component maps in a chosen coordinate system.
They are auxiliary image-style diagnostics, not substitutes for geometric vector
errors. Do not compute any numerical metric from a GIF, color map, or arrow plot.

## 3. Formulas

Let $\Omega$ be the selected set of space-time samples, $N=|\Omega|$, and let
$F_i,\hat F_i$ be reference and predicted values. All sums use this same selection.
For vector fields, squared Frobenius norms sum over components as well as samples.
The formulas below assume equally weighted samples, matching the current code.

### 3.1 Relative root mean squared error (RRMSE)

$$
\operatorname{RRMSE}(\hat F,F)
=\frac{\|\hat F-F\|_F}{\|F\|_F}.
$$

Report $100\operatorname{RRMSE}$ as a percentage. JSON stores the ratio, not the
percentage. In particular, motion uses

$$
\operatorname{RRMSE}_{\mathrm{motion}}
=\sqrt{\frac{\sum_{i\in\Omega}
\|\hat{\boldsymbol d}_i-\boldsymbol d_i\|_2^2}
{\sum_{i\in\Omega}\|\boldsymbol d_i\|_2^2}}.
$$

Both components enter one numerator and denominator; this is not the average of
component RRMSEs. If reference energy is zero, RRMSE is undefined, including when
both reference and prediction are zero. Near-zero energy can make it unstable.

### 3.2 Mean squared error and PSNR

For a scalar field or one displacement component,

$$
\operatorname{MSE}=\frac1N\sum_{i\in\Omega}(\hat F_i-F_i)^2,
\qquad \operatorname{RMSE}=\sqrt{\operatorname{MSE}},
$$

$$
\operatorname{PSNR}
=10\log_{10}\left(\frac{\Delta_F^2}{\operatorname{MSE}}\right)
\quad[\mathrm{dB}].
$$

$\Delta_F>0$ is the declared data range, with the same units as $F$. All methods
in a comparison must use the same range. Do not independently rescale predictions
or frames, and do not clip predictions before evaluation. Exact predictions have
PSNR $+\infty$. Displacement PSNR is reported separately for $d_x$ and $d_y$.

Current sequence PSNR uses the MSE pooled across all selected samples. It is not
the arithmetic mean of per-frame PSNR values. Sequence RRMSE is likewise computed
from pooled energies, not the mean of framewise RRMSEs.

### 3.3 Structural similarity (SSIM)

For a local spatial window, the standard SSIM form is

$$
\operatorname{SSIM}(F,\hat F)
=\frac{(2\mu_F\mu_{\hat F}+C_1)(2\sigma_{F\hat F}+C_2)}
{(\mu_F^2+\mu_{\hat F}^2+C_1)(\sigma_F^2+\sigma_{\hat F}^2+C_2)},
$$

$$
C_1=(0.01\Delta_F)^2,\qquad C_2=(0.03\Delta_F)^2.
$$

$\mu$ denotes local mean, $\sigma^2$ variance, and $\sigma_{F\hat F}$ covariance.
Average local scores over the declared valid window centers to obtain a frame
score. Window size, weighting, covariance convention, and boundary treatment are
part of the metric definition; specifying only "SSIM" is insufficient.

**Proposed sequence summary:** average spatial frame scores with equal time weights,

$$
\overline{\operatorname{SSIM}}_{\mathrm{spatial}}
=\frac1T\sum_{t=1}^{T}\operatorname{SSIM}_{\mathrm{spatial}}(F_t,\hat F_t).
$$

This has not replaced the existing midpoint SSIM. How to handle frames with no
valid windows must be agreed before applying this summary to masked evaluations.
Motion components remain separate; no combined vector SSIM is defined here.

### 3.4 Mean endpoint error and endpoint RMSE

Define the displacement error length at each selected point:

$$
e_i=\|\hat{\boldsymbol d}_i-\boldsymbol d_i\|_2
=\sqrt{(\hat d_{x,i}-d_{x,i})^2+(\hat d_{y,i}-d_{y,i})^2}.
$$

$$
\operatorname{Mean\ EPE}=\frac1N\sum_{i\in\Omega}e_i,
\qquad
\operatorname{Endpoint\ RMSE}=\sqrt{\frac1N\sum_{i\in\Omega}e_i^2}.
$$

Endpoint RMSE gives larger errors more weight. The historical "Motion RMSE"
means endpoint RMSE, not mean EPE. There is no additional division by the number
of displacement components: in 2D it equals $\sqrt{2}$ times scalar RMSE after
flattening both components over the same selected points.

Both measures have displacement units. For isotropic spacing $h$ in coordinate
units per image sample interval, $e_i^{\mathrm{px}}=e_i/h$. More generally,

$$
e_i^{\mathrm{px}}
=\sqrt{\left(\frac{\hat d_{x,i}-d_{x,i}}{h_x}\right)^2
+\left(\frac{\hat d_{y,i}-d_{y,i}}{h_y}\right)^2}.
$$

The current 64-sample grid spans $[-0.5,0.5]$, so $h=1/63$; GIF rendering size
and hash resolution do not enter this conversion. Pixel errors change when image
sampling changes, even for the same physical displacement error.

**Proposed reporting:** use mm when a physical coordinate calibration is available;
otherwise report a declared fixed FOV length $L_{\mathrm{FOV}}$ normalization:

$$
\operatorname{EPE}_{\mathrm{FOV}\%}
=100\frac{\operatorname{Mean\ EPE}}{L_{\mathrm{FOV}}},\qquad
\operatorname{RMSE}_{\mathrm{FOV}\%}
=100\frac{\operatorname{Endpoint\ RMSE}}{L_{\mathrm{FOV}}}.
$$

Numerator and denominator must use the same length units. The choice of FOV
length (for example, horizontal width) remains to be confirmed. This normalization
removes the pixel-unit dependence, not differences caused by sampling, ROI, or
temporal coverage. Compare models on a common physical evaluation grid and times.

### 3.5 Candidate: dynamic intensity RRMSE

Global intensity error can improve while temporal variation is suppressed. On a
fixed reference-coordinate grid with the same evaluated times at every point,
remove each field's own temporal mean:

$$
J'(q,t)=J(q,t)-\frac1T\sum_{s=1}^{T}J(q,t_s),\qquad
\hat J'(q,t)=\hat J(q,t)-\frac1T\sum_{s=1}^{T}\hat J(q,t_s).
$$

$$
\operatorname{Dynamic\ RRMSE}
=\frac{\|\hat J'-J'\|_F}{\|J'\|_F}.
$$

Report as a percentage. If the prediction is constant over time and the reference
has nonzero temporal variation, the score is 100%. If the reference has no temporal
variation, the score is undefined. This diagnostic ignores time-constant bias,
so it complements rather than replaces ordinary intensity RRMSE. It is a proposed
project diagnostic, not a metric reported by Tian et al. The definition above does
not specify changing spatial selections or missing times; those require a separate
agreement before use. No pipeline implementation or saved result is claimed.

### 3.6 Lesion activity curve RRMSE (DCE-MOBY, embedding protocol)

Adopted on 2026-10-30, following the LAC of Lozenski et al. 2024 (Table II). For a
region with partial volume $w(r,p)\in[0,1]$ at respiratory phase $p$, taken from the
slice file's indicators at each frame's own phase $p(t)$,

$$
\operatorname{LAC}_f(t)=\frac{\sum_r w(r,p(t))\,f(r,t)}{\sum_r w(r,p(t))},\qquad
\operatorname{LAC\ RRMSE}=\frac{\|\operatorname{LAC}_{\hat f}-\operatorname{LAC}_{f}\|_2}{\|\operatorname{LAC}_{f}\|_2},
$$

over all scored frames. The mask follows the breathing motion, and boundary pixels count
by their partial volume. Three regions: the whole lesion (`tumor` + `tumor_core`), its
shell (`tumor`) and its core (`tumor_core`); the two indicators are complementary partial
volumes of the lesion. The pre-injection mean uses the frames with $t<0$ (time 0 is the
injection). Curves are reported in raw p0 units; the RRMSE does not depend on the scale.
This is not the `activity_curve` of `metrics.json`, which averages over the rectangular
ROI. Defined for the embedding protocol only.

## 4. Existing evaluation conventions to preserve

- **Temporal domain:** midpoint experiments separately evaluate training-grid
  times and unseen midpoint times. One-cycle and three-cycle results cover
  different domains. Compare them only on explicitly shared evaluation times.
- **Embedding protocol (dce_moby, approved 2026-10-25):** `held_out_split = "none"`
  fits and scores the same, complete set of window frames (`all_window_frames`), as in
  Lozenski et al. 2022. It measures representation only and is reported separately from
  the held-out protocols; their scores are not interchangeable.
- **Alternate held-out split (dce_moby, approved 2026-10-23):** for plateau-schedule
  runs the midpoints are split into validation frames (window frames 1, 5, 9, ...),
  which steer the learning rate, stop and kept weights, and test frames (3, 7,
  11, ...), which are only scored. The test frames are the reported held-out result;
  compare other runs on them with `evaluation.moby_split_eval`, not with their
  all-midpoint `unseen_time_midpoints` scores.
- **Regions:** scalar reports include the full FOV and a rectangular ROI. Each
  image/intensity ROI is the bounding box of positive truth over training and
  midpoint frames, expanded by two pixels and clipped to the image bounds.
  Motion-component ROIs use the observed image's ROI. Motion vector scores
  separately use the full FOV or true observed-object foreground. These selections
  are not interchangeable. Truth-derived ROIs are toy evaluation conventions.
- **Aggregation:** global scalar errors and motion endpoint errors pool selected
  space-time samples. Changing foreground size therefore changes frame weights.
- **Ranges:** image and reference-intensity reports use `data_range=1`.
  Midpoint `ux`/`uy` reports use each component's ground-truth max minus min across
  training and midpoint arrays, with a unit-range fallback for a constant component.
  Cross-dataset component PSNR comparisons need an agreed range convention.
- **Midpoint SSIM:** the top-level `ssim` is computed on the entire space-time
  array with time treated as another windowed dimension. Spatial `per_frame_ssim`
  is saved separately. Current settings are window size 7, uniform weights,
  sample covariance, `K1=0.01`, and `K2=0.03`. Windows do not extend outside the
  evaluated full array or ROI. A too-short axis returns `null`. Arrays above 2^24
  voxels (approved 2026-10-26, for the whole DCE-MOBY series) are scored in time
  chunks with 3-frame halos; this is the same whole-array value up to floating-point
  rounding (relative difference 1.8e-15 on 1,000 DCE-MOBY frames) and is noted in
  `settings.ssim.computation`. Smaller arrays use the single whole-array call unchanged.
- **Block SSIM:** only spatial 7-by-7 windows entirely inside the selected mask
  are valid. Overall SSIM averages valid centers across all frames; it is not
  necessarily an equal-weight frame average. Do not mix it with midpoint SSIM.
- **Special values:** scalar RRMSE with zero truth energy is saved as `"undefined"`;
  infinite PSNR as `"Infinity"`; unavailable window scores as `null`. Near-zero
  motion-component frames are suppressed only in RRMSE plotting, not silently
  removed from global errors. Empty-selection handling differs across helpers;
  `motion_scores()` assumes nonempty pooled selections and is not a general
  empty-mask API.
- **No display processing:** metrics use raw predictions. Fixed display clipping
  and GIF interpolation are visualization choices only.

Existing GIF definitions remain in [PROGRESS.md](PROGRESS.md). In particular,
`reference_intensity.gif` queries fixed reference coordinates without motion;
`motion_effect.gif` applies true/predicted motion to true intensity frozen at the
model's reference time t0, with absolute image error in the third panel (definition
updated on 2026-10-15); `motion_only.gif` freezes learned intensity at that time.
The frozen-reference GIF does not replace the existing time-varying intensity
branch-replacement scores in `diagnostics`; numerical metrics are unchanged.
None of these image displays is the displacement vector itself.

## 5. Code and output map

Paths below are relative to `motion_static/`; output paths are relative to a run.
The compact layout changes storage only. Its `schema_version` is a file-format
version, not adoption of the proposed standardized metric protocol.

| Calculation | Implementation | Saved output / key |
|---|---|---|
| Scalar MSE, RMSE, RRMSE, PSNR | [metrics.py](evaluation/metrics.py): `error_metrics()` | `metrics.json`: `image`, `reference_intensity`, `motion_components.{ux,uy}`; `rrmse`, `psnr_db`, `mse`, `rmse` |
| Sequence, frame, ROI, and SSIM evaluation | [metrics.py](evaluation/metrics.py): `evaluate_sequence()` | `{image,reference_intensity}.unseen_time_midpoints`, `roi`, `per_frame_ssim`, other `per_frame_*` |
| Populate image/intensity/component reports | [evaluate.py](evaluation/evaluate.py): `scalar_report()`, `evaluate()` | One `metrics.json` containing all scalar and branch reports |
| Mean EPE, endpoint RMSE, vector RRMSE | [evaluate.py](evaluation/evaluate.py): `motion_scores()` | `metrics.json`: `motion.midpoint.foreground` and `motion.midpoint.whole_fov`; `mean_endpoint_error_pixels`, `endpoint_rmse_pixels`, `vector_rrmse` |
| Motion diagnostics with per-axis pixel spacing | [branch_metrics.py](evaluation/branch_metrics.py): `motion_statistics()` | Branch-diagnostic summaries; component and endpoint RMSE in pixels |
| Selected-block errors and spatial-window SSIM | [masked_metrics.py](evaluation/masked_metrics.py): `evaluate_masked_sequence()` | Block-run reports, using their distinct mask protocol |
| Fixed-point and region-mean temporal curve errors | [metrics.py](evaluation/metrics.py): `evaluate_sequence()` | `point_curves`, `activity_curve`, including curve `rrmse` |
| DCE-MOBY lesion activity curves and LAC RRMSE (Section 3.6) | [moby_lesion_curves.py](evaluation/moby_lesion_curves.py): `curves_for()`, `scores()` | Separate output folder: `lesion_curves.json` (`scores`, `ss_check`), `lesion_curves.npz`, `lesion_curves.png` |

`motion_scores()` uses one scalar pixel spacing and is currently used on isotropic
toy grids. The separate branch diagnostic supports `(dx,dy)` spacing. Neither
function currently standardizes physical mm or FOV-percent output. Existing
point/mean curve RRMSE is not the demeaned dynamic-intensity metric in Section 3.5.

Implementation checks on 2026-10-14:

- Inspected the functions and their callers above; no metric code was changed.
- Before artifact cleanup, confirmed the early R=2 run stored all four scalar
  reports and motion vector scores. Those old files were subsequently deleted
  at the user's request; this is a historical implementation audit.
- Ran 11 existing focused tests: eight scalar/sequence metric tests, one endpoint
  pixel-unit/foreground test, and two branch motion tests. All passed. These do
  not validate unimplemented proposals or claim exhaustive coverage.
- No training or retrospective rewriting of experiment metrics was performed.

## 6. Decisions still needed before a new common protocol

1. Confirm per-frame spatial SSIM averaging, window settings, and missing-window
   handling; preserve old space-time and block SSIM under their original meanings.
2. Confirm the primary motion unit, physical calibration or FOV normalization,
   and the role/ranges of auxiliary component PSNR and SSIM.
3. Confirm the dynamic-intensity diagnostic, its spatial selection, and time set.
4. Fix common evaluation coordinates, times, regions, and output protocol metadata
   before comparing previously unmatched runs. Historical JSON files do not yet
   carry a unified protocol-version field; do not relabel them as compliant.

## 7. Literature context

- **Tian et al., "Unsupervised Motion-Compensated Decomposition for Cardiac MRI
  Reconstruction via Neural Representation," AAAI-26.** Reviewed the user-supplied
  `00592-AAAI26.TianX-CV.pdf`, especially PDF pages 5-7. Section 5.3/Table 1 report
  reconstructed-image PSNR, SSIM, and cardiac-ROI nRMSE. Section 6.3/Figure 5 and
  Section 6.5/Figure 7 assess DVFs qualitatively through deformation grids and
  arrows. This PDF does not report ground-truth DVF EPE/RMSE/RRMSE, and does not
  specify the nRMSE denominator. Its static canonical image does not address our
  time-varying reference-intensity target; intensity changes are mentioned as a
  future challenge. [Author code repository](https://github.com/MeijiTian/MoCo-INR).
- **Wang et al., "Image Quality Assessment: From Error Visibility to Structural
  Similarity," 2004.** Source for SSIM, not for the proposed temporal aggregation
  or signed-motion interpretation. [Paper](https://www.cns.nyu.edu/pub/Lcv/wang03-reprint.pdf).
- **Middlebury optical-flow evaluation.** Uses average endpoint error as a motion
  accuracy measure. This supports the metric family, not a claim that our backward
  reference-displacement task has the same protocol as optical flow or all PACT
  studies. [Official evaluation](https://vision.middlebury.edu/flow/eval/results/results-e1.php).
