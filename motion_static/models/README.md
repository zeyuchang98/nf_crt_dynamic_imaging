# Reading the models

Start with the main model's `forward()` near the top of each file.
Inputs are rows of `(x, y, t)`, with shape `(N, 3)`.

| File | What to read here |
|---|---|
| [nested_hash_nf.py](nested_hash_nf.py) | Two hash fields: `intensity(r + motion(r, t), t)` |
| [nested_low_rank_hash_nf.py](nested_low_rank_hash_nf.py) | Existing motion hash field + low-rank reference intensity |
| [low_rank_intensity.py](low_rank_intensity.py) | 2D spatial hash basis and time MLP; reused directly by the motion-free baseline |
| [time_grid.py](time_grid.py) | Optional low-rank time network: dense 1D multi-resolution grid + MLP, R outputs |
| [oracle_branches.py](oracle_branches.py) | Torch copies of the complex phantom's true motion and reference intensity, for oracle diagnostics only |
| [nested_mlp.py](nested_mlp.py) | The same composition using two coordinate MLPs |
| [hash_nf.py](hash_nf.py) | One hash field: coordinate scaling, encoding, then MLP decoding |
| [coordinates.py](coordinates.py) | Fixed dataset bounds, normalization to [0, 1] or [-1, 1] |
| [hash_grid.py](hash_grid.py) | 2D/3D grid setup, table lookup, and bilinear/trilinear interpolation |

## Where to find a specific operation

- **Coordinate normalization:** `CoordinateNormalizer` in `coordinates.py`, called by the hash field and MLP branches.
- **Adding motion to position:** `NestedMLP.forward` and `NestedHashNF.forward`. Only `(x, y)` moves; `t` stays the same.
- **Zero motion at the reference time:** `MotionMLP.forward` and `MotionHashNF.forward`.
- **Network widths and activations:** each model's `__init__`.
- **Grid resolutions and table sizes:** `HashGridEncoding.__init__`.
- **Interpolation weights:** `HashGridEncoding.forward`, step 3.
- **Dense indexing and hashing:** `HashGridEncoding._table_indices`.
- **Loss formula:** [../losses.py](../losses.py), in `compute_training_losses()`.
- **Optimization:** [../run.py](../run.py), in `build_optimizer()` and `train()`.

The model code assumes valid experiment settings and `(N, 3)` coordinate inputs.
Boundary handling remains necessary: training includes the domain endpoints,
and nested intensity queries can move outside the original image.

## Low-rank intensity for motion-aware reconstruction

`NestedLowRankHashNF` represents `I(r,t) = sum_k b_k(r + u(r,t)) a_k(t)`.
`LowRankIntensity.spatial_basis(xy)` returns `(N,R)` spatial features from a 2D
hash grid and MLP. `temporal_coefficients(times)` takes `(N,1)` physical times,
normalizes them to `[-1,1]`, and returns `(N,R)` coefficients using a Tanh MLP.
The spatial branch never receives time. The final inner product has no additional
nonlinearity or time-dependent decoder. Thus the uncentered reference-coordinate
space-by-time matrix has rank at most R, including any static component.

The existing motion network, backward displacement convention, and zero-motion
reference anchor are retained. Spatial queries outside the original FOV retain
coordinate gradients. The original 3D encoder remains the default, with the same
parameter layout and indices for old checkpoints.

These models have no phantom-specific spatial shapes, trajectories, intensity
curves, or labels. They accept arbitrary fixed x/y/time bounds and return a
differentiable scalar field for a future PACT forward operator. The current
runner still uses direct image supervision; PACT measurement reconstruction is
not implemented by these representation experiments. R=2 and R=3 are candidate
capacities, not established rank bounds for real PACT objects. The factor basis
is nonunique and individual basis images need not correspond to physical parts.

### Motion-free low-rank baseline

The runner's `low_rank_hash_nf` option instantiates `LowRankIntensity` directly:
`I(r,t) = sum_k b_k(r) a_k(t)`. It has no motion module, warp or motion reference
anchor. Its spatial bases and time coefficients must represent the complete
moving image, including both geometry and brightness changes.

It uses the same image MSE, spatial hash, temporal MLP and optimizer rates as the
low-rank branch of the nested model. Its rank bound applies to the image matrix
at fixed observation coordinates. Evaluation reports image metrics and
`image_rank`; motion and recovered reference-intensity metrics are absent.
This baseline is not supervised with true reference-intensity labels.

### Independent motion grid resolution

For `NestedLowRankHashNF`, optional `motion_encoding` settings override the shared
encoding settings only for motion. `min_resolution` and `max_resolution` accept
either scalar cell counts or one count per coordinate axis. For example,
`[2, 2, 8]` to `[32, 32, 128]` uses coarser x/y cells while retaining the original
8-to-128 temporal grid. Intensity continues using its own 2D 8-to-128 grid.
Scalar settings retain the original state-dictionary layout and table indexing.

The coarser grid interpolates hash features before the existing MLP decoder;
it does not directly interpolate displacement control points or guarantee a
band-limited or invertible motion field. The layer count, feature width, hash
capacity limit, backward warp, and reference anchor remain unchanged. Fewer
vertices change table sizes and random-number consumption, so an identical seed
does not imply identical motion hidden weights or training batches. Intensity
is initialized first and its initial parameters remain identical at the same seed.

## Reuse with a different coordinate domain

Pass fixed bounds in the same units as your input coordinates:

```python
from motion_static.models.nested_hash_nf import NestedHashNF

bounds = ((-20, 20), (-15, 15), (0, 2))  # x in mm, y in mm, t in seconds
model = NestedHashNF(coordinate_bounds=bounds, reference_time=0)
```

`HashNeuralField` and `NestedMLP` accept the same `coordinate_bounds` argument.
The default bounds remain `(-0.5, 0.5)` on each axis for existing toy experiments.

- Hash inputs use `(coordinate - minimum) / (maximum - minimum)`.
- Coordinate MLPs map the same domain to `[-1, 1]`.
- Bounds stay fixed across batches and frames; displaced queries use the same bounds.
- Motion networks predict fractions of the spatial extent, then scale `(ux, uy)`
  back to input units before adding displacement to position.
- `reference_time` is in input time units. If omitted, it is the lower time bound.

`run.py` derives bounds from the complete training coordinate axes unless they
are supplied in the experiment config. New checkpoints save `coordinate_bounds`
and the resolved reference time in `config`. Always save and reload **config plus
state_dict**, as the training entry point does; weights alone do not describe the
coordinate domain. Older toy checkpoints still load with their original domain.
