"""Reproducible block holdouts on a (y, x, t) grid."""
import hashlib

import numpy as np


def block_split(shape, block_shape=(8, 8, 2), test_fraction=0.2, seed=0):
    """Hold out whole blocks, keeping temporal gaps short at every spatial tile.

    The first and last time blocks stay in training. Selected test time blocks
    never touch within one spatial tile. Selection uses coordinates, not labels.
    """
    shape, block_shape = np.asarray(shape), np.asarray(block_shape)
    if shape.shape != (3,) or block_shape.shape != (3,) or np.any(block_shape < 1):
        raise ValueError("shape and block_shape must contain three positive sizes (y, x, t)")
    if np.any(shape % block_shape):
        raise ValueError("Each grid size must be divisible by its block size")
    ny, nx, nt = (shape // block_shape).tolist()
    spatial_tiles = ny * nx
    total_blocks = spatial_tiles * nt
    test_count = round(total_blocks * test_fraction)
    max_per_tile = max(0, (nt - 1) // 2)
    if not 0 < test_count <= spatial_tiles * max_per_tile:
        raise ValueError("Requested test fraction cannot satisfy the temporal-gap rules")

    rng = np.random.default_rng(seed)
    counts = np.full(spatial_tiles, test_count // spatial_tiles)
    counts[rng.permutation(spatial_tiles)[:test_count % spatial_tiles]] += 1
    test_blocks = np.zeros((ny, nx, nt), dtype=bool)
    for tile, count in enumerate(counts):
        # Sample a compressed timeline, then insert one train block between
        # selected indices. This samples nonadjacent subsets without retries.
        compressed = np.sort(rng.choice(nt - 1 - count, size=count, replace=False))
        selected = 1 + compressed + np.arange(count)
        test_blocks[tile // nx, tile % nx, selected] = True

    test_mask = test_blocks
    for axis, size in enumerate(block_shape):
        test_mask = np.repeat(test_mask, size, axis=axis)
    settings = dict(
        name="blocks", block_shape_yxt=block_shape.tolist(), seed=int(seed),
        requested_test_fraction=float(test_fraction), actual_test_fraction=float(test_mask.mean()),
        total_blocks=int(total_blocks), test_blocks=int(test_count),
        train_points=int((~test_mask).sum()), test_points=int(test_mask.sum()),
        protected_first_and_last_time_blocks=True, adjacent_test_time_blocks=False,
        max_missing_frames=int(block_shape[2]),
        max_bracketing_gap_in_frame_steps=int(block_shape[2] + 1),
        mask_sha256=hashlib.sha256(test_mask.tobytes()).hexdigest())
    return dict(train_mask=~test_mask, test_mask=test_mask, test_blocks=test_blocks, settings=settings)


def interpolate_training_frames(truth, train_mask, times):
    """Interpolate along time at each pixel, reading only observed training values."""
    if not train_mask[..., 0].all() or not train_mask[..., -1].all():
        raise ValueError("Temporal interpolation requires observed endpoints")
    times = np.asarray(times)
    prediction = np.empty(truth.shape, dtype=float)
    for y, x in np.ndindex(truth.shape[:2]):
        observed = train_mask[y, x]
        prediction[y, x] = np.interp(times, times[observed], truth[y, x, observed])
    return prediction
