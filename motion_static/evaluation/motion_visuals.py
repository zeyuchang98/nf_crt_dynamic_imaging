"""Visualize motion on a shared true reference; no learned intensity or metrics."""
import numpy as np

from ..phantoms.data import coordinate_grid, get_phantom


def frozen_true_motion(data, predicted_motion, config, *, reference_time=None):
    """Return I_true(r + u_true(r,t), t0) and I_true(r + u_pred(r,t), t0).

    Displacements retain their observation times. Only the true intensity's
    time argument is frozen; the reference is evaluated continuously, not
    interpolated from a previously rendered reference image.
    """
    if reference_time is None:
        reference_time = config.get("reference_time")
    if reference_time is None:
        reference_time = config.get("coordinate_bounds", [[-.5, .5]] * 3)[2][0]
    phantom = get_phantom(config["phantom"])
    coordinates = coordinate_grid(data["x"], data["y"], data["t"])
    coordinates[:, 2] = float(reference_time)
    shape = data["truth"].shape
    images = []
    for displacement in (data["u_true"], predicted_motion):
        if displacement.shape != (*shape, 2):
            raise ValueError("Motion must have shape (height, width, times, 2)")
        warped = coordinates.copy()
        warped[:, :2] += displacement.reshape(-1, 2)
        images.append(phantom.reference_intensity(warped, config["size"]).reshape(shape))
    return tuple(images)
