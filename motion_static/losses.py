"""Training objective and individual loss terms, shared by all model variants."""
import torch


def compute_training_losses(model, coordinates, targets, config):
    """Return scalar tensors for one observed batch, retaining their gradients.

    coordinates: (batch, 3) physical (x, y, t) or (batch, 4) (x, y, z, t) coordinates.
    targets: (batch,) observed image intensities at those coordinates.

    Edit loss terms and their weighted sum here. Only ``total_loss`` is used for
    backpropagation; the other named terms are recorded for inspection. Validation
    labels and true motion/reference-intensity labels never enter this function.
    """
    predicted_image = model(coordinates)
    image_mse = (predicted_image - targets).square().mean()

    # This optional existing penalty is disabled in the default configuration.
    # Skip its forward passes entirely when its weight is zero.
    temporal_weight = config.get("temporal_weight", 0.)
    temporal_l2 = image_mse.new_zeros(())
    if temporal_weight > 0:
        temporal_l2 = intensity_temporal_l2(
            model,
            coordinates,
            delta=config["temporal_step"],
            time_max=config["coordinate_bounds"][-1][1],
        )

    total_loss = image_mse + temporal_weight * temporal_l2
    return {
        "batch_mse": image_mse,
        "temporal_l2": temporal_l2,
        "total_loss": total_loss,
    }


def compute_measurement_losses(images, predicted_data, measured_data, sigma, regularizer):
    """Objective of direct backpropagation through the CRT for a batch of whole frames.

    images: (K, size, size) predicted frames in the (y, x) layout; predicted_data = P_k f_k
    and measured_data d_k: (K, rows). `data_fidelity` is mean(((P f - d) / sigma)^2), about 1
    once the residual reaches the noise level. The objective is half of it (the
    1 / (2 sigma^2) least-squares term per measurement) plus weight * R(f), with R from
    image_regularizer. Only `total_loss` is differentiated.
    """
    data_fidelity = ((predicted_data - measured_data) / sigma).square().mean()
    regularization = image_regularizer(images, regularizer)
    total_loss = 0.5 * data_fidelity + regularizer.get("weight", 0.) * regularization
    return {"data_fidelity": data_fidelity, "regularization": regularization, "total_loss": total_loss}


def image_regularizer(images, regularizer):
    """Spatial penalty of (K, size, size) frames, from forward pixel differences.

    "squared_gradient" (ProxNF): mean(Dx f^2) + mean(Dy f^2), each mean over its differences.
    "tv" (Lozenski et al. 2022): mean over pixels of sqrt(Dx f^2 + Dy f^2 + epsilon^2),
    isotropic, on the (size - 1)^2 pixels that have both differences. "none": 0.
    Differences are per pixel, not per mm.
    """
    kind = regularizer.get("type", "none")
    if kind == "none":
        return images.new_zeros(())
    dx = images[:, :, 1:] - images[:, :, :-1]
    dy = images[:, 1:, :] - images[:, :-1, :]
    if kind == "squared_gradient":
        return dx.square().mean() + dy.square().mean()
    if kind == "tv":
        epsilon = regularizer["epsilon"]
        return (dx[:, :-1, :].square() + dy[:, :, :-1].square() + epsilon ** 2).sqrt().mean()
    raise ValueError(f"Unknown regularizer {kind!r}")


def intensity_temporal_l2(model, xyt, delta, time_max):
    """Penalize intensity's time derivative at a fixed, detached reference point.

    delta uses the same time units as xyt. No motion or intensity labels are used.
    Real contrast changes also pay this penalty; it is a preference, not a guarantee
    of correct motion separation.
    """
    with torch.no_grad():
        earlier = xyt.detach().clone()
        earlier[:, :-1] += model.motion(xyt)
        # Use a backward pair near the final time, without querying past the data.
        earlier[:, -1].clamp_(max=time_max - delta)
        later = earlier.clone()
        later[:, -1] += delta

    # Only time changes between these queries. Gradients update intensity only.
    rate = (model.intensity(later) - model.intensity(earlier)) / delta
    return rate.square().mean()
