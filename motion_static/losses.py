"""Training objective and individual loss terms, shared by all model variants."""
import torch


def compute_training_losses(model, coordinates, targets, config):
    """Return scalar tensors for one observed batch, retaining their gradients.

    coordinates: (batch, 3) physical (x, y, t) coordinates.
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
            time_max=config["coordinate_bounds"][2][1],
        )

    total_loss = image_mse + temporal_weight * temporal_l2
    return {
        "batch_mse": image_mse,
        "temporal_l2": temporal_l2,
        "total_loss": total_loss,
    }


def intensity_temporal_l2(model, xyt, delta, time_max):
    """Penalize intensity's time derivative at a fixed, detached reference point.

    delta uses the same time units as xyt. No motion or intensity labels are used.
    Real contrast changes also pay this penalty; it is a preference, not a guarantee
    of correct motion separation.
    """
    with torch.no_grad():
        earlier = xyt.detach().clone()
        earlier[:, :2] += model.motion(xyt)
        # Use a backward pair near the final time, without querying past the data.
        earlier[:, 2].clamp_(max=time_max - delta)
        later = earlier.clone()
        later[:, 2] += delta

    # Only time changes between these queries. Gradients update intensity only.
    rate = (model.intensity(later) - model.intensity(earlier)) / delta
    return rate.square().mean()
