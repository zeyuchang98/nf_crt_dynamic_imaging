"""Small training losses shared by the nested MLP and nested Hash NF."""
import torch


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
