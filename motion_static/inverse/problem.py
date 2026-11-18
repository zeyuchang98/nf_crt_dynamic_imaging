"""The 2D inverse problem of one run: measured data, per-frame operators and the pixel grid.

`InverseProblem` holds what reconstruction may use (noisy data d_k, sigma, operators P_k,
pixel coordinates, frame times) and, separately, the normalized truth, which only the
monitor and the final scoring read. The measurement file must come from the run's own
data window (same file hash, frames and intensity scale) and name the operator it used.
"""
from pathlib import Path

import numpy as np
import torch

from ..phantoms import moby
from .crt_operator import load_operator
from .frame_operator import FrameOperator
from .measurements import load_measurements


def check_measurements(metadata, data):
    """Raise unless the measurement file was simulated from this run's data window."""
    expected = dict(sha256=data["sha256"], intensity_scale_value=data["intensity_scale_value"])
    for key, value in expected.items():
        if metadata["data"][key] != value:
            raise ValueError(f"Measurements have data {key} {metadata['data'][key]}, the run {value}")
    frames = moby.window_frames(data)
    if metadata["frames"] != [int(frames[0]), int(frames[-1])]:
        raise ValueError(f"Measurements cover frames {metadata['frames']}, the run {frames[0]}-{frames[-1]}")


class InverseProblem:
    def __init__(self, config, device, adjoint=False):
        self.device = torch.device(device)
        arrays, self.metadata = load_measurements(config["inverse"]["measurements"])
        check_measurements(self.metadata, config["data"])
        operator = self.metadata["operator"]
        matrix, saved = load_operator(operator["file"])  # verifies the arrays against their hash
        if saved["operator_sha256"] != operator["operator_sha256"]:
            raise ValueError("The operator file is not the one the measurements were simulated with")
        self.operator = FrameOperator(matrix, arrays["view_degrees"], operator["num_rings"],
                                      operator["angle_step_deg"], self.device, adjoint=adjoint)
        # The scipy operator and NumPy data, for the LSQR warm start (inverse/warm_start.py).
        self.matrix, self.noisy, self.view_degrees = matrix, arrays["noisy"], arrays["view_degrees"]
        self.num_rings, self.angle_step_deg = operator["num_rings"], operator["angle_step_deg"]
        self.sigma = float(self.metadata["noise"]["sigma"])
        self.frames = arrays["frames"]
        self.times = torch.as_tensor(arrays["times"], dtype=torch.float32, device=self.device)
        self.data = torch.as_tensor(arrays["noisy"].reshape(len(self.frames), -1), device=self.device)

        entry = moby.read_window_entry(config["data"])
        if not np.array_equal(entry["t"], arrays["times"]):
            raise ValueError("Measurement times differ from the window times")
        self.size = len(entry["x"])
        # Pixel (iy, ix) at flat position iy * size + ix, the operator's column order.
        y, x = np.meshgrid(entry["y"], entry["x"], indexing="ij")
        self.xy = torch.as_tensor(np.stack([x.reshape(-1), y.reshape(-1)], axis=1), dtype=torch.float32,
                                  device=self.device)
        # Truth for monitoring and scoring only, (frames, size**2) x-fastest.
        self.truth = np.moveaxis(entry["truth"], -1, 0).reshape(len(self.frames), -1)

    def __len__(self):
        return len(self.frames)

    def coordinates(self, frames):
        """(K * size**2, 3) query points of whole frames, frame by frame, x-fastest."""
        times = self.times[torch.as_tensor(np.asarray(frames).reshape(-1), device=self.device)]
        xy = self.xy.repeat(len(times), 1)
        return torch.cat([xy, times.repeat_interleave(len(self.xy))[:, None]], dim=1)

    def render(self, model, frames):
        """Model frames (K, size**2), with gradients; POUnet renders psi(t) C^T directly."""
        frames = np.asarray(frames).reshape(-1)
        if hasattr(model, "frames"):
            return model.frames(self.times[torch.as_tensor(frames, device=self.device)])
        return model(self.coordinates(frames)).reshape(len(frames), -1)

    def predicted_data(self, images, frames):
        return self.operator.forward(images, frames)

    @torch.no_grad()
    def evaluate(self, model, frames, chunk=8):
        """Normalized data fidelity mean(((P f - d) / sigma)^2) and truth RRMSE over `frames`."""
        was_training = model.training
        model.eval()
        residual = error = norm = 0.
        try:
            for start in range(0, len(frames), chunk):
                part = np.asarray(frames[start:start + chunk])
                images = self.render(model, part)
                misfit = (self.predicted_data(images, part) - self.data[torch.as_tensor(part, device=self.device)])
                # MPS has no float64: move to the CPU before accumulating.
                residual += (misfit / self.sigma).cpu().double().square().sum().item()
                truth = self.truth[part]
                error += np.square(images.cpu().double().numpy() - truth).sum()
                norm += np.square(truth.astype(np.float64)).sum()
        finally:
            model.train(was_training)
        return dict(data_fidelity=residual / (len(frames) * self.operator.rows_per_frame),
                    truth_rrmse=float(np.sqrt(error / norm)))
