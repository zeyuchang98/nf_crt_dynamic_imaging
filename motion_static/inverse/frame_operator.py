"""Per-frame CRT operators P_k as torch sparse tensors, for training on any device.

P_k holds the rows of the full-view operator for frame k's views (crt_operator.frame_rows),
so P_k f_k is frame k's noiseless data in the (view, ring) order of the measurement file.
Frames with the same views share one tensor: with ProxNF's schedule the views repeat every
72 frames. The tensors are COO, the sparse layout whose products and backward pass work on
MPS (CSR does not, checked 2026-11-18). The adjoint P_k^T is stored separately because the
proximal-splitting gradient step needs it; direct backpropagation gets it from autograd.
"""
import numpy as np
import torch

from .crt_operator import frame_rows


def coo_tensor(matrix, device):
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.as_tensor(coo.data, dtype=torch.float32),
                                   coo.shape, check_invariants=False).coalesce().to(device)


class FrameOperator:
    def __init__(self, matrix, view_degrees, num_rings, angle_step_deg=1., device="cpu", adjoint=False):
        """matrix: full-view scipy operator; view_degrees: (frames, views) of the measurement file."""
        self.device = torch.device(device)
        patterns, self.pattern = np.unique(np.asarray(view_degrees, dtype=float), axis=0, return_inverse=True)
        self.pattern = self.pattern.reshape(-1)
        blocks = [matrix[frame_rows(views, num_rings, angle_step_deg)] for views in patterns]
        self.forward_blocks = [coo_tensor(block, self.device) for block in blocks]
        self.adjoint_blocks = [coo_tensor(block.T, self.device) for block in blocks] if adjoint else None
        self.rows_per_frame = blocks[0].shape[0]
        self.pixels = matrix.shape[1]

    def __len__(self):
        return len(self.pattern)

    def forward(self, images, frames):
        """images (K, pixels), x-fastest; frames: K window positions. Returns (K, rows_per_frame)."""
        return torch.stack([torch.sparse.mm(self.forward_blocks[self.pattern[k]], image[:, None])[:, 0]
                            for image, k in zip(images, np.asarray(frames).reshape(-1))])

    def adjoint(self, data, frames):
        """data (K, rows_per_frame) -> (K, pixels): P_k^T applied frame by frame."""
        if self.adjoint_blocks is None:
            raise ValueError("Built without adjoint blocks")
        return torch.stack([torch.sparse.mm(self.adjoint_blocks[self.pattern[k]], values[:, None])[:, 0]
                            for values, k in zip(data, np.asarray(frames).reshape(-1))])
