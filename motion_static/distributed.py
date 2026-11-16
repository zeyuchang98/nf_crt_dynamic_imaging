"""Data-parallel training over processes: one process per GPU (one per Vista node).

Every process holds the whole model and draws its own batch from its frame shard
(streaming.py). After each backward pass the weighted gradients w_p grad(loss_p) are
summed over processes, so every process applies the same Adam update to the same
parameters; the parameters are broadcast from process 0 once at the start and checked
for agreement whenever losses are logged. Process 0 writes every file.

Launch: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT in the environment
(torchrun or experiments/vista/train_multinode.sbatch), and `run.py --distributed`.
"""
import os

import torch
import torch.distributed as dist


class Context:
    """This process's place in the run; one process without torch.distributed is rank 0 of 1."""

    def __init__(self, rank=0, world_size=1, local_rank=0, backend=None):
        self.rank, self.world_size, self.local_rank, self.backend = rank, world_size, local_rank, backend
        self.weight = 1.  # w_p = |S_p| / F, set once the shards are known

    @property
    def is_root(self):
        return self.rank == 0

    @property
    def parallel(self):
        return self.world_size > 1

    def _device(self):
        # NCCL reduces CUDA tensors; gloo reduces CPU tensors.
        return torch.device("cuda", torch.cuda.current_device()) if self.backend == "nccl" else torch.device("cpu")

    def all_reduce_gradients(self, model):
        """Sum the (already weighted) gradients of every parameter over processes."""
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:  # every process must take part in every reduction
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)

    def weighted_sum(self, values):
        """sum_p w_p value_p for a mapping of name -> float (e.g. the batch losses)."""
        names = list(values)
        tensor = torch.tensor([self.weight * values[name] for name in names], dtype=torch.float64,
                              device=self._device())
        dist.all_reduce(tensor)
        return dict(zip(names, tensor.tolist()))

    def broadcast_parameters(self, model):
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, 0)

    def check_parameters(self, model):
        """Raise if the processes' parameters differ (they must stay identical)."""
        checksum = torch.stack([parameter.detach().double().sum() for parameter in model.parameters()]).sum()
        low, high = checksum.clone().to(self._device()), checksum.clone().to(self._device())
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if low.item() != high.item():
            raise RuntimeError(f"Process parameters diverged (checksums {low.item()!r} to {high.item()!r})")

    def maximum(self, value):
        tensor = torch.tensor([value], dtype=torch.float64, device=self._device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def any(self, mask):
        """Elementwise OR of a boolean numpy array over processes."""
        tensor = torch.as_tensor(mask.astype("uint8"), device=self._device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.cpu().numpy().astype(bool)

    def sum_array(self, array):
        """Elementwise sum of a float32 numpy array over processes."""
        tensor = torch.as_tensor(array, device=self._device())
        dist.all_reduce(tensor)
        return tensor.cpu().numpy()

    def all_gather(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def barrier(self):
        if self.parallel:
            dist.barrier()


def world_size_from_environment():
    return int(os.environ.get("WORLD_SIZE", 1))


def initialize(device):
    """Join the process group described by the environment; the backend follows the device."""
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.set_device(local_rank)
    backend = "nccl" if cuda else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return Context(rank, world_size, local_rank, backend)


def finalize(context):
    if context.backend is not None and dist.is_initialized():
        dist.destroy_process_group()
