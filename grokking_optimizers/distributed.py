"""
Distributed training utilities for SuperGrok v2.

Provides helpers for DDP and FSDP training with SuperGrok v2's meta-net.
Handles meta-net weight synchronization, expert count aggregation, and
FSDP parameter gathering for the Mamba scan.

Usage with DDP::

    import torch.distributed as dist
    from grokking_optimizers.distributed import setup_distributed, cleanup_distributed

    setup_distributed()
    model = DDP(model, device_ids=[local_rank])
    opt = SuperGrok2(model.parameters(), ...)
    # Training loop works as normal — distributed hooks are automatic

Usage with FSDP::

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from grokking_optimizers.distributed import setup_distributed

    setup_distributed()
    opt = SuperGrok2(model.parameters(), ...)
    SuperGrok2.exclude_meta_net_from_fsdp(opt.meta_net)
    model = FSDP(model, auto_wrap_policy=...)

Usage with torchrun::

    torchrun --nproc_per_node=4 train.py
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> int:
    """Initialize distributed training from environment variables.

    Compatible with ``torchrun`` / ``torch.distributed.launch``.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        init_method: URL for init (default: env:// from torchrun).

    Returns:
        local_rank: The local rank of this process.
    """
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size <= 1:
        return local_rank

    if init_method is None:
        init_method = "env://"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get the global rank of this process (0 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the world size (1 if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def broadcast_optimizer_state(optimizer, src: int = 0):
    """Broadcast optimizer state from src rank to all other ranks.

    Useful for ensuring consistent optimizer state after loading a checkpoint
    or after the first step when states are lazily initialized.

    Args:
        optimizer: A SuperGrok2 (or any) optimizer instance.
        src: Source rank to broadcast from.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() <= 1:
        return

    from .supergrok2 import SuperGrok2

    if isinstance(optimizer, SuperGrok2):
        # Broadcast flat state tensors
        for tensor_list in [
            optimizer._flat_exp_avgs,
            optimizer._flat_exp_avg_sqs,
            optimizer._flat_mus,
            optimizer._flat_sharpness,
            optimizer._flat_gru_states,
        ]:
            for t in tensor_list:
                if t is not None:
                    dist.broadcast(t, src=src)

        # Broadcast mamba states
        for state in optimizer._flat_mamba_fwd_states + optimizer._flat_mamba_bwd_states:
            if state is not None:
                dist.broadcast(state, src=src)

        # Broadcast meta-net parameters
        for p in optimizer.meta_net.parameters():
            dist.broadcast(p.data, src=src)

        # Broadcast expert counts
        dist.broadcast(optimizer.meta_net.expert_counts, src=src)


def wrap_model_ddp(model, device_ids=None, **kwargs):
    """Wrap a model with DDP if distributed training is active.

    Falls back to the unwrapped model if not distributed.

    Args:
        model: The model to wrap.
        device_ids: GPU device IDs (default: current device).
        **kwargs: Additional arguments for DDP.

    Returns:
        The model (DDP-wrapped if distributed, otherwise unchanged).
    """
    if not dist.is_available() or not dist.is_initialized():
        return model
    if dist.get_world_size() <= 1:
        return model

    from torch.nn.parallel import DistributedDataParallel as DDP

    if device_ids is None and torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]

    return DDP(model, device_ids=device_ids, **kwargs)
