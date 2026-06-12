"""grokking_optimizers.parallel — Phase-2 parallelism helpers (CPU-authorable).

IMPLEMENTS /workspace/.parallelism_design.md §3.4 (the tensor-granular shard
partition the existing ``ZeRO3Sharder._even_partition`` does NOT model). It is a
NEW, standalone module — it does NOT edit ``grokking_optimizers/distributed.py``
(the existing distributed layer is reused as-is; this package adds the missing
per-tensor partition as pure functions so it can be authored + unit-tested on CPU
before the 8×H100 window, per design §7).

WHY a separate package: design §3.4 requires a shard mode keyed off the optimizer
— ELEMENTWISE optimizers may use the finer flat even-partition (better balance),
but the PER-TENSOR optimizers (muon / SuperGrok11 / SuperGrok15 / SuperGrok2)
need WHOLE tensors on one rank (their P2.x stages cooperate across CTAs per
matrix/tensor on the GridBarrier substrate — design §0.4/§2.4). Splitting a
tensor mid-way (what ``_even_partition`` does) BREAKS them. :func:`shard_map.
partition_tensor_granular` assigns whole tensors greedily-by-numel for balance;
:func:`shard_map.even_partition` is the elementwise flat helper (a pure mirror of
``distributed._even_partition`` so the two layers agree without a cross-import).

PURE PYTHON: no torch, no torch.distributed, no GPU. Operates on size lists, so
it imports and tests cleanly on any box (design §7: maximize what's validated on
CPU/1-GPU).
"""

from __future__ import annotations

from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    TensorPlacement,
    even_partition,
    partition_elementwise_even,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)

__all__ = [
    "ShardPlan",
    "TensorPlacement",
    "even_partition",
    "partition_elementwise_even",
    "partition_tensor_granular",
    "shard_mode_for_optimizer",
]
