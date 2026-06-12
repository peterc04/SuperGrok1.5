"""grokking_optimizers/parallel/shard_map.py — the ZeRO shard PARTITION.

IMPLEMENTS /workspace/.parallelism_design.md §3.4 (the shard partition that
respects the per-TENSOR optimizer constraint) + the elementwise even-partition
helper (design §2.4 "shard the flat [0,total) by even-partition").

THE PROBLEM (design §3.4): ``ZeRO3Sharder._even_partition`` splits the FLAT param
buffer mid-tensor. That breaks muon / SuperGrok11 / SuperGrok15 / SuperGrok2 —
their P2.x optimizer stages (Newton–Schulz over a whole 2D weight; the per-tensor
meta-net mu/gate; SG2's segmented sort + CSA/HCA/PEER/GRU) need the WHOLE
matrix/tensor on ONE rank (design §0.4/§2.4). So the ZeRO-3 shard for those cells
must be TENSOR-GRANULAR, not element-granular.

THIS MODULE (pure functions on SIZE LISTS — no torch, no torch.distributed, no
GPU, so it is CPU/CI-testable per design §7):

  * :func:`even_partition`            — the flat element partition (mirrors
                                        ``distributed._even_partition`` exactly).
  * :func:`partition_elementwise_even`— per-rank flat slices for the ELEMENTWISE
                                        optimizers (finer, best balance).
  * :func:`partition_tensor_granular` — assign WHOLE tensors to ranks (greedy by
                                        numel, balanced) so NO tensor is split —
                                        REQUIRED for muon/SG11/SG15/SG2 (§3.4).
  * :func:`shard_mode_for_optimizer`  — key the mode off the optimizer (§3.4:
                                        elementwise → flat-even; per-tensor →
                                        tensor-granular).

It does NOT touch ``distributed.py``; a caller (e.g. a future
``fused_train_step_distributed``, design §6.2) picks the mode via
:func:`shard_mode_for_optimizer` and calls the matching partitioner.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────
#  Optimizer → shard-mode taxonomy (design §0.4 / §2.4 / §3.4).
#
#  ELEMENTWISE: shard the flat [0,total) — every per-element update is local, so
#  splitting a tensor across ranks is fine. (grokadamw/looksam/prodigy are
#  elementwise at APPLY; their global scalar is handled by a tiny all-reduce, NOT
#  by the shard partition — so they shard elementwise here.)
#
#  PER-TENSOR: a whole matrix/tensor must live on one rank (the P2.x stage reads
#  the whole tensor cooperatively). muon is per-MATRIX (NS over each 2D weight);
#  SG11/SG15/SG2 are per-TENSOR (meta-net over each tensor's elements).
# ──────────────────────────────────────────────────────────────────────────
ELEMENTWISE_OPTIMIZERS = frozenset({
    "adamw", "lion", "grokfast", "grokadamw", "looksam", "prodigy", "neuralgrok",
})
PER_TENSOR_OPTIMIZERS = frozenset({
    "muon", "supergrok11", "supergrok15", "supergrok2",
})

# OptId integers (mirror csrc/fused/sm_90/opt_components.cuh::OptId) so a caller
# holding the numeric id can map to a mode too. Single source of the spelling.
_OPTID_TO_NAME: Dict[int, str] = {
    0: "adamw", 1: "lion", 2: "grokfast", 3: "grokadamw", 4: "looksam",
    5: "prodigy", 6: "neuralgrok", 7: "muon", 8: "supergrok11",
    9: "supergrok15", 10: "supergrok2",
}

_MODE_ELEMENTWISE = "elementwise_even"
_MODE_TENSOR_GRANULAR = "tensor_granular"


def shard_mode_for_optimizer(opt) -> str:
    """Return the shard mode an optimizer requires (design §3.4).

    ``opt`` may be the optimizer NAME (str, case-insensitive) or its OptId int
    (0..10). Per-tensor cells (muon/SG11/SG15/SG2) → ``"tensor_granular"``;
    everything else → ``"elementwise_even"``.

    Raises ``ValueError`` for an unknown optimizer (loud, no silent default —
    same discipline as the kernel dispatch, design §1.3).
    """
    if isinstance(opt, int):
        if opt not in _OPTID_TO_NAME:
            raise ValueError(f"unknown OptId {opt!r} (expected 0..10)")
        name = _OPTID_TO_NAME[opt]
    else:
        name = str(opt).strip().lower()
    if name in PER_TENSOR_OPTIMIZERS:
        return _MODE_TENSOR_GRANULAR
    if name in ELEMENTWISE_OPTIMIZERS:
        return _MODE_ELEMENTWISE
    raise ValueError(
        f"unknown optimizer {opt!r} — not in the 11-optimizer taxonomy "
        f"(elementwise={sorted(ELEMENTWISE_OPTIMIZERS)}, "
        f"per_tensor={sorted(PER_TENSOR_OPTIMIZERS)})"
    )


# ──────────────────────────────────────────────────────────────────────────
#  Result types.
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TensorPlacement:
    """One tensor's whole-tensor placement under tensor-granular sharding.

    ``name``/``numel`` identify the tensor; ``rank`` is the DP rank that owns it
    in full (no split). Tensor-granular sharding emits one of these per tensor.
    """

    name: str
    numel: int
    rank: int


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """A partition result: each rank's owned (name, start, end) flat slices.

    ``slices[r]`` is the list of ``(name, start, end)`` slices rank ``r`` owns
    (``end`` exclusive, in the tensor's OWN flat index space). For tensor-granular
    mode every slice is a whole tensor ``(name, 0, numel)``; for elementwise mode
    a tensor may appear on several ranks as contiguous sub-slices.

    ``mode`` is ``"elementwise_even"`` or ``"tensor_granular"``. ``rank_numel[r]``
    is the total element count rank ``r`` owns (the balance metric).
    """

    mode: str
    world: int
    slices: List[List[Tuple[str, int, int]]]
    rank_numel: List[int]

    @property
    def total_numel(self) -> int:
        return sum(self.rank_numel)

    def imbalance(self) -> float:
        """max/mean rank load (1.0 == perfectly balanced; higher == lumpier).

        Tensor-granular is necessarily lumpier than elementwise (whole tensors
        can't be split), so this is the "honest cost, measure it" number design
        §3.4 calls for. Returns 1.0 for the degenerate world<=1.
        """
        if self.world <= 1 or self.total_numel == 0:
            return 1.0
        mean = self.total_numel / self.world
        return max(self.rank_numel) / mean if mean > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────
#  Elementwise flat partition (design §2.4).
# ──────────────────────────────────────────────────────────────────────────


def even_partition(numel: int, world: int, rank: int) -> Tuple[int, int]:
    """Contiguous, padded-even partition of ``numel`` across ``world`` ranks.

    Byte-for-byte the convention of ``distributed._even_partition`` (the
    DeepSpeed ZeRO-3 flat-partition): each rank owns ``ceil(numel/world)``
    elements (last rank clamped), so the union covers ``[0, numel)`` exactly with
    no overlap. Mirrored here (not imported) so this pure-Python module needs no
    torch import; a unit test asserts it agrees with ``distributed`` when torch
    is present.
    """
    if world <= 1:
        return 0, numel
    per = (numel + world - 1) // world
    start = min(rank * per, numel)
    end = min(start + per, numel)
    return start, end


def partition_elementwise_even(
    named_sizes: Sequence[Tuple[str, int]], world: int
) -> ShardPlan:
    """ELEMENTWISE shard: split each tensor's flat buffer evenly across ranks.

    The finer partition (design §2.4) — best balance, valid for the elementwise
    optimizers (adamw/lion/grokfast/neuralgrok + the elementwise core of
    grokadamw/looksam/prodigy). A tensor MAY be split across ranks (that's fine:
    every per-element apply is local).

    ``named_sizes`` is ``[(name, numel), ...]`` (e.g. from ``named_parameters()``
    sizes). Returns a :class:`ShardPlan` in ``"elementwise_even"`` mode.
    """
    world = max(1, int(world))
    slices: List[List[Tuple[str, int, int]]] = [[] for _ in range(world)]
    rank_numel = [0] * world
    for name, numel in named_sizes:
        numel = int(numel)
        if numel < 0:
            raise ValueError(f"tensor {name!r} has negative numel {numel}")
        for r in range(world):
            s, e = even_partition(numel, world, r)
            if e > s:
                slices[r].append((name, s, e))
                rank_numel[r] += e - s
    return ShardPlan(_MODE_ELEMENTWISE, world, slices, rank_numel)


# ──────────────────────────────────────────────────────────────────────────
#  Tensor-granular partition (design §3.4) — the new requirement.
# ──────────────────────────────────────────────────────────────────────────


def partition_tensor_granular(
    named_sizes: Sequence[Tuple[str, int]], world: int
) -> ShardPlan:
    """Assign WHOLE tensors to ranks (greedy by numel, balanced) — design §3.4.

    No tensor is split across ranks, which is REQUIRED for the per-tensor
    optimizers (muon/SuperGrok11/SuperGrok15/SuperGrok2): their P2.x stages need
    the whole matrix/tensor on one rank (design §0.4/§2.4), and the co-located
    optimizer state (m/mu/sharpness/slow/gru_state) rides with its tensor.

    ALGORITHM — greedy longest-processing-time (LPT): sort tensors descending by
    numel, then assign each to the currently least-loaded rank. This is the
    standard balanced whole-item bin-packing heuristic; it is deterministic
    (a pure function of the size list + world — design §2.7 requires deterministic
    shard boundaries) and gives a good (not optimal) balance. Ties broken by the
    lowest rank index, then by tensor name, so the result is stable across runs.

    Balance is necessarily lumpier than elementwise (the embedding + ff matrices
    dominate), so :meth:`ShardPlan.imbalance` is the honest DP-efficiency cost
    design §3.4 says to measure.

    ``named_sizes`` is ``[(name, numel), ...]``. Returns a :class:`ShardPlan` in
    ``"tensor_granular"`` mode where every owned slice is a whole tensor
    ``(name, 0, numel)``.
    """
    world = max(1, int(world))
    slices: List[List[Tuple[str, int, int]]] = [[] for _ in range(world)]
    rank_numel = [0] * world

    # Validate + materialize.
    tensors: List[Tuple[str, int]] = []
    for name, numel in named_sizes:
        numel = int(numel)
        if numel < 0:
            raise ValueError(f"tensor {name!r} has negative numel {numel}")
        tensors.append((name, numel))

    # LPT: descending by numel; tie-break by name so the order is total + stable.
    order = sorted(tensors, key=lambda t: (-t[1], t[0]))
    for name, numel in order:
        # Least-loaded rank; ties → lowest rank index (min() is stable on the
        # (load, rank) key).
        r = min(range(world), key=lambda k: (rank_numel[k], k))
        slices[r].append((name, 0, numel))
        rank_numel[r] += numel

    # Keep each rank's slice list in the input (named_parameters()) order, so the
    # owned slices are reported in a stable, layout-meaningful order (the kernel
    # addresses tensors in named_parameters() order — decoder_layout.cuh).
    input_order = {name: i for i, (name, _) in enumerate(tensors)}
    for r in range(world):
        slices[r].sort(key=lambda sl: input_order[sl[0]])

    return ShardPlan(_MODE_TENSOR_GRANULAR, world, slices, rank_numel)


def partition_for_optimizer(
    named_sizes: Sequence[Tuple[str, int]], world: int, opt
) -> ShardPlan:
    """Pick the partition mode from the optimizer (design §3.4) and run it.

    Convenience wrapper: per-tensor cells → :func:`partition_tensor_granular`;
    elementwise cells → :func:`partition_elementwise_even`.
    """
    mode = shard_mode_for_optimizer(opt)
    if mode == _MODE_TENSOR_GRANULAR:
        return partition_tensor_granular(named_sizes, world)
    return partition_elementwise_even(named_sizes, world)
