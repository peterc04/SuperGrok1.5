"""grokking_optimizers/parallel/zero3.py — ZeRO-3 PARAM partitioning,
gather/release around the fused step, and the sharded optimizer-state
checkpoint/resume round-trip.

IMPLEMENTS /workspace/.parallelism_design.md §3 (task #25 phase-2, Lane C):
  * §3.2 option (a) FULL PRE-GATHER — the locked first ZeRO-3 increment: the
    full flat param blob is reconstituted transiently for the megakernel
    launch (the kernel reads full params through P1) and RELEASED after the
    step; only the rank's shard stays resident between steps. Options (b)
    per-phase gather (rejected — P1 is barrier-free across layers) and (c)
    in-kernel NVSHMEM gather (frontier, only if (a) OOMs at the flagship
    size) are documented decisions, not built here.
  * §3.4 the shard partition respects the per-TENSOR optimizer constraint via
    grokking_optimizers.parallel.shard_map (elementwise-even for elementwise
    optimizers; tensor-granular for muon/SG11/SG15/SG2) — this module turns
    those plans into FLAT-BLOB slices (the megakernel ABI is the flat
    ``float[total]`` blob + the generated offset tables, design §0.2; the
    existing per-tensor ``ZeRO3Sharder`` shards framework params, NOT the
    kernel blob — that is the gap this module fills).
  * checkpoint/save-resume: each rank persists ONLY its param shard + opt-state
    shard (+ a loud plan fingerprint); resume must be BIT-EXACT (gated by
    tests/hw/test_zero3_roundtrip.py, building on the ef433ac sharded-opt
    bit-parity).

DESIGN NOTES:
  * The flat state layout is the kernel's: ``[m | v | extra]`` parallel to the
    flat param blob (design §0.2) — a rank's state shard is the SAME slices of
    each of the 3 (or more) state planes, kept shard-local exactly the way the
    sharded_optimizer_kernel binding consumes them ([m_sh | v_sh | extra_sh]).
  * world=1 degenerates to no-op copies (the §7.1 identity). The multi-rank
    collective path reuses the padded all_gather_into_tensor convention of
    ``ZeRO3Sharder.all_gather_params`` (one source of conventions); when
    torch.distributed is not initialized, gather_full falls back to the
    VIRTUAL-rank path (peer stores in-process) — the honest 1-GPU loopback,
    used by the round-trip test. There is no silent wrong-world fallback:
    mismatches raise.

Import-safe without torch at module import; torch is required at call time
for tensor ops (guarded with a loud error).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from grokking_optimizers.parallel.shard_map import (
    ShardPlan,
    partition_elementwise_even,
    partition_tensor_granular,
    shard_mode_for_optimizer,
)


def _torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("zero3 tensor ops require torch") from e


# ──────────────────────────────────────────────────────────────────────────
#  Flat-blob slice plan (the megakernel-ABI view of a ShardPlan).
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class FlatShardPlan:
    """Per-rank contiguous slices of the FLAT param blob [0, total).

    ``slices[r]`` = ordered list of (start, end) in flat-blob coordinates;
    within a rank the slices are ascending and non-overlapping; across ranks
    they partition [0, total) exactly. ``mode``/``world`` echo the shard_map
    plan. The fingerprint pins (mode, world, total, every boundary) so a
    checkpoint can refuse a mismatched resume LOUDLY.
    """

    mode: str
    world: int
    total: int
    slices: List[List[Tuple[int, int]]]

    def rank_numel(self, rank: int) -> int:
        return sum(e - s for s, e in self.slices[rank])

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"mode": self.mode, "world": self.world, "total": self.total,
             "slices": self.slices},
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def flat_plan_for_optimizer(named_sizes: Sequence[Tuple[str, int]], world: int,
                            opt) -> FlatShardPlan:
    """Build the FLAT-blob plan for an optimizer (§3.4 mode keying).

    ELEMENTWISE → one contiguous even slice of the WHOLE flat blob per rank
    (the partition the DP=2 loopback gate already validated — even_partition
    over [0,total)). PER-TENSOR → whole tensors per rank (tensor-granular LPT),
    mapped to flat coordinates via the named_sizes prefix sum (the same
    named_parameters() order the generated layout headers encode).
    """
    mode = shard_mode_for_optimizer(opt)
    total = sum(int(n) for _, n in named_sizes)
    world = max(1, int(world))
    offsets: Dict[str, int] = {}
    off = 0
    for name, n in named_sizes:
        offsets[name] = off
        off += int(n)

    slices: List[List[Tuple[int, int]]]
    if mode == "tensor_granular":
        plan: ShardPlan = partition_tensor_granular(named_sizes, world)
        slices = []
        for r in range(world):
            sl = [(offsets[name] + s, offsets[name] + e)
                  for name, s, e in plan.slices[r]]
            slices.append(sorted(sl))
    else:
        # Elementwise: ONE contiguous slice of the flat blob per rank — the
        # convention the sharded-opt kernel + DP loopback tests use
        # (even_partition over [0,total), NOT per-tensor slicing).
        from grokking_optimizers.parallel.shard_map import even_partition
        slices = []
        for r in range(world):
            s, e = even_partition(total, world, r)
            slices.append([(s, e)] if e > s else [])
    return FlatShardPlan(mode=mode, world=world, total=total, slices=slices)


# ──────────────────────────────────────────────────────────────────────────
#  Param store: resident shard + gather/release around the fused step (§3.2a).
# ──────────────────────────────────────────────────────────────────────────


class Zero3FlatParamStore:
    """One rank's resident param shard + the full-pre-gather/release dance.

    Lifecycle per step (design §2.1 with §3.2(a)):
        full = store.gather_full(...)        # [0] reconstitute for the kernel
        <fused megakernel step on full>      # [1] (+ ZeRO>=2 grad dance)
        <sharded-opt updates store.shard>    # [3] via shard_view()
        store.release(full)                  # keep only the shard resident
    ``gather_full`` sources non-owned slices from (in priority order):
      * ``peers``      — other ranks' stores IN-PROCESS (the 1-GPU virtual-rank
                         loopback; honest: real data movement, just one device);
      * torch.distributed all-gather on ``group`` when a job is launched;
      * world==1       — the degenerate copy.
    Anything else raises (no silent stale-params path).
    """

    def __init__(self, plan: FlatShardPlan, rank: int, full_flat=None,
                 device=None, dtype=None) -> None:
        torch = _torch()
        if not (0 <= rank < plan.world):
            raise ValueError(f"rank {rank} out of range for world {plan.world}")
        self.plan = plan
        self.rank = rank
        n = plan.rank_numel(rank)
        if full_flat is not None:
            if full_flat.numel() != plan.total:
                raise ValueError(
                    f"full_flat numel {full_flat.numel()} != plan total {plan.total}")
            parts = [full_flat[s:e] for s, e in plan.slices[rank]]
            self.shard = (torch.cat(parts).clone() if parts else
                          full_flat.new_empty(0))
        else:
            self.shard = torch.zeros(
                n, device=device, dtype=dtype if dtype is not None else torch.float32)

    # Owned-slice iteration: (flat_start, flat_end, shard_start, shard_end).
    def owned(self) -> List[Tuple[int, int, int, int]]:
        out = []
        o = 0
        for s, e in self.plan.slices[self.rank]:
            out.append((s, e, o, o + (e - s)))
            o += e - s
        return out

    def shard_view(self):
        """The mutable resident shard (the sharded-opt kernel's params_shard)."""
        return self.shard

    def gather_full(self, peers: Optional[Sequence["Zero3FlatParamStore"]] = None,
                    group=None, out=None):
        torch = _torch()
        full = out if out is not None else torch.empty(
            self.plan.total, device=self.shard.device, dtype=self.shard.dtype)
        if full.numel() != self.plan.total:
            raise ValueError("gather_full: out has wrong numel")
        # own slices first (always local).
        for fs, fe, ss, se in self.owned():
            full[fs:fe].copy_(self.shard[ss:se])
        if self.plan.world == 1:
            return full
        if peers is not None:
            stores = {p.rank: p for p in peers}
            stores[self.rank] = self
            if sorted(stores) != list(range(self.plan.world)):
                raise ValueError(
                    f"gather_full(peers): need every rank exactly once, got {sorted(stores)}")
            for r, st in stores.items():
                if r == self.rank:
                    continue
                if st.plan.fingerprint() != self.plan.fingerprint():
                    raise ValueError("gather_full(peers): plan fingerprint mismatch")
                for fs, fe, ss, se in st.owned():
                    full[fs:fe].copy_(st.shard[ss:se])
            return full
        # Real collective path (8× window): padded all_gather_into_tensor over
        # the CONTIGUOUS elementwise slices (one slice/rank). Tensor-granular
        # gathers go per-slice (lumpier; bounded by 30 tensors).
        import torch.distributed as dist  # noqa: PLC0415
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "gather_full: world>1 but neither peers given nor "
                "torch.distributed initialized — refusing a stale gather")
        if self.plan.mode == "elementwise_even":
            per = (self.plan.total + self.plan.world - 1) // self.plan.world
            buf = self.shard.new_zeros(per)
            buf[: self.shard.numel()].copy_(self.shard)
            gathered = self.shard.new_empty(per * self.plan.world)
            dist.all_gather_into_tensor(gathered, buf, group=group)
            for r in range(self.plan.world):
                s = min(r * per, self.plan.total)
                e = min(s + per, self.plan.total)
                if e > s:
                    full[s:e].copy_(gathered[r * per:r * per + (e - s)])
        else:
            # tensor-granular: broadcast each rank's slice list from its owner.
            for r in range(self.plan.world):
                for (s, e) in self.plan.slices[r]:
                    seg = full[s:e]
                    if r == self.rank:
                        # already copied from our shard above
                        pass
                    dist.broadcast(seg, src=r, group=group)
        return full

    def release(self, full) -> None:
        """End-of-step: pull the owned slices back into the resident shard.
        The caller drops ``full`` afterwards (the §3.2(a) transient buffer)."""
        if full.numel() != self.plan.total:
            raise ValueError("release: full has wrong numel")
        for fs, fe, ss, se in self.owned():
            self.shard[ss:se].copy_(full[fs:fe])


# ──────────────────────────────────────────────────────────────────────────
#  Sharded optimizer-state checkpoint (bit-exact save/resume).
# ──────────────────────────────────────────────────────────────────────────

_CKPT_VERSION = 1


def save_sharded_checkpoint(path: str, *, plan: FlatShardPlan, rank: int,
                            step: int, param_shard, opt_state_shard,
                            extra_meta: Optional[Dict] = None) -> None:
    """Persist ONE rank's shard-local step state (ZeRO-3: each rank saves only
    its 1/N — the aggregate IS the checkpoint).

    param_shard     : [shard_numel] fp32 (the resident Zero3FlatParamStore shard)
    opt_state_shard : [k * shard_numel] fp32, the kernel's shard-local
                      [m | v | extra] planes exactly as sharded_opt_step uses
                      them (k = planes; 3 for the elementwise cells).
    The plan fingerprint + (world, rank, mode, step) are stored and VALIDATED
    on load — resuming under a different partition is a hard error, never a
    silent re-shard.
    """
    torch = _torch()
    n = plan.rank_numel(rank)
    if param_shard.numel() != n:
        raise ValueError(f"param_shard numel {param_shard.numel()} != plan shard {n}")
    if opt_state_shard.numel() % max(n, 1) != 0:
        raise ValueError(
            f"opt_state_shard numel {opt_state_shard.numel()} is not a whole "
            f"multiple of shard_numel {n} — the [m|v|extra] planes are misaligned")
    obj = {
        "version": _CKPT_VERSION,
        "fingerprint": plan.fingerprint(),
        "mode": plan.mode,
        "world": plan.world,
        "rank": rank,
        "total": plan.total,
        "step": int(step),
        "param_shard": param_shard.detach().cpu().clone(),
        "opt_state_shard": opt_state_shard.detach().cpu().clone(),
        "extra_meta": dict(extra_meta or {}),
    }
    torch.save(obj, path)


def load_sharded_checkpoint(path: str, *, plan: FlatShardPlan, rank: int) -> Dict:
    """Load + LOUDLY validate one rank's shard checkpoint against the live plan.

    Returns the dict with 'param_shard'/'opt_state_shard' still on CPU (the
    caller .to(device)'s them); raises on ANY fingerprint/world/rank/total
    mismatch (design discipline: deterministic shard boundaries are part of
    the cross-rank A/A/A contract — a drifted plan must never load).
    """
    torch = _torch()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != _CKPT_VERSION:
        raise RuntimeError(f"checkpoint version {obj.get('version')} != {_CKPT_VERSION}")
    if obj["fingerprint"] != plan.fingerprint():
        raise RuntimeError(
            "sharded checkpoint plan fingerprint mismatch — the partition "
            f"(mode/world/boundaries) changed since save (saved mode={obj['mode']} "
            f"world={obj['world']}, live mode={plan.mode} world={plan.world}). "
            "Refusing to resume; re-shard explicitly if intended.")
    if obj["rank"] != rank:
        raise RuntimeError(
            f"checkpoint is rank {obj['rank']}'s shard, asked to load as rank {rank}")
    n = plan.rank_numel(rank)
    if obj["param_shard"].numel() != n:
        raise RuntimeError("checkpoint param_shard numel != plan shard numel")
    return obj


__all__ = [
    "FlatShardPlan",
    "flat_plan_for_optimizer",
    "Zero3FlatParamStore",
    "save_sharded_checkpoint",
    "load_sharded_checkpoint",
]
