"""grokking_optimizers/parallel/distributed_step.py — the rank-aware
DECOMPOSED train step (design §6.2 ``fused_train_step_distributed``).

IMPLEMENTS /workspace/.parallelism_design.md §6.2 (task #25 phase-2, Lane C) as
a NEW module wrapping the existing ``dispatch.fused_train_step`` — the §2.1
decomposition, cut at the B2 seam:

    [1] production fused step on THIS rank's batch shard  → local grad[total]
    [2] FIXED-ORDER cross-DP grad reduce (all_gather + ascending-rank fp32 sum
        — the §2.7/§6.4 A/A/A non-negotiable; never a raw NCCL reduce-scatter)
    [3] sharded-optimizer apply over the OWNED flat slice   (§2.3)
    [4] all-gather params → full updated params on every rank (§2.1[4])

HARDWARE LINEAGE: steps [1]-[4] are byte-for-byte the sequence
tests/hw/test_dp2_loopback_determinism.py proved green on 1×H100 (ef433ac,
DP=2 loopback cross-rank A/A/A); this module PROMOTES that proven worker logic
into a production-shaped, importable entry so the 8× driver and the tests call
ONE implementation. The DP=2 worker path here is re-gated by
tests/hw/test_distributed_step.py before the window.

DEPENDENCY-INJECTED SHARDED APPLY: the post-reduce-scatter optimizer kernel is
passed in as ``sharded_apply`` (signature below). The 1-GPU gates inject the
JIT ``sharded_optimizer_binding.sharded_opt_step`` (bit-parity-proven,
DELIVERABLE 1); the future production build injects the _ops entry once the
kernel lands in dispatch (csrc/bindings is another lane's surface — this
module deliberately does not import it).

KERNEL-CUT NOTE (§2.2, honest scope): the fused kernel currently always runs
its in-kernel P3 (the ``if constexpr (Par::kShardOptGrad) return;`` early-exit
is dispatch-wiring work on csrc/bindings — out of this lane's surface). Under
ZeRO>=2 this module therefore captures the grad via ``return_grad=True``,
RESETS the params the in-kernel P3 advanced, and applies the sharded kernel on
the reduce-scattered grad — numerically the §2.1 dance with a redundant P3
whose result is discarded (correctness-identical, bit-gated; the wasted P3 is
a measured inefficiency the §2.2 early-exit removes at the 8× window, never a
correctness risk). The world=1 default short-circuits to the plain fused step
(zero overhead on the single-GPU path).

Scalars: the per-step optimizer scalars come from ``_opt_scalars_from`` — the
SINGLE source the production path uses (bit-parity depends on it).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grokking_optimizers.parallel.shard_map import even_partition
from grokking_optimizers.parallel.zero3 import FlatShardPlan, Zero3FlatParamStore

# OptId mirror (opt_components.cuh) for the ELEMENTWISE ids this step's flat
# sharded apply supports today (the per-tensor cells need the tensor-granular
# full-kernel path, design §2.3 — loudly rejected below, not silently wrong).
_ELEMENTWISE_OPTID: Dict[str, int] = {"adamw": 0, "lion": 1, "grokfast": 2}

# sharded_apply(opt_id, params_shard, grad_shard, state_shard, sc: dict, step)
ShardedApply = Callable[..., None]


def _torch():
    import torch  # noqa: PLC0415
    return torch


@dataclasses.dataclass
class DistStepContext:
    """Everything rank-shaped the decomposed step needs.

    world/rank      : DP group coordinates (world=1 ⇒ degenerate single-GPU).
    plan/store/state: the rank's ZeRO-3 flat plan + resident param-shard store
                      + shard-local [m|v|extra] state planes ([3, shard_numel]).
    process_group   : torch.distributed group for the collectives (None ⇒ the
                      default group). Ignored at world=1.
    """

    world: int
    rank: int
    plan: FlatShardPlan
    store: Zero3FlatParamStore
    state: "object"               # torch.Tensor [3, shard_numel]
    process_group: Optional[object] = None


def make_dist_step_context(named_sizes: Sequence[Tuple[str, int]], world: int,
                           rank: int, opt_name: str, full_flat,
                           process_group=None) -> DistStepContext:
    """Build the rank's context from the model's named sizes + initial flat
    params (every rank passes the SAME full_flat — same-seed init, the §7.1
    convention). Elementwise opts only (loud otherwise, §2.3)."""
    torch = _torch()
    if opt_name not in _ELEMENTWISE_OPTID:
        raise ValueError(
            f"distributed_step: {opt_name!r} is not an elementwise OptId "
            f"{sorted(_ELEMENTWISE_OPTID)} — the per-tensor cells "
            f"(muon/SG11/SG15/SG2) need the tensor-granular full-kernel path "
            f"(design §2.3), which is not this flat apply.")
    from grokking_optimizers.parallel.zero3 import flat_plan_for_optimizer
    plan = flat_plan_for_optimizer(named_sizes, world, opt_name)
    store = Zero3FlatParamStore(plan, rank, full_flat=full_flat)
    state = torch.zeros(3, store.shard.numel(), device=full_flat.device,
                        dtype=torch.float32)
    return DistStepContext(world=world, rank=rank, plan=plan, store=store,
                           state=state, process_group=process_group)


def shard_batch_rows(tokens, targets, world: int, rank: int):
    """Deterministic row shard of a fixed batch, 16-aligned per rank (the
    proven _shard_batch from the DP=2 loopback gate — wgmma needs B%16==0 and
    pre-rounding to 16*world keeps the shards balanced + comparable)."""
    B = int(targets.shape[0])
    step = 16 * max(1, world)
    Bw = B - (B % step)
    s, e = even_partition(Bw, max(1, world), rank)
    return tokens[s:e].contiguous(), targets[s:e].contiguous(), Bw


def fixed_order_allreduce_grad(grad, world: int, group=None):
    """THE §2.7/§6.4 fixed-order cross-rank grad reduce: NCCL all_gather (pure
    data movement — bit-exact, order-free) then an ASCENDING-RANK fp32 sum,
    averaged by world. Identical on every rank and across reruns. world=1 ⇒
    returns grad unchanged (already the full reduced grad)."""
    if world <= 1:
        return grad
    torch = _torch()
    import torch.distributed as dist  # noqa: PLC0415
    total = grad.numel()
    gathered = torch.empty(total * world, dtype=grad.dtype, device=grad.device)
    torch.cuda.synchronize()      # the megakernel fully done before the collective
    dist.all_gather_into_tensor(gathered, grad.contiguous(), group=group)
    torch.cuda.synchronize()
    acc = torch.zeros(total, dtype=torch.float32, device=grad.device)
    for r in range(world):        # ascending rank — fixed, structural order
        acc += gathered[r * total:(r + 1) * total]
    acc /= float(world)
    return acc


def allgather_full_params(ctx: DistStepContext):
    """§2.1[4]: reconstitute the full updated params on every rank. world=1 /
    in-process virtual ranks use the store's gather; a launched job uses the
    padded all_gather convention inside Zero3FlatParamStore.gather_full."""
    if ctx.world <= 1:
        return ctx.store.gather_full()
    return ctx.store.gather_full(group=ctx.process_group)


def fused_train_step_distributed(
        model_name: str, opt_name: str, model, optimizer, tokens, targets, *,
        ctx: DistStepContext, sharded_apply: ShardedApply, state_cache: dict,
        step: int, gemm_impl: str = "wgmma",
        decompose_at_world1: bool = False):
    """ONE decomposed rank-aware step (design §6.2 [1]-[7]). Returns
    (loss_local, full_params_after) — full params are also scattered back into
    ``model`` (the existing fused_train_step contract keeps model/state_cache
    coherent for step+1).

    world=1 + decompose_at_world1=False (the production default) is the PLAIN
    fused step — literally unchanged single-GPU behavior (§6.2: "the
    single-GPU path is literally unchanged"). decompose_at_world1=True forces
    the full decomposition at world=1 — the §7.1 identity gate's mode (must be
    BIT-IDENTICAL to the plain path; gated by tests/hw/test_distributed_step.py).
    """
    torch = _torch()
    from grokking_optimizers.dispatch import fused_train_step, _opt_scalars_from
    if opt_name not in _ELEMENTWISE_OPTID:
        raise ValueError(f"elementwise OptIds only (got {opt_name!r}) — see §2.3")

    if ctx.world <= 1 and not decompose_at_world1:
        loss = fused_train_step(model_name, opt_name, model, optimizer,
                                tokens, targets, state_cache=state_cache,
                                step=step, gemm_impl=gemm_impl)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        total = sum(p.numel() for _, p in named)
        full = torch.empty(total, device=next(model.parameters()).device)
        off = 0
        for _, p in named:
            n = p.numel()
            full[off:off + n].copy_(p.data.reshape(-1))
            off += n
        return loss, full

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    dev = next(model.parameters()).device
    if total != ctx.plan.total:
        raise ValueError(f"plan total {ctx.plan.total} != model total {total}")

    # [0] (ZeRO-3) full pre-gather of params → the model the kernel will run.
    #     The store is authoritative between steps; scatter into p.data so the
    #     fused step launches on the gathered values (§3.2(a)).
    full_before = allgather_full_params(ctx)
    off = 0
    for _, p in named:
        n = p.numel()
        p.data.reshape(-1).copy_(full_before[off:off + n])
        off += n

    # [1] the fused step on THIS rank's batch shard → local grad. (The in-kernel
    #     P3 also runs and advances p.data — discarded below; see the module
    #     docstring's §2.2 note.)
    tok_sh, tgt_sh, _bw = shard_batch_rows(tokens, targets, ctx.world, ctx.rank)
    loss, grad_local = fused_train_step(model_name, opt_name, model, optimizer,
                                        tok_sh, tgt_sh, state_cache=state_cache,
                                        step=step, return_grad=True,
                                        gemm_impl=gemm_impl)

    # [2] fixed-order cross-DP grad reduce (identity at world=1).
    grad_full = fixed_order_allreduce_grad(grad_local, ctx.world,
                                           group=ctx.process_group)

    # [3] sharded apply over the OWNED slices (in place on the store's shard +
    #     the rank's [m|v|extra] planes), with the production scalars.
    sc = _opt_scalars_from(optimizer, step)
    opt_id = _ELEMENTWISE_OPTID[opt_name]
    for fs, fe, ss, se in ctx.store.owned():
        params_shard = ctx.store.shard[ss:se].contiguous()
        grad_shard = grad_full[fs:fe].contiguous()
        state_shard = ctx.state[:, ss:se].reshape(-1).contiguous()
        sharded_apply(opt_id, params_shard, grad_shard, state_shard,
                      sc["lr"], sc["beta1"], sc["beta2"], sc["eps"],
                      sc["weight_decay"], sc["bc1"], sc["bc2"],
                      sc.get("alpha", 0.98), sc.get("lamb", 2.0), step)
        ctx.store.shard[ss:se].copy_(params_shard)
        ctx.state[:, ss:se].copy_(state_shard.view(3, se - ss))

    # INVARIANT: the sharded apply above started from params-BEFORE-the-step —
    # the store was gathered before [1] and is never touched by the kernel
    # (the in-kernel P3 mutates p.data only, and that result is discarded by
    # the [4]/[5] overwrite below). This is what makes the decomposed update
    # bit-identical to a clean B2-cut kernel (the §2.2 early-exit, once wired).

    # [4] all-gather the updated shards → full params; [5] scatter into p.data
    #     so model/state_cache stay coherent for step+1.
    full_after = allgather_full_params(ctx)
    off = 0
    for _, p in named:
        n = p.numel()
        p.data.reshape(-1).copy_(full_after[off:off + n])
        off += n
    return loss, full_after


__all__ = [
    "DistStepContext",
    "make_dist_step_context",
    "shard_batch_rows",
    "fixed_order_allreduce_grad",
    "allgather_full_params",
    "fused_train_step_distributed",
]
