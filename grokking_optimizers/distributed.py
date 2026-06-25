"""grokking_optimizers.distributed — Stage 7 distributed-training layer (§8).

This is a **Python-level orchestration layer** that sits on top of the existing
optimizers and the Stage-6 fused megakernel. It introduces **no kernel changes**
and does **not** touch any optimizer `.step()` math; it only wires the per-GPU
local compute into a multi-GPU / multinode rank mesh.

Coverage of the §8 spec:

* §8.1 — classic **3D parallelism**: data (DP) × tensor (TP) × pipeline (PP).
  There is deliberately **no sequence / 4th parallel dim** — the race models are
  short-sequence (arithmetic grokking, small ViT/Mamba), so a sequence split
  buys nothing and only adds all-to-all traffic. The mesh is the standard
  DP×TP×PP rank decomposition.
* §8.3 — **ZeRO-3** full optimizer-state sharding across the DP group (SG2's
  ~5–6× optimizer-state footprint, from the CSA/HCA + PEER meta-net, needs it).
* §8.4 — orchestration via **DeepSpeed (ZeRO-3) or Megatron-LM**, NCCL on
  NVIDIA / RCCL on AMD. The fused-kernel adapter that lets the Stage-6
  single-launch megakernel satisfy the framework's separate fwd/bwd/step
  interface lives in :mod:`grokking_optimizers.megakernel_engine` (the trickiest
  integration — see that module's docstring).
* §8.5 — the megakernel components are reused as the per-GPU *local* compute;
  this module's comms layer wraps them.
* §8.6 — multinode at maximal efficiency (InfiniBand / NVLink-class fabric).
  AMD multinode trails NVIDIA under ZeRO-3 cross-node all-gathers (§2.13); the
  config records the backend so the harness can attribute the gap.

CRITICAL — **import safety**. Importing this module must succeed with no
distributed job launched, no GPU, and no DeepSpeed installed (the common case:
the self-test and `import grokking_optimizers`). Every `torch.distributed` call
is therefore guarded and lazy: nothing touches a process group at import time,
and a non-launched job degrades cleanly to a single-process world (world_size=1,
rank=0, all parallel dims = 1).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, Iterable, Optional, Tuple

# torch is a hard dependency of the package (the optimizers need it), but we
# still touch torch.distributed only behind guards, since the *backend* (NCCL/
# RCCL/gloo) and a launched job are optional.
import torch


# ───────────────────────────── ParallelConfig ────────────────────────────────


@dataclasses.dataclass
class ParallelConfig:
    """Declarative description of the 3D-parallel + ZeRO mesh (§8.1, §8.3, §8.4).

    The product ``data_parallel * tensor_parallel * pipeline_parallel`` must
    equal the launched ``world_size`` (validated lazily in
    :meth:`DistributedContext.from_config`, never at import).

    Attributes
    ----------
    data_parallel, tensor_parallel, pipeline_parallel:
        Sizes of the three parallel dims. ``1`` disables that dim. There is no
        sequence-parallel dim by design (short-sequence models, §8.1).
    expert_parallel:
        Size of the expert-parallel (EP) dim — the MoE experts are sharded
        across this many ranks (the PEER/MoE optimizers, e.g.
        ``MoEAwareSuperGrok2``, route tokens to expert shards). ``1`` (the
        default) disables EP, so the common dense case is unaffected. EP shares
        ranks with the DP group (experts are replicated/sharded over the same
        data-parallel ranks), so it does NOT enlarge ``world_size``; instead it
        sub-divides the DP dim (``data_parallel`` must be a multiple of
        ``expert_parallel``). The EP group is the set of DP peers that together
        hold one full expert set.
    zero_stage:
        ZeRO optimizer-state sharding stage, 0/1/2/3. Stage 3 shards optimizer
        state **and** gradients **and** parameters across the DP group (§8.3).
        Sharding is always over the DP group (TP/PP shard the model itself).
    backend:
        Collective backend. ``"nccl"`` on NVIDIA, ``"rccl"`` on AMD (ROCm ships
        RCCL behind the same ``"nccl"`` ProcessGroup name — see
        :meth:`resolved_backend`), ``"gloo"`` for CPU-only smoke tests.
    use_megakernel:
        When True, the per-GPU local compute is the Stage-6 fused megakernel and
        the framework is driven through the
        :mod:`grokking_optimizers.megakernel_engine` adapter (§8.4/§8.5).
    reduce_bucket_size:
        Target size, in **elements**, of the flat gradient buckets the native
        DDP all-reduce coalesces small per-param gradients into (§8.4). Larger
        buckets amortize per-collective launch latency at the cost of delaying
        the first reduce; ~25M elements (≈50 MB bf16 / 100 MB fp32) is a sane
        default for the race's small models. Honored by
        :meth:`DistributedContext.all_reduce_dp_grads` and the async-overlap
        grad hooks.
    """

    data_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    zero_stage: int = 0
    backend: str = "nccl"
    use_megakernel: bool = True
    reduce_bucket_size: int = 25_000_000

    def __post_init__(self) -> None:
        for name in ("data_parallel", "tensor_parallel", "pipeline_parallel",
                     "expert_parallel"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.zero_stage}")
        if not isinstance(self.reduce_bucket_size, int) or self.reduce_bucket_size < 1:
            raise ValueError(
                f"reduce_bucket_size must be a positive int, "
                f"got {self.reduce_bucket_size!r}")
        # EP sub-divides the DP dim (experts are sharded over DP peers), so it
        # must evenly divide it and never exceeds it.
        if self.data_parallel % self.expert_parallel != 0:
            raise ValueError(
                f"expert_parallel ({self.expert_parallel}) must evenly divide "
                f"data_parallel ({self.data_parallel}) — EP shards experts "
                f"across a subset of the DP ranks.")

    @property
    def ep(self) -> int:
        """Alias for ``expert_parallel`` (the short spelling used in configs)."""
        return self.expert_parallel

    @property
    def model_parallel_size(self) -> int:
        """TP×PP — the number of ranks one full model copy is spread across."""
        return self.tensor_parallel * self.pipeline_parallel

    @property
    def world_size(self) -> int:
        """Total ranks the config describes (DP×TP×PP)."""
        return self.data_parallel * self.tensor_parallel * self.pipeline_parallel

    def resolved_backend(self) -> str:
        """Map the logical backend to the torch ProcessGroup backend string.

        ROCm's RCCL registers under the *same* ``"nccl"`` ProcessGroup backend
        name (torch dispatches NCCL↔RCCL by the build, not the string), so
        ``"rccl"`` resolves to ``"nccl"`` for :func:`torch.distributed.init_process_group`.
        """
        return "nccl" if self.backend in ("nccl", "rccl") else self.backend

    def validate_against_world(self, world_size: int) -> None:
        """Raise if DP×TP×PP does not match the launched ``world_size``."""
        if self.world_size != world_size:
            raise ValueError(
                f"ParallelConfig DP×TP×PP = {self.data_parallel}×"
                f"{self.tensor_parallel}×{self.pipeline_parallel} = "
                f"{self.world_size} does not match launched world_size={world_size}"
            )


# ──────────────────────── distributed availability probes ─────────────────────


def _dist():
    """Return ``torch.distributed`` if importable, else ``None`` (no crash)."""
    try:
        import torch.distributed as dist  # noqa: PLC0415 — lazy by design
    except Exception:  # pragma: no cover - torch always ships dist, defensive
        return None
    return dist


def is_dist_available_and_initialized() -> bool:
    """True only when a real process group has been launched & initialized.

    This is the single guard every collective in this module funnels through, so
    the not-launched case (self-test, plain import) short-circuits to a no-op.
    """
    dist = _dist()
    return bool(dist) and dist.is_available() and dist.is_initialized()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bucketize(tensors, bucket_elems: int):
    """Group ``tensors`` into contiguous buckets of ~``bucket_elems`` elements.

    A single tensor larger than the target still forms its own (oversized)
    bucket so nothing is dropped. Order is preserved so the flat-buffer
    scatter-back in :meth:`DistributedContext.all_reduce_dp_grads` stays
    aligned. Pure-Python list math (no collectives) — unit-testable offline.
    """
    bucket: list = []
    cur = 0
    for t in tensors:
        n = t.numel()
        if bucket and cur + n > bucket_elems:
            yield bucket
            bucket, cur = [], 0
        bucket.append(t)
        cur += n
    if bucket:
        yield bucket


# ───────────────────────────── DistributedContext ────────────────────────────


@dataclasses.dataclass
class _RankMesh:
    """The (dp, tp, pp) coordinates of this rank plus the per-dim peer ranks.

    Rank-linearization order is the Megatron-LM convention: the fastest-varying
    dim is TP, then PP, then DP (``rank = ((dp * pp_size + pp) * tp_size + tp)``).
    Keeping TP contiguous puts the high-traffic tensor-parallel all-reduces on
    the tightest fabric (intra-node NVLink/Infinity-Fabric), which is what §8.6
    wants for maximal efficiency.
    """

    global_rank: int
    world_size: int
    dp: int
    tp: int
    pp: int
    ep: int
    dp_ranks: Tuple[int, ...]
    tp_ranks: Tuple[int, ...]
    pp_ranks: Tuple[int, ...]
    ep_ranks: Tuple[int, ...]


def _coords_from_rank(rank: int, dp: int, tp: int, pp: int) -> Tuple[int, int, int]:
    """Decompose a global rank into (dp, tp, pp) coords (TP fastest, then PP)."""
    tp_i = rank % tp
    pp_i = (rank // tp) % pp
    dp_i = rank // (tp * pp)
    return dp_i, tp_i, pp_i


def _rank_from_coords(dp_i: int, tp_i: int, pp_i: int, dp: int, tp: int, pp: int) -> int:
    return (dp_i * pp + pp_i) * tp + tp_i


def _build_mesh(rank: int, cfg: ParallelConfig) -> _RankMesh:
    """Pure-Python rank-mesh math (no collectives) — testable without a launch.

    Computes this rank's (dp,tp,pp) coords and the peer-rank lists for each of
    the three subgroups it belongs to. The actual ``new_group`` calls happen in
    :meth:`DistributedContext._init_groups`; this keeps the index arithmetic
    unit-testable on a single process.
    """
    dp, tp, pp = cfg.data_parallel, cfg.tensor_parallel, cfg.pipeline_parallel
    ep = cfg.expert_parallel
    dp_i, tp_i, pp_i = _coords_from_rank(rank, dp, tp, pp)

    # DP peers: vary dp_i, hold (tp_i, pp_i) — this is the ZeRO-3 shard group.
    dp_ranks = tuple(
        _rank_from_coords(d, tp_i, pp_i, dp, tp, pp) for d in range(dp)
    )
    # TP peers: vary tp_i, hold (dp_i, pp_i) — the per-layer all-reduce group.
    tp_ranks = tuple(
        _rank_from_coords(dp_i, t, pp_i, dp, tp, pp) for t in range(tp)
    )
    # PP peers: vary pp_i, hold (dp_i, tp_i) — the pipeline-stage send/recv group.
    pp_ranks = tuple(
        _rank_from_coords(dp_i, tp_i, p, dp, tp, pp) for p in range(pp)
    )
    # EP peers: the ``ep`` consecutive DP ranks that together hold one full
    # (sharded) expert set. DP is partitioned into dp/ep expert-replica blocks;
    # this rank's EP group is the block its dp_i falls in. EP coordinate is the
    # offset within that block. EP size 1 ⇒ a singleton group (no-op).
    ep_block_start = (dp_i // ep) * ep
    ep_dp_coords = range(ep_block_start, ep_block_start + ep)
    ep_ranks = tuple(
        _rank_from_coords(d, tp_i, pp_i, dp, tp, pp) for d in ep_dp_coords
    )
    ep_i = dp_i % ep
    return _RankMesh(
        global_rank=rank,
        world_size=cfg.world_size,
        dp=dp_i,
        tp=tp_i,
        pp=pp_i,
        ep=ep_i,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
        pp_ranks=pp_ranks,
        ep_ranks=ep_ranks,
    )


class DistributedContext:
    """Lazily-initialized holder of the DP/TP/PP process groups (§8.1).

    Construction never touches ``torch.distributed``. Call :meth:`initialize`
    (or use :meth:`from_config`, which initializes) inside a launched job to
    build the subgroups. With no launched job everything degrades to a
    single-process world: ``world_size == 1``, ``rank == 0``, every parallel dim
    collapses to size 1, and every collective helper becomes a no-op. That is
    exactly the self-test / plain-import path, and it must not crash.
    """

    def __init__(self, config: Optional[ParallelConfig] = None) -> None:
        self.config = config or ParallelConfig(use_megakernel=False)
        self._initialized = False
        self._mesh: Optional[_RankMesh] = None
        # Per-dim process-group handles (opaque; None ⇒ WORLD / single-process).
        self._dp_group = None
        self._tp_group = None
        self._pp_group = None
        self._ep_group = None
        # Async DDP grad-hook bookkeeping (populated by register_dp_grad_hooks).
        self._dp_grad_hook_handles: list = []
        self._pending_dp_works: list = []

    # ── construction / teardown ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: ParallelConfig) -> "DistributedContext":
        """Build a context and initialize its groups if a job is launched.

        Safe to call with no launch: returns a single-process context.
        """
        ctx = cls(config)
        ctx.initialize()
        return ctx

    def initialize(self, init_method: Optional[str] = None) -> "DistributedContext":
        """Initialize process groups *if and only if* a job is launched.

        Idempotent. With no launch (or torch.distributed unavailable) this marks
        the context single-process and returns immediately — no collective is
        ever issued, so import/self-test paths are safe.
        """
        if self._initialized:
            return self
        dist = _dist()

        # Single-process / not-launched path: build a degenerate 1-rank mesh.
        if dist is None or not dist.is_available():
            self._mesh = _build_mesh(0, ParallelConfig())
            self._initialized = True
            return self

        # If the launcher set the env vars but the user has not yet called
        # init_process_group, initialize it lazily here (idempotent guard).
        if not dist.is_initialized():
            world_size = _env_int("WORLD_SIZE", 1)
            if world_size <= 1:
                # No real job — single process. Do NOT init a group.
                self._mesh = _build_mesh(0, ParallelConfig())
                self._initialized = True
                return self
            rank = _env_int("RANK", 0)
            backend = self.config.resolved_backend()
            try:
                dist.init_process_group(
                    backend=backend, init_method=init_method,
                    world_size=world_size, rank=rank,
                )
            except Exception:
                # A misconfigured launch should degrade, not hard-crash an
                # import-time path. Fall back to single-process.
                self._mesh = _build_mesh(0, ParallelConfig())
                self._initialized = True
                return self

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.config.validate_against_world(world_size)
        self._mesh = _build_mesh(rank, self.config)
        self._init_groups()
        self._initialized = True
        return self

    def _init_groups(self) -> None:
        """Create the DP/TP/PP subgroups via ``new_group`` (launched-job only).

        ``new_group`` is a **collective** that every rank must enter for every
        group, so we iterate the full set of subgroups deterministically and
        each rank keeps the handle for the group it belongs to. This is the
        standard Megatron-LM group-construction pattern.
        """
        dist = _dist()
        assert dist is not None and self._mesh is not None
        cfg = self.config
        dp, tp, pp = cfg.data_parallel, cfg.tensor_parallel, cfg.pipeline_parallel
        ep = cfg.expert_parallel
        mesh = self._mesh

        # Build every DP group (one per (tp_i, pp_i) slice).
        for tp_i in range(tp):
            for pp_i in range(pp):
                ranks = [
                    _rank_from_coords(d, tp_i, pp_i, dp, tp, pp) for d in range(dp)
                ]
                g = dist.new_group(ranks)
                if mesh.global_rank in ranks:
                    self._dp_group = g
        # Build every EP group (one per (tp_i, pp_i, dp-block) slice): the ``ep``
        # consecutive DP ranks that hold one full expert set. Only meaningful
        # when ep > 1; with ep == 1 each group is a singleton (still entered by
        # every rank — new_group is collective). Skipped entirely for ep == 1 to
        # avoid issuing dp*tp*pp singleton-group collectives.
        if ep > 1:
            for tp_i in range(tp):
                for pp_i in range(pp):
                    for block_start in range(0, dp, ep):
                        ranks = [
                            _rank_from_coords(d, tp_i, pp_i, dp, tp, pp)
                            for d in range(block_start, block_start + ep)
                        ]
                        g = dist.new_group(ranks)
                        if mesh.global_rank in ranks:
                            self._ep_group = g
        # Build every TP group (one per (dp_i, pp_i) slice).
        for dp_i in range(dp):
            for pp_i in range(pp):
                ranks = [
                    _rank_from_coords(dp_i, t, pp_i, dp, tp, pp) for t in range(tp)
                ]
                g = dist.new_group(ranks)
                if mesh.global_rank in ranks:
                    self._tp_group = g
        # Build every PP group (one per (dp_i, tp_i) slice).
        for dp_i in range(dp):
            for tp_i in range(tp):
                ranks = [
                    _rank_from_coords(dp_i, tp_i, p, dp, tp, pp) for p in range(pp)
                ]
                g = dist.new_group(ranks)
                if mesh.global_rank in ranks:
                    self._pp_group = g

    def barrier(self) -> None:
        if is_dist_available_and_initialized():
            _dist().barrier()

    def destroy(self) -> None:
        """Tear down the default group (best-effort; safe if never launched)."""
        if is_dist_available_and_initialized():
            try:
                _dist().destroy_process_group()
            except Exception:
                pass
        self._initialized = False
        self._dp_group = self._tp_group = self._pp_group = None
        self._ep_group = None

    # ── rank / coordinate accessors (always safe) ────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def mesh(self) -> _RankMesh:
        if self._mesh is None:
            self.initialize()
        assert self._mesh is not None
        return self._mesh

    @property
    def rank(self) -> int:
        return self.mesh.global_rank

    @property
    def world_size(self) -> int:
        return self.mesh.world_size

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def dp_rank(self) -> int:
        return self.mesh.dp

    @property
    def tp_rank(self) -> int:
        return self.mesh.tp

    @property
    def pp_rank(self) -> int:
        return self.mesh.pp

    @property
    def ep_rank(self) -> int:
        return self.mesh.ep

    @property
    def dp_world_size(self) -> int:
        return self.config.data_parallel

    @property
    def tp_world_size(self) -> int:
        return self.config.tensor_parallel

    @property
    def pp_world_size(self) -> int:
        return self.config.pipeline_parallel

    @property
    def ep_world_size(self) -> int:
        return self.config.expert_parallel

    @property
    def dp_group(self):
        return self._dp_group

    @property
    def tp_group(self):
        return self._tp_group

    @property
    def pp_group(self):
        return self._pp_group

    @property
    def ep_group(self):
        return self._ep_group

    @property
    def is_pipeline_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_pipeline_last_stage(self) -> bool:
        return self.pp_rank == self.config.pipeline_parallel - 1

    # ── collective helpers (no-op single-process) ───────────────────────────

    def all_reduce_tp(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Sum-reduce a tensor across the tensor-parallel group (in place).

        The per-layer all-reduce of a column/row-parallel Linear (§8.1 TP).
        No-op when TP size is 1 or no job is launched.
        """
        if self.tp_world_size > 1 and is_dist_available_and_initialized():
            _dist().all_reduce(tensor, group=self._tp_group)
        return tensor

    def all_reduce_ep(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Sum-reduce a tensor across the expert-parallel group (in place).

        Used to combine partial expert outputs / expert-gradient stats across
        the EP shards that hold one full expert set. No-op when EP size is 1 or
        no job is launched.
        """
        if self.ep_world_size > 1 and is_dist_available_and_initialized():
            _dist().all_reduce(tensor, group=self._ep_group)
        return tensor

    def all_reduce_dp_grads(self, params: Iterable["torch.Tensor"]) -> None:
        """Average gradients across the DP group (the DDP gradient all-reduce).

        Used for ZeRO-0/1/2, where gradients are replicated then averaged. Under
        ZeRO-3 the gradient is reduce-scattered instead (see
        :class:`ZeRO3Sharder`). No-op single-process.

        Gradients are coalesced into flat buckets of ~``reduce_bucket_size``
        elements so a model with many small parameters issues a handful of large
        all-reduces instead of one per parameter — the per-collective launch
        latency is the dominant cost on the race's small models. Numerically
        identical to the prior per-param all-reduce (sum then divide by the DP
        world size); only the collective granularity changes.
        """
        if self.dp_world_size <= 1 or not is_dist_available_and_initialized():
            return
        dist = _dist()
        ws = self.dp_world_size
        grads = [p.grad for p in params if getattr(p, "grad", None) is not None]
        bucket_size = self.config.reduce_bucket_size
        for bucket in _bucketize(grads, bucket_size):
            if len(bucket) == 1:
                g = bucket[0]
                dist.all_reduce(g, group=self._dp_group)
                g.div_(ws)
                continue
            flat = torch.cat([g.reshape(-1) for g in bucket])
            dist.all_reduce(flat, group=self._dp_group)
            flat.div_(ws)
            # Scatter the reduced flat buffer back into each grad in place.
            offset = 0
            for g in bucket:
                n = g.numel()
                g.copy_(flat[offset:offset + n].view_as(g))
                offset += n

    # ── async comm/compute overlap (DDP grad hooks) ──────────────────────────

    def register_dp_grad_hooks(
        self, params: Iterable["torch.Tensor"]
    ) -> int:
        """Register backward hooks that issue **async** DP all-reduces.

        Each parameter's ``register_post_accumulate_grad_hook`` fires the moment
        that grad finishes accumulating in backward, so the reduce overlaps with
        the remaining backward compute (comm/compute overlap, §8.4). The handles
        are returned via the count and tracked on the context;
        :meth:`wait_dp_grads` must be called before the optimizer step to drain
        the in-flight collectives and apply the ``1/world`` averaging.

        No-op (returns 0) single-process or when the grad-hook API is missing.
        Returns the number of hooks registered.
        """
        if self.dp_world_size <= 1 or not is_dist_available_and_initialized():
            return 0
        self._pending_dp_works = getattr(self, "_pending_dp_works", [])
        self._dp_grad_hook_handles = getattr(self, "_dp_grad_hook_handles", [])
        ws = self.dp_world_size
        group = self._dp_group
        dist = _dist()
        count = 0
        for p in params:
            if not getattr(p, "requires_grad", False):
                continue
            if not hasattr(p, "register_post_accumulate_grad_hook"):
                # PyTorch < 2.1 — caller should use all_reduce_dp_grads instead.
                return count

            def _hook(param, _ws=ws, _grp=group, _d=dist):
                if param.grad is None:
                    return
                work = _d.all_reduce(
                    param.grad, group=_grp, async_op=True)
                # Defer the /ws division until wait() so it lands after the
                # async reduce completes.
                self._pending_dp_works.append((work, param.grad, _ws))

            self._dp_grad_hook_handles.append(
                p.register_post_accumulate_grad_hook(_hook))
            count += 1
        return count

    def wait_dp_grads(self) -> None:
        """Wait on all in-flight async DP all-reduces and finish the averaging.

        Call once after ``loss.backward()`` and before ``optimizer.step()``.
        Safe to call when no hooks were registered (no-op).
        """
        pending = getattr(self, "_pending_dp_works", None)
        if not pending:
            return
        for work, grad, ws in pending:
            if work is not None:
                work.wait()
            grad.div_(ws)
        pending.clear()

    def remove_dp_grad_hooks(self) -> None:
        """Remove any registered DP grad hooks (best-effort)."""
        for h in getattr(self, "_dp_grad_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
        self._dp_grad_hook_handles = []
        self._pending_dp_works = []


# Process-wide default context. It is single-process until something calls
# initialize() inside a launched job — constructing it touches nothing.
_DEFAULT_CONTEXT: Optional[DistributedContext] = None


def get_distributed_context(
    config: Optional[ParallelConfig] = None,
) -> DistributedContext:
    """Return the process-wide :class:`DistributedContext`, creating it lazily.

    Passing a ``config`` on first call sets the mesh shape. Safe to call from
    import-time / single-process code: it never issues a collective unless a job
    is actually launched.
    """
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = DistributedContext(config)
    return _DEFAULT_CONTEXT


# ─────────────────────────────── ZeRO-3 sharding ─────────────────────────────


def deepspeed_available() -> bool:
    """True iff DeepSpeed is importable (guarded — never raises)."""
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("deepspeed") is not None
    except Exception:  # pragma: no cover - defensive
        return False


@dataclasses.dataclass
class _ParamShard:
    """Bookkeeping for one parameter's flat slice owned by this DP rank.

    Under ZeRO-3 each DP rank owns a contiguous ``[start:end)`` slice of every
    parameter's flattened buffer (and, symmetrically, the same slice of every
    optimizer-state tensor for that param). ``numel`` is the full param size.
    """

    name: str
    numel: int
    start: int
    end: int  # exclusive

    @property
    def shard_numel(self) -> int:
        return self.end - self.start


def _even_partition(numel: int, world: int, rank: int) -> Tuple[int, int]:
    """Contiguous, padded-even partition of ``numel`` across ``world`` ranks.

    Matches the DeepSpeed ZeRO-3 flat-partition convention: each rank owns
    ``ceil(numel/world)`` elements (the last rank's slice is clamped), so the
    union covers ``[0, numel)`` exactly with no overlap.
    """
    if world <= 1:
        return 0, numel
    per = (numel + world - 1) // world
    start = min(rank * per, numel)
    end = min(start + per, numel)
    return start, end


# ───────────────────────── Tensor-parallel weight shard (§8.1 TP) ─────────────
# dist_step.md §6.C.5 / run_harness.md §0: the Megatron column/row split of the
# decoder's large per-layer weights so per-rank Nmax = kDecMaxTensorNumel/TP — the
# load-bearing memory win that makes the staged-opt (SG2) scratch fit at TP=8 (it is
# LINEAR in Nmax, so it shrinks by TP). This is NOT the ZeRO flat even-partition (which
# splits the FLAT blob arbitrarily and would break the per-layer all-reduce reassembly):
# TP needs the Megatron STRUCTURE — column-parallel tensors split on their OUTPUT dim,
# row-parallel on their INPUT dim — so the in-kernel ``tp_allreduce_sum_fixed_order``
# (tp_transport.cuh) after the row-parallel GEMMs reassembles the right thing.
#
# PURE FUNCTION on a name+shape list (no torch tensor data, no collectives, no GPU) so it
# is CPU/CI-testable exactly like shard_map.partition_tensor_granular. The caller (the
# host bringup / harness) feeds it [(name, numel)] or [(name, shape)] from
# named_parameters(); it returns per-rank numel + the per-tensor split kind so the budget
# math (flagship_budget.Nmax/tp) and the kernel ParConfig<…,TP,…> agree on the geometry.

# Megatron parallel-tensor name classification for the decoder cell (the
# named_parameters() spelling decoder_flagship_layout.cuh addresses). COLUMN-parallel
# tensors are split on their first (output-feature) dim; ROW-parallel on their second
# (input-feature) dim; everything else (embeddings, norms, final unembed, biases) is
# REPLICATED on every TP rank (Megatron leaves them whole — the all-reduce handles the
# cross-rank combine, not a weight split).
_TP_COLUMN_PARALLEL = ("in_proj", "ff.0", "qkv", "fc1", "w1", "gate_proj", "up_proj")
_TP_ROW_PARALLEL    = ("out_proj", "ff.2", "proj", "fc2", "w2", "down_proj")

# TP split kinds.
TP_REPLICATED = "replicated"
TP_COLUMN     = "column"      # split dim 0 (output features) by TP
TP_ROW        = "row"        # split dim 1 (input features) by TP


@dataclasses.dataclass(frozen=True)
class TPTensorShard:
    """One tensor's per-TP-rank shard under Megatron column/row parallelism.

    ``name``/``full_numel`` identify the tensor; ``kind`` is one of
    :data:`TP_REPLICATED` / :data:`TP_COLUMN` / :data:`TP_ROW`. ``shard_numel`` is
    this TP rank's owned element count (== full_numel for replicated tensors,
    full_numel/TP for a split tensor whose split dim is divisible by TP). For a
    split tensor ``rows``/``cols`` carry the per-rank shard's 2D shape (the kernel
    reads the matrix shape for the wgmma tiling); for a replicated/1D tensor they
    are 0. ``split_dim`` is 0 (column), 1 (row), or -1 (replicated).
    """

    name: str
    full_numel: int
    kind: str
    shard_numel: int
    rows: int
    cols: int
    split_dim: int


@dataclasses.dataclass(frozen=True)
class TPShardPlan:
    """The full TP weight-shard for one (tp, tp_rank) point (dist_step.md §6.C.5).

    ``shards`` is one :class:`TPTensorShard` per input tensor, in input order
    (named_parameters() order — the kernel addresses tensors in that order). The
    aggregate metrics drive the budget gate + the static-allocation sizing.
    """

    tp: int
    tp_rank: int
    shards: List["TPTensorShard"]

    @property
    def total_shard_numel(self) -> int:
        """This TP rank's resident param count (the per-rank model footprint)."""
        return sum(s.shard_numel for s in self.shards)

    @property
    def total_full_numel(self) -> int:
        return sum(s.full_numel for s in self.shards)

    @property
    def max_shard_numel(self) -> int:
        """Per-rank Nmax — the LARGEST per-tensor shard. This is the term the SG2
        staged-opt scratch is linear in (run_harness.md §0): at TP>1 the split
        tensors shrink to numel/TP, so the binding tensor's per-rank numel is
        what the budget uses (== kDecMaxTensorNumel/TP when the max tensor splits)."""
        return max((s.shard_numel for s in self.shards), default=0)

    @property
    def n_split(self) -> int:
        return sum(1 for s in self.shards if s.kind != TP_REPLICATED)


def _tp_classify(name: str) -> str:
    """Classify a tensor name into a Megatron TP split kind (case-insensitive
    substring match on the column/row roster). Unknown ⇒ replicated (the safe
    Megatron default: embeddings/norms/biases/unembed are never weight-split)."""
    low = name.lower()
    for key in _TP_COLUMN_PARALLEL:
        if key in low:
            return TP_COLUMN
    for key in _TP_ROW_PARALLEL:
        if key in low:
            return TP_ROW
    return TP_REPLICATED


def _as_numel_shape(size) -> Tuple[int, Tuple[int, ...]]:
    """Normalize a size entry to (numel, shape). Accepts an int numel or a shape
    tuple/list (e.g. ``p.shape``); a torch.Size is just a tuple subclass."""
    if isinstance(size, int):
        return size, (size,)
    shape = tuple(int(x) for x in size)
    numel = 1
    for x in shape:
        numel *= x
    return numel, shape


def partition_tensor_parallel(
    named_sizes, tp: int, tp_rank: int, *, model: str = "decoder",
) -> TPShardPlan:
    """Megatron column/row TP weight-shard of the decoder's per-layer weights
    (dist_step.md §6.C.5 / run_harness.md §0). Pure function on a name+shape list.

    ``named_sizes`` is ``[(name, numel), ...]`` OR ``[(name, shape), ...]`` (e.g. from
    ``[(n, p.numel()) for n, p in model.named_parameters()]`` or with ``p.shape``). For
    a split tensor the 2D matrix shape is needed to know WHICH dim to split (output vs
    input feature), so a shape is preferred; given a bare numel for a 2D weight we infer
    the split factor from ``tp`` directly (numel/tp) and report ``rows/cols = 0`` (the
    budget only needs the per-rank numel, which is numel/tp either way).

    SPLIT GEOMETRY (Megatron):
      * COLUMN-parallel (``in_proj`` 3d×d, ``ff.0`` 4d×d): split the OUTPUT-feature
        dim (dim 0) by ``tp`` → each rank holds ``(out/tp)×in`` → numel/tp. No
        all-reduce after (the split is along the output; the next op consumes the
        local columns). ``in_proj`` 3d×d at tp=8 → (3d/8)×d.
      * ROW-parallel (``out_proj`` d×d, ``ff.2`` d×4d): split the INPUT-feature dim
        (dim 1) by ``tp`` → each rank holds ``out×(in/tp)`` → numel/tp. The partial
        outputs are all-reduced after (the in-kernel ``tp_allreduce_sum_fixed_order``).
      * REPLICATED (embeddings ``tok``/``pos``, ``norm``s, final unembed, biases):
        whole tensor on every rank (Megatron never weight-splits these).

    Returns a :class:`TPShardPlan`; ``plan.max_shard_numel`` is the per-rank Nmax the
    staged-opt budget is linear in (== kDecMaxTensorNumel/tp when the max tensor splits).

    LOUD on a split dim that ``tp`` does not divide (the Megatron geometry requires
    ``out_features % tp == 0`` for column and ``in_features % tp == 0`` for row — at the
    flagship d=1600 every split dim is a multiple of 8/4/2/1, but a bad (tp, model) pair
    must fail here, never silently mis-shard). At tp==1 every tensor is replicated whole
    (the byte-identical single-GPU fold — TPShardPlan.total_shard_numel == total).
    """
    tp = int(tp)
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    if not (0 <= tp_rank < tp):
        raise ValueError(f"tp_rank {tp_rank} out of range for tp {tp}")
    if model != "decoder":
        # The column/row roster is the decoder's; other cells reuse the same
        # mechanism with their own roster. Loud rather than silently mis-classify.
        raise ValueError(
            f"partition_tensor_parallel: model={model!r} unsupported "
            f"(only 'decoder' has a Megatron col/row roster wired here)")

    shards: List[TPTensorShard] = []
    for name, size in named_sizes:
        full_numel, shape = _as_numel_shape(size)
        if full_numel < 0:
            raise ValueError(f"tensor {name!r} has negative numel {full_numel}")
        if tp == 1:
            shards.append(TPTensorShard(name, full_numel, TP_REPLICATED,
                                        full_numel, 0, 0, -1))
            continue
        kind = _tp_classify(name)
        if kind == TP_REPLICATED:
            shards.append(TPTensorShard(name, full_numel, TP_REPLICATED,
                                        full_numel, 0, 0, -1))
            continue
        # A split tensor: divide numel by tp (the Megatron col/row split of a 2D
        # weight is exactly an even division of the split-dim, hence numel/tp).
        if full_numel % tp != 0:
            raise ValueError(
                f"tensor {name!r} (numel {full_numel}, kind {kind}) is not "
                f"divisible by tp={tp} — Megatron col/row split requires the "
                f"split feature dim to be a multiple of tp")
        shard_numel = full_numel // tp
        rows = cols = 0
        split_dim = 0 if kind == TP_COLUMN else 1
        if len(shape) == 2:
            out_f, in_f = shape
            if kind == TP_COLUMN:
                if out_f % tp != 0:
                    raise ValueError(
                        f"tensor {name!r} out_features {out_f} not divisible by "
                        f"tp={tp} (column-parallel split)")
                rows, cols = out_f // tp, in_f
            else:  # TP_ROW
                if in_f % tp != 0:
                    raise ValueError(
                        f"tensor {name!r} in_features {in_f} not divisible by "
                        f"tp={tp} (row-parallel split)")
                rows, cols = out_f, in_f // tp
        shards.append(TPTensorShard(name, full_numel, kind, shard_numel,
                                    rows, cols, split_dim))
    return TPShardPlan(tp=tp, tp_rank=tp_rank, shards=shards)


class ZeRO3Sharder:
    """§8.3 ZeRO-3 full optimizer-state sharding across the DP group.

    Two backends, auto-selected:

    * **DeepSpeed** — when importable, ZeRO-3 is best handled by DeepSpeed's
      `deepspeed.initialize(...)` engine (partitioned params + reduce-scatter
      gradients + all-gather-on-use, plus offload). We don't reimplement it; we
      hand back the framework-managed shape and the megakernel adapter (§8.4)
      plugs in as the `client_optimizer`. :meth:`build_ds_config` emits the
      matching ZeRO-3 config dict.

    * **Native shim** — when DeepSpeed is absent, this class provides the same
      *pattern* with raw ``torch.distributed`` collectives:

        - :meth:`partition_optimizer_state` assigns each DP rank a contiguous
          ``1/N`` slice of every param's optimizer state (m/v/etc.), so the
          aggregate state memory is ``1/N`` per rank — the win SG2's ~5–6×
          state footprint needs.
        - :meth:`reduce_scatter_grads` replaces the DP all-reduce: each rank
          ends up with the *reduced* gradient for **only its owned slice**
          (reduce-scatter), which is what the local optimizer step consumes.
        - :meth:`all_gather_params` reconstitutes the full parameter from the
          per-rank slices after the step (all-gather-on-step), so forward/
          backward see complete weights.

    Both backends are import-safe and degrade to a no-op single-process shim
    (every collective short-circuits when DP size is 1 / no job launched).

    DOCUMENTED PARTIAL STOP (cross-param flat-bucket ZeRO-3): the native shim
    now uses the fused per-parameter ``reduce_scatter_tensor`` /
    ``all_gather_into_tensor`` primitives (half the traffic of the old
    all-reduce-then-slice, and one collective call instead of a Python list
    gather). It still issues **one collective per parameter**. A further win
    would coalesce many small params into a single flat ~``reduce_bucket_size``
    bucket per reduce-scatter/all-gather (as DeepSpeed's ``contiguous_gradients``
    does). That is NOT implemented here: a correct cross-param bucketed
    reduce-scatter must reproduce DeepSpeed's exact per-rank ownership of the
    *concatenated* buffer (padding + alignment so each rank's owned slice maps
    back to the right param sub-ranges), which is easy to get subtly wrong and
    cannot be validated without a real multi-GPU job. The risk outweighs the
    value for the race's small models, so it is deferred to the DeepSpeed path
    (``build_ds_config`` already sets ``contiguous_gradients=True`` and the
    bucket sizes). The DDP (ZeRO-0/1/2) path *is* bucketed — see
    :meth:`DistributedContext.all_reduce_dp_grads`.
    """

    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx
        self.use_deepspeed = deepspeed_available()
        # name -> _ParamShard for this rank (native shim only).
        self._shards: Dict[str, _ParamShard] = {}

    @property
    def dp_world(self) -> int:
        return self.ctx.dp_world_size

    @property
    def dp_rank(self) -> int:
        return self.ctx.dp_rank

    # ── DeepSpeed path ───────────────────────────────────────────────────────

    def build_ds_config(
        self,
        train_micro_batch_size_per_gpu: int = 1,
        gradient_accumulation_steps: int = 1,
        bf16: bool = True,
        offload_optimizer: bool = False,
    ) -> Dict:
        """Emit a DeepSpeed ZeRO-3 config dict matching this context's stage.

        Pure data — does not import or call DeepSpeed, so it is import-safe even
        when DeepSpeed is absent (useful for serializing the launch config).
        """
        # Cross-node all-gather/reduce-scatter bucket sizes. These are the knobs
        # §8.6/§2.13 cares about: on AMD multinode the cross-node all-gather is
        # the bottleneck, so larger buckets amortize launch latency. Honor the
        # context's configured ``reduce_bucket_size`` (no longer hardcoded 5M),
        # so the native DDP path and the DeepSpeed path use ONE source of truth.
        # The prefetch/persistence knobs scale off it so they stay proportional.
        bucket = self.ctx.config.reduce_bucket_size
        zero: Dict = {
            "stage": self.ctx.config.zero_stage,
            "reduce_bucket_size": bucket,
            "stage3_prefetch_bucket_size": bucket,
            "stage3_param_persistence_threshold": 1_000_000,
            "stage3_max_live_parameters": 1_000_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "contiguous_gradients": True,
            "overlap_comm": True,
        }
        if offload_optimizer:
            zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        cfg: Dict = {
            "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization": zero,
        }
        if bf16:
            cfg["bf16"] = {"enabled": True}
        return cfg

    # ── Native shim path ─────────────────────────────────────────────────────

    def partition_optimizer_state(
        self, named_params: Iterable[Tuple[str, "torch.Tensor"]]
    ) -> Dict[str, _ParamShard]:
        """Assign this DP rank its contiguous ``1/N`` slice of every parameter.

        Records the shard map used by :meth:`reduce_scatter_grads` /
        :meth:`all_gather_params`. The optimizer then only allocates state
        (m/v/…) for ``[start:end)`` of each param — the ZeRO-3 memory win.
        Single-process ⇒ every shard is the whole param (a clean degenerate).
        """
        self._shards.clear()
        world = self.dp_world
        rank = self.dp_rank
        for name, p in named_params:
            numel = p.numel()
            start, end = _even_partition(numel, world, rank)
            self._shards[name] = _ParamShard(name, numel, start, end)
        return dict(self._shards)

    def owned_slice(self, name: str, tensor: "torch.Tensor") -> "torch.Tensor":
        """Return this rank's owned flat slice of ``tensor`` (or all of it)."""
        shard = self._shards.get(name)
        flat = tensor.reshape(-1)
        if shard is None:
            return flat
        return flat[shard.start:shard.end]

    def reduce_scatter_grads(
        self, named_grads: Iterable[Tuple[str, "torch.Tensor"]]
    ) -> Dict[str, "torch.Tensor"]:
        """ZeRO-3 gradient reduce-scatter across the DP group.

        Each rank receives the *summed-then-averaged* gradient for **only its
        owned slice** of each parameter — the input the sharded local optimizer
        step consumes. Falls back to a plain copy of the local grad slice when
        DP size is 1 / no job is launched (single-process correctness).

        Implementation: a true fused ``reduce_scatter_tensor`` per parameter —
        each rank sends the whole (per-rank-padded) flat grad and receives only
        the reduced sum for its owned slice, which is half the traffic of the
        former ``all_reduce``-then-slice. The padded reduce-scatter convention
        matches :meth:`all_gather_params` / :meth:`_even_partition` (each rank
        owns ``ceil(numel/world)`` elements). Falls back to the local slice when
        DP size is 1 / no job is launched (single-process correctness).
        """
        out: Dict[str, "torch.Tensor"] = {}
        live = self.dp_world > 1 and is_dist_available_and_initialized()
        ws = self.dp_world
        if not live:
            for name, g in named_grads:
                out[name] = self.owned_slice(name, g).clone()
            return out

        dist = _dist()
        group = self.ctx.dp_group
        rstensor = getattr(dist, "reduce_scatter_tensor", None)
        for name, g in named_grads:
            flat = g.reshape(-1).contiguous()
            numel = flat.numel()
            per = (numel + ws - 1) // ws
            if rstensor is not None:
                # Pad the input to world*per so the scatter is uniform.
                padded = flat
                if per * ws != numel:
                    padded = flat.new_zeros(per * ws)
                    padded[:numel] = flat
                owned = flat.new_empty(per)
                rstensor(owned, padded, group=group)
                owned.div_(ws)
                # Trim padding off this rank's owned slice.
                shard = self._shards.get(name)
                keep = (shard.shard_numel if shard is not None
                        else max(0, min(per, numel - self.dp_rank * per)))
                out[name] = owned[:keep].clone()
            else:
                # Older torch without reduce_scatter_tensor: equivalent result
                # via all_reduce + owned slice.
                dist.all_reduce(flat, group=group)
                flat.div_(ws)
                out[name] = self.owned_slice(name, flat).clone()
        return out

    def all_gather_params(
        self, named_params: Iterable[Tuple[str, "torch.Tensor"]]
    ) -> None:
        """ZeRO-3 all-gather-on-step: reconstitute full params from DP slices.

        After each DP rank updates its owned slice, every rank needs the full
        parameter again for the next forward/backward. We use a fused
        ``all_gather_into_tensor`` (single contiguous output buffer) instead of
        the list-based ``all_gather``, then stitch the per-rank slices back into
        the param in place. No-op single-process. This is the §2.13
        cross-node-sensitive collective (the AMD multinode gap the harness
        records).
        """
        if self.dp_world <= 1 or not is_dist_available_and_initialized():
            return
        dist = _dist()
        world = self.dp_world
        group = self.ctx.dp_group
        agtensor = getattr(dist, "all_gather_into_tensor", None)
        for name, p in named_params:
            shard = self._shards.get(name)
            if shard is None:
                continue
            flat = p.reshape(-1)
            owned = flat[shard.start:shard.end].contiguous()
            # Pad to the common per-rank shard length so the gather is uniform.
            per = (shard.numel + world - 1) // world
            buf = owned.new_zeros(per)
            buf[: owned.numel()] = owned
            if agtensor is not None:
                gathered = buf.new_empty(per * world)
                agtensor(gathered, buf, group=group)
                # Stitch the contiguous [r*per:(r+1)*per] slices back in place.
                for r in range(world):
                    start = min(r * per, shard.numel)
                    end = min(start + per, shard.numel)
                    n = end - start
                    if n > 0:
                        flat[start:end].copy_(gathered[r * per:r * per + n])
            else:
                gathered = [torch.empty_like(buf) for _ in range(world)]
                dist.all_gather(gathered, buf, group=group)
                for r in range(world):
                    start = min(r * per, shard.numel)
                    end = min(start + per, shard.numel)
                    n = end - start
                    if n > 0:
                        flat[start:end].copy_(gathered[r][:n])


# Host NVSHMEM TP-team bootstrap (the launcher EDIT-E companion). Re-exported so a
# caller can `from grokking_optimizers.distributed import TPBootstrap` next to the TP
# weight-shard it pairs with. Import-safe: host_bringup is pure-Python (no torch,
# no NVSHMEM dlopen at import — discovery only).
from grokking_optimizers.host_bringup import (  # noqa: E402
    TPBootstrap,
    bootstrap_tp_team,
)

__all__ = [
    "ParallelConfig",
    "DistributedContext",
    "get_distributed_context",
    "is_dist_available_and_initialized",
    "deepspeed_available",
    "ZeRO3Sharder",
    # Tensor-parallel weight shard (§8.1 TP / dist_step.md §6.C.5).
    "partition_tensor_parallel",
    "TPShardPlan",
    "TPTensorShard",
    "TP_REPLICATED",
    "TP_COLUMN",
    "TP_ROW",
    # Host NVSHMEM TP-team bootstrap (run_harness.md / dist_step.md §6.D).
    "TPBootstrap",
    "bootstrap_tp_team",
]
