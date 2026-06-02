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
    """

    data_parallel: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    zero_stage: int = 0
    backend: str = "nccl"
    use_megakernel: bool = True

    def __post_init__(self) -> None:
        for name in ("data_parallel", "tensor_parallel", "pipeline_parallel"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.zero_stage}")

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
    dp_ranks: Tuple[int, ...]
    tp_ranks: Tuple[int, ...]
    pp_ranks: Tuple[int, ...]


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
    return _RankMesh(
        global_rank=rank,
        world_size=cfg.world_size,
        dp=dp_i,
        tp=tp_i,
        pp=pp_i,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
        pp_ranks=pp_ranks,
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
    def dp_world_size(self) -> int:
        return self.config.data_parallel

    @property
    def tp_world_size(self) -> int:
        return self.config.tensor_parallel

    @property
    def pp_world_size(self) -> int:
        return self.config.pipeline_parallel

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

    def all_reduce_dp_grads(self, params: Iterable["torch.Tensor"]) -> None:
        """Average gradients across the DP group (the DDP gradient all-reduce).

        Used for ZeRO-0/1/2, where gradients are replicated then averaged. Under
        ZeRO-3 the gradient is reduce-scattered instead (see
        :class:`ZeRO3Sharder`). No-op single-process.
        """
        if self.dp_world_size <= 1 or not is_dist_available_and_initialized():
            return
        dist = _dist()
        ws = self.dp_world_size
        for p in params:
            if getattr(p, "grad", None) is not None:
                dist.all_reduce(p.grad, group=self._dp_group)
                p.grad.div_(ws)


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
        zero: Dict = {
            "stage": self.ctx.config.zero_stage,
            # Cross-node all-gather/reduce-scatter bucket sizes. These are the
            # knobs §8.6/§2.13 cares about: on AMD multinode the cross-node
            # all-gather is the bottleneck, so larger buckets amortize launch
            # latency. Conservative defaults; the harness tunes per fabric.
            "reduce_bucket_size": 5_000_000,
            "stage3_prefetch_bucket_size": 5_000_000,
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

        Implementation note: we use ``all_reduce`` over the DP group then take
        the owned slice. That is bandwidth-equivalent in result to a true
        ``reduce_scatter`` for correctness; the production DeepSpeed path uses
        the native fused reduce-scatter (less traffic). The native shim favors
        a single robust collective that works across torch versions without the
        flat-bucket plumbing DeepSpeed already provides.
        """
        out: Dict[str, "torch.Tensor"] = {}
        live = self.dp_world > 1 and is_dist_available_and_initialized()
        dist = _dist() if live else None
        ws = self.dp_world
        for name, g in named_grads:
            if live:
                g = g.contiguous()
                dist.all_reduce(g, group=self.ctx.dp_group)
                g.div_(ws)
            out[name] = self.owned_slice(name, g).clone()
        return out

    def all_gather_params(
        self, named_params: Iterable[Tuple[str, "torch.Tensor"]]
    ) -> None:
        """ZeRO-3 all-gather-on-step: reconstitute full params from DP slices.

        After each DP rank updates its owned slice, every rank needs the full
        parameter again for the next forward/backward. We all-gather the
        owned slices and scatter them back into the param buffer in place.
        No-op single-process. This is the §2.13 cross-node-sensitive collective
        (the AMD multinode gap the harness records).
        """
        if self.dp_world <= 1 or not is_dist_available_and_initialized():
            return
        dist = _dist()
        world = self.dp_world
        group = self.ctx.dp_group
        for name, p in named_params:
            shard = self._shards.get(name)
            if shard is None:
                continue
            flat = p.reshape(-1)
            owned = flat[shard.start:shard.end].contiguous()
            # Pad to the common per-rank shard length so all_gather is uniform.
            per = (shard.numel + world - 1) // world
            buf = owned.new_zeros(per)
            buf[: owned.numel()] = owned
            gathered = [torch.empty_like(buf) for _ in range(world)]
            dist.all_gather(gathered, buf, group=group)
            # Stitch the gathered slices back into the full flat param.
            for r in range(world):
                start = min(r * per, shard.numel)
                end = min(start + per, shard.numel)
                n = end - start
                if n > 0:
                    flat[start:end].copy_(gathered[r][:n])


__all__ = [
    "ParallelConfig",
    "DistributedContext",
    "get_distributed_context",
    "is_dist_available_and_initialized",
    "deepspeed_available",
    "ZeRO3Sharder",
]
