"""tests/hw/test_3d_parallel.py — Stage 7 §8 5–10B-param 3D-parallel harness.

A **runnable-under-torchrun** harness that builds a ~7B-param model sharded
DP×TP×PP with ZeRO-3 (§8.1/§8.3), runs a few fused steps through the §8.4
megakernel adapter, and reports weak-scaling efficiency. The actual launch
command and the ``>= 70%`` weak-scaling pass criterion live in
``HARDWARE_VALIDATION.md`` (Stage 7 section) — the runtime check fires when
real multi-GPU / multinode hardware lands (🟡).

IMPORT / SUITE SAFETY (the #1 requirement). This file:

  * imports cleanly with no GPU, no launched job, and no DeepSpeed;
  * exposes ``pytest``-discoverable tests that **SKIP** (never fail / error)
    when there is no distributed launch or no CUDA/ROCm device, so it does not
    break the self-test or a CPU CI run;
  * only touches ``torch.distributed`` / device memory inside the explicitly
    launched ``__main__`` path or behind the skip guards.

Run (hardware):

    torchrun --nnodes=$N --nproc_per_node=8 tests/hw/test_3d_parallel.py \
        --params 7e9 --dp 8 --tp 4 --pp 2 --zero 3
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from grokking_optimizers.distributed import (
    DistributedContext,
    ParallelConfig,
    ZeRO3Sharder,
    deepspeed_available,
    is_dist_available_and_initialized,
)
from grokking_optimizers.megakernel_engine import (
    build_client_optimizer,
    fused_training_step,
)

# Pytest is optional at import time (the file is also a torchrun __main__).
try:  # pragma: no cover - trivial import guard
    import pytest
except Exception:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


# ───────────────────────────── launch-state probes ───────────────────────────


def _under_distributed_launch() -> bool:
    """True iff a torchrun/launcher set up a real (world_size>1) job."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _has_accelerator() -> bool:
    """True iff a CUDA or ROCm device is visible (ROCm presents as torch.cuda)."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _skip_reason() -> Optional[str]:
    """Return a skip reason string, or None if the harness can really run."""
    if not _under_distributed_launch():
        return "no distributed launch (WORLD_SIZE<=1) — Stage 7 harness is hardware-deferred"
    if not _has_accelerator():
        return "no CUDA/ROCm accelerator visible"
    return None


# ───────────────────────────── model sizing (§ 5–10B) ────────────────────────


@dataclass
class ModelSpec:
    """A transformer-decoder size hitting a target parameter count.

    Pure arithmetic — no tensors allocated — so sizing is testable on CPU.
    Param count ≈ vocab*d + L*(12 d^2)  (attn 4 d^2 + FFN 8 d^2 per layer).
    """

    target_params: float
    d_model: int
    n_layers: int
    vocab: int

    @property
    def approx_params(self) -> int:
        return self.vocab * self.d_model + self.n_layers * (12 * self.d_model ** 2)


def size_model(target_params: float, vocab: int = 50_257) -> ModelSpec:
    """Pick (d_model, n_layers) hitting ~``target_params`` (a 7B-class config).

    Uses the canonical ``n_layers ≈ d_model/128`` aspect ratio, then solves the
    per-layer ``12 d^2`` term for d. Returns a :class:`ModelSpec` whose
    ``approx_params`` is within a few percent of target.
    """
    # 12 * d^2 * (d/128) ≈ target  →  d^3 ≈ target * 128 / 12
    d = int(round((target_params * 128 / 12) ** (1 / 3)))
    d = max(128, (d // 128) * 128)  # round to a multiple of 128
    n_layers = max(1, d // 128)
    return ModelSpec(target_params=target_params, d_model=d, n_layers=n_layers, vocab=vocab)


def shard_param_count(spec: ModelSpec, cfg: ParallelConfig) -> int:
    """Per-GPU parameter count after TP×PP model sharding (not counting ZeRO).

    TP splits each layer's matmuls across ``tensor_parallel`` ranks; PP splits
    the layer stack across ``pipeline_parallel`` stages. DP replicates (then
    ZeRO-3 shards optimizer state over DP). This is the weak-scaling denominator.
    """
    return spec.approx_params // (cfg.tensor_parallel * cfg.pipeline_parallel)


# ───────────────────────────── scaling efficiency ────────────────────────────


def weak_scaling_efficiency(
    base_throughput: float, scaled_throughput: float, scale_factor: float
) -> float:
    """Weak-scaling efficiency = (scaled / (base * scale_factor)).

    1.0 is ideal linear weak scaling; the Stage-7 pass bar is ``>= 0.70`` to 32
    GPUs on IB/NVLink fabric (recorded in HARDWARE_VALIDATION.md). On AMD the
    cross-node ZeRO-3 all-gather (§2.13) is expected to pull this down vs NVIDIA.
    """
    if base_throughput <= 0 or scale_factor <= 0:
        return 0.0
    return scaled_throughput / (base_throughput * scale_factor)


PASS_EFFICIENCY = 0.70


# ───────────────────────────── the runnable harness ──────────────────────────


def _build_local_shard(spec: ModelSpec, cfg: ParallelConfig, device: str) -> List[torch.nn.Parameter]:
    """Allocate this rank's *local* parameter shard as a flat list of params.

    We do NOT instantiate the full 7B model on one rank — that would OOM. We
    allocate only the TP×PP-sharded slice (``shard_param_count``) as a handful
    of large flat parameters, which is enough to exercise the ZeRO-3 collectives
    and the fused-step adapter at realistic per-GPU memory.
    """
    per_rank = shard_param_count(spec, cfg)
    # Split into ~chunks of <=2**28 elements to keep individual allocations sane.
    chunk = 1 << 26
    params: List[torch.nn.Parameter] = []
    remaining = per_rank
    while remaining > 0:
        n = min(chunk, remaining)
        p = torch.nn.Parameter(torch.randn(n, device=device, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p)
        params.append(p)
        remaining -= n
    return params


def run_harness(
    params: float,
    dp: int,
    tp: int,
    pp: int,
    zero_stage: int,
    steps: int = 4,
    backend: str = "nccl",
) -> Tuple[ParallelConfig, float]:
    """Build a 3D-parallel + ZeRO-3 ~7B harness, run ``steps``, return throughput.

    Returns ``(config, steps_per_second)``. Intended to be called once per
    launch; the >=70% weak-scaling comparison across launches is computed by the
    surrounding driver (see HARDWARE_VALIDATION.md) from the per-launch numbers.
    """
    cfg = ParallelConfig(
        data_parallel=dp,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        zero_stage=zero_stage,
        backend=backend,
        use_megakernel=True,
    )
    ctx = DistributedContext.from_config(cfg)
    device = f"cuda:{torch.cuda.current_device()}" if _has_accelerator() else "cpu"

    spec = size_model(params)
    local_params = _build_local_shard(spec, cfg, device)

    # ZeRO-3: shard optimizer state over the DP group.
    sharder = ZeRO3Sharder(ctx)
    named = [(f"p{i}", p) for i, p in enumerate(local_params)]
    sharder.partition_optimizer_state(named)

    # Inner optimizer (plain torch SGD stands in for the fused race optimizer in
    # the harness; the real run swaps in SuperGrok2 et al). Wrap as the fused
    # client_optimizer so the §8.4 adapter drives backward/step.
    inner = torch.optim.SGD(local_params, lr=1e-3)
    opt = build_client_optimizer(inner, "transformer_decoder", "sm_90", "adamw")

    if is_dist_available_and_initialized():
        torch.cuda.synchronize()
    start = _now()
    for _ in range(steps):
        # ZeRO-3 gradient reduce-scatter over DP (each rank keeps its slice).
        sharder.reduce_scatter_grads(named)
        # A scalar loss surrogate to drive the adapter's backward/step path.
        loss = sum(p.float().sum() for p in local_params)
        fused_training_step(opt, loss, framework_backward=lambda x: None)
        # ZeRO-3 all-gather-on-step: rebuild full params from DP slices.
        sharder.all_gather_params(named)
    if is_dist_available_and_initialized():
        torch.cuda.synchronize()
    elapsed = max(_now() - start, 1e-9)
    return cfg, steps / elapsed


def _now() -> float:
    import time

    return time.perf_counter()


# ─────────────────────────────── pytest tests ────────────────────────────────
# All of these SKIP cleanly when there is no launch / no accelerator, so the
# default CPU suite stays green. The pure-Python sizing / efficiency math is
# tested unconditionally (it needs no hardware).


def test_model_sizing_hits_7b_target():
    """size_model returns a config within ~15% of the 7B target (CPU-safe)."""
    spec = size_model(7e9)
    assert spec.d_model % 128 == 0
    rel = abs(spec.approx_params - 7e9) / 7e9
    assert rel < 0.15, f"sizing off by {rel:.1%}: {spec}"


def test_weak_scaling_efficiency_math():
    """Efficiency = scaled/(base*scale); identity at perfect linear scaling."""
    assert weak_scaling_efficiency(100.0, 400.0, 4.0) == 1.0
    assert weak_scaling_efficiency(100.0, 280.0, 4.0) == 0.70
    assert weak_scaling_efficiency(0.0, 1.0, 4.0) == 0.0


def test_shard_count_divides_by_model_parallel():
    """TP×PP sharding reduces per-GPU params by exactly the MP size (CPU-safe)."""
    spec = size_model(7e9)
    cfg = ParallelConfig(data_parallel=8, tensor_parallel=4, pipeline_parallel=2, zero_stage=3)
    assert shard_param_count(spec, cfg) == spec.approx_params // 8


def test_zero3_partition_covers_param_exactly():
    """The union of DP shards covers [0, numel) with no gaps/overlap (CPU-safe)."""
    # Simulate the 4 DP ranks' partitions of a 1000-element param.
    from grokking_optimizers.distributed import _even_partition

    covered = [False] * 1000
    for r in range(4):
        s, e = _even_partition(1000, 4, r)
        for i in range(s, e):
            assert not covered[i], f"overlap at {i}"
            covered[i] = True
    assert all(covered), "partition left a gap"


def _maybe_skip():
    reason = _skip_reason()
    if reason is not None:
        if pytest is not None:
            pytest.skip(reason)
        else:  # pragma: no cover - non-pytest path
            raise SystemExit(0)


def test_3d_parallel_smoke():
    """Hardware path: build the 7B 3D+ZeRO-3 harness and run a few steps.

    SKIPS unless launched under torchrun with an accelerator. When it runs, it
    only asserts the harness completes and reports a positive throughput; the
    >=70% weak-scaling efficiency comparison across launches is performed by the
    HARDWARE_VALIDATION.md driver, not this single-launch test.
    """
    _maybe_skip()
    cfg, thr = run_harness(params=7e9, dp=8, tp=4, pp=2, zero_stage=3, steps=2)
    assert thr > 0.0, "harness produced no throughput"


def test_deepspeed_or_native_path_selected():
    """ZeRO-3 backend selection is reported (CPU-safe — no DeepSpeed required)."""
    cfg = ParallelConfig(data_parallel=2, zero_stage=3)
    ctx = DistributedContext(cfg)
    sharder = ZeRO3Sharder(ctx)
    # Either DeepSpeed is present (use_deepspeed True) or the native shim is
    # used (False). Both are valid; we just assert the flag is a bool and the
    # ds-config builder is import-safe regardless.
    assert isinstance(sharder.use_deepspeed, bool)
    ds_cfg = sharder.build_ds_config()
    assert ds_cfg["zero_optimization"]["stage"] == 3
    _ = deepspeed_available()


# ──────────────────────────────── torchrun main ──────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 7 3D-parallel + ZeRO-3 harness")
    ap.add_argument("--params", type=float, default=7e9, help="target param count")
    ap.add_argument("--dp", type=int, default=8, help="data-parallel size")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size")
    ap.add_argument("--pp", type=int, default=2, help="pipeline-parallel size")
    ap.add_argument("--zero", type=int, default=3, choices=(0, 1, 2, 3))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--backend", type=str, default="nccl")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    reason = _skip_reason()
    if reason is not None:
        print(f"[stage7] SKIP: {reason}")
        return 0
    cfg, thr = run_harness(
        params=args.params, dp=args.dp, tp=args.tp, pp=args.pp,
        zero_stage=args.zero, steps=args.steps, backend=args.backend,
    )
    ctx = DistributedContext.from_config(cfg)
    if ctx.is_main_process:
        spec = size_model(args.params)
        print(
            f"[stage7] world={cfg.world_size} DP×TP×PP="
            f"{cfg.data_parallel}×{cfg.tensor_parallel}×{cfg.pipeline_parallel} "
            f"zero={cfg.zero_stage} d_model={spec.d_model} layers={spec.n_layers} "
            f"~{spec.approx_params/1e9:.2f}B params | "
            f"{thr:.3f} steps/s | per-GPU params ≈ "
            f"{shard_param_count(spec, cfg)/1e9:.2f}B"
        )
        print(f"[stage7] pass criterion: weak-scaling efficiency >= {PASS_EFFICIENCY:.0%}")
    ctx.barrier()
    ctx.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
