"""grokking_optimizers/parallel/pipeline.py — the PIPELINE-PARALLEL schedule
(1F1B), stage partition of the megakernel step, and activation hand-off plan.

IMPLEMENTS /workspace/.parallelism_design.md §4 (task #25 phase-2, Lane C):
  * §4.1 stage partitioning — contiguous layer ranges + the per-stage TENSOR
    ownership of the megakernel's flat layout (the host mirror of the
    kernel-side ``PPStageSpec`` in csrc/fused/sm_90/pp_stage_decoder_tc.cuh;
    the two are cross-checked by tests/hw/test_pp2_loopback_determinism.py::
    test_pp2_stage_ownership_matches_python_plan).
  * §4.1 hand-off buffers — the fwd boundary payload is the acts X_in[L]
    region (bf16 [T,d], already in HBM by the acts-to-HBM design) and the bwd
    payload is the fp32 [T,d] boundary adjoint (fp32 ON PURPOSE: it is the
    bit-preserving carrier — see the layer-range patch).
  * §4.2 the 1F1B microbatch schedule (standard Megatron non-interleaved
    1F1B: warmup fwds, steady 1F1B, cooldown bwds) + a dependency-simulating
    validator + the generic host-side driver ``run_1f1b``.

PURE PYTHON, no torch required (design §7: CPU-validatable now). The driver
takes callables; the GPU test binds them to the real stage-kernel launches,
the CPU test binds recording stubs. The transport is a swap point:
``LoopbackP2P`` (in-process, single device — used by the 1-GPU tests) vs the
real ``torch.distributed`` P2P on the pp_group (8×H100 window; the existing
``DistributedContext.pp_group`` from distributed.py is the comm fabric).

HONEST SCOPE (design §4.3): PP is pure overhead at the 2-layer race depth —
these pieces exist because the FLAGSHIP ships full 4D (owner-locked) and the
machinery must be validated before the rental window. The loopback gate proves
stage-composition CORRECTNESS (bit-exactness); PP's throughput contribution is
an 8×-window measurement, never assumed.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────
#  Stage partition (§4.1).
# ──────────────────────────────────────────────────────────────────────────


def stage_layer_ranges(n_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    """Contiguous even layer split: stage s owns layers [s*L/P, (s+1)*L/P).

    Mirrors ``PPStageSpec``'s static_asserts: requires ``num_stages | n_layers``
    (loud ValueError otherwise — no silent uneven split; an uneven split is a
    DIFFERENT design decision that must be taken explicitly).
    """
    if num_stages < 1 or n_layers < 1:
        raise ValueError(f"need n_layers>=1, num_stages>=1; got {n_layers}, {num_stages}")
    if num_stages > n_layers:
        raise ValueError(f"num_stages={num_stages} > n_layers={n_layers}")
    if n_layers % num_stages != 0:
        raise ValueError(
            f"contiguous even split requires num_stages | n_layers "
            f"({num_stages} ∤ {n_layers}) — mirror of PPStageSpec's static_assert")
    per = n_layers // num_stages
    return [(s * per, (s + 1) * per) for s in range(num_stages)]


# Declarative flat-layout structure per model (named_parameters() order — the
# SINGLE structural fact the ownership map needs; counts mirror the generated
# layout headers, e.g. decoder_layout.cuh's 30 = 2 + 2*12 + 4). No numeric
# offsets here: offsets/sizes stay owned by the generated layout headers.
_MODEL_LAYOUT_STRUCTURE: Dict[str, Dict[str, object]] = {
    "decoder": {
        "n_layers": 2,
        "prefix": 2,        # tok.weight, pos.weight        → FIRST stage
        "per_layer": 12,    # in_w,in_b,out_w,out_b,n1w,n1b,n2w,n2b,ff0w,ff0b,ff2w,ff2b
        "suffix": 4,        # norm.w, norm.b, out.weight, out.bias → LAST stage
    },
}


def stage_tensor_ownership(model: str, num_stages: int) -> List[List[int]]:
    """Per-stage owned flat-tensor indices (named_parameters() order) — §4.1.

    Stage owning layers [Llo,Lhi) owns the per-layer tensor block
    [prefix + per_layer*Llo, prefix + per_layer*Lhi); the FIRST stage adds the
    prefix (embeddings); the LAST adds the suffix (final norm + head). This is
    the host mirror of ``PPStageSpec::owns_tensor`` — kept structurally
    identical and pinned by the kernel-side cross-check test.
    """
    if model not in _MODEL_LAYOUT_STRUCTURE:
        raise ValueError(
            f"unknown model {model!r} for PP ownership "
            f"(have {sorted(_MODEL_LAYOUT_STRUCTURE)}); add its layout structure "
            f"— do NOT default silently")
    st = _MODEL_LAYOUT_STRUCTURE[model]
    n_layers = int(st["n_layers"])
    prefix = int(st["prefix"])
    per_layer = int(st["per_layer"])
    suffix = int(st["suffix"])
    ranges = stage_layer_ranges(n_layers, num_stages)
    total = prefix + per_layer * n_layers + suffix
    out: List[List[int]] = []
    for s, (llo, lhi) in enumerate(ranges):
        own = list(range(prefix + per_layer * llo, prefix + per_layer * lhi))
        if s == 0:
            own = list(range(prefix)) + own
        if s == num_stages - 1:
            own += list(range(total - suffix, total))
        out.append(sorted(own))
    # Invariant: disjoint + complete (loud — a partition bug corrupts grads).
    flat = [t for o in out for t in o]
    if sorted(flat) != list(range(total)):
        raise AssertionError(f"ownership not a partition of [0,{total}): {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Hand-off buffer plan (§4.1).
# ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class HandoffSpec:
    """One stage boundary's per-microbatch payloads.

    fwd: the boundary activation X_in[L] — bf16 [T_mb, d] (T_mb = mb rows × seq).
    bwd: the boundary adjoint dh — fp32 [T_mb, d] (the bit-preserving carrier).
    """

    boundary_layer: int       # the L of X_in[L] (== upstream stage's Lhi)
    fwd_numel: int            # T_mb * d  (bf16 elements)
    fwd_bytes: int
    bwd_numel: int            # T_mb * d  (fp32 elements)
    bwd_bytes: int


def handoff_plan(d_model: int, seq: int, microbatch_rows: int,
                 n_layers: int, num_stages: int) -> List[HandoffSpec]:
    """The §4.1 hand-off plan: one spec per stage boundary (num_stages-1)."""
    ranges = stage_layer_ranges(n_layers, num_stages)
    t_mb = microbatch_rows * seq
    out = []
    for s in range(num_stages - 1):
        boundary = ranges[s][1]
        n = t_mb * d_model
        out.append(HandoffSpec(boundary_layer=boundary,
                               fwd_numel=n, fwd_bytes=2 * n,    # bf16
                               bwd_numel=n, bwd_bytes=4 * n))   # fp32
    return out


def max_inflight_microbatches(num_stages: int, num_microbatches: int,
                              stage: int) -> int:
    """1F1B liveness bound: stage s holds ≤ min(P - s, M) microbatches' fwd
    state (handoff inputs) live at once — the per-stage buffer count."""
    return min(num_stages - stage, num_microbatches)


# ──────────────────────────────────────────────────────────────────────────
#  1F1B schedule (§4.2) — standard Megatron non-interleaved.
# ──────────────────────────────────────────────────────────────────────────

Op = Tuple[str, int]          # ("fwd"|"bwd", microbatch_index)


def build_1f1b_schedule(num_stages: int, num_microbatches: int) -> List[List[Op]]:
    """Per-stage op order for non-interleaved 1F1B.

    Stage s: warmup = min(P-1-s, M) fwds; steady: (M - warmup) × (fwd, bwd)
    pairs; cooldown: the remaining warmup bwds. M >= 1; P >= 1. Total per
    stage: M fwds + M bwds, each microbatch exactly once.
    """
    if num_stages < 1 or num_microbatches < 1:
        raise ValueError("num_stages and num_microbatches must be >= 1")
    P, M = num_stages, num_microbatches
    sched: List[List[Op]] = []
    for s in range(P):
        ops: List[Op] = []
        warmup = min(P - 1 - s, M)
        steady = M - warmup
        f = 0  # next fwd mb
        b = 0  # next bwd mb
        for _ in range(warmup):
            ops.append(("fwd", f))
            f += 1
        for _ in range(steady):
            ops.append(("fwd", f))
            f += 1
            ops.append(("bwd", b))
            b += 1
        while b < M:
            ops.append(("bwd", b))
            b += 1
        sched.append(ops)
    return sched


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """The textbook 1F1B bubble: (P-1)/(M+P-1) — §4.3's honest cost number."""
    P, M = num_stages, num_microbatches
    return (P - 1) / (M + P - 1)


def validate_1f1b_schedule(sched: List[List[Op]], num_microbatches: int) -> None:
    """Simulate the cross-stage dependency graph; raise on any violation.

    Dependencies (the §4.1 dataflow):
      fwd(s, mb) needs fwd(s-1, mb)   (the fwd activation handoff)
      bwd(s, mb) needs fwd(s, mb) and bwd(s+1, mb)  (the dh handoff)
    The greedy event-loop simulation must drain every stage's queue — a stall
    with no runnable head op is a deadlock ⇒ schedule bug. Also asserts each
    stage runs every microbatch exactly once per direction and respects the
    1F1B liveness bound (max_inflight_microbatches).
    """
    P = len(sched)
    M = num_microbatches
    for s, ops in enumerate(sched):
        fwds = [mb for kind, mb in ops if kind == "fwd"]
        bwds = [mb for kind, mb in ops if kind == "bwd"]
        if sorted(fwds) != list(range(M)) or sorted(bwds) != list(range(M)):
            raise AssertionError(f"stage {s}: each mb must fwd+bwd exactly once: {ops}")
    done_f = [[False] * M for _ in range(P)]
    done_b = [[False] * M for _ in range(P)]
    heads = [0] * P
    inflight_peak = [0] * P
    inflight = [0] * P
    progressed = True
    while progressed:
        progressed = False
        for s in range(P):
            while heads[s] < len(sched[s]):
                kind, mb = sched[s][heads[s]]
                if kind == "fwd":
                    ready = (s == 0) or done_f[s - 1][mb]
                    if not ready:
                        break
                    done_f[s][mb] = True
                    inflight[s] += 1
                    inflight_peak[s] = max(inflight_peak[s], inflight[s])
                else:
                    ready = done_f[s][mb] and ((s == P - 1) or done_b[s + 1][mb])
                    if not ready:
                        break
                    done_b[s][mb] = True
                    inflight[s] -= 1
                heads[s] += 1
                progressed = True
    stalled = [s for s in range(P) if heads[s] < len(sched[s])]
    if stalled:
        raise AssertionError(
            f"1F1B schedule DEADLOCKED at stages {stalled} "
            f"(heads={heads}, lens={[len(x) for x in sched]})")
    for s in range(P):
        bound = max_inflight_microbatches(P, M, s)
        if inflight_peak[s] > bound:
            raise AssertionError(
                f"stage {s}: in-flight peak {inflight_peak[s]} exceeds the 1F1B "
                f"bound {bound} — the memory guarantee is broken")


# ──────────────────────────────────────────────────────────────────────────
#  Transport swap point + the generic host-side driver (§4.2).
# ──────────────────────────────────────────────────────────────────────────


class LoopbackP2P:
    """In-process P2P for the single-GPU loopback: send/recv keyed by
    (src_stage, dst_stage, microbatch, tag). HONEST semantics: recv of a
    never-sent key raises (a real recv would block forever — we surface the
    schedule bug loudly instead); each payload is consumed exactly once.

    The 8×H100 swap-in is ``torch.distributed`` isend/irecv on the pp_group
    (``DistributedContext.pp_group``); the driver below only touches this
    send/recv surface, so the swap changes no driver logic.
    """

    def __init__(self) -> None:
        self._box: Dict[Tuple[int, int, int, str], object] = {}
        self.sends = 0
        self.recvs = 0

    def send(self, src: int, dst: int, mb: int, tag: str, payload: object) -> None:
        key = (src, dst, mb, tag)
        if key in self._box:
            raise AssertionError(f"double-send {key} (unconsumed payload)")
        self._box[key] = payload
        self.sends += 1

    def recv(self, src: int, dst: int, mb: int, tag: str) -> object:
        key = (src, dst, mb, tag)
        if key not in self._box:
            raise AssertionError(
                f"recv-before-send {key} — schedule dependency violation "
                f"(a real P2P recv would deadlock here)")
        self.recvs += 1
        return self._box.pop(key)

    def drained(self) -> bool:
        return not self._box


def run_1f1b(num_stages: int, num_microbatches: int,
             fwd_fn: Callable[[int, int, Optional[object]], object],
             bwd_fn: Callable[[int, int, Optional[object]], Optional[object]],
             transport: Optional[LoopbackP2P] = None,
             schedule: Optional[List[List[Op]]] = None) -> LoopbackP2P:
    """Execute a (validated) 1F1B schedule against stage callables.

    fwd_fn(stage, mb, fwd_in)  -> fwd_out  (the boundary activation payload;
        fwd_in is None for stage 0, else the upstream handoff). For the LAST
        stage the return value is ignored (no downstream).
    bwd_fn(stage, mb, dh_in)   -> dh_out   (the fp32 boundary adjoint; dh_in is
        None for the LAST stage — it starts from CE; return None for stage 0).

    Single-process driver: executes ops in a dependency-respecting global
    order (the same event loop the validator uses), calling each stage's ops
    in ITS schedule order — semantically what P concurrent rank processes do,
    minus the parallelism (the 1-GPU loopback shape; the 8× driver runs one
    rank's lane per process with real P2P). Returns the transport (for
    drain/count asserts).
    """
    P, M = num_stages, num_microbatches
    sched = schedule if schedule is not None else build_1f1b_schedule(P, M)
    validate_1f1b_schedule(sched, M)
    tr = transport if transport is not None else LoopbackP2P()

    done_f = [[False] * M for _ in range(P)]
    done_b = [[False] * M for _ in range(P)]
    heads = [0] * P
    progressed = True
    while progressed:
        progressed = False
        for s in range(P):
            while heads[s] < len(sched[s]):
                kind, mb = sched[s][heads[s]]
                if kind == "fwd":
                    if not (s == 0 or done_f[s - 1][mb]):
                        break
                    fwd_in = None if s == 0 else tr.recv(s - 1, s, mb, "act")
                    out = fwd_fn(s, mb, fwd_in)
                    if s < P - 1:
                        tr.send(s, s + 1, mb, "act", out)
                    done_f[s][mb] = True
                else:
                    if not (done_f[s][mb] and (s == P - 1 or done_b[s + 1][mb])):
                        break
                    dh_in = None if s == P - 1 else tr.recv(s + 1, s, mb, "dh")
                    dh_out = bwd_fn(s, mb, dh_in)
                    if s > 0:
                        tr.send(s, s - 1, mb, "dh", dh_out)
                    done_b[s][mb] = True
                heads[s] += 1
                progressed = True
    stalled = [s for s in range(P) if heads[s] < len(sched[s])]
    if stalled:
        raise AssertionError(f"run_1f1b deadlocked at stages {stalled}")
    if not tr.drained():
        raise AssertionError(f"undrained handoffs: {list(tr._box)}")
    return tr


__all__ = [
    "stage_layer_ranges",
    "stage_tensor_ownership",
    "HandoffSpec",
    "handoff_plan",
    "max_inflight_microbatches",
    "build_1f1b_schedule",
    "bubble_fraction",
    "validate_1f1b_schedule",
    "LoopbackP2P",
    "run_1f1b",
]
