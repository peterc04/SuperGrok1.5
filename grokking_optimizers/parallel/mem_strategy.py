"""grokking_optimizers/parallel/mem_strategy.py — the MEMORY-STRATEGY PLANNER.

From (model size/shape + hardware: #GPUs, HBM/GPU, host RAM, interconnect) it
decides the per-rank memory strategy — in-HBM | optimizer host-offload | activation
recompute | layer-streaming — by MEMORY FIT, never by GPU count. It is the sibling
of the live per-rank budget arithmetic: that code sizes the IN-HBM footprint; this
one applies the strategy SAVINGS and picks the minimal strategy set that fits.

DRIVER (USER DIRECTIVE): strategy decisions MUST NOT key on GPU count. 10M-on-1GPU →
in-HBM (trivial); 10B-on-1GPU → offload+recompute+streaming; 1.5B-on-8GPU → 4D+ZeRO-3.
Same model on 1 vs 8 GPUs differs ONLY because the per-GPU budget changes — the
decision is fit(footprint(strategy), usable_hbm), full stop.

PURE PYTHON — no torch, no GPU. Unit-testable on CPU; the harness prints the plan in
--dry-run BEFORE any GPU work (the same proof contract as the live budget model).

The savings model (all per-rank, after TP/PP/ZeRO division which the budget model does):
  * OPT HOST-OFFLOAD  : optimizer state (k*total floats) lives in PINNED HOST RAM;
                        per-step it is staged in tiles, so the RESIDENT device cost is
                        ~one stage tile, not k*total. Budget: state_gb -> ~offload_tile_gb.
                        REQUIRES: host RAM >= state bytes; PCIe/NVLink bw >= step needs.
  * ACT RECOMPUTE     : store ONLY layer-boundary acts (the per-layer X_in inputs), drop
                        the interior (X_ctx/X_x1/X_gact + dY caches), recompute the layer
                        fwd in bwd. Budget: acts_gb -> acts_gb * (boundary_floats/full).
  * LAYER STREAMING   : weights live in PINNED HOST RAM; a ring keeps `stream_depth`
                        layers resident. Budget: params_gb -> params_gb * stream_depth/L.
                        REQUIRES: host RAM >= param bytes; bw >= per-layer fetch / compute.

DEPENDENCY NOTE (deviation from the memory_strategy.md spec text): the spec wrote this
module against a sibling ``grokking_optimizers.parallel.flagship_budget`` (the pending
run_harness.md NEW FILE 1) which is NOT yet on disk. The live, on-disk budget source of
truth is ``grokking_optimizers.parallel.resource_planner`` (same scratch formulas,
mirrors fused_decoder_megakernel.cuh dec_tc_*_floats). To keep this module import-safe
(the gate ``python -c "import grokking_optimizers.parallel.mem_strategy"``) AND keep ONE
source of truth for the fit arithmetic, ``fb`` below is a thin internal ADAPTER over
``resource_planner`` that exposes exactly the flagship_budget-style names the planner
body uses (per_rank_budget / dec_tc_acts_floats / auto_ncta / H100_SAFETY_GIB /
FLAGSHIP_*). When ``flagship_budget`` lands, this adapter is the only block to swap; the
public surface (Hardware / MemPlan / plan_memory_strategy) and the emitted SG_MEM_*
macros are unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from grokking_optimizers.parallel import resource_planner as _rp


# ─────────────────────────────────────────────────────────────────────────────
#  ``fb`` ADAPTER — the flagship_budget-style budget API the planner body reads,
#  implemented over the live resource_planner scratch formulas (the single source
#  of truth that mirrors fused_decoder_megakernel.cuh dec_tc_*_floats). PURE PYTHON.
#
#  This is the ONLY block that knows the underlying budget module. It hands the
#  planner the same names the memory_strategy.md spec assumed on flagship_budget:
#    H100_SAFETY_GIB, FLAGSHIP_D/VOCAB/SEQ/LAYERS/TOTAL_PARAMS,
#    dec_tc_acts_floats(B, layers), per_rank_budget(...) -> RankBudget, auto_ncta(...).
# ─────────────────────────────────────────────────────────────────────────────
class _FlagshipBudgetAdapter:
    # Flagship (GPT-2 XL tier) reference shape — matches decoder_flagship_layout.cuh
    # (d=1600, L=48, vocab=99, seq=4 ⇒ total 1,475,884,899) and resource_planner's
    # is_flagship check (:431). These are the planner's default model descriptors.
    FLAGSHIP_D: int = 1600
    FLAGSHIP_LAYERS: int = 48
    FLAGSHIP_VOCAB: int = 99
    FLAGSHIP_SEQ: int = 4
    FLAGSHIP_TOTAL_PARAMS: int = 1_475_884_899
    # The HBM safety margin (ctx + cuBLAS/cuDNN + NCCL buffers), the same 4 GiB
    # reserve resource_planner.HardwareConfig.safety_gib defaults to.
    H100_SAFETY_GIB: float = _rp.HardwareConfig.safety_gib

    # The model the planner sizes against. Set by plan_memory_strategy from its args so
    # the adapter's budget calls use the run's true shape (not just the flagship pin).
    _model: Optional["_rp.ModelConfig"] = None
    _hw: Optional["_rp.HardwareConfig"] = None

    def dec_tc_acts_floats(self, B: int, layers: int) -> int:
        """flagship_budget-style acts size: full per-layer fwd+bwd cache for `layers`
        live layers at batch B. Delegates to the live mirror (resource_planner:272)."""
        m = self._model
        d = m.d if m is not None else self.FLAGSHIP_D
        vocab = m.vocab if m is not None else self.FLAGSHIP_VOCAB
        seq = m.seq if m is not None else self.FLAGSHIP_SEQ
        return _rp.dec_tc_acts_floats(B, d, vocab, max(layers, 1), seq)

    def _resolved(self, opt: str, B: int):
        m = self._model or _rp.ModelConfig(
            d=self.FLAGSHIP_D, layers=self.FLAGSHIP_LAYERS,
            seq=self.FLAGSHIP_SEQ, vocab=self.FLAGSHIP_VOCAB,
            batch=B, optimizer=opt)
        # Honor the run's optimizer + batch on top of the model shape.
        m = dataclasses.replace(m, optimizer=opt, batch=B)
        hw = self._hw or _rp.HardwareConfig()
        return m, hw

    def per_rank_budget(self, opt: str, *, tp: int, pp: int, dp: int,
                        zero3: bool, ncta: int, B: int) -> "_RankBudget":
        """flagship_budget-style per-rank IN-HBM footprint for one (mesh, ncta) point.
        Wraps resource_planner.per_rank_budget; returns a RankBudget with the
        params_gb/state_gb/acts_gb/staged_gb fields the planner's savings model reads."""
        m, hw = self._resolved(opt, B)
        total, n_tensors, nmax = _rp.layout_arith(m)
        mesh = _rp.Mesh(dp=dp, tp=tp, pp=pp, sp=1, ep=1)
        flags = _rp.MemFlags(need_zero_offload=bool(zero3))
        # staged carve elided ONLY for adamw single-opt (the bench-layout gate); SG2
        # and the other staged opts always carry their staged scratch (resource_planner:504).
        staged_needed = opt != "adamw"
        b = _rp.per_rank_budget(m, hw, mesh, flags, ncta,
                                total=total, n_tensors=n_tensors, nmax=nmax,
                                staged_scratch_needed=staged_needed)
        return _RankBudget(params_gb=b.params, state_gb=b.state,
                           acts_gb=b.acts, staged_gb=b.staged_opt,
                           total_hbm=b.total_hbm, usable=hw.usable_hbm_gib)

    def auto_ncta(self, opt: str, *, tp: int, pp: int, dp: int, zero3: bool,
                  B: int, n_sms: int) -> int:
        """The largest nCTA on the standard ladder whose staged scratch fits the usable
        HBM (the live memory-fit knob). Mirrors resource_planner's R2 CTA-tiling walk."""
        usable = (self._hw or _rp.HardwareConfig()).usable_hbm_gib
        for cand in (n_sms, 64, 32, 16, 8, 4, 2, 1):
            if cand > n_sms:
                continue
            b = self.per_rank_budget(opt, tp=tp, pp=pp, dp=dp, zero3=zero3,
                                     ncta=cand, B=B)
            if b.total_hbm <= usable:
                return cand
        return 1


@dataclasses.dataclass(frozen=True)
class _RankBudget:
    """flagship_budget.RankBudget-shaped view: the four per-rank regions (GiB) the
    planner's savings model multiplies, plus a fit predicate."""
    params_gb: float
    state_gb: float
    acts_gb: float
    staged_gb: float
    total_hbm: float
    usable: float

    @property
    def fits(self) -> bool:
        return self.total_hbm <= self.usable


fb = _FlagshipBudgetAdapter()


# ── Hardware descriptor (the planner's only knowledge of the box). ──
@dataclasses.dataclass(frozen=True)
class Hardware:
    n_gpus: int
    hbm_gib_per_gpu: float
    host_ram_gib: float
    # effective host<->device bandwidth (GiB/s) on the bus that carries offload/stream.
    h2d_gib_s: float = 24.0          # PCIe Gen4 x16 ~24-26 GiB/s (NVLink-C2C/Grace: ~450)
    # usable fraction after CUDA ctx + handles + comm buffers (mirrors the budget model).
    usable_frac: Optional[float] = None   # if None, derive from the safety margin

    def usable_hbm_gib(self) -> float:
        if self.usable_frac is not None:
            return self.hbm_gib_per_gpu * self.usable_frac
        # mirror the live budget: capacity - 4 GiB safety (ctx/handles/NCCL).
        return self.hbm_gib_per_gpu - fb.H100_SAFETY_GIB


@dataclasses.dataclass(frozen=True)
class MemPlan:
    offload_optimizer: bool
    recompute_acts: bool
    stream_layers: bool
    stream_depth: int                # resident layers when streaming (>=2 for ring)
    ncta: int
    resident_gib: float              # the per-rank device footprint AFTER strategies
    host_gib: float                  # pinned host RAM the plan needs
    fits: bool
    reason: str

    def gate_macros(self) -> dict:
        """The -D macros the kernel/launcher build consumes (the if-constexpr gate set).
        ALL-OFF => the byte-identical in-HBM path."""
        return {
            "SG_MEM_OFFLOAD_OPT": 1 if self.offload_optimizer else 0,
            "SG_MEM_RECOMPUTE_ACTS": 1 if self.recompute_acts else 0,
            "SG_MEM_STREAM_LAYERS": 1 if self.stream_layers else 0,
            "SG_MEM_STREAM_DEPTH": self.stream_depth,
        }


# Fraction of full acts that the layer-boundary checkpoint keeps. The boundary set is
# the per-layer LAYER INPUT (X_in[li], one [T,d] per layer) — the recompute anchor. The
# interior caches (X_ctx/X_x1/X_gact + the 4 dY adjoints) are recomputed. From
# DecActs (model_stage_decoder_tc.cuh:425-433): full per-layer fwd+bwd cache bf16 elems
# = Td(X_in)+Td(X_ctx)+Td(X_x1)+Tff(X_gact) + T3d(dY_qkv)+Td(dY_a)+Tff(dY_ff0)+Td(dY_ff2).
# Boundary keeps only X_in (Td). dff=4d => full = (1+1+1+4 + 3+1+4+1)*Td = 16*Td; kept=1*Td.
_ACT_BOUNDARY_FRAC = 1.0 / 16.0      # exact for dff=4d; recomputed precisely per-shape below


def _full_acts_floats(B: int, layers: int) -> int:
    return fb.dec_tc_acts_floats(B, layers)


def _boundary_acts_floats(B: int, layers: int) -> int:
    """Acts kept under recompute: per-layer X_in (Td) + the non-layer tail (X_hn/dY_logits/dh0).
    Mirrors dec_tc_acts_floats's tail term (B*d + B*V + Td) which is NOT recomputable."""
    d, V, seq = fb.FLAGSHIP_D, fb.FLAGSHIP_VOCAB, fb.FLAGSHIP_SEQ
    if fb._model is not None:
        d, V, seq = fb._model.d, fb._model.vocab, fb._model.seq
    T = B * seq
    Td = T * d
    bf = layers * Td                 # one X_in per layer (the checkpoint anchor)
    bf += B * d + B * V + Td         # tail (final-norm/logits/dh0) — must stay
    return (bf + 1) // 2


def plan_memory_strategy(*, total_params: int, layers: int, opt: str,
                         tp: int, pp: int, dp: int, zero3: bool, B: int,
                         hw: Hardware) -> MemPlan:
    """Pick the MINIMAL strategy set that fits the per-rank HBM budget, by FIT not by
    GPU count. Order of escalation (cheapest first):
        in-HBM -> cap nCTA -> recompute acts -> offload optimizer -> stream layers.
    Each step is added ONLY if the running budget still does not fit."""
    # Tell the budget adapter the run's shape/box so its scratch math matches this plan.
    # (Pure assignment of CPU-only descriptors; no GPU, no torch.)
    _bind_budget_model(total_params=total_params, layers=layers, opt=opt, B=B, hw=hw)

    usable = hw.usable_hbm_gib()
    n_sms = 132

    # Start from the in-HBM budget model at full occupancy, then escalate.
    def budget(ncta, recompute, offload, stream_depth):
        b = fb.per_rank_budget(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, ncta=ncta, B=B)
        params_gb, state_gb, acts_gb, staged_gb = b.params_gb, b.state_gb, b.acts_gb, b.staged_gb
        host_gib = 0.0
        layers_pr = max(layers // pp, 1)
        if recompute:
            full = _full_acts_floats(B, layers_pr)
            keep = _boundary_acts_floats(B, layers_pr)
            acts_gb = acts_gb * (keep / max(full, 1))
        if offload:
            host_gib += state_gb        # state moves to pinned host
            # resident device state ~ one stage tile; model it as 1/layers of the state
            # (the launcher stages per-tensor-group), bounded below by a small floor.
            state_gb = max(state_gb / max(layers_pr, 1), 0.05)
        if stream_depth and stream_depth < layers_pr:
            host_gib += params_gb        # weights move to pinned host
            params_gb = params_gb * (stream_depth / layers_pr)
        total = params_gb + state_gb + acts_gb + staged_gb + 0.10
        return total, host_gib

    # 1) in-HBM at the largest nCTA that fits the staged scratch (the existing lever).
    for ncta in (n_sms, 64, 32, 16, 8, 4, 2, 1):
        t, h = budget(ncta, False, False, 0)
        if t <= usable:
            return MemPlan(False, False, False, 0, ncta, t, h, True,
                           f"in-HBM @nCTA={ncta}")
    base_ncta = fb.auto_ncta(opt, tp=tp, pp=pp, dp=dp, zero3=zero3, B=B, n_sms=n_sms)

    # 2) + recompute acts.
    t, h = budget(base_ncta, True, False, 0)
    if t <= usable:
        return MemPlan(False, True, False, 0, base_ncta, t, h, True,
                       f"recompute-acts @nCTA={base_ncta}")
    # 3) + offload optimizer (needs host RAM for the state).
    t, h = budget(base_ncta, True, True, 0)
    if t <= usable and h <= hw.host_ram_gib:
        return MemPlan(True, True, False, 0, base_ncta, t, h, True,
                       f"recompute+offload @nCTA={base_ncta}")
    # 4) + stream layers (resident ring of 2; needs host RAM for params too).
    depth = 2
    t, h = budget(base_ncta, True, True, depth)
    fits = (t <= usable) and (h <= hw.host_ram_gib)
    return MemPlan(True, True, True, depth, base_ncta, t, h, fits,
                   ("recompute+offload+stream(depth=2) @nCTA="
                    f"{base_ncta}" + ("" if fits else " — STILL OOM (raise TP/PP or host RAM)")))


def _bind_budget_model(*, total_params: int, layers: int, opt: str, B: int,
                       hw: Hardware) -> None:
    """Point the ``fb`` adapter at the run's true model shape + hardware so its scratch
    arithmetic matches the plan. Derives the model dims from total_params/layers when they
    differ from the flagship pin (keeps the planner correct for arbitrary sizes, not just
    1.5B). Pure CPU descriptor assignment — no GPU."""
    d, vocab, seq = fb.FLAGSHIP_D, fb.FLAGSHIP_VOCAB, fb.FLAGSHIP_SEQ
    # If this is the flagship pin (matching total + layers), keep its exact dims so the
    # committed layout table arithmetic reproduces to the byte. Otherwise solve d from the
    # 12*L*d^2 dominant term (dff=4d) so the budget tracks the requested parameter count.
    if not (layers == fb.FLAGSHIP_LAYERS
            and total_params == fb.FLAGSHIP_TOTAL_PARAMS):
        # total ≈ 12*L*d^2 (the per-layer attn+ff weight bulk dominates embeddings/norms).
        import math
        d_est = int(round(math.sqrt(max(total_params, 1) / max(12 * layers, 1))))
        d = max((d_est // 8) * 8, 8)   # keep d a multiple of 8 (TP/head divisibility)
    fb._model = _rp.ModelConfig(d=d, layers=layers, seq=seq, vocab=vocab,
                                batch=B, optimizer=opt)
    # Hardware: translate the planner Hardware into the resource_planner envelope so the
    # adapter's usable_hbm_gib / host_ram_gib match the planner's own usable_hbm_gib().
    safety = (hw.hbm_gib_per_gpu - hw.usable_hbm_gib())
    fb._hw = _rp.HardwareConfig(
        num_gpus=max(hw.n_gpus, 1),
        hbm_bytes_per_gpu=int(hw.hbm_gib_per_gpu * (1024 ** 3)),
        host_ram_bytes=int(hw.host_ram_gib * (1024 ** 3)),
        safety_gib=max(safety, 0.0),
    )


__all__ = ["Hardware", "MemPlan", "plan_memory_strategy"]
