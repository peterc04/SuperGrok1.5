"""grokking_optimizers.megakernel — Stage 6 feasibility solver + generator.

Composes the per-arch kernel primitives into one fused **L3 persistent
megakernel per (model × optimizer × arch)** — 3 models × 11 optimizers × 3
archs = 99 pipelines (§1.5 L3, §1.6 all-99, §1.8 all-three-models).

This module is the **automatic feasibility solver** (§1.9/§1.10): it reads each
component's register / shared-mem (LDS on AMD) budget from the ARCH_TABLE and
picks the highest fusion tier that fits per (model, optimizer, arch); it falls
back DOWN tiers when L3 doesn't fit, and **errors** (§1.11 — no silent slow
path) when even the lowest tier doesn't fit. The chosen plan is what the C++
generator (`csrc/fused/<arch>/`) emits, and what `fused_step` in dispatch.cpp
prefers when a fused TU is present (§1.12).

Hardware-deferred: the *measured* register counts come from `ptxas
-v`/`rocm-llvm` on real silicon; here the solver uses calibrated per-component
estimates (recorded in HARDWARE_VALIDATION.md), so the tier selection is
🟡 until the on-device register report confirms it.

🟡 WORKSTREAM-1 CAVEAT (register-pressure reduction): the component costs below
were LOWERED to reflect behavior-preserving SMEM-staging / rematerialization /
live-range-shortening in the kernels (model_stages.cuh, supergrok2_sm90.cuh)
and the Hopper setmaxnreg producer/consumer warp-group split
(fused_megakernel.cuh). The resulting L1→L3/L2 tier promotions are ESTIMATES.
ptxas register allocation does NOT match this additive model — it may keep a
staged value live anyway, or free a slot we assumed live. The compile.py
per-cell `maxrregcount` sweep (knob #1) plus on-silicon `ptxas -v` are the
real arbiters; treat every tier change here as 🟡 until silicon confirms it.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict, List, Optional, Tuple

from grokking_optimizers.dispatch import canonicalize_model

# The ARCH_TABLE (single source of truth for per-arch smem/reg/warp budgets)
# lives in compile.py. Import lazily to avoid a heavy import at module load.

MODELS: Tuple[str, ...] = ("transformer_decoder", "vit", "mamba3")
OPTIMIZERS: Tuple[str, ...] = (
    "adamw", "lion", "grokfast", "grokadamw", "looksam", "muon",
    "neuralgrok", "prodigy", "supergrok11", "supergrok15", "supergrok2",
)
# The three archs that have hand-written kernel bodies (the megakernel targets).
MEGAKERNEL_ARCHS: Tuple[str, ...] = ("sm_90", "gfx942", "tpu_v6e")


class FusionTier(enum.IntEnum):
    """Fusion tiers, highest (most fused) first. The solver picks the highest
    tier whose register + shared-memory footprint fits the arch budget."""
    L3_FWD_BWD_OPT = 3   # one persistent kernel: forward + backward + optimizer
    L2_BWD_OPT = 2       # backward + optimizer fused; forward separate
    L1_OPT_ONLY = 1      # optimizer step fused across params; fwd/bwd separate
    L0_UNFUSED = 0       # no fusion (the existing per-op path) — solver ERRORS
                         # before returning this for a megakernel request


@dataclasses.dataclass(frozen=True)
class ComponentCost:
    """Calibrated per-component resource estimate (one fusion building block).

    regs   — registers/thread the component's hot loop needs live.
    smem   — bytes of shared memory (LDS on AMD) it stages.
    These are the solver inputs; on-silicon `ptxas -v` refines them (🟡).
    """
    name: str
    regs: int
    smem: int


# ── Calibrated component costs (per model / optimizer) ────────────────────
# Forward + backward dominate registers/smem; the optimizer tail is light.
# Estimates are deliberately conservative (round up) so the solver never
# over-promises a tier that won't fit. Refined from on-device reports (🟡).

_MODEL_FWD: Dict[str, ComponentCost] = {
    # MFMA/WGMMA attention + FFN: large smem tiles, high reg pressure.
    "transformer_decoder": ComponentCost("decoder_fwd", regs=168, smem=49152),
    "vit":                 ComponentCost("vit_fwd",     regs=168, smem=49152),
    # SSM selective scan: deep recurrence, moderate smem, high regs.
    "mamba3":              ComponentCost("mamba3_fwd",  regs=176, smem=40960),
}
_MODEL_BWD: Dict[str, ComponentCost] = {
    # Backward roughly tracks forward + the saved-activation adjoints.
    # WORKSTREAM 1 (#2/#3/#4): model_backward_stage now STAGES the upstream
    # adjoint slab into shared memory (up_stage[256], +1KB smem) and
    # rematerializes x at point-of-use, and model_activation_grad was
    # live-range-shortened so u/t/du no longer co-reside. That pulls the
    # per-thread live-register working set down from the saved-adjoint estimate.
    # 🟡 estimate — ptxas refines on silicon.
    "transformer_decoder": ComponentCost("decoder_bwd", regs=168, smem=66560),
    "vit":                 ComponentCost("vit_bwd",     regs=168, smem=66560),
    "mamba3":              ComponentCost("mamba3_bwd",  regs=176, smem=50176),
}
# Optimizer-step register/ smem cost (the fused tail). Elementwise optimizers
# are cheap; the meta-net optimizers (sg11/15/2) carry a small MLP / attention.
_OPT_COST: Dict[str, ComponentCost] = {
    "adamw":       ComponentCost("adamw_opt",       regs=40, smem=0),
    "lion":        ComponentCost("lion_opt",        regs=32, smem=0),
    "grokfast":    ComponentCost("grokfast_opt",    regs=48, smem=0),
    "grokadamw":   ComponentCost("grokadamw_opt",   regs=52, smem=0),
    "looksam":     ComponentCost("looksam_opt",     regs=56, smem=2048),
    "muon":        ComponentCost("muon_opt",        regs=72, smem=4096),
    "neuralgrok":  ComponentCost("neuralgrok_opt",  regs=64, smem=1024),
    "prodigy":     ComponentCost("prodigy_opt",     regs=56, smem=2048),
    "supergrok11": ComponentCost("sg11_opt",        regs=88, smem=4096),
    "supergrok15": ComponentCost("sg15_opt",        regs=88, smem=4096),
    # SG2's CSA/HCA + PEER meta-net is the heaviest optimizer tail by far.
    # WORKSTREAM 1 (#2): the CSA/HCA online-softmax accumulator acc[64] and the
    # indexer top-k scratch best_score[64]+best_idx[64] (the dominant per-thread
    # register arrays, which spilled to local on real ptxas) are now staged into
    # the attention kernels' dynamic shared buffer + a top-k smem scratch. That
    # removes ~128 register slots of array state from the hot path; the smem
    # rises (acc staging doubles the query buffer, top-k adds block*64*8 B), but
    # smem is the near-empty resource. 🟡 estimate.
    "supergrok2":  ComponentCost("sg2_opt",         regs=96, smem=49152),
}


@dataclasses.dataclass(frozen=True)
class ArchBudget:
    name: str
    max_regs_per_thread: int
    max_smem_per_block: int
    warp_size: int
    # In a persistent megakernel we pin occupancy; the realistic per-thread
    # register ceiling for a 1-CTA-per-SM persistent kernel is the full file.
    is_pallas: bool = False


def _arch_budget(arch: str) -> ArchBudget:
    """Pull the register/smem budget for `arch` from compile.py's ARCH_TABLE."""
    from grokking_optimizers.compile import ARCH_TABLE  # lazy
    if arch not in ARCH_TABLE:
        raise KeyError(f"unknown arch '{arch}' (not in ARCH_TABLE)")
    e = ARCH_TABLE[arch]
    is_pallas = e.warp_size is None
    return ArchBudget(
        name=arch,
        # Pallas/TPU has no register/smem model exposed here; treat as
        # always-fits at the program level (XLA does its own allocation).
        max_regs_per_thread=e.max_regs_per_thread or 1 << 30,
        max_smem_per_block=e.max_smem_per_block or 1 << 30,
        warp_size=e.warp_size or 0,
        is_pallas=is_pallas,
    )


@dataclasses.dataclass(frozen=True)
class FusionPlan:
    model: str
    optimizer: str
    arch: str
    tier: FusionTier
    regs: int
    smem: int
    budget_regs: int
    budget_smem: int
    note: str = ""

    @property
    def fits(self) -> bool:
        return self.tier > FusionTier.L0_UNFUSED


# Hopper setmaxnreg producer-warp-group floor (kProducerRegsF in
# fused_megakernel.cuh). The producer WG that runs the light staging + the
# optimizer tail is bounded at this allocation; the heavy model stages run on
# the consumer WG. See _tier_cost for how the §3.4 split changes the L3 peak.
_PRODUCER_WG_REGS = 32


def _tier_cost(model: str, optimizer: str,
               tier: FusionTier) -> Tuple[int, int]:
    """Peak (regs, smem) for a given fusion tier.

    WORKSTREAM 1 (#6): the L3 fused kernel now runs as TWO Hopper warp-groups
    (setmaxnreg split, fused_megakernel.cuh). The heavy model forward/backward
    stages run on the CONSUMER warp-group; the light producer staging + the
    optimizer tail run on the PRODUCER warp-group. The SM register *ceiling* a
    single warp-group must satisfy is therefore the MAX of the two warp-groups'
    footprints, NOT the old sum (max(fwd,bwd) + opt). We bound the consumer WG
    by max(fwd,bwd) and the producer WG by the optimizer tail (which subsumes
    the producer floor), and take the max. smem is the MAX tile resident at
    once (the staged adjoint / attention buffers are summed into the component
    smem figures). 🟡 these are MODEL estimates — ptxas register allocation
    differs; the per-cell maxrregcount sweep (compile.py knob #1) + on-silicon
    `ptxas -v` are the source of truth.
    """
    fwd = _MODEL_FWD[model]
    bwd = _MODEL_BWD[model]
    opt = _OPT_COST[optimizer]
    if tier == FusionTier.L3_FWD_BWD_OPT:
        consumer_wg = max(fwd.regs, bwd.regs)          # heavy model stages
        producer_wg = max(opt.regs, _PRODUCER_WG_REGS)  # tail + staging
        regs = max(consumer_wg, producer_wg)            # §3.4 split: max, not sum
        smem = max(fwd.smem, bwd.smem, opt.smem)
    elif tier == FusionTier.L2_BWD_OPT:
        regs = bwd.regs + opt.regs
        smem = max(bwd.smem, opt.smem)
    elif tier == FusionTier.L1_OPT_ONLY:
        regs = opt.regs
        smem = opt.smem
    else:
        regs, smem = 0, 0
    return regs, smem


def solve(model: str, optimizer: str, arch: str) -> FusionPlan:
    """§1.9/§1.10 feasibility solver: pick the highest fusion tier that fits.

    Falls DOWN tiers when L3 doesn't fit; raises (§1.11 — no silent slow path)
    when even L1 (optimizer-only fusion) doesn't fit the arch budget.

    Accepts both short names ("mamba", "decoder", "vit") and canonical names
    ("mamba3", "transformer_decoder", "vit") via canonicalize_model().
    """
    model = canonicalize_model(model)
    if model not in _MODEL_FWD:
        raise KeyError(f"unknown model '{model}'")
    if optimizer not in _OPT_COST:
        raise KeyError(f"unknown optimizer '{optimizer}'")

    b = _arch_budget(arch)

    # Pallas/TPU: XLA owns allocation; the megakernel is a single fused program
    # and always "fits" at this granularity. Report L3.
    if b.is_pallas:
        return FusionPlan(model, optimizer, arch, FusionTier.L3_FWD_BWD_OPT,
                          regs=0, smem=0, budget_regs=0, budget_smem=0,
                          note="pallas/XLA: single fused program, allocator-managed")

    for tier in (FusionTier.L3_FWD_BWD_OPT,
                 FusionTier.L2_BWD_OPT,
                 FusionTier.L1_OPT_ONLY):
        regs, smem = _tier_cost(model, optimizer, tier)
        if regs <= b.max_regs_per_thread and smem <= b.max_smem_per_block:
            note = ""
            if tier != FusionTier.L3_FWD_BWD_OPT:
                # Record exactly which higher tier(s) busted, and on which
                # resource, so the coverage table is self-explaining.
                reasons = []
                for hi in (FusionTier.L3_FWD_BWD_OPT, FusionTier.L2_BWD_OPT):
                    if hi <= tier:
                        continue
                    hr, hs = _tier_cost(model, optimizer, hi)
                    over = []
                    if hr > b.max_regs_per_thread:
                        over.append(f"regs {hr}>{b.max_regs_per_thread}")
                    if hs > b.max_smem_per_block:
                        over.append(f"smem {hs}>{b.max_smem_per_block}")
                    if over:
                        reasons.append(f"{hi.name}({', '.join(over)})")
                note = f"fell back to {tier.name}; " + "; ".join(reasons)
            return FusionPlan(model, optimizer, arch, tier,
                              regs=regs, smem=smem,
                              budget_regs=b.max_regs_per_thread,
                              budget_smem=b.max_smem_per_block, note=note)

    # §1.11: even L1 doesn't fit → hard error, never a silent slow path.
    regs, smem = _tier_cost(model, optimizer, FusionTier.L1_OPT_ONLY)
    raise RuntimeError(
        f"megakernel infeasible for ({model}, {optimizer}, {arch}): even "
        f"L1 optimizer-only fusion needs regs={regs}/smem={smem} but the arch "
        f"budget is regs={b.max_regs_per_thread}/smem={b.max_smem_per_block}. "
        f"No silent slow path (§1.11) — use the unfused per-op dispatch.")


def solve_all() -> List[FusionPlan]:
    """Solve the full 99-pipeline matrix. Infeasible cells are recorded with an
    L0 plan + the reason (so the table is complete) rather than raising, so the
    caller can see coverage at a glance."""
    plans: List[FusionPlan] = []
    for arch in MEGAKERNEL_ARCHS:
        for model in MODELS:
            for opt in OPTIMIZERS:
                try:
                    plans.append(solve(model, opt, arch))
                except RuntimeError as e:
                    b = _arch_budget(arch)
                    regs, smem = _tier_cost(model, opt, FusionTier.L1_OPT_ONLY)
                    plans.append(FusionPlan(
                        model, opt, arch, FusionTier.L0_UNFUSED,
                        regs=regs, smem=smem,
                        budget_regs=b.max_regs_per_thread,
                        budget_smem=b.max_smem_per_block,
                        note=f"INFEASIBLE: {e}"))
    return plans


def coverage_summary() -> Dict[str, int]:
    """Tier coverage across all 99 pipelines (for the consolidation report)."""
    plans = solve_all()
    out: Dict[str, int] = {t.name: 0 for t in FusionTier}
    for p in plans:
        out[p.tier.name] += 1
    return out


if __name__ == "__main__":
    import json
    cov = coverage_summary()
    print("Megakernel feasibility-solver coverage (99 pipelines):")
    print(json.dumps(cov, indent=2))
    # Show any infeasible cells explicitly.
    infeasible = [p for p in solve_all() if not p.fits]
    if infeasible:
        print(f"\n{len(infeasible)} INFEASIBLE cell(s):")
        for p in infeasible:
            print(f"  ({p.model}, {p.optimizer}, {p.arch}): {p.note}")
    else:
        print("\nAll 99 pipelines feasible at L1 or higher.")
