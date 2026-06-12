"""grokking_optimizers.megakernel_codegen — Stage 6 L3 megakernel generator.

Given the feasibility solver (``megakernel.solve_all`` / ``megakernel.solve``),
this module EMITS the per-(model × optimizer × arch) megakernel source by
instantiating the ONE templated L3 megakernel from
``csrc/fused/<arch>/megakernel_demo.{cu,hip.hpp}`` at the solver-chosen tier.

You do NOT hand-write 99 kernels: the demo TU is the structural template, and
the generator re-instantiates ``l3_megakernel<Model, Optimizer>`` (sm_90 /
gfx942) or emits the Pallas program (tpu_v6e) per cell. It can emit a single
cell's source string (``--emit <model> <optimizer> <arch>``) or report the full
99-cell manifest with tiers (``--emit-all``).

The set of cells that already have a REAL, wired fused TU (the three the demo
instantiates + dispatch.cpp routes to) is :data:`WIRED_CELLS`; every other cell
is generator-emittable but not yet compiled into the extension (the manifest
flags which is which, mirroring dispatch.cpp's honesty contract, §1.12).

BUILD-TIME GENERATOR — NOT a runtime module. It is invoked from the build
(``setup.py`` materializes the per-cell sources it emits) and as a CLI::

    python -m grokking_optimizers.megakernel_codegen --emit-all
    python -m grokking_optimizers.megakernel_codegen --emit mamba3 supergrok2 sm_90
    python -m grokking_optimizers.megakernel_codegen --write-all

It has NO runtime call sites in the optimizer hot path by design; see
:func:`main` / the ``__main__`` entry below.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

from grokking_optimizers import megakernel as mk
from grokking_optimizers.megakernel import FusionPlan, FusionTier

# ── Component enum mapping (Phase 3 Stage 5 — REAL compositions) ─────────────
# The sm_90 fused cells compose the REAL optimizer device-function component
# (csrc/algorithms/<opt>.h via csrc/fused/sm_90/opt_components.cuh) with the
# REAL model stage component (csrc/fused/sm_90/model_stages.cuh). Each optimizer
# maps to its OWN OptId — every one of the 11 has real, distinct update math.
# There is NO fallback: the Phase 2 "7 optimizers → AdamW tail" breadth-faking
# is eliminated (see opt_components.cuh header).
_MODEL_ENUM: Dict[str, str] = {
    "transformer_decoder": "TransformerDecoder",
    "vit": "ViT",
    "mamba3": "Mamba3",
}
# Every optimizer → its own real OptId in opt_components.cuh::OptId. No fallback.
_OPT_ENUM: Dict[str, str] = {
    "adamw": "AdamW",
    "lion": "Lion",
    "grokfast": "Grokfast",
    "grokadamw": "GrokAdamW",
    "looksam": "LookSAM",
    "prodigy": "Prodigy",
    "neuralgrok": "NeuralGrok",
    "muon": "Muon",
    "supergrok11": "SuperGrok11",
    "supergrok15": "SuperGrok15",
    "supergrok2": "SuperGrok2",
}

# Cells whose megakernel source is materialized to disk and whose dispatch
# route exists in dispatch.cpp::wired_fused_cell. Phase 2 expanded this from
# the original 3 demo cells to all 99 (3 models × 11 optimizers × 3 archs).
# The manifest marks wired=True for all cells, since every cell now has a
# generated source TU and a dispatch route. GPU-arch cells are compile-gated
# (sm_90 → nvcc, gfx942 → hipcc); TPU cells are Pallas stubs.
WIRED_CELLS: Tuple[Tuple[str, str, str], ...] = tuple(
    (m, o, a)
    for m in ("transformer_decoder", "vit", "mamba3")
    for o in ("adamw", "lion", "grokfast", "grokadamw", "looksam", "muon",
              "neuralgrok", "prodigy", "supergrok11", "supergrok15",
              "supergrok2")
    for a in ("sm_90", "gfx942", "tpu_v6e")
)


def _opt_enum(optimizer: str) -> Tuple[str, bool]:
    """Return (OptId spelling, is_exact). is_exact is always True now: every
    optimizer has its own real OptId / real update math (no fallback)."""
    return _OPT_ENUM[optimizer], True


def _cell_symbol(model: str, optimizer: str) -> str:
    """The non-templated host-launcher symbol name for a cell."""
    return f"mega_{model}_{optimizer}"


# Per-optimizer "extra" state buffer (a single n-element slice beyond m, v) that
# the fused tail reads. The host passes `extra` = state + 2n; each cell binds it
# to its optimizer's FusedOptState field. None → optimizer needs no third
# per-element buffer (adamw/lion). neuralgrok is special: its psi-net is a small
# per-tensor weight set (not an n-slice), so it stays host-supplied (🟡) and is
# NOT bound from `extra`.
_OPT_EXTRA_FIELD: Dict[str, Optional[str]] = {
    "adamw": None,
    "lion": None,
    "grokfast": "ema",
    "grokadamw": "ema",
    "looksam": "sam_dir",
    "prodigy": "s_track",
    "neuralgrok": None,        # psi-net weights bound explicitly — see below
    "muon": "orth",
    "supergrok11": "mu",
    "supergrok15": "mu",
    "supergrok2": "smart_grad",
}

# NeuralGrok needs its psi-net weights bound from the packed `extra` buffer
# (layout W1|b1|W2|b2 in opt_components.{cuh,hip.hpp}). It does NOT use a single
# `extra` n-slice field like the other optimizers, so it gets a bespoke bind
# block instead of the generic `st.<field> = extra` / `(void)extra` line. The
# psi_b2 scalar is left host-side at 0 and read ON-DEVICE (st.psi_W2[kPsiHidden],
# where the device pointer is dereferenceable). This keeps the codegen the
# single source for the cells (the bind used to be hand-patched into the files).
_PSI_BIND_CUDA = """
    // #5: bind the psi-net weights from the packed `extra` buffer (layout in
    // opt_components.cuh: W1|b1|W2|b2). Without this the psi pointers stay null
    // and neuralgrok_psi_forward null-derefs on this real path.
    if (extra) {
        st.psi_W1 = extra + kPsiW1Off;
        st.psi_b1 = extra + kPsiB1Off;
        st.psi_W2 = extra + kPsiW2Off;
        // st.psi_b2 left at 0.0f host-side: the real psi_b2 scalar lives at
        // extra[kPsiB2Off] and is read ON-DEVICE in apply_optimizer<NeuralGrok>
        // (st.psi_W2[kPsiHidden]), where the device pointer is dereferenceable.
    }"""

_PSI_BIND_HIP = """
    // #5: bind psi-net weights from the packed `extra` buffer (W1|b1|W2|b2,
    // layout in opt_components.hip.hpp) — without this the psi pointers are null
    // and sg_psi_forward null-derefs. psi_b2 is read on-device in apply_optimizer<NeuralGrok> (st.psi_W2[kPsiHidden]).
    if (extra) {
        st.psi_W1 = extra + kPsiW1Off;
        st.psi_b1 = extra + kPsiB1Off;
        st.psi_W2 = extra + kPsiW2Off;
    }"""


# ── Per-arch source emission ─────────────────────────────────────────────

def _fuse_tier_tag(tier: FusionTier, *, lower: bool = False) -> str:
    """Map a solver ``FusionTier`` to the C++ ``FuseTier`` instantiation tag.

    The emitters only support L3 (fwd+bwd+opt) and L1 (opt-only). The old
    ``"L3" if tier == L3 else "L1"`` form SILENTLY coerced L2_BWD_OPT (and
    L0_UNFUSED) to L1 — emitting a kernel that fuses only the optimizer tail
    instead of backward+optimizer, a wrong-kernel landmine if the cost model
    ever yields an L2 cell. Fail loudly instead so the codegen is corrected.
    """
    mapping = {FusionTier.L3_FWD_BWD_OPT: "l3", FusionTier.L1_OPT_ONLY: "l1"}
    tag = mapping.get(tier)
    if tag is None:
        raise ValueError(
            f"megakernel_codegen: FusionTier {tier.name} has no emitter "
            f"mapping (only L3/L1 are emittable). The solver returned a tier "
            f"the codegen cannot emit — add an emitter or fix the cost model.")
    return tag if lower else tag.upper()


def _emit_cuda(plan: FusionPlan) -> str:
    """Emit the sm_90 .cu source for a cell as a REAL component composition.

    The cell includes the real composition substrate (fused_megakernel.cuh,
    which pulls the real csrc/algorithms/<opt>.h optimizer component and the
    real model_stages.cuh model component) and defines its launcher calling
    launch_fused_megakernel<ModelId, OptId, FuseTier> — the real, distinct
    per-optimizer update math. NO demo include, NO TEMPLATE_ONLY, NO fallback.
    """
    m_enum = _MODEL_ENUM[plan.model]
    o_enum, _ = _opt_enum(plan.optimizer)
    sym = _cell_symbol(plan.model, plan.optimizer)
    tier = _fuse_tier_tag(plan.tier)
    extra_field = _OPT_EXTRA_FIELD[plan.optimizer]
    if plan.optimizer == "neuralgrok":
        extra_line = _PSI_BIND_CUDA
    elif extra_field:
        extra_line = (f"\n    st.{extra_field} = extra;  // {plan.optimizer}'s "
                      f"third per-element state buffer")
    else:
        extra_line = ("\n    (void)extra;  // adamw/lion/neuralgrok need no "
                      "'extra' n-slice")
    return f"""// csrc/fused/sm_90/{sym}.cu  — GENERATED by megakernel_codegen.py
// Cell: ({plan.model}, {plan.optimizer}, {plan.arch})  tier={plan.tier.name}
//   regs={plan.regs}/{plan.budget_regs}  smem={plan.smem}/{plan.budget_smem}
//   {plan.note or 'fits L3'}
//
// REAL component composition (Phase 3 Stage 5): composes the real optimizer
// device-function component (csrc/algorithms/{plan.optimizer}.h via
// opt_components.cuh::apply_optimizer<OptId::{o_enum}>) with the real model
// stage component (model_stages.cuh::model_*_stage<ModelId::{m_enum}>) over the
// shared persistent-megakernel substrate. Solver fuse tier (perf placement): {tier}.
#include "csrc/fused/sm_90/fused_megakernel.cuh"

namespace sg {{ namespace fused {{ namespace sm90 {{

// Uniform host entry (the symbol fused_step dispatches to). Binds this
// optimizer's REAL state buffers: m/v (Adam moments) + the per-optimizer
// `extra` n-slice (ema/sam_dir/s_track/mu/orth/smart_grad), AND the FULL runtime
// scalar set via apply_scalars() — beta1/beta2/eps/wd/alpha/lamb/alpha_max/beta/
// gate/d_factor/neg_lr_scale/decay_factor + un-inverted bc1/bc2. This closes the
// C2 gap: every scalar apply_optimizer<{o_enum}> reads is now SET from the live
// optimizer (previously only lr was bound, freezing bc/gate/d_factor at 1.0).
//
// TIER SELECTOR (`opt_only`): this entry instantiates BOTH tiers and selects at
// runtime. opt_only=true → L1: the fused optimizer TAIL only; the kernel reads
// the REAL gradient from the global `grad` buffer (computed by the framework's
// own fwd/bwd) and applies the canonical update — this is the NUMERICALLY
// FAITHFUL path the grokking race uses (real grad in, real updated params out).
// opt_only=false → L3: the kernel ALSO runs the megakernel's element-local model
// fwd/bwd over the flat param blob (model_stages.cuh) — a SURROGATE model, NOT
// the real Transformer/ViT/Mamba graph, so its loss does not match eager. L3 is
// kept compiled (perf-placement coverage) but is not the race path; see
// BUILD_AND_VALIDATE.md. Both tiers share this identical scalar/pointer binding.
cudaError_t {sym}(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, float* extra,
        const int* sizes, const int* offsets,
        float lr, int step, const FusedScalars& scalars, bool opt_only,
        cudaStream_t stream) {{
    FusedOptState st;
    st.exp_avg = m;
    st.exp_avg_sq = v;{extra_line}
    apply_scalars(st, scalars);  // FULL scalar set (un-freezes bc1/bc2/gate/d/…)
    st.lr = lr;                  // lr also passed positionally for the L3 phase
    if (opt_only) {{
        return launch_fused_megakernel<ModelId::{m_enum}, OptId::{o_enum},
                                       FuseTier::L1>(
            ctx, params, input, acts, grad, sizes, offsets, lr, step, st, stream);
    }}
    return launch_fused_megakernel<ModelId::{m_enum}, OptId::{o_enum},
                                   FuseTier::L3>(
        ctx, params, input, acts, grad, sizes, offsets, lr, step, st, stream);
}}

}}}}}}  // namespace sg::fused::sm90
"""


def _emit_hip(plan: FusionPlan) -> str:
    """Emit the gfx942 .hip source for a cell as a REAL component composition.

    AMD twin of _emit_cuda: the cell includes fused_megakernel.hip.hpp (which
    pulls the real opt_components.hip.hpp — all 11 optimizers' real AMDGCN apply,
    NO fallback — and the real model_stages.hip.hpp) and force-instantiates its
    fused_megakernel<ModelId, OptId, FuseTier>. NO demo include, NO toy
    opt_update. The device code is AMDGCN-gate-verified; the host launch
    (hipLaunchKernelGGL) is 🟡 (MI300X-gated; no hipcc here).
    """
    m_enum = _MODEL_ENUM[plan.model]
    o_enum, _ = _opt_enum(plan.optimizer)
    sym = _cell_symbol(plan.model, plan.optimizer)
    tier = _fuse_tier_tag(plan.tier)
    extra_field = _OPT_EXTRA_FIELD[plan.optimizer]
    if plan.optimizer == "neuralgrok":
        extra_line = _PSI_BIND_HIP
    elif extra_field:
        extra_line = f"\n    st.{extra_field} = extra;"
    else:
        extra_line = "\n    (void)extra;"
    return f"""// csrc/fused/gfx942/{sym}.hip  — GENERATED by megakernel_codegen.py
// Cell: ({plan.model}, {plan.optimizer}, {plan.arch})  tier={plan.tier.name}
//   §1.13 ping-pong / 4-wave-interleave (NOT warp-specialized).
//
// REAL component composition (AMD): composes the real optimizer device-function
// component (opt_components.hip.hpp::apply_optimizer<OptId::{o_enum}>, byte-
// faithful to csrc/algorithms/{plan.optimizer}.h) with the real model stage
// component (model_stages.hip.hpp::model_*_stage<ModelId::{m_enum}>) over the
// gfx942 persistent substrate. Fuse tier {tier} (solver). AMDGCN-gate-verified;
// host hipLaunchKernelGGL is 🟡 (MI300X).
#include "csrc/fused/gfx942/fused_megakernel.hip.hpp"

#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)
namespace sg {{ namespace fused {{ namespace gfx942_mega {{

// Force BOTH tier instantiations so the device pass emits them and the free-
// standing AMDGCN gate type-checks the full expansion. The host entry selects
// L1 (opt-only, real-grad tail — the faithful path) or L3 (surrogate model fwd/
// bwd + opt) at runtime; both must exist as symbols.
template __global__ void
fused_megakernel<ModelId::{m_enum}, OptId::{o_enum}, FuseTier::L1>(
    PersistentContext, float*, const float*, float*, float*,
    const int*, const int*, float, int, FusedOptState);
template __global__ void
fused_megakernel<ModelId::{m_enum}, OptId::{o_enum}, FuseTier::L3>(
    PersistentContext, float*, const float*, float*, float*,
    const int*, const int*, float, int, FusedOptState);

}}}}}}  // namespace sg::fused::gfx942_mega
#endif

// ── HOST launcher (hipcc host pass only; 🟡 MI300X — not the bare amdgcn gate,
//    not the WITH_CUDA build). Faithful mirror of the verified sm_90 launcher:
//    binds m/v/extra + the FULL scalar set (apply_scalars, closing the C2 gap)
//    and selects the tier via opt_only. Uses launch_fused_megakernel (the
//    hang-free occupancy-capped helper) for each tier. __HIPCC__ is defined for
//    the hipcc host pass, so FusedOptState/FusedScalars/fused_megakernel are
//    visible here. ─────────────────────────────────────────────────────────────
#if defined(__HIPCC__) && !defined(__AMDGCN__)
#include <hip/hip_runtime.h>
namespace sg {{ namespace fused {{ namespace gfx942_mega {{
hipError_t {sym}(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, float* extra,
        const int* sizes, const int* offsets, float lr, int step,
        const FusedScalars& scalars, bool opt_only, hipStream_t stream) {{
    FusedOptState st;
    st.exp_avg = m; st.exp_avg_sq = v;{extra_line}
    apply_scalars(st, scalars);  // FULL scalar set (un-freezes bc1/bc2/gate/d/…)
    st.lr = lr;
    if (opt_only) {{
        return launch_fused_megakernel<ModelId::{m_enum}, OptId::{o_enum},
                                       FuseTier::L1>(
            ctx, params, input, acts, grad, sizes, offsets, lr, step, st, stream);
    }}
    return launch_fused_megakernel<ModelId::{m_enum}, OptId::{o_enum},
                                   FuseTier::L3>(
        ctx, params, input, acts, grad, sizes, offsets, lr, step, st, stream);
}}
}}}}}}  // namespace sg::fused::gfx942_mega
#endif
"""


def _emit_pallas(plan: FusionPlan) -> str:
    """Emit a tpu_v6e cell as a REAL fused composition (not a stub).

    The cell binds to the real fused program in
    csrc/backends/pallas/_pallas_fused.py::fused_step, which composes — inside
    ONE jax.jit (XLA fuses it) — the real TPU model fwd/bwd
    (_pallas_models.py) with the real per-optimizer TPU step
    (kernels/tpu/<opt> or backends/pallas/launch_<opt>). All 11 optimizers and
    3 models are wired (33/33 trace+lower verified, L1 and L3). The solver tier
    selects L3 (fwd+bwd+opt) vs L1 (opt-only).
    """
    sym = _cell_symbol(plan.model, plan.optimizer)
    tier = _fuse_tier_tag(plan.tier)
    return f'''# csrc/fused/tpu_v6e/{sym}.py
# GENERATED by megakernel_codegen.py — cell ({plan.model}, {plan.optimizer},
# {plan.arch}), tier={plan.tier.name}.
#
# REAL fused composition: binds (model={plan.model}, optimizer={plan.optimizer})
# to the real fused program csrc/backends/pallas/_pallas_fused.py::fused_step,
# which composes the real TPU model fwd/bwd with the real per-optimizer TPU
# step inside one jax.jit (XLA fuses fwd->bwd->opt for L3, opt-only for L1).
# NO stub — verified via trace_check (jax.eval_shape + jit.lower).
from functools import partial

from csrc.backends.pallas._pallas_fused import fused_step, trace_check

MODEL = "{plan.model}"
OPTIMIZER = "{plan.optimizer}"
TIER = "{tier}"

# The fused step for this cell (model+optimizer bound; tier from the solver).
step = partial(fused_step, MODEL, OPTIMIZER, tier=TIER)


def verify():
    """Trace+lower this cell's fused program (no hardware needed)."""
    return trace_check(MODEL, OPTIMIZER, TIER)
'''


def emit_cell(model: str, optimizer: str, arch: str) -> str:
    """Emit the megakernel source string for ONE (model, optimizer, arch) cell,
    at the solver-chosen tier. Raises if the solver finds the cell infeasible
    (§1.11 — no silent slow path)."""
    plan = mk.solve(model, optimizer, arch)  # raises on infeasible
    if arch == "sm_90":
        return _emit_cuda(plan)
    if arch == "gfx942":
        return _emit_hip(plan)
    if arch == "tpu_v6e":
        return _emit_pallas(plan)
    raise KeyError(f"unknown arch '{arch}' (not a megakernel target)")


# ── Manifest ─────────────────────────────────────────────────────────────

def manifest() -> List[Dict[str, object]]:
    """The full 99-cell manifest: every (model, optimizer, arch) with its
    solver tier, the template enum pair it instantiates, whether the optimizer
    tail is an exact demo case, and whether the cell is WIRED into fused_step."""
    rows: List[Dict[str, object]] = []
    for plan in mk.solve_all():
        o_enum, exact = _opt_enum(plan.optimizer)
        rows.append({
            "model": plan.model,
            "optimizer": plan.optimizer,
            "arch": plan.arch,
            "tier": plan.tier.name,
            "feasible": plan.fits,
            "model_enum": _MODEL_ENUM.get(plan.model, "?"),
            "opt_enum": o_enum,
            "opt_exact": exact,
            "wired": (plan.model, plan.optimizer, plan.arch) in WIRED_CELLS,
            "symbol": _cell_symbol(plan.model, plan.optimizer),
            "regs": plan.regs,
            "smem": plan.smem,
            "note": plan.note,
        })
    return rows


def _format_manifest(rows: List[Dict[str, object]]) -> str:
    out: List[str] = []
    out.append(f"Stage 6 megakernel manifest — {len(rows)} cells "
               f"(3 models × 11 optimizers × 3 archs)")
    # Tier coverage.
    tiers: Dict[str, int] = {}
    wired = 0
    for r in rows:
        tiers[str(r["tier"])] = tiers.get(str(r["tier"]), 0) + 1
        if r["wired"]:
            wired += 1
    out.append("  tier coverage: " +
               ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())))
    out.append(f"  wired into fused_step: {wired} cell(s) "
               f"(real fused TU + dispatch route)")
    out.append("")
    out.append(f"  {'model':<20} {'optimizer':<13} {'arch':<9} "
               f"{'tier':<16} {'wired':<6} {'exact_tail':<10} note")
    out.append("  " + "-" * 96)
    for r in rows:
        out.append(
            f"  {r['model']:<20} {r['optimizer']:<13} {r['arch']:<9} "
            f"{r['tier']:<16} {'yes' if r['wired'] else '':<6} "
            f"{'yes' if r['opt_exact'] else 'fallback':<10} "
            f"{r['note'] or ''}")
    return "\n".join(out)


# ── Materialize all cells to disk ─────────────────────────────────────

_ARCH_EXT = {"sm_90": ".cu", "gfx942": ".hip", "tpu_v6e": ".py"}
_ARCH_DIR = {"sm_90": "sm_90", "gfx942": "gfx942", "tpu_v6e": "tpu_v6e"}


def write_all(root: str = "csrc/fused") -> List[str]:
    """Materialize all 99 cell source files under *root*/<arch>/."""
    import os
    paths: List[str] = []
    for plan in mk.solve_all():
        src = emit_cell(plan.model, plan.optimizer, plan.arch)
        sym = _cell_symbol(plan.model, plan.optimizer)
        ext = _ARCH_EXT[plan.arch]
        arch_dir = os.path.join(root, _ARCH_DIR[plan.arch])
        os.makedirs(arch_dir, exist_ok=True)
        path = os.path.join(arch_dir, f"{sym}{ext}")
        with open(path, "w") as f:
            f.write(src)
        paths.append(path)
    return paths


_CELL_LAUNCHER_SIG = (
    "PersistentContext, float*, const float*, float*, float*, float*, float*, "
    "float*, const int*, const int*, float, int, const FusedScalars&, bool, "
    "cudaStream_t")


def dispatch_table_sm90() -> str:
    """Emit the generated sm_90 fused-cell dispatch table (.inc).

    Provides extern declarations for all 33 real sm_90 cell launchers and a
    dispatch_sm90_cell() that routes (model, optimizer) → the real composition
    symbol (csrc/fused/sm_90/mega_<model>_<opt>.cu). This replaces the 3
    hard-coded demo routes; every sm_90 cell is now a real composition route.
    """
    lines: List[str] = []
    lines.append("// AUTO-GENERATED by megakernel_codegen.py --dispatch-table-sm90")
    lines.append("// Do not edit by hand. Routes all 33 sm_90 cells to their real")
    lines.append("// component-composition launchers (Phase 3 Stage 5).")
    lines.append("// NOTE: include this WITHIN `namespace sg {` (it opens")
    lines.append("//       fused::sm90, i.e. sg::fused::sm90). PersistentContext")
    lines.append("//       must already be declared in sg::fused::sm90.")
    lines.append("namespace fused { namespace sm90 {")
    syms = []
    for plan in mk.solve_all():
        if plan.arch != "sm_90":
            continue
        sym = _cell_symbol(plan.model, plan.optimizer)
        syms.append((plan.model, plan.optimizer, sym))
        lines.append(f"cudaError_t {sym}({_CELL_LAUNCHER_SIG});")
    lines.append("")
    lines.append("inline cudaError_t dispatch_sm90_cell(")
    lines.append(" const std::string& model, const std::string& optimizer,")
    lines.append(" PersistentContext ctx, float* params, const float* input,")
    lines.append(" float* acts, float* grad, float* m, float* v, float* extra,")
    lines.append(" const int* sizes, const int* offsets, float lr, int step,")
    lines.append(" const FusedScalars& scalars, bool opt_only,")
    lines.append(" cudaStream_t stream, bool* found) {")
    lines.append(" *found = true;")
    for model, opt, sym in syms:
        lines.append(
            f' if (model == "{model}" && optimizer == "{opt}")')
        lines.append(
            f"  return {sym}(ctx, params, input, acts, grad, m, v, extra, sizes,"
            " offsets, lr, step, scalars, opt_only, stream);")
    lines.append(" *found = false;")
    lines.append(" return cudaSuccess;")
    lines.append("}")
    lines.append("}} // namespace fused::sm90  (within namespace sg)")
    return "\n".join(lines)


_GFX942_CELL_SIG = (
    "PersistentContext, float*, const float*, float*, float*, float*, float*, "
    "float*, const int*, const int*, float, int, const FusedScalars&, bool, "
    "hipStream_t")


def dispatch_table_gfx942() -> str:
    """Emit the generated gfx942 fused-cell dispatch table (.inc).

    AMD twin of dispatch_table_sm90: declares the 33 gfx942 host launchers
    (hipError_t, in sg::fused::gfx942_mega) and dispatch_gfx942_cell(). Compiled
    ONLY by a hipcc WITH_HIP build (🟡 MI300X); the WITH_CUDA build #if-excludes
    the branch that includes this. Include WITHIN `namespace sg {`.
    """
    lines: List[str] = []
    lines.append("// AUTO-GENERATED by megakernel_codegen.py --dispatch-table-gfx942")
    lines.append("// Do not edit by hand. hipcc/WITH_HIP only (🟡 MI300X).")
    lines.append("// PersistentContext must already be declared in sg::fused::gfx942_mega.")
    lines.append("namespace fused { namespace gfx942_mega {")
    syms = []
    for plan in mk.solve_all():
        if plan.arch != "gfx942":
            continue
        sym = _cell_symbol(plan.model, plan.optimizer)
        syms.append((plan.model, plan.optimizer, sym))
        lines.append(f"hipError_t {sym}({_GFX942_CELL_SIG});")
    lines.append("")
    lines.append("inline hipError_t dispatch_gfx942_cell(")
    lines.append(" const std::string& model, const std::string& optimizer,")
    lines.append(" PersistentContext ctx, float* params, const float* input,")
    lines.append(" float* acts, float* grad, float* m, float* v, float* extra,")
    lines.append(" const int* sizes, const int* offsets, float lr, int step,")
    lines.append(" const FusedScalars& scalars, bool opt_only,")
    lines.append(" hipStream_t stream, bool* found) {")
    lines.append(" *found = true;")
    for model, opt, sym in syms:
        lines.append(f' if (model == "{model}" && optimizer == "{opt}")')
        lines.append(
            f"  return {sym}(ctx, params, input, acts, grad, m, v, extra, sizes,"
            " offsets, lr, step, scalars, opt_only, stream);")
    lines.append(" *found = false;")
    lines.append(" return hipSuccess;")
    lines.append("}")
    lines.append("}} // namespace fused::gfx942_mega  (within namespace sg)")
    return "\n".join(lines)


# ── PHASE 1: the real transformer-decoder weight layout (single C++ source). ──
# The L3-REAL decoder megakernel (csrc/fused/sm_90/model_stages_decoder.cuh) needs
# the flat-buffer OFFSETS/SIZES of the eager model's 30 parameter tensors, in
# named_parameters() ORDER. This is THE single source of truth; the device header
# csrc/fused/sm_90/decoder_layout.cuh is GENERATED from it (so it cannot drift),
# and tests/hw/decoder_oracle.py::decoder_param_layout() builds the same table
# from the same shapes (asserted == the live model's named_parameters in the
# parity test). Keep the shape constants here == decoder_oracle.py.
_DEC_VOCAB, _DEC_D, _DEC_HEADS, _DEC_LAYERS, _DEC_SEQ = 99, 128, 4, 2, 4
_DEC_DFF = 4 * _DEC_D

# ── HILL-CLIMB BENCH VARIANT (d-scaled decoder) ──────────────────────────────
# The persistent megakernel's compute roofline is diagnosed/optimized at a LARGE
# model width. _DEC_BENCH_D is the width the d-scaled BENCHMARK build compiles at;
# it does NOT touch the production d=128 path. decoder_layout_header() emits BOTH
# layouts into ONE header, selected by the compile flag SG_DEC_BENCH_LAYOUT (set
# ONLY by the bench TU / the _ops_bench variant; UNSET → the production d=128
# constants, byte-identical default). So the committed header carries both, the
# production _ops always sees d=128, and the variant build flips ONE flag. The
# proven end-to-end d (task #13 context: parity 2.4e-07, A/A/A) is 1024.
_DEC_BENCH_D = 1024


def _decoder_param_sizes(d: int = _DEC_D) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of decoder_oracle.py
    decoder_param_spec()). 30 tensors; at d=128 total 422755. `d` is parametric
    so the d-scaled bench layout (SG_DEC_BENCH_LAYOUT) reuses the SAME shape
    formula — the per-tensor shapes are all functions of (d, dff=4d, vocab, seq),
    so a single d controls the whole table."""
    dff, v, seq = 4 * d, _DEC_VOCAB, _DEC_SEQ
    sizes = [v * d, seq * d]                      # tok, pos
    for _ in range(_DEC_LAYERS):
        sizes += [
            3 * d * d, 3 * d,                     # attn.in_proj_weight/bias
            d * d, d,                             # attn.out_proj.weight/bias
            d, d, d, d,                           # n1.w/b, n2.w/b
            dff * d, dff,                         # ff.0.weight/bias
            d * dff, d,                           # ff.2.weight/bias
        ]
    sizes += [d, d, v * d, v]                     # norm.w/b, out.weight/bias
    return sizes


def _decoder_layout_body(d: int) -> str:
    """The constants + __constant__ tables + compile-time cross-check for ONE
    decoder width `d`. Emitted into ONE of the SG_DEC_BENCH_LAYOUT branches; the
    branches are mutually exclusive at preprocess time, so reusing the SAME symbol
    names (kDecOffsets/kDecSizes/dec_layout_check) across both is safe."""
    sizes = _decoder_param_sizes(d)
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)

    def _fmt(arr):
        # 10 per line for readability.
        rows = []
        for i in range(0, len(arr), 10):
            rows.append("    " + ", ".join(str(x) for x in arr[i:i + 10]))
        return ",\n".join(rows)

    sizes_block = _fmt(sizes)
    offsets_block = _fmt(offsets)
    return f"""constexpr int SG_DEC_VOCAB  = {_DEC_VOCAB};
constexpr int SG_DEC_D      = {d};
constexpr int SG_DEC_HEADS  = {_DEC_HEADS};
constexpr int SG_DEC_LAYERS = {_DEC_LAYERS};
constexpr int SG_DEC_SEQ    = {_DEC_SEQ};
constexpr int SG_DEC_DFF    = 4 * SG_DEC_D;   // {4 * d}

constexpr int     kDecNumTensors = {n_tensors};
constexpr int64_t kDecTotalElems = {total};

// Per-tensor element offsets into the flat param blob, named_parameters() order.
__device__ __constant__ int kDecOffsets[kDecNumTensors] = {{
{offsets_block}
}};

// Per-tensor element sizes (numel), same order.
__device__ __constant__ int kDecSizes[kDecNumTensors] = {{
{sizes_block}
}};

static_assert(kDecNumTensors == {n_tensors},
              "decoder_layout: tensor count drifted. Regenerate via "
              "megakernel_codegen.py --decoder-layout.");
static_assert(kDecTotalElems == {total},
              "decoder_layout: total param count drifted. Regenerate.");

// Host-constexpr mirrors so a sum/offset cross-check folds at compile time (a
// __constant__ array can't be folded in a constexpr). These guarantee
// offsets/sizes/total agree.
namespace dec_layout_check {{
constexpr int kSizes[kDecNumTensors] = {{
{sizes_block}
}};
constexpr int kOffsets[kDecNumTensors] = {{
{offsets_block}
}};
constexpr int64_t sum_sizes() {{
    int64_t s = 0;
    for (int i = 0; i < kDecNumTensors; ++i) s += kSizes[i];
    return s;
}}
constexpr bool offsets_consistent() {{
    int64_t acc = 0;
    for (int i = 0; i < kDecNumTensors; ++i) {{
        if (kOffsets[i] != (int)acc) return false;
        acc += kSizes[i];
    }}
    return true;
}}
static_assert(sum_sizes() == kDecTotalElems,
              "decoder_layout: sum(kDecSizes) != kDecTotalElems. Regenerate.");
static_assert(offsets_consistent(),
              "decoder_layout: kDecOffsets[i] != sum(kDecSizes[0..i)). Regenerate.");
}}  // namespace dec_layout_check"""


def decoder_layout_header() -> str:
    """Emit csrc/fused/sm_90/decoder_layout.cuh (the GENERATED weight-layout
    mirror + static_asserts). Single source of truth for the L3-REAL decoder
    flat-buffer offsets/sizes; a count/total mismatch fails the BUILD.

    Carries TWO layouts under one include guard, selected by the compile flag
    SG_DEC_BENCH_LAYOUT (the hill-climb d-scaled bench-variant gate):
      * UNSET (default / production _ops)  -> d={_DEC_D} (33/33 path, byte-identical).
      * SG_DEC_BENCH_LAYOUT=1 (bench TU)   -> d={_DEC_BENCH_D} (the d-scaled roofline build).
    Because the branches are #if/#else, any ONE TU sees exactly one (d, table,
    static-assert) set consistently across every includer; the production build
    NEVER sees the bench branch (the flag is set only by the variant TU / the
    _ops_bench extension). See /workspace/.hillclimb_loop.md."""
    prod_body = _decoder_layout_body(_DEC_D)
    bench_body = _decoder_layout_body(_DEC_BENCH_D)
    return f"""#ifndef SG_FUSED_SM90_DECODER_LAYOUT_CUH_
#define SG_FUSED_SM90_DECODER_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/decoder_layout.cuh — GENERATED weight-layout mirror for the
// L3-REAL transformer-decoder megakernel (PHASE 1).
//
// AUTO-GENERATED by: python -m grokking_optimizers.megakernel_codegen \\
//     --decoder-layout > csrc/fused/sm_90/decoder_layout.cuh
// Do NOT hand-edit the numbers. SINGLE SOURCE OF TRUTH: megakernel_codegen.py
// _decoder_param_sizes() (mirrored by tests/hw/decoder_oracle.py, asserted ==
// the eager model's named_parameters() order in the parity test). The flat blob
// is torch.cat([p.reshape(-1) for _, p in model.named_parameters()]); the kernel
// addresses tensor i at params + kDecOffsets[i] for kDecSizes[i] elements.
//
// A count/total mismatch fails the BUILD loudly (a static_assert below), never
// corrupts at dispatch.
//
// ── SG_DEC_BENCH_LAYOUT (hill-climb d-scaled bench variant) ──────────────────
// Two layouts coexist under the single include guard. SG_DEC_BENCH_LAYOUT is
// UNSET on the production _ops build (→ the d={_DEC_D} branch, byte-identical to
// the historical header), and set to 1 ONLY by the d-scaled benchmark TU / the
// _ops_bench variant extension (→ the d={_DEC_BENCH_D} branch). The branches are
// mutually exclusive at preprocess time, so a TU compiles exactly one consistent
// (constants, __constant__ table, static-assert) set; production never sees the
// bench numbers. This is the loop infra for task #13 — see /workspace/.hillclimb_loop.md.
// ============================================================================

#include <cstdint>

#ifndef SG_DEC_BENCH_LAYOUT
#define SG_DEC_BENCH_LAYOUT 0
#endif

namespace sg {{ namespace fused {{ namespace sm90 {{

#if SG_DEC_BENCH_LAYOUT
// ── d-SCALED BENCH VARIANT (d={_DEC_BENCH_D}): the persistent-megakernel roofline
//    build. NOT on the production path; selected ONLY by -DSG_DEC_BENCH_LAYOUT=1. ──
{bench_body}
#else
// ── PRODUCTION (d={_DEC_D}): the 33/33 wiring_check path. Byte-identical to the
//    historical generated header (the default when SG_DEC_BENCH_LAYOUT is unset). ──
{prod_body}
#endif  // SG_DEC_BENCH_LAYOUT

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_DECODER_LAYOUT_CUH_
"""


# ── PHASE 2: the real ViT weight layout (single C++ source). ────────────────
# The L3-REAL ViT megakernel (csrc/fused/sm_90/model_stage_vit.cuh) needs the
# flat-buffer OFFSETS/SIZES of the eager model's 32 parameter tensors, in
# named_parameters() ORDER. THE single source of truth is the same shapes the
# oracle (tests/hw/vit_oracle.py::vit_param_layout()) encodes; the device header
# csrc/fused/sm_90/vit_layout.cuh is GENERATED from THIS so it cannot drift.
# CRITICAL ORDERING: cls_token (a leaf nn.Parameter on the ViT module) is yielded
# BEFORE the patch_proj submodule's params, so the first three tensors are
# cls_token, patch_proj.weight, patch_proj.bias. Keep these constants ==
# vit_oracle.py.
_VIT_VOCAB, _VIT_D, _VIT_HEADS, _VIT_LAYERS, _VIT_PATCH, _VIT_NPATCH = \
    97, 128, 4, 2, 49, 16
_VIT_DFF = 4 * _VIT_D


def _vit_param_sizes() -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of vit_oracle.py
    vit_param_layout()). 32 tensors, total 418017. cls_token leads (leaf before
    the patch_proj submodule)."""
    d, dff, v, patch, npatch = _VIT_D, _VIT_DFF, _VIT_VOCAB, _VIT_PATCH, _VIT_NPATCH
    sizes = [1 * 1 * d, d * patch, d, (npatch + 1) * d]   # cls, patch.w/b, pos
    for _ in range(_VIT_LAYERS):
        sizes += [
            3 * d * d, 3 * d,                     # attn.in_proj_weight/bias
            d * d, d,                             # attn.out_proj.weight/bias
            d, d, d, d,                           # n1.w/b, n2.w/b
            dff * d, dff,                         # ff.0.weight/bias
            d * dff, d,                           # ff.2.weight/bias
        ]
    sizes += [d, d, v * d, v]                     # norm.w/b, out.weight/bias
    return sizes


def vit_layout_header() -> str:
    """Emit csrc/fused/sm_90/vit_layout.cuh (the weight-layout mirror +
    static_asserts + the dynamic-smem budget block). Byte-identical to the
    hand-written header; a count/total/smem mismatch fails the BUILD."""
    sizes = _vit_param_sizes()
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)

    def _fmt(arr):
        rows = []
        for i in range(0, len(arr), 10):
            rows.append("    " + ", ".join(str(x) for x in arr[i:i + 10]))
        return ",\n".join(rows)

    sizes_block = _fmt(sizes)
    offsets_block = _fmt(offsets)
    return f"""#ifndef SG_FUSED_SM90_VIT_LAYOUT_CUH_
#define SG_FUSED_SM90_VIT_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/vit_layout.cuh — weight-layout mirror for the L3-REAL
// Vision-Transformer megakernel (PHASE 2).
//
// HAND-WRITTEN (PHASE 2 deliverable) — MARKED FOR CODEGEN ADOPTION. The decoder
// twin (decoder_layout.cuh) is emitted by megakernel_codegen.py --decoder-layout;
// the integrator should add a `--vit-layout` emitter (mirroring
// _decoder_param_sizes / decoder_layout_header) whose output is byte-identical to
// THIS file, then regenerate so it cannot drift. Until then, the numbers below are
// generated FROM the single source of truth — tests/hw/vit_oracle.py
// ::vit_param_layout() (asserted == the live model's named_parameters() order in
// the parity test, tests/hw/test_vit_megakernel.py) — and guarded by the
// static_asserts below. A count/total mismatch fails the BUILD loudly, never
// corrupts at dispatch.
//
// The flat blob is torch.cat([p.reshape(-1) for _, p in model.named_parameters()]);
// the kernel addresses tensor i at params + kVitOffsets[i] for kVitSizes[i] elems.
//
// named_parameters() ORDER (32 tensors) — note cls_token (a leaf nn.Parameter on
// the ViT module) is yielded BEFORE the patch_proj submodule's params:
//   0  cls_token              [1,1,128]
//   1  patch_proj.weight      [128,49]
//   2  patch_proj.bias        [128]
//   3  pos.weight             [17,128]
//   4..15   layers.0.{{attn.in_proj_w/b, attn.out_proj.w/b, n1.w/b, n2.w/b,
//                     ff.0.w/b, ff.2.w/b}}
//   16..27  layers.1.{{...same...}}
//   28 norm.weight [128]  29 norm.bias [128]  30 out.weight [97,128]  31 out.bias [97]
// ============================================================================

#include <cstdint>

namespace sg {{ namespace fused {{ namespace sm90 {{

constexpr int SG_VIT_VOCAB  = {_VIT_VOCAB};          // p (head Linear(d, p))
constexpr int SG_VIT_D      = {_VIT_D};
constexpr int SG_VIT_HEADS  = {_VIT_HEADS};
constexpr int SG_VIT_LAYERS = {_VIT_LAYERS};
constexpr int SG_VIT_PATCH  = {_VIT_PATCH};          // patch pixel count (7×7)
constexpr int SG_VIT_NPATCH = {_VIT_NPATCH};          // image patches
constexpr int SG_VIT_SEQ    = SG_VIT_NPATCH + 1;  // 17 (CLS + 16 patches)
constexpr int SG_VIT_DFF    = 4 * SG_VIT_D;       // 512

constexpr int     kVitNumTensors = {n_tensors};
constexpr int64_t kVitTotalElems = {total};

// Per-tensor element offsets into the flat param blob, named_parameters() order.
__device__ __constant__ int kVitOffsets[kVitNumTensors] = {{
{offsets_block}
}};

// Per-tensor element sizes (numel), same order.
__device__ __constant__ int kVitSizes[kVitNumTensors] = {{
{sizes_block}
}};

static_assert(kVitNumTensors == {n_tensors},
              "vit_layout: tensor count drifted. Regenerate from "
              "vit_oracle.py::vit_param_layout() (codegen adoption: --vit-layout).");
static_assert(kVitTotalElems == {total},
              "vit_layout: total param count drifted. Regenerate.");

// Host-constexpr mirrors so a sum/offset cross-check folds at compile time (a
// __constant__ array can't be folded in a constexpr). These guarantee
// offsets/sizes/total agree.
namespace vit_layout_check {{
constexpr int kSizes[kVitNumTensors] = {{
{sizes_block}
}};
constexpr int kOffsets[kVitNumTensors] = {{
{offsets_block}
}};
constexpr int64_t sum_sizes() {{
    int64_t s = 0;
    for (int i = 0; i < kVitNumTensors; ++i) s += kSizes[i];
    return s;
}}
constexpr bool offsets_consistent() {{
    int64_t acc = 0;
    for (int i = 0; i < kVitNumTensors; ++i) {{
        if (kOffsets[i] != (int)acc) return false;
        acc += kSizes[i];
    }}
    return true;
}}
static_assert(sum_sizes() == kVitTotalElems,
              "vit_layout: sum(kVitSizes) != kVitTotalElems. Regenerate.");
static_assert(offsets_consistent(),
              "vit_layout: kVitOffsets[i] != sum(kVitSizes[0..i)). Regenerate.");

// ── Per-CTA dynamic-smem budget guard. The ViT per-sample working set
//    (VitSampleSmem, model_stage_vit.cuh) is ≈ 188,080 bytes (≈ 183.67 KB) at
//    seq=17 — FAR over the 48 KB STATIC __shared__ cap (so it MUST be dynamic
//    smem), but comfortably UNDER the sm_90 per-block dynamic cap of 227 KB
//    (232448 B). This bound is the size the launcher passes to
//    cudaFuncSetAttribute(MaxDynamicSharedMemorySize) + <<<dynamicSMemBytes>>>.
//    Computed from the field list (all float, 4 B, no padding):
//      patch 16*49 + layer_in 2*17*128 + final_in 17*128 + qkv 17*384
//      + ctx 17*128 + x1 17*128 + ff0 17*512 + gact 17*512 + attn 4*17*17
//      + n1_xhat 17*128 + n1_inv 17 + n2_xhat 17*128 + n2_inv 17
//      + fn_xhat 17*128 + fn_inv 17 + dh 17*128 + logits 97 + dsc 4*17*17
//      + red 256  = 47020 floats = 188080 bytes.
//    (If VitSampleSmem changes, update BOTH this literal and the sum below;
//    fused_vit_megakernel.cuh static_asserts sizeof(VitSampleSmem) against it.) ─
constexpr int kVitSampleSmemFloats =
    16 * 49            // patch[NPATCH][PATCH]
  + 2 * 17 * 128       // layer_in[LAYERS][SEQ][D]
  + 17 * 128           // final_in[SEQ][D]
  + 17 * 384           // qkv[SEQ][3D]
  + 17 * 128           // ctx[SEQ][D]
  + 17 * 128           // x1[SEQ][D]
  + 17 * 512           // ff0[SEQ][DFF]
  + 17 * 512           // gact[SEQ][DFF]
  + 4 * 17 * 17        // attn[HEADS][SEQ][SEQ]
  + 17 * 128 + 17      // n1_xhat[SEQ][D] + n1_inv[SEQ]
  + 17 * 128 + 17      // n2_xhat[SEQ][D] + n2_inv[SEQ]
  + 17 * 128 + 17      // fn_xhat[SEQ][D] + fn_inv[SEQ]
  + 17 * 128           // dh[SEQ][D]
  + 97                 // logits[VOCAB]
  + 4 * 17 * 17        // dsc[HEADS][SEQ][SEQ]
  + 256;               // red[256]
constexpr int kVitSampleSmemBytes = kVitSampleSmemFloats * (int)sizeof(float);
static_assert(kVitSampleSmemBytes == 188080,
              "vit_layout: VitSampleSmem byte budget drifted from 188080.");
static_assert(kVitSampleSmemBytes < 227 * 1024,
              "vit_layout: VitSampleSmem exceeds the sm_90 227 KB dynamic-smem "
              "per-block cap — the megakernel could not launch.");
}}  // namespace vit_layout_check

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_VIT_LAYOUT_CUH_
"""


# ── PHASE 2: the real Mamba weight layout (single C++ source). ──────────────
# The L3-REAL Mamba megakernel (csrc/fused/sm_90/model_stage_mamba3.cuh) needs the
# flat-buffer OFFSETS/SIZES of the eager model's 28 parameter tensors, in
# named_parameters() ORDER. THE single source of truth is the same shapes the
# oracle (tests/hw/mamba_oracle.py::mamba_param_layout()) encodes; the device
# header csrc/fused/sm_90/mamba3_layout.cuh is GENERATED from THIS so it cannot
# drift. CRITICAL ORDERING: within each SelectiveSSMLayer, A_log and D (the
# module's OWN nn.Parameters) are yielded BEFORE in_proj (a submodule) — NOT the
# __init__ visual order. Keep these constants == mamba_oracle.py.
_MAMBA_VOCAB, _MAMBA_PHEAD, _MAMBA_D, _MAMBA_LAYERS, _MAMBA_SEQ = 99, 97, 128, 2, 8
_MAMBA_DINNER, _MAMBA_STATE, _MAMBA_DTRANK, _MAMBA_CONVK = 256, 16, 8, 3


def _mamba_param_sizes() -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of mamba_oracle.py
    mamba_param_layout()). 28 tensors, total 259425. Per layer the leaf-before-
    submodule order is A_log, D, in_proj, conv1d.w, conv1d.b, x_proj, dt_proj.w,
    dt_proj.b, out_proj, norm.w, norm.b."""
    d, v, ph, seq = _MAMBA_D, _MAMBA_VOCAB, _MAMBA_PHEAD, _MAMBA_SEQ
    di, st, dtr, ck = _MAMBA_DINNER, _MAMBA_STATE, _MAMBA_DTRANK, _MAMBA_CONVK
    sizes = [v * d, seq * d]                          # tok, pos
    for _ in range(_MAMBA_LAYERS):
        sizes += [
            di * st,                                  # A_log [d_inner, state]
            di,                                       # D [d_inner]
            2 * di * d,                               # in_proj [2*d_inner, d]
            di * 1 * ck,                              # conv1d.weight [d_inner,1,3]
            di,                                       # conv1d.bias [d_inner]
            (dtr + 2 * st) * di,                      # x_proj [dt_rank+2*state, d_inner]
            di * dtr,                                 # dt_proj.weight [d_inner, dt_rank]
            di,                                       # dt_proj.bias [d_inner]
            d * di,                                   # out_proj [d, d_inner]
            d, d,                                     # norm.weight/bias
        ]
    sizes += [d, d, ph * d, ph]                       # norm.w/b, out.weight/bias
    return sizes


def mamba_layout_header() -> str:
    """Emit csrc/fused/sm_90/mamba3_layout.cuh (the weight-layout mirror +
    static_asserts + the dynamic-smem budget block). Byte-identical to the
    hand-written header; a count/total/smem mismatch fails the BUILD.

    kMambaSmemFloats is the FIELD-BY-FIELD element count of MambaSampleSmem
    (model_stage_mamba3.cuh), NOT a sizeof() (this generator is host Python, not
    nvcc). The 36281 total is documented field-by-field in the emitted comment.
    """
    sizes = _mamba_param_sizes()
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)

    def _fmt(arr):
        rows = []
        for i in range(0, len(arr), 10):
            rows.append("    " + ", ".join(str(x) for x in arr[i:i + 10]))
        return ",\n".join(rows)

    sizes_block = _fmt(sizes)
    offsets_block = _fmt(offsets)
    return f"""#ifndef SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
#define SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/mamba3_layout.cuh — weight-layout mirror for the L3-REAL
// Mamba megakernel (PHASE 2).
//
// HAND-WRITTEN (not generated): megakernel_codegen.py is OFF-LIMITS for this
// phase per the owner directive, so this header is hand-derived from the SINGLE
// SOURCE OF TRUTH tests/hw/mamba_oracle.py::mamba_param_layout(), which is
// asserted == the eager model's named_parameters() ORDER + count + total in the
// parity test (tests/hw/test_mamba_megakernel.py). When megakernel_codegen.py is
// later extended with a --mamba-layout emitter (the decoder pattern), this file
// becomes its generated output; until then the numbers below are pinned by the
// static_asserts and the Python parity test. The flat blob is
// torch.cat([p.reshape(-1) for _, p in model.named_parameters()]); the kernel
// addresses tensor i at params + kMambaOffsets[i] for kMambaSizes[i] elements.
//
// CRITICAL ORDERING NOTE: PyTorch yields a module's OWN nn.Parameters before its
// submodules, so within each SelectiveSSMLayer **A_log and D come BEFORE in_proj**
// (NOT the __init__ visual order). The order below matches the verified dump.
//
// A count/total mismatch fails the BUILD loudly (the static_asserts below), never
// corrupts at dispatch.
// ============================================================================

#include <cstdint>

namespace sg {{ namespace fused {{ namespace sm90 {{

constexpr int SG_MB_VOCAB  = {_MAMBA_VOCAB};    // ntok (tok embedding rows)
constexpr int SG_MB_PHEAD  = {_MAMBA_PHEAD};    // p (head width = out Linear cols)
constexpr int SG_MB_D      = {_MAMBA_D};   // d_model
constexpr int SG_MB_LAYERS = {_MAMBA_LAYERS};     // nl
constexpr int SG_MB_SEQ    = {_MAMBA_SEQ};     // seq_len
constexpr int SG_MB_DINNER = {_MAMBA_DINNER};   // d_inner (= d * expand_factor=2)
constexpr int SG_MB_STATE  = {_MAMBA_STATE};    // state_dim
constexpr int SG_MB_DTRANK = {_MAMBA_DTRANK};     // dt_rank = max(d/16,1)
constexpr int SG_MB_CONVK  = {_MAMBA_CONVK};     // conv1d kernel size

constexpr int     kMambaNumTensors = {n_tensors};
constexpr int64_t kMambaTotalElems = {total};

// Per-tensor element offsets into the flat param blob, named_parameters() order.
__device__ __constant__ int kMambaOffsets[kMambaNumTensors] = {{
{offsets_block}
}};

// Per-tensor element sizes (numel), same order.
__device__ __constant__ int kMambaSizes[kMambaNumTensors] = {{
{sizes_block}
}};

static_assert(kMambaNumTensors == {n_tensors},
              "mamba3_layout: tensor count drifted from the oracle layout (28).");
static_assert(kMambaTotalElems == {total},
              "mamba3_layout: total param count drifted (259425).");

// Host-constexpr mirrors so a sum/offset cross-check folds at compile time (a
// __constant__ array can't be folded in a constexpr).
namespace mamba_layout_check {{
constexpr int kSizes[kMambaNumTensors] = {{
{sizes_block}
}};
constexpr int kOffsets[kMambaNumTensors] = {{
{offsets_block}
}};
constexpr int64_t sum_sizes() {{
    int64_t s = 0;
    for (int i = 0; i < kMambaNumTensors; ++i) s += kSizes[i];
    return s;
}}
constexpr bool offsets_consistent() {{
    int64_t acc = 0;
    for (int i = 0; i < kMambaNumTensors; ++i) {{
        if (kOffsets[i] != (int)acc) return false;
        acc += kSizes[i];
    }}
    return true;
}}
static_assert(sum_sizes() == kMambaTotalElems,
              "mamba3_layout: sum(kMambaSizes) != kMambaTotalElems. Re-derive "
              "from tests/hw/mamba_oracle.py::mamba_param_layout().");
static_assert(offsets_consistent(),
              "mamba3_layout: kMambaOffsets[i] != sum(kMambaSizes[0..i)).");
}}  // namespace mamba_layout_check

// ── SMEM footprint of MambaSampleSmem (model_stage_mamba3.cuh). The Mamba CTA
//    smem EXCEEDS the 48 KB static cap (d_inner=256 + both-layer caching), so the
//    launcher MUST declare it DYNAMIC and opt in via
//    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
//    kMambaSmemBytes) before launch (sm_90 has ~228 KB smem/SM; one persistent
//    block/SM still places, occupancy>=1 holds, NO CUDA graphs). This is the
//    HONEST deviation from the decoder's <48 KB static footprint — see the
//    SMEM BUDGET note in model_stage_mamba3.cuh and INTEGRATION-MAMBA.md.
//
//    The number below is a static_assert-guarded mirror of sizeof(MambaSampleSmem)
//    computed field-by-field (all fields are float, 4-byte aligned -> no padding):
//      layer_in   [2][8][128]              = 2048
//      final_in   [8][128]                 = 1024
//      act[2].{{ x_main_raw,z,conv,dt_pre,y_scan [8][256] (5*2048),
//               Bmat,Cmat [8][16] (2*128), ln_xhat [8][128], ln_inv [8] }}
//                 per layer = 11528 ; x2  = 23056
//      fn_xhat[8][128]=1024; fn_inv[8]=8; logits[97]=97
//      dh,dr [8][128] (2*1024); adj_a,adj_b,adj_c [8][256] (3*2048)
//      dbc [8][40]=320; dBmat,dCmat [8][16] (2*128); red[256]
//      ------------------------------------------------------- = 36281 floats
constexpr int64_t kMambaSmemFloats = 36281;
constexpr int64_t kMambaSmemBytes  = kMambaSmemFloats * (int64_t)sizeof(float); // 145124
static_assert(kMambaSmemBytes > 48 * 1024,
              "mamba3_layout: if the smem ever drops below 48KB, switch the "
              "launcher back to STATIC smem (no opt-in needed).");
static_assert(kMambaSmemBytes < 224 * 1024,
              "mamba3_layout: CTA smem exceeds the sm_90 ~228KB/SM budget; "
              "one-block-per-SM would fail to place (GridBarrier hang). Shrink "
              "the live set (e.g. drop a cached LayerAct buffer).");

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
"""


def dispatch_table() -> str:
    """Emit the C++ wired_fused_cell() body covering all 99 cells."""
    lines: List[str] = []
    lines.append(
        "// AUTO-GENERATED by: python -m grokking_optimizers.megakernel_codegen "
        "--dispatch-table")
    lines.append(
        "// Do NOT hand-edit. Derived from the SAME solver enumeration "
        "(megakernel.solve_all)")
    lines.append(
        "// that emits the 99 csrc/fused/<arch>/mega_*.{cu,hip,py} cells, so it "
        "cannot drift.")
    lines.append("// dispatch.cpp #includes this inside its anonymous namespace.")
    lines.append("std::string wired_fused_cell(const std::string& model,")
    lines.append(" const std::string& optimizer, int arch) {")
    seen: set = set()
    for plan in mk.solve_all():
        key = (plan.model, plan.optimizer, plan.arch)
        if key in seen:
            continue
        seen.add(key)
        tier_tag = _fuse_tier_tag(plan.tier, lower=True)
        arch_int = {"sm_90": 90, "gfx942": 942, "tpu_v6e": -1}[plan.arch]
        arch_lit = {"sm_90": "sm_90", "gfx942": "gfx942",
                    "tpu_v6e": "tpu_v6e"}[plan.arch]
        lines.append(
            f' if (arch == {arch_int} && model == "{plan.model}"'
            f' && optimizer == "{plan.optimizer}")')
        lines.append(
            f'  return "{tier_tag}:{plan.model}+{plan.optimizer}:{arch_lit}";')
    lines.append(' return "";')
    lines.append("}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="megakernel_codegen",
        description="Stage 6 L3 megakernel generator (emits per-cell source).")
    ap.add_argument("--emit", nargs=3, metavar=("MODEL", "OPTIMIZER", "ARCH"),
                    help="emit ONE cell's megakernel source to stdout")
    ap.add_argument("--emit-all", action="store_true",
                    help="report the full 99-cell manifest with tiers")
    ap.add_argument("--write-all", action="store_true",
                    help="materialize all 99 cell source files to csrc/fused/")
    ap.add_argument("--dispatch-table", action="store_true",
                    help="emit the C++ dispatch table covering all 99 cells")
    ap.add_argument("--dispatch-table-sm90", action="store_true",
                    help="emit the generated sm_90 real-composition dispatch .inc")
    ap.add_argument("--dispatch-table-gfx942", action="store_true",
                    help="emit the generated gfx942 real-composition dispatch .inc")
    ap.add_argument("--decoder-layout", action="store_true",
                    help="emit the PHASE-1 L3-REAL decoder weight-layout header "
                         "(csrc/fused/sm_90/decoder_layout.cuh)")
    ap.add_argument("--vit-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL ViT weight-layout header "
                         "(csrc/fused/sm_90/vit_layout.cuh)")
    ap.add_argument("--mamba-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL Mamba weight-layout header "
                         "(csrc/fused/sm_90/mamba3_layout.cuh)")
    args = ap.parse_args(argv)

    if args.emit:
        model, optimizer, arch = args.emit
        try:
            sys.stdout.write(emit_cell(model, optimizer, arch))
        except (KeyError, RuntimeError) as exc:
            sys.stderr.write(f"emit failed: {exc}\n")
            return 2
        return 0

    if args.emit_all:
        sys.stdout.write(_format_manifest(manifest()) + "\n")
        return 0

    if args.write_all:
        paths = write_all()
        sys.stdout.write(f"Wrote {len(paths)} cell files:\n")
        for p in paths:
            sys.stdout.write(f"  {p}\n")
        return 0

    if args.dispatch_table:
        sys.stdout.write(dispatch_table() + "\n")
        return 0

    if args.dispatch_table_sm90:
        sys.stdout.write(dispatch_table_sm90() + "\n")
        return 0

    if args.dispatch_table_gfx942:
        sys.stdout.write(dispatch_table_gfx942() + "\n")
        return 0

    if args.decoder_layout:
        sys.stdout.write(decoder_layout_header())
        return 0

    if args.vit_layout:
        sys.stdout.write(vit_layout_header())
        return 0

    if args.mamba_layout:
        sys.stdout.write(mamba_layout_header())
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
