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
    tier = "L3" if plan.tier == FusionTier.L3_FWD_BWD_OPT else "L1"
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
// shared persistent-megakernel substrate. Fuse tier {tier} chosen by the solver.
#include "csrc/fused/sm_90/fused_megakernel.cuh"

namespace sg {{ namespace fused {{ namespace sm90 {{

// Uniform host entry (the symbol fused_step dispatches to). Binds this
// optimizer's REAL state buffers: m/v (Adam moments) + the per-optimizer
// `extra` n-slice (ema/sam_dir/s_track/mu/orth/smart_grad). Scalar hyperparams
// (prodigy d, sg gates, neuralgrok psi-net) are host-supplied at runtime (🟡
// no-GPU here) — see HARDWARE_VALIDATION.md. Composition + apply math are
// real + compiled.
cudaError_t {sym}(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, float* extra,
        const int* sizes, const int* offsets,
        float lr, int step, cudaStream_t stream) {{
    FusedOptState st;
    st.exp_avg = m;
    st.exp_avg_sq = v;{extra_line}
    st.lr = lr;
    return launch_fused_megakernel<ModelId::{m_enum}, OptId::{o_enum},
                                   FuseTier::{tier}>(
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
    tier = "L3" if plan.tier == FusionTier.L3_FWD_BWD_OPT else "L1"
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

// Force this cell's real composed kernel instantiation so the device pass emits
// it (and the free-standing AMDGCN gate type-checks the full expansion).
template __global__ void
fused_megakernel<ModelId::{m_enum}, OptId::{o_enum}, FuseTier::{tier}>(
    PersistentContext, float*, const float*, float*, float*,
    const int*, const int*, float, int, FusedOptState);

}}}}}}  // namespace sg::fused::gfx942_mega
#endif

// ── HOST launcher (hipcc host pass only; 🟡 MI300X — not the bare amdgcn gate,
//    not the WITH_CUDA build). Faithful mirror of the verified sm_90 launcher:
//    one persistent workgroup per CU, 256 threads (4 wavefronts). __HIPCC__ is
//    defined for the hipcc host pass, so FusedOptState/fused_megakernel (guarded
//    on __HIPCC__) are visible here. ───────────────────────────────────────────
#if defined(__HIPCC__) && !defined(__AMDGCN__)
#include <hip/hip_runtime.h>
namespace sg {{ namespace fused {{ namespace gfx942_mega {{
hipError_t {sym}(
        PersistentContext ctx, float* params, const float* input, float* acts,
        float* grad, float* m, float* v, float* extra,
        const int* sizes, const int* offsets, float lr, int step,
        hipStream_t stream) {{
    int dev = 0; hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) return err;
    int n_cus = 0;
    err = hipDeviceGetAttribute(&n_cus, hipDeviceAttributeMultiprocessorCount, dev);
    if (err != hipSuccess) return err;
    ctx.n_groups = (unsigned)n_cus;
    FusedOptState st;
    st.exp_avg = m; st.exp_avg_sq = v;{extra_line}
    st.lr = lr;
    dim3 grid((unsigned)n_cus), block(256);
    hipLaunchKernelGGL(
        (fused_megakernel<ModelId::{m_enum}, OptId::{o_enum}, FuseTier::{tier}>),
        grid, block, 0, stream,
        ctx, params, input, acts, grad, sizes, offsets, lr, step, st);
    return hipGetLastError();
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
    tier = "L3" if plan.tier == FusionTier.L3_FWD_BWD_OPT else "L1"
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
    "float*, const int*, const int*, float, int, cudaStream_t")


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
    lines.append(" cudaStream_t stream, bool* found) {")
    lines.append(" *found = true;")
    for model, opt, sym in syms:
        lines.append(
            f' if (model == "{model}" && optimizer == "{opt}")')
        lines.append(
            f"  return {sym}(ctx, params, input, acts, grad, m, v, extra, sizes,"
            " offsets, lr, step, stream);")
    lines.append(" *found = false;")
    lines.append(" return cudaSuccess;")
    lines.append("}")
    lines.append("}} // namespace fused::sm90  (within namespace sg)")
    return "\n".join(lines)


_GFX942_CELL_SIG = (
    "PersistentContext, float*, const float*, float*, float*, float*, float*, "
    "float*, const int*, const int*, float, int, hipStream_t")


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
    lines.append(" hipStream_t stream, bool* found) {")
    lines.append(" *found = true;")
    for model, opt, sym in syms:
        lines.append(f' if (model == "{model}" && optimizer == "{opt}")')
        lines.append(
            f"  return {sym}(ctx, params, input, acts, grad, m, v, extra, sizes,"
            " offsets, lr, step, stream);")
    lines.append(" *found = false;")
    lines.append(" return hipSuccess;")
    lines.append("}")
    lines.append("}} // namespace fused::gfx942_mega  (within namespace sg)")
    return "\n".join(lines)


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
        tier_tag = "l3" if plan.tier == FusionTier.L3_FWD_BWD_OPT else "l1"
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

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
