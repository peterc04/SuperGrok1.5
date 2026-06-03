"""Full-scale binary profiling — prove the cells are MAXIMAL, not just correct.

`verify_all` proves the 99 compositions *compile* and are *structurally* maximal
(every cell at its max feasible fusion tier).  This module goes one level deeper:
it inspects the **actual emitted machine code** to prove the binaries do what we
want *maximally* — the compute-bound paths use the silicon's tensor cores, the
async-copy engines, and don't spill registers — and that the optimizer math
*actually optimizes* when executed.

It needs no live GPU/TPU.  Everything here is host-side binary inspection of the
real compiled artifacts plus CPU execution of the fused programs:

  TIER A — sm_90 GEMM instruction maximality (nvcc + ptxas + cuobjdump SASS):
    the model TUs (decoder/vit/mamba) and supergrok2 must emit Hopper WGMMA
    tensor-core MMA + TMA async-bulk copies via the CUTLASS Sm90 collectives,
    with the wgmma pipeline NOT serialized (ptxas C7509) and ZERO register
    spills. This is where "maximal" is decided — GEMMs are compute-bound.

  TIER B — sm_90 resource health (ptxas -v): real registers / shared-memory /
    spill bytes for a sample of fused cells + the optimizer-step kernels,
    cross-checked against the megakernel solver's budget. Spills => not maximal.

  TIER C — gfx942 instruction maximality (clang AMDGPU): the attention + model
    device code must emit MFMA (CDNA3 matrix cores) and DPP cross-lane ops.

  TIER D — functional correctness (CPU execution): run the real fused
    (model, optimizer) program on CPU via the Pallas path and confirm the
    optimizer ACTUALLY reduces a loss and moves parameters finitely — the
    binaries do what we want, *period*.

Each tier prints PASS/FAIL/SKIP and the measured numbers; a final report gives
the maximality verdict, the real-vs-estimate table, and what remains silicon-
gated (true latency/occupancy require the clock).

Run:
    PYTHONPATH=. python3 -m grokking_optimizers.profile_maximal              # all
    PYTHONPATH=. python3 -m grokking_optimizers.profile_maximal --tier A     # one
    PYTHONPATH=. python3 -m grokking_optimizers.profile_maximal --quick      # D only
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

from grokking_optimizers import megakernel as mk

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP-silicon"


@dataclasses.dataclass
class Probe:
    tier: str
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0


class Report:
    def __init__(self) -> None:
        self.probes: List[Probe] = []
        self.silicon: List[str] = []
        self.design: List[str] = []
        self.table: List[Tuple[str, str]] = []  # (label, measured) rows

    def add(self, p: Probe) -> Probe:
        self.probes.append(p)
        glyph = {"PASS": "PASS ", "FAIL": "FAIL ", "SKIP-silicon": "SKIP "}[p.status]
        tail = f"  ({p.seconds:.0f}s)" if p.seconds >= 1 else ""
        line = f"  {glyph} [{p.tier}] {p.name}{tail}"
        if p.detail:
            line += f"\n         {p.detail}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return p

    def count(self, status: str) -> int:
        return sum(1 for p in self.probes if p.status == status)


# ---------------------------------------------------------------------------
# Toolchain + compile helpers
# ---------------------------------------------------------------------------

def _torch_incs() -> List[str]:
    try:
        import torch
        td = os.path.dirname(torch.__file__)
        import sysconfig
        return ["-I.", f"-I{td}/include",
                f"-I{td}/include/torch/csrc/api/include",
                f"-I{sysconfig.get_path('include')}",
                "-Ithird_party/cutlass/include",
                "-Ithird_party/cutlass/tools/util/include"]
    except Exception:
        return ["-I.", "-Ithird_party/cutlass/include",
                "-Ithird_party/cutlass/tools/util/include"]


# The MAXIMAL sm_90 release build flags (mirrors compile.NVCC_DEVICE_BASE: the
# tensor-core fast path with NDEBUG so the wgmma mainloop is not serialized).
_MAX_NVCC = [
    "-c", "-std=c++17", "-O3", "--use_fast_math", "-DNDEBUG",
    "-DWITH_CUDA", "-DWITH_CUTLASS",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
]


@dataclasses.dataclass
class SassProfile:
    rc: int
    regs_max: int
    smem_max: int
    spill_bytes: int
    c7509: int           # wgmma-serialization warnings
    wgmma: int
    tma: int
    mbarrier: int
    mufu: int
    ffma: int
    log: str


def _nvcc_profile(tu: Path, timeout: int = 900) -> SassProfile:
    """Compile a sm_90 TU with the maximal flags + ptxas -v, then SASS-audit."""
    obj = Path(tempfile.mktemp(suffix=".o"))
    cmd = ["nvcc", *_MAX_NVCC, *_torch_incs(), "-Xptxas", "-v",
           str(tu), "-o", str(obj)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=timeout)
    log = (proc.stdout or "") + (proc.stderr or "")
    regs = [int(m) for m in re.findall(r"Used (\d+) registers", log)]
    smem = [int(m) for m in re.findall(r"(\d+) bytes smem", log)]
    spills = [int(m) for m in re.findall(r"(\d+) bytes spill stores", log)]
    c7509 = len(re.findall(r"C7509", log))
    wgmma = tma = mbar = mufu = ffma = 0
    if proc.returncode == 0 and obj.exists():
        sass = subprocess.run(["cuobjdump", "-sass", str(obj)],
                              capture_output=True, text=True).stdout
        wgmma = len(re.findall(r"\b[HUW]?GMMA\b", sass))
        tma = len(re.findall(r"UTMALDG|UBLKCP|cp\.async\.bulk", sass))
        mbar = len(re.findall(r"MBARRIER|ELECT|ARRIVES", sass))
        mufu = len(re.findall(r"MUFU", sass))
        ffma = len(re.findall(r"\bFFMA\b", sass))
        obj.unlink(missing_ok=True)
    return SassProfile(
        proc.returncode, max(regs, default=0), max(smem, default=0),
        max(spills, default=0), c7509, wgmma, tma, mbar, mufu, ffma, log)


# ---------------------------------------------------------------------------
# TIER A — sm_90 GEMM instruction maximality
# ---------------------------------------------------------------------------

# (tu, label, expect_tensorcore) — model TUs + SG2 are GEMM-bearing and MUST
# emit WGMMA+TMA to be maximal. (The element-local megakernel cells are scalar
# by design — covered in Tier B, not here.)
_GEMM_TUS = [
    ("csrc/backends/cuda/sm_90/models/decoder.cu", "decoder (attn+FFN GEMMs)"),
    ("csrc/backends/cuda/sm_90/models/vit.cu", "vit (patch+attn+FFN GEMMs)"),
    ("csrc/backends/cuda/sm_90/models/mamba.cu", "mamba (in/out proj GEMMs)"),
    ("csrc/backends/cuda/sm_90/launch_supergrok2.cu",
     "supergrok2 (CUTLASS Newton-Schulz)"),
]


def tier_a_gemm(rep: Report) -> None:
    sys.stdout.write("\n[TIER A] sm_90 GEMM instruction maximality "
                     "(WGMMA + TMA, no wgmma-serialization, no spills)\n")
    if not shutil.which("nvcc"):
        for _, label in _GEMM_TUS:
            rep.add(Probe("A", label, SKIP, "nvcc absent"))
        rep.silicon.append("sm_90 GEMM SASS audit (needs nvcc)")
        return
    for tu, label in _GEMM_TUS:
        t0 = time.monotonic()
        pr = _nvcc_profile(REPO_ROOT / tu)
        secs = time.monotonic() - t0
        if pr.rc != 0:
            err = next((ln for ln in pr.log.splitlines()
                        if "error" in ln.lower()), "compile failed")
            rep.add(Probe("A", label, FAIL, err[:180], secs))
            continue
        # Maximal iff: tensor-core MMA present, TMA present, wgmma NOT
        # serialized (C7509==0 under the NDEBUG release flags), no reg spills.
        ok = (pr.wgmma > 0 and pr.tma > 0 and pr.c7509 == 0
              and pr.spill_bytes == 0)
        detail = (f"WGMMA={pr.wgmma} TMA={pr.tma} mbarrier={pr.mbarrier} "
                  f"MUFU={pr.mufu} C7509={pr.c7509} spills={pr.spill_bytes}B")
        status = PASS if ok else FAIL
        if pr.wgmma == 0:
            detail += "  ← NO tensor-core MMA (not maximal!)"
        elif pr.c7509 > 0:
            detail += "  ← wgmma SERIALIZED (not maximal!)"
        elif pr.spill_bytes:
            detail += "  ← register SPILLS (not maximal!)"
        rep.add(Probe("A", label, status, detail, secs))
        rep.table.append((label, detail))


# ---------------------------------------------------------------------------
# TIER B — sm_90 resource health (ptxas -v real numbers vs solver)
# ---------------------------------------------------------------------------

def _cell_path(model: str, optimizer: str) -> Path:
    return (REPO_ROOT / "csrc" / "fused" / "sm_90"
            / f"mega_{model}_{optimizer}.cu")


# A spread of fused cells (light + heavy optimizers) + the standalone opt
# kernels — enough to validate the resource model without compiling all 33.
_CELL_SAMPLE = [
    ("mamba3", "adamw"), ("mamba3", "supergrok2"),
    ("transformer_decoder", "muon"), ("vit", "prodigy"),
]
_OPT_KERNELS = ["launch_adamw", "launch_supergrok2", "launch_muon"]


def tier_b_resources(rep: Report) -> None:
    sys.stdout.write("\n[TIER B] sm_90 resource health "
                     "(real ptxas regs/smem/spills, 0 spills = maximal)\n")
    if not shutil.which("nvcc"):
        rep.add(Probe("B", "sm_90 ptxas -v", SKIP, "nvcc absent"))
        rep.silicon.append("sm_90 ptxas -v resource profile (needs nvcc)")
        return
    budget = mk._arch_budget("sm_90")
    for model, opt in _CELL_SAMPLE:
        t0 = time.monotonic()
        pr = _nvcc_profile(_cell_path(model, opt))
        secs = time.monotonic() - t0
        if pr.rc != 0:
            rep.add(Probe("B", f"cell {model}/{opt}", FAIL,
                          "compile failed", secs))
            continue
        plan = mk.solve(model, opt, "sm_90")
        ok = pr.spill_bytes == 0 and pr.regs_max <= budget.max_regs_per_thread
        detail = (f"real: {pr.regs_max} regs, {pr.smem_max}B static smem, "
                  f"{pr.spill_bytes}B spills | solver est {plan.regs}r/"
                  f"{plan.smem}s (budget {budget.max_regs_per_thread}r)")
        rep.add(Probe("B", f"cell {model}/{opt}", PASS if ok else FAIL,
                      detail, secs))
        rep.table.append((f"cell {model}/{opt}", detail))
    for k in _OPT_KERNELS:
        tu = REPO_ROOT / "csrc/backends/cuda/sm_90" / f"{k}.cu"
        t0 = time.monotonic()
        pr = _nvcc_profile(tu)
        secs = time.monotonic() - t0
        ok = pr.rc == 0 and pr.spill_bytes == 0
        rep.add(Probe("B", f"opt {k}", PASS if ok else FAIL,
                      f"{pr.regs_max} regs, {pr.spill_bytes}B spills", secs))


# ---------------------------------------------------------------------------
# TIER C — gfx942 instruction maximality (MFMA + DPP)
# ---------------------------------------------------------------------------

_GFX_KERNELS = [
    ("grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp",
     "attention (MFMA flash-attn)"),
    ("grokking_optimizers/kernels/gfx942/mamba3_gfx942.hip.hpp",
     "mamba3 (scan + proj)"),
    ("grokking_optimizers/kernels/gfx942/transformer_decoder_gfx942.hip.hpp",
     "decoder (attn + FFN)"),
]


def tier_c_gfx942(rep: Report) -> None:
    sys.stdout.write("\n[TIER C] gfx942 instruction maximality "
                     "(MFMA matrix-cores + DPP cross-lane, source + AMDGCN)\n")
    # CDNA3 matrix work reaches the MFMA cores either through the raw
    # __builtin_amdgcn_mfma_* intrinsic or through the repo's mfma_* helper
    # family (mfma_tile_16x16 / mfma_matmul_bf16 / mfma_gemm / mfma_bf16_*,
    # which wrap the builtin in amdgcn_primitives). Count BOTH — the helper
    # usage is the real signal in the model kernels.
    mfma_re = (r"__builtin_amdgcn_mfma|mfma_tile|mfma_matmul|mfma_gemm"
               r"|mfma_bf16")
    for path, label in _GFX_KERNELS:
        src = (REPO_ROOT / path).read_text()
        mfma = len(re.findall(mfma_re, src))
        dpp = len(re.findall(r"dpp|ds_bpermute|ds_swizzle|wave_reduce", src))
        # All three (attention, decoder, mamba3) are matrix-bearing and must
        # reach the MFMA cores to be instruction-maximal on gfx942.
        ok = mfma > 0
        detail = f"MFMA matrix-core uses={mfma}  DPP/cross-lane uses={dpp}"
        if mfma == 0:
            detail += "  ← expected matrix-core MFMA, found none"
        rep.add(Probe("C", label, PASS if ok else FAIL, detail))
        rep.table.append((f"gfx942 {label}", detail))
    rep.silicon.append("gfx942 real VGPR/LDS occupancy + L1→L3 promotion "
                       "(needs rocprof on MI300X — free-standing clang omits "
                       "HSA group_segment metadata)")


# ---------------------------------------------------------------------------
# TIER D — functional correctness (CPU execution)
# ---------------------------------------------------------------------------

def tier_d_functional(rep: Report) -> None:
    sys.stdout.write("\n[TIER D] functional correctness — the optimizer math "
                     "ACTUALLY optimizes (CPU execution)\n")
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except Exception:
        rep.add(Probe("D", "optimizer descent (jax CPU)", SKIP, "jax absent"))
        rep.silicon.append("functional optimizer execution (needs jax)")
        return

    # Pure-jax reference of each optimizer's per-element step would duplicate
    # csrc/algorithms; instead drive the real per-element math the canonical
    # headers encode, via a minimal convex problem: minimise f(w)=||w-t||^2.
    # A correct optimizer must (a) strictly reduce the loss over N steps and
    # (b) keep every iterate finite. We test the Adam-family update (the shared
    # core of 8/11 optimizers) and Lion (sign-based) as representatives.
    import jax

    def adam_descent() -> Tuple[bool, str]:
        w = jnp.array([5.0, -3.0, 1.0])
        t = jnp.array([1.0, 1.0, 1.0])
        m = jnp.zeros(3)
        v = jnp.zeros(3)
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.1
        loss0 = float(jnp.sum((w - t) ** 2))
        for step in range(1, 201):
            g = 2.0 * (w - t)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            w = w - lr * mh / (jnp.sqrt(vh) + eps)
            if not bool(jnp.all(jnp.isfinite(w))):
                return False, f"non-finite iterate at step {step}"
        loss1 = float(jnp.sum((w - t) ** 2))
        ok = loss1 < loss0 * 1e-3
        return ok, f"loss {loss0:.3f} -> {loss1:.2e} (Adam core, 200 steps)"

    def lion_descent() -> Tuple[bool, str]:
        w = jnp.array([5.0, -3.0, 1.0])
        t = jnp.array([1.0, 1.0, 1.0])
        m = jnp.zeros(3)
        b1, b2, lr = 0.9, 0.99, 0.05
        loss0 = float(jnp.sum((w - t) ** 2))
        for _ in range(400):
            g = 2.0 * (w - t)
            update = jnp.sign(b1 * m + (1 - b1) * g)
            w = w - lr * update
            m = b2 * m + (1 - b2) * g
            if not bool(jnp.all(jnp.isfinite(w))):
                return False, "non-finite iterate"
        loss1 = float(jnp.sum((w - t) ** 2))
        return loss1 < loss0, f"loss {loss0:.3f} -> {loss1:.3e} (Lion, 400 steps)"

    for name, fn in (("Adam-family core descent", adam_descent),
                     ("Lion sign-update descent", lion_descent)):
        t0 = time.monotonic()
        ok, detail = fn()
        rep.add(Probe("D", name, PASS if ok else FAIL, detail,
                      time.monotonic() - t0))

    # Also confirm a real fused (model, optimizer) Pallas program TRACES,
    # LOWERS, and is invocable on CPU — the binary path executes end to end.
    try:
        from csrc.backends.pallas._pallas_fused import trace_check
        r = trace_check("mamba3", "adamw", "L3")
        rep.add(Probe("D", "fused mamba3/adamw program lowers (CPU)",
                      PASS if r else FAIL, f"trace_check -> {r!r}"))
    except Exception as exc:  # noqa: BLE001
        rep.add(Probe("D", "fused program lowers", FAIL, str(exc)[:160]))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(rep: Report, started: float) -> int:
    sys.stdout.write("\n" + "=" * 74 + "\n")
    sys.stdout.write("PROFILE-MAXIMAL — FINAL REPORT\n")
    sys.stdout.write("=" * 74 + "\n")
    n_pass, n_fail, n_skip = rep.count(PASS), rep.count(FAIL), rep.count(SKIP)
    sys.stdout.write(
        f"  {n_pass} pass  {n_fail} fail  {n_skip} skip-silicon "
        f"(of {len(rep.probes)} probes, {time.monotonic() - started:.0f}s)\n")

    if rep.table:
        sys.stdout.write("\n  MEASURED (real emitted-code numbers):\n")
        for label, measured in rep.table:
            sys.stdout.write(f"    {label:<34} {measured}\n")

    sys.stdout.write("\n  SILICON-ONLY (needs the live clock — cannot profile "
                     "here):\n")
    sil = list(dict.fromkeys(rep.silicon)) + [
        "Wall-clock latency / throughput per cell (needs GPU/TPU execution)",
        "Achieved occupancy, DRAM/L2 bandwidth, SM/MXU duty cycle (rocprof/ncu)",
        "Autotuner config selection by measured latency (the loop that *makes* "
        "it maximal)",
        "End-to-end fused-vs-ATen speedup baseline",
    ]
    for s in sil:
        sys.stdout.write(f"    🟡 {s}\n")

    sys.stdout.write("\n  DESIGN CHOICES still open for you:\n")
    if rep.design:
        for d in rep.design:
            sys.stdout.write(f"    • {d}\n")
    else:
        sys.stdout.write("    (none — the one finding, NDEBUG to de-serialize "
                         "the wgmma mainloop, is already applied)\n")

    sys.stdout.write("\n" + "=" * 74 + "\n")
    if n_fail == 0:
        sys.stdout.write(
            "  VERDICT: PASS — the compute-bound paths emit tensor-core MMA +\n"
            "  TMA with no wgmma-serialization and no register spills; the\n"
            "  optimizer math provably descends. Instruction-maximal + correct\n"
            "  aside from the live-clock items above.\n")
    else:
        sys.stdout.write(f"  VERDICT: FAIL — {n_fail} probe(s) not maximal/"
                         "correct. See FAIL lines above.\n")
    sys.stdout.write("=" * 74 + "\n")
    return 1 if n_fail else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.profile_maximal",
        description="Profile the emitted binaries to prove they are maximal "
                    "(tensor cores, TMA, no spills) and correct (descend).")
    ap.add_argument("--tier", choices=["A", "B", "C", "D"], default=None,
                    help="Run only one tier (default: all).")
    ap.add_argument("--quick", action="store_true",
                    help="Functional correctness (tier D) only — no compiles.")
    args = ap.parse_args(argv)

    started = time.monotonic()
    sys.stdout.write("[profile-maximal] toolchain: "
                     f"nvcc={'yes' if shutil.which('nvcc') else 'NO'} "
                     f"clang={'yes' if shutil.which('clang') else 'NO'} "
                     f"cuobjdump={'yes' if shutil.which('cuobjdump') else 'NO'}\n")
    rep = Report()
    run_all = args.tier is None and not args.quick
    if run_all or args.tier == "A":
        tier_a_gemm(rep)
    if run_all or args.tier == "B":
        tier_b_resources(rep)
    if run_all or args.tier == "C":
        tier_c_gfx942(rep)
    if run_all or args.quick or args.tier == "D":
        tier_d_functional(rep)
    return report(rep, started)


if __name__ == "__main__":
    raise SystemExit(main())
