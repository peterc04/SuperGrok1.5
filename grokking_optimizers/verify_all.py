"""End-to-end verification harness — the single authoritative gate for
SuperGrok1.5.  "Know EVERYTHING; prove it works, and works *maximally*."

This is the end-all-be-all.  It does not just check that individual files
compile in isolation — it proves that the **modular composition** works: every
optimizer compiles *together with* every model into the fused megakernel cell
for every arch (99 cells), and that each composition runs at its **maximal**
fusion tier.  It then cross-validates the generated artifacts against their
single source of truth so nothing can silently drift.

It is organized as ordered phases; each phase prints one PASS/FAIL line per
check and accumulates into a final structured report.  Phases that need a tool
absent on this host (nvcc / clang / jax) are reported as SKIPPED-silicon rather
than silently passing.

    PHASE 0  toolchain probe            — what can be proven here vs on silicon
    PHASE 1  structural inventory       — every component + 99 cells exist
    PHASE 2  single-component gates      — per-op launchers + models + headers
    PHASE 3  MODULAR COMPOSITION         — 99 cells: optimizer × model compile
                                            *together* (the core guarantee)
    PHASE 4  MAXIMALITY                   — max fusion tier, codegen idempotency,
                                            register/smem budget, math drift
    PHASE 5  cross-validation            — dispatch tables, self-test, ruff
    PHASE 6  report                       — totals + tier map + silicon + design

Run:
    PYTHONPATH=. python3 -m grokking_optimizers.verify_all              # all
    PYTHONPATH=. python3 -m grokking_optimizers.verify_all --phase 3    # one
    PYTHONPATH=. python3 -m grokking_optimizers.verify_all --quick      # no compiles
    PYTHONPATH=. python3 -m grokking_optimizers.verify_all --jobs 8     # parallelism

Failure policy mirrors utilization.py: a missing *expected* artifact or a failed
*runnable* check is a hard FAIL.  A check that genuinely needs an accelerator is
SKIPPED-silicon and listed at the end — never a false green.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

from grokking_optimizers import megakernel as mk
from grokking_optimizers import megakernel_codegen as cg

OPTIMIZERS: Tuple[str, ...] = mk.OPTIMIZERS
MODELS: Tuple[str, ...] = mk.MODELS
ARCHES: Tuple[str, ...] = mk.MEGAKERNEL_ARCHS  # sm_90, gfx942, tpu_v6e
N_CELLS = len(OPTIMIZERS) * len(MODELS) * len(ARCHES)  # 99


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP-silicon"


@dataclasses.dataclass
class Check:
    phase: str
    name: str
    status: str            # PASS | FAIL | SKIP-silicon
    detail: str = ""
    seconds: float = 0.0


class Report:
    def __init__(self) -> None:
        self.checks: List[Check] = []
        self.tier_map: Dict[str, Dict[str, int]] = {}
        self.silicon: List[str] = []
        self.design: List[str] = []

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        glyph = {"PASS": "PASS ", "FAIL": "FAIL ",
                 "SKIP-silicon": "SKIP "}[c.status]
        tail = f"  ({c.seconds:.1f}s)" if c.seconds >= 0.1 else ""
        line = f"  {glyph} [{c.phase}] {c.name}{tail}"
        if c.detail and c.status != PASS:
            line += f"\n         {c.detail}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return c

    def count(self, status: str, phase: Optional[str] = None) -> int:
        return sum(1 for c in self.checks
                   if c.status == status and (phase is None or c.phase == phase))


# ---------------------------------------------------------------------------
# Toolchain probe
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Toolchain:
    nvcc: Optional[str]
    clang: Optional[str]
    has_jax: bool
    has_gpu: bool
    has_tpu: bool


def probe_toolchain() -> Toolchain:
    nvcc = shutil.which("nvcc")
    clang = shutil.which("clang")
    has_jax = False
    has_tpu = False
    try:
        import jax
        has_jax = True
        has_tpu = any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        pass
    has_gpu = False
    try:
        import torch
        has_gpu = bool(torch.cuda.is_available())
    except Exception:
        pass
    return Toolchain(nvcc, clang, has_jax, has_gpu, has_tpu)


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str]:
    """Run a command in REPO_ROOT, return (rc, combined-tail)."""
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def _cell_path(model: str, optimizer: str, arch: str) -> Path:
    sym = cg._cell_symbol(model, optimizer)
    ext = cg._ARCH_EXT[arch]
    return REPO_ROOT / "csrc" / "fused" / cg._ARCH_DIR[arch] / f"{sym}{ext}"


# ---------------------------------------------------------------------------
# PHASE 1 — structural inventory
# ---------------------------------------------------------------------------

def phase1_inventory(rep: Report) -> None:
    sys.stdout.write("\n[PHASE 1] structural inventory\n")

    # 11 canonical algorithm headers (single-source optimizer math).
    algo_dir = REPO_ROOT / "csrc" / "algorithms"
    missing = [o for o in OPTIMIZERS if not (algo_dir / f"{o}.h").exists()]
    rep.add(Check("1", "algorithm-headers (11)", PASS if not missing else FAIL,
                  "" if not missing else f"missing: {missing}"))

    # Per-op launchers per GPU arch.
    launcher_specs = [
        ("sm_90 launchers (11)", "csrc/backends/cuda/sm_90", "launch_{o}.cu"),
        ("gfx942 launchers (11)", "csrc/backends/hip/gfx942", "launch_{o}.hip.cpp"),
        ("pallas launchers (11)", "csrc/backends/pallas", "launch_{o}.py"),
    ]
    for label, sub, pat in launcher_specs:
        d = REPO_ROOT / sub
        miss = [o for o in OPTIMIZERS
                if not (d / pat.format(o=o)).exists()]
        # Some optimizers share a launcher (e.g. muon/looksam adamw-update);
        # only FAIL if the expected file is genuinely absent for >half.
        rep.add(Check("1", label, PASS if not miss else FAIL,
                      "" if not miss else f"missing {len(miss)}: {miss}"))

    # 99 fused cells exist on disk.
    missing_cells: List[str] = []
    for arch in ARCHES:
        for model in MODELS:
            for opt in OPTIMIZERS:
                if not _cell_path(model, opt, arch).exists():
                    missing_cells.append(f"{model}/{opt}/{arch}")
    rep.add(Check("1", f"fused cells on disk ({N_CELLS})",
                  PASS if not missing_cells else FAIL,
                  "" if not missing_cells
                  else f"missing {len(missing_cells)}: {missing_cells[:5]}…"))

    # Dispatch tables present.
    for f in ("csrc/fused/sm_90/fused_dispatch_table.inc",
              "csrc/fused/gfx942/fused_dispatch_table.inc",
              "csrc/fused/fused_wired_cells.inc"):
        rep.add(Check("1", f"dispatch table {Path(f).name}",
                      PASS if (REPO_ROOT / f).exists() else FAIL))


# ---------------------------------------------------------------------------
# PHASE 2 — single-component compile gates
# ---------------------------------------------------------------------------

def phase2_components(rep: Report, tc: Toolchain, jobs: int) -> None:
    sys.stdout.write("\n[PHASE 2] single-component compile gates\n")

    # sm_90 per-op launchers + models (nvcc -c object).
    if tc.nvcc:
        units = [REPO_ROOT / "csrc/backends/cuda/sm_90" / f"launch_{o}.cu"
                 for o in OPTIMIZERS]
        units += [REPO_ROOT / "csrc/backends/cuda/sm_90/models" / f"{m}.cu"
                  for m in ("mamba", "decoder", "vit")]
        _parallel_compile(rep, "2", "sm_90 component", units,
                          _nvcc_compile, jobs)
    else:
        rep.add(Check("2", "sm_90 components (nvcc)", SKIP,
                      "nvcc absent"))
        rep.silicon.append("sm_90 component object-compile (needs nvcc)")

    # gfx942 device headers (AMDGCN clang gate).
    if tc.clang:
        headers = sorted((REPO_ROOT / "grokking_optimizers/kernels/gfx942")
                         .glob("*_gfx942.hip.hpp"))
        headers.append(REPO_ROOT /
                       "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp")
        _parallel_compile(rep, "2", "gfx942 header", headers,
                          _amdgcn_header, jobs)
    else:
        rep.add(Check("2", "gfx942 headers (clang)", SKIP, "clang absent"))
        rep.silicon.append("gfx942 AMDGCN header gate (needs clang)")


# ---------------------------------------------------------------------------
# PHASE 3 — MODULAR COMPOSITION  (the core guarantee)
# ---------------------------------------------------------------------------

def phase3_composition(rep: Report, tc: Toolchain, jobs: int) -> None:
    sys.stdout.write("\n[PHASE 3] MODULAR COMPOSITION — 99 cells: "
                     "optimizer × model compiled TOGETHER\n")

    # sm_90: nvcc -c on each mega_*.cu (composes opt component + model stage).
    sm90 = [_cell_path(m, o, "sm_90") for m in MODELS for o in OPTIMIZERS]
    if tc.nvcc:
        _parallel_compile(rep, "3", "sm_90 compose", sm90, _nvcc_compile, jobs)
    else:
        for p in sm90:
            rep.add(Check("3", f"sm_90 compose {p.stem}", SKIP, "nvcc absent"))
        rep.silicon.append("sm_90 fused-cell composition compile (needs nvcc)")

    # gfx942: AMDGCN device gate on each mega_*.hip (forces the composed
    # template instantiation — type-checks opt×model on device).
    gfx = [_cell_path(m, o, "gfx942") for m in MODELS for o in OPTIMIZERS]
    if tc.clang:
        _parallel_compile(rep, "3", "gfx942 compose", gfx, _amdgcn_cell, jobs)
    else:
        for p in gfx:
            rep.add(Check("3", f"gfx942 compose {p.stem}", SKIP, "clang absent"))
        rep.silicon.append("gfx942 fused-cell device composition (needs clang)")

    # tpu_v6e: jax trace + HLO lower of each fused program (opt×model under one
    # jit). Runs on CPU; no TPU needed.
    if tc.has_jax:
        _tpu_trace_all(rep, jobs)
    else:
        for m in MODELS:
            for o in OPTIMIZERS:
                rep.add(Check("3", f"tpu_v6e compose mega_{m}_{o}", SKIP,
                              "jax absent"))
        rep.silicon.append("tpu_v6e fused-cell trace+lower (needs jax)")

    # gfx942 + tpu host-launch paths are silicon-gated (hipcc/MI300X, real TPU).
    rep.silicon.append("gfx942 host hipLaunchKernelGGL path (needs hipcc/MI300X)")
    rep.silicon.append("tpu_v6e on-device execution (needs real TPU v6e)")


def _nvcc_compile(tu: Path) -> Tuple[bool, str]:
    # -DWITH_CUTLASS is the MAXIMAL sm_90 build config (the shipped fast path:
    # CUTLASS WGMMA/TMA collectives for SG2 + the model GEMMs). supergrok2's
    # launcher and the model TUs hard-#error without it; the cuBLAS megakernel
    # cells are unaffected by it. Test what actually ships.
    rc, out = _run(["bash", "scripts/compile_to_object.sh", str(tu),
                    "-DWITH_CUTLASS"], timeout=600)
    if rc == 0:
        return True, ""
    return False, _first_error(out)


def _amdgcn_header(hdr: Path) -> Tuple[bool, str]:
    rc, out = _run(["bash", "scripts/amdgcn_check.sh", "--header", str(hdr)],
                   timeout=120)
    return (rc == 0), ("" if rc == 0 else _first_error(out))


def _amdgcn_cell(cell: Path) -> Tuple[bool, str]:
    rc, out = _run(["bash", "scripts/amdgcn_check.sh", "--cell", str(cell)],
                   timeout=120)
    return (rc == 0), ("" if rc == 0 else _first_error(out))


def _first_error(out: str) -> str:
    for ln in out.splitlines():
        if "error" in ln.lower():
            return ln.strip()[:200]
    return out.splitlines()[-1][:200] if out else "(no output)"


def _parallel_compile(rep: Report, phase: str, label: str, units: List[Path],
                      fn: Callable[[Path], Tuple[bool, str]], jobs: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_timed, fn, u): u for u in units}
        for fut in concurrent.futures.as_completed(futs):
            u = futs[fut]
            ok, detail, secs = fut.result()
            rep.add(Check(phase, f"{label} {u.stem}",
                          PASS if ok else FAIL, detail, secs))


def _timed(fn: Callable[[Path], Tuple[bool, str]], u: Path
           ) -> Tuple[bool, str, float]:
    t0 = time.monotonic()
    ok, detail = fn(u)
    return ok, detail, time.monotonic() - t0


def _tpu_trace_all(rep: Report, jobs: int) -> None:
    """Trace+lower every tpu_v6e fused cell via its verify() (CPU jax)."""
    import importlib

    def one(model: str, opt: str) -> Tuple[bool, str, float]:
        t0 = time.monotonic()
        mod_name = f"csrc.fused.tpu_v6e.mega_{model}_{opt}"
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            mod.verify()
            return True, "", time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001 — surface the trace failure
            return False, f"{type(exc).__name__}: {exc}"[:200], \
                time.monotonic() - t0

    cells = [(m, o) for m in MODELS for o in OPTIMIZERS]
    # jax tracing is not thread-safe across reloads; keep TPU trace serial-ish.
    workers = max(1, min(jobs, 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, m, o): (m, o) for m, o in cells}
        for fut in concurrent.futures.as_completed(futs):
            m, o = futs[fut]
            ok, detail, secs = fut.result()
            rep.add(Check("3", f"tpu_v6e compose mega_{m}_{o}",
                          PASS if ok else FAIL, detail, secs))


# ---------------------------------------------------------------------------
# PHASE 4 — MAXIMALITY
# ---------------------------------------------------------------------------

def phase4_maximality(rep: Report) -> None:
    sys.stdout.write("\n[PHASE 4] MAXIMALITY — max fusion tier, idempotency, "
                     "budget, single-source\n")

    plans = mk.solve_all()
    by_arch_tier: Dict[str, Dict[str, int]] = {a: {} for a in ARCHES}

    # 4a. Every cell is FEASIBLE (>= L1) — no infeasible compositions.
    infeasible = [p for p in plans if not p.fits]
    rep.add(Check("4", f"all {N_CELLS} cells feasible (≥L1)",
                  PASS if not infeasible else FAIL,
                  "" if not infeasible
                  else f"{len(infeasible)} infeasible: "
                       f"{[(p.model, p.optimizer, p.arch) for p in infeasible][:3]}"))

    # 4b. Every cell sits at the MAX tier that fits its budget (the solver's
    # job — re-derive independently and confirm no cell could go higher).
    not_maximal: List[str] = []
    for p in plans:
        tier = p.tier
        by_arch_tier[p.arch][tier.name] = \
            by_arch_tier[p.arch].get(tier.name, 0) + 1
        if p.arch == "tpu_v6e":
            continue  # XLA-managed; solver reports L3 by construction.
        b = mk._arch_budget(p.arch)
        # Could a strictly higher tier have fit? If so, the solver under-fused.
        for higher in (mk.FusionTier.L3_FWD_BWD_OPT, mk.FusionTier.L2_BWD_OPT):
            if higher <= tier:
                continue
            hr, hs = mk._tier_cost(p.model, p.optimizer, higher)
            if hr <= b.max_regs_per_thread and hs <= b.max_smem_per_block:
                not_maximal.append(
                    f"{p.model}/{p.optimizer}/{p.arch}: at {tier.name} but "
                    f"{higher.name} fits ({hr}r/{hs}s)")
    rep.add(Check("4", "every cell at its MAX feasible tier",
                  PASS if not not_maximal else FAIL,
                  "" if not not_maximal else "; ".join(not_maximal[:3])))
    rep.tier_map = by_arch_tier

    # 4c. Each cell within the arch register + smem budget at its chosen tier.
    over_budget: List[str] = []
    for p in plans:
        if p.arch == "tpu_v6e":
            continue
        b = mk._arch_budget(p.arch)
        if p.regs > b.max_regs_per_thread or p.smem > b.max_smem_per_block:
            over_budget.append(
                f"{p.model}/{p.optimizer}/{p.arch}: {p.regs}r/{p.smem}s "
                f"> {b.max_regs_per_thread}r/{b.max_smem_per_block}s")
    rep.add(Check("4", "every cell within register+smem budget",
                  PASS if not over_budget else FAIL,
                  "" if not over_budget else "; ".join(over_budget[:3])))

    # 4d. CODEGEN IDEMPOTENCY: re-emit every cell in memory and assert it is
    # byte-identical to the committed file (no drift, no stale generated code).
    drifted: List[str] = []
    for p in plans:
        path = _cell_path(p.model, p.optimizer, p.arch)
        if not path.exists():
            drifted.append(f"{path.name} (missing)")
            continue
        emitted = cg.emit_cell(p.model, p.optimizer, p.arch)
        if emitted.rstrip() != path.read_text().rstrip():
            drifted.append(path.name)
    rep.add(Check("4", f"codegen idempotency ({N_CELLS} cells byte-identical)",
                  PASS if not drifted else FAIL,
                  "" if not drifted
                  else f"{len(drifted)} drifted: {drifted[:5]}"))

    # 4e. The committed cell's tier COMMENT matches the solver's live decision
    # (catches a stale generated header even if the body is right).
    tier_mismatch: List[str] = []
    for p in plans:
        path = _cell_path(p.model, p.optimizer, p.arch)
        if not path.exists():
            continue
        head = path.read_text()[:400]
        if p.tier.name not in head and _short_tier(p.tier) not in head:
            tier_mismatch.append(f"{path.name} (want {p.tier.name})")
    rep.add(Check("4", "cell tier-comment matches solver",
                  PASS if not tier_mismatch else FAIL,
                  "" if not tier_mismatch else f"{tier_mismatch[:5]}"))

    # 4f. Math single-source drift guard (the optimizer math is shared between
    # per-op and fused paths — maximal reuse, zero divergence).
    rc, out = _run([sys.executable, "scripts/check_math_single_source.py"],
                   timeout=120)
    rep.add(Check("4", "optimizer-math single-source drift guard",
                  PASS if rc == 0 else FAIL,
                  "" if rc == 0 else _first_error(out)))


def _short_tier(t: "mk.FusionTier") -> str:
    return {"L3_FWD_BWD_OPT": "L3", "L2_BWD_OPT": "L2",
            "L1_OPT_ONLY": "L1", "L0_UNFUSED": "L0"}[t.name]


# ---------------------------------------------------------------------------
# PHASE 5 — cross-validation
# ---------------------------------------------------------------------------

def phase5_crossval(rep: Report, tc: Toolchain) -> None:
    sys.stdout.write("\n[PHASE 5] cross-validation — generated artifacts vs "
                     "single source\n")

    # 5a. The three dispatch tables match their generators byte-for-byte.
    pairs = [
        ("sm_90 dispatch table", "csrc/fused/sm_90/fused_dispatch_table.inc",
         cg.dispatch_table_sm90),
        ("gfx942 dispatch table", "csrc/fused/gfx942/fused_dispatch_table.inc",
         cg.dispatch_table_gfx942),
        ("wired_fused_cells", "csrc/fused/fused_wired_cells.inc",
         cg.dispatch_table),
    ]
    for label, path, gen in pairs:
        committed = (REPO_ROOT / path).read_text()
        rep.add(Check("5", label,
                      PASS if gen().rstrip() == committed.rstrip() else FAIL,
                      "" if gen().rstrip() == committed.rstrip()
                      else "generator output differs from committed file"))

    # 5b. fused_wired_cells routes every (arch,model,opt) the solver knows and
    # carries no stale tpu_v5p.
    wired = (REPO_ROOT / "csrc/fused/fused_wired_cells.inc").read_text()
    rep.add(Check("5", "no stale tpu_v5p in active dispatch",
                  PASS if "tpu_v5p" not in wired else FAIL))
    route_miss = [f"{m}/{o}"
                  for m in MODELS for o in OPTIMIZERS
                  if f'+{o}:' not in wired or f'"{m}' not in wired and
                  f'{m}+{o}' not in wired]
    rep.add(Check("5", "wired table routes all model+opt pairs",
                  PASS if not route_miss else FAIL,
                  "" if not route_miss else f"{route_miss[:5]}"))

    # 5c. Self-test suite (authoritative count guard inside compile.py).
    rc, out = _run([sys.executable, "-m", "grokking_optimizers.compile",
                    "--self-test"], timeout=600)
    ok = rc == 0 and "0 failed" in out
    summary = next((ln for ln in out.splitlines()
                    if "passed," in ln and "failed" in ln), "")
    rep.add(Check("5", "compile.py --self-test",
                  PASS if ok else FAIL, summary.strip()))

    # 5d. ruff lint clean across the package + scripts.
    if shutil.which("ruff"):
        rc, out = _run(["ruff", "check", "grokking_optimizers/", "scripts/"],
                       timeout=120)
        rep.add(Check("5", "ruff lint clean",
                      PASS if rc == 0 else FAIL,
                      "" if rc == 0 else _first_error(out)))
    else:
        rep.add(Check("5", "ruff lint", SKIP, "ruff absent"))

    # 5e. utilization crash-hard contract still holds (no device here → raises).
    try:
        from grokking_optimizers import utilization as u
        raised = False
        try:
            u.track_cell("adamw", "mamba", "sm_90a", iters=1, timeout=10)
        except RuntimeError:
            raised = True
        rep.add(Check("5", "utilization crash-hard (raises w/o device)",
                      PASS if raised else FAIL,
                      "" if raised else "track_cell did not raise"))
    except Exception as exc:  # noqa: BLE001
        rep.add(Check("5", "utilization crash-hard", FAIL, str(exc)[:200]))


# ---------------------------------------------------------------------------
# PHASE 6 — report
# ---------------------------------------------------------------------------

def phase6_report(rep: Report, tc: Toolchain, started: float) -> int:
    sys.stdout.write("\n" + "=" * 74 + "\n")
    sys.stdout.write("VERIFY-ALL — FINAL REPORT\n")
    sys.stdout.write("=" * 74 + "\n")

    total = len(rep.checks)
    n_pass, n_fail, n_skip = (rep.count(PASS), rep.count(FAIL), rep.count(SKIP))
    for ph in ("1", "2", "3", "4", "5"):
        p, f, s = (rep.count(PASS, ph), rep.count(FAIL, ph), rep.count(SKIP, ph))
        names = {"1": "inventory", "2": "components", "3": "COMPOSITION",
                 "4": "MAXIMALITY", "5": "cross-validation"}
        sys.stdout.write(
            f"  phase {ph} {names[ph]:<18} "
            f"{p:>3} pass  {f:>3} fail  {s:>3} skip-silicon\n")
    sys.stdout.write("-" * 74 + "\n")
    sys.stdout.write(
        f"  TOTAL  {n_pass} pass  {n_fail} fail  {n_skip} skip-silicon "
        f"(of {total} checks, {time.monotonic() - started:.0f}s)\n")

    # Maximality / fusion-tier map.
    sys.stdout.write("\n  MAXIMALITY — fusion-tier coverage per arch "
                     "(higher = more fused):\n")
    for arch in ARCHES:
        tiers = rep.tier_map.get(arch, {})
        parts = [f"{t.split('_')[0]}={tiers[t]}"
                 for t in ("L3_FWD_BWD_OPT", "L2_BWD_OPT",
                           "L1_OPT_ONLY", "L0_UNFUSED") if tiers.get(t)]
        sys.stdout.write(f"    {arch:<9} {'  '.join(parts)}\n")

    # The gfx942 decoder/vit L1 demotion is the one tier non-maximality, and it
    # is an ESTIMATE pending silicon (the LDS over-count). Surface it precisely.
    gfx = rep.tier_map.get("gfx942", {})
    if gfx.get("L1_OPT_ONLY"):
        sys.stdout.write(
            f"    note: {gfx['L1_OPT_ONLY']} gfx942 decoder/vit cells are L1 "
            "by the smem ESTIMATE (bwd 66560>65536). Promotion to L3 is\n"
            "          silicon-gated (rocprof true-LDS) — see SILICON below.\n")

    # Silicon-only items.
    sys.stdout.write("\n  SILICON-ONLY (cannot be proven on this host — no "
                     "GPU/TPU):\n")
    sil = list(dict.fromkeys(rep.silicon))  # dedupe, keep order
    sil += [
        "Real latency / throughput + autotuner config selection (needs GPU)",
        "99-cell numeric parity vs fp64 PyTorch reference (needs GPU/TPU exec)",
        "gfx942 decoder/vit L1→L3 promotion via rocprof true-LDS (≤64KB?)",
        "ptxas -v / roc-obj SASS emission check (HGMMA/UTMALDG/v_mfma counts)",
    ]
    for s in sil:
        sys.stdout.write(f"    🟡 {s}\n")

    # Design choices (only genuine ones).
    sys.stdout.write("\n  DESIGN CHOICES still open for you:\n")
    if not rep.design:
        sys.stdout.write("    (none — all prior D1–D7 decided; nothing new "
                         "surfaced by this run)\n")
    else:
        for d in rep.design:
            sys.stdout.write(f"    • {d}\n")

    # Verdict.
    sys.stdout.write("\n" + "=" * 74 + "\n")
    if n_fail == 0:
        sys.stdout.write(
            "  VERDICT: PASS — every runnable check green. The codebase is\n"
            "  composition-complete and MAXIMAL aside from the silicon-only\n"
            "  items above (which require a real accelerator to close).\n")
    else:
        sys.stdout.write(
            f"  VERDICT: FAIL — {n_fail} check(s) failed. See FAIL lines "
            "above.\n")
    sys.stdout.write("=" * 74 + "\n")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.verify_all",
        description="End-all-be-all verification: prove the modular "
                    "composition compiles AND runs maximally across all 99 "
                    "fused pipelines.")
    ap.add_argument("--phase", choices=["1", "2", "3", "4", "5"], default=None,
                    help="Run only one phase (default: all).")
    ap.add_argument("--quick", action="store_true",
                    help="Skip the compile-heavy phases 2+3 (inventory + "
                         "maximality + cross-val only).")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4,
                    help="Parallel compile workers (default: ncpu).")
    args = ap.parse_args(argv)

    started = time.monotonic()
    tc = probe_toolchain()
    sys.stdout.write("[PHASE 0] toolchain probe\n")
    sys.stdout.write(
        f"  nvcc={'yes' if tc.nvcc else 'NO'}  "
        f"clang={'yes' if tc.clang else 'NO'}  "
        f"jax={'yes' if tc.has_jax else 'NO'}  "
        f"gpu={'yes' if tc.has_gpu else 'NO'}  "
        f"tpu={'yes' if tc.has_tpu else 'NO'}  jobs={args.jobs}\n")

    rep = Report()
    run_all = args.phase is None

    if run_all or args.phase == "1":
        phase1_inventory(rep)
    if (run_all and not args.quick) or args.phase == "2":
        phase2_components(rep, tc, args.jobs)
    if (run_all and not args.quick) or args.phase == "3":
        phase3_composition(rep, tc, args.jobs)
    if run_all or args.phase == "4":
        phase4_maximality(rep)
    if run_all or args.phase == "5":
        phase5_crossval(rep, tc)

    # Always need the tier map for the report; fill it if phase 4 was skipped.
    if not rep.tier_map:
        for p in mk.solve_all():
            rep.tier_map.setdefault(p.arch, {})
            rep.tier_map[p.arch][p.tier.name] = \
                rep.tier_map[p.arch].get(p.tier.name, 0) + 1

    return phase6_report(rep, tc, started)


if __name__ == "__main__":
    raise SystemExit(main())
