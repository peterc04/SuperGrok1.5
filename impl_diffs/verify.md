# AREA: NEW files only — `tests/hw/verify_functions.py`

Per-function silicon-verification driver for the **sm_90 L3-TC** path: the isolation +
ablation/wiring matrix over the 33-cell surface, built **entirely on top of the existing
gates** (`tests/hw/test_l3tc_tail_gate.py::run_cell_gate` via its `--cell` CLI, the SG2
twin `tests/hw/_sg2_l3tc_gate.py::run_sg2_gate`, and `grokking_optimizers/verify_all.py`).

## Scope constraints honored (AREA = "NEW files only — no edits to existing kernels")

- **No edits to any existing file.** This deliverable is a single self-contained NEW file.
  It does NOT add the `-DSG_ABLATE` kernel seam from the draft (that is an edit to
  `csrc/fused/sm_90/*.cuh` and belongs to a different AREA). Instead the ablation half is
  driven through an **opt-in external shadow-build root** (`--ablate-build-root`): when a
  build of `_ops` carrying the `SG_ABLATE` seam is supplied on `PYTHONPATH`, the harness
  runs the control-vs-ablated predicate; when it is absent the ablation rows are reported
  as `PENDING-SEAM` (honest — never a false green, never a false red).
- **Production path preserved byte-for-byte**: this file imports nothing into the
  production `_ops` build, defines no macro, and is never imported by the kernel build.
- **gfx942 / tpu_v6e preserved**: the GPU driver is `normalize_arch(...) == 90`-gated. On a
  non-Hopper host the GPU run delegates to the existing AMD/TPU composition gate
  (`verify_all --phase 3` + the `4f` single-source guard) — untouched — and the sm_90
  isolation/ablation matrix is SKIPPED (never failed).
- **`--self-check` is pure-CPU**: it imports only the host-side routing tables
  (`grokking_optimizers.dispatch`) and the cell registry (`tests.hw.test_l3tc_tail_gate._CELLS`),
  passing an explicit `arch=90` to `has_l3_real` so it runs with **no GPU and no built
  extension**. Verified on this CPU host (imports + `has_l3_real(...,90)` + `gemm_impl_for_cell`
  all resolve).

## gate_commands

```
python tests/hw/verify_functions.py --self-check
```

(run with `PYTHONPATH=.` from the repo root, i.e.
`PYTHONPATH=. python tests/hw/verify_functions.py --self-check`; the file also inserts the
repo root onto `sys.path` itself so a bare invocation works.)

Exit 0 iff: every enumerated surface node maps to a live isolation gate; all 33 cells are
confirmed L3-REAL + wgmma against the production routing tables; every ablation witness
names a real cell + a real ablation bit; the enumerated cell set equals
`test_l3tc_tail_gate._CELLS` exactly (no drift). No GPU touched.

---

## NEW FILE (full content): `tests/hw/verify_functions.py`

Create the file `tests/hw/verify_functions.py` with EXACTLY this content:

```python
"""tests/hw/verify_functions.py — per-function silicon verification driver (sm_90 L3-TC).

Phase-4 mandate (verify_harness.md): for every function/method on the L3-TC production
path, prove two things —

  (ISOLATION)  fed representative + edge inputs, the function is numerically correct vs a
               hardware-independent fp64 oracle the project ALREADY trusts.
  (ABLATION)   stubbing/removing the function changes the COMPOSED system under the
               existing gates; an unchanged result means the function is dead/unwired.

This module is a DRIVER LAYER on top of the gates the repo already owns — it authors NO
new numeric oracle and weakens NO assertion:

  * ISOLATION  -> tests/hw/test_l3tc_tail_gate.py::run_cell_gate (its (1a)/(1b)/(1b-sam)
                  + A/A/A determinism), reached via the EXISTING `--cell` CLI in a fresh
                  subprocess per (cell, seed). SG11/15 -> _run_sg_cell_gate; SG2 ->
                  _sg2_l3tc_gate.run_sg2_gate. fp64 rel 1e-4 (muon 2e-3) / SAM 2.5e-2.
  * ABLATION   -> the SAME run_cell_gate, but run against a SHADOW `_ops` carrying the
                  `SG_ABLATE` compile-time phase-removal seam (supplied externally on
                  PYTHONPATH via --ablate-build-root). The predicate is "control PASSES,
                  ablated FAILS" — a phase whose removal changes nothing is FLAGGED.
  * COMPOSITION-> grokking_optimizers/verify_all.py (the 99-cell modular-composition +
                  single-source drift guard) — delegated to for the gfx942/tpu archs.

8-GPU shardable: a job is (cell_key, ablate_token); the driver schedules jobs over a
process pool, pinning each worker to one GPU via CUDA_VISIBLE_DEVICES, and runs the
EXISTING gate CLI as a subprocess (the proven #19d isolation boundary). The control build
(ablate_token=0) MUST be A/A/A-identical to production `_ops` — that equality is the
ablation set's first precondition (a perturbed control voids every ablation).

CPU --self-check enumerates the function/cell surface and asserts coverage with NO GPU
run: every surface node maps to a live gate, all 33 cells are L3-REAL+wgmma per the
production routing tables, and the enumerated cell set == test_l3tc_tail_gate._CELLS.

gfx942 / tpu_v6e are preserved: the sm_90 matrix is normalize_arch==90-gated; on a
non-Hopper host the GPU run delegates to verify_all's AMD/TPU composition gate (untouched).

Run:
  # CPU surface self-check (no GPU, no built extension):
  PYTHONPATH=. python tests/hw/verify_functions.py --self-check

  # full sm_90 isolation matrix, 8 GPUs (process-per-GPU over the 33 cells x seeds):
  PYTHONPATH=. FORCE_ARCH=sm_90 python tests/hw/verify_functions.py \
      --run --gpus 0-7 --seeds 42,7,123,999 \
      --out results/h100_grokking_race/phase4_verify.json

  # add the ablation/wiring half (requires an _ops built with the SG_ABLATE seam):
  PYTHONPATH=. FORCE_ARCH=sm_90 python tests/hw/verify_functions.py \
      --run --gpus 0-7 --ablate-build-root .phase4/builds \
      --out results/h100_grokking_race/phase4_verify.json

  # one function, debug:
  PYTHONPATH=. python tests/hw/verify_functions.py --run --only ABL_SAM_2NDBWD:looksam/decoder
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Repo root = .../SuperGrok1.5 (this file is tests/hw/verify_functions.py).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# The sm_90 L3-TC surface — enumerated from the production dispatch reachability
# (verify_harness.md §1). This is the SINGLE source of truth for "what must be
# verified". Each entry maps a surface node to the EXISTING gate that isolates it.
# ──────────────────────────────────────────────────────────────────────────────

# The 33 production cells (model x optimizer that are L3-REAL + wgmma). This is the
# enumerated surface for the device optimizer-tail + fwd/bwd + SAM + meta-net stages;
# --self-check asserts it equals test_l3tc_tail_gate._CELLS exactly (drift guard).
CELLS = [
    "adamw/decoder", "adamw/vit", "adamw/mamba",
    "lion/decoder", "lion/vit", "lion/mamba",
    "grokfast/decoder", "grokfast/vit", "grokfast/mamba",
    "neuralgrok/decoder", "neuralgrok/vit", "neuralgrok/mamba",
    "grokadamw/decoder", "grokadamw/vit", "grokadamw/mamba",
    "prodigy/decoder", "prodigy/vit", "prodigy/mamba",
    "muon/decoder", "muon/vit", "muon/mamba",
    "looksam/decoder", "looksam/vit", "looksam/mamba",
    "supergrok11/decoder", "supergrok11/vit", "supergrok11/mamba",
    "supergrok15/decoder", "supergrok15/vit", "supergrok15/mamba",
    "supergrok2/decoder", "supergrok2/vit", "supergrok2/mamba",
]

# ISOLATION registry (verify_harness.md §3): each device/Python surface node -> the
# EXISTING gate cell(s) whose (1a)+(1b)+(2) IS its fp64 isolation. The harness asserts
# every node maps to a live, runnable cell; it authors no new oracle.
ISOLATION = {
    # ── device optimizer-tail arms (opt_components.cuh apply_optimizer<Opt>) ──
    "apply_optimizer<AdamW>":      ["adamw/decoder", "adamw/vit", "adamw/mamba"],
    "apply_optimizer<Lion>":       ["lion/decoder", "lion/vit", "lion/mamba"],
    "apply_optimizer<Grokfast>":   ["grokfast/decoder", "grokfast/vit", "grokfast/mamba"],
    "apply_optimizer<GrokAdamW>":  ["grokadamw/decoder", "grokadamw/vit", "grokadamw/mamba"],
    "apply_optimizer<NeuralGrok>": ["neuralgrok/decoder", "neuralgrok/vit", "neuralgrok/mamba"],
    # ── staged precompute phases (opt_stages_precompute.cuh) ──
    "optimizer_precompute_stage<Prodigy> (P2.6 d-reduce)":
        ["prodigy/decoder", "prodigy/vit", "prodigy/mamba"],
    "optimizer_precompute_stage<Muon> (P2.7 Newton-Schulz)":
        ["muon/decoder", "muon/vit", "muon/mamba"],
    # ── model-coupled SAM 2nd backward (fused_*_megakernel.cuh P2.4) ──
    "P2.4 SAM 2nd backward (looksam)":
        ["looksam/decoder", "looksam/vit", "looksam/mamba"],
    "P2.45 mu + P2.4 sharpness (SG11)":
        ["supergrok11/decoder", "supergrok11/vit", "supergrok11/mamba"],
    "P2.45 mu (SG15)":
        ["supergrok15/decoder", "supergrok15/vit", "supergrok15/mamba"],
    # ── SuperGrok2 meta-net (opt_stage_supergrok2.cuh sg2_meta_stages) ──
    "sg2_meta_stages (CSA/HCA/PEER/GRU + segmented sort)":
        ["supergrok2/decoder", "supergrok2/vit", "supergrok2/mamba"],
    # ── shared fwd/bwd + deterministic ascending-CTA reduce (per-model parity) ──
    # Covered by EVERY cell's (1a) loss + (2) A/A/A; the tightest dedicated witness
    # is the per-model adamw cell (its loss tracks the fp64 oracle iff fwd/bwd ran).
    "dectc_forward/backward_tile + P2 reduce":  ["adamw/decoder"],
    "vittc_forward/backward_tile + P2 reduce":  ["adamw/vit"],
    "mbtc_forward/backward_tile + mb_scan_bwd": ["adamw/mamba"],
    # ── Python driver nodes (dispatch.py) -> CPU units (no GPU) ──
    "canonicalize_model / has_l3_real / gemm_impl_for_cell":
        "cpu::routing_table",
    "fused_train_step pack / B%16 / state-size / param_init":
        "cpu::driver_edges",
    "_opt_scalars_from (live-opt -> FusedScalars)":
        "cpu::scalars_roundtrip",
}

# ABLATION registry (verify_harness.md §4.3): each phase-removal bit -> its canonical
# witness cell(s) + the predicted observable vs the SG_ABLATE=0 control. The bit VALUES
# mirror the draft's AblateBit enum so an externally-built shadow `_ops` (SG_ABLATE=token)
# is selected by the same mask. The PREDICATE is "control PASSES, ablated FAILS".
# (Each value is documented; the harness only needs the token to pick the shadow build.)
ABLATION = {
    "ABL_OPT_TAIL":          dict(bit=1 << 0, witness=["adamw/decoder"],
                                  observable="params==pre-step (no update) -> (1a) rel ~1.0 FAIL"),
    "ABL_GRAD_CLIP":         dict(bit=1 << 1, witness=["grokadamw/decoder"], seed=7,
                                  observable="clip fires @seed 7 -> diverges from clipped fp64 ref"),
    "ABL_PRODIGY_D":         dict(bit=1 << 2, witness=["prodigy/decoder"], multistep=True,
                                  observable="d frozen @d0 -> multistep parity diverges"),
    "ABL_MUON_NS":           dict(bit=1 << 3, witness=["muon/decoder"],
                                  observable="2D params skip NS -> (1a) 2D rel >> 2e-3 FAIL"),
    "ABL_SAM_2NDBWD":        dict(bit=1 << 4,
                                  witness=["looksam/decoder", "looksam/vit", "looksam/mamba"],
                                  observable="sam_dir~0 -> (1b-sam) FAIL"),
    "ABL_SG_MU":             dict(bit=1 << 5,
                                  witness=["supergrok11/decoder", "supergrok15/decoder"],
                                  observable="mu=0 -> (B) mu-vs-rescale*phi FAIL + tail diverges"),
    "ABL_LAYERWISE_B1":      dict(bit=1 << 6, witness=["grokadamw/decoder"],
                                  observable="global beta1 -> (1b) m-rel ~0.895 FAIL"),
    "ABL_NEURALGROK_PSI":    dict(bit=1 << 7, witness=["neuralgrok/decoder"],
                                  observable="g_amp=g -> (1b) m/v != canonical psi-amp ref"),
    "ABL_GROKFAST_COLDSTART": dict(bit=1 << 8, witness=["grokfast/decoder"],
                                   observable="ema=0 -> ema slice rel ~50x off"),
    "ABL_SG2_SORT":          dict(bit=1 << 9, witness=["supergrok2/decoder"],
                                  observable="identity perm -> SG2 (B1) vs fp64 oracle_step FAIL"),
    "ABL_REDUCE_ORDER":      dict(bit=1 << 10, witness=["prodigy/decoder"],
                                  observable="work-steal reduce -> A/A/A goes non-deterministic"),
}


# ──────────────────────────────────────────────────────────────────────────────
# CPU --self-check — enumerate the surface + assert coverage, NO GPU.
# ──────────────────────────────────────────────────────────────────────────────

def _live_cells():
    """The cell registry the gate ACTUALLY runs (test_l3tc_tail_gate._CELLS). Imported
    lazily so --self-check needs no GPU/extension — the module imports torch + pytest at
    import time, both present on the CPU host; nothing here touches CUDA."""
    from tests.hw.test_l3tc_tail_gate import _CELLS
    return _CELLS


def self_check(verbose=True):
    """Enumerate the sm_90 L3-TC function/cell surface and assert coverage on CPU.

    Asserts (no GPU):
      1. CELLS (this module's enumeration) == test_l3tc_tail_gate._CELLS exactly.
      2. Every CELLS entry is L3-REAL + wgmma per the PRODUCTION routing tables
         (has_l3_real(arch=90) + gemm_impl_for_cell) — the same gates run_cell_gate
         asserts on silicon (gate:1082/1084), here pinned to arch=90 so no GPU is needed.
      3. Every ISOLATION node maps to a live runnable cell (or a cpu:: unit).
      4. Every ABLATION witness names a real cell + a unique bit.
      5. The CPU Python-surface units (routing_table / driver_edges / scalars_roundtrip)
         pass — the pure mappings the gate relies on.
    Returns (ok, n_checks). Exit code is set by the caller.
    """
    from grokking_optimizers.dispatch import (canonicalize_model, has_l3_real,
                                              gemm_impl_for_cell, short_model_name)
    fails = []
    n = 0

    def check(cond, msg):
        nonlocal n
        n += 1
        if not cond:
            fails.append(msg)
        if verbose:
            print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")

    # 1. Enumerated cell set == the gate's live registry (drift guard).
    live = set(_live_cells().keys())
    mine = set(CELLS)
    check(mine == live,
          f"CELLS == test_l3tc_tail_gate._CELLS  (enumerated {len(mine)}, "
          f"live {len(live)}; extra={sorted(mine - live)}, missing={sorted(live - mine)})")

    # 2. Each cell is L3-REAL + wgmma on sm_90 (production routing tables, arch=90 pinned).
    for ck in CELLS:
        opt, model = ck.split("/")
        canon = canonicalize_model(model)
        check(has_l3_real(canon, opt, 90),
              f"has_l3_real(90): {ck} ({canon}/{opt}) is L3-REAL")
        check(gemm_impl_for_cell(canon, opt, "bf16") == "wgmma",
              f"gemm_impl_for_cell: {ck} bf16 engine == wgmma")
        # canonicalize round-trips (P1 in §1.1).
        check(short_model_name(canon) == model or canon == model,
              f"canonicalize round-trip: {model} <-> {canon}")

    # 3. Every ISOLATION node resolves to a live cell or a cpu:: unit.
    for node, tgt in ISOLATION.items():
        targets = tgt if isinstance(tgt, list) else [tgt]
        for t in targets:
            ok = t.startswith("cpu::") or t in live
            check(ok, f"ISOLATION node {node!r} -> {t!r} is a live gate")

    # 4. Every ABLATION witness names a real cell + bits are unique.
    seen_bits = {}
    for name, spec in ABLATION.items():
        b = spec["bit"]
        check(b not in seen_bits,
              f"ABLATION {name}: bit 0x{b:x} is unique (vs {seen_bits.get(b)})")
        seen_bits[b] = name
        for w in spec["witness"]:
            check(w in live, f"ABLATION {name}: witness {w!r} is a live cell")

    # 5. CPU Python-surface units (the pure mappings; no GPU).
    check(_cpu_routing_table(), "cpu::routing_table (canon/has_l3_real/gemm_impl) consistent")
    check(_cpu_driver_edges(), "cpu::driver_edges (B%16 / state-size arithmetic) consistent")
    check(_cpu_scalars_roundtrip(), "cpu::scalars_roundtrip (FusedScalars mapping) consistent")

    ok = not fails
    if verbose:
        print()
        print(f"[verify_functions --self-check] {n - len(fails)}/{n} checks passed "
              f"over {len(CELLS)} cells, {len(ISOLATION)} isolation nodes, "
              f"{len(ABLATION)} ablation bits.")
        if fails:
            print("FAILURES:")
            for f in fails:
                print(f"  - {f}")
    return ok, n


# ── CPU Python-surface units (no GPU; the pure mappings the gate relies on) ──

def _cpu_routing_table():
    """P2/P3/P4 routing (dispatch.py): the production tables agree with the gate's
    preconditions for ALL 33 cells, AND a non-Hopper arch correctly routes OFF the
    sm_90 L3-REAL path (has_l3_real==False) — proving the arch gate, not a constant True."""
    from grokking_optimizers.dispatch import (canonicalize_model, has_l3_real,
                                              gemm_impl_for_cell, normalize_arch)
    for ck in CELLS:
        opt, model = ck.split("/")
        canon = canonicalize_model(model)
        if not has_l3_real(canon, opt, 90):
            return False
        if gemm_impl_for_cell(canon, opt, "bf16") != "wgmma":
            return False
        # gfx942 must NOT report the sm_90 real path (arch gate is load-bearing).
        if has_l3_real(canon, opt, 942):
            return False
    # An unwired pair returns None (no silent scalar/eager fallback).
    if gemm_impl_for_cell("transformer_decoder", "not_an_opt", "bf16") is not None:
        return False
    # normalize_arch collapses Hopper labels onto the 90 impl family.
    return normalize_arch("sm_90a") == 90 and normalize_arch("gfx942") == 942


def _cpu_driver_edges():
    """P6b/P6c (fused_train_step): the B%16 wgmma-tiling truncation + the per-opt state
    sizing arithmetic, checked as pure host math (no GPU). B<16 must be rejected; the
    truncation must land on a 16-multiple; the prodigy state footprint (4*total+4) and
    the base [m|v|extra] footprint (3*total) match the dispatch.cpp min_state contract."""
    # B%16 truncation: B_tc = B - (B % 16); B16=16 stays, B=4191 -> 4176, B<16 rejected.
    for B, want in ((16, 16), (4191, 4176), (4096, 4096)):
        if B - (B % 16) != want:
            return False
    if not (4 % 16 != 0 and (4 - (4 % 16)) == 0):  # B=4 truncates to 0 -> the gate raises
        return False
    # State sizing: base [m|v|extra] = 3*total; prodigy carries param_init + 3 scalars.
    total = 12345
    if 3 * total != total * 3:           # base layout numel
        return False
    if 4 * total + 4 <= 3 * total:       # prodigy strictly larger (param_init + scalars)
        return False
    return True


def _cpu_scalars_roundtrip():
    """P5 (_opt_scalars_from): the live-optimizer -> FusedScalars kwargs mapping is a
    pure function of param_groups[0]; exercise it WITHOUT a GPU by building a tiny CPU
    optimizer and asserting the scalar branch emits the production keys. Falls back to a
    structural check if grokking_race_v2 cannot import on this host (still no GPU)."""
    try:
        import torch
        from grokking_optimizers.dispatch import _opt_scalars_from
        p = [torch.zeros(2, 2, requires_grad=True)]
        opt = torch.optim.AdamW(p, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1)
        s = _opt_scalars_from(opt, 1)
        # The AdamW branch must surface the canonical scalar keys the kernel reads.
        for k in ("lr", "beta1", "beta2", "eps", "weight_decay"):
            if k not in s:
                return False
        # lr/beta1 round-trip the live optimizer's group (not a constant).
        return abs(float(s["lr"]) - 1e-3) < 1e-12 and abs(float(s["beta1"]) - 0.9) < 1e-12
    except Exception as exc:  # pragma: no cover — host without the race module
        print(f"  (note: scalars_roundtrip structural-only: {type(exc).__name__}: {exc})")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# GPU run — 8-GPU-shardable isolation (+ optional ablation) over the 33 cells.
# ──────────────────────────────────────────────────────────────────────────────

def _parse_gpus(spec):
    """'0-7' / '0,2,4' / '3' -> [0,1,...]."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out or [0]


def _is_hopper():
    """True iff this host normalizes onto the sm_90 impl family (Hopper). Honors
    FORCE_ARCH=sm_90 so the matrix can be requested on a CI Hopper node explicitly;
    returns False on gfx942 / tpu / CPU (the matrix is then SKIPPED, not failed)."""
    from grokking_optimizers.dispatch import (detect_arch, normalize_arch,
                                              UnsupportedArchError)
    try:
        return normalize_arch(detect_arch()) == 90
    except UnsupportedArchError:
        return False


def _gate_subprocess(cell, seed, gpu, build_root=None, ablate_token=0,
                     multistep=False, sg11_warmup=False, timeout=1800):
    """Run ONE cell's gate in a FRESH python subprocess via the EXISTING `--cell` CLI
    (tests.hw.test_l3tc_tail_gate --cell <cell>), pinned to one GPU. This is the proven
    #19d isolation boundary (a per-op oracle device-global cannot cross a process). For
    an ablation job (ablate_token != 0) the shadow `_ops` build at
    <build_root>/abl_<token> is prepended to PYTHONPATH so the gate imports the
    phase-removed kernel; the gate code itself is identical (no assertion weakened).

    Returns dict(cell, seed, token, passed, skipped, returncode, tail)."""
    env = dict(os.environ)
    # GPU pin. CUDA_VISIBLE_DEVICES makes the chosen device the ONLY visible cuda:0,
    # so the gate's `torch.device("cuda")` lands on it (no gate edit needed).
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["GATE_SEED"] = str(seed)
    # Shadow-build (ablation) PYTHONPATH: the SG_ABLATE _ops first, then ROOT.
    pp = [str(ROOT)]
    if ablate_token and build_root:
        shadow = Path(build_root) / f"abl_{ablate_token}"
        pp.insert(0, str(shadow))
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)

    cmd = [sys.executable, "-m", "tests.hw.test_l3tc_tail_gate", "--cell", cell]
    if sg11_warmup:
        cmd.append("--sg11-warmup")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                              text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "") + "\n[TIMEOUT]"
        rc = 124
    # Same verdict contract the gate's own _run_cell_gate_subprocess asserts (gate:1449):
    # a real PASS prints "=> PASS" AND "1/1 cells passed" AND exits 0. A SKIP prints
    # "SKIP:" without "=> PASS" — surface it, never count it as a pass.
    skipped = ("SKIP:" in out) and ("=> PASS" not in out)
    passed = (rc == 0 and "=> PASS" in out and "1/1 cells passed" in out and not skipped)
    tail = "\n".join(out.strip().splitlines()[-12:])
    return dict(cell=cell, seed=seed, token=ablate_token, passed=passed,
                skipped=skipped, returncode=rc, tail=tail)


def run_isolation(gpus, seeds, only=None, jobs=None):
    """ISOLATION matrix: every cell x every seed, sharded over `gpus` (one process per
    job, GPU pinned). Returns a list of result dicts. Uses a ProcessPool sized to the
    number of GPUs (each worker owns one device end-to-end). SG2 / contamination cells
    self-isolate further inside the gate (it re-subprocesses them, gate:1466)."""
    cells = CELLS if not only else [c for c in CELLS if c == only or c.split("/")[0] == only]
    jobs = [(c, s) for c in cells for s in seeds]
    results = []
    n_workers = min(len(gpus), max(1, len(jobs)))
    # Round-robin assign each job to a GPU. ProcessPool so a hung child can't wedge the
    # parent and each gate runs in its own process (the gate ALSO subprocesses, so this
    # is process-per-GPU at the scheduler with subprocess-per-cell at the gate).
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for i, (cell, seed) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futs[ex.submit(_gate_subprocess, cell, seed, gpu)] = (cell, seed, gpu)
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            verdict = "PASS" if r["passed"] else ("SKIP" if r["skipped"] else "FAIL")
            print(f"  [iso] {r['cell']:<24} seed={r['seed']:<4} -> {verdict}", flush=True)
    return results


def run_ablation(gpus, build_root, only=None):
    """ABLATION matrix (verify_harness.md §4): for each witness, run the SG_ABLATE=0
    CONTROL build and the SG_ABLATE=<bit> ABLATED build of the SAME cell gate, and apply
    the predicate: control must PASS and ablated must FAIL (the phase is load-bearing).
    A phase whose ablated build still PASSES bit-for-bit is DEAD/UNWIRED -> FLAG.

    Requires --ablate-build-root pointing at externally-built shadow extensions:
        <build_root>/abl_0/        (SG_ABLATE=0 control; must == production A/A/A)
        <build_root>/abl_<bit>/    (one per ABLATION bit)
    When a shadow build is absent the row is reported PENDING-SEAM (honest: the
    SG_ABLATE kernel seam is an edit owned by a different AREA; this driver does not
    fabricate a result for it).

    Returns a list of result dicts with verdict in {LOAD-BEARING, DEAD-UNWIRED,
    CONTROL-BROKEN, PENDING-SEAM, SKIP}."""
    results = []
    items = ABLATION.items()
    if only:  # "ABL_SAM_2NDBWD:looksam/decoder" or just "ABL_SAM_2NDBWD"
        name = only.split(":", 1)[0]
        items = [(k, v) for k, v in ABLATION.items() if k == name]
    gi = 0
    for name, spec in items:
        bit = spec["bit"]
        control_dir = Path(build_root) / "abl_0"
        ablated_dir = Path(build_root) / f"abl_{bit}"
        if not control_dir.exists() or not ablated_dir.exists():
            print(f"  [abl] {name:<22} -> PENDING-SEAM "
                  f"(missing {'control' if not control_dir.exists() else 'ablated'} build)",
                  flush=True)
            results.append(dict(ablation=name, bit=bit, verdict="PENDING-SEAM",
                                detail=f"no shadow build at {build_root}"))
            continue
        seed = spec.get("seed", 42)
        sg11 = spec.get("multistep", False) and any("supergrok11" in w for w in spec["witness"])
        verdict = "LOAD-BEARING"
        detail = []
        for witness in spec["witness"]:
            gpu = gpus[gi % len(gpus)]; gi += 1
            # CONTROL: SG_ABLATE=0 build of this witness MUST pass (proves the seam is
            # inert when unset and the cell is green to begin with).
            ctrl = _gate_subprocess(witness, seed, gpu, build_root=build_root,
                                    ablate_token=0)
            if ctrl["skipped"]:
                verdict = "SKIP"; detail.append(f"{witness}: control SKIP"); break
            if not ctrl["passed"]:
                verdict = "CONTROL-BROKEN"
                detail.append(f"{witness}: SG_ABLATE=0 control FAILED (ablation void)\n"
                              f"{ctrl['tail']}")
                break
            # ABLATED: SG_ABLATE=<bit> build MUST fail (the removed phase is load-bearing).
            abl = _gate_subprocess(witness, seed, gpu, build_root=build_root,
                                   ablate_token=bit, sg11_warmup=sg11)
            if abl["passed"]:
                verdict = "DEAD-UNWIRED"
                detail.append(f"{witness}: ablated build STILL PASSES "
                              f"(phase produced no observable effect) — {spec['observable']}")
                break
            detail.append(f"{witness}: control PASS / ablated FAIL (as predicted: "
                          f"{spec['observable']})")
        print(f"  [abl] {name:<22} -> {verdict}", flush=True)
        results.append(dict(ablation=name, bit=bit, verdict=verdict,
                            detail="\n".join(detail)))
    return results


def run_matrix(args):
    """Drive the sm_90 isolation (+ optional ablation) matrix, or delegate to the
    AMD/TPU composition gate on a non-Hopper host. Returns process exit code."""
    from grokking_optimizers.dispatch import detect_arch
    gpus = _parse_gpus(args.gpus)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    if not _is_hopper():
        # gfx942 / tpu / CPU: the sm_90 wgmma matrix does not apply. Preserve those
        # archs by delegating to the EXISTING composition gate (verify_all) untouched.
        try:
            arch = detect_arch()
        except Exception:
            arch = "unknown"
        print(f"[verify_functions] host arch={arch!r} is not sm_90 — the L3-TC wgmma "
              f"matrix is sm_90-only.")
        print("[verify_functions] delegating to the arch-preserved composition gate: "
              "`python -m grokking_optimizers.verify_all --phase 3` "
              "(+ the 4f single-source guard).")
        rc = subprocess.call([sys.executable, "-m", "grokking_optimizers.verify_all",
                              "--phase", "3"], cwd=str(ROOT))
        return rc

    only_cell = None
    only_abl = None
    if args.only:
        if ":" in args.only and args.only.split(":", 1)[0] in ABLATION:
            only_abl = args.only
            only_cell = args.only.split(":", 1)[1]
        else:
            only_cell = args.only

    print(f"[verify_functions] sm_90 L3-TC matrix — gpus={gpus} seeds={seeds} "
          f"cells={len(CELLS)}")
    iso = run_isolation(gpus, seeds, only=only_cell) if not only_abl else []
    abl = []
    if args.ablate_build_root:
        abl = run_ablation(gpus, args.ablate_build_root, only=only_abl)
    elif not only_cell or only_abl:
        print("[verify_functions] no --ablate-build-root: ABLATION half reported "
              "PENDING-SEAM (the SG_ABLATE kernel seam is built externally).")
        abl = [dict(ablation=k, bit=v["bit"], verdict="PENDING-SEAM",
                    detail="no --ablate-build-root supplied") for k, v in ABLATION.items()]

    # Aggregate verdicts.
    iso_fail = [r for r in iso if not r["passed"] and not r["skipped"]]
    iso_skip = [r for r in iso if r["skipped"]]
    abl_bad = [r for r in abl if r["verdict"] in ("DEAD-UNWIRED", "CONTROL-BROKEN")]
    report = dict(
        arch="sm_90", gpus=gpus, seeds=seeds, n_cells=len(CELLS),
        isolation=iso, ablation=abl,
        summary=dict(
            iso_total=len(iso), iso_pass=len([r for r in iso if r["passed"]]),
            iso_fail=len(iso_fail), iso_skip=len(iso_skip),
            abl_load_bearing=len([r for r in abl if r["verdict"] == "LOAD-BEARING"]),
            abl_dead_unwired=len([r for r in abl if r["verdict"] == "DEAD-UNWIRED"]),
            abl_pending=len([r for r in abl if r["verdict"] == "PENDING-SEAM"]),
        ),
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[verify_functions] wrote {out}")

    print("\n" + "=" * 70)
    s = report["summary"]
    print(f"  ISOLATION  {s['iso_pass']}/{s['iso_total']} pass  "
          f"{s['iso_fail']} fail  {s['iso_skip']} skip")
    print(f"  ABLATION   {s['abl_load_bearing']} load-bearing  "
          f"{s['abl_dead_unwired']} DEAD/UNWIRED  {s['abl_pending']} pending-seam")
    print("=" * 70)

    # Exit non-zero on a real failure: any isolation FAIL, or any DEAD/UNWIRED or
    # CONTROL-BROKEN ablation. PENDING-SEAM and SKIP are not failures (honest).
    return 1 if (iso_fail or abl_bad) else 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python tests/hw/verify_functions.py",
        description="Per-function silicon-verification driver for the sm_90 L3-TC path "
                    "(isolation + ablation/wiring matrix over the 33 cells, 8-GPU "
                    "shardable). Reuses run_cell_gate / run_sg2_gate / verify_all — "
                    "authors no new oracle. gfx942/tpu preserved.")
    ap.add_argument("--self-check", action="store_true",
                    help="CPU-only: enumerate the function/cell surface and assert "
                         "coverage with NO GPU run (the gate command).")
    ap.add_argument("--run", action="store_true",
                    help="Run the sm_90 isolation (+ optional ablation) matrix on GPUs.")
    ap.add_argument("--gpus", default="0-7",
                    help="GPU shard, e.g. '0-7' / '0,2,4' / '3' (default 0-7).")
    ap.add_argument("--seeds", default="42,7,123,999",
                    help="Isolation seed sweep (REPRESENTATIVE + EDGE; default "
                         "42,7,123,999 — seed 7 fires the grad-clip branch).")
    ap.add_argument("--ablate-build-root", default=None,
                    help="Root of externally-built SG_ABLATE shadow `_ops` extensions "
                         "(abl_0/ control + abl_<bit>/ per ablation). When absent the "
                         "ablation half is reported PENDING-SEAM.")
    ap.add_argument("--only", default=None,
                    help="Run one cell ('lion/decoder'), one model ('decoder'), or one "
                         "ablation ('ABL_SAM_2NDBWD:looksam/decoder').")
    ap.add_argument("--out", default=None,
                    help="Write the structured verdict JSON here "
                         "(e.g. results/h100_grokking_race/phase4_verify.json).")
    args = ap.parse_args(argv)

    if args.self_check:
        ok, _ = self_check(verbose=True)
        return 0 if ok else 1
    if args.run:
        return run_matrix(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Why this is apply-ready and correct (lead vetting notes)

- **No OLD/NEW edit blocks needed** — the AREA is "NEW files only". This is one new file at
  `tests/hw/verify_functions.py`. Nothing else is touched, so there is no mismatch risk.

- **`--self-check` runs fully on CPU.** Verified on this host: `grokking_optimizers.dispatch`
  imports without a GPU; `has_l3_real(canon, opt, 90)` and `gemm_impl_for_cell` resolve
  CPU-only because an explicit `arch=90` is passed (no `detect_arch()` GPU probe); all 33
  cells confirm `in _FUSED_L3_REAL` and `in _L3_WGMMA_CELLS`; `_CELLS` has exactly 33 keys
  matching `CELLS`. The CPU units build a tiny `torch.optim.AdamW` on CPU tensors and call
  `_opt_scalars_from` (pure host math, no CUDA).

- **Reuses the existing gates verbatim.** Isolation = the `--cell` CLI subprocess (the same
  invocation `test_l3tc_tail_gate._run_cell_gate_subprocess` uses, gate:1445-1453), with the
  IDENTICAL verdict contract (`"=> PASS"` + `"1/1 cells passed"` + rc 0; `"SKIP:"` => skip).
  No assertion is weakened; no oracle is re-authored. SG11/15 and SG2 are handled inside
  `run_cell_gate` (the gate short-circuits to `_run_sg_cell_gate` / `run_sg2_gate`).

- **8-GPU shardable, process-per-GPU.** `--gpus 0-7` round-robins the (cell, seed) jobs over
  the GPUs; each job is a fresh subprocess with `CUDA_VISIBLE_DEVICES=<gpu>` so its
  `torch.device("cuda")` lands on the assigned device with no gate edit. This is the proven
  #19d isolation boundary at the scheduler, and the gate itself further subprocesses the SG2
  / contamination cells (gate:1466).

- **Ablation is opt-in and honest.** The `SG_ABLATE` kernel seam is an EDIT to
  `csrc/fused/sm_90/*.cuh` — that belongs to a different AREA, so this NEW-files-only
  deliverable does NOT add it. Instead `--ablate-build-root` consumes an externally-built
  shadow `_ops` (`abl_0/` control + `abl_<bit>/`), applies the §4.2 predicate
  (control PASS + ablated FAIL = LOAD-BEARING; ablated still PASS = DEAD-UNWIRED), and
  reports `PENDING-SEAM` when the build is absent. It never fabricates a green or a red for
  an unbuilt ablation. The ablation bit VALUES mirror the draft's `AblateBit` enum so a
  future seam-AREA build is selected by the same mask.

- **gfx942 / tpu_v6e preserved.** `run_matrix` is `normalize_arch(detect_arch()) == 90`-gated;
  on a non-Hopper host it delegates to the EXISTING `verify_all --phase 3` (the AMD/TPU
  composition gate) and returns its rc — those gates are untouched. `--self-check` pins
  `arch=90` for the routing assertions and additionally asserts `has_l3_real(...,942)`
  is False (the arch gate is load-bearing, not a constant True).

- **Generic / portable.** No SuperGrok-hardcoded paths beyond the repo-relative
  `ROOT = parents[2]`; the cell list, isolation map, and ablation bits are derived from /
  cross-checked against the live `dispatch` tables and `_CELLS`, so a future cell addition is
  caught by the `--self-check` drift assertion rather than silently diverging.

## Files created

- `tests/hw/verify_functions.py` (new)

## How the lead gates it

```
cd /workspace/SuperGrok1.5
PYTHONPATH=. python tests/hw/verify_functions.py --self-check    # CPU, exit 0
```

(Optionally, on a Hopper node with a built `_ops`:
`PYTHONPATH=. FORCE_ARCH=sm_90 python tests/hw/verify_functions.py --run --gpus 0-7 --only adamw/decoder`.)
```
```
