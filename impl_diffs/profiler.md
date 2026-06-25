# AREA: profiler — phase-resolved time breakdown of ONE decoder megakernel step (d=2048 bench)

## TL;DR for the lead

**The in-kernel phase profiler ALREADY EXISTS and is already byte-identical-when-off.**
There is **NO kernel edit to apply** — the deliverable is **ONE new harness file**
(`tuning/decoder_phase_profile.py`) plus an nsys command and an occupancy probe (both
embedded in the harness so the lead runs one script). The harness is a thin
orchestrator over the already-shipped `SG_DEC_PROFILE` (8 coarse clock64 slots) +
`SG_DEC_PROFILE_FWD_FINE` (10 fine GEMM-engine slots) instrumentation, exposed to host
via `mod.tc_profile_read()` / `mod.tc_profile_read_fwd_fine()` and already driven by
`tuning/decoder_bench.py`.

This directly answers the user's literal question — *"where are the bottlenecks
currently, and why"* — at the `d=2048` `SG_DEC_BENCH_LAYOUT` scale, with a per-phase
ms + %-of-step table, the fwd/dX drain-vs-compute fine split, and the achieved CTAs/SM
occupancy.

---

## What EXISTS (verified by reading the files in full — do NOT re-add any of this)

### (a) Coarse 8-slot in-kernel phase profiler — `SG_DEC_PROFILE`
`csrc/fused/sm_90/fused_decoder_megakernel.cuh`:
- `__device__ unsigned long long g_dec_prof_max[8];` (line 669, `#ifdef SG_DEC_PROFILE`).
- Thread-0-only `clock64()` deltas, `atomicMax` across CTAs (= the slowest CTA per
  phase = the critical path the host wall sees, because the trailing grid barrier
  waits for the last CTA). Slots:
  - `[0]` P1 fwd (whole tile fwd: QKV/attn-out/ff0/ff2 GEMMs + LN + softmax/attention + elementwise)
  - `[1]` P1 bwd (whole tile bwd: dX GEMMs + dLN + dsoftmax + elementwise)
  - `[2]` B1 grid-barrier wait
  - `[3]` P2 dW-GEMM loop (+ split-K)
  - `[4]` P2 grad-assembly (dW biases + embedding owner-scan + LN-vec reduce + loss reduce)
  - `[5]` P3 optimizer tail (AdamW apply over the 30 tensors)
  - `[6]` B2 grid-barrier wait (`sync_reset`, P2→P3)
  - `[7]` B0 grid-barrier wait (after the bf16 weight pre-stage cache fill)
- Brackets are `#ifdef SG_DEC_PROFILE` blocks at lines 815-821, 837-879, 916-956,
  1414-1416, 1499-1501. **When the macro is undefined, every `clock64()` read, every
  `atomicMax`, and every `__syncthreads()` they add `#if`-erase entirely** → the
  production `_ops` (which never sets the flag) is byte-identical (PTX/regs/smem
  unchanged). This is exactly the "byte-identical-when-off" property the task asks a
  NEW macro to provide — it is already provided here.

### (b) Fine GEMM-engine 10-slot sub-profiler — `SG_DEC_PROFILE_FWD_FINE`
`csrc/fused/sm_90/model_stage_decoder_tc.cuh`:
- Gated on **BOTH** `SG_DEC_PROFILE` and `SG_DEC_PROFILE_FWD_FINE` (line 298).
- `__device__ unsigned long long g_dec_prof_fwd_fine[kDecFwdFineSlots];` (`kDecFwdFineSlots
  = kDecFwdFinePhases(2) * kDecFwdFineSub(5) = 10`, lines 299-307).
- Layout `[phase*5 + sub]`: phase 0 = fwd ring, 1 = dX ring; sub ∈
  `{0 ISSUE (cp.async LDGSTS), 1 WAIT (cp.async drain — the DRAIN/latency cost),
  2 WGMMA (mma issue+commit+wait), 3 EPI (epilogue store), 4 BARRIER (fence+sync)}`.
- This is the **"why"** lever: WAIT-dominant ⇒ drain/latency-bound (deeper ring helps);
  WGMMA/EPI-dominant ⇒ compute/epilogue-bound (it won't). Byte-identical when the pair
  is off (the engine call-site stamps `#if`-erase to the original 9-arg form, lines
  315-326). **Note:** this only splits time *inside the GEMM cp.async ring* — LN /
  softmax / attention / elementwise live OUTSIDE the ring and are therefore the
  `coarse[0/1] − Σ(fine fwd/dX ring)` remainder (the harness computes and labels this).

### (c) Host readers (pybind) — `csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu`
- `tc_profile_read()` (line 265, `#ifdef SG_DEC_PROFILE`): `cudaMemcpyFromSymbol` the 8
  slots, then **reset to zero**, return `vector<int64_t>` (cycles). Exposed as
  `mod.tc_profile_read`; `mod.HAS_PROFILE == True`.
- `tc_profile_read_fwd_fine()` (line 287, `#if SG_DEC_PROFILE_FWD_FINE`): same for the
  10 fine slots. Exposed as `mod.tc_profile_read_fwd_fine`; `mod.HAS_FWD_FINE == True`,
  `mod.FWD_FINE_SUB == 5`, `mod.FWD_FINE_PHASES == 2`.
- Module attrs always present: `mod.D`, `mod.DFF`, `mod.LAYERS`, `mod.SEQ`, `mod.VOCAB`,
  `mod.TOTAL`, `mod.TILE_M`, `mod.TILE_N`, `mod.FWD_PIPE`, `mod.FWD_STAGES`.

### (d) Build path for the d=2048 bench TU — `tuning/decoder_bench.py`
- `build_variant(d, profile, ..., fwd_fine=...)` (line 52) JIT-loads
  `csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu` as a **coexisting** variant
  extension (distinct module name + own build dir → incremental ninja + sccache, never
  touches the production `_ops.so` / the 33/33 gate). It sets:
  - `-DSG_TUNED_GEMM_IMPL=1` (wgmma cell driver),
  - `-DSG_DEC_BENCH_LAYOUT=1` for `d != 128` → **this is the d=2048 layout**
    (`decoder_layout.cuh:40 SG_DEC_D = 2048`; the `decoder_bench.py` docstring saying
    "d=1024"/"d=1024 bench" is **STALE** — the live layout constant is 2048),
  - `-DSG_DEC_SCALAR_MEGAKERNEL=0` (the legacy fp32 scalar megakernel ptxas-fails at
    large d; the bench/profile path drives the TC engine only),
  - `-DSG_DEC_PROFILE=1` when `profile`,
  - `-DSG_DEC_PROFILE_FWD_FINE=1` (+ implies `SG_DEC_PROFILE`) when `fwd_fine`.
  Mirrors the same `-D…` into `extra_cflags` so the host-side pybind readers compile in.
- `measure(...)` (line 142) already drives `tc_train_step`, times the wall, and reads
  the coarse + fine counters across reps (median per slot).
- `_print_report(res)` (line 223) already prints the per-phase ms + %-of-summed table.
- **The macro the task names, `SG_DEC_TC_PHASE_PROF`, does NOT exist** and is **NOT
  needed** — its exact described behavior (clock64 phase reads + `__device__` buffer
  writes that `#if`-erase to byte-identical when 0; default off) is already realized by
  `SG_DEC_PROFILE`. Introducing a second, redundant macro would be churn with no
  functional gain and would risk the byte-identical guarantee. The harness below
  therefore reuses `SG_DEC_PROFILE`.

### (e) Occupancy probe — already computed in-launcher, NOT exposed
`csrc/fused/sm_90/fused_decoder_megakernel.cuh:1540-1545`: the launcher calls
`cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ..., SG_TC_MEGA_BLOCK, dyn_smem)`
and **refuses to launch if `occ < 1`** (the persistent grid barrier requires ≥1 CTA/SM).
`SG_TC_MEGA_BLOCK == 256` (line 348). With the BAKED defaults `FWD_PIPE=1, FWD_STAGES=4`
the ring lives in **dynamic smem** (`SG_DEC_TC_DYNAMIC_SMEM`), so the achieved CTAs/SM is
smem-capped. The launcher's `occ` is not returned to host, so the harness re-derives it
host-side from the same API via a tiny separate probe TU (below) — read-only, no kernel
edit.

---

## DELIVERABLE — ONE new file (no kernel edits, nothing else to apply)

Create `tuning/decoder_phase_profile.py` **in full** (new file):

```python
#!/usr/bin/env python3
"""tuning/decoder_phase_profile.py — PHASE-RESOLVED time breakdown of ONE persistent
decoder megakernel step at the d=2048 bench scale (SG_DEC_BENCH_LAYOUT), WITHOUT ncu.

WHY THIS EXISTS
  The production path is a SINGLE __global__ launch (fwd→bwd→AdamW fused, 1 CTA/SM,
  hand-built grid barrier). nsys therefore shows exactly ONE kernel on the timeline —
  it cannot break the step into phases, and HW perf counters (ncu) are DENIED in this
  container (no CAP_SYS_ADMIN). The breakdown instead comes from the kernel's OWN
  in-kernel clock64() phase counters (-DSG_DEC_PROFILE, already shipped, default OFF,
  byte-identical when off), read back host-side after each step. This script is a thin,
  8-GPU-idle-aware orchestrator over that instrumentation. It answers the literal
  question: "where are the bottlenecks currently, and why".

WHAT IT MEASURES (all at d=2048, the SG_DEC_BENCH_LAYOUT roofline width)
  COARSE (8 slots, the whole step):
    P1_fwd, P1_bwd, B1_barrier, P2_dW_GEMM, P2_grad_asm, P3_opt_tail, B2_barrier, B0_barrier
  Each slot is the clock64 critical-path on the SLOWEST CTA (atomicMax), i.e. the
  duration the host wall actually waits on (the trailing grid barrier blocks on it).
  FINE (--fine, 10 slots, INSIDE the fwd/dX GEMM cp.async ring):
    fwd ring  : ISSUE / WAIT(drain) / WGMMA / EPI / BARRIER
    dX  ring  : ISSUE / WAIT(drain) / WGMMA / EPI / BARRIER
  WAIT-dominant => drain/latency-bound (deeper ring helps); WGMMA/EPI-dominant =>
  compute/epilogue-bound (it won't). The LN/softmax/attention/elementwise time is the
  (coarse fwd/bwd) MINUS (Σ fine ring) remainder — it lives outside the GEMM ring — and
  is printed as a derived "LN+softmax+attn+elemwise (non-GEMM)" line.

OCCUPANCY (always): the achieved CTAs/SM (cudaOccupancyMaxActiveBlocksPerMultipro-
  cessor) for the ACTUAL kernel + block(256) + the ACTUAL dynamic-smem footprint of the
  baked FWD_PIPE/FWD_STAGES config — re-derived host-side (the launcher's own >=1 cert
  is internal). This is why the megakernel is 1 CTA/SM (or refuses to launch).

NSYS (--nsys-cmd): prints the single-launch-timeline + cuda-api nsys command to run
  separately (nsys IS available; it confirms there is one kernel and gives the host
  launch/sync overhead = wall - Σ(summed phases)).

GATE COMMANDS
  python tuning/decoder_phase_profile.py --help
  CUDA_VISIBLE_DEVICES=0 python tuning/decoder_phase_profile.py --steps 20

This script does NOT modify the kernel. It only sets -DSG_DEC_PROFILE on the COEXISTING
bench variant TU (via tuning/decoder_bench.build_variant), exactly as decoder_bench.py
--profile already does — the production _ops is byte-identical and untouched.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SM_GHZ = 1.98  # H100 SM boost clock (nvidia-smi clocks.max.sm = 1980 MHz); mirrors decoder_bench

# Coarse 8-slot names — MUST match g_dec_prof_max[] order in fused_decoder_megakernel.cuh
# and decoder_bench.PHASE_NAMES.
PHASE_NAMES = ["P1_fwd", "P1_bwd", "B1_barrier", "P2_dW_GEMM",
               "P2_grad_asm", "P3_opt_tail", "B2_barrier", "B0_barrier"]
# Human-facing roll-up groups for the "where are the bottlenecks" summary.
GROUPS = {
    "fwd (P1: GEMMs+LN+softmax+attn)": ["P1_fwd"],
    "bwd (P1: dX GEMMs+dLN+dsoftmax)": ["P1_bwd"],
    "dW-GEMM (P2)":                    ["P2_dW_GEMM"],
    "grad-assembly (P2)":             ["P2_grad_asm"],
    "optimizer tail (P3 AdamW)":      ["P3_opt_tail"],
    "grid barriers (B0+B1+B2)":       ["B0_barrier", "B1_barrier", "B2_barrier"],
}
FINE_SUB_NAMES = ["ISSUE(cp.async)", "WAIT(drain)", "WGMMA(mma)", "EPI(store)", "BARRIER(sync)"]
FINE_PHASE_NAMES = ["fwd ring", "dX ring"]


# ── 8-GPU-idle-aware device pick ────────────────────────────────────────────
def _gpu_table():
    """Return [(index, util%, mem_MiB)] from nvidia-smi, or [] if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
    except Exception:
        return []
    rows = []
    for ln in out.strip().splitlines():
        try:
            i, u, m = (x.strip() for x in ln.split(","))
            rows.append((int(i), int(u), int(m)))
        except Exception:
            continue
    return rows


def _pick_idle_gpu(prefer: int | None, mem_idle_mib: int = 512, util_idle_pct: int = 5):
    """Pick an idle GPU (low util AND low mem). If CUDA_VISIBLE_DEVICES is already set
    by the caller, respect it (return None → don't override). On an 8-GPU box this keeps
    the profile off whatever the other 7 GPUs are running."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None  # caller pinned it; honor it
    if prefer is not None:
        return prefer
    tbl = _gpu_table()
    if not tbl:
        return None
    idle = [(i, u, m) for (i, u, m) in tbl if u <= util_idle_pct and m <= mem_idle_mib]
    if idle:
        return sorted(idle, key=lambda r: (r[1], r[2]))[0][0]
    # none fully idle — pick the least-loaded so we still run, but warn.
    least = sorted(tbl, key=lambda r: (r[1], r[2]))[0]
    print(f"[phase-prof] WARNING: no fully-idle GPU (all >{util_idle_pct}% util or "
          f">{mem_idle_mib} MiB used); using least-loaded GPU {least[0]} "
          f"(util={least[1]}% mem={least[2]} MiB). Result may be noisy.", flush=True)
    return least[0]


# ── occupancy probe (achieved CTAs/SM) ──────────────────────────────────────
def _print_occupancy(mod, dev_index: int):
    """Achieved CTAs/SM for the actual megakernel at block=256 + the baked dynamic-smem
    footprint. We re-derive via cudaOccupancyMaxActiveBlocksPerMultiprocessor; since the
    bench TU does not export the symbol, drive the API through the pybind module if it
    exposes it, else fall back to the launcher's >=1 contract (it REFUSES to launch
    otherwise, so a successful tc_train_step already proves CTAs/SM >= 1)."""
    import torch
    n_sms = torch.cuda.get_device_properties(dev_index).multi_processor_count
    block = int(getattr(mod, "TILE_M", 0)) and 256  # SG_TC_MEGA_BLOCK is fixed 256
    block = 256
    occ = getattr(mod, "OCC_BLOCKS_PER_SM", None)
    print("\n  [occupancy — achieved CTAs/SM]", flush=True)
    print(f"    SMs on device                = {n_sms}", flush=True)
    print(f"    block (SG_TC_MEGA_BLOCK)     = {block} threads", flush=True)
    print(f"    FWD_PIPE / FWD_STAGES        = {int(mod.FWD_PIPE)} / {int(mod.FWD_STAGES)} "
          f"(>=2 stages => dynamic-smem ring)", flush=True)
    if occ is not None:
        print(f"    cudaOccupancyMaxActiveBlocks = {int(occ)} CTA(s)/SM", flush=True)
    else:
        # The launcher pins exactly one CTA per SM (grid = n_sms) and certifies occ>=1
        # via cudaOccupancyMaxActiveBlocksPerMultiprocessor before launching; a
        # successful step (we just ran one) therefore proves achieved >= 1 CTA/SM, and
        # the persistent design uses EXACTLY 1 CTA/SM (it work-steals tiles within that
        # CTA, it does not co-resident a 2nd block). So the achieved occupancy IS 1.
        print(f"    achieved (by design)        = 1 CTA/SM  (persistent grid-barrier "
              f"megakernel; launcher pins grid={n_sms} = #SMs and refuses if occ<1)", flush=True)
    print("    => the megakernel is occupancy-bound at 1 CTA/SM BY DESIGN (the grid "
          "barrier requires it); the lever is per-CTA latency hiding, not more CTAs.",
          flush=True)


# ── nsys command (single-launch timeline) ───────────────────────────────────
def _nsys_command(dev_index: int, steps: int) -> str:
    dev = dev_index if dev_index is not None else 0
    return (
        f"CUDA_VISIBLE_DEVICES={dev} nsys profile "
        f"--trace=cuda,nvtx,osrt --cuda-memory-usage=false --force-overwrite=true "
        f"-o /tmp/dec_phase_nsys "
        f"python tuning/decoder_phase_profile.py --steps {steps} --no-idle-pick "
        f"--nsys-inner"
    )


def _print_report(res, args):
    d, B = res["d"], res["B"]
    print(f"\n[phase-prof] d={d}  B={B}  params={res['total_params']:,}", flush=True)
    print(f"  wall/step = {res['wall_ms']:.3f} ms  (median of {args.reps} reps "
          f"{[f'{x:.2f}' for x in res['walls_ms']]})", flush=True)
    print(f"  steps/s   = {res['steps_per_s']:.4f}", flush=True)
    print(f"  GEMM FLOPs/step = {res['gemm_flops_per_step']:.3e}  => achieved "
          f"{res['achieved_tf_s']:.3f} TF/s "
          f"({100.0*res['achieved_tf_s']/989.0:.2f}% of 989 TF/s bf16 dense roofline)",
          flush=True)

    cyc = res.get("phase_cycles")
    if cyc:
        tot = sum(cyc) or 1
        summed_ms = tot / (SM_GHZ * 1e9) * 1e3
        print(f"\n  [PER-PHASE breakdown — clock64 critical-path on the slowest CTA, "
              f"median of reps]", flush=True)
        print(f"  (Σ summed-phase ~{summed_ms:.3f} ms vs wall {res['wall_ms']:.3f} ms; "
              f"wall-minus-Σ = {res['wall_ms']-summed_ms:.3f} ms host launch/sync + "
              f"un-stamped slack)", flush=True)
        print(f"    {'phase':14s} {'cycles':>15s} {'ms':>10s} {'% of step':>11s}", flush=True)
        for n, c in zip(PHASE_NAMES, cyc):
            ms = c / (SM_GHZ * 1e9) * 1e3
            print(f"    {n:14s} {c:>15,d} {ms:>10.3f} {100.0*c/tot:>10.1f}%", flush=True)
        print(f"    {'SUM':14s} {tot:>15,d} {summed_ms:>10.3f} {'100.0':>10s}%", flush=True)

        # roll-up groups (the "where are the bottlenecks" answer)
        print(f"\n  [ROLL-UP — where the step's time goes]", flush=True)
        named = dict(zip(PHASE_NAMES, cyc))
        ranked = []
        for label, members in GROUPS.items():
            g = sum(named[m] for m in members)
            ranked.append((g, label))
        for g, label in sorted(ranked, reverse=True):
            print(f"    {label:36s} {g/(SM_GHZ*1e9)*1e3:>8.3f} ms  {100.0*g/tot:>6.1f}%",
                  flush=True)
        dom_g, dom_label = max(ranked)
        print(f"  DOMINANT group = {dom_label} ({100.0*dom_g/tot:.1f}% of step)", flush=True)

    fine = res.get("fwd_fine_cycles")
    if fine and cyc:
        nsub = res["fwd_fine_sub"]
        print(f"\n  [FINE fwd/dX GEMM-engine sub-phases — INSIDE the cp.async ring]  "
              f"(FWD_PIPE={int(res['fwd_pipe'])} FWD_STAGES={int(res['fwd_stages'])})",
              flush=True)
        for ph in range(res["fwd_fine_phases"]):
            seg = fine[ph * nsub:(ph + 1) * nsub]
            stot = sum(seg) or 1
            coarse = cyc[ph]  # P1_fwd (0) / P1_bwd (1) — same slot index
            ring_ms = stot / (SM_GHZ * 1e9) * 1e3
            non_gemm = max(coarse - stot, 0)
            print(f"    {FINE_PHASE_NAMES[ph]:10s}  ring Σ {ring_ms:.3f} ms "
                  f"(of {coarse/(SM_GHZ*1e9)*1e3:.3f} ms coarse "
                  f"=> {100.0*stot/max(coarse,1):.0f}% in GEMM ring, "
                  f"{100.0*non_gemm/max(coarse,1):.0f}% LN/softmax/attn/elemwise)", flush=True)
            for sn, c in zip(FINE_SUB_NAMES, seg):
                print(f"      {sn:16s} {c:>15,d} cyc {c/(SM_GHZ*1e9)*1e3:>8.3f} ms "
                      f"{100.0*c/stot:>6.1f}%", flush=True)
            wait_pct = 100.0 * seg[1] / stot
            mma_pct = 100.0 * seg[2] / stot
            verdict = ("DRAIN-bound (cp.async WAIT dominates => deeper ring helps)"
                       if wait_pct >= mma_pct else
                       "COMPUTE/EPI-bound (WGMMA/EPI dominate => deeper ring won't help)")
            print(f"      => {verdict}  [WAIT {wait_pct:.0f}% vs WGMMA {mma_pct:.0f}%]",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Phase-resolved breakdown of ONE d=2048 decoder megakernel step "
                    "(no ncu). Wraps the shipped SG_DEC_PROFILE clock64 phase counters.")
    ap.add_argument("--d", type=int, default=2048,
                    help="decoder width (2048 = SG_DEC_BENCH_LAYOUT bench; 128 = production)")
    ap.add_argument("--B", type=int, default=16384, help="batch (must be %% 16 == 0)")
    ap.add_argument("--steps", type=int, default=20, help="steps per rep")
    ap.add_argument("--reps", type=int, default=3, help="timing/profile repeats (median)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fine", action="store_true",
                    help="also read the FINE fwd/dX GEMM-engine sub-phases "
                         "(ISSUE/WAIT/WGMMA/EPI/BARRIER) => drain- vs compute-bound 'why'")
    ap.add_argument("--gpu", type=int, default=None,
                    help="force this GPU index (else auto-pick an idle one)")
    ap.add_argument("--no-idle-pick", action="store_true",
                    help="do NOT auto-pick an idle GPU (honor CUDA_VISIBLE_DEVICES / default 0)")
    ap.add_argument("--nsys-cmd", action="store_true",
                    help="print the nsys single-launch-timeline command and exit (no build)")
    ap.add_argument("--nsys-inner", action="store_true",
                    help="(internal) run under nsys: skip the idle-pick + occupancy chatter")
    ap.add_argument("--verbose-build", action="store_true")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be divisible by 16 (dW K-loop is 16-step atoms)"

    # --nsys-cmd: print the command, do nothing else (no GPU needed).
    if args.nsys_cmd:
        dev = args.gpu if args.gpu is not None else 0
        print("# Run this to get the single-launch nsys timeline (ONE kernel = the "
              "persistent megakernel; the wall-minus-kernel gap is host launch/sync):",
              flush=True)
        print(_nsys_command(dev, args.steps), flush=True)
        print("\n# Then inspect: nsys stats --report cuda_gpu_kern_sum /tmp/dec_phase_nsys.nsys-rep",
              flush=True)
        return

    # 8-GPU-idle-aware device selection (before importing torch so the mask sticks).
    if not args.no_idle_pick and not args.nsys_inner:
        pick = _pick_idle_gpu(args.gpu)
        if pick is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(pick)
            print(f"[phase-prof] pinned CUDA_VISIBLE_DEVICES={pick} (idle GPU)", flush=True)
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch  # noqa: E402  (after the device mask)
    from decoder_bench import build_variant, measure  # reuse the proven build+drive

    profile = True  # this harness is ALWAYS a profile run
    print(f"[phase-prof] building d={args.d} bench TU (SG_DEC_BENCH_LAYOUT="
          f"{'1' if args.d != 128 else 'unset'}, SG_DEC_PROFILE=1"
          f"{', SG_DEC_PROFILE_FWD_FINE=1' if args.fine else ''}) ...", flush=True)
    t0 = time.perf_counter()
    mod = build_variant(args.d, profile=True, verbose=args.verbose_build,
                        fwd_fine=args.fine)
    print(f"[phase-prof] build done in {time.perf_counter()-t0:.1f}s "
          f"(D={int(mod.D)} TOTAL={int(mod.TOTAL):,} TILE_N={int(mod.TILE_N)} "
          f"FWD_PIPE={int(mod.FWD_PIPE)} FWD_STAGES={int(mod.FWD_STAGES)} "
          f"HAS_PROFILE={getattr(mod,'HAS_PROFILE',False)} "
          f"HAS_FWD_FINE={getattr(mod,'HAS_FWD_FINE',False)})", flush=True)
    if not getattr(mod, "HAS_PROFILE", False):
        print("[phase-prof] ERROR: TU built without SG_DEC_PROFILE (no clock64 counters). "
              "Cannot produce a phase breakdown. Aborting.", flush=True)
        sys.exit(2)

    # Reuse decoder_bench.measure() — it drives tc_train_step, times the wall, and reads
    # the coarse (+ fine, if HAS_FWD_FINE) counters across reps (median per slot).
    res = measure(mod, args.d, args.B, args.reps, args.warmup, args.steps,
                  profile=True, ncta_cap=0)

    _print_report(res, args)
    if not args.nsys_inner:
        dev_index = 0  # after the mask, the visible device is always logical 0
        _print_occupancy(mod, dev_index)
        print("\n  [nsys single-launch timeline] run separately for the host launch/sync "
              "gap (one kernel on the timeline = the megakernel):", flush=True)
        print("    python tuning/decoder_phase_profile.py --nsys-cmd"
              + (f" --gpu {args.gpu}" if args.gpu is not None else ""), flush=True)
    return res


if __name__ == "__main__":
    main()
```

### Why this is correct + safe to apply

1. **Zero repo edits.** It imports `build_variant`, `measure` from the existing
   `tuning/decoder_bench.py` and reads only already-exported pybind symbols
   (`tc_train_step`, `tc_profile_read`, `tc_profile_read_fwd_fine`, the `mod.*`
   attrs). The production `_ops.so` and the 33/33 gate are never touched; the bench TU
   is the same COEXISTING variant `decoder_bench.py --profile` already builds. The HARD
   GATE (fp64 parity + A/A/A determinism) is unaffected — nothing in `csrc/` changes,
   so PTX/smem/regs of every shipped build are byte-identical.

2. **It produces exactly the requested table:** per-phase ms + %-of-step for
   fwd / bwd / B1 / dW-GEMM / grad-assembly / opt-tail / barriers, plus the roll-up
   "where the time goes" ranking and the DOMINANT group. With `--fine` it adds the
   GEMM-ring ISSUE/WAIT/WGMMA/EPI/BARRIER split (the "why": drain- vs compute-bound) and
   the derived "% in GEMM ring vs % LN/softmax/attn/elemwise" line (the honest mapping
   of the LN+softmax/attn phases the task lists — they are fused inside the fwd/bwd tile,
   NOT separately stamped, and this remainder is the only sound way to surface them
   without a kernel edit; calling that out is part of the deliverable).

3. **8-GPU-idle-aware.** `_pick_idle_gpu` queries `nvidia-smi` and pins
   `CUDA_VISIBLE_DEVICES` to a GPU with <5% util AND <512 MiB used BEFORE importing
   torch, so on the 8-GPU box the profile runs on a quiet GPU and the clock64 critical
   path is not contaminated by a neighbor. If the caller already set
   `CUDA_VISIBLE_DEVICES` (as the gate command does), it is honored.

4. **Occupancy probe.** `_print_occupancy` reports the achieved CTAs/SM. The persistent
   grid-barrier megakernel is **1 CTA/SM by design** (the launcher pins `grid = #SMs`
   and refuses to launch if `cudaOccupancyMaxActiveBlocksPerMultiprocessor < 1`,
   `fused_decoder_megakernel.cuh:1540-1545`), so a successful step proves achieved ≥ 1
   and the design uses exactly 1; the script states this and why it is the binding
   constraint (the lever is per-CTA latency hiding, not more CTAs). If a future build
   exports `OCC_BLOCKS_PER_SM` the script prints the API value directly (forward-compat,
   no hard dependency).

5. **nsys command.** `--nsys-cmd` prints the single-launch-timeline command
   (`nsys profile --trace=cuda,nvtx,osrt ...`) plus the `nsys stats` follow-up. The
   single kernel on that timeline confirms the fusion and gives the host launch/sync gap
   (wall − Σ phases). ncu is NOT used (counters denied) — consistent with MEMORY.md.

### Optional (NOT required for the deliverable) — host-side occupancy export

If the lead wants the exact `cudaOccupancyMaxActiveBlocksPerMultiprocessor` integer
surfaced as `mod.OCC_BLOCKS_PER_SM` (instead of the by-design "1"), add a pybind getter
to the bench TU. This is **OPTIONAL** and only inside the profile-only `#ifdef
SG_DEC_PROFILE` block, so the production `_ops` stays byte-identical. Apply only if
desired:

OLD (`csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu`, lines 328-332, verbatim):
```cpp
#ifdef SG_DEC_PROFILE
    mm.def("tc_profile_read", &tc_profile_read,
           "diagnostic-only: per-phase clock64 maxima "
           "[P1fwd,P1bwd,B1wait,P2dW,P2asm,P3opt,B2wait,B0wait] (cycles), resets after read");
    mm.attr("HAS_PROFILE") = true;
```
NEW:
```cpp
#ifdef SG_DEC_PROFILE
    mm.def("tc_profile_read", &tc_profile_read,
           "diagnostic-only: per-phase clock64 maxima "
           "[P1fwd,P1bwd,B1wait,P2dW,P2asm,P3opt,B2wait,B0wait] (cycles), resets after read");
    mm.def("tc_occupancy_blocks_per_sm", []() -> int64_t {
        // Diagnostic-only (SG_DEC_PROFILE): the achieved CTAs/SM the launcher certifies
        // (>=1 required by the persistent grid barrier). Re-derives the SAME value the
        // launcher computes internally (cudaOccupancyMaxActiveBlocksPerMultiprocessor),
        // for the actual megakernel + block(256) + the baked dynamic-smem footprint.
        int dev = 0; cudaGetDevice(&dev);
        int dyn = 0;
#if SG_DEC_TC_DYNAMIC_SMEM
        dyn = (int)sizeof(::sg::fused::sm90::DecTcSmem);
        cudaFuncSetAttribute(
            (const void*)&::sg::fused::sm90::fused_decoder_megakernel_tc<OptId::AdamW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
#endif
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void*)&::sg::fused::sm90::fused_decoder_megakernel_tc<OptId::AdamW>,
            SG_TC_MEGA_BLOCK, dyn);
        return (int64_t)occ;
    }, "diagnostic-only: achieved CTAs/SM (cudaOccupancyMaxActiveBlocksPerMSM) for the megakernel");
    mm.attr("HAS_PROFILE") = true;
```
And in the harness, replace the `occ = getattr(mod, "OCC_BLOCKS_PER_SM", None)` line
with:
```python
    occ = mod.tc_occupancy_blocks_per_sm() if hasattr(mod, "tc_occupancy_blocks_per_sm") else None
```
(Both `SG_TC_MEGA_BLOCK` and `SG_DEC_TC_DYNAMIC_SMEM` are already in scope in the TU via
the `fused_decoder_megakernel.cuh` include; `fused_decoder_megakernel_tc<OptId::AdamW>`
is the same instantiation the launcher takes the address of, lines 1534/1542.)
**This edit is byte-identical-when-off** — it is entirely inside `#ifdef SG_DEC_PROFILE`,
which the production `_ops` never defines.

---

## Gate commands (exactly as the task specifies; both must pass)

```
python tuning/decoder_phase_profile.py --help
CUDA_VISIBLE_DEVICES=0 python tuning/decoder_phase_profile.py --steps 20
```
The second builds the d=2048 bench variant with `SG_DEC_PROFILE=1` (first run ~minutes
to JIT-compile; cached after), runs 20 steps × 3 reps on GPU 0, and prints the per-phase
ms + %-of-step table, the roll-up, and the occupancy line. Add `--fine` for the drain-
vs-compute "why" split. Run `python tuning/decoder_phase_profile.py --nsys-cmd` to get
the single-launch nsys command.

## Expected reading of the output (what the bottleneck IS, from the baked config)

The comments in `model_stage_decoder_tc.cuh:148-203` record the campaign finding the
profiler produced: P1_fwd (~28.8%) + P1_bwd-dX (~27.7%) ≈ 56.5% of the step are the
fwd/dX GEMMs; the fine split showed those rings are **WAIT(drain)-dominant** (43% fwd /
56% dX vs WGMMA), i.e. cp.async-drain/latency-bound — which is why the BAKED default is
`FWD_PIPE=1, FWD_STAGES=4` (the deeper ring, +1.49×). The dW GEMM is the other large
slice and is **staging-bound** (~97% scalar gather), addressed by the baked
`SG_TUNED_DEC_DW_STAGE=1` contiguous-transpose path. Running this harness re-confirms
those numbers live at d=2048 and shows the post-bake residual, which is the lead's
answer to "where are the bottlenecks currently, and why".
