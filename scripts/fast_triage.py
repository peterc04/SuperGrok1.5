"""scripts/fast_triage.py — FAST DIRECTIONAL "is this kernel change better or worse?"
triage for the SuperGrok1.5 perf inner loop.

WHY THIS EXISTS
  The campaign's rate-limiter is the ~20-30 min FULL cycle: rebuild the production
  `_ops` + the 3-seed fp64 PARITY HARD-GATE + A/A/A determinism + a d=2048 3-seed
  timing. That is the right arbiter for a KEEP — but it is far too slow to TRIAGE the
  dozens of candidate `-D SG_TUNED_*` macro-configs the hill-climb proposes. This
  harness gives a ~1-3 min directional signal so the expensive full cycle is reserved
  for the few candidates that survive triage.

  It is a THIN ORCHESTRATOR over the EXISTING machinery — it reinvents NOTHING. The
  --model switch selects which megakernel to triage (decoder DEFAULT | vit); both
  bench modules expose the SAME build_variant(profile=True)/measure(profile=True)->
  res["phase_cycles"] contract, and the relevance-gate logic is model-agnostic:
    * the per-variant build is `<bench>.build_variant(d, profile=True, …)` where <bench>
      is tuning.decoder_bench (mega_decoder_real_adamw_tc.cu) or tuning.vit_bench
      (mega_vit_real_adamw_tc.cu) — the coexisting bench TU built as a SEPARATELY-NAMED
      extension with its own torch-extensions build dir + ninja incremental rebuild +
      sccache. It is ONE cache-friendly TU, NOT the full `_ops`, and it NEVER touches
      the production `_ops.so` or the 33/33 gate.
    * the timing + per-phase clock64 read is `<bench>.measure(..., profile=True)`
      → `res["phase_cycles"]`, mapped by the model's phase table:
        decoder `decoder_bench.PHASE_NAMES` (8-slot g_dec_prof_max):
          [0]=P1_fwd [1]=P1_bwd [2]=B1_barrier [3]=P2_dW_GEMM
          [4]=P2_grad_asm [5]=P3_opt_tail [6]=B2_barrier [7]=B0_barrier
        vit `vit_bench.VIT_PHASE_NAMES` (6-slot g_vit_prof_max):
          [0]=P1_fwd [1]=P1_bwd [2]=B1_barrier [3]=P2_dW_GEMM
          [4]=P2_grad_asm [5]=P3_opt_tail
    * each variant is timed in its OWN worker subprocess (the tune_optimizers.py:645
      `subprocess.Popen([sys.executable, "-c", script])` idiom) so the baseline and the
      candidate get CLEAN, ISOLATED CUDA contexts — a wedged/OOM build of one variant
      cannot poison the other, and the warm context lives only as long as that variant.

PROFILING DOCTRINE  (measure broad · fix narrow · zoom deep only where you dig)
  * MEASURE BROAD, ALWAYS: every run reads ALL 8 phases in one clock64 shot. It is FREE
    (one read gets the whole step) and it is the ONLY way to stay honest — you cannot tell
    whether the phase you changed even MATTERS without every other phase as the denominator
    (its SHARE), and broad coverage is what catches the bottleneck MOVING (dW 60%->17% after
    the staging fix => the prize is now fwd/bwd). Never profile one phase alone and fix blind.
  * FOCUS THE FIX NARROW: attack ONE bottleneck phase per round (the highest-share one),
    even though you MEASURE all of them.
  * ZOOM DEEP ONLY WHERE YOU DIG: to find the sub-lever inside the phase under attack, add
    FINE sub-phase counters to THAT phase TEMPORARILY (e.g. split P1_fwd -> stage/MMA/
    epilogue), then remove them. Fine counters everywhere perturb what they measure + clutter.
    Resolution follows where you work.

WHAT IT REPORTS
  baseline (-D set B) vs candidate (-D set C), at the ROOFLINE scale by default
  (d=2048/B=16384 — see "HONEST SCOPE"), profile-built, ALL phases read:
    * the full per-phase breakdown (baseline | candidate | Δ% | share-of-step) — broad read;
    * the TARGETED-PHASE delta (e.g. a dW change is judged on P2_dW_GEMM) — the mechanism;
    * the RELEVANCE GATE: the targeted phase's SHARE of the step + the projected step Δ
      (= share × phase-Δ). Below --relevance-floor the phase is TOO SMALL TO MATTER and the
      verdict is IRRELEVANT no matter how good the phase-local Δ looks (the vec4-AdamW guard);
    * the TOTAL-step (wall) delta — the bottom-line "did the step get faster?", the arbiter
      of this screen, cross-checked against the projected step Δ to flag side effects;
    * a one-line verdict, STEP-level first:  TRIAGE: step <better|worse|neutral> M% (X.XX×);
      <phase> <verdict> N% [share% -> projects P%].
  Lower ms = better. NEUTRAL band is configurable (--neutral-pct, default 3%).

PHASE AUTO-TARGETING
  If --phase is omitted, the harness infers the targeted phase from the candidate's
  macro names (the phase the change is *meant* to move), so a dW staging/split-K change
  is auto-judged on P2_dW_GEMM:
    SG_TUNED_DEC_DW_*            -> P2_dW_GEMM
    SG_TUNED_DEC_GEMM_* / PIPE_* -> P1_fwd  (fwd/dX GEMM; reported with P1_bwd alongside)
    SG_TUNED_*OPT* / *ADAMW*     -> P3_opt_tail
    SG_TUNED_TILE_* (ambiguous)  -> P2_dW_GEMM + P1_fwd both shown, wall is the arbiter
  Override with --phase P2_dW_GEMM (or any PHASE_NAMES entry) when you know better.

────────────────────────────────────────────────────────────────────────────────────
EXPECTED PER-CANDIDATE TIME  (the whole point):
  The speedup over the full cycle comes from CUTTING THE EXPENSIVE PARTS — the 3-seed
  fp64/AAA gate and the full `_ops` build — NOT from shrinking the problem. We stay at the
  roofline scale so the phase-shares (hence the relevance gate) stay HONEST (PROFILING
  DOCTRINE); the kernel itself is cheap to run.
  * BUILD: the bench TU rebuilds INCREMENTALLY (ninja + sccache). First build of a given
    define-set is a cold compile (~140-160 s at d=2048). Once the cache is warm, a new
    candidate that only flips a `-D` value reuses cached objects.
  * TIME: profile-build measure() at the default d=2048/B=16384 (reps=2, iters=6) is
    ~15-30 s of GPU per variant (the kernel is ~0.92 s/step; ONE seed, not three).
  ⇒ a triage is build-dominated; the GPU time is seconds and there is NO 3-seed gate. The
    full cycle it replaces (rebuild + 3-seed fp64/AAA gate + 3-seed timing) is ~20-30 min.

────────────────────────────────────────────────────────────────────────────────────
‼  HONEST SCOPE — THIS IS A TRIAGE, NOT A VERDICT  ‼
  * It is a DIRECTIONAL signal, not a roofline number. It is faithful on DIRECTION (the
    sign of the per-phase + wall delta), not on the precise magnitude.
  * It runs at the ROOFLINE scale by default (d=2048/B=16384) so the phase-shares are the
    REAL ones (PROFILING DOCTRINE). You MAY pass a smaller --B for a faster screen, but then
    grid-fill, occupancy, and the shares drift from production — a DIRECTION-ONLY signal at
    best (the P0 lesson: pipelining looked fine at d=128 and regressed at d=2048). Keep the
    default unless you specifically want a quick direction smoke.
  * It profiles with ONE seed (the phase breakdown is data-INDEPENDENT — the kernel does
    identical work every step regardless of the RNG seed — so shares are seed-invariant;
    reps median out timing jitter, not seeds). The 3-seed sweep belongs to the VERDICT, for
    timing confidence AND the A/A/A determinism gate (a CORRECTNESS check the profile cannot
    do — e.g. the seed/scale-dependent IL=4 non-determinism).
  * It can MISLEAD on FUSED-STEP INTERACTIONS: a change that shrinks its own phase can
    still lose at the step level via OCCUPANCY (e.g. the reverted P0 pipelined-GEMM:
    depth-3/4 would not even launch — occupancy<1 — and depth-2 regressed the wall
    because the producer/consumer split slowed the real staging bottleneck). The
    per-phase signal alone would not have predicted that; the WALL delta is your guard,
    and even it is approximate at small scale.
  * It CANNOT replace the fp64 PARITY hard-gate + A/A/A DETERMINISM gate. A candidate
    can be bench-FASTER and gate-FAIL: the decoder dW GEMM_INTERLEAVE=4 was the fastest
    lever found (−9.3% bench) yet FAILED A/A/A determinism on all 10 cells (ragged
    M-atom groups at the toy d → non-deterministic group-reduce). The bench missed it;
    the gate caught it. So:
        fast_triage  =  "is it worth spending a full cycle on?"
        full cycle   =  fp64 parity + A/A/A determinism + 3-seed d=2048 time  = THE ARBITER.
    NEVER KEEP on a triage signal. A green triage earns a candidate the EXPENSIVE,
    AUTHORITATIVE full cycle; a red triage saves that cycle. The patch-protocol ratchet
    (apply → build → gate → 3-seed time → KEEP iff faster on 3+ seeds AND parity-clean,
    else REVERT) is unchanged and remains the only thing that lands a KEEP.

────────────────────────────────────────────────────────────────────────────────────
VALIDATION RECIPE — PROVE THE HARNESS IS FAITHFUL AGAINST THE KNOWN dW 2×
  The dW contiguous-layout staging is the campaign's measured ground truth
  (OPTIMIZATION_LEDGER.md, "dW contiguous-layout staging: KEEP +2.05×"):
      stage=0 / splitk=1 (scalar)      = 1889.8 ms  @ d=2048/B=16384, 3 seeds
      stage=1 / splitk=1 (contiguous)  =  920.7 ms  =>  2.05× faster
  A faithful triage MUST show: P2_dW_GEMM shrinks dramatically AND the wall ≈ halves
  when going stage=0 -> stage=1 at splitk=1. If it does NOT reproduce that direction,
  the harness is UNTRUSTWORTHY — say so and do not rely on it.

  Run the built-in validator (the main loop runs this ON THE GPU; CPU authoring does
  not):

    # Authoritative — at the roofline scale the 2.05× was measured at (slow, ~minutes):
    CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. \
      python scripts/fast_triage.py --validate --d 2048 --B 16384 --reps 3

    # Quick faithfulness smoke at the fast scale (direction only; magnitude differs):
    CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. \
      python scripts/fast_triage.py --validate

  --validate is exactly:
      baseline  = -D SG_TUNED_DEC_DW_STAGE=0 -D SG_TUNED_DEC_DW_SPLITK=1   (scalar)
      candidate = -D SG_TUNED_DEC_DW_STAGE=1 -D SG_TUNED_DEC_DW_SPLITK=1   (contiguous)
      phase     = P2_dW_GEMM
  and it asserts the candidate is FASTER on BOTH P2_dW_GEMM and the wall (and, at
  --d 2048, that the wall ratio is in a ~1.5×–2.6× window around the measured 2.05×).
  Exit 0 = harness FAITHFUL; exit 2 = harness reproduced the WRONG direction (untrustworthy).

────────────────────────────────────────────────────────────────────────────────────
TRIAGE A CANDIDATE (the normal use):
    # judge a dW split-K change against the production default, fast scale:
    CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python scripts/fast_triage.py \
        --baseline-D SG_TUNED_DEC_DW_STAGE=1 --baseline-D SG_TUNED_DEC_DW_SPLITK=1 \
        --cand-D     SG_TUNED_DEC_DW_STAGE=1 --cand-D     SG_TUNED_DEC_DW_SPLITK=2
    # explicit phase + closer-to-roofline triage:
    ... --phase P2_dW_GEMM --d 2048 --B 16384 --reps 3

  --baseline-D defaults to EMPTY (the in-header production defaults: stage=1, splitk=1,
  IL=2, …) when not given, so the common case is just a set of --cand-D overrides vs
  the shipped config.

JSON: pass --json for a machine-readable summary line (TRIAGE_JSON {...}) for the
hill-climb driver to parse; the human report always prints too.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The phase-name table is the single source of truth for the clock64 slot order; it
# lives in decoder_bench and matches mega_decoder_real_adamw_tc.cu's g_dec_prof_max.
from tuning.decoder_bench import PHASE_NAMES, SM_GHZ  # noqa: E402
# ViT phase-name table (g_vit_prof_max [6 slots]); lives in vit_bench and matches
# fused_vit_megakernel.cuh's stamping sites. Imported lazily-safe at module load so a
# decoder-only triage never pays the ViT import cost difference (both are pure-Python
# at import). The --model switch selects which table + bench module drive the run.
from tuning.vit_bench import VIT_PHASE_NAMES  # noqa: E402

# ── per-model wiring: which coexisting bench module + profile macro + clock64 phase
#    table a triage drives. The decoder is the default + the validated reference; vit
#    mirrors it (vit_bench.build_variant(profile=True)/measure(profile=True) expose the
#    SAME res["phase_cycles"] contract). The relevance-gate machinery below is model-
#    AGNOSTIC — it consumes phase_cycles + a phase-name table, nothing decoder-specific.
_MODELS = {
    "decoder": {
        "bench": "tuning.decoder_bench",
        "profile_macro": "SG_DEC_PROFILE",
        "phase_names": PHASE_NAMES,
        "d_default": 2048,
        "d_valid": (128, 2048),
        "d_help": ("decoder width. The bench has exactly TWO layouts: 2048 (roofline "
                   "scale — DEFAULT, faithful shares) and 128 (production toy). There is "
                   "NO d=1024 layout — it compiles to D=2048 and trips the mod.D==d assert."),
    },
    "vit": {
        "bench": "tuning.vit_bench",
        "profile_macro": "SG_VIT_PROFILE",
        "phase_names": VIT_PHASE_NAMES,
        "d_default": 1024,
        "d_valid": (128, 1024),
        "d_help": ("ViT width. The bench has TWO layouts: 1024 (size-ladder roofline "
                   "scale — DEFAULT) and 128 (production). --d selects the layout branch."),
    },
}

# ── phase auto-targeting: which clock64 phase a -D macro is MEANT to move ─────────
# (substring match on the macro KEY, first hit wins; order = specificity). The needles
# are model-agnostic: P2_dW_GEMM / P1_fwd / P3_opt_tail exist in BOTH the decoder and
# ViT phase tables, and the ViT macro keys (SG_TUNED_VIT_DW_*, SG_TUNED_VIT_GEMM_*, …)
# carry the same DW_/GEMM_/OPT/TILE_ substrings, so one table serves both models.
_PHASE_HINTS = [
    ("DEC_DW", "P2_dW_GEMM"),       # dW staging / split-K / dW-specific knobs (decoder)
    ("VIT_DW", "P2_dW_GEMM"),       # dW knobs (ViT)
    ("DW_", "P2_dW_GEMM"),
    ("PIPE", "P1_fwd"),             # pipelined GEMM ring (fwd/dX path)
    ("GEMM_INTERLEAVE", "P1_fwd"),  # fwd/dX interleave (dW IL shows on P2 too — wall arbitrates)
    ("GEMM_STAGES", "P1_fwd"),
    ("ADAMW", "P3_opt_tail"),
    ("OPT", "P3_opt_tail"),
    ("TILE_", "P2_dW_GEMM"),        # tile geometry touches every GEMM; default to dW, show fwd too
]


def _infer_phase(cand_defines: list[str], phase_names: list[str] = PHASE_NAMES) -> str:
    """Infer the targeted phase from the candidate macro KEYs. Defaults to the wall-
    dominant P2_dW_GEMM when nothing matches (the roofline step is P1+P2 dominated).
    `phase_names` is the active model's slot table (decoder by default); the inferred /
    default phase is guaranteed to be a member of it (P2_dW_GEMM is in both tables)."""
    keys = [d.split("=", 1)[0].lstrip("-D").strip().upper() for d in cand_defines]
    for needle, phase in _PHASE_HINTS:
        if any(needle in k for k in keys) and phase in phase_names:
            return phase
    return "P2_dW_GEMM" if "P2_dW_GEMM" in phase_names else phase_names[0]


# ── the worker body: build ONE bench variant in an isolated CUDA context, profile-
#    time it, print a single TRIAGE_RESULT JSON line. Driven via `python -c`, the
#    tune_optimizers.py:645 persistent-subprocess idiom (one warm context per variant).
#    `{bench}` is the coexisting bench module (tuning.decoder_bench | tuning.vit_bench),
#    each exposing the SAME build_variant(d, profile=True, defines=…)/measure(…,
#    profile=True, ncta_cap=…)->res["phase_cycles"] contract. `{profile_macro}` only
#    labels the HAS_PROFILE assert message.
_WORKER = r"""
import json, sys
from pathlib import Path
ROOT = Path({root!r})
sys.path.insert(0, str(ROOT))
import {bench} as db

d, B, reps, warmup, iters, ncta = {d}, {B}, {reps}, {warmup}, {iters}, {ncta}
defines = {defines!r}
try:
    mod = db.build_variant(d, profile=True, defines=defines)
    assert int(mod.D) == d, "TU D=%d != %d" % (int(mod.D), d)
    assert getattr(mod, "HAS_PROFILE", False), "variant built without -D{profile_macro}"
    res = db.measure(mod, d, B, reps=reps, warmup=warmup, iters=iters,
                     profile=True, ncta_cap=ncta)
    out = {{
        "ok": True,
        "wall_ms": res["wall_ms"],
        "walls_ms": res.get("walls_ms", []),
        "achieved_tf_s": res["achieved_tf_s"],
        "phase_cycles": res.get("phase_cycles"),
        "total_params": res["total_params"],
        "tile_n": int(getattr(mod, "TILE_N", 0)),
    }}
except Exception as e:  # noqa: BLE001
    import traceback
    out = {{"ok": False, "error": repr(e), "traceback": traceback.format_exc()}}
print("TRIAGE_RESULT " + json.dumps(out), flush=True)
"""


def _run_variant(label: str, defines: list[str], args, env_passthrough: bool = True) -> dict:
    """Build+profile-time one variant in an isolated subprocess. Returns the parsed
    TRIAGE_RESULT dict augmented with build wall + label."""
    mcfg = _MODELS[getattr(args, "model", "decoder")]
    body = _WORKER.format(
        root=str(ROOT), d=args.d, B=args.B, reps=args.reps,
        warmup=args.warmup, iters=args.iters, ncta=args.ncta_cap,
        defines=defines, bench=mcfg["bench"], profile_macro=mcfg["profile_macro"],
    )
    print(f"[triage] building+profiling variant '{label}'  defines={defines or '(prod defaults)'}",
          flush=True)
    t0 = time.perf_counter()
    # cwd=ROOT so the bench's torch-extensions build dir + sccache cache are shared
    # across variants (incremental ninja). Inherit env (CUDA_VISIBLE_DEVICES,
    # CUDA_MPS_*, TORCH_EXTENSIONS_DIR, …) from the caller verbatim.
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", body],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("TRIAGE_RESULT "):
            result = json.loads(line[len("TRIAGE_RESULT "):])
        elif args.verbose:
            sys.stdout.write("    | " + line)
    proc.wait()
    wall_build = time.perf_counter() - t0
    if result is None:
        return {"ok": False, "label": label, "defines": defines,
                "error": f"worker produced no TRIAGE_RESULT (exit {proc.returncode}); "
                         f"re-run with --verbose to see the build/run log",
                "build_s": round(wall_build, 1)}
    result["label"] = label
    result["defines"] = defines
    result["build_s"] = round(wall_build, 1)
    if result.get("ok"):
        print(f"[triage]   '{label}' done in {wall_build:.1f}s  "
              f"wall={result['wall_ms']:.3f} ms  {result['achieved_tf_s']:.3f} TF/s",
              flush=True)
    else:
        print(f"[triage]   '{label}' FAILED in {wall_build:.1f}s: {result.get('error')}",
              flush=True)
    return result


def _phase_ms(res: dict, phase: str, phase_names: list[str] = PHASE_NAMES) -> float | None:
    cyc = res.get("phase_cycles")
    if not cyc:
        return None
    idx = phase_names.index(phase)
    return cyc[idx] / (SM_GHZ * 1e9) * 1e3


def _pct_delta(base: float, cand: float) -> float:
    """Signed % change of candidate vs baseline. NEGATIVE = candidate is SMALLER/FASTER
    (good). +X% = candidate is X% slower."""
    if base == 0:
        return float("nan")
    return 100.0 * (cand - base) / base


def _verdict(delta_pct: float, neutral_pct: float) -> str:
    if delta_pct != delta_pct:  # NaN
        return "unknown"
    if delta_pct <= -neutral_pct:
        return "better"
    if delta_pct >= neutral_pct:
        return "worse"
    return "neutral"


def _report(base: dict, cand: dict, phase: str, neutral_pct: float,
            relevance_floor: float = 5.0, phase_names: list[str] = PHASE_NAMES) -> dict:
    """Build + print the human triage report. Returns the JSON summary dict.
    `phase_names` is the active model's clock64 slot table (decoder by default)."""
    print("\n" + "=" * 78, flush=True)
    print("[triage] RESULT", flush=True)
    print("=" * 78, flush=True)

    if not (base.get("ok") and cand.get("ok")):
        # surface the failing side so the loop knows whether it's a build break.
        for r in (base, cand):
            if not r.get("ok"):
                print(f"  {r['label']} build/run FAILED: {r.get('error')}", flush=True)
        summary = {"verdict": "BUILD_FAILED",
                   "baseline_ok": bool(base.get("ok")), "candidate_ok": bool(cand.get("ok")),
                   "baseline_error": base.get("error"), "candidate_error": cand.get("error")}
        print("\n  TRIAGE: BUILD_FAILED — cannot judge (a -D set failed to build/run).",
              flush=True)
        return summary

    # full per-phase table (baseline | candidate | delta), in clock64-slot order.
    bp = base["phase_cycles"]
    cp = cand["phase_cycles"]
    bt, ct = sum(bp) or 1, sum(cp) or 1
    print(f"  params={base['total_params']:,}  TILE_N={base.get('tile_n', '?')}", flush=True)
    print(f"  baseline wall={base['wall_ms']:.3f} ms ({base['achieved_tf_s']:.3f} TF/s)   "
          f"candidate wall={cand['wall_ms']:.3f} ms ({cand['achieved_tf_s']:.3f} TF/s)",
          flush=True)
    print(f"\n  {'phase':14s} {'baseline ms':>13s} {'candidate ms':>14s} {'Δ%':>9s}  "
          f"{'(b%/c% of step)':>16s}", flush=True)
    for i, name in enumerate(phase_names):
        b_ms = bp[i] / (SM_GHZ * 1e9) * 1e3
        c_ms = cp[i] / (SM_GHZ * 1e9) * 1e3
        dlt = _pct_delta(b_ms, c_ms)
        star = "  <-- targeted" if name == phase else ""
        print(f"  {name:14s} {b_ms:>13.3f} {c_ms:>14.3f} {dlt:>+8.1f}%  "
              f"{100.0*bp[i]/bt:>6.1f}%/{100.0*cp[i]/ct:>5.1f}%{star}", flush=True)

    # the two signals that matter: targeted phase + wall.
    b_phase = _phase_ms(base, phase, phase_names)
    c_phase = _phase_ms(cand, phase, phase_names)
    phase_delta = _pct_delta(b_phase, c_phase)
    wall_delta = _pct_delta(base["wall_ms"], cand["wall_ms"])
    phase_v = _verdict(phase_delta, neutral_pct)
    wall_v = _verdict(wall_delta, neutral_pct)
    # speedup factor on the wall (>1 = faster), the ledger's preferred framing.
    wall_speedup = base["wall_ms"] / cand["wall_ms"] if cand["wall_ms"] else float("nan")

    # ── RELEVANCE GATE (Amdahl) — "don't optimize what we don't care about" ──────────
    # phase-share = the targeted phase's fraction of the profiled step. The STEP can move
    # by at most that share (the phase fully eliminated), so:
    #   projected step Δ = phase_share × phase_delta   (the mechanism cross-check vs wall)
    #   max plausible step gain = −phase_share          (phase → 0)
    # Below the floor the phase is TOO SMALL TO MATTER — IRRELEVANT — no matter how good
    # its local Δ looks (the canonical trap: vec4-AdamW was bench-faster on a P3 tail that
    # is <1% of the step → real benchmark NEUTRAL). This is only meaningful because the
    # profile is at the ROOFLINE SCALE — at a shrunk d/B the shares (hence this gate) lie.
    b_share = 100.0 * bp[phase_names.index(phase)] / bt   # % of step in targeted phase
    projected_step_pct = (b_share / 100.0) * phase_delta
    max_step_gain_pct = -b_share
    relevant = b_share >= relevance_floor
    # consistency: did the wall move as the share predicts? A shortfall means the change
    # hurt ANOTHER phase (the P0 trap: dW phase shrank but the wall regressed because the
    # producer/consumer split stole threads from the real staging bottleneck).
    consistent, consistency_note = True, ""
    if relevant and abs(projected_step_pct) >= neutral_pct:
        if wall_delta > projected_step_pct + max(neutral_pct, 0.5 * abs(projected_step_pct)):
            consistent = False
            consistency_note = (f"wall {wall_delta:+.1f}% fell short of projected "
                                f"{projected_step_pct:+.1f}% → likely hurt another phase")

    print(f"\n  targeted phase = {phase}:  {b_phase:.3f} -> {c_phase:.3f} ms  "
          f"({phase_delta:+.1f}%, {phase_v})", flush=True)
    print(f"  total step (wall):         {base['wall_ms']:.3f} -> {cand['wall_ms']:.3f} ms  "
          f"({wall_delta:+.1f}%, {wall_v};  {wall_speedup:.3f}× speedup)", flush=True)
    print(f"  RELEVANCE: {phase} = {b_share:.1f}% of step (floor {relevance_floor:.1f}%) → "
          f"{'RELEVANT' if relevant else 'TOO SMALL TO MATTER'}; "
          f"projected step Δ {projected_step_pct:+.1f}% "
          f"(ceiling {max_step_gain_pct:+.1f}% if eliminated)", flush=True)
    if not consistent:
        print(f"  ⚠ INCONSISTENT: {consistency_note}.", flush=True)

    # one-line verdict — STEP-LEVEL first (the decision), gated by relevance; the phase
    # delta is the mechanism, not the verdict. Lower ms = better; negative Δ% = improvement.
    if not relevant:
        headline = (f"TRIAGE: IRRELEVANT — {phase} is only {b_share:.1f}% of the step; "
                    f"its {phase_delta:+.1f}% projects to ≤{abs(max_step_gain_pct):.1f}% step. "
                    f"Not worth a full cycle (wall {wall_v} {abs(wall_delta):.1f}%).")
    else:
        headline = (f"TRIAGE: step {wall_v} by {abs(wall_delta):.1f}% ({wall_speedup:.2f}×); "
                    f"{phase} {phase_v} {abs(phase_delta):.1f}% [{b_share:.0f}% of step → "
                    f"projects {projected_step_pct:+.1f}%]"
                    + ("" if consistent else "  ⚠ wall<projected: side-effect"))
    print(f"\n  {headline}", flush=True)
    print("  NOTE: triage only — NOT a KEEP. The fp64 parity + A/A/A gate + 3-seed "
          "d=2048 timing remain the arbiter.", flush=True)

    return {
        "verdict_phase": phase_v, "verdict_wall": wall_v,
        "targeted_phase": phase,
        "phase_share_pct": round(b_share, 2),
        "projected_step_pct": round(projected_step_pct, 2),
        "max_step_gain_pct": round(max_step_gain_pct, 2),
        "relevant": relevant,
        "relevance_floor_pct": relevance_floor,
        "consistent": consistent,
        "phase_delta_pct": round(phase_delta, 2),
        "wall_delta_pct": round(wall_delta, 2),
        "wall_speedup": round(wall_speedup, 4),
        "baseline_wall_ms": round(base["wall_ms"], 4),
        "candidate_wall_ms": round(cand["wall_ms"], 4),
        "baseline_phase_ms": round(b_phase, 4),
        "candidate_phase_ms": round(c_phase, 4),
        "baseline_tf_s": base["achieved_tf_s"],
        "candidate_tf_s": cand["achieved_tf_s"],
        "neutral_pct": neutral_pct,
    }


def _do_validate(args) -> int:
    """Reproduce the known dW 2× to prove the harness is faithful. Returns process
    exit code: 0 = faithful, 2 = reproduced the WRONG direction (untrustworthy).
    DECODER-specific (the reference KEEP is a decoder dW staging win)."""
    if getattr(args, "model", "decoder") != "decoder":
        print(f"[triage --validate] only the decoder has a measured-KEEP reference "
              f"(the dW 2.05× staging win); --model {args.model} has no validation "
              f"fixture. Run --validate without --model (or --model decoder).", flush=True)
        return 1
    print("=" * 78, flush=True)
    print("[triage --validate] FAITHFULNESS CHECK vs the measured dW 2.05× KEEP", flush=True)
    print("  baseline  = SG_TUNED_DEC_DW_STAGE=0 splitk=1  (scalar; measured 1889.8 ms @ d=2048)",
          flush=True)
    print("  candidate = SG_TUNED_DEC_DW_STAGE=1 splitk=1  (contiguous; measured  920.7 ms = 2.05×)",
          flush=True)
    print(f"  scale     = d={args.d} B={args.B} reps={args.reps}  "
          f"(the DEFAULT d=2048/B=16384 reproduces the 2.05× MAGNITUDE; pass a smaller "
          f"--B for a faster direction-only smoke)", flush=True)
    print("=" * 78, flush=True)

    base = _run_variant("scalar(stage=0)",
                        ["SG_TUNED_DEC_DW_STAGE=0", "SG_TUNED_DEC_DW_SPLITK=1"], args)
    cand = _run_variant("contiguous(stage=1)",
                        ["SG_TUNED_DEC_DW_STAGE=1", "SG_TUNED_DEC_DW_SPLITK=1"], args)
    summary = _report(base, cand, "P2_dW_GEMM", args.neutral_pct, args.relevance_floor)

    if summary.get("verdict") == "BUILD_FAILED":
        print("\n[triage --validate] INCONCLUSIVE — a variant failed to build/run "
              "(not a faithfulness verdict). Re-run with --verbose.", flush=True)
        if args.json:
            print("TRIAGE_JSON " + json.dumps({"validate": "INCONCLUSIVE", **summary}), flush=True)
        return 1

    phase_d = summary["phase_delta_pct"]
    wall_d = summary["wall_delta_pct"]
    speedup = summary["wall_speedup"]
    # Faithful iff BOTH the dW phase and the wall got materially FASTER (negative Δ).
    dir_ok = (phase_d < -args.neutral_pct) and (wall_d < -args.neutral_pct)
    # At the roofline scale ALSO require the wall speedup to land near the measured 2.05×
    # (a generous 1.5×–2.6× window — magnitude, not just sign).
    mag_ok = True
    mag_checked = (args.d >= 2048 and args.B >= 16384)
    if mag_checked:
        mag_ok = 1.5 <= speedup <= 2.6

    faithful = dir_ok and mag_ok
    print("\n" + "=" * 78, flush=True)
    if faithful:
        print(f"[triage --validate] ✅ FAITHFUL: the harness reproduced the dW win — "
              f"P2_dW_GEMM {phase_d:+.1f}%, wall {wall_d:+.1f}% ({speedup:.2f}×).",
              flush=True)
        if mag_checked:
            print(f"  magnitude check (d>=2048): {speedup:.2f}× is within the 1.5–2.6× "
                  f"window around the measured 2.05×.", flush=True)
        else:
            print("  (direction-only at the fast scale; run --validate --d 2048 --B 16384 "
                  "to also check the 2.05× magnitude.)", flush=True)
        rc = 0
    else:
        print("[triage --validate] ❌ UNFAITHFUL: the harness did NOT reproduce the known "
              "dW direction/magnitude.", flush=True)
        if not dir_ok:
            print(f"  DIRECTION WRONG: expected BOTH dW phase and wall to shrink; got "
                  f"P2_dW_GEMM {phase_d:+.1f}%, wall {wall_d:+.1f}%.", flush=True)
        if mag_checked and not mag_ok:
            print(f"  MAGNITUDE OFF: wall speedup {speedup:.2f}× is outside the 1.5–2.6× "
                  f"window around the measured 2.05× (at the roofline scale).", flush=True)
        print("  => The harness is UNTRUSTWORTHY for triage. Investigate before relying on it "
              "(check: profile build flag, phase-slot mapping, that stage=1 actually compiled "
              "the contiguous path, and the bench-variant cache is not stale).", flush=True)
        rc = 2
    print("=" * 78, flush=True)
    if args.json:
        print("TRIAGE_JSON " + json.dumps({"validate": "FAITHFUL" if faithful else "UNFAITHFUL",
                                           "direction_ok": dir_ok, "magnitude_ok": mag_ok,
                                           "magnitude_checked": mag_checked, **summary}),
              flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fast directional better/worse triage for SuperGrok1.5 kernel "
                    "macro-configs (NOT a KEEP gate — see the module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="run the dW-2× faithfulness check instead of a candidate triage "
                         "(exit 0=faithful, 2=untrustworthy). DECODER-only fixture.")
    ap.add_argument("--model", choices=tuple(_MODELS), default="decoder",
                    help="which megakernel to triage: 'decoder' (DEFAULT, the validated "
                         "reference) or 'vit'. Selects the coexisting bench module + the "
                         "clock64 phase table; the relevance-gate logic is model-agnostic.")
    ap.add_argument("--baseline-D", dest="baseline_defines", action="append", default=[],
                    metavar="KEY=VAL",
                    help="-D override for the BASELINE variant (repeatable). Empty = the "
                         "in-header production defaults (stage=1, splitk=1, IL=2, …).")
    ap.add_argument("--cand-D", dest="cand_defines", action="append", default=[],
                    metavar="KEY=VAL",
                    help="-D override for the CANDIDATE variant (repeatable), e.g. "
                         "--cand-D SG_TUNED_DEC_DW_SPLITK=2.")
    # --phase choices are the UNION across models (the decoder's 8-slot table is a
    # superset of the ViT 6-slot one); membership in the ACTIVE model's table is
    # re-validated after parsing.
    ap.add_argument("--phase", choices=PHASE_NAMES, default=None,
                    help="the clock64 phase to judge the change on. Default: inferred from "
                         "the candidate macro names (see _PHASE_HINTS). Must belong to the "
                         "selected --model's phase table.")
    # REPRESENTATIVE-BY-DEFAULT: the screen runs at the SAME scale as the roofline
    # benchmark (d=2048/B=16384 decoder; d=1024/B=16384 vit) so the phase-SHARES — which
    # drive the relevance gate — are the real ones. The speed comes from ONE seed +
    # clock64 per-phase counters + a cached targeted TU (NOT the full _ops, NOT the
    # 3-seed fp64/AAA gate), NOT from shrinking the problem. A shrunk d/B would be a
    # PROXY with a different bottleneck (the P0 lesson: pipelining looked fine at d=128,
    # regressed at d=2048). --d default resolves per-model after parsing (None sentinel).
    ap.add_argument("--d", type=int, default=None,
                    help="model width. Decoder: 2048 (roofline DEFAULT) | 128 (toy). ViT: "
                         "1024 (size-ladder roofline DEFAULT) | 128 (production). Defaults "
                         "to the selected --model's roofline width.")
    ap.add_argument("--B", type=int, default=16384,
                    help="batch (%%16==0). DEFAULT 16384 = the roofline regime where the GEMM "
                         "phases dominate. Smaller B shifts shares toward the fixed-cost "
                         "barriers/opt-tail, so the relevance gate drifts from production — a "
                         "faster but less faithful screen (a warning is printed).")
    ap.add_argument("--reps", type=int, default=2,
                    help="timing repeats (median); a screen needs direction, not the 3-seed "
                         "rigor of the gate.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=6, help="steps per rep.")
    ap.add_argument("--ncta-cap", type=int, default=0, help="0 = one CTA/SM (full saturation).")
    ap.add_argument("--neutral-pct", type=float, default=3.0,
                    help="|Δ%%| below this = NEUTRAL (absorbs timing variance).")
    ap.add_argument("--relevance-floor", type=float, default=5.0,
                    help="phase-share %% below which the targeted phase is TOO SMALL TO MATTER: "
                         "even eliminating it can't move the step beyond its share, so the "
                         "verdict is IRRELEVANT regardless of the phase-local win (the "
                         "vec4-AdamW <1%%-of-step dry-well guard). 0 disables the gate.")
    ap.add_argument("--json", action="store_true",
                    help="also print a machine-readable TRIAGE_JSON {...} line.")
    ap.add_argument("--verbose", action="store_true",
                    help="stream each variant's build/run log (prefixed with '    | ').")
    args = ap.parse_args()
    mcfg = _MODELS[args.model]
    phase_names = mcfg["phase_names"]
    # resolve the --d default + valid-set from the selected model.
    if args.d is None:
        args.d = mcfg["d_default"]
    assert args.B % 16 == 0, "B must be divisible by 16 (the dW K-loop is 16-step atoms)"
    assert args.d in mcfg["d_valid"], (
        f"the {args.model} bench has only these layouts: {mcfg['d_valid']} "
        f"(default {mcfg['d_default']} = roofline scale). d={args.d} would compile to a "
        f"different D and trip the mod.D==d assert.")
    if args.phase is not None and args.phase not in phase_names:
        ap.error(f"--phase {args.phase} is not a {args.model} phase; valid: {phase_names}")
    if args.B < 16384:
        print(f"[triage] ⚠ B={args.B} < 16384: phase-shares (hence the relevance gate) drift "
              f"from the roofline regime — faster but less faithful. Prefer --B 16384.",
              flush=True)

    if args.validate:
        return _do_validate(args)

    if not args.cand_defines:
        ap.error("a candidate needs at least one --cand-D KEY=VAL (or use --validate). "
                 "Baseline may be empty (production defaults).")

    phase = args.phase or _infer_phase(args.cand_defines, phase_names)
    if args.phase is None:
        print(f"[triage] auto-targeted phase = {phase} "
              f"(inferred from {[d.split('=',1)[0] for d in args.cand_defines]}; "
              f"override with --phase)", flush=True)

    print(f"[triage] model={args.model} scale: d={args.d} B={args.B} reps={args.reps} "
          f"iters={args.iters}  neutral=±{args.neutral_pct}%  "
          f"relevance-floor={args.relevance_floor}%", flush=True)
    base = _run_variant("baseline", args.baseline_defines, args)
    cand = _run_variant("candidate", args.cand_defines, args)
    summary = _report(base, cand, phase, args.neutral_pct, args.relevance_floor, phase_names)
    summary.update({"model": args.model,
                    "baseline_defines": args.baseline_defines,
                    "candidate_defines": args.cand_defines,
                    "d": args.d, "B": args.B})
    if args.json:
        print("TRIAGE_JSON " + json.dumps(summary), flush=True)

    # exit code: 0 if the candidate is a clear or neutral pass on the wall, 1 if the wall
    # got WORSE (the hill-climb can branch on it). BUILD_FAILED => 1.
    if summary.get("verdict") == "BUILD_FAILED":
        return 1
    return 1 if summary.get("verdict_wall") == "worse" else 0


if __name__ == "__main__":
    sys.exit(main())
