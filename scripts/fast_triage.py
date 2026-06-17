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
META-VERIFICATION — the screen's trustworthiness is MEASURED + CONTINUOUSLY VERIFIED,
PER (model, context), NOT ASSUMED. Three standing mechanisms (all additive):

  (1) CALIBRATION SUITE (`_CALIBRATION` + `--validate [--model M] [--context C]`)
      A registry of KNOWN GROUND-TRUTH results keyed by (model, context) the screen MUST
      reproduce before it is TRUSTED for that context. SEED = the campaign's measured
      decoder dW KEEP (OPTIMIZATION_LEDGER.md "dW contiguous-layout staging: KEEP +2.05×"):
          stage=0 / splitk=1 (scalar)      = 1889.8 ms  @ d=2048/B=16384, 3 seeds
          stage=1 / splitk=1 (contiguous)  =  920.7 ms  =>  2.05× faster (P2_dW_GEMM, ~97% staging)
      --validate runs the matching entries and PASSES a context ONLY if the screen
      reproduces the recorded DIRECTION and (at the measured scale) the MAGNITUDE window.
      CRITICAL — per (model,context): the decoder validation does NOT transfer to ViT /
      Mamba / Level-2. A context with NO entry reports "UNCALIBRATED — screen is direction-
      only-untrusted here" (exit 1) so trust is never assumed for an unmeasured context.
      Exit: 0 = FAITHFUL, 2 = reproduced the WRONG direction/magnitude (UNTRUSTWORTHY),
            1 = UNCALIBRATED or INCONCLUSIVE (a variant failed to build/run).

        # decoder/model-triage (the seed) — reproduces the dW 2.05× exactly as before:
        CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. \
          python scripts/fast_triage.py --validate --d 2048 --B 16384 --reps 3
        # a context with no ground-truth yet → UNCALIBRATED (exit 1), trust NOT assumed:
        ... --validate --model vit            # or --model mamba / --context level-2

  (2) REPRESENTATIVENESS ASSERTIONS (every run; `_assert_representative`)
      Each screen run asserts it measures the SAME problem the calibration was taken on:
      real scale (mod.D==d, full B) AND (where per-phase data + a calibrated share exist)
      that the measured phase-SHARES match the calibration's expected shares within a
      tolerance. Drift beyond tolerance is FLAGGED loudly ("screen no longer representative
      for <context> — shares drifted X% from calibration"): the mechanical guard against an
      unrepresentative screen silently mis-pruning the true-best. For a WALL-ONLY model
      (Mamba; novel Level-2 fusions) the share check is SKIPPED and that fact asserted.

  (3) SCREEN-VS-GATE AGREEMENT LEDGER (`.perf/screen_gate_agreement.jsonl`)
      `record_screen_gate(model, context, screen_verdict, gate_verdict)` appends one row
      whenever a screen-survivor later clears (or fails) the full fp64+3-seed gate.
      `--fidelity-report` summarises the running screen fidelity per (model,context):
      agreement rate + false-positive rate (screen-said-better, gate-said-worse). A rising
      disagreement rate = the screen is drifting for that context → surfaced. (The false-
      NEGATIVE rate needs occasionally gating a screen-REJECT to estimate; see the docstring
      of record_screen_gate.)

        CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. \
          python scripts/fast_triage.py --fidelity-report

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
from datetime import datetime, timezone
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
#    `wall_only=True` (Mamba) marks a model whose bench has NO per-phase profiler
#    (mamba_bench.measure has no profile=/phase_cycles): a screen there is DIRECTION-only,
#    so every per-phase / share claim is suppressed + asserted, never fabricated.
_MODELS = {
    "decoder": {
        "bench": "tuning.decoder_bench",
        "profile_macro": "SG_DEC_PROFILE",
        "phase_names": PHASE_NAMES,
        "d_default": 2048,
        "d_valid": (128, 2048),
        "wall_only": False,
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
        "wall_only": False,
        "d_help": ("ViT width. The bench has TWO layouts: 1024 (size-ladder roofline "
                   "scale — DEFAULT) and 128 (production). --d selects the layout branch."),
    },
    # Mamba — WALL-ONLY: mamba_bench.measure(mod, d, B, reps, warmup, iters) exposes NO
    # profile=/phase_cycles (no g_*_prof profiler wired yet, per OPTIMIZATION_LEDGER.md),
    # so a Mamba screen is DIRECTION-only. Registered so --model mamba / --validate --model
    # mamba report UNCALIBRATED cleanly (no entry yet) + the meta-verification is Mamba-ready
    # for when a Mamba KEEP lands. phase_names is borrowed for --phase membership only; no
    # per-phase data is ever read for a wall-only model.
    "mamba": {
        "bench": "tuning.mamba_bench",
        "profile_macro": "SG_MB_PROFILE",
        "phase_names": PHASE_NAMES,   # borrowed (membership only); Mamba reads NO phase_cycles
        "d_default": 1024,
        "d_valid": (128, 1024),
        "wall_only": True,
        "d_help": ("Mamba d_model. The bench has TWO layouts: 1024 (size-ladder roofline "
                   "scale — DEFAULT) and 128 (production). WALL-ONLY: no per-phase profiler."),
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


# ════════════════════════════════════════════════════════════════════════════════════
# META-VERIFICATION — the screen's trustworthiness is MEASURED + CONTINUOUSLY VERIFIED,
# per (model, context), NOT assumed. Three standing mechanisms (all additive; none changes
# the existing decoder --validate / --model routing / relevance-gate behaviour):
#
#   (1) CALIBRATION SUITE  — `_CALIBRATION`: a registry of KNOWN GROUND-TRUTH results keyed
#       by (model, context). `--validate [--model M]` runs the matching entries and PASSES
#       a context ONLY if the screen reproduces the recorded DIRECTION and (at real scale)
#       MAGNITUDE. Per-(model,context): the decoder dW result does NOT transfer to ViT /
#       Mamba / Level-2; a context with no entry reports UNCALIBRATED (direction-only-
#       untrusted) so trust is never assumed for an unmeasured context.
#
#   (2) REPRESENTATIVENESS ASSERTIONS — `_assert_representative`: on EVERY screen run, assert
#       it measures the SAME problem the calibration was taken on — real scale (mod.D==d,
#       full B), and (where per-phase data exists) that the measured phase-SHARES match the
#       calibration's expected shares within a tolerance. Drift beyond tolerance is FLAGGED
#       loudly: the mechanical guard against an unrepresentative screen silently mis-pruning.
#
#   (3) SCREEN-VS-GATE AGREEMENT LEDGER — `record_screen_gate` appends one JSONL row to
#       `.perf/screen_gate_agreement.jsonl` whenever a screen-survivor later clears (or
#       fails) the full fp64+3-seed gate; `--fidelity-report` summarises the running screen
#       fidelity per (model,context) — agreement rate + false-positive rate. A rising
#       disagreement rate = the screen is drifting for that context → surfaced.
# ════════════════════════════════════════════════════════════════════════════════════

# .perf/ is the campaign's perf-artifact dir (patches, proofs). The agreement ledger lives
# alongside them. Both paths are module-level so the API + the report agree on one location.
PERF_DIR = ROOT / ".perf"
AGREEMENT_LEDGER = PERF_DIR / "screen_gate_agreement.jsonl"


# ── (1) CALIBRATION SUITE ──────────────────────────────────────────────────────────
# A registry of KNOWN GROUND-TRUTH results the screen MUST reproduce before it is TRUSTED
# for that (model, context). The seed is the campaign's measured decoder dW KEEP (Track E,
# OPTIMIZATION_LEDGER.md): STAGE=0->1 @ splitk=1 = 1889.8 -> 920.7 ms = 2.05× @ d=2048/
# B=16384, 3 seeds; the dW was ~97% staging-bound so P2_dW_GEMM dominated the step.
#
# CONTEXT taxonomy (the THREE places we screen, the owner's explicit set):
#   "model-triage"  — the hand-driven hill-climb that calls fast_triage directly (a human /
#                     the tune_optimizers loop proposing -D SG_TUNED_* configs per model).
#   "compile.py"    — compile.py's autotuner SCREEN tier (the cheap surrogate/prune stage
#                     that drops candidates before its fp64+A/A/A pick_winner gate).
#   "level-2"       — the Level-2 variant search over NOVEL FUSION variants (wall-only:
#                     a fresh fusion has no per-phase profiler, so DIRECTION-only).
#
# Each entry is a ground-truth the screen is asserted to reproduce. Fields:
#   model, context        : the (model, context) key this calibrates.
#   desc                  : human one-liner (which KEEP/REVERT this pins).
#   baseline_defines      : the -D set for the SLOWER/baseline variant.
#   candidate_defines     : the -D set for the FASTER/candidate variant (the measured win).
#   phase                 : the targeted clock64 phase (None => wall-only context; assert
#                           DIRECTION on the wall ONLY, make NO per-phase claim).
#   d, B                  : the scale the ground-truth was MEASURED at (the magnitude scale).
#   expect_wall_dir       : "faster" | "slower" — the recorded wall direction cand-vs-base.
#   mag_window            : (lo, hi) wall-speedup window (base/cand), checked only when the
#                           run is AT the measured scale (d>=d_gt and B>=B_gt). None => no
#                           magnitude claim (direction-only ground-truth).
#   expect_phase_dir      : "faster"|"slower"|None — recorded targeted-phase direction.
#   expect_phase_share    : the targeted phase's expected SHARE-of-step % on the BASELINE
#                           (the representativeness anchor). None for wall-only contexts.
#   share_tol_pct         : absolute pp tolerance on that share before drift is FLAGGED.
#
# ADDING A GROUND-TRUTH (as new KEEPs land — decoder/vit/mamba × kernel-knob/compile.py/
# Level-2): append a dict with the measured baseline/candidate -D, scale, direction, and
# (if a profiler exists for that model) the targeted-phase share from the KEEP's profile.
# A wall-only model (Mamba; novel Level-2 fusions) MUST use phase=None / expect_phase_*=
# None / mag_window per the ledger — never fabricate a per-phase share it cannot measure.
_CALIBRATION: list[dict] = [
    {
        "model": "decoder",
        "context": "model-triage",
        "desc": "dW contiguous-layout staging KEEP +2.05× (Track E): STAGE 0->1 @ splitk=1, "
                "1889.8->920.7 ms @ d=2048/B=16384 3-seed; dW was ~97% staging-bound.",
        "baseline_defines": ["SG_TUNED_DEC_DW_STAGE=0", "SG_TUNED_DEC_DW_SPLITK=1"],
        "candidate_defines": ["SG_TUNED_DEC_DW_STAGE=1", "SG_TUNED_DEC_DW_SPLITK=1"],
        "phase": "P2_dW_GEMM",
        "d": 2048,
        "B": 16384,
        "expect_wall_dir": "faster",
        "mag_window": (1.5, 2.6),       # around the measured 2.05× (matches legacy --validate)
        "expect_phase_dir": "faster",
        "expect_phase_share": 60.0,     # P2_dW_GEMM baseline share of step (staging-bound dW
                                        # is the dominant phase pre-fix); wide tol below — the
                                        # GPU re-measure tightens this to the observed value.
        "share_tol_pct": 25.0,          # generous until re-measured on-GPU (see GPU-DEFERRED).
    },
    # ── UNCALIBRATED contexts (no measured ground-truth yet) — listed so --validate can
    #    report them explicitly as "UNCALIBRATED — screen is direction-only-untrusted here"
    #    rather than silently implying the decoder calibration transfers. Promote each to a
    #    full entry (above) when its KEEP lands and is re-measured on-GPU:
    #      ("vit", "model-triage")      — needs a ViT KEEP (the dW twin REVERTED; ViT is
    #                                     non-GEMM-bound, so the lever is elsewhere). Has a
    #                                     per-phase profiler → a future entry CAN set phase/
    #                                     expect_phase_share.
    #      ("mamba", "model-triage")    — WALL-ONLY (mamba_bench.measure has no profile=/
    #                                     phase_cycles). Any future entry MUST use phase=None.
    #      ("decoder","compile.py")     — compile.py screen tier (decoder); add when a
    #                                     compile.py-screened candidate clears the gate.
    #      ("decoder","level-2"), ("vit","level-2"), ("mamba","level-2")
    #                                   — WALL-ONLY novel-fusion variants; phase=None.
]

# Contexts the meta-verification KNOWS about (for UNCALIBRATED reporting). The screen is
# DIRECTION-ONLY-UNTRUSTED for any (model, context) here that has no _CALIBRATION entry.
_KNOWN_CONTEXTS = ("model-triage", "compile.py", "level-2")


def _calibration_for(model: str, context: str | None = None) -> list[dict]:
    """Return the calibration entries matching `model` (and `context` if given)."""
    return [e for e in _CALIBRATION
            if e["model"] == model and (context is None or e["context"] == context)]


def _is_wall_only(model: str) -> bool:
    """True if `model` has no per-phase profiler (Mamba) — screen is direction-only and
    every per-phase / share claim must be suppressed (asserted as a fact, not fabricated).
    Source of truth = the model's `wall_only` flag in _MODELS; an unknown model is treated
    as wall-only (conservative: never fabricate per-phase data for something unregistered)."""
    return _MODELS.get(model, {}).get("wall_only", True)


# ── (2) REPRESENTATIVENESS ASSERTIONS ────────────────────────────────────────────────
def _assert_representative(base: dict, cand: dict, phase: str, phase_names: list[str],
                           args, calib: dict | None) -> dict:
    """On a real screen run, assert it measures the SAME problem the calibration was taken
    on. Returns a dict {scale_ok, scale_note, share_ok, share_note, shares{}} and PRINTS a
    loud FLAG on drift. Two checks:

      SCALE  — mod.D==d (enforced in-worker too) AND the run is at the calibrated/roofline
               scale (d>=d_default AND B>=16384). A shrunk scale makes the phase-SHARES (and
               thus the relevance gate) UNREPRESENTATIVE — the P0 lesson — so it is flagged.
      SHARES — where per-phase data exists AND a calibration share is known, assert the
               measured baseline phase-SHARE is within `share_tol_pct` of the calibration's
               expected share. Drift beyond tolerance => the screen is no longer measuring
               the problem the ground-truth was taken on => FLAG ("shares drifted X%").

    For a WALL-ONLY model (no phase_cycles) the SHARE check is SKIPPED and that fact is
    asserted (share_ok=None, note="wall-only: no per-phase data — DIRECTION-only screen")."""
    model = getattr(args, "model", "decoder")
    mcfg = _MODELS[model]
    out = {"scale_ok": True, "scale_note": "", "share_ok": None, "share_note": "",
           "shares": {}}

    # SCALE: roofline-representative iff at-or-above the model's roofline width AND full B.
    at_scale = (args.d >= mcfg["d_default"]) and (args.B >= 16384)
    out["scale_ok"] = at_scale
    if not at_scale:
        out["scale_note"] = (f"d={args.d}/B={args.B} is BELOW the {model} roofline scale "
                             f"(d>={mcfg['d_default']}, B>=16384) → phase-shares (hence the "
                             f"relevance gate) are NOT representative of production")
        print(f"  ⚠ REPRESENTATIVENESS [scale]: {out['scale_note']}.", flush=True)

    # SHARES: only meaningful with per-phase data. Wall-only models assert the absence.
    bp = base.get("phase_cycles")
    if _is_wall_only(model) or not bp:
        out["share_ok"] = None
        out["share_note"] = ("wall-only: no per-phase data — DIRECTION-only screen, no share "
                             "assertion possible (asserted, not fabricated)")
        return out

    bt = sum(bp) or 1
    out["shares"] = {n: round(100.0 * bp[i] / bt, 2) for i, n in enumerate(phase_names)}
    if calib is None or calib.get("expect_phase_share") is None:
        out["share_ok"] = None
        out["share_note"] = ("no calibrated share for this (model,context) — share drift "
                             "cannot be checked (UNCALIBRATED for representativeness)")
        return out

    measured = out["shares"].get(phase)
    expected = float(calib["expect_phase_share"])
    tol = float(calib.get("share_tol_pct", 10.0))
    drift = abs(measured - expected) if measured is not None else float("inf")
    out["share_ok"] = drift <= tol
    if not out["share_ok"]:
        ctx = calib.get("context", "?")
        out["share_note"] = (f"{phase} share {measured:.1f}% drifted {drift:.1f}pp from the "
                             f"calibration's {expected:.1f}% (tol ±{tol:.0f}pp)")
        print(f"  ⚠ REPRESENTATIVENESS [shares]: screen no longer representative for "
              f"{model}/{ctx} — {phase} share drifted {drift:.1f}% from calibration "
              f"({measured:.1f}% vs {expected:.1f}%, tol ±{tol:.0f}pp).", flush=True)
    return out


# ── (3) SCREEN-VS-GATE AGREEMENT LEDGER ──────────────────────────────────────────────
def record_screen_gate(model: str, context: str, screen_verdict: str, gate_verdict: str,
                       *, candidate_defines: list[str] | None = None,
                       baseline_defines: list[str] | None = None,
                       note: str = "", ledger_path: Path = AGREEMENT_LEDGER) -> dict:
    """Append ONE row to the screen-vs-gate agreement JSONL ledger.

    CALL THIS whenever a screen-SURVIVOR later clears (or fails) the full fp64+3-seed gate,
    from whichever context ran the screen (model-triage / compile.py / Level-2). It is the
    standing record of whether the cheap screen AGREED with the authoritative gate.

      screen_verdict : the SCREEN's directional call — "better" | "worse" | "neutral"
                       (fast_triage's verdict_wall) | "irrelevant".
      gate_verdict   : the full gate's authoritative call — "kept" | "reverted"
                       (parity-clean + faster on 3 seeds => kept; else reverted) | "failed"
                       (parity/determinism RED — a correctness fail the screen cannot see).

    `agreed` is TRUE iff the screen's direction matched the gate's outcome:
        screen 'better'  agrees with gate 'kept';
        screen 'worse'/'neutral'/'irrelevant' agrees with gate 'reverted'/'failed'.
    A screen-said-better that the gate REVERTED/FAILED is a FALSE POSITIVE (the costly error
    this screen exists to bound). Returns the row dict (also appended to the JSONL).

    NOTE ON FALSE NEGATIVES: this ledger naturally captures only SURVIVORS (screen passed →
    gate ran), so it estimates the FALSE-POSITIVE rate well but UNDER-counts FALSE NEGATIVES
    (screen-said-worse that the gate would have KEPT). Estimating the false-negative rate
    requires occasionally GATING A SCREEN-REJECT on purpose and recording it here too."""
    screen_verdict = str(screen_verdict).lower()
    gate_verdict = str(gate_verdict).lower()
    agreed = ((screen_verdict == "better" and gate_verdict == "kept") or
              (screen_verdict in ("worse", "neutral", "irrelevant")
               and gate_verdict in ("reverted", "failed")))
    false_positive = (screen_verdict == "better" and gate_verdict in ("reverted", "failed"))
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "context": context,
        "screen_verdict": screen_verdict,
        "gate_verdict": gate_verdict,
        "agreed": agreed,
        "false_positive": false_positive,
        "candidate_defines": candidate_defines or [],
        "baseline_defines": baseline_defines or [],
        "note": note,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _read_ledger(ledger_path: Path = AGREEMENT_LEDGER) -> list[dict]:
    """Load the agreement JSONL (tolerant of blank/garbage lines). [] if absent."""
    if not ledger_path.exists():
        return []
    rows = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, TypeError):
            continue
    return rows


def _fidelity_report(ledger_path: Path = AGREEMENT_LEDGER, *, as_json: bool = False) -> dict:
    """Summarise running screen fidelity per (model, context) from the agreement ledger:
    sample count, AGREEMENT rate, and FALSE-POSITIVE rate (screen-said-better, gate-said-
    worse — the costly error). A rising disagreement rate for a context = the screen is
    DRIFTING there → surfaced. Prints a table; returns the per-key summary dict."""
    rows = _read_ledger(ledger_path)
    summ: dict[str, dict] = {}
    for r in rows:
        key = f"{r.get('model','?')}/{r.get('context','?')}"
        s = summ.setdefault(key, {"n": 0, "agree": 0, "fp": 0, "n_better": 0})
        s["n"] += 1
        s["agree"] += 1 if r.get("agreed") else 0
        s["fp"] += 1 if r.get("false_positive") else 0
        s["n_better"] += 1 if str(r.get("screen_verdict")).lower() == "better" else 0
    for s in summ.values():
        s["agreement_rate"] = round(s["agree"] / s["n"], 3) if s["n"] else None
        # false-positive rate is over screen-BETTER calls (the gate-confirmable ones).
        s["false_positive_rate"] = (round(s["fp"] / s["n_better"], 3)
                                    if s["n_better"] else None)

    print("\n" + "=" * 78, flush=True)
    print(f"[fidelity-report] screen-vs-gate agreement  ({ledger_path})", flush=True)
    print("=" * 78, flush=True)
    if not rows:
        print("  (ledger empty — no screen→gate pairs recorded yet. The screen's fidelity is "
              "UNMEASURED until survivors are gated and record_screen_gate() is called.)",
              flush=True)
    else:
        print(f"  {'model/context':28s} {'n':>4s} {'agree%':>8s} {'FP%':>7s}  "
              f"{'(FP = screen-better, gate-worse)':>0s}", flush=True)
        for key in sorted(summ):
            s = summ[key]
            ar = f"{100*s['agreement_rate']:.0f}" if s["agreement_rate"] is not None else "—"
            fp = f"{100*s['false_positive_rate']:.0f}" if s["false_positive_rate"] is not None else "—"
            print(f"  {key:28s} {s['n']:>4d} {ar:>7s}% {fp:>6s}%", flush=True)
        print("\n  Rising disagreement (falling agree%) for a context = the screen is "
              "DRIFTING there → re-calibrate that (model,context).", flush=True)
        print("  NOTE: this ledger sees SURVIVORS, so FP% is well-estimated but the "
              "false-NEGATIVE rate is under-counted (gate a screen-reject to estimate it).",
              flush=True)
    out = {"ledger": str(ledger_path), "n_total": len(rows), "by_context": summ}
    if as_json:
        print("FIDELITY_JSON " + json.dumps(out), flush=True)
    return out


# ── the worker body: build ONE bench variant in an isolated CUDA context, profile-
#    time it, print a single TRIAGE_RESULT JSON line. Driven via `python -c`, the
#    tune_optimizers.py:645 persistent-subprocess idiom (one warm context per variant).
#    `{bench}` is the coexisting bench module (tuning.decoder_bench | tuning.vit_bench |
#    tuning.mamba_bench). decoder/vit expose build_variant(d, profile=True, defines=…)/
#    measure(…, profile=True, ncta_cap=…)->res["phase_cycles"]; the WALL-ONLY Mamba bench
#    (`{wall_only}=True`) has the plainer build_variant(d, defines=…)/measure(mod,d,B,reps,
#    warmup,iters) with NO profiler — the worker branches on `{wall_only}` to call the right
#    signature and emits phase_cycles=None (direction-only). `{profile_macro}` labels the
#    HAS_PROFILE assert message (per-phase models only).
_WORKER = r"""
import json, sys
from pathlib import Path
ROOT = Path({root!r})
sys.path.insert(0, str(ROOT))
import {bench} as db

d, B, reps, warmup, iters, ncta = {d}, {B}, {reps}, {warmup}, {iters}, {ncta}
defines = {defines!r}
wall_only = {wall_only}
try:
    if wall_only:
        # WALL-ONLY model (Mamba): no profile flag, no per-phase counters, no ncta_cap.
        mod = db.build_variant(d, defines=defines)
        assert int(mod.D) == d, "TU D=%d != %d" % (int(mod.D), d)
        res = db.measure(mod, d, B, reps=reps, warmup=warmup, iters=iters)
        phase_cycles = None
    else:
        mod = db.build_variant(d, profile=True, defines=defines)
        assert int(mod.D) == d, "TU D=%d != %d" % (int(mod.D), d)
        assert getattr(mod, "HAS_PROFILE", False), "variant built without -D{profile_macro}"
        res = db.measure(mod, d, B, reps=reps, warmup=warmup, iters=iters,
                         profile=True, ncta_cap=ncta)
        phase_cycles = res.get("phase_cycles")
    out = {{
        "ok": True,
        "wall_ms": res["wall_ms"],
        "walls_ms": res.get("walls_ms", []),
        "achieved_tf_s": res["achieved_tf_s"],
        "phase_cycles": phase_cycles,
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
        wall_only=bool(mcfg.get("wall_only", False)),
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
            relevance_floor: float = 5.0, phase_names: list[str] = PHASE_NAMES,
            args=None, calib: dict | None = None) -> dict:
    """Build + print the human triage report. Returns the JSON summary dict.
    `phase_names` is the active model's clock64 slot table (decoder by default).
    `args`/`calib` (optional, additive) enable the REPRESENTATIVENESS ASSERTION: when
    provided, every run is checked for real-scale + (where per-phase data + a calibrated
    share exist) phase-SHARE drift vs the calibration, FLAGGED loudly on drift, and the
    result threaded into the summary under `representativeness`. Omitted => legacy behaviour."""
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

    # ── WALL-ONLY path (Mamba; novel Level-2 fusions): no per-phase counters exist, so
    #    judge DIRECTION on the wall ALONE and ASSERT that fact — no fabricated per-phase /
    #    relevance / share claim. Triggered when either side lacks phase_cycles.
    if not base.get("phase_cycles") or not cand.get("phase_cycles"):
        wall_delta = _pct_delta(base["wall_ms"], cand["wall_ms"])
        wall_v = _verdict(wall_delta, neutral_pct)
        wall_speedup = base["wall_ms"] / cand["wall_ms"] if cand["wall_ms"] else float("nan")
        print(f"  params={base['total_params']:,}  [WALL-ONLY model: no per-phase profiler]",
              flush=True)
        print(f"  baseline wall={base['wall_ms']:.3f} ms ({base['achieved_tf_s']:.3f} TF/s)   "
              f"candidate wall={cand['wall_ms']:.3f} ms ({cand['achieved_tf_s']:.3f} TF/s)",
              flush=True)
        representativeness = None
        if args is not None:
            representativeness = _assert_representative(base, cand, phase, phase_names, args, calib)
        print(f"\n  total step (wall): {base['wall_ms']:.3f} -> {cand['wall_ms']:.3f} ms  "
              f"({wall_delta:+.1f}%, {wall_v};  {wall_speedup:.3f}× speedup)", flush=True)
        print(f"\n  TRIAGE: step {wall_v} by {abs(wall_delta):.1f}% ({wall_speedup:.2f}×) "
              f"— DIRECTION-ONLY (wall-only model: no per-phase share, no relevance gate; "
              f"the wall is the sole signal).", flush=True)
        print("  NOTE: triage only — NOT a KEEP. The fp64 parity + A/A/A gate + 3-seed "
              "timing remain the arbiter.", flush=True)
        return {
            "verdict_phase": None, "verdict_wall": wall_v,
            "targeted_phase": None, "wall_only": True,
            "phase_share_pct": None, "projected_step_pct": None, "max_step_gain_pct": None,
            "relevant": None, "relevance_floor_pct": relevance_floor, "consistent": None,
            "phase_delta_pct": None,
            "wall_delta_pct": round(wall_delta, 2),
            "wall_speedup": round(wall_speedup, 4),
            "baseline_wall_ms": round(base["wall_ms"], 4),
            "candidate_wall_ms": round(cand["wall_ms"], 4),
            "baseline_phase_ms": None, "candidate_phase_ms": None,
            "baseline_tf_s": base["achieved_tf_s"], "candidate_tf_s": cand["achieved_tf_s"],
            "neutral_pct": neutral_pct, "representativeness": representativeness,
        }

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

    # REPRESENTATIVENESS ASSERTION (additive) — assert this run measures the SAME problem
    # the calibration was taken on (real scale + phase-share drift), FLAG loudly on drift.
    # Only runs when the caller threaded `args` in; legacy callers skip it untouched.
    representativeness = None
    if args is not None:
        representativeness = _assert_representative(base, cand, phase, phase_names, args, calib)

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
        "representativeness": representativeness,
    }


def _validate_entry(entry: dict, args, phase_names: list[str]) -> dict:
    """Run ONE calibration entry and judge whether the screen reproduced its recorded
    ground-truth. Returns {entry, faithful, direction_ok, magnitude_ok, magnitude_checked,
    summary, reason}. Faithful iff the recorded wall DIRECTION reproduces AND (where the
    entry has a per-phase ground-truth) the targeted-phase direction reproduces AND (at the
    measured scale) the wall speedup lands in the entry's magnitude window. A wall-only
    entry (phase=None) is judged on the WALL direction alone — no per-phase claim made."""
    model, context = entry["model"], entry["context"]
    phase = entry.get("phase")           # None => wall-only entry (no per-phase claim)
    wall_only = phase is None or _is_wall_only(model)
    expect_faster = entry.get("expect_wall_dir", "faster") == "faster"
    mag_window = entry.get("mag_window")

    print("-" * 78, flush=True)
    print(f"[validate] {model}/{context}: {entry.get('desc','(no desc)')}", flush=True)
    print(f"  baseline  = {entry['baseline_defines'] or '(prod defaults)'}", flush=True)
    print(f"  candidate = {entry['candidate_defines'] or '(prod defaults)'}", flush=True)
    print(f"  expect: wall {entry.get('expect_wall_dir','faster')}"
          + ("" if wall_only else f", {phase} {entry.get('expect_phase_dir','faster')}")
          + (f", magnitude {mag_window} (only if at d>={entry['d']}/B>={entry['B']})"
             if mag_window else " (direction-only — no magnitude ground-truth)")
          + ("  [WALL-ONLY context: no per-phase claim]" if wall_only else ""),
          flush=True)

    base = _run_variant(f"{model}:baseline", list(entry["baseline_defines"]), args)
    cand = _run_variant(f"{model}:candidate", list(entry["candidate_defines"]), args)
    # judge-phase for the report: the entry's phase, or a wall-anchored placeholder for
    # wall-only entries (P2_dW_GEMM exists in every table; its delta is reported but the
    # faithfulness verdict for a wall-only entry ignores it — wall is the sole arbiter).
    judge_phase = phase if (phase and phase in phase_names) else (
        "P2_dW_GEMM" if "P2_dW_GEMM" in phase_names else phase_names[0])
    summary = _report(base, cand, judge_phase, args.neutral_pct, args.relevance_floor,
                      phase_names, args=args, calib=entry)

    if summary.get("verdict") == "BUILD_FAILED":
        return {"entry": entry, "faithful": None, "reason": "BUILD_FAILED",
                "direction_ok": None, "magnitude_ok": None, "magnitude_checked": False,
                "summary": summary}

    wall_d = summary["wall_delta_pct"]
    speedup = summary["wall_speedup"]
    # DIRECTION: the wall must move the recorded way by more than the neutral band.
    if expect_faster:
        wall_dir_ok = wall_d < -args.neutral_pct
    else:
        wall_dir_ok = wall_d > args.neutral_pct
    # per-phase direction only where the entry makes a per-phase claim.
    if wall_only:
        phase_dir_ok = True
    else:
        phase_d = summary["phase_delta_pct"]
        phase_dir_ok = (phase_d < -args.neutral_pct) if (
            entry.get("expect_phase_dir", "faster") == "faster") else (phase_d > args.neutral_pct)
    direction_ok = wall_dir_ok and phase_dir_ok

    # MAGNITUDE: only when the run is AT (or above) the scale the ground-truth was measured.
    mag_checked = (mag_window is not None and args.d >= entry["d"] and args.B >= entry["B"])
    mag_ok = True
    if mag_checked:
        lo, hi = mag_window
        # for a "slower" entry the recorded ratio is base/cand < 1; the window already
        # encodes that, so a direct containment check is correct either way.
        mag_ok = lo <= speedup <= hi

    faithful = direction_ok and mag_ok
    return {"entry": entry, "faithful": faithful, "direction_ok": direction_ok,
            "magnitude_ok": mag_ok, "magnitude_checked": mag_checked,
            "wall_only": wall_only, "summary": summary, "reason": ""}


def _do_validate(args) -> int:
    """CALIBRATION SUITE runner — prove the screen reproduces the KNOWN ground-truths for
    the selected (model, context) before it is TRUSTED there. Per-(model,context): the
    decoder dW result does NOT transfer to ViT/Mamba/Level-2; each must have its own entry.

    Exit code:
      0 = all matching calibration entries reproduced (screen FAITHFUL for that context),
          OR (no entries but the context is direction-only-by-nature) — see below;
      2 = the screen reproduced a WRONG direction/magnitude for some entry (UNTRUSTWORTHY);
      1 = INCONCLUSIVE (a variant failed to build/run) OR UNCALIBRATED (no ground-truth for
          this (model,context) — the screen is "direction-only-untrusted" there).

    Preserves the legacy decoder behaviour exactly: the seed entry (decoder/model-triage)
    reproduces the dW 2.05× check (exit 0 faithful / 2 wrong-direction), and --validate
    without --model still runs it."""
    model = getattr(args, "model", "decoder")
    context = getattr(args, "context", None)   # None => all contexts for the model
    entries = _calibration_for(model, context)
    phase_names = _MODELS[model]["phase_names"]

    print("=" * 78, flush=True)
    ctx_lbl = context or "ALL contexts"
    print(f"[triage --validate] CALIBRATION SUITE — model={model} context={ctx_lbl}", flush=True)
    print(f"  scale = d={args.d} B={args.B} reps={args.reps}  "
          f"(at-or-above an entry's measured scale also checks its MAGNITUDE window; a "
          f"smaller --B is a faster DIRECTION-only smoke)", flush=True)

    if not entries:
        # No measured ground-truth for this (model,context). Trust is NOT assumed: report
        # UNCALIBRATED loudly. This is the per-(model,context) firewall: the decoder
        # calibration does NOT make the screen trusted for ViT/Mamba/Level-2.
        wall_only = _is_wall_only(model)
        print("=" * 78, flush=True)
        print(f"[triage --validate] ⚠ UNCALIBRATED — no ground-truth entry for "
              f"{model}/{ctx_lbl}. The screen is DIRECTION-ONLY-UNTRUSTED here: it has "
              f"never been proven to reproduce a measured result in this context.", flush=True)
        if wall_only:
            print(f"  ({model} is WALL-ONLY — no per-phase profiler — so any future entry "
                  f"will be direction-only by construction.)", flush=True)
        print(f"  Known contexts: {_KNOWN_CONTEXTS}. Add a calibration entry (a measured "
              f"KEEP/REVERT for this model+context) to _CALIBRATION before relying on the "
              f"screen here. Until then: treat its verdict as a weak directional hint.",
              flush=True)
        print("=" * 78, flush=True)
        if args.json:
            print("TRIAGE_JSON " + json.dumps({"validate": "UNCALIBRATED", "model": model,
                                               "context": context, "wall_only": wall_only}),
                  flush=True)
        return 1

    results = [_validate_entry(e, args, phase_names) for e in entries]

    # aggregate verdict over the matching entries.
    any_build_fail = any(r["faithful"] is None and r.get("reason") == "BUILD_FAILED"
                         for r in results)
    judged = [r for r in results if r["faithful"] is not None]
    all_faithful = bool(judged) and all(r["faithful"] for r in judged)

    print("\n" + "=" * 78, flush=True)
    for r in results:
        e = r["entry"]
        tag = f"{e['model']}/{e['context']}"
        if r["faithful"] is None:
            print(f"[validate] {tag}: INCONCLUSIVE ({r.get('reason')}).", flush=True)
            continue
        s = r["summary"]
        wo = " [wall-only]" if r.get("wall_only") else ""
        if r["faithful"]:
            magtxt = (f", magnitude {s['wall_speedup']:.2f}× in {e['mag_window']}"
                      if r["magnitude_checked"] else " (direction-only at this scale)")
            print(f"[validate] ✅ {tag}: FAITHFUL{wo} — wall {s['wall_delta_pct']:+.1f}% "
                  f"({s['wall_speedup']:.2f}×){magtxt}.", flush=True)
        else:
            print(f"[validate] ❌ {tag}: UNFAITHFUL{wo} — "
                  f"direction_ok={r['direction_ok']} magnitude_ok={r['magnitude_ok']}; "
                  f"wall {s['wall_delta_pct']:+.1f}% ({s['wall_speedup']:.2f}×).", flush=True)

    if any_build_fail and not judged:
        print("\n[triage --validate] INCONCLUSIVE — variant(s) failed to build/run "
              "(not a faithfulness verdict). Re-run with --verbose.", flush=True)
        rc = 1
    elif all_faithful:
        print(f"\n[triage --validate] ✅ FAITHFUL: the screen reproduced ALL "
              f"{len(judged)} calibration entr{'y' if len(judged)==1 else 'ies'} for "
              f"{model}/{ctx_lbl}. Trusted for this context.", flush=True)
        rc = 0
    else:
        print(f"\n[triage --validate] ❌ UNFAITHFUL: the screen did NOT reproduce a known "
              f"ground-truth for {model}/{ctx_lbl}. UNTRUSTWORTHY here — investigate "
              f"(profile build flag, phase-slot mapping, the candidate actually compiled "
              f"the optimized path, stale bench-variant cache) before relying on it.",
              flush=True)
        rc = 2
    print("=" * 78, flush=True)
    if args.json:
        out = {"validate": ("FAITHFUL" if rc == 0 else "UNFAITHFUL" if rc == 2
                            else "INCONCLUSIVE"),
               "model": model, "context": context, "n_entries": len(entries),
               "entries": [{"model": r["entry"]["model"], "context": r["entry"]["context"],
                            "faithful": r["faithful"], "direction_ok": r.get("direction_ok"),
                            "magnitude_ok": r.get("magnitude_ok"),
                            "magnitude_checked": r.get("magnitude_checked"),
                            "wall_delta_pct": r["summary"].get("wall_delta_pct"),
                            "wall_speedup": r["summary"].get("wall_speedup")}
                           for r in results]}
        print("TRIAGE_JSON " + json.dumps(out), flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fast directional better/worse triage for SuperGrok1.5 kernel "
                    "macro-configs (NOT a KEEP gate — see the module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="run the CALIBRATION SUITE for the selected (model,context) instead "
                         "of a candidate triage: assert the screen reproduces the known "
                         "ground-truth(s). Exit 0=faithful, 2=untrustworthy, 1=UNCALIBRATED/"
                         "inconclusive. With --model decoder (default) this reproduces the dW "
                         "2.05× check; other (model,context) pairs report UNCALIBRATED until a "
                         "ground-truth is added to _CALIBRATION.")
    ap.add_argument("--model", choices=tuple(_MODELS), default="decoder",
                    help="which megakernel to triage: 'decoder' (DEFAULT, the validated "
                         "reference), 'vit', or 'mamba' (WALL-ONLY — no per-phase profiler, "
                         "so a DIRECTION-only screen). Selects the coexisting bench module + "
                         "the clock64 phase table; the relevance-gate logic is model-agnostic.")
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
    # ── META-VERIFICATION knobs (additive) ───────────────────────────────────────────
    ap.add_argument("--context", choices=_KNOWN_CONTEXTS, default="model-triage",
                    help="which SCREENING CONTEXT this run belongs to (the calibration + "
                         "representativeness are keyed per (model,context)): 'model-triage' "
                         "(DEFAULT, the hand-driven hill-climb), 'compile.py' (the autotuner "
                         "screen tier), or 'level-2' (novel-fusion variant search, wall-only). "
                         "--validate restricts the suite to this context; a normal triage uses "
                         "it to pick the matching calibration for the representativeness check.")
    ap.add_argument("--fidelity-report", action="store_true",
                    help="print the running SCREEN-VS-GATE fidelity per (model,context) from "
                         f"the agreement ledger ({AGREEMENT_LEDGER.name}) and exit. Summarises "
                         "agreement rate + false-positive rate (screen-better, gate-worse).")
    args = ap.parse_args()

    # --fidelity-report is a pure ledger read — no model/scale needed; handle it first.
    if args.fidelity_report:
        _fidelity_report(as_json=args.json)
        return 0
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

    # REPRESENTATIVENESS anchor: the calibration entry (if any) for THIS (model,context)
    # supplies the expected phase-share the run is asserted against. None => the screen is
    # UNCALIBRATED for this context (the share-drift check can't run; verdict is a weak
    # directional hint until a ground-truth is added + --validate passes here).
    calib_entries = _calibration_for(args.model, args.context)
    calib = calib_entries[0] if calib_entries else None
    if calib is None:
        print(f"[triage] ⚠ UNCALIBRATED for {args.model}/{args.context}: no measured "
              f"ground-truth → DIRECTION-ONLY-UNTRUSTED here. Run --validate (and add a "
              f"_CALIBRATION entry) before trusting this screen for this context.", flush=True)

    print(f"[triage] model={args.model} context={args.context} scale: d={args.d} B={args.B} "
          f"reps={args.reps} iters={args.iters}  neutral=±{args.neutral_pct}%  "
          f"relevance-floor={args.relevance_floor}%", flush=True)
    base = _run_variant("baseline", args.baseline_defines, args)
    cand = _run_variant("candidate", args.cand_defines, args)
    summary = _report(base, cand, phase, args.neutral_pct, args.relevance_floor, phase_names,
                      args=args, calib=calib)
    summary.update({"model": args.model, "context": args.context,
                    "calibrated": calib is not None,
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
