#!/usr/bin/env python3
"""Phase 5 WS2 — ENFORCED optimizer-math single-source / drift guard.

Three real teeth (a comment is NOT enough — this fails the build on drift):

1. STRUCTURAL single-source (CUDA): the canonical per-element math lives once in
   csrc/algorithms/<opt>.h. BOTH CUDA consumers derive from it by #include:
     * per-op : grokking_optimizers/kernels/sm_90/<opt>_sm90.cuh
     * fused  : csrc/fused/sm_90/opt_components.cuh
   If a consumer stops #including the canonical header (i.e. reimplements the
   math), this FAILS — reimplementation is the only way the two CUDA trees could
   drift, and it is now detected.

1b. RE-INLINE detection (Phase 5 WS3): #include alone is not enough — a consumer
   could keep the #include yet still LOCALLY RE-INLINE the Adam moment-update /
   bias-corrected apply instead of CALLING the canonical step. That silently
   re-introduces drift while passing the structural check. So each includable
   CUDA consumer (kernels/sm_90/<opt>_sm90.cuh and the fused
   csrc/fused/sm_90/opt_components.cuh) is scanned for the Adam moment-update +
   apply expressions (`beta1 * exp_avg…`, `beta2 * exp_avg_sq…`, the
   `(m / bc1) / (sqrtf(v / bc2) + eps)` apply). Any match in a consumer FAILS —
   the math must come from a call into csrc/algorithms/<opt>.h, not be re-typed
   in the .cuh. The canonical algorithms/*.h headers themselves are NOT scanned
   (they are the single source and MUST contain the math); the gfx942/TPU
   re-expressions are the sanctioned transcriptions guarded by check (2).

2. CONTENT-HASH manifest (drift detection for the necessary RE-EXPRESSIONS that
   cannot #include the C header — gfx942 transcription, TPU JAX): a committed
   manifest scripts/optimizer_math_manifest.json records the normalized hash of
   each canonical csrc/algorithms/<opt>.h. This guard recomputes the hashes and
   FAILS if any differs from the manifest. So editing the canonical math forces
   a DELIBERATE `--update-manifest` (which is the acknowledgement "I changed the
   canonical math and synced the gfx942/TPU re-expressions") — silent drift is
   impossible.

Modes:
  python3 scripts/check_math_single_source.py                 # verify (exit 0/1)
  python3 scripts/check_math_single_source.py --update-manifest  # re-record hashes
The compile.py --self-test section `math_drift_guard` runs the verify path AND a
prove-it-triggers test (perturb canonical math in-memory → guard must flag).
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "optimizer_math_manifest.json")
OPTS = [
    "adamw", "lion", "grokfast", "grokadamw", "looksam", "prodigy",
    "neuralgrok", "muon", "supergrok11", "supergrok15", "supergrok2",
]


def _canonical_header(opt: str) -> str:
    return f"csrc/algorithms/{opt}.h"


def _read(path: str) -> str:
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return ""
    with open(full) as f:
        return f.read()


def _strip_comments_ws(src: str) -> str:
    """Normalize C/C++ source to its semantic content: drop // and /* */
    comments and all whitespace, so reformatting/comment edits do NOT trip the
    hash but any change to the actual math expressions does."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"\s+", "", src)
    return src


# ---------------------------------------------------------------------------
# Phase 5 WS3 — re-inline detection.
#
# The Adam moment-update + bias-corrected decoupled-WD apply is the math that
# MUST live once in csrc/algorithms/<opt>.h and reach every includable CUDA
# consumer via a CALL (e.g. adamw_step / <opt>_adam_tail), never by being
# re-typed locally. These regexes match the *moment-update / apply* expressions
# specifically (not merely the names `beta1`/`exp_avg`, which legitimately
# appear as function arguments passed INTO the canonical call).
#
# Patterns (whitespace-flexible):
#   beta1 * exp_avg[...]        -> m  = b1*ea + (1-b1)*g   first moment update
#   beta2 * exp_avg_sq[...]     -> v  = b2*eas + (1-b2)*g*g second moment update
#   beta1 * ea... / beta2 * eas -> the quantized-state variant (ea_val/eas_val)
#   (... / bc1) / (sqrtf(... / bc2) ... ) -> the bias-corrected apply
_REINLINE_PATTERNS = [
    re.compile(r"beta1\s*\*\s*exp_avg\b"),
    re.compile(r"beta2\s*\*\s*exp_avg_sq\b"),
    re.compile(r"beta1\s*\*\s*ea(?:_val)?\b"),
    re.compile(r"beta2\s*\*\s*eas(?:_val)?\b"),
    re.compile(r"/\s*bc1\s*\)\s*/\s*\(\s*sqrtf\s*\(.*?/\s*bc2"),
]


def _consumer_headers() -> list:
    """Includable CUDA consumers that MUST obtain the Adam math by call, not by
    re-inlining it. Excludes the canonical algorithms/*.h (which own the math)
    and the gfx942/TPU re-expressions (sanctioned transcriptions, guarded by the
    content-hash manifest)."""
    consumers = [f"grokking_optimizers/kernels/sm_90/{opt}_sm90.cuh"
                 for opt in OPTS]
    consumers.append("csrc/fused/sm_90/opt_components.cuh")
    return consumers


def scan_reinlined_math(path: str) -> list:
    """Return list of (lineno, pattern, line) for re-inlined Adam math in `path`.
    Comments are stripped first so a commented-out reference line (e.g. the
    bit-identity doc) is not flagged."""
    body = _read(path)
    if not body:
        return []
    hits = []
    for lineno, raw in enumerate(body.splitlines(), 1):
        # drop // comments (keep code before them); the doc comments that
        # describe the canonical math are thereby ignored.
        code = re.sub(r"//.*$", "", raw)
        for pat in _REINLINE_PATTERNS:
            if pat.search(code):
                hits.append((lineno, pat.pattern, raw.strip()))
                break
    return hits


def normalized_math_hash(opt: str) -> str:
    """SHA-256 of the comment/whitespace-stripped canonical header."""
    body = _read(_canonical_header(opt))
    return hashlib.sha256(_strip_comments_ws(body).encode()).hexdigest()


def current_hashes() -> dict:
    return {opt: normalized_math_hash(opt) for opt in OPTS}


def update_manifest() -> int:
    with open(MANIFEST, "w") as f:
        json.dump({"_doc": "WS2 drift guard: normalized hashes of the canonical "
                   "csrc/algorithms/<opt>.h math. Regenerate with "
                   "`--update-manifest` ONLY when intentionally changing the "
                   "canonical math (and after syncing the gfx942/TPU "
                   "re-expressions).",
                   "hashes": current_hashes()}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest updated: {MANIFEST}")
    return 0


def check(structural_only: bool = False) -> list:
    """Return a list of failure strings (empty = OK)."""
    failures = []

    # (1) structural single-source: per-op MUST #include the canonical header.
    for opt in OPTS:
        perop = _read(f"grokking_optimizers/kernels/sm_90/{opt}_sm90.cuh")
        if perop and _canonical_header(opt) not in perop:
            failures.append(
                f"[structural] kernels/sm_90/{opt}_sm90.cuh does NOT #include "
                f"the canonical {_canonical_header(opt)} (reimplementation → "
                f"drift). Make it #include the canonical header.")
    fused = _read("csrc/fused/sm_90/opt_components.cuh")
    if fused and "csrc/algorithms/adamw.h" not in fused:
        failures.append("[structural] opt_components.cuh lost its canonical "
                        "#includes.")

    # (1b) re-inline detection: a consumer must CALL the canonical step, never
    # re-type the Adam moment-update / apply locally.
    for path in _consumer_headers():
        for lineno, pat, line in scan_reinlined_math(path):
            failures.append(
                f"[re-inline] {path}:{lineno} re-inlines canonical Adam math "
                f"(matched /{pat}/): `{line}`. This expression MUST come from a "
                f"call into csrc/algorithms/<opt>.h (e.g. <opt>_step / "
                f"<opt>_adam_tail), not be re-typed in the consumer.")

    if structural_only:
        return failures

    # (2) content-hash manifest drift detection.
    if not os.path.exists(MANIFEST):
        failures.append(f"[manifest] {MANIFEST} missing — run --update-manifest.")
        return failures
    recorded = json.load(open(MANIFEST)).get("hashes", {})
    cur = current_hashes()
    for opt in OPTS:
        if opt not in recorded:
            failures.append(f"[manifest] no recorded hash for {opt}.")
        elif recorded[opt] != cur[opt]:
            failures.append(
                f"[manifest] DRIFT: canonical csrc/algorithms/{opt}.h math "
                f"changed (hash {cur[opt][:12]} != recorded {recorded[opt][:12]}). "
                f"If intentional: sync the gfx942 transcription "
                f"(opt_components.hip.hpp) + TPU path, then --update-manifest.")
    return failures


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--update-manifest" in argv:
        return update_manifest()
    failures = check()
    print("=== optimizer-math drift guard (WS2, ENFORCED) ===")
    print(f"canonical: csrc/algorithms/<opt>.h  ({len(OPTS)} optimizers)")
    print(f"manifest:  {os.path.relpath(MANIFEST, ROOT)}")
    if failures:
        print("\nFAILURES:")
        for fr in failures:
            print(f"  ✗ {fr}")
        return 1
    print("\nOK — single-source structural invariant holds AND no canonical-math "
          "drift vs the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
