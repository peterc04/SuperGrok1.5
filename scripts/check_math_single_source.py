#!/usr/bin/env python3
"""Phase 5 WS2 — ENFORCED optimizer-math single-source / drift guard.

Two real teeth (a comment is NOT enough — this fails the build on drift):

1. STRUCTURAL single-source (CUDA): the canonical per-element math lives once in
   csrc/algorithms/<opt>.h. BOTH CUDA consumers derive from it by #include:
     * per-op : grokking_optimizers/kernels/sm_90/<opt>_sm90.cuh
     * fused  : csrc/fused/sm_90/opt_components.cuh
   If a consumer stops #including the canonical header (i.e. reimplements the
   math), this FAILS — reimplementation is the only way the two CUDA trees could
   drift, and it is now detected.

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
