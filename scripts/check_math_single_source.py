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

# ---------------------------------------------------------------------------
# Phase 5 WS-F4 — binding-region content-hash drift detection.
#
# The pybind entrypoints in csrc/bindings/bindings.cpp are NOT the canonical
# per-element math (that lives in csrc/algorithms/<opt>.h), but they ARE
# load-bearing glue: they own the boundary validation (check_params_grads /
# check_list_len), the per-param bias-correction scalar construction
# (bc1/bc2 = 1 - beta^step), the host-side grad clipping, and the exact
# (param, grad, state-list, scalar-vector) argument plumbing each launcher reads
# in lockstep. A silent edit here (e.g. dropping a guard, reordering a list,
# changing a bias-correction formula) would not trip the algorithms/<opt>.h
# hashes yet could corrupt the update. So we also content-hash the normalized
# body of each binding entrypoint into the manifest under "binding_regions" and
# fail on drift, forcing a deliberate --update-manifest (the same acknowledgement
# discipline as the canonical-math hashes).
BINDINGS_CPP = "csrc/bindings/bindings.cpp"
BINDING_FUNCS = (
    "supergrok11_fused_step",
    "supergrok15_fused_step",
    "supergrok2_batched_step",
    "supergrok2_prepare_and_batched_step",
    "grokadamw_fused_step",
    "grokfast_fused_ema_adam_step",
    "lion_fused_step",
    "muon_fused_step",
    "neuralgrok_fused_step",
    "prodigy_fused_step",
)


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


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the full text of `fn_name`'s definition (signature .. matching `}`).

    Finds the DEFINITION line — a return type (`void`, `std::tuple<...>`, or
    `std::vector<...>`) at the start of a line followed by `fn_name(` — so a
    pybind `m.def("fn_name", &sg::fn_name)` registration or a forward call site
    is NOT mistaken for the body. Then walks from the first `{` after the
    signature, brace-matching (skipping braces inside // and /* */ comments and
    "…"/'…' literals) to the matching close. Returns "" if not found.
    """
    sig = re.compile(
        r"(?:^|\n)[ \t]*(?:void|std::tuple<[^>]*>|std::vector<[^>]*>)[ \t]+"
        + re.escape(fn_name) + r"[ \t]*\(")
    m = sig.search(src)
    if not m:
        return ""
    start = m.start() if src[m.start()] != "\n" else m.start() + 1
    # Find the opening brace of the body (the first '{' after the ')' that
    # closes the parameter list — but simply scanning for the first top-level
    # '{' from the signature start is sufficient here because the parameter
    # list contains no '{').
    i = m.end()
    n = len(src)
    while i < n and src[i] != "{":
        i += 1
    if i >= n:
        return ""
    depth = 0
    j = i
    in_line_comment = False
    in_block_comment = False
    in_str = None  # '"' or "'" while inside a literal
    while j < n:
        c = src[j]
        nxt = src[j + 1] if j + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                j += 1
        elif in_str is not None:
            if c == "\\":
                j += 1  # skip escaped char
            elif c == in_str:
                in_str = None
        elif c == "/" and nxt == "/":
            in_line_comment = True
            j += 1
        elif c == "/" and nxt == "*":
            in_block_comment = True
            j += 1
        elif c == '"' or c == "'":
            in_str = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1
    return ""  # unbalanced — treated as "not found"


def _binding_region_hash(fn_name: str) -> str:
    """SHA-256 of the comment/whitespace-stripped binding-function body. Empty
    string hashes to a sentinel so a vanished function is DETECTED as drift
    (not silently skipped)."""
    body = _extract_function_body(_read(BINDINGS_CPP), fn_name)
    return hashlib.sha256(_strip_comments_ws(body).encode()).hexdigest()


def current_binding_hashes() -> dict:
    return {fn: _binding_region_hash(fn) for fn in BINDING_FUNCS}


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


# ---------------------------------------------------------------------------
# Phase 5 WS-F5 — fused-consumer structural hardening.
#
# The single fused megakernel optimizer-tail consumer. The OLD check only grepped
# it for the one literal `csrc/algorithms/adamw.h`; that left a missing include
# for any of the OTHER 10 optimizers undetected. We now assert every consumed
# optimizer's canonical header is #included AND that the consumer actually CALLS
# that optimizer's apply symbol (so an include with no call — a dead include that
# silently routes through some other path — is surfaced).
FUSED_CONSUMER = "csrc/fused/sm_90/opt_components.cuh"

# Canonical per-element apply symbol each optimizer's tail is reached through in
# the fused consumer (the call token to grep for). Verified against the current
# opt_components.cuh: adamw_step(33), lion_step(34), grokfast_fused_step(197),
# grokadamw_step(36), looksam_apply_step(37), prodigy_apply_step(38),
# neuralgrok_apply_step(226), muon_update_step(233), sg11_sweep_b_step,
# sg15_sweep_b_step.
_FUSED_APPLY_SYMBOL = {
    "adamw": "adamw_step",
    "lion": "lion_step",
    "grokfast": "grokfast_fused_step",
    "grokadamw": "grokadamw_step",
    "looksam": "looksam_apply_step",
    "prodigy": "prodigy_apply_step",
    "neuralgrok": "neuralgrok_apply_step",
    "muon": "muon_update_step",
    "supergrok11": "sg11_sweep_b_step",
    "supergrok15": "sg15_sweep_b_step",
    # supergrok2 is intentionally absent — see SG2_SEPARATE_PATH below.
}

# SuperGrok2's fused optimizer tail does NOT flow through opt_components.cuh: its
# CSA/HCA meta-net + sg2_apply_step live on a separate launch path
# (launch_csa_hca_batched_step → supergrok2.h / supergrok2_bilevel_adjoint.h),
# which opt_components.cuh itself documents (~L44). So neither `supergrok2.h`
# nor `sg2_apply_step` appears in this consumer by design. Asserting them here
# would hard-fail on the real tree; instead this is a WARNING (the include/symbol
# IS verified for SG2 via the per-op kernels/sm_90 structural check + the
# canonical-math + binding-region hashes), with a TODO to fold SG2 into the
# unified consumer assertion once the sibling's refactor lands.
SG2_SEPARATE_PATH = "supergrok2"


def _emit_warning(msg: str) -> None:
    """Print a clearly-labelled, non-fatal warning. Used for sub-checks that are
    expected to fail mid-refactor (so the guard stays usable and exits 0) but
    must NOT pass silently."""
    sys.stderr.write(f"  [WARN] {msg}\n")
    sys.stderr.flush()


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
                   "csrc/algorithms/<opt>.h math (`hashes`) AND of the pybind "
                   "binding entrypoints in csrc/bindings/bindings.cpp "
                   "(`binding_regions`: boundary guards + bias-correction + "
                   "argument plumbing). Regenerate with `--update-manifest` ONLY "
                   "when intentionally changing the canonical math (and after "
                   "syncing the gfx942/TPU re-expressions) or the binding "
                   "entrypoints.",
                   "hashes": current_hashes(),
                   "binding_regions": current_binding_hashes()},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest updated: {MANIFEST}")
    return 0


def check(structural_only: bool = False) -> list:
    """Return a list of failure strings (empty = OK)."""
    failures = []

    # (1) structural single-source: per-op MUST #include the canonical header.
    for opt in OPTS:
        rel = f"grokking_optimizers/kernels/sm_90/{opt}_sm90.cuh"
        perop = _read(rel)
        if not perop:
            # A missing/renamed consumer would otherwise silently DISABLE drift
            # detection for this optimizer (the old `if perop and ...` guard
            # skipped absent files). Fail loudly instead.
            failures.append(
                f"[structural] expected consumer {rel} is missing/empty — "
                f"drift detection for '{opt}' would be silently disabled. "
                f"Restore the file or update OPTS.")
            continue
        if _canonical_header(opt) not in perop:
            failures.append(
                f"[structural] kernels/sm_90/{opt}_sm90.cuh does NOT #include "
                f"the canonical {_canonical_header(opt)} (reimplementation → "
                f"drift). Make it #include the canonical header.")
    # (1a) fused-consumer structural hardening (WS-F5): the single fused
    # optimizer-tail consumer MUST #include every consumed optimizer's canonical
    # header AND call its apply symbol. FAIL LOUDLY if the consumer is
    # missing/empty (the old `if fused and …` silently disabled the whole check).
    fused = _read(FUSED_CONSUMER)
    if not fused:
        failures.append(
            f"[structural] fused consumer {FUSED_CONSUMER} is missing/empty — "
            f"the fused optimizer-tail single-source check is disabled. Restore "
            f"the file.")
    else:
        for opt in OPTS:
            if opt == SG2_SEPARATE_PATH:
                # Architectural exception (separate launch path) — WARN, do not
                # FAIL. TODO(post-refactor): fold SG2 into this assertion once
                # opt_components.cuh consumes its tail.
                if _canonical_header(opt) not in fused:
                    _emit_warning(
                        f"{FUSED_CONSUMER} does not #include "
                        f"{_canonical_header(opt)} — EXPECTED: SuperGrok2's "
                        f"fused tail uses the separate launch_csa_hca path, not "
                        f"opt_components.cuh. (SG2 single-source is still "
                        f"covered by the per-op kernel + canonical-math + "
                        f"binding-region hashes.) TODO: unify post-refactor.")
                continue
            if _canonical_header(opt) not in fused:
                failures.append(
                    f"[structural] {FUSED_CONSUMER} does NOT #include the "
                    f"canonical {_canonical_header(opt)} (the fused tail would "
                    f"reimplement '{opt}' → drift). Add the #include.")
            # Apply-symbol assertion: include without a call is a dead include.
            # Make this a WARNING (not hard-fail): the sibling's in-flight
            # opt_components.cuh refactor may rename the apply symbol, and the
            # canonical-math hashes already pin the math itself — so a stale
            # symbol name here should keep the guard USABLE, not break the build.
            sym = _FUSED_APPLY_SYMBOL.get(opt)
            if sym and sym not in fused:
                _emit_warning(
                    f"{FUSED_CONSUMER} #includes {_canonical_header(opt)} but "
                    f"does not appear to CALL its apply symbol "
                    f"`{sym}` — the fused tail may route '{opt}' through a "
                    f"different/renamed path (likely the in-flight "
                    f"opt_components.cuh refactor). Re-point _FUSED_APPLY_SYMBOL "
                    f"or restore the call. TODO: harden to a hard-fail "
                    f"post-refactor.")

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
    manifest_data = json.load(open(MANIFEST))
    recorded = manifest_data.get("hashes", {})
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

    # (2b) binding-region drift detection (WS-F4): the pybind entrypoints in
    # bindings.cpp own the boundary guards + bias-correction + argument plumbing.
    recorded_bindings = manifest_data.get("binding_regions", {})
    cur_bindings = current_binding_hashes()
    for fn in BINDING_FUNCS:
        if fn not in recorded_bindings:
            failures.append(
                f"[binding] no recorded hash for {fn} — run --update-manifest.")
            continue
        # A vanished/renamed entrypoint hashes its empty body; surface that as a
        # distinct, louder message than a normal edit-drift.
        if not _extract_function_body(_read(BINDINGS_CPP), fn):
            failures.append(
                f"[binding] entrypoint {fn} not found in {BINDINGS_CPP} "
                f"(removed/renamed?). If intentional, update BINDING_FUNCS and "
                f"--update-manifest.")
        elif recorded_bindings[fn] != cur_bindings[fn]:
            failures.append(
                f"[binding] DRIFT: {BINDINGS_CPP}::{fn} changed "
                f"(hash {cur_bindings[fn][:12]} != recorded "
                f"{recorded_bindings[fn][:12]}). The binding owns the boundary "
                f"guards / bias-correction / list plumbing each launcher reads; "
                f"if the edit is intentional, --update-manifest.")
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
