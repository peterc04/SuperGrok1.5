#!/usr/bin/env python3
"""Stage 0.1 — de-inline shared blocks into real #pragma-once headers.

Fixes two real-nvcc bugs:
  (1) redefinition: each *_sm90.cuh inlines platform/types/affine/utils/ptx
      (inside its primitives block) AND #includes its algorithm header, which
      ALSO inlines types+utils -> double symbol definitions.
  (2) include-order: algorithm headers inline utils (uses WARP_SIZE) but not
      platform (defines WARP_SIZE); included before the .cuh's own platform.

Approach: extract canonical headers (verified byte/semantically identical across
all copies), each #include-ing its prerequisites so macros precede use, with
#pragma once so repeated inclusion within a TU is a no-op. Then replace every
inlined block (begin..end banner) with a single #include.

Block nesting inside a kernel header:
   PRIM { PLAT ; TYPE { AFFN } ; UTIL ; PTX ; <prim-own> }
Algorithm headers contain only: TYPE (no affine) ; UTIL.

Self-verifying: aborts (writing nothing) if banners are unbalanced, if a
canonical body is implausibly small, or if any banner survives the rewrite.
"""
import sys, pathlib, re

ROOT = pathlib.Path("/home/user/SuperGrok1.5")

# former-path : (output header path, [prereq includes])
PLAT = "csrc/common/platform.h"
TYPE = "csrc/common/types.h"
AFFN = "csrc/scan/affine2x2.h"
UTIL = "csrc/common/utils.cuh"
PTX  = "csrc/common/ptx_intrinsics.cuh"
PRIM = "csrc/backends/cuda/sm_90/primitives.cuh"

PREREQS = {
    PLAT: [],
    AFFN: [PLAT],
    TYPE: [PLAT, AFFN],
    UTIL: [PLAT, TYPE, AFFN],
    PTX:  [PLAT, TYPE, AFFN],
    PRIM: [PLAT, TYPE, AFFN, UTIL, PTX],
}
# order to emit / replace (deps first)
ALL = [PLAT, AFFN, TYPE, UTIL, PTX, PRIM]


def b_begin(former): return f"// ── inlined from former {former} ──"
def b_end(former):   return f"// ── end inlined {former} ──"


def banner_idx(lines, former):
    bs = [i for i, l in enumerate(lines) if l.strip() == b_begin(former)]
    es = [i for i, l in enumerate(lines) if l.strip() == b_end(former)]
    return bs, es


def extract_inner(full_lines, plain, former, drop_formers=()):
    """Return the body text between the FIRST begin/end of `former`, with any
    nested `drop_formers` blocks removed (their banners+content stripped)."""
    bs, es = banner_idx(plain, former)
    if not bs:
        return None
    bi, ei = bs[0], es[0]
    inner = list(range(bi + 1, ei))
    drop = set()
    for d in drop_formers:
        dbs, des = banner_idx(plain, d)
        for (db, de) in zip(dbs, des):
            if bi < db and de < ei:
                drop.update(range(db, de + 1))
    keep = [full_lines[i] for i in inner if i not in drop]
    return "".join(keep)


def main():
    sm90 = sorted((ROOT / "grokking_optimizers/kernels/sm_90").glob("*.cuh"))
    gfx = sorted((ROOT / "grokking_optimizers/kernels/gfx942").glob("*.hip.hpp"))
    algos = sorted((ROOT / "csrc/algorithms").glob("*.h"))
    targets = sm90 + gfx + algos

    # ---- 1. pick reference files and extract canonical bodies ----
    ref_kernel = ROOT / "grokking_optimizers/kernels/sm_90/adamw_sm90.cuh"
    ref_algo = ROOT / "csrc/algorithms/adamw.h"

    def load(p):
        full = p.read_text().splitlines(keepends=True)
        plain = [l.rstrip("\n") for l in full]
        return full, plain

    kf, kp = load(ref_kernel)
    af, ap = load(ref_algo)

    bodies = {}
    bodies[PLAT] = extract_inner(kf, kp, PLAT)
    bodies[AFFN] = extract_inner(kf, kp, AFFN)
    # TYPE canonical: from algorithm header (no nested affine)
    bodies[TYPE] = extract_inner(af, ap, TYPE)
    bodies[UTIL] = extract_inner(kf, kp, UTIL)
    bodies[PTX] = extract_inner(kf, kp, PTX)
    # PRIM canonical: drop the nested PLAT/TYPE(+AFFN)/UTIL/PTX, keep prim-own
    bodies[PRIM] = extract_inner(kf, kp, PRIM,
                                 drop_formers=(PLAT, TYPE, AFFN, UTIL, PTX))

    for former in ALL:
        body = bodies[former]
        if body is None:
            print(f"ABORT: could not extract {former}")
            return 2
        n = body.count("\n")
        if n < 3:
            print(f"ABORT: {former} body too small ({n} lines)")
            return 3
        print(f"EXTRACT {former} lines={n}")

    # sanity: TYPE canonical must NOT contain an affine struct (it's separate)
    if "struct Affine2x2" in bodies[TYPE]:
        print("ABORT: TYPE canonical unexpectedly contains Affine2x2")
        return 4
    # sanity: AFFN canonical MUST contain the struct
    if "struct Affine2x2" not in bodies[AFFN]:
        print("ABORT: AFFN canonical missing Affine2x2 struct")
        return 5
    # sanity: PRIM-own must contain grid_stride and NOT contain ptx_fma
    if "grid_stride" not in bodies[PRIM]:
        print("ABORT: PRIM-own missing grid_stride")
        return 6
    if "ptx_fma" in bodies[PRIM]:
        print("ABORT: PRIM-own still contains nested ptx_fma")
        return 7

    # ---- 2. write canonical headers ----
    for former in ALL:
        inc = "".join(f'#include "{p}"\n' for p in PREREQS[former])
        text = (
            "#pragma once\n"
            "// Canonical shared header (Stage 0.1 de-inline). Byte-identical to\n"
            "// the former inlined copy; prerequisites #included so platform\n"
            "// macros (GROK_CUDA/WARP_SIZE/...) precede use; #pragma once dedups.\n"
            f"{inc}{bodies[former]}"
        )
        op = ROOT / former
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(text)
        print(f"WROTE {former} bytes={len(text)}")

    # ---- 3. replace inlined blocks in every target with includes ----
    for f in targets:
        full = f.read_text().splitlines(keepends=True)
        plain = [l.rstrip("\n") for l in full]
        # collect spans for the OUTERMOST blocks only: in kernel files the outer
        # block is PRIM (covers nested ones); algorithm files have TYPE+UTIL at
        # top level. Strategy: replace top-level blocks; a block is top-level if
        # it is not contained within another block we will replace.
        spans = []  # (begin_idx, end_idx, former)
        for former in ALL:
            bs, es = banner_idx(plain, former)
            for (b, e) in zip(bs, es):
                spans.append((b, e, former))
        # keep only top-level spans (not nested inside another span)
        def contained(s, others):
            b, e, _ = s
            for (ob, oe, _) in others:
                if (ob, oe) != (b, e) and ob < b and e < oe:
                    return True
            return False
        top = [s for s in spans if not contained(s, spans)]
        # splice bottom-up
        top.sort(key=lambda s: s[0], reverse=True)
        for (b, e, former) in top:
            nl = "\n" if full[e].endswith("\n") else ""
            full[b:e + 1] = [f'#include "{former}"{nl}']
        f.write_text("".join(full))

    # ---- 4. post-condition: no shared-block banners remain anywhere ----
    orphan = 0
    for f in targets + [ROOT / p for p in ALL]:
        t = f.read_text()
        for former in ALL:
            if (b_begin(former) in t) or (b_end(former) in t):
                print(f"ORPHAN {former} in {f.relative_to(ROOT)}")
                orphan += 1
    if orphan:
        print(f"ABORT_POSTCOND orphan_banners={orphan}")
        return 8

    print("OK_STAGE0_DEINLINE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
