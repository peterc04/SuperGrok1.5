"""scripts/nvcc_baseline.py — a TRUE "regular nvcc" baseline build of the _ops
extension, for the Task-#11 three-point comparison:

    A  regular nvcc        (this script)          — plain -O3 [+ --use_fast_math],
                                                     NO compile.py ptxas tuning
    B  compile.py default  (build.sh / setup.py)  — augmented flags, in-source
                                                     SG_TUNED defaults  (A→B = the
                                                     compile.py *pipeline* benefit)
    C  compile.py JIT-tuned (autotuner winner)     — B + SG_TUNED knob search
                                                     (B→C = the *autotuner* benefit)

It compiles the SAME 6 production translation units (bindings, dispatch, the 3
mega_<model>_real_adamw_tc_launcher.cu, sg2_meta_tail.cu) as setup.py, but with a
VANILLA nvcc flag set: only the flags needed to *compile* (gencode, CUTLASS +
feature -D macros, NDEBUG, fPIC, --expt-relaxed-constexpr) plus the obvious
-O3 [+ --use_fast_math] a competent dev would write — and explicitly NONE of
compile.py's ptxas micro-tuning (--register-usage-level, --def-load-cache/
--def-store-cache, --extra-device-vectorization, --maxrregcount, -Xfatbin
-compress-all). So B-vs-A isolates exactly what those tuning flags buy.

The freshly-built .so is aliased as ``grokking_optimizers._ops`` (the proven
idiom from grokking_optimizers/tune_hook.py:_alias_variant_ops) so the production
dispatch + the fp64 gate run against THIS binary, then we time the cell with
CUDA events exactly like scripts/time_cell.py.

Usage:
  python scripts/nvcc_baseline.py --model decoder --opt adamw \
      --seeds 42,7,123 --warmup 10 --iters 30 --reps 5 [--strict-math]
Prints one TIMING json line per seed (median per-step wall-ms).
"""
import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load as _cpp_load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# 1) Resolve the EXACT production source set (mirrors setup.py's glob+filter).
# --------------------------------------------------------------------------
def _owns_extension_module(path: str) -> bool:
    if not path.endswith(".cu"):
        return False
    try:
        with open(path, "r", errors="ignore") as fh:
            return "PYBIND11_MODULE(TORCH_EXTENSION_NAME" in fh.read()
    except OSError:
        return False


def _resolve_sources() -> list:
    pats = ["csrc/backends/cuda/sm_90/*.cu",
            "csrc/backends/cuda/sm_90/models/*.cu",
            "csrc/fused/sm_90/*.cu"]
    out = []
    for pat in pats:
        out += sorted(glob.glob(os.path.join(ROOT, pat)))
    common = [os.path.join(ROOT, "csrc/bindings/bindings.cpp"),
              os.path.join(ROOT, "csrc/bindings/dispatch.cpp")]
    cu = [s for s in out
          if "_overlay" not in os.path.basename(s)
          and not s.endswith("_selftest.cu")
          and not _owns_extension_module(s)]
    return common + cu


# --------------------------------------------------------------------------
# 2) VANILLA flag sets (regular nvcc) — NO compile.py ptxas tuning.
# --------------------------------------------------------------------------
# Structural defines the kernels need to COMPILE (gate #ifdef branches); these
# are NOT optimization flags — dropping them would change which code compiles,
# not how fast it runs. Mirrors setup.py's cuda_define_macros + CUTLASS block.
_STRUCTURAL_DEFINES = [
    "-DWITH_CUDA", "-DWITH_CUTLASS",
    "-DCUTLASS_NVCC_ARCHS=90a", "-DCUTLASS_NVCC_ARCHS_SUPPORTED=90a",
    "-DCUDA_TMA_ENABLED=1", "-DCUDA_WGMMA_ENABLED=1",
    "-DCUDA_CLUSTER_ENABLED=1", "-DCUDA_FP8_ENABLED=1",
    "-DCUDA_FORCE_CDP1_IF_SUPPORTED", "-DNDEBUG",
]
# sm_90a ONLY (cubin + matching PTX). The megakernel emits wgmma.* PTX, which
# requires the ``a`` (architecture-specific) target — assembling it for plain
# ``compute_90`` makes ptxas reject every wgmma instruction ("not supported on
# .target 'sm_90'"). So the embedded-PTX line must also be compute_90a (mirrors
# setup.py's _supported_gencode(min_cc=90) Hopper/CUTLASS path).
_GENCODE = ["-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_90a,code=compute_90a"]
_INCLUDES = [".", "csrc/bindings", "csrc",
             "third_party/cutlass/include",
             "third_party/cutlass/tools/util/include"]


def _vanilla_nvcc(fast_math: bool) -> list:
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"]
    if fast_math:
        flags.append("--use_fast_math")
    return flags + _STRUCTURAL_DEFINES + _GENCODE


def _vanilla_cxx(fast_math: bool) -> list:
    flags = ["-O3", "-std=c++17", "-fPIC"]
    if fast_math:
        flags.append("-ffast-math")
    return flags + ["-DWITH_CUDA", "-DWITH_CUTLASS", "-DNDEBUG"]


def _build_vanilla(fast_math: bool):
    sources = _resolve_sources()
    # Pin the arch so torch's BuildExtension does NOT auto-detect and inject its
    # own `-gencode arch=compute_90,code=compute_90` (the non-`a` PTX target),
    # which makes ptxas reject the megakernel's wgmma.* PTX ("not supported on
    # .target 'sm_90'"). 9.0a => torch emits sm_90a/compute_90a only, matching
    # _GENCODE above and the production/bench builds. (setdefault is not enough —
    # an inherited value would leak the default; set it explicitly.)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    sys.stderr.write(f"[nvcc-baseline] {len(sources)} TUs, fast_math={fast_math}, "
                     f"TORCH_CUDA_ARCH_LIST=9.0a\n")
    sys.stderr.write("[nvcc-baseline] nvcc flags: "
                     + " ".join(_vanilla_nvcc(fast_math)) + "\n")
    # name="_ops" so PYBIND11_MODULE(TORCH_EXTENSION_NAME) emits PyInit__ops,
    # matching what grokking_optimizers._ops expects.
    mod = _cpp_load(
        name="_ops",
        sources=sources,
        extra_cflags=_vanilla_cxx(fast_math),
        extra_cuda_cflags=_vanilla_nvcc(fast_math),
        extra_include_paths=[os.path.join(ROOT, p) for p in _INCLUDES],
        build_directory=_ensure_build_dir(fast_math),
        verbose=True,
    )
    return mod


def _ensure_build_dir(fast_math: bool) -> str:
    tag = "fastmath" if fast_math else "strict"
    d = os.path.join("/workspace", "nvcc_baseline_build", tag)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 3) Alias the freshly-built .so as grokking_optimizers._ops (tune_hook idiom).
# --------------------------------------------------------------------------
def _alias_as_ops(built_module) -> None:
    # The cpp_extension.load() return value is already the imported module
    # object; evict any stale grokking_optimizers._ops (+ submodules + dispatch
    # + optimizers that memoize the resolved ops) so the production dispatch
    # binds THIS binary, then alias it.
    for k in [k for k in list(sys.modules)
              if k == "grokking_optimizers._ops"
              or k.startswith("grokking_optimizers._ops.")
              or k == "grokking_optimizers.dispatch"]:
        del sys.modules[k]
    sys.modules["grokking_optimizers._ops"] = built_module


# --------------------------------------------------------------------------
# 4) Time the cell with CUDA events (mirrors scripts/time_cell.py).
# --------------------------------------------------------------------------
def _time_cell(model: str, opt: str, seed: int, warmup: int, iters: int,
               reps: int) -> dict:
    os.environ["GATE_SEED"] = str(seed)
    import tests.hw.test_l3tc_tail_gate as G
    from grokking_optimizers.dispatch import fused_train_step  # noqa: F401
    g, c, m, data, dev = G._build_cell(model)
    tx, ty = data[0], data[1]
    spec = G._CELLS[f"{opt}/{model}"]
    opt_obj = spec["factory"](g, m.parameters(), c)

    def _step():
        fused_train_step(m, opt_obj, tx, ty)

    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()
    medians = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _step()
        end.record()
        torch.cuda.synchronize()
        medians.append(start.elapsed_time(end) / iters)
    return {"model": model, "opt": opt, "seed": seed,
            "median_ms": statistics.median(medians),
            "min_ms": min(medians),
            "walls_ms": [round(x, 4) for x in medians]}


# ==========================================================================
# TASK #11 — the d=2048 roofline 3-point comparison (compile.py vs nvcc).
# --------------------------------------------------------------------------
# The full-_ops path above builds at the PRODUCTION toy d=128 config (the only
# config the production cell + fp64 gate run at). "See the real difference"
# wants the comparison at the d=2048 ROOFLINE scale where the kernel actually
# stresses the GEMM pipe — but only the coexisting BENCH translation unit
# (csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu, -DSG_DEC_BENCH_LAYOUT=1 →
# the _DEC_BENCH_D=2048 layout branch) compiles at d=2048. So the d=2048 leg
# builds THAT single TU three ways and times its tc_train_step (the SAME path
# tuning/decoder_bench.py uses), differing ONLY in (1) the nvcc flag set and
# (2) the dW split-K macro:
#
#   A  regular nvcc        : VANILLA flags (-O3 [+--use_fast_math] -arch=sm_90a
#                            + only the structural -D needed to compile), NO
#                            compile.py ptxas tuning;            split-K = 4.
#   B  compile.py default  : compile.py's AUGMENTED flags (NVCC_DEVICE_BASE +
#                            the CUDA-version-gated ptxas tuning ladder +
#                            grokking's --use_fast_math -DNDEBUG, exactly the
#                            setup.py production line);          split-K = 4.
#   C  compile.py tuned    : same augmented flags + the TUNED knob (the
#                            in-source default);                 split-K = 2.
#
# A→B isolates the compilation-pipeline benefit (the ptxas-tuning flags);
# B→C isolates the autotuner's knob benefit; A→C is the total. fast-math is
# held IDENTICAL across A/B/C (production B uses it — see setup.py:643), so
# A→B is purely the ptxas-tuning flags, not a numerics confound.
# ==========================================================================

# Structural -D the bench TU needs to compile at d=2048 (mirrors
# tuning/decoder_bench.build_variant: the d!=128 bench branch). NOT opt flags.
_BENCH_STRUCTURAL = [
    "-DSG_TUNED_GEMM_IMPL=1",     # select the wgmma cell driver
    "-DSG_DEC_BENCH_LAYOUT=1",    # → the _DEC_BENCH_D=2048 layout branch
    "-DSG_DEC_SCALAR_MEGAKERNEL=0",  # gate off the legacy scalar megakernel
                                     # (its smem grows with d and ptxas
                                     # hard-stops at d>=768; never driven here)
]
_BENCH_GENCODE = ["-gencode=arch=compute_90a,code=sm_90a",
                  "-gencode=arch=compute_90a,code=compute_90a"]
_BENCH_TU = os.path.join(ROOT, "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu")


def _vanilla_bench_nvcc(fast_math: bool) -> list:
    """Variant A — what a competent dev would hand-write: -O3 [+--use_fast_math]
    -arch=sm_90a + the structural defines. Explicitly NONE of compile.py's
    ptxas micro-tuning (no --register-usage-level / --def-load-cache /
    --def-store-cache / --extra-device-vectorization / -Xfatbin -compress-all /
    --opt-level=3 / -v)."""
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"]
    if fast_math:
        flags.append("--use_fast_math")
    return flags + _BENCH_GENCODE + _BENCH_STRUCTURAL


def _augmented_bench_nvcc() -> list:
    """Variants B/C — the EXACT production compile.py flag line for this single
    TU on this toolchain: NVCC_DEVICE_BASE + the CUDA-version-gated ptxas
    tuning ladder (probed live) + grokking's project flags (--use_fast_math
    -DNDEBUG -lineinfo), exactly as setup.py composes it. fast-math ON to match
    A (and production)."""
    from grokking_optimizers.compile import (NVCC_DEVICE_BASE,
                                             _newer_compiler_flags)
    base = list(NVCC_DEVICE_BASE)
    # The CUDA-version-gated additions (register-usage-level=10 on 12.3+,
    # def-load-cache=ca, def-store-cache=wb, --threads, --diag-suppress, …) —
    # the SAME ladder the production/autotuner device compile uses.
    # _newer_compiler_flags returns (extra_host, extra_device); take device.
    try:
        _host, gated = _newer_compiler_flags("sm_90")
    except Exception as exc:  # pragma: no cover - never block the comparison
        sys.stderr.write(f"[nvcc-baseline] WARN: _newer_compiler_flags failed "
                         f"({exc}); proceeding with NVCC_DEVICE_BASE only\n")
        gated = []
    # grokking project flags (setup.py:643) + bench structural + gencode.
    return (base + list(gated) + ["--use_fast_math", "-DNDEBUG", "-lineinfo"]
            + _BENCH_GENCODE + _BENCH_STRUCTURAL)


def _build_bench_variant(label: str, nvcc_flags: list, splitk: int,
                         fast_math: bool, verbose: bool):
    """JIT-build the d=2048 decoder bench TU with a fully-specified nvcc flag
    set + a dW split-K override. Distinct module name + build dir so the three
    variants coexist (and never touch the production _ops.so)."""
    from torch.utils.cpp_extension import load as _load
    flags = list(nvcc_flags) + [f"-DSG_TUNED_DEC_DW_SPLITK={splitk}"]
    name = f"mega_decoder_d2048_task11_{label}_sk{splitk}"
    bdir = os.path.join("/workspace", "task11_bench_build", f"{label}_sk{splitk}")
    os.makedirs(bdir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    cxx = ["-O3", "-std=c++17"]
    if fast_math:
        cxx.append("-ffast-math")
    sys.stderr.write(f"[nvcc-baseline][{label}] split-K={splitk} fast_math={fast_math}\n")
    sys.stderr.write(f"[nvcc-baseline][{label}] nvcc flags: {' '.join(flags)}\n")
    mod = _load(name=name, sources=[_BENCH_TU],
                extra_include_paths=[ROOT],
                extra_cuda_cflags=flags,
                extra_cflags=cxx,
                build_directory=bdir,
                verbose=verbose)
    return mod


def _time_bench_seed(DB, mod, d: int, batch: int, seed: int,
                     reps: int, warmup: int, iters: int):
    """Median step-ms for ONE seed: inputs reseeded per-seed (params/tokens from
    a per-seed generator — the genuine 3-seed coverage, vs decoder_bench.measure
    which hardcodes seed 42). Shapes/work are identical across seeds, so this is
    the protocol's timing-variance probe at the d=2048 roofline (correctness +
    determinism are the SEPARATE fp64 gate). Mirrors roofline_bench.time_seed."""
    import time as _t
    dev = torch.device("cuda")
    total = int(mod.TOTAL)
    vocab, seq = int(mod.VOCAB), int(mod.SEQ)
    g = torch.Generator(device=dev).manual_seed(seed)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    tokens = torch.randint(0, vocab, (batch, seq), generator=g,
                           device=dev).to(torch.int32).contiguous()
    targets = torch.randint(0, vocab, (batch,), generator=g,
                            device=dev).to(torch.int32).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    def call():
        return mod.tc_train_step(params, tokens, targets, state, lr, beta1,
                                 beta2, eps, wd, bc1, bc2, 1, 0)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    walls = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
        walls.append((_t.perf_counter() - t0) / iters * 1e3)
    return statistics.median(walls), [round(w, 4) for w in walls]


def _run_d2048_3point(args) -> int:
    """A/B/C build + 3-seed median step-ms + TF/s + deltas at d=2048."""
    import tuning.decoder_bench as DB
    import time as _time
    fast_math = not args.strict_math
    d, batch = 2048, args.batch
    assert batch % 16 == 0, "B must be divisible by 16"

    variants = [
        ("A", "regular-nvcc (vanilla flags, split-K=4)",
         _vanilla_bench_nvcc(fast_math), 4),
        ("B", "compile.py-default (augmented flags, split-K=4)",
         _augmented_bench_nvcc(), 4),
        ("C", "compile.py-tuned (augmented flags, split-K=2)",
         _augmented_bench_nvcc(), 2),
    ]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {}
    for label, desc, nvcc_flags, splitk in variants:
        sys.stderr.write(f"\n[nvcc-baseline] ===== variant {label}: {desc} =====\n")
        t0 = _time.perf_counter()
        mod = _build_bench_variant(label, nvcc_flags, splitk, fast_math,
                                   args.verbose_build)
        build_s = _time.perf_counter() - t0
        assert int(mod.D) == d, f"variant {label} compiled D={int(mod.D)} != {d}"
        flops = DB._decoder_gemm_flops_per_step(
            d, batch, int(mod.LAYERS), int(mod.SEQ), int(mod.VOCAB))
        per_seed, per_seed_walls = {}, {}
        for s in seeds:
            med_s, walls = _time_bench_seed(DB, mod, d, batch, s, args.reps,
                                            args.warmup, args.iters)
            per_seed[s] = med_s
            per_seed_walls[s] = walls
        med = statistics.median(per_seed.values())
        tf_s = flops / (med / 1e3) / 1e12
        rec = {"variant": label, "desc": desc, "split_k": splitk,
               "fast_math": fast_math, "d": d, "B": batch,
               "total_params": int(mod.TOTAL), "build_s": round(build_s, 1),
               "per_seed_median_ms": {s: round(per_seed[s], 4) for s in seeds},
               "per_seed_walls_ms": per_seed_walls,
               "across_seed_median_ms": round(med, 4),
               "achieved_tf_s": round(tf_s, 3),
               "pct_roofline": round(100.0 * tf_s / 989.0, 3)}
        results[label] = rec
        sys.stdout.write("VARIANT " + json.dumps(rec) + "\n")
        sys.stdout.flush()

    # Deltas (negative = faster). A→B = pipeline/flags; B→C = autotuner; A→C = total.
    a, b, c = (results[k]["across_seed_median_ms"] for k in "ABC")
    deltas = {
        "A_to_B_pipeline_flags_pct": round(100.0 * (b - a) / a, 3),
        "B_to_C_autotuner_pct": round(100.0 * (c - b) / b, 3),
        "A_to_C_total_pct": round(100.0 * (c - a) / a, 3),
    }
    sys.stdout.write("DELTAS " + json.dumps(deltas) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="decoder")
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--strict-math", action="store_true",
                    help="Drop --use_fast_math too (the strictest vanilla "
                         "baseline). Default keeps --use_fast_math, isolating "
                         "ONLY compile.py's ptxas micro-tuning.")
    ap.add_argument("--bench-d2048", action="store_true",
                    help="Task-#11 d=2048 roofline 3-point comparison: build "
                         "the decoder bench TU at d=2048 three ways (A vanilla "
                         "nvcc / B compile.py-default / C compile.py-tuned) and "
                         "report 3-seed median step-ms + TF/s + the A→B/B→C/A→C "
                         "deltas. Only the adamw/decoder cell.")
    ap.add_argument("--batch", type=int, default=4096,
                    help="d=2048 bench batch (must be %%16==0). Default 4096 "
                         "matches the kernel-track split-K measurement.")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--gate-cell", default="",
                    help="After building the VANILLA production _ops, run the "
                         "fp64 parity gate for this cell (e.g. adamw/decoder) "
                         "against the freshly-aliased vanilla binary, then exit "
                         "(no timing). The correctness leg for variant A — set "
                         "GATE_SEED in the env before invoking.")
    args = ap.parse_args()

    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    if args.bench_d2048:
        return _run_d2048_3point(args)

    fast_math = not args.strict_math
    built = _build_vanilla(fast_math)
    _alias_as_ops(built)

    if args.gate_cell:
        # Correctness leg (variant A): the fp64 parity gate against the vanilla
        # binary. GATE_SEED is frozen at import-time inside the gate module, so
        # it MUST already be set in the env (the caller does this per seed).
        sys.stderr.write(f"[nvcc-baseline] built+aliased VANILLA _ops; running "
                         f"fp64 gate for {args.gate_cell} "
                         f"(GATE_SEED={os.environ.get('GATE_SEED', '42')})...\n")
        import tests.hw.test_l3tc_tail_gate as G
        ok, detail = G.run_cell_gate(args.gate_cell, verbose=True)
        sys.stdout.write(f"GATE {args.gate_cell} "
                         + json.dumps({"seed": int(os.environ.get('GATE_SEED', '42')),
                                       "pass": bool(ok),
                                       "detail": str(detail)}) + "\n")
        sys.stdout.flush()
        return 0 if ok else 1

    sys.stderr.write("[nvcc-baseline] built + aliased as _ops; timing...\n")
    for s in args.seeds.split(","):
        s = s.strip()
        if not s:
            continue
        rec = _time_cell(args.model, args.opt, int(s),
                         args.warmup, args.iters, args.reps)
        sys.stdout.write("TIMING " + json.dumps(rec) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
