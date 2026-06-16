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
_GENCODE = ["-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_90,code=compute_90"]
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
    sys.stderr.write(f"[nvcc-baseline] {len(sources)} TUs, fast_math={fast_math}\n")
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
    args = ap.parse_args()

    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    fast_math = not args.strict_math
    built = _build_vanilla(fast_math)
    _alias_as_ops(built)

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
