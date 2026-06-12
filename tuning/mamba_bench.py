"""tuning/mamba_bench.py — production-scale Mamba TC step benchmark (task #24 H1+H3 port).

Mirrors tuning/decoder_bench.py (the decoder_bench is decoder-only). JIT-compiles
the SAME Mamba TC cell TU (csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu) as a
SEPARATELY-NAMED coexisting extension (own build dir + ninja incremental), so a
before/after knob sweep (-D SG_TUNED_MB_GEMM_INTERLEAVE=1 vs default 2) rebuilds
incrementally and NEVER touches the production _ops.so or the 33/33 gate.

USAGE
  python tuning/mamba_bench.py --B 16384 --reps 5
  python tuning/mamba_bench.py --B 16384 --reps 5 -D SG_TUNED_MB_GEMM_INTERLEAVE=1

Notes:
  * Mamba dims are COMPILE-fixed (mamba3_layout); only B varies. Inputs RANDOM.
  * Times the production tc_train_step at ncta_cap=0 (one CTA/SM, full saturation).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_TC_TU = str(ROOT / "csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu")


def build_variant(name: str | None = None, verbose: bool = False,
                  defines: list[str] | None = None):
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
             "-gencode=arch=compute_90a,code=sm_90a",
             "-gencode=arch=compute_90a,code=compute_90a"]
    suffix = ""
    if defines:
        for d_ in defines:
            flags.append("-D" + d_ if not d_.startswith("-D") else d_)
        norm = "_".join(d_.lstrip("-D").replace("=", "").replace(" ", "")
                        for d_ in defines)
        suffix = "_" + norm
    if name is None:
        name = "mega_mamba_real_adamw_tc_bench" + suffix
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mod = load(name=name, sources=[_TC_TU],
               extra_include_paths=[str(ROOT)],
               extra_cuda_cflags=flags,
               extra_cflags=["-O3", "-std=c++17"],
               verbose=verbose)
    return mod


def measure(mod, B: int, reps: int, warmup: int, iters: int):
    from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
    dev = torch.device("cuda")
    total = int(mod.TOTAL)
    g = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g, device=dev).to(torch.int32).contiguous()
    targets = torch.randint(0, P_HEAD, (B,), generator=g, device=dev).to(torch.int32).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    def call():
        return mod.tc_train_step(params, tokens, targets, state, lr, beta1, beta2,
                                 eps, wd, bc1, bc2, 1, 0)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    walls = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) / iters)
    wall = statistics.median(walls)
    return {"B": B, "total_params": total, "wall_ms": wall * 1e3,
            "steps_per_s": 1.0 / wall, "walls_ms": [w * 1e3 for w in walls]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("-D", dest="defines", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be divisible by 16 (dW K-loop is 16-step atoms)"
    print(f"[mamba-bench] building variant defines={args.defines}", flush=True)
    t0 = time.perf_counter()
    mod = build_variant(verbose=args.verbose_build, defines=args.defines)
    print(f"[mamba-bench] build {time.perf_counter()-t0:.1f}s  TILE_M={int(mod.TILE_M)} "
          f"TOTAL={int(mod.TOTAL):,} TILE_N={int(mod.TILE_N)}", flush=True)
    res = measure(mod, args.B, args.reps, args.warmup, args.iters)
    print(f"\n[mamba-bench] B={res['B']}  params={res['total_params']:,}", flush=True)
    print(f"  wall/step = {res['wall_ms']:.4f} ms  "
          f"(median of {[f'{x:.3f}' for x in res['walls_ms']]})", flush=True)
    print(f"  steps/s   = {res['steps_per_s']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    main()
