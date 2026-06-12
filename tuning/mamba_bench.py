"""tuning/mamba_bench.py — Mamba TC step benchmark + d-scaled size-ladder variant (task #24).

Mirrors tuning/decoder_bench.py. JIT-compiles the SAME Mamba TC cell TU
(csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu) as a SEPARATELY-NAMED coexisting
extension (own build dir + ninja incremental), so a before/after knob sweep
(-D SG_TUNED_MB_GEMM_INTERLEAVE=1 vs default 2) — or the d-scaled size-ladder build
— rebuilds incrementally and NEVER touches the production _ops.so or the 33/33 gate.

  * --d 1024  -> -DSG_MB_BENCH_LAYOUT=1 (the d=1024 branch of mamba3_layout.cuh, the
                 size-ladder roofline width: d_inner=2048, dt_rank=64) +
                 -DSG_MB_SCALAR_MEGAKERNEL=0 (the scalar megakernel's MambaSampleSmem
                 overflows the ~228 KB/SM budget at d=1024, so gate it OFF — the TC
                 engine the bench drives uses a small static MbTcSmem). Mirrors the
                 decoder bench's -DSG_DEC_BENCH_LAYOUT=1 / -DSG_DEC_SCALAR_MEGAKERNEL=0.
  * --d 128   -> production branch (default; SG_MB_BENCH_LAYOUT unset).

USAGE
  # Size-ladder d=1024 build, 3 reps (report ms + steps/s):
  python tuning/mamba_bench.py --d 1024 --B 16384 --reps 3
  # Production d=128 B=16384 median wall:
  python tuning/mamba_bench.py --d 128 --B 16384 --reps 5
  python tuning/mamba_bench.py --d 128 --B 16384 --reps 5 -D SG_TUNED_MB_GEMM_INTERLEAVE=1

Notes:
  * Mamba dims are COMPILE-fixed (the layout is a compile-time constant); --d selects
    the branch. Inputs RANDOM. params/state are sized from the TU's exposed mod.TOTAL.
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


def build_variant(d: int = 128, name: str | None = None, verbose: bool = False,
                  defines: list[str] | None = None):
    bench = (d != 128)
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
             "-gencode=arch=compute_90a,code=sm_90a",
             "-gencode=arch=compute_90a,code=compute_90a"]
    if bench:
        # d-scaled size-ladder layout branch + gate OFF the scalar megakernel (its
        # MambaSampleSmem overflows the ~228 KB/SM budget at large d; the TC path the
        # bench drives uses the small static MbTcSmem). Mirrors the decoder bench.
        flags.append("-DSG_MB_BENCH_LAYOUT=1")
        flags.append("-DSG_MB_SCALAR_MEGAKERNEL=0")
    suffix = ""
    if defines:
        for d_ in defines:
            flags.append("-D" + d_ if not d_.startswith("-D") else d_)
        norm = "_".join(d_.lstrip("-D").replace("=", "").replace(" ", "")
                        for d_ in defines)
        suffix = "_" + norm
    if name is None:
        name = "mega_mamba_real_adamw_tc"
        name += "_bench" if bench else "_prod"
        name += suffix
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mod = load(name=name, sources=[_TC_TU],
               extra_include_paths=[str(ROOT)],
               extra_cuda_cflags=flags,
               extra_cflags=["-O3", "-std=c++17"],
               verbose=verbose)
    return mod


def measure(mod, d: int, B: int, reps: int, warmup: int, iters: int):
    # VOCAB/P_HEAD/SEQ are d-INVARIANT (token range, head width, seq len), so the
    # oracle constants are correct at every ladder width; the layout's D/DINNER scale.
    from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
    assert int(mod.D) == d, f"TU compiled at D={int(mod.D)}, expected {d}"
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
    # Analytic dense-GEMM FLOPs for one fwd+bwd Mamba step (the matmul work the
    # roofline counts; the SSM scan + conv + elementwise ≈ 0). T = B*seq rows. Per
    # layer the projections are in_proj(d->2*d_inner), x_proj(d_inner->dbc),
    # dt_proj(dt_rank->d_inner), out_proj(d_inner->d). Plus the head (d->phead, M=B).
    # Backward ≈ 2x forward (dX+dW), so fwd+bwd ≈ 3x forward.
    layers, seq = int(mod.LAYERS), int(mod.SEQ)
    di, dtr, st = int(mod.DINNER), int(mod.DTRANK), int(mod.STATE)
    dbc = dtr + 2 * st
    T = B * seq
    fwd = 0.0
    for _ in range(layers):
        fwd += 2.0 * T * (2 * di) * d     # in_proj
        fwd += 2.0 * T * dbc * di         # x_proj
        fwd += 2.0 * T * di * dtr         # dt_proj
        fwd += 2.0 * T * d * di           # out_proj
    fwd += 2.0 * B * P_HEAD * d           # head (M=B)
    flops = 3.0 * fwd
    return {"d": d, "B": B, "total_params": total, "wall_ms": wall * 1e3,
            "steps_per_s": 1.0 / wall, "walls_ms": [w * 1e3 for w in walls],
            "gemm_flops_per_step": flops, "achieved_tf_s": flops / wall / 1e12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128, help="Mamba d_model (128 prod | 1024 bench)")
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("-D", dest="defines", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be divisible by 16 (dW K-loop is 16-step atoms)"
    print(f"[mamba-bench] building variant: d={args.d} "
          f"(SG_MB_BENCH_LAYOUT={'1' if args.d != 128 else 'unset'})"
          + (f"  defines={args.defines}" if args.defines else ""), flush=True)
    t0 = time.perf_counter()
    mod = build_variant(args.d, verbose=args.verbose_build, defines=args.defines)
    print(f"[mamba-bench] build {time.perf_counter()-t0:.1f}s  D={int(mod.D)} DINNER={int(mod.DINNER)} "
          f"DTRANK={int(mod.DTRANK)} TILE_M={int(mod.TILE_M)} TOTAL={int(mod.TOTAL):,} "
          f"TILE_N={int(mod.TILE_N)}", flush=True)
    res = measure(mod, args.d, args.B, args.reps, args.warmup, args.iters)
    print(f"\n[mamba-bench] d={res['d']}  B={res['B']}  params={res['total_params']:,}", flush=True)
    print(f"  wall/step = {res['wall_ms']:.4f} ms  "
          f"(median of {[f'{x:.3f}' for x in res['walls_ms']]})", flush=True)
    print(f"  steps/s   = {res['steps_per_s']:.4f}", flush=True)
    print(f"  GEMM FLOPs/step = {res['gemm_flops_per_step']:.3e}  "
          f"=> achieved {res['achieved_tf_s']:.3f} TF/s "
          f"({100.0 * res['achieved_tf_s'] / 989.0:.2f}% of 989 TF/s bf16 dense roofline)",
          flush=True)
    return res


if __name__ == "__main__":
    main()
