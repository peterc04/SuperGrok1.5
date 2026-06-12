"""tuning/vit_bench.py — ViT TC step benchmark + d-scaled size-ladder variant (task #24).

Mirrors tuning/decoder_bench.py. JIT-compiles the SAME ViT TC cell TU
(csrc/fused/sm_90/mega_vit_real_adamw_tc.cu) as a SEPARATELY-NAMED coexisting
extension (own torch-extensions build dir + ninja incremental + sccache), so a
before/after knob sweep (-D SG_TUNED_VIT_GEMM_INTERLEAVE=1 vs default 2) — or the
d-scaled size-ladder build — rebuilds incrementally and NEVER touches the
production _ops.so or the 33/33 gate.

  * --d 1024  -> -DSG_VIT_BENCH_LAYOUT=1 (the d=1024 branch of vit_layout.cuh, the
                 size-ladder roofline width, HEADS=16 via the d/64 head-dim rule) +
                 -DSG_VIT_SCALAR_MEGAKERNEL=0 (the scalar megakernel's VitSampleSmem
                 overflows the 227 KB dynamic-smem cap at d=1024, so gate it OFF —
                 the TC engine the bench drives uses a small static VitTcSmem). Mirrors
                 the decoder bench's -DSG_DEC_BENCH_LAYOUT=1 / -DSG_DEC_SCALAR_MEGAKERNEL=0.
  * --d 128   -> production branch (default; SG_VIT_BENCH_LAYOUT unset).

USAGE
  # Size-ladder d=1024 build, 3 reps (report ms + steps/s):
  python tuning/vit_bench.py --d 1024 --B 16384 --reps 3
  # Production d=128 B=16384 median wall:
  python tuning/vit_bench.py --d 128 --B 16384 --reps 5
  # Old serial GEMM (interleave width 1) for the before/after A/B:
  python tuning/vit_bench.py --d 128 --B 16384 --reps 5 -D SG_TUNED_VIT_GEMM_INTERLEAVE=1

Notes:
  * ViT width is COMPILE-fixed (the layout is a compile-time constant); --d selects
    the branch. Inputs are RANDOM (throughput, not a parity gate — correctness is the
    test suite's job). params/state are sized from the TU's exposed mod.TOTAL.
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

_TC_TU = str(ROOT / "csrc/fused/sm_90/mega_vit_real_adamw_tc.cu")


def build_variant(d: int = 128, name: str | None = None, verbose: bool = False,
                  defines: list[str] | None = None):
    bench = (d != 128)
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
             "-gencode=arch=compute_90a,code=sm_90a",
             "-gencode=arch=compute_90a,code=compute_90a"]
    if bench:
        # d-scaled size-ladder layout branch + gate OFF the scalar megakernel
        # (its VitSampleSmem overflows the 227 KB dynamic-smem cap at large d; the TC
        # path the bench drives uses the small static VitTcSmem). Mirrors the decoder.
        flags.append("-DSG_VIT_BENCH_LAYOUT=1")
        flags.append("-DSG_VIT_SCALAR_MEGAKERNEL=0")
    suffix = ""
    if defines:
        for d_ in defines:
            flags.append("-D" + d_ if not d_.startswith("-D") else d_)
        norm = "_".join(d_.lstrip("-D").replace("=", "").replace(" ", "")
                        for d_ in defines)
        suffix = "_" + norm
    if name is None:
        name = "mega_vit_real_adamw_tc"
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
    # NUM_PATCHES/PATCH_DIM/VOCAB are d-INVARIANT (image patch geometry + head width),
    # so the oracle constants are correct at every ladder width; the layout's D scales.
    from tests.hw.vit_oracle import NUM_PATCHES, PATCH_DIM, VOCAB
    assert int(mod.D) == d, f"TU compiled at D={int(mod.D)}, expected {d}"
    dev = torch.device("cuda")
    total = int(mod.TOTAL)
    g = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    patches = torch.randn(B, NUM_PATCHES, PATCH_DIM, generator=g, device=dev).to(torch.float32).contiguous()
    targets = torch.randint(0, VOCAB, (B,), generator=g, device=dev).to(torch.int32).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    def call():
        return mod.tc_train_step(params, patches, targets, state, lr, beta1, beta2,
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
    # Analytic dense-GEMM FLOPs for one fwd+bwd ViT step (the matmul work the roofline
    # counts; LN/softmax/elementwise ≈ 0). T = B*seq rows. Per layer: in_proj(d->3d),
    # out_proj(d->d), ff0(d->4d), ff2(4d->d). Plus the CLS head (d->vocab, M=B).
    # Backward ≈ 2x forward (dX + dW), so fwd+bwd ≈ 3x forward.
    layers, seq, dff = int(mod.LAYERS), int(mod.SEQ), 4 * d
    T = B * seq
    fwd = 0.0
    for _ in range(layers):
        fwd += 2.0 * T * (3 * d) * d      # in_proj
        fwd += 2.0 * T * d * d            # out_proj
        fwd += 2.0 * T * dff * d          # ff0
        fwd += 2.0 * T * d * dff          # ff2
    fwd += 2.0 * B * VOCAB * d            # CLS head (M=B)
    flops = 3.0 * fwd
    return {"d": d, "B": B, "total_params": total, "wall_ms": wall * 1e3,
            "steps_per_s": 1.0 / wall, "walls_ms": [w * 1e3 for w in walls],
            "gemm_flops_per_step": flops, "achieved_tf_s": flops / wall / 1e12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128, help="ViT width (128 prod | 1024 bench)")
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("-D", dest="defines", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be divisible by 16 (dW K-loop is 16-step atoms)"
    print(f"[vit-bench] building variant: d={args.d} "
          f"(SG_VIT_BENCH_LAYOUT={'1' if args.d != 128 else 'unset'})"
          + (f"  defines={args.defines}" if args.defines else ""), flush=True)
    t0 = time.perf_counter()
    mod = build_variant(args.d, verbose=args.verbose_build, defines=args.defines)
    print(f"[vit-bench] build {time.perf_counter()-t0:.1f}s  D={int(mod.D)} HEADS={int(mod.HEADS)} "
          f"TILE_M={int(mod.TILE_M)} TOTAL={int(mod.TOTAL):,} TILE_N={int(mod.TILE_N)}", flush=True)
    res = measure(mod, args.d, args.B, args.reps, args.warmup, args.iters)
    print(f"\n[vit-bench] d={res['d']}  B={res['B']}  params={res['total_params']:,}", flush=True)
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
