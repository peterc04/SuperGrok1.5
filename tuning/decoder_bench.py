"""tuning/decoder_bench.py — HILL-CLIMB d-scaled decoder benchmark + phase profiler.

The persistent megakernel's compute roofline (task #13) is diagnosed/optimized at a
LARGE model width. This is the FIRST-CLASS coexisting bench-variant build: it
JIT-compiles the SAME decoder TC cell TU (csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu)
as a SEPARATELY-NAMED extension (default name "mega_decoder_real_adamw_tc_bench"), with
its OWN torch-extensions build dir + ninja incremental rebuild + sccache, so each
hill-climb iteration rebuilds incrementally and NEVER touches the production _ops.so
or the 33/33 gate.

  * -DSG_DEC_BENCH_LAYOUT=1  -> the d=1024 layout branch of decoder_layout.cuh
                               (the proven end-to-end width; A/A/A, parity 2.4e-07).
  * -DSG_DEC_PROFILE=1       -> the in-kernel clock64 per-phase timers (default OFF;
                               NEVER on the production path).

USAGE
  # Deliverable B: per-phase wall breakdown of one d=1024 adamw/decoder step.
  python tuning/decoder_bench.py --profile --d 1024 --B 16384 --reps 3
  python tuning/decoder_bench.py --profile --d 1024 --B 2048  --reps 3   # small-B
  # Throughput-only (TF/s, steps/s) without instrumentation perturbation:
  python tuning/decoder_bench.py --d 1024 --B 16384 --reps 3
  # Sanity at production width (must match the 33/33 cell):
  python tuning/decoder_bench.py --profile --d 128 --B 16384

Notes:
  * d is selected at COMPILE time (the layout is a compile-time constant); --d 1024
    sets -DSG_DEC_BENCH_LAYOUT=1, --d 128 leaves it unset (production branch).
  * Inputs are RANDOM (throughput/phase split, not a parity gate — correctness was
    proven separately). params/state are sized from the TU's exposed mod.TOTAL.
  * The clock64 stamps measure the SLOWEST CTA per phase (atomicMax), which is the
    critical path the host wall sees (the trailing grid barrier waits for it).
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

SM_GHZ = 1.98  # H100 SM boost clock (nvidia-smi clocks.max.sm = 1980 MHz)
_TC_TU = str(ROOT / "csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu")


def build_variant(d: int, profile: bool, name: str | None = None, verbose: bool = False,
                  defines: list[str] | None = None):
    """JIT-build the decoder TC cell TU as a coexisting variant extension. Distinct
    module name + own build dir => incremental ninja rebuild, sccache-friendly, and
    NO collision with the production _ops. Returns the loaded module.

    `defines` is a list of extra "-DKEY=VAL" override flags (e.g. the hill-climb
    knob sweep: -DSG_TUNED_DEC_DW_SPLITK=8). Each distinct define-set gets its own
    module name (suffix from the KEY=VAL pairs) so variants coexist + cache cleanly."""
    bench = (d != 128)
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
             "-gencode=arch=compute_90a,code=sm_90a",
             "-gencode=arch=compute_90a,code=compute_90a",
             "-DSG_TUNED_GEMM_IMPL=1"]  # select the wgmma cell driver
    if bench:
        flags.append("-DSG_DEC_BENCH_LAYOUT=1")
        # The legacy fp32 SCALAR decoder megakernel's DecSampleSmem grows with
        # SG_DEC_D and ptxas HARD-STOPS ("uses too much shared data", 0x507d8 >
        # 0x29000) at d>=768. It is NEVER used by the bench/profile driver (we drive
        # the TC engine only), so gate it OFF — exactly the decouple the
        # SG_DEC_SCALAR_MEGAKERNEL flag (HEAD 555c0bc) exists for. This also compiles
        # out the cell TU's scalar_train_step mirror (#if SG_DEC_SCALAR_MEGAKERNEL),
        # which we don't need here. Production (d=128) keeps the scalar path (default 1).
        flags.append("-DSG_DEC_SCALAR_MEGAKERNEL=0")
    if profile:
        flags.append("-DSG_DEC_PROFILE=1")
    # Hill-climb knob overrides (-DKEY=VAL). Appended LAST so they win over defaults.
    suffix = ""
    if defines:
        for d_ in defines:
            flags.append("-D" + d_ if not d_.startswith("-D") else d_)
        # build a filesystem-safe module suffix from the override KEY=VAL pairs.
        norm = "_".join(d_.lstrip("-D").replace("=", "").replace(" ", "")
                        for d_ in defines)
        suffix = "_" + norm
    if name is None:
        name = "mega_decoder_real_adamw_tc"
        name += "_bench" if bench else "_prod"
        if profile:
            name += "_prof"
        name += suffix
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    mod = load(name=name, sources=[_TC_TU],
               extra_include_paths=[str(ROOT)],
               extra_cuda_cflags=flags,
               extra_cflags=["-O3", "-std=c++17",
                             *(["-DSG_DEC_PROFILE=1"] if profile else [])],
               verbose=verbose)
    return mod


def _decoder_gemm_flops_per_step(d: int, B: int, layers: int, seq: int, vocab: int) -> float:
    """Analytic GEMM FLOPs for one fwd+bwd decoder step (the dense matmul work the
    roofline counts; LN/softmax/elementwise register ~0 in the profiler convention).
    T = B*seq rows. Per layer the GEMMs are: QKV proj (d->3d), attn-out (d->d),
    ff0 (d->dff=4d), ff2 (dff->d). Plus the output head (d->vocab). Each GEMM is
    2*M*N*K MACs*2. Backward ≈ 2x forward (dX + dW), so fwd+bwd ≈ 3x forward."""
    T = B * seq
    def gemm(M, N, K):
        return 2.0 * M * N * K
    dff = 4 * d
    fwd = 0.0
    for _ in range(layers):
        fwd += gemm(T, 3 * d, d)      # in_proj
        fwd += gemm(T, d, d)          # out_proj
        fwd += gemm(T, dff, d)        # ff.0
        fwd += gemm(T, d, dff)        # ff.2
    fwd += gemm(B, vocab, d)          # output head (last position only: M=B)
    # attention score/context GEMMs (small: seq=4) — include for completeness.
    for _ in range(layers):
        fwd += gemm(T, seq, d // 1)   # QK^T-ish (seq small; negligible)
    return 3.0 * fwd                  # fwd + bwd(dX+dW) ≈ 3x


def measure(mod, d: int, B: int, reps: int, warmup: int, iters: int,
            profile: bool, ncta_cap: int = 0):
    dev = torch.device("cuda")
    total = int(mod.TOTAL)
    layers, seq, vocab = int(mod.LAYERS), int(mod.SEQ), int(mod.VOCAB)
    assert int(mod.D) == d, f"TU compiled at D={int(mod.D)}, expected {d}"
    g = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    tokens = torch.randint(0, vocab, (B, seq), generator=g, device=dev).to(torch.int32).contiguous()
    targets = torch.randint(0, vocab, (B,), generator=g, device=dev).to(torch.int32).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2

    def call():
        return mod.tc_train_step(params, tokens, targets, state, lr, beta1, beta2,
                                 eps, wd, bc1, bc2, 1, ncta_cap)

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

    flops = _decoder_gemm_flops_per_step(d, B, layers, seq, vocab)
    res = {
        "d": d, "B": B, "total_params": total, "wall_s": wall,
        "wall_ms": wall * 1e3, "steps_per_s": 1.0 / wall,
        "gemm_flops_per_step": flops, "achieved_tf_s": flops / wall / 1e12,
        "walls_ms": [w * 1e3 for w in walls],
    }

    if profile and getattr(mod, "HAS_PROFILE", False):
        # reset, run a few steps, read the max-across-CTAs per phase (every step has
        # ~identical work, so the accumulated max == one step's per-phase critical path).
        mod.tc_profile_read()  # reset
        prof_reps = []
        for _ in range(reps):
            mod.tc_profile_read()  # reset before this rep
            for _ in range(iters):
                call()
            torch.cuda.synchronize()
            prof_reps.append(mod.tc_profile_read())
        # median per slot across reps
        cyc = [int(statistics.median(s)) for s in zip(*prof_reps)]
        res["phase_cycles"] = cyc
    return res


PHASE_NAMES = ["P1_fwd", "P1_bwd", "B1_barrier", "P2_dW_GEMM",
               "P2_grad_asm", "P3_opt_tail", "B2_barrier", "B0_barrier"]


def _print_report(res):
    d, B = res["d"], res["B"]
    print(f"\n[decoder-bench] d={d}  B={B}  params={res['total_params']:,}", flush=True)
    print(f"  wall/step = {res['wall_ms']:.3f} ms  (median of reps {[f'{x:.2f}' for x in res['walls_ms']]})",
          flush=True)
    print(f"  steps/s   = {res['steps_per_s']:.4f}", flush=True)
    print(f"  GEMM FLOPs/step = {res['gemm_flops_per_step']:.3e}  "
          f"=> achieved {res['achieved_tf_s']:.3f} TF/s "
          f"({100.0*res['achieved_tf_s']/989.0:.2f}% of 989 TF/s bf16 dense roofline)",
          flush=True)
    if "phase_cycles" in res:
        cyc = res["phase_cycles"]
        tot = sum(cyc) or 1
        # clock64 stamps measure the per-phase critical path on the SLOWEST CTA.
        summed_ms = tot / (SM_GHZ * 1e9) * 1e3
        print(f"\n  [phase breakdown — clock64 critical-path, median of reps]  "
              f"(summed-phase ~{summed_ms:.3f} ms vs wall {res['wall_ms']:.3f} ms; "
              f"the wall-minus-summed gap is host/launch + un-stamped slack)", flush=True)
        print(f"    {'phase':14s} {'cycles':>14s} {'ms':>10s}  {'% of summed':>12s}", flush=True)
        for n, c in zip(PHASE_NAMES, cyc):
            ms = c / (SM_GHZ * 1e9) * 1e3
            print(f"    {n:14s} {c:>14,d} {ms:>10.3f}  {100.0*c/tot:>11.1f}%", flush=True)
        print(f"    {'SUM':14s} {tot:>14,d} {summed_ms:>10.3f}  {'100.0':>11s}%", flush=True)
        dom = PHASE_NAMES[cyc.index(max(cyc))]
        print(f"  DOMINANT phase = {dom} ({100.0*max(cyc)/tot:.1f}% of summed-phase cycles)",
              flush=True)
        # barrier roll-up
        bar_cyc = cyc[2] + cyc[6] + cyc[7]
        print(f"  grid-barrier wait total (B0+B1+B2) = {bar_cyc:,} cyc "
              f"~{bar_cyc/(SM_GHZ*1e9)*1e3:.3f} ms ({100.0*bar_cyc/tot:.1f}% of summed)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024, help="decoder width (128 prod | 1024 bench)")
    ap.add_argument("--B", type=int, default=16384, help="batch size (must be %% 16 == 0)")
    ap.add_argument("--reps", type=int, default=3, help="timing repeats (report median)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10, help="steps per rep")
    ap.add_argument("--profile", action="store_true", help="build with -DSG_DEC_PROFILE and read per-phase clock64")
    ap.add_argument("--ncta-cap", type=int, default=0, help="0 = one CTA/SM (full saturation)")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("-D", dest="defines", action="append", default=[],
                    metavar="KEY=VAL",
                    help="extra -D compile override (repeatable), e.g. "
                         "-D SG_TUNED_DEC_DW_SPLITK=8. Each set builds its own variant.")
    args = ap.parse_args()
    assert args.B % 16 == 0, "B must be divisible by 16 (dW K-loop is 16-step atoms)"

    print(f"[decoder-bench] building variant: d={args.d} profile={args.profile} "
          f"(SG_DEC_BENCH_LAYOUT={'1' if args.d != 128 else 'unset'})"
          + (f"  defines={args.defines}" if args.defines else ""), flush=True)
    t0 = time.perf_counter()
    mod = build_variant(args.d, args.profile, verbose=args.verbose_build,
                        defines=args.defines)
    print(f"[decoder-bench] build done in {time.perf_counter()-t0:.1f}s  "
          f"(D={int(mod.D)} TOTAL={int(mod.TOTAL):,} TILE_N={int(mod.TILE_N)} "
          f"HAS_PROFILE={getattr(mod,'HAS_PROFILE',False)})", flush=True)

    res = measure(mod, args.d, args.B, args.reps, args.warmup, args.iters,
                  args.profile, args.ncta_cap)
    _print_report(res)
    return res


if __name__ == "__main__":
    main()
