#!/usr/bin/env python3
"""Build + roofline bench for ViT (d=1664/L48) OR Mamba (d=2048/L24) flagship cells
across the 5 single-launch elementwise optimizers, from ONE JIT build of a multiopt TU.

Usage: python bench_vit_mamba_multiopt.py {vit|mamba}

Mirrors the decoder roofline method: build with -include <model>_flagship_layout.cuh +
-DSG_<M>_SCALAR_MEGAKERNEL=0 + -DSG_TUNED_GEMM_IMPL=1 (the flagship_train.py recipe),
then time ~50 CUDA-event steps per opt, median step ms -> achieved TF/s (analytic GEMM
FLOPs / step_s) + arithmetic intensity (GEMM FLOPs / bytes-moved).
"""
import os, sys, statistics, time
REPO = "/workspace/SuperGrok1.5"; sys.path.insert(0, REPO); os.chdir(REPO)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch
from torch.utils.cpp_extension import load

SCRATCH = "/tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad"
CUTLASS = [REPO + "/third_party/cutlass/include", REPO + "/third_party/cutlass/tools/util/include"]
OPTS = [("adamw", 0), ("lion", 1), ("grokfast", 2), ("grokadamw", 3), ("neuralgrok", 6)]
NWARM = int(os.environ.get("BENCH_WARM", "5"))
NITER = int(os.environ.get("BENCH_ITER", "30"))
NCTA_CAP = int(os.environ.get("BENCH_NCTA", "8"))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "vit"
assert MODEL in ("vit", "mamba")

if MODEL == "vit":
    TU = SCRATCH + "/mega_vit_multiopt_tc.cu"
    INC = "csrc/fused/sm_90/vit_flagship_layout.cuh"
    # pre-define the DEFAULT layout's include guard so vit_layout.cuh is a no-op;
    # the flagship header (its own distinct guard) provides the dims via -include.
    GUARD = "-DSG_FUSED_SM90_VIT_LAYOUT_CUH_=1"
    SCALAR_OFF = "-DSG_VIT_SCALAR_MEGAKERNEL=0"
    # BENCH_LAYOUT=1 sets kVitStagedOptScratch=false -> elide the Prodigy/Muon/LookSAM/SG2
    # per-CTA staged-opt scratch (SG2's is O(Nmax*d) per CTA -> 82GB OOM at flagship d).
    # The 5 elementwise opts never touch it, so eliding is honest scoping, not suppression.
    BENCH_LAYOUT = "-DSG_VIT_BENCH_LAYOUT=1"
    NAME = "mega_vit_multiopt"
    BD = "/workspace/flagship_build/mega_vit_multiopt"
else:
    TU = SCRATCH + "/mega_mamba_multiopt_tc.cu"
    INC = "csrc/fused/sm_90/mamba_flagship_layout.cuh"
    GUARD = "-DSG_FUSED_SM90_MAMBA3_LAYOUT_CUH_=1"
    SCALAR_OFF = "-DSG_MB_SCALAR_MEGAKERNEL=0"
    BENCH_LAYOUT = "-DSG_MB_BENCH_LAYOUT=1"
    NAME = "mega_mamba_multiopt"
    BD = "/workspace/flagship_build/mega_mamba_multiopt"

def build():
    os.makedirs(BD, exist_ok=True)
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-gencode=arch=compute_90a,code=sm_90a", "-gencode=arch=compute_90a,code=compute_90a",
             "-lineinfo", "-DWITH_CUTLASS", "-DSG_TUNED_GEMM_IMPL=1", SCALAR_OFF, BENCH_LAYOUT,
             GUARD, "-include", INC]
    t0 = time.time()
    mod = load(name=NAME, sources=[TU], extra_include_paths=[REPO] + CUTLASS,
               extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"],
               build_directory=BD, verbose=True)
    print(f"[{MODEL}-multiopt] build done in {time.time()-t0:.1f}s", flush=True)
    return mod

# ---- analytic GEMM FLOPs + bytes-moved ----
def vit_gemm_flops(d, B, layers, seq, vocab):
    T = B * seq
    def gemm(M, N, K): return 2.0 * M * N * K
    dff = 4 * d
    fwd = 0.0
    # patch embed: B*npatch rows of patch->d (small); count layers' main GEMMs.
    for _ in range(layers):
        fwd += gemm(T, 3*d, d)   # qkv
        fwd += gemm(T, d, d)     # attn out
        fwd += gemm(T, dff, d)   # ff.0
        fwd += gemm(T, d, dff)   # ff.2
    fwd += gemm(B, vocab, d)     # head (cls token)
    return 3.0 * fwd

def mamba_gemm_flops(d, B, layers, seq, dinner, xproj, dff, vocab, phead):
    T = B * seq
    def gemm(M, N, K): return 2.0 * M * N * K
    fwd = 0.0
    for _ in range(layers):
        fwd += gemm(T, 2*dinner, d)   # in_proj -> x + z (d_inner each)
        fwd += gemm(T, xproj, dinner) # x_proj (dt/B/C)
        fwd += gemm(T, d, dinner)     # out_proj
        fwd += gemm(T, dff, d)        # SwiGLU up (gate+value ~ treat as dff)
        fwd += gemm(T, d, dff)        # SwiGLU down
    fwd += gemm(B, phead, d)          # head
    return 3.0 * fwd

def bytes_moved(d, B, layers, seq, total_params, act_floats_per_layer):
    w_bytes = total_params * 2 * 2          # weights bf16, ~2x (fwd+bwd)
    opt_bytes = total_params * (4*2*2 + 4*2) # m,v fp32 RW + params fp32 RW
    grad_bytes = total_params * 4 * 2        # grad fp32 reduce RW
    act_bytes = act_floats_per_layer * layers * 2 * 2  # bf16, fwd write + bwd read
    return w_bytes + opt_bytes + grad_bytes + act_bytes

def main():
    dev = "cuda"
    mod = build()
    total = int(mod.TOTAL); d = int(mod.D); layers = int(mod.LAYERS)
    seq = int(mod.SEQ); vocab = int(mod.VOCAB)
    print(f"[{MODEL}-multiopt] d={d} layers={layers} seq={seq} vocab={vocab} params={total:,}", flush=True)

    B = int(os.environ.get("BENCH_B", "256"))
    g = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    targets = torch.randint(0, vocab, (B,), generator=g, device=dev).to(torch.int32).contiguous()

    if MODEL == "vit":
        npatch, patch = int(mod.NPATCH), int(mod.PATCH)
        inp = torch.randn(B, npatch, patch, generator=g, device=dev).contiguous()
        flops = vit_gemm_flops(d, B, layers, seq, vocab)
        T = B * seq; dff = 4*d
        act_pl = (T*d + T*3*d + T*d + T*dff + T*d)  # resid+qkv+attn+ff_hidden+ff_out
    else:
        inp = torch.randint(0, vocab, (B, seq), generator=g, device=dev).to(torch.int32).contiguous()
        dinner = int(mod.DINNER)
        # mamba flagship dims from layout
        xproj = 576; dff = 4096; phead = int(mod.PHEAD)
        flops = mamba_gemm_flops(d, B, layers, seq, dinner, xproj, dff, vocab, phead)
        T = B * seq
        act_pl = (T*d + T*2*dinner + T*dinner + T*dff + T*d)

    bmoved = bytes_moved(d, B, layers, seq, total, act_pl)
    intensity = flops / bmoved
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0

    for name, oid in OPTS:
        state.zero_()
        bc1, bc2 = 1.0 - b1, 1.0 - b2
        def call(step):
            return mod.tc_train_step_opt(oid, params, inp, targets, state, lr, b1, b2,
                                         eps, wd, bc1, bc2, step, 0.98, 2.0, 0.0, NCTA_CAP)
        try:
            for s in range(1, NWARM+1):
                loss, _ = call(s)
            torch.cuda.synchronize()
            l0 = float(loss.item())
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(NITER)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(NITER)]
            for i in range(NITER):
                starts[i].record(); call(NWARM + i + 1); ends[i].record()
            torch.cuda.synchronize()
            ms = sorted(starts[i].elapsed_time(ends[i]) for i in range(NITER))
            step_ms = ms[len(ms)//2]; step_s = step_ms / 1e3
            tf_s = flops / step_s / 1e12; pct = 100.0 * tf_s / 989.0
            print(f"  [{name:11s}] step={step_ms:.3f}ms  {tf_s:.2f} TF/s  ({pct:.2f}% peak)  "
                  f"intensity={intensity:.2f} FLOP/B  loss0={l0:.3f}", flush=True)
            print(f"CSV,{MODEL},{name},{d},{total},{step_ms:.4f},{tf_s:.4f},{pct:.4f},{intensity:.4f},{flops:.4e},{bmoved:.4e}", flush=True)
        except Exception as e:
            print(f"  [{name:11s}] FAILED: {e}", flush=True)
            print(f"CSVFAIL,{MODEL},{name},{d},{total},{repr(e)[:140]}", flush=True)

if __name__ == "__main__":
    main()
