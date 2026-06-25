#!/usr/bin/env python3
"""Roofline bench for the DECODER flagship cell (d=1600/L48) across the 5 single-launch
elementwise optimizers (AdamW, Lion, Grokfast, GrokAdamW, NeuralGrok) from ONE build.

Loads the CACHED multiopt .so (mega_decoder_multiopt_tc.cu, exposes tc_train_step_opt(opt_id,...)).
Times ~50 steps with CUDA events, median step ms. Achieved TF/s = analytic GEMM FLOPs / step_s.
Arithmetic intensity = GEMM FLOPs / bytes-moved (params+acts+state, bf16 for acts/params, fp32 state).
Emits one CSV row per opt to stdout as: CSV,<model>,<opt>,<d>,<params>,<step_ms>,<tf_s>,<pct_peak>,<intensity>
"""
import os, sys, json, statistics, importlib.util
REPO = "/workspace/SuperGrok1.5"; sys.path.insert(0, REPO); os.chdir(REPO)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch

SO = "/workspace/flagship_build/mega_decoder_multiopt_adamw/mega_decoder_multiopt_adamw.so"
MODNAME = "mega_decoder_multiopt_adamw"

# OptId map (opt_components.cuh): the 5 single-launch elementwise opts in this driver.
OPTS = [("adamw", 0), ("lion", 1), ("grokfast", 2), ("grokadamw", 3), ("neuralgrok", 6)]

B = int(os.environ.get("BENCH_B", "128"))  # batch; T = B*seq rows. seq=4.
NWARM = int(os.environ.get("BENCH_WARM", "5"))
NITER = int(os.environ.get("BENCH_ITER", "50"))
NCTA_CAP = 8

def load_mod():
    spec = importlib.util.spec_from_file_location(MODNAME, SO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def decoder_gemm_flops(d, B, layers, seq, vocab):
    T = B * seq
    def gemm(M, N, K): return 2.0 * M * N * K
    dff = 4 * d
    fwd = 0.0
    for _ in range(layers):
        fwd += gemm(T, 3*d, d)   # in_proj
        fwd += gemm(T, d, d)     # out_proj
        fwd += gemm(T, dff, d)   # ff.0
        fwd += gemm(T, d, dff)   # ff.2
    fwd += gemm(B, vocab, d)     # head (last position)
    for _ in range(layers):
        fwd += gemm(T, seq, d)   # attn scores (small)
    return 3.0 * fwd            # fwd + bwd(dX+dW)

def decoder_bytes_moved(d, B, layers, seq, vocab, total_params):
    """Analytic HBM bytes/step. Weights read in fwd + read again in bwd(dX) + dW write +
    optimizer reads/writes the state. We count: params bf16 read (fwd+bwd ~2x), grad fp32 write,
    optimizer m/v fp32 read+write, params fp32 read+write (apply), plus activations bf16
    (T*d-ish per layer, read+write across fwd+bwd). bf16=2B, fp32=4B."""
    T = B * seq
    # weights: bf16, touched ~2x (fwd uses W, bwd dX uses W^T). 2 bytes each.
    w_bytes = total_params * 2 * 2
    # optimizer state: m,v fp32 read+write (8B*2 each) + params fp32 RW (8B).
    opt_bytes = total_params * (4*2*2 + 4*2)
    # grad fp32 reduce write+read.
    grad_bytes = total_params * 4 * 2
    # activations: per layer roughly T*d resid + T*3d qkv + T*dff ff hidden, bf16, fwd+bwd RW.
    dff = 4 * d
    act_per_layer = (T*d + T*3*d + T*d + T*dff + T*d) * 2  # bf16 stash
    act_bytes = act_per_layer * layers * 2  # fwd write + bwd read
    return w_bytes + opt_bytes + grad_bytes + act_bytes

def main():
    dev = "cuda"
    mod = load_mod()
    total = int(mod.TOTAL); d = int(mod.D); layers = int(mod.LAYERS)
    seq = int(mod.SEQ); vocab = int(mod.VOCAB)
    print(f"[decoder-multiopt] loaded {SO}", flush=True)
    print(f"  d={d} layers={layers} seq={seq} vocab={vocab} params={total:,}", flush=True)

    g = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
    # state >= 3*total fp32 (the multiopt driver checks numel >= 3*total).
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    tokens = torch.randint(0, vocab, (B, seq), generator=g, device=dev).to(torch.int32).contiguous()
    targets = torch.randint(0, vocab, (B,), generator=g, device=dev).to(torch.int32).contiguous()
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.98, 1e-8, 0.0

    flops = decoder_gemm_flops(d, B, layers, seq, vocab)
    bytes_moved = decoder_bytes_moved(d, B, layers, seq, vocab, total)
    intensity = flops / bytes_moved

    for name, oid in OPTS:
        # fresh state per opt (avoid NaN carryover from a different opt's math)
        state.zero_()
        bc1, bc2 = 1.0 - b1, 1.0 - b2
        def call(step):
            return mod.tc_train_step_opt(oid, params, tokens, targets, state, lr, b1, b2,
                                         eps, wd, bc1, bc2, step, 0.98, 2.0, 0.0, NCTA_CAP)
        try:
            for s in range(1, NWARM+1):
                loss, _ = call(s)
            torch.cuda.synchronize()
            l0 = float(loss.item())
            # event-timed steps
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(NITER)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(NITER)]
            for i in range(NITER):
                starts[i].record()
                call(NWARM + i + 1)
                ends[i].record()
            torch.cuda.synchronize()
            ms = sorted(starts[i].elapsed_time(ends[i]) for i in range(NITER))
            step_ms = ms[len(ms)//2]  # median
            step_s = step_ms / 1e3
            tf_s = flops / step_s / 1e12
            pct = 100.0 * tf_s / 989.0
            print(f"  [{name:11s}] step={step_ms:.3f}ms  {tf_s:.2f} TF/s  ({pct:.2f}% peak)  "
                  f"intensity={intensity:.2f} FLOP/B  loss0={l0:.3f}", flush=True)
            print(f"CSV,decoder,{name},{d},{total},{step_ms:.4f},{tf_s:.4f},{pct:.4f},{intensity:.4f},{flops:.4e},{bytes_moved:.4e}", flush=True)
        except Exception as e:
            print(f"  [{name:11s}] FAILED: {e}", flush=True)
            print(f"CSVFAIL,decoder,{name},{d},{total},{repr(e)[:120]}", flush=True)

if __name__ == "__main__":
    main()
