"""Differential-bypass profiler for the Mamba TC megakernel (mamba-lane bottleneck).

ncu SpeedOfLight is blocked in this container (ERR_NVGPUCTRPERM), so we attribute
wall to phases by NULLING OUT one phase at a time (compile-time -D flags, default-0
so the shipped kernel is unchanged) and timing the delta. Quiet GPU
(CUDA_MPS_PIPE_DIRECTORY=/nonexistent), B=16384, ncta_cap=0 (one CTA/SM = shipped).

Run: CUDA_MPS_PIPE_DIRECTORY=/nonexistent PYTHONPATH=. python tuning/_mbtc_bypass_profile.py
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
from torch.utils.cpp_extension import load
from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ

B = int(os.environ.get("MBTC_B", "16384"))
B -= B % 16
TU = os.path.join(ROOT, "csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu")


def build(name, extra_defs):
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr",
             "-gencode=arch=compute_90a,code=sm_90a",
             "-gencode=arch=compute_90a,code=compute_90a", "-lineinfo"]
    flags += [f"-D{d}" for d in extra_defs]
    return load(name=name, sources=[TU], extra_include_paths=[ROOT],
                extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"],
                verbose=False)


def time_call(call, warmup=8, iters=40):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    # Variants: name -> list of -D defines. mod_<name> is a distinct JIT module.
    variants = {
        "baseline":     [],
        "no_dw_gemm":   ["SG_MBTC_BYPASS_DW_GEMM=1"],
        "no_scan":      ["SG_MBTC_BYPASS_SCAN=1"],
        "no_embedscan": ["SG_MBTC_BYPASS_EMBED_SCAN=1"],
    }
    print(f"[mbtc-bypass] B={B}  T={B*SEQ}  quiet={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}")
    results = {}
    base_ms = None
    base_sc_ms = None
    for name, defs in variants.items():
        try:
            mod = build(f"mbtc_bp_{name}", defs)
        except Exception as e:
            print(f"  {name:14s} BUILD FAILED: {str(e)[:200]}")
            continue
        total = int(mod.TOTAL)
        gpar = torch.Generator(device=dev).manual_seed(42)
        params = (torch.randn(total, generator=gpar, device=dev) * 0.02).contiguous()
        gin = torch.Generator(device=dev).manual_seed(7)
        inp = torch.randint(0, VOCAB, (B, SEQ), generator=gin, device=dev).to(torch.int32)
        tgt = torch.randint(0, P_HEAD, (B,), generator=gin, device=dev).to(torch.int32)
        state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
        beta1, beta2 = 0.9, 0.98
        bc1, bc2 = 1.0 - beta1, 1.0 - beta2

        def tc_call():
            return mod.tc_train_step(params.clone(), inp, tgt, state, 1e-3, beta1, beta2,
                                     1e-8, 0.0, bc1, bc2, 1, 0)
        ms = time_call(tc_call)
        results[name] = {"tc_ms": ms}
        if name == "baseline":
            base_ms = ms
            # scalar baseline from the SAME module (back-to-back, same contention)
            def sc_call():
                return mod.scalar_train_step(params.clone(), inp, tgt, state, 1e-3, beta1,
                                             beta2, 1e-8, 0.0, bc1, bc2, 1)
            base_sc_ms = time_call(sc_call)
            results["scalar_baseline"] = {"tc_ms": base_sc_ms}
            print(f"  {'baseline (TC)':18s} {ms:8.3f} ms")
            print(f"  {'baseline (scalar)':18s} {base_sc_ms:8.3f} ms   "
                  f"(scalar/TC = {base_sc_ms/ms:.3f}x; TC must beat 1.0 to win)")
        else:
            delta = base_ms - ms
            pct = 100.0 * delta / base_ms
            print(f"  {name:18s} {ms:8.3f} ms   phase≈{delta:7.3f} ms ({pct:5.1f}% of TC wall)")
    out = os.path.join(ROOT, "results/h100_grokking_race/mbtc_bypass_profile.json")
    json.dump({"B": B, "T": B*SEQ, "baseline_tc_ms": base_ms,
               "baseline_scalar_ms": base_sc_ms, "variants": results}, open(out, "w"), indent=2)
    print(f"[mbtc-bypass] wrote {out}")


if __name__ == "__main__":
    main()
