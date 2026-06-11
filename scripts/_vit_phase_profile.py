"""Build the vit TC TU with -DSG_VIT_PROFILE and read the per-phase clock64
breakdown (P1 fwd, P1 bwd, P2 dW+reduce, P3 opt tail) at B=16384. Run quiet.

Picks the dominant phase → tells us which structural lever to pull next.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    dev = torch.device("cuda")
    tu = str(ROOT / "csrc/fused/sm_90/mega_vit_real_adamw_tc.cu")
    mod = load(
        name="mega_vit_real_adamw_tc_PROF",
        sources=[tu],
        extra_include_paths=[str(ROOT)],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                           "-gencode=arch=compute_90a,code=sm_90a",
                           "-DSG_VIT_PROFILE=1"],
        extra_cflags=["-O3", "-std=c++17", "-DSG_VIT_PROFILE=1"],
        verbose=False,
    )
    assert getattr(mod, "HAS_PROFILE", False), "profile build missing"
    total = int(mod.TOTAL)
    from tests.hw.vit_oracle import NUM_PATCHES, PATCH_DIM, VOCAB
    gpar = torch.Generator(device=dev).manual_seed(42)
    params = (torch.randn(total, generator=gpar, device=dev) * 0.02).contiguous()
    gin = torch.Generator(device=dev).manual_seed(7)
    inp = torch.randn(B, NUM_PATCHES, PATCH_DIM, generator=gin, device=dev).to(torch.float32)
    tgt = torch.randint(0, VOCAB, (B,), generator=gin, device=dev).to(torch.int32)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    beta1, beta2 = 0.9, 0.98
    bc1, bc2 = 1.0 - beta1, 1.0 - beta2
    lr, eps, wd = 1e-3, 1e-8, 0.0

    def call():
        return mod.tc_train_step(params.clone(), inp, tgt, state, lr, beta1, beta2,
                                 eps, wd, bc1, bc2, 1, 0)
    # warmup
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    mod.tc_profile_read()  # reset
    # measured: a few steps, then read the max-across-CTAs per phase
    import time
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        call()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / N * 1e3
    cyc = mod.tc_profile_read()  # [P1fwd, P1bwd, P2dW, P3opt] cycles (max over CTAs, over N steps? no—reset each? )
    # NOTE: g_vit_prof_max accumulates the MAX across all launches since reset (not
    # summed). Since every step has ~identical work, the max == one step's per-phase
    # critical-path cycles. Convert with the SM clock (~1.98 GHz boost on H100).
    sm_ghz = 1.98
    names = ["P1_fwd", "P1_bwd", "B1_barrier_wait", "P2_dW_GEMM",
             "P2_grad_assembly", "P3_opt_tail"]
    tot = sum(cyc) or 1
    print(f"[vit-phase] B={B} wall/step={wall_ms:.2f}ms  (clock64 critical-path per phase, 1 step)", flush=True)
    for n, c in zip(names, cyc):
        ms = c / (sm_ghz * 1e9) * 1e3
        print(f"   {n:14s}: {c:>14d} cyc  ~{ms:8.3f} ms  {100.0*c/tot:5.1f}% of summed-phase cyc", flush=True)
    print(f"   {'SUM':14s}: {tot:>14d} cyc  ~{tot/(sm_ghz*1e9)*1e3:8.3f} ms", flush=True)
    print(f"[vit-phase] DOMINANT = {names[cyc.index(max(cyc))]} ({100.0*max(cyc)/tot:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
