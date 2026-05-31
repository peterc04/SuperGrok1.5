#ifndef GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
// ============================================================================
// adamw_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'adamw'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_adamw.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for AdamW (simple, used by Muon 1D params + LookSAM).
// Algorithm: csrc/algorithms/adamw.h
//
// COMPUTE PATTERN
// Pure elementwise. Per element:
//   m = beta1 * m + (1-beta1) * g         — 1 FMA, 2 reads, 1 write
//   v = beta2 * v + (1-beta2) * g²        — 1 FMA + 1 mul, 2 reads, 1 write
//   p -= lr * (m / bc1 / (sqrt(v/bc2) + eps) + wd * p)  — div, sqrt, FMA
// Bandwidth-bound (≈ 12 mem ops per element including p, m, v, g).
//
// MFMA APPLICABILITY: none.
// AdamW is pure elementwise SIMD. No matrix multiplies. CDNA3 v_mfma_*
// instructions would be unused.
//
// WHY ATEN HERE
// Same constraint as launch_lion: `.hip.cpp` → host compiler. The ATen
// path uses `mul_().addcmul_().sqrt_()` which dispatches to rocPRIM
// elementwise kernels. Bandwidth is the bound either way.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_adamw_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    auto pack = prim::pack_valid(params, grads, exp_avgs, exp_avg_sqs);
    for (size_t i = 0; i < pack.params.size(); i++) {
        auto& p = pack.params[i];
        auto& g = pack.grads[i];
        auto& m = pack.state_a[i];
        auto& v = pack.state_b[i];

        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
