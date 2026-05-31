#ifndef GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
// ============================================================================
// grokfast_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'grokfast'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_grokfast.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Grokfast.
// Algorithm: csrc/algorithms/grokfast.h
//
// COMPUTE PATTERN
// Identical to GrokAdamW (EMA amplification + AdamW). Hyperparameters differ;
// math is structurally the same.
//
// MFMA APPLICABILITY: none. Elementwise.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_grokfast_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& ema = emas[i];

        prim::ema_update_inplace(ema, g, gf_alpha);
        auto g_amp = g.to(torch::kFloat32) + gf_lamb * ema;
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema, float alpha, float lamb
) {
    prim::ema_update_inplace(ema, grad, alpha);
    grad.add_(ema, lamb);
}

void launch_fused_grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps, float bc1, float bc2
) {
    std::vector<torch::Tensor> vp{param}, vea{exp_avg}, veas{exp_avg_sq},
                               vema{ema}, vg{grad};
    launch_grokfast_step(vp, vea, veas, vema, vg,
                         alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                         bc1, bc2);
}

void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads, std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
    }
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
