#ifndef GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
// ============================================================================
// supergrok15_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'supergrok15'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_supergrok15.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for SuperGrok v1.5.
// Algorithm: csrc/algorithms/supergrok15.h
//
// COMPUTE PATTERN
// Mixed: meta-MLP + per-coord alpha gate + sharpness backward + AdamW.
//   Per element:
//     mu = phi_mlp(grad, sharpness)        — 2-input × H × 1 MLP
//     alpha = clamp(alpha_base * (1 + mu), 0, alpha_max)
//     smart_grad = g + gate_signal * alpha * mu
//     AdamW(smart_grad)
//   Plus: sharpness EMA update (separate kernel).
//
// MFMA APPLICABILITY: same as NeuralGrok / SG11 — partial via rocBLAS dispatch
// for the MLP.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_supergrok15_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mu_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sharpnesses,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    float gate_global,
    float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& mu = mu_bufs[i];

        // Sweep A: meta-net forward
        auto x = torch::stack({g.to(torch::kFloat32).view({-1}),
                               sharpnesses[i].view({-1})}, /*dim=*/1);
        auto h = torch::tanh(torch::matmul(x, phi_W1.t()) + phi_b1);
        auto mu_flat = (torch::matmul(h, phi_W2.unsqueeze(1)) + phi_b2).view_as(g);
        mu.copy_(mu_flat);

        // Per-coord alpha, then smart_grad
        auto a_per_coord = torch::clamp(alpha_base * (1.0f + mu), 0.0f, alpha_max);
        auto smart = g.to(torch::kFloat32) + gate_global * a_per_coord * mu;

        prim::ema_update_inplace(m, smart, beta1);
        prim::ema_sq_update_inplace(v, smart, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness, float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2, float rescale, float lamb_eff, float beta1, float beta2, float lr, float wd_eff, float eps, float bc1, float bc2, int hidden_dim
) {
    // Delegate to the working launch_supergrok15_step, wrapping single tensors in vectors
    float b2_val = b2.item<float>();
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> vm{exp_avg};
    std::vector<torch::Tensor> vv{exp_avg_sq};
    std::vector<torch::Tensor> vmu{mu};
    std::vector<torch::Tensor> vg{grad};
    std::vector<torch::Tensor> vs{sharpness};
    float gate_global = lamb_eff;
    float alpha_base = alpha;
    float alpha_max = alpha;
    launch_supergrok15_step(vp, vm, vv, vmu, vg, vs,
                            W1, b1, W2, b2_val,
                            gate_global, alpha_base, alpha_max,
                            lr, beta1, beta2, eps, wd_eff, bc1, bc2);
}

void launch_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    // param += rho_over_norm * grad
    param.add_(grad.to(param.scalar_type()), rho_over_norm);
}

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup, torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    // param = backup, sharpness = (sam_grad - normal_grad)^2
    param.copy_(backup);
    auto diff = sam_grad.to(torch::kFloat32) - normal_grad.to(torch::kFloat32);
    sharpness.copy_(diff * diff);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
