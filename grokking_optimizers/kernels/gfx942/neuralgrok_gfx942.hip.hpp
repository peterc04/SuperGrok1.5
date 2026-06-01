#ifndef GROKKING_KERNELS_GFX942_NEURALGROK_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_NEURALGROK_GFX942_HIP_HPP_
// ============================================================================
// neuralgrok_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'neuralgrok'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_neuralgrok.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for NeuralGrok.
// Algorithm: csrc/algorithms/neuralgrok.h
//
// COMPUTE PATTERN
// Mixed: per-element psi-net MLP + AdamW.
//   Per element:
//     h = W1 * |g| + b1     — 1×1 × 1×H → 1×H GEMM (per element)
//     h = relu(h)
//     s = W2 * h + b2       — 1×H × H×1 → 1×1 GEMM (per element)
//     g_amp = (s * alpha + beta) * g
//     AdamW(g_amp)
//
// MFMA APPLICABILITY: partial.
// The per-element MLP is structurally GEMM-shaped but the contraction
// dimension is too small (H typically 16-32) for MFMA's 16×16×16 tile to
// give a clean win on the FIRST layer (input is 1-D scalar). The SECOND
// layer (N × H × 1) is a true matrix-vector op: if we batch across N,
// MFMA can run at full pipe.
//
// WHY ATEN HERE
// ATen + rocBLAS handles the batched layer-2 GEMM via MFMA already. The
// layer-1 (input is per-element scalar) doesn't benefit from MFMA; ATen
// emits a broadcast elementwise kernel. Hand-written fusion would save
// 1 kernel launch (≈ 3 µs).

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_neuralgrok_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    const torch::Tensor& psi_W1,
    const torch::Tensor& psi_b1,
    const torch::Tensor& psi_W2,
    float psi_b2,
    float alpha, float beta,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        // psi forward: input is |grad| flattened
        auto ag = g.abs().to(torch::kFloat32).view({-1, 1});
        auto h = torch::relu(torch::matmul(ag, psi_W1.unsqueeze(0)) + psi_b1);
        auto s = (torch::matmul(h, psi_W2.unsqueeze(1)) + psi_b2).view_as(g);

        auto g_amp = (s * alpha + beta) * g.to(torch::kFloat32);
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_fused_neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified, torch::Tensor amplifier_w1, torch::Tensor amplifier_b1, torch::Tensor amplifier_w2, torch::Tensor amplifier_b2, int hidden_dim, float alpha, float beta
) {
    // psi forward: input is |grad| per-element
    auto ag = grad.abs().to(torch::kFloat32).view({-1, 1});
    auto h = torch::relu(torch::matmul(ag, amplifier_w1.unsqueeze(0)) + amplifier_b1);
    float b2_val = amplifier_b2.item<float>();
    auto s = (torch::matmul(h, amplifier_w2.unsqueeze(1)) + b2_val).view_as(grad);
    // amplified = alpha * psi * grad + beta * grad
    amplified.copy_((s * alpha + beta) * grad.to(torch::kFloat32));
}

void launch_fused_neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor amplified_grad, float beta1, float beta2, float lr, float weight_decay, float eps, float bc1, float bc2
) {
    prim::ema_update_inplace(exp_avg, amplified_grad, beta1);
    prim::ema_sq_update_inplace(exp_avg_sq, amplified_grad, beta2);
    prim::adam_apply_inplace(param, exp_avg, exp_avg_sq, lr, bc1, bc2, eps, weight_decay);
}

void launch_fused_neuralgrok_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor grad, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2, float alpha_amp, float beta_amp, int hidden_dim, float beta1, float beta2, float lr, float weight_decay, float eps, float bc1, float bc2
) {
    float b2_val = b2.item<float>();
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> vm{exp_avg};
    std::vector<torch::Tensor> vv{exp_avg_sq};
    std::vector<torch::Tensor> vg{grad};
    launch_neuralgrok_step(vp, vm, vv, vg,
                           W1, b1, W2, b2_val,
                           alpha_amp, beta_amp,
                           lr, beta1, beta2, eps, weight_decay,
                           bc1, bc2);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_NEURALGROK_GFX942_HIP_HPP_
