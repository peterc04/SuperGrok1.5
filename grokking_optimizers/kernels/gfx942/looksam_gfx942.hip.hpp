#ifndef GROKKING_KERNELS_GFX942_LOOKSAM_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_LOOKSAM_GFX942_HIP_HPP_
// ============================================================================
// looksam_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'looksam'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_looksam.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for LookSAM.
// Algorithm: csrc/algorithms/looksam.h
//
// COMPUTE PATTERN
// Four separate ops (matches the algorithm-header structure):
//   1. perturb:     p_pert = p + rho * (g / ||g||)              — elementwise + 1 reduction
//   2. restore:     p = p_pert - rho * (g / ||g||)              — elementwise
//   3. set_direction: sam_dir = g_sam - g                       — elementwise
//   4. apply:       g_adj = (1-alpha)*g + alpha*sam_dir; AdamW(g_adj)
// The ||g|| computation in steps 1 + 2 is a global reduction.
//
// MFMA APPLICABILITY: none. Elementwise + scalar reduction.
//
// WHY ATEN HERE
// Each op is a chain of broadcasted tensor expressions. ATen + rocPRIM
// handles the elementwise + reduction patterns natively. Hand-written
// fusion would chain perturb + reduce + apply into 1 kernel instead of 3;
// gain is ≈ 1.7× and needs hardware verification.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_looksam_perturb(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& grads,
    float scale
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        backups[i].copy_(params[i]);
        params[i].add_(grads[i].to(params[i].scalar_type()), scale);
    }
}

void launch_looksam_restore(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++) {
        params[i].copy_(backups[i]);
    }
}

void launch_looksam_set_direction(
    std::vector<torch::Tensor>& sam_dirs,
    std::vector<torch::Tensor>& grads_sam,
    std::vector<torch::Tensor>& grads_orig
) {
    for (size_t i = 0; i < sam_dirs.size(); i++) {
        sam_dirs[i].copy_(grads_sam[i] - grads_orig[i]);
    }
}

void launch_looksam_apply(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& sam_dirs,
    std::vector<torch::Tensor>& grads,
    float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        auto g_adj = (1.0f - alpha) * g.to(torch::kFloat32) + alpha * sam_dirs[i];
        prim::ema_update_inplace(m, g_adj, beta1);
        prim::ema_sq_update_inplace(v, g_adj, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir, float inv_norm, float lambda, float grad_norm
) {
    // grad = (1 - lambda) * grad + lambda * grad_norm * (v_dir * inv_norm)
    auto scaled_dir = v_dir * (inv_norm * grad_norm);
    grad.mul_(1.0f - lambda).add_(scaled_dir, lambda);
}

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results /* [diff_norm, grad_norm] */
) {
    // results[0] = ||sam_grad - grad||, results[1] = ||grad||
    auto diff = sam_grad.to(torch::kFloat32) - grad.to(torch::kFloat32);
    results[0] = diff.norm();
    results[1] = grad.to(torch::kFloat32).norm();
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_LOOKSAM_GFX942_HIP_HPP_
