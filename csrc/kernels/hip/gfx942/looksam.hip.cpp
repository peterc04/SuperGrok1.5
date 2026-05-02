// =====================================================================
//  csrc/kernels/hip/gfx942/looksam.hip.cpp — ATen-based.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/looksam.hip.h"

namespace sg { namespace gfx942 {

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor backup, torch::Tensor grad,
    float rho_over_norm
) {
    if (param.numel() == 0) return;
    backup.copy_(param);
    // param += rho_over_norm * grad
    param.add_(grad, rho_over_norm);
}

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup
) {
    if (param.numel() == 0) return;
    param.copy_(backup);
}

void launch_looksam_direction_adjust_fused(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor v_dir,
    float inv_norm, float lambda, float grad_norm
) {
    (void)sam_grad;
    if (grad.numel() == 0) return;
    // grad += lambda * grad_norm * (v_dir * inv_norm)
    auto v_scaled = v_dir.to(torch::kFloat32) * inv_norm;
    auto add = (lambda * grad_norm) * v_scaled;
    grad.add_(add.to(grad.scalar_type()));
}

void launch_looksam_norm_reduce(
    torch::Tensor grad, torch::Tensor sam_grad, torch::Tensor results
) {
    TORCH_CHECK(results.numel() == 2 && results.scalar_type() == at::kFloat,
                "looksam_norm_reduce (gfx942): results must be FP32 [2]");
    auto diff  = (sam_grad - grad).to(torch::kFloat32).reshape(-1);
    auto gflat = grad.to(torch::kFloat32).reshape(-1);
    auto diff_norm = diff.dot(diff).sqrt();
    auto grad_norm = gflat.dot(gflat).sqrt();
    results.narrow(0, 0, 1).copy_(diff_norm.reshape({1}));
    results.narrow(0, 1, 1).copy_(grad_norm.reshape({1}));
}

}} // namespace sg::gfx942
