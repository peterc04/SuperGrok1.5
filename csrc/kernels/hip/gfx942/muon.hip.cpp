// =====================================================================
//  csrc/kernels/hip/gfx942/muon.hip.cpp — ATen-based.
//  No CUTLASS dependency on HIP. Matmuls handled by torch::mm (rocBLAS).
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/muon.hip.h"

namespace sg { namespace gfx942 {

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm
) {
    if (buf.numel() == 0) return;
    // buf = momentum * buf + grad
    buf.mul_(momentum).add_(grad);
    // X = buf * inv_norm
    X.copy_((buf.to(torch::kFloat32) * inv_norm).to(X.scalar_type()));
}

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c
) {
    if (X.numel() == 0) return;
    // X_out = a*X + b*AX + c*AAX
    auto out = X.to(torch::kFloat32) * a
             + AX.to(torch::kFloat32) * b
             + AAX.to(torch::kFloat32) * c;
    X_out.copy_(out.to(X_out.scalar_type()));
}

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor
) {
    if (param.numel() == 0) return;
    // param = decay_factor * param + neg_lr_scale * orth
    auto p = param.to(torch::kFloat32) * decay_factor
           + orth.to(torch::kFloat32) * neg_lr_scale;
    param.copy_(p.to(param.scalar_type()));
}

void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X,
    torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c,
    float neg_lr_scale, float decay_factor
) {
    if (param.numel() == 0) return;
    auto orth = X.to(torch::kFloat32) * a
              + AX.to(torch::kFloat32) * b
              + AAX.to(torch::kFloat32) * c;
    auto p = param.to(torch::kFloat32) * decay_factor
           + orth * neg_lr_scale;
    param.copy_(p.to(param.scalar_type()));
}

}} // namespace sg::gfx942
