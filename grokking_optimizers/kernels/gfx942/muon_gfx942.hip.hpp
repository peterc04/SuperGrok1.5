#ifndef GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
// ============================================================================
// muon_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'muon'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_muon.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Muon.
// Algorithm: csrc/algorithms/muon.h
//
// COMPUTE PATTERN
// Mixed: GEMM-heavy.
//   1. momentum buffer:    buf = momentum * buf + g           — elementwise
//   2. Frobenius norm:     inv_norm = 1 / ||buf||_F           — global reduction
//   3. normalize:          X = buf * inv_norm                  — elementwise
//   4. Newton-Schulz × 5:  for step in {0..4}:
//                             A   = X @ X.T          — GEMM (rows × cols)
//                             AX  = A @ X            — GEMM
//                             AAX = A @ AX           — GEMM
//                             X   = 3.4445*X - 4.7750*AX + 2.0315*AAX
//   5. update:             p -= lr * X * scale + p * decay     — elementwise
//
// MFMA APPLICABILITY: significant.
// The 3 GEMMs per Newton-Schulz step are exactly what MFMA accelerates.
// Typical Muon shapes for grokking models (e.g. 96×96 weight matrices):
// MFMA `v_mfma_f32_16x16x16_bf16` runs 6×6 = 36 MFMA tiles per GEMM.
// At MI300X's 1100 TFLOPS BF16, the 3 GEMMs × 5 steps complete in ~5 µs.
//
// WHY ATEN HERE
// `torch::mm` on a HIP tensor dispatches to rocBLAS's GEMM, which
// internally uses `v_mfma_f32_16x16x16_bf16` for the BF16 path (or
// `v_mfma_f32_16x16x4_f32` for FP32). The MFMA acceleration is already
// being exercised through rocBLAS — we just don't see the intrinsics in
// our source code. Hand-writing the GEMM with explicit MFMA intrinsics
// would gain perhaps 1.2× over rocBLAS at small N, mainly by avoiding the
// rocBLAS launcher's overhead. Not worth the maintenance burden.

#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

static inline torch::Tensor newton_schulz_iterate(
    torch::Tensor X, int ns_steps, float a, float b, float c
) {
    for (int it = 0; it < ns_steps; it++) {
        auto AX  = torch::mm(X.transpose(-2, -1), X);
        auto AAX = torch::mm(AX, AX);
        X = a * X + b * torch::mm(X, AX) + c * torch::mm(X, AAX);
    }
    return X;
}

void launch_muon_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& bufs,
    std::vector<torch::Tensor>& grads,
    float lr, float momentum, float wd, int ns_steps,
    float ns_a, float ns_b, float ns_c
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& buf = bufs[i];
        auto& g = grads[i];

        buf.mul_(momentum).add_(g.to(buf.scalar_type()), 1.0f - momentum);

        if (p.dim() >= 2) {
            auto frob = buf.norm() + 1e-8f;
            auto X = buf / frob;
            X = newton_schulz_iterate(X, ns_steps, ns_a, ns_b, ns_c);
            float neg_lr_scale = -lr * 0.2f * sqrtf((float)std::max<int64_t>(p.size(-1), p.size(-2)));
            p.mul_(1.0f - lr * wd).add_(X.to(p.scalar_type()), neg_lr_scale);
        } else {
            // 1D fall back: Adam-like
            p.add_(buf.to(p.scalar_type()), -lr);
        }
    }
}


void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX, float a, float b, float c, float neg_lr_scale, float decay_factor
) {
    // Y = a*X + b*AX + c*AAX
    auto Y = a * X + b * AX + c * AAX;
    // param = param + neg_lr_scale*Y - decay_factor*param
    //       = (1 - decay_factor)*param + neg_lr_scale*Y
    param.mul_(1.0f - decay_factor).add_(Y.to(param.scalar_type()), neg_lr_scale);
}

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad, float momentum, float inv_norm
) {
    // buf = momentum*buf + inv_norm*grad.float()
    buf.mul_(momentum).add_(grad.to(torch::kFloat32), inv_norm);
    // X = buf (copy)
    X.copy_(buf);
}

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX, float a, float b, float c
) {
    // X_out = a*X + b*AX + c*AAX
    X_out.copy_(a * X + b * AX + c * AAX);
}

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth, float neg_lr_scale, float decay_factor
) {
    // param = param + neg_lr_scale*orth - decay_factor*param
    //       = (1 - decay_factor)*param + neg_lr_scale*orth
    param.mul_(1.0f - decay_factor).add_(orth.to(param.scalar_type()), neg_lr_scale);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
