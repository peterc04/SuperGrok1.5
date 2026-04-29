// =====================================================================
//  bindings/_helpers.h — host-side helpers shared by per-optimizer
//  binding files.
//
//  These do NOT dispatch to per-arch kernels; they are CPU-side
//  reductions and bookkeeping that the high-level vector-signature
//  entry points share.
//
//  Origin: extracted from the deleted csrc/common/ops.cpp helpers
//  (clip_grad_norms_device_side, compute_sam_grad_norm_device_side).
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <cmath>
#include <vector>

namespace sg {

// Device-side gradient clipping: a single CPU sync instead of N.
// `grads` are mutated in place when total_norm > grad_clip_norm.
inline void clip_grad_norms_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_params,
    float grad_clip_norm
) {
    if (grad_clip_norm <= 0.0f) return;

    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }

    auto norm_sq = torch::zeros(
        {1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            auto g_flat = grads[i].to(torch::kFloat32).reshape(-1);
            norm_sq.add_(g_flat.dot(g_flat));
        }
    }
    float total_norm = std::sqrt(norm_sq.item<float>());
    if (total_norm > grad_clip_norm) {
        float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0)
                grads[i].mul_(clip_coef);
        }
    }
}

// Device-side SAM grad-norm: single CPU sync instead of N.
inline float compute_sam_grad_norm_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_grads
) {
    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }
    auto norm_sq = torch::zeros(
        {1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            auto g_flat = grads[i].to(torch::kFloat32).reshape(-1);
            norm_sq.add_(g_flat.dot(g_flat));
        }
    }
    return std::sqrt(norm_sq.item<float>()) + 1e-12f;
}

} // namespace sg
