// =====================================================================
// helpers.h — shared bindings helpers + per-arch dispatch macro.
//
// Includes:
//   - int sg::detect_arch() forward decl (implemented in dispatch.cpp)
//   - SG_DISPATCH / SG_DISPATCH_CALL macros for runtime arch selection
//   - host-side gradient norm helpers extracted from the deleted
//     csrc/common/ops.cpp.
// =====================================================================

#pragma once

#include <torch/extension.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sg {

// ── Arch detection (impl in dispatch.cpp) ────────────────────────────
int detect_arch();
inline bool is_cuda_arch(int a) { return a == 90; }
inline bool is_hip_arch(int a)  { return a == 942; }

// ── Per-arch namespace handles ───────────────────────────────────────
namespace sm90 {}
namespace gfx942 {}

} // namespace sg

// ── Dispatch macro: returns from enclosing function ──────────────────
#define SG_DISPATCH(METHOD, ...) \
    do { \
        const int sg_arch_ = ::sg::detect_arch(); \
        switch (sg_arch_) { \
            case 90:  return ::sg::sm90::METHOD(__VA_ARGS__); \
            case 942: return ::sg::gfx942::METHOD(__VA_ARGS__); \
            default: \
                throw std::runtime_error( \
                    std::string(#METHOD) + " dispatch: unsupported arch " + \
                    std::to_string(sg_arch_)); \
        } \
    } while (0)

// ── Dispatch macro: same dispatch, no return ─────────────────────────
#define SG_DISPATCH_CALL(METHOD, ...) \
    do { \
        const int sg_arch_ = ::sg::detect_arch(); \
        switch (sg_arch_) { \
            case 90:  ::sg::sm90::METHOD(__VA_ARGS__); break; \
            case 942: ::sg::gfx942::METHOD(__VA_ARGS__); break; \
            default: \
                throw std::runtime_error( \
                    std::string(#METHOD) + " dispatch: unsupported arch " + \
                    std::to_string(sg_arch_)); \
        } \
    } while (0)

namespace sg {

// ── Device-side gradient clipping: single CPU sync instead of N ──────
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

// ── Device-side SAM grad-norm: single CPU sync instead of N ──────────
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
