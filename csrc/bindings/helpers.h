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

// ── Fused (model, optimizer, arch) megakernel dispatch (impl in
//    dispatch.cpp). Declared here so bindings.cpp can bind &sg::fused_step.
void fused_step(const std::string& model, const std::string& optimizer,
                torch::Tensor params, torch::Tensor input,
                torch::Tensor grad, torch::Tensor state, float lr);

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

// ── Boundary validation for optimizer fused-step entrypoints ─────────
// The Python wrappers historically owned device/dtype/shape/contiguity
// invariants; these binding entrypoints trusted them. A non-contiguous param
// view, a device/dtype mismatch between param and grad, or a shape mismatch
// silently corrupts the in-place fused update (the launchers index a flat
// pointer with the param's numel). Validate at the C++ boundary so a bad call
// fails loudly. `where` is the entrypoint name for the error message.
//
// We check the param against its paired grad. Optimizer-state buffers (m, v,
// ema, …) are allocated by the wrapper from `torch.zeros_like(param)` so they
// inherit the param's device/dtype/shape/contiguity; the param check is the
// load-bearing one. A missing/empty grad (sparse-grad fallback) is skipped —
// the per-op loops already filter those out before dispatch.
inline void check_param_grad(
    const torch::Tensor& p,
    const torch::Tensor& g,
    const char* where
) {
    TORCH_CHECK(p.defined(), where, ": param tensor is undefined");
#if defined(WITH_HIP)
    TORCH_CHECK(p.is_hip(), where, ": param must be on a HIP device");
#else
    TORCH_CHECK(p.is_cuda(), where, ": param must be on a CUDA device");
#endif
    // A non-contiguous param view aliases a strided storage; the fused
    // launchers treat data_ptr() as a dense [numel] buffer, so a strided view
    // would read/write the wrong elements and silently corrupt the update.
    TORCH_CHECK(p.is_contiguous(), where,
                ": param must be contiguous (got a non-contiguous view); "
                "call .contiguous() in the Python wrapper before the fused step");
    if (!g.defined() || g.numel() == 0) return;  // sparse-grad: skip (filtered)
    TORCH_CHECK(g.device() == p.device(), where,
                ": grad device (", g.device().str(),
                ") != param device (", p.device().str(), ")");
    TORCH_CHECK(g.scalar_type() == p.scalar_type(), where,
                ": grad dtype (", toString(g.scalar_type()),
                ") != param dtype (", toString(p.scalar_type()), ")");
    TORCH_CHECK(g.sizes() == p.sizes(), where,
                ": grad shape (", g.sizes(), ") != param shape (",
                p.sizes(), ")");
    TORCH_CHECK(g.is_contiguous(), where, ": grad must be contiguous");
}

// Validate every (param, grad) pair in a multi-tensor fused-step entrypoint.
inline void check_params_grads(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const char* where
) {
    TORCH_CHECK(params.size() == grads.size(), where,
                ": params.size() (", params.size(),
                ") != grads.size() (", grads.size(), ")");
    for (size_t i = 0; i < params.size(); ++i)
        check_param_grad(params[i], grads[i], where);
}

// ── Device-side gradient clipping: single CPU sync instead of N ──────
inline void clip_grad_norms_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_params,
    float grad_clip_norm
) {
    if (grad_clip_norm <= 0.0f) return;

    // Collect the present grads once. Fused multi-tensor ops want a flat list.
    std::vector<torch::Tensor> present;
    present.reserve(n_params);
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0)
            present.push_back(grads[i]);
    }
    if (present.empty()) return;

    // Per-tensor L2 norms via a single fused multi-tensor reduction
    // (torch::_foreach_norm), replacing the per-tensor upcast + dot + add_
    // (≈2N kernel launches) with one foreach launch. Numerics: the L2 norm of
    // each tensor is accumulated into a fp32 sum-of-squares (squaring the
    // per-tensor norm == that tensor's sum-of-squares), matching the original
    // fp32 accumulation of the global sum-of-squares. To preserve the original
    // precision when grads are NOT already fp32, upcast those (only) to fp32
    // before the norm; fp32 grads skip the upcast entirely.
    bool all_fp32 = true;
    for (auto& g : present)
        if (g.scalar_type() != torch::kFloat32) { all_fp32 = false; break; }

    std::vector<torch::Tensor> norm_inputs;
    if (all_fp32) {
        norm_inputs = present;
    } else {
        norm_inputs.reserve(present.size());
        for (auto& g : present)
            norm_inputs.push_back(g.scalar_type() == torch::kFloat32
                                      ? g
                                      : g.to(torch::kFloat32));
    }

    auto norms = torch::_foreach_norm(norm_inputs, /*p=*/2);
    // norm_sq = sum_i (||g_i||_2)^2, accumulated in fp32 on-device, one sync.
    auto stacked = torch::stack(norms).to(torch::kFloat32);
    float total_norm =
        std::sqrt(stacked.dot(stacked).item<float>());

    if (total_norm > grad_clip_norm) {
        float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
        // Fused multi-tensor in-place scale of the present grads.
        torch::_foreach_mul_(present, clip_coef);
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
