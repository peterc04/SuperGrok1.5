#pragma once
// Canonical header (de-inlined). Body is byte-identical to the
// formerly copy-pasted block; prerequisites are included so that
// platform macros precede their use.

// HIP gfx942 (CDNA3 / MI300X) primitives — shared across all 11 launch files.
//
// Note: PyTorch routes `.hip.cpp` through the host compiler (g++/clang++),
// not through hipcc. This means primitives here cannot contain `__global__`
// kernels or `<<<...>>>` launch syntax. Instead, primitives here are
// host-side helpers (ATen tensor ops, dtype/device checks, gradient
// filtering) that the launch_*.hip.cpp files call.
//
// The actual GPU work is done by ATen / rocBLAS / hipBLAS via the
// PyTorch C++ API on the active HIP stream.

#include <torch/extension.h>
#include <vector>
#include <cstdint>

// =========================================================================
//  Post-launch error check (HIP mirror of sm_90's SG_LAUNCH_CHECK).
//
//  SG_HIP_LAUNCH_CHECK(stream) is called immediately AFTER a kernel launch /
//  HIP op. It reads hipGetLastError() to surface a launch-time failure and
//  TORCH_CHECKs it, turning a silent async error into an immediate exception.
//
//  Release (default): launch-error-only, NO synchronize (keeps async overlap).
//  Define SG_HIP_LAUNCH_CHECK_SYNC=1 (debug) to additionally
//  hipStreamSynchronize and catch errors raised during kernel EXECUTION.
//
//  Macro (not a function) so __FILE__/__LINE__ point at the launch site; the
//  do{...}while(0) wrapper makes it a single statement needing a trailing ;.
//  hipGetLastError / hipSuccess come from <hip/hip_runtime.h>, which the HIP
//  launch TUs that use this macro already include.
// =========================================================================
#ifndef SG_HIP_LAUNCH_CHECK_SYNC
#define SG_HIP_LAUNCH_CHECK_SYNC 0
#endif

#if SG_HIP_LAUNCH_CHECK_SYNC
#define SG_HIP_LAUNCH_CHECK(stream)                                          \
    do {                                                                     \
        hipError_t _sg_launch_err = hipGetLastError();                       \
        TORCH_CHECK(_sg_launch_err == hipSuccess,                            \
                    "HIP kernel launch failed: ",                            \
                    hipGetErrorString(_sg_launch_err));                      \
        hipError_t _sg_sync_err = hipStreamSynchronize(stream);              \
        TORCH_CHECK(_sg_sync_err == hipSuccess,                              \
                    "HIP kernel execution failed: ",                         \
                    hipGetErrorString(_sg_sync_err));                        \
    } while (0)
#else
#define SG_HIP_LAUNCH_CHECK(stream)                                          \
    do {                                                                     \
        (void)(stream);                                                      \
        hipError_t _sg_launch_err = hipGetLastError();                       \
        TORCH_CHECK(_sg_launch_err == hipSuccess,                            \
                    "HIP kernel launch failed: ",                            \
                    hipGetErrorString(_sg_launch_err));                      \
    } while (0)
#endif

namespace sg { namespace gfx942 { namespace primitives {

// =========================================================================
//  Validate that a tensor is on the active HIP/CUDA device.
// =========================================================================

inline void check_device(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be on a HIP/CUDA device");
}

// =========================================================================
//  Filter (param, grad, state...) tuples to skip params with undefined
//  or zero-size gradients. Returns parallel vectors of valid entries.
// =========================================================================

template <typename... Tensors>
inline bool keep_tensor(const torch::Tensor& grad) {
    return grad.defined() && grad.numel() > 0;
}

// =========================================================================
//  ATen-driven element-wise update helpers.
//  These build the optimizer math out of broadcasted tensor ops.
//  PyTorch dispatches them to hipBLAS / hipDNN / pure HIP kernels.
// =========================================================================

// In-place: m = beta1 * m + (1 - beta1) * g
inline void ema_update_inplace(
    torch::Tensor& m, const torch::Tensor& g, float beta1
) {
    m.mul_(beta1).add_(g, 1.0f - beta1);
}

// In-place: v = beta2 * v + (1 - beta2) * g^2
inline void ema_sq_update_inplace(
    torch::Tensor& v, const torch::Tensor& g, float beta2
) {
    v.mul_(beta2).addcmul_(g, g, 1.0f - beta2);
}

// In-place: p = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
inline void adam_apply_inplace(
    torch::Tensor& p, const torch::Tensor& m, const torch::Tensor& v,
    float lr, float bc1, float bc2, float eps, float wd
) {
    auto m_hat = m / bc1;  // bc1 = 1 - beta1^t (un-inverted)
    auto v_hat = v / bc2;  // bc2 = 1 - beta2^t (un-inverted)
    auto denom = v_hat.sqrt().add_(eps);
    auto update = m_hat.div_(denom).add_(p, wd);
    p.add_(update, -lr);
}

// =========================================================================
//  Tensor-pack helper for multi-tensor optimizer paths.
//  Collects valid (param, grad, ...) pairs into a contiguous std::vector.
// =========================================================================

struct TensorPack {
    std::vector<torch::Tensor> params;
    std::vector<torch::Tensor> grads;
    std::vector<torch::Tensor> state_a;
    std::vector<torch::Tensor> state_b;
};

inline TensorPack pack_valid(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& state_a = {},
    const std::vector<torch::Tensor>& state_b = {}
) {
    TensorPack out;
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        out.params.push_back(params[i]);
        out.grads.push_back(grads[i]);
        if (!state_a.empty()) out.state_a.push_back(state_a[i]);
        if (!state_b.empty()) out.state_b.push_back(state_b[i]);
    }
    return out;
}

}}} // namespace sg::gfx942::primitives
