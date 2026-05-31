#ifndef GROKKING_KERNELS_GFX942_LION_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_LION_GFX942_HIP_HPP_
// ============================================================================
// lion_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'lion'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_lion.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Lion.
// Algorithm: csrc/algorithms/lion.h
//
// COMPUTE PATTERN
// Pure elementwise. For each param element:
//   interp  = beta1 * ema + (1 - beta1) * g       — 2 reads, 1 FMA
//   update  = sign(interp)                         — 1 op
//   param  -= lr * (update + wd * param)           — 1 FMA + 1 mul + 1 sub
//   ema     = beta2 * ema + (1 - beta2) * g        — 2 reads, 1 FMA
// No reduction, no GEMM. Bandwidth-bound (~6 mem ops per element).
//
// MFMA APPLICABILITY: none.
// MFMA pipes operate on 16×16×16 tiles for matrix-matrix multiply. Lion has
// no GEMM, so there is nothing for MFMA to accelerate. The optimal CDNA3
// kernel would use 256-byte (vec4 FP32 × 64-lane wavefront) coalesced
// loads/stores via `buffer_load_dword_x4`, with sign computation in
// registers. Bandwidth (≈ 1.6 TB/s on MI300X HBM3) is the bound.
//
// WHY ATEN HERE
// `.hip.cpp` files are routed through the host compiler (g++/clang++) by
// PyTorch's cpp_extension, not through hipcc. We cannot define `__global__`
// kernels here; all GPU work goes through ATen. ATen tensor ops dispatch
// to rocPRIM / rocPRIM-thrust which already produces coalesced vectorized
// kernels for elementwise math. A hand-written `__global__` kernel would
// be slightly faster (saving 2-3 kernel launches by fusing) but the
// bandwidth bound is the same. To migrate to a hand-written kernel:
//   1. Rename `.hip.cpp` → `.hip` (PyTorch routes `.hip` through hipcc).
//   2. Add `*.hip` to the source glob in setup.py for the HIP branch.
//   3. Implement `__global__ void lion_kernel(...)` + `hipLaunchKernelGGL(...)`.

#include <torch/extension.h>
#include <vector>

// ── inlined from former csrc/backends/hip/gfx942/primitives.hpp ──
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
// ── end inlined csrc/backends/hip/gfx942/primitives.hpp ──

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_lion_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float wd
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& ea = exp_avgs[i];

        // Interpolation, sign, update
        auto interp = beta1 * ea + (1.0f - beta1) * g.to(ea.scalar_type());
        auto upd = interp.sign();
        p.add_(upd + wd * p, -lr);

        // Momentum refresh
        ea.mul_(beta2).add_(g.to(ea.scalar_type()), 1.0f - beta2);
    }
}


void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad, float lr, float beta1, float beta2, float weight_decay
) {
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> ve{exp_avg};
    std::vector<torch::Tensor> vg{grad};
    launch_lion_step(vp, ve, vg, lr, beta1, beta2, weight_decay);
}

void launch_multi_tensor_lion(
    std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& exp_avgs, std::vector<torch::Tensor>& grads, float lr, float beta1, float beta2, float wd
) {
    launch_lion_step(params, exp_avgs, grads, lr, beta1, beta2, wd);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_LION_GFX942_HIP_HPP_
