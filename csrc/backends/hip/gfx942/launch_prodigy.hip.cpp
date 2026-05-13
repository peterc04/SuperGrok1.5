// HIP gfx942 launch glue for Prodigy.
// Algorithm: csrc/algorithms/prodigy.h
//
// COMPUTE PATTERN
// Mixed: per-element + reduction.
//   Per element: r_local += g * (p_init - p) * d
//                s_local += d² * |g|
//                AdamW apply with d as the lr scale
//   Reduction:   r_global = sum(r_local) across all elements (single FP32 scalar)
//                s_global = sum(s_local) across all elements
//                d_new = max(d_prev, r_global / |s_global|)
// The reduction is the bottleneck: needs wavefront reduce → LDS tree reduce
// → cross-block (cooperative or atomic) final reduce.
//
// MFMA APPLICABILITY: none.
// The reduction needs wave-reduce (`__shfl_xor` with mask=64 on CDNA3),
// then LDS-tree across waves in a block, then a single `atomicAdd` to a
// global counter. No GEMM, no MFMA.
//
// WHY ATEN HERE
// ATen's `.sum()` dispatches to rocPRIM's segmented reduction, which on
// MI300X already uses wave-reduce + LDS-tree internally. The hand-written
// version would save the kernel launch overhead (~3 µs per launch) and
// fuse the partial r/s accumulation with the AdamW apply. Modest gain
// (~2×) that is hardware-verified or not at all.

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

void launch_prodigy_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_tracks,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& grads,
    torch::Tensor& d_t,
    float d_prev,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    // Reduce r, s across all parameters.
    auto r_sum = torch::zeros({}, d_t.options());
    auto s_sum = torch::zeros({}, d_t.options());
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& pi = param_inits[i];
        auto& g = grads[i];

        auto delta = (pi - p).to(torch::kFloat32);
        r_sum += (g.to(torch::kFloat32) * delta).sum() * d_prev;
        s_sum += (g.to(torch::kFloat32).abs().sum()) * (d_prev * d_prev);
    }

    // Update d (on-device scalar).
    auto candidate = r_sum / (s_sum.abs() + 1e-12f);
    d_t.copy_(torch::maximum(d_t.new_full({}, d_prev), candidate));

    float d_val = d_t.item<float>();

    // Apply Adam with d as effective lr.
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& st = s_tracks[i];

        auto g_scaled = d_val * g.to(torch::kFloat32);
        prim::ema_update_inplace(m, g_scaled, beta1);
        prim::ema_sq_update_inplace(v, g_scaled, beta2);
        st.add_(g.to(torch::kFloat32), d_val);
        prim::adam_apply_inplace(p, m, v, d_val, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::gfx942
