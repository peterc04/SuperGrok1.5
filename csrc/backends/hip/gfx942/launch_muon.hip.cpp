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

namespace sg { namespace hip_gfx942 { namespace primitives {

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

}}} // namespace sg::hip_gfx942::primitives
// ── end inlined csrc/backends/hip/gfx942/primitives.hpp ──

namespace sg { namespace hip_gfx942 {

namespace prim = ::sg::hip_gfx942::primitives;

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

}} // namespace sg::hip_gfx942
