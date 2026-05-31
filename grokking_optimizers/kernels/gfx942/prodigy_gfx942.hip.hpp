#ifndef GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
// ============================================================================
// prodigy_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'prodigy'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_prodigy.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
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


void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor s, torch::Tensor param_init, torch::Tensor grad, float lr, float d_lr, float beta1, float beta2, float weight_decay, float eps, float bc1, float bc2
) {
    auto d_t = torch::tensor({d_lr},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> vm{exp_avg};
    std::vector<torch::Tensor> vv{exp_avg_sq};
    std::vector<torch::Tensor> vs{s};
    std::vector<torch::Tensor> vpi{param_init};
    std::vector<torch::Tensor> vg{grad};
    launch_prodigy_step(vp, vm, vv, vs, vpi, vg,
                        d_t, d_lr,
                        beta1, beta2, eps, weight_decay, bc1, bc2);
}

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init, torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator, float eps
) {
    // numerator += sum(grad * (param - param_init))
    auto gf = grad.to(torch::kFloat32);
    auto delta = (param - param_init).to(torch::kFloat32);
    numerator.add_((gf * delta).sum());
    // denominator += sum(|s|)
    denominator.add_(s.to(torch::kFloat32).abs().sum());
}

void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& grads, std::vector<torch::Tensor>& param_inits, std::vector<torch::Tensor>& exp_avgs, std::vector<torch::Tensor>& exp_avg_sqs, std::vector<torch::Tensor>& s_bufs, std::vector<float>& bc1s, std::vector<float>& bc2s, torch::Tensor d_lr_buf, float beta1, float beta2, float lr, float wd, float eps
) {
    if (params.empty()) return;
    auto dev = params[0].device();
    float d_prev = d_lr_buf.item<float>();

    // Phase 1: accumulate r and s across all tensors
    auto r_sum = torch::zeros({}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto s_sum = torch::zeros({}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = grads[i].to(torch::kFloat32);
        auto delta = (param_inits[i] - params[i]).to(torch::kFloat32);
        r_sum += (gf * delta).sum() * d_prev;
        s_sum += gf.abs().sum() * (d_prev * d_prev);
    }

    // Phase 2: update d
    auto candidate = r_sum / (s_sum.abs() + 1e-12f);
    d_lr_buf.copy_(torch::maximum(d_lr_buf.new_full({}, d_prev), candidate));
    float d_val = d_lr_buf.item<float>();

    // Phase 3: apply Adam with d as effective lr
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = d_val * grads[i].to(torch::kFloat32);
        prim::ema_update_inplace(exp_avgs[i], gf, beta1);
        prim::ema_sq_update_inplace(exp_avg_sqs[i], gf, beta2);
        s_bufs[i].add_(grads[i].to(torch::kFloat32), d_val);
        prim::adam_apply_inplace(params[i], exp_avgs[i], exp_avg_sqs[i],
                                 d_val, bc1s[i], bc2s[i], eps, wd);
    }
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
