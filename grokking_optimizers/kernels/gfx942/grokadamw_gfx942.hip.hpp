#ifndef GROKKING_KERNELS_GFX942_GROKADAMW_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_GROKADAMW_GFX942_HIP_HPP_
// ============================================================================
// grokadamw_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'grokadamw'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_grokadamw.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for GrokAdamW.
// Algorithm: csrc/algorithms/grokadamw.h
//
// COMPUTE PATTERN
// Elementwise EMA-amplified Adam. Per element:
//   ema = alpha * ema + (1-alpha) * g
//   g_amp = g + lamb * ema
//   then AdamW(g_amp) per launch_adamw.
// 4 state tensors (p, m, v, ema) + grad → 5 mem reads + 4 writes per element.
//
// MFMA APPLICABILITY: none. Pure elementwise.
//
// WHY ATEN HERE
// Same as launch_adamw. The ema update fuses with the Adam apply only if
// we hand-write a `__global__` kernel — the ATen path generates 2 kernel
// launches (one for ema, one for adam). The fusion gain is ~1.5×; the
// bandwidth bound stays the same.

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

void launch_grokadamw_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    float alpha, float lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& ema = emas[i];

        // EMA filter
        prim::ema_update_inplace(ema, g, alpha);
        // Amplified gradient
        auto g_amp = g.to(torch::kFloat32) + lamb * ema;
        // Adam moments on g_amp
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_fused_grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps, float bc1, float bc2
) {
    std::vector<torch::Tensor> vp{param}, vea{exp_avg}, veas{exp_avg_sq},
                               vema{ema}, vg{grad};
    launch_grokadamw_step(vp, vea, veas, vema, vg,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
}

void launch_fused_grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, float clip_threshold
) {
    if (clip_threshold > 0.0f) {
        auto gn = grad.norm().item<float>();
        if (gn > clip_threshold) grad = grad.mul(clip_threshold / gn);
    }
    launch_fused_grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                                alpha, lamb, beta1, beta2, lr, weight_decay,
                                eps, bc1, bc2);
}

void launch_fused_grokadamw_step_q3(
    torch::Tensor param, torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales, torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16, torch::Tensor grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps,
    float bc1, float bc2, unsigned global_step
) {
    auto ea = exp_avg_int8.to(torch::kFloat32) * exp_avg_scales.repeat_interleave(
        exp_avg_int8.numel() / exp_avg_scales.numel());
    auto eas = exp_avg_sq_bf16.to(torch::kFloat32);
    auto ema_f = ema_bf16.to(torch::kFloat32);
    std::vector<torch::Tensor> vp{param}, vea{ea}, veas{eas},
                               vema{ema_f}, vg{grad};
    launch_grokadamw_step(vp, vea, veas, vema, vg,
                          alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                          bc1, bc2);
    auto scale = ea.abs().max();
    if (scale.item<float>() < 1e-12f) scale = torch::ones({1}, ea.options());
    exp_avg_scales.fill_(scale.item<float>() / 127.0f);
    exp_avg_int8.copy_((ea / (scale / 127.0f)).clamp(-127, 127).to(torch::kInt8));
    exp_avg_sq_bf16.copy_(eas.to(torch::kBFloat16));
    ema_bf16.copy_(ema_f.to(torch::kBFloat16));
}

void launch_multi_tensor_grokadamw(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float wd, float eps
) {
    for (size_t i = 0; i < params.size(); i++) {
        std::vector<torch::Tensor> vp{params[i]}, vea{exp_avgs[i]},
            veas{exp_avg_sqs[i]}, vema{emas[i]}, vg{grads[i]};
        launch_grokadamw_step(vp, vea, veas, vema, vg,
                              alpha, lamb, lr, beta1, beta2, eps, wd,
                              bc1s[i], bc2s[i]);
    }
}

void launch_fused_adamw_simple(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    for (size_t t = 0; t < params.size(); t++) {
        if (!grads[t].defined() || grads[t].numel() == 0) continue;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[t]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[t]));
        auto& p = params[t]; auto& g = grads[t];
        auto& m = exp_avgs[t]; auto& v = exp_avg_sqs[t];
        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_GROKADAMW_GFX942_HIP_HPP_
