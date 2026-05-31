#ifndef GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
// ============================================================================
// supergrok11_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'supergrok11'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_supergrok11.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for SuperGrok v1.1.
// Algorithm: csrc/algorithms/supergrok11.h
//
// COMPUTE PATTERN
// Mixed: per-parameter meta-MLP + cosine-similarity gating + AdamW.
//   Per element:
//     mu = phi_mlp(grad, sharpness)        — 2-input × H × 1 MLP
//     cosine = sum(grad * momentum) / (||grad|| * ||momentum||)
//     gate   = clamp(cosine, 0, 1)
//     smart_grad = g + (1-gate) * alpha * mu
//     AdamW(smart_grad)
// The cosine numerator + two denominators are global reductions.
//
// MFMA APPLICABILITY: partial (same as NeuralGrok).
// The MLP forward could route through MFMA if we batch across N; in
// practice ATen + rocBLAS already does this via the rocBLAS dispatch.

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

void launch_supergrok11_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mu_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sharpnesses,
    std::vector<torch::Tensor>& momenta,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& mu = mu_bufs[i];

        // Sweep A: meta-net forward
        auto x = torch::stack({g.to(torch::kFloat32).view({-1}),
                               sharpnesses[i].view({-1})}, /*dim=*/1);
        auto h = torch::tanh(torch::matmul(x, phi_W1.t()) + phi_b1);
        auto mu_flat = (torch::matmul(h, phi_W2.unsqueeze(1)) + phi_b2).view_as(g);
        mu.copy_(mu_flat);

        // Cosine gate
        auto gf = g.to(torch::kFloat32);
        auto mom = momenta[i];
        float dot = (gf * mom).sum().item<float>();
        float ng = gf.norm().item<float>();
        float nm = mom.norm().item<float>();
        float gate = (ng * nm > 1e-12f) ? (dot / (ng * nm)) : 0.0f;
        gate = std::min(std::max(gate, 0.0f), 1.0f);

        // Sweep B: smart_grad + Adam
        auto smart = gf + (1.0f - gate) * alpha * mu;
        prim::ema_update_inplace(m, smart, beta1);
        prim::ema_sq_update_inplace(v, smart, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness, torch::Tensor smart_grad, float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2, float rescale, int hidden_dim
) {
    // phi network forward: 2-input MLP with tanh activation
    auto gf = grad.to(torch::kFloat32).view({-1});
    auto sf = sharpness.view({-1});
    auto x = torch::stack({gf, sf}, /*dim=*/1);  // [N, 2]
    auto h = torch::tanh(torch::matmul(x, W1.t()) + b1);
    float b2_val = b2.item<float>();
    auto mu_flat = (torch::matmul(h, W2.unsqueeze(1)) + b2_val).view_as(grad) * rescale;
    mu.copy_(mu_flat);
    // smart_grad = grad + alpha * mu
    smart_grad.copy_(gf.view_as(grad) + alpha * mu_flat);
}

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor smart_grad, torch::Tensor mu, float lamb_eff, float beta1, float beta2, float lr, float wd_eff, float eps, float bc1, float bc2
) {
    // g = smart_grad + lamb_eff * mu, then Adam update
    auto g = smart_grad + lamb_eff * mu;
    prim::ema_update_inplace(exp_avg, g, beta1);
    prim::ema_sq_update_inplace(exp_avg_sq, g, beta2);
    prim::adam_apply_inplace(param, exp_avg, exp_avg_sq, lr, bc1, bc2, eps, wd_eff);
}

void launch_sg11_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    // param += rho_over_norm * grad
    param.add_(grad.to(param.scalar_type()), rho_over_norm);
}

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup, torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    // param = backup, sharpness = (sam_grad - normal_grad)^2
    param.copy_(backup);
    auto diff = sam_grad.to(torch::kFloat32) - normal_grad.to(torch::kFloat32);
    sharpness.copy_(diff * diff);
}

float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp
) {
    // cos_sim(smart_grad, mu) clamped to [0, 1]
    auto sg_f = smart_grad.to(torch::kFloat32).flatten();
    auto mu_f = mu.to(torch::kFloat32).flatten();
    float num = (sg_f * mu_f).sum().item<float>();
    float den_g = (sg_f * sg_f).sum().item<float>();
    float den_m = (mu_f * mu_f).sum().item<float>();
    float denom = sqrtf(den_g * den_m + 1e-12f);
    float gate = (denom > 0.0f) ? (num / denom) : 0.0f;
    return std::min(std::max(gate, 0.0f), 1.0f);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
