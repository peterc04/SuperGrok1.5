#pragma once
// SuperGrok v1.1 — vendor-neutral algorithm header.
//
// Reference: internal algorithm. The Python source of truth is
// `grokking_optimizers/optimizers/supergrok11.py` — this header
// reproduces the per-parameter math from `_single_param_step` and the
// batched fused-kernel logic from supergrok11_fused_step.
//
// Element-wise MLP gradient transformation with cosine-similarity gating.
// Same structure as v1.5 but the gating term is the cosine between grad
// and momentum (computed per-parameter), not a sigmoid of training accuracy.
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t.
//
// Pipeline per step (two cooperative sweeps):
//
//   Sweep A — meta-net forward:
//     mu = mlp_phi(grad, sharpness)
//     gate_num += grad * momentum
//     gate_den_g += grad * grad
//     gate_den_m += momentum * momentum
//
//   Sweep B — apply:
//     gate = clamp(gate_num / sqrt(gate_den_g * gate_den_m + eps), 0, 1)
//     smart_grad = grad + (1 - gate) * alpha * mu
//     AdamW step on smart_grad with trust-ratio scaling.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// MLP forward: tanh hidden, linear output. Phi weights: W1[H,2], b1[H], W2[H], b2.
template <int H>
__device__ __forceinline__ float sg11_phi_forward(
    const float grad_val,
    const float sharp_val,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float b2
) {
    float h_acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < H; j++) {
        float h = W1[j * 2] * grad_val + W1[j * 2 + 1] * sharp_val + b1[j];
        h = tanhf(h);
        h_acc += W2[j] * h;
    }
    return h_acc + b2;
}

// Sweep A per-element: compute mu and contribute to cosine reductions.
template <typename GradT>
__device__ __forceinline__ void sg11_sweep_a_step(
    float* __restrict__ mu_out,
    const GradT* __restrict__ grad,
    const float* __restrict__ sharpness,
    const float* __restrict__ momentum,
    const float mu_val,
    const int64_t idx,
    float& gate_num_local,
    float& gate_den_g_local,
    float& gate_den_m_local
) {
    mu_out[idx] = mu_val;
    const float g = static_cast<float>(grad[idx]);
    const float m = momentum[idx];
    gate_num_local   += g * m;
    gate_den_g_local += g * g;
    gate_den_m_local += m * m;
}

// Sweep B per-element: cosine gate + smart_grad + Adam.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg11_sweep_b_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float* __restrict__ mu,
    const float gate,            // pre-reduced cosine similarity, clamped
    const float alpha,           // meta-net strength
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t idx
) {
    const float g  = static_cast<float>(grad[idx]);
    const float p  = static_cast<float>(param[idx]);
    const float u  = mu[idx];

    const float smart_grad = g + (1.0f - gate) * alpha * u;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// Canonical bias-corrected Adam tail on a caller-supplied effective gradient
// g_eff. SINGLE source of the moment-update + decoupled-WD apply math, factored
// out of sg11_sweep_b_step so the decoupled "Adam-on (smart_grad + lamb_eff*mu)"
// kernel can reuse the exact same float ops without re-inlining them. The
// caller computes its optimizer-specific g_eff; this writes the new moments
// into exp_avg[idx]/exp_avg_sq[idx] and the new parameter into param[idx].
//
// Bit-identical to the tail of sg11_sweep_b_step:
//   m = beta1 * exp_avg[idx]    + (1 - beta1) * g_eff;
//   v = beta2 * exp_avg_sq[idx] + (1 - beta2) * g_eff * g_eff;
//   update = (m / bc1) / (sqrtf(v / bc2) + eps);
//   param[idx] = p - lr * (update + wd * p);
template <typename ParamT>
__device__ __forceinline__ void sg11_adam_tail(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float g_eff,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t idx
) {
    const float p = static_cast<float>(param[idx]);
    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_eff;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_eff * g_eff;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
