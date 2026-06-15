#pragma once
// NeuralGrok — vendor-neutral algorithm header.
//
// Reference: Wang et al. 2024, "NeuralGrok: Accelerating Grokking via
// Learned Gradient Amplification" — Adam with a learned per-element
// amplifier (psi-net).
//
// The amplifier is a 2-layer MLP that takes |grad| as input and outputs
// a multiplicative scale.
//
// Two-stage compute:
//   (1) psi_net forward:  s = mlp(|g|)  (per-element scaling factor)
//   (2) apply:            g_amp = (s * alpha + beta) * g
//                         AdamW step on g_amp
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t.
//
// The MLP weights live in constant memory on CUDA / LDS on HIP. The math
// here is the per-element body; the matmul is handled by the backend.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// 2-layer MLP forward for a single element, with weights passed inline.
// W1: [H], b1: [H], W2: [H], b2: scalar.
// Activation: ReLU on hidden layer.
template <int H>
__device__ __forceinline__ float neuralgrok_psi_forward(
    const float abs_grad,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float b2
) {
    float h_acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < H; j++) {
        float h = W1[j] * abs_grad + b1[j];
        h = (h > 0.0f) ? h : 0.0f;       // ReLU
        h_acc += W2[j] * h;
    }
    return h_acc + b2;
}

// Apply: g_amp = (s * alpha + beta) * g, then run AdamW on g_amp.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void neuralgrok_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float psi_scale,        // output of psi_net forward
    const float alpha,
    const float beta,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int64_t idx,
    const float clip_coef = 1.0f   // eager GLOBAL grad-norm clip coef (1.0 = inert); applied before psi+amp
) {
    const float g = static_cast<float>(grad[idx]) * clip_coef;
    const float p = static_cast<float>(param[idx]);

    const float g_amp = (psi_scale * alpha + beta) * g;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    const float update = (m / sg_safe_bc(bc1)) / (sqrtf(v / sg_safe_bc(bc2)) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// Canonical bias-corrected Adam tail on a caller-supplied effective gradient
// g_eff. SINGLE source of the moment-update + decoupled-WD apply math, factored
// out of neuralgrok_apply_step so the "Adam-only on a pre-amplified gradient"
// kernel can reuse the exact same float ops without re-inlining them. Writes
// the new moments into exp_avg[idx]/exp_avg_sq[idx] and the new parameter into
// param[idx].
//
// Bit-identical to the tail of neuralgrok_apply_step:
//   m = beta1 * exp_avg[idx]    + (1 - beta1) * g_eff;
//   v = beta2 * exp_avg_sq[idx] + (1 - beta2) * g_eff * g_eff;
//   update = (m / bc1) / (sqrtf(v / bc2) + eps);
//   param[idx] = p - lr * (update + wd * p);
template <typename ParamT>
__device__ __forceinline__ void neuralgrok_adam_tail(
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
    const float update = (m / sg_safe_bc(bc1)) / (sqrtf(v / sg_safe_bc(bc2)) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
