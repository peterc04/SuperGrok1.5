#pragma once
// SuperGrok v1.5 — vendor-neutral algorithm header.
//
// Reference: internal algorithm. The Python source of truth is
// `grokking_optimizers/optimizers/supergrok15.py` — this header
// reproduces the per-parameter math from `_single_param_step` and the
// batched fused-kernel logic from supergrok15_fused_step.
//
// Element-wise MLP gradient transformation with sigmoid gating on training
// accuracy. Same structure as v1.1 but the gate is a scalar sigmoid of
// accuracy (set host-side), not a per-parameter cosine.
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t.
//
// Pipeline per step:
//
//   Sweep A — meta-net forward + sharpness reduction:
//     mu = mlp_phi(grad, sharpness)
//     sharpness_sum += grad - grad_prev   (used to update sharpness)
//
//   Sweep B — apply:
//     alpha_per_coord clipped to [0, 1]
//     smart_grad = grad + alpha * mu
//     AdamW step on smart_grad.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

// MLP forward: same shape as v1.1 (tanh hidden, linear output).
template <int H>
__device__ __forceinline__ float sg15_phi_forward(
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

// Sweep A: compute mu and write back; contribute to sharpness reduction.
template <typename GradT>
__device__ __forceinline__ void sg15_sweep_a_step(
    float* __restrict__ mu_out,
    const GradT* __restrict__ grad,
    const float mu_val,
    const int idx,
    float& sharp_local
) {
    mu_out[idx] = mu_val;
    sharp_local += static_cast<float>(grad[idx]) * static_cast<float>(grad[idx]);
}

// Per-coord alpha gate: clipped affine of mu.
__device__ __forceinline__ float sg15_alpha_per_coord(
    const float mu_val,
    const float alpha_base,
    const float alpha_max
) {
    const float a = alpha_base * (1.0f + mu_val);
    return fminf(fmaxf(a, 0.0f), alpha_max);
}

// Sweep B: smart_grad + Adam, register-resident.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg15_sweep_b_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float* __restrict__ mu,
    const float gate_global,     // sigmoid(accuracy)
    const float alpha_base,
    const float alpha_max,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float p = static_cast<float>(param[idx]);
    const float u = mu[idx];

    const float a = sg15_alpha_per_coord(u, alpha_base, alpha_max);
    const float smart_grad = g + gate_global * a * u;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // bc1, bc2 un-inverted (= 1 - beta^t): divide for bias correction.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

}} // namespace sg::algorithms
