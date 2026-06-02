#pragma once
// GrokAdamW — vendor-neutral algorithm header.
//
// Reference: Lee et al. 2024, "GrokAdamW: A Faster Optimizer for Late
// Generalization" — AdamW with an EMA gradient filter that amplifies
// persistent directions (the slow-changing components of the gradient
// signal).
//
// Math (decoupled weight decay variant; bias correction via division):
//   ema_t = alpha * ema_{t-1} + (1 - alpha) * g
//   g_amp = g + lamb * ema_t
//   m_t   = beta1 * m_{t-1} + (1 - beta1) * g_amp
//   v_t   = beta2 * v_{t-1} + (1 - beta2) * g_amp^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   p    -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
//
// Calling convention: bc1, bc2 are passed un-inverted —
//   bc1 = 1 - beta1^t,  bc2 = 1 - beta2^t.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace algorithms {

template <typename ParamT, typename GradT>
__device__ __forceinline__ void grokadamw_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const GradT* __restrict__ grad,
    const float alpha,    // EMA decay (e.g. 0.98)
    const float lamb,     // amplification factor (e.g. 5.0)
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

    // EMA filter
    const float ema_new = alpha * ema[idx] + (1.0f - alpha) * g;
    ema[idx] = ema_new;

    // Amplified gradient
    const float g_amp = g + lamb * ema_new;

    // Adam moments
    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    // Bias-corrected update with decoupled weight decay.
    // bc1, bc2 are passed un-inverted (= 1 - beta^t), so divide.
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// Canonical bias-corrected Adam tail on a caller-supplied effective gradient
// g_eff. This is the SINGLE source of the moment-update + decoupled-WD apply
// math, factored out of grokadamw_step so storage-variant kernels (e.g. the
// quantized Config-3 path, which dequantizes/requantizes the state) can reuse
// the exact same float ops without re-inlining them. Produces m_out, v_out
// (the new moments) and p_out (the new parameter value) by reference; the
// caller is responsible for writing them back (FP32 or quantized).
//
// Bit-identical to the tail of grokadamw_step:
//   m = beta1 * m_prev + (1 - beta1) * g_eff;
//   v = beta2 * v_prev + (1 - beta2) * g_eff * g_eff;
//   update = (m / bc1) / (sqrtf(v / bc2) + eps);
//   p_out  = p - lr * (update + wd * p);
__device__ __forceinline__ void grokadamw_adam_tail(
    const float g_eff,
    const float p,
    const float m_prev,
    const float v_prev,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    float& m_out,
    float& v_out,
    float& p_out
) {
    const float m = beta1 * m_prev + (1.0f - beta1) * g_eff;
    const float v = beta2 * v_prev + (1.0f - beta2) * g_eff * g_eff;
    const float update = (m / bc1) / (sqrtf(v / bc2) + eps);
    m_out = m;
    v_out = v;
    p_out = p - lr * (update + wd * p);
}

}} // namespace sg::algorithms
