#pragma once
// GrokAdamW -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/grokadamw_gfx942.hip.cpp
//
// Adam with grokking-aware gradient filtering and amplification:
//   1. EMA filter: filtered = alpha * ema + (1 - alpha) * grad
//   2. Amplification: amplified = grad + lamb * filtered
//   3. Adam: moments + bias-corrected step + decoupled weight decay
//
// Includes quantized (Q3) variant with INT8 moments and BF16 EMA.
// BF16 MFMA paths for matrix ops are handled at the launch-wrapper level.
// The __global__ launch wrappers remain in csrc/kernels/ (or csrc/fused/).

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

// =========================================================================
//  Fused GrokAdamW per-element step
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void grokadamw_step(
    scalar_t* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const scalar_t* __restrict__ grad,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int idx
) {
    const float g = static_cast<float>(grad[idx]);
    const float e = ema[idx];

    // -- 1. EMA gradient filter
    const float filtered = alpha * e + (1.0f - alpha) * g;
    ema[idx] = filtered;

    // -- 2. Gradient amplification
    const float amplified = g + lamb * filtered;

    // -- 3. Adam moment updates
    const float ea_old = exp_avg[idx];
    const float easq_old = exp_avg_sq[idx];

    const float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    const float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    // -- 4. Bias-corrected Adam step with decoupled weight decay
    const float step_size = lr / bc1;
    const float rsqrt_v = rsqrtf(easq / bc2 + 1e-30f);

    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea * rsqrt_v / (1.0f + eps * rsqrt_v);
    param[idx] = static_cast<scalar_t>(p);
}

// =========================================================================
//  FP32 vec4 variant
// =========================================================================

__device__ __forceinline__ void grokadamw_step_vec4(
    float4* __restrict__ param4,
    float4* __restrict__ exp_avg4,
    float4* __restrict__ exp_avg_sq4,
    float4* __restrict__ ema4,
    const float4* __restrict__ grad4,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const int i
) {
    float4 p = param4[i];
    float4 g = grad4[i];
    float4 e = ema4[i];
    float4 ea = exp_avg4[i];
    float4 eas = exp_avg_sq4[i];

    // EMA filter
    e.x = alpha * e.x + (1.0f - alpha) * g.x;
    e.y = alpha * e.y + (1.0f - alpha) * g.y;
    e.z = alpha * e.z + (1.0f - alpha) * g.z;
    e.w = alpha * e.w + (1.0f - alpha) * g.w;
    ema4[i] = e;

    // Amplification
    float4 amp;
    amp.x = g.x + lamb * e.x;
    amp.y = g.y + lamb * e.y;
    amp.z = g.z + lamb * e.z;
    amp.w = g.w + lamb * e.w;

    // Adam moments
    ea.x = beta1 * ea.x + (1.0f - beta1) * amp.x;
    ea.y = beta1 * ea.y + (1.0f - beta1) * amp.y;
    ea.z = beta1 * ea.z + (1.0f - beta1) * amp.z;
    ea.w = beta1 * ea.w + (1.0f - beta1) * amp.w;

    eas.x = beta2 * eas.x + (1.0f - beta2) * amp.x * amp.x;
    eas.y = beta2 * eas.y + (1.0f - beta2) * amp.y * amp.y;
    eas.z = beta2 * eas.z + (1.0f - beta2) * amp.z * amp.z;
    eas.w = beta2 * eas.w + (1.0f - beta2) * amp.w * amp.w;

    exp_avg4[i] = ea;
    exp_avg_sq4[i] = eas;

    // Step
    float step_size = lr / bc1;
    float decay = 1.0f - lr * weight_decay;
    float rsqrt_x = rsqrtf(eas.x / bc2 + 1e-30f);
    float rsqrt_y = rsqrtf(eas.y / bc2 + 1e-30f);
    float rsqrt_z = rsqrtf(eas.z / bc2 + 1e-30f);
    float rsqrt_w = rsqrtf(eas.w / bc2 + 1e-30f);
    p.x = decay * p.x - step_size * ea.x * rsqrt_x / (1.0f + eps * rsqrt_x);
    p.y = decay * p.y - step_size * ea.y * rsqrt_y / (1.0f + eps * rsqrt_y);
    p.z = decay * p.z - step_size * ea.z * rsqrt_z / (1.0f + eps * rsqrt_z);
    p.w = decay * p.w - step_size * ea.w * rsqrt_w / (1.0f + eps * rsqrt_w);
    param4[i] = p;
}

// =========================================================================
//  Quantized (Q3) variant: INT8 exp_avg, BF16 exp_avg_sq/ema
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void grokadamw_step_q3(
    scalar_t* __restrict__ param,
    int8_t* __restrict__ exp_avg_int8,
    float* __restrict__ exp_avg_scales,
    __nv_bfloat16* __restrict__ exp_avg_sq_bf16,
    __nv_bfloat16* __restrict__ ema_bf16,
    const scalar_t* __restrict__ grad,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const unsigned global_step,
    const int idx,
    const int quant_block_size
) {
    unsigned rng = global_step * 2654435761u ^ (unsigned)idx;

    const float g = static_cast<float>(grad[idx]);

    int block_idx = idx / quant_block_size;
    float ea_scale = exp_avg_scales[block_idx];
    float ea_old = (float)exp_avg_int8[idx] * ea_scale;

    float easq_old = __bfloat162float(exp_avg_sq_bf16[idx]);
    float e = __bfloat162float(ema_bf16[idx]);

    float filtered = alpha * e + (1.0f - alpha) * g;
    float amplified = g + lamb * filtered;

    float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    float new_scale = fmaxf(fabsf(ea), 1e-12f) / 127.0f;
    int8_t ea_q = (int8_t)fminf(fmaxf(rintf(ea / new_scale), -127.0f), 127.0f);
    exp_avg_int8[idx] = ea_q;
    if (idx % quant_block_size == 0) {
        exp_avg_scales[block_idx] = new_scale;
    }

    exp_avg_sq_bf16[idx] = __float2bfloat16(easq);
    ema_bf16[idx] = __float2bfloat16(filtered);

    float step_size = lr / bc1;
    float rsqrt_v = rsqrtf(easq / bc2 + 1e-30f);
    float p = static_cast<float>(param[idx]);
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea * rsqrt_v / (1.0f + eps * rsqrt_v);
    param[idx] = static_cast<scalar_t>(p);
}

// =========================================================================
//  Fused grad-clip + GrokAdamW: per-element step given precomputed clip_scale
// =========================================================================

__device__ __forceinline__ void grokadamw_clip_step_element(
    float* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ ema,
    const float* __restrict__ grad,
    const float alpha,
    const float lamb,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float bc1,
    const float bc2,
    const float clip_scale,
    const int idx
) {
    float g = grad[idx] * clip_scale;
    float e = ema[idx];

    float filtered = alpha * e + (1.0f - alpha) * g;
    ema[idx] = filtered;

    float amplified = g + lamb * filtered;

    float ea_old = exp_avg[idx];
    float easq_old = exp_avg_sq[idx];

    float ea = beta1 * ea_old + (1.0f - beta1) * amplified;
    float easq = beta2 * easq_old + (1.0f - beta2) * amplified * amplified;

    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    float step_size = lr / bc1;
    float rsqrt_v = rsqrtf(easq / bc2 + 1e-30f);
    float p = param[idx];
    p *= (1.0f - lr * weight_decay);
    p -= step_size * ea * rsqrt_v / (1.0f + eps * rsqrt_v);
    param[idx] = p;
}

}}} // namespace sg::device::gfx942
