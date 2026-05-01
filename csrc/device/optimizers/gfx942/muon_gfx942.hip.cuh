#pragma once
// Muon -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/muon_gfx942.hip.cpp
//
// Newton-Schulz orthogonalization element-wise ops:
//   1. momentum_normalize: buf = mom*buf + grad; X = buf/||buf||
//   2. ns_combine: X = a*X + b*AX + c*AAX
//   3. update: p += neg_lr_scale*orth; p *= decay_factor
//   4. ns_combine_update_fused: combine + update in one pass
//
// BF16 MFMA paths for matrix ops are handled at the launch-wrapper level.
// The __global__ launch wrappers remain in csrc/kernels/ (or csrc/fused/).

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

// =========================================================================
//  momentum_normalize: buf = mom*buf + grad; X = buf * inv_norm
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void muon_momentum_normalize(
    scalar_t* __restrict__ buf,
    scalar_t* __restrict__ X,
    const scalar_t* __restrict__ grad,
    const float momentum,
    const float inv_norm,
    const int idx
) {
    float b = momentum * static_cast<float>(buf[idx]) + static_cast<float>(grad[idx]);
    buf[idx] = static_cast<scalar_t>(b);
    X[idx] = static_cast<scalar_t>(b * inv_norm);
}

// FP32 vec4 variant
__device__ __forceinline__ void muon_momentum_normalize_vec4(
    float4* __restrict__ buf4,
    float4* __restrict__ X4,
    const float4* __restrict__ grad4,
    const float momentum,
    const float inv_norm,
    const int i
) {
    float4 b = buf4[i];
    float4 g = grad4[i];

    b.x = momentum * b.x + g.x;
    b.y = momentum * b.y + g.y;
    b.z = momentum * b.z + g.z;
    b.w = momentum * b.w + g.w;
    buf4[i] = b;

    float4 x;
    x.x = b.x * inv_norm;
    x.y = b.y * inv_norm;
    x.z = b.z * inv_norm;
    x.w = b.w * inv_norm;
    X4[i] = x;
}

// =========================================================================
//  ns_combine: X_out = a*X + b*AX + c*AAX
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void muon_ns_combine(
    scalar_t* __restrict__ X_out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ AX,
    const scalar_t* __restrict__ AAX,
    const float a,
    const float b,
    const float c,
    const int idx
) {
    float x = static_cast<float>(X[idx]);
    float ax = static_cast<float>(AX[idx]);
    float aax = static_cast<float>(AAX[idx]);
    X_out[idx] = static_cast<scalar_t>(a * x + b * ax + c * aax);
}

// FP32 vec4 variant
__device__ __forceinline__ void muon_ns_combine_vec4(
    float4* __restrict__ X_out4,
    const float4* __restrict__ X4,
    const float4* __restrict__ AX4,
    const float4* __restrict__ AAX4,
    const float a,
    const float b,
    const float c,
    const int i
) {
    float4 x = X4[i];
    float4 ax = AX4[i];
    float4 aax = AAX4[i];

    float4 out;
    out.x = a * x.x + b * ax.x + c * aax.x;
    out.y = a * x.y + b * ax.y + c * aax.y;
    out.z = a * x.z + b * ax.z + c * aax.z;
    out.w = a * x.w + b * ax.w + c * aax.w;
    X_out4[i] = out;
}

// =========================================================================
//  update: p += neg_lr_scale * orth; p *= decay_factor
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void muon_update(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ orth,
    const float neg_lr_scale,
    const float decay_factor,
    const int idx
) {
    float p = static_cast<float>(param[idx]);
    float o = static_cast<float>(orth[idx]);
    p += neg_lr_scale * o;
    p *= decay_factor;
    param[idx] = static_cast<scalar_t>(p);
}

// FP32 vec4 variant
__device__ __forceinline__ void muon_update_vec4(
    float4* __restrict__ param4,
    const float4* __restrict__ orth4,
    const float neg_lr_scale,
    const float decay_factor,
    const int i
) {
    float4 p = param4[i];
    float4 o = orth4[i];

    p.x = (p.x + neg_lr_scale * o.x) * decay_factor;
    p.y = (p.y + neg_lr_scale * o.y) * decay_factor;
    p.z = (p.z + neg_lr_scale * o.z) * decay_factor;
    p.w = (p.w + neg_lr_scale * o.w) * decay_factor;
    param4[i] = p;
}

// =========================================================================
//  ns_combine_update_fused: combine + update, orth stays in register
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void muon_ns_combine_update_fused(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ AX,
    const scalar_t* __restrict__ AAX,
    const float a,
    const float b,
    const float c,
    const float neg_lr_scale,
    const float decay_factor,
    const int idx
) {
    float x = static_cast<float>(X[idx]);
    float ax = static_cast<float>(AX[idx]);
    float aax = static_cast<float>(AAX[idx]);
    float orth = a * x + b * ax + c * aax;

    float p = static_cast<float>(param[idx]);
    p += neg_lr_scale * orth;
    p *= decay_factor;
    param[idx] = static_cast<scalar_t>(p);
}

// FP32 vec4 variant
__device__ __forceinline__ void muon_ns_combine_update_fused_vec4(
    float4* __restrict__ param4,
    const float4* __restrict__ X4,
    const float4* __restrict__ AX4,
    const float4* __restrict__ AAX4,
    const float a,
    const float b,
    const float c,
    const float neg_lr_scale,
    const float decay_factor,
    const int i
) {
    float4 x = X4[i];
    float4 ax = AX4[i];
    float4 aax = AAX4[i];
    float4 p = param4[i];

    p.x = (p.x + neg_lr_scale * (a * x.x + b * ax.x + c * aax.x)) * decay_factor;
    p.y = (p.y + neg_lr_scale * (a * x.y + b * ax.y + c * aax.y)) * decay_factor;
    p.z = (p.z + neg_lr_scale * (a * x.z + b * ax.z + c * aax.z)) * decay_factor;
    p.w = (p.w + neg_lr_scale * (a * x.w + b * ax.w + c * aax.w)) * decay_factor;
    param4[i] = p;
}

}}} // namespace sg::device::gfx942
