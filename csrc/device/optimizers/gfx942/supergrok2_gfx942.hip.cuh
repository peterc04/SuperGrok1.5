#pragma once
// SuperGrok v2 -- Device-function templates for gfx942 (CDNA3 / MI300X).
// Migrated from csrc/kernels/hip/gfx942/supergrok2_{fwd,bwd}_gfx942.hip.cpp
//
// Mamba-3 + 4-Head PEER + GRU architecture kernels:
//   Forward:
//     1. input_proj_sort: project [grad, sharpness] -> [N, d_model], compute sort keys
//     2. mamba3_scan: sequential selective scan with trapezoidal disc. + RoPE
//   Backward:
//     3. bilevel_precompute: precompute projections for backward
//
// Note: gfx942 does NOT have warp-specialized kernels (those are Hopper-only).
// BF16 MFMA paths for matrix ops are handled at the launch-wrapper level.
// The __global__ launch wrappers remain in csrc/kernels/ (or csrc/fused/).

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"

namespace sg { namespace device { namespace gfx942 {

// Compile-time maximums (must match kernel source)
constexpr int SG2_MAX_D_STATE = 128;
constexpr int SG2_MAX_D_INNER = 128;
constexpr int SG2_MAX_D_MODEL = 64;

// =========================================================================
//  Forward kernel 1: Input Projection + Sort Key
//  Per-element: x[idx, d] = proj_W[d,0]*grad + proj_W[d,1]*sharp + proj_b[d]
//               sort_key = |grad|
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void input_proj_sort(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    float* __restrict__ x_out,
    float* __restrict__ sort_keys,
    int* __restrict__ sort_indices,
    const float* __restrict__ proj_W,
    const float* __restrict__ proj_b,
    const int idx,
    const int N,
    const int d_model
) {
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;

    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        x_out[idx * d_model + d] = proj_W[d * 2] * g + proj_W[d * 2 + 1] * s + proj_b[d];
    }

    sort_keys[idx] = fabsf(g);
    sort_indices[idx] = idx;
}

// =========================================================================
//  Forward kernel 2: Mamba-3 Selective Scan (per-thread, sequential)
//  One thread per d_inner dimension. State in registers.
//  Trapezoidal discretization + RoPE rotation on state pairs.
// =========================================================================

__device__ __forceinline__ void mamba3_scan_step(
    float* __restrict__ h,           // [d_state] state in registers
    const float* __restrict__ A,     // [d_state] preloaded A coefficients
    const float* __restrict__ freq,  // [d_state/2] RoPE frequencies
    const float x_val,
    const float dt_val,
    const float* __restrict__ B_vals,  // [d_state]
    const float* __restrict__ C_vals,  // [d_state]
    const float D_val,
    const float z_val,
    const int d_state,
    const int step_t,
    float* __restrict__ y_out        // single output value
) {
    const int half_d_state = d_state / 2;
    float y_acc = 0.0f;

    #pragma unroll 4
    for (int p = 0; p < half_d_state; p++) {
        int s0 = p * 2;
        int s1 = s0 + 1;

        // Trapezoidal discretization
        float dt_A0 = dt_val * A[s0];
        float dt_A1 = dt_val * A[s1];
        float dA0 = (1.0f + dt_A0 * 0.5f) / (1.0f - dt_A0 * 0.5f);
        float dA1 = (1.0f + dt_A1 * 0.5f) / (1.0f - dt_A1 * 0.5f);
        float dBx0 = B_vals[s0] * x_val * dt_val;
        float dBx1 = B_vals[s1] * x_val * dt_val;

        h[s0] = dA0 * h[s0] + dBx0;
        h[s1] = dA1 * h[s1] + dBx1;

        // RoPE rotation on state pair
        float cos_r = cosf(freq[p] * step_t);
        float sin_r = sinf(freq[p] * step_t);
        float h0_rot = h[s0] * cos_r - h[s1] * sin_r;
        float h1_rot = h[s0] * sin_r + h[s1] * cos_r;

        y_acc += C_vals[s0] * h0_rot + C_vals[s1] * h1_rot;
    }

    // Output: y + D*x, gated by silu(z)
    y_acc += D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y_acc * silu_z;
}

// =========================================================================
//  Backward: bilevel precompute per-timestep (one thread per timestep)
//  Computes input projection, dt projection, B/C projections.
// =========================================================================

__device__ __forceinline__ void bilevel_precompute_timestep(
    const float* __restrict__ x_sorted_t,   // [d_model] for this timestep
    const float* __restrict__ in_proj_W,     // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,     // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,     // [d_inner]
    const float* __restrict__ B_proj_W,      // [d_state, d_inner]
    const float* __restrict__ C_proj_W,      // [d_state, d_inner]
    float* __restrict__ pre_x_val_t,         // [d_inner] output
    float* __restrict__ pre_z_val_t,         // [d_inner] output
    float* __restrict__ pre_dt_val_t,        // [d_inner] output
    float* __restrict__ pre_B_val_t,         // [d_state] output
    float* __restrict__ pre_C_val_t,         // [d_state] output
    const int d_model,
    const int d_inner,
    const int d_state
) {
    float x_branch[SG2_MAX_D_INNER];

    // Input projection: x_branch and z
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float x_val = 0.0f, z_val = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            x_val += in_proj_W[j * d_model + d] * x_sorted_t[d];
            z_val += in_proj_W[(j + d_inner) * d_model + d] * x_sorted_t[d];
        }
        x_branch[j] = x_val;
        pre_x_val_t[j] = x_val;
        pre_z_val_t[j] = z_val;
    }

    // dt projection
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        #pragma unroll 4
        for (int k = 0; k < d_inner; k++) {
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        }
        // softplus activation
        pre_dt_val_t[j] = logf(1.0f + expf(dt_raw));
    }

    // B and C projections
    #pragma unroll 4
    for (int s = 0; s < d_state; s++) {
        float b_val = 0.0f, c_val = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < d_inner; j++) {
            b_val += B_proj_W[s * d_inner + j] * x_branch[j];
            c_val += C_proj_W[s * d_inner + j] * x_branch[j];
        }
        pre_B_val_t[s] = b_val;
        pre_C_val_t[s] = c_val;
    }
}

}}} // namespace sg::device::gfx942
