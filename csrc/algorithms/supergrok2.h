#pragma once
// SuperGrok v2 — vendor-neutral algorithm header.
// Migrated and consolidated from:
//   - csrc/device/optimizers/sm_90/supergrok2_sm90.cuh (REAL device templates)
//   - csrc/kernels/cuda/sm_90/supergrok2_fwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_bwd.cuh
//   - csrc/kernels/cuda/sm_90/supergrok2_warp_specialized.cuh
//
// Mamba-3 + 4-Head PEER + per-element GRU + Adam pipeline.
//
// Per-step pipeline:
//   (1) input_proj_sort     : [grad, sharpness] -> [N, d_model], sort keys = |grad|
//   (2) mamba3_scan         : selective scan with trapezoidal discretization + RoPE
//                              (sequential for small N, parallel Blelloch for larger N,
//                               warp-specialized on Hopper for uniform d_state)
//   (3) peer_route          : product-key expert routing, top-4 of 144 experts
//   (4) gru_step            : per-element GRU integrates expert output with temporal state
//   (5) apply               : smart_grad + Adam + trust-ratio + decoupled weight decay
//
// Backward (used by bilevel meta-learning):
//   (6) bilevel_precompute  : reproduce forward projections needed for adjoint scan
//
// The heavy math (GEMMs, parallel scans, cluster reductions) is in the
// per-backend primitives; this header contains the per-element building
// blocks that are vendor-neutral.

#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/algorithms/adamw.h"  // pulled in for the merged moe_adam_step

namespace sg { namespace algorithms {

// Compile-time maximums (must match types.h constants).
constexpr int SG2_MAX_D_STATE = 128;
constexpr int SG2_MAX_D_INNER = 128;
constexpr int SG2_MAX_D_MODEL = 64;

// =========================================================================
//  Forward (1): Input Projection + Sort Key
//  x_out[idx, d] = proj_W[d,0]*grad + proj_W[d,1]*sharp + proj_b[d]
//  sort_key      = |grad|
// =========================================================================

template <typename scalar_t>
__device__ __forceinline__ void sg2_input_proj_sort(
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
    sort_keys[idx]    = fabsf(g);
    sort_indices[idx] = idx;
}

// =========================================================================
//  Forward (2): Sequential Mamba-3 Scan (per-thread, single timestep)
//  Trapezoidal discretization + RoPE on state pairs.
// =========================================================================

__device__ __forceinline__ void sg2_mamba3_scan_step(
    float* __restrict__ h,           // [d_state] state (registers / smem)
    const float* __restrict__ A,     // [d_state] preloaded A coefficients
    const float* __restrict__ freq,  // [d_state/2] RoPE frequencies
    const float x_val,
    const float dt_val,
    const float* __restrict__ B_vals,
    const float* __restrict__ C_vals,
    const float D_val,
    const float z_val,
    const int d_state,
    const int step_t,
    float* __restrict__ y_out
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

    // y + D*x gated by silu(z)
    y_acc += D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y_acc * silu_z;
}

// =========================================================================
//  Forward (2'): Warp-Specialized Scan (Hopper consumer per-timestep)
//  Producer warp loads data into double-buffered smem; this is the
//  consumer-side recurrence for a single (di, state-pair) tuple.
// =========================================================================

__device__ __forceinline__ void sg2_scan_consumer_step(
    float* __restrict__ h,           // [2] state pair in registers
    const float A0,
    const float A1,
    const float D_val,
    const float rope_f,
    const float x_val,
    const float z_val,
    const float dt_val,
    const float B0_val,
    const float B1_val,
    const float C0_val,
    const float C1_val,
    const int t,
    float* __restrict__ y_out
) {
    float dA0 = expf(A0 * dt_val);
    float dA1 = expf(A1 * dt_val);
    float dBx0 = B0_val * x_val * dt_val;
    float dBx1 = B1_val * x_val * dt_val;

    h[0] = dA0 * h[0] + dBx0;
    h[1] = dA1 * h[1] + dBx1;

    float h0_rot = h[0] * cosf(rope_f * t) - h[1] * sinf(rope_f * t);
    float h1_rot = h[0] * sinf(rope_f * t) + h[1] * cosf(rope_f * t);

    float y = C0_val * h0_rot + C1_val * h1_rot + D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y * silu_z;
}

// =========================================================================
//  Forward (2''): Warp-Specialized Scan, d_state=16 unrolled
//  All 8 state pairs processed in one consumer thread.
// =========================================================================

constexpr int SG2_D_STATE_16 = 16;
constexpr int SG2_D_STATE_16_PAIRS = 8;

__device__ __forceinline__ void sg2_scan_consumer_step_d16(
    float h[SG2_D_STATE_16],
    const float A_vals[SG2_D_STATE_16],
    const float rope_f[SG2_D_STATE_16_PAIRS],
    const float D_val,
    const float x_val,
    const float z_val,
    const float dt_val,
    const float B[SG2_D_STATE_16],
    const float C[SG2_D_STATE_16],
    const int t,
    float* __restrict__ y_out
) {
    float y_acc = 0.0f;
    #pragma unroll
    for (int p = 0; p < SG2_D_STATE_16_PAIRS; p++) {
        int s0 = p * 2;
        int s1 = s0 + 1;

        float dA0 = expf(A_vals[s0] * dt_val);
        float dA1 = expf(A_vals[s1] * dt_val);
        float dBx0 = B[s0] * x_val * dt_val;
        float dBx1 = B[s1] * x_val * dt_val;

        h[s0] = dA0 * h[s0] + dBx0;
        h[s1] = dA1 * h[s1] + dBx1;

        float cos_r = cosf(rope_f[p] * t);
        float sin_r = sinf(rope_f[p] * t);
        float h0_rot = h[s0] * cos_r - h[s1] * sin_r;
        float h1_rot = h[s0] * sin_r + h[s1] * cos_r;

        y_acc += C[s0] * h0_rot + C[s1] * h1_rot;
    }

    y_acc += D_val * x_val;
    float silu_z = z_val / (1.0f + expf(-z_val));
    *y_out = y_acc * silu_z;
}

// =========================================================================
//  Forward (3-5): GRU + apply tail (per-element)
//  Combines temporal memory, smart_grad, and Adam update.
//  PEER routing handles its own selection on the host side; this is
//  the per-element body that consumes the routed expert output.
// =========================================================================

template <typename ParamT, typename GradT>
__device__ __forceinline__ void sg2_apply_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ mu_state,    // GRU hidden state
    const GradT* __restrict__ grad,
    const float expert_out,          // PEER expert output for this element
    const float alpha,
    const float gru_decay,
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

    // GRU step: simple gated update of mu_state with expert_out as candidate.
    const float mu_new = gru_decay * mu_state[idx] + (1.0f - gru_decay) * expert_out;
    mu_state[idx] = mu_new;

    const float smart_grad = g + alpha * mu_new;

    const float m = beta1 * exp_avg[idx]    + (1.0f - beta1) * smart_grad;
    const float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * smart_grad * smart_grad;
    exp_avg[idx]    = m;
    exp_avg_sq[idx] = v;

    const float update = (m * bc1) / (sqrtf(v * bc2) + eps);
    param[idx] = static_cast<ParamT>(p - lr * (update + wd * p));
}

// =========================================================================
//  Backward (6): Bilevel precompute per-timestep
//  Reproduces the forward projections needed for the adjoint scan.
// =========================================================================

__device__ __forceinline__ void sg2_bilevel_precompute_timestep(
    const float* __restrict__ x_sorted_t,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    float* __restrict__ pre_x_val_t,
    float* __restrict__ pre_z_val_t,
    float* __restrict__ pre_dt_val_t,
    float* __restrict__ pre_B_val_t,
    float* __restrict__ pre_C_val_t,
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
        x_branch[j]      = x_val;
        pre_x_val_t[j]   = x_val;
        pre_z_val_t[j]   = z_val;
    }

    // dt projection + softplus
    #pragma unroll 4
    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        #pragma unroll 4
        for (int k = 0; k < d_inner; k++) {
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        }
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

// ═════════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former csrc/algorithms/moe_adam.h.
//
//  Multi-tensor batched AdamW used for both standard parameter groups and
//  Mixture-of-Experts active-set updates. The MoE variant compacts the
//  active subset of expert parameters into a dense buffer, runs the same
//  per-element Adam step over that buffer, then scatters results back.
//
//  The per-element math is identical to adamw.h::adamw_step; this function
//  re-exports it under the `moe_adam_step` name to keep the launcher glue
//  symmetric across the optimizers.
// ═════════════════════════════════════════════════════════════════════════

template <typename ParamT, typename GradT>
__device__ __forceinline__ void moe_adam_step(
    ParamT* __restrict__ param,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const GradT* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float bc1,
    const float bc2,
    const int idx
) {
    adamw_step(param, exp_avg, exp_avg_sq, grad,
               lr, beta1, beta2, eps, wd, bc1, bc2, idx);
}

}} // namespace sg::algorithms
