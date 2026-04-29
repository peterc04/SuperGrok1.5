/*
 * SuperGrok v2 — CDNA4 FP6 (E3M2) Optimizer State Kernels (gfx950, MI350X)
 *
 * Split out from the former cdna4_kernels_gfx950.hip.cpp monolith.
 * Contains the 4 FP6 state kernels and their host launchers:
 *   1. cdna4_fp6_state_pack_kernel    — FP32 → FP6 (4 vals → 3 bytes)
 *   2. cdna4_fp6_state_unpack_kernel  — FP6 → FP32
 *   3. cdna4_fp6_adam_step_kernel     — Fused Adam with FP6 state
 *   4. cdna4_fp6_lamb_step_kernel     — Fused LAMB with FP6 state
 *
 * Shared FP4/FP6 helpers live in csrc/common/fp4_helpers.hip.h.
 * Math is unchanged from the monolith.
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "platform.h"
#include "../../common/fp4_helpers.hip.h"

namespace sg { namespace gfx950 {

// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: FP6 State Pack
//
//  Pack FP32 optimizer state (exp_avg, exp_avg_sq) into FP6 (E3M2).
//  4 FP6 values packed into 3 bytes for 5.33x memory reduction.
//  Processes both first and second moment in a single pass.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp6_state_pack_kernel(
    const float*    __restrict__ exp_avg,         // [N]
    const float*    __restrict__ exp_avg_sq,      // [N]
    uint8_t*        __restrict__ exp_avg_fp6,     // [N * 3 / 4] packed
    uint8_t*        __restrict__ exp_avg_sq_fp6,  // [N * 3 / 4] packed
    const float*    __restrict__ state_scale_avg, // [1] or [num_blocks]
    const float*    __restrict__ state_scale_sq,  // [1] or [num_blocks]
    int             N
) {
    // Process 4 elements at a time (4 FP6 values → 3 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = (N + 3) / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    float scale_avg = state_scale_avg[0];
    float scale_sq  = state_scale_sq[0];

    // Load 4 FP32 values for exp_avg
    float avg_vals[4];
    float sq_vals[4];
    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        avg_vals[i] = (idx < N) ? exp_avg[idx] * scale_avg : 0.0f;
        sq_vals[i]  = (idx < N) ? exp_avg_sq[idx] * scale_sq : 0.0f;
    }

    // Pack to FP6
    uint8_t* avg_out = exp_avg_fp6 + (size_t)group_idx * 3;
    uint8_t* sq_out  = exp_avg_sq_fp6 + (size_t)group_idx * 3;

    fp6_pack4(avg_vals, avg_out);
    fp6_pack4(sq_vals, sq_out);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 6: FP6 State Unpack
//
//  Unpack FP6 optimizer state back to FP32 for computation.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp6_state_unpack_kernel(
    const uint8_t*  __restrict__ exp_avg_fp6,     // [N * 3 / 4] packed
    const uint8_t*  __restrict__ exp_avg_sq_fp6,  // [N * 3 / 4] packed
    float*          __restrict__ exp_avg,         // [N]
    float*          __restrict__ exp_avg_sq,      // [N]
    const float*    __restrict__ state_scale_avg, // [1]
    const float*    __restrict__ state_scale_sq,  // [1]
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = (N + 3) / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    float inv_scale_avg = (state_scale_avg[0] != 0.0f) ? (1.0f / state_scale_avg[0]) : 1.0f;
    float inv_scale_sq  = (state_scale_sq[0] != 0.0f) ? (1.0f / state_scale_sq[0]) : 1.0f;

    const uint8_t* avg_in = exp_avg_fp6 + (size_t)group_idx * 3;
    const uint8_t* sq_in  = exp_avg_sq_fp6 + (size_t)group_idx * 3;

    float avg_vals[4], sq_vals[4];
    fp6_unpack4(avg_in, avg_vals);
    fp6_unpack4(sq_in, sq_vals);

    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        if (idx < N) {
            exp_avg[idx]    = avg_vals[i] * inv_scale_avg;
            exp_avg_sq[idx] = sq_vals[i] * inv_scale_sq;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 7: FP6 Adam Step (Fused)
//
//  Full Adam optimizer step with FP6 state:
//    1. Unpack exp_avg, exp_avg_sq from FP6
//    2. Compute Adam update: m = beta1*m + (1-beta1)*g
//                            v = beta2*v + (1-beta2)*g^2
//                            param -= lr * m_hat / (sqrt(v_hat) + eps)
//    3. Repack updated m, v to FP6
//
//  Fused to avoid full unpack→compute→repack round-trip through memory.
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp6_adam_step_kernel(
    float*          __restrict__ param,           // [N]
    const float*    __restrict__ grad,            // [N]
    uint8_t*        __restrict__ exp_avg_fp6,     // [N * 3 / 4]
    uint8_t*        __restrict__ exp_avg_sq_fp6,  // [N * 3 / 4]
    float*          __restrict__ state_scale_avg, // [1] — updated in-place
    float*          __restrict__ state_scale_sq,  // [1] — updated in-place
    float           beta1,
    float           beta2,
    float           lr,
    float           eps,
    float           weight_decay,
    float           bc1,                          // 1 / (1 - beta1^t)
    float           bc2,                          // 1 / (1 - beta2^t)
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = (N + 3) / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    float inv_scale_avg = (state_scale_avg[0] != 0.0f) ? (1.0f / state_scale_avg[0]) : 1.0f;
    float inv_scale_sq  = (state_scale_sq[0] != 0.0f) ? (1.0f / state_scale_sq[0]) : 1.0f;

    // Unpack current state
    uint8_t* avg_ptr = exp_avg_fp6 + (size_t)group_idx * 3;
    uint8_t* sq_ptr  = exp_avg_sq_fp6 + (size_t)group_idx * 3;

    float m_vals[4], v_vals[4];
    fp6_unpack4(avg_ptr, m_vals);
    fp6_unpack4(sq_ptr, v_vals);

    // Compute Adam update for each element in the group
    float new_m[4], new_v[4];
    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        if (idx < N) {
            float p = param[idx];
            float g = grad[idx];

            // Decoupled weight decay
            if (weight_decay != 0.0f) {
                p -= lr * weight_decay * p;
            }

            // Moment updates (in FP32 after dequant)
            float m = m_vals[i] * inv_scale_avg;
            float v = v_vals[i] * inv_scale_sq;

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * g * g;

            // Bias-corrected estimates
            float m_hat = m * bc1;
            float v_hat = v * bc2;

            // Parameter update
            p -= lr * m_hat / (sqrtf(v_hat) + eps);
            param[idx] = p;

            // Prepare for repacking (scale to FP6 range)
            new_m[i] = m;
            new_v[i] = v;
        } else {
            new_m[i] = 0.0f;
            new_v[i] = 0.0f;
        }
    }

    // Find new scale factors for this group (local contribution)
    // The global scale will be updated via atomicMax across all groups
    float local_max_m = 0.0f, local_max_v = 0.0f;
    for (int i = 0; i < 4; i++) {
        local_max_m = fmaxf(local_max_m, fabsf(new_m[i]));
        local_max_v = fmaxf(local_max_v, fabsf(new_v[i]));
    }

    // For simplicity, reuse existing scale. Full scale update done periodically.
    float scale_avg = state_scale_avg[0];
    float scale_sq  = state_scale_sq[0];

    // Scale and repack
    float scaled_m[4], scaled_v[4];
    for (int i = 0; i < 4; i++) {
        scaled_m[i] = (scale_avg != 0.0f) ? new_m[i] * scale_avg : new_m[i];
        scaled_v[i] = (scale_sq != 0.0f) ? new_v[i] * scale_sq : new_v[i];
    }

    fp6_pack4(scaled_m, avg_ptr);
    fp6_pack4(scaled_v, sq_ptr);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 8: FP6 LAMB Step (Fused)
//
//  LAMB (Layer-wise Adaptive Moments) with FP6 state.
//  Same fused unpack-compute-repack pattern as Adam, but with
//  layer-wise trust ratio: ratio = ||param|| / ||update||
//  param -= lr * ratio * (update + wd * param)
// ═══════════════════════════════════════════════════════════════════════

extern "C"
__launch_bounds__(256, 8)
GROK_WAVES_PER_EU(1, 8)
GROK_FLAT_WORK_GROUP_SIZE(256, 256)
__global__ void
cdna4_fp6_lamb_step_kernel(
    float*          __restrict__ param,
    const float*    __restrict__ grad,
    uint8_t*        __restrict__ exp_avg_fp6,
    uint8_t*        __restrict__ exp_avg_sq_fp6,
    float*          __restrict__ state_scale_avg,
    float*          __restrict__ state_scale_sq,
    float*          __restrict__ param_norm_out,  // [1] partial sum for param norm
    float*          __restrict__ update_norm_out, // [1] partial sum for update norm
    float           beta1,
    float           beta2,
    float           lr,
    float           eps,
    float           weight_decay,
    float           bc1,
    float           bc2,
    float           trust_ratio,                  // precomputed ||param|| / ||adam_update||
    int             N
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = (N + 3) / 4;

    if (group_idx >= num_groups) return;

    int base = group_idx * 4;
    float inv_scale_avg = (state_scale_avg[0] != 0.0f) ? (1.0f / state_scale_avg[0]) : 1.0f;
    float inv_scale_sq  = (state_scale_sq[0] != 0.0f) ? (1.0f / state_scale_sq[0]) : 1.0f;

    uint8_t* avg_ptr = exp_avg_fp6 + (size_t)group_idx * 3;
    uint8_t* sq_ptr  = exp_avg_sq_fp6 + (size_t)group_idx * 3;

    float m_vals[4], v_vals[4];
    fp6_unpack4(avg_ptr, m_vals);
    fp6_unpack4(sq_ptr, v_vals);

    float new_m[4], new_v[4];
    float local_param_sq = 0.0f;
    float local_update_sq = 0.0f;

    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        if (idx < N) {
            float p = param[idx];
            float g = grad[idx];

            float m = m_vals[i] * inv_scale_avg;
            float v = v_vals[i] * inv_scale_sq;

            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * g * g;

            float m_hat = m * bc1;
            float v_hat = v * bc2;

            // LAMB update = adam_update + weight_decay * param
            float adam_update = m_hat / (sqrtf(v_hat) + eps);
            float full_update = adam_update + weight_decay * p;

            local_param_sq += p * p;
            local_update_sq += full_update * full_update;

            // Apply trust ratio scaling
            p -= lr * trust_ratio * full_update;
            param[idx] = p;

            new_m[i] = m;
            new_v[i] = v;
        } else {
            new_m[i] = 0.0f;
            new_v[i] = 0.0f;
        }
    }

    // Contribute to partial norms for next step's trust ratio computation
    if (local_param_sq > 0.0f) {
        atomicAdd(param_norm_out, local_param_sq);
    }
    if (local_update_sq > 0.0f) {
        atomicAdd(update_norm_out, local_update_sq);
    }

    // Repack state to FP6
    float scale_avg = state_scale_avg[0];
    float scale_sq  = state_scale_sq[0];
    float scaled_m[4], scaled_v[4];
    for (int i = 0; i < 4; i++) {
        scaled_m[i] = (scale_avg != 0.0f) ? new_m[i] * scale_avg : new_m[i];
        scaled_v[i] = (scale_sq != 0.0f) ? new_v[i] * scale_sq : new_v[i];
    }

    fp6_pack4(scaled_m, avg_ptr);
    fp6_pack4(scaled_v, sq_ptr);
}


// ═══════════════════════════════════════════════════════════════════════
//  Host Launchers for FP6 State Kernels
// ═══════════════════════════════════════════════════════════════════════

void cdna4_fp6_state_pack(
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    int N
) {
    int num_groups = (N + 3) / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_fp6_state_pack_kernel<<<grid, block>>>(
        exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
        exp_avg_fp6.data_ptr<uint8_t>(), exp_avg_sq_fp6.data_ptr<uint8_t>(),
        state_scale_avg.data_ptr<float>(), state_scale_sq.data_ptr<float>(),
        N
    );
}

void cdna4_fp6_state_unpack(
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    int N
) {
    int num_groups = (N + 3) / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_fp6_state_unpack_kernel<<<grid, block>>>(
        exp_avg_fp6.data_ptr<uint8_t>(), exp_avg_sq_fp6.data_ptr<uint8_t>(),
        exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
        state_scale_avg.data_ptr<float>(), state_scale_sq.data_ptr<float>(),
        N
    );
}

void cdna4_fp6_adam_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    float beta1, float beta2, float lr, float eps,
    float weight_decay, float bc1, float bc2, int N
) {
    int num_groups = (N + 3) / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_fp6_adam_step_kernel<<<grid, block>>>(
        param.data_ptr<float>(), grad.data_ptr<float>(),
        exp_avg_fp6.data_ptr<uint8_t>(), exp_avg_sq_fp6.data_ptr<uint8_t>(),
        state_scale_avg.data_ptr<float>(), state_scale_sq.data_ptr<float>(),
        beta1, beta2, lr, eps, weight_decay, bc1, bc2, N
    );
}

void cdna4_fp6_lamb_step(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor exp_avg_fp6, torch::Tensor exp_avg_sq_fp6,
    torch::Tensor state_scale_avg, torch::Tensor state_scale_sq,
    torch::Tensor param_norm_out, torch::Tensor update_norm_out,
    float beta1, float beta2, float lr, float eps,
    float weight_decay, float bc1, float bc2, float trust_ratio, int N
) {
    int num_groups = (N + 3) / 4;
    dim3 grid((num_groups + 255) / 256);
    dim3 block(256);
    cdna4_fp6_lamb_step_kernel<<<grid, block>>>(
        param.data_ptr<float>(), grad.data_ptr<float>(),
        exp_avg_fp6.data_ptr<uint8_t>(), exp_avg_sq_fp6.data_ptr<uint8_t>(),
        state_scale_avg.data_ptr<float>(), state_scale_sq.data_ptr<float>(),
        param_norm_out.data_ptr<float>(), update_norm_out.data_ptr<float>(),
        beta1, beta2, lr, eps, weight_decay, bc1, bc2, trust_ratio, N
    );
}

} } // namespace sg::gfx950
