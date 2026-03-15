/*
 * SuperGrok v2 — Ampere-Optimized Forward Kernels (sm_80+)
 *
 * Contains two real __global__ cp.async scan kernels:
 *   - mamba3_scan_batched_cpasync_kernel: sequential batched scan with
 *     double-buffered __pipeline_memcpy_async prefetch of scan input data.
 *     While processing timestep t, timestep t+1 is prefetched from global
 *     memory to shared memory via cp.async.
 *   - mamba3_scan_combined_cpasync_kernel: sequential combined (fwd+bwd)
 *     scan with the same cp.async double-buffered prefetch pattern.
 *
 * These kernels are launched by ampere_batched_scan_and_fused_elem() in
 * this file, which is called from the Ampere batched step launcher.
 *
 * Batched step uses the refactored 3-phase pipeline:
 *   1. batched_step_setup_and_sort() — shared setup
 *   2. generic_batched_precompute() — FP32 precompute (TF32 math mode)
 *   3. ampere_batched_scan_and_fused_elem() — cp.async scan + cp.async
 *      fused_elem (uses fused_elem_step_cpasync_kernel from
 *      supergrok2_fused_elem_sm80.cu)
 *
 * Single-param step delegates to generic with TF32 mode set.
 *
 * Dispatch: ops.cpp calls these on sm_80+ GPUs.
 * Fallback: On sm_70/sm_75, the generic launchers are called instead.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"
#include "types.h"
#include "dispatch.h"
#include "ops.h"

// cp.async intrinsics (sm_80+): asynchronous global→shared memory copy
// These are compiled conditionally and only used on Ampere+
#if GROK_CUDA
#include <cuda_pipeline.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations of generic launchers (defined in generic/*.cu)
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);

void launch_mamba3_peer_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts);


// ═══════════════════════════════════════════════════════════════════════
//  Ampere cp.async Batched Sequential Scan Kernel
//
//  Double-buffered scan: while processing timestep t using shared memory
//  buffer A, asynchronously prefetch timestep t+1 data into buffer B.
//  On the next iteration, swap buffers.
//
//  This overlaps global memory latency (~400 cycles on A100) with the
//  scan compute (~800 FMAs per timestep), effectively hiding ~50% of
//  memory stalls.
//
//  Grid:  (num_params, 1, 1)  — one block per parameter
//  Block: (d_inner, 1, 1)     — one thread per scan dimension
//
//  Shared memory layout (double-buffered):
//    Buffer 0: [d_model] input data
//    Buffer 1: [d_model] input data (prefetch target)
//    Projection weights: in_proj, dt_proj, B_proj, C_proj (cached once)
//
//  The scan math is IDENTICAL to the generic sequential kernel — only
//  the memory access pattern changes.
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_scan_batched_cpasync_kernel(
    const float* __restrict__ x_sorted_packed,    // [total_N, d_model]
    float* __restrict__ scan_output_packed,        // [total_N, d_inner]
    const float* __restrict__ initial_states,      // [num_params, d_inner, d_state]
    float* __restrict__ final_states,              // [num_params, d_inner, d_state]
    const int* __restrict__ offsets,               // [num_params + 1]
    const int* __restrict__ reverse_flags,         // [num_params]
    const float* __restrict__ in_proj_W,           // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,           // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,           // [d_inner]
    const float* __restrict__ B_proj_W,            // [d_state, d_inner]
    const float* __restrict__ C_proj_W,            // [d_state, d_inner]
    const float* __restrict__ A_log,               // [d_inner, d_state]
    const float* __restrict__ D_param,             // [d_inner]
    const float* __restrict__ rope_freq,           // [d_inner, d_state/2]
    const int d_model,
    const int d_inner,
    const int d_state
) {
    const int param_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    const int reverse = reverse_flags[param_idx];

    // ── Shared memory layout ──────────────────────────────────────────
    // Double-buffer for input: 2 × d_model floats
    // Projection weights (cached once): in_proj, dt_proj, dt_bias, B_proj, C_proj
    extern __shared__ float smem[];
    float* s_input_buf0 = smem;                              // [d_model]
    float* s_input_buf1 = s_input_buf0 + d_model;            // [d_model]
    float* s_x_branch = s_input_buf1 + d_model;              // [d_inner]
    float* s_in_proj_W = s_x_branch + d_inner;               // [2*d_inner*d_model]
    float* s_dt_proj_W = s_in_proj_W + 2*d_inner*d_model;    // [d_inner*d_inner]
    float* s_dt_proj_b = s_dt_proj_W + d_inner*d_inner;      // [d_inner]
    float* s_B_proj_W = s_dt_proj_b + d_inner;               // [d_state*d_inner]
    float* s_C_proj_W = s_B_proj_W + d_state*d_inner;        // [d_state*d_inner]

    // Cooperatively load projection weights into shared memory
    for (int i = tid; i < 2*d_inner*d_model; i += d_inner)
        s_in_proj_W[i] = in_proj_W[i];
    for (int i = tid; i < d_inner*d_inner; i += d_inner)
        s_dt_proj_W[i] = dt_proj_W[i];
    for (int i = tid; i < d_inner; i += d_inner)
        s_dt_proj_b[i] = dt_proj_b[i];
    for (int i = tid; i < d_state*d_inner; i += d_inner)
        s_B_proj_W[i] = B_proj_W[i];
    for (int i = tid; i < d_state*d_inner; i += d_inner)
        s_C_proj_W[i] = C_proj_W[i];
    __syncthreads();

    // ── Load state into registers ─────────────────────────────────────
    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    const float* my_init = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];
    float D_val = D_param[tid];

    const float* my_x = x_sorted_packed + start * d_model;
    float* my_out = scan_output_packed + start * d_inner;

    // ── Prefetch first timestep using cp.async ────────────────────────
    int cur_buf = 0;
    float* buf[2] = {s_input_buf0, s_input_buf1};

    if (N > 0) {
        int first_i = reverse ? (N - 1) : 0;
        // cp.async: asynchronous global → shared memory copy
        // Each thread copies its share of d_model floats
        for (int d = tid; d < d_model; d += d_inner) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_memcpy_async(
                &buf[cur_buf][d],
                &my_x[first_i * d_model + d],
                sizeof(float));
#else
            buf[cur_buf][d] = my_x[first_i * d_model + d];
#endif
        }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_commit();
#endif
    }

    // ── Main scan loop with double-buffered prefetch ──────────────────
    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;
        int next_step = step + 1;
        int next_buf = 1 - cur_buf;

        // Start prefetch of next timestep into alternate buffer
        if (next_step < N) {
            int next_i = reverse ? (N - 1 - next_step) : next_step;
            for (int d = tid; d < d_model; d += d_inner) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                __pipeline_memcpy_async(
                    &buf[next_buf][d],
                    &my_x[next_i * d_model + d],
                    sizeof(float));
#else
                buf[next_buf][d] = my_x[next_i * d_model + d];
#endif
            }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_commit();
#endif
        }

        // Wait for current buffer to be ready
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_wait_prior(1);  // wait for all but the most recent commit
#endif
        __syncthreads();

        // ── Input projection using prefetched data ────────────────────
        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = buf[cur_buf][d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        // ── dt projection (cross-thread via shared memory) ────────────
        float dt_raw = s_dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += s_dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));

        // ── State update with trapezoidal discretization + RoPE ───────
        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += s_B_proj_W[s * d_inner + j] * s_x_branch[j];
            float B_bar = dt_val * B_val;

            int pair_idx = s / 2;
            float cos_p, sin_p;
            FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0)
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            else
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // ── Output: y = C @ h, gated by SiLU(z) + D skip ────────────
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                C_val += s_C_proj_W[s * d_inner + j] * s_x_branch[j];
            y_val += h[s] * C_val;
        }
        float silu_z = z_val / (1.0f + expf(-z_val));
        y_val = y_val * silu_z + D_val * x_val;

        my_out[i * d_inner + tid] = y_val;

        // Swap buffers for next iteration
        cur_buf = next_buf;
        __syncthreads();
    }

    // ── Write final state ─────────────────────────────────────────────
    float* my_final = final_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        my_final[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Ampere cp.async Combined Forward+Backward Scan Kernel
//
//  Grid = 2 * num_params: first half forward, second half backward.
//  Same cp.async double-buffering as the batched kernel above.
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_scan_combined_cpasync_kernel(
    const float* __restrict__ x_sorted_packed,
    float* __restrict__ fwd_scan_output,
    float* __restrict__ bwd_scan_output,
    const float* __restrict__ fwd_initial_states,
    const float* __restrict__ bwd_initial_states,
    float* __restrict__ fwd_final_states,
    float* __restrict__ bwd_final_states,
    const int* __restrict__ offsets,
    const float* __restrict__ fwd_in_proj_W,
    const float* __restrict__ fwd_dt_proj_W,
    const float* __restrict__ fwd_dt_proj_b,
    const float* __restrict__ fwd_B_proj_W,
    const float* __restrict__ fwd_C_proj_W,
    const float* __restrict__ fwd_A_log,
    const float* __restrict__ fwd_D_param,
    const float* __restrict__ fwd_rope_freq,
    const float* __restrict__ bwd_in_proj_W,
    const float* __restrict__ bwd_dt_proj_W,
    const float* __restrict__ bwd_dt_proj_b,
    const float* __restrict__ bwd_B_proj_W,
    const float* __restrict__ bwd_C_proj_W,
    const float* __restrict__ bwd_A_log,
    const float* __restrict__ bwd_D_param,
    const float* __restrict__ bwd_rope_freq,
    const int num_params,
    const int d_model,
    const int d_inner,
    const int d_state
) {
    // Determine if this block does forward or backward
    const int block_idx = blockIdx.x;
    const bool is_backward = (block_idx >= num_params);
    const int param_idx = is_backward ? (block_idx - num_params) : block_idx;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    const int reverse = is_backward ? 1 : 0;

    // Select direction-specific weights
    const float* in_proj_W = is_backward ? bwd_in_proj_W : fwd_in_proj_W;
    const float* dt_proj_W_sel = is_backward ? bwd_dt_proj_W : fwd_dt_proj_W;
    const float* dt_proj_b_sel = is_backward ? bwd_dt_proj_b : fwd_dt_proj_b;
    const float* B_proj_W_sel = is_backward ? bwd_B_proj_W : fwd_B_proj_W;
    const float* C_proj_W_sel = is_backward ? bwd_C_proj_W : fwd_C_proj_W;
    const float* A_log_sel = is_backward ? bwd_A_log : fwd_A_log;
    const float* D_sel = is_backward ? bwd_D_param : fwd_D_param;
    const float* rope_sel = is_backward ? bwd_rope_freq : fwd_rope_freq;
    const float* init_states = is_backward ? bwd_initial_states : fwd_initial_states;
    float* fin_states = is_backward ? bwd_final_states : fwd_final_states;
    float* scan_out = is_backward ? bwd_scan_output : fwd_scan_output;

    // Shared memory: double-buffer + x_branch + cached projection weights
    extern __shared__ float smem[];
    float* s_input_buf0 = smem;
    float* s_input_buf1 = s_input_buf0 + d_model;
    float* s_x_branch = s_input_buf1 + d_model;
    float* s_in_proj = s_x_branch + d_inner;
    float* s_dt_proj = s_in_proj + 2*d_inner*d_model;
    float* s_dt_bias = s_dt_proj + d_inner*d_inner;
    float* s_B_proj = s_dt_bias + d_inner;
    float* s_C_proj = s_B_proj + d_state*d_inner;

    // Load weights
    for (int i = tid; i < 2*d_inner*d_model; i += d_inner)
        s_in_proj[i] = in_proj_W[i];
    for (int i = tid; i < d_inner*d_inner; i += d_inner)
        s_dt_proj[i] = dt_proj_W_sel[i];
    for (int i = tid; i < d_inner; i += d_inner)
        s_dt_bias[i] = dt_proj_b_sel[i];
    for (int i = tid; i < d_state*d_inner; i += d_inner)
        s_B_proj[i] = B_proj_W_sel[i];
    for (int i = tid; i < d_state*d_inner; i += d_inner)
        s_C_proj[i] = C_proj_W_sel[i];
    __syncthreads();

    // Load state
    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    const float* my_init = init_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++) A[s] = -expf(A_log_sel[tid * d_state + s]);
    for (int p = 0; p < half_d_state; p++) freq[p] = rope_sel[tid * half_d_state + p];
    float D_val = D_sel[tid];

    const float* my_x = x_sorted_packed + start * d_model;
    float* my_out = scan_out + start * d_inner;

    // Double-buffer prefetch
    float* buf[2] = {s_input_buf0, s_input_buf1};
    int cur_buf = 0;

    if (N > 0) {
        int first_i = reverse ? (N - 1) : 0;
        for (int d = tid; d < d_model; d += d_inner) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_memcpy_async(&buf[cur_buf][d], &my_x[first_i * d_model + d], sizeof(float));
#else
            buf[cur_buf][d] = my_x[first_i * d_model + d];
#endif
        }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_commit();
#endif
    }

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;
        int next_buf = 1 - cur_buf;

        if (step + 1 < N) {
            int next_i = reverse ? (N - 2 - step) : (step + 1);
            for (int d = tid; d < d_model; d += d_inner) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                __pipeline_memcpy_async(&buf[next_buf][d], &my_x[next_i * d_model + d], sizeof(float));
#else
                buf[next_buf][d] = my_x[next_i * d_model + d];
#endif
            }
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_commit();
            __pipeline_wait_prior(1);
#endif
        } else {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_wait_prior(0);
#endif
        }
        __syncthreads();

        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = buf[cur_buf][d];
            x_val += s_in_proj[tid * d_model + d] * inp;
            z_val += s_in_proj[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = s_dt_bias[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += s_dt_proj[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];
        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) B_val += s_B_proj[s * d_inner + j] * s_x_branch[j];
            float B_bar = dt_val * B_val;
            int pair_idx = s / 2;
            float cos_p, sin_p;
            FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot = (s % 2 == 0)
                ? h_snap[s] * cos_p - h_snap[s+1] * sin_p
                : h_snap[s] * cos_p + h_snap[s-1] * sin_p;
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++) C_val += s_C_proj[s * d_inner + j] * s_x_branch[j];
            y_val += h[s] * C_val;
        }
        float silu_z = z_val / (1.0f + expf(-z_val));
        y_val = y_val * silu_z + D_val * x_val;

        my_out[i * d_inner + tid] = y_val;
        cur_buf = next_buf;
        __syncthreads();
    }

    float* my_final = fin_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        my_final[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Forward declaration: cp.async fused_elem kernel (defined in
//  supergrok2_fused_elem_sm80.cu)
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_elem_step_cpasync_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ sharpness,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ mu,
    float* __restrict__ gru_state,
    const float* __restrict__ fwd_scan_out,
    const float* __restrict__ bwd_scan_out,
    const float* __restrict__ out_proj_fwd_W,
    const float* __restrict__ out_proj_bwd_W,
    const float* __restrict__ gru_Wz, const float* __restrict__ gru_bz,
    const float* __restrict__ gru_Wr, const float* __restrict__ gru_br,
    const float* __restrict__ gru_Wh, const float* __restrict__ gru_bh,
    const float* __restrict__ peer_query_Ws,
    const float* __restrict__ prod_keys_A,
    const float* __restrict__ prod_keys_B,
    const float* __restrict__ expert_W1, const float* __restrict__ expert_b1,
    const float* __restrict__ expert_W2, const float* __restrict__ expert_b2,
    const float rescale, const float alpha, const float lamb_eff,
    const float beta1, const float beta2,
    const float lr, const float wd_eff, const float eps,
    const float bc1, const float bc2,
    int* __restrict__ expert_counts,
    const int N, const int d_model, const int d_inner,
    const int gru_hidden, const int num_heads, const int pk_dim,
    const int expert_hidden, const int num_experts);


// ═══════════════════════════════════════════════════════════════════════
//  Ampere Batched Scan + Fused Elem (cp.async variants)
//
//  Same structure as batched_step_scan_and_fused_elem, but:
//    - Scan: launches cp.async scan kernels instead of generic ones
//    - Fused elem: launches fused_elem_step_cpasync_kernel
//
//  For large N (>= PSCAN_THRESHOLD): uses mamba3_scan_batched_cpasync_kernel
//    (separate fwd/bwd launches, sequential per-param with cp.async prefetch)
//  For small N (< PSCAN_THRESHOLD): uses mamba3_scan_combined_cpasync_kernel
//    (combined fwd+bwd launch, sequential with cp.async prefetch)
// ═══════════════════════════════════════════════════════════════════════

void ampere_batched_scan_and_fused_elem(
    BatchedScanCtx& ctx,
    torch::Tensor fwd_pre_x, torch::Tensor fwd_pre_z, torch::Tensor fwd_pre_dt,
    torch::Tensor fwd_pre_B, torch::Tensor fwd_pre_C,
    torch::Tensor bwd_pre_x, torch::Tensor bwd_pre_z, torch::Tensor bwd_pre_dt,
    torch::Tensor bwd_pre_B, torch::Tensor bwd_pre_C,
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    if (ctx.total_N == 0) return;

    auto dev = ctx.x_sorted_packed.device();
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // cp.async scan shared memory: double-buffer + x_branch + cached projection weights
    int scan_smem = (2*d_model + d_inner + 2*d_inner*d_model + d_inner*d_inner
                     + d_inner + 2*d_state*d_inner) * (int)sizeof(float);

    if (ctx.max_N >= PSCAN_THRESHOLD) {
        // ===== cp.async BATCHED SCAN for large N =====
        // Two separate launches (fwd, bwd) using mamba3_scan_batched_cpasync_kernel
        auto rev_fwd = torch::zeros({ctx.num_params}, int_opts);
        auto rev_bwd = torch::ones({ctx.num_params}, int_opts);

        // Forward scan with cp.async prefetch
        mamba3_scan_batched_cpasync_kernel<<<ctx.num_params, d_inner, scan_smem>>>(
            ctx.x_sorted_packed.data_ptr<float>(),
            ctx.fwd_scan_packed.data_ptr<float>(),
            ctx.initial_fwd.data_ptr<float>(),
            ctx.final_fwd.data_ptr<float>(),
            ctx.offsets_t.data_ptr<int>(),
            rev_fwd.data_ptr<int>(),
            mamba_fwd_in_proj.data_ptr<float>(),
            mamba_fwd_dt_W.data_ptr<float>(),
            mamba_fwd_dt_b.data_ptr<float>(),
            mamba_fwd_B_proj.data_ptr<float>(),
            mamba_fwd_C_proj.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            d_model, d_inner, d_state
        );

        // Backward scan (reversed) with cp.async prefetch
        mamba3_scan_batched_cpasync_kernel<<<ctx.num_params, d_inner, scan_smem>>>(
            ctx.x_sorted_packed.data_ptr<float>(),
            ctx.bwd_scan_packed.data_ptr<float>(),
            ctx.initial_bwd.data_ptr<float>(),
            ctx.final_bwd.data_ptr<float>(),
            ctx.offsets_t.data_ptr<int>(),
            rev_bwd.data_ptr<int>(),
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            d_model, d_inner, d_state
        );
    } else {
        // ===== cp.async COMBINED SCAN for small N =====
        mamba3_scan_combined_cpasync_kernel<<<2 * ctx.num_params, d_inner, scan_smem>>>(
            ctx.x_sorted_packed.data_ptr<float>(),
            ctx.fwd_scan_packed.data_ptr<float>(),
            ctx.bwd_scan_packed.data_ptr<float>(),
            ctx.initial_fwd.data_ptr<float>(),
            ctx.initial_bwd.data_ptr<float>(),
            ctx.final_fwd.data_ptr<float>(),
            ctx.final_bwd.data_ptr<float>(),
            ctx.offsets_t.data_ptr<int>(),
            mamba_fwd_in_proj.data_ptr<float>(),
            mamba_fwd_dt_W.data_ptr<float>(),
            mamba_fwd_dt_b.data_ptr<float>(),
            mamba_fwd_B_proj.data_ptr<float>(),
            mamba_fwd_C_proj.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            ctx.num_params, d_model, d_inner, d_state
        );
    }

    // Unsort scan outputs + copy final states
    int gru_input_dim_val = 2 + 2 * d_model;
    int gru_row_len = gru_input_dim_val + gru_hidden;
    int smem_bytes = (2 * d_model * d_inner
                    + 3 * gru_hidden * gru_row_len
                    + 3 * gru_hidden
                    + 3 * num_experts * expert_hidden + num_experts) * sizeof(float);

    std::vector<torch::Tensor> fwd_unsorted_list(ctx.num_params);
    std::vector<torch::Tensor> bwd_unsorted_list(ctx.num_params);
    for (int p = 0; p < ctx.num_params; p++) {
        int N = ctx.N_vec[p];
        if (N == 0) continue;
        int off = ctx.seg_offsets_cpu[p];

        mamba_fwd_states[p].copy_(ctx.final_fwd[p]);
        mamba_bwd_states[p].copy_(ctx.final_bwd[p]);

        auto fwd_slice = ctx.fwd_scan_packed.narrow(0, off, N);
        auto bwd_slice = ctx.bwd_scan_packed.narrow(0, off, N);
        fwd_unsorted_list[p] = fwd_slice.index_select(0, ctx.unsort_idx_list[p]);
        bwd_unsorted_list[p] = bwd_slice.index_select(0, ctx.unsort_idx_list[p]);
    }

    // Launch cp.async fused_elem_step kernels on persistent stream pool
    constexpr int NUM_STREAMS = 4;
    static GpuStream_t streams[NUM_STREAMS] = {};
    static bool streams_initialized = false;
    if (!streams_initialized) {
        for (int s = 0; s < NUM_STREAMS; s++)
            gpuStreamCreate(&streams[s]);
        streams_initialized = true;
    }

    for (int p = 0; p < ctx.num_params; p++) {
        int N = ctx.N_vec[p];
        if (N == 0) continue;

        GpuStream_t stream = streams[p % NUM_STREAMS];
        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[p].scalar_type(), "fused_elem_step_cpasync", ([&] {
            fused_elem_step_cpasync_kernel<scalar_t><<<grid, SG2M_BLOCK, smem_bytes, stream>>>(
                params[p].data_ptr<scalar_t>(),
                grads[p].data_ptr<scalar_t>(),
                sharpness_list[p].data_ptr<scalar_t>(),
                exp_avgs[p].data_ptr<float>(),
                exp_avg_sqs[p].data_ptr<float>(),
                mus[p].data_ptr<float>(),
                gru_states[p].data_ptr<float>(),
                fwd_unsorted_list[p].data_ptr<float>(),
                bwd_unsorted_list[p].data_ptr<float>(),
                mamba_fwd_out_proj.data_ptr<float>(),
                mamba_bwd_out_proj.data_ptr<float>(),
                gru_Wz.data_ptr<float>(), gru_bz.data_ptr<float>(),
                gru_Wr.data_ptr<float>(), gru_br.data_ptr<float>(),
                gru_Wh.data_ptr<float>(), gru_bh.data_ptr<float>(),
                peer_query_Ws.data_ptr<float>(),
                prod_keys_A.data_ptr<float>(),
                prod_keys_B.data_ptr<float>(),
                expert_W1.data_ptr<float>(),
                expert_b1.data_ptr<float>(),
                expert_W2.data_ptr<float>(),
                expert_b2.data_ptr<float>(),
                rescale, alpha_mus[p], lamb_effs[p],
                beta1s[p], beta2, lr, wd_eff, eps, bc1s[p], bc2s[p],
                expert_counts.data_ptr<int>(),
                N, d_model, d_inner, gru_hidden,
                num_heads, pk_dim, expert_hidden, num_experts
            );
        }));
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        gpuStreamSynchronize(streams[s]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Ampere Forward: Per-Parameter Step — delegates to generic
//
//  The generic single-param step uses only custom CUDA kernels
//  (input_proj_sort_kernel, mamba3_scan_combined_kernel, fused_elem_step_kernel),
//  not cuBLAS/torch::mm, so TF32 mode set here has no effect on the
//  single-param path. The single-param path does not have a refactored
//  pipeline split, so delegation to generic is the correct approach.
//  TF32 mode is set for consistency with the architecture tier contract.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step_ampere(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    launch_mamba3_peer_step(
        param, grad, sharpness, exp_avg, exp_avg_sq, mu,
        gru_state, mamba_fwd_state, mamba_bwd_state,
        input_proj_W, input_proj_b,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        rescale, alpha_mu, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2,
        d_model, d_state, d_inner,
        gru_hidden, num_heads, pk_dim,
        expert_hidden, num_experts,
        expert_counts);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}


// ═══════════════════════════════════════════════════════════════════════
//  Ampere Forward: Batched Step — refactored pipeline with cp.async
//
//  Uses the refactored 3-phase pipeline:
//    1. batched_step_setup_and_sort — input projection, CUB sort, packing
//    2. generic_batched_precompute — FP32 precompute (TF32 math mode)
//    3. ampere_batched_scan_and_fused_elem — cp.async scan + cp.async fused_elem
//
//  The cp.async scan kernels (mamba3_scan_batched_cpasync_kernel,
//  mamba3_scan_combined_cpasync_kernel) use double-buffered
//  __pipeline_memcpy_async to overlap global memory loads with scan compute.
//  The fused_elem_step_cpasync_kernel uses cp.async for weight prefetch.
//  TF32 mode is set for cuBLAS math operations in the precompute phase.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_batched_step_ampere(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Phase 1: Setup + sort (shared with all tiers)
    auto ctx = batched_step_setup_and_sort(
        grads, sharpness_list, mamba_fwd_states, mamba_bwd_states,
        input_proj_W, input_proj_b, d_model, d_state, d_inner);

    if (ctx.total_N == 0) {
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        return;
    }

    // Phase 2: Precompute projections (TF32 via cuBLAS math mode)
    torch::Tensor fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C;
    torch::Tensor bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C;

    generic_batched_precompute(
        ctx, mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj,
        fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
        d_model, d_inner, d_state);

    generic_batched_precompute(
        ctx, mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj,
        bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
        d_model, d_inner, d_state);

    // Phase 3: cp.async scan + cp.async fused_elem (Ampere-specific)
    ampere_batched_scan_and_fused_elem(
        ctx,
        fwd_pre_x, fwd_pre_z, fwd_pre_dt, fwd_pre_B, fwd_pre_C,
        bwd_pre_x, bwd_pre_z, bwd_pre_dt, bwd_pre_B, bwd_pre_C,
        params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
        gru_states, mamba_fwd_states, mamba_bwd_states,
        mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
        mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
        mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
        mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
        mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
        mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
        rescale, beta2, lr, wd_eff, eps,
        d_model, d_state, d_inner,
        gru_hidden, num_heads, pk_dim,
        expert_hidden, num_experts,
        expert_counts);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}
