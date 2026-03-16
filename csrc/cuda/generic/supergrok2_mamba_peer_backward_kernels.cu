/*
 * SuperGrok v2 — Mamba-3 + PEER Bilevel Backward CUDA Kernels
 *
 * Backward kernels for bilevel meta-net optimization (Phase D).
 * Computes gradients w.r.t. all meta-net parameters given d_loss/d_smart_grad.
 *
 * Four backward kernels:
 *   1. input_proj_backward     — d_proj_W, d_proj_b from d_x
 *   2. mamba3_scan_backward    — d_mamba_weights from d_scan_out (recomputes forward)
 *   3. gru_backward            — d_gru_weights from d_gru_out
 *   4. expert_peer_backward    — d_expert_weights, d_peer_weights from d_expert_out
 *
 * Plus a forward-save scan kernel that stores intermediate states,
 * with optional gradient checkpointing (bilevel_checkpoint_interval).
 *
 * Performance features:
 *   - BilevelWorkspace: thread_local pre-allocated buffers for precompute
 *     outputs, reversed sort arrays, and gradient accumulators. Eliminates
 *     per-step torch::empty allocations.
 *   - ATen GEMM dispatch: For N >= GEMM_PRECOMPUTE_THRESHOLD (1024),
 *     projection precompute uses torch::mm_out (cuBLAS) instead of custom
 *     CUDA kernels for better Tensor Core utilization.
 *   - Shared-memory reduction for expert weight gradients (256x fewer atomics).
 *   - Dimension guards: TORCH_CHECK for d_model/d_inner/d_state vs maximums.
 *
 * All meta-net weights and gradients are FP32.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "platform.h"
#include "types.h"
#include "utils.cuh"


// ═══════════════════════════════════════════════════════════════════════
//  Bilevel Precompute Kernel
//
//  One thread per timestep. Computes input projection, dt projection,
//  B projection, C projection for all d_inner/d_state dimensions.
//  Outputs serve as both precomputed values for parallel scan AND
//  saved intermediates for backward (saved_x_branch, saved_z, saved_dt).
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void bilevel_precompute_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,    // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,    // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,    // [d_inner]
    const float* __restrict__ B_proj_W,     // [d_state, d_inner]
    const float* __restrict__ C_proj_W,     // [d_state, d_inner]
    float* __restrict__ pre_x_val,          // [N, d_inner] = saved_x_branch
    float* __restrict__ pre_z_val,          // [N, d_inner] = saved_z
    float* __restrict__ pre_dt_val,         // [N, d_inner] = saved_dt
    float* __restrict__ pre_B_val,          // [N, d_state]
    float* __restrict__ pre_C_val,          // [N, d_state]
    const int N, const int d_model, const int d_inner, const int d_state
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    float inp[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++)
        inp[d] = x_sorted[t * d_model + d];

    float x_branch[MAX_D_INNER];
    for (int j = 0; j < d_inner; j++) {
        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            x_val += in_proj_W[j * d_model + d] * inp[d];
            z_val += in_proj_W[(j + d_inner) * d_model + d] * inp[d];
        }
        x_branch[j] = x_val;
        pre_x_val[t * d_inner + j] = x_val;
        pre_z_val[t * d_inner + j] = z_val;
    }

    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        for (int k = 0; k < d_inner; k++)
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        pre_dt_val[t * d_inner + j] = dt_val;
    }

    for (int s = 0; s < d_state; s++) {
        float B_val = 0.0f, C_val = 0.0f;
        for (int j = 0; j < d_inner; j++) {
            B_val += B_proj_W[s * d_inner + j] * x_branch[j];
            C_val += C_proj_W[s * d_inner + j] * x_branch[j];
        }
        pre_B_val[t * d_state + s] = B_val;
        pre_C_val[t * d_state + s] = C_val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Softplus + Bias Kernel (used after cuBLAS GEMM for dt projection)
//
//  Applies dt_val = softplus(dt_raw + bias[j]) for each element.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void softplus_bias_kernel(
    float* __restrict__ dt_out,       // [N, d_inner] — in-place
    const float* __restrict__ bias,   // [d_inner]
    const int N, const int d_inner
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * d_inner) return;
    const int j = idx % d_inner;
    float dt_raw = dt_out[idx] + bias[j];
    dt_out[idx] = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
}


// ═══════════════════════════════════════════════════════════════════════
//  cuBLAS GEMM Precompute (ATen wrappers)
//
//  Replaces bilevel_precompute_kernel with batched GEMM calls for large N.
//  Uses ATen's torch::mm (cuBLAS under the hood) for better throughput
//  on large parameter tensors.
//
//  Projections:
//    x_branch [N,d_inner] = x_sorted [N,d_model] × in_proj_W[0:d_inner,:].T
//    z        [N,d_inner] = x_sorted [N,d_model] × in_proj_W[d_inner:,:].T
//    dt       [N,d_inner] = softplus(x_branch [N,d_inner] × dt_proj_W.T + bias)
//    B        [N,d_state] = x_branch [N,d_inner] × B_proj_W.T
//    C        [N,d_state] = x_branch [N,d_inner] × C_proj_W.T
// ═══════════════════════════════════════════════════════════════════════

static void bilevel_precompute_gemm(
    torch::Tensor x_sorted,       // [N, d_model]
    torch::Tensor in_proj_W,      // [2*d_inner, d_model]
    torch::Tensor dt_proj_W,      // [d_inner, d_inner]
    torch::Tensor dt_proj_b,      // [d_inner]
    torch::Tensor B_proj_W,       // [d_state, d_inner]
    torch::Tensor C_proj_W,       // [d_state, d_inner]
    torch::Tensor pre_x_val,      // [N, d_inner] output
    torch::Tensor pre_z_val,      // [N, d_inner] output
    torch::Tensor pre_dt_val,     // [N, d_inner] output
    torch::Tensor pre_B_val,      // [N, d_state] output
    torch::Tensor pre_C_val,      // [N, d_state] output
    int d_model, int d_inner, int d_state
) {
    const int N = x_sorted.size(0);

    // Split in_proj into x and z halves: [d_inner, d_model] each
    auto in_proj_x = in_proj_W.narrow(0, 0, d_inner);        // [d_inner, d_model]
    auto in_proj_z = in_proj_W.narrow(0, d_inner, d_inner);   // [d_inner, d_model]

    // x_branch = x_sorted @ in_proj_x.T → [N, d_inner] (written directly to pre_x_val)
    torch::mm_out(pre_x_val, x_sorted, in_proj_x.t());

    // z = x_sorted @ in_proj_z.T → [N, d_inner]
    torch::mm_out(pre_z_val, x_sorted, in_proj_z.t());

    // dt_raw = x_branch @ dt_proj_W.T → [N, d_inner], then add bias + softplus
    torch::mm_out(pre_dt_val, pre_x_val, dt_proj_W.t());
    int total_dt = N * d_inner;
    int dt_grid = (total_dt + SG2B_BLOCK - 1) / SG2B_BLOCK;
    softplus_bias_kernel<<<dt_grid, SG2B_BLOCK>>>(
        pre_dt_val.data_ptr<float>(),
        dt_proj_b.data_ptr<float>(),
        N, d_inner
    );

    // B = x_branch @ B_proj_W.T → [N, d_state]
    torch::mm_out(pre_B_val, pre_x_val, B_proj_W.t());

    // C = x_branch @ C_proj_W.T → [N, d_state]
    torch::mm_out(pre_C_val, pre_x_val, C_proj_W.t());
}


// ═══════════════════════════════════════════════════════════════════════
//  Parallel Scan with State Saving for Bilevel Forward
//
//  Blelloch parallel prefix scan (same as forward scan kernel) plus
//  writes h[t] to saved_states for backward pass.
//
//  Grid:  d_inner blocks (one per scan dimension j)
//  Block: PSCAN_BLOCK threads (power of 2)
//  Shared: 6 * block_size floats for Blelloch scan
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_parallel_scan_fwd_save_kernel(
    const float* __restrict__ pre_x_val,
    const float* __restrict__ pre_z_val,
    const float* __restrict__ pre_dt_val,
    const float* __restrict__ pre_B_val,
    const float* __restrict__ pre_C_val,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_output,        // [N, d_inner] — must be pre-zeroed
    float* __restrict__ final_state,        // [d_inner, d_state]
    float* __restrict__ saved_states,       // [N or num_ckpts, d_inner, d_state]
    const float* __restrict__ initial_state,
    const int N, const int d_inner, const int d_state,
    const int reverse, const int checkpoint_interval
) {
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];

    const int chunk_size = (N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N);
    const int my_count = max(my_end - my_start, 0);
    const int half_d_state = d_state / 2;

    float A[MAX_D_STATE], freq_arr[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[j * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq_arr[p] = rope_freq[j * half_d_state + p];
    float D_val = D_param[j];

    float h_init_all[MAX_D_STATE];
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = initial_state[j * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = 0.0f;
    }

    #define BUILD_AFFINE_SAVE(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt = pre_dt_val[(t_idx) * d_inner + j]; \
        float x_val = pre_x_val[(t_idx) * d_inner + j]; \
        float B_e = pre_B_val[(t_idx) * d_state + (s_e)]; \
        float B_o = pre_B_val[(t_idx) * d_state + (s_o)]; \
        float A_bar_e = (1.0f + dt * (A_e) / 2.0f) / (1.0f - dt * (A_e) / 2.0f + 1e-8f); \
        float A_bar_o = (1.0f + dt * (A_o) / 2.0f) / (1.0f - dt * (A_o) / 2.0f + 1e-8f); \
        float cos_v, sin_v; \
        FAST_SINCOSF(dt * (f_val), &sin_v, &cos_v); \
        (elem_out).m00 = A_bar_e * cos_v; \
        (elem_out).m01 = -A_bar_e * sin_v; \
        (elem_out).m10 = A_bar_o * sin_v; \
        (elem_out).m11 = A_bar_o * cos_v; \
        (elem_out).b0 = dt * B_e * x_val; \
        (elem_out).b1 = dt * B_o * x_val; \
    } while(0)

    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p, s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq_arr[p];
        const float h_init_e = h_init_all[s_e], h_init_o = h_init_all[s_o];

        // Step 1: Sequential scan within chunk → get chunk summary
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            Affine2x2 elem;
            BUILD_AFFINE_SAVE(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine(summary, elem);
        }

        int base = ltid * 6;
        smem[base] = summary.m00; smem[base+1] = summary.m01;
        smem[base+2] = summary.m10; smem[base+3] = summary.m11;
        smem[base+4] = summary.b0;  smem[base+5] = summary.b1;
        __syncthreads();

        // Step 2: Blelloch exclusive prefix scan on chunk summaries
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            __syncthreads();
        }

        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last] = 1.0f; smem[last+1] = 0.0f;
            smem[last+2] = 0.0f; smem[last+3] = 1.0f;
            smem[last+4] = 0.0f; smem[last+5] = 0.0f;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = num_threads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                smem[(idx-stride)*6]   = right.m00; smem[(idx-stride)*6+1] = right.m01;
                smem[(idx-stride)*6+2] = right.m10; smem[(idx-stride)*6+3] = right.m11;
                smem[(idx-stride)*6+4] = right.b0;  smem[(idx-stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            __syncthreads();
        }

        Affine2x2 prefix = {smem[ltid*6], smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // Step 3: Re-scan chunk with prefix, compute output + save states
        Affine2x2 running = prefix;
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            int seq_step = my_start + step;  // sequential step index

            Affine2x2 elem;
            BUILD_AFFINE_SAVE(t, A_e, A_o, f_val, s_e, s_o, elem);
            running = affine_combine(running, elem);

            float h_e = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;

            // Save state
            if (checkpoint_interval <= 1) {
                stream_store(&saved_states[(t * d_inner + j) * d_state + s_e], h_e);
                stream_store(&saved_states[(t * d_inner + j) * d_state + s_o], h_o);
            } else {
                bool at_seg_end = ((seq_step + 1) % checkpoint_interval == 0)
                                  || (seq_step == N - 1);
                if (at_seg_end) {
                    int ckpt_idx = seq_step / checkpoint_interval;
                    stream_store(&saved_states[(ckpt_idx * d_inner + j) * d_state + s_e], h_e);
                    stream_store(&saved_states[(ckpt_idx * d_inner + j) * d_state + s_o], h_o);
                }
            }

            // Accumulate y_val contribution from this pair
            float C_e = pre_C_val[t * d_state + s_e];
            float C_o = pre_C_val[t * d_state + s_o];
            scan_output[t * d_inner + j] += h_e * C_e + h_o * C_o;
        }

        // Last thread writes final state for this pair
        if (my_end == N && my_count > 0) {
            float h_e_final = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o_final = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;
            final_state[j * d_state + s_e] = h_e_final;
            final_state[j * d_state + s_o] = h_o_final;
        }

        __syncthreads();
    }

    #undef BUILD_AFFINE_SAVE

    // Apply SiLU gating and D skip connection
    for (int step = 0; step < my_count; step++) {
        int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
        float z = pre_z_val[t * d_inner + j];
        float silu_z = z / (1.0f + expf(-z));
        float x_val = pre_x_val[t * d_inner + j];
        scan_output[t * d_inner + j] = scan_output[t * d_inner + j] * silu_z + D_val * x_val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Batched Parallel Scan with State Saving
//
//  Single-launch kernel for all parameters. 2D grid:
//    blockIdx.x = d_inner dimension (j)
//    blockIdx.y = parameter index
//  Eliminates per-parameter kernel launch loop.
//
//  Grid:  (d_inner, num_params)
//  Block: PSCAN_BLOCK threads (power of 2)
//  Shared: 6 * block_size floats
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_batched_parallel_scan_fwd_save_kernel(
    const float* __restrict__ pre_x_val,     // [total_N, d_inner]
    const float* __restrict__ pre_z_val,     // [total_N, d_inner]
    const float* __restrict__ pre_dt_val,    // [total_N, d_inner]
    const float* __restrict__ pre_B_val,     // [total_N, d_state]
    const float* __restrict__ pre_C_val,     // [total_N, d_state]
    const float* __restrict__ A_log,         // [d_inner, d_state]
    const float* __restrict__ D_param,       // [d_inner]
    const float* __restrict__ rope_freq,     // [d_inner, d_state/2]
    float* __restrict__ scan_output,         // [total_N, d_inner] — pre-zeroed
    const float* __restrict__ initial_states, // [num_params, d_inner, d_state]
    float* __restrict__ final_states,        // [num_params, d_inner, d_state]
    float* __restrict__ saved_states,        // [total_ckpts or total_N, d_inner, d_state]
    const int* __restrict__ offsets,          // [num_params + 1]
    const int* __restrict__ reverse_flags,   // [num_params]
    const int* __restrict__ ckpt_offsets,     // [num_params + 1] or nullptr
    const int d_inner, const int d_state,
    const int checkpoint_interval
) {
    const int j = blockIdx.x;          // d_inner index
    const int param_idx = blockIdx.y;  // parameter index
    if (j >= d_inner) return;

    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;
    extern __shared__ float smem[];

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    if (N == 0) return;
    const int reverse = reverse_flags[param_idx];

    // Offset pointers to this parameter's packed data
    const float* my_pre_x = pre_x_val + start * d_inner;
    const float* my_pre_z = pre_z_val + start * d_inner;
    const float* my_pre_dt = pre_dt_val + start * d_inner;
    const float* my_pre_B = pre_B_val + start * d_state;
    const float* my_pre_C = pre_C_val + start * d_state;
    float* my_scan_out = scan_output + start * d_inner;

    const int ckpt_start = (checkpoint_interval > 1 && ckpt_offsets != nullptr)
                           ? ckpt_offsets[param_idx] : start;
    float* my_saved = saved_states + ckpt_start * d_inner * d_state;

    const int chunk_size = (N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N);
    const int my_count = max(my_end - my_start, 0);
    const int half_d_state = d_state / 2;

    float A[MAX_D_STATE], freq_arr[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[j * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq_arr[p] = rope_freq[j * half_d_state + p];
    float D_val = D_param[j];

    float h_init_all[MAX_D_STATE];
    const float* init_ptr = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        h_init_all[s] = init_ptr[j * d_state + s];

    #define BUILD_AFFINE_BATCH(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt = my_pre_dt[(t_idx) * d_inner + j]; \
        float x_val = my_pre_x[(t_idx) * d_inner + j]; \
        float B_e = my_pre_B[(t_idx) * d_state + (s_e)]; \
        float B_o = my_pre_B[(t_idx) * d_state + (s_o)]; \
        float A_bar_e = (1.0f + dt * (A_e) / 2.0f) / (1.0f - dt * (A_e) / 2.0f + 1e-8f); \
        float A_bar_o = (1.0f + dt * (A_o) / 2.0f) / (1.0f - dt * (A_o) / 2.0f + 1e-8f); \
        float cos_v, sin_v; \
        FAST_SINCOSF(dt * (f_val), &sin_v, &cos_v); \
        (elem_out).m00 = A_bar_e * cos_v; \
        (elem_out).m01 = -A_bar_e * sin_v; \
        (elem_out).m10 = A_bar_o * sin_v; \
        (elem_out).m11 = A_bar_o * cos_v; \
        (elem_out).b0 = dt * B_e * x_val; \
        (elem_out).b1 = dt * B_o * x_val; \
    } while(0)

    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p, s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq_arr[p];
        const float h_init_e = h_init_all[s_e], h_init_o = h_init_all[s_o];

        // Step 1: Sequential scan within chunk
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            Affine2x2 elem;
            BUILD_AFFINE_BATCH(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine(summary, elem);
        }

        int base = ltid * 6;
        smem[base] = summary.m00; smem[base+1] = summary.m01;
        smem[base+2] = summary.m10; smem[base+3] = summary.m11;
        smem[base+4] = summary.b0;  smem[base+5] = summary.b1;
        __syncthreads();

        // Step 2: Blelloch exclusive prefix scan
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            __syncthreads();
        }
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last] = 1.0f; smem[last+1] = 0.0f;
            smem[last+2] = 0.0f; smem[last+3] = 1.0f;
            smem[last+4] = 0.0f; smem[last+5] = 0.0f;
        }
        __syncthreads();
        for (int stride = num_threads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                smem[(idx-stride)*6]   = right.m00; smem[(idx-stride)*6+1] = right.m01;
                smem[(idx-stride)*6+2] = right.m10; smem[(idx-stride)*6+3] = right.m11;
                smem[(idx-stride)*6+4] = right.b0;  smem[(idx-stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            __syncthreads();
        }

        Affine2x2 prefix = {smem[ltid*6], smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // Step 3: Re-scan with prefix, save states
        Affine2x2 running = prefix;
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            int seq_step = my_start + step;

            Affine2x2 elem;
            BUILD_AFFINE_BATCH(t, A_e, A_o, f_val, s_e, s_o, elem);
            running = affine_combine(running, elem);

            float h_e = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;

            if (checkpoint_interval <= 1) {
                my_saved[(t * d_inner + j) * d_state + s_e] = h_e;
                my_saved[(t * d_inner + j) * d_state + s_o] = h_o;
            } else {
                bool at_seg_end = ((seq_step + 1) % checkpoint_interval == 0)
                                  || (seq_step == N - 1);
                if (at_seg_end) {
                    int ckpt_idx = seq_step / checkpoint_interval;
                    my_saved[(ckpt_idx * d_inner + j) * d_state + s_e] = h_e;
                    my_saved[(ckpt_idx * d_inner + j) * d_state + s_o] = h_o;
                }
            }

            float C_e = my_pre_C[t * d_state + s_e];
            float C_o = my_pre_C[t * d_state + s_o];
            my_scan_out[t * d_inner + j] += h_e * C_e + h_o * C_o;
        }

        // Last thread writes final state
        if (my_end == N && my_count > 0) {
            float* my_final = final_states + param_idx * d_inner * d_state;
            float h_e_f = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o_f = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;
            my_final[j * d_state + s_e] = h_e_f;
            my_final[j * d_state + s_o] = h_o_f;
        }
        __syncthreads();
    }

    #undef BUILD_AFFINE_BATCH

    // SiLU gating + D skip connection
    for (int step = 0; step < my_count; step++) {
        int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
        float z = my_pre_z[t * d_inner + j];
        float silu_z = z / (1.0f + expf(-z));
        float x_val = my_pre_x[t * d_inner + j];
        my_scan_out[t * d_inner + j] = my_scan_out[t * d_inner + j] * silu_z + D_val * x_val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Utility Kernels: Segmented Reverse and Combine+Flip
//
//  Replace per-param C++ loops calling .flip().copy_() with single
//  CUDA kernel launches for all segments.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void reverse_segments_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ offsets,
    const int d,           // stride (e.g., d_model)
    const int num_params
) {
    // Grid: total_elements = total_N * d
    // Each thread reverses one (row, col) element within its segment
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_N = offsets[num_params];
    if (global_idx >= total_N * d) return;

    const int row = global_idx / d;
    const int col = global_idx % d;

    // Binary search for segment
    int lo = 0, hi = num_params;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    const int seg_start = offsets[lo];
    const int seg_end = offsets[lo + 1];
    const int local_row = row - seg_start;
    const int N = seg_end - seg_start;
    const int reversed_row = seg_start + (N - 1 - local_row);

    dst[reversed_row * d + col] = src[row * d + col];
}

__launch_bounds__(256, 8)
__global__ void combine_fwd_bwd_kernel(
    const float* __restrict__ fwd,
    const float* __restrict__ bwd,
    float* __restrict__ out,
    const int* __restrict__ offsets,
    const int d,           // d_model
    const int num_params
) {
    // out = fwd + reverse_segments(bwd)
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_N = offsets[num_params];
    if (global_idx >= total_N * d) return;

    const int row = global_idx / d;
    const int col = global_idx % d;

    // Binary search for segment
    int lo = 0, hi = num_params;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    const int seg_start = offsets[lo];
    const int seg_end = offsets[lo + 1];
    const int local_row = row - seg_start;
    const int N = seg_end - seg_start;
    const int reversed_row = seg_start + (N - 1 - local_row);

    out[row * d + col] = fwd[row * d + col] + bwd[reversed_row * d + col];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Mamba-3 Scan Forward with State Saving
//
//  Same as mamba3_scan_kernel but writes h[step] to saved_states buffer
//  for use in backward pass. Also saves x_branch, z_val, dt_val per step.
//
//  saved_states: [N, d_inner, d_state]
//  saved_x_branch: [N, d_inner]
//  saved_z: [N, d_inner]
//  saved_dt: [N, d_inner]
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_scan_fwd_save_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,    // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,    // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,    // [d_inner]
    const float* __restrict__ B_proj_W,     // [d_state, d_inner]
    const float* __restrict__ C_proj_W,     // [d_state, d_inner]
    const float* __restrict__ A_log,        // [d_inner, d_state]
    const float* __restrict__ D_param,      // [d_inner]
    const float* __restrict__ rope_freq,    // [d_inner, d_state/2]
    float* __restrict__ scan_output,        // [N, d_inner]
    float* __restrict__ final_state,        // [d_inner, d_state]
    // Saved intermediates for backward
    float* __restrict__ saved_states,       // [N, d_inner, d_state]
    float* __restrict__ saved_x_branch,     // [N, d_inner]
    float* __restrict__ saved_z,            // [N, d_inner]
    float* __restrict__ saved_dt,           // [N, d_inner]
    const float* __restrict__ initial_state, // [d_inner, d_state] or nullptr
    const int N,
    const int d_model,
    const int d_inner,
    const int d_state,
    const int reverse,
    const int checkpoint_interval  // 0 or 1 = save every step; >1 = checkpoint every C steps
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    // Shared memory for cross-thread communication
    extern __shared__ float smem[];
    float* s_x_branch = smem;           // [d_inner]

    float h[MAX_D_STATE];
    float h_snap[MAX_D_STATE]; // snapshot for RoPE
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++) h[s] = initial_state[tid * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++) h[s] = 0.0f;
    }

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];  // paired RoPE: d_state/2 frequencies
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];

    float D_val = D_param[tid];

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        // Input projection
        float x_val = 0.0f;
        float z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = x_sorted[i * d_model + d];
            x_val += in_proj_W[tid * d_model + d] * inp;
            z_val += in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        // Save x_branch and z
        stream_store(&saved_x_branch[i * d_inner + tid], x_val);
        stream_store(&saved_z[i * d_inner + tid], z_val);

        // Write x_branch to shared memory for cross-thread access
        s_x_branch[tid] = x_val;
        __syncthreads();

        // FULL dt projection
        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++) {
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        }
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        stream_store(&saved_dt[i * d_inner + tid], dt_val);

        // Snapshot h for RoPE (fixes read-after-write)
        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        // State update with trapezoidal + RoPE
        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);

            // FULL B projection
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            float B_bar = dt_val * B_val;

            // Paired RoPE using SNAPSHOT
            int pair_idx = s / 2;
            float cos_p, sin_p;
            FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }

            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // Save state after update (checkpointed or every step)
        if (checkpoint_interval <= 1) {
            for (int s = 0; s < d_state; s++)
                stream_store(&saved_states[(i * d_inner + tid) * d_state + s], h[s]);
        } else {
            bool at_seg_end = ((step + 1) % checkpoint_interval == 0) || (step == N - 1);
            if (at_seg_end) {
                int ckpt_idx = step / checkpoint_interval;
                for (int s = 0; s < d_state; s++)
                    stream_store(&saved_states[(ckpt_idx * d_inner + tid) * d_state + s], h[s]);
            }
        }

        // FULL C projection for output
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            y_val += h[s] * C_val;
        }

        float silu_z = z_val / (1.0f + expf(-z_val));
        y_val = y_val * silu_z + D_val * x_val;

        scan_output[i * d_inner + tid] = y_val;
        __syncthreads(); // ensure all threads done before next step
    }

    for (int s = 0; s < d_state; s++)
        final_state[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1b: Batched Mamba-3 Scan Forward with State Saving
//
//  One block per parameter. Same logic as fwd_save but with packed data
//  and offset table. Saves states, x_branch, z, dt for backward.
//
//  Grid: num_params (one block per parameter)
//  Threads: d_inner per block
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_scan_fwd_save_batched_kernel(
    const float* __restrict__ x_sorted_packed,    // [total_N, d_model]
    float* __restrict__ scan_output_packed,        // [total_N, d_inner]
    const float* __restrict__ initial_states,      // [num_params, d_inner, d_state]
    float* __restrict__ final_states,              // [num_params, d_inner, d_state]
    const int* __restrict__ offsets,               // [num_params + 1]
    const int* __restrict__ reverse_flags,         // [num_params]
    // Saved intermediates (packed)
    float* __restrict__ saved_states_packed,       // [total_N, d_inner, d_state]
    float* __restrict__ saved_x_branch_packed,     // [total_N, d_inner]
    float* __restrict__ saved_z_packed,            // [total_N, d_inner]
    float* __restrict__ saved_dt_packed,           // [total_N, d_inner]
    // Shared Mamba weights
    const float* __restrict__ in_proj_W,           // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,           // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,           // [d_inner]
    const float* __restrict__ B_proj_W,            // [d_state, d_inner]
    const float* __restrict__ C_proj_W,            // [d_state, d_inner]
    const float* __restrict__ A_log,               // [d_inner, d_state]
    const float* __restrict__ D_param,             // [d_inner]
    const float* __restrict__ rope_freq,           // [d_inner, d_state/2]
    const int* __restrict__ ckpt_offsets,          // [num_params + 1] or nullptr (for checkpointing)
    const int d_model,
    const int d_inner,
    const int d_state,
    const int checkpoint_interval  // 0 or 1 = save every step
) {
    const int param_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    const int reverse = reverse_flags[param_idx];

    extern __shared__ float smem[];
    float* s_x_branch = smem;

    // State in registers — load from initial_state
    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    const float* my_init = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];

    float D_val = D_param[tid];

    const float* my_x = x_sorted_packed + start * d_model;
    float* my_out = scan_output_packed + start * d_inner;
    // For checkpointing: saved_states uses ckpt_offsets instead of element offsets
    const int ckpt_start = (checkpoint_interval > 1 && ckpt_offsets != nullptr)
                           ? ckpt_offsets[param_idx] : start;
    float* my_saved_states = saved_states_packed + ckpt_start * d_inner * d_state;
    float* my_saved_xb = saved_x_branch_packed + start * d_inner;
    float* my_saved_z = saved_z_packed + start * d_inner;
    float* my_saved_dt = saved_dt_packed + start * d_inner;

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = my_x[i * d_model + d];
            x_val += in_proj_W[tid * d_model + d] * inp;
            z_val += in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        // Save x_branch and z
        my_saved_xb[i * d_inner + tid] = x_val;
        my_stream_store(&saved_z[i * d_inner + tid], z_val);

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        my_stream_store(&saved_dt[i * d_inner + tid], dt_val);

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
            float B_bar = dt_val * B_val;
            // Paired RoPE
            int pair_idx = s / 2;
            float cos_p, sin_p;
            FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // Save state after update (checkpointed or every step)
        if (checkpoint_interval <= 1) {
            for (int s = 0; s < d_state; s++)
                my_stream_store(&saved_states[(i * d_inner + tid) * d_state + s], h[s]);
        } else {
            bool at_seg_end = ((step + 1) % checkpoint_interval == 0) || (step == N - 1);
            if (at_seg_end) {
                int ckpt_idx = step / checkpoint_interval;
                for (int s = 0; s < d_state; s++)
                    my_stream_store(&saved_states[(ckpt_idx * d_inner + tid) * d_state + s], h[s]);
            }
        }

        // Output with C projection
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            y_val += h[s] * C_val;
        }
        float silu_z = z_val / (1.0f + expf(-z_val));
        y_val = y_val * silu_z + D_val * x_val;

        my_out[i * d_inner + tid] = y_val;
        __syncthreads();
    }

    float* my_final = final_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        my_final[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Mamba-3 Scan Backward
//
//  Computes gradients through the selective scan using saved states.
//  One thread per d_inner dimension, sequential over N (reverse of forward).
//
//  Accumulates gradients for:
//    - in_proj_W, dt_proj_W, dt_proj_b, B_proj_W, C_proj_W, A_log, D, rope_freq
//  Also produces d_x_sorted for upstream backward.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_scan_backward_kernel(
    const float* __restrict__ d_scan_output,  // [N, d_inner] gradient from downstream
    const float* __restrict__ x_sorted,       // [N, d_model]
    const float* __restrict__ saved_states,   // [N, d_inner, d_state]
    const float* __restrict__ saved_x_branch, // [N, d_inner]
    const float* __restrict__ saved_z,        // [N, d_inner]
    const float* __restrict__ saved_dt,       // [N, d_inner]
    // Weights (read-only)
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    // Gradient outputs (accumulated via atomicAdd)
    float* __restrict__ d_in_proj_W,          // [2*d_inner, d_model]
    float* __restrict__ d_dt_proj_W,          // [d_inner, d_inner]
    float* __restrict__ d_dt_proj_b,          // [d_inner]
    float* __restrict__ d_A_log,              // [d_inner, d_state]
    float* __restrict__ d_D_param,            // [d_inner]
    float* __restrict__ d_rope_freq,          // [d_inner, d_state/2]
    float* __restrict__ d_x_sorted,           // [N, d_model]
    const float* __restrict__ initial_state,  // [d_inner, d_state] or nullptr
    // Two-pass outputs: per-timestep warp-reduced derivatives for GEMM
    float* __restrict__ d_C_vals_buf,         // [N, d_state] — reduced across d_inner threads
    float* __restrict__ d_B_vals_buf,         // [N, d_state] — reduced across d_inner threads
    // Dims
    const int N, const int d_model, const int d_inner, const int d_state,
    const int reverse,
    const int checkpoint_interval  // 0 or 1 = no checkpointing; >1 = recompute from checkpoints
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    // Shared memory layout (two-pass: no s_d_C_proj_W / s_d_B_proj_W):
    //   s_x_branch:    [d_inner]
    //   s_d_dt_raw:    [d_inner]
    extern __shared__ float smem[];
    float* s_x_branch = smem;                                 // [d_inner]
    float* s_d_dt_raw = smem + d_inner;                       // [d_inner]

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];  // paired RoPE: d_state/2 frequencies
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];

    float D_val = D_param[tid];

    // Accumulated gradients for this thread's parameters
    float d_D_acc = 0.0f;
    float d_A_log_acc[MAX_D_STATE];
    float d_freq_acc[MAX_D_STATE / 2];  // paired: d_state/2 frequencies
    float d_dt_proj_b_acc = 0.0f;
    for (int s = 0; s < d_state; s++) {
        d_A_log_acc[s] = 0.0f;
    }
    for (int p = 0; p < half_d_state; p++) {
        d_freq_acc[p] = 0.0f;
    }
    // Full dt_proj_W gradient (not just diagonal)
    float d_dt_proj_W_row[MAX_D_INNER];
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    // Small per-thread accumulators (16 floats each — fit in registers)
    float d_in_proj_W_x_local[MAX_D_MODEL];  // row tid of d_in_proj_W
    float d_in_proj_W_z_local[MAX_D_MODEL];  // row (tid+d_inner) of d_in_proj_W
    for (int d = 0; d < d_model; d++) {
        d_in_proj_W_x_local[d] = 0.0f;
        d_in_proj_W_z_local[d] = 0.0f;
    }

    // Gradient of state: dh[s] propagated backward through time
    float dh[MAX_D_STATE];
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    // ---- Macro-like per-step backward block (used in both paths) ----
    // We define a lambda-style approach using goto-free inline code.
    // Both checkpointed and non-checkpointed paths call the same backward
    // logic per step. The only difference is how h_curr and h_prev are obtained.

    if (checkpoint_interval <= 1) {
        // ════════════════════════════════════════════════════════════
        //  ORIGINAL PATH: saved_states has all N states
        // ════════════════════════════════════════════════════════════
        for (int step = N - 1; step >= 0; step--) {
            int i = reverse ? (N - 1 - step) : step;

            float d_out = d_scan_output[i * d_inner + tid];
            float x_val = saved_x_branch[i * d_inner + tid];
            float z_val = saved_z[i * d_inner + tid];
            float dt_val = saved_dt[i * d_inner + tid];

            s_x_branch[tid] = x_val;
            __syncthreads();

            float sig_z = 1.0f / (1.0f + expf(-z_val));
            float silu_z = z_val * sig_z;

            float y_val = 0.0f;
            for (int s = 0; s < d_state; s++) {
                float C_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                y_val += saved_states[(i * d_inner + tid) * d_state + s] * C_val;
            }

            float d_y_val = d_out * silu_z;
            float d_silu_z = d_out * y_val;
            float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
            float d_x_from_D = d_out * D_val;
            d_D_acc += d_out * x_val;

            float d_x_from_C = 0.0f;
            for (int s = 0; s < d_state; s++) {
                float h_s = saved_states[(i * d_inner + tid) * d_state + s];
                float C_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                dh[s] += d_y_val * C_val;
                float d_C_val = d_y_val * h_s;
                // Two-pass: warp reduce d_C_val across d_inner threads, write to buffer
                float d_C_reduced = warp_reduce_sum(d_C_val, d_inner, tid);
                if (tid == 0)
                    d_C_vals_buf[i * d_state + s] = d_C_reduced;
                d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
            }

            float h_prev[MAX_D_STATE];
            if (step > 0) {
                int i_prev = reverse ? (N - step) : (step - 1);
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = saved_states[(i_prev * d_inner + tid) * d_state + s];
            } else {
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = (initial_state != nullptr)
                        ? initial_state[tid * d_state + s] : 0.0f;
            }

            float dh_snap[MAX_D_STATE];
            for (int s = 0; s < d_state; s++) dh_snap[s] = dh[s];
            for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

            float d_dt_val = 0.0f;
            float d_x_from_scan = 0.0f;

            for (int s = 0; s < d_state; s++) {
                float half_dtA = dt_val * A[s] / 2.0f;
                float denom_val = 1.0f - half_dtA + 1e-8f;
                float A_bar = (1.0f + half_dtA) / denom_val;
                float B_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                float B_bar = dt_val * B_val;
                int pair_idx = s / 2;
                float cos_p, sin_p;
                FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                float h_rot;
                int partner;
                float sign;
                if (s % 2 == 0) {
                    partner = s + 1; sign = -1.0f;
                    h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
                } else {
                    partner = s - 1; sign = 1.0f;
                    h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
                }
                float d_h_s = dh_snap[s];
                float d_A_bar = d_h_s * h_rot;
                float d_h_rot = d_h_s * A_bar;
                float d_B_bar = d_h_s * x_val;
                d_x_from_scan += d_h_s * B_bar;
                d_dt_val += d_B_bar * B_val;
                float d_B_val = d_B_bar * dt_val;
                // Two-pass: warp reduce d_B_val across d_inner threads, write to buffer
                float d_B_reduced = warp_reduce_sum(d_B_val, d_inner, tid);
                if (tid == 0)
                    d_B_vals_buf[i * d_state + s] = d_B_reduced;
                d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];
                float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
                d_dt_val += d_half_dtA * A[s] / 2.0f;
                float d_A_s = d_half_dtA * dt_val / 2.0f;
                d_A_log_acc[s] += d_A_s * A[s];
                float d_h_prev_s = d_h_rot * cos_p;
                float d_h_prev_partner = d_h_rot * sign * sin_p;
                float d_cos = d_h_rot * h_prev[s];
                float d_sin = d_h_rot * sign * h_prev[partner];
                d_dt_val += (-sin_p * freq[pair_idx]) * d_cos + (cos_p * freq[pair_idx]) * d_sin;
                d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;
                dh[s] += d_h_prev_s;
                dh[partner] += d_h_prev_partner;
            }

            float dt_raw = dt_proj_b[tid];
            for (int j = 0; j < d_inner; j++)
                dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
            float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
            float d_dt_raw = d_dt_val * sig_dt;

            d_dt_proj_b_acc += d_dt_raw;
            for (int j = 0; j < d_inner; j++)
                d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];

            s_d_dt_raw[tid] = d_dt_raw;
            __syncthreads();
            float d_x_from_dt = 0.0f;
            for (int t = 0; t < d_inner; t++)
                d_x_from_dt += s_d_dt_raw[t] * dt_proj_W[t * d_inner + tid];

            float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;

            for (int d = 0; d < d_model; d++) {
                float inp = x_sorted[i * d_model + d];
                d_in_proj_W_x_local[d] += d_x_val * inp;
                d_in_proj_W_z_local[d] += d_z_val * inp;
                atomicAdd(&d_x_sorted[i * d_model + d],
                          d_x_val * in_proj_W[tid * d_model + d] +
                          d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
            }
            __syncthreads();
        }
    } else {
        // ════════════════════════════════════════════════════════════
        //  CHECKPOINTED PATH: saved_states has only checkpoint states
        //  Recompute intermediate states per segment during backward.
        // ════════════════════════════════════════════════════════════
        int num_segments = (N + checkpoint_interval - 1) / checkpoint_interval;

        // Buffer for recomputed segment states: seg_h[local][s]
        // seg_h[0] = state before first step of segment (input state)
        // seg_h[k] = state after local step k-1 (for k >= 1)
        float seg_h[(MAX_CKPT_INTERVAL + 1) * MAX_D_STATE];

        for (int seg = num_segments - 1; seg >= 0; seg--) {
            int seg_start = seg * checkpoint_interval;
            int seg_end = (seg_start + checkpoint_interval < N)
                          ? seg_start + checkpoint_interval : N;
            int seg_len = seg_end - seg_start;

            // === Phase 1: Load checkpoint input state ===
            if (seg == 0) {
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = (initial_state != nullptr)
                               ? initial_state[tid * d_state + s] : 0.0f;
            } else {
                // Checkpoint[seg-1] stores state at end of previous segment
                int ckpt_idx = seg - 1;
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = saved_states[(ckpt_idx * d_inner + tid) * d_state + s];
            }

            // === Phase 2: Forward-recompute all states in this segment ===
            for (int local = 0; local < seg_len; local++) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;

                float x_val = saved_x_branch[i * d_inner + tid];
                float dt_val = saved_dt[i * d_inner + tid];

                s_x_branch[tid] = x_val;
                __syncthreads();

                // Snapshot for RoPE
                float h_snap_r[MAX_D_STATE];
                for (int s = 0; s < d_state; s++)
                    h_snap_r[s] = seg_h[local * MAX_D_STATE + s];

                for (int s = 0; s < d_state; s++) {
                    float A_bar = (1.0f + dt_val * A[s] / 2.0f)
                                  / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
                    float B_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dt_val * B_val;

                    int pair_idx = s / 2;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    if (s % 2 == 0)
                        h_rot = h_snap_r[s] * cos_p - h_snap_r[s + 1] * sin_p;
                    else
                        h_rot = h_snap_r[s] * cos_p + h_snap_r[s - 1] * sin_p;

                    seg_h[(local + 1) * MAX_D_STATE + s] = A_bar * h_rot + B_bar * x_val;
                }
                __syncthreads();
            }

            // === Phase 3: Backward through this segment (reverse order) ===
            for (int local = seg_len - 1; local >= 0; local--) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;

                float d_out = d_scan_output[i * d_inner + tid];
                float x_val = saved_x_branch[i * d_inner + tid];
                float z_val = saved_z[i * d_inner + tid];
                float dt_val = saved_dt[i * d_inner + tid];

                s_x_branch[tid] = x_val;
                __syncthreads();

                float sig_z = 1.0f / (1.0f + expf(-z_val));
                float silu_z = z_val * sig_z;

                // Use recomputed state: seg_h[(local+1)*MAX_D_STATE + s]
                float y_val = 0.0f;
                for (int s = 0; s < d_state; s++) {
                    float C_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                    y_val += seg_h[(local + 1) * MAX_D_STATE + s] * C_val;
                }

                float d_y_val = d_out * silu_z;
                float d_silu_z = d_out * y_val;
                float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
                float d_x_from_D = d_out * D_val;
                d_D_acc += d_out * x_val;

                float d_x_from_C = 0.0f;
                for (int s = 0; s < d_state; s++) {
                    float h_s = seg_h[(local + 1) * MAX_D_STATE + s];
                    float C_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                    dh[s] += d_y_val * C_val;
                    float d_C_val = d_y_val * h_s;
                    // Two-pass: warp reduce d_C_val across d_inner threads, write to buffer
                    float d_C_reduced = warp_reduce_sum(d_C_val, d_inner, tid);
                    if (tid == 0)
                        d_C_vals_buf[i * d_state + s] = d_C_reduced;
                    d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
                }

                // h_prev from recomputed segment states
                float h_prev[MAX_D_STATE];
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = seg_h[local * MAX_D_STATE + s];

                float dh_snap[MAX_D_STATE];
                for (int s = 0; s < d_state; s++) dh_snap[s] = dh[s];
                for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

                float d_dt_val = 0.0f;
                float d_x_from_scan = 0.0f;

                for (int s = 0; s < d_state; s++) {
                    float half_dtA = dt_val * A[s] / 2.0f;
                    float denom_val = 1.0f - half_dtA + 1e-8f;
                    float A_bar = (1.0f + half_dtA) / denom_val;
                    float B_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dt_val * B_val;
                    int pair_idx = s / 2;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    int partner;
                    float sign;
                    if (s % 2 == 0) {
                        partner = s + 1; sign = -1.0f;
                        h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
                    } else {
                        partner = s - 1; sign = 1.0f;
                        h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
                    }
                    float d_h_s = dh_snap[s];
                    float d_A_bar = d_h_s * h_rot;
                    float d_h_rot = d_h_s * A_bar;
                    float d_B_bar = d_h_s * x_val;
                    d_x_from_scan += d_h_s * B_bar;
                    d_dt_val += d_B_bar * B_val;
                    float d_B_val = d_B_bar * dt_val;
                    // Two-pass: warp reduce d_B_val across d_inner threads, write to buffer
                    float d_B_reduced = warp_reduce_sum(d_B_val, d_inner, tid);
                    if (tid == 0)
                        d_B_vals_buf[i * d_state + s] = d_B_reduced;
                    d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];
                    float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
                    d_dt_val += d_half_dtA * A[s] / 2.0f;
                    float d_A_s = d_half_dtA * dt_val / 2.0f;
                    d_A_log_acc[s] += d_A_s * A[s];
                    float d_h_prev_s = d_h_rot * cos_p;
                    float d_h_prev_partner = d_h_rot * sign * sin_p;
                    float d_cos = d_h_rot * h_prev[s];
                    float d_sin = d_h_rot * sign * h_prev[partner];
                    d_dt_val += (-sin_p * freq[pair_idx]) * d_cos + (cos_p * freq[pair_idx]) * d_sin;
                    d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;
                    dh[s] += d_h_prev_s;
                    dh[partner] += d_h_prev_partner;
                }

                float dt_raw = dt_proj_b[tid];
                for (int j = 0; j < d_inner; j++)
                    dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
                float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
                float d_dt_raw = d_dt_val * sig_dt;

                d_dt_proj_b_acc += d_dt_raw;
                for (int j = 0; j < d_inner; j++)
                    d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];

                s_d_dt_raw[tid] = d_dt_raw;
                __syncthreads();
                float d_x_from_dt = 0.0f;
                for (int t = 0; t < d_inner; t++)
                    d_x_from_dt += s_d_dt_raw[t] * dt_proj_W[t * d_inner + tid];

                float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;

                for (int d = 0; d < d_model; d++) {
                    float inp = x_sorted[i * d_model + d];
                    d_in_proj_W_x_local[d] += d_x_val * inp;
                    d_in_proj_W_z_local[d] += d_z_val * inp;
                    atomicAdd(&d_x_sorted[i * d_model + d],
                              d_x_val * in_proj_W[tid * d_model + d] +
                              d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
                }
                __syncthreads();
            }
        }
    }

    // Write accumulated per-thread parameter gradients
    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    for (int j = 0; j < d_inner; j++) {
        atomicAdd(&d_dt_proj_W[tid * d_inner + j], d_dt_proj_W_row[j]);
    }
    for (int s = 0; s < d_state; s++) {
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
    }
    for (int p = 0; p < half_d_state; p++) {
        atomicAdd(&d_rope_freq[tid * half_d_state + p], d_freq_acc[p]);
    }
    // Two-pass: C_proj_W and B_proj_W gradients are computed via GEMM in the launcher
    // (d_C_vals_buf and d_B_vals_buf were filled per-timestep above)
    // Write per-thread in_proj_W gradients
    for (int d = 0; d < d_model; d++) {
        atomicAdd(&d_in_proj_W[tid * d_model + d], d_in_proj_W_x_local[d]);
        atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_in_proj_W_z_local[d]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2b: Batched Mamba-3 Scan Backward
//
//  One block per parameter. Same logic as backward kernel but with
//  packed data and offset table. Weight gradients accumulated via
//  atomicAdd (shared across all params).
//
//  Grid: num_params (one block per parameter)
//  Threads: d_inner per block
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(16, 8)
__global__ void mamba3_scan_backward_batched_kernel(
    const float* __restrict__ d_scan_output_packed,  // [total_N, d_inner]
    const float* __restrict__ x_sorted_packed,       // [total_N, d_model]
    const float* __restrict__ saved_states_packed,   // [total_N, d_inner, d_state]
    const float* __restrict__ saved_x_branch_packed, // [total_N, d_inner]
    const float* __restrict__ saved_z_packed,        // [total_N, d_inner]
    const float* __restrict__ saved_dt_packed,       // [total_N, d_inner]
    const int* __restrict__ offsets,                 // [num_params + 1]
    const int* __restrict__ reverse_flags,           // [num_params]
    // Initial states (for h_prev at step 0)
    const float* __restrict__ initial_states,        // [num_params, d_inner, d_state]
    // Weights (read-only)
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    // Gradient outputs (accumulated via atomicAdd)
    float* __restrict__ d_in_proj_W,
    float* __restrict__ d_dt_proj_W,
    float* __restrict__ d_dt_proj_b,
    float* __restrict__ d_A_log,
    float* __restrict__ d_D_param,
    float* __restrict__ d_rope_freq,
    float* __restrict__ d_x_sorted_packed,           // [total_N, d_model]
    const int* __restrict__ ckpt_offsets,            // [num_params + 1] or nullptr
    // Two-pass outputs: per-timestep warp-reduced derivatives for GEMM
    float* __restrict__ d_C_vals_buf,                // [total_N, d_state]
    float* __restrict__ d_B_vals_buf,                // [total_N, d_state]
    // Dims
    const int d_model, const int d_inner, const int d_state,
    const int checkpoint_interval  // 0 or 1 = no checkpointing
) {
    const int param_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    const int reverse = reverse_flags[param_idx];

    // Shared memory layout (two-pass: no s_d_C_proj_W / s_d_B_proj_W):
    //   s_x_branch:    [d_inner]
    //   s_d_dt_raw:    [d_inner]
    extern __shared__ float smem[];
    float* s_x_branch = smem;
    float* s_d_dt_raw = smem + d_inner;

    // Point to this param's packed data
    const float* my_d_scan = d_scan_output_packed + start * d_inner;
    const float* my_x_sorted = x_sorted_packed + start * d_model;
    // For checkpointing: saved_states uses ckpt_offsets
    const int ckpt_start = (checkpoint_interval > 1 && ckpt_offsets != nullptr)
                           ? ckpt_offsets[param_idx] : start;
    const float* my_saved_states = saved_states_packed + ckpt_start * d_inner * d_state;
    const float* my_saved_xb = saved_x_branch_packed + start * d_inner;
    const float* my_saved_z = saved_z_packed + start * d_inner;
    const float* my_saved_dt = saved_dt_packed + start * d_inner;
    float* my_d_x_sorted = d_x_sorted_packed + start * d_model;
    // Two-pass buffers for this param
    float* my_d_C_vals = d_C_vals_buf + start * d_state;
    float* my_d_B_vals = d_B_vals_buf + start * d_state;

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    const int half_d_state = d_state / 2;
    float freq[MAX_D_STATE / 2];
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[tid * half_d_state + p];

    float D_val = D_param[tid];

    float d_D_acc = 0.0f;
    float d_A_log_acc[MAX_D_STATE];
    float d_freq_acc[MAX_D_STATE / 2];
    float d_dt_proj_b_acc = 0.0f;
    for (int s = 0; s < d_state; s++) d_A_log_acc[s] = 0.0f;
    for (int p = 0; p < half_d_state; p++) d_freq_acc[p] = 0.0f;
    float d_dt_proj_W_row[MAX_D_INNER];
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    // Small per-thread accumulators (fit in registers)
    float d_in_proj_W_x_local[MAX_D_MODEL];
    float d_in_proj_W_z_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        d_in_proj_W_x_local[d] = 0.0f;
        d_in_proj_W_z_local[d] = 0.0f;
    }

    float dh[MAX_D_STATE];
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    const float* my_init = initial_states + param_idx * d_inner * d_state;

    if (checkpoint_interval <= 1) {
        // ════════════════════════════════════════════════════════
        //  ORIGINAL PATH: saved_states has all N states
        // ════════════════════════════════════════════════════════
        for (int step = N - 1; step >= 0; step--) {
            int i = reverse ? (N - 1 - step) : step;

            float d_out = my_d_scan[i * d_inner + tid];
            float x_val = my_saved_xb[i * d_inner + tid];
            float z_val = my_stream_load(&saved_z[i * d_inner + tid]);
            float dt_val = my_stream_load(&saved_dt[i * d_inner + tid]);

            s_x_branch[tid] = x_val;
            __syncthreads();

            float sig_z = 1.0f / (1.0f + expf(-z_val));
            float silu_z = z_val * sig_z;

            float y_val = 0.0f;
            for (int s = 0; s < d_state; s++) {
                float C_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                y_val += my_stream_load(&saved_states[(i * d_inner + tid) * d_state + s]) * C_val;
            }

            float d_y_val = d_out * silu_z;
            float d_silu_z = d_out * y_val;
            float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
            float d_x_from_D = d_out * D_val;
            d_D_acc += d_out * x_val;

            float d_x_from_C = 0.0f;
            for (int s = 0; s < d_state; s++) {
                float h_s = my_stream_load(&saved_states[(i * d_inner + tid) * d_state + s]);
                float C_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                dh[s] += d_y_val * C_val;
                float d_C_val = d_y_val * h_s;
                // Two-pass: warp reduce d_C_val across d_inner threads, write to buffer
                float d_C_reduced = warp_reduce_sum(d_C_val, d_inner, tid);
                if (tid == 0)
                    my_d_C_vals[i * d_state + s] = d_C_reduced;
                d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
            }

            float h_prev[MAX_D_STATE];
            if (step > 0) {
                int i_prev = reverse ? (N - step) : (step - 1);
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = my_stream_load(&saved_states[(i_prev * d_inner + tid) * d_state + s]);
            } else {
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = my_init[tid * d_state + s];
            }

            float dh_snap[MAX_D_STATE];
            for (int s = 0; s < d_state; s++) dh_snap[s] = dh[s];
            for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

            float d_dt_val = 0.0f;
            float d_x_from_scan = 0.0f;

            for (int s = 0; s < d_state; s++) {
                float half_dtA = dt_val * A[s] / 2.0f;
                float denom_val = 1.0f - half_dtA + 1e-8f;
                float A_bar = (1.0f + half_dtA) / denom_val;
                float B_val = 0.0f;
                for (int j = 0; j < d_inner; j++)
                    B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                float B_bar = dt_val * B_val;
                int pair_idx = s / 2;
                float cos_p, sin_p;
                FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                float h_rot;
                int partner;
                float sign;
                if (s % 2 == 0) {
                    partner = s + 1; sign = -1.0f;
                    h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
                } else {
                    partner = s - 1; sign = 1.0f;
                    h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
                }
                float d_h_s = dh_snap[s];
                float d_A_bar = d_h_s * h_rot;
                float d_h_rot = d_h_s * A_bar;
                float d_B_bar = d_h_s * x_val;
                d_x_from_scan += d_h_s * B_bar;
                d_dt_val += d_B_bar * B_val;
                float d_B_val = d_B_bar * dt_val;
                // Two-pass: warp reduce d_B_val across d_inner threads, write to buffer
                float d_B_reduced = warp_reduce_sum(d_B_val, d_inner, tid);
                if (tid == 0)
                    my_d_B_vals[i * d_state + s] = d_B_reduced;
                d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];
                float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
                d_dt_val += d_half_dtA * A[s] / 2.0f;
                float d_A_s = d_half_dtA * dt_val / 2.0f;
                d_A_log_acc[s] += d_A_s * A[s];
                float d_h_prev_s = d_h_rot * cos_p;
                float d_h_prev_partner = d_h_rot * sign * sin_p;
                float d_cos = d_h_rot * h_prev[s];
                float d_sin = d_h_rot * sign * h_prev[partner];
                d_dt_val += (-sin_p * freq[pair_idx]) * d_cos + (cos_p * freq[pair_idx]) * d_sin;
                d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;
                dh[s] += d_h_prev_s;
                dh[partner] += d_h_prev_partner;
            }

            float dt_raw = dt_proj_b[tid];
            for (int j = 0; j < d_inner; j++)
                dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
            float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
            float d_dt_raw = d_dt_val * sig_dt;

            d_dt_proj_b_acc += d_dt_raw;
            for (int j = 0; j < d_inner; j++)
                d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];

            s_d_dt_raw[tid] = d_dt_raw;
            __syncthreads();
            float d_x_from_dt = 0.0f;
            for (int t = 0; t < d_inner; t++)
                d_x_from_dt += s_d_dt_raw[t] * dt_proj_W[t * d_inner + tid];

            float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;

            for (int d = 0; d < d_model; d++) {
                float inp = my_x_sorted[i * d_model + d];
                d_in_proj_W_x_local[d] += d_x_val * inp;
                d_in_proj_W_z_local[d] += d_z_val * inp;
                atomicAdd(&my_d_x_sorted[i * d_model + d],
                          d_x_val * in_proj_W[tid * d_model + d] +
                          d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
            }
            __syncthreads();
        }
    } else {
        // ════════════════════════════════════════════════════════
        //  CHECKPOINTED PATH: recompute states from checkpoints
        // ════════════════════════════════════════════════════════
        int num_segments = (N + checkpoint_interval - 1) / checkpoint_interval;
        float seg_h[(MAX_CKPT_INTERVAL + 1) * MAX_D_STATE];

        for (int seg = num_segments - 1; seg >= 0; seg--) {
            int seg_start = seg * checkpoint_interval;
            int seg_end = (seg_start + checkpoint_interval < N)
                          ? seg_start + checkpoint_interval : N;
            int seg_len = seg_end - seg_start;

            // Load checkpoint input state
            if (seg == 0) {
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = my_init[tid * d_state + s];
            } else {
                int ckpt_idx = seg - 1;
                for (int s = 0; s < d_state; s++)
                    seg_h[s] = my_stream_load(&saved_states[(ckpt_idx * d_inner + tid) * d_state + s]);
            }

            // Forward-recompute segment states
            for (int local = 0; local < seg_len; local++) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float x_val = my_saved_xb[i * d_inner + tid];
                float dt_val = my_stream_load(&saved_dt[i * d_inner + tid]);
                s_x_branch[tid] = x_val;
                __syncthreads();
                float h_snap_r[MAX_D_STATE];
                for (int s = 0; s < d_state; s++)
                    h_snap_r[s] = seg_h[local * MAX_D_STATE + s];
                for (int s = 0; s < d_state; s++) {
                    float A_bar = (1.0f + dt_val * A[s] / 2.0f)
                                  / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
                    float B_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dt_val * B_val;
                    int pair_idx = s / 2;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    if (s % 2 == 0)
                        h_rot = h_snap_r[s] * cos_p - h_snap_r[s + 1] * sin_p;
                    else
                        h_rot = h_snap_r[s] * cos_p + h_snap_r[s - 1] * sin_p;
                    seg_h[(local + 1) * MAX_D_STATE + s] = A_bar * h_rot + B_bar * x_val;
                }
                __syncthreads();
            }

            // Backward through segment
            for (int local = seg_len - 1; local >= 0; local--) {
                int step = seg_start + local;
                int i = reverse ? (N - 1 - step) : step;
                float d_out = my_d_scan[i * d_inner + tid];
                float x_val = my_saved_xb[i * d_inner + tid];
                float z_val = my_stream_load(&saved_z[i * d_inner + tid]);
                float dt_val = my_stream_load(&saved_dt[i * d_inner + tid]);
                s_x_branch[tid] = x_val;
                __syncthreads();
                float sig_z = 1.0f / (1.0f + expf(-z_val));
                float silu_z = z_val * sig_z;
                float y_val = 0.0f;
                for (int s = 0; s < d_state; s++) {
                    float C_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                    y_val += seg_h[(local + 1) * MAX_D_STATE + s] * C_val;
                }
                float d_y_val = d_out * silu_z;
                float d_silu_z = d_out * y_val;
                float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
                float d_x_from_D = d_out * D_val;
                d_D_acc += d_out * x_val;
                float d_x_from_C = 0.0f;
                for (int s = 0; s < d_state; s++) {
                    float h_s = seg_h[(local + 1) * MAX_D_STATE + s];
                    float C_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
                    dh[s] += d_y_val * C_val;
                    float d_C_val = d_y_val * h_s;
                    // Two-pass: warp reduce d_C_val across d_inner threads, write to buffer
                    float d_C_reduced = warp_reduce_sum(d_C_val, d_inner, tid);
                    if (tid == 0)
                        my_d_C_vals[i * d_state + s] = d_C_reduced;
                    d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
                }
                float h_prev[MAX_D_STATE];
                for (int s = 0; s < d_state; s++)
                    h_prev[s] = seg_h[local * MAX_D_STATE + s];
                float dh_snap[MAX_D_STATE];
                for (int s = 0; s < d_state; s++) dh_snap[s] = dh[s];
                for (int s = 0; s < d_state; s++) dh[s] = 0.0f;
                float d_dt_val = 0.0f;
                float d_x_from_scan = 0.0f;
                for (int s = 0; s < d_state; s++) {
                    float half_dtA = dt_val * A[s] / 2.0f;
                    float denom_val = 1.0f - half_dtA + 1e-8f;
                    float A_bar = (1.0f + half_dtA) / denom_val;
                    float B_val = 0.0f;
                    for (int j = 0; j < d_inner; j++)
                        B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
                    float B_bar = dt_val * B_val;
                    int pair_idx = s / 2;
                    float cos_p, sin_p;
                    FAST_SINCOSF(dt_val * freq[pair_idx], &sin_p, &cos_p);
                    float h_rot;
                    int partner;
                    float sign;
                    if (s % 2 == 0) {
                        partner = s + 1; sign = -1.0f;
                        h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
                    } else {
                        partner = s - 1; sign = 1.0f;
                        h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
                    }
                    float d_h_s = dh_snap[s];
                    float d_A_bar = d_h_s * h_rot;
                    float d_h_rot = d_h_s * A_bar;
                    float d_B_bar = d_h_s * x_val;
                    d_x_from_scan += d_h_s * B_bar;
                    d_dt_val += d_B_bar * B_val;
                    float d_B_val = d_B_bar * dt_val;
                    // Two-pass: warp reduce d_B_val across d_inner threads, write to buffer
                    float d_B_reduced = warp_reduce_sum(d_B_val, d_inner, tid);
                    if (tid == 0)
                        my_d_B_vals[i * d_state + s] = d_B_reduced;
                    d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];
                    float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
                    d_dt_val += d_half_dtA * A[s] / 2.0f;
                    float d_A_s = d_half_dtA * dt_val / 2.0f;
                    d_A_log_acc[s] += d_A_s * A[s];
                    float d_h_prev_s = d_h_rot * cos_p;
                    float d_h_prev_partner = d_h_rot * sign * sin_p;
                    float d_cos = d_h_rot * h_prev[s];
                    float d_sin = d_h_rot * sign * h_prev[partner];
                    d_dt_val += (-sin_p * freq[pair_idx]) * d_cos + (cos_p * freq[pair_idx]) * d_sin;
                    d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;
                    dh[s] += d_h_prev_s;
                    dh[partner] += d_h_prev_partner;
                }
                float dt_raw = dt_proj_b[tid];
                for (int j = 0; j < d_inner; j++)
                    dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
                float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
                float d_dt_raw = d_dt_val * sig_dt;
                d_dt_proj_b_acc += d_dt_raw;
                for (int j = 0; j < d_inner; j++)
                    d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];
                s_d_dt_raw[tid] = d_dt_raw;
                __syncthreads();
                float d_x_from_dt = 0.0f;
                for (int t = 0; t < d_inner; t++)
                    d_x_from_dt += s_d_dt_raw[t] * dt_proj_W[t * d_inner + tid];
                float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;
                for (int d = 0; d < d_model; d++) {
                    float inp = my_x_sorted[i * d_model + d];
                    d_in_proj_W_x_local[d] += d_x_val * inp;
                    d_in_proj_W_z_local[d] += d_z_val * inp;
                    atomicAdd(&my_d_x_sorted[i * d_model + d],
                              d_x_val * in_proj_W[tid * d_model + d] +
                              d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
                }
                __syncthreads();
            }
        }
    }

    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    for (int j = 0; j < d_inner; j++)
        atomicAdd(&d_dt_proj_W[tid * d_inner + j], d_dt_proj_W_row[j]);
    for (int s = 0; s < d_state; s++)
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
    for (int p = 0; p < half_d_state; p++)
        atomicAdd(&d_rope_freq[tid * half_d_state + p], d_freq_acc[p]);
    // Two-pass: C_proj_W and B_proj_W gradients are computed via GEMM in the launcher
    // (d_C_vals_buf and d_B_vals_buf were filled per-timestep above)
    // Write per-thread in_proj_W gradients
    for (int d = 0; d < d_model; d++) {
        atomicAdd(&d_in_proj_W[tid * d_model + d], d_in_proj_W_x_local[d]);
        atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_in_proj_W_z_local[d]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Input Projection Backward
//
//  Forward: x[idx, d] = proj_W[d, 0] * g + proj_W[d, 1] * s + proj_b[d]
//  Backward: d_proj_W, d_proj_b from d_x
//
//  One thread per element.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
__global__ void input_proj_backward_kernel(
    const float* __restrict__ d_x,           // [N, d_model]
    const scalar_t* __restrict__ grad,       // [N]
    const scalar_t* __restrict__ sharpness,  // [N]
    float* __restrict__ d_proj_W,            // [d_model, 2] — atomicAdd
    float* __restrict__ d_proj_b,            // [d_model] — atomicAdd
    const int N,
    const int d_model
) {
    // Shared memory for per-block reduction: d_model*2 for d_proj_W + d_model for d_proj_b
    extern __shared__ float smem[];
    float* s_d_proj_W = smem;           // [d_model * 2]
    float* s_d_proj_b = smem + d_model * 2;  // [d_model]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Initialize shared memory accumulators
    for (int i = tid; i < d_model * 2; i += block_size) s_d_proj_W[i] = 0.0f;
    for (int i = tid; i < d_model; i += block_size) s_d_proj_b[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        float g = static_cast<float>(grad[idx]);
        float s = static_cast<float>(sharpness[idx]);

        for (int d = 0; d < d_model; d++) {
            float d_xd = d_x[idx * d_model + d];
            atomicAdd(&s_d_proj_W[d * 2 + 0], d_xd * g);
            atomicAdd(&s_d_proj_W[d * 2 + 1], d_xd * s);
            atomicAdd(&s_d_proj_b[d], d_xd);
        }
    }
    __syncthreads();

    // Block-level reduction: one thread per shared memory element writes to global
    for (int i = tid; i < d_model * 2; i += block_size) {
        if (s_d_proj_W[i] != 0.0f)
            atomicAdd(&d_proj_W[i], s_d_proj_W[i]);
    }
    for (int i = tid; i < d_model; i += block_size) {
        if (s_d_proj_b[i] != 0.0f)
            atomicAdd(&d_proj_b[i], s_d_proj_b[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: GRU Backward
//
//  Forward:
//    xh = cat(x, h_old)
//    z = sigmoid(Wz @ xh + bz)
//    r = sigmoid(Wr @ xh + br)
//    xrh = cat(x, r * h_old)
//    h_tilde = tanh(Wh @ xrh + bh)
//    h_new = (1-z) * h_old + z * h_tilde
//
//  Each thread handles one element. Accumulates gradients via atomicAdd.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void gru_backward_kernel(
    const float* __restrict__ d_h_new,       // [N, gru_hidden]
    const float* __restrict__ gru_input,     // [N, input_dim] (saved from forward)
    const float* __restrict__ h_old,         // [N, gru_hidden] (saved from forward)
    // Saved gate values from forward
    const float* __restrict__ z_gate,        // [N, gru_hidden]
    const float* __restrict__ r_gate,        // [N, gru_hidden]
    const float* __restrict__ h_tilde,       // [N, gru_hidden]
    // Weights
    const float* __restrict__ Wz,            // [gru_hidden, input_dim + gru_hidden]
    const float* __restrict__ Wr,
    const float* __restrict__ Wh,
    // Gradient outputs (atomicAdd)
    float* __restrict__ d_Wz,
    float* __restrict__ d_bz,               // [gru_hidden]
    float* __restrict__ d_Wr,
    float* __restrict__ d_br,
    float* __restrict__ d_Wh,
    float* __restrict__ d_bh,
    float* __restrict__ d_gru_input,         // [N, input_dim]
    // Dims
    const int N, const int input_dim, const int gru_hidden
) {
    extern __shared__ float smem[];
    // Layout: s_d_Wz[gru_hidden*total_dim] + s_d_Wr[same] + s_d_Wh[same]
    //       + s_d_bz[gru_hidden] + s_d_br[gru_hidden] + s_d_bh[gru_hidden]
    const int total_dim = input_dim + gru_hidden;
    const int w_size = gru_hidden * total_dim;
    float* s_d_Wz = smem;
    float* s_d_Wr = s_d_Wz + w_size;
    float* s_d_Wh = s_d_Wr + w_size;
    float* s_d_bz = s_d_Wh + w_size;
    float* s_d_br = s_d_bz + gru_hidden;
    float* s_d_bh = s_d_br + gru_hidden;
    const int smem_total = 3 * w_size + 3 * gru_hidden;

    const int tid = threadIdx.x;

    // Zero shared accumulators
    for (int i = tid; i < smem_total; i += blockDim.x) smem[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        for (int gh = 0; gh < gru_hidden; gh++) {
            float d_h = d_h_new[idx * gru_hidden + gh];
            float z_val = z_gate[idx * gru_hidden + gh];
            float r_val = r_gate[idx * gru_hidden + gh];
            float ht_val = h_tilde[idx * gru_hidden + gh];
            float h_old_val = h_old[idx * gru_hidden + gh];

            float d_z = d_h * (ht_val - h_old_val);
            float d_h_tilde_val = d_h * z_val;

            float d_tanh_input = d_h_tilde_val * (1.0f - ht_val * ht_val);
            float d_z_input = d_z * z_val * (1.0f - z_val);

            // Accumulate to shared memory instead of global
            atomicAdd(&s_d_bh[gh], d_tanh_input);
            for (int j = 0; j < input_dim; j++) {
                float xj = gru_input[idx * input_dim + j];
                atomicAdd(&s_d_Wh[gh * total_dim + j], d_tanh_input * xj);
            }
            for (int j = 0; j < gru_hidden; j++) {
                float rh;
                if (j == gh)
                    rh = r_val * h_old_val;
                else
                    rh = r_gate[idx * gru_hidden + j] * h_old[idx * gru_hidden + j];
                atomicAdd(&s_d_Wh[gh * total_dim + input_dim + j], d_tanh_input * rh);
            }

            atomicAdd(&s_d_bz[gh], d_z_input);
            for (int j = 0; j < total_dim; j++) {
                float xh_j;
                if (j < input_dim)
                    xh_j = gru_input[idx * input_dim + j];
                else
                    xh_j = h_old[idx * gru_hidden + (j - input_dim)];
                atomicAdd(&s_d_Wz[gh * total_dim + j], d_z_input * xh_j);
            }

            for (int j = 0; j < gru_hidden; j++) {
                float d_r_j = d_tanh_input * Wh[gh * total_dim + input_dim + j]
                            * h_old[idx * gru_hidden + j];
                float r_j = r_gate[idx * gru_hidden + j];
                float d_r_j_input = d_r_j * r_j * (1.0f - r_j);

                atomicAdd(&s_d_br[j], d_r_j_input);
                for (int k = 0; k < total_dim; k++) {
                    float xh_k;
                    if (k < input_dim)
                        xh_k = gru_input[idx * input_dim + k];
                    else
                        xh_k = h_old[idx * gru_hidden + (k - input_dim)];
                    atomicAdd(&s_d_Wr[j * total_dim + k], d_r_j_input * xh_k);
                }
                for (int k = 0; k < input_dim; k++) {
                    atomicAdd(&d_gru_input[idx * input_dim + k],
                              d_r_j_input * Wr[j * total_dim + k]);
                }
            }

            for (int j = 0; j < input_dim; j++) {
                float d_input_j = d_z_input * Wz[gh * total_dim + j]
                                + d_tanh_input * Wh[gh * total_dim + j];
                atomicAdd(&d_gru_input[idx * input_dim + j], d_input_j);
            }
        }
    }
    __syncthreads();

    // Write block sums to global
    for (int i = tid; i < w_size; i += blockDim.x) {
        if (s_d_Wz[i] != 0.0f) atomicAdd(&d_Wz[i], s_d_Wz[i]);
        if (s_d_Wr[i] != 0.0f) atomicAdd(&d_Wr[i], s_d_Wr[i]);
        if (s_d_Wh[i] != 0.0f) atomicAdd(&d_Wh[i], s_d_Wh[i]);
    }
    for (int i = tid; i < gru_hidden; i += blockDim.x) {
        if (s_d_bz[i] != 0.0f) atomicAdd(&d_bz[i], s_d_bz[i]);
        if (s_d_br[i] != 0.0f) atomicAdd(&d_br[i], s_d_br[i]);
        if (s_d_bh[i] != 0.0f) atomicAdd(&d_bh[i], s_d_bh[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: Expert + PEER Backward (Soft Routing)
//
//  Forward (bilevel, soft routing):
//    For each head:
//      query = peer_query_W @ peer_input   [N, d_model]
//      q_a = query[:, :d/2], q_b = query[:, d/2:]
//      scores_a = q_a @ keys_A.T           [N, pk_dim]
//      scores_b = q_b @ keys_B.T           [N, pk_dim]
//      top_a, top_b = topk(scores_a), topk(scores_b)
//      soft_a = softmax(top_a*10), soft_b = softmax(top_b*10)
//      routing = outer(soft_a, soft_b)     [N, topk*topk]
//      expert_indices = outer_idx(top_a_idx, top_b_idx)
//      out = sum(routing * expert(g))      [N, 1]
//
//  This kernel handles backward through expert eval + soft routing.
//  One thread per element.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void expert_peer_backward_kernel(
    const float* __restrict__ d_expert_out,    // [N, 1] (rescale * d_smart_grad)
    const float* __restrict__ grad_vals,       // [N] gradient values
    // Saved from bilevel forward
    const int* __restrict__ expert_indices,    // [N, num_heads, topk*topk]
    const float* __restrict__ routing_weights, // [N, num_heads, topk*topk]
    const float* __restrict__ saved_z_hidden,  // [N, num_heads, topk*topk, expert_hidden]
    // PEER query related
    const float* __restrict__ saved_peer_input, // [N, peer_input_dim]
    const float* __restrict__ peer_query_Ws,   // [num_heads, d_model, peer_input_dim]
    const float* __restrict__ prod_keys_A,     // [num_heads, pk_dim, d_model/2]
    const float* __restrict__ prod_keys_B,     // [num_heads, pk_dim, d_model/2]
    const float* __restrict__ saved_scores_a,  // [N, num_heads, pk_dim]
    const float* __restrict__ saved_scores_b,  // [N, num_heads, pk_dim]
    const int* __restrict__ saved_top_a_idx,   // [N, num_heads, topk]
    const int* __restrict__ saved_top_b_idx,   // [N, num_heads, topk]
    const float* __restrict__ saved_soft_a,    // [N, num_heads, topk]
    const float* __restrict__ saved_soft_b,    // [N, num_heads, topk]
    // Expert weights
    const float* __restrict__ expert_W1,       // [num_experts, expert_hidden, 1]
    const float* __restrict__ expert_W2,       // [num_experts, 1, expert_hidden]
    const float* __restrict__ expert_b2_in,    // [num_experts] — read-only bias
    // Gradient outputs
    float* __restrict__ d_expert_W1,           // [num_experts, expert_hidden]
    float* __restrict__ d_expert_b1,           // [num_experts, expert_hidden]
    float* __restrict__ d_expert_W2,           // [num_experts, expert_hidden]
    float* __restrict__ d_expert_b2,           // [num_experts]
    float* __restrict__ d_peer_query_Ws,       // [num_heads, d_model, peer_input_dim]
    float* __restrict__ d_prod_keys_A,         // [num_heads, pk_dim, d_model/2]
    float* __restrict__ d_prod_keys_B,         // [num_heads, pk_dim, d_model/2]
    float* __restrict__ d_peer_input,          // [N, peer_input_dim]
    // Dims
    const int N, const int num_heads, const int topk, const int num_active,
    const int d_model, const int pk_dim, const int expert_hidden,
    const int peer_input_dim, const int num_experts
) {
    // Shared memory accumulators for expert + routing weight gradients
    extern __shared__ float smem[];
    float* s_d_expert_W1 = smem;
    float* s_d_expert_b1 = s_d_expert_W1 + num_experts * expert_hidden;
    float* s_d_expert_W2 = s_d_expert_b1 + num_experts * expert_hidden;
    float* s_d_expert_b2 = s_d_expert_W2 + num_experts * expert_hidden;
    int total_expert_smem = 3 * num_experts * expert_hidden + num_experts;
    // Routing weight gradients in smem
    int half_d_smem = d_model / 2;
    int pqw_size = num_heads * d_model * peer_input_dim;
    int pka_size = num_heads * pk_dim * half_d_smem;
    int pkb_size = pka_size;
    float* s_d_peer_query_Ws = smem + total_expert_smem;
    float* s_d_prod_keys_A = s_d_peer_query_Ws + pqw_size;
    float* s_d_prod_keys_B = s_d_prod_keys_A + pka_size;
    int total_smem = total_expert_smem + pqw_size + pka_size + pkb_size;

    // Zero shared accumulators cooperatively
    for (int i = threadIdx.x; i < total_smem; i += blockDim.x)
        smem[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
    float d_out = d_expert_out[idx];
    float g_val = grad_vals[idx];
    int half_d = d_model / 2;

    for (int h = 0; h < num_heads; h++) {
        float d_head_out = d_out / (float)num_heads;

        // First pass: compute expert outputs and softmax backward dot products
        float dot_a[MAX_TOPK] = {};
        float dot_b[MAX_TOPK] = {};

        for (int k = 0; k < num_active; k++) {
            int a_local = k / topk;
            int b_local = k % topk;
            int ei = expert_indices[(idx * num_heads + h) * num_active + k];
            float rw = routing_weights[(idx * num_heads + h) * num_active + k];

            // Recompute out_k
            float out_k = expert_b2_in[ei];
            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                out_k += expert_W2[ei * expert_hidden + eh] * z_val;
            }
            float d_rw = d_head_out * out_k;

            float sa = saved_soft_a[(idx * num_heads + h) * topk + a_local];
            float sb = saved_soft_b[(idx * num_heads + h) * topk + b_local];
            // Accumulate per-sub-key dot products for full softmax backward
            dot_a[a_local] += (d_rw * sb) * sa;
            dot_b[b_local] += (d_rw * sa) * sb;
        }

        // Second pass: compute actual gradients with full softmax backward
        for (int k = 0; k < num_active; k++) {
            int a_local = k / topk;
            int b_local = k % topk;
            int ei = expert_indices[(idx * num_heads + h) * num_active + k];
            float rw = routing_weights[(idx * num_heads + h) * num_active + k];

            // Recompute out_k
            float out_k = expert_b2_in[ei];
            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                out_k += expert_W2[ei * expert_hidden + eh] * z_val;
            }

            float d_rw = d_head_out * out_k;
            float d_out_k = d_head_out * rw;

            // Backward through expert MLP (accumulate in shared memory)
            atomicAdd(&s_d_expert_b2[ei], d_out_k);

            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                atomicAdd(&s_d_expert_W2[ei * expert_hidden + eh], d_out_k * z_val);
                float d_z = d_out_k * expert_W2[ei * expert_hidden + eh];
                float d_pre_relu = (z_val > 0.0f) ? d_z : 0.0f;
                atomicAdd(&s_d_expert_W1[ei * expert_hidden + eh], d_pre_relu * g_val);
                atomicAdd(&s_d_expert_b1[ei * expert_hidden + eh], d_pre_relu);
            }

            // Backward through soft routing with FULL softmax backward
            float sa = saved_soft_a[(idx * num_heads + h) * topk + a_local];
            float sb = saved_soft_b[(idx * num_heads + h) * topk + b_local];

            float d_soft_a_val = d_rw * sb;
            float d_score_a = 10.0f * sa * (d_soft_a_val - dot_a[a_local]);

            float d_soft_b_val = d_rw * sa;
            float d_score_b = 10.0f * sb * (d_soft_b_val - dot_b[b_local]);

            int a_key_idx = saved_top_a_idx[(idx * num_heads + h) * topk + a_local];
            int b_key_idx = saved_top_b_idx[(idx * num_heads + h) * topk + b_local];

            for (int d = 0; d < half_d; d++) {
                float q_a_d = 0.0f;
                float q_b_d = 0.0f;
                for (int j = 0; j < peer_input_dim; j++) {
                    float pi_j = saved_peer_input[idx * peer_input_dim + j];
                    q_a_d += peer_query_Ws[(h * d_model + d) * peer_input_dim + j] * pi_j;
                    q_b_d += peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j] * pi_j;
                }

                atomicAdd(&s_d_prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d], d_score_a * q_a_d);
                atomicAdd(&s_d_prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d], d_score_b * q_b_d);

                float d_q_a_d = d_score_a * prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d];
                float d_q_b_d = d_score_b * prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d];

                for (int j = 0; j < peer_input_dim; j++) {
                    float pi_j = saved_peer_input[idx * peer_input_dim + j];
                    atomicAdd(&s_d_peer_query_Ws[(h * d_model + d) * peer_input_dim + j], d_q_a_d * pi_j);
                    atomicAdd(&s_d_peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j], d_q_b_d * pi_j);
                    atomicAdd(&d_peer_input[idx * peer_input_dim + j],
                              d_q_a_d * peer_query_Ws[(h * d_model + d) * peer_input_dim + j] +
                              d_q_b_d * peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j]);
                }
            }
        }
    }
    } // end if (idx < N)

    // Flush shared memory gradient accumulators to global memory
    __syncthreads();

    // Flush expert weight gradients
    for (int i = threadIdx.x; i < total_expert_smem; i += blockDim.x) {
        if (smem[i] != 0.0f) {
            if (i < num_experts * expert_hidden)
                atomicAdd(&d_expert_W1[i], smem[i]);
            else if (i < 2 * num_experts * expert_hidden)
                atomicAdd(&d_expert_b1[i - num_experts * expert_hidden], smem[i]);
            else if (i < 3 * num_experts * expert_hidden)
                atomicAdd(&d_expert_W2[i - 2 * num_experts * expert_hidden], smem[i]);
            else
                atomicAdd(&d_expert_b2[i - 3 * num_experts * expert_hidden], smem[i]);
        }
    }

    // Flush routing weight gradients (peer_query_Ws, prod_keys_A, prod_keys_B)
    for (int i = threadIdx.x; i < pqw_size; i += blockDim.x) {
        if (s_d_peer_query_Ws[i] != 0.0f)
            atomicAdd(&d_peer_query_Ws[i], s_d_peer_query_Ws[i]);
    }
    for (int i = threadIdx.x; i < pka_size; i += blockDim.x) {
        if (s_d_prod_keys_A[i] != 0.0f)
            atomicAdd(&d_prod_keys_A[i], s_d_prod_keys_A[i]);
    }
    for (int i = threadIdx.x; i < pkb_size; i += blockDim.x) {
        if (s_d_prod_keys_B[i] != 0.0f)
            atomicAdd(&d_prod_keys_B[i], s_d_prod_keys_B[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Out-Projection Backward Kernel
//
//  Forward: context[i, d] = sum_j(out_proj_W[d, j] * scan_out[i, j])
//  Backward: d_out_proj_W, d_scan_out from d_context
//
//  One thread per element.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 8)
__global__ void out_proj_backward_kernel(
    const float* __restrict__ d_context,       // [N, d_model]
    const float* __restrict__ scan_out,        // [N, d_inner]
    const float* __restrict__ out_proj_W,      // [d_model, d_inner]
    float* __restrict__ d_out_proj_W,          // [d_model, d_inner] — atomicAdd
    float* __restrict__ d_scan_out,            // [N, d_inner]
    const int N, const int d_model, const int d_inner
) {
    extern __shared__ float smem[];
    float* s_d_out_proj_W = smem;  // [d_model * d_inner]

    const int tid = threadIdx.x;
    const int op_size = d_model * d_inner;

    // Zero shared accumulators
    for (int i = tid; i < op_size; i += blockDim.x) s_d_out_proj_W[i] = 0.0f;
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        for (int j = 0; j < d_inner; j++) {
            float d_scan_j = 0.0f;
            float so_j = scan_out[idx * d_inner + j];
            for (int d = 0; d < d_model; d++) {
                float d_ctx = d_context[idx * d_model + d];
                d_scan_j += d_ctx * out_proj_W[d * d_inner + j];
                atomicAdd(&s_d_out_proj_W[d * d_inner + j], d_ctx * so_j);
            }
            d_scan_out[idx * d_inner + j] = d_scan_j;
        }
    }
    __syncthreads();

    // Write block sums to global
    for (int i = tid; i < op_size; i += blockDim.x) {
        if (s_d_out_proj_W[i] != 0.0f)
            atomicAdd(&d_out_proj_W[i], s_d_out_proj_W[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Pre-allocated Workspace for Bilevel Operations
//
//  Avoids per-step tensor allocations for temporary buffers.
//  Sizes grow to accommodate the largest N seen; never shrink.
// ═══════════════════════════════════════════════════════════════════════

namespace {
struct BilevelWorkspace {
    // Forward-save precompute buffers
    torch::Tensor pre_B;             // [max_N, d_state]
    torch::Tensor pre_C;             // [max_N, d_state]
    torch::Tensor x_proj;            // [max_N, d_model]
    torch::Tensor sort_keys;         // [max_N]
    torch::Tensor sort_idx;          // [max_N] (int32)
    // Backward intermediates
    torch::Tensor d_peer_input;      // [max_N, max_peer_input_dim]
    torch::Tensor d_gru_input;       // [max_N, max_gru_input_dim]
    torch::Tensor d_fwd_ctx;         // [max_N, d_model]
    torch::Tensor d_bwd_ctx;         // [max_N, d_model]
    torch::Tensor d_fwd_scan_out;    // [max_N, d_inner]
    torch::Tensor d_bwd_scan_out;    // [max_N, d_inner]
    torch::Tensor d_x_sorted_fwd;    // [max_N, d_model]
    torch::Tensor d_x_sorted_bwd;    // [max_N, d_model]
    torch::Tensor unsort_idx;        // [max_N] (int64)
    // Batched-specific
    torch::Tensor x_sorted_rev;      // [max_total_N, d_model]
    torch::Tensor d_x_sorted_fwd_bat;// [max_total_N, d_model]
    torch::Tensor d_x_sorted_bwd_bat;// [max_total_N, d_model]
    int max_N = 0;
    int max_total_N = 0;
    int d_model = 0;
    int d_inner = 0;
    int d_state = 0;
    int peer_input_dim = 0;
    int gru_input_dim = 0;

    void ensure_fwd_save(int N, int dm, int di, int ds, torch::Device dev) {
        bool preBC_ok = pre_B.defined() && pre_B.size(0) >= N;
        if (N <= max_N && dm == d_model && di == d_inner && ds == d_state
            && preBC_ok) return;
        int alloc_N = std::max(N, max_N);
        // pre_B/pre_C are shared with ensure_batched — size to max of both
        int alloc_preBC = std::max(alloc_N, max_total_N);
        auto fo = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
        auto io = torch::TensorOptions().device(dev).dtype(torch::kInt32);
        pre_B = torch::empty({alloc_preBC, ds}, fo);
        pre_C = torch::empty({alloc_preBC, ds}, fo);
        x_proj = torch::empty({alloc_N, dm}, fo);
        sort_keys = torch::empty({alloc_N}, fo);
        sort_idx = torch::empty({alloc_N}, io);
        max_N = alloc_N;
        d_model = dm;
        d_inner = di;
        d_state = ds;
    }

    void ensure_backward(int N, int dm, int di, int ds, int pid, int gid,
                         torch::Device dev) {
        auto fo = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
        auto lo = torch::TensorOptions().device(dev).dtype(torch::kLong);
        bool need_realloc = (N > max_N || dm != d_model || di != d_inner
                             || ds != d_state || pid != peer_input_dim
                             || gid != gru_input_dim);
        if (!need_realloc) return;
        int alloc_N = std::max(N, max_N);
        d_peer_input = torch::empty({alloc_N, pid}, fo);
        d_gru_input = torch::empty({alloc_N, gid}, fo);
        d_fwd_ctx = torch::empty({alloc_N, dm}, fo);
        d_bwd_ctx = torch::empty({alloc_N, dm}, fo);
        d_fwd_scan_out = torch::empty({alloc_N, di}, fo);
        d_bwd_scan_out = torch::empty({alloc_N, di}, fo);
        d_x_sorted_fwd = torch::empty({alloc_N, dm}, fo);
        d_x_sorted_bwd = torch::empty({alloc_N, dm}, fo);
        unsort_idx = torch::empty({alloc_N}, lo);
        max_N = alloc_N;
        d_model = dm;
        d_inner = di;
        d_state = ds;
        peer_input_dim = pid;
        gru_input_dim = gid;
    }

    void ensure_batched(int total_N, int dm, int di, int ds, torch::Device dev) {
        bool preBC_ok = pre_B.defined() && pre_B.size(0) >= total_N;
        if (total_N <= max_total_N && dm == d_model && di == d_inner
            && ds == d_state && preBC_ok) return;
        int alloc_N = std::max(total_N, max_total_N);
        // pre_B/pre_C are shared with ensure_fwd_save — size to max of both
        int alloc_preBC = std::max(alloc_N, max_N);
        auto fo = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
        pre_B = torch::empty({alloc_preBC, ds}, fo);
        pre_C = torch::empty({alloc_preBC, ds}, fo);
        x_sorted_rev = torch::empty({alloc_N, dm}, fo);
        d_x_sorted_fwd_bat = torch::empty({alloc_N, dm}, fo);
        d_x_sorted_bwd_bat = torch::empty({alloc_N, dm}, fo);
        max_total_N = alloc_N;
        d_model = dm;
        d_inner = di;
        d_state = ds;
    }
};
static thread_local BilevelWorkspace g_bilevel_ws;
} // namespace


// ═══════════════════════════════════════════════════════════════════════
//  Launcher: Full Bilevel Backward Pass
//
//  Given d_smart_grad, computes gradients w.r.t. all meta-net parameters.
//  Requires saved intermediates from the bilevel forward pass.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_bilevel_fwd_save(
    torch::Tensor grad,              // [N]
    torch::Tensor sharpness,         // [N]
    // Input proj weights
    torch::Tensor input_proj_W,      // [d_model, 2]
    torch::Tensor input_proj_b,      // [d_model]
    // Mamba forward weights
    torch::Tensor mamba_fwd_in_proj,
    torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b,
    torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj,
    torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D,
    torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    // Mamba backward weights
    torch::Tensor mamba_bwd_in_proj,
    torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b,
    torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj,
    torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D,
    torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    // Dims
    int d_model, int d_state, int d_inner,
    // Outputs: scan outputs + saved states
    torch::Tensor fwd_scan_out,
    torch::Tensor bwd_scan_out,
    torch::Tensor fwd_final_state,
    torch::Tensor bwd_final_state,
    torch::Tensor fwd_saved_states,
    torch::Tensor fwd_saved_x_branch,
    torch::Tensor fwd_saved_z,
    torch::Tensor fwd_saved_dt,
    torch::Tensor bwd_saved_states,
    torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z,
    torch::Tensor bwd_saved_dt,
    // Sort-related
    torch::Tensor x_sorted,          // [N, d_model] — sorted input (computed here)
    torch::Tensor sort_indices,       // [N] — sort indices (computed here)
    // Mamba initial states (for exact match with forward_for_bilevel)
    torch::Tensor fwd_initial_state,  // [d_inner, d_state] or empty
    torch::Tensor bwd_initial_state,  // [d_inner, d_state] or empty
    int checkpoint_interval            // 0 = save every step
) {
    const int N = grad.numel();
    if (N == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");
    if (checkpoint_interval > 1) {
        TORCH_CHECK(checkpoint_interval <= MAX_CKPT_INTERVAL,
            "checkpoint_interval (", checkpoint_interval, ") exceeds MAX_CKPT_INTERVAL (", MAX_CKPT_INTERVAL, ")");
    }

    auto dev = grad.device();

    // Pre-allocate workspace (reused across steps)
    g_bilevel_ws.ensure_fwd_save(N, d_model, d_inner, d_state, dev);

    // Step 1: Input projection + sort
    {
        auto g_f = grad.to(torch::kFloat32).reshape(-1);
        auto s_f = sharpness.to(torch::kFloat32).reshape(-1);
        auto inp = torch::stack({g_f, s_f}, 1);  // [N, 2]
        auto x_proj = torch::addmm(input_proj_b, inp, input_proj_W.t());

        auto sort_keys = g_f.abs();
        auto sorted = sort_keys.argsort();
        sort_indices.copy_(sorted.to(torch::kInt32));
        x_sorted.copy_(x_proj.index_select(0, sorted));
    }

    int ckpt_int = (checkpoint_interval > 1) ? checkpoint_interval : 0;

    if (N >= PSCAN_THRESHOLD) {
        // ═══ Parallel scan path: precompute + Blelloch parallel prefix scan ═══
        auto pre_B_fwd = g_bilevel_ws.pre_B.narrow(0, 0, N);
        auto pre_C_fwd = g_bilevel_ws.pre_C.narrow(0, 0, N);

        // Forward direction: precompute
        if (N >= GEMM_PRECOMPUTE_THRESHOLD) {
            bilevel_precompute_gemm(
                x_sorted, mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
                mamba_fwd_B_proj, mamba_fwd_C_proj,
                fwd_saved_x_branch, fwd_saved_z, fwd_saved_dt,
                pre_B_fwd, pre_C_fwd,
                d_model, d_inner, d_state);
        } else {
            const int pre_grid = (N + SG2B_BLOCK - 1) / SG2B_BLOCK;
            bilevel_precompute_kernel<<<pre_grid, SG2B_BLOCK>>>(
                x_sorted.data_ptr<float>(),
                mamba_fwd_in_proj.data_ptr<float>(),
                mamba_fwd_dt_W.data_ptr<float>(),
                mamba_fwd_dt_b.data_ptr<float>(),
                mamba_fwd_B_proj.data_ptr<float>(),
                mamba_fwd_C_proj.data_ptr<float>(),
                fwd_saved_x_branch.data_ptr<float>(),
                fwd_saved_z.data_ptr<float>(),
                fwd_saved_dt.data_ptr<float>(),
                pre_B_fwd.data_ptr<float>(),
                pre_C_fwd.data_ptr<float>(),
                N, d_model, d_inner, d_state);
        }

        // Forward direction: parallel scan with state saving
        fwd_scan_out.zero_();
        int block_po2 = 1;
        int actual_block = std::min(PSCAN_BLOCK, N);
        while (block_po2 < actual_block) block_po2 *= 2;
        block_po2 = std::min(block_po2, PSCAN_BLOCK);
        int pscan_smem = 6 * block_po2 * sizeof(float);

        mamba3_parallel_scan_fwd_save_kernel<<<d_inner, block_po2, pscan_smem>>>(
            fwd_saved_x_branch.data_ptr<float>(),
            fwd_saved_z.data_ptr<float>(),
            fwd_saved_dt.data_ptr<float>(),
            pre_B_fwd.data_ptr<float>(),
            pre_C_fwd.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            fwd_scan_out.data_ptr<float>(),
            fwd_final_state.data_ptr<float>(),
            fwd_saved_states.data_ptr<float>(),
            fwd_initial_state.numel() > 0
                ? fwd_initial_state.data_ptr<float>() : nullptr,
            N, d_inner, d_state, 0, ckpt_int
        );

        // Backward direction: precompute (uses reversed input)
        // Reuse pre_B/C since forward scan is done with them
        auto x_sorted_rev = x_sorted.flip(0).contiguous();
        auto pre_B_bwd = g_bilevel_ws.pre_B.narrow(0, 0, N);
        auto pre_C_bwd = g_bilevel_ws.pre_C.narrow(0, 0, N);

        if (N >= GEMM_PRECOMPUTE_THRESHOLD) {
            bilevel_precompute_gemm(
                x_sorted_rev, mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
                mamba_bwd_B_proj, mamba_bwd_C_proj,
                bwd_saved_x_branch, bwd_saved_z, bwd_saved_dt,
                pre_B_bwd, pre_C_bwd,
                d_model, d_inner, d_state);
        } else {
            const int pre_grid = (N + SG2B_BLOCK - 1) / SG2B_BLOCK;
            bilevel_precompute_kernel<<<pre_grid, SG2B_BLOCK>>>(
                x_sorted_rev.data_ptr<float>(),
                mamba_bwd_in_proj.data_ptr<float>(),
                mamba_bwd_dt_W.data_ptr<float>(),
                mamba_bwd_dt_b.data_ptr<float>(),
                mamba_bwd_B_proj.data_ptr<float>(),
                mamba_bwd_C_proj.data_ptr<float>(),
                bwd_saved_x_branch.data_ptr<float>(),
                bwd_saved_z.data_ptr<float>(),
                bwd_saved_dt.data_ptr<float>(),
                pre_B_bwd.data_ptr<float>(),
                pre_C_bwd.data_ptr<float>(),
                N, d_model, d_inner, d_state);
        }

        // Backward direction: parallel scan with state saving
        bwd_scan_out.zero_();
        mamba3_parallel_scan_fwd_save_kernel<<<d_inner, block_po2, pscan_smem>>>(
            bwd_saved_x_branch.data_ptr<float>(),
            bwd_saved_z.data_ptr<float>(),
            bwd_saved_dt.data_ptr<float>(),
            pre_B_bwd.data_ptr<float>(),
            pre_C_bwd.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            bwd_scan_out.data_ptr<float>(),
            bwd_final_state.data_ptr<float>(),
            bwd_saved_states.data_ptr<float>(),
            bwd_initial_state.numel() > 0
                ? bwd_initial_state.data_ptr<float>() : nullptr,
            N, d_inner, d_state, 0, ckpt_int
        );
    } else {
        // ═══ Sequential fallback for small N ═══
        int scan_smem = d_inner * sizeof(float);
        mamba3_scan_fwd_save_kernel<<<1, d_inner, scan_smem>>>(
            x_sorted.data_ptr<float>(),
            mamba_fwd_in_proj.data_ptr<float>(),
            mamba_fwd_dt_W.data_ptr<float>(),
            mamba_fwd_dt_b.data_ptr<float>(),
            mamba_fwd_B_proj.data_ptr<float>(),
            mamba_fwd_C_proj.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            fwd_scan_out.data_ptr<float>(),
            fwd_final_state.data_ptr<float>(),
            fwd_saved_states.data_ptr<float>(),
            fwd_saved_x_branch.data_ptr<float>(),
            fwd_saved_z.data_ptr<float>(),
            fwd_saved_dt.data_ptr<float>(),
            fwd_initial_state.numel() > 0
                ? fwd_initial_state.data_ptr<float>() : nullptr,
            N, d_model, d_inner, d_state, 0, ckpt_int
        );

        auto x_sorted_rev = x_sorted.flip(0).contiguous();
        mamba3_scan_fwd_save_kernel<<<1, d_inner, scan_smem>>>(
            x_sorted_rev.data_ptr<float>(),
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            bwd_scan_out.data_ptr<float>(),
            bwd_final_state.data_ptr<float>(),
            bwd_saved_states.data_ptr<float>(),
            bwd_saved_x_branch.data_ptr<float>(),
            bwd_saved_z.data_ptr<float>(),
            bwd_saved_dt.data_ptr<float>(),
            bwd_initial_state.numel() > 0
                ? bwd_initial_state.data_ptr<float>() : nullptr,
            N, d_model, d_inner, d_state, 0, ckpt_int
        );
    }
}


void launch_mamba3_peer_backward(
    // Upstream gradient
    torch::Tensor d_smart_grad,       // [N]
    torch::Tensor grad,               // [N] original gradient
    torch::Tensor sharpness,          // [N]
    float rescale,
    // Saved from forward
    torch::Tensor sort_indices,       // [N] int
    torch::Tensor x_sorted,          // [N, d_model]
    torch::Tensor fwd_scan_out,       // [N, d_inner]
    torch::Tensor bwd_scan_out,       // [N, d_inner]
    torch::Tensor fwd_saved_states,   // [N, d_inner, d_state]
    torch::Tensor fwd_saved_x_branch, // [N, d_inner]
    torch::Tensor fwd_saved_z,        // [N, d_inner]
    torch::Tensor fwd_saved_dt,       // [N, d_inner]
    torch::Tensor bwd_saved_states,
    torch::Tensor bwd_saved_x_branch,
    torch::Tensor bwd_saved_z,
    torch::Tensor bwd_saved_dt,
    // GRU saved intermediates
    torch::Tensor gru_input,          // [N, gru_input_dim]
    torch::Tensor gru_h_old,          // [N, gru_hidden]
    torch::Tensor gru_z_gate,         // [N, gru_hidden]
    torch::Tensor gru_r_gate,         // [N, gru_hidden]
    torch::Tensor gru_h_tilde,        // [N, gru_hidden]
    // Expert+PEER saved intermediates
    torch::Tensor peer_input,         // [N, peer_input_dim]
    torch::Tensor expert_indices,     // [N, num_heads, num_active]
    torch::Tensor routing_weights,    // [N, num_heads, num_active]
    torch::Tensor saved_z_hidden,     // [N, num_heads, num_active, expert_hidden]
    torch::Tensor saved_scores_a,     // [N, num_heads, pk_dim]
    torch::Tensor saved_scores_b,
    torch::Tensor saved_top_a_idx,    // [N, num_heads, topk]
    torch::Tensor saved_top_b_idx,
    torch::Tensor saved_soft_a,       // [N, num_heads, topk]
    torch::Tensor saved_soft_b,
    // Weights (read-only)
    torch::Tensor mamba_fwd_in_proj,
    torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b,
    torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj,
    torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D,
    torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj,
    torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b,
    torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj,
    torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D,
    torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor peer_query_Ws,
    torch::Tensor prod_keys_A,
    torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_W2,
    torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
    torch::Tensor input_proj_W,
    // Mamba initial states (for correct h_prev at step 0)
    torch::Tensor mamba_fwd_init_state,  // [d_inner, d_state] or empty
    torch::Tensor mamba_bwd_init_state,  // [d_inner, d_state] or empty
    // Gradient outputs (pre-allocated, zeroed)
    torch::Tensor d_mamba_fwd_in_proj,
    torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b,
    torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj,
    torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D,
    torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_fwd_out_proj,
    torch::Tensor d_mamba_bwd_in_proj,
    torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b,
    torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj,
    torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D,
    torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_mamba_bwd_out_proj,
    torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
    torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
    torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
    torch::Tensor d_peer_query_Ws,
    torch::Tensor d_prod_keys_A,
    torch::Tensor d_prod_keys_B,
    torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
    torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
    torch::Tensor d_input_proj_W,
    torch::Tensor d_input_proj_b,
    // Dims
    int d_model, int d_state, int d_inner,
    int gru_hidden, int gru_input_dim,
    int num_heads, int topk, int pk_dim,
    int expert_hidden, int peer_input_dim, int num_experts,
    int checkpoint_interval  // 0 = no checkpointing
) {
    const int N = d_smart_grad.numel();
    if (N == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");

    auto dev = d_smart_grad.device();

    // Pre-allocate workspace (reused across steps)
    g_bilevel_ws.ensure_backward(N, d_model, d_inner, d_state,
                                  peer_input_dim, gru_input_dim, dev);

    const int grid = (N + SG2B_BLOCK - 1) / SG2B_BLOCK;
    int num_active = topk * topk;

    // Step 1: d_expert_out = rescale * d_smart_grad
    auto d_expert_out = (d_smart_grad.reshape(-1) * rescale).contiguous();

    // Step 2: Expert + PEER backward (use workspace, zero the N-region)
    auto d_peer_input = g_bilevel_ws.d_peer_input.narrow(0, 0, N);
    d_peer_input.zero_();
    int half_d = d_model / 2;
    int expert_smem_elems = 3 * num_experts * expert_hidden + num_experts;
    int routing_smem_elems = num_heads * d_model * peer_input_dim  // peer_query_Ws
                           + num_heads * pk_dim * half_d            // prod_keys_A
                           + num_heads * pk_dim * half_d;           // prod_keys_B
    int total_smem_bytes = (expert_smem_elems + routing_smem_elems) * sizeof(float);
    expert_peer_backward_kernel<<<grid, SG2B_BLOCK, total_smem_bytes>>>(
        d_expert_out.data_ptr<float>(),
        grad.to(torch::kFloat32).reshape(-1).data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        routing_weights.data_ptr<float>(),
        saved_z_hidden.data_ptr<float>(),
        peer_input.data_ptr<float>(),
        peer_query_Ws.data_ptr<float>(),
        prod_keys_A.data_ptr<float>(),
        prod_keys_B.data_ptr<float>(),
        saved_scores_a.data_ptr<float>(),
        saved_scores_b.data_ptr<float>(),
        saved_top_a_idx.data_ptr<int>(),
        saved_top_b_idx.data_ptr<int>(),
        saved_soft_a.data_ptr<float>(),
        saved_soft_b.data_ptr<float>(),
        expert_W1.data_ptr<float>(),
        expert_W2.data_ptr<float>(),
        expert_b2_in.reshape(-1).data_ptr<float>(),
        d_expert_W1.data_ptr<float>(),
        d_expert_b1.data_ptr<float>(),
        d_expert_W2.data_ptr<float>(),
        d_expert_b2.data_ptr<float>(),
        d_peer_query_Ws.data_ptr<float>(),
        d_prod_keys_A.data_ptr<float>(),
        d_prod_keys_B.data_ptr<float>(),
        d_peer_input.data_ptr<float>(),
        N, num_heads, topk, num_active,
        d_model, pk_dim, expert_hidden,
        peer_input_dim, num_experts
    );

    // Step 3: Extract d_gru_out from d_peer_input
    // peer_input = [gru_state, fwd_ctx, bwd_ctx, g, s]
    // d_gru_out = d_peer_input[:, :gru_hidden]
    auto d_gru_out = d_peer_input.narrow(1, 0, gru_hidden).contiguous();

    // Step 4: GRU backward
    auto d_gru_input = g_bilevel_ws.d_gru_input.narrow(0, 0, N);
    d_gru_input.zero_();
    const int gru_total_dim = gru_input_dim + gru_hidden;
    const int gru_smem = (3 * gru_hidden * gru_total_dim + 3 * gru_hidden) * sizeof(float);
    gru_backward_kernel<<<grid, SG2B_BLOCK, gru_smem>>>(
        d_gru_out.data_ptr<float>(),
        gru_input.data_ptr<float>(),
        gru_h_old.data_ptr<float>(),
        gru_z_gate.data_ptr<float>(),
        gru_r_gate.data_ptr<float>(),
        gru_h_tilde.data_ptr<float>(),
        gru_Wz.data_ptr<float>(),
        gru_Wr.data_ptr<float>(),
        gru_Wh.data_ptr<float>(),
        d_gru_Wz.data_ptr<float>(),
        d_gru_bz.data_ptr<float>(),
        d_gru_Wr.data_ptr<float>(),
        d_gru_br.data_ptr<float>(),
        d_gru_Wh.data_ptr<float>(),
        d_gru_bh.data_ptr<float>(),
        d_gru_input.data_ptr<float>(),
        N, gru_input_dim, gru_hidden
    );

    // Step 5: Extract d_fwd_ctx and d_bwd_ctx from d_gru_input + d_peer_input
    // gru_input = [g, s, fwd_ctx, bwd_ctx]
    // peer_input = [gru_state, fwd_ctx, bwd_ctx, g, s]
    auto d_fwd_ctx = g_bilevel_ws.d_fwd_ctx.narrow(0, 0, N);
    auto d_bwd_ctx = g_bilevel_ws.d_bwd_ctx.narrow(0, 0, N);
    d_fwd_ctx.zero_();
    d_bwd_ctx.zero_();

    // From gru_input: fwd_ctx at offset 2, bwd_ctx at offset 2+d_model
    d_fwd_ctx.add_(d_gru_input.narrow(1, 2, d_model));
    d_bwd_ctx.add_(d_gru_input.narrow(1, 2 + d_model, d_model));

    // From peer_input: fwd_ctx at offset gru_hidden, bwd_ctx at offset gru_hidden+d_model
    d_fwd_ctx.add_(d_peer_input.narrow(1, gru_hidden, d_model));
    d_bwd_ctx.add_(d_peer_input.narrow(1, gru_hidden + d_model, d_model));

    // Step 6: Re-sort d_fwd_ctx and d_bwd_ctx (unsort was applied in forward,
    // so we need to sort these back to the sorted order)
    auto sort_idx_long = sort_indices.to(torch::kLong);
    auto d_fwd_sorted = d_fwd_ctx.index_select(0, sort_idx_long);
    auto d_bwd_sorted = d_bwd_ctx.index_select(0, sort_idx_long);
    // bwd scan was on reversed x_sorted, so flip d_bwd_sorted
    d_bwd_sorted = d_bwd_sorted.flip(0).contiguous();

    // Step 7: Out-projection backward for both directions
    auto d_fwd_scan_out = g_bilevel_ws.d_fwd_scan_out.narrow(0, 0, N);
    auto d_bwd_scan_out = g_bilevel_ws.d_bwd_scan_out.narrow(0, 0, N);
    d_fwd_scan_out.zero_();
    d_bwd_scan_out.zero_();

    int out_proj_bwd_smem = d_model * d_inner * sizeof(float);
    out_proj_backward_kernel<<<grid, SG2B_BLOCK, out_proj_bwd_smem>>>(
        d_fwd_sorted.data_ptr<float>(),
        fwd_scan_out.data_ptr<float>(),
        mamba_fwd_out_proj.data_ptr<float>(),
        d_mamba_fwd_out_proj.data_ptr<float>(),
        d_fwd_scan_out.data_ptr<float>(),
        N, d_model, d_inner
    );

    out_proj_backward_kernel<<<grid, SG2B_BLOCK, out_proj_bwd_smem>>>(
        d_bwd_sorted.data_ptr<float>(),
        bwd_scan_out.data_ptr<float>(),
        mamba_bwd_out_proj.data_ptr<float>(),
        d_mamba_bwd_out_proj.data_ptr<float>(),
        d_bwd_scan_out.data_ptr<float>(),
        N, d_model, d_inner
    );

    // Step 8: Mamba scan backward (both directions)
    // Two-pass: s_x_branch[d_inner] + s_d_dt_raw[d_inner] (no s_d_C/B_proj_W)
    int scan_smem = 2 * d_inner * sizeof(float);
    auto d_x_sorted_fwd = g_bilevel_ws.d_x_sorted_fwd.narrow(0, 0, N);
    auto d_x_sorted_bwd = g_bilevel_ws.d_x_sorted_bwd.narrow(0, 0, N);
    d_x_sorted_fwd.zero_();
    d_x_sorted_bwd.zero_();

    // Two-pass buffers for warp-reduced per-timestep derivatives
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto d_C_vals_fwd = torch::empty({N, d_state}, float_opts);
    auto d_B_vals_fwd = torch::empty({N, d_state}, float_opts);

    mamba3_scan_backward_kernel<<<1, d_inner, scan_smem>>>(
        d_fwd_scan_out.data_ptr<float>(),
        x_sorted.data_ptr<float>(),
        fwd_saved_states.data_ptr<float>(),
        fwd_saved_x_branch.data_ptr<float>(),
        fwd_saved_z.data_ptr<float>(),
        fwd_saved_dt.data_ptr<float>(),
        mamba_fwd_in_proj.data_ptr<float>(),
        mamba_fwd_dt_W.data_ptr<float>(),
        mamba_fwd_dt_b.data_ptr<float>(),
        mamba_fwd_B_proj.data_ptr<float>(),
        mamba_fwd_C_proj.data_ptr<float>(),
        mamba_fwd_A_log.data_ptr<float>(),
        mamba_fwd_D.data_ptr<float>(),
        mamba_fwd_rope.data_ptr<float>(),
        d_mamba_fwd_in_proj.data_ptr<float>(),
        d_mamba_fwd_dt_W.data_ptr<float>(),
        d_mamba_fwd_dt_b.data_ptr<float>(),
        d_mamba_fwd_A_log.data_ptr<float>(),
        d_mamba_fwd_D.data_ptr<float>(),
        d_mamba_fwd_rope.data_ptr<float>(),
        d_x_sorted_fwd.data_ptr<float>(),
        mamba_fwd_init_state.numel() > 0
            ? mamba_fwd_init_state.data_ptr<float>() : nullptr,
        d_C_vals_fwd.data_ptr<float>(),
        d_B_vals_fwd.data_ptr<float>(),
        N, d_model, d_inner, d_state, 0,
        (checkpoint_interval > 1) ? checkpoint_interval : 0
    );

    // Two-pass GEMM: d_C_proj_W += d_C_vals.T @ saved_x_branch
    // d_C_vals_fwd: [N, d_state], fwd_saved_x_branch: [N, d_inner]
    // d_C_proj_W: [d_state, d_inner] += [d_state, N] @ [N, d_inner]
    torch::mm_out(d_mamba_fwd_C_proj, d_C_vals_fwd.t(), fwd_saved_x_branch);
    torch::mm_out(d_mamba_fwd_B_proj, d_B_vals_fwd.t(), fwd_saved_x_branch);

    // Backward scan used reversed x_sorted
    auto x_sorted_rev = x_sorted.flip(0).contiguous();
    auto d_C_vals_bwd = torch::empty({N, d_state}, float_opts);
    auto d_B_vals_bwd = torch::empty({N, d_state}, float_opts);

    mamba3_scan_backward_kernel<<<1, d_inner, scan_smem>>>(
        d_bwd_scan_out.data_ptr<float>(),
        x_sorted_rev.data_ptr<float>(),
        bwd_saved_states.data_ptr<float>(),
        bwd_saved_x_branch.data_ptr<float>(),
        bwd_saved_z.data_ptr<float>(),
        bwd_saved_dt.data_ptr<float>(),
        mamba_bwd_in_proj.data_ptr<float>(),
        mamba_bwd_dt_W.data_ptr<float>(),
        mamba_bwd_dt_b.data_ptr<float>(),
        mamba_bwd_B_proj.data_ptr<float>(),
        mamba_bwd_C_proj.data_ptr<float>(),
        mamba_bwd_A_log.data_ptr<float>(),
        mamba_bwd_D.data_ptr<float>(),
        mamba_bwd_rope.data_ptr<float>(),
        d_mamba_bwd_in_proj.data_ptr<float>(),
        d_mamba_bwd_dt_W.data_ptr<float>(),
        d_mamba_bwd_dt_b.data_ptr<float>(),
        d_mamba_bwd_A_log.data_ptr<float>(),
        d_mamba_bwd_D.data_ptr<float>(),
        d_mamba_bwd_rope.data_ptr<float>(),
        d_x_sorted_bwd.data_ptr<float>(),
        mamba_bwd_init_state.numel() > 0
            ? mamba_bwd_init_state.data_ptr<float>() : nullptr,
        d_C_vals_bwd.data_ptr<float>(),
        d_B_vals_bwd.data_ptr<float>(),
        N, d_model, d_inner, d_state, 0,
        (checkpoint_interval > 1) ? checkpoint_interval : 0
    );

    // Two-pass GEMM: d_C_proj_W += d_C_vals.T @ saved_x_branch (bwd direction)
    torch::mm_out(d_mamba_bwd_C_proj, d_C_vals_bwd.t(), bwd_saved_x_branch);
    torch::mm_out(d_mamba_bwd_B_proj, d_B_vals_bwd.t(), bwd_saved_x_branch);

    // Combine d_x_sorted from both directions
    // bwd scan backward produces d for reversed input, flip back
    auto d_x_sorted = d_x_sorted_fwd + d_x_sorted_bwd.flip(0);

    // Unsort d_x_sorted back to original order for input_proj backward
    auto unsort_idx = g_bilevel_ws.unsort_idx.narrow(0, 0, N);
    unsort_idx.scatter_(0, sort_idx_long,
        torch::arange(N, torch::TensorOptions().device(dev).dtype(torch::kLong)));
    auto d_x_unsorted = d_x_sorted.index_select(0, unsort_idx);

    // Step 9: Input projection backward
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "input_proj_backward", ([&] {
        int input_proj_bwd_smem = (d_model * 3) * sizeof(float);
        input_proj_backward_kernel<scalar_t><<<grid, SG2B_BLOCK, input_proj_bwd_smem>>>(
            d_x_unsorted.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            sharpness.data_ptr<scalar_t>(),
            d_input_proj_W.data_ptr<float>(),
            d_input_proj_b.data_ptr<float>(),
            N, d_model
        );
    }));
}


// ═══════════════════════════════════════════════════════════════════════
//  Batched Bilevel Forward-Save Launcher
//
//  Packs multiple parameters' data and launches batched fwd_save kernel
//  for both scan directions. Returns packed saved intermediates.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> grads,             // [num_params] each [N_i]
    std::vector<torch::Tensor> sharpness_list,    // [num_params] each [N_i]
    // Input proj weights
    torch::Tensor input_proj_W,
    torch::Tensor input_proj_b,
    // Mamba forward weights
    torch::Tensor mamba_fwd_in_proj,
    torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b,
    torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj,
    torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D,
    torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    // Mamba backward weights
    torch::Tensor mamba_bwd_in_proj,
    torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b,
    torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj,
    torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D,
    torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    // Dims
    int d_model, int d_state, int d_inner,
    // Outputs (pre-allocated by caller)
    torch::Tensor fwd_scan_out_packed,       // [total_N, d_inner]
    torch::Tensor bwd_scan_out_packed,       // [total_N, d_inner]
    torch::Tensor fwd_saved_states_packed,   // [total_N, d_inner, d_state]
    torch::Tensor fwd_saved_xb_packed,       // [total_N, d_inner]
    torch::Tensor fwd_saved_z_packed,        // [total_N, d_inner]
    torch::Tensor fwd_saved_dt_packed,       // [total_N, d_inner]
    torch::Tensor bwd_saved_states_packed,
    torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed,
    torch::Tensor bwd_saved_dt_packed,
    torch::Tensor x_sorted_packed,           // [total_N, d_model]
    torch::Tensor offsets_t,                 // [num_params + 1]
    torch::Tensor sort_indices_packed,       // [total_N] int
    // Mamba initial states (persistent, not zeros)
    torch::Tensor fwd_initial_states,        // [num_params, d_inner, d_state]
    torch::Tensor bwd_initial_states,        // [num_params, d_inner, d_state]
    int checkpoint_interval                   // 0 = save every step
) {
    const int num_params = grads.size();
    if (num_params == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");

    auto dev = grads[0].device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Step 1: Input projection + sort per param, pack into x_sorted_packed
    std::vector<int> offsets_cpu(num_params + 1);
    offsets_cpu[0] = 0;
    for (int p = 0; p < num_params; p++) {
        int N = grads[p].numel();
        offsets_cpu[p + 1] = offsets_cpu[p] + N;

        if (N == 0) continue;

        auto g_f = grads[p].to(torch::kFloat32).reshape(-1);
        auto s_f = sharpness_list[p].to(torch::kFloat32).reshape(-1);
        auto inp = torch::stack({g_f, s_f}, 1);
        auto x_proj = torch::addmm(input_proj_b, inp, input_proj_W.t());

        auto sort_keys = g_f.abs();
        auto sorted_idx = sort_keys.argsort();
        auto sorted_idx_int = sorted_idx.to(torch::kInt32);

        // Copy sort indices
        sort_indices_packed.narrow(0, offsets_cpu[p], N).copy_(sorted_idx_int);

        // Sort and copy x_sorted
        auto x_sorted_p = x_proj.index_select(0, sorted_idx);
        x_sorted_packed.narrow(0, offsets_cpu[p], N).copy_(x_sorted_p);
    }

    offsets_t.copy_(torch::from_blob(offsets_cpu.data(), {num_params + 1},
        torch::kInt32).to(dev));

    int total_N = offsets_cpu[num_params];

    // Pre-allocate workspace for batched buffers
    g_bilevel_ws.ensure_batched(total_N, d_model, d_inner, d_state, dev);

    auto final_fwd = torch::empty({num_params, d_inner, d_state}, float_opts);
    auto final_bwd = torch::empty({num_params, d_inner, d_state}, float_opts);

    auto rev_fwd = torch::zeros({num_params}, int_opts);
    auto rev_bwd = torch::ones({num_params}, int_opts);

    int ckpt_int = (checkpoint_interval > 1) ? checkpoint_interval : 0;

    // Compute checkpoint offsets if checkpointing is enabled
    torch::Tensor ckpt_offsets_t;
    if (ckpt_int > 1) {
        std::vector<int> ckpt_offsets_cpu(num_params + 1);
        ckpt_offsets_cpu[0] = 0;
        for (int p = 0; p < num_params; p++) {
            int N = offsets_cpu[p + 1] - offsets_cpu[p];
            int num_ckpts = (N + ckpt_int - 1) / ckpt_int;
            ckpt_offsets_cpu[p + 1] = ckpt_offsets_cpu[p] + num_ckpts;
        }
        ckpt_offsets_t = torch::from_blob(ckpt_offsets_cpu.data(), {num_params + 1},
            torch::kInt32).to(dev).clone();
    }

    // Determine max N across params for parallel scan threshold
    int max_N = 0;
    for (int p = 0; p < num_params; p++) {
        int N = offsets_cpu[p + 1] - offsets_cpu[p];
        if (N > max_N) max_N = N;
    }

    if (max_N >= PSCAN_THRESHOLD) {
        // ═══ Parallel scan path: precompute + batched parallel prefix scan ═══
        // Precompute for all packed data (weights shared across params)
        auto pre_B = g_bilevel_ws.pre_B.narrow(0, 0, total_N);
        auto pre_C = g_bilevel_ws.pre_C.narrow(0, 0, total_N);

        // Forward precompute — outputs to saved_xb/z/dt directly
        if (total_N >= GEMM_PRECOMPUTE_THRESHOLD) {
            bilevel_precompute_gemm(
                x_sorted_packed, mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
                mamba_fwd_B_proj, mamba_fwd_C_proj,
                fwd_saved_xb_packed, fwd_saved_z_packed, fwd_saved_dt_packed,
                pre_B, pre_C,
                d_model, d_inner, d_state);
        } else {
            const int pre_grid = (total_N + SG2B_BLOCK - 1) / SG2B_BLOCK;
            bilevel_precompute_kernel<<<pre_grid, SG2B_BLOCK>>>(
                x_sorted_packed.data_ptr<float>(),
                mamba_fwd_in_proj.data_ptr<float>(),
                mamba_fwd_dt_W.data_ptr<float>(),
                mamba_fwd_dt_b.data_ptr<float>(),
                mamba_fwd_B_proj.data_ptr<float>(),
                mamba_fwd_C_proj.data_ptr<float>(),
                fwd_saved_xb_packed.data_ptr<float>(),
                fwd_saved_z_packed.data_ptr<float>(),
                fwd_saved_dt_packed.data_ptr<float>(),
                pre_B.data_ptr<float>(),
                pre_C.data_ptr<float>(),
                total_N, d_model, d_inner, d_state);
        }

        // Forward parallel scan — single launch for all params
        fwd_scan_out_packed.zero_();
        int block_po2 = 1;
        int actual_block = std::min(PSCAN_BLOCK, max_N);
        while (block_po2 < actual_block) block_po2 *= 2;
        block_po2 = std::min(block_po2, PSCAN_BLOCK);
        int pscan_smem = 6 * block_po2 * sizeof(float);
        dim3 grid2d(d_inner, num_params);

        mamba3_batched_parallel_scan_fwd_save_kernel<<<grid2d, block_po2, pscan_smem>>>(
            fwd_saved_xb_packed.data_ptr<float>(),
            fwd_saved_z_packed.data_ptr<float>(),
            fwd_saved_dt_packed.data_ptr<float>(),
            pre_B.data_ptr<float>(),
            pre_C.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            fwd_scan_out_packed.data_ptr<float>(),
            fwd_initial_states.data_ptr<float>(),
            final_fwd.data_ptr<float>(),
            fwd_saved_states_packed.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            rev_fwd.data_ptr<int>(),
            ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
            d_inner, d_state, ckpt_int
        );

        // Build reversed x_sorted for backward
        auto x_sorted_rev_packed = g_bilevel_ws.x_sorted_rev.narrow(0, 0, total_N);
        {
            int total_elems = total_N * d_model;
            int rev_grid = (total_elems + SG2B_BLOCK - 1) / SG2B_BLOCK;
            reverse_segments_kernel<<<rev_grid, SG2B_BLOCK>>>(
                x_sorted_packed.data_ptr<float>(),
                x_sorted_rev_packed.data_ptr<float>(),
                offsets_t.data_ptr<int>(),
                d_model, num_params
            );
        }

        // Backward precompute — reuse pre_B/C since forward scan is done with them
        auto pre_B_bwd = g_bilevel_ws.pre_B.narrow(0, 0, total_N);
        auto pre_C_bwd = g_bilevel_ws.pre_C.narrow(0, 0, total_N);

        if (total_N >= GEMM_PRECOMPUTE_THRESHOLD) {
            bilevel_precompute_gemm(
                x_sorted_rev_packed, mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
                mamba_bwd_B_proj, mamba_bwd_C_proj,
                bwd_saved_xb_packed, bwd_saved_z_packed, bwd_saved_dt_packed,
                pre_B_bwd, pre_C_bwd,
                d_model, d_inner, d_state);
        } else {
            const int pre_grid = (total_N + SG2B_BLOCK - 1) / SG2B_BLOCK;
            bilevel_precompute_kernel<<<pre_grid, SG2B_BLOCK>>>(
                x_sorted_rev_packed.data_ptr<float>(),
                mamba_bwd_in_proj.data_ptr<float>(),
                mamba_bwd_dt_W.data_ptr<float>(),
                mamba_bwd_dt_b.data_ptr<float>(),
                mamba_bwd_B_proj.data_ptr<float>(),
                mamba_bwd_C_proj.data_ptr<float>(),
                bwd_saved_xb_packed.data_ptr<float>(),
                bwd_saved_z_packed.data_ptr<float>(),
                bwd_saved_dt_packed.data_ptr<float>(),
                pre_B_bwd.data_ptr<float>(),
                pre_C_bwd.data_ptr<float>(),
                total_N, d_model, d_inner, d_state);
        }

        // Backward parallel scan — single launch
        bwd_scan_out_packed.zero_();
        mamba3_batched_parallel_scan_fwd_save_kernel<<<grid2d, block_po2, pscan_smem>>>(
            bwd_saved_xb_packed.data_ptr<float>(),
            bwd_saved_z_packed.data_ptr<float>(),
            bwd_saved_dt_packed.data_ptr<float>(),
            pre_B_bwd.data_ptr<float>(),
            pre_C_bwd.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            bwd_scan_out_packed.data_ptr<float>(),
            bwd_initial_states.data_ptr<float>(),
            final_bwd.data_ptr<float>(),
            bwd_saved_states_packed.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            rev_fwd.data_ptr<int>(),
            ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
            d_inner, d_state, ckpt_int
        );
    } else {
        // ═══ Sequential fallback for small N ═══
        int scan_smem = d_inner * sizeof(float);

        mamba3_scan_fwd_save_batched_kernel<<<num_params, d_inner, scan_smem>>>(
            x_sorted_packed.data_ptr<float>(),
            fwd_scan_out_packed.data_ptr<float>(),
            fwd_initial_states.data_ptr<float>(),
            final_fwd.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            rev_fwd.data_ptr<int>(),
            fwd_saved_states_packed.data_ptr<float>(),
            fwd_saved_xb_packed.data_ptr<float>(),
            fwd_saved_z_packed.data_ptr<float>(),
            fwd_saved_dt_packed.data_ptr<float>(),
            mamba_fwd_in_proj.data_ptr<float>(),
            mamba_fwd_dt_W.data_ptr<float>(),
            mamba_fwd_dt_b.data_ptr<float>(),
            mamba_fwd_B_proj.data_ptr<float>(),
            mamba_fwd_C_proj.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
            d_model, d_inner, d_state, ckpt_int
        );

        auto x_sorted_rev_packed = g_bilevel_ws.x_sorted_rev.narrow(0, 0, total_N);
        {
            int total_elems = total_N * d_model;
            int rev_grid = (total_elems + SG2B_BLOCK - 1) / SG2B_BLOCK;
            reverse_segments_kernel<<<rev_grid, SG2B_BLOCK>>>(
                x_sorted_packed.data_ptr<float>(),
                x_sorted_rev_packed.data_ptr<float>(),
                offsets_t.data_ptr<int>(),
                d_model, num_params
            );
        }

        mamba3_scan_fwd_save_batched_kernel<<<num_params, d_inner, scan_smem>>>(
            x_sorted_rev_packed.data_ptr<float>(),
            bwd_scan_out_packed.data_ptr<float>(),
            bwd_initial_states.data_ptr<float>(),
            final_bwd.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            rev_fwd.data_ptr<int>(),
            bwd_saved_states_packed.data_ptr<float>(),
            bwd_saved_xb_packed.data_ptr<float>(),
            bwd_saved_z_packed.data_ptr<float>(),
            bwd_saved_dt_packed.data_ptr<float>(),
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
            d_model, d_inner, d_state, ckpt_int
        );
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Batched Bilevel Backward Launcher
//
//  Takes packed saved intermediates and gradient signals, launches
//  batched backward scan kernels for both directions.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_backward_batched(
    // Packed gradient signals (from out_proj backward)
    torch::Tensor d_fwd_scan_out_packed,     // [total_N, d_inner]
    torch::Tensor d_bwd_scan_out_packed,     // [total_N, d_inner]
    // Packed saved intermediates
    torch::Tensor x_sorted_packed,           // [total_N, d_model]
    torch::Tensor fwd_saved_states_packed,   // [total_N, d_inner, d_state]
    torch::Tensor fwd_saved_xb_packed,       // [total_N, d_inner]
    torch::Tensor fwd_saved_z_packed,        // [total_N, d_inner]
    torch::Tensor fwd_saved_dt_packed,       // [total_N, d_inner]
    torch::Tensor bwd_saved_states_packed,
    torch::Tensor bwd_saved_xb_packed,
    torch::Tensor bwd_saved_z_packed,
    torch::Tensor bwd_saved_dt_packed,
    torch::Tensor offsets_t,                 // [num_params + 1]
    // Weights
    torch::Tensor mamba_fwd_in_proj,
    torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b,
    torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj,
    torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D,
    torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_bwd_in_proj,
    torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b,
    torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj,
    torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D,
    torch::Tensor mamba_bwd_rope,
    // Gradient outputs (pre-zeroed by caller)
    torch::Tensor d_mamba_fwd_in_proj,
    torch::Tensor d_mamba_fwd_dt_W,
    torch::Tensor d_mamba_fwd_dt_b,
    torch::Tensor d_mamba_fwd_B_proj,
    torch::Tensor d_mamba_fwd_C_proj,
    torch::Tensor d_mamba_fwd_A_log,
    torch::Tensor d_mamba_fwd_D,
    torch::Tensor d_mamba_fwd_rope,
    torch::Tensor d_mamba_bwd_in_proj,
    torch::Tensor d_mamba_bwd_dt_W,
    torch::Tensor d_mamba_bwd_dt_b,
    torch::Tensor d_mamba_bwd_B_proj,
    torch::Tensor d_mamba_bwd_C_proj,
    torch::Tensor d_mamba_bwd_A_log,
    torch::Tensor d_mamba_bwd_D,
    torch::Tensor d_mamba_bwd_rope,
    torch::Tensor d_x_sorted_packed,         // [total_N, d_model] output
    // Mamba initial states (persistent, for correct h_prev at step 0)
    torch::Tensor fwd_initial_states,        // [num_params, d_inner, d_state]
    torch::Tensor bwd_initial_states,        // [num_params, d_inner, d_state]
    // Dims
    int d_model, int d_state, int d_inner, int num_params,
    int checkpoint_interval  // 0 = no checkpointing
) {
    if (num_params == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");

    auto dev = d_fwd_scan_out_packed.device();
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    auto rev_fwd = torch::zeros({num_params}, int_opts);
    auto rev_bwd = torch::zeros({num_params}, int_opts);  // data was pre-reversed

    // Pre-allocate workspace (reused across steps)
    auto total_N = x_sorted_packed.size(0);
    g_bilevel_ws.ensure_batched(total_N, d_model, d_inner, d_state, dev);

    // Two-pass: s_x_branch[d_inner] + s_d_dt_raw[d_inner] (no s_d_C/B_proj_W)
    int scan_smem = 2 * d_inner * sizeof(float);
    int ckpt_int = (checkpoint_interval > 1) ? checkpoint_interval : 0;

    // Compute checkpoint offsets if needed (using offsets already on GPU)
    torch::Tensor ckpt_offsets_t;
    if (ckpt_int > 1) {
        // Read offsets from GPU to CPU for checkpoint computation
        auto offsets_cpu_tmp = offsets_t.to(torch::kCPU);
        auto offsets_ptr_tmp = offsets_cpu_tmp.data_ptr<int>();
        std::vector<int> ckpt_offsets_cpu(num_params + 1);
        ckpt_offsets_cpu[0] = 0;
        for (int p = 0; p < num_params; p++) {
            int N = offsets_ptr_tmp[p + 1] - offsets_ptr_tmp[p];
            int num_ckpts = (N + ckpt_int - 1) / ckpt_int;
            ckpt_offsets_cpu[p + 1] = ckpt_offsets_cpu[p] + num_ckpts;
        }
        ckpt_offsets_t = torch::from_blob(ckpt_offsets_cpu.data(), {num_params + 1},
            torch::kInt32).to(dev).clone();
    }

    // Two-pass buffers for warp-reduced per-timestep derivatives
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto d_C_vals_packed = torch::empty({total_N, d_state}, float_opts);
    auto d_B_vals_packed = torch::empty({total_N, d_state}, float_opts);

    // Forward direction backward scan
    auto d_x_sorted_fwd = g_bilevel_ws.d_x_sorted_fwd_bat.narrow(0, 0, total_N);
    d_x_sorted_fwd.zero_();
    mamba3_scan_backward_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        d_fwd_scan_out_packed.data_ptr<float>(),
        x_sorted_packed.data_ptr<float>(),
        fwd_saved_states_packed.data_ptr<float>(),
        fwd_saved_xb_packed.data_ptr<float>(),
        fwd_saved_z_packed.data_ptr<float>(),
        fwd_saved_dt_packed.data_ptr<float>(),
        offsets_t.data_ptr<int>(),
        rev_fwd.data_ptr<int>(),
        fwd_initial_states.data_ptr<float>(),
        mamba_fwd_in_proj.data_ptr<float>(),
        mamba_fwd_dt_W.data_ptr<float>(),
        mamba_fwd_dt_b.data_ptr<float>(),
        mamba_fwd_B_proj.data_ptr<float>(),
        mamba_fwd_C_proj.data_ptr<float>(),
        mamba_fwd_A_log.data_ptr<float>(),
        mamba_fwd_D.data_ptr<float>(),
        mamba_fwd_rope.data_ptr<float>(),
        d_mamba_fwd_in_proj.data_ptr<float>(),
        d_mamba_fwd_dt_W.data_ptr<float>(),
        d_mamba_fwd_dt_b.data_ptr<float>(),
        d_mamba_fwd_A_log.data_ptr<float>(),
        d_mamba_fwd_D.data_ptr<float>(),
        d_mamba_fwd_rope.data_ptr<float>(),
        d_x_sorted_fwd.data_ptr<float>(),
        ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
        d_C_vals_packed.data_ptr<float>(),
        d_B_vals_packed.data_ptr<float>(),
        d_model, d_inner, d_state, ckpt_int
    );

    // Two-pass GEMM: d_C_proj_W = d_C_vals_packed.T @ fwd_saved_xb_packed
    // [d_state, total_N] @ [total_N, d_inner] → [d_state, d_inner]
    torch::mm_out(d_mamba_fwd_C_proj, d_C_vals_packed.t(), fwd_saved_xb_packed);
    torch::mm_out(d_mamba_fwd_B_proj, d_B_vals_packed.t(), fwd_saved_xb_packed);

    // Build reversed x_sorted via CUDA kernel (no CPU-GPU sync)
    auto x_sorted_rev_packed = g_bilevel_ws.x_sorted_rev.narrow(0, 0, total_N);
    {
        int total_elems = total_N * d_model;
        int rev_grid = (total_elems + SG2B_BLOCK - 1) / SG2B_BLOCK;
        reverse_segments_kernel<<<rev_grid, SG2B_BLOCK>>>(
            x_sorted_packed.data_ptr<float>(),
            x_sorted_rev_packed.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            d_model, num_params
        );
    }

    // Backward direction backward scan (reuse d_C/B_vals_packed buffers)
    auto d_x_sorted_bwd = g_bilevel_ws.d_x_sorted_bwd_bat.narrow(0, 0, total_N);
    d_x_sorted_bwd.zero_();
    mamba3_scan_backward_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        d_bwd_scan_out_packed.data_ptr<float>(),
        x_sorted_rev_packed.data_ptr<float>(),
        bwd_saved_states_packed.data_ptr<float>(),
        bwd_saved_xb_packed.data_ptr<float>(),
        bwd_saved_z_packed.data_ptr<float>(),
        bwd_saved_dt_packed.data_ptr<float>(),
        offsets_t.data_ptr<int>(),
        rev_bwd.data_ptr<int>(),
        bwd_initial_states.data_ptr<float>(),
        mamba_bwd_in_proj.data_ptr<float>(),
        mamba_bwd_dt_W.data_ptr<float>(),
        mamba_bwd_dt_b.data_ptr<float>(),
        mamba_bwd_B_proj.data_ptr<float>(),
        mamba_bwd_C_proj.data_ptr<float>(),
        mamba_bwd_A_log.data_ptr<float>(),
        mamba_bwd_D.data_ptr<float>(),
        mamba_bwd_rope.data_ptr<float>(),
        d_mamba_bwd_in_proj.data_ptr<float>(),
        d_mamba_bwd_dt_W.data_ptr<float>(),
        d_mamba_bwd_dt_b.data_ptr<float>(),
        d_mamba_bwd_A_log.data_ptr<float>(),
        d_mamba_bwd_D.data_ptr<float>(),
        d_mamba_bwd_rope.data_ptr<float>(),
        d_x_sorted_bwd.data_ptr<float>(),
        ckpt_int > 1 ? ckpt_offsets_t.data_ptr<int>() : nullptr,
        d_C_vals_packed.data_ptr<float>(),
        d_B_vals_packed.data_ptr<float>(),
        d_model, d_inner, d_state, ckpt_int
    );

    // Two-pass GEMM: d_C_proj_W = d_C_vals_packed.T @ bwd_saved_xb_packed
    torch::mm_out(d_mamba_bwd_C_proj, d_C_vals_packed.t(), bwd_saved_xb_packed);
    torch::mm_out(d_mamba_bwd_B_proj, d_B_vals_packed.t(), bwd_saved_xb_packed);

    // Combine: out = fwd + reverse_segments(bwd) via single CUDA kernel
    {
        int total_elems = total_N * d_model;
        int comb_grid = (total_elems + SG2B_BLOCK - 1) / SG2B_BLOCK;
        combine_fwd_bwd_kernel<<<comb_grid, SG2B_BLOCK>>>(
            d_x_sorted_fwd.data_ptr<float>(),
            d_x_sorted_bwd.data_ptr<float>(),
            d_x_sorted_packed.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
            d_model, num_params
        );
    }
}
