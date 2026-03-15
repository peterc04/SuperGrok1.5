/*
 * SuperGrok v2 — Mamba-3 + 4-Head PEER + GRU CUDA Kernels (Forward)
 *
 * Kernels for the Mamba-3 + PEER meta-net architecture:
 *
 *   1. input_proj_sort       — Project [grad, sharpness] -> [N, d_model],
 *                              compute sort keys = |grad|
 *   2. mamba3_scan (serial)  — Sequential selective scan with trapezoidal
 *                              discretization + RoPE (one direction per call)
 *   3. precompute_kernel     — Parallel precompute of projections (dt, B, C)
 *   4. parallel_scan_kernel  — Blelloch parallel prefix scan with Affine2x2
 *                              transform composition over paired RoPE dims.
 *                              Activates for N >= PSCAN_THRESHOLD (256).
 *   5. fused_elem_step       — GRU + multi-head PEER routing + expert MLP
 *                              + mu update + Adam + weight decay.
 *                              Expert weights loaded into shared memory.
 *
 * Plus sort via thrust::sort_by_key (serial) or cub::DeviceSegmentedRadixSort
 * (batched).
 *
 * Supports FP32, FP16, and BF16 parameter tensors.
 * All meta-net state (GRU, Mamba, weights) is always FP32.
 * Dimension guards: TORCH_CHECK for d_model/d_inner vs compile-time maximums.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <algorithm>

#include "types.h"


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: Input Projection + Sort Key Computation
//
//  Each thread handles one element:
//    x[idx, d] = input_proj_W[d, 0] * grad[idx] + input_proj_W[d, 1] * sharp[idx] + input_proj_b[d]
//    sort_key[idx] = |grad[idx]|
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void input_proj_sort_kernel(
    const scalar_t* __restrict__ grad,        // [N]
    const scalar_t* __restrict__ sharpness,   // [N]
    float* __restrict__ x_out,                // [N, d_model]
    float* __restrict__ sort_keys,            // [N]
    int* __restrict__ sort_indices,            // [N]
    const float* __restrict__ proj_W,         // [d_model, 2]
    const float* __restrict__ proj_b,         // [d_model]
    const int N,
    const int d_model
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;

    for (int d = 0; d < d_model; d++) {
        x_out[idx * d_model + d] = proj_W[d * 2] * g + proj_W[d * 2 + 1] * s + proj_b[d];
    }

    sort_keys[idx] = fabsf(g);
    sort_indices[idx] = idx;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Mamba-3 Selective Scan
//
//  One thread per d_inner dimension. Sequential scan over N timesteps.
//  State in registers: d_state floats per thread.
//
//  Trapezoidal discretization:
//    A_bar = (1 + dt*A/2) / (1 - dt*A/2)
//    B_bar = dt * B
//
//  RoPE rotation applied to state before A_bar multiplication.
//
//  Called twice: once for forward, once for reversed input (backward scan).
//
//  Threads: d_inner (typically 16)
//  Grid: 1 block
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_scan_kernel(
    const float* __restrict__ x_sorted,   // [N, d_model] — sorted input
    const float* __restrict__ in_proj_W,  // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,  // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,  // [d_inner]
    const float* __restrict__ B_proj_W,   // [d_state, d_inner]
    const float* __restrict__ C_proj_W,   // [d_state, d_inner]
    const float* __restrict__ A_log,      // [d_inner, d_state]
    const float* __restrict__ D_param,    // [d_inner]
    const float* __restrict__ rope_freq,  // [d_inner, d_state/2]
    float* __restrict__ scan_output,      // [N, d_inner]
    float* __restrict__ final_state,      // [d_inner, d_state]
    const float* __restrict__ initial_state, // [d_inner, d_state] or nullptr
    const int N,
    const int d_model,
    const int d_inner,
    const int d_state,
    const int reverse               // 0 = forward, 1 = reverse
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    // Shared memory layout: x_branch + cached projection weights
    extern __shared__ float smem[];
    float* s_x_branch = smem;                              // d_inner
    float* s_in_proj_W = s_x_branch + d_inner;             // 2 * d_inner * d_model
    float* s_dt_proj_W = s_in_proj_W + 2*d_inner*d_model;  // d_inner * d_inner
    float* s_dt_proj_b = s_dt_proj_W + d_inner*d_inner;    // d_inner
    float* s_B_proj_W = s_dt_proj_b + d_inner;             // d_state * d_inner
    float* s_C_proj_W = s_B_proj_W + d_state*d_inner;      // d_state * d_inner

    // Cooperatively load all projection weights into shared memory
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

    // State in registers — load from initial_state if provided
    float h[MAX_D_STATE];
    float h_snap[MAX_D_STATE]; // snapshot for RoPE (fixes read-after-write)
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++) h[s] = initial_state[tid * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++) h[s] = 0.0f;
    }

    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++) {
        A[s] = -expf(A_log[tid * d_state + s]);
    }
    for (int p = 0; p < half_d_state; p++) {
        freq[p] = rope_freq[tid * half_d_state + p];
    }
    float D_val = D_param[tid];

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        // Input projection: each thread computes its own x and z (shared memory weights)
        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = x_sorted[i * d_model + d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        // Write x_branch to shared memory for cross-thread access
        s_x_branch[tid] = x_val;
        __syncthreads();

        // FULL dt projection: dt[tid] = sum_j(dt_proj_W[tid, j] * x_branch[j]) + dt_proj_b[tid]
        float dt_raw = s_dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++) {
            dt_raw += s_dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        }
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw)); // stable softplus

        // Snapshot h for RoPE (fixes read-after-write)
        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        // State update with trapezoidal + paired RoPE
        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);

            // FULL B projection: B[s] = sum_j(B_proj_W[s, j] * x_branch[j])
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                B_val += s_B_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            float B_bar = dt_val * B_val;

            // Paired RoPE: (2i, 2i+1) form complex pairs
            int pair_idx = s / 2;
            float cos_p, sin_p;
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }

            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // FULL C projection for output: y = sum_s(h[s] * C[s])
        // C[s] = sum_j(C_proj_W[s, j] * x_branch[j])
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                C_val += s_C_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            y_val += h[s] * C_val;
        }

        // Gated output
        float silu_z = z_val / (1.0f + expf(-z_val));
        y_val = y_val * silu_z + D_val * x_val;

        scan_output[i * d_inner + tid] = y_val;
        __syncthreads(); // ensure all threads done before next step
    }

    for (int s = 0; s < d_state; s++)
        final_state[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Phase A: Parallel Scan Pre-computation
//
//  Pre-computes ALL cross-thread-dependent quantities for all N timesteps.
//  Each thread handles one timestep, computing:
//    - x_val[j] and z_val[j] for all d_inner j (input projection)
//    - dt_val[j] for all d_inner j (full dt projection, requires x_branch)
//    - B_val[s] for all d_state s (full B projection, requires x_branch)
//    - C_val[s] for all d_state s (full C projection, requires x_branch)
//
//  Grid:  ((N + 255) / 256, 1, 1)
//  Block: (256, 1, 1)
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_parallel_precompute_kernel(
    const float* __restrict__ x_sorted,     // [N, d_model]
    const float* __restrict__ in_proj_W,    // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,    // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,    // [d_inner]
    const float* __restrict__ B_proj_W,     // [d_state, d_inner]
    const float* __restrict__ C_proj_W,     // [d_state, d_inner]
    float* __restrict__ pre_x_val,          // [N, d_inner] output
    float* __restrict__ pre_z_val,          // [N, d_inner] output
    float* __restrict__ pre_dt_val,         // [N, d_inner] output
    float* __restrict__ pre_B_val,          // [N, d_state] output
    float* __restrict__ pre_C_val,          // [N, d_state] output
    const int N,
    const int d_model,
    const int d_inner,
    const int d_state
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    // Load input for this timestep
    float inp[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++)
        inp[d] = x_sorted[t * d_model + d];

    // Compute x_val[j] and z_val[j] for all d_inner
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

    // Compute dt_val[j] for all d_inner (full projection: dt = softplus(W @ x + b))
    for (int j = 0; j < d_inner; j++) {
        float dt_raw = dt_proj_b[j];
        for (int k = 0; k < d_inner; k++)
            dt_raw += dt_proj_W[j * d_inner + k] * x_branch[k];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        pre_dt_val[t * d_inner + j] = dt_val;
    }

    // Compute B_val[s] and C_val[s] for all d_state
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
//  Phase B+C: Parallel Prefix Scan with Blelloch Algorithm
//
//  Each block handles all N timesteps for one d_inner dimension (one j).
//  Pairs are processed sequentially within the block; within each pair,
//  the N timesteps are divided into chunks, scanned in parallel via
//  the Blelloch exclusive prefix scan, then the prefix is propagated.
//
//  Output y_val contributions are accumulated in scan_output via global
//  memory RMW (no contention: each [t, j] is written by exactly one thread).
//  After all pairs, gating (SiLU) and skip connection are applied.
//
//  Grid:  (d_inner, 1, 1) for single param
//  Block: (PSCAN_BLOCK, 1, 1)
//  Shared memory: 6 * PSCAN_BLOCK * sizeof(float) for Blelloch scan
//
//  Mathematical proof of correctness:
//    Sequential: h[t] = M[t] * h[t-1] + b[t]
//    Parallel:   h[t] = (M[t] ∘ M[t-1] ∘ ... ∘ M[0]) * h[-1] + b_cumulative[t]
//    where ∘ is affine_combine, which is associative.
//    The Blelloch scan computes all prefix compositions in O(N) work and O(log N) depth.
//    The chunk-based approach uses O(N/P) sequential work + O(log P) parallel depth.
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_parallel_scan_kernel(
    const float* __restrict__ pre_x_val,    // [N, d_inner]
    const float* __restrict__ pre_z_val,    // [N, d_inner]
    const float* __restrict__ pre_dt_val,   // [N, d_inner]
    const float* __restrict__ pre_B_val,    // [N, d_state]
    const float* __restrict__ pre_C_val,    // [N, d_state]
    const float* __restrict__ A_log,        // [d_inner, d_state]
    const float* __restrict__ D_param,      // [d_inner]
    const float* __restrict__ rope_freq,    // [d_inner, d_state/2]
    float* __restrict__ scan_output,        // [N, d_inner] — must be pre-zeroed
    float* __restrict__ final_state,        // [d_inner, d_state]
    const float* __restrict__ initial_state, // [d_inner, d_state] or nullptr
    const int N,
    const int d_inner,
    const int d_state,
    const int reverse
) {
    const int j = blockIdx.x;  // d_inner index for this block
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory: Affine2x2 array for Blelloch scan [num_threads × 6 floats]
    extern __shared__ float smem[];

    // Each thread handles a contiguous chunk of timesteps
    const int chunk_size = (N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load per-d_inner constants
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[j * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[j * half_d_state + p];
    float D_val = D_param[j];

    // Load initial state (read-only throughout)
    float h_init_all[MAX_D_STATE];
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = initial_state[j * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = 0.0f;
    }

    // Helper: build affine transform for timestep t, pair p
    // (inline lambda-like macro to avoid function call overhead)
    #define BUILD_AFFINE(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt = pre_dt_val[(t_idx) * d_inner + j]; \
        float x_val = pre_x_val[(t_idx) * d_inner + j]; \
        float B_e = pre_B_val[(t_idx) * d_state + (s_e)]; \
        float B_o = pre_B_val[(t_idx) * d_state + (s_o)]; \
        float A_bar_e = (1.0f + dt * (A_e) / 2.0f) / (1.0f - dt * (A_e) / 2.0f + 1e-8f); \
        float A_bar_o = (1.0f + dt * (A_o) / 2.0f) / (1.0f - dt * (A_o) / 2.0f + 1e-8f); \
        float cos_v, sin_v; \
        __sincosf(dt * (f_val), &sin_v, &cos_v); \
        (elem_out).m00 = A_bar_e * cos_v; \
        (elem_out).m01 = -A_bar_e * sin_v; \
        (elem_out).m10 = A_bar_o * sin_v; \
        (elem_out).m11 = A_bar_o * cos_v; \
        (elem_out).b0 = dt * B_e * x_val; \
        (elem_out).b1 = dt * B_o * x_val; \
    } while(0)

    // Process each pair sequentially
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];
        const float h_init_e = h_init_all[s_e];
        const float h_init_o = h_init_all[s_o];

        // === Step 1: Sequential scan within chunk → get chunk summary ===
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            Affine2x2 elem;
            BUILD_AFFINE(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine(summary, elem);
        }

        // Store summary in shared memory
        int base = ltid * 6;
        smem[base + 0] = summary.m00; smem[base + 1] = summary.m01;
        smem[base + 2] = summary.m10; smem[base + 3] = summary.m11;
        smem[base + 4] = summary.b0;  smem[base + 5] = summary.b1;
        __syncthreads();

        // === Step 2: Blelloch exclusive prefix scan on chunk summaries ===
        // Up-sweep (reduction phase)
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

        // Set last element to identity (for exclusive scan)
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
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
                // Swap and combine
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

        // Read exclusive prefix for this thread
        Affine2x2 prefix = {smem[ltid*6], smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // === Step 3: Re-scan chunk with prefix, compute output ===
        Affine2x2 running = prefix;
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);

            Affine2x2 elem;
            BUILD_AFFINE(t, A_e, A_o, f_val, s_e, s_o, elem);
            running = affine_combine(running, elem);

            // Compute h[t] from cumulative transform + initial state
            float h_e = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;

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

        __syncthreads(); // ensure all threads done before next pair
    }

    #undef BUILD_AFFINE

    // === Phase C: Apply SiLU gating and D skip connection ===
    for (int step = 0; step < my_count; step++) {
        int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
        float z = pre_z_val[t * d_inner + j];
        float silu_z = z / (1.0f + expf(-z));
        float x_val = pre_x_val[t * d_inner + j];
        scan_output[t * d_inner + j] = scan_output[t * d_inner + j] * silu_z + D_val * x_val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4b: Batched Parallel Prefix Scan
//
//  Single-launch version of mamba3_parallel_scan_kernel for all params.
//  Uses 2D grid: blockIdx.x = d_inner index, blockIdx.y = param index.
//  Offsets table provides per-param start/end into packed arrays.
//
//  Eliminates the per-param for-loop in the launcher, allowing all
//  params' scans to execute concurrently across SMs.
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_parallel_scan_batched_kernel(
    const float* __restrict__ pre_x_val,    // [total_N, d_inner]
    const float* __restrict__ pre_z_val,    // [total_N, d_inner]
    const float* __restrict__ pre_dt_val,   // [total_N, d_inner]
    const float* __restrict__ pre_B_val,    // [total_N, d_state]
    const float* __restrict__ pre_C_val,    // [total_N, d_state]
    const float* __restrict__ A_log,        // [d_inner, d_state]
    const float* __restrict__ D_param,      // [d_inner]
    const float* __restrict__ rope_freq,    // [d_inner, d_state/2]
    float* __restrict__ scan_output,        // [total_N, d_inner] — must be pre-zeroed
    float* __restrict__ final_states,       // [num_params, d_inner, d_state]
    const float* __restrict__ initial_states, // [num_params, d_inner, d_state]
    const int* __restrict__ offsets,         // [num_params + 1]
    const int* __restrict__ reverse_flags,   // [num_params]
    const int d_inner,
    const int d_state,
    const int num_params
) {
    const int j = blockIdx.x;           // d_inner index
    const int param_idx = blockIdx.y;   // parameter index
    if (j >= d_inner || param_idx >= num_params) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Get this param's range in packed arrays
    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;
    if (N == 0) return;
    const int reverse = reverse_flags[param_idx];

    // Shared memory: Affine2x2 array for Blelloch scan [num_threads × 6 floats]
    extern __shared__ float smem[];

    // Each thread handles a contiguous chunk of timesteps
    const int chunk_size = (N + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load per-d_inner constants
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[j * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_freq[j * half_d_state + p];
    float D_val = D_param[j];

    // Load initial state for this param
    float h_init_all[MAX_D_STATE];
    const float* init_ptr = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        h_init_all[s] = init_ptr[j * d_state + s];

    // Pointers into packed arrays for this param
    const float* my_pre_x = pre_x_val + start * d_inner;
    const float* my_pre_z = pre_z_val + start * d_inner;
    const float* my_pre_dt = pre_dt_val + start * d_inner;
    const float* my_pre_B = pre_B_val + start * d_state;
    const float* my_pre_C = pre_C_val + start * d_state;
    float* my_scan_out = scan_output + start * d_inner;

    // Final state pointer for this param
    float* fin_ptr = final_states + param_idx * d_inner * d_state;

    #define BUILD_AFFINE_BAT(t_idx, A_e, A_o, f_val, s_e, s_o, elem_out) do { \
        float dt = my_pre_dt[(t_idx) * d_inner + j]; \
        float x_v = my_pre_x[(t_idx) * d_inner + j]; \
        float B_e = my_pre_B[(t_idx) * d_state + (s_e)]; \
        float B_o = my_pre_B[(t_idx) * d_state + (s_o)]; \
        float A_bar_e = (1.0f + dt * (A_e) / 2.0f) / (1.0f - dt * (A_e) / 2.0f + 1e-8f); \
        float A_bar_o = (1.0f + dt * (A_o) / 2.0f) / (1.0f - dt * (A_o) / 2.0f + 1e-8f); \
        float cos_v, sin_v; \
        __sincosf(dt * (f_val), &sin_v, &cos_v); \
        (elem_out).m00 = A_bar_e * cos_v; \
        (elem_out).m01 = -A_bar_e * sin_v; \
        (elem_out).m10 = A_bar_o * sin_v; \
        (elem_out).m11 = A_bar_o * cos_v; \
        (elem_out).b0 = dt * B_e * x_v; \
        (elem_out).b1 = dt * B_o * x_v; \
    } while(0)

    // Process each pair sequentially
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];
        const float h_init_e = h_init_all[s_e];
        const float h_init_o = h_init_all[s_o];

        // === Step 1: Sequential scan within chunk → get chunk summary ===
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
            Affine2x2 elem;
            BUILD_AFFINE_BAT(t, A_e, A_o, f_val, s_e, s_o, elem);
            summary = affine_combine(summary, elem);
        }

        // Store summary in shared memory
        int base = ltid * 6;
        smem[base + 0] = summary.m00; smem[base + 1] = summary.m01;
        smem[base + 2] = summary.m10; smem[base + 3] = summary.m11;
        smem[base + 4] = summary.b0;  smem[base + 5] = summary.b1;
        __syncthreads();

        // === Step 2: Blelloch exclusive prefix scan on chunk summaries ===
        // Up-sweep (reduction phase)
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

        // Set last element to identity (for exclusive scan)
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
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
                // Swap and combine
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

        // Read exclusive prefix for this thread
        Affine2x2 prefix = {smem[ltid*6], smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // === Step 3: Re-scan chunk with prefix, compute output ===
        Affine2x2 running = prefix;
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);

            Affine2x2 elem;
            BUILD_AFFINE_BAT(t, A_e, A_o, f_val, s_e, s_o, elem);
            running = affine_combine(running, elem);

            // Compute h[t] from cumulative transform + initial state
            float h_e = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;

            // Accumulate y_val contribution from this pair
            float C_e = my_pre_C[t * d_state + s_e];
            float C_o = my_pre_C[t * d_state + s_o];
            my_scan_out[t * d_inner + j] += h_e * C_e + h_o * C_o;
        }

        // Last thread writes final state for this pair
        if (my_end == N && my_count > 0) {
            float h_e_final = running.m00 * h_init_e + running.m01 * h_init_o + running.b0;
            float h_o_final = running.m10 * h_init_e + running.m11 * h_init_o + running.b1;
            fin_ptr[j * d_state + s_e] = h_e_final;
            fin_ptr[j * d_state + s_o] = h_o_final;
        }

        __syncthreads(); // ensure all threads done before next pair
    }

    #undef BUILD_AFFINE_BAT

    // === Phase C: Apply SiLU gating and D skip connection ===
    for (int step = 0; step < my_count; step++) {
        int t = reverse ? (N - 1 - (my_start + step)) : (my_start + step);
        float z = my_pre_z[t * d_inner + j];
        float silu_z = z / (1.0f + expf(-z));
        float x_val = my_pre_x[t * d_inner + j];
        my_scan_out[t * d_inner + j] = my_scan_out[t * d_inner + j] * silu_z + D_val * x_val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Fused Per-Element Step
//
//  GRU + multi-head PEER routing + expert MLP + mu update + Adam
//
//  Each thread handles one gradient element.
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_elem_step_kernel(
    scalar_t* __restrict__ param,             // [N] — updated in-place
    const scalar_t* __restrict__ grad,        // [N] — raw gradient
    const scalar_t* __restrict__ sharpness,   // [N]
    float* __restrict__ exp_avg,              // [N] — FP32
    float* __restrict__ exp_avg_sq,           // [N] — FP32
    float* __restrict__ mu,                   // [N] — FP32 (stored as param dtype for compatibility)
    float* __restrict__ gru_state,            // [N, gru_hidden] — updated
    const float* __restrict__ fwd_scan_out,   // [N, d_inner] — in original order
    const float* __restrict__ bwd_scan_out,   // [N, d_inner] — in original order
    // Mamba out_proj weights
    const float* __restrict__ out_proj_fwd_W, // [d_model, d_inner]
    const float* __restrict__ out_proj_bwd_W, // [d_model, d_inner]
    // GRU weights
    const float* __restrict__ gru_Wz,         // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ gru_bz,         // [gru_hidden]
    const float* __restrict__ gru_Wr,         // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ gru_br,         // [gru_hidden]
    const float* __restrict__ gru_Wh,         // [gru_hidden, gru_input_dim + gru_hidden]
    const float* __restrict__ gru_bh,         // [gru_hidden]
    // PEER weights (flattened across heads)
    const float* __restrict__ peer_query_Ws,  // [num_heads, d_model, peer_input_dim]
    const float* __restrict__ prod_keys_A,    // [num_heads, pk_dim, d_model/2]
    const float* __restrict__ prod_keys_B,    // [num_heads, pk_dim, d_model/2]
    // Expert weights
    const float* __restrict__ expert_W1,      // [num_experts, expert_hidden]
    const float* __restrict__ expert_b1,      // [num_experts, expert_hidden]
    const float* __restrict__ expert_W2,      // [num_experts, expert_hidden]
    const float* __restrict__ expert_b2,      // [num_experts]
    // Scalars
    const float rescale,
    const float alpha,          // mu EMA alpha
    const float lamb_eff,       // lamb * ramp * gate_signal
    const float beta1,
    const float beta2,
    const float lr,
    const float wd_eff,
    const float eps,
    const float bc1,
    const float bc2,
    // Expert activation counter (nullable)
    int* __restrict__ expert_counts,      // [num_experts] or nullptr
    // Dims
    const int N,
    const int d_model,
    const int d_inner,
    const int gru_hidden,
    const int num_heads,
    const int pk_dim,
    const int expert_hidden,
    const int num_experts
) {
    // Shared memory layout for weight caching:
    //   [0 .. d_model*d_inner-1]: out_proj_fwd_W
    //   [d_model*d_inner .. 2*d_model*d_inner-1]: out_proj_bwd_W
    //   Then GRU weights: 3 matrices (Wz, Wr, Wh) + 3 biases (bz, br, bh)
    extern __shared__ float smem[];

    const int gru_input_dim = 2 + 2 * d_model;
    const int gru_row_len = gru_input_dim + gru_hidden;
    const int op_size = d_model * d_inner;
    const int gru_mat_size = gru_hidden * gru_row_len;

    // Pointers into shared memory
    float* s_out_fwd = smem;
    float* s_out_bwd = smem + op_size;
    float* s_gru_Wz = s_out_bwd + op_size;
    float* s_gru_Wr = s_gru_Wz + gru_mat_size;
    float* s_gru_Wh = s_gru_Wr + gru_mat_size;
    float* s_gru_bz = s_gru_Wh + gru_mat_size;
    float* s_gru_br = s_gru_bz + gru_hidden;
    float* s_gru_bh = s_gru_br + gru_hidden;
    // Expert weights in shared memory
    float* s_expert_W1 = s_gru_bh + gru_hidden;          // [num_experts * expert_hidden]
    float* s_expert_b1 = s_expert_W1 + num_experts * expert_hidden;
    float* s_expert_W2 = s_expert_b1 + num_experts * expert_hidden;
    float* s_expert_b2 = s_expert_W2 + num_experts * expert_hidden;
    // Total smem: 2*op_size + 3*gru_mat_size + 3*gru_hidden + 3*num_experts*expert_hidden + num_experts floats

    // Cooperative loading: each thread loads some elements
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Load out_proj weights
    for (int i = tid; i < 2 * op_size; i += block_size) {
        if (i < op_size)
            smem[i] = out_proj_fwd_W[i];
        else
            smem[i] = out_proj_bwd_W[i - op_size];
    }
    // Load GRU weights
    int gru_total = 3 * gru_mat_size + 3 * gru_hidden;
    float* gru_smem_start = s_gru_Wz;
    const float* gru_gmem[] = {gru_Wz, gru_Wr, gru_Wh, gru_bz, gru_br, gru_bh};
    int gru_sizes[] = {gru_mat_size, gru_mat_size, gru_mat_size, gru_hidden, gru_hidden, gru_hidden};
    int gru_offset = 0;
    for (int seg = 0; seg < 6; seg++) {
        for (int i = tid; i < gru_sizes[seg]; i += block_size)
            gru_smem_start[gru_offset + i] = gru_gmem[seg][i];
        gru_offset += gru_sizes[seg];
    }
    // Load expert weights
    for (int i = tid; i < num_experts * expert_hidden; i += block_size) {
        s_expert_W1[i] = expert_W1[i];
        s_expert_b1[i] = expert_b1[i];
        s_expert_W2[i] = expert_W2[i];
    }
    for (int i = tid; i < num_experts; i += block_size) {
        s_expert_b2[i] = expert_b2[i];
    }

    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    // Clamp NaN/Inf gradients to zero for robustness
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;
    const int half_d = d_model / 2;
    const int peer_input_dim = gru_hidden + 2 * d_model + 2;

    // 1. Apply Mamba out_proj to get fwd_ctx and bwd_ctx (using shared memory)
    // Preload scan outputs with float4 vectorized loads
    float fwd_scan[MAX_D_INNER];
    float bwd_scan[MAX_D_INNER];
    for (int j = 0; j < d_inner; j += 4) {
        float4 fwd4 = *reinterpret_cast<const float4*>(&fwd_scan_out[idx * d_inner + j]);
        float4 bwd4 = *reinterpret_cast<const float4*>(&bwd_scan_out[idx * d_inner + j]);
        fwd_scan[j] = fwd4.x; fwd_scan[j+1] = fwd4.y; fwd_scan[j+2] = fwd4.z; fwd_scan[j+3] = fwd4.w;
        bwd_scan[j] = bwd4.x; bwd_scan[j+1] = bwd4.y; bwd_scan[j+2] = bwd4.z; bwd_scan[j+3] = bwd4.w;
    }

    float fwd_ctx[MAX_D_MODEL], bwd_ctx[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        float fwd_val = 0.0f, bwd_val = 0.0f;
        for (int j = 0; j < d_inner; j++) {
            fwd_val += s_out_fwd[d * d_inner + j] * fwd_scan[j];
            bwd_val += s_out_bwd[d * d_inner + j] * bwd_scan[j];
        }
        fwd_ctx[d] = fwd_val;
        bwd_ctx[d] = bwd_val;
    }

    // 2. GRU update (using shared memory weights)
    float h_old[MAX_GRU_HIDDEN];
    for (int j = 0; j < gru_hidden; j++) {
        h_old[j] = gru_state[idx * gru_hidden + j];
    }

    float h_new[MAX_GRU_HIDDEN];
    float z_gate[MAX_GRU_HIDDEN], r_gate[MAX_GRU_HIDDEN];
    for (int j = 0; j < gru_hidden; j++) {
        float val_z = s_gru_bz[j];
        float val_r = s_gru_br[j];
        int offset = 0;
        val_z += s_gru_Wz[j * gru_row_len + 0] * g;
        val_z += s_gru_Wz[j * gru_row_len + 1] * s;
        val_r += s_gru_Wr[j * gru_row_len + 0] * g;
        val_r += s_gru_Wr[j * gru_row_len + 1] * s;
        offset = 2;
        for (int d = 0; d < d_model; d++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + d] * fwd_ctx[d];
            val_r += s_gru_Wr[j * gru_row_len + offset + d] * fwd_ctx[d];
        }
        offset += d_model;
        for (int d = 0; d < d_model; d++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + d] * bwd_ctx[d];
            val_r += s_gru_Wr[j * gru_row_len + offset + d] * bwd_ctx[d];
        }
        offset += d_model;
        for (int k = 0; k < gru_hidden; k++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + k] * h_old[k];
            val_r += s_gru_Wr[j * gru_row_len + offset + k] * h_old[k];
        }
        z_gate[j] = 1.0f / (1.0f + expf(-val_z));
        r_gate[j] = 1.0f / (1.0f + expf(-val_r));
    }

    // Candidate: h_tilde = tanh(Wh @ [x, r*h] + bh)
    for (int j = 0; j < gru_hidden; j++) {
        float val = s_gru_bh[j];
        int offset = 0;
        val += s_gru_Wh[j * gru_row_len + 0] * g;
        val += s_gru_Wh[j * gru_row_len + 1] * s;
        offset = 2;
        for (int d = 0; d < d_model; d++)
            val += s_gru_Wh[j * gru_row_len + offset + d] * fwd_ctx[d];
        offset += d_model;
        for (int d = 0; d < d_model; d++)
            val += s_gru_Wh[j * gru_row_len + offset + d] * bwd_ctx[d];
        offset += d_model;
        for (int k = 0; k < gru_hidden; k++)
            val += s_gru_Wh[j * gru_row_len + offset + k] * (r_gate[k] * h_old[k]);
        float h_tilde = tanhf(val);
        h_new[j] = (1.0f - z_gate[j]) * h_old[j] + z_gate[j] * h_tilde;
    }

    // Write GRU state
    for (int j = 0; j < gru_hidden; j++) {
        gru_state[idx * gru_hidden + j] = h_new[j];
    }

    // 3. Multi-head PEER routing + expert evaluation
    float total_out = 0.0f;

    for (int head = 0; head < num_heads; head++) {
        // Compute query for this head
        const float* pq_W = peer_query_Ws + head * d_model * peer_input_dim;
        float query[MAX_D_MODEL];
        for (int d = 0; d < d_model; d++) {
            float val = 0.0f;
            int off = 0;
            // h_new part
            for (int k = 0; k < gru_hidden; k++)
                val += pq_W[d * peer_input_dim + off + k] * h_new[k];
            off += gru_hidden;
            // fwd_ctx
            for (int k = 0; k < d_model; k++)
                val += pq_W[d * peer_input_dim + off + k] * fwd_ctx[k];
            off += d_model;
            // bwd_ctx
            for (int k = 0; k < d_model; k++)
                val += pq_W[d * peer_input_dim + off + k] * bwd_ctx[k];
            off += d_model;
            // g, s
            val += pq_W[d * peer_input_dim + off] * g;
            val += pq_W[d * peer_input_dim + off + 1] * s;
            query[d] = val;
        }

        // Product-key routing: argmax over sub-keys
        const float* keys_A = prod_keys_A + head * pk_dim * half_d;
        const float* keys_B = prod_keys_B + head * pk_dim * half_d;

        int best_a = 0;
        float best_score_a = -1e30f;
        for (int k = 0; k < pk_dim; k++) {
            float dot = 0.0f;
            for (int d = 0; d < half_d; d++)
                dot += query[d] * __ldg(&keys_A[k * half_d + d]);
            if (dot > best_score_a) { best_score_a = dot; best_a = k; }
        }

        int best_b = 0;
        float best_score_b = -1e30f;
        for (int k = 0; k < pk_dim; k++) {
            float dot = 0.0f;
            for (int d = 0; d < half_d; d++)
                dot += query[half_d + d] * __ldg(&keys_B[k * half_d + d]);
            if (dot > best_score_b) { best_score_b = dot; best_b = k; }
        }

        int expert_idx = best_a * pk_dim + best_b;
        if (expert_idx >= num_experts) expert_idx = num_experts - 1;
        if (expert_counts != nullptr)
            atomicAdd(&expert_counts[expert_idx], 1);

        // Expert MLP: z = relu(W1 * g + b1), out = W2 @ z + b2
        // Use shared memory for expert weight reads
        float head_out = s_expert_b2[expert_idx];
        for (int h = 0; h < expert_hidden; h++) {
            float z_val = s_expert_W1[expert_idx * expert_hidden + h] * g
                        + s_expert_b1[expert_idx * expert_hidden + h];
            z_val = fmaxf(z_val, 0.0f);  // ReLU
            head_out += s_expert_W2[expert_idx * expert_hidden + h] * z_val;
        }
        total_out += head_out;
    }

    // Average over heads
    float smart_grad = g + rescale * total_out / static_cast<float>(num_heads);

    // 4. mu update: mu = alpha * mu + (1 - alpha) * raw_grad
    float mu_val = mu[idx];
    mu_val = alpha * mu_val + (1.0f - alpha) * g;
    mu[idx] = mu_val;

    // 5. effective_grad = smart_grad + lamb_eff * mu
    float fg = smart_grad + lamb_eff * mu_val;

    // 6. Adam update
    float ea = exp_avg[idx];
    float easq = exp_avg_sq[idx];
    ea = beta1 * ea + (1.0f - beta1) * fg;
    easq = beta2 * easq + (1.0f - beta2) * fg * fg;
    exp_avg[idx] = ea;
    exp_avg_sq[idx] = easq;

    float step_size = lr / bc1;
    float denom = sqrtf(easq / bc2) + eps;
    float p_val = static_cast<float>(param[idx]);
    p_val = p_val * (1.0f - lr * wd_eff) - step_size * ea / denom;
    param[idx] = static_cast<scalar_t>(p_val);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: Batched Mamba-3 Scan
//
//  Processes multiple parameters' scans in parallel: one block per param.
//  Sorted data is packed contiguously with an offset table.
//
//  Grid:  (num_params, 1, 1)
//  Block: (d_inner, 1, 1)
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_scan_batched_kernel(
    const float* __restrict__ x_sorted_packed,    // [total_N, d_model]
    float* __restrict__ scan_output_packed,        // [total_N, d_inner]
    const float* __restrict__ initial_states,      // [num_params, d_inner, d_state]
    float* __restrict__ final_states,              // [num_params, d_inner, d_state]
    const int* __restrict__ offsets,               // [num_params + 1]
    const int* __restrict__ reverse_flags,         // [num_params]
    // Shared Mamba weights (same for all params)
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

    // Shared memory layout: x_branch + cached projection weights
    extern __shared__ float smem[];
    float* s_x_branch = smem;                              // d_inner
    float* s_in_proj_W = s_x_branch + d_inner;             // 2 * d_inner * d_model
    float* s_dt_proj_W = s_in_proj_W + 2*d_inner*d_model;  // d_inner * d_inner
    float* s_dt_proj_b = s_dt_proj_W + d_inner*d_inner;    // d_inner
    float* s_B_proj_W = s_dt_proj_b + d_inner;             // d_state * d_inner
    float* s_C_proj_W = s_B_proj_W + d_state*d_inner;      // d_state * d_inner

    // Cooperatively load all projection weights into shared memory
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

    // State in registers — load from initial_state
    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    const float* my_init = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++) {
        A[s] = -expf(A_log[tid * d_state + s]);
    }
    for (int p = 0; p < half_d_state; p++) {
        freq[p] = rope_freq[tid * half_d_state + p];
    }
    float D_val = D_param[tid];

    const float* my_x = x_sorted_packed + start * d_model;
    float* my_out = scan_output_packed + start * d_inner;

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = my_x[i * d_model + d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = s_dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += s_dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw)); // stable softplus

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += s_B_proj_W[s * d_inner + j] * s_x_branch[j];
            float B_bar = dt_val * B_val;
            // Paired RoPE: (2i, 2i+1) form complex pairs
            int pair_idx = s / 2;
            float cos_p, sin_p;
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

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
        __syncthreads();
    }

    float* my_final = final_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        my_final[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: Combined Forward + Backward Batched Scan
//
//  Grid = 2 * num_params: first num_params blocks do forward scan,
//  second num_params blocks do backward scan (reversed input).
//  This avoids two kernel launches and exploits GPU parallelism.
//
//  block_idx < num_params: forward, uses fwd weights
//  block_idx >= num_params: backward (reverse), uses bwd weights
// ═══════════════════════════════════════════════════════════════════════

__global__ void mamba3_scan_combined_kernel(
    const float* __restrict__ x_sorted_packed,    // [total_N, d_model]
    float* __restrict__ fwd_scan_output,          // [total_N, d_inner]
    float* __restrict__ bwd_scan_output,          // [total_N, d_inner]
    const float* __restrict__ fwd_initial_states, // [num_params, d_inner, d_state]
    const float* __restrict__ bwd_initial_states, // [num_params, d_inner, d_state]
    float* __restrict__ fwd_final_states,         // [num_params, d_inner, d_state]
    float* __restrict__ bwd_final_states,         // [num_params, d_inner, d_state]
    const int* __restrict__ offsets,              // [num_params + 1]
    // Forward weights
    const float* __restrict__ fwd_in_proj_W,
    const float* __restrict__ fwd_dt_proj_W,
    const float* __restrict__ fwd_dt_proj_b,
    const float* __restrict__ fwd_B_proj_W,
    const float* __restrict__ fwd_C_proj_W,
    const float* __restrict__ fwd_A_log,
    const float* __restrict__ fwd_D_param,
    const float* __restrict__ fwd_rope_freq,
    // Backward weights
    const float* __restrict__ bwd_in_proj_W,
    const float* __restrict__ bwd_dt_proj_W,
    const float* __restrict__ bwd_dt_proj_b,
    const float* __restrict__ bwd_B_proj_W,
    const float* __restrict__ bwd_C_proj_W,
    const float* __restrict__ bwd_A_log,
    const float* __restrict__ bwd_D_param,
    const float* __restrict__ bwd_rope_freq,
    const int d_model,
    const int d_inner,
    const int d_state,
    const int num_params
) {
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    const bool is_bwd = (block_id >= num_params);
    const int param_idx = is_bwd ? (block_id - num_params) : block_id;
    const int reverse = is_bwd ? 1 : 0;

    // Select weight set
    const float* in_proj_W   = is_bwd ? bwd_in_proj_W   : fwd_in_proj_W;
    const float* dt_proj_W   = is_bwd ? bwd_dt_proj_W   : fwd_dt_proj_W;
    const float* dt_proj_b   = is_bwd ? bwd_dt_proj_b   : fwd_dt_proj_b;
    const float* B_proj_W    = is_bwd ? bwd_B_proj_W    : fwd_B_proj_W;
    const float* C_proj_W    = is_bwd ? bwd_C_proj_W    : fwd_C_proj_W;
    const float* A_log_ptr   = is_bwd ? bwd_A_log       : fwd_A_log;
    const float* D_param_ptr = is_bwd ? bwd_D_param     : fwd_D_param;
    const float* rope_ptr    = is_bwd ? bwd_rope_freq   : fwd_rope_freq;
    float* scan_output       = is_bwd ? bwd_scan_output : fwd_scan_output;
    const float* init_states = is_bwd ? bwd_initial_states : fwd_initial_states;
    float* fin_states        = is_bwd ? bwd_final_states   : fwd_final_states;

    const int start = offsets[param_idx];
    const int end = offsets[param_idx + 1];
    const int N = end - start;

    // Shared memory layout: x_branch + cached projection weights
    extern __shared__ float smem[];
    float* s_x_branch = smem;                              // d_inner
    float* s_in_proj_W = s_x_branch + d_inner;             // 2 * d_inner * d_model
    float* s_dt_proj_W = s_in_proj_W + 2*d_inner*d_model;  // d_inner * d_inner
    float* s_dt_proj_b = s_dt_proj_W + d_inner*d_inner;    // d_inner
    float* s_B_proj_W = s_dt_proj_b + d_inner;             // d_state * d_inner
    float* s_C_proj_W = s_B_proj_W + d_state*d_inner;      // d_state * d_inner

    // Cooperatively load all projection weights into shared memory
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

    float h[MAX_D_STATE], h_snap[MAX_D_STATE];
    const float* my_init = init_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log_ptr[tid * d_state + s]);
    for (int p = 0; p < half_d_state; p++)
        freq[p] = rope_ptr[tid * half_d_state + p];
    float D_val = D_param_ptr[tid];

    const float* my_x = x_sorted_packed + start * d_model;
    float* my_out = scan_output + start * d_inner;

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = my_x[i * d_model + d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = s_dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += s_dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw)); // stable softplus

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += s_B_proj_W[s * d_inner + j] * s_x_branch[j];
            float B_bar = dt_val * B_val;
            int pair_idx = s / 2;
            float cos_p, sin_p;
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

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
        __syncthreads();
    }

    float* my_final = fin_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++)
        my_final[tid * d_state + s] = h[s];
}


// ═══════════════════════════════════════════════════════════════════════
//  Workspace cache: pre-allocated buffers that grow as needed
// ═══════════════════════════════════════════════════════════════════════
namespace {
struct ScanWorkspace {
    torch::Tensor x_proj;       // [max_N, d_model]
    torch::Tensor sort_keys;    // [max_N]
    torch::Tensor sort_indices; // [max_N]
    torch::Tensor fwd_scan;     // [max_N, d_inner]
    torch::Tensor bwd_scan;     // [max_N, d_inner]
    // Parallel scan precompute buffers
    torch::Tensor pre_x_val;    // [max_N, d_inner]
    torch::Tensor pre_z_val;    // [max_N, d_inner]
    torch::Tensor pre_dt_val;   // [max_N, d_inner]
    torch::Tensor pre_B_val;    // [max_N, d_state]
    torch::Tensor pre_C_val;    // [max_N, d_state]
    int max_N = 0;
    int d_model = 0;
    int d_inner = 0;
    int d_state = 0;

    void ensure(int N, int dm, int di, int ds, torch::Device dev) {
        if (N <= max_N && dm == d_model && di == d_inner && ds == d_state) return;
        int alloc_N = std::max(N, max_N);
        auto fo = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
        auto io = torch::TensorOptions().device(dev).dtype(torch::kInt32);
        x_proj = torch::empty({alloc_N, dm}, fo);
        sort_keys = torch::empty({alloc_N}, fo);
        sort_indices = torch::empty({alloc_N}, io);
        fwd_scan = torch::empty({alloc_N, di}, fo);
        bwd_scan = torch::empty({alloc_N, di}, fo);
        pre_x_val = torch::empty({alloc_N, di}, fo);
        pre_z_val = torch::empty({alloc_N, di}, fo);
        pre_dt_val = torch::empty({alloc_N, di}, fo);
        pre_B_val = torch::empty({alloc_N, ds}, fo);
        pre_C_val = torch::empty({alloc_N, ds}, fo);
        max_N = alloc_N;
        d_model = dm;
        d_inner = di;
        d_state = ds;
    }
};
static thread_local ScanWorkspace g_workspace;
} // namespace

// ═══════════════════════════════════════════════════════════════════════
//  Launcher: Full Mamba-3 + PEER step (single parameter)
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_step(
    torch::Tensor param,              // [N] — updated
    torch::Tensor grad,               // [N]
    torch::Tensor sharpness,          // [N]
    torch::Tensor exp_avg,            // [N] FP32
    torch::Tensor exp_avg_sq,         // [N] FP32
    torch::Tensor mu,                 // [N] FP32
    torch::Tensor gru_state,          // [N, gru_hidden] FP32
    torch::Tensor mamba_fwd_state,    // [d_inner, d_state] FP32 or empty
    torch::Tensor mamba_bwd_state,    // [d_inner, d_state] FP32 or empty
    // Input proj weights
    torch::Tensor input_proj_W,       // [d_model, 2]
    torch::Tensor input_proj_b,       // [d_model]
    // Mamba forward weights
    torch::Tensor mamba_fwd_in_proj,  // [2*d_inner, d_model]
    torch::Tensor mamba_fwd_dt_W,     // [d_inner, d_inner]
    torch::Tensor mamba_fwd_dt_b,     // [d_inner]
    torch::Tensor mamba_fwd_B_proj,   // [d_state, d_inner]
    torch::Tensor mamba_fwd_C_proj,   // [d_state, d_inner]
    torch::Tensor mamba_fwd_A_log,    // [d_inner, d_state]
    torch::Tensor mamba_fwd_D,        // [d_inner]
    torch::Tensor mamba_fwd_rope,     // [d_inner, d_state/2]
    torch::Tensor mamba_fwd_out_proj, // [d_model, d_inner]
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
    // GRU weights
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    // PEER weights (stacked across heads)
    torch::Tensor peer_query_Ws,      // [num_heads, d_model, peer_input_dim]
    torch::Tensor prod_keys_A,        // [num_heads, pk_dim, d_model/2]
    torch::Tensor prod_keys_B,        // [num_heads, pk_dim, d_model/2]
    // Expert weights
    torch::Tensor expert_W1,          // [num_experts, expert_hidden]
    torch::Tensor expert_b1,          // [num_experts, expert_hidden]
    torch::Tensor expert_W2,          // [num_experts, expert_hidden]
    torch::Tensor expert_b2,          // [num_experts]
    // Scalars
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    // Dims
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    const int N = grad.numel();
    if (N == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(gru_hidden <= MAX_GRU_HIDDEN, "gru_hidden exceeds MAX_GRU_HIDDEN (", gru_hidden, " > ", MAX_GRU_HIDDEN, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");
    TORCH_CHECK(d_inner % 4 == 0, "d_inner must be a multiple of 4 for vectorized loads (got ", d_inner, ")");

    auto dev = grad.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Pre-allocated workspace (grows as needed, reused across calls)
    g_workspace.ensure(N, d_model, d_inner, d_state, dev);

    // Step 1: Input projection + sort key computation
    auto x_proj = g_workspace.x_proj.narrow(0, 0, N);
    auto sort_keys = g_workspace.sort_keys.narrow(0, 0, N);
    auto sort_indices = g_workspace.sort_indices.narrow(0, 0, N);

    {
        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grad.scalar_type(), "input_proj_sort", ([&] {
            input_proj_sort_kernel<scalar_t><<<grid, SG2M_BLOCK>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                x_proj.data_ptr<float>(),
                sort_keys.data_ptr<float>(),
                sort_indices.data_ptr<int>(),
                input_proj_W.data_ptr<float>(),
                input_proj_b.data_ptr<float>(),
                N, d_model
            );
        }));
    }

    // Sort by |grad| magnitude
    {
        thrust::device_ptr<float> keys_ptr(sort_keys.data_ptr<float>());
        thrust::device_ptr<int> indices_ptr(sort_indices.data_ptr<int>());
        thrust::sort_by_key(keys_ptr, keys_ptr + N, indices_ptr);
    }

    // Gather sorted x_proj
    auto x_sorted = torch::empty({N, d_model}, float_opts);
    {
        auto idx_tensor = sort_indices.to(torch::kLong);
        x_sorted = x_proj.index_select(0, idx_tensor);
    }

    // Step 2: Bidirectional Mamba-3 scan (reuse workspace for scan outputs)
    auto fwd_scan_out = g_workspace.fwd_scan.narrow(0, 0, N);
    auto new_fwd_state = torch::empty({d_inner, d_state}, float_opts);
    auto bwd_scan_out = g_workspace.bwd_scan.narrow(0, 0, N);
    auto new_bwd_state = torch::empty({d_inner, d_state}, float_opts);

    const float* fwd_init_ptr = (mamba_fwd_state.numel() > 0) ?
        mamba_fwd_state.data_ptr<float>() : nullptr;
    const float* bwd_init_ptr = (mamba_bwd_state.numel() > 0) ?
        mamba_bwd_state.data_ptr<float>() : nullptr;

    if (N >= PSCAN_THRESHOLD) {
        // ===== PARALLEL PREFIX SCAN (Blelloch) for large N =====
        // Phase A: Precompute all cross-thread-dependent values

        // Helper lambda to run parallel scan for one direction
        auto run_parallel_scan = [&](
            const float* in_proj_W, const float* dt_proj_W, const float* dt_proj_b,
            const float* B_proj_W, const float* C_proj_W,
            const float* A_log_ptr, const float* D_param_ptr, const float* rope_ptr,
            float* scan_out, float* new_state, const float* init_ptr, int rev
        ) {
            auto pre_x = g_workspace.pre_x_val.narrow(0, 0, N);
            auto pre_z = g_workspace.pre_z_val.narrow(0, 0, N);
            auto pre_dt = g_workspace.pre_dt_val.narrow(0, 0, N);
            auto pre_B = g_workspace.pre_B_val.narrow(0, 0, N);
            auto pre_C = g_workspace.pre_C_val.narrow(0, 0, N);

            // Phase A: Precompute
            const int pre_grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
            mamba3_parallel_precompute_kernel<<<pre_grid, SG2M_BLOCK>>>(
                x_sorted.data_ptr<float>(),
                in_proj_W, dt_proj_W, dt_proj_b, B_proj_W, C_proj_W,
                pre_x.data_ptr<float>(),
                pre_z.data_ptr<float>(),
                pre_dt.data_ptr<float>(),
                pre_B.data_ptr<float>(),
                pre_C.data_ptr<float>(),
                N, d_model, d_inner, d_state
            );

            // Zero scan output before parallel scan accumulates into it
            cudaMemsetAsync(scan_out, 0, N * d_inner * sizeof(float));

            // Phase B+C: Parallel prefix scan + output
            int pscan_smem = 6 * PSCAN_BLOCK * sizeof(float);
            int actual_block = std::min(PSCAN_BLOCK, N);
            // Round up to next power of 2 for Blelloch
            int block_po2 = 1;
            while (block_po2 < actual_block) block_po2 *= 2;
            block_po2 = std::min(block_po2, PSCAN_BLOCK);
            pscan_smem = 6 * block_po2 * sizeof(float);

            mamba3_parallel_scan_kernel<<<d_inner, block_po2, pscan_smem>>>(
                pre_x.data_ptr<float>(),
                pre_z.data_ptr<float>(),
                pre_dt.data_ptr<float>(),
                pre_B.data_ptr<float>(),
                pre_C.data_ptr<float>(),
                A_log_ptr, D_param_ptr, rope_ptr,
                scan_out, new_state, init_ptr,
                N, d_inner, d_state, rev
            );
        };

        // Forward scan
        run_parallel_scan(
            mamba_fwd_in_proj.data_ptr<float>(),
            mamba_fwd_dt_W.data_ptr<float>(),
            mamba_fwd_dt_b.data_ptr<float>(),
            mamba_fwd_B_proj.data_ptr<float>(),
            mamba_fwd_C_proj.data_ptr<float>(),
            mamba_fwd_A_log.data_ptr<float>(),
            mamba_fwd_D.data_ptr<float>(),
            mamba_fwd_rope.data_ptr<float>(),
            fwd_scan_out.data_ptr<float>(),
            new_fwd_state.data_ptr<float>(),
            fwd_init_ptr, 0
        );

        // Backward scan (reverse direction)
        run_parallel_scan(
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            bwd_scan_out.data_ptr<float>(),
            new_bwd_state.data_ptr<float>(),
            bwd_init_ptr, 1
        );
    } else {
        // ===== SEQUENTIAL SCAN for small N (N < PSCAN_THRESHOLD) =====
        int scan_smem = (d_inner + 2*d_inner*d_model + d_inner*d_inner + d_inner + 2*d_state*d_inner) * sizeof(float);

        mamba3_scan_kernel<<<1, d_inner, scan_smem>>>(
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
            new_fwd_state.data_ptr<float>(),
            fwd_init_ptr,
            N, d_model, d_inner, d_state, 0
        );

        mamba3_scan_kernel<<<1, d_inner, scan_smem>>>(
            x_sorted.data_ptr<float>(),
            mamba_bwd_in_proj.data_ptr<float>(),
            mamba_bwd_dt_W.data_ptr<float>(),
            mamba_bwd_dt_b.data_ptr<float>(),
            mamba_bwd_B_proj.data_ptr<float>(),
            mamba_bwd_C_proj.data_ptr<float>(),
            mamba_bwd_A_log.data_ptr<float>(),
            mamba_bwd_D.data_ptr<float>(),
            mamba_bwd_rope.data_ptr<float>(),
            bwd_scan_out.data_ptr<float>(),
            new_bwd_state.data_ptr<float>(),
            bwd_init_ptr,
            N, d_model, d_inner, d_state, 1
        );
    }

    // Copy final states back
    if (mamba_fwd_state.numel() > 0) {
        mamba_fwd_state.copy_(new_fwd_state);
    }
    if (mamba_bwd_state.numel() > 0) {
        mamba_bwd_state.copy_(new_bwd_state);
    }

    // Unsort scan outputs back to original order
    auto unsort_indices = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kLong));
    {
        auto sort_idx_long = sort_indices.to(torch::kLong);
        unsort_indices.scatter_(0, sort_idx_long,
            torch::arange(N, torch::TensorOptions().device(dev).dtype(torch::kLong)));
    }
    auto fwd_unsorted = fwd_scan_out.index_select(0, unsort_indices);
    auto bwd_unsorted = bwd_scan_out.index_select(0, unsort_indices);

    // Step 3: Fused per-element step (GRU + PEER + Expert + Adam)
    {
        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        // Shared memory: out_proj (2 * d_model * d_inner) + GRU (3 matrices + 3 biases) + expert weights
        int gru_input_dim_val = 2 + 2 * d_model;
        int gru_row_len = gru_input_dim_val + gru_hidden;
        int smem_bytes = (2 * d_model * d_inner
                        + 3 * gru_hidden * gru_row_len
                        + 3 * gru_hidden
                        + 3 * num_experts * expert_hidden + num_experts) * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            param.scalar_type(), "fused_elem_step", ([&] {
            fused_elem_step_kernel<scalar_t><<<grid, SG2M_BLOCK, smem_bytes>>>(
                param.data_ptr<scalar_t>(),
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                exp_avg.data_ptr<float>(),
                exp_avg_sq.data_ptr<float>(),
                mu.data_ptr<float>(),
                gru_state.data_ptr<float>(),
                fwd_unsorted.data_ptr<float>(),
                bwd_unsorted.data_ptr<float>(),
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
                rescale, alpha_mu, lamb_eff,
                beta1, beta2, lr, wd_eff, eps, bc1, bc2,
                expert_counts.data_ptr<int>(),
                N, d_model, d_inner, gru_hidden,
                num_heads, pk_dim, expert_hidden, num_experts
            );
        }));
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Launcher: Batched Mamba-3 + PEER step (all parameters at once)
//
//  Takes vectors of per-parameter tensors, concatenates sorted data,
//  launches batched scan, then per-parameter fused_elem_step.
// ═══════════════════════════════════════════════════════════════════════

void launch_mamba3_peer_batched_step(
    std::vector<torch::Tensor> params,          // [num_params] each [N_i]
    std::vector<torch::Tensor> grads,           // [num_params] each [N_i]
    std::vector<torch::Tensor> sharpness_list,  // [num_params] each [N_i]
    std::vector<torch::Tensor> exp_avgs,        // [num_params] each [N_i] FP32
    std::vector<torch::Tensor> exp_avg_sqs,     // [num_params] each [N_i] FP32
    std::vector<torch::Tensor> mus,             // [num_params] each [N_i] FP32
    std::vector<torch::Tensor> gru_states,      // [num_params] each [N_i, gru_hidden] FP32
    std::vector<torch::Tensor> mamba_fwd_states, // [num_params] each [d_inner, d_state] FP32
    std::vector<torch::Tensor> mamba_bwd_states, // [num_params] each [d_inner, d_state] FP32
    // Meta-net weights (shared across all params)
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
    // Per-parameter scalars (vectors)
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    // Shared scalars
    float rescale, float beta2, float lr, float wd_eff, float eps,
    // Dims
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    const int num_params = params.size();
    if (num_params == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");
    TORCH_CHECK(d_model <= MAX_D_MODEL, "d_model exceeds MAX_D_MODEL (", d_model, " > ", MAX_D_MODEL, ")");
    TORCH_CHECK(gru_hidden <= MAX_GRU_HIDDEN, "gru_hidden exceeds MAX_GRU_HIDDEN (", gru_hidden, " > ", MAX_GRU_HIDDEN, ")");
    TORCH_CHECK(d_inner <= MAX_D_INNER, "d_inner exceeds MAX_D_INNER (", d_inner, " > ", MAX_D_INNER, ")");
    TORCH_CHECK(d_inner % 4 == 0, "d_inner must be a multiple of 4 for vectorized loads (got ", d_inner, ")");

    auto dev = grads[0].device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Step 1: Input projection + CUB segmented sort for all params
    std::vector<int> N_vec(num_params);
    std::vector<torch::Tensor> x_proj_list(num_params);
    std::vector<torch::Tensor> x_sorted_list(num_params);
    std::vector<torch::Tensor> sort_idx_list(num_params);
    std::vector<torch::Tensor> unsort_idx_list(num_params);
    int total_N = 0;

    // First pass: compute total_N and per-param sizes
    for (int p = 0; p < num_params; p++) {
        N_vec[p] = grads[p].numel();
        total_N += N_vec[p];
    }
    if (total_N == 0) return;

    // Build segment offsets for CUB segmented sort
    std::vector<int> seg_offsets_cpu(num_params + 1);
    seg_offsets_cpu[0] = 0;
    for (int p = 0; p < num_params; p++)
        seg_offsets_cpu[p + 1] = seg_offsets_cpu[p] + N_vec[p];

    // Allocate packed keys/indices for all params
    auto all_keys = torch::empty({total_N}, float_opts);
    auto all_indices = torch::empty({total_N}, int_opts);
    auto all_keys_out = torch::empty({total_N}, float_opts);
    auto all_indices_out = torch::empty({total_N}, int_opts);

    // Run input_proj_sort for all params, writing into packed arrays
    for (int p = 0; p < num_params; p++) {
        int N = N_vec[p];
        if (N == 0) continue;
        int off = seg_offsets_cpu[p];

        auto x_proj = torch::empty({N, d_model}, float_opts);
        x_proj_list[p] = x_proj;

        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grads[p].scalar_type(), "input_proj_sort_batch", ([&] {
            input_proj_sort_kernel<scalar_t><<<grid, SG2M_BLOCK>>>(
                grads[p].data_ptr<scalar_t>(),
                sharpness_list[p].data_ptr<scalar_t>(),
                x_proj.data_ptr<float>(),
                all_keys.data_ptr<float>() + off,
                all_indices.data_ptr<int>() + off,
                input_proj_W.data_ptr<float>(),
                input_proj_b.data_ptr<float>(),
                N, d_model
            );
        }));
    }

    // CUB segmented sort: sort all params' keys+indices in a single call
    auto seg_offsets_t = torch::from_blob(seg_offsets_cpu.data(), {num_params + 1},
        torch::kInt32).to(dev).contiguous();

    size_t cub_temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, cub_temp_bytes,
        all_keys.data_ptr<float>(), all_keys_out.data_ptr<float>(),
        all_indices.data_ptr<int>(), all_indices_out.data_ptr<int>(),
        total_N, num_params,
        seg_offsets_t.data_ptr<int>(), seg_offsets_t.data_ptr<int>() + 1);

    auto cub_temp = torch::empty({(int64_t)cub_temp_bytes},
        torch::TensorOptions().device(dev).dtype(torch::kUInt8));
    cub::DeviceSegmentedRadixSort::SortPairs(
        cub_temp.data_ptr<void>(), cub_temp_bytes,
        all_keys.data_ptr<float>(), all_keys_out.data_ptr<float>(),
        all_indices.data_ptr<int>(), all_indices_out.data_ptr<int>(),
        total_N, num_params,
        seg_offsets_t.data_ptr<int>(), seg_offsets_t.data_ptr<int>() + 1);

    // Extract per-param sorted data
    for (int p = 0; p < num_params; p++) {
        int N = N_vec[p];
        if (N == 0) continue;
        int off = seg_offsets_cpu[p];

        sort_idx_list[p] = all_indices_out.narrow(0, off, N);
        auto idx_long = sort_idx_list[p].to(torch::kLong);
        x_sorted_list[p] = x_proj_list[p].index_select(0, idx_long);

        auto unsort = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kLong));
        unsort.scatter_(0, idx_long,
            torch::arange(N, torch::TensorOptions().device(dev).dtype(torch::kLong)));
        unsort_idx_list[p] = unsort;
    }

    // Step 2: Pack sorted data (reuse seg_offsets from step 1)
    auto& offsets_cpu = seg_offsets_cpu;
    auto offsets_t = seg_offsets_t;

    // Concatenate sorted data
    std::vector<torch::Tensor> valid_sorted;
    for (int p = 0; p < num_params; p++) {
        if (N_vec[p] > 0) valid_sorted.push_back(x_sorted_list[p]);
    }
    auto x_sorted_packed = torch::cat(valid_sorted, 0);

    // Pack initial states
    auto initial_fwd = torch::stack(mamba_fwd_states, 0);  // [num_params, d_inner, d_state]
    auto initial_bwd = torch::stack(mamba_bwd_states, 0);
    auto final_fwd = torch::empty_like(initial_fwd);
    auto final_bwd = torch::empty_like(initial_bwd);

    // Scan outputs
    auto fwd_scan_packed = torch::empty({total_N, d_inner}, float_opts);
    auto bwd_scan_packed = torch::empty({total_N, d_inner}, float_opts);

    int max_N = *std::max_element(N_vec.begin(), N_vec.end());

    if (max_N >= PSCAN_THRESHOLD) {
        // ===== PARALLEL PREFIX SCAN for large N =====
        // Phase A: Precompute all timestep quantities (shared weights, all params at once)
        auto pre_x = torch::empty({total_N, d_inner}, float_opts);
        auto pre_z = torch::empty({total_N, d_inner}, float_opts);
        auto pre_dt = torch::empty({total_N, d_inner}, float_opts);
        auto pre_B = torch::empty({total_N, d_state}, float_opts);
        auto pre_C = torch::empty({total_N, d_state}, float_opts);

        // Process each direction (fwd/bwd share same input data but different weights)
        // Compute block size from max_N (shared across all params)
        int block_po2 = 1;
        int actual_block = std::min(PSCAN_BLOCK, max_N);
        while (block_po2 < actual_block) block_po2 *= 2;
        block_po2 = std::min(block_po2, PSCAN_BLOCK);
        int pscan_smem = 6 * block_po2 * (int)sizeof(float);
        dim3 pscan_grid(d_inner, num_params);

        auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
        auto rev_fwd = torch::zeros({num_params}, int_opts);
        auto rev_bwd = torch::ones({num_params}, int_opts);

        auto run_batched_parallel_scan = [&](
            torch::Tensor in_proj_W, torch::Tensor dt_proj_W, torch::Tensor dt_proj_b,
            torch::Tensor B_proj_W, torch::Tensor C_proj_W,
            torch::Tensor A_log_t, torch::Tensor D_param_t, torch::Tensor rope_t,
            torch::Tensor initial_states_t, torch::Tensor final_states_t,
            torch::Tensor scan_packed, torch::Tensor rev_flags
        ) {
            // Phase A: precompute for all packed timesteps (all params share weights)
            const int pre_grid = (total_N + SG2M_BLOCK - 1) / SG2M_BLOCK;
            mamba3_parallel_precompute_kernel<<<pre_grid, SG2M_BLOCK>>>(
                x_sorted_packed.data_ptr<float>(),
                in_proj_W.data_ptr<float>(),
                dt_proj_W.data_ptr<float>(),
                dt_proj_b.data_ptr<float>(),
                B_proj_W.data_ptr<float>(),
                C_proj_W.data_ptr<float>(),
                pre_x.data_ptr<float>(),
                pre_z.data_ptr<float>(),
                pre_dt.data_ptr<float>(),
                pre_B.data_ptr<float>(),
                pre_C.data_ptr<float>(),
                total_N, d_model, d_inner, d_state
            );

            // Zero scan output
            cudaMemsetAsync(scan_packed.data_ptr<float>(), 0,
                total_N * d_inner * sizeof(float));

            // Phase B+C: single-launch batched parallel scan (all params at once)
            mamba3_parallel_scan_batched_kernel<<<pscan_grid, block_po2, pscan_smem>>>(
                pre_x.data_ptr<float>(),
                pre_z.data_ptr<float>(),
                pre_dt.data_ptr<float>(),
                pre_B.data_ptr<float>(),
                pre_C.data_ptr<float>(),
                A_log_t.data_ptr<float>(),
                D_param_t.data_ptr<float>(),
                rope_t.data_ptr<float>(),
                scan_packed.data_ptr<float>(),
                final_states_t.data_ptr<float>(),
                initial_states_t.data_ptr<float>(),
                offsets_t.data_ptr<int>(),
                rev_flags.data_ptr<int>(),
                d_inner, d_state, num_params
            );
        };

        // Forward scan
        run_batched_parallel_scan(
            mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj,
            mamba_fwd_A_log, mamba_fwd_D, mamba_fwd_rope,
            initial_fwd, final_fwd, fwd_scan_packed, rev_fwd
        );

        // Backward scan (reverse)
        run_batched_parallel_scan(
            mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj,
            mamba_bwd_A_log, mamba_bwd_D, mamba_bwd_rope,
            initial_bwd, final_bwd, bwd_scan_packed, rev_bwd
        );
    } else {
        // ===== SEQUENTIAL COMBINED SCAN for small N =====
        int scan_smem = (d_inner + 2*d_inner*d_model + d_inner*d_inner + d_inner + 2*d_state*d_inner) * sizeof(float);

        mamba3_scan_combined_kernel<<<2 * num_params, d_inner, scan_smem>>>(
            x_sorted_packed.data_ptr<float>(),
            fwd_scan_packed.data_ptr<float>(),
            bwd_scan_packed.data_ptr<float>(),
            initial_fwd.data_ptr<float>(),
            initial_bwd.data_ptr<float>(),
            final_fwd.data_ptr<float>(),
            final_bwd.data_ptr<float>(),
            offsets_t.data_ptr<int>(),
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
            d_model, d_inner, d_state, num_params
        );
    }

    // Step 5: Copy final states back + unsort + fused_elem_step per param
    // Pre-compute all unsorted scan outputs, then launch kernels on streams
    int gru_input_dim_val = 2 + 2 * d_model;
    int gru_row_len = gru_input_dim_val + gru_hidden;
    int smem_bytes = (2 * d_model * d_inner
                    + 3 * gru_hidden * gru_row_len
                    + 3 * gru_hidden
                    + 3 * num_experts * expert_hidden + num_experts) * sizeof(float);

    // Pre-compute unsorted scan outputs and copy final states
    std::vector<torch::Tensor> fwd_unsorted_list(num_params);
    std::vector<torch::Tensor> bwd_unsorted_list(num_params);
    for (int p = 0; p < num_params; p++) {
        int N = N_vec[p];
        if (N == 0) continue;
        int off = offsets_cpu[p];

        mamba_fwd_states[p].copy_(final_fwd[p]);
        mamba_bwd_states[p].copy_(final_bwd[p]);

        auto fwd_slice = fwd_scan_packed.narrow(0, off, N);
        auto bwd_slice = bwd_scan_packed.narrow(0, off, N);
        fwd_unsorted_list[p] = fwd_slice.index_select(0, unsort_idx_list[p]);
        bwd_unsorted_list[p] = bwd_slice.index_select(0, unsort_idx_list[p]);
    }

    // Launch fused_elem_step kernels on a persistent pool of streams
    constexpr int NUM_STREAMS = 4;
    static cudaStream_t streams[NUM_STREAMS] = {};
    static bool streams_initialized = false;
    if (!streams_initialized) {
        for (int s = 0; s < NUM_STREAMS; s++)
            cudaStreamCreate(&streams[s]);
        streams_initialized = true;
    }

    for (int p = 0; p < num_params; p++) {
        int N = N_vec[p];
        if (N == 0) continue;

        cudaStream_t stream = streams[p % NUM_STREAMS];
        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            params[p].scalar_type(), "fused_elem_step_batch", ([&] {
            fused_elem_step_kernel<scalar_t><<<grid, SG2M_BLOCK, smem_bytes, stream>>>(
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

    // Sync all streams (persistent — no destroy)
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
    }
}
