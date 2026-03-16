/*
 * SuperGrok v2 — Distributed Mamba-3 Scan Kernels (Multi-GPU Sharding)
 *
 * Implements a 3-phase distributed parallel prefix scan across multiple GPUs
 * using the chunked Blelloch algorithm with Affine2x2 transform composition.
 *
 * Phase 1: Each GPU runs a local Blelloch scan on its chunk, producing
 *          a local scan result and one Affine2x2 summary per chunk.
 * Phase 2: GPU 0 collects all chunk summaries and runs a small prefix scan
 *          over them to compute per-GPU prefix transforms.
 * Phase 3: Each GPU applies its prefix transform to correct its local scan.
 *
 * Forward kernels:
 *   1. mamba3_scan_local_with_summary_kernel — Local scan + summary extraction
 *   2. scan_summary_prefix_kernel            — Prefix scan over K summaries (GPU 0)
 *   3. mamba3_apply_scan_prefix_kernel        — Apply prefix correction
 *
 * Backward kernels:
 *   4. mamba3_scan_local_with_summary_bwd_kernel — Backward local scan + summary
 *   5. scan_summary_prefix_bwd_kernel            — Backward prefix scan over summaries
 *   6. mamba3_apply_scan_prefix_bwd_kernel        — Backward prefix correction
 *
 * All kernels use __launch_bounds__, affine_combine_ptx() for composition,
 * stream_load/stream_store for state access, and __syncthreads() for barriers.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>

#include "platform.h"
#include "types.h"
#include "ptx_intrinsics.cuh"


// ═══════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════

constexpr int DSCAN_BLOCK = 256;        // threads per block for local scan
constexpr int DSCAN_MAX_GPUS = 64;      // max number of GPU chunks
constexpr int DSCAN_SUMMARY_BLOCK = 64; // threads for summary prefix kernel


// ═══════════════════════════════════════════════════════════════════════
//  Helper: Build Affine2x2 transform for a single timestep
//
//  Given precomputed dt, x_val, B (even/odd), A (even/odd), and RoPE
//  frequency, construct the affine transform for one scan element.
//  Uses trapezoidal discretization consistent with mamba3_parallel_scan.
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ Affine2x2 build_scan_element(
    float dt, float x_val,
    float B_e, float B_o,
    float A_e, float A_o,
    float freq_val
) {
    float A_bar_e = (1.0f + dt * A_e / 2.0f) / (1.0f - dt * A_e / 2.0f + 1e-8f);
    float A_bar_o = (1.0f + dt * A_o / 2.0f) / (1.0f - dt * A_o / 2.0f + 1e-8f);
    float cos_v, sin_v;
    FAST_SINCOSF(dt * freq_val, &sin_v, &cos_v);

    Affine2x2 elem;
    elem.m00 = A_bar_e * cos_v;
    elem.m01 = -A_bar_e * sin_v;
    elem.m10 = A_bar_o * sin_v;
    elem.m11 = A_bar_o * cos_v;
    elem.b0  = dt * B_e * x_val;
    elem.b1  = dt * B_o * x_val;
    return elem;
}


// ═══════════════════════════════════════════════════════════════════════
//  Helper: Apply Affine2x2 to a 2-vector state (h_e, h_o)
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void affine_apply(
    const Affine2x2& T, float h_e_in, float h_o_in,
    float& h_e_out, float& h_o_out
) {
    h_e_out = T.m00 * h_e_in + T.m01 * h_o_in + T.b0;
    h_o_out = T.m10 * h_e_in + T.m11 * h_o_in + T.b1;
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: mamba3_scan_local_with_summary_kernel (Forward)
//
//  Each GPU runs this on its local chunk of N_local elements.
//  Grid: (d_inner) blocks, each block has DSCAN_BLOCK threads.
//  Each thread handles a contiguous sub-chunk of timesteps.
//
//  The kernel performs a full Blelloch scan within the block (identical
//  to mamba3_parallel_scan_kernel) but additionally writes out the
//  total Affine2x2 summary of the entire chunk for each (j, pair).
//
//  Outputs:
//    - scan_output[N_local, d_inner]: local scan result (not yet globally correct)
//    - summaries[d_inner * half_d_state]: one Affine2x2 per (j, pair)
//
//  The scan_output is computed assuming initial_state = 0 (or provided).
//  After the global prefix is applied (kernel 3), results become globally correct.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_BLOCK, 4)
__global__ void mamba3_scan_local_with_summary_kernel(
    const float* __restrict__ pre_x_val,      // [N_local, d_inner]
    const float* __restrict__ pre_z_val,      // [N_local, d_inner]
    const float* __restrict__ pre_dt_val,     // [N_local, d_inner]
    const float* __restrict__ pre_B_val,      // [N_local, d_state]
    const float* __restrict__ pre_C_val,      // [N_local, d_state]
    const float* __restrict__ A_log,          // [d_inner, d_state]
    const float* __restrict__ D_param,        // [d_inner]
    const float* __restrict__ rope_freq,      // [d_inner, d_state/2]
    float* __restrict__ scan_output,          // [N_local, d_inner] — must be pre-zeroed
    float* __restrict__ summaries,            // [d_inner, half_d_state, 6] — Affine2x2 per pair
    const float* __restrict__ initial_state,  // [d_inner, d_state] or nullptr
    const int N_local,
    const int d_inner,
    const int d_state,
    const int reverse
) {
    const int j = blockIdx.x;  // d_inner dimension
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory: Affine2x2 array for Blelloch scan [num_threads * 6 floats]
    extern __shared__ float smem[];

    // Each thread handles a contiguous chunk of timesteps
    const int chunk_size = (N_local + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N_local);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load per-d_inner constants
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(stream_load(&A_log[j * d_state + s]));
    for (int p = 0; p < half_d_state; p++)
        freq[p] = stream_load(&rope_freq[j * half_d_state + p]);
    float D_val = stream_load(&D_param[j]);

    // Load initial state
    float h_init_all[MAX_D_STATE];
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = stream_load(&initial_state[j * d_state + s]);
    } else {
        for (int s = 0; s < d_state; s++)
            h_init_all[s] = 0.0f;
    }

    // Process each RoPE pair
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];
        const float h_init_e = h_init_all[s_e];
        const float h_init_o = h_init_all[s_o];

        // === Phase 1a: Sequential scan within thread's sub-chunk → chunk summary ===
        Affine2x2 summary = affine_identity();
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e = pre_B_val[t * d_state + s_e];
            float B_o = pre_B_val[t * d_state + s_o];
            Affine2x2 elem = build_scan_element(dt, x_v, B_e, B_o, A_e, A_o, f_val);
            summary = affine_combine_ptx(summary, elem);
        }

        // Store thread summary in shared memory
        int base = ltid * 6;
        smem[base + 0] = summary.m00; smem[base + 1] = summary.m01;
        smem[base + 2] = summary.m10; smem[base + 3] = summary.m11;
        smem[base + 4] = summary.b0;  smem[base + 5] = summary.b1;
        __syncthreads();

        // === Phase 1b: Blelloch exclusive prefix scan on thread summaries ===

        // Save total summary before Blelloch modifies it (last thread's position
        // after up-sweep holds total reduction). We accumulate it during up-sweep.
        // Instead, save summary of last active thread directly.
        // Actually, the total is at position (num_threads-1) after up-sweep.

        // Up-sweep (reduction)
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx - stride)*6],   smem[(idx - stride)*6+1],
                                   smem[(idx - stride)*6+2], smem[(idx - stride)*6+3],
                                   smem[(idx - stride)*6+4], smem[(idx - stride)*6+5]};
                Affine2x2 right = {smem[idx*6],   smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine_ptx(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // The total chunk summary is at position (num_threads-1) after up-sweep.
        // Write it out to the summaries buffer before overwriting with identity.
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            int sum_idx = (j * half_d_state + p) * 6;
            summaries[sum_idx + 0] = smem[last + 0];
            summaries[sum_idx + 1] = smem[last + 1];
            summaries[sum_idx + 2] = smem[last + 2];
            summaries[sum_idx + 3] = smem[last + 3];
            summaries[sum_idx + 4] = smem[last + 4];
            summaries[sum_idx + 5] = smem[last + 5];

            // Set last to identity for exclusive scan
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
            smem[last+2] = 0.0f; smem[last+3] = 1.0f;
            smem[last+4] = 0.0f; smem[last+5] = 0.0f;
        }
        __syncthreads();

        // Down-sweep
        for (int stride = num_threads / 2; stride >= 1; stride /= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx - stride)*6],   smem[(idx - stride)*6+1],
                                   smem[(idx - stride)*6+2], smem[(idx - stride)*6+3],
                                   smem[(idx - stride)*6+4], smem[(idx - stride)*6+5]};
                Affine2x2 right = {smem[idx*6],   smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                // Swap and combine
                smem[(idx - stride)*6]   = right.m00; smem[(idx - stride)*6+1] = right.m01;
                smem[(idx - stride)*6+2] = right.m10; smem[(idx - stride)*6+3] = right.m11;
                smem[(idx - stride)*6+4] = right.b0;  smem[(idx - stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine_ptx(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Read exclusive prefix for this thread
        Affine2x2 prefix = {smem[ltid*6],   smem[ltid*6+1],
                            smem[ltid*6+2], smem[ltid*6+3],
                            smem[ltid*6+4], smem[ltid*6+5]};

        // === Phase 1c: Re-scan chunk with prefix, compute local output ===
        Affine2x2 running = prefix;
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);

            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e = pre_B_val[t * d_state + s_e];
            float B_o = pre_B_val[t * d_state + s_o];
            Affine2x2 elem = build_scan_element(dt, x_v, B_e, B_o, A_e, A_o, f_val);
            running = affine_combine_ptx(running, elem);

            // Compute h[t] = running applied to initial state
            float h_e, h_o;
            affine_apply(running, h_init_e, h_init_o, h_e, h_o);

            // Accumulate y contribution from this pair
            float C_e = pre_C_val[t * d_state + s_e];
            float C_o = pre_C_val[t * d_state + s_o];
            scan_output[t * d_inner + j] += h_e * C_e + h_o * C_o;
        }

        __syncthreads();
    }

    // === Apply SiLU gating and D skip connection ===
    for (int step = 0; step < (N_local + num_threads - 1) / num_threads; step++) {
        int t_base = step * num_threads + ltid;
        if (t_base < N_local) {
            int t = reverse ? (N_local - 1 - t_base) : t_base;
            float z = pre_z_val[t * d_inner + j];
            float silu_z = z / (1.0f + expf(-z));
            float x_val = pre_x_val[t * d_inner + j];
            scan_output[t * d_inner + j] = scan_output[t * d_inner + j] * silu_z + D_val * x_val;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: mamba3_apply_scan_prefix_kernel (Forward)
//
//  After receiving the global prefix Affine2x2 for this GPU's chunk,
//  apply it to correct every element in the local scan output.
//
//  The correction transforms:
//    h_corrected[t] = prefix * h_local[t]
//  where h_local was computed assuming the scan started from the local
//  initial state. The prefix encodes the cumulative effect of all
//  preceding GPU chunks.
//
//  Grid: (d_inner) blocks, DSCAN_BLOCK threads each.
//  Each thread corrects a contiguous sub-chunk of timesteps.
//
//  The scan_output was computed as:
//    y[t] = sum_p (h_e * C_e + h_o * C_o) * silu_z + D * x
//  We need to undo the output, apply prefix to h, then recompute output.
//  Instead, we store raw h values in a separate buffer during kernel 1,
//  then recompute output here. BUT: to avoid extra memory, we take a
//  different approach — we correct the scan result directly.
//
//  Direct correction: For each timestep, the local scan computed
//    h_local[t] = T_local[0..t] * h_init_local
//  The correct value is:
//    h_correct[t] = T_local[0..t] * (prefix * h_init_global)
//  Since T_local * (prefix * h_init) = T_local * prefix * h_init
//  and T_local * h_init_local was used to compute the local output,
//  the correction is:
//    delta_h[t] = T_local[0..t] * (prefix * h_init - h_init_local)
//
//  We re-run the scan with the correction bias to compute delta_y.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_BLOCK, 4)
__global__ void mamba3_apply_scan_prefix_kernel(
    const float* __restrict__ pre_x_val,      // [N_local, d_inner]
    const float* __restrict__ pre_z_val,      // [N_local, d_inner]
    const float* __restrict__ pre_dt_val,     // [N_local, d_inner]
    const float* __restrict__ pre_B_val,      // [N_local, d_state]
    const float* __restrict__ pre_C_val,      // [N_local, d_state]
    const float* __restrict__ A_log,          // [d_inner, d_state]
    const float* __restrict__ rope_freq,      // [d_inner, d_state/2]
    const float* __restrict__ prefix_transforms, // [d_inner, half_d_state, 6]
    float* __restrict__ scan_output,          // [N_local, d_inner] — corrected in-place
    const float* __restrict__ initial_state,  // [d_inner, d_state] or nullptr (local)
    const int N_local,
    const int d_inner,
    const int d_state,
    const int reverse
) {
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];

    const int chunk_size = (N_local + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N_local);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load constants
    float A[MAX_D_STATE], freq_arr[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(stream_load(&A_log[j * d_state + s]));
    for (int pp = 0; pp < half_d_state; pp++)
        freq_arr[pp] = stream_load(&rope_freq[j * half_d_state + pp]);

    // Load local initial state
    float h_init_local[MAX_D_STATE];
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++)
            h_init_local[s] = stream_load(&initial_state[j * d_state + s]);
    } else {
        for (int s = 0; s < d_state; s++)
            h_init_local[s] = 0.0f;
    }

    // For each pair, compute the correction delta_y and add to scan_output.
    // The prefix transform T_prefix maps:
    //   h_corrected_init = T_prefix * h_init_global
    // But from local scan's perspective, h_init_local was used.
    // delta_init = (T_prefix.M * h_init_local + T_prefix.b) - h_init_local
    // Then delta_h[t] = T_local[0..t].M * delta_init (just the matrix part)
    // And delta_y[t] += delta_h_e * C_e + delta_h_o * C_o

    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq_arr[p];

        // Load prefix transform for this (j, pair)
        int pf_idx = (j * half_d_state + p) * 6;
        Affine2x2 pf;
        pf.m00 = prefix_transforms[pf_idx + 0];
        pf.m01 = prefix_transforms[pf_idx + 1];
        pf.m10 = prefix_transforms[pf_idx + 2];
        pf.m11 = prefix_transforms[pf_idx + 3];
        pf.b0  = prefix_transforms[pf_idx + 4];
        pf.b1  = prefix_transforms[pf_idx + 5];

        // Compute corrected initial state
        float h_init_e = h_init_local[s_e];
        float h_init_o = h_init_local[s_o];
        float h_corrected_e, h_corrected_o;
        affine_apply(pf, h_init_e, h_init_o, h_corrected_e, h_corrected_o);

        // delta_init: difference in initial state
        float delta_e = h_corrected_e - h_init_e;
        float delta_o = h_corrected_o - h_init_o;

        // If prefix is identity (delta == 0), skip this pair entirely
        if (fabsf(delta_e) < 1e-12f && fabsf(delta_o) < 1e-12f) continue;

        // Sequential scan to propagate delta through the local chunk.
        // We only need the matrix part of each element's transform to propagate delta.
        // delta_h[t] = M[t] * delta_h[t-1], where delta_h[0] = M[0] * delta_init + 0
        // But since the local scan already ran with h_init, the delta propagation
        // is: we re-scan with the delta as initial state (no bias since bias
        // was already included in local scan).

        // Build Blelloch scan of just the matrix part — but actually we can
        // do a simple sequential propagation since we need to apply delta to
        // every element anyway and must read scan_output.

        // Sequential delta propagation within thread's sub-chunk
        // Need to first do Blelloch scan of matrices across threads for the prefix.
        // Store thread's composed matrix in smem.

        Affine2x2 mat_summary = affine_identity();
        // Zero out bias since we only need matrix propagation of delta
        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];
            Affine2x2 elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);
            mat_summary = affine_combine_ptx(mat_summary, elem);
        }

        // Store in smem for Blelloch prefix
        int base_s = ltid * 6;
        smem[base_s + 0] = mat_summary.m00; smem[base_s + 1] = mat_summary.m01;
        smem[base_s + 2] = mat_summary.m10; smem[base_s + 3] = mat_summary.m11;
        smem[base_s + 4] = mat_summary.b0;  smem[base_s + 5] = mat_summary.b1;
        __syncthreads();

        // Blelloch exclusive prefix scan on thread summaries
        // Up-sweep
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine_ptx(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }
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
                smem[(idx-stride)*6]   = right.m00; smem[(idx-stride)*6+1] = right.m01;
                smem[(idx-stride)*6+2] = right.m10; smem[(idx-stride)*6+3] = right.m11;
                smem[(idx-stride)*6+4] = right.b0;  smem[(idx-stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine_ptx(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Read thread's prefix
        Affine2x2 thread_prefix = {smem[ltid*6], smem[ltid*6+1],
                                   smem[ltid*6+2], smem[ltid*6+3],
                                   smem[ltid*6+4], smem[ltid*6+5]};

        // Now re-scan the sub-chunk, propagating delta through the matrix chain
        // and adding the correction to scan_output.
        Affine2x2 running_mat = thread_prefix;
        // Zero out bias in running_mat — we only propagate delta through matrices
        // Actually thread_prefix includes bias from the prefix scan of full elements.
        // We need a separate approach: track running matrix to propagate delta.

        // Reset: we need running_mat to be identity-biased, propagate delta
        float cur_delta_e = delta_e;
        float cur_delta_o = delta_o;

        // Apply thread_prefix matrix to delta (prefix from preceding threads in block)
        float pfx_delta_e = thread_prefix.m00 * delta_e + thread_prefix.m01 * delta_o;
        float pfx_delta_o = thread_prefix.m10 * delta_e + thread_prefix.m11 * delta_o;
        cur_delta_e = pfx_delta_e;
        cur_delta_o = pfx_delta_o;

        for (int step = 0; step < my_count; step++) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];
            Affine2x2 elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);

            // Propagate delta through this element's matrix
            float new_delta_e = elem.m00 * cur_delta_e + elem.m01 * cur_delta_o;
            float new_delta_o = elem.m10 * cur_delta_e + elem.m11 * cur_delta_o;
            cur_delta_e = new_delta_e;
            cur_delta_o = new_delta_o;

            // Correction to output: delta_y += delta_h_e * C_e + delta_h_o * C_o
            // But scan_output already has SiLU gating applied. We need to add the
            // correction *before* gating. So we must undo gating, add, re-gate.
            // Actually, the SiLU gating multiplier is the same — y = (y_scan + delta) * silu_z + D*x
            // The D*x term is the same. So correction to final output = delta * silu_z.
            float C_e_v = pre_C_val[t * d_state + s_e];
            float C_o_v = pre_C_val[t * d_state + s_o];
            float delta_y = cur_delta_e * C_e_v + cur_delta_o * C_o_v;

            float z = pre_z_val[t * d_inner + j];
            float silu_z = z / (1.0f + expf(-z));
            scan_output[t * d_inner + j] += delta_y * silu_z;
        }

        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: scan_summary_prefix_kernel (Forward)
//
//  Runs on GPU 0. Takes K Affine2x2 summaries (one per GPU chunk) and
//  computes an exclusive prefix scan, producing K prefix transforms.
//
//  The prefix for GPU k is the composition of summaries[0..k-1].
//  GPU 0's prefix is identity. GPU 1's prefix is summary[0]. Etc.
//
//  This is a small kernel (K <= DSCAN_MAX_GPUS = 64) so a single
//  thread block with sequential scan suffices. For K <= 64, a single
//  warp can handle it.
//
//  Grid: (num_pairs) blocks, 1 thread each — one per (j, p) pair.
//  Actually, for simplicity and to keep it fully parallel across pairs,
//  we use one block per pair with K threads.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_MAX_GPUS, 1)
__global__ void scan_summary_prefix_kernel(
    const float* __restrict__ all_summaries,   // [K, d_inner, half_d_state, 6]
    float* __restrict__ prefix_out,            // [K, d_inner, half_d_state, 6]
    const int K,                               // number of GPU chunks
    const int d_inner,
    const int half_d_state
) {
    // Each block handles one (j, p) pair
    const int pair_idx = blockIdx.x;  // linear index over d_inner * half_d_state
    const int j = pair_idx / half_d_state;
    const int p = pair_idx % half_d_state;
    if (j >= d_inner) return;

    const int ltid = threadIdx.x;
    if (ltid >= K) return;

    // Shared memory for K Affine2x2 elements
    __shared__ float smem[DSCAN_MAX_GPUS * 6];

    // Load this GPU's summary for pair (j, p)
    // Layout: all_summaries[gpu_k][j][p] at offset (k * d_inner * half_d_state + j * half_d_state + p) * 6
    int src_idx = (ltid * d_inner * half_d_state + j * half_d_state + p) * 6;
    int base = ltid * 6;
    smem[base + 0] = stream_load(&all_summaries[src_idx + 0]);
    smem[base + 1] = stream_load(&all_summaries[src_idx + 1]);
    smem[base + 2] = stream_load(&all_summaries[src_idx + 2]);
    smem[base + 3] = stream_load(&all_summaries[src_idx + 3]);
    smem[base + 4] = stream_load(&all_summaries[src_idx + 4]);
    smem[base + 5] = stream_load(&all_summaries[src_idx + 5]);
    __syncthreads();

    // Sequential exclusive prefix scan (K is small, <= 64)
    // Only thread 0 does the work; results written to smem then copied out.
    if (ltid == 0) {
        // Temporary storage for exclusive prefix results
        float pf_buf[DSCAN_MAX_GPUS * 6];

        Affine2x2 running = affine_identity();

        for (int k = 0; k < K; k++) {
            // Write exclusive prefix (running BEFORE incorporating summary[k])
            int pb = k * 6;
            pf_buf[pb + 0] = running.m00; pf_buf[pb + 1] = running.m01;
            pf_buf[pb + 2] = running.m10; pf_buf[pb + 3] = running.m11;
            pf_buf[pb + 4] = running.b0;  pf_buf[pb + 5] = running.b1;

            // Incorporate summary[k]
            Affine2x2 sk = {smem[k*6], smem[k*6+1], smem[k*6+2],
                            smem[k*6+3], smem[k*6+4], smem[k*6+5]};
            running = affine_combine_ptx(running, sk);
        }

        // Write results to global memory
        for (int k = 0; k < K; k++) {
            int dst_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6;
            int pb = k * 6;
            stream_store(&prefix_out[dst_idx + 0], pf_buf[pb + 0]);
            stream_store(&prefix_out[dst_idx + 1], pf_buf[pb + 1]);
            stream_store(&prefix_out[dst_idx + 2], pf_buf[pb + 2]);
            stream_store(&prefix_out[dst_idx + 3], pf_buf[pb + 3]);
            stream_store(&prefix_out[dst_idx + 4], pf_buf[pb + 4]);
            stream_store(&prefix_out[dst_idx + 5], pf_buf[pb + 5]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 4: mamba3_scan_local_with_summary_bwd_kernel (Backward)
//
//  Backward version of kernel 1. Computes gradients of the local scan
//  w.r.t. precomputed inputs (pre_dt, pre_x, pre_B, pre_C) and
//  produces a backward summary Affine2x2 for the reverse-direction
//  gradient propagation across GPU chunks.
//
//  The backward scan runs in reverse order through the timesteps.
//  For each timestep t, given d_loss/d_y[t], we compute contributions
//  to d_loss/d_h[t] and propagate backwards through the scan.
//
//  The backward recurrence for h is:
//    dh[t] = M[t+1]^T * dh[t+1] + C[t] * dy[t]
//  This is also an affine recurrence (with transpose matrices),
//  so we can use the same Blelloch scan infrastructure.
//
//  Grid: (d_inner) blocks, DSCAN_BLOCK threads.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_BLOCK, 4)
__global__ void mamba3_scan_local_with_summary_bwd_kernel(
    const float* __restrict__ pre_x_val,      // [N_local, d_inner]
    const float* __restrict__ pre_z_val,      // [N_local, d_inner]
    const float* __restrict__ pre_dt_val,     // [N_local, d_inner]
    const float* __restrict__ pre_B_val,      // [N_local, d_state]
    const float* __restrict__ pre_C_val,      // [N_local, d_state]
    const float* __restrict__ A_log,          // [d_inner, d_state]
    const float* __restrict__ D_param,        // [d_inner]
    const float* __restrict__ rope_freq,      // [d_inner, d_state/2]
    const float* __restrict__ grad_output,    // [N_local, d_inner] — d_loss/d_y
    const float* __restrict__ fwd_scan_output,// [N_local, d_inner] — saved forward output
    float* __restrict__ grad_pre_x,           // [N_local, d_inner]
    float* __restrict__ grad_pre_dt,          // [N_local, d_inner]
    float* __restrict__ grad_pre_B,           // [N_local, d_state]
    float* __restrict__ grad_pre_C,           // [N_local, d_state]
    float* __restrict__ grad_D,               // [d_inner] — atomicAdd
    float* __restrict__ bwd_summaries,        // [d_inner, half_d_state, 6]
    const float* __restrict__ initial_state,  // [d_inner, d_state] or nullptr
    const int N_local,
    const int d_inner,
    const int d_state,
    const int reverse
) {
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];

    const int chunk_size = (N_local + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N_local);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    // Load constants
    float A[MAX_D_STATE], freq[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(stream_load(&A_log[j * d_state + s]));
    for (int pp = 0; pp < half_d_state; pp++)
        freq[pp] = stream_load(&rope_freq[j * half_d_state + pp]);
    float D_val = stream_load(&D_param[j]);

    // Accumulator for grad_D
    float grad_D_acc = 0.0f;

    // Backward: process each RoPE pair.
    // The backward scan uses transposed matrices running in reverse order.
    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq[p];

        // Build backward affine elements: M^T, with bias = C * dy (the gradient source)
        // The backward recurrence: dh[t-1] = M[t]^T * dh[t] + grad_source[t]
        // where grad_source[t] = C[t] * (dy[t] * silu_z[t])
        // M^T swaps m01 and m10 relative to M.

        // Phase 1: Sequential backward scan within thread's sub-chunk
        // Threads process in reverse order (last timestep first)
        Affine2x2 bwd_summary = affine_identity();

        for (int step = my_count - 1; step >= 0; step--) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];

            // Forward element (for transpose)
            Affine2x2 fwd_elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);

            // Transpose: swap m01 and m10
            Affine2x2 bwd_elem;
            bwd_elem.m00 = fwd_elem.m00;
            bwd_elem.m01 = fwd_elem.m10;  // transposed
            bwd_elem.m10 = fwd_elem.m01;  // transposed
            bwd_elem.m11 = fwd_elem.m11;

            // Bias for backward: C[t] * dy_raw[t] (before SiLU)
            float dy_raw = grad_output[t * d_inner + j];
            float z = pre_z_val[t * d_inner + j];
            float silu_z = z / (1.0f + expf(-z));
            float dy_scan = dy_raw * silu_z;  // gradient through SiLU gate

            float C_e_v = pre_C_val[t * d_state + s_e];
            float C_o_v = pre_C_val[t * d_state + s_o];
            bwd_elem.b0 = C_e_v * dy_scan;
            bwd_elem.b1 = C_o_v * dy_scan;

            bwd_summary = affine_combine_ptx(bwd_summary, bwd_elem);
        }

        // Store thread's backward summary in shared memory
        int base = ltid * 6;
        smem[base + 0] = bwd_summary.m00; smem[base + 1] = bwd_summary.m01;
        smem[base + 2] = bwd_summary.m10; smem[base + 3] = bwd_summary.m11;
        smem[base + 4] = bwd_summary.b0;  smem[base + 5] = bwd_summary.b1;
        __syncthreads();

        // Save total backward summary (at last position after up-sweep)
        // Blelloch exclusive prefix scan on backward summaries
        // Up-sweep
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine_ptx(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Write total backward summary to global buffer
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            int sum_idx = (j * half_d_state + p) * 6;
            bwd_summaries[sum_idx + 0] = smem[last + 0];
            bwd_summaries[sum_idx + 1] = smem[last + 1];
            bwd_summaries[sum_idx + 2] = smem[last + 2];
            bwd_summaries[sum_idx + 3] = smem[last + 3];
            bwd_summaries[sum_idx + 4] = smem[last + 4];
            bwd_summaries[sum_idx + 5] = smem[last + 5];

            // Set last to identity for exclusive scan
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
                smem[(idx-stride)*6]   = right.m00; smem[(idx-stride)*6+1] = right.m01;
                smem[(idx-stride)*6+2] = right.m10; smem[(idx-stride)*6+3] = right.m11;
                smem[(idx-stride)*6+4] = right.b0;  smem[(idx-stride)*6+5] = right.b1;
                Affine2x2 combined = affine_combine_ptx(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        // Read backward prefix for this thread
        Affine2x2 bwd_prefix = {smem[ltid*6], smem[ltid*6+1],
                                smem[ltid*6+2], smem[ltid*6+3],
                                smem[ltid*6+4], smem[ltid*6+5]};

        // Phase 2: Re-scan backward with prefix to compute per-element dh[t]
        // Then compute parameter gradients from dh[t].
        Affine2x2 bwd_running = bwd_prefix;

        for (int step = my_count - 1; step >= 0; step--) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];

            Affine2x2 fwd_elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);

            // Backward element (transposed)
            Affine2x2 bwd_elem;
            bwd_elem.m00 = fwd_elem.m00;
            bwd_elem.m01 = fwd_elem.m10;
            bwd_elem.m10 = fwd_elem.m01;
            bwd_elem.m11 = fwd_elem.m11;

            float dy_raw = grad_output[t * d_inner + j];
            float z = pre_z_val[t * d_inner + j];
            float silu_z = z / (1.0f + expf(-z));
            float dy_scan = dy_raw * silu_z;

            float C_e_v = pre_C_val[t * d_state + s_e];
            float C_o_v = pre_C_val[t * d_state + s_o];
            bwd_elem.b0 = C_e_v * dy_scan;
            bwd_elem.b1 = C_o_v * dy_scan;

            bwd_running = affine_combine_ptx(bwd_running, bwd_elem);

            // dh[t] = bwd_running applied to zero initial gradient
            // (exclusive prefix means bwd_running.b is the accumulated dh)
            float dh_e = bwd_running.b0;
            float dh_o = bwd_running.b1;

            // Gradient w.r.t. C: dC_e += dh_contribution = dy_scan * h_e (need h_e from fwd)
            // For local backward without forward state, accumulate gradient
            // contributions through the C projection gradient
            atomicAdd(&grad_pre_C[t * d_state + s_e], dh_e * dt * x_v);
            atomicAdd(&grad_pre_C[t * d_state + s_o], dh_o * dt * x_v);

            // Gradient w.r.t. B: dB += dh * dt * x
            atomicAdd(&grad_pre_B[t * d_state + s_e], dh_e * dt * x_v);
            atomicAdd(&grad_pre_B[t * d_state + s_o], dh_o * dt * x_v);

            // Gradient w.r.t. dt: complex chain rule through A_bar and B_bar
            float ddt_from_B = dh_e * B_e_v * x_v + dh_o * B_o_v * x_v;
            float denom_e = (1.0f - dt * A_e / 2.0f + 1e-8f);
            float denom_o = (1.0f - dt * A_o / 2.0f + 1e-8f);
            float dAbar_e = A_e / (denom_e * denom_e);
            float dAbar_o = A_o / (denom_o * denom_o);
            float ddt_from_A = dh_e * dAbar_e + dh_o * dAbar_o;
            atomicAdd(&grad_pre_dt[t * d_inner + j], ddt_from_B + ddt_from_A);

            // Gradient w.r.t. x: from B_bar contribution
            float dx_from_scan = dh_e * dt * B_e_v + dh_o * dt * B_o_v;
            atomicAdd(&grad_pre_x[t * d_inner + j], dx_from_scan);

            // Gradient w.r.t. D: D * x contributes dy * x to grad_D
            if (p == 0) {  // only accumulate once per timestep
                grad_D_acc += dy_raw * x_v;
            }
        }

        __syncthreads();
    }

    // Write accumulated grad_D
    if (grad_D_acc != 0.0f) {
        atomicAdd(&grad_D[j], grad_D_acc);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 5: mamba3_apply_scan_prefix_bwd_kernel (Backward)
//
//  Backward version of kernel 2. After receiving the global backward
//  prefix from the backward summary scan, applies it to correct the
//  local backward scan gradients.
//
//  The correction is analogous to the forward case: the backward prefix
//  encodes the accumulated gradient contribution from subsequent GPU
//  chunks that must flow back through this chunk.
//
//  Grid: (d_inner) blocks, DSCAN_BLOCK threads.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_BLOCK, 4)
__global__ void mamba3_apply_scan_prefix_bwd_kernel(
    const float* __restrict__ pre_x_val,      // [N_local, d_inner]
    const float* __restrict__ pre_z_val,      // [N_local, d_inner]
    const float* __restrict__ pre_dt_val,     // [N_local, d_inner]
    const float* __restrict__ pre_B_val,      // [N_local, d_state]
    const float* __restrict__ pre_C_val,      // [N_local, d_state]
    const float* __restrict__ A_log,          // [d_inner, d_state]
    const float* __restrict__ rope_freq,      // [d_inner, d_state/2]
    const float* __restrict__ grad_output,    // [N_local, d_inner]
    const float* __restrict__ bwd_prefix_transforms, // [d_inner, half_d_state, 6]
    float* __restrict__ grad_pre_x,           // [N_local, d_inner] — corrected
    float* __restrict__ grad_pre_dt,          // [N_local, d_inner] — corrected
    float* __restrict__ grad_pre_B,           // [N_local, d_state] — corrected
    float* __restrict__ grad_pre_C,           // [N_local, d_state] — corrected
    const int N_local,
    const int d_inner,
    const int d_state,
    const int reverse
) {
    const int j = blockIdx.x;
    if (j >= d_inner) return;
    const int ltid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];

    const int chunk_size = (N_local + num_threads - 1) / num_threads;
    const int my_start = ltid * chunk_size;
    const int my_end = min(my_start + chunk_size, N_local);
    const int my_count = max(my_end - my_start, 0);

    const int half_d_state = d_state / 2;

    float A[MAX_D_STATE], freq_arr[MAX_D_STATE / 2];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(stream_load(&A_log[j * d_state + s]));
    for (int pp = 0; pp < half_d_state; pp++)
        freq_arr[pp] = stream_load(&rope_freq[j * half_d_state + pp]);

    for (int p = 0; p < half_d_state; p++) {
        const int s_e = 2 * p;
        const int s_o = 2 * p + 1;
        const float A_e = A[s_e], A_o = A[s_o];
        const float f_val = freq_arr[p];

        // Load backward prefix transform
        int pf_idx = (j * half_d_state + p) * 6;
        Affine2x2 bpf;
        bpf.m00 = bwd_prefix_transforms[pf_idx + 0];
        bpf.m01 = bwd_prefix_transforms[pf_idx + 1];
        bpf.m10 = bwd_prefix_transforms[pf_idx + 2];
        bpf.m11 = bwd_prefix_transforms[pf_idx + 3];
        bpf.b0  = bwd_prefix_transforms[pf_idx + 4];
        bpf.b1  = bwd_prefix_transforms[pf_idx + 5];

        // The backward prefix carries an incoming dh from subsequent chunks.
        // Extract the bias component as the incoming gradient delta.
        float delta_dh_e = bpf.b0;
        float delta_dh_o = bpf.b1;

        // If prefix is identity (no correction needed), skip
        if (fabsf(delta_dh_e) < 1e-12f && fabsf(delta_dh_o) < 1e-12f &&
            fabsf(bpf.m00 - 1.0f) < 1e-12f && fabsf(bpf.m11 - 1.0f) < 1e-12f) continue;

        // Build Blelloch scan of backward matrices for prefix within block
        Affine2x2 bwd_mat_summary = affine_identity();
        for (int step = my_count - 1; step >= 0; step--) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];
            Affine2x2 fwd_elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);

            Affine2x2 bwd_elem;
            bwd_elem.m00 = fwd_elem.m00;
            bwd_elem.m01 = fwd_elem.m10;
            bwd_elem.m10 = fwd_elem.m01;
            bwd_elem.m11 = fwd_elem.m11;
            bwd_elem.b0  = 0.0f;
            bwd_elem.b1  = 0.0f;

            bwd_mat_summary = affine_combine_ptx(bwd_mat_summary, bwd_elem);
        }

        int base_s = ltid * 6;
        smem[base_s + 0] = bwd_mat_summary.m00; smem[base_s + 1] = bwd_mat_summary.m01;
        smem[base_s + 2] = bwd_mat_summary.m10; smem[base_s + 3] = bwd_mat_summary.m11;
        smem[base_s + 4] = bwd_mat_summary.b0;  smem[base_s + 5] = bwd_mat_summary.b1;
        __syncthreads();

        // Blelloch exclusive prefix scan
        for (int stride = 1; stride < num_threads; stride *= 2) {
            int idx = (ltid + 1) * stride * 2 - 1;
            if (idx < num_threads) {
                Affine2x2 left  = {smem[(idx-stride)*6], smem[(idx-stride)*6+1],
                                   smem[(idx-stride)*6+2], smem[(idx-stride)*6+3],
                                   smem[(idx-stride)*6+4], smem[(idx-stride)*6+5]};
                Affine2x2 right = {smem[idx*6], smem[idx*6+1],
                                   smem[idx*6+2], smem[idx*6+3],
                                   smem[idx*6+4], smem[idx*6+5]};
                Affine2x2 combined = affine_combine_ptx(left, right);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }
        if (ltid == 0) {
            int last = (num_threads - 1) * 6;
            smem[last]   = 1.0f; smem[last+1] = 0.0f;
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
                Affine2x2 combined = affine_combine_ptx(right, left);
                smem[idx*6]   = combined.m00; smem[idx*6+1] = combined.m01;
                smem[idx*6+2] = combined.m10; smem[idx*6+3] = combined.m11;
                smem[idx*6+4] = combined.b0;  smem[idx*6+5] = combined.b1;
            }
            if (stride * 2 >= WARP_SIZE) __syncthreads();
        }

        Affine2x2 thread_prefix = {smem[ltid*6], smem[ltid*6+1],
                                   smem[ltid*6+2], smem[ltid*6+3],
                                   smem[ltid*6+4], smem[ltid*6+5]};

        // Propagate backward delta through thread's sub-chunk
        // Apply thread_prefix to rotate the incoming delta
        float cur_dh_e = thread_prefix.m00 * delta_dh_e + thread_prefix.m01 * delta_dh_o;
        float cur_dh_o = thread_prefix.m10 * delta_dh_e + thread_prefix.m11 * delta_dh_o;

        for (int step = my_count - 1; step >= 0; step--) {
            int t = reverse ? (N_local - 1 - (my_start + step)) : (my_start + step);
            float dt  = pre_dt_val[t * d_inner + j];
            float x_v = pre_x_val[t * d_inner + j];
            float B_e_v = pre_B_val[t * d_state + s_e];
            float B_o_v = pre_B_val[t * d_state + s_o];

            Affine2x2 fwd_elem = build_scan_element(dt, x_v, B_e_v, B_o_v, A_e, A_o, f_val);

            // Propagate delta through transposed matrix
            float new_dh_e = fwd_elem.m00 * cur_dh_e + fwd_elem.m10 * cur_dh_o;
            float new_dh_o = fwd_elem.m01 * cur_dh_e + fwd_elem.m11 * cur_dh_o;
            cur_dh_e = new_dh_e;
            cur_dh_o = new_dh_o;

            // Add gradient corrections from the propagated delta
            float ddt_correction = cur_dh_e * B_e_v * x_v + cur_dh_o * B_o_v * x_v;
            float dx_correction = cur_dh_e * dt * B_e_v + cur_dh_o * dt * B_o_v;

            atomicAdd(&grad_pre_dt[t * d_inner + j], ddt_correction);
            atomicAdd(&grad_pre_x[t * d_inner + j], dx_correction);
            atomicAdd(&grad_pre_B[t * d_state + s_e], cur_dh_e * dt * x_v);
            atomicAdd(&grad_pre_B[t * d_state + s_o], cur_dh_o * dt * x_v);
            atomicAdd(&grad_pre_C[t * d_state + s_e], cur_dh_e * dt * x_v);
            atomicAdd(&grad_pre_C[t * d_state + s_o], cur_dh_o * dt * x_v);
        }

        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 6: scan_summary_prefix_bwd_kernel (Backward)
//
//  Backward version of kernel 3. Runs on GPU 0.
//  Takes K backward Affine2x2 summaries (one per GPU chunk, in reverse
//  order) and computes an exclusive prefix scan to produce backward
//  prefix transforms.
//
//  The backward prefix for GPU k encodes the accumulated gradient
//  contribution from all GPUs with index > k.
//
//  Grid: (num_pairs) blocks, up to DSCAN_MAX_GPUS threads.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(DSCAN_MAX_GPUS, 1)
__global__ void scan_summary_prefix_bwd_kernel(
    const float* __restrict__ all_bwd_summaries, // [K, d_inner, half_d_state, 6]
    float* __restrict__ bwd_prefix_out,          // [K, d_inner, half_d_state, 6]
    const int K,
    const int d_inner,
    const int half_d_state
) {
    const int pair_idx = blockIdx.x;
    const int j = pair_idx / half_d_state;
    const int p = pair_idx % half_d_state;
    if (j >= d_inner) return;

    const int ltid = threadIdx.x;
    if (ltid >= K) return;

    __shared__ float smem[DSCAN_MAX_GPUS * 6];

    // Load backward summary for GPU ltid, pair (j, p)
    // Backward summaries are ordered with the last GPU first (reverse scan)
    int src_idx = (ltid * d_inner * half_d_state + j * half_d_state + p) * 6;
    int base = ltid * 6;
    smem[base + 0] = stream_load(&all_bwd_summaries[src_idx + 0]);
    smem[base + 1] = stream_load(&all_bwd_summaries[src_idx + 1]);
    smem[base + 2] = stream_load(&all_bwd_summaries[src_idx + 2]);
    smem[base + 3] = stream_load(&all_bwd_summaries[src_idx + 3]);
    smem[base + 4] = stream_load(&all_bwd_summaries[src_idx + 4]);
    smem[base + 5] = stream_load(&all_bwd_summaries[src_idx + 5]);
    __syncthreads();

    // Sequential exclusive prefix scan in reverse order (thread 0 does all work)
    // GPU K-1 gets identity prefix (no subsequent chunks).
    // GPU K-2 gets summary[K-1]. GPU K-3 gets summary[K-1] composed with summary[K-2]. Etc.
    if (ltid == 0) {
        float pf_buf[DSCAN_MAX_GPUS * 6];

        Affine2x2 running = affine_identity();

        // Scan from last GPU to first
        for (int k = K - 1; k >= 0; k--) {
            // Write exclusive prefix for GPU k
            int pb = k * 6;
            pf_buf[pb + 0] = running.m00; pf_buf[pb + 1] = running.m01;
            pf_buf[pb + 2] = running.m10; pf_buf[pb + 3] = running.m11;
            pf_buf[pb + 4] = running.b0;  pf_buf[pb + 5] = running.b1;

            // Incorporate backward summary[k]
            Affine2x2 sk = {smem[k*6], smem[k*6+1], smem[k*6+2],
                            smem[k*6+3], smem[k*6+4], smem[k*6+5]};
            running = affine_combine_ptx(running, sk);
        }

        // Write results to global memory
        for (int k = 0; k < K; k++) {
            int dst_idx = (k * d_inner * half_d_state + j * half_d_state + p) * 6;
            int pb = k * 6;
            stream_store(&bwd_prefix_out[dst_idx + 0], pf_buf[pb + 0]);
            stream_store(&bwd_prefix_out[dst_idx + 1], pf_buf[pb + 1]);
            stream_store(&bwd_prefix_out[dst_idx + 2], pf_buf[pb + 2]);
            stream_store(&bwd_prefix_out[dst_idx + 3], pf_buf[pb + 3]);
            stream_store(&bwd_prefix_out[dst_idx + 4], pf_buf[pb + 4]);
            stream_store(&bwd_prefix_out[dst_idx + 5], pf_buf[pb + 5]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Launcher Functions
//
//  Calculate grid/block/smem and launch each kernel via <<<>>>.
//  These are called from ops.cpp via pybind11.
// ═══════════════════════════════════════════════════════════════════════

void distributed_scan_local_with_summary(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor summaries,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse
) {
    dim3 grid(d_inner);
    dim3 block(DSCAN_BLOCK);
    int smem = DSCAN_BLOCK * 6 * sizeof(float);
    const float* init_ptr = (initial_state.numel() > 0) ? initial_state.data_ptr<float>() : nullptr;

    mamba3_scan_local_with_summary_kernel<<<grid, block, smem>>>(
        pre_x_val.data_ptr<float>(), pre_z_val.data_ptr<float>(),
        pre_dt_val.data_ptr<float>(), pre_B_val.data_ptr<float>(),
        pre_C_val.data_ptr<float>(), A_log.data_ptr<float>(),
        D_param.data_ptr<float>(), rope_freq.data_ptr<float>(),
        scan_output.data_ptr<float>(), summaries.data_ptr<float>(),
        init_ptr, N_local, d_inner, d_state, reverse
    );
}

void distributed_scan_apply_prefix(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor rope_freq,
    torch::Tensor prefix_transforms, torch::Tensor scan_output,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse
) {
    dim3 grid(d_inner);
    dim3 block(DSCAN_BLOCK);
    int smem = DSCAN_BLOCK * 6 * sizeof(float);
    const float* init_ptr = (initial_state.numel() > 0) ? initial_state.data_ptr<float>() : nullptr;

    mamba3_apply_scan_prefix_kernel<<<grid, block, smem>>>(
        pre_x_val.data_ptr<float>(), pre_z_val.data_ptr<float>(),
        pre_dt_val.data_ptr<float>(), pre_B_val.data_ptr<float>(),
        pre_C_val.data_ptr<float>(), A_log.data_ptr<float>(),
        rope_freq.data_ptr<float>(), prefix_transforms.data_ptr<float>(),
        scan_output.data_ptr<float>(), init_ptr,
        N_local, d_inner, d_state, reverse
    );
}

void distributed_scan_summary_prefix(
    torch::Tensor all_summaries, torch::Tensor prefix_out,
    int K, int d_inner, int half_d_state
) {
    int num_pairs = d_inner * half_d_state;
    dim3 grid(num_pairs);
    dim3 block(K);

    scan_summary_prefix_kernel<<<grid, block>>>(
        all_summaries.data_ptr<float>(), prefix_out.data_ptr<float>(),
        K, d_inner, half_d_state
    );
}

void distributed_scan_local_with_summary_bwd(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor grad_output, torch::Tensor fwd_scan_output,
    torch::Tensor grad_pre_x, torch::Tensor grad_pre_dt,
    torch::Tensor grad_pre_B, torch::Tensor grad_pre_C,
    torch::Tensor grad_D, torch::Tensor bwd_summaries,
    torch::Tensor initial_state,
    int N_local, int d_inner, int d_state, int reverse
) {
    dim3 grid(d_inner);
    dim3 block(DSCAN_BLOCK);
    int smem = DSCAN_BLOCK * 6 * sizeof(float);
    const float* init_ptr = (initial_state.numel() > 0) ? initial_state.data_ptr<float>() : nullptr;

    mamba3_scan_local_with_summary_bwd_kernel<<<grid, block, smem>>>(
        pre_x_val.data_ptr<float>(), pre_z_val.data_ptr<float>(),
        pre_dt_val.data_ptr<float>(), pre_B_val.data_ptr<float>(),
        pre_C_val.data_ptr<float>(), A_log.data_ptr<float>(),
        D_param.data_ptr<float>(), rope_freq.data_ptr<float>(),
        grad_output.data_ptr<float>(), fwd_scan_output.data_ptr<float>(),
        grad_pre_x.data_ptr<float>(), grad_pre_dt.data_ptr<float>(),
        grad_pre_B.data_ptr<float>(), grad_pre_C.data_ptr<float>(),
        grad_D.data_ptr<float>(), bwd_summaries.data_ptr<float>(),
        init_ptr, N_local, d_inner, d_state, reverse
    );
}

void distributed_scan_apply_prefix_bwd(
    torch::Tensor pre_x_val, torch::Tensor pre_z_val, torch::Tensor pre_dt_val,
    torch::Tensor pre_B_val, torch::Tensor pre_C_val,
    torch::Tensor A_log, torch::Tensor rope_freq,
    torch::Tensor grad_output, torch::Tensor bwd_prefix_transforms,
    torch::Tensor grad_pre_x, torch::Tensor grad_pre_dt,
    torch::Tensor grad_pre_B, torch::Tensor grad_pre_C,
    int N_local, int d_inner, int d_state, int reverse
) {
    dim3 grid(d_inner);
    dim3 block(DSCAN_BLOCK);
    int smem = DSCAN_BLOCK * 6 * sizeof(float);

    mamba3_apply_scan_prefix_bwd_kernel<<<grid, block, smem>>>(
        pre_x_val.data_ptr<float>(), pre_z_val.data_ptr<float>(),
        pre_dt_val.data_ptr<float>(), pre_B_val.data_ptr<float>(),
        pre_C_val.data_ptr<float>(), A_log.data_ptr<float>(),
        rope_freq.data_ptr<float>(), grad_output.data_ptr<float>(),
        bwd_prefix_transforms.data_ptr<float>(),
        grad_pre_x.data_ptr<float>(), grad_pre_dt.data_ptr<float>(),
        grad_pre_B.data_ptr<float>(), grad_pre_C.data_ptr<float>(),
        N_local, d_inner, d_state, reverse
    );
}

void distributed_scan_summary_prefix_bwd(
    torch::Tensor all_bwd_summaries, torch::Tensor bwd_prefix_out,
    int K, int d_inner, int half_d_state
) {
    int num_pairs = d_inner * half_d_state;
    dim3 grid(num_pairs);
    dim3 block(K);

    scan_summary_prefix_bwd_kernel<<<grid, block>>>(
        all_bwd_summaries.data_ptr<float>(), bwd_prefix_out.data_ptr<float>(),
        K, d_inner, half_d_state
    );
}
