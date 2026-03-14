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
 * Plus a forward-save scan kernel that stores intermediate states.
 *
 * All meta-net weights and gradients are FP32.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

constexpr int SG2B_BLOCK = 256;
constexpr int MAX_D_STATE = 32;


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

__global__ void mamba3_scan_fwd_save_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,    // [2*d_inner, d_model]
    const float* __restrict__ dt_proj_W,    // [d_inner, d_inner]
    const float* __restrict__ dt_proj_b,    // [d_inner]
    const float* __restrict__ B_proj_W,     // [d_state, d_inner]
    const float* __restrict__ C_proj_W,     // [d_state, d_inner]
    const float* __restrict__ A_log,        // [d_inner, d_state]
    const float* __restrict__ D_param,      // [d_inner]
    const float* __restrict__ rope_freq,    // [d_inner, d_state]
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
    const int reverse
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
        saved_x_branch[i * d_inner + tid] = x_val;
        saved_z[i * d_inner + tid] = z_val;

        // Write x_branch to shared memory for cross-thread access
        s_x_branch[tid] = x_val;
        __syncthreads();

        // FULL dt projection
        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++) {
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        }
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        saved_dt[i * d_inner + tid] = dt_val;

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
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }

            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // Save state after update
        for (int s = 0; s < d_state; s++)
            saved_states[(i * d_inner + tid) * d_state + s] = h[s];

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
    float* my_saved_states = saved_states_packed + start * d_inner * d_state;
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
        my_saved_z[i * d_inner + tid] = z_val;

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));
        my_saved_dt[i * d_inner + tid] = dt_val;

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
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            if (s % 2 == 0) {
                h_rot = h_snap[s] * cos_p - h_snap[s + 1] * sin_p;
            } else {
                h_rot = h_snap[s] * cos_p + h_snap[s - 1] * sin_p;
            }
            h[s] = A_bar * h_rot + B_bar * x_val;
        }

        // Save state after update
        for (int s = 0; s < d_state; s++)
            my_saved_states[(i * d_inner + tid) * d_state + s] = h[s];

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
    float* __restrict__ d_B_proj_W,           // [d_state, d_inner]
    float* __restrict__ d_C_proj_W,           // [d_state, d_inner]
    float* __restrict__ d_A_log,              // [d_inner, d_state]
    float* __restrict__ d_D_param,            // [d_inner]
    float* __restrict__ d_rope_freq,          // [d_inner, d_state/2]
    float* __restrict__ d_x_sorted,           // [N, d_model]
    const float* __restrict__ initial_state,  // [d_inner, d_state] or nullptr
    // Dims
    const int N, const int d_model, const int d_inner, const int d_state,
    const int reverse
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    // Shared memory for cross-thread access in backward
    extern __shared__ float smem[];
    float* s_x_branch = smem;              // [d_inner]
    float* s_d_dt_raw = smem + d_inner;    // [d_inner] — for full dt backward

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
    float d_B_proj_acc[MAX_D_STATE];
    float d_C_proj_acc[MAX_D_STATE];
    float d_dt_proj_b_acc = 0.0f;
    for (int s = 0; s < d_state; s++) {
        d_A_log_acc[s] = 0.0f;
        d_B_proj_acc[s] = 0.0f;
        d_C_proj_acc[s] = 0.0f;
    }
    for (int p = 0; p < half_d_state; p++) {
        d_freq_acc[p] = 0.0f;
    }
    // Full dt_proj_W gradient (not just diagonal)
    float d_dt_proj_W_row[32]; // max d_inner = 32
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    // Gradient of state: dh[s] propagated backward through time
    float dh[MAX_D_STATE];
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    // Iterate in reverse order of forward computation
    for (int step = N - 1; step >= 0; step--) {
        int i = reverse ? (N - 1 - step) : step;

        float d_out = d_scan_output[i * d_inner + tid];
        float x_val = saved_x_branch[i * d_inner + tid];
        float z_val = saved_z[i * d_inner + tid];
        float dt_val = saved_dt[i * d_inner + tid];

        // Load x_branch into shared memory for full projection backward
        s_x_branch[tid] = x_val;
        __syncthreads();

        // Forward: y_final = y_val * silu_z + D_val * x_val
        float sig_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sig_z;

        // Recompute y_val from saved state (using full C projection)
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            y_val += saved_states[(i * d_inner + tid) * d_state + s] * C_val;
        }

        // Backward through gated output
        float d_y_val = d_out * silu_z;
        float d_silu_z = d_out * y_val;
        float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
        float d_x_from_D = d_out * D_val;
        d_D_acc += d_out * x_val;

        // Backward through y = sum_s(h[s] * C[s]) with full C projection
        float d_x_from_C = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float h_s = saved_states[(i * d_inner + tid) * d_state + s];
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            }

            // dh[s] += d_y * C[s]
            dh[s] += d_y_val * C_val;
            // d_C_val = d_y * h[s] — accumulate for all j
            float d_C_val = d_y_val * h_s;
            for (int j = 0; j < d_inner; j++) {
                atomicAdd(&d_C_proj_W[s * d_inner + j], d_C_val * s_x_branch[j]);
            }
            // Accumulate d_x_from_C: gradient of y w.r.t. x_branch[tid] via C projection
            d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
        }

        // Get h_prev (state before this step's update)
        float h_prev[MAX_D_STATE];
        if (step > 0) {
            int i_prev = reverse ? (N - step) : (step - 1);
            for (int s = 0; s < d_state; s++)
                h_prev[s] = saved_states[(i_prev * d_inner + tid) * d_state + s];
        } else {
            // Use initial_state if provided, else zero
            for (int s = 0; s < d_state; s++)
                h_prev[s] = (initial_state != nullptr)
                    ? initial_state[tid * d_state + s] : 0.0f;
        }

        // Backward through state update: h[s] = A_bar * h_rot + B_bar * x_val
        // Fix: snapshot dh before the loop to avoid read-after-write
        float dh_snap[MAX_D_STATE];
        for (int s = 0; s < d_state; s++) dh_snap[s] = dh[s];
        for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

        float d_dt_val = 0.0f;
        float d_x_from_scan = 0.0f;

        for (int s = 0; s < d_state; s++) {
            float half_dtA = dt_val * A[s] / 2.0f;
            float denom_val = 1.0f - half_dtA + 1e-8f;
            float A_bar = (1.0f + half_dtA) / denom_val;

            // Full B projection (recompute)
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            float B_bar = dt_val * B_val;

            // Paired RoPE
            int pair_idx = s / 2;
            float cos_p, sin_p;
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            int partner;
            float sign;  // sign of sin term in h_rot formula
            if (s % 2 == 0) {
                partner = s + 1;
                sign = -1.0f;  // h_rot = h[s]*cos - h[s+1]*sin
                h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
            } else {
                partner = s - 1;
                sign = 1.0f;   // h_rot = h[s]*cos + h[s-1]*sin
                h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
            }

            // Use dh_snap instead of dh for reading
            float d_h_s = dh_snap[s];

            float d_A_bar = d_h_s * h_rot;
            float d_h_rot = d_h_s * A_bar;
            // Forward: h[s] = A_bar * h_rot + B_bar * x_val
            float d_B_bar = d_h_s * x_val;
            d_x_from_scan += d_h_s * B_bar;  // gradient through x_val

            // d_B_bar -> d_dt, d_B_val
            d_dt_val += d_B_bar * B_val;
            float d_B_val = d_B_bar * dt_val;
            // Full B projection backward
            for (int j = 0; j < d_inner; j++) {
                atomicAdd(&d_B_proj_W[s * d_inner + j], d_B_val * s_x_branch[j]);
            }
            // Gradient of B_val w.r.t. x_branch[tid]: B_proj_W[s, tid]
            d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];

            // d_A_bar -> d_dt, d_A_log
            float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
            d_dt_val += d_half_dtA * A[s] / 2.0f;
            float d_A_s = d_half_dtA * dt_val / 2.0f;
            d_A_log_acc[s] += d_A_s * A[s];

            // d_h_rot -> d_h_prev[s], d_h_prev[partner] (paired RoPE)
            float d_h_prev_s = d_h_rot * cos_p;
            float d_h_prev_partner = d_h_rot * sign * sin_p;

            // d_h_rot -> d_cos, d_sin -> d_dt, d_freq
            float d_cos = d_h_rot * h_prev[s];
            float d_sin = d_h_rot * sign * h_prev[partner];
            d_dt_val += (-sin_p * freq[pair_idx]) * d_cos + (cos_p * freq[pair_idx]) * d_sin;
            d_freq_acc[pair_idx] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;

            // Propagate dh to previous step (accumulate into fresh dh)
            dh[s] += d_h_prev_s;
            dh[partner] += d_h_prev_partner;
        }

        // Backward through dt: dt_val = softplus(dt_raw)
        // Recompute dt_raw using full projection
        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++) {
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        }
        float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
        float d_dt_raw = d_dt_val * sig_dt;

        d_dt_proj_b_acc += d_dt_raw;
        // Full dt_proj_W backward
        for (int j = 0; j < d_inner; j++) {
            d_dt_proj_W_row[j] += d_dt_raw * s_x_branch[j];
        }

        // Full dt backward using shared memory for cross-thread d_dt_raw
        s_d_dt_raw[tid] = d_dt_raw;
        __syncthreads();
        float d_x_from_dt = 0.0f;
        for (int t = 0; t < d_inner; t++) {
            d_x_from_dt += s_d_dt_raw[t] * dt_proj_W[t * d_inner + tid];
        }

        // Total d_x_val for this step
        float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;

        // Backward through input projection to d_x_sorted and d_in_proj_W
        for (int d = 0; d < d_model; d++) {
            float inp = x_sorted[i * d_model + d];
            atomicAdd(&d_in_proj_W[tid * d_model + d], d_x_val * inp);
            atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_z_val * inp);
            atomicAdd(&d_x_sorted[i * d_model + d],
                      d_x_val * in_proj_W[tid * d_model + d] +
                      d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
        }

        __syncthreads(); // ensure all threads done before next step
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
    float* __restrict__ d_B_proj_W,
    float* __restrict__ d_C_proj_W,
    float* __restrict__ d_A_log,
    float* __restrict__ d_D_param,
    float* __restrict__ d_rope_freq,
    float* __restrict__ d_x_sorted_packed,           // [total_N, d_model]
    // Dims
    const int d_model, const int d_inner, const int d_state
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
    float* s_d_dt_raw = smem + d_inner;

    // Point to this param's packed data
    const float* my_d_scan = d_scan_output_packed + start * d_inner;
    const float* my_x_sorted = x_sorted_packed + start * d_model;
    const float* my_saved_states = saved_states_packed + start * d_inner * d_state;
    const float* my_saved_xb = saved_x_branch_packed + start * d_inner;
    const float* my_saved_z = saved_z_packed + start * d_inner;
    const float* my_saved_dt = saved_dt_packed + start * d_inner;
    float* my_d_x_sorted = d_x_sorted_packed + start * d_model;

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
    float d_dt_proj_W_row[32];
    for (int j = 0; j < d_inner; j++) d_dt_proj_W_row[j] = 0.0f;

    float dh[MAX_D_STATE];
    for (int s = 0; s < d_state; s++) dh[s] = 0.0f;

    // Initial state for h_prev at step 0
    const float* my_init = initial_states + param_idx * d_inner * d_state;

    for (int step = N - 1; step >= 0; step--) {
        int i = reverse ? (N - 1 - step) : step;

        float d_out = my_d_scan[i * d_inner + tid];
        float x_val = my_saved_xb[i * d_inner + tid];
        float z_val = my_saved_z[i * d_inner + tid];
        float dt_val = my_saved_dt[i * d_inner + tid];

        s_x_branch[tid] = x_val;
        __syncthreads();

        float sig_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sig_z;

        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            y_val += my_saved_states[(i * d_inner + tid) * d_state + s] * C_val;
        }

        float d_y_val = d_out * silu_z;
        float d_silu_z = d_out * y_val;
        float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
        float d_x_from_D = d_out * D_val;
        d_D_acc += d_out * x_val;

        float d_x_from_C = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float h_s = my_saved_states[(i * d_inner + tid) * d_state + s];
            float C_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
            dh[s] += d_y_val * C_val;
            float d_C_val = d_y_val * h_s;
            for (int j = 0; j < d_inner; j++)
                atomicAdd(&d_C_proj_W[s * d_inner + j], d_C_val * s_x_branch[j]);
            d_x_from_C += d_y_val * h_s * C_proj_W[s * d_inner + tid];
        }

        // Get h_prev
        float h_prev[MAX_D_STATE];
        if (step > 0) {
            int i_prev = reverse ? (N - step) : (step - 1);
            for (int s = 0; s < d_state; s++)
                h_prev[s] = my_saved_states[(i_prev * d_inner + tid) * d_state + s];
        } else {
            // Use initial state for this param
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
            __sincosf(dt_val * freq[pair_idx], &sin_p, &cos_p);
            float h_rot;
            int partner;
            float sign;
            if (s % 2 == 0) {
                partner = s + 1;
                sign = -1.0f;
                h_rot = h_prev[s] * cos_p - h_prev[partner] * sin_p;
            } else {
                partner = s - 1;
                sign = 1.0f;
                h_rot = h_prev[s] * cos_p + h_prev[partner] * sin_p;
            }

            float d_h_s = dh_snap[s];
            float d_A_bar = d_h_s * h_rot;
            float d_h_rot = d_h_s * A_bar;
            float d_B_bar = d_h_s * x_val;
            d_x_from_scan += d_h_s * B_bar;

            d_dt_val += d_B_bar * B_val;
            float d_B_val = d_B_bar * dt_val;
            for (int j = 0; j < d_inner; j++)
                atomicAdd(&d_B_proj_W[s * d_inner + j], d_B_val * s_x_branch[j]);
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
            atomicAdd(&d_in_proj_W[tid * d_model + d], d_x_val * inp);
            atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_z_val * inp);
            atomicAdd(&my_d_x_sorted[i * d_model + d],
                      d_x_val * in_proj_W[tid * d_model + d] +
                      d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
        }

        __syncthreads();
    }

    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    for (int j = 0; j < d_inner; j++)
        atomicAdd(&d_dt_proj_W[tid * d_inner + j], d_dt_proj_W_row[j]);
    for (int s = 0; s < d_state; s++)
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
    for (int p = 0; p < half_d_state; p++)
        atomicAdd(&d_rope_freq[tid * half_d_state + p], d_freq_acc[p]);
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
__global__ void input_proj_backward_kernel(
    const float* __restrict__ d_x,           // [N, d_model]
    const scalar_t* __restrict__ grad,       // [N]
    const scalar_t* __restrict__ sharpness,  // [N]
    float* __restrict__ d_proj_W,            // [d_model, 2] — atomicAdd
    float* __restrict__ d_proj_b,            // [d_model] — atomicAdd
    const int N,
    const int d_model
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);

    for (int d = 0; d < d_model; d++) {
        float d_xd = d_x[idx * d_model + d];
        atomicAdd(&d_proj_W[d * 2 + 0], d_xd * g);
        atomicAdd(&d_proj_W[d * 2 + 1], d_xd * s);
        atomicAdd(&d_proj_b[d], d_xd);
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int total_dim = input_dim + gru_hidden;

    for (int gh = 0; gh < gru_hidden; gh++) {
        float d_h = d_h_new[idx * gru_hidden + gh];
        float z_val = z_gate[idx * gru_hidden + gh];
        float r_val = r_gate[idx * gru_hidden + gh];
        float ht_val = h_tilde[idx * gru_hidden + gh];
        float h_old_val = h_old[idx * gru_hidden + gh];

        // h_new = (1-z) * h_old + z * h_tilde
        float d_z = d_h * (ht_val - h_old_val);
        float d_h_tilde_val = d_h * z_val;
        // d_h_old from this gate: d_h * (1 - z)  [not needed if we don't backprop further into h_old]

        // d_h_tilde: tanh backward
        float d_tanh_input = d_h_tilde_val * (1.0f - ht_val * ht_val);

        // d_z: sigmoid backward
        float d_z_input = d_z * z_val * (1.0f - z_val);

        // Accumulate d_Wh, d_bh from d_tanh_input
        // Wh @ xrh + bh -> tanh -> h_tilde
        // xrh = cat(x, r * h_old)
        atomicAdd(&d_bh[gh], d_tanh_input);
        for (int j = 0; j < input_dim; j++) {
            float xj = gru_input[idx * input_dim + j];
            atomicAdd(&d_Wh[gh * total_dim + j], d_tanh_input * xj);
        }
        for (int j = 0; j < gru_hidden; j++) {
            float rh = r_val * h_old[idx * gru_hidden + j];
            // Only use r_val for the current gh dimension's r
            if (j == gh)
                rh = r_val * h_old_val;
            else
                rh = r_gate[idx * gru_hidden + j] * h_old[idx * gru_hidden + j];
            atomicAdd(&d_Wh[gh * total_dim + input_dim + j], d_tanh_input * rh);
        }

        // Accumulate d_Wz, d_bz from d_z_input
        atomicAdd(&d_bz[gh], d_z_input);
        for (int j = 0; j < total_dim; j++) {
            float xh_j;
            if (j < input_dim)
                xh_j = gru_input[idx * input_dim + j];
            else
                xh_j = h_old[idx * gru_hidden + (j - input_dim)];
            atomicAdd(&d_Wz[gh * total_dim + j], d_z_input * xh_j);
        }

        // d_r from Wh backward: per-dimension d_r[j] (not mixed)
        // Forward: xrh[input_dim+j] = r[j] * h_old[j]
        // d_xrh[input_dim+j] from this gh = d_tanh_input * Wh[gh, input_dim+j]
        // d_r[j] from this gh = d_tanh_input * Wh[gh, input_dim+j] * h_old[j]
        for (int j = 0; j < gru_hidden; j++) {
            float d_r_j = d_tanh_input * Wh[gh * total_dim + input_dim + j]
                        * h_old[idx * gru_hidden + j];
            float r_j = r_gate[idx * gru_hidden + j];
            float d_r_j_input = d_r_j * r_j * (1.0f - r_j);

            // Accumulate d_Wr[j, :] and d_br[j] from this gh's contribution
            atomicAdd(&d_br[j], d_r_j_input);
            for (int k = 0; k < total_dim; k++) {
                float xh_k;
                if (k < input_dim)
                    xh_k = gru_input[idx * input_dim + k];
                else
                    xh_k = h_old[idx * gru_hidden + (k - input_dim)];
                atomicAdd(&d_Wr[j * total_dim + k], d_r_j_input * xh_k);
            }
            // d_gru_input from Wr backward
            for (int k = 0; k < input_dim; k++) {
                atomicAdd(&d_gru_input[idx * input_dim + k],
                          d_r_j_input * Wr[j * total_dim + k]);
            }
        }

        // d_gru_input from Wz and Wh backward (Wr handled above per-dimension)
        for (int j = 0; j < input_dim; j++) {
            float d_input_j = d_z_input * Wz[gh * total_dim + j]
                            + d_tanh_input * Wh[gh * total_dim + j];
            atomicAdd(&d_gru_input[idx * input_dim + j], d_input_j);
        }
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float d_out = d_expert_out[idx];
    float g_val = grad_vals[idx];
    int half_d = d_model / 2;

    for (int h = 0; h < num_heads; h++) {
        float d_head_out = d_out / (float)num_heads;

        // First pass: compute expert outputs and softmax backward dot products
        float dot_a[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // max topk = 4
        float dot_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};

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

            // Backward through expert MLP
            atomicAdd(&d_expert_b2[ei], d_out_k);

            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                atomicAdd(&d_expert_W2[ei * expert_hidden + eh], d_out_k * z_val);
                float d_z = d_out_k * expert_W2[ei * expert_hidden + eh];
                float d_pre_relu = (z_val > 0.0f) ? d_z : 0.0f;
                atomicAdd(&d_expert_W1[ei * expert_hidden + eh], d_pre_relu * g_val);
                atomicAdd(&d_expert_b1[ei * expert_hidden + eh], d_pre_relu);
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

                atomicAdd(&d_prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d], d_score_a * q_a_d);
                atomicAdd(&d_prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d], d_score_b * q_b_d);

                float d_q_a_d = d_score_a * prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d];
                float d_q_b_d = d_score_b * prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d];

                for (int j = 0; j < peer_input_dim; j++) {
                    float pi_j = saved_peer_input[idx * peer_input_dim + j];
                    atomicAdd(&d_peer_query_Ws[(h * d_model + d) * peer_input_dim + j], d_q_a_d * pi_j);
                    atomicAdd(&d_peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j], d_q_b_d * pi_j);
                    atomicAdd(&d_peer_input[idx * peer_input_dim + j],
                              d_q_a_d * peer_query_Ws[(h * d_model + d) * peer_input_dim + j] +
                              d_q_b_d * peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j]);
                }
            }
        }
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

__global__ void out_proj_backward_kernel(
    const float* __restrict__ d_context,       // [N, d_model]
    const float* __restrict__ scan_out,        // [N, d_inner]
    const float* __restrict__ out_proj_W,      // [d_model, d_inner]
    float* __restrict__ d_out_proj_W,          // [d_model, d_inner] — atomicAdd
    float* __restrict__ d_scan_out,            // [N, d_inner]
    const int N, const int d_model, const int d_inner
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int j = 0; j < d_inner; j++) {
        float d_scan_j = 0.0f;
        float so_j = scan_out[idx * d_inner + j];
        for (int d = 0; d < d_model; d++) {
            float d_ctx = d_context[idx * d_model + d];
            d_scan_j += d_ctx * out_proj_W[d * d_inner + j];
            atomicAdd(&d_out_proj_W[d * d_inner + j], d_ctx * so_j);
        }
        d_scan_out[idx * d_inner + j] = d_scan_j;
    }
}


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
    torch::Tensor sort_indices        // [N] — sort indices (computed here)
) {
    const int N = grad.numel();
    if (N == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");

    auto dev = grad.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Step 1: Input projection + sort
    auto x_proj = torch::empty({N, d_model}, float_opts);
    auto sort_keys = torch::empty({N}, float_opts);
    auto sort_idx = torch::empty({N}, int_opts);

    {
        // Reuse the input_proj_sort_kernel from forward file
        // (it's defined in supergrok2_mamba_peer_kernels.cu)
        // For now, compute in PyTorch-compatible way
        auto g_f = grad.to(torch::kFloat32).reshape(-1);
        auto s_f = sharpness.to(torch::kFloat32).reshape(-1);
        auto inp = torch::stack({g_f, s_f}, 1);  // [N, 2]
        x_proj = torch::addmm(input_proj_b, inp, input_proj_W.t());  // [N, d_model]

        sort_keys = g_f.abs();
        auto sorted = sort_keys.argsort();
        sort_indices.copy_(sorted.to(torch::kInt32));
        x_sorted.copy_(x_proj.index_select(0, sorted));
    }

    // Shared memory for scan: x_branch
    int scan_smem = d_inner * sizeof(float);

    // Step 2: Forward scan with state saving
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
        nullptr,  // no initial_state for bilevel
        N, d_model, d_inner, d_state, 0
    );

    // Step 3: Backward scan with state saving (reversed input)
    // Need reversed x_sorted
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
        nullptr,  // no initial_state for bilevel
        N, d_model, d_inner, d_state, 0
    );
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
    int expert_hidden, int peer_input_dim, int num_experts
) {
    const int N = d_smart_grad.numel();
    if (N == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");

    auto dev = d_smart_grad.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    const int grid = (N + SG2B_BLOCK - 1) / SG2B_BLOCK;
    int num_active = topk * topk;

    // Step 1: d_expert_out = rescale * d_smart_grad
    auto d_expert_out = (d_smart_grad.reshape(-1) * rescale).contiguous();

    // Step 2: Expert + PEER backward
    auto d_peer_input = torch::zeros({N, peer_input_dim}, float_opts);
    expert_peer_backward_kernel<<<grid, SG2B_BLOCK>>>(
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
    auto d_gru_input = torch::zeros({N, gru_input_dim}, float_opts);
    gru_backward_kernel<<<grid, SG2B_BLOCK>>>(
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
    auto d_fwd_ctx = torch::zeros({N, d_model}, float_opts);
    auto d_bwd_ctx = torch::zeros({N, d_model}, float_opts);

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
    auto d_fwd_scan_out = torch::zeros({N, d_inner}, float_opts);
    auto d_bwd_scan_out = torch::zeros({N, d_inner}, float_opts);

    out_proj_backward_kernel<<<grid, SG2B_BLOCK>>>(
        d_fwd_sorted.data_ptr<float>(),
        fwd_scan_out.data_ptr<float>(),
        mamba_fwd_out_proj.data_ptr<float>(),
        d_mamba_fwd_out_proj.data_ptr<float>(),
        d_fwd_scan_out.data_ptr<float>(),
        N, d_model, d_inner
    );

    out_proj_backward_kernel<<<grid, SG2B_BLOCK>>>(
        d_bwd_sorted.data_ptr<float>(),
        bwd_scan_out.data_ptr<float>(),
        mamba_bwd_out_proj.data_ptr<float>(),
        d_mamba_bwd_out_proj.data_ptr<float>(),
        d_bwd_scan_out.data_ptr<float>(),
        N, d_model, d_inner
    );

    // Step 8: Mamba scan backward (both directions)
    int scan_smem = 2 * d_inner * sizeof(float); // s_x_branch + s_d_dt_raw
    auto d_x_sorted_fwd = torch::zeros({N, d_model}, float_opts);
    auto d_x_sorted_bwd = torch::zeros({N, d_model}, float_opts);

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
        d_mamba_fwd_B_proj.data_ptr<float>(),
        d_mamba_fwd_C_proj.data_ptr<float>(),
        d_mamba_fwd_A_log.data_ptr<float>(),
        d_mamba_fwd_D.data_ptr<float>(),
        d_mamba_fwd_rope.data_ptr<float>(),
        d_x_sorted_fwd.data_ptr<float>(),
        mamba_fwd_init_state.numel() > 0
            ? mamba_fwd_init_state.data_ptr<float>() : nullptr,
        N, d_model, d_inner, d_state, 0
    );

    // Backward scan used reversed x_sorted
    auto x_sorted_rev = x_sorted.flip(0).contiguous();
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
        d_mamba_bwd_B_proj.data_ptr<float>(),
        d_mamba_bwd_C_proj.data_ptr<float>(),
        d_mamba_bwd_A_log.data_ptr<float>(),
        d_mamba_bwd_D.data_ptr<float>(),
        d_mamba_bwd_rope.data_ptr<float>(),
        d_x_sorted_bwd.data_ptr<float>(),
        mamba_bwd_init_state.numel() > 0
            ? mamba_bwd_init_state.data_ptr<float>() : nullptr,
        N, d_model, d_inner, d_state, 0
    );

    // Combine d_x_sorted from both directions
    // bwd scan backward produces d for reversed input, flip back
    auto d_x_sorted = d_x_sorted_fwd + d_x_sorted_bwd.flip(0);

    // Unsort d_x_sorted back to original order for input_proj backward
    auto unsort_idx = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kLong));
    unsort_idx.scatter_(0, sort_idx_long,
        torch::arange(N, torch::TensorOptions().device(dev).dtype(torch::kLong)));
    auto d_x_unsorted = d_x_sorted.index_select(0, unsort_idx);

    // Step 9: Input projection backward
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "input_proj_backward", ([&] {
        input_proj_backward_kernel<scalar_t><<<grid, SG2B_BLOCK>>>(
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
    torch::Tensor sort_indices_packed        // [total_N] int
) {
    const int num_params = grads.size();
    if (num_params == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");

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

    // Zero initial states for bilevel
    auto initial_states = torch::zeros({num_params, d_inner, d_state}, float_opts);
    auto final_fwd = torch::empty({num_params, d_inner, d_state}, float_opts);
    auto final_bwd = torch::empty({num_params, d_inner, d_state}, float_opts);

    auto rev_fwd = torch::zeros({num_params}, int_opts);
    auto rev_bwd = torch::ones({num_params}, int_opts);

    int scan_smem = d_inner * sizeof(float);

    // Step 2: Forward scan with saving
    mamba3_scan_fwd_save_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        x_sorted_packed.data_ptr<float>(),
        fwd_scan_out_packed.data_ptr<float>(),
        initial_states.data_ptr<float>(),
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
        d_model, d_inner, d_state
    );

    // Step 3: Build reversed x_sorted for backward scan direction
    // Each param's portion of x_sorted_packed needs to be reversed independently
    auto x_sorted_rev_packed = torch::empty_like(x_sorted_packed);
    for (int p = 0; p < num_params; p++) {
        int N = offsets_cpu[p + 1] - offsets_cpu[p];
        if (N == 0) continue;
        auto slice = x_sorted_packed.narrow(0, offsets_cpu[p], N);
        x_sorted_rev_packed.narrow(0, offsets_cpu[p], N).copy_(slice.flip(0));
    }

    // Step 4: Backward scan with saving
    mamba3_scan_fwd_save_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        x_sorted_rev_packed.data_ptr<float>(),
        bwd_scan_out_packed.data_ptr<float>(),
        initial_states.data_ptr<float>(),
        final_bwd.data_ptr<float>(),
        offsets_t.data_ptr<int>(),
        rev_fwd.data_ptr<int>(),  // not reversed — the data is already flipped
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
        d_model, d_inner, d_state
    );
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
    // Dims
    int d_model, int d_state, int d_inner, int num_params
) {
    if (num_params == 0) return;

    TORCH_CHECK(d_state % 2 == 0, "d_state must be even for paired RoPE (got ", d_state, ")");
    TORCH_CHECK(d_state <= MAX_D_STATE, "d_state exceeds MAX_D_STATE (", d_state, " > ", MAX_D_STATE, ")");

    auto dev = d_fwd_scan_out_packed.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Zero initial states for bilevel backward h_prev at step 0
    auto initial_states = torch::zeros({num_params, d_inner, d_state}, float_opts);

    auto rev_fwd = torch::zeros({num_params}, int_opts);
    auto rev_bwd = torch::zeros({num_params}, int_opts);  // data was pre-reversed

    int scan_smem = 2 * d_inner * sizeof(float);  // s_x_branch + s_d_dt_raw

    // Forward direction backward scan
    auto d_x_sorted_fwd = torch::zeros_like(d_x_sorted_packed);
    mamba3_scan_backward_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        d_fwd_scan_out_packed.data_ptr<float>(),
        x_sorted_packed.data_ptr<float>(),
        fwd_saved_states_packed.data_ptr<float>(),
        fwd_saved_xb_packed.data_ptr<float>(),
        fwd_saved_z_packed.data_ptr<float>(),
        fwd_saved_dt_packed.data_ptr<float>(),
        offsets_t.data_ptr<int>(),
        rev_fwd.data_ptr<int>(),
        initial_states.data_ptr<float>(),
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
        d_mamba_fwd_B_proj.data_ptr<float>(),
        d_mamba_fwd_C_proj.data_ptr<float>(),
        d_mamba_fwd_A_log.data_ptr<float>(),
        d_mamba_fwd_D.data_ptr<float>(),
        d_mamba_fwd_rope.data_ptr<float>(),
        d_x_sorted_fwd.data_ptr<float>(),
        d_model, d_inner, d_state
    );

    // Build reversed x_sorted for bwd direction backward
    // The bwd scan used reversed x_sorted in forward, so we need that here too
    auto total_N = x_sorted_packed.size(0);
    auto x_sorted_rev_packed = torch::empty_like(x_sorted_packed);
    // Read offsets from GPU
    auto offsets_cpu = offsets_t.to(torch::kCPU);
    auto offsets_ptr = offsets_cpu.data_ptr<int>();
    for (int p = 0; p < num_params; p++) {
        int start = offsets_ptr[p];
        int N = offsets_ptr[p + 1] - start;
        if (N == 0) continue;
        x_sorted_rev_packed.narrow(0, start, N).copy_(
            x_sorted_packed.narrow(0, start, N).flip(0));
    }

    // Backward direction backward scan
    auto d_x_sorted_bwd = torch::zeros({total_N, d_model}, float_opts);
    mamba3_scan_backward_batched_kernel<<<num_params, d_inner, scan_smem>>>(
        d_bwd_scan_out_packed.data_ptr<float>(),
        x_sorted_rev_packed.data_ptr<float>(),
        bwd_saved_states_packed.data_ptr<float>(),
        bwd_saved_xb_packed.data_ptr<float>(),
        bwd_saved_z_packed.data_ptr<float>(),
        bwd_saved_dt_packed.data_ptr<float>(),
        offsets_t.data_ptr<int>(),
        rev_bwd.data_ptr<int>(),
        initial_states.data_ptr<float>(),
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
        d_mamba_bwd_B_proj.data_ptr<float>(),
        d_mamba_bwd_C_proj.data_ptr<float>(),
        d_mamba_bwd_A_log.data_ptr<float>(),
        d_mamba_bwd_D.data_ptr<float>(),
        d_mamba_bwd_rope.data_ptr<float>(),
        d_x_sorted_bwd.data_ptr<float>(),
        d_model, d_inner, d_state
    );

    // Combine: flip bwd back and add
    // Each param's bwd portion needs to be flipped
    for (int p = 0; p < num_params; p++) {
        int start = offsets_ptr[p];
        int N = offsets_ptr[p + 1] - start;
        if (N == 0) continue;
        auto bwd_slice = d_x_sorted_bwd.narrow(0, start, N);
        d_x_sorted_packed.narrow(0, start, N).copy_(
            d_x_sorted_fwd.narrow(0, start, N) + bwd_slice.flip(0));
    }
}
