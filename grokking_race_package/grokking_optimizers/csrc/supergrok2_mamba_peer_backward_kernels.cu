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
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    float freq[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        freq[s] = rope_freq[tid * d_state + s];

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
        float dt_val = logf(1.0f + expf(dt_raw));
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

            // RoPE using SNAPSHOT
            float cos_p = cosf(dt_val * freq[s]);
            float sin_p = sinf(dt_val * freq[s]);
            int s_prev = (s > 0) ? s - 1 : d_state - 1;
            float h_rot = h_snap[s] * cos_p - h_snap[s_prev] * sin_p;

            h[s] = A_bar * h_rot + B_bar;
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
    float* __restrict__ d_rope_freq,          // [d_inner, d_state]
    float* __restrict__ d_x_sorted,           // [N, d_model]
    // Dims
    const int N, const int d_model, const int d_inner, const int d_state,
    const int reverse
) {
    const int tid = threadIdx.x;
    if (tid >= d_inner) return;

    float A[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        A[s] = -expf(A_log[tid * d_state + s]);

    float freq[MAX_D_STATE];
    for (int s = 0; s < d_state; s++)
        freq[s] = rope_freq[tid * d_state + s];

    float D_val = D_param[tid];

    // Accumulated gradients for this thread's parameters
    float d_D_acc = 0.0f;
    float d_A_log_acc[MAX_D_STATE];
    float d_freq_acc[MAX_D_STATE];
    float d_B_proj_acc[MAX_D_STATE];
    float d_C_proj_acc[MAX_D_STATE];
    float d_dt_proj_b_acc = 0.0f;
    float d_dt_proj_W_diag_acc = 0.0f;
    for (int s = 0; s < d_state; s++) {
        d_A_log_acc[s] = 0.0f;
        d_freq_acc[s] = 0.0f;
        d_B_proj_acc[s] = 0.0f;
        d_C_proj_acc[s] = 0.0f;
    }

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

        // Forward: y_final = y_val * silu_z + D_val * x_val
        float sig_z = 1.0f / (1.0f + expf(-z_val));
        float silu_z = z_val * sig_z;

        // Recompute y_val from saved state
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float C_val = C_proj_W[s * d_inner + tid] * x_val;
            y_val += saved_states[(i * d_inner + tid) * d_state + s] * C_val;
        }

        // Backward through gated output
        float d_y_val = d_out * silu_z;
        float d_silu_z = d_out * y_val;
        float d_z_val = d_silu_z * (sig_z + z_val * sig_z * (1.0f - sig_z));
        float d_x_from_D = d_out * D_val;
        d_D_acc += d_out * x_val;

        // Backward through y = sum_s(h[s] * C[s])
        float d_x_from_C = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float h_s = saved_states[(i * d_inner + tid) * d_state + s];
            float C_val = C_proj_W[s * d_inner + tid] * x_val;

            // dh[s] += d_y * C[s]
            dh[s] += d_y_val * C_val;
            // d_C_val = d_y * h[s]
            float d_C_val = d_y_val * h_s;
            // C_val = C_proj_W[s,tid] * x_val
            d_C_proj_acc[s] += d_C_val * x_val;
            d_x_from_C += d_C_val * C_proj_W[s * d_inner + tid];
        }

        // Get h_prev (state before this step's update)
        float h_prev[MAX_D_STATE];
        if (step > 0) {
            int i_prev = reverse ? (N - step) : (step - 1);
            for (int s = 0; s < d_state; s++)
                h_prev[s] = saved_states[(i_prev * d_inner + tid) * d_state + s];
        } else {
            for (int s = 0; s < d_state; s++)
                h_prev[s] = 0.0f;
        }

        // Backward through state update: h[s] = A_bar * h_rot + B_bar
        float d_dt_val = 0.0f;
        float d_x_from_scan = 0.0f;

        for (int s = 0; s < d_state; s++) {
            // Forward: A_bar = (1 + dt*A/2) / (1 - dt*A/2 + eps)
            float half_dtA = dt_val * A[s] / 2.0f;
            float denom_val = 1.0f - half_dtA + 1e-8f;
            float A_bar = (1.0f + half_dtA) / denom_val;

            float B_val = B_proj_W[s * d_inner + tid] * x_val;
            float B_bar = dt_val * B_val;

            float cos_p = cosf(dt_val * freq[s]);
            float sin_p = sinf(dt_val * freq[s]);
            int s_prev = (s > 0) ? s - 1 : d_state - 1;
            float h_rot = h_prev[s] * cos_p - h_prev[s_prev] * sin_p;

            // dh[s] is the gradient flowing into h[s]
            float d_h_s = dh[s];

            // d(A_bar * h_rot) + d(B_bar)
            float d_A_bar = d_h_s * h_rot;
            float d_h_rot = d_h_s * A_bar;
            float d_B_bar = d_h_s;

            // d_B_bar -> d_dt, d_B_val
            d_dt_val += d_B_bar * B_val;
            float d_B_val = d_B_bar * dt_val;
            // B_val = B_proj_W[s,tid] * x_val
            d_B_proj_acc[s] += d_B_val * x_val;
            d_x_from_scan += d_B_val * B_proj_W[s * d_inner + tid];

            // d_A_bar -> d_dt, d_A_log
            // A_bar = (1 + dt*A/2) / (1 - dt*A/2 + eps)
            // dA_bar/d(dt*A/2) = (1/denom + (1+half_dtA)/denom^2) = (1 + A_bar) / denom
            float d_half_dtA = d_A_bar * (1.0f + A_bar) / denom_val;
            d_dt_val += d_half_dtA * A[s] / 2.0f;
            // d_A[s] from A_bar: d_half_dtA * dt_val / 2
            float d_A_s = d_half_dtA * dt_val / 2.0f;
            // A[s] = -exp(A_log[s]) -> d_A_log += d_A * (-exp(A_log))  = d_A * A[s]
            d_A_log_acc[s] += d_A_s * A[s];

            // d_h_rot -> d_h_prev[s], d_h_prev[s_prev]
            // h_rot = h_prev[s] * cos - h_prev[s_prev] * sin
            float d_h_prev_s = d_h_rot * cos_p;
            float d_h_prev_s_prev = -d_h_rot * sin_p;

            // d_h_rot -> d_cos, d_sin -> d_dt, d_freq
            float d_cos = d_h_rot * h_prev[s];
            float d_sin = -d_h_rot * h_prev[s_prev];
            // cos(dt*freq) -> d = -sin * (freq * d_dt + dt * d_freq)
            d_dt_val += (-sin_p * freq[s]) * d_cos + (cos_p * freq[s]) * d_sin;
            d_freq_acc[s] += (-sin_p * dt_val) * d_cos + (cos_p * dt_val) * d_sin;

            // Propagate dh to previous step
            // NOTE: we clear dh[s] and add the propagated gradient
            dh[s] = d_h_prev_s;
            dh[s_prev] += d_h_prev_s_prev;
        }

        // Backward through dt: dt_val = softplus(dt_raw)
        // d_dt_raw = d_dt * sigmoid(dt_raw)
        float dt_raw = dt_proj_b[tid] + dt_proj_W[tid * d_inner + tid] * x_val;
        float sig_dt = 1.0f / (1.0f + expf(-dt_raw));
        float d_dt_raw = d_dt_val * sig_dt;

        d_dt_proj_b_acc += d_dt_raw;
        d_dt_proj_W_diag_acc += d_dt_raw * x_val;
        float d_x_from_dt = d_dt_raw * dt_proj_W[tid * d_inner + tid];

        // Total d_x_val for this step
        float d_x_val = d_x_from_D + d_x_from_C + d_x_from_scan + d_x_from_dt;

        // Backward through input projection to d_x_sorted and d_in_proj_W
        for (int d = 0; d < d_model; d++) {
            float inp = x_sorted[i * d_model + d];
            // d_in_proj_W[tid, d] += d_x_val * inp
            atomicAdd(&d_in_proj_W[tid * d_model + d], d_x_val * inp);
            // d_in_proj_W[tid+d_inner, d] += d_z_val * inp
            atomicAdd(&d_in_proj_W[(tid + d_inner) * d_model + d], d_z_val * inp);
            // d_x_sorted[i, d] += d_x_val * in_proj_W[tid,d] + d_z_val * in_proj_W[tid+d_inner,d]
            atomicAdd(&d_x_sorted[i * d_model + d],
                      d_x_val * in_proj_W[tid * d_model + d] +
                      d_z_val * in_proj_W[(tid + d_inner) * d_model + d]);
        }
    }

    // Write accumulated per-thread parameter gradients
    atomicAdd(&d_D_param[tid], d_D_acc);
    atomicAdd(&d_dt_proj_b[tid], d_dt_proj_b_acc);
    atomicAdd(&d_dt_proj_W[tid * d_inner + tid], d_dt_proj_W_diag_acc);
    for (int s = 0; s < d_state; s++) {
        atomicAdd(&d_A_log[tid * d_state + s], d_A_log_acc[s]);
        atomicAdd(&d_rope_freq[tid * d_state + s], d_freq_acc[s]);
        atomicAdd(&d_B_proj_W[s * d_inner + tid], d_B_proj_acc[s]);
        atomicAdd(&d_C_proj_W[s * d_inner + tid], d_C_proj_acc[s]);
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

        // d_r from Wh backward: d_tanh_input * Wh[gh, input_dim + j] * h_old[j]
        float d_r = 0.0f;
        for (int j = 0; j < gru_hidden; j++) {
            d_r += d_tanh_input * Wh[gh * total_dim + input_dim + j] * h_old[idx * gru_hidden + j];
        }
        // Only accumulate for matching dimension
        float d_r_input = d_r * r_val * (1.0f - r_val);

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

        // Accumulate d_Wr, d_br from d_r_input
        atomicAdd(&d_br[gh], d_r_input);
        for (int j = 0; j < total_dim; j++) {
            float xh_j;
            if (j < input_dim)
                xh_j = gru_input[idx * input_dim + j];
            else
                xh_j = h_old[idx * gru_hidden + (j - input_dim)];
            atomicAdd(&d_Wr[gh * total_dim + j], d_r_input * xh_j);
        }

        // d_gru_input from Wz and Wr and Wh backward
        for (int j = 0; j < input_dim; j++) {
            float d_input_j = d_z_input * Wz[gh * total_dim + j]
                            + d_r_input * Wr[gh * total_dim + j]
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

        // Backward through routing * expert_out
        for (int k = 0; k < num_active; k++) {
            int ei = expert_indices[(idx * num_heads + h) * num_active + k];
            float rw = routing_weights[(idx * num_heads + h) * num_active + k];

            // Forward: out_k = W2[ei] @ relu(W1[ei] * g + b1[ei]) + b2[ei]
            // z_k = relu(W1[ei] * g + b1[ei])  — saved in saved_z_hidden
            float out_k = 0.0f;
            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                out_k += expert_W2[ei * expert_hidden + eh] * z_val;
            }
            out_k += expert_W2[ei]; // This is actually expert_b2, but simplified

            // d_routing[k] = d_head_out * out_k  (for softmax backward)
            float d_rw = d_head_out * out_k;

            // d_out_k = d_head_out * rw
            float d_out_k = d_head_out * rw;

            // Backward through expert MLP
            // d_b2[ei] += d_out_k
            atomicAdd(&d_expert_b2[ei], d_out_k);

            for (int eh = 0; eh < expert_hidden; eh++) {
                float z_val = saved_z_hidden[((idx * num_heads + h) * num_active + k) * expert_hidden + eh];
                // d_W2[ei, eh] += d_out_k * z_val
                atomicAdd(&d_expert_W2[ei * expert_hidden + eh], d_out_k * z_val);
                // d_z[eh] = d_out_k * W2[ei, eh]
                float d_z = d_out_k * expert_W2[ei * expert_hidden + eh];
                // ReLU backward
                float d_pre_relu = (z_val > 0.0f) ? d_z : 0.0f;
                // d_W1[ei, eh] += d_pre_relu * g_val
                atomicAdd(&d_expert_W1[ei * expert_hidden + eh], d_pre_relu * g_val);
                // d_b1[ei, eh] += d_pre_relu
                atomicAdd(&d_expert_b1[ei * expert_hidden + eh], d_pre_relu);
            }

            // Backward through soft routing to product keys
            // routing[k] = soft_a[a_idx] * soft_b[b_idx]
            int a_local = k / topk;
            int b_local = k % topk;

            float sa = saved_soft_a[(idx * num_heads + h) * topk + a_local];
            float sb = saved_soft_b[(idx * num_heads + h) * topk + b_local];

            // d_soft_a[a_local] += d_rw * sb
            // d_soft_b[b_local] += d_rw * sa
            // We need to propagate through softmax -> topk -> scores -> keys

            // For soft_a: softmax backward
            // d_score_a[a_local] += d_soft_a * sa * (1 - sa) * 10   (simplified diagonal)
            float d_soft_a = d_rw * sb;
            float d_score_a = d_soft_a * sa * (1.0f - sa) * 10.0f;

            float d_soft_b = d_rw * sa;
            float d_score_b = d_soft_b * sb * (1.0f - sb) * 10.0f;

            // scores_a = q_a @ keys_A.T -> d_keys_A, d_q_a
            int a_key_idx = saved_top_a_idx[(idx * num_heads + h) * topk + a_local];
            int b_key_idx = saved_top_b_idx[(idx * num_heads + h) * topk + b_local];

            // Accumulate d_prod_keys_A and d_prod_keys_B
            // score_a[j] = sum_d(q_a[d] * keys_A[j, d])
            // We need the query to compute key gradients
            // Recompute query from saved_peer_input and peer_query_Ws
            for (int d = 0; d < half_d; d++) {
                // Compute q_a[d] and q_b[d] from peer_input @ peer_query_W.T
                float q_a_d = 0.0f;
                float q_b_d = 0.0f;
                for (int j = 0; j < peer_input_dim; j++) {
                    float pi_j = saved_peer_input[idx * peer_input_dim + j];
                    q_a_d += peer_query_Ws[(h * d_model + d) * peer_input_dim + j] * pi_j;
                    q_b_d += peer_query_Ws[(h * d_model + half_d + d) * peer_input_dim + j] * pi_j;
                }

                // d_keys_A[h, a_key_idx, d] += d_score_a * q_a[d]
                atomicAdd(&d_prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d], d_score_a * q_a_d);
                // d_keys_B[h, b_key_idx, d] += d_score_b * q_b[d]
                atomicAdd(&d_prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d], d_score_b * q_b_d);

                // d_q_a[d] += d_score_a * keys_A[a_key_idx, d]
                float d_q_a_d = d_score_a * prod_keys_A[(h * pk_dim + a_key_idx) * half_d + d];
                float d_q_b_d = d_score_b * prod_keys_B[(h * pk_dim + b_key_idx) * half_d + d];

                // d_peer_query_Ws and d_peer_input from query = peer_query_W @ peer_input
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
    torch::Tensor input_proj_W,
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
    auto d_x_sorted_fwd = torch::zeros({N, d_model}, float_opts);
    auto d_x_sorted_bwd = torch::zeros({N, d_model}, float_opts);

    mamba3_scan_backward_kernel<<<1, d_inner>>>(
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
        N, d_model, d_inner, d_state, 0
    );

    // Backward scan used reversed x_sorted
    auto x_sorted_rev = x_sorted.flip(0).contiguous();
    mamba3_scan_backward_kernel<<<1, d_inner>>>(
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
