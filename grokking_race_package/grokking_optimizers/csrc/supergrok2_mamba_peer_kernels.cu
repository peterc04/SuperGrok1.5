/*
 * SuperGrok v2 — Mamba-3 + 4-Head PEER + GRU CUDA Kernels (Forward)
 *
 * Three fused kernels for the Mamba-3 + PEER meta-net architecture:
 *
 *   1. input_proj_sort     — Project [grad, sharpness] -> [N, d_model],
 *                            compute sort keys = |grad|
 *   2. mamba3_scan          — Selective scan with trapezoidal discretization
 *                            + RoPE rotation (one direction per call)
 *   3. fused_elem_step      — GRU + multi-head PEER routing + expert MLP
 *                            + mu update + Adam + weight decay
 *
 * Plus sort via thrust::sort_by_key.
 *
 * Supports FP32, FP16, and BF16 parameter tensors.
 * All meta-net state (GRU, Mamba, weights) is always FP32.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

constexpr int SG2M_BLOCK = 256;


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
    const float* __restrict__ rope_freq,  // [d_inner, d_state]
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

    // Shared memory for cross-thread communication
    extern __shared__ float smem[];
    float* s_x_branch = smem;           // [d_inner]

    // State in registers — load from initial_state if provided
    float h[32];
    float h_snap[32]; // snapshot for RoPE (fixes read-after-write)
    if (initial_state != nullptr) {
        for (int s = 0; s < d_state; s++) h[s] = initial_state[tid * d_state + s];
    } else {
        for (int s = 0; s < d_state; s++) h[s] = 0.0f;
    }

    const int half_d_state = d_state / 2;
    float A[32], freq[16];  // freq is d_state/2
    for (int s = 0; s < d_state; s++) {
        A[s] = -expf(A_log[tid * d_state + s]);
    }
    for (int p = 0; p < half_d_state; p++) {
        freq[p] = rope_freq[tid * half_d_state + p];
    }
    float D_val = D_param[tid];

    for (int step = 0; step < N; step++) {
        int i = reverse ? (N - 1 - step) : step;

        // Input projection: each thread computes its own x and z
        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float inp = x_sorted[i * d_model + d];
            x_val += in_proj_W[tid * d_model + d] * inp;
            z_val += in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        // Write x_branch to shared memory for cross-thread access
        s_x_branch[tid] = x_val;
        __syncthreads();

        // FULL dt projection: dt[tid] = sum_j(dt_proj_W[tid, j] * x_branch[j]) + dt_proj_b[tid]
        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++) {
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        }
        float dt_val = logf(1.0f + expf(dt_raw)); // softplus

        // Snapshot h for RoPE (fixes read-after-write)
        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        // State update with trapezoidal + paired RoPE
        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);

            // FULL B projection: B[s] = sum_j(B_proj_W[s, j] * x_branch[j])
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
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
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
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
    // Total smem: 2*op_size + 3*gru_mat_size + 3*gru_hidden floats

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
    float fwd_ctx[16], bwd_ctx[16];  // max d_model = 16
    for (int d = 0; d < d_model; d++) {
        float fwd_val = 0.0f, bwd_val = 0.0f;
        for (int j = 0; j < d_inner; j++) {
            fwd_val += s_out_fwd[d * d_inner + j] * fwd_scan_out[idx * d_inner + j];
            bwd_val += s_out_bwd[d * d_inner + j] * bwd_scan_out[idx * d_inner + j];
        }
        fwd_ctx[d] = fwd_val;
        bwd_ctx[d] = bwd_val;
    }

    // 2. GRU update (using shared memory weights)
    float h_old[8];  // max gru_hidden = 8
    for (int j = 0; j < gru_hidden; j++) {
        h_old[j] = gru_state[idx * gru_hidden + j];
    }

    float h_new[8];
    float z_gate[8], r_gate[8];
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
        float query[16];  // max d_model = 16
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
        if (expert_counts != nullptr)
            atomicAdd(&expert_counts[expert_idx], 1);

        // Expert MLP: z = relu(W1 * g + b1), out = W2 @ z + b2
        // Use __ldg for read-only expert weight loads (L1 cache hint)
        float head_out = __ldg(&expert_b2[expert_idx]);
        for (int h = 0; h < expert_hidden; h++) {
            float z_val = __ldg(&expert_W1[expert_idx * expert_hidden + h]) * g
                        + __ldg(&expert_b1[expert_idx * expert_hidden + h]);
            z_val = fmaxf(z_val, 0.0f);  // ReLU
            head_out += __ldg(&expert_W2[expert_idx * expert_hidden + h]) * z_val;
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
    const float* __restrict__ rope_freq,           // [d_inner, d_state]
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
    float h[32], h_snap[32];
    const float* my_init = initial_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[32], freq[16];  // freq is d_state/2
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
            x_val += in_proj_W[tid * d_model + d] * inp;
            z_val += in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = logf(1.0f + expf(dt_raw));

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
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

    extern __shared__ float smem[];
    float* s_x_branch = smem;

    float h[32], h_snap[32];
    const float* my_init = init_states + param_idx * d_inner * d_state;
    for (int s = 0; s < d_state; s++) h[s] = my_init[tid * d_state + s];

    const int half_d_state = d_state / 2;
    float A[32], freq[16];
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
            x_val += in_proj_W[tid * d_model + d] * inp;
            z_val += in_proj_W[(tid + d_inner) * d_model + d] * inp;
        }

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = dt_proj_b[tid];
        for (int j = 0; j < d_inner; j++)
            dt_raw += dt_proj_W[tid * d_inner + j] * s_x_branch[j];
        float dt_val = logf(1.0f + expf(dt_raw));

        for (int s = 0; s < d_state; s++) h_snap[s] = h[s];

        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++)
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
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
                C_val += C_proj_W[s * d_inner + j] * s_x_branch[j];
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
    torch::Tensor mamba_fwd_rope,     // [d_inner, d_state]
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

    auto dev = grad.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Step 1: Input projection + sort key computation
    auto x_proj = torch::empty({N, d_model}, float_opts);
    auto sort_keys = torch::empty({N}, float_opts);
    auto sort_indices = torch::empty({N}, int_opts);

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

    // Step 2: Bidirectional Mamba-3 scan
    auto fwd_scan_out = torch::empty({N, d_inner}, float_opts);
    auto new_fwd_state = torch::empty({d_inner, d_state}, float_opts);
    auto bwd_scan_out = torch::empty({N, d_inner}, float_opts);
    auto new_bwd_state = torch::empty({d_inner, d_state}, float_opts);

    // Shared memory for scan: x_branch only (z computed per-thread)
    int scan_smem = d_inner * sizeof(float);

    // Forward scan — pass initial_state if available
    const float* fwd_init_ptr = (mamba_fwd_state.numel() > 0) ?
        mamba_fwd_state.data_ptr<float>() : nullptr;
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
        N, d_model, d_inner, d_state, 0  // forward
    );

    // Backward scan (reverse direction)
    const float* bwd_init_ptr = (mamba_bwd_state.numel() > 0) ?
        mamba_bwd_state.data_ptr<float>() : nullptr;
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
        N, d_model, d_inner, d_state, 1  // reverse
    );

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
        // Shared memory: out_proj (2 * d_model * d_inner) + GRU (3 matrices + 3 biases)
        int gru_input_dim_val = 2 + 2 * d_model;
        int gru_row_len = gru_input_dim_val + gru_hidden;
        int smem_bytes = (2 * d_model * d_inner
                        + 3 * gru_hidden * gru_row_len
                        + 3 * gru_hidden) * sizeof(float);
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

    auto dev = grads[0].device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);

    // Step 1: Input projection + sort for all params, pack sorted data
    std::vector<int> N_vec(num_params);
    std::vector<torch::Tensor> x_sorted_list(num_params);
    std::vector<torch::Tensor> sort_idx_list(num_params);
    std::vector<torch::Tensor> unsort_idx_list(num_params);
    int total_N = 0;

    for (int p = 0; p < num_params; p++) {
        int N = grads[p].numel();
        N_vec[p] = N;
        if (N == 0) continue;

        auto x_proj = torch::empty({N, d_model}, float_opts);
        auto sort_keys = torch::empty({N}, float_opts);
        auto sort_indices = torch::empty({N}, int_opts);

        const int grid = (N + SG2M_BLOCK - 1) / SG2M_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grads[p].scalar_type(), "input_proj_sort_batch", ([&] {
            input_proj_sort_kernel<scalar_t><<<grid, SG2M_BLOCK>>>(
                grads[p].data_ptr<scalar_t>(),
                sharpness_list[p].data_ptr<scalar_t>(),
                x_proj.data_ptr<float>(),
                sort_keys.data_ptr<float>(),
                sort_indices.data_ptr<int>(),
                input_proj_W.data_ptr<float>(),
                input_proj_b.data_ptr<float>(),
                N, d_model
            );
        }));

        {
            thrust::device_ptr<float> keys_ptr(sort_keys.data_ptr<float>());
            thrust::device_ptr<int> idx_ptr(sort_indices.data_ptr<int>());
            thrust::sort_by_key(keys_ptr, keys_ptr + N, idx_ptr);
        }

        auto idx_long = sort_indices.to(torch::kLong);
        x_sorted_list[p] = x_proj.index_select(0, idx_long);
        sort_idx_list[p] = sort_indices;

        auto unsort = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kLong));
        unsort.scatter_(0, idx_long,
            torch::arange(N, torch::TensorOptions().device(dev).dtype(torch::kLong)));
        unsort_idx_list[p] = unsort;

        total_N += N;
    }

    if (total_N == 0) return;

    // Step 2: Pack sorted data and build offset table
    // Build offset table
    std::vector<int> offsets_cpu(num_params + 1);
    offsets_cpu[0] = 0;
    for (int p = 0; p < num_params; p++)
        offsets_cpu[p + 1] = offsets_cpu[p] + N_vec[p];

    auto offsets_t = torch::from_blob(offsets_cpu.data(), {num_params + 1},
        torch::kInt32).to(dev);

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

    int scan_smem = d_inner * sizeof(float);

    // Step 3+4: Combined forward + backward scan in single launch
    // Grid = 2*num_params: first half fwd, second half bwd (reversed)
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

    // Step 5: Copy final states back + unsort + fused_elem_step per param
    // Pre-compute all unsorted scan outputs, then launch kernels on streams
    int gru_input_dim_val = 2 + 2 * d_model;
    int gru_row_len = gru_input_dim_val + gru_hidden;
    int smem_bytes = (2 * d_model * d_inner
                    + 3 * gru_hidden * gru_row_len
                    + 3 * gru_hidden) * sizeof(float);

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

    // Launch fused_elem_step kernels on a pool of streams for concurrency
    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++)
        cudaStreamCreate(&streams[s]);

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

    // Sync all streams and clean up
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }
}
