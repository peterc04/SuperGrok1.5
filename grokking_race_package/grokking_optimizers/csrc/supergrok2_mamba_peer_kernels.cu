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

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

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
    float* s_z_branch = smem + d_inner; // [d_inner]

    // State in registers
    float h[32];
    float h_snap[32]; // snapshot for RoPE (fixes read-after-write)
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;

    float A[32], freq[32];
    for (int s = 0; s < d_state; s++) {
        A[s] = -expf(A_log[tid * d_state + s]);
        freq[s] = rope_freq[tid * d_state + s];
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

        // State update with trapezoidal + RoPE
        for (int s = 0; s < d_state; s++) {
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);

            // FULL B projection: B[s] = sum_j(B_proj_W[s, j] * x_branch[j])
            float B_val = 0.0f;
            for (int j = 0; j < d_inner; j++) {
                B_val += B_proj_W[s * d_inner + j] * s_x_branch[j];
            }
            float B_bar = dt_val * B_val;

            // RoPE using SNAPSHOT (not in-place updated h)
            float cos_p = cosf(dt_val * freq[s]);
            float sin_p = sinf(dt_val * freq[s]);
            int s_prev = (s > 0) ? s - 1 : d_state - 1;
            float h_rot = h_snap[s] * cos_p - h_snap[s_prev] * sin_p;

            h[s] = A_bar * h_rot + B_bar;
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

    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);
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
                dot += query[d] * keys_A[k * half_d + d];
            if (dot > best_score_a) { best_score_a = dot; best_a = k; }
        }

        int best_b = 0;
        float best_score_b = -1e30f;
        for (int k = 0; k < pk_dim; k++) {
            float dot = 0.0f;
            for (int d = 0; d < half_d; d++)
                dot += query[half_d + d] * keys_B[k * half_d + d];
            if (dot > best_score_b) { best_score_b = dot; best_b = k; }
        }

        int expert_idx = best_a * pk_dim + best_b;

        // Expert MLP: z = relu(W1 * g + b1), out = W2 @ z + b2
        float head_out = expert_b2[expert_idx];
        for (int h = 0; h < expert_hidden; h++) {
            float z_val = expert_W1[expert_idx * expert_hidden + h] * g
                        + expert_b1[expert_idx * expert_hidden + h];
            z_val = fmaxf(z_val, 0.0f);  // ReLU
            head_out += expert_W2[expert_idx * expert_hidden + h] * z_val;
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
//  Launcher: Full Mamba-3 + PEER step
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
    int expert_hidden, int num_experts
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

    // Shared memory for scan: x_branch + z_branch
    int scan_smem = 2 * d_inner * sizeof(float);

    // Forward scan
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
        N, d_model, d_inner, d_state, 0  // forward
    );

    // Backward scan (reverse direction)
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
                N, d_model, d_inner, gru_hidden,
                num_heads, pk_dim, expert_hidden, num_experts
            );
        }));
    }
}
