/*
 * SuperGrok v2 — ISAB + PEER + Recurrent Meta-Net CUDA Kernels
 *
 * Two fused kernels for the ISAB+PEER+Recurrent meta-net architecture:
 *
 *   1. isab_reduce    — Inducing points attend to all elements (N → M)
 *   2. isab_peer_elem — Per-element: recurrent update + readback from I_up
 *                       + PEER product-key routing + expert MLP + skip
 *
 * Plus the v1.5 Adam kernel is reused for the parameter update.
 *
 * Complexity: O(N × M × d) — linear in N.
 * All meta-net weights loaded into shared memory for fast access.
 *
 * Supports FP32, FP16, and BF16 parameter tensors.
 * All meta-net state (recurrent_h, weights) is always FP32.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int SG2_BLOCK = 256;


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 1: ISAB Reduction — inducing points attend to elements
//
//  Each block handles one inducing point.
//  Threads cooperate to compute attention scores over N elements,
//  then compute the weighted sum of projected element values.
//
//  Input:  x = input_proj(inp) — [N, d_model]  (pre-computed by caller)
//  Output: I_up [M, d_model]
// ═══════════════════════════════════════════════════════════════════════

__global__ void isab_reduce_kernel(
    const float* __restrict__ x,              // [N, d_model] — projected inputs
    const float* __restrict__ inducing_pts,   // [M, d_model]
    const float* __restrict__ induce_q_W,     // [d_model, d_model]
    const float* __restrict__ induce_k_W,     // [d_model, d_model]
    const float* __restrict__ induce_v_W,     // [d_model, d_model]
    float* __restrict__ I_up,                 // [M, d_model] — output
    float* __restrict__ attn_buf,             // [M, N] — global memory for attention scores
    const int N,
    const int M,
    const int d_model
) {
    const int m = blockIdx.x;  // inducing point index
    if (m >= M) return;

    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    // Layout: q_m[d_model] only — attn_raw moved to global memory
    float* q_m = smem;
    float* attn_raw = attn_buf + m * N;

    // Step 1: Compute q for this inducing point: q_m = induce_q_W @ inducing_pts[m]
    if (tid < d_model) {
        float val = 0.0f;
        for (int j = 0; j < d_model; j++) {
            val += induce_q_W[tid * d_model + j] * inducing_pts[m * d_model + j];
        }
        q_m[tid] = val;
    }
    __syncthreads();

    // Step 2: Compute attention scores: score[n] = q_m · k_n / sqrt(d_model)
    float scale = rsqrtf(static_cast<float>(d_model));
    for (int n = tid; n < N; n += blockDim.x) {
        // Compute k_n = induce_k_W @ x[n]
        float dot = 0.0f;
        for (int d = 0; d < d_model; d++) {
            float k_d = 0.0f;
            for (int j = 0; j < d_model; j++) {
                k_d += induce_k_W[d * d_model + j] * x[n * d_model + j];
            }
            dot += q_m[d] * k_d;
        }
        attn_raw[n] = dot * scale;
    }
    __syncthreads();

    // Step 3: Softmax over N (cooperative)
    // Find max
    float local_max = -1e30f;
    for (int n = tid; n < N; n += blockDim.x)
        local_max = fmaxf(local_max, attn_raw[n]);

    // Block-wide max reduction via shared memory
    __shared__ float reduce_buf[SG2_BLOCK];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        __syncthreads();
    }
    float global_max = reduce_buf[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float e = expf(attn_raw[n] - global_max);
        attn_raw[n] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }
    float global_sum = reduce_buf[0] + 1e-12f;

    // Normalize
    for (int n = tid; n < N; n += blockDim.x)
        attn_raw[n] /= global_sum;
    __syncthreads();

    // Step 4: Weighted sum of values: I_up[m,d] = sum_n attn[n] * v_n[d]
    // Each thread handles one dimension d
    if (tid < d_model) {
        float val = 0.0f;
        for (int n = 0; n < N; n++) {
            // Compute v_n[tid] = induce_v_W[tid] @ x[n]
            float v_d = 0.0f;
            for (int j = 0; j < d_model; j++) {
                v_d += induce_v_W[tid * d_model + j] * x[n * d_model + j];
            }
            val += attn_raw[n] * v_d;
        }
        I_up[m * d_model + tid] = val;
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 2: Fused per-element — recurrent + readback + PEER + expert
//
//  Each thread processes one gradient element:
//    1. Input projection: inp = [grad, sharpness] → x = input_proj(inp)
//    2. Recurrent update: h_new = tanh(W_h @ h + W_x @ inp)
//    3. Cross-attend to I_up: context = softmax(read_q(x) · I_up^T) · I_up
//    4. PEER routing: query = peer_query_W @ [h_new, context, inp]
//       Split query → scores_A, scores_B → expert_idx
//    5. Expert MLP: z = relu(W1[idx] · grad + b1[idx])
//                   out = W2[idx] · z + b2[idx]
//    6. Skip: smart_grad = grad + rescale * out
//
//  Shared memory holds: I_up, read_q_W, input_proj weights,
//                        W_h, W_x, peer_query_W, product_keys
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void isab_peer_elem_kernel(
    const scalar_t* __restrict__ grad,           // [N]
    const scalar_t* __restrict__ sharpness,      // [N]
    float* __restrict__ recurrent_h,             // [N, recurrent_dim] — updated
    scalar_t* __restrict__ smart_grad_out,       // [N] — output
    // Pre-computed from kernel 1
    const float* __restrict__ I_up,              // [M, d_model]
    // Weights (all FP32)
    const float* __restrict__ input_proj_W,      // [d_model, 2]
    const float* __restrict__ input_proj_b,      // [d_model]
    const float* __restrict__ read_q_W,          // [d_model, d_model]
    const float* __restrict__ W_h,               // [recurrent_dim, recurrent_dim]
    const float* __restrict__ W_x_W,             // [recurrent_dim, 2]
    const float* __restrict__ W_x_b,             // [recurrent_dim]
    const float* __restrict__ peer_query_W,      // [d_model, peer_input_dim]
    const float* __restrict__ product_keys_A,    // [pk_dim, d_model/2]
    const float* __restrict__ product_keys_B,    // [pk_dim, d_model/2]
    const float* __restrict__ expert_W1,         // [num_experts, expert_hidden, 1]
    const float* __restrict__ expert_b1,         // [num_experts, expert_hidden]
    const float* __restrict__ expert_W2,         // [num_experts, 1, expert_hidden]
    const float* __restrict__ expert_b2,         // [num_experts, 1]
    const float rescale,
    const int N,
    const int M,
    const int d_model,
    const int recurrent_dim,
    const int pk_dim,
    const int expert_hidden,
    const int num_experts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int half_d = d_model / 2;
    const int peer_input_dim = recurrent_dim + d_model + 2;

    // Read inputs
    const float g = static_cast<float>(grad[idx]);
    const float s = static_cast<float>(sharpness[idx]);

    // 1. Input projection: x[d] = input_proj_W[d,0]*g + input_proj_W[d,1]*s + input_proj_b[d]
    // Use stack-allocated arrays (d_model is small, typically 8)
    float x_proj[16];  // max d_model = 16
    for (int d = 0; d < d_model; d++) {
        x_proj[d] = input_proj_W[d * 2] * g + input_proj_W[d * 2 + 1] * s + input_proj_b[d];
    }

    // 2. Recurrent update: h_new = tanh(W_h @ h_old + W_x @ [g, s] + W_x_b)
    float h_new[16];  // max recurrent_dim = 16
    for (int r = 0; r < recurrent_dim; r++) {
        float val = W_x_W[r * 2] * g + W_x_W[r * 2 + 1] * s + W_x_b[r];
        for (int j = 0; j < recurrent_dim; j++) {
            val += W_h[r * recurrent_dim + j] * recurrent_h[idx * recurrent_dim + j];
        }
        h_new[r] = tanhf(val);
    }
    // Write h_new back
    for (int r = 0; r < recurrent_dim; r++) {
        recurrent_h[idx * recurrent_dim + r] = h_new[r];
    }

    // 3. Cross-attend to I_up: context = softmax(read_q(x) · I_up^T / sqrt(d)) · I_up
    // Compute read_q: rq[d] = read_q_W[d,:] @ x_proj
    float rq[16];
    for (int d = 0; d < d_model; d++) {
        float val = 0.0f;
        for (int j = 0; j < d_model; j++) {
            val += read_q_W[d * d_model + j] * x_proj[j];
        }
        rq[d] = val;
    }

    // Attention scores: score[m] = rq · I_up[m] / sqrt(d_model)
    float scale = rsqrtf(static_cast<float>(d_model));
    float attn_scores[32];  // max M = 32
    float max_score = -1e30f;
    for (int m = 0; m < M; m++) {
        float dot = 0.0f;
        for (int d = 0; d < d_model; d++) {
            dot += rq[d] * I_up[m * d_model + d];
        }
        attn_scores[m] = dot * scale;
        max_score = fmaxf(max_score, attn_scores[m]);
    }
    // Softmax
    float sum_exp = 0.0f;
    for (int m = 0; m < M; m++) {
        attn_scores[m] = expf(attn_scores[m] - max_score);
        sum_exp += attn_scores[m];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    // Weighted sum: context[d] = sum_m attn[m] * I_up[m,d]
    float context[16];
    for (int d = 0; d < d_model; d++) {
        float val = 0.0f;
        for (int m = 0; m < M; m++) {
            val += (attn_scores[m] * inv_sum) * I_up[m * d_model + d];
        }
        context[d] = val;
    }

    // 4. PEER routing
    // peer_input = [h_new, context, g, s]
    // query = peer_query_W @ peer_input
    float query[16];  // d_model
    for (int d = 0; d < d_model; d++) {
        float val = 0.0f;
        int offset = 0;
        // h_new part
        for (int j = 0; j < recurrent_dim; j++) {
            val += peer_query_W[d * peer_input_dim + offset + j] * h_new[j];
        }
        offset += recurrent_dim;
        // context part
        for (int j = 0; j < d_model; j++) {
            val += peer_query_W[d * peer_input_dim + offset + j] * context[j];
        }
        offset += d_model;
        // inp part [g, s]
        val += peer_query_W[d * peer_input_dim + offset] * g;
        val += peer_query_W[d * peer_input_dim + offset + 1] * s;
        query[d] = val;
    }

    // Product key routing: split query → q_a, q_b
    // idx_a = argmax(q_a @ keys_A^T), idx_b = argmax(q_b @ keys_B^T)
    int best_a = 0;
    float best_score_a = -1e30f;
    for (int k = 0; k < pk_dim; k++) {
        float dot = 0.0f;
        for (int d = 0; d < half_d; d++) {
            dot += query[d] * product_keys_A[k * half_d + d];
        }
        if (dot > best_score_a) {
            best_score_a = dot;
            best_a = k;
        }
    }
    int best_b = 0;
    float best_score_b = -1e30f;
    for (int k = 0; k < pk_dim; k++) {
        float dot = 0.0f;
        for (int d = 0; d < half_d; d++) {
            dot += query[half_d + d] * product_keys_B[k * half_d + d];
        }
        if (dot > best_score_b) {
            best_score_b = dot;
            best_b = k;
        }
    }
    int expert_idx = best_a * pk_dim + best_b;

    // 5. Expert MLP evaluation
    // z = relu(W1[expert] * g + b1[expert])  — W1 is [expert_hidden, 1]
    // out = W2[expert] * z + b2[expert]       — W2 is [1, expert_hidden]
    float out = expert_b2[expert_idx];
    for (int h = 0; h < expert_hidden; h++) {
        float z = expert_W1[expert_idx * expert_hidden + h] * g
                + expert_b1[expert_idx * expert_hidden + h];
        z = fmaxf(z, 0.0f);  // ReLU
        out += expert_W2[expert_idx * expert_hidden + h] * z;
    }

    // 6. Skip connection
    smart_grad_out[idx] = static_cast<scalar_t>(g + rescale * out);
}


// ═══════════════════════════════════════════════════════════════════════
//  Kernel 3: Input projection — [g, s] → [N, d_model]
//  Simple linear projection, separated to keep isab_reduce clean
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void input_proj_kernel(
    const scalar_t* __restrict__ grad,       // [N]
    const scalar_t* __restrict__ sharpness,  // [N]
    float* __restrict__ x_out,               // [N, d_model]
    const float* __restrict__ proj_W,        // [d_model, 2]
    const float* __restrict__ proj_b,        // [d_model]
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
}


// ═══════════════════════════════════════════════════════════════════════
//  C++ Launcher Functions
// ═══════════════════════════════════════════════════════════════════════

void launch_isab_peer_metanet(
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor recurrent_h,       // [N, recurrent_dim] — updated in-place
    torch::Tensor smart_grad,        // [N] — output
    // Weight tensors (all FP32)
    torch::Tensor inducing_points,   // [M, d_model]
    torch::Tensor input_proj_W,      // [d_model, 2]
    torch::Tensor input_proj_b,      // [d_model]
    torch::Tensor induce_q_W,       // [d_model, d_model]
    torch::Tensor induce_k_W,       // [d_model, d_model]
    torch::Tensor induce_v_W,       // [d_model, d_model]
    torch::Tensor read_q_W,         // [d_model, d_model]
    torch::Tensor W_h,              // [recurrent_dim, recurrent_dim]
    torch::Tensor W_x_W,            // [recurrent_dim, 2]
    torch::Tensor W_x_b,            // [recurrent_dim]
    torch::Tensor peer_query_W,     // [d_model, peer_input_dim]
    torch::Tensor product_keys_A,   // [pk_dim, d_model/2]
    torch::Tensor product_keys_B,   // [pk_dim, d_model/2]
    torch::Tensor expert_W1,        // [num_experts, expert_hidden, 1]
    torch::Tensor expert_b1,        // [num_experts, expert_hidden]
    torch::Tensor expert_W2,        // [num_experts, 1, expert_hidden]
    torch::Tensor expert_b2,        // [num_experts, 1]
    float rescale,
    int num_inducing,
    int d_model,
    int pk_dim,
    int expert_hidden,
    int recurrent_dim,
    int num_experts
) {
    const int N = grad.numel();
    if (N == 0) return;

    auto dev = grad.device();
    auto float_opts = torch::TensorOptions().device(dev).dtype(torch::kFloat32);

    // Step 1: Input projection — [N] → [N, d_model]
    auto x_proj = torch::empty({N, d_model}, float_opts);
    {
        const int grid = (N + SG2_BLOCK - 1) / SG2_BLOCK;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grad.scalar_type(), "input_proj", ([&] {
            input_proj_kernel<scalar_t><<<grid, SG2_BLOCK>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                x_proj.data_ptr<float>(),
                input_proj_W.data_ptr<float>(),
                input_proj_b.data_ptr<float>(),
                N, d_model
            );
        }));
    }

    // Step 2: ISAB reduction — inducing points attend to elements
    auto I_up = torch::empty({num_inducing, d_model}, float_opts);
    // Attention scores in global memory (avoids shared memory overflow for large N)
    auto attn_buf = torch::empty({num_inducing, N}, float_opts);
    {
        // Shared memory: q_m[d_model] only
        int smem_bytes = d_model * sizeof(float);
        isab_reduce_kernel<<<num_inducing, SG2_BLOCK, smem_bytes>>>(
            x_proj.data_ptr<float>(),
            inducing_points.data_ptr<float>(),
            induce_q_W.data_ptr<float>(),
            induce_k_W.data_ptr<float>(),
            induce_v_W.data_ptr<float>(),
            I_up.data_ptr<float>(),
            attn_buf.data_ptr<float>(),
            N, num_inducing, d_model
        );
    }

    // Step 3: Per-element fused kernel
    {
        const int grid = (N + SG2_BLOCK - 1) / SG2_BLOCK;

        // Reshape expert weights for contiguous 2D access
        auto eW1 = expert_W1.reshape({num_experts, expert_hidden}).contiguous();
        auto eW2 = expert_W2.reshape({num_experts, expert_hidden}).contiguous();
        auto eb1 = expert_b1.contiguous();
        auto eb2 = expert_b2.reshape({num_experts}).contiguous();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            grad.scalar_type(), "isab_peer_elem", ([&] {
            isab_peer_elem_kernel<scalar_t><<<grid, SG2_BLOCK>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                recurrent_h.data_ptr<float>(),
                smart_grad.data_ptr<scalar_t>(),
                I_up.data_ptr<float>(),
                input_proj_W.data_ptr<float>(),
                input_proj_b.data_ptr<float>(),
                read_q_W.data_ptr<float>(),
                W_h.data_ptr<float>(),
                W_x_W.data_ptr<float>(),
                W_x_b.data_ptr<float>(),
                peer_query_W.data_ptr<float>(),
                product_keys_A.data_ptr<float>(),
                product_keys_B.data_ptr<float>(),
                eW1.data_ptr<float>(),
                eb1.data_ptr<float>(),
                eW2.data_ptr<float>(),
                eb2.data_ptr<float>(),
                rescale,
                N, num_inducing, d_model, recurrent_dim,
                pk_dim, expert_hidden, num_experts
            );
        }));
    }
}
