/*
 * SuperGrok v2.0 — CUDA Kernels
 *
 * Replaces SuperGrok v1.5's per-element MLP meta-net with DeepSeek Sparse
 * Attention (DSA).  The meta-net processes (grad, sharpness) pairs through
 * cross-element sparse attention to produce a correction signal.
 *
 * Six kernels:
 *   1. dsa_project         — project each element to Q/K/V + indexer keys
 *   2. dsa_indexer_topk    — lightning indexer scores + top-k selection
 *   3. dsa_sparse_attention — sparse attention + output projection + skip
 *   4. fused_adam_decay     — gating blend + Adam moments + progressive wd
 *   5. sam_perturb          — worst-case parameter perturbation
 *   6. sharpness_restore    — |sam_grad - grad| + param restore
 */

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_Q     = 32;
constexpr int TILE_K     = 256;

// Maximum dimensions baked into static shared-memory sizing guards.
// The actual d_head / n_idx_heads are runtime parameters; shared memory
// is allocated dynamically so these are only used for sanity checks.
constexpr int MAX_D_HEAD     = 128;
constexpr int MAX_IDX_HEADS  = 32;
constexpr int MAX_TOP_K      = 512;


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 1: DSA Projection — project each element to Q, K, V, idx_q, idx_k
// ═══════════════════════════════════════════════════════════════════════════
//
// Each thread handles one gradient element i.
//   input_i  = [grad_i, sharpness_i]            (2 scalars)
//   Q[i]     = W_q @ input_i + b_q              -> [d_head]
//   K[i]     = W_k @ input_i + b_k              -> [d_head]
//   V[i]     = W_v @ input_i + b_v              -> [d_head]
//   idx_q[i] = W_iq @ input_i                   -> [n_idx_heads]
//   idx_k[i] = W_ik @ input_i                   -> [n_idx_heads]
//
// Projection weights are small (2 * d_head per matrix) and loaded into
// shared memory cooperatively by the thread block.

template <typename scalar_t>
__global__ void dsa_project_kernel(
    const scalar_t* __restrict__ grad,       // [N]
    const scalar_t* __restrict__ sharpness,  // [N]
    scalar_t* __restrict__ Q,                // [N, d_head]
    scalar_t* __restrict__ K,                // [N, d_head]
    scalar_t* __restrict__ V,                // [N, d_head]
    scalar_t* __restrict__ idx_q,            // [N, n_idx_heads]
    scalar_t* __restrict__ idx_k,            // [N, n_idx_heads]
    const scalar_t* __restrict__ W_q,        // [d_head, 2]
    const scalar_t* __restrict__ b_q,        // [d_head]
    const scalar_t* __restrict__ W_k,        // [d_head, 2]
    const scalar_t* __restrict__ b_k,        // [d_head]
    const scalar_t* __restrict__ W_v,        // [d_head, 2]
    const scalar_t* __restrict__ b_v,        // [d_head]
    const scalar_t* __restrict__ W_iq,       // [n_idx_heads, 2]
    const scalar_t* __restrict__ W_ik,       // [n_idx_heads, 2]
    const int N,
    const int d_head,
    const int n_idx_heads
) {
    // ── Shared memory layout ──────────────────────────────────────────
    // W_q (d_head*2) | b_q (d_head) | W_k (d_head*2) | b_k (d_head)
    // W_v (d_head*2) | b_v (d_head) | W_iq (n_idx*2) | W_ik (n_idx*2)
    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    scalar_t* sW_q  = smem;
    scalar_t* sb_q  = sW_q  + d_head * 2;
    scalar_t* sW_k  = sb_q  + d_head;
    scalar_t* sb_k  = sW_k  + d_head * 2;
    scalar_t* sW_v  = sb_k  + d_head;
    scalar_t* sb_v  = sW_v  + d_head * 2;
    scalar_t* sW_iq = sb_v  + d_head;
    scalar_t* sW_ik = sW_iq + n_idx_heads * 2;
    // Total: 3*(d_head*2 + d_head) + 2*(n_idx_heads*2) = 9*d_head + 4*n_idx_heads

    const int tid = threadIdx.x;
    const int total_weights = d_head * 3 * 3 + n_idx_heads * 4;
    // Load all weights cooperatively
    // W_q
    for (int i = tid; i < d_head * 2; i += blockDim.x)
        sW_q[i] = W_q[i];
    for (int i = tid; i < d_head; i += blockDim.x)
        sb_q[i] = b_q[i];
    // W_k
    for (int i = tid; i < d_head * 2; i += blockDim.x)
        sW_k[i] = W_k[i];
    for (int i = tid; i < d_head; i += blockDim.x)
        sb_k[i] = b_k[i];
    // W_v
    for (int i = tid; i < d_head * 2; i += blockDim.x)
        sW_v[i] = W_v[i];
    for (int i = tid; i < d_head; i += blockDim.x)
        sb_v[i] = b_v[i];
    // W_iq, W_ik (no bias)
    for (int i = tid; i < n_idx_heads * 2; i += blockDim.x)
        sW_iq[i] = W_iq[i];
    for (int i = tid; i < n_idx_heads * 2; i += blockDim.x)
        sW_ik[i] = W_ik[i];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    const scalar_t g = grad[idx];
    const scalar_t s = sharpness[idx];

    // ── Q, K, V projections ───────────────────────────────────────────
    for (int h = 0; h < d_head; h++) {
        Q[idx * d_head + h] = sW_q[h * 2] * g + sW_q[h * 2 + 1] * s + sb_q[h];
        K[idx * d_head + h] = sW_k[h * 2] * g + sW_k[h * 2 + 1] * s + sb_k[h];
        V[idx * d_head + h] = sW_v[h * 2] * g + sW_v[h * 2 + 1] * s + sb_v[h];
    }

    // ── Indexer projections (no bias) ─────────────────────────────────
    for (int h = 0; h < n_idx_heads; h++) {
        idx_q[idx * n_idx_heads + h] = sW_iq[h * 2] * g + sW_iq[h * 2 + 1] * s;
        idx_k[idx * n_idx_heads + h] = sW_ik[h * 2] * g + sW_ik[h * 2 + 1] * s;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 2: Lightning Indexer + Top-K Selection
// ═══════════════════════════════════════════════════════════════════════════
//
// Each thread block handles TILE_Q query elements.
// For each query i, we iterate over ALL key elements in tiles of TILE_K,
// computing the index score:
//   I(i,j) = sum_h( w_h * ReLU( idx_q[i,h] * idx_k[j,h] ) )
// and maintaining a running top-k via insertion sort.
//
// Writes selected_indices[i, 0..top_k-1] to global memory.

template <typename scalar_t>
__global__ void dsa_indexer_topk_kernel(
    const scalar_t* __restrict__ idx_q,          // [N, n_idx_heads]
    const scalar_t* __restrict__ idx_k,          // [N, n_idx_heads]
    const scalar_t* __restrict__ idx_weights,    // [n_idx_heads]
    int* __restrict__ selected_indices,           // [N, top_k]
    const int N,
    const int n_idx_heads,
    const int top_k
) {
    // Each thread handles one query within the TILE_Q block tile.
    const int q_idx = blockIdx.x * TILE_Q + threadIdx.x;

    // Effective top_k when N is smaller
    const int eff_k = (top_k < N) ? top_k : N;

    if (q_idx >= N) return;

    // ── Load this query's idx_q into registers ────────────────────────
    // n_idx_heads is small (e.g. 4), so register file is fine.
    float local_idx_q[MAX_IDX_HEADS];
    for (int h = 0; h < n_idx_heads; h++) {
        local_idx_q[h] = static_cast<float>(idx_q[q_idx * n_idx_heads + h]);
    }

    // ── Load indexer weights into registers ───────────────────────────
    float local_w[MAX_IDX_HEADS];
    for (int h = 0; h < n_idx_heads; h++) {
        local_w[h] = static_cast<float>(idx_weights[h]);
    }

    // ── Running top-k: parallel arrays of (score, index) ─────────────
    // Initialise with -inf scores so any real score replaces them.
    // We use a min-heap-style approach: track the minimum score in top-k
    // and only do insertion sort when a new score exceeds it.
    float  topk_scores[MAX_TOP_K];
    int    topk_indices[MAX_TOP_K];
    float  min_score = -FLT_MAX;
    int    min_pos   = 0;
    int    filled    = 0;

    for (int i = 0; i < eff_k; i++) {
        topk_scores[i]  = -FLT_MAX;
        topk_indices[i] = -1;
    }

    // ── Iterate over all keys in tiles ────────────────────────────────
    for (int k_start = 0; k_start < N; k_start += TILE_K) {
        const int k_end = (k_start + TILE_K < N) ? (k_start + TILE_K) : N;

        for (int j = k_start; j < k_end; j++) {
            // Compute index score I(q_idx, j)
            float score = 0.0f;
            for (int h = 0; h < n_idx_heads; h++) {
                float prod = local_idx_q[h]
                           * static_cast<float>(idx_k[j * n_idx_heads + h]);
                float relu = (prod > 0.0f) ? prod : 0.0f;
                score += local_w[h] * relu;
            }

            // ── Insert into running top-k if score beats current min ──
            if (filled < eff_k) {
                // Still filling: just append
                topk_scores[filled]  = score;
                topk_indices[filled] = j;
                filled++;
                // Recompute min after filling
                if (filled == eff_k) {
                    min_score = topk_scores[0];
                    min_pos   = 0;
                    for (int t = 1; t < eff_k; t++) {
                        if (topk_scores[t] < min_score) {
                            min_score = topk_scores[t];
                            min_pos   = t;
                        }
                    }
                }
            } else if (score > min_score) {
                // Replace the current minimum
                topk_scores[min_pos]  = score;
                topk_indices[min_pos] = j;
                // Find new minimum
                min_score = topk_scores[0];
                min_pos   = 0;
                for (int t = 1; t < eff_k; t++) {
                    if (topk_scores[t] < min_score) {
                        min_score = topk_scores[t];
                        min_pos   = t;
                    }
                }
            }
        }
    }

    // ── Write selected indices ────────────────────────────────────────
    for (int t = 0; t < eff_k; t++) {
        selected_indices[q_idx * top_k + t] = topk_indices[t];
    }
    // Pad remaining slots with -1 if eff_k < top_k
    for (int t = eff_k; t < top_k; t++) {
        selected_indices[q_idx * top_k + t] = -1;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 3: Sparse Attention + Output Projection + Skip Connection
// ═══════════════════════════════════════════════════════════════════════════
//
// Each thread processes one query element i:
//   1. Load Q[i] from global memory
//   2. For each of the top_k selected indices j:
//        load K[j], V[j], compute q . k / sqrt(d_head)
//   3. Softmax over k attention logits (float accumulation)
//   4. Weighted sum of V -> correction vector [d_head]
//   5. Output projection: scalar_correction = W_out @ correction + b_out
//   6. smart_grad[i] = grad[i] + rescale * scalar_correction

template <typename scalar_t>
__global__ void dsa_sparse_attention_kernel(
    const scalar_t* __restrict__ grad,              // [N]
    const scalar_t* __restrict__ Q,                 // [N, d_head]
    const scalar_t* __restrict__ K,                 // [N, d_head]
    const scalar_t* __restrict__ V,                 // [N, d_head]
    const int*      __restrict__ selected_indices,  // [N, top_k]
    scalar_t* __restrict__ smart_grad,              // [N] — output
    const scalar_t* __restrict__ W_out,             // [d_head]
    const scalar_t* __restrict__ b_out,             // [1]
    const scalar_t rescale,
    const int N,
    const int d_head,
    const int top_k
) {
    // Shared memory for W_out and b_out
    extern __shared__ char smem_raw[];
    scalar_t* smem   = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sW_out = smem;           // [d_head]
    scalar_t* sb_out = sW_out + d_head; // [1]

    const int tid = threadIdx.x;
    for (int i = tid; i < d_head; i += blockDim.x)
        sW_out[i] = W_out[i];
    if (tid == 0)
        sb_out[0] = b_out[0];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    const int eff_k = (top_k < N) ? top_k : N;

    // ── Load Q[idx] into registers ────────────────────────────────────
    float q_reg[MAX_D_HEAD];
    for (int h = 0; h < d_head; h++) {
        q_reg[h] = static_cast<float>(Q[idx * d_head + h]);
    }

    const float inv_sqrt_d = rsqrtf(static_cast<float>(d_head));

    // ── Pass 1: compute attention logits and find max (for stable softmax)
    float max_logit = -FLT_MAX;
    int   valid_count = 0;

    // We'll need logits again in pass 2, but top_k can be up to 512.
    // Store them in a local array (registers / local memory).
    float logits[MAX_TOP_K];

    for (int t = 0; t < eff_k; t++) {
        const int j = selected_indices[idx * top_k + t];
        if (j < 0 || j >= N) {
            logits[t] = -FLT_MAX;
            continue;
        }
        float dot = 0.0f;
        for (int h = 0; h < d_head; h++) {
            dot += q_reg[h] * static_cast<float>(K[j * d_head + h]);
        }
        logits[t] = dot * inv_sqrt_d;
        if (logits[t] > max_logit) max_logit = logits[t];
        valid_count++;
    }

    // Handle edge case: no valid keys
    if (valid_count == 0) {
        smart_grad[idx] = grad[idx];
        return;
    }

    // ── Pass 2: exp(logit - max) and sum ──────────────────────────────
    float sum_exp = 0.0f;
    float weights[MAX_TOP_K];
    for (int t = 0; t < eff_k; t++) {
        if (logits[t] <= -FLT_MAX * 0.5f) {
            weights[t] = 0.0f;
        } else {
            weights[t] = expf(logits[t] - max_logit);
            sum_exp += weights[t];
        }
    }

    // Normalise
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // ── Pass 3: weighted sum of V values → correction[d_head] ─────────
    float correction[MAX_D_HEAD];
    for (int h = 0; h < d_head; h++)
        correction[h] = 0.0f;

    for (int t = 0; t < eff_k; t++) {
        const float w = weights[t] * inv_sum;
        if (w == 0.0f) continue;
        const int j = selected_indices[idx * top_k + t];
        if (j < 0 || j >= N) continue;
        for (int h = 0; h < d_head; h++) {
            correction[h] += w * static_cast<float>(V[j * d_head + h]);
        }
    }

    // ── Output projection: scalar = W_out . correction + b_out ────────
    float proj = static_cast<float>(sb_out[0]);
    for (int h = 0; h < d_head; h++) {
        proj += static_cast<float>(sW_out[h]) * correction[h];
    }

    // ── Skip connection ───────────────────────────────────────────────
    smart_grad[idx] = static_cast<scalar_t>(
        static_cast<float>(grad[idx]) + static_cast<float>(rescale) * proj
    );
}


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 4: Fused gating blend + Adam moments + progressive wd + step
//            (reused from SuperGrok v1.5)
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void fused_adam_decay_kernel(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const scalar_t* __restrict__ smart_grad,
    const scalar_t* __restrict__ mu,
    const scalar_t lamb_eff,
    const scalar_t beta1,
    const scalar_t beta2,
    const scalar_t lr,
    const scalar_t wd_eff,
    const scalar_t eps,
    const scalar_t bc1,
    const scalar_t bc2,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Final gradient = smart_grad + lambda * mu
    const scalar_t fg = smart_grad[idx] + lamb_eff * mu[idx];

    // Adam moment updates
    const scalar_t ea = beta1 * exp_avg[idx]
                      + (static_cast<scalar_t>(1) - beta1) * fg;
    const scalar_t easq = beta2 * exp_avg_sq[idx]
                        + (static_cast<scalar_t>(1) - beta2) * fg * fg;
    exp_avg[idx]    = ea;
    exp_avg_sq[idx] = easq;

    // Bias-corrected step
    const scalar_t step_size = lr / bc1;
    const scalar_t denom = sqrtf(static_cast<float>(easq / bc2)) + eps;

    // Progressive weight decay + Adam step (fused)
    scalar_t p = param[idx];
    p *= (static_cast<scalar_t>(1) - lr * wd_eff);
    p -= step_size * ea / denom;
    param[idx] = p;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 5: SAM parameter perturbation (reused from SuperGrok v1.5)
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sam_perturb_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    const scalar_t rho_over_norm,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    param[idx] += rho_over_norm * grad[idx];
}


// ═══════════════════════════════════════════════════════════════════════════
//  Kernel 6: Compute sharpness + restore parameters (reused from v1.5)
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sharpness_restore_kernel(
    scalar_t* __restrict__ param,
    scalar_t* __restrict__ sharpness,
    const scalar_t* __restrict__ backup,
    const scalar_t* __restrict__ sam_grad,
    const scalar_t* __restrict__ normal_grad,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    sharpness[idx] = static_cast<scalar_t>(
        fabsf(static_cast<float>(sam_grad[idx]) - static_cast<float>(normal_grad[idx]))
    );
    param[idx] = backup[idx];
}


// ═══════════════════════════════════════════════════════════════════════════
//  C++ Dispatch Functions
// ═══════════════════════════════════════════════════════════════════════════

void launch_dsa_project(
    torch::Tensor grad,
    torch::Tensor sharpness,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor idx_q,
    torch::Tensor idx_k,
    torch::Tensor W_q,
    torch::Tensor b_q,
    torch::Tensor W_k,
    torch::Tensor b_k,
    torch::Tensor W_v,
    torch::Tensor b_v,
    torch::Tensor W_iq,
    torch::Tensor W_ik,
    int d_head,
    int n_idx_heads
) {
    const int N = grad.numel();
    if (N == 0) return;

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // smem: 3*(d_head*2 + d_head) + 2*(n_idx_heads*2)  = 9*d_head + 4*n_idx_heads
    const int smem_elems = 9 * d_head + 4 * n_idx_heads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "dsa_project", ([&] {
            const int smem_bytes = smem_elems * sizeof(scalar_t);
            dsa_project_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
                grad.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                idx_q.data_ptr<scalar_t>(),
                idx_k.data_ptr<scalar_t>(),
                W_q.data_ptr<scalar_t>(),
                b_q.data_ptr<scalar_t>(),
                W_k.data_ptr<scalar_t>(),
                b_k.data_ptr<scalar_t>(),
                W_v.data_ptr<scalar_t>(),
                b_v.data_ptr<scalar_t>(),
                W_iq.data_ptr<scalar_t>(),
                W_ik.data_ptr<scalar_t>(),
                N,
                d_head,
                n_idx_heads
            );
        })
    );
}


void launch_dsa_indexer_topk(
    torch::Tensor idx_q,
    torch::Tensor idx_k,
    torch::Tensor idx_weights,
    torch::Tensor selected_indices,
    int n_idx_heads,
    int top_k
) {
    const int N = idx_q.size(0);
    if (N == 0) return;

    // One thread per query, TILE_Q threads per block
    const int grid = (N + TILE_Q - 1) / TILE_Q;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        idx_q.scalar_type(), "dsa_indexer_topk", ([&] {
            dsa_indexer_topk_kernel<scalar_t><<<grid, TILE_Q>>>(
                idx_q.data_ptr<scalar_t>(),
                idx_k.data_ptr<scalar_t>(),
                idx_weights.data_ptr<scalar_t>(),
                selected_indices.data_ptr<int>(),
                N,
                n_idx_heads,
                top_k
            );
        })
    );
}


void launch_dsa_sparse_attention(
    torch::Tensor grad,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor selected_indices,
    torch::Tensor smart_grad,
    torch::Tensor W_out,
    torch::Tensor b_out,
    float rescale,
    int d_head,
    int top_k
) {
    const int N = grad.numel();
    if (N == 0) return;

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // smem: W_out[d_head] + b_out[1]

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "dsa_sparse_attention", ([&] {
            const int smem_bytes = (d_head + 1) * sizeof(scalar_t);
            dsa_sparse_attention_kernel<scalar_t><<<grid, BLOCK_SIZE, smem_bytes>>>(
                grad.data_ptr<scalar_t>(),
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                selected_indices.data_ptr<int>(),
                smart_grad.data_ptr<scalar_t>(),
                W_out.data_ptr<scalar_t>(),
                b_out.data_ptr<scalar_t>(),
                static_cast<scalar_t>(rescale),
                N,
                d_head,
                top_k
            );
        })
    );
}


void launch_fused_adam_decay(
    torch::Tensor param,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad,
    torch::Tensor mu,
    float lamb_eff,
    float beta1,
    float beta2,
    float lr,
    float wd_eff,
    float eps,
    float bc1,
    float bc2
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "fused_adam_decay", ([&] {
            fused_adam_decay_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                exp_avg.data_ptr<scalar_t>(),
                exp_avg_sq.data_ptr<scalar_t>(),
                smart_grad.data_ptr<scalar_t>(),
                mu.data_ptr<scalar_t>(),
                static_cast<scalar_t>(lamb_eff),
                static_cast<scalar_t>(beta1),
                static_cast<scalar_t>(beta2),
                static_cast<scalar_t>(lr),
                static_cast<scalar_t>(wd_eff),
                static_cast<scalar_t>(eps),
                static_cast<scalar_t>(bc1),
                static_cast<scalar_t>(bc2),
                N
            );
        })
    );
}


void launch_sam_perturb(
    torch::Tensor param,
    torch::Tensor grad,
    float rho_over_norm
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sam_perturb", ([&] {
            sam_perturb_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                grad.data_ptr<scalar_t>(),
                static_cast<scalar_t>(rho_over_norm),
                N
            );
        })
    );
}


void launch_sharpness_restore(
    torch::Tensor param,
    torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad,
    torch::Tensor normal_grad
) {
    const int N = param.numel();
    if (N == 0) return;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        param.scalar_type(), "sharpness_restore", ([&] {
            sharpness_restore_kernel<scalar_t><<<grid, BLOCK_SIZE>>>(
                param.data_ptr<scalar_t>(),
                sharpness.data_ptr<scalar_t>(),
                backup.data_ptr<scalar_t>(),
                sam_grad.data_ptr<scalar_t>(),
                normal_grad.data_ptr<scalar_t>(),
                N
            );
        })
    );
}
