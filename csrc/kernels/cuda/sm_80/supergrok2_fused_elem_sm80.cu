/*
 * SuperGrok v2 — Ampere-Optimized Fused Element Step (sm_80+)
 *
 * Ampere optimization for the fused per-element GRU + PEER + Expert + Adam kernel:
 *   - cp.async for asynchronous cooperative loading of expert weights into shared
 *     memory, overlapping with initial register setup and GRU weight loading.
 *   - 192KB shared memory allows caching ALL weights (out_proj + GRU + expert)
 *     simultaneously without spilling.
 *
 * The kernel math is IDENTICAL to the generic fused_elem_step_kernel.
 * Only the shared memory loading path changes to use cp.async.
 *
 * Usage: The Ampere batched launcher can optionally call this kernel instead
 * of the generic fused_elem_step_kernel when expert weights are large enough
 * to benefit from async prefetch. Currently wired via the batched step launcher.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "platform.h"
#include "types.h"

#if GROK_CUDA
#include <cuda_pipeline.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  Ampere Fused Per-Element Step with cp.async Weight Prefetch
//
//  Two-phase weight loading:
//    Phase 1: cp.async prefetch out_proj + GRU weights (needed first for
//             GRU computation)
//    Phase 2: cp.async prefetch expert weights (needed later for PEER
//             routing + expert MLP). Overlapped with GRU compute.
//
//  This hides ~400 cycles of global memory latency for expert weights
//  behind the GRU computation (~264 FMAs ≈ ~300 cycles on A100).
// ═══════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__launch_bounds__(256, 8)
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
    const int expert_hidden, const int num_experts
) {
    const int gru_input_dim = 2 + 2 * d_model;
    const int gru_row_len = gru_input_dim + gru_hidden;
    const int op_size = d_model * d_inner;
    const int gru_mat_size = gru_hidden * gru_row_len;

    extern __shared__ float smem[];

    // Layout: out_proj (2 × op_size) → GRU weights → expert weights
    float* s_out_fwd = smem;
    float* s_out_bwd = smem + op_size;
    float* s_gru_Wz = s_out_bwd + op_size;
    float* s_gru_Wr = s_gru_Wz + gru_mat_size;
    float* s_gru_Wh = s_gru_Wr + gru_mat_size;
    float* s_gru_bz = s_gru_Wh + gru_mat_size;
    float* s_gru_br = s_gru_bz + gru_hidden;
    float* s_gru_bh = s_gru_br + gru_hidden;
    float* s_expert_W1 = s_gru_bh + gru_hidden;
    float* s_expert_b1 = s_expert_W1 + num_experts * expert_hidden;
    float* s_expert_W2 = s_expert_b1 + num_experts * expert_hidden;
    float* s_expert_b2 = s_expert_W2 + num_experts * expert_hidden;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // ── Phase 1: cp.async prefetch out_proj + GRU weights ─────────────
    // These are needed immediately for the out_proj and GRU computation.
    #pragma unroll 4
    for (int i = tid; i < 2 * op_size; i += block_size) {
        const float* src = (i < op_size) ? &out_proj_fwd_W[i] : &out_proj_bwd_W[i - op_size];
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(&smem[i], src, sizeof(float));
#else
        smem[i] = *src;
#endif
    }

    // GRU weights
    const float* gru_gmem[] = {gru_Wz, gru_Wr, gru_Wh, gru_bz, gru_br, gru_bh};
    int gru_sizes[] = {gru_mat_size, gru_mat_size, gru_mat_size, gru_hidden, gru_hidden, gru_hidden};
    float* gru_dst = s_gru_Wz;
    int gru_offset = 0;
    #pragma unroll
    for (int seg = 0; seg < 6; seg++) {
        #pragma unroll 4
        for (int i = tid; i < gru_sizes[seg]; i += block_size) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            __pipeline_memcpy_async(&gru_dst[gru_offset + i], &gru_gmem[seg][i], sizeof(float));
#else
            gru_dst[gru_offset + i] = gru_gmem[seg][i];
#endif
        }
        gru_offset += gru_sizes[seg];
    }

#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
#endif

    // ── Phase 2: cp.async prefetch expert weights (overlapped with GRU) ─
    #pragma unroll 4
    for (int i = tid; i < num_experts * expert_hidden; i += block_size) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(&s_expert_W1[i], &expert_W1[i], sizeof(float));
        __pipeline_memcpy_async(&s_expert_b1[i], &expert_b1[i], sizeof(float));
        __pipeline_memcpy_async(&s_expert_W2[i], &expert_W2[i], sizeof(float));
#else
        s_expert_W1[i] = expert_W1[i];
        s_expert_b1[i] = expert_b1[i];
        s_expert_W2[i] = expert_W2[i];
#endif
    }
    #pragma unroll 4
    for (int i = tid; i < num_experts; i += block_size) {
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(&s_expert_b2[i], &expert_b2[i], sizeof(float));
#else
        s_expert_b2[i] = expert_b2[i];
#endif
    }

#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
    // Wait for Phase 1 (out_proj + GRU). Phase 2 may still be in flight.
    __pipeline_wait_prior(1);
#endif
    __syncthreads();

    // ── Per-element processing ────────────────────────────────────────
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return;

    float g = static_cast<float>(grad[idx]);
    float s = static_cast<float>(sharpness[idx]);
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(s)) s = 0.0f;
    const int half_d = d_model / 2;
    const int peer_input_dim = gru_hidden + 2 * d_model + 2;

    // 1. Out_proj: fwd_ctx, bwd_ctx from scan outputs (using shared mem weights)
    float fwd_scan[MAX_D_INNER], bwd_scan[MAX_D_INNER];
    #pragma unroll 4
    for (int j = 0; j < d_inner; j += 4) {
        float4 fwd4 = *reinterpret_cast<const float4*>(&fwd_scan_out[idx * d_inner + j]);
        float4 bwd4 = *reinterpret_cast<const float4*>(&bwd_scan_out[idx * d_inner + j]);
        fwd_scan[j] = fwd4.x; fwd_scan[j+1] = fwd4.y; fwd_scan[j+2] = fwd4.z; fwd_scan[j+3] = fwd4.w;
        bwd_scan[j] = bwd4.x; bwd_scan[j+1] = bwd4.y; bwd_scan[j+2] = bwd4.z; bwd_scan[j+3] = bwd4.w;
    }

    float fwd_ctx[MAX_D_MODEL], bwd_ctx[MAX_D_MODEL];
    #pragma unroll 4
    for (int d = 0; d < d_model; d++) {
        float fv = 0.0f, bv = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < d_inner; j++) {
            fv += s_out_fwd[d * d_inner + j] * fwd_scan[j];
            bv += s_out_bwd[d * d_inner + j] * bwd_scan[j];
        }
        fwd_ctx[d] = fv;
        bwd_ctx[d] = bv;
    }

    // 2. GRU update
    float h_old[MAX_GRU_HIDDEN];
    #pragma unroll 4
    for (int j = 0; j < gru_hidden; j++)
        h_old[j] = gru_state[idx * gru_hidden + j];

    float h_new[MAX_GRU_HIDDEN];
    #pragma unroll 4
    for (int j = 0; j < gru_hidden; j++) {
        float val_z = s_gru_bz[j], val_r = s_gru_br[j];
        val_z += s_gru_Wz[j * gru_row_len + 0] * g + s_gru_Wz[j * gru_row_len + 1] * s;
        val_r += s_gru_Wr[j * gru_row_len + 0] * g + s_gru_Wr[j * gru_row_len + 1] * s;
        int offset = 2;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + d] * fwd_ctx[d];
            val_r += s_gru_Wr[j * gru_row_len + offset + d] * fwd_ctx[d];
        }
        offset += d_model;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + d] * bwd_ctx[d];
            val_r += s_gru_Wr[j * gru_row_len + offset + d] * bwd_ctx[d];
        }
        offset += d_model;
        #pragma unroll 4
        for (int k = 0; k < gru_hidden; k++) {
            val_z += s_gru_Wz[j * gru_row_len + offset + k] * h_old[k];
            val_r += s_gru_Wr[j * gru_row_len + offset + k] * h_old[k];
        }
        float z_gate = 1.0f / (1.0f + expf(-val_z));
        float r_gate = 1.0f / (1.0f + expf(-val_r));

        float val_h = s_gru_bh[j];
        val_h += s_gru_Wh[j * gru_row_len + 0] * g + s_gru_Wh[j * gru_row_len + 1] * s;
        offset = 2;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) val_h += s_gru_Wh[j * gru_row_len + offset + d] * fwd_ctx[d];
        offset += d_model;
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) val_h += s_gru_Wh[j * gru_row_len + offset + d] * bwd_ctx[d];
        offset += d_model;
        #pragma unroll 4
        for (int k = 0; k < gru_hidden; k++) val_h += s_gru_Wh[j * gru_row_len + offset + k] * (r_gate * h_old[k]);
        float h_tilde = tanhf(val_h);
        h_new[j] = (1.0f - z_gate) * h_old[j] + z_gate * h_tilde;
    }

    #pragma unroll 4
    for (int j = 0; j < gru_hidden; j++)
        gru_state[idx * gru_hidden + j] = h_new[j];

    // Wait for Phase 2 (expert weights) before PEER routing
#if GROK_CUDA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_wait_prior(0);
#endif
    __syncthreads();

    // 3. Multi-head PEER routing + expert evaluation
    float total_out = 0.0f;
    #pragma unroll 4
    for (int head = 0; head < num_heads; head++) {
        const float* pq_W = peer_query_Ws + head * d_model * peer_input_dim;
        float query[MAX_D_MODEL];
        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            float val = 0.0f;
            int off = 0;
            #pragma unroll 4
            for (int k = 0; k < gru_hidden; k++) val += pq_W[d * peer_input_dim + off + k] * h_new[k];
            off += gru_hidden;
            #pragma unroll 4
            for (int k = 0; k < d_model; k++) val += pq_W[d * peer_input_dim + off + k] * fwd_ctx[k];
            off += d_model;
            #pragma unroll 4
            for (int k = 0; k < d_model; k++) val += pq_W[d * peer_input_dim + off + k] * bwd_ctx[k];
            off += d_model;
            val += pq_W[d * peer_input_dim + off] * g + pq_W[d * peer_input_dim + off + 1] * s;
            query[d] = val;
        }

        const float* keys_A = prod_keys_A + head * pk_dim * half_d;
        const float* keys_B = prod_keys_B + head * pk_dim * half_d;

        int best_a = 0; float best_sa = -1e30f;
        #pragma unroll 4
        for (int k = 0; k < pk_dim; k++) {
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < half_d; d++) dot += query[d] * LDG(&keys_A[k * half_d + d]);
            if (dot > best_sa) { best_sa = dot; best_a = k; }
        }
        int best_b = 0; float best_sb = -1e30f;
        #pragma unroll 4
        for (int k = 0; k < pk_dim; k++) {
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < half_d; d++) dot += query[half_d + d] * LDG(&keys_B[k * half_d + d]);
            if (dot > best_sb) { best_sb = dot; best_b = k; }
        }

        int expert_idx = best_a * pk_dim + best_b;
        if (expert_idx >= num_experts) expert_idx = num_experts - 1;
        if (expert_counts) atomicAdd(&expert_counts[expert_idx], 1);

        // Expert MLP from shared memory
        float head_out = s_expert_b2[expert_idx];
        #pragma unroll 4
        for (int h = 0; h < expert_hidden; h++) {
            float z_val = s_expert_W1[expert_idx * expert_hidden + h] * g
                        + s_expert_b1[expert_idx * expert_hidden + h];
            z_val = fmaxf(z_val, 0.0f);
            head_out += s_expert_W2[expert_idx * expert_hidden + h] * z_val;
        }
        total_out += head_out;
    }

    float smart_grad = g + rescale * total_out / (float)num_heads;

    // 4-6: mu + Adam (same as generic)
    float mu_val = mu[idx];
    mu_val = alpha * mu_val + (1.0f - alpha) * g;
    mu[idx] = mu_val;

    float fg = smart_grad + lamb_eff * mu_val;

    float ea = exp_avg[idx], easq = exp_avg_sq[idx];
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

// Explicit instantiations
template __global__ void fused_elem_step_cpasync_kernel<float>(
    float*, const float*, const float*, float*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float, float, float, float, float, float, float, float, float, float,
    int*, int, int, int, int, int, int, int, int);

template __global__ void fused_elem_step_cpasync_kernel<at::Half>(
    at::Half*, const at::Half*, const at::Half*, float*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float, float, float, float, float, float, float, float, float, float,
    int*, int, int, int, int, int, int, int, int);

template __global__ void fused_elem_step_cpasync_kernel<at::BFloat16>(
    at::BFloat16*, const at::BFloat16*, const at::BFloat16*, float*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    float, float, float, float, float, float, float, float, float, float,
    int*, int, int, int, int, int, int, int, int);
