#ifndef GROKKING_LOOKSAM_GFX942_HIP_HPP_
#define GROKKING_LOOKSAM_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

struct LookSAMState {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    float* __restrict__ sam_direction;

    static constexpr int num_state_tensors() { return 3; }
    static constexpr int state_bytes_per_element() { return 3 * sizeof(float); }
};

// ---------------------------------------------------------------------------
// Phase 1: Standard AdamW step (scalar path)
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void looksam_adamw_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    LookSAMState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // gfx942 has no __ldg; non-temporal load bypasses L2 for streaming grad data
        ParamT g_raw = __builtin_nontemporal_load(&grads[i]);
        float g = to_float<ParamT>(g_raw);

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (isnan(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (isnan(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p_f = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = beta1 * m_old + (1.0f - beta1) * g;
        float v = beta2 * v_old + (1.0f - beta2) * g * g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;

        p_f -= lr * update;

        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
        params[i] = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Standard AdamW step (vec4 path for float params)
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void looksam_adamw_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    LookSAMState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int64_t n4 = n / 4;
    float4* __restrict__ p4 = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(grads);
    float4* __restrict__ m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__ v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 p_vec = p4[i];
        // Non-temporal load for streaming grad access on gfx942
        float4 g_vec = __builtin_nontemporal_load(&g4[i]);
        float4 m_vec = m4[i];
        float4 v_vec = v4[i];

        float gs[4] = {g_vec.x, g_vec.y, g_vec.z, g_vec.w};
        float ps[4] = {p_vec.x, p_vec.y, p_vec.z, p_vec.w};
        float ms[4] = {m_vec.x, m_vec.y, m_vec.z, m_vec.w};
        float vs[4] = {v_vec.x, v_vec.y, v_vec.z, v_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (isnan(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (isnan(g)) { continue; }
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            float m = beta1 * ms[k] + (1.0f - beta1) * g;
            float v = beta2 * vs[k] + (1.0f - beta2) * g * g;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            float denom = sqrtf(v_hat) + eps;
            ps[k] -= lr * (m_hat / denom + weight_decay * ps[k]);
            ms[k] = m;
            vs[k] = v;
        }

        p4[i] = make_float4(ps[0], ps[1], ps[2], ps[3]);
        m4[i] = make_float4(ms[0], ms[1], ms[2], ms[3]);
        v4[i] = make_float4(vs[0], vs[1], vs[2], vs[3]);
    }
}

// ---------------------------------------------------------------------------
// Wavefront-level reduction helper (sum) -- 64-wide wavefront on gfx942
// Uses __shfl_xor for butterfly reduction across all 64 lanes.
// ---------------------------------------------------------------------------
__forceinline__ __device__
float looksam_wavefront_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction helper (sum).  Uses LDS (shared memory) across
// wavefronts.  Returns the result broadcast to all threads in the block.
// ---------------------------------------------------------------------------
__forceinline__ __device__
float looksam_block_reduce_sum(float val) {
    __shared__ float wavefront_sums[16]; // max 16 wavefronts per block (1024 / 64)

    const int lane = threadIdx.x & 63;
    const int wavefront_id = threadIdx.x >> 6;

    // Reduce within wavefront (64-wide)
    val = looksam_wavefront_reduce_sum(val);

    if (lane == 0) {
        wavefront_sums[wavefront_id] = val;
    }
    __syncthreads();

    // First wavefront reduces the per-wavefront sums
    const int num_wavefronts = (blockDim.x + 63) >> 6;
    val = (threadIdx.x < num_wavefronts) ? wavefront_sums[threadIdx.x] : 0.0f;
    if (wavefront_id == 0) {
        val = looksam_wavefront_reduce_sum(val);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Phase 2: SAM direction update (scalar only -- involves global reductions)
//
// Two global accumulators are provided by the caller (device pointers,
// zero-initialised before launch):
//   global_dot[0]     = sum_i  orig_grad[i] * direction[i]
//   global_norm_sq[0] = sum_i  direction[i] * direction[i]
//
// Pass 1 (looksam_direction_update_pass1):
//   1. direction[i] = alpha * direction[i] + (1-alpha) * (perturbed[i] - orig[i])
//   2. Local accumulation of dot(orig, direction) and |direction|^2
//   3. Block reduce + atomicAdd to global accumulators
//
// Pass 2 (looksam_direction_update_pass2):
//   4. adjusted_grad[i] = orig[i] - (dot / (norm_sq + eps)) * direction[i]
//   5. Write adjusted_grad back to orig_grads (in-place)
// ---------------------------------------------------------------------------
__forceinline__ __device__
void looksam_direction_update_pass1(
    const float* __restrict__ orig_grads,
    const float* __restrict__ perturbed_grads,
    float* __restrict__ sam_directions,
    float* __restrict__ global_dot,
    float* __restrict__ global_norm_sq,
    float alpha,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local_dot = 0.0f;
    float local_norm_sq = 0.0f;

    for (int64_t i = idx; i < n; i += stride) {
        float orig = __builtin_nontemporal_load(&orig_grads[i]);
        float pert = __builtin_nontemporal_load(&perturbed_grads[i]);
        float dir_old = sam_directions[i];

        float dir_new = alpha * dir_old + (1.0f - alpha) * (pert - orig);

        sam_directions[i] = dir_new;

        local_dot += orig * dir_new;
        local_norm_sq += dir_new * dir_new;
    }

    // Block-level reduction via 64-wide wavefront shuffles + LDS
    float block_dot = looksam_block_reduce_sum(local_dot);
    float block_norm_sq = looksam_block_reduce_sum(local_norm_sq);

    // Thread 0 of each block contributes to global accumulators
    if (threadIdx.x == 0) {
        atomicAdd(global_dot, block_dot);
        atomicAdd(global_norm_sq, block_norm_sq);
    }
}

__forceinline__ __device__
void looksam_direction_update_pass2(
    float* __restrict__ orig_grads,
    const float* __restrict__ sam_directions,
    const float* __restrict__ global_dot,
    const float* __restrict__ global_norm_sq,
    float eps,
    int64_t n
) {
    // Read the globally-reduced scalars (broadcast to all threads)
    float dot_val = *global_dot;
    float norm_sq_val = *global_norm_sq;
    float projection_scale = dot_val / (norm_sq_val + eps);

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float orig = orig_grads[i];
        float dir = sam_directions[i];

        float adjusted = orig - projection_scale * dir;

        orig_grads[i] = adjusted;
    }
}

// ---------------------------------------------------------------------------
// Launcher kernels
// ---------------------------------------------------------------------------

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void looksam_adamw_kernel(
    ParamT* params, const ParamT* grads, LookSAMState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, float clip_threshold
) {
    looksam_adamw_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd, bc1, bc2, clip_threshold);
}

__global__ void looksam_direction_pass1_kernel(
    const float* orig_grads,
    const float* perturbed_grads,
    float* sam_directions,
    float* global_dot,
    float* global_norm_sq,
    float alpha,
    int64_t n
) {
    looksam_direction_update_pass1(
        orig_grads, perturbed_grads, sam_directions,
        global_dot, global_norm_sq, alpha, n);
}

__global__ void looksam_direction_pass2_kernel(
    float* orig_grads,
    const float* sam_directions,
    const float* global_dot,
    const float* global_norm_sq,
    float eps,
    int64_t n
) {
    looksam_direction_update_pass2(
        orig_grads, sam_directions, global_dot, global_norm_sq, eps, n);
}

}} // namespace grokking::gfx942

#endif // GROKKING_LOOKSAM_GFX942_HIP_HPP_
