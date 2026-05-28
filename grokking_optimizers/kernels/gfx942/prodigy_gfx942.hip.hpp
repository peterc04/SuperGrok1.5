#ifndef GROKKING_PRODIGY_GFX942_HIP_HPP_
#define GROKKING_PRODIGY_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

struct ProdigyState {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    float* __restrict__ s;
    const float* __restrict__ param_init;

    static constexpr int num_state_tensors() { return 4; }
    static constexpr int state_bytes_per_element() { return 4 * sizeof(float); }
};

// ---------------------------------------------------------------------------
// Wavefront-level reduction helper (sum) -- 64-wide wavefront on gfx942
// ---------------------------------------------------------------------------
__forceinline__ __device__
float prodigy_wavefront_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction helper (sum).  Uses LDS (shared memory) across
// wavefronts.  Returns the result in thread 0 of the block.
// ---------------------------------------------------------------------------
__forceinline__ __device__
float prodigy_block_reduce_sum(float val) {
    __shared__ float wavefront_sums[16]; // max 16 wavefronts per block (1024 / 64)

    const int lane = threadIdx.x & 63;
    const int wavefront_id = threadIdx.x >> 6;

    val = prodigy_wavefront_reduce_sum(val);

    if (lane == 0) {
        wavefront_sums[wavefront_id] = val;
    }
    __syncthreads();

    const int num_wavefronts = (blockDim.x + 63) >> 6;
    val = (threadIdx.x < num_wavefronts) ? wavefront_sums[threadIdx.x] : 0.0f;
    if (wavefront_id == 0) {
        val = prodigy_wavefront_reduce_sum(val);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Pass 1: Reduction kernel
//
// Computes two global reductions and updates the gradient sum state s:
//   1. s_new[i] = beta1 * s[i] + (1 - beta1) * grad[i]
//   2. d_numerator   = sum_i  grad[i] * (param[i] - param_init[i])
//   3. d_denominator = sum_i  |s_new[i]|
//
// Global accumulators (device pointers, zero-initialised before launch):
//   global_d_numerator[0], global_d_denominator[0]
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void prodigy_reduce(
    const ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    ProdigyState state,
    int64_t n,
    float beta1,
    float* __restrict__ global_d_numerator,
    float* __restrict__ global_d_denominator,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local_numer = 0.0f;
    float local_denom = 0.0f;

    for (int64_t i = idx; i < n; i += stride) {
        // gfx942: non-temporal load for streaming grad reads
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (isnan(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (isnan(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        ParamT p_raw = __builtin_nontemporal_load(params + i);
        float p_f = to_float(p_raw);
        float p_init = state.param_init[i];
        float s_old = state.s[i];

        // Update gradient sum
        float s_new = beta1 * s_old + (1.0f - beta1) * g;
        state.s[i] = s_new;

        // Accumulate for distance estimation
        local_numer += g * (p_f - p_init);
        local_denom += fabsf(s_new);
    }

    // Block-level reduction via 64-wide wavefront shuffles + LDS
    float block_numer = prodigy_block_reduce_sum(local_numer);
    float block_denom = prodigy_block_reduce_sum(local_denom);

    if (threadIdx.x == 0) {
        atomicAdd(global_d_numerator, block_numer);
        atomicAdd(global_d_denominator, block_denom);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: AdamW update kernel with self-tuned learning rate
//
// Reads globally-reduced d_lr, then applies a standard AdamW step with
// effective_lr = d_lr * lr.
//
// The caller is expected to:
//   1. Launch prodigy_reduce_kernel (pass 1)
//   2. Launch prodigy_resolve_d_lr_kernel (1 thread) to compute
//      d_lr_new = max(d_lr_old, d_numerator / (d_denominator + eps))
//   3. Launch prodigy_update_kernel (pass 2) with the resolved d_lr
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void prodigy_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    ProdigyState state,
    int64_t n,
    float lr,
    float d_lr,                 // effective distance-aware multiplier
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const float effective_lr = d_lr * lr;

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // gfx942: non-temporal load for streaming grad reads
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (isnan(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (isnan(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p_f = to_float(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = beta1 * m_old + (1.0f - beta1) * g;
        float v = beta2 * v_old + (1.0f - beta2) * g * g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;

        p_f -= effective_lr * update;

        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
        params[i] = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: vec4 path for float params
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void prodigy_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    ProdigyState state,
    int64_t n,
    float lr,
    float d_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const float effective_lr = d_lr * lr;

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
        float4 g_vec = __builtin_nontemporal_load(g4 + i);
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
            ps[k] -= effective_lr * (m_hat / denom + weight_decay * ps[k]);
            ms[k] = m;
            vs[k] = v;
        }

        p4[i] = make_float4(ps[0], ps[1], ps[2], ps[3]);
        m4[i] = make_float4(ms[0], ms[1], ms[2], ms[3]);
        v4[i] = make_float4(vs[0], vs[1], vs[2], vs[3]);
    }
}

// ---------------------------------------------------------------------------
// Launcher kernels
// ---------------------------------------------------------------------------

// Pass 1: compute reductions and update gradient sum s
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void prodigy_reduce_kernel(
    const ParamT* params, const ParamT* grads, ProdigyState state, int64_t n,
    float beta1,
    float* global_d_numerator, float* global_d_denominator,
    float clip_threshold
) {
    prodigy_reduce<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, beta1,
        global_d_numerator, global_d_denominator, clip_threshold);
}

// Tiny helper: resolve d_lr from the reduction results.
// Launch with 1 thread, 1 block.
__global__ void prodigy_resolve_d_lr_kernel(
    const float* __restrict__ global_d_numerator,
    const float* __restrict__ global_d_denominator,
    float* __restrict__ d_lr_ptr,       // read old, write new
    float eps
) {
    float numer = *global_d_numerator;
    float denom = *global_d_denominator;
    float d_lr_old = *d_lr_ptr;
    float d_lr_new = fmaxf(d_lr_old, numer / (denom + eps));
    *d_lr_ptr = d_lr_new;
}

// Pass 2: AdamW step with self-tuned learning rate
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void prodigy_update_kernel(
    ParamT* params, const ParamT* grads, ProdigyState state, int64_t n,
    float lr, float d_lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, float clip_threshold
) {
    prodigy_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, d_lr, beta1, beta2, eps, wd,
        bc1, bc2, clip_threshold);
}

}} // namespace grokking::gfx942

#endif // GROKKING_PRODIGY_GFX942_HIP_HPP_
