#ifndef GROKKING_LION_GFX942_HIP_HPP_
#define GROKKING_LION_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking {
namespace gfx942 {

struct LionState {
    float* __restrict__ exp_avg;
    static constexpr int num_state_tensors() { return 1; }
    static constexpr int state_bytes_per_element() { return sizeof(float); }
};

// -- Scalar Lion update --

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void lion_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    LionState state,
    int64_t n,
    float lr, float beta1, float beta2, float weight_decay,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // Streaming non-temporal grad read.
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);
        g = apply_nan_policy<NAN_POLICY>(g);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        float p_f = to_float(params[i]);
        float m_old = state.exp_avg[i];

        // Lion interpolation and sign.
        float interp = beta1 * m_old + (1.0f - beta1) * g;
        float sign_val = copysignf(1.0f, interp);

        // Parameter update.
        p_f -= lr * (sign_val + weight_decay * p_f);

        // Momentum update (after param update).
        float m_new = beta2 * m_old + (1.0f - beta2) * g;

        params[i] = from_float<ParamT>(p_f);
        state.exp_avg[i] = m_new;
    }
}

// -- Vectorized float4 Lion update (float params only) --

template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void lion_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    LionState state,
    int64_t n,
    float lr, float beta1, float beta2, float weight_decay,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t n4 = n / 4;

    float4* params4 = reinterpret_cast<float4*>(params);
    const float4* grads4 = reinterpret_cast<const float4*>(grads);
    float4* m4 = reinterpret_cast<float4*>(state.exp_avg);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 g4 = __builtin_nontemporal_load(grads4 + i);
        float4 p4 = params4[i];
        float4 mo4 = m4[i];

        float g_arr[4] = {g4.x, g4.y, g4.z, g4.w};
        float p_arr[4] = {p4.x, p4.y, p4.z, p4.w};
        float m_arr[4] = {mo4.x, mo4.y, mo4.z, mo4.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = apply_nan_policy<NAN_POLICY>(g_arr[k]);
            g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

            float interp = beta1 * m_arr[k] + (1.0f - beta1) * g;
            float sign_val = copysignf(1.0f, interp);

            p_arr[k] -= lr * (sign_val + weight_decay * p_arr[k]);
            m_arr[k] = beta2 * m_arr[k] + (1.0f - beta2) * g;
        }

        params4[i] = make_float4(p_arr[0], p_arr[1], p_arr[2], p_arr[3]);
        m4[i] = make_float4(m_arr[0], m_arr[1], m_arr[2], m_arr[3]);
    }

    // Handle tail elements.
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        float g_raw = __builtin_nontemporal_load(grads + i);
        float g = apply_nan_policy<NAN_POLICY>(g_raw);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        float p_f = params[i];
        float m_old = state.exp_avg[i];

        float interp = beta1 * m_old + (1.0f - beta1) * g;
        float sign_val = copysignf(1.0f, interp);

        p_f -= lr * (sign_val + weight_decay * p_f);
        float m_new = beta2 * m_old + (1.0f - beta2) * g;

        params[i] = p_f;
        state.exp_avg[i] = m_new;
    }
}

// -- Launcher kernel --

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void lion_kernel(
    ParamT* params, const ParamT* grads, LionState state, int64_t n,
    float lr, float beta1, float beta2, float wd, float clip_threshold
) {
    lion_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, wd, clip_threshold);
}

}  // namespace gfx942
}  // namespace grokking

#endif  // GROKKING_LION_GFX942_HIP_HPP_
