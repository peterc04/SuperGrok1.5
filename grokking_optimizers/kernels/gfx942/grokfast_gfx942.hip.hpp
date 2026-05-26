#ifndef GROKKING_GROKFAST_GFX942_HIP_HPP_
#define GROKKING_GROKFAST_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

struct GrokfastState {
    float* __restrict__ ema;
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    static constexpr int num_state_tensors() { return 3; }
    static constexpr int state_bytes_per_element() { return 3 * sizeof(float); }
};

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokfast_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokfastState state,
    int64_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb,
    float bias_correction1, float bias_correction2,
    float clip_threshold = 0.0f
) {
    const float one_minus_alpha = 1.0f - alpha;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * static_cast<int64_t>(blockDim.x)) {
        float g = to_float<ParamT>(
            __builtin_nontemporal_load(grads + i));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__builtin_isnan(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__builtin_isnan(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Phase 1: EMA filter + amplification
        float ema_old = state.ema[i];
        float ema_val = alpha * ema_old + one_minus_alpha * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: AdamW on amplified gradient
        float p_f = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = beta1 * m_old + one_minus_b1 * g_amp;
        float v = beta2 * v_old + one_minus_b2 * g_amp * g_amp;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;
        p_f -= lr * update;

        state.ema[i] = ema_val;
        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
        params[i] = from_float<ParamT>(p_f);
    }
}

// Vectorized path for float params, 4 elements per thread via float4 loads/stores.
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokfast_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    GrokfastState state,
    int64_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb,
    float bias_correction1, float bias_correction2,
    float clip_threshold = 0.0f
) {
    const float one_minus_alpha = 1.0f - alpha;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    const int64_t n4 = n / 4;
    float4* __restrict__ p4 = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(grads);
    float4* __restrict__ ema4 = reinterpret_cast<float4*>(state.ema);
    float4* __restrict__ m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__ v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * static_cast<int64_t>(blockDim.x)) {
        float4 p_vec = p4[i];
        float4 g_vec = __builtin_nontemporal_load(g4 + i);
        float4 ema_vec = ema4[i];
        float4 m_vec = m4[i];
        float4 v_vec = v4[i];

        float gs[4] = {g_vec.x, g_vec.y, g_vec.z, g_vec.w};
        float ps[4] = {p_vec.x, p_vec.y, p_vec.z, p_vec.w};
        float es[4] = {ema_vec.x, ema_vec.y, ema_vec.z, ema_vec.w};
        float ms[4] = {m_vec.x, m_vec.y, m_vec.z, m_vec.w};
        float vs[4] = {v_vec.x, v_vec.y, v_vec.z, v_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__builtin_isnan(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__builtin_isnan(g)) continue;
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            float ema_val = alpha * es[k] + one_minus_alpha * g;
            float g_amp = g + lamb * ema_val;

            float m = beta1 * ms[k] + one_minus_b1 * g_amp;
            float v = beta2 * vs[k] + one_minus_b2 * g_amp * g_amp;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            ps[k] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * ps[k]);
            es[k] = ema_val;
            ms[k] = m;
            vs[k] = v;
        }

        p4[i] = make_float4(ps[0], ps[1], ps[2], ps[3]);
        ema4[i] = make_float4(es[0], es[1], es[2], es[3]);
        m4[i] = make_float4(ms[0], ms[1], ms[2], ms[3]);
        v4[i] = make_float4(vs[0], vs[1], vs[2], vs[3]);
    }
}

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void grokfast_kernel(
    ParamT* params, const ParamT* grads, GrokfastState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float alpha, float lamb, float bc1, float bc2, float clip_threshold
) {
    grokfast_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd,
        alpha, lamb, bc1, bc2, clip_threshold);
}

}} // namespace grokking::gfx942

#endif // GROKKING_GROKFAST_GFX942_HIP_HPP_
