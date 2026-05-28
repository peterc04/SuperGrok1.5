#ifndef GROKKING_GROKADAMW_GFX942_HIP_HPP_
#define GROKKING_GROKADAMW_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking {
namespace gfx942 {

struct GrokAdamWState {
    float* __restrict__ ema;
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    static constexpr int num_state_tensors() { return 3; }
    static constexpr int state_bytes_per_element() { return 3 * sizeof(float); }
};

// -- Scalar GrokAdamW update --

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokadamw_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float grad_clip,
    float bias_correction1, float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);
        g = apply_nan_policy<NAN_POLICY>(g);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        float p_f = to_float(params[i]);
        float ema_old = state.ema[i];
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        // Phase 1: EMA filter + amplification.
        float ema_val = alpha * ema_old + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: Always-on per-element clipping.
        g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

        // Phase 3: AdamW on clipped gradient.
        float m_new = beta1 * m_old + (1.0f - beta1) * g_amp;
        float v_new = beta2 * v_old + (1.0f - beta2) * g_amp * g_amp;
        float m_hat = m_new * bias_correction1;
        float v_hat = v_new * bias_correction2;
        p_f -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_f);

        state.ema[i] = ema_val;
        state.exp_avg[i] = m_new;
        state.exp_avg_sq[i] = v_new;
        params[i] = from_float<ParamT>(p_f);
    }
}

// -- Vectorized float4 GrokAdamW update (float params only) --

template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokadamw_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float grad_clip,
    float bias_correction1, float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t n4 = n / 4;

    float4* params4 = reinterpret_cast<float4*>(params);
    const float4* grads4 = reinterpret_cast<const float4*>(grads);
    float4* ema4 = reinterpret_cast<float4*>(state.ema);
    float4* m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 g4 = __builtin_nontemporal_load(grads4 + i);
        float4 p4 = params4[i];
        float4 eo4 = ema4[i];
        float4 mo4 = m4[i];
        float4 vo4 = v4[i];

        float g_arr[4] = {g4.x, g4.y, g4.z, g4.w};
        float p_arr[4] = {p4.x, p4.y, p4.z, p4.w};
        float e_arr[4] = {eo4.x, eo4.y, eo4.z, eo4.w};
        float m_arr[4] = {mo4.x, mo4.y, mo4.z, mo4.w};
        float v_arr[4] = {vo4.x, vo4.y, vo4.z, vo4.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = apply_nan_policy<NAN_POLICY>(g_arr[k]);
            g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

            float ema_val = alpha * e_arr[k] + (1.0f - alpha) * g;
            float g_amp = g + lamb * ema_val;
            g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

            float m_new = beta1 * m_arr[k] + (1.0f - beta1) * g_amp;
            float v_new = beta2 * v_arr[k] + (1.0f - beta2) * g_amp * g_amp;
            float m_hat = m_new * bias_correction1;
            float v_hat = v_new * bias_correction2;

            p_arr[k] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_arr[k]);
            e_arr[k] = ema_val;
            m_arr[k] = m_new;
            v_arr[k] = v_new;
        }

        params4[i] = make_float4(p_arr[0], p_arr[1], p_arr[2], p_arr[3]);
        ema4[i] = make_float4(e_arr[0], e_arr[1], e_arr[2], e_arr[3]);
        m4[i] = make_float4(m_arr[0], m_arr[1], m_arr[2], m_arr[3]);
        v4[i] = make_float4(v_arr[0], v_arr[1], v_arr[2], v_arr[3]);
    }

    // Handle tail elements.
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        float g_raw = __builtin_nontemporal_load(grads + i);
        float g = apply_nan_policy<NAN_POLICY>(g_raw);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        float p_f = params[i];
        float ema_old = state.ema[i];
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float ema_val = alpha * ema_old + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;
        g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

        float m_new = beta1 * m_old + (1.0f - beta1) * g_amp;
        float v_new = beta2 * v_old + (1.0f - beta2) * g_amp * g_amp;
        float m_hat = m_new * bias_correction1;
        float v_hat = v_new * bias_correction2;

        p_f -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_f);

        params[i] = p_f;
        state.ema[i] = ema_val;
        state.exp_avg[i] = m_new;
        state.exp_avg_sq[i] = v_new;
    }
}

// -- Launcher kernel --

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void grokadamw_kernel(
    ParamT* params, const ParamT* grads, GrokAdamWState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float alpha, float lamb, float grad_clip,
    float bc1, float bc2, float clip_threshold
) {
    grokadamw_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd,
        alpha, lamb, grad_clip, bc1, bc2, clip_threshold);
}

// ---------------------------------------------------------------------------
// Prefetch-pipelined scalar GrokAdamW update for gfx942: software-pipelined
// with 2 register sets.  Uses __builtin_nontemporal_load for prefetch reads.
// ---------------------------------------------------------------------------

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokadamw_update_prefetch(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float alpha, float lamb, float grad_clip,
    float bias_correction1, float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    // Load first element (current)
    float g_cur = to_float(__builtin_nontemporal_load(grads + i));
    float p_cur = to_float(params[i]);
    float ema_cur = state.ema[i];
    float m_cur = state.exp_avg[i];
    float v_cur = state.exp_avg_sq[i];

    for (; i < n; i += stride) {
        // Prefetch next iteration's data
        int64_t next = i + stride;
        float g_next, p_next, ema_next, m_next, v_next;
        bool has_next = next < n;
        if (has_next) {
            g_next = to_float(__builtin_nontemporal_load(grads + next));
            p_next = to_float(params[next]);
            ema_next = state.ema[next];
            m_next = state.exp_avg[next];
            v_next = state.exp_avg_sq[next];
        }

        // ---- Process current element ----
        float g = g_cur;
        g = apply_nan_policy<NAN_POLICY>(g);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        // Phase 1: EMA filter + amplification
        float ema_val = alpha * ema_cur + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: Always-on per-element clipping
        g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

        // Phase 3: AdamW on clipped gradient
        float m_new = beta1 * m_cur + (1.0f - beta1) * g_amp;
        float v_new = beta2 * v_cur + (1.0f - beta2) * g_amp * g_amp;
        float m_hat = m_new * bias_correction1;
        float v_hat = v_new * bias_correction2;
        float p_f = p_cur - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_cur);

        // Write back state and param
        state.ema[i] = ema_val;
        state.exp_avg[i] = m_new;
        state.exp_avg_sq[i] = v_new;
        params[i] = from_float<ParamT>(p_f);

        // Swap prefetched data into current registers
        if (has_next) {
            g_cur = g_next;
            p_cur = p_next;
            ema_cur = ema_next;
            m_cur = m_next;
            v_cur = v_next;
        }
    }
}

template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void grokadamw_kernel_prefetch(
    ParamT* params, const ParamT* grads, GrokAdamWState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float alpha, float lamb, float grad_clip,
    float bc1, float bc2, float clip_threshold
) {
    grokadamw_update_prefetch<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd,
        alpha, lamb, grad_clip, bc1, bc2, clip_threshold);
}

}  // namespace gfx942
}  // namespace grokking

#endif  // GROKKING_GROKADAMW_GFX942_HIP_HPP_
