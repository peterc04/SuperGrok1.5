#ifndef GROKKING_GROKADAMW_SM90_CUH_
#define GROKKING_GROKADAMW_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

struct GrokAdamWState {
    float* __restrict__ ema;
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;

    static constexpr int num_state_tensors() { return 3; }
    static constexpr int state_bytes_per_element() { return 3 * sizeof(float); }
};

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokadamw_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,              // EMA decay for gradient filter
    float lamb,               // amplification factor
    float grad_clip,          // always-on per-element clipping threshold
    float bias_correction1,   // host-precomputed: 1 / (1 - beta1^t)
    float bias_correction2,   // host-precomputed: 1 / (1 - beta2^t)
    float clip_threshold = 0.0f  // ENABLE_CLIP pre-filter threshold
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float g = to_float<ParamT>(__ldg(&grads[i]));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        // Optional pre-filter clip on raw gradient
        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Phase 1: EMA filter + amplification
        float ema_old = state.ema[i];
        float ema_val = alpha * ema_old + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: Always-on per-element clip on amplified gradient
        g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

        // Phase 3: AdamW on clipped+amplified gradient
        float p_f = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = beta1 * m_old + (1.0f - beta1) * g_amp;
        float v = beta2 * v_old + (1.0f - beta2) * g_amp * g_amp;

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
void grokadamw_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,
    float lamb,
    float grad_clip,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t n4 = n / 4;
    float4* __restrict__ p4 = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(grads);
    float4* __restrict__ e4 = reinterpret_cast<float4*>(state.ema);
    float4* __restrict__ m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__ v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 p_vec = p4[i];
        float4 g_vec = g4[i];
        float4 e_vec = e4[i];
        float4 m_vec = m4[i];
        float4 v_vec = v4[i];

        float gs[4] = {g_vec.x, g_vec.y, g_vec.z, g_vec.w};
        float ps[4] = {p_vec.x, p_vec.y, p_vec.z, p_vec.w};
        float es[4] = {e_vec.x, e_vec.y, e_vec.z, e_vec.w};
        float ms[4] = {m_vec.x, m_vec.y, m_vec.z, m_vec.w};
        float vs[4] = {v_vec.x, v_vec.y, v_vec.z, v_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                // vec4: zero NaN lane instead of skip (can't skip individual lanes)
                if (__isnanf(g)) { continue; }
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            // EMA filter + amplification
            float ema_val = alpha * es[k] + (1.0f - alpha) * g;
            float g_amp = g + lamb * ema_val;

            // Always-on per-element clip
            g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

            // AdamW
            float m = beta1 * ms[k] + (1.0f - beta1) * g_amp;
            float v = beta2 * vs[k] + (1.0f - beta2) * g_amp * g_amp;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            float denom = sqrtf(v_hat) + eps;
            ps[k] -= lr * (m_hat / denom + weight_decay * ps[k]);
            es[k] = ema_val;
            ms[k] = m;
            vs[k] = v;
        }

        p4[i] = make_float4(ps[0], ps[1], ps[2], ps[3]);
        e4[i] = make_float4(es[0], es[1], es[2], es[3]);
        m4[i] = make_float4(ms[0], ms[1], ms[2], ms[3]);
        v4[i] = make_float4(vs[0], vs[1], vs[2], vs[3]);
    }
}

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
// Prefetch-pipelined scalar GrokAdamW update: software-pipelined with 2
// register sets so that loads for the NEXT iteration overlap with compute
// on the current iteration.
// ---------------------------------------------------------------------------

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokadamw_update_prefetch(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokAdamWState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,
    float lamb,
    float grad_clip,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    // Load first element (current)
    float g_cur = to_float<ParamT>(__ldg(&grads[i]));
    float p_cur = to_float<ParamT>(params[i]);
    float ema_cur = state.ema[i];
    float m_cur = state.exp_avg[i];
    float v_cur = state.exp_avg_sq[i];

    for (; i < n; i += stride) {
        // Prefetch next iteration's data
        int64_t next = i + stride;
        float g_next, p_next, ema_next, m_next, v_next;
        bool has_next = next < n;
        if (has_next) {
            g_next = to_float<ParamT>(__ldg(&grads[next]));
            p_next = to_float<ParamT>(params[next]);
            ema_next = state.ema[next];
            m_next = state.exp_avg[next];
            v_next = state.exp_avg_sq[next];
        }

        // ---- Process current element ----
        float g = g_cur;

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) {
                if (has_next) { g_cur = g_next; p_cur = p_next; ema_cur = ema_next; m_cur = m_next; v_cur = v_next; }
                continue;
            }
        }

        // Optional pre-filter clip on raw gradient
        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Phase 1: EMA filter + amplification
        float ema_val = alpha * ema_cur + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: Always-on per-element clip on amplified gradient
        g_amp = fminf(fmaxf(g_amp, -grad_clip), grad_clip);

        // Phase 3: AdamW on clipped+amplified gradient
        float m = beta1 * m_cur + (1.0f - beta1) * g_amp;
        float v = beta2 * v_cur + (1.0f - beta2) * g_amp * g_amp;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_cur;
        float p_f = p_cur - lr * update;

        // Write back state and param
        state.ema[i] = ema_val;
        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
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

}} // namespace grokking::sm90

#endif // GROKKING_GROKADAMW_SM90_CUH_
