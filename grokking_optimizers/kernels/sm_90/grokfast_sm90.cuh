#ifndef GROKKING_GROKFAST_SM90_CUH_
#define GROKKING_GROKFAST_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

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
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,
    float lamb,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {
        float g = to_float(__ldg(&grads[i]));

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }
        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p = to_float(params[i]);

        // Phase 1: Grokfast EMA filter + amplification
        float ema_old = state.ema[i];
        float ema_val = alpha * ema_old + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: AdamW on the amplified gradient
        float m = beta1 * state.exp_avg[i] + (1.0f - beta1) * g_amp;
        float v = beta2 * state.exp_avg_sq[i] + (1.0f - beta2) * g_amp * g_amp;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        p = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);

        // Write back state and param
        state.ema[i] = ema_val;
        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
        params[i] = from_float<ParamT>(p);
    }
}

// Vectorized float4 variant for float params — 4x memory throughput
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokfast_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    GrokfastState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,
    float lamb,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t n4 = n >> 2;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    const float4* grads4 = reinterpret_cast<const float4*>(grads);
    float4* params4 = reinterpret_cast<float4*>(params);
    float4* ema4 = reinterpret_cast<float4*>(state.ema);
    float4* m4 = reinterpret_cast<float4*>(state.exp_avg);
    float4* v4 = reinterpret_cast<float4*>(state.exp_avg_sq);

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n4;
         i += stride) {
        float4 g4 = __ldg(&grads4[i]);
        float4 p4 = params4[i];
        float4 e4 = ema4[i];
        float4 mv = m4[i];
        float4 vv = v4[i];

        float g_arr[4] = {g4.x, g4.y, g4.z, g4.w};
        float p_arr[4] = {p4.x, p4.y, p4.z, p4.w};
        float e_arr[4] = {e4.x, e4.y, e4.z, e4.w};
        float m_arr[4] = {mv.x, mv.y, mv.z, mv.w};
        float v_arr[4] = {vv.x, vv.y, vv.z, vv.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = g_arr[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__isnanf(g)) continue;
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            float ema_val = alpha * e_arr[k] + (1.0f - alpha) * g;
            float g_amp = g + lamb * ema_val;

            float m_new = beta1 * m_arr[k] + (1.0f - beta1) * g_amp;
            float v_new = beta2 * v_arr[k] + (1.0f - beta2) * g_amp * g_amp;

            float m_hat = m_new * bias_correction1;
            float v_hat = v_new * bias_correction2;

            p_arr[k] = p_arr[k] - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_arr[k]);
            e_arr[k] = ema_val;
            m_arr[k] = m_new;
            v_arr[k] = v_new;
        }

        params4[i] = make_float4(p_arr[0], p_arr[1], p_arr[2], p_arr[3]);
        ema4[i] = make_float4(e_arr[0], e_arr[1], e_arr[2], e_arr[3]);
        m4[i] = make_float4(m_arr[0], m_arr[1], m_arr[2], m_arr[3]);
        v4[i] = make_float4(v_arr[0], v_arr[1], v_arr[2], v_arr[3]);
    }

    // Handle remaining elements (n % 4 != 0)
    const int64_t tail_start = n4 << 2;
    for (int64_t i = tail_start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {
        float g = __ldg(&grads[i]);

        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p = params[i];
        float ema_val = alpha * state.ema[i] + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        float m_new = beta1 * state.exp_avg[i] + (1.0f - beta1) * g_amp;
        float v_new = beta2 * state.exp_avg_sq[i] + (1.0f - beta2) * g_amp * g_amp;

        float m_hat = m_new * bias_correction1;
        float v_hat = v_new * bias_correction2;

        params[i] = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
        state.ema[i] = ema_val;
        state.exp_avg[i] = m_new;
        state.exp_avg_sq[i] = v_new;
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

// ---------------------------------------------------------------------------
// Prefetch-pipelined scalar Grokfast update: software-pipelined with 2
// register sets so that loads for the NEXT iteration overlap with compute
// on the current iteration.
// ---------------------------------------------------------------------------

template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void grokfast_update_prefetch(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    GrokfastState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float alpha,
    float lamb,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    // Load first element (current)
    float g_cur = to_float(__ldg(&grads[i]));
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
            g_next = to_float(__ldg(&grads[next]));
            p_next = to_float(params[next]);
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

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Phase 1: Grokfast EMA filter + amplification
        float ema_val = alpha * ema_cur + (1.0f - alpha) * g;
        float g_amp = g + lamb * ema_val;

        // Phase 2: AdamW on the amplified gradient
        float m = beta1 * m_cur + (1.0f - beta1) * g_amp;
        float v = beta2 * v_cur + (1.0f - beta2) * g_amp * g_amp;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float p_f = p_cur - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_cur);

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
__global__ void grokfast_kernel_prefetch(
    ParamT* params, const ParamT* grads, GrokfastState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float alpha, float lamb, float bc1, float bc2, float clip_threshold
) {
    grokfast_update_prefetch<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd,
        alpha, lamb, bc1, bc2, clip_threshold);
}

}} // namespace grokking::sm90

#endif // GROKKING_GROKFAST_SM90_CUH_
