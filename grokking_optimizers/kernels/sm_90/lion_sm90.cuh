#ifndef GROKKING_LION_SM90_CUH_
#define GROKKING_LION_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

struct LionState {
    float* __restrict__ exp_avg;

    static constexpr int num_state_tensors() { return 1; }
    static constexpr int state_bytes_per_element() { return sizeof(float); }
};

// Per-element Lion update for scalar paths
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void lion_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    LionState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float clip_threshold = 0.0f
) {
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * (int64_t)blockDim.x) {
        float g = to_float(__ldg(&grads[i]));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        float p_f = to_float(params[i]);
        float m_old = state.exp_avg[i];

        // Update direction: sign of interpolation between momentum and gradient
        float interp = beta1 * m_old + one_minus_b1 * g;
        float sign_val = copysignf(1.0f, interp);

        // Decoupled weight decay + signed update
        p_f = p_f - lr * (sign_val + weight_decay * p_f);

        // Momentum update (AFTER parameter update)
        float m_new = beta2 * m_old + one_minus_b2 * g;

        params[i] = from_float<ParamT>(p_f);
        state.exp_avg[i] = m_new;
    }
}

// Vectorized float path: 4 elements per thread via float4
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void lion_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    LionState state,
    int64_t n4,               // number of float4 groups (n / 4)
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    float clip_threshold = 0.0f
) {
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b2 = 1.0f - beta2;

    float4* __restrict__ p4 = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(grads);
    float4* __restrict__ m4 = reinterpret_cast<float4*>(state.exp_avg);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * (int64_t)blockDim.x) {
        float4 p = p4[i];
        float4 g = g4[i];
        float4 m = m4[i];

        // Process each of the 4 lanes
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            float* gv = &((&g.x)[lane]);
            float* pv = &((&p.x)[lane]);
            float* mv = &((&m.x)[lane]);

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(*gv)) *gv = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__isnanf(*gv)) continue;
            }

            if constexpr (ENABLE_CLIP) {
                *gv = fminf(fmaxf(*gv, -clip_threshold), clip_threshold);
            }

            float interp = beta1 * (*mv) + one_minus_b1 * (*gv);
            float sign_val = copysignf(1.0f, interp);

            *pv = *pv - lr * (sign_val + weight_decay * (*pv));
            *mv = beta2 * (*mv) + one_minus_b2 * (*gv);
        }

        p4[i] = p;
        m4[i] = m;
    }
}

// Launcher kernel (scalar)
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void lion_kernel(
    ParamT* params, const ParamT* grads, LionState state, int64_t n,
    float lr, float beta1, float beta2, float wd, float clip_threshold
) {
    lion_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, wd, clip_threshold);
}

}} // namespace grokking::sm90

#endif // GROKKING_LION_SM90_CUH_
