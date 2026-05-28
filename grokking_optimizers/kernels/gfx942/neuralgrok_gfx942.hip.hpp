#ifndef GROKKING_NEURALGROK_GFX942_HIP_HPP_
#define GROKKING_NEURALGROK_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking { namespace gfx942 {

struct NeuralGrokState {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;

    static constexpr int num_state_tensors() { return 2; }
    static constexpr int state_bytes_per_element() { return 2 * sizeof(float); }
};

// ---------------------------------------------------------------------------
// 2-layer MLP forward for a single element (psi-net).
//
// Input:  |grad|  (scalar)
// Hidden: h_j = relu(W1[j] * |grad| + b1[j])   for j in [0, hidden_dim)
// Output: scale = W_last @ h + b_last
//
// MLP weights are passed as raw float pointers (the caller controls where
// they live -- __constant__, global, etc.).
// ---------------------------------------------------------------------------
__forceinline__ __device__
float neuralgrok_mlp_forward(
    float abs_grad,
    const float* __restrict__ W1,       // [hidden_dim]
    const float* __restrict__ b1,       // [hidden_dim]
    const float* __restrict__ W_last,   // [hidden_dim]
    float b_last,
    int hidden_dim
) {
    float acc = b_last;
    for (int j = 0; j < hidden_dim; ++j) {
        float h = W1[j] * abs_grad + b1[j];
        h = fmaxf(h, 0.0f);  // ReLU
        acc += W_last[j] * h;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Per-element NeuralGrok update (scalar path)
//
// Steps:
//   1. Clip gradient:  g = clamp(g, -grad_clip, grad_clip)
//   2. MLP forward:    h = relu(W1 * |g| + b1), scale = W_last @ h + b_last
//   3. Amplify:        g_amp = g + alpha * scale * g + beta_amp * scale * sign(g)
//   4. Standard AdamW step using g_amp
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void neuralgrok_update(
    ParamT* __restrict__ params,
    const ParamT* __restrict__ grads,
    NeuralGrokState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W_last,
    float b_last,
    float alpha,
    float beta_amp,
    int hidden_dim,
    float grad_clip,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // gfx942 has no __ldg; non-temporal load bypasses L2 for streaming grad data
        ParamT g_raw = __builtin_nontemporal_load(&grads[i]);
        float g = to_float<ParamT>(g_raw);

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (isnan(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (isnan(g)) continue;
        }

        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Step 1: Clip gradient for MLP stability
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        // Step 2: MLP forward -- compute per-element amplification scale
        float abs_g = fabsf(g);
        float scale = neuralgrok_mlp_forward(abs_g, W1, b1, W_last, b_last, hidden_dim);

        // Step 3: Amplified gradient
        float sign_g = copysignf(1.0f, g);
        float g_amp = g + alpha * scale * g + beta_amp * scale * sign_g;

        // Step 4: AdamW step on g_amp
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

        state.exp_avg[i] = m;
        state.exp_avg_sq[i] = v;
        params[i] = from_float<ParamT>(p_f);
    }
}

// ---------------------------------------------------------------------------
// Vectorized path for float params -- 4 elements per thread via float4
// loads/stores.  Caller must guarantee n % 4 == 0 and 16-byte alignment.
// ---------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void neuralgrok_update_vec4(
    float* __restrict__ params,
    const float* __restrict__ grads,
    NeuralGrokState state,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W_last,
    float b_last,
    float alpha,
    float beta_amp,
    int hidden_dim,
    float grad_clip,
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

            // Step 1: Clip gradient for MLP stability
            g = fminf(fmaxf(g, -grad_clip), grad_clip);

            // Step 2: MLP forward
            float abs_g = fabsf(g);
            float scale = neuralgrok_mlp_forward(abs_g, W1, b1, W_last, b_last, hidden_dim);

            // Step 3: Amplified gradient
            float sign_g = copysignf(1.0f, g);
            float g_amp = g + alpha * scale * g + beta_amp * scale * sign_g;

            // Step 4: AdamW step
            float m = beta1 * ms[k] + (1.0f - beta1) * g_amp;
            float v = beta2 * vs[k] + (1.0f - beta2) * g_amp * g_amp;

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
// Launcher kernel
// ---------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void neuralgrok_kernel(
    ParamT* params, const ParamT* grads, NeuralGrokState state, int64_t n,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    const float* W1, const float* b1, const float* W_last, float b_last,
    float alpha, float beta_amp, int hidden_dim, float grad_clip,
    float clip_threshold
) {
    neuralgrok_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n, lr, beta1, beta2, eps, wd, bc1, bc2,
        W1, b1, W_last, b_last, alpha, beta_amp, hidden_dim, grad_clip,
        clip_threshold);
}

}} // namespace grokking::gfx942

#endif // GROKKING_NEURALGROK_GFX942_HIP_HPP_
