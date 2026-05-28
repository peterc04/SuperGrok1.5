#ifndef GROKKING_SUPERGROK15_GFX942_HIP_HPP_
#define GROKKING_SUPERGROK15_GFX942_HIP_HPP_

#include "common_gfx942.hip.hpp"

namespace grokking {
namespace gfx942 {

// Forward-declare metanet_forward from supergrok11 header; if compiling
// standalone, include supergrok11_gfx942.hip.hpp before this header or
// the fallback definition below will be used.
#ifndef GROKKING_SUPERGROK11_GFX942_HIP_HPP_
__forceinline__ __device__
float metanet_forward(
    float g,
    float sharp,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    float rescale,
    int hidden_dim
) {
    float out = __builtin_nontemporal_load(b2);

    for (int j = 0; j < hidden_dim; ++j) {
        float z = __builtin_nontemporal_load(&W1[j * 2 + 0]) * g
                + __builtin_nontemporal_load(&W1[j * 2 + 1]) * sharp
                + __builtin_nontemporal_load(&b1[j]);

        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kCoeff = 0.044715f;
        float z3 = z * z * z;
        float inner = kSqrt2OverPi * (z + kCoeff * z3);
        float h = 0.5f * z * (1.0f + tanhf(inner));

        out += __builtin_nontemporal_load(&W2[j]) * h;
    }

    return rescale * out;
}
#endif  // !GROKKING_SUPERGROK11_GFX942_HIP_HPP_

// ============================================================================
// SuperGrok v1.5 — Sigmoid gating (host-precomputed gate_signal)
//
// State tensors (4): exp_avg, exp_avg_sq, mu, sharpness
// Per-element pipeline:
//   1. Clip gradient
//   2. EMA:  mu = layer_alpha * mu + (1 - layer_alpha) * g
//   3. MetaNet forward -> correction;  smart_g = g + correction
//   4. gate_signal is a scalar passed from host (not computed per-element)
//   5. effective_g = g + ramp * lamb * gate_signal * (smart_g - g)
//   6. AdamW step with per-layer beta1 and effective weight decay
// ============================================================================

struct SuperGrok15State {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    float* __restrict__ mu;
    float* __restrict__ sharpness;

    static constexpr int num_state_tensors() { return 4; }
    static constexpr int state_bytes_per_element() { return 4 * sizeof(float); }
};

// --------------------------------------------------------------------------
// Scalar per-element update
// --------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void supergrok15_update(
    ParamT* __restrict__       params,
    const ParamT* __restrict__ grads,
    SuperGrok15State           state,
    int64_t                    n,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float grad_clip,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    float rescale,
    int   hidden_dim,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        // Streaming non-temporal grad read
        ParamT g_raw = __builtin_nontemporal_load(grads + i);
        float g = to_float(g_raw);
        g = apply_nan_policy<NAN_POLICY>(g);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

        // Step 1: always-on per-element gradient clip
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        // Step 2: EMA gradient update
        float mu_old = state.mu[i];
        float mu_val = layer_alpha * mu_old + (1.0f - layer_alpha) * g;

        // Read current sharpness
        float sharp = state.sharpness[i];

        // Step 3: MetaNet forward
        float correction = metanet_forward(g, sharp, W1, b1, W2, b2, rescale, hidden_dim);
        float smart_g = g + correction;

        // Step 4-5: gate_signal is host-precomputed (sigmoid); apply directly
        float effective_g = g + ramp * lamb * gate_signal * (smart_g - g);

        // Step 6: AdamW step using effective_g
        float p_f   = to_float(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom  = sqrtf(v_hat) + eps;
        float update = m_hat / denom + weight_decay * p_f;

        p_f -= lr * update;

        // Update sharpness (magnitude of gradient correction as a running signal)
        float new_sharp = layer_alpha * sharp + (1.0f - layer_alpha) * fabsf(correction);

        // Writeback
        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        state.mu[i]         = mu_val;
        state.sharpness[i]  = new_sharp;
        params[i]           = from_float<ParamT>(p_f);
    }
}

// --------------------------------------------------------------------------
// Vectorized path for float params, 4 elements per thread via float4.
// --------------------------------------------------------------------------
template <NanPolicy NAN_POLICY = NanPolicy::kNone, bool ENABLE_CLIP = false>
__forceinline__ __device__
void supergrok15_update_vec4(
    float* __restrict__       params,
    const float* __restrict__ grads,
    SuperGrok15State          state,
    int64_t                   n,
    float lr,
    float layer_beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,
    float grad_clip,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    float rescale,
    int   hidden_dim,
    float bias_correction1,
    float bias_correction2,
    float clip_threshold = 0.0f
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const int64_t n4 = n / 4;

    float4* __restrict__       p4  = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(grads);
    float4* __restrict__       m4  = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__       v4  = reinterpret_cast<float4*>(state.exp_avg_sq);
    float4* __restrict__       mu4 = reinterpret_cast<float4*>(state.mu);
    float4* __restrict__       sh4 = reinterpret_cast<float4*>(state.sharpness);

    for (int64_t i = idx; i < n4; i += stride) {
        float4 p_vec  = p4[i];
        float4 g_vec  = __builtin_nontemporal_load(g4 + i);
        float4 m_vec  = m4[i];
        float4 v_vec  = v4[i];
        float4 mu_vec = mu4[i];
        float4 sh_vec = sh4[i];

        float gs[4]  = {g_vec.x,  g_vec.y,  g_vec.z,  g_vec.w};
        float ps[4]  = {p_vec.x,  p_vec.y,  p_vec.z,  p_vec.w};
        float ms[4]  = {m_vec.x,  m_vec.y,  m_vec.z,  m_vec.w};
        float vs[4]  = {v_vec.x,  v_vec.y,  v_vec.z,  v_vec.w};
        float mus[4] = {mu_vec.x, mu_vec.y, mu_vec.z, mu_vec.w};
        float shs[4] = {sh_vec.x, sh_vec.y, sh_vec.z, sh_vec.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float g = apply_nan_policy<NAN_POLICY>(gs[k]);
            g = apply_clip<ENABLE_CLIP>(g, clip_threshold);

            // Always-on gradient clip
            g = fminf(fmaxf(g, -grad_clip), grad_clip);

            // EMA
            float mu_val = layer_alpha * mus[k] + (1.0f - layer_alpha) * g;

            // MetaNet
            float correction = metanet_forward(g, shs[k], W1, b1, W2, b2, rescale, hidden_dim);
            float smart_g = g + correction;

            // Host-precomputed sigmoid gate
            float effective_g = g + ramp * lamb * gate_signal * (smart_g - g);

            // AdamW
            float m = layer_beta1 * ms[k] + (1.0f - layer_beta1) * effective_g;
            float v = beta2 * vs[k] + (1.0f - beta2) * effective_g * effective_g;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            ps[k] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * ps[k]);
            ms[k]  = m;
            vs[k]  = v;
            mus[k] = mu_val;
            shs[k] = layer_alpha * shs[k] + (1.0f - layer_alpha) * fabsf(correction);
        }

        p4[i]  = make_float4(ps[0],  ps[1],  ps[2],  ps[3]);
        m4[i]  = make_float4(ms[0],  ms[1],  ms[2],  ms[3]);
        v4[i]  = make_float4(vs[0],  vs[1],  vs[2],  vs[3]);
        mu4[i] = make_float4(mus[0], mus[1], mus[2], mus[3]);
        sh4[i] = make_float4(shs[0], shs[1], shs[2], shs[3]);
    }

    // Handle tail elements
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        float g_raw = __builtin_nontemporal_load(grads + i);
        float g = apply_nan_policy<NAN_POLICY>(g_raw);
        g = apply_clip<ENABLE_CLIP>(g, clip_threshold);
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        float mu_old = state.mu[i];
        float mu_val = layer_alpha * mu_old + (1.0f - layer_alpha) * g;
        float sharp = state.sharpness[i];

        float correction = metanet_forward(g, sharp, W1, b1, W2, b2, rescale, hidden_dim);
        float smart_g = g + correction;

        float effective_g = g + ramp * lamb * gate_signal * (smart_g - g);

        float p_f   = params[i];
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        p_f -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p_f);

        float new_sharp = layer_alpha * sharp + (1.0f - layer_alpha) * fabsf(correction);

        state.exp_avg[i]    = m;
        state.exp_avg_sq[i] = v;
        state.mu[i]         = mu_val;
        state.sharpness[i]  = new_sharp;
        params[i]           = p_f;
    }
}

// --------------------------------------------------------------------------
// Global launcher kernel
// --------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void supergrok15_kernel(
    ParamT* params, const ParamT* grads, SuperGrok15State state, int64_t n,
    float lr, float layer_beta1, float beta2, float eps, float wd,
    float layer_alpha, float lamb, float ramp, float gate_signal, float grad_clip,
    const float* W1, const float* b1, const float* W2, const float* b2,
    float rescale, int hidden_dim,
    float bc1, float bc2, float clip_threshold
) {
    supergrok15_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n,
        lr, layer_beta1, beta2, eps, wd,
        layer_alpha, lamb, ramp, gate_signal, grad_clip,
        W1, b1, W2, b2, rescale, hidden_dim,
        bc1, bc2, clip_threshold);
}

}  // namespace gfx942
}  // namespace grokking

#endif  // GROKKING_SUPERGROK15_GFX942_HIP_HPP_
