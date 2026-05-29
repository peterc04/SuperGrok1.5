#ifndef GROKKING_SUPERGROK15_SM90_CUH_
#define GROKKING_SUPERGROK15_SM90_CUH_

#include "common_sm90.cuh"

namespace grokking { namespace sm90 {

struct SuperGrok15State {
    float* __restrict__ exp_avg;
    float* __restrict__ exp_avg_sq;
    float* __restrict__ mu;
    float* __restrict__ sharpness;

    static constexpr int num_state_tensors() { return 4; }
    static constexpr int state_bytes_per_element() { return 4 * sizeof(float); }
};

// --------------------------------------------------------------------------
// MetaNet forward pass (2-layer MLP):  inp=[g, sharpness]
//   h = gelu(W1 @ inp + b1)   -- W1 is (H x 2), b1 is (H,)
//   out = rescale * (W2 @ h + b2)  -- W2 is (1 x H), b2 is (1,)
// --------------------------------------------------------------------------
__forceinline__ __device__
float supergrok15_metanet_forward(
    float g,
    float sharp,
    const float* __restrict__ W1,   // H x 2  (row-major)
    const float* __restrict__ b1,   // H
    const float* __restrict__ W2,   // 1 x H  (row-major)
    const float* __restrict__ b2,   // 1
    float rescale,
    int hidden_dim
) {
    float out = __ldg(&b2[0]);

    for (int j = 0; j < hidden_dim; ++j) {
        // W1 is (H, 2): row j has W1[j*2+0] and W1[j*2+1]
        float z = __ldg(&W1[j * 2 + 0]) * g
                + __ldg(&W1[j * 2 + 1]) * sharp
                + __ldg(&b1[j]);

        // GELU approximation (tanh form)
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kCoeff = 0.044715f;
        float z3 = z * z * z;
        float inner = kSqrt2OverPi * (z + kCoeff * z3);
        float h = 0.5f * z * (1.0f + tanhf(inner));

        out += __ldg(&W2[j]) * h;
    }

    return rescale * out;
}

// --------------------------------------------------------------------------
// SuperGrok v1.5 per-element update
//   1. Clip gradient
//   2. EMA update:  mu = layer_alpha * mu + (1-layer_alpha) * g
//   3. MetaNet forward -> correction;  smart_g = g + correction
//   4. Sigmoid gate (host-precomputed gate_signal passed as scalar):
//        effective_g = g + ramp * lamb * gate_signal * (smart_g - g)
//   5. AdamW step using effective_g with effective_wd (host-precomputed)
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
    float effective_wd,         // host-precomputed effective weight decay
    float layer_alpha,
    float lamb,
    float ramp,
    float gate_signal,          // host-precomputed sigmoid gate (scalar)
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
    // Pre-compute the blending factor once (uniform across all elements)
    const float blend = ramp * lamb * gate_signal;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * (int64_t)blockDim.x) {

        float g = to_float<ParamT>(__ldg(&grads[i]));

        // NaN handling
        if constexpr (NAN_POLICY == NanPolicy::kZero) {
            if (__isnanf(g)) g = 0.0f;
        } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
            if (__isnanf(g)) continue;
        }

        // Optional pre-filter clip (template-gated)
        if constexpr (ENABLE_CLIP) {
            g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
        }

        // Step 1: always-on per-element gradient clip
        g = fminf(fmaxf(g, -grad_clip), grad_clip);

        // Step 2: EMA gradient update
        float mu_old = state.mu[i];
        float mu_val = layer_alpha * mu_old + (1.0f - layer_alpha) * g;

        // Read current sharpness
        float sharp = state.sharpness[i];

        // Step 3: MetaNet forward
        float correction = supergrok15_metanet_forward(
            g, sharp, W1, b1, W2, b2, rescale, hidden_dim);
        float smart_g = g + correction;

        // Step 4: Sigmoid gate — gate_signal is host-precomputed
        float effective_g = g + blend * (smart_g - g);

        // Step 5: AdamW step using effective_g
        float p_f   = to_float<ParamT>(params[i]);
        float m_old = state.exp_avg[i];
        float v_old = state.exp_avg_sq[i];

        float m = layer_beta1 * m_old + (1.0f - layer_beta1) * effective_g;
        float v = beta2 * v_old + (1.0f - beta2) * effective_g * effective_g;

        float m_hat = m * bias_correction1;
        float v_hat = v * bias_correction2;

        float denom  = sqrtf(v_hat) + eps;
        float update = m_hat / denom + effective_wd * p_f;

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
// Caller must guarantee n % 4 == 0 and 16-byte alignment.
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
    float effective_wd,
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
    const int64_t n4 = n / 4;
    const float blend = ramp * lamb * gate_signal;

    float4* __restrict__       p4  = reinterpret_cast<float4*>(params);
    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(grads);
    float4* __restrict__       m4  = reinterpret_cast<float4*>(state.exp_avg);
    float4* __restrict__       v4  = reinterpret_cast<float4*>(state.exp_avg_sq);
    float4* __restrict__       mu4 = reinterpret_cast<float4*>(state.mu);
    float4* __restrict__       sh4 = reinterpret_cast<float4*>(state.sharpness);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * (int64_t)blockDim.x) {

        float4 p_vec  = p4[i];
        float4 g_vec  = g4[i];
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
            float g = gs[k];

            if constexpr (NAN_POLICY == NanPolicy::kZero) {
                if (__isnanf(g)) g = 0.0f;
            } else if constexpr (NAN_POLICY == NanPolicy::kPropagate) {
                if (__isnanf(g)) continue;
            }

            if constexpr (ENABLE_CLIP) {
                g = fminf(fmaxf(g, -clip_threshold), clip_threshold);
            }

            // Always-on gradient clip
            g = fminf(fmaxf(g, -grad_clip), grad_clip);

            // EMA
            float mu_val = layer_alpha * mus[k] + (1.0f - layer_alpha) * g;

            // MetaNet
            float correction = supergrok15_metanet_forward(
                g, shs[k], W1, b1, W2, b2, rescale, hidden_dim);
            float smart_g = g + correction;

            // Sigmoid gate (host-precomputed)
            float effective_g = g + blend * (smart_g - g);

            // AdamW
            float m = layer_beta1 * ms[k] + (1.0f - layer_beta1) * effective_g;
            float v = beta2 * vs[k] + (1.0f - beta2) * effective_g * effective_g;

            float m_hat = m * bias_correction1;
            float v_hat = v * bias_correction2;

            ps[k] -= lr * (m_hat / (sqrtf(v_hat) + eps) + effective_wd * ps[k]);
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
}

// --------------------------------------------------------------------------
// Global launcher kernel
// --------------------------------------------------------------------------
template <typename ParamT, NanPolicy NAN_POLICY, bool ENABLE_CLIP>
__global__ void supergrok15_kernel(
    ParamT* params, const ParamT* grads, SuperGrok15State state, int64_t n,
    float lr, float layer_beta1, float beta2, float eps, float effective_wd,
    float layer_alpha, float lamb, float ramp, float gate_signal, float grad_clip,
    const float* W1, const float* b1, const float* W2, const float* b2,
    float rescale, int hidden_dim,
    float bc1, float bc2, float clip_threshold
) {
    supergrok15_update<ParamT, NAN_POLICY, ENABLE_CLIP>(
        params, grads, state, n,
        lr, layer_beta1, beta2, eps, effective_wd,
        layer_alpha, lamb, ramp, gate_signal, grad_clip,
        W1, b1, W2, b2, rescale, hidden_dim,
        bc1, bc2, clip_threshold);
}

}} // namespace grokking::sm90

#endif // GROKKING_SUPERGROK15_SM90_CUH_
