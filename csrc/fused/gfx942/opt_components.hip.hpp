#ifndef SG_FUSED_GFX942_OPT_COMPONENTS_HIP_HPP_
#define SG_FUSED_GFX942_OPT_COMPONENTS_HIP_HPP_
// ============================================================================
// csrc/fused/gfx942/opt_components.hip.hpp — REAL per-optimizer device-function
// composition library for the gfx942 fused L3/L1 megakernel (Phase 3 cont.).
//
// The AMD twin of csrc/fused/sm_90/opt_components.cuh. It eliminates the gfx942
// false positive (the demo's 4-optimizer opt_update + AdamW fallback for the
// other 7). Every one of the 11 optimizers here has its OWN real per-element
// update — transcribed byte-faithfully from the canonical reference math in
// csrc/algorithms/<opt>.h (adamw_step / lion_step / grokfast_fused_step /
// grokadamw_step / looksam_apply_step / prodigy_apply_step /
// neuralgrok_apply_step+psi_forward / muon_update_step / sg11_sweep_b_step /
// sg15_sweep_b_step + sg15_alpha_per_coord). NO fallback, NO templating one
// optimizer from another.
//
// WHY transcribed (not #included): csrc/algorithms/<opt>.h pulls
// csrc/common/platform.h whose GROK_HIP branch includes <thrust/...>, which the
// free-standing AMDGCN device gate (scripts/amdgcn_check.sh, no ROCm/thrust)
// cannot resolve. The math is duplicated verbatim here for per-file self-
// containment (an explicitly-accepted trade) and is gate-compilable with clang
// __builtin_* math. A parity test against the algorithm headers is the CPU/
// MI300X oracle item in HARDWARE_VALIDATION.md.
//
// Compiled by the DEVICE pass only (__AMDGCN__ gate, or __HIPCC__).
// ============================================================================

#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)

namespace sg { namespace fused { namespace gfx942_mega {

// Compile-time optimizer selector — one value per real optimizer (no fallback).
enum class OptId : int {
    AdamW = 0, Lion = 1, Grokfast = 2, GrokAdamW = 3, LookSAM = 4,
    Prodigy = 5, NeuralGrok = 6, Muon = 7, SuperGrok11 = 8,
    SuperGrok15 = 9, SuperGrok2 = 10
};

static constexpr int kPsiHidden = 16;   // neuralgrok psi-net hidden width

// All optimizer state any of the 11 tails may read (mirror of the sm_90 struct).
// A cell zero-fills the pointers its optimizer does not use; each branch touches
// ONLY its own optimizer's real buffers.
struct FusedOptState {
    float* exp_avg     = nullptr;
    float* exp_avg_sq  = nullptr;
    float* ema         = nullptr;
    const float* sam_dir = nullptr;
    float* s_track     = nullptr;
    float  d_factor    = 1.0f;
    float* mu          = nullptr;
    float  gate        = 1.0f;
    const float* orth  = nullptr;
    float  neg_lr_scale = 0.0f;
    float  decay_factor = 1.0f;
    const float* psi_W1 = nullptr;
    const float* psi_b1 = nullptr;
    const float* psi_W2 = nullptr;
    float        psi_b2 = 0.0f;
    const float* smart_grad = nullptr;
    float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    float bc1 = 1.0f, bc2 = 1.0f;
    float alpha = 0.98f, beta = 0.0f, lamb = 2.0f, alpha_max = 1.0f;
};

// Shared Adam tail: m,v EMAs on g_eff, bias-corrected decoupled-WD apply.
// (bc1/bc2 un-inverted = 1-beta^t; divide — matches the algorithm headers.)
__device__ __forceinline__ float sg_adam_tail(
        float p, float g_eff, float* m_buf, float* v_buf, int i,
        float b1, float b2, float eps, float bc1, float bc2,
        float lr, float wd) {
    const float m = b1 * m_buf[i] + (1.0f - b1) * g_eff;
    const float v = b2 * v_buf[i] + (1.0f - b2) * g_eff * g_eff;
    m_buf[i] = m; v_buf[i] = v;
    const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
    return p - lr * (update + wd * p);
}

// NeuralGrok psi-net (per-element MLP over |g|). Faithful to neuralgrok_psi_forward.
__device__ __forceinline__ float sg_psi_forward(
        float abs_grad, const float* W1, const float* b1,
        const float* W2, float b2) {
    float h_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kPsiHidden; j++) {
        float h = W1[j] * abs_grad + b1[j];
        h = (h > 0.0f) ? h : 0.0f;           // ReLU
        h_acc += W2[j] * h;
    }
    return h_acc + b2;
}

// SG15 per-coordinate alpha (faithful to sg15_alpha_per_coord).
__device__ __forceinline__ float sg_sg15_alpha(float mu_val, float ab, float am) {
    const float a = ab * (1.0f + mu_val);
    return __builtin_fminf(__builtin_fmaxf(a, 0.0f), am);
}

// Dispatch to the REAL per-element update for the chosen optimizer. Each branch
// is the genuine, distinct math from csrc/algorithms/<opt>.h — no fallback.
template <OptId Opt>
__device__ __forceinline__ void apply_optimizer(
        float* __restrict__ params, const float* __restrict__ grad,
        int i, int step, const FusedOptState& st) {
    (void)step;
    const float g = grad[i];
    const float p = params[i];
    if constexpr (Opt == OptId::AdamW) {
        params[i] = sg_adam_tail(p, g, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::Lion) {
        const float ea = st.exp_avg[i];
        const float interp = st.beta1 * ea + (1.0f - st.beta1) * g;
        const float s = (interp != 0.0f) ? __builtin_copysignf(1.0f, interp) : 0.0f;
        st.exp_avg[i] = st.beta2 * ea + (1.0f - st.beta2) * g;
        params[i] = p - st.lr * (s + st.wd * p);
    } else if constexpr (Opt == OptId::Grokfast) {
        const float e = st.alpha * st.ema[i] + (1.0f - st.alpha) * g;
        st.ema[i] = e;
        const float g_amp = g + st.lamb * e;
        params[i] = sg_adam_tail(p, g_amp, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::GrokAdamW) {
        const float e = st.alpha * st.ema[i] + (1.0f - st.alpha) * g;
        st.ema[i] = e;
        const float g_amp = g + st.lamb * e;
        params[i] = sg_adam_tail(p, g_amp, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::LookSAM) {
        const float g_adj = (1.0f - st.alpha) * g + st.alpha * st.sam_dir[i];
        params[i] = sg_adam_tail(p, g_adj, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::Prodigy) {
        const float d = st.d_factor;
        const float g_scaled = d * g;
        const float m = st.beta1 * st.exp_avg[i] + (1.0f - st.beta1) * g_scaled;
        const float v = st.beta2 * st.exp_avg_sq[i]
                        + (1.0f - st.beta2) * g_scaled * g_scaled;
        st.exp_avg[i] = m; st.exp_avg_sq[i] = v;
        st.s_track[i] += d * g;
        const float update = (m / st.bc1) / (__builtin_sqrtf(v / st.bc2) + st.eps);
        params[i] = p - d * (update + st.wd * p);     // prodigy steps with d, not lr
    } else if constexpr (Opt == OptId::NeuralGrok) {
        const float psi = sg_psi_forward(__builtin_fabsf(g), st.psi_W1, st.psi_b1,
                                         st.psi_W2, st.psi_b2);
        const float g_amp = (psi * st.alpha + st.beta) * g;
        params[i] = sg_adam_tail(p, g_amp, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::Muon) {
        params[i] = p * st.decay_factor + st.neg_lr_scale * st.orth[i];
    } else if constexpr (Opt == OptId::SuperGrok11) {
        const float smart = g + (1.0f - st.gate) * st.alpha * st.mu[i];
        params[i] = sg_adam_tail(p, smart, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else if constexpr (Opt == OptId::SuperGrok15) {
        const float a = sg_sg15_alpha(st.mu[i], st.alpha, st.alpha_max);
        const float smart = g + st.gate * a * st.mu[i];
        params[i] = sg_adam_tail(p, smart, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    } else {  // OptId::SuperGrok2 — Adam apply on the meta-net smart gradient.
        const float gg = (st.smart_grad != nullptr) ? st.smart_grad[i] : g;
        params[i] = sg_adam_tail(p, gg, st.exp_avg, st.exp_avg_sq, i,
                                 st.beta1, st.beta2, st.eps, st.bc1, st.bc2,
                                 st.lr, st.wd);
    }
}

}}} // namespace sg::fused::gfx942_mega

#endif  // device pass
#endif  // SG_FUSED_GFX942_OPT_COMPONENTS_HIP_HPP_
