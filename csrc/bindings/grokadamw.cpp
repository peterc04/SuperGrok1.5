// =====================================================================
//  bindings/grokadamw.cpp — runtime dispatch to per-arch GrokAdamW launchers
//
//  Each arch has its launcher symbols inside a sg::<arch> namespace
//  defined in csrc/kernels/{cuda/<sm>,hip/gfx942}/grokadamw_<arch>.cu.
//  This dispatcher picks one based on detect_arch() and raises if the
//  detected arch is unsupported.
//
//  Pybind11 registration is done from csrc/bindings/module.cpp.
// =====================================================================

#include "bindings.h"
#include "_helpers.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// Forward-declare the launcher signatures inside each arch namespace.
// Signatures must match the definitions in the per-arch .cu files.
// ---------------------------------------------------------------------

namespace sg {

#define DECLARE_GROKADAMW_LAUNCHERS(NS)                              \
    namespace NS {                                                   \
        void launch_fused_grokadamw_step(                            \
            torch::Tensor param, torch::Tensor exp_avg,              \
            torch::Tensor exp_avg_sq, torch::Tensor ema,             \
            torch::Tensor grad,                                      \
            float alpha, float lamb,                                 \
            float beta1, float beta2,                                \
            float lr, float weight_decay,                            \
            float eps, float bc1, float bc2);                        \
        void launch_fused_grokadamw_clip_step(                       \
            torch::Tensor param, torch::Tensor exp_avg,              \
            torch::Tensor exp_avg_sq, torch::Tensor ema,             \
            torch::Tensor grad,                                      \
            float alpha, float lamb,                                 \
            float beta1, float beta2,                                \
            float lr, float weight_decay,                            \
            float eps, float bc1, float bc2,                         \
            float clip_threshold);                                   \
        void launch_fused_grokadamw_step_q3(                         \
            torch::Tensor param,                                     \
            torch::Tensor exp_avg_int8,                              \
            torch::Tensor exp_avg_scales,                            \
            torch::Tensor exp_avg_sq_bf16,                           \
            torch::Tensor ema_bf16,                                  \
            torch::Tensor grad,                                      \
            float alpha, float lamb,                                 \
            float beta1, float beta2,                                \
            float lr, float weight_decay,                            \
            float eps, float bc1, float bc2,                         \
            unsigned global_step);                                   \
    }

DECLARE_GROKADAMW_LAUNCHERS(sm80)
DECLARE_GROKADAMW_LAUNCHERS(sm90)
DECLARE_GROKADAMW_LAUNCHERS(sm100)
DECLARE_GROKADAMW_LAUNCHERS(gfx942)

DECLARE_GROKADAMW_LAUNCHERS(sm89) DECLARE_GROKADAMW_LAUNCHERS(sm103) DECLARE_GROKADAMW_LAUNCHERS(sm120) DECLARE_GROKADAMW_LAUNCHERS(gfx950)
#undef DECLARE_GROKADAMW_LAUNCHERS

// Multi-tensor variants live in csrc/kernels/<arch>/multi_tensor_<arch>.cu.
// One declaration per supported arch.
#define DECLARE_MT_GROKADAMW(NS)                                                \
    namespace NS {                                                              \
        void launch_multi_tensor_grokadamw(                                     \
            std::vector<torch::Tensor>& params,                                 \
            std::vector<torch::Tensor>& exp_avgs,                               \
            std::vector<torch::Tensor>& exp_avg_sqs,                            \
            std::vector<torch::Tensor>& emas,                                   \
            std::vector<torch::Tensor>& grads,                                  \
            std::vector<float>& bc1s, std::vector<float>& bc2s,                 \
            float alpha, float lamb, float beta1, float beta2,                  \
            float lr, float wd, float eps);                                     \
        void launch_fused_adamw_simple(                                         \
            std::vector<torch::Tensor>& params,                                 \
            std::vector<torch::Tensor>& exp_avgs,                               \
            std::vector<torch::Tensor>& exp_avg_sqs,                            \
            std::vector<torch::Tensor>& grads,                                  \
            std::vector<int64_t>& steps,                                        \
            float beta1, float beta2, float lr, float wd, float eps);           \
    }

DECLARE_MT_GROKADAMW(sm80) DECLARE_MT_GROKADAMW(sm89)
DECLARE_MT_GROKADAMW(sm90) DECLARE_MT_GROKADAMW(sm100)
DECLARE_MT_GROKADAMW(sm103) DECLARE_MT_GROKADAMW(sm120)
DECLARE_MT_GROKADAMW(gfx942) DECLARE_MT_GROKADAMW(gfx950)
#undef DECLARE_MT_GROKADAMW

// ---------------------------------------------------------------------
// Public entry points called from Python.
// ---------------------------------------------------------------------

#define DISPATCH_GROKADAMW(METHOD, ...)                                       \
    do {                                                                      \
        const int a = sg::detect_arch();                                      \
        switch (a) {                                                          \
            case 80:  return sg::sm80::METHOD(__VA_ARGS__);                   \
            case 89:  return sg::sm89::METHOD(__VA_ARGS__);                   \
            case 90:  return sg::sm90::METHOD(__VA_ARGS__);                   \
            case 100: return sg::sm100::METHOD(__VA_ARGS__);                  \
            case 103: return sg::sm103::METHOD(__VA_ARGS__);                  \
            case 120: return sg::sm120::METHOD(__VA_ARGS__);                  \
            case 942: return sg::gfx942::METHOD(__VA_ARGS__);                 \
            case 950: return sg::gfx950::METHOD(__VA_ARGS__);                 \
            default:                                                          \
                throw std::runtime_error(                                     \
                    "GrokAdamW dispatch: unsupported arch " +                 \
                    std::to_string(a));                                       \
        }                                                                     \
    } while (0)

void grokadamw_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2)
{
    DISPATCH_GROKADAMW(launch_fused_grokadamw_step,
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

void grokadamw_clip_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor ema,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    float clip_threshold)
{
    DISPATCH_GROKADAMW(launch_fused_grokadamw_clip_step,
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
        clip_threshold);
}

void grokadamw_step_q3(
    torch::Tensor param,
    torch::Tensor exp_avg_int8,
    torch::Tensor exp_avg_scales,
    torch::Tensor exp_avg_sq_bf16,
    torch::Tensor ema_bf16,
    torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2,
    float lr, float weight_decay,
    float eps, float bc1, float bc2,
    unsigned global_step)
{
    DISPATCH_GROKADAMW(launch_fused_grokadamw_step_q3,
        param, exp_avg_int8, exp_avg_scales, exp_avg_sq_bf16, ema_bf16,
        grad, alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2,
        global_step);
}

// ---------------------------------------------------------------------
// High-level vector-signature entry point — restored from pre-refactor
// csrc/common/ops.cpp::grokadamw_fused_step. Loops over per-param
// scalars (bc1, bc2) on the host, then calls the multi-tensor launcher
// in the arch-detected namespace.
// ---------------------------------------------------------------------
void grokadamw_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<int64_t>& steps,
    float alpha, float lamb_grok,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm
) {
    const size_t n_params = params.size();
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);
    if (n_params == 0) return;

    std::vector<torch::Tensor> vp, vg, vea, veas, vema;
    std::vector<float> bc1_vec, bc2_vec;
    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        steps[i] += 1;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
        vp.push_back(params[i]); vg.push_back(grads[i]);
        vea.push_back(exp_avgs[i]); veas.push_back(exp_avg_sqs[i]);
        vema.push_back(emas[i]);
        bc1_vec.push_back(bc1); bc2_vec.push_back(bc2);
    }
    if (vp.empty()) return;

    DISPATCH_GROKADAMW(launch_multi_tensor_grokadamw,
        vp, vea, veas, vema, vg, bc1_vec, bc2_vec,
        alpha, lamb_grok, beta1, beta2, lr, wd, eps);
}

// Shared simple AdamW (used by Muon and LookSAM 1D-param paths).
void fused_adamw_simple_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<int64_t>& steps,
    float beta1, float beta2, float lr, float wd, float eps
) {
    if (params.empty()) return;
    DISPATCH_GROKADAMW(launch_fused_adamw_simple,
        params, exp_avgs, exp_avg_sqs, grads, steps,
        beta1, beta2, lr, wd, eps);
}

#undef DISPATCH_GROKADAMW

} // namespace sg
