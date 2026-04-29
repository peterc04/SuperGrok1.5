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

#include <stdexcept>
#include <string>

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

#undef DISPATCH_GROKADAMW

} // namespace sg
