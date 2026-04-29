// bindings/grokfast.cpp — runtime dispatch to per-arch Grokfast launchers.
#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_GROKFAST(NS)                                                  \
    namespace NS {                                                            \
        void launch_fused_grokfast_ema(                                       \
            torch::Tensor grad, torch::Tensor ema,                            \
            float alpha, float lamb);                                         \
        void launch_fused_grokfast_adam(                                      \
            torch::Tensor param, torch::Tensor exp_avg,                       \
            torch::Tensor exp_avg_sq, torch::Tensor ema,                      \
            torch::Tensor grad,                                               \
            float alpha, float lamb,                                          \
            float beta1, float beta2,                                         \
            float lr, float weight_decay,                                     \
            float eps, float bc1, float bc2);                                 \
    }

DECLARE_GROKFAST(sm80) DECLARE_GROKFAST(sm90)
DECLARE_GROKFAST(sm100) DECLARE_GROKFAST(gfx942)
#undef DECLARE_GROKFAST

void grokfast_ema(torch::Tensor grad, torch::Tensor ema, float alpha, float lamb) {
    SG_DISPATCH(launch_fused_grokfast_ema, grad, ema, alpha, lamb);
}

void grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2)
{
    SG_DISPATCH(launch_fused_grokfast_adam,
        param, exp_avg, exp_avg_sq, ema, grad,
        alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

} // namespace sg
