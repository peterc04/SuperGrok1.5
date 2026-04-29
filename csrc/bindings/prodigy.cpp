// bindings/prodigy.cpp — runtime dispatch to per-arch Prodigy launchers.
#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_PRODIGY(NS)                                                   \
    namespace NS {                                                            \
        void launch_fused_prodigy_step(                                       \
            torch::Tensor param, torch::Tensor exp_avg,                       \
            torch::Tensor exp_avg_sq, torch::Tensor s,                        \
            torch::Tensor param_init, torch::Tensor grad,                     \
            float lr, float d_lr,                                             \
            float beta1, float beta2,                                         \
            float weight_decay,                                               \
            float eps, float bc1, float bc2);                                 \
        void launch_prodigy_dlr_reduce(                                       \
            torch::Tensor grad, torch::Tensor param,                          \
            torch::Tensor param_init, torch::Tensor s,                        \
            torch::Tensor numerator, torch::Tensor denominator,               \
            float eps);                                                       \
    }

DECLARE_PRODIGY(sm80) DECLARE_PRODIGY(sm90)
DECLARE_PRODIGY(sm100) DECLARE_PRODIGY(gfx942)
DECLARE_PRODIGY(sm89) DECLARE_PRODIGY(sm103) DECLARE_PRODIGY(sm120) DECLARE_PRODIGY(gfx950)
#undef DECLARE_PRODIGY

void prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor s, torch::Tensor param_init, torch::Tensor grad,
    float lr, float d_lr, float beta1, float beta2, float weight_decay,
    float eps, float bc1, float bc2)
{
    SG_DISPATCH(launch_fused_prodigy_step,
        param, exp_avg, exp_avg_sq, s, param_init, grad,
        lr, d_lr, beta1, beta2, weight_decay, eps, bc1, bc2);
}

void prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init,
    torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator,
    float eps)
{
    SG_DISPATCH(launch_prodigy_dlr_reduce,
        grad, param, param_init, s, numerator, denominator, eps);
}

} // namespace sg
