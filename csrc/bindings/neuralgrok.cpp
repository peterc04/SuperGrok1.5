// bindings/neuralgrok.cpp — runtime dispatch to per-arch NeuralGrok launchers.
//
// TODO(structural-refactor): the amplifier MLP launchers vary by hidden-dim
// templating; the current bindings expose the generic untemplated entry
// point. Templated specializations (H=16/32/64/128) are dispatched inside
// the per-arch launcher.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_NEURALGROK(NS)                                                \
    namespace NS {                                                            \
        void launch_fused_neuralgrok_amplifier(                               \
            torch::Tensor grad, torch::Tensor amplified,                      \
            torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,           \
            torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,           \
            int hidden_dim, float alpha, float beta);                         \
        void launch_fused_neuralgrok_adam(                                    \
            torch::Tensor param, torch::Tensor exp_avg,                       \
            torch::Tensor exp_avg_sq, torch::Tensor amplified_grad,           \
            float beta1, float beta2,                                         \
            float lr, float weight_decay,                                     \
            float eps, float bc1, float bc2);                                 \
    }

DECLARE_NEURALGROK(sm80) DECLARE_NEURALGROK(sm90)
DECLARE_NEURALGROK(sm100) DECLARE_NEURALGROK(gfx942)
#undef DECLARE_NEURALGROK

void neuralgrok_amplifier(
    torch::Tensor grad, torch::Tensor amplified,
    torch::Tensor amplifier_w1, torch::Tensor amplifier_b1,
    torch::Tensor amplifier_w2, torch::Tensor amplifier_b2,
    int hidden_dim, float alpha, float beta)
{
    SG_DISPATCH(launch_fused_neuralgrok_amplifier,
        grad, amplified, amplifier_w1, amplifier_b1,
        amplifier_w2, amplifier_b2, hidden_dim, alpha, beta);
}

void neuralgrok_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor amplified_grad,
    float beta1, float beta2, float lr, float weight_decay,
    float eps, float bc1, float bc2)
{
    SG_DISPATCH(launch_fused_neuralgrok_adam,
        param, exp_avg, exp_avg_sq, amplified_grad,
        beta1, beta2, lr, weight_decay, eps, bc1, bc2);
}

} // namespace sg
