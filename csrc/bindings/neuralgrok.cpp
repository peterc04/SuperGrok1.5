// bindings/neuralgrok.cpp — runtime dispatch to per-arch NeuralGrok launchers.
//
// TODO(structural-refactor): the amplifier MLP launchers vary by hidden-dim
// templating; the current bindings expose the generic untemplated entry
// point. Templated specializations (H=16/32/64/128) are dispatched inside
// the per-arch launcher.

#include "_dispatch_macro.h"
#include "_helpers.h"

#include <cmath>
#include <vector>

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
        void launch_fused_neuralgrok_full_step(                               \
            torch::Tensor param, torch::Tensor exp_avg,                       \
            torch::Tensor exp_avg_sq, torch::Tensor grad,                     \
            torch::Tensor W1, torch::Tensor b1,                               \
            torch::Tensor W2, torch::Tensor b2,                               \
            float alpha_amp, float beta_amp, int hidden_dim,                  \
            float beta1, float beta2, float lr, float weight_decay,           \
            float eps, float bc1, float bc2);                                 \
    }

DECLARE_NEURALGROK(sm80) DECLARE_NEURALGROK(sm90)
DECLARE_NEURALGROK(sm100) DECLARE_NEURALGROK(gfx942)
DECLARE_NEURALGROK(sm89) DECLARE_NEURALGROK(sm103) DECLARE_NEURALGROK(sm120) DECLARE_NEURALGROK(gfx950)
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

// Pre-refactor csrc/common/ops.cpp::neuralgrok_fused_step (full single
// kernel: amplifier MLP + Adam fused per-param).
void neuralgrok_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<int64_t>& steps,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float alpha_amp, float beta_amp, int hidden_dim,
    float beta1, float beta2, float lr, float wd,
    float eps, float grad_clip_norm
) {
    const size_t n_params = params.size();
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));
        SG_DISPATCH_CALL(launch_fused_neuralgrok_full_step,
            params[i], exp_avgs[i], exp_avg_sqs[i], grads[i],
            W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim,
            beta1, beta2, lr, wd, eps, bc1, bc2);
    }
}

} // namespace sg
