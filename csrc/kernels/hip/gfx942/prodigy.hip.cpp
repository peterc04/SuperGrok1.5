// =====================================================================
//  csrc/kernels/hip/gfx942/prodigy.hip.cpp — ATen-based.
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/prodigy.hip.h"

namespace sg { namespace gfx942 {

void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq, torch::Tensor s,
    torch::Tensor /*param_init*/, torch::Tensor grad,
    float lr, float d_lr,
    float beta1, float beta2,
    float weight_decay,
    float eps, float bc1, float bc2
) {
    (void)lr; // lr unused inside the per-element step (d_lr is the scaler).
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat &&
                exp_avg_sq.scalar_type() == at::kFloat &&
                s.scalar_type() == at::kFloat,
                "prodigy_step (gfx942): state tensors must be FP32");
    if (param.numel() == 0) return;

    auto p_f32 = param.to(torch::kFloat32);
    auto g_f32 = grad.to(torch::kFloat32);

    // m = beta1 * m + (1 - beta1) * g
    exp_avg.mul_(beta1).add_(g_f32, 1.0f - beta1);
    // v = beta2 * v + (1 - beta2) * g^2
    exp_avg_sq.mul_(beta2).addcmul_(g_f32, g_f32, 1.0f - beta2);
    // s = beta2 * s + (1 - beta2) * d_lr * g
    s.mul_(beta2).add_(g_f32, (1.0f - beta2) * d_lr);

    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto u = m_hat / (v_hat.sqrt() + eps) + weight_decay * p_f32;
    param.copy_((p_f32 - d_lr * u).to(param.scalar_type()));
}

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param,
    torch::Tensor param_init, torch::Tensor s,
    torch::Tensor numerator, torch::Tensor denominator,
    float eps
) {
    if (grad.numel() == 0) return;
    // numerator   += sum(grad * (param_init - param))
    // denominator += sum(|s|) + eps
    auto g  = grad.to(torch::kFloat32).reshape(-1);
    auto p  = param.to(torch::kFloat32).reshape(-1);
    auto pi = param_init.to(torch::kFloat32).reshape(-1);
    auto sf = s.to(torch::kFloat32).reshape(-1);

    auto num_contrib = (g * (pi - p)).sum().reshape({1});
    auto den_contrib = (sf.abs().sum() + eps).reshape({1});
    numerator.add_(num_contrib);
    denominator.add_(den_contrib);
}

void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<float>& bc1s, std::vector<float>& bc2s,
    torch::Tensor d_lr_buf,
    float beta1, float beta2, float lr, float wd, float eps
) {
    const size_t T = params.size();
    TORCH_CHECK(grads.size() == T && param_inits.size() == T &&
                exp_avgs.size() == T && exp_avg_sqs.size() == T &&
                s_bufs.size() == T && bc1s.size() == T && bc2s.size() == T,
                "prodigy multi-tensor (gfx942): vector size mismatch");
    if (T == 0) return;

    auto opts = torch::TensorOptions()
        .device(d_lr_buf.device()).dtype(torch::kFloat32);
    auto numerator   = torch::zeros({1}, opts);
    auto denominator = torch::zeros({1}, opts);

    for (size_t i = 0; i < T; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        launch_prodigy_dlr_reduce(grads[i], params[i], param_inits[i],
                                  s_bufs[i], numerator, denominator, eps);
    }

    const float num_v = numerator.item<float>();
    const float den_v = denominator.item<float>();
    const float d_lr_old = d_lr_buf.item<float>();
    float d_lr_new = d_lr_old;
    if (den_v > 0.0f) {
        const float candidate = lr * num_v / den_v;
        if (candidate > d_lr_old) d_lr_new = candidate;
    }
    d_lr_buf.fill_(d_lr_new);

    for (size_t i = 0; i < T; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        launch_fused_prodigy_step(
            params[i], exp_avgs[i], exp_avg_sqs[i], s_bufs[i],
            param_inits[i], grads[i],
            lr, d_lr_new, beta1, beta2, wd, eps, bc1s[i], bc2s[i]);
    }
}

}} // namespace sg::gfx942
