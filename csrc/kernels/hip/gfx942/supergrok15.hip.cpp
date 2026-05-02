// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok15.hip.cpp
//
//  gfx942 SuperGrok v1.5 launchers. Algorithm mirrors sm_90; uses ATen
//  tensor ops (which route through rocBLAS for matmuls on HIP).
// =====================================================================

#include "csrc/kernels/hip/gfx942/_common.hip.h"
#include "csrc/kernels/hip/gfx942/supergrok15.hip.h"

namespace sg { namespace gfx942 {

void launch_fused_supergrok15_full_step(
    torch::Tensor param,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor mu, torch::Tensor grad,
    torch::Tensor sharpness, float alpha,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, float lamb_eff,
    float beta1, float beta2,
    float lr, float wd_eff, float eps,
    float bc1, float bc2, int hidden_dim
) {
    (void)hidden_dim;

    // mu EMA
    mu.mul_(1.0f - alpha).add_(grad, alpha);

    auto g_flat = grad.flatten().to(torch::kFloat32);
    int64_t N = g_flat.size(0);

    float s_val = sharpness.item<float>();
    auto inp = torch::empty({N, 2}, g_flat.options());
    inp.select(1, 0).copy_(g_flat);
    inp.select(1, 1).fill_(s_val);

    auto W1_f = W1.to(torch::kFloat32);
    auto b1_f = b1.to(torch::kFloat32);
    auto W2_f = W2.to(torch::kFloat32);
    auto b2_f = b2.to(torch::kFloat32);

    auto hidden     = torch::addmm(b1_f, inp, W1_f.t()).gelu();
    auto out        = torch::addmm(b2_f, hidden, W2_f.t());
    auto smart_grad = out.reshape_as(param).mul_(rescale);

    exp_avg.mul_(beta1).add_(smart_grad, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(smart_grad, smart_grad, 1.0f - beta2);

    auto m_hat = exp_avg / bc1;
    auto v_hat = exp_avg_sq / bc2;
    auto update = m_hat / (v_hat.sqrt() + eps);

    param.mul_(1.0f - lr * wd_eff).add_(update, -lr * lamb_eff);
}

void launch_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    param.add_(grad, rho_over_norm);
}

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness,
    torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    sharpness.copy_((sam_grad - normal_grad).abs());
    param.copy_(backup);
}

}} // namespace sg::gfx942
