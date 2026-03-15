/*
 * Grokking Optimizers — C++ Operations
 *
 * High-level optimizer step functions that process all parameters in tight
 * C++ loops. Dispatches to CUDA kernels when available.
 *
 * Covers: SuperGrok v1.5, SuperGrok v2, SuperGrok v1.1, GrokAdamW,
 *         NeuralGrok, Prodigy, Grokfast, Lion, LookSAM, Muon
 *
 * All optimizer state (moments, EMA buffers) is kept in FP32.
 * Parameters and gradients can be FP32, FP16, or BF16.
 */

#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>


// ───────────────────────────────────────────────────────────────────────
//  Helper: device-side gradient clipping (single CPU sync instead of N)
// ───────────────────────────────────────────────────────────────────────
static void clip_grad_norms_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_params,
    float grad_clip_norm
) {
    if (grad_clip_norm <= 0.0f) return;

    // Find device from first valid grad
    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }

    // Accumulate norm^2 on device — all .norm() calls are async kernel launches
    auto norm_sq = torch::zeros({1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            norm_sq.add_(grads[i].norm().to(torch::kFloat32).pow(2));
        }
    }
    // Single CPU sync
    float total_norm = std::sqrt(norm_sq.item<float>());
    if (total_norm > grad_clip_norm) {
        float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0)
                grads[i].mul_(clip_coef);
        }
    }
}

// Helper: device-side SAM grad norm (single CPU sync instead of N)
static float compute_sam_grad_norm_device_side(
    std::vector<torch::Tensor>& grads,
    size_t n_grads
) {
    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }
    auto norm_sq = torch::zeros({1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    for (size_t i = 0; i < n_grads; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            norm_sq.add_(grads[i].norm().to(torch::kFloat32).pow(2));
        }
    }
    return std::sqrt(norm_sq.item<float>()) + 1e-12f;
}


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v1.5 — Main Step
// ═══════════════════════════════════════════════════════════════════════

void supergrok15_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mus,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<int64_t>& steps,
    std::vector<float>& layer_alphas,
    std::vector<float>& layer_beta1s,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float beta2, float lr, float wd_eff, float eps,
    float lamb, float ramp, float gate_signal,
    float grad_clip_norm
) {
    const size_t n_params = params.size();

    // Gradient clipping (device-side — single CPU sync)
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    float lamb_eff = 0.0f;
    if (ramp > 0.0f) {
        lamb_eff = ramp * gate_signal * lamb;
    }

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        steps[i] += 1;
        int64_t step = steps[i];
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_supergrok15_full_step(
                params[i], exp_avgs[i], exp_avg_sqs[i], mus[i],
                grads[i], sharpness_cache[i], alpha,
                W1, b1, W2, b2, rescale,
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2,
                hidden_dim);
            continue;
        }
#endif
        auto smart_grad = torch::empty_like(params[i]);
        // CPU fallback using ATen
        mus[i].mul_(alpha).add_(grads[i], 1.0f - alpha);
        auto shape = grads[i].sizes().vec();
        auto flat_g = grads[i].reshape({-1, 1}).to(torch::kFloat32);
        auto flat_s = sharpness_cache[i].reshape({-1, 1}).to(torch::kFloat32);
        auto inp = torch::cat({flat_g, flat_s}, 1);
        auto z = torch::addmm(b1.to(torch::kFloat32), inp, W1.to(torch::kFloat32).t());
        auto act = torch::gelu(z);
        auto out = torch::addmm(b2.to(torch::kFloat32), act, W2.to(torch::kFloat32).t());
        smart_grad.copy_((flat_g + rescale * out).reshape(shape));

        auto fg = smart_grad.to(torch::kFloat32) + lamb_eff * mus[i].to(torch::kFloat32);
        exp_avgs[i].mul_(beta1).add_(fg, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(fg, fg, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom = (exp_avg_sqs[i] / bc2).sqrt_().add_(eps);
        params[i].mul_(1.0f - lr * wd_eff);
        params[i].addcdiv_(exp_avgs[i], denom, -step_size);
    }
}

std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    // Device-side grad norm (single CPU sync)
    float grad_norm = compute_sam_grad_norm_device_side(grads, grads.size());
    float rho_over_norm = rho / grad_norm;

    std::vector<torch::Tensor> backups;
    backups.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].clone());
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_sam_perturb(params[i], grads[i], rho_over_norm);
            continue;
        }
#endif
        params[i].add_(grads[i], rho_over_norm);
    }
    return backups;
}

void supergrok15_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!sam_grads[i].defined() || !normal_grads[i].defined()
            || sam_grads[i].numel() == 0)
        {
            params[i].copy_(backups[i]);
            continue;
        }
#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_sharpness_restore(
                params[i], sharpness_cache[i], backups[i],
                sam_grads[i], normal_grads[i]);
            continue;
        }
#endif
        sharpness_cache[i] = (sam_grads[i] - normal_grads[i]).abs();
        params[i].copy_(backups[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v1.1 — Main Step (cosine gating)
// ═══════════════════════════════════════════════════════════════════════

void supergrok11_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mus,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<int64_t>& steps,
    std::vector<float>& layer_alphas,
    std::vector<float>& layer_beta1s,
    torch::Tensor W1, torch::Tensor b1,
    torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim,
    float beta2, float lr, float wd_eff, float eps,
    float lamb, float ramp, float gate_temperature,
    float grad_clip_norm
) {
    const size_t n_params = params.size();

    // Gradient clipping (device-side — single CPU sync)
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        steps[i] += 1;
        int64_t step = steps[i];
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

        auto smart_grad = torch::empty_like(params[i]);

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            // Step 1: Fused mu EMA + meta-net to get smart_grad (used for cosine gate)
            launch_sg11_mu_metanet(
                mus[i], grads[i], sharpness_cache[i], smart_grad, alpha,
                W1, b1, W2, b2, rescale, hidden_dim);

            // Step 2: Fused cosine gate reduction (single kernel, single CPU sync)
            float gate = compute_cosine_gate_fused(smart_grad, mus[i], gate_temperature);
            float lamb_eff = ramp > 0.0f ? ramp * gate * lamb : 0.0f;

            launch_sg11_adam_decay(
                params[i], exp_avgs[i], exp_avg_sqs[i], smart_grad, mus[i],
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback
        mus[i].mul_(alpha).add_(grads[i], 1.0f - alpha);
        auto shape = grads[i].sizes().vec();
        auto flat_g = grads[i].reshape({-1, 1}).to(torch::kFloat32);
        auto flat_s = sharpness_cache[i].reshape({-1, 1}).to(torch::kFloat32);
        auto inp = torch::cat({flat_g, flat_s}, 1);
        auto z = torch::addmm(b1.to(torch::kFloat32), inp, W1.to(torch::kFloat32).t());
        auto act = torch::gelu(z);
        auto out = torch::addmm(b2.to(torch::kFloat32), act, W2.to(torch::kFloat32).t());
        smart_grad.copy_((flat_g + rescale * out).reshape(shape));

        // Cosine gating (batched reduction — single sync)
        auto sg_flat = smart_grad.reshape(-1).to(torch::kFloat32);
        auto mu_flat = mus[i].reshape(-1).to(torch::kFloat32);
        auto vals = torch::stack({(sg_flat * mu_flat).sum(), sg_flat.norm(), mu_flat.norm()});
        float cos_sim = vals[0].item<float>() /
            (vals[1].item<float>() * vals[2].item<float>() + 1e-8f);
        float gate = 1.0f / (1.0f + std::exp(-gate_temperature * cos_sim));
        float lamb_eff = ramp > 0.0f ? ramp * gate * lamb : 0.0f;

        auto fg = smart_grad.to(torch::kFloat32) + lamb_eff * mus[i].to(torch::kFloat32);
        exp_avgs[i].mul_(beta1).add_(fg, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(fg, fg, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom_t = (exp_avg_sqs[i] / bc2).sqrt_().add_(eps);
        params[i].mul_(1.0f - lr * wd_eff);
        params[i].addcdiv_(exp_avgs[i], denom_t, -step_size);
    }
}

std::vector<torch::Tensor> supergrok11_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    // Reuse same logic as v1.5
    return supergrok15_sam_perturb_all(params, grads, rho);
}

void supergrok11_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads
) {
    supergrok15_sharpness_restore_all(params, sharpness_cache, backups, sam_grads, normal_grads);
}


// ═══════════════════════════════════════════════════════════════════════
//  GrokAdamW — Fused EMA Filter + Amplification + Adam
// ═══════════════════════════════════════════════════════════════════════

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

    // Gradient clipping (device-side — single CPU sync)
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        // step already incremented by Python caller
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_grokadamw_step(
                params[i], exp_avgs[i], exp_avg_sqs[i], emas[i], grads[i],
                alpha, lamb_grok, beta1, beta2, lr, wd, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback
        auto g_f = grads[i].to(torch::kFloat32);
        emas[i].mul_(alpha).add_(g_f, 1.0f - alpha);
        auto amplified = g_f + lamb_grok * emas[i];
        exp_avgs[i].mul_(beta1).add_(amplified, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(amplified, amplified, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom_t = (exp_avg_sqs[i] / bc2).sqrt_().add_(eps);
        params[i].mul_(1.0f - lr * wd);
        params[i].addcdiv_(exp_avgs[i], denom_t, -step_size);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  NeuralGrok — Fused MLP Amplifier + Adam
// ═══════════════════════════════════════════════════════════════════════

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

    // Gradient clipping (device-side — single CPU sync)
    clip_grad_norms_device_side(grads, n_params, grad_clip_norm);

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        // step already incremented by Python caller
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            // Single fused kernel: amplifier + adam in one pass
            // (amplified_grad stays in registers, never hits GMEM)
            launch_fused_neuralgrok_full_step(
                params[i], exp_avgs[i], exp_avg_sqs[i], grads[i],
                W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim,
                beta1, beta2, lr, wd, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback
        auto amplified_grad = torch::empty_like(params[i]);
        auto g_f = grads[i].reshape({-1, 1}).to(torch::kFloat32);
        auto z = torch::addmm(b1.to(torch::kFloat32), g_f, W1.to(torch::kFloat32).t());
        auto act = torch::relu(z);
        auto scale = torch::addmm(b2.to(torch::kFloat32), act, W2.to(torch::kFloat32).t());
        auto amp = g_f * (alpha_amp * scale + beta_amp);
        amplified_grad.copy_(amp.reshape(grads[i].sizes()));

        auto amp_f = amplified_grad.to(torch::kFloat32);
        exp_avgs[i].mul_(beta1).add_(amp_f, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(amp_f, amp_f, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom_t = (exp_avg_sqs[i] / bc2).sqrt_().add_(eps);
        params[i].mul_(1.0f - lr * wd);
        params[i].addcdiv_(exp_avgs[i], denom_t, -step_size);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Prodigy — Distance-Aware Self-Tuning Adam
// ═══════════════════════════════════════════════════════════════════════

float prodigy_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_bufs,
    std::vector<torch::Tensor>& param_inits,
    std::vector<int64_t>& steps,
    float d_lr,
    float beta1, float beta2, float lr, float wd,
    float eps
) {
    const size_t n_params = params.size();

    // Step 1: Compute d_lr update (device-side accumulation — single CPU sync)
    // Find device
    torch::Device dev(torch::kCPU);
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            dev = grads[i].device();
            break;
        }
    }

    auto num_acc = torch::zeros({1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto den_acc = torch::zeros({1}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_prodigy_dlr_reduce(
                grads[i], params[i], param_inits[i], s_bufs[i],
                num_acc, den_acc);
            continue;
        }
#endif
        auto g_f = grads[i].to(torch::kFloat32);
        auto diff = (params[i] - param_inits[i]).to(torch::kFloat32);
        num_acc.add_((g_f.flatten() * diff.flatten()).sum());
        den_acc.add_(s_bufs[i].to(torch::kFloat32).sum());
    }

    // Single CPU sync for both values
    auto results = torch::cat({num_acc, den_acc}).cpu();
    float numerator = results[0].item<float>();
    float denominator = results[1].item<float>();

    // Update d_lr
    if (denominator > 1e-30f) {
        d_lr = std::max(d_lr, std::abs(numerator) / (denominator + 1e-12f));
    }

    // Step 2: Per-parameter Adam update with d_lr
    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        // step already incremented by Python caller
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_prodigy_step(
                params[i], exp_avgs[i], exp_avg_sqs[i], s_bufs[i], grads[i],
                d_lr, beta1, beta2, lr, wd, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback
        auto g_f = grads[i].to(torch::kFloat32);
        s_bufs[i].mul_(beta2).addcmul_(g_f * d_lr, g_f * d_lr, 1.0f - beta2);
        exp_avgs[i].mul_(beta1).add_(g_f * d_lr, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(g_f * d_lr, g_f * d_lr, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom_t = (exp_avg_sqs[i] / bc2).sqrt_().add_(d_lr * eps);
        params[i].mul_(1.0f - lr * d_lr * wd);
        params[i].addcdiv_(exp_avgs[i], denom_t, -step_size);
    }

    return d_lr;
}


// ═══════════════════════════════════════════════════════════════════════
//  Grokfast — Fused EMA + Amplification
// ═══════════════════════════════════════════════════════════════════════

void grokfast_fused_step(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha,
    float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
#ifdef WITH_CUDA
        if (grads[i].is_cuda()) {
            launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
            continue;
        }
#endif
        auto g_f = grads[i].to(torch::kFloat32);
        ema_bufs[i].mul_(alpha).add_(g_f, 1.0f - alpha);
        grads[i].copy_(g_f + lamb * ema_bufs[i]);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Lion — Fused Step
// ═══════════════════════════════════════════════════════════════════════

void lion_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    float lr,
    float beta1,
    float beta2,
    float wd
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_lion_step(params[i], exp_avgs[i], grads[i],
                                   lr, beta1, beta2, wd);
            continue;
        }
#endif
        auto g_f = grads[i].to(torch::kFloat32);
        auto interp = beta1 * exp_avgs[i] + (1.0f - beta1) * g_f;
        params[i].add_(interp.sign_().add_(params[i], wd), -lr);
        exp_avgs[i].mul_(beta2).add_(g_f, 1.0f - beta2);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  LookSAM — Fused Operations
// ═══════════════════════════════════════════════════════════════════════

std::vector<torch::Tensor> looksam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho
) {
    // Device-side grad norm (single CPU sync)
    float grad_norm = compute_sam_grad_norm_device_side(grads, grads.size());
    float rho_over_norm = rho / grad_norm;

    std::vector<torch::Tensor> backups;
    backups.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        backups.push_back(params[i].clone());
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_looksam_perturb(params[i], grads[i], rho_over_norm);
            continue;
        }
#endif
        params[i].add_(grads[i], rho_over_norm);
    }
    return backups;
}

void looksam_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++) {
#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_looksam_restore(params[i], backups[i]);
            continue;
        }
#endif
        params[i].copy_(backups[i]);
    }
}

void looksam_compute_directions(
    std::vector<torch::Tensor>& v_dirs,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads
) {
    for (size_t i = 0; i < v_dirs.size(); i++) {
        if (!sam_grads[i].defined() || !normal_grads[i].defined()
            || sam_grads[i].numel() == 0) continue;

        auto diff = (sam_grads[i] - normal_grads[i]).to(torch::kFloat32);
        float norm = diff.norm().item<float>();
        if (norm < 1e-12f) continue;
        float inv_norm = 1.0f / norm;

#ifdef WITH_CUDA
        if (v_dirs[i].is_cuda()) {
            auto sg_f = sam_grads[i].to(v_dirs[i].dtype());
            auto ng_f = normal_grads[i].to(v_dirs[i].dtype());
            launch_looksam_direction(v_dirs[i], sg_f, ng_f, inv_norm);
            continue;
        }
#endif
        v_dirs[i] = diff * inv_norm;
    }
}

void looksam_adjust_grads(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& v_dirs,
    float la
) {
    for (size_t i = 0; i < grads.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        if (!v_dirs[i].defined() || v_dirs[i].numel() == 0) continue;

        float grad_norm = grads[i].norm().item<float>();
        float la_times_gnorm = la * grad_norm;

#ifdef WITH_CUDA
        if (grads[i].is_cuda()) {
            auto vd_typed = v_dirs[i].to(grads[i].dtype());
            launch_looksam_adjust(grads[i], vd_typed, la_times_gnorm);
            continue;
        }
#endif
        grads[i].add_(v_dirs[i], la_times_gnorm);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Muon — Fused Newton-Schulz Step
// ═══════════════════════════════════════════════════════════════════════

void muon_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& bufs,
    float momentum,
    float lr,
    float wd,
    int ns_steps
) {
    constexpr float NS_A = 3.4445f;
    constexpr float NS_B = -4.7750f;
    constexpr float NS_C = 2.0315f;

    // Phase 1: Update all momentum buffers and compute norms asynchronously
    std::vector<torch::Tensor> norm_tensors;
    std::vector<size_t> valid_indices;
    norm_tensors.reserve(params.size());
    valid_indices.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        bufs[i].mul_(momentum).add_(grads[i]);
        norm_tensors.push_back(bufs[i].norm());  // async kernel launch
        valid_indices.push_back(i);
    }

    // Single CPU sync: stack all norms and transfer to CPU
    std::vector<float> norms_cpu;
    if (!norm_tensors.empty()) {
        auto norms_stacked = torch::stack(norm_tensors).cpu();
        auto norms_ptr = norms_stacked.data_ptr<float>();
        norms_cpu.assign(norms_ptr, norms_ptr + norm_tensors.size());
    }

    // Phase 2: Newton-Schulz iterations using pre-computed norms
    for (size_t vi = 0; vi < valid_indices.size(); vi++) {
        size_t i = valid_indices[vi];
        auto& p = params[i];
        auto& buf = bufs[i];

        float buf_norm = norms_cpu[vi] + 1e-7f;
        float inv_norm = 1.0f / buf_norm;

        auto X = buf * inv_norm;

#ifdef WITH_CUDA
        bool use_cuda = p.is_cuda();
#else
        bool use_cuda = false;
#endif

        for (int step = 0; step < ns_steps; step++) {
            auto A = torch::mm(X, X.t());
            auto AX = torch::mm(A, X);
            auto AAX = torch::mm(A, AX);

            if (use_cuda) {
#ifdef WITH_CUDA
                auto X_new = torch::empty_like(X);
                launch_muon_ns_combine(X_new, X, AX, AAX, NS_A, NS_B, NS_C);
                X = X_new;
#endif
            } else {
                X = NS_A * X + NS_B * AX + NS_C * AAX;
            }
        }

        int64_t rows = p.size(0);
        int64_t cols = p.size(1);
        float max_dim = static_cast<float>(std::max(rows, cols));
        float scale_factor = 0.2f * std::sqrt(max_dim);
        float neg_lr_scale = -lr * scale_factor / std::sqrt(max_dim);
        float decay_factor = 1.0f - lr * wd;

        if (use_cuda) {
#ifdef WITH_CUDA
            auto X_typed = X.to(p.dtype());
            launch_muon_update(p, X_typed, neg_lr_scale, decay_factor);
#endif
        } else {
            p.add_(X, neg_lr_scale);
            p.mul_(decay_factor);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v2 — Mamba-3+PEER Per-Parameter Step (dispatch to CUDA)
// ═══════════════════════════════════════════════════════════════════════

void supergrok2_mamba_peer_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    torch::Tensor mamba_fwd_state, torch::Tensor mamba_bwd_state,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    if (grad.numel() == 0) return;

#ifdef WITH_CUDA
    if (param.is_cuda()) {
        launch_mamba3_peer_step(
            param, grad, sharpness, exp_avg, exp_avg_sq, mu,
            gru_state, mamba_fwd_state, mamba_bwd_state,
            input_proj_W, input_proj_b,
            mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
            mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
            mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
            mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
            gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
            peer_query_Ws, prod_keys_A, prod_keys_B,
            expert_W1, expert_b1, expert_W2, expert_b2,
            rescale, alpha_mu, lamb_eff,
            beta1, beta2, lr, wd_eff, eps, bc1, bc2,
            d_model, d_state, d_inner,
            gru_hidden, num_heads, pk_dim,
            expert_hidden, num_experts,
            expert_counts);
        return;
    }
#endif
    throw std::runtime_error(
        "supergrok2_mamba_peer_step requires CUDA tensors. "
        "Use the Python meta-net fallback for CPU.");
}


void supergrok2_mamba_peer_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    std::vector<torch::Tensor> mamba_fwd_states,
    std::vector<torch::Tensor> mamba_bwd_states,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor mamba_fwd_in_proj, torch::Tensor mamba_fwd_dt_W,
    torch::Tensor mamba_fwd_dt_b, torch::Tensor mamba_fwd_B_proj,
    torch::Tensor mamba_fwd_C_proj, torch::Tensor mamba_fwd_A_log,
    torch::Tensor mamba_fwd_D, torch::Tensor mamba_fwd_rope,
    torch::Tensor mamba_fwd_out_proj,
    torch::Tensor mamba_bwd_in_proj, torch::Tensor mamba_bwd_dt_W,
    torch::Tensor mamba_bwd_dt_b, torch::Tensor mamba_bwd_B_proj,
    torch::Tensor mamba_bwd_C_proj, torch::Tensor mamba_bwd_A_log,
    torch::Tensor mamba_bwd_D, torch::Tensor mamba_bwd_rope,
    torch::Tensor mamba_bwd_out_proj,
    torch::Tensor gru_Wz, torch::Tensor gru_bz,
    torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s, std::vector<float> bc1s, std::vector<float> bc2s,
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int d_state, int d_inner,
    int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    torch::Tensor expert_counts
) {
    if (params.empty()) return;
#ifdef WITH_CUDA
    if (params[0].is_cuda()) {
        launch_mamba3_peer_batched_step(
            params, grads, sharpness_list, exp_avgs, exp_avg_sqs, mus,
            gru_states, mamba_fwd_states, mamba_bwd_states,
            input_proj_W, input_proj_b,
            mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
            mamba_fwd_D, mamba_fwd_rope, mamba_fwd_out_proj,
            mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
            mamba_bwd_D, mamba_bwd_rope, mamba_bwd_out_proj,
            gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
            peer_query_Ws, prod_keys_A, prod_keys_B,
            expert_W1, expert_b1, expert_W2, expert_b2,
            alpha_mus, lamb_effs, beta1s, bc1s, bc2s,
            rescale, beta2, lr, wd_eff, eps,
            d_model, d_state, d_inner,
            gru_hidden, num_heads, pk_dim,
            expert_hidden, num_experts,
            expert_counts);
        return;
    }
#endif
    throw std::runtime_error(
        "supergrok2_mamba_peer_batched_step requires CUDA tensors.");
}


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v2 — Bilevel Forward (with state saving) + Backward
// ═══════════════════════════════════════════════════════════════════════

// These are thin wrappers that dispatch to the CUDA launchers.
// The actual heavy lifting is in supergrok2_mamba_peer_backward_kernels.cu.
// No CPU fallback — bilevel always runs on CUDA.

// (The launch_mamba3_peer_bilevel_fwd_save and launch_mamba3_peer_backward
//  functions are called directly via pybind11 — no extra wrapper needed
//  since they already have the right signatures.)


// ═══════════════════════════════════════════════════════════════════════
//  pybind11 Bindings
// ═══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Grokking Optimizers — C++/CUDA fused operations for all optimizers";

    // ── SuperGrok v1.5 ───────────────────────────────────────────────
    m.def("supergrok15_fused_step", &supergrok15_fused_step,
          "SuperGrok1.5: fused mu + meta-net + gating + adam + progressive wd",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("mus"), py::arg("sharpness_cache"),
          py::arg("steps"),
          py::arg("layer_alphas"), py::arg("layer_beta1s"),
          py::arg("W1"), py::arg("b1"), py::arg("W2"), py::arg("b2"),
          py::arg("rescale"), py::arg("hidden_dim"),
          py::arg("beta2"), py::arg("lr"), py::arg("wd_eff"), py::arg("eps"),
          py::arg("lamb"), py::arg("ramp"), py::arg("gate_signal"),
          py::arg("grad_clip_norm"));

    m.def("supergrok15_sam_perturb_all", &supergrok15_sam_perturb_all,
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("supergrok15_sharpness_restore_all", &supergrok15_sharpness_restore_all,
          py::arg("params"), py::arg("sharpness_cache"),
          py::arg("backups"), py::arg("sam_grads"), py::arg("normal_grads"));

    // ── SuperGrok v1.1 ──────────────────────────────────────────────
    m.def("supergrok11_fused_step", &supergrok11_fused_step,
          "SuperGrok1.1: meta-net + cosine gating + adam + wd",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("mus"), py::arg("sharpness_cache"),
          py::arg("steps"),
          py::arg("layer_alphas"), py::arg("layer_beta1s"),
          py::arg("W1"), py::arg("b1"), py::arg("W2"), py::arg("b2"),
          py::arg("rescale"), py::arg("hidden_dim"),
          py::arg("beta2"), py::arg("lr"), py::arg("wd_eff"), py::arg("eps"),
          py::arg("lamb"), py::arg("ramp"), py::arg("gate_temperature"),
          py::arg("grad_clip_norm"));

    m.def("supergrok11_sam_perturb_all", &supergrok11_sam_perturb_all,
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("supergrok11_sharpness_restore_all", &supergrok11_sharpness_restore_all,
          py::arg("params"), py::arg("sharpness_cache"),
          py::arg("backups"), py::arg("sam_grads"), py::arg("normal_grads"));

    // ── GrokAdamW ───────────────────────────────────────────────────
    m.def("grokadamw_fused_step", &grokadamw_fused_step,
          "GrokAdamW: EMA filter + amplification + adam",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"), py::arg("emas"),
          py::arg("steps"),
          py::arg("alpha"), py::arg("lamb"),
          py::arg("beta1"), py::arg("beta2"), py::arg("lr"), py::arg("wd"),
          py::arg("eps"), py::arg("grad_clip_norm"));

    // ── NeuralGrok ──────────────────────────────────────────────────
    m.def("neuralgrok_fused_step", &neuralgrok_fused_step,
          "NeuralGrok: MLP gradient amplifier + adam",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("steps"),
          py::arg("W1"), py::arg("b1"), py::arg("W2"), py::arg("b2"),
          py::arg("alpha_amp"), py::arg("beta_amp"), py::arg("hidden_dim"),
          py::arg("beta1"), py::arg("beta2"), py::arg("lr"), py::arg("wd"),
          py::arg("eps"), py::arg("grad_clip_norm"));

    // ── Prodigy ─────────────────────────────────────────────────────
    m.def("prodigy_fused_step", &prodigy_fused_step,
          "Prodigy: distance-aware self-tuning adam",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("s_bufs"), py::arg("param_inits"),
          py::arg("steps"), py::arg("d_lr"),
          py::arg("beta1"), py::arg("beta2"), py::arg("lr"), py::arg("wd"),
          py::arg("eps"));

    // ── Grokfast ────────────────────────────────────────────────────
    m.def("grokfast_fused_step", &grokfast_fused_step,
          "Grokfast: fused EMA update + gradient amplification",
          py::arg("grads"), py::arg("ema_bufs"),
          py::arg("alpha"), py::arg("lamb"));

    // ── Lion ────────────────────────────────────────────────────────
    m.def("lion_fused_step", &lion_fused_step,
          "Lion: fused momentum interp + sign + update + decay",
          py::arg("params"), py::arg("grads"), py::arg("exp_avgs"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("wd"));

    // ── LookSAM ────────────────────────────────────────────────────
    m.def("looksam_perturb_all", &looksam_perturb_all,
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("looksam_restore_all", &looksam_restore_all,
          py::arg("params"), py::arg("backups"));

    m.def("looksam_compute_directions", &looksam_compute_directions,
          py::arg("v_dirs"), py::arg("sam_grads"), py::arg("normal_grads"));

    m.def("looksam_adjust_grads", &looksam_adjust_grads,
          py::arg("grads"), py::arg("v_dirs"), py::arg("la"));

    // ── Muon ────────────────────────────────────────────────────────
    m.def("muon_fused_step", &muon_fused_step,
          "Muon: momentum + Newton-Schulz ortho + update + WD",
          py::arg("params"), py::arg("grads"), py::arg("bufs"),
          py::arg("momentum"), py::arg("lr"), py::arg("wd"),
          py::arg("ns_steps"));

    // ── SuperGrok v2 Mamba-3+PEER ────────────────────────────────────
    m.def("supergrok2_mamba_peer_step", &supergrok2_mamba_peer_step,
          "SuperGrok2: per-param Mamba-3+PEER meta-net + mu + Adam + WD",
          py::arg("param"), py::arg("grad"), py::arg("sharpness"),
          py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("mu"),
          py::arg("gru_state"),
          py::arg("mamba_fwd_state"), py::arg("mamba_bwd_state"),
          py::arg("input_proj_W"), py::arg("input_proj_b"),
          py::arg("mamba_fwd_in_proj"), py::arg("mamba_fwd_dt_W"),
          py::arg("mamba_fwd_dt_b"), py::arg("mamba_fwd_B_proj"),
          py::arg("mamba_fwd_C_proj"), py::arg("mamba_fwd_A_log"),
          py::arg("mamba_fwd_D"), py::arg("mamba_fwd_rope"),
          py::arg("mamba_fwd_out_proj"),
          py::arg("mamba_bwd_in_proj"), py::arg("mamba_bwd_dt_W"),
          py::arg("mamba_bwd_dt_b"), py::arg("mamba_bwd_B_proj"),
          py::arg("mamba_bwd_C_proj"), py::arg("mamba_bwd_A_log"),
          py::arg("mamba_bwd_D"), py::arg("mamba_bwd_rope"),
          py::arg("mamba_bwd_out_proj"),
          py::arg("gru_Wz"), py::arg("gru_bz"),
          py::arg("gru_Wr"), py::arg("gru_br"),
          py::arg("gru_Wh"), py::arg("gru_bh"),
          py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
          py::arg("expert_W1"), py::arg("expert_b1"),
          py::arg("expert_W2"), py::arg("expert_b2"),
          py::arg("rescale"), py::arg("alpha_mu"), py::arg("lamb_eff"),
          py::arg("beta1"), py::arg("beta2"), py::arg("lr"),
          py::arg("wd_eff"), py::arg("eps"),
          py::arg("bc1"), py::arg("bc2"),
          py::arg("d_model"), py::arg("d_state"), py::arg("d_inner"),
          py::arg("gru_hidden"), py::arg("num_heads"), py::arg("pk_dim"),
          py::arg("expert_hidden"), py::arg("num_experts"),
          py::arg("expert_counts"));

    // ── SuperGrok v2 Batched Step ──────────────────────────────────────
    m.def("supergrok2_mamba_peer_batched_step", &supergrok2_mamba_peer_batched_step,
          "SuperGrok2: batched Mamba-3+PEER step for all params at once",
          py::arg("params"), py::arg("grads"), py::arg("sharpness_list"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"), py::arg("mus"),
          py::arg("gru_states"), py::arg("mamba_fwd_states"), py::arg("mamba_bwd_states"),
          py::arg("input_proj_W"), py::arg("input_proj_b"),
          py::arg("mamba_fwd_in_proj"), py::arg("mamba_fwd_dt_W"),
          py::arg("mamba_fwd_dt_b"), py::arg("mamba_fwd_B_proj"),
          py::arg("mamba_fwd_C_proj"), py::arg("mamba_fwd_A_log"),
          py::arg("mamba_fwd_D"), py::arg("mamba_fwd_rope"),
          py::arg("mamba_fwd_out_proj"),
          py::arg("mamba_bwd_in_proj"), py::arg("mamba_bwd_dt_W"),
          py::arg("mamba_bwd_dt_b"), py::arg("mamba_bwd_B_proj"),
          py::arg("mamba_bwd_C_proj"), py::arg("mamba_bwd_A_log"),
          py::arg("mamba_bwd_D"), py::arg("mamba_bwd_rope"),
          py::arg("mamba_bwd_out_proj"),
          py::arg("gru_Wz"), py::arg("gru_bz"),
          py::arg("gru_Wr"), py::arg("gru_br"),
          py::arg("gru_Wh"), py::arg("gru_bh"),
          py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
          py::arg("expert_W1"), py::arg("expert_b1"),
          py::arg("expert_W2"), py::arg("expert_b2"),
          py::arg("alpha_mus"), py::arg("lamb_effs"),
          py::arg("beta1s"), py::arg("bc1s"), py::arg("bc2s"),
          py::arg("rescale"), py::arg("beta2"), py::arg("lr"),
          py::arg("wd_eff"), py::arg("eps"),
          py::arg("d_model"), py::arg("d_state"), py::arg("d_inner"),
          py::arg("gru_hidden"), py::arg("num_heads"), py::arg("pk_dim"),
          py::arg("expert_hidden"), py::arg("num_experts"),
          py::arg("expert_counts"));

#ifdef WITH_CUDA
    // ── SuperGrok v2 Bilevel Forward (state-saving) ──────────────────
    m.def("supergrok2_bilevel_fwd_save", &launch_mamba3_peer_bilevel_fwd_save,
          "SuperGrok2 bilevel: forward scan with state saving for backward");

    // ── SuperGrok v2 Bilevel Backward ────────────────────────────────
    m.def("supergrok2_bilevel_backward", &launch_mamba3_peer_backward,
          "SuperGrok2 bilevel: full backward through meta-net");

    // ── SuperGrok v2 Batched Bilevel Forward-Save ─────────────────────
    m.def("supergrok2_bilevel_fwd_save_batched", &launch_mamba3_peer_bilevel_fwd_save_batched,
          "SuperGrok2 bilevel: batched forward scan with state saving");

    // ── SuperGrok v2 Batched Bilevel Backward ─────────────────────────
    m.def("supergrok2_bilevel_backward_batched", &launch_mamba3_peer_backward_batched,
          "SuperGrok2 bilevel: batched backward through scan");
#endif
}
