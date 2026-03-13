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

    // Gradient clipping (global norm)
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0)
                    grads[i].mul_(clip_coef);
            }
        }
    }

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

        auto smart_grad = torch::empty_like(params[i]);

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_mu_metanet(
                mus[i], grads[i], sharpness_cache[i], smart_grad, alpha,
                W1, b1, W2, b2, rescale, hidden_dim);
            launch_fused_adam_decay(
                params[i], exp_avgs[i], exp_avg_sqs[i], smart_grad, mus[i],
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
            continue;
        }
#endif
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
    float total_norm_sq = 0.0f;
    for (size_t i = 0; i < grads.size(); i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            float n = grads[i].norm().item<float>();
            total_norm_sq += n * n;
        }
    }
    float grad_norm = std::sqrt(total_norm_sq) + 1e-12f;
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
//  SuperGrok v2 — Main Step (DSA meta-net)
// ═══════════════════════════════════════════════════════════════════════

void supergrok2_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mus,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<int64_t>& steps,
    std::vector<float>& layer_alphas,
    std::vector<float>& layer_beta1s,
    torch::Tensor W_q, torch::Tensor b_q,
    torch::Tensor W_k, torch::Tensor b_k,
    torch::Tensor W_v, torch::Tensor b_v,
    torch::Tensor W_iq, torch::Tensor W_ik,
    torch::Tensor w_idx,
    torch::Tensor W_out, torch::Tensor b_out,
    float rescale, int d_head, int n_idx_heads, int top_k,
    float beta2, float lr, float wd_eff, float eps,
    float lamb, float ramp, float gate_signal,
    float grad_clip_norm
) {
    const size_t n_params = params.size();

    // Gradient clipping
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0)
                    grads[i].mul_(clip_coef);
            }
        }
    }

    float lamb_eff = 0.0f;
    if (ramp > 0.0f) {
        lamb_eff = ramp * gate_signal * lamb;
    }

    // Pre-allocate DSA buffers outside the per-parameter loop
    int max_N = 0;
    for (size_t i = 0; i < n_params; i++) {
        if (grads[i].defined() && grads[i].numel() > 0)
            max_N = std::max(max_N, (int)grads[i].numel());
    }

    torch::Tensor Q_buf, K_buf, V_buf, idx_q_buf, idx_k_buf, selected_buf;
#ifdef WITH_CUDA
    if (max_N > 0 && n_params > 0 && params[0].is_cuda()) {
        auto dev = params[0].device();
        auto opts_f = torch::TensorOptions().device(dev).dtype(torch::kFloat32);
        Q_buf = torch::empty({max_N, d_head}, opts_f);
        K_buf = torch::empty({max_N, d_head}, opts_f);
        V_buf = torch::empty({max_N, d_head}, opts_f);
        idx_q_buf = torch::empty({max_N, n_idx_heads}, opts_f);
        idx_k_buf = torch::empty({max_N, n_idx_heads}, opts_f);
        int eff_top_k_max = std::min(top_k, max_N);
        selected_buf = torch::empty({max_N, eff_top_k_max}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
    }
#endif

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        steps[i] += 1;
        int64_t step = steps[i];
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

        // mu EMA update
        mus[i].mul_(alpha).add_(grads[i], 1.0f - alpha);

        int N = grads[i].numel();
        int eff_top_k = std::min(top_k, N);
        auto smart_grad = torch::empty_like(params[i]);

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            // Use pre-allocated buffers with narrow views
            auto Q = Q_buf.narrow(0, 0, N);
            auto K = K_buf.narrow(0, 0, N);
            auto V = V_buf.narrow(0, 0, N);
            auto idx_q = idx_q_buf.narrow(0, 0, N);
            auto idx_k = idx_k_buf.narrow(0, 0, N);
            auto selected = selected_buf.narrow(0, 0, N).narrow(1, 0, eff_top_k);

            // Step 1: Project to Q, K, V, idx_q, idx_k
            launch_dsa_project(
                grads[i], sharpness_cache[i],
                Q, K, V, idx_q, idx_k,
                W_q, b_q, W_k, b_k, W_v, b_v,
                W_iq, W_ik, d_head, n_idx_heads);

            // Step 2: Lightning indexer + top-k selection
            launch_dsa_indexer_topk(
                idx_q, idx_k, w_idx, selected,
                n_idx_heads, eff_top_k);

            // Step 3: Sparse attention + skip connection → smart_grad
            launch_dsa_sparse_attention(
                Q, K, V, selected,
                grads[i], smart_grad,
                W_out, b_out,
                rescale, d_head, eff_top_k);

            // Step 4: Adam update (reuse v1.5 kernel)
            launch_fused_adam_decay(
                params[i], exp_avgs[i], exp_avg_sqs[i], smart_grad, mus[i],
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback: simplified dense attention (for small N)
        auto g_f = grads[i].reshape({-1, 1}).to(torch::kFloat32);
        auto s_f = sharpness_cache[i].reshape({-1, 1}).to(torch::kFloat32);
        auto inp = torch::cat({g_f, s_f}, 1);  // [N, 2]

        auto q = torch::addmm(b_q.to(torch::kFloat32), inp, W_q.to(torch::kFloat32).t());  // [N, d_head]
        auto k = torch::addmm(b_k.to(torch::kFloat32), inp, W_k.to(torch::kFloat32).t());
        auto v = torch::addmm(b_v.to(torch::kFloat32), inp, W_v.to(torch::kFloat32).t());

        // Dense attention (CPU fallback ignores sparsity for simplicity)
        float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
        auto scores = torch::mm(q, k.t()) * scale;
        auto attn = torch::softmax(scores, -1);
        auto context = torch::mm(attn, v);

        // Project back to scalar
        auto correction = torch::addmm(b_out.to(torch::kFloat32), context, W_out.to(torch::kFloat32).t());
        auto result = g_f + rescale * correction;
        smart_grad.copy_(result.reshape(grads[i].sizes()));

        // Adam update
        auto fg = smart_grad.to(torch::kFloat32) + lamb_eff * mus[i].to(torch::kFloat32);
        exp_avgs[i].mul_(beta1).add_(fg, 1.0f - beta1);
        exp_avg_sqs[i].mul_(beta2).addcmul_(fg, fg, 1.0f - beta2);
        float step_size = lr / bc1;
        auto denom = (exp_avg_sqs[i] / bc2).sqrt_().add_(eps);
        params[i].mul_(1.0f - lr * wd_eff);
        params[i].addcdiv_(exp_avgs[i], denom, -step_size);
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

    // Gradient clipping
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0)
                    grads[i].mul_(clip_coef);
            }
        }
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

        auto smart_grad = torch::empty_like(params[i]);

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_sg11_mu_metanet(
                mus[i], grads[i], sharpness_cache[i], smart_grad, alpha,
                W1, b1, W2, b2, rescale, hidden_dim);

            // Per-parameter cosine gating
            float gate = compute_cosine_gate(smart_grad, mus[i], gate_temperature);
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

        // Cosine gating
        auto sg_flat = smart_grad.reshape(-1).to(torch::kFloat32);
        auto mu_flat = mus[i].reshape(-1).to(torch::kFloat32);
        float dot_val = torch::dot(sg_flat, mu_flat).item<float>();
        float sg_norm = sg_flat.norm().item<float>();
        float mu_norm = mu_flat.norm().item<float>();
        float cos_sim = dot_val / (sg_norm * mu_norm + 1e-8f);
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

    // Gradient clipping
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0)
                    grads[i].mul_(clip_coef);
            }
        }
    }

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        steps[i] += 1;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_grokadamw_step(
                params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], emas[i],
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

    // Gradient clipping
    if (grad_clip_norm > 0.0f) {
        float total_norm_sq = 0.0f;
        for (size_t i = 0; i < n_params; i++) {
            if (grads[i].defined() && grads[i].numel() > 0) {
                float norm = grads[i].norm().item<float>();
                total_norm_sq += norm * norm;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        if (total_norm > grad_clip_norm) {
            float clip_coef = grad_clip_norm / (total_norm + 1e-6f);
            for (size_t i = 0; i < n_params; i++) {
                if (grads[i].defined() && grads[i].numel() > 0)
                    grads[i].mul_(clip_coef);
            }
        }
    }

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        steps[i] += 1;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

        auto amplified_grad = torch::empty_like(params[i]);

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_neuralgrok_amplifier(
                grads[i], amplified_grad,
                W1, b1, W2, b2, alpha_amp, beta_amp, hidden_dim);
            launch_fused_neuralgrok_adam(
                params[i], exp_avgs[i], exp_avg_sqs[i], amplified_grad,
                beta1, beta2, lr, wd, eps, bc1, bc2);
            continue;
        }
#endif
        // CPU fallback
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

    // Step 1: Compute d_lr update (reduction across all params)
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            auto num_out = torch::zeros({1}, torch::TensorOptions().device(params[i].device()).dtype(torch::kFloat32));
            auto den_out = torch::zeros({1}, torch::TensorOptions().device(params[i].device()).dtype(torch::kFloat32));
            launch_prodigy_dlr_reduce(
                grads[i], params[i], param_inits[i], s_bufs[i],
                num_out, den_out);
            numerator += num_out.item<float>();
            denominator += den_out.item<float>();
            continue;
        }
#endif
        auto g_f = grads[i].to(torch::kFloat32);
        auto diff = (params[i] - param_inits[i]).to(torch::kFloat32);
        numerator += torch::dot(g_f.flatten(), diff.flatten()).item<float>();
        denominator += s_bufs[i].to(torch::kFloat32).sum().item<float>();
    }

    // Update d_lr
    if (denominator > 1e-30f) {
        d_lr = std::max(d_lr, std::abs(numerator) / (denominator + 1e-12f));
    }

    // Step 2: Per-parameter Adam update with d_lr
    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        steps[i] += 1;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(steps[i]));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(steps[i]));

#ifdef WITH_CUDA
        if (params[i].is_cuda()) {
            launch_fused_prodigy_step(
                params[i], grads[i], exp_avgs[i], exp_avg_sqs[i], s_bufs[i],
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
    float total_norm_sq = 0.0f;
    for (size_t i = 0; i < grads.size(); i++) {
        if (grads[i].defined() && grads[i].numel() > 0) {
            float n = grads[i].norm().item<float>();
            total_norm_sq += n * n;
        }
    }
    float grad_norm = std::sqrt(total_norm_sq) + 1e-12f;
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

    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        auto& p = params[i];
        auto& g = grads[i];
        auto& buf = bufs[i];

        buf.mul_(momentum).add_(g);

        float buf_norm = buf.norm().item<float>() + 1e-7f;
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

    // ── SuperGrok v2 ────────────────────────────────────────────────
    m.def("supergrok2_fused_step", &supergrok2_fused_step,
          "SuperGrok2: DSA meta-net + adam + progressive wd",
          py::arg("params"), py::arg("grads"),
          py::arg("exp_avgs"), py::arg("exp_avg_sqs"),
          py::arg("mus"), py::arg("sharpness_cache"),
          py::arg("steps"),
          py::arg("layer_alphas"), py::arg("layer_beta1s"),
          py::arg("W_q"), py::arg("b_q"),
          py::arg("W_k"), py::arg("b_k"),
          py::arg("W_v"), py::arg("b_v"),
          py::arg("W_iq"), py::arg("W_ik"),
          py::arg("w_idx"),
          py::arg("W_out"), py::arg("b_out"),
          py::arg("rescale"), py::arg("d_head"), py::arg("n_idx_heads"), py::arg("top_k"),
          py::arg("beta2"), py::arg("lr"), py::arg("wd_eff"), py::arg("eps"),
          py::arg("lamb"), py::arg("ramp"), py::arg("gate_signal"),
          py::arg("grad_clip_norm"));

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
}
