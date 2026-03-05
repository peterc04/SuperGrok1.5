/*
 * Grokking Race — C++ Operations
 *
 * Eliminates Python overhead by processing all parameters in tight C++ loops.
 * Dispatches to CUDA kernels when available, falls back to ATen ops on CPU.
 *
 * Covers: SuperGrok v1.5, Grokfast, Lion, LookSAM, Muon
 */

#include "ops.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v1.5 — CPU Fallbacks
// ═══════════════════════════════════════════════════════════════════════

static void cpu_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int H
) {
    mu.mul_(alpha).add_(grad, 1.0f - alpha);
    auto shape = grad.sizes().vec();
    auto flat_g = grad.reshape({-1, 1});
    auto flat_s = sharpness.reshape({-1, 1});
    auto inp = torch::cat({flat_g, flat_s}, 1);
    auto z = torch::addmm(b1, inp, W1.t());
    auto act = torch::gelu(z);
    auto out = torch::addmm(b2, act, W2.t());
    auto result = flat_g + rescale * out;
    smart_grad.copy_(result.reshape(shape));
}

static void cpu_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2,
    float lr, float wd_eff, float eps, float bc1, float bc2
) {
    auto fg = smart_grad + lamb_eff * mu;
    exp_avg.mul_(beta1).add_(fg, 1.0f - beta1);
    exp_avg_sq.mul_(beta2).addcmul_(fg, fg, 1.0f - beta2);
    float step_size = lr / bc1;
    auto denom = (exp_avg_sq / bc2).sqrt_().add_(eps);
    param.mul_(1.0f - lr * wd_eff);
    param.addcdiv_(exp_avg, denom, -step_size);
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

    // Per-parameter update
    for (size_t i = 0; i < n_params; i++) {
        if (!grads[i].defined() || grads[i].numel() == 0)
            continue;

        auto& p = params[i];
        auto& g = grads[i];
        auto& ea = exp_avgs[i];
        auto& easq = exp_avg_sqs[i];
        auto& mu = mus[i];
        auto& sharp = sharpness_cache[i];

        steps[i] += 1;
        int64_t step = steps[i];
        float alpha = layer_alphas[i];
        float beta1 = layer_beta1s[i];
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

        auto smart_grad = torch::empty_like(p);

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = p.is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_mu_metanet(
                mu, g, sharp, smart_grad, alpha,
                W1, b1, W2, b2, rescale, hidden_dim);
#endif
        } else {
            cpu_mu_metanet(mu, g, sharp, smart_grad, alpha,
                           W1, b1, W2, b2, rescale, hidden_dim);
        }

        // Sigmoid gating (pre-computed from training accuracy in Python)
        float lamb_eff = 0.0f;
        if (ramp > 0.0f) {
            lamb_eff = ramp * gate_signal * lamb;
        }

        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_adam_decay(
                p, ea, easq, smart_grad, mu,
                lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
#endif
        } else {
            cpu_adam_decay(p, ea, easq, smart_grad, mu,
                          lamb_eff, beta1, beta2, lr, wd_eff, eps, bc1, bc2);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v1.5 — SAM Operations
// ═══════════════════════════════════════════════════════════════════════

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
        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_sam_perturb(params[i], grads[i], rho_over_norm);
#endif
        } else {
            params[i].add_(grads[i], rho_over_norm);
        }
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
        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_sharpness_restore(
                params[i], sharpness_cache[i], backups[i],
                sam_grads[i], normal_grads[i]);
#endif
        } else {
            sharpness_cache[i] = (sam_grads[i] - normal_grads[i]).abs();
            params[i].copy_(backups[i]);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Grokfast — Fused EMA + Amplification (all params in C++ loop)
// ═══════════════════════════════════════════════════════════════════════

void grokfast_fused_step(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha,
    float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = grads[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
#endif
        } else {
            // CPU fallback: EMA update + amplification
            ema_bufs[i].mul_(alpha).add_(grads[i], 1.0f - alpha);
            grads[i].add_(ema_bufs[i], lamb);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Lion — Fused Step (all params in C++ loop)
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

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_fused_lion_step(
                params[i], exp_avgs[i], grads[i],
                lr, beta1, beta2, wd);
#endif
        } else {
            // CPU fallback
            auto interp = beta1 * exp_avgs[i] + (1.0f - beta1) * grads[i];
            params[i].add_(interp.sign_().add_(params[i], wd), -lr);
            exp_avgs[i].mul_(beta2).add_(grads[i], 1.0f - beta2);
        }
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
    // Compute global gradient norm
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

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_looksam_perturb(params[i], grads[i], rho_over_norm);
#endif
        } else {
            params[i].add_(grads[i], rho_over_norm);
        }
    }
    return backups;
}

void looksam_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups
) {
    for (size_t i = 0; i < params.size(); i++) {
        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = params[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_looksam_restore(params[i], backups[i]);
#endif
        } else {
            params[i].copy_(backups[i]);
        }
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

        // Compute norm of difference
        auto diff = sam_grads[i] - normal_grads[i];
        float norm = diff.norm().item<float>();
        if (norm < 1e-12f) continue;
        float inv_norm = 1.0f / norm;

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = v_dirs[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_looksam_direction(v_dirs[i], sam_grads[i], normal_grads[i], inv_norm);
#endif
        } else {
            v_dirs[i] = diff * inv_norm;
        }
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

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = grads[i].is_cuda();
#endif
        if (use_cuda) {
#ifdef WITH_CUDA
            launch_looksam_adjust(grads[i], v_dirs[i], la_times_gnorm);
#endif
        } else {
            grads[i].add_(v_dirs[i], la_times_gnorm);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Muon — Fused Newton-Schulz Step (all 2D params in C++ loop)
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

        // Momentum accumulation: buf = momentum * buf + grad
        buf.mul_(momentum).add_(g);

        // Normalize: X = buf / ||buf||
        float buf_norm = buf.norm().item<float>() + 1e-7f;
        float inv_norm = 1.0f / buf_norm;

        // Newton-Schulz iteration (using ATen matmul for the matrix ops)
        auto X = buf * inv_norm;

        bool use_cuda = false;
#ifdef WITH_CUDA
        use_cuda = p.is_cuda();
#endif

        for (int step = 0; step < ns_steps; step++) {
            // A = X @ X^T  (always use ATen matmul — cuBLAS under the hood)
            auto A = torch::mm(X, X.t());
            // AX = A @ X
            auto AX = torch::mm(A, X);
            // AAX = A @ AX
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

        // Parameter update + weight decay
        int64_t rows = p.size(0);
        int64_t cols = p.size(1);
        float max_dim = static_cast<float>(std::max(rows, cols));
        float scale_factor = 0.2f * std::sqrt(max_dim);
        float neg_lr_scale = -lr * scale_factor / std::sqrt(max_dim);
        float decay_factor = 1.0f - lr * wd;

        if (use_cuda) {
#ifdef WITH_CUDA
            launch_muon_update(p, X, neg_lr_scale, decay_factor);
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
    m.doc() = "Grokking Race — C++/CUDA fused operations for all optimizers";

    // ── SuperGrok v1.5 ───────────────────────────────────────────────
    m.def("fused_step", &supergrok15_fused_step,
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

    m.def("sam_perturb_all", &supergrok15_sam_perturb_all,
          "SuperGrok1.5: SAM perturbation, return backups",
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("sharpness_restore_all", &supergrok15_sharpness_restore_all,
          "SuperGrok1.5: compute sharpness and restore from backups",
          py::arg("params"), py::arg("sharpness_cache"),
          py::arg("backups"), py::arg("sam_grads"), py::arg("normal_grads"));

    // ── Grokfast ─────────────────────────────────────────────────────
    m.def("grokfast_fused_step", &grokfast_fused_step,
          "Grokfast: fused EMA update + gradient amplification",
          py::arg("grads"), py::arg("ema_bufs"),
          py::arg("alpha"), py::arg("lamb"));

    // ── Lion ─────────────────────────────────────────────────────────
    m.def("lion_fused_step", &lion_fused_step,
          "Lion: fused momentum interp + sign + update + decay",
          py::arg("params"), py::arg("grads"), py::arg("exp_avgs"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("wd"));

    // ── LookSAM ─────────────────────────────────────────────────────
    m.def("looksam_perturb_all", &looksam_perturb_all,
          "LookSAM: perturb all params, return backups",
          py::arg("params"), py::arg("grads"), py::arg("rho"));

    m.def("looksam_restore_all", &looksam_restore_all,
          "LookSAM: restore all params from backups",
          py::arg("params"), py::arg("backups"));

    m.def("looksam_compute_directions", &looksam_compute_directions,
          "LookSAM: compute normalized sharpness direction vectors",
          py::arg("v_dirs"), py::arg("sam_grads"), py::arg("normal_grads"));

    m.def("looksam_adjust_grads", &looksam_adjust_grads,
          "LookSAM: adjust gradients using cached direction",
          py::arg("grads"), py::arg("v_dirs"), py::arg("la"));

    // ── Muon ─────────────────────────────────────────────────────────
    m.def("muon_fused_step", &muon_fused_step,
          "Muon: momentum + Newton-Schulz ortho + update + WD",
          py::arg("params"), py::arg("grads"), py::arg("bufs"),
          py::arg("momentum"), py::arg("lr"), py::arg("wd"),
          py::arg("ns_steps"));
}
