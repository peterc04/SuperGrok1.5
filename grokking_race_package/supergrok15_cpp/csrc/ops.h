/*
 * Grokking Race — C++ Operations Header
 *
 * Declares kernel launchers and high-level optimizer step functions for:
 *   - SuperGrok v1.5 (fused mu + meta-net + gating + adam + wd)
 *   - Grokfast (fused EMA + gradient amplification)
 *   - Lion (fused momentum interpolation + sign + update + decay)
 *   - LookSAM (SAM perturbation/restore + direction + adjust)
 *   - Muon (momentum + Newton-Schulz + update)
 */
#pragma once
#include <torch/extension.h>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════
//  CUDA Kernel Launchers (defined in respective .cu files)
// ═══════════════════════════════════════════════════════════════════════

#ifdef WITH_CUDA

// ── SuperGrok v1.5 (kernels.cu) ──────────────────────────────────────
void launch_fused_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor smart_grad, float alpha,
    torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2,
    float rescale, int hidden_dim);

void launch_fused_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor smart_grad, torch::Tensor mu,
    float lamb_eff, float beta1, float beta2, float lr, float wd_eff,
    float eps, float bc1, float bc2);

void launch_sam_perturb(torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup,
    torch::Tensor sam_grad, torch::Tensor normal_grad);

// ── Grokfast (grokfast_kernels.cu) ──────────────────────────────────
void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema,
    float alpha, float lamb);

// ── Lion (lion_kernels.cu) ──────────────────────────────────────────
void launch_fused_lion_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor grad,
    float lr, float beta1, float beta2, float wd);

// ── LookSAM (looksam_kernels.cu) ───────────────────────────────────
void launch_looksam_direction(
    torch::Tensor v_dir, torch::Tensor sam_grad, torch::Tensor normal_grad,
    float inv_norm);

void launch_looksam_adjust(
    torch::Tensor grad, torch::Tensor v_dir, float la_times_gnorm);

void launch_looksam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm);

void launch_looksam_restore(
    torch::Tensor param, torch::Tensor backup);

// ── Muon (muon_kernels.cu) ─────────────────────────────────────────
void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad,
    float momentum, float inv_norm);

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX,
    float a, float b, float c);

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth,
    float neg_lr_scale, float decay_factor);

#endif  // WITH_CUDA


// ═══════════════════════════════════════════════════════════════════════
//  High-Level C++ Operations (called from Python via pybind11)
// ═══════════════════════════════════════════════════════════════════════

// ── SuperGrok v1.5 ───────────────────────────────────────────────────

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
    float grad_clip_norm);

std::vector<torch::Tensor> supergrok15_sam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

void supergrok15_sharpness_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& sharpness_cache,
    std::vector<torch::Tensor>& backups,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);

// ── Grokfast ─────────────────────────────────────────────────────────

// Process all parameters: EMA update + gradient amplification in one C++ loop
void grokfast_fused_step(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& ema_bufs,
    float alpha,
    float lamb);

// ── Lion ─────────────────────────────────────────────────────────────

// Process all parameters: interpolation + sign + update + decay
void lion_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& exp_avgs,
    float lr,
    float beta1,
    float beta2,
    float wd);

// ── LookSAM ─────────────────────────────────────────────────────────

// SAM closure: perturb all params, return backups
std::vector<torch::Tensor> looksam_perturb_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    float rho);

// Restore all params from backups
void looksam_restore_all(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& backups);

// Compute normalized direction vectors: v_dir = (sam_grad - normal_grad) / ||...||
void looksam_compute_directions(
    std::vector<torch::Tensor>& v_dirs,
    std::vector<torch::Tensor>& sam_grads,
    std::vector<torch::Tensor>& normal_grads);

// Adjust gradients using cached direction: grad += lambda * ||grad|| * v_dir
void looksam_adjust_grads(
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& v_dirs,
    float la);

// ── Muon ─────────────────────────────────────────────────────────────

// Full Muon step for one 2D parameter:
// momentum accumulation + Newton-Schulz orthogonalization + param update + WD
void muon_fused_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& bufs,
    float momentum,
    float lr,
    float wd,
    int ns_steps);
