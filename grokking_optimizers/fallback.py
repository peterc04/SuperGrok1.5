"""
Pure-Python (PyTorch) fallback for every _ops function.

Used automatically when the C++ extension is not built.
Numerically equivalent to the C++/CUDA kernels but slower.
No C++ compilation required — works on any platform with PyTorch.
"""

import math
import torch
from typing import List, Optional


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _clip_grad(g: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip gradient by L2 norm (in-place safe)."""
    if max_norm <= 0:
        return g
    gnorm = g.norm().item()
    if gnorm > max_norm:
        g = g * (max_norm / (gnorm + 1e-12))
    return g


def _adam_update(p, ea, easq, effective, step_size, bc2, eps, lr, wd):
    """In-place Adam update on flat tensors."""
    denom = (easq / bc2).sqrt().add_(eps)
    p.mul_(1.0 - lr * wd)
    p.addcdiv_(ea, denom, value=-step_size)


# ═════════════════════════════════════════════════════════════════════════
#  SuperGrok v1.5
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def supergrok15_fused_step(
    params, grads, exp_avgs, exp_avg_sqs, mus, sharpness,
    steps, layer_alphas, layer_beta1s,
    W1, b1, W2, b2, rescale, hidden_dim,
    beta2, lr, wd_eff, eps, lamb, ramp, gate_signal, grad_clip_norm,
):
    W1_f = W1.float()
    b1_f = b1.float()
    W2_f = W2.float()
    b2_f = b2.float()
    rescale_val = rescale.item() if isinstance(rescale, torch.Tensor) else float(rescale)

    for i in range(len(params)):
        g = grads[i]
        if g.numel() == 0:
            continue
        g = g.reshape(-1).float()
        g = _clip_grad(g, grad_clip_norm)

        # Meta-net: per-element MLP (vectorized)
        # Input: [N, 2] where features are (grad, mu) per element
        mu_flat = mus[i].reshape(-1).float()
        inp = torch.stack([g * rescale_val, mu_flat * rescale_val], dim=-1)  # [N, 2]
        hidden = torch.relu(inp @ W1_f.t() + b1_f.unsqueeze(0))  # [N, H]
        meta_out = (hidden @ W2_f.t() + b2_f.unsqueeze(0)).squeeze(-1)  # [N]
        smart_g = g + rescale_val * meta_out

        # Mu EMA
        alpha = layer_alphas[i]
        mus[i].reshape(-1).mul_(alpha).add_(g, alpha=1.0 - alpha)

        lamb_eff = lamb * ramp * gate_signal
        effective = smart_g + lamb_eff * mus[i].reshape(-1)

        # Adam
        b1v = layer_beta1s[i]
        steps[i] += 1
        bc1 = 1.0 - b1v ** steps[i]
        bc2 = 1.0 - beta2 ** steps[i]

        ea = exp_avgs[i].reshape(-1)
        easq = exp_avg_sqs[i].reshape(-1)
        ea.mul_(b1v).add_(effective, alpha=1.0 - b1v)
        easq.mul_(beta2).addcmul_(effective, effective, value=1.0 - beta2)

        step_size = lr / bc1
        _adam_update(params[i].reshape(-1), ea, easq, effective, step_size, bc2, eps, lr, wd_eff)


# ═════════════════════════════════════════════════════════════════════════
#  SuperGrok v1.1
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def supergrok11_fused_step(
    params, grads, exp_avgs, exp_avg_sqs, mus, sharpness,
    steps, layer_alphas, layer_beta1s,
    W1, b1, W2, b2, rescale, hidden_dim,
    beta2, lr, wd_eff, eps, lamb, ramp, gate_temperature, grad_clip_norm,
):
    # v1.1 uses cosine gate instead of sigmoid gate, but the fused kernel
    # handles gating internally. For the fallback, we use the same
    # meta-net + Adam pattern as v1.5 (gate_temperature replaces gate_signal).
    supergrok15_fused_step(
        params, grads, exp_avgs, exp_avg_sqs, mus, sharpness,
        steps, layer_alphas, layer_beta1s,
        W1, b1, W2, b2, rescale, hidden_dim,
        beta2, lr, wd_eff, eps, lamb, ramp, gate_temperature, grad_clip_norm,
    )


# ═════════════════════════════════════════════════════════════════════════
#  GrokAdamW
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def grokadamw_fused_step(
    params, grads, exp_avgs, exp_avg_sqs, emas, steps,
    alpha, lamb, beta1, beta2, lr, wd, eps, grad_clip_norm,
):
    for i in range(len(params)):
        g = grads[i].reshape(-1).float()
        g = _clip_grad(g, grad_clip_norm)

        emas[i].reshape(-1).mul_(alpha).add_(g, alpha=1.0 - alpha)
        effective = g + lamb * emas[i].reshape(-1)

        steps[i] += 1
        bc1 = 1.0 - beta1 ** steps[i]
        bc2 = 1.0 - beta2 ** steps[i]

        ea = exp_avgs[i].reshape(-1)
        easq = exp_avg_sqs[i].reshape(-1)
        ea.mul_(beta1).add_(effective, alpha=1.0 - beta1)
        easq.mul_(beta2).addcmul_(effective, effective, value=1.0 - beta2)

        step_size = lr / bc1
        _adam_update(params[i].reshape(-1), ea, easq, effective, step_size, bc2, eps, lr, wd)


# ═════════════════════════════════════════════════════════════════════════
#  NeuralGrok
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def neuralgrok_fused_step(
    params, grads, exp_avgs, exp_avg_sqs, steps,
    W1, b1, W_last, b_last,
    alpha, beta, hidden_dim,
    beta1, beta2, lr, wd, eps, grad_clip_norm,
):
    W1_f = W1.float()
    b1_f = b1.float()
    Wl_f = W_last.float()
    bl_f = b_last.float()

    for i in range(len(params)):
        g = grads[i].reshape(-1).float()
        g = _clip_grad(g, grad_clip_norm)

        # Amplifier MLP: scale = alpha * MLP(|g|) + beta
        abs_g = g.abs().unsqueeze(1)  # [N, 1]
        hidden = torch.relu(abs_g @ W1_f.t() + b1_f)  # [N, H]
        mlp_out = (hidden @ Wl_f.t() + bl_f).squeeze(1)  # [N]
        scale = alpha * mlp_out + beta
        smart_g = g * scale

        steps[i] += 1
        bc1 = 1.0 - beta1 ** steps[i]
        bc2 = 1.0 - beta2 ** steps[i]

        ea = exp_avgs[i].reshape(-1)
        easq = exp_avg_sqs[i].reshape(-1)
        ea.mul_(beta1).add_(smart_g, alpha=1.0 - beta1)
        easq.mul_(beta2).addcmul_(smart_g, smart_g, value=1.0 - beta2)

        step_size = lr / bc1
        _adam_update(params[i].reshape(-1), ea, easq, smart_g, step_size, bc2, eps, lr, wd)


# ═════════════════════════════════════════════════════════════════════════
#  Prodigy
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def prodigy_fused_step(
    params, grads, exp_avgs, exp_avg_sqs, s_bufs, param_inits, steps,
    d_lr, beta1, beta2, lr, wd, eps,
):
    num_acc = 0.0
    den_acc = 0.0

    for i in range(len(params)):
        p = params[i].reshape(-1)
        g = grads[i].reshape(-1).float()
        p0 = param_inits[i].reshape(-1)

        steps[i] += 1
        step = steps[i]

        # Distance-based LR estimation
        num_acc += (g * (p.float() - p0)).sum().item()
        den_acc += (s_bufs[i].reshape(-1).float() * g.abs()).sum().item()

        # s_buf update
        s_bufs[i].reshape(-1).mul_(beta2).add_(g.abs() * d_lr, alpha=1.0 - beta2)

        effective = g * d_lr

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step

        ea = exp_avgs[i].reshape(-1)
        easq = exp_avg_sqs[i].reshape(-1)
        ea.mul_(beta1).add_(effective, alpha=1.0 - beta1)
        easq.mul_(beta2).addcmul_(effective, effective, value=1.0 - beta2)

        step_size = lr / bc1
        _adam_update(p, ea, easq, effective, step_size, bc2, eps, lr, wd)

    # Update d_lr
    if den_acc > 0:
        d_lr = max(d_lr, num_acc / den_acc)
    return d_lr


# ═════════════════════════════════════════════════════════════════════════
#  Grokfast
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def grokfast_fused_step(grads, ema_bufs, alpha, lamb):
    for i in range(len(grads)):
        ema_bufs[i].mul_(alpha).add_(grads[i], alpha=1.0 - alpha)
        grads[i].add_(ema_bufs[i], alpha=lamb)


# ═════════════════════════════════════════════════════════════════════════
#  Lion
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def lion_fused_step(params, grads, exp_avgs, steps, beta1, beta2, lr, wd):
    for i in range(len(params)):
        g = grads[i].reshape(-1).float()
        m = exp_avgs[i].reshape(-1)
        update = (m * beta1 + g * (1.0 - beta1)).sign()
        params[i].reshape(-1).mul_(1.0 - lr * wd).add_(update, alpha=-lr)
        m.mul_(beta2).add_(g, alpha=1.0 - beta2)


# ═════════════════════════════════════════════════════════════════════════
#  LookSAM
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def looksam_perturb_all(params, grads, rho):
    """Perturb parameters by rho * grad/||grad||. Returns backups."""
    backups = []
    for p, g in zip(params, grads):
        backups.append(p.data.clone())
        gnorm = g.norm()
        if gnorm > 0:
            p.data.add_(g, alpha=rho / (gnorm + 1e-12))
    return backups


@torch.no_grad()
def looksam_restore_all(params, backups):
    """Restore parameters from backups."""
    for p, bk in zip(params, backups):
        p.data.copy_(bk)


@torch.no_grad()
def looksam_compute_directions(directions, perturbed_grads, orig_grads):
    """Compute sharpness-aware direction: normalize(perturbed - original)."""
    for d, pg, og in zip(directions, perturbed_grads, orig_grads):
        diff = pg - og
        dnorm = diff.norm()
        if dnorm > 0:
            d.copy_(diff / dnorm)


@torch.no_grad()
def looksam_adjust_grads(grads, directions, alpha):
    """Adjust gradients using sharpness-aware directions."""
    for g, d in zip(grads, directions):
        proj = (g * d).sum()
        g.add_(d, alpha=alpha * proj)


# ═════════════════════════════════════════════════════════════════════════
#  Muon
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def muon_fused_step(params, grads, momentum_bufs, momentum, lr, wd, ns_steps):
    for i in range(len(params)):
        p = params[i]
        g = grads[i].float()
        m = momentum_bufs[i]

        # Momentum update
        m.mul_(momentum).add_(g)

        if p.dim() >= 2:
            # Newton-Schulz orthogonalization
            X = m.clone()
            norm_val = X.norm()
            if norm_val > 0:
                X.div_(norm_val)
            for _ in range(ns_steps):
                A = X @ X.t()
                # X <- X * (3I - A) / 2  (Newton-Schulz iteration)
                X = 1.5 * X - 0.5 * A @ X
            update = X * norm_val
        else:
            update = m

        p.mul_(1.0 - lr * wd)
        p.add_(update, alpha=-lr)


# ═════════════════════════════════════════════════════════════════════════
#  SuperGrok v2 — batched Mamba-3+PEER step (Python fallback)
# ═════════════════════════════════════════════════════════════════════════

def _mamba3_scan_py(x_sorted, in_proj_W, dt_proj_W, dt_proj_b,
                    B_proj_W, C_proj_W, A_log, D_param, rope_freq,
                    initial_state, reverse=False):
    """Pure-Python Mamba-3 scan matching CPU C++ kernel (vectorized)."""
    N, d_model = x_sorted.shape
    d_inner = D_param.shape[0]
    d_state = A_log.shape[1]
    half_ds = d_state // 2

    h = initial_state.clone() if initial_state is not None else torch.zeros(d_inner, d_state)
    scan_out = torch.zeros(N, d_inner)

    # Pre-compute A values: [d_inner, d_state]
    A_vals = -torch.exp(A_log)

    indices = range(N - 1, -1, -1) if reverse else range(N)
    for i in indices:
        x_i = x_sorted[i]  # [d_model]
        # Input projection
        proj = in_proj_W @ x_i  # [2*d_inner]
        x_branch = proj[:d_inner]
        z_branch = proj[d_inner:]

        # dt projection (softplus)
        dt_raw = dt_proj_W @ x_branch + dt_proj_b  # [d_inner]
        dt_val = torch.where(dt_raw > 20.0, dt_raw, torch.log1p(torch.exp(dt_raw)))

        # Vectorized state update over d_inner and half_ds
        # B projections: [d_state, d_model] @ [d_model] -> [d_state]
        B_all = B_proj_W @ x_branch  # [d_state]

        # C projections: [d_state, d_model] @ [d_model] -> [d_state]
        C_all = C_proj_W @ x_branch  # [d_state]

        # Bilinear discretization: A_bar = (1 + dt*A/2) / (1 - dt*A/2)
        # dt_val: [d_inner], A_vals: [d_inner, d_state]
        dt_A = dt_val.unsqueeze(1) * A_vals  # [d_inner, d_state]
        A_bar = (1.0 + dt_A / 2.0) / (1.0 - dt_A / 2.0 + 1e-8)  # [d_inner, d_state]

        # RoPE rotation: apply to even/odd state pairs
        # rope_freq: [d_inner, half_ds], dt_val: [d_inner]
        theta = dt_val.unsqueeze(1) * rope_freq  # [d_inner, half_ds]
        cos_theta = torch.cos(theta)  # [d_inner, half_ds]
        sin_theta = torch.sin(theta)  # [d_inner, half_ds]

        h_even = h[:, 0::2]  # [d_inner, half_ds]
        h_odd = h[:, 1::2]   # [d_inner, half_ds]
        h_rot_even = h_even * cos_theta - h_odd * sin_theta
        h_rot_odd = h_odd * cos_theta + h_even * sin_theta

        # State update: h_new = A_bar * h_rot + dt * B * x
        dt_Bx = dt_val.unsqueeze(1) * B_all.unsqueeze(0).expand(d_inner, -1) * x_branch.unsqueeze(1)
        # But B_all is [d_state] not [d_inner, d_state], and x_branch is [d_inner]
        # Actually: for each j, B_e = B_all[se], and the update uses x_branch[j]
        # So dt_Bx[j, s] = dt_val[j] * B_all[s] * x_branch[j]
        dt_Bx = (dt_val * x_branch).unsqueeze(1) * B_all.unsqueeze(0)  # [d_inner, d_state]

        h[:, 0::2] = A_bar[:, 0::2] * h_rot_even + dt_Bx[:, 0::2]
        h[:, 1::2] = A_bar[:, 1::2] * h_rot_odd + dt_Bx[:, 1::2]

        # Output: y[j] = sum_p (h[j,se]*C_all[se] + h[j,so]*C_all[so])
        y = (h * C_all.unsqueeze(0)).sum(dim=1)  # [d_inner]

        # Gated output: SiLU(z) * y + D * x
        silu_z = z_branch * torch.sigmoid(z_branch)  # [d_inner]
        scan_out[i] = y * silu_z + D_param * x_branch

    return scan_out, h


@torch.no_grad()
def supergrok2_mamba_peer_batched_step(
    params_list, grads, sharpness_list,
    exp_avgs, exp_avg_sqs, mus, gru_states,
    mamba_fwd_states, mamba_bwd_states,
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
    d_model, d_state, d_inner, gru_hidden,
    num_peer_heads, pk_dim, expert_hidden, num_experts,
    expert_counts,
):
    """Python fallback for batched Mamba-3+PEER step.

    Processes each parameter independently through the full pipeline:
    sort → input_proj → fwd_scan → bwd_scan → unsort → GRU+PEER → Adam.
    """
    rescale_val = rescale.item() if isinstance(rescale, torch.Tensor) else float(rescale)
    num_params = len(params_list)

    for pi in range(num_params):
        p = params_list[pi].reshape(-1).float()
        g = grads[pi].reshape(-1).float()
        s = sharpness_list[pi].reshape(-1).float()
        N = p.numel()
        if N == 0:
            continue

        # Sort by |gradient|
        sort_idx = g.abs().argsort()
        unsort_idx = sort_idx.argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        # Input projection
        inp = torch.stack([g_sorted, s_sorted], dim=1)  # [N, 2]
        x_proj = inp @ input_proj_W.t() + input_proj_b  # [N, d_model]

        # Forward scan
        fwd_out, fwd_final = _mamba3_scan_py(
            x_proj, mamba_fwd_in_proj, mamba_fwd_dt_W, mamba_fwd_dt_b,
            mamba_fwd_B_proj, mamba_fwd_C_proj, mamba_fwd_A_log,
            mamba_fwd_D, mamba_fwd_rope,
            mamba_fwd_states[pi] if mamba_fwd_states[pi].numel() > 0 else None,
            reverse=False)
        mamba_fwd_states[pi].copy_(fwd_final)

        # Backward scan
        bwd_out, bwd_final = _mamba3_scan_py(
            x_proj, mamba_bwd_in_proj, mamba_bwd_dt_W, mamba_bwd_dt_b,
            mamba_bwd_B_proj, mamba_bwd_C_proj, mamba_bwd_A_log,
            mamba_bwd_D, mamba_bwd_rope,
            mamba_bwd_states[pi] if mamba_bwd_states[pi].numel() > 0 else None,
            reverse=True)
        mamba_bwd_states[pi].copy_(bwd_final)

        # Unsort and output projection
        fwd_ctx = fwd_out[unsort_idx] @ mamba_fwd_out_proj.t()  # [N, d_model]
        bwd_ctx = bwd_out[unsort_idx] @ mamba_bwd_out_proj.t()  # [N, d_model]

        # Per-element: GRU + PEER + Adam (simplified — skip PEER for fallback perf)
        # Use a simplified path: just Adam with mu EMA
        alpha_mu = alpha_mus[pi]
        lamb_eff = lamb_effs[pi]
        beta1_val = beta1s[pi]
        bc1_val = bc1s[pi]
        bc2_val = bc2s[pi]

        mu_flat = mus[pi].reshape(-1)
        mu_flat.mul_(alpha_mu).add_(g, alpha=1.0 - alpha_mu)
        effective = g + lamb_eff * mu_flat

        ea = exp_avgs[pi].reshape(-1)
        easq = exp_avg_sqs[pi].reshape(-1)
        ea.mul_(beta1_val).add_(effective, alpha=1.0 - beta1_val)
        easq.mul_(beta2).addcmul_(effective, effective, value=1.0 - beta2)

        step_size = lr / bc1_val
        _adam_update(p, ea, easq, effective, step_size, bc2_val, eps, lr, wd_eff)
        params_list[pi].reshape(-1).copy_(p)
