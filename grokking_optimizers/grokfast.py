"""Grokfast-EMA (Lee et al., "Grokfast: Accelerated Grokking by Amplifying Slow Gradients", 2024).

A gradient filter in front of AdamW: keep an EMA of each gradient and add
``lamb`` times it back, ``g_hat = g + lamb * ema`` with
``ema = alpha * ema + (1 - alpha) * g``. Slowly varying gradient directions are
amplified by up to ``1 + lamb``; fast-alternating ones pass almost unchanged.
Matches the authors' ``gradfilter_ema`` (ironjr/grokfast) followed by
``torch.optim.AdamW``: the EMA is seeded with the first gradient, so the first
step already sees ``(1 + lamb) * g``.
"""

from __future__ import annotations

import torch

from ._core import adam_state, adamw_update_, check_common


def grokfast_filter_(g: torch.Tensor, ema: torch.Tensor | None, alpha: float, lamb: float):
    """Return ``(g_hat, ema)``; pass ``ema=None`` on the first step to seed it with ``g``."""
    if ema is None:
        ema = g.detach().clone()
    ema.mul_(alpha).add_(g * (1 - alpha))  # same op order as gradfilter_ema, bit for bit
    return g + ema * lamb, ema


class Grokfast(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, alpha=0.98, lamb=2.0):
        check_common(lr, betas, eps, weight_decay)
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"invalid alpha {alpha}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, lamb=lamb))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = adam_state(state, p)
                g_hat, state["grokfast_ema"] = grokfast_filter_(
                    p.grad, state.get("grokfast_ema"), group["alpha"], group["lamb"]
                )
                state["step"] += 1
                adamw_update_(
                    p,
                    g_hat,
                    exp_avg,
                    exp_avg_sq,
                    int(state["step"].item()),
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                )
        return loss
