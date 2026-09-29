"""AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019).

The grokking baseline: with lr=1e-3, weight_decay=1.0 and betas=(0.9, 0.98) this
is the recipe of Power et al. (2022). Identical to ``torch.optim.AdamW`` (it is
checked bit for bit in the tests); it exists here so every optimizer in the race
shares one Adam tail (:func:`grokking_optimizers._core.adamw_update_`).
"""

from __future__ import annotations

import torch

from ._core import adam_state, adamw_update_, check_common


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        check_common(lr, betas, eps, weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

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
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                exp_avg, exp_avg_sq = adam_state(state, p)
                state["step"] += 1
                adamw_update_(
                    p,
                    p.grad,
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
