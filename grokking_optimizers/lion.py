"""Lion (Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023, Algorithm 2).

One momentum buffer per weight and a sign update: every coordinate moves by
exactly ``lr``. Because of that it wants an lr 3-10x smaller and a weight decay
3-10x larger than AdamW. Same operation order as the authors' reference
(google/automl ``lion_pytorch.py``); the tests check it step for step.
"""

from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m = state["exp_avg"]
                p.mul_(1 - lr * wd)
                update = m * beta1 + g * (1 - beta1)
                p.add_(update.sign_(), alpha=-lr)
                m.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss
