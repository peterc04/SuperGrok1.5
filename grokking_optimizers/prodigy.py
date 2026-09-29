"""Prodigy (Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", 2023).

Adam driven by one global, learned step size ``d`` (Algorithm 3, Adam version):

    m = beta1 m + (1 - beta1) d g            v = beta2 v + (1 - beta2) d^2 g^2
    r = beta3 r + (1 - beta3) d^2 <g, x0 - x>   (scalar)
    s = beta3 s + (1 - beta3) d^2 g             (vector, same shape as x)
    d = max(d, r / ||s||_1)                     (d only grows)
    x = x - lr d m / (sqrt(v) + d eps)

with ``beta3 = sqrt(beta2)`` by default. Leave ``lr`` at 1.0 (it multiplies the
estimate). Step semantics match the authors' ``prodigyopt`` 1.1.2 exactly
(checked in the tests), with two implementation differences that do not change
the numbers: one host sync per step instead of two per tensor, and ``p.grad`` is
never modified in place. Hyperparameters that ``prodigyopt`` silently reads from
the first parameter group only must be identical across groups here.

Memory: ``exp_avg``, ``exp_avg_sq``, ``s`` and the initial point ``p0``: 16 B per
parameter in fp32, twice AdamW. ``slice_p=k`` keeps ``s`` and ``p0`` for every
k-th coordinate only (the authors suggest 11 for large models).

Caveat for grokking: with decoupled weight decay applied as ``wd * d * lr`` and
a constant ``lr``, a large ``weight_decay`` (the race's 1.0) can make ``d``
ratchet upward; the authors recommend ``weight_decay <= 0.1`` and a cosine schedule.
"""

from __future__ import annotations

import math

import torch

_SHARED = ("betas", "beta3", "d0", "d_coef", "growth_rate", "use_bias_correction", "decouple", "safeguard_warmup")


class Prodigy(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        beta3=None,
        eps=1e-8,
        weight_decay=0.0,
        decouple=True,
        use_bias_correction=False,
        safeguard_warmup=False,
        d0=1e-6,
        d_coef=1.0,
        growth_rate=float("inf"),
        slice_p=1,
    ):
        if not d0 > 0.0:
            raise ValueError(f"invalid d0 {d0}")
        if not lr > 0.0:
            raise ValueError(f"invalid lr {lr}")
        if not eps > 0.0:
            raise ValueError(f"invalid eps {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError(f"invalid beta3 {beta3}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay {weight_decay}")
        if not growth_rate >= 1.0:
            raise ValueError(f"invalid growth_rate {growth_rate}")
        if int(slice_p) != slice_p or slice_p < 1:
            raise ValueError(f"invalid slice_p {slice_p}")
        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            use_bias_correction=use_bias_correction,
            safeguard_warmup=safeguard_warmup,
            d0=d0,
            d_coef=d_coef,
            growth_rate=growth_rate,
            slice_p=int(slice_p),
            # the global estimator state lives in every group so it survives state_dict()
            d=d0,
            d_max=d0,
            d_numerator=0.0,
            d_denom=0.0,
            d_hat=d0,
            k=0,
        )
        super().__init__(params, defaults)
        self._check_groups()

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        if len(self.param_groups) > 1:
            self._check_groups()

    def _check_groups(self):
        g0 = self.param_groups[0]
        for g in self.param_groups[1:]:
            for key in _SHARED:
                if g[key] != g0[key]:
                    raise ValueError(f"Prodigy: {key!r} must be the same in every parameter group")
        lr = max(g["lr"] for g in self.param_groups)
        if any(g["lr"] not in (lr, 0.0) for g in self.param_groups):
            raise ValueError("Prodigy: parameter groups may only use the common lr or 0")

    @property
    def d(self) -> float:
        return self.param_groups[0]["d"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g0 = self.param_groups[0]
        beta1, beta2 = g0["betas"]
        beta3 = g0["beta3"] if g0["beta3"] is not None else math.sqrt(beta2)
        k, d, d_max, d0 = g0["k"], g0["d"], g0["d_max"], g0["d0"]
        lr = max(g["lr"] for g in self.param_groups)
        bias_correction = (1 - beta2 ** (k + 1)) ** 0.5 / (1 - beta1 ** (k + 1)) if g0["use_bias_correction"] else 1
        dlr = d * lr * bias_correction
        d_numerator = g0["d_numerator"] * beta3

        # pass 1: moments, s and the per-tensor partial sums of <g, x0 - x> and |s|_1
        work, dots, l1s = [], [], []
        for group in self.param_groups:
            decay, slice_p = group["weight_decay"], group["slice_p"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if decay != 0 and not g0["decouple"]:
                    grad = grad.add(p, alpha=decay)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.flatten()[::slice_p])
                    state["p0"] = (
                        p.flatten()[::slice_p].clone() if p.any() else torch.tensor(0, device=p.device, dtype=p.dtype)
                    )
                    if beta1 > 0:
                        state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                work.append((p, state, grad, group))
                if group["lr"] > 0.0:
                    sliced = grad.flatten()[::slice_p]
                    dots.append(torch.dot(sliced, state["p0"] - p.flatten()[::slice_p]))
                    if beta1 > 0:
                        state["exp_avg"].mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))
                    s_alpha = (d / d0) * d if g0["safeguard_warmup"] else (d / d0) * dlr
                    state["s"].mul_(beta3).add_(sliced, alpha=s_alpha)
                    l1s.append(state["s"].abs().sum())

        vals = torch.stack(dots + l1s).tolist() if dots else []  # one host sync
        delta_numerator = 0.0
        for v in vals[: len(dots)]:
            delta_numerator += (d / d0) * dlr * v
        d_denom = 0.0
        for v in vals[len(dots) :]:
            d_denom += v
        if d_denom == 0:
            return loss

        d_hat = d
        if lr > 0.0:
            d_numerator += delta_numerator
            d_hat = g0["d_coef"] * d_numerator / d_denom
            if d == d0:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * g0["growth_rate"])
        for group in self.param_groups:
            group.update(d_numerator=d_numerator, d_denom=d_denom, d=d, d_max=d_max, d_hat=d_hat, k=k + 1)

        # pass 2: apply
        for p, state, grad, group in work:
            state["step"] += 1
            denom = state["exp_avg_sq"].sqrt().add_(d * group["eps"])
            if group["weight_decay"] != 0 and g0["decouple"]:
                p.add_(p, alpha=-group["weight_decay"] * dlr)
            if beta1 > 0:
                p.addcdiv_(state["exp_avg"], denom, value=-dlr)
            else:
                p.addcdiv_(grad, denom, value=-dlr * d)
        return loss
