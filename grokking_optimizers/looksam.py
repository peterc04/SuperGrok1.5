"""LookSAM (Liu et al., "Towards Efficient and Scalable Sharpness-Aware Minimization", CVPR 2022, Alg. 1)
on an AdamW base.

SAM (Foret et al., 2021) steps with the gradient ``g_s`` taken at the nearby
worst-case point ``w + rho * g / ||g||``, which costs a second forward and
backward pass every step. LookSAM pays that only every ``k``-th step:

* SAM step (every ``k``-th, starting with the first): compute ``g_s`` through the
  closure, use it for the update, and cache its component orthogonal to ``g``,
  ``g_v = g_s - (<g_s, g> / ||g||^2) g``.
* the ``k - 1`` steps in between: use ``g + alpha * (||g|| / ||g_v||) * g_v``, so the
  flatness push always has norm ``alpha * ||g||``.

All norms and inner products are over the whole parameter vector (the paper's
global form), accumulated in float64. The modified gradient then goes through
the same AdamW update as the baseline (:func:`_core.adamw_update_`), so
``rho = 0`` is exactly AdamW and ``k = 1`` is exactly SAM + AdamW (tests).

Training-loop contract: ``needs_closure``. The closure must recompute the loss
on the same batch at the current (perturbed) weights, backpropagate and return
the loss, without side effects on training statistics (the race freezes the
MoE router's load counters inside it).
"""

from __future__ import annotations

import torch

from ._core import adam_state, adamw_update_, check_common


def _dot(a_list, b_list) -> torch.Tensor:
    if not a_list:
        return torch.zeros((), dtype=torch.float64)
    return torch.stack([(a.double() * b.double()).sum() for a, b in zip(a_list, b_list)]).sum()


class LookSAM(torch.optim.Optimizer):
    needs_closure = True
    uses_step_loss = True  # the loop passes the step's loss, for the sharpness diagnostic

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        rho=0.05,
        k=5,
        alpha=0.7,
        norm_eps=1e-12,
    ):
        check_common(lr, betas, eps, weight_decay)
        if rho < 0 or alpha < 0 or int(k) != k or k < 1:
            raise ValueError("need rho >= 0, alpha >= 0 and an integer k >= 1")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.rho, self.k, self.alpha, self.norm_eps = rho, int(k), alpha, norm_eps
        self.t = 0  # completed steps; step t is a SAM step iff t % k == 0
        self.gv_norm: torch.Tensor | None = None
        self.last_sharpness: torch.Tensor | None = None  # L(w + eps) - L(w) at the last SAM step

    def upcoming_step_kind(self) -> str:
        return "sam" if self.t % self.k == 0 else "reuse"

    def diagnostics(self) -> dict:
        return {
            "gv_norm": None if self.gv_norm is None else float(self.gv_norm),
            "sharpness": None if self.last_sharpness is None else float(self.last_sharpness),
        }

    def _with_grad(self):
        return [p for group in self.param_groups for p in group["params"] if p.grad is not None]

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        """``p.grad`` must hold ``g`` at the current weights; ``closure`` is used on SAM steps."""
        ps = self._with_grad()
        if self.t % self.k == 0:
            if closure is None:
                raise RuntimeError("LookSAM: SAM steps need a closure that re-runs forward and backward")
            self._sam(ps, closure, loss)
        elif self.gv_norm is not None:
            g_norm = _dot([p.grad for p in ps], [p.grad for p in ps]).sqrt()
            coef = self.alpha * g_norm / (self.gv_norm + self.norm_eps)
            for p in ps:
                g_v = self.state[p].get("g_v")
                if g_v is not None:
                    p.grad.add_(g_v * coef.to(g_v.dtype))
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
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
        self.t += 1

    def _sam(self, ps, closure, loss):
        g = [p.grad.detach().clone() for p in ps]
        gg = _dot(g, g)
        scale = self.rho / (gg.sqrt() + self.norm_eps)
        backup = [p.detach().clone() for p in ps]
        for p, gi in zip(ps, g):
            p.add_((gi.double() * scale).to(p.dtype))
        with torch.enable_grad():
            perturbed_loss = closure()
        g_s = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ps]
        for p, b in zip(ps, backup):  # exact restore
            p.copy_(b)
        # no eps here: a residual (eps / ||g||^2) g would be amplified to alpha * ||g|| on reuse steps
        c = torch.where(gg > 0, _dot(g_s, g) / gg.clamp_min(1e-300), torch.zeros_like(gg))
        g_v = [gs - c.to(gs.dtype) * gi for gs, gi in zip(g_s, g)]
        self.gv_norm = _dot(g_v, g_v).sqrt()
        for p, gv, gs in zip(ps, g_v, g_s):
            self.state[p]["g_v"] = gv
            p.grad = gs  # the SAM step itself uses g_s
        with_grad = set(ps)
        for group in self.param_groups:  # no gradient at w (an idle expert): none after the probe either
            for p in group["params"]:
                if p not in with_grad:
                    p.grad = None
        if loss is not None and perturbed_loss is not None:
            self.last_sharpness = (perturbed_loss.detach() - loss.detach()).double()

    def state_dict(self):
        sd = super().state_dict()
        sd["looksam"] = {"t": self.t, "gv_norm": self.gv_norm}
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)  # do not consume the caller's dict
        extra = state_dict.pop("looksam", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self.t, self.gv_norm = extra["t"], extra["gv_norm"]
