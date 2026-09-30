"""GrokAdamW (E. Hartford / cognitivecomputations, pip ``grokadamw`` 0.1.2; no paper).

AdamW on a Grokfast-filtered gradient plus three heuristics, exactly as the
published code does them:

1. each tensor's gradient is clipped to norm ``gradient_clipping`` on its own;
2. "layer-wise" momentum ``beta1_i = beta1 * (1 - gamma) ** i``, where ``i``
   counts the tensors *with a gradient* in the parameter group, in order (it is a
   tensor index, not a transformer-layer index; when an MoE expert gets no tokens
   on a step, every later tensor's ``beta1`` shifts for that step, exactly as in
   the published code. At the race's full batch every expert gets tokens);
3. an adaptive EMA decay ``alpha = alpha_init * exp(-kappa * signal)`` driven by
   a train/held-out loss gap (``set_losses``; with no losses set, alpha stays at
   ``alpha_init``).

Then ``ema = alpha * ema + (1 - alpha) * g`` (EMA starts at zero),
``g_hat = g + lamb * ema`` and an Adam step on ``g_hat`` with ``eps`` added to
``sqrt(v)`` and decoupled weight decay.

The published code bias-corrects the first moment with the *global* ``beta1``
while ``m`` uses ``beta1_i``, which inflates the first steps of deep tensors by
up to ``1 / (1 - beta1)``. That is reproduced by default;
``bias_correction1="layer"`` corrects it. With ``gamma=0`` the method is
Grokfast (zero-initialised EMA) + a per-tensor clip. Deliberate differences from
the published file that do not change the numbers: state lives in
``self.state`` on the parameter's device (not CPU-side ``group['state']``), so
checkpoints round-trip.
"""

from __future__ import annotations

import math

import torch

from ._core import check_common


def grokking_signal(train_loss: float | None, heldout_loss: float | None) -> float:
    """The published default signal: the non-negative loss gap over the larger loss, in [0, 1]."""
    if train_loss is None or heldout_loss is None:
        return 0.0
    top = max(train_loss, heldout_loss)
    return max(0.0, heldout_loss - train_loss) / top if top > 0 else 0.0


class GrokAdamW(torch.optim.Optimizer):
    wants_losses = True

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
        *,
        bias_correction1="published",
    ):
        check_common(lr, betas, eps, weight_decay)
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"invalid alpha_init {alpha_init}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"invalid gamma {gamma}")
        if bias_correction1 not in ("published", "layer"):
            raise ValueError(f"bias_correction1 must be 'published' or 'layer', got {bias_correction1!r}")
        self.bias_correction1 = bias_correction1
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                alpha_init=alpha_init,
                lamb=lamb,
                gamma=gamma,
                grokking_signal_decay_rate=grokking_signal_decay_rate,
                gradient_clipping=gradient_clipping,
                train_loss=None,
                eval_loss=None,
            ),
        )

    def set_losses(self, train_loss: float | None, heldout_loss: float | None, train_acc: float | None = None) -> None:
        """Feed the grokking signal (the published code reads these from each group); ``train_acc`` is unused."""
        for group in self.param_groups:
            group["train_loss"], group["eval_loss"] = train_loss, heldout_loss

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd, eps, lamb = group["lr"], group["weight_decay"], group["eps"], group["lamb"]
            signal = grokking_signal(group["train_loss"], group["eval_loss"])
            alpha = group["alpha_init"] * math.exp(-group["grokking_signal_decay_rate"] * signal)
            with_grad = [p for p in group["params"] if p.grad is not None]
            for i, p in enumerate(with_grad):
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("GrokAdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["grok_ema"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                if group["gradient_clipping"] > 0:
                    torch.nn.utils.clip_grad_norm_(p, group["gradient_clipping"])  # in place, per tensor
                beta1_i = beta1 * (1 - group["gamma"]) ** i
                ema, m, v = state["grok_ema"], state["exp_avg"], state["exp_avg_sq"]
                ema.mul_(alpha).add_(g, alpha=1 - alpha)
                g_hat = g + lamb * ema
                m.mul_(beta1_i).add_(g_hat, alpha=1 - beta1_i)
                v.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)
                bc1 = 1 - (beta1_i if self.bias_correction1 == "layer" else beta1) ** t
                step_size = lr * math.sqrt(1 - beta2**t) / bc1
                p.mul_(1 - lr * wd)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
        return loss
