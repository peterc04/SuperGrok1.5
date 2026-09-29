"""Shared building blocks for the optimizers in this package.

Most of the grokking optimizers are "AdamW with a modified gradient". They all
finish with :func:`adamw_update_`, which reproduces
``torch.optim.AdamW(foreach=False)`` bit for bit on fp32 tensors, so a
difference between two optimizers can only come from what they do *before* the
Adam tail.
"""

from __future__ import annotations

import torch
from torch import Tensor


def adamw_update_(
    p: Tensor,
    g: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """One decoupled-weight-decay Adam step on ``p`` in place.

    ``step`` is this parameter's own step count *after* incrementing (1 on the
    first update). Bias corrections are per parameter, as in torch.
    """
    if weight_decay != 0:
        p.mul_(1 - lr * weight_decay)
    exp_avg.lerp_(g, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
    p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def adam_state(state: dict, p: Tensor) -> tuple[Tensor, Tensor]:
    """Lazily create torch-compatible Adam state (``step``, ``exp_avg``, ``exp_avg_sq``)."""
    if "step" not in state:
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    return state["exp_avg"], state["exp_avg_sq"]


def check_common(lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
    if lr < 0:
        raise ValueError(f"invalid lr {lr}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"invalid betas {betas}")
    if eps < 0:
        raise ValueError(f"invalid eps {eps}")
    if weight_decay < 0:
        raise ValueError(f"invalid weight_decay {weight_decay}")


def state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Bytes held in tensors of ``optimizer.state`` (excludes the parameters themselves)."""
    seen, total = set(), 0

    def visit(obj):
        nonlocal total
        if isinstance(obj, Tensor):
            key = (obj.data_ptr(), obj.numel())
            if key not in seen and obj.numel() > 1:
                seen.add(key)
                total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)

    for st in optimizer.state.values():
        visit(st)
    extra = getattr(optimizer, "extra_state_bytes", None)
    return total + (extra() if callable(extra) else 0)
