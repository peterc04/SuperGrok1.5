"""Muon with the Kimi K3 "Per-Head Muon" split, plus an auxiliary AdamW.

Muon (Keller Jordan, 2024) keeps one momentum buffer per weight matrix and
replaces the update with an approximation of its polar factor ``U V^T``,
computed by a few Newton-Schulz iterations. This is the Moonlight / Kimi form
(arXiv 2502.16982, 2507.20534, 2607.24653), also the one DeepSeek-V4 used:

    M = mu M + G                        heavy-ball momentum
    U = G + mu M                        Nesterov input
    O = NewtonSchulz(U)                 per independent block, in bf16
    W = W (1 - lr wd) - lr * update_rms * sqrt(max(rows, cols)) * O

The ``update_rms * sqrt(max(rows, cols))`` factor makes the update RMS about
``update_rms`` (0.2), the same as AdamW's, so AdamW's learning rate and weight
decay carry over. Scale uses the shape of each *block*: scaling per-head blocks
by the full matrix's shape would double their update.

Blocks. A matrix group's ``row_blocks`` splits each weight's rows into that many
equal matrices that are orthogonalized separately. Two reasons to split:

* ``wo_a`` in DeepSeek-V4.1 stores one independent matrix per output group, so it
  must always be split (orthogonalizing the stack would be a different, wrong update);
* Kimi K3 Per-Head Muon splits the query (and key/value) projections by head so a
  few high-magnitude heads cannot dominate the shared polar factor. In
  DeepSeek-V4.1's MQA that means ``wq_b`` (and the indexer's ``wq_b``); the single
  shared KV head has nothing to split.

Parameters that are not hidden matrices (embeddings, the output head, norm gains,
biases, attention sinks, mHC static biases/scales, Engram tables and gains) go
to the auxiliary AdamW groups (``use_muon=False``). Use :func:`muon_param_groups`
to build the groups from a model's ``param_roles()``. Leading dimensions of a
Muon parameter are treated as a batch of independent matrices.

QK-Clip (Kimi K2) is available as :meth:`Muon.qk_clip`; it is off unless called.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ._core import adam_state, adamw_update_

KELLER = (3.4445, -4.7750, 2.0315)
NS_KELLER = (KELLER,) * 5
NS_DEEPSEEK_V4 = (KELLER,) * 8 + ((2.0, -1.5, 0.5),) * 2  # hybrid schedule: last two steps settle singular values at 1


@torch.no_grad()
def newton_schulz(
    g: torch.Tensor,
    schedule: Sequence[tuple[float, float, float]] = NS_KELLER,
    eps: float = 1e-7,
    dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Approximate orthogonalization of the last two dims (batched), Keller's quintic iteration."""
    x = g if dtype is None else g.to(dtype)
    tall = g.size(-2) > g.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in schedule:
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    flops_depend_on_routing = True  # Newton-Schulz runs for each matrix with a gradient

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_schedule: Sequence[tuple[float, float, float]] = NS_KELLER,
        ns_eps=1e-7,
        ns_dtype: torch.dtype | None = torch.bfloat16,
        update_rms=0.2,
        row_blocks=1,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        if lr < 0 or weight_decay < 0 or not 0.0 <= momentum < 1.0:
            raise ValueError("invalid lr / weight_decay / momentum")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_schedule=tuple(ns_schedule),
            ns_eps=ns_eps,
            ns_dtype=ns_dtype,
            update_rms=update_rms,
            row_blocks=row_blocks,
            use_muon=True,
            betas=tuple(adamw_betas),
            eps=adamw_eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(f"Muon group got a {p.ndim}-D parameter; route it to a use_muon=False group")
                if p.shape[-2] % group["row_blocks"]:
                    raise ValueError(f"{p.shape[-2]} rows are not divisible into {group['row_blocks']} blocks")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon_group if group["use_muon"] else self._adamw_group)(group)
        return loss

    def _muon_group(self, group):
        lr, wd, mu, nb = group["lr"], group["weight_decay"], group["momentum"], group["row_blocks"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.float()
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(g)
            u = g.add(buf, alpha=mu) if group["nesterov"] else buf
            rows, cols = p.shape[-2], p.shape[-1]
            blocks = u.reshape(*p.shape[:-2], nb, rows // nb, cols)
            o = newton_schulz(blocks, group["ns_schedule"], group["ns_eps"], group["ns_dtype"])
            scale = group["update_rms"] * math.sqrt(max(rows // nb, cols))
            p.mul_(1 - lr * wd)
            p.add_(o.reshape(p.shape).to(p.dtype), alpha=-lr * scale)

    def _adamw_group(self, group):
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

    @torch.no_grad()
    def qk_clip(self, stats, tau: float = 100.0):
        """Kimi K2 QK-Clip for shared-KV attention; call after ``step()``.

        ``stats``: iterable of ``(wq_b, n_heads, max_logit)`` with ``max_logit`` [n_heads], the largest
        pre-softmax logit each head produced this step. Heads above ``tau`` have their query block scaled
        by ``tau / max_logit`` (the key is shared by all heads, so only the query side can be clipped).
        """
        for w, n_heads, max_logit in stats:
            gamma = (tau / max_logit.float()).clamp(max=1.0)
            w.view(n_heads, -1, w.shape[-1]).mul_(gamma.view(-1, 1, 1).to(w.dtype))


def muon_param_groups(
    model,
    *,
    lr,
    weight_decay,
    adamw_lr=None,
    adamw_weight_decay=None,
    per_head=True,
    decay_mask=None,
    lr_scales=None,
    **muon_overrides,
):
    """Muon / AdamW parameter groups from ``model.param_roles()``.

    Hidden matrices (``kind == "matrix"``) go to Muon, grouped by their row-block count:
    ``blocks`` (mandatory independent matrices, e.g. ``wo_a`` per output group) times
    ``head_blocks`` when ``per_head`` (Kimi K3 Per-Head Muon). Everything else goes to AdamW.
    ``decay_mask`` / ``lr_scales``: optional {name: bool} / {name: float} per-parameter
    policies (e.g. DeepSeek's no-decay set and 5x Engram lr) applied to both families.
    """
    roles = model.param_roles()
    adamw_lr = lr if adamw_lr is None else adamw_lr
    adamw_weight_decay = weight_decay if adamw_weight_decay is None else adamw_weight_decay
    buckets: dict[tuple, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = roles[name]
        decays = True if decay_mask is None else decay_mask[name]
        scale = 1.0 if lr_scales is None else lr_scales[name]
        if role.kind == "matrix":
            key = ("muon", role.blocks * (role.head_blocks if per_head else 1), decays, scale)
        else:
            key = ("adamw", 1, decays, scale)
        buckets.setdefault(key, []).append(p)
    groups = []
    for (family, row_blocks, decays, scale), params in buckets.items():
        if family == "muon":
            groups.append(
                dict(
                    params=params,
                    use_muon=True,
                    row_blocks=row_blocks,
                    lr=lr * scale,
                    weight_decay=weight_decay if decays else 0.0,
                    **muon_overrides,
                )
            )
        else:
            groups.append(
                dict(
                    params=params,
                    use_muon=False,
                    lr=adamw_lr * scale,
                    weight_decay=adamw_weight_decay if decays else 0.0,
                )
            )
    return groups
