"""The race's optimizers, as plain PyTorch ``torch.optim.Optimizer`` subclasses (CPU or GPU).

Each is checked against the published algorithm or the authors' own code in
``tests/``. See ``docs/ALGORITHMS.md`` for how they work and what they cost.

Race hyperparameters (``OptimizerSpec.race_defaults``) follow one rule: every
Adam-family method shares the grokking base recipe of Power et al. (2022),
lr 1e-3, betas (0.9, 0.98), decoupled weight decay 1.0, and adds only its own
mechanism on top, so a difference in the race is attributable to that mechanism.
Lion and Prodigy use their own step-size conventions (see their docstrings).

Training-loop contract (see ``grokking_race/trainer.py``): optimizers may declare
``needs_closure``, ``needs_meta_loss``, ``needs_train_meta_loss``,
``uses_step_loss`` or ``wants_losses``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ._core import state_bytes
from .adamw import AdamW
from .grokadamw import GrokAdamW
from .grokfast import Grokfast
from .lion import Lion
from .looksam import LookSAM
from .muon import Muon, muon_param_groups
from .neuralgrok import NeuralGrok
from .prodigy import Prodigy
from .shampoo import Shampoo
from .supergrok11 import SharpnessMetaNet, SuperGrok11, block_layer_ids

BASE = dict(lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    cls: type
    race_defaults: dict = field(default_factory=dict)
    build: Callable | None = None  # (model, **hp) -> Optimizer; default: cls(model.parameters(), **hp)


def _param_groups(model, policy: str, lr: float, weight_decay: float, allow_lr_scale: bool = True):
    """One group (``uniform``: every parameter decayed, the grokking convention) or DeepSeek's
    per-role policy (``deepseek``: no decay on sinks / mHC static terms / Engram tables, Engram lr x5)."""
    if policy == "uniform":
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    if policy != "deepseek":
        raise ValueError(f"unknown parameter policy {policy!r}")
    roles, buckets = model.param_roles(), {}
    for name, p in model.named_parameters():
        role = roles[name]
        scale = role.lr_scale if allow_lr_scale else 1.0
        buckets.setdefault((role.decay, scale), []).append(p)
    return [{"params": ps, "lr": lr * s, "weight_decay": weight_decay if d else 0.0} for (d, s), ps in buckets.items()]


def _default_build(cls, allow_lr_scale=True):
    def build(model, policy="uniform", **hp):
        groups = _param_groups(model, policy, hp.get("lr", 1e-3), hp.get("weight_decay", 0.0), allow_lr_scale)
        return cls(groups, **hp)

    return build


def _build_muon(
    model, policy="uniform", per_head=True, lr=1e-3, weight_decay=1.0, adamw_lr=None, adamw_betas=(0.9, 0.98), **hp
):
    decay_mask = lr_scales = None
    if policy == "deepseek":
        roles = model.param_roles()
        decay_mask = {n: r.decay for n, r in roles.items()}
        lr_scales = {n: r.lr_scale for n, r in roles.items()}
    groups = muon_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        adamw_lr=adamw_lr,
        per_head=per_head,
        decay_mask=decay_mask,
        lr_scales=lr_scales,
    )
    return Muon(groups, lr=lr, weight_decay=weight_decay, adamw_betas=adamw_betas, **hp)


def _build_supergrok11(model, policy="uniform", layer_ids="blocks", **hp):
    """``layer_ids="blocks"``: layer-wise beta1 counts transformer blocks; ``"tensor"``: the legacy tensor
    index (position in the model's parameter order, whatever the grouping)."""
    groups = _param_groups(model, policy, hp.get("lr", 1e-3), hp.get("weight_decay", 1.0))
    if layer_ids in ("blocks", "tensor"):
        named = list(model.named_parameters())
        ids = block_layer_ids([n for n, _ in named]) if layer_ids == "blocks" else range(len(named))
        by_param = dict(zip((p for _, p in named), ids))
        layer_ids = [by_param[p] for g in groups for p in g["params"]]
    return SuperGrok11(groups, layer_ids=layer_ids, **hp)


def _build_shampoo(model, policy="uniform", shampoo_on="matrices", lr=1e-3, weight_decay=1.0, **hp):
    """Shampoo on the hidden matrices (``shampoo_on="matrices"``: the parameters Muon orthogonalizes) and
    Adam, Shampoo's own grafting method, on embeddings, tables, norms and other vectors. ``"all"``
    preconditions every parameter, tables included (memory and eigendecompositions grow with the table)."""
    if shampoo_on not in ("matrices", "all"):
        raise ValueError(f"shampoo_on must be 'matrices' or 'all', got {shampoo_on!r}")
    if policy not in ("uniform", "deepseek"):
        raise ValueError(f"unknown parameter policy {policy!r}")
    roles = model.param_roles() if hasattr(model, "param_roles") else None
    buckets: dict = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = roles[name] if roles is not None else None
        is_matrix = role.kind == "matrix" if role is not None else p.ndim >= 2
        decay, scale = (role.decay, role.lr_scale) if (policy == "deepseek" and role is not None) else (True, 1.0)
        buckets.setdefault((shampoo_on == "all" or is_matrix, decay, scale), []).append(p)
    groups = [
        {"params": ps, "use_shampoo": use, "lr": lr * s, "weight_decay": weight_decay if d else 0.0}
        for (use, d, s), ps in buckets.items()
    ]
    return Shampoo(groups, lr=lr, weight_decay=weight_decay, **hp)


# The legacy race's SuperGrok 1.1 settings (lamb 1.0 was the last value raced) with every component on.
SUPERGROK11 = dict(
    BASE,
    alpha_init=0.98,
    lamb=1.0,
    gamma=0.1,
    kappa=0.1,
    warmup_steps=100,
    warmup_ramp=100,
    gradient_clipping=1.0,
    gate_temperature=5.0,
    alpha_update_freq=50,
    sam_rho=0.05,
    sam_every=10,
    meta_hidden_dim=32,
    meta_lr=1e-4,
    meta_update_freq=5,
    zero_grad_policy="mask",
    layer_ids="blocks",
)

OPTIMIZERS: dict[str, OptimizerSpec] = {
    spec.name: spec
    for spec in [
        OptimizerSpec("adamw", AdamW, dict(BASE)),
        OptimizerSpec("lion", Lion, dict(lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0)),
        OptimizerSpec("grokfast", Grokfast, dict(BASE, alpha=0.98, lamb=2.0)),
        OptimizerSpec(
            "grokadamw",
            GrokAdamW,
            dict(BASE, alpha_init=0.98, lamb=2.0, gamma=0.1, grokking_signal_decay_rate=0.1, gradient_clipping=1.0),
        ),
        OptimizerSpec("looksam", LookSAM, dict(BASE, rho=0.05, k=5, alpha=0.7)),
        OptimizerSpec(
            "prodigy",
            Prodigy,
            dict(lr=1.0, betas=(0.9, 0.98), weight_decay=1.0),
            _default_build(Prodigy, allow_lr_scale=False),
        ),
        OptimizerSpec(
            "neuralgrok",
            NeuralGrok,
            dict(BASE, decoupled_weight_decay=True, amp_hidden_dims=(128, 128), amp_c=1.0, meta_every=4, meta_lr=1e-4),
        ),
        OptimizerSpec(
            "muon",
            Muon,
            dict(
                lr=1e-3,
                weight_decay=1.0,
                momentum=0.95,
                nesterov=True,
                update_rms=0.2,
                per_head=True,
                adamw_betas=(0.9, 0.98),
            ),
            _build_muon,
        ),
        # Meta's Distributed Shampoo "replace Adam" recipe on the shared base: Adam grafting (step sizes are Adam's)
        OptimizerSpec(
            "shampoo",
            Shampoo,
            dict(
                BASE,
                epsilon=1e-12,
                grafting_beta2=0.98,
                grafting_epsilon=1e-8,
                max_preconditioner_dim=1024,
                precondition_frequency=10,
                start_preconditioning_step=-1,
                shampoo_on="matrices",
            ),
            _build_shampoo,
        ),
        OptimizerSpec("supergrok11", SuperGrok11, dict(SUPERGROK11), _build_supergrok11),
        # control: the same optimizer with the learned correction and its SAM probe off, i.e. AdamW with
        # block-wise beta1 and a per-tensor clip, on the same data (it carves the same meta split)
        OptimizerSpec(
            "supergrok11_frozen",
            SuperGrok11,
            dict(SUPERGROK11, lamb=0.0, meta_update_freq=0, sam_rho=0.0),
            _build_supergrok11,
        ),
    ]
}


def get_spec(name: str) -> OptimizerSpec:
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}; choose from {list(OPTIMIZERS)}")
    return OPTIMIZERS[name]


def build_optimizer(name: str, model: torch.nn.Module, policy: str = "uniform", **hp) -> torch.optim.Optimizer:
    """Build ``name`` for ``model`` with the race defaults, overridden by ``hp``."""
    spec = get_spec(name)
    hp = {**spec.race_defaults, **hp}
    build = spec.build or _default_build(spec.cls)
    return build(model, policy=policy, **hp)


__all__ = [
    "AdamW",
    "GrokAdamW",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "NeuralGrok",
    "Prodigy",
    "SharpnessMetaNet",
    "Shampoo",
    "SuperGrok11",
    "OPTIMIZERS",
    "OptimizerSpec",
    "block_layer_ids",
    "build_optimizer",
    "get_spec",
    "muon_param_groups",
    "state_bytes",
]
