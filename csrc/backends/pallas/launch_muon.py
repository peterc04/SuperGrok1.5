"""Pallas/TPU launch glue for Muon.

Algorithm: csrc/algorithms/muon.h
Self-contained — inlines the functional Newton-Schulz implementation
that previously lived in supergrok2_jax_tpu/simple_optimizers_jax.py.

2D weights: Newton-Schulz orthogonalization
1D parameters: AdamW fallback
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple


class MuonState(NamedTuple):
    """Per-parameter state for Muon."""
    momentum_buf: jnp.ndarray


class MuonConfig(NamedTuple):
    """Muon hyperparameters."""
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    ns_steps: int = 5


def init_muon_state(param: jnp.ndarray) -> MuonState:
    return MuonState(momentum_buf=jnp.zeros_like(param))


def _newton_schulz_ortho(X: jnp.ndarray, ns_steps: int) -> jnp.ndarray:
    """Newton-Schulz orthogonalization.

    Muon paper coefficients: a=3.4445, b=-4.7750, c=2.0315.
    Iterates: X <- a*X + b*(X @ X.T) @ X + c*X @ (X.T @ X)
    Converges to the nearest orthogonal matrix.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    def ns_body(_, X):
        A = X @ X.T
        return X * a + (A @ X) * b + (X @ (X.T @ X)) * c

    return lax.fori_loop(0, ns_steps, ns_body, X)


def muon_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: MuonState,
    config: MuonConfig,
) -> Tuple[jnp.ndarray, MuonState]:
    """One Muon step. Newton-Schulz orthogonalized momentum."""
    g = grad.astype(jnp.float32)
    new_m = config.momentum * state.momentum_buf + g

    if param.ndim >= 2:
        X = new_m
        norm_val = jnp.linalg.norm(X)
        X_normed = jnp.where(norm_val > 0, X / norm_val, X)
        X_ortho = _newton_schulz_ortho(X_normed, config.ns_steps)
        update = X_ortho * norm_val
    else:
        update = new_m

    new_p = param * (1.0 - config.lr * config.weight_decay) - config.lr * update
    return new_p, MuonState(momentum_buf=new_m)


def newton_schulz_iterate(
    X: jnp.ndarray, ns_steps: int, a: float, b: float, c: float
) -> jnp.ndarray:
    """Unrolled Newton-Schulz variant used by the fused-TU launcher."""
    for _ in range(ns_steps):
        AX = X.T @ X
        AAX = AX @ AX
        X = a * X + b * (X @ AX) + c * (X @ AAX)
    return X


def launch_muon_step(
    param: jnp.ndarray,
    buf: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    momentum: float,
    wd: float,
    ns_steps: int = 5,
    ns_a: float = 3.4445,
    ns_b: float = -4.7750,
    ns_c: float = 2.0315,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-tensor Muon step (fused-TU contract)."""
    new_buf = momentum * buf + (1.0 - momentum) * grad
    if param.ndim >= 2:
        frob = jnp.linalg.norm(new_buf) + 1e-8
        X = new_buf / frob
        X = newton_schulz_iterate(X, ns_steps, ns_a, ns_b, ns_c)
        max_dim = max(param.shape[-1], param.shape[-2])
        neg_lr_scale = -lr * 0.2 * jnp.sqrt(jnp.float32(max_dim))
        new_param = param * (1.0 - lr * wd) + neg_lr_scale * X
    else:
        new_param = param - lr * new_buf
    return new_param, new_buf
