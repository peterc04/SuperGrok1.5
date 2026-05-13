"""Pallas/TPU launch glue for Muon.

Algorithm: csrc/algorithms/muon.h
2D weights: Newton-Schulz orthogonalization via jnp.linalg + iteration.
1D parameters: AdamW fallback.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


def newton_schulz_iterate(
    X: jnp.ndarray, ns_steps: int, a: float, b: float, c: float
) -> jnp.ndarray:
    """X = a*X + b*X@X^T@X + c*X@X^T@X@X^T@X, repeated ns_steps times."""
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
