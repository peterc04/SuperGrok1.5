"""Pallas/TPU launch glue for AdamW.

Algorithm: csrc/algorithms/adamw.h
Self-contained — inlines the per-tensor AdamW math (formerly imported
from primitives.py::adamw_step).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


def launch_adamw_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-tensor AdamW step. Returns new (param, exp_avg, exp_avg_sq)."""
    m = beta1 * exp_avg + (1.0 - beta1) * grad
    v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    m_hat = m / bc1
    v_hat = v / bc2
    update = m_hat / (jnp.sqrt(v_hat) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v


launch_adamw_step_jit = jax.jit(
    launch_adamw_step, static_argnames=("lr", "beta1", "beta2", "eps", "wd")
)
