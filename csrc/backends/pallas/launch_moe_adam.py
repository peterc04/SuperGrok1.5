"""Pallas/TPU launch glue for MoE/Adam multi-tensor.

Algorithm: csrc/algorithms/moe_adam.h
Self-contained — the optimizer math is the standard AdamW update;
expert-aware routing is handled in the model wrapper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


def launch_moe_adam_step(
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
    """Per-tensor AdamW step for MoE-aware optimizer."""
    m = beta1 * exp_avg + (1.0 - beta1) * grad
    v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    update = (m * bc1) / (jnp.sqrt(v * bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v


launch_moe_adam_step_jit = jax.jit(
    launch_moe_adam_step,
    static_argnames=("lr", "beta1", "beta2", "eps", "wd"),
)
