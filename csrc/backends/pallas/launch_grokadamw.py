"""Pallas/TPU launch glue for GrokAdamW.

Algorithm: csrc/algorithms/grokadamw.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import ema_update, adamw_step


def launch_grokadamw_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    ema: jnp.ndarray,
    grad: jnp.ndarray,
    alpha: float,
    lamb: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-tensor GrokAdamW step. Returns (param, exp_avg, exp_avg_sq, ema)."""
    new_ema = ema_update(ema, grad, alpha)
    g_amp = grad + lamb * new_ema
    new_param, new_m, new_v = adamw_step(
        param, exp_avg, exp_avg_sq, g_amp,
        lr, beta1, beta2, eps, wd, bc1, bc2,
    )
    return new_param, new_m, new_v, new_ema


launch_grokadamw_step_jit = jax.jit(
    launch_grokadamw_step,
    static_argnames=("alpha", "lamb", "lr", "beta1", "beta2", "eps", "wd"),
)
