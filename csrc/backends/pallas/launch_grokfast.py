"""Pallas/TPU launch glue for Grokfast (fused EMA + Adam).

Algorithm: csrc/algorithms/grokfast.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import ema_update, adamw_step


def launch_grokfast_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    ema: jnp.ndarray,
    grad: jnp.ndarray,
    gf_alpha: float,
    gf_lamb: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    new_ema = ema_update(ema, grad, gf_alpha)
    g_amp = grad + gf_lamb * new_ema
    new_param, new_m, new_v = adamw_step(
        param, exp_avg, exp_avg_sq, g_amp,
        lr, beta1, beta2, eps, wd, bc1, bc2,
    )
    return new_param, new_m, new_v, new_ema


launch_grokfast_step_jit = jax.jit(
    launch_grokfast_step,
    static_argnames=("gf_alpha", "gf_lamb", "lr", "beta1", "beta2", "eps", "wd"),
)
