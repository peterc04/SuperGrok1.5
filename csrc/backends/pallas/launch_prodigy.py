"""Pallas/TPU launch glue for Prodigy.

Algorithm: csrc/algorithms/prodigy.h
On TPU the d_t update is fused into the JIT graph; no separate
device-resident scalar is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


def launch_prodigy_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    s_track: jnp.ndarray,
    param_init: jnp.ndarray,
    grad: jnp.ndarray,
    d_prev: float,
    beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    # Reduce r, s
    delta = param_init - param
    r_sum = jnp.sum(grad * delta) * d_prev
    s_sum = jnp.sum(jnp.abs(grad)) * (d_prev * d_prev)
    candidate = r_sum / (jnp.abs(s_sum) + 1e-12)
    d = jnp.maximum(d_prev, candidate)

    # Apply with d as effective lr
    g_scaled = d * grad
    new_param, new_m, new_v = adamw_step(
        param, exp_avg, exp_avg_sq, g_scaled,
        d, beta1, beta2, eps, wd, bc1, bc2,
    )
    new_s = s_track + d * grad
    return new_param, new_m, new_v, new_s, d
