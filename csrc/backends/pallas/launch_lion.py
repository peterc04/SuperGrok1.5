"""Pallas/TPU launch glue for Lion.

Algorithm: csrc/algorithms/lion.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import lion_step


def launch_lion_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return lion_step(param, exp_avg, grad, lr, beta1, beta2, wd)


launch_lion_step_jit = jax.jit(
    launch_lion_step, static_argnames=("lr", "beta1", "beta2", "wd")
)
