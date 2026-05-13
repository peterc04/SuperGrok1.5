"""Pallas/TPU launch glue for AdamW.

Algorithm: csrc/algorithms/adamw.h (C++ math spec; Python mirror in
csrc/backends/pallas/primitives.py::adamw_step).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


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
    return adamw_step(param, exp_avg, exp_avg_sq, grad,
                      lr, beta1, beta2, eps, wd, bc1, bc2)


# JIT-compiled variant for hot loops.
launch_adamw_step_jit = jax.jit(
    launch_adamw_step, static_argnames=("lr", "beta1", "beta2", "eps", "wd")
)
