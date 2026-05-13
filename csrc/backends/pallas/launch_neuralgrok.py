"""Pallas/TPU launch glue for NeuralGrok.

Algorithm: csrc/algorithms/neuralgrok.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


def psi_forward(
    abs_grad: jnp.ndarray,
    W1: jnp.ndarray, b1: jnp.ndarray,
    W2: jnp.ndarray, b2: float,
) -> jnp.ndarray:
    """2-layer MLP forward: ReLU hidden, linear output."""
    ag = abs_grad.reshape(-1, 1)
    h = jnp.maximum(ag @ W1[None, :] + b1, 0.0)
    s = h @ W2[:, None] + b2
    return s.reshape(abs_grad.shape)


def launch_neuralgrok_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    grad: jnp.ndarray,
    psi_W1: jnp.ndarray, psi_b1: jnp.ndarray,
    psi_W2: jnp.ndarray, psi_b2: float,
    alpha: float, beta: float,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = psi_forward(jnp.abs(grad), psi_W1, psi_b1, psi_W2, psi_b2)
    g_amp = (s * alpha + beta) * grad
    return adamw_step(param, exp_avg, exp_avg_sq, g_amp,
                      lr, beta1, beta2, eps, wd, bc1, bc2)
