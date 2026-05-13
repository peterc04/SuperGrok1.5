"""Pallas/TPU launch glue for SuperGrok v1.1.

Algorithm: csrc/algorithms/supergrok11.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


def phi_forward(
    grad: jnp.ndarray, sharpness: jnp.ndarray,
    W1: jnp.ndarray, b1: jnp.ndarray,
    W2: jnp.ndarray, b2: float,
) -> jnp.ndarray:
    x = jnp.stack([grad.reshape(-1), sharpness.reshape(-1)], axis=1)
    h = jnp.tanh(x @ W1.T + b1)
    mu = (h @ W2[:, None] + b2).reshape(grad.shape)
    return mu


def launch_supergrok11_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    mu_buf: jnp.ndarray,
    grad: jnp.ndarray,
    sharpness: jnp.ndarray,
    momentum: jnp.ndarray,
    phi_W1: jnp.ndarray, phi_b1: jnp.ndarray,
    phi_W2: jnp.ndarray, phi_b2: float,
    alpha: float,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> Tuple[jnp.ndarray, ...]:
    mu = phi_forward(grad, sharpness, phi_W1, phi_b1, phi_W2, phi_b2)

    # Cosine gate
    gn = jnp.sum(grad * momentum)
    gdg = jnp.sum(grad * grad)
    gdm = jnp.sum(momentum * momentum)
    denom = jnp.sqrt(gdg * gdm + 1e-12)
    gate = jnp.clip(gn / denom, 0.0, 1.0)

    smart = grad + (1.0 - gate) * alpha * mu
    new_param, new_m, new_v = adamw_step(
        param, exp_avg, exp_avg_sq, smart,
        lr, beta1, beta2, eps, wd, bc1, bc2,
    )
    return new_param, new_m, new_v, mu
