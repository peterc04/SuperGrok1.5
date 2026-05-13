"""Pallas/TPU launch glue for LookSAM (4 operations).

Algorithm: csrc/algorithms/looksam.h
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from csrc.backends.pallas.primitives import adamw_step


def launch_looksam_perturb(
    param: jnp.ndarray, grad: jnp.ndarray, scale: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (perturbed_param, backup_of_original)."""
    backup = param
    return param + scale * grad, backup


def launch_looksam_restore(backup: jnp.ndarray) -> jnp.ndarray:
    return backup


def launch_looksam_set_direction(
    grad_sam: jnp.ndarray, grad_orig: jnp.ndarray
) -> jnp.ndarray:
    return grad_sam - grad_orig


def launch_looksam_apply(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    sam_dir: jnp.ndarray,
    grad: jnp.ndarray,
    alpha: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    g_adj = (1.0 - alpha) * grad + alpha * sam_dir
    return adamw_step(param, exp_avg, exp_avg_sq, g_adj,
                      lr, beta1, beta2, eps, wd, bc1, bc2)
