"""Pallas/TPU launch glue for LookSAM (4 operations).

Algorithm: csrc/algorithms/looksam.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/simple_optimizers_jax.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class LookSAMState(NamedTuple):
    """Per-parameter state for LookSAM."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    direction: jnp.ndarray    # sharpness-aware direction
    step: jnp.ndarray


class LookSAMConfig(NamedTuple):
    """LookSAM hyperparameters."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    rho: float = 0.05        # SAM perturbation radius
    sam_alpha: float = 0.5   # direction blending coefficient


def init_looksam_state(param: jnp.ndarray) -> LookSAMState:
    flat = param.reshape(-1)
    return LookSAMState(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        direction=jnp.zeros_like(flat),
        step=jnp.array(0, dtype=jnp.int32),
    )


def looksam_perturb(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    rho: float,
) -> jnp.ndarray:
    """Perturb parameter for SAM: p + rho * g / ||g||.

    Returns perturbed parameter (original is unchanged — functional).
    """
    g = grad.reshape(-1).astype(jnp.float32)
    gnorm = jnp.linalg.norm(g)
    perturbation = jnp.where(gnorm > 0, rho * g / (gnorm + 1e-12), jnp.zeros_like(g))
    return (param.reshape(-1) + perturbation).reshape(param.shape)


def looksam_compute_direction(
    perturbed_grad: jnp.ndarray,
    orig_grad: jnp.ndarray,
) -> jnp.ndarray:
    """Compute sharpness-aware direction: normalize(perturbed - original)."""
    diff = perturbed_grad.reshape(-1) - orig_grad.reshape(-1)
    dnorm = jnp.linalg.norm(diff)
    return jnp.where(dnorm > 0, diff / dnorm, jnp.zeros_like(diff))


def looksam_adjust_grad(
    grad: jnp.ndarray,
    direction: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Adjust gradient using sharpness direction: g + alpha * (g·d) * d."""
    g = grad.reshape(-1).astype(jnp.float32)
    proj = jnp.sum(g * direction)
    return (g + alpha * proj * direction).reshape(grad.shape)


def looksam_adam_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: LookSAMState,
    config: LookSAMConfig,
) -> Tuple[jnp.ndarray, LookSAMState]:
    """Adam step for LookSAM (after gradient adjustment)."""
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * g
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * g ** 2

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step
    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = LookSAMState(
        exp_avg=new_ea, exp_avg_sq=new_easq,
        direction=state.direction, step=step)
    return new_p.reshape(param.shape), new_state


# ---------------------------------------------------------------------------
# Per-tensor fused-TU contract (4 separate operations called by race driver)
# ---------------------------------------------------------------------------

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
    m = beta1 * exp_avg + (1.0 - beta1) * g_adj
    v = beta2 * exp_avg_sq + (1.0 - beta2) * g_adj * g_adj
    update = (m * bc1) / (jnp.sqrt(v * bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v
