"""Pallas/TPU launch glue for Lion.

Algorithm: csrc/algorithms/lion.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/simple_optimizers_jax.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class LionState(NamedTuple):
    """Per-parameter state for Lion."""
    exp_avg: jnp.ndarray  # momentum


class LionConfig(NamedTuple):
    """Lion hyperparameters."""
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1


def init_lion_state(param: jnp.ndarray) -> LionState:
    return LionState(exp_avg=jnp.zeros(param.reshape(-1).shape))


def lion_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: LionState,
    config: LionConfig,
) -> Tuple[jnp.ndarray, LionState]:
    """One Lion step. update = sign(beta1*m + (1-beta1)*g)."""
    g = grad.reshape(-1).astype(jnp.float32)
    m = state.exp_avg

    update = jnp.sign(config.beta1 * m + (1.0 - config.beta1) * g)

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - config.lr * update

    new_m = config.beta2 * m + (1.0 - config.beta2) * g

    return new_p.reshape(param.shape), LionState(exp_avg=new_m)


def launch_lion_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Tensor-level launch used by fused TU stubs.

    Mirrors the per-tensor primitive contract: returns (new_param, new_m).
    """
    interp = beta1 * exp_avg + (1.0 - beta1) * grad
    update = jnp.sign(interp)
    new_param = param - lr * (update + wd * param)
    new_exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    return new_param, new_exp_avg


launch_lion_step_jit = jax.jit(
    launch_lion_step, static_argnames=("lr", "beta1", "beta2", "wd")
)
