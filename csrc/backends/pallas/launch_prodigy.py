"""Pallas/TPU launch glue for Prodigy.

Algorithm: csrc/algorithms/prodigy.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/simple_optimizers_jax.py.

On TPU the d_t update is fused into the JIT graph; no separate device-
resident scalar is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class ProdigyState(NamedTuple):
    """Per-parameter state for Prodigy."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    s_buf: jnp.ndarray       # distance-weighted accumulator
    param_init: jnp.ndarray  # initial parameter snapshot
    step: jnp.ndarray        # scalar int32


class ProdigyConfig(NamedTuple):
    """Prodigy hyperparameters."""
    lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    d_lr_init: float = 1.0


def init_prodigy_state(param: jnp.ndarray) -> ProdigyState:
    flat = param.reshape(-1)
    return ProdigyState(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        s_buf=jnp.zeros_like(flat),
        param_init=flat.copy(),
        step=jnp.array(0, dtype=jnp.int32),
    )


def prodigy_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: ProdigyState,
    config: ProdigyConfig,
    d_lr: float,
) -> Tuple[jnp.ndarray, ProdigyState, float]:
    """One Prodigy step. Returns (new_param, new_state, new_d_lr).

    Distance-based adaptive LR estimation + Adam.
    """
    step = state.step + 1
    p_flat = param.reshape(-1).astype(jnp.float32)
    g = grad.reshape(-1).astype(jnp.float32)

    num = jnp.sum(g * (p_flat - state.param_init))
    den = jnp.sum(state.s_buf * jnp.abs(g))

    # Keep d_lr as a device scalar — never force a host sync via float(...),
    # which would block the device queue and serialise the step.
    new_d_lr = jnp.where(den > 0, jnp.maximum(d_lr, num / den), d_lr)

    new_s = config.beta2 * state.s_buf + (1.0 - config.beta2) * jnp.abs(g) * d_lr
    effective = g * d_lr

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step

    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * effective
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * effective ** 2

    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = ProdigyState(
        exp_avg=new_ea, exp_avg_sq=new_easq, s_buf=new_s,
        param_init=state.param_init, step=step)
    return new_p.reshape(param.shape), new_state, new_d_lr


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
    """Per-tensor Prodigy step (fused-TU contract)."""
    delta = param_init - param
    r_sum = jnp.sum(grad * delta) * d_prev
    s_sum = jnp.sum(jnp.abs(grad)) * (d_prev * d_prev)
    candidate = r_sum / (jnp.abs(s_sum) + 1e-12)
    d = jnp.maximum(d_prev, candidate)

    g_scaled = d * grad
    m = beta1 * exp_avg + (1.0 - beta1) * g_scaled
    v = beta2 * exp_avg_sq + (1.0 - beta2) * g_scaled * g_scaled
    update = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_param = param - d * (update + wd * param)
    new_s = s_track + d * grad
    return new_param, m, v, new_s, d
