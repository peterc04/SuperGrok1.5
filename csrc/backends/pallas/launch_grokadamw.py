"""Pallas/TPU launch glue for GrokAdamW.

Algorithm: csrc/algorithms/grokadamw.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/simple_optimizers_jax.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class GrokAdamWState(NamedTuple):
    """Per-parameter state for GrokAdamW."""
    exp_avg: jnp.ndarray       # first moment
    exp_avg_sq: jnp.ndarray    # second moment
    ema: jnp.ndarray           # gradient EMA for amplification
    step: jnp.ndarray          # scalar int32


class GrokAdamWConfig(NamedTuple):
    """GrokAdamW hyperparameters."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    alpha: float = 0.98
    lamb: float = 5.0
    gradient_clipping: float = 10.0


def init_grokadamw_state(param: jnp.ndarray) -> GrokAdamWState:
    flat = param.reshape(-1)
    return GrokAdamWState(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        ema=jnp.zeros_like(flat),
        step=jnp.array(0, dtype=jnp.int32),
    )


def grokadamw_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: GrokAdamWState,
    config: GrokAdamWConfig,
) -> Tuple[jnp.ndarray, GrokAdamWState]:
    """One GrokAdamW step. EMA amplification followed by Adam."""
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    new_ema = config.alpha * state.ema + (1.0 - config.alpha) * g
    effective = g + config.lamb * new_ema

    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * effective
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * effective ** 2

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step
    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = GrokAdamWState(
        exp_avg=new_ea, exp_avg_sq=new_easq, ema=new_ema, step=step)
    return new_p.reshape(param.shape), new_state


def launch_grokadamw_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    ema: jnp.ndarray,
    grad: jnp.ndarray,
    alpha: float,
    lamb: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-tensor GrokAdamW step (fused-TU contract)."""
    new_ema = alpha * ema + (1.0 - alpha) * grad
    g_amp = grad + lamb * new_ema
    m = beta1 * exp_avg + (1.0 - beta1) * g_amp
    v = beta2 * exp_avg_sq + (1.0 - beta2) * g_amp * g_amp
    update = (m * bc1) / (jnp.sqrt(v * bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v, new_ema


launch_grokadamw_step_jit = jax.jit(
    launch_grokadamw_step,
    static_argnames=("alpha", "lamb", "lr", "beta1", "beta2", "eps", "wd"),
)
