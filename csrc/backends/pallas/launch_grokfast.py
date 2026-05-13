"""Pallas/TPU launch glue for Grokfast.

Algorithm: csrc/algorithms/grokfast.h
Self-contained — inlines the Grokfast EMA amplification used both as a
preprocessing primitive and inside the fused Adam launcher.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class GrokfastState(NamedTuple):
    """Per-parameter state for Grokfast."""
    ema: jnp.ndarray  # gradient EMA


class GrokfastConfig(NamedTuple):
    """Grokfast hyperparameters."""
    alpha: float = 0.98
    lamb: float = 5.0


def init_grokfast_state(param: jnp.ndarray) -> GrokfastState:
    return GrokfastState(ema=jnp.zeros(param.reshape(-1).shape))


def grokfast_amplify(
    grad: jnp.ndarray,
    state: GrokfastState,
    config: GrokfastConfig,
) -> Tuple[jnp.ndarray, GrokfastState]:
    """Grokfast EMA amplification (pre-processing).

    Returns amplified gradient and updated state; pair with any base
    optimizer (typically AdamW) for the full update.
    """
    g = grad.reshape(-1).astype(jnp.float32)
    new_ema = config.alpha * state.ema + (1.0 - config.alpha) * g
    amplified = g + config.lamb * new_ema
    return amplified.reshape(grad.shape), GrokfastState(ema=new_ema)


def launch_grokfast_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    ema: jnp.ndarray,
    grad: jnp.ndarray,
    gf_alpha: float,
    gf_lamb: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-tensor Grokfast + Adam step (fused-TU contract)."""
    new_ema = gf_alpha * ema + (1.0 - gf_alpha) * grad
    g_amp = grad + gf_lamb * new_ema
    m = beta1 * exp_avg + (1.0 - beta1) * g_amp
    v = beta2 * exp_avg_sq + (1.0 - beta2) * g_amp * g_amp
    update = (m * bc1) / (jnp.sqrt(v * bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v, new_ema


launch_grokfast_step_jit = jax.jit(
    launch_grokfast_step,
    static_argnames=("gf_alpha", "gf_lamb", "lr", "beta1", "beta2", "eps", "wd"),
)
