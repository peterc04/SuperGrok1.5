"""Pallas/TPU launch glue for NeuralGrok.

Algorithm: csrc/algorithms/neuralgrok.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/metanet_optimizers_jax.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class NeuralGrokState(NamedTuple):
    """Per-parameter state for NeuralGrok."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    step: jnp.ndarray


class NeuralGrokWeights(NamedTuple):
    """Amplifier MLP weights for NeuralGrok."""
    W1: jnp.ndarray      # [hidden_dim, 1]
    b1: jnp.ndarray      # [hidden_dim]
    W_last: jnp.ndarray  # [1, hidden_dim]
    b_last: jnp.ndarray  # [1]
    alpha: float
    beta: float


class NeuralGrokConfig(NamedTuple):
    """NeuralGrok hyperparameters."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    gradient_clipping: float = 10.0
    hidden_dim: int = 32


def init_neuralgrok_state(param: jnp.ndarray) -> NeuralGrokState:
    flat = param.reshape(-1)
    return NeuralGrokState(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        step=jnp.array(0, dtype=jnp.int32),
    )


def neuralgrok_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: NeuralGrokState,
    weights: NeuralGrokWeights,
    config: NeuralGrokConfig,
) -> Tuple[jnp.ndarray, NeuralGrokState]:
    """One NeuralGrok step. scale = alpha*MLP(|g|) + beta; smart_grad = g*scale; Adam."""
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    abs_g = jnp.abs(g)[:, None]
    hidden = jax.nn.relu(abs_g @ weights.W1.T + weights.b1)
    mlp_out = (hidden @ weights.W_last.T + weights.b_last).squeeze(-1)
    scale = weights.alpha * mlp_out + weights.beta
    smart_g = g * scale

    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * smart_g
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * smart_g ** 2

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step
    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = NeuralGrokState(exp_avg=new_ea, exp_avg_sq=new_easq, step=step)
    return new_p.reshape(param.shape), new_state


# ---------------------------------------------------------------------------
# Per-tensor fused-TU contract
# ---------------------------------------------------------------------------

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
    m = beta1 * exp_avg + (1.0 - beta1) * g_amp
    v = beta2 * exp_avg_sq + (1.0 - beta2) * g_amp * g_amp
    update = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v
