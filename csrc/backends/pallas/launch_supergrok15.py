"""Pallas/TPU launch glue for SuperGrok v1.5.

Algorithm: csrc/algorithms/supergrok15.h
Self-contained — inlines the functional JAX implementation that previously
lived in supergrok2_jax_tpu/metanet_optimizers_jax.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class SuperGrok15State(NamedTuple):
    """Per-parameter state for SuperGrok v1.5."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    mu: jnp.ndarray           # gradient EMA
    sharpness: jnp.ndarray    # updated externally via SAM
    step: jnp.ndarray         # scalar int32


class SuperGrok15Weights(NamedTuple):
    """Meta-net weights for SuperGrok v1.5 (2-layer MLP)."""
    W1: jnp.ndarray   # [hidden_dim, 1]
    b1: jnp.ndarray   # [hidden_dim]
    W2: jnp.ndarray   # [1, hidden_dim]
    b2: jnp.ndarray   # [1]
    rescale: float


class SuperGrok15Config(NamedTuple):
    """SuperGrok v1.5 hyperparameters."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    alpha: float = 0.98
    lamb: float = 2.0
    gradient_clipping: float = 1.0
    hidden_dim: int = 32
    ramp: float = 1.0
    gate_signal: float = 1.0


def init_supergrok15_state(param: jnp.ndarray) -> SuperGrok15State:
    flat = param.reshape(-1)
    return SuperGrok15State(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        mu=jnp.zeros_like(flat),
        sharpness=jnp.zeros_like(flat),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _meta_mlp_forward(
    g: jnp.ndarray,
    W1: jnp.ndarray, b1: jnp.ndarray,
    W2: jnp.ndarray, b2: jnp.ndarray,
    rescale: float,
) -> jnp.ndarray:
    """Per-element meta MLP: smart_grad = grad + rescale * MLP(rescale*grad)."""
    scaled = g * rescale
    hidden = jax.nn.relu(scaled[:, None] * W1.T + b1)
    mlp_out = (hidden @ W2.T + b2).squeeze(-1)
    return g + rescale * mlp_out


def supergrok15_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: SuperGrok15State,
    weights: SuperGrok15Weights,
    config: SuperGrok15Config,
) -> Tuple[jnp.ndarray, SuperGrok15State]:
    """One SuperGrok v1.5 step (meta-net MLP -> mu EMA -> Adam)."""
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    smart_g = _meta_mlp_forward(
        g, weights.W1, weights.b1, weights.W2, weights.b2, weights.rescale)

    new_mu = config.alpha * state.mu + (1.0 - config.alpha) * g
    lamb_eff = config.lamb * config.ramp * config.gate_signal
    effective = smart_g + lamb_eff * new_mu

    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * effective
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * effective ** 2

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step
    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = SuperGrok15State(
        exp_avg=new_ea, exp_avg_sq=new_easq, mu=new_mu,
        sharpness=state.sharpness, step=step)
    return new_p.reshape(param.shape), new_state


# ---------------------------------------------------------------------------
# Per-tensor fused-TU contract (different signature shape; consumed by
# csrc/fused/tpu_v5p/fused_*_supergrok15_*.py stubs)
# ---------------------------------------------------------------------------

def phi_forward(
    grad: jnp.ndarray, sharpness: jnp.ndarray,
    W1: jnp.ndarray, b1: jnp.ndarray,
    W2: jnp.ndarray, b2: float,
) -> jnp.ndarray:
    x = jnp.stack([grad.reshape(-1), sharpness.reshape(-1)], axis=1)
    h = jnp.tanh(x @ W1.T + b1)
    mu = (h @ W2[:, None] + b2).reshape(grad.shape)
    return mu


def launch_supergrok15_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    mu_buf: jnp.ndarray,
    grad: jnp.ndarray,
    sharpness: jnp.ndarray,
    phi_W1: jnp.ndarray, phi_b1: jnp.ndarray,
    phi_W2: jnp.ndarray, phi_b2: float,
    gate_global: float,
    alpha_base: float, alpha_max: float,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> Tuple[jnp.ndarray, ...]:
    mu = phi_forward(grad, sharpness, phi_W1, phi_b1, phi_W2, phi_b2)
    a_per_coord = jnp.clip(alpha_base * (1.0 + mu), 0.0, alpha_max)
    smart = grad + gate_global * a_per_coord * mu

    m = beta1 * exp_avg + (1.0 - beta1) * smart
    v = beta2 * exp_avg_sq + (1.0 - beta2) * smart * smart
    update = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v, mu
