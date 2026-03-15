"""
Meta-Net Optimizers — JAX/TPU Implementation

Pure-functional JAX implementations of optimizers that use a learned meta-net
to transform gradients: SuperGrok v1.5, SuperGrok v1.1, NeuralGrok.

Each optimizer follows the pattern:
    (params, grads, state, weights, config) → (new_params, new_state)

No in-place mutation. Compatible with jax.jit, jax.grad, jax.vmap.

Meta-net architectures:
    - SuperGrok v1.5/v1.1: 2-layer MLP (grad → smart_grad) + EMA mu + Adam
    - NeuralGrok: Amplifier MLP (|grad| → scale) → grad * scale + Adam
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


# ═════════════════════════════════════════════════════════════════════════
#  SuperGrok v1.5
# ═════════════════════════════════════════════════════════════════════════

class SuperGrok15State(NamedTuple):
    """Per-parameter state for SuperGrok v1.5."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    mu: jnp.ndarray           # gradient EMA
    sharpness: jnp.ndarray    # sharpness estimate (updated externally)
    step: jnp.ndarray         # scalar int32


class SuperGrok15Weights(NamedTuple):
    """Meta-net weights for SuperGrok v1.5 (2-layer MLP)."""
    W1: jnp.ndarray   # [hidden_dim, 1] (input is scalar per-element)
    b1: jnp.ndarray   # [hidden_dim]
    W2: jnp.ndarray   # [1, hidden_dim]
    b2: jnp.ndarray   # [1]
    rescale: float     # output rescaling factor


class SuperGrok15Config(NamedTuple):
    """SuperGrok v1.5 hyperparameters."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    alpha: float = 0.98       # mu EMA coefficient
    lamb: float = 2.0         # mu amplification
    gradient_clipping: float = 1.0
    hidden_dim: int = 32
    ramp: float = 1.0         # warmup ramp (pre-computed)
    gate_signal: float = 1.0  # gating signal (pre-computed)


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
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray,
    rescale: float,
) -> jnp.ndarray:
    """Meta-net MLP: smart_grad = grad + rescale * W2 @ relu(W1 @ (rescale*grad) + b1) + b2.

    Applied element-wise (each element is a separate 1D input).

    Args:
        g: [N] gradient vector
        W1: [hidden_dim, 1]
        b1: [hidden_dim]
        W2: [1, hidden_dim]
        b2: [1]
        rescale: scalar

    Returns:
        smart_grad: [N]
    """
    # Per-element MLP: input is scalar grad, output is scalar modification
    scaled = g * rescale                            # [N]
    hidden = jax.nn.relu(scaled[:, None] * W1.T + b1)  # [N, hidden_dim]
    mlp_out = (hidden @ W2.T + b2).squeeze(-1)     # [N]
    return g + rescale * mlp_out


def supergrok15_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: SuperGrok15State,
    weights: SuperGrok15Weights,
    config: SuperGrok15Config,
) -> Tuple[jnp.ndarray, SuperGrok15State]:
    """One SuperGrok v1.5 step.

    Math: meta-net MLP → mu EMA amplification → Adam.
    """
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    # Gradient clipping
    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    # Meta-net forward
    smart_g = _meta_mlp_forward(g, weights.W1, weights.b1, weights.W2, weights.b2, weights.rescale)

    # Mu EMA
    new_mu = config.alpha * state.mu + (1.0 - config.alpha) * g
    lamb_eff = config.lamb * config.ramp * config.gate_signal
    effective = smart_g + lamb_eff * new_mu

    # Adam
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


# ═════════════════════════════════════════════════════════════════════════
#  SuperGrok v1.1
# ═════════════════════════════════════════════════════════════════════════

class SuperGrok11State(NamedTuple):
    """Per-parameter state for SuperGrok v1.1."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    mu: jnp.ndarray
    sharpness: jnp.ndarray
    step: jnp.ndarray


class SuperGrok11Weights(NamedTuple):
    """Meta-net weights for SuperGrok v1.1 (same architecture as v1.5)."""
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    rescale: float


class SuperGrok11Config(NamedTuple):
    """SuperGrok v1.1 hyperparameters."""
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
    gate_temperature: float = 1.0  # v1.1 uses cosine gate (temperature-based)


def init_supergrok11_state(param: jnp.ndarray) -> SuperGrok11State:
    flat = param.reshape(-1)
    return SuperGrok11State(
        exp_avg=jnp.zeros_like(flat),
        exp_avg_sq=jnp.zeros_like(flat),
        mu=jnp.zeros_like(flat),
        sharpness=jnp.zeros_like(flat),
        step=jnp.array(0, dtype=jnp.int32),
    )


def supergrok11_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: SuperGrok11State,
    weights: SuperGrok11Weights,
    config: SuperGrok11Config,
) -> Tuple[jnp.ndarray, SuperGrok11State]:
    """One SuperGrok v1.1 step.

    Math: same as v1.5 but with cosine-gate temperature instead of sigmoid gate.
    """
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    # Gradient clipping
    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    # Meta-net forward (identical architecture to v1.5)
    smart_g = _meta_mlp_forward(g, weights.W1, weights.b1, weights.W2, weights.b2, weights.rescale)

    # Mu EMA with cosine gate (temperature replaces sigmoid gate_signal)
    new_mu = config.alpha * state.mu + (1.0 - config.alpha) * g
    lamb_eff = config.lamb * config.ramp * config.gate_temperature
    effective = smart_g + lamb_eff * new_mu

    # Adam
    new_ea = config.beta1 * state.exp_avg + (1.0 - config.beta1) * effective
    new_easq = config.beta2 * state.exp_avg_sq + (1.0 - config.beta2) * effective ** 2

    bc1 = 1.0 - config.beta1 ** step
    bc2 = 1.0 - config.beta2 ** step
    step_size = config.lr / bc1
    denom = jnp.sqrt(new_easq / bc2) + config.eps

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - step_size * new_ea / denom

    new_state = SuperGrok11State(
        exp_avg=new_ea, exp_avg_sq=new_easq, mu=new_mu,
        sharpness=state.sharpness, step=step)
    return new_p.reshape(param.shape), new_state


# ═════════════════════════════════════════════════════════════════════════
#  NeuralGrok
# ═════════════════════════════════════════════════════════════════════════

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
    alpha: float          # scale coefficient
    beta: float           # scale offset


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
    """One NeuralGrok step.

    Math: scale = alpha * MLP(|grad|) + beta, smart_grad = grad * scale, then Adam.
    """
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    # Gradient clipping
    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    # Amplifier MLP: input is |grad|, output is per-element scale
    abs_g = jnp.abs(g)[:, None]                       # [N, 1]
    hidden = jax.nn.relu(abs_g @ weights.W1.T + weights.b1)  # [N, hidden_dim]
    mlp_out = (hidden @ weights.W_last.T + weights.b_last).squeeze(-1)  # [N]
    scale = weights.alpha * mlp_out + weights.beta
    smart_g = g * scale

    # Adam
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
