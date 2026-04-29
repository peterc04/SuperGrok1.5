"""
Simple Optimizers — JAX/TPU Implementation

Pure-functional JAX implementations of all simple (non-meta-net) optimizers.
Each optimizer follows the pattern:
    (params, grads, state, config) → (new_params, new_state)

No in-place mutation. Compatible with jax.jit, jax.grad, jax.vmap.

Optimizers:
    - GrokAdamW: Adam with EMA-amplified gradients
    - Lion: Sign-based optimizer (Chen et al. 2023)
    - Grokfast: EMA gradient amplification (pre-processing)
    - Prodigy: Distance-based adaptive learning rate
    - Muon: Newton-Schulz orthogonalized momentum
    - LookSAM: Sharpness-Aware Minimization with lookahead
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple, Any


# ═════════════════════════════════════════════════════════════════════════
#  GrokAdamW
# ═════════════════════════════════════════════════════════════════════════

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
    """One GrokAdamW step. Math: EMA amplification + Adam."""
    step = state.step + 1
    g = grad.reshape(-1).astype(jnp.float32)

    # Gradient clipping
    gnorm = jnp.linalg.norm(g)
    g = jnp.where(gnorm > config.gradient_clipping,
                  g * (config.gradient_clipping / (gnorm + 1e-12)), g)

    # EMA amplification
    new_ema = config.alpha * state.ema + (1.0 - config.alpha) * g
    effective = g + config.lamb * new_ema

    # Adam
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


# ═════════════════════════════════════════════════════════════════════════
#  Lion
# ═════════════════════════════════════════════════════════════════════════

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
    """One Lion step. Math: sign(beta1*m + (1-beta1)*g)."""
    g = grad.reshape(-1).astype(jnp.float32)
    m = state.exp_avg

    # Update = sign(interpolation)
    update = jnp.sign(config.beta1 * m + (1.0 - config.beta1) * g)

    p_flat = param.reshape(-1)
    new_p = p_flat * (1.0 - config.lr * config.weight_decay) - config.lr * update

    # Momentum update (for next step)
    new_m = config.beta2 * m + (1.0 - config.beta2) * g

    return new_p.reshape(param.shape), LionState(exp_avg=new_m)


# ═════════════════════════════════════════════════════════════════════════
#  Grokfast
# ═════════════════════════════════════════════════════════════════════════

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
    """Grokfast EMA amplification (pre-processing step).

    Returns amplified gradient and updated state.
    Use with any base optimizer (e.g., AdamW).
    """
    g = grad.reshape(-1).astype(jnp.float32)
    new_ema = config.alpha * state.ema + (1.0 - config.alpha) * g
    amplified = g + config.lamb * new_ema
    return amplified.reshape(grad.shape), GrokfastState(ema=new_ema)


# ═════════════════════════════════════════════════════════════════════════
#  Prodigy
# ═════════════════════════════════════════════════════════════════════════

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

    Math: distance-based adaptive LR estimation + Adam.
    """
    step = state.step + 1
    p_flat = param.reshape(-1).astype(jnp.float32)
    g = grad.reshape(-1).astype(jnp.float32)

    # Distance-based LR estimation
    num = jnp.sum(g * (p_flat - state.param_init))
    den = jnp.sum(state.s_buf * jnp.abs(g))

    new_d_lr = jnp.where(den > 0, jnp.maximum(d_lr, num / den), d_lr)
    new_d_lr_float = float(new_d_lr)

    # s_buf update
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
    return new_p.reshape(param.shape), new_state, new_d_lr_float


# ═════════════════════════════════════════════════════════════════════════
#  Muon
# ═════════════════════════════════════════════════════════════════════════

class MuonState(NamedTuple):
    """Per-parameter state for Muon."""
    momentum_buf: jnp.ndarray


class MuonConfig(NamedTuple):
    """Muon hyperparameters."""
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    ns_steps: int = 5


def init_muon_state(param: jnp.ndarray) -> MuonState:
    return MuonState(momentum_buf=jnp.zeros_like(param))


def _newton_schulz_ortho(X: jnp.ndarray, ns_steps: int) -> jnp.ndarray:
    """Newton-Schulz orthogonalization.

    Muon paper coefficients: a=3.4445, b=-4.7750, c=2.0315.
    Iterates: X <- a*X + b*(X @ X.T) @ X + c*X @ (X.T @ X)
    Converges to nearest orthogonal matrix.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    def ns_body(_, X):
        A = X @ X.T
        return X * a + (A @ X) * b + (X @ (X.T @ X)) * c

    return lax.fori_loop(0, ns_steps, ns_body, X)


def muon_step(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    state: MuonState,
    config: MuonConfig,
) -> Tuple[jnp.ndarray, MuonState]:
    """One Muon step. Math: Newton-Schulz orthogonalized momentum."""
    g = grad.astype(jnp.float32)
    new_m = config.momentum * state.momentum_buf + g

    # Newton-Schulz orthogonalization for 2D+ params
    if param.ndim >= 2:
        X = new_m
        norm_val = jnp.linalg.norm(X)
        X_normed = jnp.where(norm_val > 0, X / norm_val, X)
        X_ortho = _newton_schulz_ortho(X_normed, config.ns_steps)
        update = X_ortho * norm_val
    else:
        update = new_m

    new_p = param * (1.0 - config.lr * config.weight_decay) - config.lr * update
    return new_p, MuonState(momentum_buf=new_m)


# ═════════════════════════════════════════════════════════════════════════
#  LookSAM
# ═════════════════════════════════════════════════════════════════════════

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
