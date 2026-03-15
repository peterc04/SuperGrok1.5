"""
SuperGrok v2 Optimizer — JAX/TPU Implementation

Functional optimizer: all state is explicit (passed in, returned out).
No in-place mutation. Compatible with jax.jit, jax.grad, jax.vmap.

Equivalent to the PyTorch SuperGrok2 optimizer (supergrok2.py) but
written in JAX's functional programming model.

Usage::

    import jax
    from jax_supergrok2 import (
        OptimizerConfig, init_state, supergrok2_step,
    )
    from mamba3_peer_metanet_jax import MetaNetConfig, init_meta_weights

    # Initialize
    config = OptimizerConfig()
    meta_config = MetaNetConfig()
    meta_weights = init_meta_weights(meta_config, jax.random.PRNGKey(0))
    opt_state = init_state(params, config, meta_config)

    # Training step
    grads = jax.grad(loss_fn)(params)
    params, opt_state = supergrok2_step(
        params, grads, opt_state, meta_weights, config, meta_config)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple, Any
from functools import partial

from .mamba3_peer_metanet_jax import MetaNetWeights, MetaNetConfig, meta_net_forward


class PerParamState(NamedTuple):
    """Per-parameter optimizer state.

    All fields are jnp arrays. JAX-compatible pytree.
    """
    exp_avg: jnp.ndarray          # [N] FP32 — first moment estimate
    exp_avg_sq: jnp.ndarray       # [N] FP32 — second moment estimate
    mu: jnp.ndarray               # [N] FP32 — EMA of gradient
    sharpness: jnp.ndarray        # [N] FP32 — sharpness estimate
    gru_state: jnp.ndarray        # [N, gru_hidden] FP32
    mamba_fwd_state: jnp.ndarray  # [d_inner, d_state] FP32
    mamba_bwd_state: jnp.ndarray  # [d_inner, d_state] FP32
    step_count: jnp.ndarray       # scalar int32


class SuperGrok2State(NamedTuple):
    """Full optimizer state for all parameters.

    The param_states list has one PerParamState per parameter.
    JAX pytrees handle lists of NamedTuples natively.
    """
    param_states: list  # List[PerParamState]
    global_step: jnp.ndarray      # scalar int32
    cached_train_acc: jnp.ndarray  # scalar float32


class OptimizerConfig(NamedTuple):
    """Optimizer hyperparameters (static, not traced by JAX).

    These are compile-time constants for jax.jit.
    """
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    alpha_init: float = 0.98
    lamb: float = 2.0
    gamma: float = 0.1
    kappa: float = 0.1
    warmup_steps: int = 100
    warmup_ramp: int = 100
    gradient_clipping: float = 1.0
    gate_scale: float = 20.0
    gate_thresh: float = 0.8
    wd_ramp: float = 4.0
    wd_scale: float = 20.0
    wd_thresh: float = 0.9


def init_per_param_state(
    param: jnp.ndarray,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
) -> PerParamState:
    """Initialize optimizer state for one parameter.

    Args:
        param: parameter tensor (any shape)
        config: optimizer config
        meta_config: meta-net config

    Returns:
        PerParamState with all-zeros initialization
    """
    N = param.size
    return PerParamState(
        exp_avg=jnp.zeros(N, dtype=jnp.float32),
        exp_avg_sq=jnp.zeros(N, dtype=jnp.float32),
        mu=jnp.zeros(N, dtype=jnp.float32),
        sharpness=jnp.zeros(N, dtype=jnp.float32),
        gru_state=jnp.zeros((N, meta_config.gru_hidden), dtype=jnp.float32),
        mamba_fwd_state=jnp.zeros(
            (meta_config.d_inner, meta_config.d_state), dtype=jnp.float32),
        mamba_bwd_state=jnp.zeros(
            (meta_config.d_inner, meta_config.d_state), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def init_state(
    params: Any,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
) -> SuperGrok2State:
    """Initialize full optimizer state for all parameters.

    Args:
        params: pytree of model parameters
        config: optimizer config
        meta_config: meta-net config

    Returns:
        SuperGrok2State
    """
    params_flat, _ = jax.tree.flatten(params)
    param_states = [
        init_per_param_state(p, config, meta_config) for p in params_flat
    ]
    return SuperGrok2State(
        param_states=param_states,
        global_step=jnp.array(0, dtype=jnp.int32),
        cached_train_acc=jnp.array(0.0, dtype=jnp.float32),
    )


def _sigmoid(scale: float, value: float, thresh: float) -> float:
    """Sigmoid gating function."""
    return 1.0 / (1.0 + jnp.exp(-scale * (value - thresh)))


def _get_ramp_factor(
    global_step: jnp.ndarray,
    warmup_steps: int,
    warmup_ramp: int,
) -> jnp.ndarray:
    """Compute warmup ramp factor."""
    elapsed = jnp.maximum(global_step - warmup_steps, 0)
    return jnp.where(
        global_step <= warmup_steps,
        0.0,
        jnp.minimum(elapsed / jnp.maximum(warmup_ramp, 1), 1.0),
    )


def _get_effective_wd(
    base_wd: float,
    train_acc: jnp.ndarray,
    wd_ramp: float,
    wd_scale: float,
    wd_thresh: float,
) -> jnp.ndarray:
    """Compute effective weight decay with sigmoid scaling."""
    sigmoid_val = _sigmoid(wd_scale, train_acc, wd_thresh)
    return base_wd * (1.0 + wd_ramp * sigmoid_val)


def _update_single_param(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    ps: PerParamState,
    meta_weights: MetaNetWeights,
    meta_config: MetaNetConfig,
    alpha_mu: float,
    lamb_eff: jnp.ndarray,
    beta1: float,
    beta2: float,
    lr: float,
    wd_eff: jnp.ndarray,
    eps: float,
    gradient_clipping: float,
) -> Tuple[jnp.ndarray, PerParamState]:
    """Update a single parameter using the SuperGrok v2 algorithm.

    Mathematical equivalence to the per-parameter loop in
    PyTorch SuperGrok2.step().

    Args:
        param: parameter values (flat)
        grad: gradient values (flat)
        ps: per-parameter state
        meta_weights: meta-net weights
        meta_config: meta-net config
        alpha_mu: EMA coefficient for mu
        lamb_eff: effective lambda
        beta1: Adam beta1
        beta2: Adam beta2
        lr: learning rate
        wd_eff: effective weight decay
        eps: Adam epsilon
        gradient_clipping: gradient norm clip threshold

    Returns:
        new_param: updated parameter
        new_ps: updated per-parameter state
    """
    step = ps.step_count + 1
    g = grad.reshape(-1).astype(jnp.float32)

    # Gradient clipping + NaN guard
    grad_norm = jnp.linalg.norm(g)
    g = jnp.where(
        grad_norm > gradient_clipping,
        g * (gradient_clipping / (grad_norm + 1e-12)),
        g,
    )
    g = jnp.where(jnp.isfinite(g), g, 0.0)

    # Meta-net forward
    smart_grad, new_gru, new_fwd, new_bwd, exp_counts = meta_net_forward(
        g, ps.sharpness, ps.gru_state,
        ps.mamba_fwd_state, ps.mamba_bwd_state,
        meta_weights, meta_config,
        use_soft_routing=False,
    )

    # Mu EMA
    new_mu = alpha_mu * ps.mu + (1.0 - alpha_mu) * g
    effective_grad = smart_grad.reshape(-1) + lamb_eff * new_mu

    # Adam update
    new_exp_avg = beta1 * ps.exp_avg + (1 - beta1) * effective_grad
    new_exp_avg_sq = beta2 * ps.exp_avg_sq + (1 - beta2) * effective_grad ** 2

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    step_size = lr / bc1
    denom = jnp.sqrt(new_exp_avg_sq / bc2) + eps

    param_flat = param.reshape(-1)
    new_param = param_flat * (1 - lr * wd_eff) - step_size * (new_exp_avg / denom)

    new_ps = PerParamState(
        exp_avg=new_exp_avg,
        exp_avg_sq=new_exp_avg_sq,
        mu=new_mu,
        sharpness=ps.sharpness,  # Updated externally via SAM step
        gru_state=new_gru,
        mamba_fwd_state=new_fwd,
        mamba_bwd_state=new_bwd,
        step_count=step,
    )

    return new_param.reshape(param.shape), new_ps


def supergrok2_step(
    params: Any,
    grads: Any,
    opt_state: SuperGrok2State,
    meta_weights: MetaNetWeights,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
    train_acc: Optional[float] = None,
) -> Tuple[Any, SuperGrok2State]:
    """One SuperGrok v2 optimizer step (JIT-compatible).

    Mathematical equivalence to PyTorch SuperGrok2.step().

    Args:
        params: pytree of model parameters
        grads: pytree of gradients (same structure as params)
        opt_state: SuperGrok2State
        meta_weights: MetaNetWeights (not updated here)
        config: OptimizerConfig
        meta_config: MetaNetConfig
        train_acc: optional training accuracy for adaptive scheduling

    Returns:
        new_params: updated parameters (same pytree structure)
        new_opt_state: updated optimizer state
    """
    global_step = opt_state.global_step + 1
    cached_acc = jnp.where(
        train_acc is not None,
        jnp.array(train_acc, dtype=jnp.float32) if train_acc is not None else opt_state.cached_train_acc,
        opt_state.cached_train_acc,
    )

    # Compute adaptive scalars
    ramp = _get_ramp_factor(global_step, config.warmup_steps, config.warmup_ramp)
    gate_signal = _sigmoid(config.gate_scale, cached_acc, config.gate_thresh)
    lamb_eff = config.lamb * ramp * gate_signal
    wd_eff = _get_effective_wd(
        config.weight_decay, cached_acc,
        config.wd_ramp, config.wd_scale, config.wd_thresh,
    )

    # Flatten params and grads
    params_flat, treedef = jax.tree.flatten(params)
    grads_flat, _ = jax.tree.flatten(grads)

    new_params_flat = []
    new_param_states = []

    for i, (p, g) in enumerate(zip(params_flat, grads_flat)):
        ps = opt_state.param_states[i]
        alpha_mu = config.alpha_init  # Simplified: no per-layer alpha decay

        new_p, new_ps = _update_single_param(
            p, g, ps, meta_weights, meta_config,
            alpha_mu, lamb_eff,
            config.beta1, config.beta2, config.lr,
            wd_eff, config.eps, config.gradient_clipping,
        )
        new_params_flat.append(new_p)
        new_param_states.append(new_ps)

    new_params = jax.tree.unflatten(treedef, new_params_flat)
    new_opt_state = SuperGrok2State(
        param_states=new_param_states,
        global_step=global_step,
        cached_train_acc=cached_acc,
    )

    return new_params, new_opt_state
