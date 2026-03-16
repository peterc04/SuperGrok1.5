"""
Bilevel Meta-Net Optimization for SuperGrok v2 (JAX)

JAX advantage: jax.grad automatically differentiates through
lax.associative_scan, the GRU, and PEER routing. No manual backward
implementation needed (vs 1000+ lines of custom CUDA backward kernels).

Mathematical formulation (identical to PyTorch):
  meta_loss = -sum_i sum_n (smart_grad_i[n] * val_grad_unit_i[n])
  where:
    smart_grad_i = meta_net_forward(train_grad_i, sharpness_i, ...)
    val_grad_unit_i = val_grad_i / ||val_grad_i||

  meta_grads = d(meta_loss) / d(meta_weights)
  meta_weights -= meta_lr * meta_grads
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple, Optional

from .mamba3_peer_metanet_jax import MetaNetWeights, MetaNetConfig, meta_net_forward
from .supergrok2_jax import SuperGrok2State


def bilevel_step(
    params: Any,
    train_grads: Any,
    val_x: jnp.ndarray,
    val_y: jnp.ndarray,
    opt_state: SuperGrok2State,
    meta_weights: MetaNetWeights,
    config: Any,
    meta_config: MetaNetConfig,
    model_fn: Callable,
    loss_fn: Callable,
    meta_lr: float = 1e-4,
) -> Tuple[MetaNetWeights, float]:
    """Bilevel meta-net training step using jax.grad.

    Differentiates through the meta-net forward pass (including
    lax.associative_scan) to update meta_weights.

    Key advantage over PyTorch: jax.grad handles all backward computation
    automatically. No custom backward kernels needed.

    Args:
        params: model parameter pytree
        train_grads: training gradients pytree (same structure as params)
        val_x: validation inputs
        val_y: validation targets
        opt_state: current optimizer state
        meta_weights: current meta-net weights (to be updated)
        config: OptimizerConfig
        meta_config: MetaNetConfig
        model_fn: function(params, x) -> logits
        loss_fn: function(logits, y) -> scalar loss
        meta_lr: meta-learning rate

    Returns:
        new_meta_weights: updated meta-net weights
        meta_loss_val: scalar meta-loss value
    """
    params_flat, _ = jax.tree.flatten(params)
    grads_flat, _ = jax.tree.flatten(train_grads)

    # Compute validation gradients (detached from meta-net graph)
    val_loss = loss_fn(model_fn(params, val_x), val_y)
    val_grads = jax.grad(lambda p: loss_fn(model_fn(p, val_x), val_y))(params)
    val_grads_flat, _ = jax.tree.flatten(val_grads)

    def meta_loss_fn(mw):
        """Meta-loss as a function of meta_weights.

        This is the function we differentiate to get meta_grads.
        jax.grad will automatically backpropagate through:
          - meta_net_forward (including lax.associative_scan)
          - The GRU cell
          - The PEER routing (soft routing for differentiability)
          - The expert MLP
        """
        total_loss = jnp.array(0.0)

        for i, (g, vg) in enumerate(zip(grads_flat, val_grads_flat)):
            ps = opt_state.param_states[i]

            # Forward through meta-net with soft routing (differentiable)
            smart_grad, _, _, _, _ = meta_net_forward(
                g.reshape(-1), ps.sharpness,
                ps.gru_state, ps.mamba_fwd_state, ps.mamba_bwd_state,
                mw, meta_config, use_soft_routing=True,
            )

            # Validation gradient direction
            vg_flat = vg.reshape(-1).astype(jnp.float32)
            vg_norm = jnp.linalg.norm(vg_flat)
            vg_unit = jnp.where(vg_norm > 1e-12, vg_flat / vg_norm, vg_flat)

            # Align smart_grad with validation gradient
            total_loss = total_loss - jnp.sum(smart_grad.reshape(-1) * vg_unit)

        return total_loss

    # Compute meta-loss and meta-gradients via jax.grad
    meta_loss_val, meta_grads = jax.value_and_grad(meta_loss_fn)(meta_weights)

    # Apply meta gradient update (simple SGD)
    new_meta_weights = jax.tree.map(
        lambda w, g: w - meta_lr * g,
        meta_weights, meta_grads,
    )

    return new_meta_weights, meta_loss_val
