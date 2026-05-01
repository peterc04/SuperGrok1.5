"""SuperGrok v1.1 device-function template for TPU v5p (128-wide MXU).

SG11 operations as pure JAX ops. The meta-net MLP(2->H->1) uses
JAX's vectorized operations which compile to efficient XLA HLO.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def sg11_mu_metanet(mu, grad, sharp, alpha, W1, b1, W2, b2, rescale):
    """EMA update + meta-net inference.

    mu = alpha * mu + (1-alpha) * grad
    MLP: Linear(2,H) -> GELU -> Linear(H,1)
    smart_grad = grad + rescale * mlp_out
    """
    mu = alpha * mu + (1.0 - alpha) * grad
    # MLP inference: W1 is [H, 2], b1 is [H], W2 is [H], b2 is scalar
    inputs = jnp.stack([grad, sharp], axis=-1)  # [..., 2]
    hidden = jax.nn.gelu(jnp.einsum('...i,hi->...h', inputs, W1) + b1)
    mlp_out = jnp.sum(W2 * hidden, axis=-1) + b2
    smart_grad = grad + rescale * mlp_out
    return mu, smart_grad


def sg11_adam_cosine_gate(param, exp_avg, exp_avg_sq, smart_grad,
                          lamb_eff, beta1, beta2, lr, weight_decay, eps, bc1, bc2):
    """Adam with cosine-gated gradient (lamb_eff precomputed)."""
    ea = beta1 * exp_avg + (1.0 - beta1) * lamb_eff * smart_grad
    easq = beta2 * exp_avg_sq + (1.0 - beta2) * (lamb_eff * smart_grad) ** 2
    step_size = lr / bc1
    denom = jnp.sqrt(easq / bc2) + eps
    param = param * (1.0 - lr * weight_decay) - step_size * ea / denom
    return param, ea, easq
