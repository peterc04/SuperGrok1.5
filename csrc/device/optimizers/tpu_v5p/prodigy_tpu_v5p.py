"""Prodigy device-function template for TPU v5p (128-wide MXU).

Prodigy's distance-aware Adam variant as pure JAX ops.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def prodigy_step(param, exp_avg, exp_avg_sq, s, grad,
                 d_lr, beta1, beta2, lr, weight_decay, eps, bc1, bc2):
    """Prodigy per-element step.

    s = beta2 * s + (1-beta2) * d_lr^2 * grad^2
    exp_avg = beta1 * exp_avg + (1-beta1) * d_lr * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * d_lr^2 * grad^2
    param *= (1 - lr * d_lr * weight_decay)
    param -= lr * exp_avg / (bc1 * (sqrt(exp_avg_sq/bc2) + d_lr*eps))
    """
    d_lr_g = d_lr * grad
    d_lr_g_sq = d_lr * d_lr * grad * grad
    s = beta2 * s + (1.0 - beta2) * d_lr_g_sq
    ea = beta1 * exp_avg + (1.0 - beta1) * d_lr_g
    easq = beta2 * exp_avg_sq + (1.0 - beta2) * d_lr_g_sq
    denom = jnp.sqrt(easq / bc2) + d_lr * eps
    param = param * (1.0 - lr * d_lr * weight_decay) - lr * ea / (bc1 * denom)
    return param, ea, easq, s
