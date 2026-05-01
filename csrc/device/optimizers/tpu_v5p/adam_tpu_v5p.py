"""Adam (multi-tensor) device-function template for TPU v5p (128-wide MXU).

On TPU, multi-tensor apply is handled natively by JAX's vmap/pmap
transformations. These templates provide the per-parameter step logic.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def grokadamw_step(param, exp_avg, exp_avg_sq, ema, grad,
                   alpha, lamb, beta1, beta2, lr, weight_decay, eps, bc1, bc2):
    """GrokAdamW step as pure JAX (XLA-compiled for TPU MXU)."""
    filtered = alpha * ema + (1.0 - alpha) * grad
    amplified = grad + lamb * filtered
    ea = beta1 * exp_avg + (1.0 - beta1) * amplified
    easq = beta2 * exp_avg_sq + (1.0 - beta2) * amplified ** 2
    step_size = lr / bc1
    denom = jnp.sqrt(easq / bc2) + eps
    param = param * (1.0 - lr * weight_decay) - step_size * ea / denom
    return param, ea, easq, filtered
