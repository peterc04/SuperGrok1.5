"""Grokfast device-function template for TPU v5p (128-wide MXU).

EMA gradient accumulation + amplification as pure JAX ops.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def grokfast_ema_step(grad, ema, alpha, lamb):
    """EMA update + gradient amplification.

    ema = alpha * ema + (1 - alpha) * grad
    grad_out = grad + lamb * ema
    """
    ema = alpha * ema + (1.0 - alpha) * grad
    grad_out = grad + lamb * ema
    return grad_out, ema
