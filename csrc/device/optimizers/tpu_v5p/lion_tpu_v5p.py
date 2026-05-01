"""Lion device-function template for TPU v5p (128-wide MXU).

Lion's sign-based update is expressed as pure JAX ops which XLA compiles
to efficient TPU code. No custom Pallas kernels needed.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def lion_step(param, exp_avg, grad, lr, beta1, beta2, wd):
    """Lion optimizer step.

    update  = sign(beta1 * exp_avg + (1 - beta1) * grad)
    param  -= lr * (update + wd * param)
    exp_avg = beta2 * exp_avg + (1 - beta2) * grad
    """
    interp = beta1 * exp_avg + (1.0 - beta1) * grad
    update = jnp.sign(interp)
    param = param - lr * (update + wd * param)
    exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    return param, exp_avg
