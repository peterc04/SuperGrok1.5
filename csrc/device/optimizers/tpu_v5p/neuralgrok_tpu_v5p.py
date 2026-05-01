"""TODO: Device-function template for neuralgrok on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def neuralgrok_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("neuralgrok TPU v5p device template")
