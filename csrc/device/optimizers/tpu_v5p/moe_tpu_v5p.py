"""TODO: Device-function template for moe on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def moe_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("moe TPU v5p device template")
