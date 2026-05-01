"""TODO: Device-function template for prodigy on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def prodigy_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("prodigy TPU v5p device template")
