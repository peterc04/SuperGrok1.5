"""TODO: Device-function template for lion on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def lion_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("lion TPU v5p device template")
