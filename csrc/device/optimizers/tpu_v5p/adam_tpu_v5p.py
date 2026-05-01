"""TODO: Device-function template for adam on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def adam_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("adam TPU v5p device template")
