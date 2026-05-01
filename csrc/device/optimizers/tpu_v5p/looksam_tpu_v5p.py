"""TODO: Device-function template for looksam on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def looksam_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("looksam TPU v5p device template")
