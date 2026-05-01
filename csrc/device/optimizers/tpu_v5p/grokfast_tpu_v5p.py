"""TODO: Device-function template for grokfast on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def grokfast_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("grokfast TPU v5p device template")
