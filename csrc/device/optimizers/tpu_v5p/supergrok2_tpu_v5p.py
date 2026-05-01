"""TODO: Device-function template for supergrok2 on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def supergrok2_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("supergrok2 TPU v5p device template")
