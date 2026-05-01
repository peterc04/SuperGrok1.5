"""TODO: Device-function template for supergrok15 on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def supergrok15_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("supergrok15 TPU v5p device template")
