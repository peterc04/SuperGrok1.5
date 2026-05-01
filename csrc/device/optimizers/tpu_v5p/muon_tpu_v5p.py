"""TODO: Device-function template for muon on TPU v5p (128-wide MXU)."""

import jax
import jax.numpy as jnp


def muon_step(params, grads, state, lr):
    """TODO: Port from Pallas kernel implementation."""
    raise NotImplementedError("muon TPU v5p device template")
