"""TODO: Fused forward+backward device template for mamba on TPU v5p."""

import jax
import jax.numpy as jnp


def mamba_forward(params, inputs):
    """TODO: Implement fused forward pass."""
    raise NotImplementedError("mamba_forward TPU v5p")


def mamba_backward(params, grad_output):
    """TODO: Implement fused backward pass."""
    raise NotImplementedError("mamba_backward TPU v5p")
