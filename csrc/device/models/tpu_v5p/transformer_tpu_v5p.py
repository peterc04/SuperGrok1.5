"""TODO: Fused forward+backward device template for transformer on TPU v5p."""

import jax
import jax.numpy as jnp


def transformer_forward(params, inputs):
    """TODO: Implement fused forward pass."""
    raise NotImplementedError("transformer_forward TPU v5p")


def transformer_backward(params, grad_output):
    """TODO: Implement fused backward pass."""
    raise NotImplementedError("transformer_backward TPU v5p")
