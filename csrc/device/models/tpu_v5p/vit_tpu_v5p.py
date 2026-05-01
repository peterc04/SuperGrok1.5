"""TODO: Fused forward+backward device template for vit on TPU v5p."""

import jax
import jax.numpy as jnp


def vit_forward(params, inputs):
    """TODO: Implement fused forward pass."""
    raise NotImplementedError("vit_forward TPU v5p")


def vit_backward(params, grad_output):
    """TODO: Implement fused backward pass."""
    raise NotImplementedError("vit_backward TPU v5p")
