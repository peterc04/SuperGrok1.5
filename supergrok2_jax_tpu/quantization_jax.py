"""
Quantization Utilities for SuperGrok v2 (JAX)

Mirrors the quantization formats from the PyTorch implementation
(grokking_optimizers/quantization.py):
  - INT8 symmetric per-tensor quantization
  - INT4 GPTQ-style packing (future)
  - MXFP4 microscaling FP4 (future)

On TPU, these are primarily useful for reducing memory footprint of
expert weights. XLA handles the dequantization fusion.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class Int8QuantizedTensor(NamedTuple):
    """INT8 symmetric quantized tensor."""
    data: jnp.ndarray    # int8 quantized values
    scale: jnp.ndarray   # float32 scale factor


def quantize_int8_symmetric(tensor: jnp.ndarray) -> Int8QuantizedTensor:
    """Quantize a float32 tensor to INT8 symmetric.

    scale = max(|tensor|) / 127
    quantized = round(tensor / scale)

    Args:
        tensor: float32 tensor to quantize

    Returns:
        Int8QuantizedTensor with data and scale
    """
    abs_max = jnp.max(jnp.abs(tensor))
    scale = abs_max / 127.0
    scale = jnp.maximum(scale, 1e-8)  # avoid division by zero
    quantized = jnp.round(tensor / scale).astype(jnp.int8)
    return Int8QuantizedTensor(data=quantized, scale=scale)


def dequantize_int8(qt: Int8QuantizedTensor) -> jnp.ndarray:
    """Dequantize INT8 tensor back to float32.

    Args:
        qt: Int8QuantizedTensor

    Returns:
        float32 tensor
    """
    return qt.data.astype(jnp.float32) * qt.scale


def quantize_expert_weights_int8(
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray,
) -> dict:
    """Quantize expert MLP weights to INT8.

    Args:
        W1: [num_experts, expert_hidden, 1]
        b1: [num_experts, expert_hidden]
        W2: [num_experts, 1, expert_hidden]
        b2: [num_experts, 1]

    Returns:
        dict with quantized weights and scales
    """
    return {
        'W1': quantize_int8_symmetric(W1),
        'b1': quantize_int8_symmetric(b1),
        'W2': quantize_int8_symmetric(W2),
        'b2': quantize_int8_symmetric(b2),
    }
