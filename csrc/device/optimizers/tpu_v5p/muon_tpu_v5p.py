"""Muon device-function template for TPU v5p (128-wide MXU).

Muon's Newton-Schulz orthogonalization uses matrix multiplications
which map directly to JAX/XLA matmul (leveraging MXU). The element-wise
kernels (momentum_normalize, ns_combine, update) are expressed as pure
JAX ops which XLA compiles to efficient TPU code.

The Pallas runtime path (csrc/kernels/tpu/_pallas_kernels.py) is used
only for the Mamba scan operations; Muon does not need custom Pallas kernels.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def muon_momentum_normalize(buf, grad, momentum, inv_norm):
    """buf = momentum * buf + grad; X = buf * inv_norm"""
    buf = momentum * buf + grad
    X = buf * inv_norm
    return buf, X


def muon_ns_combine(X, AX, AAX, a, b, c):
    """X_out = a*X + b*AX + c*AAX"""
    return a * X + b * AX + c * AAX


def muon_update(param, orth, neg_lr_scale, decay_factor):
    """p = (p + neg_lr_scale * orth) * decay_factor"""
    return (param + neg_lr_scale * orth) * decay_factor


def muon_ns_combine_update_fused(param, X, AX, AAX, a, b, c, neg_lr_scale, decay_factor):
    """Fused ns_combine + update: eliminates intermediate write."""
    orth = a * X + b * AX + c * AAX
    return (param + neg_lr_scale * orth) * decay_factor
