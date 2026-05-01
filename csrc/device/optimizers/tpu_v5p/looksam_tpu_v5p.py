"""LookSAM device-function template for TPU v5p (128-wide MXU).

LookSAM operations as pure JAX ops (XLA-compiled for TPU).
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import TILE_SIZE


def looksam_direction(sam_grad, normal_grad, inv_norm):
    """v_dir = (sam_grad - normal_grad) * inv_norm"""
    return (sam_grad - normal_grad) * inv_norm


def looksam_adjust(grad, v_dir, la_times_gnorm):
    """grad += la_times_gnorm * v_dir"""
    return grad + la_times_gnorm * v_dir


def looksam_perturb(param, grad, rho_over_norm):
    """param += rho_over_norm * grad"""
    return param + rho_over_norm * grad


def looksam_restore(backup):
    """Restore param from backup."""
    return backup
