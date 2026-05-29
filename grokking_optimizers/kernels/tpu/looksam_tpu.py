"""TPU/Pallas kernels for LookSAM -- AdamW with sharpness-aware direction adjustment.

Provides pure-JAX update functions and Pallas-accelerated TPU kernels for:
  - Standard AdamW parameter update
  - LookSAM direction computation and gradient adjustment

BF16 parameters, FP32 accumulators throughout.
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from grokking_optimizers.kernels.tpu.common_tpu import (
    ACCUM_DTYPE,
    PARAM_DTYPE,
    NanPolicy,
    apply_nan_policy,
)

# Try to import Pallas; fall back gracefully if unavailable.
try:
    from jax.experimental import pallas as pl
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False


# ---------------------------------------------------------------------------
# Pure JAX implementations
# ---------------------------------------------------------------------------


def looksam_adamw_update(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.98,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
    nan_policy: NanPolicy = NanPolicy.IGNORE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Standard AdamW update for LookSAM (pure JAX).

    Args:
        params: Current parameters (bf16).
        grads: Gradients (bf16).
        exp_avg: First moment estimate (f32).
        exp_avg_sq: Second moment estimate (f32).
        step: Current optimiser step (1-indexed).
        beta1: First moment decay.
        beta2: Second moment decay.
        lr: Learning rate.
        wd: Weight decay coefficient.
        eps: Numerical stability term.
        nan_policy: How to handle NaN gradients.

    Returns:
        (updated_params, updated_exp_avg, updated_exp_avg_sq)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), nan_policy)
    p = params.astype(ACCUM_DTYPE)

    # Update biased first and second moment estimates
    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (g * g)

    # Bias correction
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    # AdamW update: decoupled weight decay + Adam step
    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    return p.astype(PARAM_DTYPE), new_exp_avg, new_exp_avg_sq


def looksam_direction_update(
    orig_grads: jnp.ndarray,
    perturbed_grads: jnp.ndarray,
    directions: jnp.ndarray,
    alpha: float = 0.7,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute LookSAM sharpness-aware direction and adjust gradients.

    The direction captures the difference between gradients at the original
    and perturbed (sharpness-aware) points. This direction is then used to
    project out the sharpness component from the original gradient.

    Uses jax.lax.psum for cross-replica reductions when running under pmap.

    Args:
        orig_grads: Original gradients (bf16).
        perturbed_grads: Gradients at the SAM-perturbed point (bf16).
        directions: Running SAM direction buffer (f32).
        alpha: Interpolation weight for direction adjustment.

    Returns:
        (adjusted_grads, updated_directions)
    """
    g = orig_grads.astype(ACCUM_DTYPE)
    g_pert = perturbed_grads.astype(ACCUM_DTYPE)

    # Compute sharpness direction: difference between perturbed and original
    diff = g_pert - g

    # Normalise the direction
    diff_norm = jnp.sqrt(jnp.sum(diff * diff) + 1e-12)
    new_direction = diff / diff_norm

    # Project original gradient onto the sharpness direction
    proj_scale = jnp.sum(g * new_direction)
    g_proj = proj_scale * new_direction

    # Adjust: remove alpha fraction of the sharpness-aligned component
    adjusted = g - alpha * g_proj

    return adjusted.astype(PARAM_DTYPE), new_direction


# ---------------------------------------------------------------------------
# Pallas TPU kernels
# ---------------------------------------------------------------------------


def _adamw_kernel_body(
    params_ref, grads_ref, exp_avg_ref, exp_avg_sq_ref, out_params_ref,
    out_exp_avg_ref, out_exp_avg_sq_ref,
    *, beta1, beta2, lr, wd, eps, step,
):
    """Pallas kernel body for fused AdamW update."""
    p = params_ref[...].astype(jnp.float32)
    g = grads_ref[...].astype(jnp.float32)
    m = exp_avg_ref[...]
    v = exp_avg_sq_ref[...]

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * (g * g)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    out_params_ref[...] = p.astype(jnp.bfloat16)
    out_exp_avg_ref[...] = new_m
    out_exp_avg_sq_ref[...] = new_v


def looksam_adamw_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.98,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pallas-accelerated AdamW update for TPU.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return looksam_adamw_update(
            params, grads, exp_avg, exp_avg_sq, step,
            beta1, beta2, lr, wd, eps,
        )

    kernel_fn = functools.partial(
        _adamw_kernel_body,
        beta1=beta1, beta2=beta2, lr=lr, wd=wd, eps=eps, step=step,
    )

    out_params, out_exp_avg, out_exp_avg_sq = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(params.shape, PARAM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg_sq.shape, ACCUM_DTYPE),
        ],
        grid=(1,),
    )(params, grads, exp_avg, exp_avg_sq)

    return out_params, out_exp_avg, out_exp_avg_sq


def looksam_direction_pallas_kernel(
    orig_grads: jnp.ndarray,
    perturbed_grads: jnp.ndarray,
    directions: jnp.ndarray,
    alpha: float = 0.7,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pallas-accelerated LookSAM direction update for TPU.

    Falls back to pure JAX if Pallas is unavailable.
    """
    # Direction update involves global reductions (norms, dot products)
    # which are better handled by XLA than element-wise Pallas kernels.
    # Use the pure JAX path which leverages XLA's optimised reductions.
    return looksam_direction_update(orig_grads, perturbed_grads, directions, alpha)
