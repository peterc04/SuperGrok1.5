"""TPU/Pallas kernels for Muon -- Newton-Schulz orthogonalisation optimizer.

Muon applies orthogonalised gradient updates to 2D (matrix) parameters via
iterative Newton-Schulz orthogonalisation, while falling back to standard
AdamW for 1D parameters.

Newton-Schulz iteration (for approximate polar decomposition):
    X_{k+1} = a * X_k + b * (X_k @ X_k^T) @ X_k
              + c * ((X_k @ X_k^T) @ (X_k @ X_k^T)) @ X_k
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315

Implemented via jax.lax.fori_loop for TPU-efficient iteration.

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

try:
    from jax.experimental import pallas as pl
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False


# Newton-Schulz coefficients for the cubic iteration
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


# ---------------------------------------------------------------------------
# Pure JAX implementations
# ---------------------------------------------------------------------------


def _newton_schulz_orthogonalise(
    G: jnp.ndarray,
    ns_steps: int = 5,
) -> jnp.ndarray:
    """Approximate polar decomposition via Newton-Schulz iteration.

    Given a matrix G, computes the nearest orthogonal matrix via iterative
    Newton-Schulz steps:
        X_{k+1} = a*X + b*(X @ X^T) @ X + c*((X @ X^T)^2) @ X

    The matrix is first normalised to have unit spectral norm (approx).

    Args:
        G: Input matrix [M, N] (f32).
        ns_steps: Number of Newton-Schulz iterations.

    Returns:
        Orthogonalised matrix [M, N] (f32).
    """
    # Normalise by approximate spectral norm (Frobenius / sqrt(min(M,N)))
    M, N = G.shape
    norm_factor = jnp.sqrt(jnp.sum(G * G)) / jnp.sqrt(jnp.minimum(M, N).astype(jnp.float32))
    X = G / (norm_factor + 1e-12)

    def _ns_step(i, X_cur):
        # XXT = X @ X^T  [M, M]
        XXT = X_cur @ X_cur.T
        # XXT2 = XXT @ XXT  [M, M]
        XXT2 = XXT @ XXT
        # X_next = a*X + b*(XXT @ X) + c*(XXT2 @ X)
        X_next = _NS_A * X_cur + _NS_B * (XXT @ X_cur) + _NS_C * (XXT2 @ X_cur)
        return X_next

    X = lax.fori_loop(0, ns_steps, _ns_step, X)
    return X


def muon_update_2d(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    momentum_buf: jnp.ndarray,
    momentum_coeff: float = 0.95,
    lr: float = 0.02,
    wd: float = 1.0,
    ns_steps: int = 5,
    nan_policy: NanPolicy = NanPolicy.IGNORE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Muon update for 2D parameters: momentum + Newton-Schulz (pure JAX).

    Steps:
      1. Momentum: buf = momentum * buf + grad
      2. Newton-Schulz orthogonalise the momentum buffer
      3. Weight decay + scaled update

    Args:
        params: Current 2D parameters (bf16).
        grads: Gradients (bf16).
        momentum_buf: Momentum buffer (f32).
        momentum_coeff: Momentum coefficient.
        lr: Learning rate.
        wd: Weight decay coefficient.
        ns_steps: Number of Newton-Schulz iterations.
        nan_policy: How to handle NaN gradients.

    Returns:
        (updated_params, updated_momentum_buf)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), nan_policy)
    p = params.astype(ACCUM_DTYPE)

    # Momentum accumulation
    new_buf = momentum_coeff * momentum_buf + g

    # Newton-Schulz orthogonalisation (requires 2D input)
    ortho = _newton_schulz_orthogonalise(new_buf, ns_steps)

    # Weight decay + update
    p = p - lr * (ortho + wd * p)

    return p.astype(PARAM_DTYPE), new_buf


def muon_adamw_update(
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
    """Standard AdamW update for non-2D (1D) Muon parameters (pure JAX).

    For biases, layer norms, embeddings, and other non-matrix parameters.

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

    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (g * g)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    return p.astype(PARAM_DTYPE), new_exp_avg, new_exp_avg_sq


# ---------------------------------------------------------------------------
# Pallas TPU kernels
# ---------------------------------------------------------------------------


def _muon_adamw_kernel_body(
    params_ref, grads_ref, exp_avg_ref, exp_avg_sq_ref,
    out_params_ref, out_exp_avg_ref, out_exp_avg_sq_ref,
    *, beta1, beta2, lr, wd, eps, step,
):
    """Pallas kernel body for fused AdamW update (1D Muon params)."""
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


def muon_2d_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    momentum_buf: jnp.ndarray,
    momentum_coeff: float = 0.95,
    lr: float = 0.02,
    wd: float = 1.0,
    ns_steps: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pallas-accelerated Muon 2D update for TPU.

    The Newton-Schulz iteration involves matrix multiplications that benefit
    from XLA's MXU mapping. We use the pure JAX path which XLA compiles
    to efficient TPU matmul instructions. The momentum and weight-decay
    portions are element-wise and could use Pallas, but the NS iteration
    dominates compute, so a single XLA-compiled function is preferred.

    Falls back to pure JAX if Pallas is unavailable.
    """
    # Newton-Schulz involves matrix multiplications (MXU-bound), not
    # element-wise ops. XLA does an excellent job mapping these to TPU's MXU.
    # Using pure JAX for the entire update is optimal here.
    return muon_update_2d(
        params, grads, momentum_buf, momentum_coeff, lr, wd, ns_steps,
    )


def muon_adamw_pallas_kernel(
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
    """Pallas-accelerated AdamW update for 1D Muon params on TPU.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return muon_adamw_update(
            params, grads, exp_avg, exp_avg_sq, step,
            beta1, beta2, lr, wd, eps,
        )

    kernel_fn = functools.partial(
        _muon_adamw_kernel_body,
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
