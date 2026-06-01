"""TPU/Pallas kernels for NeuralGrok -- Adam with a learned MLP gradient amplifier.

The amplifier is a lightweight 2-layer MLP that maps gradient magnitudes to
per-element scale factors, enabling adaptive gradient amplification.

MLP forward:  h = relu(W1 @ |g| + b1)
              scale = W_last @ h + b_last
Amplified:    g_amp = g + alpha * scale * g + beta_amp * scale * sign(g)

AdamW is then applied to g_amp.

BF16 parameters, FP32 accumulators throughout.
"""

# ----------------------------------------------------------------------------
# TPU TREE STATUS (post-consolidation): REFERENCE / SPEC.
# The AUTHORITATIVE, executed Pallas path for tpu_v5p is
# csrc/backends/pallas/launch_<opt>.py (resolved by grokking_optimizers/
# profile.py and the race driver); the live Pallas kernels live in
# csrc/backends/pallas/_pallas_kernels.py. This *_tpu.py file is the
# arch-reference specification of the same math (JAX/jnp), kept in the
# unified kernel tree alongside the sm_90/.cuh and gfx942/.hip.hpp headers.
# It is NOT imported by the production path (no duplicated executing kernel
# bodies). TPU has no inline-asm concept; lowering is via XLA/Pallas BlockSpec.
# Dependency direction is intentionally the reverse of sm_90/gfx942: there the
# kernel-tree header is canonical and csrc/backends shims to it; for Pallas the
# csrc/backends launcher stays canonical because profile.py executes it by path.
# ----------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pure JAX implementation
# ---------------------------------------------------------------------------


def neuralgrok_update(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    step: int,
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W_last: jnp.ndarray,
    b_last: jnp.ndarray,
    alpha: float = 10.0,
    beta_amp: float = 4.0,
    hidden_dim: int = 128,
    beta1: float = 0.9,
    beta2: float = 0.98,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
    grad_clip: float = 1.0,
    nan_policy: NanPolicy = NanPolicy.IGNORE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NeuralGrok update with MLP gradient amplifier (pure JAX).

    The amplifier is a 2-layer MLP: relu(W1 @ |g| + b1) -> W_last @ h + b_last.
    The resulting scale factor is used to amplify gradients before AdamW.

    Args:
        params: Current parameters (bf16).
        grads: Gradients (bf16).
        exp_avg: First moment estimate (f32).
        exp_avg_sq: Second moment estimate (f32).
        step: Current optimiser step (1-indexed).
        W1: First layer weights [hidden_dim, 1] (f32).
        b1: First layer bias [hidden_dim] (f32).
        W_last: Last layer weights [1, hidden_dim] (f32).
        b_last: Last layer bias [1] (f32).
        alpha: Primary amplification scale factor.
        beta_amp: Secondary amplification scale factor (sign-based).
        hidden_dim: Width of the amplifier hidden layer.
        beta1: First moment decay.
        beta2: Second moment decay.
        lr: Learning rate.
        wd: Weight decay coefficient.
        eps: Numerical stability term.
        grad_clip: Maximum gradient norm for per-element clipping.
        nan_policy: How to handle NaN gradients.

    Returns:
        (updated_params, updated_exp_avg, updated_exp_avg_sq)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), nan_policy)
    p = params.astype(ACCUM_DTYPE)

    # Per-element gradient clipping
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # Flatten for MLP evaluation
    orig_shape = g.shape
    g_flat = g.reshape(-1)

    # MLP forward: operates on |g| per element
    # W1: [hidden_dim, 1], b1: [hidden_dim]
    # Input: |g| reshaped to [N, 1]
    abs_g = jnp.abs(g_flat)  # [N]

    # Vectorised MLP: broadcast across all elements
    # h = relu(W1 @ |g_i| + b1) for each element i
    # W1 is [H, 1], abs_g is [N] -> abs_g[:, None] is [N, 1]
    # h = relu(abs_g[:, None] * W1.T + b1) -> [N, H]
    h = jax.nn.relu(abs_g[:, None] * W1.astype(ACCUM_DTYPE).T + b1.astype(ACCUM_DTYPE))  # [N, H]

    # scale = W_last @ h + b_last -> [N, 1]
    scale = h @ W_last.astype(ACCUM_DTYPE).T + b_last.astype(ACCUM_DTYPE)  # [N, 1]
    scale = scale.squeeze(-1)  # [N]

    # Amplified gradient: g + alpha * scale * g + beta_amp * scale * sign(g)
    g_amp = g_flat + alpha * scale * g_flat + beta_amp * scale * jnp.sign(g_flat)
    g_amp = g_amp.reshape(orig_shape)

    # AdamW on amplified gradient
    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g_amp
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (g_amp * g_amp)

    # Bias correction
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    return p.astype(PARAM_DTYPE), new_exp_avg, new_exp_avg_sq


# ---------------------------------------------------------------------------
# Pallas TPU kernel
# ---------------------------------------------------------------------------


def _neuralgrok_kernel_body(
    params_ref, grads_ref, exp_avg_ref, exp_avg_sq_ref,
    W1_ref, b1_ref, W_last_ref, b_last_ref,
    out_params_ref, out_exp_avg_ref, out_exp_avg_sq_ref,
    *, alpha, beta_amp, beta1, beta2, lr, wd, eps, step, grad_clip,
):
    """Pallas kernel body for fused NeuralGrok update.

    Fuses gradient clipping, MLP amplification, and AdamW into a single kernel.
    """
    p = params_ref[...].astype(jnp.float32)
    g = grads_ref[...].astype(jnp.float32)
    m = exp_avg_ref[...]
    v = exp_avg_sq_ref[...]
    w1 = W1_ref[...]   # [H, 1]
    b1_val = b1_ref[...]   # [H]
    wl = W_last_ref[...]   # [1, H]
    bl = b_last_ref[...]   # [1]

    # Per-element gradient clipping
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # MLP amplification (element-wise, flattened)
    g_flat = g.reshape(-1)
    abs_g = jnp.abs(g_flat)
    h = jax.nn.relu(abs_g[:, None] * w1.T + b1_val)
    scale = (h @ wl.T + bl).squeeze(-1)

    g_amp = g_flat + alpha * scale * g_flat + beta_amp * scale * jnp.sign(g_flat)
    g_amp = g_amp.reshape(g.shape)

    # AdamW
    new_m = beta1 * m + (1.0 - beta1) * g_amp
    new_v = beta2 * v + (1.0 - beta2) * (g_amp * g_amp)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    out_params_ref[...] = p.astype(jnp.bfloat16)
    out_exp_avg_ref[...] = new_m
    out_exp_avg_sq_ref[...] = new_v


def neuralgrok_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    step: int,
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W_last: jnp.ndarray,
    b_last: jnp.ndarray,
    alpha: float = 10.0,
    beta_amp: float = 4.0,
    hidden_dim: int = 128,
    beta1: float = 0.9,
    beta2: float = 0.98,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
    grad_clip: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pallas-accelerated NeuralGrok update for TPU.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return neuralgrok_update(
            params, grads, exp_avg, exp_avg_sq, step,
            W1, b1, W_last, b_last,
            alpha, beta_amp, hidden_dim,
            beta1, beta2, lr, wd, eps, grad_clip,
        )

    kernel_fn = functools.partial(
        _neuralgrok_kernel_body,
        alpha=alpha, beta_amp=beta_amp,
        beta1=beta1, beta2=beta2, lr=lr, wd=wd, eps=eps,
        step=step, grad_clip=grad_clip,
    )

    out_params, out_exp_avg, out_exp_avg_sq = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(params.shape, PARAM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg_sq.shape, ACCUM_DTYPE),
        ],
        grid=(1,),
    )(params, grads, exp_avg, exp_avg_sq, W1, b1, W_last, b_last)

    return out_params, out_exp_avg, out_exp_avg_sq
