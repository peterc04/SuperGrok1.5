"""TPU/Pallas kernels for SuperGrok v1.5 -- sigmoid-gated meta-net optimizer.

Same architecture as v1.1 (SharpnessMetaNet + AdamW) but replaces per-element
cosine similarity gating with a host-precomputed scalar gate_signal (sigmoid of
training accuracy). This simplifies the per-element kernel while keeping the
adaptive scheduling behaviour.

MetaNet:  inp = stack(g, sharpness)
          h = gelu(W1 @ inp + b1)
          correction = rescale * (W2 @ h + b2)
          smart_g = g + correction

Effective:  effective_g = g + ramp * lamb * gate_signal * (smart_g - g)

AdamW is then applied to effective_g with per-layer beta1 and alpha scaling.

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


# ---------------------------------------------------------------------------
# Pure JAX implementation
# ---------------------------------------------------------------------------


def supergrok15_update(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    mu: jnp.ndarray,
    sharpness: jnp.ndarray,
    step: int,
    layer_alpha: float,
    layer_beta1: float,
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray,
    rescale: float,
    hidden_dim: int = 32,
    beta2: float = 0.999,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
    lamb: float = 2.0,
    ramp: float = 1.0,
    gate_signal: float = 0.0,
    grad_clip: float = 1.0,
    nan_policy: NanPolicy = NanPolicy.IGNORE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SuperGrok v1.5 update with host-precomputed gate signal (pure JAX).

    Identical to v1.1 except the gate is a scalar ``gate_signal`` computed on
    the host via sigmoid(gate_scale * (train_acc - gate_thresh)), rather than
    per-element cosine similarity.

    Args:
        params: Current parameters (bf16).
        grads: Gradients (bf16).
        exp_avg: First moment estimate (f32).
        exp_avg_sq: Second moment estimate (f32).
        mu: Momentum buffer (f32).
        sharpness: Per-element sharpness estimate (f32).
        step: Current optimiser step (1-indexed).
        layer_alpha: Per-layer alpha scaling factor.
        layer_beta1: Per-layer first moment decay.
        W1: MetaNet first layer weights [hidden_dim, 2] (f32).
        b1: MetaNet first layer bias [hidden_dim] (f32).
        W2: MetaNet last layer weights [1, hidden_dim] (f32).
        b2: MetaNet last layer bias [1] (f32).
        rescale: MetaNet output rescaling factor.
        hidden_dim: Width of meta-net hidden layer.
        beta2: Second moment decay.
        lr: Learning rate.
        wd: Weight decay coefficient (may include progressive WD).
        eps: Numerical stability term.
        lamb: Gating strength multiplier.
        ramp: Warmup ramp factor (0 during warmup, 1 after).
        gate_signal: Host-precomputed scalar gate (sigmoid of train accuracy).
        grad_clip: Maximum gradient norm.
        nan_policy: How to handle NaN gradients.

    Returns:
        (updated_params, updated_exp_avg, updated_exp_avg_sq, updated_mu)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), nan_policy)
    p = params.astype(ACCUM_DTYPE)
    s = sharpness.astype(ACCUM_DTYPE)

    # Per-element gradient clipping
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # MetaNet forward: inp = stack(g, sharpness) per element
    orig_shape = g.shape
    g_flat = g.reshape(-1)
    s_flat = s.reshape(-1)

    inp = jnp.stack([g_flat, s_flat], axis=-1)  # [N, 2]
    h = jax.nn.gelu(inp @ W1.astype(ACCUM_DTYPE).T + b1.astype(ACCUM_DTYPE))  # [N, H]
    correction = rescale * (h @ W2.astype(ACCUM_DTYPE).T + b2.astype(ACCUM_DTYPE))  # [N, 1]
    correction = correction.squeeze(-1)  # [N]

    smart_g_flat = g_flat + correction
    smart_g = smart_g_flat.reshape(orig_shape)

    # Effective gradient with host-precomputed gate signal (scalar)
    effective_g = g + ramp * lamb * gate_signal * (smart_g - g)

    # AdamW with per-layer beta1
    new_exp_avg = layer_beta1 * exp_avg + (1.0 - layer_beta1) * effective_g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (effective_g * effective_g)

    # Bias correction
    bc1 = 1.0 - layer_beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    # Momentum update
    new_mu = layer_beta1 * mu + (1.0 - layer_beta1) * effective_g

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    return p.astype(PARAM_DTYPE), new_exp_avg, new_exp_avg_sq, new_mu


# ---------------------------------------------------------------------------
# Pallas TPU kernel
# ---------------------------------------------------------------------------


def _supergrok15_kernel_body(
    params_ref, grads_ref, exp_avg_ref, exp_avg_sq_ref, mu_ref,
    sharpness_ref, W1_ref, b1_ref, W2_ref, b2_ref,
    out_params_ref, out_exp_avg_ref, out_exp_avg_sq_ref, out_mu_ref,
    *, rescale, layer_alpha, layer_beta1, beta2, lr, wd, eps,
    lamb, ramp, gate_signal, step, grad_clip,
):
    """Pallas kernel body for fused SuperGrok v1.5 update."""
    p = params_ref[...].astype(jnp.float32)
    g = grads_ref[...].astype(jnp.float32)
    m = exp_avg_ref[...]
    v = exp_avg_sq_ref[...]
    mu_val = mu_ref[...]
    s = sharpness_ref[...].astype(jnp.float32)
    w1 = W1_ref[...]
    b1_val = b1_ref[...]
    w2 = W2_ref[...]
    b2_val = b2_ref[...]

    # Gradient clipping
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # MetaNet forward
    g_flat = g.reshape(-1)
    s_flat = s.reshape(-1)
    inp = jnp.stack([g_flat, s_flat], axis=-1)
    h = jax.nn.gelu(inp @ w1.T + b1_val)
    correction = rescale * (h @ w2.T + b2_val)
    smart_g_flat = g_flat + correction.squeeze(-1)
    smart_g = smart_g_flat.reshape(g.shape)

    # Host-precomputed scalar gate
    effective_g = g + ramp * lamb * gate_signal * (smart_g - g)

    # AdamW
    new_m = layer_beta1 * m + (1.0 - layer_beta1) * effective_g
    new_v = beta2 * v + (1.0 - beta2) * (effective_g * effective_g)
    new_mu = layer_beta1 * mu_val + (1.0 - layer_beta1) * effective_g

    bc1 = 1.0 - layer_beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    out_params_ref[...] = p.astype(jnp.bfloat16)
    out_exp_avg_ref[...] = new_m
    out_exp_avg_sq_ref[...] = new_v
    out_mu_ref[...] = new_mu


def supergrok15_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    mu: jnp.ndarray,
    sharpness: jnp.ndarray,
    step: int,
    layer_alpha: float,
    layer_beta1: float,
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray,
    rescale: float,
    hidden_dim: int = 32,
    beta2: float = 0.999,
    lr: float = 1e-3,
    wd: float = 1.0,
    eps: float = 1e-8,
    lamb: float = 2.0,
    ramp: float = 1.0,
    gate_signal: float = 0.0,
    grad_clip: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pallas-accelerated SuperGrok v1.5 update for TPU.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return supergrok15_update(
            params, grads, exp_avg, exp_avg_sq, mu, sharpness, step,
            layer_alpha, layer_beta1, W1, b1, W2, b2, rescale,
            hidden_dim, beta2, lr, wd, eps, lamb, ramp, gate_signal,
            grad_clip,
        )

    kernel_fn = functools.partial(
        _supergrok15_kernel_body,
        rescale=rescale, layer_alpha=layer_alpha, layer_beta1=layer_beta1,
        beta2=beta2, lr=lr, wd=wd, eps=eps,
        lamb=lamb, ramp=ramp, gate_signal=gate_signal,
        step=step, grad_clip=grad_clip,
    )

    out_params, out_exp_avg, out_exp_avg_sq, out_mu = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(params.shape, PARAM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg_sq.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(mu.shape, ACCUM_DTYPE),
        ],
        grid=(1,),
    )(params, grads, exp_avg, exp_avg_sq, mu, sharpness, W1, b1, W2, b2)

    return out_params, out_exp_avg, out_exp_avg_sq, out_mu
