"""TPU/Pallas kernels for Prodigy — distance-aware self-tuning Adam.

Prodigy automatically estimates the optimal learning rate by tracking the
distance between current parameters and their initial values.  Two-phase
update: first compute adaptive d_lr from gradient-parameter distance
statistics, then apply AdamW with effective lr = d_lr * lr.

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


def prodigy_update(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    s: jnp.ndarray,
    param_init: jnp.ndarray,
    step: int,
    d_lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1.0,
    wd: float = 1.0,
    eps: float = 1e-8,
    nan_policy: NanPolicy = NanPolicy.IGNORE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Prodigy distance-aware Adam update (pure JAX).

    Two-phase algorithm:
      Phase 1 — Compute d_numerator = sum(grad * (param - param_init))
                and d_denominator = sum(|s|) to update d_lr.
      Phase 2 — Apply AdamW with effective_lr = d_lr * lr.

    Args:
        params: Current parameters (bf16).
        grads: Gradients (bf16).
        exp_avg: First moment estimate (f32).
        exp_avg_sq: Second moment estimate (f32).
        s: Accumulated gradient signal buffer (f32).
        param_init: Initial parameter snapshot (f32).
        step: Current optimiser step (1-indexed).
        d_lr: Current adaptive learning rate scalar.
        beta1: First moment decay.
        beta2: Second moment decay.
        lr: Outer learning rate multiplier.
        wd: Weight decay coefficient.
        eps: Numerical stability term.
        nan_policy: How to handle NaN gradients.

    Returns:
        (updated_params, updated_exp_avg, updated_exp_avg_sq, updated_s, new_d_lr)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), nan_policy)
    p = params.astype(ACCUM_DTYPE)
    p0 = param_init.astype(ACCUM_DTYPE)

    # Phase 1: Distance-aware d_lr estimation
    # d_numerator tracks how much the gradient aligns with parameter displacement
    d_numerator = jnp.sum(g * (p - p0))
    # Update signal accumulator: s tracks exponentially weighted gradient signal
    new_s = beta2 * s + (1.0 - beta2) * d_lr * g
    d_denominator = jnp.sum(jnp.abs(new_s))

    # Update d_lr: ratio of alignment to accumulated signal magnitude
    new_d_lr = jnp.where(
        d_denominator > 0.0,
        jnp.maximum(d_lr, d_numerator / (d_denominator + 1e-12)),
        d_lr,
    )

    # Phase 2: AdamW with effective lr = d_lr * lr
    effective_lr = new_d_lr * lr

    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (g * g)

    # Bias correction
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    # Decoupled weight decay + Adam step
    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - effective_lr * (update + wd * p)

    return (
        p.astype(PARAM_DTYPE),
        new_exp_avg,
        new_exp_avg_sq,
        new_s,
        new_d_lr,
    )


# ---------------------------------------------------------------------------
# Pallas TPU kernel
# ---------------------------------------------------------------------------


def _prodigy_kernel_body(
    params_ref, grads_ref, exp_avg_ref, exp_avg_sq_ref, s_ref, param_init_ref,
    out_params_ref, out_exp_avg_ref, out_exp_avg_sq_ref, out_s_ref,
    *, beta1, beta2, effective_lr, wd, eps, step, d_lr,
):
    """Pallas kernel body for the element-wise portion of Prodigy update.

    NOTE: d_lr update (the global reduction) is computed on the host; this
    kernel only handles the per-element AdamW step with the pre-computed
    effective_lr = d_lr * lr.
    """
    p = params_ref[...].astype(jnp.float32)
    g = grads_ref[...].astype(jnp.float32)
    m = exp_avg_ref[...]
    v = exp_avg_sq_ref[...]
    s_val = s_ref[...]

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * (g * g)
    new_s = beta2 * s_val + (1.0 - beta2) * d_lr * g

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_m / bc1
    v_hat = new_v / bc2

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - effective_lr * (update + wd * p)

    out_params_ref[...] = p.astype(jnp.bfloat16)
    out_exp_avg_ref[...] = new_m
    out_exp_avg_sq_ref[...] = new_v
    out_s_ref[...] = new_s


def prodigy_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    s: jnp.ndarray,
    param_init: jnp.ndarray,
    step: int,
    d_lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1.0,
    wd: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Pallas-accelerated Prodigy update for TPU.

    The d_lr update requires global reductions and is computed in pure JAX.
    The element-wise AdamW portion is dispatched to a Pallas kernel.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return prodigy_update(
            params, grads, exp_avg, exp_avg_sq, s, param_init,
            step, d_lr, beta1, beta2, lr, wd, eps,
        )

    # Phase 1: Global reductions for d_lr (pure JAX — XLA handles these well)
    g_f32 = grads.astype(ACCUM_DTYPE)
    p_f32 = params.astype(ACCUM_DTYPE)
    p0_f32 = param_init.astype(ACCUM_DTYPE)

    d_numerator = jnp.sum(g_f32 * (p_f32 - p0_f32))
    temp_s = beta2 * s + (1.0 - beta2) * d_lr * g_f32
    d_denominator = jnp.sum(jnp.abs(temp_s))
    new_d_lr = jnp.where(
        d_denominator > 0.0,
        jnp.maximum(d_lr, d_numerator / (d_denominator + 1e-12)),
        d_lr,
    )
    effective_lr = new_d_lr * lr

    # Phase 2: Pallas kernel for element-wise update
    kernel_fn = functools.partial(
        _prodigy_kernel_body,
        beta1=beta1, beta2=beta2, effective_lr=effective_lr,
        wd=wd, eps=eps, step=step, d_lr=d_lr,
    )

    out_params, out_exp_avg, out_exp_avg_sq, out_s = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(params.shape, PARAM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(exp_avg_sq.shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(s.shape, ACCUM_DTYPE),
        ],
        grid=(1,),
    )(params, grads, exp_avg, exp_avg_sq, s, param_init)

    return out_params, out_exp_avg, out_exp_avg_sq, out_s, new_d_lr
