"""TPU v5p Pallas module for the Mamba-3 selective state-space model.

Implements the full Mamba-3 forward and backward passes using JAX/Pallas
primitives optimised for TPU: lax.associative_scan for the SSM core,
jnp.dot for MXU matmuls, and lax.conv_general_dilated for depthwise conv1d.
BF16 parameters, FP32 accumulators throughout.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import lax

from grokking_optimizers.kernels.tpu.common_tpu import (
    ACCUM_DTYPE,
    PARAM_DTYPE,
    NanPolicy,
    apply_nan_policy,
    to_bf16,
    to_f32,
)

# ---------------------------------------------------------------------------
# Resource declarations (API compat with CUDA/HIP headers)
# ---------------------------------------------------------------------------

SSM_SCAN_VMEM_TILES: int = 8
CONV1D_SMEM_BYTES: int = 0  # TPU has no shared memory


# ---------------------------------------------------------------------------
# Weight dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Mamba3LayerWeights:
    """Weights for a single Mamba-3 layer."""

    norm_g: jnp.ndarray           # [D_MODEL]
    norm_b: jnp.ndarray           # [D_MODEL]
    in_proj_w: jnp.ndarray        # [2*D_INNER, D_MODEL]
    conv_w: jnp.ndarray           # [D_INNER, 3]
    conv_b: jnp.ndarray           # [D_INNER]
    x_proj_w: jnp.ndarray         # [DT_RANK+2*D_STATE, D_INNER]
    dt_proj_w: jnp.ndarray        # [D_INNER, DT_RANK]
    dt_proj_b: jnp.ndarray        # [D_INNER]
    A_log: jnp.ndarray            # [D_INNER, D_STATE]
    D_param: jnp.ndarray          # [D_INNER]
    out_proj_w: jnp.ndarray       # [D_MODEL, D_INNER]
    rope_freq: jnp.ndarray        # [D_INNER, D_STATE//2]


@dataclasses.dataclass
class Mamba3State:
    """Full model state for Mamba-3."""

    tok_embed: jnp.ndarray        # [VOCAB, D_MODEL]
    layer_weights: list           # list of Mamba3LayerWeights
    final_norm_g: jnp.ndarray     # [D_MODEL]
    final_norm_b: jnp.ndarray     # [D_MODEL]
    lm_head_w: jnp.ndarray | None # [VOCAB, D_MODEL] or None if tied
    lm_head_b: jnp.ndarray        # [VOCAB]
    d_model: int
    d_inner: int
    d_state: int
    dt_rank: int
    tied_embeddings: bool


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


def param_bytes(d_model: int, d_inner: int, d_state: int,
                dt_rank: int, vocab_size: int, n_layers: int,
                tied: bool = True) -> int:
    """Total parameter bytes (BF16 = 2 bytes per element)."""
    bpe = 2  # bf16
    embed = vocab_size * d_model
    per_layer = (
        2 * d_model                          # norm_g, norm_b
        + 2 * d_inner * d_model              # in_proj_w
        + d_inner * 3 + d_inner              # conv_w, conv_b
        + (dt_rank + 2 * d_state) * d_inner  # x_proj_w
        + d_inner * dt_rank + d_inner        # dt_proj_w, dt_proj_b
        + d_inner * d_state                  # A_log
        + d_inner                            # D_param
        + d_model * d_inner                  # out_proj_w
        + d_inner * (d_state // 2)           # rope_freq
    )
    head = 0 if tied else vocab_size * d_model
    head += vocab_size  # lm_head_b
    final_norm = 2 * d_model
    total_elems = embed + n_layers * per_layer + head + final_norm
    return total_elems * bpe


def num_param_tensors(n_layers: int) -> int:
    """Total number of parameter tensors in the model."""
    per_layer = 12  # fields in Mamba3LayerWeights
    global_tensors = 5  # tok_embed, final_norm_g, final_norm_b, lm_head_w, lm_head_b
    return global_tensors + n_layers * per_layer


def activation_bytes_per_layer(batch: int, seq_len: int, d_model: int,
                               d_inner: int, d_state: int,
                               dt_rank: int) -> int:
    """Activation memory per layer in bytes (FP32 = 4 bytes)."""
    bpe = 4  # f32 activations
    acts = (
        batch * seq_len * d_model              # residual
        + batch * seq_len * 2 * d_inner        # xz after in_proj
        + batch * seq_len * d_inner            # x_main after conv
        + batch * seq_len * d_inner            # x_main after silu
        + batch * seq_len * (dt_rank + 2 * d_state)  # x_proj output
        + batch * seq_len * d_inner            # dt after softplus
        + batch * seq_len * d_inner * d_state  # ssm hidden states
        + batch * seq_len * d_inner            # y from scan
        + batch * seq_len * d_inner            # gated output
        + batch * seq_len * d_model            # out_proj
    )
    return acts * bpe


# ---------------------------------------------------------------------------
# Per-layer forward functions
# ---------------------------------------------------------------------------


def embed_forward(tok_embed: jnp.ndarray, ids: jnp.ndarray) -> jnp.ndarray:
    """Token embedding lookup."""
    return tok_embed[ids]


def rmsnorm_forward(x: jnp.ndarray, gamma: jnp.ndarray,
                    eps: float = 1e-6) -> jnp.ndarray:
    """RMSNorm: x * gamma * rsqrt(mean(x^2) + eps)."""
    x_f32 = to_f32(x)
    rms = jnp.sqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return to_bf16(x_f32 / rms * to_f32(gamma))


def in_proj_forward(x: jnp.ndarray,
                    w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Input projection -> (x_main, z), each [B, S, D_INNER]."""
    xz = jnp.dot(to_f32(x), to_f32(w).T)  # [B, S, 2*D_INNER]
    xz = to_bf16(xz)
    d_inner = w.shape[0] // 2
    x_main = xz[..., :d_inner]
    z = xz[..., d_inner:]
    return x_main, z


def conv1d_forward(x: jnp.ndarray, w: jnp.ndarray,
                   b: jnp.ndarray) -> jnp.ndarray:
    """Depthwise conv1d (k=3, pad=1) followed by SiLU activation."""
    # x: [B, S, D_INNER] -> transpose to [B, D_INNER, S] for conv
    x_t = jnp.transpose(to_f32(x), (0, 2, 1))  # [B, D_INNER, S]
    d_inner = x.shape[-1]
    # w: [D_INNER, 3] -> reshape to [D_INNER, 1, 3] for depthwise
    kernel = to_f32(w)[:, None, :]  # [D_INNER, 1, 3]
    out = lax.conv_general_dilated(
        x_t,                             # [B, D_INNER, S]
        kernel,                          # [D_INNER, 1, 3]
        window_strides=(1,),
        padding=((1, 1),),               # pad=1 on each side for k=3
        feature_group_count=d_inner,
        dimension_numbers=('NCS', 'OIS', 'NCS'),
    )  # [B, D_INNER, S]
    out = out + to_f32(b)[None, :, None]
    # SiLU activation
    out = out * jax.nn.sigmoid(out)
    return to_bf16(jnp.transpose(out, (0, 2, 1)))  # [B, S, D_INNER]


def x_proj_forward(x: jnp.ndarray, w: jnp.ndarray, dt_rank: int,
                   d_state: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """X projection -> (dt_raw, B_ssm, C_ssm)."""
    proj = jnp.dot(to_f32(x), to_f32(w).T)  # [B, S, DT_RANK + 2*D_STATE]
    proj = to_bf16(proj)
    dt_raw = proj[..., :dt_rank]
    B_ssm = proj[..., dt_rank:dt_rank + d_state]
    C_ssm = proj[..., dt_rank + d_state:]
    return dt_raw, B_ssm, C_ssm


def dt_proj_forward(dt_raw: jnp.ndarray, w: jnp.ndarray,
                    b: jnp.ndarray) -> jnp.ndarray:
    """dt projection: linear + bias + softplus."""
    dt = jnp.dot(to_f32(dt_raw), to_f32(w).T) + to_f32(b)  # [B, S, D_INNER]
    dt = jnp.log1p(jnp.exp(dt))  # softplus
    return to_bf16(dt)


def rope_apply_ssm(state: jnp.ndarray,
                   freq: jnp.ndarray) -> jnp.ndarray:
    """Paired RoPE rotation on SSM hidden state."""
    # state: [B, S, D_INNER, D_STATE], freq: [D_INNER, D_STATE//2]
    state_f32 = to_f32(state)
    half = state_f32.shape[-1] // 2
    s0 = state_f32[..., :half]
    s1 = state_f32[..., half:]
    cos_f = jnp.cos(to_f32(freq))  # [D_INNER, D_STATE//2]
    sin_f = jnp.sin(to_f32(freq))
    r0 = s0 * cos_f - s1 * sin_f
    r1 = s0 * sin_f + s1 * cos_f
    return to_bf16(jnp.concatenate([r0, r1], axis=-1))


def selective_scan_forward(
    x: jnp.ndarray,
    dt: jnp.ndarray,
    A_log: jnp.ndarray,
    B_ssm: jnp.ndarray,
    C_ssm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Core SSM scan via lax.associative_scan (TPU-native tree scan)."""
    # x: [B, S, D_INNER], dt: [B, S, D_INNER]
    # A_log: [D_INNER, D_STATE], B_ssm: [B, S, D_STATE], C_ssm: [B, S, D_STATE]
    x_f32 = to_f32(x)
    dt_f32 = to_f32(dt)
    A = jnp.exp(to_f32(A_log))  # [D_INNER, D_STATE]

    # Discretization
    A_bar = jnp.exp(dt_f32[..., None] * A[None, None, :, :])  # [B, S, D_INNER, D_STATE]
    B_bar = dt_f32[..., None] * to_f32(B_ssm)[:, :, None, :]  # [B, S, D_INNER, D_STATE]
    x_bar = B_bar * x_f32[..., None]  # [B, S, D_INNER, D_STATE]

    # Associative scan: combine (a1, b1) o (a2, b2) = (a1*a2, a2*b1 + b2)
    def _combine(carry: tuple[jnp.ndarray, jnp.ndarray],
                 incoming: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        a1, b1 = carry
        a2, b2 = incoming
        return a1 * a2, a2 * b1 + b2

    # Scan along sequence axis (axis=1)
    A_scan, h_scan = lax.associative_scan(
        _combine, (A_bar, x_bar), axis=1
    )  # both [B, S, D_INNER, D_STATE]

    # Output: y[t] = sum_over_state(C[t] * h[t])
    y = jnp.sum(to_f32(C_ssm)[:, :, None, :] * h_scan, axis=-1)  # [B, S, D_INNER]
    return to_bf16(y), to_bf16(h_scan)


def gate_multiply_forward(y: jnp.ndarray, x_main: jnp.ndarray,
                          D_param: jnp.ndarray,
                          z: jnp.ndarray) -> jnp.ndarray:
    """Gated output: (y + x_main * D) * silu(z)."""
    y_f32 = to_f32(y) + to_f32(x_main) * to_f32(D_param)
    z_f32 = to_f32(z)
    gated = y_f32 * (z_f32 * jax.nn.sigmoid(z_f32))  # silu(z) = z * sigmoid(z)
    return to_bf16(gated)


def out_proj_forward(y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Output projection."""
    return to_bf16(jnp.dot(to_f32(y), to_f32(w).T))


def residual_add(x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
    """Residual connection."""
    return to_bf16(to_f32(x) + to_f32(residual))


def lm_head_forward(x: jnp.ndarray, state: Mamba3State) -> jnp.ndarray:
    """Final LM head projection to logits."""
    w = state.tok_embed if state.tied_embeddings else state.lm_head_w
    logits = jnp.dot(to_f32(x), to_f32(w).T) + to_f32(state.lm_head_b)
    return logits  # keep f32 for loss


# ---------------------------------------------------------------------------
# Per-layer backward functions
# ---------------------------------------------------------------------------


def rmsnorm_backward(dy: jnp.ndarray, x: jnp.ndarray,
                     gamma: jnp.ndarray,
                     eps: float = 1e-6) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for RMSNorm -> (dx, dgamma)."""
    x_f32 = to_f32(x)
    dy_f32 = to_f32(dy)
    g_f32 = to_f32(gamma)
    d = x_f32.shape[-1]
    var = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps
    rms_inv = jnp.rsqrt(var)
    x_hat = x_f32 * rms_inv
    dgamma = jnp.sum(dy_f32 * x_hat, axis=tuple(range(dy_f32.ndim - 1)))
    dx_hat = dy_f32 * g_f32
    dx = rms_inv * (dx_hat - x_hat * jnp.mean(dx_hat * x_hat, axis=-1, keepdims=True))
    return to_bf16(dx), to_bf16(dgamma)


def in_proj_backward(dy_main: jnp.ndarray, dy_z: jnp.ndarray,
                     x: jnp.ndarray,
                     w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for in_proj -> (dx, dw)."""
    dy = jnp.concatenate([to_f32(dy_main), to_f32(dy_z)], axis=-1)
    x_f32 = to_f32(x)
    w_f32 = to_f32(w)
    dx = jnp.dot(dy, w_f32)  # [B, S, D_MODEL]
    # dw: [2*D_INNER, D_MODEL] = dy^T @ x, summed over batch and seq
    dw = jnp.einsum('bsi,bsj->ij', dy, x_f32)
    return to_bf16(dx), to_bf16(dw)


def conv1d_backward(dy: jnp.ndarray, x: jnp.ndarray, w: jnp.ndarray,
                    b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for conv1d + SiLU -> (dx, dw, db)."""
    # Recompute forward for SiLU gradient
    x_f32 = to_f32(x)
    x_t = jnp.transpose(x_f32, (0, 2, 1))  # [B, D_INNER, S]
    d_inner = x.shape[-1]
    kernel = to_f32(w)[:, None, :]
    conv_out = lax.conv_general_dilated(
        x_t, kernel, window_strides=(1,), padding=((1, 1),),
        feature_group_count=d_inner,
        dimension_numbers=('NCS', 'OIS', 'NCS'),
    ) + to_f32(b)[None, :, None]
    sig = jax.nn.sigmoid(conv_out)
    silu_grad = sig * (1.0 + conv_out * (1.0 - sig))

    dy_t = jnp.transpose(to_f32(dy), (0, 2, 1))  # [B, D_INNER, S]
    d_conv = dy_t * silu_grad  # gradient through SiLU

    # db: sum over batch and seq
    db = jnp.sum(d_conv, axis=(0, 2))  # [D_INNER]

    # dw via cross-correlation of x_t with d_conv
    # Use conv to compute gradient w.r.t. kernel
    # Flip the operation: dx_t via transposed conv
    kernel_flip = kernel[:, :, ::-1]
    dx_t = lax.conv_general_dilated(
        d_conv, kernel_flip, window_strides=(1,), padding=((1, 1),),
        feature_group_count=d_inner,
        dimension_numbers=('NCS', 'OIS', 'NCS'),
    )
    dx = jnp.transpose(dx_t, (0, 2, 1))

    # dw: [D_INNER, 3] — correlate d_conv with x_t
    # For depthwise conv, each filter channel is independent
    dw = jnp.zeros_like(to_f32(w))  # [D_INNER, 3]
    x_padded = jnp.pad(x_t, ((0, 0), (0, 0), (1, 1)))  # [B, D_INNER, S+2]
    for k in range(3):
        dw = dw.at[:, k].set(
            jnp.sum(d_conv * x_padded[:, :, k:k + x_t.shape[2]], axis=(0, 2))
        )

    return to_bf16(dx), to_bf16(dw), to_bf16(db)


def x_proj_backward(d_dt_raw: jnp.ndarray, d_B: jnp.ndarray,
                    d_C: jnp.ndarray, x: jnp.ndarray,
                    w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for x_proj -> (dx, dw)."""
    d_proj = jnp.concatenate([to_f32(d_dt_raw), to_f32(d_B), to_f32(d_C)], axis=-1)
    x_f32 = to_f32(x)
    w_f32 = to_f32(w)
    dx = jnp.dot(d_proj, w_f32)
    dw = jnp.einsum('bsi,bsj->ij', d_proj, x_f32)
    return to_bf16(dx), to_bf16(dw)


def dt_proj_backward(d_dt: jnp.ndarray, dt_raw: jnp.ndarray,
                     w: jnp.ndarray,
                     b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for dt_proj (linear + softplus) -> (d_dt_raw, dw, db)."""
    # Recompute pre-softplus value
    pre_sp = jnp.dot(to_f32(dt_raw), to_f32(w).T) + to_f32(b)
    # softplus gradient: sigmoid(x)
    sp_grad = jax.nn.sigmoid(pre_sp)
    d_pre = to_f32(d_dt) * sp_grad  # [B, S, D_INNER]

    db = jnp.sum(d_pre, axis=(0, 1))  # [D_INNER]
    dw = jnp.einsum('bsi,bsj->ij', d_pre, to_f32(dt_raw))  # [D_INNER, DT_RANK]
    d_dt_raw = jnp.dot(d_pre, to_f32(w))  # [B, S, DT_RANK]
    return to_bf16(d_dt_raw), to_bf16(dw), to_bf16(db)


def rope_apply_ssm_backward(d_rotated: jnp.ndarray,
                            state: jnp.ndarray,
                            freq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for RoPE on SSM state -> (d_state, d_freq)."""
    d_rot_f32 = to_f32(d_rotated)
    state_f32 = to_f32(state)
    half = state_f32.shape[-1] // 2
    s0, s1 = state_f32[..., :half], state_f32[..., half:]
    cos_f = jnp.cos(to_f32(freq))
    sin_f = jnp.sin(to_f32(freq))
    dr0, dr1 = d_rot_f32[..., :half], d_rot_f32[..., half:]

    ds0 = dr0 * cos_f + dr1 * sin_f
    ds1 = -dr0 * sin_f + dr1 * cos_f
    d_state = jnp.concatenate([ds0, ds1], axis=-1)

    d_cos = dr0 * s0 + dr1 * s1  # partial w.r.t. cos
    d_sin = -dr0 * s1 + dr1 * s0  # partial w.r.t. sin
    d_freq = -d_cos * sin_f + d_sin * cos_f
    # Sum over batch, seq, d_inner dims (freq is [D_INNER, D_STATE//2])
    d_freq = jnp.sum(d_freq, axis=tuple(range(d_freq.ndim - 2)))

    return to_bf16(d_state), to_bf16(d_freq)


def selective_scan_backward(
    dy: jnp.ndarray,
    x: jnp.ndarray,
    dt: jnp.ndarray,
    A_log: jnp.ndarray,
    B_ssm: jnp.ndarray,
    C_ssm: jnp.ndarray,
    h_scan: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for selective_scan -> (dx, d_dt, d_A_log, d_B, d_C)."""
    dy_f32 = to_f32(dy)
    x_f32 = to_f32(x)
    dt_f32 = to_f32(dt)
    h_f32 = to_f32(h_scan)
    C_f32 = to_f32(C_ssm)
    B_f32 = to_f32(B_ssm)
    A = jnp.exp(to_f32(A_log))

    # y = sum(C * h, axis=-1) -> dh from dy
    dh = dy_f32[..., None] * C_f32[:, :, None, :]  # [B, S, D_INNER, D_STATE]
    d_C = jnp.sum(dy_f32[..., None] * h_f32, axis=2)  # -> wrong shape
    # Correct: dy: [B,S,D_INNER], h: [B,S,D_INNER,D_STATE], C: [B,S,D_STATE]
    # y_t = sum_n(C_t_n * h_t_d_n) summed over d_state for each d_inner -> then sum d_inner? No.
    # y[b,s,d] = sum_n C[b,s,n] * h[b,s,d,n]
    # d_C[b,s,n] = sum_d dy[b,s,d] * h[b,s,d,n]
    d_C = jnp.einsum('bsd,bsdn->bsn', dy_f32, h_f32)

    # Backward through associative scan: reverse scan
    A_bar = jnp.exp(dt_f32[..., None] * A[None, None, :, :])
    B_bar = dt_f32[..., None] * B_f32[:, :, None, :]

    # Reverse scan for dh propagation
    def _combine_bwd(carry: tuple[jnp.ndarray, jnp.ndarray],
                     incoming: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        a1, b1 = carry
        a2, b2 = incoming
        return a1 * a2, a2 * b1 + b2

    # Flip along sequence dim for backward scan
    A_bar_flip = A_bar[:, ::-1]
    dh_flip = dh[:, ::-1]
    _, dh_accum = lax.associative_scan(_combine_bwd, (A_bar_flip, dh_flip), axis=1)
    dh_accum = dh_accum[:, ::-1]  # [B, S, D_INNER, D_STATE]

    # dx: from B_bar * x contribution
    dx = jnp.sum(dh_accum * B_bar, axis=-1)  # [B, S, D_INNER]

    # d_B: from B_bar = dt * B -> dh_accum * x * dt
    d_B = jnp.einsum('bsdn,bsd,bsd->bsn', dh_accum, x_f32, dt_f32)

    # d_dt: from A_bar and B_bar
    # d_A_bar contribution via h[t-1]
    h_prev = jnp.concatenate([jnp.zeros_like(h_f32[:, :1]), h_f32[:, :-1]], axis=1)
    d_A_bar = dh_accum * h_prev
    d_dt_from_A = jnp.sum(d_A_bar * A_bar * A[None, None, :, :], axis=-1)
    d_dt_from_B = jnp.sum(dh_accum * x_f32[..., None] * B_f32[:, :, None, :], axis=-1)
    d_dt = d_dt_from_A + d_dt_from_B  # [B, S, D_INNER]

    # d_A_log: chain through exp
    d_A_log = jnp.sum(d_A_bar * A_bar * dt_f32[..., None], axis=(0, 1)) * A

    return (
        to_bf16(dx),
        to_bf16(d_dt),
        to_bf16(d_A_log),
        to_bf16(d_B),
        to_bf16(d_C),
    )


def gate_multiply_backward(
    d_out: jnp.ndarray,
    y: jnp.ndarray,
    x_main: jnp.ndarray,
    D_param: jnp.ndarray,
    z: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for gate_multiply -> (dy, dx_main, d_D, dz)."""
    d_out_f32 = to_f32(d_out)
    y_f32 = to_f32(y)
    x_f32 = to_f32(x_main)
    D_f32 = to_f32(D_param)
    z_f32 = to_f32(z)

    sig_z = jax.nn.sigmoid(z_f32)
    silu_z = z_f32 * sig_z
    combined = y_f32 + x_f32 * D_f32

    # d(combined * silu_z)
    d_combined = d_out_f32 * silu_z
    d_silu_z = d_out_f32 * combined
    # silu grad: sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z)) = sigmoid(z)(1 + z(1-sigmoid(z)))
    d_z = d_silu_z * sig_z * (1.0 + z_f32 * (1.0 - sig_z))

    dy = d_combined
    dx_main = d_combined * D_f32
    d_D = jnp.sum(d_combined * x_f32, axis=(0, 1))  # [D_INNER]

    return to_bf16(dy), to_bf16(dx_main), to_bf16(d_D), to_bf16(d_z)


def out_proj_backward(dy: jnp.ndarray, y: jnp.ndarray,
                      w: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for out_proj -> (dy_in, dw)."""
    dy_f32 = to_f32(dy)
    dy_in = jnp.dot(dy_f32, to_f32(w))  # [B, S, D_INNER]
    dw = jnp.einsum('bsi,bsj->ij', dy_f32, to_f32(y))  # [D_MODEL, D_INNER]
    return to_bf16(dy_in), to_bf16(dw)


def lm_head_backward(d_logits: jnp.ndarray,
                     x: jnp.ndarray,
                     state: Mamba3State) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for lm_head -> (dx, d_head_w, d_head_b)."""
    d_logits_f32 = to_f32(d_logits)
    x_f32 = to_f32(x)
    w = state.tok_embed if state.tied_embeddings else state.lm_head_w
    dx = jnp.dot(d_logits_f32, to_f32(w))
    dw = jnp.einsum('bsi,bsj->ij', d_logits_f32, x_f32)
    db = jnp.sum(d_logits_f32, axis=(0, 1))
    return to_bf16(dx), to_bf16(dw), to_bf16(db)


# ---------------------------------------------------------------------------
# Single-layer forward / backward
# ---------------------------------------------------------------------------


def _layer_forward(
    lw: Mamba3LayerWeights, x: jnp.ndarray, dt_rank: int, d_state: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Forward pass for one Mamba-3 layer, returning output and saved activations."""
    residual = x

    # RMSNorm
    x_norm = rmsnorm_forward(x, lw.norm_g)

    # Input projection
    x_main, z = in_proj_forward(x_norm, lw.in_proj_w)

    # Conv1d + SiLU
    x_conv = conv1d_forward(x_main, lw.conv_w, lw.conv_b)

    # X projection -> dt_raw, B, C
    dt_raw, B_ssm, C_ssm = x_proj_forward(x_conv, lw.x_proj_w, dt_rank, d_state)

    # dt projection + softplus
    dt = dt_proj_forward(dt_raw, lw.dt_proj_w, lw.dt_proj_b)

    # Selective scan
    y, h_scan = selective_scan_forward(x_conv, dt, lw.A_log, B_ssm, C_ssm)

    # RoPE on hidden state (for next-step use in recurrent mode)
    h_roped = rope_apply_ssm(h_scan, lw.rope_freq)

    # Gate multiply
    gated = gate_multiply_forward(y, x_conv, lw.D_param, z)

    # Output projection + residual
    out = out_proj_forward(gated, lw.out_proj_w)
    out = residual_add(out, residual)

    saved = {
        'residual': residual,
        'x_norm': x_norm,
        'x_main': x_main,
        'z': z,
        'x_conv': x_conv,
        'dt_raw': dt_raw,
        'dt': dt,
        'B_ssm': B_ssm,
        'C_ssm': C_ssm,
        'y': y,
        'h_scan': h_scan,
        'h_roped': h_roped,
        'gated': gated,
    }
    return out, saved


def _layer_backward(
    lw: Mamba3LayerWeights, d_out: jnp.ndarray,
    saved: dict[str, jnp.ndarray], dt_rank: int, d_state: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Backward pass for one Mamba-3 layer."""
    # Residual path: d_residual = d_out
    d_residual = d_out

    # out_proj backward
    d_gated, d_out_proj_w = out_proj_backward(d_out, saved['gated'], lw.out_proj_w)

    # gate_multiply backward
    d_y, d_x_conv_gate, d_D, d_z = gate_multiply_backward(
        d_gated, saved['y'], saved['x_conv'], lw.D_param, saved['z']
    )

    # selective_scan backward
    d_x_scan, d_dt, d_A_log, d_B, d_C = selective_scan_backward(
        d_y, saved['x_conv'], saved['dt'], lw.A_log,
        saved['B_ssm'], saved['C_ssm'], saved['h_scan'],
    )

    # dt_proj backward
    d_dt_raw, d_dt_proj_w, d_dt_proj_b = dt_proj_backward(
        d_dt, saved['dt_raw'], lw.dt_proj_w, lw.dt_proj_b,
    )

    # x_proj backward
    d_x_xproj, d_x_proj_w = x_proj_backward(
        d_dt_raw, d_B, d_C, saved['x_conv'], lw.x_proj_w,
    )

    # Accumulate gradients flowing into x_conv
    d_x_conv_total = to_bf16(
        to_f32(d_x_conv_gate) + to_f32(d_x_scan) + to_f32(d_x_xproj)
    )

    # conv1d backward
    d_x_main, d_conv_w, d_conv_b = conv1d_backward(
        d_x_conv_total, saved['x_main'], lw.conv_w, lw.conv_b,
    )

    # in_proj backward
    d_x_norm, d_in_proj_w = in_proj_backward(
        d_x_main, d_z, saved['x_norm'], lw.in_proj_w,
    )

    # rmsnorm backward
    d_x, d_norm_g = rmsnorm_backward(d_x_norm, saved['residual'], lw.norm_g)

    # Add residual gradient
    d_x = to_bf16(to_f32(d_x) + to_f32(d_residual))

    # RoPE freq gradient (accumulated but not backpropped further here)
    _, d_rope_freq = rope_apply_ssm_backward(
        jnp.zeros_like(saved['h_roped']),  # placeholder; no downstream grad on h_roped in training
        saved['h_scan'],
        lw.rope_freq,
    )

    grads = {
        'norm_g': d_norm_g,
        'norm_b': jnp.zeros_like(lw.norm_b),  # RMSNorm has no bias grad from norm_b usage
        'in_proj_w': d_in_proj_w,
        'conv_w': d_conv_w,
        'conv_b': d_conv_b,
        'x_proj_w': d_x_proj_w,
        'dt_proj_w': d_dt_proj_w,
        'dt_proj_b': d_dt_proj_b,
        'A_log': d_A_log,
        'D_param': d_D,
        'out_proj_w': d_out_proj_w,
        'rope_freq': d_rope_freq,
    }
    return d_x, grads


# ---------------------------------------------------------------------------
# Full forward / backward
# ---------------------------------------------------------------------------


def mamba3_forward(
    state: Mamba3State,
    input_ids: jnp.ndarray,
    activations: dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Full Mamba-3 forward. Uses lax.scan over layers."""
    if activations is None:
        activations = {}

    x = embed_forward(state.tok_embed, input_ids)  # [B, S, D_MODEL]

    layer_saved_list: list[dict[str, jnp.ndarray]] = []

    def _scan_body(carry: jnp.ndarray, lw: Mamba3LayerWeights) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        out, saved = _layer_forward(lw, carry, state.dt_rank, state.d_state)
        return out, saved

    # Stack layer weights for lax.scan — each leaf becomes [N_LAYERS, ...]
    stacked_lw = jax.tree.map(lambda *arrs: jnp.stack(arrs), *state.layer_weights)
    x, all_saved = lax.scan(_scan_body, x, stacked_lw)

    # Final norm
    x = rmsnorm_forward(x, state.final_norm_g)

    # LM head
    logits = lm_head_forward(x, state)

    activations['pre_head'] = x
    activations['all_layer_saved'] = all_saved
    activations['input_ids'] = input_ids

    return logits, activations


def mamba3_backward(
    state: Mamba3State,
    d_logits: jnp.ndarray,
    activations: dict[str, Any],
) -> dict[str, Any]:
    """Full Mamba-3 backward."""
    # LM head backward
    dx, d_head_w, d_head_b = lm_head_backward(
        d_logits, activations['pre_head'], state,
    )

    # Final norm backward
    dx, d_final_norm_g = rmsnorm_backward(dx, activations['pre_head'], state.final_norm_g)

    # Backward through layers in reverse
    all_saved = activations['all_layer_saved']
    n_layers = len(state.layer_weights)

    def _scan_body_bwd(
        carry: jnp.ndarray,
        scan_inputs: tuple[Mamba3LayerWeights, dict[str, jnp.ndarray]],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        lw, saved = scan_inputs
        d_x, grads = _layer_backward(lw, carry, saved, state.dt_rank, state.d_state)
        return d_x, grads

    # Reverse the stacked arrays along axis 0 for backward traversal
    stacked_lw = jax.tree.map(lambda *arrs: jnp.stack(arrs), *state.layer_weights)
    stacked_lw_rev = jax.tree.map(lambda t: t[::-1], stacked_lw)
    all_saved_rev = jax.tree.map(lambda t: t[::-1], all_saved)

    dx, all_grads_rev = lax.scan(_scan_body_bwd, dx, (stacked_lw_rev, all_saved_rev))
    # Reverse grads back to original layer order
    all_grads = jax.tree.map(lambda t: t[::-1], all_grads_rev)

    # Embedding gradient
    input_ids = activations['input_ids']
    d_tok_embed = jnp.zeros_like(to_f32(state.tok_embed))
    d_tok_embed = d_tok_embed.at[input_ids].add(to_f32(dx))
    if state.tied_embeddings:
        d_tok_embed = d_tok_embed + to_f32(d_head_w)

    grads = {
        'tok_embed': to_bf16(d_tok_embed),
        'layer_grads': all_grads,
        'final_norm_g': d_final_norm_g,
        'final_norm_b': jnp.zeros_like(state.final_norm_b),
        'lm_head_w': to_bf16(d_head_w) if not state.tied_embeddings else None,
        'lm_head_b': to_bf16(d_head_b),
    }
    return grads
