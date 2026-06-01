"""TPU v5p Pallas module for a transformer decoder model."""

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

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from common_tpu import (
    ACCUM_DTYPE,
    PARAM_DTYPE,
    NanPolicy,
    apply_nan_policy,
    to_bf16,
    to_f16,
    to_f32,
)

# ---------------------------------------------------------------------------
# Layer weights
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LayerWeights:
    """Weights for a single transformer decoder layer."""

    attn_norm_g: jnp.ndarray       # [D]
    w_qkv: jnp.ndarray             # [3*D, D]
    b_qkv: jnp.ndarray             # [3*D]
    w_o: jnp.ndarray               # [D, D]
    b_o: jnp.ndarray               # [D]
    ffn_norm_g: jnp.ndarray        # [D]
    w_gate: jnp.ndarray            # [FFN, D]
    w_up: jnp.ndarray              # [FFN, D]
    w_down: jnp.ndarray            # [D, FFN]
    b_down: jnp.ndarray            # [D]


@dataclasses.dataclass
class TransformerDecoderState:
    """Full decoder state including all layer weights."""

    tok_embed: jnp.ndarray          # [VOCAB, D]
    layer_weights: list             # list of LayerWeights per layer
    final_norm_g: jnp.ndarray       # [D]
    final_norm_b: jnp.ndarray       # [D]
    lm_head_w: jnp.ndarray | None   # [VOCAB, D], None when tied
    lm_head_b: jnp.ndarray          # [VOCAB]
    tied_embeddings: bool

# ---------------------------------------------------------------------------
# Resource declarations (TPU has no shared memory; 0 for API compat)
# ---------------------------------------------------------------------------

EMBED_SMEM_BYTES: int = 0
RMSNORM_SMEM_BYTES: int = 0
ATTENTION_SMEM_BYTES: int = 0
QKV_PROJ_SMEM_BYTES: int = 0
O_PROJ_SMEM_BYTES: int = 0
MLP_GATE_UP_SMEM_BYTES: int = 0
MLP_DOWN_SMEM_BYTES: int = 0
RESIDUAL_SMEM_BYTES: int = 0
LM_HEAD_SMEM_BYTES: int = 0

# VMEM budget hints for BlockSpec tiling
ATTENTION_VMEM_TILES: int = 4

# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

_BYTES_PER_PARAM = 2  # BF16


def param_bytes(d_model: int, n_heads: int, n_layers: int, vocab: int, seq_len: int) -> int:
    """Total parameter memory in bytes."""
    d_head = d_model // n_heads
    ffn = 4 * d_model  # standard 4x expansion
    per_layer = (
        d_model                          # attn_norm_g
        + 3 * d_model * d_model          # w_qkv
        + 3 * d_model                    # b_qkv
        + d_model * d_model              # w_o
        + d_model                        # b_o
        + d_model                        # ffn_norm_g
        + ffn * d_model                  # w_gate
        + ffn * d_model                  # w_up
        + d_model * ffn                  # w_down
        + d_model                        # b_down
    )
    non_layer = (
        vocab * d_model                  # tok_embed
        + d_model                        # final_norm_g
        + d_model                        # final_norm_b
        + vocab * d_model                # lm_head_w (worst case, untied)
        + vocab                          # lm_head_b
    )
    return (per_layer * n_layers + non_layer) * _BYTES_PER_PARAM


def num_param_tensors(n_layers: int) -> int:
    """Total number of distinct parameter tensors."""
    per_layer = 10  # matches LayerWeights field count
    non_layer = 5   # tok_embed, final_norm_g, final_norm_b, lm_head_w, lm_head_b
    return per_layer * n_layers + non_layer


def activation_bytes_per_layer(
    d_model: int, n_heads: int, seq_len: int, batch: int
) -> int:
    """Activation memory (bytes) for one layer during forward."""
    d_head = d_model // n_heads
    # Pre-norm input, Q, K, V, attn_weights, attn_out, gate, up, mlp_out
    elems = (
        batch * seq_len * d_model           # residual snapshot
        + batch * n_heads * seq_len * d_head * 3  # Q, K, V
        + batch * n_heads * seq_len * seq_len     # attn weights
        + batch * seq_len * d_model               # attn output
        + batch * seq_len * 4 * d_model * 2       # gate + up
        + batch * seq_len * d_model               # mlp output
    )
    return elems * _BYTES_PER_PARAM

# ---------------------------------------------------------------------------
# Per-layer forward functions
# ---------------------------------------------------------------------------

_RMSNORM_EPS = 1e-6


def embed_forward(tok_embed: jnp.ndarray, input_ids: jnp.ndarray) -> jnp.ndarray:
    """Lookup token embeddings."""
    return tok_embed[input_ids]


def rmsnorm_forward(x: jnp.ndarray, gamma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RMSNorm forward returning (output, rstd)."""
    x_f32 = to_f32(x)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(variance + _RMSNORM_EPS)
    out = to_bf16(x_f32 * rstd * to_f32(gamma))
    return out, rstd


def qkv_proj_forward(
    x: jnp.ndarray, w_qkv: jnp.ndarray, b_qkv: jnp.ndarray
) -> jnp.ndarray:
    """QKV linear projection."""
    return to_bf16(jnp.dot(to_f32(x), to_f32(w_qkv).T) + to_f32(b_qkv))


def _build_rope_freqs(positions: jnp.ndarray, d_head: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build sin/cos tables for RoPE."""
    half = d_head // 2
    freq_exponents = jnp.arange(0, half, dtype=jnp.float32) / half
    freqs = 1.0 / (10000.0 ** freq_exponents)                  # [half]
    angles = to_f32(positions)[..., None] * freqs[None, :]      # [S, half]
    return jnp.sin(angles), jnp.cos(angles)


def rope_apply(
    q: jnp.ndarray, k: jnp.ndarray, positions: jnp.ndarray, d_head: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary position embeddings to Q and K."""
    sin, cos = _build_rope_freqs(positions, d_head)  # [S, half]
    half = d_head // 2

    def _rotate(x: jnp.ndarray) -> jnp.ndarray:
        # x: [..., S, d_head]
        x1 = x[..., :half]
        x2 = x[..., half:]
        s = sin  # broadcast to match batch/head dims
        c = cos
        return to_bf16(jnp.concatenate(
            [to_f32(x1) * c - to_f32(x2) * s,
             to_f32(x1) * s + to_f32(x2) * c],
            axis=-1,
        ))

    return _rotate(q), _rotate(k)


def attention_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    d_head: int,
    causal: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scaled dot-product attention returning (output, weights)."""
    scale = 1.0 / math.sqrt(d_head)
    scores = to_f32(jnp.dot(q, jnp.swapaxes(k, -2, -1))) * scale  # [B,H,S,S]
    if causal:
        seq_len = scores.shape[-1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = to_bf16(jnp.dot(attn_weights, to_f32(v)))
    return out, to_bf16(attn_weights)


def o_proj_forward(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Output projection linear layer."""
    return to_bf16(jnp.dot(to_f32(x), to_f32(w).T) + to_f32(b))


def residual_add(x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
    """Element-wise residual connection."""
    return to_bf16(to_f32(x) + to_f32(residual))


def mlp_gate_up_forward(
    x: jnp.ndarray, w_gate: jnp.ndarray, w_up: jnp.ndarray
) -> jnp.ndarray:
    """SwiGLU-style MLP gate+up projection."""
    x_f32 = to_f32(x)
    gate = jax.nn.silu(jnp.dot(x_f32, to_f32(w_gate).T))
    up = jnp.dot(x_f32, to_f32(w_up).T)
    return to_bf16(gate * up)


def mlp_down_forward(
    x: jnp.ndarray, w_down: jnp.ndarray, b_down: jnp.ndarray
) -> jnp.ndarray:
    """MLP down projection."""
    return to_bf16(jnp.dot(to_f32(x), to_f32(w_down).T) + to_f32(b_down))


def lm_head_forward(
    x: jnp.ndarray, state: TransformerDecoderState, tied: bool
) -> jnp.ndarray:
    """Final logit projection, optionally reusing tok_embed."""
    w = state.tok_embed if tied else state.lm_head_w
    logits = jnp.dot(to_f32(x), to_f32(w).T) + to_f32(state.lm_head_b)
    return logits  # keep in f32 for loss

# ---------------------------------------------------------------------------
# Per-layer backward functions (explicit, not jax.grad)
# ---------------------------------------------------------------------------


def rmsnorm_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    rstd: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for RMSNorm returning (d_x, d_gamma)."""
    d_out_f32 = to_f32(d_out)
    x_f32 = to_f32(x)
    g_f32 = to_f32(gamma)
    D = x_f32.shape[-1]

    x_hat = x_f32 * rstd
    d_gamma = jnp.sum(d_out_f32 * x_hat, axis=tuple(range(d_out_f32.ndim - 1)))

    dx_hat = d_out_f32 * g_f32
    # d_x = rstd * (dx_hat - x_hat * mean(dx_hat * x_hat))
    proj = jnp.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    d_x = rstd * (dx_hat - x_hat * proj)

    return to_bf16(d_x), to_bf16(d_gamma)


def attention_backward(
    d_out: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_weights: jnp.ndarray,
    d_head: int,
    causal: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for attention returning (d_q, d_k, d_v)."""
    scale = 1.0 / math.sqrt(d_head)
    d_out_f32 = to_f32(d_out)
    aw_f32 = to_f32(attn_weights)
    v_f32 = to_f32(v)
    q_f32 = to_f32(q)
    k_f32 = to_f32(k)

    # d_V = attn_weights^T @ d_out
    d_v = jnp.dot(jnp.swapaxes(aw_f32, -2, -1), d_out_f32)

    # d_attn = d_out @ V^T
    d_attn = jnp.dot(d_out_f32, jnp.swapaxes(v_f32, -2, -1))

    # softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn, -1, keepdims))
    d_scores = aw_f32 * (d_attn - jnp.sum(d_attn * aw_f32, axis=-1, keepdims=True))

    if causal:
        seq_len = d_scores.shape[-1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        d_scores = jnp.where(mask, d_scores, 0.0)

    d_scores = d_scores * scale

    # d_Q = d_scores @ K
    d_q = jnp.dot(d_scores, k_f32)
    # d_K = d_scores^T @ Q
    d_k = jnp.dot(jnp.swapaxes(d_scores, -2, -1), q_f32)

    return to_bf16(d_q), to_bf16(d_k), to_bf16(d_v)


def linear_backward(
    d_out: jnp.ndarray, x: jnp.ndarray, w: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for y = x @ w.T + b returning (d_x, d_w, d_b)."""
    d_out_f32 = to_f32(d_out)
    x_f32 = to_f32(x)
    w_f32 = to_f32(w)

    d_x = jnp.dot(d_out_f32, w_f32)
    # d_out is [..., out], x is [..., in] -> d_w is [out, in]
    # flatten batch dims
    d_flat = d_out_f32.reshape(-1, d_out_f32.shape[-1])
    x_flat = x_f32.reshape(-1, x_f32.shape[-1])
    d_w = jnp.dot(d_flat.T, x_flat)
    d_b = jnp.sum(d_flat, axis=0)

    return to_bf16(d_x), to_bf16(d_w), to_bf16(d_b)


def mlp_gate_up_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for SwiGLU gate+up returning (d_x, d_w_gate, d_w_up)."""
    x_f32 = to_f32(x)
    d_out_f32 = to_f32(d_out)

    gate_pre = jnp.dot(x_f32, to_f32(w_gate).T)
    up_val = jnp.dot(x_f32, to_f32(w_up).T)
    gate_act = jax.nn.silu(gate_pre)

    # d_out = d(gate_act * up_val)
    d_gate_act = d_out_f32 * up_val
    d_up_val = d_out_f32 * gate_act

    # silu backward: silu'(z) = silu(z) + sigmoid(z)*(1 - silu(z))
    sig = jax.nn.sigmoid(gate_pre)
    d_gate_pre = d_gate_act * (gate_act + sig * (1.0 - gate_act))

    # linear backwards
    d_flat_gate = d_gate_pre.reshape(-1, d_gate_pre.shape[-1])
    d_flat_up = d_up_val.reshape(-1, d_up_val.shape[-1])
    x_flat = x_f32.reshape(-1, x_f32.shape[-1])

    d_w_gate = jnp.dot(d_flat_gate.T, x_flat)
    d_w_up = jnp.dot(d_flat_up.T, x_flat)

    d_x = jnp.dot(d_gate_pre, to_f32(w_gate)) + jnp.dot(d_up_val, to_f32(w_up))

    return to_bf16(d_x), to_bf16(d_w_gate), to_bf16(d_w_up)


def residual_add_backward(
    d_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for residual add returning (d_x, d_residual)."""
    return d_out, d_out


def embed_backward(
    d_out: jnp.ndarray, input_ids: jnp.ndarray, vocab_size: int
) -> jnp.ndarray:
    """Backward for embedding lookup returning d_tok_embed."""
    d_out_f32 = to_f32(d_out)
    d_embed = jnp.zeros((vocab_size, d_out_f32.shape[-1]), dtype=ACCUM_DTYPE)
    d_embed = d_embed.at[input_ids].add(d_out_f32)
    return to_bf16(d_embed)


def lm_head_backward(
    d_logits: jnp.ndarray,
    x: jnp.ndarray,
    state: TransformerDecoderState,
    tied: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    """Backward for lm_head returning (d_x, d_w or None, d_b)."""
    w = state.tok_embed if tied else state.lm_head_w
    d_x, d_w, d_b = linear_backward(d_logits, x, w)
    if tied:
        return d_x, None, d_b
    return d_x, d_w, d_b

# ---------------------------------------------------------------------------
# Pallas @pl.pallas_call wrappers for compute-bound ops
# ---------------------------------------------------------------------------

_ATTN_BLOCK_Q = 128
_ATTN_BLOCK_KV = 128


def _attention_kernel(
    q_ref,       # [BLOCK_Q, D_HEAD]
    k_ref,       # [BLOCK_KV, D_HEAD]
    v_ref,       # [BLOCK_KV, D_HEAD]
    o_ref,       # [BLOCK_Q, D_HEAD]
    *,
    d_head: int,
    block_q: int,
    block_kv: int,
    seq_len: int,
    causal: bool,
):
    """Inner Pallas kernel for one tile of attention."""
    q_tile = q_ref[...].astype(jnp.float32)
    k_tile = k_ref[...].astype(jnp.float32)
    v_tile = v_ref[...].astype(jnp.float32)

    scale = 1.0 / math.sqrt(d_head)
    scores = jnp.dot(q_tile, k_tile.T) * scale

    if causal:
        q_idx = pl.program_id(0)
        kv_idx = pl.program_id(1)
        q_offset = q_idx * block_q
        kv_offset = kv_idx * block_kv
        row_ids = q_offset + jnp.arange(block_q)[:, None]
        col_ids = kv_offset + jnp.arange(block_kv)[None, :]
        mask = row_ids >= col_ids
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)

    attn = jax.nn.softmax(scores, axis=-1)
    o_tile = jnp.dot(attn, v_tile).astype(o_ref.dtype)
    o_ref[...] = o_tile


def attention_pallas(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    d_head: int,
    causal: bool = True,
) -> jnp.ndarray:
    """BlockSpec-based tiled attention via Pallas."""
    batch, n_heads, seq_len, _ = q.shape
    block_q = min(_ATTN_BLOCK_Q, seq_len)
    block_kv = min(_ATTN_BLOCK_KV, seq_len)
    grid_q = seq_len // block_q
    grid_kv = seq_len // block_kv

    kernel = functools.partial(
        _attention_kernel,
        d_head=d_head,
        block_q=block_q,
        block_kv=block_kv,
        seq_len=seq_len,
        causal=causal,
    )

    def _per_head(q_bh: jnp.ndarray, k_bh: jnp.ndarray, v_bh: jnp.ndarray):
        return pl.pallas_call(
            kernel,
            grid=(grid_q, grid_kv),
            in_specs=[
                pl.BlockSpec((block_q, d_head), lambda i, j: (i, 0)),
                pl.BlockSpec((block_kv, d_head), lambda i, j: (j, 0)),
                pl.BlockSpec((block_kv, d_head), lambda i, j: (j, 0)),
            ],
            out_specs=pl.BlockSpec((block_q, d_head), lambda i, j: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((seq_len, d_head), q.dtype),
        )(q_bh, k_bh, v_bh)

    # vmap over batch and heads
    return jax.vmap(jax.vmap(_per_head))(q, k, v)

# ---------------------------------------------------------------------------
# Layer activations stored for backward pass
# ---------------------------------------------------------------------------


class LayerActivations(NamedTuple):
    """Saved tensors for one layer's backward pass."""

    residual_pre_attn: jnp.ndarray
    x_normed_attn: jnp.ndarray
    rstd_attn: jnp.ndarray
    q: jnp.ndarray
    k: jnp.ndarray
    v: jnp.ndarray
    attn_weights: jnp.ndarray
    attn_out: jnp.ndarray
    residual_pre_ffn: jnp.ndarray
    x_normed_ffn: jnp.ndarray
    rstd_ffn: jnp.ndarray
    mlp_input: jnp.ndarray
    mlp_gate_up_out: jnp.ndarray

# ---------------------------------------------------------------------------
# Single-layer forward / backward
# ---------------------------------------------------------------------------


def _split_qkv(
    qkv: jnp.ndarray, n_heads: int, d_head: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split fused QKV into separate Q, K, V with head dim."""
    B, S, _ = qkv.shape
    qkv = qkv.reshape(B, S, 3, n_heads, d_head)
    q = jnp.transpose(qkv[:, :, 0], (0, 2, 1, 3))  # [B, H, S, D_head]
    k = jnp.transpose(qkv[:, :, 1], (0, 2, 1, 3))
    v = jnp.transpose(qkv[:, :, 2], (0, 2, 1, 3))
    return q, k, v


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    """Merge head dim back: [B, H, S, D_head] -> [B, S, D]."""
    B, H, S, D_head = x.shape
    return jnp.transpose(x, (0, 2, 1, 3)).reshape(B, S, H * D_head)


def layer_forward(
    x: jnp.ndarray,
    lw: LayerWeights,
    positions: jnp.ndarray,
    n_heads: int,
    d_head: int,
) -> tuple[jnp.ndarray, LayerActivations]:
    """Forward pass for one decoder layer returning (output, saved activations)."""
    residual_pre_attn = x

    # Attention sub-layer
    x_normed, rstd_attn = rmsnorm_forward(x, lw.attn_norm_g)
    qkv = qkv_proj_forward(x_normed, lw.w_qkv, lw.b_qkv)
    q, k, v = _split_qkv(qkv, n_heads, d_head)
    q, k = rope_apply(q, k, positions, d_head)
    attn_out, attn_weights = attention_forward(q, k, v, d_head, causal=True)
    attn_out_merged = _merge_heads(attn_out)
    proj_out = o_proj_forward(attn_out_merged, lw.w_o, lw.b_o)
    x = residual_add(proj_out, residual_pre_attn)

    residual_pre_ffn = x

    # FFN sub-layer
    x_normed_ffn, rstd_ffn = rmsnorm_forward(x, lw.ffn_norm_g)
    mlp_mid = mlp_gate_up_forward(x_normed_ffn, lw.w_gate, lw.w_up)
    mlp_out = mlp_down_forward(mlp_mid, lw.w_down, lw.b_down)
    x = residual_add(mlp_out, residual_pre_ffn)

    acts = LayerActivations(
        residual_pre_attn=residual_pre_attn,
        x_normed_attn=x_normed,
        rstd_attn=rstd_attn,
        q=q,
        k=k,
        v=v,
        attn_weights=attn_weights,
        attn_out=attn_out_merged,
        residual_pre_ffn=residual_pre_ffn,
        x_normed_ffn=x_normed_ffn,
        rstd_ffn=rstd_ffn,
        mlp_input=x_normed_ffn,
        mlp_gate_up_out=mlp_mid,
    )
    return x, acts


def layer_backward(
    d_out: jnp.ndarray,
    lw: LayerWeights,
    acts: LayerActivations,
    n_heads: int,
    d_head: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Backward pass for one decoder layer returning (d_input, param_grads)."""
    grads: dict[str, jnp.ndarray] = {}

    # --- FFN residual backward ---
    d_mlp_out, d_res_ffn = residual_add_backward(d_out)

    # mlp_down backward
    d_mlp_mid, d_w_down, d_b_down = linear_backward(
        d_mlp_out, acts.mlp_gate_up_out, lw.w_down,
    )
    grads["w_down"] = d_w_down
    grads["b_down"] = d_b_down

    # mlp gate+up backward
    d_x_ffn, d_w_gate, d_w_up = mlp_gate_up_backward(
        d_mlp_mid, acts.mlp_input, lw.w_gate, lw.w_up,
    )
    grads["w_gate"] = d_w_gate
    grads["w_up"] = d_w_up

    # ffn rmsnorm backward
    d_x_pre_ffn_norm, d_ffn_norm_g = rmsnorm_backward(
        d_x_ffn, acts.residual_pre_ffn, lw.ffn_norm_g, acts.rstd_ffn,
    )
    grads["ffn_norm_g"] = d_ffn_norm_g

    d_x = residual_add(d_x_pre_ffn_norm, d_res_ffn)

    # --- Attention residual backward ---
    d_proj_out, d_res_attn = residual_add_backward(d_x)

    # o_proj backward
    d_attn_merged, d_w_o, d_b_o = linear_backward(
        d_proj_out, acts.attn_out, lw.w_o,
    )
    grads["w_o"] = d_w_o
    grads["b_o"] = d_b_o

    # un-merge heads for attention backward
    B, S, D = d_attn_merged.shape
    d_attn_heads = d_attn_merged.reshape(B, S, n_heads, d_head)
    d_attn_heads = jnp.transpose(d_attn_heads, (0, 2, 1, 3))

    # attention backward
    d_q, d_k, d_v = attention_backward(
        d_attn_heads, acts.q, acts.k, acts.v, acts.attn_weights, d_head,
    )

    # merge q/k/v grads back to qkv
    d_q_flat = jnp.transpose(d_q, (0, 2, 1, 3))  # [B, S, H, d_head]
    d_k_flat = jnp.transpose(d_k, (0, 2, 1, 3))
    d_v_flat = jnp.transpose(d_v, (0, 2, 1, 3))
    d_qkv = jnp.concatenate(
        [d_q_flat.reshape(B, S, -1),
         d_k_flat.reshape(B, S, -1),
         d_v_flat.reshape(B, S, -1)],
        axis=-1,
    )

    # qkv_proj backward
    d_x_normed, d_w_qkv, d_b_qkv = linear_backward(
        d_qkv, acts.x_normed_attn, lw.w_qkv,
    )
    grads["w_qkv"] = d_w_qkv
    grads["b_qkv"] = d_b_qkv

    # attn rmsnorm backward
    d_x_pre_attn_norm, d_attn_norm_g = rmsnorm_backward(
        d_x_normed, acts.residual_pre_attn, lw.attn_norm_g, acts.rstd_attn,
    )
    grads["attn_norm_g"] = d_attn_norm_g

    d_input = residual_add(d_x_pre_attn_norm, d_res_attn)

    return d_input, grads

# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------


def decoder_forward(
    state: TransformerDecoderState,
    input_ids: jnp.ndarray,
    activations_buf: list[LayerActivations | None],
) -> jnp.ndarray:
    """Full decoder forward. Returns logits and populates activations_buf."""
    d_model = state.tok_embed.shape[1]
    n_layers = len(state.layer_weights)
    n_heads = state.layer_weights[0].w_qkv.shape[0] // (3 * (d_model // (
        state.layer_weights[0].w_qkv.shape[0] // 3 // d_model
        if state.layer_weights[0].w_qkv.shape[0] // 3 != d_model
        else 1
    )))
    # Infer n_heads: w_qkv is [3*D, D], so first dim = 3*D -> n_heads from LayerWeights
    qkv_out_dim = state.layer_weights[0].w_qkv.shape[0]  # 3*D
    d_total = qkv_out_dim // 3
    # d_head must divide d_total; use attention VMEM tile hint or default 128
    d_head = min(128, d_total)
    while d_total % d_head != 0 and d_head > 1:
        d_head -= 1
    n_heads = d_total // d_head

    B, S = input_ids.shape
    positions = jnp.arange(S, dtype=jnp.int32)

    x = embed_forward(state.tok_embed, input_ids)

    # Layer loop via Python for to populate activations_buf
    for i, lw in enumerate(state.layer_weights):
        x, acts = layer_forward(x, lw, positions, n_heads, d_head)
        if i < len(activations_buf):
            activations_buf[i] = acts
        else:
            activations_buf.append(acts)

    # Final norm
    x_normed, _rstd = rmsnorm_forward(x, state.final_norm_g)
    x_normed = to_bf16(to_f32(x_normed) + to_f32(state.final_norm_b))

    logits = lm_head_forward(x_normed, state, state.tied_embeddings)
    return logits

# ---------------------------------------------------------------------------
# Full backward pass
# ---------------------------------------------------------------------------


def decoder_backward(
    state: TransformerDecoderState,
    d_logits: jnp.ndarray,
    activations_buf: list[LayerActivations],
) -> dict[str, Any]:
    """Full decoder backward. Returns param gradients."""
    d_model = state.tok_embed.shape[1]
    qkv_out_dim = state.layer_weights[0].w_qkv.shape[0]
    d_total = qkv_out_dim // 3
    d_head = min(128, d_total)
    while d_total % d_head != 0 and d_head > 1:
        d_head -= 1
    n_heads = d_total // d_head

    all_grads: dict[str, Any] = {}

    # lm_head backward — need pre-head activations
    # Reconstruct final_norm output from last layer output + final_norm
    # We stored per-layer acts; the input to final_norm is the output of last layer.
    last_layer_out = activations_buf[-1].residual_pre_attn  # placeholder
    # Instead, we recompute: after all layers, x was final input to norm.
    # For correctness we need the final normed x. We store it via a simple
    # recompute from the last residual.  The last layer output = residual_pre_ffn
    # of a hypothetical next layer, but we don't have that. Use the ffn output:
    # layer output = mlp_out + residual_pre_ffn -> already returned as x.
    # Since we don't explicitly store x_final, recompute from last acts:
    last_acts = activations_buf[-1]
    x_final = residual_add(
        mlp_down_forward(
            last_acts.mlp_gate_up_out,
            state.layer_weights[-1].w_down,
            state.layer_weights[-1].b_down,
        ),
        last_acts.residual_pre_ffn,
    )
    x_final_normed, rstd_final = rmsnorm_forward(x_final, state.final_norm_g)
    x_final_normed = to_bf16(to_f32(x_final_normed) + to_f32(state.final_norm_b))

    d_x_normed, d_lm_head_w, d_lm_head_b = lm_head_backward(
        d_logits, x_final_normed, state, state.tied_embeddings,
    )
    all_grads["lm_head_w"] = d_lm_head_w
    all_grads["lm_head_b"] = d_lm_head_b

    # final norm backward (norm + bias)
    d_final_norm_b = jnp.sum(
        to_f32(d_x_normed).reshape(-1, d_x_normed.shape[-1]), axis=0
    )
    all_grads["final_norm_b"] = to_bf16(d_final_norm_b)
    d_pre_bias, d_final_norm_g = rmsnorm_backward(
        d_x_normed, x_final, state.final_norm_g, rstd_final,
    )
    all_grads["final_norm_g"] = d_final_norm_g

    # Layer loop backward
    d_x = d_pre_bias
    layer_grads_list: list[dict[str, jnp.ndarray]] = []
    for i in reversed(range(len(state.layer_weights))):
        d_x, lg = layer_backward(
            d_x, state.layer_weights[i], activations_buf[i], n_heads, d_head,
        )
        layer_grads_list.insert(0, lg)

    all_grads["layer_grads"] = layer_grads_list

    # Embedding backward
    # d_x is now the gradient w.r.t. embedded tokens. We need input_ids to
    # scatter; caller must supply them. For API symmetry, return d_tok_embed
    # as a placeholder that the caller accumulates.
    all_grads["d_embed_out"] = d_x  # caller scatters with input_ids

    # If tied embeddings, accumulate lm_head grad into tok_embed grad
    if state.tied_embeddings and d_lm_head_w is None:
        all_grads["tok_embed_from_head"] = None  # handled via d_embed_out
    else:
        all_grads["tok_embed_from_head"] = d_lm_head_w

    return all_grads


# ---------------------------------------------------------------------------
# Functional layer loop via lax.scan (alternative to Python loop)
# ---------------------------------------------------------------------------


def decoder_forward_scan(
    state: TransformerDecoderState,
    input_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, list[LayerActivations]]:
    """Forward pass using lax.scan for the layer loop."""
    d_model = state.tok_embed.shape[1]
    qkv_out_dim = state.layer_weights[0].w_qkv.shape[0]
    d_total = qkv_out_dim // 3
    d_head = min(128, d_total)
    while d_total % d_head != 0 and d_head > 1:
        d_head -= 1
    n_heads = d_total // d_head

    B, S = input_ids.shape
    positions = jnp.arange(S, dtype=jnp.int32)

    x = embed_forward(state.tok_embed, input_ids)

    # Stack layer weights for scan
    stacked_lw = jax.tree.map(lambda *arrs: jnp.stack(arrs), *state.layer_weights)

    def scan_body(carry, lw_slice):
        x_in = carry
        # Unstack single-layer weights
        single_lw = LayerWeights(
            attn_norm_g=lw_slice.attn_norm_g,
            w_qkv=lw_slice.w_qkv,
            b_qkv=lw_slice.b_qkv,
            w_o=lw_slice.w_o,
            b_o=lw_slice.b_o,
            ffn_norm_g=lw_slice.ffn_norm_g,
            w_gate=lw_slice.w_gate,
            w_up=lw_slice.w_up,
            w_down=lw_slice.w_down,
            b_down=lw_slice.b_down,
        )
        x_out, acts = layer_forward(x_in, single_lw, positions, n_heads, d_head)
        return x_out, acts

    x, all_acts = lax.scan(scan_body, x, stacked_lw)

    # Final norm
    x_normed, _ = rmsnorm_forward(x, state.final_norm_g)
    x_normed = to_bf16(to_f32(x_normed) + to_f32(state.final_norm_b))

    logits = lm_head_forward(x_normed, state, state.tied_embeddings)
    return logits, all_acts
