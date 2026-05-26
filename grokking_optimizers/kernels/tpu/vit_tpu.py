"""TPU v5p Pallas module for a Vision Transformer (ViT).

Follows the same contract as the CUDA/HIP headers but in Python/JAX.
Uses jnp.dot for MXU matmuls, lax.scan for layer loops, BF16 params
with FP32 accumulators for norms and softmax, and non-causal attention
with learned position embeddings.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from grokking_optimizers.kernels.tpu.common_tpu import (
    ACCUM_DTYPE,
    PARAM_DTYPE,
    NanPolicy,
    apply_nan_policy,
    to_bf16,
    to_f16,
    to_f32,
)

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

D_MODEL: int = 128
N_HEADS: int = 4
D_HEAD: int = D_MODEL // N_HEADS  # 32
FFN_HIDDEN: int = 4 * D_MODEL     # 512
SEQ_LEN: int = 17                 # 16 patches + 1 CLS token

# ---------------------------------------------------------------------------
# Resource declarations (API compat with CUDA/HIP headers)
# ---------------------------------------------------------------------------

ATTENTION_VMEM_TILES: int = 2   # small attention fits in fewer tiles
PATCH_EMBED_SMEM_BYTES: int = 0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ViTLayerWeights:
    """Weights for a single ViT encoder layer."""

    qkv_w: jnp.ndarray     # [3*D, D]
    qkv_b: jnp.ndarray     # [3*D]
    out_w: jnp.ndarray      # [D, D]
    out_b: jnp.ndarray      # [D]
    norm1_g: jnp.ndarray    # [D]
    norm1_b: jnp.ndarray    # [D]
    gate_w: jnp.ndarray     # [FFN_HIDDEN, D]
    up_w: jnp.ndarray       # [FFN_HIDDEN, D]
    down_w: jnp.ndarray     # [D, FFN_HIDDEN]
    down_b: jnp.ndarray     # [D]
    norm2_g: jnp.ndarray    # [D]
    norm2_b: jnp.ndarray    # [D]


@dataclasses.dataclass
class ViTState:
    """Full model state for a Vision Transformer."""

    patch_proj_w: jnp.ndarray   # [D, PATCH_DIM]
    patch_proj_b: jnp.ndarray   # [D]
    cls_token: jnp.ndarray      # [D]
    pos_embed: jnp.ndarray      # [SEQ_LEN, D]
    layer_weights: list         # list of ViTLayerWeights
    final_norm_g: jnp.ndarray   # [D]
    final_norm_b: jnp.ndarray   # [D]
    head_w: jnp.ndarray         # [N_CLASSES, D]
    head_b: jnp.ndarray         # [N_CLASSES]
    d_model: int
    n_heads: int
    n_classes: int
    tied_embeddings: bool


# Register dataclasses as JAX pytrees so jax.tree.map can traverse their fields.
_LAYER_FIELDS = [f.name for f in dataclasses.fields(ViTLayerWeights)]


def _layer_flatten(lw: ViTLayerWeights) -> tuple[list, None]:
    return [getattr(lw, f) for f in _LAYER_FIELDS], None


def _layer_unflatten(_aux: None, children: list) -> ViTLayerWeights:
    return ViTLayerWeights(**dict(zip(_LAYER_FIELDS, children)))


jax.tree_util.register_pytree_node(ViTLayerWeights, _layer_flatten, _layer_unflatten)

# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


def param_bytes(
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_classes: int,
    patch_dim: int,
    seq_len: int = SEQ_LEN,
    param_dtype: jnp.dtype = PARAM_DTYPE,
) -> int:
    """Total parameter memory in bytes."""
    bytes_per = jnp.dtype(param_dtype).itemsize
    ffn_hidden = 4 * d_model

    # patch_proj_w + patch_proj_b
    total = d_model * patch_dim + d_model
    # cls_token
    total += d_model
    # pos_embed
    total += seq_len * d_model
    # per-layer
    per_layer = (
        3 * d_model * d_model   # qkv_w
        + 3 * d_model           # qkv_b
        + d_model * d_model     # out_w
        + d_model               # out_b
        + d_model               # norm1_g
        + d_model               # norm1_b
        + ffn_hidden * d_model  # gate_w
        + ffn_hidden * d_model  # up_w
        + d_model * ffn_hidden  # down_w
        + d_model               # down_b
        + d_model               # norm2_g
        + d_model               # norm2_b
    )
    total += n_layers * per_layer
    # final_norm_g + final_norm_b
    total += d_model + d_model
    # head_w + head_b
    total += n_classes * d_model + n_classes

    return total * bytes_per


def num_param_tensors(n_layers: int) -> int:
    """Number of distinct weight tensors across the full model."""
    # patch_proj_w(1) + patch_proj_b(1) + cls_token(1) + pos_embed(1)
    # + per-layer(12 each)
    # + final_norm_g(1) + final_norm_b(1) + head_w(1) + head_b(1)
    return 4 + n_layers * 12 + 4


def activation_bytes_per_layer(
    batch: int,
    seq_len: int,
    d_model: int,
) -> int:
    """Activation scratch memory per layer in bytes (FP32 accumulators)."""
    ffn_hidden = 4 * d_model
    tokens = batch * seq_len
    # saved_input + post_norm + qkv + attn_out + o_proj + gate_pre + up + mlp_down
    floats = (
        tokens * d_model        # saved input for backward
        + tokens * d_model      # post-norm
        + tokens * 3 * d_model  # qkv
        + tokens * d_model      # attention output
        + tokens * d_model      # o_proj output
        + tokens * ffn_hidden   # gate pre-activation (saved for SiLU backward)
        + tokens * ffn_hidden   # up projection
        + tokens * d_model      # mlp_down output
    )
    return floats * 4  # fp32

# ---------------------------------------------------------------------------
# Forward pass — per-layer functions
# ---------------------------------------------------------------------------


def patch_embed_forward(
    patches: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Linear projection of image patches: [B, N_PATCHES, PATCH_DIM] -> [B, N_PATCHES, D]."""
    return jnp.dot(patches, w.T) + b


def pos_embed_add_forward(
    x: jnp.ndarray,
    cls_token: jnp.ndarray,
    pos_embed: jnp.ndarray,
) -> jnp.ndarray:
    """Prepend CLS token and add learned positional embeddings."""
    batch = x.shape[0]
    # Expand cls_token [D] -> [B, 1, D]
    cls_expanded = jnp.broadcast_to(cls_token[None, None, :], (batch, 1, x.shape[-1]))
    # Concatenate: [B, 1, D] ; [B, N_PATCHES, D] -> [B, SEQ_LEN, D]
    h = jnp.concatenate([cls_expanded, x], axis=1)
    # Add positional embeddings [SEQ_LEN, D] broadcast over batch
    return h + pos_embed[None, :, :]


def rmsnorm_forward(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """RMSNorm: y = x / rms(x) * gamma."""
    x_f32 = to_f32(x)
    ms = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    rms_inv = jax.lax.rsqrt(ms + eps)
    return to_bf16(x_f32 * rms_inv * to_f32(gamma))


def qkv_proj_forward(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    n_heads: int = N_HEADS,
) -> jnp.ndarray:
    """QKV linear projection reshaped for multi-head attention."""
    batch, seq, d = x.shape
    d_head = d // n_heads
    # Linear: [B, S, D] @ [D, 3*D] -> [B, S, 3*D]
    qkv = jnp.dot(x, w.T) + b
    # Reshape to [B, S, 3, H, D_H]
    return qkv.reshape(batch, seq, 3, n_heads, d_head)


def attention_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    d_head: int = D_HEAD,
    causal: bool = False,
) -> jnp.ndarray:
    """Scaled dot-product attention (NON-CAUSAL by default for ViT)."""
    # q, k, v: [B, H, S, D_H]
    scale = 1.0 / math.sqrt(d_head)
    # Attention scores: [B, H, S, S] — use matmul for batched dims
    attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9), k=1)
        attn_weights = attn_weights + mask

    # Softmax in FP32 for numerical stability
    attn_weights = to_f32(attn_weights)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    attn_weights = to_bf16(attn_weights)

    # Weighted sum: [B, H, S, D_H]
    return jnp.matmul(attn_weights, v)


def o_proj_forward(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Output projection after attention: linear."""
    return jnp.dot(x, w.T) + b


def residual_add(x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
    """Elementwise residual addition."""
    return x + residual


def mlp_gate_up_forward(
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
) -> jnp.ndarray:
    """Gated MLP: silu(x @ w_gate.T) * (x @ w_up.T)."""
    gate = jax.nn.silu(jnp.dot(x, w_gate.T))
    up = jnp.dot(x, w_up.T)
    return gate * up


def mlp_down_forward(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Down projection in MLP: linear."""
    return jnp.dot(x, w.T) + b


def classifier_head_forward(
    h: jnp.ndarray,
    state: ViTState,
) -> jnp.ndarray:
    """Extract CLS token, apply final norm, and project to logits."""
    # h: [B, SEQ_LEN, D] -> cls: [B, D]
    cls_hidden = h[:, 0, :]
    # Final layernorm (using RMSNorm + bias shift)
    cls_f32 = to_f32(cls_hidden)
    ms = jnp.mean(cls_f32 ** 2, axis=-1, keepdims=True)
    rms_inv = jax.lax.rsqrt(ms + 1e-6)
    normed = cls_f32 * rms_inv * to_f32(state.final_norm_g) + to_f32(state.final_norm_b)
    normed = to_bf16(normed)
    # Linear head: [B, D] @ [D, N_CLASSES] -> [B, N_CLASSES]
    if state.tied_embeddings:
        logits = jnp.dot(normed, state.patch_proj_w) + state.head_b
    else:
        logits = jnp.dot(normed, state.head_w.T) + state.head_b
    return logits

# ---------------------------------------------------------------------------
# Single transformer layer forward
# ---------------------------------------------------------------------------


def _vit_layer_forward(
    h: jnp.ndarray,
    layer: ViTLayerWeights,
    n_heads: int,
) -> jnp.ndarray:
    """Forward pass through one ViT encoder layer."""
    d_model = h.shape[-1]
    d_head = d_model // n_heads

    # --- Self-attention block ---
    residual = h
    # QKV projection: [B, S, D] -> [B, S, 3, H, D_H]
    qkv = qkv_proj_forward(h, layer.qkv_w, layer.qkv_b, n_heads)
    q = qkv[:, :, 0, :, :]  # [B, S, H, D_H]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]
    # Transpose to [B, H, S, D_H] for attention
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    # Non-causal attention
    attn_out = attention_forward(q, k, v, d_head, causal=False)
    # Transpose back: [B, H, S, D_H] -> [B, S, H, D_H] -> [B, S, D]
    attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], d_model)
    # Output projection
    attn_out = o_proj_forward(attn_out, layer.out_w, layer.out_b)
    # Residual + RMSNorm
    h = rmsnorm_forward(residual_add(attn_out, residual), layer.norm1_g)

    # --- MLP block ---
    residual = h
    mlp_out = mlp_gate_up_forward(h, layer.gate_w, layer.up_w)
    mlp_out = mlp_down_forward(mlp_out, layer.down_w, layer.down_b)
    # Residual + RMSNorm
    h = rmsnorm_forward(residual_add(mlp_out, residual), layer.norm2_g)

    return h

# ---------------------------------------------------------------------------
# Backward pass — per-layer functions
# ---------------------------------------------------------------------------


def rmsnorm_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    eps: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward pass for RMSNorm. Returns (d_x, d_gamma)."""
    x_f32 = to_f32(x)
    d_out_f32 = to_f32(d_out)
    gamma_f32 = to_f32(gamma)

    ms = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    rms_inv = jax.lax.rsqrt(ms + eps)

    d_model = x_f32.shape[-1]
    # Normalized input
    x_hat = x_f32 * rms_inv
    # d_gamma: sum over batch and seq dimensions
    d_gamma = jnp.sum(d_out_f32 * x_hat, axis=tuple(range(d_out_f32.ndim - 1)))

    # d_x
    dot = jnp.sum(d_out_f32 * x_f32 * gamma_f32, axis=-1, keepdims=True)
    coeff = rms_inv ** 3 / d_model
    d_x = rms_inv * gamma_f32 * d_out_f32 - coeff * x_f32 * dot

    return to_bf16(d_x), to_bf16(d_gamma)


def patch_embed_backward(
    d_out: jnp.ndarray,
    patches: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for patch embedding. Returns (d_patches, d_w, d_b)."""
    d_out_f32 = to_f32(d_out)
    # d_patches = d_out @ w
    d_patches = jnp.dot(d_out_f32, to_f32(w))
    # d_w = d_out^T @ patches  (summed over batch)
    d_w = jnp.einsum('bnd,bnp->dp', d_out_f32, to_f32(patches))
    # d_b = sum over batch and patches dims
    d_b = jnp.sum(d_out_f32, axis=(0, 1))
    return to_bf16(d_patches), to_bf16(d_w), to_bf16(d_b)


def pos_embed_add_backward(
    d_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for pos_embed_add. Returns (d_x, d_cls_token, d_pos_embed)."""
    # d_out: [B, SEQ_LEN, D]
    # CLS token gradient: sum over batch from position 0
    d_cls_token = jnp.sum(d_out[:, 0, :], axis=0)
    # Patch gradients: all positions except CLS
    d_x = d_out[:, 1:, :]
    # Positional embedding gradient: sum over batch
    d_pos_embed = jnp.sum(d_out, axis=0)
    return d_x, d_cls_token, d_pos_embed


def qkv_proj_backward(
    d_qkv: jnp.ndarray,
    x: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for QKV projection. Returns (d_x, d_w, d_b)."""
    d_qkv_f32 = to_f32(d_qkv)
    # d_qkv is [B, S, 3*D] after reshaping from [B, S, 3, H, D_H]
    batch, seq = d_qkv_f32.shape[0], d_qkv_f32.shape[1]
    d_qkv_flat = d_qkv_f32.reshape(batch, seq, -1)
    # d_x = d_qkv @ w
    d_x = jnp.dot(d_qkv_flat, to_f32(w))
    # d_w = d_qkv^T @ x (summed over batch)
    d_w = jnp.einsum('bsi,bsj->ij', d_qkv_flat, to_f32(x))
    # d_b = sum over batch and seq
    d_b = jnp.sum(d_qkv_flat, axis=(0, 1))
    return to_bf16(d_x), to_bf16(d_w), to_bf16(d_b)


def attention_backward(
    d_out: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    d_head: int = D_HEAD,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for non-causal scaled dot-product attention. Returns (d_q, d_k, d_v)."""
    scale = 1.0 / math.sqrt(d_head)
    # Recompute attention weights
    attn_weights = to_f32(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale)
    probs = jax.nn.softmax(attn_weights, axis=-1)

    d_out_f32 = to_f32(d_out)
    probs_f32 = probs

    # d_v = probs^T @ d_out
    d_v = jnp.matmul(jnp.swapaxes(probs_f32, -2, -1), d_out_f32)
    # d_probs = d_out @ v^T
    d_probs = jnp.matmul(d_out_f32, jnp.swapaxes(to_f32(v), -2, -1))
    # Softmax backward: d_scores = probs * (d_probs - sum(d_probs * probs, axis=-1, keepdims))
    dot_pp = jnp.sum(d_probs * probs_f32, axis=-1, keepdims=True)
    d_scores = probs_f32 * (d_probs - dot_pp) * scale
    # d_q = d_scores @ k
    d_q = jnp.matmul(d_scores, to_f32(k))
    # d_k = d_scores^T @ q
    d_k = jnp.matmul(jnp.swapaxes(d_scores, -2, -1), to_f32(q))

    return to_bf16(d_q), to_bf16(d_k), to_bf16(d_v)


def o_proj_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for output projection. Returns (d_x, d_w, d_b)."""
    d_out_f32 = to_f32(d_out)
    d_x = jnp.dot(d_out_f32, to_f32(w))
    d_w = jnp.einsum('bsi,bsj->ij', d_out_f32, to_f32(x))
    d_b = jnp.sum(d_out_f32, axis=(0, 1))
    return to_bf16(d_x), to_bf16(d_w), to_bf16(d_b)


def residual_add_backward(
    d_out: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward for residual addition. Returns (d_x, d_residual)."""
    return d_out, d_out


def mlp_gate_up_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    w_gate: jnp.ndarray,
    w_up: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for gated MLP up projection. Returns (d_x, d_w_gate, d_w_up, gate_pre)."""
    x_f32 = to_f32(x)
    d_out_f32 = to_f32(d_out)
    w_gate_f32 = to_f32(w_gate)
    w_up_f32 = to_f32(w_up)

    # Recompute forward intermediates
    gate_pre = jnp.dot(x_f32, w_gate_f32.T)
    gate_act = jax.nn.silu(gate_pre)
    up = jnp.dot(x_f32, w_up_f32.T)

    # d_gate_act = d_out * up,  d_up = d_out * gate_act
    d_gate_act = d_out_f32 * up
    d_up = d_out_f32 * gate_act

    # SiLU backward: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sig = jax.nn.sigmoid(gate_pre)
    d_gate_pre = d_gate_act * sig * (1.0 + gate_pre * (1.0 - sig))

    # d_x = d_gate_pre @ w_gate + d_up @ w_up
    d_x = jnp.dot(d_gate_pre, w_gate_f32) + jnp.dot(d_up, w_up_f32)
    # d_w_gate = d_gate_pre^T @ x
    d_w_gate = jnp.einsum('bsi,bsj->ij', d_gate_pre, x_f32)
    # d_w_up = d_up^T @ x
    d_w_up = jnp.einsum('bsi,bsj->ij', d_up, x_f32)

    return to_bf16(d_x), to_bf16(d_w_gate), to_bf16(d_w_up), gate_pre


def mlp_down_backward(
    d_out: jnp.ndarray,
    x: jnp.ndarray,
    w: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for MLP down projection. Returns (d_x, d_w, d_b)."""
    d_out_f32 = to_f32(d_out)
    d_x = jnp.dot(d_out_f32, to_f32(w))
    d_w = jnp.einsum('bsi,bsj->ij', d_out_f32, to_f32(x))
    d_b = jnp.sum(d_out_f32, axis=(0, 1))
    return to_bf16(d_x), to_bf16(d_w), to_bf16(d_b)


def classifier_head_backward(
    d_logits: jnp.ndarray,
    h: jnp.ndarray,
    state: ViTState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward for classifier head. Returns (d_h, d_final_norm_g, d_final_norm_b, d_head_w, d_head_b)."""
    d_logits_f32 = to_f32(d_logits)
    cls_hidden = h[:, 0, :]  # [B, D]

    # --- Head linear backward ---
    if state.tied_embeddings:
        head_w_f32 = to_f32(state.patch_proj_w).T  # [D, PATCH_DIM] -> treat as [D, N_CLASSES]
    else:
        head_w_f32 = to_f32(state.head_w)  # [N_CLASSES, D]

    d_head_b = jnp.sum(d_logits_f32, axis=0)

    if state.tied_embeddings:
        # d_normed = d_logits @ patch_proj_w^T -> [B, D]
        d_normed = jnp.dot(d_logits_f32, to_f32(state.patch_proj_w))
        d_head_w = jnp.zeros_like(state.head_w, dtype=PARAM_DTYPE)
    else:
        d_normed = jnp.dot(d_logits_f32, head_w_f32)  # [B, D]
        d_head_w = jnp.einsum('bi,bj->ij', d_logits_f32, to_f32(cls_hidden))

    # --- Final norm backward (RMSNorm + bias) ---
    cls_f32 = to_f32(cls_hidden)
    gamma_f32 = to_f32(state.final_norm_g)
    ms = jnp.mean(cls_f32 ** 2, axis=-1, keepdims=True)
    rms_inv = jax.lax.rsqrt(ms + 1e-6)

    d_normed_f32 = to_f32(d_normed)
    x_hat = cls_f32 * rms_inv
    d_final_norm_g = jnp.sum(d_normed_f32 * x_hat, axis=0)
    d_final_norm_b = jnp.sum(d_normed_f32, axis=0)

    d_model_dim = cls_f32.shape[-1]
    dot = jnp.sum(d_normed_f32 * cls_f32 * gamma_f32, axis=-1, keepdims=True)
    coeff = rms_inv ** 3 / d_model_dim
    d_cls = rms_inv * gamma_f32 * d_normed_f32 - coeff * cls_f32 * dot

    # Scatter CLS gradient back into full sequence
    batch = h.shape[0]
    seq = h.shape[1]
    d_model_size = h.shape[2]
    d_h = jnp.zeros((batch, seq, d_model_size), dtype=PARAM_DTYPE)
    d_h = d_h.at[:, 0, :].set(to_bf16(d_cls))

    return d_h, to_bf16(d_final_norm_g), to_bf16(d_final_norm_b), to_bf16(d_head_w), to_bf16(d_head_b)

# ---------------------------------------------------------------------------
# Single layer backward
# ---------------------------------------------------------------------------


def _vit_layer_backward(
    d_h: jnp.ndarray,
    h_input: jnp.ndarray,
    layer: ViTLayerWeights,
    n_heads: int,
) -> tuple[jnp.ndarray, ViTLayerWeights]:
    """Backward pass through one ViT encoder layer. Returns (d_input, d_layer_weights)."""
    d_model = h_input.shape[-1]
    d_head = d_model // n_heads

    # --- Recompute forward for saved activations ---
    # Attention sub-block
    qkv = qkv_proj_forward(h_input, layer.qkv_w, layer.qkv_b, n_heads)
    q = jnp.transpose(qkv[:, :, 0, :, :], (0, 2, 1, 3))
    k = jnp.transpose(qkv[:, :, 1, :, :], (0, 2, 1, 3))
    v = jnp.transpose(qkv[:, :, 2, :, :], (0, 2, 1, 3))
    attn_out_heads = attention_forward(q, k, v, d_head, causal=False)
    attn_out = jnp.transpose(attn_out_heads, (0, 2, 1, 3))
    attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], d_model)
    o_proj_out = o_proj_forward(attn_out, layer.out_w, layer.out_b)
    h_after_attn = residual_add(o_proj_out, h_input)
    h_normed1 = rmsnorm_forward(h_after_attn, layer.norm1_g)

    # MLP sub-block
    mlp_up_out = mlp_gate_up_forward(h_normed1, layer.gate_w, layer.up_w)
    mlp_out = mlp_down_forward(mlp_up_out, layer.down_w, layer.down_b)
    h_after_mlp = residual_add(mlp_out, h_normed1)

    # --- Backward through MLP block ---
    # RMSNorm2 backward
    d_pre_norm2, d_norm2_g = rmsnorm_backward(d_h, h_after_mlp, layer.norm2_g)

    # Residual backward
    d_mlp_out, d_residual_mlp = residual_add_backward(d_pre_norm2)

    # MLP down backward
    d_mlp_act, d_down_w, d_down_b = mlp_down_backward(d_mlp_out, mlp_up_out, layer.down_w)

    # MLP gate+up backward
    d_mlp_input, d_gate_w, d_up_w, _ = mlp_gate_up_backward(
        d_mlp_act, h_normed1, layer.gate_w, layer.up_w,
    )

    # Accumulate gradient from residual
    d_h_normed1 = d_mlp_input + d_residual_mlp

    # --- Backward through attention block ---
    # RMSNorm1 backward
    d_pre_norm1, d_norm1_g = rmsnorm_backward(d_h_normed1, h_after_attn, layer.norm1_g)

    # Residual backward
    d_o_proj, d_residual_attn = residual_add_backward(d_pre_norm1)

    # Output projection backward
    d_attn_out, d_out_w, d_out_b = o_proj_backward(d_o_proj, attn_out, layer.out_w)

    # Reshape d_attn_out for attention backward: [B, S, D] -> [B, H, S, D_H]
    batch, seq = d_attn_out.shape[0], d_attn_out.shape[1]
    d_attn_heads = d_attn_out.reshape(batch, seq, n_heads, d_head)
    d_attn_heads = jnp.transpose(d_attn_heads, (0, 2, 1, 3))

    # Attention backward
    d_q, d_k, d_v = attention_backward(d_attn_heads, q, k, v, d_head)

    # Transpose back and recombine: [B, H, S, D_H] -> [B, S, 3, H, D_H] -> [B, S, 3*D]
    d_q = jnp.transpose(d_q, (0, 2, 1, 3))
    d_k = jnp.transpose(d_k, (0, 2, 1, 3))
    d_v = jnp.transpose(d_v, (0, 2, 1, 3))
    d_qkv = jnp.stack([d_q, d_k, d_v], axis=2)  # [B, S, 3, H, D_H]
    d_qkv_flat = d_qkv.reshape(batch, seq, 3 * d_model)

    # QKV projection backward
    d_qkv_input, d_qkv_w, d_qkv_b = qkv_proj_backward(d_qkv_flat, h_input, layer.qkv_w)

    # Accumulate gradient from residual
    d_input = d_qkv_input + d_residual_attn

    # Assemble gradient dataclass
    d_layer = ViTLayerWeights(
        qkv_w=d_qkv_w,
        qkv_b=d_qkv_b,
        out_w=d_out_w,
        out_b=d_out_b,
        norm1_g=d_norm1_g,
        norm1_b=jnp.zeros_like(layer.norm1_b),  # RMSNorm has no bias; placeholder
        gate_w=d_gate_w,
        up_w=d_up_w,
        down_w=d_down_w,
        down_b=d_down_b,
        norm2_g=d_norm2_g,
        norm2_b=jnp.zeros_like(layer.norm2_b),
    )

    return d_input, d_layer

# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------


def vit_forward(
    state: ViTState,
    patches: jnp.ndarray,
    activations: dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Full ViT forward. Uses lax.scan over layers."""
    if activations is None:
        activations = {}

    # Patch embedding: [B, N_PATCHES, PATCH_DIM] -> [B, N_PATCHES, D]
    h = patch_embed_forward(patches, state.patch_proj_w, state.patch_proj_b)

    # Prepend CLS and add positional embeddings: [B, SEQ_LEN, D]
    h = pos_embed_add_forward(h, state.cls_token, state.pos_embed)
    activations['post_embed'] = h

    # Layer loop via lax.scan
    def scan_body(
        carry: jnp.ndarray,
        layer: ViTLayerWeights,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Scan body: one transformer layer. Returns (h_out, h_in_saved)."""
        h_in = carry
        h_out = _vit_layer_forward(h_in, layer, state.n_heads)
        return h_out, h_in

    # Stack layer weights for scan (each field becomes [N_LAYERS, ...])
    stacked_layers = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *state.layer_weights,
    )

    h, saved_inputs = lax.scan(scan_body, h, stacked_layers)
    activations['layer_inputs'] = saved_inputs
    activations['pre_head'] = h

    # Classifier head
    logits = classifier_head_forward(h, state)

    return logits, activations

# ---------------------------------------------------------------------------
# Full backward pass
# ---------------------------------------------------------------------------


def vit_backward(
    state: ViTState,
    d_logits: jnp.ndarray,
    activations: dict[str, Any],
) -> dict[str, Any]:
    """Full ViT backward."""
    h_pre_head = activations['pre_head']
    layer_inputs = activations['layer_inputs']  # [N_LAYERS, B, SEQ_LEN, D]
    h_post_embed = activations['post_embed']

    # Classifier head backward
    d_h, d_final_norm_g, d_final_norm_b, d_head_w, d_head_b = classifier_head_backward(
        d_logits, h_pre_head, state,
    )

    # Layer loop backward via lax.scan (reverse)
    stacked_layers = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *state.layer_weights,
    )

    def scan_body_bwd(
        carry: tuple[jnp.ndarray, int],
        xs: tuple[jnp.ndarray, Any],
    ) -> tuple[tuple[jnp.ndarray, int], Any]:
        """Reverse scan body: one layer backward."""
        d_h_cur, layer_idx = carry
        h_input, layer_w = xs
        d_input, d_layer = _vit_layer_backward(d_h_cur, h_input, layer_w, state.n_heads)
        return (d_input, layer_idx - 1), d_layer

    # Reverse the saved inputs and layer weights for backward scan
    reversed_inputs = jnp.flip(layer_inputs, axis=0)
    reversed_layers = jax.tree.map(lambda x: jnp.flip(x, axis=0), stacked_layers)

    n_layers = len(state.layer_weights)
    (d_h_out, _), d_stacked_layers = lax.scan(
        scan_body_bwd,
        (d_h, n_layers - 1),
        (reversed_inputs, reversed_layers),
    )

    # Un-reverse layer gradients
    d_stacked_layers = jax.tree.map(lambda x: jnp.flip(x, axis=0), d_stacked_layers)

    # Positional embedding backward
    d_patches_pre, d_cls_token, d_pos_embed = pos_embed_add_backward(d_h_out)

    # Patch embedding backward
    # We need original patches; recover from h_post_embed is not directly possible,
    # so the caller must provide them in activations if needed.
    patches = activations.get('patches', None)
    if patches is not None:
        d_patches, d_patch_proj_w, d_patch_proj_b = patch_embed_backward(
            d_patches_pre, patches, state.patch_proj_w,
        )
    else:
        d_patch_proj_w = jnp.zeros_like(state.patch_proj_w)
        d_patch_proj_b = jnp.zeros_like(state.patch_proj_b)

    # Unstack layer gradients into list
    d_layer_weights = []
    for i in range(n_layers):
        d_layer_weights.append(
            jax.tree.map(lambda x: x[i], d_stacked_layers)
        )

    grads = {
        'patch_proj_w': d_patch_proj_w,
        'patch_proj_b': d_patch_proj_b,
        'cls_token': d_cls_token,
        'pos_embed': d_pos_embed,
        'layer_weights': d_layer_weights,
        'final_norm_g': d_final_norm_g,
        'final_norm_b': d_final_norm_b,
        'head_w': d_head_w,
        'head_b': d_head_b,
    }

    return grads
