"""TPU v5p model surface for Decoder Transformer, ViT, and Mamba.

Provides Pallas/JAX implementations of three models used by the grokking
race driver.  Each model exposes forward and backward functions suitable
for TPU v5p (128-wide MXU tiles).  All Pallas kernels are wrapped in
try/except with pure-JAX fallbacks so the code remains functional when
the Pallas API changes or is unavailable.

Models
------
- Decoder Transformer  (causal self-attention)
- Vision Transformer   (ViT, full attention, patch-based)
- Mamba                 (selective state-space model)
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention as _splash_fn
    _HAS_SPLASH = True
except (ImportError, AttributeError):
    _HAS_SPLASH = False

# Reuse optimizer-side scan from _pallas_kernels (DO NOT reimplement)
try:
    from ._pallas_kernels import pallas_affine_scan, _associative_combine
    _HAS_PALLAS_SCAN = True
except ImportError:
    _HAS_PALLAS_SCAN = False


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Frozen configuration shared by all three model families."""
    batch_size: int
    seq_len: int
    d_model: int
    n_heads: int
    d_head: int
    n_layers: int
    vocab_size: int
    ffn_expansion: int = 4
    # ViT-specific
    patch_dim: int = 0
    num_patches: int = 0
    num_classes: int = 0
    # Mamba-specific
    d_state: int = 0
    expand_factor: int = 0
    d_inner: int = 0
    conv_kernel: int = 0
    dt_rank: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _layer_norm(x, gamma, beta, eps=1e-5):
    """Standard layer normalisation."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


# ═══════════════════════════════════════════════════════════════════════
#  Shared attention (used by Decoder and ViT)
# ═══════════════════════════════════════════════════════════════════════

def _hand_tiled_attention_forward(q, k, v, *, causal: bool, tile_size=128):
    """Hand-tiled QKV attention in BF16 with optional causal mask.

    Args:
        q: [batch, n_heads, seq_len, d_head]
        k: [batch, n_heads, seq_len, d_head]
        v: [batch, n_heads, seq_len, d_head]
        causal: whether to apply a causal mask
        tile_size: tile width for Pallas grid (unused in JAX fallback)

    Returns:
        (out, softmax_lse) where out has same shape as q and
        softmax_lse is [batch, n_heads, seq_len].
    """
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)

    scale = q.shape[-1] ** -0.5
    attn_weights = jnp.einsum('bhid,bhjd->bhij', q, k) * scale

    if causal:
        seq_len = q.shape[2]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn_weights = jnp.where(mask[None, None], attn_weights, -1e9)

    # Numerically-stable softmax — keep log-sum-exp for backward
    max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    shifted = attn_weights - max_score
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    softmax_lse = jnp.squeeze(max_score + jnp.log(sum_exp), axis=-1)
    attn_probs = exp_shifted / sum_exp

    out = jnp.einsum('bhij,bhjd->bhid', attn_probs, v)
    return out, softmax_lse


def attention_forward(q, k, v, *, causal: bool, tile_size=128):
    """Multi-head attention forward — splash_attention first, hand-tiled fallback.

    Args:
        q, k, v: [batch, n_heads, seq_len, d_head]
        causal: whether to apply a causal mask
        tile_size: Pallas tile size (default 128 for v5p)

    Returns:
        (out, softmax_lse)
    """
    if _HAS_SPLASH:
        try:
            out = _splash_fn(q, k, v, is_causal=causal) if callable(_splash_fn) else None
            if out is None:
                raise RuntimeError("splash fallback")
            # Compute softmax_lse for backward compatibility
            scale = q.shape[-1] ** -0.5
            attn_weights = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
            if causal:
                seq_len = q.shape[2]
                cmask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
                attn_weights = jnp.where(cmask[None, None], attn_weights, -1e9)
            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            softmax_lse = jnp.squeeze(
                max_score + jnp.log(jnp.sum(jnp.exp(attn_weights - max_score), axis=-1, keepdims=True)),
                axis=-1,
            )
            return out, softmax_lse
        except Exception:
            pass
    return _hand_tiled_attention_forward(q, k, v, causal=causal, tile_size=tile_size)


def attention_backward(grad_out, q, k, v, out, softmax_lse, *, causal: bool, tile_size=128):
    """Attention backward pass via JAX autodiff.

    Args:
        grad_out: gradient w.r.t. attention output [batch, n_heads, seq_len, d_head]
        q, k, v: original inputs
        out: forward output (unused — recomputed internally)
        softmax_lse: log-sum-exp from forward (unused — recomputed internally)
        causal: whether causal mask was applied
        tile_size: Pallas tile size

    Returns:
        (grad_q, grad_k, grad_v) with same shapes as (q, k, v).
    """
    def _fwd(q_, k_, v_):
        o, _ = attention_forward(q_, k_, v_, causal=causal, tile_size=tile_size)
        return o

    grad_q, grad_k, grad_v = jax.grad(_fwd, argnums=(0, 1, 2))(q, k, v)
    return grad_q, grad_k, grad_v


# ═══════════════════════════════════════════════════════════════════════
#  Decoder Transformer
# ═══════════════════════════════════════════════════════════════════════

class DecoderWeights(NamedTuple):
    """Weight tuple for an L-layer causal Decoder Transformer."""
    embed: jnp.ndarray            # [vocab_size, d_model]
    pos_embed: jnp.ndarray        # [seq_len, d_model]
    ln_attn_gamma: jnp.ndarray    # [n_layers, d_model]
    ln_attn_beta: jnp.ndarray     # [n_layers, d_model]
    w_qkv: jnp.ndarray            # [n_layers, d_model, 3 * n_heads * d_head]
    w_o: jnp.ndarray              # [n_layers, n_heads * d_head, d_model]
    ln_ffn_gamma: jnp.ndarray     # [n_layers, d_model]
    ln_ffn_beta: jnp.ndarray      # [n_layers, d_model]
    w_up: jnp.ndarray             # [n_layers, d_model, ffn_expansion * d_model]
    w_down: jnp.ndarray           # [n_layers, ffn_expansion * d_model, d_model]
    ln_final_gamma: jnp.ndarray   # [d_model]
    ln_final_beta: jnp.ndarray    # [d_model]
    w_vocab: jnp.ndarray          # [d_model, vocab_size]


@jax.jit
def decoder_forward(tokens, weights: DecoderWeights, cfg: ModelConfig):
    """Decoder Transformer forward pass.

    Embedding + positional -> L layers of (LN + causal attention + LN + FFN
    with GELU) -> final LN -> last-token logits via vocab head.

    Args:
        tokens: [batch, seq_len] integer token ids
        weights: DecoderWeights named tuple
        cfg: ModelConfig

    Returns:
        logits: [batch, vocab_size] logits from the last token position
    """
    B, S = tokens.shape
    hd = cfg.n_heads * cfg.d_head

    # Embedding + positional
    x = weights.embed[tokens] + weights.pos_embed[None, :S, :]  # [B, S, d_model]

    saved_activations = []
    for layer in range(cfg.n_layers):
        # Pre-LN attention
        x_ln = _layer_norm(x, weights.ln_attn_gamma[layer], weights.ln_attn_beta[layer])
        qkv = x_ln @ weights.w_qkv[layer]  # [B, S, 3*hd]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B, S, hd]

        # Reshape to [B, n_heads, S, d_head]
        q = q.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)

        attn_out, _ = attention_forward(q, k, v, causal=True)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, hd)  # [B, S, hd]
        attn_out = attn_out @ weights.w_o[layer]  # [B, S, d_model]
        x = x + attn_out

        # Pre-LN FFN
        x_ln2 = _layer_norm(x, weights.ln_ffn_gamma[layer], weights.ln_ffn_beta[layer])
        hidden = jax.nn.gelu(x_ln2 @ weights.w_up[layer])
        ffn_out = hidden @ weights.w_down[layer]
        x = x + ffn_out

        saved_activations.append(x)

    # Final LN + last-token logits
    x = _layer_norm(x, weights.ln_final_gamma, weights.ln_final_beta)
    last_token = x[:, -1, :]  # [B, d_model]
    logits = last_token @ weights.w_vocab  # [B, vocab_size]
    return logits


def decoder_backward(grad_logits, tokens, weights, cfg, saved_activations):
    """Decoder Transformer backward pass via jax.grad.

    Args:
        grad_logits: gradient w.r.t. logits [batch, vocab_size]
        tokens: original input tokens
        weights: DecoderWeights
        cfg: ModelConfig
        saved_activations: unused (recomputed by jax.grad)

    Returns:
        Gradients w.r.t. weights (same structure as DecoderWeights).
    """
    def _loss_fn(w):
        logits = decoder_forward(tokens, w, cfg)
        return jnp.sum(logits * grad_logits)

    return jax.grad(_loss_fn)(weights)


# ═══════════════════════════════════════════════════════════════════════
#  Vision Transformer (ViT)
# ═══════════════════════════════════════════════════════════════════════

class ViTWeights(NamedTuple):
    """Weight tuple for an L-layer ViT."""
    w_patch: jnp.ndarray          # [patch_dim, d_model]
    cls_token: jnp.ndarray        # [1, 1, d_model]
    pos_embed: jnp.ndarray        # [1, num_patches + 1, d_model]
    ln_attn_gamma: jnp.ndarray    # [n_layers, d_model]
    ln_attn_beta: jnp.ndarray     # [n_layers, d_model]
    w_qkv: jnp.ndarray            # [n_layers, d_model, 3 * n_heads * d_head]
    w_o: jnp.ndarray              # [n_layers, n_heads * d_head, d_model]
    ln_ffn_gamma: jnp.ndarray     # [n_layers, d_model]
    ln_ffn_beta: jnp.ndarray      # [n_layers, d_model]
    w_up: jnp.ndarray             # [n_layers, d_model, ffn_expansion * d_model]
    w_down: jnp.ndarray           # [n_layers, ffn_expansion * d_model, d_model]
    ln_final_gamma: jnp.ndarray   # [d_model]
    ln_final_beta: jnp.ndarray    # [d_model]
    w_classify: jnp.ndarray       # [d_model, num_classes]


def _unfold_patches(image, patch_size):
    """Unfold a 2D image into non-overlapping patches.

    Args:
        image: [batch, H, W, C] or [batch, H, W] input image.
            For the ViT surface we expect 28x14 images unfolded into
            16 patches of 7x7.
        patch_size: spatial extent of each square patch.

    Returns:
        patches: [batch, num_patches, patch_dim] where
        patch_dim = patch_size * patch_size * C.
    """
    if image.ndim == 3:
        image = image[..., None]  # add channel dim
    B, H, W, C = image.shape
    nH = H // patch_size
    nW = W // patch_size
    # Reshape to [B, nH, patch_size, nW, patch_size, C]
    x = image.reshape(B, nH, patch_size, nW, patch_size, C)
    # Transpose to [B, nH, nW, patch_size, patch_size, C]
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Flatten to [B, num_patches, patch_dim]
    return x.reshape(B, nH * nW, patch_size * patch_size * C)


def vit_patch_project(image, w_patch):
    """Standalone patch projection — useful as a test surface.

    Args:
        image: [batch, H, W, C]
        w_patch: [patch_dim, d_model]

    Returns:
        projected: [batch, num_patches, d_model]
    """
    patch_size = int(w_patch.shape[0] ** 0.5)  # infer from patch_dim
    if image.ndim == 3:
        image = image[..., None]
    C = image.shape[-1]
    patch_size = int((w_patch.shape[0] // C) ** 0.5)
    patches = _unfold_patches(image, patch_size)
    return patches @ w_patch


@jax.jit
def vit_forward(image, weights: ViTWeights, cfg: ModelConfig):
    """ViT forward pass.

    Unfold 28x14 image into 16 7x7 patches -> patch projection ->
    prepend CLS token -> positional -> L layers (LN + full attention +
    LN + FFN) -> final LN -> CLS token -> classify head.

    Args:
        image: [batch, H, W] or [batch, H, W, C] input image
        weights: ViTWeights named tuple
        cfg: ModelConfig

    Returns:
        logits: [batch, num_classes]
    """
    if image.ndim == 3:
        image = image[..., None]
    B = image.shape[0]
    C = image.shape[-1]
    hd = cfg.n_heads * cfg.d_head

    # Patch projection
    patch_size = int((cfg.patch_dim // C) ** 0.5) if cfg.patch_dim > 0 else 7
    patches = _unfold_patches(image, patch_size)  # [B, num_patches, patch_dim]
    x = patches @ weights.w_patch  # [B, num_patches, d_model]

    # Prepend CLS token
    cls_tokens = jnp.broadcast_to(weights.cls_token, (B, 1, cfg.d_model))
    x = jnp.concatenate([cls_tokens, x], axis=1)  # [B, num_patches+1, d_model]

    # Positional embedding
    x = x + weights.pos_embed[:, :x.shape[1], :]

    S = x.shape[1]
    for layer in range(cfg.n_layers):
        # Pre-LN attention (full, not causal)
        x_ln = _layer_norm(x, weights.ln_attn_gamma[layer], weights.ln_attn_beta[layer])
        qkv = x_ln @ weights.w_qkv[layer]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)

        attn_out, _ = attention_forward(q, k, v, causal=False)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, hd)
        attn_out = attn_out @ weights.w_o[layer]
        x = x + attn_out

        # Pre-LN FFN
        x_ln2 = _layer_norm(x, weights.ln_ffn_gamma[layer], weights.ln_ffn_beta[layer])
        hidden = jax.nn.gelu(x_ln2 @ weights.w_up[layer])
        ffn_out = hidden @ weights.w_down[layer]
        x = x + ffn_out

    # Final LN -> CLS token -> classify
    x = _layer_norm(x, weights.ln_final_gamma, weights.ln_final_beta)
    cls_out = x[:, 0, :]  # [B, d_model]
    logits = cls_out @ weights.w_classify  # [B, num_classes]
    return logits


def vit_backward(grad_logits, image, weights, cfg):
    """ViT backward pass via jax.grad.

    Args:
        grad_logits: gradient w.r.t. logits [batch, num_classes]
        image: original input image
        weights: ViTWeights
        cfg: ModelConfig

    Returns:
        Gradients w.r.t. weights (same structure as ViTWeights).
    """
    def _loss_fn(w):
        logits = vit_forward(image, w, cfg)
        return jnp.sum(logits * grad_logits)

    return jax.grad(_loss_fn)(weights)


# ═══════════════════════════════════════════════════════════════════════
#  Mamba (selective state-space model)
# ═══════════════════════════════════════════════════════════════════════

class MambaLayerWeights(NamedTuple):
    """Weight tuple for a single Mamba layer."""
    ln_gamma: jnp.ndarray       # [d_model]
    ln_beta: jnp.ndarray        # [d_model]
    w_in: jnp.ndarray           # [d_model, 2 * d_inner]
    conv_w: jnp.ndarray         # [d_inner, 1, conv_kernel]
    conv_b: jnp.ndarray         # [d_inner]
    w_BCdt: jnp.ndarray         # [d_inner, d_state + d_state + dt_rank]
    dt_proj: jnp.ndarray        # [dt_rank, d_inner]
    dt_bias: jnp.ndarray        # [d_inner]
    A_log: jnp.ndarray          # [d_inner, d_state]
    D: jnp.ndarray              # [d_inner]
    w_out: jnp.ndarray          # [d_inner, d_model]


class MambaWeights(NamedTuple):
    """Weight tuple for an L-layer Mamba model."""
    embed: jnp.ndarray                       # [vocab_size, d_model]
    layers: Tuple[MambaLayerWeights, ...]     # length n_layers
    ln_final_gamma: jnp.ndarray              # [d_model]
    ln_final_beta: jnp.ndarray               # [d_model]
    w_vocab: jnp.ndarray                     # [d_model, vocab_size]


# -- Selective scan -------------------------------------------------------

def _jax_selective_scan_fallback(x, B_mat, C_mat, dt, A_log):
    """Pure-JAX selective scan using lax.associative_scan with Affine2x2 combine.

    Args:
        x: [batch, seq_len, d_inner]
        B_mat: [batch, seq_len, d_state]
        C_mat: [batch, seq_len, d_state]
        dt: [batch, seq_len, d_inner]
        A_log: [d_inner, d_state]

    Returns:
        y: [batch, seq_len, d_inner]
    """
    A = -jnp.exp(A_log)  # [d_inner, d_state]
    # Discretize: dA = exp(dt * A), dB = dt * B * x
    dt_exp = dt[..., None]  # [B, S, d_inner, 1]
    dA = jnp.exp(dt_exp * A[None, None, :, :])  # [B, S, d_inner, d_state]
    dB_x = dt_exp * B_mat[:, :, None, :] * x[..., None]  # [B, S, d_inner, d_state]

    # Affine2x2 associative scan: state_{t} = dA_t * state_{t-1} + dB_x_t
    Ms = dA       # [B, S, d_inner, d_state]
    bs = dB_x     # [B, S, d_inner, d_state]

    def _combine(left, right):
        M_l, b_l = left
        M_r, b_r = right
        return (M_r * M_l, M_r * b_l + b_r)

    final_Ms, final_bs = lax.associative_scan(_combine, (Ms, bs), axis=1)
    # final_bs is the state at each timestep: [B, S, d_inner, d_state]
    # Output: y_t = C_t . state_t  (dot over d_state)
    y = jnp.einsum('bsdn,bsn->bsd', final_bs, C_mat)
    return y


def mamba_selective_scan(x, B_mat, C_mat, dt, A_log):
    """Selective scan -- tries Pallas pallas_affine_scan, falls back to JAX.

    Args:
        x: [batch, seq_len, d_inner]
        B_mat: [batch, seq_len, d_state]
        C_mat: [batch, seq_len, d_state]
        dt: [batch, seq_len, d_inner]
        A_log: [d_inner, d_state]

    Returns:
        y: [batch, seq_len, d_inner]
    """
    try:
        from ._pallas_kernels import pallas_affine_scan
        A = -jnp.exp(A_log)
        dt_exp = dt[..., None]
        dA = jnp.exp(dt_exp * A[None, None, :, :])
        dB_x = dt_exp * B_mat[:, :, None, :] * x[..., None]
        N = dA.shape[1]
        d_inner = dA.shape[2]
        B_dim = dA.shape[0]
        d_state = dA.shape[3]
        Ms = dA.reshape(B_dim, N, d_inner * d_state)
        bs = dB_x.reshape(B_dim, N, d_inner * d_state)
        out_Ms, out_bs = pallas_affine_scan(Ms, bs, N, d_inner * d_state)
        states = out_bs.reshape(B_dim, N, d_inner, d_state)
        y = jnp.einsum('bsdn,bsn->bsd', states, C_mat)
        return y
    except Exception:
        return _jax_selective_scan_fallback(x, B_mat, C_mat, dt, A_log)


# -- Mamba layer -----------------------------------------------------------

def _mamba_layer(x, layer_w: MambaLayerWeights, cfg: ModelConfig):
    """Single Mamba layer: LN -> in_proj -> conv1d -> SSM -> gate -> out_proj -> residual.

    Args:
        x: [batch, seq_len, d_model] input activations
        layer_w: MambaLayerWeights for this layer
        cfg: ModelConfig

    Returns:
        x_out: [batch, seq_len, d_model] with residual connection
    """
    residual = x
    x = _layer_norm(x, layer_w.ln_gamma, layer_w.ln_beta)

    # In-projection -> split into x_ssm and z (gate)
    xz = x @ layer_w.w_in  # [B, S, 2 * d_inner]
    x_ssm, z = jnp.split(xz, 2, axis=-1)  # each [B, S, d_inner]

    # Depthwise conv1d via lax.conv_general_dilated (groups=d_inner)
    # conv_w: [d_inner, 1, conv_kernel] -- interpreted as [out, in/groups, spatial]
    # Input needs to be [B, d_inner, S] for 1D conv
    x_conv = x_ssm.transpose(0, 2, 1)  # [B, d_inner, S]
    # Pad on the left so output length == input length (causal)
    pad_len = cfg.conv_kernel - 1
    x_conv = jnp.pad(x_conv, ((0, 0), (0, 0), (pad_len, 0)))
    x_conv = lax.conv_general_dilated(
        x_conv,                          # [B, d_inner, S + pad]
        layer_w.conv_w,                  # [d_inner, 1, conv_kernel]
        window_strides=(1,),
        padding='VALID',
        feature_group_count=cfg.d_inner,
    )  # [B, d_inner, S]
    x_conv = x_conv + layer_w.conv_b[:, None]  # bias
    x_conv = jax.nn.silu(x_conv)
    x_conv = x_conv.transpose(0, 2, 1)  # [B, S, d_inner]

    # BCdt projection
    BCdt = x_conv @ layer_w.w_BCdt  # [B, S, d_state + d_state + dt_rank]
    B_mat = BCdt[:, :, :cfg.d_state]
    C_mat = BCdt[:, :, cfg.d_state:2 * cfg.d_state]
    dt_raw = BCdt[:, :, 2 * cfg.d_state:]  # [B, S, dt_rank]

    # dt projection + softplus
    dt = dt_raw @ layer_w.dt_proj + layer_w.dt_bias[None, None, :]  # [B, S, d_inner]
    dt = jax.nn.softplus(dt)

    # Selective scan
    y = mamba_selective_scan(x_conv, B_mat, C_mat, dt, layer_w.A_log)

    # Skip connection with D
    y = y + x_conv * layer_w.D[None, None, :]

    # Gate with SiLU(z)
    y = y * jax.nn.silu(z)

    # Out projection + residual
    out = y @ layer_w.w_out  # [B, S, d_model]
    return out + residual


def mamba_layer_forward(x, layer_w: MambaLayerWeights, cfg: ModelConfig):
    """Standalone single Mamba layer -- useful as a test surface.

    Args:
        x: [batch, seq_len, d_model]
        layer_w: MambaLayerWeights
        cfg: ModelConfig

    Returns:
        [batch, seq_len, d_model]
    """
    return _mamba_layer(x, layer_w, cfg)


@jax.jit
def mamba_forward(tokens, weights: MambaWeights, cfg: ModelConfig):
    """Mamba model forward pass.

    Embedding -> L layers -> final LN -> last-token logits.

    Args:
        tokens: [batch, seq_len] integer token ids
        weights: MambaWeights named tuple
        cfg: ModelConfig

    Returns:
        logits: [batch, vocab_size]
    """
    x = weights.embed[tokens]  # [B, S, d_model]

    for layer_idx in range(cfg.n_layers):
        x = _mamba_layer(x, weights.layers[layer_idx], cfg)

    # Final LN + last-token logits
    x = _layer_norm(x, weights.ln_final_gamma, weights.ln_final_beta)
    last_token = x[:, -1, :]  # [B, d_model]
    logits = last_token @ weights.w_vocab  # [B, vocab_size]
    return logits


def mamba_backward(grad_logits, tokens, weights, cfg):
    """Mamba backward pass via jax.grad.

    Args:
        grad_logits: gradient w.r.t. logits [batch, vocab_size]
        tokens: original input tokens
        weights: MambaWeights
        cfg: ModelConfig

    Returns:
        Gradients w.r.t. weights (same structure as MambaWeights).
    """
    def _loss_fn(w):
        logits = mamba_forward(tokens, w, cfg)
        return jnp.sum(logits * grad_logits)

    return jax.grad(_loss_fn)(weights)


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "ModelConfig",
    "_layer_norm",
    "attention_forward",
    "attention_backward",
    "DecoderWeights",
    "decoder_forward",
    "decoder_backward",
    "ViTWeights",
    "vit_forward",
    "vit_backward",
    "vit_patch_project",
    "MambaLayerWeights",
    "MambaWeights",
    "mamba_forward",
    "mamba_backward",
    "mamba_layer_forward",
    "mamba_selective_scan",
]
