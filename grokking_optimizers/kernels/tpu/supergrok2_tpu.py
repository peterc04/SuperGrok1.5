"""TPU/Pallas kernels for SuperGrok v2 -- CSA/HCA hybrid attention + PEER + GRU.

DeepSeek-V4-style learned-optimizer meta-net (see /tmp/csa_hca_spec.md, esp.
SS2, SS3b). The flattened parameter-element sequence is mixed by compressed
attention, replacing the previous Mamba-3 bidirectional selective scan. The
GRU + PEER + expert-MLP + AdamW apply tail is KEPT VERBATIM (only the sequence
mixer changes -- spec SS3b). Combines:
  - CSA: Compressed Sparse Attention (compress m=4 + lightning-indexer top-k +
    sliding window) -> csa_ctx  (fine/local context; was the mamba fwd scan)
  - HCA: Heavily Compressed Attention (heavy compress m'=128 + dense attention +
    sliding window) -> hca_ctx  (global coarse context; was the mamba bwd scan)
  - 4-Head PEER product-key expert routing                 (UNCHANGED)
  - Per-element GRU temporal memory                         (UNCHANGED)
  - AdamW with sigmoid gating and adaptive scheduling       (UNCHANGED)

CSA mechanics (spec SS2):
  c_kv[j]  = sum_w softmax(compress_w)[w] * x[j*m + w]      (KV compression)
  qI       = (x @ idx_DQ) @ idx_UQ ;  kI = compress(x @ idx_K)  (rank R indexer)
  I[t,s]   = qI[t] . kI[s] / sqrt(d)  -> keep top-k compressed entries per query
  out      = softmax(Q.K^T / sqrt(head_dim)) . V  over (top-k U window), MQA KV.
HCA mechanics: stride-m' mean pool -> Nh entries; every query attends DENSELY to
all Nh compressed entries (+window). No top-k.

Key TPU optimisations:
  - reshape + mean for O(1) KV compression (no sequential scan)
  - einsum + jax.lax.top_k for the lightning indexer selection
  - jnp.einsum + jax.nn.softmax for attention (XLA -> MXU)
  - jax.vmap for expert MLP evaluation (vectorised across experts)
  - Pallas kernels for element-wise fused AdamW

Attention is STATELESS across optimizer steps (unlike the carried Mamba state),
so the mamba_fwd_state / mamba_bwd_state entries are dropped; the GRU state stays.
BF16 parameters, FP32 accumulators throughout.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

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
# Component 1a: shared attention helpers (compression + sliding-window mask)
# ---------------------------------------------------------------------------


def _compress_kv(
    kv: jnp.ndarray,        # [N, d_model]
    compress_w: jnp.ndarray,  # [window] or None -> uniform mean pool
    m: int,                 # compression stride
    window: int,            # pooling window (>= m); for HCA window == m
) -> jnp.ndarray:
    """Pool a `window`-wide block at stride `m` -> [Nc, d_model] (spec SS2.1).

    c_kv[j] = sum_w softmax(compress_w)[w] * kv[j*m + w]. Implemented with a
    reshape into [Nc, m, d_model] (zero-padded) and a weighted mean -- no scan.
    When window > m the spec pools overlapping windows; for the reference path we
    use the stride-m non-overlapping block (window collapses to m) so the reshape
    stays exact, which is the common m == window case.
    """
    N, d_model = kv.shape
    Nc = (N + m - 1) // m
    pad = Nc * m - N
    if pad:
        kv = jnp.concatenate([kv, jnp.zeros((pad, d_model), kv.dtype)], axis=0)
    blocks = kv.reshape(Nc, m, d_model)  # [Nc, m, d_model]

    if compress_w is None:
        weights = jnp.full((m,), 1.0 / m, dtype=ACCUM_DTYPE)
    else:
        # softmax over the first m entries of the learned window weights
        w = compress_w[:m].astype(ACCUM_DTYPE)
        weights = jax.nn.softmax(w)
    return jnp.einsum('cmd,m->cd', blocks.astype(ACCUM_DTYPE), weights)  # [Nc, d_model]


def _sliding_window_bias(N: int, window: int) -> jnp.ndarray:
    """Additive attention bias [N, N] allowing only the last `window` causal
    tokens (token t attends to t-window+1 .. t). 0 where allowed, -inf else."""
    t = jnp.arange(N)[:, None]
    s = jnp.arange(N)[None, :]
    allowed = (s <= t) & (s > t - window)
    return jnp.where(allowed, 0.0, -jnp.inf).astype(ACCUM_DTYPE)


# ---------------------------------------------------------------------------
# Component 1b: CSA -- Compressed Sparse Attention (spec SS2)
# ---------------------------------------------------------------------------


def sg2_csa_attention(
    x: jnp.ndarray,          # [N, d_model] projected feature sequence
    q_W: jnp.ndarray,        # [d_model, d_model]
    k_W: jnp.ndarray,        # [d_model, d_model]
    v_W: jnp.ndarray,        # [d_model, d_model]
    out_W: jnp.ndarray,      # [d_model, d_model]
    compress_w: jnp.ndarray,  # [csa_window] learned KV pool weights
    idx_DQ: jnp.ndarray,     # [d_model, indexer_rank]
    idx_UQ: jnp.ndarray,     # [indexer_rank, d_model]  (kept for parity; see note)
    idx_K: jnp.ndarray,      # [d_model, indexer_rank]
    n_heads: int,
    csa_compress: int,
    csa_window: int,
    csa_topk: int,
) -> jnp.ndarray:
    """Compressed Sparse Attention -> csa_ctx [N, d_model] (spec SS2 CSA).

    1. KV compression at stride m=csa_compress with learned pooling.
    2. Lightning indexer: low-rank query qI scores compressed indexer keys kI;
       keep top-k compressed entries per query (jax.lax.top_k).
    3. Sliding window: also attend to the last csa_window raw tokens (causal).
    4. Attention: softmax(Q.Kc^T / sqrt(head_dim)) . Vc over selected entries,
       plus the windowed raw attention; KV shared across heads (multi-query).
    """
    x = x.astype(ACCUM_DTYPE)
    N, d_model = x.shape
    head_dim = d_model // n_heads
    scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, ACCUM_DTYPE))

    q = x @ q_W.T  # [N, d_model]
    k = x @ k_W.T
    v = x @ v_W.T

    c_k = _compress_kv(k, compress_w, csa_compress, csa_window)  # [Nc, d_model]
    c_v = _compress_kv(v, compress_w, csa_compress, csa_window)  # [Nc, d_model]
    Nc = c_k.shape[0]

    # Lightning indexer (spec SS2.2): qI = (x @ idx_DQ) @ idx_UQ then projected
    # back to rank space against kI. We score directly in rank space: the rank-R
    # indexer query z = x @ idx_DQ and indexer keys kI = compress(x @ idx_K).
    R = idx_DQ.shape[1]
    z = x @ idx_DQ                       # [N, R]
    kI = _compress_kv(x @ idx_K, compress_w, csa_compress, csa_window)  # [Nc, R]
    idx_scores = (z @ kI.T) / jnp.sqrt(jnp.asarray(R, ACCUM_DTYPE))  # [N, Nc]

    k_keep = min(csa_topk, Nc)
    _, top_idx = jax.lax.top_k(idx_scores, k_keep)  # [N, k_keep] compressed ids

    # Per-head multi-query attention over the selected compressed entries.
    def _attend_head(h):
        off = h * head_dim
        q_h = q[:, off:off + head_dim]                  # [N, head_dim]
        ck_h = c_k[:, off:off + head_dim]               # [Nc, head_dim]
        cv_h = c_v[:, off:off + head_dim]
        # Gather selected compressed keys/values per query: [N, k_keep, head_dim]
        sel_k = ck_h[top_idx]
        sel_v = cv_h[top_idx]
        comp_scores = jnp.einsum('nd,nkd->nk', q_h, sel_k) * scale  # [N, k_keep]
        comp_p = jax.nn.softmax(comp_scores, axis=-1)
        comp_ctx = jnp.einsum('nk,nkd->nd', comp_p, sel_v)          # [N, head_dim]

        # Sliding-window raw attention (causal, last csa_window tokens).
        rk_h = k[:, off:off + head_dim]
        rv_h = v[:, off:off + head_dim]
        win_scores = jnp.einsum('nd,md->nm', q_h, rk_h) * scale     # [N, N]
        win_scores = win_scores + _sliding_window_bias(N, csa_window)
        win_p = jax.nn.softmax(win_scores, axis=-1)
        win_ctx = win_p @ rv_h                                       # [N, head_dim]

        return 0.5 * (comp_ctx + win_ctx)                           # [N, head_dim]

    heads = [_attend_head(h) for h in range(n_heads)]
    ctx = jnp.concatenate(heads, axis=-1)  # [N, d_model]
    return ctx @ out_W.T                    # output projection [N, d_model]


# ---------------------------------------------------------------------------
# Component 1c: HCA -- Heavily Compressed Attention (spec SS2)
# ---------------------------------------------------------------------------


def sg2_hca_attention(
    x: jnp.ndarray,          # [N, d_model] projected feature sequence
    q_W: jnp.ndarray,        # [d_model, d_model]
    k_W: jnp.ndarray,        # [d_model, d_model]
    v_W: jnp.ndarray,        # [d_model, d_model]
    out_W: jnp.ndarray,      # [d_model, d_model]
    n_heads: int,
    hca_compress: int,
    csa_window: int,
) -> jnp.ndarray:
    """Heavily Compressed Attention -> hca_ctx [N, d_model] (spec SS2 HCA).

    Stride-m' mean pool (m'=hca_compress) -> Nh compressed entries; every query
    attends DENSELY to all Nh entries (no top-k) plus the sliding window.
    """
    x = x.astype(ACCUM_DTYPE)
    N, d_model = x.shape
    head_dim = d_model // n_heads
    scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, ACCUM_DTYPE))

    q = x @ q_W.T
    k = x @ k_W.T
    v = x @ v_W.T

    c_k = _compress_kv(k, None, hca_compress, hca_compress)  # [Nh, d_model] mean
    c_v = _compress_kv(v, None, hca_compress, hca_compress)

    def _attend_head(h):
        off = h * head_dim
        q_h = q[:, off:off + head_dim]
        ck_h = c_k[:, off:off + head_dim]
        cv_h = c_v[:, off:off + head_dim]
        dense_scores = jnp.einsum('nd,md->nm', q_h, ck_h) * scale  # [N, Nh] dense
        dense_p = jax.nn.softmax(dense_scores, axis=-1)
        dense_ctx = dense_p @ cv_h                                  # [N, head_dim]

        rk_h = k[:, off:off + head_dim]
        rv_h = v[:, off:off + head_dim]
        win_scores = jnp.einsum('nd,md->nm', q_h, rk_h) * scale
        win_scores = win_scores + _sliding_window_bias(N, csa_window)
        win_p = jax.nn.softmax(win_scores, axis=-1)
        win_ctx = win_p @ rv_h

        return 0.5 * (dense_ctx + win_ctx)

    heads = [_attend_head(h) for h in range(n_heads)]
    ctx = jnp.concatenate(heads, axis=-1)
    return ctx @ out_W.T


# ---------------------------------------------------------------------------
# Component 2: GRU update
# ---------------------------------------------------------------------------


def sg2_gru_update(
    input: jnp.ndarray,
    h_old: jnp.ndarray,
    W_z: jnp.ndarray,
    W_r: jnp.ndarray,
    W_h: jnp.ndarray,
    b_z: jnp.ndarray,
    b_r: jnp.ndarray,
    b_h: jnp.ndarray,
) -> jnp.ndarray:
    """Per-element GRU update for temporal gradient memory.

    Standard GRU equations:
        z = sigmoid(W_z @ [input, h_old] + b_z)     (update gate)
        r = sigmoid(W_r @ [input, h_old] + b_r)     (reset gate)
        h_tilde = tanh(W_h @ [input, r * h_old] + b_h)  (candidate)
        h_new = (1 - z) * h_old + z * h_tilde

    Args:
        input: GRU input [N, input_dim] (f32).
        h_old: Previous hidden state [N, hidden_dim] (f32).
        W_z: Update gate weights [hidden_dim, input_dim + hidden_dim] (f32).
        W_r: Reset gate weights [hidden_dim, input_dim + hidden_dim] (f32).
        W_h: Candidate weights [hidden_dim, input_dim + hidden_dim] (f32).
        b_z: Update gate bias [hidden_dim] (f32).
        b_r: Reset gate bias [hidden_dim] (f32).
        b_h: Candidate bias [hidden_dim] (f32).

    Returns:
        h_new: Updated hidden state [N, hidden_dim] (f32).
    """
    input_f32 = input.astype(ACCUM_DTYPE)
    h_f32 = h_old.astype(ACCUM_DTYPE)

    # Concatenate input and hidden state
    xh = jnp.concatenate([input_f32, h_f32], axis=-1)  # [N, in+hid]

    # Gates
    z = jax.nn.sigmoid(xh @ W_z.T + b_z)  # [N, hid]
    r = jax.nn.sigmoid(xh @ W_r.T + b_r)  # [N, hid]

    # Candidate with reset gate
    xrh = jnp.concatenate([input_f32, r * h_f32], axis=-1)
    h_tilde = jnp.tanh(xrh @ W_h.T + b_h)  # [N, hid]

    # Update
    h_new = (1.0 - z) * h_f32 + z * h_tilde

    return h_new


# ---------------------------------------------------------------------------
# Component 3: PEER product-key routing
# ---------------------------------------------------------------------------


def sg2_peer_routing(
    input: jnp.ndarray,
    query_Ws: jnp.ndarray,
    keys_A: jnp.ndarray,
    keys_B: jnp.ndarray,
    pk_dim: int,
    num_heads: int,
) -> jnp.ndarray:
    """Product-key routing for multi-head expert selection.

    For each head:
      1. Compute query = W_query @ input
      2. Split query into (q_a, q_b) halves
      3. Find argmax(q_a @ keys_A^T) and argmax(q_b @ keys_B^T)
      4. Expert index = idx_a * pk_dim + idx_b

    Args:
        input: Routing input [N, input_dim] (f32).
        query_Ws: Stacked query projections [num_heads, d_model, input_dim] (f32).
        keys_A: Product keys A [num_heads, pk_dim, d_model // 2] (f32).
        keys_B: Product keys B [num_heads, pk_dim, d_model // 2] (f32).
        pk_dim: Size of each product-key codebook.
        num_heads: Number of PEER heads.

    Returns:
        expert_indices: Selected expert indices [num_heads, N] (int32).
    """
    input_f32 = input.astype(ACCUM_DTYPE)
    half_d = query_Ws.shape[1] // 2

    def _route_one_head(h):
        # query: [N, d_model]
        query = input_f32 @ query_Ws[h].T
        q_a = query[:, :half_d]  # [N, d_model//2]
        q_b = query[:, half_d:]  # [N, d_model//2]

        # Product-key lookup: argmax of dot products
        scores_a = q_a @ keys_A[h].T  # [N, pk_dim]
        scores_b = q_b @ keys_B[h].T  # [N, pk_dim]
        idx_a = jnp.argmax(scores_a, axis=-1)  # [N]
        idx_b = jnp.argmax(scores_b, axis=-1)  # [N]

        return idx_a * pk_dim + idx_b  # [N]

    # Vectorise across heads
    expert_indices = jax.vmap(_route_one_head)(jnp.arange(num_heads))  # [num_heads, N]

    return expert_indices


# ---------------------------------------------------------------------------
# Component 4: Expert MLP evaluation
# ---------------------------------------------------------------------------


def sg2_expert_mlp(
    g: jnp.ndarray,
    expert_indices: jnp.ndarray,
    expert_W1: jnp.ndarray,
    expert_b1: jnp.ndarray,
    expert_W2: jnp.ndarray,
    expert_b2: jnp.ndarray,
    rescale: float,
    num_heads: int,
) -> jnp.ndarray:
    """Evaluate expert MLPs for routed gradient elements.

    For each head, each element selects one expert. The expert MLP:
        z = relu(W1 @ g + b1)
        out = W2 @ z + b2

    Outputs are averaged across heads and rescaled.

    Args:
        g: Gradient values [N] (f32).
        expert_indices: Selected expert indices [num_heads, N] (int32).
        expert_W1: Expert first layer weights [num_experts, expert_hidden, 1] (f32).
        expert_b1: Expert first layer biases [num_experts, expert_hidden] (f32).
        expert_W2: Expert second layer weights [num_experts, 1, expert_hidden] (f32).
        expert_b2: Expert second layer biases [num_experts, 1] (f32).
        rescale: Output rescaling factor.
        num_heads: Number of PEER heads.

    Returns:
        smart_grad: Corrected gradient [N] (f32).
    """
    g_f32 = g.astype(ACCUM_DTYPE)
    N = g_f32.shape[0]

    total_out = jnp.zeros(N, dtype=ACCUM_DTYPE)

    def _eval_one_head(h_idx):
        """Evaluate expert MLPs for a single head."""
        indices = expert_indices[h_idx]  # [N]

        # Gather expert weights for selected experts
        w1 = expert_W1[indices]  # [N, expert_hidden, 1]
        b1_val = expert_b1[indices]  # [N, expert_hidden]
        w2 = expert_W2[indices]  # [N, 1, expert_hidden]
        b2_val = expert_b2[indices]  # [N, 1]

        # MLP forward per element (batched via gathered weights)
        # z = relu(W1 @ g_i + b1)
        g_col = g_f32[:, None, None]  # [N, 1, 1]
        z = jax.nn.relu(jnp.squeeze(w1 @ g_col, axis=-1) + b1_val)  # [N, expert_hidden]
        # out = W2 @ z + b2
        z_col = z[:, :, None]  # [N, expert_hidden, 1]
        out = jnp.squeeze(w2 @ z_col, axis=(-2, -1)) + b2_val.squeeze(-1)  # [N]
        return out

    # Evaluate each head and accumulate
    for h in range(num_heads):
        total_out = total_out + _eval_one_head(h)

    total_out = total_out / num_heads

    # smart_grad = g + rescale * expert_correction
    smart_grad = g_f32 + rescale * total_out

    return smart_grad


# ---------------------------------------------------------------------------
# Component 5: Full SuperGrok v2 update step
# ---------------------------------------------------------------------------


def sg2_update(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    state: Dict[str, Any],
    meta_weights: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Full SuperGrok v2 update step orchestrating all components (pure JAX).

    Orchestrates (spec SS3b -- only the sequence mixer changed vs. the old
    Mamba-3 meta-net):
      1. Gradient clipping
      2. Sort by |gradient| magnitude
      3. Input projection
      4. CSA + HCA hybrid attention (csa_ctx local, hca_ctx global)
      5. Per-element GRU temporal memory
      6. 4-Head PEER product-key expert routing
      7. Expert MLP evaluation
      8. Gated gradient blending
      9. AdamW parameter update

    Args:
        params: Current parameters (bf16).
        grads: Gradients (bf16).
        state: Mutable optimizer state dict containing:
            - exp_avg: First moment estimate (f32).
            - exp_avg_sq: Second moment estimate (f32).
            - mu: Momentum buffer (f32).
            - sharpness: Per-element sharpness (f32).
            - gru_state: GRU hidden state [N, gru_hidden] (f32).
            - step: Current step (int).
          (Attention is stateless across steps -- no mamba_fwd/bwd_state.)
        meta_weights: Meta-network weights dict containing:
            - input_proj_W, input_proj_b
            - csa_q_W, csa_k_W, csa_v_W, csa_out_W, csa_compress_w,
              csa_idx_DQ, csa_idx_UQ, csa_idx_K   (CSA layer)
            - hca_q_W, hca_k_W, hca_v_W, hca_out_W (HCA layer)
            - gru_W_z, gru_W_r, gru_W_h, gru_b_z, gru_b_r, gru_b_h
            - peer_query_Ws: [num_heads, d_model, peer_input_dim]
            - product_keys_A, product_keys_B: [num_heads, pk_dim, d_model//2]
            - expert_W1, expert_b1, expert_W2, expert_b2
            - rescale, d_model, pk_dim, num_peer_heads, n_heads,
              csa_compress, csa_window, csa_topk, hca_compress, indexer_rank
        hyperparams: Optimizer hyperparameters dict containing:
            - layer_beta1, beta2, lr, wd, eps
            - lamb, ramp, gate_signal, grad_clip

    Returns:
        (updated_params, updated_state)
    """
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), NanPolicy.IGNORE)
    p = params.astype(ACCUM_DTYPE)

    # Unpack hyperparams
    layer_beta1 = hyperparams['layer_beta1']
    beta2 = hyperparams['beta2']
    lr = hyperparams['lr']
    wd = hyperparams['wd']
    eps = hyperparams['eps']
    lamb = hyperparams['lamb']
    ramp = hyperparams['ramp']
    gate_signal = hyperparams['gate_signal']
    grad_clip = hyperparams['grad_clip']

    # Unpack meta-network dimensions
    d_model = meta_weights['d_model']
    pk_dim = meta_weights['pk_dim']
    num_heads = meta_weights['num_peer_heads']
    rescale = meta_weights['rescale']
    gru_hidden = meta_weights.get('gru_hidden', 4)
    # CSA / HCA attention dims
    n_heads = meta_weights.get('n_heads', 2)
    csa_compress = meta_weights.get('csa_compress', 4)
    csa_window = meta_weights.get('csa_window', 8)
    csa_topk = meta_weights.get('csa_topk', 16)
    hca_compress = meta_weights.get('hca_compress', 128)
    indexer_rank = meta_weights.get('indexer_rank', 4)

    # Unpack state (attention is stateless -> no mamba scan state)
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    mu = state['mu']
    sharpness = state['sharpness']
    gru_state_val = state['gru_state']
    step = state['step']

    # --- 1. Gradient clipping ---
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # --- 2. Sort by |gradient| magnitude ---
    g_flat = g.reshape(-1)
    s_flat = sharpness.astype(ACCUM_DTYPE).reshape(-1)
    N = g_flat.shape[0]

    sort_idx = jnp.argsort(jnp.abs(g_flat))
    g_sorted = g_flat[sort_idx]
    s_sorted = s_flat[sort_idx]

    # --- 3. Input projection ---
    inp = jnp.stack([g_sorted, s_sorted], axis=-1)  # [N, 2]
    x = inp @ meta_weights['input_proj_W'].T + meta_weights['input_proj_b']  # [N, d_model]

    # --- 4. CSA + HCA hybrid attention (spec SS2, SS3b) ---
    # CSA -> local/fine context (replaces mamba fwd); HCA -> global coarse
    # context (replaces mamba bwd). Both operate on the sorted, projected seq x.
    csa_out = sg2_csa_attention(
        x,
        meta_weights['csa_q_W'], meta_weights['csa_k_W'],
        meta_weights['csa_v_W'], meta_weights['csa_out_W'],
        meta_weights['csa_compress_w'],
        meta_weights['csa_idx_DQ'], meta_weights['csa_idx_UQ'],
        meta_weights['csa_idx_K'],
        n_heads, csa_compress, csa_window, csa_topk,
    )  # [N, d_model]
    hca_out = sg2_hca_attention(
        x,
        meta_weights['hca_q_W'], meta_weights['hca_k_W'],
        meta_weights['hca_v_W'], meta_weights['hca_out_W'],
        n_heads, hca_compress, csa_window,
    )  # [N, d_model]

    # Unsort back to original element order
    unsort_idx = jnp.argsort(sort_idx)
    csa_ctx = csa_out[unsort_idx]  # [N, d_model]
    hca_ctx = hca_out[unsort_idx]  # [N, d_model]

    # --- 5. Per-element GRU temporal memory ---
    gru_input = jnp.concatenate([
        g_flat[:, None], s_flat[:, None],
        csa_ctx, hca_ctx,
    ], axis=-1)  # [N, 2 + 2*d_model]

    new_gru = sg2_gru_update(
        gru_input, gru_state_val,
        meta_weights['gru_W_z'], meta_weights['gru_W_r'], meta_weights['gru_W_h'],
        meta_weights['gru_b_z'], meta_weights['gru_b_r'], meta_weights['gru_b_h'],
    )

    # --- 6. PEER product-key expert routing ---
    peer_input = jnp.concatenate([
        new_gru, csa_ctx, hca_ctx,
        g_flat[:, None], s_flat[:, None],
    ], axis=-1)  # [N, gru_hidden + 2*d_model + 2]

    expert_indices = sg2_peer_routing(
        peer_input,
        meta_weights['peer_query_Ws'],
        meta_weights['product_keys_A'],
        meta_weights['product_keys_B'],
        pk_dim,
        num_heads,
    )  # [num_heads, N]

    # --- 7. Expert MLP evaluation ---
    smart_g_flat = sg2_expert_mlp(
        g_flat, expert_indices,
        meta_weights['expert_W1'], meta_weights['expert_b1'],
        meta_weights['expert_W2'], meta_weights['expert_b2'],
        rescale, num_heads,
    )  # [N]

    smart_g = smart_g_flat.reshape(g.shape)

    # --- 8. Gated gradient blending ---
    effective_g = g + ramp * lamb * gate_signal * (smart_g - g)

    # --- 9. AdamW parameter update ---
    new_exp_avg = layer_beta1 * exp_avg + (1.0 - layer_beta1) * effective_g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (effective_g * effective_g)

    bc1 = 1.0 - layer_beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = new_exp_avg / bc1
    v_hat = new_exp_avg_sq / bc2

    new_mu = layer_beta1 * mu + (1.0 - layer_beta1) * effective_g

    update = m_hat / (jnp.sqrt(v_hat) + eps)
    p = p - lr * (update + wd * p)

    # --- Build updated state (attention is stateless across steps) ---
    new_state = {
        'exp_avg': new_exp_avg,
        'exp_avg_sq': new_exp_avg_sq,
        'mu': new_mu,
        'sharpness': sharpness,
        'gru_state': new_gru,
        'step': step + 1,
    }

    return p.astype(PARAM_DTYPE), new_state


# ---------------------------------------------------------------------------
# Pallas TPU kernel (element-wise AdamW portion)
# ---------------------------------------------------------------------------


def _sg2_adamw_kernel_body(
    params_ref, effective_g_ref, exp_avg_ref, exp_avg_sq_ref, mu_ref,
    out_params_ref, out_exp_avg_ref, out_exp_avg_sq_ref, out_mu_ref,
    *, layer_beta1, beta2, lr, wd, eps, step,
):
    """Pallas kernel body for the AdamW portion of SuperGrok v2."""
    p = params_ref[...].astype(jnp.float32)
    eg = effective_g_ref[...]
    m = exp_avg_ref[...]
    v = exp_avg_sq_ref[...]
    mu_val = mu_ref[...]

    new_m = layer_beta1 * m + (1.0 - layer_beta1) * eg
    new_v = beta2 * v + (1.0 - beta2) * (eg * eg)
    new_mu = layer_beta1 * mu_val + (1.0 - layer_beta1) * eg

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


def sg2_pallas_kernel(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    state: Dict[str, Any],
    meta_weights: Dict[str, Any],
    hyperparams: Dict[str, Any],
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Pallas-accelerated SuperGrok v2 update for TPU.

    The meta-network (Mamba scan, GRU, PEER routing, expert MLP) involves
    matrix multiplications and complex control flow best handled by XLA.
    The final AdamW step is dispatched to a Pallas kernel when available.

    Falls back to pure JAX if Pallas is unavailable.
    """
    if not _HAS_PALLAS:
        return sg2_update(params, grads, state, meta_weights, hyperparams)

    # Run the meta-network pipeline in pure JAX (matmul-heavy, XLA-optimal)
    g = apply_nan_policy(grads.astype(ACCUM_DTYPE), NanPolicy.IGNORE)
    p = params.astype(ACCUM_DTYPE)

    layer_beta1 = hyperparams['layer_beta1']
    beta2 = hyperparams['beta2']
    lr = hyperparams['lr']
    wd = hyperparams['wd']
    eps = hyperparams['eps']
    lamb = hyperparams['lamb']
    ramp = hyperparams['ramp']
    gate_signal = hyperparams['gate_signal']
    grad_clip = hyperparams['grad_clip']

    d_model = meta_weights['d_model']
    pk_dim = meta_weights['pk_dim']
    num_heads = meta_weights['num_peer_heads']
    rescale = meta_weights['rescale']
    n_heads = meta_weights.get('n_heads', 2)
    csa_compress = meta_weights.get('csa_compress', 4)
    csa_window = meta_weights.get('csa_window', 8)
    csa_topk = meta_weights.get('csa_topk', 16)
    hca_compress = meta_weights.get('hca_compress', 128)
    indexer_rank = meta_weights.get('indexer_rank', 4)

    step = state['step']

    # Gradient clipping
    g_norm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
    clip_factor = jnp.minimum(1.0, grad_clip / g_norm)
    g = g * clip_factor

    # Sort, project, scan, GRU, route, expert eval (all pure JAX)
    g_flat = g.reshape(-1)
    s_flat = state['sharpness'].astype(ACCUM_DTYPE).reshape(-1)
    N = g_flat.shape[0]

    sort_idx = jnp.argsort(jnp.abs(g_flat))
    g_sorted = g_flat[sort_idx]
    s_sorted = s_flat[sort_idx]

    inp = jnp.stack([g_sorted, s_sorted], axis=-1)
    x = inp @ meta_weights['input_proj_W'].T + meta_weights['input_proj_b']

    csa_out = sg2_csa_attention(
        x,
        meta_weights['csa_q_W'], meta_weights['csa_k_W'],
        meta_weights['csa_v_W'], meta_weights['csa_out_W'],
        meta_weights['csa_compress_w'],
        meta_weights['csa_idx_DQ'], meta_weights['csa_idx_UQ'],
        meta_weights['csa_idx_K'],
        n_heads, csa_compress, csa_window, csa_topk,
    )
    hca_out = sg2_hca_attention(
        x,
        meta_weights['hca_q_W'], meta_weights['hca_k_W'],
        meta_weights['hca_v_W'], meta_weights['hca_out_W'],
        n_heads, hca_compress, csa_window,
    )

    unsort_idx = jnp.argsort(sort_idx)
    csa_ctx = csa_out[unsort_idx]
    hca_ctx = hca_out[unsort_idx]

    gru_input = jnp.concatenate([
        g_flat[:, None], s_flat[:, None], csa_ctx, hca_ctx,
    ], axis=-1)
    new_gru = sg2_gru_update(
        gru_input, state['gru_state'],
        meta_weights['gru_W_z'], meta_weights['gru_W_r'], meta_weights['gru_W_h'],
        meta_weights['gru_b_z'], meta_weights['gru_b_r'], meta_weights['gru_b_h'],
    )

    peer_input = jnp.concatenate([
        new_gru, csa_ctx, hca_ctx, g_flat[:, None], s_flat[:, None],
    ], axis=-1)
    expert_indices = sg2_peer_routing(
        peer_input, meta_weights['peer_query_Ws'],
        meta_weights['product_keys_A'], meta_weights['product_keys_B'],
        pk_dim, num_heads,
    )
    smart_g_flat = sg2_expert_mlp(
        g_flat, expert_indices,
        meta_weights['expert_W1'], meta_weights['expert_b1'],
        meta_weights['expert_W2'], meta_weights['expert_b2'],
        rescale, num_heads,
    )
    smart_g = smart_g_flat.reshape(g.shape)

    # Gated blending
    effective_g = g + ramp * lamb * gate_signal * (smart_g - g)

    # Dispatch AdamW to Pallas kernel
    kernel_fn = functools.partial(
        _sg2_adamw_kernel_body,
        layer_beta1=layer_beta1, beta2=beta2, lr=lr, wd=wd, eps=eps,
        step=step,
    )

    out_params, out_exp_avg, out_exp_avg_sq, out_mu = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(params.shape, PARAM_DTYPE),
            jax.ShapeDtypeStruct(state['exp_avg'].shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(state['exp_avg_sq'].shape, ACCUM_DTYPE),
            jax.ShapeDtypeStruct(state['mu'].shape, ACCUM_DTYPE),
        ],
        grid=(1,),
    )(params, effective_g.astype(ACCUM_DTYPE), state['exp_avg'],
      state['exp_avg_sq'], state['mu'])

    new_state = {
        'exp_avg': out_exp_avg,
        'exp_avg_sq': out_exp_avg_sq,
        'mu': out_mu,
        'sharpness': state['sharpness'],
        'gru_state': new_gru,
        'step': step + 1,
    }

    return out_params, new_state
