"""TPU/Pallas kernels for SuperGrok v2 -- Mamba-3 + PEER + GRU meta-net optimizer.

The most complex optimizer in the SuperGrok family. Combines:
  - Mamba-3 bidirectional selective state-space scan (via associative_scan)
  - 4-Head PEER product-key expert routing
  - Per-element GRU temporal memory
  - AdamW with sigmoid gating and adaptive scheduling

Key TPU optimisations:
  - jax.lax.associative_scan for O(log N) parallel Mamba scans
  - jax.vmap for expert MLP evaluation (vectorised across experts)
  - Pure JAX for matmul-heavy paths (XLA -> MXU)
  - Pallas kernels for element-wise fused AdamW

BF16 parameters, FP32 accumulators throughout.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

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
# Component 1: Mamba selective scan via associative_scan
# ---------------------------------------------------------------------------


def sg2_mamba_scan(
    x: jnp.ndarray,
    state: jnp.ndarray,
    A_log: jnp.ndarray,
    B_proj: jnp.ndarray,
    C_proj: jnp.ndarray,
    D: jnp.ndarray,
    dt_proj_W: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Mamba-3 selective state-space scan using associative_scan.

    Uses the 2x2 affine operator for the parallel scan:
        (M, b) where M = diag(exp(A * dt)), b = B * x
        Combine: (M1, b1) o (M2, b2) = (M1 * M2, M2 * b1 + b2)

    This parallelises the inherently sequential SSM scan on TPU,
    achieving O(log N) depth instead of O(N).

    Args:
        x: Input sequence [N, d_inner] (f32). N = flattened param elements.
        state: Previous SSM hidden state [d_inner, d_state] (f32).
        A_log: Log of state transition matrix [d_inner, d_state] (f32).
        B_proj: B projection weights [d_state, d_inner] (f32).
        C_proj: C projection weights [d_state, d_inner] (f32).
        D: Skip connection parameter [d_inner] (f32).
        dt_proj_W: Timestep projection weights [d_inner, d_inner] (f32).

    Returns:
        (output, new_state) where output is [N, d_inner] and
        new_state is [d_inner, d_state].
    """
    N = x.shape[0]
    d_inner = x.shape[1] if x.ndim > 1 else D.shape[0]
    d_state = A_log.shape[1]

    # Ensure x is 2D: [N, d_inner]
    x_f32 = x.astype(ACCUM_DTYPE)
    if x_f32.ndim == 1:
        x_f32 = x_f32[:, None]  # [N, 1] -- will broadcast

    # Compute dt via softplus of projection
    dt = jnp.log1p(jnp.exp(x_f32 @ dt_proj_W.T))  # [N, d_inner]

    # Compute B and C from projections
    B = x_f32 @ B_proj.T  # [N, d_state]
    C = x_f32 @ C_proj.T  # [N, d_state]

    # Discretisation
    A = jnp.exp(A_log)  # [d_inner, d_state]
    # A_bar = exp(dt * A): [N, d_inner, d_state]
    A_bar = jnp.exp(dt[:, :, None] * A[None, :, :])
    # B_bar = dt * B: [N, d_inner, d_state]
    B_bar = dt[:, :, None] * B[:, None, :]
    # x_bar = B_bar * x: [N, d_inner, d_state]
    x_bar = B_bar * x_f32[:, :, None]

    # Associative scan along sequence axis (axis=0)
    # Operator: (a1, b1) o (a2, b2) = (a1 * a2, a2 * b1 + b2)
    def _combine(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        incoming: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a1, b1 = carry
        a2, b2 = incoming
        return a1 * a2, a2 * b1 + b2

    # Incorporate initial state into the first element
    # h[0] = A_bar[0] * state + x_bar[0]
    init_contrib = A_bar[0] * state[None, :, :]  # [1, d_inner, d_state] -> broadcast
    x_bar_adj = x_bar.at[0].add(init_contrib.squeeze(0))

    _, h_scan = lax.associative_scan(
        _combine, (A_bar, x_bar_adj), axis=0,
    )  # h_scan: [N, d_inner, d_state]

    # Output: y[n] = sum_s(C[n, s] * h[n, :, s])
    y = jnp.sum(C[:, None, :] * h_scan, axis=-1)  # [N, d_inner]

    # Skip connection
    y = y + D[None, :] * x_f32

    # New state = last hidden state
    new_state = h_scan[-1]  # [d_inner, d_state]

    return y, new_state


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

    Orchestrates:
      1. Gradient clipping
      2. Sort by |gradient| magnitude
      3. Input projection
      4. Bidirectional Mamba-3 scan (forward + backward)
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
            - mamba_fwd_state: Forward SSM state [d_inner, d_state] (f32).
            - mamba_bwd_state: Backward SSM state [d_inner, d_state] (f32).
            - step: Current step (int).
        meta_weights: Meta-network weights dict containing:
            - input_proj_W, input_proj_b
            - mamba_fwd_*: Forward Mamba weights
            - mamba_bwd_*: Backward Mamba weights
            - gru_W_z, gru_W_r, gru_W_h, gru_b_z, gru_b_r, gru_b_h
            - peer_query_Ws: [num_heads, d_model, peer_input_dim]
            - product_keys_A, product_keys_B: [num_heads, pk_dim, d_model//2]
            - expert_W1, expert_b1, expert_W2, expert_b2
            - rescale, d_model, d_state, d_inner, pk_dim, num_peer_heads
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
    d_state = meta_weights['d_state']
    d_inner = meta_weights['d_inner']
    pk_dim = meta_weights['pk_dim']
    num_heads = meta_weights['num_peer_heads']
    rescale = meta_weights['rescale']
    gru_hidden = meta_weights.get('gru_hidden', 4)

    # Unpack state
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    mu = state['mu']
    sharpness = state['sharpness']
    gru_state_val = state['gru_state']
    mamba_fwd_state = state['mamba_fwd_state']
    mamba_bwd_state = state['mamba_bwd_state']
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

    # --- 4. Bidirectional Mamba-3 scan ---
    # Forward scan
    fwd_out, new_fwd_state = sg2_mamba_scan(
        x, mamba_fwd_state,
        meta_weights['mamba_fwd_A_log'],
        meta_weights['mamba_fwd_B_proj'],
        meta_weights['mamba_fwd_C_proj'],
        meta_weights['mamba_fwd_D'],
        meta_weights['mamba_fwd_dt_proj_W'],
    )

    # Backward scan (flip input, flip output)
    bwd_out, new_bwd_state = sg2_mamba_scan(
        x[::-1], mamba_bwd_state,
        meta_weights['mamba_bwd_A_log'],
        meta_weights['mamba_bwd_B_proj'],
        meta_weights['mamba_bwd_C_proj'],
        meta_weights['mamba_bwd_D'],
        meta_weights['mamba_bwd_dt_proj_W'],
    )
    bwd_out = bwd_out[::-1]

    # Unsort back to original element order
    unsort_idx = jnp.argsort(sort_idx)
    fwd_ctx = fwd_out[unsort_idx]  # [N, d_model]
    bwd_ctx = bwd_out[unsort_idx]  # [N, d_model]

    # --- 5. Per-element GRU temporal memory ---
    gru_input = jnp.concatenate([
        g_flat[:, None], s_flat[:, None],
        fwd_ctx, bwd_ctx,
    ], axis=-1)  # [N, 2 + 2*d_model]

    new_gru = sg2_gru_update(
        gru_input, gru_state_val,
        meta_weights['gru_W_z'], meta_weights['gru_W_r'], meta_weights['gru_W_h'],
        meta_weights['gru_b_z'], meta_weights['gru_b_r'], meta_weights['gru_b_h'],
    )

    # --- 6. PEER product-key expert routing ---
    peer_input = jnp.concatenate([
        new_gru, fwd_ctx, bwd_ctx,
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

    # --- Build updated state ---
    new_state = {
        'exp_avg': new_exp_avg,
        'exp_avg_sq': new_exp_avg_sq,
        'mu': new_mu,
        'sharpness': sharpness,
        'gru_state': new_gru,
        'mamba_fwd_state': new_fwd_state,
        'mamba_bwd_state': new_bwd_state,
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

    fwd_out, new_fwd_state = sg2_mamba_scan(
        x, state['mamba_fwd_state'],
        meta_weights['mamba_fwd_A_log'], meta_weights['mamba_fwd_B_proj'],
        meta_weights['mamba_fwd_C_proj'], meta_weights['mamba_fwd_D'],
        meta_weights['mamba_fwd_dt_proj_W'],
    )
    bwd_out, new_bwd_state = sg2_mamba_scan(
        x[::-1], state['mamba_bwd_state'],
        meta_weights['mamba_bwd_A_log'], meta_weights['mamba_bwd_B_proj'],
        meta_weights['mamba_bwd_C_proj'], meta_weights['mamba_bwd_D'],
        meta_weights['mamba_bwd_dt_proj_W'],
    )
    bwd_out = bwd_out[::-1]

    unsort_idx = jnp.argsort(sort_idx)
    fwd_ctx = fwd_out[unsort_idx]
    bwd_ctx = bwd_out[unsort_idx]

    gru_input = jnp.concatenate([
        g_flat[:, None], s_flat[:, None], fwd_ctx, bwd_ctx,
    ], axis=-1)
    new_gru = sg2_gru_update(
        gru_input, state['gru_state'],
        meta_weights['gru_W_z'], meta_weights['gru_W_r'], meta_weights['gru_W_h'],
        meta_weights['gru_b_z'], meta_weights['gru_b_r'], meta_weights['gru_b_h'],
    )

    peer_input = jnp.concatenate([
        new_gru, fwd_ctx, bwd_ctx, g_flat[:, None], s_flat[:, None],
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
        'mamba_fwd_state': new_fwd_state,
        'mamba_bwd_state': new_bwd_state,
        'step': step + 1,
    }

    return out_params, new_state
