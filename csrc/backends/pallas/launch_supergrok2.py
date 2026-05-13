"""Pallas/TPU launch glue for SuperGrok v2.

Algorithm: csrc/algorithms/supergrok2.h
Self-contained — collapses the entire functional JAX rewrite that
previously lived in supergrok2_jax_tpu/ (supergrok2_jax.py,
mamba3_peer_metanet_jax.py, scan.py, gru.py, peer.py, bilevel.py,
quantization_jax.py) into one file. Matches the policy that each
optimizer launch is fully self-contained, even if the resulting file
is large.

Architectural map (all in this file):
  1. Mamba-3 scan primitives        (parallel associative scan)
  2. Per-element GRU cell           (temporal memory)
  3. Multi-head PEER routing         (soft and hard variants)
  4. Composed meta-net forward pass (sort, scan, GRU, route, skip)
  5. SuperGrok v2 per-parameter step
  6. Bilevel meta-net optimization step
  7. INT8/INT4 quantization helpers
  8. v5p Pallas kernel re-exports
  9. fused-TU contract: launch_supergrok2_step
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from jax import lax
from typing import Any, Callable, NamedTuple, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
#  Section 1 — Mamba-3 scan
# ════════════════════════════════════════════════════════════════════════════


class MambaScanWeights(NamedTuple):
    """Weights for one direction of the Mamba-3 scan."""
    in_proj_W: jnp.ndarray      # [2*d_inner, d_model]
    dt_proj_W: jnp.ndarray      # [d_inner, d_inner]
    dt_proj_b: jnp.ndarray      # [d_inner]
    B_proj_W: jnp.ndarray       # [d_state, d_inner]
    C_proj_W: jnp.ndarray       # [d_state, d_inner]
    A_log: jnp.ndarray          # [d_inner, d_state]
    D: jnp.ndarray              # [d_inner]
    rope_freq: jnp.ndarray      # [d_inner, d_state//2]
    out_proj_W: jnp.ndarray     # [d_model, d_inner]


def _build_affine_transforms(
    dt: jnp.ndarray,
    x_branch: jnp.ndarray,
    B: jnp.ndarray,
    A_log: jnp.ndarray,
    rope_freq: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build per-timestep affine transforms for the associative scan.

    For each timestep t, state pair p, and inner dimension j, the recurrence
    h_pair[t] = M[t] @ h_pair[t-1] + b[t] is encoded as (M[t], b[t]).
    """
    A = -jnp.exp(A_log)
    half_dtA = dt[:, :, None] * A[None, :, :] / 2.0
    A_bar = (1.0 + half_dtA) / (1.0 - half_dtA + 1e-8)
    B_bar = dt[:, :, None] * B[:, None, :]

    phase = dt[:, :, None] * rope_freq[None, :, :]
    cos_p = jnp.cos(phase)
    sin_p = jnp.sin(phase)

    A_bar_e = A_bar[:, :, 0::2]
    A_bar_o = A_bar[:, :, 1::2]

    M_00 = A_bar_e * cos_p
    M_01 = -A_bar_e * sin_p
    M_10 = A_bar_o * sin_p
    M_11 = A_bar_o * cos_p
    Ms = jnp.stack([
        jnp.stack([M_00, M_01], axis=-1),
        jnp.stack([M_10, M_11], axis=-1),
    ], axis=-2)

    B_bar_e = B_bar[:, :, 0::2]
    B_bar_o = B_bar[:, :, 1::2]
    bs = jnp.stack([
        B_bar_e * x_branch[:, :, None],
        B_bar_o * x_branch[:, :, None],
    ], axis=-1)
    return Ms, bs


def _associative_combine(left, right):
    """Compose two affine transforms: (M_r @ M_l, M_r @ b_l + b_r)."""
    M_l, b_l = left
    M_r, b_r = right
    M_out = jnp.einsum('...ij,...jk->...ik', M_r, M_l)
    b_out = jnp.einsum('...ij,...j->...i', M_r, b_l) + b_r
    return M_out, b_out


def mamba3_scan(
    x_sorted: jnp.ndarray,
    weights: MambaScanWeights,
    initial_state: Optional[jnp.ndarray] = None,
    reverse: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Bidirectional Mamba-3 scan using JAX associative_scan.

    Mathematical equivalence to the CUDA Blelloch scan:
      h[t] = A_bar[t] * RoPE(h[t-1]) + B_bar[t] * x[t]
      y[t] = sum_s(h[t,:,s] * C[t,s]) * silu(z[t]) + D * x_branch[t]
      output[t] = y[t] @ out_proj.T
    """
    N, d_model = x_sorted.shape
    d_inner = weights.in_proj_W.shape[0] // 2
    d_state = weights.A_log.shape[1]
    half_d_state = d_state // 2

    if reverse:
        x_sorted = x_sorted[::-1]

    xz = x_sorted @ weights.in_proj_W.T
    x_branch = xz[:, :d_inner]
    z_branch = xz[:, d_inner:]

    dt_raw = x_branch @ weights.dt_proj_W.T + weights.dt_proj_b
    dt = jax.nn.softplus(dt_raw)
    B = x_branch @ weights.B_proj_W.T
    C = x_branch @ weights.C_proj_W.T

    Ms, bs = _build_affine_transforms(dt, x_branch, B, weights.A_log, weights.rope_freq)

    cumulative_M, cumulative_b = lax.associative_scan(
        _associative_combine, (Ms, bs), axis=0)

    if initial_state is not None:
        h_init_pairs = initial_state.reshape(d_inner, half_d_state, 2)
    else:
        h_init_pairs = jnp.zeros((d_inner, half_d_state, 2))

    h_pairs = (
        jnp.einsum('...ij,...j->...i', cumulative_M, h_init_pairs[None, :, :, :])
        + cumulative_b
    )

    h_even = h_pairs[..., 0]
    h_odd = h_pairs[..., 1]
    h = jnp.zeros((N, d_inner, d_state))
    h = h.at[:, :, 0::2].set(h_even)
    h = h.at[:, :, 1::2].set(h_odd)

    y = jnp.sum(h * C[:, None, :], axis=-1)
    y = y * jax.nn.silu(z_branch) + weights.D[None, :] * x_branch
    output = y @ weights.out_proj_W.T

    final_state = h[-1]

    if reverse:
        output = output[::-1]

    return output, final_state


# ════════════════════════════════════════════════════════════════════════════
#  Section 2 — Per-element GRU
# ════════════════════════════════════════════════════════════════════════════


class GRUWeights(NamedTuple):
    """Weights for the per-element GRU."""
    Wz: jnp.ndarray
    bz: jnp.ndarray
    Wr: jnp.ndarray
    br: jnp.ndarray
    Wh: jnp.ndarray
    bh: jnp.ndarray


def mini_gru(
    x: jnp.ndarray,
    h_old: jnp.ndarray,
    weights: GRUWeights,
) -> jnp.ndarray:
    """Per-element GRU update (no scan across N; embarrassingly parallel)."""
    xh = jnp.concatenate([x, h_old], axis=-1)
    z = jax.nn.sigmoid(xh @ weights.Wz.T + weights.bz)
    r = jax.nn.sigmoid(xh @ weights.Wr.T + weights.br)
    xrh = jnp.concatenate([x, r * h_old], axis=-1)
    h_tilde = jnp.tanh(xrh @ weights.Wh.T + weights.bh)
    h_new = (1 - z) * h_old + z * h_tilde
    return h_new


# ════════════════════════════════════════════════════════════════════════════
#  Section 3 — Multi-head PEER routing
# ════════════════════════════════════════════════════════════════════════════


def peer_expert_forward(
    peer_input: jnp.ndarray,
    grad: jnp.ndarray,
    peer_query_Ws: jnp.ndarray,
    prod_keys_A: jnp.ndarray,
    prod_keys_B: jnp.ndarray,
    expert_W1: jnp.ndarray,
    expert_b1: jnp.ndarray,
    expert_W2: jnp.ndarray,
    expert_b2: jnp.ndarray,
    num_heads: int,
    pk_dim: int,
    num_experts: int,
    topk: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-head PEER routing with product-key expert selection (soft, bilevel)."""
    N = peer_input.shape[0]
    half_d = peer_query_Ws.shape[1] // 2
    num_active = topk * topk

    total_out = jnp.zeros(N)
    expert_counts = jnp.zeros(num_experts, dtype=jnp.int32)

    def _head_forward(h, carry):
        total_out, expert_counts = carry

        query = peer_input @ peer_query_Ws[h].T
        q_a = query[:, :half_d]
        q_b = query[:, half_d:]

        scores_a = q_a @ prod_keys_A[h].T
        scores_b = q_b @ prod_keys_B[h].T

        top_a_vals, top_a_idx = jax.lax.top_k(scores_a, topk)
        top_b_vals, top_b_idx = jax.lax.top_k(scores_b, topk)

        soft_a = jax.nn.softmax(top_a_vals * 10.0, axis=-1)
        soft_b = jax.nn.softmax(top_b_vals * 10.0, axis=-1)

        expert_idx = (
            top_a_idx[:, :, None] * pk_dim + top_b_idx[:, None, :]
        ).reshape(N, num_active)
        routing_weights = (
            soft_a[:, :, None] * soft_b[:, None, :]
        ).reshape(N, num_active)

        ew1 = expert_W1[expert_idx]
        eb1 = expert_b1[expert_idx]
        ew2 = expert_W2[expert_idx]
        eb2 = expert_b2[expert_idx]

        g_exp = grad[:, None, None, None]
        z_hidden = jax.nn.relu((ew1 * g_exp).squeeze(-1) + eb1)
        out = (
            jnp.matmul(ew2, z_hidden[..., None]).squeeze(-1).squeeze(-1)
            + eb2.squeeze(-1)
        )

        head_out = (routing_weights * out).sum(axis=1)
        total_out = total_out + head_out
        expert_counts = expert_counts.at[expert_idx.reshape(-1)].add(1)
        return total_out, expert_counts

    for h in range(num_heads):
        total_out, expert_counts = _head_forward(h, (total_out, expert_counts))

    return total_out / num_heads, expert_counts


def peer_expert_forward_hard(
    peer_input: jnp.ndarray,
    grad: jnp.ndarray,
    peer_query_Ws: jnp.ndarray,
    prod_keys_A: jnp.ndarray,
    prod_keys_B: jnp.ndarray,
    expert_W1: jnp.ndarray,
    expert_b1: jnp.ndarray,
    expert_W2: jnp.ndarray,
    expert_b2: jnp.ndarray,
    num_heads: int,
    pk_dim: int,
    num_experts: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Hard PEER routing (argmax, non-differentiable). Used outside bilevel."""
    N = peer_input.shape[0]
    half_d = peer_query_Ws.shape[1] // 2

    total_out = jnp.zeros(N)
    expert_counts = jnp.zeros(num_experts, dtype=jnp.int32)

    for h in range(num_heads):
        query = peer_input @ peer_query_Ws[h].T
        q_a = query[:, :half_d]
        q_b = query[:, half_d:]

        idx_a = jnp.argmax(q_a @ prod_keys_A[h].T, axis=-1)
        idx_b = jnp.argmax(q_b @ prod_keys_B[h].T, axis=-1)
        expert_idx = idx_a * pk_dim + idx_b

        W1 = expert_W1[expert_idx]
        b1 = expert_b1[expert_idx]
        W2 = expert_W2[expert_idx]
        b2 = expert_b2[expert_idx]

        g = grad[:, None, None]
        z = jax.nn.relu(jnp.matmul(W1, g).squeeze(-1) + b1)
        out = jnp.matmul(W2, z[:, :, None]).squeeze(-1).squeeze(-1) + b2.squeeze(-1)

        total_out = total_out + out
        expert_counts = expert_counts.at[expert_idx].add(1)

    return total_out / num_heads, expert_counts


# ════════════════════════════════════════════════════════════════════════════
#  Section 4 — Meta-net composition
# ════════════════════════════════════════════════════════════════════════════


class MetaNetWeights(NamedTuple):
    """All weights for the SuperGrok v2 meta-net."""
    input_proj_W: jnp.ndarray   # [d_model, 2]
    input_proj_b: jnp.ndarray   # [d_model]
    mamba_fwd: MambaScanWeights
    mamba_bwd: MambaScanWeights
    gru: GRUWeights
    peer_query_Ws: jnp.ndarray
    prod_keys_A: jnp.ndarray
    prod_keys_B: jnp.ndarray
    expert_W1: jnp.ndarray
    expert_b1: jnp.ndarray
    expert_W2: jnp.ndarray
    expert_b2: jnp.ndarray
    rescale: jnp.ndarray
    expert_counts: jnp.ndarray


class MetaNetConfig(NamedTuple):
    """Configuration for the meta-net (static, not traced by JAX)."""
    d_model: int = 8
    d_state: int = 16
    d_inner: int = 16
    num_peer_heads: int = 4
    num_experts: int = 144
    expert_hidden: int = 16
    gru_hidden: int = 4
    pk_dim: int = 12
    rescale: float = 0.1


def init_meta_weights(
    config: MetaNetConfig,
    key: jnp.ndarray,
) -> MetaNetWeights:
    """Initialize meta-net weights with the same scheme as PyTorch."""
    d_model = config.d_model
    d_state = config.d_state
    d_inner = config.d_inner
    num_heads = config.num_peer_heads
    num_experts = config.num_experts
    expert_hidden = config.expert_hidden
    gru_hidden = config.gru_hidden
    pk_dim = config.pk_dim

    keys = jax.random.split(key, 50)
    ki = iter(range(50))

    def _k():
        return keys[next(ki)]

    def _linear(k, out_dim, in_dim):
        bound = 1.0 / math.sqrt(in_dim)
        return jax.random.uniform(k, (out_dim, in_dim), minval=-bound, maxval=bound)

    def _bias(k, dim, in_dim):
        bound = 1.0 / math.sqrt(in_dim)
        return jax.random.uniform(k, (dim,), minval=-bound, maxval=bound)

    input_proj_W = _linear(_k(), d_model, 2)
    input_proj_b = _bias(_k(), d_model, 2)

    mamba_fwd = MambaScanWeights(
        in_proj_W=_linear(_k(), 2 * d_inner, d_model),
        dt_proj_W=_linear(_k(), d_inner, d_inner),
        dt_proj_b=_bias(_k(), d_inner, d_inner),
        B_proj_W=_linear(_k(), d_state, d_inner),
        C_proj_W=_linear(_k(), d_state, d_inner),
        A_log=jnp.log(jnp.linspace(1, d_state, d_state))[None, :].repeat(d_inner, axis=0),
        D=jnp.ones(d_inner),
        rope_freq=jax.random.normal(_k(), (d_inner, d_state // 2)) * 0.01,
        out_proj_W=_linear(_k(), d_model, d_inner),
    )

    mamba_bwd = MambaScanWeights(
        in_proj_W=_linear(_k(), 2 * d_inner, d_model),
        dt_proj_W=_linear(_k(), d_inner, d_inner),
        dt_proj_b=_bias(_k(), d_inner, d_inner),
        B_proj_W=_linear(_k(), d_state, d_inner),
        C_proj_W=_linear(_k(), d_state, d_inner),
        A_log=jnp.log(jnp.linspace(1, d_state, d_state))[None, :].repeat(d_inner, axis=0),
        D=jnp.ones(d_inner),
        rope_freq=jax.random.normal(_k(), (d_inner, d_state // 2)) * 0.01,
        out_proj_W=_linear(_k(), d_model, d_inner),
    )

    gru_input_dim = 2 + 2 * d_model
    gru_total_dim = gru_input_dim + gru_hidden
    gru = GRUWeights(
        Wz=_linear(_k(), gru_hidden, gru_total_dim),
        bz=jnp.zeros(gru_hidden),
        Wr=_linear(_k(), gru_hidden, gru_total_dim),
        br=jnp.zeros(gru_hidden),
        Wh=_linear(_k(), gru_hidden, gru_total_dim),
        bh=jnp.zeros(gru_hidden),
    )

    peer_input_dim = gru_hidden + 2 * d_model + 2
    peer_query_Ws = jnp.stack([
        _linear(_k(), d_model, peer_input_dim)
        for _ in range(num_heads)
    ])
    prod_keys_A = jax.random.normal(_k(), (num_heads, pk_dim, d_model // 2)) * 0.02
    prod_keys_B = jax.random.normal(_k(), (num_heads, pk_dim, d_model // 2)) * 0.02

    expert_W1 = jax.random.normal(_k(), (num_experts, expert_hidden, 1)) * 0.02
    expert_b1 = jnp.zeros((num_experts, expert_hidden))
    expert_W2 = jax.random.normal(_k(), (num_experts, 1, expert_hidden)) * 0.02
    expert_b2 = jnp.zeros((num_experts, 1))

    return MetaNetWeights(
        input_proj_W=input_proj_W,
        input_proj_b=input_proj_b,
        mamba_fwd=mamba_fwd,
        mamba_bwd=mamba_bwd,
        gru=gru,
        peer_query_Ws=peer_query_Ws,
        prod_keys_A=prod_keys_A,
        prod_keys_B=prod_keys_B,
        expert_W1=expert_W1,
        expert_b1=expert_b1,
        expert_W2=expert_W2,
        expert_b2=expert_b2,
        rescale=jnp.array(config.rescale),
        expert_counts=jnp.zeros(num_experts, dtype=jnp.int32),
    )


def meta_net_forward(
    grad: jnp.ndarray,
    sharpness: jnp.ndarray,
    gru_state: jnp.ndarray,
    mamba_fwd_state: Optional[jnp.ndarray],
    mamba_bwd_state: Optional[jnp.ndarray],
    meta_weights: MetaNetWeights,
    config: MetaNetConfig,
    use_soft_routing: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full SuperGrok v2 meta-net forward pass.

    Equivalent to PyTorch Mamba3PEERMetaNet.forward (hard) or
    forward_for_bilevel (soft).
    """
    N = grad.shape[0]
    g = grad.reshape(-1).astype(jnp.float32)
    s = sharpness.reshape(-1).astype(jnp.float32)

    sort_idx = jnp.argsort(jnp.abs(g))
    g_sorted = g[sort_idx]
    s_sorted = s[sort_idx]

    inp = jnp.stack([g_sorted, s_sorted], axis=-1)
    x = inp @ meta_weights.input_proj_W.T + meta_weights.input_proj_b

    fwd_out, new_fwd = mamba3_scan(x, meta_weights.mamba_fwd, mamba_fwd_state, reverse=False)
    bwd_out, new_bwd = mamba3_scan(x, meta_weights.mamba_bwd, mamba_bwd_state, reverse=True)

    unsort_idx = jnp.argsort(sort_idx)
    fwd_ctx = fwd_out[unsort_idx]
    bwd_ctx = bwd_out[unsort_idx]

    gru_input = jnp.concatenate([
        g[:, None], s[:, None], fwd_ctx, bwd_ctx
    ], axis=-1)
    new_gru = mini_gru(gru_input, gru_state.astype(jnp.float32), meta_weights.gru)

    peer_input = jnp.concatenate([
        new_gru, fwd_ctx, bwd_ctx, g[:, None], s[:, None]
    ], axis=-1)

    if use_soft_routing:
        expert_out, expert_counts = peer_expert_forward(
            peer_input, g,
            meta_weights.peer_query_Ws, meta_weights.prod_keys_A,
            meta_weights.prod_keys_B,
            meta_weights.expert_W1, meta_weights.expert_b1,
            meta_weights.expert_W2, meta_weights.expert_b2,
            config.num_peer_heads, config.pk_dim, config.num_experts,
        )
    else:
        expert_out, expert_counts = peer_expert_forward_hard(
            peer_input, g,
            meta_weights.peer_query_Ws, meta_weights.prod_keys_A,
            meta_weights.prod_keys_B,
            meta_weights.expert_W1, meta_weights.expert_b1,
            meta_weights.expert_W2, meta_weights.expert_b2,
            config.num_peer_heads, config.pk_dim, config.num_experts,
        )

    smart_grad = g + meta_weights.rescale * expert_out
    return smart_grad.reshape(grad.shape), new_gru, new_fwd, new_bwd, expert_counts


# ════════════════════════════════════════════════════════════════════════════
#  Section 5 — SuperGrok v2 optimizer step
# ════════════════════════════════════════════════════════════════════════════


class PerParamState(NamedTuple):
    """Per-parameter optimizer state. JAX-compatible pytree."""
    exp_avg: jnp.ndarray
    exp_avg_sq: jnp.ndarray
    mu: jnp.ndarray
    sharpness: jnp.ndarray
    gru_state: jnp.ndarray
    mamba_fwd_state: jnp.ndarray
    mamba_bwd_state: jnp.ndarray
    step_count: jnp.ndarray


class SuperGrok2State(NamedTuple):
    """Full optimizer state for all parameters."""
    param_states: list
    global_step: jnp.ndarray
    cached_train_acc: jnp.ndarray


class OptimizerConfig(NamedTuple):
    """SuperGrok v2 hyperparameters (static, not traced by JAX)."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1.0
    alpha_init: float = 0.98
    lamb: float = 2.0
    gamma: float = 0.1
    kappa: float = 0.1
    warmup_steps: int = 100
    warmup_ramp: int = 100
    gradient_clipping: float = 1.0
    gate_scale: float = 20.0
    gate_thresh: float = 0.8
    wd_ramp: float = 4.0
    wd_scale: float = 20.0
    wd_thresh: float = 0.9


def init_per_param_state(
    param: jnp.ndarray,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
) -> PerParamState:
    """Initialize optimizer state for one parameter."""
    N = param.size
    return PerParamState(
        exp_avg=jnp.zeros(N, dtype=jnp.float32),
        exp_avg_sq=jnp.zeros(N, dtype=jnp.float32),
        mu=jnp.zeros(N, dtype=jnp.float32),
        sharpness=jnp.zeros(N, dtype=jnp.float32),
        gru_state=jnp.zeros((N, meta_config.gru_hidden), dtype=jnp.float32),
        mamba_fwd_state=jnp.zeros(
            (meta_config.d_inner, meta_config.d_state), dtype=jnp.float32),
        mamba_bwd_state=jnp.zeros(
            (meta_config.d_inner, meta_config.d_state), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def init_state(
    params: Any,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
) -> SuperGrok2State:
    """Initialize full optimizer state for all parameters."""
    params_flat, _ = jax.tree.flatten(params)
    param_states = [
        init_per_param_state(p, config, meta_config) for p in params_flat
    ]
    return SuperGrok2State(
        param_states=param_states,
        global_step=jnp.array(0, dtype=jnp.int32),
        cached_train_acc=jnp.array(0.0, dtype=jnp.float32),
    )


def _sigmoid(scale: float, value: float, thresh: float) -> float:
    return 1.0 / (1.0 + jnp.exp(-scale * (value - thresh)))


def _get_ramp_factor(
    global_step: jnp.ndarray,
    warmup_steps: int,
    warmup_ramp: int,
) -> jnp.ndarray:
    elapsed = jnp.maximum(global_step - warmup_steps, 0)
    return jnp.where(
        global_step <= warmup_steps,
        0.0,
        jnp.minimum(elapsed / jnp.maximum(warmup_ramp, 1), 1.0),
    )


def _get_effective_wd(
    base_wd: float,
    train_acc: jnp.ndarray,
    wd_ramp: float,
    wd_scale: float,
    wd_thresh: float,
) -> jnp.ndarray:
    sigmoid_val = _sigmoid(wd_scale, train_acc, wd_thresh)
    return base_wd * (1.0 + wd_ramp * sigmoid_val)


def _update_single_param(
    param: jnp.ndarray,
    grad: jnp.ndarray,
    ps: PerParamState,
    meta_weights: MetaNetWeights,
    meta_config: MetaNetConfig,
    alpha_mu: float,
    lamb_eff: jnp.ndarray,
    beta1: float,
    beta2: float,
    lr: float,
    wd_eff: jnp.ndarray,
    eps: float,
    gradient_clipping: float,
) -> Tuple[jnp.ndarray, PerParamState]:
    """Update a single parameter using the SuperGrok v2 algorithm."""
    step = ps.step_count + 1
    g = grad.reshape(-1).astype(jnp.float32)

    grad_norm = jnp.linalg.norm(g)
    g = jnp.where(
        grad_norm > gradient_clipping,
        g * (gradient_clipping / (grad_norm + 1e-12)),
        g,
    )
    g = jnp.where(jnp.isfinite(g), g, 0.0)

    smart_grad, new_gru, new_fwd, new_bwd, _ = meta_net_forward(
        g, ps.sharpness, ps.gru_state,
        ps.mamba_fwd_state, ps.mamba_bwd_state,
        meta_weights, meta_config,
        use_soft_routing=False,
    )

    new_mu = alpha_mu * ps.mu + (1.0 - alpha_mu) * g
    effective_grad = smart_grad.reshape(-1) + lamb_eff * new_mu

    new_exp_avg = beta1 * ps.exp_avg + (1 - beta1) * effective_grad
    new_exp_avg_sq = beta2 * ps.exp_avg_sq + (1 - beta2) * effective_grad ** 2

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    step_size = lr / bc1
    denom = jnp.sqrt(new_exp_avg_sq / bc2) + eps

    param_flat = param.reshape(-1)
    new_param = param_flat * (1 - lr * wd_eff) - step_size * (new_exp_avg / denom)

    new_ps = PerParamState(
        exp_avg=new_exp_avg,
        exp_avg_sq=new_exp_avg_sq,
        mu=new_mu,
        sharpness=ps.sharpness,
        gru_state=new_gru,
        mamba_fwd_state=new_fwd,
        mamba_bwd_state=new_bwd,
        step_count=step,
    )
    return new_param.reshape(param.shape), new_ps


def supergrok2_step(
    params: Any,
    grads: Any,
    opt_state: SuperGrok2State,
    meta_weights: MetaNetWeights,
    config: OptimizerConfig,
    meta_config: MetaNetConfig,
    train_acc: Optional[float] = None,
) -> Tuple[Any, SuperGrok2State]:
    """One SuperGrok v2 optimizer step (JIT-compatible)."""
    global_step = opt_state.global_step + 1
    cached_acc = (
        jnp.array(train_acc, dtype=jnp.float32)
        if train_acc is not None
        else opt_state.cached_train_acc
    )

    ramp = _get_ramp_factor(global_step, config.warmup_steps, config.warmup_ramp)
    gate_signal = _sigmoid(config.gate_scale, cached_acc, config.gate_thresh)
    lamb_eff = config.lamb * ramp * gate_signal
    wd_eff = _get_effective_wd(
        config.weight_decay, cached_acc,
        config.wd_ramp, config.wd_scale, config.wd_thresh,
    )

    params_flat, treedef = jax.tree.flatten(params)
    grads_flat, _ = jax.tree.flatten(grads)

    new_params_flat = []
    new_param_states = []

    for i, (p, g) in enumerate(zip(params_flat, grads_flat)):
        ps = opt_state.param_states[i]
        alpha_mu = config.alpha_init

        new_p, new_ps = _update_single_param(
            p, g, ps, meta_weights, meta_config,
            alpha_mu, lamb_eff,
            config.beta1, config.beta2, config.lr,
            wd_eff, config.eps, config.gradient_clipping,
        )
        new_params_flat.append(new_p)
        new_param_states.append(new_ps)

    new_params = jax.tree.unflatten(treedef, new_params_flat)
    new_opt_state = SuperGrok2State(
        param_states=new_param_states,
        global_step=global_step,
        cached_train_acc=cached_acc,
    )
    return new_params, new_opt_state


# ════════════════════════════════════════════════════════════════════════════
#  Section 6 — Bilevel meta-net optimization
# ════════════════════════════════════════════════════════════════════════════


def bilevel_step(
    params: Any,
    train_grads: Any,
    val_x: jnp.ndarray,
    val_y: jnp.ndarray,
    opt_state: SuperGrok2State,
    meta_weights: MetaNetWeights,
    config: Any,
    meta_config: MetaNetConfig,
    model_fn: Callable,
    loss_fn: Callable,
    meta_lr: float = 1e-4,
) -> Tuple[MetaNetWeights, float]:
    """Bilevel meta-net training step using jax.grad.

    Differentiates through the meta-net forward (including lax.associative_scan)
    to update meta_weights. No custom backward kernel needed.
    """
    params_flat, _ = jax.tree.flatten(params)
    grads_flat, _ = jax.tree.flatten(train_grads)

    val_loss = loss_fn(model_fn(params, val_x), val_y)
    val_grads = jax.grad(lambda p: loss_fn(model_fn(p, val_x), val_y))(params)
    val_grads_flat, _ = jax.tree.flatten(val_grads)

    def meta_loss_fn(mw):
        total_loss = jnp.array(0.0)
        for i, (g, vg) in enumerate(zip(grads_flat, val_grads_flat)):
            ps = opt_state.param_states[i]

            smart_grad, _, _, _, _ = meta_net_forward(
                g.reshape(-1), ps.sharpness,
                ps.gru_state, ps.mamba_fwd_state, ps.mamba_bwd_state,
                mw, meta_config, use_soft_routing=True,
            )

            vg_flat = vg.reshape(-1).astype(jnp.float32)
            vg_norm = jnp.linalg.norm(vg_flat)
            vg_unit = jnp.where(vg_norm > 1e-12, vg_flat / vg_norm, vg_flat)

            total_loss = total_loss - jnp.sum(smart_grad.reshape(-1) * vg_unit)
        return total_loss

    meta_loss_val, meta_grads = jax.value_and_grad(meta_loss_fn)(meta_weights)
    new_meta_weights = jax.tree.map(
        lambda w, g: w - meta_lr * g,
        meta_weights, meta_grads,
    )
    return new_meta_weights, meta_loss_val


# ════════════════════════════════════════════════════════════════════════════
#  Section 7 — INT8/INT4 quantization helpers
# ════════════════════════════════════════════════════════════════════════════


class Int8QuantizedTensor(NamedTuple):
    """INT8 symmetric quantized tensor."""
    data: jnp.ndarray
    scale: jnp.ndarray


def quantize_int8_symmetric(tensor: jnp.ndarray) -> Int8QuantizedTensor:
    """Symmetric INT8 quantization: scale = max(|tensor|) / 127."""
    abs_max = jnp.max(jnp.abs(tensor))
    scale = abs_max / 127.0
    scale = jnp.maximum(scale, 1e-8)
    quantized = jnp.round(tensor / scale).astype(jnp.int8)
    return Int8QuantizedTensor(data=quantized, scale=scale)


def dequantize_int8(qt: Int8QuantizedTensor) -> jnp.ndarray:
    """Dequantize INT8 tensor back to float32."""
    return qt.data.astype(jnp.float32) * qt.scale


def quantize_expert_weights_int8(
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray,
) -> dict:
    """Quantize expert MLP weights to INT8 (used for memory-bound runs)."""
    return {
        'W1': quantize_int8_symmetric(W1),
        'b1': quantize_int8_symmetric(b1),
        'W2': quantize_int8_symmetric(W2),
        'b2': quantize_int8_symmetric(b2),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Section 8 — v5p Pallas kernel re-exports + fused-TU contract
# ════════════════════════════════════════════════════════════════════════════


# ── inlined from former csrc/backends/pallas/primitives.py ──
#
# TPU/Pallas detection helpers + re-exports of the Pallas kernels in
# csrc/backends/pallas/_pallas_kernels.py. Duplicated across every
# Pallas launch file by design (self-containment).


def detect_tpu_version() -> str:
    """Return the TPU generation string ('v4'/'v5e'/'v5p'/'v6e') or 'cpu'."""
    try:
        devs = jax.devices()
        if not devs:
            return "cpu"
        kind = (getattr(devs[0], "device_kind", "") or "").lower()
        for v in ("v6e", "v5p", "v5e", "v4"):
            if v in kind:
                return v
        return "v4"
    except Exception:
        return "v4"


def mxu_tile_width() -> int:
    """MXU lane width for the active TPU. 128 for v4/v5e/v5p, 256 for v6e."""
    return 256 if detect_tpu_version() == "v6e" else 128


try:
    from csrc.backends.pallas._pallas_kernels import (
        mamba3_scan_pallas_tile128,
        mamba3_scan_pallas_tile256,
        pallas_persistent_scan_fused_elem_tile128,
        pallas_persistent_scan_fused_elem_tile256,
        pallas_fused_gru_peer,
        tpu_auto_select_scan,
        tpu_auto_select_fused_scan_elem,
    )
    _PALLAS_AVAILABLE = True
    _SG2_PALLAS_OK = True
except ImportError:
    _PALLAS_AVAILABLE = False
    _SG2_PALLAS_OK = False


def pallas_available() -> bool:
    return _PALLAS_AVAILABLE
# ── end inlined csrc/backends/pallas/primitives.py ──


def launch_supergrok2_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    mu_state: jnp.ndarray,
    grad: jnp.ndarray,
    sharpness: jnp.ndarray,
    A_log: jnp.ndarray,
    proj_W: jnp.ndarray, proj_b: jnp.ndarray,
    alpha: float, gru_decay: float,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> Tuple[jnp.ndarray, ...]:
    """Per-tensor SG2 step (fused-TU contract).

    Pure-JAX fallback that treats SG2 as `smart_grad = grad + alpha*mu` +
    Adam. The Pallas-accelerated path (when available) is wired through
    the supergrok2_step entrypoint above.
    """
    smart = grad + alpha * mu_state
    m = beta1 * exp_avg + (1.0 - beta1) * smart
    v = beta2 * exp_avg_sq + (1.0 - beta2) * smart * smart
    update = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_param = param - lr * (update + wd * param)
    new_mu = gru_decay * mu_state + (1.0 - gru_decay) * smart
    return new_param, m, v, new_mu


# ═════════════════════════════════════════════════════════════════════════
#  MoE/Adam multi-tensor — folded in from former launch_moe_adam.py.
#
#  Algorithm: csrc/algorithms/supergrok2.h::moe_adam_step (wraps adamw_step)
#  Self-contained — the optimizer math is the standard AdamW update;
#  expert-aware routing is handled in the model wrapper.
# ═════════════════════════════════════════════════════════════════════════


def launch_moe_adam_step(
    param: jnp.ndarray,
    exp_avg: jnp.ndarray,
    exp_avg_sq: jnp.ndarray,
    grad: jnp.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    bc1: float,
    bc2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-tensor AdamW step for MoE-aware optimizer."""
    m = beta1 * exp_avg + (1.0 - beta1) * grad
    v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    update = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_param = param - lr * (update + wd * param)
    return new_param, m, v


launch_moe_adam_step_jit = jax.jit(
    launch_moe_adam_step,
    static_argnames=("lr", "beta1", "beta2", "eps", "wd"),
)
