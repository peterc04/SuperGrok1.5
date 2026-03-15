"""
Mamba-3 + 4-Head PEER + Per-Element GRU Meta-Net for SuperGrok v2 (JAX)

Composes scan.py, gru.py, and peer.py into the full meta-net forward pass.
All state is explicit (passed in, returned out) — no mutation.

Architecture (identical to PyTorch Mamba3PEERMetaNet):
  1. Sort by |gradient| magnitude
  2. Input projection: [grad, sharpness] -> [N, d_model]
  3. Bidirectional Mamba-3 scan (forward + backward)
  4. Unsort to original order
  5. Per-element GRU update
  6. Multi-head PEER routing + expert MLP
  7. Skip connection: smart_grad = grad + rescale * expert_output
"""

import math
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple

from .scan import MambaScanWeights, mamba3_scan
from .gru import GRUWeights, mini_gru
from .peer import peer_expert_forward, peer_expert_forward_hard


class MetaNetWeights(NamedTuple):
    """All weights for the SuperGrok v2 meta-net.

    JAX-compatible NamedTuple — all fields are jnp arrays.
    Can be passed through jax.jit without retracing.
    Registered as a pytree automatically by JAX (NamedTuple support).
    """
    # Input projection
    input_proj_W: jnp.ndarray   # [d_model, 2]
    input_proj_b: jnp.ndarray   # [d_model]

    # Mamba forward scan
    mamba_fwd: MambaScanWeights

    # Mamba backward scan
    mamba_bwd: MambaScanWeights

    # GRU
    gru: GRUWeights

    # PEER routing
    peer_query_Ws: jnp.ndarray  # [num_heads, d_model, peer_input_dim]
    prod_keys_A: jnp.ndarray    # [num_heads, pk_dim, d_model//2]
    prod_keys_B: jnp.ndarray    # [num_heads, pk_dim, d_model//2]

    # Expert pool
    expert_W1: jnp.ndarray      # [num_experts, expert_hidden, 1]
    expert_b1: jnp.ndarray      # [num_experts, expert_hidden]
    expert_W2: jnp.ndarray      # [num_experts, 1, expert_hidden]
    expert_b2: jnp.ndarray      # [num_experts, 1]

    # Scalars (stored as arrays for JIT compatibility)
    rescale: jnp.ndarray        # scalar float

    # Expert tracking (mutable buffer — updated outside JIT)
    expert_counts: jnp.ndarray  # [num_experts] int32


class MetaNetConfig(NamedTuple):
    """Configuration for the meta-net (static, not traced by JAX)."""
    d_model: int = 8
    d_state: int = 16
    d_inner: int = 16  # d_model * mamba_expand
    num_peer_heads: int = 4
    num_experts: int = 144
    expert_hidden: int = 16
    gru_hidden: int = 4
    pk_dim: int = 12  # sqrt(num_experts)
    rescale: float = 0.1


def init_meta_weights(
    config: MetaNetConfig,
    key: jnp.ndarray,
) -> MetaNetWeights:
    """Initialize meta-net weights with the same scheme as PyTorch.

    Args:
        config: MetaNetConfig
        key: JAX PRNG key

    Returns:
        MetaNetWeights with random initialization
    """
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

    # Linear layer init: Kaiming uniform (PyTorch default)
    def _linear(k, out_dim, in_dim):
        bound = 1.0 / math.sqrt(in_dim)
        return jax.random.uniform(k, (out_dim, in_dim), minval=-bound, maxval=bound)

    def _bias(k, dim, in_dim):
        bound = 1.0 / math.sqrt(in_dim)
        return jax.random.uniform(k, (dim,), minval=-bound, maxval=bound)

    # Input projection
    input_proj_W = _linear(_k(), d_model, 2)
    input_proj_b = _bias(_k(), d_model, 2)

    # Mamba forward
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

    # Mamba backward
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

    # GRU
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

    # PEER routing
    peer_input_dim = gru_hidden + 2 * d_model + 2
    peer_query_Ws = jnp.stack([
        _linear(_k(), d_model, peer_input_dim)
        for _ in range(num_heads)
    ])
    prod_keys_A = jax.random.normal(_k(), (num_heads, pk_dim, d_model // 2)) * 0.02
    prod_keys_B = jax.random.normal(_k(), (num_heads, pk_dim, d_model // 2)) * 0.02

    # Expert pool
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

    Mathematical equivalence to PyTorch Mamba3PEERMetaNet.forward (hard routing)
    or forward_for_bilevel (soft routing).

    Args:
        grad: [N] flattened gradient
        sharpness: [N] flattened sharpness
        gru_state: [N, gru_hidden] persistent per-element state
        mamba_fwd_state: [d_inner, d_state] or None
        mamba_bwd_state: [d_inner, d_state] or None
        meta_weights: MetaNetWeights
        config: MetaNetConfig
        use_soft_routing: if True, use differentiable top-k soft routing (bilevel)

    Returns:
        smart_grad: [N] modified gradient
        new_gru_state: [N, gru_hidden]
        new_mamba_fwd_state: [d_inner, d_state]
        new_mamba_bwd_state: [d_inner, d_state]
        expert_counts: [num_experts] int32
    """
    N = grad.shape[0]
    g = grad.reshape(-1).astype(jnp.float32)
    s = sharpness.reshape(-1).astype(jnp.float32)

    # 1. Sort by gradient magnitude
    sort_idx = jnp.argsort(jnp.abs(g))
    g_sorted = g[sort_idx]
    s_sorted = s[sort_idx]

    # 2. Input projection
    inp = jnp.stack([g_sorted, s_sorted], axis=-1)  # [N, 2]
    x = inp @ meta_weights.input_proj_W.T + meta_weights.input_proj_b  # [N, d_model]

    # 3. Bidirectional Mamba-3 scan
    fwd_out, new_fwd = mamba3_scan(x, meta_weights.mamba_fwd, mamba_fwd_state, reverse=False)
    bwd_out, new_bwd = mamba3_scan(x, meta_weights.mamba_bwd, mamba_bwd_state, reverse=True)

    # 4. Unsort to original order
    unsort_idx = jnp.argsort(sort_idx)
    fwd_ctx = fwd_out[unsort_idx]  # [N, d_model]
    bwd_ctx = bwd_out[unsort_idx]  # [N, d_model]

    # 5. Per-element GRU update
    gru_input = jnp.concatenate([
        g[:, None], s[:, None], fwd_ctx, bwd_ctx
    ], axis=-1)  # [N, 2 + 2*d_model]
    new_gru = mini_gru(gru_input, gru_state.astype(jnp.float32), meta_weights.gru)

    # 6. Multi-head PEER routing
    peer_input = jnp.concatenate([
        new_gru, fwd_ctx, bwd_ctx, g[:, None], s[:, None]
    ], axis=-1)  # [N, gru_hidden + 2*d_model + 2]

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

    # 7. Skip connection
    smart_grad = g + meta_weights.rescale * expert_out

    return smart_grad.reshape(grad.shape), new_gru, new_fwd, new_bwd, expert_counts
