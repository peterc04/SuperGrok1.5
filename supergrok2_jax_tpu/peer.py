"""
Multi-Head PEER Routing + Expert MLP for SuperGrok v2 (JAX)

Mathematical formulation (identical to PyTorch):
  For each head h:
    query = peer_input @ query_W[h].T                    # [N, d_model]
    q_a, q_b = split(query, d_model//2)
    scores_a = q_a @ keys_A[h].T                          # [N, pk_dim]
    scores_b = q_b @ keys_B[h].T                          # [N, pk_dim]
    top_a = topk(scores_a, k=4)
    top_b = topk(scores_b, k=4)
    expert_idx = top_a_idx[:,:,None] * pk_dim + top_b_idx[:,None,:]  # [N, 16]
    routing_w = softmax(top_a * 10) x softmax(top_b * 10)            # [N, 16]
    expert_out = sum_k(routing_w[k] * MLP_expert[k](grad))

Product-key routing selects 16 experts (4x4 grid) per element per head.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


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
    """Multi-head PEER routing with product-key expert selection.

    Mathematical equivalence to PyTorch Mamba3PEERMetaNet.forward_for_bilevel
    (PEER routing section).

    Args:
        peer_input: [N, peer_input_dim]
        grad: [N] gradient values (scalar input to expert MLPs)
        peer_query_Ws: [num_heads, d_model, peer_input_dim] query projections
        prod_keys_A: [num_heads, pk_dim, d_model//2] product keys A
        prod_keys_B: [num_heads, pk_dim, d_model//2] product keys B
        expert_W1: [num_experts, expert_hidden, 1] first layer weights
        expert_b1: [num_experts, expert_hidden] first layer biases
        expert_W2: [num_experts, 1, expert_hidden] second layer weights
        expert_b2: [num_experts, 1] second layer biases
        num_heads: number of PEER routing heads
        pk_dim: product key dimension (sqrt(num_experts))
        num_experts: total number of experts
        topk: top-k per sub-key (default 4)

    Returns:
        expert_output: [N] averaged expert output across heads
        expert_counts: [num_experts] int32 activation counts
    """
    N = peer_input.shape[0]
    half_d = peer_query_Ws.shape[1] // 2
    num_active = topk * topk

    total_out = jnp.zeros(N)
    expert_counts = jnp.zeros(num_experts, dtype=jnp.int32)

    def _head_forward(h, carry):
        total_out, expert_counts = carry

        # Query projection
        query = peer_input @ peer_query_Ws[h].T  # [N, d_model]
        q_a = query[:, :half_d]
        q_b = query[:, half_d:]

        # Product-key scores
        scores_a = q_a @ prod_keys_A[h].T  # [N, pk_dim]
        scores_b = q_b @ prod_keys_B[h].T  # [N, pk_dim]

        # Top-k selection
        top_a_vals, top_a_idx = jax.lax.top_k(scores_a, topk)  # [N, topk]
        top_b_vals, top_b_idx = jax.lax.top_k(scores_b, topk)  # [N, topk]

        # Soft routing weights
        soft_a = jax.nn.softmax(top_a_vals * 10.0, axis=-1)  # [N, topk]
        soft_b = jax.nn.softmax(top_b_vals * 10.0, axis=-1)  # [N, topk]

        # Expert indices: [N, topk*topk]
        expert_idx = (
            top_a_idx[:, :, None] * pk_dim + top_b_idx[:, None, :]
        ).reshape(N, num_active)
        routing_weights = (
            soft_a[:, :, None] * soft_b[:, None, :]
        ).reshape(N, num_active)

        # Expert MLP: gather weights for selected experts
        # expert_W1: [num_experts, expert_hidden, 1]
        ew1 = expert_W1[expert_idx]   # [N, num_active, expert_hidden, 1]
        eb1 = expert_b1[expert_idx]   # [N, num_active, expert_hidden]
        ew2 = expert_W2[expert_idx]   # [N, num_active, 1, expert_hidden]
        eb2 = expert_b2[expert_idx]   # [N, num_active, 1]

        # Forward: z = relu(W1 * grad + b1), out = W2 * z + b2
        g_exp = grad[:, None, None, None]  # [N, 1, 1, 1]
        z_hidden = jax.nn.relu(
            (ew1 * g_exp).squeeze(-1) + eb1
        )  # [N, num_active, expert_hidden]
        out = (
            jnp.matmul(ew2, z_hidden[..., None]).squeeze(-1).squeeze(-1)
            + eb2.squeeze(-1)
        )  # [N, num_active]

        head_out = (routing_weights * out).sum(axis=1)  # [N]
        total_out = total_out + head_out

        # Track expert activations
        expert_counts = expert_counts.at[expert_idx.reshape(-1)].add(1)

        return total_out, expert_counts

    # Unrolled loop over heads (small fixed number, JIT-friendly)
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
    """Hard PEER routing (argmax, non-differentiable) for the forward step.

    Selects 1 expert per head (vs topk*topk for soft routing).
    Used in the regular optimizer step (not bilevel).
    """
    N = peer_input.shape[0]
    half_d = peer_query_Ws.shape[1] // 2

    total_out = jnp.zeros(N)
    expert_counts = jnp.zeros(num_experts, dtype=jnp.int32)

    for h in range(num_heads):
        query = peer_input @ peer_query_Ws[h].T
        q_a = query[:, :half_d]
        q_b = query[:, half_d:]

        idx_a = jnp.argmax(q_a @ prod_keys_A[h].T, axis=-1)  # [N]
        idx_b = jnp.argmax(q_b @ prod_keys_B[h].T, axis=-1)  # [N]
        expert_idx = idx_a * pk_dim + idx_b  # [N]

        # Expert MLP
        W1 = expert_W1[expert_idx]  # [N, expert_hidden, 1]
        b1 = expert_b1[expert_idx]  # [N, expert_hidden]
        W2 = expert_W2[expert_idx]  # [N, 1, expert_hidden]
        b2 = expert_b2[expert_idx]  # [N, 1]

        g = grad[:, None, None]  # [N, 1, 1]
        z = jax.nn.relu(jnp.matmul(W1, g).squeeze(-1) + b1)  # [N, expert_hidden]
        out = jnp.matmul(W2, z[:, :, None]).squeeze(-1).squeeze(-1) + b2.squeeze(-1)  # [N]

        total_out = total_out + out
        expert_counts = expert_counts.at[expert_idx].add(1)

    return total_out / num_heads, expert_counts
