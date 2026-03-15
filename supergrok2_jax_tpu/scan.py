"""
Mamba-3 Bidirectional Scan via jax.lax.associative_scan

Mathematical formulation (identical to CUDA):
  For each d_inner dimension j, the scan over N timesteps computes:
    h[t] = A_bar[t] * rot(h[t-1]) + B_bar[t] * x[t]

  where:
    A_bar[t] = (1 + dt*A/2) / (1 - dt*A/2)    (trapezoidal discretization)
    B_bar[t] = dt * B[t]
    rot() applies paired RoPE rotation to even/odd state pairs

  The RoPE rotation + A_bar scaling can be expressed as a 2x2 affine
  transform per state pair, enabling parallel prefix scan via
  lax.associative_scan with matrix composition as the binary operator.

JAX advantage:
  - lax.associative_scan handles all parallelization automatically
  - XLA compiles to optimal scan for target hardware (TPU systolic array or GPU)
  - ~40 lines vs ~500 lines of Blelloch CUDA
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Optional, Tuple


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

    Args:
        dt: [N, d_inner] softplus-activated dt values
        x_branch: [N, d_inner] input branch
        B: [N, d_state] B projection
        A_log: [d_inner, d_state] log of negative A diagonal
        rope_freq: [d_inner, d_state//2] RoPE frequencies

    Returns:
        Ms: [N, d_inner, d_state//2, 2, 2] rotation+scaling matrices
        bs: [N, d_inner, d_state//2, 2] bias vectors
    """
    A = -jnp.exp(A_log)  # [d_inner, d_state]

    # A_bar via trapezoidal discretization: [N, d_inner, d_state]
    half_dtA = dt[:, :, None] * A[None, :, :] / 2.0
    A_bar = (1.0 + half_dtA) / (1.0 - half_dtA + 1e-8)

    # B_bar: [N, d_inner, d_state]
    B_bar = dt[:, :, None] * B[:, None, :]

    # RoPE phase: [N, d_inner, d_state//2]
    phase = dt[:, :, None] * rope_freq[None, :, :]
    cos_p = jnp.cos(phase)
    sin_p = jnp.sin(phase)

    # Split A_bar into even/odd pairs
    A_bar_e = A_bar[:, :, 0::2]  # [N, d_inner, d_state//2]
    A_bar_o = A_bar[:, :, 1::2]

    # Build 2x2 matrices: rotation * scaling
    # M = [[A_e*cos, -A_e*sin],
    #      [A_o*sin,  A_o*cos]]
    # Shape: [N, d_inner, d_state//2, 2, 2]
    M_00 = A_bar_e * cos_p
    M_01 = -A_bar_e * sin_p
    M_10 = A_bar_o * sin_p
    M_11 = A_bar_o * cos_p
    Ms = jnp.stack([
        jnp.stack([M_00, M_01], axis=-1),
        jnp.stack([M_10, M_11], axis=-1),
    ], axis=-2)

    # Build bias vectors: [N, d_inner, d_state//2, 2]
    B_bar_e = B_bar[:, :, 0::2]
    B_bar_o = B_bar[:, :, 1::2]
    bs = jnp.stack([
        B_bar_e * x_branch[:, :, None],
        B_bar_o * x_branch[:, :, None],
    ], axis=-1)

    return Ms, bs


def _associative_combine(left, right):
    """Compose two affine transforms: (M_r @ M_l, M_r @ b_l + b_r).

    This operator is associative: combine(combine(a,b), c) == combine(a, combine(b,c))

    Each element is (M, b) where:
      M: [..., 2, 2] matrix
      b: [..., 2] vector

    The composition rule for affine transforms h' = M*h + b is:
      (M2, b2) ∘ (M1, b1) = (M2 @ M1, M2 @ b1 + b2)
    """
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

    Mathematical equivalence to CUDA:
      h[t] = A_bar[t] * RoPE(h[t-1]) + B_bar[t] * x[t]
      y[t] = sum_s(h[t,:,s] * C[t,s]) * silu(z[t]) + D * x_branch[t]
      output[t] = y[t] @ out_proj.T

    Args:
        x_sorted: [N, d_model] sorted input features
        weights: MambaScanWeights namedtuple
        initial_state: [d_inner, d_state] or None
        reverse: if True, scan in reverse direction (for backward scan)

    Returns:
        output: [N, d_model] scan output projected to d_model
        final_state: [d_inner, d_state]
    """
    N, d_model = x_sorted.shape
    d_inner = weights.in_proj_W.shape[0] // 2
    d_state = weights.A_log.shape[1]
    half_d_state = d_state // 2

    if reverse:
        x_sorted = x_sorted[::-1]

    # Input projection: [N, d_model] -> [N, 2*d_inner]
    xz = x_sorted @ weights.in_proj_W.T
    x_branch = xz[:, :d_inner]   # [N, d_inner]
    z_branch = xz[:, d_inner:]   # [N, d_inner]

    # Selective parameters (input-dependent)
    dt_raw = x_branch @ weights.dt_proj_W.T + weights.dt_proj_b  # [N, d_inner]
    dt = jax.nn.softplus(dt_raw)                                  # [N, d_inner]
    B = x_branch @ weights.B_proj_W.T                             # [N, d_state]
    C = x_branch @ weights.C_proj_W.T                             # [N, d_state]

    # Build affine transforms for associative scan
    Ms, bs = _build_affine_transforms(dt, x_branch, B, weights.A_log, weights.rope_freq)
    # Ms: [N, d_inner, half_d_state, 2, 2]
    # bs: [N, d_inner, half_d_state, 2]

    # Run associative scan: O(N) work, O(log N) depth
    cumulative_M, cumulative_b = lax.associative_scan(
        _associative_combine, (Ms, bs), axis=0)

    # Apply initial state
    if initial_state is not None:
        h_init_pairs = initial_state.reshape(d_inner, half_d_state, 2)
    else:
        h_init_pairs = jnp.zeros((d_inner, half_d_state, 2))

    # h[t] = cumulative_M[t] @ h_init + cumulative_b[t]
    h_pairs = (
        jnp.einsum('...ij,...j->...i', cumulative_M, h_init_pairs[None, :, :, :])
        + cumulative_b
    )
    # h_pairs: [N, d_inner, half_d_state, 2]

    # Reshape to [N, d_inner, d_state] by interleaving even/odd
    h_even = h_pairs[..., 0]  # [N, d_inner, half_d_state]
    h_odd = h_pairs[..., 1]   # [N, d_inner, half_d_state]
    h = jnp.zeros((N, d_inner, d_state))
    h = h.at[:, :, 0::2].set(h_even)
    h = h.at[:, :, 1::2].set(h_odd)

    # Output: y[t] = sum_s(h[t, j, s] * C[t, s])
    # C: [N, d_state], h: [N, d_inner, d_state]
    y = jnp.sum(h * C[:, None, :], axis=-1)  # [N, d_inner]

    # Gated output + skip connection
    y = y * jax.nn.silu(z_branch) + weights.D[None, :] * x_branch

    # Output projection: [N, d_inner] -> [N, d_model]
    output = y @ weights.out_proj_W.T

    # Final state: last timestep's h
    final_state = h[-1]  # [d_inner, d_state]

    if reverse:
        output = output[::-1]

    return output, final_state
