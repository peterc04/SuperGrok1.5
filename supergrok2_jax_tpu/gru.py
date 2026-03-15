"""
Per-Element GRU Cell for SuperGrok v2 (JAX)

Mathematical formulation (identical to PyTorch MiniGRU):
  z = sigmoid(W_z @ [x, h] + b_z)           # update gate
  r = sigmoid(W_r @ [x, h] + b_r)           # reset gate
  h_tilde = tanh(W_h @ [x, r*h] + b_h)      # candidate
  h_new = (1 - z) * h + z * h_tilde          # interpolation

The GRU is applied independently per element (no scan across N),
so it's embarrassingly parallel and maps directly to JAX matrix ops.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class GRUWeights(NamedTuple):
    """Weights for the per-element GRU."""
    Wz: jnp.ndarray   # [gru_hidden, input_dim + gru_hidden]
    bz: jnp.ndarray   # [gru_hidden]
    Wr: jnp.ndarray   # [gru_hidden, input_dim + gru_hidden]
    br: jnp.ndarray   # [gru_hidden]
    Wh: jnp.ndarray   # [gru_hidden, input_dim + gru_hidden]
    bh: jnp.ndarray   # [gru_hidden]


def mini_gru(
    x: jnp.ndarray,
    h_old: jnp.ndarray,
    weights: GRUWeights,
) -> jnp.ndarray:
    """Per-element GRU update.

    Mathematical equivalence to PyTorch MiniGRU.forward:
      xh = cat([x, h_old])
      z = sigmoid(xh @ Wz.T + bz)
      r = sigmoid(xh @ Wr.T + br)
      xrh = cat([x, r * h_old])
      h_tilde = tanh(xrh @ Wh.T + bh)
      h_new = (1-z) * h_old + z * h_tilde

    Args:
        x: [N, input_dim] input features
        h_old: [N, gru_hidden] previous hidden state
        weights: GRUWeights namedtuple

    Returns:
        h_new: [N, gru_hidden] updated hidden state
    """
    # Concatenate input and hidden state
    xh = jnp.concatenate([x, h_old], axis=-1)  # [N, input_dim + gru_hidden]

    # Gates
    z = jax.nn.sigmoid(xh @ weights.Wz.T + weights.bz)  # [N, gru_hidden]
    r = jax.nn.sigmoid(xh @ weights.Wr.T + weights.br)  # [N, gru_hidden]

    # Candidate
    xrh = jnp.concatenate([x, r * h_old], axis=-1)
    h_tilde = jnp.tanh(xrh @ weights.Wh.T + weights.bh)  # [N, gru_hidden]

    # Interpolation
    h_new = (1 - z) * h_old + z * h_tilde

    return h_new
