"""SuperGrok v2 device-function template for TPU v5p (128-wide MXU).

Imports Pallas-optimized scan kernels from the TPU runtime path.
The Mamba-3 scan and fused element step use Pallas tile-128 kernels
that are already functional on v5p hardware.
"""

import jax
import jax.numpy as jnp

# Import Pallas-optimized scan operations from the TPU v5p runtime
from csrc.kernels.tpu.v5p import (
    mamba3_scan,
    pallas_persistent_scan_fused_elem,
    vmem_persistent_expert_mlp,
    TILE_SIZE,
)


def supergrok2_mamba3_scan(
    x_sorted, in_proj_W, dt_proj_W, dt_proj_b,
    B_proj_W, C_proj_W, A_log, D_param, rope_freq,
    initial_state, N, d_model, d_inner, d_state, reverse=False
):
    """Mamba-3 selective scan on TPU v5p via Pallas tile-128.

    Delegates to the Pallas-optimized scan kernel which tiles the
    sequential scan across 128-wide MXU tiles for high throughput.
    """
    return mamba3_scan(
        x_sorted, in_proj_W, dt_proj_W, dt_proj_b,
        B_proj_W, C_proj_W, A_log, D_param, rope_freq,
        initial_state, N, d_model, d_inner, d_state,
        reverse=reverse, tile_size=TILE_SIZE,
    )


def supergrok2_fused_elem_step(
    scan_out_fwd, scan_out_bwd, gru_state, expert_weights,
    peer_weights, param, exp_avg, exp_avg_sq, grad, sharpness,
    mu, d_model, d_inner, num_experts, num_heads,
    beta1, beta2, lr, weight_decay, eps, bc1, bc2
):
    """Fused element step (GRU + PEER + Adam) on TPU v5p via Pallas.

    Uses VMEM-persistent expert weights for efficient MoE routing.
    """
    return pallas_persistent_scan_fused_elem(
        scan_out_fwd, scan_out_bwd, gru_state, expert_weights,
        peer_weights, param, exp_avg, exp_avg_sq, grad, sharpness,
        mu, d_model, d_inner, num_experts, num_heads,
        beta1, beta2, lr, weight_decay, eps, bc1, bc2,
        tile_size=TILE_SIZE,
    )
