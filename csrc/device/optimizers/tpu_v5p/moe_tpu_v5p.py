"""MoE device-function template for TPU v5p (128-wide MXU).

Imports VMEM-persistent expert MLP from the Pallas runtime for
efficient MoE routing on TPU v5p hardware.
"""

import jax
import jax.numpy as jnp

from csrc.kernels.tpu.v5p import (
    vmem_persistent_expert_mlp,
    TILE_SIZE,
)


def moe_expert_forward(input_data, gate_logits, W1, b1, W2, b2,
                       threshold, num_experts, input_dim, expert_dim):
    """Dynamic expert forward on TPU using VMEM-persistent weights.

    Leverages the Pallas vmem_persistent_expert_mlp kernel for
    efficient expert routing without HBM round-trips.
    """
    return vmem_persistent_expert_mlp(
        input_data, gate_logits, W1, b1, W2, b2,
        threshold=threshold, num_experts=num_experts,
        input_dim=input_dim, expert_dim=expert_dim,
        tile_size=TILE_SIZE,
    )
