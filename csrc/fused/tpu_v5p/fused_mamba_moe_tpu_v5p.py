"""Fused (model, optimizer, arch) instantiation: mamba + moe on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.moe_tpu_v5p import moe_step


def fused_mamba_moe_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + moe on TPU v5p."""
    raise NotImplementedError("fused_mamba_moe_tpu_v5p")
