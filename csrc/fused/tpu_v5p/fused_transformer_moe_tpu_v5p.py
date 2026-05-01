"""Fused (model, optimizer, arch) instantiation: transformer + moe on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.moe_tpu_v5p import moe_step


def fused_transformer_moe_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + moe on TPU v5p."""
    raise NotImplementedError("fused_transformer_moe_tpu_v5p")
