"""Fused (model, optimizer, arch) instantiation: transformer + neuralgrok on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.neuralgrok_tpu_v5p import neuralgrok_step


def fused_transformer_neuralgrok_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + neuralgrok on TPU v5p."""
    raise NotImplementedError("fused_transformer_neuralgrok_tpu_v5p")
