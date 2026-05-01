"""Fused (model, optimizer, arch) instantiation: transformer + grokfast on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.grokfast_tpu_v5p import grokfast_step


def fused_transformer_grokfast_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + grokfast on TPU v5p."""
    raise NotImplementedError("fused_transformer_grokfast_tpu_v5p")
