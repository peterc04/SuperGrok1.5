"""Fused (model, optimizer, arch) instantiation: transformer + looksam on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.looksam_tpu_v5p import looksam_step


def fused_transformer_looksam_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + looksam on TPU v5p."""
    raise NotImplementedError("fused_transformer_looksam_tpu_v5p")
