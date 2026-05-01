"""Fused (model, optimizer, arch) instantiation: transformer + prodigy on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.prodigy_tpu_v5p import prodigy_step


def fused_transformer_prodigy_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + prodigy on TPU v5p."""
    raise NotImplementedError("fused_transformer_prodigy_tpu_v5p")
