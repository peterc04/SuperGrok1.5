"""Fused (model, optimizer, arch) instantiation: transformer + supergrok15 on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.supergrok15_tpu_v5p import supergrok15_step


def fused_transformer_supergrok15_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + supergrok15 on TPU v5p."""
    raise NotImplementedError("fused_transformer_supergrok15_tpu_v5p")
