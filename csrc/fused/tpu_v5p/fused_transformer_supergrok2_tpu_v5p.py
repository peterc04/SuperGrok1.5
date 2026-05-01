"""Fused (model, optimizer, arch) instantiation: transformer + supergrok2 on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.supergrok2_tpu_v5p import supergrok2_step


def fused_transformer_supergrok2_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + supergrok2 on TPU v5p."""
    raise NotImplementedError("fused_transformer_supergrok2_tpu_v5p")
