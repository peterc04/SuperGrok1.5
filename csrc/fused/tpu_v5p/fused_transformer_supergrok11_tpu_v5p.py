"""Fused (model, optimizer, arch) instantiation: transformer + supergrok11 on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.supergrok11_tpu_v5p import supergrok11_step


def fused_transformer_supergrok11_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + supergrok11 on TPU v5p."""
    raise NotImplementedError("fused_transformer_supergrok11_tpu_v5p")
