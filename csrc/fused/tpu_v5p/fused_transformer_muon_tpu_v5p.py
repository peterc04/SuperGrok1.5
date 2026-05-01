"""Fused (model, optimizer, arch) instantiation: transformer + muon on TPU v5p."""

from csrc.device.models.tpu_v5p.transformer_tpu_v5p import transformer_forward, transformer_backward
from csrc.device.optimizers.tpu_v5p.muon_tpu_v5p import muon_step


def fused_transformer_muon_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for transformer + muon on TPU v5p."""
    raise NotImplementedError("fused_transformer_muon_tpu_v5p")
