"""Fused (model, optimizer, arch) instantiation: vit + muon on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.muon_tpu_v5p import muon_step


def fused_vit_muon_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + muon on TPU v5p."""
    raise NotImplementedError("fused_vit_muon_tpu_v5p")
