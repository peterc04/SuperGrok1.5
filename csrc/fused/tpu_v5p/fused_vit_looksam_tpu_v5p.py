"""Fused (model, optimizer, arch) instantiation: vit + looksam on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.looksam_tpu_v5p import looksam_step


def fused_vit_looksam_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + looksam on TPU v5p."""
    raise NotImplementedError("fused_vit_looksam_tpu_v5p")
