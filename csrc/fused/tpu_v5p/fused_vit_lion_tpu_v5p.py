"""Fused (model, optimizer, arch) instantiation: vit + lion on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.lion_tpu_v5p import lion_step


def fused_vit_lion_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + lion on TPU v5p."""
    raise NotImplementedError("fused_vit_lion_tpu_v5p")
