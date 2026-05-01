"""Fused (model, optimizer, arch) instantiation: vit + neuralgrok on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.neuralgrok_tpu_v5p import neuralgrok_step


def fused_vit_neuralgrok_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + neuralgrok on TPU v5p."""
    raise NotImplementedError("fused_vit_neuralgrok_tpu_v5p")
