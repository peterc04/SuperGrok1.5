"""Fused (model, optimizer, arch) instantiation: vit + grokfast on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.grokfast_tpu_v5p import grokfast_step


def fused_vit_grokfast_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + grokfast on TPU v5p."""
    raise NotImplementedError("fused_vit_grokfast_tpu_v5p")
