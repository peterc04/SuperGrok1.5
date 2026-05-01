"""Fused (model, optimizer, arch) instantiation: vit + prodigy on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.prodigy_tpu_v5p import prodigy_step


def fused_vit_prodigy_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + prodigy on TPU v5p."""
    raise NotImplementedError("fused_vit_prodigy_tpu_v5p")
