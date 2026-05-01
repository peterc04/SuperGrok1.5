"""Fused (model, optimizer, arch) instantiation: vit + supergrok15 on TPU v5p."""

from csrc.device.models.tpu_v5p.vit_tpu_v5p import vit_forward, vit_backward
from csrc.device.optimizers.tpu_v5p.supergrok15_tpu_v5p import supergrok15_step


def fused_vit_supergrok15_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for vit + supergrok15 on TPU v5p."""
    raise NotImplementedError("fused_vit_supergrok15_tpu_v5p")
