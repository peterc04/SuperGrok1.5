"""Fused (model, optimizer, arch) instantiation: mamba + prodigy on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.prodigy_tpu_v5p import prodigy_step


def fused_mamba_prodigy_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + prodigy on TPU v5p."""
    raise NotImplementedError("fused_mamba_prodigy_tpu_v5p")
