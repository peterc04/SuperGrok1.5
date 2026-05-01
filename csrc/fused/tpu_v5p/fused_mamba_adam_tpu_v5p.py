"""Fused (model, optimizer, arch) instantiation: mamba + adam on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.adam_tpu_v5p import adam_step


def fused_mamba_adam_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + adam on TPU v5p."""
    raise NotImplementedError("fused_mamba_adam_tpu_v5p")
