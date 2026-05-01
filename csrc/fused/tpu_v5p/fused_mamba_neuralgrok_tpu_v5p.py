"""Fused (model, optimizer, arch) instantiation: mamba + neuralgrok on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.neuralgrok_tpu_v5p import neuralgrok_step


def fused_mamba_neuralgrok_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + neuralgrok on TPU v5p."""
    raise NotImplementedError("fused_mamba_neuralgrok_tpu_v5p")
