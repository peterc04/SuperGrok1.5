"""Fused (model, optimizer, arch) instantiation: mamba + looksam on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.looksam_tpu_v5p import looksam_step


def fused_mamba_looksam_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + looksam on TPU v5p."""
    raise NotImplementedError("fused_mamba_looksam_tpu_v5p")
