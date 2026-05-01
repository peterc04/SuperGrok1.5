"""Fused (model, optimizer, arch) instantiation: mamba + supergrok15 on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.supergrok15_tpu_v5p import supergrok15_step


def fused_mamba_supergrok15_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + supergrok15 on TPU v5p."""
    raise NotImplementedError("fused_mamba_supergrok15_tpu_v5p")
