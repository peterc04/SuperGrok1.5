"""Fused (model, optimizer, arch) instantiation: mamba + supergrok2 on TPU v5p."""

from csrc.device.models.tpu_v5p.mamba_tpu_v5p import mamba_forward, mamba_backward
from csrc.device.optimizers.tpu_v5p.supergrok2_tpu_v5p import supergrok2_step


def fused_mamba_supergrok2_step(params, inputs, state, lr):
    """TODO: Fused forward-backward-update for mamba + supergrok2 on TPU v5p."""
    raise NotImplementedError("fused_mamba_supergrok2_tpu_v5p")
