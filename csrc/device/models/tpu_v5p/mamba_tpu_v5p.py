"""Mamba SSM device-function template for TPU v5p (128-wide MXU).

Re-exports the real Pallas/JAX implementation that lives in the central
``csrc/kernels/tpu/_pallas_models.py`` module so v5p, v6e, and the dispatch
shim stay in lockstep. The selective scan inside ``mamba_forward``
auto-dispatches to the tile-128 Pallas affine scan when ``_HAS_PALLAS``
is true and falls back to ``jax.lax.associative_scan`` otherwise.
"""

from csrc.kernels.tpu._pallas_models import (
    mamba_forward as _mamba_forward,
    mamba_backward as _mamba_backward,
    mamba_layer_forward,
    mamba_selective_scan,
    MambaWeights,
    MambaLayerWeights,
    ModelConfig,
)


def mamba_forward(tokens, weights: MambaWeights, cfg: ModelConfig):
    """Mamba forward pass on TPU v5p.

    Delegates to :func:`csrc.kernels.tpu._pallas_models.mamba_forward`.
    """
    return _mamba_forward(tokens, weights, cfg)


def mamba_backward(grad_logits, tokens, weights: MambaWeights, cfg: ModelConfig):
    """Mamba backward pass on TPU v5p.

    Delegates to :func:`csrc.kernels.tpu._pallas_models.mamba_backward`.
    """
    return _mamba_backward(grad_logits, tokens, weights, cfg)


__all__ = [
    "mamba_forward",
    "mamba_backward",
    "mamba_layer_forward",
    "mamba_selective_scan",
    "MambaWeights",
    "MambaLayerWeights",
    "ModelConfig",
]
