"""Vision Transformer device-function template for TPU v5p (128-wide MXU).

Re-exports the real Pallas/JAX implementation that lives in the central
``csrc/kernels/tpu/_pallas_models.py`` module so v5p, v6e, and the dispatch
shim stay in lockstep.
"""

from csrc.kernels.tpu._pallas_models import (
    vit_forward as _vit_forward,
    vit_backward as _vit_backward,
    vit_patch_project,
    ViTWeights,
    ModelConfig,
)


def vit_forward(image, weights: ViTWeights, cfg: ModelConfig):
    """ViT forward pass on TPU v5p.

    Delegates to :func:`csrc.kernels.tpu._pallas_models.vit_forward`.
    """
    return _vit_forward(image, weights, cfg)


def vit_backward(grad_logits, image, weights: ViTWeights, cfg: ModelConfig):
    """ViT backward pass on TPU v5p.

    Delegates to :func:`csrc.kernels.tpu._pallas_models.vit_backward`.
    """
    return _vit_backward(grad_logits, image, weights, cfg)


__all__ = [
    "vit_forward",
    "vit_backward",
    "vit_patch_project",
    "ViTWeights",
    "ModelConfig",
]
