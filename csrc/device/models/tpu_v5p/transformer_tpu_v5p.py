"""Decoder Transformer device-function template for TPU v5p (128-wide MXU).

Re-exports the real Pallas/JAX implementation that lives in the central
``csrc/kernels/tpu/_pallas_models.py`` module. The naming alias here
(``transformer_*``) matches the per-arch device header naming convention
used by the build scaffolding while keeping the implementation in a single
place to avoid drift between v5p, v6e, and the dispatch shim.
"""

from csrc.kernels.tpu._pallas_models import (
    decoder_forward as _decoder_forward,
    decoder_backward as _decoder_backward,
    DecoderWeights,
    ModelConfig,
)


def transformer_forward(tokens, weights: DecoderWeights, cfg: ModelConfig):
    """Decoder Transformer forward pass on TPU v5p.

    Delegates to the shared Pallas/JAX implementation in
    :mod:`csrc.kernels.tpu._pallas_models`. The 128-wide MXU tiling is
    handled inside the shared attention / scan kernels.
    """
    return _decoder_forward(tokens, weights, cfg)


def transformer_backward(grad_logits, tokens, weights: DecoderWeights,
                         cfg: ModelConfig, saved_activations=None):
    """Decoder Transformer backward pass on TPU v5p.

    Delegates to :func:`csrc.kernels.tpu._pallas_models.decoder_backward`,
    which uses ``jax.grad`` over the JIT-compiled forward.
    """
    return _decoder_backward(grad_logits, tokens, weights, cfg, saved_activations)


__all__ = [
    "transformer_forward",
    "transformer_backward",
    "DecoderWeights",
    "ModelConfig",
]
