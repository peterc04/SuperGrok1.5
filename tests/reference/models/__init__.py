"""Reference model implementations (PyTorch nn.Module).

Each submodule provides a verbatim copy of the corresponding class
defined in ``grokking_race_v2.py`` PART 1, plus a ``make_<model>``
factory returning a fresh instance with default config:

    from tests.reference.models.decoder_ref import make_decoder
    from tests.reference.models.vit_ref import make_vit
    from tests.reference.models.mamba_ref import make_mamba

These are intended for use as the ground-truth oracle in parity tests
against the fused-kernel bindings registered on ``_ops.models``. Each
file is self-contained and only imports from ``torch`` and the standard
library.
"""

from .decoder_ref import Transformer, DecoderBlock, make_decoder
from .vit_ref import ViT, EncoderBlock, make_vit
from .mamba_ref import MambaModel, SelectiveSSMLayer, make_mamba

__all__ = [
    "Transformer",
    "DecoderBlock",
    "make_decoder",
    "ViT",
    "EncoderBlock",
    "make_vit",
    "MambaModel",
    "SelectiveSSMLayer",
    "make_mamba",
]
