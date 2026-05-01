"""TPU Pallas kernel dispatch.

Per-TPU-version Pallas kernels live in ``csrc/kernels/tpu/v5p/`` and
``csrc/kernels/tpu/v6e/``. This module exposes ``detect_tpu_version()`` and a
small selector that imports the right module at runtime.

The shared implementation that both versions delegate to lives in
``_pallas_kernels.py`` (moved from ``supergrok2_jax_tpu/pallas_kernels.py``).
v5p re-exports its tile-128 variants; v6e re-exports the tile-256 variants.
The two variants share the same Python source for the algorithm; the
divergence is purely the tile size.
"""

from __future__ import annotations

# Re-export shared helpers / detection from the internal pallas module so
# downstream code can keep importing from ``csrc.kernels.tpu`` directly.
from ._pallas_kernels import (
    detect_tpu_version,
    _associative_combine,
    _HAS_PALLAS,
)


def get_kernels(kind=None):
    """Return the per-version kernel module (or kernel dict) for the active TPU.

    Args:
        kind: optional string — ``'optimizers'`` or ``'models'``.  When
            provided, returns a *dict* of kernel callables from the
            per-version sub-module's ``get_kernels(kind)``.  When
            ``None`` (the default, for backward compatibility), returns
            the sub-module itself.

    Raises:
        ``ImportError`` if the TPU version is unknown or no Pallas
        backend is available.
    """
    version = detect_tpu_version()
    if version in ("v4", "v5e", "v5p"):
        from . import v5p as mod
    elif version == "v6e":
        from . import v6e as mod
    else:
        raise ImportError(
            f"Unknown TPU version {version!r}; supported: v4/v5e/v5p, v6e. "
            "Set PALLAS_TPU_VERSION env var to override.")

    if kind is not None:
        return mod.get_kernels(kind)
    return mod


__all__ = ["detect_tpu_version", "get_kernels", "_HAS_PALLAS"]
