"""Pallas / JAX primitives — vendor-specific TPU helpers.

The optimizer math itself is inlined per launch_<optimizer>.py file
(matching the self-contained-optimizer policy used elsewhere in the
codebase). What remains here is only the vendor-specific surface:

  - TPU version detection (v4 / v5e / v5p / v6e)
  - MXU lane width helper
  - Re-exports of the working Pallas kernels in csrc/kernels/tpu/
    (consumed by launch_supergrok2.py)
"""

from __future__ import annotations

import jax


def detect_tpu_version() -> str:
    """Return the TPU generation string (\"v4\", \"v5e\", \"v5p\", \"v6e\") or \"cpu\"."""
    try:
        devs = jax.devices()
        if not devs:
            return "cpu"
        kind = getattr(devs[0], "device_kind", "") or ""
        kind = kind.lower()
        for v in ("v6e", "v5p", "v5e", "v4"):
            if v in kind:
                return v
        return "v4"  # sentinel
    except Exception:
        return "v4"


def mxu_tile_width() -> int:
    """MXU lane width for the active TPU. 128 for v4/v5e/v5p, 256 for v6e."""
    v = detect_tpu_version()
    return 256 if v == "v6e" else 128


try:
    from csrc.backends.pallas._pallas_kernels import (
        mamba3_scan_pallas_tile128,
        mamba3_scan_pallas_tile256,
        pallas_persistent_scan_fused_elem_tile128,
        pallas_persistent_scan_fused_elem_tile256,
        pallas_fused_gru_peer,
        tpu_auto_select_scan,
        tpu_auto_select_fused_scan_elem,
    )
    _PALLAS_AVAILABLE = True
except ImportError:
    _PALLAS_AVAILABLE = False


def pallas_available() -> bool:
    return _PALLAS_AVAILABLE
