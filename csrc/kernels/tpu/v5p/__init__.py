"""TPU v4 / v5e / v5p Pallas kernels (128-wide MXU tiles).

Re-exports the tile-128 variants from ``csrc/kernels/tpu/_pallas_kernels.py``.
v5p additionally enables VMEM-persistent expert weights via the
``vmem_persistent_expert_mlp`` path; the version check inside that helper
honors ``detect_tpu_version()`` so the residency policy fires correctly.

When the algorithm needs to diverge between v4/v5e/v5p in a way that goes
beyond tile size, this module is the place to add the divergence (split
into per-version submodules and have ``__init__.py`` pick the right one).
"""

from .._pallas_kernels import (
    mamba3_scan_pallas_tile128 as mamba3_scan,
    pallas_persistent_scan_fused_elem_tile128 as pallas_persistent_scan_fused_elem,
    vmem_persistent_expert_mlp,
    sharded_mamba3_scan,           # multi-device, tile_size kwarg defaults to 128
    detect_tpu_version,
)

TILE_SIZE = 128

__all__ = [
    "mamba3_scan",
    "pallas_persistent_scan_fused_elem",
    "vmem_persistent_expert_mlp",
    "sharded_mamba3_scan",
    "detect_tpu_version",
    "TILE_SIZE",
]
