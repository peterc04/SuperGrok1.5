"""TPU v6e Pallas kernels (256-wide MXU tiles).

Re-exports the tile-256 variants from ``csrc/kernels/tpu/_pallas_kernels.py``.
v6e benefits from both the wider MXU tile size and the VMEM-persistent
expert-weight residency policy.

When v6e-specific divergence is needed beyond tile size (e.g. v6e-only
HBM layout hints, larger VMEM working set), put the new code here.
"""

from .._pallas_kernels import (
    mamba3_scan_pallas_tile256 as mamba3_scan,
    pallas_persistent_scan_fused_elem_tile256 as pallas_persistent_scan_fused_elem,
    vmem_persistent_expert_mlp,
    sharded_mamba3_scan,  # multi-device, pass tile_size=256 explicitly
    detect_tpu_version,
)

TILE_SIZE = 256

__all__ = [
    "mamba3_scan",
    "pallas_persistent_scan_fused_elem",
    "vmem_persistent_expert_mlp",
    "sharded_mamba3_scan",
    "detect_tpu_version",
    "TILE_SIZE",
]
