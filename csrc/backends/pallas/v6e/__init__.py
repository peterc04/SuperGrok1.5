"""TPU v6e (Trillium) Pallas kernels (256-wide MXU tiles).

Re-exports the tile-256 variants from
``csrc/backends/pallas/_pallas_kernels.py``. Trillium's MXU is twice as wide
as the v4/v5e/v5p generations, so the affine prefix-scan and persistent
fused-scan kernels process 256-element tiles (vs. 128) to keep the MXU fully
fed. v6e additionally enables VMEM-persistent expert weights via the
``vmem_persistent_expert_mlp`` path; the version check inside that helper
honors ``detect_tpu_version()`` so the residency policy fires correctly on
both v5p and v6e.

This mirrors the sibling ``v5p`` package (the 128-wide family). Both
generations remain supported — ``detect_tpu_version()`` selects the right one
at runtime — but ``tpu_v6e`` is the primary shipped TPU target.
"""

import functools

from .._pallas_kernels import (
    mamba3_scan_pallas_tile256 as mamba3_scan,
    pallas_persistent_scan_fused_elem_tile256 as pallas_persistent_scan_fused_elem,
    vmem_persistent_expert_mlp,
    sharded_mamba3_scan as _sharded_mamba3_scan,
    detect_tpu_version,
)

from .._pallas_models import (
    attention_forward,
    attention_backward,
    decoder_forward,
    decoder_backward,
    vit_forward,
    vit_backward,
    vit_patch_project,
    mamba_forward,
    mamba_backward,
    mamba_layer_forward,
    mamba_selective_scan,
)

TILE_SIZE = 256

# Multi-device scan defaults to the 256-wide tile on v6e (vs. 128 on v5p).
sharded_mamba3_scan = functools.partial(_sharded_mamba3_scan, tile_size=256)

_OPTIMIZER_KERNELS = {
    "mamba3_scan": mamba3_scan,
    "pallas_persistent_scan_fused_elem": pallas_persistent_scan_fused_elem,
    "vmem_persistent_expert_mlp": vmem_persistent_expert_mlp,
    "sharded_mamba3_scan": sharded_mamba3_scan,
}

_MODEL_KERNELS = {
    "attention_forward": attention_forward,
    "attention_backward": attention_backward,
    "decoder_forward": decoder_forward,
    "decoder_backward": decoder_backward,
    "vit_forward": vit_forward,
    "vit_backward": vit_backward,
    "vit_patch_project": vit_patch_project,
    "mamba_forward": mamba_forward,
    "mamba_backward": mamba_backward,
    "mamba_layer_forward": mamba_layer_forward,
    "mamba_selective_scan": mamba_selective_scan,
}


def get_kernels(kind="optimizers"):
    """Return a dict of kernel callables for the requested kind.

    Args:
        kind: ``'optimizers'`` (default) for the existing optimizer/scan
            kernels, or ``'models'`` for the model forward/backward kernels.

    Returns:
        Dict mapping kernel name to callable.

    Raises:
        ValueError: if *kind* is not recognised.
    """
    if kind == "optimizers":
        return dict(_OPTIMIZER_KERNELS)
    if kind == "models":
        return dict(_MODEL_KERNELS)
    raise ValueError(
        f"Unknown kernel kind {kind!r}; expected 'optimizers' or 'models'."
    )


__all__ = [
    "mamba3_scan",
    "pallas_persistent_scan_fused_elem",
    "vmem_persistent_expert_mlp",
    "sharded_mamba3_scan",
    "detect_tpu_version",
    "TILE_SIZE",
    "get_kernels",
]
