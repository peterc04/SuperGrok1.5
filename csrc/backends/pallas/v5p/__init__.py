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

TILE_SIZE = 128

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
