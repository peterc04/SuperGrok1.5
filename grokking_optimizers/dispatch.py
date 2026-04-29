"""Runtime hardware detection for the all-specialized kernel architecture.

Supported arches (post-refactor): sm_80, sm_90, sm_100, gfx942.

Anything else raises ``UnsupportedArchError``. There is no tier fallback chain
and no generic-kernel path. ``FORCE_ARCH`` env var continues to work for
testing on hosts that have multiple bindings compiled in.

See REFACTOR_PLAN.md and csrc/kernels/README.md for the underlying policy.
"""

from __future__ import annotations

import functools
import os

import torch


SUPPORTED_ARCHES = (80, 90, 100, 942)


class UnsupportedArchError(RuntimeError):
    """Raised when the detected arch is not one of {sm_80, sm_90, sm_100, gfx942}."""


# ----------------------------------------------------------------------
# Vendor / backend
# ----------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_gpu_vendor() -> str:
    """GPU vendor: 'nvidia', 'amd', or 'none' (no GPU)."""
    if not torch.cuda.is_available():
        return 'none'
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return 'amd'
    return 'nvidia'


@functools.lru_cache(maxsize=1)
def get_backend() -> str:
    """Active backend: 'cuda', 'hip', or 'cpu'."""
    if not torch.cuda.is_available():
        return 'cpu'
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return 'hip'
    return 'cuda'


@functools.lru_cache(maxsize=1)
def get_warp_size() -> int:
    """Warp/wavefront size: 32 (NVIDIA), 64 (AMD CDNA)."""
    if get_gpu_vendor() == 'amd':
        return 64
    return 32


# ----------------------------------------------------------------------
# Arch detection
# ----------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> int:
    """Detected GPU arch as one of {80, 90, 100, 942}.

    For NVIDIA, the SM number is rounded into one of the supported tiers:
    sm_80/sm_86/sm_89 → 80 (Ampere family routes to sm_80 binding).
    sm_90 → 90, sm_100+ → 100.

    For AMD, only gfx942 is supported.

    Honors FORCE_ARCH env var.

    Raises ``UnsupportedArchError`` if the detected arch is not supported.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        try:
            arch = int(force)
        except ValueError:
            raise UnsupportedArchError(f"FORCE_ARCH={force!r} is not an integer")
        if arch not in SUPPORTED_ARCHES:
            raise UnsupportedArchError(
                f"FORCE_ARCH={arch} not in supported set {SUPPORTED_ARCHES}")
        return arch

    if not torch.cuda.is_available():
        raise UnsupportedArchError(
            "No CUDA/HIP device available. SuperGrok kernels are GPU-only "
            "(CPU build is for testing). Set FORCE_ARCH=<80|90|100|942> "
            "for cross-arch testing.")

    vendor = get_gpu_vendor()
    if vendor == 'nvidia':
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm in (80, 86, 89):
            return 80   # Ampere family routes to the sm_80 binding
        if sm == 90:
            return 90
        if sm >= 100:
            return 100
        raise UnsupportedArchError(
            f"Detected sm_{sm}; only sm_80, sm_90, sm_100 are supported. "
            "sm_70/sm_75 (Volta/Turing) and pre-Ampere arches are no longer "
            "supported. Use FORCE_CUDA=1 to force-build a binding.")

    if vendor == 'amd':
        prop = torch.cuda.get_device_properties(0)
        # gcnArchName looks like 'gfx942:sramecc+:xnack-'; take the prefix.
        arch_name = (prop.gcnArchName or '').split(':')[0]
        if arch_name == 'gfx942':
            return 942
        raise UnsupportedArchError(
            f"Detected {arch_name!r}; only gfx942 (MI300X) is supported. "
            "gfx908/gfx90a/gfx950 are no longer supported.")

    raise UnsupportedArchError(f"Unknown GPU vendor {vendor!r}")


# ----------------------------------------------------------------------
# Feature predicates (used by tests and Python-side precision pickers)
# ----------------------------------------------------------------------

def supports_bf16() -> bool:
    """Native BF16 matmul. True on every supported arch."""
    return True


def supports_tf32() -> bool:
    """TF32 tensor cores (NVIDIA Ampere+)."""
    return get_gpu_vendor() == 'nvidia'


def supports_fp8() -> bool:
    """FP8 E4M3 tensor cores (Hopper sm_90+ or Blackwell sm_100+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_async_copy() -> bool:
    """cp.async (Ampere+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia'


def supports_tma() -> bool:
    """TMA bulk copy (Hopper sm_90+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_block_clusters() -> bool:
    """Thread block clusters (Hopper sm_90+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_nvfp4() -> bool:
    """Native NVFP4 (Blackwell sm_100+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 100


def supports_matrix_cores() -> bool:
    """Matrix cores: AMD MFMA on gfx942, or NVIDIA Tensor Cores on sm_80+."""
    return True


# ----------------------------------------------------------------------
# Human-readable labels
# ----------------------------------------------------------------------

def get_arch_label() -> str:
    """Human-readable label for the detected arch."""
    try:
        arch = get_gpu_arch()
    except UnsupportedArchError as exc:
        return f"unsupported ({exc})"
    return {
        80:  "Ampere family (sm_80 binding)",
        90:  "Hopper (sm_90)",
        100: "Blackwell (sm_100)",
        942: "MI300X (gfx942)",
    }[arch]


# ----------------------------------------------------------------------
# Convenience for the optimizer code: assert the active arch is one we
# have a binding for, and surface a clear error if not.
# ----------------------------------------------------------------------

def assert_supported_arch() -> int:
    """Returns the detected arch or raises UnsupportedArchError."""
    return get_gpu_arch()
