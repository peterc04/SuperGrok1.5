"""Runtime hardware detection for the 3-arch fused kernel architecture.

Supported arches (3-arch active set):
    NVIDIA: sm_90 (Hopper — H100, H200)
    AMD:    gfx942 (CDNA3 — MI300X, MI300A)
    TPU:    v5p (handled via JAX backend)

Anything else raises ``UnsupportedArchError``. There is no tier fallback chain
and no generic-kernel path. ``FORCE_ARCH`` env var continues to work for
testing on hosts that have multiple bindings compiled in.

See REFACTOR_PLAN.md (esp. §10) and csrc/kernels/README.md for policy.
"""

from __future__ import annotations

import functools
import os
from typing import Union

import torch


SUPPORTED_ARCHES = (90, 942, "tpu_v5p")


class UnsupportedArchError(RuntimeError):
    """Raised when the detected arch is not one of {sm_90, gfx942, tpu_v5p}."""


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
    """Detected GPU arch as one of {90, 942}.

    For NVIDIA: only sm_90 (Hopper) is supported. Everything else raises.

    For AMD: only gfx942 is supported. Everything else raises.

    Honors FORCE_ARCH env var.

    Raises ``UnsupportedArchError`` if the detected arch is not supported.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        # Allow "tpu_v5p" to pass through for detect_arch(), but not here
        if force == "tpu_v5p":
            raise UnsupportedArchError(
                "FORCE_ARCH=tpu_v5p is not a GPU arch; use detect_arch() instead")
        try:
            arch = int(force)
        except ValueError:
            raise UnsupportedArchError(
                f"FORCE_ARCH={force!r} is not a valid arch identifier")
        if arch not in (90, 942):
            raise UnsupportedArchError(
                f"FORCE_ARCH={arch} not in supported GPU set {{90, 942}}. "
                f"3-arch active set is: sm_90, gfx942, tpu_v5p.")
        return arch

    if not torch.cuda.is_available():
        raise UnsupportedArchError(
            "No CUDA/HIP device available. SuperGrok kernels require sm_90 "
            "(Hopper) or gfx942 (MI300X/MI300A). Set FORCE_ARCH=<90|942> "
            "for cross-arch testing.")

    vendor = get_gpu_vendor()
    if vendor == 'nvidia':
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm == 90:
            return 90     # Hopper (H100, H200)
        raise UnsupportedArchError(
            f"Detected sm_{sm}; only sm_90 (Hopper) is supported in the "
            "3-arch active set. Other NVIDIA arches (sm_80, sm_89, sm_100, "
            "sm_103, sm_120) have been removed from the active set.")

    if vendor == 'amd':
        prop = torch.cuda.get_device_properties(0)
        # gcnArchName looks like 'gfx942:sramecc+:xnack-'; take the prefix.
        arch_name = (prop.gcnArchName or '').split(':')[0]
        if arch_name == 'gfx942':
            return 942
        raise UnsupportedArchError(
            f"Detected {arch_name!r}; only gfx942 (MI300X/MI300A) is "
            "supported in the 3-arch active set. gfx950 (MI350X/MI355X) "
            "and other AMD arches have been removed from the active set.")

    raise UnsupportedArchError(f"Unknown GPU vendor {vendor!r}")


@functools.lru_cache(maxsize=1)
def detect_arch() -> Union[int, str]:
    """Detect active arch: returns 90, 942, or "tpu_v5p".

    Detection order:
      1. FORCE_ARCH env var (accepts 90, 942, or "tpu_v5p")
      2. TPU detection via JAX
      3. GPU detection via get_gpu_arch()

    Raises ``UnsupportedArchError`` if no supported arch is found.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        if force == "tpu_v5p":
            return "tpu_v5p"
        try:
            arch = int(force)
        except ValueError:
            raise UnsupportedArchError(
                f"FORCE_ARCH={force!r} not in supported set {{90, 942, tpu_v5p}}")
        if arch not in (90, 942):
            raise UnsupportedArchError(
                f"FORCE_ARCH={arch} not in supported set {{90, 942, tpu_v5p}}")
        return arch

    # Try TPU detection first (check if JAX is available and running on TPU)
    try:
        import jax  # noqa: F401
        import jax.devices
        devices = jax.devices()
        if devices and any(d.platform == 'tpu' for d in devices):
            return "tpu_v5p"
    except (ImportError, RuntimeError):
        pass

    # Fall back to GPU detection
    return get_gpu_arch()


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
    """FP8 E4M3 tensor cores (Hopper sm_90+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_async_copy() -> bool:
    """cp.async (NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia'


def supports_tma() -> bool:
    """TMA bulk copy (Hopper sm_90)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_block_clusters() -> bool:
    """Thread block clusters (Hopper sm_90)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_matrix_cores() -> bool:
    """Matrix cores: AMD MFMA on gfx942, or NVIDIA Tensor Cores on sm_90."""
    return True


# ----------------------------------------------------------------------
# Human-readable labels
# ----------------------------------------------------------------------

def get_arch_label() -> str:
    """Human-readable label for the detected arch."""
    try:
        arch = detect_arch()
    except UnsupportedArchError as exc:
        return f"unsupported ({exc})"
    return {
        90:       "Hopper (sm_90) — H100, H200",
        942:      "CDNA3 (gfx942) — MI300X / MI300A",
        "tpu_v5p": "TPU v5p",
    }[arch]


# ----------------------------------------------------------------------
# Convenience for the optimizer code: assert the active arch is one we
# have a binding for, and surface a clear error if not.
# ----------------------------------------------------------------------

def assert_supported_arch() -> Union[int, str]:
    """Returns the detected arch or raises UnsupportedArchError."""
    return detect_arch()
