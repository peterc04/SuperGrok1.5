"""Runtime hardware detection for the all-specialized kernel architecture.

Supported arches (8 GPU + 2 TPU):
    NVIDIA: sm_80, sm_89, sm_90, sm_100, sm_103, sm_120
    AMD:    gfx942, gfx950
    TPU:    v5p, v6e (handled by csrc/kernels/tpu/, not this module)

Anything else raises ``UnsupportedArchError``. There is no tier fallback chain
and no generic-kernel path. ``FORCE_ARCH`` env var continues to work for
testing on hosts that have multiple bindings compiled in.

See REFACTOR_PLAN.md (esp. §10) and csrc/kernels/README.md for policy.
"""

from __future__ import annotations

import functools
import os

import torch


SUPPORTED_ARCHES = (80, 89, 90, 100, 103, 120, 942, 950)


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
    """Detected GPU arch as one of {80, 89, 90, 100, 103, 120, 942, 950}.

    For NVIDIA: sm_80/sm_86 → 80 (Ampere family routes to sm_80 binding);
    sm_89 → 89; sm_90 → 90; sm_100 → 100; sm_103 → 103; sm_120+ → 120.

    For AMD: gfx942 → 942; gfx950 → 950. Anything else raises.

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
            "(CPU build is for testing). Set FORCE_ARCH=<80|89|90|100|103|"
            "120|942|950> for cross-arch testing.")

    vendor = get_gpu_vendor()
    if vendor == 'nvidia':
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm in (80, 86):
            return 80     # A100/A30/A10/RTX 30 (Ampere family) → sm_80 binding
        if sm == 89:
            return 89     # Ada (RTX 40, L40, L40S)
        if sm == 90:
            return 90     # Hopper (H100, H200)
        if sm == 100:
            return 100    # Datacenter Blackwell (B100/B200/GB200)
        if sm == 103:
            return 103    # Blackwell Ultra (B300, GB300 NVL72)
        if sm == 120 or sm > 120:
            return 120    # Consumer Blackwell (RTX 50, RTX PRO 6000)
        raise UnsupportedArchError(
            f"Detected sm_{sm}; supported NVIDIA arches are sm_80, sm_89, "
            "sm_90, sm_100, sm_103, sm_120. Pre-Ampere (sm_70/75) is "
            "permanently unsupported. Use FORCE_CUDA=1 to force-build a "
            "binding.")

    if vendor == 'amd':
        prop = torch.cuda.get_device_properties(0)
        # gcnArchName looks like 'gfx942:sramecc+:xnack-'; take the prefix.
        arch_name = (prop.gcnArchName or '').split(':')[0]
        if arch_name == 'gfx942':
            return 942
        if arch_name == 'gfx950':
            return 950
        raise UnsupportedArchError(
            f"Detected {arch_name!r}; supported AMD arches are gfx942 "
            "(MI300X/MI300A) and gfx950 (MI350X/MI355X). gfx908/gfx90a "
            "(MI100/MI200) and RDNA cards are unsupported.")

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


def supports_nvfp4_accelerated() -> bool:
    """Blackwell Ultra (sm_103+) has 1.5x NVFP4 throughput vs sm_100."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() == 103


def supports_consumer_blackwell() -> bool:
    """Consumer Blackwell (sm_120): 128KB shared memory, no DSMT."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() == 120


def supports_fp4_mfma() -> bool:
    """Native FP4 expert MFMA (AMD CDNA4 / gfx950)."""
    return get_gpu_vendor() == 'amd' and get_gpu_arch() == 950


def supports_fp6_state() -> bool:
    """Native FP6 E3M2 optimizer state (AMD CDNA4 / gfx950)."""
    return get_gpu_vendor() == 'amd' and get_gpu_arch() == 950


def supports_24_sparsity() -> bool:
    """Hardware 2:4 structured sparsity (AMD CDNA4 / gfx950)."""
    return get_gpu_vendor() == 'amd' and get_gpu_arch() == 950


def supports_matrix_cores() -> bool:
    """Matrix cores: AMD MFMA on gfx942/gfx950, or NVIDIA Tensor Cores on sm_80+."""
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
        80:  "Ampere family (sm_80 binding) — A100/A30/A10/RTX 30",
        89:  "Ada (sm_89) — RTX 40, L40, L40S",
        90:  "Hopper (sm_90) — H100, H200",
        100: "Datacenter Blackwell (sm_100) — B100/B200/GB200",
        103: "Blackwell Ultra (sm_103) — B300, GB300 NVL72",
        120: "Consumer Blackwell (sm_120) — RTX 50, RTX PRO 6000 Blackwell",
        942: "CDNA3 (gfx942) — MI300X / MI300A",
        950: "CDNA4 (gfx950) — MI350X / MI355X",
    }[arch]


# ----------------------------------------------------------------------
# Convenience for the optimizer code: assert the active arch is one we
# have a binding for, and surface a clear error if not.
# ----------------------------------------------------------------------

def assert_supported_arch() -> int:
    """Returns the detected arch or raises UnsupportedArchError."""
    return get_gpu_arch()
