"""Runtime hardware detection and kernel dispatch.

Detects GPU vendor (NVIDIA/AMD) and architecture at import time.
Provides dispatch helpers for selecting the optimal kernel variant per hardware.

Set FORCE_ARCH=<sm_number> to override detected architecture for testing.
E.g., FORCE_ARCH=80 forces Ampere tier on any GPU.
"""

import functools
import os
import torch


@functools.lru_cache(maxsize=1)
def get_gpu_vendor() -> str:
    """GPU vendor: 'nvidia', 'amd', or 'none'."""
    if not torch.cuda.is_available():
        return 'none'
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return 'amd'
    return 'nvidia'


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> int:
    """SM compute capability as integer (e.g., 75 for T4, 80 for A100).
    For AMD, returns the GCN arch number (e.g., 90 for gfx90a, 94 for gfx942).
    Returns 0 if no GPU available.
    Honors FORCE_ARCH env var for testing."""
    force = os.environ.get('FORCE_ARCH')
    if force:
        return int(force)
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@functools.lru_cache(maxsize=1)
def get_backend() -> str:
    """Active backend: 'cuda', 'hip', or 'cpu'."""
    if torch.cuda.is_available():
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return 'hip'
        return 'cuda'
    return 'cpu'


@functools.lru_cache(maxsize=1)
def get_warp_size() -> int:
    """Warp/wavefront size for the current GPU.
    NVIDIA: 32, AMD CDNA (MI100/200/300): 64, AMD RDNA: 32."""
    if get_gpu_vendor() == 'amd':
        # CDNA architectures use wavefront-64
        # RDNA architectures use wavefront-32 but we target data center GPUs
        prop = torch.cuda.get_device_properties(0)
        name = prop.name.lower()
        if 'rdna' in name or 'rx' in name:
            return 32
        return 64  # CDNA default
    return 32  # NVIDIA


def get_arch_tier() -> str:
    """NVIDIA architecture tier: 'blackwell', 'hopper', 'ampere', or 'generic'."""
    if get_gpu_vendor() == 'amd':
        return 'generic'
    arch = get_gpu_arch()
    if arch >= 100:
        return 'blackwell'
    if arch >= 90:
        return 'hopper'
    if arch >= 80:
        return 'ampere'
    return 'generic'


@functools.lru_cache(maxsize=1)
def get_amd_tier() -> str:
    """AMD architecture tier: 'cdna4', 'cdna3', 'cdna2', or 'generic'.

    FORCE_ARCH convention (matches C++ get_amd_tier):
      1200 → cdna4 (gfx1200, full GCN arch number)
      120  → cdna4 (from get_device_capability major*10+minor)
      942  → cdna3 (full GCN arch number)
      94   → cdna3 (from get_device_capability major*10+minor)
      90   → cdna2 (gfx90a)
      908  → generic (MI100, full GCN arch)
      else → generic
    """
    if get_gpu_vendor() != 'amd':
        return 'generic'

    # Handle FORCE_ARCH directly to match C++ convention
    force = os.environ.get('FORCE_ARCH')
    if force:
        arch = int(force)
        if arch >= 1200: return 'cdna4'   # gfx1200/gfx1201 (full arch number)
        if arch >= 942:  return 'cdna3'   # gfx942 (full arch number)
        if arch == 120:  return 'cdna4'   # capability format (12, 0)
        if arch == 94:   return 'cdna3'   # capability format (9, 4)
        if arch == 90:   return 'cdna2'   # gfx90a
        return 'generic'                   # gfx908, etc.

    # Real hardware: get_gpu_arch returns major*10+minor
    arch = get_gpu_arch()
    if arch >= 120:  # gfx1200 → (12, 0) → 120
        return 'cdna4'
    if arch >= 94:   # gfx942 → (9, 4) → 94
        return 'cdna3'
    if arch >= 90:   # gfx90a → (9, 0) → 90
        return 'cdna2'
    return 'generic'


def get_amd_label() -> str:
    """Human-readable label for AMD GPU tier."""
    tier = get_amd_tier()
    labels = {
        'cdna4': 'MI400 (gfx1200, CDNA4)',
        'cdna3': 'MI300X (gfx942, CDNA3)',
        'cdna2': 'MI250 (gfx90a, CDNA2)',
        'generic': 'AMD GPU (generic CDNA)',
    }
    return labels.get(tier, 'AMD GPU')


def supports_fp8() -> bool:
    """FP8 Tensor Cores (Ada Lovelace sm_89+ or Hopper sm_90+)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 89


def supports_bf16() -> bool:
    """Native BF16 (Ampere sm_80+ or AMD CDNA gfx90a+)."""
    if get_gpu_vendor() == 'amd':
        return get_gpu_arch() >= 90  # gfx90a+ has BF16 matrix cores
    return get_gpu_arch() >= 80


def supports_tf32() -> bool:
    """TF32 Tensor Core mode (Ampere sm_80+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 80


def supports_async_copy() -> bool:
    """cp.async global->shared (Ampere sm_80+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 80


def supports_tma() -> bool:
    """Tensor Memory Accelerator (Hopper sm_90+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_block_clusters() -> bool:
    """Thread Block Clusters (Hopper sm_90+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 90


def supports_nvfp4() -> bool:
    """NVFP4 native (Blackwell sm_100+, NVIDIA only)."""
    return get_gpu_vendor() == 'nvidia' and get_gpu_arch() >= 100


def supports_fp4_mfma() -> bool:
    """Native FP4 MFMA instructions (AMD CDNA4 gfx1200+)."""
    return get_amd_tier() == 'cdna4'


def supports_fp6() -> bool:
    """Enhanced FP6 support (AMD CDNA3+: gfx940/gfx941/gfx942 and gfx1200+)."""
    return get_amd_tier() in ('cdna3', 'cdna4')


def supports_matrix_cores() -> bool:
    """AMD Matrix Cores (CDNA gfx908+)."""
    return get_gpu_vendor() == 'amd' and get_gpu_arch() >= 90


def get_arch_label() -> str:
    """Human-readable label for the detected GPU."""
    arch = get_gpu_arch()
    vendor = get_gpu_vendor()

    if vendor == 'amd':
        labels = {
            120: "MI400 (gfx1200)",
            90: "MI200 (gfx90a)",
            94: "MI300X (gfx942)",
        }
        return labels.get(arch, f"AMD GPU (gfx{arch})")

    labels = {
        0: "CPU (no GPU)",
        70: "V100 (sm_70)",
        75: "T4 (sm_75)",
        80: "A100 (sm_80)",
        86: "RTX 3090 / A10 (sm_86)",
        89: "L4 / RTX 4090 (sm_89)",
        90: "H100 (sm_90)",
        100: "B200 (sm_100)",
    }
    return labels.get(arch, f"Unknown (sm_{arch})")
