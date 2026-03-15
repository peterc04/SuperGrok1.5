"""Runtime hardware detection and kernel dispatch.

Detects GPU architecture at import time and provides dispatch helpers
for selecting the optimal kernel variant per hardware.
"""

import functools
import torch


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> int:
    """SM compute capability as integer (e.g., 75 for T4, 80 for A100).
    Returns 0 if no CUDA GPU available."""
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


def supports_fp8() -> bool:
    """FP8 Tensor Cores (Ada Lovelace sm_89+ or Hopper sm_90+)."""
    return get_gpu_arch() >= 89


def supports_bf16() -> bool:
    """Native BF16 (Ampere sm_80+)."""
    return get_gpu_arch() >= 80


def supports_tf32() -> bool:
    """TF32 Tensor Core mode (Ampere sm_80+)."""
    return get_gpu_arch() >= 80


def supports_async_copy() -> bool:
    """cp.async global->shared (Ampere sm_80+)."""
    return get_gpu_arch() >= 80


def supports_tma() -> bool:
    """Tensor Memory Accelerator (Hopper sm_90+)."""
    return get_gpu_arch() >= 90


def supports_block_clusters() -> bool:
    """Thread Block Clusters (Hopper sm_90+)."""
    return get_gpu_arch() >= 90


def supports_nvfp4() -> bool:
    """NVFP4 native (Blackwell sm_100+)."""
    return get_gpu_arch() >= 100


def get_arch_label() -> str:
    """Human-readable label for the detected GPU."""
    arch = get_gpu_arch()
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
