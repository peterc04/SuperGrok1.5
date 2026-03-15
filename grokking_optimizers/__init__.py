"""
Grokking Optimizers — C++/CUDA/HIP Accelerated Optimizer Suite

All optimizers use custom GPU kernels for maximum performance.
Supports NVIDIA CUDA and AMD ROCm/HIP backends.
Supports FP32, FP16, and BF16 parameter tensors.

Usage:
    from grokking_optimizers import SuperGrok15, SuperGrok2, GrokAdamW, ...
"""

__version__ = "2.0.0"

try:
    from grokking_optimizers import _ops  # noqa: F401
except ImportError as e:
    raise ImportError(
        "grokking_optimizers C++/CUDA/HIP extension not found. "
        "Build with: pip install -e . (from the repo root). "
        f"Original error: {e}"
    ) from e

from .supergrok15 import SuperGrok15, SharpnessMetaNet
from .supergrok2 import SuperGrok2
from .mamba3_peer_metanet import Mamba3PEERMetaNet, Mamba3ScanBlock, MiniGRU
from .supergrok11 import SuperGrok11
from .grokadamw import GrokAdamW
from .neuralgrok import NeuralGrok
from .prodigy import Prodigy
from .grokfast import Grokfast
from .lion import Lion
from .looksam import LookSAM
from .muon import Muon
from .cuda_graph_optimizer import CUDAGraphOptimizer
from .dispatch import (
    get_gpu_arch, get_gpu_vendor, get_backend, get_arch_label,
    get_warp_size, supports_bf16, supports_fp8, supports_tf32,
    supports_matrix_cores,
)
from .quantization import PrecisionConfig

__all__ = [
    "SuperGrok15", "SharpnessMetaNet",
    "SuperGrok2", "Mamba3PEERMetaNet", "Mamba3ScanBlock", "MiniGRU",
    "SuperGrok11",
    "GrokAdamW",
    "NeuralGrok",
    "Prodigy",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "CUDAGraphOptimizer",
    "get_gpu_arch", "get_gpu_vendor", "get_backend", "get_arch_label",
    "get_warp_size", "supports_bf16", "supports_fp8", "supports_tf32",
    "supports_matrix_cores",
    "PrecisionConfig",
]
