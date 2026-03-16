"""
Grokking Optimizers — C++/CUDA/HIP/CPU Accelerated Optimizer Suite

All optimizers use custom GPU kernels for maximum performance.
Supports NVIDIA CUDA, AMD ROCm/HIP, and CPU-only (for debugging/testing).
Supports FP32, FP16, and BF16 parameter tensors.

Usage:
    from grokking_optimizers import SuperGrok15, SuperGrok2, GrokAdamW, ...
"""

__version__ = "2.1.0"

# ── Backend capability flags ─────────────────────────────────────────
# Three-tier detection:
#   _HAS_OPS  = True  → C++ extension loaded (GPU or CPU build)
#   _HAS_CUDA = True  → GPU kernels available (CUDA or HIP build)
#   _HAS_CPU_OPS = True → CPU C++ kernels available (CPU build)
# If _HAS_OPS is False, the pure-Python fallback in _python_fallback.py
# is used automatically by every optimizer.
_HAS_OPS = False
_HAS_CUDA = False
_HAS_CPU_OPS = False

try:
    import torch  # noqa: F401 — must import torch first to load libc10.so
    from . import _ops  # noqa: F401
    _HAS_OPS = True
    _HAS_CUDA = hasattr(_ops, 'supergrok2_mamba_peer_batched_step')
    _HAS_CPU_OPS = hasattr(_ops, 'supergrok2_cpu_step')
except Exception:
    _ops = None

from .supergrok15 import SuperGrok15, SharpnessMetaNet
from .supergrok2 import SuperGrok2, CompiledSuperGrok2
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
    supports_matrix_cores, supports_nvfp4,
    get_amd_tier, get_amd_label,
)
from .quantization import PrecisionConfig
from .distributed import (
    setup_distributed, cleanup_distributed,
    get_rank, get_world_size, is_main_process,
    broadcast_optimizer_state, wrap_model_ddp,
)
from .overlap_distributed import OverlappedOptimizer, OverlappedSuperGrok2
from .gradient_compression import INT8GradientCompressor, PowerSGDCompressor
from .partial_graph import PartialGraphOptimizer
from .sparse_gradients import SparseGradientHandler
from .pipelined_optimizer import PipelinedOptimizer

__all__ = [
    "SuperGrok15", "SharpnessMetaNet",
    "SuperGrok2", "CompiledSuperGrok2",
    "Mamba3PEERMetaNet", "Mamba3ScanBlock", "MiniGRU",
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
    "supports_matrix_cores", "supports_nvfp4",
    "get_amd_tier", "get_amd_label",
    "PrecisionConfig",
    "CompiledSuperGrok2",
    "setup_distributed", "cleanup_distributed",
    "get_rank", "get_world_size", "is_main_process",
    "broadcast_optimizer_state", "wrap_model_ddp",
    "_HAS_OPS", "_HAS_CUDA", "_HAS_CPU_OPS",
    "PartialGraphOptimizer",
    "SparseGradientHandler",
    "PipelinedOptimizer",
]
