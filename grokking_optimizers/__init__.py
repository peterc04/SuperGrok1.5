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
# No fallbacks. If the C++ extension is not built, get_ops() fails loudly.
# _HAS_OPS is always True in production. Tests mock this.
from grokking_optimizers._ops_loader import get_ops

import torch  # noqa: F401 — must import torch first to load libc10.so

_HAS_OPS = True  # Always True in production. Tests mock this.

def _get_ops():
    return get_ops()

try:
    _ops = get_ops()
    _HAS_CUDA = hasattr(_ops, 'supergrok2_mamba_peer_batched_step')
    _HAS_CPU_OPS = hasattr(_ops, 'supergrok2_cpu_step')
except RuntimeError:
    # Extension not built — flags remain for import-time checks,
    # but any actual use of _ops will fail loudly via get_ops()
    _HAS_CUDA = False
    _HAS_CPU_OPS = False

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
from .gradient_hook_optimizer import GradientHookOptimizer
from .async_supergrok2 import AsyncSuperGrok2

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
    "GradientHookOptimizer",
    "AsyncSuperGrok2",
]
