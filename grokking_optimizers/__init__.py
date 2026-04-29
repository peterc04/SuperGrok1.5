"""
Grokking Optimizers — All-Specialized Per-Arch Kernel Suite

Every optimizer is hand-written and fully specialized for one of:
  sm_80 (Ampere family)  sm_90 (Hopper)  sm_100 (Blackwell)  gfx942 (MI300X)

There is no generic-kernel fallback and no tier fallback chain. The build
fails on unsupported arches; the CPU build is for testing only.

Usage:
    from grokking_optimizers import SuperGrok15, SuperGrok2, GrokAdamW, ...
"""

__version__ = "3.0.0"

# ── Backend capability flags ─────────────────────────────────────────
# No fallbacks. If the C++ extension is not built or the host arch is
# not supported, get_ops() fails loudly.
from grokking_optimizers._ops_loader import get_ops

import torch  # noqa: F401 — must import torch first to load libc10.so

_HAS_OPS = True  # Always True in production. Tests mock this.

def _get_ops():
    return get_ops()

try:
    _ops = get_ops()
    # Capability probes used by some Python paths to skip features not
    # registered in the current build (e.g. SG2 batched step is the
    # primary entry point that gates the meta-net pipeline).
    _HAS_CUDA = hasattr(_ops, 'supergrok2_mamba_peer_batched_step') or \
                hasattr(_ops, 'sg15_fused_step')  # new bindings layer
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
    supports_nvfp4_accelerated, supports_consumer_blackwell,
    supports_fp4_mfma, supports_fp6_state, supports_24_sparsity,
    SUPPORTED_ARCHES, UnsupportedArchError, assert_supported_arch,
)
from .quantization import PrecisionConfig
from .distributed import (
    setup_distributed, cleanup_distributed,
    get_rank, get_world_size, is_main_process,
    broadcast_optimizer_state, wrap_model_ddp,
)
from .moe_deep import MoEAwareSuperGrok2
from .overlap_distributed import OverlappedOptimizer
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
    "supports_nvfp4_accelerated", "supports_consumer_blackwell",
    "supports_fp4_mfma", "supports_fp6_state", "supports_24_sparsity",
    "SUPPORTED_ARCHES", "UnsupportedArchError", "assert_supported_arch",
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
    "MoEAwareSuperGrok2",
    "OverlappedOptimizer",
]
