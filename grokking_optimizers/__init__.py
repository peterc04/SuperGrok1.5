"""Grokking Optimizers — vendor-neutral algorithm + per-vendor backend.

Package layout:

  grokking_optimizers/
    optimizers/      11 torch.optim.Optimizer subclasses (the race optimizers)
    extensions/      shared infrastructure used by the SG2 family
                     (Mamba3 meta-net, quantization, gradient hooks)
    dispatch.py      arch detection + fused kernel routing + get_ops()
    fallback.py      pure-Python reference implementations (testing only)

Usage:
    from grokking_optimizers import SuperGrok2, Lion, GrokAdamW, ...
"""

__version__ = "3.0.0"

import torch  # noqa: F401 — load libc10.so first

# --------------------------------------------------------------------------
# C++ extension capability probes
# --------------------------------------------------------------------------

from grokking_optimizers.dispatch import get_ops, detect_arch, UnsupportedArchError


_HAS_OPS = True


def _get_ops():
    return get_ops()


try:
    _ops = get_ops()
    _HAS_CUDA = hasattr(_ops, "supergrok2_mamba_peer_batched_step") or \
                hasattr(_ops, "sg15_fused_step")
    _HAS_CPU_OPS = hasattr(_ops, "supergrok2_cpu_step")
except RuntimeError:
    _HAS_CUDA = False
    _HAS_CPU_OPS = False


# --------------------------------------------------------------------------
# Core optimizers (11)
# --------------------------------------------------------------------------

from grokking_optimizers.optimizers.supergrok2 import SuperGrok2, CompiledSuperGrok2
from grokking_optimizers.optimizers.supergrok15 import SuperGrok15, SharpnessMetaNet
from grokking_optimizers.optimizers.supergrok11 import SuperGrok11
from grokking_optimizers.optimizers.grokadamw import GrokAdamW
from grokking_optimizers.optimizers.grokfast import Grokfast
from grokking_optimizers.optimizers.lion import Lion
from grokking_optimizers.optimizers.looksam import LookSAM
from grokking_optimizers.optimizers.muon import Muon
from grokking_optimizers.optimizers.neuralgrok import NeuralGrok
from grokking_optimizers.optimizers.prodigy import Prodigy
from grokking_optimizers.optimizers.moe_adam import MoEAwareSuperGrok2


# --------------------------------------------------------------------------
# Extensions (auxiliary — shared infrastructure used by SG2 family)
# --------------------------------------------------------------------------

from grokking_optimizers.extensions.mamba3_peer_metanet import (
    Mamba3PEERMetaNet, Mamba3ScanBlock, MiniGRU,
)
from grokking_optimizers.extensions.quantization import PrecisionConfig
from grokking_optimizers.extensions.gradient_hook_optimizer import GradientHookOptimizer


# --------------------------------------------------------------------------
# Dispatch / arch helpers
# --------------------------------------------------------------------------

from grokking_optimizers.dispatch import (
    get_gpu_arch, get_gpu_vendor, get_backend, get_arch_label,
    get_warp_size, supports_bf16, supports_fp8, supports_tf32,
    supports_matrix_cores, supports_nvfp4,
    supports_nvfp4_accelerated, supports_consumer_blackwell,
    supports_fp4_mfma, supports_fp6_state, supports_24_sparsity,
    SUPPORTED_ARCHES, assert_supported_arch,
)


__all__ = [
    # 11 core optimizers
    "SuperGrok2", "CompiledSuperGrok2",
    "SuperGrok15", "SharpnessMetaNet",
    "SuperGrok11",
    "GrokAdamW",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "NeuralGrok",
    "Prodigy",
    "MoEAwareSuperGrok2",

    # Extensions (shared infrastructure)
    "Mamba3PEERMetaNet", "Mamba3ScanBlock", "MiniGRU",
    "PrecisionConfig",
    "GradientHookOptimizer",

    # Dispatch helpers
    "get_gpu_arch", "get_gpu_vendor", "get_backend", "get_arch_label",
    "get_warp_size", "supports_bf16", "supports_fp8", "supports_tf32",
    "supports_matrix_cores", "supports_nvfp4",
    "supports_nvfp4_accelerated", "supports_consumer_blackwell",
    "supports_fp4_mfma", "supports_fp6_state", "supports_24_sparsity",
    "SUPPORTED_ARCHES", "UnsupportedArchError", "assert_supported_arch",
    "detect_arch",

    # Capability flags
    "_HAS_OPS", "_HAS_CUDA", "_HAS_CPU_OPS",
]
