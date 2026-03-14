"""
Grokking Optimizers — C++/CUDA Accelerated Optimizer Suite

All optimizers use custom CUDA kernels for maximum performance.
Supports FP32, FP16, and BF16 parameter tensors.

Usage:
    from grokking_optimizers import SuperGrok15, SuperGrok2, GrokAdamW, ...
"""

__version__ = "2.0.0"

try:
    from grokking_optimizers import _ops  # noqa: F401
except ImportError as e:
    raise ImportError(
        "grokking_optimizers C++/CUDA extension not found. "
        "Build with: pip install -e . (from the grokking_optimizers directory). "
        f"Original error: {e}"
    ) from e

from .supergrok15 import SuperGrok15, SharpnessMetaNet
from .supergrok2 import SuperGrok2
from .isab_peer_metanet import ISABPEERMetaNet
from .supergrok11 import SuperGrok11
from .grokadamw import GrokAdamW
from .neuralgrok import NeuralGrok
from .prodigy import Prodigy
from .grokfast import Grokfast
from .lion import Lion
from .looksam import LookSAM
from .muon import Muon
from .cuda_graph_optimizer import CUDAGraphOptimizer

__all__ = [
    "SuperGrok15", "SharpnessMetaNet",
    "SuperGrok2", "ISABPEERMetaNet",
    "SuperGrok11",
    "GrokAdamW",
    "NeuralGrok",
    "Prodigy",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "CUDAGraphOptimizer",
]
