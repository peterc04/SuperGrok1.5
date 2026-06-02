"""Grokking Optimizers — vendor-neutral algorithm + per-vendor backend.

Package layout:

  grokking_optimizers/
    optimizers/         11 torch.optim.Optimizer subclasses (the race optimizers)
    dispatch.py         arch detection + fused kernel routing + get_ops()

Each optimizer file is fully self-contained — the Mamba3+PEER metanet and
PrecisionConfig live inside supergrok2.py; SharpnessMetaNet is duplicated
between supergrok15.py and supergrok11.py. Those implementation-detail
classes are reachable via
`from grokking_optimizers.optimizers.supergrok2 import X` when needed.

The C++ kernel is the only execution path — if the extension isn't built or
a kernel raises, the exception propagates. There is no Python reference
fallback.

Usage:
    from grokking_optimizers import SuperGrok2, Lion, GrokAdamW, ...
"""

__version__ = "3.0.0"

import torch  # noqa: F401 — load libc10.so first

# --------------------------------------------------------------------------
# C++ extension capability probes
# --------------------------------------------------------------------------

from grokking_optimizers.dispatch import get_ops, detect_arch, UnsupportedArchError


_ops = get_ops()
_HAS_OPS = bool(_ops)
_HAS_CUDA = hasattr(_ops, "supergrok2_mamba_peer_batched_step") or \
            hasattr(_ops, "sg15_fused_step")
_HAS_CPU_OPS = hasattr(_ops, "supergrok2_cpu_step")


def _get_ops():
    return get_ops()


# --------------------------------------------------------------------------
# Core optimizers (11)
# --------------------------------------------------------------------------

from grokking_optimizers.optimizers.supergrok2 import (
    SuperGrok2, CompiledSuperGrok2, MoEAwareSuperGrok2,
)
from grokking_optimizers.optimizers.supergrok15 import SuperGrok15
from grokking_optimizers.optimizers.supergrok11 import SuperGrok11
from grokking_optimizers.optimizers.adamw import AdamW
from grokking_optimizers.optimizers.grokadamw import GrokAdamW
from grokking_optimizers.optimizers.grokfast import Grokfast
from grokking_optimizers.optimizers.lion import Lion
from grokking_optimizers.optimizers.looksam import LookSAM
from grokking_optimizers.optimizers.muon import Muon
from grokking_optimizers.optimizers.neuralgrok import NeuralGrok
from grokking_optimizers.optimizers.prodigy import Prodigy


# --------------------------------------------------------------------------
# Dispatch / arch helpers
# --------------------------------------------------------------------------

from grokking_optimizers.dispatch import (
    get_gpu_arch, get_gpu_vendor, get_backend, get_arch_label,
    get_warp_size, supports_bf16, supports_fp8, supports_tf32,
    supports_matrix_cores,
    SUPPORTED_ARCHES, assert_supported_arch,
)


# Single source of truth for every supported GPU/TPU architecture. See
# grokking_optimizers/compile.py for the ArchEntry dataclass and the table
# itself. ARCH_INFO is a derived 6-key dict kept for backward compatibility
# with older call sites that read e.g. ARCH_INFO[arch]["vendor"].
#
# These four symbols live in compile.py — a ~23k-LOC module whose parse cost
# we do NOT want to pay on every ``import grokking_optimizers``. They are
# resolved lazily via the module-level __getattr__ below (PEP 562), so the
# import is deferred until ARCH_TABLE/ARCH_INFO/ArchEntry/get_arch_entry is
# first accessed.
_COMPILE_LAZY_EXPORTS = frozenset(
    {"ARCH_TABLE", "ARCH_INFO", "ArchEntry", "get_arch_entry"})


def __getattr__(name):
    # PEP 562 module-level __getattr__: only invoked for names not found in the
    # module's normal namespace, so it costs nothing until one of the lazy
    # compile.py exports is actually accessed.
    if name in _COMPILE_LAZY_EXPORTS:
        import importlib
        compile_mod = importlib.import_module("grokking_optimizers.compile")
        value = getattr(compile_mod, name)
        globals()[name] = value  # cache so __getattr__ isn't hit again
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 12 core optimizers (11 grokking + AdamW baseline)
    "AdamW",
    "SuperGrok2", "CompiledSuperGrok2",
    "SuperGrok15",
    "SuperGrok11",
    "GrokAdamW",
    "Grokfast",
    "Lion",
    "LookSAM",
    "Muon",
    "NeuralGrok",
    "Prodigy",
    "MoEAwareSuperGrok2",

    # Dispatch helpers
    "get_gpu_arch", "get_gpu_vendor", "get_backend", "get_arch_label",
    "get_warp_size", "supports_bf16", "supports_fp8", "supports_tf32",
    "supports_matrix_cores",
    "SUPPORTED_ARCHES", "UnsupportedArchError", "assert_supported_arch",
    "detect_arch",

    # ARCH_TABLE — single source of truth for every supported arch
    "ARCH_TABLE", "ARCH_INFO", "ArchEntry", "get_arch_entry",

    # Capability flags
    "_HAS_OPS", "_HAS_CUDA", "_HAS_CPU_OPS",
]
