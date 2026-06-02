"""Runtime hardware detection for the fused kernel architecture.

The kernels are single-source-per-vendor and the C++ extension is built as a
multi-arch fat binary (every NVIDIA CC / AMD gfx the toolchain accepts; see
setup.py). Detection therefore normalises to a VENDOR impl selector:

    NVIDIA (any sm_70..sm_120) -> 90  (the sg::sm90 impl)
    AMD    (any gfx906..gfx1201) -> 942 (the sg::gfx942 impl)
    TPU                          -> "tpu_v5p" (handled via JAX backend)

The driver loads the matching per-SM/-gfx code from the fat binary; the host
only needs to pick the vendor impl. Feature predicates (``supports_fp8`` etc.)
are keyed to the REAL compute capability via ``get_device_sm()``, not the
normalised selector. ``FORCE_ARCH`` accepts any of the above forms.
"""

from __future__ import annotations

import functools
import os
from typing import Union

import torch


SUPPORTED_ARCHES = (90, 942, "tpu_v5p")


class UnsupportedArchError(RuntimeError):
    """Raised when the detected arch is not one of {sm_90, gfx942, tpu_v5p}."""


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

def _normalize_force_arch(force: str):
    """Map a FORCE_ARCH string to the vendor impl selector, or None for TPU.

    Returns 90 (NVIDIA), 942 (AMD), or "tpu_v5p". Raises on unrecognised input.
    """
    if force.startswith('tpu'):
        return "tpu_v5p"
    # AMD: any gfx target (and the bare 942 form).
    if force == '942' or force.startswith('gfx'):
        return 942
    # NVIDIA: sm_* / smXX, or a bare numeric compute capability.
    if force.startswith('sm'):
        return 90
    if force.isdigit():
        return 90
    raise UnsupportedArchError(
        f"FORCE_ARCH={force!r} not recognized. Use an NVIDIA arch "
        f"(sm_70..sm_120 or a numeric CC), an AMD arch (gfx906..gfx1201), "
        f"or tpu_v5p.")


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> int:
    """Detected GPU vendor impl selector: 90 (NVIDIA) or 942 (AMD).

    Any NVIDIA device (sm_70..sm_120) normalises to 90; any AMD gfx device
    normalises to 942. The fat binary carries the matching per-arch code.

    Honors FORCE_ARCH. Raises ``UnsupportedArchError`` if no GPU is available
    or the vendor is unknown.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        sel = _normalize_force_arch(force)
        if sel == "tpu_v5p":
            raise UnsupportedArchError(
                "FORCE_ARCH=tpu_v5p is not a GPU arch; use detect_arch() instead")
        return sel

    if not torch.cuda.is_available():
        raise UnsupportedArchError(
            "No CUDA/HIP device available. Set FORCE_ARCH (e.g. 90 for any "
            "NVIDIA, 942 for any AMD) for cross-arch testing.")

    vendor = get_gpu_vendor()
    if vendor == 'nvidia':
        return 90     # any sm_70..sm_120 -> the sm90 impl
    if vendor == 'amd':
        return 942    # any gfx906..gfx1201 -> the gfx942 impl
    raise UnsupportedArchError(f"Unknown GPU vendor {vendor!r}")


@functools.lru_cache(maxsize=1)
def get_device_sm():
    """Real NVIDIA compute capability (e.g. 70, 80, 90, 120), or None.

    None when there is no NVIDIA device (CPU host or AMD). Used by the feature
    predicates, which need the true capability rather than the normalised
    vendor selector returned by ``get_gpu_arch()``.
    """
    if get_gpu_vendor() != 'nvidia':
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@functools.lru_cache(maxsize=1)
def detect_arch() -> Union[int, str]:
    """Detect active arch: returns 90, 942, or "tpu_v5p".

    Detection order:
      1. FORCE_ARCH env var (accepts 90, 942, or "tpu_v5p")
      2. TPU detection via JAX
      3. GPU detection via get_gpu_arch()

    Raises ``UnsupportedArchError`` if no supported arch is found.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        return _normalize_force_arch(force)

    # Try TPU detection first (check if JAX is available and running on TPU)
    try:
        import jax  # noqa: F401
        import jax.devices
        devices = jax.devices()
        if devices and any(d.platform == 'tpu' for d in devices):
            return "tpu_v5p"
    except (ImportError, RuntimeError):
        pass

    # Fall back to GPU detection
    return get_gpu_arch()


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
    """FP8 E4M3 tensor cores (Hopper sm_90+). Keyed to the real device CC."""
    return (get_device_sm() or 0) >= 90


def supports_async_copy() -> bool:
    """cp.async (NVIDIA Ampere sm_80+)."""
    return (get_device_sm() or 0) >= 80


def supports_tma() -> bool:
    """TMA bulk copy (Hopper sm_90+). Keyed to the real device CC."""
    return (get_device_sm() or 0) >= 90


def supports_block_clusters() -> bool:
    """Thread block clusters (Hopper sm_90+). Keyed to the real device CC."""
    return (get_device_sm() or 0) >= 90


def supports_matrix_cores() -> bool:
    """Matrix cores: AMD MFMA on gfx942, or NVIDIA Tensor Cores on sm_90."""
    return True


# ----------------------------------------------------------------------
# Human-readable labels
# ----------------------------------------------------------------------

def get_arch_label() -> str:
    """Human-readable label for the detected arch."""
    try:
        arch = detect_arch()
    except UnsupportedArchError as exc:
        return f"unsupported ({exc})"
    return {
        90:       "Hopper (sm_90) — H100, H200",
        942:      "CDNA3 (gfx942) — MI300X / MI300A",
        "tpu_v5p": "TPU v5p",
    }[arch]


# ----------------------------------------------------------------------
# Convenience for the optimizer code: assert the active arch is one we
# have a binding for, and surface a clear error if not.
# ----------------------------------------------------------------------

def assert_supported_arch() -> Union[int, str]:
    """Returns the detected arch or raises UnsupportedArchError."""
    return detect_arch()


# ----------------------------------------------------------------------
# C++ extension loader (consolidated from _ops_loader.py).
# Loads the per-arch specialized C++ extension. Raises on first kernel
# attribute access if the extension isn't built — there is no Python
# fallback path, but `import grokking_optimizers` succeeds either way so
# that profiling / tooling code can introspect the package without a
# working build.
# ----------------------------------------------------------------------


class _LazyOps:
    """Lazy proxy for the compiled `grokking_optimizers._ops` extension.

    Resolves on first attribute access. `hasattr(_ops, "foo")` correctly
    returns False when the extension isn't built; direct attribute access
    raises AttributeError with a descriptive build hint. This preserves the
    no-fallback contract (kernels are unreachable without a build) while
    allowing import-time introspection.
    """
    __slots__ = ("_real", "_error", "_attr_cache")

    def __init__(self):
        object.__setattr__(self, "_real", None)
        object.__setattr__(self, "_error", None)
        # Memoize resolved kernel attributes so repeated `_ops.fn` access on
        # the hot path skips the module getattr after the first lookup.
        object.__setattr__(self, "_attr_cache", {})

    def _resolve(self):
        if self._real is not None:
            return self._real
        if self._error is not None:
            return None  # cached failure
        try:
            import importlib
            real = importlib.import_module("grokking_optimizers._ops")
            object.__setattr__(self, "_real", real)
            return real
        except ImportError as e:
            object.__setattr__(self, "_error", e)
            return None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        cache = self._attr_cache
        cached = cache.get(name)
        if cached is not None:
            return cached
        real = self._resolve()
        if real is None:
            raise AttributeError(
                f"grokking_optimizers._ops.{name}: C++ extension not built. "
                f"Run `pip install -e .` (supported arches: {SUPPORTED_ARCHES}). "
                f"Original ImportError: {self._error}"
            )
        attr = getattr(real, name)
        cache[name] = attr
        return attr

    def bind(self, name):
        """Resolve ``name`` once and return the underlying kernel callable.

        Callers (optimizers) cache the returned bound function on the instance
        so subsequent steps skip the proxy ``__getattr__`` entirely. Raises the
        same descriptive error as attribute access if the extension is unbuilt.
        """
        return self.__getattr__(name)

    def __bool__(self):
        return self._resolve() is not None

    def __repr__(self):
        real = self._resolve()
        if real is None:
            return f"<_LazyOps unbuilt: {self._error}>"
        return f"<_LazyOps wrapping {real!r}>"


_cached_ops = _LazyOps()


def get_ops():
    """Return the lazy `_ops` proxy. Never raises at call time."""
    return _cached_ops


# ----------------------------------------------------------------------
# Fused (model, optimizer, arch) kernel registry.
#
# This registry is intentionally EMPTY: the live fused execution path is the
# megakernel engine (``grokking_optimizers.megakernel_engine``) keyed on the
# canonical names in ``megakernel.MODELS`` / ``megakernel.OPTIMIZERS``. Nothing
# in the package populates ``_FUSED_REGISTRY``.
#
# The registry + ``register_fused`` / ``has_fused`` / ``dispatch_fused`` are
# retained only because ``grokking_race_v2.py`` imports ``has_fused`` /
# ``dispatch_fused`` as an optional fast-path probe: ``has_fused(...)`` returns
# False (empty registry) so the race always falls back to the eager path. They
# are kept as a stable no-op shim rather than deleted to avoid breaking that
# importer. ``MODELS`` / ``OPTIMIZERS`` here are validation lists of the
# canonical model/optimizer identifiers; ``MODELS`` mirrors the canonical
# ``megakernel.MODELS`` triple.
# ----------------------------------------------------------------------

MODELS = ("transformer_decoder", "vit", "mamba3")
OPTIMIZERS = ("grokadamw", "grokfast", "lion", "looksam", "moe_adam", "muon",
              "neuralgrok", "prodigy", "supergrok2", "supergrok15", "supergrok11")

_FUSED_REGISTRY = {}


def register_fused(model, optimizer, arch):
    """Register a fused (model, optimizer, arch) kernel. Currently unused — the
    live path is the megakernel engine; kept as a stable shim (see module note).
    """
    def decorator(fn):
        _FUSED_REGISTRY[(model, optimizer, arch)] = fn
        return fn
    return decorator


def has_fused(model, optimizer, arch=None):
    """Whether a fused kernel is registered for (model, optimizer, arch).

    Always False with the (empty) built-in registry — callers fall back to the
    eager path. See the module note above.
    """
    if arch is None:
        arch = detect_arch()
    return (model, optimizer, arch) in _FUSED_REGISTRY


def dispatch_fused(model, optimizer, params, inputs, grads, state, lr, arch=None):
    """Dispatch to a registered fused kernel, or raise KeyError if none.

    Unused by the package's live path (kept as a shim for grokking_race_v2.py).
    """
    if arch is None:
        arch = detect_arch()
    key = (model, optimizer, arch)
    if key not in _FUSED_REGISTRY:
        raise KeyError(
            f"No fused kernel for {key}. "
            f"Available: {list(_FUSED_REGISTRY.keys())}"
        )
    return _FUSED_REGISTRY[key](params, inputs, grads, state, lr)
