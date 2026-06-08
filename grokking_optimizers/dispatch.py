"""Runtime hardware detection for the fused kernel architecture.

Detection is **arch-honest**: ``detect_arch()`` / ``get_gpu_arch()`` report the
REAL device architecture, not a collapsed vendor selector. An A100 reports
``sm_80``, an L40S ``sm_89``, an H100 ``sm_90a`` — they are NOT all flattened to
``90``. AMD reports its real ``gfx<...>`` target; TPU reports ``tpu_v6e``.

    NVIDIA: sm_70 / sm_75 / sm_80 / sm_86 / sm_89 / sm_90a / sm_100a / ...
    AMD:    gfx906 / gfx908 / gfx90a / gfx942 / gfx950 / gfx110x / ...
    TPU:    tpu_v6e

The C++ extension is still built as a multi-arch fat binary (one impl per
vendor family; see setup.py), so the *fat-binary impl selector* is a separate
concern from the reported arch. ``normalize_arch()`` maps a real arch to its
impl-family selector (NVIDIA → 90, AMD → 942, TPU → "tpu_v6e") for the code
paths that load the matching per-SM/-gfx code from the fat binary. Reporting
stays honest; only the binary loader normalises.

Feature predicates (``supports_fp8`` etc.) are keyed to the REAL compute
capability via ``get_device_sm()``. ``FORCE_ARCH`` accepts any real arch form
(``sm_80``, ``gfx942``, ``tpu_v6e``, or a bare numeric CC like ``80``/``90``).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Union

import torch


# ----------------------------------------------------------------------
# Structured logging (spec §6b). A single module logger, env-controlled
# level via GROK_LOG_LEVEL (e.g. DEBUG/INFO/WARNING; default WARNING).
# Replaces ad-hoc print()s and is the channel for arch-detection / ops-
# resolution diagnostics. We attach a NullHandler so importing the package
# never configures the root logger (library-friendly); the level is only
# applied when the env var is set, so default behaviour is quiet.
# ----------------------------------------------------------------------
logger = logging.getLogger("grokking_optimizers.dispatch")
logger.addHandler(logging.NullHandler())


def _configure_logging_from_env() -> None:
    level_name = os.environ.get("GROK_LOG_LEVEL")
    if not level_name:
        return
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        # Only add a stream handler if the user opted in via the env var and
        # no real handler exists yet (don't duplicate on re-import).
        if not any(not isinstance(h, logging.NullHandler)
                   for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(name)s %(levelname)s] %(message)s"))
            logger.addHandler(handler)


_configure_logging_from_env()


# ----------------------------------------------------------------------
# Model name normalization — single point of truth for short → canonical.
#
# Short names ("mamba", "decoder", "vit") are the user-facing API exposed
# by profile.MODELS and the CLI. Canonical names ("mamba3",
# "transformer_decoder", "vit") are the identifiers used by the
# megakernel engine, fused cells, and C++ dispatch. canonicalize_model()
# is the ONLY place this mapping is defined.
# ----------------------------------------------------------------------

_MODEL_CANONICAL = {
    "mamba": "mamba3",
    "decoder": "transformer_decoder",
    "vit": "vit",
    # Canonical names map to themselves (idempotent).
    "mamba3": "mamba3",
    "transformer_decoder": "transformer_decoder",
}

_MODEL_SHORT = {v: k for k, v in {
    "mamba": "mamba3",
    "decoder": "transformer_decoder",
    "vit": "vit",
}.items()}


def canonicalize_model(name: str) -> str:
    """Map a user-facing model name to the canonical internal name.

    Short names ("mamba", "decoder", "vit") are the user API (profile.MODELS);
    canonical names ("mamba3", "transformer_decoder", "vit") are the megakernel/
    fused-cell identifiers. This function accepts either form and returns the
    canonical form. Raises ValueError for unknown names.
    """
    canon = _MODEL_CANONICAL.get(name)
    if canon is None:
        raise ValueError(
            f"Unknown model name {name!r}. "
            f"Accepted: {sorted(_MODEL_CANONICAL.keys())}"
        )
    return canon


def short_model_name(canonical: str) -> str:
    """Reverse of canonicalize_model: canonical → short user-facing name."""
    short = _MODEL_SHORT.get(canonical, canonical)
    return short


# Every real arch the package recognises (matches the canonical keys in
# compile.ARCH_TABLE). Reporting is honest against this set; the fat-binary
# impl selector (90/942) is a separate, normalised value (see normalize_arch).
SUPPORTED_ARCHES = (
    # NVIDIA — real compute capabilities (the ``a`` suffix is the canonical
    # spelling for the arch-conditional variants, matching compile.ARCH_TABLE).
    "sm_70", "sm_75", "sm_80", "sm_86", "sm_89",
    "sm_90a", "sm_100a", "sm_103a", "sm_120a",
    # AMD — real gfx targets.
    "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1030", "gfx1100", "gfx1101", "gfx1102",
    "gfx1151", "gfx1200", "gfx1201",
    # TPU.
    "tpu_v6e",
)

# The two GPU fat-binary impl families the extension actually ships. Real
# arches normalise onto one of these (plus "tpu_v6e") via normalize_arch().
IMPL_FAMILIES = (90, 942, "tpu_v6e")


class UnsupportedArchError(RuntimeError):
    """Raised when the detected arch is not one of {sm_90, gfx942, tpu_v6e}."""


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

# Real NVIDIA compute capabilities the package recognises, in ascending
# order, mapped to their canonical arch-label spelling. A device CC is
# resolved to the highest label it is >= to (so a future sm_91 still resolves
# to the sm_90a kernel family, never silently dropping to fp32).
_NVIDIA_CC_LABELS = (
    (70, "sm_70"), (75, "sm_75"), (80, "sm_80"), (86, "sm_86"),
    (89, "sm_89"), (90, "sm_90a"), (100, "sm_100a"), (103, "sm_103a"),
    (120, "sm_120a"),
)


def _sm_to_arch_label(cc: int) -> str:
    """Map a real NVIDIA compute capability int to its canonical arch label.

    Honest: an A100 (CC 80) → ``sm_80``, an L40S (CC 89) → ``sm_89``, an H100
    (CC 90) → ``sm_90a``. A CC above the highest known entry resolves to that
    highest label (forward-compatible), a CC below the lowest raises.
    """
    label = None
    for threshold, name in _NVIDIA_CC_LABELS:
        if cc >= threshold:
            label = name
    if label is None:
        raise UnsupportedArchError(
            f"NVIDIA compute capability {cc} is below the minimum supported "
            f"(sm_70).")
    return label


def _normalize_force_arch(force: str) -> Union[int, str]:
    """Map a FORCE_ARCH string to the REAL arch it names (honest).

    Returns a real arch label (``sm_80``/``gfx942``/...) or ``tpu_v6e``. A bare
    numeric value is treated as an NVIDIA compute capability and resolved to its
    real label. Raises on unrecognised input. Use :func:`normalize_arch` on the
    result when an impl-family selector (90/942) is needed.
    """
    if force.startswith('tpu'):
        return "tpu_v6e"
    # AMD: any gfx target. Honor the exact gfx label; the bare ``942`` form maps
    # to gfx942 for back-compat.
    if force.startswith('gfx'):
        return force
    if force == '942':
        return "gfx942"
    # NVIDIA: sm_* label (normalise the bare ``sm_90`` alias to ``sm_90a`` etc.)
    if force.startswith('sm_'):
        try:
            cc = int(force[3:].rstrip('a'))
        except ValueError:
            raise UnsupportedArchError(
                f"FORCE_ARCH={force!r} not recognized.") from None
        return _sm_to_arch_label(cc)
    if force.startswith('sm'):
        try:
            cc = int(force[2:].rstrip('a'))
        except ValueError:
            raise UnsupportedArchError(
                f"FORCE_ARCH={force!r} not recognized.") from None
        return _sm_to_arch_label(cc)
    # Bare numeric compute capability (e.g. ``80`` for A100, ``90`` for H100).
    if force.isdigit():
        return _sm_to_arch_label(int(force))
    raise UnsupportedArchError(
        f"FORCE_ARCH={force!r} not recognized. Use an NVIDIA arch "
        f"(sm_70..sm_120a or a numeric CC), an AMD arch (gfx906..gfx1201), "
        f"or tpu_v6e.")


def normalize_arch(arch: Union[int, str]) -> Union[int, str]:
    """Map a real arch to its fat-binary impl-family selector.

    NVIDIA labels (``sm_*``) and bare CC ints → ``90``; AMD ``gfx*`` (and the
    bare ``942``) → ``942``; ``tpu_v6e`` → ``"tpu_v6e"``. This is the ONLY place
    the NVIDIA→90 / AMD→942 collapse happens, and it is used solely by the code
    that loads the matching per-arch code from the multi-arch fat binary — never
    for *reporting* the device's real arch.
    """
    if arch == "tpu_v6e":
        return "tpu_v6e"
    if isinstance(arch, int):
        return 942 if arch == 942 else 90
    if arch.startswith('gfx'):
        return 942
    if arch.startswith('sm'):
        return 90
    raise UnsupportedArchError(f"Cannot normalize unknown arch {arch!r}")


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> Union[int, str]:
    """Detected REAL GPU arch label: ``sm_80`` / ``sm_90a`` / ``gfx942`` / ...

    Reports the device's actual compute capability (NVIDIA) or gfx target (AMD)
    — NOT a collapsed vendor selector. Use :func:`normalize_arch` on the result
    when the fat-binary impl-family int (90/942) is needed.

    Honors FORCE_ARCH. Raises ``UnsupportedArchError`` if no GPU is available
    or the vendor is unknown.
    """
    force = os.environ.get('FORCE_ARCH')
    if force:
        sel = _normalize_force_arch(force)
        if sel == "tpu_v6e":
            raise UnsupportedArchError(
                "FORCE_ARCH=tpu_v6e is not a GPU arch; use detect_arch() instead")
        return sel

    if not torch.cuda.is_available():
        raise UnsupportedArchError(
            "No CUDA/HIP device available. Set FORCE_ARCH (e.g. sm_80 for an "
            "A100, sm_90a for an H100, gfx942 for MI300X) for cross-arch "
            "testing.")

    vendor = get_gpu_vendor()
    if vendor == 'nvidia':
        return _sm_to_arch_label(get_device_sm())  # real CC -> real label
    if vendor == 'amd':
        return _amd_gfx_label()                     # real gfx target
    raise UnsupportedArchError(f"Unknown GPU vendor {vendor!r}")


@functools.lru_cache(maxsize=1)
def _amd_gfx_label() -> str:
    """Real AMD gfx target string (e.g. ``gfx942``), or ``gfx942`` if unknown.

    Reads the device's gcnArchName via torch and strips any ``:xnack`` feature
    suffix. Falls back to ``gfx942`` (the shipped CDNA3 impl) only when the
    runtime exposes no arch name.
    """
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        name = getattr(props, "gcnArchName", None) or ""
    except Exception:
        name = ""
    name = name.split(":", 1)[0].strip()
    if name.startswith("gfx"):
        return name
    return "gfx942"


@functools.lru_cache(maxsize=1)
def get_device_sm():
    """Real NVIDIA compute capability (e.g. 70, 80, 90, 120), or None.

    None when there is no NVIDIA device (CPU host or AMD). Used by the feature
    predicates that want the raw integer CC; ``get_gpu_arch()`` returns the
    same capability as a canonical arch *label*.
    """
    if get_gpu_vendor() != 'nvidia':
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@functools.lru_cache(maxsize=1)
def detect_arch() -> Union[int, str]:
    """Detect the active REAL arch: a label like ``sm_80`` / ``sm_90a`` /
    ``gfx942``, or ``"tpu_v6e"``.

    Detection order:
      1. FORCE_ARCH env var (accepts sm_*, gfx*, a numeric CC, or tpu_v6e)
      2. TPU detection via JAX
      3. GPU detection via get_gpu_arch()

    The result is the device's honest arch — an A100 reports ``sm_80``, never a
    collapsed ``90``. Apply :func:`normalize_arch` for the fat-binary impl
    selector. Raises ``UnsupportedArchError`` if no supported arch is found.
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
            return "tpu_v6e"
    except (ImportError, RuntimeError):
        pass

    # Fall back to GPU detection
    arch = get_gpu_arch()
    logger.debug("detect_arch resolved real arch=%r (impl-family=%r)",
                 arch, normalize_arch(arch))
    return arch


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

_ARCH_LABELS = {
    "sm_70":   "Volta (sm_70) — V100",
    "sm_75":   "Turing (sm_75) — T4 / RTX 20xx",
    "sm_80":   "Ampere (sm_80) — A100",
    "sm_86":   "Ampere (sm_86) — A10 / RTX 30xx",
    "sm_89":   "Ada (sm_89) — L40S / RTX 40xx",
    "sm_90a":  "Hopper (sm_90a) — H100, H200",
    "sm_100a": "Blackwell (sm_100a) — B100 / B200",
    "sm_103a": "Blackwell (sm_103a)",
    "sm_120a": "Blackwell (sm_120a) — RTX 50xx",
    "gfx942":  "CDNA3 (gfx942) — MI300X / MI300A",
    "gfx950":  "CDNA3.5 (gfx950) — MI325X",
    "tpu_v6e": "TPU v6e",
}


def get_arch_label() -> str:
    """Human-readable label for the detected (real) arch."""
    try:
        arch = detect_arch()
    except UnsupportedArchError as exc:
        return f"unsupported ({exc})"
    label = _ARCH_LABELS.get(arch)
    if label is not None:
        return label
    # Any recognised-but-unlabelled real arch (e.g. an older gfx) still gets an
    # honest string rather than a KeyError.
    return str(arch)


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
            logger.debug("grokking_optimizers._ops not importable "
                         "(extension unbuilt?): %s", e)
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


# Kernel entry points whose presence indicates a usable GPU build. We probe a
# representative spread (the two SG2 fused steps + the SG15 step) rather than
# one name, so a partial/renamed build is still reported honestly.
_KERNEL_PROBE_NAMES = (
    "supergrok2_prepare_and_batched_step",
    "supergrok2_batched_step",
    "supergrok2_mamba_peer_batched_step",
    "sg15_fused_step",
)


def has_kernels() -> bool:
    """True iff the compiled C++/CUDA/HIP kernel extension is importable AND
    exposes the fused optimizer kernels.

    This is the honest capability flag (spec §6a): the optimizers have NO
    pure-PyTorch fallback for ``.step()`` — they hard-require this extension and
    a GPU. Gate any GPU-only code on ``grokking_optimizers.has_kernels()``
    instead of assuming a fallback exists. Returns False on a CPU-only / unbuilt
    install (import still succeeds for introspection/tooling).
    """
    ops = _cached_ops
    if not ops:
        return False
    return any(hasattr(ops, name) for name in _KERNEL_PROBE_NAMES)


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
SHORT_MODELS = tuple(short_model_name(m) for m in MODELS)
OPTIMIZERS = ("adamw", "grokadamw", "grokfast", "lion", "looksam", "muon",
              "neuralgrok", "prodigy", "supergrok11", "supergrok15", "supergrok2")
OPT_CLASS = {
    "adamw":       "AdamW",
    "grokadamw":   "GrokAdamW",
    "grokfast":    "Grokfast",
    "lion":        "Lion",
    "looksam":     "LookSAM",
    "muon":        "Muon",
    "neuralgrok":  "NeuralGrok",
    "prodigy":     "Prodigy",
    "supergrok11": "SuperGrok11",
    "supergrok15": "SuperGrok15",
    "supergrok2":  "SuperGrok2",
}

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
    model = canonicalize_model(model)
    return (model, optimizer, arch) in _FUSED_REGISTRY


def dispatch_fused(model, optimizer, params, inputs, grads, state, lr, arch=None):
    """Dispatch to a registered fused kernel, or raise KeyError if none.

    Unused by the package's live path (kept as a shim for grokking_race_v2.py).
    """
    if arch is None:
        arch = detect_arch()
    model = canonicalize_model(model)
    key = (model, optimizer, arch)
    if key not in _FUSED_REGISTRY:
        raise KeyError(
            f"No fused kernel for {key}. "
            f"Available: {list(_FUSED_REGISTRY.keys())}"
        )
    return _FUSED_REGISTRY[key](params, inputs, grads, state, lr)
