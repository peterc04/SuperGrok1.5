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
import sys
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
# Fused (model, optimizer, arch) kernel registry + L1 megakernel readiness.
#
# THE LIVE FUSED PATH (lever (a)). ``register_fused`` populates ``_FUSED_REGISTRY``
# for the cells on the READINESS whitelist (``_FUSED_READY``). For a whitelisted
# (model, optimizer) cell, ``has_fused(...)`` returns True and ``dispatch_fused``
# runs the L1 fused optimizer-tail megakernel via ``ops.fused_step`` —
# numerically the canonical optimizer update over the REAL gradient the framework
# already computed (the C++ ``opt_only=True`` path; dispatch.cpp/opt_components).
#
# L1 vs L3 (the tier landscape after PHASE 1):
#   * L1 (this whitelist, {adamw,lion}×3 models): the fused optimizer TAIL only —
#     consumes the real ``p.grad`` the framework computed and applies the
#     canonical update. Faithful + validatable; the cell stays model-agnostic.
#   * L3-SURROGATE (the 33 generated cells' opt_only=False path): an element-local
#     SURROGATE model (csrc/fused/sm_90/model_stages.cuh: acts=GELU(param+input),
#     grad=acts*GELU'(param)) over the flat param blob — NOT the real graph; its
#     loss cannot match eager, so it is NOT used by the race. Still compiled
#     (perf-placement coverage) but unreachable on the race path.
#   * L3-REAL (PHASE 1; ONLY (transformer_decoder, adamw) on sm_90): the TRUE
#     fused megakernel — ONE persistent kernel runs the REAL decoder fwd+bwd +
#     AdamW, no surrogate, no intermediate launches (model_stages_decoder.cuh +
#     fused_decoder_megakernel.cuh + mega_decoder_real_adamw.cu, transcribed from
#     the verified oracle). Its loss DOES match eager (validated to 1e-5).
#     Reached via ``fused_train_step`` / ``has_l3_real`` (the L3-REAL tier marker
#     ``_FUSED_L3_REAL``), which REPLACES the eager fwd+bwd+opt for that one cell.
# So "L3 can't match eager" is true for the SURROGATE L3 cells but FALSE for the
# L3-REAL decoder×adamw cell. See BUILD_AND_VALIDATE.md §PHASE-1.
#
# WHY THIS WHITELIST (adamw, lion). The L1 tail's only non-pointer inputs are the
# scalars (lr/betas/eps/wd/bc1/bc2) and the persistent m|v state — all directly
# computable from the live optimizer + a step counter, with NO separate per-step
# precompute. The other 9 optimizers need a precomputed per-step quantity the L1
# tail reads but does not itself produce — prodigy's adaptive ``d``, grokfast/
# grokadamw's slow-grad EMA seeding, looksam's SAM direction, muon's NS-orth
# direction, SG11/15's reduced gate + mu, SG2's meta-net smart-grad. Those are
# honest-staged behind the readiness gate (a loud one-time TODO at run start)
# rather than wired with placeholder scalars (which would silently degrade the
# math — the exact suppression the owner forbids). ``register_fused`` is the seam
# to add them once their precompute is plumbed + validated.
#
# NOTE (L1 is model-agnostic): the L1 tail (fused_optimizer_stage<Opt>) never
# touches the model stages, so a cell's result depends only on the optimizer, not
# the model. All 3 models × {adamw, lion} are therefore equivalent on L1 and all
# are whitelisted; the parity test needs only ONE model per optimizer.
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

# READINESS whitelist: (canonical_model, optimizer) cells cleared for the L1
# megakernel path this pass. Everything else uses the existing per-op/eager path.
# Kept as the SINGLE source of truth for "which cell may take the megakernel"; the
# validation script (tests/hw/test_megakernel_vs_eager.py) asserts each of these
# matches its eager reference before it is trusted on hardware.
_FUSED_READY_OPTIMIZERS = ("adamw", "lion")
_FUSED_READY = frozenset(
    (m, o) for m in MODELS for o in _FUSED_READY_OPTIMIZERS
)

# PHASE 1+2 — L3-REAL tier marker. These (canonical_model, optimizer) cells have a
# TRUE L3 fused megakernel: ONE persistent kernel runs the REAL model fwd+bwd +
# AdamW (no surrogate, no intermediate launches), transcribed from the verified
# per-model oracle:
#   * (transformer_decoder, adamw)  — PHASE 1 (model_stages_decoder.cuh +
#                                     fused_decoder_megakernel.cuh)
#   * (vit, adamw)                  — PHASE 2 (model_stage_vit.cuh +
#                                     fused_vit_megakernel.cuh; dynamic smem)
#   * (mamba3, adamw)               — PHASE 2 (model_stage_mamba3.cuh +
#                                     fused_mamba_megakernel.cuh; dynamic smem)
# NOTE the CANONICAL name "mamba3" (canonicalize_model maps the user "mamba" →
# "mamba3"; the C++ dispatch + wired-cell table also use "mamba3"). These are the
# ONLY L3-REAL cells — every other cell's L3 path is still the element-local
# SURROGATE (loud honesty: do NOT use opt_only=False for them). The race reaches
# these cells via fused_train_step() (NOT the L1 per-tensor fused_optimizer_step).
# Kept as a SET so the tier semantics are a single source of truth the race + the
# parity tests both read.
# OWNER BASELINE DIRECTIVE (all 33 cells on L3-TC): the decoder/vit wgmma fwd+bwd is
# optimizer-independent, so the SINGLE-LAUNCH optimizer tails (lion/grokfast/grokadamw
# /neuralgrok) compose into the SAME TC driver via apply_optimizer<Opt> (the 14/0-
# apply-parity math). They are L3-REAL on the wgmma path. Added per-model as each
# model's launcher gains the opt_id switch (decoder first; vit/mamba follow). The
# STAGED (prodigy/muon/SG11/SG15), model-coupled (looksam) and SG2 cells are NOT here:
# their L3-TC needs a precompute stage / 2nd backward / sharpness ABI not in the
# single-launch path (INTEGRATION-OPTSTAGES verdict table) — wiring_check fails them
# loud with the cited blocker, which is the directive's "report converted/blocked".
_FUSED_L3_REAL = frozenset({
    ("transformer_decoder", "adamw"),
    ("vit", "adamw"),
    ("mamba3", "adamw"),
    # decoder + vit single-launch TC tails (opt_id switch wired in the
    # mega_{decoder,vit}_real_adamw_tc_launcher.cu + dispatch.cpp).
    ("transformer_decoder", "lion"),
    ("vit", "lion"),
    # grokfast (cycle 2): CONVERTED. The ema cold-start blocker is fixed in
    # apply_optimizer<Grokfast> (opt_components.cuh) — at step==1 it seeds ema=grad,
    # so grokfast_fused_step's e_new = alpha*g + (1-alpha)*g = g matches the eager
    # ema=grad0 seed (grokfast.py _group_cache). The state-aware tail gate now passes
    # params AND state (ema rel < 1e-4) for {decoder,vit}×grokfast; A/A/A clean.
    # grokfast's constructor has NO grad_clip, and _opt_scalars_from already forwards
    # grokfast_alpha/grokfast_lamb, so the cold-start was the ONLY remaining gap.
    ("transformer_decoder", "grokfast"),
    ("vit", "grokfast"),
    # mamba (cycle-2 directive (c)): the mamba TC kernel is now wired in-_ops
    # (mega_mamba_real_adamw_tc_launcher.cu + dispatch.cpp wgmma branch). mamba×adamw
    # was already L3-REAL (scalar engine); it now ALSO has the wgmma path. lion +
    # grokfast join via the OptId-generic launcher (opt_id 1/2). The 0.46× scalar-wins
    # is a perf fact the roofline reports, not a correctness block. grokadamw/neuralgrok
    # are NOT mamba TC tails (grokadamw's 3-mechanism gap; neuralgrok host-coupled).
    ("mamba3", "lion"),
    ("mamba3", "grokfast"),
    # neuralgrok (decoder + vit): CONVERTED. The amplifier psi-net MLP is already
    # in the TC driver (apply_optimizer<NeuralGrok>, opt_components.cuh:238-251),
    # and the decoder/vit TC launchers bind st.psi_W1/b1/W2 from the `extra` state
    # slice. The missing seam was HOST-SIDE: (1) the psi-net pack must be written
    # into extra[0..3H+1] (fused_train_step now scatters NeuralGrok.psi_pack() in
    # every step), and (2) the amplifier is trained host-side on a cadence
    # (NeuralGrok.maybe_train_amplifier; the kernel cannot run the lookahead
    # autograd). alpha+beta are forwarded by _opt_scalars_from so the kernel's
    # (psi*alpha+beta)*g uses the live neural_alpha/neural_beta. State-gate clean:
    # the kernel and the real eager NeuralGrok consume the SAME TC-reduced grad and
    # the SAME psi weights, so m/v match to fp32 reorder tol (no per-element `extra`
    # ema for neuralgrok — extra carries the psi PACK, which the kernel does not
    # write, so it survives the step). neuralgrok's grad_clip is a GLOBAL grad-norm
    # clip in the eager BINDING (helpers.h clip_grad_norms_device_side), NOT in the
    # neuralgrok.h algorithm the kernel implements; the L3-TC gate asserts the clip
    # is INERT (global grad-norm <= grad_clip) at the gated step so the parity is
    # real, not a hollow pass. mamba×neuralgrok stays OUT (no mamba TC neuralgrok
    # tail — the mamba launcher's opt_id switch does not include NeuralGrok).
    ("transformer_decoder", "neuralgrok"),
    ("vit", "neuralgrok"),
    # grokadamw (decoder): CONVERTED. The THREE eager mechanisms ALL land now (the
    # ema cold-start was already staged in apply_optimizer<GrokAdamW>); the cell is
    # no longer a hollow pass:
    #   (i)  PER-TENSOR LAYER-WISE β1 = β1·(1-γ)^layer (γ via FusedScalars.gamma).
    #        Applied in the kernel's P3 work-steal loop where the task id t == the
    #        flat named_parameters() layer index (kDecOffsets order == eager
    #        enumeration), with bc1 ALSO rebased (1-β1_i^step) so m_hat=m/bc1 matches
    #        eager (β1-only re-fails 1a ~9.6× on the deep layer). bc2 stays global.
    #        This is the mechanism that failed the STEP-1 state gate (m-rel 0.895);
    #        it now passes.
    #   (ii) GLOBAL grad-norm clip to grad_clip (FusedScalars.grad_clip). The eager
    #        clip_grad_norms_device_side (helpers.h) is a GLOBAL norm over ALL
    #        tensors, one clip_coef = grad_clip/(‖g‖₂+1e-6) when ‖g‖₂>grad_clip. The
    #        kernel computes it ON-DEVICE in a P2.5 deterministic ascending-CTA
    #        reduction over the reduced grad → clip_coef, applied per-element in the
    #        tail. grad_out is NOT mutated (the return_grad oracle + the eager-side
    #        clip both see the unclipped grad). Inert at step 1 (‖g‖₂≈0.72<1) but
    #        FIRES by ~step 50 — so the single-step gate is BLIND to it; honest
    #        registration rests on a MULTI-STEP parity (kernel-with-clip tracks eager
    #        to fp32-reorder tol; kernel-without-clip reproduces the ~2e-4 divergence).
    #   (iii) adaptive α = α·exp(-κ·signal): a genuine NO-OP in-context. No race/gate
    #        path feeds (train_loss, val_loss) to GrokAdamW.step(), so eager α stays
    #        at α_init = the kernel's static α for ALL steps. Faithful, not dropped.
    ("transformer_decoder", "grokadamw"),
})

_FUSED_REGISTRY = {}


def fused_ready_cells():
    """The (canonical_model, optimizer) pairs cleared for the L1 megakernel path.

    Single source of truth for the readiness whitelist; the race and the
    validation script both read it so they cannot disagree on which cell is live.
    """
    return frozenset(_FUSED_READY)


def fused_l3_real_cells():
    """The (canonical_model, optimizer) pairs with a TRUE L3 fused megakernel
    (real fwd+bwd+opt in one persistent kernel). Currently only
    (transformer_decoder, adamw). Single source of truth for the L3-REAL tier."""
    return frozenset(_FUSED_L3_REAL)


def has_l3_real(model, optimizer, arch=None) -> bool:
    """Whether the TRUE L3 fused megakernel (real fwd+bwd+opt) is available for
    (model, optimizer) on the detected arch. True ONLY for the L3-REAL cell(s) on
    sm_90 with a compiled fused TU. The decoder real path is sm_90-only (the
    nvcc-compiled mega_decoder_real_adamw.cu); gfx942 keeps the eager path."""
    if arch is None:
        try:
            arch = detect_arch()
        except UnsupportedArchError:
            return False
    try:
        model = canonicalize_model(model)
    except ValueError:
        return False
    impl = normalize_arch(arch) if not isinstance(arch, int) else arch
    if impl != 90:   # the real decoder fwd/bwd .cu is sm_90-only for PHASE 1
        return False
    return (model, optimizer) in _FUSED_L3_REAL


def register_fused(model, optimizer, arch):
    """Register a fused (model, optimizer, arch) kernel callable.

    Used to populate ``_FUSED_REGISTRY`` for the readiness-whitelisted cells (see
    ``_register_ready_cells`` below). The registered callable takes the same
    ``(params, inputs, grads, state, lr)`` shape ``dispatch_fused`` forwards.
    """
    def decorator(fn):
        _FUSED_REGISTRY[(canonicalize_model(model), optimizer, arch)] = fn
        return fn
    return decorator


def has_fused(model, optimizer, arch=None):
    """Whether the L1 megakernel path is available for (model, optimizer, arch).

    True iff the cell is on the readiness whitelist (``_FUSED_READY``) for the
    detected arch. The arch must be a real GPU arch with a compiled fused TU
    (sm_90 / gfx942); on any other arch (CPU host, TPU) this returns False so the
    caller uses the eager path. Tolerant of an unknown model name (returns False).
    """
    if arch is None:
        try:
            arch = detect_arch()
        except UnsupportedArchError:
            return False
    try:
        model = canonicalize_model(model)
    except ValueError:
        return False
    # The compiled fused TU exists only for the GPU impl families (sm_90/gfx942).
    impl = normalize_arch(arch) if not isinstance(arch, int) else arch
    if impl not in (90, 942):
        return False
    return (model, optimizer) in _FUSED_READY


def dispatch_fused(model, optimizer, params, inputs, grads, state, lr, arch=None):
    """Dispatch to a registered fused kernel, or raise KeyError if none.

    The whitelisted cells register a callable here via ``register_fused``; the
    race reaches the megakernel through the higher-level
    :func:`fused_optimizer_step` (which assembles persistent state + live
    scalars), so this remains the low-level registry shim.
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


# ----------------------------------------------------------------------
# L1 megakernel optimizer-step driver (the live race path for whitelisted cells).
# ----------------------------------------------------------------------

# One-time run-start announcement guard (so the readiness summary prints ONCE per
# process, not per step — the owner directive: "says so ONCE loudly at run start").
_FUSED_ANNOUNCED = False


def announce_fused_readiness(force: bool = False) -> None:
    """Print, ONCE per process, which cells take the L1 megakernel path.

    Loud + honest at run start: lists the whitelisted (model, optimizer) cells
    that will use the fused optimizer-tail megakernel and notes that every other
    cell uses the eager/per-op path. Idempotent (guarded by ``_FUSED_ANNOUNCED``).
    """
    global _FUSED_ANNOUNCED
    if _FUSED_ANNOUNCED and not force:
        return
    _FUSED_ANNOUNCED = True
    ready = sorted(f"{short_model_name(m)}:{o}" for (m, o) in _FUSED_READY)
    l3_real = sorted(f"{short_model_name(m)}:{o}" for (m, o) in _FUSED_L3_REAL)
    msg = (
        f"[fused] L3-REAL fused-train path (real fwd+bwd+opt in ONE persistent "
        f"kernel) ENABLED for {len(l3_real)} cell(s): {', '.join(l3_real)}. "
        f"L1 fused-optimizer-tail path ENABLED for {len(ready)} cell(s): "
        f"{', '.join(ready)}. All other (model, optimizer) cells use the "
        f"eager/per-op path (the surrogate L3 cells are compiled but unused by "
        f"the race; see BUILD_AND_VALIDATE.md §PHASE-1).")
    # Print to stderr so the run-start banner is ALWAYS visible (the module
    # logger has a NullHandler by default, so logger.warning would be silent
    # unless GROK_LOG_LEVEL is set). Also log it for structured-log consumers.
    print(msg, file=sys.stderr, flush=True)
    logger.warning(msg)


# Per-optimizer extra-state requirement for the L1 tail's `extra` (3rd n-slice).
# Only the whitelisted optimizers are listed; adamw/lion need no extra slice (it
# is allocated and zeroed but never read — rebase_state<Opt> guards it out).
_OPT_USES_EXTRA = {
    "adamw": False,
    "lion":  False,
}


def _opt_scalars_from(optimizer, step):
    """Extract the FULL scalar set for ``ops.fused_step`` from a live optimizer.

    Pulls lr/betas/eps/weight_decay from ``param_groups[0]`` (so a scheduled lr is
    honored) and computes the UN-INVERTED bias corrections bc1 = 1 - beta1**step,
    bc2 = 1 - beta2**step from the shared step counter. One (bc1, bc2) pair is
    correct because the race steps every parameter together (the megakernel treats
    each flat param as one task; see opt_components.cuh FusedScalars note).

    Returns a kwargs dict for ``ops.fused_step`` (only the fields the whitelisted
    optimizers consume are set; the rest keep their inert ABI defaults).
    """
    g = optimizer.param_groups[0]
    betas = g.get("betas", (0.9, 0.999))
    beta1, beta2 = float(betas[0]), float(betas[1])
    lr = float(g.get("lr", 1e-3))
    eps = float(g.get("eps", 1e-8))
    wd = float(g.get("weight_decay", 0.0))
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    out = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd,
               bc1=bc1, bc2=bc2, step=int(step))
    # grokfast/grokadamw: the in-kernel apply_optimizer<Opt> reads st.alpha/st.lamb
    # (the EMA decay + amplification). Pull the REAL configured values from the
    # optimizer's param_groups so the kernel uses the live mechanism — NOT the
    # FusedScalars struct default (which only coincidentally matches 0.98/2.0).
    # grokfast uses grokfast_alpha/grokfast_lamb; grokadamw uses alpha/lamb.
    if "grokfast_alpha" in g:
        out["alpha"] = float(g["grokfast_alpha"])
        out["lamb"] = float(g.get("grokfast_lamb", 2.0))
    elif "alpha" in g and "beta" in g and "lamb" not in g:
        # neuralgrok: the in-kernel apply_optimizer<NeuralGrok> computes
        # g_amp = (psi*alpha + beta)*g (opt_components.cuh:250 / neuralgrok.h:70),
        # reading st.alpha AND st.beta. NeuralGrok's param_groups carry `alpha`
        # (amplification scale) and `beta` (affine psi term) but NO `lamb`, which
        # distinguishes it from grokadamw below. Forward BOTH so the kernel uses the
        # live configured values (neural_alpha=10.0, neural_beta=4.0) instead of the
        # FusedScalars defaults (0.98/0.0). `beta` lands in scalars["beta"], which
        # fused_step maps to the FusedScalars.beta field the apply reads.
        out["alpha"] = float(g["alpha"])
        out["beta"] = float(g["beta"])
    elif "alpha" in g and "lamb" in g:   # grokadamw (also lion has no alpha key)
        out["lamb"] = float(g["lamb"])
        # GrokAdamW decoder L3-TC — ALL THREE mechanisms thread here:
        #  (iii) ADAPTIVE α = α_init·exp(-κ·signal). The L3 path REPLACES the eager
        #        opt.step() (where α adaptation lives), so we must pass the LIVE α_t
        #        here or the adaptation is dropped on the fused path (suppression in
        #        the race, which feeds train/val losses on a cadence). When the
        #        optimizer exposes _alpha_for_group (a GrokAdamW), use it: it returns
        #        α_init until a (train_loss,val_loss) signal has been set (so the
        #        single-step gate + early steps are unchanged) and α_init·exp(-κ·
        #        signal) once the race/parity sets opt._grok_signal/_signal_active
        #        BEFORE the step. Falls back to the static group α otherwise.
        if hasattr(optimizer, "_alpha_for_group"):
            out["alpha"] = float(optimizer._alpha_for_group(g))
        else:
            out["alpha"] = float(g["alpha"])
        #  (i) layer-wise β1 = β1·(1-γ)^layer (kernel P3, per-tensor) and (ii) the
        #      GLOBAL grad-norm clip to grad_clip (kernel P2.5). Both are grokadamw-
        #      only param_group keys; every other optimizer lacks them, so they
        #      default to the inert FusedScalars value (0.0 ⇒ single global β1 / no
        #      clip). They make the cell a faithful conversion, not a hollow pass.
        if "gamma" in g:
            out["gamma"] = float(g["gamma"])
        if "grad_clip" in g:
            out["grad_clip"] = float(g["grad_clip"])
    return out


def fused_optimizer_step(model_name, opt_name, torch_module, optimizer, *,
                         state_cache, step):
    """Run ONE L1 fused optimizer-tail step over the live model's parameters.

    This is the live race path for a readiness-whitelisted cell. The caller has
    ALREADY run the real forward + backward, so every trainable parameter carries
    its real ``p.grad``; this applies the canonical optimizer update to each
    parameter in-place via the fused megakernel (``ops.fused_step`` with
    ``opt_only=True`` → the L1 real-grad tail).

    State ownership (critical): because the fused path REPLACES ``optimizer.step()``
    the torch optimizer's own ``.state`` never fills, so we keep our OWN persistent
    per-parameter ``[m|v|extra]`` (one contiguous 3n fp32 buffer) in
    ``state_cache`` (a dict the caller threads across steps). Reallocating it per
    step would reset momentum every step (the optimizer would never grok); we
    allocate once per parameter and reuse.

    Args:
        model_name: user/canonical model name (canonicalized internally).
        opt_name: optimizer key (must be on the readiness whitelist).
        torch_module: the ``nn.Module`` being trained (provides named params+grads).
        optimizer: the live torch optimizer (source of lr/betas/eps/wd).
        state_cache: dict persisted across steps: id(param) -> 3n fp32 state tensor.
        step: 1-based step counter (drives bias correction).

    Returns True on success. Raises if the cell is not whitelisted or a param is
    in an unsupported layout (no silent corruption).
    """
    model_c = canonicalize_model(model_name)
    if (model_c, opt_name) not in _FUSED_READY:
        raise KeyError(
            f"fused_optimizer_step: cell ({model_c}, {opt_name}) is not on the "
            f"L1 readiness whitelist {sorted(_FUSED_READY)}")
    ops = get_ops()
    if not hasattr(ops, "fused_step"):
        raise RuntimeError("ops.fused_step unavailable; build the extension.")

    scalars = _opt_scalars_from(optimizer, step)

    for _, p in torch_module.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            # No grad for this param this step (e.g. an unused head). Skip it
            # rather than feed a stale/zero buffer — the eager step would also be
            # a no-op for a None grad.
            continue
        # The L1 tail indexes raw dense float memory; a non-contiguous param/grad
        # or a non-fp32 dtype would silently address the wrong elements. fused_step
        # does NOT run check_param_grad (that guard is on the per-op path), so we
        # enforce it here. amp is off by default in the race (fp32 params).
        if p.dtype != torch.float32 or g.dtype != torch.float32:
            raise RuntimeError(
                f"fused_optimizer_step requires fp32 params AND grads (got "
                f"param {p.dtype}, grad {g.dtype}); the L1 megakernel tail reads "
                f"raw float memory. Disable AMP for the fused path.")
        if g.is_sparse:
            raise RuntimeError("fused_optimizer_step does not support sparse grads")
        param = p.data if p.data.is_contiguous() else p.data.contiguous()
        grad = g if g.is_contiguous() else g.contiguous()
        n = param.numel()
        key = id(p)
        st = state_cache.get(key)
        if st is None or st.numel() != 3 * n:
            # Persistent [m|v|extra] state, zero-init ONCE per parameter.
            st = torch.zeros(3 * n, dtype=torch.float32, device=param.device)
            state_cache[key] = st
        # input is unused on the L1 tail (auto in = input.numel()? : p); pass the
        # param as a harmless non-empty placeholder so the C++ side does not fall
        # back to the acts buffer. grad is the REAL gradient just computed.
        ops.fused_step(model_c, opt_name, param.view(-1), param.view(-1),
                       grad.view(-1), st, scalars["lr"],
                       beta1=scalars["beta1"], beta2=scalars["beta2"],
                       eps=scalars["eps"], weight_decay=scalars["weight_decay"],
                       bc1=scalars["bc1"], bc2=scalars["bc2"],
                       step=scalars["step"], opt_only=True)
        # If .contiguous() copied, write the update back into the param storage.
        if param.data_ptr() != p.data.data_ptr():
            p.data.copy_(param.view_as(p.data))
    return True


# Per-model flat-buffer element counts. Each MUST equal the C++/CUDA total the
# corresponding .cu static-asserts (kDecTotalElems / kVitTotalElems /
# kMambaTotalElems). The Python wrapper builds the flat param/state buffers to
# this size. The no-GPU layout tests cross-check these against the oracle totals.
_DECODER_TOTAL_ELEMS = 422755   # small decoder (vocab=99,d=128,heads=4,layers=2,seq=4)
_DECODER_SEQ = 4
_VIT_TOTAL_ELEMS = 418017       # small ViT (p=97,d=128,heads=4,layers=2,patch=49,npatch=16)
_VIT_PATCH_ELEMS = 16 * 49      # 784 patch pixels per sample ([16][49] row-major)
_MAMBA_TOTAL_ELEMS = 259425     # small Mamba (ntok=99,p=97,d=128,layers=2,seq=8)
_MAMBA_SEQ = 8

# L3-REAL per-(canonical_model) ABI spec for fused_train_step. Each model carries
# its (token/patch) input through the single `input` tensor with the packing
# dispatch.cpp reads:
#   * "kind"  — "int_tokens" (decoder/mamba: int32 tokens[B*seq]++targets[B]) or
#               "float_patches" (vit: float32 patches[B*per]++int-target-bits[B]
#               BIT-REINTERPRETED into the trailing float slots, NOT a value cast)
#   * "total" — flat param element count (== the .cu's static_asserted total)
#   * "per"   — per-sample input width (seq for tokens; patch-pixel count for vit)
_L3_REAL_SPEC = {
    "transformer_decoder": {"kind": "int_tokens", "total": _DECODER_TOTAL_ELEMS,
                            "per": _DECODER_SEQ},
    "mamba3":              {"kind": "int_tokens", "total": _MAMBA_TOTAL_ELEMS,
                            "per": _MAMBA_SEQ},
    "vit":                 {"kind": "float_patches", "total": _VIT_TOTAL_ELEMS,
                            "per": _VIT_PATCH_ELEMS},
}


# Per-(canonical_model, gemm_engine) READINESS for the L3-REAL train path. The
# GEMM engine is the owner directive's GEMM_IMPL axis (task 1): "wgmma" runs the
# bf16 tensor-core launcher (mega_{decoder,vit}_real_adamw_tc_launcher.cu, wired
# into _ops), "scalar" runs the shipped fp32 megakernel. Mamba has NO wgmma
# launcher (the measured scalar-wins carve-out, 905a4bb 0.46×), so requesting
# wgmma for it is unsupported and FAILS LOUD (dispatch.cpp TORCH_CHECK) — never a
# silent scalar run under a wgmma label. Single source of truth the race + tests
# read so the path map cannot drift.
_L3_WGMMA_CELLS = frozenset({
    ("transformer_decoder", "adamw"),
    ("vit", "adamw"),
    # mamba (cycle-2 directive (c)): the mamba TC kernel (launch_fused_mamba_megakernel_tc)
    # is now WIRED into _ops via mega_mamba_real_adamw_tc_launcher.cu + the dispatch.cpp
    # wgmma branch. The 0.46× scalar-wins carve-out is a PERFORMANCE fact (scan-dominated)
    # the roofline surfaces — not a correctness reason (test_mamba_tc 5/5). adamw/lion/
    # grokfast route to the TC path at bf16; grokadamw/neuralgrok are NOT mamba TC tails.
    ("mamba3", "adamw"),
    ("mamba3", "lion"),
    ("mamba3", "grokfast"),
    # OWNER BASELINE: single-launch optimizer tails on the bf16 TC decoder/vit driver.
    # Each entry's tail runs in-kernel via apply_optimizer<Opt> over the TC-reduced
    # grad; added per-model as the launcher opt_id switch lands.
    ("transformer_decoder", "lion"),
    ("vit", "lion"),
    # grokfast (cycle 2): the ema cold-start (state-aware tail blocker (a)) is fixed in
    # apply_optimizer<Grokfast> (opt_components.cuh: step==1 → ema=grad, matching the
    # eager ema=grad0 seed). State + params now match the real Grokfast at step 1
    # (state-gate clean). grokadamw stays excluded — its per-tensor layer-wise beta1
    # AND per-tensor grad-norm clip (bindings.cpp clip_grad_norms_device_side) are not
    # representable in the single global FusedScalars; see the _FUSED_L3_REAL note.
    ("transformer_decoder", "grokfast"),
    ("vit", "grokfast"),
    # neuralgrok (decoder + vit): the bf16 TC launcher dispatches opt_id=6
    # (OptId::NeuralGrok) → apply_optimizer<NeuralGrok> over the TC-reduced grad,
    # binding the psi-net from the `extra` slice the host fills via psi_pack(). The
    # fwd+bwd is the SAME validated wgmma decoder/vit kernel; only the per-element
    # tail differs. mamba×neuralgrok is NOT here (no mamba TC neuralgrok tail).
    ("transformer_decoder", "neuralgrok"),
    ("vit", "neuralgrok"),
    # grokadamw (decoder): CONVERTED. All THREE eager mechanisms now land in the
    # bf16 TC path (opt_id=3 → apply_optimizer<GrokAdamW>): (i) per-tensor
    # layer-wise β1=β1·(1-γ)^layer + rebased bc1 (kernel P3, t==flat layer index);
    # (ii) GLOBAL grad-norm clip to grad_clip (kernel P2.5 deterministic global-norm
    # reduction → clip_coef, applied to g in the tail, grad_out NOT mutated);
    # (iii) adaptive-α is a no-op in-context (no train/val losses fed to .step()),
    # so the static α is faithful. γ/grad_clip thread via FusedScalars (gamma,
    # grad_clip appended). Validated: single-step state-gate + a MULTI-STEP parity
    # (the clip fires by ~step 50; kernel tracks eager to fp32-reorder tol). vit is
    # NOT here yet (this conversion is decoder-only per the cell order).
    ("transformer_decoder", "grokadamw"),
})


def gemm_impl_for_cell(model_name, opt_name, precision):
    """The GEMM engine token ("wgmma" | "scalar") for an L3-REAL cell at `precision`.

    Path-matched semantics (replaces the old fp32-only gate, owner directive task 1):
      * precision == "bf16"  → "wgmma" for the TC-capable cells (decoder/vit/mamba ×
                               {adamw, lion, grokfast} as wired in _L3_WGMMA_CELLS),
                               "scalar" otherwise. mamba is NOW a wgmma cell (cycle-2
                               directive (c)) — the 0.46× scalar-wins is a perf fact the
                               roofline reports, not a correctness carve-out.
      * precision == "fp32"  → "scalar" for ADAMW cells (the fp32 owner-computes
                               megakernel exists only for adamw, all 3 models); None for
                               the non-adamw single-launch tails (lion/grokfast) — they
                               have NO scalar real fwd+bwd+opt kernel, only the wgmma one,
                               so at fp32 the caller declines to eager (the honest path).
                               Returning "scalar" for them would route to the SURROGATE
                               cell (wired_fused_cell) → a dtype throw on the token input,
                               a loud-but-wrong fallthrough; None avoids it cleanly.
    Any other precision returns None → the caller declines the L3 path entirely
    (fp16-AMP / tf32 have no in-kernel carrier here; eager is the honest path).

    The engine returned is the ACTUAL engine dispatch.cpp will run (it has no silent
    fallback: an unsupported wgmma request throws), so a successful fused_train_step
    with this token PROVES that engine executed — the basis of the path report.
    """
    model_c = canonicalize_model(model_name)
    if precision == "fp32":
        # Only adamw has a scalar real fwd+bwd+opt megakernel. Non-adamw L3-REAL cells
        # (lion) are wgmma-only → decline at fp32 (eager), never the surrogate.
        return "scalar" if opt_name == "adamw" else None
    if precision == "bf16":
        return "wgmma" if (model_c, opt_name) in _L3_WGMMA_CELLS else "scalar"
    return None


def fused_train_step(model_name, opt_name, torch_module, optimizer, tokens,
                     targets, *, state_cache, step, return_grad=False,
                     gemm_impl="scalar",
                     lr=None, betas=None, weight_decay=None, eps=None):
    """Run ONE TRUE L3 fused TRAIN step (real fwd+bwd+opt) for an L3-REAL cell.

    PHASE 1+2: for an L3-REAL cell ((transformer_decoder|vit|mamba3, adamw)) this
    REPLACES the eager forward + backward + optimizer.step() with ONE persistent
    megakernel that runs the REAL model forward+backward AND the AdamW update —
    real model math, real optimizer math, ZERO intermediate kernel launches (the
    owner rejected CUDA graphs; this is the chosen path). The caller MUST therefore
    NOT run its own forward/backward/step for this cell; it uses the LOSS this
    returns for logging.

    The input path is carried through the existing ``ops.fused_step`` ABI (whose
    pybind arity is pinned), packed PER MODEL into the single ``input`` tensor:
      * int_tokens (decoder/mamba): ``input`` = int32 [B*(per+1)] = tokens[B*per]
        ++ targets[B]  (per = seq: decoder 4, mamba 8).  ``tokens`` is [B, per].
      * float_patches (vit): ``input`` = float32 [B*(per+1)] = patches[B*per] ++
        targets BIT-REINTERPRETED into the trailing float slots (per = 16*49 = 784;
        ``tokens`` here is the float patch tensor [B, 16, 49]). The targets are
        carried via ``tensor.view(torch.int32)/view(torch.float32)`` bit-reinterpret
        (NOT a value cast) so they are lossless for ALL int32 — dispatch.cpp reads
        them back with reinterpret_cast<const int*>. This mirrors the decoder's
        int pack: one contiguous ``input`` tensor.
    ``params`` = the flat concat of ``named_parameters()`` (in order); ``state`` =
    [m|v|extra] (3*total) + a trailing loss slot the kernel writes the mean
    cross-entropy into. We read that slot back and return it.

    State/params ownership (critical): the megakernel updates the FLAT param
    buffer in place, so we keep BOTH the flat param buffer AND the [m|v|extra|loss]
    state in ``state_cache`` (keyed by the model id), allocated ONCE and reused —
    reallocating per step would reset momentum and the run would never grok. After
    the kernel we scatter the updated flat params back into the live
    ``p.data`` (named_parameters() order), so the model carries the new weights for
    the next eager eval.

    Args:
        model_name/opt_name: must be an L3-REAL cell ((decoder|vit|mamba, adamw)).
        torch_module: the live model nn.Module (NOT torch.compile-wrapped — the
            flat layout assumes the eager named_parameters() order).
        optimizer: live torch optimizer (source of lr/betas/eps/wd).
        tokens: the per-sample input. int64/int32 [B, kSeq] token ids in [0, vocab)
            for decoder/mamba; the float patch tensor [B, 16, 49] for vit (the arg
            keeps its name for ABI symmetry across models).
        targets: int64/int32 [B] target ids.
        state_cache: dict persisted across steps (canonical model name -> {flat,
            state, grad_out, names}).
        step: 1-based step counter (drives bias correction).
        return_grad: if True, also return the reduced weight-grad buffer (a flat
            [total] tensor; slice it by the model's layout offsets for per-tensor
            grads) — used by the parity test to compare the kernel's backward
            against the oracle. Default False (the race only needs the loss).
        gemm_impl: GEMM-engine token forwarded to ops.fused_step (owner directive
            task 1). "scalar" (default) → the shipped fp32 owner-computes
            megakernel; "wgmma" → the bf16 tensor-core launcher (decoder/vit only;
            wired into _ops via mega_{decoder,vit}_real_adamw_tc_launcher.cu). On the
            "wgmma" path the batch B is truncated to the largest multiple of 16 ≤ B
            (the TC dW K-loop is 16-step atoms; the race full batch is not ÷16), for
            both tokens and targets consistently. dispatch.cpp has NO silent
            fallback: an unsupported wgmma request (e.g. mamba) throws, so a
            successful return PROVES the requested engine executed.

    Returns the scalar training loss (mean CE) as a float; or (loss, grad) when
    ``return_grad=True``. Raises if the cell is not L3-REAL or a param layout is
    unexpected (no silent corruption).

    lr/betas/weight_decay/eps: ACCEPTED for signature-compatibility with the parity
    test helpers (test_{vit,mamba,decoder}_megakernel.py pass them explicitly per
    INTEGRATION-VIT.md §6), but the AUTHORITATIVE scalars come from
    ``optimizer.param_groups[0]`` via _opt_scalars_from — exactly as the decoder
    path does (which passes none of them). They are accepted-and-ignored here so a
    scheduled lr on the optimizer is always honored from the single source; the
    test's _Opt stand-in carries the same values in param_groups[0], so there is no
    divergence.
    """
    del lr, betas, weight_decay, eps  # see docstring: optimizer.param_groups is authoritative
    model_c = canonicalize_model(model_name)
    if (model_c, opt_name) not in _FUSED_L3_REAL:
        raise KeyError(
            f"fused_train_step: cell ({model_c}, {opt_name}) is not an L3-REAL "
            f"cell {sorted(_FUSED_L3_REAL)}")
    spec = _L3_REAL_SPEC.get(model_c)
    if spec is None:
        raise KeyError(
            f"fused_train_step: no L3-REAL ABI spec for model {model_c!r} "
            f"(known: {sorted(_L3_REAL_SPEC)})")
    ops = get_ops()
    if not hasattr(ops, "fused_step"):
        raise RuntimeError("ops.fused_step unavailable; build the extension.")

    # Collect the trainable params in named_parameters() ORDER (the flat layout).
    named = [(n, p) for n, p in torch_module.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named)
    if total != spec["total"]:
        raise RuntimeError(
            f"fused_train_step: {model_c} has {total} params but the L3-REAL "
            f"megakernel layout expects {spec['total']}. The model arch changed — "
            f"regenerate csrc/fused/sm_90/{model_c}_layout.cuh (or vit_layout/"
            f"mamba3_layout) and update the _*_TOTAL_ELEMS mirror.")
    p0 = named[0][1]
    if p0.dtype != torch.float32:
        raise RuntimeError(
            "fused_train_step requires fp32 params (the megakernel reads raw "
            "float memory). Disable AMP for the L3-REAL fused path.")
    device = p0.device

    cache = state_cache.get(model_c)
    if cache is None or cache["flat"].numel() != total:
        # Persistent flat param mirror + [m|v|extra]+loss state + a SEPARATE
        # reduced-grad output buffer (NOT `flat` — `flat` is params; reusing it
        # for grad would corrupt the weights). All allocated ONCE and reused
        # (reallocating state per step would reset momentum → never grok).
        flat = torch.empty(total, dtype=torch.float32, device=device)
        state = torch.zeros(3 * total + 1, dtype=torch.float32, device=device)
        grad_out = torch.zeros(total, dtype=torch.float32, device=device)
        names = [n for n, _ in named]
        cache = dict(flat=flat, state=state, grad_out=grad_out, names=names)
        state_cache[model_c] = cache
    elif cache["names"] != [n for n, _ in named]:
        raise RuntimeError(
            "fused_train_step: named_parameters() order changed between steps; "
            "the flat layout would be wrong. Do not re-create the model mid-run.")
    flat = cache["flat"]
    state = cache["state"]
    grad_out = cache["grad_out"]

    # Pack the CURRENT params into the flat buffer (named_parameters() order). The
    # kernel updates `flat` in place; we scatter it back afterward. (Copying in is
    # cheap relative to the fwd/bwd; it also picks up any eager change to a param.)
    off = 0
    for _, p in named:
        n = p.numel()
        flat[off:off + n].copy_(p.data.reshape(-1))
        off += n

    # Pack the per-sample input ++ targets into ONE contiguous `input` tensor, the
    # packing dispatch.cpp reads for this model (int tokens vs float patches).
    B = int(tokens.shape[0])
    per = spec["per"]
    # WGMMA TILING REQUIREMENT (not suppression): the TC launchers process the dW
    # K-loop in 16-step atoms and require B %% 16 == 0 (they return cudaErrorInvalidValue
    # otherwise → dispatch throws). The race full batch is not a multiple of 16
    # (e.g. decoder 4191, vit 4234), so on the wgmma path we truncate to the largest
    # multiple of 16 ≤ B, dropping the trailing < 16 samples for BOTH tokens and
    # targets consistently. This is a kernel input-shape constraint, identical to
    # what tuning.roofline.measure_tc_cell already does; the <0.4%% batch delta vs the
    # eager competitors is negligible and logged. The scalar path takes any B.
    if gemm_impl == "wgmma":
        B_tc = B - (B % 16)
        if B_tc < 16:
            raise RuntimeError(
                f"fused_train_step: wgmma path needs B>=16 (got B={B}); the TC "
                f"dW K-loop is 16-step atoms. Use a larger batch or gemm_impl='scalar'.")
        if B_tc != B:
            tokens = tokens[:B_tc]
            targets = targets[:B_tc]
            B = B_tc
    if spec["kind"] == "int_tokens":
        # decoder/mamba: int32 tokens[B*per] ++ int32 targets[B]. tokens is [B,per].
        S = int(tokens.shape[1])
        if S != per:
            raise RuntimeError(
                f"fused_train_step: {model_c} seq is {S} but the L3-REAL kernel "
                f"is compiled for kSeq={per}.")
        tok_i = tokens.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
        tgt_i = targets.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
        packed = torch.empty(B * (per + 1), dtype=torch.int32, device=device)
        packed[: B * per].copy_(tok_i)
        packed[B * per:].copy_(tgt_i)
    elif spec["kind"] == "float_patches":
        # vit: float32 patches[B*per] ++ int-target-bits[B] BIT-REINTERPRETED into
        # the trailing float slots. tokens here is the float patch tensor [B,16,49].
        n_in = int(tokens.numel() // B) if B else 0
        if n_in != per:
            raise RuntimeError(
                f"fused_train_step: {model_c} per-sample patch width is {n_in} "
                f"but the L3-REAL kernel is compiled for {per} (16*49). patches "
                f"must be [B, 16, 49] (or any [B, ...] with 784 elems/sample).")
        patch_f = tokens.to(device=device, dtype=torch.float32).contiguous().reshape(-1)
        # Bit-reinterpret the int32 targets into float slots (NOT a value cast):
        # int32 -> view as float32 copies the BIT PATTERN, lossless for all int32.
        tgt_bits = (targets.to(device=device, dtype=torch.int32).contiguous()
                    .reshape(-1).view(torch.float32))
        packed = torch.empty(B * (per + 1), dtype=torch.float32, device=device)
        packed[: B * per].copy_(patch_f)
        packed[B * per:].copy_(tgt_bits)
    else:
        raise RuntimeError(
            f"fused_train_step: unknown input kind {spec['kind']!r} for {model_c}")

    # neuralgrok SEAM: scatter the FRESH amplifier psi-net pack into the HEAD of the
    # `extra` state slice (state[2*total : 2*total + 3H+1]) before the kernel runs.
    # The TC launcher binds st.psi_W1/b1/W2 from extra[0/H/2H] (kPsiW1Off/...), so
    # the kernel reads the host-owned amplifier weights from here; the kernel's
    # apply_optimizer<NeuralGrok> reads but NEVER writes `extra` for neuralgrok (it
    # writes only m/v), so the pack we lay down is exactly what the device evaluates.
    # We re-lay it EVERY step (cheap, 49 floats) so a host-trained amplifier
    # (maybe_train_amplifier) is picked up on the very next launch — the snapshot
    # refresh the megakernel path needs (it cannot call mark_amplifier_dirty back
    # into the kernel). psi_pack() asserts hidden_dim==16 (the kernel's compile-time
    # psi width) so a mismatched MLP fails LOUD instead of mis-binding the extra
    # slice. Guard on the method so a non-NeuralGrok optimizer (the parity-test
    # stand-ins) never trips it.
    if opt_name == "neuralgrok":
        if not hasattr(optimizer, "psi_pack"):
            raise RuntimeError(
                "fused_train_step: neuralgrok L3 cell requires an optimizer exposing "
                "psi_pack() to fill the kernel's psi `extra` slice; got "
                f"{type(optimizer).__name__}. Pass a NeuralGrok instance.")
        pack = optimizer.psi_pack(device=device)
        npk = pack.numel()
        if 2 * total + npk > state.numel() - 1:   # -1: the trailing loss slot
            raise RuntimeError(
                f"fused_train_step: psi pack ({npk} floats) does not fit the extra "
                f"slice (state has {state.numel()}, needs >= {2 * total + npk + 1}).")
        state[2 * total:2 * total + npk].copy_(pack)

    scalars = _opt_scalars_from(optimizer, step)
    # opt_only=False selects the L3-REAL decoder path in dispatch.cpp (which reads
    # tokens/targets from `input` and runs the real fwd+bwd+adamw). The 33 surrogate
    # cells are NOT reached (only this exact model+optimizer routes to the real .cu).
    # `grad_out` is the ABI grad arg: the kernel writes the deterministically-reduced
    # weight grad there (and consumes it in the optimizer tail WITHOUT overwriting),
    # so after the call grad_out holds exactly the grad the AdamW step used — the
    # parity test slices it per-tensor against the oracle (the keystone check).
    # Forward alpha/lamb when the optimizer carries them (grokfast/grokadamw): the
    # in-kernel apply_optimizer<Opt> reads st.alpha/st.lamb. Omitting them would leave
    # the FusedScalars defaults (0.98/2.0) — wrong if the cell ran a configured value.
    _extra_scalars = {}
    if "alpha" in scalars:
        _extra_scalars["alpha"] = scalars["alpha"]
    if "lamb" in scalars:
        _extra_scalars["lamb"] = scalars["lamb"]
    # neuralgrok's affine psi term: the kernel reads st.beta in (psi*alpha+beta)*g.
    # _opt_scalars_from sets scalars["beta"] only for neuralgrok; omitting it would
    # leave the FusedScalars.beta default (0.0) and drop the +beta amplification.
    if "beta" in scalars:
        _extra_scalars["beta"] = scalars["beta"]
    # grokadamw decoder L3-TC mechanisms (i)+(ii): the kernel reads st.gamma
    # (layer-wise β1 = β1·(1-γ)^layer, P3) and st.grad_clip (GLOBAL grad-norm clip,
    # P2.5). _opt_scalars_from sets these only for grokadamw; omitting them leaves
    # the inert FusedScalars defaults (0.0 ⇒ single global β1 / no clip), which is
    # the hollow-pass the gate names. Forward them so the real mechanisms execute.
    if "gamma" in scalars:
        _extra_scalars["gamma"] = scalars["gamma"]
    if "grad_clip" in scalars:
        _extra_scalars["grad_clip"] = scalars["grad_clip"]
    ops.fused_step(model_c, opt_name, flat, packed, grad_out, state, scalars["lr"],
                   beta1=scalars["beta1"], beta2=scalars["beta2"],
                   eps=scalars["eps"], weight_decay=scalars["weight_decay"],
                   bc1=scalars["bc1"], bc2=scalars["bc2"],
                   step=scalars["step"], opt_only=False, gemm_impl=gemm_impl,
                   **_extra_scalars)

    # Scatter the updated flat params back into the live model (same order).
    off = 0
    for _, p in named:
        n = p.numel()
        p.data.reshape(-1).copy_(flat[off:off + n])
        off += n

    # The kernel wrote the mean CE loss into state[3*total]; read it back.
    loss = float(state[3 * total].item())
    if return_grad:
        # grad_out is the persistent buffer; clone so the caller's snapshot is not
        # overwritten by the next step. Returned as a flat [total] tensor (slice it
        # by the decoder layout offsets to get per-tensor grads).
        return loss, grad_out.clone()
    return loss


def _register_ready_cells():
    """Populate ``_FUSED_REGISTRY`` for every readiness-whitelisted cell.

    The registered callable is a thin shim so ``dispatch_fused`` / ``has_fused``
    see a real entry; the live race uses :func:`fused_optimizer_step` directly
    (it owns the persistent state + per-step scalars the low-level registry shim
    cannot carry). Registering for the detected GPU arch only (the compiled TU).
    """
    try:
        arch = detect_arch()
    except UnsupportedArchError:
        return
    impl = normalize_arch(arch) if not isinstance(arch, int) else arch
    if impl not in (90, 942):
        return
    for (m, o) in _FUSED_READY:
        def _shim(params, inputs, grads, state, lr, _m=m, _o=o):
            ops = get_ops()
            ops.fused_step(_m, _o, params, inputs, grads, state, float(lr),
                           opt_only=True)
            return params
        register_fused(m, o, arch)(_shim)


# Populate the registry at import (no-op / silent on CPU + non-GPU arches).
_register_ready_cells()
