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
    # real, not a hollow pass. mamba×neuralgrok is ALSO wired (the wave-2 mamba lane
    # below) — the mamba TC launcher now DOES include the OptId::NeuralGrok case.
    ("transformer_decoder", "neuralgrok"),
    ("vit", "neuralgrok"),
    # neuralgrok (mamba — wave-2 mamba lane): the mamba TC launcher gained the
    # OptId::NeuralGrok case (mega_mamba_real_adamw_tc_launcher.cu) and the mamba
    # dispatch.cpp cap widened to the generic mb_opt_id>=0 (mirrors decoder/vit). The
    # psi-net pack is model-INDEPENDENT (fused_train_step scatters psi_pack() into the
    # `extra` slice for any model), and apply_optimizer<NeuralGrok> is a pure
    # elementwise tail over the deterministically-reduced grad — so the SAME psi-net
    # MLP runs on the mamba TC-reduced grad. State-gate clean by the same argument as
    # decoder/vit (kernel + eager consume the SAME grad + psi weights). The amplifier
    # host-training cadence is the eager NeuralGrok's concern; the L3-TC step runs the
    # deployed psi MLP faithfully (the gate anchors (1b) to canonical neuralgrok.h).
    ("mamba3", "neuralgrok"),
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
    # grokadamw (vit): CONVERTED (wave-2 vit lane). Same 3 mechanisms on the vit TC
    # kernel — fused_vit_megakernel_tc gained the IDENTICAL P2.5 global grad-norm clip
    # + P3 per-tensor layer-wise β1 (t == flat kVitOffsets index; cls_token t=0), and
    # the vit launcher already dispatches opt_id=3. Single-step state-gate + multi-step
    # parity validated (the clip is the step-1-inert mechanism the single-step gate is
    # blind to, so the multi-step parity is the load-bearing check, exactly as decoder).
    ("vit", "grokadamw"),
    # grokadamw (mamba — wave-2 mamba lane): the mamba TC kernel gained the IDENTICAL
    # 3-mechanism support as decoder/vit — P2.5 GLOBAL grad-norm clip (deterministic
    # ascending-CTA reduce over the reduced grad → clip_coef) + P3 per-tensor
    # layer-wise β1 = β1·(1-γ)^t (t == flat kMambaOffsets index, rebased bc1) +
    # adaptive-α (static in-context = the LIVE α the host computes pre-launch and
    # forwards via FusedScalars). γ/grad_clip thread through FusedScalars. The mamba
    # TC launcher dispatches opt_id=3 (OptId::GrokAdamW). ema = the `extra` slice
    # (cold-start seed at step 1, identical to decoder). Single persistent wgmma
    # launch — the clip + β1-rebase are IN-KERNEL stages, no extra launch.
    ("mamba3", "grokadamw"),
    # prodigy (decoder): CONVERTED (wave-2 decoder lane). STAGED global-d. The
    # adaptive learning rate d (a cross-ALL-tensors reduction over EVERY element of
    # EVERY tensor) is computed IN-KERNEL as a new phase (P2.6, between the grad
    # reduction B2 and the optimizer tail P3) — NOT a separate launch, so it stays a
    # SINGLE persistent megakernel. The phase byte-matches the live eager multi-tensor
    # estimator (prodigy_sm90.cuh:465-544 → prodigy.py): per-CTA (r,s) owner-computes
    # reduction (no float atomic, deterministic) → beta3-EMA decay of the PERSISTED
    # (r_ema,s_ema) scalars → d = max(d_prev, d_coef·r_ema/|s_ema|). The cell extends
    # the state buffer to carry the trajectory anchor param_init + the 3 persisted
    # estimator scalars (fused_train_step sizes it 4*total+4). At step 1 param_init==
    # params ⇒ r=0 ⇒ d=d0 (cold-start matches eager _d_lr=d0); the d-adaptation fires
    # at step≥2, so the single-step state-gate is necessary-not-sufficient and the
    # MULTI-STEP parity (kernel tracks eager; a d-frozen control diverges) is the
    # load-bearing check, exactly as grokadamw's clip. mamba×prodigy is now CONVERTED
    # too (wave-2 mamba lane — see the ("mamba3","prodigy") entry below).
    ("transformer_decoder", "prodigy"),
    # prodigy (vit): CONVERTED (wave-2 vit lane). The SAME STAGED global-d P2.6 phase
    # on the vit TC kernel (fused_vit_megakernel_tc): per-CTA (r,s) owner-computes
    # reduction over kVitSizes/kVitOffsets → beta3-EMA decay of the persisted (r_ema,
    # s_ema) → d=max(d_prev,d_coef·r_ema/|s_ema|). The vit launcher binds s_track/
    # param_init/prodigy_persist + routes opt_id=5; fused_train_step's 4*total+4 state
    # sizing + param_init seeding are model-agnostic (keyed on opt_name=="prodigy"), so
    # they apply to vit unchanged. Same MULTI-STEP load-bearing check (d fires step≥2).
    ("vit", "prodigy"),
    # prodigy (mamba): BLOCKED — A/A/A determinism FAILS on the mamba TC kernel. This is a
    # SEPARATE defect from the decoder/vit prodigy P2.6 nondeterminism that is now RESOLVED
    # (see the ✓ note below); do NOT conflate them. The P2.6 d-reduction kernel + launcher
    # binding ARE LANDED (dormant, if-constexpr'd), but the production routing is intentionally
    # NOT registered here: the prodigy<Opt> mamba TC kernel produces NON-DETERMINISTIC
    # loss+grad across A/A/A re-runs from a bit-identical init (loss 4.7554/4.7555/4.7553;
    # grad maxd ~1e-2 vs the DETERMINISTIC adamw grad on the SAME input — every one of the 28
    # grad tensors differs). Note the SYMPTOM (~1e-2 grad drift) is NOT the decoder/vit class
    # (which the fixed-partition P2.6 below resolved) — the mamba block is in the shared mamba
    # scan/forward, not the prodigy reduction. Ruled OUT for mamba: NOT the 4*total+4 state
    # buffer (adamw forced to that size is bit-identical), NOT opt_reduce/acts overlap (offset
    # probe: opt_reduce [132789961,132790226) vs acts [0,55056384) — disjoint), NOT an intra-
    # block smem race (compute-sanitizer racecheck = 0 hazards) NOR a __syncthreads/barrier
    # misuse (synccheck = 0 errors). Until the mamba forward is fixed, mamba×prodigy stays on
    # the eager/per-op path (no silent non-deterministic production cell — the no-suppression
    # rule).
    #
    # ✓ RESOLVED — decoder+vit prodigy A/A/A (was the ⚠ escalation; now FIXED):
    #   ("transformer_decoder","prodigy") and ("vit","prodigy") are REGISTERED-CONVERTED
    #   (in _FUSED_L3_REAL + _L3_WGMMA_CELLS above) and now PASS A/A/A bit-exact via
    #   test_l3tc_tail_gate — step 1 (loss 4.775414 ×3, params rel 8.8e-10) AND step>=40 with
    #   d adapted off d0 (params bit-identical maxd=0, d_lr 4.834174e-04 ×3); multistep parity
    #   tracks eager to <1e-4. FIX: the megakernel prodigy P2.6 (fused_{decoder,vit}_
    #   megakernel.cuh) was rewritten from the work-steal q.next_block() drain + extra
    #   bar.sync_reset() to a FIXED-PARTITION reduce — each CTA owns a fixed contiguous element
    #   range of the flat [0,total) arrays (params/param_init/grad are parallel flat blobs, so a
    #   contiguous flat-range Σ == the cross-all-tensors Σ), published to per-CTA (r,s) slots →
    #   ONE plain grid barrier → CTA0 ascending owner-sum → EMA/update_d → broadcast. This is the
    #   IDENTICAL deterministic shape as the A/A/A-clean GrokAdamW P2.5 grad-norm reduce; the
    #   per-element contribution still CALLS canonical prodigy_partials_step (prodigy.h, single
    #   source). The work-steal drain made each CTA's claimed-tensor SUBSET timing-dependent, so
    #   its per-CTA partial regrouped the same values differently → an fp32-order-dependent d at
    #   step>=2 (a COMPONENT_CONTRACT deterministic-reduction violation). ("vit","muon") (P2.7
    #   NS) was always A/A/A-CLEAN (per-matrix cooperative, no work-steal drain). NOTE: the
    #   opt_stages_precompute.cuh prodigy_precompute_reduce_phaseA/B helpers (the old work-steal
    #   form) are now UNUSED by decoder/vit (kept for reference / the dormant mamba path).
    #   mamba×prodigy remains the only blocked prodigy cell, on its OWN mamba-forward symptom.
    # muon (vit): CONVERTED (wave-2 vit lane). STAGED grid-cooperative Newton-Schulz.
    # The NS orthogonalization of the 2D weights (kVitMuon2D: 11 matrices) is an
    # IN-KERNEL phase (P2.7, between B2 and P3) — all CTAs cooperate on one matrix at a
    # time (buf=μ·buf+g with buf the PERSISTENT m-slice, ‖buf‖→inv_norm→X, then 5×NS
    # {A=XXᵀ, AX, AAX, combine}), then the muon.h apply; the 1D/non-2D weights take the
    # AdamW tail in P3 (muon.py auto-split by p.ndim). Still a SINGLE persistent launch
    # (no separate NS launch — the in-kernel grid-cooperative form places one CTA/SM:
    # VitTcSmem is static ~7KB, NS buffers are HBM). The vit launcher binds the momentum
    # to st.exp_avg + routes opt_id=7. mamba×muon is now CONVERTED too (see the
    # ("mamba3","muon") entry below — single forward + NS precompute, NO 2nd-forward race).
    ("vit", "muon"),
    # muon (decoder — wave-2 decoder lane): the SAME STAGED grid-cooperative Newton-
    # Schulz P2.7 phase ported onto the decoder TC kernel (fused_decoder_megakernel_tc):
    # the 11 2D weights (dectc::kDecMuon2D — incl. tok[99,128]/pos[4,128], shapes no vit
    # muon matrix had) are orthogonalized in-kernel (buf=μ·buf+g with buf the PERSISTENT
    # m-slice, ‖buf‖→inv_norm→X, 5×NS {A=XXᵀ,AX,AAX,combine}, then the muon.h apply); the
    # 1D/non-2D weights take the AdamW aux tail in P3 (muon.py auto-split by p.ndim). The
    # decoder launcher binds the momentum to st.exp_avg + routes opt_id=7 (case OptId::Muon).
    # The NS scratch is carved in dec_tc_workspace_floats (dec_tc_muon_floats), after the
    # Prodigy reduce slots — so the 5 already-green decoder cells stay byte-identical.
    # The aux_lr/aux_betas plumbing + 4*total state sizing are model-agnostic. Still a
    # SINGLE persistent launch.
    ("transformer_decoder", "muon"),
    # muon (mamba — wave-2 mamba lane): the SAME in-kernel P2.7 grid-cooperative Newton-
    # Schulz ported onto fused_mamba_megakernel_tc (the grid-cooperative NS phase, model-
    # agnostic in opt_stages_precompute.cuh). The 13 ndim==2 mamba weights (mbtc::kMbMuon2D
    # — tok[99,128]/pos[8,128]/A_log[256,16]/in_proj[512,128]/x_proj[40,256]/dt_proj[256,8]/
    # out_proj[128,256] × 2 layers + out.weight[97,128]) are orthogonalized in-kernel
    # (buf=μ·buf+g with buf the PERSISTENT m-slice, ‖buf‖→inv_norm→X, 5×NS {A=XXᵀ,AX,AAX,
    # combine}, then the muon.h apply); the 1D/non-2D weights (D, conv1d.w/b, dt_proj.bias,
    # LN γ/β, out.bias) take the AdamW aux tail in P3 (muon.py auto-split by p.ndim — the
    # ndim==3 conv1d.weight also routes to the aux tail). The mamba launcher binds the
    # momentum to st.exp_avg + routes opt_id=7 (case OptId::Muon). The NS scratch is carved
    # in mb_tc_workspace_floats (mb_tc_muon_floats), after the LookSAM scratch — so the
    # already-green mamba cells stay byte-identical. CRITICAL DETERMINISM NOTE: unlike the
    # SAM-2nd-pass mamba cells (looksam/SG11/15) which re-run the mamba forward and hit the
    # shared mamba-forward A/A/A race, muon/mamba is a SINGLE forward + NS precompute (no 2nd
    # forward), so it does NOT hit that race — VERIFIED A/A/A bit-exact by test_l3tc_tail_gate
    # (muon/mamba). Still a SINGLE persistent launch.
    ("mamba3", "muon"),
    # prodigy (mamba): CONVERTED — the A/A/A race is FIXED (register-pressure wgmma-accumulator
    # spill in the mamba TC backward; see the test_l3tc_tail_gate "prodigy/mamba" note + the
    # fused-dB/dC-reduce + a_save-drop + __noinline__ fix in model_stage_mamba{3,_tc}.cuh).
    # prodigy/mamba now PASSES A/A/A bit-exact on the production fused_train_step path
    # (loss/grad/param maxd=0 ×4). The dispatch.cpp mamba×prodigy carve-out is lifted.
    ("mamba3", "prodigy"),
    # looksam (mamba): CONVERTED — same root cause + fix (the SAM block inlines the heavy
    # fwd+bwd a SECOND time; __noinline__ + the scan-bwd footprint cut keep the megakernel
    # frame under the spill threshold). looksam/mamba now PASSES A/A/A bit-exact on the
    # production path (SAM step AND SAM-off step; loss/grad/param maxd=0 ×4). The dispatch.cpp
    # mb_looksam carve-out is lifted. (supergrok11/15 mamba stay dormant — code-ABSENT SG path
    # in the mamba kernel/launcher, a feature port, not this determinism fix.)
    ("mamba3", "looksam"),
    # looksam (decoder — SAM-tier lane): CONVERTED. The MODEL-COUPLED SAM 2nd backward
    # (st.sam_dir = g_sam − g, INTEGRATION-OPTSTAGES §6) is now an IN-KERNEL phase (P2.4,
    # between B2 and P2.5/P3) on the decoder TC kernel (fused_decoder_megakernel.cuh):
    # on every-k SAM steps it computes the GLOBAL ‖g‖ (deterministic ascending-CTA reduce,
    # IDENTICAL shape to GrokAdamW's P2.5), perturbs p'=p+(rho/‖g‖)·g (backup saved for the
    # exact restore), runs a FULL SECOND in-kernel fwd+bwd at p' (re-invoking dectc_forward_
    # tile/dectc_backward_tile + the deterministic grad assembly into a SEPARATE sam_grad
    # buffer so the first grad is untouched), writes sam_dir=g_sam−g into the PERSISTENT
    # `extra` state slice, and restores p. On the k−1 intervening steps (looksam_sam==0) the
    # cached sam_dir is reused verbatim, NO 2nd pass. The apply tail (apply_optimizer<LookSAM>,
    # already in opt_components.cuh) blends g_adj=(1−α)g+α·sam_dir and runs AdamW. Still a
    # SINGLE persistent launch (the SAM 2nd backward is an in-kernel phase, the same shape as
    # Prodigy's P2.6 / Muon's P2.7). The SAM-step gate is host-computed from the every-k cadence
    # ((step-1)%k==0, _opt_scalars_from) and threaded via FusedScalars.looksam_sam; rho rides
    # FusedScalars.rho, α the existing alpha. The transient backup+sam_grad buffers are carved
    # in dec_tc_workspace_floats (dec_tc_looksam_floats, after the Muon scratch — the prior
    # cells stay byte-identical). The decoder launcher binds st.sam_dir to the extra slice +
    # routes opt_id=4 (case OptId::LookSAM). DETERMINISM: every reduction is the deterministic
    # ascending shape; the 2nd fwd+bwd reuses the A/A/A-clean first-pass machinery → A/A/A by
    # construction. vit + mamba looksam follow (the same in-kernel phase ported per model).
    ("transformer_decoder", "looksam"),
    # looksam (vit — SAM-tier lane): CONVERTED. The IDENTICAL in-kernel P2.4 SAM 2nd backward,
    # ported onto the ViT TC kernel (fused_vit_megakernel.cuh): the ViT tile fns (vittc_forward_
    # tile/vittc_backward_tile + vittc_dw_*/vittc_clspos_owner_scan/vittc_lnvec_reduce) run the
    # 2nd fwd+bwd at p' into a SEPARATE sam_grad buffer; the transient backup+sam_grad live in
    # vit_tc_workspace_floats (vit_tc_looksam_floats, after the Muon scratch — prior cells
    # byte-identical). The vit launcher binds st.sam_dir to the extra slice + routes opt_id=4
    # (case OptId::LookSAM). vit's forward is deterministic (vit Prodigy/Muon are A/A/A-green),
    # so the SAM 2nd backward is A/A/A by construction (same fixed reductions as the decoder).
    ("vit", "looksam"),
    # looksam (mamba): BLOCKED — A/A/A determinism FAILS on the mamba TC kernel (same class
    # as mamba×prodigy). The in-kernel P2.4 SAM 2nd backward IS code-landed + compile-clean
    # (fused_mamba_megakernel.cuh, if-constexpr'd → mamba×{adamw,lion,grokfast} stay bit-
    # identical A/A/A, and the mamba launcher routes opt_id=4), but the LookSAM template
    # INSTANTIATION (fused_mamba_megakernel_tc<LookSAM>) is NOT registered here: its grad is
    # NON-DETERMINISTIC across bit-identical re-runs (loss varies ~5e-5, grad maxΔ~1e-3, and
    # intermittent NaN). MEASURED: the non-determinism appears EVEN on a SAM-OFF step (no 2nd
    # backward), so it is NOT in the SAM phase code (the sam_dir reductions are fixed-ownership
    # mirrors of P1+P2) — it is the SAME latent shared mamba scan/forward race that blocks
    # mamba×prodigy, exposed here by the LookSAM instantiation's register/occupancy profile
    # (the if-constexpr'd P2.4 changes ptxas allocation). Registering it would ship a non-
    # deterministic cell (no-suppression violation). dispatch.cpp gates it to eager via the
    # mb_looksam carve-out (mirroring the mamba×prodigy carve-out). Lift BOTH once the mamba
    # forward is fixed by its owner (megakernel_common.cuh GridBarrier / model_stage_mamba3.cuh).
    # ✓ decoder + vit looksam ARE registered-converted (A/A/A bit-exact); only mamba is blocked.
    # supergrok11 (decoder — SAM-tier lane): CONVERTED. SG11 REUSES the IDENTICAL in-kernel
    # SAM 2nd backward as looksam (P2.4) — but the elementwise write differs: it stores
    # sharpness=(g_sam−g)² (supergrok11_sm90.cuh:246, NOT looksam's sam_dir=g_sam−g), the
    # 2nd MLP input. ON TOP of that 2nd pass runs a per-TENSOR meta-net mu precompute (P2.45,
    # staged into the P3 drain): mu=sg_rescale·phi(g,sharpness) over each tensor + a per-tensor
    # cosine gate (clamp(cos(g,mu),0,1)), via sg11_precompute_mu_and_gate_for_tensor<32>
    # (opt_stages_precompute.cuh, CPU-fp64-validated 9/9). The apply tail (sg11_sweep_b_step:
    # smart_grad=g+(1−gate)·alpha·mu, AdamW) runs in P3. SharpnessMetaNet hidden_dim=32
    # (NOT 64 — a known prior bug). The phi weights + sharpness ride the extended state buffer
    # (4*total+1+kSgPhiPack); fused_train_step scatters the phi pack each step (like NeuralGrok's
    # psi pack) and sizes the state. _opt_scalars_from forwards sg_rescale + alpha + rho (the SAM
    # radius) + looksam_sam (the SAM-step gate, same every-k cadence). The decoder launcher binds
    # st.mu/st.sharpness/st.sg_phi_* + routes opt_id=8. Single persistent launch (SAM 2nd backward
    # + mu precompute are in-kernel phases). A/A/A by construction (same fixed reductions as
    # looksam/the first pass). mamba×supergrok11 is BLOCKED (shared mamba-forward A/A/A race).
    ("transformer_decoder", "supergrok11"),
    # supergrok11 (vit — SAM-tier lane): CONVERTED. The IDENTICAL in-kernel SAM 2nd backward +
    # per-tensor mu precompute ported onto the ViT TC kernel (fused_vit_megakernel.cuh): the ViT
    # tile fns run the 2nd fwd+bwd at p' → sharpness=(g_sam−g)², the P2.45 phase fills mu over
    # kVitSizes/kVitOffsets, the apply tail runs in P3. The vit launcher binds st.mu/st.sharpness/
    # st.sg_phi_* + routes opt_id=8. vit's forward is deterministic (vit looksam/prodigy/muon are
    # A/A/A-green), so SG11 is A/A/A by construction. mamba×supergrok11 stays blocked.
    ("vit", "supergrok11"),
    # supergrok15 (decoder — SAM-tier lane): CONVERTED. SAME SAM 2nd backward + per-tensor mu
    # precompute as SG11, but SIMPLER tail: the gate is a HOST SCALAR (st.gate=sigmoid(accuracy)),
    # NO per-tensor cosine stage — so the precompute is just mu=sg_rescale·phi(g,sharpness) via
    # sg15_precompute_mu_for_tensor<32>. The apply tail (sg15_sweep_b_step) does the per-coord
    # alpha clip (clamp(alpha·(1+mu),0,alpha_max)) + smart_grad=g+gate·a·mu + AdamW. SharpnessMetaNet
    # hidden_dim=32. State/phi-pack/scalars identical to SG11 (the launcher routes opt_id=9; the
    # gate is the host scalar, not per-tensor). Single persistent launch; A/A/A by construction.
    # mamba×supergrok15 is BLOCKED (shared mamba-forward A/A/A race).
    ("transformer_decoder", "supergrok15"),
    # supergrok15 (vit — SAM-tier lane): CONVERTED. The IDENTICAL SAM 2nd backward + mu precompute
    # on the ViT TC kernel; host-scalar gate. The vit launcher routes opt_id=9. mamba stays blocked.
    ("vit", "supergrok15"),
    # supergrok2 (decoder/vit — the LAST/hardest cell): CONVERTED via the DEDICATED
    # ops.sg2_fused_step entry (sg2_fused_step in dispatch.cpp → mega_{decoder,vit}_sg2_tc).
    # The FULL CSA/HCA/PEER/GRU meta-net AS the optimizer phase: P3-SG2 runs the in-kernel
    # SEGMENTED SORT (STAGE -1, index-tie-break strategy A) → S0..S5, reading st.sharpness
    # from the SAM 2nd backward (P2.4, shared with SG11/15). Weights from HBM; per-tensor
    # scalar arrays via the dedicated entry. Single-step + 200-step parity vs the per-op
    # oracle (ops.supergrok2_batched_step, SAME low-rank indexer) < 1e-5; A/A/A bit-exact.
    # HONEST CAVEAT: SG2 won't GROK (CSA lightning-indexer fidelity gap, N>64 — the gate's
    # N>64 probe REPORTS it). mamba×supergrok2 BLOCKED (SAM double-forward + segmented sort
    # re-trip the shared mamba-forward A/A/A race; code-absent in the mamba kernel/launcher).
    ("transformer_decoder", "supergrok2"),
    ("vit", "supergrok2"),
    # supergrok11/15 (mamba — wave-4 mamba SG lane): CONVERTED. The decoder/vit SG11/15
    # in-kernel SAM 2nd backward (P2.4, sharpness=(g_sam−g)²) + the per-tensor meta-net mu
    # precompute (P2.45/P3, mu=sg_rescale·phi(g,sharpness); SG11 also the cosine gate) are
    # PORTED onto the mamba TC kernel (fused_mamba_megakernel.cuh) + launcher (opt_id 8/9).
    # The shared-mamba-forward A/A/A race that blocked the SAM 2nd-pass mamba cells is FIXED
    # (commit 0b57f7e — register-pressure wgmma-accumulator spill; it un-dormanted
    # looksam/mamba, which proved the SAM double-forward path is race-free). SG11/15 ride the
    # SAME single-launch TC tail (the meta-net mu is a per-element/per-tensor body, NOT the
    # full CSA/HCA meta-net of SG2). The phi pack + sharpness ride the extended state buffer
    # (4*total+1+kSgPhiPack); fused_train_step scatters the phi pack each step (model-agnostic,
    # shared with decoder/vit) and sizes the state. HONEST CAVEAT: SG11/15 mamba reach L3-TC +
    # single-step parity but WON'T GROK on L3 (the meta-net is untrained — a separate
    # owner-approved host-training task, NOT done here). A/A/A re-verified by the tail gate; if
    # either trips the race it is gated OUT of _L3_WGMMA_CELLS (landed dormant — no
    # non-deterministic production cell).
    ("mamba3", "supergrok11"),
    ("mamba3", "supergrok15"),
    # supergrok2 (mamba — the LAST/hardest cell): the FULL CSA/HCA/PEER/GRU meta-net + the
    # in-kernel segmented sort (STAGE -1) + the SAM 2nd backward, PORTED onto the mamba
    # megakernel (P3-SG2) + the DEDICATED mega_mamba_sg2_tc launcher entry (the meta-net weight
    # bundle + per-tensor scalar ARRAYS ride ops.sg2_fused_step, NOT the generic fused_step).
    # The SAM double-forward + segmented sort re-exercise the shared mamba forward (the
    # .sg2_spec.md-flagged risk), but the now-fixed race (0b57f7e) COVERS it: the tail gate
    # confirms A/A/A bit-determinism (loss 4.754265 ×3; grad/param/sharp/gru-eq all True) +
    # single-step parity vs the per-op oracle (WORST 3.1e-7 < 1e-5). HONEST CAVEAT: SG2 mamba
    # won't GROK (the CSA idx_UQ fidelity gap, same as decoder/vit — out of scope; the gate's
    # N>64 CSA fidelity probe REPORTS it). The DEDICATED entry path is reached because
    # has_l3_real is True; the _L3_WGMMA_CELLS membership below is the production gate.
    ("mamba3", "supergrok2"),
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
    # prodigy (decoder L3-TC, STAGED global-d): the kernel's P2.6 d-reduction reads
    # st.d0/st.d_coef/st.beta3. Prodigy's param_groups carry d0/d_coef; beta3 is
    # sqrt(beta2) (the canonical persistent-EMA decay, prodigy.py:178). Distinguish
    # from the other branches by the d0 key (only Prodigy has it). Forward them so
    # the kernel's estimator matches the eager multi-tensor path byte-for-byte.
    elif "d0" in g and "d_coef" in g:
        import math as _math
        out["d0"] = float(g["d0"])
        out["d_coef"] = float(g["d_coef"])
        out["beta3"] = _math.sqrt(beta2)
    elif (hasattr(optimizer, "meta_net") and hasattr(optimizer, "sam_rho")
          and hasattr(getattr(optimizer, "meta_net", None), "num_experts")):
        # supergrok2 (decoder/vit L3-TC, DEDICATED entry). MUST precede the SG11/15
        # branch below: SG2's CSAHCAMetaNet ALSO has .rescale (so that branch's
        # distinguisher would WRONGLY match it), but SG2's meta-net is a CSAHCAMetaNet
        # (has .num_experts/.csa_compress) which the SharpnessMetaNet (SG11/15) lacks.
        # The PER-TENSOR scalars (alpha_i/gru_decay_i/lamb_eff/beta1_i/bc1_i/bc2_i) do
        # NOT fit the global FusedScalars POD — they are computed + passed as device
        # ARRAYS by the SG2 branch of fused_train_step (via ops.sg2_fused_step), NOT
        # here. This branch only sets the SHARED scalars (lr/beta2/wd/eps already in
        # `out`) + the SAM 2nd-backward gate (rho = sam_rho, looksam_sam = the every-k
        # SAM-step cadence). The SAM cadence: train_supergrok2 fires sam_step on a
        # dynamic _get_effective_sam_freq; for the L3 path we use the sam_freq_min base
        # and fire at step 1 + every sam_freq_min steps (a faithful base cadence; the
        # adaptive freq depends on metrics the L3 step does not feed). REPORTED in the
        # gate. wd_eff = the progressive WD ramp (same _get_effective_wd as eager).
        out["rho"] = float(getattr(optimizer, "sam_rho", 0.05))
        sfmin = max(1, int(getattr(optimizer, "sam_freq_min", 3)))
        out["looksam_sam"] = 1.0 if (int(step) % sfmin == 0 or int(step) == 1) else 0.0
        # decoupled-WD ramp (eager _get_effective_wd); fall back to the group wd.
        try:
            out["weight_decay"] = float(optimizer._get_effective_wd(wd))
        except Exception:
            out["weight_decay"] = wd
        return out
    elif (hasattr(optimizer, "meta_net") and hasattr(optimizer, "sam_rho")
          and hasattr(getattr(optimizer, "meta_net", None), "rescale")):
        # supergrok11 / supergrok15 (decoder/vit L3-TC, MODEL-COUPLED SAM 2nd backward +
        # per-tensor meta-net mu). The SG optimizers carry their hyperparameters on the
        # INSTANCE (not param_groups: the group has only lr/betas/eps/wd from `defaults`),
        # so read them from the optimizer object. The distinguisher from SG2 (which ALSO has
        # meta_net + sam_rho but a CSAHCAMetaNet) is meta_net.rescale — a SharpnessMetaNet
        # attribute SG2's meta-net lacks; SG2 is not an L3-REAL cell anyway, but this keeps
        # the branch unambiguous. We FAITHFULLY reproduce the eager binding's scalar mapping
        # so the in-kernel apply tail (sg{11,15}_sweep_b_step) consumes the SAME effective
        # scalars the eager supergrok{11,15}_fused_step would:
        #
        #   sg_rescale = meta_net.rescale (the SharpnessMetaNet output scale; mu =
        #                rescale·phi(g, sharpness)). The phi WEIGHTS (W1/b1/W2/b2) are
        #                scattered into the state buffer by fused_train_step (the kernel
        #                reads them from there); only this scalar is in the FusedScalars POD.
        #   rho        = sam_rho (the SAM perturbation radius; the in-kernel P2.4 perturb
        #                uses st.rho/‖g‖, identical to the eager sam_perturb_all).
        #   looksam_sam= the SAM-step gate from the every-(2·meta_update_freq) cadence the
        #                race uses (train_supergrok: sam_freq = max(1, muf*2)); we recompute
        #                it from `step` so it does NOT depend on the eager step()/global_step
        #                having run (mirrors the looksam looksam_sam computation).
        #   ramp       = the warmup ramp factor (eager _get_ramp_factor at this step).
        #   base_alpha = the cached meta-net strength (eager _cached_alpha = alpha_init until
        #                a (loss,acc) signal updates it; the L3 path does not feed losses, so
        #                it stays alpha_init — faithful, like grokadamw's static-α in-context).
        #   layer_alpha= base_alpha·(1-gamma_alpha)^(max_idx-idx), clamped [0,1]. gamma_alpha
        #                defaults to 0 (both SG), so it is UNIFORM = clamp(base_alpha,0,1) for
        #                every tensor — representable as the single FusedScalars.alpha.
        import math as _math
        rescale = float(optimizer.meta_net.get_weights()[4])
        out["sg_rescale"] = rescale
        out["rho"] = float(getattr(optimizer, "sam_rho", 0.05))
        # SAM cadence: the race fires sam_step every sam_freq = max(1, meta_update_freq*2)
        # steps (train_supergrok). The eager should-SAM check is step % sam_freq == 0 (1-based
        # `step` here is the authoritative counter the L3 path passes). At step 1 the gate (k=…)
        # decides if the 2nd backward runs; on intervening steps the cached sharpness is reused.
        muf = int(getattr(optimizer, "meta_update_freq", 5))
        sam_freq = max(1, muf * 2)
        out["looksam_sam"] = 1.0 if (int(step) % sam_freq == 0 or int(step) == 1) else 0.0
        # ramp + base_alpha + layer_alpha (uniform with gamma_alpha=0). _get_ramp_factor reads
        # _global_step; the L3 path owns `step`, so compute the ramp from `step` directly to be
        # independent of whether eager step() ran (warmup_steps/warmup_ramp are instance attrs).
        ws = int(getattr(optimizer, "warmup_steps", 100))
        wr = max(1, int(getattr(optimizer, "warmup_ramp", 100)))
        gstep = int(step)
        ramp = 0.0 if gstep <= ws else min(1.0, (gstep - ws) / wr)
        base_alpha = float(getattr(optimizer, "_cached_alpha",
                                   getattr(optimizer, "alpha_init", 0.98)))
        layer_alpha = max(0.0, min(1.0, base_alpha))   # gamma_alpha=0 ⇒ uniform
        lamb = float(getattr(optimizer, "lamb", 2.0))
        # The kernel's sg11_sweep_b_step does smart_grad = g + (1-gate)·alpha·mu; the eager
        # SG11 binding does g_eff = g + ramp·lamb·(1-gate)·layer_alpha·mu (lamb_eff =
        # ramp·lamb·(1-gate)·alpha). Fold ramp·lamb into alpha so the single canonical term
        # matches: alpha_kernel = ramp·lamb·layer_alpha. The kernel's sg15_sweep_b_step does
        # smart_grad = g + gate_global·a·mu with a = clamp(alpha_base·(1+mu),0,alpha_max); the
        # eager SG15 binding passes gate_global = ramp·gate_signal·lamb, alpha_base = alpha_max
        # = layer_alpha. So SG15's `alpha`/`alpha_max` stay = layer_alpha and the gate carries
        # ramp·gate_signal·lamb (set below); SG11's alpha carries ramp·lamb·layer_alpha and its
        # gate is the per-tensor cosine (computed in-kernel — NOT a host scalar).
        is_sg15 = hasattr(optimizer, "_get_gate_signal")
        if is_sg15:
            out["alpha"] = layer_alpha
            out["alpha_max"] = layer_alpha
            # gate_global = ramp·gate_signal·lamb (the eager lamb_eff for SG15). gate_signal =
            # sigmoid(gate_scale·(acc - gate_thresh)); _cached_train_acc=0 at step 1 ⇒ a small
            # positive sigmoid. Read it from the optimizer (it owns _get_gate_signal).
            try:
                gate_signal = float(optimizer._get_gate_signal())
            except Exception:
                gate_signal = 0.0
            out["gate"] = (ramp * gate_signal * lamb) if ramp > 0.0 else 0.0
        else:
            # SG11: alpha absorbs ramp·lamb·layer_alpha; the gate is the per-tensor cosine the
            # kernel computes in P2.45 (NOT passed as a scalar — leave FusedScalars.gate inert).
            out["alpha"] = (ramp * lamb * layer_alpha) if ramp > 0.0 else 0.0
        return out
    elif "rho" in g and "k" in g and "alpha" in g:
        # looksam (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM 2nd backward): the kernel's
        # P2.4 phase reads st.rho (perturbation radius) + st.looksam_sam (the every-k
        # SAM-step gate) and st.alpha (the cached-direction blend, looksam.h:81). LookSAM's
        # param_group carries `rho`, `k`, `alpha` (and NO d0/momentum/grokfast_alpha/beta,
        # which distinguishes it from the branches above). Forward them so the kernel runs
        # the live mechanism. The SAM cadence: eager should_sam_step() is _global_step % k
        # == 0 with _global_step starting at 0 and incrementing in step() — so SAM fires at
        # global_step ∈ {0, k, 2k} ↔ 1-based `step` ∈ {1, 1+k, 1+2k} ↔ (step-1) % k == 0.
        # We compute the flag from `step` (the authoritative 1-based counter the L3 path
        # passes) so it does NOT depend on whether the eager step()/global_step ran.
        out["alpha"] = float(g["alpha"])
        out["rho"] = float(g["rho"])
        kk = int(g.get("k", 5))
        out["looksam_sam"] = 1.0 if ((int(step) - 1) % max(kk, 1) == 0) else 0.0
    elif "momentum" in g and "betas" not in g:
        # muon (vit L3-TC, STAGED grid-cooperative NS): the kernel's P2.7 reads the
        # momentum as st.beta1 (buf = β1·buf + g, muon.h:43-44). Muon's 2D param_group
        # carries `momentum` (0.95), NOT `betas` (so `betas` defaulted to (0.9,0.999)
        # above — WRONG for the buf decay). Override beta1 = momentum. lr is already the
        # 2D group's muon_lr (param_groups[0]); wd flows through; the kernel computes
        # neg_lr_scale/decay_factor on-device from lr+wd.
        out["beta1"] = float(g["momentum"])
        out["bc1"] = 1.0 - out["beta1"] ** step
        # The 1D/non-2D weights are a SEPARATE eager AdamW group (muon.py:115-125) with
        # INDEPENDENT hyperparameters: adamw_lr/adamw_betas/adamw_eps. The single
        # FusedScalars lr/beta1/beta2 carry the 2D-group values (0.02/0.95/...), so the
        # 1D AdamW tail (kernel P3, OptId::Muon branch) reads the aux_* fields instead.
        # Find the adamw group (group_type=="adamw", or the group that has `betas` —
        # the 2D group has `momentum` and no `betas`). Forward its lr/betas → aux_*;
        # eps maps to the shared `eps` (above). If there is no 1D group (all-2D model),
        # aux_* keep the eager-default ABI values (harmless — no 1D tensors to apply to).
        aux_g = None
        for pg in optimizer.param_groups:
            if pg.get("group_type") == "adamw" or "betas" in pg:
                aux_g = pg
                break
        if aux_g is not None:
            a_betas = aux_g.get("betas", (0.9, 0.98))
            out["aux_lr"] = float(aux_g.get("lr", 1e-3))
            out["aux_beta1"] = float(a_betas[0])
            out["aux_beta2"] = float(a_betas[1])
            # eps: the 1D AdamW group's adamw_eps. The kernel's 1D tail uses st.eps
            # (the shared `eps` scalar). Bind it from the adamw group so the tail's eps
            # matches eager (both default 1e-8; explicit so a non-default adamw_eps is
            # honored). The 2D NS path does not use eps, so this is safe to override.
            out["eps"] = float(aux_g.get("eps", eps))
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
    # neuralgrok (mamba — wave-2): bf16 TC launcher routes opt_id=6
    # (OptId::NeuralGrok) over the mamba TC-reduced grad; psi pack scattered into the
    # `extra` slice host-side (model-independent). gemm_impl_for_cell → "wgmma".
    ("mamba3", "neuralgrok"),
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
    # grokadamw (vit): CONVERTED (wave-2 vit lane). The SAME 3-mechanism conversion
    # on the vit TC kernel: fused_vit_megakernel_tc now has the IDENTICAL P2.5 global
    # grad-norm clip (deterministic ascending-CTA Σg² → clip_coef) and the P3 per-tensor
    # layer-wise β1=β1·(1-γ)^t + rebased bc1 (t == flat kVitOffsets layer index, cls_token
    # is t=0). The vit launcher already routes opt_id=3 (OptId::GrokAdamW). γ/grad_clip
    # thread via FusedScalars. Validated: single-step state-gate + the MULTI-STEP parity
    # (the clip is inert at step 1 but fires by ~step 50; kernel-with-clip tracks eager,
    # kernel-without reproduces the ~2e-4 divergence).
    ("vit", "grokadamw"),
    # grokadamw (mamba — wave-2): bf16 TC launcher routes opt_id=3 (OptId::GrokAdamW)
    # over the mamba TC-reduced grad; the kernel's P2.5 global grad-norm clip + P3
    # per-tensor layer-wise β1 land in fused_mamba_megakernel_tc (ported from
    # decoder/vit). gemm_impl_for_cell → "wgmma".
    ("mamba3", "grokadamw"),
    # prodigy (decoder): CONVERTED (wave-2 decoder lane). STAGED global-d, computed
    # IN-KERNEL (P2.6, between B2 and P3) — still a SINGLE persistent wgmma launch.
    # dispatch.cpp's wgmma_tail_opt_id("prodigy")=5 routes it onto the TC driver; the
    # decoder launcher (mega_decoder_real_adamw_tc_launcher.cu) binds param_init /
    # prodigy_persist / s_track and dispatches OptId::Prodigy. The d-estimate reduction
    # byte-matches the eager multi-tensor estimator. mamba×prodigy is now wired too
    # (see the ("mamba3","prodigy") entry below).
    ("transformer_decoder", "prodigy"),
    # prodigy (vit): CONVERTED (wave-2 vit lane). The SAME STAGED global-d P2.6 phase
    # on the vit TC kernel; the vit launcher binds param_init/prodigy_persist/s_track
    # and dispatches OptId::Prodigy (opt_id=5). Single persistent wgmma launch.
    ("vit", "prodigy"),
    # NOTE: mamba×prodigy is NOT here — A/A/A determinism FAILS (scheduling-exposed race
    # in the shared mamba forward exposed by prodigy's register pressure; see the
    # _FUSED_L3_REAL block above). The kernel/launcher code is landed-dormant; the cell
    # stays blocked until the racy component is fixed.
    # muon (vit): CONVERTED (wave-2 vit lane). STAGED grid-cooperative Newton-Schulz —
    # in-kernel P2.7 (NS orthogonalization of the 11 2D weights, all CTAs per matrix),
    # 1D weights → AdamW tail in P3. Single persistent wgmma launch (opt_id=7); the vit
    # launcher binds the momentum to the persistent m-slice (st.exp_avg).
    ("vit", "muon"),
    # muon (decoder — wave-2 decoder lane): the SAME in-kernel P2.7 grid-cooperative NS
    # ported onto fused_decoder_megakernel_tc (11 2D weights incl. tok[99,128]/pos[4,128]),
    # 1D weights → AdamW aux tail. dispatch.cpp's wgmma_tail_opt_id("muon")=7 routes it
    # onto the decoder TC driver, the launcher dispatches case OptId::Muon. WITHOUT this
    # entry gemm_impl_for_cell returns "scalar" → the scalar adamw-only decoder path →
    # an Int/Float dtype throw on the token input (the bug this fixes). With it →
    # "wgmma", the real TC path.
    ("transformer_decoder", "muon"),
    # muon (mamba — wave-2 mamba lane): the SAME in-kernel P2.7 grid-cooperative NS ported
    # onto fused_mamba_megakernel_tc (13 ndim==2 weights: mbtc::kMbMuon2D). dispatch.cpp's
    # wgmma_tail_opt_id("muon")=7 routes it onto the mamba TC driver, the launcher dispatches
    # case OptId::Muon; mb_tc_tail is true for muon (NOT sg/prodigy/looksam) so it routes
    # unconditionally. WITHOUT this entry gemm_impl_for_cell returns "scalar" for mamba×muon
    # → the scalar adamw-only mamba path throws on the non-adamw optimizer; WITH it → "wgmma".
    # muon/mamba is a SINGLE forward + NS precompute (NOT a 2nd forward), so it does NOT hit
    # the shared mamba-forward A/A/A race that blocks looksam/prodigy/SG mamba — VERIFIED
    # A/A/A bit-exact by test_l3tc_tail_gate (muon/mamba). Single persistent wgmma launch.
    ("mamba3", "muon"),
    # prodigy/mamba + looksam/mamba: CONVERTED (A/A/A race fixed — see _FUSED_L3_REAL notes
    # above + test_l3tc_tail_gate). gemm_impl_for_cell must return "wgmma" for them so the real
    # mamba TC tail (prodigy P2.6 / looksam P2.4 SAM 2nd backward) routes instead of the scalar
    # adamw-only path. Both PASS A/A/A bit-exact on the production path.
    ("mamba3", "prodigy"),
    ("mamba3", "looksam"),
    # looksam (decoder — SAM-tier lane): CONVERTED. The SAM 2nd backward is the in-kernel
    # P2.4 phase (perturb→2nd fwd+bwd→sam_dir=g_sam−g) on the decoder TC kernel; opt_id=4
    # routes it onto the TC driver (dispatch.cpp wgmma_tail_opt_id("looksam")=4 → case
    # OptId::LookSAM in the decoder launcher). WITHOUT this entry gemm_impl_for_cell returns
    # "scalar" → the scalar adamw-only decoder path → an Int/Float dtype throw on the token
    # input; WITH it → "wgmma", the real TC path. vit×looksam joins (the same in-kernel P2.4
    # phase ported to the vit TC kernel). mamba×looksam follows once the mamba phase lands.
    ("transformer_decoder", "looksam"),
    # looksam (vit — SAM-tier lane): CONVERTED. The IDENTICAL in-kernel P2.4 SAM 2nd backward
    # on the ViT TC kernel (fused_vit_megakernel.cuh); opt_id=4 routes it onto the vit TC driver
    # (wgmma_tail_opt_id("looksam")=4 → case OptId::LookSAM in the vit launcher). WITHOUT this
    # entry gemm_impl_for_cell returns "scalar" → the scalar adamw-only vit path → throws; WITH
    # it → "wgmma", the real TC path. mamba×looksam is BLOCKED (deliberately NOT registered
    # here): its in-kernel P2.4 IS code-landed but the LookSAM mamba instantiation FAILS A/A/A
    # (the same latent shared mamba-forward race as mamba×prodigy — non-deterministic EVEN on
    # SAM-OFF steps, so NOT the SAM code). Absent this registry entry gemm_impl_for_cell returns
    # "scalar" for mamba×looksam, and has_l3_real is False, so fused_train_step declines the L3
    # path and the race runs eager LookSAM — never a non-deterministic wgmma cell. dispatch.cpp's
    # mb_looksam carve-out is the C++-side guard (env SG_MAMBA_LOOKSAM_PROBE to observe the
    # landed-dormant tail). Lift once the mamba forward is fixed by its owner.
    ("vit", "looksam"),
    # supergrok11 (decoder/vit — SAM-tier lane): CONVERTED. The in-kernel SAM 2nd backward
    # (P2.4, sharpness=(g_sam−g)²) + per-tensor meta-net mu/gate precompute (P2.45) route onto
    # the {decoder,vit} TC driver via opt_id=8 (dispatch.cpp wgmma_tail_opt_id("supergrok11")=8
    # → case OptId::SuperGrok11). WITHOUT these entries gemm_impl_for_cell returns "scalar" → the
    # scalar adamw-only path → a dtype/loud throw; WITH them → "wgmma", the real TC path. mamba×
    # supergrok11 is BLOCKED (deliberately NOT registered): the SG SAM 2nd backward re-runs the
    # mamba forward TWICE, exposing the SAME shared mamba-forward A/A/A race as mamba×prodigy/
    # looksam; the mamba megakernel/launcher carry NO SG case (code-absent block). has_l3_real is
    # False for mamba×SG → fused_train_step declines → eager. dispatch.cpp's mb_is_sg carve-out
    # is the C++-side guard. Lift once the mamba forward is fixed by its owner.
    ("transformer_decoder", "supergrok11"),
    ("vit", "supergrok11"),
    # supergrok15 (decoder/vit — SAM-tier lane): CONVERTED. SAME SAM 2nd backward + mu precompute
    # as SG11, host-scalar gate (no per-tensor cosine). opt_id=9 routes onto the TC driver
    # (wgmma_tail_opt_id("supergrok15")=9 → case OptId::SuperGrok15). mamba×supergrok15 is BLOCKED
    # (same shared mamba-forward race; code-absent in the mamba kernel/launcher).
    ("transformer_decoder", "supergrok15"),
    ("vit", "supergrok15"),
    # supergrok2 (decoder/vit — the LAST/hardest cell): CONVERTED via the DEDICATED
    # ops.sg2_fused_step entry. SG2's optimizer phase is the FULL CSA/HCA/PEER/GRU
    # meta-net run AS in-kernel stages (P3-SG2): STAGE -1 in-kernel SEGMENTED SORT
    # (|grad| ascending, index tie-break — strategy A) → S0..S5 (CSA/HCA/GRU/PEER/
    # apply), reading st.sharpness from the SAM 2nd backward (P2.4, shared with SG11/15).
    # The meta-net WEIGHTS come from HBM (the 35 KB bundle does NOT fit smem alongside
    # DecTcSmem/VitSampleSmem under the 48 KB cap) + per-tensor scalar ARRAYS ride the
    # dedicated entry (they don't fit the global FusedScalars). Single-step + 200-step
    # parity vs ops.supergrok2_batched_step (the per-op oracle, SAME low-rank indexer)
    # < 1e-5; A/A/A bit-exact via the index-tie-break sort. HONEST CAVEAT: SG2 reaches
    # L3-TC + single-step parity but WON'T GROK — the CSA lightning-indexer fidelity gap
    # (kernel drops idx_UQ, /sqrt(rank) vs /sqrt(d), diverges for N>64); the gate's N>64
    # CSA fidelity PROBE REPORTS this (not a regression). mamba×supergrok2 is BLOCKED
    # (SAM double-forward + segmented sort re-trip the shared mamba-forward A/A/A race;
    # the mamba kernel/launcher carry no SG2 case — code-absent).
    ("transformer_decoder", "supergrok2"),
    ("vit", "supergrok2"),
    # supergrok11/15 (mamba — wave-4 mamba SG lane): CONVERTED. The bf16 TC mamba launcher
    # routes opt_id 8/9 (OptId::SuperGrok11/15) over the mamba TC-reduced grad; the kernel's
    # P2.4 SAM 2nd backward (sharpness=(g_sam−g)²) + P2.45/P3 per-tensor meta-net mu precompute
    # land in fused_mamba_megakernel_tc (ported from decoder/vit). The shared-mamba-forward
    # A/A/A race that blocked them is fixed (commit 0b57f7e — looksam/mamba proved the SAM
    # double-forward is race-free); the tail gate re-verifies A/A/A. gemm_impl_for_cell →
    # "wgmma". If either trips the race the entry is REMOVED here (landed dormant; the
    # _FUSED_L3_REAL entry would then route to eager via has_l3_real-only-on-membership? no —
    # see the dormant carve below). They reach L3-TC but WON'T GROK (meta-net untrained).
    ("mamba3", "supergrok11"),
    ("mamba3", "supergrok15"),
    # supergrok2 (mamba — the LAST/hardest cell): CONVERTED via the DEDICATED
    # ops.sg2_fused_step → mega_mamba_sg2_tc entry (the FULL CSA/HCA/PEER/GRU meta-net AS the
    # optimizer phase: P3-SG2 in-kernel SEGMENTED SORT (STAGE -1) + SAM 2nd backward + the
    # sg2_meta_stages CSA/HCA/GRU/PEER pipeline, ported onto the mamba megakernel + launcher).
    # The SAM double-forward + segmented sort re-exercise the shared mamba forward; the now-
    # fixed race (0b57f7e) COVERS it — the tail gate confirms A/A/A bit-determinism. Won't GROK
    # (CSA idx_UQ fidelity gap, out of scope). If A/A/A ever re-trips, REMOVE this membership
    # (the kernel/launcher stay landed) — but it is A/A/A-clean as shipped.
    ("mamba3", "supergrok2"),
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
        # State layout is [m|v|extra]+loss = 3*total+1 for every cell EXCEPT prodigy
        # (decoder L3-TC, STAGED global-d), which needs a LARGER buffer carrying its
        # trajectory anchor + persisted estimator scalars:
        #   [m | v | extra/s_track | loss | param_init(total) | r_ema | s_ema | d_lr]
        #   = 4*total + 4. The TC launcher carves param_init at loss_slot+1 and the
        # 3 scalars after it (matches dispatch.cpp's prodigy state-size check). The
        # extras zero-init; param_init is seeded = params at step 1 below (so r=0 at
        # step 1 ⇒ d stays at d0, matching eager). d_lr's zero-init is overridden by
        # the kernel's step-1 d0 cold-start (st.d0), so d_lr starts at d0 like eager.
        # SuperGrok11/15 (meta-net mu + SAM 2nd backward) need a LARGER buffer carrying the
        # PERSISTED sharpness + the phi-net weight pack:
        #   [m | v | mu | loss | sharpness(total) | phi_pack(4H+1)] = 4*total + 1 + kSgPhiPack
        #   (the TC launcher carves sharpness at loss_slot+1 and the phi pack after it,
        #   matching dispatch.cpp's SG state-size check). The extras zero-init; sharpness
        #   stays 0 until the first SAM step writes it (eager-faithful: _flat_sharpness
        #   zero-init). H=32 (the kernel's compile-time kSgPhiHidden).
        _sg_phi_pack = 4 * 32 + 1
        if opt_name == "prodigy":
            _state_floats = 4 * total + 4
        elif opt_name in ("supergrok11", "supergrok15"):
            _state_floats = 4 * total + 1 + _sg_phi_pack
        elif opt_name == "supergrok2":
            # SuperGrok2 (DEDICATED entry) state: [m | v | mu | loss | sharpness(total)
            # | slow(total) | gru_state(total*GH)] = (5+GH)*total + 1, GH=gru_hidden=4.
            # perm/unsort are built IN-KERNEL into the per-CTA workspace (NOT state).
            # The launcher carves sharpness at loss_slot+1, slow after it, gru_state
            # after slow. mu reuses the 3rd slice; m/v the first two. zero-init.
            _SG2_GH = 4
            _state_floats = (5 + _SG2_GH) * total + 1
        else:
            _state_floats = 3 * total + 1
        state = torch.zeros(_state_floats, dtype=torch.float32, device=device)
        grad_out = torch.zeros(total, dtype=torch.float32, device=device)
        names = [n for n, _ in named]
        cache = dict(flat=flat, state=state, grad_out=grad_out, names=names,
                     param_init_seeded=False)
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

    # PRODIGY trajectory anchor (p0): seed param_init = the CURRENT params ONCE, at
    # the FIRST step this cell is driven (mirrors eager prodigy.py's
    # state["param_init"] = p.detach().clone() at the first step). param_init lives at
    # state[3*total+1 : 4*total+1] (right after the loss slot, before the 3 persisted
    # estimator scalars). Seeding from `flat` (which holds the current params) makes
    # r = Σ d²·<g, p0−p> = 0 at step 1 ⇒ d stays at d0 — the eager-faithful cold
    # start. Guarded so step≥2 never overwrites the anchor (the distance from p0 is
    # the whole point of the estimator). The flag persists in the cache across steps.
    if opt_name == "prodigy" and not cache.get("param_init_seeded", False):
        state[3 * total + 1:4 * total + 1].copy_(flat)
        cache["param_init_seeded"] = True

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

    # SuperGrok11/15 SEAM: scatter the FRESH SharpnessMetaNet phi-net weight pack into the
    # state slice the kernel reads (state[4*total+1 : 4*total+1+kSgPhiPack], right after the
    # PERSISTED sharpness buffer at state[3*total+1 : 4*total+1]). The TC launcher binds
    # st.sg_phi_W1/b1/W2 from there (sg_phi_b2 read on-device from sg_phi_W2[H]); the kernel's
    # P2.45 mu precompute reads the host-owned meta-net weights. We re-lay it EVERY step (cheap,
    # 129 floats) so a host-trained meta-net (meta_step) is picked up on the very next launch
    # (the snapshot refresh the megakernel path needs — it cannot run the bilevel autograd).
    # Pack layout matches opt_components.cuh's kSgPhi*Off: [W1(2H) | b1(H) | W2(H) | b2(1)]
    # (H=32). The kernel's sg11_phi_forward reads W1 as [H,2] row-major; the meta-net's
    # Linear(2,H).weight is [H,2] (out=H rows, in=2 cols) → row-major flatten = [j*2],[j*2+1],
    # exactly what the kernel indexes. W2 is the Linear(H,1).weight [1,H] → flatten = [H]. b2 is
    # the scalar Linear(H,1).bias. Guard on the SG instance (meta_net) so a non-SG optimizer
    # (the parity-test stand-ins) never trips it.
    if opt_name in ("supergrok11", "supergrok15"):
        if not hasattr(optimizer, "meta_net"):
            raise RuntimeError(
                "fused_train_step: supergrok11/15 L3 cell requires an optimizer exposing "
                f"meta_net (a SharpnessMetaNet) to fill the kernel's phi pack; got "
                f"{type(optimizer).__name__}.")
        W1, b1, W2, b2, _rescale = optimizer.meta_net.get_weights()
        H = optimizer.meta_net.hidden_dim
        if H != 32:
            raise RuntimeError(
                f"fused_train_step: supergrok11/15 SharpnessMetaNet hidden_dim is {H} but the "
                f"kernel is compiled for kSgPhiHidden=32 (the known prior 64-bug). Build the "
                f"meta-net with meta_hidden_dim=32.")
        pack = torch.empty(4 * H + 1, dtype=torch.float32, device=device)
        pack[0:2 * H].copy_(W1.reshape(-1).to(device=device, dtype=torch.float32))   # W1 [H,2]→row-major
        pack[2 * H:3 * H].copy_(b1.reshape(-1).to(device=device, dtype=torch.float32))
        pack[3 * H:4 * H].copy_(W2.reshape(-1).to(device=device, dtype=torch.float32))  # W2 [1,H]→[H]
        pack[4 * H].copy_(torch.as_tensor(float(b2.reshape(-1)[0]), dtype=torch.float32, device=device))
        phi_off = 4 * total + 1
        if phi_off + pack.numel() > state.numel():
            raise RuntimeError(
                f"fused_train_step: phi pack ({pack.numel()} floats) does not fit "
                f"(state has {state.numel()}, needs >= {phi_off + pack.numel()}).")
        state[phi_off:phi_off + pack.numel()].copy_(pack)

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
    # prodigy decoder L3-TC (STAGED global-d): the kernel's P2.6 d-reduction reads
    # st.d0/st.d_coef/st.beta3. _opt_scalars_from sets these only for prodigy;
    # omitting them leaves the inert FusedScalars defaults (d_coef=1, beta3=0) —
    # which would DROP the persistent EMA and the candidate scale (the d-estimate
    # would be the instantaneous form, diverging from eager at step≥2). Forward them.
    if "d0" in scalars:
        _extra_scalars["d0"] = scalars["d0"]
    if "d_coef" in scalars:
        _extra_scalars["d_coef"] = scalars["d_coef"]
    if "beta3" in scalars:
        _extra_scalars["beta3"] = scalars["beta3"]
    # muon vit L3-TC (STAGED NS): the 1D/non-2D weights take a SEPARATE eager AdamW
    # group with INDEPENDENT lr/betas (adamw_lr/adamw_betas). _opt_scalars_from sets
    # aux_lr/aux_beta1/aux_beta2 only for muon; omitting them leaves the eager-default
    # ABI values and the 1D tail would silently use the 2D-group lr/beta1 (the bug the
    # 2D/1D split diagnostic caught: 1D param rel 19.0). Forward them so the kernel's
    # Muon P3 1D AdamW tail matches the eager non-2D group (it device-computes the
    # aux bias-corrections from aux_beta^step).
    if "aux_lr" in scalars:
        _extra_scalars["aux_lr"] = scalars["aux_lr"]
    if "aux_beta1" in scalars:
        _extra_scalars["aux_beta1"] = scalars["aux_beta1"]
    if "aux_beta2" in scalars:
        _extra_scalars["aux_beta2"] = scalars["aux_beta2"]
    # looksam (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM 2nd backward): the kernel's
    # P2.4 phase reads st.rho + st.looksam_sam. _opt_scalars_from sets these only for
    # looksam; omitting them leaves the inert FusedScalars defaults (rho=0 ⇒ no perturb,
    # looksam_sam=0 ⇒ no 2nd backward), which would DROP the SAM direction entirely (the
    # cell would degenerate to plain AdamW on g). Forward them so the in-kernel perturb→
    # 2nd fwd+bwd→sam_dir=g_sam−g fires on the every-k cadence.
    if "rho" in scalars:
        _extra_scalars["rho"] = scalars["rho"]
    if "looksam_sam" in scalars:
        _extra_scalars["looksam_sam"] = scalars["looksam_sam"]
    # supergrok11/15 (decoder/vit L3-TC, meta-net mu): the kernel's P2.45 mu precompute reads
    # st.sg_rescale (mu = rescale·phi(g, sharpness)); SG15's apply also reads st.gate (the host
    # sigmoid(accuracy)) + alpha_max. _opt_scalars_from sets these only for SG; omitting them
    # leaves the inert FusedScalars defaults (sg_rescale=0 ⇒ mu=0 ⇒ the SG tail degenerates to
    # AdamW on g). Forward them so the live meta-net mechanism executes. (alpha/gate/alpha_max
    # are forwarded by the generic _extra_scalars branches above when present in scalars.)
    if "sg_rescale" in scalars:
        _extra_scalars["sg_rescale"] = scalars["sg_rescale"]
    if "alpha_max" in scalars:
        _extra_scalars["alpha_max"] = scalars["alpha_max"]
    if "gate" in scalars:
        _extra_scalars["gate"] = scalars["gate"]

    # ── SuperGrok2 DEDICATED entry (decoder/vit L3-TC). SG2 routes through
    #    ops.sg2_fused_step (NOT ops.fused_step): its optimizer phase is the FULL
    #    CSA/HCA/PEER/GRU meta-net (in-kernel segmented sort + SAM 2nd backward +
    #    sg2_meta_stages), needing the meta-net weight bundle + per-tensor scalar
    #    ARRAYS the generic ABI cannot carry. We build BOTH here (the per-tensor
    #    scalars in named_parameters() == flat layout order) and call the dedicated
    #    binding; everything else (param pack-in, scatter-back, loss read) is shared.
    if opt_name == "supergrok2":
        if not hasattr(optimizer, "meta_net"):
            raise RuntimeError(
                "fused_train_step: supergrok2 L3 cell requires an optimizer exposing "
                f"meta_net (a CSAHCAMetaNet); got {type(optimizer).__name__}.")
        net = optimizer.meta_net
        GH = int(getattr(net, "gru_hidden", 4))
        _SG2_GH = 4
        if GH != _SG2_GH:
            raise RuntimeError(
                f"fused_train_step: supergrok2 meta-net gru_hidden is {GH} but the "
                f"kernel is compiled for SG2Dims gru_hidden={_SG2_GH}.")
        # Meta-net weight bundle, mapped to the kernel ABI keys (get_weights() uses
        # gru_W_z; the kernel takes gru_Wz). peer_queries/product_keys are per-head
        # lists → stack. expert_W1/W2 → [E, hidden]. fp32 + contiguous.
        w = net.get_weights(None)
        ne = int(net.num_experts)
        def _f32c(t):
            t = t if t.dtype == torch.float32 else t.float()
            return t.contiguous().to(device=device)
        peer_Ws = _f32c(torch.stack([q.float().contiguous() for q in w['peer_queries']]))
        pkA = _f32c(torch.stack([k.float().contiguous() for k in w['product_keys_A']]))
        pkB = _f32c(torch.stack([k.float().contiguous() for k in w['product_keys_B']]))
        # Per-tensor scalar arrays (length P = #tensors), in named_parameters() order.
        # Formulas mirror supergrok2.py step_compiled (:2258-2280): alpha_i =
        # clamp(base_alpha·layer_alpha[i],0,1); beta1_i = layer_beta1[i]; gru_decay==
        # beta1; lamb_eff = lamb·ramp·gate_signal (uniform); bc1_i=1-beta1_i^step,
        # bc2_i=1-beta2^step. The optimizer builds _flat_layer_{alphas,beta1s} from its
        # param_groups (== model.parameters() == named order). If a SG2 run ever
        # reorders, the (n,off) layout would drift — guarded by the count check.
        P = len(named)
        import math as _m
        beta1_g = float(optimizer.param_groups[0].get("betas", (0.9, 0.999))[0])
        beta2_g = float(optimizer.param_groups[0].get("betas", (0.9, 0.999))[1])
        base_alpha = float(getattr(optimizer, "_cached_alpha",
                                   getattr(optimizer, "alpha_init", 0.98)))
        try:
            ramp = float(optimizer._get_ramp_factor())
        except Exception:
            ramp = 1.0
        try:
            gate_signal = float(optimizer._get_gate_signal())
        except Exception:
            gate_signal = 1.0
        lamb = float(getattr(optimizer, "lamb", 2.0))
        lamb_eff = lamb * ramp * gate_signal
        la = getattr(optimizer, "_flat_layer_alphas", None)
        lb = getattr(optimizer, "_flat_layer_beta1s", None)
        # FAIL LOUD if the optimizer's per-layer tables don't cover all P tensors:
        # silently falling back to alpha=1.0/beta1_g for a missing tail would diverge
        # from the eager per-tensor scalars WITHOUT an error (the per-op oracle would
        # IndexError instead). In the wired path the optimizer is built over
        # model.parameters() so len == P; a mismatch is a real misconfiguration.
        if la is not None and len(la) != P:
            raise RuntimeError(
                f"fused_train_step: supergrok2 _flat_layer_alphas has {len(la)} "
                f"entries but the model has {P} trainable tensors — the optimizer "
                f"must wrap the SAME params the L3 layout enumerates.")
        if lb is not None and len(lb) != P:
            raise RuntimeError(
                f"fused_train_step: supergrok2 _flat_layer_beta1s has {len(lb)} "
                f"entries but the model has {P} trainable tensors.")
        alpha_a = []; gd_a = []; le_a = []; b1_a = []; bc1_a = []; bc2_a = []
        st_step = int(scalars["step"])
        for i in range(P):
            la_i = float(la[i]) if (la is not None and i < len(la)) else 1.0
            b1_i = float(lb[i]) if (lb is not None and i < len(lb)) else beta1_g
            a_i = max(0.0, min(1.0, base_alpha * la_i))
            alpha_a.append(a_i); gd_a.append(b1_i); le_a.append(lamb_eff)
            b1_a.append(b1_i)
            bc1_a.append(1.0 - b1_i ** st_step)
            bc2_a.append(1.0 - beta2_g ** st_step)
        def _arr(v):
            return torch.tensor(v, dtype=torch.float32, device=device).contiguous()
        rescale = float(getattr(net, "rescale", 0.1))
        ops.sg2_fused_step(
            model_c, flat, packed, grad_out, state,
            _f32c(w['input_proj_W']), _f32c(w['input_proj_b']),
            _f32c(w['csa_q_W']), _f32c(w['csa_k_W']), _f32c(w['csa_v_W']), _f32c(w['csa_out_W']),
            _f32c(w['csa_compress_w']), _f32c(w['csa_idx_DQ']), _f32c(w['csa_idx_K']),
            _f32c(w['hca_q_W']), _f32c(w['hca_k_W']), _f32c(w['hca_v_W']), _f32c(w['hca_out_W']),
            _f32c(w['gru_W_z']), _f32c(w['gru_b_z']), _f32c(w['gru_W_r']), _f32c(w['gru_b_r']),
            _f32c(w['gru_W_h']), _f32c(w['gru_b_h']),
            peer_Ws, pkA, pkB,
            _f32c(w['expert_W1'].reshape(ne, -1)), _f32c(w['expert_b1']),
            _f32c(w['expert_W2'].reshape(ne, -1)), _f32c(w['expert_b2'].reshape(-1)),
            _arr(alpha_a), _arr(gd_a), _arr(le_a), _arr(b1_a), _arr(bc1_a), _arr(bc2_a),
            rescale, beta2_g, float(scalars["lr"]),
            float(scalars["weight_decay"]), float(scalars["eps"]),
            float(scalars.get("rho", 0.05)), float(scalars.get("looksam_sam", 0.0)),
            int(scalars["step"]))
    else:
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
