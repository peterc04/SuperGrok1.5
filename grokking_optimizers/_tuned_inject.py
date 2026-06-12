"""Autotuner → product-build linkage: per-TU tuned-flag injection.

This module is the single source of truth for lever (b): making the
``compile.py`` autotuner's winning launch parameters ACTIVE in the shipped
``grokking_optimizers/_ops*.so`` instead of decorative.

Producer  : ``grokking_optimizers.compile.build_jit`` calls
            :func:`export_winner` the moment a JIT winner is decided,
            persisting per-(arch, model, optimizer) winners to the canonical
            JSON at ``grokking_optimizers/_kernel_tuned.json``.
Consumer  : ``setup.py``'s ``BuildExtension`` subclass reads that JSON and,
            per CUDA translation unit, appends ``-DSG_TUNED_*`` macros (and
            ``--maxrregcount`` when nonzero) to the nvcc flags of the TUs
            that belong to the matching optimizer — see
            :func:`source_extra_nvcc_flags` and :func:`compute_source_flags`.

SCHEMA (#12) — the JSON is keyed ``{arch: {model: {optimizer: combo}}}`` (the
canonical short ``model`` token), so tuning the same optimizer on two models no
longer last-wins; each ``mega_<model>_<opt>.cu`` cell gets ITS model's winner
(:func:`model_for_source`), shared ``launch_<opt>.cu`` launchers fall back to a
deterministic model's winner. The OLD flat ``{arch: {optimizer: combo}}`` shape
is still READ (backward-compat, :func:`_lookup_combo`) and MIGRATED in place on
the next :func:`export_winner` write, so a pre-#12 JSON keeps working.

IMPORT DISCIPLINE — this module is *pure stdlib* and MUST stay that way.
It is loaded by ``setup.py`` (which already imports torch, but should not
re-import the heavy package) and by ``tuning/test_build_injection.py``
(CPU-only, no torch / no GPU). Both load it by FILE PATH so that importing
it never triggers ``grokking_optimizers/__init__.py`` (which does
``import torch`` + ``get_ops()`` and would probe CUDA). Do NOT add any
``grokking_optimizers``-relative imports or third-party imports here.

SCOPE — the five canonical SAFE per-TU dims (block / vec / unroll /
async_depth as ``-DSG_TUNED_*`` macros + maxrregcount as a real
``--maxrregcount=N`` ptxas flag) are hardcoded in :data:`MACROS` as the
always-known floor. Review fix P0.2: the exporter ALSO persists EVERY OTHER
winner dim that carries a macro in the live search space (cluster_shape, and
any feature macro a kernel begins consuming), together with the dim→macro map
the producer derived. The consumer (:func:`source_extra_nvcc_flags`) reads that
persisted ``_macros`` map so it can emit ``-D<macro>=<v>`` for dims it has no
hardcoded entry for — WITHOUT importing compile.py (this module stays pure
stdlib). Tuple dims (cluster_shape) emit nvcc-safe per-axis scalar macros
``<MACRO>_0/_1/_2`` plus ``<MACRO>_VOLUME`` exactly as compile.py's
``resolve_macros`` does (a single ``-Dfoo=2,1,1`` is rejected by the -D lexer).
An unknown macro (no map entry, not in MACROS) is IGNORED with a one-line
warning so a stale setup.py degrades gracefully (task P0.2).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Canonical paths.
# --------------------------------------------------------------------------
# This file lives at grokking_optimizers/_tuned_inject.py; the JSON sits next
# to it at grokking_optimizers/_kernel_tuned.json, and the repo root is the
# parent of the grokking_optimizers package directory.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)
KERNEL_TUNED_JSON = os.path.join(_PKG_DIR, "_kernel_tuned.json")

# --------------------------------------------------------------------------
# Arch-key canonicalization.
# --------------------------------------------------------------------------
# compile.py's ARCH_TABLE maps the user-facing "sm_90" to the internal
# "sm_90a" (the architecture-specific Hopper target). The JSON and every
# consumer here use the SHORT, user-facing key ("sm_90") so the exporter and
# the setup.py reader agree on one string. canonical_arch_key() folds the
# "a"/"f" suffix variants back to the short form.
def canonical_arch_key(arch: str) -> str:
    """Fold an internal arch string (e.g. ``"sm_90a"``) to its short JSON key.

    ``"sm_90a" -> "sm_90"``, ``"sm_120f" -> "sm_120"``, ``"gfx942" -> "gfx942"``,
    ``"sm_90" -> "sm_90"``. Unknown strings pass through unchanged.
    """
    a = str(arch).strip()
    if a.startswith("sm_"):
        # strip a trailing single arch-letter suffix ("a"/"f") if present.
        if a and a[-1].isalpha() and a[:-1].rsplit("_", 1)[-1].isdigit():
            return a[:-1]
    return a


# --------------------------------------------------------------------------
# SOURCE OF TRUTH — the five SAFE per-TU dims.
#
# Each entry maps a winner-combo key (the search-space dim name used by
# compile.py, e.g. "block") to how it is emitted on the compiler command
# line. There are two kinds:
#   kind="macro": emitted as ``-D<macro>=<value>`` (an nvcc preprocessor
#                 define). The macro name MUST match the ``#ifndef`` guard
#                 in the kernel header so the override actually takes — see
#                 grokking_optimizers/kernels/sm_90/adamw_sm90.cuh lines
#                 31-41 (SG_TUNED_BLOCK_SIZE / VEC_WIDTH / UNROLL /
#                 ASYNC_DEPTH; the other per-opt headers inline the same
#                 four guards).
#   kind="flag": emitted as a real ptxas/codegen flag, NOT a -D macro.
#                ``maxrregcount`` -> ``--maxrregcount=<n>`` (CUDA). A value
#                of 0 means "unset" and emits NOTHING (the historical
#                default — nvcc's own register allocator decides).
#
# ``defaults`` records the in-header #ifndef default for the macro dims, so a
# CPU-only unit test can assert "no drift" between this table and the header
# without a GPU. Keep this dict in lockstep with adamw_sm90.cuh:31-41.
# --------------------------------------------------------------------------
MACROS: Dict[str, Dict[str, Any]] = {
    "block":       {"kind": "macro", "macro": "SG_TUNED_BLOCK_SIZE", "default": 256},
    "vec":         {"kind": "macro", "macro": "SG_TUNED_VEC_WIDTH",  "default": 4},
    "unroll":      {"kind": "macro", "macro": "SG_TUNED_UNROLL",     "default": 1},
    "async_depth": {"kind": "macro", "macro": "SG_TUNED_ASYNC_DEPTH", "default": 2},
    # Real ptxas flag (no preprocessor macro). 0 == unset == emit nothing.
    "maxrregcount": {"kind": "flag", "flag": "--maxrregcount", "default": 0},
}

# Order in which to emit flags for a TU (stable output → deterministic
# build.ninja diffs and a predictable unit-test expectation).
_EMIT_ORDER = ["block", "vec", "unroll", "async_depth", "maxrregcount"]

# The canonical optimizer tokens (mirrors
# grokking_optimizers.dispatch.OPTIMIZERS). Used to validate filename→opt
# parses. Kept as a literal so this stdlib-only module never imports the
# heavier dispatch module (which pulls torch).
OPTIMIZERS: Tuple[str, ...] = (
    "adamw", "grokadamw", "grokfast", "lion", "looksam", "muon",
    "neuralgrok", "prodigy", "supergrok11", "supergrok15", "supergrok2",
)

# #12 per-model schema — short model keys used in the nested JSON, and the
# filename-token → short-key map for the megakernel cell TUs. The cell TUs are
# named mega_<model>_<opt>.cu where <model> is the CANONICAL token
# (transformer_decoder / mamba3 / vit); the JSON keys are the SHORT names
# (decoder / mamba / vit) the CLI + dispatch.short_model_name use. Kept as a
# literal so this stdlib-only module never imports dispatch.
MODELS: Tuple[str, ...] = ("decoder", "vit", "mamba")
_MODEL_TOKEN_TO_SHORT: Dict[str, str] = {
    "decoder": "decoder", "transformer_decoder": "decoder",
    "vit": "vit",
    "mamba": "mamba", "mamba3": "mamba",
}


# --------------------------------------------------------------------------
# Drift guard (design item 1c): verify the MACROS table against the kernel
# header's #ifndef defaults. CPU-only, grep-free at build time — the table
# carries its own expected defaults and a unit test (and an optional
# best-effort build-time check) compares them to the committed header.
# --------------------------------------------------------------------------
def header_default_macros() -> Dict[str, int]:
    """The macro→default mapping this module asserts the headers ship.

    A drift between this and the actual ``#ifndef`` guard in
    ``adamw_sm90.cuh`` would mean we emit a ``-D`` for a macro the kernel no
    longer reads (silent no-op) or miss one it added. The unit test compares
    this against a parse of the real header.
    """
    return {spec["macro"]: spec["default"]
            for spec in MACROS.values() if spec["kind"] == "macro"}


# --------------------------------------------------------------------------
# TU → optimizer mapping (the ambiguous-TU policy).
#
# A CUDA TU gets per-optimizer tuned flags ONLY when its filename encodes an
# optimizer by the repo's naming convention:
#   * per-optimizer launcher:  csrc/backends/cuda/sm_90/launch_<opt>.cu
#       (the matching kernels/sm_90/<opt>_sm90.cuh is HEADER-INCLUDED by this
#        TU, so the macros applied here reach the kernel body)
#   * megakernel cell:         csrc/fused/sm_90/mega_<model>_<opt>.cu
#
# EVERYTHING ELSE gets NO per-opt flags (returns None):
#   * bindings (bindings.cpp, dispatch.cpp) — not even CUDA
#   * model-only TUs (models/<model>.cu, mma.cuh, primitives.cuh, ...)
#   * common/utils TUs
# This is the deliberate "ambiguous TU ⇒ defaults" behavior: a TU that does
# not name exactly one optimizer keeps the in-header defaults.
# --------------------------------------------------------------------------
def optimizer_for_source(path: str) -> Optional[str]:
    """Return the optimizer a CUDA TU belongs to, or ``None`` if ambiguous.

    Matches ``launch_<opt>.cu`` and ``mega_<model>_<opt>.cu`` by basename;
    the ``<opt>`` token must be one of :data:`OPTIMIZERS`. Greedy on the
    optimizer suffix so ``mega_transformer_decoder_supergrok2.cu`` resolves
    to ``supergrok2`` (not a shorter prefix). Non-``.cu`` paths and any TU
    that does not encode exactly one known optimizer return ``None``.
    """
    base = os.path.basename(str(path))
    # Only CUDA TUs carry nvcc flags. (.cpp bindings, .py launchers: never.)
    if not base.endswith(".cu"):
        return None
    stem = base[:-len(".cu")]

    if stem.startswith("launch_"):
        cand = stem[len("launch_"):]
        return cand if cand in OPTIMIZERS else None

    if stem.startswith("mega_"):
        # mega_<model>_<opt>: the optimizer is the LAST underscore-group that
        # forms a known optimizer token. Test the longest known-opt suffix.
        rest = stem[len("mega_"):]
        for opt in sorted(OPTIMIZERS, key=len, reverse=True):
            if rest == opt or rest.endswith("_" + opt):
                return opt
        return None

    # models/<model>.cu, primitives.cuh, mma.cuh, etc. — no single optimizer.
    return None


def model_for_source(path: str) -> Optional[str]:
    """Return the SHORT model token a megakernel-cell TU belongs to, or ``None``.

    #12 per-model schema: the INJECTED cells are named ``mega_<model>_<opt>.cu``
    where ``<model>`` is the canonical token (``transformer_decoder`` /
    ``mamba3`` / ``vit``); this returns the SHORT key (``decoder`` / ``mamba`` /
    ``vit``) used in the nested JSON so the consumer can select that cell's
    per-model winner. Returns ``None`` for a per-optimizer LAUNCHER TU
    (``launch_<opt>.cu`` is shared across models — it has no single model) and
    for any TU that does not encode exactly one known optimizer — including the
    standalone real-TC cells ``mega_<model>_real_adamw_tc.cu`` (whose suffix is
    ``_real_adamw_tc``, not a bare optimizer, so :func:`optimizer_for_source`
    already returns ``None``; those are NOT ninja-injection targets — they get
    their tuned tile macros from compile.py's real-TC-step timer build, not from
    the per-TU ninja pass). The ``<model>`` is the prefix between ``mega_`` and
    the trailing ``_<opt>`` token, matched longest-first against the known model
    tokens so ``transformer_decoder`` wins over a shorter prefix."""
    base = os.path.basename(str(path))
    if not base.endswith(".cu"):
        return None
    stem = base[:-len(".cu")]
    if not stem.startswith("mega_"):
        return None
    opt = optimizer_for_source(path)
    if opt is None:
        return None
    rest = stem[len("mega_"):]
    # strip the trailing optimizer token to leave the model token(s).
    if rest.endswith("_" + opt):
        model_tok = rest[: -(len("_" + opt))]
    else:  # rest == opt (no model component) — shouldn't reach here for a cell.
        return None
    for known in sorted(_MODEL_TOKEN_TO_SHORT, key=len, reverse=True):
        if model_tok == known or model_tok.startswith(known + "_"):
            return _MODEL_TOKEN_TO_SHORT[known]
    return None


# --------------------------------------------------------------------------
# Flag computation.
# --------------------------------------------------------------------------
# Warn at most once per (arch, dim) about an unmapped macro dim, so a stale
# setup.py degrades gracefully (P0.2) without spamming the build log.
_WARNED_UNMAPPED: set = set()

# Arch-level bookkeeping keys that live alongside the model/optimizer sub-blocks
# and must NOT be mistaken for a model name during shape detection.
_ARCH_RESERVED_KEYS = frozenset({"_macros", "_meta"})


def _arch_block_is_nested(arch_block: Dict[str, Any]) -> bool:
    """#12 — distinguish the NEW per-model nested shape
    ``{model: {optimizer: combo}}`` from the OLD flat shape
    ``{optimizer: combo}`` for ONE arch block.

    Nested iff some non-reserved key maps to a dict that itself contains at
    least one known-OPTIMIZER key whose value is a dict (a model→opt→combo
    nesting). The old flat shape keys directly by optimizer, whose combo dict's
    values are scalars/lists (never an optimizer→dict nesting), so it returns
    False. An empty / malformed block is treated as flat (the safe default —
    the flat reader simply finds nothing)."""
    for k, v in arch_block.items():
        if k in _ARCH_RESERVED_KEYS:
            continue
        if isinstance(v, dict) and any(
                opt in v and isinstance(v[opt], dict) for opt in OPTIMIZERS):
            return True
    return False


def _lookup_combo(arch_block: Dict[str, Any], optimizer: str,
                  model: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve the winner combo for ``optimizer`` under ``arch_block``, reading
    BOTH the new per-model nested shape and the old flat shape (#12).

    * NEW nested ``{model: {optimizer: combo}}``:
        - ``model`` given  → ``arch_block[model][optimizer]``.
        - ``model`` None   → first model sub-block (deterministic: the JSON's
          sorted model keys) that records this optimizer. This is the
          shared-launcher-TU case (``launch_<opt>.cu`` spans all models): it
          keeps the historical "one winner applies to every TU of this opt"
          semantics, now picking a stable model's winner.
    * OLD flat ``{optimizer: combo}``: ``arch_block[optimizer]`` (``model`` is
      ignored — the old schema had no model dimension).

    Returns the combo dict or ``None`` (absent / wrong shape)."""
    if _arch_block_is_nested(arch_block):
        if model is not None:
            sub = arch_block.get(model)
            if isinstance(sub, dict):
                combo = sub.get(optimizer)
                return combo if isinstance(combo, dict) else None
            return None
        # No model context (shared launcher TU): pick a deterministic model
        # sub-block that has this optimizer.
        for mkey in sorted(k for k in arch_block
                           if k not in _ARCH_RESERVED_KEYS):
            sub = arch_block.get(mkey)
            if isinstance(sub, dict) and isinstance(sub.get(optimizer), dict):
                return sub[optimizer]
        return None
    # Old flat shape.
    combo = arch_block.get(optimizer)
    return combo if isinstance(combo, dict) else None


def source_extra_nvcc_flags(optimizer: Optional[str],
                            tuned: Optional[Dict[str, Any]],
                            arch_key: str,
                            model: Optional[str] = None) -> List[str]:
    """Extra nvcc flags for ONE optimizer's TUs under ``arch_key``.

    Returns ``[]`` when there is nothing tuned for this (arch, optimizer)
    — i.e. when ``optimizer is None`` (ambiguous TU), ``tuned`` is empty/None
    (no JSON), the arch is absent, or the optimizer has no recorded winner.
    Same value is applied to every TU of the same optimizer (design 1b).

    #12 — ``model`` (the SHORT model token for a ``mega_<model>_<opt>.cu`` cell,
    via :func:`model_for_source`) selects that cell's per-model winner from the
    NEW nested ``{arch: {model: {optimizer: ...}}}`` schema. ``None`` (a shared
    ``launch_<opt>.cu`` launcher, or any caller that doesn't know the model)
    falls back to a deterministic model's winner. The OLD flat
    ``{arch: {optimizer: ...}}`` schema is still read (``model`` ignored) so a
    pre-#12 JSON keeps working unchanged. See :func:`_lookup_combo`.

    Deterministic order: the five hardcoded SAFE dims first (``_EMIT_ORDER``),
    then any EXTRA persisted macro dims (P0.2) in sorted order. Scalar macro
    dims become ``-D<macro>=<v>``; tuple dims (cluster_shape) become per-axis
    ``-D<macro>_0/_1/_2`` + ``-D<macro>_VOLUME`` (nvcc rejects ``-Dfoo=2,1,1``);
    ``maxrregcount`` becomes ``--maxrregcount=<n>`` only when ``n > 0``. The
    extra dims' macro names come from the persisted ``_macros`` map; an
    UNKNOWN dim (not SAFE and not in the map) is ignored with a one-line
    warning (graceful degradation for a stale reader).
    """
    if optimizer is None or not tuned:
        return []
    canon = canonical_arch_key(arch_key)
    arch_block = tuned.get(canon)
    if not isinstance(arch_block, dict):
        return []
    combo = _lookup_combo(arch_block, optimizer, model)
    if not isinstance(combo, dict):
        return []
    # P0.2 — producer-persisted dim→macro map for the extra (non-SAFE) dims.
    macro_map = arch_block.get("_macros")
    if not isinstance(macro_map, dict):
        macro_map = {}

    flags: List[str] = []
    # (1) hardcoded SAFE floor, in canonical emit order.
    for key in _EMIT_ORDER:
        if key not in combo:
            continue
        spec = MACROS.get(key)
        if spec is None:
            continue
        value = combo[key]
        if spec["kind"] == "macro":
            flags.append(f"-D{spec['macro']}={_macro_value(value)}")
        elif spec["kind"] == "flag":
            # maxrregcount: 0/None/unset => emit nothing (nvcc default alloc).
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            if n > 0:
                flags.append(f"{spec['flag']}={n}")
    # (2) extra persisted macro dims (cluster_shape + any consumed feature
    # macro), sorted for deterministic build.ninja output. Skip the SAFE dims
    # (already emitted), the model provenance, and any internal bookkeeping.
    for key in sorted(combo.keys()):
        if key in MACROS or key in _NON_DIM_KEYS:
            continue
        value = combo[key]
        macro = macro_map.get(key)
        if not macro:
            # Unknown dim with no mapping — a stale setup.py can't know its
            # macro name. Ignore it (graceful degradation) but say so once.
            wk = (canon, key)
            if wk not in _WARNED_UNMAPPED:
                _WARNED_UNMAPPED.add(wk)
                import sys as _sys
                _sys.stderr.write(
                    f"  [tuned-inject] WARNING: winner dim {key!r} for "
                    f"{optimizer}/{canon} has no macro mapping in the JSON's "
                    f"_macros table; skipping it (the kernel keeps its in-header "
                    f"default). Re-export with a current compile.py to map it.\n")
            continue
        flags.extend(_emit_macro_flags(macro, value))
    return flags


def _emit_macro_flags(macro: str, value: Any) -> List[str]:
    """Emit nvcc-safe ``-D`` flags for one macro dim's value.

    Mirrors compile.py's ``resolve_macros``: a tuple/list value (cluster_shape)
    becomes per-axis ``-D<macro>_0/_1/_2`` plus ``-D<macro>_VOLUME=<product>``
    (a single ``-Dfoo=2,1,1`` is rejected by the -D lexer); a scalar becomes
    ``-D<macro>=<v>``."""
    if isinstance(value, (tuple, list)):
        out: List[str] = []
        comps = list(value)
        for i, comp in enumerate(comps):
            out.append(f"-D{macro}_{i}={_macro_value(comp)}")
        vol = 1
        for comp in comps:
            try:
                vol *= int(comp)
            except (TypeError, ValueError):
                vol = 0
                break
        out.append(f"-D{macro}_VOLUME={vol}")
        return out
    return [f"-D{macro}={_macro_value(value)}"]


def _macro_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def compute_source_flags(sources: List[str],
                         tuned: Optional[Dict[str, Any]],
                         arch_key: str) -> Dict[str, List[str]]:
    """Map each source path to its extra nvcc flag list (only nonempty ones).

    The product of the whole module: feed it the extension's ``sources`` and
    the loaded JSON; it returns ``{abspath_or_path: [flags...]}`` for exactly
    the TUs that resolve to a tuned optimizer. Sources with no per-opt flags
    are OMITTED from the dict (callers treat "absent" as "stock flags").
    Keys preserve the input path string (callers match against the same
    ``sources`` they passed). A per-(optimizer, model) flag list is computed
    once and shared across TUs with the same (optimizer, model).

    #12 — each ``mega_<model>_<opt>.cu`` cell is keyed by its OWN model
    (:func:`model_for_source`) so it picks up THAT model's per-model winner from
    the nested schema; shared ``launch_<opt>.cu`` launchers carry ``model=None``
    and get a deterministic model's winner. Under the OLD flat schema the model
    is ignored, so behaviour is identical to before.
    """
    out: Dict[str, List[str]] = {}
    cache: Dict[Tuple[Optional[str], Optional[str]], List[str]] = {}
    for src in sources:
        opt = optimizer_for_source(src)
        if opt is None:
            continue
        model = model_for_source(src)   # None for shared launcher TUs
        ck = (opt, model)
        if ck not in cache:
            cache[ck] = source_extra_nvcc_flags(opt, tuned, arch_key, model)
        flags = cache[ck]
        if flags:
            out[src] = list(flags)
    return out


# --------------------------------------------------------------------------
# JSON load (consumer side — setup.py).
# --------------------------------------------------------------------------
def load_tuned(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the canonical tuned JSON. ``None`` if absent or unreadable.

    A missing / malformed JSON is NOT an error — it means "no winners yet",
    and the build proceeds with in-header defaults (graceful degradation,
    design 1b). The caller logs a one-line notice in that case.
    """
    p = path or KERNEL_TUNED_JSON
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# JSON export (producer side — compile.py build_jit).
# --------------------------------------------------------------------------
# The five SAFE per-TU dims are ALWAYS persisted (the MACROS floor). Review fix
# P0.2: EVERY other winner dim that carries a macro in the live search space is
# persisted too (so a winning cluster_shape / feature-macro value actually
# reaches the product build instead of being silently dropped). Bookkeeping keys
# (timing_ms, config_key, stage_won, fast-math markers, ...) are still dropped —
# the JSON is a STABLE handoff contract, not a dump of the trial record.
_PERSIST_KEYS = tuple(MACROS.keys())  # block, vec, unroll, async_depth, maxrregcount

# Combo keys that are autotuner bookkeeping, never a tunable macro dim — always
# dropped from the persisted payload regardless of the macro map.
_NON_DIM_KEYS = frozenset({
    "timing_ms", "config_key", "stage_won", "_fastmath", "origin", "status",
    "error", "value_ms", "trial_num", "stage", "host", "model",
})


def _coerce_scalar(value: Any) -> Optional[Any]:
    """Coerce a winner value to a JSON-stable scalar, or None to skip."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, str)):
        return value
    return None


def _winner_payload(combo: Dict[str, Any],
                    macro_map: Optional[Dict[str, str]] = None
                    ) -> Dict[str, Any]:
    """Project a winner combo down to the persisted dims.

    P0.2: persists (a) the five hardcoded SAFE dims (always), and (b) EVERY
    extra combo key present in ``macro_map`` (the producer-derived dim→macro
    map from the live search space). Scalars persist as-is; tuple/list dims
    (cluster_shape) persist as a list so the consumer can emit per-axis macros.
    Keys that are neither a SAFE dim nor in the macro map are dropped (they are
    bookkeeping or non-macro flags)."""
    payload: Dict[str, Any] = {}
    # (a) hardcoded SAFE floor.
    for key in _PERSIST_KEYS:
        if key not in combo:
            continue
        value = combo[key]
        if MACROS[key]["kind"] == "flag":
            try:
                payload[key] = int(value)  # maxrregcount (0 == unset)
            except (TypeError, ValueError):
                continue
        else:
            sv = _coerce_scalar(value)
            if sv is not None:
                payload[key] = sv
    # (b) every extra macro-carrying dim the producer derived from the space.
    for key, _macro in (macro_map or {}).items():
        if key in payload or key in _NON_DIM_KEYS or key not in combo:
            continue
        value = combo[key]
        if isinstance(value, (tuple, list)):
            # Persist tuple dims (e.g. cluster_shape) as a list of ints/scalars;
            # the consumer emits FOO_0/_1/_2 + FOO_VOLUME (nvcc-safe).
            payload[key] = [_coerce_scalar(v) for v in value
                            if _coerce_scalar(v) is not None]
        else:
            sv = _coerce_scalar(value)
            if sv is not None:
                payload[key] = sv
    return payload


def export_winner(optimizer: str,
                 model: str,
                 arch: str,
                 combo: Dict[str, Any],
                 *,
                 path: Optional[str] = None,
                 source: str = "compile.build_jit",
                 version_hash: Optional[str] = None,
                 macro_map: Optional[Dict[str, str]] = None) -> bool:
    """Persist ONE (arch, model, optimizer) winner to the canonical JSON.

    READ-MERGE-WRITE (design 1a): ``build_jit`` runs a single (model,
    optimizer), so this merges into any existing JSON rather than overwriting —
    tuning every (model, optimizer) in sequence accumulates one entry each. The
    write is atomic (tmp file + ``os.replace``) so a reader never sees a
    half-written file.

    #12 PER-MODEL SCHEMA: the winner is keyed by ``{arch: {model: {optimizer:
    combo}}}`` (the canonical short ``model`` token), so tuning the same
    optimizer on a SECOND model NO LONGER clobbers the first — each (model,
    optimizer) is its own slot. An existing OLD-flat arch block
    (``{optimizer: combo}``, pre-#12) is MIGRATED in place on the first write:
    each flat ``optimizer`` entry is moved under its recorded ``model``
    provenance (defaulting to this call's ``model`` if absent), then the new
    winner is merged into the nested tree. The consumer reads both shapes
    (:func:`_lookup_combo`), so a half-migrated or still-flat JSON stays valid.

    ``macro_map`` (P0.2) is the producer-derived ``{dim_name: SG_TUNED_MACRO}``
    map for THIS arch's live search space. The extra (non-SAFE) dims it names
    are persisted in the winner, and the map itself is recorded in the arch
    block under ``_macros`` so the stdlib-only consumer
    (:func:`source_extra_nvcc_flags`) can emit ``-D<macro>=<v>`` for them
    WITHOUT importing compile.py. When ``None`` (legacy callers), behaviour is
    the old SAFE-5-only export.

    Returns ``True`` on success, ``False`` on any failure. This function
    MUST NEVER raise into the caller — a failed persist is a loud warning,
    never a failed tuning run (design 1a). The caller is responsible for the
    warning text; this returns False and leaves the existing JSON intact.

    NOTE on concurrency: atomic rename makes each write self-consistent, but
    naive read-merge-write loses entries if multiple per-optimizer PROCESSES
    race. The runbook directs operators to tune optimizers SEQUENTIALLY (or
    funnel winners through one process); see AUTOTUNE_LINKAGE.md.
    """
    try:
        target = path or KERNEL_TUNED_JSON
        arch_key = canonical_arch_key(arch)
        payload = _winner_payload(combo, macro_map)

        # Read-merge: start from any existing JSON so sibling (model, optimizer)
        # winners survive. A corrupt existing file is discarded (we cannot
        # safely merge into garbage) but reported via the return value path.
        existing = load_tuned(target) or {}
        if not isinstance(existing, dict):
            existing = {}
        arch_block = existing.get(arch_key)
        if not isinstance(arch_block, dict):
            arch_block = {}
        # #12 — migrate an old-flat arch block ({optimizer: combo}) to the
        # nested per-model shape ({model: {optimizer: combo}}) before merging, so
        # a pre-#12 JSON is upgraded in place instead of mixing shapes. Reserved
        # keys (_macros) are left at the arch level.
        if not _arch_block_is_nested(arch_block):
            migrated: Dict[str, Any] = {}
            for k in list(arch_block.keys()):
                if k in _ARCH_RESERVED_KEYS:
                    migrated[k] = arch_block[k]
                    continue
                old_entry = arch_block[k]
                if k in OPTIMIZERS and isinstance(old_entry, dict):
                    # provenance model recorded inside the flat entry, else this
                    # call's model (best available).
                    prov = old_entry.get("model") or model
                    migrated.setdefault(prov, {})[k] = old_entry
                else:
                    # Unknown/foreign key — preserve verbatim so we never drop
                    # data we don't understand.
                    migrated[k] = old_entry
            arch_block = migrated
        entry = dict(payload)
        entry["model"] = model  # provenance: which model's sweep won this.
        model_block = arch_block.get(model)
        if not isinstance(model_block, dict):
            model_block = {}
        model_block[optimizer] = entry
        arch_block[model] = model_block
        # P0.2 — record the dim→macro map for this arch so the consumer can
        # apply the extra (non-SAFE) macros. Merge so sibling optimizers'
        # contributions accumulate; a per-arch map is shared across its opts +
        # models (it describes the arch's search-space macro names, not a
        # per-model thing), so it stays at the ARCH level.
        if macro_map:
            existing_map = arch_block.get("_macros")
            if not isinstance(existing_map, dict):
                existing_map = {}
            existing_map.update({k: str(v) for k, v in macro_map.items() if v})
            arch_block["_macros"] = existing_map
        existing[arch_key] = arch_block

        meta = existing.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["source"] = source
        meta["last_optimizer"] = optimizer
        meta["last_arch"] = arch_key
        if version_hash is not None:
            meta["compile_py_version_hash"] = version_hash
        try:
            import datetime
            meta["timestamp"] = datetime.datetime.now().isoformat()
        except Exception:  # pragma: no cover - datetime import never fails
            pass
        existing["_meta"] = meta

        _atomic_write_json(target, existing)
        return True
    except Exception:  # noqa: BLE001 — persist must never break tuning.
        return False


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` as pretty JSON atomically (tmp in same dir + replace)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._kernel_tuned.", suffix=".tmp",
                               dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Ninja build.ninja rewriting (consumer side — setup.py's BuildExtension).
#
# These live here (not in setup.py) so they are importable for CPU-only
# testing without running setup()'s torch import + GPU probe. setup.py's
# TunedBuildExtension monkeypatches torch's _write_ninja_file to call the
# original writer and then :func:`inject_overrides_into_ninja` on the result.
# --------------------------------------------------------------------------
def inject_overrides_into_ninja(ninja_path: str,
                                sources: List[str],
                                objects: List[str],
                                base_cuda_post_cflags: List[str],
                                tuned: Optional[Dict[str, Any]],
                                arch_key: str,
                                quote=None) -> int:
    """Insert per-build-statement ``cuda_post_cflags`` overrides into a ninja
    file for every CUDA TU that maps to a tuned optimizer.

    The override line carries the FULL flag list (``base_cuda_post_cflags`` +
    that optimizer's tuned extras) so ninja never self-references the variable
    (a wrong eval order would silently drop the base arch/-O3/fast-math flags).
    Returns the number of build edges that received an override (0 when there
    is nothing tuned, no base flags, or no matching CUDA edge).

    ``quote`` is an optional callable (``shlex.quote``) applied to each extra
    flag so its quoting matches torch's already-quoted ``base`` list; when
    None the extras are emitted verbatim (the safe per-TU flags contain no
    shell metacharacters).
    """
    if not base_cuda_post_cflags:
        return 0
    per_src = compute_source_flags(list(sources), tuned, arch_key)
    if not per_src:
        return 0
    if quote is None:
        quote = lambda s: s  # noqa: E731 — identity; safe flags need no quoting.

    base = list(base_cuda_post_cflags)
    # Map each source's ABSOLUTE object path to its override line. ninja writes
    # build edges with os.path.abspath(object), escaping ':' as '$:' and ' '
    # as '$ '; we match on the unescaped object path.
    overrides: Dict[str, str] = {}
    for src, obj in zip(sources, objects):
        flags = per_src.get(src)
        if not flags:
            continue
        abs_obj = os.path.abspath(obj)
        full = " ".join(base + [quote(f) for f in flags])
        overrides[abs_obj] = f"  cuda_post_cflags = {full}"
    if not overrides:
        return 0

    with open(ninja_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    out: List[str] = []
    injected = 0
    for line in lines:
        out.append(line)
        stripped = line.rstrip("\n")
        if stripped.startswith("build ") and "cuda_compile" in stripped:
            target = parse_ninja_build_target(stripped)
            if target is not None and target in overrides:
                out.append(overrides[target] + "\n")
                injected += 1
    if injected:
        with open(ninja_path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    return injected


def parse_ninja_build_target(stmt: str) -> Optional[str]:
    """Return the (unescaped) object path from a ninja
    ``build <obj>: <rule> <ins>`` statement, or ``None``.

    ninja escapes ``:`` as ``$:`` and `` `` as ``$ `` inside paths; we walk
    the token, treating ``$<c>`` as a literal pair, and stop at the first
    UNescaped ``:`` (the rule separator).
    """
    if not stmt.startswith("build "):
        return None
    rest = stmt[len("build "):]
    i = 0
    obj_chars: List[str] = []
    found = False
    while i < len(rest):
        c = rest[i]
        if c == "$" and i + 1 < len(rest):
            obj_chars.append(rest[i:i + 2])
            i += 2
            continue
        if c == ":":
            found = True
            break
        obj_chars.append(c)
        i += 1
    if not found:
        return None
    token = "".join(obj_chars).strip()
    return token.replace("$:", ":").replace("$ ", " ")
