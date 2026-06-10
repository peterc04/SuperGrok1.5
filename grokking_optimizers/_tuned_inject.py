"""Autotuner → product-build linkage: per-TU tuned-flag injection.

This module is the single source of truth for lever (b): making the
``compile.py`` autotuner's winning launch parameters ACTIVE in the shipped
``grokking_optimizers/_ops*.so`` instead of decorative.

Producer  : ``grokking_optimizers.compile.build_jit`` calls
            :func:`export_winner` the moment a JIT winner is decided,
            persisting per-(arch, optimizer) winners to the canonical JSON
            at ``grokking_optimizers/_kernel_tuned.json``.
Consumer  : ``setup.py``'s ``BuildExtension`` subclass reads that JSON and,
            per CUDA translation unit, appends ``-DSG_TUNED_*`` macros (and
            ``--maxrregcount`` when nonzero) to the nvcc flags of the TUs
            that belong to the matching optimizer — see
            :func:`source_extra_nvcc_flags` and :func:`compute_source_flags`.

IMPORT DISCIPLINE — this module is *pure stdlib* and MUST stay that way.
It is loaded by ``setup.py`` (which already imports torch, but should not
re-import the heavy package) and by ``tuning/test_build_injection.py``
(CPU-only, no torch / no GPU). Both load it by FILE PATH so that importing
it never triggers ``grokking_optimizers/__init__.py`` (which does
``import torch`` + ``get_ops()`` and would probe CUDA). Do NOT add any
``grokking_optimizers``-relative imports or third-party imports here.

SCOPE (phase 1 — the five SAFE per-TU dims only): block / vec / unroll /
async_depth as ``-DSG_TUNED_*`` macros, plus maxrregcount as a real
``--maxrregcount=N`` ptxas flag. ``cluster_shape`` and the feature macros
(TMA / WGMMA / fp8 / swizzle / ...) are component-scoped and riskier
(they change which kernel specialization compiles); they are deliberately
phase-2 and are NOT emitted here.
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


# --------------------------------------------------------------------------
# Flag computation.
# --------------------------------------------------------------------------
def source_extra_nvcc_flags(optimizer: Optional[str],
                            tuned: Optional[Dict[str, Any]],
                            arch_key: str) -> List[str]:
    """Extra nvcc flags for ONE optimizer's TUs under ``arch_key``.

    Returns ``[]`` when there is nothing tuned for this (arch, optimizer)
    — i.e. when ``optimizer is None`` (ambiguous TU), ``tuned`` is empty/None
    (no JSON), the arch is absent, or the optimizer has no recorded winner.
    Same value is applied to every TU of the same optimizer (design 1b).

    The emitted list is deterministic (``_EMIT_ORDER``); macro dims become
    ``-D<macro>=<v>`` and ``maxrregcount`` becomes ``--maxrregcount=<n>``
    only when ``n > 0``.
    """
    if optimizer is None or not tuned:
        return []
    arch_block = tuned.get(canonical_arch_key(arch_key))
    if not isinstance(arch_block, dict):
        return []
    combo = arch_block.get(optimizer)
    if not isinstance(combo, dict):
        return []

    flags: List[str] = []
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
    return flags


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
    ``sources`` they passed). A per-optimizer flag list is computed once and
    shared across that optimizer's TUs.
    """
    out: Dict[str, List[str]] = {}
    cache: Dict[Optional[str], List[str]] = {}
    for src in sources:
        opt = optimizer_for_source(src)
        if opt is None:
            continue
        if opt not in cache:
            cache[opt] = source_extra_nvcc_flags(opt, tuned, arch_key)
        flags = cache[opt]
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
# Only the safe per-TU dims are persisted; everything else in the winning
# combo (timing_ms, config_key, feature macros, fast-math markers, ...) is
# dropped from the JSON. The JSON is a STABLE handoff contract, not a dump of
# the autotuner's internal trial record.
_PERSIST_KEYS = tuple(MACROS.keys())  # block, vec, unroll, async_depth, maxrregcount


def _winner_payload(combo: Dict[str, Any]) -> Dict[str, Any]:
    """Project a winner combo down to the persisted per-TU dims."""
    payload: Dict[str, Any] = {}
    for key in _PERSIST_KEYS:
        if key not in combo:
            continue
        value = combo[key]
        if MACROS[key]["kind"] == "flag":
            # maxrregcount stored as int (0 == unset).
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(value, bool):
            payload[key] = bool(value)
        elif isinstance(value, (int, float, str)):
            payload[key] = value
        # tuples/lists (e.g. cluster_shape) are intentionally NOT persisted.
    return payload


def export_winner(optimizer: str,
                 model: str,
                 arch: str,
                 combo: Dict[str, Any],
                 *,
                 path: Optional[str] = None,
                 source: str = "compile.build_jit",
                 version_hash: Optional[str] = None) -> bool:
    """Persist ONE (arch, optimizer) winner to the canonical JSON.

    READ-MERGE-WRITE (design 1a): ``build_jit`` runs a single optimizer, so
    this merges into any existing JSON rather than overwriting — tuning all
    11 optimizers in sequence accumulates 11 entries. The write is atomic
    (tmp file + ``os.replace``) so a reader never sees a half-written file.

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
        payload = _winner_payload(combo)

        # Read-merge: start from any existing JSON so sibling optimizers'
        # winners survive. A corrupt existing file is discarded (we cannot
        # safely merge into garbage) but reported via the return value path.
        existing = load_tuned(target) or {}
        if not isinstance(existing, dict):
            existing = {}
        arch_block = existing.get(arch_key)
        if not isinstance(arch_block, dict):
            arch_block = {}
        entry = dict(payload)
        entry["model"] = model  # provenance: which model's sweep won this.
        arch_block[optimizer] = entry
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
