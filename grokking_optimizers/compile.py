"""grokking_optimizers.compile — Targeted per-(optimizer, model, arch) build
with a portable AOT-then-JIT cache, YAML-driven search space, Bayesian
or Exhaustive autotune, optional PGO, and runtime-split AOT/JIT
subprocesses.

This is the development-time companion to ``setup.py``. Where
``pip install -e .`` builds the full ``grokking_optimizers._ops``
extension with default flags, this module compiles a focused,
**maximally optimised** build for a single ``(optimizer, model, arch)``
triple. The pipeline is split into two halves so a CPU-only host can
do the heavy AOT compile and ship a cache file to the target GPU host,
which then does the JIT autotune sweep.

Two-phase pipeline
==================

**AOT phase (any host with the toolchain — no GPU required).**

  1. Resolve sources for the chosen arch (bindings + every launcher +
     every model TU).
  2. Hash the source set, the host cflags, the device cflags, the
     resolved search-space, and the PGO state. Look up
     ``(optimizer, model, arch)`` in the cache. If all hashes match an
     entry with ``aot_completed_at != None`` and ``pgo_enabled`` agrees
     and the recorded artefact still exists → **cache hit**.
  3. Otherwise build with ``torch.utils.cpp_extension.load`` (ninja,
     ``MAX_JOBS=$(nproc)``, sccache wiring when on PATH), arch-tuned
     codegen + LTO + perf flags, and the ``SG_BUILD_*`` macros so
     headers can ``#if`` out unused specialisations.
  4. If ``--pgo`` is set, the AOT phase runs the 3-pass loop:
       instrument → workload → use. The PGO state is recorded so
       subsequent runs short-circuit when the workload hash matches.
  5. Record the artefact path, size, mtime, and SHA-256 in the cache.
     Mark ``aot_completed_at``. Save cache to disk.

**JIT autotune phase (target GPU only).**

  1. Load the resolved search space from ``configs/search_space.yaml``.
     Generate the cartesian product → apply static pre-filter rules →
     log the elimination count.
  2. Two autotune modes:
       - ``--mode exhaustive``: every survivor is built and timed.
         Cache is written every N trials so a Ctrl-C is recoverable.
       - ``--mode bayesian`` (default): Optuna TPE for ``--bayesian-trials``
         iterations + top-K refinement with ±2-step neighbours. The
         Optuna study persists to ``<cache_dir>/optuna_<opt>_<model>_<arch>.db``
         for cross-run resume.
  3. Timing is done by a **persistent subprocess worker** that holds a
     warm CUDA / HIP context for the full sweep, using CUDA-graph
     replay where the kernel supports it. Per-variant timeout falls
     back to one-shot subprocess timing on worker crash.
  4. Pick the winning combo (lowest median ms). Set ``cache.tuned_config``.
     Mark ``jit_completed_at``.
  5. Rewrite ``csrc/algorithms/tuned_configs.h`` with the winning
     macros so downstream consumers (setup.py builds, IDE Intellisense)
     pick up the tuned defaults.
  6. Rebuild the primary artefact with the tuned configs baked in.

Runtime split
=============
``--runtime {aot, jit, both}`` controls which subprocesses run.
``both`` (default) spawns an AOT subprocess (CPU-only env) followed by
a JIT subprocess (GPU env). Each subprocess re-enters ``main`` with
the single-phase flag and reads/writes the same on-disk cache. This
means AOT and JIT can be tuned independently (env vars, library
versions, memory limits) without their settings bleeding into each
other. ``--aot-only`` / ``--jit-only`` remain as aliases.

Cache schema v3
===============
The on-disk format is JSON; ``CACHE_VERSION == 3``. Forward-compatible
with v2 (auto-migrated on load). See ``INTERFACES.md`` for the full
shape. Headline additions over v2:

    mode                 "exhaustive" | "bayesian" | None
    pgo_enabled          bool
    pgo_profile_dir      str | None
    pgo_workload_hash    str | None
    pgo_completed_at     str | None
    pgo_host             host dict
    search_space_hash    str (SHA-256 of resolved YAML space)
    bayesian_trials      [trial_record, …]   (stage="tpe" | "refine")

Per-entry hashes (``source_hash``, ``host_cflags_hash``,
``device_cflags_hash``, ``search_space_hash``, plus ``pgo_enabled``)
all gate AOT freshness; any change invalidates the entry.

Usage
=====

CLI::

    # End-to-end on a GPU host (AOT + Bayesian JIT autotune + profile)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json \\
        --mode bayesian --bayesian-trials 500

    # AOT-only on a CPU host with PGO (ships the cache)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json --aot-only --pgo \\
        --pgo-workload scripts/pgo_workload.py --pgo-steps 1000 \\
        --aot-artifact-dir build/compiled/aot_artifacts

    # JIT autotune only on the GPU host (consumes the cache)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json --jit-only --mode exhaustive

Importable::

    from grokking_optimizers.compile import build, build_aot, build_jit, CompileCache
    cache = CompileCache(Path("build/.compile_cache.json"))
    so_path = build(optimizer="supergrok2", model="mamba", arch="sm_90",
                    cache=cache, autotune_mode="bayesian",
                    bayesian_trials=500, pgo=False)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from grokking_optimizers.profile import (
    ARCH_INFO,
    ARCHES,
    MODELS,
    NCPUS,
    OPTIMIZERS,
    OPT_CLASS,  # noqa: F401  (re-exported for back-compat)
    REPO_ROOT,
    _dispatch_profile,
    child_env,
    env_overlay,
    make_progress,
)

from grokking_optimizers import search_space as _ss
from grokking_optimizers import pgo as _pgo


CACHE_VERSION = 3
DEFAULT_CACHE_NAME = ".compile_cache.json"
DEFAULT_SEARCH_SPACE = REPO_ROOT / "configs" / "search_space.yaml"
DEFAULT_PGO_WORKLOAD = REPO_ROOT / "scripts" / "pgo_workload.py"
JIT_CACHE_FLUSH_EVERY = 5   # save cache every N completed JIT trials

# How many trials Bayesian "quick" mode runs (vs the full 500 default).
QUICK_BAYESIAN_TRIALS = 25


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_sources(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        rel = (str(p.relative_to(REPO_ROOT))
               if str(p).startswith(str(REPO_ROOT)) else str(p))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(p).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_flags(flags: List[str]) -> str:
    return _sha256_str("\0".join(flags))


# ---------------------------------------------------------------------------
# Persistent cache (in-memory dict, JSON on disk, atomic save) — v3 schema
# ---------------------------------------------------------------------------

def _current_host() -> dict:
    """Capture host identity for the cache provenance trail."""
    try:
        import torch
        torch_v = getattr(torch, "__version__", None)
        cuda_v = getattr(torch.version, "cuda", None)
        hip_v = getattr(torch.version, "hip", None)
    except ImportError:
        torch_v = cuda_v = hip_v = None
    return {
        "recorded_at": datetime.datetime.now().isoformat(),
        "platform":    platform.platform(),
        "python":      sys.version.split()[0],
        "torch":       torch_v,
        "cuda":        cuda_v,
        "hip":         hip_v,
        "ncpus":       NCPUS,
    }


_V3_DEFAULTS: Dict[str, Any] = {
    "mode":               None,
    "pgo_enabled":        False,
    "pgo_profile_dir":    None,
    "pgo_workload_hash":  None,
    "pgo_completed_at":   None,
    "pgo_host":           None,
    "search_space_hash":  None,
    "bayesian_trials":    [],
}


def _fresh_entry() -> dict:
    return {
        "source_hash":         None,
        "host_cflags_hash":    None,
        "device_cflags_hash":  None,
        "primary_artifact":    None,
        "variant_artifacts":   {},
        "sweep_history":       [],
        "tuned_config":        None,
        "aot_completed_at":    None,
        "jit_completed_at":    None,
        "aot_host":            None,
        "jit_host":            None,
        **{k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in _V3_DEFAULTS.items()},
    }


def _migrate_v2_to_v3(data: dict) -> dict:
    """Forward-migrate a v2 cache: add v3 keys with defaults, bump version."""
    data["version"] = CACHE_VERSION
    data.setdefault("entries", {})
    for k, entry in list(data["entries"].items()):
        if not isinstance(entry, dict):
            continue
        for nk, nv in _V3_DEFAULTS.items():
            entry.setdefault(nk, list(nv) if isinstance(nv, list)
                             else dict(nv) if isinstance(nv, dict) else nv)
    data["migrated_from_v2_at"] = datetime.datetime.now().isoformat()
    return data


class CompileCache:
    """Persistent build cache.

    The cache lives in-memory as a Python dict for the duration of a
    ``build()`` call and is written back to disk atomically at the end
    (or via explicit ``.save()``). Mutations are tracked via the
    ``_dirty`` flag so a no-op build doesn't touch the file. The cache
    is JSON; see module docstring + ``INTERFACES.md`` for the v3 shape.
    """

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._dirty = False
        self._data = self._load()

    def _fresh(self) -> dict:
        return {
            "version":      CACHE_VERSION,
            "created_at":   datetime.datetime.now().isoformat(),
            "host_history": [_current_host()],
            "entries":      {},
        }

    def _load(self) -> dict:
        if self.path is None or not self.path.exists():
            data = self._fresh()
            self._dirty = True
            return data
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            backup = self.path.with_suffix(self.path.suffix + ".corrupt.bak")
            try:
                self.path.rename(backup)
            except OSError:
                pass
            data = self._fresh()
            self._dirty = True
            return data
        version = data.get("version")
        if version == CACHE_VERSION:
            data.setdefault("host_history", []).append(_current_host())
            self._dirty = True
            return data
        if version == 2:
            # Back up the v2 file untouched, then migrate the loaded data.
            backup = self.path.with_suffix(self.path.suffix + ".v2.bak")
            try:
                shutil.copy2(self.path, backup)
            except OSError:
                pass
            data = _migrate_v2_to_v3(data)
            data.setdefault("host_history", []).append(_current_host())
            self._dirty = True
            return data
        # Older / unknown version → archive and start fresh.
        backup = self.path.with_suffix(
            self.path.suffix + f".v{version or 0}.bak")
        try:
            shutil.move(self.path, backup)
        except OSError:
            pass
        data = self._fresh()
        self._dirty = True
        return data

    def save(self) -> None:
        """Atomically write to disk via tmp-file rename."""
        if self.path is None or not self._dirty:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
            tmp.replace(self.path)
            self._dirty = False

    def key(self, opt: str, model: str, arch: str) -> str:
        return f"{opt}/{model}/{arch}"

    def get(self, opt: str, model: str, arch: str) -> dict:
        with self._lock:
            k = self.key(opt, model, arch)
            entry = self._data["entries"].setdefault(k, _fresh_entry())
            # Defensive: stale v2 entries that escaped migration.
            for nk, nv in _V3_DEFAULTS.items():
                entry.setdefault(nk, list(nv) if isinstance(nv, list)
                                 else dict(nv) if isinstance(nv, dict) else nv)
            self._dirty = True
            return entry

    # ── Freshness ────────────────────────────────────────────────────

    def is_aot_fresh(self, opt: str, model: str, arch: str,
                     source_hash: str, host_flags_hash: str,
                     device_flags_hash: str,
                     pgo_enabled: bool = False,
                     pgo_workload_hash: Optional[str] = None,
                     search_space_hash: Optional[str] = None) -> bool:
        e = self.get(opt, model, arch)
        if not e["aot_completed_at"]:
            return False
        if e["source_hash"] != source_hash:
            return False
        if e["host_cflags_hash"] != host_flags_hash:
            return False
        if e["device_cflags_hash"] != device_flags_hash:
            return False
        if bool(e.get("pgo_enabled")) != bool(pgo_enabled):
            return False
        if pgo_enabled and pgo_workload_hash is not None:
            if e.get("pgo_workload_hash") != pgo_workload_hash:
                return False
        if (search_space_hash is not None
                and e.get("search_space_hash") not in (None, search_space_hash)):
            return False
        art = e["primary_artifact"]
        if not art:
            return False
        p = Path(art["path"])
        try:
            return p.exists() and p.stat().st_size == art.get("size")
        except OSError:
            return False

    def is_jit_fresh(self, opt: str, model: str, arch: str,
                     search_space_hash: Optional[str] = None) -> bool:
        e = self.get(opt, model, arch)
        if not (e["jit_completed_at"] and e["tuned_config"]):
            return False
        if (search_space_hash is not None
                and e.get("search_space_hash") not in (None, search_space_hash)):
            return False
        return True

    # ── Recorders ────────────────────────────────────────────────────

    def record_aot(self, opt: str, model: str, arch: str, *,
                   source_hash: str, host_flags_hash: str,
                   device_flags_hash: str,
                   so_path: Optional[Path],
                   pgo_enabled: bool = False,
                   pgo_profile_dir: Optional[Path] = None,
                   pgo_workload_hash: Optional[str] = None,
                   search_space_hash: Optional[str] = None) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["source_hash"]        = source_hash
            e["host_cflags_hash"]   = host_flags_hash
            e["device_cflags_hash"] = device_flags_hash
            e["pgo_enabled"]        = bool(pgo_enabled)
            if pgo_profile_dir is not None:
                e["pgo_profile_dir"] = str(pgo_profile_dir)
            if pgo_workload_hash is not None:
                e["pgo_workload_hash"] = pgo_workload_hash
            if search_space_hash is not None:
                e["search_space_hash"] = search_space_hash
            if so_path is not None and Path(so_path).exists():
                stat = Path(so_path).stat()
                e["primary_artifact"] = {
                    "path":   str(so_path),
                    "size":   stat.st_size,
                    "mtime":  stat.st_mtime,
                    "sha256": _sha256_file(Path(so_path)),
                }
            e["aot_completed_at"] = datetime.datetime.now().isoformat()
            e["aot_host"]         = _current_host()
            self._dirty = True

    def record_pgo(self, opt: str, model: str, arch: str, *,
                   profile_dir: Path, workload_hash: str) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["pgo_enabled"]        = True
            e["pgo_profile_dir"]    = str(profile_dir)
            e["pgo_workload_hash"]  = workload_hash
            e["pgo_completed_at"]   = datetime.datetime.now().isoformat()
            e["pgo_host"]           = _current_host()
            self._dirty = True

    def record_variant(self, opt: str, model: str, arch: str,
                       config_key: str, so_path: Optional[Path]) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            if so_path is not None and Path(so_path).exists():
                stat = Path(so_path).stat()
                e["variant_artifacts"][config_key] = {
                    "path":  str(so_path),
                    "size":  stat.st_size,
                    "mtime": stat.st_mtime,
                }
                self._dirty = True

    def record_trial(self, opt: str, model: str, arch: str,
                     trial: Dict[str, Any]) -> None:
        """Append a trial record to both bayesian_trials and sweep_history."""
        with self._lock:
            e = self.get(opt, model, arch)
            e["bayesian_trials"].append(trial)
            e["sweep_history"].append(trial)
            self._dirty = True

    def record_sweep(self, opt: str, model: str, arch: str, *,
                     config: dict, timing_ms: float, **extras) -> None:
        """Back-compat: simple grid record for non-bayesian sweeps."""
        with self._lock:
            e = self.get(opt, model, arch)
            e["sweep_history"].append({
                "stage":        "exhaustive",
                "config":       config,
                "timing_ms":    timing_ms,
                "host":         _current_host(),
                "recorded_at":  datetime.datetime.now().isoformat(),
                **extras,
            })
            self._dirty = True

    def set_tuned(self, opt: str, model: str, arch: str,
                  config: dict, *, mode: Optional[str] = None,
                  search_space_hash: Optional[str] = None) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["tuned_config"]      = config
            e["jit_completed_at"]  = datetime.datetime.now().isoformat()
            e["jit_host"]          = _current_host()
            if mode is not None:
                e["mode"] = mode
            if search_space_hash is not None:
                e["search_space_hash"] = search_space_hash
            self._dirty = True


# ---------------------------------------------------------------------------
# BuildSpec
# ---------------------------------------------------------------------------

@dataclass
class BuildSpec:
    optimizer: str
    model: str
    arch: str
    out_dir: Path
    autotune: bool = True
    autotune_mode: str = "bayesian"   # "exhaustive" | "bayesian"
    profile: bool = True
    verbose: bool = False
    extra_macros: List[str] = field(default_factory=list)
    runtime: str = "both"             # "aot" | "jit" | "both"
    aot_only: bool = False
    jit_only: bool = False
    search_space_path: Optional[Path] = None
    aot_artifact_dir: Optional[Path] = None
    pgo: bool = False
    pgo_workload: Optional[Path] = None
    pgo_steps: int = 1000
    bayesian_trials: int = 500
    top_k: int = 20
    seed: int = 0
    debug_symbols: bool = False
    # §12 A1 / A2 — Hyperband pruner + transfer learning
    pruner: str = "none"              # "none" | "median" | "hyperband"
    transfer_learning: bool = False


def _validate(spec: BuildSpec) -> None:
    if spec.optimizer not in OPTIMIZERS:
        raise ValueError(
            f"optimizer={spec.optimizer!r} not in {list(OPTIMIZERS)}")
    if spec.model not in MODELS:
        raise ValueError(f"model={spec.model!r} not in {list(MODELS)}")
    if spec.arch not in ARCHES:
        raise ValueError(f"arch={spec.arch!r} not in {list(ARCHES)}")
    if spec.autotune_mode not in ("exhaustive", "bayesian"):
        raise ValueError(
            f"autotune_mode={spec.autotune_mode!r} not in "
            "{'exhaustive', 'bayesian'}")
    if spec.runtime not in ("aot", "jit", "both"):
        raise ValueError(
            f"runtime={spec.runtime!r} not in {{'aot', 'jit', 'both'}}")
    if spec.pruner not in ("none", "median", "hyperband"):
        raise ValueError(
            f"pruner={spec.pruner!r} not in {{'none', 'median', 'hyperband'}}")


def _collect_sibling_trials(cache: CompileCache, opt: str, model: str,
                            arch: str) -> List[Dict[str, Any]]:
    """§12 A2 — transfer learning: gather successful trials from
    sibling-optimizer studies on the same (model, arch). The returned
    list is ready to pass to ``bayesian.run_bayesian(seed_trials=...)``.

    Sibling = same (model, arch), different optimizer. Failed trials
    (``timing_ms is None``) are skipped so the surrogate isn't poisoned.
    """
    sibling_trials: List[Dict[str, Any]] = []
    entries = cache._data.get("entries", {})
    suffix = f"/{model}/{arch}"
    for key, entry in entries.items():
        if not key.endswith(suffix):
            continue
        if key.startswith(f"{opt}/"):
            continue  # don't reseed from the same optimizer
        for t in entry.get("bayesian_trials", []):
            if t.get("timing_ms") is not None and "config" in t:
                sibling_trials.append(t)
    return sibling_trials


# ---------------------------------------------------------------------------
# Source / flag resolution
# ---------------------------------------------------------------------------

def _resolve_sources(spec: BuildSpec) -> List[Path]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        return []
    backend = REPO_ROOT / "csrc/backends" / info["subdir"]
    bindings = sorted((REPO_ROOT / "csrc/bindings").glob("*.cpp"))
    launchers: List[Path] = []
    for g in info["launcher_glob"]:
        launchers.extend(sorted(backend.glob(g)))
    models: List[Path] = []
    for g in info["model_glob"]:
        models.extend(sorted((backend / "models").glob(g)))
    return bindings + launchers + models


def _build_macros(spec: BuildSpec) -> List[str]:
    macros = [
        f"-DSG_BUILD_OPTIMIZER_{spec.optimizer.upper()}=1",
        f"-DSG_BUILD_MODEL_{spec.model.upper()}=1",
        f"-D{ARCH_INFO[spec.arch]['macro']}=1",
        "-DSG_VERBOSE=1",
    ]
    return macros + list(spec.extra_macros)


# ---- Performance flag bases (see INTERFACES.md §9) ------------------------

HOST_CFLAGS_BASE = [
    "-O3", "-std=c++17", "-fPIC",
    "-flto=full", "-march=native", "-mtune=native",
    "-fno-semantic-interposition", "-fvisibility=hidden",
    "-fdata-sections", "-ffunction-sections",
    "-fno-math-errno", "-fno-trapping-math",
    "-fomit-frame-pointer",
    "-ffast-math", "-funroll-loops",
]

NVCC_DEVICE_BASE = [
    "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
    "--expt-relaxed-constexpr",
    "--threads", "8",
    "-Xfatbin", "-compress-all",
    "-Xptxas", "-O3", "-Xptxas", "-v", "-Xptxas", "--warn-on-spills",
    "-Xptxas", "--allow-expensive-optimizations=true",
    "-Xptxas", "--def-load-cache=ca",
    "-Xptxas", "--def-store-cache=wb",
    "--extra-device-vectorization",
    "-Xcompiler", "-fPIC", "-Xcompiler", "-flto=full",
    "--resource-usage",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
    "-dlto",
]

HIPCC_DEVICE_BASE = [
    "-O3", "-std=c++17", "-DWITH_HIP",
    "-ffast-math", "-fPIC",
    "--offload-arch=gfx942",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-amdgpu-internalize-symbols",
    "-fgpu-flush-denormals-to-zero",
    "-Rpass-analysis=kernel-resource-usage",
    "-flto",
]

LDFLAGS_BASE = [
    "-flto=full", "-Wl,--as-needed",
    "-Wl,--gc-sections", "-Wl,-O3",
    "-Wl,--icf=all",
]


def _host_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        return []
    base = list(HOST_CFLAGS_BASE) + [f"-D{info['host_define']}"]
    if spec.debug_symbols or spec.profile:
        base += ["-ggdb"]
    return base + _build_macros(spec)


def _device_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "cuda":
        base = list(NVCC_DEVICE_BASE)
        if spec.debug_symbols or spec.profile:
            base += ["-lineinfo", "--generate-line-info"]
        if (REPO_ROOT / "third_party/cutlass/include").exists():
            base += ["-DWITH_CUTLASS", "-DCUTLASS_NVCC_ARCHS=90a"]
        return base + _build_macros(spec)
    if info["vendor"] == "hip":
        base = list(HIPCC_DEVICE_BASE)
        if spec.debug_symbols or spec.profile:
            base += ["-ggdb"]
        return base + _build_macros(spec)
    return []


def _ldflags(spec: BuildSpec) -> List[str]:
    if ARCH_INFO[spec.arch]["vendor"] == "pallas":
        return []
    return list(LDFLAGS_BASE)


def _include_paths() -> List[str]:
    return [str(REPO_ROOT / "csrc/bindings"), str(REPO_ROOT)]


# ---------------------------------------------------------------------------
# Compile-wrapper env wiring — ccache → sccache → unwrapped (§12 C2 / I1)
# ---------------------------------------------------------------------------

def _writable_cache_dir(name: str) -> Optional[Path]:
    """Prefer /dev/shm/<name> (ramdisk); fall back to ~/.cache/<name>."""
    for candidate in (Path(f"/dev/shm/{name}"), Path.home() / ".cache" / name):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return None


def _sccache_env() -> Dict[str, str]:
    """Detect ccache (preferred for host TUs) and sccache (preferred for
    NVCC), wire whichever is present. Honour ``SCCACHE_REDIS_ENDPOINT``
    if set (§12 I1 — Redis-backed shared cache across hosts).

    The host CC/CXX get ccache when it's on PATH (3-4.5× faster than
    sccache on local C/C++ object retrieval per the sccache project's
    own measurements). NVCC always goes through sccache when sccache is
    on PATH (after vllm#13697 the CUDA hash bug is fixed)."""
    out: Dict[str, str] = {}
    ccache = shutil.which("ccache")
    sccache = shutil.which("sccache")

    if ccache:
        host_dir = _writable_cache_dir("ccache")
        if host_dir is not None:
            out["CCACHE_DIR"] = str(host_dir)
            out["CC"]  = f"{ccache} {os.environ.get('CC',  'cc')}"
            out["CXX"] = f"{ccache} {os.environ.get('CXX', 'c++')}"

    if sccache:
        sc_dir = _writable_cache_dir("sccache")
        if sc_dir is not None:
            out.setdefault("SCCACHE_DIR", str(sc_dir))
            # If ccache didn't claim host wrappers, sccache takes them.
            out.setdefault("CC",  f"{sccache} {os.environ.get('CC',  'cc')}")
            out.setdefault("CXX", f"{sccache} {os.environ.get('CXX', 'c++')}")
            out["CUDA_NVCC_EXECUTABLE"] = f"{sccache} nvcc"
        # Redis backend (§12 I1) propagates unconditionally if user set it.
        for k in ("SCCACHE_REDIS_ENDPOINT", "SCCACHE_REDIS", "SCCACHE_S3_BUCKET"):
            if k in os.environ:
                out[k] = os.environ[k]

    return out


# ---------------------------------------------------------------------------
# Toolchain version probe — §12 C1
# ---------------------------------------------------------------------------

def _probe_nvcc_version() -> Optional[Tuple[int, int]]:
    """Return (major, minor) of nvcc on PATH, or None."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True,
                                      timeout=10, stderr=subprocess.STDOUT)
    except (subprocess.SubprocessError, OSError):
        return None
    # Format: "release 12.6, V12.6.85"
    import re
    m = re.search(r"release\s+(\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _probe_hipcc_version() -> Optional[Tuple[int, int]]:
    """Return (major, minor) of HIP on PATH, or None."""
    hipcc = shutil.which("hipcc")
    if not hipcc:
        return None
    try:
        out = subprocess.check_output([hipcc, "--version"], text=True,
                                      timeout=10, stderr=subprocess.STDOUT)
    except (subprocess.SubprocessError, OSError):
        return None
    import re
    m = re.search(r"HIP version[:\s]+(\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _newer_compiler_flags(arch: str, report=None) -> Tuple[List[str], List[str]]:
    """Return (extra_host, extra_device) flags that are safe additions
    when the detected toolchain is new enough. §12 C1 — pure autodetect,
    no-op on older toolchains."""
    extra_host: List[str] = []
    extra_device: List[str] = []
    if arch in ("sm_90",):
        ver = _probe_nvcc_version()
        if ver:
            if report:
                report.write(f"  [toolchain] nvcc {ver[0]}.{ver[1]}\n")
            if ver >= (12, 6):
                # NVCC 12.6+ supports --split-compile for opt-phase parallelism.
                extra_device += [f"--split-compile={NCPUS}"]
                if report:
                    report.write(f"  [toolchain] enabling "
                                 f"--split-compile={NCPUS} (NVCC ≥12.6)\n")
        elif report:
            report.write("  [toolchain] nvcc not on PATH; "
                         "skipping version-gated flags\n")
    elif arch == "gfx942":
        ver = _probe_hipcc_version()
        if ver and report:
            report.write(f"  [toolchain] HIP {ver[0]}.{ver[1]}\n")
        elif report:
            report.write("  [toolchain] hipcc not on PATH; "
                         "skipping version-gated flags\n")
    return extra_host, extra_device


# ---------------------------------------------------------------------------
# Build driver — torch.utils.cpp_extension.load with ninja
# ---------------------------------------------------------------------------

def _torch_load(spec: BuildSpec, sources: List[Path],
                host_cflags: List[str], device_cflags: List[str],
                ldflags: List[str], report,
                module_suffix: str = "") -> Optional[Path]:
    """Compile via torch.utils.cpp_extension.load (ninja, MAX_JOBS=nproc)."""
    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        report.write(f"  [load] torch not importable: {exc}\n")
        return None

    module_name = (
        f"grokking_compiled_{spec.optimizer}_{spec.model}_{spec.arch}"
        f"{module_suffix}"
    )
    build_dir = spec.out_dir / module_name
    build_dir.mkdir(parents=True, exist_ok=True)

    report.write(f"  module:    {module_name}\n")
    report.write(f"  build dir: {build_dir}\n")
    report.write(f"  sources:   {len(sources)} files\n")
    report.write(f"  host:      {' '.join(host_cflags)}\n")
    report.write(f"  device:    {' '.join(device_cflags)}\n")
    report.write(f"  ldflags:   {' '.join(ldflags)}\n")
    report.flush()

    with_cuda = ARCH_INFO[spec.arch]["vendor"] in ("cuda", "hip")
    # §12 C1 — newer compiler probe; auto-append --split-compile etc.
    extra_host, extra_device = _newer_compiler_flags(spec.arch, report)
    if extra_host:
        host_cflags = list(host_cflags) + extra_host
    if extra_device:
        device_cflags = list(device_cflags) + extra_device
    overlay = {
        "MAX_JOBS": NCPUS,
        "NINJA_STATUS": "[%f/%t %es] ",
        "TORCH_CUDA_VERBOSE_BUILD": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": NCPUS,
        "NVCC_THREADS": "8",
        **_sccache_env(),
    }
    if "CCACHE_DIR" in overlay:
        report.write(f"  ccache:    {overlay['CCACHE_DIR']}\n")
    if "SCCACHE_DIR" in overlay:
        report.write(f"  sccache:   {overlay['SCCACHE_DIR']}\n")
    if "SCCACHE_REDIS_ENDPOINT" in overlay:
        report.write(f"  sccache Redis: {overlay['SCCACHE_REDIS_ENDPOINT']}\n")
    with env_overlay(**overlay):
        t0 = time.monotonic()
        try:
            load(
                name=module_name,
                sources=[str(s) for s in sources],
                extra_cflags=host_cflags,
                extra_cuda_cflags=device_cflags,
                extra_ldflags=ldflags,
                extra_include_paths=_include_paths(),
                build_directory=str(build_dir),
                verbose=spec.verbose,
                with_cuda=with_cuda,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            report.write(f"\n[build FAILED after {elapsed:.1f}s]\n{exc}\n")
            return None

    elapsed = time.monotonic() - t0
    so = next(build_dir.glob(f"{module_name}*.so"), None)
    report.write(f"\n[build OK in {elapsed:.1f}s] -> {so}\n")
    return so


# ---------------------------------------------------------------------------
# JIT autotune — orchestrates Bayesian / Exhaustive sweep + worker
# ---------------------------------------------------------------------------

def _variant_macros(config: Dict[str, Any], dims: List[Dict[str, Any]],
                    target: str) -> List[str]:
    """Macros + extra-flag overrides for one config × target."""
    macros = _ss.resolve_macros(config, dims, target)
    extra = _ss.resolve_extra_nvcc_flags(config, dims) if target == "device" else []
    extra_hip = _ss.resolve_extra_hipcc_flags(config, dims) if target == "device" else []
    # Only one of CUDA / HIP applies per arch.
    return macros + (extra if "--maxrregcount" not in " ".join(extra_hip) else extra_hip)


def _make_variant_timer(spec: BuildSpec, sources: List[Path],
                        host_cflags_base: List[str],
                        device_cflags_base: List[str],
                        ldflags: List[str], dims: List[Dict[str, Any]],
                        cache: CompileCache,
                        worker,                       # Optional[TimingWorker]
                        report,
                        progress_state: Dict[str, Any]):
    """Return a closure ``timer(config) -> result dict | None`` for the
    Bayesian/Exhaustive driver. Builds the variant .so, records it in
    the cache, then asks the worker to time it (fallback: one-shot
    subprocess)."""

    def timer(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ckey = _ss.config_key(config)
        host_extra = _variant_macros(config, dims, "host")
        device_extra = _variant_macros(config, dims, "device")

        # Per-variant flush of the running ETA window.
        progress_state["last_start"] = time.monotonic()

        variant_so = _torch_load(
            spec, sources,
            host_cflags_base + host_extra,
            device_cflags_base + device_extra,
            ldflags, report,
            module_suffix=f"_{_short_key(ckey)}",
        )
        if variant_so is None:
            return None
        cache.record_variant(spec.optimizer, spec.model, spec.arch,
                             ckey, variant_so)

        result = None
        if worker is not None and worker.alive():
            result = worker.time(variant_so)
            if result is None:
                report.write(f"    [worker time failed for {ckey}; "
                             "restart + fallback]\n")
                worker.restart()
        if result is None:
            result = _time_variant_oneshot(
                variant_so, OPT_CLASS[spec.optimizer], report=report)
        # Update rolling window
        elapsed = time.monotonic() - progress_state["last_start"]
        progress_state["window"].append(elapsed)
        if len(progress_state["window"]) > 20:
            progress_state["window"].pop(0)
        return result

    return timer


def _short_key(ckey: str) -> str:
    """Shorten a config key for use in a directory name (avoids OS path limits)."""
    if len(ckey) <= 80:
        return ckey
    return hashlib.sha1(ckey.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# One-shot timing fallback (kept from v2 for worker-crash resilience)
# ---------------------------------------------------------------------------

_TIMING_SCRIPT = r"""
import sys, json, importlib.util, traceback
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "torch.cuda.is_available() == False"}))
        sys.exit(1)
    so_path = {so_path!r}
    if "grokking_optimizers._ops" in sys.modules:
        del sys.modules["grokking_optimizers._ops"]
    spec = importlib.util.spec_from_file_location("grokking_optimizers._ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["grokking_optimizers._ops"] = mod
    from grokking_optimizers import {opt_class}
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn({size}, {size}, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    for _ in range({warmup}):
        p.grad = g.clone()
        opt = {opt_class}([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()
    timings = []
    for _ in range({iters}):
        p.grad = g.clone()
        opt = {opt_class}([p], lr=1e-3)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    print(json.dumps({
        "timing_ms": timings[len(timings) // 2],
        "min_ms":    timings[0],
        "max_ms":    timings[-1],
        "n":         len(timings),
    }))
except Exception as exc:
    print(json.dumps({"error": str(exc), "tb": traceback.format_exc()}))
    sys.exit(1)
"""


def _time_variant_oneshot(variant_so: Path, opt_class: str, *,
                          size: int = 4096, warmup: int = 5, iters: int = 21,
                          timeout: int = 180, report=None) -> Optional[Dict[str, Any]]:
    body = _TIMING_SCRIPT.format(
        so_path=str(variant_so),
        opt_class=opt_class, size=size, warmup=warmup, iters=iters,
    )
    fd, path = tempfile.mkstemp(suffix="_time.py")
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, path], cwd=REPO_ROOT, env=child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        out = (proc.stdout or "").strip()
        last = next((ln for ln in reversed(out.splitlines())
                     if ln.startswith("{")), None)
        if last is None:
            if report is not None:
                report.write(f"    [oneshot no-json]\n{out}\n")
            return None
        try:
            result = json.loads(last)
        except json.JSONDecodeError:
            if report is not None:
                report.write(f"    [oneshot json-decode error]\n{out}\n")
            return None
        if "error" in result:
            if report is not None:
                report.write(f"    [oneshot error: {result['error']}]\n")
            return None
        return result
    except subprocess.TimeoutExpired:
        if report is not None:
            report.write(f"    [oneshot timeout after {timeout}s]\n")
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Autotune drivers
# ---------------------------------------------------------------------------

def _jit_autotune(spec: BuildSpec, sources: List[Path],
                  host_cflags: List[str], device_cflags: List[str],
                  ldflags: List[str], cache: CompileCache,
                  report) -> Optional[Dict[str, Any]]:
    """Load the YAML space, prefilter, then dispatch to bayesian or
    exhaustive driver. Returns the winning config dict (with at least
    the dims set as the search space + ``timing_ms``)."""
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except ImportError:
        gpu_ok = False
    if not gpu_ok:
        report.write("  [jit-autotune] no GPU visible — skipping. Run JIT "
                     "phase on the target GPU host with this cache.\n")
        return None
    if ARCH_INFO[spec.arch]["vendor"] == "pallas":
        report.write("  [jit-autotune] pallas backend; no C++ tuning.\n")
        return None

    yaml_path = (spec.search_space_path or DEFAULT_SEARCH_SPACE)
    report.write(f"  [search-space] {yaml_path}\n")
    space = _ss.load_yaml(yaml_path)
    if spec.arch not in space:
        report.write(f"  [jit-autotune] no search space for arch={spec.arch}\n")
        return None
    space_hash = _ss.hash_space(space, spec.arch)
    report.write(f"  [search-space] hash={space_hash[:16]}\n")

    if cache.is_jit_fresh(spec.optimizer, spec.model, spec.arch,
                          search_space_hash=space_hash):
        tuned = cache.get(spec.optimizer, spec.model, spec.arch)["tuned_config"]
        report.write(f"  [jit-autotune] cache hit: tuned={tuned}\n")
        return tuned

    all_configs = _ss.cartesian(space, spec.arch)
    survivors, eliminated = _ss.prefilter(
        all_configs, space[spec.arch].get("prefilter", {}))
    report.write(f"  [prefilter] {len(all_configs)} candidates → "
                 f"{len(survivors)} survivors ({eliminated} eliminated)\n")
    if not survivors:
        report.write("  [jit-autotune] no survivors after prefilter.\n")
        return None

    # Spawn the persistent worker.
    from grokking_optimizers.timing_worker import TimingWorker
    worker = TimingWorker(opt_class=OPT_CLASS[spec.optimizer])
    if not worker.start():
        report.write("  [worker] start FAILED; falling back to one-shot per variant.\n")
        worker = None
    else:
        report.write("  [worker] persistent timing worker is up.\n")

    progress_state = {"last_start": time.monotonic(), "window": []}
    timer = _make_variant_timer(
        spec, sources, host_cflags, device_cflags, ldflags,
        space[spec.arch]["dims"], cache, worker, report, progress_state)

    dims = space[spec.arch]["dims"]
    try:
        if spec.autotune_mode == "exhaustive":
            winning = _run_exhaustive(spec, survivors, dims, timer, cache,
                                      space_hash, report)
        else:
            winning = _run_bayesian(spec, survivors, space, dims, timer, cache,
                                    space_hash, report)
    finally:
        if worker is not None:
            worker.stop()
    return winning


def _run_exhaustive(spec: BuildSpec, configs: List[Dict[str, Any]],
                    dims: List[Dict[str, Any]],
                    timer, cache: CompileCache, space_hash: str,
                    report) -> Optional[Dict[str, Any]]:
    report.write(f"\n  [exhaustive] sweeping {len(configs)} survivors\n")
    step, close = make_progress(len(configs),
                                f"jit-exhaustive {spec.optimizer}/{spec.arch}")
    best: Optional[Dict[str, Any]] = None
    try:
        for i, cfg in enumerate(configs, 1):
            ckey = _ss.config_key(cfg)
            report.write(f"\n  [{i}/{len(configs)}] {ckey}\n")
            report.flush()
            t0 = time.monotonic()
            result = timer(cfg)
            elapsed = time.monotonic() - t0
            trial = {
                "trial_num":   i,
                "stage":       "exhaustive",
                "config":      cfg,
                "config_key":  ckey,
                "timing_ms":   result["timing_ms"] if result else None,
                "min_ms":      result["min_ms"]    if result else None,
                "max_ms":      result["max_ms"]    if result else None,
                "n":           result["n"]         if result else None,
                "host":        _current_host(),
                "recorded_at": datetime.datetime.now().isoformat(),
                "status":      "ok" if result else "fail",
                "build_s":     elapsed,
            }
            cache.record_trial(spec.optimizer, spec.model, spec.arch, trial)
            if result is not None:
                ms = result["timing_ms"]
                report.write(f"    median={ms:.4f}ms\n")
                if best is None or ms < (best.get("timing_ms") or float("inf")):
                    best = {**cfg, "timing_ms": ms, "config_key": ckey}
            else:
                report.write(f"    FAIL ({elapsed:.1f}s)\n")
            step(f"best={best['timing_ms']:.3f}ms" if best else "no winner yet")
            if (i % JIT_CACHE_FLUSH_EVERY) == 0:
                cache.save()
    finally:
        close()
    if best is None:
        report.write("\n  [exhaustive] no successful variants — "
                     "leaving tuned_config unset.\n")
        return None
    report.write(f"\n  [exhaustive] WINNER: {best['config_key']} "
                 f"@ {best['timing_ms']:.4f}ms\n")
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, best,
                    mode="exhaustive", search_space_hash=space_hash)
    return best


def _run_bayesian(spec: BuildSpec, prefiltered: List[Dict[str, Any]],
                  space: Dict[str, Any], dims: List[Dict[str, Any]],
                  timer, cache: CompileCache, space_hash: str,
                  report) -> Optional[Dict[str, Any]]:
    from grokking_optimizers.bayesian import (
        run_bayesian, topk_refine, pick_winner)

    n_trials = spec.bayesian_trials
    report.write(f"\n  [bayesian] TPE stage with n_trials={n_trials}, "
                 f"seed={spec.seed}, pruner={spec.pruner}\n")

    # §12 A2 — Transfer learning: seed from sibling-optimizer trials.
    seed_trials: Optional[List[Dict[str, Any]]] = None
    if spec.transfer_learning:
        siblings = _collect_sibling_trials(cache, spec.optimizer,
                                           spec.model, spec.arch)
        if siblings:
            report.write(f"  [bayesian] transfer-learning: seeding from "
                         f"{len(siblings)} sibling-optimizer trials\n")
            seed_trials = siblings
        else:
            report.write("  [bayesian] transfer-learning: no sibling "
                         "trials available; cold-starting\n")

    # Optuna study persistence (resumable across runs).
    storage = (spec.out_dir
               / f"optuna_{spec.optimizer}_{spec.model}_{spec.arch}.db")
    step1, close1 = make_progress(
        n_trials, f"jit-tpe {spec.optimizer}/{spec.arch}")

    def progress1(done, total, cfg):
        step1(f"trial {done}/{total} key={_ss.config_key(cfg)[:24]}…")

    try:
        tpe_trials = run_bayesian(
            spec.arch, space, n_trials=n_trials, seed=spec.seed,
            storage=storage,
            study_name=f"sg_{spec.optimizer}_{spec.model}_{spec.arch}",
            timer=timer, progress=progress1, host=_current_host(),
            prefiltered=prefiltered,
            pruner=spec.pruner,
            seed_trials=seed_trials,
        )
    finally:
        close1()

    for t in tpe_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
    cache.save()

    report.write(f"  [bayesian] TPE produced {len(tpe_trials)} trials; "
                 f"{sum(1 for t in tpe_trials if t['timing_ms'] is not None)} "
                 "succeeded.\n")

    # Stage 2: refine the top-K with ±2-step neighbours.
    report.write(f"\n  [bayesian] refine stage top_k={spec.top_k}\n")
    refine_inputs = [t for t in tpe_trials if t["timing_ms"] is not None]
    refine_inputs.sort(key=lambda t: t["timing_ms"])
    n_refine_est = sum(1 for _ in _neighbour_estimate(
        refine_inputs[:spec.top_k], dims, prefiltered))
    step2, close2 = make_progress(
        max(n_refine_est, 1), f"jit-refine {spec.optimizer}/{spec.arch}")

    def progress2(done, total, cfg):
        step2(f"refine {done}/{total} key={_ss.config_key(cfg)[:24]}…")

    try:
        refine_trials = topk_refine(
            tpe_trials, space, spec.arch,
            top_k=spec.top_k, timer=timer,
            progress=progress2, host=_current_host(),
            prefiltered=prefiltered,
        )
    finally:
        close2()
    for t in refine_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
    cache.save()

    winner = pick_winner(tpe_trials + refine_trials)
    if winner is None:
        report.write("\n  [bayesian] no successful trials — "
                     "leaving tuned_config unset.\n")
        return None
    out = dict(winner["config"])
    out["timing_ms"] = winner["timing_ms"]
    out["config_key"] = _ss.config_key(out)
    out["stage_won"] = winner["stage"]
    report.write(f"\n  [bayesian] WINNER ({winner['stage']}): "
                 f"{out['config_key']} @ {out['timing_ms']:.4f}ms\n")
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, out,
                    mode="bayesian", search_space_hash=space_hash)
    return out


def _neighbour_estimate(seeds: List[Dict[str, Any]],
                        dims: List[Dict[str, Any]],
                        prefiltered: List[Dict[str, Any]]):
    """Estimate the refine-stage trial count (best-effort, for progress bar)."""
    feasible = {_ss.config_key(c) for c in prefiltered}
    seen = set()
    from grokking_optimizers.bayesian import _step_neighbours
    for s in seeds:
        base = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in s["config"].items()}
        for d in dims:
            for nb in _step_neighbours(base.get(d["name"]), d["values"], 2):
                cfg = dict(base)
                cfg[d["name"]] = nb
                k = _ss.config_key(cfg)
                if k in seen or (feasible and k not in feasible):
                    continue
                seen.add(k)
                yield cfg


# ---------------------------------------------------------------------------
# tuned_configs.h — written from the cache's winning config
# ---------------------------------------------------------------------------

def _write_tuned_configs_header(combo: Dict[str, Any], optimizer: str,
                                model: str, arch: str, report) -> Path:
    tuned_h = REPO_ROOT / "csrc/algorithms/tuned_configs.h"
    tuned_h.parent.mkdir(parents=True, exist_ok=True)
    macros: List[str] = []
    # Try to load the space to map dim -> macro
    try:
        space = _ss.load_yaml(DEFAULT_SEARCH_SPACE)
        dims = space.get(arch, {}).get("dims", [])
    except Exception:
        dims = []
    name_to_macro = {d["name"]: d.get("macro") for d in dims}
    for k, v in combo.items():
        if k in ("timing_ms", "config_key", "stage_won"):
            continue
        macro = name_to_macro.get(k)
        if not macro:
            # Fallback for back-compat keys (block / vec / unroll).
            backcompat = {
                "block":  "SG_TUNED_BLOCK_SIZE",
                "vec":    "SG_TUNED_VEC_WIDTH",
                "unroll": "SG_TUNED_UNROLL",
            }
            macro = backcompat.get(k)
        if macro:
            macros.append((macro, v))
    body = ["// Auto-generated by grokking_optimizers.compile JIT autotune.",
            "// Do not edit by hand — re-run with --jit-only to refresh.",
            f"// Winning combo: optimizer={optimizer} model={model} arch={arch}",
            f"// Median timing: {combo.get('timing_ms', 0.0):.4f} ms",
            f"// Generated: {datetime.datetime.now().isoformat()}",
            "#pragma once"]
    for macro, value in macros:
        if isinstance(value, bool):
            val_text = "1" if value else "0"
        elif isinstance(value, tuple):
            val_text = "{" + ", ".join(str(x) for x in value) + "}"
        else:
            val_text = str(value)
        body.append(f"#ifndef {macro}")
        body.append(f"#define {macro} {val_text}")
        body.append("#endif")
    tuned_h.write_text("\n".join(body) + "\n")
    report.write(f"  [tuned_configs.h] wrote {tuned_h.relative_to(REPO_ROOT)}\n")
    return tuned_h


# ---------------------------------------------------------------------------
# AOT and JIT halves (importable; called by main() in single-phase mode)
# ---------------------------------------------------------------------------

def build_aot(spec: BuildSpec, cache: CompileCache, report) -> Optional[Path]:
    """Run only the AOT portion of the build. No GPU access needed.

    When ``spec.pgo`` is True, runs the 3-pass instrument → workload →
    use loop. Otherwise a single AOT build."""
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        launcher = (REPO_ROOT / "csrc/backends/pallas"
                    / f"launch_{spec.optimizer}.py")
        report.write("\n[pallas] no C++ compile; Python launcher only\n")
        report.write(f"  launcher: {launcher}\n")
        report.write(f"  exists:   {launcher.exists()}\n")
        return None

    sources = _resolve_sources(spec)
    host_cflags = _host_cflags(spec)
    device_cflags = _device_cflags(spec)
    ldflags = _ldflags(spec)
    source_hash = _hash_sources(sources) if sources else "pallas"
    host_hash = _hash_flags(host_cflags)
    device_hash = _hash_flags(device_cflags)

    # Resolve search-space hash (gates AOT freshness too)
    space_hash = None
    yaml_path = spec.search_space_path or DEFAULT_SEARCH_SPACE
    try:
        space_hash = _ss.hash_space(_ss.load_yaml(yaml_path), spec.arch)
    except Exception as exc:
        report.write(f"  [search-space] could not hash: {exc}\n")

    workload_hash = None
    if spec.pgo:
        workload = spec.pgo_workload or DEFAULT_PGO_WORKLOAD
        workload_hash = _pgo.hash_workload(workload, spec.pgo_steps)
        report.write(f"  [pgo] workload={workload} steps={spec.pgo_steps} "
                     f"hash={workload_hash[:16]}\n")

    if cache.is_aot_fresh(spec.optimizer, spec.model, spec.arch,
                          source_hash, host_hash, device_hash,
                          pgo_enabled=spec.pgo,
                          pgo_workload_hash=workload_hash,
                          search_space_hash=space_hash):
        art = cache.get(spec.optimizer, spec.model, spec.arch)["primary_artifact"]
        so_path = Path(art["path"])
        report.write(f"  [aot cache HIT] reusing {so_path} "
                     f"(size={art['size']} "
                     f"sha256={art['sha256'][:16]}…)\n")
        return so_path

    if spec.pgo:
        return _build_aot_pgo(spec, cache, sources, host_cflags, device_cflags,
                              ldflags, source_hash, host_hash, device_hash,
                              workload_hash, space_hash, report)

    report.write("  [aot cache MISS] building primary artefact...\n")
    so_path = _torch_load(spec, sources, host_cflags, device_cflags, ldflags,
                          report)
    if so_path is None:
        return None
    so_path = _publish_aot_artifact(spec, so_path, report)
    cache.record_aot(
        spec.optimizer, spec.model, spec.arch,
        source_hash=source_hash,
        host_flags_hash=host_hash,
        device_flags_hash=device_hash,
        so_path=so_path,
        pgo_enabled=False,
        search_space_hash=space_hash,
    )
    cache.save()
    return so_path


def _build_aot_pgo(spec: BuildSpec, cache: CompileCache, sources: List[Path],
                   host_cflags: List[str], device_cflags: List[str],
                   ldflags: List[str], source_hash: str, host_hash: str,
                   device_hash: str, workload_hash: str,
                   space_hash: Optional[str], report) -> Optional[Path]:
    """Three-pass PGO loop: instrument → collect → use."""
    profile_dir = spec.out_dir / "pgo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: instrumented build ────────────────────────────────────
    report.write("\n  [pgo 1/3] building instrumented .so\n")
    inst_host, inst_device, inst_ld = _pgo.instrument_flags(
        spec.arch, profile_dir, host_cflags, device_cflags, ldflags)
    inst_spec = BuildSpec(**{**spec.__dict__})
    inst_spec.extra_macros = list(spec.extra_macros) + ["-DSG_PGO_INSTRUMENT=1"]
    inst_so = _torch_load(inst_spec, sources, inst_host, inst_device,
                          inst_ld, report, module_suffix="_pgo_instrument")
    if inst_so is None:
        report.write("  [pgo 1/3] instrumented build FAILED\n")
        return None

    # ── Pass 2: collect workload ──────────────────────────────────────
    report.write("\n  [pgo 2/3] running workload to collect profile data\n")
    workload = spec.pgo_workload or DEFAULT_PGO_WORKLOAD
    ok = _pgo.collect_workload(
        workload,
        so_path=inst_so,
        opt_class=OPT_CLASS[spec.optimizer],
        model=spec.model,
        arch=spec.arch,
        profile_dir=profile_dir,
        steps=spec.pgo_steps,
        report=report,
    )
    if not ok:
        report.write("  [pgo 2/3] workload FAILED or produced no profile data\n")
        return None
    cache.record_pgo(spec.optimizer, spec.model, spec.arch,
                     profile_dir=profile_dir, workload_hash=workload_hash)
    cache.save()

    # ── Pass 3: profile-use build ─────────────────────────────────────
    report.write("\n  [pgo 3/3] rebuilding with -fprofile-use\n")
    use_host, use_device, use_ld = _pgo.use_flags(
        spec.arch, profile_dir, host_cflags, device_cflags, ldflags)
    so_path = _torch_load(spec, sources, use_host, use_device, use_ld,
                          report, module_suffix="_pgo")
    if so_path is None:
        report.write("  [pgo 3/3] profile-use build FAILED\n")
        return None
    so_path = _publish_aot_artifact(spec, so_path, report)
    cache.record_aot(
        spec.optimizer, spec.model, spec.arch,
        source_hash=source_hash,
        host_flags_hash=_hash_flags(use_host),
        device_flags_hash=_hash_flags(use_device),
        so_path=so_path,
        pgo_enabled=True,
        pgo_profile_dir=profile_dir,
        pgo_workload_hash=workload_hash,
        search_space_hash=space_hash,
    )
    cache.save()
    return so_path


def _publish_aot_artifact(spec: BuildSpec, so_path: Path, report) -> Path:
    """If ``--aot-artifact-dir`` is set, copy the .so into the shared dir
    so a downstream JIT runtime (possibly on another host) can pick it up.
    Returns the published path (or the original)."""
    if spec.aot_artifact_dir is None:
        return so_path
    dest_dir = Path(spec.aot_artifact_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / so_path.name
    if dest.resolve() != so_path.resolve():
        try:
            shutil.copy2(so_path, dest)
        except OSError as exc:
            report.write(f"  [aot publish] copy failed: {exc} (keeping {so_path})\n")
            return so_path
        report.write(f"  [aot publish] {so_path} -> {dest}\n")
    return dest


def build_jit(spec: BuildSpec, cache: CompileCache, report) -> Optional[Path]:
    """Run only the JIT autotune + final-link half. Requires GPU."""
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        report.write("\n[pallas] JIT phase no-op (Python-only backend)\n")
        return None

    # Sanity: AOT must have run already on some host with matching hashes.
    e = cache.get(spec.optimizer, spec.model, spec.arch)
    if not e["aot_completed_at"]:
        report.write("  [jit] cache has no AOT entry; running AOT now.\n")
        so_aot = build_aot(spec, cache, report)
        if so_aot is None:
            return None

    sources = _resolve_sources(spec)
    host_cflags = _host_cflags(spec)
    device_cflags = _device_cflags(spec)
    ldflags = _ldflags(spec)

    tuned = _jit_autotune(spec, sources, host_cflags, device_cflags, ldflags,
                          cache, report)
    cache.save()

    if tuned is None:
        report.write("  [jit] no tuned config — keeping AOT primary artefact.\n")
        # Return the AOT primary artefact if present.
        art = e.get("primary_artifact")
        return Path(art["path"]) if art else None

    # Final pass: rebuild with tuned macros baked in.
    space_hash = e.get("search_space_hash")
    _write_tuned_configs_header(tuned, spec.optimizer, spec.model, spec.arch,
                                report)

    # Try to assemble macros via the resolved YAML space; fall back to
    # the tuned dict's literal keys.
    extra_host: List[str] = []
    extra_device: List[str] = []
    try:
        space = _ss.load_yaml(spec.search_space_path or DEFAULT_SEARCH_SPACE)
        dims = space.get(spec.arch, {}).get("dims", [])
        extra_host = _variant_macros(tuned, dims, "host")
        extra_device = _variant_macros(tuned, dims, "device")
    except Exception:
        for k, v in tuned.items():
            if k in ("timing_ms", "config_key", "stage_won"):
                continue
            macro = {"block": "SG_TUNED_BLOCK_SIZE",
                     "vec": "SG_TUNED_VEC_WIDTH",
                     "unroll": "SG_TUNED_UNROLL"}.get(k)
            if macro:
                flag = f"-D{macro}={v}"
                extra_host.append(flag)
                extra_device.append(flag)

    so_path = _torch_load(spec, sources,
                          host_cflags + extra_host,
                          device_cflags + extra_device,
                          ldflags, report, module_suffix="_tuned")
    if so_path is not None:
        so_path = _publish_aot_artifact(spec, so_path, report)
        # Record this as the "current primary" — its hashes include the tuned macros
        cache.record_aot(
            spec.optimizer, spec.model, spec.arch,
            source_hash=_hash_sources(sources),
            host_flags_hash=_hash_flags(host_cflags + extra_host),
            device_flags_hash=_hash_flags(device_cflags + extra_device),
            so_path=so_path,
            pgo_enabled=bool(e.get("pgo_enabled")),
            pgo_workload_hash=e.get("pgo_workload_hash"),
            search_space_hash=space_hash,
        )
        cache.save()
    return so_path


# ---------------------------------------------------------------------------
# Public entry — orchestrates AOT and JIT (in-process or via subprocess split)
# ---------------------------------------------------------------------------

def build(
    optimizer: str,
    model: str,
    arch: str,
    *,
    cache: Optional[CompileCache] = None,
    cache_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    aot_only: bool = False,
    jit_only: bool = False,
    runtime: str = "both",
    autotune: bool = True,
    autotune_mode: str = "bayesian",
    profile: bool = True,
    report_path: Optional[Path] = None,
    verbose: bool = False,
    extra_macros: Optional[Iterable[str]] = None,
    search_space_path: Optional[Path] = None,
    aot_artifact_dir: Optional[Path] = None,
    pgo: bool = False,
    pgo_workload: Optional[Path] = None,
    pgo_steps: int = 1000,
    bayesian_trials: int = 500,
    top_k: int = 20,
    seed: int = 0,
    debug_symbols: bool = False,
    pruner: str = "none",
    transfer_learning: bool = False,
) -> Optional[Path]:
    """In-process orchestrator. ``main`` handles subprocess split.

    When called from Python, this does not fork: AOT and JIT run in the
    same process. Use ``main(['--runtime', 'both', ...])`` for the
    subprocess-isolated workflow."""
    if aot_only:
        runtime = "aot"
    if jit_only:
        runtime = "jit"

    spec = BuildSpec(
        optimizer=optimizer, model=model, arch=arch,
        out_dir=(out_dir or (REPO_ROOT / "build" / "compiled")).resolve(),
        autotune=autotune, autotune_mode=autotune_mode, profile=profile,
        verbose=verbose, extra_macros=list(extra_macros or []),
        runtime=runtime,
        aot_only=(runtime == "aot"),
        jit_only=(runtime == "jit"),
        search_space_path=search_space_path,
        aot_artifact_dir=aot_artifact_dir,
        pgo=pgo, pgo_workload=pgo_workload, pgo_steps=pgo_steps,
        bayesian_trials=bayesian_trials, top_k=top_k, seed=seed,
        debug_symbols=debug_symbols,
        pruner=pruner, transfer_learning=transfer_learning,
    )
    _validate(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if cache is None:
        cp = Path(cache_path) if cache_path else (spec.out_dir / DEFAULT_CACHE_NAME)
        cache = CompileCache(cp)

    report_path = report_path or (
        spec.out_dir / f"compile_{optimizer}_{model}_{arch}.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    info = ARCH_INFO[arch]
    phases = ["resolve"]
    if info["vendor"] != "pallas":
        if runtime in ("aot", "both"):
            phases.append("aot")
        if runtime in ("jit", "both") and autotune:
            phases.append("jit-autotune")
        if runtime in ("jit", "both"):
            phases.append("final")
    if profile and runtime != "aot":
        phases.append("profile")
    step, close = make_progress(len(phases), f"{optimizer}/{model}/{arch}")
    so_path: Optional[Path] = None

    try:
        with open(report_path, "w") as report:
            report.write("# grokking_optimizers.compile — targeted build\n")
            report.write(f"# Generated:        {datetime.datetime.now().isoformat()}\n")
            report.write(f"# Optimizer:        {optimizer}\n")
            report.write(f"# Model:            {model}\n")
            report.write(f"# Arch:             {arch} (vendor={info['vendor']})\n")
            report.write(f"# CPU cores:        {NCPUS} (ninja -j)\n")
            report.write(f"# Out dir:          {spec.out_dir}\n")
            report.write(f"# Cache:            {cache.path}\n")
            report.write(f"# Runtime:          {runtime}\n")
            report.write(f"# Autotune:         {autotune} (mode={autotune_mode})\n")
            report.write(f"# Bayesian trials:  {bayesian_trials} (top_k={top_k})\n")
            report.write(f"# PGO:              {pgo}\n")
            if pgo:
                report.write(f"#   workload:       {pgo_workload or DEFAULT_PGO_WORKLOAD}\n")
                report.write(f"#   steps:          {pgo_steps}\n")
            report.write(f"# Search space:     {search_space_path or DEFAULT_SEARCH_SPACE}\n")
            report.write(f"# AOT artefact dir: {aot_artifact_dir}\n")
            report.write(f"# Debug symbols:    {debug_symbols}\n")
            report.write(f"# Profile:          {profile}\n")
            step("resolve")

            if info["vendor"] == "pallas":
                build_aot(spec, cache, report)  # logs pallas no-op
            else:
                if runtime in ("aot", "both"):
                    report.write("\n--- AOT PHASE ---\n")
                    so_path = build_aot(spec, cache, report)
                    step("aot")
                if runtime in ("jit", "both") and autotune:
                    report.write("\n--- JIT AUTOTUNE PHASE ---\n")
                    so_path = build_jit(spec, cache, report) or so_path
                    step("jit-autotune")
                if runtime in ("jit", "both"):
                    if "final" in phases:
                        step("final")

            if profile and runtime != "aot":
                report.write("\n--- PROFILE PASS ---\n")
                _dispatch_profile(optimizer, model, arch, report)
                step("profile")

            report.write(f"\n# Cache:    {cache.path}\n")
            report.write(f"# Final .so: {so_path}\n")
    finally:
        cache.save()
        close()
    return so_path


# ---------------------------------------------------------------------------
# CLI — runtime split happens here
# ---------------------------------------------------------------------------

def _spawn_phase(argv: List[str], phase: str) -> int:
    """Re-exec self with the chosen phase. Returns the subprocess returncode."""
    phase_flag = {"aot": "--aot-only", "jit": "--jit-only"}[phase]
    # Strip any prior --runtime / --aot-only / --jit-only flags from the
    # user's argv so we don't double-spec.
    cleaned: List[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--runtime",):
            skip_next = True
            continue
        if tok in ("--aot-only", "--jit-only"):
            continue
        if tok.startswith("--runtime="):
            continue
        cleaned.append(tok)
    cmd = [sys.executable, "-m", "grokking_optimizers.compile",
           phase_flag] + cleaned
    sys.stdout.write(f"[runtime split] spawning {phase}: {' '.join(cmd)}\n")
    sys.stdout.flush()
    return subprocess.call(cmd, env=os.environ.copy())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.compile",
        description="Targeted per-(optimizer, model, arch) build pipeline. "
                    "v3 cache · YAML search space · Bayesian/Exhaustive · PGO · "
                    "split AOT/JIT runtimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--optimizer", "-O", required=True, choices=OPTIMIZERS,
                        help="Optimizer name (csrc/algorithms/<name>.h)")
    parser.add_argument("--model", "-M", required=True, choices=MODELS,
                        help="Model name (csrc/backends/*/models/<name>.*)")
    parser.add_argument("--arch", "-A", required=True, choices=ARCHES,
                        help="Target arch")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "build" / "compiled",
                        help="Build artifact directory (default: build/compiled)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Cache file path (default: <out>/.compile_cache.json).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Report file path (default: <out>/compile_<O>_<M>_<A>.txt)")

    # ── Runtime / phase ──────────────────────────────────────────────
    parser.add_argument("--runtime", choices=("aot", "jit", "both"),
                        default="both",
                        help="Which phase to run in this process. 'both' "
                             "spawns AOT then JIT subprocesses sequentially.")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--aot-only", action="store_true",
                       help="Alias for --runtime aot.")
    phase.add_argument("--jit-only", action="store_true",
                       help="Alias for --runtime jit.")

    # ── Autotune mode ────────────────────────────────────────────────
    parser.add_argument("--mode", choices=("exhaustive", "bayesian"),
                        default="bayesian",
                        help="Autotune mode (default: bayesian).")
    parser.add_argument("--bayesian-trials", type=int, default=500,
                        help="Trials for Bayesian TPE stage (default: 500).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K winners refined in the second stage "
                             "(default: 20).")
    parser.add_argument("--quick", action="store_true",
                        help=f"Debug shortcut: bayesian mode with "
                             f"{QUICK_BAYESIAN_TRIALS} trials.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bayesian sampler seed (default: 0).")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Skip JIT autotune even when a GPU is visible.")
    # §12 A1 — Hyperband / median pruner option
    parser.add_argument("--pruner", choices=("none", "median", "hyperband"),
                        default="none",
                        help="Optuna pruner. Default: none. 'hyperband' "
                             "enables Successive Halving (§12 A1).")
    # §12 A2 — Transfer learning seeding
    parser.add_argument("--transfer-learning", action="store_true",
                        help="Seed Bayesian TPE from sibling-optimizer "
                             "trials on the same (model, arch). §12 A2.")

    # ── Search space + PGO ──────────────────────────────────────────
    parser.add_argument("--search-space", type=Path,
                        default=DEFAULT_SEARCH_SPACE,
                        help="YAML search-space file "
                             "(default: configs/search_space.yaml).")
    parser.add_argument("--pgo", action="store_true",
                        help="Enable 3-pass PGO loop (instrument → "
                             "workload → use). Doubles AOT compile time.")
    parser.add_argument("--pgo-workload", type=Path,
                        default=DEFAULT_PGO_WORKLOAD,
                        help="PGO workload script.")
    parser.add_argument("--pgo-steps", type=int, default=1000,
                        help="N optimizer.step() calls during profile "
                             "collection (default: 1000).")

    # ── Cross-host artefact dir ─────────────────────────────────────
    parser.add_argument("--aot-artifact-dir", type=Path, default=None,
                        help="Directory to publish the AOT .so to so a "
                             "JIT host on another machine can pick it up.")

    # ── Misc ────────────────────────────────────────────────────────
    parser.add_argument("--no-profile", action="store_true",
                        help="Skip ncu/rocprof/jax.profiler pass.")
    parser.add_argument("--debug-symbols", action="store_true",
                        help="Add -ggdb / -lineinfo (auto-on with --profile).")
    parser.add_argument("-D", dest="extra_macros", action="append", default=[],
                        help="Extra -D<MACRO[=VALUE]>. Repeatable.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose torch cpp_extension output")
    args = parser.parse_args(argv)

    # Resolve aliases.
    if args.aot_only:
        args.runtime = "aot"
    if args.jit_only:
        args.runtime = "jit"
    if args.quick:
        args.mode = "bayesian"
        args.bayesian_trials = QUICK_BAYESIAN_TRIALS

    # ── Runtime split: spawn AOT then JIT, then return ──────────────
    if args.runtime == "both":
        argv_in = list(argv) if argv is not None else sys.argv[1:]
        rc_aot = _spawn_phase(argv_in, "aot")
        if rc_aot != 0:
            sys.stderr.write(f"[runtime split] AOT subprocess returned "
                             f"{rc_aot}; skipping JIT.\n")
            return rc_aot
        rc_jit = _spawn_phase(argv_in, "jit")
        return rc_jit

    extra = [m if m.startswith("-D") else f"-D{m}"
             for m in args.extra_macros]

    so = build(
        optimizer=args.optimizer, model=args.model, arch=args.arch,
        cache_path=args.cache,
        out_dir=args.out,
        aot_only=(args.runtime == "aot"),
        jit_only=(args.runtime == "jit"),
        runtime=args.runtime,
        autotune=not args.no_autotune,
        autotune_mode=args.mode,
        profile=not args.no_profile,
        report_path=args.report,
        verbose=args.verbose,
        extra_macros=extra,
        search_space_path=args.search_space,
        aot_artifact_dir=args.aot_artifact_dir,
        pgo=args.pgo,
        pgo_workload=args.pgo_workload,
        pgo_steps=args.pgo_steps,
        bayesian_trials=args.bayesian_trials,
        top_k=args.top_k,
        seed=args.seed,
        debug_symbols=args.debug_symbols,
        pruner=args.pruner,
        transfer_learning=args.transfer_learning,
    )

    report = args.report or (
        args.out / f"compile_{args.optimizer}_{args.model}_{args.arch}.txt")
    sys.stdout.write(f"{report}\n")
    return 0 if (so is not None
                 or ARCH_INFO[args.arch]["vendor"] == "pallas"
                 or args.runtime == "aot") else 1


if __name__ == "__main__":
    raise SystemExit(main())
