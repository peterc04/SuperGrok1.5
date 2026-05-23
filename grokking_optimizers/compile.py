"""grokking_optimizers.compile — Targeted per-(optimizer, model, arch) build
with a portable AOT-then-JIT cache.

Companion to ``setup.py``. Where ``pip install -e .`` builds the full
``grokking_optimizers._ops`` extension with the default arch flags, this
module compiles a focused, maximally-optimised build for a single
``(optimizer, model, arch)`` triple. The pipeline is split into two halves
so a CPU-only host can do the heavy AOT compile and ship a cache file to
the target GPU host, which then does the JIT autotune sweep.

Two-phase pipeline
==================

**AOT phase (runs on any host with the toolchain — no GPU required).**

  1. Resolve sources for the chosen arch (bindings + every launcher + every
     model TU = 18 files).
  2. Hash the source set, the host cflags, and the device cflags.
  3. Look up ``(optimizer, model, arch)`` in the cache file. If all three
     hashes match an entry with ``aot_completed_at != None`` and the
     recorded primary artefact still exists with the recorded size →
     **cache hit**, skip rebuild and reuse the .so.
  4. Otherwise build with ``torch.utils.cpp_extension.load`` (ninja,
     ``MAX_JOBS=$(nproc)``), arch-tuned codegen + LTO, and the
     ``SG_BUILD_*`` metaprog macros so headers can ``#if`` out unused
     specialisations.
  5. Record the artefact path, size, mtime, and SHA-256 in the cache.
     Mark ``aot_completed_at``. Save cache to disk.

  The AOT artefact is portable in *knowledge* (hashes, tuned configs)
  across machines; the ``.so`` path itself is host-local and gets
  re-validated on each run.

**JIT autotune phase (runs only when ``torch.cuda.is_available()``).**

  1. Choose a search space: ``--exhaustive`` (75 configs),
     ``--default`` (36 configs), or ``--quick`` (8 configs).
  2. For each ``(block_size, vec_width, unroll)`` combo:
       - Build a variant .so with
         ``-DSG_TUNED_BLOCK_SIZE=B -DSG_TUNED_VEC_WIDTH=V -DSG_TUNED_UNROLL=U``
         (each variant gets its own module name and build_directory so the
         ninja per-file cache hits on the second+ build). The cache also
         remembers which variant configs have been built before.
       - Load that .so as ``grokking_optimizers._ops`` into a fresh
         subprocess, run ``opt.step()`` on a 4096×4096 tensor, time it
         with ``torch.cuda.Event``, take the median of N iterations.
       - Record ``config, timing_ms`` in ``cache.sweep_history``.
  3. Pick the winning combo (lowest median timing). Set
     ``cache.tuned_config``. Mark ``jit_completed_at``.
  4. Rewrite ``csrc/algorithms/tuned_configs.h`` with the winning macros
     so downstream consumers (setup.py builds, IDE Intellisense) pick up
     the tuned defaults.
  5. Rebuild the primary artefact with the tuned configs baked in.
  6. Save cache to disk.

The cache is loaded once at the start of ``build()`` into an in-memory
dict and mutated throughout — no per-step file I/O — then written back
atomically at the end. Pass ``--cache <path>`` to use a specific cache
file; the default lives at ``<out>/.compile_cache.json``.

Cache portability
=================

The on-disk format is JSON keyed by ``optimizer/model/arch``. Each entry
stores:

    source_hash        SHA-256 of the source-set contents and order.
    host_cflags_hash   SHA-256 of the host C++ flag string.
    device_cflags_hash SHA-256 of the CUDA/HIP flag string.
    primary_artifact   {path, size, mtime, sha256}  — host-local.
    variant_artifacts  {config_key -> {path, size, mtime}} — host-local.
    sweep_history      [{config, timing_ms, host, recorded_at}]
    tuned_config       {block, vec, unroll, timing_ms} or None.
    aot_completed_at   ISO timestamp or None.
    jit_completed_at   ISO timestamp or None.
    aot_host           {platform, python, torch, cuda, hip, ncpus}
    jit_host           same shape, populated on the GPU host

When the cache moves to a new machine, the hashes + tuned configs +
sweep history travel; the local artefact paths get rebuilt on first AOT
run there. JIT-sweep results carry across hosts of the same arch, so a
machine that's already swept doesn't need to repeat the sweep — a fresh
GPU host of the same arch reuses the winning config directly.

Usage
=====

CLI::

    # End-to-end on a GPU host (AOT + JIT autotune + profile)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json --exhaustive

    # AOT-only on a CPU host (writes cache; no GPU needed)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json --aot-only

    # JIT autotune only on the GPU host (consumes the cache from the CPU host)
    python -m grokking_optimizers.compile \\
        --optimizer supergrok2 --model mamba --arch sm_90 \\
        --cache build/.compile_cache.json --jit-only

Importable::

    from grokking_optimizers.compile import build, CompileCache
    cache = CompileCache(Path("build/.compile_cache.json"))
    so_path = build(optimizer="supergrok2", model="mamba", arch="sm_90",
                    cache=cache, autotune_mode="exhaustive")
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


CACHE_VERSION = 2
DEFAULT_CACHE_NAME = ".compile_cache.json"

# Per-arch autotune search spaces. The CDNA3 wavefront is 64 so larger
# blocks pay off; sm_90's warp is 32 so 128–512 hits the sweet spot.
# Vec widths reflect the available vector-load width on each arch.
EXHAUSTIVE_SPACE = {
    "sm_90":  {"block": [64, 128, 256, 512, 1024],
               "vec":   [1, 2, 4],
               "unroll":[1, 2, 4, 8, 16]},
    "gfx942": {"block": [64, 128, 256, 512, 1024],
               "vec":   [1, 2, 4],
               "unroll":[1, 2, 4, 8, 16]},
}

DEFAULT_SPACE = {
    "sm_90":  {"block": [128, 256, 512, 1024],
               "vec":   [1, 2, 4],
               "unroll":[4, 8, 16]},
    "gfx942": {"block": [128, 256, 512, 1024],
               "vec":   [1, 2, 4],
               "unroll":[4, 8, 16]},
}

QUICK_SPACE = {
    "sm_90":  {"block": [128, 256], "vec": [2, 4], "unroll": [4, 8]},
    "gfx942": {"block": [256, 512], "vec": [2, 4], "unroll": [4, 8]},
}


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
        rel = str(p.relative_to(REPO_ROOT) if str(p).startswith(str(REPO_ROOT))
                  else p)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(p).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_flags(flags: List[str]) -> str:
    return _sha256_str("\0".join(flags))


# ---------------------------------------------------------------------------
# Persistent cache (in-memory dict, JSON on disk, atomic save)
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


class CompileCache:
    """Persistent build cache.

    The cache lives in-memory as a Python dict for the duration of a
    ``build()`` call and is written back to disk atomically at the end
    (or via explicit ``.save()``). Mutations are tracked via the
    ``_dirty`` flag so a no-op build doesn't touch the file.

    Schema is JSON; see module docstring for the field list. Cross-host
    portability: source / flag hashes and sweep history travel; local
    artefact paths get re-validated on each host.
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
        if data.get("version") != CACHE_VERSION:
            backup = self.path.with_suffix(
                self.path.suffix + f".v{data.get('version', 0)}.bak")
            try:
                self.path.rename(backup)
            except OSError:
                pass
            data = self._fresh()
            self._dirty = True
            return data
        # Append current host to provenance trail.
        data.setdefault("host_history", []).append(_current_host())
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
        """Get-or-create the entry for this combo. Returns a live dict
        reference; mutations propagate."""
        with self._lock:
            k = self.key(opt, model, arch)
            entry = self._data["entries"].setdefault(k, {
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
            })
            self._dirty = True
            return entry

    def is_aot_fresh(self, opt: str, model: str, arch: str,
                     source_hash: str, host_flags_hash: str,
                     device_flags_hash: str) -> bool:
        e = self.get(opt, model, arch)
        if not e["aot_completed_at"]:
            return False
        if e["source_hash"] != source_hash:
            return False
        if e["host_cflags_hash"] != host_flags_hash:
            return False
        if e["device_cflags_hash"] != device_flags_hash:
            return False
        art = e["primary_artifact"]
        if not art:
            return False
        p = Path(art["path"])
        try:
            return p.exists() and p.stat().st_size == art.get("size")
        except OSError:
            return False

    def is_jit_fresh(self, opt: str, model: str, arch: str) -> bool:
        e = self.get(opt, model, arch)
        return bool(e["jit_completed_at"] and e["tuned_config"])

    def record_aot(self, opt: str, model: str, arch: str, *,
                   source_hash: str, host_flags_hash: str,
                   device_flags_hash: str, so_path: Optional[Path]) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["source_hash"]        = source_hash
            e["host_cflags_hash"]   = host_flags_hash
            e["device_cflags_hash"] = device_flags_hash
            if so_path is not None and so_path.exists():
                stat = so_path.stat()
                e["primary_artifact"] = {
                    "path":   str(so_path),
                    "size":   stat.st_size,
                    "mtime":  stat.st_mtime,
                    "sha256": _sha256_file(so_path),
                }
            e["aot_completed_at"] = datetime.datetime.now().isoformat()
            e["aot_host"]         = _current_host()
            self._dirty = True

    def record_variant(self, opt: str, model: str, arch: str,
                       config_key: str, so_path: Optional[Path]) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            if so_path is not None and so_path.exists():
                stat = so_path.stat()
                e["variant_artifacts"][config_key] = {
                    "path":  str(so_path),
                    "size":  stat.st_size,
                    "mtime": stat.st_mtime,
                }
                self._dirty = True

    def record_sweep(self, opt: str, model: str, arch: str, *,
                     config: dict, timing_ms: float, **extras) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["sweep_history"].append({
                "config":       config,
                "timing_ms":    timing_ms,
                "host":         _current_host(),
                **extras,
            })
            self._dirty = True

    def set_tuned(self, opt: str, model: str, arch: str,
                  config: dict) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["tuned_config"]      = config
            e["jit_completed_at"]  = datetime.datetime.now().isoformat()
            e["jit_host"]          = _current_host()
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
    autotune_mode: str = "default"   # "quick" | "default" | "exhaustive"
    profile: bool = True
    verbose: bool = False
    extra_macros: List[str] = field(default_factory=list)
    aot_only: bool = False
    jit_only: bool = False


def _validate(spec: BuildSpec) -> None:
    if spec.optimizer not in OPTIMIZERS:
        raise ValueError(
            f"optimizer={spec.optimizer!r} not in {list(OPTIMIZERS)}")
    if spec.model not in MODELS:
        raise ValueError(f"model={spec.model!r} not in {list(MODELS)}")
    if spec.arch not in ARCHES:
        raise ValueError(f"arch={spec.arch!r} not in {list(ARCHES)}")
    if spec.autotune_mode not in ("quick", "default", "exhaustive"):
        raise ValueError(
            f"autotune_mode={spec.autotune_mode!r} not in "
            "{'quick', 'default', 'exhaustive'}")


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


def _host_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "pallas":
        return []
    base = [
        "-O3", "-std=c++17", f"-D{info['host_define']}",
        "-ffast-math", "-funroll-loops", "-fPIC",
        "-flto",
    ]
    return base + _build_macros(spec)


def _device_cflags(spec: BuildSpec) -> List[str]:
    info = ARCH_INFO[spec.arch]
    if info["vendor"] == "cuda":
        base = [
            "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas", "-O3",
            "-Xptxas", "-v",
            "-Xptxas", "--warn-on-spills",
            "-Xcompiler", "-fPIC",
            "-Xcompiler", "-flto",
            "--resource-usage",
            "--generate-line-info",
            "--maxrregcount=255",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
            "-dlto",
        ]
        if (REPO_ROOT / "third_party/cutlass/include").exists():
            base += ["-DWITH_CUTLASS", "-DCUTLASS_NVCC_ARCHS=90a"]
        return base + _build_macros(spec)
    if info["vendor"] == "hip":
        return [
            "-O3", "-std=c++17", "-DWITH_HIP",
            "-ffast-math", "-fPIC",
            "--offload-arch=gfx942",
            "-Rpass-analysis=kernel-resource-usage",
            "-ggdb",
            "-flto",
        ] + _build_macros(spec)
    return []


def _ldflags(spec: BuildSpec) -> List[str]:
    if ARCH_INFO[spec.arch]["vendor"] == "pallas":
        return []
    return ["-flto", "-Wl,--as-needed"]


def _include_paths() -> List[str]:
    return [
        str(REPO_ROOT / "csrc/bindings"),
        str(REPO_ROOT),
    ]


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
    with env_overlay(
        MAX_JOBS=NCPUS,
        NINJA_STATUS="[%f/%t %es] ",
        TORCH_CUDA_VERBOSE_BUILD="1",
        CMAKE_BUILD_PARALLEL_LEVEL=NCPUS,
    ):
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
# JIT autotune: variant build + on-device timing in an isolated subprocess
# ---------------------------------------------------------------------------

TIMING_SCRIPT = r"""
import sys, json, importlib.util, traceback
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "torch.cuda.is_available() == False"}))
        sys.exit(1)
    so_path = {so_path!r}
    spec = importlib.util.spec_from_file_location("grokking_optimizers._ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["grokking_optimizers._ops"] = mod
    from grokking_optimizers import {opt_class}
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn({size}, {size}, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    # Warmup
    for _ in range({warmup}):
        p.grad = g.clone()
        opt = {opt_class}([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()
    # Time N iterations; report median for robustness against jitter
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
    median = timings[len(timings) // 2]
    print(json.dumps({
        "timing_ms": median,
        "min_ms":    timings[0],
        "max_ms":    timings[-1],
        "n":         len(timings),
    }))
except Exception as exc:
    print(json.dumps({"error": str(exc), "tb": traceback.format_exc()}))
    sys.exit(1)
"""


def _time_variant(variant_so: Path, opt_class: str, *,
                  size: int = 4096, warmup: int = 5, iters: int = 21,
                  timeout: int = 180, report=None) -> Optional[dict]:
    """Subprocess-isolate the timing of ``variant_so`` so each variant
    starts from a clean Python module state.

    Returns ``{timing_ms, min_ms, max_ms, n}`` or ``None`` on failure.
    """
    body = TIMING_SCRIPT.format(
        so_path=str(variant_so),
        opt_class=opt_class,
        size=size,
        warmup=warmup,
        iters=iters,
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
        if report is not None:
            report.write(f"    [time-subprocess exit={proc.returncode}]\n")
        # Parse the last JSON line (script prints one line on success)
        last = next((ln for ln in reversed(out.splitlines())
                     if ln.startswith("{")), None)
        if last is None:
            if report is not None:
                report.write(f"    [time-subprocess no-json output]\n{out}\n")
            return None
        try:
            result = json.loads(last)
        except json.JSONDecodeError:
            if report is not None:
                report.write(f"    [time-subprocess json-decode error]\n{out}\n")
            return None
        if "error" in result:
            if report is not None:
                report.write(f"    [time-subprocess error: {result['error']}]\n")
            return None
        return result
    except subprocess.TimeoutExpired:
        if report is not None:
            report.write(f"    [time-subprocess timeout after {timeout}s]\n")
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _select_search_space(arch: str, mode: str) -> dict:
    table = {
        "quick":      QUICK_SPACE,
        "default":    DEFAULT_SPACE,
        "exhaustive": EXHAUSTIVE_SPACE,
    }[mode]
    if arch not in table:
        raise ValueError(f"no search space for arch={arch}")
    return table[arch]


def _config_key(b: int, v: int, u: int) -> str:
    return f"b{b}_v{v}_u{u}"


def _jit_autotune(spec: BuildSpec, sources: List[Path],
                  host_cflags: List[str], device_cflags: List[str],
                  ldflags: List[str], cache: CompileCache,
                  report) -> Optional[dict]:
    """Sweep the per-arch config grid on the live GPU and pick the
    fastest median timing.

    Each variant build reuses the same ``build_directory`` family so
    ninja's per-file cache makes the second-and-later variant builds
    near-instant (only the .o files affected by the changed
    ``-DSG_TUNED_*`` macros rebuild).
    """
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except ImportError:
        gpu_ok = False
    if not gpu_ok:
        report.write("  [jit-autotune] no GPU visible — skipping; run on "
                     "target arch with this cache to complete JIT phase.\n")
        return None
    if ARCH_INFO[spec.arch]["vendor"] == "pallas":
        report.write("  [jit-autotune] pallas backend; no C++ tuning.\n")
        return None

    if cache.is_jit_fresh(spec.optimizer, spec.model, spec.arch):
        tuned = cache.get(spec.optimizer, spec.model, spec.arch)["tuned_config"]
        report.write(f"  [jit-autotune] cache hit: tuned={tuned}\n")
        return tuned

    space = _select_search_space(spec.arch, spec.autotune_mode)
    combos = [(b, v, u)
              for b in space["block"]
              for v in space["vec"]
              for u in space["unroll"]]
    report.write(f"  [jit-autotune] mode={spec.autotune_mode} — sweeping "
                 f"{len(combos)} configs "
                 f"({len(space['block'])}×{len(space['vec'])}×"
                 f"{len(space['unroll'])} = block × vec × unroll)\n")

    opt_class = OPT_CLASS[spec.optimizer]
    best: Optional[dict] = None
    for i, (b, v, u) in enumerate(combos, 1):
        ckey = _config_key(b, v, u)
        cfg_macros = [
            f"-DSG_TUNED_BLOCK_SIZE={b}",
            f"-DSG_TUNED_VEC_WIDTH={v}",
            f"-DSG_TUNED_UNROLL={u}",
        ]
        report.write(f"\n  [{i}/{len(combos)}] variant {ckey}: building... ")
        report.flush()
        variant_so = _torch_load(
            spec, sources,
            host_cflags + cfg_macros,
            device_cflags + cfg_macros,
            ldflags, report,
            module_suffix=f"_{ckey}",
        )
        if variant_so is None:
            report.write(f"  [{ckey}] variant build FAILED; skipping.\n")
            continue
        cache.record_variant(spec.optimizer, spec.model, spec.arch,
                             ckey, variant_so)
        report.write(f"  [{ckey}] timing... ")
        report.flush()
        timing = _time_variant(variant_so, opt_class, report=report)
        if timing is None:
            report.write(f"  [{ckey}] timing FAILED; skipping.\n")
            continue
        ms = timing["timing_ms"]
        report.write(f"  [{ckey}] median={ms:.4f}ms "
                     f"(min={timing['min_ms']:.4f} max={timing['max_ms']:.4f} "
                     f"n={timing['n']})\n")
        cache.record_sweep(
            spec.optimizer, spec.model, spec.arch,
            config={"block": b, "vec": v, "unroll": u},
            timing_ms=ms, min_ms=timing["min_ms"], max_ms=timing["max_ms"],
            n=timing["n"], config_key=ckey,
        )
        if best is None or ms < best["timing_ms"]:
            best = {"block": b, "vec": v, "unroll": u,
                    "timing_ms": ms, "config_key": ckey}

    if best is None:
        report.write("\n  [jit-autotune] no successful variants — "
                     "leaving tuned_config unset.\n")
        return None

    report.write(f"\n  [jit-autotune] WINNER: {best['config_key']} "
                 f"@ {best['timing_ms']:.4f}ms median\n")
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, best)
    return best


# ---------------------------------------------------------------------------
# tuned_configs.h — written from the cache's winning config
# ---------------------------------------------------------------------------

def _write_tuned_configs_header(combo: dict, optimizer: str, model: str,
                                arch: str, report) -> Path:
    tuned_h = REPO_ROOT / "csrc/algorithms/tuned_configs.h"
    tuned_h.write_text(textwrap.dedent(f"""\
        // Auto-generated by grokking_optimizers.compile JIT autotune.
        // Do not edit by hand — re-run with --jit-only to refresh.
        //
        // Winning combo: optimizer={optimizer} model={model} arch={arch}
        // block_size={combo['block']} vec_width={combo['vec']} unroll={combo['unroll']}
        // median timing: {combo['timing_ms']:.4f} ms
        // Generated: {datetime.datetime.now().isoformat()}
        #pragma once
        #ifndef SG_TUNED_BLOCK_SIZE
        #define SG_TUNED_BLOCK_SIZE {combo['block']}
        #endif
        #ifndef SG_TUNED_VEC_WIDTH
        #define SG_TUNED_VEC_WIDTH {combo['vec']}
        #endif
        #ifndef SG_TUNED_UNROLL
        #define SG_TUNED_UNROLL {combo['unroll']}
        #endif
    """))
    report.write(f"  [tuned_configs.h] wrote {tuned_h.relative_to(REPO_ROOT)}\n")
    return tuned_h


# ---------------------------------------------------------------------------
# Public entry
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
    autotune: bool = True,
    autotune_mode: str = "default",
    profile: bool = True,
    report_path: Optional[Path] = None,
    verbose: bool = False,
    extra_macros: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Run the targeted compile pipeline.

    Either pass an existing ``cache`` (a ``CompileCache`` instance held
    in memory across multiple ``build()`` calls) or ``cache_path`` (the
    path will be loaded once into memory and saved at the end). If
    neither is given, a cache is loaded from ``<out_dir>/.compile_cache.json``.

    ``aot_only=True`` runs the CPU-portable AOT half only (writes cache
    with ``aot_completed_at`` set; leaves ``jit_completed_at = None``).
    ``jit_only=True`` skips the AOT phase (cache must already have a
    matching AOT entry from a previous run / shipped from another host)
    and only runs the JIT autotune sweep on the active GPU.

    Returns the path to the produced .so, or ``None`` for pallas /
    on failure.
    """
    spec = BuildSpec(
        optimizer=optimizer,
        model=model,
        arch=arch,
        out_dir=(out_dir or (REPO_ROOT / "build" / "compiled")).resolve(),
        autotune=autotune,
        autotune_mode=autotune_mode,
        profile=profile,
        verbose=verbose,
        extra_macros=list(extra_macros or []),
        aot_only=aot_only,
        jit_only=jit_only,
    )
    _validate(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if cache is None:
        cp = Path(cache_path) if cache_path else (spec.out_dir / DEFAULT_CACHE_NAME)
        cache = CompileCache(cp)

    report_path = report_path or (
        spec.out_dir / f"compile_{optimizer}_{model}_{arch}.txt"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    info = ARCH_INFO[arch]
    phases: List[str] = ["resolve"]
    if info["vendor"] != "pallas":
        if not jit_only:
            phases += ["aot"]
        if autotune and not aot_only and info["vendor"] != "pallas":
            phases += ["jit-autotune"]
        if not aot_only:
            phases += ["final"]
    if profile and not aot_only:
        phases += ["profile"]

    step, close = make_progress(len(phases), f"{optimizer}/{model}/{arch}")
    so_path: Optional[Path] = None

    try:
        with open(report_path, "w") as report:
            report.write("# grokking_optimizers.compile — targeted build\n")
            report.write(f"# Generated:   {datetime.datetime.now().isoformat()}\n")
            report.write(f"# Optimizer:   {optimizer}\n")
            report.write(f"# Model:       {model}\n")
            report.write(f"# Arch:        {arch} (vendor={info['vendor']})\n")
            report.write(f"# CPU cores:   {NCPUS} (ninja -j)\n")
            report.write(f"# Out dir:     {spec.out_dir}\n")
            report.write(f"# Cache:       {cache.path}\n")
            report.write(f"# AOT only:    {aot_only}\n")
            report.write(f"# JIT only:    {jit_only}\n")
            report.write(f"# Autotune:    {autotune} (mode={autotune_mode})\n")
            report.write(f"# Profile:     {profile}\n")

            sources = _resolve_sources(spec)
            host_cflags = _host_cflags(spec)
            device_cflags = _device_cflags(spec)
            ldflags = _ldflags(spec)
            source_hash = (_hash_sources(sources) if sources else "pallas")
            host_hash   = _hash_flags(host_cflags)
            device_hash = _hash_flags(device_cflags)
            report.write(f"\n[resolve] sources={len(sources)} "
                         f"source_hash={source_hash[:16]} "
                         f"host_hash={host_hash[:16]} "
                         f"device_hash={device_hash[:16]}\n")
            step("resolve")

            if info["vendor"] == "pallas":
                launcher = (REPO_ROOT / "csrc/backends/pallas"
                            / f"launch_{optimizer}.py")
                report.write("\n[pallas] no C++ compile; Python launcher only\n")
                report.write(f"  launcher: {launcher}\n")
                report.write(f"  exists:   {launcher.exists()}\n")
                cache.save()
            else:
                # ─── AOT phase ───────────────────────────────────────────
                if not jit_only:
                    report.write("\n--- AOT PHASE ---\n")
                    if cache.is_aot_fresh(optimizer, model, arch,
                                          source_hash, host_hash, device_hash):
                        art = cache.get(optimizer, model, arch)["primary_artifact"]
                        so_path = Path(art["path"])
                        report.write(f"  [cache HIT] reusing primary "
                                     f"artefact: {so_path}\n"
                                     f"             size={art['size']} "
                                     f"sha256={art['sha256'][:16]}…\n")
                    else:
                        report.write("  [cache MISS] building primary "
                                     "artefact...\n")
                        so_path = _torch_load(
                            spec, sources, host_cflags, device_cflags,
                            ldflags, report,
                        )
                        cache.record_aot(
                            optimizer, model, arch,
                            source_hash=source_hash,
                            host_flags_hash=host_hash,
                            device_flags_hash=device_hash,
                            so_path=so_path,
                        )
                        cache.save()  # persist AOT result before JIT begins
                    step("aot")

                # ─── JIT autotune phase (target hardware only) ───────────
                tuned: Optional[dict] = None
                if autotune and not aot_only:
                    report.write("\n--- JIT AUTOTUNE PHASE ---\n")
                    tuned = _jit_autotune(
                        spec, sources, host_cflags, device_cflags,
                        ldflags, cache, report,
                    )
                    cache.save()
                    step("jit-autotune")

                # ─── Final build with tuned configs baked into header ────
                if not aot_only:
                    report.write("\n--- FINAL PASS ---\n")
                    if tuned is not None:
                        _write_tuned_configs_header(
                            tuned, optimizer, model, arch, report)
                        so_path = _torch_load(
                            spec, sources,
                            host_cflags + [
                                f"-DSG_TUNED_BLOCK_SIZE={tuned['block']}",
                                f"-DSG_TUNED_VEC_WIDTH={tuned['vec']}",
                                f"-DSG_TUNED_UNROLL={tuned['unroll']}",
                            ],
                            device_cflags + [
                                f"-DSG_TUNED_BLOCK_SIZE={tuned['block']}",
                                f"-DSG_TUNED_VEC_WIDTH={tuned['vec']}",
                                f"-DSG_TUNED_UNROLL={tuned['unroll']}",
                            ],
                            ldflags, report,
                            module_suffix="_tuned",
                        )
                        cache.record_aot(
                            optimizer, model, arch,
                            source_hash=source_hash,
                            host_flags_hash=_hash_flags(host_cflags + [
                                f"-DSG_TUNED_BLOCK_SIZE={tuned['block']}",
                                f"-DSG_TUNED_VEC_WIDTH={tuned['vec']}",
                                f"-DSG_TUNED_UNROLL={tuned['unroll']}",
                            ]),
                            device_flags_hash=_hash_flags(device_cflags + [
                                f"-DSG_TUNED_BLOCK_SIZE={tuned['block']}",
                                f"-DSG_TUNED_VEC_WIDTH={tuned['vec']}",
                                f"-DSG_TUNED_UNROLL={tuned['unroll']}",
                            ]),
                            so_path=so_path,
                        )
                    else:
                        report.write("  [final] no tuned config — keeping "
                                     "AOT primary artefact as final.\n")
                    cache.save()
                    step("final")

            # ─── Profile pass (delegated) ────────────────────────────────
            if profile and not aot_only:
                report.write("\n--- PROFILE PASS "
                             "(via grokking_optimizers.profile) ---\n")
                _dispatch_profile(optimizer, model, arch, report)
                step("profile")

            report.write(f"\n# Cache:    {cache.path}\n")
            report.write(f"# Final .so: {so_path}\n")
    finally:
        cache.save()
        close()

    return so_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grokking_optimizers.compile",
        description="Targeted per-(optimizer, model, arch) build pipeline "
                    "with portable AOT-then-JIT cache.",
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
                        help="Cache file path (default: <out>/.compile_cache.json). "
                             "Loaded into memory at start; written atomically at end.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Report file path (default: <out>/compile_<O>_<M>_<A>.txt)")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--aot-only", action="store_true",
                       help="Run only the AOT (CPU-portable) phase. No GPU required. "
                            "Cache gets aot_completed_at; ship it to the GPU host.")
    phase.add_argument("--jit-only", action="store_true",
                       help="Skip AOT (assume the cache already has it from "
                            "a previous run / another host); run only the "
                            "JIT autotune sweep on the active GPU.")
    parser.add_argument("--no-autotune", action="store_true",
                       help="Skip the JIT autotune sweep even when a GPU "
                            "is visible (build with default tuned configs).")
    space = parser.add_mutually_exclusive_group()
    space.add_argument("--quick", dest="autotune_mode", action="store_const",
                       const="quick",
                       help="Quick autotune (8 configs).")
    space.add_argument("--exhaustive", dest="autotune_mode",
                       action="store_const", const="exhaustive",
                       help="Exhaustive autotune (75 configs — maximally "
                            "optimised, targets ~100%% utilisation).")
    parser.add_argument("--no-profile", action="store_true",
                        help="Skip the ncu/rocprof/jax.profiler pass.")
    parser.add_argument("-D", dest="extra_macros", action="append", default=[],
                        help="Extra -D<MACRO[=VALUE]>. Repeatable.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose torch cpp_extension output")
    args = parser.parse_args(argv)

    extra = [m if m.startswith("-D") else f"-D{m}"
             for m in args.extra_macros]
    autotune_mode = args.autotune_mode or "default"

    so = build(
        optimizer=args.optimizer, model=args.model, arch=args.arch,
        cache_path=args.cache,
        out_dir=args.out,
        aot_only=args.aot_only,
        jit_only=args.jit_only,
        autotune=not args.no_autotune,
        autotune_mode=autotune_mode,
        profile=not args.no_profile,
        report_path=args.report,
        verbose=args.verbose,
        extra_macros=extra,
    )

    report = args.report or (
        args.out / f"compile_{args.optimizer}_{args.model}_{args.arch}.txt"
    )
    sys.stdout.write(f"{report}\n")
    return 0 if (so is not None
                 or ARCH_INFO[args.arch]["vendor"] == "pallas"
                 or args.aot_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
