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

  1. Load the resolved search space — by default, the COMPLETE programmatic
     space from build_full_search_space() (~billions of candidates per arch).
     Override with --search-space <path/to/your.yaml> to use a smaller
     curated space.
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
                    bayesian_trials=None,  # None = multi-criterion auto-stop
                    max_tune_seconds=900,  # 15-minute wall-clock cap
                    pgo=False)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from grokking_optimizers.profile import (
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
# ARCH_INFO / ARCH_TABLE are defined later in this file (single source of
# truth); profile.py imports them lazily to avoid a circular import.

import itertools
import math

import optuna
from optuna.samplers import TPESampler
import yaml


CACHE_VERSION = 3
DEFAULT_CACHE_NAME = ".compile_cache.json"
DEFAULT_PGO_WORKLOAD = Path(__file__).resolve()  # absorbed from scripts/pgo_workload.py
JIT_CACHE_FLUSH_EVERY = 5   # save cache every N completed JIT trials


# ---------------------------------------------------------------------------
# ARCH_TABLE — single source of truth for every GPU / TPU architecture
# ---------------------------------------------------------------------------
#
# Every codepath that needs to know "what arch are we building for?" reads
# from this table. The previous design had three separate sources (the
# ``ARCH_INFO`` dict in profile.py, the ``min_cuda_for_arch`` dict in
# _preflight_toolchain, and hardcoded `-gencode=arch=compute_90,code=sm_90`
# / ``--offload-arch=gfx942`` strings in NVCC_DEVICE_BASE / HIPCC_DEVICE_BASE).
# Adding a new arch meant editing all three plus a handful of ``if arch ==
# "sm_90"`` branches scattered through the file. ARCH_TABLE collapses all of
# that into one declarative dataclass per arch.
#
# Per-arch search-space builders are wired in *after* ``_sm90_full_space``
# / ``_gfx942_full_space`` are defined (further down in this file). Other
# archs leave ``search_space_builder=None`` until Stream 2 fills them in;
# this is intentional and not a bug.

from dataclasses import dataclass, field  # noqa: E402  (re-import for clarity)


@dataclass
class ArchEntry:
    """Declarative spec for one GPU/TPU architecture.

    Every consumer in compile.py / profile.py reads from this; never
    hardcode arch-specific behaviour outside this table.
    """
    vendor: str                                # "cuda" | "hip" | "pallas"
    display_name: str
    subdir: str
    launcher_glob: Tuple[str, ...]
    model_glob: Tuple[str, ...]
    macro: str                                 # SG_BUILD_ARCH_*
    host_define: Optional[str]                 # WITH_CUDA / WITH_HIP / None
    min_toolchain_version: Tuple[int, ...]     # (M, m) CUDA/ROCm; (M, m, p) JAX
    arch_suffix: str                           # "a" for Hopper/Blackwell, else ""
    nvcc_gencode: List[str]                    # per-arch -gencode flags (empty for non-CUDA)
    hipcc_offload_arch: str                    # e.g. "gfx942"; "" for non-HIP
    cutlass_arch: Optional[int]                # CUTLASS_ARCH_MMA_SM* int; None for non-CUDA
    max_smem_per_block: Optional[int]          # bytes (CUDA smem / HIP LDS); None for Pallas
    warp_size: Optional[int]                   # 32 CUDA/RDNA, 64 CDNA, None Pallas
    max_regs_per_thread: Optional[int]         # 255 CUDA/HIP, None Pallas
    max_threads_per_block: Optional[int]       # 1024 CUDA/HIP, None Pallas
    features: frozenset                        # capability flag strings (see docstring)
    search_space_builder: Optional[Callable[[], Dict[str, Any]]] = None


# ---- NVCC gencode helper -------------------------------------------------
#
# For each CUDA arch we emit two -gencode flags:
#   1. compute_XX[a],code=sm_XX[a]  — SASS for the target SM
#   2. compute_XX,code=compute_XX   — PTX fallback so older drivers can JIT
#      forward to newer hardware that didn't exist at compile time.
# The "a" suffix on Hopper+ tells NVCC to emit arch-specific instructions
# (TMA, wgmma, tcgen05 etc.) that the non-"a" variant rejects.

def _nvcc_gencode_pair(num: int, suffix: str = "") -> List[str]:
    sm = f"sm_{num}{suffix}"
    compute = f"compute_{num}{suffix}"
    compute_fallback = f"compute_{num}"   # PTX fallback is always non-"a"
    return [
        f"-gencode=arch={compute},code={sm}",
        f"-gencode=arch={compute_fallback},code={compute_fallback}",
    ]


# ---- Feature-flag mnemonics (see ARCH_TABLE.features) --------------------
#
# tma, wgmma, cluster ......... Hopper (sm_90a) async-copy + cluster launch
# fp8 ........................ Ada (sm_89) / Blackwell / RDNA4
# fp4, tcgen05 ............... Blackwell datacenter (sm_100a/103a)
# mfma ....................... CDNA matrix-fused-multiply-add
# bf16_mfma / fp8_mfma / fp4_mfma ... CDNA2/3/4 type-specific MFMA
# wmma ....................... RDNA wave-matrix-multiply-accumulate
# sparsecore ................. TPU sparse embedding co-processor
# async_copy ................. cp.async (Ampere+)
# dpp, tgsplit ............... RDNA wave32 primitives
# cuda_graph, cooperative_groups, dyn_parallelism ... universal CUDA features

# Tuple form of feature flags used to build frozensets compactly below.
_F_BASE_CUDA = ("cuda_graph", "cooperative_groups", "dyn_parallelism")
_F_AMPERE    = _F_BASE_CUDA + ("async_copy", "bf16")
_F_ADA       = _F_AMPERE + ("fp8",)
_F_HOPPER    = _F_ADA + ("tma", "wgmma", "cluster")
_F_BLACKWELL = _F_HOPPER + ("fp4", "tcgen05")
_F_BLACKWELL_CONSUMER = _F_ADA + ("fp4",)     # sm_120a: fp8+fp4, no tma/wgmma/cluster


_ARCH_TABLE_PRIMARY: Dict[str, ArchEntry] = {

    # =================== NVIDIA / CUDA ===================
    "sm_75": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA T4 (Turing)",
        subdir="cuda/sm_75",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM75",
        host_define="WITH_CUDA",
        min_toolchain_version=(10, 0),
        arch_suffix="",
        nvcc_gencode=_nvcc_gencode_pair(75),
        hipcc_offload_arch="",
        cutlass_arch=75,
        max_smem_per_block=100 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_BASE_CUDA),
    ),

    "sm_80": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA A100 (Ampere)",
        subdir="cuda/sm_80",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM80",
        host_define="WITH_CUDA",
        min_toolchain_version=(11, 0),
        arch_suffix="",
        nvcc_gencode=_nvcc_gencode_pair(80),
        hipcc_offload_arch="",
        cutlass_arch=80,
        max_smem_per_block=164 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_AMPERE),
    ),

    "sm_86": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA A10/RTX 30xx (Ampere)",
        subdir="cuda/sm_86",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM86",
        host_define="WITH_CUDA",
        min_toolchain_version=(11, 1),
        arch_suffix="",
        nvcc_gencode=_nvcc_gencode_pair(86),
        hipcc_offload_arch="",
        cutlass_arch=86,
        max_smem_per_block=164 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_AMPERE),
    ),

    "sm_89": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA L4/L40/RTX 40xx (Ada Lovelace)",
        subdir="cuda/sm_89",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM89",
        host_define="WITH_CUDA",
        min_toolchain_version=(11, 8),
        arch_suffix="",
        nvcc_gencode=_nvcc_gencode_pair(89),
        hipcc_offload_arch="",
        cutlass_arch=89,
        max_smem_per_block=164 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_ADA),
    ),

    "sm_90a": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA H100/H200 (Hopper)",
        subdir="cuda/sm_90",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM90",
        host_define="WITH_CUDA",
        min_toolchain_version=(12, 0),
        arch_suffix="a",
        nvcc_gencode=_nvcc_gencode_pair(90, "a"),
        hipcc_offload_arch="",
        cutlass_arch=90,
        max_smem_per_block=228 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_HOPPER),
    ),

    "sm_100a": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA B100/B200/GB200 (Blackwell)",
        subdir="cuda/sm_100",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM100",
        host_define="WITH_CUDA",
        min_toolchain_version=(12, 8),
        arch_suffix="a",
        nvcc_gencode=_nvcc_gencode_pair(100, "a"),
        hipcc_offload_arch="",
        cutlass_arch=100,
        max_smem_per_block=232 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_BLACKWELL),
    ),

    "sm_103a": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA B300 (Blackwell Ultra)",
        subdir="cuda/sm_103",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM103",
        host_define="WITH_CUDA",
        min_toolchain_version=(12, 9),
        arch_suffix="a",
        nvcc_gencode=_nvcc_gencode_pair(103, "a"),
        hipcc_offload_arch="",
        cutlass_arch=103,
        max_smem_per_block=232 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_BLACKWELL),
    ),

    "sm_120a": ArchEntry(
        vendor="cuda",
        display_name="NVIDIA RTX 50xx / RTX 6000 Pro Blackwell (consumer)",
        subdir="cuda/sm_120",
        launcher_glob=("launch_*.cu",),
        model_glob=("*.cu",),
        macro="SG_BUILD_ARCH_SM120",
        host_define="WITH_CUDA",
        min_toolchain_version=(12, 8),
        arch_suffix="a",
        nvcc_gencode=_nvcc_gencode_pair(120, "a"),
        hipcc_offload_arch="",
        cutlass_arch=120,
        max_smem_per_block=100 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset(_F_BLACKWELL_CONSUMER),
    ),

    # =================== AMD / HIP / ROCm ===================
    "gfx906": ArchEntry(
        vendor="hip",
        display_name="AMD MI50 (Vega20)",
        subdir="hip/gfx906",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX906",
        host_define="WITH_HIP",
        min_toolchain_version=(3, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx906",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=64,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"mfma"}),
    ),

    "gfx908": ArchEntry(
        vendor="hip",
        display_name="AMD MI100 (CDNA1)",
        subdir="hip/gfx908",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX908",
        host_define="WITH_HIP",
        min_toolchain_version=(3, 5),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx908",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=64,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"mfma", "bf16_mfma"}),
    ),

    "gfx90a": ArchEntry(
        vendor="hip",
        display_name="AMD MI200/MI250 (CDNA2)",
        subdir="hip/gfx90a",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX90A",
        host_define="WITH_HIP",
        min_toolchain_version=(4, 5),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx90a",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=64,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"mfma", "bf16_mfma", "mfma_xdl"}),
    ),

    "gfx942": ArchEntry(
        vendor="hip",
        display_name="AMD MI300X/MI300A (CDNA3)",
        subdir="hip/gfx942",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX942",
        host_define="WITH_HIP",
        min_toolchain_version=(6, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx942",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=64,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"mfma", "bf16_mfma", "fp8_mfma", "mfma_xdl"}),
    ),

    "gfx950": ArchEntry(
        vendor="hip",
        display_name="AMD MI350X/MI355X (CDNA4)",
        subdir="hip/gfx950",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX950",
        host_define="WITH_HIP",
        min_toolchain_version=(6, 2),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx950",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=64,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"mfma", "bf16_mfma", "fp8_mfma", "fp4_mfma",
                            "mfma_xdl", "mfma_4x_smfmac"}),
    ),

    "gfx1030": ArchEntry(
        vendor="hip",
        display_name="AMD RX 6000 (RDNA2)",
        subdir="hip/gfx1030",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1030",
        host_define="WITH_HIP",
        min_toolchain_version=(4, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1030",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma"}),
    ),

    "gfx1100": ArchEntry(
        vendor="hip",
        display_name="AMD RX 7900 (RDNA3)",
        subdir="hip/gfx1100",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1100",
        host_define="WITH_HIP",
        min_toolchain_version=(5, 5),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1100",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit"}),
    ),

    "gfx1101": ArchEntry(
        vendor="hip",
        display_name="AMD RX 7800 (RDNA3)",
        subdir="hip/gfx1101",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1101",
        host_define="WITH_HIP",
        min_toolchain_version=(5, 5),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1101",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit"}),
    ),

    "gfx1102": ArchEntry(
        vendor="hip",
        display_name="AMD RX 7600 (RDNA3)",
        subdir="hip/gfx1102",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1102",
        host_define="WITH_HIP",
        min_toolchain_version=(5, 5),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1102",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit"}),
    ),

    "gfx1151": ArchEntry(
        vendor="hip",
        display_name="AMD Strix Halo (RDNA3.5)",
        subdir="hip/gfx1151",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1151",
        host_define="WITH_HIP",
        min_toolchain_version=(6, 1),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1151",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit"}),
    ),

    "gfx1200": ArchEntry(
        vendor="hip",
        display_name="AMD RX 9000 (RDNA4)",
        subdir="hip/gfx1200",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1200",
        host_define="WITH_HIP",
        min_toolchain_version=(7, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1200",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit", "fp8"}),
    ),

    "gfx1201": ArchEntry(
        vendor="hip",
        display_name="AMD RX 9070 (RDNA4)",
        subdir="hip/gfx1201",
        launcher_glob=("launch_*.hip.cpp", "launch_*.hip"),
        model_glob=("*.hip.cpp",),
        macro="SG_BUILD_ARCH_GFX1201",
        host_define="WITH_HIP",
        min_toolchain_version=(7, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="gfx1201",
        cutlass_arch=None,
        max_smem_per_block=64 * 1024,
        warp_size=32,
        max_regs_per_thread=255,
        max_threads_per_block=1024,
        features=frozenset({"wmma", "dpp", "tgsplit", "fp8"}),
    ),

    # =================== Google / Pallas / XLA / Mosaic ===================
    "tpu_v4": ArchEntry(
        vendor="pallas",
        display_name="Google TPU v4",
        subdir="pallas",
        launcher_glob=("launch_*.py",),
        model_glob=(),
        macro="SG_BUILD_ARCH_TPU_V4",
        host_define=None,
        min_toolchain_version=(0, 4, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="",
        cutlass_arch=None,
        max_smem_per_block=None,
        warp_size=None,
        max_regs_per_thread=None,
        max_threads_per_block=None,
        features=frozenset(),
    ),

    "tpu_v5e": ArchEntry(
        vendor="pallas",
        display_name="Google TPU v5e",
        subdir="pallas",
        launcher_glob=("launch_*.py",),
        model_glob=(),
        macro="SG_BUILD_ARCH_TPU_V5E",
        host_define=None,
        min_toolchain_version=(0, 4, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="",
        cutlass_arch=None,
        max_smem_per_block=None,
        warp_size=None,
        max_regs_per_thread=None,
        max_threads_per_block=None,
        features=frozenset(),
    ),

    "tpu_v5p": ArchEntry(
        vendor="pallas",
        display_name="Google TPU v5p",
        subdir="pallas",
        launcher_glob=("launch_*.py",),
        model_glob=(),
        macro="SG_BUILD_ARCH_TPU_V5P",
        host_define=None,
        min_toolchain_version=(0, 4, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="",
        cutlass_arch=None,
        max_smem_per_block=None,
        warp_size=None,
        max_regs_per_thread=None,
        max_threads_per_block=None,
        features=frozenset({"sparsecore"}),
    ),

    "tpu_v6e": ArchEntry(
        vendor="pallas",
        display_name="Google TPU v6e (Trillium)",
        subdir="pallas",
        launcher_glob=("launch_*.py",),
        model_glob=(),
        macro="SG_BUILD_ARCH_TPU_V6E",
        host_define=None,
        min_toolchain_version=(0, 4, 30),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="",
        cutlass_arch=None,
        max_smem_per_block=None,
        warp_size=None,
        max_regs_per_thread=None,
        max_threads_per_block=None,
        features=frozenset(),
    ),

    "tpu_v7": ArchEntry(
        vendor="pallas",
        display_name="Google TPU v7 (Ironwood)",
        subdir="pallas",
        launcher_glob=("launch_*.py",),
        model_glob=(),
        macro="SG_BUILD_ARCH_TPU_V7",
        host_define=None,
        min_toolchain_version=(0, 5, 0),
        arch_suffix="",
        nvcc_gencode=[],
        hipcc_offload_arch="",
        cutlass_arch=None,
        max_smem_per_block=None,
        warp_size=None,
        max_regs_per_thread=None,
        max_threads_per_block=None,
        features=frozenset(),
    ),
}


# ---- Backward-compat aliases (canonical -> alias) ------------------------
# Users typing "--arch sm_90" continue to work; the canonical entry is the
# "a"-suffixed variant. Both keys map to the SAME ArchEntry object, so
# later patches (e.g. ``search_space_builder=_sm90_full_space``) propagate
# transparently. Tested by test_arch_table_completeness below.
_ARCH_TABLE_ALIASES: Dict[str, str] = {
    "sm_90":  "sm_90a",
    "sm_100": "sm_100a",
    "sm_103": "sm_103a",
    "sm_120": "sm_120a",
}


# Full ARCH_TABLE with both canonical and alias keys pointing at the same
# ArchEntry instances. Order: primary entries first, then aliases.
ARCH_TABLE: Dict[str, ArchEntry] = {
    **_ARCH_TABLE_PRIMARY,
    **{alias: _ARCH_TABLE_PRIMARY[canonical]
       for alias, canonical in _ARCH_TABLE_ALIASES.items()},
}


# ---- Legacy 6-key view (ARCH_INFO) — keep all old call sites working -----
# Every read like ``ARCH_INFO[arch]["vendor"]`` continues to resolve. The
# derived dict is built from ARCH_TABLE so it stays in sync; profile.py
# imports this shim instead of defining its own ARCH_INFO.
def _build_legacy_arch_info() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for arch, entry in ARCH_TABLE.items():
        out[arch] = {
            "vendor":        entry.vendor,
            "subdir":        entry.subdir,
            "launcher_glob": entry.launcher_glob,
            "model_glob":    entry.model_glob,
            "macro":         entry.macro,
            "host_define":   entry.host_define,
        }
    return out


ARCH_INFO: Dict[str, Dict[str, Any]] = _build_legacy_arch_info()


def get_arch_entry(arch: str) -> ArchEntry:
    """Look up the ArchEntry for ``arch``, resolving aliases.

    Raises KeyError with the list of valid arches if the lookup fails.
    """
    try:
        return ARCH_TABLE[arch]
    except KeyError:
        raise KeyError(
            f"arch={arch!r} not in ARCH_TABLE; "
            f"valid: {sorted(ARCH_TABLE.keys())}")


# How many trials Bayesian "quick" mode runs (vs the full 500 default).
QUICK_BAYESIAN_TRIALS = 25


# Sentinel display value used in reports when no external YAML is supplied.
DEFAULT_SEARCH_SPACE = "<full-programmatic>"


# ---------------------------------------------------------------------------
# Programmatic search-space builder — the COMPLETE space, not a curated list
# ---------------------------------------------------------------------------
#
# Earlier revisions of this file embedded a hand-written YAML blob with a
# small curated set of values per tuning dimension (e.g. block ∈ {64, 128,
# 256, 512, 1024}). That biased the autotuner — Bayesian TPE could only
# sample what we'd pre-decided was "reasonable". This was a mistake: the
# whole point of an extensive optimization run is to discover non-obvious
# winners, and pre-curating values defeats that.
#
# build_full_search_space() now generates the search space programmatically
# from hardware specs:
#   - every warp-aligned (sm_90) / wavefront-aligned (gfx942) block size
#   - the full range of vector widths the load/store instructions support
#   - every power-of-2 unroll factor up to a generous cap
#   - every valid pipeline depth, register-budget, cluster shape, etc.
#
# The result is a Cartesian product of ~10⁹ candidates per arch — far too
# many to materialize. Three consequences:
#   1. ``cartesian()`` is now a generator (was a list).
#   2. ``cartesian_count(space, arch)`` returns the size without iteration.
#   3. ``ss_prefilter()`` accepts an iterable and streams survivors.
# Bayesian TPE doesn't care about the size — it samples one config at a
# time using the per-dim value lists. Exhaustive mode also streams, with a
# cap (--exhaustive-cap) to prevent infinite enumeration.
#
# Users can still override the entire space via --search-space <path.yaml>.


def _dim(name: str, dtype: str, values: List[Any], macro: Optional[str],
         applies_to: List[str], kind: str = "macro") -> Dict[str, Any]:
    """One tuning dimension.

    ``kind`` is one of:
      - ``"macro"`` (default): becomes a ``-DSG_TUNED_*=VAL`` flag passed to nvcc/hipcc.
      - ``"pallas_kwarg"``: passed as a keyword argument to ``pl.pallas_call``.
      - ``"nvcc_flag"`` / ``"hipcc_flag"``: promoted to a bare compiler flag
        (handled out-of-band, e.g. ``maxrregcount``).
    """
    return {"name": name, "type": dtype, "values": values,
            "macro": macro, "applies_to": applies_to, "kind": kind}


# ===========================================================================
# Stream 2: per-arch search-space builders
# ===========================================================================
#
# Each ``_xxx_full_space()`` consults ``ARCH_TABLE[arch].features`` to decide
# which dims to emit. The builders share infrastructure via helper functions
# below; arch-specific logic lives in the per-arch builder.
#
# Cardinality target: 10^6 .. 10^13 per CUDA/HIP arch; 10 .. 10^6 per Pallas
# arch (the Pallas spaces are kwargs to ``pl.pallas_call`` not -D macros).


# ---- Shared dim builders --------------------------------------------------

def _maxrregcount_values(arch_key: str) -> List[int]:
    """Per-arch maxrregcount range, capped at max_regs_per_thread."""
    entry = ARCH_TABLE[arch_key]
    cap = entry.max_regs_per_thread or 255
    # Per spec: [24, 28, 32, ..., 248] in 4-step increments, capped per arch.
    floor_n = max(24, (cap // 8) * 4 // 4 * 4)  # at least max_regs/8 *4 step
    floor_n = max(24, (cap // 8))
    # Snap floor to a multiple of 4.
    floor_n = max(24, (floor_n + 3) // 4 * 4)
    vals = list(range(floor_n, min(249, cap + 1), 4))
    if cap >= 255 and 255 not in vals:
        vals.append(255)
    return vals or [cap]


def _cuda_block_values(arch_key: str) -> List[int]:
    entry = ARCH_TABLE[arch_key]
    max_threads = entry.max_threads_per_block or 1024
    # CUDA: warp_size=32, step 32.
    return list(range(32, max_threads + 1, 32))


def _cdna_block_values(arch_key: str) -> List[int]:
    entry = ARCH_TABLE[arch_key]
    max_threads = entry.max_threads_per_block or 1024
    # CDNA: wavefront=64, step 64.
    return list(range(64, max_threads + 1, 64))


def _rdna_block_values(arch_key: str) -> List[int]:
    entry = ARCH_TABLE[arch_key]
    max_threads = entry.max_threads_per_block or 1024
    # RDNA: wave32 native, but blocks step in 32.
    return list(range(32, max_threads + 1, 32))


def _hopper_cluster_shapes() -> List[List[int]]:
    """Every (m, n, p) ∈ {1,2,4,8}³ with m·n·p ≤ 8 (Hopper hw limit)."""
    out: List[List[int]] = []
    for m in (1, 2, 4, 8):
        for n in (1, 2, 4, 8):
            for p in (1, 2, 4, 8):
                if m * n * p <= 8:
                    out.append([m, n, p])
    return out


def _blackwell_cluster_shapes() -> List[List[int]]:
    """Blackwell datacenter — cluster volume up to 16 CTAs (vs Hopper's 8)."""
    out: List[List[int]] = []
    for m in (1, 2, 4, 8, 16):
        for n in (1, 2, 4, 8, 16):
            for p in (1, 2, 4, 8):
                if m * n * p <= 16:
                    out.append([m, n, p])
    return out


# ---- Shared prefilter rule builders --------------------------------------

def _cuda_common_rules(arch_key: str) -> List[Dict[str, str]]:
    entry = ARCH_TABLE[arch_key]
    warp = entry.warp_size or 32
    max_threads = entry.max_threads_per_block or 1024
    smem = entry.max_smem_per_block or (48 * 1024)
    return [
        {"name": "block_le_max_threads",
         "expr": f"block <= {max_threads}"},
        {"name": "block_warp_aligned",
         "expr": f"block % {warp} == 0"},
        {"name": "vec_block_alignment",
         "expr": "block % (vec * 4) == 0"},
        # Smem heuristic: block * num_stages * 4 * vec ≤ max_smem_per_block.
        {"name": "smem_budget",
         "expr": f"(block * num_stages * 4 * vec) <= {smem}"},
    ]


def _cdna_common_rules(arch_key: str) -> List[Dict[str, str]]:
    entry = ARCH_TABLE[arch_key]
    max_threads = entry.max_threads_per_block or 1024
    smem = entry.max_smem_per_block or (64 * 1024)
    return [
        {"name": "block_le_max_threads",
         "expr": f"block <= {max_threads}"},
        {"name": "block_wave64_aligned",
         "expr": "block % 64 == 0"},
        {"name": "vec_block_alignment",
         "expr": "block % (vec * 4) == 0"},
        {"name": "lds_budget",
         "expr": f"(block * num_stages * 4 * vec) <= {smem}"},
        # Combined waves_per_eu × per-thread reg budget heuristic.
        {"name": "occupancy_budget",
         "expr": "(block * (waves_per_eu if waves_per_eu else 1) * vec * 4) <= 65536"},
    ]


def _rdna_common_rules(arch_key: str) -> List[Dict[str, str]]:
    entry = ARCH_TABLE[arch_key]
    max_threads = entry.max_threads_per_block or 1024
    smem = entry.max_smem_per_block or (64 * 1024)
    return [
        {"name": "block_le_max_threads",
         "expr": f"block <= {max_threads}"},
        {"name": "block_wave_aligned",
         "expr": "block % 32 == 0"},
        {"name": "vec_block_alignment",
         "expr": "block % (vec * 4) == 0"},
        {"name": "lds_budget",
         "expr": f"(block * num_stages * 4 * vec) <= {smem}"},
    ]


# ---- Generic CUDA builder -------------------------------------------------

def _build_cuda_space(arch_key: str,
                     vec_values: Optional[List[int]] = None,
                     unroll_values: Optional[List[int]] = None,
                     stages_values: Optional[List[int]] = None,
                     ) -> Dict[str, Any]:
    """Construct a CUDA arch's full search space from feature flags."""
    entry = ARCH_TABLE[arch_key]
    features = entry.features

    if vec_values is None:
        vec_values = [1, 2, 4, 8, 16]
    if unroll_values is None:
        unroll_values = [1, 2, 4, 8, 16, 32, 64, 128]
    if stages_values is None:
        stages_values = list(range(1, 9))

    dims: List[Dict[str, Any]] = [
        _dim("block", "int", _cuda_block_values(arch_key),
             "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
        _dim("vec", "int", vec_values,
             "SG_TUNED_VEC_WIDTH", ["host", "device"]),
        _dim("unroll", "int", unroll_values,
             "SG_TUNED_UNROLL", ["host", "device"]),
        _dim("num_stages", "int", stages_values,
             "SG_TUNED_NUM_STAGES", ["device"]),
        _dim("maxrregcount", "int", _maxrregcount_values(arch_key),
             None, ["device"]),
        _dim("swizzle", "int", [0, 32, 64, 128, 256],
             "SG_TUNED_SWIZZLE", ["device"]),
    ]

    rules = list(_cuda_common_rules(arch_key))

    if "tma" in features:
        dims.append(_dim("tma_descriptors", "int", [0, 1, 2, 4, 8],
                         "SG_TUNED_TMA_DESCRIPTORS", ["device"]))
        dims.append(_dim("async_depth", "int", list(range(1, 17)),
                         "SG_TUNED_ASYNC_DEPTH", ["device"]))
        rules.append({"name": "async_depth_stages",
                      "expr": "async_depth >= num_stages - 1"})

    if "wgmma" in features:
        # sm_90a: base wgmma shapes; sm_100a/103a add tcgen05 variants.
        wgmma_shapes = ["m64n8k16", "m64n16k16", "m64n32k16",
                        "m64n64k16", "m64n128k16", "m64n256k16"]
        if "tcgen05" in features:
            wgmma_shapes += ["m128n128k16_tcgen05", "m128n256k16_tcgen05",
                             "m256n256k16_tcgen05"]
        dims.append(_dim("wgmma_shape", "enum", wgmma_shapes,
                         "SG_TUNED_WGMMA_SHAPE", ["device"]))
        dims.append(_dim("warp_specialization", "int", [0, 1],
                         "SG_TUNED_WARP_SPECIALIZATION", ["device"]))

    if "cluster" in features:
        if "tcgen05" in features:
            cluster_shapes = _blackwell_cluster_shapes()
            volume_cap = 16
        else:
            cluster_shapes = _hopper_cluster_shapes()
            volume_cap = 8
        dims.append(_dim("cluster_shape", "tuple", cluster_shapes,
                         "SG_TUNED_CLUSTER_SHAPE", ["device"]))
        rules.append({"name": "cluster_volume",
                      "expr": f"cluster_shape[0] * cluster_shape[1] * cluster_shape[2] <= {volume_cap}"})

    if "fp8" in features:
        dims.append(_dim("fp8_layout", "enum", ["none", "e4m3", "e5m2"],
                         "SG_TUNED_FP8_LAYOUT", ["device"]))

    if "fp4" in features:
        dims.append(_dim("fp4_layout", "enum", ["none", "e2m1"],
                         "SG_TUNED_FP4_LAYOUT", ["device"]))

    if "tcgen05" in features:
        dims.append(_dim("tcgen05_variant", "enum",
                         ["tma_a", "tma_b", "mma", "tma_mma"],
                         "SG_TUNED_TCGEN05_VARIANT", ["device"]))

    return {
        "dims": dims,
        "prefilter": {
            "register_pressure_max": entry.max_regs_per_thread or 255,
            "smem_budget_bytes": entry.max_smem_per_block or (48 * 1024),
            "rules": rules,
        },
    }


# ---- Generic HIP builders -------------------------------------------------

def _build_cdna_space(arch_key: str,
                     mfma_shapes: Optional[List[str]] = None,
                     vec_values: Optional[List[int]] = None,
                     unroll_values: Optional[List[int]] = None,
                     stages_values: Optional[List[int]] = None,
                     extra_features: Optional[List[str]] = None,
                     ) -> Dict[str, Any]:
    """CDNA (gfx9xx) builder. Includes waves_per_eu, mfma, no tgsplit."""
    entry = ARCH_TABLE[arch_key]
    features = entry.features
    extras = set(extra_features or [])

    if vec_values is None:
        vec_values = [1, 2, 4, 8, 16]
    if unroll_values is None:
        unroll_values = [1, 2, 4, 8, 16, 32, 64, 128]
    if stages_values is None:
        stages_values = list(range(1, 9))

    dims: List[Dict[str, Any]] = [
        _dim("block", "int", _cdna_block_values(arch_key),
             "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
        _dim("vec", "int", vec_values,
             "SG_TUNED_VEC_WIDTH", ["host", "device"]),
        _dim("unroll", "int", unroll_values,
             "SG_TUNED_UNROLL", ["host", "device"]),
        _dim("num_stages", "int", stages_values,
             "SG_TUNED_NUM_STAGES", ["device"]),
        _dim("maxrregcount", "int", _maxrregcount_values(arch_key),
             None, ["device"]),
        _dim("lds_padding", "int", [0, 4, 8, 16, 32],
             "SG_TUNED_LDS_PADDING", ["device"]),
        _dim("waves_per_eu", "int", [0, 1, 2, 3, 4, 6, 8, 10],
             "SG_TUNED_WAVES_PER_EU", ["device"]),
    ]

    if "mfma" in features:
        if mfma_shapes is None:
            mfma_shapes = ["m16n16k4", "m32n32k2", "m4n4k1"]
        dims.append(_dim("mfma_shape", "enum", mfma_shapes,
                         "SG_TUNED_MFMA_SHAPE", ["device"]))

    if "fp8_mfma" in features:
        dims.append(_dim("fp8_mfma_layout", "enum",
                         ["none", "e4m3", "e5m2"],
                         "SG_TUNED_FP8_MFMA_LAYOUT", ["device"]))

    if "fp4_mfma" in features:
        dims.append(_dim("fp4_mfma_layout", "enum",
                         ["none", "e2m1"],
                         "SG_TUNED_FP4_MFMA_LAYOUT", ["device"]))

    # CDNA3+ scheduler hints.
    if "cdna3_plus" in extras:
        dims.append(_dim("scheduler_hint", "enum",
                         ["none", "iglp", "sched_group_barrier"],
                         "SG_TUNED_SCHEDULER_HINT", ["device"]))

    return {
        "dims": dims,
        "prefilter": {
            "register_pressure_max": entry.max_regs_per_thread or 255,
            "waves_per_eu_max": 10,
            "smem_budget_bytes": entry.max_smem_per_block or (64 * 1024),
            "rules": _cdna_common_rules(arch_key),
        },
    }


def _build_rdna_space(arch_key: str,
                     vec_values: Optional[List[int]] = None,
                     unroll_values: Optional[List[int]] = None,
                     stages_values: Optional[List[int]] = None,
                     wmma_shapes: Optional[List[str]] = None,
                     ) -> Dict[str, Any]:
    """RDNA (gfx10xx+) builder. Includes wmma, dpp, tgsplit; no waves_per_eu."""
    entry = ARCH_TABLE[arch_key]
    features = entry.features

    if vec_values is None:
        vec_values = [1, 2, 4, 8, 16]
    if unroll_values is None:
        unroll_values = [1, 2, 4, 8, 16, 32, 64, 128]
    if stages_values is None:
        stages_values = list(range(1, 9))

    dims: List[Dict[str, Any]] = [
        _dim("block", "int", _rdna_block_values(arch_key),
             "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
        _dim("vec", "int", vec_values,
             "SG_TUNED_VEC_WIDTH", ["host", "device"]),
        _dim("unroll", "int", unroll_values,
             "SG_TUNED_UNROLL", ["host", "device"]),
        _dim("num_stages", "int", stages_values,
             "SG_TUNED_NUM_STAGES", ["device"]),
        _dim("maxrregcount", "int", _maxrregcount_values(arch_key),
             None, ["device"]),
        _dim("lds_padding", "int", [0, 4, 8, 16, 32],
             "SG_TUNED_LDS_PADDING", ["device"]),
    ]

    if "wmma" in features:
        if wmma_shapes is None:
            wmma_shapes = ["m16n16k16_fp16", "m16n16k16_bf16",
                           "m16n16k32_int8"]
        dims.append(_dim("wmma_shape", "enum", wmma_shapes,
                         "SG_TUNED_WMMA_SHAPE", ["device"]))

    if "dpp" in features:
        dims.append(_dim("dpp_modifier", "enum",
                         ["none", "quad_perm", "row_shr", "wave_shr"],
                         "SG_TUNED_DPP_MODIFIER", ["device"]))

    if "tgsplit" in features:
        dims.append(_dim("tgsplit", "int", [0, 1],
                         "SG_TUNED_TGSPLIT", ["device"]))

    if "fp8" in features:
        dims.append(_dim("fp8_layout", "enum", ["none", "e4m3", "e5m2"],
                         "SG_TUNED_FP8_LAYOUT", ["device"]))

    return {
        "dims": dims,
        "prefilter": {
            "register_pressure_max": entry.max_regs_per_thread or 255,
            "smem_budget_bytes": entry.max_smem_per_block or (64 * 1024),
            "rules": _rdna_common_rules(arch_key),
        },
    }


# ---- Per-arch CUDA builders ----------------------------------------------

def _sm75_full_space() -> Dict[str, Any]:
    # Turing: no async_copy/bf16; smaller smem (100 KB); drop vec=16 and large unroll.
    return _build_cuda_space(
        "sm_75",
        vec_values=[1, 2, 4, 8],
        unroll_values=[1, 2, 4, 8, 16, 32, 64],
        stages_values=[1, 2, 3, 4],
    )


def _sm80_full_space() -> Dict[str, Any]:
    return _build_cuda_space("sm_80")


def _sm86_full_space() -> Dict[str, Any]:
    return _build_cuda_space("sm_86")


def _sm89_full_space() -> Dict[str, Any]:
    return _build_cuda_space("sm_89")


def _sm90_full_space() -> Dict[str, Any]:
    """Hopper: tma + wgmma + cluster + fp8. Keep the original tight shape."""
    arch_key = "sm_90a"
    entry = ARCH_TABLE[arch_key]
    return {
        "dims": [
            _dim("block", "int", list(range(32, 1025, 32)),
                 "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
            _dim("vec", "int", [1, 2, 4, 8, 16],
                 "SG_TUNED_VEC_WIDTH", ["host", "device"]),
            _dim("unroll", "int", [1, 2, 4, 8, 16, 32, 64, 128],
                 "SG_TUNED_UNROLL", ["host", "device"]),
            _dim("num_stages", "int", list(range(1, 9)),
                 "SG_TUNED_NUM_STAGES", ["device"]),
            _dim("maxrregcount", "int", list(range(32, 253, 4)) + [255],
                 None, ["device"]),
            _dim("cluster_shape", "tuple", _hopper_cluster_shapes(),
                 "SG_TUNED_CLUSTER_SHAPE", ["device"]),
            _dim("swizzle", "enum", ["none", "xor2", "xor4", "xor8", "xor16"],
                 "SG_TUNED_SWIZZLE", ["device"]),
            _dim("warp_specialization", "bool", [False, True],
                 "SG_TUNED_WARP_SPECIALIZATION", ["device"]),
            _dim("tma", "bool", [False, True],
                 "SG_TUNED_TMA", ["device"]),
            _dim("async_depth", "int", list(range(1, 17)),
                 "SG_TUNED_ASYNC_DEPTH", ["device"]),
        ],
        "prefilter": {
            "register_pressure_max": 255,
            "smem_budget_bytes": entry.max_smem_per_block or 232448,
            "rules": [
                {"name": "warps_per_block", "expr": "(block // 32) <= 32"},
                {"name": "vec_block_alignment",
                 "expr": "block % (vec * 4) == 0"},
                {"name": "stages_block",
                 "expr": "num_stages * vec <= block // 32"},
                {"name": "tma_requires_block",
                 "expr": "(not tma) or block >= 128"},
                {"name": "warpspec_requires_block",
                 "expr": "(not warp_specialization) or block >= 128"},
                {"name": "cluster_volume",
                 "expr": "cluster_shape[0] * cluster_shape[1] * cluster_shape[2] <= 8"},
                {"name": "async_depth_stages",
                 "expr": "async_depth >= num_stages - 1"},
            ],
        },
    }


def _sm100a_full_space() -> Dict[str, Any]:
    return _build_cuda_space("sm_100a")


def _sm103a_full_space() -> Dict[str, Any]:
    return _build_cuda_space("sm_103a")


def _sm120a_full_space() -> Dict[str, Any]:
    # Blackwell consumer: fp8 + fp4 but no tma/wgmma/cluster/tcgen05.
    return _build_cuda_space("sm_120a")


# ---- Per-arch HIP builders -----------------------------------------------

def _gfx906_full_space() -> Dict[str, Any]:
    # Vega20: minimal MFMA shapes, shorter pipeline, drop vec=16.
    return _build_cdna_space(
        "gfx906",
        mfma_shapes=["m16n16k4", "m32n32k2", "m4n4k1"],
        vec_values=[1, 2, 4, 8],
        unroll_values=[1, 2, 4, 8, 16, 32, 64],
        stages_values=[1, 2, 3, 4],
    )


def _gfx908_full_space() -> Dict[str, Any]:
    return _build_cdna_space(
        "gfx908",
        mfma_shapes=["m16n16k4", "m32n32k2", "m4n4k1",
                     "m16n16k16_bf16", "m32n32k8_bf16"],
    )


def _gfx90a_full_space() -> Dict[str, Any]:
    return _build_cdna_space(
        "gfx90a",
        mfma_shapes=["m16n16k4", "m32n32k2", "m4n4k1",
                     "m16n16k16_bf16", "m32n32k8_bf16",
                     "m16n16k4_fp64", "m32n32k4_fp64"],
    )


def _gfx942_full_space() -> Dict[str, Any]:
    """CDNA3 (MI300X): full MFMA matrix + fp8 + scheduler hints."""
    arch_key = "gfx942"
    entry = ARCH_TABLE[arch_key]
    return {
        "dims": [
            _dim("block", "int", list(range(64, 1025, 64)),
                 "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
            _dim("vec", "int", [1, 2, 4, 8],
                 "SG_TUNED_VEC_WIDTH", ["host", "device"]),
            _dim("unroll", "int", [1, 2, 4, 8, 16, 32, 64, 128],
                 "SG_TUNED_UNROLL", ["host", "device"]),
            _dim("num_stages", "int", list(range(1, 9)),
                 "SG_TUNED_NUM_STAGES", ["device"]),
            _dim("maxrregcount", "int", list(range(32, 257, 4)),
                 None, ["device"]),
            _dim("waves_per_eu", "int", list(range(1, 11)),
                 "SG_TUNED_WAVES_PER_EU", ["device"]),
            _dim("lds_padding", "int", [0, 1, 2, 4, 8],
                 "SG_TUNED_LDS_PADDING", ["device"]),
            _dim("mfma_shape", "enum",
                 ["m16n16k16", "m32n32k8", "m16n16k32", "m32n32k16",
                  "m16n16k4f64", "m32n32k4f64", "m16n16k64fp8",
                  "m32n32k32fp8", "m16n16k16f32", "m32n32k8f32"],
                 "SG_TUNED_MFMA_SHAPE", ["device"]),
            _dim("scheduler_hint", "enum",
                 ["default", "llvm", "iglp_max_throughput",
                  "iglp_max_throughput_v2", "iglp_gemm", "none"],
                 "SG_TUNED_SCHEDULER_HINT", ["device"]),
        ],
        "prefilter": {
            "register_pressure_max": 256,
            "waves_per_eu_max": 10,
            "smem_budget_bytes": entry.max_smem_per_block or 65536,
            "rules": [
                {"name": "wave_alignment", "expr": "block % 64 == 0"},
                {"name": "waves_per_block", "expr": "(block // 64) <= 16"},
                {"name": "waves_per_eu_total",
                 "expr": "waves_per_eu * (block // 64) <= 20"},
                {"name": "vec_block_alignment",
                 "expr": "block % (vec * 4) == 0"},
                {"name": "mfma_block_min", "expr": "block >= 64"},
            ],
        },
    }


def _gfx950_full_space() -> Dict[str, Any]:
    # CDNA4 (MI350X): adds fp4_mfma.
    return _build_cdna_space(
        "gfx950",
        mfma_shapes=["m16n16k16", "m32n32k8", "m16n16k32",
                     "m32n32k16", "m16n16k64fp8", "m32n32k32fp8",
                     "m16n16k128fp4", "m32n32k64fp4"],
        extra_features=["cdna3_plus"],
    )


def _gfx1030_full_space() -> Dict[str, Any]:
    # RDNA2: no wmma, no dpp/tgsplit features; minimal RDNA shape.
    return _build_rdna_space(
        "gfx1030",
        vec_values=[1, 2, 4, 8],
        unroll_values=[1, 2, 4, 8, 16, 32, 64],
        stages_values=[1, 2, 3, 4],
        wmma_shapes=None,  # gfx1030 has wmma feature flag
    )


def _gfx1100_full_space() -> Dict[str, Any]:
    """RDNA3 (RX 7900, also reused for gfx1101/gfx1102)."""
    return _build_rdna_space(
        "gfx1100",
        wmma_shapes=["m16n16k16_fp16", "m16n16k16_bf16",
                     "m16n16k32_int8", "m16n16k32_int4"],
    )


def _gfx1151_full_space() -> Dict[str, Any]:
    # RDNA3.5 (Strix Halo APU): same WMMA shapes as RDNA3.
    return _build_rdna_space(
        "gfx1151",
        wmma_shapes=["m16n16k16_fp16", "m16n16k16_bf16",
                     "m16n16k32_int8", "m16n16k32_int4"],
    )


def _gfx1200_full_space() -> Dict[str, Any]:
    """RDNA4 (RX 9000, also reused for gfx1201). Adds fp8 + new WMMA shapes."""
    return _build_rdna_space(
        "gfx1200",
        wmma_shapes=["m16n16k16_fp16", "m16n16k16_bf16",
                     "m16n16k32_int8", "m16n16k32_int4",
                     "m16n16k32_fp8_e4m3", "m16n16k32_fp8_e5m2",
                     "m16n16k64_fp4"],
    )


# ---- Per-arch Pallas builders --------------------------------------------

def _pallas_common_dims(extra_block_shapes: Optional[List[Tuple[int, int]]] = None,
                       num_warps: Optional[List[int]] = None,
                       num_stages: Optional[List[int]] = None,
                       ) -> List[Dict[str, Any]]:
    block_shapes = [(64, 64), (128, 128), (256, 256), (64, 256), (256, 64)]
    if extra_block_shapes:
        block_shapes = block_shapes + list(extra_block_shapes)
    if num_warps is None:
        num_warps = [1, 2, 4, 8]
    if num_stages is None:
        num_stages = [1, 2, 3, 4]
    return [
        _dim("block_shape", "tuple", block_shapes,
             None, ["device"], kind="pallas_kwarg"),
        _dim("num_warps", "int", num_warps,
             None, ["device"], kind="pallas_kwarg"),
        _dim("num_stages", "int", num_stages,
             None, ["device"], kind="pallas_kwarg"),
        _dim("dimension_semantics", "tuple",
             [("parallel", "parallel"),
              ("parallel", "sequential"),
              ("sequential", "parallel")],
             None, ["device"], kind="pallas_kwarg"),
    ]


def _pallas_common_prefilter() -> Dict[str, Any]:
    # Pallas spaces are already tiny — just basic positivity.
    return {
        "rules": [
            {"name": "warps_positive", "expr": "num_warps >= 1"},
            {"name": "stages_positive", "expr": "num_stages >= 1"},
        ],
    }


def _tpu_v4_full_space() -> Dict[str, Any]:
    return {
        "dims": _pallas_common_dims(),
        "prefilter": _pallas_common_prefilter(),
    }


def _tpu_v5e_full_space() -> Dict[str, Any]:
    return {
        "dims": _pallas_common_dims(),
        "prefilter": _pallas_common_prefilter(),
    }


def _tpu_v5p_full_space() -> Dict[str, Any]:
    dims = _pallas_common_dims()
    # sparsecore feature → optional sparsecore_axis dim.
    if "sparsecore" in ARCH_TABLE["tpu_v5p"].features:
        dims.append(_dim("sparsecore_axis", "enum", ["none", "0", "1"],
                         None, ["device"], kind="pallas_kwarg"))
    return {
        "dims": dims,
        "prefilter": _pallas_common_prefilter(),
    }


def _tpu_v6e_full_space() -> Dict[str, Any]:
    # Trillium: add core_count.
    dims = _pallas_common_dims()
    dims.append(_dim("core_count", "int", [1, 2],
                     None, ["device"], kind="pallas_kwarg"))
    return {
        "dims": dims,
        "prefilter": _pallas_common_prefilter(),
    }


def _tpu_v7_full_space() -> Dict[str, Any]:
    # Ironwood: larger blocks.
    dims = _pallas_common_dims(
        extra_block_shapes=[(512, 512), (1024, 256), (256, 1024)],
    )
    return {
        "dims": dims,
        "prefilter": _pallas_common_prefilter(),
    }


# ---- Wire builders into ARCH_TABLE ---------------------------------------

# Maps canonical arch keys to their builder. RDNA3 variants (gfx1101/gfx1102)
# reuse the gfx1100 builder; RDNA4 gfx1201 reuses gfx1200.
_ARCH_BUILDERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "sm_75":    _sm75_full_space,
    "sm_80":    _sm80_full_space,
    "sm_86":    _sm86_full_space,
    "sm_89":    _sm89_full_space,
    "sm_90a":   _sm90_full_space,
    "sm_100a":  _sm100a_full_space,
    "sm_103a":  _sm103a_full_space,
    "sm_120a":  _sm120a_full_space,
    "gfx906":   _gfx906_full_space,
    "gfx908":   _gfx908_full_space,
    "gfx90a":   _gfx90a_full_space,
    "gfx942":   _gfx942_full_space,
    "gfx950":   _gfx950_full_space,
    "gfx1030":  _gfx1030_full_space,
    "gfx1100":  _gfx1100_full_space,
    "gfx1101":  _gfx1100_full_space,
    "gfx1102":  _gfx1100_full_space,
    "gfx1151":  _gfx1151_full_space,
    "gfx1200":  _gfx1200_full_space,
    "gfx1201":  _gfx1200_full_space,
    "tpu_v4":   _tpu_v4_full_space,
    "tpu_v5e":  _tpu_v5e_full_space,
    "tpu_v5p":  _tpu_v5p_full_space,
    "tpu_v6e":  _tpu_v6e_full_space,
    "tpu_v7":   _tpu_v7_full_space,
}


def _populate_search_space_builders() -> None:
    """Attach the per-arch search-space builder to every ArchEntry.

    Runs at module import time, after ARCH_TABLE is fully defined and after
    every ``_xxx_full_space()`` function exists. Both the canonical key and
    any aliases share the same ArchEntry instance, so assigning once is
    sufficient — but the dict-iteration form below is robust either way.
    Falls back to ``object.__setattr__`` if ArchEntry ever becomes frozen.
    """
    for arch, builder in _ARCH_BUILDERS.items():
        if arch not in ARCH_TABLE:
            continue
        entry = ARCH_TABLE[arch]
        try:
            entry.search_space_builder = builder
        except (AttributeError, TypeError):
            object.__setattr__(entry, "search_space_builder", builder)


_populate_search_space_builders()


def _canonical_arches() -> List[str]:
    """Distinct canonical arch keys (skips aliases that share an entry)."""
    seen_ids: set = set()
    out: List[str] = []
    for arch, entry in ARCH_TABLE.items():
        if id(entry) in seen_ids:
            continue
        seen_ids.add(id(entry))
        out.append(arch)
    return out


def build_full_search_space() -> Dict[str, Any]:
    """Return the COMPLETE per-arch search space for every arch in ARCH_TABLE.

    Iterates ``ARCH_TABLE.keys()`` (skipping alias entries that share the
    same ``ArchEntry`` instance as a canonical key) and calls each entry's
    ``search_space_builder``. Result is ``{arch: builder_result}``.

    For backward compatibility, alias keys (``sm_90``, ``sm_100``, etc.)
    receive the SAME built dict as their canonical counterpart so a call
    like ``--arch sm_90`` still resolves a search space via ``space[arch]``.
    The canonical 'a'-suffixed key is the source of truth; aliases reuse it.

    Per-arch Cartesian product is enormous (10^6..10^13 per CUDA/HIP, 10..10^6
    per Pallas) — never materialize as a list; use ``cartesian()`` /
    ``cartesian_count()`` / ``ss_prefilter()`` instead.

    Override the whole thing with ``--search-space <path.yaml>`` if you
    genuinely want a smaller curated space (e.g. for fast CI sweeps).
    """
    out: Dict[str, Any] = {}
    # First pass: build for each canonical arch.
    canonical_built: Dict[int, Dict[str, Any]] = {}  # id(entry) -> built dict
    for arch in _canonical_arches():
        entry = ARCH_TABLE[arch]
        builder = entry.search_space_builder
        if builder is None:
            continue
        built = builder()
        out[arch] = built
        canonical_built[id(entry)] = built
    # Second pass: replicate to aliases (shares the same ArchEntry instance).
    for arch, entry in ARCH_TABLE.items():
        if arch in out:
            continue
        if id(entry) in canonical_built:
            out[arch] = canonical_built[id(entry)]
    return out


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


# ===========================================================================
# search_space — YAML-driven autotune search space (absorbed from
# grokking_optimizers/search_space.py)
# ===========================================================================

class SearchSpaceError(ValueError):
    """Raised when the YAML is missing required keys or has a bad type."""


_VALID_TYPES = {"int", "bool", "enum", "tuple"}
_VALID_TARGETS = {"host", "device"}


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load the YAML file and validate the per-arch shape."""
    path = Path(path)
    if not path.exists():
        raise SearchSpaceError(f"search-space YAML not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SearchSpaceError(
            f"search-space YAML must be a top-level dict; got {type(raw).__name__}")
    for arch, block in raw.items():
        _validate_arch(arch, block)
    return raw


def load_embedded_search_space() -> Dict[str, Any]:
    """Return the COMPLETE programmatic search space (no curation).

    Equivalent to ``build_full_search_space()`` plus the standard shape
    validation. Bayesian TPE will sample this space; ``cartesian_count``
    reports its size (billions per arch); ``cartesian()`` streams the
    Cartesian product; ``ss_prefilter()`` filters lazily."""
    raw = build_full_search_space()
    for arch, block in raw.items():
        _validate_arch(arch, block)
    return raw


def get_search_space(path: Optional[Path]) -> Dict[str, Any]:
    """Return the search space dict.

    When ``path`` is None, returns the full programmatic space from
    ``build_full_search_space()``. When ``path`` points at a YAML file,
    loads that — for users who genuinely want a smaller curated space.
    """
    if path is None:
        return load_embedded_search_space()
    return load_yaml(path)


def _validate_arch(arch: str, block: Any) -> None:
    if not isinstance(block, dict):
        raise SearchSpaceError(f"{arch}: expected dict, got {type(block).__name__}")
    dims = block.get("dims", [])
    if not isinstance(dims, list):
        raise SearchSpaceError(f"{arch}.dims must be a list")
    seen_names: set = set()
    for dim in dims:
        _validate_dim(arch, dim, seen_names)
        seen_names.add(dim["name"])
    prefilter_block = block.get("prefilter", {})
    if prefilter_block and not isinstance(prefilter_block, dict):
        raise SearchSpaceError(f"{arch}.prefilter must be a dict")
    rules = prefilter_block.get("rules", []) if prefilter_block else []
    if rules and not isinstance(rules, list):
        raise SearchSpaceError(f"{arch}.prefilter.rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict) or "expr" not in rule:
            raise SearchSpaceError(
                f"{arch}.prefilter.rules each entry must be a dict with 'expr'")


def _validate_dim(arch: str, dim: Any, seen: set) -> None:
    if not isinstance(dim, dict):
        raise SearchSpaceError(f"{arch}: dim entry must be a dict, got {type(dim).__name__}")
    for required in ("name", "type", "values"):
        if required not in dim:
            raise SearchSpaceError(f"{arch}: dim missing '{required}': {dim}")
    name = dim["name"]
    if not isinstance(name, str):
        raise SearchSpaceError(f"{arch}: dim 'name' must be a str: {dim}")
    if name in seen:
        raise SearchSpaceError(f"{arch}: duplicate dim name {name!r}")
    if dim["type"] not in _VALID_TYPES:
        raise SearchSpaceError(
            f"{arch}.{name}: type {dim['type']!r} not in {_VALID_TYPES}")
    if not isinstance(dim["values"], list) or not dim["values"]:
        raise SearchSpaceError(f"{arch}.{name}: values must be a non-empty list")
    applies_to = dim.get("applies_to", ["host", "device"])
    if not isinstance(applies_to, list):
        raise SearchSpaceError(f"{arch}.{name}: applies_to must be a list")
    bad = set(applies_to) - _VALID_TARGETS
    if bad:
        raise SearchSpaceError(
            f"{arch}.{name}: applies_to has unknown targets {bad}")


def cartesian(space: Dict[str, Any], arch: str) -> Iterator[Dict[str, Any]]:
    """Stream the full Cartesian product as {dim_name: value} dicts.

    Generator (lazy). The full programmatic search space is billions of
    configs per arch — never call ``list(cartesian(...))`` on it. Use
    ``cartesian_count()`` for the size, and pipe this iterator through
    ``ss_prefilter()`` which yields survivors lazily.
    """
    if arch not in space:
        return
    dims = space[arch].get("dims", [])
    if not dims:
        return
    names = [d["name"] for d in dims]
    values = [d["values"] for d in dims]
    for combo in itertools.product(*values):
        cfg: Dict[str, Any] = {}
        for n, v in zip(names, combo):
            cfg[n] = tuple(v) if isinstance(v, list) else v
        yield cfg


def cartesian_count(space: Dict[str, Any], arch: str) -> int:
    """Return the size of ``cartesian(space, arch)`` without iterating.

    Product of len(dim.values) across every dim. For the full programmatic
    space this is in the billions per arch — cheap to compute, used for
    the ``[prefilter] N candidates → ...`` display line.
    """
    if arch not in space:
        return 0
    dims = space[arch].get("dims", [])
    if not dims:
        return 0
    n = 1
    for d in dims:
        n *= len(d.get("values", []) or [])
    return n


def ss_prefilter(configs: Iterable[Dict[str, Any]],
                 prefilter_spec: Dict[str, Any],
                 max_survivors: Optional[int] = None,
                 ) -> Tuple[List[Dict[str, Any]], int]:
    """Apply the static pruning rules. Returns (survivors, eliminated_count).

    Accepts any iterable for ``configs`` (typically the lazy
    ``cartesian()`` generator). Survivors are materialized as a list —
    callers needing to also stream survivors should use
    ``iter_prefilter()`` below.

    When ``max_survivors`` is set, the loop **breaks early** as soon as
    that many survivors are collected. ``eliminated_count`` reflects
    only what was rejected up to that point — there may be many more
    candidates beyond that the prefilter never visited. Use this to
    cap exhaustive sweeps against the full multi-billion-config space.
    """
    if not prefilter_spec:
        if max_survivors is None:
            return list(configs), 0
        return [c for c, _ in zip(configs, range(max_survivors))], 0
    survivors: List[Dict[str, Any]] = []
    eliminated = 0
    for cfg, passed in _iter_prefilter_with_status(configs, prefilter_spec):
        if passed:
            survivors.append(cfg)
            if max_survivors is not None and len(survivors) >= max_survivors:
                break
        else:
            eliminated += 1
    return survivors, eliminated


def iter_prefilter(configs: Iterable[Dict[str, Any]],
                   prefilter_spec: Dict[str, Any]
                   ) -> Iterator[Dict[str, Any]]:
    """Yield surviving configs lazily. Use when you can't afford to
    materialize the survivor list (full programmatic space → millions
    or tens of millions of survivors). Eliminated count is NOT tracked
    in this variant — use ``ss_prefilter`` if you need it."""
    for cfg, passed in _iter_prefilter_with_status(configs, prefilter_spec):
        if passed:
            yield cfg


_PREFILTER_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "abs": abs,
    "int": int, "bool": bool, "True": True, "False": False,
}


def compile_feasibility_check(prefilter_spec: Dict[str, Any]
                              ) -> Callable[[Dict[str, Any]], bool]:
    """Return a single-config feasibility predicate.

    Used by Bayesian TPE to validate each suggestion in O(rules) time
    without ever materializing the prefiltered survivor set. With the
    full programmatic search space (billions of candidates), enumerating
    survivors up-front is infeasible — but checking one TPE-sampled
    cfg against the rules is cheap.
    """
    rules: List[Dict[str, Any]] = (prefilter_spec or {}).get("rules", []) or []
    compiled = [compile(r["expr"], "<prefilter>", "eval") for r in rules]

    def check(cfg: Dict[str, Any]) -> bool:
        if not compiled:
            return True
        for code in compiled:
            try:
                env = dict(cfg)
                env["__builtins__"] = _PREFILTER_SAFE_BUILTINS
                if not bool(eval(code, env, env)):  # noqa: S307 — sandboxed
                    return False
            except Exception:
                return False
        return True
    return check


def _iter_prefilter_with_status(configs: Iterable[Dict[str, Any]],
                                prefilter_spec: Dict[str, Any]
                                ) -> Iterator[Tuple[Dict[str, Any], bool]]:
    """Internal helper: yield (cfg, passed) pairs as we evaluate each one."""
    check = compile_feasibility_check(prefilter_spec)
    for cfg in configs:
        yield cfg, check(cfg)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    return str(value)


def resolve_macros(config: Dict[str, Any], dim_specs: List[Dict[str, Any]],
                   target: str) -> List[str]:
    """Return the ``-DFOO=VAL`` flags for a config restricted to ``target``."""
    if target not in _VALID_TARGETS:
        raise ValueError(f"target={target!r} not in {_VALID_TARGETS}")
    out: List[str] = []
    for spec in dim_specs:
        macro = spec.get("macro")
        applies_to = spec.get("applies_to", ["host", "device"])
        if target not in applies_to:
            continue
        if not macro:
            continue
        name = spec["name"]
        if name not in config:
            continue
        out.append(f"-D{macro}={_format_value(config[name])}")
    return out


def resolve_extra_nvcc_flags(config: Dict[str, Any],
                              dim_specs: List[Dict[str, Any]],
                              arch: Optional[str] = None) -> List[str]:
    """Some dims become bare NVCC flags rather than ``-D`` macros, plus
    per-arch feature gates and per-config dtype/layout flags.

    Stream 3: replaces the previous ``if arch == "sm_90"`` branching
    with feature-table lookups so any NVIDIA arch with the right
    capability picks up the corresponding macro automatically.

    Currently emits:
      - ``maxrregcount`` -> ``--maxrregcount=N``
      - feature macros for arch capabilities (TMA, fp4/fp8, cluster, ...)
      - layout macros for tuned dtypes (e.g. ``fp8_layout=e4m3`` -> ``-DCUDA_FP8_E4M3=1``)
    """
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.append(f"--maxrregcount={int(v)}")

    if arch and arch in ARCH_TABLE:
        entry = get_arch_entry(arch)
        if entry.vendor == "cuda":
            feats = entry.features
            if "tma" in feats:
                out.append("-DCUDA_TMA_ENABLED=1")
            if "wgmma" in feats:
                out.append("-DCUDA_WGMMA_ENABLED=1")
            if "cluster" in feats:
                out.append("-DCUDA_CLUSTER_ENABLED=1")
            if "fp8" in feats:
                out.append("-DCUDA_FP8_ENABLED=1")
            if "fp4" in feats:
                out.append("-DCUDA_FP4_ENABLED=1")
            if "tcgen05" in feats:
                out.append("-DCUDA_TCGEN05_ENABLED=1")
            # FP8 / FP4 sub-format layouts driven by search-space dim values.
            # Stream 2's space exposes these as tunables; map the chosen
            # variant to a -D so the header picks the right template.
            fp8 = str(config.get("fp8_layout", "")).lower()
            if fp8 in ("e4m3", "e5m2"):
                out.append(f"-DCUDA_FP8_{fp8.upper()}=1")
            fp4 = str(config.get("fp4_layout", "")).lower()
            if fp4 in ("e2m1", "mx"):
                out.append(f"-DCUDA_FP4_{fp4.upper()}=1")
    return out


def resolve_extra_hipcc_flags(config: Dict[str, Any],
                               dim_specs: List[Dict[str, Any]],
                               arch: Optional[str] = None) -> List[str]:
    """HIPCC analogue of resolve_extra_nvcc_flags.

    Currently emits:
      - ``maxrregcount`` -> ``-mllvm -amdgpu-max-num-vgprs=N``
      - feature macros for AMDGPU capabilities (MFMA / WMMA / fp8 / fp4 / dpp / tgsplit)
      - layout macros for tuned dtypes (``fp8_layout``, ``fp4_layout``)
    """
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.extend(["-mllvm", f"-amdgpu-max-num-vgprs={int(v)}"])

    if arch and arch in ARCH_TABLE:
        entry = get_arch_entry(arch)
        if entry.vendor == "hip":
            feats = entry.features
            if "mfma" in feats:
                out.append("-DAMDGPU_MFMA_ENABLED=1")
            if "wmma" in feats:
                out.append("-DAMDGPU_WMMA_ENABLED=1")
            if "bf16_mfma" in feats:
                out.append("-DAMDGPU_BF16_MFMA=1")
            if "fp8_mfma" in feats:
                out.append("-DAMDGPU_FP8_MFMA=1")
            if "fp4_mfma" in feats:
                out.append("-DAMDGPU_FP4_MFMA=1")
            if "tgsplit" in feats:
                out.append("-DAMDGPU_TGSPLIT=1")
            if "dpp" in feats:
                out.append("-DAMDGPU_DPP=1")
            if "fp8" in feats:
                out.append("-DAMDGPU_FP8_ENABLED=1")
            fp8 = str(config.get("fp8_layout", "")).lower()
            if fp8 in ("e4m3", "e5m2", "bf8", "fnuz"):
                out.append(f"-DAMDGPU_FP8_{fp8.upper()}=1")
            fp4 = str(config.get("fp4_layout", "")).lower()
            if fp4 in ("e2m1", "mx", "ocp"):
                out.append(f"-DAMDGPU_FP4_{fp4.upper()}=1")
    return out


def hash_space(space: Dict[str, Any], arch: str) -> str:
    """Stable SHA-256 of the per-arch space (sorted JSON, no whitespace)."""
    if arch not in space:
        return hashlib.sha256(b"").hexdigest()
    block = space[arch]
    payload = json.dumps(block, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_key(config: Dict[str, Any]) -> str:
    """Compact, deterministic key — used as cache.variant_artifacts subkey."""
    parts = []
    for k in sorted(config.keys()):
        parts.append(f"{k}={_format_value(config[k])}")
    return "_".join(parts)


# ===========================================================================
# pgo — Profile-Guided Optimisation driver (absorbed from
# grokking_optimizers/pgo.py)
# ===========================================================================

def instrument_flags(
    arch: str,
    profile_dir: Path,
    host_cflags: List[str],
    device_cflags: List[str],
    ldflags: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Append profile-generate flags. ``profile_dir`` is the .gcda output dir."""
    profile_dir = Path(profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    h = list(host_cflags) + [
        f"-fprofile-generate={profile_dir}",
        "-fprofile-update=atomic",
    ]
    d: List[str] = list(device_cflags)
    entry = get_arch_entry(arch) if arch in ARCH_TABLE else None
    if entry is not None and entry.vendor == "cuda":
        # NVCC needs -Xcompiler to forward flags to the host compiler.
        d += [
            "-Xcompiler", f"-fprofile-generate={profile_dir}",
            "-Xcompiler", "-fprofile-update=atomic",
        ]
    elif entry is not None and entry.vendor == "hip":
        # HIPCC drives clang directly; flags pass straight through.
        d += [
            f"-fprofile-generate={profile_dir}",
            "-fprofile-update=atomic",
        ]
    l = list(ldflags) + [f"-fprofile-generate={profile_dir}"]
    return h, d, l


def use_flags(
    arch: str,
    profile_dir: Path,
    host_cflags: List[str],
    device_cflags: List[str],
    ldflags: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Append profile-use flags."""
    profile_dir = Path(profile_dir).resolve()

    h = list(host_cflags) + [
        f"-fprofile-use={profile_dir}",
        "-fprofile-correction",
    ]
    d: List[str] = list(device_cflags)
    entry = get_arch_entry(arch) if arch in ARCH_TABLE else None
    if entry is not None and entry.vendor == "cuda":
        d += [
            "-Xcompiler", f"-fprofile-use={profile_dir}",
            "-Xcompiler", "-fprofile-correction",
        ]
    elif entry is not None and entry.vendor == "hip":
        d += [
            f"-fprofile-use={profile_dir}",
            "-fprofile-correction",
        ]
    l = list(ldflags) + [f"-fprofile-use={profile_dir}"]
    return h, d, l


def hash_workload(workload_script: Path, steps: int) -> str:
    """SHA-256 of (workload file contents, step count)."""
    h = hashlib.sha256()
    try:
        h.update(Path(workload_script).read_bytes())
    except OSError:
        h.update(b"<missing-workload>")
    h.update(b"\0")
    h.update(str(int(steps)).encode("ascii"))
    return h.hexdigest()


def collect_workload(
    workload_script: Path,
    *,
    so_path: Path,
    opt_class: str,
    model: str,
    arch: str,
    profile_dir: Path,
    steps: int = 1000,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    report=None,
) -> bool:
    """Run the workload subprocess and confirm profile files appear.

    Returns ``True`` if at least one ``.gcda`` (or any file) was created
    in ``profile_dir`` and the workload exited cleanly.
    """
    workload_script = Path(workload_script).resolve()
    if not workload_script.exists():
        if report:
            report.write(f"  [pgo collect] workload not found: {workload_script}\n")
        return False
    profile_dir = Path(profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_size for p in profile_dir.glob("**/*") if p.is_file()}

    sub_env = (env or os.environ).copy()
    sub_env["LLVM_PROFILE_FILE"] = str(profile_dir / "default_%p.profraw")
    sub_env["GCOV_PREFIX"] = str(profile_dir)

    cmd = [
        sys.executable, str(workload_script),
        "--so",   str(so_path),
        "--opt",  opt_class,
        "--model", model,
        "--arch", arch,
        "--steps", str(int(steps)),
    ]
    if report:
        report.write(f"\n$ {' '.join(cmd)}\n")
        report.flush()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, env=sub_env,
        )
    except subprocess.TimeoutExpired:
        if report:
            report.write(f"  [pgo collect] TIMEOUT after {timeout}s\n")
        return False
    elapsed = time.monotonic() - t0
    if report:
        report.write(proc.stdout or "")
        report.write(f"\n[exit {proc.returncode} in {elapsed:.1f}s]\n")
    if proc.returncode != 0:
        return False

    after = {p: p.stat().st_size for p in profile_dir.glob("**/*") if p.is_file()}
    new_or_grown = [p for p, sz in after.items()
                    if before.get(p, -1) != sz]
    if not new_or_grown:
        if report:
            report.write(
                f"  [pgo collect] no profile files appeared under {profile_dir}\n")
        return False
    if report:
        report.write(
            f"  [pgo collect] {len(new_or_grown)} profile file(s) updated\n")
    return True


# ===========================================================================
# pgo_workload — default PGO workload (absorbed from scripts/pgo_workload.py)
# ===========================================================================

def _pgo_workload_main() -> int:
    """Default PGO workload entry point (was scripts/pgo_workload.py)."""
    import argparse as _ap
    parser = _ap.ArgumentParser(
        description="Default PGO workload — runs N optimizer steps")
    parser.add_argument("--so", type=Path, required=True,
                        help="Path to the instrumented .so")
    parser.add_argument("--opt", required=True,
                        help="Optimizer class name (e.g. Lion, AdamW)")
    parser.add_argument("--model", default="mamba",
                        help="Model identifier (informational; not loaded)")
    parser.add_argument("--arch", default="sm_90",
                        help="Arch identifier (informational; not loaded)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of optimizer.step() calls")
    parser.add_argument("--size", type=int, default=2048,
                        help="Parameter size (size x size float32 tensor)")
    args = parser.parse_args()

    if args.so:
        import importlib.util as _ilu
        if "grokking_optimizers._ops" in sys.modules:
            del sys.modules["grokking_optimizers._ops"]
        _spec = _ilu.spec_from_file_location(
            "grokking_optimizers._ops", str(args.so))
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"could not load .so: {args.so}")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # type: ignore[arg-type]
        sys.modules["grokking_optimizers._ops"] = _mod

    import torch
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, args.opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(args.size, args.size, device=device, dtype=torch.float32))
    target = torch.randn_like(p)

    opt = OptCls([p], lr=1e-3)
    for _ in range(args.steps):
        loss = ((p - target) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"PGO workload OK ({args.opt}/{args.model}/{args.arch}): "
          f"{args.steps} steps, size={args.size}, device={device}")
    return 0


# ===========================================================================
# bench_graph — CUDA / HIP graph-replay timing (absorbed from
# grokking_optimizers/bench_graph.py)
# ===========================================================================

def _build_param(opt_class_name: str, size: int) -> Any:
    import torch
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(size, size, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    p.grad = g
    return p, g


def cuda_graph_median_ms(opt_class: str, *, size: int = 4096,
                         warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """Time ``opt.step()`` under a CUDA graph replay."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_median_ms requires torch.cuda.is_available()")
    from grokking_optimizers import _ops  # noqa: F401 — must be loaded first
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)

    p, g = _build_param(opt_class, size)
    opt = OptCls([p], lr=1e-3)

    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt.step()
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        p.grad = g.clone()
        opt.step()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        p.grad.copy_(g)
        opt.step()

    timings = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return {
        "timing_ms": float(timings[len(timings) // 2]),
        "min_ms":    float(timings[0]),
        "max_ms":    float(timings[-1]),
        "n":         len(timings),
    }


def hip_graph_median_ms(opt_class: str, *, size: int = 4096,
                        warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """HIP analogue — reuses cuda_graph_median_ms (ROCm uses same namespace)."""
    return cuda_graph_median_ms(opt_class, size=size, warmup=warmup, iters=iters)


def event_median_ms(opt_class: str, *, size: int = 4096,
                    warmup: int = 5, iters: int = 21) -> Dict[str, float]:
    """Fallback timer for archs / backends that do not support graph capture."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("event_median_ms requires torch.cuda.is_available()")
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)

    p, g = _build_param(opt_class, size)
    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return {
        "timing_ms": float(timings[len(timings) // 2]),
        "min_ms":    float(timings[0]),
        "max_ms":    float(timings[-1]),
        "n":         len(timings),
    }


# ===========================================================================
# timing_worker — persistent timing subprocess (absorbed from
# grokking_optimizers/timing_worker.py)
# ===========================================================================

_WORKER_BODY = r"""
import sys, json, importlib.util, traceback, time
try:
    import torch
except ImportError as exc:
    sys.stdout.write(json.dumps({"error": "torch import failed: " + str(exc)}) + "\n")
    sys.stdout.flush()
    sys.exit(1)


def _load_so(so_path):
    if "grokking_optimizers._ops" in sys.modules:
        del sys.modules["grokking_optimizers._ops"]
    spec = importlib.util.spec_from_file_location(
        "grokking_optimizers._ops", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["grokking_optimizers._ops"] = mod
    return mod


def _bg_build_param(opt_class_name, size):
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(size, size, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    p.grad = g
    return p, g


def _bg_cuda_graph_median_ms(opt_class, *, size=4096, warmup=5, iters=21):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_median_ms requires torch.cuda.is_available()")
    from grokking_optimizers import _ops  # noqa: F401
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)

    p, g = _bg_build_param(opt_class, size)
    opt = OptCls([p], lr=1e-3)

    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt.step()
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        p.grad = g.clone()
        opt.step()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        p.grad.copy_(g)
        opt.step()

    timings = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    timings.sort()
    return {
        "timing_ms": float(timings[len(timings) // 2]),
        "min_ms":    float(timings[0]),
        "max_ms":    float(timings[-1]),
        "n":         len(timings),
    }


def _time_with_graph(opt_class, size, warmup, iters):
    try:
        return _bg_cuda_graph_median_ms(
            opt_class, size=size, warmup=warmup, iters=iters)
    except Exception:
        return None


def _time_with_events(opt_class, size, warmup, iters):
    from importlib import import_module
    grok = import_module("grokking_optimizers")
    OptCls = getattr(grok, opt_class)
    torch.manual_seed(0)
    p = torch.nn.Parameter(
        torch.randn(size, size, device="cuda", dtype=torch.float32))
    g = torch.randn_like(p)
    for _ in range(max(1, warmup)):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        opt.step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        p.grad = g.clone()
        opt = OptCls([p], lr=1e-3)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return {
        "timing_ms": float(times[len(times) // 2]),
        "min_ms":    float(times[0]),
        "max_ms":    float(times[-1]),
        "n":         len(times),
    }


def main():
    if not torch.cuda.is_available():
        sys.stdout.write(json.dumps(
            {"error": "torch.cuda.is_available() == False"}) + "\n")
        sys.stdout.flush()
        sys.exit(2)
    torch.cuda.synchronize()
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    use_graph = True
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps(
                {"error": "bad request: " + str(exc)}) + "\n")
            sys.stdout.flush()
            continue
        op = req.get("op")
        if op == "ping":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            continue
        if op == "shutdown":
            sys.stdout.write(json.dumps({"bye": True}) + "\n")
            sys.stdout.flush()
            return
        if op != "time":
            sys.stdout.write(json.dumps(
                {"error": "unknown op " + str(op)}) + "\n")
            sys.stdout.flush()
            continue
        try:
            so_path = req["so_path"]
            opt_class = req["opt_class"]
            size = int(req.get("size", 4096))
            warmup = int(req.get("warmup", 5))
            iters = int(req.get("iters", 21))
            _load_so(so_path)
            result = None
            if use_graph and req.get("use_cuda_graph", True):
                result = _time_with_graph(opt_class, size, warmup, iters)
            if result is None:
                result = _time_with_events(opt_class, size, warmup, iters)
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps(
                {"error": str(exc), "tb": traceback.format_exc()}) + "\n")
            sys.stdout.flush()


main()
"""


# ---------------------------------------------------------------------------
# PallasTimer — TPU/Pallas analog of TimingWorker
# ---------------------------------------------------------------------------
#
# CUDA/HIP backends time a variant by building a .so and replaying it via a
# warm subprocess. Pallas has no .so — every "build" is a JAX trace + XLA
# compile. PallasTimer caches the jitted callable per kwargs combo, then
# replays N iterations in-process with jax.block_until_ready() to force
# device sync. JAX is imported lazily so this class is importable on
# CPU-only / no-JAX hosts; time_config() raises RuntimeError if JAX or the
# launcher module is unavailable.

class PallasTimer:
    """Time a Pallas kernel by JIT-compiling pl.pallas_call(...) and replaying
    N iterations. Python-only, no .so to load."""

    def __init__(self, launcher_path: Path, optimizer: str, *,
                 warmup: int = 5, iters: int = 21, problem_size: int = 4096):
        self.launcher_path = Path(launcher_path)
        self.optimizer = optimizer
        self.warmup = warmup
        self.iters = iters
        self.problem_size = problem_size
        self._cached_jit: Dict[Tuple, Any] = {}   # key (kwargs frozen) -> jit-compiled fn
        self._launch_fn = None
        self._launch_sig = None

    def _load_launcher(self):
        """Lazily import jax + the launcher module. Raises RuntimeError on
        failure (so PallasTimer stays importable on hosts without JAX)."""
        if self._launch_fn is not None:
            return
        try:
            import importlib.util
            import inspect
        except ImportError as exc:  # pragma: no cover (stdlib)
            raise RuntimeError(f"Pallas requires importlib/inspect: {exc}")
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"Pallas requires jax: {exc}")
        if not self.launcher_path.is_file():
            raise RuntimeError(
                f"Pallas launcher not found: {self.launcher_path}")
        spec = importlib.util.spec_from_file_location(
            f"_pallas_launcher_{self.optimizer}", str(self.launcher_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"could not load spec for {self.launcher_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Auto-discover launch_<opt>_step_jit / launch_<opt>_step / first launch_*
        launch_fn = None
        for cand in (f"launch_{self.optimizer}_step_jit",
                     f"launch_{self.optimizer}_step"):
            if hasattr(mod, cand):
                launch_fn = getattr(mod, cand)
                break
        if launch_fn is None:
            for name in sorted(dir(mod)):
                if name.startswith("launch_") and callable(getattr(mod, name)):
                    launch_fn = getattr(mod, name)
                    break
        if launch_fn is None:
            raise RuntimeError(f"no launch_* in {self.launcher_path}")
        self._launch_fn = launch_fn
        try:
            self._launch_sig = inspect.signature(launch_fn)
        except (TypeError, ValueError):
            self._launch_sig = None

    def _build_args(self, kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """Synthesize positional+scalar arguments for the launcher based on
        its signature. Tensor-shaped params get a problem_size vector;
        scalar params draw from a small defaults table; any kwargs the
        caller passed override those defaults."""
        import jax.numpy as jnp

        N = self.problem_size
        scalar_defaults: Dict[str, Any] = {
            "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
            "wd": 0.01, "bc1": 1.0, "bc2": 1.0,
            "alpha": 0.98, "lamb": 5.0, "gamma": 0.01,
            "rho": 0.05, "d_coef": 1.0, "scale": 1.0, "step": 1,
            "t": 1, "k": 1,
        }
        dummy_arr_fn = lambda: jnp.zeros((N,), dtype=jnp.float32)
        dummy_grad_fn = lambda: jnp.ones((N,), dtype=jnp.float32)

        # If we have a signature, walk it and fill each param.
        pos_args: List[Any] = []
        kw_args: Dict[str, Any] = {}
        if self._launch_sig is not None:
            tensor_slot = 0
            tensor_makers = [dummy_arr_fn, dummy_grad_fn, dummy_arr_fn,
                             dummy_arr_fn, dummy_arr_fn, dummy_arr_fn]
            for name, p in self._launch_sig.parameters.items():
                if name in kwargs:
                    val = kwargs[name]
                elif name in scalar_defaults:
                    val = scalar_defaults[name]
                else:
                    ann = p.annotation
                    if ann is float or "float" in str(ann):
                        val = 1e-3
                    elif ann is int or "int" in str(ann):
                        val = 1
                    else:
                        maker = tensor_makers[tensor_slot] \
                            if tensor_slot < len(tensor_makers) else dummy_arr_fn
                        val = maker()
                        tensor_slot += 1
                # If callable (closure-bound), evaluate.
                if callable(val) and not isinstance(val, (int, float, bool, str)):
                    try:
                        val = val()
                    except Exception:
                        pass
                kw_args[name] = val
            return pos_args, kw_args

        # No signature — fall back to a minimal AdamW-style argpack.
        pos_args = [dummy_arr_fn(), dummy_grad_fn(),
                    dummy_arr_fn(), dummy_arr_fn(), jnp.float32(1e-3)]
        return pos_args, dict(kwargs)

    def time_config(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return median ms over self.iters replays as a result dict, or
        None on launcher signature mismatch / runtime error."""
        self._load_launcher()
        import jax
        import jax.numpy as jnp  # noqa: F401
        import functools
        import time as _time

        # Freeze kwargs into a hashable cache key.
        key = tuple(sorted((k, (tuple(v) if isinstance(v, list) else v))
                           for k, v in kwargs.items()))
        if key not in self._cached_jit:
            # Build the call (kwargs are NOT static_argnames-friendly across
            # arbitrary launcher signatures; bake them in via partial then
            # jit the resulting tensor-only callable).
            launch_fn = self._launch_fn
            assert launch_fn is not None

            def _call(*args, **kw):
                return launch_fn(*args, **kw)

            self._cached_jit[key] = jax.jit(_call)

        jit_fn = self._cached_jit[key]
        pos_args, kw_args = self._build_args(kwargs)

        # Warmup — also catches signature mismatches early.
        try:
            for _ in range(self.warmup):
                out = jit_fn(*pos_args, **kw_args)
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready()
                    if hasattr(x, "block_until_ready") else x, out)
        except TypeError:
            # Signature mismatch — retry with positional-only fallback.
            try:
                if self._launch_sig is not None:
                    # Drop kwargs that aren't in signature; use positional.
                    pos_args = list(kw_args.values())
                    kw_args = {}
                for _ in range(self.warmup):
                    out = jit_fn(*pos_args, **kw_args)
                    jax.tree_util.tree_map(
                        lambda x: x.block_until_ready()
                        if hasattr(x, "block_until_ready") else x, out)
            except Exception:
                return None
        except Exception:
            return None

        # Timed iterations.
        times: List[float] = []
        try:
            for _ in range(self.iters):
                t0 = _time.perf_counter()
                out = jit_fn(*pos_args, **kw_args)
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready()
                    if hasattr(x, "block_until_ready") else x, out)
                times.append((_time.perf_counter() - t0) * 1000.0)
        except Exception:
            return None
        times.sort()
        return {
            "timing_ms": times[len(times) // 2],
            "min_ms":    times[0],
            "max_ms":    times[-1],
            "n":         len(times),
        }


class TimingWorker:
    """Persistent timing subprocess; one warm CUDA context for an entire sweep.

    A background watchdog thread pings the worker every 30s. If a ping
    fails or no pong arrives, and the last good pong was >60s ago, the
    subprocess is SIGKILL'd and re-started transparently so callers can
    keep using the same worker object across a sweep even when the
    target GPU wedges or OOMs.
    """

    def __init__(self, opt_class: str, *,
                 size: int = 4096, warmup: int = 5, iters: int = 21,
                 use_cuda_graph: bool = True,
                 timeout_per_variant: int = 180,
                 env: Optional[Dict[str, str]] = None,
                 env_overlay: Optional[Dict[str, str]] = None,
                 cwd: Optional[Path] = None,
                 python: Optional[str] = None,
                 watchdog_interval_s: float = 30.0,
                 watchdog_grace_s: float = 60.0,
                 enable_watchdog: bool = True):
        self.opt_class = opt_class
        self.size = size
        self.warmup = warmup
        self.iters = iters
        self.use_cuda_graph = use_cuda_graph
        self.timeout = timeout_per_variant
        # Merge env_overlay (per-GPU overrides) on top of the base env so
        # MultiGPUTimingPool can pin each worker to a single device via
        # CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES without mutating the
        # parent process env.
        base_env = env if env is not None else os.environ.copy()
        if env_overlay:
            base_env = dict(base_env)
            base_env.update(env_overlay)
        self.env = base_env
        self.env_overlay = dict(env_overlay) if env_overlay else None
        self.cwd = cwd
        self.python = python or sys.executable
        self._proc: Optional[subprocess.Popen] = None
        self._error_log: list = []
        # ---- watchdog ----
        self._watchdog_interval = float(watchdog_interval_s)
        self._watchdog_grace = float(watchdog_grace_s)
        self._enable_watchdog = bool(enable_watchdog)
        self._watchdog_stop = threading.Event()
        self._last_pong_ts = time.time()
        self._watchdog_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Spawn the subprocess and wait for its ``{"ready": true}`` ack."""
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._proc = subprocess.Popen(
            [self.python, "-u", "-c", _WORKER_BODY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
            cwd=str(self.cwd) if self.cwd else None,
        )
        ready = self._read_line(timeout=30)
        if not ready or ready.get("ready") is not True:
            err = ready.get("error") if ready else "no response"
            self._error_log.append(("start", err))
            self.stop()
            return False
        self._last_pong_ts = time.time()
        self._ensure_watchdog()
        return True

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restart(self) -> bool:
        # Preserve watchdog across restarts: only stop the subprocess,
        # not the watchdog thread.
        self._terminate_proc()
        ok = self.start()
        return ok

    def stop(self) -> None:
        # Tell the watchdog to exit first so it doesn't race with the
        # shutdown handshake below by trying to send a ping mid-tear-down.
        self._watchdog_stop.set()
        wd = self._watchdog_thread
        if wd is not None and wd.is_alive() and wd is not threading.current_thread():
            wd.join(timeout=2.0)
        self._watchdog_thread = None
        self._terminate_proc()

    def _terminate_proc(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin:
                try:
                    self._proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    # ------------------------------------------------------------------
    # Watchdog — pings the worker every ``watchdog_interval_s``. If no
    # pong arrives within ``watchdog_grace_s`` of the last good pong,
    # SIGKILL the subprocess and restart it.
    # ------------------------------------------------------------------

    def _ensure_watchdog(self) -> None:
        if not self._enable_watchdog:
            return
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True,
            name=f"TimingWorker-watchdog-{id(self):x}")
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            # wait() returns True if the event was set (stop signal)
            if self._watchdog_stop.wait(self._watchdog_interval):
                return
            try:
                ok = self.ping(_from_watchdog=True)
                if ok:
                    self._last_pong_ts = time.time()
                    continue
            except Exception as exc:  # noqa: BLE001 — log + continue to age check
                self._error_log.append(("watchdog_ping", str(exc)))
            # No pong this cycle — check how stale the last good pong is.
            age = time.time() - self._last_pong_ts
            if age > self._watchdog_grace:
                self._error_log.append(("watchdog", f"no pong for {age:.1f}s; SIGKILL+restart"))
                self._force_restart()
                # After restart, prime the timestamp so we don't immediately
                # re-trigger on the next missed-cycle wraparound.
                self._last_pong_ts = time.time()

    def _force_restart(self) -> None:
        """SIGKILL the subprocess (no graceful shutdown) and respawn."""
        import signal
        with self._watchdog_lock:
            try:
                if self._proc and self._proc.poll() is None:
                    os.kill(self._proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                if self._proc:
                    self._proc.wait(timeout=2.0)
            except Exception:
                pass
            self._proc = None
            # Re-spawn without re-entering the watchdog plumbing.
            try:
                self._proc = subprocess.Popen(
                    [self.python, "-u", "-c", _WORKER_BODY],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=self.env,
                    cwd=str(self.cwd) if self.cwd else None,
                )
                ready = self._read_line(timeout=30)
                if not ready or ready.get("ready") is not True:
                    err = ready.get("error") if ready else "no response"
                    self._error_log.append(("watchdog_restart", err))
                    if self._proc:
                        try: self._proc.kill()
                        except Exception: pass
                    self._proc = None
            except Exception as exc:  # noqa: BLE001
                self._error_log.append(("watchdog_restart_spawn", str(exc)))
                self._proc = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def time(self, variant_so: Path) -> Optional[Dict[str, Any]]:
        """Time a variant .so. Returns the result dict or ``None`` on failure."""
        if not self.alive():
            self._error_log.append(("time", "worker not alive"))
            return None
        req = {
            "op":             "time",
            "so_path":        str(variant_so),
            "opt_class":      self.opt_class,
            "size":           self.size,
            "warmup":         self.warmup,
            "iters":          self.iters,
            "use_cuda_graph": self.use_cuda_graph,
        }
        with self._io_lock:
            try:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._error_log.append(("write", str(exc)))
                return None
            result = self._read_line(timeout=self.timeout)
        if result is None:
            return None
        if "error" in result:
            self._error_log.append(("time", result.get("error", "?")))
            return None
        # A successful timing call doubles as a liveness signal for the
        # watchdog — refresh the pong timestamp so we don't trip the grace
        # period during a long-running compile burst.
        self._last_pong_ts = time.time()
        return result

    def ping(self, *, _from_watchdog: bool = False) -> bool:
        """Send a no-op ping to the worker; return True on pong within 5s.

        Sharing the same stdin/stdout requires serialising against any
        concurrent ``time()`` request — both are short, so a single mutex
        suffices.
        """
        if not self.alive():
            return False
        with self._io_lock:
            try:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(json.dumps({"op": "ping"}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return False
            line = self._read_line(timeout=5)
        return bool(line and line.get("ok"))

    @property
    def error_log(self) -> list:
        return list(self._error_log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_line(self, timeout: float) -> Optional[Dict[str, Any]]:
        assert self._proc and self._proc.stdout
        import time as _time
        deadline = _time.monotonic() + timeout
        line = ""
        while _time.monotonic() < deadline:
            if self._proc.poll() is not None:
                tail = self._proc.stdout.read() or ""
                if tail:
                    self._error_log.append(("died", tail.strip()[-2000:]))
                return None
            ch = self._proc.stdout.read(1)
            if not ch:
                _time.sleep(0.01)
                continue
            if ch == "\n":
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    self._error_log.append(("decode", line[:500]))
                    return None
            line += ch
        self._error_log.append(("timeout", f"after {timeout}s"))
        return None


class MultiGPUTimingPool:
    """Fan a JIT autotune sweep across every visible GPU via work-stealing.

    When ``CUDA_VISIBLE_DEVICES`` (or ``HIP_VISIBLE_DEVICES`` for HIP)
    enumerates more than one device, this pool spawns one
    :class:`TimingWorker` per device with that device pinned via
    ``*_VISIBLE_DEVICES`` in the worker's env overlay.

    Dispatch model: each worker has a dedicated dispatcher thread that
    pulls items off a shared ``queue.Queue``. A fast worker that finishes
    early grabs the next pending item ahead of a slow sibling — no
    round-robin starvation. Items submitted to a dead worker are bounced
    back onto the queue for a live sibling to pick up.

    The public surface (``start`` / ``stop`` / ``alive`` / ``time(variant_so)``)
    matches :class:`TimingWorker` so callers can swap one for the other.
    Multiple producer threads may call ``time()`` concurrently.
    """

    _CUDA = "CUDA_VISIBLE_DEVICES"
    _HIP = "HIP_VISIBLE_DEVICES"

    @classmethod
    def env_var_for(cls, vendor: str) -> str:
        return cls._HIP if vendor == "hip" else cls._CUDA

    @classmethod
    def visible_devices(cls, vendor: str) -> List[str]:
        """Parse ``CUDA_/HIP_VISIBLE_DEVICES``. Falls back to ``["0"]``
        when the variable is unset or empty (single-GPU default)."""
        env_var = cls.env_var_for(vendor)
        raw = os.environ.get(env_var, "")
        devices = [d.strip() for d in raw.split(",") if d.strip()]
        return devices or ["0"]

    def __init__(self, opt_class: str, vendor: str = "cuda", **worker_kwargs):
        self.opt_class = opt_class
        self.vendor = vendor
        self.devices = self.visible_devices(vendor)
        self._env_var = self.env_var_for(vendor)
        self.workers: List[TimingWorker] = []
        for dev in self.devices:
            w = TimingWorker(
                opt_class,
                env_overlay={self._env_var: dev},
                **worker_kwargs,
            )
            self.workers.append(w)
        # Work-stealing dispatch plumbing — set up lazily in start() so
        # that workers that failed to spawn are dropped from the
        # dispatcher rotation before any threads are launched.
        self._queue: "queue.Queue" = queue.Queue()
        self._stopped = threading.Event()
        self._dispatch_threads: List[threading.Thread] = []
        self._started = False

    # ---- lifecycle mirroring TimingWorker ----------------------------

    def start(self) -> bool:
        """Start every per-device worker. Returns True iff at least one
        came up.

        Failed workers are dropped from the rotation but the pool is
        still considered usable as long as ≥1 worker is alive — matches
        the existing ``TimingWorker`` fallback semantics in
        ``_make_variant_timer``. After workers are up, spawn one
        dispatcher thread per live worker that pulls from the shared
        work queue (work-stealing).
        """
        any_ok = False
        live: List[TimingWorker] = []
        for w in self.workers:
            if w.start():
                live.append(w)
                any_ok = True
        self.workers = live
        self._spawn_dispatchers()
        self._started = True
        return any_ok

    def _spawn_dispatchers(self) -> None:
        """Start one dispatcher thread per live worker (idempotent)."""
        if self._dispatch_threads:
            return
        for idx, w in enumerate(self.workers):
            name = f"MultiGPUPool-dispatch-{idx}-{self.devices[idx] if idx < len(self.devices) else idx}"
            t = threading.Thread(
                target=self._dispatch_loop,
                args=(idx, w),
                daemon=True,
                name=name,
            )
            t.start()
            self._dispatch_threads.append(t)

    def alive(self) -> bool:
        return any(w.alive() for w in self.workers)

    def stop(self) -> None:
        # Signal dispatchers to drain & exit, then enqueue one sentinel
        # per dispatcher so each loop wakes up from its blocking ``get()``.
        self._stopped.set()
        for _ in self._dispatch_threads:
            try:
                self._queue.put(None)
            except Exception:
                pass
        for t in self._dispatch_threads:
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
        self._dispatch_threads = []
        for w in self.workers:
            try:
                w.stop()
            except Exception:
                pass

    def restart(self) -> bool:
        any_ok = False
        for w in self.workers:
            try:
                if w.restart():
                    any_ok = True
            except Exception:
                pass
        return any_ok

    # ---- work-stealing dispatcher -----------------------------------

    def _dispatch_loop(self, idx: int, worker: Any) -> None:
        """Per-worker thread: pull (payload, future) tuples off the
        shared queue, run them on ``worker``, fulfil the future.

        If the assigned worker is dead, re-enqueue the item so a live
        sibling can grab it, then briefly yield. A ``None`` sentinel
        exits the loop (drain-on-shutdown).
        """
        while not self._stopped.is_set():
            try:
                item = self._queue.get()
            except Exception:
                return
            if item is None:
                # Sentinel: graceful shutdown.
                try:
                    self._queue.task_done()
                except Exception:
                    pass
                return
            payload, fut = item
            try:
                if not worker.alive():
                    # Bounce the item back so a sibling can pick it up.
                    # If we're the last one standing, fail the future
                    # rather than busy-looping forever.
                    live_siblings = sum(
                        1 for w in self.workers
                        if w is not worker and w.alive()
                    )
                    if live_siblings == 0:
                        fut.set_exception(
                            RuntimeError(
                                "MultiGPUTimingPool: no live workers"))
                    else:
                        self._queue.put(item)
                        time.sleep(0.05)
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass
                    continue
                variant_so, opt_class, kwargs = payload
                # Real TimingWorker.time only takes variant_so (opt_class
                # was bound at construction). Pass extras only when the
                # caller supplied something, to keep both production and
                # mock-worker test paths working.
                if not opt_class and not kwargs:
                    result = worker.time(variant_so)
                else:
                    result = worker.time(variant_so, opt_class, **kwargs)
                fut.set_result(result)
            except Exception as exc:  # noqa: BLE001 — propagate to caller
                try:
                    fut.set_exception(exc)
                except Exception:
                    pass
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    # ---- timing API mirroring TimingWorker ---------------------------

    def time(self, variant_so: Path, opt_class: str = "", **kwargs):
        """Enqueue a timing request and block until any dispatcher
        returns a result. Thread-safe: multiple producer threads may
        call ``time()`` concurrently — the queue serialises submissions
        and the dispatchers steal work as they become idle.

        Returns the dict from the underlying worker, ``None`` if the
        worker reported a failure, or raises if no live worker remains.
        """
        if not self.workers:
            return None
        # Lazy dispatcher spawn — supports callers that construct the
        # pool and call .time() without an explicit .start() (e.g. tests
        # that splice in mock workers via __new__).
        if not self._dispatch_threads:
            self._spawn_dispatchers()
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        self._queue.put(((variant_so, opt_class, kwargs), fut))
        return fut.result()

    @property
    def error_log(self) -> list:
        merged: list = []
        for i, w in enumerate(self.workers):
            for entry in w.error_log:
                merged.append((f"gpu{i}",) + tuple(entry))
        return merged

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


# ===========================================================================
# bayesian — Optuna TPE-driven autotune (absorbed from
# grokking_optimizers/bayesian.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Stream 5 — Bayesian early-stopping (BayesianEarlyStopper, elbow detector)
# ---------------------------------------------------------------------------
#
# Replaces hardcoded ``bayesian_trials=500`` / ``top_k=20`` knobs with a
# multi-criterion auto-stop loop. The class exposes ``observe(value, params)``
# (called per completed trial) and ``should_stop()`` (polled before asking
# Optuna for the next trial). Triggers, evaluated in order:
#
#   (a) best-so-far plateau   — no relative improvement > ``min_delta_rel``
#                               for ``patience`` consecutive trials.
#   (b) EI exhaustion         — rolling Expected-Improvement estimate falls
#                               below ``ei_floor`` (currently a no-op slot —
#                               TPE doesn't expose EI directly; reserved for
#                               a Stream-future GP sampler).
#   (c) coverage saturation   — count of distinct (dim_name, value) pairs
#                               grows by less than ``coverage_growth_floor``
#                               per trial over the trailing ``patience``
#                               window — i.e. TPE is just resampling the
#                               same handful of corners.
#   (d) wall-clock budget     — ``max_seconds`` from stopper construction.
#   (e) hard ceiling          — ``max_trials`` (sanity guard, default 1M).
#
# Auto patience: ``max(50, trial_count // 10)`` so the patience window
# scales with how much exploration has already happened.

def _hashable(v):
    """Make a search-space value hashable for the coverage set."""
    if isinstance(v, list):
        return tuple(v)
    return v


class BayesianEarlyStopper:
    """Multi-criterion stopper for Optuna autotune.

    Stops when ANY of the following triggers (whichever fires first):
      (a) Best-so-far plateau: no improvement > ``min_delta_rel`` for
          ``patience`` trials.
      (b) EI exhaustion: rolling EI estimate falls below ``ei_floor``
          (reserved; TPE samplers don't expose EI here).
      (c) Coverage saturation: new-(dim_name, value) tuples per trial
          < ``coverage_growth_floor``.
      (d) Wall-clock budget: ``time.time() - start_time > max_seconds``.
      (e) Hard ceiling: ``trial_count >= max_trials`` (sanity, default 1M).
    """

    def __init__(self, *,
                 min_delta_rel: float = 0.005,
                 patience: Optional[int] = None,
                 ei_floor: float = 1e-6,
                 coverage_growth_floor: float = 0.001,
                 max_seconds: Optional[float] = None,
                 max_trials: int = 1_000_000):
        self.min_delta_rel = min_delta_rel
        self._patience_override = patience  # None → auto = max(50, 0.1*N)
        self.ei_floor = ei_floor
        self.coverage_growth_floor = coverage_growth_floor
        self.max_seconds = max_seconds
        self.max_trials = max_trials
        self.start_time = time.time()
        self.best = math.inf
        self.last_improve_trial = 0
        self.coverage_set: set = set()  # (dim_name, value) tuples
        self.coverage_history: List[int] = []  # cumulative |coverage_set| per trial
        self.trial_count = 0
        self.stop_reason: Optional[str] = None

    @property
    def patience(self) -> int:
        if self._patience_override is not None:
            return self._patience_override
        return max(50, self.trial_count // 10)

    def observe(self, trial_value: float, trial_params: Dict[str, Any]) -> None:
        self.trial_count += 1
        # Plateau tracking — only count finite improvements as such.
        if math.isfinite(trial_value) and \
                trial_value < self.best * (1 - self.min_delta_rel):
            self.best = trial_value
            self.last_improve_trial = self.trial_count
        # Coverage tracking
        for k, v in trial_params.items():
            self.coverage_set.add((k, _hashable(v)))
        self.coverage_history.append(len(self.coverage_set))

    def should_stop(self) -> bool:
        if self.stop_reason is not None:
            return True
        # (e) hard ceiling
        if self.trial_count >= self.max_trials:
            self.stop_reason = f"hard_ceiling:{self.max_trials}"
            return True
        # (d) wall-clock
        if self.max_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_seconds:
                self.stop_reason = f"max_seconds:{int(elapsed)}s"
                return True
        # Need a warm-up of at least patience+10 trials before other criteria fire
        if self.trial_count < self.patience + 10:
            return False
        # (a) best-so-far plateau
        if (self.trial_count - self.last_improve_trial) >= self.patience:
            self.stop_reason = f"plateau:no_improvement_in_{self.patience}"
            return True
        # (c) coverage saturation: growth over last `patience` trials
        if len(self.coverage_history) >= self.patience:
            window_growth = (self.coverage_history[-1]
                             - self.coverage_history[-self.patience])
            growth_rate = window_growth / max(1, self.patience)
            if growth_rate < self.coverage_growth_floor:
                self.stop_reason = (
                    f"coverage_saturated:growth_rate={growth_rate:.4f}")
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stopper state for cache persistence."""
        return {
            "min_delta_rel": self.min_delta_rel,
            "patience_override": self._patience_override,
            "ei_floor": self.ei_floor,
            "coverage_growth_floor": self.coverage_growth_floor,
            "max_seconds": self.max_seconds,
            "max_trials": self.max_trials,
            "trial_count": self.trial_count,
            "best": self.best,
            "last_improve_trial": self.last_improve_trial,
            "coverage_size": len(self.coverage_set),
            "stop_reason": self.stop_reason,
        }


def _detect_topk_elbow(records: List[Dict[str, Any]]) -> int:
    """Detect the elbow in a sorted-by-timing trial list.

    Sort successful records by ``timing_ms`` (ascending), compute the
    second discrete differences over the timing series, return the index
    where the curvature peaks (i.e. the knee where good trials end and
    long-tail trials begin). Capped at 50 and at len(records).

    Fallback: ``min(50, len(records) // 4)`` when there's too little
    data to fit a curvature estimate or all timings are degenerate.
    """
    ok = [r for r in records if (r.get("status") == "ok"
                                  or r.get("timing_ms") is not None)]
    n = len(ok)
    if n == 0:
        return 0
    if n < 5:
        return min(50, max(1, n))
    # Extract timings (records may use 'timing_ms' or 'value_ms')
    def _val(r):
        v = r.get("timing_ms")
        if v is None:
            v = r.get("value_ms")
        return float(v) if v is not None else math.inf
    times = sorted(_val(r) for r in ok)
    times = [t for t in times if math.isfinite(t)]
    if len(times) < 5:
        return min(50, max(1, len(times) or 1))
    # Second discrete differences: t[i-1] - 2*t[i] + t[i+1]
    d2 = [times[i - 1] - 2.0 * times[i] + times[i + 1]
          for i in range(1, len(times) - 1)]
    # Peak negative curvature would be a max of |d2|; we want the first
    # index where d2 spikes substantially. Take argmax of -d2 (since
    # the knee is where the second derivative becomes large and negative
    # in a convex-then-flat curve).
    if not d2 or all(abs(x) < 1e-12 for x in d2):
        return min(50, max(1, len(times) // 4))
    # The knee index in the original 'times' array is d2_idx + 1
    knee = max(range(len(d2)), key=lambda i: -d2[i]) + 1
    # Ensure at least a few seeds even if curvature is shallow
    knee = max(knee, min(3, len(times)))
    return min(50, knee)


def _make_pruner(name: str):
    """Return an Optuna pruner. ``name`` in {none, median, hyperband}."""
    name = (name or "none").lower()
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=2)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=21, reduction_factor=3)
    raise ValueError(f"unknown pruner {name!r}")


TimerResult = Optional[Dict[str, Any]]
Timer = Callable[[Dict[str, Any]], TimerResult]
ProgressCb = Optional[Callable[[int, int, Dict[str, Any]], None]]


def _suggest(trial: optuna.Trial, dims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate dim specs into Optuna ``suggest_*`` calls."""
    cfg: Dict[str, Any] = {}
    for dim in dims:
        name = dim["name"]
        values = dim["values"]
        suggest_vals = [tuple(v) if isinstance(v, list) else v for v in values]
        cfg[name] = trial.suggest_categorical(name, suggest_vals)
    return cfg


def _make_trial_record(stage: str, trial_num: int, cfg: Dict[str, Any],
                       result: TimerResult, host: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    # Stream 10: pull the per-variant numerical-validation tag set by
    # _make_variant_timer (keyed by config_key). Trials that never went
    # through the variant timer (e.g. synthetic self-test trials, or
    # infeasible / build-fail trials) get "skipped" by default.
    ckey = config_key(cfg)
    num_status = _LAST_NUMERICAL_STATUS.get(ckey, "skipped")
    return {
        "trial_num":   trial_num,
        "stage":       stage,
        "config":      {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
        "config_key":  ckey,
        "timing_ms":   result["timing_ms"] if result else None,
        "min_ms":      result["min_ms"]    if result else None,
        "max_ms":      result["max_ms"]    if result else None,
        "n":           result["n"]         if result else None,
        "host":        host,
        "numerical_status": num_status,
        "recorded_at": datetime.datetime.now().isoformat(),
    }


def _coerce_timer_result(raw: Any) -> Tuple[Optional[Dict[str, Any]], float]:
    """Normalise a timer return value.

    The C++/CUDA variant timer returns ``{"timing_ms": float, ...}`` (or
    ``None`` on failure). Synthetic test timers may return a bare float.
    Returns ``(result_dict_or_None, scalar_value_for_optuna)``.
    """
    if raw is None:
        return None, math.inf
    if isinstance(raw, dict):
        v = raw.get("timing_ms")
        return raw, float(v) if v is not None else math.inf
    if isinstance(raw, (int, float)):
        v = float(raw)
        return ({"timing_ms": v, "min_ms": v, "max_ms": v, "n": 1},
                v if math.isfinite(v) else math.inf)
    raise TypeError(f"timer returned unsupported type {type(raw).__name__}")


def run_bayesian(
    arch: str,
    space: Dict[str, Any],
    *,
    n_trials: Optional[int] = None,
    seed: int = 0,
    storage: Optional[Path] = None,
    study_name: str = "sg_tune",
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
    pruner: str = "none",
    seed_trials: Optional[List[Dict[str, Any]]] = None,
    stopper: Optional["BayesianEarlyStopper"] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the TPE stage with multi-criterion early stopping.

    Returns ``(records, stop_info)`` where ``stop_info`` is the
    ``BayesianEarlyStopper.to_dict()`` snapshot capturing why the loop
    halted (plateau / coverage saturated / wall-clock budget / manual
    ``n_trials`` cap).

    ``n_trials`` is now optional: when ``None`` (default), the loop
    runs until the stopper fires. When set, it acts as a hard cap on
    top of the stopper's other criteria — useful for explicit
    reproducible budgets (and for the ``--quick`` and self-test paths).
    """
    dims = space[arch]["dims"]
    # Build a per-config feasibility predicate from the prefilter rules
    # so we can validate TPE suggestions in O(rules) without enumerating
    # the (billions-of-configs) full Cartesian survivor set.
    is_feasible = compile_feasibility_check(
        space[arch].get("prefilter", {}))

    if stopper is None:
        stopper = BayesianEarlyStopper()

    # Startup trials: enough to cold-start TPE. When n_trials is set
    # use the historical 10% heuristic; otherwise lean on the stopper's
    # min-warmup (patience + 10) for the same effect.
    if n_trials is not None:
        startup = max(10, n_trials // 10)
    else:
        startup = max(10, (stopper.patience + 10) // 5)
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=startup,
        multivariate=True,
        constant_liar=False,
    )

    storage_url = None
    if storage is not None:
        storage = Path(storage)
        storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{storage}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=_make_pruner(pruner),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    if seed_trials:
        dim_names = [d["name"] for d in dims]
        dim_value_sets = {d["name"]: [tuple(v) if isinstance(v, list) else v
                                      for v in d["values"]]
                          for d in dims}
        for t in seed_trials:
            cfg = t.get("config") or {}
            tms = t.get("timing_ms")
            if tms is None:
                continue
            params: Dict[str, Any] = {}
            distributions: Dict[str, optuna.distributions.BaseDistribution] = {}
            ok = True
            for name in dim_names:
                if name not in cfg:
                    ok = False
                    break
                val = cfg[name]
                val = tuple(val) if isinstance(val, list) else val
                if val not in dim_value_sets[name]:
                    ok = False
                    break
                params[name] = val
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    dim_value_sets[name])
            if not ok:
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(tms),
                ))
            except Exception:
                continue

    records: List[Dict[str, Any]] = []
    # For progress reporting when no manual cap, use stopper's hard ceiling
    # as the denominator — it'll never be reached, but the bar shows movement.
    total_for_progress = n_trials if n_trials is not None else stopper.max_trials

    while True:
        if n_trials is not None and len(records) >= n_trials:
            if stopper.stop_reason is None:
                stopper.stop_reason = f"manual_n_trials:{n_trials}"
            break
        if stopper.should_stop():
            break
        trial = study.ask()
        cfg = _suggest(trial, dims)
        if not is_feasible(cfg):
            study.tell(trial, math.inf,
                       state=optuna.trial.TrialState.PRUNED)
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "infeasible"
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg)
            continue
        try:
            raw = timer(cfg)
        except Exception as exc:
            study.tell(trial, math.inf,
                       state=optuna.trial.TrialState.FAIL)
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "fail"
            rec["error"] = str(exc)
            records.append(rec)
            if progress:
                progress(len(records), total_for_progress, cfg)
            stopper.observe(math.inf, cfg)
            continue
        result, value = _coerce_timer_result(raw)
        if not math.isfinite(value):
            study.tell(trial, math.inf,
                       state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, value)
        rec = _make_trial_record("tpe", trial.number, cfg, result, host=host)
        rec["status"] = "ok" if result is not None else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(len(records), total_for_progress, cfg)
        stopper.observe(value, cfg)

    return records, stopper.to_dict()


def _step_neighbours(value: Any, values: List[Any], radius: int = 2
                     ) -> List[Any]:
    """Return up to ``2*radius`` neighbours of ``value`` along its dim's
    ordered ``values`` list (radius-2 = +/-2 steps)."""
    normed = [tuple(v) if isinstance(v, list) else v for v in values]
    val = tuple(value) if isinstance(value, list) else value
    if val not in normed:
        return []
    idx = normed.index(val)
    lo = max(0, idx - radius)
    hi = min(len(normed), idx + radius + 1)
    return [normed[i] for i in range(lo, hi) if normed[i] != val]


def topk_refine(
    bayes_trials: List[Dict[str, Any]],
    space: Dict[str, Any],
    arch: str,
    *,
    top_k: Optional[int] = None,
    radius: int = 2,
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """For each of the top-K TPE trials, time the +/-radius-step
    neighbours along every dim. Returns the refine-stage records.

    When ``top_k`` is ``None`` the seed count is chosen automatically
    via ``_detect_topk_elbow`` over the sorted success list — small for
    sharply-peaked spaces, larger when the curve flattens late."""
    dims = space[arch]["dims"]
    is_feasible = compile_feasibility_check(
        space[arch].get("prefilter", {}))

    successes = [t for t in bayes_trials if t.get("timing_ms") is not None]
    successes.sort(key=lambda t: t["timing_ms"])
    if top_k is None:
        top_k = _detect_topk_elbow(successes) or min(50, max(1, len(successes) // 4 or 1))
    seeds = successes[:top_k]

    seen_keys: set = {t["config_key"] for t in bayes_trials}
    candidate_cfgs: List[Dict[str, Any]] = []

    for seed_trial in seeds:
        base = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in seed_trial["config"].items()}
        for dim in dims:
            name = dim["name"]
            for nb in _step_neighbours(base.get(name), dim["values"], radius):
                cfg = dict(base)
                cfg[name] = nb
                k = config_key(cfg)
                if k in seen_keys:
                    continue
                if not is_feasible(cfg):
                    continue
                seen_keys.add(k)
                candidate_cfgs.append(cfg)

    records: List[Dict[str, Any]] = []
    total = len(candidate_cfgs)
    for i, cfg in enumerate(candidate_cfgs, 1):
        raw = timer(cfg)
        try:
            result, _ = _coerce_timer_result(raw)
        except TypeError:
            result = raw if isinstance(raw, dict) else None
        rec = _make_trial_record("refine", i, cfg, result, host=host)
        rec["status"] = "ok" if result else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(i, total, cfg)
    return records


def pick_winner(all_trials: List[Dict[str, Any]], *,
                strict_numerics: bool = False) -> Optional[Dict[str, Any]]:
    """Lowest timing across all stages, after numerical-validation filtering.

    Stream 10 filter rules:
      * Always exclude trials whose ``numerical_status`` is
        ``"numerical_fail"`` — that variant produced an output outside
        tolerance vs. the reference and is unsafe to ship as the winner.
      * When ``strict_numerics=True``, only trials tagged
        ``"deterministic"`` are eligible (i.e. bit-identical to the
        reference AND bit-identical across a 3x re-run).

    Returns the winning trial record (with ``config`` and ``timing_ms``)
    or ``None`` if no trial is eligible.
    """
    finished = [t for t in all_trials if t.get("timing_ms") is not None]
    # Trials produced before Stream 10 lack the numerical_status field;
    # treat them as "skipped" so the existing winner-selection logic
    # behaves identically for legacy caches.
    eligible = [t for t in finished
                if t.get("numerical_status", "skipped") != "numerical_fail"]
    if strict_numerics:
        eligible = [t for t in eligible
                    if t.get("numerical_status") == "deterministic"]
    if not eligible:
        return None
    return min(eligible, key=lambda t: t["timing_ms"])


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


class _DebugTee:
    """File-like wrapper that mirrors every write to a secondary stream.

    Used when --debug is enabled to stream the build report to stderr in
    real time without changing the existing report.write(...) call sites.
    """
    __slots__ = ("primary", "mirror")

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror

    def write(self, s):
        self.primary.write(s)
        self.primary.flush()
        try:
            self.mirror.write(s)
            self.mirror.flush()
        except Exception:
            pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        self.primary.flush()
        try:
            self.mirror.flush()
        except Exception:
            pass


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
                  search_space_hash: Optional[str] = None,
                  early_stop_info: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            e = self.get(opt, model, arch)
            e["tuned_config"]      = config
            e["jit_completed_at"]  = datetime.datetime.now().isoformat()
            e["jit_host"]          = _current_host()
            if mode is not None:
                e["mode"] = mode
            if search_space_hash is not None:
                e["search_space_hash"] = search_space_hash
            # Stream-5 additive field: stopper diagnostics for this run.
            # Backward-compatible — readers that don't know about it just
            # ignore the extra key.
            if early_stop_info is not None:
                e["early_stop_info"] = early_stop_info
            self._dirty = True

    # ── Garbage collection ────────────────────────────────────────────

    def prune(self, *, max_age_days: int = 30, keep_top_n: int = 100,
              dry_run: bool = False) -> Dict[str, Any]:
        """Drop variant_artifacts older than ``max_age_days`` AND those not
        in the top-N timings per ``(opt, model, arch)`` group.

        Returns a summary dict ``{dropped, kept, bytes_freed,
        max_age_days, keep_top_n, dry_run, entries_scanned}``.

        Timing rank is taken from each entry's ``sweep_history`` +
        ``bayesian_trials`` (matched by ``config_key``); variants with
        no recorded timing are treated as ``inf`` and pruned first.
        Pruning is conservative: the tuned winner (``tuned_config``) is
        always kept, even if it falls outside ``keep_top_n`` or is older
        than ``max_age_days``.
        """
        now = time.time()
        cutoff_ts = now - max_age_days * 86400
        dropped = 0
        kept = 0
        bytes_freed = 0
        entries_scanned = 0

        with self._lock:
            entries = self._data.get("entries", {})
            for entry_key, entry in entries.items():
                # Cache key format: "<opt>/<model>/<arch>". Skip anything
                # that doesn't match — defensive against bad data.
                parts = entry_key.split("/")
                if len(parts) != 3:
                    continue
                entries_scanned += 1
                variants = entry.get("variant_artifacts") or {}
                if not isinstance(variants, dict) or not variants:
                    continue

                # Per-ckey best timing from trial history. We scan both
                # buckets so a v2-style cache (sweep_history only) still
                # gets ranked correctly.
                best_ms: Dict[str, float] = {}
                for src in ("sweep_history", "bayesian_trials"):
                    for t in entry.get(src) or []:
                        if not isinstance(t, dict):
                            continue
                        ck = t.get("config_key")
                        tms = t.get("timing_ms")
                        if ck is None or tms is None:
                            continue
                        prior = best_ms.get(ck, math.inf)
                        if tms < prior:
                            best_ms[ck] = tms

                # Always preserve the tuned winner's ckey if present.
                tuned = entry.get("tuned_config") or {}
                tuned_ckey = tuned.get("config_key") if isinstance(tuned, dict) else None

                # Rank ckeys by best timing (ascending); unscored ckeys
                # go to the tail with inf so they're prunable first.
                ranked = sorted(
                    variants.keys(),
                    key=lambda ck: best_ms.get(ck, math.inf),
                )
                top_keep = set(ranked[:max(0, int(keep_top_n))])
                if tuned_ckey:
                    top_keep.add(tuned_ckey)

                # Pass 1: classify + delete .so files.
                to_drop_keys: List[str] = []
                for ckey, vrec in list(variants.items()):
                    if not isinstance(vrec, dict):
                        # Garbage — drop on sight.
                        to_drop_keys.append(ckey)
                        continue
                    mtime = vrec.get("mtime")
                    if mtime is None:
                        # Treat unknown mtime as "old" so we don't keep
                        # ancient garbage forever, but spare the tuned
                        # winner.
                        age_ok = (ckey == tuned_ckey)
                    else:
                        age_ok = mtime >= cutoff_ts or (ckey == tuned_ckey)
                    rank_ok = ckey in top_keep

                    if age_ok and rank_ok:
                        kept += 1
                        continue

                    # Drop: try to unlink the .so, then forget the record.
                    so_path = vrec.get("path") or vrec.get("so_path")
                    if so_path:
                        p = Path(so_path)
                        try:
                            if p.exists():
                                sz = p.stat().st_size
                                bytes_freed += sz
                                if not dry_run:
                                    try:
                                        p.unlink()
                                    except OSError:
                                        pass
                        except OSError:
                            pass
                    to_drop_keys.append(ckey)
                    dropped += 1

                # Pass 2: actually remove the records from the entry.
                if not dry_run and to_drop_keys:
                    for ckey in to_drop_keys:
                        variants.pop(ckey, None)
                    self._dirty = True

            if not dry_run and self._dirty:
                self.save()

        return {
            "dropped":         dropped,
            "kept":            kept,
            "bytes_freed":     bytes_freed,
            "max_age_days":    max_age_days,
            "keep_top_n":      keep_top_n,
            "dry_run":         bool(dry_run),
            "entries_scanned": entries_scanned,
        }


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
    # Bayesian autotune budget.
    # bayesian_trials=None ⇒ Stream-5 multi-criterion auto early-stop.
    # An integer here acts as a hard cap on top of the stopper.
    bayesian_trials: Optional[int] = None
    top_k: Optional[int] = None       # None ⇒ elbow detection in topk_refine
    # Stream-5 stopper knobs (only consulted when bayesian_trials is None).
    max_tune_seconds: Optional[float] = None
    min_improvement: float = 0.005
    patience: Optional[int] = None    # None ⇒ auto = max(50, 0.1*N)
    seed: int = 0
    debug_symbols: bool = False
    debug: bool = False               # mirror report to stderr + print every subproc
    # §12 A1 / A2 — Hyperband pruner + transfer learning
    pruner: str = "none"              # "none" | "median" | "hyperband"
    transfer_learning: bool = False
    # Stream 7 — NVRTC / hipRTC runtime kernel specialization.
    # When True, build() calls kernel_registry.initialize_registry() at the
    # tail of the orchestrator to pre-warm a per-arch KernelRegistry.
    enable_runtime_specialization: bool = False
    # Stream 6 — Jinja2 kernel emitter. When True, _variant_macros routes
    # through grokking_optimizers.codegen.emit_variant_source so each
    # variant is built from a freshly rendered source file instead of
    # one fixed source re-compiled with different -D macros.
    enable_emitter: bool = False
    _emitted_sources: Dict[str, Path] = field(default_factory=dict)
    # Stream 8 — device-side PGO (CUPTI / rocprof / XLA HLO dump)
    enable_device_pgo: bool = False
    # Stream 9 — variant cache GC. Auto-prune runs at the END of a
    # successful JIT autotune pass; controls map 1:1 to CompileCache.prune().
    prune_after_autotune: bool = True
    prune_max_age_days: int = 30
    prune_keep_top_n: int = 100
    # Stream 10 — per-variant numerical / differential validation
    strict_numerics: bool = False     # require bit-identical determinism for winner
    aot_so_path: Optional[Path] = None  # populated by build_aot; used by variant timer


def _ensure_nvcc_on_path() -> Optional[str]:
    """Locate nvcc and prepend its directory to PATH if it's missing.

    Searches in priority order:
      1. PATH (already there)
      2. $CUDA_HOME/bin/nvcc, $CUDA_PATH/bin/nvcc
      3. Standard system locations: /usr/local/cuda/bin, /opt/cuda/bin
      4. Versioned dirs: /usr/local/cuda-<ver>/bin (sorted newest first)
      5. **NVIDIA PyPI wheels**: <site-packages>/nvidia/cuda_nvcc/bin/nvcc
         (this is the only nvcc available on Colab CPU runtimes after
         ``pip install nvidia-cuda-nvcc-cu12``)

    When found, prepends the bin dir to PATH and AUTO-CORRECTS
    ``CUDA_HOME`` if it points at a non-existent path. (Common case:
    user manually set ``os.environ["CUDA_HOME"] = "/usr/local/cuda"``
    but apt installed nvcc to ``/usr/bin/nvcc`` — without this fix,
    torch's cpp_extension writes ninja with the stale phantom path.)
    Returns the resolved nvcc path or None.
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        _reconcile_cuda_home(nvcc)
        return nvcc
    candidates: List[Path] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    candidates.append(Path("/usr/local/nvidia/cuda/bin/nvcc"))
    candidates.append(Path("/opt/cuda/bin/nvcc"))
    for parent in (Path("/usr/local"), Path("/opt")):
        if parent.is_dir():
            try:
                for child in sorted(parent.iterdir(), reverse=True):
                    if child.name.startswith("cuda-") or child.name.startswith("cuda_"):
                        candidates.append(child / "bin" / "nvcc")
            except (OSError, PermissionError):
                pass
    # NVIDIA PyPI wheels — the rescue path for Colab CPU runtimes.
    try:
        import site
        site_dirs: List[Path] = []
        for getter in (site.getsitepackages, site.getusersitepackages):
            try:
                got = getter()
                if isinstance(got, str):
                    site_dirs.append(Path(got))
                else:
                    site_dirs.extend(Path(p) for p in got)
            except Exception:
                pass
        for sd in site_dirs:
            for cuda_pkg in ("cuda_nvcc", "cuda-nvcc"):
                nvcc_p = sd / "nvidia" / cuda_pkg / "bin" / "nvcc"
                candidates.append(nvcc_p)
    except Exception:
        pass

    for nvcc_p in candidates:
        try:
            if nvcc_p.is_file() and os.access(nvcc_p, os.X_OK):
                nvcc_dir = str(nvcc_p.parent)
                current_path = os.environ.get("PATH", "")
                if nvcc_dir not in current_path.split(os.pathsep):
                    os.environ["PATH"] = f"{nvcc_dir}{os.pathsep}{current_path}"
                # For PyPI wheels, libcudart lives in a sibling nvidia/cuda_runtime
                # directory; make sure LD_LIBRARY_PATH and CUDA_HOME's lib64
                # discovery work.
                if "nvidia" in nvcc_p.parts:
                    nvidia_root = nvcc_p.parents[2]  # <site>/nvidia
                    lib_dirs = []
                    for pkg in ("cuda_runtime", "cuda_cudart", "cuda_nvrtc",
                                "cuda_cccl"):
                        ld = nvidia_root / pkg / "lib"
                        if ld.is_dir():
                            lib_dirs.append(str(ld))
                    if lib_dirs:
                        existing = os.environ.get("LD_LIBRARY_PATH", "")
                        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                            lib_dirs + ([existing] if existing else []))
                sys.stderr.write(f"[compile] discovered nvcc at {nvcc_p}\n")
                _reconcile_cuda_home(str(nvcc_p))
                return str(nvcc_p)
        except (OSError, PermissionError):
            continue
    return None


def _reconcile_cuda_home(nvcc_path: str) -> None:
    """Make ``os.environ["CUDA_HOME"]`` consistent with the actual nvcc
    location. Sets/overwrites CUDA_HOME when it points at a path that
    doesn't contain the real nvcc binary, then forces torch's
    cpp_extension to re-read it.

    Example fix: user sets ``CUDA_HOME=/usr/local/cuda`` (which doesn't
    exist) but apt installs nvcc to ``/usr/bin/nvcc``. Without this,
    torch joins the stale CUDA_HOME with ``/bin/nvcc`` and writes a
    bad path into build.ninja → exit-127.
    """
    nvcc = Path(nvcc_path).resolve()
    if not nvcc.is_file():
        return
    # Derive the "CUDA root" — parent of bin/. For NVIDIA's standard
    # layout (/usr/local/cuda-12.6/bin/nvcc), root = /usr/local/cuda-12.6.
    # For Debian's multiarch layout (/usr/bin/nvcc), root = /usr.
    actual_root = nvcc.parent.parent
    env_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    needs_fix = (
        not env_root
        or not Path(env_root).is_dir()
        or not (Path(env_root) / "bin" / "nvcc").is_file()
    )
    if needs_fix:
        sys.stderr.write(
            f"[compile] correcting CUDA_HOME: {env_root or '<unset>'} -> "
            f"{actual_root} (derived from {nvcc})\n"
        )
        os.environ["CUDA_HOME"] = str(actual_root)
        # Make sure $CUDA_HOME/bin is also on PATH so subprocess shells
        # invoking nvcc by its bin dir work.
        bin_dir = str(actual_root / "bin")
        current = os.environ.get("PATH", "")
        if bin_dir not in current.split(os.pathsep):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current}"
        _refresh_torch_cuda_home(force=True)


def _refresh_torch_cuda_home(force: bool = False) -> None:
    """Force ``torch.utils.cpp_extension`` to re-read CUDA_HOME / ROCM_HOME.

    torch caches these at module-import time. If a user sets
    ``os.environ["CUDA_HOME"]`` AFTER torch was imported (common in
    Colab/Jupyter where torch is pre-loaded at kernel startup), the
    cached value stays at ``None`` and the build fails with
    "CUDA_HOME environment variable is not set" even though the env
    var is present. This patches the cached values from os.environ.

    With ``force=True``, overwrites the cache even when it already has
    a non-None value (used after ``_reconcile_cuda_home`` corrected a
    stale CUDA_HOME).
    """
    try:
        import torch.utils.cpp_extension as cppext
    except Exception:
        return
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda:
        cached = getattr(cppext, "CUDA_HOME", None)
        if force or cached in (None, "") or cached != cuda:
            cppext.CUDA_HOME = cuda
            sys.stderr.write(f"[compile] refreshed torch CUDA_HOME = {cuda}\n")
    rocm = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    if rocm:
        cached = getattr(cppext, "ROCM_HOME", None)
        if force or cached in (None, "") or cached != rocm:
            cppext.ROCM_HOME = rocm
            sys.stderr.write(f"[compile] refreshed torch ROCM_HOME = {rocm}\n")


def _sudo_prefix() -> List[str]:
    """Return ['sudo'] when needed, [] when running as root, [] when
    sudo doesn't exist (we'll try the bare command and let it fail)."""
    try:
        if os.geteuid() == 0:
            return []
    except AttributeError:
        pass  # Windows
    s = shutil.which("sudo")
    return [s] if s else []


# ---------------------------------------------------------------------------
# Per-arch toolchain-version targeting — §12 Stream 12
# ---------------------------------------------------------------------------
#
# Bootstrapping must pick a CUDA/ROCm version that's *at least* high enough
# for the target arch (sm_90 needs CUDA 12.0+, sm_120a needs 12.8+, gfx950
# needs ROCm 6.2+, gfx1200/1201 need 7.0+). The helpers below answer
# "what's the minimum version we should install?" by reading
# ARCH_TABLE[arch].min_toolchain_version and (for CUDA) reconciling with
# torch.version.cuda so we don't downgrade below torch's bundled runtime.

def _target_cuda_version_for_arch(arch: str) -> Tuple[int, int]:
    """Return min CUDA (major, minor) needed for this arch. Falls back to
    torch.version.cuda if higher.

    Returns (0, 0) for non-CUDA archs (caller must check).
    """
    if arch not in ARCH_TABLE:
        return (12, 0)
    entry = ARCH_TABLE[arch]
    if entry.vendor != "cuda":
        return (0, 0)
    min_v = entry.min_toolchain_version  # e.g. (12, 0) for sm_90a
    # Compare to torch's bundled CUDA so we don't pick something older
    # than the runtime torch was linked against.
    try:
        import torch
        tv = getattr(torch.version, "cuda", None)
        if tv:
            parts = tv.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                torch_v: Tuple[int, ...] = (int(parts[0]), int(parts[1]))
            else:
                torch_v = min_v
        else:
            torch_v = min_v
    except Exception:
        torch_v = min_v
    # Pick the HIGHER of the two (typed as 2-tuple for the return contract).
    chosen = max(min_v[:2], torch_v[:2])
    return (chosen[0], chosen[1])


def _target_rocm_version_for_arch(arch: str) -> Tuple[int, int]:
    """Return min ROCm (major, minor) needed for this arch.

    Returns (0, 0) for non-HIP archs (caller must check).
    """
    if arch not in ARCH_TABLE or ARCH_TABLE[arch].vendor != "hip":
        return (0, 0)
    v = ARCH_TABLE[arch].min_toolchain_version
    return (v[0], v[1])


def _bootstrap_cuda_via_conda(stream) -> bool:
    """Install via conda from the nvidia channel. Works on Linux, macOS,
    Windows — no sudo. Best when CONDA_PREFIX is set (active env)."""
    conda = shutil.which("mamba") or shutil.which("conda")
    if not conda:
        return False
    in_env = bool(os.environ.get("CONDA_PREFIX"))
    stream.write(f"[bootstrap] trying {Path(conda).name} install "
                 f"(in conda env: {in_env}) — cross-platform, no sudo\n")
    stream.flush()
    rc = subprocess.call([
        conda, "install", "-y", "-c", "nvidia",
        "cuda-nvcc", "cuda-runtime", "cuda-cccl", "cuda-nvrtc",
    ])
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_nvidia_apt_repo(stream, arch: Optional[str] = None) -> bool:
    """Add NVIDIA's official CUDA apt repo and install cuda-toolkit-XX-Y.

    Preferred over stock ``nvidia-cuda-toolkit`` on Debian/Ubuntu because:
      - Stock package is often years old (CUDA 11.5 on Ubuntu 22.04 —
        too old for sm_90, which needs CUDA 12.0+).
      - NVIDIA's repo installs to ``/usr/local/cuda-<ver>/`` with a
        ``/usr/local/cuda`` symlink — the exact layout torch's
        ``cpp_extension`` expects (bin/, lib64/, include/).
      - Version is picked to be max(arch_min, torch.version.cuda) — never
        below the arch requirement (sm_120a → 12.8, sm_103a → 12.9), never
        below torch's bundled CUDA.

    Probes the available cuda-toolkit packages and installs the newest
    one that matches the per-arch target. Falls back through known good
    versions (12.6, 12.4, 12.3 …) if no exact match exists in the repo
    metadata.
    """
    apt = shutil.which("apt-get") or shutil.which("apt")
    if not apt:
        return False
    # Detect distro + arch via /etc/os-release + uname
    os_release: Dict[str, str] = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os_release[k] = v.strip('"')
    except Exception:
        return False
    distro_id = os_release.get("ID", "").lower()
    version_id = os_release.get("VERSION_ID", "").replace(".", "")
    if distro_id == "ubuntu":
        if version_id not in ("1804", "2004", "2204", "2404"):
            return False
        repo_path = f"ubuntu{version_id}"
    elif distro_id == "debian":
        if version_id not in ("11", "12"):
            return False
        repo_path = f"debian{version_id}"
    else:
        return False
    import platform as _plat
    machine = _plat.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch_seg = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch_seg = "sbsa"
    else:
        return False

    # Pick CUDA toolkit version: max(arch_min, torch.version.cuda).
    # Per-arch targeting ensures sm_120a gets 12.8+, sm_103a gets 12.9+, etc.
    target_ver = "12-6"
    if arch and arch in ARCH_TABLE and ARCH_TABLE[arch].vendor == "cuda":
        tv = _target_cuda_version_for_arch(arch)
        if tv >= (12, 0):
            target_ver = f"{tv[0]}-{tv[1]}"
    else:
        try:
            import torch
            v = (torch.version.cuda or "").strip()
            parts = v.split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                if int(parts[0]) >= 12:
                    target_ver = f"{parts[0]}-{parts[1]}"
        except Exception:
            pass

    keyring_url = (f"https://developer.download.nvidia.com/compute/cuda/repos/"
                   f"{repo_path}/{arch_seg}/cuda-keyring_1.1-1_all.deb")
    stream.write(f"[bootstrap] trying NVIDIA's official apt repo for "
                 f"{distro_id} {version_id} ({arch_seg}); target "
                 f"cuda-toolkit-{target_ver}\n")
    stream.flush()

    import tempfile
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    sudo = _sudo_prefix()
    fetched = False
    keyring_deb = tempfile.NamedTemporaryFile(suffix=".deb", delete=False).name
    try:
        for downloader in (["wget", "-q", "-O", keyring_deb, keyring_url],
                           ["curl", "-sLfo", keyring_deb, keyring_url]):
            if shutil.which(downloader[0]):
                if subprocess.call(downloader) == 0 and \
                        os.path.getsize(keyring_deb) > 0:
                    fetched = True
                    break
        if not fetched:
            stream.write(f"[bootstrap] could not download keyring from "
                         f"{keyring_url}; skipping NVIDIA-repo path\n")
            return False
        rc = subprocess.call(sudo + ["dpkg", "-i", keyring_deb], env=env)
        if rc != 0:
            stream.write("[bootstrap] dpkg -i cuda-keyring FAILED\n")
            return False
        if subprocess.call(sudo + [apt, "update", "-qq"], env=env) != 0:
            stream.write("[bootstrap] apt update after adding NVIDIA repo "
                         "FAILED\n")
            return False
        candidates = [f"cuda-toolkit-{target_ver}", "cuda-toolkit-12-6",
                      "cuda-toolkit-12-4", "cuda-toolkit-12-3",
                      "cuda-toolkit-12-2", "cuda-toolkit-12-1",
                      "cuda-toolkit-12-0", "cuda-toolkit"]
        seen: set = set()
        for pkg in candidates:
            if pkg in seen:
                continue
            seen.add(pkg)
            stream.write(f"[bootstrap] apt install {pkg}\n")
            stream.flush()
            rc = subprocess.call(sudo + [apt, "install", "-y", "-qq", pkg],
                                 env=env)
            if rc == 0:
                # NVIDIA's package puts everything under /usr/local/cuda-<ver>/
                for cuda_dir in [Path("/usr/local/cuda")] + sorted(
                        Path("/usr/local").glob("cuda-*"), reverse=True):
                    if (cuda_dir / "bin" / "nvcc").is_file():
                        os.environ["CUDA_HOME"] = str(cuda_dir)
                        bin_path = str(cuda_dir / "bin")
                        current = os.environ.get("PATH", "")
                        if bin_path not in current.split(os.pathsep):
                            os.environ["PATH"] = f"{bin_path}{os.pathsep}{current}"
                        break
                if _ensure_nvcc_on_path():
                    return True
    finally:
        try:
            os.unlink(keyring_deb)
        except OSError:
            pass
    return False


def _bootstrap_cuda_via_apt(stream) -> bool:
    """Debian / Ubuntu / Colab / Mint — stock distro package.

    Note: this is often years behind (e.g. CUDA 11.5 on Ubuntu 22.04),
    so ``_bootstrap_cuda_via_nvidia_apt_repo`` is preferred for sm_90
    and other newer arches. Kept as a fallback for hosts where adding
    NVIDIA's repo failed (corp firewall, custom apt config, etc.)."""
    apt = shutil.which("apt-get") or shutil.which("apt")
    if not apt:
        return False
    stream.write("[bootstrap] trying apt-get install nvidia-cuda-toolkit "
                 "(stock distro package; may be older than CUDA 12)\n")
    stream.flush()
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    sudo = _sudo_prefix()
    subprocess.call(sudo + [apt, "update", "-qq"], env=env)
    rc = subprocess.call(
        sudo + [apt, "install", "-y", "-qq", "nvidia-cuda-toolkit"], env=env)
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_dnf(stream) -> bool:
    """Fedora / RHEL 8+ / Rocky / Alma. Tries `cuda` first
    (works once NVIDIA repo is added), then `nvidia-cuda-toolkit`
    (RPMFusion)."""
    dnf = shutil.which("dnf")
    if not dnf:
        return False
    sudo = _sudo_prefix()
    for pkg in ("cuda", "nvidia-cuda-toolkit"):
        stream.write(f"[bootstrap] trying dnf install {pkg}\n")
        stream.flush()
        rc = subprocess.call(sudo + [dnf, "install", "-y", pkg])
        if rc == 0 and _ensure_nvcc_on_path():
            return True
    return False


def _bootstrap_cuda_via_yum(stream) -> bool:
    """RHEL 7 / CentOS 7."""
    yum = shutil.which("yum")
    if not yum:
        return False
    sudo = _sudo_prefix()
    for pkg in ("cuda", "nvidia-cuda-toolkit"):
        stream.write(f"[bootstrap] trying yum install {pkg}\n")
        stream.flush()
        rc = subprocess.call(sudo + [yum, "install", "-y", pkg])
        if rc == 0 and _ensure_nvcc_on_path():
            return True
    return False


def _bootstrap_cuda_via_zypper(stream) -> bool:
    """openSUSE / SLES."""
    zypper = shutil.which("zypper")
    if not zypper:
        return False
    sudo = _sudo_prefix()
    for pkg in ("cuda", "nvidia-cuda-toolkit"):
        stream.write(f"[bootstrap] trying zypper install {pkg}\n")
        stream.flush()
        rc = subprocess.call(sudo + [zypper, "--non-interactive", "install",
                                     "-y", pkg])
        if rc == 0 and _ensure_nvcc_on_path():
            return True
    return False


def _bootstrap_cuda_via_pacman(stream) -> bool:
    """Arch / Manjaro / EndeavourOS — `cuda` is in extra repo."""
    pac = shutil.which("pacman")
    if not pac:
        return False
    stream.write("[bootstrap] trying pacman -S cuda\n")
    stream.flush()
    sudo = _sudo_prefix()
    rc = subprocess.call(sudo + [pac, "-S", "--noconfirm", "cuda"])
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_apk(stream) -> bool:
    """Alpine Linux — `cuda` is in the testing repo."""
    apk = shutil.which("apk")
    if not apk:
        return False
    stream.write("[bootstrap] trying apk add cuda\n")
    stream.flush()
    sudo = _sudo_prefix()
    rc = subprocess.call(sudo + [apk, "add", "--no-cache", "cuda"])
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_brew(stream) -> bool:
    """macOS. CUDA was discontinued on Mac after 10.13 / CUDA 10.2, so
    this almost never succeeds — but we try the Homebrew formula in
    case the user has an old Tesla-era Mac."""
    brew = shutil.which("brew")
    if not brew:
        return False
    stream.write("[bootstrap] trying brew install --cask cuda "
                 "(macOS — typically only works on pre-2019 systems)\n")
    stream.flush()
    rc = subprocess.call([brew, "install", "--cask", "cuda"])
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_winget(stream) -> bool:
    """Windows — Microsoft App Installer / winget."""
    winget = shutil.which("winget")
    if not winget:
        return False
    stream.write("[bootstrap] trying winget install Nvidia.CUDA\n")
    stream.flush()
    rc = subprocess.call([winget, "install", "--id", "Nvidia.CUDA",
                          "--silent", "--accept-package-agreements",
                          "--accept-source-agreements"])
    return rc == 0 and _ensure_nvcc_on_path() is not None


def _bootstrap_cuda_via_pypi_wheels(stream) -> bool:
    """LAST RESORT: NVIDIA's PyPI wheels. These ship ptxas, libnvvm,
    libcudart, and the headers — but NOT the nvcc compiler driver.
    Useful for filling in the dependency surface AFTER nvcc has been
    installed by another method. Listed as a fallback for the rare
    case where the wheels include enough to satisfy build attempts."""
    preferred = "cu12"
    try:
        import torch
        v = (torch.version.cuda or "").strip()
        if v.startswith("13"):
            preferred = "cu13"
        elif v.startswith("12"):
            preferred = "cu12"
        elif v.startswith("11"):
            preferred = "cu11"
    except Exception:
        pass
    seen: List[str] = []
    for tag in (preferred, "cu12", "cu11"):
        if tag in seen:
            continue
        seen.append(tag)
        pkgs = [f"nvidia-cuda-nvcc-{tag}", f"nvidia-cuda-runtime-{tag}",
                f"nvidia-cuda-cccl-{tag}", f"nvidia-cuda-nvrtc-{tag}"]
        stream.write(f"[bootstrap] trying NVIDIA PyPI wheels ({tag}) — "
                     "note: does NOT include nvcc binary\n")
        stream.flush()
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                              *pkgs])
        if rc == 0 and _ensure_nvcc_on_path():
            return True
    return False


def bootstrap_cuda_toolkit(stream=None, arch: Optional[str] = None) -> bool:
    """Install the CUDA toolkit (nvcc + runtime + headers) on demand.

    Works on any host where one of these is available:
      - conda  (preferred — cross-platform, no sudo, no repo setup)
      - apt-get/apt  (Debian / Ubuntu / Colab / Mint)
      - dnf  (Fedora / RHEL 8+ / Rocky / Alma)
      - yum  (RHEL 7 / CentOS 7)
      - zypper  (openSUSE / SLES)
      - pacman  (Arch / Manjaro / EndeavourOS)
      - apk  (Alpine Linux)
      - brew  (macOS — legacy CUDA only)
      - winget  (Windows 10+)
      - pip + NVIDIA PyPI wheels  (partial — no nvcc, but useful libs)

    Probes them in priority order: conda first if you're in a conda env
    (no sudo), then any system package manager that's on PATH, then
    PyPI wheels as a partial fallback. Returns True iff nvcc is on PATH
    afterwards.

    ``arch`` (optional) targets a specific CUDA version. When given, the
    NVIDIA apt-repo path installs the cuda-toolkit-XX-Y package matching
    ``max(ARCH_TABLE[arch].min_toolchain_version, torch.version.cuda)``
    so sm_120a / sm_103a get CUDA 12.8+ / 12.9+ as required.
    """
    if stream is None:
        stream = sys.stderr
    if shutil.which("nvcc") or _ensure_nvcc_on_path():
        # Already have nvcc — but if it's < CUDA 12.0 we may still want
        # to upgrade since sm_90 / sm_100 / sm_103 / sm_120 won't build.
        ver = _probe_nvcc_version()
        if ver and ver < (12, 0):
            stream.write(f"[bootstrap] found nvcc {ver[0]}.{ver[1]} but it's "
                         "older than CUDA 12.0 (required for sm_90+); will "
                         "still try to install a newer toolkit\n")
        else:
            stream.write("[bootstrap] nvcc already available; skipping install\n")
            return True

    in_conda_env = bool(os.environ.get("CONDA_PREFIX"))
    # Priority order: prefer conda when we're in an active env (no sudo,
    # cross-platform). Otherwise prefer the host's native package manager.
    methods: List[Tuple[str, Any]] = []
    if in_conda_env:
        methods.append(("conda", _bootstrap_cuda_via_conda))
    # Native system package managers — one of these matches on any
    # Linux/macOS/Windows host.
    methods.extend([
        # Prefer NVIDIA's official apt repo on Debian/Ubuntu — it
        # installs CUDA 12.x to /usr/local/cuda/ (the layout torch
        # expects), whereas the stock package is often CUDA 11.x.
        ("nvidia-apt", _bootstrap_cuda_via_nvidia_apt_repo),
        ("apt",      _bootstrap_cuda_via_apt),
        ("dnf",      _bootstrap_cuda_via_dnf),
        ("yum",      _bootstrap_cuda_via_yum),
        ("zypper",   _bootstrap_cuda_via_zypper),
        ("pacman",   _bootstrap_cuda_via_pacman),
        ("apk",      _bootstrap_cuda_via_apk),
        ("brew",     _bootstrap_cuda_via_brew),
        ("winget",   _bootstrap_cuda_via_winget),
    ])
    # If we have conda available but weren't in an env, try it as a
    # late fallback (creates / modifies base env, less ideal).
    if not in_conda_env and (shutil.which("mamba") or shutil.which("conda")):
        methods.append(("conda", _bootstrap_cuda_via_conda))
    # PyPI as final, partial fallback.
    methods.append(("pip-wheels", _bootstrap_cuda_via_pypi_wheels))

    tried: List[str] = []
    for name, fn in methods:
        try:
            # The NVIDIA apt-repo path knows how to pick the right toolkit
            # version per target arch; all other methods take stream only.
            if name == "nvidia-apt":
                ok = fn(stream, arch)
            else:
                ok = fn(stream)
        except Exception as exc:
            stream.write(f"[bootstrap] {name} raised {type(exc).__name__}: "
                         f"{exc}\n")
            tried.append(f"{name} (errored)")
            continue
        tried.append(name)
        if ok:
            found = _ensure_nvcc_on_path()
            if found:
                _refresh_torch_cuda_home()
                stream.write(f"[bootstrap] OK ({name}) — nvcc at {found}\n")
                return True
            stream.write(f"[bootstrap] {name} reported success but nvcc "
                         "still not findable; continuing\n")

    stream.write(
        "\n[bootstrap] FAILED to obtain nvcc on this host.\n"
        f"  Attempted: {tried}\n"
        "  Manual install — pick the one that matches your environment:\n"
        "    conda install -c nvidia cuda-nvcc cuda-runtime    # any OS, conda env\n"
        "    sudo apt-get install nvidia-cuda-toolkit          # Debian / Ubuntu / Colab\n"
        "    sudo dnf install cuda                             # Fedora / RHEL 8+ (NVIDIA repo)\n"
        "    sudo zypper install cuda                          # openSUSE / SLES\n"
        "    sudo pacman -S cuda                               # Arch / Manjaro\n"
        "    sudo apk add cuda                                 # Alpine\n"
        "    winget install Nvidia.CUDA                        # Windows 10+\n"
        "    https://developer.nvidia.com/cuda-downloads       # official .run installer\n"
        "  NOTE: NVIDIA's PyPI wheels (nvidia-cuda-nvcc-cuXX) ship ptxas, libnvvm,\n"
        "  libcudart, and headers — but NOT the nvcc compiler driver. The bootstrap\n"
        "  attempts them as a partial fallback to populate the dependency surface.\n"
    )
    return False


# ---------------------------------------------------------------------------
# ROCm bootstrap — mirrors bootstrap_cuda_toolkit (multi-pm probe) — Stream 12
# ---------------------------------------------------------------------------
#
# Same pattern as CUDA: try several package managers in priority order, give
# up loudly with a manual-install recipe at the end. Picks per-arch target
# version via ``_target_rocm_version_for_arch`` so gfx950 gets ROCm 6.2+ and
# gfx1200/gfx1201 get ROCm 7.0+.

def _bootstrap_rocm_via_amd_apt_repo(stream, arch: str) -> bool:
    """Install ROCm via AMD's official apt repo.

    Picks version per target arch: gfx942/gfx950 → 6.x, gfx1200/gfx1201 → 7.x.
    The repo layout is ``https://repo.radeon.com/rocm/apt/<ver>/`` and the
    GPG key lives at ``https://repo.radeon.com/rocm/rocm.gpg.key``.
    """
    if not shutil.which("apt-get"):
        return False
    sudo = _sudo_prefix()
    target = _target_rocm_version_for_arch(arch)
    if target == (0, 0):
        # Non-HIP arch — bail out cleanly.
        return False
    rocm_ver_str = f"{target[0]}.{target[1]}"
    rocm_key = "https://repo.radeon.com/rocm/rocm.gpg.key"
    rocm_repo_url = f"https://repo.radeon.com/rocm/apt/{rocm_ver_str}"
    stream.write(f"[bootstrap_rocm] trying AMD apt repo for ROCm "
                 f"{rocm_ver_str} (arch={arch})\n")
    stream.flush()
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    try:
        # Fetch key (try wget first, then curl)
        key_path = "/tmp/rocm.gpg.key"
        fetched = False
        for downloader in (
            ["wget", "-q", "-O", key_path, rocm_key],
            ["curl", "-sLfo", key_path, rocm_key],
        ):
            if shutil.which(downloader[0]):
                if subprocess.call(downloader) == 0:
                    fetched = True
                    break
        if not fetched:
            stream.write(f"[bootstrap_rocm:apt] could not download GPG key "
                         f"from {rocm_key}\n")
            return False
        subprocess.run(sudo + ["apt-key", "add", key_path],
                       check=True, timeout=30, env=env)
        # Add repo
        sources_line = f"deb [arch=amd64] {rocm_repo_url} ubuntu main"
        list_path = "/etc/apt/sources.list.d/rocm.list"
        subprocess.run(
            sudo + ["bash", "-c", f"echo '{sources_line}' > {list_path}"],
            check=True, timeout=10, env=env)
        subprocess.run(sudo + ["apt-get", "update", "-qq"],
                       check=True, timeout=120, env=env)
        # Install
        subprocess.run(
            sudo + ["apt-get", "install", "-y", "rocm-dev", "hip-dev"],
            check=True, timeout=900, env=env)
        stream.write(f"[bootstrap_rocm] installed ROCm {rocm_ver_str} via "
                     "AMD apt repo\n")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError) as exc:
        stream.write(f"[bootstrap_rocm:apt] FAILED: {exc}\n")
        return False


def _bootstrap_rocm_via_apt_stock(stream, arch: str) -> bool:
    """Ubuntu stock rocm-hip-runtime / rocm-dev packages. Fallback when the
    AMD apt repo isn't reachable (corp firewall, custom apt config)."""
    if not shutil.which("apt-get"):
        return False
    sudo = _sudo_prefix()
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    stream.write("[bootstrap_rocm] trying stock apt-get install "
                 "rocm-dev rocm-libs hipcc\n")
    stream.flush()
    try:
        subprocess.run(sudo + ["apt-get", "update", "-qq"],
                       check=True, timeout=120, env=env)
        subprocess.run(
            sudo + ["apt-get", "install", "-y", "rocm-dev", "rocm-libs",
                    "hipcc"],
            check=True, timeout=900, env=env)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return False


def _bootstrap_rocm_via_dnf(stream, arch: str) -> bool:
    """Fedora / RHEL 8+ / Rocky / Alma — rocm-hip-devel."""
    if not shutil.which("dnf"):
        return False
    sudo = _sudo_prefix()
    stream.write("[bootstrap_rocm] trying dnf install rocm-hip-devel\n")
    stream.flush()
    try:
        subprocess.run(sudo + ["dnf", "install", "-y", "rocm-hip-devel"],
                       check=True, timeout=900)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return False


def _bootstrap_rocm_via_zypper(stream, arch: str) -> bool:
    """openSUSE / SLES — rocm-hip-devel."""
    if not shutil.which("zypper"):
        return False
    sudo = _sudo_prefix()
    stream.write("[bootstrap_rocm] trying zypper install rocm-hip-devel\n")
    stream.flush()
    try:
        subprocess.run(
            sudo + ["zypper", "--non-interactive", "install", "-y",
                    "rocm-hip-devel"],
            check=True, timeout=900)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return False


def bootstrap_rocm_toolkit(stream=None, arch: str = "gfx942") -> bool:
    """Universal ROCm bootstrap. Tries AMD's official repo first, falls back
    to distro stock packages. Returns True iff hipcc is on PATH after.

    The probe order matches ``bootstrap_cuda_toolkit``: prefer the vendor's
    own repo (it ships the version we want), then distro stock as a partial
    fallback, then dnf/zypper.
    """
    if stream is None:
        stream = sys.stderr
    if shutil.which("hipcc"):
        stream.write("[bootstrap_rocm] hipcc already on PATH; skipping\n")
        return True
    methods: List[Tuple[str, Callable[..., bool]]] = [
        ("amd-apt",       _bootstrap_rocm_via_amd_apt_repo),
        ("apt-stock",     _bootstrap_rocm_via_apt_stock),
        ("dnf",           _bootstrap_rocm_via_dnf),
        ("zypper",        _bootstrap_rocm_via_zypper),
    ]
    tried: List[str] = []
    for name, fn in methods:
        try:
            ok = fn(stream, arch)
        except Exception as exc:
            stream.write(f"[bootstrap_rocm] {name} raised "
                         f"{type(exc).__name__}: {exc}\n")
            tried.append(f"{name} (errored)")
            continue
        tried.append(name)
        if ok and shutil.which("hipcc"):
            stream.write(f"[bootstrap_rocm] OK ({name}) — hipcc at "
                         f"{shutil.which('hipcc')}\n")
            return True
    stream.write(
        "\n[bootstrap_rocm] FAILED to install ROCm on this host.\n"
        f"  Attempted: {tried}\n"
        "  Manual install — pick the one matching your environment:\n"
        "    sudo apt-get install rocm-hip-sdk        # Debian / Ubuntu\n"
        "    sudo dnf install rocm-hip-devel          # Fedora / RHEL\n"
        "    sudo zypper install rocm-hip-devel       # openSUSE\n"
        "    sudo pacman -S rocm-hip-sdk              # Arch\n"
        "    https://rocm.docs.amd.com/projects/install-on-linux/\n"
        "  Then set os.environ['ROCM_PATH'] = '/opt/rocm' and prepend\n"
        "  $ROCM_PATH/bin to PATH.\n"
    )
    return False


# ---------------------------------------------------------------------------
# JAX/TPU bootstrap — Stream 12
# ---------------------------------------------------------------------------

def bootstrap_jax_tpu(stream=None, arch: str = "tpu_v5p") -> bool:
    """Detect TPU runtime via ``jax.devices()``; install ``jax[tpu]`` from
    the libtpu_releases bucket when missing. Returns True iff a TPU device
    is visible after.

    On a non-TPU host this returns False (no TPU to bind to), which is the
    correct outcome — the caller (e.g. self-test) only requires "no crash".
    """
    if stream is None:
        stream = sys.stderr
    # Probe first: skip the install if a TPU is already visible.
    try:
        import jax
        devs = jax.devices()
        if any(getattr(d, "platform", "") == "tpu" for d in devs):
            stream.write(f"[bootstrap_jax] TPU already visible: {devs}\n")
            return True
    except (ImportError, RuntimeError):
        pass
    except Exception as exc:
        stream.write(f"[bootstrap_jax] jax.devices() raised "
                     f"{type(exc).__name__}: {exc}; continuing to install\n")
    # Pick a minimum jax version per arch from ARCH_TABLE.
    min_jax_ver = ">=0.4.30"
    if arch in ARCH_TABLE and ARCH_TABLE[arch].vendor == "pallas":
        need = ARCH_TABLE[arch].min_toolchain_version
        if len(need) >= 3:
            min_jax_ver = f">={need[0]}.{need[1]}.{need[2]}"
        elif len(need) == 2:
            min_jax_ver = f">={need[0]}.{need[1]}"
    stream.write(f"[bootstrap_jax] installing jax[tpu]{min_jax_ver} "
                 f"(arch={arch})\n")
    stream.flush()
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             f"jax[tpu]{min_jax_ver}",
             "-f", "https://storage.googleapis.com/jax-releases/"
                   "libtpu_releases.html"],
            check=True, timeout=600)
        stream.write("[bootstrap_jax] installed jax[tpu]\n")
        # Re-probe after install. Reload jax so a stale `import jax` from
        # before the install doesn't shadow the new install.
        import importlib
        if "jax" in sys.modules:
            try:
                importlib.reload(sys.modules["jax"])
            except Exception:
                pass
        import jax
        devs = jax.devices()
        ok = any(getattr(d, "platform", "") == "tpu" for d in devs)
        stream.write(f"[bootstrap_jax] post-install devices={devs}\n")
        return ok
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ImportError, RuntimeError, OSError) as exc:
        stream.write(f"[bootstrap_jax] FAILED: {exc}\n")
        return False


# ---------------------------------------------------------------------------
# Unified bootstrap entry — dispatch by vendor — Stream 12
# ---------------------------------------------------------------------------

def bootstrap_toolchain(arch: str, stream=None) -> bool:
    """Dispatch to the right per-vendor bootstrap based on ARCH_TABLE.

    Returns the underlying bootstrap function's True/False. Unknown arches
    log a message and return False; the caller can treat that as "skip".
    """
    if stream is None:
        stream = sys.stderr
    if arch not in ARCH_TABLE:
        stream.write(f"[bootstrap] unknown arch {arch}\n")
        return False
    vendor = ARCH_TABLE[arch].vendor
    if vendor == "cuda":
        return bootstrap_cuda_toolkit(stream, arch=arch)
    elif vendor == "hip":
        return bootstrap_rocm_toolkit(stream, arch=arch)
    elif vendor == "pallas":
        return bootstrap_jax_tpu(stream, arch=arch)
    stream.write(f"[bootstrap] unknown vendor {vendor!r} for arch {arch}\n")
    return False


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


def _preflight_toolchain(arch: str) -> List[str]:
    """Pre-flight: probe the toolchain for the chosen arch.

    Returns a list of human-readable diagnostic lines (warnings, paths,
    versions) — the build proceeds whether or not the toolchain looks
    healthy; failures get surfaced by build_aot() with a much louder
    error including the actual compiler stderr. The purpose here is to
    give the user a clear ``[preflight]`` block at the top of the run
    so they can spot a missing dependency before waiting for the build
    to fail."""
    lines: List[str] = []
    vendor = get_arch_entry(arch).vendor
    lines.append(f"[preflight] arch={arch} vendor={vendor}")

    if vendor == "cuda":
        nvcc = _ensure_nvcc_on_path()
        cuda_home = os.environ.get("CUDA_HOME", "")
        lines.append(f"[preflight] CUDA_HOME={cuda_home or '<unset>'}")
        if nvcc:
            lines.append(f"[preflight] nvcc={nvcc}")
            nvcc_ver = _probe_nvcc_version()
            try:
                out = subprocess.check_output([nvcc, "--version"], text=True,
                                              timeout=10).strip().splitlines()
                if out:
                    lines.append(f"[preflight] {out[-1]}")
            except Exception:
                pass
            # Hard requirement: each CUDA arch has a min nvcc version (see
            # ARCH_TABLE[arch].min_toolchain_version). Warn loudly if the
            # detected nvcc is too old — otherwise the build dies with
            # cryptic ptxas errors.
            need = get_arch_entry(arch).min_toolchain_version
            if nvcc_ver and need and nvcc_ver < need:
                lines.append(
                    f"[preflight] WARNING: nvcc {nvcc_ver[0]}.{nvcc_ver[1]} "
                    f"is too old for {arch} (needs CUDA {need[0]}.{need[1]}+).\n"
                    f"  This is usually the stock distro package "
                    f"(nvidia-cuda-toolkit on Ubuntu 22.04 = CUDA 11.5).\n"
                    f"  Fix: install NVIDIA's official cuda-toolkit-12-x via\n"
                    f"  their apt repo, or call build(bootstrap_cuda=True)\n"
                    f"  which prefers nvidia-apt over stock apt and pulls\n"
                    f"  a CUDA 12.x that supports {arch}."
                )
        else:
            lines.append(
                "[preflight] WARNING: nvcc NOT found. The build will fail "
                "with 'CUDA_HOME environment variable is not set' or "
                "similar. To fix:\n"
                "  - Verify the CUDA Toolkit is installed (look for "
                "/usr/local/cuda*/bin/nvcc).\n"
                "  - On Colab CPU runtimes there's no nvcc; switch to a "
                "GPU runtime (Runtime → Change runtime type → GPU).\n"
                "  - On hosts where nvcc lives at a versioned path, "
                "set CUDA_HOME accordingly:\n"
                "      os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'"
            )
        # Check libcudart presence (cpp_extension needs it to link)
        if cuda_home:
            for lib in ("lib64/libcudart.so", "lib/libcudart.so"):
                if (Path(cuda_home) / lib).exists():
                    lines.append(f"[preflight] libcudart: {Path(cuda_home) / lib}")
                    break
            else:
                lines.append(f"[preflight] WARNING: libcudart.so not under {cuda_home}/lib[64]")
    elif vendor == "hip":
        hipcc = shutil.which("hipcc")
        rocm = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME", "")
        lines.append(f"[preflight] ROCM_PATH={rocm or '<unset>'}")
        if hipcc:
            lines.append(f"[preflight] hipcc={hipcc}")
        else:
            lines.append(
                "[preflight] WARNING: hipcc NOT found. Install ROCm and "
                "set os.environ['ROCM_PATH'] = '/opt/rocm' (or wherever "
                "it lives) then prepend $ROCM_PATH/bin to PATH.")
    elif vendor == "pallas":
        try:
            import jax  # noqa: F401
            lines.append(f"[preflight] jax={jax.__version__}")
        except Exception as e:
            lines.append(
                f"[preflight] WARNING: jax not importable: {e}. "
                "For TPU, pip install 'jax[tpu]' -f "
                "https://storage.googleapis.com/jax-releases/libtpu_releases.html")

    # Universal checks
    for tool in ("ninja", "g++"):
        p = shutil.which(tool)
        if p:
            lines.append(f"[preflight] {tool}={p}")
        else:
            lines.append(f"[preflight] WARNING: {tool} NOT on PATH")

    # ── Per-arch toolchain min-version PASS/FAIL — Stream 12 ─────────
    # Explicit one-line judgment per arch so the [preflight] block can be
    # grep'd for FAIL by CI / wrapper scripts without parsing every WARNING.
    entry = ARCH_TABLE.get(arch)
    if entry is None:
        lines.append(f"[preflight] arch {arch} not in ARCH_TABLE — FAIL")
        return lines
    need = entry.min_toolchain_version
    if entry.vendor == "cuda":
        have = _probe_nvcc_version()
        if have is None:
            lines.append(
                f"[preflight] arch={arch} need CUDA>={need[0]}.{need[1]}: "
                "nvcc not found — FAIL")
        elif have < (need[0], need[1]):
            lines.append(
                f"[preflight] arch={arch} need CUDA>={need[0]}.{need[1]} "
                f"have {have[0]}.{have[1]} — FAIL")
        else:
            lines.append(
                f"[preflight] arch={arch} need CUDA>={need[0]}.{need[1]} "
                f"have {have[0]}.{have[1]} — PASS")
    elif entry.vendor == "hip":
        have = _probe_hipcc_version()
        if have is None:
            lines.append(
                f"[preflight] arch={arch} need ROCm>={need[0]}.{need[1]}: "
                "hipcc not found — FAIL")
        elif have < (need[0], need[1]):
            lines.append(
                f"[preflight] arch={arch} need ROCm>={need[0]}.{need[1]} "
                f"have {have[0]}.{have[1]} — FAIL")
        else:
            lines.append(
                f"[preflight] arch={arch} need ROCm>={need[0]}.{need[1]} "
                f"have {have[0]}.{have[1]} — PASS")
    elif entry.vendor == "pallas":
        try:
            import jax
            try:
                jv = tuple(int(x) for x in jax.__version__.split(".")[:3])
            except (ValueError, AttributeError):
                jv = (0, 0, 0)
            need3 = need if len(need) == 3 else (need[0], need[1], 0)
            cmp_jv = jv[:len(need3)]
            need_str = ".".join(str(x) for x in need3)
            if cmp_jv >= need3:
                lines.append(
                    f"[preflight] arch={arch} JAX {jax.__version__} "
                    f">= {need_str} — PASS")
            else:
                lines.append(
                    f"[preflight] arch={arch} JAX {jax.__version__} "
                    f"< {need_str} — FAIL")
        except ImportError:
            lines.append(f"[preflight] arch={arch} JAX not installed — FAIL")
    return lines


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
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "pallas":
        return []
    backend = REPO_ROOT / "csrc/backends" / entry.subdir
    bindings = sorted((REPO_ROOT / "csrc/bindings").glob("*.cpp"))
    launchers: List[Path] = []
    for g in entry.launcher_glob:
        launchers.extend(sorted(backend.glob(g)))
    models: List[Path] = []
    for g in entry.model_glob:
        models.extend(sorted((backend / "models").glob(g)))
    return bindings + launchers + models


def _build_macros(spec: BuildSpec) -> List[str]:
    entry = get_arch_entry(spec.arch)
    macros = [
        f"-DSG_BUILD_OPTIMIZER_{spec.optimizer.upper()}=1",
        f"-DSG_BUILD_MODEL_{spec.model.upper()}=1",
        f"-D{entry.macro}=1",
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

# NVCC base flags — arch-specific -gencode is appended per-build from
# ARCH_TABLE[arch].nvcc_gencode in _device_cflags(). Keeping the base
# table arch-agnostic means a new CUDA arch is one ARCH_TABLE entry, not
# a code change here.
#
# Stream 3: every PTXAS / fatbin / device-link knob that helps perf is
# pulled into this list. There is exactly ONE ``-Xptxas --opt-level=3``
# (the duplicate ``-Xptxas -O3`` from earlier revisions was removed —
# nvcc treats both spellings identically and warns about the duplicate).
# Version-gated additions (CUDA 12.0+ register-usage-level, 13.0+
# --minimal, 12.6+ --split-compile) live in ``_newer_compiler_flags``.
NVCC_DEVICE_BASE = [
    "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
    "--expt-relaxed-constexpr",
    "--threads", "8",
    "-Xfatbin", "-compress-all",
    # NOTE: keep exactly one PTXAS opt-level flag. ``--opt-level=3`` is the
    # documented long form; earlier revisions also had ``-Xptxas -O3`` which
    # nvcc accepted but warned was redundant.
    "-Xptxas", "--opt-level=3",
    "-Xptxas", "-v", "-Xptxas", "--warn-on-spills",
    "-Xptxas", "--allow-expensive-optimizations=true",
    "-Xptxas", "--def-load-cache=ca",
    "-Xptxas", "--def-store-cache=wb",
    "--extra-device-vectorization",
    "-Xcompiler", "-fPIC", "-Xcompiler", "-flto=full",
    "-Xcompiler", "-fno-strict-aliasing",
    # Quiet the linker on big templated kernels — these warnings are
    # CUTLASS/cuBLASLt routine and never actionable.
    "-Xnvlink", "--suppress-stack-size-warning",
    # CUDA 12+ deprecates a couple of constexpr-related diags that fire
    # in third-party headers (CCCL, CUTLASS). Suppress so build logs stay
    # actionable. --diag-suppress is silently ignored by older nvcc.
    "--diag-suppress=20012,20013",
    "--resource-usage",
    "-dlto",
    # Pin the device-link step to LTO too (idempotent with -dlto on the
    # main line but explicit so the device link doesn't silently downgrade
    # when the host driver passes its own --device-link-options).
    "--device-link-options=-dlto",
]

# HIPCC base flags — --offload-arch is appended per-build from
# ARCH_TABLE[arch].hipcc_offload_arch in _device_cflags().
#
# Stream 3: every AMDGPU LLVM knob that helps perf is in this list. The
# wave-size / cumode toggles (which are NOT arch-agnostic) live in
# _device_cflags(spec) and are gated by ARCH_TABLE[arch].warp_size.
HIPCC_DEVICE_BASE = [
    "-O3", "-std=c++17", "-DWITH_HIP",
    "-ffast-math", "-fPIC",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-amdgpu-internalize-symbols",
    # ROCm-LLVM perf knobs (Stream 3): aggressive unroll, module-scope
    # LDS lowering, larger alloca-to-vector promotion limit, vector SROA
    # element bump, and merge-m0 transform. All five are documented
    # AMDGPU backend options safe on every gfx target.
    "-mllvm", "--amdgpu-unroll-threshold=1000",
    "-mllvm", "--amdgpu-enable-lower-module-lds-strategy=module",
    "-mllvm", "--amdgpu-promote-alloca-to-vector-limit=512",
    "-mllvm", "--amdgpu-sroa-vector-elements=8",
    "-mllvm", "--amdgpu-enable-merge-m0",
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
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "pallas":
        return []
    base = list(HOST_CFLAGS_BASE) + [f"-D{entry.host_define}"]
    if spec.debug_symbols or spec.profile:
        base += ["-ggdb"]
    return base + _build_macros(spec)


def _device_cflags(spec: BuildSpec) -> List[str]:
    """Build the full per-arch device-compiler flag list.

    Stream 3: every arch-specific knob (gencode, offload-arch, CDNA/RDNA
    wave-size selection, fp4/fp8/cluster macros) is appended here based
    on ARCH_TABLE[spec.arch] — there are no remaining ``if arch == "sm_90"``
    branches scattered through the file. Version-gated additions (CUDA
    12.0+ register-usage-level, CUDA 13.0+ --minimal, etc.) live in
    ``_newer_compiler_flags`` so the gate is colocated with the probe.
    """
    entry = get_arch_entry(spec.arch)
    feats = entry.features

    if entry.vendor == "cuda":
        base = list(NVCC_DEVICE_BASE)
        # Per-arch -gencode pair (target SASS + PTX fallback) from ARCH_TABLE.
        # Stream 1 wired the SASS pair. Stream 3 guarantees a PTX fallback is
        # always present (defensive: if a future ARCH_TABLE edit drops the
        # fallback, splice it in here so older drivers can still JIT).
        gencodes = list(entry.nvcc_gencode)
        sm_num = entry.cutlass_arch
        if (sm_num is not None
                and not any(",code=compute_" in g for g in gencodes)):
            gencodes.append(
                f"-gencode=arch=compute_{sm_num},code=compute_{sm_num}")
        base += gencodes

        if spec.debug_symbols or spec.profile:
            base += ["-lineinfo", "--generate-line-info"]

        # CUTLASS integration: emit both the legacy single-arch token and
        # the modern multi-arch list (CUTLASS_NVCC_ARCHS_SUPPORTED). The
        # arch_suffix encodes the "a" qualifier needed for Hopper+ to
        # unlock TMA / wgmma / tcgen05 instructions.
        cutlass_arch_token = (
            f"{entry.cutlass_arch}{entry.arch_suffix}"
            if entry.cutlass_arch is not None else "")
        if (REPO_ROOT / "third_party/cutlass/include").exists():
            base += ["-DWITH_CUTLASS",
                     f"-DCUTLASS_NVCC_ARCHS={cutlass_arch_token}"]
        if cutlass_arch_token:
            # Independent of whether CUTLASS headers are vendored — the
            # supported-arch macro is consumed by host code that probes
            # availability at compile time.
            base += [f"-DCUTLASS_NVCC_ARCHS_SUPPORTED={cutlass_arch_token}"]

        # Feature-gated capability macros consumed by template specialisations.
        if "tma" in feats:
            base += ["-DCUDA_TMA_ENABLED=1"]
        if "wgmma" in feats:
            base += ["-DCUDA_WGMMA_ENABLED=1"]
        if "cluster" in feats:
            base += ["-DCUDA_CLUSTER_ENABLED=1"]
        if "fp8" in feats:
            base += ["-DCUDA_FP8_ENABLED=1"]
        if "fp4" in feats:
            base += ["-DCUDA_FP4_ENABLED=1"]
        if "tcgen05" in feats:
            base += ["-DCUDA_TCGEN05_ENABLED=1"]
        if "dyn_parallelism" in feats:
            base += ["-DCUDA_FORCE_CDP1_IF_SUPPORTED"]

        return base + _build_macros(spec)

    if entry.vendor == "hip":
        base = list(HIPCC_DEVICE_BASE)
        # Per-arch --offload-arch from ARCH_TABLE.
        if entry.hipcc_offload_arch:
            base += [f"--offload-arch={entry.hipcc_offload_arch}"]

        # CDNA (gfx9xx, wave64) gets -mcumode for compute-unit mode (vs the
        # default WGP mode that pairs CUs in RDNA). RDNA (gfx10xx+, wave32)
        # instead gets the wave32 enable + tgsplit toggle.
        if entry.warp_size == 64:
            base += ["-mcumode"]
        elif entry.warp_size == 32:
            base += ["-mtgsplit", "-mwavefrontsize32"]

        # Feature-gated AMDGPU capability macros (consumed by header specs).
        if "mfma" in feats:
            base += ["-DAMDGPU_MFMA_ENABLED=1"]
        if "wmma" in feats:
            base += ["-DAMDGPU_WMMA_ENABLED=1"]
        if "bf16_mfma" in feats:
            base += ["-DAMDGPU_BF16_MFMA=1"]
        if "fp8_mfma" in feats:
            base += ["-DAMDGPU_FP8_MFMA=1"]
        if "fp4_mfma" in feats:
            base += ["-DAMDGPU_FP4_MFMA=1"]
        if "tgsplit" in feats:
            base += ["-DAMDGPU_TGSPLIT=1"]
        if "dpp" in feats:
            base += ["-DAMDGPU_DPP=1"]
        if "fp8" in feats:
            base += ["-DAMDGPU_FP8_ENABLED=1"]

        # Debug-only: -fgpu-sanitize is opt-in (it adds nontrivial overhead).
        if spec.debug_symbols:
            base += ["-fgpu-sanitize"]
        if spec.debug_symbols or spec.profile:
            base += ["-ggdb"]
        return base + _build_macros(spec)
    return []


def _ldflags(spec: BuildSpec) -> List[str]:
    if get_arch_entry(spec.arch).vendor == "pallas":
        return []
    return list(LDFLAGS_BASE)


# ---- XLA / Pallas worker env (Stream 3) ----------------------------------
# When the build target is a Pallas / XLA arch (tpu_*) we don't emit C++
# flags — there is no host compiler. Instead, the perf knobs live in the
# environment variables consumed by the XLA backend at JIT time. The
# orchestrator merges this dict into the worker env at call sites; this
# helper is the single source of truth for which flags we set.

_XLA_FLAGS_BASE: Tuple[str, ...] = (
    "--xla_gpu_autotune_level=4",
    "--xla_gpu_dump_autotuned_gemm_fusions=true",
    "--xla_gpu_enable_triton_gemm=true",
    "--xla_gpu_enable_cublaslt=true",
    "--xla_gpu_enable_cudnn_fmha=true",
    "--xla_gpu_enable_async_collectives=true",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,CUBLAS,CUDNN,COLLECTIVES",
    "--xla_gpu_graph_level=3",
)


def _xla_env(arch: str, out_dir: Path) -> Dict[str, str]:
    """Return env-var dict for the XLA / Pallas worker subprocess.

    Empty dict for non-Pallas archs (so callers can unconditionally
    ``env.update(_xla_env(arch, out_dir))``). Stream 3 callers: the AOT
    + JIT subprocess spawners merge this on top of ``child_env()`` for
    tpu_* targets.
    """
    if arch not in ARCH_TABLE:
        return {}
    entry = get_arch_entry(arch)
    if entry.vendor != "pallas":
        return {}
    cache_dir = Path(out_dir) / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "XLA_FLAGS": " ".join(_XLA_FLAGS_BASE),
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        # Cache every compile — default skips short ones (≥1s by default).
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    }


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
    no-op on older toolchains. Dispatch is by ARCH_TABLE vendor so every
    CUDA arch (not just sm_90) benefits from version-gated flags.

    Stream 3: this is the single home for every version-gated NVCC / HIPCC
    flag. Arch-feature-gated flags (TMA, fp8, MFMA macros, cumode, etc.)
    live in ``_device_cflags(spec)`` since those depend on the static
    ArchEntry, not on the runtime toolchain probe.
    """
    extra_host: List[str] = []
    extra_device: List[str] = []
    if arch not in ARCH_TABLE:
        return extra_host, extra_device
    entry = get_arch_entry(arch)
    if entry.vendor == "cuda":
        ver = _probe_nvcc_version()
        if ver:
            if report:
                report.write(f"  [toolchain] nvcc {ver[0]}.{ver[1]}\n")
            # CUDA 12.0+: --register-usage-level=10 enables PTXAS to use
            # the full register budget (default 5 leaves slots on the table).
            if ver >= (12, 0):
                extra_device += ["-Xptxas", "--register-usage-level=10"]
                if report:
                    report.write("  [toolchain] enabling "
                                 "-Xptxas --register-usage-level=10 "
                                 "(CUDA ≥12.0)\n")
            # CUDA 11.4+: explicit device-link-options=-dlto belt-and-suspenders.
            # (NVCC_DEVICE_BASE already passes it on the main line; this is a
            # second copy with an explicit option-name spelling for the rare
            # multi-stage link.) Idempotent — nvcc dedupes.
            if ver >= (11, 4):
                extra_device += ["--device-link-options=-dlto"]
            if ver >= (12, 6):
                # NVCC 12.6+ supports --split-compile for opt-phase parallelism.
                extra_device += [f"--split-compile={NCPUS}"]
                if report:
                    report.write(f"  [toolchain] enabling "
                                 f"--split-compile={NCPUS} (NVCC ≥12.6)\n")
            # CUDA 13.0+: --minimal disables features the build doesn't need
            # (cuRTC, cudadevrt) to shrink the fatbin. Strict gate — earlier
            # nvcc rejects the flag.
            if ver >= (13, 0):
                extra_device += ["--minimal"]
                if report:
                    report.write("  [toolchain] enabling --minimal "
                                 "(NVCC ≥13.0)\n")
        elif report:
            report.write("  [toolchain] nvcc not on PATH; "
                         "skipping version-gated flags\n")
    elif entry.vendor == "hip":
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

    with_cuda = get_arch_entry(spec.arch).vendor in ("cuda", "hip")
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
            # Surface the actual compiler/linker error that torch's
            # cpp_extension swallowed. Look in the build directory for
            # ninja's stderr/log capture; these have the real diagnostics.
            try:
                import traceback as _tb
                tb_str = _tb.format_exc()
                report.write("\n[build FAILED — full traceback]\n")
                report.write(tb_str)
            except Exception:
                pass
            # Dump every log-like file in the build dir
            for log_name in ("build.ninja", ".ninja_log", "build.log",
                             "compile_commands.json"):
                lp = build_dir / log_name
                if lp.is_file():
                    try:
                        content = lp.read_text(errors="replace")
                    except Exception as read_exc:
                        report.write(f"\n[could not read {log_name}: {read_exc}]\n")
                        continue
                    report.write(f"\n[{log_name}] (head 4000 chars)\n")
                    report.write(content[:4000])
                    if len(content) > 4000:
                        report.write(f"\n... ({len(content) - 4000} more chars truncated)\n")
            # Try invoking ninja directly to capture stderr from the
            # actual nvcc/g++/hipcc subprocess — this is the gold path
            # for diagnosing "Error building extension" from torch.
            try:
                ninja_bin = shutil.which("ninja")
                if ninja_bin and (build_dir / "build.ninja").is_file():
                    report.write(f"\n[re-running ninja directly: {ninja_bin} -C {build_dir}]\n")
                    proc = subprocess.run(
                        [ninja_bin, "-C", str(build_dir), "-v"],
                        capture_output=True, text=True, timeout=120,
                    )
                    if proc.stdout:
                        report.write(f"\n[ninja stdout]\n{proc.stdout[:6000]}\n")
                    if proc.stderr:
                        report.write(f"\n[ninja stderr]\n{proc.stderr[:6000]}\n")
                    report.write(f"\n[ninja exit code: {proc.returncode}]\n")
            except Exception as re_exc:
                report.write(f"\n[ninja re-run failed: {re_exc}]\n")
            # Toolchain sanity check (so the user can see what's missing)
            report.write("\n[toolchain probe]\n")
            for tool in ("nvcc", "hipcc", "g++", "gcc", "ld", "ninja"):
                pth = shutil.which(tool)
                report.write(f"  {tool}: {pth or '<NOT FOUND on PATH>'}\n")
                if pth:
                    try:
                        ver = subprocess.run([pth, "--version"],
                                             capture_output=True, text=True,
                                             timeout=10)
                        first_line = (ver.stdout or ver.stderr).splitlines()[:1]
                        if first_line:
                            report.write(f"    -> {first_line[0]}\n")
                    except Exception:
                        pass
            cuda_home = os.environ.get("CUDA_HOME", "")
            if cuda_home:
                report.write(f"\n[CUDA_HOME probe: {cuda_home}]\n")
                for sub in ("", "bin/nvcc", "lib64/libcudart.so",
                            "include/cuda_runtime.h"):
                    p = Path(cuda_home) / sub if sub else Path(cuda_home)
                    report.write(f"  {p}: "
                                 f"{'EXISTS' if p.exists() else '<MISSING>'}\n")
            return None

    elapsed = time.monotonic() - t0
    so = next(build_dir.glob(f"{module_name}*.so"), None)
    report.write(f"\n[build OK in {elapsed:.1f}s] -> {so}\n")
    return so


# ---------------------------------------------------------------------------
# JIT autotune — orchestrates Bayesian / Exhaustive sweep + worker
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stream 10 — per-variant numerical / differential validation
# ---------------------------------------------------------------------------
#
# For each candidate variant we time we also do a forward pass on a fixed
# input and compare the post-step parameter tensor against a known-good
# reference output (captured once from the unoptimised AOT primary
# artefact). Each variant is tagged with one of:
#
#   "ok"                 — output is within tolerance for its dtype
#   "numerical_fail"     — output is outside tolerance
#   "deterministic"      — bit-identical to reference (or 3 re-runs are
#                          bit-identical when --strict-numerics is set)
#   "non_deterministic"  — within-tolerance but not bit-identical under
#                          3x re-run (only set in strict mode)
#   "skipped"            — no reference available (e.g. Pallas), or the
#                          numerical check itself errored
#
# Variants tagged "numerical_fail" are excluded from the winner pool in
# ``pick_winner``. When ``--strict-numerics`` is set, only variants tagged
# "deterministic" are eligible.
#
# Per-dtype tolerances (rtol, atol). Looser as the dtype gets narrower;
# fp4 is the loosest because rounding error is huge. fp32 is tight: any
# kernel that disagrees here is almost certainly wrong.
TOLERANCES: Dict[str, Tuple[float, float]] = {
    "fp32":    (1e-5, 1e-6),
    "float32": (1e-5, 1e-6),
    "fp16":    (1e-3, 1e-4),
    "float16": (1e-3, 1e-4),
    "bf16":    (1e-3, 1e-4),
    "bfloat16":(1e-3, 1e-4),
    "fp8":     (1e-2, 1e-3),
    "fp4":     (5e-2, 1e-2),
}


# Sidecar dict — the variant timer writes a numerical_status entry here
# keyed by config_key(); _make_trial_record reads it back so the existing
# ``timer(cfg) -> result-dict`` signature stays unchanged.
_LAST_NUMERICAL_STATUS: Dict[str, str] = {}


def _capture_reference_output(aot_so_path: Path, opt_class: str,
                              size: int, dtype: str,
                              out_dir: Path) -> Path:
    """Run the AOT optimiser once and save the post-step parameter tensor
    as a .npy file. Cached per (opt, size, dtype) — re-uses an existing
    snapshot if one is already on disk."""
    ref_path = out_dir / f"ref_output_{opt_class}_{size}_{dtype}.npy"
    if ref_path.exists():
        return ref_path
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tiny subprocess: load the AOT .so via torch.ops.load_library, run a
    # single fused-step call, dump the resulting param tensor with numpy.
    # Wrapped in a try/except so a missing torch op falls through to
    # "skipped" upstream rather than crashing the entire autotune.
    script = textwrap.dedent(f"""
        import os, sys, numpy as np
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        import torch
        torch.ops.load_library(r'''{aot_so_path}''')
        torch.manual_seed(0)
        dtype = getattr(torch, '{dtype}')
        param = torch.zeros({size}, dtype=dtype, device='cuda')
        grad  = torch.ones( {size}, dtype=dtype, device='cuda')
        m     = torch.zeros({size}, dtype=dtype, device='cuda')
        v     = torch.zeros({size}, dtype=dtype, device='cuda')
        op = getattr(torch.ops.grokking_optimizers,
                     'fused_{opt_class.lower()}_simple_step', None)
        if op is None:
            sys.stderr.write('no fused op registered for {opt_class}\\n')
            sys.exit(2)
        op(param, grad, m, v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0)
        np.save(r'''{ref_path}''', param.detach().cpu().numpy())
        print('OK')
    """)
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not ref_path.exists():
        raise RuntimeError(
            f"reference capture failed (rc={r.returncode}): "
            f"stderr={r.stderr[-500:]}")
    return ref_path


def _compare_outputs(ref_path: Path, candidate_path: Path,
                     dtype: str) -> Tuple[str, float]:
    """Compare two .npy tensors. Returns (status, max_relative_error).

    status is one of "ok" | "numerical_fail" | "deterministic". The
    "deterministic" tag is set only when the two arrays are bit-identical
    (np.array_equal). Strict-mode 3x re-run determinism is checked
    separately by _check_determinism_3x.
    """
    import numpy as np
    rtol, atol = TOLERANCES.get(dtype, (1e-3, 1e-4))
    a = np.load(ref_path)
    b = np.load(candidate_path)
    if a.shape != b.shape:
        return ("numerical_fail", float("inf"))
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rel = diff / (np.abs(a).astype(np.float64) + atol)
    max_rel = float(rel.max()) if rel.size else 0.0
    if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
        if np.array_equal(a, b):
            return ("deterministic", max_rel)
        return ("ok", max_rel)
    return ("numerical_fail", max_rel)


def _dump_variant_output(variant_so: Path, opt_class: str, size: int,
                         dtype: str, out_path: Path,
                         timeout: int = 120) -> bool:
    """Run the variant .so once and dump its post-step param tensor to
    ``out_path``. Returns True on success."""
    script = textwrap.dedent(f"""
        import os, sys, numpy as np
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        import torch
        torch.ops.load_library(r'''{variant_so}''')
        torch.manual_seed(0)
        dtype = getattr(torch, '{dtype}')
        param = torch.zeros({size}, dtype=dtype, device='cuda')
        grad  = torch.ones( {size}, dtype=dtype, device='cuda')
        m     = torch.zeros({size}, dtype=dtype, device='cuda')
        v     = torch.zeros({size}, dtype=dtype, device='cuda')
        op = getattr(torch.ops.grokking_optimizers,
                     'fused_{opt_class.lower()}_simple_step', None)
        if op is None:
            sys.exit(2)
        op(param, grad, m, v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0)
        np.save(r'''{out_path}''', param.detach().cpu().numpy())
        print('OK')
    """)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and out_path.exists()
    except Exception:
        return False


def _check_determinism_3x(variant_so: Path, opt_class: str, size: int,
                          dtype: str, out_dir: Path) -> bool:
    """Run the variant 3 times; return True iff all 3 outputs are
    bit-identical to each other."""
    import numpy as np
    paths: List[Path] = []
    for i in range(3):
        p = out_dir / f"_det_check_{i}_{opt_class}_{size}_{dtype}.npy"
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
        if not _dump_variant_output(variant_so, opt_class, size, dtype, p):
            return False
        paths.append(p)
    if len(paths) < 2:
        return False
    a = np.load(paths[0])
    for p in paths[1:]:
        if not np.array_equal(a, np.load(p)):
            return False
    return True


def _variant_macros(config: Dict[str, Any], dims: List[Dict[str, Any]],
                    target: str,
                    spec: Optional["BuildSpec"] = None,
                    arch: Optional[str] = None) -> List[str]:
    """Macros + extra-flag overrides for one config × target.

    Stream 3: ``arch`` is threaded through so the NVCC/HIPCC extra
    resolvers can emit per-arch feature macros (TMA, MFMA, fp8/fp4, ...).

    Stream 6 hook: when ``spec.enable_emitter`` is True, the device-side
    call routes through ``grokking_optimizers.codegen.emit_variant_source``
    to render a per-variant source file. The emitted path is stashed on
    ``spec._emitted_sources[config_key]`` so the (Stream 10-owned)
    ``_make_variant_timer`` can swap it into its ``sources`` list without
    a signature change. Residual host-side macros are still appended to
    the cflags so host code that needs them keeps working. On any
    emitter failure we fall through to the legacy macros-only path.
    """
    # Stream 3 — if arch wasn't passed explicitly, lift it off spec.
    if arch is None and spec is not None:
        arch = getattr(spec, "arch", None)
    if spec is not None and getattr(spec, "enable_emitter", False) \
            and target == "device":
        try:
            from grokking_optimizers.codegen import (
                emit_variant_source, CodegenError)
            emitted_path, residual = emit_variant_source(
                config, dims, spec.optimizer, spec.arch, spec.out_dir)
            spec._emitted_sources[config_key(config)] = emitted_path
            macros_only = resolve_macros(config, dims, target)
            return macros_only + residual
        except Exception:
            pass
    macros = resolve_macros(config, dims, target)
    if target != "device":
        return macros
    # Pick the resolver that matches the arch's vendor.
    if arch and arch in ARCH_TABLE:
        vendor = get_arch_entry(arch).vendor
        if vendor == "cuda":
            return macros + resolve_extra_nvcc_flags(config, dims, arch)
        if vendor == "hip":
            return macros + resolve_extra_hipcc_flags(config, dims, arch)
        return macros  # pallas — no native device flags
    # Backward-compat fallback: emit whichever resolver produced a maxrregcount.
    extra = resolve_extra_nvcc_flags(config, dims)
    extra_hip = resolve_extra_hipcc_flags(config, dims)
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
    subprocess).

    For Pallas/TPU specs, returns a Pallas-flavoured closure that skips
    the variant-.so build entirely and just runs PallasTimer on the
    config dict directly (each config is a kwargs bundle for the jitted
    launcher)."""

    # ---- Pallas/TPU path: no .so, no worker — PallasTimer in-process. ----
    if get_arch_entry(spec.arch).vendor == "pallas":
        launcher_path = (REPO_ROOT / "csrc/backends/pallas"
                         / f"launch_{spec.optimizer}.py")
        pallas_timer = PallasTimer(launcher_path, spec.optimizer)

        def pallas_closure(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            ckey = config_key(config)
            progress_state["last_start"] = time.monotonic()
            try:
                result = pallas_timer.time_config(config)
            except RuntimeError as exc:
                report.write(f"    [pallas time error {ckey}: {exc}]\n")
                result = None
            elapsed = time.monotonic() - progress_state["last_start"]
            progress_state["window"].append(elapsed)
            if len(progress_state["window"]) > 20:
                progress_state["window"].pop(0)
            return result

        return pallas_closure

    # Stream 10 — numerical validation context.
    arch_entry = get_arch_entry(spec.arch)
    aot_so = getattr(spec, "aot_so_path", None)
    strict = bool(getattr(spec, "strict_numerics", False))
    # Pallas has no per-variant .so; skip numerical validation entirely.
    numerics_enabled = (arch_entry.vendor != "pallas"
                        and aot_so is not None
                        and Path(aot_so).exists())
    variant_dump_dir = spec.out_dir / "variants"
    variant_dump_dir.mkdir(parents=True, exist_ok=True)
    # The reference output is captured lazily on the first variant so we
    # don't pay for it when the autotune cache hits AOT-only or when the
    # AOT .so doesn't expose the expected fused op.
    ref_state: Dict[str, Any] = {"path": None, "tried": False,
                                 "size": 4096, "dtype": "float32"}

    def _resolve_ref() -> Optional[Path]:
        if not numerics_enabled:
            return None
        if ref_state["path"] is not None:
            return ref_state["path"]
        if ref_state["tried"]:
            return None
        ref_state["tried"] = True
        try:
            p = _capture_reference_output(
                Path(aot_so), OPT_CLASS[spec.optimizer],
                ref_state["size"], ref_state["dtype"], spec.out_dir)
            ref_state["path"] = p
            return p
        except Exception as exc:
            report.write(f"  [numerical] reference capture skipped: {exc}\n")
            return None

    def timer(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ckey = config_key(config)
        host_extra = _variant_macros(config, dims, "host",
                                      spec=spec, arch=spec.arch)
        device_extra = _variant_macros(config, dims, "device",
                                        spec=spec, arch=spec.arch)

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
            _LAST_NUMERICAL_STATUS[ckey] = "skipped"
            return None
        cache.record_variant(spec.optimizer, spec.model, spec.arch,
                             ckey, variant_so)

        # Tell the inline timing script to dump the post-step param tensor
        # so we can numerically compare against the reference. The env var
        # is consumed by both the persistent worker subprocess and the
        # one-shot fallback (they re-exec child Python with child_env()).
        out_dump = variant_dump_dir / f"_out_{_short_key(ckey)}.npy"
        if out_dump.exists():
            try:
                out_dump.unlink()
            except OSError:
                pass
        prior_dump = os.environ.get("SG_DUMP_OUTPUT")
        os.environ["SG_DUMP_OUTPUT"] = str(out_dump)
        try:
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
        finally:
            if prior_dump is None:
                os.environ.pop("SG_DUMP_OUTPUT", None)
            else:
                os.environ["SG_DUMP_OUTPUT"] = prior_dump

        # Update rolling window before we do the numerical work — the
        # numerical phase is sequential and we don't want it polluting
        # the per-variant ETA estimate.
        elapsed = time.monotonic() - progress_state["last_start"]
        progress_state["window"].append(elapsed)
        if len(progress_state["window"]) > 20:
            progress_state["window"].pop(0)

        # ── Numerical validation pass ───────────────────────────────────
        num_status = "skipped"
        if result is not None and numerics_enabled:
            ref_path = _resolve_ref()
            if ref_path is not None and out_dump.exists():
                try:
                    num_status, max_rel = _compare_outputs(
                        ref_path, out_dump, ref_state["dtype"])
                    report.write(
                        f"    [numerical] {ckey[:24]} {num_status} "
                        f"(max_rel={max_rel:.3e})\n")
                    # Strict mode: only "deterministic" trials are eligible
                    # to win, so promote within-tolerance variants by
                    # re-running 3x and checking for bit-identical outputs.
                    if (strict and num_status in ("ok", "deterministic")):
                        det = _check_determinism_3x(
                            variant_so, OPT_CLASS[spec.optimizer],
                            ref_state["size"], ref_state["dtype"],
                            variant_dump_dir)
                        num_status = ("deterministic" if det
                                      else "non_deterministic")
                        report.write(
                            f"    [numerical:strict] {ckey[:24]} "
                            f"3x re-run -> {num_status}\n")
                except Exception as exc:
                    report.write(
                        f"    [numerical] {ckey[:24]} skipped: {exc}\n")
                    num_status = "skipped"
        _LAST_NUMERICAL_STATUS[ckey] = num_status
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
import sys, os, json, importlib.util, traceback
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
    # Stream 10: dump post-step parameter tensor for numerical validation
    # when SG_DUMP_OUTPUT is set. Side-effect only — does NOT change the
    # JSON timing output that the caller parses.
    dump_path = os.environ.get("SG_DUMP_OUTPUT")
    if dump_path:
        try:
            import numpy as _np
            _np.save(dump_path, p.detach().cpu().numpy())
        except Exception as _dexc:
            # Don't let a dump failure mask the timing result — the
            # numerical layer will see the missing file and tag "skipped".
            sys.stderr.write("[dump-output] " + repr(_dexc) + "\n")
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
    # Pallas/TPU has no torch.cuda; route to the Pallas autotune driver,
    # which uses PallasTimer (JAX in-process) instead of variant .so timing.
    if get_arch_entry(spec.arch).vendor == "pallas":
        report.write("  [jit-autotune] pallas backend — using PallasTimer.\n")
        try:
            space = get_search_space(spec.search_space_path)
        except Exception as exc:
            report.write(f"  [jit-autotune] search space load failed: {exc}\n")
            space = {spec.arch: {"dims": [], "prefilter": {"rules": []}}}
        arch_space = space.get(spec.arch, {"dims": [], "prefilter": {"rules": []}})
        if arch_space.get("dims"):
            try:
                configs = list(itertools.islice(
                    cartesian({spec.arch: arch_space}, spec.arch),
                    1_000_000))
            except Exception as exc:
                report.write(f"  [jit-autotune] cartesian failed: {exc}\n")
                configs = []
        else:
            configs = []
        return _pallas_autotune(spec, sources, configs, report, cache)
    if not gpu_ok:
        report.write("  [jit-autotune] no GPU visible — skipping. Run JIT "
                     "phase on the target GPU host with this cache.\n")
        return None

    yaml_path = spec.search_space_path or DEFAULT_SEARCH_SPACE
    report.write(f"  [search-space] {yaml_path}\n")
    space = get_search_space(spec.search_space_path)
    if spec.arch not in space:
        report.write(f"  [jit-autotune] no search space for arch={spec.arch}\n")
        return None
    space_hash = hash_space(space, spec.arch)
    report.write(f"  [search-space] hash={space_hash[:16]}\n")

    if cache.is_jit_fresh(spec.optimizer, spec.model, spec.arch,
                          search_space_hash=space_hash):
        tuned = cache.get(spec.optimizer, spec.model, spec.arch)["tuned_config"]
        report.write(f"  [jit-autotune] cache hit: tuned={tuned}\n")
        return tuned

    # The full programmatic search space is billions of configs per arch.
    # Never materialize the Cartesian product. cartesian_count() gives the
    # total size cheaply for the display line. Bayesian mode uses an inline
    # per-config feasibility predicate (compile_feasibility_check) and never
    # iterates the Cartesian. Exhaustive mode streams survivors with a 1M
    # cap to bound memory.
    total_candidates = cartesian_count(space, spec.arch)
    survivors: List[Dict[str, Any]] = []
    if spec.autotune_mode == "exhaustive":
        exhaustive_cap = 1_000_000
        survivors, eliminated = ss_prefilter(
            cartesian(space, spec.arch),
            space[spec.arch].get("prefilter", {}),
            max_survivors=exhaustive_cap,
        )
        capped = len(survivors) >= exhaustive_cap
        cap_note = f" (capped at {exhaustive_cap:,})" if capped else ""
        report.write(f"  [prefilter] {total_candidates:,} candidates → "
                     f"{len(survivors):,} survivors{cap_note} "
                     f"({eliminated:,} eliminated en route)\n")
        if not survivors:
            report.write("  [jit-autotune] no survivors after prefilter.\n")
            return None
    else:
        # Bayesian: skip materialization entirely. TPE samples per-dim values
        # via Optuna's suggest_categorical and validates each suggestion with
        # the inline feasibility predicate. The reported "survivors" count is
        # an estimate from a small Cartesian slice (best-effort, for UX).
        sample_size = 100_000
        survivors_est = sum(
            1 for c, ok in _iter_prefilter_with_status(
                itertools.islice(cartesian(space, spec.arch), sample_size),
                space[spec.arch].get("prefilter", {})) if ok)
        rate = survivors_est / sample_size if sample_size else 0.0
        est_survivors = int(total_candidates * rate)
        report.write(f"  [prefilter] {total_candidates:,} candidates "
                     f"(~{est_survivors:,} feasible @ {rate*100:.1f}% sampled "
                     f"pass rate); Bayesian TPE samples directly — no "
                     f"materialization\n")

    # Spawn the persistent worker(s). When CUDA_VISIBLE_DEVICES /
    # HIP_VISIBLE_DEVICES enumerates >1 device, fan the sweep across
    # every device via MultiGPUTimingPool. The pool's public surface
    # matches TimingWorker, so the timer closure below is unchanged.
    vendor = get_arch_entry(spec.arch).vendor
    visible = MultiGPUTimingPool.visible_devices(vendor)
    worker: Optional[Any] = None
    if len(visible) > 1:
        report.write(f"  [worker] {len(visible)} GPUs visible "
                     f"({','.join(visible)}); spawning MultiGPUTimingPool.\n")
        pool = MultiGPUTimingPool(OPT_CLASS[spec.optimizer], vendor=vendor)
        if pool.start():
            worker = pool
            report.write(f"  [worker] multi-GPU pool up with "
                         f"{len(pool.workers)} worker(s).\n")
        else:
            report.write("  [worker] multi-GPU pool failed to start; "
                         "falling back to single worker.\n")
    if worker is None:
        single = TimingWorker(opt_class=OPT_CLASS[spec.optimizer])
        if not single.start():
            report.write("  [worker] start FAILED; falling back to "
                         "one-shot per variant.\n")
            worker = None
        else:
            report.write("  [worker] persistent timing worker is up.\n")
            worker = single

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
            try:
                worker.stop()
            except Exception:
                pass

    # Auto-prune the variant cache so a long-running autotune campaign
    # doesn't accumulate gigabytes of stale .so files. Only runs on a
    # successful sweep; opt out with spec.prune_after_autotune=False.
    if winning is not None and getattr(spec, "prune_after_autotune", True):
        try:
            summary = cache.prune(
                max_age_days=spec.prune_max_age_days,
                keep_top_n=spec.prune_keep_top_n,
            )
            report.write(
                f"  [auto-prune] dropped={summary['dropped']} "
                f"kept={summary['kept']} "
                f"freed={summary['bytes_freed']/1e6:.2f} MB "
                f"(max_age_days={summary['max_age_days']}, "
                f"keep_top_n={summary['keep_top_n']})\n")
        except Exception as exc:  # noqa: BLE001 — never let GC fail the build
            report.write(f"  [auto-prune] skipped: {exc}\n")

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
            ckey = config_key(cfg)
            report.write(f"\n  [{i}/{len(configs)}] {ckey}\n")
            report.flush()
            t0 = time.monotonic()
            result = timer(cfg)
            elapsed = time.monotonic() - t0
            # Stream 10: per-variant numerical-validation tag is stashed
            # by the variant timer under config_key(cfg).
            num_status = _LAST_NUMERICAL_STATUS.get(ckey, "skipped")
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
                "numerical_status": num_status,
                "recorded_at": datetime.datetime.now().isoformat(),
                "status":      "ok" if result else "fail",
                "build_s":     elapsed,
            }
            cache.record_trial(spec.optimizer, spec.model, spec.arch, trial)
            if result is not None:
                ms = result["timing_ms"]
                # Stream 10 — exclude numerically failing variants from
                # the running best; in strict mode require deterministic.
                if num_status == "numerical_fail":
                    report.write(f"    median={ms:.4f}ms  "
                                 f"[EXCLUDED: numerical_fail]\n")
                elif spec.strict_numerics and num_status != "deterministic":
                    report.write(f"    median={ms:.4f}ms  "
                                 f"[EXCLUDED: strict requires deterministic, "
                                 f"got {num_status}]\n")
                else:
                    report.write(f"    median={ms:.4f}ms ({num_status})\n")
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
    n_trials = spec.bayesian_trials  # may be None ⇒ auto stopper
    # Build a stopper from BuildSpec; honours --max-tune-seconds / --patience
    # / --min-improvement when provided, otherwise pure auto-mode.
    stopper = BayesianEarlyStopper(
        min_delta_rel=spec.min_improvement,
        patience=spec.patience,
        max_seconds=spec.max_tune_seconds,
    )
    budget_desc = (f"n_trials={n_trials} (manual cap)"
                   if n_trials is not None
                   else "auto early-stop")
    if spec.max_tune_seconds is not None:
        budget_desc += f", max_tune_seconds={spec.max_tune_seconds}"
    if spec.patience is not None:
        budget_desc += f", patience={spec.patience}"
    budget_desc += f", min_improvement={spec.min_improvement}"
    report.write(f"\n  [bayesian] TPE stage with {budget_desc}, "
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
    # Progress bar denominator: explicit cap if set, else the stopper's
    # hard ceiling (the bar will simply move slowly; it's a UX hint).
    progress_total = n_trials if n_trials is not None else 1000
    step1, close1 = make_progress(
        progress_total, f"jit-tpe {spec.optimizer}/{spec.arch}")

    def progress1(done, total, cfg):
        step1(f"trial {done}/{total} key={config_key(cfg)[:24]}…")

    try:
        tpe_trials, stop_info = run_bayesian(
            spec.arch, space, n_trials=n_trials, seed=spec.seed,
            storage=storage,
            study_name=f"sg_{spec.optimizer}_{spec.model}_{spec.arch}",
            timer=timer, progress=progress1, host=_current_host(),
            prefiltered=prefiltered,
            pruner=spec.pruner,
            seed_trials=seed_trials,
            stopper=stopper,
        )
    finally:
        close1()

    for t in tpe_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
    cache.save()

    n_ok = sum(1 for t in tpe_trials if t['timing_ms'] is not None)
    report.write(f"  [bayesian] TPE produced {len(tpe_trials)} trials; "
                 f"{n_ok} succeeded. "
                 f"stop_reason={stop_info.get('stop_reason')}\n")

    # Stage 2: refine the top-K with ±2-step neighbours.
    # spec.top_k=None ⇒ topk_refine uses elbow detection.
    refine_inputs = [t for t in tpe_trials if t["timing_ms"] is not None]
    refine_inputs.sort(key=lambda t: t["timing_ms"])
    # Pre-compute effective top_k for progress estimation + reporting.
    effective_top_k = (spec.top_k if spec.top_k is not None
                       else _detect_topk_elbow(refine_inputs))
    report.write(
        f"\n  [bayesian] refine stage top_k={effective_top_k}"
        f"{' (elbow-detected)' if spec.top_k is None else ''}\n")
    feasibility_check = compile_feasibility_check(
        space[spec.arch].get("prefilter", {}))
    n_refine_est = sum(1 for _ in _neighbour_estimate(
        refine_inputs[:effective_top_k], dims, feasibility_check))
    step2, close2 = make_progress(
        max(n_refine_est, 1), f"jit-refine {spec.optimizer}/{spec.arch}")

    def progress2(done, total, cfg):
        step2(f"refine {done}/{total} key={config_key(cfg)[:24]}…")

    try:
        refine_trials = topk_refine(
            tpe_trials, space, spec.arch,
            top_k=effective_top_k, timer=timer,
            progress=progress2, host=_current_host(),
            prefiltered=prefiltered,
        )
    finally:
        close2()
    for t in refine_trials:
        cache.record_trial(spec.optimizer, spec.model, spec.arch, t)
    cache.save()

    winner = pick_winner(tpe_trials + refine_trials,
                         strict_numerics=spec.strict_numerics)
    if winner is None:
        # Note: pick_winner may have returned None purely because no
        # surviving trial passed numerical validation (or, in strict
        # mode, no trial was bit-identical-deterministic across the
        # 3x re-run). Report the breakdown so the user can lower
        # --strict-numerics or widen the search space.
        finished = [t for t in (tpe_trials + refine_trials)
                    if t.get("timing_ms") is not None]
        ns_counts: Dict[str, int] = {}
        for t in finished:
            s = t.get("numerical_status", "skipped")
            ns_counts[s] = ns_counts.get(s, 0) + 1
        report.write(
            f"\n  [bayesian] no successful trials — "
            f"leaving tuned_config unset.\n"
            f"  [bayesian] numerical_status breakdown of "
            f"timed trials: {ns_counts} "
            f"(strict_numerics={spec.strict_numerics})\n")
        return None
    out = dict(winner["config"])
    out["timing_ms"] = winner["timing_ms"]
    out["config_key"] = config_key(out)
    out["stage_won"] = winner["stage"]
    report.write(f"\n  [bayesian] WINNER ({winner['stage']}): "
                 f"{out['config_key']} @ {out['timing_ms']:.4f}ms\n")
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, out,
                    mode="bayesian", search_space_hash=space_hash,
                    early_stop_info=stop_info)
    return out


# ---------------------------------------------------------------------------
# Pallas/TPU autotune driver — Python-only, no .so build, JSON manifest output
# ---------------------------------------------------------------------------
#
# Mirrors _run_bayesian's contract but skips the variant-build step entirely:
# each "config" is just a kwargs dict that PallasTimer feeds into the jitted
# launcher. The winning config is persisted as a JSON manifest under
# ``out_dir / tuned_pallas_<opt>_<model>_<arch>.json`` so the runtime side
# (or another host) can re-import it without re-tuning.
#
# Cache key tuple: (optimizer, model, tpu_arch, block_spec, jax_version,
# libtpu_version). The same compile-cache machinery is used; the block_spec
# and version pieces are derived best-effort and folded into the search-space
# hash so changes invalidate.

def _pallas_versions() -> Dict[str, str]:
    """Best-effort probe of the local jax + libtpu install. Used as part of
    the Pallas autotune cache key — different JAX/libtpu pairings can produce
    different optimal kwargs."""
    out: Dict[str, str] = {}
    try:
        import jax
        out["jax"] = getattr(jax, "__version__", "unknown")
    except ImportError:
        out["jax"] = "absent"
    try:
        import libtpu  # type: ignore
        out["libtpu"] = getattr(libtpu, "__version__", "unknown")
    except ImportError:
        out["libtpu"] = "absent"
    return out


def _pallas_autotune(spec: BuildSpec, sources: List[Path],
                     configs: List[Dict[str, Any]], report,
                     cache: CompileCache) -> Optional[Dict[str, Any]]:
    """Pallas analog of _run_bayesian. Times every config via PallasTimer
    (or, for empty/tiny spaces, just times the no-kwargs default once) and
    writes the winner as a JSON manifest. Returns the winning config dict
    or ``None`` if every trial failed."""
    versions = _pallas_versions()
    block_spec = "default"   # placeholder for future Pallas BlockSpec tuning
    report.write(f"\n  [pallas-autotune] jax={versions['jax']} "
                 f"libtpu={versions['libtpu']} block_spec={block_spec}\n")

    launcher_path = (REPO_ROOT / "csrc/backends/pallas"
                     / f"launch_{spec.optimizer}.py")
    if not launcher_path.is_file():
        report.write(f"  [pallas-autotune] launcher missing: {launcher_path}\n")
        return None

    # Materialize the config list (Pallas search spaces are small — < 1M).
    if not configs:
        # No dimensions in the Pallas space yet (Stream 2 will fill them in).
        # Time the zero-kwargs default so we still produce a winner and a
        # manifest with median timing.
        configs = [{}]
    report.write(f"  [pallas-autotune] candidates: {len(configs)}\n")

    # Build the per-trial timer.
    progress_state: Dict[str, Any] = {"last_start": time.monotonic(), "window": []}
    timer = _make_variant_timer(
        spec, sources, [], [], [], [], cache, None, report, progress_state)

    step, close = make_progress(
        len(configs), f"pallas-autotune {spec.optimizer}/{spec.arch}")

    all_trials: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    try:
        for i, cfg in enumerate(configs, 1):
            ckey = config_key(cfg)
            report.write(f"  [{i}/{len(configs)}] {ckey or '<default>'}\n")
            report.flush()
            result = timer(cfg)
            trial = {
                "trial_num":   i,
                "stage":       "pallas",
                "config":      cfg,
                "config_key":  ckey,
                "timing_ms":   result["timing_ms"] if result else None,
                "min_ms":      result["min_ms"]    if result else None,
                "max_ms":      result["max_ms"]    if result else None,
                "n":           result["n"]         if result else None,
                "host":        _current_host(),
                "recorded_at": datetime.datetime.now().isoformat(),
                "status":      "ok" if result else "fail",
            }
            all_trials.append(trial)
            cache.record_trial(spec.optimizer, spec.model, spec.arch, trial)
            if result is not None and (best is None
                                       or result["timing_ms"] < best["timing_ms"]):
                best = {"config": cfg, "timing_ms": result["timing_ms"],
                        "trial_num": i}
            step(f"trial {i}/{len(configs)} key={ckey[:24] or '<default>'}…")
            if i % JIT_CACHE_FLUSH_EVERY == 0:
                cache.save()
    finally:
        close()
        cache.save()

    if best is None:
        report.write("\n  [pallas-autotune] no successful trials.\n")
        return None

    winner = dict(best["config"])
    winner["timing_ms"] = best["timing_ms"]
    winner["config_key"] = config_key(best["config"])
    winner["stage_won"] = "pallas"

    # Persist the JSON manifest the runtime will consume.
    manifest_path = (spec.out_dir
                     / f"tuned_pallas_{spec.optimizer}_{spec.model}_{spec.arch}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "optimizer":  spec.optimizer,
        "model":      spec.model,
        "tpu_arch":   spec.arch,
        "block_spec": block_spec,
        "jax_version":    versions["jax"],
        "libtpu_version": versions["libtpu"],
        "launcher":   str(launcher_path),
        "tuned_kwargs": {k: v for k, v in winner.items()
                         if k not in ("timing_ms", "config_key", "stage_won")},
        "timing_ms":  winner["timing_ms"],
        "n_trials":   len(all_trials),
        "n_ok":       sum(1 for t in all_trials if t["status"] == "ok"),
        "host":       _current_host(),
        "recorded_at": datetime.datetime.now().isoformat(),
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report.write(f"  [pallas-autotune] manifest: {manifest_path}\n")
    except OSError as exc:
        report.write(f"  [pallas-autotune] manifest write failed: {exc}\n")

    # Roll the same fact into the compile cache for downstream consumers.
    space_key = _sha256_str(json.dumps(
        {"opt": spec.optimizer, "model": spec.model, "arch": spec.arch,
         "block_spec": block_spec, **versions}, sort_keys=True))
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, winner,
                    mode="pallas", search_space_hash=space_key)
    report.write(f"  [pallas-autotune] WINNER: {winner['config_key'] or '<default>'} "
                 f"@ {winner['timing_ms']:.4f}ms\n")
    return winner


def _neighbour_estimate(seeds: List[Dict[str, Any]],
                        dims: List[Dict[str, Any]],
                        is_feasible: Callable[[Dict[str, Any]], bool]):
    """Estimate the refine-stage trial count (best-effort, for progress bar)."""
    seen = set()
    for s in seeds:
        base = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in s["config"].items()}
        for d in dims:
            for nb in _step_neighbours(base.get(d["name"]), d["values"], 2):
                cfg = dict(base)
                cfg[d["name"]] = nb
                k = config_key(cfg)
                if k in seen or not is_feasible(cfg):
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
        space = load_embedded_search_space()
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
    _ensure_nvcc_on_path()
    _refresh_torch_cuda_home()
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "pallas":
        # TPU/Pallas has no AOT phase — no nvcc, no .so, no cpp_extension.
        # Return a sentinel Path so downstream consumers (_publish_aot_artifact,
        # build_jit) can detect the no-op without crashing on a missing file.
        launcher = (REPO_ROOT / "csrc/backends/pallas"
                    / f"launch_{spec.optimizer}.py")
        report.write("\n[build_aot] no-op for Pallas (TPU has no AOT phase)\n")
        report.write(f"  launcher: {launcher}\n")
        report.write(f"  exists:   {launcher.exists()}\n")
        return Path("pallas-noop")

    sources = _resolve_sources(spec)
    host_cflags = _host_cflags(spec)
    device_cflags = _device_cflags(spec)
    ldflags = _ldflags(spec)
    source_hash = _hash_sources(sources) if sources else "pallas"
    host_hash = _hash_flags(host_cflags)
    device_hash = _hash_flags(device_cflags)

    # Resolve search-space hash (gates AOT freshness too)
    space_hash = None
    try:
        space_hash = hash_space(get_search_space(spec.search_space_path), spec.arch)
    except Exception as exc:
        report.write(f"  [search-space] could not hash: {exc}\n")

    workload_hash = None
    if spec.pgo:
        workload = spec.pgo_workload or DEFAULT_PGO_WORKLOAD
        workload_hash = hash_workload(workload, spec.pgo_steps)
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
    inst_host, inst_device, inst_ld = instrument_flags(
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
    ok = collect_workload(
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
    use_host, use_device, use_ld = use_flags(
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

    # ── Stream 8 hook: device-side PGO (CUPTI / rocprof / XLA HLO) ────
    # NVCC strips LLVM PGO from device code, so the 3-pass loop above only
    # optimizes the host launchers. This hook complements it with vendor-
    # specific stall sampling whose output (a JSON sidecar) the Bayesian
    # autotuner can use to enqueue biased trials. The hook is a no-op
    # unless ``spec.enable_device_pgo`` is True.
    if getattr(spec, "enable_device_pgo", False):
        try:
            from grokking_optimizers.device_profiling import (
                run_device_pgo_round,
            )
            device_workload_cmd = [
                sys.executable, str(workload),
                "--so",    str(so_path),
                "--opt",   OPT_CLASS[spec.optimizer],
                "--model", spec.model,
                "--arch",  spec.arch,
                "--steps", str(int(spec.pgo_steps)),
            ]
            run_device_pgo_round(
                spec, device_workload_cmd, spec.out_dir, report)
        except ImportError:
            pass

    return so_path


def _publish_aot_artifact(spec: BuildSpec, so_path: Path, report) -> Path:
    """If ``--aot-artifact-dir`` is set, copy the .so into the shared dir
    so a downstream JIT runtime (possibly on another host) can pick it up.
    Returns the published path (or the original)."""
    # Pallas sentinel — nothing to publish.
    if str(so_path) == "pallas-noop":
        return so_path
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
    _ensure_nvcc_on_path()
    _refresh_torch_cuda_home()
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "pallas":
        # Pallas/TPU JIT phase = PallasTimer-driven autotune. No .so output;
        # winning kwargs are persisted as a JSON manifest under spec.out_dir.
        report.write("\n[build_jit] pallas path — routing to _pallas_autotune\n")
        # Sources are conceptual here (a single .py launcher); pass an empty
        # list — _pallas_autotune locates the launcher itself.
        try:
            space = build_full_search_space()
        except Exception as exc:
            report.write(f"  [pallas] build_full_search_space failed: {exc}\n")
            space = {spec.arch: {"dims": [], "prefilter": {"rules": []}}}
        arch_space = space.get(spec.arch, {"dims": [], "prefilter": {"rules": []}})
        if arch_space.get("dims"):
            try:
                configs = list(itertools.islice(
                    cartesian({spec.arch: arch_space}, spec.arch),
                    1_000_000))
            except Exception as exc:
                report.write(f"  [pallas] cartesian failed: {exc}\n")
                configs = []
        else:
            configs = []
        winner = _pallas_autotune(spec, [], configs, report, cache)
        cache.save()
        manifest_path = (spec.out_dir /
                         f"tuned_pallas_{spec.optimizer}_{spec.model}_{spec.arch}.json")
        if winner is None:
            report.write("  [pallas] no winner produced.\n")
            return None
        return manifest_path if manifest_path.is_file() else Path("pallas-noop")

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
        space = get_search_space(spec.search_space_path)
        dims = space.get(spec.arch, {}).get("dims", [])
        extra_host = _variant_macros(tuned, dims, "host",
                                      spec=spec, arch=spec.arch)
        extra_device = _variant_macros(tuned, dims, "device",
                                        spec=spec, arch=spec.arch)
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
    bayesian_trials: Optional[int] = None,
    top_k: Optional[int] = None,
    max_tune_seconds: Optional[float] = None,
    min_improvement: float = 0.005,
    patience: Optional[int] = None,
    seed: int = 0,
    debug_symbols: bool = False,
    debug: bool = False,
    bootstrap_cuda: bool = False,
    bootstrap_rocm: bool = False,
    bootstrap_jax: bool = False,
    pruner: str = "none",
    transfer_learning: bool = False,
    enable_runtime_specialization: bool = False,
    config: Optional[Any] = None,
    enable_emitter: bool = False,
    enable_device_pgo: bool = False,
    prune_after_autotune: bool = True,
    prune_max_age_days: int = 30,
    prune_keep_top_n: int = 100,
    strict_numerics: bool = False,
) -> Optional[Path]:
    """In-process orchestrator. ``main`` handles subprocess split.

    When called from Python, this does not fork: AOT and JIT run in the
    same process. Use ``main(['--runtime', 'both', ...])`` for the
    subprocess-isolated workflow.

    ``debug=True`` mirrors every line of the build report to stderr in
    real time, prints every spawned subprocess and ninja invocation
    (via verbose=True), and emits per-phase banners with timestamps."""
    if aot_only:
        runtime = "aot"
    if jit_only:
        runtime = "jit"

    # Discover nvcc on the filesystem if it's not on PATH (common when
    # CUDA lives at /usr/local/cuda-<ver>/, in a pip-installed nvidia
    # wheel, or under the bare /usr/local/cuda/bin the user added).
    _ensure_nvcc_on_path()
    # On-demand: install the right toolchain if missing and the caller
    # opted in. Vendor dispatch ensures HIP archs get hipcc (not nvcc) and
    # Pallas archs get jax[tpu] (not nvcc). Handles Colab CPU runtimes,
    # fresh CI, mixed-vendor sweeps, etc.
    _vendor = get_arch_entry(arch).vendor
    if bootstrap_cuda and _vendor == "cuda":
        if not shutil.which("nvcc"):
            bootstrap_cuda_toolkit(arch=arch)
    elif bootstrap_rocm and _vendor == "hip":
        if not shutil.which("hipcc"):
            bootstrap_rocm_toolkit(arch=arch)
    elif bootstrap_jax and _vendor == "pallas":
        bootstrap_jax_tpu(arch=arch)
    # Force torch to re-read CUDA_HOME / ROCM_HOME from os.environ even if
    # it was imported (and cached None) before the user set the env vars.
    _refresh_torch_cuda_home()

    # Hard pre-flight: if we're targeting a C++-compiled arch and the
    # toolchain is genuinely missing, fail loudly NOW with the install
    # recipe instead of letting ninja produce a confusing exit-127 error.
    if get_arch_entry(arch).vendor == "cuda" and not shutil.which("nvcc"):
        raise RuntimeError(
            "nvcc not found on PATH and could not be auto-discovered.\n"
            "\n"
            "AUTO-FIX: call build(..., bootstrap_cuda=True). This probes\n"
            "conda / apt / dnf / yum / zypper / pacman / apk / brew / winget\n"
            "in priority order and installs nvcc via whichever is available.\n"
            "\n"
            "MANUAL OPTIONS — pick the one that matches your environment:\n"
            "  conda install -c nvidia cuda-nvcc cuda-runtime    # any OS, conda env\n"
            "  sudo apt-get install nvidia-cuda-toolkit          # Debian / Ubuntu / Colab\n"
            "  sudo dnf install cuda                             # Fedora / RHEL 8+ (NVIDIA repo)\n"
            "  sudo yum install cuda                             # RHEL / CentOS 7\n"
            "  sudo zypper install cuda                          # openSUSE / SLES\n"
            "  sudo pacman -S cuda                               # Arch / Manjaro\n"
            "  sudo apk add cuda                                 # Alpine Linux\n"
            "  winget install Nvidia.CUDA                        # Windows 10+\n"
            "  https://developer.nvidia.com/cuda-downloads       # official .run installer\n"
            "\n"
            "NOTE: NVIDIA's PyPI wheels (nvidia-cuda-nvcc-cuXX) ship ptxas,\n"
            "libnvvm, libcudart, and headers — but NOT the nvcc compiler\n"
            "driver. Use one of the above to get nvcc.\n"
            "\n"
            "Re-run with debug=True to see the [preflight] / [CUDA_HOME probe]\n"
            "blocks that show exactly what compile.py searched."
        )
    if get_arch_entry(arch).vendor == "hip" and not shutil.which("hipcc"):
        raise RuntimeError(
            "hipcc not found on PATH. ROCm install — pick the one matching\n"
            "your environment:\n"
            "  sudo apt-get install rocm-hip-sdk                 # Debian / Ubuntu\n"
            "  sudo dnf install rocm-hip-devel                   # Fedora / RHEL\n"
            "  sudo zypper install rocm-hip                      # openSUSE\n"
            "  sudo pacman -S rocm-hip-sdk                       # Arch\n"
            "  https://rocm.docs.amd.com/projects/install-on-linux/\n"
            "Then set os.environ['ROCM_PATH'] = '/opt/rocm' (or wherever it\n"
            "lives) and prepend $ROCM_PATH/bin to PATH."
        )

    if debug:
        verbose = True

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
        bayesian_trials=bayesian_trials, top_k=top_k,
        max_tune_seconds=max_tune_seconds,
        min_improvement=min_improvement,
        patience=patience,
        seed=seed,
        debug_symbols=debug_symbols, debug=debug,
        pruner=pruner, transfer_learning=transfer_learning,
        enable_runtime_specialization=enable_runtime_specialization,
        enable_emitter=enable_emitter,
        enable_device_pgo=enable_device_pgo,
        prune_after_autotune=prune_after_autotune,
        prune_max_age_days=prune_max_age_days,
        prune_keep_top_n=prune_keep_top_n,
        strict_numerics=strict_numerics,
    )

    # Stream 11: optionally load project config and apply to spec. Strictly
    # backward-compatible — if no override is present in the config, the
    # spec is left untouched.
    if config is None or not isinstance(config, dict):
        try:
            from grokking_optimizers.compile_config import (
                load_config as _load_cfg,
                apply_to_buildspec as _apply_cfg,
            )
            _cfg_arg = config if isinstance(config, (str, Path)) else None
            project_cfg = _load_cfg(Path(_cfg_arg) if _cfg_arg else None)
        except Exception:
            project_cfg = {}
    else:
        project_cfg = config
    try:
        from grokking_optimizers.compile_config import apply_to_buildspec as _apply_cfg2
        _apply_cfg2(spec, project_cfg)
    except Exception:
        pass

    _validate(spec)
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if cache is None:
        cp = Path(cache_path) if cache_path else (spec.out_dir / DEFAULT_CACHE_NAME)
        cache = CompileCache(cp)

    report_path = report_path or (
        spec.out_dir / f"compile_{optimizer}_{model}_{arch}.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    entry = get_arch_entry(arch)
    phases = ["resolve"]
    if entry.vendor != "pallas":
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

    if debug:
        bar = "=" * 72
        _ts = datetime.datetime.now().isoformat()
        _cuda = os.environ.get("CUDA_HOME", "<unset>")
        _rocm = os.environ.get("ROCM_PATH", "<unset>")
        _path = os.environ.get("PATH", "")[:200]
        _fc = os.environ.get("FORCE_CUDA", "<unset>")
        _tcal = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>")
        sys.stderr.write(
            f"\n{bar}\n"
            f"[debug] grokking_optimizers.compile starting at {_ts}\n"
            f"[debug] target:   {optimizer}/{model}/{arch} (vendor={entry.vendor})\n"
            f"[debug] runtime:  {runtime}  autotune={autotune} ({autotune_mode})  "
            f"pgo={pgo}  profile={profile}\n"
            f"[debug] phases:   {phases}\n"
            f"[debug] out_dir:  {spec.out_dir}\n"
            f"[debug] cache:    {cache.path}\n"
            f"[debug] report:   {report_path}\n"
            f"[debug] env:      CUDA_HOME={_cuda}  ROCM_PATH={_rocm}\n"
            f"[debug] env:      PATH={_path}...\n"
            f"[debug] env:      FORCE_CUDA={_fc}  TORCH_CUDA_ARCH_LIST={_tcal}\n"
            f"{bar}\n\n"
        )
        sys.stderr.flush()

    try:
        with open(report_path, "w") as report_file:
            report = _DebugTee(report_file, sys.stderr) if debug else report_file
            report.write("# grokking_optimizers.compile — targeted build\n")
            report.write(f"# Generated:        {datetime.datetime.now().isoformat()}\n")
            report.write(f"# Optimizer:        {optimizer}\n")
            report.write(f"# Model:            {model}\n")
            report.write(f"# Arch:             {arch} (vendor={entry.vendor})\n")
            report.write(f"# CPU cores:        {NCPUS} (ninja -j)\n")
            report.write(f"# Out dir:          {spec.out_dir}\n")
            report.write(f"# Cache:            {cache.path}\n")
            report.write(f"# Runtime:          {runtime}\n")
            report.write(f"# Autotune:         {autotune} (mode={autotune_mode})\n")
            bt_disp = bayesian_trials if bayesian_trials is not None else "auto"
            tk_disp = top_k if top_k is not None else "auto"
            report.write(f"# Bayesian trials:  {bt_disp} (top_k={tk_disp}"
                         f" patience={patience or 'auto'}"
                         f" max_tune_seconds={max_tune_seconds or 'unbounded'}"
                         f" min_improvement={min_improvement})\n")
            report.write(f"# PGO:              {pgo}\n")
            if pgo:
                report.write(f"#   workload:       {pgo_workload or DEFAULT_PGO_WORKLOAD}\n")
                report.write(f"#   steps:          {pgo_steps}\n")
            report.write(f"# Search space:     {search_space_path or DEFAULT_SEARCH_SPACE}\n")
            report.write(f"# AOT artefact dir: {aot_artifact_dir}\n")
            report.write(f"# Debug symbols:    {debug_symbols}\n")
            report.write(f"# Profile:          {profile}\n")

            # Pre-flight: dump toolchain visibility so the user sees what
            # the build can actually find BEFORE we waste 4 phases.
            report.write("\n")
            for line in _preflight_toolchain(arch):
                report.write(line + "\n")
            step("resolve")

            if entry.vendor == "pallas":
                build_aot(spec, cache, report)  # logs pallas no-op
            else:
                if runtime in ("aot", "both"):
                    report.write("\n--- AOT PHASE ---\n")
                    so_path = build_aot(spec, cache, report)
                    # Stream 10: stash the AOT primary artefact path on
                    # the spec so the variant timer can use it as the
                    # numerical-validation reference.
                    if so_path is not None:
                        spec.aot_so_path = so_path
                    step("aot")
                if runtime in ("jit", "both") and autotune:
                    report.write("\n--- JIT AUTOTUNE PHASE ---\n")
                    # If we entered jit-only without a fresh AOT this run,
                    # try to surface the cached AOT primary artefact for
                    # the numerical-validation reference.
                    if spec.aot_so_path is None:
                        try:
                            _e = cache.get(spec.optimizer, spec.model,
                                           spec.arch)
                            _art = _e.get("primary_artifact") if _e else None
                            if _art and _art.get("path"):
                                spec.aot_so_path = Path(_art["path"])
                        except Exception:
                            pass
                    so_path = build_jit(spec, cache, report) or so_path
                    step("jit-autotune")
                if runtime in ("jit", "both"):
                    if "final" in phases:
                        step("final")

            if profile and runtime != "aot":
                report.write("\n--- PROFILE PASS ---\n")
                _dispatch_profile(optimizer, model, arch, report)
                step("profile")

            # Stream 7 — NVRTC / hipRTC kernel-registry pre-warm.
            if enable_runtime_specialization:
                from grokking_optimizers.kernel_registry import initialize_registry
                initialize_registry(spec, report)

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
    if "--self-test" in (argv if argv is not None else sys.argv[1:]):
        return _self_test()

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
    parser.add_argument("--bayesian-trials", type=int, default=None,
                        help="Manual trial-count override; default = "
                             "multi-criterion auto early-stop "
                             "(plateau + coverage saturation + wall-clock).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-K winners refined in the second stage "
                             "(default: auto elbow detection).")
    parser.add_argument("--max-tune-seconds", type=float, default=None,
                        help="Wall-clock budget for auto autotune (seconds). "
                             "Stops the TPE loop when exceeded.")
    parser.add_argument("--min-improvement", type=float, default=0.005,
                        help="Min relative improvement to reset the plateau "
                             "counter (default: 0.005 = 0.5%%).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Trial patience for plateau detection "
                             "(default: auto = max(50, 0.1 * n_completed)).")
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
                        default=None,
                        help="YAML search-space file "
                             "(default: full programmatic space from "
                             "build_full_search_space() — billions of "
                             "candidates per arch).")
    parser.add_argument("--pgo", action="store_true",
                        help="Enable 3-pass PGO loop (instrument → "
                             "workload → use). Doubles AOT compile time.")
    parser.add_argument("--pgo-workload", type=Path,
                        default=DEFAULT_PGO_WORKLOAD,
                        help="PGO workload script.")
    parser.add_argument("--pgo-steps", type=int, default=1000,
                        help="N optimizer.step() calls during profile "
                             "collection (default: 1000).")
    # Stream 8 — device-side PGO (complements LLVM PGO which nvcc strips
    # from device code). Collects CUPTI / rocprof / XLA stall info into a
    # JSON sidecar consumed by the Bayesian autotuner.
    parser.add_argument("--enable-device-pgo", action="store_true",
                        help="Collect CUPTI / rocprof / XLA stall info as a "
                             "PGO sidecar after the standard 3-pass loop. "
                             "No-op when --pgo is off or the profiler tool "
                             "(nsys / rocprof) is unavailable.")

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
    parser.add_argument("--debug", action="store_true",
                        help="Mirror the full build report to stderr in real "
                             "time, force --verbose, print every spawned "
                             "subprocess + every nvcc/g++/hipcc invocation, "
                             "and emit per-phase banners with timestamps.")
    parser.add_argument("--bootstrap-cuda", action="store_true",
                        help="If nvcc isn't found after probing the usual "
                             "locations, pip-install the NVIDIA CUDA toolkit "
                             "wheels (nvcc + runtime + CCCL + NVRTC) and "
                             "retry. Use this on Colab CPU runtimes or any "
                             "fresh host without a system CUDA install.")
    # Stream 7 — runtime kernel specialization via NVRTC / hipRTC.
    parser.add_argument("--enable-runtime-specialization", action="store_true",
                        help="Pre-warm a per-arch KernelRegistry that JIT-"
                             "specializes hot kernels via NVRTC (CUDA) / "
                             "hipRTC (HIP) with shape-class constants baked "
                             "in. CUBINs are cached under "
                             "<out>/nvrtc_cache. CPU-only hosts without "
                             "cuda-python degrade gracefully — the build "
                             "still succeeds.")
    # Stream 11 — TOML project config
    parser.add_argument("--config", default=None,
                        help="Path to project config TOML "
                             "(default: ./compile_config.toml or packaged default)")
    # Stream 6 — kernel emission backend
    parser.add_argument("--enable-emitter", action="store_true",
                        help="Route each variant through "
                             "grokking_optimizers.codegen.emit_variant_source "
                             "so the autotuner compiles a freshly rendered "
                             "Jinja2 template per config instead of "
                             "re-compiling one fixed source with -D macros. "
                             "Falls back to macros-only on any emitter error.")
    # Stream 12 — toolchain bootstrap for HIP + Pallas archs.
    parser.add_argument("--bootstrap-rocm", action="store_true",
                        help="Install the ROCm toolchain (hipcc + rocm-dev) "
                             "if missing. HIP archs (gfx*) only — picks the "
                             "ROCm version per ARCH_TABLE[arch] (e.g. gfx950 "
                             "→ ROCm 6.2+, gfx1200 → 7.0+).")
    parser.add_argument("--bootstrap-jax", action="store_true",
                        help="Install jax[tpu] from the libtpu_releases "
                             "bucket if no TPU device is visible. Pallas/TPU "
                             "archs (tpu_v*) only.")

    # ── Variant-cache GC (Stream 9) ─────────────────────────────────
    parser.add_argument("--prune", action="store_true",
                        help="Prune the variant cache and exit (no build). "
                             "Honours --prune-max-age-days / --prune-keep-top-n.")
    parser.add_argument("--prune-max-age-days", type=int, default=30,
                        help="Variants older than N days are dropped "
                             "(default: 30).")
    parser.add_argument("--prune-keep-top-n", type=int, default=100,
                        help="Per (opt, model, arch) keep only the top-N "
                             "fastest variants (default: 100).")
    parser.add_argument("--no-auto-prune", action="store_true",
                        help="Skip the auto-prune step at the end of a "
                             "successful JIT autotune pass.")
    # Stream 10 — numerical / differential validation
    parser.add_argument("--strict-numerics", action="store_true",
                        help="Require bit-identical determinism for the "
                             "winning variant. Variants tagged "
                             "numerical_fail are always excluded; with this "
                             "flag, only variants that produce the same "
                             "output as the AOT reference AND are "
                             "bit-identical across a 3x re-run "
                             "(tag=deterministic) are eligible.")
    args = parser.parse_args(argv)

    # Stream 11: pre-load the project config so module-level defaults can be
    # consulted by downstream consumers. Behavior is identical to today when
    # no config is present (the loader returns {}).
    try:
        from grokking_optimizers.compile_config import load_config as _load_cfg
        project_cfg = _load_cfg(Path(args.config) if args.config else None)
    except FileNotFoundError:
        raise
    except Exception:
        project_cfg = {}
    # Note: we deliberately do NOT auto-override args.arch, even when
    # archs.default is set — that would surprise users who passed --arch.
    if args.debug:
        args.verbose = True

    # Resolve aliases.
    if args.aot_only:
        args.runtime = "aot"
    if args.jit_only:
        args.runtime = "jit"
    if args.quick:
        args.mode = "bayesian"
        args.bayesian_trials = QUICK_BAYESIAN_TRIALS

    # ── --prune mode: GC the variant cache and exit, no build ──────
    if args.prune:
        cache_path = args.cache or (args.out / DEFAULT_CACHE_NAME)
        cache = CompileCache(cache_path)
        summary = cache.prune(
            max_age_days=args.prune_max_age_days,
            keep_top_n=args.prune_keep_top_n,
        )
        sys.stdout.write(
            f"[prune] cache={cache_path} "
            f"entries_scanned={summary['entries_scanned']} "
            f"dropped={summary['dropped']} kept={summary['kept']} "
            f"freed={summary['bytes_freed']/1e6:.2f}MB "
            f"(max_age_days={summary['max_age_days']}, "
            f"keep_top_n={summary['keep_top_n']})\n")
        return 0

    # ── Runtime split: spawn AOT then JIT, then return ──────────────
    if args.runtime == "both":
        # Do the (possibly slow) toolchain bootstrap ONCE in the parent so
        # AOT and JIT subprocesses inherit the discovered nvcc/hipcc via
        # PATH/CUDA_HOME/ROCM_PATH — no per-subprocess re-install. Vendor
        # dispatch matches build()'s logic so the right installer fires.
        _vendor = get_arch_entry(args.arch).vendor
        if args.bootstrap_cuda and _vendor == "cuda":
            if not shutil.which("nvcc"):
                _ensure_nvcc_on_path() or bootstrap_cuda_toolkit(arch=args.arch)
        elif args.bootstrap_rocm and _vendor == "hip":
            if not shutil.which("hipcc"):
                bootstrap_rocm_toolkit(arch=args.arch)
        elif args.bootstrap_jax and _vendor == "pallas":
            bootstrap_jax_tpu(arch=args.arch)
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
        max_tune_seconds=args.max_tune_seconds,
        min_improvement=args.min_improvement,
        patience=args.patience,
        seed=args.seed,
        debug_symbols=args.debug_symbols,
        debug=args.debug,
        bootstrap_cuda=args.bootstrap_cuda,
        bootstrap_rocm=args.bootstrap_rocm,
        bootstrap_jax=args.bootstrap_jax,
        pruner=args.pruner,
        transfer_learning=args.transfer_learning,
        enable_runtime_specialization=args.enable_runtime_specialization,
        config=project_cfg,
        enable_emitter=args.enable_emitter,
        enable_device_pgo=args.enable_device_pgo,
        prune_after_autotune=not args.no_auto_prune,
        prune_max_age_days=args.prune_max_age_days,
        prune_keep_top_n=args.prune_keep_top_n,
        strict_numerics=args.strict_numerics,
    )

    report = args.report or (
        args.out / f"compile_{args.optimizer}_{args.model}_{args.arch}.txt")
    sys.stdout.write(f"{report}\n")
    return 0 if (so is not None
                 or get_arch_entry(args.arch).vendor == "pallas"
                 or args.runtime == "aot") else 1


def _self_test() -> int:
    """Run inline self-checks. Returns 0 on success, 1 on failure."""
    import shutil
    import tempfile

    failures = 0
    passed = 0

    def _run(name, fn):
        nonlocal failures, passed
        try:
            fn()
            sys.stdout.write(f"  PASS: {name}\n")
            passed += 1
        except Exception as exc:
            sys.stdout.write(f"  FAIL: {name}: {exc}\n")
            failures += 1

    sys.stdout.write("[self-test] search_space\n")

    def test_load_yaml_validates_shape():
        td = Path(tempfile.mkdtemp())
        try:
            p = td / "bad.yaml"
            p.write_text("sm_90:\n  dims:\n    - name: block\n")
            try:
                load_yaml(p)
                raise AssertionError("expected SearchSpaceError")
            except SearchSpaceError:
                pass
        finally:
            shutil.rmtree(td)

    def test_load_yaml_rejects_duplicate_dim():
        td = Path(tempfile.mkdtemp())
        try:
            p = td / "dup.yaml"
            p.write_text(
                "sm_90:\n  dims:\n"
                "    - {name: block, type: int, values: [64]}\n"
                "    - {name: block, type: int, values: [128]}\n")
            try:
                load_yaml(p)
                raise AssertionError("expected SearchSpaceError")
            except SearchSpaceError:
                pass
        finally:
            shutil.rmtree(td)

    def test_real_yaml_loads():
        space = load_embedded_search_space()
        assert "sm_90" in space
        assert "gfx942" in space
        # Full programmatic space — every dim should have a non-trivial value list.
        sm90_dims = space["sm_90"]["dims"]
        assert len(sm90_dims) >= 10
        for d in sm90_dims:
            assert d["values"], f"dim {d['name']} has empty values list"

    def test_cartesian_counts():
        """The COMPLETE space is huge — verify by counting product of value
        list lengths, not by materializing the iterator (which would yield
        billions of dicts)."""
        space = load_embedded_search_space()
        count = cartesian_count(space, "sm_90")
        # Sanity bound — full sm_90 space should be at least 100 million combos
        # and well under 10^12.
        assert count > 100_000_000, f"sm_90 count too small: {count}"
        assert count < 10**12, f"sm_90 count unreasonably large: {count}"
        # Iterator yields exactly that many items — verify on a tiny slice.
        it = cartesian(space, "sm_90")
        first_few = list(itertools.islice(it, 5))
        assert len(first_few) == 5
        # Each yielded dict should have all dims.
        names = {d["name"] for d in space["sm_90"]["dims"]}
        assert set(first_few[0].keys()) == names

    def test_prefilter_eliminates():
        """Stream the prefilter against a CAPPED slice of the full space —
        verifies the prefilter actually rejects some configs, without
        materializing the billion-config Cartesian product."""
        space = load_embedded_search_space()
        # Take the first 50k Cartesian items and run prefilter on them.
        slice_iter = itertools.islice(cartesian(space, "sm_90"), 50_000)
        survivors, eliminated = ss_prefilter(
            slice_iter, space["sm_90"]["prefilter"])
        # Some configs must pass (block=32, vec=1 + reasonable values).
        # Some must fail (e.g. block=32 with stages=8 violates stages_block).
        assert len(survivors) + eliminated == 50_000
        assert eliminated > 0, "prefilter eliminated nothing — rules broken?"
        assert len(survivors) > 0, "prefilter eliminated everything — rules too strict?"

    def test_config_key_deterministic():
        cfg1 = {"block": 256, "vec": 4, "unroll": 8}
        cfg2 = {"unroll": 8, "block": 256, "vec": 4}
        assert config_key(cfg1) == config_key(cfg2)

    def test_hash_space_stable():
        space = load_embedded_search_space()
        h1 = hash_space(space, "sm_90")
        h2 = hash_space(space, "sm_90")
        assert h1 == h2
        assert h1 != hash_space(space, "gfx942")

    def test_per_arch_search_space():
        """Stream 2: every canonical arch in ARCH_TABLE has a builder
        wired in, and its Cartesian product cardinality falls in the
        arch-appropriate bounds (CUDA/HIP: 10^6..10^13; Pallas: 10..10^6)."""
        space = build_full_search_space()
        all_arches = _canonical_arches()
        assert len(all_arches) >= 20, f"only {len(all_arches)} canonical arches"
        for arch in all_arches:
            assert arch in space, f"{arch} missing from full search space"
            cnt = cartesian_count(space, arch)
            vendor = ARCH_TABLE[arch].vendor
            if vendor in ("cuda", "hip"):
                assert 1_000_000 <= cnt <= 10**13, (
                    f"{arch}: count {cnt} out of CUDA/HIP bounds")
            else:  # pallas
                assert 10 <= cnt <= 10**6, (
                    f"{arch}: pallas count {cnt} out of bounds")
            sys.stdout.write(
                f"    [per_arch] {arch}({vendor}) cardinality={cnt:,}\n")

    def test_alias_search_space_consistency():
        """Aliases (sm_90, sm_100, ...) must return the same space dict
        object as their canonical 'a' counterpart."""
        space = build_full_search_space()
        for alias, canonical in [("sm_90", "sm_90a"),
                                  ("sm_100", "sm_100a"),
                                  ("sm_103", "sm_103a"),
                                  ("sm_120", "sm_120a")]:
            assert alias in space and canonical in space
            assert space[alias] is space[canonical], (
                f"alias {alias} doesn't share dict with {canonical}")

    _run("load_yaml_validates_shape", test_load_yaml_validates_shape)
    _run("load_yaml_rejects_duplicate_dim", test_load_yaml_rejects_duplicate_dim)
    _run("embedded_yaml_loads", test_real_yaml_loads)
    _run("cartesian_counts", test_cartesian_counts)
    _run("prefilter_eliminates", test_prefilter_eliminates)
    _run("config_key_deterministic", test_config_key_deterministic)
    _run("hash_space_stable", test_hash_space_stable)
    _run("per_arch_search_space", test_per_arch_search_space)
    _run("alias_search_space_consistency", test_alias_search_space_consistency)

    sys.stdout.write("[self-test] pgo\n")

    def test_hash_workload_deterministic():
        td = Path(tempfile.mkdtemp())
        try:
            s = td / "w.py"
            s.write_text("print('hi')")
            assert hash_workload(s, 1000) == hash_workload(s, 1000)
        finally:
            shutil.rmtree(td)

    def test_hash_workload_changes():
        td = Path(tempfile.mkdtemp())
        try:
            s = td / "w.py"
            s.write_text("a = 1")
            h1 = hash_workload(s, 1000)
            s.write_text("a = 2")
            h2 = hash_workload(s, 1000)
            assert h1 != h2
            assert hash_workload(s, 1000) != hash_workload(s, 2000)
        finally:
            shutil.rmtree(td)

    def test_instrument_flags_cuda():
        td = Path(tempfile.mkdtemp())
        try:
            h, d, l = instrument_flags("sm_90", td, ["-O3"], ["-O3"], ["-flto"])
            assert any("-fprofile-generate" in f for f in h)
            assert any("-fprofile-generate" in f for f in d)
        finally:
            shutil.rmtree(td)

    def test_use_flags_round_trip():
        td = Path(tempfile.mkdtemp())
        try:
            h, d, l = use_flags("sm_90", td, ["-O3"], ["-O3"], ["-flto"])
            assert any("-fprofile-use" in f for f in h)
            assert any("-fprofile-correction" in f for f in h)
        finally:
            shutil.rmtree(td)

    _run("hash_workload_deterministic", test_hash_workload_deterministic)
    _run("hash_workload_changes", test_hash_workload_changes)
    _run("instrument_flags_cuda", test_instrument_flags_cuda)
    _run("use_flags_round_trip", test_use_flags_round_trip)

    sys.stdout.write("[self-test] device_profiling\n")

    def test_device_profiling_import():
        from grokking_optimizers import device_profiling  # noqa: F401

    def test_stall_to_bias_mapping():
        from grokking_optimizers import device_profiling
        hints = device_profiling._stall_to_bias_hints({
            "long_scoreboard": 0.4,
            "not_selected": 0.2,
        })
        assert "swizzle" in hints, hints
        assert "waves_per_eu" in hints, hints
        # Specific value recommendations must populate concrete values.
        assert 64 in hints["swizzle"], hints["swizzle"]
        assert 128 in hints["swizzle"], hints["swizzle"]

    def test_stall_to_bias_empty_input():
        from grokking_optimizers import device_profiling
        assert device_profiling._stall_to_bias_hints({}) == {}
        # All-low fractions: dim names still seeded (top-5 above 5%
        # threshold), but no specific value recommendations.
        h = device_profiling._stall_to_bias_hints({"long_scoreboard": 0.01})
        assert h == {}, f"low-fraction stall should not bias: {h}"

    def test_bias_trial_queue_enqueues():
        from grokking_optimizers import device_profiling

        class _MockStudy:
            def __init__(self):
                self.queued = []

            def enqueue_trial(self, cfg):
                self.queued.append(cfg)

        ms = _MockStudy()
        tiny_space = {"dims": [
            {"name": "swizzle", "type": "int",
             "values": [0, 64, 128, 256],
             "target": "device", "macro": "SWZ"},
            {"name": "block", "type": "int",
             "values": [32, 64, 128, 256],
             "target": "device", "macro": "BLK"},
        ]}
        fake_stall = {"bias_hints": {"swizzle": [64, 128]}}
        n = device_profiling.bias_trial_queue(
            ms, fake_stall, tiny_space, "sm_90a")
        assert n == 2, f"expected 2 enqueued, got {n}"
        assert len(ms.queued) == 2
        for cfg in ms.queued:
            assert cfg["swizzle"] in (64, 128)
            assert cfg["block"] == 128  # middle value of [32,64,128,256]

    def test_bias_trial_queue_empty():
        from grokking_optimizers import device_profiling

        class _MockStudy:
            def __init__(self): self.queued = []
            def enqueue_trial(self, cfg): self.queued.append(cfg)

        ms = _MockStudy()
        assert device_profiling.bias_trial_queue(
            ms, None, {"dims": []}, "sm_90a") == 0
        assert device_profiling.bias_trial_queue(
            ms, {}, {"dims": []}, "sm_90a") == 0
        assert device_profiling.bias_trial_queue(
            ms, {"bias_hints": {}}, {"dims": []}, "sm_90a") == 0

    def test_run_device_pgo_round_disabled():
        """When spec.enable_device_pgo=False, hook returns None and emits
        no work — critical because every existing PGO build path runs
        through here."""
        from grokking_optimizers import device_profiling

        class _DummySpec:
            arch = "sm_90"
            enable_device_pgo = False

        import io as _io
        rep = _io.StringIO()
        result = device_profiling.run_device_pgo_round(
            _DummySpec(), ["echo", "ignored"], Path(tempfile.mkdtemp()), rep)
        assert result is None
        assert rep.getvalue() == "", f"expected silent no-op, got {rep.getvalue()!r}"

    def test_stall_sidecar_round_trip():
        from grokking_optimizers import device_profiling
        td = Path(tempfile.mkdtemp())
        try:
            info = {
                "arch": "sm_90a",
                "tool": "nsys",
                "stall_reasons": {"long_scoreboard": 0.42},
                "bias_hints": {"swizzle": [64, 128]},
            }
            p = device_profiling.write_stall_sidecar(info, td)
            assert p.exists()
            assert p.name == "device_stall_info.json"
            loaded = device_profiling.read_stall_sidecar(td)
            assert loaded == info
        finally:
            shutil.rmtree(td)

    def test_buildspec_has_device_pgo_field():
        """Stream 8 wiring: BuildSpec must carry the flag so it propagates
        from CLI -> build() -> spec -> _build_aot_pgo -> hook."""
        spec = BuildSpec(
            optimizer="lion", model="mamba", arch="sm_90",
            out_dir=Path("/tmp"))
        assert hasattr(spec, "enable_device_pgo")
        assert spec.enable_device_pgo is False
        spec2 = BuildSpec(
            optimizer="lion", model="mamba", arch="sm_90",
            out_dir=Path("/tmp"), enable_device_pgo=True)
        assert spec2.enable_device_pgo is True

    _run("device_profiling_import", test_device_profiling_import)
    _run("stall_to_bias_mapping", test_stall_to_bias_mapping)
    _run("stall_to_bias_empty_input", test_stall_to_bias_empty_input)
    _run("bias_trial_queue_enqueues", test_bias_trial_queue_enqueues)
    _run("bias_trial_queue_empty", test_bias_trial_queue_empty)
    _run("run_device_pgo_round_disabled", test_run_device_pgo_round_disabled)
    _run("stall_sidecar_round_trip", test_stall_sidecar_round_trip)
    _run("buildspec_has_device_pgo_field", test_buildspec_has_device_pgo_field)

    # ---- Stream 3: per-arch native flag emission ----
    sys.stdout.write("[self-test] flags\n")

    def _spec(arch: str) -> "BuildSpec":
        return BuildSpec(
            optimizer="supergrok2",
            model="mamba",
            arch=arch,
            out_dir=Path(tempfile.gettempdir()) / f"sg_st_{arch}",
        )

    # Canonical archs we expect Stream 1 + 3 to fully support. Aliases
    # (sm_90, sm_100, ...) are tested via their canonical entries.
    _CANONICAL_ARCHES_S3 = [
        # NVIDIA
        "sm_75", "sm_80", "sm_86", "sm_89",
        "sm_90a", "sm_100a", "sm_103a", "sm_120a",
        # AMD
        "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
        "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151",
        "gfx1200", "gfx1201",
    ]

    def test_per_arch_native_flags():
        """_device_cflags must produce a non-empty, arch-correct flag list
        for every canonical NVIDIA/AMD arch, with no cross-contamination."""
        for arch in _CANONICAL_ARCHES_S3:
            flags = _device_cflags(_spec(arch))
            assert flags, f"{arch}: empty device cflags"
            entry = get_arch_entry(arch)

            if entry.vendor == "cuda":
                joined = " ".join(flags)
                # Correct -gencode for this arch (canonical entry already
                # carries the suffix-bearing token).
                expect_sm = f"sm_{entry.cutlass_arch}{entry.arch_suffix}"
                assert f"code={expect_sm}" in joined, \
                    f"{arch}: missing code={expect_sm} in {joined[:200]}"
                # No --offload-arch (that's HIP).
                assert "--offload-arch" not in joined, \
                    f"{arch}: CUDA arch leaked HIP --offload-arch"
                # No OTHER CUDA arch's SASS code= token appears.
                for other in _CANONICAL_ARCHES_S3:
                    o_entry = get_arch_entry(other)
                    if o_entry.vendor != "cuda":
                        continue
                    if o_entry.cutlass_arch == entry.cutlass_arch:
                        continue
                    other_sm = f"code=sm_{o_entry.cutlass_arch}{o_entry.arch_suffix}"
                    assert other_sm not in joined, \
                        f"{arch}: leaked {other_sm}"
                # Feature gating — Hopper+ gets TMA, Blackwell gets fp4.
                if "tma" in entry.features:
                    assert "-DCUDA_TMA_ENABLED=1" in flags
                else:
                    assert "-DCUDA_TMA_ENABLED=1" not in flags
                if "fp4" in entry.features:
                    assert "-DCUDA_FP4_ENABLED=1" in flags

            elif entry.vendor == "hip":
                joined = " ".join(flags)
                assert f"--offload-arch={entry.hipcc_offload_arch}" in joined, \
                    f"{arch}: missing --offload-arch={entry.hipcc_offload_arch}"
                assert "-gencode" not in joined, \
                    f"{arch}: HIP arch leaked CUDA -gencode"
                # Wave-size gating.
                if entry.warp_size == 64:
                    assert "-mcumode" in flags, f"{arch}: CDNA missing -mcumode"
                    assert "-mwavefrontsize32" not in flags
                elif entry.warp_size == 32:
                    assert "-mwavefrontsize32" in flags, \
                        f"{arch}: RDNA missing -mwavefrontsize32"
                    assert "-mcumode" not in flags
                # MFMA / WMMA macro gating.
                if "mfma" in entry.features:
                    assert "-DAMDGPU_MFMA_ENABLED=1" in flags
                if "wmma" in entry.features:
                    assert "-DAMDGPU_WMMA_ENABLED=1" in flags

    def test_flag_base_superset_regression():
        """The current sm_90 / gfx942 flag lists must be a STRICT superset
        of the pre-Stream-3 base flag list. Guards against accidental
        regressions on these two canonical archs.
        """
        sm90 = set(_device_cflags(_spec("sm_90a")))
        legacy_sm90 = {
            "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "--extra-device-vectorization",
            "--resource-usage", "-dlto",
        }
        missing = legacy_sm90 - sm90
        assert not missing, f"sm_90 lost legacy flags: {missing}"

        gfx942 = set(_device_cflags(_spec("gfx942")))
        legacy_gfx942 = {
            "-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math", "-fPIC",
            "-fgpu-flush-denormals-to-zero", "-flto",
            "--offload-arch=gfx942",
        }
        missing_amd = legacy_gfx942 - gfx942
        assert not missing_amd, f"gfx942 lost legacy flags: {missing_amd}"

    def test_nvcc_no_duplicate_ptxas_o3():
        """Exactly one PTXAS opt-level flag — duplicate -Xptxas -O3 must
        not reappear after the Stream 3 cleanup."""
        flags = _device_cflags(_spec("sm_90a"))
        opt_hits = sum(1 for f in flags
                       if f in ("--opt-level=3",) or f == "-O3")
        # We expect one "-O3" (top-level NVCC) and one "--opt-level=3" (PTXAS)
        # — never two of the same thing.
        assert flags.count("-O3") == 1, \
            f"sm_90 has {flags.count('-O3')} -O3 flags, expected 1"
        assert flags.count("--opt-level=3") == 1, \
            f"sm_90 has {flags.count('--opt-level=3')} --opt-level=3 flags"

    def test_resolve_extra_feature_macros():
        """resolve_extra_nvcc_flags / resolve_extra_hipcc_flags emit
        per-arch feature macros driven by ARCH_TABLE.features (not by
        if-arch-==-string branches)."""
        # Hopper TMA / wgmma
        flags = resolve_extra_nvcc_flags({}, [], "sm_90a")
        assert "-DCUDA_TMA_ENABLED=1" in flags
        assert "-DCUDA_WGMMA_ENABLED=1" in flags
        assert "-DCUDA_CLUSTER_ENABLED=1" in flags
        # Blackwell fp4 + tcgen05
        flags = resolve_extra_nvcc_flags({}, [], "sm_100a")
        assert "-DCUDA_FP4_ENABLED=1" in flags
        assert "-DCUDA_TCGEN05_ENABLED=1" in flags
        # Consumer Blackwell has fp4 but no tma/wgmma/cluster.
        flags = resolve_extra_nvcc_flags({}, [], "sm_120a")
        assert "-DCUDA_FP4_ENABLED=1" in flags
        assert "-DCUDA_TMA_ENABLED=1" not in flags
        # CDNA3 has fp8 MFMA.
        flags = resolve_extra_hipcc_flags({}, [], "gfx942")
        assert "-DAMDGPU_FP8_MFMA=1" in flags
        assert "-DAMDGPU_MFMA_ENABLED=1" in flags
        # RDNA3 has WMMA + dpp + tgsplit, but not MFMA.
        flags = resolve_extra_hipcc_flags({}, [], "gfx1100")
        assert "-DAMDGPU_WMMA_ENABLED=1" in flags
        assert "-DAMDGPU_DPP=1" in flags
        assert "-DAMDGPU_TGSPLIT=1" in flags
        assert "-DAMDGPU_MFMA_ENABLED=1" not in flags
        # Layout dim values map to dtype macros.
        flags = resolve_extra_nvcc_flags({"fp8_layout": "e4m3"}, [], "sm_90a")
        assert "-DCUDA_FP8_E4M3=1" in flags
        flags = resolve_extra_hipcc_flags({"fp4_layout": "ocp"}, [], "gfx950")
        assert "-DAMDGPU_FP4_OCP=1" in flags

    def test_xla_env():
        """_xla_env returns the canonical XLA_FLAGS for Pallas archs and
        an empty dict for non-Pallas archs."""
        td = Path(tempfile.mkdtemp())
        try:
            env = _xla_env("tpu_v5p", td)
            assert env, "tpu_v5p should produce a non-empty env"
            assert "XLA_FLAGS" in env
            assert "xla_gpu_autotune_level" in env["XLA_FLAGS"]
            assert "xla_gpu_enable_triton_gemm=true" in env["XLA_FLAGS"]
            assert "xla_gpu_graph_level=3" in env["XLA_FLAGS"]
            assert env["JAX_COMPILATION_CACHE_DIR"] == str(td / "jax_cache")
            assert env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
            # Cache dir is created lazily.
            assert (td / "jax_cache").is_dir()
            # Non-Pallas arch returns empty.
            assert _xla_env("sm_90a", td) == {}
            assert _xla_env("gfx942", td) == {}
            assert _xla_env("nonexistent", td) == {}
        finally:
            shutil.rmtree(td)

    _run("per_arch_native_flags", test_per_arch_native_flags)
    _run("flag_base_superset_regression", test_flag_base_superset_regression)
    _run("nvcc_no_duplicate_ptxas_o3", test_nvcc_no_duplicate_ptxas_o3)
    _run("resolve_extra_feature_macros", test_resolve_extra_feature_macros)
    _run("xla_env", test_xla_env)

    sys.stdout.write("[self-test] bayesian\n")

    def _tiny_space():
        return {"sm_90": {"dims": [
            {"name": "block", "type": "int", "values": [64, 128, 256, 512],
             "macro": "SG_TUNED_BLOCK_SIZE", "applies_to": ["host", "device"]},
            {"name": "vec", "type": "int", "values": [1, 2, 4],
             "macro": "SG_TUNED_VEC_WIDTH", "applies_to": ["host", "device"]},
            {"name": "unroll", "type": "int", "values": [1, 2, 4, 8],
             "macro": "SG_TUNED_UNROLL", "applies_to": ["host", "device"]},
        ], "prefilter": {"rules": []}}}

    def _synthetic_timer(cfg):
        score = ((cfg["block"] - 256) / 256.0) ** 2
        score += ((cfg["vec"] - 4) / 4.0) ** 2
        score += ((cfg["unroll"] - 8) / 8.0) ** 2
        score += 0.05
        return {"timing_ms": score, "min_ms": score - 0.01,
                "max_ms": score + 0.01, "n": 21}

    def test_bayesian_finds_winner():
        td = Path(tempfile.mkdtemp())
        try:
            trials, _stop_info = run_bayesian(
                "sm_90", _tiny_space(), n_trials=40, seed=0,
                storage=td / "study.db", timer=_synthetic_timer)
            assert len(trials) == 40
            w = pick_winner(trials)
            assert w is not None and w["timing_ms"] < 0.5
        finally:
            shutil.rmtree(td)

    def test_topk_refine_generates_neighbours():
        td = Path(tempfile.mkdtemp())
        try:
            trials, _stop_info = run_bayesian(
                "sm_90", _tiny_space(), n_trials=20, seed=0,
                storage=td / "study.db", timer=_synthetic_timer)
            refines = topk_refine(trials, _tiny_space(), "sm_90",
                                  top_k=5, radius=2, timer=_synthetic_timer)
            assert all(t["stage"] == "refine" for t in refines)
        finally:
            shutil.rmtree(td)

    _run("bayesian_finds_winner", test_bayesian_finds_winner)
    _run("topk_refine_generates_neighbours", test_topk_refine_generates_neighbours)

    sys.stdout.write("[self-test] early_stopping\n")

    import random

    def _es_tiny_space():
        return {"sm_90": {"dims": [
            {"name": "block", "type": "int",
             "values": [64, 128, 192, 256, 320, 384, 512],
             "macro": "BLOCK", "applies_to": ["host", "device"]},
            {"name": "vec", "type": "int",
             "values": [1, 2, 4, 8],
             "macro": "VEC", "applies_to": ["host", "device"]},
        ], "prefilter": {"rules": []}}}

    def test_early_stopper_triggers():
        random.seed(0)

        def _es_synthetic_timer(cfg):
            # Optimum at block=256, vec=4. Distance-based timing.
            bd = abs(cfg.get("block", 128) - 256) / 256.0
            vd = abs(cfg.get("vec", 1) - 4) / 4.0
            return 0.1 + bd + vd + 0.01 * random.random()

        stopper = BayesianEarlyStopper(
            min_delta_rel=0.005, patience=50, max_trials=2000)
        records, stop_info = run_bayesian(
            "sm_90", _es_tiny_space(), n_trials=None, seed=0,
            timer=_es_synthetic_timer, stopper=stopper)
        assert 30 <= len(records) <= 1500, \
            f"unexpected trial count {len(records)}"
        assert stop_info["stop_reason"] is not None, \
            f"stopper did not record a reason: {stop_info}"
        sys.stdout.write(
            f"    (stopper triggered after {len(records)} trials, "
            f"reason={stop_info['stop_reason']})\n")

    def test_early_stopper_wall_clock():
        stopper2 = BayesianEarlyStopper(max_seconds=0.5)
        records2, stop_info2 = run_bayesian(
            "sm_90", _es_tiny_space(), n_trials=None, seed=0,
            timer=lambda c: (time.sleep(0.05) or 0.1),
            stopper=stopper2)
        assert "max_seconds" in (stop_info2.get("stop_reason") or ""), \
            f"expected max_seconds stop, got {stop_info2.get('stop_reason')}"
        sys.stdout.write(
            f"    (wall-clock stop after {len(records2)} trials)\n")

    def test_topk_elbow_detection():
        fake_records = [
            {"status": "ok", "timing_ms": v, "stage": "bayesian",
             "trial_num": i, "config": {"block": 128 * (i + 1)},
             "error": None}
            for i, v in enumerate([0.1, 0.11, 0.12, 0.5, 0.6, 0.7, 0.8])
        ]
        elbow = _detect_topk_elbow(fake_records)
        assert 1 <= elbow <= 5, f"elbow={elbow}"
        sys.stdout.write(f"    (topk elbow detected at index {elbow})\n")

    def test_stopper_to_dict_serializable():
        s = BayesianEarlyStopper(max_seconds=1.0, patience=10)
        s.observe(0.5, {"block": 128, "vec": 4})
        s.observe(0.4, {"block": 256, "vec": 4})
        d = s.to_dict()
        # Round-trip through JSON to confirm cache-persistence safety.
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped["trial_count"] == 2
        assert round_tripped["best"] == 0.4
        assert round_tripped["coverage_size"] == 3  # (block,128), (block,256), (vec,4)

    _run("early_stopper_triggers", test_early_stopper_triggers)
    _run("early_stopper_wall_clock", test_early_stopper_wall_clock)
    _run("topk_elbow_detection", test_topk_elbow_detection)
    _run("stopper_to_dict_serializable", test_stopper_to_dict_serializable)

    sys.stdout.write("[self-test] cache\n")

    def test_v2_to_v3_migration():
        td = Path(tempfile.mkdtemp())
        try:
            cp = td / "cache.json"
            cp.write_text(json.dumps({
                "version": 2, "created_at": "2026-04-10T12:00:00",
                "host_history": [{"platform": "old"}],
                "entries": {"lion/mamba/sm_90": {
                    "source_hash": "abc", "host_cflags_hash": "def",
                    "device_cflags_hash": "ghi", "primary_artifact": None,
                    "variant_artifacts": {}, "sweep_history": [],
                    "tuned_config": None,
                    "aot_completed_at": "2026-04-10T12:00:00",
                    "jit_completed_at": None, "aot_host": {}, "jit_host": None,
                }}}, indent=2))
            cache = CompileCache(cp)
            assert cache._data["version"] == CACHE_VERSION
            e = cache._data["entries"]["lion/mamba/sm_90"]
            assert e["pgo_enabled"] is False
            assert e["bayesian_trials"] == []
        finally:
            shutil.rmtree(td)

    def test_cache_round_trips():
        td = Path(tempfile.mkdtemp())
        try:
            cp = td / "fresh.json"
            cache = CompileCache(cp)
            cache.record_aot("lion", "mamba", "sm_90",
                             source_hash="s", host_flags_hash="h",
                             device_flags_hash="d", so_path=None,
                             pgo_enabled=True, pgo_workload_hash="w",
                             search_space_hash="ss")
            cache.save()
            reloaded = CompileCache(cp)
            e = reloaded._data["entries"]["lion/mamba/sm_90"]
            assert e["pgo_enabled"] is True
            assert e["pgo_workload_hash"] == "w"
        finally:
            shutil.rmtree(td)

    _run("v2_to_v3_migration", test_v2_to_v3_migration)
    _run("cache_round_trips", test_cache_round_trips)

    def test_cache_prune():
        """Populate 5 variants, prune to top-2 by timing, verify 3 dropped
        + .so files unlinked + bytes_freed accounted for."""
        td = Path(tempfile.mkdtemp())
        try:
            cp = td / "cache.json"
            cache = CompileCache(cp)
            # Create 5 fake .so files with content so bytes_freed is non-zero.
            so_paths: List[Path] = []
            for i in range(5):
                p = td / f"v{i}.so"
                p.write_bytes(b"x" * (1024 * (i + 1)))
                so_paths.append(p)
                cache.record_variant("lion", "mamba", "sm_90",
                                     config_key=f"block-{128 + i*32}",
                                     so_path=p)
                # And synthesise a trial record so prune can rank by timing.
                cache.record_trial("lion", "mamba", "sm_90", {
                    "trial_num":   i,
                    "stage":       "exhaustive",
                    "config":      {"block": 128 + i*32},
                    "config_key":  f"block-{128 + i*32}",
                    # i=0 is the fastest, i=4 is the slowest.
                    "timing_ms":   0.1 + i * 0.5,
                    "min_ms":      0.09 + i * 0.5,
                    "max_ms":      0.12 + i * 0.5,
                    "n":           21,
                    "host":        {},
                    "recorded_at": datetime.datetime.now().isoformat(),
                })
            summary = cache.prune(max_age_days=30, keep_top_n=2)
            assert summary["kept"] == 2, summary
            assert summary["dropped"] == 3, summary
            # bytes_freed should be the sum of sizes for v2/v3/v4
            # (i=2 → 3KB, i=3 → 4KB, i=4 → 5KB = 12KB total).
            assert summary["bytes_freed"] == (3 + 4 + 5) * 1024, summary
            # The three pruned .so files are gone; the top-2 survive.
            assert not so_paths[2].exists()
            assert not so_paths[3].exists()
            assert not so_paths[4].exists()
            assert so_paths[0].exists()
            assert so_paths[1].exists()
            # Entry's variant_artifacts now has only the kept ckeys.
            e = cache.get("lion", "mamba", "sm_90")
            assert set(e["variant_artifacts"].keys()) == {"block-128", "block-160"}, \
                list(e["variant_artifacts"].keys())
        finally:
            shutil.rmtree(td)

    def test_cache_prune_dry_run():
        """dry_run reports what *would* be dropped without touching anything."""
        td = Path(tempfile.mkdtemp())
        try:
            cp = td / "cache.json"
            cache = CompileCache(cp)
            for i in range(3):
                p = td / f"v{i}.so"
                p.write_bytes(b"z" * 100)
                cache.record_variant("lion", "mamba", "sm_90",
                                     config_key=f"k{i}", so_path=p)
                cache.record_trial("lion", "mamba", "sm_90", {
                    "trial_num": i, "stage": "exhaustive",
                    "config": {"i": i}, "config_key": f"k{i}",
                    "timing_ms": float(i + 1), "min_ms": float(i),
                    "max_ms": float(i + 2), "n": 21, "host": {},
                    "recorded_at": datetime.datetime.now().isoformat(),
                })
            summary = cache.prune(max_age_days=30, keep_top_n=1, dry_run=True)
            assert summary["dropped"] == 2 and summary["kept"] == 1, summary
            assert summary["dry_run"] is True
            # Nothing actually removed.
            assert (td / "v0.so").exists()
            assert (td / "v1.so").exists()
            assert (td / "v2.so").exists()
            e = cache.get("lion", "mamba", "sm_90")
            assert len(e["variant_artifacts"]) == 3
        finally:
            shutil.rmtree(td)

    _run("cache_prune", test_cache_prune)
    _run("cache_prune_dry_run", test_cache_prune_dry_run)

    sys.stdout.write("[self-test] multi_gpu_pool\n")

    def test_visible_devices_default():
        """With CUDA_VISIBLE_DEVICES unset, default to a single '0' device."""
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            devs = MultiGPUTimingPool.visible_devices("cuda")
            assert devs == ["0"], devs
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_visible_devices_multi():
        """Comma-separated env var → list of devices in order."""
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
        try:
            devs = MultiGPUTimingPool.visible_devices("cuda")
            assert devs == ["0", "1", "3"], devs
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_visible_devices_hip():
        """HIP backend reads HIP_VISIBLE_DEVICES, not CUDA_VISIBLE_DEVICES."""
        saved_hip = os.environ.get("HIP_VISIBLE_DEVICES")
        saved_cuda = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["HIP_VISIBLE_DEVICES"] = "2,4"
        try:
            devs = MultiGPUTimingPool.visible_devices("hip")
            assert devs == ["2", "4"], devs
            # CUDA env var is irrelevant for HIP.
            assert MultiGPUTimingPool.env_var_for("hip") == "HIP_VISIBLE_DEVICES"
            assert MultiGPUTimingPool.env_var_for("cuda") == "CUDA_VISIBLE_DEVICES"
        finally:
            if saved_hip is None:
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
            else:
                os.environ["HIP_VISIBLE_DEVICES"] = saved_hip
            if saved_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda

    def test_pool_constructs_per_device_workers():
        """Constructing the pool spawns one TimingWorker per visible device,
        each with the appropriate env_overlay (no subprocesses are started
        because we never call .start())."""
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        try:
            pool = MultiGPUTimingPool("Lion", vendor="cuda",
                                      enable_watchdog=False)
            assert len(pool.workers) == 2
            assert pool.workers[0].env_overlay == {"CUDA_VISIBLE_DEVICES": "0"}
            assert pool.workers[1].env_overlay == {"CUDA_VISIBLE_DEVICES": "1"}
            assert pool.workers[0].env["CUDA_VISIBLE_DEVICES"] == "0"
            assert pool.workers[1].env["CUDA_VISIBLE_DEVICES"] == "1"
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    _run("visible_devices_default", test_visible_devices_default)
    _run("visible_devices_multi", test_visible_devices_multi)
    _run("visible_devices_hip", test_visible_devices_hip)
    _run("pool_constructs_per_device_workers", test_pool_constructs_per_device_workers)

    def test_work_stealing_fast_worker_dominates():
        """Build a 2-worker pool with mock workers of different latencies.
        Submit 10 jobs; assert fast worker handles >=6 of them and total
        wall-clock is < (slow worker × number of jobs) ≈ no serialization."""

        class _MockWorker:
            def __init__(self, latency_s: float, name: str):
                self.latency = latency_s
                self.name = name
                self.calls = 0
                self._dead = False

            def alive(self) -> bool:
                return not self._dead

            def time(self, variant_so, opt_class="", **kw):
                self.calls += 1
                time.sleep(self.latency)
                return {"variant_so": str(variant_so),
                        "median_ms": self.latency * 1000,
                        "worker": self.name}

            def stop(self):
                self._dead = True

        fast = _MockWorker(0.01, "fast")
        slow = _MockWorker(0.10, "slow")

        # Build a pool but skip the real TimingWorker spawn — splice mocks in.
        pool = MultiGPUTimingPool.__new__(MultiGPUTimingPool)
        pool.vendor = "cuda"
        pool.devices = ["0", "1"]
        pool.workers = [fast, slow]
        pool._queue = queue.Queue()
        pool._stopped = threading.Event()
        pool._dispatch_threads = []
        for idx, w in enumerate(pool.workers):
            t = threading.Thread(
                target=pool._dispatch_loop, args=(idx, w), daemon=True)
            t.start()
            pool._dispatch_threads.append(t)

        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = [ex.submit(pool.time, Path(f"/tmp/variant_{i}.so"), "Adamw")
                    for i in range(10)]
            results = [f.result() for f in futs]
        wall = time.time() - t0
        pool.stop()

        fast_calls = sum(1 for r in results if r["worker"] == "fast")
        slow_calls = sum(1 for r in results if r["worker"] == "slow")
        assert fast_calls + slow_calls == 10, (fast_calls, slow_calls)
        assert fast_calls >= 6, (
            f"work-stealing failed — fast={fast_calls} slow={slow_calls}; "
            f"fast worker should drain the queue ahead of the slow one")
        # Pure serialization would take 10*0.1=1.0s. Work-stealing should be
        # bounded by ceil(10/2) * slow_latency = 5*0.1 = 0.5s — give us 2x
        # slack for thread scheduling.
        assert wall < 1.0, f"wall-clock {wall:.3f}s suggests serialization"
        sys.stdout.write(
            f"    (fast={fast_calls} slow={slow_calls} wall={wall:.3f}s)\n")

    _run("multigpu_work_stealing", test_work_stealing_fast_worker_dominates)

    sys.stdout.write("[self-test] kernel_headers\n")

    def _read_kernel(p: Path) -> str:
        return p.read_text(encoding="utf-8")

    KERNEL_DIR = REPO_ROOT / "grokking_optimizers" / "kernels"

    def test_elementwise_headers():
        sm90_dir = KERNEL_DIR / "sm_90"
        gfx942_dir = KERNEL_DIR / "gfx942"
        for opt in ("adamw", "lion", "grokfast", "grokadamw"):
            sm = sm90_dir / f"{opt}_sm90.cuh"
            gfx = gfx942_dir / f"{opt}_gfx942.hip.hpp"
            assert sm.is_file(), f"Missing {sm}"
            assert gfx.is_file(), f"Missing {gfx}"
            sm_src = _read_kernel(sm)
            gfx_src = _read_kernel(gfx)
            assert f"{opt}_update(" in sm_src
            assert f"{opt}_update(" in gfx_src
            assert f"{opt}_kernel(" in sm_src
            assert f"{opt}_kernel(" in gfx_src
            assert "namespace grokking" in sm_src
            assert "namespace grokking" in gfx_src

    def test_model_headers():
        models = [
            ("transformer_decoder", "sm_90", "transformer_decoder_sm90.cuh"),
            ("transformer_decoder", "gfx942", "transformer_decoder_gfx942.hip.hpp"),
            ("transformer_decoder", "tpu", "transformer_decoder_tpu.py"),
            ("mamba3", "sm_90", "mamba3_sm90.cuh"),
            ("mamba3", "gfx942", "mamba3_gfx942.hip.hpp"),
            ("mamba3", "tpu", "mamba3_tpu.py"),
            ("vit", "sm_90", "vit_sm90.cuh"),
            ("vit", "gfx942", "vit_gfx942.hip.hpp"),
            ("vit", "tpu", "vit_tpu.py"),
        ]
        for model, arch, fname in models:
            p = KERNEL_DIR / arch / fname
            assert p.is_file(), f"Missing {p}"
            src = _read_kernel(p)
            assert "param_bytes" in src or "Sizes" in src or "SMEM_BYTES" in src, \
                f"{fname} missing size helpers"

    def test_optimizer_model_cross():
        sm90_dir = KERNEL_DIR / "sm_90"
        gfx942_dir = KERNEL_DIR / "gfx942"
        tpu_dir = KERNEL_DIR / "tpu"
        opts = ("adamw", "lion", "grokfast", "grokadamw")
        models_sm90 = ("transformer_decoder_sm90.cuh", "mamba3_sm90.cuh", "vit_sm90.cuh")
        models_gfx = ("transformer_decoder_gfx942.hip.hpp", "mamba3_gfx942.hip.hpp",
                      "vit_gfx942.hip.hpp")
        models_tpu = ("transformer_decoder_tpu.py", "mamba3_tpu.py", "vit_tpu.py")
        for opt in opts:
            assert (sm90_dir / f"{opt}_sm90.cuh").is_file()
            assert (gfx942_dir / f"{opt}_gfx942.hip.hpp").is_file()
        for m in models_sm90:
            assert (sm90_dir / m).is_file()
        for m in models_gfx:
            assert (gfx942_dir / m).is_file()
        for m in models_tpu:
            assert (tpu_dir / m).is_file()
        assert (sm90_dir / "common_sm90.cuh").is_file()
        assert (gfx942_dir / "common_gfx942.hip.hpp").is_file()
        assert (tpu_dir / "common_tpu.py").is_file()

    _run("elementwise_headers", test_elementwise_headers)
    _run("model_headers", test_model_headers)
    _run("optimizer_model_cross", test_optimizer_model_cross)

    sys.stdout.write("[self-test] arch_table\n")

    def test_arch_table_completeness():
        """Every arch in scope is present with all required fields filled,
        alias resolution works, and the ARCH_INFO derived view exposes the
        legacy 6 keys for backward compatibility."""
        expected_canonical = {
            # NVIDIA / CUDA
            "sm_75", "sm_80", "sm_86", "sm_89",
            "sm_90a", "sm_100a", "sm_103a", "sm_120a",
            # AMD / HIP
            "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
            "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151",
            "gfx1200", "gfx1201",
            # Google / Pallas
            "tpu_v4", "tpu_v5e", "tpu_v5p", "tpu_v6e", "tpu_v7",
        }
        missing = expected_canonical - set(ARCH_TABLE.keys())
        assert not missing, f"ARCH_TABLE missing canonical arches: {missing}"

        # Every entry must have every required field non-None where applicable
        # to its vendor.
        for arch, entry in ARCH_TABLE.items():
            assert entry.vendor in ("cuda", "hip", "pallas"), \
                f"{arch}: bad vendor {entry.vendor!r}"
            assert entry.display_name, f"{arch}: empty display_name"
            assert entry.subdir, f"{arch}: empty subdir"
            assert entry.macro.startswith("SG_BUILD_ARCH_"), \
                f"{arch}: bad macro {entry.macro!r}"
            assert isinstance(entry.min_toolchain_version, tuple) and \
                len(entry.min_toolchain_version) >= 2, \
                f"{arch}: bad min_toolchain_version"
            assert isinstance(entry.features, frozenset), \
                f"{arch}: features must be a frozenset"
            if entry.vendor == "cuda":
                assert entry.host_define == "WITH_CUDA", \
                    f"{arch}: host_define={entry.host_define!r}"
                assert entry.nvcc_gencode, f"{arch}: empty nvcc_gencode"
                assert entry.cutlass_arch is not None, \
                    f"{arch}: cutlass_arch must be set for CUDA"
                assert entry.warp_size == 32, f"{arch}: warp_size!=32"
            elif entry.vendor == "hip":
                assert entry.host_define == "WITH_HIP", \
                    f"{arch}: host_define={entry.host_define!r}"
                assert entry.hipcc_offload_arch, \
                    f"{arch}: empty hipcc_offload_arch"
                assert entry.warp_size in (32, 64), \
                    f"{arch}: warp_size={entry.warp_size!r}"
            else:  # pallas
                assert entry.host_define is None, \
                    f"{arch}: pallas host_define must be None"
                assert entry.warp_size is None
                assert entry.max_threads_per_block is None

        # Alias resolution: sm_90 → sm_90a (same object), etc.
        assert ARCH_TABLE["sm_90"] is ARCH_TABLE["sm_90a"], \
            "sm_90 alias must resolve to sm_90a"
        assert ARCH_TABLE["sm_100"] is ARCH_TABLE["sm_100a"]
        assert ARCH_TABLE["sm_103"] is ARCH_TABLE["sm_103a"]
        assert ARCH_TABLE["sm_120"] is ARCH_TABLE["sm_120a"]

        # get_arch_entry must work for aliases too.
        assert get_arch_entry("sm_90").macro == "SG_BUILD_ARCH_SM90"
        assert get_arch_entry("sm_120a").vendor == "cuda"

        # Legacy 6-key derived view must contain the expected keys.
        legacy_keys = {"vendor", "subdir", "launcher_glob", "model_glob",
                       "macro", "host_define"}
        for arch in ARCH_TABLE:
            assert arch in ARCH_INFO, f"ARCH_INFO missing {arch}"
            assert set(ARCH_INFO[arch].keys()) == legacy_keys, \
                f"ARCH_INFO[{arch}] keys {sorted(ARCH_INFO[arch].keys())} " \
                f"!= {sorted(legacy_keys)}"
            # And the values must match the ArchEntry source.
            assert ARCH_INFO[arch]["vendor"] == ARCH_TABLE[arch].vendor
            assert ARCH_INFO[arch]["macro"] == ARCH_TABLE[arch].macro

        # Spot-check min_toolchain_version values.
        assert get_arch_entry("sm_90a").min_toolchain_version == (12, 0)
        assert get_arch_entry("sm_120a").min_toolchain_version == (12, 8)
        assert get_arch_entry("gfx942").min_toolchain_version == (6, 0)
        assert get_arch_entry("tpu_v6e").min_toolchain_version == (0, 4, 30)

        # Feature-flag spot-checks.
        assert "tma" in get_arch_entry("sm_90a").features
        assert "fp4" in get_arch_entry("sm_100a").features
        assert "tma" not in get_arch_entry("sm_120a").features  # consumer Blackwell
        assert "fp8" in get_arch_entry("sm_89").features
        assert "fp8_mfma" in get_arch_entry("gfx942").features
        assert "fp4_mfma" in get_arch_entry("gfx950").features
        assert "wmma" in get_arch_entry("gfx1100").features
        assert "sparsecore" in get_arch_entry("tpu_v5p").features

        # Existing search-space builders are wired into ARCH_TABLE.
        assert ARCH_TABLE["sm_90a"].search_space_builder is _sm90_full_space
        assert ARCH_TABLE["gfx942"].search_space_builder is _gfx942_full_space

    def test_arch_table_gencode_format():
        """nvcc_gencode entries must follow the documented format and include
        BOTH a SASS pair and a PTX fallback for each CUDA arch."""
        for arch, entry in ARCH_TABLE.items():
            if entry.vendor != "cuda":
                assert entry.nvcc_gencode == [], \
                    f"{arch}: non-CUDA entries must have empty nvcc_gencode"
                continue
            assert len(entry.nvcc_gencode) == 2, \
                f"{arch}: expected 2 -gencode flags, got {entry.nvcc_gencode}"
            sass, ptx = entry.nvcc_gencode
            assert sass.startswith("-gencode=arch=compute_"), sass
            assert ptx.startswith("-gencode=arch=compute_"), ptx
            # PTX fallback compute_XX,code=compute_XX (non-"a" — drivers
            # forward to newer hw via JIT).
            assert ",code=compute_" in ptx, \
                f"{arch}: PTX fallback malformed: {ptx}"

    _run("arch_table_completeness", test_arch_table_completeness)
    _run("arch_table_gencode_format", test_arch_table_gencode_format)

    sys.stdout.write("[self-test] kernel_registry\n")

    def test_kernel_registry_importable():
        from grokking_optimizers import kernel_registry  # noqa: F401

    def test_shape_class_buckets():
        from grokking_optimizers.kernel_registry import _shape_class
        assert _shape_class((100,)) == "tiny"
        assert _shape_class(()) == "tiny"        # scalar == 1 element
        # 256 elements is still inside the < 1024 tiny bucket.
        assert _shape_class((256,)) == "tiny"
        assert _shape_class((4096,)) == "small"
        assert _shape_class((512, 512)) == "medium"
        assert _shape_class((4096, 4096)) in ("large", "huge")
        assert _shape_class((1 << 30,)) == "huge"

    def test_kernel_registry_construct_cuda():
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            reg = kernel_registry.KernelRegistry("sm_90a", td)
            assert reg.vendor == "cuda"
            assert reg.cache_dir == td
        finally:
            shutil.rmtree(td)

    def test_kernel_registry_rejects_pallas():
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            try:
                kernel_registry.KernelRegistry("tpu_v5p", td)
            except kernel_registry.RegistryError:
                return
            raise AssertionError(
                "expected RegistryError for pallas arch tpu_v5p")
        finally:
            shutil.rmtree(td)

    def test_kernel_registry_rejects_unknown_arch():
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            try:
                kernel_registry.KernelRegistry("sm_bogus", td)
            except kernel_registry.RegistryError:
                return
            raise AssertionError("expected RegistryError for unknown arch")
        finally:
            shutil.rmtree(td)

    def test_kernel_registry_nvrtc_compile_or_skip():
        """NVRTC compile path — skip cleanly on hosts without cuda-python."""
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            reg = kernel_registry.KernelRegistry("sm_90a", td)
            try:
                handle = reg.dispatch("adamw", "fp32", (4096,))
            except kernel_registry.RegistryError as exc:
                # Acceptable on CPU-only hosts without cuda-python.
                sys.stdout.write(f"    [nvrtc skipped] {exc}\n")
                return
            cubin_path = handle.cubin_path
            assert cubin_path.exists()
            assert cubin_path.stat().st_size > 0
            sys.stdout.write(
                f"    [nvrtc compiled] {cubin_path.stat().st_size}B cubin\n")
        finally:
            shutil.rmtree(td)

    def test_initialize_registry_disabled_by_default():
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            spec = BuildSpec(optimizer="adamw", model="mamba", arch="sm_90a",
                             out_dir=Path(td))
            assert kernel_registry.initialize_registry(spec) is None
        finally:
            shutil.rmtree(td)

    def test_loaded_kernel_call_or_skip():
        """Live NVRTC → cuModuleLoadData → cuLaunchKernel round-trip.

        Skips cleanly if cuda-python or a CUDA-capable GPU isn't available.
        """
        # Gate 1: cuda-python
        try:
            try:
                from cuda.bindings import driver as _drv  # noqa: F401
                from cuda.bindings import nvrtc as _nvrtc  # noqa: F401
            except ImportError:
                from cuda import cuda as _drv  # noqa: F401
                from cuda import nvrtc as _nvrtc  # noqa: F401
        except ImportError:
            sys.stdout.write("    [nvrtc-live] cuda-python unavailable — skip\n")
            return
        # Gate 2: GPU visibility (be defensive — torch.cuda is the easy probe).
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                sys.stdout.write("    [nvrtc-live] no CUDA device — skip\n")
                return
        except Exception:
            sys.stdout.write("    [nvrtc-live] torch probe failed — skip\n")
            return
        # Gate 3: ARCH_TABLE entry for the local SKU.
        try:
            major, minor = torch.cuda.get_device_capability(0)
        except Exception as exc:
            sys.stdout.write(
                f"    [nvrtc-live] get_device_capability failed: {exc} — skip\n")
            return
        sm = f"sm_{major}{minor}"
        if sm not in ARCH_TABLE:
            # Try the "a" variant.
            if (sm + "a") in ARCH_TABLE:
                sm = sm + "a"
            else:
                sys.stdout.write(
                    f"    [nvrtc-live] {sm} not in ARCH_TABLE — skip\n")
                return
        # Compile + load + launch.
        from grokking_optimizers import kernel_registry
        td = Path(tempfile.mkdtemp())
        try:
            reg = kernel_registry.KernelRegistry(sm, td)
            handle = reg.dispatch("copy", "fp32", (64,))
            # Allocate device memory via torch as a convenience wrapper.
            n = 64
            d_in = torch.arange(n, dtype=torch.float32, device="cuda")
            d_out = torch.zeros(n, dtype=torch.float32, device="cuda")
            handle(d_out.data_ptr(), d_in.data_ptr(), n,
                   grid=((n + 31) // 32, 1, 1), block=(32, 1, 1))
            torch.cuda.synchronize()
            assert torch.allclose(d_out, d_in), \
                f"copy kernel output mismatch: out={d_out[:4]} in={d_in[:4]}"
            sys.stdout.write(
                f"    [nvrtc-live] {sm} copy kernel round-tripped "
                f"{n} fp32 elements\n")
        finally:
            shutil.rmtree(td)

    _run("kernel_registry_importable", test_kernel_registry_importable)
    _run("kernel_registry_shape_buckets", test_shape_class_buckets)
    _run("kernel_registry_construct_cuda", test_kernel_registry_construct_cuda)
    _run("kernel_registry_rejects_pallas", test_kernel_registry_rejects_pallas)
    _run("kernel_registry_rejects_unknown_arch",
         test_kernel_registry_rejects_unknown_arch)
    _run("kernel_registry_nvrtc_compile_or_skip",
         test_kernel_registry_nvrtc_compile_or_skip)
    _run("kernel_registry_initialize_disabled",
         test_initialize_registry_disabled_by_default)
    _run("loaded_kernel_call_or_skip", test_loaded_kernel_call_or_skip)

    # ----- Stream 11: compile_config -----
    sys.stdout.write("[self-test] compile_config\n")

    def test_compile_config_default_loads():
        from grokking_optimizers import compile_config
        cfg = compile_config.load_config()
        assert isinstance(cfg, dict), type(cfg)
        assert "project" in cfg
        sys.stdout.write(
            f"    (default config has {len(cfg)} sections)\n")

    def test_compile_config_cwd_override():
        from grokking_optimizers import compile_config
        with tempfile.TemporaryDirectory() as td:
            cwd_cfg = Path(td) / "compile_config.toml"
            cwd_cfg.write_text(
                "[codegen]\n"
                "enable_emitter = true\n"
                "[archs]\n"
                'default = "sm_100a"\n'
            )
            saved_cwd = os.getcwd()
            try:
                os.chdir(td)
                cfg2 = compile_config.load_config()
                assert cfg2.get("codegen", {}).get("enable_emitter") is True
                assert cfg2.get("archs", {}).get("default") == "sm_100a"
            finally:
                os.chdir(saved_cwd)

    def test_compile_config_apply_noop_on_empty():
        from grokking_optimizers import compile_config

        class _MockSpec:
            enable_emitter = False
            enable_runtime_specialization = False
            enable_device_pgo = False
            strict_numerics = False
            prune_after_autotune = True
            prune_max_age_days = 30
            prune_keep_top_n = 100

        ms = _MockSpec()
        compile_config.apply_to_buildspec(ms, {})
        assert ms.enable_emitter is False
        assert ms.enable_runtime_specialization is False
        assert ms.enable_device_pgo is False
        assert ms.strict_numerics is False

    _run("compile_config_default_loads", test_compile_config_default_loads)
    _run("compile_config_cwd_override", test_compile_config_cwd_override)
    _run("compile_config_apply_noop_on_empty",
         test_compile_config_apply_noop_on_empty)

    # ─────────────────────────────────────────────────────────────────
    # Stream 6 — codegen / Jinja2 kernel emitter
    # ─────────────────────────────────────────────────────────────────
    sys.stdout.write("[self-test] codegen\n")

    def test_codegen_import():
        from grokking_optimizers import codegen as _cg
        assert hasattr(_cg, "emit_variant_source")
        assert hasattr(_cg, "find_template")
        assert hasattr(_cg, "validate_template_render")
        assert hasattr(_cg, "validate_with_nvcc_dryrun")

    def test_codegen_templates_dir():
        # Templates are now bundled as a dict inside compile.py — the
        # legacy TEMPLATE_ROOT constant still exists but the dict is the
        # source of truth.
        from grokking_optimizers import codegen as _cg
        assert isinstance(_cg._BUNDLED_TEMPLATES, dict)
        assert len(_cg._BUNDLED_TEMPLATES) >= 1, \
            "no bundled templates registered in _BUNDLED_TEMPLATES"

    def test_codegen_template_count():
        from grokking_optimizers import codegen as _cg
        names = list(_cg._BUNDLED_TEMPLATES.keys())
        assert len(names) >= 3, \
            f"expected >= 3 bundled templates, found {len(names)}: {names}"

    def test_codegen_jinja2_or_skip():
        # Gracefully no-op if jinja2 isn't available — the rest of
        # the suite then prints SKIPPED for the render checks below.
        try:
            import jinja2  # noqa: F401
        except ImportError:
            sys.stdout.write(
                "    (jinja2 unavailable; skipping render checks)\n")

    def test_codegen_adamw_sm90_renders():
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # silently skip — graceful degradation
        from grokking_optimizers import codegen as _cg
        ok, msg = _cg.validate_template_render("adamw", "sm_90a")
        assert ok, f"adamw/sm_90a render failed: {msg}"

    def test_codegen_emit_returns_path():
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # skip
        from grokking_optimizers import codegen as _cg
        sm90_dims = build_full_search_space()["sm_90"]["dims"][:3]
        cfg = {d["name"]: d["values"][0] for d in sm90_dims}
        td = Path(tempfile.mkdtemp())
        try:
            emitted, residual = _cg.emit_variant_source(
                cfg, sm90_dims, "adamw", "sm_90a", td)
            assert emitted.exists(), f"emitted file missing: {emitted}"
            assert emitted.suffix == ".cu", \
                f"expected .cu, got {emitted.suffix}"
            assert isinstance(residual, list)
            body = emitted.read_text()
            assert "AUTO-GENERATED" in body
            assert f"SG_BLOCK_SIZE   {cfg['block']}" in body \
                or f"SG_BLOCK_SIZE  {cfg['block']}" in body, \
                f"emitted source missing baked block size: {body[:400]}"
            # Re-emit with same config → cache hit, same path.
            emitted2, _ = _cg.emit_variant_source(
                cfg, sm90_dims, "adamw", "sm_90a", td)
            assert emitted2 == emitted, \
                f"cache miss on identical config: {emitted2} vs {emitted}"
        finally:
            shutil.rmtree(td)

    def test_codegen_nvcc_dryrun_graceful():
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # skip
        from grokking_optimizers import codegen as _cg
        td = Path(tempfile.mkdtemp())
        try:
            sm90_dims = build_full_search_space()["sm_90"]["dims"][:3]
            cfg = {d["name"]: d["values"][0] for d in sm90_dims}
            emitted, _ = _cg.emit_variant_source(
                cfg, sm90_dims, "adamw", "sm_90a", td)
            ok, msg = _cg.validate_with_nvcc_dryrun(emitted)
            # ok must be True either because nvcc validated the source
            # OR because nvcc is unavailable (CI host). The "unavailable"
            # branch is the expected path on this CPU-only worktree.
            assert ok, f"nvcc dryrun failed: {msg}"
        finally:
            shutil.rmtree(td)

    def test_codegen_find_template_fallback():
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # skip
        from grokking_optimizers import codegen as _cg
        # find_template now returns the dict KEY (a string), not a Path.
        p = _cg.find_template("adamw", "sm_90a")
        assert p == "adamw_sm_90a.cu.j2", p
        # Unknown optimizer → None (caller raises CodegenError).
        assert _cg.find_template("nonexistent_opt", "sm_90a") is None

    def test_codegen_unknown_template_raises():
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # skip
        from grokking_optimizers import codegen as _cg
        td = Path(tempfile.mkdtemp())
        try:
            try:
                _cg.emit_variant_source(
                    {}, [], "nonexistent_opt", "sm_90a", td)
                raise AssertionError("expected CodegenError")
            except _cg.CodegenError:
                pass
        finally:
            shutil.rmtree(td)

    def test_codegen_variant_macros_hook():
        """Round-trip through _variant_macros with enable_emitter=True."""
        try:
            import jinja2  # noqa: F401
        except ImportError:
            return  # skip
        td = Path(tempfile.mkdtemp())
        try:
            spec = BuildSpec(
                optimizer="adamw", model="mamba", arch="sm_90a",
                out_dir=td, enable_emitter=True)
            sm90_dims = build_full_search_space()["sm_90"]["dims"][:3]
            cfg = {d["name"]: d["values"][0] for d in sm90_dims}
            # Device path triggers the emitter and stashes a path.
            flags = _variant_macros(cfg, sm90_dims, "device", spec=spec)
            assert isinstance(flags, list)
            ck = config_key(cfg)
            assert ck in spec._emitted_sources, \
                f"emitted_sources missing {ck}: {spec._emitted_sources}"
            assert spec._emitted_sources[ck].exists()
            # When enable_emitter=False, behavior is unchanged.
            spec2 = BuildSpec(optimizer="adamw", model="mamba",
                              arch="sm_90a", out_dir=td)
            flags2 = _variant_macros(cfg, sm90_dims, "device", spec=spec2)
            assert spec2._emitted_sources == {}
            assert isinstance(flags2, list)
        finally:
            shutil.rmtree(td)

    def test_cutlass_gemm_emitter_or_skip():
        """CUTLASS GEMM emitter: skip cleanly without cutlass-python."""
        from grokking_optimizers import codegen as _cg
        try:
            import cutlass  # type: ignore  # noqa: F401
        except ImportError:
            # Expected on CPU-only / minimal hosts. Verify the stub raises
            # the right error.
            td = Path(tempfile.mkdtemp())
            try:
                try:
                    _cg.emit_cutlass_gemm_variants(
                        "sm_90a", (256, 256, 256), "fp16", td)
                    raise AssertionError("expected CodegenError without cutlass-python")
                except _cg.CodegenError as exc:
                    assert "cutlass" in str(exc).lower()
            finally:
                shutil.rmtree(td)
            return
        # cutlass-python IS installed — assert a real variant is emitted.
        td = Path(tempfile.mkdtemp())
        try:
            variants = _cg.emit_cutlass_gemm_variants(
                "sm_90a", (256, 256, 256), "fp16", td)
            assert len(variants) >= 1, f"expected >= 1 variant, got {len(variants)}"
            path, meta = variants[0]
            assert path.exists() and path.suffix == ".cu"
            assert "tile" in meta or "tile_shape" in meta or "tile_description" in meta
            src = path.read_text()
            assert "GemmUniversal" in src or "Gemm" in src
            assert "extern \"C\"" in src or "extern \"C\"" in src.replace('\\"', '"')
            # Unsupported arch must raise.
            try:
                _cg.emit_cutlass_gemm_variants("sm_75", (256, 256, 256), "fp16", td)
                raise AssertionError("expected CodegenError for unsupported arch sm_75")
            except _cg.CodegenError:
                pass
        finally:
            shutil.rmtree(td)

    _run("codegen_import", test_codegen_import)
    _run("codegen_templates_dir", test_codegen_templates_dir)
    _run("codegen_template_count", test_codegen_template_count)
    _run("codegen_jinja2_probe", test_codegen_jinja2_or_skip)
    _run("codegen_adamw_sm90_renders", test_codegen_adamw_sm90_renders)
    _run("codegen_emit_returns_path", test_codegen_emit_returns_path)
    _run("codegen_nvcc_dryrun_graceful", test_codegen_nvcc_dryrun_graceful)
    _run("codegen_find_template_fallback", test_codegen_find_template_fallback)
    _run("codegen_unknown_template_raises", test_codegen_unknown_template_raises)
    _run("codegen_variant_macros_hook", test_codegen_variant_macros_hook)
    _run("cutlass_gemm_emitter_or_skip", test_cutlass_gemm_emitter_or_skip)

    # ── Stream 12: toolchain bootstrap ──────────────────────────────
    sys.stdout.write("[self-test] toolchain_bootstrap\n")

    def test_bootstrap_helpers_importable():
        """All new bootstrap helpers + version-targeting functions are
        importable from the module — guards against typos / missed exports."""
        from grokking_optimizers.compile import (  # noqa: F401
            bootstrap_rocm_toolkit,
            bootstrap_jax_tpu,
            bootstrap_toolchain,
            _target_cuda_version_for_arch,
            _target_rocm_version_for_arch,
        )

    def test_target_version_lookup():
        """Per-arch min-version lookup matches ARCH_TABLE entries."""
        # ROCm: gfx950 needs 6.2; gfx1200 needs 7.0.
        assert _target_rocm_version_for_arch("gfx950") == (6, 2), \
            _target_rocm_version_for_arch("gfx950")
        assert _target_rocm_version_for_arch("gfx1200") == (7, 0), \
            _target_rocm_version_for_arch("gfx1200")
        # Non-HIP arches return (0, 0) so callers can branch cleanly.
        assert _target_rocm_version_for_arch("sm_90a") == (0, 0)
        assert _target_rocm_version_for_arch("tpu_v5p") == (0, 0)

    def test_cuda_target_respects_arch_min():
        """CUDA targeting picks max(arch_min, torch_cuda). For each arch the
        returned version must be >= ARCH_TABLE[arch].min_toolchain_version."""
        for arch in ("sm_75", "sm_80", "sm_90a", "sm_100a", "sm_103a",
                     "sm_120a"):
            tv = _target_cuda_version_for_arch(arch)
            need = ARCH_TABLE[arch].min_toolchain_version
            assert tv[0] >= need[0] or (
                tv[0] == need[0] and tv[1] >= need[1]), \
                f"{arch}: target {tv} below min {need}"
        # Non-CUDA arches return (0, 0).
        assert _target_cuda_version_for_arch("gfx942") == (0, 0)
        assert _target_cuda_version_for_arch("tpu_v5p") == (0, 0)

    def test_preflight_emits_judgment_per_arch():
        """_preflight_toolchain must emit at least one PASS/FAIL line per
        arch across all three vendors so CI can grep for failures."""
        for arch in ("sm_75", "sm_90a", "sm_120a", "gfx942", "gfx1200",
                     "tpu_v5p"):
            lines = _preflight_toolchain(arch)
            has_judgment = any(
                ("PASS" in ln or "FAIL" in ln) and f"arch={arch}" in ln
                for ln in lines
            )
            assert has_judgment, \
                f"{arch}: no per-arch PASS/FAIL in preflight: {lines}"

    def test_bootstrap_toolchain_dispatch_no_crash():
        """bootstrap_toolchain must dispatch by vendor without crashing on
        a host that doesn't actually have nvcc/hipcc/jax[tpu]. Functions
        return False on missing installers — that's the expected outcome
        in a sandboxed self-test."""
        import io
        buf = io.StringIO()
        # The cuda path on a host with nvcc already returns True quickly
        # (it short-circuits on shutil.which("nvcc")); without nvcc it
        # tries package managers and may return False — both are OK here.
        # We only assert "no exception escapes".
        for arch in ("sm_75", "gfx942", "tpu_v5p"):
            try:
                # Skip live bootstrap of TPU/ROCm when their tools are
                # missing — these self-tests must not actually install.
                # Probe-only behaviour: shutil.which() short-circuits.
                if arch == "sm_75" and not shutil.which("nvcc"):
                    # bootstrap_cuda_toolkit() would try apt — skip the
                    # actual install attempt in self-test.
                    continue
                if arch == "gfx942" and not shutil.which("hipcc"):
                    continue
                # tpu_v5p with no jax installed: bootstrap_jax_tpu would
                # try to pip-install — skip too.
                if arch == "tpu_v5p":
                    try:
                        import jax  # noqa: F401
                        devs = jax.devices()
                        if not any(getattr(d, "platform", "") == "tpu"
                                   for d in devs):
                            continue
                    except Exception:
                        continue
                _ = bootstrap_toolchain(arch, buf)
            except Exception as exc:
                raise AssertionError(
                    f"bootstrap_toolchain({arch}) crashed: "
                    f"{type(exc).__name__}: {exc}") from exc
        # And: unknown arch returns False, doesn't crash.
        assert bootstrap_toolchain("not_a_real_arch", buf) is False

    _run("bootstrap_helpers_importable", test_bootstrap_helpers_importable)
    _run("target_version_lookup", test_target_version_lookup)
    _run("cuda_target_respects_arch_min", test_cuda_target_respects_arch_min)
    _run("preflight_emits_judgment_per_arch",
         test_preflight_emits_judgment_per_arch)
    _run("bootstrap_toolchain_dispatch_no_crash",
         test_bootstrap_toolchain_dispatch_no_crash)

    sys.stdout.write("[self-test] pallas\n")

    def test_pallas_timer_importable_without_jax():
        # PallasTimer must be importable on hosts that don't have JAX.
        # We can't really uninstall JAX here, but we can verify the class
        # constructs and only raises when time_config() is actually called.
        td = Path(tempfile.mkdtemp())
        try:
            t = PallasTimer(td / "nonexistent_launcher.py", "adamw")
            assert t.optimizer == "adamw"
            assert t._launch_fn is None       # lazy — not yet loaded
            assert t._cached_jit == {}
            # time_config should raise RuntimeError, not crash on import.
            try:
                t.time_config({})
            except RuntimeError:
                pass
            else:
                # If JAX is installed AND the launcher path doesn't exist,
                # _load_launcher() raises RuntimeError("launcher not found").
                # Either way RuntimeError is the expected failure mode.
                raise AssertionError(
                    "time_config should have raised RuntimeError")
        finally:
            shutil.rmtree(td)

    def test_pallas_build_aot_noop():
        # build_aot must return the Path("pallas-noop") sentinel for tpu_v5p
        # without trying to load nvcc or torch.cpp_extension.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            spec = BuildSpec(
                optimizer="adamw", model="mamba3", arch="tpu_v5p",
                out_dir=tdp,
            )

            class _DummyReport:
                def write(self, s): pass
                def flush(self): pass

            cache_path = tdp / "cache.json"
            cache = CompileCache(cache_path)
            out = build_aot(spec, cache, _DummyReport())
            assert out is not None, "build_aot returned None for pallas"
            assert str(out) == "pallas-noop", \
                f"expected pallas-noop, got {out!r}"

    def test_pallas_publish_handles_sentinel():
        # _publish_aot_artifact must accept the sentinel without copying.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            spec = BuildSpec(
                optimizer="adamw", model="mamba3", arch="tpu_v5p",
                out_dir=tdp, aot_artifact_dir=tdp / "published",
            )

            class _DummyReport:
                def write(self, s): pass
                def flush(self): pass

            out = _publish_aot_artifact(spec, Path("pallas-noop"),
                                        _DummyReport())
            assert str(out) == "pallas-noop"

    _run("pallas_timer_importable", test_pallas_timer_importable_without_jax)
    _run("pallas_build_aot_noop", test_pallas_build_aot_noop)
    _run("pallas_publish_handles_sentinel", test_pallas_publish_handles_sentinel)

    # ── Stream 10 — numerical / differential validation ──────────────
    sys.stdout.write("[self-test] numerical_validation\n")

    def test_compare_outputs_tolerant():
        import numpy as np
        td = Path(tempfile.mkdtemp())
        try:
            ref = td / "ref.npy"
            cand_ok = td / "ok.npy"
            np.save(ref,     np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.save(cand_ok, np.array([1.0, 2.0, 3.0 + 1e-7],
                                      dtype=np.float32))
            status, _ = _compare_outputs(ref, cand_ok, "fp32")
            assert status in ("ok", "deterministic"), status
        finally:
            shutil.rmtree(td)

    def test_compare_outputs_fail():
        import numpy as np
        td = Path(tempfile.mkdtemp())
        try:
            ref = td / "ref.npy"
            cand_bad = td / "bad.npy"
            np.save(ref,      np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.save(cand_bad, np.array([1.0, 2.0, 100.0], dtype=np.float32))
            status, _ = _compare_outputs(ref, cand_bad, "fp32")
            assert status == "numerical_fail", status
        finally:
            shutil.rmtree(td)

    def test_compare_outputs_deterministic():
        import numpy as np
        td = Path(tempfile.mkdtemp())
        try:
            ref = td / "ref.npy"
            cand_det = td / "det.npy"
            np.save(ref,      np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.save(cand_det, np.array([1.0, 2.0, 3.0], dtype=np.float32))
            status, _ = _compare_outputs(ref, cand_det, "fp32")
            assert status == "deterministic", status
        finally:
            shutil.rmtree(td)

    def test_compare_outputs_shape_mismatch():
        import numpy as np
        td = Path(tempfile.mkdtemp())
        try:
            ref = td / "ref.npy"
            cand = td / "cand.npy"
            np.save(ref,  np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.save(cand, np.array([1.0, 2.0],      dtype=np.float32))
            status, rel = _compare_outputs(ref, cand, "fp32")
            assert status == "numerical_fail" and rel == float("inf")
        finally:
            shutil.rmtree(td)

    def test_tolerances_table_shape():
        # Both narrow and wide dtypes must be present and (rtol, atol)
        # tuples. fp4 must be looser than fp32.
        for k in ("fp32", "fp16", "bf16", "fp8", "fp4"):
            assert k in TOLERANCES, k
            rtol, atol = TOLERANCES[k]
            assert rtol > 0 and atol > 0
        assert TOLERANCES["fp4"][0] > TOLERANCES["fp32"][0]

    def test_pick_winner_filters_numerical_fail():
        trials = [
            {"status": "ok", "value_ms": 0.5, "timing_ms": 0.5,
             "numerical_status": "deterministic",
             "stage": "b", "trial_num": 0, "config": {}, "error": None},
            {"status": "ok", "value_ms": 0.3, "timing_ms": 0.3,
             "numerical_status": "numerical_fail",
             "stage": "b", "trial_num": 1, "config": {}, "error": None},
            {"status": "ok", "value_ms": 0.4, "timing_ms": 0.4,
             "numerical_status": "ok",
             "stage": "b", "trial_num": 2, "config": {}, "error": None},
        ]
        w = pick_winner(trials)
        assert w is not None and w["timing_ms"] == 0.4, w
        w_strict = pick_winner(trials, strict_numerics=True)
        assert w_strict is not None and w_strict["timing_ms"] == 0.5, w_strict

    def test_pick_winner_strict_returns_none_when_no_deterministic():
        trials = [
            {"status": "ok", "timing_ms": 0.3,
             "numerical_status": "ok",
             "stage": "b", "trial_num": 0, "config": {}},
            {"status": "ok", "timing_ms": 0.4,
             "numerical_status": "non_deterministic",
             "stage": "b", "trial_num": 1, "config": {}},
        ]
        # Non-strict mode picks the faster "ok" trial.
        w = pick_winner(trials)
        assert w is not None and w["timing_ms"] == 0.3
        # Strict mode requires "deterministic" — neither qualifies.
        assert pick_winner(trials, strict_numerics=True) is None

    def test_make_trial_record_pulls_numerical_status():
        # Prime the sidecar.
        cfg = {"block": 128, "vec": 2, "unroll": 4}
        ck = config_key(cfg)
        _LAST_NUMERICAL_STATUS[ck] = "ok"
        try:
            rec = _make_trial_record("tpe", 0, cfg,
                                     {"timing_ms": 1.0, "min_ms": 0.9,
                                      "max_ms": 1.1, "n": 21})
            assert rec["numerical_status"] == "ok", rec
            # Unknown config_key falls back to "skipped".
            rec2 = _make_trial_record("tpe", 1, {"block": 999, "vec": 1,
                                                 "unroll": 1},
                                      {"timing_ms": 1.0, "min_ms": 0.9,
                                       "max_ms": 1.1, "n": 21})
            assert rec2["numerical_status"] == "skipped", rec2
        finally:
            _LAST_NUMERICAL_STATUS.pop(ck, None)

    _run("tolerances_table_shape", test_tolerances_table_shape)
    _run("compare_outputs_tolerant", test_compare_outputs_tolerant)
    _run("compare_outputs_numerical_fail", test_compare_outputs_fail)
    _run("compare_outputs_deterministic", test_compare_outputs_deterministic)
    _run("compare_outputs_shape_mismatch", test_compare_outputs_shape_mismatch)
    _run("pick_winner_excludes_numerical_fail",
         test_pick_winner_filters_numerical_fail)
    _run("pick_winner_strict_requires_deterministic",
         test_pick_winner_strict_returns_none_when_no_deterministic)
    _run("make_trial_record_propagates_numerical_status",
         test_make_trial_record_pulls_numerical_status)

    sys.stdout.write(f"\n[self-test] {passed} passed, {failures} failed\n")
    return 1 if failures else 0


# ==============================================================================
# CONSOLIDATED MODULES — formerly grokking_optimizers/codegen.py,
# kernel_registry.py, device_profiling.py, compile_config.py + templates/*.j2.
#
# Inlined here so the entire pipeline lives in a single compile.py file. The
# four legacy module paths are also registered in sys.modules at the bottom
# of this block, so legacy callers using
# ``from grokking_optimizers.codegen import emit_variant_source`` (etc.) keep
# resolving without code changes.
# ==============================================================================

import re as _re_consolidated  # used by toml fallback parser + nsys stall parser

# ------------------------------------------------------------------------------
# Bundled Jinja2 templates (formerly grokking_optimizers/templates/*.j2)
# ------------------------------------------------------------------------------

_BUNDLED_TEMPLATES: Dict[str, str] = {
    "adamw_sm_90a.cu.j2": r"""// AUTO-GENERATED by grokking_optimizers.codegen — DO NOT EDIT
// Optimizer:   AdamW
// Arch:        {{ arch }} ({{ display_name }})
// Vendor:      {{ vendor }}
// Generated config (baked in at template-render time):
//   block        = {{ block }}
//   vec          = {{ vec }}
//   unroll       = {{ unroll }}
//   num_stages   = {{ num_stages | default(2) }}
{% if 'cluster_shape' in config %}
//   cluster_shape = {{ cluster_shape }}
{% endif %}
{% if 'swizzle' in config %}
//   swizzle      = {{ swizzle }}
{% endif %}
//
// Hopper-specific knobs auto-emitted from the arch features
// ({{ features | join(', ') }})
{% if 'wgmma' in features %}
#define SG_USE_WGMMA 1
{% endif %}
{% if 'tma' in features %}
#define SG_USE_TMA 1
{% endif %}
{% if 'cluster' in features %}
#define SG_USE_CLUSTER 1
{% endif %}

#define SG_BLOCK_SIZE   {{ block }}
#define SG_VEC_WIDTH    {{ vec }}
#define SG_UNROLL       {{ unroll }}
#define SG_NUM_STAGES   {{ num_stages | default(2) }}
{% if 'async_depth' in config %}
#define SG_ASYNC_DEPTH  {{ async_depth }}
{% endif %}

// The actual per-element AdamW math + grid-stride loop primitives live
// in the project headers below; the emitted source only injects the
// tuned constants so the compiler can specialise the kernel body.
#include "csrc/algorithms/adamw.h"

namespace sg::sm90 {

// Tuned launch bounds — the compiler uses this to pick a register
// budget that fits SG_BLOCK_SIZE warps without spilling.
template <int BlockSize = SG_BLOCK_SIZE, int VecWidth = SG_VEC_WIDTH>
__global__ __launch_bounds__(BlockSize)
void adamw_step_kernel(float* __restrict__ p,
                       float* __restrict__ m,
                       float* __restrict__ v,
                       const float* __restrict__ g,
                       float lr, float beta1, float beta2,
                       float eps, float wd, float bc1, float bc2,
                       int n) {
    constexpr int kVec = VecWidth;
    const int gid = blockIdx.x * BlockSize + threadIdx.x;
    const int stride = BlockSize * gridDim.x;
    #pragma unroll SG_UNROLL
    for (int i = gid * kVec; i < n; i += stride * kVec) {
        #pragma unroll
        for (int k = 0; k < kVec; ++k) {
            const int idx = i + k;
            if (idx >= n) break;
            adamw_step_elem(p + idx, m + idx, v + idx, g[idx],
                            lr, beta1, beta2, eps, wd, bc1, bc2);
        }
    }
}

extern "C" void launch_adamw_step(float* p, float* m, float* v,
                                  const float* g, float lr, float beta1,
                                  float beta2, float eps, float wd,
                                  float bc1, float bc2, int n,
                                  cudaStream_t stream) {
    const int threads = SG_BLOCK_SIZE;
    const int items_per_block = threads * SG_VEC_WIDTH;
    const int blocks = (n + items_per_block - 1) / items_per_block;
    adamw_step_kernel<SG_BLOCK_SIZE, SG_VEC_WIDTH>
        <<<blocks, threads, 0, stream>>>(p, m, v, g, lr, beta1, beta2,
                                         eps, wd, bc1, bc2, n);
}

} // namespace sg::sm90
""",

    "adamw_generic.cu.j2": r"""// AUTO-GENERATED by grokking_optimizers.codegen — DO NOT EDIT
// Optimizer:   AdamW (generic CUDA fallback)
// Arch:        {{ arch }} ({{ display_name }})
// Vendor:      {{ vendor }}
//
// This template targets pre-Hopper CUDA (sm_75 / sm_80 / sm_86 / sm_89).
// No TMA, wgmma, or cluster launch — just a clean block-strided loop
// with the autotuned block / vec / unroll constants baked in.
//
// Tuned config:
//   block      = {{ block | default(256) }}
//   vec        = {{ vec | default(4) }}
//   unroll     = {{ unroll | default(4) }}
//   num_stages = {{ num_stages | default(2) }}

#define SG_BLOCK_SIZE  {{ block | default(256) }}
#define SG_VEC_WIDTH   {{ vec | default(4) }}
#define SG_UNROLL      {{ unroll | default(4) }}

#include "csrc/algorithms/adamw.h"

namespace sg::generic {

__global__ __launch_bounds__(SG_BLOCK_SIZE)
void adamw_step_kernel(float* __restrict__ p,
                       float* __restrict__ m,
                       float* __restrict__ v,
                       const float* __restrict__ g,
                       float lr, float beta1, float beta2,
                       float eps, float wd, float bc1, float bc2,
                       int n) {
    const int gid = blockIdx.x * SG_BLOCK_SIZE + threadIdx.x;
    const int stride = SG_BLOCK_SIZE * gridDim.x;
    #pragma unroll SG_UNROLL
    for (int i = gid * SG_VEC_WIDTH; i < n; i += stride * SG_VEC_WIDTH) {
        #pragma unroll
        for (int k = 0; k < SG_VEC_WIDTH; ++k) {
            const int idx = i + k;
            if (idx >= n) break;
            adamw_step_elem(p + idx, m + idx, v + idx, g[idx],
                            lr, beta1, beta2, eps, wd, bc1, bc2);
        }
    }
}

extern "C" void launch_adamw_step(float* p, float* m, float* v,
                                  const float* g, float lr, float beta1,
                                  float beta2, float eps, float wd,
                                  float bc1, float bc2, int n,
                                  cudaStream_t stream) {
    const int threads = SG_BLOCK_SIZE;
    const int items_per_block = threads * SG_VEC_WIDTH;
    const int blocks = (n + items_per_block - 1) / items_per_block;
    adamw_step_kernel<<<blocks, threads, 0, stream>>>(
        p, m, v, g, lr, beta1, beta2, eps, wd, bc1, bc2, n);
}

} // namespace sg::generic
""",

    "adamw_gfx942.hip.cpp.j2": r"""// AUTO-GENERATED by grokking_optimizers.codegen — DO NOT EDIT
// Optimizer:   AdamW
// Arch:        {{ arch }} ({{ display_name }})
// Vendor:      {{ vendor }}
// Wavefront:   {{ warp_size }} lanes (CDNA3)
//
// Tuned config:
//   block          = {{ block }}
//   vec            = {{ vec }}
//   unroll         = {{ unroll }}
//   num_stages     = {{ num_stages | default(2) }}
{% if 'waves_per_eu' in config %}
//   waves_per_eu   = {{ waves_per_eu }}
{% endif %}
{% if 'lds_padding' in config %}
//   lds_padding    = {{ lds_padding }}
{% endif %}
{% if 'mfma_shape' in config %}
//   mfma_shape     = {{ mfma_shape }}
{% endif %}
{% if 'scheduler_hint' in config %}
//   scheduler_hint = {{ scheduler_hint }}
{% endif %}

#define SG_BLOCK_SIZE  {{ block }}
#define SG_VEC_WIDTH   {{ vec }}
#define SG_UNROLL      {{ unroll }}
#define SG_NUM_STAGES  {{ num_stages | default(2) }}
{% if 'mfma' in features %}
#define SG_USE_MFMA 1
{% endif %}
{% if 'fp8_mfma' in features %}
#define SG_USE_FP8_MFMA 1
{% endif %}

#include <hip/hip_runtime.h>
#include "csrc/algorithms/adamw.h"

namespace sg::gfx942 {

template <int BlockSize = SG_BLOCK_SIZE, int VecWidth = SG_VEC_WIDTH>
__global__ __launch_bounds__(BlockSize)
void adamw_step_kernel(float* __restrict__ p,
                       float* __restrict__ m,
                       float* __restrict__ v,
                       const float* __restrict__ g,
                       float lr, float beta1, float beta2,
                       float eps, float wd, float bc1, float bc2,
                       int n) {
    const int gid = blockIdx.x * BlockSize + threadIdx.x;
    const int stride = BlockSize * gridDim.x;
    #pragma unroll SG_UNROLL
    for (int i = gid * VecWidth; i < n; i += stride * VecWidth) {
        #pragma unroll
        for (int k = 0; k < VecWidth; ++k) {
            const int idx = i + k;
            if (idx >= n) break;
            adamw_step_elem(p + idx, m + idx, v + idx, g[idx],
                            lr, beta1, beta2, eps, wd, bc1, bc2);
        }
    }
}

extern "C" void launch_adamw_step(float* p, float* m, float* v,
                                  const float* g, float lr, float beta1,
                                  float beta2, float eps, float wd,
                                  float bc1, float bc2, int n,
                                  hipStream_t stream) {
    const int threads = SG_BLOCK_SIZE;
    const int items_per_block = threads * SG_VEC_WIDTH;
    const int blocks = (n + items_per_block - 1) / items_per_block;
    hipLaunchKernelGGL(
        (adamw_step_kernel<SG_BLOCK_SIZE, SG_VEC_WIDTH>),
        dim3(blocks), dim3(threads), 0, stream,
        p, m, v, g, lr, beta1, beta2, eps, wd, bc1, bc2, n);
}

} // namespace sg::gfx942
""",

    "adamw_pallas.py.j2": r'''# AUTO-GENERATED by grokking_optimizers.codegen — DO NOT EDIT
# Optimizer:   AdamW
# Arch:        {{ arch }} ({{ display_name }})
# Vendor:      {{ vendor }} (JAX/Pallas — TPU target)
#
# Tuned config baked into the Pallas kernel call:
#   block      = {{ block | default(128) }}
#   vec        = {{ vec | default(4) }}
#   unroll     = {{ unroll | default(4) }}
{% if 'num_stages' in config %}
#   num_stages = {{ num_stages }}
{% endif %}

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


SG_BLOCK_SIZE = {{ block | default(128) }}
SG_VEC_WIDTH = {{ vec | default(4) }}
SG_UNROLL = {{ unroll | default(4) }}


def _adamw_step_pallas_kernel(p_ref, m_ref, g_ref, lr, beta1, beta2,
                              eps, wd, bc1, bc2,
                              p_out_ref, m_out_ref, v_out_ref):
    """Per-block Pallas kernel — runs on one TPU core's HBM slice."""
    p = p_ref[...]
    m = m_ref[...]
    g = g_ref[...]
    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * (m * m) + (1.0 - beta2) * (g * g)
    m_hat = new_m / bc1
    v_hat = new_v / bc2
    new_p = p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p)
    p_out_ref[...] = new_p
    m_out_ref[...] = new_m
    v_out_ref[...] = new_v


def launch_adamw_step(p, m, v, g, lr, beta1, beta2, eps, wd, bc1, bc2):
    """JIT-compiled AdamW step. Block size = {{ block | default(128) }}."""
    grid = (max(1, p.size // SG_BLOCK_SIZE),)
    fn = pl.pallas_call(
        lambda *a: _adamw_step_pallas_kernel(*a, lr, beta1, beta2,
                                              eps, wd, bc1, bc2),
        out_shape=[
            jax.ShapeDtypeStruct(p.shape, p.dtype),
            jax.ShapeDtypeStruct(m.shape, m.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        grid=grid,
    )
    return fn(p, m, g)
''',
}

# TEMPLATE_ROOT — kept as the legacy filesystem location for any caller that
# wants to write out emitted sources nearby. Templates themselves are NOT
# loaded from this path anymore — they live in ``_BUNDLED_TEMPLATES`` above.
TEMPLATE_ROOT = Path(__file__).parent / "templates"


# ------------------------------------------------------------------------------
# Codegen (formerly grokking_optimizers/codegen.py)
# ------------------------------------------------------------------------------

class CodegenError(RuntimeError):
    """Raised when the emitter cannot produce a valid source file."""


def _jinja_env():
    """Construct a Jinja2 Environment backed by the bundled-template dict."""
    try:
        import jinja2
    except ImportError as exc:
        raise CodegenError(
            "codegen requires jinja2 — pip install jinja2") from exc
    return jinja2.Environment(
        loader=jinja2.DictLoader(_BUNDLED_TEMPLATES),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


_VENDOR_EXT: Dict[str, str] = {
    "cuda":   ".cu",
    "hip":    ".hip.cpp",
    "pallas": ".py",
}


def _candidate_template_names(optimizer: str, arch: str,
                              vendor: str) -> List[str]:
    """Probe order: <opt>_<arch>.<ext>.j2 → <opt>_<vendor>.<ext>.j2 → <opt>_generic.<ext>.j2."""
    ext = _VENDOR_EXT.get(vendor, ".cu")
    names: List[str] = [f"{optimizer}_{arch}{ext}.j2"]
    if arch.endswith("a") and arch.startswith("sm_"):
        names.append(f"{optimizer}_{arch[:-1]}{ext}.j2")
    names.append(f"{optimizer}_{vendor}{ext}.j2")
    names.append(f"{optimizer}_generic{ext}.j2")
    return names


def find_template(optimizer: str, arch: str,
                  vendor: Optional[str] = None) -> Optional[str]:
    """Locate the highest-priority bundled template name for (optimizer, arch).

    Returns the template KEY (e.g. ``"adamw_sm_90a.cu.j2"``) into
    ``_BUNDLED_TEMPLATES`` — NOT a filesystem path — or ``None`` when no
    matching template is bundled.
    """
    if vendor is None:
        try:
            vendor = get_arch_entry(arch).vendor
        except KeyError:
            vendor = "cuda"
    for name in _candidate_template_names(optimizer, arch, vendor):
        if name in _BUNDLED_TEMPLATES:
            return name
    return None


def _json_default(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, frozenset):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _canonical_json(ctx: Dict[str, Any]) -> str:
    return json.dumps(ctx, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _build_context(config: Dict[str, Any], arch: str) -> Dict[str, Any]:
    entry = get_arch_entry(arch)
    ctx: Dict[str, Any] = {
        "config": dict(config),
        "arch": arch,
        "vendor": entry.vendor,
        "features": sorted(entry.features),
        "warp_size": entry.warp_size,
        "max_smem_per_block": entry.max_smem_per_block,
        "max_regs_per_thread": entry.max_regs_per_thread,
        "max_threads_per_block": entry.max_threads_per_block,
        "macro": entry.macro,
        "display_name": entry.display_name,
    }
    for k, v in config.items():
        if k not in ctx:
            ctx[k] = v
    return ctx


def emit_variant_source(config: Dict[str, Any],
                        dims: List[Dict[str, Any]],
                        optimizer: str,
                        arch: str,
                        out_dir: Path) -> Tuple[Path, List[str]]:
    """Render a bundled template; return (emitted_path, residual_macros).

    Files cached by SHA-256(template_source + canonical-JSON ctx).
    """
    tpl_name = find_template(optimizer, arch)
    if tpl_name is None:
        raise CodegenError(
            f"no bundled template for optimizer={optimizer!r} arch={arch!r}")
    tpl_src = _BUNDLED_TEMPLATES[tpl_name]
    env = _jinja_env()
    try:
        tpl = env.get_template(tpl_name)
    except Exception as exc:
        raise CodegenError(
            f"jinja2 could not load bundled template {tpl_name}: {exc}"
        ) from exc

    ctx = _build_context(config, arch)
    try:
        rendered = tpl.render(**ctx)
    except Exception as exc:
        raise CodegenError(
            f"template {tpl_name} failed to render: {exc}") from exc

    key = hashlib.sha256(
        tpl_src.encode("utf-8") + b"\0" + _canonical_json(ctx).encode("utf-8")
    ).hexdigest()[:16]
    vendor = get_arch_entry(arch).vendor
    ext = _VENDOR_EXT.get(vendor, ".cu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"emitted_{optimizer}_{arch}_{key}{ext}"
    if not out_path.exists():
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        tmp.replace(out_path)
    return out_path, _compute_residual_macros(config, dims)


def _compute_residual_macros(config: Dict[str, Any],
                             dims: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in dims:
        macro = d.get("macro")
        if not macro:
            continue
        name = d.get("name")
        if name is None or name not in config:
            continue
        force = (d.get("kind") == "force_macro")
        applies = d.get("applies_to") or []
        is_host = ("host" in applies) or (d.get("target") == "host")
        if force or is_host:
            v = config[name]
            if isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = 1 if v else 0
            out.append(f"-D{macro}={v}")
    return out


def list_available_templates() -> List[Tuple[str, str]]:
    """Enumerate (optimizer, arch_or_vendor) pairs of bundled templates."""
    out: List[Tuple[str, str]] = []
    for fname in sorted(_BUNDLED_TEMPLATES):
        name = fname[: -len(".j2")]
        for ext in (".hip.cpp", ".cu", ".py"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        if "_" in name:
            opt, suffix = name.split("_", 1)
            out.append((opt, suffix))
        else:
            out.append((name, ""))
    return out


def validate_template_render(optimizer: str, arch: str
                             ) -> Tuple[bool, str]:
    """Render template with a canned minimal config; return (ok, err)."""
    try:
        if arch not in ARCH_TABLE:
            return False, f"unknown arch {arch!r}"
        space = build_full_search_space()
        space_key = arch
        if space_key not in space and arch.endswith("a"):
            space_key = arch[:-1]
        block = space.get(space_key, {})
        dims = block.get("dims", []) if isinstance(block, dict) else []
        cfg: Dict[str, Any] = {}
        for d in dims:
            vs = d.get("values") or []
            if vs:
                cfg[d["name"]] = vs[0]
        with tempfile.TemporaryDirectory() as td:
            emit_variant_source(cfg, dims, optimizer, arch, Path(td))
        return True, ""
    except CodegenError as exc:
        return False, f"CodegenError: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_with_nvcc_dryrun(emitted: Path) -> Tuple[bool, str]:
    """``nvcc --cuda --dryrun`` syntax check; graceful when nvcc missing."""
    if not shutil.which("nvcc"):
        return True, "nvcc unavailable; skipping validation"
    if emitted.suffix not in (".cu", ".cuh"):
        return True, f"non-CUDA source {emitted.suffix}; skipping nvcc check"
    try:
        r = subprocess.run(
            ["nvcc", "--cuda", "--dryrun", str(emitted)],
            capture_output=True, text=True, timeout=30,
        )
        ok = (r.returncode == 0)
        return ok, (r.stderr or r.stdout or "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return True, f"nvcc invocation failed ({type(exc).__name__}): {exc}"


# ---- CUTLASS GEMM emitter helpers -----------------------------------------
#
# AMD CK equivalent: TODO — when the ROCm composable-kernel Python frontend
# stabilises, mirror this function as emit_ck_gemm_variants(arch, ...)
# for gfx9xx/gfx12xx targets. Do NOT silently fall back to CUTLASS for HIP.

# Hand-curated fallback sweep used when cutlass.op.Gemm.tile_descriptions()
# is unavailable on the installed cutlass-python release. Each entry is
# (threadblock_shape, cluster_shape, stages, kernel_schedule, epilogue_schedule).
_CUTLASS_FALLBACK_VARIANTS: Tuple[Tuple[Tuple[int, int, int],
                                        Tuple[int, int, int],
                                        int, str, str], ...] = (
    ((128, 128, 32), (1, 1, 1), 3,
     "KernelTmaWarpSpecializedCooperative", "EpilogueTmaWarpSpecialized"),
    ((128, 128, 64), (1, 1, 1), 3,
     "KernelTmaWarpSpecializedCooperative", "EpilogueTmaWarpSpecialized"),
    ((256, 128, 32), (2, 1, 1), 3,
     "KernelTmaWarpSpecializedCooperative", "EpilogueTmaWarpSpecialized"),
    ((256, 128, 64), (2, 1, 1), 3,
     "KernelTmaWarpSpecializedCooperative", "EpilogueTmaWarpSpecialized"),
)

# Map our dtype strings → (CUTLASS C++ type, CUTLASS python element enum name).
_CUTLASS_DTYPE_MAP: Dict[str, Tuple[str, str]] = {
    "fp16":     ("cutlass::half_t",     "float16"),
    "f16":      ("cutlass::half_t",     "float16"),
    "half":     ("cutlass::half_t",     "float16"),
    "bf16":     ("cutlass::bfloat16_t", "bfloat16"),
    "bfloat16": ("cutlass::bfloat16_t", "bfloat16"),
    "fp32":     ("float",               "float32"),
    "f32":      ("float",               "float32"),
    "float":    ("float",               "float32"),
    "tf32":     ("cutlass::tfloat32_t", "tfloat32"),
    "fp8":      ("cutlass::float_e4m3_t", "float_e4m3"),
    "e4m3":     ("cutlass::float_e4m3_t", "float_e4m3"),
    "e5m2":     ("cutlass::float_e5m2_t", "float_e5m2"),
}


def _cutlass_variant_key(tile: Tuple[int, int, int],
                         cluster: Tuple[int, int, int],
                         stages: int,
                         schedule: str) -> str:
    """Build a filesystem- and C-identifier-safe variant key."""
    t = "x".join(str(int(v)) for v in tile)
    c = "x".join(str(int(v)) for v in cluster)
    sch = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in schedule)
    sch = sch.strip("_") or "default"
    return f"t{t}_c{c}_s{int(stages)}_{sch}"


def _cutlass_tuple3(obj: Any, default: Tuple[int, int, int]
                    ) -> Tuple[int, int, int]:
    """Best-effort extraction of a 3-tuple from a cutlass shape object."""
    if obj is None:
        return default
    # Plain iterable.
    try:
        seq = tuple(int(x) for x in obj)
        if len(seq) >= 3:
            return (seq[0], seq[1], seq[2])
        if len(seq) == 2:
            return (seq[0], seq[1], default[2])
    except (TypeError, ValueError):
        pass
    # Attribute style (.m/.n/.k, .x/.y/.z).
    for triple in (("m", "n", "k"), ("x", "y", "z")):
        try:
            return (int(getattr(obj, triple[0])),
                    int(getattr(obj, triple[1])),
                    int(getattr(obj, triple[2])))
        except (AttributeError, TypeError, ValueError):
            continue
    return default


def _cutlass_schedule_str(obj: Any, default: str) -> str:
    """Best-effort name extraction from a cutlass schedule enum / object."""
    if obj is None:
        return default
    for attr in ("name", "value", "__name__"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v:
            return v
    s = str(obj)
    return s if s else default


def _enumerate_cutlass_variants(arch: str,
                                problem_shape: Tuple[int, int, int],
                                dtype: str
                                ) -> List[Dict[str, Any]]:
    """Probe cutlass.op.Gemm.tile_descriptions(); fall back to a curated sweep.

    Returns a list of metadata dicts; one per variant.
    """
    fallback: List[Dict[str, Any]] = []
    for tile, cluster, stages, ksched, esched in _CUTLASS_FALLBACK_VARIANTS:
        fallback.append({
            "tile": tile, "cluster": cluster, "stages": stages,
            "schedule": ksched, "epilogue": esched, "source": "fallback",
        })

    try:
        import cutlass  # type: ignore
    except ImportError:
        # Caller already guarded against this — defensive return.
        return fallback

    cc = ARCH_TABLE[arch].cutlass_arch
    cpp_dt, py_dt_name = _CUTLASS_DTYPE_MAP.get(
        dtype, ("float", "float32"))
    M, N, K = (int(problem_shape[0]),
               int(problem_shape[1]),
               int(problem_shape[2]))

    # Resolve dtype on the cutlass module — fall back to fp32 if missing.
    elem = None
    try:
        elem = getattr(cutlass.DataType, py_dt_name, None) \
            if hasattr(cutlass, "DataType") else None
    except Exception:
        elem = None

    # Build a Gemm op as defensively as possible — different cutlass-python
    # releases (2.x/3.x/4.x) accept slightly different kwargs.
    gemm = None
    for kwargs in (
        {"element": elem, "cc": cc} if elem is not None
            else {"cc": cc},
        {"element_A": elem, "element_B": elem, "element_C": elem,
         "element_D": elem, "cc": cc} if elem is not None
            else {"cc": cc},
        {"cc": cc},
        {},
    ):
        try:
            gemm = cutlass.op.Gemm(**kwargs)
            break
        except Exception:
            continue
    if gemm is None:
        return fallback

    # Probe tile_descriptions() — if unavailable, return curated fallback.
    tds = None
    try:
        tds_fn = getattr(gemm, "tile_descriptions", None)
        if callable(tds_fn):
            tds = tds_fn()
    except Exception:
        tds = None
    if not tds:
        return fallback

    out: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for td in tds:
        tile = _cutlass_tuple3(
            getattr(td, "threadblock_shape", None), (128, 128, 32))
        cluster = _cutlass_tuple3(
            getattr(td, "cluster_shape", None), (1, 1, 1))
        stages = 3
        try:
            stages = int(getattr(td, "stages", 3) or 3)
        except (TypeError, ValueError):
            stages = 3
        ksched = _cutlass_schedule_str(
            getattr(td, "kernel_schedule", None),
            "KernelTmaWarpSpecializedCooperative")
        esched = _cutlass_schedule_str(
            getattr(td, "epilogue_schedule", None),
            "EpilogueTmaWarpSpecialized")
        key = _cutlass_variant_key(tile, cluster, stages, ksched)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "tile": tile, "cluster": cluster, "stages": stages,
            "schedule": ksched, "epilogue": esched, "source": "tile_descriptions",
        })
        # Cap the sweep so a single call doesn't emit thousands of files.
        if len(out) >= 64:
            break

    _ = (cpp_dt, M, N, K)  # used by caller for source emission
    return out if out else fallback


def _render_cutlass_gemm_source(arch: str,
                                problem_shape: Tuple[int, int, int],
                                dtype: str,
                                variant: Dict[str, Any],
                                variant_key: str) -> str:
    """Render a standalone .cu source for one (arch, dtype, tile-variant)."""
    cpp_dt, _ = _CUTLASS_DTYPE_MAP.get(dtype, ("float", "float32"))
    cc = ARCH_TABLE[arch].cutlass_arch or 90
    tile = variant["tile"]
    cluster = variant["cluster"]
    stages = int(variant["stages"])
    ksched = str(variant["schedule"])
    esched = str(variant["epilogue"])
    M, N, K = problem_shape
    fn_name = f"launch_{variant_key}"
    return f"""// AUTO-GENERATED by emit_cutlass_gemm_variants — DO NOT EDIT
// Arch:        {arch} (CUTLASS_ARCH_MMA_SM{cc})
// Dtype:       {dtype}  (C++ type: {cpp_dt})
// Problem:     M={M}, N={N}, K={K}
// Tile:        {tile[0]}x{tile[1]}x{tile[2]}
// Cluster:     {cluster[0]}x{cluster[1]}x{cluster[2]}
// Stages:      {stages}
// Schedule:    {ksched}
// Epilogue:    {esched}

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/host_tensor.h>

namespace sg_cutlass_gen {{

using ElementA       = {cpp_dt};
using ElementB       = {cpp_dt};
using ElementC       = {cpp_dt};
using ElementD       = {cpp_dt};
using ElementAccum   = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape    = cute::Shape<cute::Int<{tile[0]}>, cute::Int<{tile[1]}>, cute::Int<{tile[2]}>>;
using ClusterShape = cute::Shape<cute::Int<{cluster[0]}>, cute::Int<{cluster[1]}>, cute::Int<{cluster[2]}>>;

using ArchTag        = cutlass::arch::Sm{cc};
using OperatorClass  = cutlass::arch::OpClassTensorOp;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementCompute,
        ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
        ElementAccum,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}}  // namespace sg_cutlass_gen

extern "C" int {fn_name}(void* A, void* B, void* C, void* D,
                         int M, int N, int K,
                         float alpha, float beta,
                         cudaStream_t stream) {{
    using namespace sg_cutlass_gen;
    typename Gemm::Arguments args{{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {{M, N, K, 1}},
        {{
            static_cast<ElementA const*>(A), {{K, cute::Int<1>{{}}, 0}},
            static_cast<ElementB const*>(B), {{K, cute::Int<1>{{}}, 0}},
        }},
        {{
            {{ElementCompute(alpha), ElementCompute(beta)}},
            static_cast<ElementC const*>(C), {{N, cute::Int<1>{{}}, 0}},
            static_cast<ElementD*>(D),       {{N, cute::Int<1>{{}}, 0}},
        }}
    }};
    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {{
        return static_cast<int>(status);
    }}
    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {{
        cudaError_t merr = cudaMallocAsync(&workspace, workspace_size, stream);
        if (merr != cudaSuccess) {{
            return static_cast<int>(cutlass::Status::kErrorMemoryAllocation);
        }}
    }}
    status = gemm.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {{
        status = gemm(stream);
    }}
    if (workspace != nullptr) {{
        cudaFreeAsync(workspace, stream);
    }}
    return static_cast<int>(status);
}}
"""


def emit_cutlass_gemm_variants(arch: str,
                               problem_shape: Tuple[int, int, int],
                               dtype: str,
                               out_dir: Path
                               ) -> List[Tuple[Path, Dict[str, Any]]]:
    """Emit standalone CUTLASS GEMM ``.cu`` files for one (arch, dtype, MNK).

    Enumerates {tile x cluster x stages x schedule x epilogue} variants the
    locally-installed cutlass-python supports for the target. Each variant
    becomes one ``.cu`` file under ``out_dir`` exporting a ``extern "C"``
    launcher; the returned list pairs the emitted path with a metadata dict.

    Re-invocation with the same variant key skips re-emission (filename-based
    cache). Errors inside the cutlass-python call path (other than
    ``ImportError``) are wrapped as ``CodegenError`` so callers can fall back
    to the template-only emitter gracefully.

    Scope: NVIDIA sm_90a (Hopper) and sm_100a (Blackwell) only — the cluster /
    TMA / wgmma plumbing assumed below isn't valid on earlier SMs and there's
    no clean AMD CK equivalent in-tree yet (see module-level TODO above).
    """
    try:
        import cutlass  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise CodegenError(
            "cutlass-python required for GEMM emitter — install with "
            "`pip install nvidia-cutlass` (3.5+) on a CUDA-capable host"
        ) from exc

    if arch not in ("sm_90a", "sm_100a"):
        raise CodegenError(
            "CUTLASS GEMM emitter supports sm_90a and sm_100a only")

    if arch not in ARCH_TABLE:
        raise CodegenError(f"unknown arch {arch!r}")

    if dtype not in _CUTLASS_DTYPE_MAP:
        raise CodegenError(
            f"unsupported dtype {dtype!r} for CUTLASS GEMM emitter; "
            f"known: {sorted(_CUTLASS_DTYPE_MAP)}")

    try:
        M, N, K = (int(problem_shape[0]),
                   int(problem_shape[1]),
                   int(problem_shape[2]))
    except (TypeError, ValueError, IndexError) as exc:
        raise CodegenError(
            f"problem_shape must be a 3-tuple (M, N, K), got {problem_shape!r}"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        variants = _enumerate_cutlass_variants(arch, (M, N, K), dtype)
    except Exception as exc:  # wrap any cutlass-python failure as CodegenError
        raise CodegenError(
            f"cutlass-python variant enumeration failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not variants:
        raise CodegenError(
            f"no CUTLASS GEMM variants emitted for arch={arch} dtype={dtype} "
            f"shape={(M, N, K)}")

    emitted: List[Tuple[Path, Dict[str, Any]]] = []
    for var in variants:
        var_key = _cutlass_variant_key(
            var["tile"], var["cluster"], var["stages"], var["schedule"])
        fname = (f"cutlass_gemm_{arch}_{dtype}_"
                 f"{M}x{N}x{K}_{var_key}.cu")
        out_path = out_dir / fname
        meta = {
            "tile": var["tile"],
            "cluster": var["cluster"],
            "stages": var["stages"],
            "schedule": var["schedule"],
            "epilogue": var["epilogue"],
            "arch": arch,
            "dtype": dtype,
            "mnk": (M, N, K),
            "variant_key": var_key,
            "source": var.get("source", "unknown"),
        }
        if out_path.exists() and out_path.stat().st_size > 0:
            emitted.append((out_path, meta))
            continue
        try:
            src = _render_cutlass_gemm_source(
                arch, (M, N, K), dtype, var, var_key)
        except Exception as exc:
            raise CodegenError(
                f"CUTLASS GEMM source render failed for variant {var_key}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(src, encoding="utf-8")
        tmp.replace(out_path)
        emitted.append((out_path, meta))

    return emitted


# ------------------------------------------------------------------------------
# Kernel registry (formerly grokking_optimizers/kernel_registry.py)
# ------------------------------------------------------------------------------

class RegistryError(RuntimeError):
    """Raised when runtime specialization cannot proceed."""


_SHAPE_BUCKETS: Tuple[Tuple[int, str], ...] = (
    (1024,        "tiny"),
    (65536,       "small"),
    (4_194_304,   "medium"),
    (268_435_456, "large"),
)
_SHAPE_HINT_SIZE: Dict[str, int] = {
    "tiny":   256,
    "small":  4_096,
    "medium": 65_536,
    "large":  1_048_576,
    "huge":   16_777_216,
}


def _shape_class(shape: Tuple[int, ...]) -> str:
    n = 1
    for d in shape:
        n *= int(d)
    for upper, label in _SHAPE_BUCKETS:
        if n < upper:
            return label
    return "huge"


def _shape_class_size(cls: str) -> int:
    return _SHAPE_HINT_SIZE.get(cls, 4_096)


_DTYPE_C_TYPE: Dict[str, str] = {
    "fp32":  "float", "f32":   "float", "float": "float",
    "fp16":  "__half", "f16":   "__half", "half":  "__half",
    "bf16":  "__nv_bfloat16", "bfloat16": "__nv_bfloat16",
    "fp64":  "double", "f64":   "double", "double": "double",
}


def _resolve_ctype(dtype: str) -> str:
    return _DTYPE_C_TYPE.get(dtype, dtype)


def _default_template_provider(op: str, arch: str) -> str:
    return r"""
extern "C" __global__ void specialized_{OP}_kernel({DTYPE}* out, const {DTYPE}* in, int n) {{
    constexpr int kSizeHint = {SIZE_HINT};
    constexpr int kShapeDims = {SHAPE_DIMS};
    (void)kSizeHint; (void)kShapeDims;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}}
""".replace("{OP}", op)


class KernelRegistry:
    """Sub-µs runtime-specialized kernel lookup via NVRTC / hipRTC."""

    def __init__(self, arch: str, cache_dir: Path,
                 template_provider: Optional[Callable[[str, str], str]] = None):
        if arch not in ARCH_TABLE:
            raise RegistryError(f"unknown arch {arch!r}")
        entry = ARCH_TABLE[arch]
        if entry.vendor == "pallas":
            raise RegistryError(
                f"KernelRegistry not applicable to Pallas arch {arch!r}; "
                "use JAX/XLA jit directly.")
        self.arch = arch
        self.vendor = entry.vendor
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._handle_cache: Dict[str, Any] = {}
        self._template_provider = (template_provider
                                   or _default_template_provider)

    def _key(self, op: str, dtype: str, shape_cls: str) -> str:
        return hashlib.sha256(
            f"{self.arch}|{op}|{dtype}|{shape_cls}".encode()
        ).hexdigest()[:24]

    def dispatch(self, op: str, dtype: str, shape: Tuple[int, ...]):
        """Return a callable kernel handle. Sub-µs on cache hit."""
        shape_cls = _shape_class(tuple(shape))
        key = self._key(op, dtype, shape_cls)
        with self._lock:
            cached = self._handle_cache.get(key)
            if cached is not None:
                return cached
        cubin_path = self.cache_dir / f"{key}.cubin"
        if cubin_path.exists() and cubin_path.stat().st_size > 0:
            handle = self._load_cubin(cubin_path, op)
        else:
            cubin = self._compile(op, dtype, shape_cls, tuple(shape))
            tmp_path = cubin_path.with_suffix(".cubin.tmp")
            tmp_path.write_bytes(cubin)
            os.replace(tmp_path, cubin_path)
            handle = self._load_cubin(cubin_path, op)
        with self._lock:
            self._handle_cache[key] = handle
        return handle

    def _compile(self, op: str, dtype: str, shape_cls: str,
                 example_shape: Tuple[int, ...]) -> bytes:
        src = self._template_provider(op, self.arch).format(
            DTYPE=_resolve_ctype(dtype),
            SHAPE_DIMS=len(example_shape),
            SIZE_HINT=_shape_class_size(shape_cls),
        )
        if self.vendor == "cuda":
            return self._nvrtc_compile(src)
        if self.vendor == "hip":
            return self._hiprtc_compile(src)
        raise RegistryError(f"no rt-compile path for vendor {self.vendor!r}")

    def _nvrtc_compile(self, src: str) -> bytes:
        try:
            from cuda.bindings import nvrtc  # type: ignore
        except ImportError:
            try:
                from cuda import nvrtc  # type: ignore
            except ImportError as exc:
                raise RegistryError(
                    "NVRTC requires cuda-python (pip install cuda-python)"
                ) from exc
        entry = ARCH_TABLE[self.arch]
        compute = f"compute_{entry.cutlass_arch}{entry.arch_suffix}"
        create_res = nvrtc.nvrtcCreateProgram(
            src.encode(), b"kernel.cu", 0, [], [])
        prog = _first_payload(create_res)
        if prog is None:
            raise RegistryError(f"nvrtcCreateProgram failed: {create_res!r}")
        opts = [f"-arch={compute}".encode(),
                b"-default-device",
                b"--std=c++17"]
        compile_res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if _err_value(compile_res) != 0:
            log = ""
            try:
                log_sz = _first_payload(nvrtc.nvrtcGetProgramLogSize(prog)) or 0
                if log_sz:
                    buf = bytearray(log_sz)
                    nvrtc.nvrtcGetProgramLog(prog, buf)
                    log = buf.decode("utf-8", "replace")
            except Exception:
                pass
            raise RegistryError(f"NVRTC compile failed: {compile_res!r}\n{log}")
        cubin_ok = (hasattr(nvrtc, "nvrtcGetCUBINSize")
                    and hasattr(nvrtc, "nvrtcGetCUBIN"))
        if cubin_ok:
            size = _first_payload(nvrtc.nvrtcGetCUBINSize(prog)) or 0
            if size > 0:
                buf = bytearray(size)
                nvrtc.nvrtcGetCUBIN(prog, buf)
                return bytes(buf)
        size = _first_payload(nvrtc.nvrtcGetPTXSize(prog)) or 0
        buf = bytearray(size)
        nvrtc.nvrtcGetPTX(prog, buf)
        return bytes(buf)

    def _hiprtc_compile(self, src: str) -> bytes:
        try:
            from hip import hiprtc  # type: ignore
        except ImportError as exc:
            raise RegistryError(
                "hipRTC requires hip-python (pip install hip-python)"
            ) from exc
        offload = ARCH_TABLE[self.arch].hipcc_offload_arch
        prog = _first_payload(
            hiprtc.hiprtcCreateProgram(src.encode(), b"kernel.hip", 0, [], []))
        if prog is None:
            raise RegistryError("hiprtcCreateProgram failed")
        opts = [f"--offload-arch={offload}".encode()]
        compile_res = hiprtc.hiprtcCompileProgram(prog, len(opts), opts)
        if _err_value(compile_res) != 0:
            raise RegistryError(f"hipRTC compile failed: {compile_res!r}")
        size = _first_payload(hiprtc.hiprtcGetCodeSize(prog)) or 0
        buf = bytearray(size)
        hiprtc.hiprtcGetCode(prog, buf)
        return bytes(buf)

    def _load_cubin(self, cubin_path: Path, op: str):
        return _LoadedKernel(cubin_path, op, vendor=self.vendor)


def _first_payload(res):
    if isinstance(res, tuple):
        if len(res) >= 2:
            return res[1]
        return None
    return res


def _err_value(res) -> int:
    if isinstance(res, tuple) and res:
        err = res[0]
        v = getattr(err, "value", err)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    return 0


def _load_cuda_driver():
    """Import cuda-python driver bindings (modern path first, legacy fallback).

    Returns the bindings module (e.g. ``cuda.bindings.driver`` or
    ``cuda.cuda``). Raises :class:`RegistryError` if neither is installed.
    """
    try:
        from cuda.bindings import driver as _drv  # type: ignore
        return _drv
    except ImportError:
        pass
    try:
        from cuda import cuda as _drv  # type: ignore
        return _drv
    except ImportError as exc:
        raise RegistryError(
            "cuda-python required for live launch "
            "(pip install cuda-python)") from exc


def _load_hip_driver():
    """Import hip-python driver bindings. Raises RegistryError on miss."""
    try:
        from hip import hip as _hip_drv  # type: ignore
        return _hip_drv
    except ImportError as exc:
        raise RegistryError(
            "hip-python required for live launch "
            "(pip install hip-python)") from exc


class _LoadedKernel:
    """Loaded cubin handle. Real cuModule / hipModule lookup happens lazily
    on the first __call__ so construction stays cheap and stays importable
    on hosts without cuda-python / hip-python."""

    __slots__ = ("cubin_path", "op", "vendor", "_module", "_function", "_lock")

    def __init__(self, cubin_path: Path, op: str, vendor: str = "cuda"):
        self.cubin_path = cubin_path
        self.op = op
        self.vendor = vendor
        self._module = None
        self._function = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (f"<Kernel op={self.op} cubin={self.cubin_path.name} "
                f"vendor={self.vendor}>")

    def _load(self) -> None:
        """Lazy module load + function lookup. Idempotent under the lock."""
        if self._function is not None:
            return
        with self._lock:
            if self._function is not None:
                return
            cubin_bytes = self.cubin_path.read_bytes()
            fn_name = f"specialized_{self.op}_kernel".encode()
            if self.vendor == "cuda":
                drv = _load_cuda_driver()
                if not hasattr(drv, "cuModuleLoadData") \
                        or not hasattr(drv, "cuModuleGetFunction"):
                    raise RegistryError(
                        "cuda-python bindings missing cuModuleLoadData / "
                        "cuModuleGetFunction — please upgrade cuda-python")
                mod_res = drv.cuModuleLoadData(cubin_bytes)
                if _err_value(mod_res) != 0:
                    raise RegistryError(
                        f"cuModuleLoadData failed: {mod_res!r}")
                module = _first_payload(mod_res)
                if module is None:
                    raise RegistryError(
                        f"cuModuleLoadData returned no module: {mod_res!r}")
                fn_res = drv.cuModuleGetFunction(module, fn_name)
                if _err_value(fn_res) != 0:
                    raise RegistryError(
                        f"cuModuleGetFunction({fn_name!r}) failed: {fn_res!r}")
                function = _first_payload(fn_res)
                if function is None:
                    raise RegistryError(
                        f"cuModuleGetFunction returned nothing: {fn_res!r}")
                self._module = module
                self._function = function
            elif self.vendor == "hip":
                drv = _load_hip_driver()
                if not hasattr(drv, "hipModuleLoadData") \
                        or not hasattr(drv, "hipModuleGetFunction"):
                    raise RegistryError(
                        "hip-python bindings missing hipModuleLoadData / "
                        "hipModuleGetFunction — please upgrade hip-python")
                mod_res = drv.hipModuleLoadData(cubin_bytes)
                if _err_value(mod_res) != 0:
                    raise RegistryError(
                        f"hipModuleLoadData failed: {mod_res!r}")
                module = _first_payload(mod_res)
                if module is None:
                    raise RegistryError(
                        f"hipModuleLoadData returned no module: {mod_res!r}")
                fn_res = drv.hipModuleGetFunction(module, fn_name)
                if _err_value(fn_res) != 0:
                    raise RegistryError(
                        f"hipModuleGetFunction({fn_name!r}) failed: "
                        f"{fn_res!r}")
                function = _first_payload(fn_res)
                if function is None:
                    raise RegistryError(
                        f"hipModuleGetFunction returned nothing: {fn_res!r}")
                self._module = module
                self._function = function
            else:
                raise RegistryError(
                    f"_LoadedKernel: unsupported vendor {self.vendor!r}")

    def _pack_args(self, args):
        """Pack Python args into a ctypes void** array suitable for
        cuLaunchKernel / hipModuleLaunchKernel.

        Each Python int is treated as a raw device-pointer-sized integer
        (CUdeviceptr / uint64). Floats become C doubles. Already-prepared
        ctypes objects pass through unchanged. Bools become ints.
        """
        import ctypes
        boxed = []
        for a in args:
            if isinstance(a, bool):
                boxed.append(ctypes.c_int(int(a)))
            elif isinstance(a, int):
                # Treat int as 64-bit (device pointer OR scalar int). cubin
                # kernels we generate take `int n` (32-bit) plus pointers
                # (64-bit). Use c_int for ints that fit, c_uint64 otherwise.
                if -(1 << 31) <= a < (1 << 31):
                    boxed.append(ctypes.c_int(a))
                else:
                    boxed.append(ctypes.c_uint64(a))
            elif isinstance(a, float):
                boxed.append(ctypes.c_double(a))
            else:
                # Already a ctypes object (or buffer); pass through.
                boxed.append(a)
        # Heuristic: kernels emitted by _default_template_provider take
        # (out_ptr, in_ptr, int n). Coerce the first two ints to c_uint64
        # if they were demoted to c_int above (likely raw device pointers).
        # We can only know this from the kernel signature; the caller knows,
        # so we leave it to the caller to pass real ctypes objects when they
        # need a specific layout.
        ptr_array = (ctypes.c_void_p * len(boxed))()
        # Keep refs alive via a list returned alongside.
        for i, b in enumerate(boxed):
            ptr_array[i] = ctypes.cast(ctypes.pointer(b), ctypes.c_void_p)
        return ptr_array, boxed

    def __call__(self, *args, grid=(1, 1, 1), block=(1, 1, 1),
                 shared: int = 0, stream: int = 0) -> None:
        """Pack args via ctypes void**, call cuLaunchKernel /
        hipModuleLaunchKernel.

        ``args`` must be Python ints (treated as raw device pointers /
        scalar ints), floats, or already-allocated ctypes objects.
        Tensors should be unwrapped to their ``.data_ptr()`` ints by the
        caller.
        """
        import ctypes
        self._load()
        # Default-template kernels take (out_ptr*, in_ptr*, int n). Caller
        # passes plain ints for the pointers AND the count, but pointers
        # must be 64-bit while n must be 32-bit. Re-coerce by position.
        coerced = []
        for idx, a in enumerate(args):
            if isinstance(a, int) and not isinstance(a, bool):
                # By convention, the last int arg is the scalar count
                # (32-bit). Everything else that's an int is treated as a
                # device pointer (64-bit). This matches the
                # _default_template_provider kernel signature.
                if idx == len(args) - 1:
                    coerced.append(ctypes.c_int(a))
                else:
                    coerced.append(ctypes.c_uint64(a))
            elif isinstance(a, float):
                coerced.append(ctypes.c_float(a))
            elif isinstance(a, bool):
                coerced.append(ctypes.c_int(int(a)))
            else:
                coerced.append(a)
        n = len(coerced)
        ptr_array = (ctypes.c_void_p * n)()
        for i, b in enumerate(coerced):
            ptr_array[i] = ctypes.cast(ctypes.pointer(b), ctypes.c_void_p)

        gx, gy, gz = (int(grid[0]), int(grid[1]), int(grid[2]))
        bx, by, bz = (int(block[0]), int(block[1]), int(block[2]))

        if self.vendor == "cuda":
            drv = _load_cuda_driver()
            if not hasattr(drv, "cuLaunchKernel"):
                raise RegistryError(
                    "cuda-python bindings missing cuLaunchKernel — "
                    "please upgrade cuda-python")
            kernel_params = ctypes.addressof(ptr_array)
            launch_res = drv.cuLaunchKernel(
                self._function,
                gx, gy, gz,
                bx, by, bz,
                int(shared),
                int(stream),
                kernel_params,
                0,  # extra
            )
            if _err_value(launch_res) != 0:
                raise RegistryError(
                    f"cuLaunchKernel failed: {launch_res!r}")
        elif self.vendor == "hip":
            drv = _load_hip_driver()
            if not hasattr(drv, "hipModuleLaunchKernel"):
                raise RegistryError(
                    "hip-python bindings missing hipModuleLaunchKernel — "
                    "please upgrade hip-python")
            kernel_params = ctypes.addressof(ptr_array)
            launch_res = drv.hipModuleLaunchKernel(
                self._function,
                gx, gy, gz,
                bx, by, bz,
                int(shared),
                int(stream),
                kernel_params,
                0,  # extra
            )
            if _err_value(launch_res) != 0:
                raise RegistryError(
                    f"hipModuleLaunchKernel failed: {launch_res!r}")
        else:
            raise RegistryError(
                f"_LoadedKernel.__call__: unsupported vendor {self.vendor!r}")


_REGISTRY: Dict[str, KernelRegistry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_registry(arch: str,
                 cache_dir: Optional[Path] = None) -> KernelRegistry:
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(arch)
        if existing is not None:
            return existing
        cd = cache_dir or (
            Path(os.environ.get("HOME", "/tmp"))
            / ".cache" / "supergrok" / "nvrtc")
        reg = KernelRegistry(arch, cd)
        _REGISTRY[arch] = reg
        return reg


def initialize_registry(spec, report=None) -> Optional[KernelRegistry]:
    """Pre-warm the registry from build() when enable_runtime_specialization=True."""
    if not getattr(spec, "enable_runtime_specialization", False):
        return None
    try:
        cache_dir = Path(spec.out_dir) / "nvrtc_cache"
        reg = get_registry(spec.arch, cache_dir=cache_dir)
    except RegistryError as exc:
        if report is not None:
            report.write(f"[nvrtc] disabled: {exc}\n")
        return None
    op = getattr(spec, "optimizer", "kernel")
    for dtype in ("fp32", "fp64"):
        try:
            reg.dispatch(op, dtype, (1024,))
        except RegistryError as exc:
            if report is not None:
                report.write(f"[nvrtc] {dtype} prewarm skipped: {exc}\n")
    if report is not None:
        report.write(f"[nvrtc] registry initialized for {spec.arch} "
                     f"(cache={reg.cache_dir})\n")
    return reg


# ------------------------------------------------------------------------------
# Device-side PGO (formerly grokking_optimizers/device_profiling.py)
# ------------------------------------------------------------------------------

class DeviceProfilingError(RuntimeError):
    """Raised by callers that opt into strict mode (not used internally)."""


STALL_DIM_HINTS: Dict[str, List[str]] = {
    "long_scoreboard":     ["swizzle", "lds_padding", "vec"],
    "not_selected":        ["block", "waves_per_eu", "maxrregcount"],
    "math_pipe_throttle":  ["unroll", "num_stages"],
    "memory_throttle":     ["block", "vec", "async_depth", "num_stages"],
    "tex_throttle":        ["block", "vec"],
    "barrier":             ["block", "warp_specialization"],
    "wait":                ["num_stages", "async_depth"],
    "imc_miss":            ["block", "unroll"],
    "lg_throttle":         ["lds_padding", "swizzle"],
    "dispatch_stall":      ["block", "maxrregcount"],
    "vmem_lat":            ["waves_per_eu", "vec"],
    "lds_bank_conflict":   ["lds_padding"],
    "valu_dep":            ["unroll", "num_stages"],
}


def collect_nvidia_stalls(workload_cmd: List[str], out_dir: Path,
                          *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    nsys = shutil.which("nsys") or shutil.which("/usr/local/cuda/bin/nsys")
    if nsys is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cupti_pc_sample.qdrep"
    try:
        proc = subprocess.run(
            [nsys, "profile",
             "-o", str(report.with_suffix("")),
             "--sample=cpu",
             "--cuda-memory-usage=true",
             "--cuda-stack=true",
             "--stats=true"] + list(workload_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"nsys exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "nsys timeout"}
    except OSError as exc:
        return {"error": f"nsys spawn failed: {exc}"}
    stall_reasons = _parse_nsys_stall_section(proc.stdout or "")
    return {"tool": "nsys", "report": str(report),
            "stall_reasons": stall_reasons,
            "bias_hints": _stall_to_bias_hints(stall_reasons)}


def _parse_nsys_stall_section(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pct_re = _re_consolidated.compile(r"(\d+\.?\d*)\s*%")
    for line in text.splitlines():
        low = line.lower()
        for reason in STALL_DIM_HINTS:
            if reason in low:
                m = pct_re.search(line)
                if m:
                    try:
                        out[reason] = float(m.group(1)) / 100.0
                    except ValueError:
                        continue
    return out


def collect_amd_stalls(workload_cmd: List[str], out_dir: Path,
                       *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    rocprof = shutil.which("rocprof") or shutil.which("/opt/rocm/bin/rocprof")
    if rocprof is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "rocprof_stats.csv"
    try:
        proc = subprocess.run(
            [rocprof, "--stats", "-o", str(out_csv)] + list(workload_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"rocprof exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "rocprof timeout"}
    except OSError as exc:
        return {"error": f"rocprof spawn failed: {exc}"}
    stall_reasons = _parse_rocprof_csv(out_csv)
    return {"tool": "rocprof", "report": str(out_csv),
            "stall_reasons": stall_reasons,
            "bias_hints": _stall_to_bias_hints(stall_reasons)}


def _parse_rocprof_csv(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    out: Dict[str, float] = {}
    try:
        lines = csv_path.read_text(errors="ignore").splitlines()
        if not lines:
            return {}
        header = [c.strip() for c in lines[0].split(",")]
        for line in lines[1:]:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            for k, v in row.items():
                kl = k.lower()
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if "wait" in kl:
                    out["vmem_lat"] = out.get("vmem_lat", 0.0) + val
                if "lds" in kl:
                    out["lds_bank_conflict"] = (
                        out.get("lds_bank_conflict", 0.0) + val)
                if "valu" in kl:
                    out["valu_dep"] = out.get("valu_dep", 0.0) + val
    except Exception:
        return {}
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def collect_pallas_stalls(workload_cmd: List[str], out_dir: Path,
                          *, timeout: int = 600) -> Optional[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = out_dir / "xla_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    extra_flags = (
        f"--xla_gpu_dump_autotuned_gemm_fusions=true "
        f"--xla_dump_to={dump_dir}"
    )
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + extra_flags).strip()
    try:
        proc = subprocess.run(
            list(workload_cmd), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return {"error": f"xla workload exit {proc.returncode}",
                    "stderr": (proc.stderr or "")[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "xla workload timeout"}
    except OSError as exc:
        return {"error": f"xla workload spawn failed: {exc}"}
    stall_reasons = _parse_xla_dump(dump_dir)
    return {"tool": "xla_dump", "report": str(dump_dir),
            "stall_reasons": stall_reasons,
            "bias_hints": _stall_to_bias_hints(stall_reasons)}


def _parse_xla_dump(dump_dir: Path) -> Dict[str, float]:
    if not dump_dir.exists():
        return {}
    out: Dict[str, float] = {}
    for f in dump_dir.glob("*.txt"):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if "memory_throttle" in low:
                out["memory_throttle"] = out.get("memory_throttle", 0.0) + 0.05
            if "barrier" in low:
                out["barrier"] = out.get("barrier", 0.0) + 0.05
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def collect_device_stalls(arch: str, workload_cmd: List[str],
                          out_dir: Path) -> Optional[Dict[str, Any]]:
    if arch not in ARCH_TABLE:
        return None
    vendor = ARCH_TABLE[arch].vendor
    if vendor == "cuda":
        result = collect_nvidia_stalls(workload_cmd, out_dir)
    elif vendor == "hip":
        result = collect_amd_stalls(workload_cmd, out_dir)
    elif vendor == "pallas":
        result = collect_pallas_stalls(workload_cmd, out_dir)
    else:
        return None
    if result is not None:
        result.setdefault("arch", arch)
    return result


def _stall_to_bias_hints(stall_reasons: Dict[str, float]
                         ) -> Dict[str, List[Any]]:
    hints: Dict[str, List[Any]] = {}
    if not stall_reasons:
        return hints
    top = sorted(stall_reasons.items(), key=lambda x: -x[1])[:5]
    for reason, frac in top:
        if frac < 0.05:
            continue
        for dim in STALL_DIM_HINTS.get(reason, []):
            hints.setdefault(dim, [])
    if stall_reasons.get("long_scoreboard", 0) > 0.2:
        hints.setdefault("swizzle", []).extend([64, 128])
        hints.setdefault("lds_padding", []).extend([8, 16])
    if stall_reasons.get("not_selected", 0) > 0.15:
        hints.setdefault("waves_per_eu", []).extend([2, 4])
        hints.setdefault("maxrregcount", []).extend([64, 96])
    if stall_reasons.get("memory_throttle", 0) > 0.2:
        hints.setdefault("vec", []).extend([4, 8])
        hints.setdefault("async_depth", []).extend([4, 8])
    if stall_reasons.get("math_pipe_throttle", 0) > 0.2:
        hints.setdefault("unroll", []).extend([4, 8])
        hints.setdefault("num_stages", []).extend([3, 4, 5])
    if stall_reasons.get("lds_bank_conflict", 0) > 0.15:
        hints.setdefault("lds_padding", []).extend([4, 8, 16])
    for k in list(hints.keys()):
        hints[k] = sorted(set(hints[k]))
    return hints


def write_stall_sidecar(stall_info: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "device_stall_info.json"
    path.write_text(json.dumps(stall_info, indent=2, default=str))
    return path


def read_stall_sidecar(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = out_dir / "device_stall_info.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def bias_trial_queue(study, stall_info: Optional[Dict[str, Any]],
                     space: Dict[str, Any], arch: str,
                     *, max_enqueued: int = 25) -> int:
    if not stall_info or "bias_hints" not in stall_info:
        return 0
    hints = stall_info["bias_hints"]
    if not hints:
        return 0
    dims = space.get("dims", []) if isinstance(space, dict) else []
    if not dims:
        return 0
    base: Dict[str, Any] = {}
    for d in dims:
        name = d.get("name")
        vals = d.get("values") or []
        if name is None or not vals:
            continue
        base[name] = vals[len(vals) // 2]
    enqueued = 0
    for dim_name, preferred_vals in hints.items():
        if dim_name not in base:
            continue
        for val in preferred_vals:
            cfg = dict(base)
            cfg[dim_name] = val
            try:
                study.enqueue_trial(cfg)
                enqueued += 1
                if enqueued >= max_enqueued:
                    return enqueued
            except Exception:
                continue
    return enqueued


def run_device_pgo_round(spec, workload_cmd: List[str],
                         out_dir: Path, report=None) -> Optional[Path]:
    if not getattr(spec, "enable_device_pgo", False):
        return None
    try:
        stall_info = collect_device_stalls(spec.arch, workload_cmd, out_dir)
        if stall_info is None:
            if report is not None:
                report.write(f"  [device-pgo] no profiler available for arch "
                             f"{spec.arch}\n")
            return None
        if "error" in stall_info:
            if report is not None:
                report.write(f"  [device-pgo] {stall_info['error']}\n")
            return None
        sidecar = write_stall_sidecar(stall_info, out_dir)
        if report is not None:
            report.write(
                f"  [device-pgo] {stall_info.get('tool')} -> {sidecar}; "
                f"{len(stall_info.get('stall_reasons', {}))} stall "
                f"categories, "
                f"{len(stall_info.get('bias_hints', {}))} bias hints\n")
        return sidecar
    except Exception as exc:
        if report is not None:
            report.write(f"  [device-pgo] FAILED: {exc}\n")
        return None


# ------------------------------------------------------------------------------
# TOML project config (formerly grokking_optimizers/compile_config.{py,toml})
# ------------------------------------------------------------------------------

# Default config — was grokking_optimizers/compile_config.toml.
_DEFAULT_PROJECT_CONFIG: Dict[str, Any] = {
    "project": {"name": "supergrok", "version": "2.0.0"},
    "sources": {
        "cuda_root":      "csrc/backends/cuda",
        "hip_root":       "csrc/backends/hip",
        "pallas_root":    "csrc/backends/pallas",
        "algorithms_dir": "csrc/algorithms",
        "bindings_dir":   "csrc/bindings",
        "include_paths":  {"extra": []},
    },
    "optimizers": {
        "enabled": ["adamw", "lion", "muon", "prodigy", "grokadamw",
                    "grokfast", "looksam", "neuralgrok",
                    "supergrok11", "supergrok15", "supergrok2"],
    },
    "models": {"enabled": ["mamba3", "transformer_decoder", "vit"]},
    "archs":  {"default": "sm_90a", "allowed": []},
    "pgo":    {"workload_script": "", "steps": 1000},
    "autotune": {
        "min_improvement": 0.005,
        "patience": 0,
        "max_seconds": 0,
    },
    "codegen": {
        "enable_emitter": False,
        "template_dir": "grokking_optimizers/templates",
    },
    "runtime_specialization": {"enable": False, "cache_dir": ""},
    "device_pgo": {"enable": False},
    "cache": {
        "auto_prune_after_jit": True,
        "max_age_days": 30,
        "keep_top_n": 100,
    },
    "numerics": {"strict": False},
}

# Filenames kept as constants for any caller that still wants to point at a
# user-supplied TOML file alongside the inlined defaults.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "compile_config.toml"
CWD_CONFIG_NAME = "compile_config.toml"


def _load_toml_file(path: Path) -> Dict[str, Any]:
    """Use stdlib tomllib (3.11+), fall back to tomli, then a tiny parser."""
    if sys.version_info >= (3, 11):
        import tomllib
        with path.open("rb") as f:
            return tomllib.load(f)
    try:
        import tomli
        with path.open("rb") as f:
            return tomli.load(f)
    except ImportError:
        data: Dict[str, Any] = {}
        section = None
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
            m = _re_consolidated.match(r"\[([\w.]+)\]", line)
            if m:
                section = m.group(1)
                cur = data
                for part in section.split("."):
                    cur = cur.setdefault(part, {})
                continue
            m = _re_consolidated.match(r"(\w+)\s*=\s*(.+)", line)
            if m and section is not None:
                key, raw = m.group(1), m.group(2).strip()
                cur = data
                for part in section.split("."):
                    cur = cur.setdefault(part, {})
                try:
                    v = eval(raw, {"__builtins__": {}},
                             {"true": True, "false": False})
                except Exception:
                    v = raw.strip('"').strip("'")
                cur[key] = v
        return data


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Merge in priority order: caller path > CWD compile_config.toml > inlined defaults."""
    layers: List[Dict[str, Any]] = [_DEFAULT_PROJECT_CONFIG]
    cwd_path = Path.cwd() / CWD_CONFIG_NAME
    if cwd_path.exists():
        try:
            layers.append(_load_toml_file(cwd_path))
        except Exception:
            pass
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--config: {p} not found")
        layers.append(_load_toml_file(p))
    return _deep_merge(*layers)


def _deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if (k in out and isinstance(out[k], dict)
                    and isinstance(v, dict)):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
    return out


def apply_to_buildspec(spec, config: Dict[str, Any]) -> None:
    """Mutate spec in place: apply config feature toggles when off in spec."""
    if "codegen" in config:
        if not getattr(spec, "enable_emitter", False) and \
                config["codegen"].get("enable_emitter"):
            spec.enable_emitter = True
    if "runtime_specialization" in config:
        if not getattr(spec, "enable_runtime_specialization", False) and \
                config["runtime_specialization"].get("enable"):
            spec.enable_runtime_specialization = True
    if "device_pgo" in config:
        if not getattr(spec, "enable_device_pgo", False) and \
                config["device_pgo"].get("enable"):
            spec.enable_device_pgo = True
    if "numerics" in config:
        if not getattr(spec, "strict_numerics", False) and \
                config["numerics"].get("strict"):
            spec.strict_numerics = True
    if "cache" in config:
        if not config["cache"].get("auto_prune_after_jit", True):
            spec.prune_after_autotune = False
        spec.prune_max_age_days = int(
            config["cache"].get("max_age_days", 30))
        spec.prune_keep_top_n = int(
            config["cache"].get("keep_top_n", 100))


def project_sources(config: Dict[str, Any], vendor: str,
                    arch: str) -> Dict[str, Path]:
    src = config.get("sources", {})
    root_key = {"cuda": "cuda_root", "hip": "hip_root",
                "pallas": "pallas_root"}[vendor]
    root = src.get(root_key, f"csrc/backends/{vendor}")
    if vendor == "pallas":
        return {"backend": Path(root)}
    return {"backend": Path(root) / arch}


def allowed_optimizers(config: Dict[str, Any]) -> Optional[list]:
    opts = config.get("optimizers", {}).get("enabled", [])
    return opts if opts else None


def allowed_models(config: Dict[str, Any]) -> Optional[list]:
    mds = config.get("models", {}).get("enabled", [])
    return mds if mds else None


def allowed_archs(config: Dict[str, Any]) -> Optional[list]:
    ar = config.get("archs", {}).get("allowed", [])
    return ar if ar else None


def default_arch(config: Dict[str, Any]) -> Optional[str]:
    return config.get("archs", {}).get("default") or None


# ------------------------------------------------------------------------------
# Legacy import shim — register the four absorbed modules in sys.modules so
# existing ``from grokking_optimizers.codegen import X`` (etc.) imports still
# resolve. Each alias points at this module, so every absorbed top-level
# symbol is reachable via the legacy dotted path.
# ------------------------------------------------------------------------------

def _register_consolidated_module_aliases() -> None:
    """Make compile.py answer to its four absorbed submodule paths."""
    me = sys.modules[__name__]
    for legacy_name in ("codegen", "kernel_registry",
                        "device_profiling", "compile_config"):
        sys.modules[f"grokking_optimizers.{legacy_name}"] = me
    # If the package object is reachable, set attributes too so
    # ``from grokking_optimizers import codegen`` resolves cleanly.
    pkg = sys.modules.get("grokking_optimizers")
    if pkg is not None:
        for legacy_name in ("codegen", "kernel_registry",
                            "device_profiling", "compile_config"):
            setattr(pkg, legacy_name, me)


_register_consolidated_module_aliases()



if __name__ == "__main__":
    raise SystemExit(main())
