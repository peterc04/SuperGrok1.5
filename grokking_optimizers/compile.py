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
         applies_to: List[str]) -> Dict[str, Any]:
    return {"name": name, "type": dtype, "values": values,
            "macro": macro, "applies_to": applies_to}


def _hopper_cluster_shapes() -> List[List[int]]:
    """Every (m, n, p) ∈ {1,2,4,8}³ with m·n·p ≤ 8 (Hopper hw limit)."""
    out: List[List[int]] = []
    for m in (1, 2, 4, 8):
        for n in (1, 2, 4, 8):
            for p in (1, 2, 4, 8):
                if m * n * p <= 8:
                    out.append([m, n, p])
    return out


def _sm90_full_space() -> Dict[str, Any]:
    return {
        "dims": [
            # block: every warp-multiple from 32 up to the 1024 thread-per-CTA max.
            _dim("block", "int", list(range(32, 1025, 32)),
                 "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
            # vec: 1..16 — covers byte/short/int/long via LDG.E.{32,64,128}.
            _dim("vec", "int", [1, 2, 4, 8, 16],
                 "SG_TUNED_VEC_WIDTH", ["host", "device"]),
            # unroll: powers of 2 up to 128 — broad sweep of compiler hints.
            _dim("unroll", "int", [1, 2, 4, 8, 16, 32, 64, 128],
                 "SG_TUNED_UNROLL", ["host", "device"]),
            # num_stages: full Hopper async-pipeline depth range.
            _dim("num_stages", "int", list(range(1, 9)),
                 "SG_TUNED_NUM_STAGES", ["device"]),
            # maxrregcount: every 4 from 32 to 252, plus the hard cap 255.
            _dim("maxrregcount", "int", list(range(32, 253, 4)) + [255],
                 None, ["device"]),  # promoted to NVCC flag, not -D
            # cluster_shape: every Hopper-legal (m, n, p) cluster geometry.
            _dim("cluster_shape", "tuple", _hopper_cluster_shapes(),
                 "SG_TUNED_CLUSTER_SHAPE", ["device"]),
            # swizzle: shared-memory bank-conflict avoidance patterns.
            _dim("swizzle", "enum", ["none", "xor2", "xor4", "xor8", "xor16"],
                 "SG_TUNED_SWIZZLE", ["device"]),
            _dim("warp_specialization", "bool", [False, True],
                 "SG_TUNED_WARP_SPECIALIZATION", ["device"]),
            _dim("tma", "bool", [False, True],
                 "SG_TUNED_TMA", ["device"]),
            # async_depth: 1..16 (cp.async pipeline depth on Hopper).
            _dim("async_depth", "int", list(range(1, 17)),
                 "SG_TUNED_ASYNC_DEPTH", ["device"]),
        ],
        "prefilter": {
            "register_pressure_max": 255,
            "smem_budget_bytes": 232448,   # 227 KB usable smem per Hopper SM
            "rules": [
                # Hopper: ≤32 warps per block.
                {"name": "warps_per_block", "expr": "(block // 32) <= 32"},
                # Vec loads need block % (vec * bytes_per_elem) == 0.
                {"name": "vec_block_alignment",
                 "expr": "block % (vec * 4) == 0"},
                # Pipelined kernels need enough warps to hide latency.
                {"name": "stages_block",
                 "expr": "num_stages * vec <= block // 32"},
                # TMA + warp specialisation need ≥ 1 dedicated warp.
                {"name": "tma_requires_block",
                 "expr": "(not tma) or block >= 128"},
                {"name": "warpspec_requires_block",
                 "expr": "(not warp_specialization) or block >= 128"},
                # Cluster volume limit is 8 CTAs (Hopper hardware cap).
                {"name": "cluster_volume",
                 "expr": "cluster_shape[0] * cluster_shape[1] * cluster_shape[2] <= 8"},
                # Async depth ≥ stages-1 to keep the pipeline busy.
                {"name": "async_depth_stages",
                 "expr": "async_depth >= num_stages - 1"},
            ],
        },
    }


def _gfx942_full_space() -> Dict[str, Any]:
    return {
        "dims": [
            # block: every wavefront-multiple (64) up to 1024 threads/CTA.
            _dim("block", "int", list(range(64, 1025, 64)),
                 "SG_TUNED_BLOCK_SIZE", ["host", "device"]),
            # vec: 1..8 — CDNA dwordx4 caps the meaningful vec at 4 for f32.
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
            # MFMA tile shapes — every public shape MI300X exposes.
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
            "smem_budget_bytes": 65536,   # 64 KB LDS per CU on CDNA3
            "rules": [
                # CDNA3 wavefront = 64; block must be a multiple.
                {"name": "wave_alignment", "expr": "block % 64 == 0"},
                # ≤16 waves per block (1024 threads / 64 lanes).
                {"name": "waves_per_block", "expr": "(block // 64) <= 16"},
                # Occupancy ceiling: waves_per_eu * waves_per_block ≤ 20.
                {"name": "waves_per_eu_total",
                 "expr": "waves_per_eu * (block // 64) <= 20"},
                {"name": "vec_block_alignment",
                 "expr": "block % (vec * 4) == 0"},
                {"name": "mfma_block_min", "expr": "block >= 64"},
            ],
        },
    }


def build_full_search_space() -> Dict[str, Any]:
    """Return the COMPLETE per-arch search space.

    No curated value lists; no hand-picked subsets. Every dim covers the
    full range its hardware actually supports. The Cartesian product is
    in the billions per arch — never materialize it as a list; use the
    streaming ``cartesian()`` iterator and ``cartesian_count()`` instead.

    Override with ``--search-space <path.yaml>`` if you genuinely want a
    smaller curated space (e.g. for fast CI sweeps); the YAML schema is
    the same shape as this dict.
    """
    return {
        "sm_90":   _sm90_full_space(),
        "gfx942":  _gfx942_full_space(),
        "tpu_v5p": {"dims": [], "prefilter": {"rules": []}},
    }


# Wire the existing search-space builders into the ARCH_TABLE entries.
# Stream 2 owns extending these to the rest of the arches; the assignment
# propagates through aliases because both keys reference the same ArchEntry.
ARCH_TABLE["sm_90a"].search_space_builder = _sm90_full_space
ARCH_TABLE["gfx942"].search_space_builder = _gfx942_full_space


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
                             dim_specs: List[Dict[str, Any]]) -> List[str]:
    """Some dims become bare NVCC/HIPCC flags rather than ``-D`` macros.
    Currently: ``maxrregcount`` -> ``--maxrregcount=N``."""
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.append(f"--maxrregcount={int(v)}")
    return out


def resolve_extra_hipcc_flags(config: Dict[str, Any],
                              dim_specs: List[Dict[str, Any]]) -> List[str]:
    """HIPCC analogue. ``maxrregcount`` -> ``-mllvm -amdgpu-max-num-vgprs=N``."""
    out: List[str] = []
    for spec in dim_specs:
        if spec.get("macro") is None and spec["name"] == "maxrregcount":
            v = config.get("maxrregcount")
            if v is not None:
                out.extend(["-mllvm", f"-amdgpu-max-num-vgprs={int(v)}"])
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


class TimingWorker:
    """Persistent timing subprocess; one warm CUDA context for an entire sweep."""

    def __init__(self, opt_class: str, *,
                 size: int = 4096, warmup: int = 5, iters: int = 21,
                 use_cuda_graph: bool = True,
                 timeout_per_variant: int = 180,
                 env: Optional[Dict[str, str]] = None,
                 cwd: Optional[Path] = None,
                 python: Optional[str] = None):
        self.opt_class = opt_class
        self.size = size
        self.warmup = warmup
        self.iters = iters
        self.use_cuda_graph = use_cuda_graph
        self.timeout = timeout_per_variant
        self.env = env or os.environ.copy()
        self.cwd = cwd
        self.python = python or sys.executable
        self._proc: Optional[subprocess.Popen] = None
        self._error_log: list = []

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
        return True

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def stop(self) -> None:
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
        return result

    def ping(self) -> bool:
        if not self.alive():
            return False
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


# ===========================================================================
# bayesian — Optuna TPE-driven autotune (absorbed from
# grokking_optimizers/bayesian.py)
# ===========================================================================

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
    return {
        "trial_num":   trial_num,
        "stage":       stage,
        "config":      {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
        "config_key":  config_key(cfg),
        "timing_ms":   result["timing_ms"] if result else None,
        "min_ms":      result["min_ms"]    if result else None,
        "max_ms":      result["max_ms"]    if result else None,
        "n":           result["n"]         if result else None,
        "host":        host,
        "recorded_at": datetime.datetime.now().isoformat(),
    }


def run_bayesian(
    arch: str,
    space: Dict[str, Any],
    *,
    n_trials: int = 500,
    seed: int = 0,
    storage: Optional[Path] = None,
    study_name: str = "sg_tune",
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
    pruner: str = "none",
    seed_trials: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run the TPE stage. Returns the list of trial records."""
    dims = space[arch]["dims"]
    # Build a per-config feasibility predicate from the prefilter rules
    # so we can validate TPE suggestions in O(rules) without enumerating
    # the (billions-of-configs) full Cartesian survivor set.
    is_feasible = compile_feasibility_check(
        space[arch].get("prefilter", {}))

    sampler = TPESampler(
        seed=seed,
        n_startup_trials=max(10, n_trials // 10),
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
        seeded = 0
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
                seeded += 1
            except Exception:
                continue

    records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest(trial, dims)
        if not is_feasible(cfg):
            rec = _make_trial_record("tpe", trial.number, cfg, None, host=host)
            rec["status"] = "infeasible"
            records.append(rec)
            if progress:
                progress(trial.number + 1, n_trials, cfg)
            return math.inf
        result = timer(cfg)
        rec = _make_trial_record("tpe", trial.number, cfg, result, host=host)
        rec["status"] = "ok" if result else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(trial.number + 1, n_trials, cfg)
        if result is None:
            return math.inf
        return float(result["timing_ms"])

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   show_progress_bar=False)
    return records


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
    top_k: int = 20,
    radius: int = 2,
    timer: Timer,
    progress: ProgressCb = None,
    host: Optional[Dict[str, Any]] = None,
    prefiltered: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """For each of the top-K TPE trials, time the +/-radius-step
    neighbours along every dim. Returns the refine-stage records."""
    dims = space[arch]["dims"]
    is_feasible = compile_feasibility_check(
        space[arch].get("prefilter", {}))

    successes = [t for t in bayes_trials if t["timing_ms"] is not None]
    successes.sort(key=lambda t: t["timing_ms"])
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
        result = timer(cfg)
        rec = _make_trial_record("refine", i, cfg, result, host=host)
        rec["status"] = "ok" if result else "build_or_time_fail"
        records.append(rec)
        if progress:
            progress(i, total, cfg)
    return records


def pick_winner(all_trials: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Lowest timing across all stages. Returns the winning trial record
    (with ``config`` and ``timing_ms``) or ``None``."""
    finished = [t for t in all_trials if t["timing_ms"] is not None]
    if not finished:
        return None
    return min(finished, key=lambda t: t["timing_ms"])


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
    debug: bool = False               # mirror report to stderr + print every subproc
    # §12 A1 / A2 — Hyperband pruner + transfer learning
    pruner: str = "none"              # "none" | "median" | "hyperband"
    transfer_learning: bool = False
    # Stream 8 — device-side PGO (CUPTI / rocprof / XLA HLO dump)
    enable_device_pgo: bool = False


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


def _bootstrap_cuda_via_nvidia_apt_repo(stream) -> bool:
    """Add NVIDIA's official CUDA apt repo and install cuda-toolkit-XX-Y.

    Preferred over stock ``nvidia-cuda-toolkit`` on Debian/Ubuntu because:
      - Stock package is often years old (CUDA 11.5 on Ubuntu 22.04 —
        too old for sm_90, which needs CUDA 12.0+).
      - NVIDIA's repo installs to ``/usr/local/cuda-<ver>/`` with a
        ``/usr/local/cuda`` symlink — the exact layout torch's
        ``cpp_extension`` expects (bin/, lib64/, include/).
      - Version is picked to match ``torch.version.cuda`` when possible.

    Probes the available cuda-toolkit packages and installs the newest
    one ≥12.0 that matches torch's bundled CUDA. Falls back to 12.6,
    12.4 if no exact match exists in the repo metadata.
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

    # Pick CUDA toolkit version matching torch's bundled CUDA when possible.
    target_ver = "12-6"
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


def bootstrap_cuda_toolkit(stream=None) -> bool:
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
    "-dlto",
]

# HIPCC base flags — --offload-arch is appended per-build from
# ARCH_TABLE[arch].hipcc_offload_arch in _device_cflags().
HIPCC_DEVICE_BASE = [
    "-O3", "-std=c++17", "-DWITH_HIP",
    "-ffast-math", "-fPIC",
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
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "pallas":
        return []
    base = list(HOST_CFLAGS_BASE) + [f"-D{entry.host_define}"]
    if spec.debug_symbols or spec.profile:
        base += ["-ggdb"]
    return base + _build_macros(spec)


def _device_cflags(spec: BuildSpec) -> List[str]:
    entry = get_arch_entry(spec.arch)
    if entry.vendor == "cuda":
        base = list(NVCC_DEVICE_BASE)
        # Per-arch -gencode pair (target SASS + PTX fallback) from ARCH_TABLE.
        base += list(entry.nvcc_gencode)
        if spec.debug_symbols or spec.profile:
            base += ["-lineinfo", "--generate-line-info"]
        if (REPO_ROOT / "third_party/cutlass/include").exists():
            cutlass_arch_token = (
                f"{entry.cutlass_arch}{entry.arch_suffix}"
                if entry.cutlass_arch is not None else "")
            base += ["-DWITH_CUTLASS",
                     f"-DCUTLASS_NVCC_ARCHS={cutlass_arch_token}"]
        return base + _build_macros(spec)
    if entry.vendor == "hip":
        base = list(HIPCC_DEVICE_BASE)
        # Per-arch --offload-arch from ARCH_TABLE.
        if entry.hipcc_offload_arch:
            base += [f"--offload-arch={entry.hipcc_offload_arch}"]
        if spec.debug_symbols or spec.profile:
            base += ["-ggdb"]
        return base + _build_macros(spec)
    return []


def _ldflags(spec: BuildSpec) -> List[str]:
    if get_arch_entry(spec.arch).vendor == "pallas":
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
    no-op on older toolchains. Dispatch is by ARCH_TABLE vendor so every
    CUDA arch (not just sm_90) benefits from version-gated flags."""
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
            if ver >= (12, 6):
                # NVCC 12.6+ supports --split-compile for opt-phase parallelism.
                extra_device += [f"--split-compile={NCPUS}"]
                if report:
                    report.write(f"  [toolchain] enabling "
                                 f"--split-compile={NCPUS} (NVCC ≥12.6)\n")
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

def _variant_macros(config: Dict[str, Any], dims: List[Dict[str, Any]],
                    target: str) -> List[str]:
    """Macros + extra-flag overrides for one config × target."""
    macros = resolve_macros(config, dims, target)
    extra = resolve_extra_nvcc_flags(config, dims) if target == "device" else []
    extra_hip = resolve_extra_hipcc_flags(config, dims) if target == "device" else []
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
        ckey = config_key(config)
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
    if get_arch_entry(spec.arch).vendor == "pallas":
        report.write("  [jit-autotune] pallas backend; no C++ tuning.\n")
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

    # Spawn the persistent worker.
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
            ckey = config_key(cfg)
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
        step1(f"trial {done}/{total} key={config_key(cfg)[:24]}…")

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
    feasibility_check = compile_feasibility_check(
        space[spec.arch].get("prefilter", {}))
    n_refine_est = sum(1 for _ in _neighbour_estimate(
        refine_inputs[:spec.top_k], dims, feasibility_check))
    step2, close2 = make_progress(
        max(n_refine_est, 1), f"jit-refine {spec.optimizer}/{spec.arch}")

    def progress2(done, total, cfg):
        step2(f"refine {done}/{total} key={config_key(cfg)[:24]}…")

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
    out["config_key"] = config_key(out)
    out["stage_won"] = winner["stage"]
    report.write(f"\n  [bayesian] WINNER ({winner['stage']}): "
                 f"{out['config_key']} @ {out['timing_ms']:.4f}ms\n")
    cache.set_tuned(spec.optimizer, spec.model, spec.arch, out,
                    mode="bayesian", search_space_hash=space_hash)
    return out


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
        space = get_search_space(spec.search_space_path)
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
    debug: bool = False,
    bootstrap_cuda: bool = False,
    pruner: str = "none",
    transfer_learning: bool = False,
    enable_device_pgo: bool = False,
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
    # On-demand: pip-install the NVIDIA wheels if nvcc is still missing
    # and the caller opted in. Handles Colab CPU runtimes, fresh CI, etc.
    if bootstrap_cuda and get_arch_entry(arch).vendor == "cuda":
        if not shutil.which("nvcc"):
            bootstrap_cuda_toolkit()
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
        bayesian_trials=bayesian_trials, top_k=top_k, seed=seed,
        debug_symbols=debug_symbols, debug=debug,
        pruner=pruner, transfer_learning=transfer_learning,
        enable_device_pgo=enable_device_pgo,
    )
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
            report.write(f"# Bayesian trials:  {bayesian_trials} (top_k={top_k})\n")
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
    args = parser.parse_args(argv)
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

    # ── Runtime split: spawn AOT then JIT, then return ──────────────
    if args.runtime == "both":
        # Do the (possibly slow) CUDA toolkit bootstrap ONCE in the
        # parent so AOT and JIT subprocesses inherit the discovered
        # nvcc via PATH/CUDA_HOME — no per-subprocess re-install.
        if args.bootstrap_cuda and get_arch_entry(args.arch).vendor == "cuda":
            if not shutil.which("nvcc"):
                _ensure_nvcc_on_path() or bootstrap_cuda_toolkit()
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
        debug=args.debug,
        bootstrap_cuda=args.bootstrap_cuda,
        pruner=args.pruner,
        transfer_learning=args.transfer_learning,
        enable_device_pgo=args.enable_device_pgo,
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

    _run("load_yaml_validates_shape", test_load_yaml_validates_shape)
    _run("load_yaml_rejects_duplicate_dim", test_load_yaml_rejects_duplicate_dim)
    _run("embedded_yaml_loads", test_real_yaml_loads)
    _run("cartesian_counts", test_cartesian_counts)
    _run("prefilter_eliminates", test_prefilter_eliminates)
    _run("config_key_deterministic", test_config_key_deterministic)
    _run("hash_space_stable", test_hash_space_stable)

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
            trials = run_bayesian(
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
            trials = run_bayesian(
                "sm_90", _tiny_space(), n_trials=20, seed=0,
                storage=td / "study.db", timer=_synthetic_timer)
            refines = topk_refine(trials, _tiny_space(), "sm_90",
                                  top_k=5, radius=2, timer=_synthetic_timer)
            assert all(t["stage"] == "refine" for t in refines)
        finally:
            shutil.rmtree(td)

    _run("bayesian_finds_winner", test_bayesian_finds_winner)
    _run("topk_refine_generates_neighbours", test_topk_refine_generates_neighbours)

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

    sys.stdout.write(f"\n[self-test] {passed} passed, {failures} failed\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
