# REFRESH.md — SuperGrok1.5 Reference

A compact, granular catch-up. Plain language. Each kernel, file, and optimizer gets its own entry. No fluff, no withholding.

> **POST-REFACTOR.** The all-specialized per-arch refactor + arch-matrix expansion is complete. Eight GPU arches + two TPU versions are first-class citizens; there is no fallback chain and no generic-kernel tier. Subsequently the per-arch overlays have been folded into per-arch namespaces, the high-level vector-signature bindings have been restored from the deleted `csrc/common/ops.cpp`, and the cross-arch agreement test bodies have been filled in. **§0 below is the current state.** §1–§20 remain the pre-refactor reference and are still useful for understanding individual kernels and optimizers; they are not the source of truth on layout, dispatch, or bindings. Engineering work picks up from §0.5 ("What is NOT yet done") + §21 (overlay merges) + §22 (bindings state).

---

## 0. Post-refactor state (NEW)

### Supported arches (no fallback)

After the §10 expansion, eight GPU arches plus two TPU versions are supported:

- **NVIDIA**:
  - `sm_80` — Ampere family. A100, A30, A10. RTX 30-series and sm_86 route here.
  - `sm_89` — Ada Lovelace. RTX 40-series, L40, L40S.
  - `sm_90` — Hopper. H100, H200.
  - `sm_100` — Datacenter Blackwell. B100, B200, GB200.
  - `sm_103` — Blackwell Ultra. B300, GB300 NVL72. NVFP4 hot path.
  - `sm_120` — Consumer Blackwell. RTX 50-series, RTX PRO 6000 Blackwell. 128KB shared memory per SM.
- **AMD**:
  - `gfx942` — CDNA3. MI300X, MI300A.
  - `gfx950` — CDNA4. MI350X, MI355X. Native FP4 expert MFMA, FP6 state, 2:4 sparsity.
- **TPU** (JAX path): `v5p` (128-wide MXU), `v6e` (256-wide MXU).
- **CPU**: testing only, not a runtime fallback.

Anything else raises `UnsupportedArchError` or fails the build. Permanently unsupported: V100, T4, RTX 20-series, MI100 (gfx908), MI200 (gfx90a), AMD RDNA, TPU v3/v4/v5e.

### Filesystem layout

```
csrc/
├── common/          (shared headers; tuned_configs.h with 8-row tables)
├── bindings/        (per-optimizer dispatchers + pybind11 module)
│                     _helpers.h with clip-grad / SAM-norm host helpers
│                     _dispatch_macro.h with SG_DISPATCH + SG_DISPATCH_CALL
└── kernels/
    ├── cuda/
    │   ├── sm_80/   (17 wrapped baselines + 5 arch-tuned files in
    │   │            sg::sm80 namespace: muon TF32 inlined into canonical;
    │   │            metanet_optimizers, metanet_cpasync_variants,
    │   │            supergrok2_{backward,fused_elem,scan} folded in)
    │   ├── sm_89/   (17 wrapped baselines, ported from sm_90)
    │   ├── sm_90/   (17 wrapped baselines + 3 arch-tuned files in
    │   │            sg::sm90 namespace: muon FP8 inlined into canonical;
    │   │            supergrok2_{backward,scan,warp_specialized} folded in)
    │   ├── sm_100/  (17 wrapped baselines + 2 arch-tuned files in
    │   │            sg::sm100 namespace: supergrok2_{precompute,scan}
    │   │            folded in. Blackwell delegating overlay deleted —
    │   │            per-arch baselines are sufficient)
    │   ├── sm_103/  (17 wrapped baselines, ported from sm_100)
    │   └── sm_120/  (17 wrapped baselines, ported from sm_100)
    ├── hip/
    │   ├── gfx942/  (17 wrapped baselines + 1 arch-tuned file in
    │   │            sg::gfx942 namespace: muon BF16 MFMA inlined into
    │   │            canonical; supergrok2_gfx942 folded in. Trivial
    │   │            metanet delegating overlay deleted)
    │   └── gfx950/  (17 wrapped baselines + cdna4_kernels_gfx950.hip.cpp
    │                 in sg::gfx950 namespace: FP4/FP6/2:4 sparsity
    │                 kernels recovered from git history. Per-feature
    │                 split deferred — see §21)
    ├── tpu/
    │   ├── _pallas_kernels.py  (shared Pallas implementation)
    │   ├── v5p/                 (tile-128 re-export for v4/v5e/v5p)
    │   ├── v6e/                 (tile-256 re-export for v6e)
    │   └── __init__.py          (detect_tpu_version + get_kernels)
    └── cpu/         (avx512/, neon/, scalar; testing only)
autotune/            (tune.py, grids.py with NVFP4/FP4/FP6 entries,
                     cutlass_profile.py for sm_89/103a/120a, runner.py)
```

Removed entirely: `csrc/cuda/generic/`, `csrc/cuda/generated/`,
`csrc/cuda/sm_75/86/89/`, `csrc/hip/cdna2/cdna3/cdna4/`, `csrc/cpu/`
(moved to `csrc/kernels/cpu/`), `csrc/common/{ops.h,ops.cpp,dispatch.h}`,
`grokking_optimizers/jit/`, `grokking_optimizers/jit_kernels.py`,
`codegen/`. All `*_overlay.*` files have been merged into per-arch
namespaced kernels (this session's work — see §21).

### Per-arch kernel pattern

Each per-arch source file is wrapped in `namespace sg::<arch> { ... }`
so the eight translation units do not collide on launcher / kernel
symbols at link time. The 8 × 17 = 136 wrapped baselines start out
with identical math; per-arch divergence (cp.async, TMA, MFMA, FP8
paths, warp specialization) is layered on top via the merged-overlay
files in §21 and via direct edits to canonical launchers (e.g. the
muon Ampere TF32 / Hopper FP8 / CDNA3 BF16 MFMA paths inlined this
session — see §21). Cross-arch numerical agreement is guarded by
`tests/test_cross_arch_agreement.py` (now with filled-in test bodies
for every optimizer including a Muon 2D-param harness).

The `*_overlay.*` naming convention is gone. All overlay files have
been folded into per-arch namespaced canonical files.

### Bindings layer

- `csrc/bindings/bindings.h` — shared declarations and arch namespace anchors
- `csrc/bindings/dispatch.cpp` — `detect_arch()`: returns one of
  `{80, 89, 90, 100, 103, 120, 942, 950}` or raises (no fallback chain)
- `csrc/bindings/_dispatch_macro.h` — two macros:
  - `SG_DISPATCH(method, args...)` — early-returns from the enclosing
    function. Use at the tail of a per-tensor wrapper.
  - `SG_DISPATCH_CALL(method, args...)` — same dispatch with `break`
    instead of `return`. Use inside loops.
- `csrc/bindings/_helpers.h` — host-side helpers shared by per-optimizer
  binding files: `clip_grad_norms_device_side`,
  `compute_sam_grad_norm_device_side`. Both do single-CPU-sync norm
  reductions over a `std::vector<torch::Tensor>` (extracted verbatim
  from the deleted `csrc/common/ops.cpp`).
- `csrc/bindings/<optimizer>.cpp` — per-optimizer dispatcher. Each file
  exposes both per-tensor wrappers (escape hatches for tests) and
  high-level vector-signature entry points matching the pre-refactor
  `ops.cpp`. The Python optimizers in `grokking_optimizers/*.py` call
  the latter (e.g. `_ops.muon_fused_step(params, grads, bufs, ...)`).
- `csrc/bindings/module.cpp` — pybind11 aggregator. Registers every
  vector-signature entry point + every per-tensor wrapper. SG v2 entry
  points remain unregistered — see §22.

### dispatch.py / __init__.py changes

- `dispatch.get_gpu_arch()` returns one of `{80, 90, 100, 942}` or raises `UnsupportedArchError`
- Tier helpers (`get_arch_tier`, `get_amd_tier`, `get_amd_label`) are gone
- `assert_supported_arch()` and `SUPPORTED_ARCHES` are new public surface
- `__version__` bumped to `3.0.0` (breaking)
- `_HAS_OPS` simplified: extension or error

### Autotune

`autotune/tune.py` is the offline tuner: runs each kernel × arch × shape × config grid, picks the median-fastest, writes `csrc/common/tuned_configs.h`. The header is committed. CUTLASS GEMMs (SG2 projections, Muon Newton-Schulz) go through `cutlass_profile.py`. Currently scaffolding — `runner.py:bench()` and `cutlass_profile.py:profile_gemm()` raise `NotImplementedError`; they need a hardware-equipped session to wire to `torch.utils.cpp_extension` and the CUTLASS profiler binary.

### TPU dispatch

Pallas kernels moved from `supergrok2_jax_tpu/pallas_kernels.py` to `csrc/kernels/tpu/`. The shared implementation is `_pallas_kernels.py`; v5p re-exports the tile-128 variants, v6e re-exports the tile-256 variants. `csrc/kernels/tpu/__init__.py:get_kernels()` selects based on `detect_tpu_version()`. `supergrok2_jax_tpu/pallas_kernels.py` is now a backwards-compat shim that re-exports from the new path.

### Code statistics (post-refactor)

180,845 lines of code across 267 tracked files. Breakdown by language:

| Language | Files | Lines | Notes |
|----------|------:|------:|-------|
| CUDA C++ (`.cu` / `.cuh`) | 119 | 112,822 | Per-arch wrapped baselines (17 optimizers × 6 NVIDIA arches = 102 files) + 14 `*_overlay.cu` files preserved as future-merge targets + 2 shared `.cuh` headers (`ptx_intrinsics.cuh`, `utils.cuh`) + 1 quantization kernel. |
| HIP C++ (`.hip.cpp`) | 38 | 38,050 | Per-arch wrapped baselines for gfx942 + gfx950 (17 optimizers × 2 = 34) + 3 gfx942 `*_overlay` files + 1 recovered CDNA4 FP4/FP6/2:4 sparsity overlay (`cdna4_kernels_gfx950_overlay.hip.cpp`, 2491 lines). |
| Python (`.py`) | 71 | 21,342 | `grokking_optimizers/` package (eleven optimizers + dispatch + bindings + distributed + quantization + cuda_graph wrappers), `supergrok2_jax_tpu/` JAX implementation, `csrc/kernels/tpu/_pallas_kernels.py` (1190 lines), `autotune/` scripts, tests. |
| C++ host code (`.cpp` / `.h`) | 31 | 5,523 | `csrc/bindings/` per-optimizer dispatchers + module aggregator (~16 files), shared headers (`platform.h`, `types.h`, `quantization.h`, `tuned_configs.h`, `bindings.h`), CPU testing-only sources (`csrc/kernels/cpu/`). |
| Markdown docs (`.md`) | 7 | 3,089 | `README.md`, `REFRESH.md` (this file), `REFACTOR_PLAN.md`, `ANALYSIS.md`, plus per-tree READMEs (`csrc/kernels/README.md`, `csrc/kernels/hip/README_HIP.md`, `autotune/README.md`). |
| Config (TOML) | 1 | 19 | `pyproject.toml`. |
| **Total** | **267** | **180,845** | |

The CUDA + HIP totals include large amounts of structurally-identical baseline content (the 68 + 68 wrapped baselines from the original 4-arch refactor + the §10 expansion are byte-equivalent within each optimizer's per-arch family at this stage). Real divergence — cp.async vs TMA vs MFMA, FP8 vs NVFP4 vs FP4 paths, warp specialization — is added per-arch under hardware-validated tuning passes; the cross-arch numerical agreement test (`tests/test_cross_arch_agreement.py`) catches drift.

Lines of code in the eight `*_overlay.*` pre-tuned files (Hopper FP8 / Ampere cp.async / Blackwell TMA / gfx942 BF16 MFMA / CDNA4 FP4-FP6) account for roughly 12,000 lines of the CUDA/HIP totals; these files are excluded from the build until merged into the canonical per-arch kernels.

### 0.5 What is NOT yet done

The all-specialized refactor, arch-matrix expansion, overlay merges,
SG v2 binding wiring, ninja build wrapper, autotune execution layer,
and CUTLASS scaffolding are now all complete. Remaining work is
engineering-focused, deferred until hardware. See §25 for a structured
list and §24 for the per-arch per-optimizer plan.

Items that have been **completed since the previous REFRESH.md edit**:

- **SG v2 bindings**: all seven SG2 entry points
  (`supergrok2_mamba_peer_step`, `supergrok2_mamba_peer_batched_step`,
  `supergrok2_bilevel_fwd_save{_batched}`,
  `supergrok2_bilevel_backward{_batched}`,
  `supergrok2_prepare_and_batched_step`) are wired in
  `csrc/bindings/supergrok2.cpp` and registered in
  `csrc/bindings/module.cpp`. Each is a thin SG_DISPATCH wrapper
  around the per-arch launcher. See §22.1.
- **Inlining of arch-suffixed launchers**: the §21.4 inlining pass
  is complete for sm_80 (Ampere TF32 wrap inlined into supergrok15 /
  supergrok11 / neuralgrok / sg2 backward / sg2 scan canonicals),
  sm_100 (Blackwell prefix dropped from supergrok2_precompute /
  supergrok2_scan symbol names), and gfx942 (CDNA3 BF16 MFMA inlined
  into mamba_peer_batched_step canonical). sm_90 cleaned up trivial
  delegators; sm_90 FP8 fast-path inlining is **deferred** because
  the existing hopper-suffixed launcher's non-FP8 fallback called
  `ampere_*` symbols defined only in `sg::sm80` (not visible from
  `sg::sm90`). Full FP8 inline requires the architecture restructuring
  noted in §25.
- **gfx950 CDNA4 split**: the 2491-line `cdna4_kernels_gfx950.hip.cpp`
  monolith has been split into four per-feature files
  (`fp4_expert_kernels`, `fp6_state_kernels`, `sparse24_kernels`,
  `fused_combos`) with shared FP4/FP6 helpers extracted into
  `csrc/common/fp4_helpers.hip.h`. All helpers are
  `__device__ static __forceinline__` so each TU gets internal-linkage
  copies and avoids ODR errors.
- **Ninja-backed build with multi-arch AOT fatbin**: `setup.py` now
  uses `BuildExtension.with_options(use_ninja=True)`, emits
  `-gencode arch=compute_X,code=sm_X` for every supported arch plus
  PTX embedding of `compute_120` for forward-compat driver JIT, and
  detects ccache/sccache via `CMAKE_*_COMPILER_LAUNCHER`. `build.sh`
  wraps `pip install -e . --no-build-isolation -v` with a tqdm
  progress bar that parses ninja `[N/M]` lines, plus `--autotune`
  (two-pass), `--debug` (cuda-gdb), and `--profile` (NCU) modes.
  `pyproject.toml` adds `ninja` to build-system requires; bumped
  to version 3.0.0.
- **Autotune execution layer**: `autotune/runner.py:bench()` records
  one `(start, stop)` CUDA event pair per iteration with a single
  post-loop `torch.cuda.synchronize()`, returns median microseconds.
  `autotune/cutlass_profile.py:profile_gemm()` invokes the
  `cutlass_profiler` binary, parses CSV output (column-name drift
  tolerant), returns sorted-by-latency results. `autotune/tune.py`
  orchestrates per-arch grids and writes winners between
  `// AUTOTUNE_BEGIN` / `// AUTOTUNE_END` markers in
  `tuned_configs.h`. The C++ `LaunchConfig` table itself is left
  intact for hand-tuned fallback values.
- **CUTLASS migration scaffold**: `third_party/cutlass` is registered
  as a v3.6.0 submodule (not cloned in this session — the user runs
  `git submodule update --init` later). `setup.py` adds CUTLASS
  include dirs + `-DCUTLASS_NVCC_ARCHS=90a/100a/103a/120a` + a new
  `WITH_CUTLASS=1` opt-in flag. `csrc/kernels/cuda/_cutlass_gemm.cuh`
  exposes `cutlass_gemm_fp16/bf16` and `cutlass_dt_proj_fused`
  helpers (the latter currently runs the unfused linear-combo GEMM
  and relies on the pre-existing `softplus_bias_kernel` post-pass —
  fused softplus epilogue is deferred per §25). Muon Newton-Schulz
  GEMMs (sm_90/100/103/120) and the SG2 five-projection set
  (`in_proj_x`, `in_proj_z`, `dt_proj`, `B_proj`, `C_proj` — actually
  in `supergrok2_bwd_*.cu`'s `bilevel_precompute_gemm` helper, not
  the fwd files as initially expected) are wrapped in
  `#ifdef WITH_CUTLASS` so the existing torch::mm path stays the
  default until the user opts in. sm_80/sm_89 keep cuBLAS, gfx942/950
  keep rocBLAS. `tests/test_cutlass_parity.py` asserts CUTLASS output
  matches cuBLAS within 1e-3 (FP16) or 1e-4 (BF16).
- **Build-error post-merge audit**: 41 stale `#include "ops.h"` and
  `#include "dispatch.h"` references stripped from per-arch kernel
  TUs (both headers were deleted in `682eab4` but the includes
  survived). `BatchedScanCtx` struct (used by 9 SG2 forward TUs)
  recovered into `csrc/common/types.h`. `ArchTier` /
  `StatePrecision` / `ExpertPrecision` enums recovered into a new
  `csrc/common/arch_tier.h` shim with constexpr `kArchTier` picked
  per-TU via `SG_ARCH_<X>` preprocessor switches; legacy `ArchTier::X`
  call sites in distributed_pipeline TUs continue to work without
  body edits. The `csrc/kernels/hip/gfx942/supergrok2_gfx942.hip.cpp`
  delegating overlay was deleted (its remaining bodies were trivial
  passthrough wrappers to symbols that no longer exist).
- **Cross-arch agreement test bodies**: filled in for every
  optimizer including a Muon 2D-param harness (`tests/test_cross_arch_agreement.py`).

Items that **remain deferred** (engineering work — see §25 for detail):

- Real per-arch kernel divergence beyond Muon (the 8 × 17 = 136
  wrapped baselines are still byte-identical modulo namespace).
- Hopper FP8 fast-path inlining for SG2 (`launch_mamba3_peer_batched_step`
  on sm_90) and Blackwell warp-specialized scan activation
  beyond the renamed `_warp_specialized` declarators.
- Fused softplus epilogue in CUTLASS for SG2 `dt_proj`.
- Real autotune output (placeholders remain in `tuned_configs.h`).
- CI updates for the eight-row arch matrix.

### Migration commit series

For navigation, the structural-refactor commits:

- `895c32e` — `REFACTOR_PLAN.md`
- `866d42b` — create new directory tree
- `864499c` — `git mv` 27 existing arch-specific files into new tree
- `e180964` — rename pre-existing arch-tuned files to `*_overlay`
- `5d8085a` — wrap 17 generic kernels into 68 per-arch baselines
- `4f71d02` — bindings layer + worked-example GrokAdamW dispatcher
- `9eb21d4` — per-optimizer bindings + pybind11 module aggregator
- `c8022cc` — refactor dispatch.py / loader / __init__.py
- `e9d22cf` — move Pallas TPU kernels to `csrc/kernels/tpu/`
- `6a50d3e` — autotune scaffolding + `tuned_configs.h`
- `f422ada` — rewrite `setup.py` for new tree
- `3683945` — `tests/test_cross_arch_agreement.py`
- `6307566` — refactor `tests/test_amd_hip.py` and `tests/test_all_arches.py`
- `8c2280d` — delete unsupported arches (sm_75/86/89, cdna2/cdna4)
- `8725e3a` — delete `csrc/cuda/generic/` (17 files) + `csrc/cuda/generated/` (30 files)
- `95b77e0` — delete `csrc/cpu/` tree
- `104f3ff` — delete `grokking_optimizers/jit/` and `codegen/`
- `682eab4` — delete `csrc/common/{ops.h,ops.cpp,dispatch.h}`

§10 arch matrix expansion commits:

- `dd1e0a6` — `REFACTOR_PLAN.md §10` (arch expansion plan)
- `8da3d80` — create directory tree for sm_89, sm_103, sm_120, gfx950
- `bf157b4` — port 17 sm_90 baselines → sm_89 (Ada Lovelace)
- `e2545a8` — port 17 sm_100 baselines → sm_103 (Blackwell Ultra)
- `50925ae` — port 17 sm_100 baselines → sm_120 (consumer Blackwell)
- `98f8190` — port 17 gfx942 baselines → gfx950 (CDNA4)
- `02348cc` — recover CDNA4 FP4/FP6/2:4 sparsity overlay (2491 lines) into gfx950
- `813ffdb` — drop now-redundant `.gitkeep` placeholders
- `5b4218b` — extend bindings layer (`bindings.h`, `_dispatch_macro.h`, `dispatch.cpp`, all per-optimizer dispatchers) to 8 arches
- `0fe9cc4` — extend `tuned_configs.h` (`ArchId` enum + table widths 4→8) and `setup.py` (`-gencode` for sm_89/103/120, `--offload-arch=gfx950`)
- `40954ae` — extend `dispatch.py` (SUPPORTED_ARCHES, detection, label table) and `autotune/` (NVFP4/FP4/FP6 grids, sm_89/103a/120a CUTLASS targets)
- `58b9e54` — extend tests (`test_cross_arch_agreement.py`, `test_all_arches.py`, `test_amd_hip.py` covering both gfx942 + gfx950)

---

## Contents (pre-refactor reference; sections below describe the OLD layout — see §0, §21, §22, §23 for current state)

1. Repo layout
2. Project state
3. Optimizers
4. Python infrastructure
5. csrc/common — shared headers
6. csrc/cuda/generic — generic kernels
7. csrc/cuda/sm_80 — Ampere
8. csrc/cuda/sm_90 — Hopper
9. csrc/cuda/sm_100 — Blackwell
10. csrc/hip — AMD ROCm
11. csrc/quantization — quantization kernels
12. Algorithms
13. JAX/TPU
14. Tests
15. Benchmarks
16. Codegen
17. Build
18. Recent commits
19. Known gaps
20. Quick reference

---

## 1. Repo layout

(Post-refactor. The pre-refactor tree had `csrc/cuda/generic/`,
`csrc/cuda/generated/`, `csrc/cpu/`, `csrc/hip/cdna2/3/4/`, `codegen/`,
and `grokking_optimizers/jit/`. All deleted in the structural refactor —
see §0 migration commit series.)

- `grokking_optimizers/` — Python package, eleven optimizers plus infra
- `supergrok2_jax_tpu/` — JAX/TPU port of the suite (Pallas kernels live in `csrc/kernels/tpu/`)
- `csrc/common/` — shared headers (`platform.h`, `types.h`, `ptx_intrinsics.cuh`, `utils.cuh`, `quantization.h`) plus `tuned_configs.h` (autotune output)
- `csrc/bindings/` — per-optimizer dispatchers (`grokadamw.cpp`, `lion.cpp`, …) + arch detection (`dispatch.cpp`) + pybind11 module aggregator (`module.cpp`)
- `csrc/kernels/cuda/sm_80/` `sm_89/` `sm_90/` `sm_100/` `sm_103/` `sm_120/` — NVIDIA per-arch specialized kernels (six arches)
- `csrc/kernels/hip/gfx942/` `gfx950/` — AMD per-arch specialized kernels (two arches)
- `csrc/kernels/tpu/v5p/` `v6e/` — TPU Pallas kernels per version (re-export tile-128 / tile-256 from shared `_pallas_kernels.py`)
- `csrc/kernels/cpu/` — CPU implementations with AVX-512 / NEON SIMD (testing only, not a runtime fallback)
- `csrc/quantization/` — quantization kernels (FP8, INT8, INT4, MXFP4)
- `autotune/` — offline tuning: `tune.py`, `grids.py`, `runner.py`, `cutlass_profile.py`. Replaces the deleted `grokking_optimizers/jit/`.
- `tests/` — nine test files (the refactor added `test_cross_arch_agreement.py` and renamed `test_all_tiers.py` → `test_all_arches.py`)
- `benchmarks/` — three benchmark scripts
- `setup.py` — build entry, detects backend, supports the eight GPU arches
- `README.md` — user docs
- `REFRESH.md` — this file
- `REFACTOR_PLAN.md` — refactor design (steps 1-9 + §10 arch expansion)
- `ANALYSIS.md` — internal review with bug findings and optimization opportunities

## 2. Project state

- Branch: `claude/custom-optimizer-analysis-HFYhg`
- Working tree: clean
- Size: 180,845 LOC across 267 tracked files (see §0 Code statistics for the per-language breakdown)
- Package version: `3.0.0` (breaking refactor — bumped in commit `c8022cc`)
- Backends supported: NVIDIA `sm_80, sm_89, sm_90, sm_100, sm_103, sm_120`; AMD `gfx942, gfx950`; TPU `v5p, v6e`. CPU build for testing only.
- Permanently unsupported: V100, T4, RTX 20-series, MI100 (gfx908), MI200 (gfx90a), AMD RDNA, TPU v3/v4/v5e.
- Status: structural refactor complete; per-arch hand-tuning + CUTLASS migration + autotune execution deferred to hardware-validated sessions (see §0 "What is NOT yet done").
- Recent focus: arch matrix expansion (6 NVIDIA + 2 AMD = 8 GPU arches, up from 4); per-arch wrapped baselines; bindings layer split; autotune scaffolding.
- Architecture: SuperGrok v2 design settled long ago; the refactor changes how kernels are organized (single per-arch source-of-truth instead of generic + overlay), not what they compute.
- Last 5 commits, newest first:
  - `8a100ac` — REFRESH.md §0 with 8-arch supported set + expansion commits
  - `58b9e54` — extend tests for sm_89/103/120 + gfx950
  - `40954ae` — extend dispatch.py + autotune for new arches
  - `0fe9cc4` — extend tuned_configs.h + setup.py for 8-arch matrix
  - `5b4218b` — extend bindings layer to recognize 4 new arches

## 3. Optimizers

Eleven total. Each entry: purpose, state per param, hyperparameters with defaults, fused kernel name, Python fallback availability.

### 3.1 SuperGrok v2 (`supergrok2.py`)

- Purpose: flagship. Mamba-3 + 4-head PEER + per-element GRU + 144-expert pool, per-element learned gradient correction, on top of Adam with SAM and bilevel meta-learning.
- State per param: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`, `gru_states[N, gru_hidden]`, `mamba_fwd_states[d_inner, d_state]`, `mamba_bwd_states[d_inner, d_state]`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1.0, alpha_init=0.98, lamb=2.0, kappa=0.1, gradient_clipping=1.0, d_model=8, d_state=16, mamba_expand=2, num_peer_heads=4, num_experts=144, expert_hidden=16, gru_hidden=4, meta_rescale=0.1, recycle_interval=100, recycle_threshold=0.001, sam_rho=0.05, projection_precision='auto', state_precision='fp32' or 'config3'
- Fused kernel: `_ops.supergrok2_prepare_and_batched_step`
- Bilevel kernel: `_ops.supergrok2_bilevel_fwd_save_batched` + `_ops.supergrok2_bilevel_backward`
- Python fallback: yes, full
- Distributed: meta-grad allreduce, expert count allreduce, mamba state broadcast from rank 0
- FSDP: meta-net excluded from sharding via `exclude_meta_net_from_fsdp`
- Compilable: `CompiledSuperGrok2` wrapper for CUDA graph capture

### 3.2 SuperGrok v1.5 (`supergrok15.py`)

- Purpose: simpler v2. Replaces Mamba+PEER+GRU with a 2-input 2-layer MLP.
- State per param: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=0.98, lamb=2.0, kappa=0.1, sam_rho=0.05, hidden_dim=32 (also 16/64/128 specialized)
- Fused kernel: `supergrok15_fused_step`
- Python fallback: no
- Special: register-resident smart_grad in fused full-step kernel

### 3.3 SuperGrok v1.1 (`supergrok11.py`)

- Purpose: v1.5 with cosine-similarity gating instead of sigmoid-on-accuracy.
- State per param: same as v1.5
- Hyperparameters: same as v1.5, plus gate_temperature=5.0, meta_update_freq=5
- Fused kernel: `supergrok11_fused_step`
- Reduction kernel: `cosine_gate_reduce_kernel` — fused dot/norm/norm reduction
- Python fallback: no

### 3.4 GrokAdamW (`grokadamw.py`)

- Purpose: AdamW with EMA gradient filter and persistent-direction amplification.
- State per param: `exp_avg`, `exp_avg_sq`, `ema`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=0.98, lamb=5.0, grad_clip=1.0
- Fused kernel: `grokadamw_fused_step`
- Quantized variant: `_q3` kernel — INT8 per-block exp_avg + BF16 stochastic-rounded exp_avg_sq + ema
- Python fallback: no (CPU build has C++ implementation)

### 3.5 NeuralGrok (`neuralgrok.py`)

- Purpose: AdamW with learned MLP amplifier on |grad|.
- State per param: `exp_avg`, `exp_avg_sq`
- Amplifier: 2- or 3-layer MLP, input |grad|, output multiplicative scale
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=10.0, beta=4.0, num_layers=3, hidden_dim=128, inner_steps=1
- Fused kernel: `neuralgrok_fused_step`
- Python fallback: no

### 3.6 Prodigy (`prodigy.py`)

- Purpose: self-tuning Adam. Estimates `d_lr` from cumulative parameter-space distance. Set lr=1.0 and let it auto-tune.
- State per param: `exp_avg`, `exp_avg_sq`, `s`, `param_init`
- Hyperparameters: lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1.0
- Fused kernel: `prodigy_fused_step`
- Reduction kernel: `prodigy_dlr_reduce_kernel` — computes new `d_lr` via global reduction
- Python fallback: no

### 3.7 Grokfast (`grokfast.py`)

- Purpose: simplest grokking-aware AdamW. EMA + amplification.
- State per param: `ema`, `exp_avg`, `exp_avg_sq`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, grokfast_alpha=0.98, grokfast_lamb=2.0
- Fused kernel: `grokfast_fused_ema_adam_step`
- Python fallback: no

### 3.8 Lion (`lion.py`)

- Purpose: sign-based Adam alternative (EvoLved Sign Momentum).
- State per param: `exp_avg` (momentum buffer)
- Hyperparameters: lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0
- Fused kernel: `lion_fused_step`
- Multi-tensor variant: yes — fuses many small params into one launch
- Python fallback: yes, in CPU C++

### 3.9 LookSAM (`looksam.py`)

- Purpose: AdamW with periodic SAM (every k steps) instead of every-step SAM.
- State per param: `exp_avg`, `exp_avg_sq`, `sam_direction`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, rho=0.05, k=5, alpha=0.7
- Fused kernel: `fused_adamw_simple_step` (regular), manual SAM step
- Python fallback: yes, in CPU C++

### 3.10 Muon (`muon.py`)

- Purpose: dual optimizer. Newton-Schulz orthogonalization for 2D weights, AdamW for 1D.
- State (2D): `momentum_buffer`
- State (1D): `exp_avg`, `exp_avg_sq`
- Hyperparameters (2D): lr=0.02, momentum=0.95, weight_decay=1.0, ns_steps=5
- Hyperparameters (1D): adamw_lr=1e-3, adamw_betas=(0.9, 0.98), adamw_eps=1e-8
- Fused kernels: `muon_fused_step` (2D), `fused_adamw_simple_step` (1D)
- Python fallback: yes, in CPU C++

### 3.11 Mamba3PEERMetaNet (`mamba3_peer_metanet.py`)

- Purpose: meta-net used internally by SuperGrok v2; not a standalone optimizer.
- Submodules: `Mamba3ScanBlock`, `MiniGRU`, PEER router (in same file), expert MLP pool
- Trained by SuperGrok v2's bilevel update
- Has full pure-PyTorch CPU fallback path

## 4. Python infrastructure

### `dispatch.py`
- Detects backend at runtime, no GPU import required.
- `get_gpu_vendor()` → 'nvidia' | 'amd' | 'none'
- `get_gpu_arch()` → SM number (NVIDIA) or CDNA arch (AMD)
- `get_backend()` → 'cuda' | 'hip' | 'cpu'
- `get_warp_size()` → 32 (NVIDIA) or 64 (AMD CDNA)
- `get_arch_tier()` → 'blackwell'|'hopper'|'ampere'|'generic'
- `get_amd_tier()` → 'cdna4'|'cdna3'|'cdna2'|'generic'
- `supports_bf16/fp8/tf32/tma/block_clusters/matrix_cores/nvfp4` predicates
- Env override: `FORCE_ARCH=N`

### `quantization.py`
- `PrecisionConfig` class with three knobs:
  - projection precision: `'fp32'|'tf32'|'bf16'|'fp8'|'mxfp4'|'nvfp4'|'auto'`
  - expert precision: `'fp32'|'int8'|'int4'|'auto'`
  - state precision: always FP32
- `convert_projection_weights(w)` → (quantized, scale)
- `convert_expert_weights(w1, b1, w2, b2)` → dict with mode + tensors
- Auto chain: nvfp4 → mxfp4 → fp8 → bf16 → fp32
- Optional dynamic-precision mode that lowers precision as training stabilizes

### `cuda_graph_optimizer.py`
- `CUDAGraphOptimizer(opt, warmup_steps=3)` — wraps any optimizer
- `CompiledSuperGrok2` — SuperGrok v2 specialization
- Records first non-warmup step as graph, replays after
- Auto-invalidates when kwargs passed to step()
- `invalidate()` method for manual reset
- ~2-3× speedup small models, ~1.5× large

### `distributed.py`
- `setup_distributed(backend='nccl')` — torchrun-style init
- `cleanup_distributed()`, `get_rank()`, `get_world_size()`, `is_main_process()`
- `broadcast_optimizer_state(opt, src=0)` — align ranks
- `wrap_model_ddp(model)` — DDP wrapper
- SuperGrok v2 private helpers:
  - `_is_distributed()`
  - `_allreduce_meta_grads()` — sum + divide by world size
  - `_allreduce_expert_counts()` — for recycling consistency
  - `_sync_mamba_states()` — broadcast from rank 0
  - `_gather_full_grad_fsdp()` — context manager for FSDP
  - `exclude_meta_net_from_fsdp(meta_net)` — keep meta-net replicated

### `jit/` directory
- Optional runtime kernel specialization, cached in `~/.cache/supergrok2/`
- `specializer.py` — base class + `ModelConfig`
- `cuda_specializer.py`, `hip_specializer.py`, `tpu_specializer.py`, `cpu_specializer.py`
- `smem_layout.py` — shared memory layout optimization
- `block_size_optimizer.py` — tile size selection
- `gcn_scheduler.py` — AMD GCN wavefront scheduling
- `ptx_scheduler.py` — NVIDIA PTX instruction scheduling
- Falls back to pre-compiled `_ops` if anything fails

### `__init__.py`
- Exports all eleven optimizers
- Meta-net classes: Mamba3PEERMetaNet, Mamba3ScanBlock, MiniGRU, SharpnessMetaNet
- Wrappers: CompiledSuperGrok2, CUDAGraphOptimizer, OverlappedOptimizer, PipelinedOptimizer, GradientHookOptimizer, AsyncSuperGrok2, MoEAwareSuperGrok2
- Distributed helpers, dispatch helpers, PrecisionConfig
- Flags: `_HAS_OPS`, `_HAS_CUDA`, `_HAS_CPU_OPS`

## 5. csrc/common — shared headers

### `platform.h`
- Single-source CUDA/HIP abstraction.
- Sets `GROK_CUDA=1` or `GROK_HIP=1`.
- Warp size: 32 (CUDA), 64 on CDNA via `__AMDGCN_WAVEFRONT_SIZE__`.
- Macros remap intrinsics:
  - `SHFL_DOWN` → `__shfl_down_sync` (CUDA, with masks) / `__shfl_down` (HIP)
  - `FAST_SINCOSF` → `__sincosf` (CUDA) / `sincosf` (HIP)
  - `LDG` → `__ldg` (CUDA, read-only cache) / `*ptr` (HIP)
  - `FULL_WARP_MASK` → `0xFFFFFFFF` (CUDA) / `0` (HIP)
- Stream type alias: `GpuStream_t`
- Error checking: `gpuGetLastError`, `gpuDeviceSynchronize`, `gpuGetDeviceProperties`
- CUB/hipcub namespace alias for portable CUB calls
- Non-temporal I/O (bypass L2):
  - CUDA sm_80+: inline PTX `ld.global.nc` and `st.global.wt`
  - HIP: `__builtin_nontemporal_load/store`
  - float4 vectorized variants
- AMD occupancy attributes: `GROK_WAVES_PER_EU(min, max)`, `GROK_FLAT_WORK_GROUP_SIZE(min, max)` (no-op on CUDA)

### `types.h`
- Compile-time constants:
  - `MAX_D_STATE = 32`
  - `MAX_D_INNER = 32`
  - `MAX_D_MODEL = 16`
  - `MAX_GRU_HIDDEN = 8`
  - `MAX_EXPERT_HIDDEN = 16`
  - `MAX_TOPK = 4`
  - `PSCAN_BLOCK = 512` (Blelloch threads/block)
  - `PSCAN_THRESHOLD = 256` (use sequential below this N)
  - `GEMM_PRECOMPUTE_THRESHOLD = 1024` (use cuBLAS GEMM above this N)
- `Affine2x2` struct: 4 floats matrix + 2 floats bias = 6 floats
- `affine_combine()` — portable C++ composition

### `ptx_intrinsics.cuh`
- `affine_combine_ptx(left, right)` — 12 FMAs in 3 waves, ~10 cycles
- `softplus_ptx(x)` — `ex2.approx` + `lg2.approx` + branchless saturation, ~2 cycles
- `fast_exp_ptx(x)` — `ex2.approx`, 1 cycle
- `stochastic_round_ptx(x, rand_bits)` — `cvt.rmi` + `selp`, branchless
- `gru_gates_ptx(...)` — interleaved sigmoid pair using `ex2.approx` + `rcp.approx`
- HIP fallbacks use standard math library

### `utils.cuh`
- `warp_reduce_sum(val, d_inner, tid)` — warp-shuffle reduction adapting to warp size
- `hash_prng(step, idx)` — Philox-like deterministic PRNG, no state buffer
- BF16 and INT8 stochastic rounding helpers
- `fast_rsqrt_nr(x)` — `rsqrt.approx.f32` + Newton-Raphson refinement
- `ptx_fma`, `ptx_exp2`, `ptx_log2`, `ptx_expf`, `ptx_tanhf`, `ptx_sigmoidf`
- `ptx_expert_mlp_forward<H>` — templated, fully unrollable
- `ptx_int8_stochastic_round` — uses `prmt.b32` byte permutation

### `ops.h` / `ops.cpp`
- C++ binding layer; ~79 kernel launchers
- `ops.h` declares all launchers
- `ops.cpp` is high-level glue per optimizer step
- Decides parallel vs sequential scan, GEMM vs custom precompute
- CPU fallback via PyTorch ATen ops (correct, slow), guarded by `WITH_CUDA`/`WITH_HIP`

### `quantization.h`
- `PrecisionMode` enum: FP32, TF32, BF16, FP8_E4M3, INT8_SYM, INT4_GPTQ, MXFP4
- Device-side dequant helpers:
  - `dequant_int8(q, scale)` — symmetric, per-tensor
  - `dequant_int4(packed, which, scale, zero)` — group_size=32, asymmetric
  - `dequant_mxfp4(packed, which, shared_exp)` — block_size=32 shared exponent
  - `fp4_e2m1_to_float` — lookup table {0, 0.5, 1, 1.5, 2, 3, 4, 6}

### `dispatch.h`
- C++ side of arch detection
- NVIDIA tiers: GENERIC (sm_70/75), AMPERE (sm_80–89), HOPPER (sm_90), BLACKWELL (sm_100)
- AMD tiers: GENERIC (gfx908/90a/942), CDNA4 (gfx950)
- `get_sm_arch()` via `cudaGetDeviceProperties`, respects `FORCE_ARCH` env var
- `StatePrecision` enum: FP32, CONFIG4 (INT8 state), FP6 (CDNA4)
- `ExpertPrecision` enum: FP32, INT8, INT4, MXFP4, FP4 (CDNA4)

## 6. csrc/cuda/generic — generic kernels

### `supergrok2_mamba_peer_kernels.cu` (forward path)

- **`input_proj_sort_kernel`** — projects `[grad, sharpness]` to `[N, d_model]`, emits `|grad|` as sort key plus identity index permutation. Clips NaN/Inf to zero. 256 threads/block, one element per thread, `#pragma unroll 4` on d_model loop.
- **`mamba3_scan_kernel`** — sequential selective scan, used when N < 256. 16 threads per param (one per d_inner). Per timestep: x-branch and z-gate via shared in_proj_W, dt via softplus_ptx, B and C projections via shared x_branch, trapezoidal state recurrence with paired RoPE rotation via FAST_SINCOSF, gated output `y * silu(z) + D * x`. Reverse flag drives backward bidirectional pass.
- **`mamba3_parallel_precompute_kernel`** — precomputes `pre_x_val`, `pre_z_val`, `pre_dt_val`, `pre_B_val`, `pre_C_val` for all timesteps in parallel, no inter-timestep dependencies. Used when 256 ≤ N < 1024. 256 threads/block, one timestep per thread.
- **`mamba3_parallel_scan_kernel`** — Blelloch parallel prefix scan over Affine2x2 transforms. PSCAN_BLOCK=512 threads/block. Three phases:
  1. Each thread sequentially scans a chunk to produce one Affine2x2 summary
  2. Up-sweep + down-sweep on summaries in shared memory (12KB for 6 floats × 512 threads)
  3. Each thread re-scans its chunk applying its prefix, accumulates into output
  Skips `__syncthreads()` for stride < WARP_SIZE.
- **`fused_elem_step_kernel`** — the per-element step. One thread per element. Sequence: load fwd/bwd Mamba scan outputs (float4 vectorized for d_inner ≤ 4), project to d_model contexts, non-temporal load of GRU state, GRU update with `gru_gates_ptx` for sigmoid pair, non-temporal store of new GRU state, build query per PEER head, score against 12 product keys per half via `LDG` cached loads, hard-route to one expert per head, evaluate 2-layer expert MLP from shared memory, accumulate weighted output, atomic-add expert counter, smart_grad = grad + rescale × meta_out, slow EMA update, effective grad = smart + λ × mu, Adam moment updates, parameter update with decoupled weight decay. Shared memory ~8.5 KB per block.

### `supergrok2_mamba_peer_backward_kernels.cu` (backward path)

- **`bilevel_precompute_kernel`** — same as forward parallel precompute, used for backward replay
- **`softplus_bias_kernel`** — applies `softplus(x + bias)` element-wise, used after cuBLAS dt projection
- **`bilevel_precompute_gemm`** — wraps `torch::mm_out` calls (cuBLAS path) when N ≥ 1024. Splits in_proj into x and z halves; runs in_proj_x, in_proj_z, dt_proj, B_proj, C_proj as 5 GEMMs. Calls `softplus_bias_kernel` after dt.
- **`mamba3_parallel_scan_fwd_save_kernel`** — same as forward parallel scan but saves selected hidden states to `saved_states` for backward. Checkpoint policy: save every state if `checkpoint_interval ≤ 1`, else save every Cth state.
- **`mamba3_scan_fwd_save_kernel`** — sequential variant for small N
- **`mamba3_scan_backward_kernel`** — backward scan. Walks timesteps in reverse. Per step: backprop through SiLU gating, through C projection (two-pass with warp reductions to a `d_C_vals_buf`, then backward GEMM for d_C_proj_W), through trapezoidal-discretized affine recurrence and RoPE, through dt projection, through B projection, through input projection. Block-local shared-memory accumulators for weight gradients, atomicAdd flush at block end.
- **`input_proj_backward_kernel`** — outer-product accumulation of `d_x` against `[grad, sharpness]` into proj_W and proj_b. Block-local accumulator + atomic flush.
- **`gru_backward_kernel`** — gradients w.r.t. Wz, Wr, Wh and bz, br, bh and gru_input. Unrolls gate logic carefully, accumulates via shared memory.
- **`expert_peer_backward_kernel`** — two-pass for softmax-backward coupling:
  - Pass 1: accumulate softmax dot products
  - Pass 2: full softmax-backward + gradient accumulation into expert weights, product keys, query weights
- **`out_proj_backward_kernel`** — outer-product accumulation for d_out_proj_W

### `supergrok15_kernels.cu`

- **`fused_mu_metanet_kernel`** — updates mu EMA, evaluates 2-input MLP per element with weights in shared memory, fast-GELU activation, output is smart_grad = grad + rescale × mlp_out
- **`fused_adam_decay_kernel`** — final blend with mu, Adam moments update with `fast_rsqrt_nr`, progressive decoupled weight decay. Non-temporal stores. Float4 fast path.
- **`sam_perturb_kernel`** — `param[i] += rho_over_norm × grad[i]`. Float4 fast path.
- **`sharpness_restore_kernel`** — `sharpness[i] = |sam_grad - normal_grad|`, restore param from backup
- **`fused_supergrok15_full_step_kernel`** — fuses mu_metanet + adam_decay. Smart_grad is register-resident, never hits global memory. ~50% bandwidth reduction.
- Templated specializations for H=16/32/64/128 with full unrolling. Runtime variant uses `#pragma unroll 4`.

### `supergrok11_kernels.cu`

- **`launch_sg11_mu_metanet`** — same as v1.5 mu_metanet but with cosine gating
- **`compute_cosine_gate`** — ATen helper that computes cos_sim between smart_grad and mu, passes through temperature sigmoid
- **`cosine_gate_reduce_kernel`** — fused 3-quantity reduction (dot, |sg|², |mu|²). Warp shuffle within warp, atomicAdd per warp into globals.
- **`compute_cosine_gate_fused`** — ATen wrapper around the reduce kernel
- **`launch_sg11_adam_decay`** — same shape as v1.5, takes lamb_eff = ramp × cos_gate × base_lamb
- **`fused_sg11_full_step_kernel`** — fused full step with cosine gate input

### `grokadamw_kernels.cu`

- **`fused_grokadamw_step_kernel`** — EMA update, gradient amplification, Adam moments, decoupled WD, parameter update. Non-temporal I/O for state. Float4 fast path.
- **`fused_grokadamw_step_q3_kernel`** — quantized variant. INT8 per-block exp_avg with FP32 per-block scales (block_size=8). BF16 stochastic-rounded exp_avg_sq and ema using `hash_prng`. ~50% optimizer state memory reduction.

### `neuralgrok_kernels.cu`

- **`fused_neuralgrok_amplifier_kernel`** — amplifier MLP per element. Linear(1→H), ReLU, Linear(H→1). Weights cooperatively loaded into shared memory. `amplified_grad = grad × (alpha × mlp_out + beta)`.
- **`fused_neuralgrok_full_step_kernel`** — fuses amplifier + Adam, amplified_grad register-resident
- Templated H=16/32/64/128

### `prodigy_kernels.cu`

- **`prodigy_dlr_reduce_kernel`** — global reduction, computes numerator (Σ grad × distance) and denominator (Σ s) via warp shuffles and per-warp atomicAdd. New `d_lr = sqrt(num / denom + eps)`.
- **`fused_prodigy_step_kernel`** — moment updates scaled by `d_lr`, s update, parameter update with `lr × d_lr × wd`.

### `grokfast_kernels.cu`

- **`fused_grokfast_ema_kernel`** — standalone EMA update + amplification (used in non-fused paths)
- **`fused_grokfast_adam_kernel`** — fused full step. Amplified grad register-resident.

### `lion_kernels.cu`

- **`fused_lion_step_kernel`** — interpolated direction `β1 × m + (1-β1) × grad`, sign extraction via `copysignf`, parameter update with decoupled WD, momentum EMA update with β2. Non-temporal I/O. Float4 fast path.

### `looksam_kernels.cu`

- **`looksam_norm_reduce_kernel`** — fused two-norm reduction: `|sam_grad - normal_grad|²` and `|grad|²` in one pass. Warp shuffles + per-warp atomic.
- **`looksam_direction_kernel`** — `v_dir[i] = (sam_grad - normal_grad) × inv_norm`
- **`looksam_direction_adjust_fused_kernel`** — fused direction + gradient adjustment. v_dir register-resident.

### `muon_kernels.cu`

- **`muon_momentum_normalize_kernel`** — momentum EMA update + division by Frobenius norm
- **`muon_ns_combine_kernel`** — Newton-Schulz inner: `out = a × X + b × AX + c × AAX` with hand-tuned coefficients (a=3, b=-3, c=1 default). AX and AAX are computed by separate cuBLAS matmul calls outside this kernel.
- **`muon_ns_combine_update_fused_kernel`** — final NS iteration combine + parameter update fused. Orthogonalized direction register-resident.

### `moe_deep_kernels.cu`

- **`moe_dynamic_expert_load_kernel`** — load only active experts' weights into shared memory based on gate logits
- **`moe_dynamic_expert_fwd_kernel`** — forward through dynamically loaded subset
- **`moe_dynamic_expert_bwd_kernel`** — backward through dynamic loading
- **`moe_filter_active_params_kernel`** — compact parameter index list to active experts only
- **`moe_scan_compacted_kernel`** — Mamba scan on compacted subset
- **`moe_scatter_results_kernel`** — scatter results back to full positions
- **`moe_count_expert_activations_kernel`** — atomicAdd per expert
- **`moe_compute_load_balance_loss_kernel`** — auxiliary loss for uniform expert utilization
- **`moe_apply_frequency_scaling_kernel`** — per-expert lr scaling by activation frequency

### `multi_tensor_optimizer_kernels.cu`

- Single kernel launch for many small parameter tensors. 2D grid: blockIdx.y selects param, threads in row iterate via grid-stride.
- Supports: GrokAdamW, Lion, Grokfast EMA, Prodigy step
- Pointer-packing: param pointers packed once on host, transferred to device, indexed by blockIdx.y
- Saves 100-500 ms/step on transformers with many small params

### `multi_tensor_prepare.cu`

- Fuses per-step preparation into one kernel: gradient norm, clipping, NaN/Inf replace, bias correction, per-param scalars
- One block per parameter, parallel reduction within block via warp shuffles + shared memory

### `distributed_scan_kernels.cu`

- **`mamba3_scan_local_with_summary_kernel`** — each GPU runs local Blelloch scan on its chunk, produces one Affine2x2 summary
- **`scan_summary_prefix_kernel`** — gathers summaries on rank 0, computes per-GPU prefix corrections via small prefix scan
- **`mamba3_apply_scan_prefix_kernel`** — each GPU applies its prefix to local output
- Communication: ~6 floats per GPU per scan
- Backward variants for gradient computation

## 7. csrc/cuda/sm_80 — Ampere

Headline optimization: `cp.async` double-buffered prefetch. Overlaps multi-hundred-cycle global memory latency with scan compute, hides ~50% of memory stalls.

- **`supergrok2_scan_sm80.cu`** — sequential scan with cp.async prefetch
  - **`mamba3_scan_batched_cpasync_kernel`** — batched sequential scan, double-buffered shared memory
  - **`mamba3_scan_combined_cpasync_kernel`** — forward + backward scan fused
- **`supergrok2_backward_sm80.cu`** — backward scan with cp.async
- **`supergrok2_fused_elem_sm80.cu`**
  - **`fused_elem_step_cpasync_kernel`** — per-element step with cp.async-prefetched expert weights
- **`metanet_optimizers_sm80.cu`** — Ampere-tuned optimizer kernel templates
- **`metanet_cpasync_variants_sm80.cu`** — cp.async variants for the meta-net optimizers
- **`muon_sm80.cu`** — Muon for Ampere with TF32 GEMMs via cuBLAS

Precision: TF32 for projection matmuls (transparent via cuBLAS). 192KB shared memory per SM.

## 8. csrc/cuda/sm_90 — Hopper

Headline optimization: FP8 E4M3 projection precompute via cuBLAS for N ≥ 4096. ~2× speedup vs BF16 (905 vs 452 TFLOPS on H100). Device-side absmax computation avoids host-device sync.

- **`supergrok2_scan_sm90.cu`** — FP8 precompute integrated with scan
  - **`hopper_fp8_gemm`** — cuBLAS GEMM with FP8 inputs and FP32 accumulation, scale = absmax / 448.0 (FP8 E4M3 max)
- **`supergrok2_backward_sm90.cu`** — backward with FP8 projection backward GEMMs
- **`supergrok2_warp_specialized_sm90.cu`** — uses Hopper distributed shared memory; producer/consumer warp specialization (one warp loads, another computes)
- **`metanet_optimizers_sm90.cu`** — Hopper-tuned optimizer kernels
- **`muon_sm90.cu`** — Muon for Hopper with FP8 GEMMs

Note: TMA is **not** used for the scan because per-timestep scattered reads (sort permutation) are poorly suited to TMA's bulk-copy descriptor model.

228KB shared memory. Thread block clusters supported.

## 9. csrc/cuda/sm_100 — Blackwell

Conservative tier. Most heavy features (TMEM, MMA.2SM, native NVFP4) deferred to Hopper FP8 fallback with documented delegation.

- **`supergrok2_sm100.cu`** — TMA bulk-copy kernels for expert weights
  - **`fused_elem_step_tma_kernel`** — per-element step with TMA-prefetched expert weights (single-thread initiation, hardware-managed transfer)
- **`supergrok2_precompute_sm100.cu`** — FP4 precompute scaffolding
- **`supergrok2_scan_sm100.cu`** — scan with TMA

Hardware features available:
- TMA (Tensor Memory Accelerator): hardware-managed asynchronous bulk copy via descriptors
- FP4 E2M1 native matrix multiply: `mfma_f32_32x32x8_fp4`, 8× elements per instruction
- TMEM: on-chip tensor memory (currently unused)

Fallback chain: Blackwell → Hopper FP8 → Ampere → Generic.

## 10. csrc/hip — AMD ROCm

Wavefront 64 throughout (CDNA). All kernels use `WARP_SIZE` from `platform.h` for portability.

### `cdna2/` (gfx90a, MI250)

- **`supergrok2_scan_cdna2.hip.cpp`** — baseline CDNA scan, MFMA `mfma_f32_16x16x4` for matrix ops
- 8MB L2, 220 CUs

### `cdna3/` (gfx942, MI300X)

- **`supergrok2_cdna3.hip.cpp`** — BF16 MFMA projection precompute
  - **`cdna3_precompute_bf16`** — runs in_proj_x, in_proj_z, dt_proj, B_proj, C_proj as BF16 matmuls via rocBLAS, output cast back to FP32. Dispatches to `MFMA_F32_32x32x8_BF16`, ~2× FP32 MFMA throughput.
- 304 CUs, 256MB L2 (meta-net always resident)

### `cdna4/` (gfx950, MI350X)

- **`cdna4_kernels.hip.cpp`** — native FP4/FP6/2:4 sparsity scaffolding

FP4 expert kernels:
- **`cdna4_fp4_expert_load`** — dequantize FP4 weights to FP32
- **`cdna4_fp4_expert_fwd`** — forward with FP4 expert weights via `mfma_f32_32x32x8_fp4`
- **`cdna4_fp4_expert_bwd`** — backward with gradient accumulation
- **`cdna4_fp4_quantize_experts`** — re-quantize expert gradients

FP6 state kernels (E3M2 native):
- **`cdna4_fp6_state_pack`** — FP32 → FP6 + per-block scale
- **`cdna4_fp6_state_unpack`** — FP6 → FP32
- **`cdna4_fp6_adam_step`** — Adam directly on FP6 state
- **`cdna4_fp6_lamb_step`** — LAMB on FP6 state

2:4 sparsity:
- **`cdna4_sparse24_select`** — select 2 non-zeros from each group of 4
- **`cdna4_sparse24_apply_mask`** — mask gradients to 2:4 pattern
- **`cdna4_sparse24_project`** — project momentum to sparse pattern
- **`cdna4_sparse24_densify`** — convert sparse → dense

Fused combos:
- **`cdna4_fp4_sparse24_fused_expert`** — expert MLP with FP4 weights + 2:4 sparsity
- **`cdna4_supergrok15_full_step`** — full v1.5 step on FP6 state + FP4 experts

512 CUs, 288MB L2.

### `README_HIP.md`

- Notes on wavefront-64 specific tuning, sync-skip behavior, MFMA dispatch

## 11. csrc/quantization — quantization kernels

### `quantization_kernels.cu`

FP8 E4M3:
- Two-phase kernel.
- Phase 1: warp-shuffle reduction within warp + atomicAdd to global accumulator → absmax
- Phase 2: rescale by `absmax / 448.0` (FP8 E4M3 max), quantize element by element with float4 vectorization, write uint8 + FP32 scale

INT8 symmetric:
- Same reduction pattern, limit 127, signed output
- `q = clamp(round(x / scale), -127, 127)`, `scale = absmax / 127`

INT4 GPTQ-style:
- Group-wise (group_size=32)
- Per-group min/max → scale and zero-point
- Asymmetric: `[min, max] → [0, 15]`
- Two values packed per byte (low/high nibble)

MXFP4:
- Per-block (block_size=32) shared 8-bit exponent
- 4-bit FP4 magnitudes per element + separate sign bit
- Magnitude lookup: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`
- Block scale: `2^(shared_exp - 127)`

Dequant helpers in `csrc/common/quantization.h` are inlined into optimizer kernels (e.g., expert weight dequant in shared memory).

## 12. Algorithms

### Affine2x2 Mamba encoding
- Mamba-3 recurrence: `h[t] = A_bar[t] × RoPE_rotate(h[t-1]) + B_bar[t] × x[t]`
- Split state into RoPE-paired even/odd dims → recurrence is an affine map on a 2D vector
- 2×2 matrix folds in: state-transition `A_bar`, RoPE rotation
- 2-vector bias holds: input contribution `B_bar × x`
- Composable: `(M_r, b_r) ∘ (M_l, b_l) = (M_r × M_l, M_r × b_l + b_r)`
- Associativity → eligible for parallel prefix scan
- Storage: 6 floats per element

### 12-FMA composition
- Affine2x2 composition: 8 FMAs for matrix product + 4 FMAs for matrix-vector + bias = 12 FMAs
- Inline PTX `affine_combine_ptx` arranges in 3 waves of 4 FMAs each:
  - Wave 0: 4 independent matrix products (no dependencies, fills both FMA pipelines)
  - Wave 1: 4 dependent accumulations + 2 bias starts
  - Wave 2: 2 final bias accumulations
- ~10 cycles total vs ~40+ for naive C++

### Blelloch parallel prefix scan
- Two-phase associative-operator scan
- Up-sweep: combine pairs at strides 1, 2, 4, …, leaves a root holding full composition
- Down-sweep: distribute exclusive prefixes back down with identity at root
- Work: O(N), depth: O(log N)
- Each leaf is one thread's chunk summary; each combine is one `affine_combine_ptx`
- Skip `__syncthreads()` for stride < WARP_SIZE (lanes implicitly synced)

### Bilevel checkpointing
- Naive backward stores all N timesteps × d_inner × d_state floats per param (e.g., 20 GB for N=100K)
- Checkpoint every C-th state (C = `checkpoint_interval`), recompute intermediates from nearest checkpoint
- Memory savings: ~(C-1)/C
- Backward compute: ~2× with C=256
- Default C=1 (full save), tunable via `bilevel_checkpoint_interval`

### Register-resident smart_grad
- Pattern shared by v1.5, v1.1, NeuralGrok, Grokfast, Muon final iter
- Smart_grad / amplified_grad / orthogonalized direction held in CUDA register
- Immediately consumed by Adam update in same kernel
- Avoids N writes + N reads of intermediate
- ~50% bandwidth reduction → 20-30% throughput gain

### Non-temporal I/O
- Optimizer state read-modify-write per step, not reused within step
- `stream_load`/`stream_store` use PTX `ld.global.nc` / `st.global.wt` on Ampere+
- HIP: `__builtin_nontemporal_load/store`
- Bypasses L2, leaves cache for hot data (weights, scan output)
- float and float4 variants

### PTX hot-path intrinsics
- `ex2.approx`: 1-cycle approximation of 2^x
- `lg2.approx`: 1-cycle approximation of log2(x)
- `rcp.approx`: 1-cycle reciprocal
- `softplus_ptx`: 2 cycles via ex2 + lg2 + selp
- `fast_exp_ptx`: 1 cycle via ex2.approx after multiply by log2(e)
- `gru_gates_ptx`: interleaved sigmoid pair (both pipelines busy)
- `stochastic_round_ptx`: branchless via cvt.rmi + selp
- 1-2 ULP error, acceptable for averaged optimizer state

### Warp-shuffle reductions
- Used in: cosine gate reduce, prodigy d_lr reduce, looksam norm reduce, expert backward
- Within warp: `__shfl_down_sync` butterfly at strides 16, 8, 4, 2, 1 (CUDA) or 32, 16, … (HIP-64)
- Cross-warp: lane 0 atomicAdds to global accumulator
- Avoids shared memory bottlenecks for small reductions

### Product-key PEER routing
- Score N elements against E experts in O(√E) work, not O(E)
- Split query into two halves, score against √E sub-keys per half
- Top-K each half → outer product → top-K(K²) candidates
- For E=144: 12-key sub-scoring × 2 = 24 dot products, vs 144 naive

### Cooperative shared-memory weight loading
- Used in: meta-net MLP weights, GRU weights, expert weights, prod keys
- All threads in block load disjoint slices of weights into shared memory
- One `__syncthreads()` then per-thread access at ~5 cycle latency vs 100+ for global

## 13. JAX/TPU

Functional rewrite of the suite. ~300 lines of core logic vs ~2000 lines of CUDA.

### Modules in `supergrok2_jax_tpu/`

- **`supergrok2_jax.py`** — main optimizer loop, `PerParamState`, `SuperGrok2State`, `OptimizerConfig`, `supergrok2_step`
- **`mamba3_peer_metanet_jax.py`** — meta-net architecture, `MetaNetWeights`, `init_meta_weights`, `meta_net_forward`
- **`scan.py`** — Mamba-3 scan via `jax.lax.associative_scan` with Affine2x2 combine operator (~40 lines)
- **`gru.py`** — `mini_gru` cell
- **`peer.py`** — `peer_expert_forward` (soft routing for bilevel), `peer_expert_forward_hard` (argmax for forward step)
- **`bilevel.py`** — `bilevel_step` using `jax.grad`, no custom backward needed
- **`pallas_kernels.py`** — optional Pallas kernels with try/except fallback
- **`sharding.py`** — `create_mesh`, `shard_params`, multi-host helpers
- **`simple_optimizers_jax.py`** — GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM
- **`metanet_optimizers_jax.py`** — SuperGrok v1.5, v1.1, NeuralGrok
- **`quantization_jax.py`** — INT8 symmetric quantization round-trip
- **`bridge.py`** — PyTorch ↔ JAX weight conversion + test vector export

### Key differences vs CUDA
- Functional: state is immutable pytrees, threaded through each step
- No custom backward: `jax.grad` autodiffs through `lax.associative_scan`
- Sharding declarative: `jax.sharding.NamedSharding`, `PartitionSpec`
- Multi-host: `jax.distributed.initialize`, `lax.pmean`, `lax.all_gather`

### Pallas kernels
- **`pallas_affine_scan`** — tiles scan into 128-element (v4/v5) or 256-element (v6e) blocks; intra-tile sequential scan + cross-tile prefix
- **`pallas_fused_gru_peer`** — fuses GRU + PEER routing + expert MLP, intermediates in VMEM
- **`vmem_persistent_expert_mlp`** — `eviction_policy="none"` keeps expert weights resident across tiles on v5p/v6e
- All wrapped in try/except, fall back to pure JAX if Pallas API changes

### TPU version detection
- `detect_tpu_version()` reads `jax.devices()[0].device_kind`
- Returns 'v4', 'v5e', 'v5p', 'v6e'
- Drives tile size and VMEM policy

### Feature gaps vs CUDA
- SAM perturbation not fully integrated (sharpness field exists, perturbation not wired)
- No explicit bilevel checkpointing (XLA's automatic rematerialization handles activations)
- INT8 only (no FP8/INT4/MXFP4/FP6)
- Expert load balancing minimal (counts tracked, no auxiliary loss)

## 14. Tests

Eight files, ~2,964 LOC. README still says six (stale). Total test points ~82.

> **Post-refactor update.** `tests/test_cross_arch_agreement.py` — the
> safety net for the all-specialized refactor — is no longer a stub.
> Every per-optimizer test (`test_<optimizer>_cross_arch`) runs the
> optimizer for N steps under each available `FORCE_ARCH` and asserts
> elementwise `allclose()` across arches with tolerance `1e-4`. Arches
> not compiled into the local extension are skipped. The Muon harness
> uses 2D params (64×64 matrix) since its Newton-Schulz orthogonalization
> requires `dim ≥ 2`. The whole class skips if no GPU is available.
> See §0.5 + §21 for what arches are expected to agree.

> The legacy CPU-fallback test `tests/test_cpu_fallback.py` had a
> `test_setup_cpu_sources` checker that referenced `csrc/cpu/generic/`
> paths deleted in `95b77e0`. It is now `test_setup_kernel_sources`,
> which checks `setup.py` walks the post-refactor `csrc/kernels/` tree.

### `test_supergrok2.py` (27 sections, 12A–12AA)
- 12A — import and build
- 12B — sequential vs parallel scan equivalence (N from 1 to 1024)
- 12C — forward step correctness (param changes, state populated)
- 12D — bilevel meta-learning correctness
- 12E — two-pass backward equivalence for scan weight gradients
- 12F — expert recycling stability (50 steps)
- 12G — gradient checkpointing equivalence (interval=1 vs 8)
- 12H — edge cases (N=0, N=1, zero grad, large grad, FP16 params)
- 12I — all 11 optimizers construct + step
- 12J — memory leak check (200 steps, <10% growth)
- 12K — two-pass GEMM backward reproducibility (max diff <1e-4)
- 12L — batched parallel scan single-launch with bitwise reproducibility
- 12M — dispatch detection (Python/C++ agreement)
- 12N — precision config auto-selection
- 12O — projection precision FP32 vs auto equivalence
- 12P — dispatch convergence (10 steps)
- 12Q — platform/vendor detection
- 12R — INT8 symmetric quantization round-trip
- 12S — INT4 GPTQ packing correctness
- 12T — MXFP4 quantization
- 12U — dynamic precision selection
- 12V — expert FP32 passthrough
- 12W — distributed helpers (DDP hooks, no-op without dist)
- 12X — CompiledSuperGrok2 wrapper (warmup/capture/replay)
- 12Y — `step_compiled` method
- 12Z — FSDP exclusion helper
- 12AA — distributed module imports

### `test_matrix.py`
- Cross-platform correctness matrix
- Runs 10 optimizers (excludes Mamba3PEERMetaNet which is internal)
- 5 steps per config, validates no NaN, measures step time
- Honors `FORCE_ARCH` env var

### `test_all_tiers.py`
- Validates dispatch correctness across NVIDIA tiers (generic, Ampere, Hopper)
- Sets `SUPERGROK_FORCE_ARCH` and runs `test_matrix.py` for each

### `test_cpu_fallback.py` (12 sections)
- `_HAS_*` flag sanity
- Python fallback module existence (13 variants)
- Strict `_ops` import in optimizer files
- Numerical correctness for Lion, GrokAdamW, Grokfast EMA, LookSAM, Muon Newton-Schulz
- CPU C++ extension completeness
- Importability of all optimizers
- Prodigy `d_lr` return value
- `setup.py` CPU sources listing

### `test_jax_matrix.py`
- Same matrix as PyTorch but for JAX optimizers
- 10 JAX optimizers, validates param changes and no NaN

### `test_amd_hip.py` (6 sections)
- `platform.h` adherence (no raw CUDA in generic)
- AMD tier detection via FORCE_ARCH
- PrecisionConfig auto for CDNA2/3 (BF16)
- `get_amd_label()` GPU labels
- GCN arch parsing (MI100/MI250/MI300X, three-digit codes)
- Wavefront-64 sync skip behavior

### `test_new_features.py` (7 sections)
- float4 vectorized GrokAdamW with alignment fallback
- OverlappedOptimizer distributed wrapper
- INT8 / PowerSGD gradient compression
- Pallas scan fallback
- Interleaved states layout
- Sparse gradient mask inference
- Partial CUDA graph optimizer

### `test_training_aware.py` (7 sections)
- Non-temporal stream_load/store correctness
- Q3 quantized states valid (no NaN, loss decreases)
- Q3 matches FP32 direction (cosine similarity > 0.99)
- Stochastic rounding unbiasedness
- No `.item()` calls in hot path
- PipelinedOptimizer equivalence
- training_benchmark script error-free

### Notable gap
- No explicit fused-CUDA-vs-Python-fallback bitwise/numerical agreement test (called out in `ANALYSIS.md`)

## 15. Benchmarks

### `benchmark_supergrok2.py`
- Models: tiny (h=32), small (h=64), medium (h=128), large (h=256), xlarge (h=512)
- Optimizers: all 11 + AdamW baseline
- Metrics: step time (ms), peak GPU memory (MB) by category, throughput (params/sec)
- Phases: 10 warmup + 100 timed
- Same init, same data (batch=32), same seed across optimizers
- Flags: `--optimizer`, `--model-size`, `--include-bilevel`, `--per-tier`, `--verbose`

### `autotune.py`
- Per-GPU profiling, results cached at `~/.cache/supergrok/autotune_{gpu_key}.json`
- GPU key: hash of device name + SM + total memory
- Profiles: scan block-size throughput, projection precision (FP32/TF32/BF16/FP8/MXFP4), memory
- Note: `PSCAN_BLOCK` is constexpr; changing requires rebuild
- Flags: `--dry-run`, `--force`, `--verbose`

### `training_benchmark.py`
- End-to-end grokking-style training run
- Reports loss/accuracy curves over time
- For comparing convergence speed and memory efficiency

### Fairness notes (from `ANALYSIS.md` §3)
- Same init, multi-GPU round-robin, multi-seed bands ✓
- SuperGrok optimizers do extra per-step work (meta-net forward, SAM, bilevel) — wall-clock not directly comparable
- SuperGrok bilevel uses validation data → information advantage vs other optimizers
- Missing baseline: standalone SAM/GSAM

## 16. Codegen

Development-time scripts. Generated outputs are checked in; not run at build.

### `generate_kernels.py`
- Generates GrokAdamW Q3 kernels (INT8 per-block exp_avg + BF16 stochastic-rounded)
- Generates `compute_absmax_scale_kernel.cu`
- Generates `muon_update_generated.cu` with non-temporal I/O
- Scalar (S) and float4 (V) variants

### `generate_sg2_kernels.py`
- Template-based from `kernel_specs.yaml`
- Generates Ampere (sm_80) and Hopper (sm_90) optimizer kernel variants
- Features: cp.async (Ampere), FP8 E4M3 (Hopper)
- Output goes to `csrc/cuda/sm_80/` and `csrc/cuda/sm_90/`

### `kernel_specs.yaml`
- Lists 12+ optimizer specs
- Each spec: block_size, launch_bounds, state vars + quantization formats, scalars, update math
- Variant axes: GPU (S_F_D, V_F_D, S_Q_D, V_Q_D, S_F_M, V_F_M, S_Q_M, V_Q_M), CPU (cpu_F, cpu_Q)
- Templates use placeholders: STATE_LOAD/STORE, PARAM_LOAD/STORE, GRAD_STORE, EXTRA_LOAD
- GrokAdamW has 16 GPU + 2 CPU variants

### `common_macros.j2`
- Jinja2 macros: synchronization, warp reductions, memory access patterns

## 17. Build

`setup.py` at repo root.

### Backend detection
- HIP: `torch.version.hip is not None`
- CUDA: `torch.cuda.is_available()` (or `FORCE_CUDA=1` for build-only)
- Falls back to CPU otherwise

### CUDA path (WITH_CUDA)
- Generic sources: 18 files (optimizer kernels, distributed scan, MoE, quantization)
- sm_80: 6 files (cp.async scan, backward, fused_elem, optimizers, cpasync variants, muon)
- sm_90: 5 files (FP8 scan, backward, warp-specialized, optimizers, muon)
- sm_100: 3 files (TMA kernels)
- Auto-detects generated sources in `csrc/cuda/generated/`
- Flags: `nvcc -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr`
- Arches: `-gencode arch=compute_{70,75,80,86,89,90,100},code=sm_*`
- Override: `TORCH_CUDA_ARCH_LIST` env var

### HIP path (WITH_HIP)
- Generic sources: same 18
- CDNA-specific: gfx90a (CDNA2), gfx942 (CDNA3, 3 files), gfx950 (CDNA4, 1 file)
- Flags: `hipcc -O3 -std=c++17 --offload-arch=gfx908,gfx90a,gfx942,gfx950`

### CPU path
- 7 core sources + generated
- SIMD: AVX-512 (x86_64) or NEON (ARM64) detected via `-march=native`
- Flags: `g++ -O3 -std=c++17 -fopenmp -ffast-math -funroll-loops`
- OpenMP parallelism at parameter level

### Total
- ~67 source files on CUDA path
- Clean CUDA build for full arch matrix: several minutes
- Editable install: `pip install -e .`

## 18. Recent commits

Newest first.

- **`ea968b6`** — Fix critical bugs and apply optimizations across all optimizer components
  - NeuralGrok: fix `_single_param_step` crash (wrong function name, missing `step_list`)
  - JAX/TPU Pallas: fix tile corruption, infinite recursion in persistent scan
  - SuperGrok v2: replace `except Exception: pass` with `RuntimeError` + warning
  - C++ `dispatch.h`: fix AMD GCN arch parsing for 3-digit codes
  - C++ `ops.cpp`: batch CPU syncs, vectorize Mamba-3 inner loops, cache `g × d_lr` in Prodigy
  - Python fallback: eliminate `.item()` calls in hot path

- **`a6323c9`** — Fix `_single_param_step` bugs in muon, prodigy, grokadamw

- **`6c48166`** — Fix potential out-of-bounds read in AMD `gcnArchName` parsing (3-digit codes)

- **`dbe3ef4`** — Fix 9 bugs and apply FP32 skip optimization across optimizer suite (skip `.to(kFloat32)` when already FP32)

- **`1d930db`** — Wire dead fused kernels and eliminate Python `adamw_step` bottleneck

### Trajectory
- Architecture settled, focus on hot-path performance and correctness
- Recent: kernel fusions, register-resident intermediates, non-temporal I/O, reduction kernel improvements
- Architecture coverage: Hopper FP8 added, CDNA3 BF16 MFMA added, Blackwell + CDNA4 scaffolded
- Bug fix backlog draining: silent exception swallowing, redundant forward passes, meta-net device placement, `id`-based caching fragility, single-param-step bugs

## 19. Known gaps

### Optimization opportunities (`ANALYSIS.md` §8)

| # | Optimization | Impact | Difficulty |
|---|---|---|---|
| 1 | Fuse v1.1 cosine gate into full_step kernel | Medium | Easy |
| 2 | Fuse NeuralGrok amplifier + Adam | Low | Easy |
| 3 | Cache meta-net weights across steps | Low-Medium | Easy |
| 4 | Pre-allocate scan workspace buffers | Low-Medium | Easy |
| 5 | Persistent CUDA streams | Very Low | Easy |
| 6 | Skip `.to(kFloat32)` when already FP32 | Very Low | Easy (partial done) |
| 7 | Switch meta-net GELU from tanh to sigmoid form | Low | Easy |
| 8 | Custom cosine-gate reduction kernel | Low | Medium |
| 9 | Batch Muon Newton-Schulz across 2D params | Low | Medium |
| 10 | CUB segmented sort for batched gradient sort | Low | Medium |

### Design concerns (`ANALYSIS.md` §2)
- Peak weight decay aggressive: sigmoid scheduler can multiply base WD by 20 → effective 5.0 → ~99.3% shrinkage over 1000 steps
- Memorization detection binary: sharp threshold at training_acc=0.995, transition not smooth
- SuperGrok bilevel uses validation data → information advantage in benchmark comparisons

### Test gap
- No explicit fused-CUDA-vs-Python-fallback bitwise/numerical agreement test

### Architecture-specific gaps
- Blackwell: TMEM, MMA.2SM, NVFP4 native — scaffolded, delegates to Hopper FP8
- CDNA4: native FP4 expert, FP6 state — scaffolded, delegates to next-lower tier

### JAX/TPU gaps
- SAM not fully integrated (sharpness field exists, perturbation not wired)
- No explicit bilevel checkpointing (XLA rematerialization handles)
- Quantization INT8 only (no FP8/INT4/MXFP4/FP6)
- Expert load balancing minimal

### Documentation staleness
- README test count: says 6 files, actual 8
- ANALYSIS.md test point count: says 67, actual ~82
- Codegen relationship to setup.py not visible to readers

## 20. Quick reference

### Optimizer feature matrix

| Optimizer | Meta-net | State tensors | Decoupled WD | SAM | Bilevel | Grokking | Fused kernel | Python fallback |
|-----------|----------|---------------|--------------|-----|---------|----------|--------------|-----------------|
| SuperGrok2 | Mamba3+PEER+GRU | 7 | ✓ | ✓ functional | ✓ | ✓ | ✓ | ✓ full |
| SuperGrok15 | MLP 2-layer | 4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| SuperGrok11 | MLP + cosine gate | 4 | ✓ | ✓ | ✓ meta_step | ✓ | ✓ | ✗ |
| GrokAdamW | EMA filter | 3 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ (CPU C++) |
| NeuralGrok | Learned MLP | 2 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Prodigy | distance-aware | 4+init | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Grokfast | EMA amplify | 3 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Lion | momentum | 1 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ CPU |
| LookSAM | AdamW + periodic SAM | 3 | ✓ | ✓ periodic | ✗ | ✗ | ✓ | ✓ CPU |
| Muon | NS ortho 2D / Adam 1D | 1 or 3 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ CPU |

### Compile-time constants

| Constant | Value | Used where |
|----------|-------|------------|
| `MAX_D_STATE` | 32 | scan state dim cap |
| `MAX_D_INNER` | 32 | Mamba inner dim cap |
| `MAX_D_MODEL` | 16 | projection dim cap |
| `MAX_GRU_HIDDEN` | 8 | GRU hidden cap |
| `MAX_EXPERT_HIDDEN` | 16 | expert MLP cap |
| `MAX_TOPK` | 4 | PEER top-k |
| `PSCAN_BLOCK` | 512 | Blelloch threads/block |
| `PSCAN_THRESHOLD` | 256 | seq vs parallel scan switch |
| `GEMM_PRECOMPUTE_THRESHOLD` | 1024 | custom vs cuBLAS precompute switch |

### Decision tree for SuperGrok v2 forward
- N < 256 → sequential `mamba3_scan_kernel`
- 256 ≤ N < 1024 → `mamba3_parallel_precompute_kernel` + `mamba3_parallel_scan_kernel`
- N ≥ 1024 → `bilevel_precompute_gemm` (cuBLAS) + `mamba3_parallel_scan_kernel`

### Architecture tier fallback chains
- NVIDIA: Blackwell → Hopper → Ampere → Generic
- AMD: CDNA4 → CDNA3 → Generic

### Precision auto-selection chain
- nvfp4 → mxfp4 → fp8 → bf16 → fp32

### Where to find
- Optimizer Python: `grokking_optimizers/<name>.py`
- Optimizer JAX: `supergrok2_jax_tpu/<name>_jax.py` or in `simple_optimizers_jax.py`/`metanet_optimizers_jax.py`
- Optimizer kernel: `csrc/cuda/generic/<name>_kernels.cu`
- Arch-specific kernel: `csrc/cuda/sm_{80,90,100}/<name>_sm{80,90,100}.cu`
- HIP kernel: `csrc/hip/cdna{2,3,4}/<name>_cdna{2,3,4}.hip.cpp`
- Common headers: `csrc/common/`
- Quantization kernels: `csrc/quantization/`
- C++ binding: `csrc/common/ops.h`, `csrc/common/ops.cpp`
- Tests: `tests/test_<topic>.py`
- Benchmarks: `benchmarks/<name>.py`
- Codegen: `codegen/<name>.py` + `kernel_specs.yaml`
- Build: `setup.py`
- User docs: `README.md`
- Internal review: `ANALYSIS.md`


---

## 21. Overlay merges (this session)

The 19 `*_overlay.*` files that predated the all-specialized refactor have been folded into the per-arch kernel tree. Three patterns were used:

### 21.1. Inlined into canonical launchers (math-mode-style overlays)

For overlays that wrapped a canonical launcher with a Tensor-Core math mode change, the wrap behavior was inlined directly into the canonical `launch_X` body. The arch-suffixed wrapper function is gone.

- `csrc/kernels/cuda/sm_80/muon_sm80.cu` — `launch_muon_fused_step` now opens an `AmpereTF32Scope` RAII helper at the top, setting cuBLAS to `CUBLAS_TF32_TENSOR_OP_MATH` for the Newton-Schulz GEMMs (~2× over FP32 on A100). Restored on scope exit.
- `csrc/kernels/cuda/sm_90/muon_sm90.cu` — `launch_muon_fused_step` now uses a `hopper_fp8_mm` helper (`cublasGemmEx` with `CUDA_R_8F_E4M3` inputs, FP32 accumulation). The leading `X^T @ X` GEMM uses FP8 when both dims ≥ 64; smaller GEMMs stay FP32. ~4× over FP32 on H100.
- `csrc/kernels/hip/gfx942/muon_gfx942.hip.cpp` — `launch_muon_fused_step` now has a CDNA3 BF16 MFMA fast path at the top: when M ≥ 128 and 2D, converts the buffer to BF16 and runs the full NS chain through `torch::mm` (rocBLAS dispatches to `MFMA_F32_32x32x8_BF16`, ~2×). Below M = 128 falls through to FP32.

### 21.2. Deleted (trivial delegators)

These overlays did nothing but delegate to a generic launcher — the wrap was a no-op once the canonical was renamed into a per-arch namespace.

- `csrc/kernels/cuda/sm_90/metanet_optimizers_sm90_overlay.cu` (Hopper metanet wrappers that delegated to `*_ampere`)
- `csrc/kernels/cuda/sm_100/supergrok2_sm100_overlay.cu` (Blackwell SG2 that delegated to `*_hopper`)
- `csrc/kernels/hip/gfx942/metanet_optimizers_gfx942_overlay.hip.cpp` (CDNA3 metanet wrappers that delegated to generic)

### 21.3. Renamed + namespace-wrapped (arch-tuned new code)

These overlays added genuinely new kernels (cp.async pipelining, warp specialization, TMA scaffolding, FP4/FP6/2:4-sparsity). They could not be discarded without losing engineering work, and a clean inline-into-canonical replacement requires hardware to verify. Each was renamed (drop `_overlay` suffix), wrapped in `namespace sg::<arch> { ... }`, and had its `#include "dispatch.h"` / `#include "ops.h"` stripped.

| Old path | New path | Arch | Lines |
|----------|----------|------|------:|
| `cuda/sm_80/metanet_optimizers_sm80_overlay.cu` | `metanet_optimizers_sm80.cu` | sm80 | 706 |
| `cuda/sm_80/metanet_cpasync_variants_sm80_overlay.cu` | `metanet_cpasync_variants_sm80.cu` | sm80 | 1477 |
| `cuda/sm_80/supergrok2_backward_sm80_overlay.cu` | `supergrok2_backward_sm80.cu` | sm80 | 576 |
| `cuda/sm_80/supergrok2_fused_elem_sm80_overlay.cu` | `supergrok2_fused_elem_sm80.cu` | sm80 | 358 |
| `cuda/sm_80/supergrok2_scan_sm80_overlay.cu` | `supergrok2_scan_sm80.cu` | sm80 | 980 |
| `cuda/sm_90/supergrok2_backward_sm90_overlay.cu` | `supergrok2_backward_sm90.cu` | sm90 | 203 |
| `cuda/sm_90/supergrok2_scan_sm90_overlay.cu` | `supergrok2_scan_sm90.cu` | sm90 | 431 |
| `cuda/sm_90/supergrok2_warp_specialized_sm90_overlay.cu` | `supergrok2_warp_specialized_sm90.cu` | sm90 | 445 |
| `cuda/sm_100/supergrok2_precompute_sm100_overlay.cu` | `supergrok2_precompute_sm100.cu` | sm100 | 239 |
| `cuda/sm_100/supergrok2_scan_sm100_overlay.cu` | `supergrok2_scan_sm100.cu` | sm100 | 862 |
| `hip/gfx942/supergrok2_gfx942_overlay.hip.cpp` | `supergrok2_gfx942.hip.cpp` | gfx942 | 464 |
| `hip/gfx950/cdna4_kernels_gfx950_overlay.hip.cpp` | `cdna4_kernels_gfx950.hip.cpp` | gfx950 | 2491 |

These files now participate in the build. The per-feature split of the cdna4 kernels file (FP4 expert / FP6 state / 2:4 sparsity / fused combos) is deferred — the 14 kernels in that file share FP4 helper functions at the top that would need either duplication or extraction into a header.

### 21.4. Inlining the arch-suffixed variants into canonical (post-refactor)

The arch-suffixed launcher variants in §21.3 (`launch_X_ampere`, `launch_X_hopper`, `launch_X_cdna3`) coexist with the wrapped baseline `launch_X` inside the same per-arch namespace. The bindings layer currently calls `launch_X` (the wrapped baseline). To activate the arch-tuned path, either:

  a. Replace the canonical `launch_X` body with the arch-suffixed body (drop the suffix). This requires reading both bodies per kernel and verifying behavior identity — best done on hardware.
  b. Update the per-optimizer binding to call the arch-suffixed launcher when the detected arch matches. Less invasive but adds a second dispatch decision.

Both approaches are mechanical given a working build. Until then, the build links both and only the wrapped baseline is reachable from Python.

---

## 22. Bindings layer state

The pybind11 module surface is restored from the deleted 1700-line `csrc/common/ops.cpp`. Every Python-facing entry point that the optimizers in `grokking_optimizers/*.py` call is registered, except SG v2.

### 22.1. Registered entry points (callable from Python)

Each per-optimizer `csrc/bindings/<optimizer>.cpp` defines two API surfaces:

  - **High-level vector-signature** — primary contract. Takes
    `std::vector<torch::Tensor>&`, runs host-side bookkeeping
    (per-step `bc1/bc2`, gradient clipping, batched norm reductions
    with single CPU sync), then dispatches into the per-arch
    multi-tensor or per-tensor launcher via `SG_DISPATCH_CALL`.
    Names match the deleted `ops.cpp` exactly so Python optimizers
    work unchanged.
  - **Per-tensor wrappers** — escape hatches for tests. Take individual
    `torch::Tensor` args and use the early-returning `SG_DISPATCH`.

Both are registered in `csrc/bindings/module.cpp`. The vector-signature names below match what `grokking_optimizers/*.py` calls via `_ops.<name>`:

| Optimizer | Vector entry points (registered) |
|-----------|----------------------------------|
| GrokAdamW | `grokadamw_fused_step`, `fused_adamw_simple_step` |
| Lion | `lion_fused_step` |
| Grokfast | `grokfast_fused_step` (EMA only), `grokfast_fused_ema_adam_step` |
| Prodigy | `prodigy_fused_step` (returns updated `d_lr`) |
| NeuralGrok | `neuralgrok_fused_step` |
| LookSAM | `looksam_perturb_all`, `looksam_restore_all`, `looksam_compute_directions_and_adjust` |
| Muon | `muon_fused_step` |
| SG v1.5 | `supergrok15_fused_step`, `supergrok15_sam_perturb_all`, `supergrok15_sharpness_restore_all` |
| SG v1.1 | `supergrok11_fused_step`, `supergrok11_sam_perturb_all`, `supergrok11_sharpness_restore_all` |
| MoE | `moe_dynamic_expert_{load,fwd,bwd}`, `moe_filter_active_params`, `moe_scan_compacted`, `moe_scatter_results`, `moe_count_expert_activations`, `moe_compute_load_balance_loss`, `moe_apply_frequency_scaling` |
| Distributed scan | `distributed_scan_phase{1,2,3}` |
| Quantization | `fp8_e4m3_quantize`, `int8_symmetric_quantize`, `int4_gptq_quantize`, `mxfp4_quantize` |

### 22.2. Stubbed (NOT yet registered) — SG v2

`csrc/bindings/supergrok2.cpp` is scaffolded with a `DECLARE_SG2(NS)` macro shape and a clear TODO block. The per-arch launchers exist:

  - `sg::<arch>::launch_mamba3_peer_step` (50+ args)
  - `sg::<arch>::launch_mamba3_peer_batched_step`
  - `sg::<arch>::launch_mamba3_peer_bilevel_fwd_save`
  - `sg::<arch>::launch_mamba3_peer_bilevel_fwd_save_batched`
  - `sg::<arch>::launch_mamba3_peer_backward`
  - `sg::<arch>::launch_mamba3_peer_backward_batched`

in `csrc/kernels/<arch>/supergrok2_{fwd,bwd}_<arch>.{cu,hip.cpp}`. The bindings need thin SG_DISPATCH-style wrappers that take the same args as each per-arch launcher. The signatures are 30–60 args each — large but mechanical to fill in given a build to verify against. Until that work is done, `_ops.supergrok2_*` raises `AttributeError` and the SG v2 Python optimizer falls into its meta-net Python fallback path. Reference for exact wrapper bodies: `git show 682eab4^:csrc/common/ops.cpp` (lines 908–1199, 1499–1611 in `ops.cpp`).

### 22.3. Stale-reference audit

The following stale references were cleaned up:

  - `grokking_optimizers/supergrok2.py` — removed `from grokking_optimizers.jit import create_specializer` (the JIT package was deleted in `104f3ff`).
  - `grokking_optimizers/jit_kernels.py` — deleted (self-contained runtime JIT compiler that nothing imported; obsolete under the no-fallback policy).
  - `tests/test_cpu_fallback.py:test_setup_cpu_sources` — replaced with `test_setup_kernel_sources` (the old test referenced `csrc/cpu/generic/` paths that were deleted in `95b77e0`).

`csrc/bindings/supergrok2.cpp` and a handful of overlay files retain `TODO(post-refactor)` markers to flag work that needs hardware to complete.

---

## 23. Where engineering work picks up

This section was the original "what to do next" pointer. After the
post-refactor sweep + CUTLASS scaffold + ninja build + autotune
wiring + SG v2 binding completion, **§24 (per-arch per-optimizer
rundown) and §25 (engineering remaining) supersede this section.**
Items below are kept as historical reference for the navigation path
that led to the current state. For the live work-remaining list,
read §25.

The codebase compiles to a working _ops Python extension if a build succeeds (this session has not run the build). Once the user runs `bash build.sh` (or `pip install -e .`) and a GPU is available:

1. **Verify build** — `bash build.sh --no-autotune` runs `pip install -e . --no-build-isolation -v` through ninja with a tqdm progress bar. Any signature drift the C++ compiler catches will land in `build.log`. Most likely sources of build-time signature drift were the SG v1.1 `launch_sg11_*` signatures, muon launcher arg orders, and SG v1.5's 21-arg full-step signature — all rewritten this session to match the canonical kernel signatures (see §22.1).
2. **Run cross-arch agreement** — `pytest tests/test_cross_arch_agreement.py`. This is the regression net for any future per-arch kernel divergence; `FORCE_ARCH=<n>` cycles through compiled-in arches.
3. **Run CUTLASS parity** — `git submodule update --init --recursive third_party/cutlass`, then `WITH_CUTLASS=1 bash build.sh`, then `pytest tests/test_cutlass_parity.py`. Verifies CUTLASS GEMM output matches cuBLAS within FP tolerance for Muon NS and SG2 projections on Hopper+/Blackwell.
4. **Run autotune** — `bash build.sh --autotune` does a stub-config build, runs `python autotune/tune.py` to sweep grids, writes winners between the `// AUTOTUNE_BEGIN` / `// AUTOTUNE_END` markers in `csrc/common/tuned_configs.h`, then rebuilds.
5. **Profile hot paths** — `bash build.sh --profile` builds with `-lineinfo` and runs `ncu --set full` against `benchmarks/profile_smoke.py` (5 steps × 11 optimizers).
6. **Walk the per-arch per-optimizer rundown** — see §24 for the hot-path-per-cell map; pick a (optimizer, arch) cell and start hand-tuning. §25 lists the highest-payoff items in priority order.

For each kernel, the hot path lives in:

| Kernel topic | sm_80 file | sm_90 file | sm_100 file | gfx942 file | gfx950 file |
|---|---|---|---|---|---|
| Muon (NS + update) | `muon_sm80.cu` (TF32) | `muon_sm90.cu` (FP8) | `muon_sm100.cu` (baseline) | `muon_gfx942.hip.cpp` (BF16 MFMA) | `muon_gfx950.hip.cpp` (baseline) |
| SG2 forward/scan | `supergrok2_fwd_sm80.cu` + `supergrok2_scan_sm80.cu` (cp.async) + `metanet_cpasync_variants_sm80.cu` | `supergrok2_fwd_sm90.cu` + `supergrok2_warp_specialized_sm90.cu` | `supergrok2_fwd_sm100.cu` + `supergrok2_precompute_sm100.cu` (TMA scaffolding) | `supergrok2_fwd_gfx942.hip.cpp` + `supergrok2_gfx942.hip.cpp` (BF16 MFMA precompute) | `supergrok2_fwd_gfx950.hip.cpp` + `cdna4_kernels_gfx950.hip.cpp` (FP4 expert MFMA) |
| SG2 backward | `supergrok2_bwd_sm80.cu` + `supergrok2_backward_sm80.cu` | `supergrok2_bwd_sm90.cu` + `supergrok2_backward_sm90.cu` | `supergrok2_bwd_sm100.cu` | `supergrok2_bwd_gfx942.hip.cpp` | `supergrok2_bwd_gfx950.hip.cpp` |
| SG v1.5/1.1 metanet | `metanet_optimizers_sm80.cu` (cp.async pipelined weight load) | (baseline, FP8 deferred — small MLP) | (baseline) | (baseline, BF16 MFMA marginal — small MLP) | (baseline) |
| FP6 state / FP4 expert / 2:4 sparsity | n/a | n/a | n/a | n/a | `cdna4_kernels_gfx950.hip.cpp` |


---

## 24. Per-arch per-optimizer rundown

This section walks every optimizer over every supported arch and
describes the shape of the hot path: what's identical to the
canonical math, what's arch-specific, where tensor cores are wired,
which precision is in flight, what memory tier holds what, and where
hand-tuning is still pending. Pure English; no code or pseudocode.
The goal is a printable map that lets a hardware-equipped engineer
walk into any (optimizer, arch) cell and know what to expect before
opening the file.

### 24.1 SuperGrok v2 — Mamba-3+PEER+GRU meta-net

The largest optimizer. Each step runs a recurrent affine prefix scan
over packed segments, an in-place GRU, a PEER product-key router into
a stack of FP32 expert MLPs, and an Adam+Lamb update. The hot paths
are the prefix scan, the projection GEMMs (`in_proj_x`, `in_proj_z`,
`dt_proj`, `B_proj`, `C_proj`), and the expert MLP.

**sm_80 (Ampere — A100 / A30 / A10):** canonical math; uses Ampere
TF32 tensor cores via cuBLAS for projection GEMMs; the prefix scan
and expert MLP both have cp.async double-buffered shared-memory
loads (the variants in `metanet_cpasync_variants_sm80.cu` were
inlined into the canonical full-step launcher this session). Shared
memory budget per SM is 164 KB on A100 / 100 KB on RTX 30; the scan
kernel sizes its cp.async stages to fit either. Register pressure
is high (~96 per thread); occupancy targets are 2 blocks/SM. Expert
weights live in shared memory pinned with `__ldg` for L1 hits;
`bc1`/`bc2` arrive as scalars. GEMMs go through cuBLAS (TF32).
Hand-tune: pipeline-depth = 3 vs 4, segment block size for the
prefix scan, expert-tile shape per `(num_experts, expert_hidden)`.
Gotcha: A10 has only 24 GB; SG2 with `d_state=128` and large
`expert_hidden` blows past the workspace limit — fall back to
`d_state=64`.

**sm_89 (Ada Lovelace — RTX 40 / L40 / L40S):** canonical math;
identical to sm_80 today (the wrapped baseline was ported from
sm_90 in commit `bf157b4`). FP8 E4M3 tensor cores are present on
Ada but the SG2 projections currently use FP16 → no FP8 path is
active. Shared memory per SM is 100 KB. cuBLAS TF32 path; CUTLASS
not yet engaged on this arch (kept on cuBLAS — see §0.5 CUTLASS
note). Hand-tune: small-batch occupancy (sm_89 has 128 SMs at most
on L40S, often half on RTX 40-series); FP8 path for the projection
GEMMs would give the largest wins on consumer Ada. Gotcha: L40S
lacks NVLink, so the `distributed_*pipeline` files fall back to
PCIe ring all-gather.

**sm_90 (Hopper — H100 / H200):** canonical math plus FP8 E4M3
CUTLASS GEMMs for the five projections when `WITH_CUTLASS=1` is
set. The warp-specialized scan kernel
(`launch_scan_warp_specialized` and `launch_scan_warp_specialized_d16`
in `supergrok2_warp_specialized_sm90.cu`) is declared in the
`sg::sm90` namespace but is **not yet wired** into the canonical
`launch_mamba3_peer_batched_step` — see §25. Shared memory is 228
KB per SM (Hopper's bonus); register pressure tighter (~80 per
thread to keep 2 blocks/SM at the launch bounds). TMA descriptors
are scaffolded in `supergrok2_precompute_sm100.cu` but not on
sm_90. GEMMs: CUTLASS sm_90a (FP8 E4M3 inputs, FP32 accumulate)
when opted in, else cuBLAS. Hand-tune: warp-specialization
producer/consumer ratio; CTA cluster size (Hopper supports 16-CTA
clusters via DSMEM); tail-effect on small batches. Gotcha: the
suffixed-launcher's non-FP8 fallback referenced `ampere_*` symbols
visible only from `sg::sm80`; full FP8 inline is deferred — see §25.

**sm_100 (Datacenter Blackwell — B100 / B200 / GB200):** canonical
math; TMA-pre-compute scaffolding lives in
`supergrok2_precompute_sm100.cu` (was `_blackwell` suffix; renamed
this session). Shared memory budget per SM is 228 KB (same as
Hopper). Register pressure is very high (~112 per thread for the
fused full-step launcher); occupancy is 1 block/SM at full feature
set. NVFP4 is **not** yet active here — that lives on sm_103. GEMMs
go through CUTLASS sm_100a when `WITH_CUTLASS=1`, else cuBLAS.
Hand-tune: TMA descriptor reuse across scan segments; 4th-gen
tensor core utilization for the dt_proj fused softplus epilogue
(currently unfused — see §25). Gotcha: B100/GB200 cluster topology
favors 16-CTA clusters; the scan kernel does not yet exploit
DSMEM-shared cross-CTA reductions.

**sm_103 (Blackwell Ultra — B300 / GB300 NVL72):** canonical math;
the **NVFP4 hot path** lives here for the projections via CUTLASS
sm_103a target (NVFP4 is a 4-bit FP format with shared exponent
blocks of 16; native to Blackwell Ultra tensor cores). Shared memory
per SM matches sm_100 (228 KB). Register pressure similar. GEMMs:
CUTLASS sm_103a when `WITH_CUTLASS=1`, else cuBLAS (cuBLAS does NOT
have NVFP4 on Blackwell Ultra at the time of this writing — opting
out of CUTLASS here means falling back to FP16). Hand-tune: NVFP4
block-scaling factor calibration; the autotune grid in
`autotune/grids.py` already lists the sm_103a NVFP4 entries, the
profiler binary needs running on hardware. Gotcha: NVFP4 requires
careful per-block scale handling; numerical accuracy validation must
guard against scale-overflow.

**sm_120 (Consumer Blackwell — RTX 50 / RTX PRO 6000):** canonical
math; uses CUTLASS sm_120a target when `WITH_CUTLASS=1`. **Shared
memory is 128 KB per SM** — significantly less than sm_100 / sm_103's
228 KB. The scan and full-step kernels need re-tuned tile sizes for
this constraint; current placeholder values in `tuned_configs.h`
match sm_100, which will under-occupy on sm_120. GEMMs: CUTLASS
sm_120a or cuBLAS. Hand-tune: shared-memory tile reduction (likely
halve the segment block size); FP4 / NVFP4 are present on RTX PRO
6000 but the consumer RTX 50 cards have varying tensor-core mix.
Gotcha: RTX PRO 6000 has 96 GB; consumer RTX 5090 has 32 GB —
respect the workspace ceiling.

**gfx942 (CDNA3 — MI300X / MI300A):** canonical math plus a CDNA3
BF16 MFMA fast path inlined into `launch_mamba3_peer_batched_step`
(this session — §21.1). The pipeline runs setup+sort →
`cdna3_precompute_bf16` (BF16 MFMA via `torch::mm` →
`MFMA_F32_32x32x8_BF16`) → scan+fused-elem. LDS budget per CU is
64 KB; register pressure is moderate (~64 VGPRs). MI300X has 192 GB
HBM3 — workspace effectively unlimited. GEMMs: rocBLAS
(MFMA-backed BF16 matmul). Hand-tune: BF16 MFMA tile shape, LDS
double-buffering depth, async copy queue depth. Gotcha: MI300X
unified memory between CPU and GPU on MI300A means the `param`
buffer can live in CPU memory pages; verify the kernel sees
device pointers.

**gfx950 (CDNA4 — MI350X / MI355X):** canonical math; the
**FP4 expert MFMA** path lives in `fp4_expert_kernels_gfx950.hip.cpp`
(post-split — §21.3). The expert weights are stored as packed FP4
(8 values per uint32) in HBM; loaded and dequantized to FP32 via
helpers in `csrc/common/fp4_helpers.hip.h`; the MMA itself uses
`__builtin_amdgcn_mfma_f32_16x16x128_fp4`. FP6 E3M2 state packing
lives in `fp6_state_kernels_gfx950.hip.cpp`; 2:4 structured sparsity
in `sparse24_kernels_gfx950.hip.cpp`. LDS per CU is 64 KB. GEMMs:
rocBLAS for the FP32 fallback; native FP4 MMA for expert weights;
no CUTLASS here. Hand-tune: FP4 quant scale calibration, LDS-bank
conflict avoidance, FP6 unpack throughput vs the affine-scan
recurrence. Gotcha: stochastic rounding for FP4 quant uses a Philox
hash (`philox_hash` in fp4_helpers); the seed must be deterministic
across distributed ranks.

**TPU v5p (128-wide MXU):** the JAX/Pallas implementation in
`csrc/kernels/tpu/v5p/` (re-exports from `_pallas_kernels.py`) tiles
the prefix scan and projections for the 128-lane MXU; `dt_proj`'s
softplus epilogue is naturally fused in Pallas via XLA fusion. No
FP4/FP6/FP8; uses BF16 throughout. HBM pressure is 32 GB per chip;
v5p pods scale to 8960 chips. Hand-tune: pjit sharding spec for
`d_state` and `num_experts`; the host XLA cache should be warmed.
Gotcha: TPU v5p does NOT support custom CUDA kernels — the entire
SG2 path runs through Pallas; `_ops.supergrok2_*` is unused on TPU.

**TPU v6e (256-wide MXU):** identical math to v5p; tiled for the
256-lane MXU instead. The Pallas kernel module re-exports tile-256
variants via `csrc/kernels/tpu/v6e/`. Effective throughput per chip
is roughly 2× v5p for projection GEMMs; the prefix scan is bound
by HBM bandwidth in both cases and gains less. Hand-tune: lane-256
tile shape for the expert MLP; this is wider than the typical
expert hidden size (8–32) so most tiles will be padded — the Pallas
kernel needs explicit tile slicing to avoid wasted MXU cycles.

### 24.2 SuperGrok v1.5

A grokking-aware optimizer with a small two-layer MLP meta-net
(`hidden_dim` ∈ {16, 32, 64, 128} typical), Lamb-style trust-ratio
update, fused SAM perturb / sharpness restore, and per-step
gradient clipping. The hot path is the fused 21-arg full-step
launcher.

**sm_80:** canonical math; the cp.async-pipelined weight-load
variant (formerly `_ampere`) is now the canonical body of
`launch_fused_supergrok15_full_step` — sets cuBLAS to
`CUBLAS_TF32_TENSOR_OP_MATH` via the `AmpereTF32Scope` RAII helper,
then dispatches to one of four templated H={16,32,64,128} fast
paths or the runtime-H cp.async kernel. Shared memory budget is
~16 KB per block (4 weight tiles × hidden_dim floats + 1 scratch).
Register pressure is low (~48); occupancy 4 blocks/SM. GEMMs go
through cuBLAS (TF32) for the projection step, but the meta-net
itself is small enough to fit in registers per block. Hand-tune:
hidden_dim-specialized launch bounds; cp.async stages.

**sm_89:** canonical math; baseline ported from sm_90. FP8 path on
the meta-net is **not yet active** — the MLP is small (~64-256
params) so FP8 buys little vs the cp.async weight-pipelining win.
cuBLAS TF32 for the trust-ratio cuBLAS reduction. Hand-tune: at
sm_89's 100 KB shared memory, slightly lower occupancy is OK because
the per-block smem footprint is small.

**sm_90:** canonical math; baseline. FP8 deferred — the meta-net
weights (`W1`, `b1`, `W2`, `b2`) are FP32 in the current Python
optimizer and do not benefit from FP8 conversion at scale. The
warp-specialized scan does not apply (the SG1.5 step is element-wise
in the param dimension, not a recurrence). GEMMs: none — the meta-net
runs through the per-thread inline ALU. Hand-tune: hidden-dim 128
specialization for Hopper's larger register file.

**sm_100:** canonical math; baseline. TMA scaffolding does not apply
(no large GEMM in the hot path). Smem budget ample (228 KB/SM).
Hand-tune: launch bounds for B100/B200 SM count differences; not
much to gain from Blackwell-specific features here vs sm_90.

**sm_103:** canonical math; baseline. NVFP4 inapplicable to a small
meta-net. Same as sm_100.

**sm_120:** canonical math; baseline. Smem 128 KB/SM but the SG1.5
per-block footprint is ~16 KB, so the constraint doesn't bite.
Tile launch on RTX 50 / PRO 6000 should target high occupancy.

**gfx942:** canonical math; baseline. BF16 MFMA marginal for the
small MLP — the matmul is too small to amortize MFMA setup cost.
LDS budget per CU is 64 KB; well under.

**gfx950:** canonical math; baseline. FP4 expert MFMA path doesn't
apply (no expert MoE in SG1.5). Inherits gfx942's "MFMA marginal"
gotcha.

**TPU v5p:** Pallas implementation tiles the meta-net 2-layer MLP
across the 128-wide MXU; bias-corrections and Lamb trust-ratio run
in parallel via XLA. BF16. SAM perturb/restore are fused into
`pjit` graph regions.

**TPU v6e:** identical to v5p; tile 256. The meta-net is small
enough that v6e's 2× throughput advantage shows up only in the
projection step, not the meta-net itself. Hand-tune: avoid padding
the MXU below 256 lanes when `hidden_dim < 256` — pack multiple
parameters' meta-nets into a single MXU launch.

### 24.3 SuperGrok v1.1

Predecessor of SG v1.5 — same grokking-aware MLP meta-net structure
but with a simpler 2-phase pipeline: `launch_sg11_mu_metanet` →
runtime cosine-gate computation → `launch_sg11_adam_decay`. SAM
perturb/restore are also exposed.

**sm_80:** canonical math; baseline mu_metanet kernel wraps in
`AmpereTF32Scope` for the meta-net GEMM. Smem budget similar to
SG1.5 (~16 KB/block). Register pressure low (~48). cuBLAS TF32 GEMM.
Hand-tune: cosine-gate fusion into the metanet kernel (currently a
separate kernel call).

**sm_89:** baseline. Same notes as SG1.5/sm_89.

**sm_90:** baseline. Hopper FP8 deferred for the same MLP-too-small
reason as SG1.5.

**sm_100:** baseline. TMA inapplicable.

**sm_103:** baseline. NVFP4 inapplicable.

**sm_120:** baseline. Smem 128 KB constraint doesn't bite.

**gfx942:** baseline. BF16 MFMA marginal.

**gfx950:** baseline.

**TPU v5p:** Pallas tiling for 128-wide MXU. The mu_metanet → cosine
gate → adam_decay pipeline runs as a single XLA graph, so the
intermediate `cosine_gate` value never leaves SRAM.

**TPU v6e:** tile-256. Same shape as v5p; 2× throughput on the
metanet GEMM.

### 24.4 GrokAdamW

Plain AdamW with grokking-detection scheduling and slow/fast
parameter ramps. The hot path is `grokadamw_fused_step` which
merges the parameter update, decay, and bias-correction across all
parameters in one launch.

**sm_80:** canonical math; baseline. cp.async pipelining for the
multi-tensor parameter list (each block handles one parameter).
Cuda Graph capture friendly. Smem budget ~4 KB/block. Register
pressure low. cuBLAS GEMM not used (purely element-wise).
Hand-tune: per-block param batching to coalesce small parameter
tensors into larger work units.

**sm_89:** baseline. Identical to sm_80 today.

**sm_90:** baseline. Could exploit DSMEM for cross-CTA reductions
when computing the global gradient norm; not yet wired.

**sm_100:** baseline. TMA inapplicable.

**sm_103:** baseline. Lamb path could use NVFP4 for the trust-ratio
GEMM; very low priority — small relative cost.

**sm_120:** baseline. Smem 128 KB/SM is fine for this lightweight
optimizer.

**gfx942:** baseline. LDS budget under-used.

**gfx950:** baseline. FP4 expert MFMA inapplicable.

**TPU v5p:** the AdamW step is naturally vectorizable in Pallas;
runs at MXU peak throughput when shape is divisible by 128.

**TPU v6e:** tile-256.

### 24.5 NeuralGrok

Grokking optimizer with a two-layer MLP "amplifier" that scales the
effective gradient direction. Smaller than SG1.5's meta-net but has
a per-step amplifier-net pass.

**sm_80:** canonical math; baseline amplifier kernel inherits the
TF32 wrap from sm_80's metanet variant inlining. Smem ~8 KB/block.
Hand-tune: amplifier hidden-dim specialization analogous to SG1.5.

**sm_89:** baseline.

**sm_90:** baseline. Same FP8-deferred reasoning as SG1.5.

**sm_100:** baseline.

**sm_103:** baseline. NVFP4 inapplicable.

**sm_120:** baseline. Smem easy.

**gfx942:** baseline.

**gfx950:** baseline.

**TPU v5p:** Pallas tile-128 for the amplifier 2-layer MLP.

**TPU v6e:** tile-256.

### 24.6 Prodigy

Adaptive learning-rate optimizer with a `d_lr` adaptation that
accumulates per-step. Returns the updated `d_lr` from the C++
binding. Hot path: `prodigy_fused_step`.

**sm_80:** canonical math; baseline. Element-wise update; no GEMM.
Smem trivial. cp.async unhelpful (no shared-memory weight reuse).
Hand-tune: warps-per-block for maximum occupancy on a single fused
update + reduction.

**sm_89:** baseline. Same.

**sm_90:** baseline. DSMEM cross-CTA reductions could improve the
adaptive-LR aggregation; deferred.

**sm_100:** baseline.

**sm_103:** baseline.

**sm_120:** baseline. Smem inapplicable (no smem usage).

**gfx942:** baseline. CDNA3 BF16 MFMA inapplicable (no GEMM).

**gfx950:** baseline.

**TPU v5p:** Pallas; the d_lr accumulation is a scalar reduction
across all parameter tensors — a parallel reduction is implicit in
Pallas via XLA.

**TPU v6e:** tile-256.

### 24.7 Grokfast

Two-mode optimizer: an EMA-only mode (`grokfast_fused_step`) and a
GrokFast-EMA + Adam variant (`grokfast_fused_ema_adam_step`). Both
run a single fused element-wise pass per parameter.

**sm_80:** canonical math; baseline. Element-wise; no GEMM. Smem
trivial. Hand-tune: combined parameter batching for many small
tensors (currently one block per parameter).

**sm_89:** baseline.

**sm_90:** baseline.

**sm_100:** baseline.

**sm_103:** baseline.

**sm_120:** baseline.

**gfx942:** baseline.

**gfx950:** baseline.

**TPU v5p:** trivially Pallas-vectorizable.

**TPU v6e:** tile-256.

### 24.8 Lion

Sign-momentum optimizer. The simplest hot path: one fused
sign(beta1·m + (1-beta1)·g) + decay update.

**sm_80:** canonical math; baseline. Pure element-wise; no GEMM.
Sign function compiles to a `selp` PTX instruction (branchless,
warp-uniform).

**sm_89:** baseline.

**sm_90:** baseline.

**sm_100:** baseline.

**sm_103:** baseline.

**sm_120:** baseline.

**gfx942:** baseline. AMD's equivalent of `selp` — compiles to
`v_cndmask_b32`.

**gfx950:** baseline.

**TPU v5p:** Pallas trivially vectorizable; sign function is a
single XLA op.

**TPU v6e:** tile-256.

### 24.9 LookSAM

Sharpness-Aware Minimization variant with periodic direction caching.
Three entry points: `looksam_perturb_all` (param + clones backup),
`looksam_restore_all` (param ← backup), and
`looksam_compute_directions_and_adjust` (batched 2-sync norm
reductions). The norm reductions are the bottleneck.

**sm_80:** canonical math; baseline. The 2-sync reduction (vs N
syncs in the naive form) is the key optimization: stacks all
parameter norms into one device tensor and does a single CPU sync
to read them. Smem trivial. Hand-tune: warp-shuffle reduction over
the per-parameter chunks.

**sm_89:** baseline.

**sm_90:** baseline. DSMEM could combine all CTAs' reductions
without going to global memory; deferred.

**sm_100:** baseline.

**sm_103:** baseline.

**sm_120:** baseline.

**gfx942:** baseline. Wave-reduction primitives on CDNA3 are
similar to NVIDIA's warp-shuffle.

**gfx950:** baseline.

**TPU v5p:** Pallas; the global-norm reduction is a single XLA
all-reduce at the host side.

**TPU v6e:** tile-256.

### 24.10 Muon

Newton-Schulz orthogonalization optimizer for 2D parameters. The hot
path is the 5-step iteration `X ← (a·X + b·X·Xᵀ·X + c·X·Xᵀ·X·Xᵀ·X)`
with constants (a, b, c) = (3.4445, -4.7750, 2.0315). Two GEMMs
per step. Has been the canonical worked-example for arch divergence:
TF32 on Ampere, FP8 on Hopper, BF16 MFMA on CDNA3 — all inlined this
session into the canonical `launch_muon_fused_step` body (§21.1).

**sm_80:** canonical math + Ampere TF32 fast path. Opens an
`AmpereTF32Scope` RAII helper, which sets the cuBLAS handle to
`CUBLAS_TF32_TENSOR_OP_MATH` for the duration of the NS chain;
restores on scope exit. ~2× speedup over plain FP32 on A100. cuBLAS
is the GEMM engine; CUTLASS not engaged on sm_80 (kept on cuBLAS
per the Task 5 spec). Hand-tune: NS-step granularity (current is
unrolled across all 5 steps in one launcher).

**sm_89:** canonical math + TF32 (same as sm_80). FP8 GEMMs not yet
wired here — would buy ~4× over FP32 like Hopper, but consumer Ada's
small SM count limits the absolute win.

**sm_90:** canonical math + Hopper FP8 fast path. Uses a
`hopper_fp8_mm` helper (`cublasGemmEx` with `CUDA_R_8F_E4M3` inputs,
FP32 accumulation) for the leading `X·Xᵀ` GEMM when both dims ≥ 64;
smaller GEMMs stay FP32. ~4× speedup on H100. With `WITH_CUTLASS=1`
the GEMM goes through CUTLASS sm_90a's FP16/BF16 paths instead of
cuBLAS FP8 — the choice between them is currently per-build, not
per-shape; revisit on hardware. Hand-tune: when to cross over from
cuBLAS-FP8 to CUTLASS-FP16; FP8 quant scale calibration.

**sm_100:** canonical math; baseline (no FP8 wrap inlined). CUTLASS
sm_100a routes the GEMMs when `WITH_CUTLASS=1`. The 4th-gen tensor
core on Blackwell could use FP4, but the NS chain is small (typical
2D param is 64×64 to 1024×1024) and FP4 quant overhead is non-trivial.
Hand-tune: NS-iteration count for very small matrices (could drop
from 5 to 3 with a tighter spectral-norm bound).

**sm_103:** canonical math. NVFP4 hot path possible via CUTLASS
sm_103a but not yet activated for Muon (deferred — the spectral norm
bounds for NVFP4 vs FP16 NS are not yet validated).

**sm_120:** canonical math; CUTLASS sm_120a path for the GEMM.
Smem 128 KB — adequate.

**gfx942:** canonical math + CDNA3 BF16 MFMA fast path. When M ≥ 128
and the param is 2D, converts to BF16 and runs the full NS chain
through `torch::mm` (rocBLAS dispatches to `MFMA_F32_32x32x8_BF16`).
~2× speedup. Below M=128 falls back to FP32. CUTLASS not engaged on
AMD.

**gfx950:** canonical math; baseline. CDNA4 has FP4 MMA but Muon is
not (yet) routed through it — same reason as sm_103: spectral-norm
validation needed. rocBLAS for the GEMM.

**TPU v5p:** Pallas; the NS chain is 5 matmuls per step, all
naturally MXU-accelerated. BF16. Tile-128.

**TPU v6e:** tile-256. The NS GEMMs are usually too small (64–1024)
to fully fill the 256-wide MXU; expect modest speedup over v5p.

### 24.11 MoE / Mamba3PEER (auxiliary entries)

Beyond SG2's own MoE-routed expert MLP, a separate MoE binding
surface exposes nine entries
(`moe_dynamic_expert_{load,fwd,bwd}`, `moe_filter_active_params`,
`moe_scan_compacted`, `moe_scatter_results`,
`moe_count_expert_activations`, `moe_compute_load_balance_loss`,
`moe_apply_frequency_scaling`) for use outside SG2. These power the
Mamba3PEER block in `grokking_optimizers/mamba3_peer_metanet.py`
which can run independently of any optimizer.

**sm_80:** canonical math; baseline. Token routing uses a top-k
selection kernel; expert weights loaded with cp.async pipelining.
Hand-tune: top-k threshold for L1-vs-shared-memory expert weight
caching.

**sm_89:** baseline.

**sm_90:** baseline. Hopper warp specialization is the highest-value
wire-up here (each expert's MLP becomes a producer/consumer pipeline).

**sm_100:** baseline. TMA descriptor reuse across expert loads is
the highest-value Blackwell win.

**sm_103:** baseline. NVFP4 expert weights on sm_103a would be a
direct MoE quantization win — needs profiling against accuracy.

**sm_120:** baseline. Smem 128 KB constrains how many experts fit
in cache simultaneously.

**gfx942:** baseline. BF16 MFMA expert MLP is the natural fit.

**gfx950:** **the FP4 expert MFMA hot path** — uses
`fp4_expert_kernels_gfx950.hip.cpp` directly via the MoE bindings;
8 FP4 weights packed per uint32_t in HBM; dequantized to FP32 in
LDS via `fp4_helpers.hip.h`. Stochastic rounding with Philox.
This is the canonical "real divergence" implementation across the
arch matrix.

**TPU v5p:** Pallas; expert routing in a single XLA shard map; v5p's
8-way TPU pod can hold the full expert table in HBM.

**TPU v6e:** tile-256. v6e's 256-wide MXU pairs naturally with
expert hidden dims of 256+; smaller experts pad and waste throughput.

---

## 25. Engineering work remaining

The all-specialized refactor, eight-arch expansion, overlay merges,
SG v2 binding wiring, ninja build wrapper, autotune execution layer,
and CUTLASS migration scaffolding are now complete. What remains is
real per-arch hand-tuning and a small set of items that need
hardware to validate. Rough order of expected payoff:

**1. Hopper FP8 fast-path inlining for SG2 batched step.** The
`launch_mamba3_peer_batched_step_hopper` body had real FP8 E4M3
projections via `cublasGemmEx` (helpers `hopper_fp8_gemm` and
`hopper_precompute_fp8`) but its non-FP8 fallback called
`ampere_batched_scan_and_fused_elem` — defined only in `sg::sm80`
and invisible from `sg::sm90` at link time. To activate, the
fallback path needs restructuring: either copy the `ampere_*`
helpers into `sg::sm90` (preferred — keeps math identical), or
replace the batched_step with a direct in-namespace pipeline that
matches sm_80's structure. After that, the FP8 helpers can be
inlined into the canonical body. Expected ~2× over the current
torch::mm on H100.

**2. Hopper warp-specialized scan activation.** The
`launch_scan_warp_specialized` and `launch_scan_warp_specialized_d16`
declarations in `supergrok2_warp_specialized_sm90.cu` are unwired
from the canonical scan launcher. To activate, the canonical
batched-step needs a code-path that picks the warp-specialized
variant when `d_state` is uniform across all parameters in the
batch — typically true for SG2. Expected ~1.5× on H100/H200 for
long-segment workloads.

**3. Real autotune output for tuned_configs.h.** All 17 optimizers
× 8 GPU arches = 136 entries currently use placeholder
`LaunchConfig` values that match hand-coded `__launch_bounds__` in
the per-arch baselines. Run `bash build.sh --autotune` on hardware:
this does a stub-config build, runs `python autotune/tune.py` to
sweep grids, writes winners between the
`// AUTOTUNE_BEGIN`/`// AUTOTUNE_END` markers in
`csrc/common/tuned_configs.h`, then rebuilds. Expected 5–30%
launch-config wins per arch.

**4. Fused softplus epilogue in CUTLASS for SG2 dt_proj.** The
current `cutlass_dt_proj_fused` runs the unfused linear-combo GEMM
plus a separate `softplus_bias_kernel` post-pass. CUTLASS 3.x's
`EpilogueOp` template can fuse `softplus(x + bias)` into the GEMM
tail, saving one elementwise pass over the dt activation. Math is
identical; the API surface change is internal to CUTLASS.

**5. NVFP4 path for Blackwell Ultra (sm_103) projections.** The
CUTLASS sm_103a target is wired in `setup.py` but the SG2 Python
optimizer still passes FP16 / BF16 for the projection inputs. To
activate, the projection precompute (in the Python pre-step) needs
an NVFP4 quantization pass with proper block-scaling factors;
`autotune/grids.py` already lists the sm_103a NVFP4 entries for
the autotune sweep.

**6. sm_120 retuned tile sizes.** Consumer Blackwell has 128 KB
shared memory per SM versus sm_100 / sm_103's 228 KB. Current
placeholder `tuned_configs.h` values for sm_120 mirror sm_100 and
will under-occupy. The autotune sweep above will detect this; the
specific kernels affected are SG2's batched scan and the metanet
cp.async variants.

**7. CDNA4 FP4 / FP6 / 2:4 sparsity engagement beyond MoE.**
Currently only the MoE expert path uses gfx950's native FP4 MFMA.
Wiring NVFP4-equivalent FP4 into the SG2 projections, FP6 state
into the scan recurrence, and 2:4 sparsity into the dt_proj weights
are all open per-experiment opportunities. Profiling required.

**8. DSMEM for cross-CTA reductions on Hopper / Blackwell.** Norm
reductions (LookSAM, GrokAdamW, Prodigy, the SAM step in SG1.5/1.1)
all currently round-trip through global memory. DSMEM (distributed
shared memory across CTA clusters, available on sm_90+) can do
cross-CTA reductions without that round-trip. Expected ~5–10% on
the global-norm step.

**9. Per-feature gfx950 file split refinement.** The post-split
gfx950 files (`fp4_expert`, `fp6_state`, `sparse24`, `fused_combos`)
currently use `__device__ static __forceinline__` helpers in
`fp4_helpers.hip.h` to avoid ODR. Each TU gets its own internal
copy of every helper; the `__constant__` LUT is wrapped in an
anonymous namespace for the same reason. If a future refactor wants
shared helpers (single copy in the binary), they need to move to a
non-template `.cpp` file with explicit `extern` declarations from
each TU.

**10. CI matrix for the eight-row arch sweep.** Tests
(`test_amd_hip.py`, `test_all_arches.py`,
`test_cross_arch_agreement.py` with the Muon 2D harness,
`test_cutlass_parity.py`) exist but the CI runner needs configuring
to exercise the full {sm_80, sm_89, sm_90, sm_100, sm_103, sm_120,
gfx942, gfx950} × {test_*.py} matrix. The cross-arch agreement
test honors `FORCE_ARCH=<n>` so a single multi-build CI image can
run the full matrix.

**11. CPU SIMD test paths.** The `csrc/kernels/cpu/{avx512,neon}/`
files exist but are testing-only and not exercised under any
public test. A small `tests/test_cpu_simd.py` that runs each
optimizer for a few steps on CPU would catch SIMD regressions
without needing a GPU. Low priority.

**12. PyPI-distributable wheel.** Current build is `pip install -e .`
only. An `auditwheel`-compatible binary wheel build for the
eight-arch fatbin would make distribution trivial; needs a CI
host with all toolchains (CUDA + ROCm + AVX-512). Not urgent
while the project is under active iteration.

