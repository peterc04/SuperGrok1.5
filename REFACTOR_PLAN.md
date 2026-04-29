# REFACTOR_PLAN.md — All-Specialized Kernel Architecture

Status: **Step 1 only — proposal for review. Nothing destructive happens until this is approved.**

This plan covers the migration from the current "generic kernels with arch-specialized variants" structure to "all-specialized kernels, no generic tier, no fallback chain."

## Contents

1. Goals recap
2. Target filesystem layout
3. File-by-file migration map
4. Explicit deletion list
5. New files to be created
6. Refactor list (changes without deletion)
7. Risks and mitigations
8. Open questions
9. Execution plan (steps 2–10)

---

## 1. Goals recap

- All kernels hand-written and arch-specialized. Generic tier deleted.
- Supported arches: sm_80, sm_90, sm_100, gfx942. Build fails on anything else.
- CUTLASS adopted only for SG2 projection GEMMs and Muon Newton-Schulz GEMMs. Other optimizers stay hand-rolled.
- `jit/` runtime specialization replaced by offline `autotune/tune.py` → committed `tuned_configs.h`.
- Pallas TPU kernels colocated under `csrc/kernels/tpu/`, framework-side TPU detection picks the right Pallas kernel.

Untouched / load-bearing:
- Affine2x2 PTX composition in `ptx_intrinsics.cuh`.
- Blelloch parallel scan algorithm itself (tile sizes / prefetch can change).
- Canonical reduction pattern (warp shuffle + per-warp atomicAdd).

## 2. Target filesystem layout

```
SuperGrok1.5/
├── README.md
├── REFRESH.md
├── ANALYSIS.md
├── REFACTOR_PLAN.md            (this file; can be deleted post-merge)
├── pyproject.toml
├── setup.py                    (rewritten: only sm_80/90/100 + gfx942)
├── grokking_race_v2.py
│
├── grokking_optimizers/        (Python optimizer package)
│   ├── __init__.py
│   ├── _ops_loader.py
│   ├── _adamw_helper.py
│   ├── _python_fallback.py     (kept for SG2 only; CPU/test path)
│   ├── dispatch.py             (rewritten: arch detection + exact kernel pick, no fallback chain)
│   ├── quantization.py
│   ├── distributed.py
│   ├── distributed_scan.py
│   ├── cuda_graph_optimizer.py
│   ├── async_supergrok2.py
│   ├── overlap_distributed.py
│   ├── pipelined_optimizer.py
│   ├── partial_graph.py
│   ├── gradient_hook_optimizer.py
│   ├── gradient_compression.py
│   ├── interleaved_states.py
│   ├── moe_deep.py
│   ├── sparse_gradients.py
│   ├── torch_compile_integration.py
│   ├── jit_kernels.py          (TBD: see open questions)
│   ├── grokadamw.py
│   ├── grokfast.py
│   ├── lion.py
│   ├── looksam.py
│   ├── muon.py
│   ├── neuralgrok.py
│   ├── prodigy.py
│   ├── supergrok11.py
│   ├── supergrok15.py
│   ├── supergrok2.py
│   └── mamba3_peer_metanet.py
│
├── supergrok2_jax_tpu/          (JAX/TPU stays here; Pallas kernels moved out to csrc/)
│   ├── __init__.py
│   ├── supergrok2_jax.py
│   ├── mamba3_peer_metanet_jax.py
│   ├── scan.py
│   ├── gru.py
│   ├── peer.py
│   ├── bilevel.py
│   ├── sharding.py             (extended: TPU version detection picks Pallas kernel from csrc/kernels/tpu/)
│   ├── simple_optimizers_jax.py
│   ├── metanet_optimizers_jax.py
│   ├── quantization_jax.py
│   ├── bridge.py
│   ├── distributed_example.py
│   └── tests/
│
├── csrc/
│   ├── common/                 (shared headers + tuned configs)
│   │   ├── platform.h          (simplified: only sm_80/90/100 + gfx942)
│   │   ├── types.h
│   │   ├── ptx_intrinsics.cuh  (UNTOUCHED)
│   │   ├── utils.cuh
│   │   ├── quantization.h
│   │   └── tuned_configs.h     (NEW: autotune output, committed)
│   │
│   ├── bindings/               (NEW: split per optimizer, replaces ops.cpp/ops.h)
│   │   ├── bindings.h          (entry-point declarations)
│   │   ├── module.cpp          (PYBIND11_MODULE definition; aggregates all bindings)
│   │   ├── dispatch.cpp        (arch detection + exact kernel pick, no fallback)
│   │   ├── grokadamw.cpp
│   │   ├── grokfast.cpp
│   │   ├── lion.cpp
│   │   ├── looksam.cpp
│   │   ├── muon.cpp
│   │   ├── neuralgrok.cpp
│   │   ├── prodigy.cpp
│   │   ├── supergrok11.cpp
│   │   ├── supergrok15.cpp
│   │   ├── supergrok2.cpp
│   │   ├── multi_tensor.cpp
│   │   ├── moe.cpp
│   │   ├── distributed_scan.cpp
│   │   └── quantization.cpp
│   │
│   ├── kernels/
│   │   ├── cuda/
│   │   │   ├── sm_80/          (Ampere — cp.async)
│   │   │   │   ├── grokadamw_sm80.cu
│   │   │   │   ├── grokfast_sm80.cu
│   │   │   │   ├── lion_sm80.cu
│   │   │   │   ├── looksam_sm80.cu
│   │   │   │   ├── muon_sm80.cu
│   │   │   │   ├── neuralgrok_sm80.cu
│   │   │   │   ├── prodigy_sm80.cu
│   │   │   │   ├── supergrok11_sm80.cu
│   │   │   │   ├── supergrok15_sm80.cu
│   │   │   │   ├── supergrok2_fwd_sm80.cu
│   │   │   │   ├── supergrok2_bwd_sm80.cu
│   │   │   │   ├── multi_tensor_sm80.cu
│   │   │   │   ├── moe_sm80.cu
│   │   │   │   └── distributed_scan_sm80.cu
│   │   │   ├── sm_90/          (Hopper — FP8, warp specialization, DSMT)
│   │   │   │   └── … (mirror of sm_80 file list, _sm90 suffix)
│   │   │   └── sm_100/         (Blackwell — TMA, FP4 scaffolding)
│   │   │       └── … (mirror, _sm100 suffix)
│   │   ├── hip/
│   │   │   └── gfx942/         (CDNA3 — BF16 MFMA)
│   │   │       └── … (mirror of sm_80 file list, _gfx942.hip.cpp suffix)
│   │   ├── tpu/
│   │   │   ├── v5p/            (NEW: split from supergrok2_jax_tpu/pallas_kernels.py)
│   │   │   │   ├── affine_scan_v5p.py
│   │   │   │   ├── fused_gru_peer_v5p.py
│   │   │   │   └── persistent_scan_fused_elem_v5p.py
│   │   │   └── v6e/            (NEW: same shape, 256-wide tiles)
│   │   │       └── …
│   │   └── cpu/                (testing only; preserved sources, codegen output dropped)
│   │       ├── cpu_ops.cpp
│   │       ├── cpu_kernels.cpp
│   │       ├── all_optimizers_cpu.cpp
│   │       ├── supergrok2_scan_cpu.cpp
│   │       ├── distributed_scan_cpu.cpp
│   │       ├── moe_cpu.cpp
│   │       ├── sg2_fused_scan_elem_cpu.cpp
│   │       ├── avx512/simd_kernels.cpp
│   │       └── neon/simd_kernels.cpp
│   │
│   └── quantization/
│       └── quantization_kernels.cu  (will be split per arch as part of migration)
│
├── autotune/                   (NEW: replaces grokking_optimizers/jit/)
│   ├── tune.py                 (entry point; runs grids, writes csrc/common/tuned_configs.h)
│   ├── grids.py                (per-kernel param grids)
│   ├── runner.py               (microbench harness)
│   └── cutlass_profile.py      (wraps CUTLASS profiler for SG2 + Muon GEMMs)
│
├── tests/                      (kept; some tests will be updated for new arch policy)
└── benchmarks/                 (kept; benchmarks/autotune.py: see open questions)
```

## 3. File-by-file migration map

Format: `current path` → `new path` (or DELETE / REFACTOR-IN-PLACE / KEEP).

### `csrc/common/`
- `platform.h` → KEEP. Simplify: drop sm_70/75/86/89, gfx908/gfx90a/gfx950 macro paths.
- `types.h` → KEEP.
- `ptx_intrinsics.cuh` → KEEP. **Untouched (load-bearing).**
- `utils.cuh` → KEEP.
- `quantization.h` → KEEP.
- `dispatch.h` → DELETE. Replaced by `csrc/bindings/dispatch.cpp` (no fallback chain).
- `ops.h` → DELETE. Replaced by `csrc/bindings/bindings.h`.
- `ops.cpp` → DELETE. Replaced by per-optimizer files in `csrc/bindings/`.

### `csrc/cuda/generic/` (DELETE entire directory after migration)

For each existing file, the kernel becomes specialized variants in `csrc/kernels/cuda/{sm_80,sm_90,sm_100}/` and `csrc/kernels/hip/gfx942/`:

- `grokadamw_kernels.cu` → `kernels/cuda/{sm_80,sm_90,sm_100}/grokadamw_sm{80,90,100}.cu`, `kernels/hip/gfx942/grokadamw_gfx942.hip.cpp`
- `grokfast_kernels.cu` → mirror
- `lion_kernels.cu` → mirror
- `looksam_kernels.cu` → mirror
- `muon_kernels.cu` → mirror (note: SM90/100 use CUTLASS for NS GEMMs)
- `neuralgrok_kernels.cu` → mirror
- `prodigy_kernels.cu` → mirror
- `supergrok11_kernels.cu` → mirror
- `supergrok15_kernels.cu` → mirror
- `supergrok2_mamba_peer_kernels.cu` → split per arch as `supergrok2_fwd_sm{80,90,100}.cu` and `supergrok2_fwd_gfx942.hip.cpp`. SM90/100 use CUTLASS for projection GEMMs.
- `supergrok2_mamba_peer_backward_kernels.cu` → split per arch as `supergrok2_bwd_sm{80,90,100}.cu` and `supergrok2_bwd_gfx942.hip.cpp`. SM90/100 use CUTLASS for projection-backward GEMMs.
- `multi_tensor_optimizer_kernels.cu` → `multi_tensor_sm{80,90,100}.cu` and `multi_tensor_gfx942.hip.cpp`
- `multi_tensor_prepare.cu` → folded into `multi_tensor_*` files per arch
- `moe_deep_kernels.cu` → `moe_sm{80,90,100}.cu` and `moe_gfx942.hip.cpp`
- `distributed_scan_kernels.cu` → `distributed_scan_sm{80,90,100}.cu` and `distributed_scan_gfx942.hip.cpp`
- `distributed_scan_pipeline.cu` → folded into `distributed_scan_*` files per arch
- `distributed_pipeline.cu` → folded into `distributed_scan_*` files per arch

### `csrc/cuda/sm_80/` → `csrc/kernels/cuda/sm_80/`

Existing files **move** and merge with the migrated kernels above:
- `metanet_optimizers_sm80.cu` → merged contents go into per-optimizer files
- `metanet_cpasync_variants_sm80.cu` → merged into `supergrok{11,15,2}_sm80.cu` cp.async paths
- `muon_sm80.cu` → merged into `muon_sm80.cu` (note: same name in new tree)
- `supergrok2_backward_sm80.cu` → contents merged into `supergrok2_bwd_sm80.cu`
- `supergrok2_fused_elem_sm80.cu` → contents merged into `supergrok2_fwd_sm80.cu` (per-element step section)
- `supergrok2_scan_sm80.cu` → contents merged into `supergrok2_fwd_sm80.cu` (scan section)

### `csrc/cuda/sm_90/` → `csrc/kernels/cuda/sm_90/`

Existing files **move** and merge:
- `metanet_optimizers_sm90.cu` → split into per-optimizer files
- `muon_sm90.cu` → `muon_sm90.cu` (extended with CUTLASS NS GEMMs)
- `supergrok2_backward_sm90.cu` → `supergrok2_bwd_sm90.cu` (extended with CUTLASS projection-backward GEMMs)
- `supergrok2_scan_sm90.cu` → `supergrok2_fwd_sm90.cu` (scan section, extended with CUTLASS projections)
- `supergrok2_warp_specialized_sm90.cu` → folded into `supergrok2_fwd_sm90.cu`

### `csrc/cuda/sm_100/` → `csrc/kernels/cuda/sm_100/`

Existing files **move** and merge:
- `supergrok2_sm100.cu` → split into `supergrok2_fwd_sm100.cu` and `supergrok2_bwd_sm100.cu`
- `supergrok2_precompute_sm100.cu` → folded into `supergrok2_fwd_sm100.cu`
- `supergrok2_scan_sm100.cu` → folded into `supergrok2_fwd_sm100.cu`

### `csrc/cuda/sm_75/`, `csrc/cuda/sm_86/`, `csrc/cuda/sm_89/`

DELETE. Empty directories for unsupported arches.

### `csrc/cuda/generated/` (DELETE entire directory)

All 30 codegen-produced files. Replaced by hand-written specialized variants in `csrc/kernels/cuda/{sm_80,sm_90,sm_100}/`.

### `csrc/hip/`

- `csrc/hip/cdna2/supergrok2_scan_cdna2.hip.cpp` → DELETE. CDNA2 (gfx90a) unsupported.
- `csrc/hip/cdna3/metanet_optimizers_cdna3.hip.cpp` → MOVE+SPLIT to per-optimizer files in `csrc/kernels/hip/gfx942/`
- `csrc/hip/cdna3/muon_cdna3.hip.cpp` → MOVE to `csrc/kernels/hip/gfx942/muon_gfx942.hip.cpp`
- `csrc/hip/cdna3/supergrok2_cdna3.hip.cpp` → MOVE to `csrc/kernels/hip/gfx942/supergrok2_{fwd,bwd}_gfx942.hip.cpp` (split forward/backward)
- `csrc/hip/cdna4/cdna4_kernels.hip.cpp` → DELETE. CDNA4 (gfx950) unsupported per spec.
- `csrc/hip/README_HIP.md` → MOVE to `csrc/kernels/hip/README_HIP.md`

### `csrc/cpu/`

Move under new tree, drop codegen output, keep handwritten + SIMD:

- `csrc/cpu/cpu_ops.cpp` → `csrc/kernels/cpu/cpu_ops.cpp`
- `csrc/cpu/cpu_kernels.cpp` → `csrc/kernels/cpu/cpu_kernels.cpp`
- `csrc/cpu/distributed_scan_cpu.cpp` → `csrc/kernels/cpu/distributed_scan_cpu.cpp`
- `csrc/cpu/moe_cpu.cpp` → `csrc/kernels/cpu/moe_cpu.cpp`
- `csrc/cpu/sg2_fused_scan_elem_cpu.cpp` → `csrc/kernels/cpu/sg2_fused_scan_elem_cpu.cpp`
- `csrc/cpu/avx512/simd_kernels.cpp` → `csrc/kernels/cpu/avx512/simd_kernels.cpp`
- `csrc/cpu/neon/simd_kernels.cpp` → `csrc/kernels/cpu/neon/simd_kernels.cpp`
- `csrc/cpu/generic/all_optimizers_cpu.cpp` → `csrc/kernels/cpu/all_optimizers_cpu.cpp`
- `csrc/cpu/generic/supergrok2_scan_cpu.cpp` → `csrc/kernels/cpu/supergrok2_scan_cpu.cpp`
- `csrc/cpu/generated/` → DELETE entire subdirectory (21 codegen files; consolidate equivalent functionality into `all_optimizers_cpu.cpp`).

### `csrc/quantization/`

- `quantization_kernels.cu` → split per arch as `csrc/kernels/cuda/{sm_80,sm_90,sm_100}/quantization_sm{80,90,100}.cu` and `csrc/kernels/hip/gfx942/quantization_gfx942.hip.cpp`. The Python `csrc/bindings/quantization.cpp` dispatches.

### `grokking_optimizers/jit/` (DELETE entire directory)

- `__init__.py` → DELETE
- `block_size_optimizer.py` → DELETE (offline autotune supersedes)
- `cpu_specializer.py` → DELETE
- `cuda_specializer.py` → DELETE
- `gcn_scheduler.py` → DELETE
- `hip_specializer.py` → DELETE
- `multi_gpu_optimizer.py` → DELETE
- `ptx_scheduler.py` → DELETE
- `smem_layout.py` → DELETE
- `specializer.py` → DELETE
- `tpu_specializer.py` → DELETE
- `templates/` → DELETE

### `grokking_optimizers/dispatch.py` → REFACTOR-IN-PLACE

Drop tier fallback chain. Detect arch (one of sm_80, sm_90, sm_100, gfx942), pick the exact specialized binding, raise `RuntimeError` if the arch is anything else. Keep the platform/vendor query helpers used by tests.

### `grokking_optimizers/__init__.py` → REFACTOR-IN-PLACE

Drop `_HAS_*` flag exports gated on optional fallbacks. Replace with a single `_HAS_OPS` flag plus the canonical "supported arch or error" import logic.

### `grokking_optimizers/_ops_loader.py` → REFACTOR-IN-PLACE

Same: load extension or raise. No multi-tier fallback.

### `grokking_optimizers/_python_fallback.py` → KEEP

Used as the SG2 reference implementation for CPU testing and the v2 full-Python fallback (per spec, SG2 is the only optimizer with a robust Python fallback). Not part of the runtime fallback chain being deleted.

### `grokking_optimizers/jit_kernels.py` → TBD (see open questions)

### `supergrok2_jax_tpu/pallas_kernels.py` → MOVE+SPLIT

Pallas kernels move to `csrc/kernels/tpu/v5p/` and `csrc/kernels/tpu/v6e/`. The remaining JAX modules in `supergrok2_jax_tpu/` stay where they are; `sharding.py` (or a new `tpu_dispatch.py`) calls `detect_tpu_version()` and imports the right variant from `csrc/kernels/tpu/`.

### `codegen/` → DELETE entire directory

- `generate_kernels.py` → DELETE
- `generate_sg2_kernels.py` → DELETE
- `kernel_specs.yaml` → DELETE
- `templates/` → DELETE

### `benchmarks/autotune.py` → TBD (see open questions)

### `setup.py` → REFACTOR-IN-PLACE

- Drop `-gencode arch=compute_{70,75,86,89},code=sm_*`.
- Keep only sm_80, sm_90, sm_100.
- HIP: drop `gfx908,gfx90a,gfx950` from `--offload-arch`. Keep only `gfx942`.
- Source list: replace `csrc/cuda/generic/*.cu` and `csrc/cuda/generated/*.cu` collectors with explicit lists from `csrc/kernels/cuda/{sm_80,sm_90,sm_100}/*.cu`. Same for HIP.
- Add CUTLASS include paths (header-only, third-party submodule or vendored) for SG2 + Muon GEMM compilation units.
- Bake `tuned_configs.h` into the include path.
- Build fails clean if the host arch is unsupported and `FORCE_CUDA=1` is not set.

### Tests (`tests/`)

- `test_supergrok2.py` → REFACTOR-IN-PLACE. Some sections (12N–12Q dispatch / arch detection) need to be updated for the new policy: detected arch must be exactly one of {sm_80, sm_90, sm_100, gfx942}, otherwise raise.
- `test_matrix.py` → REFACTOR-IN-PLACE. Drop `FORCE_ARCH=75` / generic-tier paths.
- `test_all_tiers.py` → RENAME to `test_all_arches.py`. Drop generic tier from the matrix; add gfx942 row.
- `test_amd_hip.py` → REFACTOR-IN-PLACE. Drop CDNA2/CDNA4 cases. Test gfx942 only.
- `test_cpu_fallback.py` → REFACTOR-IN-PLACE. CPU is now testing-only, so adjust naming and assertions.
- `test_jax_matrix.py` → KEEP. Add TPU version detection + Pallas-from-`csrc/kernels/tpu/` path test.
- `test_new_features.py` → KEEP.
- `test_training_aware.py` → KEEP.
- **NEW**: `tests/test_cross_arch_agreement.py` — per-optimizer numerical agreement test that runs all four arches on the same inputs and asserts elementwise agreement (modulo a tolerance for FP rounding). Skips arches not present on the host.

## 4. Explicit deletion list

Deletions happen only **after** equivalent specialized variants exist in the new tree, the new bindings compile, and tests pass on the migrated optimizer. Each deletion is a separate commit so it can be reverted if a regression slips through.

### Directories to delete (entire trees)

1. `csrc/cuda/generic/` (17 files)
2. `csrc/cuda/generated/` (30 files)
3. `csrc/cuda/sm_75/` (empty)
4. `csrc/cuda/sm_86/` (empty)
5. `csrc/cuda/sm_89/` (empty)
6. `csrc/cpu/generated/` (21 files)
7. `csrc/cpu/generic/` (after files are moved to `csrc/kernels/cpu/`)
8. `csrc/cpu/avx512/` (after move)
9. `csrc/cpu/neon/` (after move)
10. `csrc/cpu/` (after all files move out)
11. `csrc/hip/cdna2/` (1 file; CDNA2 unsupported)
12. `csrc/hip/cdna3/` (3 files; after move to `csrc/kernels/hip/gfx942/`)
13. `csrc/hip/cdna4/` (1 file; CDNA4 unsupported)
14. `csrc/hip/` (after subdirs deleted; README moves)
15. `grokking_optimizers/jit/` (10 files + `templates/`)
16. `codegen/` (3 files + `templates/`)

### Individual files to delete

- `csrc/common/dispatch.h`
- `csrc/common/ops.h`
- `csrc/common/ops.cpp`

### Files moved (not deleted)

These are deletion-equivalent at the old path but the content lives elsewhere:

- All `csrc/cuda/sm_{80,90,100}/*.cu` → `csrc/kernels/cuda/sm_{80,90,100}/`
- `csrc/cpu/*.cpp` (top level) → `csrc/kernels/cpu/`
- `csrc/cpu/{avx512,neon}/simd_kernels.cpp` → `csrc/kernels/cpu/{avx512,neon}/`
- `csrc/cpu/generic/*.cpp` → `csrc/kernels/cpu/`
- `csrc/hip/cdna3/*.hip.cpp` → `csrc/kernels/hip/gfx942/`
- `csrc/hip/README_HIP.md` → `csrc/kernels/hip/README_HIP.md`
- `csrc/quantization/quantization_kernels.cu` → split per arch under `csrc/kernels/{cuda,hip}/<arch>/quantization_<arch>.cu` (or `.hip.cpp`)
- `supergrok2_jax_tpu/pallas_kernels.py` → split into `csrc/kernels/tpu/{v5p,v6e}/`

### Total file deletion count (after migration finishes)

- 17 generic CUDA files
- 30 generated CUDA files
- 21 generated CPU files
- 3 directories that go empty (sm_75/86/89)
- 1 CDNA2 file + 1 CDNA4 file
- 10 jit files + jit templates dir
- 3 codegen files + codegen templates dir
- 3 individual files in `csrc/common/` (dispatch.h, ops.h, ops.cpp)

Approximate net: ~85 files deleted, ~30 directories emptied or removed.

### NOT deleted

- `csrc/common/{platform.h, types.h, ptx_intrinsics.cuh, utils.cuh, quantization.h}`
- `grokking_optimizers/_python_fallback.py` (SG2 reference impl, used by tests)
- `grokking_optimizers/dispatch.py` (refactored, not deleted)
- All other Python files in `grokking_optimizers/`
- `supergrok2_jax_tpu/` (Pallas kernels move out, rest stays)
- All tests (some are refactored)
- All benchmarks
- `README.md`, `REFRESH.md`, `ANALYSIS.md`

## 5. New files to be created

### `csrc/common/`
- `tuned_configs.h` — autotune output. Tables of `__launch_bounds__`, BLOCK_M/N, STAGES, etc. per (kernel, arch, shape bucket). Committed.

### `csrc/bindings/` (entirely new directory)
- `bindings.h` — declares all Python-callable entry points
- `module.cpp` — `PYBIND11_MODULE(_ops, m)` registration; aggregates per-optimizer registrations
- `dispatch.cpp` — single-arch detection + exact kernel selection; raises on unsupported
- `grokadamw.cpp`, `grokfast.cpp`, `lion.cpp`, `looksam.cpp`, `muon.cpp`, `neuralgrok.cpp`, `prodigy.cpp`, `supergrok11.cpp`, `supergrok15.cpp`, `supergrok2.cpp` — per-optimizer launchers
- `multi_tensor.cpp`, `moe.cpp`, `distributed_scan.cpp`, `quantization.cpp` — supporting bindings

### `csrc/kernels/cuda/sm_80/`, `csrc/kernels/cuda/sm_90/`, `csrc/kernels/cuda/sm_100/`

Per-optimizer specialized files (see § 2 layout). Each kernel in its own translation unit, with `__launch_bounds__` populated from `tuned_configs.h`. SG2 forward and backward have CUTLASS-backed projection GEMMs in sm_90/sm_100 only; sm_80 keeps cuBLAS+cp.async (CUTLASS sm_80 path TBD; see open questions).

### `csrc/kernels/hip/gfx942/`

Mirror of the CUDA file list, `.hip.cpp` extension. Same kernel logic as sm_90 baseline; arch-specific primitives swapped (MFMA where appropriate, BF16 matmul through rocBLAS for the projection step until/unless we add a CUTLASS-equivalent on AMD).

### `csrc/kernels/tpu/v5p/`, `csrc/kernels/tpu/v6e/`

Pallas kernels split out from `supergrok2_jax_tpu/pallas_kernels.py`. Each TPU version gets its own files:
- `affine_scan_<ver>.py` — Pallas tiled affine scan
- `fused_gru_peer_<ver>.py` — Pallas fused GRU + PEER
- `persistent_scan_fused_elem_<ver>.py` — persistent scan + elementwise

### `csrc/kernels/cpu/`

Files moved from `csrc/cpu/` (see § 3). No new files; just relocated.

### `autotune/` (entirely new directory)

- `tune.py` — entry point. Parses CLI, walks each kernel's grid, runs microbench, picks winners, emits `csrc/common/tuned_configs.h`.
- `grids.py` — per-kernel parameter grids (BLOCK_M, BLOCK_N, STAGES, NUM_WARPS, etc.).
- `runner.py` — microbench harness. Builds a tiny test extension, runs N iterations, reports median latency.
- `cutlass_profile.py` — wraps the CUTLASS profiler binary for SG2 projection GEMMs and Muon NS GEMMs. Parses CUTLASS profiler CSV and converts to `tuned_configs.h` entries.
- `README.md` — usage, prerequisites, how to add a new kernel grid.

### Tests

- `tests/test_cross_arch_agreement.py` — per-optimizer numerical agreement test. Runs each available arch on the same inputs, asserts elementwise agreement within FP tolerance. Skips missing arches (so a host with only sm_90 doesn't fail; CI matrix exercises all four).

## 6. Refactor list (non-delete)

These files stay at their current path but their contents change.

### `csrc/common/platform.h`
- Drop sm_70/75/86/89 macro paths (the file currently has logic gated on `__CUDA_ARCH__ < 800` etc.).
- Drop gfx908/gfx90a/gfx950 paths.
- Keep `GROK_CUDA`/`GROK_HIP` selection.
- Keep warp size logic (32 vs 64).
- Keep stream/error aliases, CUB/hipcub alias.
- Keep non-temporal I/O helpers (still relevant for sm_80+).

### `grokking_optimizers/dispatch.py`
- `get_gpu_arch()` returns one of `80`, `90`, `100`, `942` (with `942` denoting gfx942) or raises.
- Drop `get_arch_tier()`, `get_amd_tier()` — no tiers anymore.
- Keep predicates that probe hardware features (`supports_bf16`, `supports_fp8`, etc.) for use in tests, but they no longer drive runtime kernel selection.
- Drop fallback-chain logic.
- `FORCE_ARCH` env var continues to work for tests that want to exercise a specific binding (CI matrix).

### `grokking_optimizers/__init__.py`
- Drop `_HAS_OPS` truthy fallback to Python-only mode for non-SG2 optimizers. Either the extension loads cleanly for the detected arch or import fails.
- SG2's Python fallback is opt-in via `SUPERGROK2_PYTHON_FALLBACK=1` for testing.

### `grokking_optimizers/_ops_loader.py`
- Same change: load or raise. No silent fallback.

### `setup.py`
- Source list rewritten to walk `csrc/kernels/{cuda,hip,cpu}/` and `csrc/bindings/`.
- `-gencode` list trimmed to sm_80, sm_90, sm_100.
- `--offload-arch` trimmed to `gfx942`.
- CUTLASS include path added (vendored or submodule, see open questions).
- Build error if host arch is unsupported and `FORCE_CUDA` is not set.

### `supergrok2_jax_tpu/sharding.py`
- Add TPU dispatch: `detect_tpu_version()` → import from `csrc/kernels/tpu/v5p/` or `csrc/kernels/tpu/v6e/`.
- Remove try/except fallback to pure-JAX where it gated on Pallas API stability — keep pure-JAX as an opt-in code path, not a silent fallback.

### `tests/test_supergrok2.py`
- Sections 12N (precision config auto-selection), 12O (projection precision FP32 vs auto), 12P (dispatch convergence), 12Q (platform/vendor detection): adjust assertions to match the new "exactly one of {80, 90, 100, 942}, otherwise raise" policy.
- Drop any test branch that relied on a Volta/Turing fallback.

### `tests/test_matrix.py`
- Drop the generic-tier (sm_75) row. Add gfx942 row. Skip rows whose arch the host can't run.

### `tests/test_amd_hip.py`
- Drop CDNA2/CDNA4 cases. Test gfx942 only.

### `tests/test_cpu_fallback.py`
- Reframe as "CPU build sanity check," not "fallback when GPU is missing."

### `benchmarks/autotune.py`
- See open questions. Likely deleted or repurposed.

## 7. Risks and mitigations

**Risk: silent numerical drift across arches.**
Mitigation: `tests/test_cross_arch_agreement.py` is mandatory. CI matrix exercises sm_80, sm_90, sm_100, gfx942. Same inputs, elementwise diff under tolerance. Any drift fails the build.

**Risk: lost coverage for unsupported arches.**
Mitigation: explicitly documented. `setup.py` errors on Volta/Turing/Ampere-mid (sm_86/89)/RDNA. Existing user docs (README.md) need a "Hardware support" section update before this lands.

**Risk: CUTLASS sm_80 path is real work, not just a swap.**
Mitigation: keep cuBLAS+cp.async as the sm_80 path for the SG2 projection GEMMs initially. Migrate sm_80 to CUTLASS in a follow-up. SG2 sm_90 is the primary CUTLASS target; sm_100 follows once the sm_90 epilogue is settled.

**Risk: dt_proj fused softplus epilogue is a new CUTLASS epilogue, not a stock one.**
Mitigation: prototype the epilogue against a CUTLASS unit test before wiring it into the SG2 forward path. Verify against the existing `softplus_bias_kernel` output bit-for-bit (after agreed FP tolerance).

**Risk: per-optimizer cross-arch math drift when one arch gets a fix and the others don't.**
Mitigation: every math change is an explicit pass across all four arches. The cross-arch test detects drift; the rule is "if you change the formula in one variant, you must update all variants in the same commit."

**Risk: `tuned_configs.h` becomes stale silently.**
Mitigation: bake autotune output into the build via `#include "tuned_configs.h"`. If the file is missing or stale, the build either fails or emits a warning (TBD; see open questions). The autotune script is run on demand on hardware; a CI job verifies the committed file is consistent with the current grid definitions but does not necessarily re-tune.

**Risk: `__launch_bounds__` from autotune output is wrong for shapes outside the tuned grid.**
Mitigation: the grid covers a representative range of shapes (small / medium / large per optimizer). Out-of-grid shapes round to the nearest tuned bucket. Document this in `autotune/README.md`.

**Risk: codegen deletion loses Q3/Q4 quantized state variants that aren't yet hand-written.**
Mitigation: before deleting `codegen/` and `csrc/cuda/generated/`, port the Q3 quantized state path for GrokAdamW (the only quantized optimizer state in active use; the others are scaffolded). Q4 variants in `csrc/cpu/generated/` are CPU-only and dropped per spec (`csrc/cpu/` is testing-only).

**Risk: TPU detection path picks the wrong Pallas kernel.**
Mitigation: keep `detect_tpu_version()` simple (read `device_kind`, return `'v5p'` or `'v6e'`). Raise on unknown. Add a test that asserts the detection returns one of the expected values when run on each TPU.

**Risk: Pallas API instability between JAX versions.**
Mitigation: pin a tested JAX version range in `pyproject.toml`. The fallback to pure-JAX is removed but a pinned dependency is the new mitigation. JAX upgrades become a deliberate event with a test-suite gate.

**Risk: long migration window leaves the tree half-converted.**
Mitigation: each optimizer is migrated end-to-end in a small commit series (specialized variant for sm_90, then sm_80, then sm_100, then gfx942, then cross-arch test, then delete the corresponding generic file). The generic kernel for that optimizer is not deleted until its specialized variants pass tests on every supported arch. The build should never break.

**Risk: deleting `jit/` removes runtime hooks that downstream code imports.**
Mitigation: `grep -r "from grokking_optimizers.jit"` and `grep -r "from grokking_optimizers import jit"` to find all import sites. SG2's Python optimizer wraps JIT instantiation in try/except; migration removes the try/except and the import.

**Risk: `dispatch.py` refactor breaks `__init__.py` re-exports.**
Mitigation: `__init__.py` imports a known surface from `dispatch.py`. The refactor preserves `get_gpu_arch`, `get_gpu_vendor`, `get_backend`, `get_warp_size`, and the feature predicates; only tier helpers (`get_arch_tier`, `get_amd_tier`) and fallback-chain helpers go away. Verify with the test suite.

**Risk: forgetting to update `README.md` and `REFRESH.md` to reflect the new arch policy.**
Mitigation: documentation updates are part of step 8 (build system) before deletions in step 9. The CI job that verifies docs are not stale is added at the same time.

## 8. Open questions

### Q1. `grokking_optimizers/jit_kernels.py`

This file lives at the top of `grokking_optimizers/`, alongside `jit/` (the directory). They appear to be different: `jit/` is the runtime specializer system, `jit_kernels.py` is something else (possibly torch.jit-style tracing for specific paths, or shape-specialized launcher caching). Should this file be:
- (a) deleted along with `jit/`, or
- (b) kept and refactored to use `tuned_configs.h`?

Default if unspecified: keep until I confirm what it does.

### Q2. CUTLASS dependency mode

Three options for getting CUTLASS into the build:
- (a) **Git submodule** at `third_party/cutlass`. Pros: pinned exact commit, easy to reproduce. Cons: clone-time bandwidth.
- (b) **Vendored snapshot** under `third_party/cutlass/` (committed to repo). Pros: works offline, no submodule init. Cons: large repo footprint (~50 MB of headers).
- (c) **System dependency** via `find_package(NvidiaCutlass)` or pip-installable `cutlass-cpp-headers`. Pros: lightweight repo. Cons: build env complexity, version drift.

Default if unspecified: (a) submodule pinned to a known-good CUTLASS release tag.

### Q3. CUTLASS on AMD?

CUTLASS proper is NVIDIA-only. AMD's equivalent is Composable Kernel (CK). Two options for the gfx942 BF16 MFMA path on the SG2 projections:
- (a) Keep rocBLAS BF16 GEMM as-is on gfx942 (existing path; works).
- (b) Adopt Composable Kernel for gfx942 BF16 MFMA + softplus epilogue.

Default if unspecified: (a). The migration spec says "CUTLASS for SG2 projections and Muon NS." On gfx942, rocBLAS is the equivalent — keeping it is consistent with the spec's intent without dragging in a second kernel library.

### Q4. CUTLASS on sm_80?

The spec says CUTLASS for SG2 projections and Muon NS. For sm_80 specifically, two options:
- (a) **CUTLASS sm_80 path with TF32 tensor cores**. Pros: arch consistency. Cons: more sm_80 work for marginal gain over the existing cuBLAS path.
- (b) **Keep cuBLAS+cp.async on sm_80**, only adopt CUTLASS for sm_90+.

Default if unspecified: (a) — full CUTLASS adoption per the goals statement. We can stage it: sm_90 first, then sm_80 in a follow-up commit, with the sm_80 path temporarily using cuBLAS until the CUTLASS variant lands.

### Q5. `benchmarks/autotune.py` disposition

This is the existing per-GPU profiling utility. The new `autotune/tune.py` is a build-time tuning script. Two options:
- (a) Delete `benchmarks/autotune.py` (subsumed by `autotune/tune.py`).
- (b) Keep `benchmarks/autotune.py` as a runtime "what-is-my-GPU + sanity check" tool, separate from build-time tuning.

Default if unspecified: (a) — simpler. The runtime "what is my GPU" function lives in `dispatch.py`.

### Q6. `tuned_configs.h` staleness policy

Two options:
- (a) **Hard fail** the build if `tuned_configs.h` is missing or older than `autotune/grids.py`.
- (b) **Soft warn**: build proceeds with default configs, prints a recommendation to re-run autotune.

Default if unspecified: (b) for developer ergonomics, with a CI job that hard-fails when the committed file is inconsistent with the grid.

### Q7. `csrc/cpu/generated/` — drop entirely or consolidate?

The 21 generated CPU files include Q3/Q4 quantized state variants for several optimizers. Currently they exist because the codegen path produces them. Two options:
- (a) **Drop entirely**. Hand-written `all_optimizers_cpu.cpp` becomes the only CPU source.
- (b) **Consolidate**: hand-write equivalent functionality (symmetric INT8 / INT4 / MXFP4 quant state for the optimizers that have it on GPU) into the existing `all_optimizers_cpu.cpp`.

Default if unspecified: (b) for the optimizers that genuinely have a quantized GPU state (GrokAdamW Q3); (a) for everything else (drop unused variants).

### Q8. Autotune output format

Three options for `tuned_configs.h`:
- (a) `static constexpr` tables of `LaunchConfig` structs keyed by enum (kernel, arch, shape bucket).
- (b) X-macros: `#define KERNEL_LIST(X) X(grokadamw, sm_90, BLOCK=128, …)` etc.
- (c) Lookup function generated by autotune (`get_launch_config(kernel_id, arch_id, n)` returns a struct).

Default if unspecified: (a) for compile-time visibility; reuses the same shape buckets as `autotune/grids.py`.

### Q9. SG2 Python fallback — keep, demote, or delete?

The spec says "no fallbacks." But `_python_fallback.py` is the SG2 reference implementation used by tests for correctness checking. It is also the only working code path on a CPU host without a CUDA build. Options:
- (a) Keep as-is. It is opt-in (set `SUPERGROK2_PYTHON_FALLBACK=1`) and never silent-fallback.
- (b) Demote to `tests/_sg2_reference.py` (test-only, not user-facing).
- (c) Delete; tests use the CPU C++ extension instead.

Default if unspecified: (a) — used by `tests/test_supergrok2.py` as a reference; demoting is fine but invasive, deleting loses the reference.

### Q10. Branch / commit cadence

Each optimizer migration is its own commit pair: (i) add specialized variants + bindings + tests, (ii) delete generic. Or interleave aggressively: run on each commit. Two options:
- (a) **One PR per optimizer**, ten total. Long branch life, lots of context per review.
- (b) **All in one branch** (`claude/custom-optimizer-analysis-HFYhg`) with one commit per logical step. Easier to revert individual steps.

Default if unspecified: (b) — already on the branch the user wants to develop on.

### Q11. CI matrix

The cross-arch agreement test requires running all four arches. Two options:
- (a) **One CI job per arch**, four jobs total, parallel.
- (b) **Single job per push, host-arch-only**, full matrix runs nightly.

Default if unspecified: (a) for fast feedback, with an opt-out flag for ad-hoc runs.

### Q12. `csrc/quantization/` — collapse into kernels tree?

`quantization_kernels.cu` is currently a single file under `csrc/quantization/`. The migration splits it per arch. Two options:
- (a) **Move per-arch into `csrc/kernels/{cuda,hip}/<arch>/quantization_<arch>.cu`** and delete `csrc/quantization/`.
- (b) **Keep `csrc/quantization/` as its own tree** with arch subdirs.

Default if unspecified: (a) — consistent with the rest of the kernels tree.

## 9. Execution plan

Steps follow the order specified by the user in the original prompt. Each step is a separate set of commits. Step 1 is this document.

### Step 1 — Plan (this commit)

Write `REFACTOR_PLAN.md`, commit, stop. Wait for review.

### Step 2 — GrokAdamW sm_90 (first specialized kernel)

Concrete commits inside this step:
1. Create directory tree: `csrc/kernels/cuda/sm_90/`, `csrc/bindings/`, `csrc/common/tuned_configs.h` (initially empty stub with default configs).
2. Write `csrc/kernels/cuda/sm_90/grokadamw_sm90.cu` — fully specialized Hopper version. cp.async + DSMT-aware shared memory layout. `__launch_bounds__` from stub `tuned_configs.h`. Float4 vectorized fast path retained.
3. Write `csrc/bindings/grokadamw.cpp` — pybind11 entry point. Calls the sm_90 kernel after asserting `get_sm_arch() == 90`.
4. Wire it into `csrc/bindings/module.cpp` (or a temporary parallel module while the migration is in flight).
5. Update `setup.py` to compile the new file alongside the existing extension. Both grokadamw paths exist temporarily.
6. Add a smoke test: `tests/test_grokadamw_sm90.py` runs the optimizer for a few steps and compares against the existing path on the same hardware.
7. Run `benchmarks/benchmark_supergrok2.py --optimizer grokadamw` on the hardware to confirm parity or improvement vs the generic kernel.
8. Commit each of the above as a separate commit with a clear message.

### Step 3 — GrokAdamW sm_80, sm_100, gfx942 + cross-arch test

Concrete commits:
1. Port `grokadamw_sm90.cu` to `grokadamw_sm80.cu`. Swap cp.async double-buffering details where they diverge; keep the math identical.
2. Port to `grokadamw_sm100.cu`. Swap to TMA prefetch where applicable.
3. Port to `grokadamw_gfx942.hip.cpp`. Use BF16 MFMA where it helps (note: GrokAdamW has no GEMM, so this mostly matters for the optional Q3 INT8/BF16 state path).
4. Update `csrc/bindings/grokadamw.cpp` to dispatch to the right specialized kernel based on `get_sm_arch()`. Raise on anything else.
5. Write `tests/test_cross_arch_agreement.py` with a `test_grokadamw_cross_arch` case. Skips arches not present on the host.
6. Delete `csrc/cuda/generic/grokadamw_kernels.cu`. Remove from `setup.py` source list.
7. Run the full test suite. Fix anything that breaks. Commit.

### Steps 4 — Migrate the rest of the optimizers

Order, easiest to hardest:

- **Step 4a**: Lion (single-state momentum, no MLP, no GEMM)
- **Step 4b**: Grokfast (EMA + Adam, no MLP, no GEMM)
- **Step 4c**: Prodigy (Adam + global reduction for `d_lr`)
- **Step 4d**: NeuralGrok (small MLP amplifier)
- **Step 4e**: LookSAM (Adam + periodic SAM)
- **Step 4f**: SuperGrok v1.5 (MetaNet MLP + SAM + bilevel)
- **Step 4g**: SuperGrok v1.1 (MetaNet MLP + cosine gate + bilevel)
- **Step 4h**: Muon (cuBLAS NS GEMMs → CUTLASS for sm_90, sm_100)
- **Step 4i**: SuperGrok v2 (Mamba+PEER+GRU+expert, projection cuBLAS → CUTLASS for sm_90, sm_100; bilevel checkpointing kept)

Each follows the same template as steps 2–3: sm_90 first, then sm_80/100/gfx942, then cross-arch test, then delete the corresponding generic file.

### Step 5 — CUTLASS migration (overlaps with 4h, 4i)

When migrating Muon's sm_90 kernel: introduce CUTLASS for the X·Xᵀ·X NS GEMM pattern. Stage the work:
1. Add CUTLASS as third-party dependency (per Q2 default: git submodule).
2. Update `setup.py` include path.
3. Write a CUTLASS-backed Muon NS GEMM in `csrc/kernels/cuda/sm_90/muon_sm90.cu`.
4. Compare against existing cuBLAS path; assert numerical agreement; benchmark.

When migrating SG2 forward sm_90:
1. Replace `bilevel_precompute_gemm`'s 5 cuBLAS matmuls with 5 CUTLASS GEMMs.
2. Implement a fused softplus+bias epilogue for the dt projection GEMM. Verify against existing `softplus_bias_kernel` output bit-for-bit (within tolerance).
3. Other 4 projection GEMMs use stock CUTLASS epilogues.
4. Document the choice for sm_80 (per Q4 default: stage CUTLASS sm_80 in a follow-up; ship sm_80 with cuBLAS+cp.async first).

### Step 6 — TPU dispatch

1. Create `csrc/kernels/tpu/v5p/` and `csrc/kernels/tpu/v6e/`.
2. Split `supergrok2_jax_tpu/pallas_kernels.py` into the per-version files listed in § 5.
3. Refactor `supergrok2_jax_tpu/sharding.py` (or a new `supergrok2_jax_tpu/tpu_dispatch.py`) to call `detect_tpu_version()` and import from the right kernel directory.
4. Update `tests/test_supergrok2_jax.py` to assert TPU dispatch picks the correct version.
5. Delete `supergrok2_jax_tpu/pallas_kernels.py`.

### Step 7 — Autotune

1. Create `autotune/` directory with `tune.py`, `grids.py`, `runner.py`, `cutlass_profile.py`, `README.md`.
2. Implement the CLI: `python autotune/tune.py --kernel grokadamw --arch sm_90 --output csrc/common/tuned_configs.h`.
3. Run on each available arch on hardware (user runs).
4. Commit `tuned_configs.h` to repo.
5. Verify build picks up the new launch bounds and the autotune-tuned kernels do not regress.

### Step 8 — Build system

1. Update `setup.py` source list: walk `csrc/kernels/{cuda,hip,cpu}/` and `csrc/bindings/`, drop `csrc/cuda/generic/`, drop `csrc/cuda/generated/`, drop `csrc/cpu/generated/`.
2. Update `-gencode` and `--offload-arch` to the supported arches only.
3. Add CUTLASS include path.
4. Add `tuned_configs.h` include path.
5. Build error if host arch is unsupported and `FORCE_CUDA` is not set.
6. Update `README.md` "Hardware support" section.

### Step 9 — Deletions

In sequence:
1. `csrc/cuda/generic/` (now empty after step 4 migrated each file).
2. `csrc/cuda/generated/` (codegen output; obsolete after step 4).
3. `csrc/cuda/sm_75/`, `sm_86/`, `sm_89/` (empty unsupported arches).
4. `csrc/cpu/generated/` (per Q7 default: drop entirely after consolidating into `all_optimizers_cpu.cpp`).
5. `csrc/cpu/` (after files moved to `csrc/kernels/cpu/`).
6. `csrc/hip/cdna2/`, `csrc/hip/cdna4/` (unsupported arches).
7. `csrc/hip/cdna3/` (after files moved to `csrc/kernels/hip/gfx942/`).
8. `csrc/hip/` (after subdirs deleted).
9. `csrc/common/{ops.h,ops.cpp,dispatch.h}` (replaced by `csrc/bindings/`).
10. `grokking_optimizers/jit/` (entire directory).
11. `codegen/` (entire directory).
12. `benchmarks/autotune.py` (per Q5 default).

Each in its own commit with a clear "Delete X (replaced by Y)" message.

### Step 10 — Test suite

1. Run the full test suite. Address anything that fails.
2. Update `tests/test_all_tiers.py` → `tests/test_all_arches.py` (drop generic-tier rows, add gfx942).
3. Update `tests/test_amd_hip.py` to test gfx942 only.
4. Update `tests/test_cpu_fallback.py` framing.
5. Add the four-row matrix to CI.
6. Tag the commit so the migration is bisectable.

---

## 10. Arch matrix expansion (planned follow-on)

Triggered after the original 4-arch refactor (sm_80, sm_90, sm_100, gfx942) is fully complete and tested. Expands the supported set to eight GPU arches plus two TPU versions.

### 10.1 Expanded supported set

NVIDIA:
- `sm_80` (Ampere) — A100, A30 (existing)
- `sm_89` (Ada) — RTX 40-series, L40, L40S (NEW)
- `sm_90` (Hopper) — H100, H200 (existing)
- `sm_100` (Blackwell datacenter) — B100, B200, GB200 (existing)
- `sm_103` (Blackwell Ultra) — B300, GB300 NVL72 (NEW)
- `sm_120` (Blackwell consumer) — RTX 50-series, RTX PRO 6000 Blackwell (NEW)

AMD:
- `gfx942` (CDNA3) — MI300X, MI300A (existing)
- `gfx950` (CDNA4) — MI350X, MI355X (NEW)

TPU (JAX path, unchanged):
- `v5p`, `v6e`

Still unsupported (build fails or `dispatch.get_gpu_arch()` raises):
- V100, T4 (sm_70/75)
- Pre-Ada consumer (sm_86 — Ampere RTX 30-series)
- MI100 (gfx908), MI200 (gfx90a), and explicitly NOT gfx950 prior to its first-class promotion in this expansion
- AMD RDNA cards
- TPU v3, v4, v5e

### 10.2 Per-arch specialization notes

**sm_89 (Ada Lovelace).** 4th-gen tensor cores with FP8 E4M3/E5M2. No TMA, no thread block clusters, no DSMT. Closest baseline is sm_90 minus Hopper-specific features. Strategy: port from `sm_90` and strip TMA / DSMT / warp-specialization paths. FP8 GEMMs go through cuBLASLt or the CUTLASS sm_89 path. `cp.async` carries over from sm_80, so the Ampere prefetch pattern stays. No CTA cluster intrinsics.

**sm_103 (Blackwell Ultra).** `compute_100f` family, binary-compatible with sm_100 but with ~50% more NVFP4 compute and an accelerated softmax. Strategy: port the optimizer per-element kernels from `sm_100` unchanged (memory-bound, no win from extra tensor core throughput). Specialize SG2 projection GEMMs and Muon Newton-Schulz GEMMs via the CUTLASS `sm_103a` target to exploit native NVFP4. Custom NVFP4 epilogue paths feed into `tuned_configs.h` from `autotune/cutlass_profile.py`.

**sm_120 (Blackwell consumer).** 128 KB shared memory per SM (vs 228 KB on sm_100). No DSMT. Different tensor core configuration than datacenter Blackwell. Strategy: port from `sm_100` with reduced shared-memory budgets in the tile-resident expert-weight kernels and the SG2 fused-elem kernel; switch to consumer tensor core paths for FP8 / FP16 GEMMs. CUTLASS `sm_120a` target for GEMMs.

**gfx950 (CDNA4).** Native FP4 expert MFMA (`mfma_f32_32x32x8_fp4`), native FP6 E3M2 state, 2:4 sparsity. Earlier work in `csrc/hip/cdna4/cdna4_kernels.hip.cpp` (deleted in commit `8c2280d` but recoverable from git history) provides a starting point. Strategy: port the 17 baselines from `gfx942`, then promote the recovered FP4/FP6/2:4 paths to first-class status under `csrc/kernels/hip/gfx950/`. FP4 expert weights and FP6 optimizer state become the optimized path on this arch (cf. spec §3 quantization).

### 10.3 Files to add

Mirror the existing `csrc/kernels/<backend>/<arch>/` structure for each new arch. Per-arch counts match the existing sm_90 column (17 wrapped baselines per optimizer):

- `csrc/kernels/cuda/sm_89/<optimizer>_sm89.cu` × 17, plus CUTLASS sm_89 SG2/Muon GEMMs
- `csrc/kernels/cuda/sm_103/<optimizer>_sm103.cu` × 17, plus CUTLASS sm_103a NVFP4 SG2/Muon GEMMs
- `csrc/kernels/cuda/sm_120/<optimizer>_sm120.cu` × 17, plus CUTLASS sm_120a SG2/Muon GEMMs
- `csrc/kernels/hip/gfx950/<optimizer>_gfx950.hip.cpp` × 17, plus FP4 expert + FP6 state specializations

### 10.4 Files to update

- `csrc/bindings/dispatch.cpp` — add cases for 89, 103, 120, 950 in the supported-arch switch; `detect_arch_from_device()` recognizes each SM number explicitly; raise message lists the new supported set
- `csrc/bindings/_dispatch_macro.h` — extend `SG_DISPATCH` switch to all 8 arches
- `csrc/bindings/bindings.h` — anchor `namespace sm89`, `sm103`, `sm120`, `gfx950`
- `csrc/bindings/<optimizer>.cpp` (every per-optimizer file) — declare per-arch launchers in the new namespaces; `SG_DISPATCH` macro picks up the new cases automatically once `_dispatch_macro.h` is updated
- `csrc/common/tuned_configs.h` — extend `ArchId` enum with `ARCH_SM89=4, ARCH_SM103=5, ARCH_SM120=6, ARCH_GFX950=7`; bump `NUM_ARCHES` from 4 to 8; widen the per-kernel tables from `[4][buckets]` to `[8][buckets]`
- `setup.py` — append `-gencode arch=compute_{89,103,120},code=sm_{89,103,120}` to the nvcc flags; append `--offload-arch=gfx950` to the hipcc flags
- `autotune/grids.py` — per-kernel grids inherit the existing axes for sm_89/103/120; add NVFP4-specific entries for sm_103 (`tile_shape × split_K` for the SG2 GEMMs); add FP4/FP6 entries for gfx950 (`fp4_expert_tile`, `fp6_state_block_size`)
- `autotune/cutlass_profile.py` — add sm_89, sm_103a, sm_120a as profiler targets; sm_103a needs the NVFP4 epilogue probe; sm_120a uses the consumer tensor core profile
- `grokking_optimizers/dispatch.py` — `SUPPORTED_ARCHES = (80, 89, 90, 100, 103, 120, 942, 950)`; `get_gpu_arch()` recognizes sm_89/103/120 and gfx950 explicitly; `get_arch_label()` table grows to 8 entries
- `tests/test_cross_arch_agreement.py` — `SUPPORTED` tuple grows to all 8; per-optimizer harnesses iterate them
- `tests/test_amd_hip.py` — add a gfx950 detection test alongside gfx942; legacy reject list (gfx908/gfx90a) stays
- `tests/test_all_arches.py` — `ARCHES` list grows from 4 to 8 rows; the FORCE_ARCH probe gates each per the `_arch_available` pattern
- `REFRESH.md §0` — supported set, fallback chain, layout table, and migration commit series all updated

### 10.5 Order of operations

Each step is its own commit. Per-optimizer ports group into commit batches per arch.

1. Show the updated REFACTOR_PLAN.md (this section). **Stop here for approval before proceeding.**
2. Port each optimizer's `sm_90` kernel to `sm_89` (closest base). Strip TMA, DSMT, warp specialization. One commit per optimizer.
3. Port each optimizer's `sm_100` kernel to `sm_103` and `sm_120`. For `sm_120`, additionally reduce shared-memory budgets to fit 128 KB SM.
4. Port each optimizer's `gfx942` kernel to `gfx950`, then add the FP4/FP6 specializations (recovered from `csrc/hip/cdna4/cdna4_kernels.hip.cpp` git history).
5. Extend `tests/test_cross_arch_agreement.py` to all 8 arches. Run on hardware as available.
6. Migrate CUTLASS GEMMs (SG2 projections, Muon NS) to sm_89, sm_103a, sm_120a targets.
7. Run autotune on each new arch on hardware; commit `tuned_configs.h` updates.

### 10.6 Constraints (carried over from the original refactor)

- One file per Write/Edit, ~150 lines max, commit per logical step. No codegen / no Jinja templates.
- The original 4-arch refactor must be fully merged and tested before this expansion starts. Concretely: all bindings TODOs filled in (`csrc/bindings/supergrok2.cpp`, secondary `multi_tensor.cpp` / `moe.cpp` launchers), build verified on at least one arch, pytest passes the four-row arch matrix.
- Affine2x2 PTX composition (`csrc/common/ptx_intrinsics.cuh`) remains untouched.
- Blelloch parallel scan algorithm itself is untouched; tile sizes and prefetch may change.
- Canonical reduction pattern (warp shuffle + per-warp atomicAdd) is untouched.
- Math is identical across all 8 arches; arch-specific changes are limited to primitives (cp.async / TMA / MFMA / FP4/FP6/FP8 paths, shared memory layout, tensor core configuration).
- Every per-arch kernel pulls `__launch_bounds__` from `csrc/common/tuned_configs.h`.
- The cross-arch numerical agreement test (`tests/test_cross_arch_agreement.py`) is mandatory and gates every arch port.

### 10.7 Final coverage target

After expansion completes, the project supports:

- A100, A30 (sm_80)
- RTX 40-series, L40, L40S (sm_89)
- H100, H200 (sm_90)
- B100, B200, GB200 (sm_100)
- B300, GB300 NVL72 (sm_103)
- RTX 50-series, RTX PRO 6000 Blackwell (sm_120)
- MI300X, MI300A (gfx942)
- MI350X, MI355X (gfx950)
- TPU v5p, v6e

Permanently unsupported: V100, T4, RTX 20/30-series, MI100, MI200, RDNA, TPU v3/v4/v5e.

---

## Stop here — review before execution

Nothing destructive happens until this plan is approved. After approval:
- Step 1 (this) is done.
- Step 2 of the original refactor begins with `csrc/kernels/cuda/sm_90/grokadamw_sm90.cu`.
- The §10 expansion does NOT begin until the original 4-arch refactor is complete (per §10.6 gating).
- I will pause and ask after each step for confirmation before continuing.
