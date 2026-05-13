# Refactor Audit — Phase 1

Pre-refactor inventory of every source file in the repository, its target
location in the new architecture, and any stub/placeholder status discovered.

This audit is read-only: no files have been moved or deleted yet.

---

## Executive summary

The codebase has the right *bones* — three vendor backends, eleven optimizer
algorithms, three model definitions, and a clean Python frontend — but the
on-disk layout overstates completion. The biggest discoveries:

1. **The `csrc/device/` tree is largely placeholder.** Most files contain
   `// TODO: Port full implementation from kernel` markers in function bodies.
   The real per-arch kernel logic lives in `csrc/kernels/cuda/sm_90/*.cu/cuh`
   and `csrc/kernels/hip/gfx942/*.hip.cpp/h`, not in `csrc/device/`.
2. **All 99 fused TUs are stubs.** Every file in `csrc/fused/<arch>/` is a
   one-line include + TODO comment. None instantiate a real fused kernel.
3. **GFX942 SuperGrok2 is honestly stubbed.** The fwd, bwd, and warp-spec
   launchers throw `std::runtime_error` with a clear message. The build matrix
   should reflect this (⛔ not ✅).
4. **Six device model templates are pure TODOs.** All
   `csrc/device/models/<arch>/{transformer,vit,mamba}_*.cuh` files contain
   empty function bodies with TODO comments. Models are *actually* implemented
   in `csrc/kernels/cuda/sm_90/models/*.cuh` and the gfx942 delegation wrappers.
5. **TPU v5p device files are real.** Both
   `csrc/device/models/tpu_v5p/*.py` and
   `csrc/device/optimizers/tpu_v5p/*.py` are thin re-exports from the actual
   working Pallas/JAX kernels in `csrc/kernels/tpu/_pallas_models.py` and
   `_pallas_kernels.py`.
6. **The Python frontend already follows the target pattern.** Every optimizer
   is a `torch.optim.Optimizer` subclass that calls into `_ops.<fn>_step`.
   The dispatch chain (Python class → C++ binding → SG_DISPATCH macro →
   per-arch launcher → kernel) is already in place; the refactor will
   tighten it, not invent it.

## Preconditions and deviations from prompt

The prompt asks to "preserve all autotune config infrastructure
(`tuned_configs.h`, `autotune/tune.py`)" and references `tests/` — both
`autotune/` and `tests/` were deleted in the prior turn at the user's
explicit request. `tuned_configs.h` still exists at `csrc/common/`. The
refactor proceeds without re-creating `autotune/` or `tests/`. This will
be noted in REFACTOR_NOTES.md when complete.

---

## Status legend

For each file:

- **REAL** — has actual logic, will be migrated content-preserving
- **STUB** — placeholder with TODO comments, will be deleted (logic lives elsewhere)
- **DELEGATION** — re-exports from a real implementation file
- **THROWS** — runtime-throwing stub kept for link compatibility
- **GENERATED** — produced by autotune (none currently)

For each row, the **Target** column is where it lives after the refactor.

---

## Top-level

| Path | Status | Target |
|------|--------|--------|
| `README.md` | REAL | `README.md` (rewrite in Phase 11) |
| `grokking_race_v2.py` | REAL | unchanged |
| `setup.py` | REAL | `setup.py` (update source globs in Phase 8) |
| `build.sh` | REAL | unchanged |
| `pyproject.toml` | REAL | unchanged |
| `.gitignore`, `.gitmodules` | REAL | unchanged |
| `third_party/.gitkeep` | REAL | unchanged |

---

## `csrc/common/` — shared headers

All real, no stubs. Stays in place.

| File | Status | Target |
|------|--------|--------|
| `platform.h` | REAL | `csrc/common/platform.h` |
| `types.h` | REAL | `csrc/common/types.h` (Affine2x2 extracted in Phase 4) |
| `utils.cuh` | REAL | `csrc/common/utils.cuh` |
| `ptx_intrinsics.cuh` | REAL | `csrc/common/ptx_intrinsics.cuh` |
| `tuned_configs.h` | REAL | `csrc/common/tuned_configs.h` |
| `quantization.h` | REAL | `csrc/common/quantization.h` |
| `arch_tier.h` | REAL | `csrc/common/arch_tier.h` |
| `fp4_helpers.hip.h` | REAL | `csrc/common/fp4_helpers.hip.h` (used by future gfx950) |

---

## `csrc/device/` — DELETE ENTIRELY after migration

### `csrc/device/models/sm_90/` (3 files — all STUBS)

| File | Status | Giveaway | Target |
|------|--------|----------|--------|
| `transformer_sm90.cuh` | STUB | `// TODO: Fused forward+backward device template for transformer on sm_90.` | DELETE |
| `vit_sm90.cuh` | STUB | `// TODO: Fused forward+backward device template for vit on sm_90.` | DELETE |
| `mamba_sm90.cuh` | STUB | `// TODO: Fused forward+backward device template for mamba on sm_90.` | DELETE |

Real model implementations remain at `csrc/kernels/cuda/sm_90/models/*.cuh`
and will move to `csrc/backends/cuda/sm_90/models/` in Phase 5.

### `csrc/device/models/gfx942/` (3 files — all STUBS)

| File | Status | Giveaway | Target |
|------|--------|----------|--------|
| `transformer_gfx942.hip.cuh` | STUB | `// TODO: Fused forward+backward device template for transformer on gfx942.` | DELETE |
| `vit_gfx942.hip.cuh` | STUB | `// TODO: Fused forward+backward device template for vit on gfx942.` | DELETE |
| `mamba_gfx942.hip.cuh` | STUB | `// TODO: Fused forward+backward device template for mamba on gfx942.` | DELETE |

### `csrc/device/models/tpu_v5p/` (3 files — DELEGATION)

| File | Status | Target |
|------|--------|--------|
| `transformer_tpu_v5p.py` | DELEGATION → `csrc/kernels/tpu/_pallas_models.py` | content folds into `csrc/backends/pallas/models/decoder.py` |
| `vit_tpu_v5p.py` | DELEGATION → `csrc/kernels/tpu/_pallas_models.py` | content folds into `csrc/backends/pallas/models/vit.py` |
| `mamba_tpu_v5p.py` | DELEGATION → `csrc/kernels/tpu/_pallas_models.py` | content folds into `csrc/backends/pallas/models/mamba.py` |

### `csrc/device/optimizers/sm_90/` (12 files)

| File | Status | Notes | Target |
|------|--------|-------|--------|
| `adam_sm90.cuh` | STUB | 4 functions with `// TODO: Port full implementation` | DELETE (math goes to `csrc/algorithms/adamw.h` + `moe_adam.h`) |
| `grokadamw_sm90.cuh` | STUB | TODO markers | DELETE → `csrc/algorithms/grokadamw.h` |
| `grokfast_sm90.cuh` | STUB | 2 TODO functions | DELETE → `csrc/algorithms/grokfast.h` |
| `lion_sm90.cuh` | REAL | `lion_step`, `lion_step_vec4` fully implemented | EXTRACT math → `csrc/algorithms/lion.h`, then DELETE |
| `looksam_sm90.cuh` | STUB | 5 TODO functions | DELETE → `csrc/algorithms/looksam.h` |
| `moe_sm90.cuh` | STUB | 4 TODO functions | DELETE → `csrc/algorithms/moe_adam.h` |
| `muon_sm90.cuh` | STUB | TODO markers | DELETE → `csrc/algorithms/muon.h` |
| `neuralgrok_sm90.cuh` | STUB | 2 TODO functions | DELETE → `csrc/algorithms/neuralgrok.h` |
| `prodigy_sm90.cuh` | STUB | 2 TODO functions, commented reference | DELETE → `csrc/algorithms/prodigy.h` |
| `supergrok11_sm90.cuh` | STUB | 4 TODO functions | DELETE → `csrc/algorithms/supergrok11.h` |
| `supergrok15_sm90.cuh` | STUB | 4 TODO functions | DELETE → `csrc/algorithms/supergrok15.h` |
| `supergrok2_sm90.cuh` | REAL | `input_proj_sort`, `mamba3_scan_step`, scan_warp_specialized variants, `bilevel_precompute_timestep` all fully implemented | EXTRACT math → `csrc/algorithms/supergrok2.h`, then DELETE |

### `csrc/device/optimizers/gfx942/` (12 files)

Same shape as sm_90. The only real file is `supergrok2_gfx942.hip.cuh`
(identical algorithm math as sm_90 version). All others are stubs.

| File | Status | Target |
|------|--------|--------|
| `adam_gfx942.hip.cuh` | STUB | DELETE |
| `grokadamw_gfx942.hip.cuh` | STUB | DELETE |
| `grokfast_gfx942.hip.cuh` | STUB | DELETE |
| `lion_gfx942.hip.cuh` | STUB | DELETE |
| `looksam_gfx942.hip.cuh` | STUB | DELETE |
| `moe_gfx942.hip.cuh` | STUB | DELETE |
| `muon_gfx942.hip.cuh` | STUB | DELETE |
| `neuralgrok_gfx942.hip.cuh` | STUB | DELETE |
| `prodigy_gfx942.hip.cuh` | STUB | DELETE |
| `supergrok11_gfx942.hip.cuh` | STUB | DELETE |
| `supergrok15_gfx942.hip.cuh` | STUB | DELETE |
| `supergrok2_gfx942.hip.cuh` | REAL (mirror of sm_90) | math already extracted to `csrc/algorithms/supergrok2.h` |

### `csrc/device/optimizers/tpu_v5p/` (11 files — DELEGATION)

| File | Status | Target |
|------|--------|--------|
| `adam_tpu_v5p.py` | DELEGATION | content folds into `csrc/backends/pallas/launch_adamw.py` (or `moe_adam.py`) |
| `grokfast_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_grokfast.py` |
| `lion_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_lion.py` |
| `looksam_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_looksam.py` |
| `moe_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_moe_adam.py` |
| `muon_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_muon.py` |
| `neuralgrok_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_neuralgrok.py` |
| `prodigy_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_prodigy.py` |
| `supergrok11_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_supergrok11.py` |
| `supergrok15_tpu_v5p.py` | DELEGATION | → `csrc/backends/pallas/launch_supergrok15.py` |
| `supergrok2_tpu_v5p.py` | DELEGATION → `csrc/kernels/tpu/v5p` and `_pallas_kernels` | → `csrc/backends/pallas/launch_supergrok2.py` |

Note: there is no `grokadamw_tpu_v5p.py` in the current tree — GrokAdamW on
TPU is currently implemented via the JAX `simple_optimizers_jax.py`. Phase 5
will create a real `launch_grokadamw.py` calling that path.

---

## `csrc/kernels/cuda/` — sm_90 real kernels

### `csrc/kernels/cuda/_cutlass_gemm.cuh`

| Status | Target |
|--------|--------|
| REAL | `csrc/backends/cuda/sm_90/mma.cuh` (rename) |

CUTLASS wrappers and the fused softplus epilogue. Only used by Muon and SG2.

### `csrc/kernels/cuda/sm_90/` (11 optimizer .cu/.cuh pairs)

Pattern: each `.cuh` is the real kernel header (templates + launcher
declarations), each `.cu` is the explicit instantiation TU. These are the
authoritative implementations.

| `.cuh` file | Status | Algorithm math target | Launch glue target |
|-------------|--------|------------------------|---------------------|
| `adamw.cuh` | REAL (some templates marked TODO at line 184 but the inline-step math is real) | `csrc/algorithms/adamw.h` | `csrc/backends/cuda/sm_90/launch_adamw.cu` |
| `grokadamw.cuh` | REAL | `csrc/algorithms/grokadamw.h` | `csrc/backends/cuda/sm_90/launch_grokadamw.cu` |
| `grokfast.cuh` | REAL | `csrc/algorithms/grokfast.h` | `csrc/backends/cuda/sm_90/launch_grokfast.cu` |
| `lion.cuh` | REAL | `csrc/algorithms/lion.h` | `csrc/backends/cuda/sm_90/launch_lion.cu` |
| `looksam.cuh` | REAL | `csrc/algorithms/looksam.h` | `csrc/backends/cuda/sm_90/launch_looksam.cu` |
| `muon.cuh` | REAL (line 440 has FP8 stub guarded by `__has_include`) | `csrc/algorithms/muon.h` | `csrc/backends/cuda/sm_90/launch_muon.cu` |
| `neuralgrok.cuh` | REAL (line 215 has link-resolvable symbol stubs) | `csrc/algorithms/neuralgrok.h` | `csrc/backends/cuda/sm_90/launch_neuralgrok.cu` |
| `prodigy.cuh` | REAL | `csrc/algorithms/prodigy.h` | `csrc/backends/cuda/sm_90/launch_prodigy.cu` |
| `supergrok11.cuh` | REAL | `csrc/algorithms/supergrok11.h` | `csrc/backends/cuda/sm_90/launch_supergrok11.cu` |
| `supergrok15.cuh` | REAL | `csrc/algorithms/supergrok15.h` | `csrc/backends/cuda/sm_90/launch_supergrok15.cu` |
| `supergrok2_fwd.cuh` | REAL (placeholder array references on lines 452-453 in `.cu`) | merge → `csrc/algorithms/supergrok2.h` | merge → `csrc/backends/cuda/sm_90/launch_supergrok2.cu` |
| `supergrok2_bwd.cuh` | REAL | merge → `csrc/algorithms/supergrok2.h` | merge → `csrc/backends/cuda/sm_90/launch_supergrok2.cu` |
| `supergrok2_warp_specialized.cuh` | REAL | merge → `csrc/algorithms/supergrok2.h` | merge → `csrc/backends/cuda/sm_90/launch_supergrok2.cu` |

There is no `moe.cuh` in `csrc/kernels/cuda/sm_90/` — MoE/Adam is currently
served by the `multi_tensor` path in adamw.cuh + a binding-side dispatcher.
Phase 5 needs to add `csrc/backends/cuda/sm_90/launch_moe_adam.cu`.

### `csrc/kernels/cuda/sm_90/models/` (5 files)

| File | Status | Target |
|------|--------|--------|
| `attention.cuh` | REAL | `csrc/backends/cuda/sm_90/models/attention.cuh` |
| `decoder.cuh` | REAL | merge into `csrc/backends/cuda/sm_90/models/decoder.cu` |
| `decoder.cu` | REAL (instantiation TU) | `csrc/backends/cuda/sm_90/models/decoder.cu` |
| `vit.cuh` | REAL | merge into `csrc/backends/cuda/sm_90/models/vit.cu` |
| `vit.cu` | REAL (instantiation TU) | `csrc/backends/cuda/sm_90/models/vit.cu` |
| `mamba.cuh` | REAL | merge into `csrc/backends/cuda/sm_90/models/mamba.cu` |
| `mamba.cu` | REAL (instantiation TU) | `csrc/backends/cuda/sm_90/models/mamba.cu` |
| `mamba_scan_adapter.cuh` | REAL | `csrc/scan/mamba_scan_adapter.cuh` (RECLASSIFIED — used by both model and SG2 optimizer) |

Models will be rewritten as `models/decoder.h` (vendor-neutral definition) +
per-arch `models/decoder.cu` (real kernel TU). The current `.cuh`/`.cu` split
on sm_90 collapses into a single .cu containing both the kernel definitions
and the launcher.

---

## `csrc/kernels/hip/gfx942/` — HIP kernels

### `csrc/kernels/hip/gfx942/` (11 optimizers, `.hip.h`/`.hip.cpp` pairs)

Pattern: `.hip.h` declares launchers in `sg::gfx942` namespace; `.hip.cpp`
implements them via ATen tensor ops (because .hip.cpp goes through host
compiler, not hipcc — discovery from prior session). Most are REAL.

| File | Status | Notes |
|------|--------|-------|
| `_common.hip.h` | REAL (common includes) | → `csrc/backends/hip/gfx942/primitives.hpp` (consolidated) |
| `adamw.hip.h/cpp` | REAL | → `launch_adamw.hip.cpp` |
| `grokadamw.hip.h/cpp` | REAL | → `launch_grokadamw.hip.cpp` |
| `grokfast.hip.h/cpp` | REAL | → `launch_grokfast.hip.cpp` |
| `lion.hip.h/cpp` | REAL | → `launch_lion.hip.cpp` |
| `looksam.hip.h/cpp` | REAL | → `launch_looksam.hip.cpp` |
| `muon.hip.h/cpp` | REAL | → `launch_muon.hip.cpp` |
| `neuralgrok.hip.h/cpp` | REAL | → `launch_neuralgrok.hip.cpp` |
| `prodigy.hip.h/cpp` | REAL | → `launch_prodigy.hip.cpp` |
| `supergrok11.hip.h/cpp` | REAL | → `launch_supergrok11.hip.cpp` |
| `supergrok15.hip.h/cpp` | REAL | → `launch_supergrok15.hip.cpp` |
| `supergrok2_fwd.hip.h/cpp` | **THROWS** | merge → `launch_supergrok2.hip.cpp` (raises NotImplementedError) |
| `supergrok2_bwd.hip.h/cpp` | **THROWS** | merge → `launch_supergrok2.hip.cpp` (raises NotImplementedError) |
| `supergrok2_warp_specialized.hip.h/cpp` | **THROWS** (intentionally empty .cpp) | merge → `launch_supergrok2.hip.cpp` |

The MoE/Adam multi-tensor on gfx942 is served by the multi_tensor binding
+ ATen ops in the `adamw.hip.cpp` (`launch_fused_adamw_simple` handles
multi-tensor batches). Phase 10 will verify it actually runs on hardware
and decide its status flag.

### `csrc/kernels/hip/gfx942/models/` (5 files — DELEGATION)

| File | Status | Target |
|------|--------|--------|
| `attention.hip.h` | DELEGATION | `csrc/backends/hip/gfx942/models/attention.hip.h` (or merged into decoder/vit) |
| `decoder.hip.h/cpp` | DELEGATION → sm_90 via hipify | `csrc/backends/hip/gfx942/models/decoder.hip.cpp` |
| `vit.hip.h/cpp` | DELEGATION → sm_90 via hipify | `csrc/backends/hip/gfx942/models/vit.hip.cpp` |
| `mamba.hip.h/cpp` | DELEGATION → sm_90 via hipify | `csrc/backends/hip/gfx942/models/mamba.hip.cpp` |
| `mamba_scan_adapter.hip.h` | REAL (line 165 has `return hipErrorNotReady; // stub` — single error path) | `csrc/scan/mamba_scan_adapter.hip.h` (RECLASSIFIED) |

---

## `csrc/kernels/tpu/` — Pallas/JAX

| File | Status | Target |
|------|--------|--------|
| `__init__.py` | REAL | `csrc/backends/pallas/__init__.py` |
| `_pallas_kernels.py` | REAL (line 195: `Currently a stub — implement if profiling shows...`) | merge into `csrc/backends/pallas/primitives.py` + per-launch files |
| `_pallas_models.py` | REAL | split into `csrc/backends/pallas/models/{decoder,vit,mamba}.py` |
| `v5p/__init__.py` | REAL (re-exports) | `csrc/backends/pallas/v5p/__init__.py` |

---

## `csrc/fused/` — 99 stub TUs (KEEP STRUCTURE, UPDATE INCLUDES)

All files in `csrc/fused/{sm_90,gfx942,tpu_v5p}/` are stubs. The directory
structure stays (these are the 99 build targets the prompt mentions); only
the `#include` paths inside change in Phase 8.

### `csrc/fused/sm_90/` (33 files — all STUB)

Pattern: `fused_<model>_<optimizer>_sm90.cu`. Each file:
```c
#include "csrc/device/models/sm_90/<model>_sm90.cuh"     // OLD
#include "csrc/device/optimizers/sm_90/<optimizer>_sm90.cuh"  // OLD
// TODO: Instantiate fused forward-backward-update kernel
```

After Phase 8, these become:
```c
#include "csrc/models/<model>.h"
#include "csrc/algorithms/<optimizer>.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"
// TODO: Instantiate fused forward-backward-update kernel
```

The 33 files are: `{transformer, vit, mamba} × {adam, grokfast, lion,
looksam, moe, muon, neuralgrok, prodigy, supergrok11, supergrok15,
supergrok2}`. NOTE: `grokadamw` is not currently a row in fused/. Phase 8
will reconcile this (either add it or treat `adam` as the alias for it; the
audit shows it's `fused_<model>_adam_sm90.cu`, not `fused_<model>_grokadamw_sm90.cu`,
so the "11 optimizers" list in fused/ is: adam, grokfast, lion, looksam,
moe, muon, neuralgrok, prodigy, supergrok11, supergrok15, supergrok2 — only
11 because `adam` = `grokadamw`).

### `csrc/fused/gfx942/` (33 files — all STUB)

Same pattern with `.hip.cpp` extension.

### `csrc/fused/tpu_v5p/` (33 files — all STUB)

Same pattern with `.py` extension, raises `NotImplementedError`.

---

## `csrc/bindings/` — THICK dispatchers (KEEP, RENAME PATHS)

The bindings layer is real and does meaningful work (gradient filtering,
clipping, SAM norm, vector packing). It stays but consolidates.

| File | Status | Target |
|------|--------|--------|
| `_dispatch_macro.h` | REAL | merge into `csrc/bindings/dispatch.cpp` (or keep as private header) |
| `_helpers.h` | REAL | `csrc/bindings/helpers.h` (rename, no underscore) |
| `bindings.h` | REAL | merge into `csrc/bindings/dispatch.cpp` |
| `module.cpp` | REAL (pybind11 entry, ~450 LOC) | `csrc/bindings/bindings.cpp` (rename) |
| `dispatch.cpp` | REAL (`fused_step` throws for all 99 combos at line 114-122) | `csrc/bindings/dispatch.cpp` |
| `models_module.cpp` | REAL (model submodule registrar) | merge into `csrc/bindings/bindings.cpp` |
| `models_decoder.cpp`, `models_mamba.cpp`, `models_vit.cpp` | REAL | merge into `csrc/bindings/bindings.cpp` |
| `lion.cpp`, `muon.cpp`, `prodigy.cpp`, etc. (per-optimizer dispatchers) | REAL | merge into `csrc/bindings/bindings.cpp` |
| `multi_tensor.cpp` | REAL (multi-tensor AdamW path) | merge into `csrc/bindings/bindings.cpp` |
| `moe.cpp` | REAL | merge into `csrc/bindings/bindings.cpp` |
| `quantization.cpp` | REAL (FP8/INT8/INT4/MXFP4 quantizers) | KEEP as `csrc/bindings/quantization.cpp` |
| `distributed_scan.cpp` | REAL | KEEP as `csrc/bindings/distributed_scan.cpp` |
| `supergrok2.cpp`, `supergrok15.cpp`, `supergrok11.cpp`, etc. | REAL | merge into `csrc/bindings/bindings.cpp` |
| `grokadamw.cpp`, `grokfast.cpp`, `looksam.cpp`, `neuralgrok.cpp` | REAL | merge into `csrc/bindings/bindings.cpp` |

The per-optimizer dispatcher .cpp files each contain ~50-100 LOC of CPU-side
gradient filtering + `SG_DISPATCH(launch_<method>, ...)` calls. They could
either remain as individual files (current layout) or consolidate into one
`bindings.cpp`. The prompt's target uses one `bindings.cpp` + `dispatch.cpp` +
`helpers.h`, so we consolidate.

---

## `grokking_optimizers/` — Python frontend (RESTRUCTURE in Phase 9)

Current: 30 files, ~9,500 LOC total, with `supergrok2.py` alone at 2073 LOC.

### Core optimizers (11 — KEEP AS torch.optim.Optimizer subclasses)

| File | LOC | Target |
|------|-----|--------|
| `lion.py` | 115 | `grokking_optimizers/optimizers/lion.py` |
| `grokfast.py` | 152 | `grokking_optimizers/optimizers/grokfast.py` |
| `prodigy.py` | 157 | `grokking_optimizers/optimizers/prodigy.py` |
| `grokadamw.py` | 163 | `grokking_optimizers/optimizers/grokadamw.py` |
| `moe_deep.py` | 163 | `grokking_optimizers/optimizers/moe_adam.py` (rename) |
| `muon.py` | 225 | `grokking_optimizers/optimizers/muon.py` |
| `looksam.py` | 260 | `grokking_optimizers/optimizers/looksam.py` |
| `neuralgrok.py` | 270 | `grokking_optimizers/optimizers/neuralgrok.py` |
| `supergrok11.py` | 341 | `grokking_optimizers/optimizers/supergrok11.py` |
| `supergrok15.py` | 522 | `grokking_optimizers/optimizers/supergrok15.py` |
| `supergrok2.py` | 2073 | `grokking_optimizers/optimizers/supergrok2.py` (KEEP big, do not split) |

### Infrastructure (KEEP and refactor)

| File | LOC | Target |
|------|-----|--------|
| `__init__.py` | 109 | `grokking_optimizers/__init__.py` (clean re-exports) |
| `dispatch.py` | 229 | `grokking_optimizers/dispatch.py` |
| `fused_dispatch.py` | 41 | merge into `grokking_optimizers/dispatch.py` |
| `_ops_loader.py` | 30 | merge into `grokking_optimizers/dispatch.py` |
| `_adamw_helper.py` | 22 | merge into `grokking_optimizers/optimizers/grokadamw.py` |
| `_python_fallback.py` | 487 | `grokking_optimizers/fallback.py` (rename) |

### Auxiliary (DECISION REQUIRED — keep or remove?)

These exist in the repo but the public API surface (`__init__.py`) only
exports the 11 core optimizers plus a few helpers. Each adds maintenance
surface and may be partially or fully unused after the cleanup:

| File | LOC | Recommendation |
|------|-----|----------------|
| `async_supergrok2.py` | 539 | KEEP (referenced by `__init__.py`) |
| `cuda_graph_optimizer.py` | 163 | KEEP if exported |
| `distributed.py` | 171 | KEEP (distributed training support) |
| `distributed_scan.py` | 671 | KEEP (distributed scan kernels) |
| `gradient_compression.py` | 77 | KEEP if exported |
| `gradient_hook_optimizer.py` | 342 | KEEP if exported |
| `interleaved_states.py` | 45 | KEEP if exported |
| `mamba3_peer_metanet.py` | 572 | KEEP (used by SuperGrok2) |
| `overlap_distributed.py` | 103 | KEEP if exported |
| `partial_graph.py` | 64 | KEEP if exported |
| `pipelined_optimizer.py` | 96 | KEEP if exported |
| `quantization.py` | 636 | KEEP (referenced) |
| `sparse_gradients.py` | 33 | KEEP if exported |
| `torch_compile_integration.py` | 593 | KEEP if exported |

I will preserve all of these as-is unless `grokking_race_v2.py` is fully
checkable as not importing them. Phase 9 will move them under
`grokking_optimizers/extensions/` to keep the core directory uncluttered.

---

## `supergrok2_jax_tpu/` — JAX rewrite (UNTOUCHED except imports)

| File | Status | Target |
|------|--------|--------|
| `__init__.py` | REAL | unchanged |
| `bilevel.py` | REAL | unchanged |
| `bridge.py` | REAL | unchanged |
| `gru.py` | REAL | unchanged |
| `mamba3_peer_metanet_jax.py` | REAL | unchanged |
| `metanet_optimizers_jax.py` | REAL | unchanged |
| `pallas_kernels.py` | REAL | unchanged |
| `peer.py` | REAL | unchanged |
| `quantization_jax.py` | REAL | unchanged |
| `scan.py` | REAL | unchanged |
| `sharding.py` | REAL | unchanged |
| `simple_optimizers_jax.py` | REAL | unchanged |
| `supergrok2_jax.py` | REAL | unchanged |

Any import path that references the old `csrc/kernels/tpu/_pallas_*` will be
updated in Phase 8.

---

## File count after refactor — target

Per the prompt:

- 11 algorithm headers (`csrc/algorithms/*.h`)
- 3 model definitions (`csrc/models/*.h`)
- 8 common shared headers (`csrc/common/*`)
- 2 scan shared files (`csrc/scan/affine2x2.h`, `csrc/scan/mamba_scan_adapter.{cuh,hip.h}`)
- 3 × (primitives + 3 model TUs + 11 launch TUs) = 3 × 15 = 45 backend files
- 3 bindings files (`bindings.cpp`, `dispatch.cpp`, `helpers.h`)
- 99 fused TU stubs (unchanged in number)

That's roughly 11 + 3 + 8 + 2 + 45 + 3 = 72 source files + 99 fused TUs.
Plus the Python frontend (~30 files), JAX rewrite (~13 files), and top-level
build/docs.

The "50 source files compose into 99 fused build targets" framing in the
prompt counts the 11 algorithms + 3 models + 33 launch files + 3 primitives
= 50 vendor-meaningful files.

---

## Decisions to capture in REFACTOR_NOTES.md

1. `autotune/` and `tests/` were removed in the prior cleanup turn. The
   refactor proceeds without them. `tuned_configs.h` is preserved.
2. The `_python_fallback.py` file (487 LOC) becomes `fallback.py` —
   pure-python reference implementations remain functional.
3. Per-optimizer C++ dispatcher .cpp files consolidate into one
   `bindings.cpp`. This is a deliberate simplification matching the prompt's
   target layout.
4. The MoE/Adam optimizer is named "Adam multi-tensor" in some places and
   "MoE/Adam mt" in the build matrix. The new canonical name is `moe_adam`
   in C++ and `MoEAwareAdam` (or similar) in Python.
5. `csrc/fused/<arch>/` keeps all 99 stub files (the build-target count); we
   only update their includes. The prompt explicitly says "KEEP AS-IS" for
   the count.
6. Auxiliary Python optimizers (CUDA graph, gradient hooks, distributed,
   pipelined, etc.) move under `grokking_optimizers/extensions/` to keep the
   core optimizer surface clean. If they break, they will be left as-is and
   noted; they are not part of the 11 core optimizers.

---

## Phase ordering reaffirmed

The 12 phases in the prompt are the plan; this audit captures the pre-state
each phase will transform. Phase 2 begins next: algorithm extraction.
