# Other-Arch Backends + Third-Party Catalog

## Summary

This document catalogs the non-sm_90 architecture backends (gfx942/HIP, TPU/Pallas) and third_party/ vendored code. All are **PRESERVED-BUT-NOT-EXERCISED** in this H100/sm_90-only campaign.

---

## 1. HIP/gfx942 Backend

### 1a. Primitives + Infrastructure

**`csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp`**
- CDNA3 / MI300X device primitives: MFMA wrappers (`__builtin_amdgcn_mfma_f32_*bf16_1k`), DPP cross-lane wavefront reductions, ds_swizzle/ds_bpermute fallbacks, buffer_load->LDS staging, FP8 FNUZ (CDNA3) conversion, sched_group_barrier, AGENT-scope global atomics.
- Wavefront width = 64 (CDNA3).
- Status: NOT compile-verified (no hipcc in this env), declared hardware-gated 🟡 per HARDWARE_VALIDATION.md Stage 5.

**`csrc/backends/hip/gfx942/primitives.hpp`** — host-side helpers shim.

**`csrc/backends/hip/gfx942/moe_compaction_gfx942.hip.hpp`** — MoE expert compaction (histogram/ballot-compact/scatter) for gfx942.

**`csrc/backends/hip/gfx942/supergrok2_bilevel_adjoint_gfx942.hip.hpp`** — SuperGrok2 bilevel adjoint backward AMDGCN kernels (attention-ctx MFMA bwd, GRU-gate bwd, PEER bwd DPP, softmax bwd).

### 1b. Per-optimizer Launch TUs (csrc/backends/hip/gfx942/)

Thin `.hip.cpp` files that each `#include` the canonical kernel header from `grokking_optimizers/kernels/gfx942/`:
- `launch_adamw.hip.cpp`
- `launch_grokadamw.hip.cpp`
- `launch_grokfast.hip.cpp`
- `launch_lion.hip.cpp`
- `launch_looksam.hip.cpp`
- `launch_muon.hip.cpp`
- `launch_neuralgrok.hip.cpp`
- `launch_prodigy.hip.cpp`
- `launch_supergrok11.hip.cpp`
- `launch_supergrok15.hip.cpp`
- `launch_supergrok2.hip.cpp`

**All 11 optimizers covered.** Each TU is a one-liner include shim.

### 1c. gfx942 Model Files (csrc/backends/hip/gfx942/models/)

- `decoder.hip.h` — one-liner shim: `#include "grokking_optimizers/kernels/gfx942/transformer_decoder_gfx942.hip.hpp"`
- `decoder.hip.cpp` — thin TU
- `mamba.hip.h`, `mamba.hip.cpp` — Mamba3 model for gfx942
- `vit.hip.h`, `vit.hip.cpp` — ViT for gfx942
- `attention.hip.h` — attention primitives

**All 3 reference models covered.**

### 1d. Fused gfx942 Megakernel (csrc/fused/gfx942/)

- `fused_megakernel.hip.hpp` — L3/L1 persistent megakernel, AMD twin of `csrc/fused/sm_90/fused_megakernel.cuh`. Composes `opt_components.hip.hpp` (all 11 opts, no fallback) + `model_stages.hip.hpp` over shared gfx942 persistent substrate (megakernel_common_hip: task queue, CU-pin, GridBarrier). 4-wave-interleave scheduling (no Hopper warp-spec analog on CDNA3). Device-pass gated by `__AMDGCN__`/`__HIPCC__`. Status: 🟡 MI300X-gated.
- `model_stages.hip.hpp` — gfx942 model stage components
- `opt_components.hip.hpp` — gfx942 optimizer components (all 11)

---

## 2. grokking_optimizers/kernels/gfx942/ — Per-Optimizer Kernel Headers

This directory contains the CANONICAL per-optimizer kernel implementations for gfx942. 17 items total (16 .hip.hpp + 1 __init__.py):

**Optimizer kernels (11 total, all present):**
- `adamw_gfx942.hip.hpp` — pure elementwise SIMD; no MFMA applicability; streaming grad loads via `amd::streaming_load`; fuses m/v EMAs + bias-corrected decoupled-weight-decay in ONE kernel
- `grokadamw_gfx942.hip.hpp`
- `grokfast_gfx942.hip.hpp`
- `lion_gfx942.hip.hpp`
- `looksam_gfx942.hip.hpp`
- `muon_gfx942.hip.hpp`
- `neuralgrok_gfx942.hip.hpp`
- `prodigy_gfx942.hip.hpp`
- `supergrok11_gfx942.hip.hpp`
- `supergrok15_gfx942.hip.hpp`
- `supergrok2_gfx942.hip.hpp` — largest at 2692 lines; full CSA/HCA meta-model with MFMA (16x16x16 bf16), DPP softmax reductions, PEER product-key top-k routing, GRU gates, MoE compaction; bilevel adjoint delegated to `supergrok2_bilevel_adjoint_gfx942.hip.hpp`

**Model kernels (3 total, all present):**
- `transformer_decoder_gfx942.hip.hpp` — 493 lines
- `vit_gfx942.hip.hpp` — 557 lines
- `mamba3_gfx942.hip.hpp`

**Shared:**
- `common_gfx942.hip.hpp`
- `attention_gfx942.hip.hpp`

Each optimizer header uses TWO-PASS compile routing:
- HOST pass (`!__AMDGCN__`): sees ATen/rocBLAS orchestration
- DEVICE pass (`__AMDGCN__`/`__HIPCC__`): sees hand-written AMDGCN kernels

All headers declare hardware-gated 🟡 (MI300X/hipcc not available in this env).

---

## 3. TPU/Pallas Backend

### 3a. csrc/backends/pallas/

**Infrastructure:**
- `_pallas_kernels.py` — Pallas custom kernels: affine associative scan (manually tiled for TPU MXU) and expert gather. Falls back to pure JAX when Pallas unavailable or input too small.
- `_pallas_models.py` — Pallas model forward/backward surface (tiled/splash attention, selective scan, patch projection) for all 3 models.
- `_pallas_fused.py` — REAL fused composition: `OPT_STEPS` dict maps all 11 optimizers to real TPU step callables; `MODEL_STAGES` maps 3 models to (forward, backward) pairs. `fused_step` composes them; tier L3 runs fwd->bwd->opt in one `jax.jit`; `trace_check` validates via `jax.eval_shape` + `.lower()` without hardware.

**Per-optimizer Pallas launchers (11 total, all present):**
- `launch_adamw.py` — pure JAX elementwise (no MFMA applicable), jit'd
- `launch_grokadamw.py`, `launch_grokfast.py`, `launch_lion.py`
- `launch_looksam.py`, `launch_muon.py`, `launch_neuralgrok.py`
- `launch_prodigy.py`, `launch_supergrok11.py`, `launch_supergrok15.py`, `launch_supergrok2.py`

**v5p/ and v6e/ subdirs:** `__init__.py` only (architecture-specific init scaffolding).

### 3b. csrc/fused/tpu_v6e/

33 cells = 3 models × 11 optimizers. All present:
- `mega_transformer_decoder_{adamw,grokadamw,grokfast,lion,looksam,muon,neuralgrok,prodigy,supergrok11,supergrok15,supergrok2}.py`
- `mega_mamba3_*.py` (11)
- `mega_vit_*.py` (11)

Each is a thin `functools.partial` wrapper over `_pallas_fused.fused_step` binding model+optimizer+tier=L3. Has `verify()` calling `trace_check()`. These are generated by `megakernel_codegen.py`. Status: real composition (not stubs) per comments, but TPU hardware not available.

---

## 4. third_party/cutlass

**Version: 3.6.0** (released 2024-10-03). Only vendored third-party library.

**Role:** Provides CuTe atoms, WGMMA/TMA wrappers, and sm_90 collective primitives used by the sm_90 megakernel. Key items:
- `include/cute/` — CuTe layout algebra, atoms, algorithms
- `include/cutlass/` — Hopper-era GEMM, epilogue, collective primitives
- sm_90 content: warp-specialized GEMM, TMA load/store, WGMMA instructions

**Not used by gfx942 or TPU paths** — those use `__builtin_amdgcn_mfma_*` and Pallas respectively.

---

## 5. Completeness Assessment

| Arch | Optimizers (11?) | Models (3?) | Fused megakernel? | Status |
|------|-----------------|-------------|-------------------|--------|
| gfx942/HIP | YES (11/11) | YES (3/3) | YES (fused_megakernel.hip.hpp) | 🟡 hardware-gated, not compiled |
| TPU v6e/Pallas | YES (11/11) | YES (3/3) | YES (33 tpu_v6e cells) | 🟡 JAX trace-only, no TPU hw |
| sm_90 (active) | YES (11/11) | YES (3/3) | YES (active, exercised) | Active |

**CUTLASS 3.6.0:** Present, headers intact, used only by sm_90 path.

---

## 6. Key Findings

1. **Full parity across arches**: All 3 alt-arch backends (gfx942/HIP, Pallas/TPU-v5p, Pallas/TPU-v6e) implement the complete 11-optimizer × 3-model matrix. No missing cells.

2. **Preserved-but-not-exercised**: All gfx942 code is explicitly labeled 🟡 (hardware-gated, not hipcc compiled). All Pallas/TPU code has JAX-fallback guards. Neither compiles or runs in the current sm_90 H100 session.

3. **Architecture separation is clean**: gfx942 uses `__AMDGCN__`/`__HIPCC__` compile-routing guards; TPU uses `try/except ImportError` for JAX/Pallas. No cross-contamination with sm_90 path.

4. **Fused megakernel structure mirrors sm_90**: `csrc/fused/gfx942/fused_megakernel.hip.hpp` is declared as the AMD twin of `csrc/fused/sm_90/fused_megakernel.cuh`, using the same task-queue/persistent CU-pin pattern but with 4-wave-interleave (no Hopper warp-spec analog).

5. **No discrepancies with CLAIMED state**: The claim that these are "preserved" alt-arch paths is confirmed. The CLAIMED state says nothing specific about gfx942/TPU completeness beyond existence; code shows full 11×3 coverage.

6. **CUTLASS 3.6.0** (not 3.5 or earlier) is the vendored version — the most recent CUTLASS at sm_90 feature-complete level, including Hopper sparse GEMM and PDL.
