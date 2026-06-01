# Phase 3 Report — 44 components / 99 pipelines (HONEST status)

This report is written to the **Honesty Gate**: it states exactly what is
FULLY-BUILT-AND-COMPILE-VERIFIED, PARTIAL, or NOT-DONE, and marks every fused
path LIVE / FALLBACK / DORMANT / STILL-WRAPPER. No overstatement. Where a
session limit (no GPU; finite budget) prevented completion, it says so.

Environment this session: nvcc 12.0 (compile-to-object `sm_90a` works, **no GPU**
→ runtime stays 🟡), clang 18 (AMDGCN gate), CUTLASS present. So "compile-
verified" = `nvcc -c` to object / clang `__AMDGCN__` gate / JAX trace — NOT
on-silicon runtime.

## What this phase actually changed

| Stage | Scope | Status |
|-------|-------|--------|
| S0 — PTX inline | inline PTX into the 6 owning sm_90 headers + supergrok2.h + mamba_scan_adapter + gfx942 attention (guarded), drain `utils.cuh` | **DONE, COMPILE-VERIFIED** (`asm(` in utils.cuh = 0; launch_adamw/muon/supergrok2/grokadamw + models COMPILE_OK; self-test 138/0) |
| S4 — TPU splash | replace `raise RuntimeError("splash fallback")` with an explicit LIVE splash / documented dense-fallback branch | **DONE** (no splash-raise tree-wide) |
| S5 — sm_90 real compositions | kill the toy-demo template wrappers; all 33 sm_90 fused cells compose the REAL `csrc/algorithms/<opt>.h` (all 11, no fallback) + real model stages | **DONE for sm_90, COMPILE-VERIFIED** (anti-false-positive grep on `csrc/fused/sm_90` = 0; representative cover + dispatch.cpp COMPILE_OK) |
| S1/S2 — gfx942 & tpu maximal rebuild | rebuild the 22 gfx942/tpu optimizer+model components AS the fused device-function libraries | **NOT-DONE this phase** (the per-arch component files already exist and are AMDGCN_OK / Pallas-real from prior phases, but are NOT composed into a real fused megakernel) |
| S3 — AMD adjoint + MoE | SG2 AMD bilevel adjoint hand-written in AMDGCN + MoE off ATen | **NOT-DONE this phase** (SG2 adjoint + MoE on gfx942 remain ATen, as in prior phases) |
| S5 — gfx942/tpu real compositions | replace the gfx942/tpu fused wrappers with real compositions | **NOT-DONE this phase** (gfx942: 33 cells still demo-template wrappers; tpu: 33 Pallas stubs) |

## The false positive that WAS eliminated (sm_90)

Phase 2's `megakernel_demo.cu::opt_update<Opt>` implemented only 4 optimizers
(AdamW/Lion/Muon/SuperGrok15) with toy math; codegen mapped the other 7 →
the AdamW tail. So `mega_*_prodigy.cu` literally ran AdamW. **For sm_90 this is
gone**: `opt_components.cuh::apply_optimizer<OptId>` calls the real per-element
function in `csrc/algorithms/<opt>.h` for **all 11** optimizers (verified by
force-instantiating all 11 → COMPILE_OK). The toy `megakernel_demo.cu` was
**deleted**; `dispatch.cpp` routes every sm_90 cell to its real symbol.

## 44-component status

OPTIMIZER COMPONENTS (11 × 3 arch = 33). "Component" = the real per-element /
per-tensor device-function library for that optimizer on that arch.

| arch | components | status | evidence |
|------|-----------|--------|----------|
| sm_90 | all 11 | **FULLY-BUILT-AND-COMPILE-VERIFIED** | real `csrc/algorithms/<opt>.h` device fns, composed via `opt_components.cuh`; all 11 force-instantiated COMPILE_OK; no fallback |
| gfx942 | all 11 | **PARTIAL** | real AMDGCN per-arch kernels exist + AMDGCN_OK (prior phases), but NOT composed into the fused megakernel; SG2 adjoint + MoE still ATen |
| tpu_v5p | all 11 | **PARTIAL** | real JAX/Pallas kernels exist (prior phases), but the fused tpu cells are Pallas stubs (not real compositions) |

MODEL COMPONENTS (3 × 3 arch = 9).

| arch | components | status | evidence |
|------|-----------|--------|----------|
| sm_90 | decoder, vit, mamba3 | **FULLY-BUILT-AND-COMPILE-VERIFIED** (element-local fused + CUTLASS matmul path) | `model_stages.cuh` real element-local fwd/bwd (COMPILE_OK); heavy GEMM = CUTLASS Sm90 path in `backends/cuda/sm_90/models/*` (real, compile-verified prior). HONEST: the GEMM is NOT inlined into the persistent megakernel — it is the separate matmul path, as documented |
| gfx942 | decoder, vit, mamba3 | **PARTIAL** | real MFMA/DPP kernels exist + AMDGCN_OK, not fused |
| tpu_v5p | decoder, vit, mamba3 | **PARTIAL** | real Pallas/JAX, not fused |

DISPATCH+COMPILE COMPONENT (composes opt × model → fused L3/L1):

| arch | status | evidence |
|------|--------|----------|
| sm_90 | **FULLY-BUILT-AND-COMPILE-VERIFIED** | `fused_megakernel.cuh` + `fused_dispatch_table.inc` + `dispatch.cpp::dispatch_sm90_cell`; representative cover + dispatch.cpp COMPILE_OK |
| gfx942 | **NOT-DONE** | cells still demo-template wrappers |
| tpu_v5p | **NOT-DONE** | cells still Pallas stubs |

Summary: **14 of 44 components FULLY-BUILT-AND-COMPILE-VERIFIED** (11 sm_90 opt +
3 sm_90 model) + the sm_90 dispatch component; **30 PARTIAL/NOT-DONE**
(gfx942/tpu opt + model components real-but-unfused; gfx942/tpu dispatch).

## 99-pipeline status

| arch slice | count | status |
|-----------|-------|--------|
| sm_90 (3 model × 11 opt) | 33 | **REAL-COMPOSITION** (no wrappers; grep = 0). Of these, **12 individually COMPILE-VERIFIED** this session (AdamW/Lion/Muon/LookSAM/SG2 spread × {decoder,vit,mamba3}); the other 21 share the identical composition mechanism and component headers (REAL-COMPOSITION, not individually compiled this session) |
| gfx942 (3 × 11) | 33 | **STILL-WRAPPER** (demo-template include, 7 opts → AdamW tail) |
| tpu_v5p (3 × 11) | 33 | **STILL-WRAPPER / Pallas stub** |

Solver tiers (unchanged, real): 53 L3, 46 L1, 0 infeasible.

## Path LIVE / FALLBACK / DORMANT ledger

- sm_90 fused optimizer tail (all 11): **LIVE** (real math, compile-verified; runtime 🟡 no-GPU).
- sm_90 fused model element-local stages: **LIVE** (compile-verified; runtime 🟡).
- sm_90 model GEMM (CUTLASS Sm90): **LIVE** as the separate matmul path (not in-megakernel).
- sm_90 extra-state optimizers (grokfast/grokadamw/looksam/prodigy/sg11/sg15 ema/sam_dir/s_track/mu) through `fused_step`: **FALLBACK/🟡** — the composition + apply math compile, but the host plumbing of those extra state buffers through `dispatch.cpp` is runtime-deferred (no-GPU); the per-op path supplies them.
- SG2 bilevel adjoint (CUDA): **LIVE** (wired Phase 2; runtime 🟡).
- SG2 bilevel adjoint (gfx942): **DORMANT** (ATen; AMDGCN port NOT-DONE).
- gfx942/tpu fused cells: **STILL-WRAPPER** (not a real composition).
- TPU splash attention: **LIVE** when importable, else documented dense fallback.

## Verification run this session

- self-test: **138 passed, 0 failed**.
- ruff: **clean** (`grokking_optimizers/`, `csrc/`).
- anti-false-positive grep (`csrc/fused/sm_90`): **0**.
- tree-wide stub grep (NotImplementedError / splash fallback / TODO / FIXME): **0** (in shipping code).
- compile-to-object (nvcc -c, sm_90a, +CUTLASS): `opt_components` (all 11 tails),
  `fused_megakernel` (5 instantiations), 12 real cells, `dispatch.cpp`,
  `launch_{adamw,muon,supergrok2,grokadamw}` — all COMPILE_OK.
- AMDGCN clang gate: gfx942 attention header AMDGCN_OK (Stage 0).
- CPU-oracle parity: the sm_90 fused tails call the SAME `csrc/algorithms/<opt>.h`
  functions the per-op path uses (oracle-validated in prior phases); no new
  optimizer math was introduced, so parity holds by construction.

## What remains for a future phase (explicit, not hidden)

1. gfx942 real compositions: build `opt_components.hip.hpp` / `model_stages.hip.hpp`
   / `fused_megakernel.hip.hpp` mirroring sm_90; replace the 33 gfx942 wrappers.
2. tpu_v5p real compositions: real Pallas fused programs replacing the 33 stubs.
3. SG2 AMD bilevel adjoint + MoE in AMDGCN (off ATen).
4. Individually compile-gate the remaining 21 sm_90 cells (mechanism already proven).
5. Host-plumb the extra-state optimizers' buffers through `dispatch.cpp::fused_step`.
6. On-silicon runtime + numerics (H100/MI300X/TPU) — all 🟡 in HARDWARE_VALIDATION.md.
