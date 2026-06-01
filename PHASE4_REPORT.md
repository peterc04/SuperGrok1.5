# Phase 4 Report — register pass, AMD device-pass, FP32 tensor-core, tree reconciliation

Written to the Honesty Gate. Every path is marked LIVE / FALLBACK / DORMANT;
perf/tier/numeric claims that need real silicon are marked 🟡. Verified ONCE in a
single consolidated pass (Stage V) after all six workstreams — not per stage.

Environment: nvcc 12.0 (`nvcc -c sm_90a` works, **no GPU**), clang 18 (AMDGCN
gate), CUTLASS, JAX 0.10.1. "Compile-verified" = object-compile / clang gate /
JAX lower — NOT on-silicon execution.

## Workstream outcomes

| WS | scope | outcome | verification |
|----|-------|---------|--------------|
| **1** | SG2 + L3 register-pressure reduction (behavior-preserving) | DONE | SMEM staging (SG2 CSA/HCA softmax accumulator + indexer top-k; model-bwd adjoint), rematerialization + live-range shortening (GELU′), `setmaxnreg` warp-group split (256-thread block = producer WG `dealloc<32>` / consumer WG `alloc<200>`), register-cap wired as a per-cell autotuner knob in compile.py. The 3 fused headers + SG2-L3 cell **nvcc -c COMPILE_OK**. Bit-identical by construction (storage-class / schedule only). |
| **2** | AMD device-pass fill (11 optimizers) | DONE (gap was small) | device passes were largely already filled; the genuine gap — **Muon Newton-Schulz device MFMA** — was added (real 16×16×16 bf16 MFMA, sched_group_barrier interleave). All touched gfx942 files **AMDGCN_OK**. |
| **3** | FP32 tensor-core path for models | DONE | TF32 (`cutlass::tfloat32_t`) Sm90 collective added to `mma.cuh` (`sm90_run_gemm_tf32[_bt]`); wired in decoder/vit/mamba/attention FP32 sites; scalar kept as last-resort + `-DSG_FORCE_SCALAR_FP32` escape. `models/{decoder,vit,mamba}.cu` **COMPILE_OK**. |
| **4** | reconcile the two optimizer-math trees | DONE | found they were ALREADY single-source: `kernels/sm_90/<opt>_sm90.cuh` AND `opt_components.cuh` both `#include csrc/algorithms/<opt>.h`. Added `csrc/algorithms/SOURCE_OF_TRUTH.md` + `scripts/check_math_single_source.py` (structural guard, exit 0). |
| **5** | TPU `kernels/tpu/` cleanup | DONE (filled) | tree was used (not dead) but partial (7 opt). Added the 4 base (`adamw/lion/grokfast/grokadamw`) as REFERENCE re-export shims of the authoritative `launch_<opt>.py` → symmetric all-11, single source, no duplication. |
| **6** | 44-component + 99-pipeline status tables | DONE | below + appended to BUILD_REPORT.md. |

## WS1 before→after fuse-tier table (🟡 ESTIMATES — ptxas is the real arbiter)

| arch | before L3 | after L3 | before L1 | after L1 | note |
|------|-----------|----------|-----------|----------|------|
| sm_90 | 10 | **33** | 23 | **0** | all 33 cells now fit L3 under the staged-register model |
| gfx942 | 10 | **11** | 23 | **22** | mamba3 ×11 promoted; decoder/vit stay L1 — their staged L3 smem (66560B) exceeds gfx942's 64KB LDS (an honest *smem* limit, not register) |
| tpu_v5p | 33 | 33 | 0 | 0 | XLA-managed (unchanged) |
| **total** | **53** | **77** | **46** | **22** | |

🟡 These tiers are **estimates** from the solver's additive register model
(megakernel.py). ptxas register allocation differs (it may keep a "staged"
value live, or free a slot the model assumed live). The compile.py per-cell
`maxrregcount` sweep + on-silicon `ptxas -v` are the real arbiters. Stated in
the megakernel.py docstring and `_tier_cost`.

## 44-component status

OPTIMIZER COMPONENTS (11 × 3 = 33):

| arch | status | evidence |
|------|--------|----------|
| sm_90 (×11) | FULLY-BUILT, **nvcc-object-verified** | canonical `csrc/algorithms/<opt>.h`; per-op + fused both `#include` it (WS4 guard); SG2-L3 fused cell COMPILE_OK |
| gfx942 (×11) | FULLY-BUILT, **clang-amdgcn-gate-verified** | device passes filled; reducers use DPP; Muon NS + SG2 use MFMA; ATen only in `!__AMDGCN__` host fallback |
| tpu_v5p (×11) | FULLY-BUILT, **jax-lower-verified** | all 11 in `kernels/tpu` (WS5) + `_pallas_fused` 66/66 trace+lower |

MODEL COMPONENTS (3 × 3 = 9):

| arch | status | evidence |
|------|--------|----------|
| sm_90 (×3) | FULLY-BUILT, **nvcc-object-verified** | `models/{decoder,vit,mamba}.cu` COMPILE_OK; TF32 tensor-core path LIVE + scalar FALLBACK (WS3) |
| gfx942 (×3) | FULLY-BUILT, **clang-amdgcn-gate-verified** | MFMA/DPP model kernels |
| tpu_v5p (×3) | FULLY-BUILT, **jax-lower-verified** | `_pallas_models.py` fwd/bwd |

DISPATCH + COMPILE (2):

| component | status | evidence |
|-----------|--------|----------|
| dispatch (sm_90 nvcc / gfx942 hipcc-🟡 / tpu unified py) | FULLY-BUILT | sm_90 `dispatch_sm90_cell` COMPILE_OK; gfx942 `#if WITH_HIP` route structural (🟡 hipcc); tpu via `dispatch_fused_megakernel` (trace) |
| compile/codegen (autotuner register-cap sweep + 99-cell generator) | FULLY-BUILT, **self-test-verified** | compile.py register-cap sweep; self-test 138/0 |

**44/44 components built.** Each is verified at its arch's gate level (nvcc
object / clang amdgcn / jax lower). On-silicon execution + numerics are 🟡.

## 99-pipeline status (post-WS1 tiers)

| arch | count | status | tiers |
|------|-------|--------|-------|
| sm_90 | 33 | REAL-COMPOSITION-COMPILED (representative + WS1/WS3 TUs nvcc -c; 0 wrappers) | 33 L3 🟡 |
| gfx942 | 33 | REAL-COMPOSITION-GATE-VERIFIED (clang amdgcn; 0 wrappers) | 11 L3 / 22 L1 🟡 |
| tpu_v5p | 33 | REAL-COMPOSITION-TRACE-VERIFIED (66/66 lower) | 33 L3 |

**0 / 99 STILL-WRAPPER** (anti-false-positive grep = 0 across `csrc/fused/`).

## gfx942 DPP / MFMA / ATen (Phase-4 surface)
- DPP: 13 files (all reduction optimizers — LookSAM/Prodigy/Muon/SG11/SG15 — covered).
- MFMA: 9 files (added Muon Newton-Schulz this phase; SG2 fwd + adjoint already MFMA).
- ATen: 11 files — all in `!__AMDGCN__` host-orchestration / CPU-fallback (correct by design).

## LIVE / FALLBACK / DORMANT ledger
- sm_90 fused L3 (model stages + 11 real optimizer tails), register-reduced: **LIVE** (compiled; tiers + runtime 🟡).
- sm_90 model FP32 TF32 tensor-core path: **LIVE**; scalar FP32: **FALLBACK** (`-DSG_FORCE_SCALAR_FP32`). TF32 ≈10-bit mantissa — accepted FP32-TC precision, not bit-identical to scalar (🟡 accuracy note, not a bug).
- gfx942 device passes (11 opt + SG2 adjoint + MoE): **LIVE** on device pass; ATen host = **FALLBACK** (CPU build). MI300X numerics 🟡.
- TPU fused programs (33): **LIVE** (trace+lower); on-TPU runtime 🟡.
- WS4 CUDA single-source: **enforced** (guard script). gfx942/TPU re-expressions: cross-referenced manual-sync.

## What remains (genuinely external)
1. **On-silicon execution + numeric parity** (H100 / MI300X / TPU v5p) — the only remaining class; nothing was executed on an accelerator. All 🟡 in HARDWARE_VALIDATION.md.
2. **ptxas register confirmation** of the WS1 L3 tier promotions (the 53→77 is a modeled estimate).
3. gfx942 decoder/vit L3 is smem-bound (66560B > 64KB LDS) — would need a smaller model tile (out of scope).
