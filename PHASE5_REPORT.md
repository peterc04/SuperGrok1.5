# Phase 5 Report — AMD device-pass live-wiring + enforced drift guard + system-wide verification

Written to the NO-CORNERS standard. Every claim has a command/result; every
path is LIVE / FALLBACK / DORMANT; stale prompt premises are corrected with
evidence rather than papered over. Verified ONCE, system-wide (Stage V).

Environment: nvcc 12.0 (`nvcc -c sm_90a`, **no GPU**), clang 18 AMDGPU gate
(`scripts/amdgcn_check.sh`), CUTLASS, JAX 0.10.1. No hipcc, no MI300X/H100/TPU.
"Verified" = object-compile / clang-device-gate / JAX-lower / self-test — never
on-silicon execution.

## Correction of stale prompt premises (verified)

The Phase 5 prompt described a pre-Phase-4 snapshot. Actual state at Phase 5
start (commands in V.7):
- "SG2 gfx942 backward still throws (forward only)" — **FALSE**. `launch_csa_hca_backward` already called `sg2adj::bilevel_backward_driver` (real reverse-mode VJP); the file's own line said "NO throw remains". The "throws/forward-only" notes (supergrok2_gfx942 L81, bindings.cpp L1889) were **stale comments** — now corrected.
- "DPP=5, MFMA=7, apply-steps still ATen" — **stale**. All 11 device passes already had real AMDGCN kernels; DPP was in 10 files, MFMA in 7; all 4 reducers already used DPP wave-64 trees.

The ONE real gap (confirmed): the gfx942 host launchers called ATen `prim::`
**unconditionally** — the real device kernels existed but were **DORMANT** (never
dispatched; `hipLaunchKernelGGL` was comment-only). That is the genuine Phase 5
fix.

## Workstream outcomes

| WS | scope | outcome |
|----|-------|---------|
| **1** | AMD device-pass: make the device kernels the LIVE path | DONE (live-wired). All 11 optimizers + SG2 fwd/backward/MoE: host launchers now dispatch the (already AMDGCN_OK) device kernels under `#if defined(__HIPCC__)`, with ATen as the `#else` CPU fallback. SG2 device bilevel adjoint live-wired. All stale "throws/forward-only/dormant" comments corrected. **Device side AMDGCN_OK (19/19 headers); the hipLaunchKernelGGL host glue is 🟡 (no hipcc here).** |
| **2** | enforce the drift guard (was comment-only) | DONE. `scripts/check_math_single_source.py` now has TWO real teeth: structural single-source (per-op + fused both `#include` the canonical `csrc/algorithms/<opt>.h`) + a content-hash manifest (`scripts/optimizer_math_manifest.json`) that FAILS on any canonical-math change. Wired into `--self-test` as `math_drift_guard` (3 tests) incl. a **prove-it-triggers** test (perturb canonical math → guard flags). self-test 138→**141**. |
| **3** | regenerate stale fused-cell comments | DONE. `--write-all` regenerated all 99 cells' tier/reg/smem comments from the LIVE solver; stale `tier=L1_OPT_ONLY` count 46→22 (= solver's actual 22 L1). Generator-emitted so they can't drift. |
| **4** | fix report prose + 44-table | DONE. Removed the stale "one templated L3 megakernel / 3 demo cells" line in BUILD_REPORT (demos were deleted Phase 3; all 99 are real compositions); the 44-component + 99-pipeline tables are below + in BUILD_REPORT. |
| **5** | populate HARDWARE_VALIDATION per-cell matrix | DONE. Full 99-cell × 3-arch matrix (generated from the live solver) inserted: each cell's fuse tier + numeric-oracle reference + 🟡. |

## Stage V — system-wide verification (single pass; every result with its command)

| check | command | result |
|-------|---------|--------|
| V.1 sm_90 compile | `compile_to_object.sh dispatch.cpp / mega_vit_muon.cu -DWITH_CUTLASS` | COMPILE_OK |
| V.1 gfx942 gate (ALL) | `amdgcn_check.sh --header` over every gfx942 header | **19/19 AMDGCN_OK** |
| V.2 config sources | mirror setup.py globs per config | WITH_CUDA: 49 srcs / 0 missing / 33 fused; WITH_HIP: 46 / 0 / 33; setup.py globs all present |
| V.3 emission sweep | `emit_cell` for all 99 | **99/99 REAL-COMPOSITION**, 0 demo/TEMPLATE_ONLY |
| V.4 5-way consistency | solver tier ↔ cell-comment tier ↔ dispatch route ↔ component-real | **99/99 consistent, 0 mismatch** |
| V.5 anti-false-positive | forbidden-pattern grep, whole tree | 0 actual occurrences (4 benign: a codegen docstring saying "NO TEMPLATE_ONLY"; `-DXXX` example flag; 2 honest "ROCm Composable-Kernel Python frontend not in-tree yet" TODOs) |
| V.6 self-test | `compile --self-test` (×3) | **141 passed, 0 failed** (stable) |
| V.6 ruff | `ruff check grokking_optimizers/ scripts/` | clean |
| V.6 drift guard | `check_math_single_source.py` + the prove-it-triggers self-test | enforced; triggers on divergence |

## V.7 deltas — gfx942 (commands in the run log)

| metric | before P5 | after P5 | note |
|--------|-----------|----------|------|
| device-dispatch host launchers (`#if defined(__HIPCC__)`) | 0 | **11** | the real change: DORMANT → LIVE-on-hipcc |
| real `hipLaunchKernelGGL` call sites (kernels/gfx942) | 0 (comment-only) | **25** | device kernels now actually launched |
| DPP files | 10 | 10 | already done (Phase 4); all 4 reducers use DPP wave-64 |
| MFMA files | 7 | 7 | already done; MFMA only for matmul opts (Muon NS, SG2) — forcing it into elementwise opts would be fake |
| ATen refs (kernels/gfx942) | 609 | **744 (UP)** | HONEST: ATen is RETAINED as the `#else` CPU fallback (per the two-pass design) + the device dispatch adds host `.data_ptr()/.numel()` extraction. "ATen drops" contradicts "ATen remains the CPU-host fallback" — the device path is added alongside, not by deleting the fallback. |

## 44-component status

| group | sm_90 | gfx942 | tpu_v6e | verification |
|-------|-------|--------|---------|--------------|
| 11 optimizers | FULLY-BUILT (nvcc-object) | FULLY-BUILT, device LIVE-on-hipcc (clang-amdgcn-gate; host-launch 🟡) | FULLY-BUILT (jax-lower) | per arch |
| 3 models | FULLY-BUILT (nvcc-object; TF32 path) | FULLY-BUILT (clang-amdgcn) | FULLY-BUILT (jax-lower) | per arch |
| dispatch | FULLY-BUILT (nvcc, `dispatch_sm90_cell`) | FULLY-BUILT structural, 🟡 hipcc (`#if WITH_HIP dispatch_gfx942_cell`) | FULLY-BUILT (`dispatch_fused_megakernel`/`_pallas_fused`) | — |
| compile/codegen + drift-guard | FULLY-BUILT, self-test-verified (141/0) | | | — |
**44/44 built**; each verified at its arch gate. On-silicon execution/numerics 🟡.

## 99-pipeline status (live-solver tiers, post-Phase-4 re-tier)
- sm_90: 33 REAL-COMPOSITION-COMPILED (33 L3 🟡-estimate)
- gfx942: 33 REAL-COMPOSITION-GATE-VERIFIED (11 L3 / 22 L1; device LIVE-on-hipcc 🟡)
- tpu_v6e: 33 REAL-COMPOSITION-TRACE-VERIFIED (33 L3)
**0 STILL-WRAPPER.** 5-way consistent (V.4).

## LIVE / FALLBACK / DORMANT ledger
- gfx942 per-element apply (all 11): device kernel **LIVE on hipcc** (`#if __HIPCC__`); ATen `#else` = CPU **FALLBACK**. (was DORMANT — fixed.) MI300X numerics 🟡.
- SG2 gfx942 forward (CSA/HCA MFMA + PEER + GRU): **LIVE on hipcc**; ATen FALLBACK.
- SG2 gfx942 backward: ATen `bilevel_backward_driver` **LIVE** (CPU, functional, no throw); AMDGCN device adjoint **LIVE on hipcc**; numerics 🟡.
- SG2 gfx942 MoE (filter/scatter/histogram): device **LIVE on hipcc**; ATen FALLBACK.
- sm_90 fused L3 + TF32 model path: **LIVE** (compiled; tiers + numerics 🟡).
- drift guard: **LIVE + enforced** in `--self-test`.

## What remains hardware-locked (gap #7) — the ONLY remaining class
Nothing was executed on an accelerator (none in this env). All of: the
hipLaunchKernelGGL host-launch glue (needs hipcc), gfx942/SG2-adjoint/MoE numeric
parity (MI300X), the WS1-Phase-4 L3 re-tier confirmation (`ptxas -v`, H100), and
TPU runtime (v5p) — are 🟡 in HARDWARE_VALIDATION.md (now a complete 99-cell
checklist). Everything else is closed at "implemented + system-verified
(CPU/clang/nvcc)".
