# Phase 7 Report — targeted cleanup + AMD apply-step vectorization (final impl pass)

Written to the NO-CORNERS standard: every count carries its before→after delta +
the command; honest about non-equivalence and per-optimizer limits. Verified once
(Stage V) after all six workstreams.

Environment: nvcc 12.0 (`nvcc -c sm_90a`, no GPU), clang 18 AMDGPU gate, CUTLASS,
JAX 0.10.1. All runtime/numeric claims are 🟡 (no accelerator); the rest is
compile/clang/trace/self-test verified.

## Workstreams (all 6 done)

- **W1 — delete dead kernels/tpu**: 16 files deleted; `_pallas_fused.py` redirected
  its 7 imports to the canonical pallas launchers; `v5p/__init__` + compile.py
  self-tests redirected to pallas. `grep` code-imports of `kernels.tpu`: **5 → 0**.
  HONEST: the pallas launchers are NOT bit-identical to the deleted duplicate for
  looksam (Δ6.6e-3), prodigy (Δ2.4e5), muon, supergrok2 — they are the intended
  canonical path; this is a dedup, not a refactor; TPU numeric parity is 🟡.
- **W2 — de-inline 3 optimizers**: grokadamw/neuralgrok/supergrok11 — inline Adam
  tail → new canonical `<opt>_adam_tail` in `csrc/algorithms/`. Bit-identical
  (expression-tree, vs git HEAD). 3 TUs nvcc -c COMPILE_OK.
- **W3 — harden drift guard**: added re-inline detection (scans sm_90 consumers
  for re-typed moment-update/apply). Optimizer-math-outside-algorithms count
  (sm_90 consumers): **2 → 0**. Guard passes clean AND triggers on injection.
- **W4 — generator-driven dispatch**: `wired_fused_cell()` (93 hand-pasted lines)
  → `#include fused_wired_cells.inc` (generated). Table = 99/99 cells, 0 drift.
- **W5 — AMD apply vectorization**: all 11 gfx942 apply-steps → f32x4 (128-bit)
  streaming load/store + scalar tail. f32x4-apply coverage: **~3/11 → 11/11**.
  DPP reductions / MFMA untouched. Per-optimizer honesty: SG2's main CSA/HCA
  apply tail stays ATen (fused into the per-row attention pipeline, coupled-WD —
  not a standalone elementwise kernel); its standalone MoE/Adam apply IS f32x4.
- **W6 — README/BUILD_REPORT**: corrected to the true post-cleanup state (no
  kernels/tpu; pallas canonical for all 11 TPU; enforced re-inline guard;
  generator dispatch; vectorized AMD), reframed Phase-6 honestly.

## Stage V (single consolidated pass; each with command)

| check | command | result |
|-------|---------|--------|
| self-test | `compile --self-test` | **156 passed, 0 failed** |
| ruff | `ruff check grokking_optimizers/ scripts/ pallas/` | clean |
| gfx942 gate | `amdgcn_check.sh --header` ×14 | **14/14 AMDGCN_OK** |
| sm_90 compile | `compile_to_object.sh` launch_{grokadamw,neuralgrok,supergrok11}.cu + dispatch.cpp | COMPILE_OK |
| drift guard | hardened guard | PASS clean; **TRIGGERS** on injected re-inline; restores |
| dispatch table | generated vs solve_all | **99/99**, 0 drift |
| dead tree | kernels/tpu files + code imports | **0 / 0** |
| anti-false-positive | whole-tree grep | 0 real (1 benign codegen docstring "NO TEMPLATE_ONLY") |
| f32x4 apply | grep ×11 | **11/11** |

## LIVE / FALLBACK / DORMANT
- sm_90 fused L3/L1 + TF32 model GEMM: **LIVE** (nvcc-object); tiers/runtime 🟡.
- gfx942 device kernels (11 opt f32x4-vectorized + SG2 fwd/bwd/MoE): **LIVE on
  hipcc**; ATen **FALLBACK** (CPU); host-launch + numerics 🟡.
- TPU Pallas fused (33, canonical pallas math): **LIVE** (trace+lower); runtime 🟡.
- drift guard (incl. re-inline detection): **LIVE + enforced**.
- DORMANT: none.

## Bottom line
Implementation-maximal across sm_90 / gfx942 / tpu. Single canonical math source
per component, **enforced** (incl. re-inline detection); no dead duplicate trees;
generator-driven dispatch; vectorized AMD apply-steps. The ONLY remaining class
is **gap #7 — on-silicon validation** (H100 / MI300X / TPU v6e) per
`HARDWARE_VALIDATION.md`, to move 🟡 → ✅.
