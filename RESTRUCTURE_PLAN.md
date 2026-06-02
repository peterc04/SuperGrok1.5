# RESTRUCTURE_PLAN.md — Phase 6 Stage 0 (inventory + plan; NO code moved yet)

> Mandate: produce this plan and commit it BEFORE moving/deleting anything.
> Behavior preservation is the hard requirement; a documented stop beats a
> silent half-merge.

## Executive finding (read first)

**The restructure's GOAL — "one canonical source of truth per component, no
parallel math trees, no dead duplicates" — is ALREADY MET.** It was achieved
incrementally across Phases 4–5 (the `csrc/algorithms` canonicalization, the
WS2 enforced drift guard, the WS5 TPU consolidation). This Stage-0 inventory
verifies that with evidence below. Consequently:

- There is **NO duplicate CUDA math tree to merge** (verified: all 11
  `kernels/sm_90/<opt>_sm90.cuh` `#include` the canonical `csrc/algorithms/
  <opt>.h` and call its `<opt>_step`; the drift guard enforces this in
  `--self-test`).
- There is **NO dead `kernels/tpu/` tree to delete** (verified: 4 live importers;
  it holds the single math source for 7 TPU optimizers).
- The few remaining re-expressions (gfx942 device transcription; 2 sm_90 fused
  variant tails) are **documented, necessary, and drift-guard-tracked** — not
  duplicate trees.

**Therefore a forced tree-merge is NOT warranted and would be UNSAFE**: it would
rewrite what `setup.py` compiles, what the fused cells include, and what bindings
call, on a codebase that is currently system-verified (self-test 141/0, 19/19
AMDGCN_OK, 99/99 5-way-consistent) — for **zero structural gain** (there is
nothing to de-duplicate) and real regression risk that **cannot be
behavior-proven here** (no GPU for the device paths). Per the phase's own safety
clause, the correct action is the **documented stop** on the merge, plus the
genuine, zero-risk value-adds: this plan (canonical-architecture documentation),
the README rewrite (Stage 3), and branch hygiene (Stage 2).

The sections below are the full inventory/map/graph the mandate requires.

---

## 0.1 File inventory (LOC + role)

Total tracked source: **80,119 LOC**. Key trees:

| tree | files | LOC | role |
|------|------:|----:|------|
| `csrc/algorithms/` | 12 | 2271 | **CANONICAL-MATH** — the per-element optimizer math (`<opt>.h`), the SG2 bilevel adjoint, `SOURCE_OF_TRUTH.md`. ONE definition per optimizer. |
| `grokking_optimizers/kernels/sm_90/` | 17 | 9262 | **OPTIMIZER (launch wrappers, thin)** + **MODEL (CUTLASS, heavy)**. Optimizer files 132–377 LOC each `#include` the canonical header (no math dup). Model files (supergrok2 2556, vit 1539, decoder 1426, mamba3 775, attention 679) are the **canonical CUDA model home**. |
| `grokking_optimizers/kernels/gfx942/` | 17 | 9672 | **OPTIMIZER/MODEL (AMDGCN device kernels)** — the single gfx942 device source per component. |
| `grokking_optimizers/kernels/tpu/` | 16 | 5032 | **OPTIMIZER/MODEL (JAX reference)** — single math source for 7 TPU optimizers + 3 models; 4 base optimizers are pure re-export SHIMS of `pallas/launch_*`. |
| `csrc/backends/cuda/sm_90/` | 21 | 1621 | **SHIM (pure entry points, ~5 LOC each)** `launch_<x>.cu` `#include` the kernel header; + `models/*.cu` (90 LOC wrappers). Zero math. |
| `csrc/backends/hip/gfx942/` | 22 | 1986 | **SHIM + SUPPORT** — `launch_*.hip.cpp` entry points, `amdgcn_primitives.hip.hpp`, the SG2 device adjoint + MoE compaction. |
| `csrc/backends/pallas/` | 16 | 4875 | **OPTIMIZER/MODEL (TPU executed path)** — `launch_<opt>.py` (canonical for the 4 base opts), `_pallas_models.py`, `_pallas_fused.py` (the 99-cell TPU composer), `_pallas_kernels.py`. |
| `csrc/fused/` | 74 | 3306 | **DISPATCH/COMPOSITION** — `opt_components.{cuh,hip.hpp}`, `model_stages.{cuh,hip.hpp}`, `fused_megakernel.*`, the 99 `mega_<model>_<opt>` cells, the substrate. |

## 0.2 Duplication map (per component → single home, verified)

**11 optimizers (per arch):**

| arch | single canonical home | consumers (zero-math, reference it) | duplicate? |
|------|----------------------|-------------------------------------|------------|
| sm_90 | `csrc/algorithms/<opt>.h` (elementwise math) | `kernels/sm_90/<opt>_sm90.cuh` (`#include` + launch), `csrc/fused/sm_90/opt_components.cuh` (`#include` + `apply_optimizer`) | **NO** — `#include`, drift-guard-enforced |
| gfx942 | `kernels/gfx942/<opt>_gfx942.hip.hpp` (device kernel) | `launch_<opt>.hip.cpp` (entry); `csrc/fused/gfx942/opt_components.hip.hpp` is a **documented byte-faithful transcription** of `csrc/algorithms` (thrust blocks `#include` on the bare AMDGCN gate — the one necessary re-expression, cross-referenced + drift-guard-noted) | **NO** (necessary re-expression, tracked) |
| tpu_v5p | 7 opts: `kernels/tpu/<opt>_tpu.py`; 4 base: `pallas/launch_<opt>.py` | `_pallas_fused.py` imports each; the 4 `kernels/tpu/<base>_tpu.py` are pure re-export shims | **NO** — one math home per opt; shims carry zero math |

**3 models (per arch):** sm_90 → `kernels/sm_90/<model>_sm90.cuh` (canonical CUTLASS), wrapped by `backends/cuda/sm_90/models/<model>.cuh`; gfx942 → `kernels/gfx942/<model>_gfx942.hip.hpp`; tpu → `_pallas_models.py`. **No duplicates.**

**Two known sub-expression re-expressions (NOT trees, documented):**
1. gfx942 `opt_components.hip.hpp` — transcription of the algorithm math (thrust constraint); already factored to a shared `sg_adam_tail` helper (9 uses) + cross-referenced.
2. `kernels/sm_90/{supergrok11,neuralgrok}_sm90.cuh` — each has ONE per-op fused apply kernel (`sg11_adam_decay_kernel`, etc.) that embeds the bias-corrected Adam tail with optimizer-specific gradient blending (e.g. `g = smart_grad + lamb_eff*mu`). These are the **per-op path's own kernel** (not a copy of `adamw_step` — the blend differs), so they are single-source for that path. Folding the bare 4-line tail into a shared helper is possible but (a) low gain (2 sites), (b) touches working CUTLASS-adjacent kernels, (c) **not runtime-behavior-provable here** → deferred, documented, NOT performed (no unsafe micro-merge).

## 0.3 Dependency graph

- **`setup.py` compiles:** CUDA = `csrc/bindings/*.cpp` + `csrc/backends/cuda/sm_90/*.cu` + `models/*.cu` + `csrc/fused/sm_90/*.cu` (49 srcs). HIP = `bindings` + `csrc/backends/hip/gfx942/*.hip.cpp` + `*.hip` + `csrc/fused/gfx942/*.hip` (46 srcs). Both verified 0-missing in Phase 5 V.2.
- **Fused cells `#include`:** sm_90 cell → `fused_megakernel.cuh` → `opt_components.cuh` (→ `csrc/algorithms/<opt>.h`) + `model_stages.cuh`. gfx942 cell → `fused_megakernel.hip.hpp` → `opt_components.hip.hpp` + `model_stages.hip.hpp`. NO demo includes, NO TEMPLATE_ONLY (grep = 0).
- **Bindings call:** `dispatch.cpp::fused_step` → `dispatch_sm90_cell` / `#if WITH_HIP dispatch_gfx942_cell`; the per-op `*_fused_step` bindings → `kernels/sm_90/<opt>_sm90.cuh` launchers → `algorithms/<opt>.h`.
- **Python imports:** `_pallas_fused.py` ← `kernels.tpu.<7 opts>` + `pallas.launch_<4 base>`; `megakernel_codegen.py` (doc ref to `kernels/tpu`); `launch_supergrok2.py`; `v5p/__init__.py`. Removing `kernels/tpu` would break all four → **must NOT delete** (confirms 0.2).

## 0.4 Migration order

Because there are no duplicate trees to merge, the "migration" reduces to
documentation + hygiene, each leaving self-test ≥141/0:

1. **(this commit) Stage 0** — `RESTRUCTURE_PLAN.md`. No code moved.
2. **Stage 2** — branch-prune list (read-only recommendation; no remote deletion without user sign-off).
3. **Stage 3** — README rewrite to document the (already-canonical) 44→99 layout.
4. **Stage V** — re-run the full system verification to confirm nothing regressed.

No file move/delete is scheduled, because Stage 0 found none that is both safe
and beneficial. If a future phase wants the `kernels/tpu` 7 + `pallas/launch` 4
unified into one directory for tidiness, the safe sequence is: (a) move the 4
base math into `kernels/tpu`, (b) make `pallas/launch_<base>` re-export from
there, (c) `--self-test` + TPU trace, (d) commit. That is a pure Python
relocation (importer-verifiable, no compile/hardware risk) — but it is cosmetic
(each opt already has exactly one math home) and is therefore left as an
explicitly-optional future item, not done here.

## 0.5 Rollback

Every step is a single revertible commit; nothing is force-pushed. If any step's
`--self-test` drops below 141/0 or any gate (`amdgcn_check.sh`,
`compile_to_object.sh`, the drift guard) fails, `git revert <commit>` restores
the prior green state (this Stage-0 commit moves no code, so it is itself a no-op
rollback point). Because no source is moved/deleted in this phase, there is no
dangling-reference risk to roll back from.
