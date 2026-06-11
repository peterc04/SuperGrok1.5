# Fused-megakernel component contract (portability spec)

Owner requirement (2026-06-09): every (model × optimizer × arch) configuration —
all 3 models × 11 optimizers — composes into ONE persistent megakernel binary,
and the building blocks are **individually portable**: a third project lifts the
headers and instantiates a fused trainer without dragging this repo along. The
meta-programming lives in headers; the dispatch layer is generated and is the
only composition seam.

## Component taxonomy (header-only, self-contained)

| component | header(s) | contract |
|---|---|---|
| **Substrate** | `csrc/fused/megakernel_common.cuh` | `PersistentContext` (task counter, grid barrier, generation), one-CTA-per-SM persistent launch idiom. Depends on CUDA runtime only. |
| **Optimizer math** | `csrc/algorithms/<opt>.h` (×11) | Pure per-element apply math (`<opt>_step(...)`), no CUDA dependencies beyond `__device__` qualifiers, no state allocation, no I/O. Single source of truth — everything else `#include`s these (enforced by `scripts/check_math_single_source.py`). Already portable as-is. |
| **Optimizer tails** | `csrc/fused/sm_90/opt_components.cuh` | `apply_optimizer<OptId>(FusedOptState&, FusedScalars const&, ...)` — templates composing the algorithm headers into megakernel stages. Consumes the documented POD structs only. |
| **Model stages** | `csrc/fused/sm_90/model_stage_<model>.cuh` (decoder / vit / mamba3 — ONE HEADER PER MODEL, not a monolith) | Provides `model_<m>_forward_stage<Dims>` and `model_<m>_backward_stage<Dims>`: batch-slice-local fwd/bwd with all activations CTA-local, writing per-CTA weight-grad partials to a caller-provided workspace. Templated on compile-time dims (d_model, n_layers, n_heads, seq, vocab). Includes ONLY the substrate header + CUDA. No globals, no repo build flags (every `SG_TUNED_*` consumed via `#ifndef` defaults). |
| **Composition** | `csrc/fused/sm_90/fused_megakernel.cuh` | `launch_fused_megakernel<ModelId, OptId, Tier>` — stage pipeline: fwd → bwd → grid barrier → deterministic weight-grad reduction → optimizer tail. The only place stages and tails meet. |
| **Cells** | `csrc/fused/sm_90/mega_<model>_<opt>.cu` (×33) | GENERATED 36-line instantiations (`megakernel_codegen.py`). Never hand-edited. |
| **Dispatch** | `csrc/bindings/dispatch.cpp` + generated `fused_wired_cells.inc` / `fused_dispatch_table.inc` | `(model, optimizer, arch) → cell` routing. Generated from the same enumeration as the cells, so it cannot drift. Loud-gated: unwired/unready cells throw, never silently degrade. |

## Portability rules (apply to every new/edited component)
1. **Self-containment**: a component header compiles with `#include <cuda_runtime.h>`
   + its declared includes only. Include-what-you-use; no transitive reliance.
2. **No hidden state**: no globals, no statics with side effects; all state flows
   through the documented POD structs (`PersistentContext`, `FusedOptState`,
   `FusedScalars`) or explicit pointers.
3. **Compile-time configuration only via documented template params or
   `#ifndef`-defaulted macros** (`SG_TUNED_*`); absent definitions must yield a
   correct (if untuned) kernel.
4. **The lift test**: copying `megakernel_common.cuh` + one `model_stage_*.cuh`
   + `opt_components.cuh` + the relevant `csrc/algorithms/*.h` into a fresh
   project and instantiating `launch_fused_megakernel<M,O,Tier>` must build and
   run. Each phase's validation includes compiling a minimal out-of-tree TU
   (tests/hw/test_component_portability.cu) that includes ONLY those headers.
5. **Generated artifacts are never edited by hand** — change the generator,
   regenerate everything, `git diff` must be generator-consistent.
6. **No placeholder math** on any wired path (owner no-suppression directive);
   surrogate stages exist only behind explicit unreadiness gates and throw at
   dispatch.

## Phase status

> **Boundary (per `HARDWARE_VALIDATION.md` §2, 2026-06-09):** the L3 fused
> model×optimizer megacells are **compile-verified only** on sm_90 — the
> on-silicon H100 race exercises the eager model + fused-*optimizer* (L1) path,
> not the L3 megakernel, so "validated vs eager" below remains the open gate for
> every phase. No phase is silicon-complete.

- Phase 1 (in flight): real decoder fwd+bwd stages. The per-model split has
  landed (`model_stage_decoder*.cuh` + `fused_decoder_megakernel.cuh` exist per
  this contract); the **validated-vs-eager** gate is still open — these stages
  are compile-verified only (§2), not yet runtime/numeric-checked on silicon.
- Phase 2: vit + mamba3 stage headers (`model_stage_{vit,mamba3}*.cuh` +
  `fused_{vit,mamba}_megakernel.cuh` present, built against this contract
  natively); same compile-verified-only boundary applies.
- Phase 3: optimizer precompute stages in-kernel for the 9 non-trivial tails;
  SuperGrok2's CSA/HCA/PEER/GRU meta as in-kernel stages (the full
  "vit×sg2 in one binary").
