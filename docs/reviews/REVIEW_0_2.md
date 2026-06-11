# REVIEW_0_2 — Stage-0 header de-inline + Stage-2 L2 persistence

Correctness re-review (Opus 4.8, max effort, 2026-06-01) of branch
`claude/custom-optimizer-analysis-HFYhg`. Read-mostly; fix only clear bugs.
Prior work done partly off the pinned model — re-verified rigorously from git
history and the live tree.

**Verdict summary: NO BUGS FOUND. Both stages CORRECT. Zero code changes made.**

Gate after review (no edits, baseline re-confirmed):
- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/launch_adamw.cu` → **COMPILE_OK**
- `python -m grokking_optimizers.compile --self-test` → **137 passed, 1 failed**
  (the failure is the pre-existing `flag_base_superset_regression`; 0 net regressions)
- `ruff check grokking_optimizers/` → **All checks passed!** (scoped lint of the
  reviewed launcher tree; see ruff note below)

---

## Part A — Stage 0 header de-inline (pure refactor)

The de-inline landed in commit `bed06aa` ("hoist platform defines for real-nvcc
compilation"). It created the canonical `#pragma once` headers
`csrc/common/{platform.h,types.h,utils.cuh}` (NEW in that commit) and re-created
`csrc/scan/affine2x2.h` (originally added in `817ab77`, deleted in `9804937`'s
"full inlining" pass, restored here). Stage-3.0 follow-up `bc91f05` removed dead
hand-PTX.

### A1 — No behavioral change (byte/semantic identity) → CORRECT
Verified the canonical bodies against the pre-de-inline *inlined* copies
(`bed06aa~1:csrc/algorithms/adamw.h`):
- `types.h` constants — **identical names AND values** (MAX_D_STATE=128,
  MAX_D_INNER=128, MAX_D_MODEL=64, MAX_GRU_HIDDEN=8, MAX_EXPERT_HIDDEN=16,
  MAX_TOPK=4, MAX_CKPT_INTERVAL=32, SG2M_BLOCK=256, SG2B_BLOCK=256,
  PSCAN_BLOCK=512, PSCAN_THRESHOLD=256, GEMM_PRECOMPUTE_THRESHOLD=1024).
  `csrc/common/types.h:24-36`.
- `utils.cuh::warp_reduce_sum` — `diff` against pre-de-inline body: **byte-identical**.
- `affine2x2.h::affine_combine` (12-FMA inline-PTX block) — **byte-identical** to
  the live scan primitive; `csrc/scan/affine2x2.h:36-73`.
- `platform.h::stream_load/stream_store/*4` — present and unchanged.
- **No DROPPED / DUPLICATED function**: each canonical symbol is defined in
  EXACTLY ONE file (grep across `*.cu|*.cuh|*.h`): `MAX_D_STATE` → only
  `types.h`; `warp_reduce_sum(float,int,int)` → only `utils.cuh`;
  `#define WARP_SIZE` → only `platform.h`; `struct Affine2x2` → only
  `affine2x2.h`. No residual text-inlined copy survives anywhere.
- The two similarly-named INT8 helpers are NOT duplicates:
  `types.h::float_to_int8_stochastic_branchless` (line 56) and
  `utils.cuh::float_to_int8_stochastic` (line 72) are distinct functions with
  distinct bodies; both existed pre-de-inline.

### A2 — Include topology acyclic + complete (WARP_SIZE-before-def) → CORRECT
- `platform.h` — no local deps; **DEFINES** WARP_SIZE (l.79), GROK_CUDA/HIP
  (l.25-31), FULL_WARP_MASK (l.167), SHFL_DOWN_SYNC (l.96), stream_load/store.
- `affine2x2.h` — `#include platform.h` (l.5); uses GROK_CUDA (l.41). ✓ prereq present.
- `types.h` — `#include platform.h, affine2x2.h` (l.5-6). ✓
- `utils.cuh` — `#include platform.h, types.h, affine2x2.h` (l.5-7); uses
  WARP_SIZE/FULL_WARP_MASK (l.31), SHFL_DOWN_SYNC (l.33) — all from the included
  platform.h. ✓ **The original "WARP_SIZE used before defined" bug is fixed**:
  every header that USES a platform macro INCLUDES platform.h first.
- `primitives.cuh`, `mma.cuh`, `mamba_scan_adapter.cuh` — each
  `#include {platform,types,affine2x2,utils}` in that order (l.5-8).
- **No cycle**: platform (leaf) ← affine2x2 ← types ← utils ← primitives/mma/
  adapter. types includes BOTH platform and affine2x2 directly; `#pragma once` +
  platform-first ordering makes the diamond safe.

### A3 — #pragma once → CORRECT
All 8 canonical headers (`platform.h`, `types.h`, `affine2x2.h`, `utils.cuh`,
`sm_90/primitives.cuh`, `sm_90/mma.cuh`, `scan/mamba_scan_adapter.cuh`,
`hip/gfx942/primitives.hpp`) have `#pragma once` as line 1. No ODR risk.

### A4 — Stage-3.0 dead-code deletion → CORRECT
- `csrc/common/ptx_intrinsics.cuh` is deleted (file absent; only a tombstone
  comment in `primitives.cuh:9` references it). No `#include` of it survives.
- The 3 functions removed from `utils.cuh` (`ptx_log2`, `ptx_affine_combine`,
  `ptx_expert_mlp_forward`) have **0 real call sites** — grep finds only tombstone
  comments and the `STAGE3_PTX_AUDIT.md` audit doc; no compiled reference.
- **KEPT functions all still defined AND used** (call-site counts, excluding
  defs/audit):
  - `ptx_exp2` (utils.cuh:114) — private base, used by expf/tanhf/sigmoidf
    internally (l.125/131/138). Kept correctly.
  - `ptx_expf` — 31 sites · `ptx_tanhf` — 2 · `ptx_sigmoidf` — 2 ·
    `ptx_fma` — 6 · `fast_rsqrt_nr` — 6 ·
    `ptx_int8_stochastic_round` — 1 (grokadamw_sm90.cuh:160) ·
    `affine_combine` (affine2x2.h, live 12-FMA scan primitive) — used by
    mamba_scan_adapter, supergrok2, types.h re-export.
  All defined, all live. No live function was deleted.

---

## Part B — Stage 2 L2 persistence (`prim::L2PersistScope`)

Definition: `csrc/backends/cuda/sm_90/primitives.cuh:197-278`.

### B1 — RAII helper is correct CUDA runtime API → CORRECT
Verified field/enum names against the CUDA runtime-API reference:
- Union type `cudaStreamAttrValue`, member `accessPolicyWindow` of type
  `cudaAccessPolicyWindow` with fields `base_ptr` (void*), `num_bytes` (size_t),
  `hitRatio` (float), `hitProp`/`missProp` (cudaAccessProperty) — **all match**
  (l.236-241).
- Enum `cudaStreamAttributeAccessPolicyWindow` passed to
  `cudaStreamSetAttribute(stream, attr_id, &val)` — correct (l.242-243).
- `cudaAccessPropertyPersisting/Streaming/Normal` — correct. `missProp` is
  Streaming on set (valid; miss must be Normal|Streaming, never Persisting) and
  Normal on reset (valid).
- **Destructor reset is complete and leak-free** (l.252-267): (a) resets the
  stream window to num_bytes=0 / Normal props → disables persistence for future
  ops on the stream; (b) `cudaCtxResetPersistingL2Cache()` (correct `void`
  signature) → demotes currently-persisting lines to normal; (c)
  `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0)` → releases the reserved
  carve-out. The dtor only runs its body when `active_==true`, i.e. exactly when
  construction completed all three side-effects — so the reset mirrors the setup.
  No leak.

### B2 — Gating safe + correctly ordered → CORRECT
Constructor short-circuits, IN ORDER, BEFORE the first `cudaDeviceSetLimit`
side-effect (l.234):
1. `ENABLE_L2_PERSIST==0` → whole body `#if`-compiled out (l.204/247).
2. `p0==nullptr || n0==0` → return (l.205).
3. `cudaGetDevice` fail → return (l.208).
4. `cc_major < 9` → return (l.212). (Locals default 0, so a failed
   `cudaDeviceGetAttribute` also returns safely.)
5. `max_persist <= 0` → return (l.218).
6. `span==0 || span > max_persist` → return (l.231).
Only after all six does it reserve + set the window. Order is correct; no
side-effect happens before a gate can veto it.

### B3 — Span over non-adjacent buffers → CORRECT (efficiency-only caveat, documented)
`span = max(end) - min(start)` over the ≤2 buffers (l.221-229). If the two
buffers are NON-adjacent, the window over-covers the gap (unrelated memory).
**This is NOT a correctness problem**: a persisting access-policy window only
changes L2 *eviction priority* for lines touched within it — it never moves,
re-orders, or alters data. Over-covering only wastes carve-out capacity / dilutes
hit-rate (an efficiency footnote). Moreover, when buffers are far apart the span
typically exceeds `max_persist` and gate #6 no-ops the whole scope — so the
practical effect is "persistence simply disabled", which is safe. `hi - lo` is a
`char*` difference cast to `size_t`; since `lo` is always the min and `hi` the
max, the difference is non-negative — no underflow.

### B4 — Pure HINT, numerics-invariant → CORRECT (key safety property holds)
The helper's ONLY runtime calls are `cudaGetDevice`, `cudaDeviceGetAttribute`,
`cudaDeviceSetLimit`, `cudaStreamSetAttribute`, `cudaCtxResetPersistingL2Cache`.
NONE read or write tensor data — it sets a cache *policy* only. Persisting L2 is
a residency hint; it cannot change kernel outputs. Numerics-invariant confirmed.

### B5 — Wiring into the 11 launchers → CORRECT
All 11 optimizer launchers in `grokking_optimizers/kernels/sm_90/` construct the
scope **immediately after** the `auto stream = ...getCurrentCUDAStream()` line,
covering that optimizer's real, in-scope hot state-buffer parameters:

| launcher | line | buffers persisted (verified real params + hot state) |
|----------|------|------------------------------------------------------|
| adamw        | 106 | exp_avg (m) + exp_avg_sq (v) |
| grokadamw    |  75 | exp_avg + exp_avg_sq |
| grokfast     |  76 | exp_avg + exp_avg_sq |
| neuralgrok   |  91 | exp_avg + exp_avg_sq |
| prodigy      | 116 | exp_avg + exp_avg_sq |
| supergrok11  | 117 | exp_avg + exp_avg_sq |
| supergrok15  | 109 | exp_avg + exp_avg_sq |
| supergrok2   |1387 | exp_avg + exp_avg_sq (CSA/HCA step) |
| looksam      | 168 | exp_avg + exp_avg_sq (SAM apply step) |
| lion         |  79 | exp_avg (single momentum — verified real param of `launch_lion_step`) |
| muon         |  94 | buf (momentum — verified real param of `launch_muon_momentum_normalize`) |

Spot-checked signatures: Lion's `exp_avg` and Muon's `buf` are genuine function
parameters and are each optimizer's actual hot momentum state. All 11 users
`#include csrc/backends/cuda/sm_90/primitives.cuh` and alias
`namespace prim = ::sg::sm90::primitives;` — `prim::L2PersistScope` resolves.

---

## Ruff note (scope)
Repo-wide `ruff check .` reports 372 errors, but **all of them are pre-existing
and outside this review's scope**: 370 in `grokking_race_v2.py` (E701/E702
multi-statement lines), 1 in `bench_backends.py`, 1 in `scripts/stage0_deinline.py`
(E401). `ruff.toml` excludes `csrc/`. The reviewed launcher tree
(`grokking_optimizers/`) is **ruff-clean**. The working tree is unmodified by this
review, so no new lint was introduced. (Not fixed: those files are owned elsewhere
and the task is scoped to the de-inline + L2 work.)

## Fixes applied
None. No clear bug was found in either stage.
