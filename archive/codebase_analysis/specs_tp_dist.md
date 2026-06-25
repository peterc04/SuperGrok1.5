# specs_tp_dist.md — Digest: TP/Distributed/Adaptive-Parallelism Spec Files

Agent: specs_tp_dist subagent
Source files: /workspace/impl_diffs/{tp_kernel,tp_nvshmem,dist_step,run_harness,adaptive_parallelism,resource_fit_planner,memory_strategy,size_adaptive}.md
All 8 files read in full.

---

## 1. VERIFIED LIVE STATE (from adaptive_parallelism.md §0)

These edits are ALREADY APPLIED in the on-disk codebase as of the most recent commits:

- **Megakernel IS templated** `template <OptId Opt, class Par=par::SingleGPU>` with trailing `CommCtx comm={}`: `fused_decoder_megakernel.cuh:674-681`
- **ParConfig IS 5-param** `<DP,TP,PP,SP,ZeROStage Z>`: `parallel_config.cuh:55`
- **CommCtx IS widened** with tp_sym_heap/tp_heap_stride_floats/tp_team_local_pe/tp_team_n_pes: `parallel_config.cuh:106-138`
- **NvshmemTransport IS hardened** (team-scoped, nvshmem_quiet, make_transport_from_comm): `tp_transport.cuh:168-277`
- **Python distributed.py HAS FULL EP plumbing**: `distributed.py:64-128,228-292,403-418,491-555`

---

## 2. CONFIG-DERIVATION MECHANISM (the central design thesis)

### 2A. infer_parallel_config() — auto_config.py (EDIT C of adaptive_parallelism.md)

NEW file: `grokking_optimizers/parallel/auto_config.py`

Pure Python. Takes `(model_cfg: dict, hw_cfg: dict) -> dict` with keys matching ParConfig template params.

**Decision rules (in order):**

1. **TP**: largest power-of-2 dividing `num_gpus`, bounded by `hw_cfg["nvlink_width"]`, subject to `model_cfg["d"] % TP == 0`. Default=1 (no NVLink).
2. **PP**: only when `(params_per_tp_rank > usable_hbm) and (L % PP == 0)`. PP fills the depth gap after TP. Default=1.
3. **DP**: `num_gpus // (TP * PP)`. Fills remainder.
4. **SP**: pinned=1 for this campaign (sequence models are SP-eligible — decoder/vit/mamba — but SP is deferred; the field exists for future use).
5. **EP**: only for model-level MoE (`num_experts > 1` or `is_moe`). Current 3 models have no model experts → EP=1 always. Note: sg2_num_experts (optimizer's PEER) does NOT trigger EP.

### 2B. plan_execution() — resource_planner.py (NEW, resource_fit_planner.md)

NEW file: `grokking_optimizers/parallel/resource_planner.py`

Signature: `plan_execution(model_cfg: dict, hw_cfg: dict) -> ExecutionPlan`

**NEVER branches on GPU count.** Branches on `fit(footprint) <= usable_hbm`.

**Memory escalation ladder (R0 → R5):**
- R0: in-HBM (no extra strategy); try first
- R1: ZeRO-3 sharding over DP
- R1b: raise PP (when R1 still doesn't fit and L%PP==0)
- R2: CTA-tile cap (auto_ncta ladder)
- R3: activation recompute
- R4: layer streaming (pinned-host weights + ring of kStreamDepth device slots)
- R5: host offload (element-local opts only — NOT Prodigy/Muon/SG2)

**per_rank_budget():** params + state + acts + staged_opt. TP shrinks Nmax; ZeRO-3 shards over DP.

**infer_mesh():** TP=largest pow2 dividing num_gpus bounded by nvlink_width AND d%TP==0; PP only when needed and L%PP==0; DP fills rest; EP sub-divides DP for MoE.

**emit_compile_flags():** exact -D flags: `-DSG_FLAGSHIP_TP/PP/DP/ZERO`, `-DSG_DEC_BENCH_LAYOUT`, `-DSG_DEC_RECOMPUTE`, `-DSG_DEC_LAYER_STREAM`, `-DSG_DEC_HOST_OFFLOAD`

**_template_inst():** `launch_fused_decoder_megakernel_tc<Opt, ParConfig<DP,TP,PP,1,Z>>`

**Worked examples (verified numerically):**
- 10M/1GPU: trivial, R0 in-HBM, all defaults
- 1.5B/8GPU: → TP=8, R0 (fits), no offload
- 10B/1GPU: → full stack (R5 offload) — SG2 on 1 GPU is structurally unfittable; planner downgrades to adamw
- 10B/8GPU: → TP=8, R0 no offload

---

## 3. ADAPTIVE 3D→5D PARALLELISM LOGIC

ParConfig template (already live): `template <int DP, int TP, int PP, int SP, ZeROStage Z>`

EP as 6th trailing defaulted param (EDIT A, adaptive_parallelism.md): `template <int DP, int TP, int PP, int SP, ZeROStage Z, int EP=1>`

**Derived gate predicates (all static constexpr):**
- `kTPComm = (TP > 1)` — enables in-kernel NVSHMEM TP all-reduce
- `kEPComm = (EP > 1)` — enables EP MoE routing (future)
- `kIsSingleGPU = (DP==1 && TP==1 && PP==1 && SP==1 && Z==ZeROStage::Z0 && EP==1)`

**TP reduce points (4 in fwd+bwd):**
1. out_proj fwd (row-parallel, all-reduce after)
2. ff2 fwd (row-parallel, all-reduce after)
2'. ff0-dX bwd
1'. in_proj-dX bwd

**All-reduce implementation:** `tp_allreduce_sum_fixed_order` — ascending-pe fp32 fixed-order reduce for A/A/A determinism. NOT `nvshmemx_float_sum_reduce` (unspecified order → ULP drift).

**Grid-lockstep P1 constraint (CRITICAL):** On kTPComm path, ALL CTAs on a GPU process the SAME tile each round to avoid deadlock with `tr.rendezvous(bar)`. Naïve per-CTA rendezvous deadlocks.

---

## 4. SIZE-ADAPTIVE CTA-TILING SELECTOR (size_adaptive.md)

### SizeConfig template (EDIT A, parallel_config.cuh):

```cpp
template <bool CtaTile, int CtasPerTile, int ClusterDim, int TileN>
struct SizeConfig { ... };
using SizeSmall = SizeConfig<false, 1, 1, SG_TUNED_TILE_N>;  // byte-identical default
using SizeLarge = SizeConfig<true,  2, 2, SG_TUNED_TILE_N>;  // CTA-tiling ON
```

**Second template axis** on the megakernel, alongside Par:
- `template <OptId Opt, class Par=SingleGPU, class Sz=SizeSmall>`
- Orthogonal: `<Opt,ParTP8,SizeLarge>` = TP all-reduce + CTA-tiling; `<Opt,SingleGPU,SizeSmall>` = today

### CTA-tiling decision rule (_dec_is_large predicate):

LARGE tier (CTA-tiling ON) when either:
1. **WIDTH**: `d >= 1024` — in_proj/ff GEMM N-range wide enough that 2 CTAs per tile each get a full wgmma N-tile
2. **GRID UNDER-FILL**: `ceil(T/TILE_M) < n_sms` — fewer token tiles than SMs → idle SMs → grid barrier waits on slowest

SMALL otherwise (d=128 production race: d<1024 AND enough tiles to fill grid → persistent 1-CTA/SM shape wins).

**Named tiers:**
- production (d=128): SMALL → par::SizeSmall, knobs = live defaults
- bench (d=2048): LARGE → par::SizeLarge (d>=1024), knobs = live defaults (this increment)
- flagship (d=1600,L=48): LARGE → par::SizeLarge (d>=1024), knobs = live defaults

### auto_ncta ladder:

132 → 64 → 32 → 16 → 8 → 4 → 2 → 1 (picks largest that fits HBM budget)

**sg2_ws_stride = ~91.277×Nmax floats/CTA** (verified numerically in run_harness.md; NOT the "~50" header comment estimate).

### Budget at flagship d=1600, L=48, 8×H100:

usable HBM = 80×1000³/1024³ − 4.0 = ~70.5 GiB

| Config | HBM | Status |
|--------|-----|--------|
| TP8/nCTA=132/adamw | 66.39 GiB | FITS |
| TP8/nCTA=132/sg2   | 70.52 GiB | OOM (0.01 over) |
| TP8/nCTA=64/sg2    | 40.92 GiB | FITS (auto-cap) |

**Recommendation:** TP8×DP1×PP1+ZeRO-3; 10 opts at nCTA=132, SG2 auto-caps to nCTA=64.

---

## 5. MEMORY STRATEGY GATES (memory_strategy.md)

### What EXISTS on disk (NOT spec'd, already there):
- ZeRO-3 (`zero3.py`)
- nCTA cap / auto_ncta
- Full DecActs materialization (all-layer acts in HBM, no checkpoint boundary)
- All-layer bf16 weights pre-staged via DecWBf (no streaming)
- NO host offload in fused kernel, NO recompute, NO streaming

### MemConfig template (EDIT A2, mem_config.cuh — NEW file):

```cpp
template <bool OffloadOpt, bool RecomputeActs, bool StreamLayers, int StreamDepth>
struct MemConfig { ... };
using InHbm = MemConfig<false, false, false, 0>;  // byte-identical default
```

**Gate thresholds:** all keyed on `fit(footprint) <= usable_hbm` (70.5 GiB for H100). No hardcoded GPU count.

### Memory strategy options:

| Option | Gate | Mechanism | Constraint |
|--------|------|-----------|------------|
| (A) In-HBM | default (R0) | no change | baseline |
| (B) ZeRO-3 | R1 | full-grad reduce-scatter over DP | DistStepContext sharding |
| (C) Host offload | R5 | element-local OptIds (NOT Prodigy/Muon/SG2); cudaMemcpyAsync tile staging | state→pinned host |
| (D) Act recompute | R3 | store only X_in[L] (1/16 of full acts at dff=4d); recompute fwd in bwd | touches hot bwd tile |
| (E) Layer streaming | R4 | pinned host weights + ring of kStreamDepth device slots | HARDEST; partially breaks single-launch invariant |

**Bug fix (dist_step.md):** ZeRO-3 full-grad all-reduce is 47 GB at flagship×8 — needs replace with `fixed_order_reduce_scatter_grad`.

---

## 6. PENDING SPEC EDITS (not yet committed, status per spec)

### Apply-ready NOW (pure Python / CPU-compilable POD):
- `auto_config.py` → `infer_parallel_config()` (EDIT C, adaptive_parallelism.md)
- `resource_planner.py` → `plan_execution()` (EDIT, resource_fit_planner.md)
- `mem_strategy.py` → MemConfig + escalation logic (EDIT A1, memory_strategy.md)
- `mem_config.cuh` → MemConfig template (EDIT A2, memory_strategy.md)
- `parallel_config.cuh` → EP as 6th param (EDIT A, adaptive_parallelism.md)
- `parallel_config.cuh` → CommCtx EP fields (EDIT B, adaptive_parallelism.md)
- `parallel_config.cuh` → SizeConfig + SizeSmall/SizeLarge aliases (EDIT A, size_adaptive.md)
- `parallel/__init__.py` → export auto_config + resource_planner (EDIT D, HOOK 1)
- `megakernel_codegen.py` → decoder_knobs_for_size() + DEC_SIZE_TIERS + CLI (EDIT B, size_adaptive.md)
- `compile.py` → size_tier dim + _decoder_size_tier_pins() (EDIT D, size_adaptive.md)
- `zero3.py` cold-start shard device default fix (dist_step.md)
- `test_distributed_step.py` _PARITY_TOL: 3e-5 → 5e-4 (dist_step.md)
- `tests/test_resource_planner.py` (NEW, resource_fit_planner.md)
- `tests/test_parallel_instantiation.py` EP point allow-list extension (EDIT E, adaptive_parallelism.md)
- `tuning/flagship_distributed.py` (NEW harness, run_harness.md)
- `grokking_optimizers/parallel/flagship_budget.py` (NEW, run_harness.md)

### Kernel-track (require GPU build/validate loop):
- `fused_decoder_megakernel.cuh` → `<class Sz=SizeSmall>` template param + P1 CTA-tile gate seam + launcher shape selector (EDIT C, size_adaptive.md)
- `fused_decoder_megakernel.cuh` → 4 TP reduce points + grid-lockstep P1 restructure (EDIT C/D, tp_kernel.md)
- `model_stage_decoder_tc.cuh` → `<Par,Transport>` threading + 4 reduce-point inserts (EDIT D, tp_kernel.md)
- `mega_decoder_real_adamw_tc_launcher.cu` → symmetric heap allocator split + CommCtx population + TP dispatch (EDIT E, tp_kernel.md)

### Scoped (deferred, NOT byte-exact — require GPU window):
- §7 CTA-tiled (occupancy>1/cluster) megakernel variant — barrier, tile ownership, smem (size_adaptive.md §7)
- DistStepContext 4D extension (dist_step.md §6)
- reduce-scatter-grad replacing full all-reduce (dist_step.md §6.B)

---

## 7. NVSHMEM STATUS RECONCILIATION

- **tp_nvshmem.md §0 (OLDER):** Claims NVSHMEM NOT installed.
- **tp_kernel.md §0 (NEWER, authoritative):** NVSHMEM 3.7.0 IS installed at `/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem`. `nvshmemx_barrier_block` present in 3.7.0 (resolves §3.1 follow-up in tp_nvshmem.md).

**Symmetric heap:** nvshmem_malloc'd memory required for NVSHMEM reduces. Plain cudaMalloc workspace NOT addressable via nvshmem_ptr. ~26 MB/PE at flagship TP=8 (NOT the 216 MB loopback figure).

---

## 8. KEY DISCREPANCIES FOUND

1. **tp_nvshmem.md §0 vs tp_kernel.md §0:** NVSHMEM installed vs not. tp_kernel.md is authoritative.
2. **resource_fit_planner.md internal note:** Claims "adaptive_parallelism.md / size_adaptive.md are NOT present in /workspace/impl_diffs" — WRONG. Both files exist there. Likely written before those files were placed.
3. **sg2_ws_stride header comment "~50":** The run_harness.md numerically verifies it is ~91.277×Nmax floats/CTA. Header comment is stale.

---

## 9. COMMIT CROSS-CHECK

| Commit | Description | Status |
|--------|-------------|--------|
| 5733af5 | TP foundation (megakernel templated, CommCtx base) | APPLIED — confirmed by adaptive_parallelism.md §0 live state checks |
| 531d87e | tp_remainder (NvshmemTransport hardened, make_transport_from_comm) | APPLIED — tp_transport.cuh:168-277 confirmed live |
| 81f1bfb | resource_planner | SPEC WRITTEN, apply status unverified (new file not yet on disk at time of spec) |
| edec531 | memory_strategy | SPEC WRITTEN, apply status unverified |
| c1230dc | ep_size (EP as 6th param, parallel_config.cuh) | SPEC WRITTEN, apply status unverified |

The 3 "SPEC WRITTEN" commits may or may not have landed; the adaptive_parallelism.md §0 only verifies through commit 531d87e.
