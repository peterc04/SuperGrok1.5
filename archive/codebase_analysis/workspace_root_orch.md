# Workspace Root Orchestration Digest
**SuperGrok2 / SuperGrok1.5 — Top-Level Ledger + Orchestration Scripts**
**Analysis date:** 2026-06-25 (session-teleport reconstruction)

Files fully read: PROGRESS.md, PHASE0_CONTEXT.md, PHASE0_CONTEXT_v2.md, COMPILE_RECONCILE.md,
LEDGER.json, impl_workflow.js, phase_drafts.js, phase0c_read.js, phase0c_gapfill.js,
plus inspection of: phase6/tp_datapath_fix_WIP.patch, phase6/flagship_11opt_ranking.txt,
W_workspace_root.txt (group index).

---

## 1. What These Files Are

The `/workspace` root files (outside the repo `/workspace/SuperGrok1.5`) are the
**campaign orchestration layer** — the persistent ledger, context, and agent-spawning
scripts for the 2026-06-24/25 "SuperGrok2 Hardening Run" conducted after a session
teleport onto an 8xH100 node.

| File | Role |
|---|---|
| `PROGRESS.md` | Human-readable running ledger: phase status, validated results, blockers, event log |
| `PHASE0_CONTEXT.md` | Phase 0 v1 output: full contradiction list (C1–C15), env, architecture, bugs, phase implications |
| `PHASE0_CONTEXT_v2.md` | Phase 0 v2 addendum: upgraded-L3-TC clarification, prebuilt artifact map, owner priorities |
| `COMPILE_RECONCILE.md` | Phase-2/Step-0 gate: status of every compile.py capability claim (PRESENT / PARTIAL / ABSENT) |
| `LEDGER.json` | Machine-readable state snapshot (HEAD c29ed4e, written 2026-06-24T21:40Z — stale by session-end) |
| `impl_workflow.js` | 5-agent parallel implementation workflow: each agent emits exact edit specs to `/workspace/impl_diffs/` |
| `phase_drafts.js` | 9-agent Phase 2/3/4/5 design+draft workflow (datasets, compile additions, verify harness, dead code) |
| `phase0c_read.js` | ~65-agent exhaustive cover-to-cover mega-read: compile.py in 14 shards, all CUDA/C++/Python, prebuilt binary disasm, 8-shard session archive mining |
| `phase0c_gapfill.js` | 8-bundled-agent re-run of rate-limit-failed areas from phase0c_read |

---

## 2. Phase Plan (as extracted from PROGRESS.md + PHASE0_CONTEXT.md)

The campaign had 7 planned phases, of which Phase 0 was completed and heavily expanded:

| Phase | Title | Status per PROGRESS.md |
|---|---|---|
| 0 | Codebase reread + contradiction sweep | **REDO (exhaustive)**: v1 was grep-skimmed; relaunched 80 agents (~11+69) for literal cover-to-cover read incl. prebuilt binary disasm + session archive mining. v1 deliverables (PHASE0_CONTEXT.md + COMPILE_RECONCILE.md) stand; v2 (PHASE0_CONTEXT_v2.md) adds upgraded-kernel corrections |
| 1 | Megakernel baseline (measure-only) | pending (gate: Phase 0 done) |
| 2 | Compile-file reconcile/harden/extend | pending (gate: Phase 0 done) |
| 3 | Dataset integration | pending |
| 4 | Per-function silicon verification | pending |
| 5 | Cleanup (bugs + dead code) | pending |
| 6 | Full profiling + roofline + final report | pending |

However, PROGRESS.md documents that the campaign ran far beyond Phase 0. By session end (2026-06-25)
large portions of Phases 1/2/4/5/6 were actually completed and validated, as documented in the event
log and milestone sections of PROGRESS.md. The formal phase status table was not updated to reflect
this — the milestone sections are the authoritative completion record.

---

## 3. The Orchestration Workflow Architecture

### phase0c_read.js (primary mega-read)
- **65 parallel agents** spawned via `parallel()` harness
- `compile.py` split into **14 line-range shards** (each ~2350 lines of the 32,900-line file)
- **Source-file agents** (30+): every CUDA kernel, Python module, test, doc, tuning script
- **Binary/artifact agents**: `_ops.so` disasm, `_dectc_codegen` variants, `task11_bench_build`
- **Session-archive agents**: 8 shards of `claude_session_archive/` mining for high-value terms
- Output: per-area digests to `/workspace/phase0c/` (confirmed ~53 .md files present)
- Schema: `{area, files_fully_read, top_findings, bugs, dead_code, digest_path}`

### phase0c_gapfill.js (gap-fill for rate-limited failures)
- **8 bundled agents** (low concurrency) for areas that hit rate limits
- 7 bundles: kernels-core, substrate, opt-tail, opt-py+bindings, tests, harness+profile, docs, tuning+scripts+preserve
- Same output schema but `digest_paths` (plural array)
- Filled the 33 areas that didn't complete in the initial mega-read

### phase_drafts.js (design/draft for Phases 2/3/4/5)
- **9 parallel agents** each writing design + draft code/diff to `/workspace/`
- Explicitly READ-ONLY on main repo; lead applies vetted drafts
- Areas: FineWeb decoder, ImageNet ViT, GiftEval Mamba, shared harness changes, L2-persistence dim, smem carveout, negative cache + bugs, verification harness, dead-code removal
- Schema: `{phase, area, design_summary, draft_path, files_to_change, risks}`

### impl_workflow.js (apply-ready edit specs)
- **5 parallel agents** producing exact-edit specs to `/workspace/impl_diffs/`
- Areas: compile.py (bugfixes + S3.4/S1.4/S1.5), CuTe GEMM rewrite (wgmma.cuh), flagship codegen, datasets Layer A + dead-code, Phase-4 verification harness
- Schema: `{area, spec_path, files_changed, gate_commands, confidence, risks}`
- Rules: exact-match OLD/NEW blocks, byte-identical-at-default for new knobs, gfx942/tpu_v6e preserved, no SuperGrok-hardcoding

---

## 4. Contradiction List C1–C15 (from PHASE0_CONTEXT.md — codebase wins)

These were the full set of contradictions found between the campaign prompt assumptions and the live code:

| # | Prompt assertion | Live code reality |
|---|---|---|
| **C1** | "33 megakernels / 33 cells" = 33 fused binaries | 3 real per-model `_tc` kernels over `OptId`; the 33 generated per-cell `.cu` are dead (missing `fused_megakernel.cuh`, unrouted, won't compile). `dispatch.cpp:240-243` |
| **C2** | "fwd→barrier→bwd→barrier→opt" phase layout | **P0/B0/P1(fwd+bwd fused)/B1/P2(reduce)/B2/P3(opt)** — fwd&bwd share P1; a reduce phase exists. `fused_decoder_megakernel.cuh:18-37` |
| **C3** | "no HBM round-trip" for fused intermediates | Grad partials reduced THROUGH HBM in P2; only inter-launch + grad-materialization round-trips removed. `fused_decoder_megakernel.cuh:207,257-261` |
| **C4** | EarlyStopper triggers on 95% TEST accuracy | Stops on **VAL** (`early_stop_on='val'`); test is post-hoc. `grokking_race_v2.py:269,690` |
| **C5** | mod-97 "~4656 samples" | 4656 = test split at 50/50; full datasets are 9312/9409. `grokking_race_v2.py:306,331,2428` |
| **C6** | "step budget e.g. 15k" | Configured budget is **20,000** (`early_stop_max_steps`). `grokking_race_v2.py:2314,2405` |
| **C7** | `MultiGPUTimingPool` is the cross-GPU timing pool for 33 cells | It shards **autotune variants of one cell**; no 33-across-8 scheduler exists. `compile.py:4874,4909,16382` |
| **C8** | Parallelism "already in codebase" implies a cell-fanout scheduler | None exists; training DP/TP/PP+ZeRO is tests-only; `.parallelism_design.md:3` is design-only |
| **C9** | WGMMA/TMA is the megakernel substrate | In-kernel substrate uses **cp.async**, not TMA; TMA only in CUTLASS collectives; Mamba3 stage is scalar. `tile_pipeline.cuh:199-204` |
| **C10** | FP4/NVFP4/MXFP/FP8 precision present | Only FP8/FP4 autotuner **dims**; NVFP4/MXFP absent; **FP4 Blackwell-only → inactive on sm_90a**; no FP8/FP4 in kernels. `compile.py:547-548,1918,1922` |
| **C11** | Stage-1/3 ADD items are ABSENT | Mostly PARTIAL/present — see COMPILE_RECONCILE.md |
| **C12** | "move sm_90 yellow→green" implies undone | Campaign files show sm_90 L3 already silicon-validated; `HARDWARE_VALIDATION.md` 🟡 is stale doc-drift |
| **C13** | `SG_TUNED_{DEC,VIT,MB}_DW_SPLITK` all live | **MB split-K REMOVED** 2026-06-17. `compile.py:2225-2228` |
| **C14** | Drift guard protects `.py`↔`.h` | Guard hashes only `.h`; SG11/15/2 `.py` un-guarded drift surface; GrokAdamW EMA re-inline uncaught. `check_math_single_source.py:195-201,251-254` |
| **C15** | Archived `PHASE3_REPORT.md` = prompt's Phase 3 | Those are OLD restructuring phases, not the prompt's Phase 3 (datasets) |

---

## 5. Verified Bugs (from PHASE0_CONTEXT.md and v2)

1. **Dead `device_profiling` import** (`compile.py:~17652`): `from grokking_optimizers.device_profiling import run_device_pgo_round` → `ModuleNotFoundError` (verified live). Device-PGO round hook is a no-op or crash.
2. **`_MMA_NATIVE_LOADS_WIRED=False`** (`compile.py:30179`): compile.py's synthetic wgmma PTX never reaches an artifact. Only scalar fallback generated by the autotuner.
3. **Inert ABI guard**: `GROK_ABI_SCHEMA=1` exported in `bindings.cpp:115` but no Python-side `__abi_schema__` assertion → stale `.so` / new wrapper mismatch silently mis-marshals.
4. **`profile_maximal` 11/0 only with allowlist**: raw run is 6P/5F/9SKIP where all 5 fails = ONE runtime-dead CUTLASS TF32-RS GEMM spilling 8B.
5. **Dead code**: in-file `MambaModel`/`SelectiveSSMLayer` (`grokking_race_v2.py:434-489`), `_maybe_wrap_cuda_graph` no-op shim (`:895`).
6. **`SG_TUNED_MB_DW_SPLITK` removed** 2026-06-17: docs/prompt list all three but MB is gone.
7. **`resolve_extra_hipcc_flags`** doesn't skip `-1` maxrregcount sentinel (would emit `=-1`) — gfx942-only bug.

### Bugs surfaced by the 8-GPU TP run (PROGRESS.md §"8-GPU FLAGSHIP RUN")
These were discovered AFTER Phase 0 by the first live 8-GPU flagship training run:

- **Bug A (IMA — invalid memory access)**: 87 invalid 4-byte global writes in `fused_decoder_megakernel_tc<AdamW,ParConfig<8,8,1,1,Z3>>+0x1dc30` (wild ptr ~35 GiB below the NVSHMEM heap). Root cause: workspace cudaMalloc fails at flagship scale; null workspace → wild-pointer launch. **Fixed in `tp_datapath_fix_WIP.patch`** via OOM-safe cudaMalloc + null-workspace guard.
- **Bug B (no per-rank weight offset)**: `dectc_wbf_convert/dec_bind` reads FULL weight matrices identically on every rank → all ranks compute slice-0 → all-reduce sums 8 identical partials (degenerate). **Fixed in `tp_datapath_fix_WIP.patch`** by switching to full-width replicated computation (not weight-sharded) on the kTPComm path.
- **Bug C (head divisibility)**: flagship `kHeads=25` not `%TP=8` → `Hloc=3, Dloc=200` but invariant `Dloc==Hloc*kDhead=192` violated. **Resolved by Bug B fix** (full-width replicated attention removes the head-shard requirement; the kTPComm path now uses the same `dectc_attn_fwd_tile` as SingleGPU).

---

## 6. Archive Gaps

| # | Gap | Impact |
|---|---|---|
| **Gap #1** | Generated cells (`megakernel_codegen.py --write-all`) `#include fused_megakernel.cuh` + `model_stages.cuh` — both removed in pure-L3-TC refactor (commit `8b30ea8`); not in HEAD. **33 generated per-cell `.cu` DO NOT COMPILE**. | Affects per-cell standalone profiling. Real path = 3 `_tc` kernels. |
| **Gap #2** | `csrc/backends/cuda/sm_90/launch_<opt>.cu` (11 files) + `models/{decoder,vit,mamba}.cu` (3 files) referenced by `verify_all`/`profile`/`profile_maximal` but **in no git ref and have no generator**. | Affects per-component profiling + `verify_all` Phase-1/2 structural gates. Not the live megakernels. |

---

## 7. COMPILE_RECONCILE.md — True Status of Phase-2 Capabilities

### Already Present (verify-only):
- Optuna TPE sampler, XGBoost cost model + LCB pruning, BayesianEarlyStopper, Hyperband pruner
- Cross-run transfer warm-start (keyed by full `(opt,model,arch)`, NOT arch alone — contradicts prompt)
- Two-phase AOT→JIT, TimingWorker + CUDA-graph replay, MultiGPUTimingPool (autotune-variant only)
- Schema-inferred `-D` dims, ptxas flag tuning, all caching layers (ccache/sccache/Redis/CompileCache)
- Malformed-asm-leak guard, `async_depth`/`__launch_bounds__` tuned dims
- Split-K `SG_TUNED_{DEC,VIT}_DW_SPLITK` (MB removed); inline PTX (hard-disabled)

### Genuinely Absent (need authoring):
- **S1.4**: `cudaAccessPolicyWindow` / L2-persistence as tuned dim
- **S1.5**: smem carveout / max-dyn-smem tuned knobs
- **S3.4**: Cross-run negative cache + bloom dedup

### Partial (finish the missing aspect only):
- **S1.1**: SASS audit exists standalone in `profile_maximal.py:285`; NOT wired into compile.py loop
- **S1.2**: `ptxas -v` parsed (`compile.py:13492`), spill-gate is post-timing (`6910`); missing PRE-timing occupancy/spill prune
- **S3.1**: NVRTC real + default-on (`31484,17946`); compile-while-bench pipeline absent (explicit TODO `15119-15128`)
- **S3.2**: `CompileCache.prune(age,top_n)` exists + loop-wired; missing size-cap + LRU/LFU + ref-count pinning
- **S3.3**: sha256 closure drives invalidation; NVRTC cubin store content-addressed; primary artifact cache not CAS

### Blocked (needs `ncu`):
- **S1.3**: Measured achieved-occupancy/L2/DRAM features — `ncu` HW counters BLOCKED (`ERR_NVGPUCTRPERM`)

### Defer (kernel-dependent):
- **S2.1**: Stream-K, **S2.3**: cluster-multicast/warp-spec

---

## 8. What the Campaign Actually Delivered (PROGRESS.md milestone sections)

By the time PROGRESS.md was last updated, the following were completed (far beyond Phase 0):

1. **CuTe atoms steps 1-3**: `wgmma.cuh` rewritten behind `SG_TUNED_GEMM_ENGINE` (default 0=hand PTX, 1=CuTe). Validated bit-identical through the real decoder megakernel. Driver: `/workspace/phase1/cute_decoder_validate.py`.

2. **Flagship codegen (d,layers,vocab,seq)**: `megakernel_codegen.py` parameterized; `decoder_flagship_layout.cuh` emitted (d=1600, L=48, 582 tensors, kDecTotalElems=1,475,884,899). Production header byte-identical.

3. **Flagship TC megakernel compiles + fits**: ptxas: 255 regs, 24.8KB smem (layer-independent — streams layers, not caches all L), 23.5KB stack. 1 CTA/SM fits. Smem wall (feared ∝-kLayers) does not apply to TC path.

4. **dW-spec L-generalization applied**: 13 edits in `model_stage_decoder_tc.cuh` + 3 in `fused_decoder_megakernel.cuh` + 2 in `pp_stage_decoder_tc.cuh`. `kDecNumDwSpecs=4L+1`, formulas replace brace-lists. Byte-identical at L=2.

5. **Flagship 1.5B decoder runs end-to-end on silicon**: `flagship_smoke.py` at d=1600/L=48. loss=4.585047 finite (≈ln(99)), all-finite grads, 3×A/A/A bit-identical. L=48 dW-generalization validated on silicon.

6. **NVSHMEM TP all-reduce validated on 8 GPUs**: UID bootstrap (no MPI), team_split_strided, collective nvshmem_malloc, in-kernel device all-reduce. 8-GPU PASS (expected=36.0, got=36.0 bit-exact). NVSHMEM_DISABLE_NVLS=1 + NCCL_NVLS_ENABLE=0 auto-set (NVLink P2P path).

7. **All 3 flagship models launch**: Mamba smem redesign (19.56MB → 192.97KB via layer-streaming + HBM proxy). Flagship Mamba TC launched: 1.265B params, loss 4.577 (≈ln97), finite grads.

8. **Roofline deliverable**: nsys-measured 10 cells. Decoder 1.14-1.58 TF/s = 0.12-0.16% of 989 TF/s (occupancy-bound, not HBM-bound). ViT 0.27-0.28 TF/s = ~0.03%. Files: `/workspace/phase6/roofline_flagship.{png,csv}`.

9. **Dead-code cleanup**: Removed 8,089,083 lines / 528 files (95.7% of repo text). True source ~361K / 1047 text files. Build/imports intact; all 3 layouts byte-identical post-removal.

10. **11-opt decoder ranking benchmark**: 9/11 opts banked (muon + supergrok2 = FITS-but-slow). `/workspace/phase6/flagship_11opt_ranking.txt`. This is a fixed-batch OVERFIT benchmark (B=16, 100 steps), not a real-data run.

11. **TP data-path fix (WIP, ungated)**: `phase6/tp_datapath_fix_WIP.patch` (358 lines). Bugs A+B fixed:
    - OOM-safe cudaMalloc + null-workspace guard (Bug A root cause)
    - Full-width replicated TP (replaces Megatron weight-shard, fixes Bug B and Bug C by elimination)
    - The kTPComm path now computes every projection full-width, routes result/P through the device-NVSHMEM all-reduce (identity), exercising NVLink without requiring head-divisibility or per-rank weight packing.

---

## 9. LEDGER.json vs PROGRESS.md Discrepancy

`LEDGER.json` was written at `2026-06-24T21:40Z` with HEAD `c29ed4e` and shows all phases 1-6 as "pending". This is the session's STARTING state snapshot.

`PROGRESS.md` is the running ledger and reflects the full session (2026-06-24 through 2026-06-25). It records many completed milestones and shows HEAD evolving through commits: b92442b → 5733af5 → ed1bb55 → ... → (the closure commit mentioned in the assignment as e69df73).

**The LEDGER.json is definitively stale** — it was not updated after session start. PROGRESS.md is the authoritative state record.

---

## 10. Key Config-Derivation and Adaptivity Mechanisms (as documented)

From PROGRESS.md and the orchestration scripts' design notes:

1. **PARALLELISM auto_config**: 3D base (DP×TP×PP). SP (sequence-parallel) becomes 4th axis at long sequences. EP (expert-parallel) becomes 5th axis for MoE. ZeRO-3 is orthogonal sharding. The claim in PROGRESS.md §"INTEGRATION STATE" that "4D + ZeRO-3" is the current operating point for the 3 dense models is consistent with the 3D-5D design space. Current models are 4D.

2. **CTA tiling selector**: Size-adaptive selector chooses CTA tiling from workload/occupancy. `size_adaptive.md` in `impl_diffs/`. Occupancy>1 cluster body is scoped future work.

3. **Memory strategy gates**: `memory_strategy.md` in `impl_diffs/`. Offload / recompute / streaming chosen by fit — single GPU can train 10B+ by trading compute/bandwidth. Capacity does not dictate GPU count; the planner does. `resource_fit_planner.md` in `impl_diffs/`.

4. **Megakernel self-specialization**: if-constexpr folds in exactly the machinery the config needs (distributed→all-reduce, single→none; large→CTA-tiling, small→none; MoE→EP). Demonstrated by `tp_datapath_fix_WIP.patch` which shows `if constexpr (Par::kTPComm)` gating the NVSHMEM all-reduce path.

5. **The flagship 509 GB → fit progression**: PROGRESS.md §"FLAGSHIP 1.5B DECODER RUNS END-TO-END" explicitly documents that at full flagship dims (d=1600, nCTA=132), the optimizer scratch is 509 GB — beyond any single GPU. This is direct evidence that flagship REQUIRES 4D+ZeRO-3 sharding (TP shrinks per-rank Nmax; ZeRO-3 shards params + state + opt). This is the concrete mechanism behind the "self-adapting" claim.

---

## 11. Critical State Discrepancies vs CLAIMED State (per assignment prompt)

The assignment prompt claims:
- "DONE+validated: ...full TP; cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs"
- "#1 REMAINING: TP data-path fix — A+B fixed in phase6/tp_datapath_fix_WIP.patch (ungated)"

PROGRESS.md confirms:
- NVSHMEM 8-GPU all-reduce IS validated (the smoke test PASS at expected=36.0)
- The flagship 8-GPU megakernel TP training run surfaced 3 bugs (not the smoke test)
- The patch fixes A+B by switching to full-width replicated computation
- Bug C is "unconfirmed" status — the patch eliminates its root cause (head-shard) by design

**Key discrepancy**: PROGRESS.md §"INTEGRATION STATE" says the 8-GPU run needs:
(1) TP attention head-localization, (2) host NVSHMEM team bootstrap, (3) host weight-shard + torchrun harness.
But the patch (`tp_datapath_fix_WIP.patch`) removes head-localization (replaces with full-width replicated).
This is a deliberate design choice (correctness model comment at patch line 57-78) — not an omission.

**Benchmark status discrepancy**: The claimed "11-opt decoder ranking benchmark" in PROGRESS.md is
a **fixed-batch OVERFIT** run (B=16, 100 steps), explicitly not a real-data benchmark.
The assignment prompt treats this as validated; it is a placeholder.

**Roofline completeness**: PROGRESS.md claims "pending" for the "full 33-cell roofline" but documents
a 10-cell nsys-measured roofline as deliverable #1. The 33-cell full roofline was not completed.

---

## 12. Summary of What Was Genuinely Validated vs Stubbed vs In-Flight

| Item | State |
|---|---|
| CuTe atoms steps 1-3 | VALIDATED (bit-identical through real decoder) |
| Flagship codegen (d,layers,vocab,seq) | VALIDATED (byte-identical prod header + ptxas fits) |
| dW-spec L-generalization (L=48) | VALIDATED on silicon (flagship_smoke.py 4.585047 finite, A/A/A) |
| NVSHMEM 8-GPU all-reduce smoke | VALIDATED (36.0 bit-exact all 8 ranks) |
| 3 flagship models launch (single GPU) | VALIDATED (decoder, ViT, Mamba with smem redesign) |
| TP data-path fix (Bugs A+B) | IN-FLIGHT (patch written, not yet applied to HEAD, ungated) |
| 11-opt ranking benchmark | STUBBED (overfit placeholder B=16/100 steps, not real data) |
| Full 33-cell roofline | PARTIAL (10 cells measured; full 33-cell not done) |
| Real-data (FineWeb/ImageNet/GiftEval) | NOT STARTED (datasets Layer A specs in impl_diffs/datasets.md) |
| Resource fit planner | SPEC WRITTEN (`resource_fit_planner.md`) — integration status unclear |
| EP 5th axis / adaptive_parallelism | SPEC WRITTEN (`adaptive_parallelism.md`) — byte-identical seam, not exercised |
| Memory strategy (offload/recompute) | SPEC WRITTEN (`memory_strategy.md`) — integration status unclear |
| Dead-code cleanup | VALIDATED (8.09M lines removed, all 3 layouts byte-identical post) |
| ncu HW counters | PERMANENTLY BLOCKED in this container (ERR_NVGPUCTRPERM) |
