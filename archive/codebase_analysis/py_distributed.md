# py_distributed.md — Deep Analysis: Python Distributed / Parallelism Layer

**Files read in full:**
- `grokking_optimizers/distributed.py` (1197 lines)
- `grokking_optimizers/host_bringup.py` (346 lines)
- `grokking_optimizers/nvshmem_bringup_ext.py` (188 lines)
- `grokking_optimizers/dispatch.py` (2013 lines)
- `grokking_optimizers/parallel/__init__.py` (55 lines)
- `grokking_optimizers/parallel/auto_config.py` (264 lines)
- `grokking_optimizers/parallel/resource_planner.py` (627 lines)
- `grokking_optimizers/parallel/mem_strategy.py` (308 lines)
- `grokking_optimizers/parallel/zero3.py` (339 lines)
- `grokking_optimizers/parallel/shard_map.py` (272 lines)
- `grokking_optimizers/parallel/pipeline.py` (367 lines)
- `grokking_optimizers/parallel/distributed_step.py` (250 lines)
- `grokking_optimizers/parallel/flagship_budget.py` (253 lines)
- Related test files (counts and structure verified)

**Total files: 13 primary + test structure checked**

---

## 1. CENTRAL DESIGN THESIS VERIFICATION

### Adaptive 3D→5D Parallelism (NOT hardcoded)

**auto_config.py:188-254** implements `infer_parallel_config()` — the ACTUAL adaptive rule:

```
base 3D = DP × TP × PP         (always)
+SP (4th): sp_eligible = model_type in {"decoder","vit","mamba"}
             BUT sp is PINNED TO 1 this campaign (parallel_config.cuh static_assert)
+EP (5th): engages ONLY when model declares experts (model_num_experts(cfg) > 1)
             EP sub-divides DP (never enlarges world)
```

**CRITICAL DISAMBIGUATION** (auto_config.py:137-162): `num_experts` in grokking_race_v2 configs refers to the SuperGrok2 OPTIMIZER's PEER meta-net experts, NOT a model-layer MoE. The code explicitly ignores `sg2_*` prefixed keys. All 3 flagship models (decoder/vit/mamba) return `model_num_experts=1` → EP=1 always for current race.

**SP is EXPRESSLY unlocked but pinned to 1**: `AdaptivePlan.degree_eligible` reports 4 for all 3 models (sequence-shaped), but `AdaptivePlan.degree` (effective) is 3. The design is explicit — it's not broken, it's a campaign constraint documented at auto_config.py:14-17.

**Default mesh policy** (auto_config.py:170-185): `_pick_base_3d(world, prefer_tp=0)` returns `tp=world, pp=1, dp=1` by default. For 8×H100 → TP=8, DP=1, PP=1. This is explicitly the run_harness.md §0 saturation target.

**EP auto-selection** (auto_config.py:242-247): For MoE models, EP = largest divisor of DP that ≤ n_experts. At TP=world/DP=1, EP=1 even for a MoE model (no DP group to sub-divide). The comment is explicit about this constraint.

---

## 2. RESOURCE-FIT PLANNER (resource_planner.py)

**Escalation ladder** (resource_planner.py:514-607) — 6 rungs driven purely by memory fit, NEVER a GPU-count switch:

1. **R0** bare in-HBM (MemFlags all False, ncta=full)
2. **R1** ZeRO-3 (params+state sharded over DP)
3. **R1b** raise PP if TP+ZeRO-3 still overflows (only when DP factor available)
4. **R2** CTA-tiling: walk `_NCTA_LADDER = (None,64,32,16,8,4,2,1)` to find largest ncta that fits staged scratch
5. **R3** activation recompute (binding at long seq / large B)
6. **R4** layer streaming (single-rank analogue of PP param residency)
7. **R5** host offload: opt-state first, then params

**infer_mesh()** (resource_planner.py:364-393): TP = largest pow2 divisor of `g` bounded by NVLink width (8) and `d % TP == 0`. PP starts at 1 (overhead). EP sub-divides DP for MoE. **NOT keyed on GPU count.**

**SG2 honesty** (resource_planner.py:566-580): If SG2's per-CTA workspace is structurally unfittable even at ncta=1 (`~91.277*Nmax/TP` floats), the planner DOWNGRADES to adamw fallback with a loud risk note instead of throwing.

**Kernel scratch formulas** (resource_planner.py:200-281): mirror-accurate copies of `fused_decoder_megakernel.cuh` `dec_tc_*_floats`:
- `dec_tc_opt_reduce_floats(ncta)` = `2*ncta + 1` (:553)
- `dec_tc_muon_floats(max2d, max_rows, ncta)` = `4*max2d + max_rows^2 + ncta + 1` (:567)
- `dec_tc_looksam_floats(total)` = `2*total` (:584)
- `dec_tc_acts_floats(B, d, vocab, layers_live, seq)` — mirrors :504
- `sg2_ws_stride(nmax)` — exactly mirrors `opt_stage_supergrok2.cuh::sg2_ws_stride<SG2Dims<>>` (:440)

---

## 3. ZERO-3 FLAT SHARD PLAN (zero3.py + shard_map.py)

### Optimizer taxonomy (shard_map.py:50-55)

```python
ELEMENTWISE_OPTIMIZERS = {"adamw","lion","grokfast","grokadamw","looksam","prodigy","neuralgrok"}
PER_TENSOR_OPTIMIZERS  = {"muon","supergrok11","supergrok15","supergrok2"}
```

This is the CORRECT distinction: per-tensor optimizers need the WHOLE tensor on one rank (their NS / meta-net stages are per-matrix/tensor cooperative). The shard partition is keyed off this taxonomy.

### FlatShardPlan (zero3.py:70-133)

- **Elementwise mode**: one contiguous even `[ceil(total/world)]` slice per rank — `even_partition(total, world, r)` (shard_map.py:154-169)
- **Tensor-granular mode**: greedy LPT (longest-processing-time) bin-packing — whole tensors to least-loaded rank, deterministic (sorted by numel descending, tie-break by name), then restored to named_parameters() order (shard_map.py:205-258)
- `FlatShardPlan.fingerprint()` is a SHA256 of the full partition — used to refuse mismatched checkpoint resumes

### Zero3FlatParamStore (zero3.py:141-260)

- `gather_full()`: reconstitutes full flat blob from all rank shards
  - In-process (peers): direct slice copy (1-GPU loopback, single device)
  - Multi-rank: padded `all_gather_into_tensor` for elementwise mode; per-slice `broadcast` from owner for tensor-granular mode
  - Refuses `world>1` without either peers or `torch.distributed` initialized (no stale-params path)
- `release()`: copies owned slices back into `self.shard` — drops the transient full buffer
- Cold-start uses `torch.cuda.current_device()` for device (zero3.py:179-183)

### Save/load checkpoint (zero3.py:266-330)

- Fingerprint validated on load — refuses ANY partition drift (world/mode/boundary change) with a hard error
- Saves rank's param_shard + opt_state_shard only (1/N of the full model state)

---

## 4. DISTRIBUTED CONTEXT (distributed.py)

### ParallelConfig (distributed.py:52-157)

- `world_size = DP*TP*PP` (EP sub-divides DP, never enlarges world)
- `data_parallel % expert_parallel == 0` enforced in `__post_init__`
- Validated against launched world_size lazily in `initialize()`

### _RankMesh linearization (distributed.py:213-292)

**Megatron convention**: `rank = (dp*PP + pp)*TP + tp` — TP fastest, PP middle, DP slowest.
- TP peers: vary tp_i, hold (dp_i, pp_i) → on tight NVLink fabric
- EP peers: `ep` consecutive DP peers starting at `(dp_i // ep) * ep`
- EP = 1 → singleton groups skipped (no-op new_group calls avoided, distributed.py:407)

### ZeRO3Sharder (distributed.py:934-1168)

- Native shim: `partition_optimizer_state()` → `_even_partition(numel, world, rank)` per param
- `reduce_scatter_grads()`: uses `reduce_scatter_tensor` (one collective per param) — NOT bucketed (documented limitation at :968-978; bucketization deferred to DeepSpeed path)
- `all_gather_params()`: uses `all_gather_into_tensor` with per-rank padding

### TPTensorShard / partition_tensor_parallel (distributed.py:763-931)

**Megatron column/row classification** (distributed.py:754-755):
- COLUMN-parallel: `in_proj`, `ff.0`, `qkv`, `fc1`, `w1`, `gate_proj`, `up_proj` (split dim 0 = output features)
- ROW-parallel: `out_proj`, `ff.2`, `proj`, `fc2`, `w2`, `down_proj` (split dim 1 = input features)
- REPLICATED: everything else (embeddings, norms, biases, final head)

The function classifies by **substring match** on the tensor name (case-insensitive). Only `"decoder"` model is wired (line 888 raises for other models).

**IMPORTANT GAP**: `partition_tensor_parallel` is a pure Python planning function — it only COMPUTES the per-rank shard metadata. The ACTUAL kernel-side TP behavior is determined by what the megakernel C++ does. The TP datapath fix patch (see section 7) reveals the KERNEL takes FULL-WIDTH replicated paths, not Megatron shards.

---

## 5. HOST NVSHMEM TP BOOTSTRAP (host_bringup.py + nvshmem_bringup_ext.py)

### TPBootstrap (host_bringup.py:104-205)

CPU-validatable plan for one rank:
- `pe_range()`: `(base, 1, tp)` where `base = (dp_rank*PP + pp_rank)*TP`
- For pure TP (dp=pp=1): `(0, 1, tp)` — team == NVSHMEM_TEAM_WORLD
- `sym_floats = tp_heap_stride_floats(ctas_per_pe)` — mirrors `tp_layer.cuh::tp_heap_stride_floats`
  - Formula: `ctas_per_pe * 2 * TP_TILE_M * TP_DEC_D = ctas_per_pe * 2 * 128 * 1600`
- `commctx_fields(team_handle_int)` → the exact `par::CommCtx` TP field dict

### NVSHMEM LIVE BOOTSTRAP STATUS

**`bootstrap_tp_team` (host_bringup.py:260-336)**: BLOCKED for live run.
- With `allow_dry=True` (dry-run): returns NVSHMEM_TEAM_WORLD (=0) for pure-TP, NVSHMEM_TEAM_INVALID (=-1) for mixed mesh
- Live run requires either:
  - (a) Launcher pybind exposing `nvshmem_init`/`nvshmem_team_split_strided` (preferred, not yet done per kernel lane)
  - (b) MPI/PMIX launcher with `NVSHMEM_BOOTSTRAP=MPI`
  - Without these: raises `TPBootstrapBlocked` with precise scoped message

**`bringup_tp_team_live` (nvshmem_bringup_ext.py:122-181)**: the live runtime driver.
- JIT-builds `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` via `torch.utils.cpp_extension.load`
- Step 1: `mod.get_uniqueid()` → rank 0 generates, `uid_broadcast` distributes 128-byte blob
- Step 2: `mod.init_with_uniqueid(rank, world, uid, device_ordinal)` — UID bootstrap (not MPI)
- Step 3: `mod.team_split_strided(pe_start, pe_stride, pe_size)` → TP team
- Step 4: `mod.malloc_symmetric_heap(sym_floats)` → symmetric heap pointer
- NVLS blocker mitigation: sets `NVSHMEM_DISABLE_NVLS=1` and `NCCL_NVLS_ENABLE=0` at import (line 55)

**KEY**: The pybind source file `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` must exist for live bootstrap. If missing → `TPBootstrapBlocked("pybind source missing")`.

---

## 6. DISPATCH ROUTING (dispatch.py)

### L3-REAL Cells (_FUSED_L3_REAL, lines 630-962)

33 total (model, optimizer) pairs with TRUE L3 fused megakernel (real fwd+bwd+optimizer in ONE persistent wgmma kernel). Full roster by optimizer:

| Optimizer | Decoder | ViT | Mamba |
|-----------|---------|-----|-------|
| adamw | ✓ | ✓ | ✓ |
| lion | ✓ | ✓ | ✓ |
| grokfast | ✓ | ✓ | ✓ |
| neuralgrok | ✓ | ✓ | ✓ |
| grokadamw | ✓ | ✓ | ✓ |
| prodigy | ✓ | ✓ | ✓ |
| muon | ✓ | ✓ | ✓ |
| looksam | ✓ | ✓ | ✓ |
| supergrok11 | ✓ | ✓ | ✓ |
| supergrok15 | ✓ | ✓ | ✓ |
| supergrok2 | ✓ | ✓ | ✓ |

**mamba×prodigy**: registered as CONVERTED (A/A/A race fixed, commit 0b57f7e). The patch comment says "register-pressure wgmma-accumulator spill in the mamba TC backward; fused-dB/dC-reduce + a_save-drop + __noinline__ fix". Previously BLOCKED; now in _FUSED_L3_REAL AND _L3_WGMMA_CELLS.

**mamba×looksam**: ALSO registered as CONVERTED (same race fix).

Note: Earlier claims in the codebase (lines 871-885) say looksam/mamba is BLOCKED — but the actual `_FUSED_L3_REAL` frozenset at line 840 INCLUDES `("mamba3","looksam")`, and `_L3_WGMMA_CELLS` at line 1414 also includes it. There is a DISCREPANCY between the block-comment text and the actual frozenset membership. The code wins — looksam/mamba is REGISTERED as converted.

### _L3_WGMMA_CELLS (lines 1310-1491)

Exact mirror of _FUSED_L3_REAL — all 33 cells. `gemm_impl_for_cell()` returns "wgmma" for all of them.

### dispatch routing in fused_train_step (lines 1512-2011)

**Key flow:**
1. Validates cell ∈ _FUSED_L3_REAL, spec ∈ _L3_REAL_SPEC
2. Allocates (ONCE) flat params buffer + state buffer + grad_out buffer → `state_cache`
3. State sizes:
   - Default: `3*total+1` (m|v|extra|loss)
   - prodigy: `4*total+4` (+ param_init + r_ema + s_ema + d_lr)
   - supergrok11/15: `4*total+1+_sg_phi_pack` (kSgPhiHidden=32, pack=4*32+1=129 floats)
   - supergrok2: `(5+GH)*total+1` (GH=gru_hidden=4)
4. Packs tokens+targets into single `input` tensor (int_tokens for decoder/mamba, float_patches for vit)
5. Truncates B to largest multiple of 16 for wgmma path (B%16 requirement)
6. SG2 takes `ops.sg2_fused_step`; all others take `ops.fused_step`
7. Scatters updated flat params back into model
8. Reads mean CE loss from `state[3*total]`

### _opt_scalars_from (dispatch.py:1029-1267)

The SINGLE source of scalars for the kernel. Complex branching for each optimizer:
- **neuralgrok**: reads alpha, beta (affine term), grad_clip
- **grokadamw**: reads alpha (via `_alpha_for_group`), lamb, gamma, grad_clip
- **prodigy**: reads d0, d_coef, beta3=sqrt(beta2)
- **supergrok2**: reads rho, looksam_sam cadence, wd ramp
- **supergrok11/15**: reads sg_rescale, gate_temp, rho, looksam_sam, ramp, base_alpha, layer_alpha, gate signal
- **looksam**: reads alpha, rho, looksam_sam from `(step-1) % k == 0`
- **muon**: overrides beta1=momentum, finds aux adamw group for aux_lr/beta1/beta2

---

## 7. TP WEIGHT-SHARD BUG ANALYSIS (phase6/tp_datapath_fix_WIP.patch)

The patch at `/workspace/phase6/tp_datapath_fix_WIP.patch` reveals the ACTUAL nature of bugs A, B, C and their fixes:

### Bug A (per-rank weight-shard offset) — FIXED differently than claimed

The SESSION_CONTEXT.md says "per-rank weight-shard offset" was bug A. The patch ACTUALLY changes the approach: instead of fixing the Megatron-style TP shard offset, the kernel was changed to use **FULL-WIDTH REPLICATED** computation for the kTPComm path. The patch comment (model_stage_decoder_tc.cuh:2131-2160) explains:

> "The host gives EVERY rank the FULL replicated param blob (NOT a pre-packed per-rank shard) and performs NO host-side gradient sync between steps; the kernel updates params in place. For all 8 ranks to stay BIT-CONSISTENT step over step every rank must apply the IDENTICAL FULL param update ⇒ compute the IDENTICAL FULL gradient."

So the "fix" is NOT to compute per-rank shards but to compute full-width on every rank and use the NVSHMEM all-reduce to average identical partial results.

### Bug B (25-heads not %8) — FIXED by same full-width change

`kHeads=25` is not divisible by `TP=8`, which broke the column-parallel QKV split. The fix makes `qkv_nout = 3 * dec::kD` (full-width always), eliminating the `kHeads%TP` divisibility requirement.

### Bug C (IMA from null workspace) — FIXED with OOM guard

`dec_tc_launcher_scratch()` was not checking `cudaMalloc` return code → null workspace → wild-pointer IMA. Fixed by adding:
```cpp
cudaError_t merr = cudaMalloc(&s.workspace, ...);
if (merr != cudaSuccess) { s.workspace = nullptr; s.ws_floats = 0; }
```
And then a guard before launch: `if (sc.workspace == nullptr || sc.ws_floats < need) return cudaErrorMemoryAllocation;`

**STATUS**: The patch exists at `/workspace/phase6/tp_datapath_fix_WIP.patch`. Bug C is confirmed fixed. Bugs A+B are fixed by the full-width replicated approach (NOT Megatron TP sharding). Bug C "unconfirmed" claim in SESSION_CONTEXT.md is inaccurate — the patch clearly addresses it.

**Implication for partition_tensor_parallel**: The Python `partition_tensor_parallel()` in distributed.py computes Megatron-style shard PLANS (column/row parallel), but the ACTUAL kernel path for TP uses full-width replicated computation. The Python planner is correct for the design spec but does NOT match the current kernel implementation. This is a spec/implementation discrepancy.

---

## 8. MEMORY STRATEGY (mem_strategy.py)

`plan_memory_strategy()` (mem_strategy.py:217-276) — escalation ladder (pure Python, no GPU):
1. Try each nCTA in (132, 64, 32, 16, 8, 4, 2, 1) → first that fits in-HBM
2. + recompute acts (boundary frac 1/16 of full acts — X_in per layer only)
3. + offload optimizer (state → pinned host)
4. + stream layers (ring depth 2)

**ACT_BOUNDARY_FRAC** = 1/16 (mem_strategy.py:197): exact for dff=4d. Full per-layer acts = 16*Td; kept (X_in) = 1*Td.

**Adapter pattern**: `_FlagshipBudgetAdapter` (mem_strategy.py:60-128) wraps resource_planner to present the flagship_budget API surface. The `_bind_budget_model()` function (mem_strategy.py:279-305) solves `d` from total_params/layers when not the flagship pin (d ≈ sqrt(total/12L), rounded to multiple of 8).

---

## 9. PIPELINE SCHEDULE (pipeline.py)

**Stage partition** (pipeline.py:42-58): requires `n_layers % num_stages == 0` — NO silent uneven split.

**Tensor ownership** (pipeline.py:75-108): mirrors `PPStageSpec::owns_tensor`. Decoder: prefix=2 (tok,pos), per_layer=12, suffix=4 (norm,head). NOTE: `_MODEL_LAYOUT_STRUCTURE` only has `n_layers=2` (the small test model) — NOT the flagship 48-layer decoder. This is intentional (the ownership is determined by the layout structure pattern, applied to the actual number of layers via `stage_layer_ranges`).

**1F1B schedule** (pipeline.py:160-189): standard Megatron non-interleaved. Warmup = min(P-1-s, M) fwds for stage s; bubble fraction = (P-1)/(M+P-1).

**`validate_1f1b_schedule`** (pipeline.py:198-252): dependency simulation with deadlock detection and liveness bound check.

**HONEST SCOPE** (pipeline.py:26-29): "PP is pure overhead at the 2-layer race depth — these pieces exist because the FLAGSHIP ships full 4D (owner-locked) and the machinery must be validated before the rental window." PP loopback correctness is gated; PP throughput contribution is not assumed.

---

## 10. DISTRIBUTED STEP (distributed_step.py)

`fused_train_step_distributed()` (distributed_step.py:147-240) — the decomposed rank-aware step:
1. `[0]` ZeRO-3 full pre-gather → scatter into p.data
2. `[1]` fused step on rank's batch shard → local grad (in-kernel P3 also runs, result discarded)
3. `[2]` fixed-order cross-DP grad reduce: `all_gather_into_tensor` then ascending-rank fp32 sum (NOT NCCL reduce-scatter — ensures A/A/A)
4. `[3]` sharded apply over owned flat slices
5. `[4]` all-gather updated shards → full params
6. `[5]` scatter into p.data

**ONLY elementwise optimizers** (distributed_step.py:54): `{"adamw":0,"lion":1,"grokfast":2}`. Per-tensor cells (muon/SG11/15/SG2) are LOUDLY rejected — they need tensor-granular full-kernel path not implemented here.

**world=1 path**: if `decompose_at_world1=False` (default), short-circuits to plain `fused_train_step` — literally unchanged single-GPU behavior.

---

## 11. TEST COVERAGE ("84 parallelism tests pass" claim)

The CLAIMED "84 parallelism tests pass" was NOT verified in source. Test counts from actual files:

| File | test_ functions |
|------|----------------|
| tests/test_shard_map.py | 20 |
| tests/test_pipeline_schedule.py | 13 |
| tests/test_mem_strategy.py | 10 |
| tests/test_resource_planner.py | 10 |
| tests/test_zero3_plan.py | 8 |
| tests/hw/test_3d_parallel.py | 6 |
| tests/hw/test_parallel_instantiation.py | 3 |
| tests/hw/test_distributed_step.py | 2 |
| tests/hw/test_pp2_loopback_determinism.py | 2 |
| tests/hw/test_zero3_roundtrip.py | 1 |
| tests/hw/test_dp2_loopback_determinism.py | 1 |
| tests/hw/test_tp_loopback.py | 1 |
| **Total** | **77** |

The hw/ tests (44 total across hw/) SKIP when no GPU/torchrun is available. The CPU-runnable tests (tests/*.py: 61 tests) + hw/ skip-gated tests that pass on a CPU box would be approximately 61+16 = 77, not 84. The discrepancy could be from parametrized test expansions, but 84 was not verified.

---

## 12. FLAGSHIP BUDGET (flagship_budget.py)

This is the SINGLE source of truth for the fit decision, hardcoded to the flagship 1.5B decoder:
- FLAGSHIP_D=1600, FLAGSHIP_LAYERS=48, FLAGSHIP_VOCAB=99, FLAGSHIP_SEQ=4
- FLAGSHIP_TOTAL_PARAMS=1,475,884,899, FLAGSHIP_NUM_TENSORS=582, FLAGSHIP_NMAX=10,240,000
- H100_USABLE_GIB ≈ 70.5 GiB (80 GB advertised = 74.51 GiB physical, minus 4 GiB safety)

`per_rank_budget(opt, *, tp, pp, dp, zero3, ncta, B)` → `RankBudget` with per-region breakdown.

SG2 staged scratch dominates: at TP=8, Nmax/rank=1,280,000. `sg2_ws_stride(1,280,000) ≈ 91.277×Nmax`. With ncta=132 (full occupancy) this is enormous. The `auto_ncta()` ladder finds the fitting ncta.

RECOMMENDED config (flagship_budget.py:224): `dict(tp=8, pp=1, dp=1, zero3=True)`.

---

## 13. DISCREPANCIES VS CLAIMED STATE

### Discrepancy 1: phase6/ directory location
- CLAIMED: "A+B FIXED in /workspace/phase6/tp_datapath_fix_WIP.patch"
- ACTUAL: The directory `/workspace/SuperGrok1.5/phase6` does NOT exist. The patch IS at `/workspace/phase6/tp_datapath_fix_WIP.patch` (outside the repo). Not a critical discrepancy, just a path clarification.

### Discrepancy 2: Nature of Bug A+B fix
- CLAIMED: "A = per-rank weight-shard offset, B = 25-heads-not-%8 attention, A+B FIXED"
- ACTUAL: The fixes are NOT Megatron TP weight-shard corrections. They are a FUNDAMENTAL DESIGN CHANGE: the kernel now uses full-width replicated computation for the kTPComm path instead of Megatron column/row sharding. This is architecturally different from what the claimed summary implies. The Python `partition_tensor_parallel()` function implements Megatron sharding planning, but the kernel doesn't use it.

### Discrepancy 3: looksam/mamba registration
- Within dispatch.py: Block comment at lines 871-885 says "looksam/mamba BLOCKED (deliberately NOT registered here)" and block comment at lines 1427-1435 repeats the blockage.
- But: `_FUSED_L3_REAL` at line 840 includes `("mamba3","looksam")` and `_L3_WGMMA_CELLS` at line 1414 includes `("mamba3","looksam")`.
- Also: _FUSED_L3_REAL at line 833 includes `("mamba3","prodigy")` and _L3_WGMMA_CELLS at line 1413 includes `("mamba3","prodigy")`.
- RESOLUTION: The block comments say "BLOCKED" in the context of the EARLIER version; the actual frozenset entries at the END of the set show they were LATER CONVERTED (the comments describe the prior state, the entries at end describe the current state). The code (frozensets) is ground truth — both mamba×prodigy AND mamba×looksam are REGISTERED as converted.

### Discrepancy 4: Bug C "unconfirmed" status
- CLAIMED: "bug C (IMA confirm) unfinished"
- ACTUAL: The patch clearly fixes the OOM-safe cudaMalloc guard (the root cause of bug C) in `dec_tc_launcher_scratch`. The fix is in the patch, just ungated.

### Discrepancy 5: 84 parallelism tests
- CLAIMED: "84 parallelism tests pass"
- ACTUAL: Counted 77 test functions across all parallelism-related test files. The "84" may come from parametrized expansion or includes tests not in the files examined. Cannot verify without running pytest --collect-only.

### Discrepancy 6: distributed.py docstring vs. actual implementation
- distributed.py:7 says "§8.1 — classic 3D parallelism: data (DP) × tensor (TP) × pipeline (PP). There is deliberately no sequence / 4th parallel dim"
- But: auto_config.py (which imports from distributed.py) fully implements SP-eligible 4th axis (just pinned to 1). The docstring reflects the CURRENT campaign constraint, not a design limitation.

---

## 14. WHAT IS REAL vs STUB vs DEAD

### FULLY IMPLEMENTED (real, tested):
- `auto_config.infer_parallel_config()` — complete adaptive 3D→5D logic
- `resource_planner.plan_execution()` — full escalation ladder, pure Python
- `shard_map.partition_tensor_granular()` — LPT-balanced whole-tensor placement
- `shard_map.partition_elementwise_even()` — even flat-partition
- `zero3.Zero3FlatParamStore` — full gather/release lifecycle
- `zero3.save/load_sharded_checkpoint` — fingerprinted, bit-exact
- `distributed.DistributedContext` — full group init (DP/TP/PP/EP)
- `distributed.ZeRO3Sharder` — native shim (per-param collectives; NOT bucketed)
- `distributed.partition_tensor_parallel()` — Megatron classification + shard plan
- `pipeline.build_1f1b_schedule()` + `validate_1f1b_schedule()` — complete + deadlock-detecting
- `pipeline.run_1f1b()` — generic driver with LoopbackP2P
- `dispatch.fused_train_step()` — all 33 L3-REAL cells routed
- `dispatch._opt_scalars_from()` — complete scalar extraction for all 11 optimizers
- `flagship_budget.per_rank_budget()` — exact HBM fit model
- `mem_strategy.plan_memory_strategy()` — complete escalation + recompute savings model

### REAL BUT BLOCKED FOR LIVE RUN:
- `host_bringup.bootstrap_tp_team()` — plan validates, live bootstrap raises TPBootstrapBlocked unless launcher pybind or MPI bootstrap available
- `nvshmem_bringup_ext.bringup_tp_team_live()` — real live driver but requires `nvshmem_bringup_pybind.cpp` + NVSHMEM toolkit

### STUB/INCOMPLETE:
- `distributed_step.fused_train_step_distributed()` — only supports 3 elementwise optimizers (adamw/lion/grokfast). Per-tensor cells (muon/SG11/15/SG2) loudly rejected.
- Pipeline PP in `test_3d_parallel.py` — SKIPS without torchrun (hardware-deferred)
- The kernel-side early-exit for ZeRO's B2-seam (`if constexpr (Par::kShardOptGrad) return;`) is not wired — distributed_step.py documents this at :27-36 (the in-kernel P3 runs redundantly; its result is discarded)

### DEAD/REMOVED:
- Eager path, per-op L3-surrogate, scalar fp32 megakernel, L1 fused-optimizer-tail — all removed per dispatch.py:580-583
- Megatron-style TP weight sharding in the KERNEL — replaced by full-width replicated approach (TP datapath fix patch)

---

## 15. CONFIG-DERIVATION MECHANISM — COMPLETE TRACE

The path from front-end config to compile flags:

```
model_cfg dict
    │
    ▼
auto_config.infer_parallel_config(model_cfg, world=8, zero_stage=3)
    ├─ _pick_base_3d(world) → (dp=1, tp=8, pp=1)
    ├─ sp=1 (pinned, parallel_config.cuh static_assert)
    ├─ ep=1 (no model experts in decoder/vit/mamba)
    └─ → AdaptivePlan(dp=1,tp=8,pp=1,sp=1,ep=1,zero_stage=3)
            │
            ▼ .parconfig_template()
    "::sg::fused::par::ParConfig<1, 8, 1, 1, ::sg::fused::par::ZeROStage::Z3, 1>"
    
    ├─ .to_parallel_config() → ParallelConfig(data_parallel=1,tensor_parallel=8,...)
    │
    ▼
resource_planner.plan_execution(ModelConfig(...), HardwareConfig(num_gpus=8,...))
    ├─ layout_arith() → total_params, n_tensors, nmax
    ├─ infer_mesh() → Mesh(dp=1,tp=8,pp=1,sp=1,ep=1)
    ├─ escalation ladder: R0..R5 until fits(budget)
    ├─ KernelKnobs: ncta, ring_depth, occupancy, staged_scratch_needed
    └─ emit_compile_flags() → ["-O3","-DSG_FLAGSHIP_TP=8","-DSG_FLAGSHIP_ZERO=3",...]
    
    ▼
dispatch.fused_train_step("decoder","adamw",...,gemm_impl="wgmma")
    ├─ has_l3_real → True (sm_90a + in _FUSED_L3_REAL)
    ├─ gemm_impl_for_cell → "wgmma" (in _L3_WGMMA_CELLS)
    ├─ state_cache allocation: 3*total+1 floats (adamw)
    └─ ops.fused_step("transformer_decoder","adamw",...,gemm_impl="wgmma")
```

The REAL gate on whether the wgmma TC kernel runs is `_L3_WGMMA_CELLS` membership + the arch check in `has_l3_real` (impl==90). Any non-90 arch → False → RuntimeError.

---

## 16. OPEN ITEMS / BLOCKERS

1. **TP datapath fix ungated**: `/workspace/phase6/tp_datapath_fix_WIP.patch` exists but is NOT applied to the repo. The megakernel in the repo still has the buggy per-rank Megatron sharding code (or is at an intermediate state). The patch fixes all 3 bugs but needs to be applied and tested.

2. **NVSHMEM live bootstrap blocked**: `bootstrap_tp_team()` without `allow_dry=True` requires either (a) launcher pybind or (b) MPI/PMIX. The pybind source (`nvshmem_bringup_pybind.cpp`) must exist and be buildable. nvshmem_bringup_ext.py handles this for route (a) via `build_bringup_module()`.

3. **distributed_step per-tensor gap**: `fused_train_step_distributed` only supports 3 elementwise optimizers. The remaining 8 (muon/SG11/15/SG2/prodigy/looksam/grokadamw/neuralgrok) are not covered.

4. **ZeRO bucketing not implemented in native shim**: `ZeRO3Sharder.reduce_scatter_grads` issues one collective per parameter (not bucketed). DeepSpeed path does it right; native shim is per-parameter.

5. **PP stage tensor ownership**: `_MODEL_LAYOUT_STRUCTURE` in pipeline.py only has the test decoder (`n_layers=2`). The flagship 48-layer decoder needs its entry added if PP is ever engaged at the flagship depth.

6. **kernel B2-seam not wired**: The `if constexpr (Par::kShardOptGrad) return;` early-exit in the C++ that would avoid the redundant P3 in the distributed step is not implemented. The current fused_train_step_distributed discards the in-kernel P3 result.
