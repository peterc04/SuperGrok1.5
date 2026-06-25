# Phase6 Deliverables: Exhaustive Analysis

## Overview

The `/workspace/phase6/` directory contains the complete orchestration history and deliverable
artifacts for the final campaign phase of SuperGrok2 (repo SuperGrok1.5). This analysis covers
every file listed in P_phase6_deliverables.txt: the 11-optimizer ranking, roofline CSV,
TP data-path fix patch, all JS workflow scripts, bench scripts, and logs.

---

## 1. The 11-Optimizer Ranking (flagship_11opt_ranking.{json,txt})

### Source and Method
- `staged_opt_plumbing/flagship_staged_run.py` runs each optimizer on the flagship 1.5B decoder
  (d=1600, L=48, 1,475,884,899 params) with B=16 fixed batch (overfit), 100 steps, ncta=4 for
  generic opts, ncta=1 for SG2.
- `staged_opt_plumbing/aggregate_11opt.py` reads per-opt staged_{opt}_s0.json results and
  produces the ranked JSON and readable table.
- The per-opt JSON files live in the scratchpad at
  `/tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad/staged_{opt}_s0.json`.
- The 9 JSON files in `staged_opt_plumbing/` (staged_adamw_s0.json, staged_grokfast_s0.json,
  etc.) are committed copies of those results.

### Complete Ranking (flagship_11opt_ranking.txt:1-23)

```
rank  optimizer      id       lr  ncta  steps   first     last  descent  ms/step    status
   1  neuralgrok      6  2.0e-03     4    100  4.5850   2.6860   1.8991     5725        OK
   2  grokadamw       3  2.0e-03     4    100  4.5850   2.6860   1.8991     5751        OK
   3  adamw           0  3.0e-03     4    100  4.5850   2.6860   1.8991     4212        OK
   4  grokfast        2  2.0e-03     4    100  4.5850   2.6860   1.8991     4694        OK
   5  prodigy         5  1.0e+00     4    100  4.5850   2.6860   1.8990     5448        OK
   6  lion            1  3.0e-04     4    100  4.5850   2.6876   1.8975     3686        OK
   7  supergrok11     8  3.0e-03     4    100  4.5850   2.6968   1.8882     9578        OK
   8  supergrok15     9  3.0e-03     4    100  4.5850   2.6974   1.8877     9048        OK
   9  looksam         4  3.0e-03     4    100  4.5850   2.7329   1.8522     7318        OK
  --  muon           --       --     1     --      --       --       --       -- FITS/SLOW
  --  supergrok2     --       --     1     --      --       --       --       -- FITS/SLOW
```

### Gate Status (flagship_11opt_ranking.json:11-13)
```json
"gate_all_finite": true,
"gate_all_descending": true,
"gate_all_11_runnable": true
```

### Key Notes on FITS_BUT_SLOW cases
- **muon** (json:1556-1558): ncta forced to 4 (OOM at >=6). Memory: 3-plane state 16.5 GiB +
  params + grad + workspace ~66 GiB < 79 GiB. LAUNCHES and runs at 100% util. SLOW because
  grid-cooperative Newton-Schulz orthogonalizes 195 2D weights x 5 NS iters x 3 naive fp32
  matmuls (largest A=6400x6400xK=1600 = 65 GFLOP) on only 4 SMs = hours/step.
- **supergrok2** (json:1562-1565): ncta=1 resource-planner deep limit. Per-CTA CSA/HCA/PEER/GRU
  meta-net workspace = ~3.7 GiB/CTA at flagship ff width (full occupancy = 270 GiB OOM). FITS at
  ncta=1: 8-plane state (slow-plane aliased onto dead LookSAM workspace) 47 GiB + params 5.5 GiB
  + grad (aliased onto dead LookSAM sam_backup) + 20 GiB opt-agnostic workspace ~= 74 GiB < 79 GiB.
  LAUNCHES at 100% util. SLOW: per-element meta-net + SAM 2nd backward on SINGLE SM = tens-of-min/step.

### Memory Engineering (METHODS.md:34-48)
The opt-agnostic workspace carves ALL four staged regions unconditionally. Two zero-copy aliases
enable the heaviest opts to FIT:
- SG2 grad -> aliased onto dead LookSAM sam_backup region
- SG2 slow (grokfast EMA) state plane -> aliased onto dead LookSAM sam_grad region

Effective nCTA: 4 for generic opts (workspace ~31 GiB; total ~70 GiB fits), 1 for SG2.

### Architecture (METHODS.md:1-62)
The scratchpad JIT TU `mega_decoder_staged_tc.cu` extends the existing elementwise multiopt
driver to staged optimizers. Two pybind entries:
- `tc_train_step_opt(opt_id, ...)`: OptId 0-9 dispatch
- `sg2_train_step(...)`: Dedicated SG2 path with 26 meta-net weight packs

NO committed-source edit -- build via include only, mirroring the launcher dispatch exactly.

---

## 2. Roofline Deliverable (roofline_flagship.csv, logs)

### Coverage: 10/33 cells (PARTIAL)

The roofline_flagship.csv contains exactly 10 rows:
- **decoder x 5 opts** (adamw, grokadamw, grokfast, lion, neuralgrok): using nsys timing
- **ViT x 5 opts** (adamw, grokadamw, grokfast, lion, neuralgrok): using nsys timing
- **Mamba**: ZERO cells — ALL FAILED

CSV columns (roofline_flagship.csv:1):
```
model,opt,d,params,batch_B,ncta_cap_SMs,nsys_kernel_per_step_ns,step_ms,
achieved_tf_s,pct_of_989tfs_peak,arith_intensity_flop_per_byte,
gemm_flops_per_step,bytes_moved_per_step,nsys_kernel_instances
```

### Measured Performance

#### Decoder cells (B=128, ncta_cap=8, nsys timing):
```
adamw:     step=3107.77ms, 1.458 TF/s, 0.147% peak, intensity=82.82 FLOP/byte, 35 kernel instances
grokadamw: step=3883.36ms, 1.167 TF/s, 0.118% peak, intensity=82.82 FLOP/byte, 35 kernel instances
grokfast:  step=3366.34ms, 1.346 TF/s, 0.136% peak, intensity=82.82 FLOP/byte, 35 kernel instances
lion:      step=2859.99ms, 1.584 TF/s, 0.160% peak, intensity=82.82 FLOP/byte, 35 kernel instances
neuralgrok:step=3964.24ms, 1.143 TF/s, 0.116% peak, intensity=82.82 FLOP/byte, 35 kernel instances
```

#### ViT cells (B=64, ncta_cap=4, nsys timing):
```
adamw:     step=37208.15ms, 0.280 TF/s, 0.028% peak, intensity=170.85 FLOP/byte, 8 kernel instances
grokadamw: step=38660.74ms, 0.269 TF/s, 0.027% peak, intensity=170.85 FLOP/byte, 8 kernel instances
grokfast:  step=37789.81ms, 0.276 TF/s, 0.028% peak, intensity=170.85 FLOP/byte, 8 kernel instances
lion:      step=36699.44ms, 0.284 TF/s, 0.029% peak, intensity=170.85 FLOP/byte, 8 kernel instances
neuralgrok:step=38482.16ms, 0.271 TF/s, 0.027% peak, intensity=170.85 FLOP/byte, 8 kernel instances
```

#### Mamba cells: ALL FAILED (mamba_run.log:24-53)
"CUDA error: invalid argument" for all 5 opts (adamw, lion, grokfast, grokadamw, neuralgrok).
The mamba flagship TC megakernel requests ~19.56 MB dynamic smem (88x over the 227 KB H100 cap)
and cannot launch. This is the known pre-existing blocker.

### Saturation Sweep (dec_sat.log)
The sweep_decoder_sat.py found the best achievable single-GPU decoder throughput:
- BEST: 8.46 TF/s (0.85% peak) at B=512, cap=32
- B=128, cap=8: 1.46 TF/s (0.15% peak) — the cell in roofline_flagship.csv
- B=512, cap=16: 6.40 TF/s (0.65% peak)
- Anything above cap=64 OOMs at B=128 (workspace is nCTA-scaled)

### CUDA Event vs nsys Consistency
dec_probe.log (CUDA event): decoder/adamw = 3112.919ms, 1.46 TF/s, 0.147% peak
roofline_flagship.csv (nsys):   decoder/adamw = 3107.771ms, 1.458 TF/s, 0.147% peak
Difference is < 0.2%, indicating consistent measurement methodology.

### Pending (16-23 of 33 cells)
Missing from roofline_flagship.csv:
- 6 decoder optimizer cells (prodigy, muon, looksam, supergrok11, supergrok15, supergrok2)
- 6 ViT optimizer cells (same 6)
- ALL 11 Mamba cells (blocked by smem launch failure)
Total pending: 23/33 cells

---

## 3. TP Data-Path Fix (tp_datapath_fix_WIP.patch)

### Status: WIP PATCH EXISTS, NOT APPLIED TO COMMITTED SOURCE

The patch is a git diff format, targeting:
- `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu`
- `csrc/fused/sm_90/model_stage_decoder_tc.cuh`

### Bug A Fix: OOM-Safe Workspace (Launcher)
- OLD: `cudaMalloc(&s.workspace, ...)` + unconditional `s.ws_floats = need_floats`
- NEW: Check `cudaError_t merr = cudaMalloc(...)`. On failure: `s.workspace = nullptr; s.ws_floats = 0; (void)cudaGetLastError()` (clear sticky error).
- NEW guard before launch: `if (sc.workspace == nullptr || sc.ws_floats < need) return cudaErrorMemoryAllocation`
- Applied to both `mega_decoder_real_adamw_tc` (patch:36) and `mega_decoder_sg2_tc` (patch:44).
- Rationale (patch:11-17): the flagship production-layout TC workspace is hundreds of GB (the staged-opt LookSAM 2·total + SG2 Nmax·nCTA carves). On 80 GB GPU the cudaMalloc FAILS. The null workspace made the kernel's lnvec/acts writes index a wild base (the +0x1dc30 IMA seen on the TP8 run).

### Bug B Fix: Full-Width Replicated TP (model_stage_decoder_tc.cuh)
The patch introduces the **FULL-WIDTH REPLICATED TP APPROACH** (not genuine Megatron sharding):

**Design rationale** (patch:56-79):
> The host gives EVERY rank the FULL replicated param blob (NOT a pre-packed per-rank shard)
> and performs NO host-side gradient sync between steps; the kernel updates params in place.
> For all 8 ranks to stay BIT-CONSISTENT step over step every rank must apply the IDENTICAL
> FULL param update => compute the IDENTICAL FULL gradient. A Megatron weight-shard (each rank
> a distinct dW slice) would diverge the replicated params unless a full-weight grad all-reduce
> is added — which the tiny tile-sized symmetric heap cannot carry.
>
> So the kTPComm path computes EVERY projection FULL-WIDTH REPLICATED (the same math as
> SingleGPU, incl. attention at the full kHeads=25 — this also removes the kHeads%TP!=0 /
> Dloc==Hloc·kDhead invariant violation that was bug B), and the four reduce points still
> drive the REAL device-NVSHMEM all-reduce over NVLink: each rank publishes (full_result / P)
> and the ascending-pe fixed-order sum of P identical copies reconstructs full_result.

The `kTpInvP = Par::kTPComm ? (1.0f / P) : 1.0f` constant drives the divide-then-sum identity.

**Forward pass changes** (patch lines 87-167):
- QKV projection: `qkv_nout = 3 * dec::kD` (was `kTPComm ? (3*kD)/kTP : 3*kD`)
- Attention: now always calls `dectc_attn_fwd_tile` (full-width, same as SingleGPU). Previously had kTPComm branch calling `dectc_attn_fwd_tile_tp` with Hloc=kHeads/kTP.
- out_proj: full-width GEMM, then kTPComm publishes (a/P) -> ascending-pe sum -> a
- ff0: `ff0_nout = dec::kDff` (was `kTPComm ? kDff/kTP : kDff`)
- ff2: full-width GEMM, then kTPComm publishes (ff2/P) -> ascending-pe sum -> ff2

**Backward pass changes** (patch lines 222-358):
- Same pattern: `ff0_nout = dec::kDff`, `qkv_nout = 3*dec::kD` unconditionally
- ff2 dX: full-width dgact (was local dff/P with comm-free note)
- ff0 dX: full-width dx1, then kTPComm all-reduce identity (②')
- out_proj dX: `dctx_nin = dec::kD` (was `kTPComm ? kD/kTP : kD`)
- Attention bwd: always calls `dectc_attn_bwd_tile` (removes kTPComm head-shard path)
- in_proj dX: full-width, then kTPComm all-reduce identity (①')

### Bug C Status
The patch comment explicitly states (patch:73-78):
> "writes ONLY full-width buffers (matching dec_acts_bind) — eliminating the out-of-bounds
> shard-width writes (bug C). A future genuine weight-shard needs the host to pre-pack per-rank
> shards AND a whole-weight grad all-reduce; that is scoped, not done here."

Bug C is CLAIMED resolved as a consequence of full-width approach. However:
- This is theoretical — no compute-sanitizer run post-patch has been performed
- The patch is WIP and not applied to committed source
- The 8-GPU re-run to confirm "0 errors" is still pending

### Critical Design Note
The patch implements a FUNDAMENTALLY DIFFERENT TP APPROACH from the original design:
- Original bringup_parallel.js intended: genuine Megatron column/row sharding (per-rank weight shard)
- Patch implements: full-width replicated + allreduce identity (exercises NVLink path but doesn't reduce FLOP/memory per rank)
- "A future genuine weight-shard" is explicitly scoped out

---

## 4. JS Orchestration Workflow History

### Campaign Sequence (inferred from file content and cross-references)

#### Phase: Distributed Scope
**distributed_scope_workflow.js**: 3 READ-ONLY agents scoping 4D+ZeRO-3 distributed flagship.
- `dist_step`: Integration gap in distributed_step.py + ZeRO-3 wiring; produced impl_diffs/dist_step.md
- `tp_nvshmem`: In-kernel device-NVSHMEM all-reduce spec; produced impl_diffs/tp_nvshmem.md
- `run_harness`: 8-GPU torchrun harness + per-rank memory math; produced impl_diffs/run_harness.md

#### Phase: Next Phase (Read-Only Specs)
**next_phase_workflow.js**: 3 READ-ONLY investigation agents.
- `flagship_dw`: dW-spec L-generalization for decoder (hardcoded 9-spec -> 4*kLayers+1); produced impl_diffs/flagship_dw.md
- `profiler`: Phase-resolved decoder profiler (ncu-free, clock64); produced impl_diffs/profiler.md
- `tma`: TMA step-4 GEMM-load rewrite spec; produced impl_diffs/tma.md

#### Phase: Parallel Tracks (Read-Only Specs)
**parallel_tracks_workflow.js**: 5 READ-ONLY spec tracks.
- `tp_kernel`: TP Par-template + in-kernel NVSHMEM all-reduce; produced impl_diffs/tp_kernel.md
- `tma_wire`: TMA megakernel wiring; produced impl_diffs/tma_wire.md
- `vit_flagship`: ViT flagship layout + dW-gen; produced impl_diffs/vit_flagship.md
- `mamba_flagship`: Mamba flagship layout + dW-gen; produced impl_diffs/mamba_flagship.md
- `datasets`: Layer-A data plumbing; produced impl_diffs/datasets_v2.md

#### Phase: Apply Tracks
**apply_tracks_parallel.js**: 3 WORKTREE write-agents applying specs (ViT flagship, Mamba flagship, datasets).
**apply_flagships_workflow.js**: Similar 3-agent apply (ViT flagship, Mamba flagship, datasets).

#### Phase: Adaptive Design
**adaptive_design_workflow.js**: 2 READ-ONLY design agents.
- `adaptive_parallelism`: 3D-5D + EP as 5th axis; produced impl_diffs/adaptive_parallelism.md
- `size_adaptive`: Size/config-adaptive CTA-tiling selector; produced impl_diffs/size_adaptive.md

#### Phase: Apply Remaining (Parallel)
**apply_remaining_parallel.js**: 3 isolated write-agents on disjoint file sets.
- `tp_remainder`: TP kernel C.3 + D + E (grid-lockstep + reduce-points + launcher heap)
- `memory_strategy`: mem_strategy.py planner + mem_config.cuh gate POD
- `ep_size`: EP 5th axis + size-adaptive CTA-tiling seams

#### Phase: Resource Fit
**resource_fit_workflow.js**: 2 READ-ONLY agents.
- `resource_fit_planner`: plan_execution(model_cfg, hw_cfg) -> ExecutionPlan; produced impl_diffs/resource_fit_planner.md
- `memory_strategy`: Survey + scope of offload/recompute/streaming; produced impl_diffs/memory_strategy.md

#### Phase: Dead Code Analysis
**deadcode_analysis.js**: 2 READ-ONLY agents analyzing ~8.09M lines of removable code.
- `deadcode_artifacts`: _dectc_codegen/ (64 files, ~7.95M lines nvcc intermediates) + _scan/ + claude_session_archive/
- `deadcode_source`: tc_dump_outproj_operands, dead model classes, archive-gap dead cells

#### Phase: Kernel Redesign
**kernel_redesign_design.js**: 2 READ-ONLY design agents.
- `mamba_smem_redesign`: Layer-streaming to reduce MambaSampleSmem from 19.56 MB to <227 KB
- `vit_forkb`: ViT Fork-B grad-partial elimination (nCTA*total workspace blocker)

#### Phase: Bringup
**bringup_parallel.js**: 2 WORKTREE agents.
- `attention_shard`: TP attention head-localization (Hloc=kHeads/kTP per rank)
- `host_bringup`: NVSHMEM bootstrap + weight-shard + torchrun harness

#### Phase: Mamba Redesign + Rerun
**mamba_redesign_apply.js**: Apply Mamba smem redesign (Level A + B).
**mamba_rerun_workflow.js**: Re-run Mamba flagship apply agent.

#### Phase: ViT/Mamba TP
**vit_mamba_tp.js**: Mirror decoder TP track to ViT and Mamba.
- `vit_tp`: Mirror decoder TP (attn head-shard + 4 reduce points + launcher sym-heap)
- `mamba_tp`: Mirror decoder TP for SSM (no heads, shard in_proj/out_proj)

#### Phase: Lever3 Overlap
**lever3_overlap.js**: Saturate 8 idle H100s with flagship-scale decoder training while TP impl lands.
- Results fed into staged_opt_plumbing/staged_{opt}_s0.json and aggregate_11opt.py

#### Phase: Finish Line
**finish_line.js**: 3 parallel worktree agents closing gaps.
- `staged_opt_plumbing`: All 11 optimizers at flagship single-GPU -> ranking (DONE, gated)
- `nvshmem_pybind`: Launcher NVSHMEM-init pybind + 2-GPU smoke
- `mamba_test_fix`: Mamba test recalibration (2 pre-existing failures)

#### Phase: Flagship 8-GPU Run
**flagship_8gpu_run.js**: North-star one-model-across-8 run via TP8 + in-kernel NVSHMEM.
- Preceded the discovery of the 3 megakernel bugs (A, B, C)

#### Phase: TP Data-Path Fix
**tp_datapath_fix.js**: Fix the 3 bugs, re-run 8-GPU, confirm cross-rank agreement + loss descent.
- Produced tp_datapath_fix_WIP.patch (not yet applied)

#### Phase: Roofline Campaign
**roofline_campaign.js**: nsys-profiled roofline measurement of flagship cells.
- Produced roofline_flagship.csv (10/33 cells; Mamba blocked)

---

## 5. Bench Scripts

### bench_decoder_multiopt.py
- Loads the cached multiopt .so from `/workspace/flagship_build/mega_decoder_multiopt_adamw/`
- Times 5 elementwise opts (adamw=0, lion=1, grokfast=2, grokadamw=3, neuralgrok=6) with CUDA events
- Analytical GEMM FLOPs: fwd(T×3d×d + T×d×d + T×dff×d + T×d×dff + B×vocab×d) × 3 (fwd+bwd)
- Outputs CSV rows to stdout

### bench_vit_mamba_multiopt.py
- Builds one JIT TU (`mega_vit_multiopt_tc.cu` or `mega_mamba_multiopt_tc.cu`) against the
  flagship layout with BENCH_LAYOUT=1 to elide staged-opt scratch
- ViT: B=256, NCTA_CAP=8; Mamba: same but all cells fail at flagship scale

### make_roofline.py
- Aggregates CSV rows from dec_probe.log, vit_run.log, mamba_run.log (+ sat logs)
- Plots the H100 bf16 roofline (989 TF/s peak, 3.35 TB/s HBM3 bandwidth)
- Outputs roofline_flagship.csv + roofline_flagship.png

### sweep_decoder_sat.py
- Sweeps (B, ncta_cap) configs to find the saturation ceiling (best achievable TF/s)
- BEST: 8.46 TF/s (0.85% peak) at B=512, cap=32 (dec_sat.log:35)
- Shows memory pressure: cap>=132 OOMs at all tested batch sizes

### sweep_vit_sat.py
- Similar saturation sweep for ViT (only probe log vit_probe1.log shows limited data)
- vit_probe1.log only shows GPU memory after params (6.4 GB) and state (25.5 GB), no timing

### vit_probe1.py
- Memory probe for ViT flagship: configs (B, cap) = (16,1), (16,2), (16,4), (32,2), (48,2), (16,8)
- vit_probe1.log:1-5 shows build from cache + memory only (no timing results logged)

---

## 6. Staged Opt Plumbing (staged_opt_plumbing/)

### METHODS.md
Complete write-up of the staged-opt plumbing approach: what was built, memory engineering,
and the final result summary.

### flagship_staged_run.py
The runner script orchestrating all 11 optimizers:
- DEFAULT_LR per opt: adamw=3e-3, lion=3e-4, grokfast=2e-3, grokadamw=2e-3, neuralgrok=2e-3,
  looksam=3e-3, prodigy=1.0, muon=3e-3, supergrok11=3e-3, supergrok15=3e-3, supergrok2=1e-3
- SG2 build uses a separate build tag to avoid clobbering cached .so
- State sizes: SG2=(3+1+GH)*total+1; others=4*total+1+total+phi_pack
- SAM cadence for looksam/sg11/sg15: every 5 steps (step%5==1)
- Prodigy anchor p0 seeded = params at step 1

### aggregate_11opt.py
Aggregates per-opt JSON -> flagship_11opt_ranking.json + .txt.
- Ranking key: loss_last (lower = better) for OK opts
- EXCEPTIONS dict for muon/supergrok2: FITS_BUT_SLOW with detailed notes
- Gate logic: gate_all_finite checks all non-slow opts are finite; gate_all_descending
  checks loss_last < loss_first for all present opts

### staged_{opt}_s0.json
Per-opt real benchmark data. Example from staged_adamw_s0.json:
- d=1600, L=48, 1,475,884,899 params, ncta=4, 100 steps, B=16
- loss_first=4.585, loss_last=2.686, step_time_s=4.212s (4212ms/step)
- GPU: "1" (ran on GPU 1, not GPU 0 which was reserved for gates)

---

## 7. Scratch Rescue Files (scratch_rescue/)

These are intermediate .cu files used during development:
- `mega_decoder_multiopt_tc.cu`: The multiopt driver (5 elementwise opts, opt_id dispatch)
- `mega_decoder_staged_tc.cu`: Extended to all 11 opts including staged
- `mega_decoder_sg2_tc.cu`: SG2-dedicated path
- `mega_decoder_staged_tc.cu` (in staged_opt_plumbing/): The committed version of the staged TU
- `mega_vit_multiopt_tc.cu`: ViT multiopt driver
- `mega_mamba_multiopt_tc.cu`: Mamba multiopt driver (builds successfully; runs fail at flagship scale)
- Various probe/smoke files: mb_size.cu, mb_smoke.cu, mbtc_probe.cu, mem_cfg_probe.cu, sz.cu

---

## 8. Key Log Analysis

### dec_probe.log (decoder baseline, CUDA event timed)
5 decoder cells from the multiopt driver (B=128, ncta_cap=8):
- lion: fastest at 2870ms/step (1.58 TF/s)
- adamw: 3113ms (1.46 TF/s)
- grokfast: 3486ms (1.30 TF/s)
- grokadamw: 4020ms (1.13 TF/s)
- neuralgrok: 4003ms (1.13 TF/s)
Arithmetic intensity = 82.82 FLOP/byte for all (same model, same B).

### vit_probe1.log (ViT memory probe)
Only logs memory: params=6.4 GB, state (3x)=25.5 GB. No timing results.
(The actual ViT timing is in vit_run.log.)

### vit_run.log (ViT timing, CUDA event timed)
Build from cache (0.1s), d=1664, L=48, B=64(?):
- All 5 opts in range 17.5-19.9 seconds/step, 0.13-0.15 TF/s, 0.01-0.02% peak
- ViT is much slower than decoder (longer smem stalls, larger seq dimension)
- Intensity = 170.85 FLOP/byte (higher than decoder's 82.82 due to larger d=1664)

Wait: the roofline_flagship.csv shows ViT B=64 with nsys data. The vit_run.log shows
seq=17 (ViT patches), vocab=97. The actual batch used for the CSV was B=64 (from the bench script
default BENCH_B=256 env var, but the CSV shows 64 -- this may have been manually adjusted).

Actually looking at vit_run.log more carefully: the log shows `[vit-multiopt] d=1664 layers=48
seq=17 vocab=97 params=1,596,200,417` -- but the CSV shows batch_B=64. The bench_vit_mamba_multiopt.py
default is BENCH_B=256, so there may have been manual adjustment to B=64.

### mamba_run.log (Mamba flagship run FAILED)
Build succeeded (82.3s), d=2048, L=24, params=1,265,411,169.
ALL 5 opts FAILED with "CUDA error: invalid argument". This is the smem launch failure
(19.56 MB dyn_smem requested, 227 KB cap). The mamba smem redesign spec and apply workflow
exist (mamba_redesign_apply.js, impl_diffs/mamba_smem_redesign.md) but it's unclear if the
redesign was successfully applied to committed source.

---

## 9. Config-Derivation Mechanism (as manifested in phase6)

### resource_fit_workflow.js
The resource fit planner spec defines:
```
plan_execution(model_cfg, hw_cfg) -> ExecutionPlan:
  - model_cfg: d, layers, seq, vocab, num_experts, is_sequence
  - hw_cfg: num_gpus, hbm_bytes_per_gpu, host_ram_bytes, nvlink/pcie
  - Outputs: (DP,TP,PP,SP,EP degrees), (need_zero_offload, need_activation_recompute,
             need_layer_streaming, need_param_offload), (cta_tiling, ring_depth)
```

Key design rule (resource_fit_workflow.js:20): "strategy decisions must NOT key on GPU count.
A SINGLE GPU can host a 10B+ model even for TRAINING."

### adaptive_design_workflow.js  
Defines the 3D-5D adaptive parallelism:
- Base: DP x TP x PP (always)
- +SP (4th): if is_sequence model (decoder / ViT-patches / Mamba are sequence, SP-eligible)
- +EP (5th): if num_experts > 1 (MoE)
- ZeRO-3 is orthogonal sharding, NOT an axis

The auto_config.py infer_parallel_config function (NEW file, part of applied spec):
- Returns ParConfig + per-axis degrees
- Honors device count (8 H100s) + run_harness.md mesh math

### Size-Adaptive Kernel (size_adaptive.md)
SizeConfig<CtaTile, CtasPerTile, ClusterDim, TileN> + decoder_knobs_for_size selector:
- CTA-tiling ON for LARGE configs (more SMs to fill, fixes the 20% grid-barrier idle at d=2048)
- OFF for SMALL configs (persistent 1-CTA/SM wins; overhead dominates)
- NOTE: Actual CTA-tiled occupancy>1 kernel body is a scoped follow-on, not implemented

---

## 10. Discrepancies vs Claimed State

### Discrepancy 1: TP Data-Path Fix Status
CLAIMED: "A+B fixed in phase6/tp_datapath_fix_WIP.patch (ungated)"
ACTUAL: The patch EXISTS in phase6/ but is NOT applied to committed source. The term "fixed"
overstates: the patch is a WIP ready-to-apply diff. The 8-GPU re-run with compute-sanitizer
to confirm 0 IMA errors has NOT been performed.

### Discrepancy 2: Bug C Confirmation
CLAIMED: "bug C unconfirmed"
ACTUAL: The patch CLAIMS bug C is resolved as a consequence of full-width replicated approach
(patch comment: "eliminating the out-of-bounds shard-width writes (bug C)"). The claim is
theoretical based on the design argument. No empirical confirmation via compute-sanitizer.

### Discrepancy 3: Roofline Deliverable Completeness
CLAIMED: "roofline deliverable" (in DONE list)
ACTUAL: Only 10/33 cells measured. 23 cells are pending (6 decoder opts + 6 ViT opts +
11 Mamba opts). Mamba is fully blocked by smem issue. The "full 33-cell roofline" is
correctly listed under "remaining" items.

### Discrepancy 4: TP Design Change Not Surfaced
CLAIMED: "cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs"
ACTUAL: The 8-GPU flagship run discovered bugs that required a fundamental approach change.
The WIP patch converts from genuine Megatron weight-sharded TP to "full-width replicated +
allreduce identity" (mathematical no-op for correctness but doesn't reduce FLOP/memory per rank).
This is a significant design divergence not surfaced in the summary.

### Discrepancy 5: 11-Opt Benchmark Description
CLAIMED: "11-opt decoder ranking (overfit placeholder)"
ACTUAL: The benchmark is concrete and validated (not a placeholder). The json/txt files contain
real GPU execution data with full 100-step loss trajectories. All 9 OK opts show genuine descent
from 4.585 to ~2.686. The term "overfit placeholder" refers to the FIXED-BATCH (non-real-data)
nature -- the loss descends because it's overfitting a fixed batch, not generalizing.
"placeholder" is misleading; it's genuine measured data, just on synthetic/fixed data.

### Discrepancy 6: Mamba Flagship Status
CLAIMED: "3 flagship models LAUNCH"
ACTUAL: Mamba at FLAGSHIP DIMS (d=2048/L24) does NOT launch. mamba_run.log confirms
cudaErrorInvalidValue for all opts. The mamba_redesign_apply.js workflow exists to fix this,
but the committed mamba_flagship_layout.cuh + megakernel still has the smem blocker.
The "LAUNCH" claim likely refers to the SMALL-SIZE (d=128/L=2) test layout, not flagship.

---

## 11. Summary of Phase6 State

| Deliverable | Status |
|-------------|--------|
| 11-opt ranking (decoder) | DONE, validated, real GPU data |
| Roofline CSV (10 cells) | PARTIAL (10/33; Mamba blocked) |
| TP data-path fix patch | WIP (not applied) |
| 8-GPU re-run post-fix | PENDING |
| Mamba smem redesign | SPEC exists; apply status unclear |
| ViT Fork-B | SPEC exists; apply status unclear |
| resource_fit_planner | SPEC in impl_diffs/, not verified as live code |
| EP 5th axis + size-adaptive | SPEC applied (apply_remaining_parallel.js) |
| Real-data benchmark | PENDING (blocked by TP fix) |
| Full 33-cell roofline | PENDING (blocked by Mamba + 23 missing cells) |
