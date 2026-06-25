# Memory and History Digest
*Slice: .session_memory/ (12 standing-rules files) + older 06-12 campaign reports*

Generated: 2026-06-25

---

## Part 1: Session Memory — Standing User Directives

### 1.1 MEMORY.md (index file)
File: `/workspace/SuperGrok1.5/.session_memory/MEMORY.md`

Lists 8 active memory modules. Provides a quick-reference index. All from session originSessionId `6354dc07-b50f-40a0-8748-5189102539d3`.

---

### 1.2 supergrok-working-prefs.md
**L3-TC kernels ONLY** (no scalar/naive path). Production path = persistent fused wgmma megakernel with `SG_TUNED_GEMM_IMPL=1`.

Upgraded default knobs (already `#ifndef`-set):
- Decoder: `SG_TUNED_DEC_FWD_PIPE=1`/`FWD_STAGES=4` (+1.49x), `DW_STAGE=1` (+2.05x)
- ViT: `VIT_P1_SUBTILE_S=8` (4.02x)

**Prebuilt binaries exist** — do NOT recompile unless necessary:
- `_ops*.so`, `tune11_out/*/*.so`, `task11_bench_build/{A_sk4,B_sk4,C_sk2}/*.so`, `nvcc_baseline_build/`, `_dectc_codegen/*/*.{cubin,ptx,fatbin}`

**Compile cache**: `.fast_build_env.sh` with sccache+ccache; `.build_cache` (1.3G) committed.

**Read exhaustively** (no grep-skimming). Authoritative reference = `CODEBASE_EXPLAINED.md`; live state = `SESSION_STATE.md`/`PLANNING_INPUT.md`/`.perf/phase1_status_audit.md`.

**Parallelize aggressively** with parallel agents. Minimize GPU hours.

---

### 1.3 supergrok-autonomy.md
**Never pause to ask priority/what-next** at milestones. User wants maximal throughput. Proceed autonomously, drive critical path as lead, fan out independent tracks. Only ask when genuinely blocked on decisions only user can make (external/destructive actions). Report progress + ETAs but keep moving.

---

### 1.4 supergrok-execution-style.md
The REAL product: a **portable, self-adapting, max-performance training stack** (PyTorch-shaped: high-level Python over CUDA/C++ backend). Two core properties:
1. **Portability** — every component drops into anyone's project, config-driven
2. **Self-designing megakernels** — autotuner co-generates optimal kernel for ANY workload (10M→1.5B→bigger)

Validation: 11-optimizer ranking benchmark across 3 ~1.5B models (decoder/ViT/Mamba-3) on real datasets (FineWeb-Edu/ImageNet-1k/GiftEvalPretrain) with 4D parallelism (DP×TP×PP×SP) + ZeRO-3. SP is active at scale (was pinned 1 only for seq=4 toy).

**Critical path** = flagship kernel regen -> datasets -> real 1.5B training.

**4D+ZeRO-3** is HOW 8 GPUs get saturated: ONE 1.5B model distributed across all 8.

User frustrated by: mis-scaling (toy vs 1.5B), fumbled GPU orchestration, over-asking.

---

### 1.5 supergrok-adaptive-parallelism.md
**Central design thesis** (user directive 2026-06-25):

**Part 1: Adaptive 3D–5D parallelism, inferred from front-end params:**
- Base 3D = DP × TP × PP (always)
- +SP (4th axis) IF sequence model
- +EP (5th axis, expert parallelism) IF MoE model
- Current 3 flagships (decoder/ViT/Mamba) are sequence non-MoE → 4D
- EP is a NEW 5th axis to add + front-end → ParConfig inference function
- `ParConfig<DP,TP,PP,SP,Z>` exists in `parallel_config.cuh` but EP axis not yet added

**Part 2: Kernels self-specialize by size/config:**
- CTA-tiling helps at large sizes (fill SMs), hurts at small sizes (overhead)
- Size-thresholded knob selection in `megakernel_codegen.py`
- CTA-tiling ties to bottleneck lever ② (20% grid-barrier idle at d=2048)

**Unifying principle (if-constexpr):** megakernel TEMPLATED on deployment config → folds in exactly the machinery config needs:
- distributed → emits all-reduce/TP/parallelism
- single-GPU → none (byte-identical)
- large size → CTA-tiling ON; small → OFF
- SAME mechanism for EP, SP, CTA-tiling = more `if constexpr`-gated branches

**Robust resource-fit planner (NOT GPU-count):**
- Given (model size/shape + hardware: #GPUs, HBM/GPU, host RAM, interconnect) → decide:
  - parallelism degree (3D-5D)
  - memory strategy (in-HBM | ZeRO-offload | activation-recompute | layer-streaming | host-offload)
  - kernel knobs (CTA-tiling, ring depth, occupancy)
- 10M-on-1-GPU → trivial; 10B-on-1-GPU → offload+recompute+streaming+CTA-tiling; 1.5B-on-8-GPU → 4D+ZeRO-3
- Single GPU CAN train 10B+ by trading compute/bandwidth for capacity

---

### 1.6 supergrok-frontend-api.md
**Fixed (library surface):** 3 model architectures (decoder/ViT/Mamba-3) + 11 optimizers. Parameterized by SIZE (any size from 10M→1.5B→10B+).

**NOT fixed — datasets are PLUGGABLE:** FineWeb-Edu/ImageNet-1k/GiftEvalPretrain are provided implementations of a PLUGGABLE dataset interface. User must be able to connect their own dataset (streaming train iterator + fixed eval probe). Interface must be a generic PROTOCOL, not a 3-way hardcode.

**Flow (PyTorch-shaped):** instantiate model + pick optimizer + pass dataset → backend SELF-SPECIALIZES: codegen emits layout for that size, resource planner decides parallelism+memory strategy+CTA-tiling, COMPILES (cached). Config change → cached recompile (megakernel is size-pinned at compile time).

---

### 1.7 flagship-distributed-config.md
**The winning 4D mesh:** TP=8 · DP=1 · PP=1 · ZeRO-3

**Why TP=8 is needed:**
- SG2 staged-opt scratch: `dec_tc_sg2_floats = nCTA · ~91·Nmax` (linear in Nmax)
- Nmax = largest tensor = ff weight `dff·d = 10.24M` at d=1600
- TP=8: Nmax=1.28M/rank → SG2 scratch shrinks 509 GB → ~58 GB/rank
- Result: 10/11 optimizers run at 1-CTA/SM (nCTA=132, 66-68 GiB/rank); SG2 auto-caps to nCTA=64 (40.9 GiB)
- Single-GPU dense does NOT fit staged opts (509 GB scratch)

**Critical verified state (as of flagship-distributed-config.md):**
- DP + host-ZeRO-3: WIRED (`fused_train_step_distributed`)
- TP/PP: STUBBED — production launcher `launch_fused_decoder_megakernel_tc<OptId>` NOT yet templated on `ParConfig`/`CommCtx` (zero refs in fused_decoder_megakernel.cuh)
- Wiring TP needs: template kernel+launcher on `Par`, symmetric-heap TP-comm-slot allocator, in-kernel `tp_allreduce_sum_fixed_order` via device NVSHMEM
- Apply-ready specs: `/workspace/impl_diffs/{dist_step,tp_nvshmem,run_harness}.md`

**DISCREPANCY vs CLAIMED STATE:** The claimed "DONE+validated: full TP; cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs" conflicts with flagship-distributed-config.md which says TP/PP are STUBBED and the production launcher is not templated. This memory was written 2026-06-25 and appears authoritative.

---

### 1.8 ncu-blocked-runpod.md
`ncu` returns `ERR_NVGPUCTRPERM` on RunPod 8xH100 container — `CAP_SYS_ADMIN`/`CAP_PERFMON`/`CAP_SYS_PTRACE` not in bounding set. **Unfixable from inside container.**

Fix requires pod relaunch with `--cap-add=SYS_ADMIN` (RunPod custom template / support request).

**Counter-free fallbacks:** nsys timeline, cuobjdump -sass/nvdisasm/ptxas -v, CUDA-event wall-clock → throughput + analytical-FLOP roofline, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

---

### 1.9 nvshmem-installed.md
NVSHMEM **3.7.0** installed 2026-06-25 via `pip install nvidia-nvshmem-cu12`.
- Header: `/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem/include/nvshmem.h`
- Host lib: `.../nvshmem/lib/libnvshmem_host.so.3`
- **Device bitcode for sm_90**: `.../nvshmem/lib/libnvshmem_device_sm_90.bc` + `libnvshmem_device.a`
- Compile verified: `nvcc -std=c++17 -arch=sm_90a -rdc=true -I.../nvshmem/include -c <tu>.cu` → rc=0
- NVSHMEM_HOME = `/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem`
- `-DSG_HAS_NVSHMEM=1` gate is now buildable (was absent in 06-12 campaign)
- Caveats: device calls need `-rdc=true` + link `libnvshmem_device.a`; operands must be in `nvshmem_malloc` SYMMETRIC heap; multi-GPU needs NVSHMEM bootstrap

**Note:** 06-12 campaign's Phase-2 REPORT.md says NVSHMEM was NOT installed then — this is a key difference between old and current state.

---

### 1.10 supergrok-queued-deliverables.md
Two explicitly queued post-build deliverables:

1. **33-cell flagship roofline graph** (11 optimizers × 3 models): ncu-FREE (analytical arithmetic intensity on x-axis vs achieved TF/s from CUDA-event wallclock), plotted against H100 bf16 roofline. Deliver as matplotlib PNG.

2. **Comprehensive line-by-line dead-code cleanup** + total LOC + per-language LOC/percentage report. Do when tree is STABLE (not during parallel integration edits). Provably-dead only. After cleanup: cloc-style count.

Both are POST-BUILD deliverables.

---

### 1.11 supergrok-cutlass-cute-directive.md
**Use CUTLASS/CuTe** for megakernel GEMMs instead of hand-rolled wgmma (`csrc/backends/cuda/sm_90/wgmma.cuh`).

**Why:** hand-rolled wgmma runs well below cuBLAS/CUTLASS-class (decoder GEMM phases ~462ms vs ~40ms ideal; 6.48% roofline).

**Critical nuance:** Cannot drop in CUTLASS's host-launched `CollectiveMma` — it owns its own grid/launch, incompatible with 1-CTA/SM persistent kernel. Must use **CuTe DEVICE-SIDE ATOMS** composed inside megakernel:
- wgmma MMA atom (`SM90_64xNx16_F32BF16BF16_SS`)
- SM90_TMA_LOAD copy atoms (real TMA, replacing current cp.async ring)
- CuTe swizzle layouts
- `cutlass::pipeline` for multi-stage + warp-specialization
All device-callable between grid barriers, preserving one-launch fusion.

This is ARCH-level work (substantial GEMM substrate rewrite). It's the LEAD task for arch phase after general work (compile, bug-fixes, dead-code, datasets, 4D+ZeRO-3, NVSHMEM, verification, profiling).

**Status (from CLAIMED state):** "CuTe-atom GEMM engine (bit-identical, SG_TUNED_GEMM_ENGINE)" is CLAIMED done and validated. This memory predates that milestone and was a directive to do it — if the claim is true, this was fulfilled.

---

### 1.12 vit-tc-forkb-already-ported.md
**Key finding (2026-06-25 campaign correction):** task #31 ("port decoder Fork-B grad-partial elimination to ViT") is STALE. Already done in production ViT TC path.

**Two ViT megakernels:**
- Scalar `fused_vit_megakernel` (#if SG_VIT_SCALAR_MEGAKERNEL, ~L184-349): HAS nCTA*total grad partial; allocated by gate-only `scalar_train_step` (mega_vit_real_adamw_tc.cu:335). Compiled OUT at flagship. Never shipped.
- TC `fused_vit_megakernel_tc` (#if SG_TUNED_GEMM_IMPL==WGMMA, L503+): production path. Workspace = `vit_tc_workspace_floats` (L479), NO nCTA*total. Already has: HBM bf16 acts, P2 output-stationary dW, split-K dW, cls/pos owner-scan.

**One thing ViT didn't adopt:** Decoder uses `SG_TUNED_DEC_DW_SPLITK=1` (after adding contiguous-transpose staging SG_TUNED_DEC_DW_STAGE=1). ViT still uses `SG_TUNED_VIT_DW_SPLITK=4` AND has NO contiguous-transpose staging → carries `vit_dw_part_floats(4)` = 25.5 GB at flagship d=1664.

**Apply-ready spec:** `/workspace/impl_diffs/vit_forkb.md`
- EDIT 2A (SAFE): flip `SG_TUNED_VIT_DW_SPLITK` 4→1: -25.5 GB. CAVEAT: G=1 dW is SLOWER without staging.
- EDIT 2B (LARGE, OUT OF SCOPE): port decoder contiguous-transpose dW staging.

**Real flagship 80GB blocker:** `vit_tc_acts_floats` (Fork-B HBM bf16 acts) ~379 GB at grid-saturating batch. The 25.5 GB dW partial does NOT fix the 80GB constraint.

---

## Part 2: Older Campaign Reports (06-12, branch claude/h100-audit-maximal)

### 2.1 .audit_notes.md — H100 Audit Working Notes (06-12 campaign)

**Branch:** `claude/h100-audit-maximal` (STALE — current branch is `claude/custom-optimizer-analysis-HFYhg`)

**Environment:** H100 80GB HBM3, sm_90, CUDA 12.4, torch 2.4.1+cu124, 1.5TB RAM, 224 cores.

**Baseline gate state:**
- self-test: 229 pass / 2 fail
- verify_all: 67 pass / 2 fail / 67 skip-silicon
- 33 sm_90 cells compile
- 2 failures (compile.py sccache CXX path bug + utilization_track_cell_crashes inverted logic)

**Optimizer diagnoses (key findings):**

SG11/SG15:
- Root cause: `SG11_H/SG15_H = 64` hardcoded but canonical SharpnessMetaNet `hidden_dim = 32`
- Fix: change to 32 in respective .cuh files
- Secondary (SG11): FROZEN STEP COUNTER — `_flat_steps` passed by value; `steps[i]+=1` in binding doesn't persist → step frozen at 0 → bc1/bc2 pinned at t=1 → Adam denom ~50x inflated
- Fix applied: increment `_flat_steps` in Python (supergrok11.py ~250, supergrok15.py ~314) + remove binding steps[i]+=1
- SG11 also has mu applied TWICE + gate polarity inverted vs canonical sg11.h:101

Prodigy:
- Root cause: degree bug — d-update is degree -1 in d (carries 1/d_prev). d0=1e-6 → 1/d_prev=1e6 catapults d: 1e-6→0.185 in ONE step → max clamp → diverges
- Fix: restore canonical EMA d-adaptation (persistent d_numerator EMA w/ beta3=sqrt(beta2), ||s||_1 from running s buffer)
- EMA fix applied → Prodigy GROKS @ step 1000 (tst 0.998 peak)

SG2:
- Root causes: zero-GRU placeholder + descending-vs-ascending sort
- Fix: matrix-GRU reconstruction in `csa_hca_step_one` + flip sort to ascending + gru_state writeback
- RISK: CSA/HCA CUDA kernel fidelity UNVERIFIED

**Race results (06-12):**
- After fixes: 8/11 grok
- Remaining 3 failing: SG11 (mu-path fix regressed it → flat), SG15 (memorize-then-collapse, meta-net gate dynamics), SG2 (flat despite GRU+sort fix — CSA/HCA kernel-fidelity gap)
- Verdict: 3 SuperGrok-family optimizers numerically correct (parity 11/0) but bilevel meta-net DYNAMICS destabilize post-memorization

**Profile maximality (agent a6e853f8):**
- WGMMA/TMA claim holds: 256 HGMMA + 622 TMA (UTMALDG), all CUTLASS Sm90 collective
- Binary sm_90a-ONLY, 0 spills everywhere live (except dead CUTLASS TF32-RS GEMM)
- All 5 failures = ONE kernel spilling 8B (CUTLASS TF32 A-transposed RS GEMM, runtime-dead)

**Race coverage (agent aa6e79db):**
- Race = fused optimizer kernels × eager PyTorch models
- GOTCHA: race `adamw` uses STOCK torch.optim.AdamW(fused=True), NOT the pkg adamw kernel
- Model kernels ARE callable: ops.models.{decoder,vit,mamba}_{forward,backward} (15 entries)
- Mamba race uses python fallback scan (mamba_scan_ext absent), never hits mamba3_sm90

---

### 2.2 .morning_report.md — Overnight run 2026-06-12

**Branch:** `claude/h100-audit-maximal`. HEAD = `642e360`.

**Performance achieved overnight:**
| metric | bedtime | morning | factor |
|---|---|---|---|
| decoder d=1024 B=16k | 2084 ms / 4.75 TF/s | **967.3 ms / 10.2 TF/s** | **2.15x** |
| decoder d=128 | 107.6 ms | **32.8 ms** | **3.28x** |
| vit d=128 | 166.6 ms | **123.6 ms** | 1.35x |
| mamba d=128 | 108.0 ms | **102.7 ms** | 1.05x |
| 33-cell roofline | 0.38% mean | **1.15% mean** | ~3x |

**4 kept optimizations:**
1. H1: M-atom-interleaved wgmma pipeline (2 atoms share staged B-tile) — d=1024 -22%
2. H2: O(d·T) embed-grad CSR + single-owner bias reduce — grad_asm 60x speedup
3. H3: dW-tile interleave — dW -30.7%
4. Ports: H1+H3 → vit+mamba (vit eliminated ALL register spills)

**Honest failures:**
- cp.async mbarrier ring: ATTEMPTED, cleanly REVERTED. 3 structural blockers: (a) fp32 weights → cp.async can't convert, (b) dW operands transposed-strided → TMA-with-transpose needed, (c) engine at 255 regs WITH 2.6KB spills — zero headroom for ring state
- Tuner GPU validation: INCOMPLETE — bindings.cpp pins PYBIND11_MODULE(_ops) → JIT variant module can never import. Partial patch at `/workspace/.orphan_12_jit_fix.diff`
- SG2 workspace: 199GB @ d=1024. Workspace redesign needed.
- test_mamba_tc 3-fail: proven PRE-EXISTING

**Phase-2 pre-tests:** sharded-opt BIT-IDENTICAL 9/9 cells; DP=2 loopback cross-rank A/A/A bit-exact; CUDA-graph step capture bit-exact x5. Only NVSHMEM-TP + real scaling measurements need 8x.

---

### 2.3 .parallelism_design.md — Phase-2 Contract

**Status:** DESIGN-ONLY (task #22). CPU-only authored, no GPU testing.

**Key design decisions:**

**ParConfig struct** (`parallel_config.cuh`):
```cpp
template <int DP, int TP, int PP, int SP, ZeROStage Z>
struct ParConfig {
    static constexpr bool kIsSingleGPU = (DP==1 && TP==1 && PP==1 && SP==1);
    static constexpr bool kEmitComm    = !kIsSingleGPU;
    static constexpr bool kShardParams = (Z == ZeROStage::Z3);
    static constexpr bool kTPComm      = (TP > 1);
    static constexpr bool kPPStage     = (PP > 1);
    // SP is expressible but pinned to 1 (static_assert)
};
```

**Megakernel cut at B2 seam:** B2 is the clean cut between [fwd+bwd → local grad] and [optimizer over grad → updated params]. Multi-GPU decomposition: `[fwd+bwd megakernel] → reduce-scatter(grad) → [sharded optimizer kernel] → all-gather(params)`.

**ZeRO decomposition:**
- Under ZeRO≥2: megakernel returns at B2 (P3 gated by `if constexpr (!Par::kShardOptGrad)`)
- Separate sharded-optimizer kernel (`sharded_optimizer_kernel.cuh`) consumes reduce-scattered grad

**Optimizer sharding constraints:**
- Elementwise (trivial): AdamW, Lion, Grokfast, NeuralGrok, elementwise core of GrokAdamW/LookSAM/Prodigy
- Prodigy/GrokAdamW/LookSAM: also need tiny all-reduce of global scalars (2-3 floats)
- Per-MATRIX (Muon): whole matrix must live on one rank → tensor-granular shard
- Per-TENSOR (SG11/SG15/SG2): meta-net operates on whole tensor → tensor-granular shard

**What already exists (as of 06-12):**
- `grokking_optimizers/distributed.py`: `ParallelConfig`, `DistributedContext`, `ZeRO3Sharder`
- `megakernel_engine.py`: `MegakernelOptimizer`/`FusedBackwardHook` framework adapter
- `tests/hw/test_3d_parallel.py`: torchrun harness + CPU-testable mesh/shard/efficiency tests

**Gaps the Phase-2 design fills:** megakernel fused launch (single call producing grad[total]) vs framework-managed separate fwd/bwd/step; per-TENSOR optimizer constraint not modeled; TP/PP are config fields with no kernel realization; no graph capture; no NVSHMEM.

---

### 2.4 .regpressure/REPORT.md — Static Register Pressure Campaign

**All TC kernels: 255 registers, 1 CTA/SM** (255 regs pins to 65K regs/CTA = full SM file → 1 CTA/SM occupancy by design; smem 9.7-43.9 KB of 227 KB max, never the limiter).

**Critical findings:**
- `--maxrregcount` is SILENTLY IGNORED for these kernels (launch_bounds takes precedence)
- Spill bytes are the pressure metric, not register count

**Baseline spill inventory:**
- Decoder: 7 single-pass cells = 0 spills; LookSAM/SG11/SG15 = ~15252 sp_st; SG2 = ~15424 sp_st
- ViT: 7 light cells = 0-8 sp_st; LookSAM/SG11/SG15 = ~15020; SG2 = ~18288
- Mamba: ALL cells have spills (5692-5852 sp_st for 7 simple cells; 9452-10180 for SAM-family) — mamba uses `__noinline__` device functions

**Attribution:** wgmma accumulator array owns the margin. `WgmmaAccum<128>` = 64 fp32 regs/fragment; kIL=2 (H1 win) keeps TWO live = 128 regs in K-loop. **Halving accumulator** (either `SG_TUNED_DEC_GEMM_INTERLEAVE=1` or `TILE_N=64`) → 253 regs / 0 spills for every single-pass tail.

**Patches authored (NOT applied to main tree):**
- `0001-decoder-bf16-weight-prestage.patch`: bf16 weight cache (`dec_wbf_cache`) to enable future cp.async ring
- `0002-decoder-sam-scoped-outline.patch`: `__noinline__` scope on SAM second pass → forces regalloc boundary
- `0003-vit-sam-scoped-outline.patch`: same for ViT
- `0004-mamba-scope-noinline.patch`: mamba callee outline
- `0005-decoder-cpasync-ring-fwddx.patch`: cp.async double-buffered ring for fwd/dX

---

### 2.5 .regpressure/RING_REPORT.md — cp.async Ring (Lane D)

**What:** cp.async double-buffered ring for decoder fwd/dX GEMM engine. Replaces synchronous per-element staging with 16-byte `cp.async.cg.shared.global` (silicon-validated `primitives.cuh::cp_async_cg_16`).

**Key design choices:**
- `if constexpr (kRingAsync)` gated compile-time branch
- All 256 threads both produce and consume (collective barrier IS the correct signal at prefetch distance 1)
- dW path keeps lambda sources → synchronous engine (transposed-strided reads block ring)
- C1-T: transposed section in bf16 weight cache for dX B-operand (cache cost: 786 KB → 1.57 MB at d=128)

**Register/spill impact:**
- Single-pass cells: ZERO spills preserved (ring's in-loop regs absorbed exactly as margin predicted)
- SAM cells: ±120 B delta (≤1.5%) from 7.9-8.6 KB base → **negligible**

**PTX validation:** HGMMA sequence preserved verbatim (ring changes WHEN/HOW bytes move, not WHAT they are). Bit-identical accumulation by construction.

**Status:** authored as patch `0005-decoder-cpasync-ring-fwddx.patch`. NOT applied to tree (2026-06-12).

---

### 2.6 .phase2/REPORT.md — Phase-2 Authoring Report

**Date:** 2026-06-12. **Constraint:** CPU-only lane (no GPU process launched).

**NVSHMEM status (06-12):** NOT INSTALLED. TP authored against transport interface with bit-exact single-process loopback transport. `NvshmemTransport` compiled only under `-DSG_HAS_NVSHMEM`.

**What was authored (committed):**
- `csrc/fused/sm_90/tp_transport.cuh` — TP transport seam (loopback + NVSHMEM surface + fixed-order reduce)
- `csrc/fused/sm_90/tp_layer.cuh` — TP shard geometry/table/pack maps + sharded wgmma tile functions + reduce-point insertion map
- `csrc/fused/sm_90/pp_stage_decoder_tc.cuh` — PP stage spec/ownership + stage kernels + launchers (patch-gated)
- `tests/hw/tp_loopback_binding.cu`, `tests/hw/pp_stage_binding.cu`
- `grokking_optimizers/parallel/pipeline.py` — 1F1B schedule/driver/LoopbackP2P
- `grokking_optimizers/parallel/zero3.py` — FlatShardPlan, Zero3FlatParamStore, sharded checkpoint
- `grokking_optimizers/parallel/distributed_step.py` — `fused_train_step_distributed`

**GPU tests authored but NOT run:**
- `tests/hw/test_tp_loopback.py`, `tests/hw/test_pp2_loopback_determinism.py`
- `tests/hw/test_zero3_roundtrip.py`, `tests/hw/test_distributed_step.py`

**CPU tests RUN GREEN:** `tests/test_pipeline_schedule.py`, `tests/test_zero3_plan.py` — 55 passed

**Patches (NOT applied to tree):**
- `.phase2/patches/0001-dectc-layer-range-pp.patch` — layer-range templates on decoder tc tile functions. Required by PP stage header/test.
- `.phase2/patches/0002-parallel-init-exports.patch` — OPTIONAL re-exports in `grokking_optimizers/parallel/__init__.py`

**PTX identity:** patched vs unpatched fused_decoder_megakernel_tc<AdamW>: 16,543 PTX lines each; sole delta = one `mov.u32` scheduled one line earlier (scheduling jitter, zero semantic delta).

**Genuinely needs 8x:**
1. NVSHMEM-TP transport validation + go/no-go [THE residual]
2. TP insertion into production kernel body (transport-choice-dependent, at 4 marked points in tp_layer.cuh)
3. Real scaling measurements (DP 1→8, ZeRO-3 OOM threshold, PP bubble/microbatch sweeps)
4. Cross-rank graph capture with collectives
5. PP real P2P transport swap
6. ViT/Mamba PP/TP twins

---

### 2.7 .perf/phase1_status_audit.md + patches

The `.perf/` directory contains:
- `phase1_status_audit.md` — Phase-1 status audit
- `M0_mamba_integration_scaffold.patch` — Mamba integration scaffold
- `M0_mamba_wgmma_projections.patch` — Mamba wgmma projections

These are Phase-1 Mamba integration patches from the older campaign.

---

## Part 3: Discrepancies vs Claimed State

The CLAIMED state (from RESUME.md/PROGRESS.md/SESSION_CONTEXT.md) vs what the memory files show:

1. **"full TP; cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs"** — flagship-distributed-config.md (written 2026-06-25, same day as the current campaign) says TP/PP are STUBBED and production launcher is NOT templated on ParConfig/CommCtx. The session memory is authoritative over the claimed state summary.

2. **"11-optimizer ranking" / "overfit placeholder"** — .audit_notes.md from the older campaign shows 8/11 grok achieved with 3 SuperGrok-family optimizers failing due to bilevel meta-net dynamics. This was on the OLD branch (claude/h100-audit-maximal) with the old decoder (mod-97 grokking task, d=128). The current campaign is a different benchmark (1.5B flagship).

3. **NVSHMEM:** OLD campaign (06-12) = NOT installed. Current campaign = installed (3.7.0). This is a resolved blocker.

4. **The cp.async ring** was authored as a patch (06-12) but NOT applied to the main tree. Its status in the current branch is unknown from memory alone.

5. **The regpressure patches** (0001-0005) were authored 06-12 as patches NOT applied to the main tree. Whether they were applied in the current branch is unknown from memory alone.

---

## Summary: Key Standing Directives

| Directive | Status |
|---|---|
| L3-TC kernels ONLY (no scalar) | ACTIVE |
| Use prebuilt binaries when available | ACTIVE |
| Read code exhaustively (no grep-skim) | ACTIVE |
| Parallelize aggressively with agents | ACTIVE |
| Never pause to ask priority/what-next | ACTIVE |
| CuTe device-side atoms for megakernel GEMMs | CLAIMED done (SG_TUNED_GEMM_ENGINE) |
| ncu blocked on RunPod — use counter-free fallbacks | ACTIVE (unless pod relaunched) |
| NVSHMEM 3.7.0 installed — in-kernel device all-reduce buildable | ACTIVE |
| 33-cell roofline graph (post-build deliverable) | QUEUED |
| Dead-code cleanup + LOC report (post-build deliverable) | QUEUED |
| Adaptive 3D-5D parallelism, self-specializing kernel | DESIGN DIRECTIVE |
| Robust resource-fit planner (not GPU-count-based) | DESIGN DIRECTIVE |
| Dataset interface must be PLUGGABLE (not 3-way hardcode) | DESIGN DIRECTIVE |
| EP as 5th axis (MoE) + front-end inference function | TO DO |
