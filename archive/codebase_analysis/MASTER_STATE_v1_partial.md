# SuperGrok2 — MASTER STATE (authoritative synthesis)

> Synthesis lead consolidation of the 33-reader analysis + direct code/git verification.
> Repo: `/workspace/SuperGrok1.5` · branch `claude/custom-optimizer-analysis-HFYhg` · **HEAD `e69df73`** (verified live).
> Date of synthesis: 2026-06-25. Code-evidence is preferred over summary-doc claims throughout;
> every claim is tagged with its strength: **[verified]** (I checked the tree/git), **[doc]** (single source doc),
> **[doc-corroborated]** (multiple docs agree), **[claimed-unverified]** (asserted in a log/ledger, not re-checked here).

---

## 1. EXECUTIVE SUMMARY

**SuperGrok2** is a portable, self-adapting, maximum-performance training stack — PyTorch-shaped Python over
hand-authored CUDA/C++ — whose defining idea is that **one entire training step runs as a single persistent
fused CUDA megakernel** per `(model × optimizer)` cell: one `__global__` launch executes P0 zero → P1 fused
fwd+bwd → P2 deterministic grad reduce → P3 optimizer tail, with phases separated only by an in-kernel,
hand-built sense-reversing **GridBarrier** (no per-step relaunch, no cooperative launch; one CTA per SM).
It spans **3 model families** (transformer-decoder, ViT, Mamba-3) × **11 optimizers** (AdamW, Lion, Grokfast,
GrokAdamW, LookSAM, Prodigy, NeuralGrok, Muon, SuperGrok1.1, SuperGrok1.5, SuperGrok2) × **3 arch backends**
(sm_90 Hopper / gfx942 CDNA3 / tpu_v6e), composed by a generator-emitted dispatch table into 99 cells (33 per
arch). Correctness is a **HARD gate**: fp64 parity (rel 1e-4) **AND** A/A/A bit-determinism (`torch.equal` ×3),
with all legal rewrites *transport-only* (the ascending-k fp32 reduction order is preserved). `compile.py`
(32,900 lines [verified]) is the Optuna-TPE superoptimizer/autotuner. The latest campaign added a CuTe-atom GEMM
engine, three ~1.5B **flagship** layouts, full 3D–5D parallelism (DP×TP×PP×SP + ZeRO-3) with an **in-kernel
device-NVSHMEM all-reduce**, a resource planner, and a memory-strategy layer.

**THE SINGLE "WHERE WE LEFT OFF" STATEMENT:**
The hardest piece — cross-GPU **8-GPU NVSHMEM TP all-reduce** — is bit-exact validated (2/4/8-GPU smokes), and
all three flagship models launch single-GPU. The *one* thing blocking the real "one 1.5B model trained across all
8 H100s" run is the **TP megakernel data-path fix**: the WIP patch `/workspace/phase6/tp_datapath_fix_WIP.patch`
[verified present, 23.7 KB, 2 files] code-fixes bugs **A** (per-rank weight divergence → solved by full-width
replicated compute) and **B** (head divisibility, `kHeads=25 % TP=8`), but it is **UNGATED and uncommitted**, and
bug **C** (confirm the +0x1dc30 IMA is cleared under compute-sanitizer on the live 8-GPU run, with cross-rank loss
agreeing + descending, and SingleGPU pytest still 19/19 byte-identical) **remains open**. ETA ~1–2 hr of GPU work.

---

## 2. ARCHITECTURE (the full stack)

### 2.1 Front-end API & component model
- **44 components → 99 cells** = (11 opt × 3 model × 3 arch = 33+9 model-cells +2 dispatch/compile). sm_90 slice
  = **33** real `(model,opt)` cells. [doc-corroborated: README §1; LEDGER.ground_truth]
- **opt_id integer contract** (hand-maintained, *no auto drift-guard* — flagged gap): `adamw=0, lion=1, grokfast=2,
  grokadamw=3, looksam=4, prodigy=5, neuralgrok=6, muon=7, supergrok11=8, supergrok15=9, supergrok2=10`. The
  Python list, the C++ dispatch, and `OptId` in `opt_components.cuh` must agree by hand. [doc]
  `csrc/fused/sm_90/opt_components.cuh` exists (524 lines) [verified]; dispatch lives at
  **`csrc/bindings/dispatch.cpp`** [verified — *not* `csrc/fused/sm_90/dispatch.cpp` as several digests cite].
- Python entry: `grokking_optimizers/dispatch.py` (2,013 lines [verified]); prebuilt `_ops.so` exposes 5 callables
  in the fused build: `detect_arch, fused_step, sg2_fused_step, sg2_meta_optimizer_tail, sg2_ws_stride`.
  [doc: PROGRESS "KEY PHASE-0 FINDINGS #2"] SG2 has a dedicated launcher/entry (`ops.sg2_fused_step`).
- **Single-source math guarantee**: `scripts/check_math_single_source.py` (wired into `--self-test` as
  `math_drift_guard`) fails the build if a consumer stops `#include`-ing the canonical `csrc/algorithms/<opt>.h`,
  if Adam moment math is re-inlined, or if canonical math changes without `--update-manifest`. LIVE + clean
  (exit 0, one expected SG2 WARN). [doc-corroborated] **Gap:** the `.py` optimizer sources (SG11/15/2) are not
  checked against the `.h`, and a GrokAdamW EMA re-inline is uncaught. [doc: LEDGER C14]

### 2.2 The 3 models (race scale → flagship scale)
| model | race/dev scale | flagship (~1.5B) | notes |
|---|---|---|---|
| transformer-decoder | d128/h4/L2/vocab99/seq4 | **d=1600, h=25, L=48, 582 tensors, 1,475,884,899 params** | causal; counting-sort owner-scan embedding; cleanest (Fork-B + layer-independent smem) |
| ViT | patch49/16patch/d128/h4/L2/cls97 | **d=1664, L=48, 584 tensors (~1.596B)** | full bidirectional attn; CLS@pos0; `kSeq=17→kTileM=1088` |
| Mamba-3 | d128/L2/state128/head64/seq8 | **d=2048, h=32, L=24 (~1.265B)** | genuine Mamba-3 SISO (arXiv 2603.15569); batch-parallel scalar register scan (seq=8); wgmma dW machinery bypassed |
[doc-corroborated: LEDGER flagship_scale; PROGRESS session-2; README]

### 2.3 The 11 optimizers (the P3 tail)
The optimizer is **not a separate kernel** — it is **P3 of the same megakernel**. State is one flat caller-owned
buffer `[m | v | extra]` over `3*total` floats + a loss slot; `extra` is overloaded per optimizer (grokfast/
grokadamw EMA, OR Prodigy `s`, OR LookSAM `sam_dir`, OR SG11/15 `mu`). `apply_optimizer<Opt>` is an `if constexpr`
ladder; every branch calls canonical `csrc/algorithms/<opt>.h`. No-silent-fallback: launcher `switch`
`default: cudaErrorInvalidValue` → thrown runtime_error. [doc: CODEBASE_EXPLAINED]
Epilogue-fusability (DESIGN App-A): **5/11 fully fusable** (adamw/lion/grokfast/grokadamw/neuralgrok);
prodigy/sg11/sg15 need one reduce phase; muon/sg2 not fusable. [doc]

### 2.4 The persistent L3-TC megakernel substrate (the heart)
- **Entry points** (templated on `<Opt>`, now also on `<ParConfig<...>>`): `fused_decoder_megakernel_tc`
  (`csrc/fused/sm_90/fused_decoder_megakernel.cuh`, 1631 lines [verified]), `fused_vit_megakernel_tc`,
  `fused_mamba_megakernel_tc`. Launched via plain `<<<grid,block,dyn_smem,stream>>>` — **NOT cooperative**.
- **gridDim.x == #SMs, one persistent CTA/SM** (`cudaDevAttrMultiProcessorCount`); hard occupancy gate ≥1 or
  `cudaErrorLaunchOutOfResources` (refuse, don't hang). SM pin via `%smid`. [doc: CODEBASE_EXPLAINED §1–2]
- **GridBarrier** at `csrc/fused/megakernel_common.cuh` [verified path — *not* `csrc/fused/sm_90/`]: two global
  atomics (`g_arrived`,`g_generation`) + sense-reversing generation; one release + one acquire `__threadfence()`
  (an extra fence is a >10% perf + correctness footgun); `__nanosleep` exp backoff cap 1024ns; `sync_reset` folds
  task-queue zeroing into the last-arriver critical section, collapsing 4 barriers → 2.
- **TaskQueue** (work-stealing): single atomic `g_next_task`; the fp32 summation order is never affected
  (determinism by construction).
- **5-phase / 4-barrier layout** (decoder reference): P0 zero + bf16 weight pre-stage → B0 → P1 token-tile
  fwd+bwd (barrier-free within tile, loss fp32 per-CTA) → B1 → P2 dW GEMM + grad assembly + LN-vec reduce +
  fp64 loss reduce by CTA0 → B2 (sync_reset) → optional staged P2.x phases → P3 optimizer tail (work-steal, no
  trailing barrier). Staged insertions, each `if constexpr`-gated so every other cell is byte-identical: P2.4 SAM
  2nd backward (looksam/sg11/sg15/sg2); P2.5 grad-norm clip (grokadamw/neuralgrok); P2.6 Prodigy d-reduce (fixed
  partition for determinism); P2.7 Muon Newton-Schulz (5 iters); P2.45/P3-SG2 meta-net.
- **GEMM = hand-rolled in-kernel ss-wgmma** at `csrc/backends/cuda/sm_90/wgmma.cuh` (775 lines [verified]):
  `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`, 64-bit smem descriptor, fixed **ascending-k** order
  (first ScaleD=0, rest ScaleD=1). bf16 inputs, **fp32 accumulation**. Race precision is **bf16**, not TF32.
- **CuTe-atom GEMM engine (NEW, landed)**: `wgmma.cuh` rewritten to CuTe device atoms (`cute::GmmaDescriptor`,
  `MMA_64xNx16_F32BF16BF16_SS::fma`, `warpgroup_arrive/commit/wait`) behind **`SG_TUNED_GEMM_ENGINE`**
  (default 0 = hand PTX, byte-identical; 1 = CuTe). Validated bit-equal end-to-end through the real decoder
  megakernel (ENGINE0 vs ENGINE1 loss+grad maxabs 0.0; ENGINE1 fp64 rel 2.85e-5). [claimed-unverified: PROGRESS
  session-2; driver `phase1/cute_decoder_validate.py`]
- **cp.async ring, not TMA** (`tile_pipeline.cuh`): decoder `SG_TUNED_DEC_FWD_PIPE` baked at 1 (deepened ring,
  +1.49×), `SG_TUNED_DEC_FWD_STAGES` baked at 4. TMA is the deferred perf step (host-built `CUtensorMap` only).

### 2.5 The codegen / autotune / compile brain
- **`grokking_optimizers/compile.py`** — 32,900 lines [verified]. Per-(opt,model,arch) autotuner. Two-phase AOT
  (`build_aot`, cache JSON CACHE_VERSION==4) / JIT (`build_jit`). **Optuna TPE Bayesian** (default), SQLite-
  persisted/resumable, `BayesianEarlyStopper`, top-K ±2 local refine. Learned **CostModel**
  (XGBoost→sklearn-GBR→numpy-ridge), default-ON, prunes pre-build. `MultiGPUTimingPool` (compile.py:4874) shards
  *autotune variants of one cell* across visible GPUs — **NOT** a 33-cells-across-8-GPU scheduler (cells run
  sequentially on dev0). [doc: LEDGER C7/C8]
- **fp64-gate-in-loop** (landed `af9b720`): `_default_correctness_hook` re-runs `run_cell_gate` (fp64 + A/A/A)
  and demotes failing top-K, fail-closed; winner left UNSET if none clear. **RG4-scale caveat:** the hook builds
  at `grokking_race_v2.DEFAULT_CONFIG` (**d=128, L=2**) despite "PRODUCTION scale (d=2048)" comments — non-default
  shapes must be re-gated at d=2048 by hand. [doc: docs_core §5]
- **`megakernel_codegen.py`** parameterizes the layout by `(d,layers,vocab,seq,heads)`; defaults == old globals
  → production header byte-identical; `--decoder-layout-flagship` emits the d=1600/L48 header. [claimed]
- **Self-test expected count = `_SELF_TEST_EXPECTED_COUNT = 265`** at `compile.py:26861` [verified]. The README's
  "156 passed / 152/152" is therefore **stale**.

### 2.6 3D–5D parallelism + ZeRO-3 (NEW)
- North-star: **4D parallelism = DP×TP×PP×SP** (sequence-parallel the 4th axis) **+ ZeRO-3**, optional 5th **EP**
  axis (byte-identical future-MoE seam; current 3 models are 4D). [doc: PROGRESS REFINED GOAL]
- `csrc/fused/sm_90/parallel_config.cuh` (279 lines [verified]) — the `ParConfig<TP,...,Z3>` template the
  megakernel is now generic over. `csrc/fused/sm_90/tp_transport.cuh` (304 lines [verified]) — `NvshmemTransport`
  (in-kernel device-NVSHMEM all-reduce over NVLink, gated `-DSG_HAS_NVSHMEM=1`).
- **Cross-GPU bring-up VALIDATED**: `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` (357 lines [verified]) +
  `nvshmem_bringup_ext.py` + `tests/hw/nvshmem_smoke.py`: UID bootstrap (no MPI/PMI) over torch.dist →
  `nvshmemx_hostlib_init_attr` → `team_split_strided` → collective `nvshmem_malloc` → in-kernel device all-reduce.
  Smokes **2/4/8-GPU PASS** (8-GPU expected 36.0 got 36.0 bit-exact on all 8 ranks). Container quirk (NVLink-SHARP
  multicast forbidden) mitigated via `NVSHMEM_DISABLE_NVLS=1 + NCCL_NVLS_ENABLE=0`. [claimed-unverified: PROGRESS
  HEAD 03bd3f0; bring-up source verified present]
- **CPU foundation**: 84 parallelism tests PASS (LEDGER says 14/15 GPU tests, one grokfast parity 3.77e-5 vs 3e-5
  tol); dp2-loopback runs the real reduce-scatter/sharded-opt/all-gather, cross-rank + A/A/A bit-identical. [doc]

### 2.7 Resource planner + memory-strategy + datasets
- **Resource planner** (`81f1bfb`): robust workload×hardware → execution config (parallelism + memory strategy +
  kernel knobs); 10/10 tests. [claimed] **memory-strategy** (offload/recompute/stream gates) for large-on-few-GPU.
  **size-adaptive CTA-tiling selector**. Design principle: the kernel is templated on the deployment config and
  `if constexpr` folds in EXACTLY the machinery the config needs (distributed→all-reduce, single→none).
- **Datasets layer**: Layer-A pluggable `data_source` seam landed (`7f9e772`). **Layer-B (real data) NOT wired** —
  current data is mod-97 (decoder a·b⁻¹; ViT MNIST a+b; Mamba chained div); no FineWeb/ImageNet/GiftEval yet
  (greenfield). `evaluate()` forwards the whole held-out tensor (catastrophic at flagship); EarlyStopper stops on
  VAL acc≥0.95 not test. [doc: LEDGER ground_truth]
- **Benchmark spec**: MODE=D = 11 opt × 5 seeds[42,123,456,1337,3407] × 4 splits[0.10,0.25,0.50,0.80] × 3 models
  = 660 runs; step_budget 20000. [doc: LEDGER]

---

## 3. WHAT IS DONE & VALIDATED (with evidence)

| Item | Evidence | Strength |
|---|---|---|
| Single-GPU sm_90 path is sound | 2026-06-17 8-agent audit: 33-cell dispatch consistent Python≡C++≡opt-id; fp64+A/A/A genuinely anchored | [doc-corroborated] |
| Decoder/ViT/Mamba SingleGPU pytest | decoder **19/19**, ViT **21/21** byte-identical; Mamba **3/5** (2 PRE-EXISTING fails: B_bias-tol + obsolete proj_dw) | [claimed: RESUME §3] |
| CuTe-atom GEMM engine bit-identical | ENGINE0 vs ENGINE1 loss+grad maxabs 0.0 through real decoder; fp64 rel 2.85e-5; A/A/A deterministic | [claimed: PROGRESS S2; driver exists] |
| Flagship decoder runs on silicon (L=48) | `flagship_smoke.py`: 1,475,884,899 params, full fwd→bwd→AdamW on 1×H100; loss 4.585 ≈ ln(99)=4.595; all 1.476e9 grads finite; A/A/A bit-identical ×3 (4.585046768188477) | [claimed: PROGRESS; logs flagship_smoke{,2,3,4}.log] |
| Flagship TC megakernel fits 1 CTA/SM @ d=1600/L48 | ptxas: 255 regs, 25,360 B static smem, 23.5 KB stack; smem is LAYER-INDEPENDENT (kernel streams layers; acts→HBM) → RISK-2 (smem wall) OVERTURNED | [claimed: PROGRESS S2#3] |
| All 3 flagships LAUNCH | Mamba smem redesign 19.56MB→192.97KB (<227KB cap) via layer-streaming + scratch-to-HBM (`9936308`); Mamba TC launched 1.265B, loss 4.577≈ln97 | [claimed: PROGRESS] |
| 8-GPU NVSHMEM TP all-reduce | smokes 2/4/8-GPU PASS, 8-GPU 36.0 bit-exact on all ranks (HEAD `03bd3f0`) | [claimed; source verified] |
| Dead-code cleanup | removed 8,089,083 lines / 528 files of provably-dead artifacts (95.7% of repo text) `8643cc2`; true source ~361K / 1047 files; layouts byte-identical post-removal | [verified: git log] |
| Flagship roofline deliverable #1 | `/workspace/phase6/roofline_flagship.{png,csv}` — nsys-measured 10 cells; occupancy-bound | [verified: files present] |
| Decoder bottleneck map (d=2048/L2/B16384) | 617.43 ms/step, GEMM 64.15 TF/s = **6.49%** of 989 TF/s; P1_fwd 27.6% + P1_bwd 27.3% + P2_dW 16.6% (GEMM-bearing 71.5%); B1 barrier idle 16.7%; P3 tail 5.9% | [doc: PROGRESS BOTTLENECK MAP] |
| Resource planner / parallelism CPU tests | 84 parallelism tests pass; 35 parallel/config/resource tests pass; resource planner 10/10 | [claimed] |

---

## 4. WHAT IS IN-FLIGHT / REMAINING (ordered, with ETAs)

1. **★#1 — TP DATA-PATH FIX (the live one-model-across-8 training run).** The 8-GPU run surfaced 3 committed-
   source megakernel bugs the tiny-FFN loopback never exercised (compute-sanitizer scoped):
   - **Bug A — per-rank weight divergence** (PROGRESS bug #2): `dectc_wbf_convert`/`dec_bind` read the FULL weight
     matrices identically on every rank (`comm.tp_rank` unused) → all ranks compute slice-0 → all-reduce sums 8
     identical partials (degenerate). **FIXED** in the WIP patch by switching the kTPComm path to **full-width
     replicated compute** (every rank computes the identical full gradient; the 4 reduce-points publish
     `result/P` and the ascending-pe sum reconstructs it — a mathematical identity that still genuinely exercises
     the NVLink all-reduce). [verified in patch lines 52–80]
   - **Bug B — head divisibility** (PROGRESS bug #3): flagship `kHeads=25` not `%TP=8` → `Hloc=3,Dloc=200` but the
     invariant `Dloc==Hloc·kDhead=192` is violated. **FIXED** by the same full-width-replicated attention (head-
     shard demoted to a future opt). [verified in patch]
   - **Bug C — IMA (OPEN)** (PROGRESS bug #1): 87 invalid 4-byte global writes in
     `fused_decoder_megakernel_tc<AdamW,ParConfig<8,8,1,1,Z3>>+0x1dc30` (wild ptr ~35 GiB below the NVSHMEM heap).
     The WIP patch adds an **OOM-safe workspace guard** (the flagship workspace is hundreds of GB; a failed
     `cudaMalloc` left a null/stale base → the wild write) returning `cudaErrorMemoryAllocation` instead of a
     silent null-base launch [verified in patch, launcher hunk]. **Still to DO:** apply the patch, rebuild,
     re-run 8-GPU, and **confirm under compute-sanitizer that the IMA is gone**, cross-rank loss agrees + descends.
   - **State of the patch:** `/workspace/phase6/tp_datapath_fix_WIP.patch` exists (23.7 KB, touches
     `mega_decoder_real_adamw_tc_launcher.cu` + `model_stage_decoder_tc.cuh`), is **UNGATED and NOT committed/
     applied** [verified]. Gate to close: SingleGPU pytest **19/19 byte-identical** + 8-GPU sanitizer-clean +
     cross-rank loss agrees + descends. **ETA ~1–2 hr.**
2. **Real-data benchmark (Layer-B).** Wire FineWeb-Edu / ImageNet-1k / GiftEval into the datasets Layer-A seam
   (`impl_diffs/datasets_v2.md`), replace mod-97/overfit, run the real 11×3 ranking. **ETA ~3–5 hr + the run
   (GPU-hrs–days).**
3. **Full 33-cell roofline** (Mamba now launches): re-run `/workspace/phase6/roofline_campaign.js` for all
   3 models × 11 opts (nsys). **ETA ~1–2 hr.**
4. **ViT re-measure** at the saturating batch (~2k) — the roofline `ncta=4` was a conservative artifact. **~0.5 hr.**
5. **Per-model TP extension (ViT/Mamba)** for the full 11-opt distributed benchmark; then the benchmark RUN.
6. Optional/known-debt: `VIT_DW_SPLITK 4→1` (−25.5 GB byte-identical); 56-line Mamba dead-source removal
   (`tc_dump_outproj_operands` + sole-caller test); Mamba/SG2 single-GPU occupancy (`ncta=1`) needs fewer
   always-on opt carves or a 2nd-GPU shard; **P3 optimizer tail un-autotuned** (autotune produced 0 winners —
   SESSION directive "optimizers must be MAXED not just fused"); GEMM perf roadmap = TMA(step4, ~1.4× on
   issue) + wgmma-accumulator-pipeline (the real ~15× lever, C7515 serialization) + barrier load-balance.

---

## 5. KNOWN BUGS / DEAD CODE / ARCHIVE GAPS (consolidated)

**Active numeric-path bugs:** *(none confirmed — drift audit = 0/11 active.)*

**Bugs (non-numeric / scale / harness):**
- **BUG-04**: mamba staged-opt scratch un-gated → ~199 GiB OOM at d≥1024 (blocks muon/neuralgrok/looksam mamba at
  scale; VRAM sizing bug, not a missing optimizer). [doc]
- **SG2 flagship workspace**: per-CTA meta-net scratch O(50·Nmax)·nCTA ≈ **509 GB** at d=1600 (DEEP LIMIT; needs
  streamed redesign OR TP-sharding to shrink per-rank Nmax). [doc: flagship_smoke]
- **TP IMA (bug C)** — open, see §4.1.
- `has_kernels()` returns False on a healthy `_ops.so` (probes 4 removed symbols, `dispatch.py:~542`); cleanup-#10
  regression in dev/verification harnesses; **verify_all/profilers DOA** (target removed TUs). [doc: 2026-06-17 audit]
- `--grad-hooks` eager path raises; `FusedOptState st;` uninit (benign for AdamW); `PIPE==2` engine half-wired
  (latent null-deref if built; PIPE=0/1 safe); zero3 non-contig broadcast (verification struck this + mamba-uninit
  + PIPE==2-compiles as FALSE/benign). [doc: audit, with verification corrections]
- `device_profiling` import dead (ModuleNotFoundError); `_MMA_NATIVE_LOADS_WIRED=False` disables compile.py wgmma
  PTX (synth GEMM falls to scalar triple-loop); inert ABI guard (`GROK_ABI_SCHEMA` exported, no py assert);
  dead `MambaModel`/`SelectiveSSMLayer` + no-op `_maybe_wrap_cuda_graph`. [doc: LEDGER verified_bugs]

**Gate-coverage caveats (8/11 optimizers — MISSING/toy-scale gates, NOT confirmed drift):** grokadamw/prodigy
multistep gate missing; muon/neuralgrok blocked by BUG-04; looksam looser SAM tol; sg11 warmup gate CLI-only not
CI; sg15 no warmup gate; sg2 CSA oracle co-wrong (HIGH). The `51098e0` SG11 fix = cos(grad,MU)→cos(grad,MOMENTUM)
in the staged gate. green: adamw, lion, grokfast. [doc-corroborated]

**Dead code / ghost modules (verified):**
- `grokking_optimizers/codegen.py` — **does NOT exist** [verified]; emitter ghost-import → macros-only fallback,
  `_emitted_sources` never populated. `grokking_optimizers/compile_config.py` — does NOT exist (in-file load_config).
- `csrc/backends/cuda/sm_90/mma.cuh` — **still EXISTS** (the 820-LoC host-launched CUTLASS path) [verified];
  audit-flagged DEAD (0 includers on the persistent path, used only for L1 Muon-NS/SG2). **It was NOT removed by
  the 8.09M cleanup** — contradicting digests that imply removal.
- `SG_TUNED_*` GEMM-stage/swizzle/tma/wgmma_shape dims = dead/auto-pinned (no kernel `#ifndef` reads them).

**Archive gaps (generated-but-uncompilable):**
- **#1**: `megakernel_codegen.py --write-all` 33 per-cell `mega_<model>_<opt>.cu` `#include fused_megakernel.cuh`
  + call `launch_fused_megakernel<...>` — BOTH removed in the pure-L3-TC refactor (`8b30ea8`) → generated cells
  DO NOT COMPILE. The real runtime path = the 6 reference `mega_<model>_real_adamw_tc.cu` `_tc` cells. [doc]
- **#2**: `launch_<opt>.cu` + `models/*.cu` 5-LOC shims referenced by verify_all/profile in NO git ref + no
  generator → missing (affects per-component profiling, not the 33 fused cells). [doc]

---

## 6. DISCREPANCIES & RECONCILIATION

Rated **confirmed** (I re-checked) or **suspected** (single-source, plausible).

1. **HANDOFF.md is STALE — different campaign. [CONFIRMED]** `HANDOFF.md` is dated **2026-06-12** and scoped to
   branch **`claude/h100-audit-maximal`** [verified via `git log -1 HANDOFF.md`]; it describes the *previous*
   campaign (33-cell L3-TC roofline, `.regpressure/` static patch series, Lane A–E). The current campaign is on
   `claude/custom-optimizer-analysis-HFYhg`. **Use `RESUME.md` (2026-06-25) [verified date], not HANDOFF.md.**
2. **HEAD pointer disagreement across the state files. [CONFIRMED]** `LEDGER.json` says `head: c29ed4e` (the
   clone point, 2026-06-24); `PROGRESS.md` header says "Branch HEAD c29ed4e" but its body narrates progression to
   `03bd3f0`/`e69df73`; **the live tree HEAD is `e69df73`** [verified `git rev-parse`]. LEDGER.json is simply
   frozen at Phase-0 (2026-06-24T21:40Z) and was never advanced. Reconcile: **`e69df73` is authoritative.**
3. **Core architecture docs PREDATE the campaign. [CONFIRMED]** `CODEBASE_EXPLAINED.md` and `SESSION_STATE.md`
   are dated 2026-06-17 and never mention NVSHMEM TP, the CuTe-atom engine (`SG_TUNED_GEMM_ENGINE`), the resource
   planner, the 3 flagship layouts, or the 8.09M-line cleanup. They are the correct *architectural reference* but
   STALE on live state. Git log [verified] confirms all that work landed after them (`5733af5`…`9936308`,`03bd3f0`).
4. **README "CUTLASS Sm90 collectives for the model GEMMs" vs reality. [CONFIRMED]** The shipping persistent
   megakernel uses **hand-rolled in-kernel ss-wgmma** (`wgmma.cuh` [verified, 775 L]); CUTLASS `mma.cuh` is
   host-launched-only, used for L1 Muon-NS/SG2, and audit-flagged DEAD (still present [verified]).
5. **README "Implementation-maximal, only remaining work is on-silicon validation" vs audit/session. [CONFIRMED]**
   Contradicted by: P3 tails un-autotuned, 8/11 gate-coverage caveats, 6.48% roofline with fixable inefficiency,
   BUG-04 open, the TP data-path bug C open, and Layer-B datasets greenfield. README is aspirational.
6. **README self-test "156 passed / 152/152" vs code. [CONFIRMED]** `compile.py:26861` sets
   `_SELF_TEST_EXPECTED_COUNT = 265` [verified]; the README counts are stale.
7. **README TF32-for-FP32 vs bf16 race precision. [CONFIRMED via docs]** DESIGN §5/App-B: race precision is bf16;
   TF32 only as the `mma.cuh` L1 fallback.
8. **DESIGN-TC-PIPELINE.md "No in-kernel wgmma anywhere / scalar fp32 triple loop". [CONFIRMED]** Superseded —
   `wgmma.cuh` exists [verified]. Keep DESIGN only as the historical rationale for the now-shipped TC engine.
9. **ENV_SNAPSHOT egg pin `github.com/peterc04/SuperGrok1.5@4af83c3` vs "LOCAL-ONLY never push" + HEAD e69df73.
   [CONFIRMED]** The pinned SHA `4af83c3` is far behind HEAD and references a github remote that the doctrine says
   is never pushed. Treat the pin as a packaging artifact; the working tree is the source of truth.
10. **TP bug LABELS are inconsistent across docs. [CONFIRMED]** PROGRESS numbers them (1)=IMA, (2)=weight-offset,
    (3)=head-divisibility; RESUME labels A=weight-shard, B=full-width-attn(head), C=IMA; the **patch comment
    itself** calls head-invariant "bug B" and out-of-bounds shard writes "bug C" (a third mapping). The *work* is
    unambiguous (verified in the patch): weight-divergence + head-divisibility are CODE-FIXED via full-width
    replication; the IMA OOM-guard is added; the live-sanitizer CONFIRMATION is the open item. Adopt the RESUME
    A/B/C labels (matches the task framing: A/B fixed, C open) and note the cross-doc inconsistency.
11. **"33 megakernels" vs binaries. [CONFIRMED]** The 33 generated per-cell `.cu` are DEAD (won't compile, archive
    gap #1); the *real* runtime path is **3 per-model `_tc` kernels over `OptId`** (compiled, run via `_ops.so`).
    "33 cells" is a logical/dispatch count, not 33 binaries.
12. **Mamba flagship "UNLAUNCHABLE (#30)" (roofline section) vs "ALL 3 LAUNCH". [CONFIRMED — resolved in sequence]**
    The roofline section recorded Mamba unlaunchable (19.56MB smem) as a *finding*; the later smem redesign
    (`9936308`/`d75d178`) resolved it (→192.97KB). Same doc, different times; the LATER state holds.
13. **ncu HW counters DENIED. [CONFIRMED — environmental]** `ERR_NVGPUCTRPERM`, no CAP_SYS_ADMIN in container;
    all roofline/occupancy is nsys + static (cuobjdump/ptxas) + CUDA-event wallclock + analytical FLOP/byte. Not a
    bug; a measurement constraint that must be flagged to the owner.
14. **Flagship 11-opt ranking is an OVERFIT placeholder. [CONFIRMED]** `/workspace/phase6/flagship_11opt_ranking.txt`
    [verified] header: "B=16, fixed-batch overfit"; 9/11 OK, muon+sg2 "FITS/SLOW" (ncta=1). Not a real ranking —
    it proves runnability, not optimizer quality. The real ranking awaits Layer-B data (§4.2).

---

## 7. EXACT RESUME PLAYBOOK

### 7.0 Restore the instance (deps lived outside /workspace, deleted on closure)
```bash
pip install nvidia-nvshmem-cu12 optuna ruff nvidia-ml-py
cd /workspace/SuperGrok1.5
git config user.email "<owner-email>" && git config user.name "SuperGrok2 session"
mkdir -p /root/.claude/projects/-/memory && cp .session_memory/*.md /root/.claude/projects/-/memory/
export NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
export NVSHMEM_DISABLE_NVLS=1 NCCL_NVLS_ENABLE=0
# build env (from HANDOFF pod constraints — verify on the new pod):
export TORCH_EXTENSIONS_DIR=/workspace/SuperGrok1.5/build/torch_ext   # /dev/shm is NOEXEC
export CUDA_HOME=/usr/local/cuda                                       # 12.4; PATH nvcc is a caching shim
export TORCH_CUDA_ARCH_LIST=9.0a
# MAX_JOBS=24 for JIT variant builds (cicc ~5.9 GiB/TU; 200 GiB cgroup)
```

### 7.1 Confirm git state
```bash
git -C /workspace/SuperGrok1.5 rev-parse HEAD          # expect e69df73
git -C /workspace/SuperGrok1.5 branch --show-current   # claude/custom-optimizer-analysis-HFYhg
git -C /workspace/SuperGrok1.5 status --short           # only .pytest_cache churn expected
```

### 7.2 ★ The 8-GPU TP data-path fix (the #1 item)
```bash
cd /workspace/SuperGrok1.5
# 1. Regression baseline BEFORE the patch (must hold AFTER):
pytest tests/hw/test_decoder_tc.py -m hw -q -s          # expect 19/19, byte-identical SingleGPU
# 2. Apply the WIP fix (bugs A + B + the OOM/IMA guard):
git apply /workspace/phase6/tp_datapath_fix_WIP.patch
# 3. Rebuild the TP8 scratch path (torch JIT load() OMITS -rdc -dlink → use the manual 3-step build):
bash tuning/_tp8_build.sh
# 4. Run one flagship 1.5B decoder across all 8 H100s (TP8 + ZeRO-3, in-kernel NVSHMEM all-reduce):
torchrun --nproc_per_node=8 tuning/_tp8_run.py --steps 30 --ncta 64
#    (_tp8_run.py: UID bootstrap → team_split(0,1,8) → sym malloc → JIT-build _tp8_scratch_pybind.cu
#     with the flagship layout force-included + -DSG_HAS_NVSHMEM=1 -rdc=true → TP-shard 1.476B weights)
# 5. Bug C gate — confirm under compute-sanitizer (memcheck) that the +0x1dc30 IMA is GONE:
compute-sanitizer --tool memcheck torchrun --nproc_per_node=8 tuning/_tp8_run.py --steps 3 --ncta 64
```
**GATES (all must pass to close):** SingleGPU pytest **19/19 byte-identical** · 8-GPU run **sanitizer-clean
(0 IMA)** · **cross-rank loss agrees** · **loss descends**. Then commit the patch (LOCAL-ONLY, never push) with
the verdict.

### 7.3 After the TP fix (in order): §4.2 Layer-B datasets → §4.3 full 33-cell roofline → §4.4 ViT re-measure →
per-model TP extension → the 11×3 benchmark RUN.

### Standing constraints
- Commits are **LOCAL-ONLY, never push.** HARD gate = fp64 parity (rel 1e-4; muon 2e-3; SAM 2.5e-2/3e-2) **AND**
  A/A/A `torch.equal` ×3. All rewrites transport-only (preserve ascending-k fp32 order). ncu HW counters DENIED
  (nsys/static/wallclock only). One heavy GPU build at a time; never `kill -9` CUDA clients under MPS.

---

## 8. MAP OF THE VOLUME (where everything lives)

**Repo (the source of truth):** `/workspace/SuperGrok1.5` (branch `claude/custom-optimizer-analysis-HFYhg`,
HEAD `e69df73`).
- `csrc/algorithms/<opt>.h` — canonical per-element optimizer math (one def/opt; `SOURCE_OF_TRUTH.md` is the
  contract; SG2 bilevel adjoint in `supergrok2_bilevel_adjoint.h`).
- `csrc/fused/sm_90/` — `opt_components.cuh` (524 L), `fused_{decoder,vit,mamba}_megakernel.cuh`,
  `model_stage_*_tc.cuh`, `mega_<model>_real_adamw_tc{,_launcher}.cu` (the **real** runtime cells),
  `parallel_config.cuh` (279 L), `tp_transport.cuh` (304 L), `nvshmem_bringup_pybind.cpp` (357 L).
- `csrc/fused/megakernel_common.cuh` — **GridBarrier/TaskQueue** (note: `csrc/fused/`, not `…/sm_90/`).
- `csrc/bindings/dispatch.cpp` — the C++ dispatch (note path; opt_id contract here).
- `csrc/backends/cuda/sm_90/wgmma.cuh` (775 L, the in-kernel ss-wgmma + CuTe engine) · `mma.cuh` (DEAD CUTLASS
  host path, still present) · `csrc/backends/{hip/gfx942,pallas}` (preserved, not exercised this campaign).
- `grokking_optimizers/` — `compile.py` (32,900 L autotuner), `dispatch.py` (2,013 L), `megakernel_codegen.py`,
  `verify_all.py` (DOA). **Ghost (do not exist):** `codegen.py`, `compile_config.py`.
- `tuning/` — `_tp8_build.sh`, `_tp8_run.py`, `_tp8_scratch_pybind.cu` (the committed 8-GPU run wiring),
  `flagship_distributed.py`, `decoder_bench.py` (`--profile`).
- `tests/hw/` — `test_decoder_tc.py` (19/19 gate), `test_l3tc_tail_gate.py` (fp64 + A/A/A), `nvshmem_smoke.py`.

**Live campaign state (root of /workspace):**
- `PROGRESS.md` — the running ledger (most current narrative; 2026-06-25).
- `SuperGrok1.5/RESUME.md` — **the resume guide (current, 2026-06-25).**
- `LEDGER.json` — machine ledger, **frozen at Phase-0 (2026-06-24, HEAD c29ed4e)** — stale on HEAD.
- `PHASE0_CONTEXT{,_v2}.md`, `COMPILE_RECONCILE.md` — Phase-0 reconciliation deliverables.
- `SuperGrok1.5/HANDOFF.md` — **STALE (2026-06-12, prior `h100-audit-maximal` campaign); ignore for current state.**

**Specs (apply-ready design work):** `/workspace/impl_diffs/*.md` (25 files incl. `tp_kernel.md`, `tma_wire.md`,
`mamba_flagship.md`, `datasets_v2.md`, `flagship_dw.md`, `resource_fit_planner.md`, `deadcode_{artifacts,source}.md`).

**Deliverables / runnable workflows:** `/workspace/phase6/` — `roofline_flagship.{png,csv}` (deliverable #1),
`flagship_11opt_ranking.{json,txt}` (OVERFIT placeholder), `tp_datapath_fix_WIP.patch` (**the #1 fix, ungated**),
`*.js` workflows (`roofline_campaign`, `flagship_8gpu_run`, `tp_datapath_fix`, `finish_line`, …).
Flagship runners + logs: `/workspace/phase1/flagship_{train,smoke}.py`, `cute_decoder_validate.py`,
`decoder_phase_baseline.log`.

**Machine-output / duplicate / snapshot (NOT source):**
- `/workspace/SuperGrok1.5/.claude/worktrees/wf_*/` — **dozens of per-workflow worktree clones** (each a full repo
  copy: `csrc/…`, `grokking_optimizers/…`). These are reader/worker scratch, not authoritative — ignore when
  citing.
- `/workspace/{phase0,phase0b,phase0c,phase1..phase6,tune_out,task11_bench_build,flagship_build,cute_*,wt_preTP,
  impl_diffs}` — workflow scratch/build outputs and analysis fan-out.
- `/workspace/SuperGrok1.5/.session_memory/` — standing-rules memory backup (restore per §7.0).
- `_analysis/` — this synthesis + `docs_core.md` reader digest.
- `race_run1.log` (1.1 MB), `verify_all_baseline.log` (degraded pre-fix), `git_reset.log`, `cutlass_init.log`.

---
*End MASTER_STATE.md*
