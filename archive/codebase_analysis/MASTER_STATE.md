# SuperGrok2 — MASTER STATE (authoritative synthesis)

> Synthesis-lead consolidation of the 33-reader analysis + first-hand code/git re-verification.
> Repo: `/workspace/SuperGrok1.5` · branch `claude/custom-optimizer-analysis-HFYhg`
> **HEAD `0668904` [verified `git rev-parse HEAD`]** — *not* `e69df73`; there are **4 commits past the
> closure commit** (`e69df73 → 1956f48 → 392ae82 → 35237b2 → 0668904`). Every doc that says "HEAD e69df73"
> (including the prior MASTER_STATE.md) is stale.
> Synthesis date: 2026-06-25. Evidence tags: **[verified]** = I checked the live tree/git this session;
> **[doc]** = single summary doc; **[doc-corroborated]** = multiple docs agree; **[log]** = measured run
> artifact; **[claimed]** = asserted in RESUME/PROGRESS/LEDGER, not independently confirmable.

---

## 1. EXECUTIVE SUMMARY

**SuperGrok2** is a portable, self-adapting, maximum-performance training stack — a PyTorch-shaped Python
front-end over hand-authored CUDA/C++ — whose defining idea is that **one entire training step runs as a
single persistent fused CUDA megakernel**: one `__global__` launch (one CTA per SM, *not* cooperative) executes
**P0 zero → B0 → P1 fused fwd+bwd → B1 → P2 deterministic dW reduce (through HBM) → B2 → P3 optimizer tail**,
phases separated only by an in-kernel sense-reversing **GridBarrier** (`megakernel_common.cuh:147-255`). The
matrix work is hand-rolled in-kernel **ss-wgmma** (`wgmma.cuh`), *not* CUTLASS collectives and *not* TMA
(only Ampere `cp.async`/LDGSTS is used). The system spans **3 model families** (transformer-decoder, ViT,
Mamba-3) × **11 optimizers** (AdamW, Lion, Grokfast, GrokAdamW, LookSAM, Prodigy, NeuralGrok, Muon,
SuperGrok1.1/1.5/2) × **3 arch backends** (sm_90 Hopper / gfx942 CDNA3 / tpu_v6e), nominally 99 cells. Execution
config is **derived, never GPU-count-switched**: a resource-fit planner picks a 3D→5D mesh (DP×TP×PP +SP +EP)
× ZeRO-3 × memory strategy from *workload × hardware fit*, and the megakernel self-specializes via compile-time
`if constexpr` so the SingleGPU/InHbm path is **byte-identical** to the legacy kernel. Correctness is a HARD
gate: fp64 parity (rel ~1e-4) **AND** A/A/A bit-determinism (`torch.equal` ×3), all rewrites transport-only.

**Single most important "where we left off":** The **sm_90 single-GPU L3-TC path is real, built, and the
prebuilt `_ops.so` is committed in-tree [verified]**; the 11-optimizer flagship decoder ranking ran on real
GPU data (9/11 OK, 2 fits-but-slow) [log]. The **#1 open item is the 8-GPU TP data-path fix**: the patch at
`/workspace/phase6/tp_datapath_fix_WIP.patch` is **NOT applied to the live tree** [verified — live
`model_stage_decoder_tc.cuh:2139` still has the buggy Megatron-shard `qkv_nout = (3*kD)/Par::kTP`], and even
the fix is a **full-width-replicated validation scaffold (a mathematical identity), not real model sharding**
— it proves the in-kernel NVSHMEM all-reduce works on 8 GPUs but yields **zero compute/memory reduction** and
does **not** put a 1.5B model "1/8 per GPU." Genuine weight-sharded TP, the full 33-cell roofline, and any
real-data benchmark remain future work.

---

## 2. ARCHITECTURE — the full stack

### 2.0 Layering (top → bottom)
- **Front-end (PyTorch-shaped, `__version__=3.0.0`, `grokking_optimizers/__init__.py:32`):** 11
  `torch.optim.Optimizer` subclasses that are **pure config/state holders** — every eager `.step()` raises
  `NotImplementedError('L3-TC megakernel only')`. Host-side meta-net training (`meta_step`/`bilevel_step`/
  `sam_step`) is retained for SG11/SG15/SG2. **3 models fixed, 11 optimizers fixed; the dataset layer is the
  one pluggable seam** (`dataset_sources.py`, Layer-A, default-off unless `data_source != 'modular'`).
- **Dispatch/runtime (`grokking_optimizers/dispatch.py`, `csrc/.../dispatch.cpp`):** string-keyed router; HARD
  gate `TORCH_CHECK(gemm_impl=='wgmma')` (`dispatch.cpp:707`) — no eager/scalar fallback survives.
- **Megakernel substrate (`csrc/fused/sm_90/`):** GridBarrier + TaskQueue + PersistentContext + per-model TC
  megakernels + 11 optimizer device-functions.
- **Codegen/autotune brain (`grokking_optimizers/compile.py` 32,900 lines, `megakernel_codegen.py`,
  `megakernel.py`):** feasibility solver + Optuna/Bayesian autotuner + compile cache.
- **Distributed/parallelism (`grokking_optimizers/parallel/`, `distributed.py`):** auto-config + resource
  planner + ZeRO-3 + TP classification + NVSHMEM bootstrap.

### 2.1 The 3 models (two scales)
| Model | Production (grokking science) | Flagship (~1.5B) | Anchor |
|---|---|---|---|
| transformer-decoder | d=128, L=2, vocab=99, kSeq=4; total elems **422,755** | **d=1600, L=48, h=25 (~1.476B; kHeads=25 is non-%8 → the TP bug)** | `dispatch.cpp:454-577`; `grokking_race_v2.py:249` |
| ViT | d=128, 32 tensors | **d=1664, L=48, heads=16, kDhead=104 (non-pow2; ~1.60B, 584 tensors)** | `vit_flagship_layout.cuh:32-41,169-173` |
| Mamba-3 | d=128, 45 tensors; complex exp-trapezoidal selective scan, fp64-oracle ~2e-6 | **d=2048, L=24, state=128, 485 tensors (~1.27–1.53B)** | `mamba3_layout.cuh`, `mamba_flagship_layout.cuh:47` |

Mamba-3 is the real arXiv 2603.15569 block (no conv1d, no SiLU on SSM input, BCNorm + B/C biases, complex
state via RoPE trick). Inside the L3 megakernel the **Mamba mixer is scalar** (scan-dominated); the "TC" Mamba
megakernel calls the same scalar `mb_forward_sample`/`mb_backward_sample` — its wgmma machinery is dormant
Mamba-1 residue (`fused_mamba_megakernel.cuh:548-554`). Mamba TC measured **0.46× vs scalar** [log] — wired
deliberately to expose the honest negative result.

### 2.2 The 11 optimizers (single source of truth)
`csrc/algorithms/<opt>.h` are the canonical per-element step functions; **both** the per-op sm_90 kernels and
the fused L3 megakernel `#include` and call the same `_step()` — drift is impossible by construction, enforced
by `check_math_single_source.py` (3 teeth: structural include, re-inline detection, SHA-256 content manifest).
`OptId` enum `AdamW=0 … SuperGrok2=10` (`opt_components.cuh:53`). `FusedScalars` POD has **26 float fields**
on sm_90 (`dispatch.cpp:204-237`). Per-optimizer state sizing (`dispatch.cpp`): default `3*total+1`;
prodigy `4*total+4`; SG11/15 `4*total+1+129`; SG2 `9*total+1`. Documented historical bug-fixes now in the
canonical headers: Prodigy degree-2 numerator + L1 norm; SG1.1 gate_temperature; SG2 restored grokfast
`lamb_eff*slow_new`; Muon inverted weight-decay; GrokAdamW Q3 floor→ceil.

### 2.3 Persistent L3-TC megakernel substrate
- **GridBarrier** (`megakernel_common.cuh:147-255`): 2 global atomics + sense-reversing generation +
  `__nanosleep` backoff; `sync_reset()` folds the task-counter reset into the last-arriver critical section,
  cutting **4→2** barriers per L3 step.
- **wgmma.cuh** (dual engine): `SG_TUNED_GEMM_ENGINE=0` (default, shipped) = hand-PTX ss-wgmma for
  N∈{8,16,32,64,96,128}; `=1` routes the identical ABI through **CuTe device atoms**
  (`cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS::fma`). ENGINE=1 is present + designed bit-identical but
  **never shipped/validated on hardware** (parity proof is CPU-synthesis docs in `/workspace/cute_plan/`).
- **No TMA**: `cuTensorMapEncodeTiled` is host-only in CuTe 3.6.0; TMA is explicitly phase-2. SASS confirms
  `UTMALDG=0`, `cp.async (LDGSTS)=412`, `GMMA/HGMMA=1415` in the shipped `.so` [log, `scan_machine.md`].
- All 6 production sm_90 TC TUs are present [verified]: `mega_{decoder,vit,mamba}_real_adamw_tc.cu` +
  `*_launcher.cu`. **No `fused_megakernel.cuh` and no 33 codegen `mega_<model>_<opt>.cu` cells exist** — the
  production path is 3 model-specific headers, not the unified codegen substrate (see §5 ARCHIVE GAPS).

### 2.4 Codegen / autotune / compile brain
`compile.py` = two-phase AOT (CPU-only baseline .so) + JIT (GPU Bayesian sweep): XGBoost→sklearn→Ridge cost
model, 5-criterion `BayesianEarlyStopper`, `MultiGPUTimingPool` with per-GPU calibration, JSON
`CompileCache v4` with flock, ccache/sccache masquerade, `ARCH_TABLE` (H100 sm_90a: max_smem 228KB,
peak 989.4 TF/s). `_LIVE_TUNING_DIMS` = {block, vec, unroll, async_depth, cluster_shape, maxrregcount, tile_m,
tile_n, dec_dw_splitk, vit_dw_splitk} (`mb_dw_splitk` removed 2026-06-17). **Inert by default:**
`_MMA_NATIVE_LOADS_WIRED=False` (synth GEMM falls back to scalar triple-loop), `device_profiling` import is a
no-op (module missing — see §5), `enable_synth_codegen/enable_polyhedral=False`.

### 2.5 3D→5D parallelism + ZeRO-3 (the substrate)
`ParConfig<DP,TP,PP,SP,ZeROStage,EP=1>` (`parallel_config.cuh:75`): all fields `static constexpr`;
`kIsSingleGPU=(DP==TP==PP==SP==EP==1)`, `kEmitComm=!kIsSingleGPU` folds **all** NVSHMEM symbols away on the
SingleGPU path [verified]. **`static_assert(SP==1)` at `:99`** — SP axis is *expressible but pinned to 1 this
campaign* [verified]. `using SingleGPU = ParConfig<1,1,1,1,Z0>` (`:111`) is the byte-identical default arg.
In-kernel TP all-reduce: `LoopbackTransport` (single-process honest sim) + `NvshmemTransport` (NVLink
`nvshmem_ptr` direct load/store, TEAM-scoped); the reduction is `tp_allreduce_sum_fixed_order` — **ascending-pe
fp32 fixed order**, the A/A/A structural guarantee (`tp_transport.cuh:244`). 4 TP reduce points
(out_proj fwd, ff2 fwd, ff0-dX bwd, in_proj-dX bwd). ZeRO-2/3 sharded optimizer kernel
(`sharded_optimizer_kernel.cuh:119`) is flat grid-stride, no GridBarrier, reuses `apply_optimizer<Opt>`
verbatim; elementwise-drivable for AdamW/Lion/Grokfast/NeuralGrok, per-tensor opts need the full persistent
megakernel.

### 2.6 Resource planner + datasets
`resource_planner.py:plan_execution()` (R0–R5 escalation ladder) and `auto_config.py:infer_parallel_config()`
— detailed in §2b. Datasets: `dataset_sources.py` is the Layer-A pluggable seam; **only deterministic
synthetic stubs exist** (no FineWeb/ImageNet/GiftEval). LM stub seq hardcoded to 4; ViT npatch=16; mamba
seq=8.

---

## 2b. CONFIG-DERIVATION & ADAPTIVITY — the central thesis (mechanism + how-real)

The thesis: *the system derives its whole execution plan from `workload × hardware fit`, with **no GPU-count
hardcoding** anywhere in the decision path.* This is **substantially real in the Python planner and the
compile-time C++ template machinery, and substantially aspirational in the kernel data-path** (TP doesn't yet
shard a model; SizeLarge body is a stub). Mechanism, layer by layer:

**(A) Resource-fit planner — inputs→outputs, decision procedure** (`resource_planner.py:514-607`).
`plan_execution()` **never branches on GPU count**; it branches on `fit(footprint) ≤ usable_hbm`
(usable ≈ 70.5 GiB/H100). The escalation ladder: **R0** bare in-HBM → **R1** ZeRO-3 → **R1b** raise PP →
**R2** CTA-tile down (`auto_ncta` ladder 132→64→32→16→8→4→2→1, pick largest where staged scratch fits) →
**R3** activation recompute → **R4** layer streaming → **R5** host offload. `infer_mesh()` (`:370-393`):
`TP = largest pow2 divisor of num_gpus bounded by nvlink_width(8) AND d%TP==0`; PP starts at 1, raised only
when `params_per_tp_rank > usable_hbm AND L%PP==0`; DP fills the rest; EP sub-divides DP for MoE. Worked
example: **10B on 1 GPU is plannable** via offload+recompute+streaming (no GPU-count gate); **SG2 on 1 GPU is
structurally unfittable** → planner downgrades to AdamW (a known deep limit). Robustness: **complete and real
as a planner** (all 13 `parallel/`+`distributed` modules are substantive, no stubs [doc-corroborated]).

**(B) Adaptive 3D→5D selection** (`auto_config.py:188-254`). Base `DP×TP×PP`; **+SP (4th)** when the model is
a sequence model (all 3 flagships eligible, **pinned sp=1** this campaign by `static_assert`); **+EP (5th)**
when the model declares `num_experts>1` (`model_num_experts()` ignores the SG2 *optimizer*-expert keys → all 3
flagships are dense, ep=1); **ZeRO-3 is orthogonal**, not an axis. Default policy for 8×H100 is `tp=8,dp=1,pp=1`
— but that is *derived from the fit/nvlink rule*, **not a GPU-count switch** [verified rule]. EP folds away
byte-identically when `kEPComm=false`. Robustness: the *selection logic* is real; **EP is not yet instantiated
in any compiled cell and SP is asserted==1 → the actual operating point today is 3D (DP×TP×PP)**, with 4D/5D
expressible-but-dormant.

**(C) Size-adaptive CTA-tiling selector** (`SizeConfig`, `megakernel_codegen.py:651-671`). Predicate
`_dec_is_large()`: `d≥1024 OR token_tiles<n_sms` → `SizeLarge`; production d=128 → `SizeSmall`. `SizeLarge`
(`parallel_config.cuh:199`) and the `if constexpr(Sz::kCtaTile)` gate exist, **but the CTA-tiled execution body
(§7) is a NO-OP stub** — `SizeLarge` currently degenerates to `SizeSmall` behavior [verified alias present;
finding-confirmed stub]. So the selector *names* a tiling that does not yet change SASS. This is the single
biggest gap between the adaptivity thesis and the kernel reality, and is why the megakernel saturates at
B≈2k and sits at ~0.15–6.5% of roofline (occupancy-bound at `ncta_cap/132` SMs, not compute/BW-bound).

**(D) Memory strategy** (`MemConfig<OffloadOpt,RecomputeActs,StreamLayers,StreamDepth>`, `mem_config.cuh:25`).
`InHbm=<false,false,false,0>` is the byte-identical default; the non-default modes trade compute for capacity
so even 1 GPU can hold a huge model. Layer streaming partially breaks the single-launch invariant (pinned host
weights + a ring of `kStreamDepth` device slots) and is the hardest, least-mature mode. The Mamba-3 flagship
smem redesign (commit `9936308`) implements the streamed-smem path: `kMbStreamSmem` is compile-time TRUE only
when all-layers smem >227KB; it cuts **19.57MB→~193KB** (`kMambaSmemBytes=20,513,956 → kMbStreamSmemBytes
≤227·1024`, `static_assert` at `mamba_flagship_layout.cuh:355` [verified both present]). **BUT** this is
compile-time FALSE for production d=128 (dead on the production path) and there is **no flagship Mamba launch
TU**, so it cannot actually be exercised yet (§4/§5).

**(E) Megakernel if-constexpr self-specialization.** Every `apply_optimizer<Opt>` branch is `if constexpr`,
and every comm/shard/tiling/mem branch is gated on a `ParConfig`/`SizeConfig`/`MemConfig` compile-time bool, so
a degenerate instantiation (`<Opt, SingleGPU, SizeSmall, InHbm>`) folds away **all** comm/shard/tiling code and
is provably byte-identical to the legacy kernel (the PTX-diff gate). This is **real and verified in the
template structure** [verified: `kEmitComm`, `if constexpr` gates present].

**Net honesty assessment.** The *config-derivation brain* (planner + auto-config + compile-time template axes
+ memory ladder) is genuinely implemented and GPU-count-agnostic. The *delivery* of its two headline payoffs —
(i) sharding one big model across N GPUs, (ii) CTA-tiling to fill 132 SMs — is **not yet realized**: TP is a
replicated identity scaffold and SizeLarge is a stub. The adaptivity is "the decision is real, the execution
of the hard cases is scaffolded."

---

## 3. WHAT IS DONE & VALIDATED (with evidence)

1. **sm_90 single-GPU L3-TC megakernel — built & shipped.** Prebuilt `_ops.cpython-311-x86_64-linux-gnu.so`
   is committed in-tree (commit `35237b2` "prebuilt _ops.so + ccache/sccache + ninja") [verified `find`]. This
   **de-stales** the py_root_scripts reader's "build.log shows only Python editable install → would crash"
   blocker — a CUDA `.so` is present now.
2. **33/33 wiring gate green** (`results/h100_grokking_race/wiring_check.json`, 2026-06-24): `total_cells=33,
   converted_l3_tc=33, blocked=0, fraction=1.0`, 132 l3_steps fired, 0 errors [verified file]. Note the
   nuance in §5: this is *observation-based conversion of the 3 per-model TC kernels over OptId*, not 33
   distinct compiled cells.
3. **11-optimizer flagship decoder ranking — real GPU data** (`phase6/flagship_11opt_ranking.json`): d=1600,
   L=48, 1.476B params, B=16, 100 steps, loss 4.585→2.69; `gate_all_finite/descending/11_runnable=true`;
   9/11 OK, 2 fits-but-slow (muon forced ncta=4; SG2 ncta=1 deep limit). Ranking: neuralgrok > grokadamw >
   adamw > grokfast > prodigy > lion > sg11 > sg15 > looksam > {muon, sg2} [log, verified head].
4. **Optimizer math correctness:** 11/0 fp64 parity + 8/11 grokking on the mod-97 d=128 decoder (the 3
   SuperGrok DNFs are research-owned meta-net dynamics) [doc-corroborated, HARDWARE_VALIDATION.md 2026-06-09].
   Canonical headers carry all documented bug-fixes; `check_math_single_source.py` enforces single-source.
5. **ViT TC Fork-B** validated (21/21 gates), wired, dW-output-stationary P2; ViT TC workspace has no
   `nCTA*total` partial term (already Fork-B) [finding].
6. **Structural dead-code cleanup real:** commit `8643cc2` removed 8,089,083 lines / 528 files (95.7% of repo
   text; `_dectc_codegen/` 348MB + `_scan/`); all 3 production layouts byte-identical post-removal
   [verified git log + snapshot_diff].
7. **CuTe 3.6.0 present**; `SG_TUNED_GEMM_ENGINE=1` CuTe-atom path is in source and designed bit-identical
   (not validated on HW).
8. **Partial roofline (10 cells)** measured with nsys (`phase6/roofline_flagship.csv`): decoder/adamw
   3107ms 0.147% peak; vit/adamw 37208ms 0.028% peak — **occupancy-bound, not compute/BW-bound** [verified
   csv]. Best single-GPU decoder throughput 8.46 TF/s (0.85% peak) at B=512/cap=32 [log `dec_sat.log`].

---

## 4. WHAT IS IN-FLIGHT / REMAINING (ordered)

**#1 — TP data-path fix (the live blocker). [verified NOT applied]**
- Patch `/workspace/phase6/tp_datapath_fix_WIP.patch` (23,749 bytes) is a **WIP git diff, not applied** —
  live `model_stage_decoder_tc.cuh:2139` still computes `qkv_nout = Par::kTPComm ? (3*dec::kD)/Par::kTP :
  3*dec::kD` (the buggy Megatron-shard form). Bugs **A (per-rank workspace OOM→null→IMA)** and **B (kHeads=25
  % TP=8 ≠ 0 head-split violation)** are *addressed in the patch*; **C (the +0x1dc30 IMA)** is claimed-resolved
  by the patch but **not confirmed by any compute-sanitizer run**.
- **Crucial nuance (CRUX_TP_DATAPATH.md, lead first-hand):** the fix **abandons real sharding** and computes
  **FULL-WIDTH REPLICATED** on every rank, publishing `(full_result/P)` and ascending-pe summing P identical
  copies (a mathematical identity Σ P·(x/P)=x). It is a **validation scaffold**: it proves the in-kernel
  NVSHMEM NVLink all-reduce + TP plumbing work bit-consistently on 8 GPUs, but yields **zero compute reduction
  and zero per-rank memory reduction**, and the OOM guard converts a wild-pointer IMA into a clean
  `cudaErrorMemoryAllocation` — it does **not** make the flagship fit. **Genuine weight-sharded TP (pre-packed
  per-rank shards + whole-weight grad all-reduce) is explicitly scoped-out, not done.** ETA: apply+gate is
  hours; genuine sharding is a multi-day effort.
- ETA gates: SingleGPU pytest 19/19 byte-identical + 8-GPU run (no IMA, cross-rank loss agrees & descends).

**#2 — Full 33-cell roofline.** Only 10/33 measured (decoder×5 + ViT×5 elementwise opts). 6 decoder staged-opt
+ 6 ViT staged-opt + all 11 Mamba cells missing. Mamba was blocked by smem; redesign landed (commit `9936308`)
but no flagship Mamba launch TU exists, so Mamba roofline is still un-runnable as shipped. ETA: 1–2 GPU-days
once a flagship Mamba TU + nCTA caps are wired.

**#3 — Real-data benchmark.** FineWeb-Edu / ImageNet-1k / GiftEval **not wired** (grep=0 in repo); current
ranking is a fixed-batch synthetic overfit. Spec exists (`impl_diffs/datasets*.md`). ETA: days.

**#4 — Flagship launch TUs missing for Mamba & ViT.** No `mega_*_flagship.cu`; flagship layouts are complete
headers with `static_assert`s but cannot be launched without new TUs. Mamba TP at flagship is *doubly* blocked
(`static_assert(!kTPComm||!kMbStreamSmem)` at `model_stage_mamba_tc.cuh:362-365` + no TU).

**#5 — Per-model TP for ViT/Mamba.** Only the decoder TP data-path exists; ViT/Mamba TP not extended for the
11-opt benchmark.

**#6 — Megakernel performance.** ~0.15–6.5% of 989 TF/s; the lever is **multi-CTA-per-tensor tiling**
(`SizeLarge` §7 body — currently a stub), then TMA + wgmma-accum pipeline (`C7515` serialization warning
unresolved in all SASS), then the 20% grid-barrier idle.

**#7 — SizeLarge §7 CTA-tiled body, EP axis instantiation, layer-streaming maturation** — the adaptivity
payoffs that are scaffolded but not delivered (see §2b).

**#8 — distributed_step** supports only 3 elementwise optimizers (adamw/lion/grokfast); the other 8 are loudly
rejected. ZeRO-3 native shim issues one collective per parameter (un-bucketed).

---

## 5. KNOWN BUGS / DEAD CODE / ARCHIVE GAPS (consolidated)

**Confirmed bugs / gaps:**
- **TP patch ungated** (#1 above) [verified].
- **NVSHMEM not installed** [verified `ls .../nvidia/` → no nvshmem dir]. `NvshmemTransport` and
  `nvshmem_bringup_pybind` require it + `-DSG_HAS_NVSHMEM=1`. Memory note "nvshmem-installed 3.7.0" was
  session-specific; the pip install was deleted on closure. **This contradicts the [claimed] "8-GPU NVSHMEM
  TP all-reduce VALIDATED"** — currently un-rebuildable from the live filesystem.
- **`device_profiling.py` missing** → `compile.py:17652` import fails silently (device-PGO is a no-op) AND
  8 self-test subtests fail `ImportError`, which **breaks `verify_all.py` phase 5c** (regex needs "0 failed").
  The real functions are inlined into `compile.py:32002+`.
- **`flagship_distributed.py:406` wiring gap:** calls `tc_train_step` (tp_size=1 SingleGPU), **not**
  `tc_train_step_tp8` — so its `--nvshmem` flag sets up the heap but the in-kernel TP all-reduce **never
  fires**; its `assert lmax<1e-9` passes trivially (each rank computes identically). The *only* harness that
  actually exercises TP is `tuning/_tp8_run.py` (`tc_train_step_tp8`).
- **gfx942 `FusedScalars` has 15/26 fields** → GrokAdamW/Prodigy/Muon/LookSAM/SG11/15 cells are structurally
  broken on AMD (`dispatch.cpp:377-380`).
- **Mamba flagship cannot launch** at d=2048/L24 as shipped (19.56MB smem vs 227KB cap); `mamba_run.log`
  records all 5 cells failing `cudaErrorInvalidValue` (predates/does not reflect the streamed-smem redesign).
- **SizeLarge body is a NO-OP stub** (§2b-C).
- **`wiring_check.py` references removed APIs** (`_try_fused_step`, `_FUSED_ABI_STALE`) per the phase0 digest →
  AttributeError at `check_cell` — yet `wiring_check.json` (2026-06-24) shows 33/33. The JSON is from a *prior*
  working version of the gate or a different entry path; **reconcile before trusting the gate to re-run**.
- **SG2 won't grok on the L3 path:** in-kernel CSA lightning-indexer drops `idx_UQ` and scales scores
  `/sqrt(rank)` not `/sqrt(d)` → diverges from the eager net for N>64. `mamba×SG2` always raises.
- **NeuralGrok learned amplifier is inert** (frozen at random init) on the kernel path — the only optimizer
  whose headline mechanism does not function [tx_subagents].
- **Version drift:** `compile_config.toml` says `2.0.0`; `pyproject.toml`/`setup.py` say `3.0.0`.
- **split-K default mismatch:** `compile_config.toml` records `dec/vit/mb_dw_splitk=4` but task11 measured
  split-K=2 is 2.1% faster.

**Archive gaps:**
- `csrc/fused/sm_90/fused_megakernel.cuh` does **not exist**; the 33 codegen `mega_<model>_<opt>.cu` cells do
  **not exist** [verified — only 6 `*_real_adamw_tc*.cu` TUs present]. `WIRED_CELLS`/`COMPONENT_CONTRACT.md`
  claims of "×33 generated cells, never hand-edited" are false for sm_90. gfx942 (33 `.hip`) + tpu_v6e
  (33 `.py`) *do* exist.
- `launch_<opt>.cu` shims + `models/*.cu` missing from any git ref (affects `verify_all` structural gates).

**Dead code:**
- `_moe_step` raises `NotImplementedError` (`supergrok2.py:2221`) — MoE compaction kernels unreachable.
- `CompiledSuperGrok2` / `step_full` / `_bilevel_step_cuda` all raise — autograd bilevel only.
- 56-line dead `tc_dump_outproj_operands` in `mega_mamba_real_adamw_tc.cu` (unconditional `TORCH_CHECK(false)`).
- in-file `MambaModel`/`SelectiveSSMLayer` (old Mamba-1) in `grokking_race_v2.py` superseded by import.
- `mma.cuh` (~820 LOC CUTLASS GEMM, zero includers) flagged for removal.

---

## 6. DISCREPANCIES & RECONCILIATION (claimed-state vs code/test reality)

| # | Discrepancy | Rating | Reconciliation |
|---|---|---|---|
| D1 | RESUME/docs say **HEAD `e69df73`**; prior MASTER_STATE.md repeats it | **confirmed** | Actual HEAD `0668904` [verified]; 4 commits past closure (analysis/build-cache commits). Use `0668904`. |
| D2 | "8-GPU in-kernel NVSHMEM TP all-reduce **VALIDATED**" (RESUME §3) | **confirmed misleading** | (a) NVSHMEM **not installed** now [verified]; (b) TP patch **not applied** [verified]; (c) `flagship_distributed.py` never fires the TP path; (d) even applied, the fix is a replicated **identity scaffold**, not real sharding. What was validated = transport-layer smoke via `_tp8_run.py`, not training-correct model parallelism. |
| D3 | "TP data-path A+B **fixed**" | **confirmed overstated** | Fix exists only as an **ungated WIP patch**; live tree still has the buggy `qkv_nout/Par::kTP` [verified]. Bug C unconfirmed (no sanitizer run). |
| D4 | SESSION_CONTEXT: bug A = "per-rank weight-shard offset" | **confirmed mischaracterized** | The fix **removes Megatron sharding entirely** (full-width replicated), it does not fix an offset (CRUX_TP_DATAPATH). |
| D5 | "Build DONE+validated" vs py_root_scripts "only Python editable install → would crash" | **resolved** | Both partly stale: prebuilt `_ops.so` **is** now committed (`35237b2`) [verified]. The build blocker is closed; runtime validation beyond 2026-06-09 H100 session is still 🟡. |
| D6 | "33/33 cells wired" / COMPONENT_CONTRACT "×33 generated cells" | **confirmed false-as-worded** | 0/33 codegen cells on disk; production = 3 per-model TC kernels over OptId routed by `dispatch.cpp`. `wiring_check.json` 33/33 = observed conversions of those 3 kernels, not 33 binaries. |
| D7 | dispatch.py block-comments say mamba×{prodigy,looksam,sg2} **BLOCKED** | **confirmed stale comments** | `_FUSED_L3_REAL`/`_L3_WGMMA_CELLS` frozensets (ground truth) **include** them; A/A/A race fixed (commit `0b57f7e`). Comments are pre-fix text. |
| D8 | "4D/5D parallelism" claim | **confirmed overstated** | `static_assert(SP==1)` + EP not instantiated → operating point is **3D (DP×TP×PP)**; 4D/5D expressible-dormant. |
| D9 | `supergrok2.h:549-578` TODO "fused path needs slow_state+lamb_eff" | **suspected (split entry points)** | csrc_optimizers reader: `opt_stage_supergrok2.cuh` **already** calls `sg2_apply_step(slow_state, lamb_eff)`. BUT csrc_common reader says `opt_components.cuh::apply_optimizer<SuperGrok2>` still uses bare `adamw_step` — **two fused entry points; the TODO may apply to one and not the other.** Needs a targeted read before closing. |
| D10 | README "CUTLASS Sm90 collectives (TMA+WGMMA) for model GEMMs" | **confirmed wrong** | Shipping GEMM is hand-rolled ss-wgmma; CUTLASS/`mma.cuh` is host-only & dead. TMA `UTMALDG=0` in SASS. |
| D11 | README/PERF "~2% is a hardware ceiling" | **confirmed wrong (self-superseded)** | PERF_ANALYSIS.md self-marks SUPERSEDED; dW contiguous-staging gave +2.05×, proving the ceiling was a staging artifact. Real efficiency is occupancy-bound (`ncta_cap/132`). |
| D12 | HANDOFF.md state (roofline 1.15%/1.29%, ~25/33 cells, `.regpressure/.phase2`) | **confirmed stale (different campaign)** | HANDOFF.md is 2026-06-12 branch `claude/h100-audit-maximal`; superseded by 6.48% decoder and 33/33. Ignore for current branch. |
| D13 | LEDGER.json phases 1-6 "pending" | **confirmed stale** | Written 2026-06-24T21:40Z at HEAD `c29ed4e`; PROGRESS.md documents phases 1/4/5/6 milestones done. Never re-synced. |
| D14 | "11-opt ranking overfit placeholder" | **confirmed-but-real-data** | It IS fixed-batch synthetic (overfit), but the measurements are genuine GPU runs, not a placeholder. |
| D15 | `compile_config.toml` cost_model/emitter `enable=False` vs `build()` defaults `True` | **confirmed asymmetry** | TOML wins when a config file loads; direct Python callers get them ON. Benign but real. |
| D16 | SG11 gate: bindings.cpp comment "sigmoid(t·cos)" vs `supergrok11_sm90.cuh` bare clamp | **confirmed (code authoritative)** | The sigmoid-gate switch (task #21) was in-flight at a prior session compaction; canonical header `supergrok11.h:72` does sigmoid, but the per-op sm_90 kernel may still clamp. Verify before trusting SG11 gate semantics. |

---

## 7. EXACT RESUME PLAYBOOK

**Environment / git:**
```bash
cd /workspace/SuperGrok1.5
git rev-parse HEAD            # expect 0668904... (NOT e69df73)
git status --short            # only .pyc + .pytest_cache dirty; source tree clean
find . -name "_ops*.so" -path "*/grokking_optimizers/*"   # prebuilt CUDA .so present
ls /usr/local/lib/python3.11/dist-packages/nvidia/ | grep -i nvshmem  # EMPTY → must reinstall for TP
```
NVSHMEM is gone — for any TP work: `pip install nvidia-nvshmem-cu12` and rebuild the launcher with
`-DSG_HAS_NVSHMEM=1` (3-step manual build in `tuning/_tp8_build.sh`: compile `-rdc=true`, device-link
`-lnvshmem_device`, host link — torch JIT omits the `-dlink` step).

**The #1 fix — apply + gate the TP data-path patch:**
```bash
cd /workspace/SuperGrok1.5
git apply --stat /workspace/phase6/tp_datapath_fix_WIP.patch   # inspect first
git apply        /workspace/phase6/tp_datapath_fix_WIP.patch
# rebuild _ops.so (ccache/sccache warm in .build_cache/); then gate:
#   (a) SingleGPU byte-identity: pytest tests/ -k "tail_gate or l3tc" -q   → 19/19, torch.equal ×3
#   (b) 8-GPU TP path is _tp8_run.py (NOT flagship_distributed.py):
cd /workspace/SuperGrok1.5/tuning && bash _tp8_build.sh && torchrun --nproc_per_node=8 _tp8_run.py
#   gate: 0 IMA under compute-sanitizer, cross-rank loss agrees (dloss<1e-6) AND descends
```
Remember the NVSHMEM ordering constraint: the scratch `.so` must be `dlopen`'d **before** `nvshmem_init`
(`_tp8_run.py:130-142`), else NULL device state → IMA. `_tp8_scratch_pybind.cu:47` uses
`ParConfig<DP=8,TP=8,PP=1,SP=1,Z3>` — verify `DP=8` is intended for a pure-TP8 mesh (suspected should be DP=1).

**After TP gates green:** wire a flagship Mamba launch TU (uses `kMbStreamSmem` streamed path) + nCTA caps →
run the remaining 23 roofline cells → then real-data datasets (`impl_diffs/datasets_v2.md`).

**Single-GPU smoke that already works:** `mega_decoder_real_adamw_tc` fits 80GB only with `ncta_cap=8` +
AdamW + `SG_DEC_BENCH_LAYOUT=1`. The 11-opt flagship ranking is reproducible via the phase6 staged-opt
plumbing (zero-copy aliasing lets SG2 fit at ncta=1).

**Do NOT trust:** `flagship_distributed.py` for TP (fires SingleGPU); `wiring_check.py` until its removed-API
refs are reconciled; `verify_all.py` phase 5c (broken by missing `device_profiling.py`).

---

## 8. MAP OF THE VOLUME

**Live repo `/workspace/SuperGrok1.5/` (HEAD `0668904`, branch `claude/custom-optimizer-analysis-HFYhg`):**
- `csrc/algorithms/<opt>.h` — 11 canonical optimizer math headers (single source of truth) + `SOURCE_OF_TRUTH.md`.
- `csrc/fused/sm_90/` — the megakernel substrate: `wgmma.cuh`, `megakernel_common.cuh` (GridBarrier),
  `parallel_config.cuh`/`mem_config.cuh`, `tp_transport.cuh`/`tp_layer.cuh`, `opt_components.cuh`/
  `opt_stages_precompute.cuh`/`opt_stage_supergrok2.cuh`, `model_stage_{decoder,vit,mamba}_tc.cuh`,
  `fused_{decoder,vit,mamba}_megakernel.cuh`, 6 `mega_*_real_adamw_tc{,_launcher}.cu`, `*_flagship_layout.cuh`,
  `sharded_optimizer_kernel.cuh`, `nvshmem_bringup_pybind.cpp`, `fused_dispatch_table.inc`.
  **Missing (archive gap):** `fused_megakernel.cuh`, 33 codegen cells, `launch_<opt>.cu`, `models/*.cu`.
- `csrc/common,bindings` — `_ops.so` pybind (`dispatch.cpp`, `bindings.cpp`), GridBarrier/TaskQueue/platform.
- `csrc/backends/hip/gfx942/` + `grokking_optimizers/kernels/gfx942/` — AMD twin (structurally complete,
  HW-gated). `csrc/backends/pallas/` + `csrc/fused/tpu_v6e/` — TPU twin (trace-only). `third_party/cutlass/` 3.6.0.
- `grokking_optimizers/` — front-end: 11 optimizer classes, `compile.py` (32.9k lines), `megakernel_codegen.py`,
  `megakernel.py`, `megakernel_engine.py`, `dispatch.py`, `distributed.py`, `parallel/` (auto_config,
  resource_planner, zero3, shard_map, pipeline), `host_bringup.py`, `nvshmem_bringup_ext.py`,
  `dataset_sources.py`, `_tuned_inject.py`, `tune_hook.py`, `lowprec.py`, `profile*.py`, `verify_all.py`,
  prebuilt **`_ops.cpython-311-x86_64-linux-gnu.so`**.
- `tuning/` — `flagship_distributed.py` (⚠ SingleGPU dispatch), **`_tp8_run.py`/`_tp8_build.sh`/
  `_tp8_scratch_pybind.cu` (the real TP harness)**, `*_bench.py`, `roofline.py`, `tune_optimizers.py`.
- `tests/` — tail gates, multistep parity, `tests/hw/`. `scripts/`, `examples/`, `docs/reviews/`.
- Root docs (mixed freshness): `CODEBASE_EXPLAINED.md` (best, 2026-06-17), `HARDWARE_VALIDATION.md`,
  `OPTIMIZATION_LEDGER.md`, `PHASE1_CAMPAIGN.md`, `RESUME.md`/`SESSION_STATE.md` (HEAD claim stale);
  **stale/different-campaign:** `HANDOFF.md`, `DESIGN-TC-PIPELINE.md`.
- `results/h100_grokking_race/wiring_check.json` — MUST STAY (referenced by README/tooling).

**Outside the repo (`/workspace/`):**
- `phase6/` — deliverables: **`tp_datapath_fix_WIP.patch` (the #1 item, NOT applied)**,
  `flagship_11opt_ranking.json`, `roofline_flagship.csv` (10 cells), bench logs (`mamba_run.log`,
  `dec_sat.log`), `staged_opt_plumbing/`, ~18 orchestration `.js` files.
- `PROGRESS.md` (running ledger), `LEDGER.json` (stale snapshot), `PHASE0_CONTEXT.md`, `COMPILE_RECONCILE.md`,
  `impl_diffs/` (16 apply-ready specs, **none applied**), `cute_plan/` (CuTe/TMA synthesis, not implemented),
  `.session_memory/` (active 2026-06-25 directives), `.audit_notes/.regpressure/.phase2/` (2026-06-12 campaign,
  stale), `wt_preTP/` (pre-teleport snapshot; holds the removed 348MB `_dectc_codegen/` machine output).
- `_analysis/` — the 33 reader digests + this `MASTER_STATE.md` + crux deep-dives
  (`CRUX_TP_DATAPATH.md`, `CRUX_CONFIG_DERIVATION.md`).

**Machine-output / duplicate / snapshot (don't mistake for source):** `wt_preTP/_dectc_codegen/` (348MB PTX/SASS),
`wt_preTP/_scan/`, `nvcc_baseline_build/`, `build/compiled/`, all `*.pyc`/`.pytest_cache`, SASS census dumps
(`phase1/ops_sass_census.txt`), Optuna journal logs. The `.so` files under `.claude/worktrees/*` are
per-worktree build copies, not the canonical artifact.

---

*Synthesis complete. Headline: HEAD is `0668904` (not e69df73); sm_90 single-GPU L3-TC is built & shipped with
a committed `_ops.so`; the 8-GPU TP fix is an unapplied WIP patch that is itself only a replicated validation
scaffold (no real sharding), NVSHMEM is uninstalled, and the "validated on 8 GPUs" claim is not currently
reproducible. The config-derivation brain is real; its two hardest payoffs (model sharding, CTA-tiling) are
scaffolded, not delivered.*
