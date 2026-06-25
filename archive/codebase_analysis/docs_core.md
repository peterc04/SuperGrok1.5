# docs_core.md — Canonical architecture digest from the repo-root core docs

Slice: the repo-root architectural/mental-model docs of SuperGrok2 (repo `SuperGrok1.5`).
Read IN FULL: `README.md` (730L), `CODEBASE_EXPLAINED.md` (501L / 115K — the exhaustive reference),
`DESIGN-TC-PIPELINE.md` (680L), `CODEBASE_AUDIT.md` (40L), `CODEBASE_AUDIT_FINDINGS.md` (94L),
`SUPEROPTIMIZER_L2_PLAN.md` (211L), `SUPEROPTIMIZER_SCOPING.md` (370L), `SESSION_STATE.md` (124L),
`ENV_SNAPSHOT.txt` (180L). All file:line cites below are from these docs (which themselves cite code);
I verified a few load-bearing facts against the live tree (git, file existence) — see "Ground-truth checks".

---

## 0. TL;DR — the canonical mental model

SuperGrok2 trains **3 model families** (transformer-decoder, ViT, Mamba-3) under **11 optimizers**, each
compiled as **one persistent fused CUDA megakernel** on H100 sm_90 (bf16 wgmma + fp32 accum). The
defining decision: **one `__global__` launch runs the entire training step** — P0 zero → P1 fwd+bwd →
P2 deterministic grad reduce → P3 optimizer tail — with phases separated only by an in-kernel,
hand-built **sense-reversing GridBarrier** (no per-step relaunch, no cooperative launch, optimizer state
HBM/L2-resident). Cost: **one CTA per SM** (the barrier needs all CTAs co-resident).
`compile.py` is the **superoptimizer/autotuner**. Correctness is a **HARD gate**: fp64 parity (rel 1e-4;
SAM 2.5e-2/3e-2) **AND** A/A/A bit-determinism. Legal rewrites are **transport-only** (preserve
ascending-k fp32 reduction order).

## 1. The 44 → 99 architecture (README §1, README:421-489)

- **44 components**, one canonical home each:
  - optimizer × arch = 11 × 3 = **33**
  - model × arch = 3 × 3 = **9**
  - dispatch + compile = **2**
- **11 optimizers:** AdamW, Lion, Grokfast, GrokAdamW, LookSAM, Prodigy, NeuralGrok, Muon, SuperGrok1.1,
  SuperGrok1.5, SuperGrok2. **3 models:** Transformer-Decoder, ViT, Mamba-3. **3 archs:** sm_90 (Hopper),
  gfx942 (CDNA3/MI300X), tpu_v6e.
- Dispatch/compile composes any (optimizer × model × arch) into **99 fused pipelines** (33 per arch).
  Each cell is a real composition of canonical component device-functions (anti-false-positive sweep = 0).
- **opt_id integer contract** (hand-maintained, no auto drift-guard — flagged gap): `adamw=0, lion=1,
  grokfast=2, grokadamw=3, looksam=4, prodigy=5, neuralgrok=6, muon=7, supergrok11=8, supergrok15=9,
  supergrok2=10` (`dispatch.cpp:591-616` ↔ `OptId` in `opt_components.cuh:53-57`).

### Canonical directory layout (README:444-474)
- `csrc/algorithms/<opt>.h` — CANONICAL per-element optimizer math (one def/opt). SG2 bilevel adjoint in
  `supergrok2_bilevel_adjoint.h`. `SOURCE_OF_TRUTH.md` is the contract.
- `csrc/fused/sm_90/` — `opt_components.cuh` (apply_optimizer<OptId>→algorithms), `model_stage_*`,
  `fused_*_megakernel.cuh` (the composition seam), `mega_<model>_<opt>.{cu}` (cells).
- `csrc/backends/{cuda/sm_90,hip/gfx942,pallas}` — entry shims / AMDGCN device / TPU Pallas math.
- gfx942 mirror + tpu_v6e (24-line `.py` cells). The C++ fused dispatch table is generator-emitted
  (`csrc/fused/fused_wired_cells.inc`) so it cannot hand-sync-drift.

### Single-source guarantee (README:476-488)
`scripts/check_math_single_source.py` (wired into `--self-test` as `math_drift_guard`) fails the build on
3 triggers: (1) a consumer stops `#include`-ing the canonical header; (2) re-inline of Adam
moment-update/apply locally (Phase-7 re-inline detection); (3) canonical math changes without
`--update-manifest` (content-hash manifest `scripts/optimizer_math_manifest.json`).

## 2. Per-arch story (README §2, README:493-512)

- **sm_90:** inlined PTX (`rsqrt.approx`, `ex2.approx`, `fma.rn`, `redux.sync`); README claims **CUTLASS
  Sm90 collectives (TMA+WGMMA)** + a TF32 (`tfloat32_t`) path for FP32. **NOTE/DISCREPANCY:** the
  shipping persistent-megakernel GEMM is **hand-rolled in-kernel ss-wgmma** (`wgmma.cuh`) that
  deliberately does NOT use CUTLASS; CUTLASS (`mma.cuh`) is host-launched and used only for the L1 Muon-NS
  / SG2 `dt_proj` paths and is "explicitly REJECTED for the persistent-megakernel path"
  (CODEBASE_EXPLAINED §4, `wgmma.cuh:14-18`). Race precision is **bf16**, not TF32 (DESIGN §5, Appendix B).
- **gfx942:** hand-written AMDGCN (`__builtin_amdgcn_mfma_*` bf16 16×16, DPP wave-64 reductions, FNUZ
  FP8). LIVE on `#if __HIPCC__`; ATen/rocBLAS = CPU fallback. SG2 adjoint + MoE compaction are real device code.
- **tpu_v6e:** Pallas (`pl.pallas_call`+`BlockSpec`) composed into one `jax.jit` program per cell;
  `lax.associative_scan` for Mamba. 256-wide MXU tile.

## 3. The fused-megakernel substrate (the heart)

### Persistent kernel + GridBarrier (CODEBASE_EXPLAINED §1-§2)
- Entry points: `fused_decoder_megakernel_tc<Opt>` (`fused_decoder_megakernel.cuh:672-678`, launched
  :1511-1569 via plain `<<<grid,block,dyn_smem,stream>>>` — NOT cooperative), ViT
  `fused_vit_megakernel_tc` (`fused_vit_megakernel.cuh:1178`), Mamba `fused_mamba_megakernel_tc<Opt>`
  (`fused_mamba_megakernel.cuh:527-532`).
- **gridDim.x == #SMs, one persistent CTA/SM**; launcher reads `cudaDevAttrMultiProcessorCount`
  (:1547-1549). Hard occupancy gate ≥1 or `cudaErrorLaunchOutOfResources` (refuse, don't hang;
  :1540-1545). SM pin via `%smid` (`megakernel_common.cuh:65-73`).
- Host seams: `mega_<model>_real_adamw_tc_launcher.cu` expose ONE non-template host launcher with a
  pointers/ints + `FusedScalars` POD ABI (so `dispatch.cpp` extern-declares without header types). The
  launcher exists because the Fork-B TC cell driver owns its own `PYBIND11_MODULE` and is dropped from
  `_ops` by setup.py's `_collect()` (PyInit collision). Compiled `-DSG_TUNED_GEMM_IMPL=1` for the wgmma branch.
- **GridBarrier** (`megakernel_common.cuh:147-255`): two global atomics (`g_arrived`, `g_generation`) +
  sense-reversing generation; release/acquire `__threadfence()` pair (§1.14, one before publish/one after
  wait — loose extra fence is a >10% perf + correctness footgun); `__nanosleep` exp backoff cap 1024ns;
  `sync_reset` folds task-queue zeroing into the last-arriver critical section → collapses 4 barriers → 2.
- **TaskQueue** (work-stealing): single global atomic `g_next_task`; idle CTA steals next tensor. Order of
  the fp32 summation is never affected (determinism by construction).

### 5-phase / 4-barrier layout (decoder = reference; CODEBASE_EXPLAINED §3)
P0 zero + bf16 weight pre-stage → **B0** → P1 token-tile fwd+bwd GEMMs (barrier-free within tile, loss in
fp32 per-CTA) → **B1** → P2 dW GEMM + grad assembly + LN-vec reduce + **fp64 loss reduce** by CTA0 →
**B2 (sync_reset)** → optional staged P2.x phases → P3 optimizer tail (work-steal, `apply_optimizer<Opt>`,
no trailing barrier). Staged insertions, each `if constexpr`-gated so every other cell is byte-identical:
- P2.4 SAM 2nd backward (looksam/sg11/sg15/sg2) — full 2nd in-kernel fwd+bwd, side channel sam_dir / sharpness.
- P2.5 grad-norm clip (grokadamw/neuralgrok).
- P2.6 Prodigy d-reduce (FIXED partition, not work-steal → determinism).
- P2.7 Muon Newton-Schulz (grid-cooperative, 5 iters over 2D weights).
- P2.45/P3-SG2 (SG11/15 meta mu precompute; full SG2 CSA/HCA/PEER/GRU meta-net).

### wgmma substrate (`wgmma.cuh`, CODEBASE_EXPLAINED §4)
Hand-rolled ss-wgmma `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`. 64-bit smem descriptor
bit-packing (16B-align mandatory); `SwizzleMode` NONE=0/128B=1/64B=2/32B=3 (NOT size-ascending — footgun);
ships `kSwizzleNone`. `WgmmaAccum<N>` holds N/2 regs; one shared fragment decode for store+gate.
`wgmma_mainloop_kchain` issues k_steps in **fixed ascending-k** (first ScaleD=0, rest ScaleD=1); descriptor
gen is a callback so the same mainloop serves unpipelined + pipelined paths bit-identically.

### cp.async ring (`tile_pipeline.cuh` + decoder PIPE/STAGES macros, CODEBASE_EXPLAINED §5)
Uses **cp.async (.cg.16), not TMA** (TMA = phase-2). `TilePipeline<N,Depth>` producer/consumer ring,
each mbarrier expects full 128-warpgroup arrivals. Decoder selector:
- `SG_TUNED_DEC_GEMM_STAGES` default 2.
- `SG_TUNED_DEC_FWD_PIPE` **baked at 1** (0=shipped S=2 ring; 1=deepened all-threads ring; 2=full
  producer/consumer mbarrier engine). **Tournament verdict: entry-a deeper ring WON +1.49×; entry-b
  PIPE=2 LOST (857 vs 618 ms).**
- `SG_TUNED_DEC_FWD_STAGES` 2..4 **baked at 4**.
- Deeper rings (PIPE&&STAGES>2) route to **dynamic smem** (`SG_DEC_TC_DYNAMIC_SMEM`) via
  `cudaFuncAttributeMaxDynamicSharedMemorySize`; static cap 48KB, dynamic budget ~228KB/SM.

### Per-model specifics
- **Decoder** (§6): causal attention; embedding **counting-sort owner-scan** (deterministic CSR + perm,
  ascending-t); dW split-K (`SG_TUNED_DEC_DW_SPLITK`, default **G=1** now — single-CTA dW 2.05× faster
  than G=2-scalar); dW staging (`SG_TUNED_DEC_DW_STAGE=1`) — dW is **staging-bound (~97% staging / ~3%
  wgmma)**, so the lever is a contiguous K-major pre-transpose (transport-only).
- **ViT** (§7): full **bidirectional** attention; patch-proj Linear(49→128) + CLS token at pos 0 + pos[17];
  head on pos 0; `kSeq=17` → `kTileM=1088`. **B1 sub-tile lever `SG_TUNED_VIT_P1_SUBTILE_S` baked at 8**
  (#1 ViT lever, 4.02× — at B=1024 only 16 P1 tiles so 116/132 CTAs idle; sub-tiling by S whole samples
  grows n_tiles). Default 64 = byte-identical OFF.
- **Mamba3** (§8): genuine Mamba-3 SISO (arXiv 2603.15569, ICLR 2026), exponential-trapezoidal selective
  scan; **batch-parallel scalar scan** in registers (seq=8 register exploit, no checkpointing; c-loop
  intentionally NOT unrolled or >255 regs). `MambaSampleSmem` dynamic ~215KB at d=128 → **cannot place
  past ~d≈142** (smem-bound, 1 CTA/SM); d=2048 impossible on the megakernel. The wgmma-projection dW
  machinery is **bypassed** — production Mamba TC = scalar scan in the persistent substrate.

## 4. The optimizer tail (CODEBASE_EXPLAINED "optimizer files")

The optimizer **is not a separate kernel** — it is P3 of the same megakernel. State is one flat
caller-owned buffer `[m | v | extra]` over `3*total` floats + loss slot; `extra` is overloaded per
optimizer (grokfast/grokadamw EMA, OR Prodigy `s`, OR LookSAM `sam_dir`, OR SG11/15 `mu`). Staged opts
extend past loss slot; host sizes per-cell in `dispatch.cpp` (`min_state`: 4*total+4 Prodigy,
4*total+1+kSgPhiPack SG11/15, else 3*total+1). `rebase_state<Opt>` adds tensor flat offset to per-element
pointers only. `apply_optimizer<Opt>` is an `if constexpr` ladder; every branch calls canonical
`csrc/algorithms/<opt>.h`. No-silent-fallback: launcher `switch` `default: return cudaErrorInvalidValue` →
thrown runtime_error. SG2 has a **dedicated launcher + dispatch entry** (`ops.sg2_fused_step`,
`dispatch.py:1973-1988`); its 26-pointer meta-net bundle staged to HBM.

### Perf reality (CODEBASE_EXPLAINED): fusion maxed, tail NOT
Fusion thesis fully realized (one launch, 0 intermediate launches; only host bubble = one `.item()` D2H).
But P3 tail is **~5.9% of the decoder step (~38ms/618ms)** and **un-autotuned** (autotune produced 0
winners). `perf_maxed:yes` only for adamw/lion/grokfast (and flagged "thin").

### Correctness reality: 3 green, 8 gate-COVERAGE caveats (NOT confirmed drift)
**green:** adamw, lion, grokfast (33/33×3-seed pytest at d=128). 8 caveats are MISSING/toy-scale gates,
not active math drift (drift audit = 0/11 active). grokadamw/prodigy multistep gate missing; muon/neuralgrok
blocked by **BUG-04** (mamba staged-opt scratch un-gated → ~199 GiB OOM at d≥1024, a VRAM sizing bug not a
missing optimizer); looksam looser SAM tol; sg11 warmup gate CLI-only not CI; sg15 no warmup gate; sg2 CSA
oracle co-wrong (HIGH). The **51098e0 SG11 fix** = cos(grad,MU)→cos(grad,MOMENTUM) in the staged gate.

## 5. compile.py — the superoptimizer/autotuner (CODEBASE_EXPLAINED §"compile.py")

32,900-line per-(opt,model,arch) autotuner. Two-phase AOT (CPU, `build_aot`, cache JSON
CACHE_VERSION==4) / JIT (GPU, `build_jit`). Optuna **TPE Bayesian** (default), SQLite-persisted +
resumable, no fixed budget (`BayesianEarlyStopper`); top-K ±2 local refine. Learned **CostModel**
(XGBoost→sklearn-GBR→numpy-ridge) **default-ON**, prunes before building, cold-floor 100 timings,
multi-fidelity. Winners baked per-TU via `_tuned_inject.py` (nested per-model JSON
`{arch:{model:{optimizer:combo}}}`; SAFE `MACROS` table = mega_block/tile_m/tile_n/dec_dw_splitk/
vit_dw_splitk/prod_regs/cons_regs + maxrregcount). **No-JSON build is byte-identical.**

**fp64-gate-in-loop** (task #28, landed `af9b720`): `_default_correctness_hook` builds a `pick_winner`
hook that re-runs `run_cell_gate` (fp64 + A/A/A) and demotes failing top-K candidates, fail-closed; if
none clear it the winner is left UNSET. **RG4-scale caveat (real discrepancy):** the hook's comments
claim "PRODUCTION scale (d=2048)" but `_build_cell` uses `grokking_race_v2.DEFAULT_CONFIG` (d=128, L=2) —
the toy grokking scale; non-default-shape winners must be re-gated at d=2048 manually.

**Dead/ghost-module caveats (verified true in tree):** `grokking_optimizers/codegen.py` does NOT exist
(emitter ghost import → macros-only fallback, `_emitted_sources` never populated);
`grokking_optimizers/compile_config.py` does NOT exist (in-file load_config used instead);
`SG_TUNED_*` GEMM-stage/swizzle/tma/wgmma_shape dims are **dead/auto-pinned** (no kernel `#ifndef` reads them).

## 6. SUPEROPTIMIZER_SCOPING.md (#25) — measured NO-GO

Recommendation: **No-go on a full general GPU-kernel superoptimizer; continue #24 + hand-author the named
levers.** Break-even needs kernel/arch breadth; we have 3 fixed models × 11 opts × ONE arch (sm_90a), a
small enumerated lever set, and already own ~80% of a practical superoptimizer in compile.py. The novel
capability (auto-discovery of structural rewrites of a fused training megakernel) is exactly what NO
surveyed IR can express. Key binding constraints (also in L2_PLAN §1):
- **C1 transport-only legality** — ascending-k fp32 reduction order is non-associative; any reassociation
  fails the A/A/A gate.
- **C2 no IR expresses the persistent fused megakernel** — CuTe/Triton/MLIR-Linalg/polyhedral are all
  collective/single-op shaped; a host-launched collective can't run device-side inside the persistent grid.
- STOKE on the dW inner loop = no-go (win is above the instruction window; NVIDIA has no SASS assembler).
- Recommended cheap probe: re-express ONE decoder fwd in_proj GEMM in CuTe-DSL and race it (~3-5 days).

## 7. SUPEROPTIMIZER_L2_PLAN.md (#27) — owner overrode the recommendation → GO

Owner chose **(b): make Level-2 REAL, correctness-gated** (supersedes the *recommendation* not the
*constraints*). Re-spec: do NOT re-synthesize the GEMM (the hand engine is near-maximal); **max the shared
primitive first, then synthesize the FUSION** (the fp32 epilogue passes). Status map (RESOLVED): all 5
generative back-ends structurally wired but **non-functional on the real path** — codegen emitter module
missing, synth GEMM is a stub (`_MMA_NATIVE_LOADS_WIRED=False` → scalar triple-loop), polyhedral is a toy
+ unreachable, CUTLASS/CK are host-launch-only and never origin-stamped; **no trial has ever carried a
generated origin.** What IS real: elementwise/reduce/scan lowerers emit compilable CUDA; the winner
source-swap + #16 strict-validation gate would correctly REJECT today's toy outputs (no fake-green). Phases:
A1 max the primitive, A2 fp64-oracle-in-loop (LANDED af9b720), B CuTe calibration probe, C0 emitter unblock
(re-target to a device-inlinable `__device__` fragment), C fusion synthesis, D polyhedral on fp32 epilogues.

## 8. DESIGN-TC-PIPELINE.md — **STALE earlier design doc** (important caveat)

Scope line: **branch `claude/h100-audit-maximal`** (the EARLIER campaign, same as stale HANDOFF.md). It is
an implementation CONTRACT written when there was **"No in-kernel wgmma anywhere"** — "Every model-stage
GEMM is a scalar fp32 owner-computes triple loop" (DESIGN:32). It plans the wgmma/TC rewrite (Fork B
dW-output-stationary, bf16 policy, R1/R2 sequencing, 7 new `SG_TUNED_*` dims) that has SINCE BEEN
IMPLEMENTED (wgmma.cuh now exists, verified). Treat DESIGN-TC-PIPELINE.md as the historical design rationale
for the now-shipped TC engine, NOT current state. Useful canonical facts it pins (code-verified at the
time): decoder train batch **B=int(97²·0.5)=4704** (the `4191` in comments is a stale val-split number);
decoder **vocab=99** (`SG_DEC_VOCAB=99`); mamba 28 tensors/259,425 elems; vit 32 tensors/418,017 elems.
Its own docstring "5 phases/4 barriers" vs call sites "4 phases/3 barriers" is noted as stale (DESIGN:29).
Appendix A epilogue-fusion verdict: **5/11 fully fusable** (adamw/lion/grokfast/grokadamw/neuralgrok);
prodigy/sg11/sg15 need one reduce phase; muon/sg2 not fusable.

## 9. Roofline / perf reality (CODEBASE_EXPLAINED §4)

Peaks (H100 SXM): bf16 TC **989.4 TF/s**, HBM3 **3.35 TB/s**, ridge ≈295 FLOP/B. Measured (derived from
wall-time, NOT formally scored; FLOP numerator counts dense GEMMs only):
| model | config | ms/step | % of 989 peak | bound |
|---|---|---|---|---|
| decoder | d=2048, B=16384 | **618** | **6.48%** | latency/phase-serialization |
| vit | d=2048, B=1024 | ~1434 (was 5759) | ~0.74% | was B1 load-imbalance (fixed S=8) |
| mamba3 | d=128 (smem-capped ~d142) | 221.6 | 0.0303% | scan-dominated |

618ms decoder phase split: P1_fwd 178.5 (27.6%), P1_bwd 176.7 (27.3%), B1_barrier 108.4 (16.8%),
P2_dW_GEMM 106.7 (16.5%), P3_opt_tail 38.0 (5.9%), B2 20.1, P2_grad_asm 16.8, B0 0.9. GEMM phases ≈55%
(74.9% w/ dW); grid barriers ≈20% pure serialization idle. **6.48% is mostly fixable inefficiency**
(barrier serialization, sub-cuBLAS hand engine, un-overlapped fp32 epilogues, 1-CTA/SM occupancy), NOT a
non-GEMM physics floor; ideal TC time = 40.0ms. **Levers WON:** decoder PIPE=1/STAGES=4 +1.49×; dW
contiguous K-major staging +2.05×; ViT B1 sub-tile S=8 4.02×. **Levers LOST:** PIPE=2 producer/consumer
−11.6% (dW is staging-bound not drain-bound); GEMM interleave IL2→4 fastest but A/A/A FAILED on ragged
atoms (bench missed, gate caught); ViT dW-staging port +4.5% & gate-RED → reverted (decoder win does NOT
generalize).

## 10. The HARD gate + ratchet (CODEBASE_EXPLAINED "correctness methodology")

(1) **fp64 parity** vs pure-fp64 oracle (`test_l3tc_tail_gate.py`, `run_cell_gate`): params max-rel 1e-4
(muon 2e-3 for NS), **state `[m|v|extra]` 1e-4 = the decisive check** (params blind to per-layer β1 /
state init at step 1), SAM sharpness 3e-2 / sam_dir 2.5e-2. (2) **A/A/A bit-determinism** via `torch.equal`
across 3 runs (loss/grad/params; SAM cells add mu/sharpness). ANDed = HARD gate; autotuner wired to same gate.
**Ratchet:** apply→build→fp64+A/A/A→3-seed timing → KEEP iff faster on 3+ seeds AND parity-clean else
REVERT (parity is a gate not a tiebreaker; neutral reverts; every revert root-caused). Risky NEEDS-PARITY
knobs `#if`-erase to byte-identical OFF defaults.
**Fake-green catalogue:** macro-drift table-vs-header guard; SG11 co-wrong cos oracle; autotune-hook
d=128-vs-d=2048 scale lie; SG2 mirrored-oracle (non-voting probes); `_emitted_sources` keying bug (fixed);
IL=4/atomicAdd determinism trap.

## 11. CODEBASE_AUDIT(.md/_FINDINGS.md) — the recurring audit + 2026-06-17 findings

Process: after every landed step, full-tree line-by-line Opus swarm (range agents + 3 cross-cutting:
dead-file reachability, dead-code, consistency); findings dedup into `CODEBASE_AUDIT_FINDINGS.md`;
conservative deletion (prove-dead, flag-and-keep).
2026-06-17 8-agent findings headline: **shipping single-GPU path is sound** (33-cell dispatch consistent
Python≡C++≡opt-id, fp64+A/A/A genuinely anchored, dormant Phase-2/HIP/Pallas correct+isolated). Defects
cluster in 3 buckets NONE on the numeric path: (A) cleanup-#10 regression in dev/verification harnesses +
`has_kernels()` returns False on a healthy `_ops.so` (probes 4 removed symbols, `dispatch.py:~542`); (B)
dead weight (820-LoC unused CUTLASS `mma.cuh`, committed build artifacts, ~750MB never-read scratch); (C)
doc drift. P0 items: verify_all/profilers DOA (target removed TUs); `--grad-hooks` eager path raises;
`FusedOptState st;` uninit (benign for AdamW); `PIPE==2` half-wired null-deref if built; zero3 non-contig
broadcast. Verification struck 3 rows as FALSE (mamba uninit benign, zero3, PIPE==2 actually compiles).
Flag-and-KEEP: HIP+Pallas, Phase-2 scaffolding, fp64 oracles, `.STOP_TUNING` sentinel.

## 12. SESSION_STATE.md — live resume state (planning session, ~2026-06-17)

Session ENDED cleanly (planning/strategy). NEW directives: (1) **optimizers must be MAXED not just fused**
(P3 tail ~5.9%, un-autotuned, 8/11 gate caveats); (2) **high % roofline at EVERY scale** (decoder 6.48%
@d2048 largely fixable, target 30-50%+; hard floor only at d=128 tiny matmuls); (3) owner bringing a full
engineering plan, hybrid labor split. Shipped+validated this session: ViT B1 S=8 bake (b0d41f8),
macro-table drift fix (319b96d), front-load fix + phase-1 audit (64feb14), decoder PIPE=1/STAGES=4 bake,
SG11 correctness fix (51098e0), build-cache persistence. Binaries on volume: `_ops*.so` (33M), `build/`
(90M). `.STOP_TUNING` restored (autotuner OFF by default). Self-contained venv
`/workspace/venv_selfcontained` (4.9G). Open queue: confirm-build ViT S=8, optimizer-max, gate-hardening,
BUG-04, C0 fragment emitter, perf levers, fast-triage. Constraints: **commits LOCAL-ONLY, never push**;
fp64+A/A/A HARD gate; Ultracode ON.

## 13. ENV_SNAPSHOT.txt — the 180-package pin

py 3.11.10, **torch 2.4.1+cu124 / torchaudio 2.4.1+cu124 / torchvision 0.19.1+cu124**, CUDA 12.4 toolkit,
nvidia-*-cu12 12.4.x, nccl 2.20.5, triton 3.0.0, jax 0.4.38/jaxlib 0.4.38, optuna 4.9.0, xgboost 2.1.4,
scikit-learn 1.9.0, prodigyopt 1.1.2, pynvml 11.5.3, pytest 9.1.0, ruff 0.15.17, ninja 1.13.0, numpy 1.26.3.
Editable install pin: `-e git+https://github.com/peterc04/SuperGrok1.5@4af83c3527431aa6f8c8460fb03bf2ba43823290#egg=grokking_optimizers`
(NOTE: a github remote `peterc04/SuperGrok1.5` exists in the pin despite the "LOCAL-ONLY never push" rule;
the pinned SHA 4af83c3 is NOT the current HEAD e69df73).

---

## Ground-truth checks I ran against the live tree
- `git branch --show-current` = **claude/custom-optimizer-analysis-HFYhg**; HEAD = **e69df73** "closure:
  resume docs + session memory backup + ... 8-GPU run wiring" — MATCHES the CLAIMED state.
- Recent commits confirm the claimed campaign: `03bd3f0 finish: nvshmem_pybind`, `9936308 mamba flagship
  smem redesign (layer-stream + scratch-to-HBM)`, `8643cc2 cleanup: remove 8.09M lines of provably-dead
  committed artifacts`, `5e084ca vit/mamba TP track`.
- `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` EXISTS; `NVSHMEM`/`nvshmem` + TP files (`tp_layer.cuh`,
  `tp_transport.cuh`, `parallel_config.cuh`) present in tree.
- `grokking_optimizers/codegen.py` and `grokking_optimizers/compile_config.py` do **NOT** exist (confirms
  the ghost-module caveats in CODEBASE_EXPLAINED §"build env").
- `csrc/backends/cuda/sm_90/wgmma.cuh` **EXISTS** (confirms DESIGN-TC-PIPELINE.md's "no in-kernel wgmma" is stale).

## 14. Config-Derivation Mechanism — Mechanism-Level Detail (KEY GOAL)

This is the central design thesis: "no hardcoded rule, codebase ROBUSTLY DERIVES execution config from workload × hardware fit." Here is what the core docs reveal at mechanism level.

### 14.1 Grid/occupancy derivation (REAL, runtime)
- Grid = `cudaDevAttrMultiProcessorCount` → exact SM count at launch time (`fused_decoder_megakernel.cuh:1547-1549`)
- `ncta_cap=0` (default) = full saturation; test override possible
- Hard occupancy gate: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with actual dynamic smem + 200-reg consumer → if occ<1: `return cudaErrorLaunchOutOfResources` (NOT a hang; `fused_decoder_megakernel.cuh:1540-1545`). This is the "refuse, not hang" guarantee.
- SM-pinning via `%smid` intrinsic (`megakernel_common.cuh:65-73`) keeps optimizer state L2-warm on its owning SM.

### 14.2 Fusion tier selection (feasibility solver — REAL, build-time)
- Location: `grokking_optimizers/megakernel.py`, function `solve_all`
- Inputs: arch register/smem budget limits, per-cell register/smem footprint estimates
- Algorithm: picks highest tier (L3=fwd+bwd+opt fused, L1=optimizer-only) that fits within budget
- Output: 77/99 L3, 22/99 L1 (estimates, `ptxas -v` is the silicon arbiter)
- Winner is then refined by per-cell `maxrregcount` autotuner sweep in `compile.py`
- Note: this is a build-time decision, not runtime — the tier is baked into which `.cu` file is compiled

### 14.3 CTA tiling (REAL, build-time via autotuner + tournament)
- `SG_TUNED_TILE_M` (default 128): picked by autotuner from {64, 128, 256} via Optuna-TPE
- `SG_TUNED_TILE_N` (default 128): picked by autotuner from {64, 128, 256}
- `SG_TUNED_DEC_FWD_PIPE` (baked at 1): won by tournament (PIPE=2 lost -11.6%)
- `SG_TUNED_DEC_FWD_STAGES` (baked at 4): won by tournament (+1.49×)
- `SG_TUNED_DEC_DW_STAGE` (baked at 1): won by tournament (+2.05×)
- `SG_TUNED_VIT_P1_SUBTILE_S` (baked at 8): won by tournament (4.02×)
- `SG_TUNED_PROD_REGS=40 / SG_TUNED_CONS_REGS=232`: picked by autotuner
- Dead dims pinned (no kernel reads): GEMM-stage/interleave/swizzle/tma — auto-excluded from search

### 14.4 Dynamic smem allocation (REAL, runtime)
- Static cap = 48 KB; dynamic cap = 228 KB/SM (H100)
- Gate: `SG_DEC_TC_DYNAMIC_SMEM = (PIPE && STAGES>2)` → `extern __shared__ char[]` + `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(DecTcSmem))`
- Mamba always uses dynamic smem (`mamba3_layout.cuh:257-264`), smem ~215 KB at d=128
- ViT always uses dynamic smem, ~188 KB at d=128
- Decoder: uses dynamic smem only when PIPE=1 && STAGES=4 (current default)
- Budget enforcement: `static_assert(kMambaSmemBytes <= 228*1024)` + occupancy-refuse gate

### 14.5 Memory strategy (PARTIAL — recompute vs HBM-store)
- Current: within P1, backward recomputes each layer from `layer_in` (`dec_recompute_layer`) to stay under smem cap
- Planned (Fork-B design in DESIGN-TC-PIPELINE.md): dW output-stationary means cross-tile activations MUST be in HBM; within-tile recompute can stay
- The decision of recompute-vs-store is a BUILD-TIME choice per GEMM based on smem budget table (§4.3 of DESIGN-TC-PIPELINE.md), not a runtime decision
- **NOT IMPLEMENTED**: a runtime "memory-strategy gate" that dynamically switches offload/recompute based on available HBM. The smem budget calculations happen at design/build time.

### 14.6 Parallelism derivation (PARTIAL — scaffold real, auto-config unclear)
- `grokking_optimizers/distributed.py` (976 lines): `ParallelConfig` + `DistributedContext`
- 3D mesh: DP×TP×PP rank assignment (Megatron-style, TP innermost)
- ZeRO-3 sharding over NCCL/RCCL is ORTHOGONAL (not an axis)
- SP (sequence-parallel) and EP (expert-parallel) are in the design space but dense models use 3D
- All `torch.distributed` guarded → single-rank = no-op
- `FusedBackwardHook` / `MegakernelOptimizer` in `megakernel_engine.py` reconciles fused launch with framework
- **GAP**: These docs do NOT show a complete auto-inference algorithm for optimal DP/TP/PP from (model_size, seq_len, HBM). The "resource planner" mentioned in the CLAIMED state is not documented in these 9 files. The ParallelConfig exists but how it is auto-populated from workload analysis is not described here.

### 14.7 Self-specialization via if-constexpr (REAL, compile-time)
- The megakernel self-specializes via `if constexpr` on OptId: each staged phase (P2.4 SAM, P2.5 clip, P2.6 d-reduce, P2.7 NS, P2.45 SG-mu) is gated so every other cell is byte-identical (`:101-107` pattern)
- `apply_optimizer<Opt>` is an `if constexpr` ladder calling canonical algorithms
- The 33 sm_90 cells are physically distinct `.cu` files generated by `megakernel_codegen.py` — each cell includes EXACTLY the machinery its (model, optimizer) combination needs

### 14.8 Portability (REAL)
- `compile_config.toml` declares `tune_hook` → project-specific timing callback
- `BuildSpec.tune_hook` seam (`compile.py:8611`): any project exposes `fn(*, so_path, model, optimizer, arch, regime, seed) -> {output, elapsed_ms}`
- compile.py compares variant vs strict-math reference build — knowing nothing about the op
- `ArchEntry` table in compile.py is the single source of truth per arch (vendor, gencode, smem/reg/warp limits, features, roofline constants)
- Per-arch space builders (`_sm90_full_space`, `_build_cuda/cdna/rdna/pallas_space`) dispatch on vendor

### 14.9 Autotuner → product build linkage (REAL)
- `compile.py` JIT winner → `_kernel_tuned.json` (nested per-model: `{arch: {model: {optimizer: combo}}}`)
- `setup.py TunedBuildExtension` reads JSON + monkeypatches ninja writer → per-TU `-DSG_TUNED_*` + `--maxrregcount`
- TU→optimizer mapping: `launch_<opt>.cu` and `mega_<model>_<opt>.cu` get their optimizer's flags; bindings/model-only TUs get none
- Absent JSON = byte-identical to committed kernel (no `-D` flags emitted)

### 14.10 Summary: What is ACTUALLY auto-derived vs what is claimed

| Claimed adaptive behavior | Reality |
|---|---|
| Grid/occupancy from hardware | REAL — runtime SM count + occupancy gate |
| Fusion tier from register/smem budget | REAL — build-time via feasibility solver |
| CTA tiling from workload/occupancy | REAL — autotuner (empirical tournament) |
| Memory strategy (offload/recompute/streaming) | PARTIAL — smem budget decisions at build time; no runtime HBM-availability gate |
| Parallelism (3D→5D) auto-inferred | PARTIAL — scaffold (ParallelConfig) real; auto-derivation algorithm not documented in these files |
| Megakernel self-specializes by config | REAL — if-constexpr compile-time specialization |
| Portability via config (any project) | REAL — tune_hook seam + ArchEntry table |

---

## Discrepancies between these CORE docs and the CLAIMED current state
1. **The core docs PREDATE the claimed campaign.** CODEBASE_EXPLAINED.md and SESSION_STATE.md are dated
   2026-06-17 (a planning session) and describe the system BEFORE the TP / NVSHMEM / CuTe-atom GEMM engine
   / 3-flagship-LAUNCH / 8.09M-line cleanup work. None of them mention `SG_TUNED_GEMM_ENGINE`, NVSHMEM TP
   all-reduce, the resource planner, or the flagship dims (decoder d1600/L48, ViT d1664/L48, Mamba
   d2048/L24 per the prompt). The roofline numbers here are at d=2048 (decoder) / d=128 (mamba), not the
   flagship dims. So these docs are the *architectural reference* but are STALE on the live campaign state.
2. **README's "CUTLASS Sm90 collectives for the model GEMMs" vs the in-kernel hand-rolled wgmma reality.**
   The shipping persistent-megakernel GEMM is hand-rolled ss-wgmma; CUTLASS/`mma.cuh` is host-launched,
   used only for L1 Muon-NS/SG2, and the 2026-06-17 audit flags `mma.cuh` as DEAD (820 LoC, zero includers)
   recommended for removal.
3. **README's TF32-for-FP32 framing vs bf16 race precision** (DESIGN §5/App-B: bf16 is the owner decision;
   TF32 only as the `mma.cuh` L1 fallback).
4. **DESIGN-TC-PIPELINE.md scope is branch `claude/h100-audit-maximal`** (the stale HANDOFF era) and
   describes a pre-wgmma scalar state — fully superseded; keep only as design rationale.
5. **README "Implementation-maximal ... only remaining work is on-silicon validation"** is contradicted by
   SESSION_STATE/CODEBASE_EXPLAINED: P3 tails un-autotuned, 8/11 optimizers carry gate-coverage caveats,
   decoder at 6.48% roofline with fixable inefficiency, BUG-04 open. The README is aspirational vs the
   audit's "shipping single-GPU path is sound but 3 defect buckets + perf headroom remain."
6. **ENV pin SHA 4af83c3 (github peterc04) vs HEAD e69df73 local-only** — the github remote + old SHA in
   ENV_SNAPSHOT.txt contradicts the "LOCAL-ONLY, never push" doctrine and is well behind HEAD.
7. **verify_all/profile_maximal pass-counts vary across docs** (README cites 152/152, 156/0 self-test,
   17/17, 23/23; SESSION/audit cite self-test 265/0). The self-test expected count is `_SELF_TEST_EXPECTED_COUNT
   = 265` per CODEBASE_EXPLAINED — so README's "156 passed / 152/152" is stale vs the 265 count.
