# Phase-0 Document Digests: Exhaustive Summary and Discrepancy Analysis

Agent: phase0_doc_digests  
Source group: P_phase0_digests.txt (85 entries)  
Date: 2026-06-25  
Files read: 80+ (every non-cs_ .md in the group, plus .json/.txt artifacts)

---

## EXECUTIVE SUMMARY

The Phase-0 digests collectively describe the full historical arc of the SuperGrok2 project from early
"Stages 0-7" restructuring through the current L3-TC persistent wgmma megakernel campaign. The claimed
current state is: 33/33 cells L3-TC wgmma on sm_90, roofline measured at d=2048, decoder 618ms/6.48%,
wiring 33/33, tail-gate 33/33, A/A/A. **The code corroborates this for most claims but with important
caveats around gating completeness, tool correctness, and documentation drift.**

---

## PART 1: ARCHITECTURE DOCUMENTS (doc_arch, doc_hwval, doc_misc, doc_plan1, doc_plan2)

### doc_arch.md — archived_reports/ (Phases 3-7, BUILD, FIX2, MANDATE, MIGRATION, from_disk_backup/)

**What these reports describe:** The EARLIER "Stages 0-7" maximal-build campaign that produced the
99-cell surrogate composition + AMD/TPU ports. These are NOT current production truth.

Key historical facts (all cite archived_reports/):
- PHASE3_REPORT: 44/44 components built, 42 gate/trace-verified. All 33 sm_90 cells nvcc -c COMPILE_OK.
  THE PRE-TC STATE: scalar fused megakernel + separate host-launched CUTLASS GEMM (NOT in-megakernel).
  Self-test grew from 137→138→141→156→205→220 across phases.
- PHASE4_REPORT: WS1 register-pressure reduction (SMEM staging + setmaxnreg producer/consumer 256-thread
  split, consumer 200/producer 32). WS3 TF32 path added. Phase-4 L3 estimate: 77/99 (vs earlier 53/77).
- PHASE5_REPORT: AMD device-pass live-wiring; drift guard 2-tooth (structural single-source +
  content-hash manifest). gfx942 host launchers dispatched device kernels under #if defined(__HIPCC__).
- PHASE6_REPORT: "goal already met." No tree-merge performed. README rewritten 3235→~210 lines.
- PHASE7_REPORT: deleted 16 dead files, AMD vectorized f32x4 (11/11 AMDGCN_OK). Self-test 156/0.
- BUILD_REPORT: S1C decoder/ViT 11 matmuls → Sm90 TMA+WGMMA, SASS 64 HGMMA + 50 UTMALDG confirmed.
- FIX2_REPORT: compile.py audit — ON-DEVICE VALIDATION (#9/#10/#16) STRUCTURALLY BROKEN (TimingWorker
  frozen env never sees SG_DUMP_OUTPUT → num_status stays "skipped" → #16 gate drops EVERY generated
  variant; oracle shape mismatch → numerical_fail). Self-test 220/0 but gates never ran.
- MANDATE_REPORT: compile.py universalized to zero-manifest NVCC-parity compiler. 23-item status table.
- MIGRATION_REPORT: KEY — original production tree DOES NOT compile in include order on its own
  (launch_<opt>.cu uses WARP_SIZE/GROK_CUDA before #defining — "build compiles cleanly" was never
  literally true with real nvcc; --self-test validates Python pipeline NOT kernel compilation).
- H100_MAXIMALITY_REPORT (from_disk_backup/): **THE ONLY CONFIRMED ✅ on real silicon (2026-06-09)**.
  Shipped _ops.so 33.9MB built Jun 9 05:03. WGMMA/HGMMA=256 (144 HGMMA.64x128x8.F32.TF32 + 56 BF16
  + 56 F16), TMA=622 (580 UTMALDG.3D). GAP-1 (only fail): CUTLASS TF32-RS GEMM spills 8B → runtime-
  DEAD (vit_run_gemm_atb<float> returns cudaErrorNotSupported) → allowlisted. Max REG = 168/255.
- ROOFLINE (from_disk_backup/): **MEASUREMENT-INTEGRITY INCIDENT** — GPU NOT quiet during measurement
  (SIGSTOPped tuner fleet resuming every ~30s; 2nd unlaunched roofline PID holding 18.7GB/50% util).
  Headline (fp32, d=128): decoder TC 34.8st/s, eager-lion 57.4st/s (**eager >=4.9x L3**). TC beats
  eager on mamba (1.504 vs 1.051 TF/s). HONEST NEGATIVE: L3 scalar megakernel slower than eager at
  this scale.
- power_findings (from_disk_backup/): **SG2 NOT glorified AdamW but has BROKEN parts**: sam_step fails
  100% (perturbed_params non-leaf, supergrok2.py:1751, pre-existing git-blamed 6941457); bilevel ~50%;
  grokfast lamb runs away. At that time dispatch._FUSED_REGISTRY empty → race used eager+L1, NOT L3.
- vit_findings (from_disk_backup/): dW 4.15us/k-step = STAGING-LATENCY-BOUND (NOT memory-bound 0.15%
  HBM). Redundant vittc_dw_biases 132x-redundancy removal + other fixes = 716→206ms (3.48x), 1.32→4.60
  TF/s.

**CRITICAL CROSS-REPORT NOTE**: The archived reports describe the PRE-L3-TC state (scalar fused
megakernel + separate host-launched CUTLASS GEMM). Current production = L3-TC persistent wgmma
megakernel where wgmma runs INSIDE the persistent kernel. The ROOFLINE.md / power_findings
"dispatch._FUSED_REGISTRY empty → race uses eager+L1" is SUPERSEDED by all-33 L3-TC wiring.

---

### doc_hwval.md — HARDWARE_VALIDATION.md (1876 lines)

**Key finding: HARDWARE_VALIDATION.md is STALE.** The 99-cell matrix is frozen at 2026-06-09
(L1 optimizer-only audit). All 99 cells remain marked 🟡 in the doc despite the campaign having
wired, gated, and silicon-tested 33 L3-TC cells.

Real silicon ✅ status (the ONLY confirmed on-silicon in the doc):
- §2 H100 audit 2026-06-09: build/link/import/run ✅, L1 per-op fused optimizer kernels parity
  11/0, maximality 11/0, race 8/11. BOUNDARY: the 33 L3 megacells were compile-verified only.
- Stage 0: device link + cuobjdump SASS (WGMMA/UTMALDG) confirmed ✅.

Status of all deferred items: 🟡 (ncu/profiling, L2 hit-rate uplift, occupancy/SM-pin, TMA-reuse,
DSMEM, gfx942/TPU runtime, every item in §1C/Stage2/3/4/5/6/7).

Key per-arch tier facts (§1 matrix):
- transformer + vit: sm_90=L3 🟡, gfx942=L1 🟡, tpu_v6e=L3 🟡
- mamba: sm_90=L3 🟡, gfx942=L3 🟡, tpu_v6e=L3 🟡

Stage 6 in HARDWARE_VALIDATION says "3 cells wired" (mamba3+adamw, decoder+lion, vit+supergrok15) —
this is from an early demo. Production now wires all 33 via _L3_WGMMA_CELLS (dispatch.py).

SG2 bilevel adjoint (§1A): 24 weight-grad buffers, 20 are 🟢 full analytic; 4 are 🟡 (d_csa_compress_w
softmax-pool VJP; d_csa_idx_DQ/UQ/K=0 by stop-grad). 2 residual 🟡: GRU-gate recompute fallback drops
biases if caller passes empty gru gates; output-buffer zero-init is caller contract with no guard.
REVIEW_1A (Opus 4.8): ALL 24 buffers match autograd to fp32 (max abs ≤6e-9) across 4 configs.

---

### doc_misc.md — docs/reviews/, examples/, root .md files

**docs/reviews/ highlights:**
- REVIEW_0_2: Stage-0 header de-inline + Stage-2 L2-persist — NO BUGS FOUND.
- REVIEW_1A: SG2 bilevel adjoint — NO BUG. d_csa_idx_{DQ,UQ,K}=0 is MATHEMATICALLY CORRECT.
- REVIEW_1B: 9 SG2 MoE-compaction kernels — CORRECT + EQUIVALENT cross-backend. NOTE: _moe_step
  raises NotImplementedError on first line (dead at runtime).
- REVIEW_1C: Stage-1C decoder/ViT tensor-core GEMMs — ALL 11 CORRECT, no bugs.

**examples/autotune_demo/:** Real H100 sweep 24/24 configs, HONEST NEGATIVE: 1.04x speedup over
naive (4% — whole family ~14x slower than cuBLAS). USES FALLBACK PATH (not `python -m grokking_optimizers.compile` e2e because compile.py _validate() hard-rejects foreign optimizer name).

**examples/toy_tune_project/:** Three compile.py PORTABILITY GAPS documented: Gap 1 (tune-hook seam
half-wired), Gap 2 (csrc/fused glob hardcoded), Gap 3 (optimizer taxonomy hardcoded). Gap 1 noted
CLOSED in autotune_demo RESULT.md; Gaps 2+3 still open.

**Root .md plan/spec files:**
- .sam_spec.md: SAM 2nd-pass builder spec — in-kernel SAM between B2 and P2.5, workspace carved LAST
  so 6 green cells stay byte-identical.
- .sg2_spec.md: SG2 full meta-net BUILT; 4 deliverables remain; HONEST CAVEAT: won't GROK (CSA
  indexer drops idx_UQ → diverges for N>64).
- .hillclimb_loop.md: decoder d=1024 PROVEN (parity 2.4e-7, A/A/A); ~0.49% roofline at d=128 AND
  d=1024 (FLAT). cp.async mbarrier ring ATTEMPTED+REVERTED (3 blockers).
- .morning_report.md: HEADLINE: decoder d=1024 2084→967.3ms (2.15x); all-33 roofline mean 1.15%.
  4 KEPT (H1/H2/H3/Ports), cp.async ring reverted. 8x DECISION pending owner.

**ENV_SNAPSHOT.txt:** torch 2.4.1+cu124, jax 0.4.38, triton 3.0.0, optuna 4.9.0, pytest 9.1.0.
Repo: -e git+...peterc04/SuperGrok1.5@4af83c35.

---

### doc_plan1.md — PLANNING_INPUT, OPTIMIZATION_LEDGER, COMPILE_AUDIT, AUTOTUNE_LINKAGE, PERF_ANALYSIS

**PLANNING_INPUT.md (2026-06-17):**
- Done/maxed: 33 cells, one persistent kernel, ONE launch, 1 CTA/SM, hand-built GridBarrier.
- Shipped wins: decoder PIPE=1/STAGES=4 (+1.49x, 920→618ms); ViT B1 sub-tile S=8 (4.02x, pending confirm).
- BUG-04: mamba kMbStagedOptScratch un-gated → OOM at d≥1024 for muon/neuralgrok/looksam mamba cells.

**OPTIMIZATION_LEDGER.md (THE perf ledger):**
- Track A static patches (.regpressure/0001..0005): 0001 bf16-prestage KEEP; 0002 decoder SAM-scoped
  outline KEEP; 0003 vit SAM outline REVERT (+5% slower); 0004 mamba scope-noinline KEEP (-4.4%);
  0005 decoder cp.async ring KEEP (-14.2%).
- KEY WIN 1 — dW contiguous-layout staging KEEP +2.05x (SG_TUNED_DEC_DW_STAGE=1):
  grid-cooperative pre-transpose writes dY/X once/step to K-contiguous scratch → reuses kRingAsync.
  @d=2048/B=16384: 1889.8→920.7ms. fp64+A/A/A GREEN 11/11.
- KEY WIN 2 — decoder fwd/dX deeper cp.async ring KEEP +1.49x (SG_TUNED_DEC_FWD_PIPE=1, STAGES=4):
  deepen kRingAsync ring. Blocked earlier by DecTcSmem static→dynamic conversion (S=3 static 51.3KB
  > 48KB cap; dynamic S=3 25360B fits 227KB). @d=2048: 920.6→618.5ms. fp64+A/A/A GREEN 11/11.
  CUMULATIVE: 1889→618ms ~3.05x; roofline 2.08%→6.475%.
- IL=4 GEMM-interleave REJECT: bench -9.3% (fastest lever) BUT A/A/A FAILED ALL 10 decoder cells
  (ragged M-atom group-reduce non-deterministic at d=128). KEY LESSON: bench-first necessary NOT sufficient.
- ViT bottleneck @d=2048: B1_barrier 51.2% DOMINANT (load imbalance); P2_dW only 3.0%.
- META-LESSON: decoder GEMM-staging win does NOT generalize (ViT=non-GEMM-surface-bound, Mamba=scan-bound).

**COMPILE_AUDIT.md (audit of compile.py 31,461 lines):**
- P0 BLOCKERS: (1) fp64 oracle NOT wired into pick_winner (uses fp32 self-consistency). (2) polyhedral/
  CUTLASS/CK winners ship TEMPLATE while reporting generated origin (fake-green). (3) IL=4 determinism
  trap open by default. (4) fast-math cache fm_sig drops version-gated flags → stale .so can win.
- P1: CLI INVERTS "maximal" — plain invocation runs all generative layers OFF. #24 roofline objective
  ABSENT. #23 tiered spill PARTIAL — parsed spill bytes consumed by NOTHING. "Byte-identical untuned
  build" FALSE — sm_90 builder canonical values diverge from kernel #define defaults.
- P2: Level-2 superoptimizer ~70% scaffold (polyhedral=identity-copy, synth GEMM never updates m/v,
  native wgmma STUB _MMA_NATIVE_LOADS_WIRED=False). compile_config module import (17473) DOES NOT EXIST
  → ModuleNotFoundError swallowed → all TOML config knobs silently ignored.
- P3 DEAD: 14 stall-reason features permanently zero; dead dims mb_dw_splitk removed 2026-06-17;
  mb_gemm_stages/mb_gemm_interleave/min_blocks still auto-pinned dead. Blackwell routes generic → no TC.

**AUTOTUNE_LINKAGE.md:**
- Linkage: compile.py build_jit → _kernel_tuned.json → setup.py TunedBuildExtension injects per-TU flags.
- 5 SAFE dims: SG_TUNED_MEGA_BLOCK/TILE_M/TILE_N/dec_dw_splitk/vit_dw_splitk/prod_regs/cons_regs +
  --maxrregcount. Old block/vec/unroll/async_depth DELETED (were for removed eager kernels).
- IMPORTANT schema quirk NOW FIXED: was one winner per (arch, optimizer) LAST-WINS; now nested
  {arch:{model:{optimizer:combo}}} per model.

**PERF_ANALYSIS.md (SUPERSEDED IN PART):** P0 dW pipelined GEMM REVERTED (-11.6%). dW contiguous-
staging redirect = +2.05x KEEP. Post-dW bottleneck: decoder=fwd/dX drain-bound + B1 barrier 19%;
ViT=B1-barrier load-imbalance 51% (dW only 3%); Mamba=scan-bound.

---

### doc_plan2.md — .campaign_plan, PHASE1_CAMPAIGN, BUILD_AND_VALIDATE, DESIGN-TC-PIPELINE, MAMBA3_REFERENCE, SUPEROPTIMIZER_*, INTEGRATION-*, HANDOFF

**PARALLELISM DECISION FINAL (owner 2026-06-12):** 4D + ZeRO-3 + MAX BATCH. NOT a fixed "if
num_gpus==1". Base = 3D (DP×TP×PP); SP axis must be a LOUD build error (static_assert); EP axis
for MoE. ZeRO-3 is ORTHOGONAL sharding layered on top.

**Phase 1 CAMPAIGN highlights:**
- 3 canonical model sizes (owner-locked 2026-06-16): decoder GPT-2 XL d=1600/L48/h25 ~1.5B; vit
  ViT-G/14 d=1664/L48/h16/MLP8192 ~1.8B; mamba Mamba-3 d=2048/L24/state128/head_dim64/d_ff4096 ~1.528B.
- #14 FINDINGS: megakernel ALREADY multi-CTA-tiled (131 CTAs); smem CLIFF REFUTED — TC megakernel smem
  is d-INDEPENDENT (acts stage through HBM workspace): decoder 13.7KB/vit 7.7KB/mamba 14.8KB FLAT
  128→2048 all <<228KB. Width-sensitive = REGISTERS/SPILLS not smem.
- Cp.async ring BLOCKED: (a) fp32 weights converted on READ (__float2bfloat16), cp.async can't convert;
  (b) dW both operands transposed-strided → needs TMA-with-transpose; (c) 255 regs WITH 2636B spills
  already → zero headroom for mbarrier/ring.

**DESIGN-TC-PIPELINE.md** is an OLDER planning doc — pre-TC state (scalar triple loops). Its smem
numbers (decoder 42KB/mamba 145124B/vit 188080B) are the SCALAR SampleSmem, contradicted by #14's
measured TC megakernel d-independent 13.7/14.8/7.7KB. Scalar fallback kernel is DEAD (wiring 33/33).

**MAMBA3_REFERENCE.md:** Mamba-3 (arXiv 2603.15569 ICLR 2026) reversible foundation. Reference +
fp64 oracle + manual backward matches autograd ≤4e-15 (PASS). Mamba-3 upgrade in PROGRESS: Phase 1
DONE+validated; Phases 2-4 (L3-TC megakernel) NOT done — "CUDA megakernel untouched."

**SUPEROPTIMIZER_SCOPING.md:** NO-GO on full general superoptimizer. DECISIVE CONSTRAINT: fp32 add
non-associative → any reassociation fails A/A/A determinism → legal space = TRANSPORT-ONLY rewrites.

**INTEGRATION-MAMBA.md:** Describes Mamba-1 (NOT Mamba-3 of MAMBA3_REFERENCE). MambaSampleSmem=145124B.

**HANDOFF.md (2026-06-12, branch claude/h100-audit-maximal):** 19 overnight commits through 642e360
(33/33 real L3-TC wgmma, roofline mean 1.15%). Phase-2 authored (TP loopback NVSHMEM behind
-DSG_HAS_NVSHMEM seam). NVSHMEM NOT INSTALLED in env (verified).

**KEY CONTRADICTIONS WITHIN doc_plan2:**
1. DESIGN-TC-PIPELINE "NO in-kernel wgmma; scalar triple loops" = PRE-build state, now IMPLEMENTED.
2. BUILD_AND_VALIDATE surrogate L3 model stages (GELU(params+input)) now DEAD on race path.
3. INTEGRATION-MAMBA describes Mamba-1 (28 tensors/259425); MAMBA3_REFERENCE = Mamba-3 upgrade (NOT
   in production megakernel yet — "CUDA megakernel untouched").
4. C7515 wgmma serialization STILL PRESENT after H1; true cross-k overlap needs cp.async producer/
   consumer mbarrier ring = BIG rewrite, blocked on 3 hard constraints.

---

## PART 2: TEST FILES (t_gate, t_tc, t_nonhw, t_oracles, t_rest)

### t_gate.md — test_l3tc_tail_gate.py + sg2_kernel_mirror.py

**test_l3tc_tail_gate.py (1512 lines):** THE per-cell L3-TC conversion gate.
- 33 cells registered in _CELLS; all 3 models × 11 optimizers including supergrok2 (factory=None,
  routes to dedicated _sg2_l3tc_gate.run_sg2_gate).
- Gate (1) = fp64 + A/A/A. Precondition: has_l3_real AND gemm_impl=="wgmma".
- SG11/15 dedicated gate (_run_sg_cell_gate): 4 surfaces including sharpness vs PURE-fp64 2nd backward.
- run_sg11_warmup_gate: discriminates cos(grad,mu) [wrong] vs cos(grad,momentum) [canonical]; MUST be
  run but is CLI-only (`--sg11-warmup`) — NOT pytest-collected (RG1 gap).
- _CONTAMINATION_ISOLATED_OPTS: SG2 isolated into subprocess (device-global leaks in 33-cell run).
- _BLOCKED_EVIDENCE={} (grokadamw/vit now CONVERTED so empty).

**sg2_kernel_mirror.py (1001 lines):** Single-threaded structural mirror of opt_stage_supergrok2.cuh.
Catches index/dead-buffer bug class. Oracle is LOW-RANK kernel path (qI=x@idx_DQ, kI=pooled@idx_K,
rank-dim dot), NOT full-rank eager CSAHCAMetaNet. Agreement ~1e-12 confirms structural correctness.

---

### t_tc.md — test_decoder_tc, test_vit_tc, test_mamba_tc, oracles, mirrors

Architecture constants (single source of truth for flat layouts):
- decoder: VOCAB=99, D=128, H=4, L=2, SEQ=4, D_FF=512 → 30 tensors, 422755 total.
- vit: D=128, H=4, L=2, PATCH_DIM=49, NUM_PATCHES=16, SEQ=17 → 32 tensors, 418017 total.
- mamba (Mamba-3 toy): d=128, nl=2, state_dim=128, head_dim=64 → 45 tensors, 593713 total.
- mamba_oracle (Mamba-1): 28 tensors, 259425 total.

Key gate: _bf16_faithful_oracle (the R2.3 named reference): model fwd+bwd in fp64 but with EVERY
value the TC kernel STORES rounded to bf16 at SAME points. Real bug shows rel~1; bf16-storage ~bf16-floor.

Loss tol tightened to 1e-4 (was 5e-3) AFTER a bias-omission bug rode at 2.52e-4 with fixed kernel at
2.85e-5 — calibration provenance documented.

PART 1 micro-gates: three operand orientations (fwd Y=X@Wᵀ, dX=dY@W, dW multi-k-step). test_dw_random_
multistep is THE stride-bug gate (single tile passes with wrong k-stride).

---

### t_nonhw.md — conftest.py, test_shard_map, test_pipeline_schedule, test_zero3_plan, tpu tests

test_pallas_parity_interpret.py: TPU v6e Pallas optimizer parity (PRESERVED-BY-DESIGN). Runs 11
optimizers in interpret=True (CPU). Notable:
- muon NOT in pallas_call harness (NS orthogonalization is matmul+reduce, not per-element body).
- test_supergrok2_live_update_parity: LIVE sg2_update path with DIFFERENTIAL reconstruction (real
  meta_net runs in reference so smart_g cancels, leaving grokfast+Adam tail in fp64).

test_pipeline_schedule.py: CPU gates for 1F1B. LAST stage STRICTLY ALTERNATES fwd→bwd with zero warmup
(key property that lets PP stage kernel keep last stage's fwd+bwd FUSED in one launch).

test_zero3_plan.py: ZeRO-3 flat-blob plan + checkpoint round-trip. Mismatched plan raises RuntimeError.

---

### t_oracles.md — mamba3_oracle, test_reference_parity, test_multistep_parity, mamba_oracle

**test_reference_parity.py:** fp64 reference ref_<opt>_step (11 optimizers). Single source = csrc/
algorithms/<opt>.h transcription. Key facts:
- ref_muon_step: buf=momentum·buf+g (NO (1-momentum)); NS coeffs (3.4445,-4.7750,2.0315).
- ref_prodigy_partials: r=Σg·(p_init-p)·d_prev²; s=Σd_prev²·|g| (degree-2 d_prev, L1 norm |g|).
  A stale revision used d_prev^1 + signed g.
- ref_sg2_apply_step: lamb_eff RESTORED (was (void)-ed/dropped before fix).
- ref_sg_phi_forward: EXACT-erf GELU (NOT tanh — important for correctness).

GPU half test_kernel_matches_reference_gpu: ONLY AdamW closed-form parity (atol 1e-4/rtol 1e-3);
others smoke (finite + moved) — 10 of 11 optimizers never get numeric GPU check in CI.

mamba3_oracle.py: Mamba-3 complex scan fwd/bwd with 2-adjoint carries (gh + gv coupling from t+1 β-term).
Foundation gate manual_vs_autograd ≤1e-10 per param. VERIFIED.

---

### t_rest.md — remaining tests/hw/*.py + *.cu

**Key parallelism gates:**
- test_sharded_optimizer.py (DELIVERABLE 1): DP=1 sharded_optimizer_kernel == in-kernel P3 fused tail
  BIT-IDENTICAL for adamw/lion/grokfast × 3 models.
- test_tp_loopback.py (§5.1/§5.2): TP∈{2,4} virtual ranks via tp_loopback_binding.cu. 5 asserts
  including dW/db slice-exactness. The NVSHMEM transport swap is the 8×H100 task (no math change).
- test_pp2_loopback_determinism.py: REQUIRES `.phase2/patches/0001-dectc-layer-range-pp.patch` (SKIPS
  without it).
- test_step_graph_capture.py: CUDA graph of [ops.fused_step → sharded_optimizer_kernel<Opt>]. Docstring
  HONESTLY documents: cross-rank collectives NOT captured (megakernel grabs all SMs while peer waits).
- test_3d_parallel.py: 7B DP×TP×PP+ZeRO3 smoke SKIPS without WORLD>1 + accelerator. PASS bar
  weak-scaling ≥0.70 (never run in this session).

**Mamba probes:**
- _mamba_race_probe.py + _mamba_prodigy_probe.cu: diagnostic that localized the __noinline__ +
  scan-bwd footprint reduction fix for mamba×prodigy/looksam A/A/A race.
- test_l3tc_tail_gate._CELLS now register prodigy/mamba + looksam/mamba as CONVERTED (race FIXED).

---

## PART 3: PROFILING AND TUNING (go_profile, go_opt_b, tune_a, tune_b, perf_dir, regpressure, results_chr)

### go_profile.md — profile.py, profile_maximal.py, utilization.py, verify_all.py, _tuned_inject.py

**profile_maximal.py (key design):**
- Dead-spill allowlist: EXACTLY ONE allowlisted spiller — CUTLASS TF32-RS GEMM (MMA_64x128x8_F32TF32
  TF32_RS_TN + MainloopSm90TmaGmmaRmemAWarpSpecialized + tfloat32_t — all 3 required, FAIL-CLOSED).
  Runtime-DEAD: vit_run_gemm_atb<float> returns cudaErrorNotSupported → scalar fallback.
- SASS audit: counts wgmma (HGMMA/UGMMA/WGMMA), TMA (UTMALDG/UBLKCP/cp.async.bulk), mbarrier, mufu, ffma.

**verify_all.py (authoritative 99-cell gate):**
- 6 phases. Phase 3 = THE CORE: 99 cells compose (sm_90 via compile_to_object.sh, gfx942 via
  amdgcn_check.sh, tpu_v6e via trace_check).
- NOTE (Phase 6, lines 599-606): gfx942 decoder/vit L1 demotion is an ESTIMATE pending silicon
  (LDS over-count bwd 66560>65536). This is the one tier non-maximality — silicon-gated.

**_tuned_inject.py MACROS TABLE (LIVE sm_90 macros):**
| dim | macro | default | consumer file |
|-----|-------|---------|---------------|
| mega_block | SG_TUNED_MEGA_BLOCK | 256 | megakernel_common.cuh:50 |
| tile_m | SG_TUNED_TILE_M | 128 | wgmma.cuh:136 |
| tile_n | SG_TUNED_TILE_N | 128 | wgmma.cuh:133 |
| dec_dw_splitk | SG_TUNED_DEC_DW_SPLITK | 1 | model_stage_decoder_tc.cuh:101 |
| vit_dw_splitk | SG_TUNED_VIT_DW_SPLITK | 4 | model_stage_vit_tc.cuh:105 |
| prod_regs | SG_TUNED_PROD_REGS | 40 | tile_pipeline.cuh:92 |
| cons_regs | SG_TUNED_CONS_REGS | 232 | tile_pipeline.cuh:95 |
| maxrregcount | --maxrregcount | 0 (unset → emit nothing) | ptxas flag |

mb_dw_splitk REMOVED 2026-06-17 (Mamba-3 TC rewrite dropped output-stationary dW split-K).
DRIFT: compile_config.toml still lists mb_dw_splitk and dec_dw default=4 (vs MACROS default=1).

---

### go_opt_b.md — neuralgrok, muon, supergrok11/15/2

**PROJECT-TRUTH alignment:** Every optimizer is now PURE L3-TC. eager .step() REMOVED → NotImplementedError
in muon.py:154, neuralgrok.py, supergrok11.py:555, supergrok15.py:584, supergrok2.py:2377.

**neuralgrok.py:**
- KERNEL_PSI_HIDDEN = 16 MUST equal kernel kPsiHidden (opt_components.cuh static constexpr int kPsiHidden=16).
- _Amplifier default hidden_dim=128, num_layers=3 — MISMATCH with required 16/2. OPTIMIZER_CONFIGS
  must override or psi_pack() raises.
- A3 FIX: train_amplifier_step now rebuilds amplified update DIFFERENTIABLY through the amplifier
  (was autograd-unreachable before).

**supergrok11/15:** SharpnessMetaNet hidden_dim: dispatch.py seam at :1774-1810 asserts H==32
(the known 64-bug guard). SG11 gate = cos(grad,momentum); SG15 gate = training-accuracy scalar.

**supergrok2.py (2397 lines):**
- idx_UQ is NOT in the kernel bundle (only idx_DQ + idx_K; kernel fuses DQ·UQ and K·UQ).
- bilevel_step PURE L3-TC: C++ VJP DROPPED; ALWAYS routes to _bilevel_step_autograd.
- _HAS_CUDA is now False (pybinds removed from bindings.cpp).
- _moe_step raises NotImplementedError on line 1 (dead at runtime), sg2_fused_step call inside it
  references kernels that are REMOVED.
- CompiledSuperGrok2._capture_graph calls removed step_compiled → would fail.

---

### tune_a.md — roofline.py, tune_optimizers.py, decoder/vit/mamba_bench.py, precision_analysis.py

**roofline.py BATCH SATURATION finding (HARD-CODED tables):**
TC megakernel SATURATES at B≈2k (one-CTA/SM occupancy-pinned, 16-row dW atom × 132 SMs ≈ 2112) and
DECLINES past 16k. Eager-lion keeps climbing to ~32-65k. Path to higher fraction = multi-CTA-per-tensor
tiling, NOT bigger batch.

**tune_optimizers.py:** KILL-SWITCH: .STOP_TUNING sentinel EXISTS in repo root (0 bytes) → module
refuses to run. INTENDED kill-switch.

**mamba_bench.py:** d=1024 blocked (MambaSampleSmem overflows ~228KB/SM at d_inner=2048). Throughput-
only (no profiler/phase split at d=128).

---

### tune_b.md — tuning/_*.py probes/validators

**oracle_trust_audit.json (3 CONFIRMED co-wrong oracles):**
1. SG2-B1 CSA indexer co-wrong: sg2_kernel_mirror::oracle_step implements KERNEL's LOW-RANK indexer
   (idx_UQ DROPPED, /sqrt(rank)) not canonical full-rank path. N>64 fidelity probe ONLY REPORTS
   (sets no _ok flag; verdict passes silently). NOT fixed by gate.
2. SG11 cold-gate CI-blind: test_l3tc_cell_gate runs COLD step-1 (m0=zeros → cos_t=0 → gate=0.5 fixed).
   Discriminating warmup gate is CLI-only. phantom test_sg11_gate_warmup_catches_momentum_bug has NO
   DEFINITION in repo.
3. SG11 gate-oracle is LOCAL TWIN: _sg11_gate_oracle_fp64 and _sg11_gate_mirror_fp32 compute IDENTICAL
   expression; a co-edit swapping BOTH twins to cos(grad,mu) passes cleanly.

**test_coverage_gap_audit.json (16 ranked gaps RG1-RG16):**
- RG1 HIGH: SG11/15 warmup gate never pytest-collected; SG15 has NO warm path (hard-asserts opt=="supergrok11").
- RG2 HIGH: multistep parity ONLY grokadamw+prodigy ONLY decoder; muon/looksam/sg11/sg15/sg2 single-step-only.
- RG3 HIGH: ViT SG_TUNED_VIT_P1_SUBTILE_S<64 ON-path NEVER compiled (+4.02x lever ships UNTESTED).
- RG4 HIGH: autotune correctness hook runs d=128 but docstring says 'PRODUCTION-scale d=2048' (co-derivation shape).
- RG5 HIGH: opt_stages_precompute.cuh PRODUCTION path validated ONLY vs CPU Python fp32 mirrors.
- RG6 HIGH: every fp64+A/A/A gate at toy d=128 single-CTA; multi-CTA reductions ungated.

**author_queue1/2/3 authored ready-to-gate diffs (NOT yet applied):**
- staged_opt_scratch_gate.diff: BUG-04 fix (kMbStagedOptScratch gate, byte-identical).
- int64_offsets.diff: widen offsets int32→int64 (byte-identical at current scale).
- gate_hardening.diff: real sg11 warmup gate + SG2 fullrank oracle.
- phase2_pp_readiness.diff: PP bring-up across 6 files.
- compile_keying_bug.diff: fix _emitted_sources KEYING bug (false-green on generative origin).
- autotuner_new_knobs.diff (RG3): vit_p1_subtile_s [64,32,16,8] swept. NOTE: run.log shows this
  diff "did not apply cleanly → skip" in live front-load.
- m0_mamba_body_rewire.diff: M0 mamba wgmma projection under SG_TUNED_MB_PROJ_WGMMA (byte-identical OFF).

---

### perf_dir.md — .perf/ (audits + reprofiles + author_queues)

**phase1_status_audit.md roofline:**
| model | config | ms/step | TF/s | % of 989 | bound |
|-------|--------|---------|------|-----------|-------|
| decoder | d2048 B16384 | 618.5 | 64.0 | **6.48%** | latency/serialization |
| vit | d2048 B1024 | 5759 base → **~1434 (S=8, 4.02x)** | 1.83→7.3 | 0.185→0.74% | B1 load-imbalance |
| mamba3 | d128 B4096 | 221.6 | 0.30 | **0.0303%** | scan-dominated |

Decoder reprofile (clock64 critical-path summed ~646ms vs wall 616ms):
P1_fwd 27.6% / P1_bwd 27.3% / B1_barrier 16.8% / P2_dW 16.5% / P2_grad_asm 2.6% / P3_opt_tail **5.9%** (38ms) / B2 3.1% / B0 0.1%.
Grid-barrier wait total B0+B1+B2 ~129.4ms (20.0%).

fwd_fine reprofile: COMPUTE/EPI-bound (WAIT 9.9%, WGMMA 46.6%), NOT drain-bound. VERIFIER REFUTE:
decoder fwd is NO LONGER drain-bound (deeper-ring KEEP fixed it).

Autotune FAILED: rc=2/rc=1/rc=124 across all decoder/vit runs; autotune_decoder_adamw.log stalls at
"0/4 [00:00<?, ?phase/s]" — zero usable winners. front-load FIXED CUDA_HOME=/usr/local/cuda-12.4 and
g++-cached PATH issues AFTER the run.

---

### regpressure.md — .regpressure/** + .phase2/**

**.regpressure/ (static register-pressure campaign, HEAD 642e360):**
- KEY FACT: every TC kernel reports Used 255 registers (__launch_bounds__(256) fills the file).
  Occupancy = 1 CTA/SM by design. Spill BYTES are the pressure metric, not reg count.
- --maxrregcount SILENTLY IGNORED (launch_bounds wins; verified byte-identical at 240/224/192).
- WgmmaAccum<128> = 64 fp32 regs/fragment; kIL=2 keeps 2 live = 128 regs. Halving IL or TILE_N=64
  → 253 regs / 0 spills.
- SAM-coupled cells (looksam/sg11/sg15/sg2) spill ~15KB on decoder/vit (duplicated engine body P1+P2.4).
- mamba spills ~5.8KB on EVERY cell (callee-inclusive from __noinline__ tile fns).
- PATCH VERDICTS: 0002 decoder SAM-scoped KEEP; 0003 vit SAM REVERT (+5.11% SLOWER); 0004 mamba
  scope-noinline KEEP; 0005 decoder cp.async ring KEEP (ALL decoder cells -14.23% mean).

**roofline_BASE.json key numbers (bf16-forced, all 33 rows, engine=wgmma):**
- decoder: adamw 4.868/1.71%, lion 4.843/1.70%, supergrok2 0.218/0.07% (788ms outlier, ~20x slower).
- vit: adamw 5.628/1.81%, supergrok2 0.916/0.28% (804ms).
- mamba: adamw 2.007/0.82%, neuralgrok 0.708/0.28%, supergrok2 0.242/0.09% (860ms).

**.phase2/ (multi-GPU Phase-2 authoring):**
- NVSHMEM NOT INSTALLED (verified in env) → TP against LoopbackTransport (bit-exact loopback, testable).
- CPU-only: all GPU tests AUTHORED + nvcc-compiled but NOT RUN.
- PHASE2_REVIEW_FINDINGS.md: multi-GPU shard MATH CORRECT, scaffolding COMPLETE.
  Only finding: F6a [MINOR] — tp_layer.cuh:56-70 line citations drifted stale (decoder header grew
  1100→1946 lines) → FIXED (comment-only commit 2ff75a5).
- PP=2 gate REQUIRES `.phase2/patches/0001-dectc-layer-range-pp.patch` (applied separately; SKIPS loudly without it). The PP patch was confirmed NOT BIT-ROTTED in this phase review (PTX identical, only scheduling jitter in one mov instruction).

---

### results_chr.md — results/h100_grokking_race/ + results/tuning/ + tune11_out/

**Race result (README.md, real H100 2026-06-09, L1 per-op optimizer path):**
8/11 grok. Ranked: Muon (400), Prodigy (1000 — NOT sustained, final 0.007), Grokfast (2600), AdamW (3000),
LookSAM (3200), Lion (4000), NeuralGrok (4800), GrokAdamW (5000). DNF: SG1.5 (peak 0.918), SG1.1 (0.020),
SG2 (0.017). DNFs = research-owned dynamics NOT kernel bugs.

**IMPORTANT CONTRADICTION with ROOFLINE.md:** wiring_check.json = 33/33 L3-TC(wgmma); ROOFLINE.md says
"only adamw trio has real L3-TC megakernel." RECONCILIATION: temporal — wiring_check.json is the LATER
state (all-33 L3-TC); ROOFLINE.md and POWER_PROFILING.md are from the earlier eager+L1 era.

**mamba TC vs scalar inconsistency:** mbtc_bypass_profile.json shows TC 2.15× faster than scalar at
d=128 B=16384 (JIT bench path). ROOFLINE.md ships SCALAR megakernel for mamba×adamw (flagging TC as
0.46× slower at production op-point). Two different measurement contexts; production routing choice (scalar)
is the validated one.

**Mamba-3 does NOT toy-grok** at d=128 (RETUNE_STATUS 2026-06-16) — memorizes train ~step 80-100 but
val/test pinned at ~0.01. No valid tuned_configs_mamba.json exists.

**tune11_out/ autotuner run (adamw/decoder/sm_90a, bayesian, 10-cap, PGO=off):**
- SEARCH-SPACE BUG: SG_TUNED_TILE_N=256 sampled but unbuildable (wgmma bf16 N must be ≤128) → 5/9
  trials fail at wgmma.cuh:578 static_assert.
- GATE-FEEDBACK BUG: optuna's objective is timing-only. All 3 buildable+timed trials (14.71/15.13/17.84ms)
  → numerical_fail (variant hooks diverge ~0.08 from ref vs 1e-4 gate). Optuna marks them COMPLETE/FINITE.
- gemm_impl lever is DEAD: source .cu files hard-#define SG_TUNED_GEMM_IMPL 1 overriding tuner's
  -DSG_TUNED_GEMM_IMPL=SG_GEMM_IMPL_SCALAR (redefinition warnings on every TU).
- CLOCK-LOCK WARNING: clocks NOT locked (-lgc failed rc=4) → timings reflect boost/thermal noise.
- tuned_config: null — NO usable tuned config produced. Production stays on untuned baseline .so.

**build/ (prior pgo_instrument campaign):**
- All 12 ViT cells FAILED at g++-cached -v probe (CXX issue, not code bugs).
- Both decoder cells: incomplete partial .o builds, NO .so produced.
- .compile_cache.json: zero successful builds.

---

### preserve_trees.md — gfx942 (HIP) + tpu_v6e/pallas PRESERVED

Structure CONFIRMED for all four trees:
- csrc/backends/hip/gfx942/: two-pass single-header design; ATen host orchestration + REAL AMDGCN device
  kernels; HARDWARE-GATED 🟡 (device-compile-verified via amdgcn_check.sh; MI300X numeric parity DEFERRED).
- csrc/fused/gfx942/: 33 generated mega cells; fused_megakernel.hip.hpp REAL composition L3/L1.
- kernels/gfx942/: 16 files including 93KB supergrok2_gfx942.hip.hpp + 91KB mamba3_gfx942.hip.hpp.
- csrc/backends/pallas/: 11 launch modules; _pallas_kernels.py (Pallas affine scan + expert gather);
  v5p/v6e split (128 vs 256 tile width). PRESERVED-BY-DESIGN, never flag dead.
- csrc/fused/tpu_v6e/: 33 generated cells. Each calls real _pallas_fused.fused_step (XLA fuses).

---

### scripts_all.md — scripts/*.py + scripts/*.sh

**check_math_single_source.py:**
- BINDING_FUNCS=() empty — eager per-op pybind entrypoints REMOVED; production = unified fused_step ABI.
- _consumer_headers = ONLY csrc/fused/sm_90/opt_components.cuh (per-op kernels/sm_90/*_sm90.cuh removed).
- SG2_SEPARATE_PATH: missing #include is WARN not FAIL (TODO: unify post-refactor).

**nvcc_baseline.py:** The correct 3-point Task-#11 comparison (A=regular nvcc, B=compile.py default,
C=compile.py tuned). _GENCODE sm_90a ONLY (plain compute_90 makes ptxas reject every wgmma).

**fast_triage.py:** CALIBRATION — only ONE calibrated ground-truth: decoder/model-triage dW staging
KEEP +2.05× (STAGE 0→1, 1889.8→920.7ms @d=2048). All other pairs UNCALIBRATED.

**diag_neuralgrok_seed123.py:** REAL gate-side defect: per-op eager neuralgrok compiled at NG_H=64 while
deployed megakernel uses kPsiHidden=16 → OOB reads of 48 floats/weight, seed-dependent sign flips.
Verdict: gate at fault (anchor to fp64-H16 oracle), NOT the kernel. Per-op NG_H=64 is a separate defect.

**.STOP_TUNING sentinel EXISTS in repo root** (0 bytes) → tune_optimizers.py refuses to run.

---

### dispatch.md — grokking_optimizers/dispatch.py

**Config-derivation (THE mechanism for the central design thesis):**
- `_FUSED_L3_REAL` frozenset (dispatch.py:630-962): **33 cells (verified)** — ALL 3 models × 11
  optimizers. This is the single source of truth. BUT STALE DOC: docstring at :604/617 says "Currently
  only (transformer_decoder, adamw)" — drift, NOT a logic bug.
- `has_l3_real()` gates the real path to impl==90 (sm_90 ONLY) — gfx942/tpu keep eager path here.
- `_L3_WGMMA_CELLS` frozenset: IDENTICAL membership to _FUSED_L3_REAL (all 33). Non-members → hard-fail.
- State-layout: prodigy 4*total+4; SG11/15 4*total+1+kSgPhiPack(=4*32+1=129); SG2 (5+GH)*total+1,
  GH=4 → 9*total+1; else 3*total+1.
- SG11/15 seam: W1[H,2], W2[H], H=32; RuntimeError if H≠32.
- SG2 dispatch: builds 26-tensor weight bundle via net.get_weights(None); 6 per-tensor scalar arrays;
  calls ops.sg2_fused_step; verified 1:1 against bindings.cpp:228 and dispatch.cpp:1205.

**STALE COMMENTS in dispatch.py:**
1. fused_train_step docstring says default gemm_impl="scalar" — actual default="wgmma", scalar REMOVED.
2. _FUSED_L3_REAL docstring says "Currently only (transformer_decoder, adamw)" — now all 33 cells.
3. _L3_WGMMA_CELLS comments say "returns scalar" — now returns None → hard-fail.

---

## PART 4: OTHER KEY PHASE0 FILES

### phase0/PRIOR_STATE_AND_CHECKLIST.md

HARDWARE_VALIDATION.md is STALE (doc-drift): the campaign superseded the L3-megacell rows on silicon
but never updated the doc's markers. 

**Prompt assertion checks:**
- 33 megakernels on sm_90 → CONFIRMED (ls csrc/fused/sm_90/mega_*.cu = exactly 33).
- Parallelism strategy = 4D + ZeRO-3 + max batch (NOT hardcoded rules) → CONFIRMED present but DORMANT
  (CPU-tested, GPU-UNRUN; PP patch bit-rotted concern now REFUTED per .phase2/REPORT.md).
- mod-97 tasks CURRENT; FineWeb/ImageNet/GiftEval = NOT STARTED → CONFIRMED.
- L3 megakernel structure is B0→P1(fwd+bwd)→B1→P2(cross-CTA grad reduce)→B2→P3(opt). THREE barriers,
  fwd+bwd together (NOT barrier-separated as a naive reading might suggest).

**Known open bugs from prior sessions:**
- BUG-04 mamba staged-opt scratch gate (OOM at d≥1024 for muon/neuralgrok/looksam mamba).
- SG2 workspace OOM: dec_sg2_ws ~199GB at d=1024, carved UNCONDITIONALLY.
- CODEBASE_AUDIT_FINDINGS P0: has_kernels() returns False on healthy _ops.so (probes removed symbols);
  verify_all.py/profile.py/etc. DOA (drive deleted eager .step()); SG11 mu-path regression.
- compile.py P0 fake-green holes (fp64 oracle not in pick_winner; IL=4 non-determinism open by default).
- Phase-2 distributed NOT fully bring-up-ready on GPU (PP patch: REFUTED re bit-rot; but GPU-unrun).

### phase0b/sm90_wgmma_substrate.md

TWO distinct WGMMA paths (must NOT conflate):
1. Hand-rolled in-kernel ss-WGMMA (wgmma.cuh + tile_pipeline.cuh + warp_specialize.cuh). Uses cp.async,
   NOT TMA. This is the L3 persistent megakernel path.
2. CUTLASS host-launched collective GEMM (mma.cuh). Where TMA actually lives (cuTensorMapEncode).
   HOST-LAUNCHED and device-NON-callable. EXPLICITLY REJECTED for L3 persistent-megakernel path.

wgmma.cuh:18: "This header is the only WGMMA path inside the kernel."

cp.async ring in tile_pipeline.cuh: uses cp.async.cg.16 NOT TMA. Phase-1 staging is cp.async; TMA
is explicitly phase-2.

SG_TUNED_TILE_N default 128; valid: {8,16,32,64,96,128} — N=256 is ILLEGAL (static_assert at wgmma.cuh:578,
confirmed by tune11_out search-space bug).

---

## DISCREPANCY ANALYSIS vs CLAIMED CURRENT STATE

The CLAIMED current state (from RESUME.md/PROGRESS.md/SESSION_CONTEXT.md) has these specific claims:

| Claim | Evidence | Verdict |
|-------|----------|---------|
| "DONE+validated: CuTe-atom GEMM engine (bit-identical, SG_TUNED_GEMM_ENGINE)" | Not directly evidenced in these docs; tune11_out shows no SG_TUNED_GEMM_ENGINE in search space or macros. This term may refer to the hand-rolled ss-wgmma engine (wgmma.cuh). | CANNOT CONFIRM this specific term |
| "3 flagship models LAUNCH" | Confirmed: decoder/vit/mamba3 all wired, wiring_check.json 33/33 | CONFIRMED |
| "full TP; cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs" | .phase2/REPORT.md: NVSHMEM NOT INSTALLED; TP authored against LoopbackTransport; GPU-UNRUN. This is a CONTRADICTION with the claimed state. | CONTRADICTED — .phase2 docs say NVSHMEM not installed, GPU tests not run |
| "resource planner" | parallel_config.cuh ParConfig<DP,TP,PP,ZeRO> with static_asserts; PLANNING_INPUT mentions it. Exists as static compile-time derivation, not runtime. | PARTIAL — exists as static config, adaptivity is compile-time not runtime |
| "dead-code cleanup (removed 8.09M lines)" | Not quantified in these docs; Phase7 deleted 16 files. Plausible for historical restructuring. | UNVERIFIED specific number |
| "roofline deliverable" | results_chr/ROOFLINE.md + .perf/phase1_status_audit confirm decoder 6.48%, with honest caveats | CONFIRMED |
| "11-opt decoder ranking (overfit placeholder)" | grokking_race_v2.py results + RETUNE_STATUS confirm race data exists | CONFIRMED |
| "#1 REMAINING: TP data-path fix — A per-rank weight-shard offset, B 25-heads-not-%8 attention, C resulting IMA; A+B fixed in phase6/tp_datapath_fix_WIP.patch" | Not directly evidenced in phase0 digests. The .phase2 content describes LoopbackTransport (no real NVSHMEM), so the claim of "cross-GPU NVSHMEM validated" is already contradicted. | NOT CONFIRMED — the .phase2 evidence predates this claim |
| "HANDOFF.md is from EARLIER campaign (2026-06-12, branch claude/h100-audit-maximal)" | Confirmed — HANDOFF.md dated 2026-06-12, branch claude/h100-audit-maximal. | CONFIRMED |

---

## KEY FACTS SUMMARY (with file:line cites)

1. **All 33 cells in _FUSED_L3_REAL and _L3_WGMMA_CELLS** (dispatch.py:630-962, :1310-1491). Stale docstring says "only (transformer_decoder, adamw)" — code is authoritative.

2. **fused_train_step default gemm_impl="wgmma"** (dispatch.py:1514). Scalar engine REMOVED. Any uncovered cell HARD-FAILS, never silent fallback.

3. **Decoder roofline: 618.5ms / 64.0 TF/s / 6.48%** at d=2048 B=16384 (phase1_status_audit.md). Achieved via two compound wins: dW contiguous-staging +2.05x + fwd/dX deeper cp.async ring +1.49x = ~3.05x cumulative.

4. **HARDWARE_VALIDATION.md frozen at 2026-06-09** (L1 optimizer-only audit; all 99 cells still show 🟡 despite L3-TC campaign completing on real silicon). Doc-drift, not code bug.

5. **tune11_out search-space bug**: SG_TUNED_TILE_N=256 sampled but illegal (wgmma N≤128). 5/9 trials fail at wgmma.cuh:578. All 3 buildable trials numerical_fail. tuned_config=null. (build_tune11.md)

6. **All TC kernels: 255 regs, 1 CTA/SM by design**. --maxrregcount silently ignored under __launch_bounds__. Spill bytes are the pressure metric. (regpressure/REPORT.md)

7. **SG11 warmup gate CI-blind** (RG1): run_sg11_warmup_gate is CLI-only; phantom test_sg11_gate_warmup_catches_momentum_bug has NO definition. SG15 has NO warm path at all. (oracle_trust_audit.json, test_coverage_gap_audit.json)

8. **SG2 CSA indexer co-wrong** (oracle_trust_audit.json): sg2_kernel_mirror oracle implements LOW-RANK path (idx_UQ dropped), not canonical full-rank. N>64 fidelity probe report-only (no _ok flag in verdict).

9. **ViT RG3 gap**: SG_TUNED_VIT_P1_SUBTILE_S<64 ON-path NEVER compiled — not in any test/autotuner. The #1 perf lever (+4.02x) ships UNTESTED. (test_coverage_gap_audit.json)

10. **Phase-2 GPU tests AUTHORED but NOT RUN**: NVSHMEM not installed; TP against LoopbackTransport; PP=2 gate requires applying .phase2/patches/0001 manually (SKIPS without). Shard math independently audited clean (PHASE2_REVIEW_FINDINGS.md). (regpressure/.phase2/REPORT.md)

11. **BUG-04 open**: mamba kMbStagedOptScratch un-gated → OOM at d≥1024 for muon/neuralgrok/looksam mamba cells. staged_opt_scratch_gate.diff authored but not applied. (PLANNING_INPUT.md, author_queue1)

12. **Mamba-3 does NOT toy-grok** at d=128 (val/test pinned ~0.01). No valid tuned_configs_mamba.json. (results_chr/results/tuning/RETUNE_STATUS_2026-06-16.md)

13. **Two WGMMA paths**: hand-rolled ss-wgmma (in-kernel, cp.async, this IS the megakernel path) vs CUTLASS host-launched collective (TMA lives here, explicitly REJECTED for persistent megakernel). (phase0b/sm90_wgmma_substrate.md, wgmma.cuh:13-18)

14. **compile.py Level-2 superoptimizer ~70% scaffold**: polyhedral=identity-copy, synth GEMM stub (_MMA_NATIVE_LOADS_WIRED=False), codegen emitter module DOES NOT EXIST (ghost import). (COMPILE_AUDIT.md, SUPEROPTIMIZER_L2_PLAN.md)

15. **Optuna gate-feedback bug**: objective = timing only; numerical parity gate NOT fed back into optuna study. Would promote wrong kernel if tuned_config threshold reached. (build_tune11.md:B.4)

16. **ADAPTIVITY: config derivation is STATIC** (compile-time ParConfig template), NOT runtime workload-fit. The codebase derives parallelism via static parallel_config.cuh template parameters, not a dynamic resource-fit planner that reads hardware at runtime. The "self-adapting" description in the design thesis is aspirational for Phase-2; today it is a compile-time configuration system. (PRIOR_STATE_AND_CHECKLIST.md, .phase2/REPORT.md)

17. **Parallelism is 5D capable (3D base + optional SP + EP) but current deployment is 3D** (DP×TP×PP with ZeRO-3 orthogonal). SP axis must be a LOUD static_assert error (test_parallel_instantiation.py). EP is planned for MoE but not activated. (doc_plan2, t_rest)

18. **Optimizer tails via _opt_scalars_from()**: discriminated by if/elif chain on param_group attributes (dispatch.py:1029-1267). Order matters. SG2 dispatches via dedicated ops.sg2_fused_step with 26 weight tensors + 6 per-tensor scalar arrays. The chain is correct and internally consistent (§11 of dispatch.md).

19. **res_tune_base.txt ptxas evidence**: All 11 OptId instantiations of fused_decoder/mamba/vit_megakernel_tc show REG:255, STACK 672-2256B (optimizer-dependent, SAM cells higher), SHARED 10724-44852B (model-dependent). sg2_meta_optimizer_megakernel has REG:128, STACK:1056, SHARED:37140. (res_tune_base.txt)

20. **_KERNEL_PROBE_NAMES = ("fused_step", "sg2_fused_step")** (dispatch.py:545-548). Only two megakernel entries shipped. Every eager per-op binding removed. has_kernels() probes ONLY these two. CODEBASE_AUDIT: has_kernels() returns False on healthy _ops.so (probes 4 removed symbols) — this is a BUG if has_kernels() still probes removed names. (PRIOR_STATE_AND_CHECKLIST.md)

---

## OPEN ITEMS (BUGS, TODOS, BLOCKERS)

1. **BUG-04** (OPEN, BLOCKER): mamba staged-opt scratch un-gated → OOM at d≥1024 for muon/neuralgrok/looksam mamba cells. Fix authored (staged_opt_scratch_gate.diff), not applied.

2. **SG2 workspace OOM** (OPEN): dec_sg2_ws ~199GB at d=1024, carved unconditionally (even for adamw). Blocks decoder_bench at HEAD.

3. **SG11 cold-gate CI-blind** (OPEN, HIGH): warmup gate not pytest-collected; phantom test definition. SG15 has NO warm path at all. Fix authored (gate_hardening.diff).

4. **SG2 CSA indexer co-wrong** (OPEN, HIGH): oracle validates low-rank path, not canonical full-rank. N>64 fidelity probe report-only. Fix in gate_hardening.diff.

5. **RG3: ViT S<64 subtile untested** (OPEN, HIGH): #1 perf lever (+4.02x) ships without gate.

6. **tune11 search-space bug** (OPEN): TILE_N=256 in tuner domain but illegal. Causes 5/9 trial failures.

7. **tune11 gate-feedback bug** (OPEN): optuna objective = timing only; numerical gate decoupled.

8. **compile.py superoptimizer scaffold** (OPEN, non-blocking): 70% built; polyhedral/synth stubs don't work. compile_keying_bug.diff fixes the ghost-import emitter source-key bug.

9. **Phase-2 GPU bring-up** (OPEN, BLOCKED): NVSHMEM not installed; all GPU tests unrun; TP loopback validated only on single-GPU LoopbackTransport.

10. **HARDWARE_VALIDATION.md stale** (OPEN, doc-only): never updated to reflect 33/33 L3-TC campaign results.

11. **Mamba-3 not wired into production megakernel** (OPEN): "CUDA megakernel untouched" per MAMBA3_REFERENCE. Current mamba megakernel = Mamba-1 structure.

12. **has_kernels() returns False** (OPEN, BUG): probes removed symbols per CODEBASE_AUDIT. Breaks _HAS_CUDA and verify_all.py/etc. which are reportedly DOA.

13. **P3 opt_tail 5.9% (38ms/618ms)** (OPEN, perf): not perf-maxed; autotune FAILED (zero winners).

14. **ViT confirm-build DEFERRED**: ViT B1 sub-tile S=8 BAKED (b0d41f8, +4.02x) but "final confirm-build was DEFERRED (GPU busy)" — SESSION_STATE.md:26-28.

---

## SELF-CONSISTENCY CHECKS ON CLAIMED STATE

The claimed state says "cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs" but:
- .phase2/REPORT.md explicitly states "NVSHMEM NOT INSTALLED (verified find/pip/ldconfig)"
- Every GPU test is "AUTHORED + nvcc-compiled but NOT RUN"
- TP validation is via LoopbackTransport (single-GPU simulation), not real NVSHMEM

This is a direct DISCREPANCY between the claimed state and the phase0b/phase0c documentation evidence.

The claimed state says "TP data-path fix in phase6/tp_datapath_fix_WIP.patch (ungated)" but the phase0
docs describe a system where TP is not yet silicon-validated (authored, CPU-tested only). This is
plausibly a LATER development after the phase0 docs were written, but cannot be confirmed from the
phase0 digests alone.

The claimed "resource planner" is a compile-time ParConfig template, not a runtime workload-fit
planner. The design thesis's "self-adapting" language is aspirational; today's implementation is
static/config-derived at compile time.
