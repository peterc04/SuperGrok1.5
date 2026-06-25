# Digest: Integration Docs, Performance Analysis, Hardware Validation
## Assigned files: HARDWARE_VALIDATION.md, PHASE1_CAMPAIGN.md, INTEGRATION-MAMBA.md, INTEGRATION-NOTES.md, INTEGRATION-OPTSTAGES.md, INTEGRATION-VIT.md, MAMBA3_REFERENCE.md, OPTIMIZATION_LEDGER.md, PERF_ANALYSIS.md, AUTOTUNE_LINKAGE.md, BUILD_AND_VALIDATE.md, COMPILE_AUDIT.md, PLANNING_INPUT.md, HANDOFF.md

---

## CRITICAL FLAG: HANDOFF.md is from the OLDER campaign

HANDOFF.md (41 lines) explicitly states:
- "Updated 2026-06-12 ~22:05Z"
- Branch: **`claude/h100-audit-maximal`** (NOT the current `claude/custom-optimizer-analysis-HFYhg`)
- References `.regpressure/` patch series and `.phase2/` directory (not present in the current workspace listing)
- Commits referenced: 642e360, ab8c313, 821fee5 — different from HEAD e69df73 claimed in current state
- Roofline: "mean 1.15%/median 1.29%; decoder d=1024 2.15×, production 3.28× vs two nights ago"
- Resume queue: GPU-gate `.regpressure/0001..0005` series, Phase-2 GPU runbook, bounded tuner sweep, vit/mamba PP/TP twins, SG2 workspace redesign — different from current priorities

**Discrepancy**: The current RESUME.md/SESSION_CONTEXT.md claim TP was validated on 8 GPUs. HANDOFF.md says "8× SIGNAL: when 1-5 done = Phase 1 solid + parallelism 1-GPU-validated → PushNotification owner 'provision 8xH100 now'" — meaning at 2026-06-12 the 8× wasn't provisioned yet. This validates that HANDOFF.md is from an entirely different campaign than the current state.

---

## HARDWARE_VALIDATION.md (1876 lines) — Key Findings

### Status Legend
- 🟡 = implemented + compile-verified only, NOT hardware-validated
- ✅ = bit-level reference-checked + profiled on real target accelerator

### Cell Status Matrix (§1)
ALL 99 cells (33 model×optimizer × 3 archs) are **🟡 (not silicon-validated)** for the L3 fused megakernels.

**Silicon-validated (2026-06-09, real H100 80GB HBM3, CUDA 12.4, torch 2.4.1):**
- Build/link/import/run: confirmed working
- Per-op fused sm_90 optimizer kernels: **11/0 numeric parity pass, 11/0 maximality pass**
- 8/11 grokking race (Muon and Prodigy fastest; 3 SuperGrok DNFs are research-owned meta-net dynamics)
- No wgmma serialization (C7509=0), no live register spills in production cells

**Important boundary**: The L3 fused model×optimizer megacells (`csrc/fused/sm_90/mega_*.cu`, the 99-cell matrix) are **compile-verified only**. The race uses the eager model + fused-optimizer (L1) path. `dispatch.has_fused` is empty by design for the race.

**Note on the race path**: "dispatch.has_fused is empty by design" means the 33 sm_90 L3 megacells are NOT runtime/numeric-validated — they all stay 🟡.

### Verified SASS emission (Stage 1C, decoder + ViT)
- decoder.o: **64 HGMMA + 50 UTMALDG** (both confirmed)
- vit.o: **64 HGMMA + 50 UTMALDG** (both confirmed)
- "wgmma" literal = 0 in SASS (PTX mnemonic; SASS = HGMMA)

### Stage 1A — SG2 bilevel backward
- 24 weight-grad buffers implemented with full analytic adjoint
- 20 buffers are 🟢 (full analytic, expected bit-parity on device); 4 are 🟡 (csa_compress_w, csa_idx_DQ/UQ/K)
- Two residual bugs found by line-by-line review:
  - **GRU-gate recompute fallback drops biases** (sm90:1684/1687/1691, gfx942:875/878/882) — inexact if empty gates passed
  - **Output-buffer zero-init is a caller contract with no guard** — bindings do NOT zero 24 `d_*` buffers; harness must `zero_()` before each call
- Hardware validation: DEFERRED (no GPU in environment)

### Stage 2 — L2 persistence
- Wired into all 11 optimizers via `prim::L2PersistScope` (RAII, `cudaStreamSetAttribute` + `cudaAccessPolicyWindow`)
- Gated by `ENABLE_L2_PERSIST` + runtime size check
- Hardware checks: 🟡 ALL DEFERRED

### Stages 3.1/3.2 — PTX maximization
- `redux.sync.add.u32` for integer warp reductions — PTX verified ("REDUX.SUM UR6, R2" in SASS via standalone probe)
- `cp.async.{cg,ca}.shared.global` for attention softmax and CSA/HCA staging
- Depth knob `SG_TUNED_ASYNC_DEPTH` confirmed live (distinct SASS per depth 1/2/4)
- Hardware checks: 🟡 ALL DEFERRED

### Stages 4.1/4.2 — TMA descriptor cache, DSMEM cross-CTA reductions
- TMA descriptor cache: `Sm90GemmCache<Gemm>`, 16-slot direct-mapped, FNV-1a key, correct eviction
- DSMEM: `cluster_reduce_sum_f32_dsmem` (real Hopper DSMEM tree). Wired only for Prodigy r/s sums; others (LookSAM, Muon, attention softmax, LayerNorm) are not applicable
- `ENABLE_DSMEM_REDUCE` defaults to OFF until on-silicon checks pass
- Hardware checks: 🟡 ALL DEFERRED

### Stage 5 — AMD-native gfx942
- Mamba3, attention, decoder, ViT, SG2, all 11 optimizers: two-pass HOST/DEVICE pattern implemented
- Each file has `AMDGCN_OK` compile verification from `amdgcn_check.sh`
- Real MFMA (16×16×16 bf16), DPP reductions, LDS handoff scans written
- Live `hipLaunchKernelGGL` wiring DEFERRED (no hipcc/MI300X in environment; TU migration `.hip.cpp → .hip` pending)
- Hardware checks: 🟡 ALL DEFERRED

### gfx942 drift-closure pass
Three semantic bugs ported from sm_90 to gfx942 ATen launchers:
1. **Muon weight decay INVERTED** (was retaining ~lr·wd instead of ~(1−lr·wd)) — fixed by `param.mul_(decay_factor)`
2. **grokadamw Q3 per-block scale index overrun** — floor→ceil division fix
3. **SG2 MoE dynamic-expert backward nondeterminism** — default now deterministic one-hot contraction (vs `index_add_`)
All fixes CPU-only; on-silicon validation: 🟡

### Stage 6 — 99 megakernels + solver
- Scheduler/barrier substrate: task-queue atomicAdd, work-stealing, SM pinning via `%smid`, hand-built GridBarrier with sense-reversing generation counter
- Demo megakernel: `l3_megakernel<Model, Optimizer>` with 3 stages (fwd → barrier → bwd → barrier → opt)
- sm_90 forward stage uses §3.4 warp-spec (elect/mbarrier/setmaxnreg)
- 3 wired cells: (mamba3, adamw), (transformer_decoder, lion), (vit, supergrok15)
- Generator manifest: 53 L3 / 46 L1 / 0 infeasible
- Autotuner `_resolve_sources` has a latent bug: generated `mega_*.cu` TUs aren't globbed → `e2e_smoke` path undefined for `mega_vit_neuralgrok` etc.
- Hardware checks: 🟡 ALL DEFERRED

### Stage 7 — Distributed
- `grokking_optimizers/distributed.py`: `ParallelConfig`, `DistributedContext`, `ZeRO3Sharder`
- `grokking_optimizers/megakernel_engine.py`: `MegakernelOptimizer`, `FusedBackwardHook`
- Import-clean; CPU-safe tests pass; GPU/distributed runtime: 🟡 DEFERRED

---

## PHASE1_CAMPAIGN.md (614 lines) — Campaign Plan and Status

### Project Phases
- Phase 1: single-GPU foundation, all 33 cells fp64-validated, roofline-converged, portable autotuner, trained models
- Phase 2: 4D + ZeRO-3 multi-GPU (8×H100)
- No formal "Phase 3"

### Flagship Model Sizes (owner-locked 2026-06-16)
| Model | Arch | Config | Params |
|---|---|---|---|
| decoder | GPT-2 XL | d=1600, L=48, h=25 | ~1.5B |
| vit | ViT-G/14 | d=1664, L=48, h=16, MLP=8192 | ~1.8B |
| mamba | Mamba-3 | d=2048, L=24, state=128, head_dim=64, d_ff=4096 | **1.528B** |

Note: These are the *flagship* configs. The grokking science RACE stays at toy config (d=128).

### Mamba-3 Upgrade Status (§2)
- Phase 1 (reference model + oracle): ☑ DONE+validated
  - `grokking_optimizers/mamba3_block.py` + `tests/hw/mamba3_oracle.py`
  - Oracle PASS: fp64 finite + 35 params differentiable + fp32≈fp64 (1.25e-6) + FD≈autograd (4.8e-8)
  - 1.473B at d=2048/L=24/state=128
- Phases 2-4 (megakernel, re-gate, HIP/TPU): ☐ PENDING

### Roofline Baselines (3-seed median, before optimization ratchet)
| cell | d | B | median ms | TF/s | % roofline |
|---|---|---|---|---|---|
| adamw/decoder | 2048 | 4096 | 515.4 | 19.2 | 1.94% |
| adamw/vit | 2048 | 1024 | 5759.2 | 1.83 | 0.185% |
| adamw/mamba | 128 | 4096 | 221.6 | 0.30 | 0.03% |

### Compile.py Optimization Track (5 rounds)
- NARROW loop: 9 KEEP / 0 REVERT / 20 SKIP — DRY WELL
- BROAD loop: 2 KEEP (compile-01 single-cell scoping 2.55× build speedup, compile-10 memoize), 2 DEFER, 5 neutral/SKIP → STOP (3-consecutive criterion)
- Self-test held at 236/6 (6 pre-existing drift guards)

### Kernel Track Findings
- Decoder: split-K 4→2 KEEP (−2.5%); IL=4 REJECT (A/A/A determinism fail at d=128 ragged atoms); vec4 NEUTRAL (<1% of step); TILE_N=64 SKIP (+51%); TILE_M=256 SKIP (+27.7%) → STOP
- ViT: split-K=2 NEUTRAL (within ±1.7% noise); bottleneck is NON-GEMM, not dW
- Mamba: can't run at d=2048 — smem-bound, max ~d=142 under H100 227 KB cap

### #11 Autotuner vs nvcc Result
| Variant | flags | split-K | median ms | TF/s | % roofline |
|---|---|---|---|---|---|
| A — vanilla nvcc | `-O3 --use_fast_math -arch=sm_90a` | 4 | 500.49 | 19.79 | 2.00% |
| B — compile.py default | augmented ptxas flags | 4 | 502.63 | 19.70 | 1.99% |
| C — compile.py tuned | augmented + split-K=2 | 2 | 492.14 | 20.12 | 2.03% |

A→B = +0.43% (neutral; ptxas flags inert at this megakernel); B→C = −2.09% (tuned split-K); A→C = −1.67% total.

### Phase-1 Checklist Status
- ☐ Mamba-3 trained model live + 11 mamba cells re-gated (3 seeds)
- ☐ Optimization ratchet complete (both tracks; stop criteria hit)
- ☐ All 33 cells parity-clean (fp64 gate) on seeds {42,7,123}
- ☐ Roofline-converged at d=2048
- ☐ Autotuner validated at scale (#11 done)
- ☐ Pre-race `tuned_configs.json` generated + committed
- ☐ Pre-existing #10-aftermath self-test drift-guards fixed
- ☐ Everything persisted + hand-off doc complete

---

## INTEGRATION-MAMBA.md (438 lines)

### Purpose
Contract for wiring the real Mamba-3 forward+backward as persistent megakernel stages. Math+structure proven on CPU; seam not yet wired.

### What Ships (CPU-validated)
| file | status |
|---|---|
| `tests/hw/mamba_oracle.py` | ✅ fp64, loss diff 8.9e-16, worst grad rel 1.3e-15 (28/28 tensors) |
| `tests/hw/mamba_kernel_mirror.py` | ✅ matches oracle, worst grad rel <1e-6 |
| `csrc/fused/sm_90/model_stage_mamba3.cuh` | source-complete, not built into project |
| `csrc/fused/sm_90/mamba3_layout.cuh` | source-complete, 28 tensors, 259425 total |
| `tests/hw/test_mamba_megakernel.py` | no-GPU gates PASS; GPU gates SKIP pending seam |

### Architecture (pinned toy config: d=128, nl=2)
- 28 parameter tensors, 259,425 total params
- seq=8, head reads LAST position
- int32 input `[B,8]` + targets `[B]`
- Critical: `kMambaSmemBytes = 145,124 bytes` (≈141.72 KB) — requires dynamic smem opt-in

### Key Integration Requirements
- **MUST** declare dynamic smem (`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, dyn_smem)`)
- batch-parallel (NOT work-steal queue — fp32 sums non-associative)
- `n_tasks=28` (kMambaNumTensors) for reduce + optimizer phases
- Workspace: `n_ctas × 259425` fp32 (~137 MB at 132 SMs)
- Only `mamba × adamw` on sm_90; other optimizers/models/gfx942 stay eager (L1)

---

## INTEGRATION-VIT.md (381 lines)

### Purpose
Contract for wiring real ViT forward+backward as persistent megakernel stages. Source-complete, not built.

### What Ships (CPU-validated)
| file | status |
|---|---|
| `tests/hw/vit_oracle.py` | ✅ fp64, loss rel 0, worst grad rel 1.0e-15 (32/32) |
| `tests/hw/vit_kernel_mirror.py` | ✅ loss rel 2.0e-16, worst grad rel 9.9e-16 |
| `csrc/fused/sm_90/model_stage_vit.cuh` | source-complete, not built |
| `csrc/fused/sm_90/fused_vit_megakernel.cuh` | source-complete, incl. dynamic-smem opt-in |
| `csrc/fused/sm_90/vit_layout.cuh` | source-complete, 32 tensors, 418017 total |

### Architecture (pinned toy config: d=128, h=4, nl=2)
- 32 parameter tensors, 418,017 total params
- `VitSampleSmem = 188,080 bytes (≈183.67 KB)` — dynamic smem required
- INPUT is FLOAT image patches `[B,16,49]`, NOT int tokens
- NO causal mask (FULL attention)
- Head reads CLS position 0

### ABI
- Input `[B*16*49 + B]` float: patches + targets bit-cast to float slots
- Workspace: ~221 MB at 132 SMs
- Only `vit × adamw` on sm_90

---

## INTEGRATION-NOTES.md (427 lines)

### Purpose
Contract for `csrc/fused/sm_90/opt_stage_supergrok2.cuh` — the full SG2 meta-net as in-kernel optimizer stages. Replaces ~15-20 launches/tensor with 1 argsort prep + 1 persistent kernel.

### Key Facts
- `launch_sg2_meta_optimizer_tail<Dims, ParamT, GradT>` runs entire SG2 step for ALL params in ONE persistent kernel
- The argsort is the ONE explicitly-retained pre-kernel step (honesty rail #5)
- fp64 structural mirror (`tests/hw/sg2_kernel_mirror.py`): CPU mirror vs oracle → machine-epsilon agreement (~1e-16) across N∈{5,17,64,200} and 200-step trajectory — PASSES
- GPU tests `pytest.skip` pending binding creation
- Sort tie-handling is subtle: must reproduce plain `torch.sort` (NOT stable) to stay bit-aligned with parity oracle

### ABI Gap Flagged
`sharpness` is model-coupled (from SAM 2nd backward). `FusedOptState` has NO `sharpness` field and NO phi-weight fields. Two integration options: (1) extend ABI, (2) side-channel pointers. Until wired, SG11/SG15 CANNOT run fully in-kernel.

---

## INTEGRATION-OPTSTAGES.md (365 lines)

### Purpose
Contract for `csrc/fused/sm_90/opt_stages_precompute.cuh` — the per-step precompute stages needed for 9 non-trivial optimizers.

### Verdicts
| optimizer | verdict |
|---|---|
| grokfast, grokadamw, neuralgrok | NOTHING-NEEDED (EMA/psi fused in apply) |
| looksam | MODEL-COUPLED (st.sam_dir from model stage 2nd bwd) |
| prodigy | STAGED (cross-all-tensors d reduction) |
| muon | STAGED (Newton-Schulz per-matrix grid-cooperative) |
| supergrok11 | STAGED (per-tensor mu + cosine gate) |
| supergrok15 | STAGED (per-tensor mu; gate = host scalar) |
| supergrok2 | SKIP (sibling-owned) |

### Validation Status
- CPU fp64 mirrors: `tests/hw/test_opt_stages.py` — **9/9 pass**
- CUDA compile / GPU run: DEFERRED

### Notable Finding: SG11 Gate Formula
`compute_cosine_gate_fused` (`supergrok11_sm90.cuh:280-285`) computes `clamp(<sg,mu> / sqrt(‖sg‖²·‖mu‖² + 1e-12), 0, 1)` — this IGNORES `gate_temp` despite `bindings.cpp:1276` comment claiming `sigmoid(t·cos)`. The function does a bare clamp. The comment is wrong; the code is authoritative.

---

## MAMBA3_REFERENCE.md (544 lines)

### Purpose
Reference implementation of full Llama-style Mamba-3 (arXiv 2603.15569, ICLR 2026).

### Status
- Code: `grokking_optimizers/mamba3_block.py` (`Mamba3Layer`, `SwiGLU_MLP`, `Mamba3Block`, `Mamba3Model`, `RMSNorm`)
- Oracle: `tests/hw/mamba3_oracle.py` (fp64 verified, ≤4e-15 per param + input)
- NOT wired into production or CUDA megakernel
- 1.5277B at d=2048/nl=24/state=128 (paper config)

### Key Architecture Differences vs Mamba-1
| Aspect | Mamba-1 | Mamba-3 |
|---|---|---|
| Short conv1d | depthwise k=3 NON-causal | DROPPED |
| SSM-input SiLU | after conv | DROPPED |
| `dt/A/lambda` scope | per-channel | **per head** |
| Discretization | exponential-Euler (2-term) | **exponential-trapezoidal (3-term)** |
| State | real diagonal | **complex → 2×2 real rotations** |
| lambda (gate) | n/a | `sigma(u_t)`, data-dependent |
| BCNorm | none | RMSNorm on B,C |
| B,C biases | none | all-ones-init channel-wise |

### Architecture: Llama-style (24 mixer + 24 SwiGLU blocks = "nl=24" counts MIXER blocks)
Alternating Mamba-3 mixer sub-block and SwiGLU MLP sub-block, both pre-norm + residual.
SwiGLU: `down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))`, inner dim d_ff=4096.

### Open Ambiguities (for CUDA kernel phase)
- (a) SiLU on SSM input: paper says obviate conv "and its accompanying activation" → lean DROP (matches current impl)
- (b) rotation dt: per-head dt (implemented, matches Mamba-2/3)

---

## OPTIMIZATION_LEDGER.md (168 lines)

### Comprehensive Track Summary

**Track A (static patches .regpressure/0001-0005)**
- 0001 bf16-prestage: **KEEP** (enables cp.async ring 0005)
- 0002 decoder SAM-scoped outline: **KEEP** (reduced register pressure SAM cells)
- 0003 ViT SAM-scoped outline: **REVERT** (+5% slower)
- 0004 mamba scope-noinline: **KEEP** (−4.4% single-pass)
- 0005 decoder cp.async ring: **KEEP** (−14.2% across ALL decoder cells)

**Track B (compile.py)**
- NARROW: 9 KEEP / 0 REVERT, dry-well STOP
- BROAD: 2 KEEP (single-cell scoping 2.55×, `_owns_extension_module_tu` memoize ~11×), 5 neutral/SKIP → STOP

**Track C (kernel roofline d=2048)**
- Split-K 4→2: **KEEP −2.5%** (commit a625227)
- IL=4: **REJECT** (A/A/A determinism fail — ragged atoms at d=128)
- vec4 AdamW: **NEUTRAL** (P3 tail <1% of step)
- TILE_N=64: **SKIP +51%**, TILE_M=256: **SKIP +27.7%**

**Track D (#11 validation measurement)**
- A→C = −1.67% total (the autotuner's split-K knob, not ptxas flags)

**Track E (structural)**
- **P0 decoder dW pipelined GEMM: REVERT** (−11.6%; staging-bound not drain-bound)
- **dW contiguous-layout staging: KEEP +2.05×** (920.7 ms vs 1889.8 ms; roofline 2.08%→4.35%)
- **Decoder fwd/dX deeper cp.async ring STAGES=4: KEEP +1.49×** (618.5 ms; 6.475% roofline)
- **ViT dW staging twin: REVERT** (IMA, root-caused as pointer-carve gate missing; after fix only +4.5% and fp64 gate RED on SG2/vit)
- **Mamba M0 (wgmma projections): DEFER** (scan-dominated at d=128; blocked by scan data-flow rewrite)

**CUMULATIVE decoder**: 1889 → 618 ms ≈ **3.05×** from session start; **6.475% of 989 TF/s bf16 roofline**

**META-LESSON**: Decoder = dW/fwd-GEMM-bound (the 2× lived there). ViT = non-GEMM-surface-bound (dW only 3% of step). Mamba = scan-bound. The decoder lever does NOT generalize to ViT or Mamba.

**ViT Bottleneck Map (profiled 2026-06-17, d=2048/B=1024)**:
- B1 barrier 51.2% (load imbalance — dominant)
- P1_bwd 23.7%
- P1_fwd 21.7%
- P2_dW_GEMM 3.0% (vindicates meta-lesson)
- grad_asm 0.1%, opt_tail 0.3%

---

## PERF_ANALYSIS.md (69 lines)

**STATUS: SUPERSEDED IN PART (2026-06-17)**

The document explicitly flags itself as partially superseded. The original framing ("the ~2% roofline is a hardware/structural ceiling"; "P0 pipelined GEMM is the built-but-unused #1 lever") is out of date:
- P0 was REVERTED (staging-bound, not drain-bound; −11.6%)
- dW contiguous-staging redirect is the actual win (+2.05×)
- The ~2% was a STAGING artifact, not a hardware ceiling

**Current bottleneck map (post wins)**:
- Decoder: fwd/dX 56.5% (drain-bound) + B1 barrier 19%
- ViT: B1-barrier load-imbalance 51%
- Mamba: scan-dominated (M0 deferred)

**Remaining levers in the document** (still valid even if framing changed):
- P1: fuse inter-GEMM fp32 epilogues
- P2: ViT B1 grid-barrier load imbalance
- M0: Mamba wgmma projections + output-stationary dW (94% of FLOPs at 1.5B)
- M1: Mamba smem/HBM restructure to fit d=2048
- M2: chunked/associative parallel scan (HIGH RISK — FP-bit-different)

---

## AUTOTUNE_LINKAGE.md (238 lines)

### Linkage Architecture
```
compile.py build_jit --write-on-win --> grokking_optimizers/_kernel_tuned.json
                                              |
pip install -e . / ./build.sh                v
  setup.py TunedBuildExtension --reads JSON, injects per-TU nvcc flags --> _ops*.so
```

### Safe Tuned Dims (per-TU)
- `-DSG_TUNED_BLOCK_SIZE`, `-DSG_TUNED_VEC_WIDTH`, `-DSG_TUNED_UNROLL`, `-DSG_TUNED_ASYNC_DEPTH`, `--maxrregcount=N`
- Applied to `launch_<opt>.cu` and `mega_<model>_<opt>.cu` TUs
- Bindings, model-only TUs, common headers get NO per-optimizer flags

### Historical Issue (now fixed)
- "The autotuner was decorative" — the winner only reached a header that nothing on the install path included
- Now fixed: `TunedBuildExtension` reads JSON and injects flags into `build.ninja`

### Key Gotchas
- One winner per `(arch, optimizer)` — no model dimension; tuning model B overwrites model A's winner
- JSON absence → byte-identical to before (in-header defaults)
- `./build.sh --autotune` is GONE (it called a nonexistent `autotune/tune.py`)
- Secondary header `csrc/algorithms/tuned_configs.h` is back-compat consumer; canonical handoff is `_kernel_tuned.json`

---

## BUILD_AND_VALIDATE.md (453 lines)

### Key Discrepancy: L3 vs L1 in the Race
**The L3 fused megakernel does NOT compute real model fwd/bwd** for most cells. The model stages in `csrc/fused/sm_90/model_stages.cuh` are a **SURROGATE** (`acts[gi] = GELU(params[gi] + input[gi])`), not real attention/matmul/CE.

Therefore: lever (a) is wired as **L1 optimizer tail** (keep real PyTorch fwd+bwd, replace `optimizer.step()` with L1 fused megakernel `opt_only=True`).

### PHASE 1 (TRUE L3 for decoder × adamw)
The decoder × adamw cell IS the real megakernel:
- `csrc/fused/sm_90/mega_decoder_real_adamw.cu` includes `fused_decoder_megakernel.cuh` with real model math
- Source-complete; implementing agent did NOT build or run GPU work
- Described architecture (toy grokking config): `Transformer(nl=2, d=128, h=4, ntok=99, seq=4)` — 30 tensors, 422,755 total params

### Readiness Whitelist (L1 fused in race)
**`decoder:adamw, vit:adamw, mamba:adamw, decoder:lion, vit:lion, mamba:lion`** (6 cells)

Other 9 optimizers are behind readiness gate because they need precomputed quantities (prodigy's d, grokfast EMA, muon NS-orth, SG11/15 gate, SG2 meta-net smart-grad) that the L1 tail doesn't itself produce. "Wiring those with placeholder scalars would silently degrade the math (the suppression the owner forbids)."

### Bug Fixed in BUILD_AND_VALIDATE.md
AdamW loop was passing `"grokadamw"` (routing plain-AdamW state into the grokfast-EMA cell — wrong math); now corrected to `"adamw"`.

---

## COMPILE_AUDIT.md (181 lines)

### P0 Blockers (4 items — must fix before trusting tuned winners)

1. **fp64 oracle NOT wired into winner path** — `pick_winner` uses same-dtype fp32 self-consistency, never the fp64 ground-truth gate. The IL=4 trap was caught only by the out-of-band hardware gate, NOT the tuner. Fix: inject fp64+A/A/A hook into `pick_winner`.

2. **Polyhedral/CUTLASS/CK winners ship the TEMPLATE while reporting generated origin (fake-green)** — `build_jit`'s final source-swap checks only `== ORIGIN_SYNTH`; polyhedral leaves `final_sources = sources` (template) yet records `origin=polyhedral`. Fix: generalize to any `origin in _VALIDATION_REQUIRED_ORIGINS`.

3. **IL=4 determinism trap OPEN by default** — `pick_winner` only enforces `deterministic` tag when `strict_numerics=True`; the default path lets `non_deterministic` (atomicAdd / IL=4 reduction-order) configs WIN. Fix: reject unconditionally.

4. **Fast-math cache signature drops version-gated flags** — stale `.so` can win post-upgrade. Fix: fold version-gated tuples into `fm_sig`.

### P1 Maximality Issues (5 items)

5. **CLI INVERTS "maximal"** — a plain invocation has PGO, device-PGO, emitter, synth, polyhedral, runtime-specialization, transfer-learning all OFF (7 `store_true` flags override `build()`'s True defaults). Fix: tri-state the flags.

6. **#24 roofline objective ABSENT** — objective is raw `timing_ms`, not %-of-roofline. Fix: add `peak_tf`/`peak_bw` to `ArchEntry`; switch to maximize %-roofline.

7. **#23 tiered spill mgmt PARTIAL** — `_parse_ptxas_v_stderr` parses spill bytes into sidecar but NOTHING reacts (no penalty, no escalation, not cost-model features). Fix: nonzero local-spill = tier breach.

8. **Byte-identical "untuned build" claims are FALSE** — `dec_dw_splitk` first value 4 vs kernel default 1; `vit_dw_splitk` 4 vs 1; `cons_regs` 200 vs 232; `prod_regs` 32 vs 40. Fix: reorder value lists to match kernel defaults.

9. **SASS not in scope as inspection** — `cuobjdump --dump-sass` never run during tuning. (Optional but owner asked for PTX-AND-SASS.)

### Level-2 Superoptimizer (~70% scaffold)
- Polyhedral: `apply_schedule` discards real kernel signature, emits hardcoded identity copy
- Synth/OpGraph: "AdamW" patterns never update m/v (numerically wrong); wrong ABI symbol (always fails validation)
- Native wgmma mainloop: `_MMA_NATIVE_LOADS_WIRED=False` (hard-OFF stub)
- CUTLASS/CK emitters: real but not wired into decoder/vit/mamba megakernel timing path

**Key note**: compile_config module import in `build()` (17473) **does not exist** → `ModuleNotFoundError` swallowed → all TOML config knobs silently ignored on live path.

---

## PLANNING_INPUT.md (55 lines)

### Current State (2026-06-17, post structural wins)

**Roofline (%-of-989-TF/s bf16 peak; NOT formally scored — #24 pending)**:
| model | ms/step | % of peak | bound |
|---|---|---|---|
| decoder d=2048 | 618 | 6.48% | latency / phase-serialization |
| vit d=2048 | ~1434 (post-S=8 sub-tile fix) | ~0.74% | was B1 load-imbalance (fixed) |
| mamba3 d=128 | 222 | 0.03% | scan-dominated (can't scale past ~d=142) |

**Decoder opportunity map (618 ms step)**:
1. P1_fwd 27.6% + P1_bwd 27.3% = ~55% (WGMMA-bound; lever = GEMM efficiency)
2. Grid-barriers (B0+B1+B2) ~20% (phase serialization)
3. P2_dW_GEMM 16.5% (staging-bound)
4. P3 optimizer tail 5.9% (UN-AUTOTUNED)
5. Non-GEMM CUDA-core (hidden share)

**Open gate-coverage caveats (NOT confirmed math bugs)**:
- grokadamw/prodigy: multi-step parity gate missing (vit/mamba)
- supergrok11: warm-up gate CLI-only; supergrok15: NO warm-up gate
- supergrok2: CSA oracle co-wrong (HIGH)
- muon/neuralgrok/looksam: BUG-04 (mamba staged-opt scratch un-gated → OOM at d≥1024)

---

## Stage 5 AMD-native (gfx942) — Additional Detail from HARDWARE_VALIDATION.md

### Models (all follow two-pass HOST/DEVICE pattern, all AMDGCN_OK compile-verified)

**mamba3_gfx942.hip.hpp**: Real 16×16×16 bf16 MFMA (`amd::mfma_bf16_16x16x16`) for in_proj/out_proj/x_proj; DPP reductions for RMSNorm fwd+bwd; per-lane sequential + Blelloch work-efficient scan for SSM selective scan. `hipLaunchKernelGGL` wiring DEFERRED (no hipcc/MI300X).

**attention_gfx942.hip.hpp**: `attention_gfx942_fwd_mfma<32,kCausal>` + `_bwd_mfma` — QKᵀ, scale+mask, DPP row softmax, O=PV — all via 16×16×16 MFMA tiles. Softmax max uses `wave_reduce_max_dpp` (same DPP shape as add-butterfly but `fmaxf`). LDS budget: ≤8 KB for grokking shapes, well within 64 KB CDNA3.

**transformer_decoder_gfx942.hip.hpp**: `decoder_gfx942_mfma_gemm` (single MFMA GEMM driver for QKV/out/FFN-up/FFN-down), `decoder_gfx942_attention`, `decoder_gfx942_layernorm_fwd` (two DPP reductions for mean+var), `decoder_gfx942_gelu` (tanh-approx via `__builtin_tanhf`). GELU constants identical to sm_90 decoder.

**vit_gfx942.hip.hpp**: `vit_gfx942_matmul_bias` (single MFMA GEMM for patch-embed/QKV/out/MLP/head), `vit_gfx942_attention` (online softmax with running max via DPP fmaxf), `vit_gfx942_layernorm_fwd`. LDS budget: `kAttnMaxS=240` (62,400 B < 65,536 B); static_assert guards it.

**All 5 gfx942 model files**: AMDGCN_OK; live hipLaunchKernelGGL pending `.hip.cpp → .hip` migration on MI300X.

### gfx942 Optimizer Kernels

**Reduction-bearing (5 files: looksam/muon/prodigy/sg11/sg15)**: Each has a real AMDGCN wave→block→AGENT-atomic reduction kernel (DPP wavefront sum → LDS block tree → `amd::atomic_add_agent_f32` AGENT scope). The APPLY stays on ATen. AMDGCN_OK for all 5.

**Elementwise (5 files: adamw/lion/grokfast/grokadamw/neuralgrok)**: Grid-stride `__global__` kernels. Per-element math copied verbatim from `csrc/algorithms/<opt>.h`. Streaming loads via `amd::streaming_load`. APPLY stays ATen. AMDGCN_OK for all 5.

**supergrok2_gfx942.hip.hpp**: HOST pass (ATen+rocBLAS) unchanged; DEVICE pass adds real MFMA CSA/HCA attention (`sg2_csa_attention_fwd_mfma<4>`, `sg2_hca_attention_fwd_mfma<4>`), PEER product-key routing (`sg2_peer_route_kernel`), GRU gates (`sg2_gru_gate_kernel`). Bilevel adjoint and MoE compaction stay on ATen. AMDGCN_OK.

### gfx942 Drift-Closure Pass — Semantic Bugs Ported from sm_90

Three semantic fixes applied to gfx942 ATen launchers (CPU-only; no on-silicon validation):
1. **Muon weight decay INVERTED** (LIVE BUG): was `param.mul_(1 - decay_factor)` instead of `param.mul_(decay_factor)` — retained ~lr·wd (~2%) instead of ~(1−lr·wd) (~98%) per step. Fixed: `param.mul_(decay_factor)`.
2. **grokadamw Q3 per-block scale index overrun**: floor division → `q_block_size = ceil(numel / num_scales)` + `narrow(0,0,numel)`.
3. **SG2 MoE dynamic-expert backward nondeterminism**: `index_add_` (atomic, nondeterministic) → one-hot contraction default (deterministic); `SG2_ATOMIC_MOE_BWD=1` restores fast path.

---

## Stage 6 — 99 Megakernels + Solver — Additional Detail

### Scheduler/Barrier Substrate (csrc/fused/megakernel_common.cuh + megakernel_common_hip.hip.hpp)

- **Task-queue scheduler**: global atomic `g_next_task`; each persistent CTA pulls via `atomicAdd`; leader broadcasts through shared/LDS.
- **Work-stealing**: NO static partition; idle CTA just calls `next()` again — the `atomicAdd` IS the steal.
- **SM/CU pinning**: `%smid` on sm_90; `HW_ID` via `__builtin_amdgcn_s_getreg` on gfx942. Grid = one CTA/SM.
- **Hand-built GridBarrier**: two global atomics (`arrived`, `generation`) + sense-reversing generation counter. NO cooperative launch. Reusable because fast CTA cannot lap slow one — the consumer waits on generation, not `arrived` reset.
- **sm_90 warp-spec**: elect/mbarrier/setmaxnreg (§3.4 primitives). gfx942 uses 4-wave-interleave ping-pong (no TMA/WGMMA analog on CDNA3).

### Demo Megakernel Structure

`l3_megakernel<Model, Optimizer>`: `forward_stage → GridBarrier → backward_stage → GridBarrier → optimizer_stage`. Forward stage on sm_90 exercises warp-spec (producer warp 0 elect+mbarrier, consumer warps 1–3). Task queue reset inside the kernel at grid barrier (last arriver, all CTAs quiesced — race-free).

### Generator Manifest (`megakernel_codegen.py --emit-all`)
```
tier coverage: L1_OPT_ONLY=46, L3_FWD_BWD_OPT=53
wired into fused_step: 3 cell(s)
```
3 wired cells: (mamba3, adamw), (transformer_decoder, lion), (vit, supergrok15). All others throw "no fused TU; use per-op path".

### Autotuner Bug in _resolve_sources

Generated `mega_*.cu` TUs aren't globbed → `e2e_smoke` path undefined for `mega_vit_neuralgrok` etc. Production `_ops` race path is unaffected; deferred.

---

## Phase 2-4 Additions (from HARDWARE_VALIDATION.md §Phase 2/3/4)

### Stage P2-1: SG2 Bilevel Adjoint Wired Live
`bilevel_step()` dispatches to C++ hand-written VJP (`supergrok2_bilevel_backward`) when extension built with bilevel support. Numerical parity of C++ VJP vs autograd on real hardware: DEFERRED.

### Stage P2-2: 99 Megakernel Cells Emitted
`megakernel_codegen.py --write-all` materializes 33 sm_90 .cu + 33 gfx942 .hip + 33 tpu_v6e .py stubs. `dispatch.cpp::wired_fused_cell` expanded to route all 99. Solver: 53 L3 / 46 L1 / 0 infeasible. On-silicon compile gate DEFERRED.

### Stage P2-4: TPU Parity — pallas_expert_gather Implemented
Real Pallas-tiled gather kernel replaces pure-JAX stub. All TPU optimizer/model kernels now have JAX/Pallas implementations. On-silicon profile DEFERRED.

### Phase 3: Real Component Compositions (sm_90)
All 33 sm_90 cells individually nvcc -c COMPILE_OK. Real `csrc/algorithms/<opt>.h` math (no AdamW fallback) fused in all 11 optimizer tails. gfx942 AMDGCN_OK. TPU 66/66 trace+lower. On-silicon numeric parity: DEFERRED.

### Phase 4: Stage V — Consolidated Verification
Self-test 156/0; ruff clean; nvcc -c COMPILE_OK on model/{decoder,vit,mamba}.cu + WS1 cells; AMDGCN_OK on all touched gfx942 files. Remaining: real accelerator parity for all 99 cells.

---

## PERF_ANALYSIS.md — Superseded Framing

The document explicitly states: **"⚠ SUPERSEDED IN PART (2026-06-17)"**. Its original framing ("the ~2% roofline is a hardware/structural ceiling", "P0 pipelined GEMM is the #1 built-but-unused lever") is OUT OF DATE:
- P0 was REVERTED (staging-bound, −11.6% slower)
- dW contiguous-staging is the actual +2.05× KEEP (proved ~2% was a staging artifact, not a ceiling)
- Post-dW bottleneck map: decoder = fwd/dX 56.5% drain-bound + B1 19%; ViT = B1 load-imbalance 51% (dW only 3%); Mamba = scan-bound

However its lever table remains informative: P1 (fuse inter-GEMM fp32 epilogues, ~1.2-1.6× multiplicative), P2 (ViT B1 imbalance), M0/M1/M2 (Mamba wgmma projections + smem + chunked scan).

---

## Summary of Discrepancies vs CLAIMED State

### HANDOFF.md (confirmed stale/different campaign)
- Branch: `claude/h100-audit-maximal` vs current `claude/custom-optimizer-analysis-HFYhg` ← CONFIRMED DIFFERENT
- Date: 2026-06-12 ← older than current session
- Roofline: 1.15%/1.29% ← superseded by 6.48% (decoder) in PLANNING_INPUT.md
- Infrastructure: `.regpressure/`, `.phase2/` ← not visible in current workspace listing
- Resume queue: completely different items than current priorities

### CLAIMED "NVSHMEM TP all-reduce VALIDATED on 8 GPUs"
- NONE of the docs I read mention NVSHMEM validation on 8 GPUs
- HARDWARE_VALIDATION.md §6 distributed scaling is entirely 🟡 DEFERRED
- Stage 7 distributed tests skip unless `WORLD_SIZE>1` and CUDA device visible
- HANDOFF.md (2026-06-12) says "8× SIGNAL" is pending (not yet done)
- This claim is UNSUBSTANTIATED in these documents — it may be in later docs from a subsequent session not in this assignment

### CLAIMED "11-opt decoder ranking (overfit placeholder)"
- The 8/11 grokking race result IS confirmed by HARDWARE_VALIDATION.md §2
- "overfit placeholder" is described: the 3 SuperGrok DNFs are research-owned meta-net dynamics, not kernel bugs (frozen-meta-net control reduces SuperGrok1.1 to AdamW → groks at step 2,700)
- CONFIRMED

### CLAIMED "roofline deliverable"
- PLANNING_INPUT.md shows decoder 6.48%, ViT 0.74%, Mamba 0.03%
- But note COMPILE_AUDIT.md #6: roofline objective not yet in the autotuner (#24 pending)
- These are derived numbers, not formally scored by the autotuner
- PARTIALLY CONFIRMED (numbers exist, not yet formal)

### CLAIMED "dead-code cleanup (removed 8.09M lines)"
- Not verifiable from these docs; not mentioned in any of the 14 files assigned

### CLAIMED "phase6/tp_datapath_fix_WIP.patch"
- Not mentioned in any of the 14 files assigned; appears to be from a more recent session
