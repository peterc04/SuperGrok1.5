# Task Outputs Digest — task_outputs_all.md
## Source: 140 concatenated final structured reports from session workflows

---

## Overview

The file `/tmp/claude-0/.../task_outputs_all.md` is 16,282 lines containing concatenated structured outputs from ~140 orchestrated workflow agents spanning the full session history. Reports cover distinct campaign phases from the June 2026 session. This digest mines the concrete deliverable results, decisions, and state — flagging contradictions with the claimed current state in RESUME.md/PROGRESS.md.

---

## Phase Summary: What the Agents Actually Did and Found

### 1. R2.3 Decoder TC Fork-B Driver (Task b0bw35x0w — LARGEST output, lines 1–2486)

**Task**: Author the Fork-B decoder TC cell driver — the phase-restructured fwd+bwd+AdamW persistent kernel built on the VALIDATED engine (the perf milestone).

**Actual outcome**: Driver was AUTHORED, DEBUGGED, and VALIDATED. Key journey:

- Wrote tile-batched fwd+bwd driver in `model_stage_decoder_tc.cuh`, replacing the `#error` in `fused_decoder_megakernel.cuh` under `SG_GEMM_IMPL_WGMMA`, plus new TC TU `mega_decoder_real_adamw_tc.cu` + test gates in `test_decoder_tc.py`.
- **Bugs found and fixed during authoring** (all fixed before final validation):
  1. **ff0 dX and in_proj dX had Kin/Nout swapped** — caused buffer overflow (4× into `sc.x1`) that corrupted all downstream adjoints. Fixed by correcting `(kDff, kD)` → `(kD, kDff)` for ff0 dX and `(3*kD, kD)` → `(kD, 3*kD)` for in_proj dX.
  2. **dW double-mbase**: `dectc_dw_run_tile` passed `mbase0=mbase` to `tc_gemm_block_unpipelined` AND added `mbase+m` in the accessor, causing double-offset since engine already adds `mbase` internally. Fixed by removing the extra `mbase+` in srcA/out accessors.
  3. **Per-layer LN cache overwrite**: `qkv`, `ff0pre`, `attn`, `n1x`/`n1i`/`n2x`/`n2i` were single-buffered per CTA — forward layer 1 overwrote layer 0's caches. Backward for layer 0 read layer 1's caches. Fixed by making these per-layer arrays `[kLayers]`.
  4. **work/work2 aliasing in backward**: Fixed by adding a `work2` fp32 buffer to `DecTileScratch`.
  5. **Unused `hnb` variable**: Removed dead code.

**Final validated gate results** (all PASS):
- Loss rel vs bf16-faithful oracle: **2.5e-4** (well within 1e-3 target)
- Grad parity vs bf16-faithful oracle: layer-0 ≈ layer-1 (max weight rel 0.098 < 0.15 tol, max bias 0.070 < 0.08 tol). ISO gate (dW GEMM on kernel's own bf16 acts) = **3.3e-7** (GEMM is bit-exact on own operands; confirms residual is bf16 noise)
- Determinism A/A/A: bit-exact
- nCTA=2 (multi-tile-per-CTA path): worst rel 9.813e-02 < 0.15 ✓
- 50-step trajectory: loss 4.8152 → 0.0281, finite, decreasing ✓
- **Scalar path unchanged**: compiles clean, 19 passed (byte-identical)

**SASS audit** (fresh from rebuilt .so):
- **18× HGMMA.64x128x16.F32.BF16** (real bf16 fp32-acc warpgroup MMAs: 8 fwd + 8 dX + 9 dW, head/CE/attention/LN are scalar fp32 — head M=32<64 minimum wgmma atom)
- **9× HGMMA.64x8x16.F16 (all write RZ)** — dead ptxas scaffolding for wgmma pipeline choreography, no functional effect
- **27× WARPGROUP.ARRIVE/DEPBAR.LE** (fence/wait choreography)
- **0 bytes spill stores, 0 bytes spill loads**
- 251 registers, 1152-byte stack frame, 8628 bytes smem, occ=1

**Matched step-time** (both at nCTA=132, same contention):
- **SCALAR: 12,053 µs/step** vs **TC: 6,690 µs/step** → **TC is 1.8× faster than scalar**
- Note: prior capped run (nCTA=4) showed 23,952 µs which was an artifact of memory pressure, not representative

**DEVATIONS from Fork-B contract** (documented in DESIGN-TC-PIPELINE.md):
- AdamW is NOT fused in the dW epilogue; it runs as a separate P3 phase (3-barrier structure: B0 zero-lnvec, B1 acts-complete, B2 grad-ready). This is a deliberate correctness-first choice.
- Head/CE remains scalar per-sample (M=32 < wgmma m64 minimum; correct per design)
- Engine is unpipelined (C7515 serialization remark; correctness-first tradeoff)
- The `ncta_cap` parameter (not in the original design) was added to let tests run under fleet memory pressure

**Memory environment constraint**: GPU saturated by fleet (~79 GB used), leaving <700 MB free. Launch needs ~53 MB local-memory reservation (1152 B/thread × 132 SM × 256 threads) + workspace. The TC workspace at nCTA=132 is ~222 MB — comparable to the scalar workspace (~223 MB). Runs were only possible during brief fleet idle windows.

**Files delivered** (uncommitted per task constraint):
- `csrc/fused/sm_90/model_stage_decoder_tc.cuh` (279 → 1134 lines)
- `csrc/fused/sm_90/fused_decoder_megakernel.cuh` (#error replaced)
- `csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu` (NEW, 188 lines)
- `tests/hw/test_decoder_tc.py` (760 lines, with PART2 full-cell gates + bf16-faithful oracle + ISO gate)

**DISCREPANCY vs claimed state**: The claimed state says "A+B fixed in phase6/tp_datapath_fix_WIP.patch (ungated), bug C unconfirmed" — but that refers to the TP data-path fix for the 8-GPU run, which is a DIFFERENT task. The R2.3 decoder TC driver bugs A/B/C (per-layer cache, double-mbase dW, Kin/Nout swap) are ALL fixed in the above.

---

### 2. 8-GPU North-Star TP8 Run (Task w78epgggr — lines 9596–9750)

**Task**: Run ONE flagship 1.5B decoder across all 8 H100s via TP8 + in-kernel device-NVSHMEM all-reduce.

**Actual outcome**: **RUNTIME FAULT on first launch, 0 steps completed.**

**What was validated**:
- NVSHMEM bring-up smoke on 8 ranks: PASS (barrier, team_split(0,1,8), symmetric malloc, cross-rank all-reduce = 36.0 bit-exact)
- Dry-run TP8 plan for all 8 ranks: per-rank Nmax = 1,280,000 (== kDecMaxTensorNumel/TP), budget 40.92 GiB FITS, sym heap 13.1 MB, PE range (0,1,8) ✓
- Flagship TC megakernel built with `-DSG_HAS_NVSHMEM=1 -rdc=true` + manual device-link against libnvshmem_device
- Real device NvshmemTransport confirmed linked (symbols nvshmemx_barrier_block + nvshmemi_transfer_quiet/amo/enforce)

**The blocker** (exact): compute-sanitizer caught **87 "Invalid __global__ write of size 4 bytes"** inside `fused_decoder_megakernel_tc<AdamW, ParConfig<8,8,1,1,Z3>>+0x1dc30`, by thread (192,0,0) block(11,0,0). Faulting address ~4.93 GB is ~35 GiB BELOW the NVSHMEM symmetric heap → wild pointer from the in-kernel TP path.

**Root causes identified** (in committed kernel source):
1. **No per-rank weight-shard offset**: `dectc_wbf_convert` reads FULL weight matrices identically on every rank; `comm.tp_rank` is never used to offset. All 8 ranks compute the same slice-0, so even if IMA were fixed the math would be wrong.
2. **Head divisibility**: flagship `SG_DEC_HEADS=25` is NOT divisible by TP=8. `Hloc=kHeads/TP=3`, `Dloc=kD/TP=200`, but invariant requires `Dloc==Hloc*kDhead = 3*64 = 192 ≠ 200`. Head-localized attention runs in an inconsistent partial-width regime.
3. **Coverage gap**: TP loopback gate only exercised a standalone FFN at d=128/dff=512 — never the full flagship decoder megakernel at d=1600/L=48. "Compiles + loopback math correct" did not cover real megakernel's TP activation/heap indexing.

**Build-system gaps** (worked around, non-committed):
- `torch.utils.cpp_extension.load()` does NOT emit the nvcc `-dlink` for `-rdc=true` TUs → manual 3-step build needed (compile → dlink against libnvshmem_device → host link)
- Committed pybind (`mega_decoder_real_adamw_tc.cu::tc_train_step`) calls the SingleGPU launcher — never fires the in-kernel TP path. TP8 dispatch is only in the 18-arg `tp_size=8` arm. Reached via non-committed scratch pybind.

**Non-committed scratch wiring**:
- `tuning/_tp8_scratch_pybind.cu`
- `tuning/_tp8_build.sh`
- `tuning/_tp8_run.py`

**Sanitizer log**: `/tmp/claude-0/.../scratchpad/san_rank0.log`

**MATCH to claimed state**: The claimed "TP data-path fix" with bugs A (per-rank weight-shard offset) and B (25-heads-not-%8 attention) **matches exactly** what the compute-sanitizer identified. The claim that A+B are fixed in `phase6/tp_datapath_fix_WIP.patch (ungated)` — i.e., the fix was authored but NOT applied/committed — is consistent with this output showing the IMA still present on the day of the run. Bug C (IMA itself, a consequence of A+B) is listed as "unconfirmed" in the claimed state, but this run CONFIRMED it: compute-sanitizer showed IMA on first launch.

---

### 3. Roofline Campaign (Task w9fa0p8eh — lines 9752–9805)

**Task**: Sustained GPU-saturating roofline campaign for flagship cells.

**Actual outcome**: **10 cells measured; 5 mamba cells blocked.**

**Results** (nsys-based, all-nsys ncu-free):
- Decoder d=1600/L48 (1.476B): **1.14–1.58 TF/s** (0.116–0.160% of 989 TF/s bf16 ceiling, I=82.8 FLOP/B, ncta_cap=8/B=128)
- ViT d=1664/L48 (1.596B): **0.27–0.28 TF/s** (0.027–0.029%, I=170.8, ncta_cap=4/B=64 — memory-forced)
- Mamba d=2048/L24: **BLOCKED** — flagship megakernel needs **19.56 MB dynamic smem per block** (88× over H100's 227 KB cap). Architectural limit of the Mamba-3 both-layers-in-smem design (`model_stage_mamba3.cuh:142`, `kMambaSmemBytes = 20,513,956`). Cannot be fixed by launcher params.

**Graph and data**:
- `/workspace/phase6/roofline_flagship.png`
- `/workspace/phase6/roofline_flagship.csv` (10 rows)

**DISCREPANCY vs claimed state**: The claimed "full 33-cell roofline" is NOT yet done. The current session deliverable is a **10-cell roofline** (decoder × 5 + ViT × 5, elementwise optimizers only). Staged-opt cells (Prodigy/Muon/LookSAM/SG2/SuperGrok2 × 3 models = ~18 cells) are blocked because `SG_*_BENCH_LAYOUT=1` disables per-CTA scratch, and mamba is unlaunchable. The 33-cell target remains open.

---

### 4. ViT and Mamba TP Track (Task w967olfbt — lines 9652–9750)

**Task**: Mirror decoder TP track onto ViT and Mamba.

**Actual outcome**: **Both ViT and Mamba TP tracks committed, 3/3 gates each.**

**ViT TP commit**: `ccdf80773d56198d24b528855a48782134953651`
- 21/21 pytest (SingleGPU byte-identical), NVSHMEM RDC compile COMPILE_OK
- Attention head-shard scoped out (same as decoder), 4 reduce points implemented

**Mamba TP commit**: `5e084cadb69cc8306928ddc12991f1b0c78f9a1a`
- 3/5 pytest PASS (2 pre-existing failures: obsolete `tc_dump_outproj_operands` + B_bias bf16-floor tol calibration issue)
- NVSHMEM RDC compile COMPILE_OK
- SSM body replicated (not sharded); projection GEMMs handle TP reduce points
- 2 pre-existing failures are NOT regression: identical before and after TP edits

---

### 5. TP Attention Head-Localization + Host Bringup (Task w4u1htj6y — lines 9087–9183)

**Task**: TP attention head-localization (kernel) + host NVSHMEM bootstrap + weight-shard + torchrun harness.

**Actual outcome**: **2/2 gates passed; commits 8e8f3d10 and 0b66169f.**

**Attention shard (kernel)**:
- Added `dectc_attn_fwd_tile_tp` / `dectc_attn_bwd_tile_tp` to `model_stage_decoder_tc.cuh` — parameterized by `Hloc=kHeads/TP`, `Dloc=kD/TP`
- `dec::kHeads==4` is NOT divisible by 8, so `Hloc==0` for `kHeads%TP!=0` → loops are no-ops; math exact only for head-divisible configs
- 19/19 pytest (SingleGPU byte-identical) + ParTP8 NVSHMEM RDC COMPILE_OK
- Deviation: for `kHeads%TP!=0`, `Hloc=0` makes attention no-op rather than static_assert — avoids breaking the mandatory compile gate for head-indivisible configs

**Host bringup** (committed pure Python, no csrc edits):
- `grokking_optimizers/distributed.py` — `partition_tensor_parallel` (Megatron col/row split), per-rank Nmax = 1,280,000 (== kDecMaxTensorNumel/8 exactly)
- `grokking_optimizers/host_bringup.py` — `TPBootstrap`, `bootstrap_tp_team`; live path raises `TPBootstrapBlocked` (no nvshmem Python binding on box)
- `grokking_optimizers/parallel/flagship_budget.py`
- `tuning/flagship_distributed.py` — dry-run validates plan/mesh/shard for all 8 ranks, rc=0
- Regression: 30 passed, 2 skipped, 0 failed

---

### 6. Dead-Code Analysis (Task w4pgw2cm6 — lines 9001–9087)

**Actual outcome**: 8,089,083 removable lines identified.

- `_dectc_codegen/` (64 files, ~7.95M lines of nvcc intermediate dumps)
- `_scan/` (43 files, ~91K lines)
- `_scan_prep.sh` + `_scan_prep.log`
- `claude_session_archive/` (419 files, ~87K lines)
- **results/ KEPT** (curated reference benchmark data, referenced by 7 tuning scripts)
- True source after removal: ~361,010 lines (1,047 text files) = 4.27% of current repo

Only one provably-dead SOURCE code path: `tc_dump_outproj_operands` on Mamba scalar-TC cell (TORCH_CHECK(false) stub; called only by test that was SKIPPED or errors).

---

### 7. Adaptive Parallelism Design Specs (Task wyf2usg0z — lines 16177–16282)

**Task**: Design specs for 3D-5D adaptive parallelism and size-adaptive kernel specialization.

**Actual outcome**: Specs authored at `/workspace/impl_diffs/adaptive_parallelism.md` and `/workspace/impl_diffs/size_adaptive.md`. **READ-ONLY, NOT applied.**

**Key findings on current state of adaptivity**:
- EP (Expert Parallelism as 5th axis) is **INERT for all current models**: `num_experts` keys in `grokking_race_v2.py` are ALL for the SuperGrok2 OPTIMIZER's PEER meta-net, NOT model-level MoE. Front-end returns EP=1 for every race model. EP is a future seam.
- Mamba's `_FUSED_ABI_STALE` constraint: the 33 sm_90 `.cu` cells (old granular per-opt cells) were already deleted; the 33 cells are now the multi-opt .so approach.
- CTA-tiling (occupancy > 1) is a **scoped follow-on** (GridBarrier assumes 1 CTA/SM; occupancy>1 needs Hopper thread-block clusters). Current production = SizeSmall = 1 CTA/SM.

---

### 8. Benchmarks and Race Results

**Task b3yufkmax / b9rfuzrpz / related** — Grokking race runs visible in the task outputs:

From `b9rfuzrpz.output` (lines 3519–3665): Live race across 8 GPUs, showing multiple optimizers running simultaneously. SuperGrok2 visible at step 22 with `tl=4.626` (not yet grokking). GrokAdamW at 3999/4000 steps. Multiple DNF annotations (AdamW/Grokfast/GrokAdamW finished within ~16s each).

**From `b2msxyjbc.output`** (line 2524): DEC saturation sweep results:
```
BEST: 8.46 TF/s (0.85% peak) at B=512 cap=32 step=2143.41ms
```
This is the decoder performance with cap=32 (32 CTAs out of 132). At ncta_cap=32, the kernel achieves 8.46 TF/s = 0.85% of peak. At ncta_cap=8/B=128 (roofline measurement): ~1.46 TF/s = 0.15% of peak.

**From `bfqr3ixph.output`** (lines 4551–4706): Phase 6 multi-model probe results:
- Decoder: `adamw 1.46 TF/s, lion 1.58 TF/s, grokfast 1.30 TF/s, grokadamw 1.13 TF/s, neuralgrok 1.13 TF/s`
- Mamba: `CSVFAIL mamba,* CUDA error: invalid argument` (all 5 opts blocked — the 19.56 MB smem issue)
- ViT: `CSVFAIL vit,* CUDA out of memory / TC vit megakernel launch failed: out of memory`

**VIT OOM pattern**: ViT flagship (d=1664/L48, 1.596B) requires 91.75 GiB or 82.03 GiB depending on batch — always exceeds single 79.2 GiB H100. The TC path also OOMs: "TC vit megakernel launch failed: out of memory" at ncta_cap=8. ViT can only run at ncta_cap=4/B=64 as shown in the roofline.

---

### 9. Decoder TC L=48 Flagship Compile (Task bgjqpxhqv — lines 4673–4706)

**CRITICAL FINDING**: The decoder TC kernel at L=48 (flagship layout) has **SIGNIFICANT SPILLS**:

```
ptxas info: 23616 bytes stack frame, 3956 bytes spill stores, 3976 bytes spill loads
ptxas info: Used 255 registers, 23616 bytes cumulative stack size, 25360 bytes smem
```

This is for `fused_decoder_megakernel_tc<AdamW, ParConfig<1,1,1,1,Z3>>` at d=1600/L=48. Compare to the test kernel (d=128/L=2): 0 spills, 251 regs, 1152-byte stack.

The COMPILE_OK is for the TU; the kernel itself has 3956/3976 spill stores/loads at flagship size. This is a significant performance concern — spills indicate the register file is being exhausted at flagship dims (L=48 means 48× more weight loads/GEMM calls per tile).

Also notable: `SG_TUNED_DEC_FWD_PIPE=1` with `SG_TUNED_DEC_FWD_STAGES=4` is baked into the default. Confirmed at:
- `model_stage_decoder_tc.cuh:186`: `#define SG_TUNED_DEC_FWD_PIPE 1` (entry-a deeper cp.async ring, +1.49×)
- `model_stage_decoder_tc.cuh:189`: `#define SG_TUNED_DEC_FWD_STAGES 4` (gated 11/11 × 3 seeds @ 618ms/6.477%)

---

### 10. Path Label / Wiring Guard / Roofline Re-run (Task b6imoc4rd — lines 2947–3069)

**Task**: Wiring guard — every train_* logs which path executed; roofline rows carry the path; FAIL loud if silent degrade.

**Actual outcome**: Edits applied to:
- `grokking_race_v2.py`: Added `LAST_L3_ENGINE` module-level dict + `B%16==0` truncation for wgmma path; precision gate flip to path-matched (bf16 for decoder/vit TC, scalar stays "L3-scalar bf16")
- `tuning/roofline.py`: Roofline wrapper reads `LAST_L3_ENGINE` to label rows with REAL path (TC vs scalar) and set matching ceiling

Gate results from `b6uj1wzxu` (line 3073): `GATE_EXIT=0` and "19 passed, 1 warning in 109.94s (0:01:49)"

**Existing roofline.json was STALE**: `path_family: None, path: None` everywhere — predates path-labeling code.

---

### 11. Mamba TC Gate Status (Session h100-audit-maximal, from archive lines 4388–4451)

From the claude_session_archive subagent results (the HANDOFF.md era, branch `claude/h100-audit-maximal`):

Final test result: **5/5 passed** on `test_mamba_tc.py`:
```
test_tc_single_step_grad_parity  PASSED  loss rel=1.79e-05, STRUCTURAL: layer0≈layer1, bf16-floor=1.615e-02
test_tc_proj_dw_exact_on_own_operands  PASSED  ISO rel=2.957e-07
test_tc_determinism  PASSED  bit-identical grad max-Δ=0
test_tc_short_trajectory  PASSED  loss[0]=4.8045 loss[49]=0.0004
test_tc_step_time_vs_scalar  PASSED  TC=17.767ms scalar=8.221ms ratio=0.46×
```

Note: `ratio=0.46×` means **Mamba TC is SLOWER than scalar** (scalar wins at 8.2ms vs TC at 17.8ms). This is consistent with the claimed state that mamba keeps scalar for measured performance.

But also from the same archive era (`claude/h100-audit-maximal`):
- Earlier run showed `test_tc_proj_dw_exact_on_own_operands FAILED proj-dW ISO rel 1.05e+00` — the dW bug was present in the HEAD baseline
- After fixes: `test_tc_single_step_grad_parity PASSED` (5/5)

---

### 12. TP IMA at 8 GPU Scale (Task b7ozprtf7 — lines 3080–3141)

Direct evidence of the TP IMA bug:
```
[rank3]: RuntimeError: CUDA error: an illegal memory access was encountered
[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered
```
Date: 2026-06-25 07:23:25. This is the CURRENT BLOCKER, confirmed live in the task outputs.

---

### 13. Misc Important Outputs

**`b0fboqo5q.output` (line 2486)**: `19 passed, 1 warning in 155.82s (0:02:35)` — likely the full decoder TC suite.

**`b3at107mo.output` (line 2539)**: `EXIT=0; 28 passed, 2 skipped, 271 deselected, 1 warning in 21.18s` — fast test subset.

**`b3x56sup5.output` (line 2562)**: `[result] LAUNCHED OK. loss=4.577683 finite_loss=True finite_grad=True grad_absmax=6.2168e+00` — single step launch works.

**`bg9w9f60r.output` (line 4664)**: `L=2 keystone gate FINAL` — 19 passed in 158.20s. The L=2 decoder TC gates all pass.

**`bgjqpxhqv.output` (line 4673)**: L=48 COMPILE OK but with 3956 bytes spill stores/loads at flagship dims.

**`bgavyweg7.output` (line 4673)**: `COMPILE_OK tu=csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu` in 50.183s.

**`b1t7w1pu0.output` (line 2508)**: `VIT_EXIT=0 COMPILE_OK tu=csrc/fused/sm_90/mega_vit_real_adamw_tc.cu`

**`b2lnmf961.output` (line 2515)**: `COMPILE_OK tu=csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu`

**`bc0ndjspg.output` (line 4477)**: TP8 scratch build: `sg_tp8_scratch.so` built (1965936 bytes), 1 symbol, 3 device nvshmem symbols pulled.

**`bscta4zud.output` (line 8248)**: JIT autotune FAILED for adamw/decoder/sm_90 — build returned None. Runtime JIT autotune path is broken (AOT-only path works).

**`bhm4pa6ur.output` (line 4695)**: 33 cells found, lists adamw/decoder, adamw/mamba, adamw/vit, grokadamw/decoder...

**`brpqw95la.output` (line 8047)**: SG2 TC build — BUILD_EXIT=0, 0 spills, 255 regs, 25360 smem for ParConfig<1,1,1,1,Z3>, ParConfig<8,8,1,1,Z3>. These both compile clean.

---

## Key Discrepancies vs Claimed Current State

### Confirmed vs Claimed
| Claim | Status from task outputs |
|-------|--------------------------|
| CuTe-atom GEMM engine 13/13 VALIDATED | CONFIRMED (repeatedly referenced as foundation) |
| 3 flagship models LAUNCH | CONFIRMED (decoder single-GPU OK; ViT OOMs at ncta>4; Mamba blocked by 19.56MB smem) |
| Full TP compiled + NVSHMEM validated (loopback) | CONFIRMED |
| Cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED on 8 GPUs | PARTIALLY CONFIRMED: NVSHMEM team smoke passes; the FULL MEGAKERNEL TP path has IMA — "cross-GPU in-kernel NVSHMEM TP all-reduce VALIDATED" is an overstatement if it means end-to-end training steps |
| 11-opt decoder ranking (overfit placeholder) | NOT SEEN in task outputs — this may be in other outputs not captured here |
| Roofline deliverable done | PARTIALLY: 10/33 cells measured (decoder 5 + ViT 5); mamba blocked by architectural smem limit; staged-opt cells blocked |
| Dead-code cleanup (removed 8.09M lines) | CONFIRMED as analyzed (spec at `/workspace/impl_diffs/deadcode_artifacts.md`) |
| Resource planner DONE | NOT directly evidenced in these task outputs |
| TP data-path fix A+B in phase6/tp_datapath_fix_WIP.patch, ungated | CONFIRMED CONSISTENT: fix was authored but the IMA (bug C) is confirmed live on 2026-06-25 |
| Bug C (IMA) unconfirmed | DISPROVED: IMA is CONFIRMED by compute-sanitizer (task w78epgggr) |
| TC step is 1.8x faster than scalar (decoder L=2) | CONFIRMED (6690 vs 12053 µs) |
| Decoder TC L=48 flagship has spills | CONFIRMED: 3956/3976 spill stores/loads at flagship dims |
| Mamba TC is slower than scalar | CONFIRMED: 17.767ms TC vs 8.221ms scalar (0.46× ratio) |

### Additions Not in Claimed State
1. **EP axis is INERT** for all 3 current models — no race model has model-level MoE experts.
2. **ViT flagship OOMs at single-GPU ncta>4** — requires B<<full to run at all.
3. **JIT autotune (runtime=jit) is broken** — build() returns None; AOT-only path works.
4. **Mamba TC has 2 pre-existing test failures** (obsolete tc_dump_outproj_operands + B_bias tol) — these are not kernel bugs but pre-existing test issues.
5. **Decoder TC L=48 has 23616-byte stack frame and ~4KB spills** — the R2.3 driver validated at L=2 was spill-free; the flagship scale (L=48) has spills due to register pressure.

---

## Summary State Assessment

- **DONE and validated**: Decoder TC Fork-B driver (L=2, all gates pass, 1.8× speedup vs scalar, committed to files though not to git per task constraint). TP host bringup, attention shard, ViT/Mamba TP tracks (compile gates). Dead-code analysis (spec only). Partial roofline (10 cells).
- **IN FLIGHT**: TP data-path fix (patch authored, not applied to committed source). Bug C (IMA in full TP megakernel) confirmed live, not fixed. Full 33-cell roofline blocked by mamba smem limit and staged-opt OOM.
- **BLOCKED**: Mamba flagship unlaunchable on H100 (architectural 19.56MB smem limit). ViT flagship memory-constrained (only 4 CTAs at B=64). 8-GPU end-to-end training still 0 steps completed.
- **NOT YET STARTED in these outputs**: 11-optimizer decoder ranking benchmark with real data (only placeholder confirmed).
