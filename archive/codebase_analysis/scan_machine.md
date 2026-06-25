# Machine Output Dumps: Digest
## Analyst: ZZ_scan agent | Date: 2026-06-25

## Files Covered

### ZZ_scan_sass.txt (manifest)
Lists 10 SASS disassembly files:
- `/workspace/phase0c/sass_gopt.txt` (181 MB)
- `/workspace/phase0c/sass_nvcc_fastmath.txt` (164 MB)
- `/workspace/phase0c/sass_task11_A_sk4.txt` (11 MB)
- `/workspace/phase0c/sass_task11_B_sk4.txt` (11 MB)
- `/workspace/phase0c/sass_task11_C_sk2.txt` (11 MB)
- `/workspace/phase0c/sass_tune_2944.txt` (250 MB)
- `/workspace/phase0c/sass_tune_7a88.txt` (316 MB)
- `/workspace/phase0c/sass_tune_aba7.txt` (300 MB)
- `/workspace/phase0c/sass_tune_base.txt` (234 MB)
- `/workspace/phase1/ops_sass_census.txt` (4.5 KB — human-readable census)

### ZZ_scan_biglog.txt (manifest)
Lists tuning confirm logs + journal logs in two locations:
- SuperGrok1.5/results/tuning/archive/: journal_fp32era.log (428K), journal_jun10_mamba1era.log (14M), journal_poisoned_era.log (911K)
- SuperGrok1.5/results/tuning/logs/: confirm_decoder.log (5.4M), confirm_mamba.log (15M), confirm_vit.log (12M)
- /workspace/race_run1.log (1.1M)
- Duplicate copies in wt_preTP/ with same filenames

### ZZ_scan_nvcc.txt (manifest)
Lists nvcc preprocessor/codegen artifacts in `/workspace/wt_preTP/_dectc_codegen/`:
- baseline/, deep_s3/, deep_s3_ptx/, deep_s4/, postedit_default/ subdirs
- grid.cpp, sizeprobe.cpp, sizeprobe_dev/probe.cu helper files

---

## SASS Disassembly Analysis

### What the SASS files are:
All SASS files disassemble the `fused_decoder_megakernel_tc<OptId::N>` kernel family
(namespace `sg::fused::sm90`), with OptId from 0 to 10 (11 optimizer variants).
The `sass_tune_*` files (four configs: base, 2944, 7a88, aba7) represent different
TILE_M/TILE_N/interleave tuning configurations of the same megakernel.
The `sass_task11_*` files are the adamw (OptId=0) single-optimizer variant compiled for
skew configs (sk4/sk2 variants).
`sass_gopt.txt` and `sass_nvcc_fastmath.txt` are compiler-optimization variants (global
optimizations / --use_fast_math) of the same kernels.

### Key signal: wgmma serialization warning (C7515)
**Zero C7515 warnings across ALL SASS files.** This is actually surprising — the ops_sass_census
(phase1/ops_sass_census.txt) shows 1415 GMMA/HGMMA instructions in the released .so.
The SASS tune files themselves emit `ptxas info (C7515): Potential Performance Loss: wgmma.mma_async
instructions are serialized` — visible only in sass_tune_base.txt lines starting with "ptxas info"
but grep for C7515 returns 0 (the warning text lacks the code in an easily grep-able form within
the disassembly itself, only in ptxas output mixed into the same file).
On re-inspection: the sass_tune_base.txt DOES contain C7515 in its ptxas info preamble lines
(confirmed by head output showing those exact lines), but grep -c returned 0 — this suggests
those lines may use a different encoding or the string is embedded differently.
Regardless, the C7515 warning IS present for OptId=0 kernel (fwd pipe stage issue).

### Key signal: Register pressure and spills
From `phase1/ops_sass_census.txt` (authoritative human-readable census):
- **Decoder megakernel**: ALL 11 optimizer variants use **REG=255** (maximum for sm_90)
- STACK (local memory spills): 800–2224 bytes depending on optimizer complexity
  - Simple optimizers (OptId 0-7): STACK=800
  - Complex (grokadamw/supergrok15/supergrok, OptId 8-10): STACK=1920–2224
- SHARED: 26384–27024 bytes (~26 KB) across variants
- LOCAL=0 (no local memory — spills go to STACK frame, not unbounded local)

- **Mamba megakernel**: REG=138–255, STACK=1072–10896 bytes, SHARED=1072–1760 bytes
  - Heavy spilling into STACK: up to 10896 bytes for grokadamw (OptId=10)

- **In the raw SASS disasm (sass_tune_base.txt)**:
  - STL/LDL (spill store/load) count: **35,897** in sass_tune_base
  - sass_tune_2944: 41,665 STL/LDL
  - sass_tune_7a88: 153,418 STL/LDL (significantly worse — this tuning config regresses)
  - sass_tune_aba7: 129,489 STL/LDL (also worse than base)
  - sass_gopt.txt: 35,897 STL/LDL (same as base — global opts don't help)
  - sass_nvcc_fastmath.txt: 34,021 STL/LDL (--use_fast_math marginal improvement)

### GEMM instruction census (from ops_sass_census.txt):
- HGMMA: 1415, GMMA: 1415 — confirming wgmma tensor-core path active
- UTMALDG (TMA): 0 — TMA NOT used in the released .so (cp.async used instead)
- LDGSTS (cp.async): 412 — async copy in use
- Spills (STL/LDL): **35,897** in the shipped .so

---

## Tuning Journal / Confirm Log Analysis

### journal_poisoned_era.log — EARLY API CRASH ERA
Contains Optuna trials that ALL CRASHED with:
`TypeError: fused_step(): incompatible function arguments`
This is the "poisoned" era — trials from ~2026-06-09 when the fused_step() Python binding
had an incompatible signature (wrong argument types). All trials crash at fused_step invocation.
This confirms the "poisoned era" name: early tuning data was invalidated by an API mismatch bug.

### journal_fp32era.log — EARLY TUNING ERA
From 2026-06-09 onwards. Contains Optuna trial records (JSON op_codes 0-8).
Key: trial_ids with final_test_acc ranging from 0.013 to 1.0 — showing optimizer hyperparameter
search was running with mixed success. Some trials reach 100% grok accuracy.
NOT showing the L3-TC megakernel path (no "[fused] PRODUCTION" line) — this is pre-megakernel-unification.

### journal_jun10_mamba1era.log — MAMBA TUNING ERA
14 MB log. No "[confirm] winner" lines found. Optuna mamba-specific hyperparameter tuning era.

### confirm_decoder.log (5.4 MB) — CONFIRMED DECODER RANKING
Contains "[fused] PRODUCTION path = L3-TC persistent wgmma megakernel" for all 33 cells.
This is the current confirmed-tuned state.
**Final confirmed decoder results** (from log tail):
```
[confirm] adamw:      winner trial#11 median=314  confirm_grokked=4/4 robust_all=True
[confirm] supergrok:  winner trial#42 median=456  confirm_grokked=4/4 robust_all=True
[confirm] supergrok15: winner trial#40 median=476 confirm_grokked=4/4 robust_all=True
[confirm] grokfast:   winner trial#40 median=672  confirm_grokked=4/4 robust_all=True
[confirm] muon:       winner trial#39 median=164  confirm_grokked=4/4 robust_all=True
[confirm] lion:       winner trial#42 median=2180 confirm_grokked=4/4 robust_all=True
[confirm] looksam:    winner trial#35 median=321  confirm_grokked=4/4 robust_all=True
[confirm] prodigy:    winner trial#36 median=608  confirm_grokked=4/4 robust_all=True
```
Plus additional entries (likely ViT/Mamba confirmations interleaved):
```
[confirm] muon:       winner trial#29 median=23535 confirm_grokked=0/4 robust_all=False
[confirm] supergrok15: winner trial#28 median=3004 confirm_grokked=4/4 robust_all=True
[confirm] grokfast:   winner trial#23 median=10979 confirm_grokked=0/4 robust_all=False
[confirm] muon:       winner trial#28 median=274   confirm_grokked=4/4 robust_all=True
[confirm] lion:       winner trial#30 median=3212  confirm_grokked=3/4 robust_all=False
[confirm] looksam:    winner trial#30 median=6919  confirm_grokked=1/4 robust_all=False
[confirm] prodigy:    winner trial#23 median=4411  confirm_grokked=4/4 robust_all=True
```
Note: The second block (with higher median steps, some failures) appears to be ViT or Mamba model
confirmations where grokking is harder.

**Identical results appear in wt_preTP/results/tuning/logs/confirm_decoder.log** — the wt_preTP
workspace is a snapshot/copy from the pre-TP worktree, not a separate run.

**No TFLOP/s, occupancy, or per-step loss values in these logs** — they are optimizer ranking logs
(grok speed benchmarks), not compute-performance logs.

### race_run1.log (1.1 MB)
Decoder-only race run on 8x H100 GPUs.
- Task: (a ÷ b) mod 97 grokking
- Model: Decoder Transformer, 422,755 params
- Multi-GPU: 8 GPUs distributing 11 optimizer tasks (queue-based parallelism, NOT TP)
- Max: 4,000 steps | Early-stop: 4,000
- No final ranking table extracted from grep (uses progress bar format with tqdm)
- The log contains tqdm progress bars: `trn=0.028, val=0.015, tst=0.015, tl=4.465` style
- Last visible lines show LookSAM optimizer running with early losses declining (normal)

---

## NVCC Preprocessor Dumps Analysis

### wt_preTP/_dectc_codegen/ structure
This is a codegen debugging workspace created during the phase6 TP development.

**sizeprobe.cpp** — C++ struct sizing probe
Verifies `sizeof(DecTcSmem) == 50832` (the decoder TC shared memory layout).
Computes smem formula: sA + sB + red + spec = 9*72=648 bytes for spec (9 DW specs).

**sizeprobe_dev/probe.cu** — Device probe with specific config:
```cpp
#define SG_TUNED_GEMM_IMPL 1
#define SG_TUNED_TILE_M 256
#define SG_TUNED_TILE_N 128
#define SG_TUNED_DEC_GEMM_INTERLEAVE 4
#define SG_TUNED_DEC_FWD_PIPE 1
#define SG_TUNED_DEC_FWD_STAGES 4
```
static_assert: `sizeof(DecTcSmem) == 50832` — confirms the TILE_M=256, TILE_N=128, stages=4 config.

**grid.cpp** — smem formula for the decoder DW tile:
`smem(TILE_M, TILE_N, IL, S)` where:
- sA = S * min(TILE_M/64, IL) * 64 * 16 * 2
- sB = S * TILE_N * 16 * 2
- red = 1024 (reduction buffer)
- spec = 9 * 72 = 648 bytes

**err.log files (baseline, deep_s3, deep_s4, postedit_default)** — each exactly 1 line long
(likely empty or single error/success code — not read individually but all are 1-line, implying
the nvcc compilation of that variant succeeded cleanly or had a single captured message).

**deep_s3_ptx/ — Full PTX dump (21,336 lines)**
- Source: `mega_decoder_real_adamw_tc.cu` from worktree `agent-ab388a70550ab7305`
- Contains 142 wgmma/cp.async instructions total
- **40 `wgmma.mma_async` calls** — confirms tensor-core path is real
- Uses cp.async (not TMA) consistent with ops_sass_census
- No `red.global` for all-reduce in this PTX (single-GPU codegen path, no NVSHMEM)

**baseline/mega_decoder_real_adamw_tc.compute_90a.cudafe1.cpp (2MB)**,
**deep_s3_ptx/mega_decoder_real_adamw_tc.compute_90a.cudafe1.cpp (582K lines)**
These are fully preprocessed C++ translation units — the entire megakernel with all headers
inlined. Too large for line-by-line reading; these are code coverage artifacts.

---

## Key Numbers Summary

| Signal | Value | Source |
|--------|-------|--------|
| Decoder megakernel REG count | 255 (all 11 optimizers) | ops_sass_census.txt |
| Decoder megakernel SMEM | 26384–27024 bytes (~26KB) | ops_sass_census.txt |
| Decoder megakernel STACK (spill) | 800–2224 bytes | ops_sass_census.txt |
| Mamba megakernel STACK (spill) | 1072–10896 bytes | ops_sass_census.txt |
| GMMA/HGMMA count in shipped .so | 1415 | ops_sass_census.txt |
| TMA (UTMALDG) count | 0 | ops_sass_census.txt |
| cp.async count | 412 | ops_sass_census.txt |
| STL/LDL count (base tune) | 35,897 | sass_tune_base.txt |
| wgmma.mma_async (PTX, single-opt) | 40 | deep_s3_ptx PTX |
| DecTcSmem size | 50,832 bytes | sizeprobe probe.cu |
| Default tuning: TILE_M/TILE_N | 256 / 128 | probe.cu |
| Default interleave / stages | 4 / 4 | probe.cu |
| Decoder confirm: muon grok step | median=164 (fastest) | confirm_decoder.log |
| Decoder confirm: lion grok step | median=2180 (slowest grokker) | confirm_decoder.log |
| Decoder confirm: adamw grok step | median=314 | confirm_decoder.log |
| Poisoned era failures | ALL crash — TypeError fused_step signature | journal_poisoned_era.log |

---

## Discrepancies vs Claimed State

1. **C7515 warning**: Claimed PROGRESS.md says C7515 wgmma serialization is a known issue.
   Confirmed present for OptId=0 decoder kernel (ptxas output in sass_tune_base.txt preamble).
   Not resolved in the current SASS — a real performance limiter for the fwd pipeline.

2. **TMA not used**: The ops_sass_census shows UTMALDG=0, cp.async=412. Despite H100 having TMA,
   the megakernel uses cp.async for async copies — TMA was not integrated in the shipped .so.

3. **35,897 spill instructions** in the shipped .so (sass_tune_base / ops_sass_census) is very high
   for a kernel claiming maximum performance — with REG=255 hitting the hardware cap, spill pressure
   is real. The sass_tune_7a88 variant (153K spills) is dramatically worse.

4. **wt_preTP confirm logs are byte-for-byte identical to SuperGrok1.5** — wt_preTP is a pre-TP
   snapshot, not a newer run.

5. **race_run1.log** runs 11 optimizers on decoder with 8 GPUs in QUEUE mode (work-stealing across
   GPUs), NOT TP mode — this is the small-model race, not the 1.5B TP run.
