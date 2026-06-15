# VIT L3-TC bottleneck — measured diagnosis (B=16384, quiet)
## Baseline: 716.85ms/step, 1.32 TF/s, frac_of_roofline=0.0016 (0.16%)
## After acc[17]→single-acc fix (committed to model_stage_vit_tc.cuh): 580.69ms, 1.63 TF/s, frac 0.0019 (-19%). 21/21 parity PASS.

## Phase breakdown (clock64, SG_VIT_PROFILE):
  P1_fwd:        52.8ms (9.1%)
  P1_bwd:        58.3ms (10.1%)
  P2_dW+reduce: 467.3ms (80.8%)  <-- DOMINANT
  P3_opt_tail:    0.15ms (0.0%)

## P2 dW root cause (MEASURED, not inferred):
- dW = dY^T @ X, K=T=278528. ~52 tiles (ff0=16, ff2=16, in=12, out=4, head=2, patch=2) over 132 CTAs.
- Standalone 1-m-atom dW (Nout=64,N=128,K=T) = 72.2ms, 4.15 us/k-step, LINEAR in K (full/half=2.11).
- 4.15us/kstep = ~8200 cyc for ONE wgmma m64n128k16 (~256cyc) + 2 staging copies + 2 syncthreads + full wgmma_wait_group<0>.
- => STAGING-LATENCY-BOUND + NO PIPELINING. NOT memory-bound (0.15% HBM), NOT compute (0.048%), NOT monolith (standalone even slower per-work), NOT register-spill.
- Engine = tc_gemm_block_unpipelined: stage A + stage B + sync + 1 wgmma + WAIT + sync, per k-step, fully serial.

## Levers (all structural, gated by dW parity gates in test_vit_tc.py):
1. PIPELINE the k-loop (double-buffer staging, overlap with wgmma) — kills the ~8000cyc/kstep serialization. SG_TUNED_PIPE_DEPTH.
2. COALESCE transposed staging reads (dY[k*Nout+m] stride-Nout, X[k*N+n]).
3. SPLIT-K across CTAs (52 tiles → ~80 idle SMs; split K=T contraction + reduction). SG_TUNED dW split.

## Scope: model_stage_vit_tc.cuh (vit-local engine copy, in-scope). All 11 vit cells inherit. No commit.

## CORRECTION after splitting P2 (refined 6-phase profile):
  P2_dW_GEMM     = 190ms (13%)   <- NOT the bottleneck (earlier 467 lumped P2 together)
  P2_grad_assembly = 871ms (59.5%) <- THE REAL DOMINANT
  B1_barrier_wait = 142ms (9.7%)
  (SUM over-counts: maxes don't add + extra syncs; RATIOS are the signal)

## grad_assembly root causes (MEASURED):
- vittc_dw_biases: db[o]=Σ_k dY[k,o]. (1) RUNS REDUNDANTLY ON ALL 132 CTAs (no nCTA guard!),
  (2) serial K=278528 loop per output element, (3) stride-Nout UNCOALESCED reads. 132x wasted.
- vittc_clspos_owner_scan pos-rows: `for t in [0,T): if t%kSeq==p` — scans ALL T but keeps 1/17,
  17x wasteful + uncoalesced; only kSeq=17 CTAs active (115 idle).
- vittc_lnvec_reduce: distributed + small — fine.

## FIX (safe, big win, parity-preserved):
1. dw_biases: partition (weight,out-col) across CTAs (kill 132x redundancy) + coalesce + parallel-K.
2. clspos pos-scan: index t = p, p+kSeq, p+2*kSeq,... (stride-kSeq direct, no 17x scan); spread across more CTAs.
Gated by 21/21 + A/A/A (sums must match to fp32 reorder tol; keep ascending order).

## PROGRESS (all gated 21/21):
  716.85ms (baseline)
  → 580.69ms  acc[17]→single-acc (C7515 spill fix)
  → 370.04ms  dw_biases 132x-redundancy removed + clspos stride-kSeq
  → 204.66ms  dw_biases warp-parallel K-reduction (grad_assembly 871→7.6ms)
  = 3.5x faster so far.

## Current phase split @ 204ms (ratios):
  P1_fwd 53ms (20.5%) | P1_bwd 58ms (22.4%) | B1_wait 64ms (24.4%) | dW_GEMM 78ms (29.7%) | grad_assembly 7.6ms (2.9%)
  → dW_GEMM now dominant. Next: coalesce dW staging (thread-decomp swap, no transpose — advisor), then maybe pipeline.
  → B1_wait 64ms (load imbalance at B1) also notable; P1 fwd+bwd 111ms is now a big share too.

## DECISION (banked 3.5x; advisor-confirmed):
- Coalesced dW staging REVERTED: m-major HBM read coalescing introduced a compensating smem bank conflict (dst mn*8 stride) → neutral-to-worse (78→87ms). 4th from-reasoning mechanism guess refuted by measurement; ncu blocked so stop hand-optimizing the staging blind.
- KEPT (all 21/21 gated, propagate to lion/grokfast vit via shared header):
  (1) single accumulator (C7515 spill removal), (2) dw_biases 132x-redundancy removal + clspos stride-kSeq, (3) dw_biases warp-parallel K-reduction.
- REMAINING levers are SG_TUNED dims for compile.py (NOT hand work, per req#9):
  * B1_wait 64ms = load imbalance (256 tiles/132 CTAs) → ncta dimension.
  * dW_GEMM 87ms serialization → SG_TUNED_PIPE_DEPTH + split-K.
- frac is occupancy/AI-pinned at d=128 (ROOFLINE.md) → low-single-digit ceiling; "flat" = marginal returns there.
- Close loop: rebuild _ops.so → tail_gate {lion,adamw,grokfast}/vit + wiring_check --models vit → clean non-PROFILE roofline frac.

## FINAL — CLEAN (non-PROFILE) production path, B=16384:
  wall/step: 716.85ms → 206.00ms  (3.48x)
  achieved:  1.32 → 4.60 TF/s
  frac_of_roofline: 0.0016 → 0.0055  (bf16 ceiling 840 TF/s; AI 250; occupancy/AI-pinned at d=128 per ROOFLINE.md)
  scalar/TC speedup: 1.66x → 5.04x

## GATES (all on rebuilt production _ops.so):
  tail_gate lion/vit:     PASS (params/state rel 0.0, A/A/A det)
  tail_gate adamw/vit:    PASS (params rel 5.7e-8, state ≤9.5e-7, A/A/A det)
  tail_gate grokfast/vit: PASS (params rel 2.8e-8, ema 0.0, A/A/A det)
  wiring_check --models vit: 3/11 wgmma (adamw,lion,grokfast); 8 blocked loud+cited
  test_vit_tc.py: 21/21 (run 3x across fix sequence)

## PER-CELL (the 3 share the fwd+bwd+reduce megakernel; opt tail P3=0.15ms negligible → same frac):
  adamw/vit:    4.60 TF/s, frac 0.0055
  lion/vit:     4.60 TF/s, frac 0.0055
  grokfast/vit: 4.60 TF/s, frac 0.0055

## HEAD-LOOP QUANTIFICATION (advisor's last lead — settled):
  per-sample head/CE loops (lines 714 fwd / 879 bwd): fwd_head 0.58ms, bwd_head 0.23ms = ~0.8ms total (~0.4% of step).
  → NOT a bottleneck. P1_fwd/bwd (111ms) is the unpipelined layer-GEMMs + attention, NOT the serial head.
  → No further IN-SCOPE structural win. Remaining levers are compile.py autotuner (req#9) / cross-lane pipelined GEMM (#32/#42).

## HONEST FRAMING (not "flat"):
  frac 0.0055 is IMPROVABLE (ROOFLINE.md: multi-CTA-per-tensor tiling is the path; my profile shows dW at 0.048% compute / B1_wait 64ms idle — serialization/imbalance-bound, not hardware-limited). Reported as "maxed within scope (3 real bug-fixes, 3.48x), further gains via cited autotuner/structural levers" — NOT roofline-flat.

## CITED REMAINING LEVERS (out of this lane's edit scope):
  - dW_GEMM 78ms + P1 GEMMs: unpipelined tc_gemm_block_unpipelined → SG_TUNED_PIPE_DEPTH + the validated tc_pipelined_gemm_m64nNk16 (needs transpose-staging or accessor-pipeline; cross-lane #32/#42).
  - B1_wait 64ms: 256 tiles/132 CTAs imbalance → ncta / split-K (autotuner).
  - vit dims (SG_TUNED_TILE_N, SG_TUNED_VIT_TILE_M) exposed in-header but NOT registered in compile.py → compile.py-lane (#35/#39).
  - SG_VIT_TC_MEGA_BLOCK is a hard #define (not SG_TUNED) — candidate for the compile.py lane.
