# PLANNING_INPUT — accurate state to write the engineering plan against (2026-06-17)

Purpose: give the owner a faithful, verified snapshot so the plan targets real headroom,
doesn't re-specify done work, and leaves the empirically-uncertain bits to the gate loop.
Source: phase1-status-audit (workflow wcrdmvuvq, refute-by-default) + the front-load re-profile.

## What is DONE / maxed (do not re-plan)
- **Fusion architecture = maxed**: model fwd + bwd + optimizer update run in ONE persistent
  `__global__` kernel, ONE launch, 1 CTA/SM, hand-built GridBarrier, zero intermediate launches.
  All 33 cells (3 models x 11 opts). This IS the one-binary-one-launch design.
- **Shipped perf wins**: decoder PIPE=1/STAGES=4 (+1.49x, 920->618 ms); ViT B1 sub-tile S=8
  (4.02x, baked b0d41f8, pending a confirm-build).
- **Decoder fwd/dX is now WGMMA-compute-bound** internally (deeper ring hid the drain) — so MORE
  buffering (S>4) is measured-unlikely to help on the decoder.

## Roofline (derived-from-measured-time; %-of-989-TF/s bf16 peak. NOT formally scored — #24 pending)
| model | ms/step | % of peak | bound |
|---|---|---|---|
| decoder d2048 | 618 | 6.48% | latency / phase-serialization |
| vit d2048 | ~1434 (post-S=8) | ~0.74% | was B1 load-imbalance (fixed) |
| mamba3 d128 | 222 | 0.03% | scan-dominated (can't scale past ~d142, smem cap) |
NONE is compute- or bandwidth-bound. The GEMM-only roofline UNDERSTATES true utilization (the step
is mostly non-GEMM fp32 CUDA-core work that fills the wall but not the FLOP numerator).

## Opportunity map (decoder 618 ms step — where the headroom is, ranked)
1. **P1_fwd 27.6% + P1_bwd 27.3% = ~55%** — token-tile GEMMs. WGMMA-bound internally. Lever = GEMM
   efficiency (tiling/TILE_M, interleave IL, occupancy, register pressure), NOT more buffering.
2. **Grid-barriers (B0+B1+B2) ~20%** — phase serialization. Lever = phase overlap / fewer barriers / async.
3. **P2_dW_GEMM 16.5%** — dW operand staging-bound. Lever = DW_STAGE method, split-K (G).
4. **P3 optimizer tail 5.9%** — UN-AUTOTUNED. This is the "max the optimizer kernels" target.
5. **Non-GEMM CUDA-core (attn/LN/GELU/embed/CE)** — large hidden share (fills wall, not in roofline %).

## Optimizer readiness (the "max the optimizers" surface)
Fusion maxed; tail KERNELS not perf-maxed (5.9% of step, no autotune data yet — front-load is now
generating it). 3 green (adamw/lion/grokfast), 8 gate-coverage caveats (NOT confirmed math bugs):
- grokadamw/prodigy: multi-step parity gate missing (vit/mamba).
- supergrok11: warm-up gate CLI-only (not CI-collected); supergrok15: NO warm-up gate.
- supergrok2: CSA oracle co-wrong (HIGH) + non-voting fidelity probe.
- muon/neuralgrok/looksam: BUG-04 (mamba staged-opt scratch un-gated -> OOM at d>=1024).

## Open decisions the PLAN should make (left to owner, not the loop)
1. Optimizer-max scope: all 11 or production set? Attack the 5.9% tail, or the bigger non-GEMM share?
2. Priority order across the opportunity map: GEMM-eff (55%) vs barrier-overlap (20%) vs dW (16.5%) vs opt-tail (6%)?
3. Correctness bar: re-gate all cells at d=2048 (currently d=128) before trusting perf changes? (Recommended yes.)
4. Mamba: fix BUG-04 + push toward scale, or freeze at d=128?
5. Numerics: keep transport-only (ascending-k fp32) hard constraint, or allow gated reassociation for specific kernels?

## Leave to the GATE LOOP (don't pre-decide in the plan — empirically arbitrated)
- Exact tile sizes / IL / register caps / split-K G / pipe depth per cell (the autotuner + ratchet pick these).
- Which candidate perf levers actually WIN (PIPE=2 looked right and lost; the hardware is the arbiter).

## Hard constraints (non-negotiable)
- fp64 parity (rel 1e-4; SAM 2.5e-2) + A/A/A bit-determinism = the gate. Keep/revert ratchet.
- One GPU now; Phase 2 (4D DP x TP x PP + ZeRO-3) is hardware-gated on the 8xH100.
- Commits LOCAL-ONLY (never push).
