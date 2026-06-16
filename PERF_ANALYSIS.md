# SuperGrok1.5 — Master Performance Analysis (1B+ scale)

Synthesis of three independent, read-only, file:line-grounded analyses (decoder+ViT, Mamba-3, shared substrate + multi-GPU), 2026-06-16. Goal: maximize all three model megakernels for 1B+ scale, single- and multi-GPU. Every lever is checked against the **fp64 parity hard-gate + A/A/A determinism**; that is a HARD gate, not a tiebreaker.

## TL;DR — the headline

**The ~2% roofline is structural, not a knob/scale problem — and the single highest-ROI fix already exists in the codebase, built and silicon-validated, but unused.**

- Measured floor (`.task11_perf_authoritative.log`, `PHASE1_CAMPAIGN.md:377-426`): decoder d=2048 = **2.0%** of the 989 TF/s bf16 roofline; ViT = **0.19%**; Mamba-3 can't place at d=2048 (smem-bound), stuck at d=128 (0.03%).
- Root cause (all 3 agents independently converged): **(1) no wgmma software pipelining** — the GEMM engine drains every K-step (`wgmma_wait_group<0>` per step; `wait_group<1>/<N>` appears nowhere); **(2) the fused step is mostly non-GEMM fp32 CUDA-core work** (attention/LayerNorm/GELU/bias/residual) that the roofline numerator (GEMM-FLOPs only) ignores but that fills the denominator.
- **Scaling alone is a dry well**: decoder is the SAME 2% at d=1024 and d=2048 → scale-invariant cap. Bigger batch is *necessary* for grid-fill (ViT P1 needs B≥8448 for 132 SMs) but cannot amortize the per-K stall or the GEMM fragmentation. Est. <0.2 abs-% from further scaling on current code; the 5–15× lives in the structural levers.

## The #1 lever (P0): wire in the dormant pipelined GEMM engine

A complete, **mbarrier producer/consumer, register-rebalanced (`setmaxnreg` 232/40), depth-tunable** pipelined wgmma GEMM exists and is **bit-identical by construction** (same ascending-K issue order — the determinism gate "c"):
- `csrc/backends/cuda/sm_90/tile_pipeline.cuh:272` — `tc_pipelined_gemm_m64nNk16<N, Depth>`, `SG_TUNED_PIPE_DEPTH ∈ {2,3,4}`. Exercised only by `wgmma_selftest.cu:139`.

Today the megakernels instead run:
- **ViT**: fully unpipelined `tc_gemm_block_unpipelined` (`model_stage_vit_tc.cuh:311-407`; `:346` "no S-stage ring") — measured ~8000-of-8200-cycle bubble per dW K-step (`archived_reports/from_disk_backup/vit_findings.md`: staging-latency-bound, NOT compute, NOT memory).
- **Decoder**: a hand-inlined **S=2** cp.async ring for fwd/dX only (`model_stage_decoder_tc.cuh:558-632`, capped by `static_assert` at `:120-121`); the **dW GEMM is unpipelined** (lambda sources → `kRingAsync=false` → synchronous `else` at `:633`, `:1544`).

**Change:** route fwd/dX/dW through the depth-3–4 producer/consumer engine; sweep `SG_TUNED_PIPE_DEPTH={2,3,4}`. Engineering: adapt the engine (dynamic-smem, one-tile turnkey) into the persistent megakernel's static-smem multi-tile loop; re-check the launcher occupancy gate still returns ≥1 block/SM (Depth=4 operand ring ≈ 32 KB ≪ 227 KB cap). **Parity: SAFE by construction** (reduction-order-invariant). **Risk:** engineering (adapter) + register budget (consumer 232 regs + fp32 accumulator; SAM optimizers already spill → gate depth per-optimizer). **Est. impact: 2–4× ViT, 1.3–2× decoder** (gated by the residual fp32-epilogue wall → P1 below). Shared engine ⇒ one change fixes both models. **More valuable under TP** (see multi-GPU).

## Prioritized lever set

| # | Lever | Files | Impact | Parity/determinism |
|---|---|---|---|---|
| **P0** | Pipelined GEMM engine (above) | `tile_pipeline.cuh:272`; `model_stage_{decoder,vit}_tc.cuh` GEMM calls | **Largest.** 2–4× ViT, 1.3–2× dec | SAFE (bit-identical, gate-c) |
| **P1** | Fuse the inter-GEMM fp32 epilogues (bias/round into the GEMM epilogue lambda; residual into LN's first read) + overlap epilogue of GEMM k with staging of k+1 | `model_stage_decoder_tc.cuh:1046-1188` + ViT twin (~10 `__syncthreads`-separated fp32 passes/layer) | MED-HIGH, multiplicative with P0 (~1.2–1.6×) | SAFE **iff** fp32 fold order preserved (oracle-matching, `:1071-1075`); no atomics/ragged → A/A/A-safe |
| **P2** | ViT B1 grid-barrier load imbalance (tile-count vs SM-count; measured 24.4%) | `fused_vit_megakernel.cuh:588,616`; tune `ncta_cap` / tile size | MED for ViT, 0 for decoder | SAFE (scheduling only) |
| **M0** | **Mamba-3: wgmma the projections + decoder-style output-stationary dW** | `model_stage_mamba3.cuh:361-376,440-470,784-794,853-874,1107-1183`; pattern `model_stage_decoder_tc.cuh:93-99,470-677` | **Unblocks Mamba-3 at scale.** Projections are **94% of FLOPs at 1.5B** (not the scan); also collapses the **668 GB** per-CTA grad-workspace wall (`fused_mamba_megakernel.cuh:173-174`) to tile-local; rides the smem fix | SAFE (proven substrate, ascending-k, owner-computes dW; scan untouched) |
| **M1** | Mamba-3 smem/HBM restructure to fit d=2048 (acts→HBM, one layer at a time) | `model_stage_mamba3.cuh:142-202`; cap `mamba3_layout.cuh:275` (2.5 MB needed vs 227 KB) | Enabler (can't run d=2048 without it) | SAFE (recompute = bit-identical) |
| **M2** | Mamba-3 chunked/associative parallel scan (long-seq enabler) | `model_stage_mamba3.cuh:521-770`; `MAMBA3_REFERENCE.md:96-115` (cumulative-product form, Prop 4) | Enables seq≥2048 training (serial scan spills regs there) | **HIGH RISK** — algebraically exact but FP-bit-different → re-anchor the fp64 oracle to the chunked form; gate from C=L downward; keep sequential behind a flag |
| S1 | 128B smem swizzle (kills bank conflicts) | `wgmma.cuh:157-172` (`kSwizzleNone`) | small-med | **PARITY-RISKY** (descriptor layout changes) — defer + gate |
| S2 | TMA-with-transpose to stream dW operands | `tile_pipeline.cuh:199` (deferred) | med (dW ~30% of ViT) | SAFE landing, high eng cost |
| S3 | `wgmma_issue_n256` + TILE_N=256 | `wgmma.cuh:578-579` (N≤128 locked) | small-med | register-bound risk (may break occ≥1) |

**Confirmed dry wells (do NOT pursue):** optimizer P3 tail micro-opts (<1% of step, vec4 NEUTRAL); GEMM-tile knob cycling (converged — TILE_N=64 +51%, IL=4 broke A/A/A); further batch/d scaling on current code (d=1024→2048 flat).

## The CUTLASS tension
`csrc/backends/cuda/sm_90/mma.cuh` is a real Sm90 TMA+wgmma+fp32-accum collective, but it's **host-launched** and "explicitly REJECTED for the persistent-megakernel path" (`wgmma.cuh:16-18`) — a host collective can't run device-side inside the persistent grid. So the fix is **not** "switch to CUTLASS"; it's **bringing CUTLASS-class technique (multi-stage pipeline + TMA + swizzle + warp-specialization) into the hand-rolled engine** — and the primitives (Mbarrier, `setmaxnreg`, cp.async, the TilePipeline ring) already exist; they're just unwired.

## Single- vs multi-GPU (the header-specialization axis)
- `ParConfig<1,1,1,1>` (`parallel_config.cuh:86`, `kEmitComm=false`) ⇒ every `if constexpr (kEmitComm)` comm branch folds away ⇒ the multi-GPU build is **byte-identical to single-GPU** (PTX-diff gate). So **all multi-GPU perf work is authorable now behind `if constexpr` with zero single-GPU regression.**
- **Counterintuitive but first-order:** TP *shrinks* per-GPU GEMMs (column/row Megatron split → `d/P`-wide shards, `tp_layer.cuh`), which makes per-GPU wgmma utilization **worse** → the P0 pipelining lever is **more** important under TP, and argues for **preferring DP+ZeRO+PP over wide TP** until GEMMs are pipeline-saturated.
- **Author-now (single-H100, loopback/CPU-gateable):** ParConfig plumbing + PTX-diff gate; TP math via `LoopbackTransport` (`tp_transport.cuh:95`, fixed-ascending-pe reduce = A/A/A-safe); sharded-optimizer DP=1 parity; PP stage kernels; **per-TP-degree sharded-shape tile specializations**; comm-overlap seams. **8×H100-only:** real NVSHMEM device-TP + its go/no-go gate; whether NVSHMEM reg-pressure breaks the zero-margin 1-CTA/SM occ≥1 invariant; weak-scaling; cross-rank A/A/A at real DP.

## Recommended FIRST action (empirical, before big rewrites)
Capture the authoritative d=2048 phase breakdowns that don't exist yet (only d=128/d=1024 runs exist), to rank P0/P1/P2 precisely:
- Decoder: `python tuning/decoder_bench.py --profile --d 2048 --B 16384` (8-phase clock64 split, `decoder_bench.py:185`).
- ViT: `scripts/_vit_phase_profile.py` adapted to the d=2048 bench layout (settles the `VIT_TC_001` head-CE question — a measured profile at d=128 found head-CE ≈ 0.4%, contradicting the campaign ledger's "highest-headroom" claim; re-measure at d=2048).

## Execution protocol
fp64-gated ratchet per [[feedback-patch-protocol]]: apply → build → fp64 parity HARD-gate + A/A/A determinism → 3-seed timing at d=2048 → KEEP iff faster on 3+ seeds AND parity-clean, else REVERT. M2 (chunked scan) additionally requires the fp64 chunked-oracle ≡ sequential-oracle proof first, and stays behind a compile flag until it passes 3 seeds.

## Ledger corrections this analysis forces
1. "roofline-converged / well-tuned" was **knob-convergence**, not roofline-closure (structural gap remains).
2. Mamba "scan-dominated" (`model_stage_mamba_tc.cuh:6-9`) is a **d=128 artifact**; projections = 94% at 1.5B.
3. The Mamba per-CTA full-grad-partial reduce is a hard **668 GB OOM at 1.5B** — must go output-stationary before any scale-up.
4. ViT's "highest-headroom = head-CE (VIT_TC_001)" is contradicted by a measured profile (head-CE ≈ 0.4%); the real ViT lever is **P0 pipelining** + the B1 imbalance.
