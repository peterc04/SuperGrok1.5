# Optimization Ledger — every KEEP / REVERT / NEUTRAL / SKIP, with the *why*

Canonical record of every optimization decision on SuperGrok1.5, per the owner's directive
(2026-06-16): **document every keep and revert with a detailed explanation of why.** Verdict
rule ([[feedback-patch-protocol]]): apply → build → **fp64 parity HARD-gate + A/A/A determinism**
→ 3-seed timing at the roofline scale → **KEEP iff faster on 3+ seeds AND parity-clean, else
REVERT**. Parity/determinism is a gate, not a tiebreaker. "Neutral" (no measurable gain) counts
as not-positive. On any REVERT, root-cause the *mechanism* and let it redirect the next decision.

Legend: **KEEP** = measured win, retained · **REVERT** = applied then undone · **NEUTRAL** = no
measurable change (treated as not-positive) · **SKIP/REJECT** = bench-tested, not landed.

---

## Track A — static-optimization patch series (`.regpressure/0001..0005`, P1–P5)
Verdicts also in `.regpressure/PATCH_SERIES_VERDICTS.txt`. Composition committed `fc14d14`.

| Patch | Change | Verdict | Why |
|---|---|---|---|
| 0001 bf16-prestage | pre-stage bf16 operands | **KEEP** | In the baseline; enables the cp.async ring (0005). Foundational. |
| 0002 decoder SAM-scoped outline | `__noinline__` the SAM 2nd-pass on decoder | **KEEP** | Cut register pressure on the SAM-coupled cells; faster + parity-clean on 3 seeds. |
| 0003 vit SAM-scoped outline | same idea on ViT | **REVERT** | **+5% slower.** ViT's structure differs; the outline added call overhead exceeding the spill savings. |
| 0004 mamba scope-noinline | scope-limit the mamba pass | **KEEP** | **−4.4%** single-pass; parity-clean. |
| 0005 decoder cp.async-ring | S=2 cp.async double-buffer ring on fwd/dX | **KEEP** | **−14.2% across ALL decoder cells** — the biggest static win. (Note: dW couldn't take this — its transposed-strided sources can't cp.async; that gap motivated the later P0 attempt.) |

## Track B — compile.py autotuner (self-test gated; build/tune-time measured)
- **NARROW micro-opt loop** (`e3b9b71→19912db`): **9 KEEP / 0 REVERT**, then dry-well STOP. All bit-neutral micro-opts (codegen cleanups); each verified self-test-green. Too narrow in scope (the owner later asked for the BROAD scope).
- **BROAD loop** (`62a9128`, `9a0645c`, `deb46ef`): **2 KEEP / 2 DEFER / 5 neutral → STOP.**
  - compile-01 single-cell source scoping — **KEEP**: build only the tuned cell's TU + reuse other models' AOT `.o`. Real build-time win.
  - compile-10 memoize `_owns_extension_module_tu` — **KEEP**: removes redundant TU-ownership recomputation.
  - 5 others **NEUTRAL** (no measurable build/tune-time gain) → 3-consecutive-neutral STOP.

## Track C — kernel roofline cycling @ d=2048 (model + optimizer stages)
| Lever | Change | Verdict | Why |
|---|---|---|---|
| decoder DW split-K | `SG_TUNED_DEC_DW_SPLITK` 4→2 | **KEEP −2.5%** (`a625227`) | Faster on 3 seeds @ d=2048; parity + A/A/A clean. The one kernel-track win. |
| decoder GEMM interleave | `SG_TUNED_DEC_GEMM_INTERLEAVE` 2→4 | **REJECT** | Bench −9.3% (fastest lever found) BUT **A/A/A determinism FAILED** on all 10 decoder cells: IL=4 makes the dW M-atom groups 4-wide, and at the toy d=128 the ragged atom counts (qkv 6→4+2, ff 8→4+4) make the 4-wide dW group-reduce non-deterministic. Ran fine + faster at d=2048 (atoms divide evenly) so the bench missed it; **the gate caught it.** KEY LESSON: bench-first is necessary, NOT sufficient — every winner must clear the full A/A/A gate. |
| elemwise vec4 AdamW | wire `adamw_step_vec4` into the P3 tail | **NEUTRAL** (`+0.05%`) | Bit-identical + A/A/A-safe, but the P3 optimizer tail is **<1% of the d=2048 step** (P1 fwd/bwd + P2 dW dominate) → no measurable gain. CROSS-TRACK FINDING: the whole optimizer-track candidate family targets that <1% region → a dry well at scale; wins live in the model GEMM. |
| decoder output-tile N | `SG_TUNED_TILE_N` 128→64 | **SKIP +51%** | Smaller output tiles murder tensor-core utilization (1.32% vs 1.94% roofline). |
| decoder TILE_M | `SG_TUNED_TILE_M` →256 | **SKIP +27.7%** | Slower. |
| decoder DW split-K | 4→8 | **SKIP +2.6%** | At d=2048/B=4096 the dW tiles already fill the grid; 2× split-K just doubles the partial-reduce + workspace traffic. |
| ViT DW split-K | 4→2 (mirror decoder) | **NEUTRAL** | Not a reliable-on-3-seeds win. ViT @ d=2048 is 0.185% roofline, bottlenecked on the non-GEMM head-CE/LN/attn surface, not the dW split. |
| mamba (any) | — | **N/A (blocked)** | Mamba-3 megakernel can't place at d=2048 (smem-bound, ~215 KB one-CTA/SM); no d=2048 cycling possible. Its roofline stays at d=128. |
**Result: decoder STOP (3-consecutive not-positive: IL=4 reject → vec4 neutral → TILE_N skip); the decoder L3-TC GEMM is knob-converged at TILE 128×128/IL=2/STAGES=2/cp.async-ring.** Net kernel-track win = the −2.5% split-K.

## Track D — #11 validation (not a keep/revert; a measurement)
compile.py vs regular nvcc @ d=2048 (adamw/decoder, `20013f7`): **A→B = +0.43%** (ptxas-tuning flags — NEUTRAL; the kernel is smem/occupancy-bound, not codegen-bound), **B→C = −2.09%** (the autotuned split-K knob — the real win, corroborates Track C), **A→C = −1.67%** total. Takeaway: compile.py's value at scale is the tuned knob, not the flag pipeline.

## Track E — perf campaign (structural, 1B+ scale; fp64-gated)
| Lever | Change | Verdict | Why (detailed) |
|---|---|---|---|
| **P0 decoder dW pipelined GEMM** | add `tc_gemm_block_pipelined` (mbarrier producer/consumer ring), wire decoder dW, sweep `SG_TUNED_PIPE_DEPTH` {1,2,3,4} | **REVERT** (2026-06-16) | **fp64 gate PASSED (11/11 decoder cells — correct), but perf REGRESSED:** depth-1 (unpipelined fallback) baseline = 1921.8 ms; depth-2 (pipelined) = **2144.8 ms (−11.6% slower)**; depth-3/4 **won't launch** ("too many resources" → occupancy <1, and the persistent grid-barrier requires ≥1 block/SM). **ROOT CAUSE: the dW is STAGING-bound, not drain-bound** — `vit_findings.md` clock64 shows ~97% of each dW K-step is scalar transposed-strided operand staging + syncs, only ~3% is the wgmma. Pipelining overlaps staging↔MMA, but you can't hide a 97% staging cost behind a 3% MMA (the consumer finishes instantly and stalls on the producer), and the producer/consumer split *reduces* the threads doing the already-slow staging + adds ring/barrier overhead → the real bottleneck got worse. (The impl also used `MaxAtomsM=kDecDwIL` not 1, inflating each ring slot → the depth-3/4 resource overflow.) **REDIRECT:** the dW's real lever is faster **STAGING** — TMA-with-transpose to *stream* the transposed operands (promoted from "defer" to PRIMARY), or a contiguous layout killing the scalar gather — NOT MMA pipelining. Pipelining may still help the *drain-bound* fwd/dX (27% of the step, cp.async, not staging-bound) — untested. The doomed **ViT-dW twin was dropped pre-emptively** (ViT dW is even more staging-dominated). Reverted in source; the `_ops` `.so` is still the slower depth-2 build (rebuild to baseline next cycle). |

### Pending (Track E, not yet decided)
- Mamba-3 M0 — wgmma-projection (in_proj/out_proj/SwiGLU) + output-stationary-dW INFRA **authored** (worktree `worktree-agent-a2a7fdbeab1836b95`; saved `.perf/M0_mamba_wgmma_projections.patch`; +380 lines in `model_stage_mamba_tc.cuh`, reuses the plain `tc_gemm_block_unpipelined` engine; syntax-clean incl. the full mega_mamba TU; determinism-safe by construction — ascending-k, output-stationary, no atomics). **DORMANT** (not a KEEP yet — the existing scalar megakernel path is byte-identical/untouched, so zero perf change until wired in). Remaining substantial step = the **final integration**: rewire `fused_mamba_megakernel_tc`'s per-sample scalar fwd/bwd dispatch → decoder-style P1 token-tile-fwd + P2 dW-stationary phases; then merge+gate the complete M0. (x_proj/dt_proj deferred — scan-coupling / sub-m64; smem-fit = M1; chunked scan = M2.)
- dW TMA-with-transpose staging — the redirected primary lever (confirm staging-bound via intra-dW profile first).
- P1 epilogue fusion; P2 ViT B1 barrier imbalance; #23 compile.py tiered spill management.

---
## Track E — dW contiguous-layout staging: **KEEP +2.05× — the campaign's first major structural win** (2026-06-16)
The Track-E redirect lever (after the P0 pipelining revert). `SG_TUNED_DEC_DW_STAGE=1` (Option B, contiguous-layout): a cheap grid-cooperative pre-transpose writes each weight's dY/X **once per step** into K-contiguous scratch → the dW reuses the proven `kRingAsync` cp.async ring (the −14.2% fwd/dX win) instead of the scalar transposed-strided gather. **Measured @ d=2048/B=16384, 3 seeds:** stage=0/sk=1 scalar **1889.8 ms** → stage=1/sk=1 contiguous **920.7 ms = 2.05× faster**; vs production stage=0/sk=2 (1925.5 ms) = **2.09×**. Roofline **2.08% → 4.35% (doubled)**. **fp64 PARITY + A/A/A GATE GREEN: 11/11 decoder cells × seeds {42,7,123}.** Production default set to **stage=1, splitk=1** — **SUPERSEDES the Track-C split-K=2 KEEP** (split-K was a scalar-dW grid-fill mitigation; with fast staging the single-CTA dW wins). **Mechanism validated:** the dW was staging-bound (~97% staging) → fixing the *staging* (not pipelining the MMA, the reverted P0) was the lever. **KEY REFRAME: the megakernel's ~2% roofline was a STAGING artifact, not a hardware ceiling.** Next: the same fix on ViT dW (even more staging-dominated → likely a bigger win) + Mamba.

*Maintained going forward: every future apply→gate→verdict lands here with its measured numbers + the mechanistic why.*
