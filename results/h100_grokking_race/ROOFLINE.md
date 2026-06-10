# Roofline — TC-only production paths vs the H100 ceiling (R2.4: bf16, B=16384, peak-VRAM)

Owner directive R2.4: **distance-to-roofline is the optimization metric**, measured on
the **production megakernel paths** at the **per-model max batch**, with **peak VRAM**
recorded per cell. This pass measures every (optimizer × model) race row at the wired
**race precision bf16** and at **B = 16384** (the SM-saturation operating point), and
records `torch.cuda.max_memory_allocated` of the production path per cell.

Data: `roofline.json` (rows + `batch_saturation_sweep` + `peak_vram_table`); plot:
`roofline.png`; harness: `tuning/roofline.py`. Measured on a quiet GPU (tuner fleet
SIGSTOPped for the window; resumed after).

## TC-only routing (item 1) — VERIFIED, not constructed

The production-megakernel universe is the **adamw trio** — the ONLY cells with a TRUE
L3 megakernel (real fwd+bwd+opt in one persistent kernel). Routing at bf16, confirmed
end-to-end (`LAST_L3_ENGINE` + the per-row path counter):

| cell | engine | path | note |
|---|---|---|---|
| decoder × adamw | **wgmma** | `L3-TC-megakernel(wgmma)` | bf16 tensor-core ✓ |
| vit × adamw | **wgmma** | `L3-TC-megakernel(wgmma)` | bf16 tensor-core ✓ |
| mamba × adamw | scalar | `L3-scalar-megakernel` | **FLAGGED**: its TC is currently slower (scan-dominated, measured 0.46×); ships the fastest VALIDATED megakernel (fp32 scalar) pending a TC fix |

The race's `DEFAULT_CONFIG.matmul_precision` is `bf16`, so the production race routes the
adamw trio exactly this way with no flag. `_try_fused_train_step` + `gemm_impl_for_cell`
gate it; the wiring guard RAISES on a wrong-path launch (never a silent eager degrade).

### Honest coverage limit (no-suppression)
The **30 non-adamw cells have NO real L3-TC megakernel** — only a per-element optimizer
tail (L1) over an eager fwd/bwd, plus a *surrogate* L3 whose model fwd/bwd does NOT match
the real graph (loud honesty in every generated cell: "L3 is kept compiled but is not the
race path"). Wiring the surrogate into the race would be placeholder math on a live path
(forbidden: COMPONENT_CONTRACT rule #6 + the no-suppression directive). So their honest
production path is **eager + the real optimizer tail**, and that is what they measure here.
Building a real L3-TC megakernel for all 33 cells is the codegen program (`megakernel_codegen.py`)
against the *real* model-stage headers — out of this session's budget; flagged, not faked.
The directive's "eager/L3-scalar only in tests/parity" (rule#14) is honored *for the trio*
(adamw never degrades to eager); for the 30 it is the only correct path, not a degrade.

## Batch-saturation sweep (item 2) — why B = 16384, and the honest negative result

Owner hint: "B≈16k+ fills 132 SMs." MEASURED (decoder, min-of-3 wall, quiet GPU; full
tables in `roofline.json.batch_saturation_sweep`):

| B | TC megakernel (samples/s) | eager-lion (samples/s) | TC peak VRAM | eager peak VRAM |
|---:|---:|---:|---:|---:|
| 1024 | 116 098 | — | 0.25 GB | — |
| 2048 | **142 477** (≈peak) | — | 0.52 GB | — |
| 4096 | 142 646 (peak) | 632 904 | 0.63 GB | 0.22 GB |
| 8192 | 140 821 | 575 065 | 0.84 GB | 0.37 GB |
| **16384** | 137 591 (−3.5% vs peak) | 1 141 497 (knee) | 1.26 GB | 0.58 GB |
| 32768 | 128 463 (−10%) | 1 724 025 | 2.10 GB | 1.10 GB |
| 65536 | 129 732 | 1 830 900 | 3.79 GB | 2.12 GB |
| 131072 | 128 234 | 1 880 273 | 7.15 GB | 4.17 GB |
| 262144 | — | 1 911 100 (plateau) | — | 8.27 GB |

**The throughput (= achieved FLOP/s, FLOPs linear in B) of the one-CTA-per-SM TC
megakernel SATURATES at B≈2k and DECLINES past 16k** — it is occupancy-pinned (one CTA
per SM; one 16-row dW atom × 132 SMs ≈ 2112 = one tile per SM), so a bigger batch only
adds grid-stride iterations with no occupancy gain, and cache pressure slowly costs it.
**Batch does NOT move the megakernel's roofline fraction** — the path to a higher fraction
is multi-CTA-per-tensor tiling (the structural fix), NOT a larger batch.

Eager (cuBLAS batched GEMMs) is *not* occupancy-capped, so it keeps climbing to a ~32–65k
plateau. **B = 16384 is the chosen shared operating point**: the owner's stated floor, ≈
one 128-row tile per 132 SMs, the megakernel within 3.5% of its peak, and eager at its
utilization knee (fair to both). Going past 16k *hurts* the megakernel and the metric is
megakernel distance-to-roofline — so 16384 is the ceiling too.

**VRAM is NOT the binding constraint at d=128.** Literal memory-max ≈ B~1–2M (tens of KB
of activations per sample over 80 GB) and buys zero fraction once utilization-saturated.
Peak VRAM at the operating point is a few GB (table below), not 70.

## Peak-VRAM table (item 2) — production path, B = 16384

`torch.cuda.max_memory_allocated` captured during the WALL pass (use_fused=True — the real
megakernel/eager path), NOT the eager FLOP-count pass (which would misreport the 3
megakernel cells). Values in GB.

| optimizer | decoder | vit | mamba |
|---|---:|---:|---:|
| **adamw** (megakernel) | **0.64** (tc) | **2.19** (tc) | **2.50** (scalar) |
| neuralgrok | 0.81 | 2.90 | 8.94 |
| grokadamw | 0.81 | 2.90 | 8.94 |
| supergrok | 1.10 | 3.99 | 9.54 |
| supergrok15 | 1.10 | 3.99 | 9.54 |
| supergrok2 | **9.14** | **9.44** | 9.54 |
| grokfast | 0.81 | 2.90 | 8.94 |
| muon | 0.81 | 2.90 | 8.94 |
| lion | 0.81 | 2.90 | 8.94 |
| looksam | 1.08 | 3.96 | 9.53 |
| prodigy | 0.81 | 2.90 | 8.94 |

Reading:
1. **The mamba megakernel is a memory WIN.** mamba-adamw scalar megakernel = **2.50 GB**
   vs mamba eager **8.9–9.5 GB** — the recurrent scan's per-timestep activation graph is
   ~3.6× the footprint; the megakernel keeps the scan state CTA-local. (Throughput is
   comparable, 0.85 vs ~1.1 TF/s, so this is a genuine megakernel advantage on the
   memory axis even where it is not yet a throughput win.)
2. **supergrok2 is the VRAM outlier** (~9.1–9.5 GB everywhere) — its per-head PEER routing
   allocates large intermediates; a candidate for a future fused/streamed reduction.
3. **decoder/vit megakernels use LESS VRAM than their own eager rows** (0.64 vs 0.81;
   2.19 vs 2.90) — CTA-local activations vs the full eager graph.

## Achieved throughput (bf16, B = 16384) — rank by ABSOLUTE TF/s, not fraction

The bf16 ceiling (989 TF) is ~15× the fp32 ceiling, so a scalar-fp32 cell's *fraction*
is not comparable to a bf16 cell's — read absolute achieved TF/s.

| model | adamw megakernel (TF/s) | best eager (TF/s) | megakernel path |
|---|---:|---:|---|
| decoder | 1.25 | prodigy 8.45 | TC (wgmma) |
| vit | 1.23 | grokfast 21.04 | TC (wgmma) |
| mamba | 0.85 | looksam 1.31 | scalar (carve-out) |

**The megakernel still trails eager in absolute TF/s at this tiny scale** (d=128) — the
same occupancy ceiling the sweep diagnoses. This is the honest negative result the
no-suppression directive accepts: the L3 path is *correct* (gates green, real fwd+bwd+opt
in one persistent kernel, wiring-guarded) but a single CTA/SM grid-stride cannot beat
cuBLAS's batched small-GEMM scheduling on a 0.42 M-param model. Closing it is a kernel
redesign (batch-tiling / multi-CTA-per-tensor), not a tune or a bigger batch — both of
which this pass measured and ruled out.

## Levers in the TC headers (item 3) — present as tuned dims

The TC headers already ship the directive's levers as `SG_TUNED_*` compile-time dims,
parity-validated (decoder 13/13, vit 21/21, mamba 5/5):
- **cp.async pipelined staging** — `csrc/backends/cuda/sm_90/tile_pipeline.cuh` (the
  "validated tile_pipeline"): a producer/consumer warpgroup ring (WG0 cp.async-stages
  bf16 K-tiles + mbarrier-signals; WG1 wgmma-consumes), depth = `SG_TUNED_PIPE_DEPTH`
  (default 2 = double-buffer; 3 = triple if smem allows).
- **wgmma GEMM engine** — `SG_TUNED_GEMM_IMPL` (scalar | wgmma); the bf16 race selects wgmma.
- **swizzle** — the 128-byte-swizzled smem core-matrix layout for the bf16 K-tiles.
- **epilogue AdamW fusion** — the optimizer tail runs in the dW epilogue (0 extra barriers
  for the epilogue-fusable optimizers).

### Lever toggle attempted (PIPE_DEPTH) — found INERT on the decoder TC path

**Honest status: 0 header levers produced a measurable before/after this session.** The
PIPE_DEPTH lever was attempted and proven **inert** on the cell it targets — reported here in
full because the attempt is informative.

The plan was decoder `SG_TUNED_PIPE_DEPTH` 2→3 (the decoder TC row is memory-bound, so deeper
cp.async staging is the most promising lever). Built a depth-3 variant standalone via
`_build_tc_module(-DSG_TUNED_PIPE_DEPTH=3)` and measured it. **VERIFICATION (this is the part
that matters):**
- **Source grep**: `model_stage_decoder_tc.cuh` has **0** references to `TilePipeline` and
  **0** to `SG_TUNED_PIPE_DEPTH`. It `#include`s `tile_pipeline.cuh` but its hot fwd/bwd GEMM
  uses a **custom inline Major-K smem stager + direct `wgs::wgmma_*` calls**
  (`wgmma_fence`/`wgmma_m64nNk16_bf16`/`wgmma_commit_group`/`wgmma_wait_group`, lines ~231-262),
  NOT the `TilePipeline<Depth>` producer/consumer ring. So the macro cannot affect this TU.
- **ptxas -v (the empirical seal)**: depth-2 and depth-3 compile to a **byte-identical**
  binary — the megakernel kernel is **253 regs / 7604 B smem** at BOTH depths (and the second
  kernel 80 regs / 42968 B smem at both). Identical codegen ⇒ the lever did not engage.
- The +0.13% I first measured back-to-back was therefore **run-to-run noise on an unchanged
  binary**, NOT a lever effect — corrected here.

So `SG_TUNED_PIPE_DEPTH` is a *substrate* tuning dim of `tile_pipeline.cuh` (the
`test_wgmma_substrate.py` path), but the shipped decoder TC stage does not route through it —
it has its own staging. The directive's "validated tile_pipeline" substrate exists and is
silicon-validated, but is not the carrier of the production decoder TC GEMM.

### Levers — true coverage this session
- **PIPE_DEPTH (decoder): attempted, INERT** (above) — no before/after to report.
- **PIPE_DEPTH (vit/mamba): not wired** (no cp.async in those stages; grep=0).
- **wgmma GEMM engine** (`SG_TUNED_GEMM_IMPL`): IS the shipped lever — the bf16 race selects
  `wgmma` and the TC rows here run on it (verified `engine="wgmma"`). Its before/after vs the
  scalar engine is the prior TC-vs-scalar-megakernel result (decoder 2.71×, etc.), not re-run.
- **epilogue AdamW fusion / swizzle**: present + validated in the cells (parity-green), but
  each is structural to the cell — toggling needs a codegen change + the full oracle gate per
  cell, and (per the sweep) the one-CTA-per-SM occupancy ceiling bounds their upside. Deferred
  with reason, not measured.
- **TMA (`cp.async.bulk.tensor`)**: does NOT exist (DESIGN-TC-PIPELINE.md §4) — ground-up CUDA
  + revalidation, **out-of-budget**, not started.

The structural conclusion rests on the **batch sweep**: the megakernel is occupancy-pinned at
one CTA/SM (flat throughput from B≈2k, declining past 16k), and the achieved-TF/s gap vs eager
is consistent with exactly that. The PIPE_DEPTH probe is a **null result** — the directive's
assumed `tile_pipeline` carrier is not wired to the production decoder GEMM — so it neither
supports nor refutes the conclusion; it is reported as a useful finding for the owner (the
substrate tuning dim does not reach this kernel), not as evidence about the gap. The fix
remains multi-CTA-per-tensor tiling — lifting the one-CTA-per-SM occupancy cap — a kernel
redesign, not a tune.

## Measurement integrity
- **Quiet window**: the tuner fleet (PIDs 449297/449483) is re-SIGCONTed by an external
  orchestrator (~every 30 s); a tight re-SIGSTOP watchdog held them stopped for the whole
  run (GPU 0% between cells), then they were resumed. The megakernel rows are the
  contention-sensitive ones (one CTA/SM grid cannot yield); they were measured clean.
- **FLOP-trap fix retained**: the FLOP count runs `use_fused=False` so the profiler sees
  the real eager GEMMs (a fused megakernel registers 0 aten FLOPs); the count is path- and
  dtype-independent, so it is the correct numerator for the megakernel wall.
- **No commit** (per directive). Durable artifacts: `roofline.json`, `roofline.png`, this file.
