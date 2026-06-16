# Superoptimizer Scoping — go/no-go for a full GPU-kernel superoptimizer on SuperGrok1.5

**Author:** kernel-perf analysis pass, 2026-06-16. CPU/analysis only (no GPU, no builds).
**Status:** decision document. Uncommitted by design (the main loop commits).
**Audience:** the owner, for a go/no-go on building the *full* superoptimizer vs. continuing the
incremental **#24** path (widening `compile.py`'s joint search over hand-authored structural
variants).

---

## 0. TL;DR — the recommendation

**No-go on the full, general GPU-kernel superoptimizer. Continue #24 + hand-author the handful of
named structural levers.** The break-even for a general correctness-gated search engine is *kernel
and architecture breadth*; we have **3 fixed models × 11 optimizers on one arch (sm_90a)**, the
known wins are a *small, already-enumerated set of structural levers* (P0/P1/M0/M1, dW staging), and
**we already own ~80% of a practical superoptimizer** in `compile.py`: a self-configuring
Optuna-TPE search over structure-macros × PTX-hints × ptxas-flags, fp64-gated, scored by real
d=2048 timing, with a *wired* XGBoost/ridge learned cost model and *real-but-off-by-default*
generative back-ends (OpGraph synth-codegen, libclang+islpy polyhedral, CUTLASS/CK emitters).

The thing a full superoptimizer would add — *automatic discovery of structural rewrites of a fused
training megakernel* — is exactly the thing every surveyed autoscheduler **cannot express** (they
target inference / single ops, not a persistent fwd+bwd+optimizer megakernel behind a hand-built
grid barrier), and the thing our concrete bottleneck **does not need** (the dW lever is a known
gmem-layout change a human already designed and partially wired: `SG_TUNED_DEC_DW_STAGE=1`).

**If you want to spend anything here, spend it on the one cheap, measurable probe in §6** (re-express
ONE decoder GEMM in CuTe-DSL/Triton and race its autoscheduled result against our hand-tuned engine
on the *isolated* GEMM). That probe costs ~3–5 person-days and either (a) reveals a real
auto-scheduling gap worth more investment, or (b) confirms hand-authoring + #24 captures the value —
without a multi-month blind commit. **A STOKE pilot on the inner loop is a no-go** (§4): the win is
structurally above the instruction window, and NVIDIA has no assembler to search at the SASS level.

---

## 1. Where we actually are (grounded)

The numbers and structure below are read directly from the codebase; citations are `file:line`.

### 1.1 The system
- **ONE persistent CUDA megakernel per (model × optimizer) cell** fuses the entire training step
  (P0 zero → P1 fwd+bwd → P2 deterministic grad reduce → P3 optimizer tail), separated only by a
  **hand-built sense-reversing `GridBarrier`** (`csrc/fused/megakernel_common.cuh:120-153`). The
  barrier **needs no cooperative launch** so it scales past the cooperative-grid CTA cap
  (`megakernel_common.cuh:28-29`); the host launches exactly `gridDim.x == #SMs` persistent CTAs
  (`:59`). **This is the single most important structural fact for IR selection** (§2): a
  *host-launched* collective (CUTLASS device::Gemm, the CuTe device GEMM API, an MLIR/Linalg-lowered
  op) cannot be invoked device-side *inside* this persistent grid.
- **33 cells** = 3 models (`transformer_decoder`, `vit`, `mamba3`) × 11 optimizers
  (`fused_dispatch_table.inc`), generated from ONE structural template
  `l3_megakernel<Model, Optimizer>` by `grokking_optimizers/megakernel_codegen.py` (you do **not**
  hand-write 33 kernels — the demo TU is the template, re-instantiated per cell).
- The GEMM engine is **hand-rolled in-kernel ss-wgmma** (`csrc/backends/cuda/sm_90/wgmma.cuh`),
  consumed via thin orientation wrappers (`model_stage_decoder_tc.cuh:717+`) and an unpipelined /
  cp.async-ring tile loop (`tc_gemm_block_unpipelined`, `:508-715`).

### 1.2 Current performance and *why* it's capped
- decoder d=2048 = **2.0%** of the 989 TF/s bf16 roofline; ViT = **0.19%**; Mamba-3 can't place at
  d=2048 (smem-bound) (`PERF_ANALYSIS.md:9`).
- **Structurally capped, not knob-bound:** decoder is the same 2% at d=1024 and d=2048
  (scale-invariant cap, `PERF_ANALYSIS.md:11`); the kernel-knob track hit a 3-consecutive-not-positive
  STOP and is **knob-converged** at TILE 128×128/IL=2/STAGES=2/cp.async-ring
  (`OPTIMIZATION_LEDGER.md:44`).
- Root cause: (1) the **dW GEMM is staging-bound** — ~97% of each dW K-step is the transposed-strided
  scalar operand gather, ~3% is the wgmma (`OPTIMIZATION_LEDGER.md:52`, `PERF_ANALYSIS.md:19`);
  (2) the fused step is mostly **non-GEMM fp32 CUDA-core work** the roofline numerator ignores.
- **The P0 pipelining lever was tried and reverted** (`OPTIMIZATION_LEDGER.md:52`): fp64 gate passed
  (11/11) but perf regressed −11.6% — pipelining the MMA is the wrong lever when the bottleneck is
  staging. The redirect (documented, not yet landed): faster *staging* via contiguous K-major
  pre-transpose (`SG_TUNED_DEC_DW_STAGE=1`, `model_stage_decoder_tc.cuh:114-150`) or
  TMA-with-transpose.

### 1.3 What `compile.py` already is (this is decisive for ROI)
`compile.py` is **already a practical superoptimizer over its macro/flag space** — not a stub. Verified:

| Capability | Status | Evidence |
|---|---|---|
| Self-configuring search space | **REAL, used** | `_discover_kernel_knobs` scans committed kernels for `#ifndef SG_TUNED_X`/`#define` guards → a new knob enters the space with zero edits; producer→consumer liveness audit drops dead macros. Cartesian ~10⁶–10¹³/arch. |
| Joint structure × PTX-hint × ptxas-flag | **REAL, used** | `TILE_M/N`, `*_DW_SPLITK`, `async_depth`, `cluster_shape`, `swizzle`, `wgmma_shape`, `prod_regs/cons_regs`, warp-spec → `-DSG_TUNED_*`; plus real `--maxrregcount`, `-Xptxas --opt-level`, `--allow-expensive-optimizations`. |
| Search algorithm | **REAL, used** | Optuna **TPE Bayesian** (`run_bayesian`, `TPESampler(multivariate=True)`) + **top-K ±2-step local refine** (`topk_refine`) + multi-criterion `BayesianEarlyStopper`. Opt-in exhaustive/Cartesian mode also exists. |
| Learned cost model (Ansor-style surrogate) | **REAL, wired, default-ON** | `class CostModel`: XGBoost → sklearn-GBR → numpy-ridge fallback; **vetoes candidates inside the TPE loop** (`ms_pred > threshold and high_confidence` → `cost_model_pruned` → Optuna `PRUNED`); cold-start floor of 100 real timings; online-fit + warm-start + persisted; multi-fidelity 2nd prune stage. |
| Correctness gate (in-loop) | **REAL** but **NOT fp64** | strict-math (`-DSG_STRICT_MATH=1`, fast-math-OFF) AOT build's own fp32 output is the oracle; `np.allclose` per-dtype tol + a `np.array_equal` 3× determinism check. Generated/transformed origins are **ineligible to win unless they recorded an oracle PASS** (`_VALIDATION_REQUIRED_ORIGINS`). |
| Correctness gate (fp64 ground truth) | **REAL, separate layer** | `tests/hw/test_l3tc_tail_gate.py` (+ `*_oracle.py`, `*_kernel_mirror.py`): bf16-TC kernel vs pure-**fp64** canonical refs (`ref_<opt>_step`), rel-tol 1e-4 (SAM surfaces 2.5e-2/3e-2), + A/A/A bit-identical determinism via `torch.equal`. **This does NOT call the autotuner — it gates the production step directly.** |
| Measurement | **REAL, used** | `TimingWorker` persistent CUDA-context subprocess + watchdog + multi-GPU pool; CUDA-graph median at `--size 2048`; `.so` reuse keyed on sources+flags hash (skips ~125s recompiles, `compile.py:7589`); sqlite `compiled_cache.db`; resumable Optuna study. |
| Generative codegen (the "real superoptimizer") | **REAL but OFF by default** | OpGraph synth-codegen (fused AdamW/flash-attn/Newton-Schulz patterns → real CUDA elementwise/reduce/GEMM emitters), polyhedral (libclang AST + islpy deps, honest conservative fallback), CUTLASS/CK GEMM emitters. Gated `enable_*=false`; exercised by self-tests; **not proven on the production megakernels**; origin-tagged only when a transformed source actually compiles. |
| Objective | **roofline-aware** | optimization target is *distance-to-roofline* (`tuning/roofline.py`: achieved FLOP/s = FLOPs/wall), tracked at 10-step granularity; the surrogate predicts `median_ms`. |

**Honest gaps in `compile.py` a "full superoptimizer" would need to close:** (1) the in-loop oracle is
strict-AOT-fp32 self-consistency, not fp64 — fp64 ground truth lives only in the hardware test layer;
(2) there is **no e-graph / equality-saturation / true program-synthesis search over semantics** — the
"generative" layer is a *fixed pattern library* + polyhedral reschedule, not search over rewrites;
(3) the generative back-ends are unproven at scale and off; (4) `compiled_cache.db` is empty in this
checkout (no GPU builds persisted).

### 1.4 What IRs are already vendored
- **CUTLASS v3.6.0 + CuTe** — PRESENT and **used**, but only on the **host-launched** path:
  `csrc/backends/cuda/sm_90/mma.cuh` (`#ifdef WITH_CUTLASS`) builds a Sm90 `GemmUniversalAdapter`
  (TMA+WGMMA) for Muon Newton-Schulz and SuperGrok2 `dt_proj`. The header itself states the boundary:
  "The CUTLASS GEMM is a self-contained kernel that OWNS its TMA mainloop … Pairing the elect/mbarrier
  primitives with TMA belongs to the … hand-written TMA path, NOT here" (`mma.cuh:148-160`). It is
  **explicitly rejected for the persistent-megakernel path** (`wgmma.cuh:14-18`).
- **Triton** — ABSENT (no `@triton.jit`/`tl.`; not a dependency).
- **MLIR / Linalg / IREE / TVM / Ansor / OpenXLA** — ABSENT for kernel gen (only an XLA/Pallas
  scaffold for the TPU path).
- **Polyhedral (islpy / libclang)** — PRESENT but only as `compile.py`'s opt-in, off-by-default
  scaffold; soft deps, not in `install_requires`; not wired into the sm_90 megakernels.

---

## 2. IR options for Level-2 structural auto-scheduling

The question that dominates every row: **can it express our FUSED TRAINING megakernel** — one
persistent kernel that does fwd + bwd + dW + the optimizer step, with a deterministic cross-CTA grad
reduce, behind a hand-built grid barrier? This is *not* the workload any of these tools is built for.
Autoschedulers target **inference / single ops / forward graphs**; a fused fwd+bwd+optimizer
persistent megakernel with cross-CTA reductions and an in-kernel optimizer is **out of distribution
for all of them**. Treat this as the central obstacle, not a footnote.

### 2.1 CuTe-DSL (CUTLASS-native, Python)
- **Substrate match:** *best of the four* — CuTe is the native abstraction over exactly our wgmma/TMA
  substrate, and we already vendor CUTLASS 3.6 + CuTe and use it (host-side) in `mma.cuh`. Layouts,
  swizzles, TMA descriptors, warp-specialized mainloops are first-class.
- **Can it express the fused training megakernel?** **No, not as one persistent device kernel.** The
  CuTe-DSL / CUTLASS programming model is *collective-kernel-shaped*: you author a GEMM (or a fused
  epilogue) that CUTLASS launches and that owns its own mainloop/epilogue. Our megakernel is the
  *inverse*: a persistent grid we own, into which GEMMs are *embedded* as device-side tile loops
  between grid barriers, with a cross-CTA reduce and an optimizer phase living in the same launch.
  There is no CuTe construct for "persistent grid that runs a backward pass and an Adam step between
  two grid barriers." You could express the *individual GEMMs* in CuTe and the *epilogue fusion* in a
  CUTLASS collective epilogue — but not the whole step, and not without re-introducing the
  host-launch boundary the persistent design exists to avoid.
- **Autoscheduler maturity on Hopper/wgmma/TMA:** CuTe gives you the *vocabulary* and CUTLASS ships
  expert-tuned Hopper kernels, but CuTe-DSL is **not an autoscheduler** — there is no Ansor-style cost
  model or search; you (or CUTLASS's `CollectiveBuilder` heuristics) pick the schedule. So "adopt
  CuTe-DSL" buys *expression*, not *automatic search* — and we already have the search (`compile.py`).
- **Rewrite cost:** to get value inside the megakernel you'd re-express the in-kernel GEMM tile loop
  in CuTe device code and prove it composes with our grid barrier / persistent loop / smem budget —
  **weeks per GEMM family**, and the persistent-launch composition is unproven (likely the hardest
  part). To get value *outside* (host-launched per-GEMM), you abandon fusion.
- **fp64 gate integration:** clean — a CuTe GEMM has a well-defined I/O contract; run it through the
  same `test_l3tc_tail_gate` fp64 oracle. But bit-exact A/A/A requires the same ascending-k reduction
  order; CuTe's split-K/stream-K reorders the sum and would fail the determinism gate as-is.
- **Verdict:** highest-leverage IR for a *probe* (it matches our substrate and is already vendored),
  but it **cannot express the fused training megakernel** and is not itself a search engine.

### 2.2 Triton
- **Substrate match:** good for elementwise/attention/forward GEMMs; Hopper wgmma/TMA support exists
  but lags hand-tuned CUTLASS for the last 2× on big GEMMs.
- **Can it express the fused training megakernel?** **No.** Triton's model is grid-of-independent-tiles
  per `@triton.jit` kernel; it has no persistent-grid-barrier / cross-CTA-reduce-in-one-launch /
  in-kernel-optimizer construct. You'd write *many* Triton kernels (one per phase) and launch them
  separately — i.e. **un-fuse** the megakernel, which is the opposite of the design.
- **Autoscheduler maturity:** Triton's autotuner is a **config grid search** (`@triton.autotune` over
  `num_warps`/`num_stages`/block sizes) — *less* capable than what `compile.py` already does (TPE +
  learned surrogate + multi-fidelity). No structural search.
- **Rewrite cost:** ABSENT today (new dependency); per-kernel rewrites are days each but the
  *fusion-breaking* is architectural.
- **fp64 gate:** same story as CuTe — clean per-kernel contract; reduction-order determinism is the
  catch.
- **Verdict:** viable only as a *forward-GEMM / attention probe*; structurally cannot host the fused
  training step; its search is weaker than ours.

### 2.3 MLIR-Linalg (IREE / custom pipeline)
- **Substrate match:** Linalg can represent GEMM/conv/elementwise and there are Hopper lowerings, but
  driving wgmma/TMA well end-to-end is research-grade and brittle.
- **Can it express the fused training megakernel?** **No, and worst rewrite cost of the four.** Linalg
  models tensor algebra (a *forward* dataflow graph); backward + the optimizer update + a deterministic
  cross-CTA reduction + a persistent grid barrier are not Linalg ops. You would be authoring a custom
  dialect + lowering pipeline for a one-of-a-kind kernel shape — a multi-quarter compiler project with
  no guarantee the Hopper codegen beats our hand-tuned engine.
- **Autoscheduler maturity on Hopper/wgmma/TMA:** the least mature for *persistent fused* shapes;
  IREE/Linalg autoscheduling targets inference graphs.
- **fp64 gate:** the gate would sit at the whole-kernel boundary (same fp64 oracle), but you'd first
  have to *build* the kernel through the pipeline before you could gate it.
- **Verdict:** not worth it for 3 fixed models. This is the option whose break-even most clearly
  requires breadth we don't have.

### 2.4 Polyhedral (Pluto-style; islpy/libclang already scaffolded in `compile.py`)
- **Substrate match:** polyhedral excels at *affine loop-nest reschedule* (tiling, fusion, skewing,
  interchange) of static-control loops — and `compile.py` already has the libclang+islpy plumbing
  (off by default).
- **Can it express the fused training megakernel?** It does not *replace* the kernel; it *reschedules
  loops within* it. The hot loops are wgmma tile loops whose schedule is constrained by (a) the wgmma
  `.aligned` collective requirement (all 128 warpgroup threads issue uniformly, `wgmma.cuh:423`),
  (b) the ascending-k determinism contract, and (c) smem capacity. Polyhedral reschedule that respects
  all three reduces to choices our macro space already covers (STAGES, INTERLEAVE, split-K).
- **Autoscheduler maturity on Hopper/wgmma/TMA:** polyhedral tools have **no model of wgmma/TMA**;
  they schedule scalar affine loops, then you still hand-map to the tensor-core substrate. The match
  to our substrate is weak precisely where the perf lives.
- **fp64 gate:** any reschedule that reorders the fp32 accumulation breaks bit-exact A/A/A; only
  order-preserving reschedules pass — a narrow set.
- **Verdict:** the scaffold is fine to keep as an *experiment* on the non-GEMM fp32 epilogue passes
  (P1 fusion candidates), but it cannot reach the GEMM bottleneck and isn't a general win.

### 2.5 IR summary
None of the four can express the fused training megakernel as one persistent kernel; **CuTe is the
only one whose substrate matches ours and that we already own**, and even it is collective-shaped,
not persistent-megakernel-shaped, and is not itself a search engine. The honest conclusion: the
*expression layer* is not our gap — our gap is structural rewrites, and no surveyed autoscheduler
discovers those for this kernel shape.

---

## 3. Search method

The space is **structure × realization × flags**:
- *structure*: tile shapes, split-K, interleave, pipeline depth, staging method, warp-spec, fusion of
  the fp32 epilogue passes, the dW operand layout;
- *realization*: how each structural choice is lowered (cp.async ring vs synchronous stager vs
  TMA-with-transpose);
- *flags*: PTX hints + ptxas (`--maxrregcount`, opt-level, expensive-opts).

Methods, judged against what we already have:

| Method | What it is | Status here | Verdict for us |
|---|---|---|---|
| **Measured search (Optuna-TPE + local refine)** | Bayesian global + ±2-step neighbor refine, every candidate timed | **Already built & default** (`run_bayesian`/`topk_refine`) | The right backbone; keep it. |
| **Learned cost model (Ansor-style)** | surrogate predicts runtime, prunes candidates before building | **Already built, wired, default-ON** (`CostModel`, XGBoost/ridge, cold-floor 100, multi-fidelity) | Already have it; the value is *fewer ~125s builds*, not new wins. |
| **Stochastic / MCMC over programs (STOKE)** | MCMC over instruction sequences with correctness+perf cost | **ABSENT** | No-go for our bottleneck (§4): win is above the instruction window; no SASS assembler. |
| **Exhaustive** | enumerate the Cartesian space | opt-in mode exists | Intractable for the full megakernel; fine for tiny pinned sub-spaces. |

**How correctness-equivalence is enforced (the fp64 oracle):** today, two-tier — the in-loop
strict-AOT-fp32 self-consistency + 3× determinism gate inside `compile.py`, and the authoritative
**fp64** parity + A/A/A gate in `tests/hw/test_l3tc_tail_gate.py` driving the production step. **A
full superoptimizer that generates novel structure would need the fp64 oracle wired *into the search
loop*** (not just the post-hoc hardware gate), because in-loop strict-fp32 self-consistency does not
prove fp64 correctness of a *newly synthesized* variant — it only proves it matches the strict build
of the *same source*. That wiring is a real, bounded piece of work and a prerequisite for trusting any
generative back-end's winners.

**The decisive constraint across every method:** bit-exact A/A/A determinism requires preserving the
**ascending-k fp32 reduction order**. fp32 add is non-associative, so *any* search that reassociates
the sum (a faster reduction tree, stream-K, a different split-K reduce order) changes bits and fails
the determinism gate. This collapses the *legal* realization space to **transport-only** rewrites
(move the same operands to the same smem in the same k-order). Relaxing the oracle to ULP-tolerance
to unlock reassociation would re-litigate the determinism contract the whole project rests on — and
would *still* not help the dW, whose bottleneck is the load pattern, not the sum.

---

## 4. Component-level STOKE on the hot K-step inner loop

The full megakernel is far too big for STOKE; the hot **dW K-step inner loop** (staging + wgmma, ~tens
of instructions) is the only plausibly-STOKE-able unit. **Verdict: no-go.** Grounded reasons:

1. **The win is structurally above the instruction window.** The dW is staging-bound (~97% staging /
   ~3% wgmma, `OPTIMIZATION_LEDGER.md:52`) and the documented lever is to **stop doing the strided
   gather at all** — pre-transpose dY/X into a contiguous K-major gmem scratch so rows become
   K-contiguous + 16B-aligned, flipping `DecTileSrcIsGmem` true and letting the *already-validated*
   cp.async ring stream them (`SG_TUNED_DEC_DW_STAGE=1`, `model_stage_decoder_tc.cuh:114-150`,
   transpose pass `:1660-1692`), or TMA-with-transpose. Both are **changes to the *input* side of the
   I/O contract** (a different gmem layout + a separate pre-transpose pass). STOKE optimizes a fixed
   instruction sequence *for a fixed I/O contract*; it cannot say "first run a different kernel to
   rewrite memory, then a different sequence becomes optimal." The lever is out of its search space by
   construction — and a human already found it and partially wired it.
2. **No SASS substrate.** NVIDIA SASS is **not officially assemblable** (no public assembler, no
   documented `ptxas` round-trip; the toolchain here references only ptxas flags, no
   `nvdisasm`/`cuobjdump`-edit path). STOKE's premise (mutate assembly → reassemble → measure) has no
   substrate on NVIDIA. The realistic levels are **PTX** (assemblable, but ptxas re-schedules /
   re-allocates so you don't control final SASS — and the body is already near-minimal PTX: single
   inline-asm wgmma, single-op barriers/fences, `LDG.U16`/`STS.U16` with no instruction-count fat) or
   **CUDA-C/intrinsic source** (real leverage — but the leverage is *choosing a staging strategy*,
   which is exactly the discrete macro space `compile.py` already sweeps: `STAGES`, `INTERLEAVE`,
   `SPLITK`, `DW_STAGE`, `kRingAsync`).
3. **The exact oracle the project requires forbids the interesting rewrites.** A unit oracle is cheap
   and would be *bit-exact* (feed a known dY/X tile, compare the fp32 accumulator) — but exactness +
   ascending-k means only **transport-only** rewrites are legal; STOKE can't touch the arithmetic, and
   the arithmetic isn't the bottleneck.
4. **The one local coalescing micro-rewrite STOKE would plausibly find was already tried and reverted**
   (m-major HBM coalescing introduced a compensating smem bank conflict → neutral-to-worse). The fix
   has to be holistic (layout + transport together), i.e. structural, not peephole.

STOKE could only pay off if pointed at a **compute-bound / instruction-count-bound** inner loop. We
don't have one on the critical path — our hot loop is latency-bound on an uncoalesced gather.

---

## 5. Effort / ROI

Person-time is rough order-of-magnitude (one experienced GPU+compiler engineer). "Gain vs #24" is the
*expected marginal* gain over continuing the incremental path, **given the bottleneck is a known
structural lever and `compile.py` already searches the flag/macro space**.

| Option | Person-time | Expected gain vs #24 | Risk | Notes |
|---|---|---:|---|---|
| **#24 (baseline): widen `compile.py` joint search over hand-authored structural variants** | ongoing (in flight) | — (the reference) | low | Author each lever as a `SG_TUNED_*` variant; the autotuner searches it; fp64-gated ratchet. Captures P0-redirect (dW staging), P1 epilogue fusion, M0/M1, swizzle, TMA-transpose **as they're authored**. |
| Hand-author the named levers (dW `STAGE=1`, M0 wgmma-projections, P1 fusion, ViT B1) | ~1–3 wk each | **High** — these are the measured 5–15× headroom (`PERF_ANALYSIS.md`) | med (per-lever eng + parity) | This is where the real perf is. Each is bounded, fp64-gated, revertible. |
| **CuTe-DSL probe** (re-express ONE decoder GEMM, race vs hand engine, *isolated*) | ~3–5 days | Diagnostic (info, not perf) | low | The recommended probe (§6). Tells us if an autoscheduler beats our hand-tuned GEMM *at all* before any big bet. |
| Triton probe (forward GEMM / attention only) | ~1 wk | Diagnostic, lower relevance | low | Only touches forward ops; can't host fusion; search weaker than ours. |
| Wire fp64 oracle *into* the search loop + turn on `compile.py` generative back-ends (synth-codegen / polyhedral) on the megakernel | ~3–6 wk | **Low-to-uncertain** | high | Back-ends are real but unproven at scale; the pattern library doesn't contain our bottleneck's fix; e-graph/equality-sat search over semantics doesn't exist here and would be a research build. |
| Full MLIR-Linalg / custom-dialect auto-scheduling of the megakernel | **multi-quarter** | **Speculative** | very high | Cannot express the fused training kernel without a bespoke dialect+pipeline; no guarantee of beating the hand engine. |
| Component STOKE on the dW inner loop | ~2–4 wk | **~0** (likely negative ROI) | high | §4: win is above instruction window; no SASS; oracle forbids the useful rewrites; the local rewrite it'd find was reverted. |

### 5.1 The break-even
A general correctness-gated superoptimizer amortizes its build cost across **kernel/architecture
breadth** — many distinct kernels, multiple GPUs/arches, evolving ops, where re-deriving schedules by
hand is the bottleneck. **Our situation is the opposite of the break-even case:**
- **3 fixed models, 11 optimizers, ONE arch (sm_90a).** The 33 cells already share ONE structural
  template and ONE GEMM engine — a lever fixed once (P0-redirect, M0) propagates to all relevant cells
  by construction (`PERF_ANALYSIS.md:22` "shared engine ⇒ one change fixes both models").
- The headroom lives in a **small, already-enumerated set of structural levers** (P0/P1/P2, M0/M1/M2,
  S1–S3 in `PERF_ANALYSIS.md`), not in an open-ended schedule space a human can't enumerate.
- We **already own the search engine** for the flag/macro/structural-variant space. The full
  superoptimizer's *only* novel capability — automatic discovery of structural rewrites of a fused
  training megakernel — is precisely what no autoscheduler can express for this kernel shape (§2).

So the marginal value of the full superoptimizer over #24-plus-hand-authoring is **low and high-risk**,
while its cost is **high**. #24 + hand-authoring the handful of levers captures most of the value.

---

## 6. Phased plan IF go (start with a cheap probe, not a blind commit)

If the owner wants to *de-risk* the auto-scheduling question before committing, do the **minimal
measurable probe first** and gate expansion on its result.

### Phase 0 — the probe (~3–5 person-days, low risk) [RECOMMENDED if spending anything]
**Re-express ONE isolated decoder GEMM in CuTe-DSL and race its autoscheduled result against our
hand-tuned engine.** Pick the **fwd in_proj GEMM** (clean shape: M = token tile, N = 3d, K = d; *not*
the dW — the dW's lever is layout, not scheduling, so it would mislead the probe). Concretely:
1. Author the GEMM as a standalone CuTe/CUTLASS Sm90 collective (we already vendor CUTLASS 3.6 + use
   it host-side in `mma.cuh`, so the toolchain is in place).
2. Feed it the *same* bf16 operands the in-kernel engine sees; compare the fp32 output against the
   existing `test_decoder_tc.py` micro-gate (bit-match where ascending-k is preserved; ULP-bounded
   otherwise, recorded honestly).
3. Time both *in isolation* at d=2048 with the existing `TimingWorker` harness.

**Decision rule:**
- If CuTe's autoscheduled GEMM is **meaningfully faster** than our hand-tuned engine *on the isolated
  GEMM* → there is a real auto-scheduling gap. Proceed to Phase 1, but scoped to **host-launchable
  GEMM families only** (the fusion question is still unsolved — see §2.1).
- If it is **within noise or slower** (the likely outcome given the bottleneck is staging/layout and
  non-GEMM fp32 work, not GEMM scheduling) → **stop**; the value is in hand-authoring the structural
  levers + #24. Record the result in `OPTIMIZATION_LEDGER.md` as a measured dry-well.

(Optional cheaper sub-probe: a Triton version of the same GEMM, ~1 day, as a second data point. Skip if
Phase 0 already answers the question.)

### Phase 1 — only if Phase 0 shows a real gap (~3–6 wk)
Wire the **fp64 oracle into the search loop** (prerequisite to trust any generated variant), then turn
on **one** `compile.py` generative back-end (start with the CUTLASS emitter, since the probe validated
CuTe) on the **host-launchable GEMM families**, gated by the fp64 oracle + A/A/A. Measure against the
hand engine on 3 seeds at d=2048. Keep only on a measured win (the existing ratchet). **Do not** attempt
the persistent-megakernel-internal auto-scheduling until/unless a host-launched GEMM win is demonstrated
*and* a persistent-launch composition story exists — that remains the hardest, least-proven part.

### Phase 2 — only if Phase 1 wins compound (re-evaluate)
Revisit polyhedral reschedule on the **non-GEMM fp32 epilogue passes** (the P1 fusion candidates), where
affine-loop transforms are actually applicable, and the broader generative search. Re-run the
break-even (§5.1) with real Phase-1 numbers before any multi-month commit.

**Throughout:** every step lands in `OPTIMIZATION_LEDGER.md` with its measured numbers and the
mechanistic why (the existing protocol), so the "big bet" only ever advances on evidence.

---

## 7. The honest one-paragraph answer

For *our* situation — 3 fixed models, 11 optimizers, one arch, a structurally-capped megakernel whose
known wins are a short list of human-identifiable structural levers, and an existing `compile.py` that
is already a fp64-gated, roofline-scored, learned-cost-model-pruned Bayesian superoptimizer over the
flag/macro/structural-variant space — **a full general GPU-kernel superoptimizer is not worth
building.** No surveyed IR (CuTe, Triton, MLIR-Linalg, polyhedral) can even express the fused training
megakernel as one persistent kernel, and the only one matching our substrate (CuTe) is collective-shaped
and not itself a search engine. Component-level STOKE can't reach our bottleneck (the lever is a gmem
layout change above the instruction window, and NVIDIA has no SASS assembler to search). **#24 +
hand-authoring the handful of levers (dW staging redirect, M0, P1 fusion) captures most of the value at
a fraction of the cost and risk.** If you want to spend anything to *confirm* this, run the Phase-0 CuTe
probe (~3–5 days) and let the measurement decide.
