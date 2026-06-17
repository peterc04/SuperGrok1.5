# Level-2 Superoptimizer — the GO plan (task #27)

**Status:** active design. Supersedes the **recommendation** of `SUPEROPTIMIZER_SCOPING.md` (#25) —
**not** its constraints. The owner reviewed the scoping doc's measured *no-go* and chose **(b): make
the Level-2 superoptimizer REAL**, correctness-gated. This doc is the engineering plan that honors
that decision while respecting the constraints the scoping pass established as *physics, not
preference*. CPU/analysis authoring; every landed step rides the fp64-gated ratchet
([[feedback-patch-protocol]]).

> **Back-end status (RESOLVED — evidence-backed map, agent `a69400abba8373c4d`):** all five
> generative back-ends are *structurally* wired (origin-tagging, the winner source-swap, the #16
> validation gate are REAL + correct) but **non-functional on the real path today**: the
> `grokking_optimizers.codegen` emitter module **does not exist** (ghost import → the production build
> bypasses it, globbing hand-written TUs via `_resolve_sources` L10492; `enable_emitter` defaults True
> but hits the `except` at L14646 → macros-only, `_emitted_sources` never populated); the synth **GEMM
> is an explicit stub** (`_MMA_NATIVE_LOADS_WIRED=False` L30178 → scalar triple-loop; the native wgmma
> bodies are dead + bogus); **polyhedral is a toy** (identity-copy without libclang; fragile token-join
> + unsound `(0,)` deps with it; and unreachable anyway — its hook reads the `_emitted_sources` the dead
> emitter never sets); **CUTLASS/CK are real emitters but host-launch-only** (`GemmUniversalAdapter`
> owns its grid — C2) AND never origin-stamped, so they can't win. Soft deps (libclang/islpy/cutlass)
> all absent; **no trial record has ever carried a generated origin** (nothing has been generated, let
> alone won). **What IS real and load-bearing:** the elementwise/reduce/scan lowerers emit *compilable
> CUDA* (verified L26292/26305); the winner source-swap (`build_jit` L17847) + the #16 strict-validation
> gate (`pick_winner` L6882, fail-closed) would correctly REJECT today's toy outputs; input synthesis is
> correctly wired. This map **sharpens** Phase C (below); it does not change §0–§2.

---

## 0. What "make it REAL" means here (the re-spec)

The owner's correction reframed the task twice, and both reframings are load-bearing:

1. **NOT "re-derive wgmma from scratch."** The in-kernel ss-wgmma engine
   (`csrc/backends/cuda/sm_90/wgmma.cuh`, consumed via `model_stage_decoder_tc.cuh:717+`) is
   hand-tuned and, per the scoping survey, unlikely to be beaten by any autoscheduler *on the
   isolated GEMM*. Re-synthesizing the GEMM is low-value.
2. **"Max the shared primitive FIRST, then synthesize the FUSION."** The owner explicitly rejected
   naive engine-reuse because *"the engine itself may not be maximal"* — reuse on a sub-maximal base
   propagates the deficit to every fused structure. So Level-2's value is **automatic synthesis +
   correctness-gated selection of the transport/fusion structural rewrites**, composed over a GEMM
   primitive that has *already* been driven to maximality by the #22 structural track.

So the deliverable is: **turn the hand-identified structural levers (dW staging redirect, the P1
fp32-epilogue fusion class, M0 projection-wgmma, the ViT B1 reshape) from manual one-offs into a
searchable + (where tractable) *synthesizable* space that compile.py's generative layer drives and
the fp64 + A/A/A gate certifies.** Level-2 = Level-1's flag/macro search (already real) **plus** a
*generative* structural-rewrite layer that emits new sources, not just new `-DSG_TUNED_*` values.

## 1. The constraints that bind regardless of GO (do not relitigate)

These came out of the scoping survey and are architectural facts. The plan is built *around* them.

- **C1 — Transport-only legality.** Bit-exact A/A/A determinism requires preserving the **ascending-k
  fp32 reduction order** (`SUPEROPTIMIZER_SCOPING.md:232`). fp32 add is non-associative; *any* rewrite
  that reassociates the accumulation (faster reduction tree, stream-K, reordered split-K) changes bits
  and fails the determinism gate. ⟹ the legal generative space is **transport-only**: move the *same*
  operands to the *same* smem in the *same* k-order, faster. Synthesis is constrained to this class.
- **C2 — No IR expresses the persistent fused megakernel.** CuTe / Triton / MLIR-Linalg / polyhedral
  are all collective- or single-op-shaped; none can host "a persistent grid that runs fwd+bwd+dW+the
  optimizer step between hand-built grid barriers" (`:111-201`). ⟹ generative back-ends produce
  **device-side-embeddable tile/epilogue fragments** spliced into the existing megakernel, OR
  **host-launchable GEMM families** — never a replacement kernel. A host-launched collective cannot
  run device-side inside the persistent grid (`wgmma.cuh:14-18`).
- **C3 — fp64 must move into the search loop.** Today the in-loop oracle is strict-AOT-fp32
  self-consistency; fp64 ground truth lives only in `tests/hw/test_l3tc_tail_gate.py` and does **not**
  call the autotuner (`:223-230`). Self-consistency proves a variant matches the strict build of the
  *same* source — it does **not** prove a *newly synthesized* source is fp64-correct. ⟹ wiring the
  fp64 oracle into the search loop is the **hard prerequisite** for trusting any generative winner.
- **C4 — The pattern library must actually contain the bottleneck's fix.** The scoping gap (`:89-94`):
  the generative layer is a *fixed pattern library* + polyhedral reschedule, not search over semantics.
  If the library can't express the dW staging redirect / the P1 fusion, turning it "on" yields nothing.
  ⟹ Phase C's first job is to *extend the pattern library to the transport/fusion rewrites we know we
  want* — synthesis fills the search **within** that class, it does not invent the class.

## 2. Preconditions (Phase A) — must be green before C/D land

**A1 — Max the shared GEMM primitive (the #22 structural track).** Level-2 composes a *maximal*
primitive. Status:
  - decoder fwd/dX deeper cp.async ring — **KEPT** `a89f6f1` (+1.49×, gate-green 11/11×3).
  - decoder dW contiguous K-major staging — **KEPT** (+2.05×).
  - decoder PIPE=2 producer/consumer engine — **tournament in flight** (background build in the main
    worktree). Its KEEP/REVERT verdict fixes whether the maximal decoder primitive is PIPE=1/STAGES=4
    or PIPE=2/Depth-D. Level-2 reuse waits on this verdict for the decoder family.
  - ViT B1-barrier load-imbalance (51%, the #1 ViT lever) — roadmap workflow in flight.
  - Mamba M0 projection-wgmma — scaffold authored + gated OFF (byte-identical), **stashed/deferred**
    (`.perf/M0_mamba_integration_scaffold.patch`); revisit after the decoder/ViT primitives settle and
    the Mamba profiler is wired (the meta-lesson: profile each model before porting a lever).
  *Exit:* each model's GEMM primitive is at a measured local max on 3 seeds, ledger-recorded.

**A2 — Wire the fp64 oracle into the compile.py search loop (C3).** This is the single most important
enabling piece and is *independent of which back-end we turn on*. Concretely: give `BuildSpec` /
`pick_winner` a correctness hook that, for any candidate whose origin is in
`_SOURCE_GENERATING_ORIGINS` (i.e. a *generated/transformed* source, not a flag-only variant), runs
the fp64 parity + A/A/A determinism check (the `run_cell_gate` path already seamed in `af9b720`) and
makes the candidate **ineligible to win** unless it records a fp64 PASS — generalizing the existing
`_VALIDATION_REQUIRED_ORIGINS` / strict-fp32 gate to true fp64. Flag-only variants keep the cheap
in-loop strict-fp32 oracle (they can't change semantics by construction). *Exit:* a deliberately-wrong
synthesized variant is rejected in-loop by the fp64 hook (negative self-test), and a known-good one
passes; self-test count goes up.

## 3. Phase B — the calibration probe (cheap, do it even on GO)

Reframed from the scoping doc's go/no-go probe into a **calibration** under the GO decision: re-express
**one isolated decoder fwd in_proj GEMM** (clean shape M=token-tile, N=3d, K=d — *not* the dW, whose
lever is layout) in CuTe-DSL/CUTLASS Sm90 (toolchain already vendored, used host-side in `mma.cuh`),
feed it the *same* bf16 operands the in-kernel engine sees, gate fp32 output against the
`test_decoder_tc.py` micro-gate (bit-match where ascending-k preserved), time both in isolation at
d=2048 via the existing `TimingWorker`. ~3–5 days.

**What it decides under GO (not whether to proceed, but where synthesis aims):**
- CuTe meaningfully **faster** than the *maxed* hand engine ⟹ a real GEMM-scheduling gap exists →
  Phase C may include host-launchable GEMM-family emission (CUTLASS emitter), scoped by C2.
- CuTe **within noise / slower** (the scoping-predicted likely outcome, since the bottleneck is
  staging/layout + non-GEMM fp32 work, not GEMM scheduling) ⟹ synthesis focuses **entirely on the
  FUSION / transport rewrites**, not GEMM codegen — which is the owner's stated target anyway. Either
  way the probe is informative and bounded; record the number in the ledger as the calibration point.

## 4. Phase C — the real generative Level-2 (sharpened by the status map)

The map reframes Phase C decisively: **the synth GEMM is a bogus stub, but the elementwise/reduce
lowerers are REAL and emit compilable CUDA.** That is exactly the owner's re-spec — *do not synthesize
the GEMM* (reuse the maxed hand primitive, §0), *synthesize the FUSION* (the fp32 epilogue passes,
which are elementwise/reduce). So Phase C deliberately **sidesteps `_MMA_NATIVE_LOADS_WIRED`/the GEMM
synth entirely** and rides the working lowerers + the already-real source-swap/#16 gate. The generative
layer must (a) emit **transport-only** rewrites (C1) that are (b) **device-embeddable** into the
persistent megakernel (C2), (c) composed over the **maxed** primitive (§0/A1), (d) **fp64-gated
in-loop** (C3/A2), and (e) drawn from a library **extended to contain our levers** (C4).

**C0 — foundational unblock (prerequisite for any generated source on the real path).** The map found
three hard blockers, in order: (i) the **`grokking_optimizers.codegen` emitter module is missing** — the
production build never reaches the emitter (`_resolve_sources` L10492 globs hand-written TUs;
`_variant_macros` L14646 fails-open to macros-only). Either restore that module to render the *real
megakernel TU*, or — cleaner given C2 — **re-target the emitter to emit a device-inlinable fragment**
(a `__device__` tile/epilogue function spliced between grid barriers), NOT a standalone `__global__`
(today's synth emits launchable kernels with their own grid — useless for the persistent megakernel).
(ii) install + verify the **soft deps** actually needed (islpy for sound polyhedral deps; libclang for
the body-lift) — skip cutlass (host-launch, C2). (iii) the synth's elementwise/reduce path emits a
standalone kernel too; the C0 re-target fixes both. **Until C0 lands, the generative layer cannot touch
the production build at all** — this is the real cost the scoping no-go was pricing.

Candidate targets once C0 + A2 are green, in increasing risk:

- **C-fusion-1 (P1 fp32-epilogue fusion):** synthesize fusions of the ~10 `__syncthreads`-separated
  fp32 passes/layer (bias/round/residual/LN-first-read) into the GEMM epilogue lambda
  (`model_stage_decoder_tc.cuh:1046-1188` + ViT twin) — order-preserving fold, A/A/A-safe iff the fp32
  fold order is preserved (`:1071-1075`). This is the most tractable *and* highest-value synthesis
  target: affine, non-GEMM, no reduction reorder. Polyhedral (Phase D) and OpGraph synth both plausibly
  reach it.
- **C-transport-1 (dW staging variants):** parameterize + synthesize the contiguous K-major
  pre-transpose staging (`SG_TUNED_DEC_DW_STAGE`) as a *generated* transport pass rather than a single
  hand-wired macro — let the generative layer emit the variant set, fp64-gate, pick the winner.
- **C-gemm-family (only if Phase B showed a gap):** CUTLASS emitter on host-launchable GEMM families
  (Muon NS, SuperGrok2 `dt_proj`) — these already live host-side, so C2 is satisfied.

Each lands one at a time on the ratchet. **Do not** attempt persistent-megakernel-internal
auto-scheduling of the GEMM until a host-launched GEMM win is demonstrated *and* a persistent-launch
composition story exists — the scoping doc flags this as the hardest, least-proven part (`:340-346`).

## 5. Phase D — polyhedral on the non-GEMM fp32 epilogues

The libclang+islpy scaffold is order-sensitive, so confine it to the **affine fp32 epilogue passes**
(the P1 fusion candidates) where order-preserving tiling/fusion/interchange is legal (C1). It cannot
reach the GEMM substrate (no model of wgmma/TMA, `:188`). Treat as an *experiment* that either
compounds the C-fusion-1 win or is recorded as a measured dry-well.

## 6. Gating + ledger discipline (the spine, non-negotiable)

Every generated/transformed candidate, at every phase: **(1)** fp64 parity (rel-tol 1e-4; SAM
surfaces 2.5e-2/3e-2) **(2)** A/A/A bit-identical determinism via `torch.equal` **(3)** 3-seed timing
at d=2048 — KEEP iff faster on 3+ seeds AND parity-clean, else REVERT, root-caused into the ledger.
The fp64 gate (A2) runs *in-loop* for generating origins; the hardware gate
(`test_l3tc_tail_gate.py`) remains the final authority on the production step. Microbench screens
(fast_triage) may PRUNE but never PICK (the representativeness invariant).

## 7. Honest risk register

| risk | severity | mitigation |
|---|---|---|
| Pattern library doesn't contain our levers (C4) | high | Phase C step 0 = extend the library to the named transport/fusion rewrites *before* turning search on; the status map tells us how far the existing emitters already get. |
| No e-graph / equality-saturation / semantic search exists here (`:90-92`) | med | We are NOT building one — transport-only legality (C1) collapses the useful space to a small, enumerable set; TPE + the extended pattern library covers it. |
| Persistent-launch composition of any generated GEMM is unproven (`:346`) | high | Gate behind a demonstrated host-launched win first; never block the megakernel path on it. |
| fp64-in-loop adds ~60s/candidate (the gate cost) | low-med | Only generating-origin candidates pay it; flag-only variants keep the cheap oracle; the cost-model prunes before the gate. |
| Synthesis effort exceeds the value vs #24 (the scoping no-go's core point) | med | Phase B calibration + one-lever-at-a-time ratchet means we *measure* the marginal value continuously and can stop at any phase with the ledger as the record. |

## 8. Sequencing vs. the in-flight work (status map RESOLVED)

The concrete near-term path is now fixed:
1. **A2 (fp64-oracle-in-loop) — ALREADY LANDED (af9b720), verified by code read.** The selection side
   is complete: `_default_correctness_hook` (L7213) builds the fp64-parity + A/A/A gate via
   `run_cell_gate`, `_jit_autotune` installs it on the GPU path (L7236), and `pick_winner`'s P0-1 path
   (L6926-6959) re-checks the top-K candidates against it — demote-and-fall-through, fail-closed — so a
   winner that merely shares an fp32 rounding error with the strict oracle (the IL=4 trap) can't be
   finalized. The #16 gate (L6882) already makes generated origins ineligible without a strict-oracle
   PASS. **Remaining A2 gap is Phase-C-coupled, not standalone:** the hook runs `run_cell_gate(cell_key)`
   against the *currently-built* `_ops` (L7269 comment), so for a truly *synthesized* candidate the
   candidate's generated source must be rebuilt+published *before* the hook fires (today only the final
   winner is rebuilt in the verify phase). That per-candidate rebuild-before-fp64-gate is wired in C0,
   when generated candidates first exist — there are none today, so A2 is correct as-is for the
   flag/macro path that actually ships.
2. **A1 (max the primitive)** completes as the #22 tournaments resolve — the PIPE=2-vs-PIPE=1/S=4 gate
   (in flight) fixes the maximal decoder primitive; ViT B1 (#29) is the bigger lever next.
3. **Phase B (CuTe calibration probe)** — authorable any time; informs whether synthesis aims at the
   GEMM (unlikely) or stays on the fusion (likely).
4. **C0 (foundational emitter unblock)** then **Phase C (fusion synthesis on the working lowerers)** —
   the large build; gated behind A2 green + C0 + the maxed primitive.

So #27 advances *in parallel* with the structural ratchet: A2 + Phase B are CPU-authorable now while the
GPU runs the perf gates; C0/C is the post-A2 build. Honest framing for the owner: the scoping no-go was
pricing exactly the C0 cost (a missing emitter module + absent deps + a re-target from launchable
kernels to device fragments). The owner's GO is taken — we start with the genuinely-tractable, already-
half-built pieces (A2 + the real source-swap/validation gate) and let the ratchet decide each step.
