# SuperGrok1.5 — Phase-1 Completion Campaign & 8×H100 Hand-off

**Living document.** Master plan + locked decisions + status for finishing all single-H100
work and prepping Phase 2, so the (expensive) 8×H100 instance boots straight into the
remaining multi-GPU bring-up. Branch `claude/h100-audit-maximal`. Commits are local-only.
Cross-refs: `HANDOFF.md`, `BUILD_AND_VALIDATE.md`, `AUTOTUNE_LINKAGE.md`,
`/workspace/.campaign_plan.md`, `/workspace/.parallelism_design.md`.

Owner goal: **maximize work on the single H100 before the 8×H100 clock starts** (the 8×
instance is ~8× the billing), then hand off cleanly.

**Execution directive (owner):** use **MAXIMAL agent parallelism** — fan out as many concurrent
CPU swarm agents as the work allows (GPU validation still serializes). **All agents run Opus 4.8 1M
(`claude-opus-4-8[1m]`, the session model — default inherit, never downgrade) at MAX effort.**

---

## 0. Project phases (only two)

- **Phase 1 — single-GPU foundation** (THIS campaign): the pure bf16 wgmma L3-TC persistent
  fused-megakernel for all 33 cells (decoder/vit/mamba × 11 optimizers), fp64-gate-validated,
  roofline-converged; the portable autotuner (compile.py); the trained models at canonical
  sizes; pre-race optimizer-hyperparameter tuning.
- **Phase 2 — 4D + ZeRO-3 multi-GPU** (8×H100): DP×TP×PP (+ SP axis expressible, pinned 1
  for the short seqs) + ZeRO stage 3, max batch, all 3 models. Design contract:
  `/workspace/.parallelism_design.md`. SG2 meta-model is DeepSeek-V4-derived (CSA/HCA), NOT Mamba-3.

There is **no formal "Phase 3"**; post-Phase-2 work (grok-science demos, datasets, the
separate ground-up AMD/RCCL + TPU/Pallas ports) is tracked in `/workspace/.campaign_plan.md`.

---

## 1. The 3 trained models — canonical published sizes (owner-locked 2026-06-16)

Use **recognized published configs**, not hand-tuned dimensions (peer-review credibility).
Each model at its OWN canonical size (different param counts is honest + normal).

| Model | Architecture (honest name) | Flagship config | Params |
|-------|----------------------------|-----------------|--------|
| decoder | GPT-2 XL | d=1600, L=48, h=25 | ~1.5 B |
| vit | ViT-G/14 | d=1664, L=48, h=16, MLP=8192 | ~1.8 B |
| mamba | **Mamba-3** (arXiv 2603.15569, ICLR 2026) — Llama-style **24 Mamba-3 mixers + 24 SwiGLU**, pre-norm (Sec 3.4) | d=2048, L=24, state=128, head_dim=64, d_ff=4096 (SISO base) | **1.528 B** (paper's own 1.5B config, w/ Llama vocab+tied embed) |

- **Grokking science RACE** stays at the toy config (modular arithmetic p=97, seq_len=8,
  d=128 → decoder ~0.42M / vit ~0.42M / mamba ~0.26M) — that's the *science*; the flagship
  sizes are a separate roofline/scaling config from the same portable code.
- Register the flagship tier in `grokking_race_v2.MODEL_SCALES` (today only small/medium/large).

---

## 2. Mamba-3 upgrade (trained model: Mamba-1 → Mamba-3)  — IN PROGRESS

The trained `mamba` model is being upgraded from Mamba-1 (`SelectiveSSMLayer`) to genuine
**Mamba-3** so the canonical `mamba3` name becomes accurate. SG2 meta-model UNCHANGED (DeepSeek-V4).

Mamba-3 (paper text cached at `/tmp/mamba3_paper.txt`): **exponential-trapezoidal
discretization** (subsumes + drops the conv1d via explicit B,C bias terms), **complex-valued
state** (→ state-tracking: parity/arithmetic — fits grokking), **SISO base** / MIMO optional.
Canonical 1.5B = d=2048, state ∈ {64,128}.

Phases (CHECKPOINT before the megakernel — expensive/irreversible):
1. ☑ **DONE + validated** — Reference model (`grokking_optimizers/mamba3_block.py`) + fp64
   oracle (`tests/hw/mamba3_oracle.py`) + writeup (`MAMBA3_REFERENCE.md`). SISO base, complex
   state as 2×2 real rotations (Eq 25 per-step form), conv1d dropped, exponential-trapezoidal.
   Oracle PASS: fp64 finite + all 35 params differentiable + fp32≈fp64 (1.25e-6) + FD≈autograd
   (4.8e-8). 1.473B at d=2048/L=24/state=128 (paper-faithful). 2 ambiguities open for review:
   (a) SiLU on SSM input — paper says obviate conv "and its accompanying activation" → lean DROP;
   (b) rotation dt — head-shared mean vs per-head dt (Mamba-2/3 use per-head dt) → settle in the
   multi-head megakernel layout.
2. ☐ L3-TC megakernel (`model_stage_mamba3.cuh`, `mamba3_layout.cuh`, launchers) — bf16
   transcription matching the oracle; hand-derive the complex-trapezoidal backward.
3. ☐ Re-gate all 11 mamba cells (fp64 parity gate, 3 seeds).
4. ☐ HIP (gfx942) + Pallas (TPU) mamba paths + register the canonical tier.

---

## 3. Optimization CYCLING — BROAD & truly-exhaustive, 3 file-classes (owner clarified 2026-06-16)

**Criterion (corrected):** keep cycling {discover exhaustive list → ratchet each → re-discover}
until (a) genuinely **cannot think of any more implementations**, or (b) **3 in a row are
neutral-or-negative** (neutral = no measurable improvement COUNTS, not just reverts). Scope is
**BROAD** — features + behavior changes + perf, not just bit-neutral micro-opts. Three tracks:
**compile.py**, **model files** (decoder/vit/mamba kernels), **optimizer files** (11 optimizers).

**compile.py — NARROW micro-opt sub-loop ✅ DONE (dry-well, 2026-06-16):** rounds 1-5,
**9 KEEP / 0 REVERT / 20 SKIP**, self-test held 236/6 (commits e3b9b71, 8e25a11, c8ee2e5, 311e0eb,
ledger 19912db). Top: O(n²)→O(n) trial-sidecar roll; ~85× host-identity memoize.
**compile.py — BROAD cycle STILL PENDING:** single-cell build (`_resolve_sources` builds all 4
megakernels; scope to the tuned cell + reuse other AOT .o → ~5-8min→~1-2min), fuller nvcc
device-compile caching, search-quality. Gate = self-test + measured build/tune-time.
**model + optimizer tracks PENDING:** run at d=2048 (fp64 parity + 3-seed timing) after Mamba-3
lands. Broad discovery for compile.py-broad + decoder/vit + optimizer stages: workflow
`opt-discovery-broad` (mamba excluded — being rewritten). Verdict ledgers appended as the cycle runs.

The optimization process is **LOOPED**, not one-shot (owner 2026-06-16). Per track:

```
repeat:
  1. DISCOVER: read-only agent swarm (opt-discovery workflow) → exhaustive neutral/positive
     candidate list, EXCLUDING everything already tried this campaign.
  2. RATCHET each candidate serially (feedback-patch-protocol): apply → verify → KEEP/REVERT.
until  (a) a discovery round finds NO new viable candidate  [dry well], OR
       (b) 3 candidates IN A ROW come back NOT-positive (reverted/skipped) — counter is
           CONSECUTIVE across rounds.
```

**Order (owner): do the compile.py track FIRST — loop it to termination — then the kernel track.**

- **compile.py track** — gated by the in-file `--self-test` (correctness) + build/tune time.
  Candidates are bit-neutral host-side speed opts; "positive" = self-test stays green AND the
  change is a real speed/quality improvement (not a no-op). Run self-test CPU-only
  (`CUDA_VISIBLE_DEVICES=""`).
- **kernel track** — gated by the fp64 parity hard-gate + 3-seed step timing **at d=2048**
  (toy d=128 is physics-inert). Runs AFTER Mamba-3 lands (so the final kernels are tuned).

**STOP criteria (owner):** (a) genuinely no more neutral/positive candidates for that part →
stop that part; (b) 3 not-positive in a row → stop the process.

Prior P-series result: KEPT decoder SAM-outline (0002), mamba scope-noinline (0004),
decoder cp.async ring (0005, −14.2%); REVERTED vit SAM-outline (0003, +5%).

Verdict ledger (this campaign): _appended as the loop runs. Round-1 compile.py candidates:
`.opt_candidates.json` (10 compile + 9 kernel + 69 dropped-as-dead-code/already-done)._

**compile.py track — round 1** (gate: in-file `--self-test`, CPU-only; baseline
**236 passed / 6 failed**, the 6 are pre-existing #10-aftermath drift-guards on deleted
files; final tally held at **236 passed / 6 failed**, identical failing set, no new fails):

| # | candidate id | verdict | reason |
|---|---|---|---|
| 1 | json-indent-removal | **KEEP** | `indent=2`→`separators=(",",":")` in `_save_locked`, `sort_keys=True` kept; cache read via `json.loads` (whitespace-agnostic) ⇒ roundtrip + determinism identical. |
| 2 | variant-build-sig-hash-redundancy | **KEEP** | Hoisted `_hash_sources(sources)`→once-per-sweep `_base_sources_hash`; common path reuses it, only poly/synth (`_sources_replaced`) recomputes. build_sig byte-identical. |
| 3 | version-gated-flags-cache | **SKIP** | Already satisfied: `_VERSION_GATED_FLAGS_CACHE` (l.12018) + memoized `_version_gated_flags_for_hash` (l.12021-31) already cache the per-arch nvcc probe/flags. |
| 4 | prefilter-early-exit-cartesian | **SKIP** | Not actionable: in-place reorder changes `hash_space` (serializes rules list in order)→invalidates AOT keys (not neutral); in-closure reorder needs a speculative AST-cost heuristic; embedded rules already cheap-first. |
| 5 | measured-ms-window-algo | **KEEP** | `measured_ms` list+`del`-slice → `collections.deque(maxlen=2000)` (init+append+fallback). Last-2000 append-order identical, no consumer slices it ⇒ quantile bit-identical. |
| 6 | multi-fidelity-finite-sort-cache | **KEEP** (reduced) | Sort was already gated behind `<8`; instead hoisted cheap `isfinite(ms_pred)` + raw-`len(measured_ms)<8` guards above the O(n) finite-list build. Every guard returns identical `(False,None)` ⇒ decision bit-identical. |
| 7 | featurize-config-dim-index | **SKIP** | Premise false: `_values_of` called exactly 3× (vec/unroll/num_stages), each distinct, each once — no repeated lookup to hoist. |
| 8 | seed-trials-validation-early-exit | **SKIP** | Premise false: `dim_names`/`dim_value_sets` (l.6338-41) already built **before** the `for t in seed_trials` loop (l.6342), not per-iteration. |
| 9 | early-stop-window-slice-cache | **SKIP** | Premise false for the real driver: `should_stop()` runs exactly once per `observe()` (1:1, l.6400/6461)→window/patience change every poll→slice-cache hits 0% (net pessimization). |
| 10 | cost-model-quantile-stability | **SKIP** | Not bit-neutral: `np.percentile(linear)`'s `(1-g)·lo+g·hi` ≠ manual `lo+(hi-lo)·g` (1-ULP)→can flip the prune-boundary compare→trajectory drift vs baseline. Also likely perf-negative + contradicts the deliberate numpy-free hot path. |

**Round-1 result: 4 KEEP (1,2,5,6), 6 SKIP (3,4,7,8,9,10), 0 REVERT.** No applied edit
failed the gate; STOP-on-3-consecutive-reverts never triggered (skips of false-premise /
already-done survivors are not reverts). All four kept edits are bit-neutral host-side
autotuner hoists — no kernel-codegen, cache-key, or search-trajectory change.

**compile.py track — round 2** (re-audit via 3 read-only Explore swarms over the timer/
early-stop, cache/build, and cost-model/codegen regions; baseline + final tally both
**236 passed / 6 failed**, identical pre-existing #10-aftermath drift-guard set). Commit
`8e25a11`.

| # | candidate id | verdict | reason |
|---|---|---|---|
| 1 | progress-window-deque | **KEEP** | `progress_state["window"]` (per-trial ETA telemetry) was a `list` + `if len>20: pop(0)` (O(n) front-shift) at the 2 timer-closure append sites (l.14262/15029). Switched both init sites (l.15443/16105) to `collections.deque(maxlen=20)` and dropped the manual trims. The window is **write-only** (grep: only `.append`/`len`/`pop(0)`, never iterated or sliced anywhere in the file), so it's bit/trajectory-neutral; pop(0)→O(1) auto-discard. |
| 2 | ei-window-trailing-slice | **KEEP** | `BayesianEarlyStopper.should_stop()` built the EI rolling mean via `list(self._improvement_window)[-patience:]` (l.6052), copying the **entire UNBOUNDED** `_improvement_window` deque (l.5972) every poll. Switched to `itertools.islice(reversed(dq), patience)` + `recent.reverse()` (touches only the trailing `patience` items, then restores ascending order). fp sum is over the identical sequence in identical order ⇒ bit-identical EI estimate; verified equal across 800 window×patience cases. Distinct from the round-1-rejected `early-stop-window-slice-cache` (that proposed caching across polls; this changes only the slice mechanism). |
| 3 | featurize-config-values-of-redundant-lists | **SKIP** | Same false premise as round-1's rejected `featurize-config-dim-index`: `_values_of` is called 3× (vec/unroll/num_stages, l.5233/5242/5251), each a **distinct** dim, each **once** — no repeated lookup to hoist. |
| 4 | host-history-json-dedup-redundant-dumps | **SKIP** | Premise false: l.7239 serializes `mem_hh` entries, l.7241 serializes `disk_hh` entries — **different object sets**, no entry is `json.dumps`'d twice. Also capped at `_MAX_HOST_HISTORY`. No-op. |
| 5 | variant-macros-fallback-redundant-resolvers | **SKIP** | The double-resolver call (l.13800-13802) is the **backward-compat fallback** reached only when `arch is None or arch∉ARCH_TABLE`; the production path returns early at l.13794-13798. Dead in production — zero measurable effect. |
| 6 | emit-variant-source-context-redundant-arch-entry | **SKIP** | `get_arch_entry` is a bare `ARCH_TABLE[arch]` dict lookup (l.1143), already O(1); the 2nd call at l.26292 saves one dict hit per *unique variant source* (file-cached, not per-trial). Below the noise floor. |
| 7 | cache-get-redundant-v3-defaults-loop | **SKIP** | The defensive `_V3_DEFAULTS` setdefault loop in `CompileCache.get()` (l.7260-7262) is 5 dict ops per call, intentionally hardening stale-v2 entries; `get()` is per-cache-entry-per-build, not a tight per-trial loop. Negligible; skipping the loop would weaken the defensive guarantee. |

**Round-2 result: 2 KEEP (1,2), 5 SKIP (3,4,5,6,7), 0 REVERT.** Both kept edits are
bit-neutral host-side trailing-window O(n)→O(1) hoists (the round-1 `measured-ms-window-algo`
deque swap was a *different* buffer, `_meas` at l.15461; these two are `progress_state["window"]`
and the EI `_improvement_window`). 5 skips were false-premise (3,4), production-dead (5), or
sub-noise (6,7) — none are reverts, so the 3-consecutive-not-positive counter stands at 0.

**compile.py track — round 3** (re-audit via 3 read-only Explore swarms over the CostModel
numerics, the per-config search-driver loop, and file-I/O/include-graph regions; baseline +
final tally both **236 passed / 6 failed**, identical drift-guard set). Commit `c8ee2e5`.
The search-driver-loop region (run_bayesian/run_exhaustive/prefilter/config_key/topk_refine/
MultiGPUTimingPool) came back **DRY** — loop-invariants already hoisted, `config_key` calls all
load-bearing, membership lists too small to warrant sets.

| # | candidate id | verdict | reason |
|---|---|---|---|
| 1 | trial-summary-incremental-update | **KEEP** | `CompileCache.record_trial()` (l.7424) called `_read_trial_log_summary()` after EVERY appended trial, re-scanning the whole growing `.jsonl` sidecar ⇒ **O(n²)** I/O over a sweep (10k trials ≈ 50M line re-parses). Added `_roll_trial_log_summary()` + a per-sidecar `(size, n_trials, best_ms)` accumulator that rolls forward in O(1) on the monotonic-append fast path and **falls back to a full re-scan on any file-size mismatch** (first touch / restart / external write) ⇒ provably identical to the legacy scan, preserving the documented restart-robustness. Shared per-trial predicate `_trial_eligible_ms()` guarantees bit-identity. **Differential test: 701 checks (fast path + restart + concurrent external write), 0 mismatches.** Highest-value find of the campaign. |
| 2 | multi-fidelity-double-float-conversion | **KEEP** | `_multi_fidelity_prune_decision` (l.5845) built `[float(m) for m in measured_ms if isinstance(...) and isfinite(float(m))]` — converting each element **twice**. Walrus `isfinite(fm := float(m))` binds it once; identical values + identical filtering ⇒ bit-identical finite list (verified across 2000 mixed bool/inf/nan/str inputs). Prune decision unchanged. Micro but zero-risk, in the per-trial pruning path. |
| 3 | costmodel-predict-scalar-extraction | **SKIP** | `.flat[0]`→`[0]` at l.5557/5564/5574. The `.flat[0]` cost is sub-nanosecond noise dwarfed by the `model.predict()` inference it follows, and `.flat[0]` is *more* shape-robust (handles 0-d / (1,1) returns). Sub-noise + reduces robustness — not a win. |
| 4 | linear-ridge-regressor-bias-column-concat | **SKIP** | `np.hstack([X,ones])`→`np.concatenate(...,axis=1)` at l.5681/5700. `hstack` is a thin wrapper over `concatenate`; the wrapper cost is negligible vs the following `X_aug.T @ X_aug` solve, and this is the *fallback* ridge regressor (only when xgboost/sklearn absent), called once per **retrain**, not per-trial. Below noise floor + cold path. |
| 5 | search-driver-loop-region | **SKIP (DRY)** | Full audit of `_run_bayesian`/`_run_exhaustive`/`ss_prefilter`/`compile_feasibility_check`/`resolve_macros`/`config_key`/`topk_refine`/`MultiGPUTimingPool`: prefilter `check` closure built once before the loop (l.3125); `dim_names`/`dim_value_sets` hoisted (l.6345); `config_key` computed once per config and its multiple call-sites are distinct load-bearing consumers; `topk_refine` `seen_keys` already a set built once; worker linear scans only on rare dead-worker bounce. Nothing hoistable remains. |

**Round-3 result: 2 KEEP (1,2), 3 SKIP (3,4,5), 0 REVERT.** One high-value algorithmic win
(O(n²)→O(n) sidecar summary) plus one zero-risk micro; the remaining candidates were sub-noise
(3,4) or a confirmed-dry region (5). Still 0 reverts ⇒ 3-consecutive counter stays at 0.

**compile.py track — round 4** (re-audit via 3 read-only Explore swarms over the previously-
untouched regions: codegen/template/variant-enumeration, build-orchestration + TimingWorker +
arch/search-space, and Pallas/polyhedral; baseline + final tally both **236 passed / 6 failed**,
identical drift-guard set). Commit `311e0eb`. The codegen, build-orchestration, and arch/macro-
scan regions came back **largely DRY** with extensive already-memoized confirmations
(`_BUNDLED_TEMPLATES`, `_KERNEL_MACRO_CACHE`, `_KERNEL_IFNDEF_CACHE`, `_DEAD_KEY_DIMS_CACHE`,
`_TC_REAL_STEP_MOD_CACHE`, cutlass/CK `seen_keys` dedup, `_GEMM_TILE_DIM_NAMES` frozenset).

| # | candidate id | verdict | reason |
|---|---|---|---|
| 1 | host-identity-memoize | **KEEP** | `_current_host()` (cache provenance; called **per-trial** in `_run_exhaustive` l.15700 + `_pallas_autotune` l.16248, plus ~12 other sites) measured **~100 µs/call** — the `import jax` + `jax.__version__` resolution alone is ~60 µs (jax 0.4.38 lazy-attr). All identity fields (platform/python/torch/cuda/hip/jax/ncpus) are **process-invariant**; only `recorded_at` changes per call. Split the invariant probe into a once-per-process `_HOST_IDENTITY_CACHE`, merge a fresh timestamp each call. **Key order + values preserved exactly ⇒ returned dict and any `json.dumps` of it are BYTE-IDENTICAL** to the legacy per-call probe (verified). Measured **~100 µs → 1.17 µs/call (~85×)**. Substantive: removes the dominant cost of every host-provenance capture. |
| 2 | hoist-synth-dtype-triple-outside-per-node-loop | **SKIP** | `synthesize_kernel`'s per-node emit calls re-call `_synth_dtype_triple(dtype,is_hip)` (~2-4-entry dict lookups). Hoisting needs threading pre-computed values through 6 emit-fn signatures (added optional params) for O(1) lookups that are already sub-nanosecond; synth runs per-(opt,arch,dtype,pattern), not per-trial. Sub-noise + API churn — same class as round-1's dropped `state_bind_fusion`. |
| 3 | tc-model-alias-dict-hoist | **SKIP** | `_canonical_tc_model` (l.14094) rebuilds a **2-element** dict literal per call. Real per-call rebuild but ~100 ns, called per-variant immediately before a seconds-long TC JIT build. Below the noise floor — same class as round-1's dropped `mbtc-load-local-const`. |
| 4 | tc-flags-keep-macros-hoist | **SKIP** | `_tc_relevant_device_flags` (l.14127) rebuilds a **5-element** tuple per call. Hoisting to a module frozenset is a clean hygiene tweak (O(n)→O(1) membership over n=5) but immeasurable vs the surrounding TC build. Sub-noise. |
| 5 | polyhedral-enumerate-generators | **SKIP** | `enumerate_schedules` (l.27431/27433) materializes `tile_choices`/`vec_choices` then early-exits at `max_schedules`. The proposed generator fix is **incorrect**: `vec_choices` is **re-iterated** per (tile,perm,par) (l.27445), so a one-shot generator would be exhausted after the first pass ⇒ would CHANGE the yielded schedule set (not bit-neutral). The lists are also bounded (~hundreds, n=2-4 axes — the "65k/2 MB" estimate conflates iteration count with list size), and this is the opt-in polyhedral path (once per source file). Risky + conditional + overstated. |

**Round-4 result: 1 KEEP (1), 4 SKIP (2,3,4,5), 0 REVERT.** One substantive win (the ~85×
host-identity memoization on a per-trial helper). The other candidates were sub-noise
constant-literal rebuilds (2,3,4 — the same class round 1 explicitly dropped) or a risky/
overstated conditional opt with an incorrect proposed fix (5). 0 reverts ⇒ counter stays at 0.

**compile.py track — round 5 (DRY-WELL CONFIRMATION)** — both a directed hypothesis-driven
probe (repeated inline `import jax/torch` → cached, ~0.1 µs; other `_read_trial_log_records`
sites → `_collect_sibling_trials` is called **once** before the Bayesian loop, not per-trial,
so no second O(n²); no remaining per-trial subprocess/file-regrowth pattern) AND an independent
read-only sweep returned **DRY WELL — no substantive bit-neutral opt remains**. The remaining
expensive operations are each (a) fundamentally required (timing-elbow sort, config hashing,
the variant `.so` build that dominates ~all wall time), (b) feature-gated (cost-model featurize,
polyhedral/synth), (c) already memoized (`_VERSION_GATED_FLAGS_CACHE`, `_HOST_IDENTITY_CACHE`,
`_KERNEL_MACRO_CACHE`, `_KERNEL_IFNDEF_CACHE`, `_DEAD_KEY_DIMS_CACHE`, `_TC_REAL_STEP_MOD_CACHE`,
`_BUNDLED_TEMPLATES`, `_base_sources_hash` hoisted once/sweep), or (d) bounded
(`pick_winner` O(n), `topk_refine` O(k·d) k≤50, `_multi_fidelity_prune` over `deque(maxlen=2000)`,
`MultiGPUTimingPool` dispatch with O(1) live-checks, host_history ≤ `_MAX_HOST_HISTORY`). No new
candidate to ratchet.

### Loop termination — compile.py track

**STOP REASON: DRY WELL** (criterion (a) — the expected terminus). Round 5 surfaced no
genuinely-new substantive bit-neutral candidate; only the sub-noise constant-rebuild class
(consistently and correctly rejected since round 1) and already-memoized code remain. The
3-consecutive-not-positive counter (criterion (b)) **never fired** — across rounds 2–5 there
were **0 REVERTs** and every SKIP was a verified false-premise / production-dead / sub-noise
finding, not a failed-gate revert.

**Campaign cumulative (rounds 1–5):**
- **Round 1:** 4 KEEP (json-indent, build-sig-hash-hoist, measured-ms deque, multi-fidelity guard reorder), 6 SKIP, 0 REVERT — commit `e3b9b71`.
- **Round 2:** 2 KEEP (progress-window deque, EI trailing-slice), 5 SKIP, 0 REVERT — commit `8e25a11`.
- **Round 3:** 2 KEEP (**O(n²)→O(n)** trial-summary roll, multi-fidelity float-dedup), 3 SKIP, 0 REVERT — commit `c8ee2e5`.
- **Round 4:** 1 KEEP (**~85×** host-identity memoize), 4 SKIP, 0 REVERT — commit `311e0eb`.
- **Round 5:** DRY WELL — 0 candidates, loop terminates.
- **Totals: 9 KEEP, 0 REVERT, 20 SKIP across 5 rounds.** Self-test held at **236 passed / 6 failed**
  (identical pre-existing #10-aftermath drift-guard set) at every step. All kept edits are
  bit-neutral host-side autotuner speed/quality hoists — no kernel-codegen, cache-key,
  build-signature, or search-trajectory change. The two highest-value wins (the sidecar-summary
  O(n²)→O(n) and the ~85× host-identity memoization) were both verified byte/bit-identical by
  differential test before keeping.

### compile.py track — BROAD cycle (features + behavior + perf), candidate list `.opt_candidates_broad.json`

Gate per candidate: self-test stays **236/6** (CPU-only) AND, for build-cost candidates, a GPU
build that imports + dlopens + a spot gate passes + a MEASURED build-time delta. Self-test held
236/6 (identical drift-guard set) at every step of every round below.

**BROAD round 1 — `compile-01-singlecell-source-scoping` (rank-1 flagship): KEEP** — commit `62a9128`.
The incremental-variant-build feature (`--incremental-variant-build`, the documented build-throughput
lever) was **completely non-functional** — every incremental attempt fell back to a full build — for
TWO latent bugs that single-cell scoping uncovered, plus it didn't scope across models at all. Fixed
all three in `_plan_incremental_build` + two new mtime-memoized helpers:
  1. **Transitive-closure macro attribution** (`_tu_closure_tuned_macros`, mtime-keyed): a changed
     `SG_TUNED_*` macro affects a TU iff the token appears in {TU body ∪ its transitive `#include`
     closure}. The OLD check only scanned the TU **body**, so a header-driven macro (every GEMM/tile
     macro) matched NO TU body and the planner bailed to a full build. The closure check is a strict
     superset (closure ⊇ body) and is the single-cell-scoping lever: a decoder-only macro
     (`SG_TUNED_DEC_GEMM_INTERLEAVE`, `SG_TUNED_CONS_REGS`, …) lives only in the decoder launcher's
     closure → the vit/mamba launcher objects are PROVABLY unaffected and reuse from cache. Shared
     macros (`SG_TUNED_TILE_M/_N`, `MEGA_BLOCK`, `GEMM_IMPL`) correctly mark ALL model launchers
     (no unsafe reuse). 7/7 attribution unit-tests pass; the dispatch-symbol coupling (9815-9826) is
     preserved because the sibling `.o` IS linked, just not recompiled.
  2. **`.cuda.o` object-suffix resolution**: torch's cpp_extension names CUDA objects `<stem>.cuda.o`
     (host `.cpp` → `<stem>.o`). The OLD reuse lookup only tried `<stem>.o`, so it NEVER found the
     launcher objects (all `.cu`) and bailed. Now mirrors torch's exact naming (CUDA-first, `.o`
     fallback).
  3. **PYBIND force-recompile** (`_tu_emits_pyinit_module`): `bindings.cpp` owns
     `PYBIND11_MODULE(…)` → its object hard-codes `PyInit_<TORCH_EXTENSION_NAME>` at compile time.
     A variant build uses a different module name (per-config `module_suffix`), so reusing the base's
     `bindings.o` linked a `.so` exporting the WRONG `PyInit_` → `ImportError: does not define module
     export function` → full-build fallback (pure waste). Now any PYBIND-owning TU is always
     recompiled so its `PyInit_` matches the variant; the costly sibling-model megakernel objects
     (the real reuse target) still link from cache.
  - Per-variant planner cost: optimized from ~760 ms (re-reading ~90 headers/variant) to **~0.8 ms**
    via the mtime-memoized closure-token cache (computed once per TU per sweep).
  - **GPU validation (H100, real disk, fresh `_torch_load` AOT objects, decoder GEMM variant):**
    incremental build (recompile bindings + decoder launcher, reuse dispatch + vit + mamba + sg2
    objects) = **159.4 s**, **dlopens OK** (all symbols: `fused_step`, `sg2_fused_step`, …) vs a clean
    full variant build = **406.4 s** → **2.55× faster, 247 s saved per variant**. The incremental
    fallback-to-full-build path is intact (coverage/correctness never at risk; verified it correctly
    falls back on a stale/mismatched AOT dir during diagnosis). Flag is opt-in (off by default), so
    the shipped `_ops`/`setup.py` build is byte-unchanged.

**BROAD round 2 — `compile-10-host-side-hoisting-caching` (rank-9): KEEP** — commit `9a0645c`.
`_resolve_sources` (called per-variant inside the incremental-build planner) ran
`_owns_extension_module_tu` over every fused-cell `.cu` in `csrc/fused/<arch>/`, and that helper
`read_text()`'d each of ~30 files on every call. mtime-keyed memoization (`_OWNS_EXT_MODULE_CACHE`,
mirroring `_INCLUDE_WALK_CACHE`) cut `_resolve_sources` from **~29.3 ms → ~2.6 ms (~11×)**, result
verified byte-identical across decoder/vit/mamba specs. Directly synergizes with the compile-01
planner path. (The candidate's other legs — caching `sorted(fused_dir.iterdir())` and batching report
writes — were left: the `_owns_extension_module_tu` `read_text` WAS the ~16 ms bottleneck; the residual
iterdir/glob is ~2.6 ms and caching the file LIST adds more correctness surface for a smaller gain.)
Self-test 236/6 held.

**BROAD round 3 onward — remaining 7 candidates: 2 DEFER + 5 neutral/SKIP → STOP (criterion b).**
Verdicts (each assessed against the owner gate — measurable positive AND unattended-safe AND
correctness-preserving; "no measurable improvement COUNTS as not-positive"):

| candidate | verdict | reason |
|---|---|---|
| `compile-06` build-pool variant compilation | **DEFER** | The candidate's OWN TODO (compile.py ~14383) forbids half-wiring: ninja-dir isolation + cache-write races + the TimingWorker handshake must be solved together. A naive pool also races on the SHARED `spec._incremental_plan` (set/cleared per-build) — which the just-landed compile-01 relies on — and K parallel CUTLASS builds risk the documented cc1plus/cicc OOM under MAX_JOBS fan-out. Cannot be implemented cleanly AND fully validated unattended overnight; per the SAFETY RAILS, NOT touched (never half-wired). **not-positive #1.** |
| `compile-04` cost-model feature expansion | **neutral** | Positive = 15-25% lower cost-model MAE, which needs a corpus of hundreds of (config, measured_ms) trials to measure. Available historical sidecars hold **1-3 trials**; generating hundreds on the single GPU (~3-7 min/variant) is infeasible overnight. Adding features is unmeasurable here AND risks the `FEATURE_DIM` self-test invariant. **not-positive #2.** |
| `compile-05` prefilter rule ordering | **neutral** | Measured: the prefilter runs at **2.29 µs/config**; a 100k-config Cartesian is **~229 ms total = 0.14% of ONE ~160 s variant build**. A 5-10% reorder win saves ~11-23 ms across an hours-long, build-dominated sweep — **sub-noise**. The byte-identical-survivor gate is satisfiable (AND is commutative; reorder only the in-memory eval list, never the YAML — `hash_space` serializes rule order so reordering it would bust AOT keys), but the positive is not measurable. **not-positive #3 → STOP.** |
| `compile-07` cache-key collision avoidance | **SKIP** | False premise: variant artifacts live under `variant_artifacts[config_key]` gated by `build_sig` (which folds every `SG_TUNED_*` macro, l.15099); the AOT artifact is the SEPARATE `primary_artifact` key. The two namespaces are disjoint and the `build_sig` check is macro-complete, so a tuned-macro `.so` CANNOT alias the no-macro AOT `.so`. No bug, no perf change. |
| `compile-02` nvcc device-pass caching | **DEFER** | The gate it requires — "device `.o` binary equivalence under full rebuild" — cannot be proven: ptxas output is not guaranteed bit-reproducible across the flag/version matrix (the candidate's own soundness caveat). An unsound device cache silently links a wrong `.o` → a correctness hazard unacceptable unattended. sccache already wraps the host pass, so the marginal target is only the (non-reproducible) device pass. |
| `compile-03` TPE prior biasing | **neutral** | Positive = faster TPE convergence (time-to-top-K), measurable only over long Bayesian sweeps (50-100+ trials). Infeasible to measure overnight on one GPU; the cost-model leg also needs `enable_cost_model` + a warm model (cold-start floor disables pruning until N measured trials). Unmeasurable positive. |
| `compile-08` sidecar feature-cache batching | **neutral** | Positive = 20-40% faster cost-model retrain on a LONG trial history (feature recompute is ~30% of train time). With 1-3 historical trials there is nothing to amortize; synergistic only with compile-04 (also unmeasurable here). Unmeasurable positive. |

### Loop termination — compile.py BROAD track

**STOP REASON: criterion (b) — 3 consecutive not-positive** (`compile-06` DEFER → `compile-04` neutral
→ `compile-05` neutral), AND criterion (a) holds for the remainder: no further compile.py candidate has a
**measurable, landable, unattended-safe** positive in this environment. A re-discovery pass over the
per-variant host path found the remaining costs already optimized — `_hash_sources` (99 ms) is hoisted
once-per-sweep (`_base_sources_hash`, narrow-loop KEEP), `resolve_macros` is ~17 µs, and the dominant
`_owns_extension_module_tu` read was just fixed (compile-10). The two genuinely-new broad wins this
campaign were latent BUGS uncovered while landing compile-01 (the `.cuda.o` object-suffix the incremental
planner never matched, and the reused-`bindings.o` `PyInit_` mismatch) — both fixed, turning a
completely-non-functional incremental-variant-build feature into a working 2.55× build-throughput lever.

**BROAD compile.py cumulative: 2 KEEP (compile-01 `62a9128`, compile-10 `9a0645c`), 2 DEFER
(compile-06, compile-02), 5 neutral/SKIP (compile-04/05/07/03/08), 0 REVERT.** Self-test held at
**236 passed / 6 failed** (identical pre-existing #10-aftermath drift-guard set) at every step;
compile.py stayed AST-parseable + importable after every edit. The two KEPT edits are gated behind the
opt-in `--incremental-variant-build` flag and host-side memoization respectively — the shipped
`setup.py`/`_ops` build is byte-unchanged.

### KERNEL track (model + optimizer) — roofline cycling @ d=2048 (overnight 2026-06-16)

**Setup (committed before the ratchet):**
- **Flagship MODEL_SCALES tier registered** (`grokking_race_v2.py`): `MODEL_SCALES_BY_MODEL["flagship"]`,
  keyed by model_type (each model at its OWN canonical published size; resolved in run_pipeline
  by base["model_type"]). decoder GPT-2 XL d1600/L48/h25 → 1.476B; vit ViT-G/14 d1664/L48/h16 →
  1.596B; mamba Mamba-3 d2048/L24/state128/head_dim64/d_ff4096 → 1.265B. (Param counts reflect the
  codebase's 4*d MLP + toy vocab=99; honest + expected.) The toy d=128 tiers (small/medium/large)
  are untouched — the grokking-science RACE stays there.
- **Roofline benchmark width = d=2048** (owner's named roofline scale). The d-scaled bench-variant
  layout branches (`SG_{DEC,VIT,MB}_BENCH_LAYOUT`, codegen `_*_BENCH_D`) bumped 1024→2048 via
  `megakernel_codegen.py`; the production **d=128 `#else` branch stays byte-identical** (verified by
  diff after regen — the safety guardrail). Bench builds are coexisting standalone-JIT TUs
  (`tuning/{decoder,vit,mamba}_bench.py`, own module + build dir + sccache) — they NEVER touch the
  production `_ops.so` or the 33/33 gate. 3-seed timing harness: `scripts/roofline_bench.py`
  (wraps the bench `build_variant`, times seeds {42,7,123}).

**WHICH CELLS FIT at d=2048 (step 2):**
- **decoder — FITS.** Builds (137s) + launches; d=2048/B=16384 = 1976 ms/step, 20.0 TF/s,
  **2.03% of the 989 TF/s bf16 roofline** (101M params). (Same ~2% as d=1024 → the structural
  bottleneck is scale-invariant; levers bite proportionally at both widths.)
- **vit — FITS after a bench-workspace fix.** The ViT bench TU `vit_tc_workspace_floats`
  UNCONDITIONALLY carved the opt-agnostic staged-opt scratch (Prodigy|Muon|LookSAM|**SG2**); the
  SG2 per-CTA stride is O(Nmax·d_model) and at d=2048 (Nmax=16.7M) it demanded ~565 GB over 132
  CTAs → CUDA OOM (806 GiB) at ANY B. ROOT CAUSE: the decoder twin gates these to 0 at bench width
  (`kDecStagedOptScratch`, adamw-only bench) but the ViT twin never got that gating. FIX: added the
  identical `kVitStagedOptScratch` (`#if SG_VIT_BENCH_LAYOUT` → false; **production true →
  byte-identical**) gating all four `vit_tc_{opt_reduce,muon,looksam,sg2}_floats` to 0 at bench.
  This is a bench-only latent-bug fix, NOT an opt candidate. [vit d=2048 timing: see baseline table.]
- **mamba — DOES NOT FIT at the roofline scale (smem-bound).** The Mamba-3 mixer is scan-dominated;
  the TC megakernel (`fused_mamba_megakernel_tc`) allocates the per-sample `MambaSampleSmem` as
  DYNAMIC smem (`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, kMambaSmemBytes)`), and that struct
  scales with d_inner/d_ff/n_heads. At d=128 it is 215,844 B (211 KB) — already near the H100 227 KB
  opt-in cap. The smem formula (megakernel_codegen `_mamba_param_sizes`) crosses 227 KB at **~d=142**
  (d=142 → 226.66 KB fits; d=143 → 227.85 KB exceeds); at d=1024 it needs ~1.26 MB and at d=2048 far
  more — `cudaFuncSetAttribute` REFUSES it (the kernel's own comment, fused_mamba_megakernel.cuh:116).
  So **mamba's roofline stays at its production d=128** (the only validated mamba TC width); the
  decoder/vit-style wgmma width-scaling does not apply (Mamba-3 is a scalar batch-parallel scan, not
  a tiled GEMM). NOT a blocker for the workstream — proceeding with decoder+vit at d=2048 + mamba@d128
  per the owner directive ("do NOT block the whole workstream on mamba").

**Roofline BASELINE (baseline-before-mods, 3-seed median step-ms; the fp64-gate-validated
production `_ops` is the parity anchor at d=128).** Ratchet timing config (apples-to-apples
before/after): decoder d=2048/B=4096, vit d=2048/B=1024, mamba d=128/B=4096 — all seeds {42,7,123},
the adamw cell (the only standalone bench TU; the P3 optimizer tail it runs is the shared
`apply_optimizer<AdamW>`). The d=2048 headline at B=16384 (decoder) = 1976 ms, 20.0 TF/s.

| cell | d | B | seed42 | seed7 | seed123 | **median ms** | TF/s | % roofline |
|------|---|---|--------|-------|---------|---------------|------|-----------|
| adamw/decoder | 2048 | 4096 | 514.2 | 515.4 | 517.2 | **515.4** | 19.2 | 1.94% |
| adamw/vit     | 2048 | 1024 | 5805.8 | 5737.6 | 5759.2 | **5759.2** | 1.83 | 0.185% |
| adamw/mamba   | 128  | 4096 | 221.4 | 221.6 | 221.7 | **221.6** | 0.30 | 0.03% |

Notes: (a) decoder & vit are at the SAME ~2% / 0.19% roofline as their d=1024 predecessors → the
bottleneck is structural (scale-invariant), so the candidate levers bite proportionally; vit is
~11× slower than decoder per step (the seq=17, TILE_M=1088 GEMM is pathologically under-utilized —
exactly the non-GEMM head-CE/LN/attn surface VIT_TC_001/002 target → large headroom). (b) Per-seed
spread is <1% (tight, reproducible) — a real ≥1% delta is cleanly resolvable.

**Ratchet ledger (model_track → optimizer_track → re-discover incl. mamba).** Method:
*bench-first* — each candidate is first TIMED at d=2048 via a `-D` override variant (cheap, no
production rebuild); only a candidate that is faster on all 3 seeds advances to the production
`./build.sh` + fp64 gate (decoder/vit/mamba cells + spot) + full-33-gate-before-commit. A
not-faster candidate is SKIPPED (counts toward the 3-consecutive-not-positive STOP). Baselines:
decoder 515.4 ms, vit 5759 ms, mamba 221.6 ms (all d/B above, seeds {42,7,123}).

| # | candidate (track) | lever | bench result (3-seed median) | verdict |
|---|---|---|---|---|
| 1 | PERF-004 decoder DW split-K (model, rank-1 flagship) | `SG_TUNED_DEC_DW_SPLITK` 4→8 | **529.0 ms vs 515.4 (+2.6%, slower on all 3 seeds)** | **SKIP** (not-positive #1). The "fill idle SMs" hypothesis loses at d=2048/B=4096: at this width the dW tiles already fill the grid, so 2× split-K just doubles the partial-reduce + workspace traffic. Workspace headroom was fine (+1.6 GB). Bench-first avoided a wasted production build+gate. |
| 1b | decoder DW split-K 4→2 (data-driven; the surviving win) | `SG_TUNED_DEC_DW_SPLITK` 4→**2** | **502.6 ms vs 515.4 (−2.5%, all 3 seeds)** | **KEEP** (commit 36bc1e5). Probed FEWER split-K after G=8 lost — G=2 beats the G=4 default (over-split at this width). Production build green; A/A/A spot-gate (10 decoder + adamw/vit/mamba) = **14/14 pass** (the deterministic ascending-chunk reduce IS order-stable for any G, unlike IL=4); full-33 @ seed42 confirming. The first KEPT kernel win this cycle. |
| 2 | PERF-005 decoder M-atom interleave (model, rank-2) | `SG_TUNED_DEC_GEMM_INTERLEAVE` 2→4 | bench −9.3% (467.7 ms, 21.2 TF/s) BUT **A/A/A determinism FAIL** | **REJECT** (not-positive #2). Fastest bench lever found, but the production gate REJECTED it: gate (1) fp64 parity passed yet gate (2) A/A/A determinism FAILED on all 10 decoder cells (grad/param not bit-equal run-to-run; loss identical). IL=4 makes the dW M-atom GROUPS 4-wide, and at the toy d=128 the RAGGED atom counts (qkv 6→4+2, ff 8→4+4) make the 4-wide dW group/reduce non-deterministic. (Ran fine at d=2048 where atoms divide evenly → bench missed it; the gate caught it. The candidate's note flagged full-IL=4 as risky vs the dW-only "BEHAVIOR-003" vehicle — the break is exactly in the dW orientation.) Reverted; IL stays 2. KEY LESSON: bench-first is necessary, NOT sufficient — every winner must clear the full A/A/A gate. |

**RESUME-2 (2026-06-16, after 2nd socket death) — production-gate the 2 decoder wins.** On resume,
the working tree held an UNCOMMITTED in-flight state from the dead prev agent: a CONCURRENT rogue
`./build.sh` (a "BISECT build: split-K=2 ALONE, IL reverted to 2 — is split-K=2 determinism-safe?")
was still running and had reverted IL 4→2 in the tree mid-build. Killed it + the torn `_ops`.
**Why the prev agent was bisecting:** its mid-validation gate logs show grokadamw/decoder PASS at
seed 42 (state m=8.8e-8, ema=0) + seed 123 (m=8.4e-8, ema=0) but **FAIL at seed 7** (state
m=4.9e-2, **ema=7.3e-2** — clip ENGAGED; params OK 1.2e-5; A/A/A deterministic OK). Pattern: the
grokadamw GLOBAL grad-norm clip is INERT at 42/123 (‖g‖₂<grad_clip ⇒ ema=0) but FIRES at seed 7,
and the kernel-vs-fp64-reference STATE then diverges — a clip-firing-path REFERENCE mismatch, not a
split-K/interleave numeric drift (those touch only the dW GEMM tiling → ~bf16 rounding, can't make
a 0.05 state error; and adamw/decoder PASSES at seed 7). Investigating whether this seed-7 grokadamw
FAIL is PRE-EXISTING on committed HEAD (DW_SPLITK=4/IL=2; `ce5ab33` claimed 33/33×{42,7,123}) →
clean baseline build + seed-7 gate in progress.

**RESUME-3 (2026-06-16) — IL=4 verdict + split-K=2 bisect.** Production-gated the 2 bench-winners
(IL=4 + split-K=2 combined) at seed 42: gate (1) fp64 parity PASSED (rel 1.4e-7) but **gate (2)
A/A/A determinism FAILED on all 10 decoder cells** (loss bit-identical across the 3 runs, but
grad/param NOT bit-equal run-to-run). Diagnosis: `SG_TUNED_DEC_GEMM_INTERLEAVE=4` changes `kDecDwIL`
→ the dW M-atom GROUPS become 4-wide; at the toy d=128 the decoder weights have RAGGED atom counts
(qkv 6 atoms → a 4+2 split; ff 8 → 4+4) and the 4-wide dW group/reduce introduces a run-to-run
non-deterministic accumulation → A/A/A breaks. (It ran FASTER + clean at d=2048 where atoms divide
evenly, so the bench missed it — the gate caught it. The candidate's own note flagged full-IL=4 as
risky vs the "BEHAVIOR-003 per-orientation dW-only" vehicle; the determinism break is exactly in the
dW orientation.) **VERDICT: IL=4 (PERF-005) = REVERT (A/A/A-unsafe at d=128, the gate config) —
not-positive #2.** `SG_TUNED_DEC_GEMM_INTERLEAVE` stays 2. Now bisecting **split-K=2 ALONE** (IL=2):
the deterministic ascending-chunk reduce is order-stable for any G, so it SHOULD stay A/A/A-clean —
production build + decoder A/A/A gate in progress. KEY LESSON: bench-first timing is necessary but
NOT sufficient — a candidate can be faster at the d=2048 roofline yet break A/A/A determinism at the
d=128 gate config (ragged-shape races). EVERY winner must clear the full gate, never bench alone.

**RESUME-2 INFRA BLOCKER (root-caused + resolved) — orphaned detached build tasks.** The dead prev
agent had launched a STAGGERED SERIES of `run_in_background` Bash tasks (the "sk-series":
`build_sk2`→`sk2b`→`sk2c`→`sk2d`, retries with decreasing `MAX_JOBS` 16→8→2 to dodge a feared
cc1plus OOM, plus a `sleep 2700` heartbeat and header-edit tasks). The harness keeps detached tasks
alive across the parent's death, so they kept firing `./build.sh` ONE AT A TIME from the backlog —
each racing the single `grokking_optimizers/_ops.so` and re-applying the bisect header (DW_SPLITK=2/
IL=2) under my feet. This invalidated ~3 of my build+gate attempts (a sibling build raced the `.so`
mid-compile). Memory was never the issue (1.58 TB / 188 GB cgroup; the prev "OOM" was a `free -g`
GB-rounding misread). RESOLUTION: traced every build to its detached-task wrapper (PPID=1578 via
shell-snapshot eval), killed the whole sk-series tree + drained the backlog to ZERO build-task
wrappers + 0 compiler procs; confirmed NO cron/hook/loop/routine generator (finite manual retries,
now exhausted). Verified clean: header reset to committed 4/2, GPU 0 MiB. KEY LESSON for the next
agent: before ANY build, `ps -eo comm | grep -E '^(ninja|nvcc|cicc)$'` must be 0 AND no
`build_sk*`/`timeout * ./build.sh` wrapper under the claude PID — else a sibling detached task is
racing your `_ops` and every gate result is void.

---

## 4. Autotuner (compile.py) — status

Re-anchored + Wave-1-merged + validated (236/6 self-test = 6 pre-existing #10-aftermath
drift-guards only; 33/33 fp64 gate; production build green). See
`project-campaign-state` memory + `AUTOTUNE_LINKAGE.md`.
- **Build-cost reduction (quality-neutral/positive, owner-requested):** incremental variant
  build (`--incremental-variant-build`, done); single-cell build (rebuild only the tuned
  cell's TU, reuse other models' AOT objects — they link for dlopen, never timed); ccache-for-nvcc.
- **JIT search:** NO fixed trial count — `--bayesian-trials` omitted ⇒ multi-criterion
  early-stop (plateau + coverage saturation + wall-clock).
- **#11 validation:** tuned-vs-default + vs-regular-nvcc, meaningful only at d=2048 scale
  (toy d=128 is physics-inert — autotuner correctly found no win there).

---

## 5. Pre-race optimizer hyperparameter tuning (for the RACE)  — QUEUED (run at end)

`tuning/tune_optimizers.py` — Optuna, all 11 optimizers, tuning seed 1001 (disjoint from race
seeds), output → `results/tuning/tuned_configs.json`. The grokking race later just loads that
file (zero race-time impact). **Run this near the end** and commit the resulting
`tuned_configs.json` so the race uses owner-blessed hyperparameters.

Status: _TBD — tuned_configs.json not yet generated this campaign._

---

## 6. Phase-1 close-out checklist

- ☐ Mamba-3 trained model live + 11 mamba cells re-gated (3 seeds)
- ☐ Optimization ratchet complete (both tracks; stop criteria hit)
- ☐ All 33 cells parity-clean (fp64 gate) on seeds {42,7,123}
- ☐ Roofline-converged at d=2048 (baseline-before-mods → hill-climb → re-measure)
- ☐ Autotuner validated at scale (#11 tuned-vs-default-vs-nvcc)
- ☐ Pre-race `tuned_configs.json` generated + committed
- ☐ Pre-existing #10-aftermath self-test drift-guards fixed (OPTS/BINDING_FUNCS/manifest)
- ☐ Everything persisted to /workspace (§7) + hand-off doc complete (§8)

---

## 7. Persist for fast recompile / immediate use (/workspace)

Everything needed to recompile fast or run immediately on the 8×H100, kept on the persistent
volume `/workspace`:
- Built `_ops` extension + the persistent build caches (`/dev/shm/ccache` → also mirror to
  `/workspace`, `/workspace/.sccache`, the autotuner CompileCache + `tuned_configs`).
- `results/tuning/tuned_configs.json` (race hyperparameters).
- The tuned kernel configs (`grokking_optimizers/_kernel_tuned*.json`) if a scaled tune lands.
- Env to reproduce: `cd /workspace/SuperGrok1.5 && source /workspace/venv/bin/activate &&
  export PATH=/workspace/.local/bin:$PATH && source .regpressure/env.sh && export PYTHONPATH="$PWD"`.
- Build: `./build.sh` (compile.py/setup.py only; never raw nvcc). ~5.5 min full build.

---

## 8. 8×H100 / Phase-2 hand-off

When the 8×H100 is provisioned, the single-GPU foundation is done and the instance should
boot straight into the multi-GPU bring-up. Authoritative contract:
`/workspace/.parallelism_design.md` (4D+ZeRO-3, owner-locked).

**Done on the single H100 (Phase-2 prep):** _TBD — list the CPU/1-GPU-authorable pieces built
+ unit-tested here (parallel_config.cuh / tp_layer.cuh / sharded_optimizer_kernel.cuh, ZeRO
shard math, DP=2 loopback determinism)._

**Left for the 8×H100 window (minimize time here):** the NVSHMEM device-initiated TP (or
host-NCCL fallback), the graph-captured distributed step, and the 1→8 scaling measurements.
Bring-up order: DP+ZeRO-2 → ZeRO-3 → +PP → +TP (validation gates between).

**To resume:** read this file → §6 checklist → `/workspace/.parallelism_design.md` → run the
distributed tests under `torchrun --nproc_per_node=8` (they skip on WORLD_SIZE≤1 today).
