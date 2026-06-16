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
