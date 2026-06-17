# compile.py — line-by-line audit (proper + maximal?) — 2026-06-16

Full line-by-line audit of `grokking_optimizers/compile.py` (31,461 lines) + the Level-2
"superoptimizing additions", run as an 11-agent swarm (10 contiguous line-ranges + 1
cross-cutting mandate tracer), each checking **correct · complete · maximal · no-dead-code**
against the owner philosophy (MAXIMAL search space; NO fixed trial budget / early-stopping;
PTX **and** SASS in scope; correctness is a HARD gate) and the known gaps (#23 tiered spill,
#24 roofline objective, fp64-gate wiring).

## Executive summary
**The search BACKBONE and the correctness GATE machinery are production-grade and honest.** The
auto-knob scan (`#ifndef SG_TUNED_*`), the ~10⁶–10¹³ per-arch space, the real convergence-based
early-stopping (`n_trials=None`, `max_trials=1M` is only a safety rail), the transitive-include
cache-hashing, the default-ON learned cost model that genuinely prunes, and the
`_VALIDATION_REQUIRED_ORIGINS × {ok,deterministic}` winner gate are all real and well-defended.
The 242-case self-test suite is fully wired with a runtime count-guard.

**The gaps are four themes:** (A) the gate is the wrong oracle (fp32 self-consistency, not fp64);
(B) the maximal machinery is wired OFF by default and the objective is raw-ms not %-roofline;
(C) the Level-2 superoptimizer is ~70% scaffold that the gate correctly prevents from ever
winning — *except* one fake-green hole; (D) cache toolchain-identity blind spots.

---
## P0 — BLOCKERS (correctness / fake-green). Fix before trusting any autotuned winner.

1. **fp64 oracle NOT wired into the winner path.** `pick_winner` (6659–6701) accepts winners
   on a *same-dtype strict-fp32 self-consistency* oracle (`_capture_reference_output` 13816,
   `_compare_outputs` 13878 `np.allclose`) + a 3× same-size determinism tag — never the fp64
   ground-truth gate (`tests/hw/test_l3tc_tail_gate.py`). Two kernels sharing a rounding error
   agree and pass. **The IL=4 trap (bench-faster, determinism-FAIL) was caught only by the
   out-of-band hardware gate, not the tuner.** FIX: inject an fp64+A/A/A hook (like `tune_hook`)
   that re-checks the top-K candidate winners at production scale (d=2048) and demotes any
   failure to ineligible — make fp64 a hard precondition of `pick_winner`.

2. **Polyhedral / CUTLASS / CK winners ship the TEMPLATE while reporting a generated origin
   (fake-green).** `build_jit`'s final source-swap (17183) checks ONLY `== ORIGIN_SYNTH`; a
   polyhedral winner leaves `final_sources = sources` (the template, 17181) yet records
   `origin=polyhedral` at the winning timing. Reachable because `build()` defaults
   `enable_polyhedral=True` (17276) + the mirror (17509) force-enables it. FIX: generalize 17183
   to any `origin in _VALIDATION_REQUIRED_ORIGINS` with a stashed `_emitted_sources[ckey:origin]`;
   reuse the synth warn-and-fall-back-to-template branch (17191).

3. **IL=4 determinism trap OPEN by default.** `pick_winner` only *enforces* the `deterministic`
   tag when `strict_numerics=True` (6696); the default path lets a `non_deterministic` config
   (atomicAdd / IL=4 reduction-order) that merely clears `np.allclose` WIN. FIX: reject
   `numerical_status == "non_deterministic"` unconditionally (it's already measured for free);
   reserve the opt-out for an explicit `--allow-nondeterministic`. (Also default `strict_numerics=True`.)

4. **Fast-math cache signature drops version-gated flags → stale `.so` can win post-upgrade.**
   `fm_sig` (14118) omits `_version_gated_flags_for_hash` that the main `build_sig` folds in
   (15226). After a gate-crossing compiler upgrade a stale fast-math `.so` is re-served as a
   trial and (fast-math being a validation-required origin) can win on stale timing. FIX: fold
   the version-gated tuples into `fm_sig` exactly as the main path does.

---
## P1 — MAXIMALITY (the owner philosophy). High leverage; mostly turning ON what exists.

5. **The CLI INVERTS "maximal".** A plain `python -m grokking_optimizers.compile -O adamw -M
   decoder` runs with **PGO, device-PGO, emitter, synth, polyhedral, runtime-specialization,
   transfer-learning all OFF** and `pruner=none` — the seven `store_true` flags + pruner default
   (18808–18834) OVERRIDE `build()`'s documented `True` maximal defaults (17269–17276). The
   default invocation is maximally *conservative*. FIX (highest leverage, ~1–2h): tri-state the
   flags (`--no-X`, `default=None`, pass `getattr(args,X,None)`) so absent → build()'s True
   survives — exactly how `--cost-model`/`--multi-fidelity` already do it. Also `--pruner`
   default → `hyperband` (or pick one source of truth: build()=hyperband vs BuildSpec=none).

6. **#24 roofline objective ABSENT.** The objective is raw `timing_ms` end-to-end (Optuna
   `minimize`, 6335; winner `min(key=timing_ms)`, 6701). No per-arch peak-TFLOP, no
   `achieved_tf = FLOPs/wall`, no %-of-roofline, no sub-ceiling flag — the roofline %-numbers
   live only in the offline `tuning/roofline.py`. FIX: add `peak_tf`/`peak_bw` to `ArchEntry`;
   compute achieved-TF in the timer; switch the objective to maximize %-roofline; flag a winner
   that lands below a ceiling threshold (so a structurally-capped 2–4% result is surfaced, not
   reported as "tuned"). [#24]

7. **#23 tiered register-spill mgmt — PARTIAL, nothing reacts.** The hard parts exist — a
   `maxrregcount` cap sweep (2568) and a working ptxas spill-bytes parser (`_parse_ptxas_v_stderr`
   12885) — but the parsed `spill_stores`/`stack_frame` are recorded to the sidecar and consumed
   by NOTHING. No penalty, no escalation, no enforcement of "never local/DRAM", and they are not
   cost-model features (which use only derived proxies like `maxrregcount/255`). FIX: in
   `pick_winner`/`_make_trial_record`, treat nonzero local-spill as a tier breach (exclude or
   penalize), add `spill_stores`/`stack_frame` as measured cost-model features, and make
   `--def-load-cache`/`-Xptxas -dlcm`/`--register-usage-level` *tuned dims* (today pinned base
   constants, 12083–12155). Add the genuinely-uncapped maxrregcount point (sentinel → omit flag;
   today `0` is aliased to `=255`, different SASS). [#23 + B-flag-resolution]

8. **Byte-identical "untuned build" claims are FALSE.** The sm_90 builder's first/canonical
   values diverge from the kernel `#define` defaults: `dec_dw_splitk` 4 vs **1**, `vit_dw_splitk`
   4 vs **1**, `cons_regs` 200 vs **232**, `prod_regs` 32 vs **40** (A, ~2121–2178). The no-JSON
   build is therefore NOT the committed default — it would NOT reproduce the dW `splitk=1` KEEP.
   FIX: reorder each value list so the first element equals the in-kernel default. (Direct,
   safe, protects the landed dW win.)

9. **SASS is not in scope as inspection.** PTX/ptxas flags are tuned, but `cuobjdump
   --dump-sass`/`nvdisasm` is never run (`--sass-only` only trims the fatbin). SASS *rewriting* is
   genuinely blocked (no assembler — see SUPEROPTIMIZER_SCOPING.md), but SASS *inspection* is
   feasible. FIX (optional, for the PTX-AND-SASS ask): a read-only `cuobjdump --dump-sass` pass
   recording real reg-alloc / spill-instructions / instruction-mix into the trial sidecar to
   inform selection.

---
## P2 — Level-2 superoptimizer is ~70% scaffold (J). Gate prevents wins, so no corruption — but no maximality added today either.
- **Polyhedral** (27147): `apply_schedule` discards the real kernel signature and emits a fixed
  `launch_polyhedral_kernel(out,in,n)` — body is a hardcoded identity copy when libclang body-lift
  returns None (the common path); dependence analysis `_heuristic_dep_vectors` (27464) is a stub
  returning all-parallel `(0,)` vectors; loop bounds are token-scraped (fabricated trip counts).
- **Synth/OpGraph** (27947): the "AdamW"/"fused Adam" patterns (28267) never update m/v (consume
  stale moments → numerically wrong as a training step); synth `.cu` exports the wrong ABI symbol
  (`synth_elementwise_*`, not `launch_<opt>_step`) so the loader can't call it → always
  `numerical_fail`. Native wgmma/tcgen05/mfma/wmma mainloop is a hard-OFF zero-fragment STUB
  (`_MMA_NATIVE_LOADS_WIRED=False`, 28740) → synth GEMMs fall back to a scalar triple-loop.
  `pattern_bilevel_fusion` just concatenates graphs (no real fusion). `_emit_scan_cuda` max-scan
  uses identity `0` not `-INF` + single-block only (bug).
- **CUTLASS/CK** emitters are real + compilable but never wired into the decoder/vit/mamba
  megakernel timing path (only reachable for muon/sg2 GEMM patterns under synth).
- **Decision needed (owner fork):** implement these for real (large) vs honestly mark dormant
  (set the Python `build()` defaults to match the CLI `False`, fix the scoping-doc claim, and
  gate behind explicit opt-in) so the default sweep doesn't waste compiles on layers that cannot
  win. Recommended near-term: **honest-dormant + fix blocker #2**; revisit "implement for real"
  only if Level-0/1 dries up.

## P2 — cache toolchain-identity (F)
- nvcc **patch** version not hashed — same-gate-bucket upgrade (12.6.0→12.6.3) silently HITs a
  stale binary. FIX: fold `nvcc --version` major.minor.patch into the version-gated hash.
- JIT-freshness + PGO-record hashes are inconsistent with the AOT version-gated basis → JIT
  re-serves a winner picked for a different binary; **PGO pass-3 record hash can NEVER match its
  freshness check (16977) → every build re-runs full 3-pass PGO** (cache-thrash). FIX: record with
  the same basis the freshness check uses.
- `compile_config` module imported by `build()` (17473) **does not exist** → `ModuleNotFoundError`
  swallowed → all TOML config knobs silently ignored on the live path (the in-file
  `apply_to_buildspec`/`DEFAULT_CONFIG` run only in self-tests). FIX: ship the module or route
  `build()` through the in-file `load_config`.

## P2 — timing methodology (C)
- **Multi-GPU clock-lock is half-off:** `_GpuClockLock` pins only GPU 0 (device_index=0,
  hardcoded) while `MultiGPUTimingPool` times across all visible GPUs on free-boost clocks. Also
  wrong under `CUDA_VISIBLE_DEVICES` remap. FIX: one lock per pooled device.
- **L2 flush sized at exactly L2 capacity + write-only** → under-evicts (biases toward
  L2-resident configs). FIX: ~2× L2 and/or read+write sweep.
- Dead duplicate top-level timers (`cuda_graph_median_ms`/`event_median_ms`, 3819–3927) diverged
  from the live `_bg_*` worker copies; the dead `event_median_ms` still carries the
  default-generator crash the live copy was fixed for. FIX: delete or single-source.
- In-graph `p.grad.copy_(g)` (64 MB D2D) inside the timed region inflates every variant +
  re-warms L2. FIX: capture only `opt.step()`.

## P3 — cost model (D) & dead code
- 14 stall-reason features are **permanently zero** (`cost_model_state["stall_info"]` init None,
  never written) — device-PGO stall data goes to a separate local, never fed back. Consistent
  (zero at train+predict) so not a skew bug, but the feature set isn't the maximal one claimed.
- `_cost_model_compute_feature_dim()` is **dead** (never called); the `FEATURE_DIM=14` hardcode is
  guarded by a self-test the comment promises but that **does not exist** → silent schema-drift
  risk. Seed-prediction sibling rows leak into `mae_val` (optimistic).
- Phantom dead search dims: ~~`mb_dw_splitk`~~ **(REMOVED 2026-06-17 — Mamba-3 TC rewrite dropped
  the output-stationary dW split-K; macro no longer #defined. Also corrected the `dec_dw_splitk`
  table default 4→1 drift the header ratchet introduced. `test_macro_drift_against_header` now
  green.)** / `mb_gemm_stages`/`mb_gemm_interleave` + `min_blocks` (still no kernel macro →
  auto-pinned dead, never sweep; comments claim otherwise — separate cleanup). Blackwell
  sm_100a/103a/120a route through the generic builder → no TC tile surface (multi-arch maximality
  is really single-arch sm_90). ~10 stranded section-header comments in the self-test tail (nit).

---
## VERIFIED-GOOD (so we don't "fix" what's right)
Auto-knob scan (no curated lists) · real convergence early-stopping (no hidden fixed budget) ·
transitive-include cache hashing (stale-header class closed) · cost model default-ON + actually
prunes (dual rejection caps, cold-start floor, conservative multi-fidelity) · the
`_VALIDATION_REQUIRED_ORIGINS` winner gate is real and fail-closed for synth · the post-main
range is all live (242-case self-test, no orphan functions) · fp32 compare upcasts to float64 +
`equal_nan=False` (NaN can't masquerade as ok).

---
## Fix priority (front-loaded; ~12–20h total, most value in the first ~6–8h)
1. **P0 correctness (≈3–5h):** #1 fp64-gate hook · #3 reject non-deterministic by default ·
   #2 generalize the origin source-swap · #4 fm_sig version flags. → makes winners trustworthy.
2. **#8 byte-identical defaults (≈30m):** reorder value lists to the kernel defaults. (Protects
   the dW KEEP; do first — trivial + safe.)
3. **#5 CLI maximal-flip (≈1–2h):** tri-state the seven layers + pruner. → turns the existing
   machinery ON.
4. **#6 #24 roofline objective (≈3–5h)** · **#7 #23 tiered-spill reaction + PTX dim widening (≈4–6h).**
5. **P2 cache-identity + timing fixes (≈2–4h).**
6. **Superopt fork (owner decision):** honest-dormant (≈1–2h) vs implement-for-real (large).

*Sources: 11 audit agents over the full file; line refs are to compile.py @ HEAD ba19e29 unless noted.*
