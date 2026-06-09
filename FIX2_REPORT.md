# FIX #2 — FINAL REPORT

Branch: `claude/custom-optimizer-analysis-HFYhg`. Self-test **220/0**, ruff clean,
math-drift PASS, `--dry-run-all-archs` sm_90a PASS. Audit converged in **2 iterations**.

---

## 1. DEFECT 1 — fast-math (#6) was dead code; now a real, oracle-validated variant

**Before:** `_FAST_MATH_VARIANTS` / `_fast_math_variant_flags` had only a self-test
call site; `_LAST_VARIANT_ORIGIN` was never stamped `"fastmath"` in the real path.
The sweep never produced/timed/validated a fast-math variant. Only "remove from base"
worked.

**After (commits `5026818`, `c7e9…`):**
- `_time_validate_fastmath_variants` builds each `_FAST_MATH_VARIANTS` entry ADDITIVELY
  on the strict build's flags (base lists untouched), times it, validates its output
  against the SAME strict oracle, stamps `origin=ORIGIN_FASTMATH` + numerical_status,
  and emits a SEPARATE trial into a sink.
- The timer runs the pass on each NEW-BEST config (bounds extra compile cost, always
  covers the eventual winner). Both drivers drain the sink into `pick_winner`.
  `_run_exhaustive` now selects via `pick_winner` too — closing a latent bug where its
  inline selection bypassed the #16 gate entirely.
- Added flags feed `build_sig`/cache hash; `build_jit`'s final rebuild re-applies the
  fast-math flags when the winner carries `_fastmath` (Bug 1 fix below).
- Loud-not-silent: N/A log for Pallas, WARNING when zero variants generated on CUDA/HIP.
- **Footgun killed (1.6):** `ORIGIN_*` single source of truth; every origin write, the
  gate set, and tests use the constants. A stray `"fast_math"`/`"fast-math"` can no
  longer silently bypass the gate.

**INTEGRATION evidence** (the test class that would have caught the miss —
`_self_test_fastmath_integration`, 7 tests, all green):
- `fastmath_variants_actually_produced` — a real (mock-timed) pass yields 2
  `origin=fastmath` trials that enter the stream.
- `fastmath_divergent_marked_fail_then_gated` — a divergent variant → `numerical_fail`
  → `pick_winner` refuses it even when fastest.
- `fastmath_skipped_never_wins_validated_can` — unvalidated (`skipped`) fast-math can
  NEVER win; validated CAN.
- `fastmath_pallas_is_na_loud`, `fastmath_wired_into_sweep_drivers`,
  `fastmath_winner_reapplies_flags_in_final_build`, `origin_constants_single_source_of_truth`.

**Grep proof:** `_fast_math_variant_flags`/`_FAST_MATH_VARIANTS` now have non-test call
sites (compile.py:11944-11945); the only `"fastmath"` literal is `ORIGIN_FASTMATH`.

**GPU-deferred proof (H100):** a fast-math variant built, validated vs the strict oracle,
and won-or-rejected on a live sweep —
`PYTHONPATH=. python3 grokking_optimizers/compile.py --optimizer adamw --model decoder --arch sm_90a --runtime jit --mode bayesian`
(blocked by the escalated validation-mechanism fix in §3).

---

## 2. DEFECT 2 — silent broad-except → §2A degradation visibility

**Before:** ~154 broad `except Exception` blocks degraded silently (pass/continue/
assign-default, no log/raise). **After:**
- `_debug_swallow(component, exc, detail)` — one stderr line at `_COMPILE_LOG_LEVEL>=2`,
  quiet by default. Applied to 151 benign optional-feature/cleanup/observability
  swallows. **AST proof:** zero pass/continue broad-excepts remain in production code
  (`test_no_silent_broad_except_in_production`). Iteration-2 audit verified all 156 sites
  preserve control flow and reference the correct exception variable.
- **Priority (calibration):** a per-device failure WARNS naming the device (offset
  dropped, ranking may skew) instead of silently appending None; a TOTAL failure RAISES
  when `require_calibration=True` (cross-GPU modes), instead of a degenerate all-1.0.
- **§2A crash payload:** `_render_repro_state` dumps resolved host+device flags, arch,
  optimizer/model/dtype/shape config, sources, include paths, and env (CUDA/HIP/JAX home
  + sccache) into the build-failure report — a crash is fixable from its output alone.
  Existing crash visibility (summary/traceback/ninja-logs/worker-tb/CLI-tb) preserved.

**Acceptance:** `_self_test_silent_degradation` (6 tests) — debug-swallow quiet/loud,
calibration-warns, total-failure-raises-when-required, require-kwarg-isolated,
repro-state-payload-complete, AST no-silent-broad-except.

---

## 3. NEEDS YOUR CONFIRMATION — pre-existing validation-mechanism mismatch (GPU-gated)

**This is the one item I did NOT auto-apply** (per apply-vs-escalate: pre-existing,
GPU-unverifiable, perf-vs-correctness tradeoff). It is the single live defect both audit
iterations converged on.

**Finding:** on-device numeric validation (#9/#10/#16) is structurally broken and has
never actually run:
1. **Persistent-worker no-dump:** the timer sets `os.environ["SG_DUMP_OUTPUT"]` in the
   PARENT, but the persistent `TimingWorker` subprocess was spawned earlier with a frozen
   env — it never sees the var, never dumps. Only the one-shot fallback dumps. So under
   the normal (persistent-worker) path, `num_status` stays `"skipped"` → `pick_winner`'s
   #16 gate drops every generated variant (synth/polyhedral/cutlass/ck/fastmath). The new
   fast-math path replicates this same pattern.
2. **Oracle↔timing semantics mismatch:** the oracle (`_capture_reference_output`) runs ONE
   raw-op call on a 1-D `(size,)`=(4096,) tensor via `_render_arg_construction`; the timing
   side-dump (`_TIMING_SCRIPT`) dumps `p` after warmup+iters `opt.step()` calls on a 2-D
   `(4096,4096)` tensor. `_compare_outputs` hits a shape mismatch → `numerical_fail` even
   on the one-shot path. So validation either skips (worker) or fails-on-shape (one-shot)
   — it never produces a meaningful PASS.

**Recommended fix (decision-ready):** decouple validation from timing — in the timer's
numerical pass, replace the `SG_DUMP_OUTPUT` side-channel with an explicit
`_dump_variant_output(variant_so, opt, ref_state["size"], ref_state["dtype"], out_dump,
entry=ref_state["entry"], regime="normal", seed=0)`. That helper ALREADY exists and uses
`_render_arg_construction` — IDENTICAL inputs/shape/op-call/step-count to the oracle — so
the comparison becomes valid AND it runs as a fresh subprocess (no frozen-env issue),
fixing both sub-issues at the root. (Store the chosen entry in `ref_state["entry"]` in
`_resolve_ref`.)

**The tradeoff you need to decide (why I escalated):** an explicit dump-run is +1
subprocess per VALIDATED config. Options:
- **(A) validate every config** — fully correct, ~2× sweep subprocess cost.
- **(B) validate only what can win** (recommended): generated-origins always (#16 needs
  it) + strict configs only on new-best or under `--strict-numerics`. Bounds cost,
  guarantees the winner + every eligible generated variant is validated. Needs a gating
  policy (the design choice).
- **(C) leave as-is** — fast-math/synth/etc. remain fail-closed (can't win) until a real
  GPU validation path exists; strict-only winners ship correctly.

Blast radius: the timer's numerical pass + `_resolve_ref` (store entry) + the fast-math
pass. GPU-required to prove the numeric result; the input-matching correctness is
structurally provable (both sides call `_render_arg_construction`). My recommendation: **B**.

---

## 4. DEAD-CODE-VS-TESTS SCAN (the NEW required dimension)

An AST scan for mandate features with NO non-test caller found **4 implemented-but-unwired
features** (the exact class that hid the fast-math miss) — all now wired:

| Feature | Was | Now wired into |
|---------|-----|----------------|
| `_discover_entry_points` / `_select_tunable_entry` (#4/#22) | test-only | `_resolve_ref` (Tier-2 oracle FALLBACK: template-first, discovery on miss) |
| `_strict_math_flags` (#9) | test-only | `_resolve_ref` (oracle fast-math-contamination guard, loud warn) |
| `_synthesize_input_spec` (#10) | test-only | `_render_arg_construction` (records reproducible input spec / validates regime) |

**Institutional guard added:** `test_mandate_features_have_nontest_callsites` — an AST test
that FAILS if any of 21 mandate features is reachable only from `_self_test_*`/`test_*`.
This would have caught fast-math AND these four; it permanently guards the whole bug class.

Full call-site proof (all OK after wiring): fast-math, clock-lock, L2-flush, discovery,
march policy, synth dtype/pattern, variant-reuse, repro-state, debug-swallow, strict-math,
arg-construction, input-spec, synth-codegen, polyhedral, registry-init, oracle-capture.

---

## 5. AUTO-APPLIED AUDIT FIXES (each with covering test)

1. **Bug 1 — fast-math winner shipped the wrong `.so`** (interaction audit): `_fastmath`
   is not a search-space dim, so `_variant_macros` dropped the fast-math flags and
   `build_jit`'s final rebuild produced a STRICT `.so` that couldn't reproduce the winning
   timing. Fix: re-apply `_fast_math_variant_flags(name, vendor)` when the winner carries
   `_fastmath`; the recorded `device_flags_hash` now reflects it. Test:
   `fastmath_winner_reapplies_flags_in_final_build`.
2. **4 dead features wired** (§4) + institutional guard test.
3. **Cosmetic:** corrected the contamination-warning flag-count over-count (log-only).

---

## 6. VERIFICATION EVIDENCE

```
--self-test                              → 220 passed, 0 failed   (was 206 → +14: 6 fastmath-integration,
                                                                    6 silent-degradation, mandate-features guard, bug-1)
ruff check .                             → All checks passed!
scripts/check_math_single_source.py      → OK (no drift)
--dry-run-all-archs                      → sm_90a PASS
AST: silent pass/continue broad-excepts  → 0 in production
AST: mandate features w/ non-test caller → all OK
```

Capability-regression (Invariant #4): self-test/ruff/drift/dry-run all green; no
`-ffast-math`/`--use_fast_math`/`-DNDEBUG` back in base lists; `pick_winner` still rejects
unvalidated generated variants (10-case logic + the new integration tests); crash/build/
worker/CLI dumps still fire; `_run_exhaustive` winner persists via `set_tuned`.

---

## 7. AUDIT ITERATIONS & TERMINATION

- **Iteration 1:** whole-system interaction trace (oracle↔pick_winner↔drivers, sink↔cache
  hashing, SG_DUMP race, variant-reuse↔sidecars, debug-swallow injection, cost-model↔
  stopper, `_LAST_*` races) + dead-code-vs-tests AST scan. Found: Bug 1 (fixed), 4 unwired
  features (fixed), the validation-mismatch (escalated).
- **Iteration 2:** re-audit of all code introduced this session (bulk transform at scale,
  discovery wiring, regime-raise, genericness, locks, cache-hash fast-math axis). Found:
  **nothing new** beyond the escalated validation-mismatch; 2 cosmetic notes (1 fixed, 1
  documented — adversarial regimes wired-but-defaulted-to-normal).

**Terminated:** iteration 2 found no new defects → converged.
