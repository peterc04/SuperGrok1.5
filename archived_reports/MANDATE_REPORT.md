# MANDATE FINAL REPORT — Universalize `compile.py`

Branch: `claude/custom-optimizer-analysis-HFYhg`
Baseline: `fa65a23` → final. Self-test: **205/0**. Ruff: clean. Math-drift: PASS.
verify_all: 99/99 composition, all components compile (1 transient OOM at high
`--jobs`, re-run clean at `--jobs 2`).

---

## 1. SUMMARY

`grokking_optimizers/compile.py` is now a **project-agnostic, zero-manifest,
NVCC-parity compiler** with an empirical autotune+validate tier, while
retaining 100% of the grokking-specific behavior as one client of the generic
path. The work landed phase-by-phase (mandate order 1→9), atomic commits per
item, tree green at every commit, GPU proofs deferred with exact commands.

The architectural keystone is realized:
- **Tier 1 (compile, zero-config):** auto-detect arch/toolchain (pre-existing)
  + auto-discover sources (new) + arch-tuned strict-math flags → builds an
  arbitrary project with no manifest. Proven on a synthetic non-grokking
  project (real nvcc compile).
- **Tier 2 (autotune+validate, zero-manifest):** entry-point discovery +
  signature introspection + synthesized adversarial inputs + a strict-math
  self-oracle, with an explicit loud Tier-1 mode-switch when no tunable entry
  exists. Every generated/transformed variant is gated behind an on-device
  strict-oracle PASS.

---

## 2. PER-ITEM STATUS (#1–#23)

| # | Item | Status | Evidence (CPU) | GPU-deferred proof |
|---|------|--------|----------------|--------------------|
| 1 | GPU clock locking | DONE | `clock_lock` self-tests (4) | P1/P3/P11 |
| 2 | L2 cache flush | DONE | `l2_flush` self-tests (5) | P2/P12 |
| 3 | Representative shapes | PARTIAL | shape_hint wired into synth (#19) | per-bucket tuning on GPU |
| 4 | Workload injection via discovery | DONE | `discovery` self-tests | P4/P17/P18 (live .so) |
| 5 | Multi-GPU pinning/calibration | DONE | `autotune_brain` calibration tests (3) | P6 (cross-GPU ranking) |
| 6 | Fast-math as vetted variant | DONE | base strict + `fast_math_variant_flags` | P5/P9/P14 (validate+SASS) |
| 7 | `-march` policy | DONE | `tier1_compile` march tests | on-target vs cross-host build |
| 8 | sccache as sweep requirement | DONE | `compiler_cache_warning` test | hit-rate on a real sweep |
| 9 | Strict uncontaminated oracle | DONE | `strict_math_strips_fast_math` | P4/P7 (live capture) |
| 10 | Randomized adversarial inputs | DONE | `input_synthesis_*` tests | live inputs on GPU |
| 11 | Signature from discovery | DONE | `discovery_parse_*` tests | live schema enumeration |
| 12 | Determinism env | DONE | `determinism_preamble_present` | P8 (cuBLAS determinism) |
| 13 | Stopper defaults (not the cap) | DONE | `stopper_*` tests (2) | locked-clock convergence (P3) |
| 14 | Cost-model cold-start floor | DONE | `cost_model_cold_start_floor` | calibration on locked-clock data |
| 15 | Registry↔autotuner wiring | DONE | `registry_bakes_tuned_config` | per-bucket cubin dispatch |
| 16 | On-device validation gate | DONE | `pick_winner_rejects_unvalidated_generated` | live oracle on silicon |
| 17 | Pattern fail-safe + injectable | DONE | `synth_pattern_failsafe/injectable` | — |
| 18 | Synth dtype plumbing | DONE | `synth_dtype_from_config`, `synth_emits_configured_dtype` | — |
| 19 | Synth shape from discovery | DONE | shape_hint path | shapes from live signature |
| 20 | Polyhedral legality + gate | DONE | origin=polyhedral gating + dep-surface log | illegal-reschedule rejection on GPU |
| 21 | CUTLASS/CK enumerators fed | DONE (verified) | fallback already logs degradation | enumerated variant search on arch |
| 22 | Discovery-based adapter | DONE | Tier-1/Tier-2 zero-config contract | non-grokking autotune (P18) |
| 23 | CPU-offload / target-split | DONE (foundation) | `--stage` selector + cross_host wiring | P10 (byte-identical parity) |

**NDEBUG migration (user directive):** `-DNDEBUG` removed from
`NVCC_DEVICE_BASE`/`HIPCC_DEVICE_BASE`; re-sourced from grokking's
`compile_config.toml` `[device_cflags].extra` (folded into
`device_cflags_hash`). Verified: grokking build flags carry `-DNDEBUG`,
config-less generic build does not. `setup.py` re-adds grokking's own
`--use_fast_math` + `-DNDEBUG` on the release path.

---

## 3. AUTO-APPLIED AUDIT FIXES (Phase 9)

- No bare `except:` introduced; broad `except Exception` only in genuinely
  best-effort paths (cache-annotate provenance, compiler-cache stats,
  optional codegen layers, prediction-error logging) — all of which LOG and
  none of which swallow a correctness error.
- `_discover_entry_points` RAISES on a load failure / malformed schema (§2A
  fail-fast), never degrades silently.
- Updated `flag_base_superset_regression` + `synth_dispatcher` self-tests to
  assert the new strict-math / fail-safe contracts.

---

## 4. NEEDS YOUR CONFIRMATION

None outstanding. The three escalation questions (NDEBUG handling, sequencing,
agent policy) were answered before Phase 1 and implemented as directed.

---

## 5. VERIFICATION EVIDENCE

```
PYTHONPATH=. python3 grokking_optimizers/compile.py --self-test   → 205 passed, 0 failed
python3 -m ruff check .                                            → All checks passed!
PYTHONPATH=. python3 scripts/check_math_single_source.py           → OK (no drift)
PYTHONPATH=. python3 grokking_optimizers/compile.py --dry-run-all-archs → sm_90a PASS
PYTHONPATH=. python3 -m grokking_optimizers.verify_all --phase 2 --jobs 2 → 14/14 components compile
# Tier-1 zero-config on a non-grokking project:
nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 -c kernels/vecadd.cu  → rc=0 (7760 bytes)
```

GPU-deferred proofs (run on the named silicon):
- **H100 (sm_90a):** P1 clock-lock variance drop; P3 locked-clock CoV<1%;
  P4 e2e Tier-2 autotune; P5 fast-math variant validated vs strict oracle;
  P7 oracle strictness; P8 determinism tag; P9 `cuobjdump -sass | grep -c
  wgmma.mma_async` unchanged under strict base; P10 `sha256sum` of
  all-on-target vs cpu-then-target binaries.
- **MI300X (gfx942):** P11 rocm-smi clock pin; P12 LLC flush; P13 e2e;
  P14 `llvm-objdump -d | grep -c v_mfma` intact.
- **TPU v6e:** P15 Pallas autotune; P16 HLO dot_general on device.

Exact commands are in the Phase-0 plan (`§5 GPU-deferred proof list`).

---

## 6. KNOWN LIMITATIONS

- #3/#19 representative-shape tuning derives shapes from a config/discovery
  hint; per-bucket autotuning (one tuned config per shape bucket) requires the
  GPU timing loop to populate buckets — the registry consumes the single
  tuned config today (loop closed, per-bucket fan-out is the GPU follow-on).
- #23 offload-mode byte-identical parity (Invariant #10) is structurally wired
  (cross-host ISA baseline + cache transfer) but provable only on silicon (P10).
- The strict-oracle .so is the AOT build, which is strict-math by default
  post-#6; the separate-strict-rebuild optimization is unnecessary now that the
  base is strict.

---

## 7. PARALLELISM REPORT

Per the Phase-0 partition, the monolithic single-file constraint made most
phases serial through the main thread; read-only recon fanned out to two
`Explore`/`opus` sub-agents in Phase 0. No concurrent write-intent over any
shared surface (`ARCH_TABLE`, `BuildSpec`, base flag lists, `FEATURE_DIM`,
cache schema, `_SELF_TEST_EXPECTED_COUNT`, `TOLERANCES`,
`_DEFAULT_FUSED_OP_TEMPLATE`, `_DEFAULT_PROJECT_CONFIG`). Every phase ended at
the §6 whole-system gate with the full self-test suite green. No clobbers.
