# Session checkpoint / resume state (LIVE — updated frequently for termination-safety)

> Resume after an unpredictable termination. **Source of truth = git commits on `claude/h100-audit-maximal`
> (LOCAL-ONLY, never push) on `/workspace` (persistent).** Read: this doc → `OPTIMIZATION_LEDGER.md` →
> the captured roadmaps in `.perf/*.json` → `git log --oneline -15` + `git status`.

## HEAD lineage (most-recent first; all committed = durable)
- `<checkpoint commits>` capture roadmaps: `2ca44be` needs-care-unblock · `749f70c` phase2 bring-up · `b1d22e7` perf-sweep + SESSION_STATE · (+ bug-hunt capture just committed).
- `eff3e78` audit: 5 verified-safe fixes — **PENDING the combined fp64 gate (gate agent in flight); REVERT if red**.
- `5c73b5d` decoder **BAKE PIPE=1/STAGES=4** (ships +1.49×; gate-validated 618ms/6.477%, 11/11×3).
- `fda3454` ViT **B1 sub-tiling** knob `SG_TUNED_VIT_P1_SUBTILE_S` — gated OFF=byte-identical; **pending flip-gate**.
- earlier: `a3371f6`/`1e0d194` docs, `df13655` dead-artifact cleanup, `74260dd` PIPE=2 wiring (PIPE=2 LOST its gate).

## In-flight (VOLATILE; read-only analysis → re-runnable, NO code at risk)
- **gate agent `a3a0c742`** (GPU): rebuild `_ops` + full fp64+A/A/A gate validating `5c73b5d`+`eff3e78`. Verdict → GREEN keep; RED `git revert eff3e78` (D; C pre-validated) + re-gate.
- **C0-design workflow `wddaf8dcg`**: #27 Level-2 C0 MVP build spec.

## Captured analysis roadmaps (durable, in `.perf/`)
`perf_lever_sweep_roadmap.json` · `phase2_bringup_plan.json` · `needs_care_unblock_plan.json` · `megakernel_bug_hunt.json`. **Analysis is now SATURATED — execution (serial GPU + commits) is the bottleneck; do NOT spawn more analysis workflows.**

## Serial GPU ratchet QUEUE (priority order; single H100, one build+gate at a time; delegate each gate, I decide keep/commit)
1. **(awaiting)** combined gate verdict of `5c73b5d`+`eff3e78`.
2. **SG11 cosine-gate correctness fix** (HIGH — correctness>perf): `opt_stages_precompute.cuh:540-547` reduces `cos(grad,mu)` → must be `cos(grad,st.exp_avg)` (momentum); route via canonical `sg11_sweep_a_step`; FIX the co-wrong oracles (`test_opt_stages.py:201-226`, `test_l3tc_tail_gate.py:777-778`) + re-gate with a WARM-UP step (step-1 coincidence blinds it) + harden `check_math_single_source.py`. Affects all 3 SG11 cells. **Re-verify the operand vs live code before editing.**
3. **ViT B1 flip-gate** (#1 perf lever, 51%): `-DSG_TUNED_VIT_P1_SUBTILE_S ∈ {16,8,4}` at **d2048/B1024** (NOT B16384). NEEDS-PARITY; avoid S=1.
4. **Re-profile (gate-free diagnostics)** — the relevance gate is on STALE numbers: re-profile decoder @618ms (`decoder_bench.py --profile`); wire the mamba per-phase profiler (perf-sweep rank-1: fix `fused_mamba_megakernel.cuh:397-405` slot bracketing + add launcher readout + `mamba_bench.py --profile`).
5. **decoder TILE_M 128→64** (#30) — RE-EVALUATE after #4 (B1 share may be below the relevance floor post-1.49×).
6. perf-lever-sweep's other surviving levers (ranks 3+, in the roadmap json).

## Lower-urgency / deferred (CPU; not blocking the local ratchet)
- **CI-unblock** (`needs_care_unblock_plan.json`): re-anchor `verify_all.py` + `ci.yml` (compile_to_object + cpp_structural + docs_consistency) + README — **only matters when pushed (local-only now)**. Fix #4 is an owner decision (README rewrite).
- **mma.cuh removal** (zero-ref confirmed) → drops `third_party/cutlass` (118M) = build-speedup; touches the build → gate it.
- **Phase-2 PP fixes** (`phase2_bringup_plan.json`): regenerate patch 0001, fix the PP kernel signature/carve, factor a shared `dec_tc_bind_workspace` helper (touches the production decoder TU → parity-gated). Front-load before the 8×H100 lands.
- **M0 mamba scaffold**: `git stash@{0}` + `.perf/M0_mamba_integration_scaffold.patch` (gated OFF; revisit after the mamba profiler is wired).

## Mode / constraints
Ultracode ON (xhigh + workflows), Opus 4.8 1M, `dontAsk`. Agents work in the MAIN tree despite worktree isolation + may `git stash`/`checkout` → **serialize anything that builds/gates/mutates**; read-only analysis fans out freely.

## Resume protocol
1. `git log --oneline -15` + `git status --porcelain`. 2. If the combined-gate verdict for `eff3e78` is unknown, **re-run it** before trusting D. 3. Work the serial GPU queue above in priority order (SG11 correctness fix is high). 4. Capture every workflow `.output` from `/tmp` → `.perf/` + commit promptly (termination-safety). 5. Commit progress frequently; refresh THIS doc at each major checkpoint.
