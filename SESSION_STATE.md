# Session checkpoint / resume state (LIVE — updated frequently for termination-safety)

> Operational checkpoint so a fresh or resumed session picks up instantly after an unpredictable
> termination. **Source of truth = git commits on `claude/h100-audit-maximal` (LOCAL-ONLY, never push).**
> Resume by reading: this doc → `OPTIMIZATION_LEDGER.md` (canonical keep/revert) → `SUPEROPTIMIZER_L2_PLAN.md`
> (#27) → `git log --oneline -10` + `git status`.

## HEAD lineage (most-recent first)
- `eff3e78` audit: 5 verified-safe fixes — **PENDING the combined fp64 gate; REVERT if red** (this is D; C below is pre-validated).
- `5c73b5d` decoder **BAKE PIPE=1/STAGES=4** — ships the +1.49× deeper-ring KEEP (gate-validated 618ms/6.477%, 11/11×3).
- `fda3454` ViT **B1 sub-tiling** knob `SG_TUNED_VIT_P1_SUBTILE_S` — gated OFF=byte-identical (sha256-proven); **pending flip-gate**.
- `a3371f6` / `1e0d194` docs (Level-2 GO-plan + audit synthesis). `df13655` dead-artifact cleanup. `74260dd` PIPE=2 wiring (PIPE=2 LOST its gate).

## In-flight — VOLATILE (lost on termination; all READ-ONLY analysis → recoverable by re-running; NO code at risk)
- **gate agent `a3a0c742`**: rebuild `_ops` + full fp64+A/A/A gate (all cells, 3 seeds) validating `5c73b5d`+`eff3e78`. On verdict: GREEN → keep; RED → `git revert eff3e78` (D, the audit-fixes — C is pre-validated) and re-gate.
- **workflows**: `wt30eghrl` perf-lever-sweep · `wxi1vdtsf` phase2-bringup-readiness · `w9fpf0ir4` megakernel-bug-hunt · (C0-design launching). Each returns a roadmap/triage → ACT: perf levers → GPU ratchet; confirmed bugs → fp64-gated fix; phase2 → CPU fixes + 8×H100 bring-up plan; C0 → the #27 Level-2 build.

## Queued GPU ratchet — SERIAL (single H100; one build+gate at a time; delegate each gate to an agent, I decide keep/commit)
1. (awaiting verdict) combined gate of `5c73b5d`+`eff3e78`.
2. **ViT B1 flip-gate** — the #1 lever (51%): `-DSG_TUNED_VIT_P1_SUBTILE_S ∈ {16,8,4}` at **d2048/B1024** (NOT B16384 or the B1 share is understated). Sweep, avoid S=1 (U-curve). NEEDS-PARITY (LN-vec/loss fp32 reassoc under 1e-4).
3. **decoder TILE_M 128→64** (task #30, gate-cheap, ~1.33×→1.14× B1 spread).
4. the perf-lever-sweep roadmap's surviving levers.

## Saved work (persistent)
- **M0 mamba wgmma-projections scaffold** — `git stash@{0}` AND `.perf/M0_mamba_integration_scaffold.patch` (gated OFF=byte-identical; deferred until the mamba profiler is wired).
- `.perf/`: `vit_b1_subtile.diff` (=`fda3454`), `fwd_dx_pipelined_engine.diff` (PIPE=2 — LOST, superseded), `audit_safe_fixes.diff` (=`eff3e78`).

## Settings / mode
Ultracode ON (xhigh + dynamic workflows), Opus 4.8 1M, `dontAsk` perms. Agents operate in the MAIN tree despite worktree isolation + may `git stash` for byte-identity checks → **serialize anything that builds/gates/mutates**; read-only analysis fans out freely.

## Resume protocol
1. `git log --oneline -10` + `git status --porcelain`. 2. If the combined-gate verdict is unknown, **re-run it** (rebuild `_ops` + `tests/hw/test_l3tc_tail_gate.py`, 3 seeds) before trusting `eff3e78`. 3. Resume the serial GPU ratchet (queue above) + act on any landed workflow roadmaps. 4. Keep read-only analysis workflows saturated (ultracode); commit progress frequently; update THIS doc each step.
