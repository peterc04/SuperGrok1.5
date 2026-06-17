# Session checkpoint (LIVE) — FRONT-LOAD MODE for the usage-reset gap

> Operator away ~8–12h after this usage window. A ~13h **autonomous GPU batch** is launched (or
> launching) to keep the H100 productive, **recording verdicts for review** (NO auto-commit — the
> keep/revert ratchet stays the operator's call). Source of truth = git on `/workspace` (LOCAL-ONLY).
> Resume: read this → `.perf/batch/` (the front-load output) → the `.perf/*.json` roadmaps → `git log`.

## Shipped + validated THIS session (committed, durable)
- **decoder PIPE=1/STAGES=4 BAKED** — the +1.49× fwd/dX deeper-ring KEEP, now in the shipped `_ops` (gate: 619ms/6.466%, 33/33 × 3 seeds GREEN). PIPE=2 LOST its tournament.
- **5 audit cleanups** (`eff3e78`) — has_kernels, ~750MB scratch-workspace removal, grad-hooks fail-fast, decoder-mirror test, scan-adapter removal. Gate-confirmed.
- **SG11 correctness fix** (`51098e0`) — cos(grad,**mu**)→cos(grad,**momentum**) in the L3-TC staged gate (was gate-blind: co-wrong oracle + step-1 coincidence). Gated old-FAILS / new-PASSES (×3 seeds) / 33-cell GREEN / A/A/A. Found by the bug-hunt.
- dead-artifact cleanup, GO-plan, 11 captured analysis roadmaps in `.perf/*.json`.

## In-flight (volatile; act on landing)
- **ViT B1 flip-gate** (agent `a2e9168d`, sonnet) — the #1 perf lever (51%); sweeps SUBTILE_S∈{16,8,4} @ d2048/B1024. ON LANDING: **bake the winning `-DSG_TUNED_VIT_P1_SUBTILE_S` + commit** (or "OFF stays").
- **scale-correctness-audit** (`we4dguzvd`) — capture on landing.
- **author-fixes** (`w3w9n3npc`, WRITE worktree) — emits `.perf/ci_verify_all_reanchor.diff` + `.perf/c0_r1fold_mvp.diff`; apply+gate on return.

## The 12h FRONT-LOAD batch — `.perf/batch/run_12h_frontload.sh` (launched after ViT B1 frees the GPU)
RECORDS to `.perf/batch/`, NO auto-commit. Steps: (1) preflight self-test (abort-guard); (2) `rm .STOP_TUNING`; (3) **decoder re-profile @618ms** (coarse + --fwd-fine) → `decoder_reprofile_*.txt` — re-grounds the STALE relevance gate (the 56.5% fwd/dX + 19% B1 were from the superseded 920ms build); (4) **autotuner sweep, 33 cells** (`-M -O --jit-only`, 24m/cell ≈13h) → winners in `grokking_optimizers/_kernel_tuned.json` + `autotune_*.log` + cost-model cache. **REVIEW CAVEAT:** the in-loop gate runs at d=128 (RG4) → RE-GATE winners at d=2048 before shipping.

## Serial GPU queue (on return; single H100, one gate at a time; I decide keep/commit)
1. **Review the batch output** (`.perf/batch/DONE` + the re-profile phase table + the autotuner winners).
2. **Bake the ViT B1 winner** (if the agent didn't land before launch).
3. **VRAM opt_id workspace-gate** (`vram_efficiency_audit.json`) — byte-identical ~36GB; thread opt_id into the 3 `*_tc_workspace_floats` sizers (LookSAM/SG2/Muon→0 off-path). Gate = PTX-diff + A/A/A on adamw. **High-value, cheap.**
4. **Decoder P1-epilogue-fusion** (`perf_lever_impl_design.json`, knob SG_TUNED_DEC_EPI_FUSE) — only after the re-profile confirms the foldable share clears the relevance floor. The only perf lever that survived adversarial verification.
5. **Gate-hardening workstream** (`oracle_trust_audit.json` + `test_coverage_gap_audit.json`): RG1 wire run_sg11_warmup_gate into a pytest-collected hw test + generalize to SG15 (**load-bearing — the SG11 fix isn't CI-regression-guarded without it**); RG4 reconcile the autotune hook's d2048-vs-d128 scale lie; de-mirror the SG2 CSA oracle (full-rank) + make its fidelity probe voting; anchor the SG11 fp64 oracle operand. (The gate-hardening WRITE workflow failed on a backtick bug — re-author.)
6. **C0-MVP r1-fold** (`.perf/c0_r1fold_mvp.diff`, byte-identical OFF) + **mamba profiler wiring** → then M0 (output-stationary dW, 668GB OOM wall) + M1 (acts→HBM, d=2048 placement).
7. **CI-unblock** (`needs_care_unblock_plan.json`) — LOW-urgency (local-only, no CI running); fix #4 (cpp_structural/docs_consistency/README) is an owner decision.

## Mode / constraints (load-bearing for a resume)
Ultracode ON; models cost-calibrated (sonnet bulk, opus for subtle/correctness). Commits LOCAL-ONLY, never push. **Workflow worktree-isolation HOLDS** (verified — write agents don't leak into main); but **serialize anything that builds/gates the main tree** (single GPU + the main-tree git-stash hazard); read-only + worktree-write fan out freely. The fp64 + A/A/A gate is the HARD gate; the legal rewrite space is transport-only (ascending-k); structural/perf changes are NEEDS-PARITY (gate-arbitrated).

## Resume protocol
1. `git log --oneline -20` + `git status --porcelain`. 2. `cat .perf/batch/DONE` + tail `.perf/batch/run.log`; review `decoder_reprofile_*.txt` + `_kernel_tuned.json` (re-gate winners at d=2048 before trusting). 3. Work the serial GPU queue ↑ in order. 4. Bake/commit ViT B1 + apply the authored `.perf/*.diff` (each fp64-gated). 5. Commit frequently; refresh THIS doc each step.
