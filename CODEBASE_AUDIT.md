# Codebase audit — repeatable full-tree line-by-line swarm (PROCESS)

Owner directive (2026-06-16): **after EVERY landed step, run a comprehensive line-by-line analysis of
the ENTIRE codebase** (Opus 4.8 1M, max effort) — exactly like the compile.py audit (COMPILE_AUDIT.md),
but tree-wide — looking for **(1) bugs, (2) dead code to clean, (3) dead files to clean.** This doc is
the repeatable harness; findings accumulate in `CODEBASE_AUDIT_FINDINGS.md`.

## "A step"
A completed unit of LANDED work: a commit — a KEEP/REVERT, a fix batch, a perf lever. The audit runs
*after* the step's changes are committed + gate-green (auditing a mid-flight, changing tree is wasteful
and the P0 compile.py agent / perf gates must finish + commit first).

## The swarm (per run)
1. **Enumerate** the tree (`git ls-files`), partition the real source into ~contiguous ranges on
   file/section boundaries (~2.5–4k lines/agent) across: `csrc/{fused,backends,kernels,common}`,
   `grokking_optimizers/`, `tuning/`, `scripts/`, `tests/`, `examples/`, top-level.
2. **Range agents** (general-purpose, Opus 4.8, parallel, READ-ONLY): line-by-line; each returns
   structured findings — `SEVERITY{blocker|major|minor|nit} · file:line · CATEGORY{bug | dead-code |
   dead-file | not-maximal | correctness-risk | duplication} · desc · concrete fix`. Same skeptical
   mandate as the compile.py audit (stubs/TODO/swallowed-except/present-but-OFF/not-strongest).
3. **Cross-cutting agents:** (a) **DEAD-FILE graph** — import/include/build/test reachability across
   the whole tree (which `.py/.cuh/.cu/.cpp/.h` are never imported/included/compiled/referenced →
   dead-file candidates); (b) **DEAD-CODE** — functions/symbols never called; (c) **CONSISTENCY** —
   duplicated/diverged logic (e.g. the dead timer twins in compile.py).
4. **Synthesize** → append to `CODEBASE_AUDIT_FINDINGS.md`, **deduped** against prior runs (carry
   unchanged findings by reference; report NEW issues + the step's changed blast-radius in full).
5. **Fix (gated)** per [[feedback-patch-protocol]]: fp64 gate + 242-case self-test stay green.

## Deletion discipline (CONSERVATIVE — repo is under active development)
- PROVE a file/path is dead (no import/include/build/test/doc reference) before removing.
- **FLAG-AND-KEEP** (report, don't delete) anything ambiguous or that I did not create.
- KEEP: binaries, build caches, HIP + Pallas backends, dev scaffolding, docs, fp64 oracles, campaign
  files, dormant-but-intended infra (e.g. the parallelism scaffolding, the M0 Mamba patch).
- Re-validate after every deletion; revert anything that goes red.

## Efficiency (faithful, not wasteful)
First run = exhaustive full sweep. Per-step runs = full re-sweep with **deduped output** + deep-focus
on the changed blast radius + always re-run the (cheap) dead-file/dead-code reachability graph (catches
files newly-orphaned by the step). Milestone runs = full exhaustive re-read. Cost is real and is
budgeted into the Phase-1 ETA.
