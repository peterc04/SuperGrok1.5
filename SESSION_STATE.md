# Session checkpoint (LIVE) — SESSION ENDED CLEANLY (instance being closed)

> The operator ended this session (a planning/strategy session) and is closing the instance.
> EVERYTHING is persisted to /workspace (the persistent volume). To resume, read in this order:
> 1. THIS file, 2. `CODEBASE_EXPLAINED.md` (the full architectural reference), 3. `PLANNING_INPUT.md`
> + `.perf/phase1_status_audit.md`, 4. `git log --oneline -20`. Source of truth = git on /workspace
> (LOCAL-ONLY, never push). Build caches are now volume-backed — `source .fast_build_env.sh` = fast recompile.

## NEW DIRECTIVES from this session (load-bearing — these change priorities)
1. **Optimizers must be MAXED**, not just fused. The fusion (one-binary-one-launch; the optimizer is
   the P3 tail of the persistent megakernel, state resident, zero extra launch) is DONE. But the
   optimizer TAIL KERNELS are NOT perf-maxed (~5.9% of the decoder step, un-autotuned), and 8/11 carry
   gate-coverage caveats. This is now an explicit objective alongside the model kernels.
2. **High % of roofline at EVERY scale** is the design goal (owner was explicit: not only at 1B+).
   REFRAME (I corrected an earlier over-pessimistic take): decoder 6.48% @ d=2048 is largely FIXABLE
   inefficiency — ~20% grid-barrier serialization + sub-cuBLAS hand-rolled wgmma + non-GEMM serialized
   (not overlapped) under the matmuls + 1-CTA/SM low occupancy — NOT a non-GEMM physics floor. Target
   30-50%+ @ d=2048. A genuine hard floor exists only at the very smallest shapes (d=128 tiny matmuls).
   Thesis: the files "optimize for the project they are put in" (scale-adaptive strategy).
3. **The owner is bringing a full engineering plan** (objectives + structural how + priority order +
   done-criteria). Division of labor = HYBRID: owner owns the what/structure/priorities; I implement +
   empirically gate + surface where the plan meets reality. Leave tile-sizes / which-lever-wins to the
   gate loop (empirical). Est: a good plan ~1.3-1.6x faster end-to-end + kills misdirection/busywork risk.

## Shipped + validated THIS session (committed, durable)
- **ViT B1 sub-tile S=8 BAKED** (`b0d41f8`) — the **#1 lever, 4.02x** (5768->1434 ms @ d2048/B1024;
  B1_barrier 51.2%->41.9%). Sweep a2d8fe79: 36/36 gate calls clean (4 S x 3 seeds x 11 cells), A/A/A held.
  **Final confirm-build on resume** (was deferred — GPU was busy).
- **macro-table drift fix** (`319b96d`) — `dec_dw_splitk` table=4 vs header=1 (the 06-16 ratchet) + dead
  `mb_dw_splitk` knob; a fake-green hole (drift guard not in `--self-test`) hid it. Fixed across 5 files;
  guard now green; self-test 265/0.
- **front-load fix + phase-1 audit** (`64feb14`) — the 12h autotune front-load was failing in 8 min on
  two bugs, both fixed at source: `-M transformer_decoder` -> CLI wants `decoder`; `g++-cached` (CXX
  ccache wrapper) was on no PATH dir so torch's ABI probe killed every `--jit-only` build -> symlinked
  into /workspace/.local/bin. Smoke-tested: builds now reach the PGO+autotune phase. Also captured
  `.perf/phase1_status_audit.md`.
- **decoder PIPE=1/STAGES=4 BAKED** (earlier) — +1.49x fwd/dX deeper-ring KEEP (618ms/6.466%, 33/33 x3).
  PIPE=2 LOST its tournament.
- **SG11 correctness fix** (`51098e0`) — cos(grad,mu)->cos(grad,momentum) in the staged gate.
- **Build-cache persistence** (this commit) — sccache+ccache moved ramdisk->`.build_cache/` (volume).

## Verified current state — READ THESE
- **CODEBASE_EXPLAINED.md** — exhaustive architectural reference: the model megakernels + L3-TC
  substrate, the 11 optimizers + fused P3 tail, compile.py (the superoptimizer / autotuner + the
  Level-2 work), WHY it is latency-bound + the roofline reality, and the correctness methodology.
- **PLANNING_INPUT.md** — opportunity map (decoder 618ms: ~55% GEMM / ~20% barriers / 16.5% dW / 6% opt
  tail), the 11-optimizer readiness matrix, the open decisions the owner's plan should make.
- **.perf/phase1_status_audit.md** — the refute-by-default audit (roofline numbers + optimizer matrix +
  GPU-util findings), all numbers reproduced.

## Build / binaries / cache — persistence (IMPORTANT for fast resume)
- Binaries on the volume (survive instance close): `grokking_optimizers/_ops*.so` (33M), `build/` (90M).
  DO NOT clean `build/` (task #7's deferred cleanup is now superseded by the owner directive to KEEP binaries).
- Build caches MOVED ramdisk->volume: `.build_cache/sccache` (19M) + `.build_cache/ccache` (831M),
  gitignored. `.fast_build_env.sh` now points SCCACHE_DIR/CCACHE_DIR there, so `source .fast_build_env.sh`
  yields fast recompile immediately on resume. (`/dev/shm/tmp` stays as regenerable nvcc scratch.)
- `.STOP_TUNING` restored (autotuner DISABLED by default; the front-load script removes it for a run).

## Front-load batch — STOPPED (was fixed + running; halted for the clean shutdown)
`.perf/batch/run_12h_frontload.sh` was fixed + running real autotune this session, then STOPPED for the
instance close (it cannot cover an absence while the instance is down). The decoder RE-PROFILE completed:
`.perf/batch/decoder_reprofile_{coarse,fwdfine}.txt` (618ms/6.48%; fwd/dX rings now WGMMA-compute-bound,
WAIT 9%/6% vs WGMMA 47%/41% — the deeper ring hid the drain). Re-launch when you want a tuning run.
Re-gate any `_kernel_tuned.json` winners at d=2048 (the in-loop gate runs at d=128, RG4).

## Open queue (on resume; single H100, serialize anything that builds/gates the main tree)
1. **Confirm-build the ViT S=8 bake** + production re-gate (the only deferred verification of a shipped win).
2. **Optimizer-max** (NEW): autotune the P3 tail per-cell + structural opt (the front-load gives the first data).
3. **Gate-hardening** (makes 8 caveat cells trustworthy — prerequisite for trusting any optimizer-perf):
   SG11/SG15 warm-up -> CI-collected pytest; grokadamw/prodigy multi-step parity; SG2 CSA oracle de-mirror;
   re-gate d=128 cells at d=2048. (oracle_trust_audit.json + test_coverage_gap_audit.json.)
4. **BUG-04** — mamba staged-opt scratch gate (kMbStagedOptScratch byte-identical) — unblocks muon/neuralgrok/looksam mamba cells (OOM at d>=1024).
5. **C0 fragment emitter** (#27) — re-author MYSELF in the main tree (the leaked partial is in
   `.perf/leaked_implement_net_new_partial.diff`; specs in `.perf/critic_and_design.json` + level2_c0_spec).
6. **Perf levers** — decoder TILE_M 128->64 (#30), P1-epilogue-fusion (#31, fix the fold-B OOB first),
   barrier-overlap, dW staging, toward the "high-roofline-every-scale" goal.
7. **fast-triage harness** (#26) — accelerates the whole perf loop.

## Mode / constraints (load-bearing)
Ultracode ON; models cost-calibrated. Commits LOCAL-ONLY, never push. The fp64 (rel 1e-4; SAM 2.5e-2) +
A/A/A bit-determinism gate is the HARD gate; legal rewrites are transport-only (ascending-k fp32);
structural/perf changes are NEEDS-PARITY (gate-arbitrated); keep/revert ratchet (better on 3+ seeds else
revert). LESSON: Workflow worktree-isolation has leaked into the main tree once — do correctness-sensitive
WRITES myself in the main tree; fan out read-only/design workflows freely. Serialize anything that builds/gates.

## Resume protocol
1. `git log --oneline -20` + `git status --porcelain`. 2. `source .fast_build_env.sh` (restores fast cache).
3. Read CODEBASE_EXPLAINED.md + PLANNING_INPUT.md. 4. Take the owner's engineering plan; implement + gate
   each item through the fp64 + A/A/A ratchet. 5. Confirm-build the ViT S=8 bake first. 6. Commit frequently;
   refresh THIS doc each step.
