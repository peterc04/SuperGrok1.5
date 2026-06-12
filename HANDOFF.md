# HANDOFF — SuperGrok1.5 H100 megakernel campaign
**Updated 2026-06-12 ~22:05Z, instance about to be cut by owner (usage limit). Everything needed to resume is on the /workspace network volume — KEEP THE VOLUME.**

## Fresh-session bootstrap (new instance, new session)
1. Restore standing memory: `cp -r /workspace/SuperGrok1.5/.claude-memory-backup/* ~/.claude/projects/<project-slug>/memory/` (or just read the files — they ARE the standing rules: no-suppression, validation ladder, owner pings, provisioning signals, pod constraints).
2. `git -C /workspace/SuperGrok1.5 log --oneline -25` — the durable record. Branch: **`claude/h100-audit-maximal`**, all commits LOCAL-ONLY (never push).
3. `CUDA_MPS_PIPE_DIRECTORY=/nonexistent python wiring_check.py --require-all` — must say 33/33 L3-TC (wgmma). ALWAYS run before any roofline (measurement-trap rule).
4. Read `.regpressure/REPORT.md` + `.regpressure/RING_REPORT.md` + `.phase2/REPORT.md` + `.phase2/RUNBOOK.md` — the live deliverables.
5. If `git status` shows tracked-file modifications: that's the killed GPU-gating lane's half-applied patch — `git checkout -- .` back to HEAD; the patch series in `.regpressure/*.patch` is the source of truth, re-apply per protocol below.

## State at cut (owner directive: "do everything you can without the 8×, signal when needed")
**DONE today (commits):**
- 19 overnight commits through `642e360` (all 33 cells real L3-TC wgmma; 33-cell tail-gate green in ONE invocation; roofline mean 1.15%/median 1.29%; decoder d=1024 2.15×, production 3.28× vs two nights ago). Morning report: `/workspace/.morning_report.md`.
- `ab8c313` — **Phase-2 authoring complete** (Lane C): TP loopback transport+layer (NVSHMEM behind `-DSG_HAS_NVSHMEM` compile seam, NOT installed here), PP stage kernels + 1F1B + PP=2 loopback gate, ZeRO-3 FlatShardPlan/gather-release/sharded-ckpt, §6.2 distributed step. 55 CPU tests green; GPU tests authored but NOT run → `.phase2/RUNBOOK.md` (~45-60 min on 1×H100). Patches for tracked files: `.phase2/patches/0001,0002` (PTX-identity proven for 0001).
- `821fee5` — **#12 tuner JIT path FIXED** (Lane A): `PYBIND11_MODULE(TORCH_EXTENSION_NAME)` root fix + 8 compile.py repairs (worker eviction, JSON-line filter, private-generator event fallback, brace escape, TORCH_CUDA_ARCH_LIST=9.0a pin, PYTORCH_NVCC wrap, MAX_JOBS guard). Proven: variant built+imported+timed on GPU end-to-end; wiring 33/33; tail-gate 33/33 in 314s; A/A/A 30/30 bit-identical (+3 SG2 isolated). Peer stash patch rejected 4/4 with citations (superseded by `41e6525`/`b208f00` at HEAD or inverted premise) — `stash@{0}` kept + extracted to `.peer_tuner_patch.diff`.
- **Static optimization series authored & statically verified** (Lanes B+D), in `.regpressure/`:
  `0001`-decoder-bf16-weight-prestage (ring blocker (a) eliminated; spill-neutral; halved fwd/dX weight HBM reads = GPU-gated claim) → `0002`-decoder-sam-scoped-outline (SAM spills 15252→~7950, −48%; single-pass cells SASS-identical) → `0003`-vit-sam-scoped-outline (SAM −17%, SG2 −31%) → `0004`-mamba-scope-noinline (single-pass 5848→1032, −82%; RISK: returns single-pass to inline topology — gate A/A/A hard) → `0005`-decoder-cpasync-ring-fwddx (on 0001+0002; LDGSTS contract fully in SASS, wgmma web untouched, single-pass stay 0-spill; falsifiability notes in RING_REPORT).
  Apply order vs `642e360` (re-apply on `821fee5` works — Lane A touched only bindings.cpp+compile.py; use `git apply -3` if needed). Full baselines/methodology/parse tools in `.regpressure/` (`parse_ptxas.py`, `parse2.py`, `compile_one.sh`).

**IN-FLIGHT at cut (killed with instance):** Lane E was GPU-gating the 0001→0005 series, still in step-0 BASELINES at `821fee5` (evidence streaming to `.regpressure/gpu/`: `prod_*_BASE.json`, `chain_BASE.log`, `tailgate_baseline.log`, `wiring_baseline.log`). No patch had been applied yet when this handoff was written; verify with `git status`/`git stash list`.

## Resume queue (in order)
1. **GPU-gate the series** (was Lane E): baseline at HEAD (3 reps = noise floor; may already exist in `.regpressure/gpu/`) → per patch IN ORDER: apply → incremental rebuild → `wiring_check --require-all` → full tail-gate (`pytest tests/hw/test_l3tc_tail_gate.py -m hw -q -s`, expect 33) → re-time affected cells (locked knees: decoder@1024 B≈4096, vit B≈16k, mamba B≤4096; production d=128 all models) → keep-if-better-than-noise-and-parity-green else revert. **0001 judged AFTER 0005** (keep both if ring wins; else 0001 only if independently better). **0004**: any A/A/A waver across 3 repeats = revert. Parity/determinism failure = instant revert. Commit kept patches separately with verdict numbers; update reports with GPU VERDICTS section; final composition gets wiring + one full tail-gate + roofline json refresh.
2. **Phase-2 GPU runbook** (`.phase2/RUNBOOK.md`, ~45-60 min): loopback TP∈{2,4}, PP=2 bit-identity, ZeRO-3 round-trip, dist-step world=1/2. Apply `.phase2/patches/*` first (check REPORT).
3. **Bounded tuner sweep LAST** (60 trials, revert-if-not-better) — after kernels are final, so it tunes the final code. The JIT path works post-`821fee5`. Known gaps: sweep numerical-oracle template targets torch.ops.* (pybind-only project → validation ran "skipped"-shaped; parity safety comes from the tail-gate, run it after applying tuned flags); CUDA-graph capture of fused step fails ≥1024 (pre-existing; event-timer fallback works).
4. **vit/mamba PP/TP twins** (1-GPU, after runbook proves decoder versions).
5. **SG2 workspace redesign** (stretch; 199 GB at d=1024 problem).
6. **8× SIGNAL**: when 1-5 done = Phase 1 solid + parallelism 1-GPU-validated → PushNotification owner "provision 8xH100 now"; one rental window = NVSHMEM-TP validation + §5.4 go/no-go + TP insertion at the 4 marked points + scaling measurements (DP 1→8, ZeRO-3 OOM threshold, PP bubble) + cross-rank graph capture + real P2P swap → tear down. mi300x/tpu_v6e signal comes only after the FULL H100 work is locked.

## Pod constraints (this pod generation — verify on new instance)
- Docker memory cgroup **200 GiB** despite huge host RAM: `MAX_JOBS=24` for JIT variant builds (cicc ≈5.9 GiB/TU); main 56-TU `_ops` ninja at full -j is OK (~8.5 min cold); NEVER two heavy builds concurrently.
- **`/dev/shm` is NOEXEC**: nothing dlopen-able there → `TORCH_EXTENSIONS_DIR=/workspace/SuperGrok1.5/build/torch_ext`; sccache + nvcc temps on /dev/shm fine: `SCCACHE_DIR=/dev/shm/sccache TMPDIR=/dev/shm/tmp` (sccache cache dies with instance — first build cold).
- PATH `nvcc` is a caching shim (breaks sccache wrapping): `export CUDA_HOME=/usr/local/cuda` (12.4).
- MPS: SIGTERM only, never kill -9 CUDA clients (server wedge). Quiet GPU for timing, ONE heavy GPU process at a time. `bash /workspace/.cleanup.sh` when disk tightens.
- nvcc 12.4 cicc segfaults on tile-fn-noinline-without-GEMM-handling shapes (Lane B hazard note).

## Where everything lives (all on the volume)
`/workspace/SuperGrok1.5` repo (branch `claude/h100-audit-maximal`) · `.regpressure/` static series + reports + GPU baselines · `.phase2/` parallelism reports/runbook/patches · `/workspace/.morning_report.md` overnight full report · `/workspace/.parallelism_design.md` design contract · `.claude-memory-backup/` standing-rules memory · `build/compiled/{smoke12_acceptance,tailgate_full,wiring_check_verdict}.log` Lane A evidence · `stash@{0}` peer patch provenance (+ `.peer_tuner_patch.diff`).

## Roofline reality (unchanged, don't re-derive)
Numerator = profiler GEMM/conv FLOPs only; tiny models are mostly non-GEMM → absolute % of bf16 peak stays ~1% at these dims — PHYSICS, not missing optimization. Relative wins + spill/bandwidth reductions are the real metric at this scale; the 0.2-0.6%→1.15% climb came from real structural fixes. Raising absolute % = bigger models = the owner's scale-up decision, separate from this ladder.
