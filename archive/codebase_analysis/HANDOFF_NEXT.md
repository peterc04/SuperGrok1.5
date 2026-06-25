# HANDOFF — SuperGrok2, for the next instance (8×H100)

**Written 2026-06-25 at the end of a 1×H100 analysis session, in preparation for moving to an 8×H100 pod.**
This file is the single entry point. It was written by an instance that read this codebase cover-to-cover
(see `analysis/`). Read this top-to-bottom, then `analysis/MASTER_STATE.md`, then start work.

> ⚠️ **Trust git + this handoff over the older docs.** `LEDGER.json`/`PROGRESS.md` headers are frozen at an
> early commit, and `CODEBASE_EXPLAINED.md`/`SESSION_STATE.md`/`HANDOFF.md` PREDATE this campaign. The §5
> "corrections" below are the load-bearing truths the stale docs hide.

---

## 0. TL;DR — the one thing that matters
The hard part is **done and validated**: the cross-GPU in-kernel **device-NVSHMEM TP all-reduce is bit-exact on
8 GPUs**, and all 3 ~1.5B flagship models launch single-GPU. The **only** thing between you and the real
"one-1.5B-model-across-8-H100s" training run is finishing the **TP data-path fix** (`analysis/phase6/
tp_datapath_fix_WIP.patch`) — and you are now on hardware (8×H100) where it can actually run. **But read §5.1
first: that patch is a correctness/validation scaffold, not yet real model sharding.**

## 1. RESUME ON THE NEW POD
1. **Mount the volume at `/workspace`** (same path — session-resume + build-cache hits depend on it).
2. Restore the environment (`RESUME.md` §1): `pip install nvidia-nvshmem-cu12 optuna ruff nvidia-ml-py`;
   `git config user.email "<your-email>"`; the NVSHMEM env (`NVSHMEM_HOME=…/nvidia/nvshmem`,
   `NVSHMEM_DISABLE_NVLS=1 NCCL_NVLS_ENABLE=0`); `cp .session_memory/*.md /root/.claude/projects/-/memory/`.
3. **If the base image matches** (torch 2.4.1+cu124, py3.11, CUDA 12.4) the committed `_ops.so` + `.build_cache/`
   give a near-zero-rebuild start. If not, rebuild — ccache still accelerates it.
4. Git: branch `claude/custom-optimizer-analysis-HFYhg`, HEAD = the closure commit. The `mamba_test_fix` branch
   (see §5.4) is NOT merged — decide whether to adopt it.

## 2. WHAT SUPERGROK2 IS (concise)
A portable, self-adapting, max-performance training stack: PyTorch-shaped Python over a **persistent fused
"L3-TC" CUDA megakernel** (bf16 wgmma + fp32 accum). ONE `__global__` launch runs a whole training step:
**P0 zero → P1 fused fwd+bwd → P2 deterministic ascending-CTA grad reduce → P3 optimizer tail**, phases split
only by a hand-built **sense-reversing GridBarrier** (1 CTA/SM, non-cooperative launch). It spans **3 models**
(transformer-decoder, ViT, Mamba-3) × **11 optimizers** × **3 archs** (sm_90 / gfx942-MI300X / tpu_v6e-Pallas),
composed by a generator-emitted dispatch table. Validated by an **11-optimizer ranking** × 3 ~1.5B flagships
(decoder d1600/L48, ViT d1664/L48, Mamba d2048/L24). Correctness is a HARD gate: fp64 parity (rel 1e-4) AND
A/A/A bit-determinism; all rewrites transport-only (ascending-k fp32 order preserved). `grokking_optimizers/
compile.py` (32,900 lines) is the Optuna-TPE superoptimizer/autotuner. **Self-adapting** = the config is DERIVED
from workload×hardware, never hardcoded by GPU-count (see §5.2).

## 3. DONE & VALIDATED (evidence in analysis/)
- CuTe-atom GEMM engine, bit-identical, behind `SG_TUNED_GEMM_ENGINE` (default 0 = shipped PTX).
- 3 flagship layouts emitted + launch single-GPU; Mamba unblocked via the smem redesign (19.56MB→193KB,
  `kMbStreamSmem`). Decoder L=48 silicon run: loss=ln(99), A/A/A deterministic.
- Full TP machinery + **8-GPU NVSHMEM all-reduce bit-exact** (2/4/8-GPU smokes, 36.0 exact).
- Resource planner (`parallel/resource_planner.py`), ZeRO-3, CTA-tiling, dead-code cleanup (−8.09M lines).
- Decoder pytest gates green; gfx942 AMD backend compiles 14/14 (complete 11-opt mirror); `_scan` security clean.

## 4. THE #1 TASK — the real 8-GPU TP training run
The WIP patch applies CLEANLY on HEAD (verified: blobs `e84ec16`/`973115b` == patch base). Steps:
```
cd /workspace/SuperGrok1.5
git apply analysis/phase6/tp_datapath_fix_WIP.patch          # 2 files, ungated, SingleGPU path unchanged
pytest tests/hw/test_decoder_tc.py -m hw -q -s               # GATE: must stay byte-identical (19/19)
bash tuning/_tp8_build.sh                                     # manual 3-step RDC/-dlink build (torch JIT omits -dlink)
torchrun --nproc_per_node=8 tuning/_tp8_run.py               # the tp_size=8 arm via the launcher
compute-sanitizer --tool memcheck torchrun --nproc_per_node=8 tuning/_tp8_run.py --steps 3   # close bug C
```
GATE for "done": 0 IMA under sanitizer + cross-rank loss agrees + loss descends + SingleGPU pytest still 19/19.

## 5. CRITICAL CORRECTIONS (what the stale docs hide — verify these first-hand, cites in analysis/CRUX_*.md)
**5.1 — The TP fix is a VALIDATION SCAFFOLD, not real model sharding.** (`analysis/CRUX_TP_DATAPATH.md`)
The patch makes the `kTPComm` path compute **FULL-WIDTH REPLICATED** on every rank (same math as SingleGPU,
full kHeads=25), then routes the 4 reduce points through an **identity** all-reduce of (x/P). This proves the
NVSHMEM data path end-to-end (bit-consistent, descending) but gives **zero compute/memory reduction** — every
rank computes the whole model. It fixes the 3 bugs (A rank-divergence, B 25%8 head split, C the IMA + an
OOM-safe workspace guard). **Genuine weight-sharding (host pre-packed shards + whole-weight grad all-reduce) is
explicitly scoped-not-done.** So the flagship 1.5B still won't *fit* sharded across 8 until that lands — the OOM
guard just turns the wild-pointer IMA into a clean `cudaErrorMemoryAllocation`. This is the real next frontier.

**5.2 — The resource planner is REAL but AHEAD OF THE KERNEL.** (`analysis/CRUX_CONFIG_DERIVATION.md`)
`parallel/resource_planner.py::plan_execution` is a rigorous fit-driven ladder (ZeRO-3 → raise PP → **CTA-tiling**
→ recompute → layer-stream → host-offload), faithful to the live scratch formulas, emitting exact `-D` flags +
`ParConfig<dp,tp,pp,sp,z>`. Adaptive **3D→5D** (DP×TP×PP, +SP for sequences, +EP for MoE), never `if num_gpus==1`.
BUT for the decoder, `SG_DEC_RECOMPUTE`/`SG_DEC_LAYER_STREAM`/`SG_DEC_HOST_OFFLOAD` have **0 kernel refs** —
emitted-but-unimplemented. Real decoder levers today: TP(replicated), ZeRO-3, CTA-tiling(nCTA), bench-layout
elision. Mamba has real layer-streaming. Planner tests assert only the arithmetic, not that the kernel honors flags.

**5.3 — "33 megakernels" is misleading.** The 33 generated per-cell `.cu` are DEAD/uncompilable (they `#include`
a removed header). The REAL path is **3 per-model `_tc` kernels dispatched over `OptId`**. The flagship 11-opt
ranking is an **overfit placeholder** (B=16, mod-97), not a real benchmark — real ranking needs Layer-B datasets.

**5.4 — `mamba_test_fix` (branch `3df7ee9`) is un-merged.** It recalibrates `tests/hw/test_mamba_tc.py`
(per-tensor bf16-floor-calibrated tols + one documented skip) and resolves the "Mamba 3/5 failing" status — NOT
a loosening. HEAD has the OLD version. Adopt via `git checkout 3df7ee9 -- tests/hw/test_mamba_tc.py` if desired.

**5.5 — Scalar vs TC.** The decoder file has TWO megakernels: scalar fp32 (default, correctness oracle) and the
wgmma "L3-TC" (`-DSG_TUNED_GEMM_IMPL=1`, the campaign/flagship path). Don't confuse their gates (TC = 13/13
test_decoder_tc; scalar = test_megakernel_vs_eager). **5.6 — ncu counters are env-DENIED** → all roofline numbers
are nsys/static/wallclock, not counter-scored.

## 6. REMAINING WORK (ordered)
1. **Finish the 8-GPU TP run** (§4) — apply WIP patch, close bug C under sanitizer. ~1–2 hr.
2. **Real model sharding** (§5.1) — host pre-packed per-rank shards + whole-weight grad all-reduce, so the 1.5B
   actually fits sharded across 8 (the OOM scaffold is not a fit). The genuine north-star item.
3. **Wire the planner's unimplemented decoder strategies** (§5.2): recompute / layer-stream / host-offload, or
   document them as Mamba-only. Needed for the "10B-on-1-GPU" claim to be real for the decoder.
4. **Real-data benchmark (Layer-B):** wire FineWeb-Edu/ImageNet-1k/GiftEval into the datasets Layer-A seam
   (`analysis/impl_diffs/datasets_v2.md`); replace mod-97; run the real 11×3 ranking.
5. **Full 33-cell roofline** (now Mamba launches) + ViT re-measure at the saturating batch.
6. Adopt `mamba_test_fix` (§5.4). Fix the `tune_out` build bug (PYTHONPATH + include-path when building from
   `/workspace` not the repo root — see `analysis/coverage/misc_repo_buildout.md`).

## 7. HOW TO WORK (methodology — the owner's standing directives; see `.session_memory/`)
- **Read the ENTIRE codebase cover-to-cover before acting** — like the analysis in `analysis/` was produced.
  Don't grep-skim; literally read every source/doc/spec. The corpus is ~28MB of real code/docs (the 40GB volume
  is mostly build cache + duplicate worktrees — see `analysis/COVERAGE_LEDGER.md`).
- **Max parallelism, hardware-bound not latency-bound:** use as many parallel agents + Workflows as possible.
  Use **git-worktree isolation per track** (one worktree per spec/agent so parallel builds don't clobber) and
  **integrate via cherry-pick** into the mainline — this is exactly how this campaign was built (14 such tracks).
- **RATE-LIMIT LESSON (learned the hard way):** ~16 concurrent Opus agents over MB-scale reads trips a
  server-side 429 and kills them. Read on **Sonnet**, gate concurrency to **waves of ≤6**, wrap each agent in a
  2–3× **retry**, and **don't run two big workflows concurrently** (they steal each other's rate-limit budget and
  trigger expensive retries). Reserve Opus for synthesis/verify.
- **Proceed autonomously** — don't ask priority/what-next; the owner course-corrects.
- **L3-TC kernels only; use prebuilt binaries + compile-file caching; never recompile from scratch.**

## 8. MAP — where everything is
- `analysis/MASTER_STATE.md` — the full synthesized state (architecture, done/in-flight, discrepancies, resume).
- `analysis/CRUX_TP_DATAPATH.md`, `analysis/CRUX_CONFIG_DERIVATION.md` — first-hand deep-dives (§5.1/§5.2).
- `analysis/HISTORY_FINDINGS.md` — the git-history reconstruction (§5.4).
- `analysis/COVERAGE_LEDGER.md` — proof every file was accounted for; what's regenerable vs essential.
- `analysis/<area>.md` + `analysis/coverage/<area>.md` — the per-slice digests (the cover-to-cover read).
- `analysis/impl_diffs/` — apply-ready design specs. `analysis/phase6/` — deliverables + the WIP TP patch +
  re-runnable workflow scripts. `analysis/session_history/` — scrubbed prior-session conversation narratives.
- `RESUME.md` — env-restore detail. `.session_memory/` — the owner's standing directives.
- Source: `csrc/fused/sm_90/` (megakernels), `csrc/algorithms/` (11 optimizer math headers),
  `grokking_optimizers/` (the Python brain: compile.py, codegen, distributed, parallel/).
