# SuperGrok2 — FULL SESSION CONTEXT (2026-06-24/25)

**To resume: read this + RESUME.md + PROGRESS.md, then continue.** Full raw detail is in
`.session_context/` (the literal transcript `transcript_*.jsonl` 9.8M, 340 subagent transcripts, 140
task-output reports). Durable facts are in `.session_memory/*.md` (restore to /root/.claude per RESUME.md §1).

## MISSION (user-refined)
SuperGrok2 is a **portable, self-adapting, max-performance training stack** (PyTorch-shaped: Python over a
CUDA/C++ persistent fused **L3-TC megakernel**). Two core properties: (1) PORTABILITY — every component drops
into any project, config-driven, never hardcoded; (2) SELF-DESIGNING megakernels — kernel+autotuner
co-generate the optimal kernel for the workload (10M-on-1-GPU → 1.5B → 10B+), each at max perf. Validated by
the **11-optimizer ranking benchmark** (lowest val loss / most stable per fixed step budget) × **3 ~1.5B
flagship models** (decoder d1600/L48, ViT d1664/L48, Mamba d2048/L24) on real datasets
(FineWeb-Edu/ImageNet-1k/GiftEvalPretrain).

## KEY USER DIRECTIVES (the decision rules — see .session_memory/)
- **L3-TC kernels ONLY** (the upgraded persistent wgmma path); use prebuilt binaries + the compile-file
  caching; never recompile from scratch. (supergrok-working-prefs)
- **CuTe/CUTLASS device atoms** over the hand-rolled wgmma (the #1 perf lever). DONE + validated bit-identical.
- **Adaptive 3D–5D parallelism** auto-inferred from front-end params: base 3D (DP×TP×PP); +SP if sequence;
  +EP (expert parallelism, 5th) if MoE. (supergrok-adaptive-parallelism)
- **Self-specialize by config, not GPU-count**: the megakernel is templated on its deployment config;
  if-constexpr folds in exactly the machinery needed — distributed→all-reduce, single→none; large→CTA-tiling,
  small→none. A single GPU can train 10B+ (offload/recompute/stream) → decide from **robust workload×hardware
  fit**, never `if num_gpus==1`. (supergrok-adaptive-parallelism, the resource planner)
- **PyTorch-shaped front-end**: call 1 of 3 models (any size) + 1 of 11 opts + YOUR OWN dataset → backend
  self-specializes + compiles. 3 models + 11 opts fixed; **datasets PLUGGABLE** (not confined to the 3).
  (supergrok-frontend-api)
- **Max parallelism, hardware-bound**: use as many agents/workflows as possible; don't be Claude/latency-bound.
- **Proceed autonomously**: don't ask priority/what-next questions; user course-corrects. (supergrok-autonomy)

## SESSION ARC (what happened, in order)
1. **CuTe atoms** (wgmma.cuh, SG_TUNED_GEMM_ENGINE) — validated bit-identical to the hand engine through the
   real decoder megakernel (loss+grad maxabs 0.0, fp64 rel 2.85e-5, A/A/A).
2. **Bottleneck profile** (no ncu — denied; clock64 + nsys): d=2048 decoder = GEMM 72% of step @ 6.5% roofline,
   grid-barrier idle 20%. Fine ring: WGMMA 46% + ISSUE(cp.async) 37% + WAIT 10% → TMA helps ISSUE not latency.
3. **Flagship codegen**: parameterized the emitter by (d,layers,vocab,seq); emitted decoder_flagship_layout.cuh
   (1.476B params). Production headers byte-identical.
4. **dW-generalization**: the decoder backward was hardcoded for 2 layers (5 places); generalized to
   dec::kLayers via closed-form formulas, byte-identical at L=2 (gate 19/19), validated at L=48 on silicon.
5. **Flagship decoder RUNS + TRAINS** single-GPU (loss ln(99)=4.585→2.69 overfit descent; A/A/A deterministic).
6. **Distributed**: TP foundation (in-kernel device-NVSHMEM all-reduce + CommCtx + Par-template), 2 distributed
   fixes (28 tests). **NVSHMEM 3.7.0 installed** (sm_90 device bitcode). Resource planner (10/10), EP/3D-5D
   auto_config, size-adaptive CTA-tiling, memory-strategy, datasets Layer-A.
7. **Full TP for all 3 models** (decoder/ViT/Mamba) — mirrored the decoder track.
8. **Dead-code cleanup**: removed 8.09M lines of committed artifacts (nvcc dumps/scan/session logs); true
   source ~361K. LOC report in PROGRESS.md.
9. **Roofline (deliverable #1)**: /workspace/phase6/roofline_flagship.{png,csv}, 10 cells, occupancy-bound.
10. **Per-model kernel limits found**: Mamba flagship UNLAUNCHABLE (19.56MB smem/block) → **REDESIGNED**
    (layer-stream + scratch-to-HBM → 193KB, now launches); ViT not actually blocked (re-measure).
11. **11-opt flagship-decoder ranking** (overfit placeholder): NeuralGrok≈GrokAdamW≈AdamW≈GrokFast≈Prodigy ≫
    Lion > SG11 > SG15 > LookSAM; Muon/SG2 fit-but-slow. /workspace/phase6/flagship_11opt_ranking.{json,txt}.
12. **Cross-GPU NVSHMEM TP all-reduce VALIDATED on 8 GPUs** (UID bootstrap, bit-exact 2/4/8-GPU). Wildcard resolved.
13. **Live one-model-across-8 run** surfaced **3 megakernel-TP-data-path bugs** (per-rank weight-shard offset,
    25-head÷8 attention, the resulting IMA) — A+B FIXED in /workspace/phase6/tp_datapath_fix_WIP.patch (ungated),
    bug C (IMA confirm) unfinished. THIS IS THE #1 RESUME ITEM.

## STATE + REMAINING → see RESUME.md (git chain, what's done/validated, the ~5–8 hr remaining work, ETA).
## DELIVERABLES IN /workspace: phase6/roofline_flagship.png+csv, phase6/flagship_11opt_ranking.*, impl_diffs/*.md
   (all the apply-ready design specs), PROGRESS.md (the running ledger), tuning/_tp8_* (the 8-GPU run wiring).
