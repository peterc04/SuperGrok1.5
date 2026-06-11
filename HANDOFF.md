# HANDOFF — SuperGrok1.5 H100 megakernel campaign

**For a fresh session:** read `MEMORY.md` (auto-loads, 14 standing rules), then run
`CUDA_MPS_PIPE_DIRECTORY=/nonexistent python wiring_check.py --require-all` for live
per-cell state, then `cat results/h100_grokking_race/roofline.json` for perf. This
file captures only what those can't: in-flight granularity + open decisions.

## State (2026-06-11, commit 405654f)
- **~18/33 cells real L3-TC wgmma** (wiring_check is the source of truth, not this number).
  Done: adamw/lion/grokfast/grokadamw/neuralgrok/prodigy across models; vit also muon.
  Blocked-with-reason (printed by wiring_check): the staged SAM optimizers
  (supergrok11/15, looksam) on decoder+mamba; neuralgrok/decoder shows "eager" until
  its committed cell rebuilds into _ops; mamba muon/sg2.
- **Component model:** 14 files (3 model stage headers `model_stage_<model>_tc.cuh` +
  11 opt tails) compose into 33 generated cells. Fix a header → recompose, never edit
  cells. Build: `FORCE_CUDA=1 ./build.sh` (or compile.py incremental; `*_selftest.cu`
  /`*_tc_launcher.cu` auto-excluded from `_ops` by setup.py content-filter).

## Headline perf (B=16384, quiet GPU)
- mamba TC: **0.46×→2.15× vs scalar** (bottleneck FIXED, the win)
- vit TC: **+3.48×, 4.60 TF/s** (3 structural bugs fixed: acc-array spill, 132× redundant
  bias reduce, strided clspos scan)
- decoder grokadamw: 3-mechanism cell landed (per-layer β1, global clip P2.5, adaptive α)
- **Absolute roofline fraction stays ~0.2–0.6% of bf16 peak — this is PHYSICS of tiny
  models, not missing optimization** (see "Roofline reality" below). Relative wins are real.

## In-flight when this was written (these agents die with the session)
- Wave-2 workflow (wf_694d43eb-253): 3 lanes converting remaining staged cells.
  decoder: muon→sg11→sg15→looksam→sg2. mamba: prodigy→muon→sg11→sg15→looksam→sg2 +
  fix mamba determinism gate fails. vit lane: DONE (committed 0cd63c8).
- Resume the campaign: per cell → tail_gate(state+A/A/A) + wiring_check + roofline quiet,
  maxed before next, blocked=loud+cited, commit per batch. Templates proven in vit lane.

## Open decisions for the owner
1. **mamba×adamw default = wgmma** (TC-only rule) but TC was 0.46× scalar pre-fix; now
   2.15× so the flip is justified — confirm it stays.
2. **Phase-decomposition** (offered, not yet run): clock64 instrumentation in the decoder
   TC kernel can break each cell's wall into GEMM/attention/elementwise/optimizer/barrier
   — tells the TRUE achievable ceiling per cell. Run before declaring cells "maxed".
3. **Scale-up vs optimize** (owner's live question): the roofline bars are low because the
   models are tiny (d=128, seq 4-17, 0.4M params, ~45µs math/step), NOT because kernels
   are unoptimized. Raising the bars meaningfully = bigger models/problem, which changes
   the experiment. Owner deciding whether that's in scope.

## Roofline reality (so a fresh session doesn't re-discover it)
- Roofline numerator = torch.profiler GEMM/conv FLOPs only; elementwise counts ZERO.
  These models are mostly NOT GEMM (tiny seq → scalar attention; LN/softmax/CE/optimizer
  are memory-bound) → even perfect GEMMs read low-% of the flat 989 TF/s peak.
- One-CTA-per-SM cooperative megakernels saturate at B≈2k, decline past 16k → max batch
  helps GEMM size but not occupancy. Multi-CTA-per-tensor tiling is the open lever.

## Provisioning (owner cost-optimal): signal mi300x+TPU at race-launch (pinned checkout).
gfx942 + TPU pre-silicon lanes committed (D1 + Pallas interpret 229-green); day-1 = bring-up.

## Earlier roofline measurement TRAP (don't repeat): a chart once measured eager+L1
because the bf16 precision gate declined L3. wiring_check.py exists to prevent this —
ALWAYS run it before any roofline; a row's `path`/`l3_engine` must say wgmma.
