---
name: megakernel-validation-ladder-gaps
description: CPU structural mirrors do NOT catch flat-stride/race bugs — only GPU-vs-fp64-oracle gates do; plus calibrated parity-gate tolerances for fp32 training kernels
metadata: 
  node_type: memory
  type: project
  originSessionId: 122b3aee-fa74-43c0-936d-39ff1eca854f
---

Lesson from the L3-real decoder no-learn bug (2026-06-10, fixed in 5cb32b8):

**The validation ladder oracle(fp64)→mirror(structural)→CUDA has a hole**: the
CPU mirror indexes buffers *semantically* (right-sized 2D tensors), so it cannot
see (a) flat-stride vs declared-width mismatches in smem scratch reuse, or
(b) any race. The decoder kernel passed mirror at 1e-12 while every grad
upstream of attention was ~1e5 wrong on GPU (packed-stride linear_bwd writing
into a kDff-wide buffer). Same latent bug existed in the ViT header (never yet
GPU-run) — found by pattern-audit, ~70 seams checked across vit/mamba/sg2.

**How to apply:**
- A megakernel cell is NOT validated until the GPU single-step gate (kernel grad
  vs fp64 oracle, per-tensor) has run on silicon. Treat mirror-green as
  "transcription plausible", nothing more.
- When a helper takes raw float* + a width, make row strides EXPLICIT ld*
  parameters (no defaults) — the [[no-functionality-suppression]] of indexing.
- Debug signature worth remembering: loss pinned at exactly ln(vocab), acc at
  1/vocab = zero/garbage grads + live weight decay shrinking params to uniform
  logits. Forward fine + upstream-of-X grads garbage = first corruption is at
  X's backward seam (backward runs reverse; everything computed after the bad
  stage in TIME inherits the poison).
- Calibrated gate tolerances for fp32 train-step parity vs eager (worked out
  with controls, recorded in test_megakernel_vs_eager.py):
  - params-after-1-step needs an ABS floor of 0.05*lr: AdamW step-1 update is
    lr·g/(|g|+eps); on true-zero-grad elements both implementations hold fp32
    reduction noise |δg|~5e-10 → |Δp| up to lr·δg/eps. Pure-rel gates also
    collapse on zero-init tensors.
  - 200-step final-params gate = 3× the MEASURED chaos floor (eager-vs-eager
    with one 1-ulp perturbation separates to ~6e-3 on grokking decoders). The
    sharp gate is the per-step LOSS curve (<1e-3), which a systematic formula
    error shifts immediately.
- kill -9 of a CUDA client under MPS can wedge the server in "Teardown in
  progress" → ALL new clients get error 807 AND existing clients throttle
  (~10× slower steps). Recovery: stop clients, `echo quit | nvidia-cuda-mps-control`,
  restart daemon + fleet. Prefer SIGTERM for CUDA processes under MPS.

STALE-BINARY TRAP (2026-06-11): a gate "NaN/non-determinism" failure was a
STALE _ops.so compiled from a prior session's vanished WIP source — clean
rebuild + compute-sanitizer (initcheck/racecheck 0) cleared it, no live bug.
Before chasing a kernel gate failure: FORCE_CUDA=1 ./build.sh fresh, confirm
the .so postdates the source. Working-tree can also carry a prior session's
uncommitted WIP behind a "clean" git snapshot (Teleport auto-stash) — verify.
