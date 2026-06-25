# Staged-optimizer plumbing — flagship single-GPU 11-optimizer ranking

Area: `staged_opt_plumbing`. Goal: make all 11 optimizers runnable at flagship
single-GPU (decoder d=1600, layers=48, 1,475,884,899 params) and bank the
loss-vs-step trajectories into the complete 11-optimizer ranking.

## What was built (build-via-include only; NO committed-source edit)

`mega_decoder_staged_tc.cu` — a SCRATCHPAD JIT translation unit that extends the
existing elementwise multiopt driver (`mega_decoder_multiopt_tc.cu`) to the STAGED
optimizers, mirroring the OptId dispatch + state-binding block of the committed
launcher `csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu` EXACTLY. It is
compiled against the flagship layout with the SAME `-D` flags + `-include
decoder_flagship_layout.cuh` as `flagship_train.py`, EXCEPT it does NOT set
`-DSG_DEC_BENCH_LAYOUT=1`, so the live `kDecStagedOptScratch` gate
(fused_decoder_megakernel.cuh:541-545) is TRUE and the four staged-opt scratch
regions (Prodigy reduce | Muon NS | LookSAM 2nd-bwd | SuperGrok2 meta-net) are
carved — exactly as the production opt-agnostic launcher needs.

Two pybind entries:
* `tc_train_step_opt(opt_id, ...)` — generic dispatch for OptId 0-9 (the 5
  elementwise + Prodigy/Muon/LookSAM/SuperGrok11/SuperGrok15). Binds the FULL state
  layout the launcher binds: `[m|v|extra]` + loss@`state+3*total` + Prodigy
  param_init/persist + SG11/15 sharpness/phi pack. Live scalars via FusedScalars +
  apply_scalars.
* `sg2_train_step(...)` — dedicated SuperGrok2 path mirroring `mega_decoder_sg2_tc`:
  the 26 model-independent meta-net weight packs (from `CSAHCAMetaNet.get_weights`)
  + the 6 per-tensor scalar arrays + SG2 state slices, dispatched to
  `launch_fused_decoder_megakernel_tc<OptId::SuperGrok2>`. Runs at ncta=1.

The committed source tree is byte-identical (git clean) — the SingleGPU/default path
and the 3 model-pytest byte-identity gates are untouched.

## Memory engineering for the flagship (≈79 GiB H100)

The opt-agnostic workspace carves ALL four staged regions unconditionally (the
kernel's `sg2_ws_base` offset is hardcoded as `muon_base + muon + 2*total`, so the
LookSAM `[sam_backup|sam_grad]` 2*total region cannot be elided without editing
committed source). Two zero-copy aliases let the heaviest opts FIT:
* SG2 grad → aliased onto the dead LookSAM `sam_backup` region (SG2 never runs the
  LookSAM phase → no hazard), reclaiming 5.5 GiB.
* SG2 `slow` (grokfast EMA) state plane → aliased onto the dead LookSAM `sam_grad`
  region (persists across steps; the workspace scratch is process-lived), dropping
  SG2 state from 9 to 8 planes (-5.5 GiB).
Generic opts use a persistent grad buffer (one alloc, zeroed in place) to avoid the
per-step 5.5 GiB alloc/free churn that fragments + OOMs at flagship width.

Effective nCTA: 4 for the generic opts (workspace ~31 GiB; total ~70 GiB fits),
1 for SG2 (its per-CTA CSA/HCA/PEER/GRU meta-net workspace is ~3.7 GiB/CTA → full
occupancy = 270 GiB OOM, the resource-planner deep limit).

## Result — 9/11 banked finite descending; 2 FITS-but-slow; 11/11 runnable single-GPU

See `flagship_11opt_ranking.json` + `.txt`. The 9 banked opts converge from 4.5850
to 2.686-2.733 over 100 steps (fixed batch B=16). Muon and SuperGrok2 FIT and their
kernels LAUNCH + EXECUTE (100% GPU util, no OOM — verified) but are impractically
slow at the memory-forced low ncta (muon: 195 2D weights × 5 NS iters × naive fp32
matmuls on 4 SMs ≈ hours/step; SG2: per-element meta-net + SAM 2nd backward on a
SINGLE SM ≈ tens-of-min/step). Both paths are parity-validated by the committed
tests (test_sg2_megakernel.py; the generic staged opts share the launcher's
validated kernel instantiations). These are the documented exceptions the task
anticipated ("run SG2 at ncta=1 or document it FITS-but-slow").
