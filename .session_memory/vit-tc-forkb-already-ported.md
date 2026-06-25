# ViT TC Fork-B grad-partial: ALREADY PORTED (campaign-correcting finding, 2026-06-25)

Investigated task #31 ("port decoder Fork-B grad-partial elimination to ViT"). The
premise is STALE — it is already done in the production ViT TC path.

## Two ViT megakernels in fused_vit_megakernel.cuh
- SCALAR `fused_vit_megakernel` (#if SG_VIT_SCALAR_MEGAKERNEL, ~L184-349): the ONLY
  one with the nCTA*total grad partial. Allocated by gate-only `scalar_train_step`
  (mega_vit_real_adamw_tc.cu:335 = `n_sms*total + n_sms + 1` = the 51GB term).
  Compiled OUT at flagship/bench (VitSampleSmem > 227KB smem cap). NEVER shipped.
- TC/Fork-B `fused_vit_megakernel_tc` (#if SG_TUNED_GEMM_IMPL==WGMMA, L503+): the
  production "L3-TC persistent wgmma megakernel". Workspace = vit_tc_workspace_floats
  (L479) which has NO nCTA*total. Already has: HBM bf16 acts (VitActs/vit_acts_bind),
  P2 output-stationary dW (vittc_dw_run_tile, owner gt%nCTA), split-K dW
  (vittc_dw_run_tile_splitk/_reduce_splitk, vit_dw_part_floats), cls/pos owner-scan.
  Launcher comment (mega_vit_real_adamw_tc_launcher.cu:14) says it verbatim.

## The ONE thing ViT didn't adopt from the decoder
Decoder: SG_TUNED_DEC_DW_SPLITK=1 (after adding contiguous-transpose staging
SG_TUNED_DEC_DW_STAGE=1, which made single-CTA dW 2.05x faster) ⇒ dec dw_part==0.
ViT: SG_TUNED_VIT_DW_SPLITK=4 (model_stage_vit_tc.cuh:107-109) AND has NO
contiguous-transpose staging at all (no SG_TUNED_VIT_DW_STAGE / dYt / Xt /
vit_dw_transpose_operands). So ViT still carries vit_dw_part_floats(4) = 25.5 GB at
flagship d=1664.

## Apply-ready spec: /workspace/impl_diffs/vit_forkb.md
- EDIT 2A (SAFE, in-scope): flip SG_TUNED_VIT_DW_SPLITK 4->1. dw_part->0, kernel takes
  single-CTA branch (byte-identical to G=4 reduce at G=1). -25.5GB at flagship,
  test_vit_tc.py stays green (grad-parity vs bf16 oracle + determinism, both G-agnostic).
  CAVEAT: without staging the G=1 dW is SLOWER (scalar gather, no grid-fill) — a
  memory<->dW-speed trade. NO launcher/workspace edit needed (all keys off kVitDwSplitK).
- EDIT 2B (LARGE, OUT OF SCOPE): port decoder contiguous-transpose dW staging (~150
  LOC, full symbol map in spec) to make G=1 fast. It's a SPEED enabler, NOT a memory
  fix (transpose scratch is also batch-bound, tens of GB at flagship).

## The REAL flagship 80GB blocker (NOT the grad partial)
vit_tc_acts_floats (Fork-B HBM bf16 acts) ~= 379 GB at the grid-saturating batch
(B>=8448 to fill 132 tiles of kTileM=1088; bench uses B=8704). Removing the 25.5GB
dW partial does NOT bring 379GB under 80GB. The acts buffer is an
activation-memory/recompute/batch problem, separate from grad partials.
Per tuning/roofline.py BATCH_SATURATION_SWEEP: the 1-CTA/SM megakernel saturates at
B~2k and "VRAM is NOT the binding constraint" (peak VRAM <8GB even at B=131072,
d=128). The real cap on ncta is occupancy (1 CTA/SM), not HBM. At B~2k the workspace
already fits ncta_cap=8.

VERDICT: literal port = no-op (already done). Only byte-identical win = DW_SPLITK=1
(-25.5GB). "ncta_cap=8 within 80GB at flagship" unreachable via grad-partial changes.
