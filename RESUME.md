# SuperGrok2 — RESUME GUIDE (instance-closure recovery)

Written 2026-06-25 at end of session. **Everything outside `/workspace` was deleted on closure** — this
guide restores the rest. The git repo (`/workspace/SuperGrok1.5/.git`), all deliverables
(`/workspace/phase6`), specs (`/workspace/impl_diffs`), `PROGRESS.md`, and the memory backup
(`.session_memory/`) are all under `/workspace` and survived.

## 1. RESTORE THE INSTANCE (reinstall the deps that lived outside /workspace)
```bash
pip install nvidia-nvshmem-cu12 optuna ruff nvidia-ml-py     # session-added deps (deleted on closure)
cd /workspace/SuperGrok1.5
git config user.email "pcoy400@gmail.com" && git config user.name "SuperGrok2 session"
# restore the persistent memory (originals were at /root/.claude/... = deleted):
mkdir -p /root/.claude/projects/-/memory && cp .session_memory/*.md /root/.claude/projects/-/memory/
# NVSHMEM env for the 8-GPU path:
export NVSHMEM_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem
export NVSHMEM_DISABLE_NVLS=1 NCCL_NVLS_ENABLE=0
```
`detect_arch`/imports need torch+cuda (in the base image). ncu HW counters stay DENIED (container);
nsys works. See `.session_memory/ncu-blocked-runpod.md`, `nvshmem-installed.md`.

## 2. GIT STATE
Branch `claude/custom-optimizer-analysis-HFYhg`. Chain (newest first):
`03bd3f0 nvshmem_pybind → 9936308 mamba smem redesign → 8643cc2 cleanup(-8.09M lines) → 5e084ca vit/mamba TP
→ ... → c1230dc ep_size → 81f1bfb resource planner → 7f9e772 datasets → ed1bb55 ViT flagship → 5733af5 TP
foundation → b92442b session base`. (Plus this closure commit adding RESUME.md + .session_memory + tuning/_tp8_*.)

## 3. WHAT'S DONE + VALIDATED (the full self-adapting stack)
- CuTe-atom GEMM engine (bit-identical, behind SG_TUNED_GEMM_ENGINE).
- 3 flagship models LAUNCH (~1.5B): decoder, ViT, Mamba (Mamba via the smem redesign 19.56MB→193KB).
- Full TP (Par-template + grid-lockstep loop + 4 reduce points + sym-heap launcher) for all 3 models.
- **Cross-GPU in-kernel device-NVSHMEM TP all-reduce VALIDATED on 8 GPUs** (bit-exact; 2/4/8-GPU smokes).
- Resource planner (workload×hardware→config, 10/10 tests); EP/3D-5D auto_config; size-adaptive CTA-tiling
  selector; memory-strategy (offload/recompute/stream gates); datasets Layer-A (pluggable bring-your-own).
- Dead-code cleanup: removed 8.09M lines of artifacts; true source ~361K. LOC report in PROGRESS.md.
- DELIVERABLE #1 roofline graph: /workspace/phase6/roofline_flagship.{png,csv} (10 cells; occupancy-bound).
- 11-optimizer flagship-decoder ranking (OVERFIT placeholder): /workspace/phase6/flagship_11opt_ranking.{json,txt}.
- Decoder pytest 19/19, ViT 21/21 byte-identical; Mamba 3/5 (2 PRE-EXISTING fails: B_bias-tol + obsolete proj_dw).

## 4. REMAINING WORK (ETA ~5–8 hr impl, then the benchmark RUN = GPU-hrs)
1. **TP data-path fix** (the live one-model-across-8 TRAINING run): the 8-GPU run surfaced 3 megakernel bugs.
   Bug A (per-rank weight-shard offset) + B (full-width attn on kTPComm for 25 heads) are FIXED in the WIP
   patch `/workspace/phase6/tp_datapath_fix_WIP.patch` (UNGATED — apply, then finish bug C: confirm IMA cleared
   under compute-sanitizer). Re-run: `tuning/_tp8_build.sh` + `torchrun --nproc_per_node=8 tuning/_tp8_run.py`
   (the scratch pybind `tuning/_tp8_scratch_pybind.cu` calls the launcher's tp_size=8 arm). Gate: decoder
   pytest 19/19 + 8-GPU sanitizer-clean + cross-rank loss agrees + descends. ~1–2 hr. (Details: PROGRESS.md
   "8-GPU FLAGSHIP RUN" section.)
2. **Real-data benchmark** (Layer-B): wire FineWeb-Edu/ImageNet-1k/GiftEval into the datasets Layer-A seam
   (impl_diffs/datasets_v2.md), replace the mod-97/overfit, run the real 11×3 ranking. ~3–5 hr + the run.
3. **Full 33-cell roofline** (Mamba now launches): re-run the campaign /workspace/phase6/roofline_campaign.js
   for all 3 models × 11 opts (nsys). ~1–2 hr.
4. **ViT re-measure** at the saturating batch (~2k) — the roofline ncta=4 was a conservative artifact. ~0.5 hr.
5. Optional: VIT_DW_SPLITK 4→1 (−25.5GB byte-id); the 56-line Mamba dead-source removal (impl_diffs/deadcode_source.md);
   Mamba/SG2 single-GPU occupancy (ncta) needs fewer always-on opt carves or a 2nd-GPU shard.

## 5. WHERE THINGS ARE
- Apply-ready specs (all the design work): `/workspace/impl_diffs/*.md`.
- Workflows (re-runnable): `/workspace/phase6/*.js` (roofline_campaign, flagship_8gpu_run, tp_datapath_fix, etc.).
- Deliverables: `/workspace/phase6/{roofline_flagship.png,.csv, flagship_11opt_ranking.*}`.
- 8-GPU run wiring: `tuning/_tp8_{build.sh,run.py,scratch_pybind.cu}` (committed).
- Flagship runners: `/workspace/phase1/flagship_{train,smoke}.py`, `tuning/flagship_distributed.py`.
- Full running ledger: `/workspace/PROGRESS.md`. Memory: `.session_memory/` (restore per §1).
