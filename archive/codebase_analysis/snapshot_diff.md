# Snapshot Diff: wt_preTP vs SuperGrok1.5 Live Tree

**Date:** 2026-06-25  
**Snapshot path:** `/workspace/wt_preTP/`  
**Live tree:** `/workspace/SuperGrok1.5/` (HEAD: e69df73, branch: claude/custom-optimizer-analysis-HFYhg)

---

## 1. Summary

The snapshot `wt_preTP` is a worktree copy taken just before (or at) the teleport. Comparing it against the live tree confirms:

- Dead code cleanup (8.09M lines, commit `8643cc2`) is real and complete — `_dectc_codegen/` (348 MB) and `_scan/` exist only in the snapshot.
- Post-snapshot commits added: NVSHMEM bringup pybind (`03bd3f0`), Mamba flagship smem redesign (`9936308`), TP track for ViT/Mamba (`5e084ca`), and closure docs (`e69df73`).
- No uncommitted source changes in the live tree (only `.pyc` files differ from HEAD).
- `.regpressure/` and `phase2/` referenced in HANDOFF.md are at `/workspace/` root, not inside either tree — from earlier campaign (branch `claude/h100-audit-maximal`), not the current one.

---

## 2. Files/Dirs Only in Snapshot (wt_preTP) — Dead Code or Removed

### `_dectc_codegen/` (348 MB)
- **Subdirs:** `baseline/`, `deep_s3/`, `deep_s3_ptx/`, `deep_s4/`, `grid/`, `postedit_default/`, `sizeprobe/`, `sizeprobe_dev/`
- **Files:** `grid.cpp`, `sizeprobe.cpp`
- **Status:** Dead code, removed in commit `8643cc2`. These were old codegen artifacts from an earlier decoder TC pipeline exploration. The live tree does not contain them.

### `_scan/` (43 files)
- **Files:** `rc_chunk_00` through `rc_chunk_29`, `review_candidates.txt`, `text_files.txt`, `text_hashes.tsv`, `text_unique.txt`, `all_files.txt`, `binary_files.txt`, `hits_*.txt`, `pat_*.txt`
- **Also at root:** `_scan_prep.log`, `_scan_prep.sh`
- **Status:** Dead. Scanning/audit artifacts from an earlier codebase review phase. Removed in same cleanup commit.

### `claude_session_archive/`
- **Contents:** `projects/` subdirectory only
- **Status:** Historical session archive. Not in live tree.

---

## 3. Files/Dirs Only in Live Tree (SuperGrok1.5) — Post-Snapshot Additions

### New Source Files (committed post-snapshot)

| File | Commit | Purpose |
|------|--------|---------|
| `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` | `03bd3f0` | NVSHMEM host-side pybind extension |
| `grokking_optimizers/nvshmem_bringup_ext.py` | `03bd3f0` | Python wrapper for NVSHMEM bringup |
| `tests/hw/nvshmem_smoke.py` | `03bd3f0` | Smoke test for NVSHMEM |
| `tuning/_tp8_build.sh` | `e69df73` | TP=8 build script for live 8-GPU run |
| `tuning/_tp8_run.py` | `e69df73` | TP=8 run driver |
| `tuning/_tp8_scratch_pybind.cu` | `e69df73` | TP=8 scratch pybind CUDA |

### New Session/Docs (post-teleport, not committed to source)
- `RESUME.md`, `SESSION_CONTEXT.md`, `.claude/`, `.session_context/`, `.session_memory/`

### New Parallel Module pycache (implies new .py sources committed earlier)
- `grokking_optimizers/parallel/__pycache__/`: `auto_config`, `flagship_budget`, `mem_strategy`, `resource_planner` pycs

### New hw test pycache (implies hw tests added post-snapshot)
- `_mamba_fill_test`, `decoder_kernel_mirror`, `test_3d_parallel`, `test_decoder_tc`, `test_distributed_step`, `test_dp2_loopback_determinism`, `test_l3tc_tail_gate`, `test_mamba_megakernel`, `test_mb3_scalar`, `test_multistep_parity`

### New unit test pycache
- `test_mem_strategy`, `test_pipeline_schedule`, `test_resource_planner`, `test_shard_map`, `test_zero3_plan`

### `third_party/cutlass/`
- The snapshot had an empty/stub cutlass dir; the live tree has the full CUTLASS submodule with `.git`, `include/`, `examples/`, `tools/`, etc.

---

## 4. Source Files That Differ (live is newer)

All differences are from the TP track (commit `5e084ca`) and Mamba flagship smem redesign (commit `9936308`).

### `csrc/fused/sm_90/fused_mamba_megakernel.cuh`
- **Lines ~97-99:** Added `#include parallel_config.cuh` + `tp_transport.cuh`
- **Line ~135:** `static_assert` now conditional: `kMbStreamSmem || sizeof(MambaSampleSmem)==kMambaSmemBytes`
- **New function `mb_tc_dyn_smem_bytes()`:** Returns `sizeof(MambaSampleSmem)` on streamed flagship path, `kMambaSmemBytes` on SMALL/bench path
- **Kernel template:** `<OptId Opt>` → `<OptId Opt, class Par = ::sg::fused::par::SingleGPU>`; launcher signature gains `CommCtx comm = {}`
- **Workspace layout change:** `acts_base = ws`, `part_base = acts_base + nCTA * mb_acts_stride_floats()` — layer-streaming scratch prefix carved before gradient partials
- **Comment note (line ~515):** Workspace comment now shows `[nCTA*acts_stride | nCTA*total | loss(nCTA) | ...]`

### `csrc/fused/sm_90/fused_vit_megakernel.cuh`
- **Lines ~101-103:** Added `#include parallel_config.cuh` + `tp_transport.cuh`
- **Kernel template gains Par + CommCtx** (same pattern as Mamba)
- **TP-aware forward/backward:** `if constexpr (!Par::kTPComm)` guard around default single-GPU tile loop; TP branch calls `_impl<Par,Transport>` bodies
- **New budget helpers:** `vit_tp_tile_slot_floats()` = `kTileM * vit::kD`; `vit_tp_heap_stride_floats(ctas_per_pe)` = `ctas_per_pe * 2 * vit_tp_tile_slot_floats()`

### `csrc/fused/sm_90/model_stage_mamba_tc.cuh`
- **New includes:** `parallel_config.cuh`, `tp_transport.cuh`
- **Layer-streaming re-exports:** `MbActsHbm`, `mb_acts_stride_floats`, `mb_acts_perlayer_floats`, `mb_acts_transient_floats`, `mb_acts_bind` re-exported from `sg::fused::sm90` into `mbtc::` namespace
- **Mamba-3 TP shard table (lines ~169+):** Full comment table of column/row splits across 45-tensor Mamba weight layout:
  - `in_proj(7)`: COLUMN split → no all-reduce needed on fwd
  - `out_proj(15)`: ROW split → fwd all-reduce ①
  - `gate(17)/up(18)`: COLUMN split → no reduce
  - `down(19)`: ROW split → fwd all-reduce ②
  - `x_proj/dt_proj/scan internals/norms/tok/pos/head/D/A_log`: REPLICATED
  - Note: SSM (x_proj→dt/A/lam/B/C→scan) runs on FULL d_inner (no attention head-shard touch)

### `csrc/fused/sm_90/mega_mamba_real_adamw_tc_launcher.cu`
- **`#if SG_HAS_NVSHMEM` blocks:** NVSHMEM headers only included when `-DSG_HAS_NVSHMEM=1`
- **`MbTcLauncherScratch` struct gains:** `float* tp_sym_heap = nullptr` and `int64_t tp_sym_floats = 0`
- **`mb_tc_ensure_tp_sym_heap()`:** Collective function to allocate/grow symmetric TP-slot heap via `nvshmem_malloc`; mirrors `dec_tc_ensure_tp_sym_heap` pattern

### Also differ (contain TP/layer-streaming changes):
- `mega_mamba_real_adamw_tc.cu` — main Mamba TC kernel launcher body
- `mamba3_layout.cuh` — layout helper changes for TP
- `mamba_flagship_layout.cuh` — flagship-path layout
- `mega_vit_real_adamw_tc_launcher.cu` — ViT launcher TP heap support
- `model_stage_vit_tc.cuh` — ViT TP-aware stage bodies
- `model_stage_mamba3.cuh` — Mamba3 stage with layer-streaming

---

## 5. Uncommitted Changes in Live Tree

**Only `.pyc` files** (21 modified pycache files). No uncommitted source changes. Live tree is clean at HEAD `e69df73`.

---

## 6. Dead Code Cleanup Verification

The CLAIMED state says "removed 8.09M lines" in commit `8643cc2`. This is CONFIRMED:
- `_dectc_codegen/` (348 MB of codegen artifacts) exists in snapshot, absent in live ✓
- `_scan/` (audit artifacts, 43 files) exists in snapshot, absent in live ✓
- No other large dead-code directories found missing from live

The `.regpressure/` and `phase2/` directories referenced in HANDOFF.md (branch `claude/h100-audit-maximal`):
- These exist at `/workspace/.regpressure/` and `/workspace/phase2/` (workspace root, not inside SuperGrok1.5)
- They are NOT inside `wt_preTP/SuperGrok1.5/` and were never part of the current branch's committed state
- HANDOFF.md note that these are from an earlier campaign is CONFIRMED

---

## 7. Valuable Content in Snapshot Only

The only potentially valuable content uniquely in the snapshot (not in live or workspace root):

1. **`_dectc_codegen/`**: Baseline and deep-exploration PTX/SASS outputs from earlier decoder TC codegen experiments (`deep_s3`, `deep_s4`, `grid`). Could be useful for register pressure / occupancy reference but are byproducts of removed code paths. Size 348 MB.

2. **`_scan/` chunks**: `rc_chunk_00..rc_chunk_29` are raw scan results of the full codebase; `review_candidates.txt` and `text_unique.txt` could have reference value for auditing what was removed. Minor value.

3. **`claude_session_archive/projects/`**: Historical session data. No source value.

**Conclusion:** Nothing in the snapshot represents missing source code that should be recovered. The dead-code removal was clean and intentional.
