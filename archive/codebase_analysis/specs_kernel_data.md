# specs_kernel_data.md — Digest of impl_diffs kernel/data specs

Agent assignment: read IN FULL all 16 impl_diffs spec files and record design intent,
applied-vs-on-disk status, and gate conditions for each.

Date: 2026-06-25. All paths relative to /workspace/SuperGrok1.5 unless noted.

---

## 1. cute.md — CuTe GEMM Engine (SG_TUNED_GEMM_ENGINE)

### Design Intent
Replace the hand-rolled inline PTX wgmma instructions in `csrc/backends/cuda/sm_90/wgmma.cuh`
with CuTe device atoms from CUTLASS, behind a compile-time knob `SG_TUNED_GEMM_ENGINE`.
Maintains byte-identity with the existing implementation when the knob is OFF (default=0).
ENGINE=1 routes through `cute::SM90::GMMA::MMA_64xNx16_F32BF16BF16_SS::fma`,
`cute::warpgroup_arrive`, `cute::warpgroup_commit_batch`, `cute::warpgroup_wait`, while
preserving the same `sg::sm90::wgs::` ABI.

### Applied vs On-disk Status
APPLY-READY (not yet applied). Single-file change with 5 concrete edits to wgmma.cuh.
The entire CuTe code is `#if`-erased when `SG_TUNED_GEMM_ENGINE=0`. No CuTe code
exists in the live tree under this guard.

### Key Facts
- Default `SG_TUNED_GEMM_ENGINE=0` is provably byte-identical (entire block is `#if 0`-erased)
- ENGINE=1 uses `cute::GmmaDescriptor` + CuTe atoms; same sg::sm90::wgs:: ABI preserved
- One honest runtime difference: scale_D predicate becomes runtime vs compile-time immediate
  (NOT a math difference; both produce the same numeric output)
- Precondition: ENGINE=1 requires `-DWITH_CUTLASS`

### Gate Commands
```
nvcc wgmma_selftest.cu -DSG_TUNED_GEMM_ENGINE=1 -DWITH_CUTLASS ... -o wgmma_test
python tests/hw/test_decoder_tc.py -k wgmma
```

---

## 2. tma.md — TMA Staging Sub-Knob (SG_TUNED_GEMM_TMA)

### Design Intent
Step 4 of the CuTe integration: add TMA (Tensor Memory Accelerator) load staging as a
sub-knob `SG_TUNED_GEMM_TMA` (requires ENGINE=1). Adds:
- Host header `csrc/backends/cuda/sm_90/cute_tma_desc.h` — `sg_build_tma_desc_kmajor<TILE_MN>()`
  that constructs a `CUtensorMap` for a K-major 2D weight tile
- Device primitive `tma_load_kmajor_tile(tma_desc, mbar, smem_tile, crd_mn, crd_k)` using
  `cute::SM90_TMA_LOAD::copy` + `Mbarrier::arrive_expect_tx/try_wait`

### Applied vs On-disk Status
Sections 2-4 edits are APPLY-READY. Section 5 (megakernel wiring, which is tma_wire.md)
is documented but NOT applied here.
VERIFIED NOT YET APPLIED: `grep SG_TUNED_GEMM_TMA csrc/backends/cuda/sm_90/wgmma.cuh` → no hits.
`cute_tma_desc.h` does NOT exist in the live tree.

### Key Facts
- Sub-knob hierarchy: TMA=1 requires ENGINE=1; both default to 0 (byte-identical)
- tok.workspace is cudaMalloc'd once per device, reused across steps → TMA descriptors valid
- dW transposed-strided gather NOT TMA-reachable without `SG_TUNED_DEC_DW_STAGE=1`
- Host builder: `sg_build_tma_desc_kmajor<TILE_MN>(const __nv_bfloat16* base, int rows, int K)`
- kDecTmaNumDesc = 16 * kLayers (8 weight + 8 acts operands per layer)

---

## 3. tma_wire.md — TMA Megakernel Wiring (Section 5 from tma.md)

### Design Intent
Full wiring of TMA into the persistent megakernel launcher. Applies across 7+ files:
PersistentContext struct, dispatch.cpp CUDA mirror, dec_acts_bind/dec_wbf_bind host/device,
tc_gemm_block_unpipelined parameter additions, and the Stage_k_tma lambda.

### Applied vs On-disk Status
APPLY-READY but NOT YET APPLIED to the live tree. Contains exact OLD→NEW edit blocks.
`SG_TUNED_GEMM_TMA` absent from wgmma.cuh confirmed (not applied).

### Key Facts
- PersistentContext additions: `const void* g_tma_desc = nullptr; int n_tma_desc = 0;`
  (trailing-defaulted so ABI forward-compatible)
- dispatch.cpp CUDA mirror struct must be updated in lockstep
- SG_DEC_TMA_* macro idiom mirrors SG_DEC_PIPE_BARS_ARG pattern (byte-identical when OFF)
- tc_gemm_block_unpipelined gets `SG_DEC_TMA_TILE_PARAMS` trailing defaulted params
- Stage_k_tma lambda uses mbarrier arrive_expect_tx/try_wait pattern
- Risks: wbf_f offset helper duplicates kernel carve; B operand n0 + CTA-local mbarrier
  need lead completion verification; sg2 +1 realign unverifiable as read-only

---

## 4. flagship.md — Decoder Flagship Layout Emitter

### Design Intent
Parameterize `megakernel_codegen.py`'s `_decoder_param_sizes` and `_decoder_layout_body`
to accept `(d, layers, vocab, seq, heads)` instead of hardcoded constants. Add a new
`decoder_flagship_layout_header()` emitter for d=1600, L=48, heads=25, vocab=99, seq=4.
Emit to `csrc/fused/sm_90/decoder_flagship_layout.cuh`.

### Applied vs On-disk Status
APPLY-READY, not yet applied. Existing `--decoder-layout` output byte-identity verified.

### Key Facts
- kDecNumTensors = 2 + 12*48 + 4 = 582
- kDecTotalElems = 1,475,884,899 (fits int64_t)
- _DEC_FLAGSHIP_D=1600, _DEC_FLAGSHIP_HEADS=25, _DEC_FLAGSHIP_LAYERS=48
- Gate: `python -m grokking_optimizers.megakernel_codegen --decoder-layout-flagship`
- RISK 1: consumer kernel is L=2-pinned until flagship_dw.md edits land
- RISK 2: smem budget may not fit at L=48 (per-tile smem ∝ kLayers); needs compile probe

---

## 5. flagship_dw.md — Decoder L-Generalized Backward Enumerations

### Design Intent
L-generalize the decoder TC backward: replace hardcoded `__constant__` array tables
with closed-form constexpr formula accessors. Edits span:
- `model_stage_decoder_tc.cuh` (Edits 1.1-1.13)
- `fused_decoder_megakernel.cuh` (Edits 2.1-2.3)
- `pp_stage_decoder_tc.cuh` (Edits 3.1-3.2)

### Applied vs On-disk Status
APPLY-READY, not yet applied.

### Key Facts
- kDecNumDwSpecs = 4*kLayers+1 (=9 at L=2, byte-identical)
- kNumLnVec = 4*kLayers+2 (=10 at L=2, byte-identical)
- kDecNumMuon2D = 4*kLayers+3 (=11 at L=2, byte-identical)
- dec_lnvec_tensor_idx(v): `6 + 12*(v/4) + (v%4)` for v<4L, else `2+12*L+(v-Lx4)`
- dec_muon_2d(mi): closed formula for tok/pos/per-layer/head
- dec_is_muon_2d(t): `t∈{0,1} OR t==head.out OR ((t-2)%12∈{0,2,8,10})`
- All formulas PROVEN byte-identical at L=2 by replaying literal tables
- DecTcSmem::spec[9] → spec[kDecNumDwSpecs] (Edit 2.1, byte-identical at L=2)
- RISK: Muon table→formula PTX NOT byte-identical (different code path, but Muon
  cell not exercised by AdamW gate)

---

## 6. profiler.md — Decoder Phase Profiler Harness

### Design Intent
Add analysis tooling for the existing profiler (SG_DEC_PROFILE, SG_DEC_PROFILE_FWD_FINE)
as a new Python file `tuning/decoder_phase_profile.py`. No kernel edits; the profiler
already exists. The spec documents: slot mappings, campaign findings, and a new analysis
script.

### Applied vs On-disk Status
Kernel profiler already exists and is ON-DISK. New Python harness file is APPLY-READY
(create new file only). SG_DEC_TC_PHASE_PROF does NOT exist and is NOT needed.

### Key Facts
- 8 coarse slots: [0] P1_fwd, [1] P1_bwd, [2] B1_barrier, [3] P2_dW_GEMM,
  [4] P2_grad_asm, [5] P3_opt_tail, [6] B2_barrier, [7] B0_barrier
- 10 fine slots: fwd ring {ISSUE,WAIT(drain),WGMMA,EPI,BARRIER} × dX ring
- Campaign finding: P1_fwd 28.8% + P1_bwd 27.7% ≈ 56.5% of total time
- fwd ring WAIT-dominant (43% WAIT vs WGMMA) → motivation for FWD_PIPE=1, FWD_STAGES=4
- d=2048 bench uses SG_DEC_BENCH_LAYOUT (docstring says d=1024 but actual constant is 2048)
- Occupancy: 1 CTA/SM by design; SG_TC_MEGA_BLOCK=256

---

## 7. mamba_flagship.md — Mamba-3 Flagship Layout Emitter + L-Generalized Backward

### Design Intent
Mirror decoder flagship for Mamba-3: parameterize `megakernel_codegen.py` for Mamba,
add `mamba_flagship_layout_header()` for d=2048/L=24, and L-generalize `kMbMuon2D[]`
in `model_stage_mamba_tc.cuh` + `fused_mamba_megakernel.cuh`.

### Applied vs On-disk Status
APPLY-READY, not yet applied.

### Key Facts
- CRITICAL: n_heads=64 at d=2048 (derived d_inner//head_dim = 4096//64), NOT config's num_heads:32
- _MAMBA_FLAGSHIP_D=2048, _MAMBA_FLAGSHIP_LAYERS=24
- kMambaNumTensors = 2 + 20*24 + 3 = 485
- kMambaTotalElems = 1,265,411,169
- BUG FIXED: kMbMuonMaxNumel was kXProj*kDInner (too small at flagship; in_proj is largest)
  FIX: kMbMuonMaxNumel = kMambaMaxTensorNumel (layout-derived, exact at every width)
- kMbNumMuon2D = 2+7*L+1 = 7*L+3 (byte-identical at L=2 if old table had 17 entries)
- kMbNumDwSpecs = 8 (fixed dormant constant, not layer-scaled)
- MbTcSmem::spec[8] → spec[kMbNumDwSpecs]
- Gate typo found: task's gate uses SG_FUSED_SM90_MAMBA_LAYOUT_CUH_ (missing "3"),
  real guard is SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_

---

## 8. mamba_smem_redesign.md — Mamba-3 Shared Memory Redesign

### Design Intent
Diagnose and fix the showstopper: the Mamba-3 TC kernel uses `MambaSampleSmem` as
dynamic smem (NOT `MbTcSmem`). At flagship (d=2048), kMambaSmemBytes = 20,513,956 bytes,
far over the 227 KB cudaFuncSetAttribute cap → kernel cannot launch.

Design two redesign levels:
- Level A (exact spec): one LayerAct + kMbActsRing=2 ring; cross-layer acts to HBM MbActsHbm
  → 19.56 MB → 1.74 MB (still > 227 KB cap)
- Level B (structure only): stream SEQ×{DINNER,DFF,D} scratch to HBM → ~120.85 KB (fits)

### Applied vs On-disk Status
APPLY-READY spec for kernel redesign; NOT YET APPLIED in live tree.

### Key Facts
- Prod d=128: 210.79 KB (fits); Flagship d=2048: 19.564 MB (88× over cap)
- MambaSampleSmem at flagship: L*SEQ*D (layer_in) + L*LayerAct accounts for 95.4%
- Gate predicate: kMbStreamSmem = (kMbAllLayersSmemFloats * sizeof(float)) > (227*1024)
- False at d=128/1024 → byte-identical; True at flagship
- Helper functions: mb_smem_la(li): kMbStreamSmem ? 0 : li; mb_ring(li): kMbStreamSmem ? (li%kMbActsRing) : li
- Workspace adds at front: nCTA * mb_acts_stride_floats() floats (zero on non-streamed)
- kMbStreamSmemBytes = 1,827,684 B = 1784.85 KB (Level A) — still won't launch
- Level B target: ~120.85 KB (fits within 227 KB cap)

---

## 9. vit_flagship.md — ViT Flagship Layout Emitter + L-Generalized Backward

### Design Intent
Mirror decoder flagship for ViT: parameterize `megakernel_codegen.py` for ViT,
add `vit_flagship_layout_header()` for d=1664, L=48, heads=16.
L-generalize backward enumerations across:
- `megakernel_codegen.py` (Edits 1.1-1.5)
- `model_stage_vit_tc.cuh` (Edits 2.1-2.12)
- `fused_vit_megakernel.cuh` (Edits 3.1-3.3)

### Applied vs On-disk Status
APPLY-READY, not yet applied.

### Key Facts
- _VIT_FLAGSHIP_D=1664, _VIT_FLAGSHIP_HEADS=16, _VIT_FLAGSHIP_LAYERS=48
- heads=16 explicitly (NOT d//64=26 rule)
- kVitNumTensors = 4 + 12*48 + 4 = 584 (4 lead tensors vs decoder's 2)
- kVitTotalElems = 1,596,200,417
- Per-layer base = 4 + 12*li (ViT has 4 lead tensors vs decoder's 2)
- kVitNumDwSpecs = 4*L+2 (=10 at L=2); kNumLnVec = 4*L+2 (=10); kVitNumMuon2D = 2+4*L+1 (=11)
- vit_lnvec_tensor_idx(v): `8 + 12*(v/4) + (v%4)` for v<4L, else `4+12*L+(v-Lx4)`
- VitSampleSmem at flagship ≈ 2,304,784 B (2250.77 KB) — scalar megakernel gated OFF
- VitTcSmem spec[] grows by +7.36 KB at L=48 (only smem-dependent change)
- Edit 2.11: gn_normw/gn_normb use `(4*L+{0,1})*kD` instead of `8*kD`/`9*kD`
- Already L-general (no edit needed): VitActs, VitTileScratch, vittc_backward_tile
  per-layer LN slots, kVitDwMaxTiles, kVitMuonMaxNumel, kLnVecElems
- NO pp_stage equivalent for ViT (symbols only in model_stage_vit_tc.cuh + fused_vit_megakernel.cuh)
- Gate 3 typo: uses -DSG_DEC_SCALAR_MEGAKERNEL=0 (decoder gate, no-op for ViT TU)

---

## 10. vit_forkb.md — ViT Fork-B Grad-Partial Elimination Status

### Design Intent
Audit whether ViT TC kernel needs Fork-B (eliminate per-CTA full-grad partial).
Finding: the TC kernel already has Fork-B; only one small change still needed.

### Applied vs On-disk Status
HEADLINE FINDING: ViT TC persistent megakernel ALREADY has Fork-B (the per-CTA
full-grad partial is ALREADY eliminated). Only ONE change remains.

### Key Facts
- Scalar path (SG_VIT_SCALAR_MEGAKERNEL, gated off at flagship) HAS the nCTA*total partial
- TC path does NOT — it already uses HBM bf16 acts (VitActs) with no nCTA*total term
- vit_tc_workspace_floats has NO nCTA*total term in the live tree
- Flagship B=8704: acts buffer ~379 GB dominates; ncta_cap=8 within 80 GB NOT achievable
  via grad-partial changes alone
- ONLY remaining change: EDIT 2A — SG_TUNED_VIT_DW_SPLITK default 4→1 (-25.5 GB dW partial)
- G=1 is bit-identical to G=4 at G=1 (one chunk = full-K ascending-k accumulate)
- CAVEAT: G=1 dW will be SLOWER at flagship without contiguous-transpose staging (2B, out of scope)
- Insertion location: model_stage_vit_tc.cuh:107-109

---

## 11. datasets.md — Layer-A Data Plumbing (v1 draft)

### Design Intent
Add scaled-dataset support to `grokking_race_v2.py` via a new DEFAULT-OFF config axis
`c['data_source']`. Create new module `grokking_optimizers/dataset_sources.py` with
deterministic stub loaders for decoder/mamba/vit. Remove `_maybe_wrap_cuda_graph` dead code.
The mod-97 path must be BYTE-IDENTICAL when data_source=="modular" (the default).

### Applied vs On-disk Status
APPLY-READY (v1 draft — see datasets_v2.md for validated v2 with unique OLD→NEW blocks).

### Key Facts
- New DEFAULT_CONFIG knobs: data_source, train_batch_size, train_view_rows, eval_probe_rows,
  eval_micro_batch, early_stop_mode, early_stop_plateau_patience, early_stop_plateau_min_delta
- make_data_for_task dispatches: data_source=="modular" → literal original 4 lines
  (byte-identical); else → lazy import make_source_for_task
- evaluate() gains micro_batch param (default 0 → byte-identical single-shot)
- EarlyStopper gains mode="acc" axis with loss-PLATEAU branch for LM/forecasting
- Dead-code: _maybe_wrap_cuda_graph def (pure identity, returns opt unchanged) + 6 call sites
- KEEP: MambaModel/SelectiveSSMLayer (live, used by test_mamba_megakernel.py)
- KEEP: _maybe_checkpoint (used in Transformer.forward, ViT.forward, MambaModel.forward)
- Note: v1 Edit 2.9b told lead to apply 5 shared call sites by line number
  (v2 provides unique multi-line OLD blocks for all 6 sites)

---

## 12. datasets_v2.md — Layer-A Data Plumbing (v2, validated 2026-06-25)

### Design Intent
Same as datasets.md but fully re-validated against the live file. Provides unique
exact-match OLD→NEW blocks for ALL 6 _maybe_wrap_cuda_graph call sites.

### Applied vs On-disk Status
APPLY-READY (preferred over v1). Re-verified against live grokking_race_v2.py (2505 lines)
with updated line numbers and unique multi-line context blocks for each site.

### Key Facts
- Dead-code KEEP proof confirmed: test_mamba_megakernel.py:37-39 does
  `import grokking_race_v2 as g; g.MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2, ...)`
- Confirmed: evaluate() has exactly ONE real caller (_fin:815)
- Confirmed: EarlyStopper has exactly ONE caller (_stopper:691)
- Live grep: exactly 7 occurrences of _maybe_wrap_cuda_graph (def:895 + 6 calls at
  983, 1090, 1387, 1424, 1452, 1512)
- Site 6 unique anchor (train_prodigy:1510-1512): `opt=Prodigy(m.parameters(), ...)`
  above the call
- 5 of 11 train loops never called the shim (train_neuralgrok, supergrok, supergrok15,
  supergrok2, looksam) — they are untouched
- Net: -11 lines (4 def + 1 blank + 6 calls); behavior-preserving (opt=opt identity)

---

## 13. compile.md — grokking_optimizers/compile.py + dispatch.py Improvements

### Design Intent
Four parts: Bug fixes (3) and new tuning dimensions (2).
- Bug #1: inline run_device_pgo_round (drop self-alias import + swallowed ImportError)
- Bug #2: gfx942 maxrregcount sentinel (-1) must OMIT `-amdgpu-max-num-vgprs` flag
- Bug #3: ABI-schema guard in dispatch.py (_LazyOps._resolve checks __abi_schema__)
- S1.4: L2-persistence tuned dims (l2_persist, l2_hit_ratio, l2_setaside_pct)
- S1.5: smem-carveout + max-dyn-smem tuned dims (smem_carveout, max_dyn_smem_kb)
- S3.4: cross-run negative cache (bloom dedup over proven-bad config hashes)

### Applied vs On-disk Status
APPLY-READY, not yet applied. Verified against live compile.py.

### Key Facts
- Bug #2: gfx942 dim is range(32,256,4) (sentinel NOT the first value); bug surfaces
  only on a cross-arch config carrying maxrregcount=-1
- Bug #3: EXPECTED_ABI_SCHEMA=1 in dispatch.py; _LazyOps.__bool__ swallows mismatch
  so has_kernels() stays a clean predicate
- Negcache: DEFAULT OFF (enable_negcache:bool=False); only caches DETERMINISTIC rejects
  (infeasible, cost_model_pruned, numerical_fail) — NOT transient build/time fail
- _NegCacheBloom: fixed-capacity bloom (m=1<<21, k=13) with double-hashing + base64-JSON
  serialization; persists across runs in CompileCache JSON
- S1.4 L2 dims: l2_persist [False,True], l2_hit_ratio [1.0,0.75,0.5,0.25] (float!),
  l2_setaside_pct [100,75,50,25]; first value=today's behavior, auto-pinned dead until
  primitives.cuh gains #ifndef SG_TUNED_L2_* guards
- S1.5 smem dims: smem_carveout [-1,100,0,25,50,75], max_dyn_smem_kb [0,100,164,200,228];
  sentinels: -1=OMIT carveout; 0=exact sizeof
- l2_hit_ratio is the project's ONLY float-valued macro dim; resolve_macros gets
  explicit `elif isinstance(value, float): f"-D{macro}={value!r}f"` branch
- _SELF_TEST_EXPECTED_COUNT: 265 → 267 (1 Bug#2 case + 1 negcache case)
- New keep_macros in _tc_relevant_device_flags: SG_TUNED_L2_*, SG_TUNED_SMEM_CARVEOUT,
  SG_TUNED_MAX_DYN_SMEM_KB
- Gate: `python -m grokking_optimizers.compile --self-test` + `ruff check grokking_optimizers/`

---

## 14. verify.md — Per-Function Silicon Verification Driver

### Design Intent
Create a new file `tests/hw/verify_functions.py` as a driver layer over existing gates.
No edits to any existing file (AREA = "NEW files only"). Provides:
- CPU --self-check: enumerate the 33-cell L3-TC surface and assert coverage
- GPU --run: 8-GPU-shardable isolation matrix (existing run_cell_gate CLI via subprocess)
- Optional ablation half (requires external shadow _ops with SG_ABLATE seam)

### Applied vs On-disk Status
APPLY-READY (create new file). File does NOT exist in live tree.

### Key Facts
- 33 CELLS enumerated (adamw/lion/grokfast/neuralgrok/grokadamw/prodigy/muon/
  looksam/supergrok11/supergrok15/supergrok2 × decoder/vit/mamba)
- ISOLATION registry: 14 surface nodes mapped to live gate cells + 3 cpu:: units
  (cpu::routing_table, cpu::driver_edges, cpu::scalars_roundtrip)
- ABLATION registry: 11 bits (ABL_OPT_TAIL through ABL_REDUCE_ORDER)
- --self-check runs FULLY ON CPU: pinned arch=90, no GPU/extension needed
- _gate_subprocess verdict contract: rc==0 AND "=> PASS" AND "1/1 cells passed" AND no SKIP
- PENDING-SEAM verdict when shadow build absent (honest — never false green/red)
- gfx942/tpu preservation: normalize_arch(detect_arch())==90 gate; non-Hopper delegates
  to verify_all --phase 3
- Gate: `PYTHONPATH=. python tests/hw/verify_functions.py --self-check`

---

## 15. deadcode_artifacts.md — Non-Source Artifact Removal

### Design Intent
Remove 528 provably-dead committed artifacts from the live repo:
- `_dectc_codegen/` — nvcc --keep intermediate dumps (64 tracked files, 7.95M lines)
- `_scan/` — secrets/PII scan-prep manifests and grep dumps (43 tracked files)
- `_scan_prep.sh` + `_scan_prep.log` — the scan-prep generator (2 files)
- `claude_session_archive/` — Claude session tool-result dumps (419 tracked files)

### Applied vs On-disk Status
APPLY-READY removal list (analysis; not yet applied). Proven dead via reachability
search. No source/build/test references to any of the four artifact sets.

### Key Facts
- Total removal: 528 tracked files, 8,089,083 text lines (95.73% of repo text lines!)
- After removal: true source = 4.27% of current committed repo lines
- Dominant removal: _dectc_codegen/ (.ii=5,649,695 + .cpp=1,744,986 + .gpu=331,496
  + .ptx=126,272 + .c=58,072 ≈ 7.91M lines)
- MANIFEST.in/setup.py/packaging unaffected (none of these artifact sets are shipped)
- `results/` (101 tracked files): FLAGGED TO KEEP — referenced by README.md,
  HARDWARE_VALIDATION.md, tuning scripts, wiring_check.py
- Git rm commands: `git rm -r _dectc_codegen/ _scan/ claude_session_archive/`
  and `git rm _scan_prep.sh _scan_prep.log`
- Post-apply verify: grep for removed names must return no matches in *.py/*.cu/*.cuh/*.cpp

---

## 16. deadcode_source.md — Provably-Dead Production Source Code

### Design Intent
Identify and remove dead code from the TRUE SOURCE (grokking_optimizers/*.py, csrc/**,
tests/**, tuning/**, root *.py). Conservative: only removes if zero reachable callers.

### Applied vs On-disk Status
APPLY-READY. Exactly ONE dead code path found (56 lines total).

### Key Facts
- REMOVABLE: `tc_dump_outproj_operands` on the Mamba scalar-TC cell
  - C++ function body: unconditional TORCH_CHECK(false, "obsolete...") — can never return
  - Registered in pybind11 PYBIND11_MODULE
  - Sole caller: `test_tc_proj_dw_exact_on_own_operands` (test_mamba_tc.py:551)
    → always RuntimeErrors on hardware; SKIPPED on non-sm_90; not xfail-marked
  - Removal: 1A (19 lines C++ function), 1B (2 lines pybind def), 1C (35 lines test function)
  - OPTIONAL: 1D updates stale docstring bullet (prose only)
- NOT REMOVABLE:
  - MambaModel/SelectiveSSMLayer — LIVE (test_mamba_megakernel.py:37-39)
  - _maybe_wrap_cuda_graph — does NOT exist in current tree (grep returns 0)
    (owned by datasets_v2.md, not this AREA, to avoid double-removal)
  - Dead generated mega_<model>_<opt>.cu cells — NOT present on disk (already deleted)
  - launch_<opt>.cu / models/*.cu shims — absent on disk; references are in LIVE verify_all.py
- Siblings decoder/vit tc_dump_ff2_operands are REAL working functions — KEEP
- Remaining mamba TC gates stay: test_tc_grad_parity*/keystone, test_tc_determinism,
  test_tc_short_trajectory, test_tc_step_time_vs_scalar

---

## Cross-Cutting Summary

### Spec Status Overview
| Spec | Status | Risk Level |
|------|--------|------------|
| cute.md | Apply-ready, NOT applied | Low (byte-identical default) |
| tma.md | Apply-ready (partial: sections 2-4), NOT applied | Low |
| tma_wire.md | Apply-ready, NOT applied | Medium (7+ files, ABI) |
| flagship.md | Apply-ready, NOT applied | Low (codegen only) |
| flagship_dw.md | Apply-ready, NOT applied | Low (byte-identical at L=2) |
| profiler.md | Kernel profiler ON-DISK; new .py file apply-ready | Low |
| mamba_flagship.md | Apply-ready, NOT applied | Medium (n_heads bug, smem) |
| mamba_smem_redesign.md | Apply-ready spec; NOT applied | HIGH (smem redesign) |
| vit_flagship.md | Apply-ready, NOT applied | Low (byte-identical at L=2) |
| vit_forkb.md | TC kernel already has Fork-B; 1 small edit remaining | Low |
| datasets.md | Apply-ready (v1); use v2 | Low (byte-identical default) |
| datasets_v2.md | Apply-ready (v2 validated), NOT applied | Low |
| compile.md | Apply-ready, NOT applied | Low (gated/pinned defaults) |
| verify.md | Apply-ready (new file only), NOT applied | Low (no existing edits) |
| deadcode_artifacts.md | Apply-ready removal list, NOT applied | Low (artifacts only) |
| deadcode_source.md | Apply-ready (56 lines), NOT applied | Low (pure dead code) |

### Byte-Identity Guarantees
Every spec honors the project's core invariant: new knobs must be byte-identical at
default values. Key mechanisms:
- SG_TUNED_GEMM_ENGINE=0 → #if-erases all CuTe code
- SG_TUNED_GEMM_TMA=0 → SG_DEC_TMA_* macros collapse to existing code
- data_source="modular" → literal original 4 lines in make_data_for_task
- enable_negcache=False → zero probe/harvest calls, pre-knob code path
- L2 dims first value=today's behavior, auto-pinned until kernel headers land
- smem_carveout=-1 sentinel → launcher SKIPS cudaFuncSetAttribute
- All L-generalized formulas proven byte-identical at L=2

### Critical Blockers
1. Mamba-3 flagship launch impossible: 19.56 MB dynamic smem >> 227 KB cap
   → requires full mamba_smem_redesign.md implementation (Level B target ~120.85 KB)
2. Decoder L=2-pinned consumer kernel: until flagship_dw.md lands, the consumer
   processes only 2 of 48 layers even with the flagship layout header
3. TMA not yet wired: cute_tma_desc.h absent, SG_TUNED_GEMM_TMA not in wgmma.cuh

### Gate Command Summary
- CuTe engine: `nvcc wgmma_selftest.cu -DSG_TUNED_GEMM_ENGINE=1`
- Decoder flagship: `python -m grokking_optimizers.megakernel_codegen --decoder-layout-flagship`
- Mamba flagship: `python -m grokking_optimizers.megakernel_codegen --mamba-layout-flagship`
- ViT flagship: `python -m grokking_optimizers.megakernel_codegen --vit-layout-flagship`
- Compile self-test: `python -m grokking_optimizers.compile --self-test`
- Verify self-check: `PYTHONPATH=. python tests/hw/verify_functions.py --self-check`
- Mod-97 smoke: `python grokking_race_v2.py --gpus 2 --optimizers adamw --num-seeds 1 --early-stop-max-steps 300 --no-status-server --output /tmp/smoke`
