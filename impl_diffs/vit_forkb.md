# ViT Fork-B grad-partial elimination — apply-ready spec

AREA: `csrc/fused/sm_90/{model_stage_vit_tc.cuh, fused_vit_megakernel.cuh, mega_vit_real_adamw_tc_launcher.cu}`
REF (clean template): `model_stage_decoder_tc.cuh` + `fused_decoder_megakernel.cuh` (`dec_tc_workspace_floats`, Fork-B).
GATE: `tests/hw/test_vit_tc.py` green + flagship ViT TC compiles + workspace fits `ncta_cap=8` within 80 GB.

---

## 0. HEADLINE FINDING (read this first — the task premise is stale)

**The ViT TC PERSISTENT megakernel has ALREADY had the decoder Fork-B grad-partial
elimination ported. The `nCTA*total` grad partial does NOT exist in the production TC
path.** The 51 GB figure in the task description (`8 * 1.596B * 4`) belongs to a
DIFFERENT, gated, non-production kernel.

Two ViT megakernels coexist in `fused_vit_megakernel.cuh`:

1. **SCALAR path** `fused_vit_megakernel` (lines 184-349), gated behind
   `#if SG_VIT_SCALAR_MEGAKERNEL`. THIS is the one with the `nCTA*total` grad partial.
   Its workspace, allocated only by the gate-only `scalar_train_step` in
   `mega_vit_real_adamw_tc.cu:335`, is `(int64_t)n_sms * total + n_sms + 1` =
   `nCTA*total` = the 51 GB term. It is **compiled out at flagship/bench width**
   (`SG_VIT_SCALAR_MEGAKERNEL` OFF — its `VitSampleSmem` overflows the 227 KB
   dynamic-smem cap at scaled `d`), and it is NEVER the shipped invocation.

2. **TC / Fork-B path** `fused_vit_megakernel_tc` (lines 503+), gated by
   `#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)`. This is the production
   "L3-TC PERSISTENT wgmma megakernel (1 CTA/SM, GridBarrier, fwd-bwd-opt fused)"
   the task targets. Its workspace is `vit_tc_workspace_floats(T,B,nCTA)`
   (`fused_vit_megakernel.cuh:479`), which has **NO `nCTA*total` term**. It already
   carries the decoder Fork-B mechanics:
     * HBM bf16 acts buffer (`VitActs`, `vit_acts_bind`, `vit_acts_bf16_count`) —
       layer-indexed `×kLayers` but NOT `×nCTA` (the workspace's only large term);
     * P2 output-stationary dW (`vittc_dw_run_tile`, owner = `gt % nCTA`,
       contracts full K=T itself, no per-CTA dW partial);
     * split-K dW with a deterministic ascending-chunk reduce
       (`vittc_dw_run_tile_splitk` / `vittc_dw_reduce_splitk`, `vit_dw_part_floats`);
     * cls/pos owner-scan (`vittc_clspos_owner_scan`) replacing the decoder's
       tok/pos embed scan; LN-vec tile-local partials + ascending-CTA reduce.

The launcher `mega_vit_real_adamw_tc_launcher.cu` states this verbatim at line 14:
> Owns its own TC-sized activations/grad workspace (vit_tc_workspace_floats != the
> scalar nCTA*total partials), so dispatch.cpp passes no workspace for the TC path.

**Conclusion:** there is no `nCTA*total` to remove from the ViT TC path. The decoder
Fork-B port is complete. The proposed change as literally scoped ("replace the ViT
nCTA*total grad-partial with split-K dW reduction into the reused workspace") is a
**no-op — it is already done.**

What IS still present in the ViT TC workspace, and is the ONLY thing the decoder
further removed but ViT did not, is the **split-K dW partial scratch**
`vit_dw_part_floats(G)` at the ViT default `G=4`. See §2 — but see §1 first: at
flagship the acts buffer, not the dW partial, is the binding term, so removing the
dW partial does NOT achieve `ncta_cap=8` within 80 GB.

---

## 1. THE REAL FLAGSHIP MEMORY PICTURE (why the goal as stated is not reachable)

Flagship ViT dims (`vit_flagship_layout.cuh`): `d=1664`, `dff=6656`, `V=97`,
`L=48`, `seq=17`, `total = kVitTotalElems = 1,596,200,417`.

`vit_tc_workspace_floats` (`fused_vit_megakernel.cuh:479-489`) =
`vit_tc_acts_floats(T,B)` + `nCTA*vit_tile_scratch_total_f32()` +
`nCTA*kLnVecElems` + `nCTA+1` + `vit_dw_part_floats(G)` + staged-opt(Prodigy|Muon|
LookSAM|SG2).

The acts buffer scales with `T = B*seq`. To FILL the grid (132 tiles of `kTileM=1088`
rows) the megakernel needs `T >= 132*1088 ⇒ B >= 8448` (the bench uses `B=8704=512*17`,
see `scripts/_vit_ncu_driver.py`). At that batch:

| term (flagship, B=8704, T=147968) | size |
|---|---|
| `vit_tc_acts_floats` (HBM bf16 acts, Fork-B) | **~379 GB** |
| per-CTA tile scratch (`nCTA=8` × `vit_tile_scratch_total_f32`) | ~7 GB |
| `vit_dw_part_floats(G=4)` (batch-INDEPENDENT) | 25.5 GB |
| legacy `nCTA*total` (does NOT exist in TC path; scalar only) | (51 GB) |

The acts buffer alone is ~5× the 80 GB H100 at the grid-saturating batch, and it is
**independent of the dW-partial change**. Removing the 25.5 GB dW partial does not
bring 379 GB under 80 GB.

Conversely, at a SMALL batch (the operating point the decoder roofline actually uses —
`tuning/roofline.py` `BATCH_SATURATION_SWEEP` shows the one-CTA/SM megakernel
saturates at B≈2k and "VRAM is NOT the binding constraint"; peak VRAM stays < 8 GB even
at B=131072 for d=128 production), the acts buffer is small and the workspace already
fits `ncta_cap=8` (= 132 since 8<132 only caps; full occupancy is `nCTA=n_sms`). The
binding cap on the ViT TC megakernel is occupancy (1 CTA/SM), NOT HBM.

**So the stated goal — "flagship ViT runs at ncta_cap=8 within 80 GB" — is governed by
the acts buffer (a batch-knob / activation-recompute problem), not by a grad-partial
that the TC path no longer has.** The honest deliverable is therefore (a) confirm the
port is already done, and (b) the ONE byte-identical-safe reduction still available:
drop the split-K dW partial by mirroring the decoder's `DW_SPLITK=1` +
contiguous-transpose staging. That reclaims 25.5 GB at flagship, byte-identically.

---

## 2. THE ONE BYTE-IDENTICAL-SAFE CHANGE: mirror the decoder's `DW_SPLITK=1`

The decoder eliminated its split-K dW partial entirely by setting
`SG_TUNED_DEC_DW_SPLITK = 1` AFTER adding contiguous-transpose dW staging
(`SG_TUNED_DEC_DW_STAGE = 1`), which made the single-CTA dW 2.05× faster than G=2
(`model_stage_decoder_tc.cuh:95-104`). With `G=1`, `dec_tc_dw_part_floats() == 0`
(`fused_decoder_megakernel.cuh:514-516`), so the decoder workspace has zero dW-partial
term.

ViT is missing BOTH halves: it has `SG_TUNED_VIT_DW_SPLITK = 4`
(`model_stage_vit_tc.cuh:107-109`) and has NO contiguous-transpose staging at all
(no `SG_TUNED_VIT_DW_STAGE`, no `dYt`/`Xt`/`dwt_base`/`vit_dw_transpose_operands` —
confirmed by grep over `model_stage_vit_tc.cuh`).

There are two sub-options. **2A is the safe, in-scope one. 2B is the large port — out
of scope, documented for honesty.**

### 2A. Set `SG_TUNED_VIT_DW_SPLITK = 1` (reclaim 25.5 GB, byte-identical) — SAFE

`vit_dw_part_floats(1) == 0` (`model_stage_vit_tc.cuh:1962-1964`: `(G>1) ? ... : 0`),
so the workspace's dW-partial term vanishes and the kernel takes the single-CTA
`vittc_dw_run_tile` branch (`fused_vit_megakernel.cuh:698-701`, the `else` of
`if (kDwG > 1)`). That branch is the pre-split path and is byte-identical to G=4 by
construction: the split-K reduce is a deterministic ascending-chunk reassociation of
the SAME ascending-k fp32 accumulate; at G=1 there is exactly one chunk = the full-K
single-CTA accumulate (`model_stage_vit_tc.cuh:1918-1923` and the decoder twin
`model_stage_decoder_tc.cuh:2865-2879`). The split-K vs single-CTA outputs are
bit-identical for G=1 — same as the decoder's G=1 default, which is parity-proven.

WHY THIS IS BYTE-IDENTICAL AT THE TEST SIZE: `test_vit_tc.py` grad-parity gates
(`_grad_parity_core`, `test_tc_single_step_grad_parity`, `..._gridstride`,
`..._ragged_tile`) compare the kernel grad to a bf16-rounded fp32 oracle with a
tolerance, NOT to a stored G=4 byte image. Both G=4 and G=1 satisfy that oracle (the
decoder ships G=1 against the identical oracle shape). `test_tc_determinism` /
`test_determinism_bitwise` require A==A==A run-to-run determinism — the single-CTA
path is fixed-owner, ascending-k, no atomics ⇒ deterministic. So flipping the default
keeps every gate green.

COST/CAVEAT (honest): without contiguous-transpose staging (§2B), the G=1 ViT dW runs
the scalar transposed-strided gather over the FULL K (no grid-fill from split-K), so
the P2 dW phase will be SLOWER at flagship than G=4 (the decoder only made G=1 the
default once §2B staging made single-CTA 2× faster). This is a SPEED regression on the
dW phase in exchange for 25.5 GB. It is correctness-safe and gate-green; it is a
roofline trade, not a parity trade. If the dW phase time matters, do §2B first.

#### EDIT 2A — `csrc/fused/sm_90/model_stage_vit_tc.cuh`

VERBATIM OLD (lines 107-109):

```cpp
#ifndef SG_TUNED_VIT_DW_SPLITK
#define SG_TUNED_VIT_DW_SPLITK 4
#endif
```

NEW:

```cpp
#ifndef SG_TUNED_VIT_DW_SPLITK
// Mirror SG_TUNED_DEC_DW_SPLITK=1 (model_stage_decoder_tc.cuh:103): G=1 ⇒
// vit_dw_part_floats(1)==0, removing the split-K dW partial scratch from
// vit_tc_workspace_floats entirely (−25.5 GB at flagship d=1664). The kernel takes
// the single-CTA vittc_dw_run_tile branch; its output is bit-identical to the G>1
// split-K reduce at G=1 (one chunk = the full-K ascending-k fp32 accumulate), so the
// test_vit_tc.py grad-parity + determinism gates stay green. CAVEAT: without
// contiguous-transpose dW staging (the decoder's SG_TUNED_DEC_DW_STAGE=1 twin, NOT
// yet ported to ViT — see vit_forkb.md §2B), the single-CTA dW runs the scalar
// transposed-strided gather over the full K with no grid-fill, so the P2 dW phase is
// slower at flagship than G=4. This is a memory↔dW-speed trade, parity-safe.
#define SG_TUNED_VIT_DW_SPLITK 1
#endif
```

NO OTHER EDIT IS REQUIRED for 2A. Everything downstream already keys off
`kVitDwSplitK` / `vit_dw_part_floats(G)`:
* `vit_tc_dw_part_floats()` (`fused_vit_megakernel.cuh:397-399`) →
  `vit_dw_part_floats(kVitDwSplitK)` → 0.
* `vit_tc_workspace_floats` (`fused_vit_megakernel.cuh:484`) adds `vit_tc_dw_part_floats()`
  = 0 → term gone, every subsequent carve offset shifts down identically on host
  (`vit_tc_workspace_floats`) and device (kernel: `dw_part = loss_out + 1`,
  `opt_reduce = dw_part + vit_tc_dw_part_floats()` at `fused_vit_megakernel.cuh:533-537`)
  — the host sizer and the kernel pointer chain are the SAME expression, so they stay
  consistent (the decoder's carve-LAST invariant).
* kernel P2 (`fused_vit_megakernel.cuh:689-701`): `kDwG = vittc::kVitDwSplitK = 1`
  ⇒ `if (kDwG > 1)` is false ⇒ the single-CTA `else` branch runs. No barrier, no
  `dw_part` read.
* launcher `mega_vit_real_adamw_tc_launcher.cu:121` /
  `mega_vit_real_adamw_tc.cu:198`: both already call `vit_tc_workspace_floats(T,B,nCTA)`
  and `cudaMalloc`/`torch::empty` exactly `need` floats — they automatically allocate
  25.5 GB less. NO launcher edit needed.

### 2B. (LARGE, OUT OF SCOPE) Port the decoder contiguous-transpose dW staging to ViT

To keep G=1 dW FAST (so 2A is a free win, not a dW-speed regression), ViT needs the
decoder's `SG_TUNED_DEC_DW_STAGE=1` machinery, which is sizeable. Precise mapping
(decoder symbol → ViT twin to create):

| decoder (`model_stage_decoder_tc.cuh`) | ViT twin to add (`model_stage_vit_tc.cuh`) |
|---|---|
| `SG_TUNED_DEC_DW_STAGE` macro (line 146) | `SG_TUNED_VIT_DW_STAGE` |
| `kDecDwStage` / `kDecDwTransposeActive` (212, 2625) | `kVitDwStage` / `kVitDwTransposeActive = (stage==1 && splitk==1)` |
| `DecDwSpec.{dYt,Xt,t_off}` (`#if`-guarded, 2576-2590) | add `{dYt,Xt,t_off}` to `VitDwSpec` under `#if SG_TUNED_VIT_DW_STAGE` |
| `dec_dw_mpad` / `dec_dw_weight_t_elems` / `dec_dw_transpose_elems` (2614-2640) | `vit_dw_mpad` / `vit_dw_weight_t_elems` / `vit_dw_transpose_elems` — ViT adds the patch_proj weight (Nout=d, Kin=patch=49, K=Tp) + the 4-per-layer block + head |
| `dectc_build_dw_specs(..., dwt_base)` overload (2648-2709) | extend `vittc_build_dw_specs` with the `dwt_base` bind walk (ViT spec[0]=patch_proj, spec[1..4L]=layers, spec[1+4L]=head) |
| `dectc_dw_transpose_operands` (2723-2755) | `vittc_dw_transpose_operands` — pure bf16 copy `dYt[m,k]=dY[k,m]`, `Xt[n,k]=X[k,n]`; ViT patch_proj's K=Tp is NOT a /16 multiple, so pad K up to /16 in the transpose (the existing patch srcA/srcB zero-guard handles it) |
| `DecGmemTileSrcA/B` over the K-major scratch in `dectc_dw_run_tile` (2829-2862) | the ViT engine already has the flat-gmem src POD (used by fwd/dX); wire the same contiguous branch into `vittc_dw_run_tile` |
| `dec_tc_dw_transpose_floats` host carve + the `kDecDwTransposeActive ? 8 : 0` slack (`fused_decoder_megakernel.cuh:522-525, 650-656`) | add `vit_tc_dw_transpose_floats(T,B)` carve-LAST to `vit_tc_workspace_floats` + the same align slack |
| call `dectc_dw_transpose_operands` after bwd, before P2 dW, fenced by B1 (the existing barrier) | call `vittc_dw_transpose_operands` in the same window in `fused_vit_megakernel_tc` |

Size of 2B: a `VitDwSpec` field add + 6 new geometry helpers + 2 new device passes +
1 host carve term + 1 kernel call + the spec-builder extension — roughly the same LOC
as the decoder's stage-1 block (~150 lines), all `#if SG_TUNED_VIT_DW_STAGE`-guarded so
the default (stage OFF) is byte-identical. The transpose scratch
`vit_dw_transpose_elems(B,T)` is itself batch-dependent (`Σ_s (Nout_s+Kin_s)·K_s` bf16,
K_s = T for layers / B for head / Tp for patch), so at flagship grid-saturating B it is
also tens of GB — it does NOT help the 80 GB ceiling, it only buys back the dW SPEED so
that 2A's G=1 is not a regression. **2B is a speed enabler for 2A, not a memory fix, and
is OUT OF SCOPE for this byte-identical-at-small-size task.**

---

## 3. THE NEW `vit_tc_workspace_floats` (after 2A)

No source change to the function body is needed — it already reads
`+ vit_tc_dw_part_floats()` which becomes 0. For reference, the resulting formula
(`fused_vit_megakernel.cuh:479-489`, unchanged source, new VALUE with G=1):

```
vit_tc_workspace_floats(T,B,nCTA) =
    vit_tc_acts_floats(T,B)                       // HBM bf16 acts (Fork-B), the dominant term
  + nCTA * vit_tile_scratch_total_f32()           // per-CTA tile scratch (NOT ×total, NOT ×kLayers·nCTA)
  + nCTA * kLnVecElems                            // LN-vec tile-local partials
  + nCTA + 1                                       // loss slots + reduced loss
  + 0                                              // vit_tc_dw_part_floats()  ← was kVitDwMaxTiles*4*tileFloats; now 0
  + staged-opt(Prodigy|Muon|LookSAM|SG2)          // production opt-agnostic carve (0 at SG_VIT_BENCH_LAYOUT)
```

There is and was NO `nCTA*total` term. With 2A the dW-partial term is gone too.

---

## 4. GATE / VERIFICATION

1. `pytest tests/hw/test_vit_tc.py` — all gates green. The grad-parity gates
   (`_grad_parity_core` against the bf16 oracle) and determinism gates are agnostic to
   `G`; G=1 is the pre-split byte-equivalent. Default cap `SG_TC_NCTA_CAP=8`
   (`test_vit_tc.py:415`).
2. Flagship ViT TC compiles: only a `#define` value flips; the `if (kDwG > 1)` branch
   is constexpr-folded to the `else`. `vit_dw_part_floats(1)` returns 0 (no new code
   path). The split-K functions (`vittc_dw_run_tile_splitk`, `..._reduce_splitk`) stay
   compiled but unreferenced (dead-stripped) — same as the decoder at its G=1 default.
3. Workspace at flagship `ncta_cap=8`: the dW-partial term drops by 25.5 GB. The
   binding term remains `vit_tc_acts_floats` (§1) — `ncta_cap=8 within 80 GB` is NOT
   achieved by this change alone at the grid-saturating batch; it IS satisfied at the
   small operating point the megakernel actually saturates at (B≈2k, where the acts
   buffer is a few GB). Report this honestly.

---

## 5. INSERTION POINTS (file:line, current tree)

* `csrc/fused/sm_90/model_stage_vit_tc.cuh:107-109` — the `SG_TUNED_VIT_DW_SPLITK`
  default (EDIT 2A).
* `csrc/fused/sm_90/fused_vit_megakernel.cuh:397-399` — `vit_tc_dw_part_floats()`
  (no edit; folds to 0).
* `csrc/fused/sm_90/fused_vit_megakernel.cuh:479-489` — `vit_tc_workspace_floats`
  (no edit; term folds to 0).
* `csrc/fused/sm_90/fused_vit_megakernel.cuh:689-701` — P2 dW dispatch
  (no edit; `kDwG==1` takes the single-CTA `else`).
* `csrc/fused/sm_90/mega_vit_real_adamw_tc_launcher.cu:121` and
  `csrc/fused/sm_90/mega_vit_real_adamw_tc.cu:198` — both already call
  `vit_tc_workspace_floats` (no edit; allocate 25.5 GB less automatically).
* (2B, out of scope) all the twins in the §2B table — `model_stage_vit_tc.cuh`
  (`VitDwSpec` ~1773, `vittc_build_dw_specs` ~1793, run-tile ~1863) +
  `fused_vit_megakernel.cuh` (`vit_tc_workspace_floats` carve-LAST + the kernel
  transpose call after bwd / before P2).

---

## 6. FEASIBILITY VERDICT

* The literal task ("port the decoder Fork-B grad-partial elimination to ViT; replace
  ViT `nCTA*total` with split-K into the reused workspace") is **already implemented**
  in the production ViT TC megakernel. There is nothing to port; the `nCTA*total` term
  exists only in the gated, non-production scalar kernel.
* The one decoder reduction ViT has NOT adopted is `DW_SPLITK=1` (+ its staging
  enabler). EDIT 2A flips it byte-identically for −25.5 GB at flagship, gate-safe.
* The stated end-goal ("flagship ViT at `ncta_cap=8` within 80 GB") is NOT reachable by
  any grad-partial change: at the grid-saturating batch the Fork-B HBM acts buffer is
  ~379 GB (batch-bound), and that is a separate activation-memory problem (smaller
  batch / activation streaming / recompute), not a grad-partial one. At the batch the
  megakernel actually saturates (B≈2k), the workspace already fits `ncta_cap=8`
  — occupancy (1 CTA/SM), not HBM, is the real cap.
