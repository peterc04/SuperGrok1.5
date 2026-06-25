# csrc/fused/sm_90 — ViT + Mamba Component Analysis

## Slice assignment
Files analyzed: mamba3_layout.cuh, mamba_flagship_layout.cuh, model_stage_mamba3.cuh,
model_stage_mamba_tc.cuh, fused_mamba_megakernel.cuh, vit_layout.cuh, vit_flagship_layout.cuh,
model_stage_vit.cuh, model_stage_vit_tc.cuh, fused_vit_megakernel.cuh,
mega_mamba_real_adamw_tc.cu, mega_mamba_real_adamw_tc_launcher.cu,
mega_vit_real_adamw_tc.cu, mega_vit_real_adamw_tc_launcher.cu,
grokking_optimizers/mamba3_block.py

All files in `/workspace/SuperGrok1.5/csrc/fused/sm_90/` (canonical HEAD).

---

## 1. Mamba smem redesign: gate `kMbStreamSmem`

### Gate mechanism (mamba3_layout.cuh:287-307, mamba_flagship_layout.cuh:323-357)

The gate is a **compile-time `constexpr bool`** derived purely from layout constants:

```
kMbAllLayersSmemFloats = LAYERS*SEQ*D + SEQ*D + LAYERS*kMbOneLayerActFloats
    + (SEQ*D + SEQ + PHEAD) + 2*SEQ*D + 3*SEQ*DINNER + 2*SEQ*DFF
    + SEQ*XPROJ + SEQ*STATEC*2*2 + SEQ*STATEC + 64

kMbStreamSmem = (kMbAllLayersSmemFloats * sizeof(float)) > (227 * 1024)
```

**Production (d=128, L=2):** kMbAllLayersSmemFloats ≈ 52,700 floats → ~210.8 KB → kMbStreamSmem = **FALSE** → old struct unchanged, byte-identical.

**Flagship (d=2048, L=24, n_heads=64):** kMbAllLayersSmemFloats = 5,128,489 floats → 20,513,956 bytes = **19.57 MB** → kMbStreamSmem = **TRUE** → streamed struct.

The "19.56MB" figure in CLAIMED state is confirmed correct (mamba_flagship_layout.cuh:312: `kMambaSmemBytes = 20513956`).

### Streamed smem size computation (VERIFIED)

When kMbStreamSmem=TRUE, the struct caches only:
- `layer_in` ring (2 deep × SEQ×D): 2*8*2048 = 32,768 floats
- one `LayerAct` **small-cache only** (no x_in/z/y_scan/h1/g_pre/u_mlp): kMbStreamOneLayerActFloats = 8 * (1+128+192+64+256+4+256+1) = **7,216 floats**
- fn_xhat[1][D] + fn_r[SEQ] + logits[PHEAD]: 2048+8+97 = 2,153 floats
- xproj[SEQ][XPROJ]: 8*576 = 4,608 floats
- dBbar/dCbar[SEQ][STATEC][2]: 8*64*4 = 2,048 floats
- dtheta[SEQ][STATEC]: 8*64 = 512 floats
- red[64]: 64 floats

Total: **49,369 floats × 4 = 197,476 bytes ≈ 192.85 KB ≈ 193 KB**

The "193KB" claim is confirmed. mamba_flagship_layout.cuh:355 static_assert confirms it fits the H100 227KB cap.

### Level B: scratch-to-HBM proxy mechanism (model_stage_mamba3.cuh:156-198)

The big SEQ×{DINNER,DFF,D} buffers become `MbHbmBuf2D<Cols>` proxy structs (8 bytes, a `float* base_`). The `MbBuf2D<Rows,Cols>` alias selects:
- **SMALL**: `float[Rows][Cols]` (real smem array)
- **STREAMED**: `MbHbmBuf2D<Cols>` (proxy, HBM-backed)

`mb_set_base(field, ptr)` is overloaded: no-op for real arrays, pointer-set for proxies. This lets the **same source text** (same `buf[s][c]` access pattern) compile for both paths.

Fields moved to HBM: `x_in`, `z`, `y_scan`, `h1`, `g_pre`, `u_mlp` (per-layer big buffers), plus transient backward buffers `final_in`, `dh`, `dr`, `dr2`, `adj_a/b/c`, `wff_a/b`.

HBM layout per CTA (model_stage_mamba3.cuh:324-343):
- Per layer: SEQ*D (layer_in) + small_cache_exact + 3*SEQ*DINNER + SEQ*D + 2*SEQ*DFF
- Transient: 4*SEQ*D + 3*SEQ*DINNER + 2*SEQ*DFF

`mb_acts_stride_floats()` returns 0 when !kMbStreamSmem → zero carve on SMALL/bench.

---

## 2. Mamba flagship dimensions (mamba_flagship_layout.cuh)

| Constant | Value | Derivation |
|----------|-------|-----------|
| D (d_model) | 2048 | flagship config |
| LAYERS | 24 | flagship config |
| n_heads | 64 | d_inner/head_dim = 4096/64 |
| d_inner | 4096 | = 2*D |
| dt_rank | 128 | max(d/16, 1) |
| Nc (STATEC) | 64 | N/2, N=128 |
| d_ff | 4096 | mlp_ratio * d |
| XPROJ | 576 | dt_rank + 2*n_heads + 5*Nc = 128+128+320 |
| kMambaNumTensors | **485** | 2 + 20*L + 3 = 2 + 480 + 3 |
| kMambaTotalElems | **1,265,411,169** | ~1.27B params |

Note: kMambaNumTensors=485 (not 45 like d=128 production). The per-layer 20-tensor block at flat 2+20*li for 24 layers.

---

## 3. ViT flagship dimensions (vit_flagship_layout.cuh)

| Constant | Value | Note |
|----------|-------|------|
| D | 1664 | flagship config |
| LAYERS | 48 | flagship config |
| HEADS | 16 | d/head_dim = 1664/104... actually D/HEADS = 104 per head |
| SEQ | 17 | 16 patches + CLS |
| DFF | 6656 | = 4*D |
| kVitNumTensors | **584** | 4 + 12*L + 4 = 4 + 576 + 4 |
| kVitTotalElems | **1,596,200,417** | ~1.60B params |

**Discrepancy vs claim**: The brief says "ViT d1664/L48" which matches. But: the ViT flagship comment says "heads=16" (vit_flagship_layout.cuh:33). At d=1664 and kDhead should be d/heads = 1664/16 = 104. This is an unusual head dimension (not a power of 2).

---

## 4. ViT Fork-B grad-partial architecture (model_stage_vit_tc.cuh)

### Four ViT deltas vs decoder TC twin (model_stage_vit_tc.cuh:33-48):
1. **Full (bidirectional) attention** — no causal mask
2. **Patch-proj Linear(49→128)** replaces token embedding; CLS token prepended
3. **CLS pos-0 head** — final-norm + head on position 0 (vs decoder's last position)
4. **kSeq=17 → kTileM=LCM(17,64)=1088** (17 stacked m64 atoms)

### Fork-B structure:
- **P1**: token-tile-parallel fwd+bwd-dX through all layers, barrier-free within tile. Writes HBM bf16 VitActs (X_in, X_ctx, X_x1, X_gact, dY_qkv, dY_a, dY_ff0, dY_ff2 per layer; X_patch, X_hn, dY_logits, dh0)
- **P2**: dW-output-stationary — each weight matrix dW tile owned by ONE CTA, contracts full K ascending-t, no float atomics → deterministic

### Tunable knobs (model_stage_vit_tc.cuh):
- `SG_TUNED_VIT_TILE_M = 1088` (LCM(17,64))
- `SG_TUNED_VIT_DW_SPLITK = 4` (split-K factor for the ~52 dW tiles)
- `SG_TUNED_VIT_GEMM_INTERLEAVE = 2` (M-atom interleave; LIFTED from decoder H1+H3 win)
- `SG_TUNED_VIT_P1_SUBTILE_S = 8` (BAKED default from 2026-06-17; 4.02x vs S=64 at d2048/B1024)

### P1 sub-tiling mechanism (model_stage_vit_tc.cuh:127-172):
Grid strides over sub-tiles of S whole samples (S=8 → kVitP1SubtileRows=136 rows) instead of 1088-row tiles. At S=8, 16 P1 tiles → 16*8=128 sub-tiles → nearly all 132 SMs active. Parity: LN-affine grads and loss are valid fp32 reassociations (gate tolerance 1e-4).

### LN vector-grad partials (model_stage_vit_tc.cuh:277-293):
`kNumLnVec = 4*L+2` dense slots per CTA (n1.w/b, n2.w/b ×L, norm.w/b). L-general formula `vit_lnvec_tensor_idx(v)` replaces the old 10-slot static table (needed for L=48 flagship).

---

## 5. What LAUNCHES vs is blocked

### Mamba
| Kernel | State | Evidence |
|--------|-------|----------|
| `fused_mamba_megakernel<Opt>` (scalar fp32) | **LIVE, validated** | SG_MB_SCALAR_MEGAKERNEL=1 default; dynamic smem opt-in implemented; 33/33 wiring gates |
| `fused_mamba_megakernel_tc<Opt,Par>` (wgmma TC) | **WIRED, but SCAN-DOMINATED scalar** | mega_mamba_real_adamw_tc_launcher.cu wired into _ops; BUT kernel body at line 548-554: "the 'TC' megakernel runs the VALIDATED scalar per-sample fwd+bwd... The Mamba-3 mixer is scan-dominated"; no wgmma for Mamba fwd/bwd |
| Flagship Mamba (d=2048, L=24) | **LAYOUT READY, NO TU** | mamba_flagship_layout.cuh exists as standalone header; no flagship-specific mega_mamba_*_flagship*.cu TU in sm_90/ |
| Mamba TP at flagship | **BLOCKED** | static_assert in model_stage_mamba_tc.cuh:362-365: `!Par::kTPComm || !kMbStreamSmem` — "kTPComm + layer-streamed smem not supported" |

**Key honesty note**: The "TC" megakernel for Mamba is NOT a wgmma-accelerated Mamba fwd/bwd. The body reads: it calls `mb_forward_sample` / `mb_backward_sample` (the scalar scan), then runs the full optimizer tail with the proper workspace layout. The wgmma machinery in model_stage_mamba_tc.cuh (MbTcSmem, kMbAtomsPerSlot, etc.) is **dormant** — leftover from Mamba-1 days, marked "dormant MbTcSmem" at line 75. The MbDwSpec array is kept (kMbNumDwSpecs=8) only for struct size stability.

### ViT
| Kernel | State | Evidence |
|--------|-------|----------|
| `fused_vit_megakernel<Opt>` (scalar fp32) | **LIVE, validated** | SG_VIT_SCALAR_MEGAKERNEL=1 default; 33/33 wiring |
| `fused_vit_megakernel_tc<Opt,Par>` | **TRUE wgmma Fork-B, WIRED** | mega_vit_real_adamw_tc_launcher.cu in _ops; genuine HGMMA for 6 linear families; 21/21 gates per comment |
| Flagship ViT (d=1664, L=48) | **LAYOUT READY, NO TU** | vit_flagship_layout.cuh exists; no flagship TU |
| ViT scalar megakernel at bench/flagship | **GATED OFF** | SG_VIT_SCALAR_MEGAKERNEL=0 at bench width (d>128 smem > 227KB) |

---

## 6. Smem budgets summary

### Production d=128

| Model | smem type | Bytes | Cap status |
|-------|-----------|-------|-----------|
| Mamba scalar (SMALL) | dynamic | 215,844 B (≈210.8KB) | < 227KB ✓ |
| Mamba TC (SMALL, same struct) | dynamic | 215,844 B | < 227KB ✓ |
| ViT scalar | dynamic | 188,080 B (≈183.7KB) | < 227KB ✓ |
| ViT TC (VitTcSmem) | static | ≈7KB (A+B tiles + red + specs) | < 48KB ✓ |
| Mamba TC (MbTcSmem) | static | ≈16.6KB | < 48KB ✓ |

### Flagship (d=2048 Mamba, d=1664 ViT)

| Model | smem type | Bytes | Cap status |
|-------|-----------|-------|-----------|
| Mamba flagship (streamed) | dynamic | 197,476 B (≈193KB) | < 227KB ✓ |
| Mamba flagship (all-layers, DEAD) | — | 20,513,956 B (19.57MB) | EXCEEDS cap |
| ViT flagship scalar (DEAD, gated OFF) | — | 2,304,784 B (≈2.25MB) | EXCEEDS cap |
| ViT flagship TC | static | ≈7KB | < 48KB ✓ |

---

## 7. Mamba TP implementation (model_stage_mamba_tc.cuh:172-604)

**Shard table** (mb_tp_split_of, line 222-232):
- in_proj (block+7): COL (split output rows = column-parallel)
- out_proj (block+15): ROW (row-parallel → forward all-reduce ①)
- gate (block+17): COL
- up (block+18): COL
- down (block+19): ROW (row-parallel → forward all-reduce ②)
- everything else (tok/pos/norm/head/x_proj/dt/B,C norms/biases/D/A_log): REPLICATED

**4 reduce points**:
- ① out_proj forward: partial publish → fixed-order ascending-PE reduce → mix_out
- ② down forward: partial publish → reduce → mlp_out
- ①' in_proj backward dX: partial → reduce → dxn (column-parallel dX)
- ②' gate+up backward dX: partial → reduce → dh1n

**SSM body is REPLICATED on every rank** (the entire selective scan including x_proj/dt internals). Head-sharded selective scan is documented as a "deep follow-up, NOT this mechanical mirror" (line 342-343).

**kTPComm lockstep contract** (line 326-343): The kTPComm P1 grid-lockstep-over-samples ensures every CTA reaches all 4 rendezvous points uniformly. An `active` flag lets empty rounds still reach every rendezvous.

**Critical blocker**: `static_assert(!Par::kTPComm || !kMbStreamSmem)` at line 362-365 — TP+flagship (streamed-smem) is explicitly blocked with: "the TP body caches all layers; stream it before TP-at-flagship."

---

## 8. ViT TP implementation (model_stage_vit_tc.cuh via parallel_config.cuh)

Analogous to Mamba TP: the TP seam uses the same tp_transport.cuh primitives (LoopbackTransport/NvshmemTransport) and par::ParConfig/par::CommCtx. The VitActs HBM buffer carries the bf16 activations used by both the scalar sample processing and the TC dW path.

---

## 9. Mamba-3 model math (mamba3_block.py + model_stage_mamba3.cuh)

The Python file is the verified reference (arXiv 2603.15569, ICLR 2026). Key design decisions:
- **No conv1d** — dropped (exponential-trapezoidal recurrence implies implicit width-2 convolution + B,C biases replace it)
- **No SiLU on SSM input** — x_main feeds SSM/projections directly
- **BCNorm**: RMSNorm on Br/Bi/Cr/Ci, then per-channel biases B_bias/Bhat_bias/C_bias/Chat_bias
- **Complex state via RoPE trick**: 2-vector per complex coordinate; rotation angle phi = dt*theta (per-head per-coord)
- **Scan recurrence** (Eq 25): h_t = alpha*(R@h_{t-1}) + beta*(R@(Bbar_{t-1}*x_{t-1})) + gamma*Bbar_t*x_t

The CUDA kernel uses the **seq=8 exploit**: each thread owns SSM channel j, holds complex state in registers, unrolls t=0..7. No smem for scan state.

**Backward**: recomputes forward keeping h_hist[seq+1][Nc][2] and v_hist[seq+1][Nc][2] in registers. Width-2 coupling: v_{t-1} feeds step t-1's gamma term AND step t's beta term.

---

## 10. Model-stage tensor structure

### MambaWeights (model_stage_mamba3.cuh:408-436)
Typed view over flat param blob:
- tok[kVocab, kD], pos[kSeq, kD]
- Per Layer: mixn_w, A_log, D, B_bias, Bhat_bias, C_bias, Chat_bias, in_w, x_proj_w, dt_proj_w, dt_proj_b, B_norm_w, Bhat_norm_w, C_norm_w, Chat_norm_w, out_w, mlpn_w, gate_w, up_w, down_w (20 per layer)
- norm_w, out_w, out_b

Named_parameters() order: tok/pos → per block (mixer_norm OWN params first, then submodules) → norm/head.

### VitWeights (model_stage_vit.cuh:151-175)
- cls[d], patch_w[d, patch], patch_b[d], pos[kSeq, d]
- Per Layer: in_w[3d,d], in_b[3d], out_w[d,d], out_b[d], n1_w/b[d], n2_w/b[d], ff0_w[4d,d], ff0_b[4d], ff2_w[d,4d], ff2_b[d] (12 per layer)
- norm_w[d], norm_b[d], out_w[kVocab,d], out_b[kVocab]

---

## 11. Optimizer integration

### Mamba (fused_mamba_megakernel.cuh)
- `mamba_rebase_state<Opt>()` rebases per-element state pointers to tensor slice
- P3 work-steals kMambaNumTensors tasks, calls `apply_optimizer<Opt>(p, gg, i, step, ts)`
- Workspace layout: `[nCTA*acts_stride | nCTA*total | loss(nCTA) | loss_out | opt_reduce(2nCTA+1) | sam_backup+sam_grad(2*total) | muon | sg2]`
- `mb_tc_workspace_floats(T, nCTA)` is the canonical sizing formula

### Muon 2D-weight table (model_stage_mamba_tc.cuh:98-153)
L-general formula `mb_muon_2d(mi)` and `mb_is_muon_2d(t)` — replaces static 17-entry table at L=2:
- kMbNumMuon2D = 2 + 7*L + 1 = 17 at L=2, 171 at L=24
- kMbMuonMaxNumel = kMambaMaxTensorNumel (layout-derived, not d=128-pinned 65536)
- kMbMuonMaxRows = 2*d_inner (in_proj rows are the widest)

### ViT Muon 2D-weight table (model_stage_vit_tc.cuh:183-230)
L-general formula `vit_muon_2d(mi)` and `vit_is_muon_2d(t)`:
- kVitNumMuon2D = 2 + 4*L + 1 = 11 at L=2, 195 at L=48

---

## 12. Kernel launch infrastructure

### Mamba scalar launcher (fused_mamba_megakernel.cuh:314-362)
Three-step dynamic smem opt-in:
1. `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, dyn_smem)`
2. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with `dyn_smem`
3. `<<<grid, block, dyn_smem, stream>>>`
- `dyn_smem = mb_tc_dyn_smem_bytes()`: kMbStreamSmem ? sizeof(MambaSampleSmem) : kMambaSmemBytes
- `assert(occ >= 1)` or returns cudaErrorLaunchOutOfResources

### Mamba TC launcher (mega_mamba_real_adamw_tc_launcher.cu)
Uses cudaMalloc scratch (MbTcLauncherScratch, process-lived, one per device). Wired into _ops. Optional NVSHMEM symmetric TP-slot heap (`tp_sym_heap`) when SG_HAS_NVSHMEM=1.

### ViT scalar launcher (fused_vit_megakernel.cuh:300-348)
Identical three-step pattern; dyn_smem = sizeof(VitSampleSmem) = 188,080 B.

### ViT TC launcher (mega_vit_real_adamw_tc_launcher.cu)
Uses cudaMalloc scratch (VitTcLauncherScratch). Also has NVSHMEM symmetric TP-slot heap when SG_HAS_NVSHMEM.

---

## 13. Build-system notes

- `mega_mamba_real_adamw_tc.cu` and `mega_vit_real_adamw_tc.cu` own their own PYBIND11_MODULE → **auto-excluded from _ops** by setup.py's content-based glob filter. These are parity gate drivers (test_mamba_tc.py, test_vit_tc.py).
- `mega_mamba_real_adamw_tc_launcher.cu` and `mega_vit_real_adamw_tc_launcher.cu` do NOT own a pybind module → **INCLUDED in _ops**. These are the production wiring points.
- Both launcher TUs define `#define SG_TUNED_GEMM_IMPL 1` in-source, so they link the wgmma branch without preprocessor flags from the build system.

---

## 14. Discrepancies and concerns

### Discrepancy 1: Mamba "TC" megakernel is misleadingly named
The `fused_mamba_megakernel_tc` (wgmma token) **does not run wgmma for Mamba forward/backward**. The scalar per-sample scan (`mb_forward_sample` / `mb_backward_sample`) is called verbatim. The wgmma infrastructure in model_stage_mamba_tc.cuh (MbTcSmem ring, kMbAtomsPerSlot, MbDwSpec) is dormant. The "TC" designates the infrastructure path (workspace layout, optimizer tail, NVSHMEM seam), not tensor-core acceleration of the scan.

### Discrepancy 2: kMbStreamSmem is FALSE for production d=128
The flagship streamed-smem redesign (Layer-B, MbHbmBuf2D) is compile-time DEAD at d=128. At d=128, kMbStreamSmem=false, so MbBuf2D<Rows,Cols> resolves to real `float[Rows][Cols]` smem arrays. The MbHbmBuf2D proxy code compiles but is never instantiated. The `mb_acts_stride_floats()` returns 0. No HBM scratch is carved for the small model.

### Discrepancy 3: ViT flagship head count / head dimension
vit_flagship_layout.cuh line 33 states `SG_VIT_HEADS = 16` at d=1664. This gives kDhead = 1664/16 = 104. This is an unusual non-power-of-2 head dimension. The wgmma attention tile (per-head softmax, score tiles) assumes dimensions compatible with the m64nNk16 atom — a 17×17 attention matrix per head is already non-standard; a 104-column value tensor is padded to 128 for wgmma. This is likely fine since the scalar path handles the attention and the TC path handles only the linear projections.

### Discrepancy 4: Flagship layouts have no launch TU
Both mamba_flagship_layout.cuh and vit_flagship_layout.cuh are standalone headers described as "included INSTEAD OF" the regular layout files. There is no `mega_mamba_flagship*.cu` or `mega_vit_flagship*.cu` in sm_90/. The flagship model CANNOT be launched without creating a new TU that includes the flagship layout header.

### Discrepancy 5: TP+Mamba+Flagship is doubly blocked
At flagship scale: (a) kMbStreamSmem=TRUE → the static_assert at model_stage_mamba_tc.cuh:362-365 blocks TP+streaming, and (b) no flagship TU exists. This means the 1.27B Mamba model cannot run with TP in the current codebase.

### Discrepancy 6: Mamba TC scalar-win carve-out
mega_mamba_real_adamw_tc_launcher.cu:24-28 admits: "mamba×adamw was excluded from _L3_WGMMA_CELLS as a measured scalar-WINS carve-out (0.46×, the selective-scan/conv1d dominate)". The "TC" megakernel is wired now per cycle-2 directive "for the roofline to report the honest 0.46×", not because it's faster.

---

## 15. Config-derivation mechanism

The smem gate (kMbStreamSmem) IS a config-derived adaptive decision — but it is **compile-time**, not runtime. The selector folds over the model dimensions (D, LAYERS, SEQ, etc.) at compile time via `constexpr bool`. At runtime, the kernel body uses `if constexpr (kMbStreamSmem)` to select the struct layout and binding paths.

This is consistent with the central design thesis's "self-specializes by deployment config via if-constexpr" claim — but the adaptation is between TWO compile-time configurations (d=128 production and d=2048 flagship), not a runtime decision. There is no runtime planner that selects the streamed vs all-layers struct.

---

## Summary table

| Component | Status | Evidence |
|-----------|--------|---------|
| Mamba d=128 scalar megakernel | DONE, VALIDATED | SG_MB_SCALAR_MEGAKERNEL=1, 33/33 gates |
| Mamba d=128 TC megakernel | WIRED, SCAN-DOMINATED | mega_mamba_real_adamw_tc_launcher.cu; no actual wgmma for scan |
| Mamba flagship layout (d=2048) | LAYOUT ONLY | mamba_flagship_layout.cuh; no TU |
| Mamba kMbStreamSmem gate | IMPLEMENTED | compile-time constexpr; 19.57MB→193KB confirmed |
| Mamba TP (SMALL scale) | IMPLEMENTED | mb_forward_sample_tp/mb_backward_sample_tp, 4 reduce points |
| Mamba TP (flagship scale) | BLOCKED | static_assert kTPComm+kMbStreamSmem |
| ViT d=128 scalar megakernel | DONE, VALIDATED | SG_VIT_SCALAR_MEGAKERNEL=1, 33/33 gates |
| ViT d=128 TC megakernel (Fork-B) | DONE, VALIDATED | mega_vit_real_adamw_tc_launcher.cu; 21/21 gates; true wgmma |
| ViT P1 sub-tiling (S=8) | BAKED DEFAULT | SG_TUNED_VIT_P1_SUBTILE_S=8, 4.02x speedup confirmed |
| ViT flagship layout (d=1664) | LAYOUT ONLY | vit_flagship_layout.cuh; no flagship TU |
| Mamba3Block Python reference | COMPLETE | mamba3_block.py; full Eq.25 scan with complex state |
