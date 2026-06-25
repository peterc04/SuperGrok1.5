# Mamba TC megakernel — LAYER-STREAMING smem redesign

AREA: `csrc/fused/sm_90/{model_stage_mamba3.cuh, fused_mamba_megakernel.cuh, model_stage_mamba_tc.cuh}`
GATE: byte-identical at the SMALL (production d=128 / test) size — `tests/hw/test_mamba_tc.py` stays green; the redesign only changes behavior at FLAGSHIP scale (d=2048, L=24).
GOAL: make the per-sample `MambaSampleSmem` LAYER-INDEPENDENT (one layer + a small ring), mirroring the decoder's layer-independent `DecTcSmem`; move the acts that must persist across layers to the HBM workspace (DecActs-style), so the launch `dyn_smem` request drops below the H100 227 KB opt-in cap.

---

## 0. Confirmed live diagnosis (READ THIS FIRST — it corrects an existing doc)

The production TC path is `fused_mamba_megakernel_tc<Opt,Par>` (`fused_mamba_megakernel.cuh`). Two facts, both verified by reading the file:

1. The TC kernel **uses `MambaSampleSmem` as its dynamic smem**, NOT the small static `MbTcSmem`:
   - `fused_mamba_megakernel.cuh:546-547`:
     ```
     extern __shared__ char mamba_tc_smem_raw[];
     MambaSampleSmem& sm = *reinterpret_cast<MambaSampleSmem*>(mamba_tc_smem_raw);
     ```
   - `MbTcSmem` (line 419) is **declared but never instantiated** in the TC kernel body. It is dormant (Mamba-1-shaped, carried for the obsolete wgmma dW machinery). So the claim in `impl_diffs/mamba_flagship.md:660` ("the TC engine uses the small d-independent static MbTcSmem, NOT MambaSampleSmem") is **incorrect** for the scalar TC path. The TC kernel's `sm` is `MambaSampleSmem`.

2. The launcher requests `kMambaSmemBytes` of dynamic smem:
   - `fused_mamba_megakernel.cuh:1311-1313`:
     ```
     const int dyn_smem = (int)kMambaSmemBytes;
     err = cudaFuncSetAttribute((const void*)&fused_mamba_megakernel_tc<Opt, Par>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
     ```
   - and again at `<<<grid, block, dyn_smem, stream>>>` (line 1340).

`kMambaSmemBytes = sizeof(MambaSampleSmem)` (pinned by `static_assert` at `model_stage_mamba3.cuh:206` and `fused_mamba_megakernel.cuh:138`). At flagship (`mamba_flagship_layout.cuh`: d=2048, L=24, d_inner=4096, n_heads=64, dt_rank=128, d_ff=4096) `sizeof(MambaSampleSmem) = 20,513,956 B ≈ 19.564 MB` → `cudaFuncSetAttribute` returns `cudaErrorInvalidValue` → `launch_fused_mamba_megakernel_tc` returns before launch → **UNLAUNCHABLE**. This matches the prompt's number exactly.

Note: `fused_mamba_megakernel_tc` is compiled under `SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA` (line 96) and is INDEPENDENT of `SG_MB_SCALAR_MEGAKERNEL` (which gates only the *legacy fp32* `fused_mamba_megakernel`, lines 187-355). So gating the legacy scalar kernel OFF does **not** rescue the TC path — the TC path is the live consumer.

---

## 1. Exact smem accounting (verified field-by-field against `MambaSampleSmem`)

`MambaSampleSmem` (model_stage_mamba3.cuh:142-202) in floats. `SEQ=8`, `PHEAD=97` everywhere.

Per-config dims:

| dim     | prod d=128 | flagship d=2048 |
|---------|-----------:|----------------:|
| D       | 128        | 2048            |
| L       | 2          | 24              |
| DINNER  | 256        | 4096            |
| NHEADS  | 4          | 64              |
| DTRANK  | 8          | 128             |
| STATEC  | 64         | 64              |
| XPROJ   | 336        | 576             |
| DFF     | 256        | 4096            |

One `LayerAct` (floats) = `SEQ*(1 + 2*DINNER + DTRANK + 3*NHEADS + STATEC + 4*STATEC + 4 + 4*STATEC + DINNER + D + 1 + 2*DFF)`
(fields: mixn_r, x_in, z, dt_lr, dt_pre/A_mod/u_lam, theta, Br/Bi/Cr/Ci, 4 rms recips, Bbar+Cbar, y_scan, h1, mlpn_r, g_pre+u_mlp).
- prod: **16,080 floats** (64,320 B)
- flagship: **187,440 floats** (732.19 KB)

Total `sizeof(MambaSampleSmem)` floats = `L*SEQ*D (layer_in) + SEQ*D (final_in) + L*LayerAct + (SEQ*D + SEQ + PHEAD) (fn_xhat/fn_r/logits) + 2*SEQ*D (dh,dr) + 3*SEQ*DINNER (adj_a/b/c) + 2*SEQ*DFF (wff_a/b) + SEQ*XPROJ (xproj) + SEQ*STATEC*2*2 (dBbar,dCbar) + SEQ*STATEC (dtheta) + 64 (red)`:

- **prod: 53,961 floats = 215,844 B = 210.79 KB** — matches `kMambaSmemFloats = 53961` (mamba3_layout.cuh:270). Under the 227 KB cap → test green today.
- **flagship: 5,128,489 floats = 20,513,956 B = 19.564 MB** — 88× over cap → unlaunchable.

The `×L` terms are `layer_in[L]` (L·SEQ·D) and `act[L]` (L·LayerAct). At flagship those two are `393,216 + 4,498,560 = 4,891,776` floats = **95.4 %** of the 5,128,489 total. Removing the `×L` (one layer + ring) is the dominant win.

### One-layer-streamed footprint (this is the honesty pivot)

Replace `act[L] → act[1]` and `layer_in[L] → layer_in_ring[kRing]` with `kRing=2`:

- **prod one-layer: 37,881 floats = 151,524 B = 147.97 KB** (well under cap; SMALL not affected because SMALL is gated to the OLD struct — see §3).
- **flagship one-layer: 456,921 floats = 1,827,684 B = 1784.85 KB** — **STILL ~7.86× over the 227 KB cap.**

Why: at d=2048 the **per-sample working set itself** is large. Each `SEQ×DINNER`/`SEQ×DFF`/`SEQ×D` buffer is 64–128 KB. Even with L=1 there are ~13 such buffers:
`layer_in(ring2) 128KB, la.x_in 128, la.z 128, la.y_scan 128, la.g_pre 128, la.u_mlp 128, adj_a 128, adj_b 128, adj_c 128, wff_a 128, wff_b 128` (= 1408 KB) plus `final_in 64, la.h1 64, dh 64, dr 64, fn_xhat 64` and the small scan caches.

**Honest conclusion:** the redesign **as scoped (one-layer + small ring)** is **NOT sufficient to launch at flagship** by itself. It is correct, byte-identical-safe at SMALL, and removes 95 % of the smem, but flagship needs the **big `SEQ×{DINNER,DFF,D}` scratch buffers also moved to HBM (or tiled over d_inner/d_ff)** to clear 227 KB. This spec gives BOTH levels:

- **Level A (this spec is exact on):** layer-independent struct + ring + per-layer acts in HBM (the prompt's core ask). Byte-identical at SMALL; flagship `dyn_smem` drops from 19.56 MB to 1.74 MB but does **not** yet fit.
- **Level B (structure + exact formulas, not full verbatim body):** additionally stream the big `SEQ×{DINNER,DFF}` GEMM scratch through HBM so the smem-resident set is the ~120.85 KB of small scan/Nc/head caches + ring + reductions (computed below), which fits 227 KB. This is the deeper rewrite the prompt anticipates ("if a full streamed rewrite is too large for an exact spec, give the precise STRUCTURE + the exact smem/workspace formulas + the gated insertion points").

If big `SEQ×{DINNER,DFF,D}` scratch (`x_in,z,y_scan,g_pre,u_mlp,adj_a,adj_b,adj_c,wff_a,wff_b,h1,dh,dr,final_in,layer_in`) goes to HBM, the smem-resident remainder is:
`la.dt_lr 4 + la.dt/A/u 6 + la.theta 2 + la.B/C raw 8 + la.Bbar/Cbar 8 + la.misc_r 0.19 + fn caches 64.41 + xproj 18 + dBbar 4 + dCbar 4 + dtheta 2 + red 0.25` ≈ **120.85 KB** → fits the 227 KB cap. (`fn_xhat` at 64 KB is itself over-allocated — only the last position is used; an honest Level-B sub-opt is to shrink it to `[1][D]`, dropping ~56 KB more, but that touches the head/final-norm fwd+bwd and is out of the byte-identical-at-small minimal change.)

---

## 2. The decoder template we are mirroring (why it stays layer-independent)

`DecTcSmem` (fused_decoder_megakernel.cuh:357-411): holds ONLY `sA`/`sB` ring tiles, `red[256]`, `spec[]` — **no `[kLayers]` arrays**. All per-layer activations live in HBM in `DecActs` (model_stage_decoder_tc.cuh:425-463), `[li]`-indexed, carved from the FRONT of the same workspace the host already allocates, bound at runtime by `dec_acts_bind(p, T, B)` from a `__nv_bfloat16*` base with a running `off` cursor. The decoder forward writes each layer's needed inputs/adjoints to `DecActs.X_*[li]` / `dY_*[li]`; the backward reads them back per layer. `DecTcSmem` is sized by `ring × tile`, NOT `× kLayers`.

We mirror this for Mamba: `MambaSampleSmem` becomes one-`LayerAct` + a `kRing`-deep `layer_in` ring; the cross-layer acts move to a new `MbActsHbm` region carved from `tok.workspace`, `[li]`-indexed and bound by `mb_acts_bind`. The Mamba unit of work is a SAMPLE (one CTA per sample), so the HBM region is per-CTA (`nCTA` copies), not per-T — see the workspace formula in §6.

---

## 3. The byte-identical-at-SMALL gate (the crux of the HARD GATE)

The flagship layout is a **standalone header** (`mamba_flagship_layout.cuh`) included *instead of* `mamba3_layout.cuh`, with byte-identical symbol names; it sets `SG_MB_D=2048`, `SG_MB_LAYERS=24`, … as compile-time constants. So we can pick the struct at **compile time** with a predicate that is FALSE for prod/bench and TRUE for flagship. The cleanest predicate is "does the all-layers struct fit under the opt-in cap": define it from the existing dims so the SMALL TU literally compiles the OLD struct (→ byte-identical PTX, identical `sizeof`, identical launcher cert).

Add to `mamba3_layout.cuh` AND `mamba_flagship_layout.cuh` (both, since flagship is standalone), in `namespace sg::fused::sm90`, after the dims:

```
// Per-CTA all-layers MambaSampleSmem float count, computed from the layout dims.
// This is the SAME field formula model_stage_mamba3.cuh's struct realizes; it is
// used ONLY to pick the streamed vs all-layers struct at compile time. If it ever
// drifts from sizeof(MambaSampleSmem) the existing static_assert (==kMambaSmemBytes)
// still guards correctness — this constant only drives the gate.
constexpr int64_t kMbOneLayerActFloats =
    (int64_t)SG_MB_SEQ * (1 + 2*SG_MB_DINNER + SG_MB_DTRANK + 3*SG_MB_NHEADS
        + SG_MB_STATEC + 4*SG_MB_STATEC + 4 + 4*SG_MB_STATEC
        + SG_MB_DINNER + SG_MB_D + 1 + 2*SG_MB_DFF);
constexpr int64_t kMbAllLayersSmemFloats =
    (int64_t)SG_MB_LAYERS*SG_MB_SEQ*SG_MB_D + SG_MB_SEQ*SG_MB_D
    + (int64_t)SG_MB_LAYERS*kMbOneLayerActFloats
    + (SG_MB_SEQ*SG_MB_D + SG_MB_SEQ + SG_MB_PHEAD)
    + 2*SG_MB_SEQ*SG_MB_D
    + 3*SG_MB_SEQ*SG_MB_DINNER + 2*SG_MB_SEQ*SG_MB_DFF
    + SG_MB_SEQ*SG_MB_XPROJ
    + SG_MB_SEQ*SG_MB_STATEC*2*2 + SG_MB_SEQ*SG_MB_STATEC + 64;
// TRUE only when the all-layers struct does NOT fit the H100 227KB opt-in cap.
// prod (210.79KB) / bench: FALSE → OLD struct, byte-identical. flagship (19.56MB): TRUE.
constexpr bool kMbStreamSmem =
    (kMbAllLayersSmemFloats * (int64_t)sizeof(float)) > (227 * 1024);
constexpr int  kMbActsRing = 2;   // layer_in ring depth (streamed path only)
```

`kMbStreamSmem` is `false` at d=128 and d=1024 (210.79 KB / smaller fits), `true` at flagship. Every NEW code path below is `if constexpr (kMbStreamSmem)` / `#if`-gated so the SMALL TU is textually unchanged.

---

## 4. EDIT 1 — the gated `MambaSampleSmem` (one-layer + ring)

FILE: `csrc/fused/sm_90/model_stage_mamba3.cuh`

The OLD struct stays VERBATIM as the `!kMbStreamSmem` branch (so SMALL gets bit-identical PTX). The streamed branch drops `act[kLayers] → act[1]` (still `act[mb::kLayers]` typed but `kLayers→1` via a layer-count alias) and `layer_in[kLayers] → layer_in[kMbActsRing]`.

The minimal, lowest-risk way to keep the body code (which references `sm->act[li]`, `sm->layer_in[li]`) unchanged at SMALL while shrinking at flagship is to introduce a **smem layer-extent alias** and a **ring index helper**, then index with them. At SMALL the alias == `kLayers` and ring == `kLayers`, so every existing `sm->act[li]` / `sm->layer_in[li]` access is unchanged (`li` already in `[0,kLayers)`). At flagship the alias == 1 and the body uses `sm->act[mb_smem_li(li)]` (always 0) and `sm->layer_in[mb_ring(li)]`.

### VERBATIM OLD (model_stage_mamba3.cuh:142-208)

```
struct MambaSampleSmem {
    // Cross-block residual stream: the INPUT to each block (= residual x), and
    // the final-block output feeding the head norm.
    float layer_in[mb::kLayers][mb::kSeq][mb::kD];   // block input (residual)
    float final_in[mb::kSeq][mb::kD];                // final-block output -> head
    // Per-block cached forward activations (both blocks): the values the backward
    // reads. The block-level "h1" (= x + mixer_out) is cached so mlp_norm's input
    // and the mixer-residual reconstruction are available in the backward.
    struct LayerAct {
```
...
```
        float g_pre[mb::kSeq][mb::kDff];       // gate_proj out (pre-SiLU)
        float u_mlp[mb::kSeq][mb::kDff];       // up_proj out
    } act[mb::kLayers];
```
...
```
    float red[64];
};
// SAFETY: the launcher opts into kMambaSmemBytes of DYNAMIC smem (mamba3_layout.cuh).
// PIN the layout constant to the actual struct here so a field added without
// updating kMambaSmemFloats fails the BUILD (vs. silently under-allocating).
static_assert((int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "model_stage_mamba3: sizeof(MambaSampleSmem) drifted from "
              "kMambaSmemBytes (mamba3_layout.cuh). Update kMambaSmemFloats.");
```

### NEW (replace the two `[mb::kLayers]` array extents + the static_assert; keep LayerAct body verbatim)

Introduce, ABOVE the struct:

```
// ── Layer-streaming smem extent (decoder DecTcSmem mirror). On the SMALL/bench
//    path (kMbStreamSmem==false) the per-sample smem caches ALL layers exactly as
//    before → byte-identical. On the flagship path it caches ONE layer + a ring of
//    block-inputs; the cross-layer acts persist in the HBM MbActsHbm workspace
//    (mb_acts_bind), exactly as the decoder keeps per-layer acts in DecActs. ──
constexpr int kMbSmemLayers = kMbStreamSmem ? 1            : mb::kLayers;
constexpr int kMbLayerInRing = kMbStreamSmem ? kMbActsRing : mb::kLayers;
// Map a model layer index li∈[0,kLayers) to its smem LayerAct slot / layer_in ring
// slot. SMALL: identity (li). Flagship: act always slot 0; layer_in ring = li%ring.
__device__ __forceinline__ int mb_smem_la(int li) { return kMbStreamSmem ? 0 : li; }
__device__ __forceinline__ int mb_ring(int li)    { return kMbStreamSmem ? (li % kMbActsRing) : li; }
```

Then change ONLY the two array extents and the static_assert in the struct:

```
struct MambaSampleSmem {
    float layer_in[kMbLayerInRing][mb::kSeq][mb::kD];   // block input (residual); ring on the streamed path
    float final_in[mb::kSeq][mb::kD];                // final-block output -> head
    struct LayerAct {
        ... (BODY UNCHANGED — verbatim) ...
        float u_mlp[mb::kSeq][mb::kDff];   // up_proj out
    } act[kMbSmemLayers];                 // one layer on the streamed path; all layers on SMALL
    ... (fn_xhat .. red UNCHANGED) ...
    float red[64];
};
// SMALL/bench (kMbStreamSmem==false): sizeof == kMambaSmemBytes EXACTLY (the field
// formula is unchanged → byte-identical to the shipped struct). On the streamed
// flagship path the launcher requests kMbStreamSmemBytes (mamba3_layout.cuh), not
// kMambaSmemBytes, so the pin is conditional.
static_assert(kMbStreamSmem || (int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "model_stage_mamba3: sizeof(MambaSampleSmem) drifted from "
              "kMambaSmemBytes (mamba3_layout.cuh). Update kMambaSmemFloats.");
static_assert(!kMbStreamSmem || (int64_t)sizeof(MambaSampleSmem) == kMbStreamSmemBytes,
              "model_stage_mamba3: streamed sizeof(MambaSampleSmem) drifted from "
              "kMbStreamSmemBytes (mamba3_layout.cuh).");
```

> Byte-identity proof at SMALL: `kMbStreamSmem==false` ⇒ `kMbSmemLayers==mb::kLayers`, `kMbLayerInRing==mb::kLayers`, `mb_smem_la(li)==li`, `mb_ring(li)==li`. The struct's array extents are textually `mb::kLayers` after constant folding, so the struct, its `sizeof`, and every body access are bit-identical to the shipped code. The first static_assert is the original; the second is dead (`!kMbStreamSmem`→true short-circuits). `kMbStreamSmemBytes` must exist in BOTH layout headers (see §6).

---

## 5. EDIT 2 — the cross-layer HBM acts region (`MbActsHbm`, DecActs mirror)

FILE: `csrc/fused/sm_90/model_stage_mamba_tc.cuh` (alongside the other Mamba-TC workspace helpers in `namespace mbtc`), OR `model_stage_mamba3.cuh` next to `MambaWeights`. Place it where `mb::` dims + `MambaSampleSmem::LayerAct` are visible.

On the streamed path the **backward recomputes per layer from the cached forward acts**, so the HBM region must hold, per layer, everything the backward reads that the one-layer smem can no longer keep across the layer loop:

- `layer_in[li]` (the block residual input — needed both as the next-layer producer AND by the mixer/mlp-norm raw-x backward) — `SEQ*D` per layer.
- the full `LayerAct` per layer (the backward's `a = &sm->act[li]` source) — `kMbOneLayerActFloats` per layer.

`final_in`, `fn_xhat`, `fn_r`, `logits`, and the `dh/dr/adj_*/wff_*/xproj/dBbar/dCbar/dtheta/red` scratch stay in smem (they are NOT cross-layer — they are consumed within a single layer's fwd or bwd step, or are the head/final tail). So Level A keeps those in smem (→ the 1784.85 KB flagship footprint; see §1 honesty note). Level B additionally streams the big scratch (see §7).

The acts persist in **bf16** (DecActs is bf16) to halve HBM traffic — BUT the Mamba scan/coefficient backward is matched to the fp64 oracle to ~2e-6 and reads these acts in fp32 math. Storing the *cross-layer cache* in bf16 would perturb the flagship numerics (acceptable — flagship has no parity oracle), but to keep the SAME device code reading fp32 we store the acts in **fp32** here. (bf16 is a later memory-bound optimization, parallel to the decoder Fork-B bf16 acts; not required to launch.)

```
// ── HBM cross-layer acts (decoder DecActs mirror), streamed path only. Per CTA
//    (one sample at a time), per model layer li: the block input layer_in[li] and
//    the full LayerAct the backward replays. Carved from tok.workspace (see
//    mb_tc_workspace_floats). fp32 (the scan bwd reads fp32; bf16 is a later mem opt). ──
struct MbActsHbm {
    float* layer_in;   // [kLayers][SEQ*D]                 block residual input per layer
    float* act;        // [kLayers][kMbOneLayerActFloats]  full LayerAct per layer (POD-copied)
};
// Per-CTA float stride of the streamed acts (0 on the non-streamed path → no carve).
__host__ __device__ __forceinline__ int64_t mb_acts_stride_floats() {
    if (!kMbStreamSmem) return 0;
    return (int64_t)mb::kLayers * ((int64_t)mb::kSeq * mb::kD + kMbOneLayerActFloats);
}
// Bind this CTA's slice (base = the acts region front + cta*stride).
__device__ __forceinline__ MbActsHbm mb_acts_bind(float* base_cta) {
    MbActsHbm a;
    a.layer_in = base_cta;
    a.act      = base_cta + (int64_t)mb::kLayers * mb::kSeq * mb::kD;
    return a;
}
// Accessors: the li-th layer's layer_in row / LayerAct.
__device__ __forceinline__ float* mb_acts_layer_in(const MbActsHbm& a, int li) {
    return a.layer_in + (int64_t)li * mb::kSeq * mb::kD;
}
__device__ __forceinline__ MambaSampleSmem::LayerAct* mb_acts_act(const MbActsHbm& a, int li) {
    return reinterpret_cast<MambaSampleSmem::LayerAct*>(
        a.act + (int64_t)li * kMbOneLayerActFloats);
}
```

> `kMbOneLayerActFloats` must equal `sizeof(MambaSampleSmem::LayerAct)/sizeof(float)`; add `static_assert((int64_t)sizeof(MambaSampleSmem::LayerAct) == kMbOneLayerActFloats*(int64_t)sizeof(float), ...)` next to the struct so a field add to `LayerAct` fails the build instead of corrupting the HBM stride.

---

## 6. EDIT 3 — layout-header smem byte constant + workspace formula

### 6a. `mamba3_layout.cuh` AND `mamba_flagship_layout.cuh`

Add the streamed-smem byte constant beside `kMambaSmemFloats`/`kMambaSmemBytes`. It is the one-layer footprint = `kMbAllLayersSmemFloats - (L-1)*(SEQ*D + kMbOneLayerActFloats) - (kLayers-kMbActsRing)*SEQ*D` — or, cleaner, recompute directly:

```
// Streamed (flagship) per-CTA MambaSampleSmem float count: one LayerAct + a
// kMbActsRing-deep layer_in ring (the rest of the struct is layer-independent).
constexpr int64_t kMbStreamSmemFloats =
    (int64_t)kMbActsRing*SG_MB_SEQ*SG_MB_D + SG_MB_SEQ*SG_MB_D   // layer_in(ring) + final_in
    + kMbOneLayerActFloats                                        // one LayerAct
    + (SG_MB_SEQ*SG_MB_D + SG_MB_SEQ + SG_MB_PHEAD)               // fn_xhat,fn_r,logits
    + 2*SG_MB_SEQ*SG_MB_D                                          // dh,dr
    + 3*SG_MB_SEQ*SG_MB_DINNER + 2*SG_MB_SEQ*SG_MB_DFF            // adj_a/b/c, wff_a/b
    + SG_MB_SEQ*SG_MB_XPROJ                                        // xproj
    + SG_MB_SEQ*SG_MB_STATEC*2*2 + SG_MB_SEQ*SG_MB_STATEC + 64;   // dBbar,dCbar,dtheta,red
constexpr int64_t kMbStreamSmemBytes = kMbStreamSmemFloats * (int64_t)sizeof(float);
```

Flagship value: `kMbStreamSmemFloats = 456,921` → `kMbStreamSmemBytes = 1,827,684 B = 1784.85 KB`. (Define on BOTH headers, since flagship is standalone.) On the prod/bench headers this constant is computed but unused (`kMbStreamSmem==false`).

> HONEST FLAG: 1784.85 KB > 227 KB. Do NOT add a `static_assert(kMbStreamSmemBytes <= 227*1024)` until Level B (§7) is applied — it would fail the flagship build. Level A reduces the request 11× (19.56 MB → 1.74 MB) and removes the `cudaErrorInvalidValue` *cause* (the all-layers `×L`), but `cudaFuncSetAttribute` will still reject 1.74 MB. **Level A alone does NOT make the flagship launchable.** Level B does.

### 6b. workspace formula — `fused_mamba_megakernel.cuh:510-527` (`mb_tc_workspace_floats`)

Add the per-CTA acts region (zero-width on the non-streamed path → SMALL workspace byte-identical).

VERBATIM OLD (510-527):
```
__host__ __device__ __forceinline__ int64_t mb_tc_workspace_floats(int T, int nCTA) {
    (void)T;
    // MAMBA-3 scalar design: the per-CTA partial is the FULL grad [total] (not the
    // acts + tile-scratch + dW-split + non-GEMM-partial of the old wgmma Fork-B).
    // Layout order MUST match the kernel's pointer derivations:
    //   [nCTA*total | loss(nCTA) | loss_out(1) | opt_reduce(2nCTA+1) |
    //    sam_backup+sam_grad(2*total) | muon | sg2 (| profiler)].
    return (int64_t)nCTA * kMambaTotalElems          // per-CTA FULL grad partial
         + nCTA + 1                                  // loss slots + reduced loss
         + mb_tc_opt_reduce_floats(nCTA)             // STAGED-opt (Prodigy) reduce slots
         + mb_tc_looksam_floats()                    // SAM 2nd-bwd scratch [sam_backup|sam_grad]
         + mb_tc_muon_floats(nCTA)                   // Muon NS per-matrix scratch
         + mb_tc_sg2_floats(nCTA)                    // SuperGrok2 meta-net per-CTA scratch (carve-LAST)
#if SG_MB_TC_PROFILE
         + (int64_t)nCTA * SG_MBTC_PROF_SLOTS * 2 + 2  // phase-profiler (doubles=2 floats) + align pad
#endif
         ;
}
```

NEW — add the acts term **at the FRONT** of the carve (mirror DecActs which carves from the workspace front), so every existing region's relative offset is preserved on the non-streamed path (term is `0` there → no shift → SMALL byte-identical):
```
__host__ __device__ __forceinline__ int64_t mb_tc_workspace_floats(int T, int nCTA) {
    (void)T;
    // STREAMED (flagship) acts region (decoder DecActs mirror), carved FRONT.
    // Zero on the SMALL/bench path (mb_acts_stride_floats()==0) → byte-identical.
    return (int64_t)nCTA * mbtc::mb_acts_stride_floats()   // [nCTA] per-CTA cross-layer acts
         + (int64_t)nCTA * kMambaTotalElems          // per-CTA FULL grad partial
         + nCTA + 1
         + mb_tc_opt_reduce_floats(nCTA)
         + mb_tc_looksam_floats()
         + mb_tc_muon_floats(nCTA)
         + mb_tc_sg2_floats(nCTA)
#if SG_MB_TC_PROFILE
         + (int64_t)nCTA * SG_MBTC_PROF_SLOTS * 2 + 2
#endif
         ;
}
```

And in the kernel's workspace partition (`fused_mamba_megakernel.cuh:565-611`), carve the acts base FIRST and shift `part_base` past it:

VERBATIM OLD (565-568):
```
    float* ws = tok.workspace;
    const int64_t total_p = kMambaTotalElems;
    float* part_base = ws;                                  // [nCTA * total]
    float* loss_part = part_base + (int64_t)nCTA * total_p; // [nCTA]
```
NEW:
```
    float* ws = tok.workspace;
    const int64_t total_p = kMambaTotalElems;
    // STREAMED acts region (front); zero-width on SMALL/bench → part_base == ws.
    float* acts_base = ws;                                   // [nCTA * mb_acts_stride_floats()]
    float* part_base = acts_base + (int64_t)nCTA * mbtc::mb_acts_stride_floats();
    float* loss_part = part_base + (int64_t)nCTA * total_p; // [nCTA]
```
and bind this CTA's acts after `my_part` (line 611):
```
    float* my_part = part_base + (int64_t)cta * total_p;
    MbActsHbm acts = mbtc::mb_acts_bind(acts_base + (int64_t)cta * mbtc::mb_acts_stride_floats());
    (void)acts;   // referenced only on the streamed fwd/bwd (if constexpr kMbStreamSmem)
```

> SMALL byte-identity: `mb_acts_stride_floats()==0` ⇒ `acts_base==ws`, `part_base==ws`, every downstream offset unchanged → the SMALL workspace, the P0 zero loop, P2 reduce, and all optimizer-tail carves are bit-identical.

---

## 7. EDIT 4 — the per-layer fwd/bwd loop restructure (the streaming)

FILE: `csrc/fused/sm_90/model_stage_mamba3.cuh`, `mb_forward_sample` (908-977) and `mb_backward_sample` (1218-1328). These are the device fns the TC kernel calls (`fused_mamba_megakernel.cuh:637-638`).

The restructure must take `MbActsHbm acts` (added param), and on `if constexpr (kMbStreamSmem)` spill/refill the one-layer smem to/from HBM at the layer boundary. On `!kMbStreamSmem` the body is the SHIPPED code verbatim (the new param is unused). Because `MambaSampleSmem::LayerAct` is POD, the spill is a flat copy.

### 7a. Add the acts param (both fns + their `_tp` wrappers + the two call sites)

`mb_forward_sample(const MambaWeights& w, const int* tokens_s, int target, MambaSampleSmem* sm)`
→ `mb_forward_sample(const MambaWeights& w, const int* tokens_s, int target, MambaSampleSmem* sm, mbtc::MbActsHbm acts = {})`
(default `{}` keeps the SMALL call sites and the `_tp` SingleGPU forwarders unchanged; the kernel passes `acts` explicitly.) Same for `mb_backward_sample`. Update the two call sites in `fused_mamba_megakernel.cuh:637-638` to pass `acts`, and the `_tp` SingleGPU path (`model_stage_mamba_tc.cuh:345, 464`) to forward `acts` (default `{}` already covers the existing signature, but add the explicit forward for the streamed flagship-TP case).

### 7b. forward layer loop (mb_forward_sample:918-946)

VERBATIM OLD (918-946):
```
    for (int li = 0; li < mb::kLayers; ++li) {
        const MambaWeights::Layer& L = w.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        float* hin = &sm->layer_in[li][0][0];   // block input (residual)
        // --- mixer sub-block: h1 = hin + mixer(RMSNorm_mix(hin)) ---
        // xhat written to throwaway scratch (adj_c) — only the recip is cached; the
        // backward recomputes xhat from the raw block input layer_in[li].
        mb_rmsnorm_fwd(hin, L.mixn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mixn_r[0],
                       sm->red, mb::kD, mb::kD);   // xn -> sm->dr
        // mix_out written into adj_b at WIDTH-kD stride (s*kD+j); read it the same way.
        float* mo = &sm->adj_b[0][0];
        mb_mixer_fwd(L, &sm->dr[0][0], a, mo, sm);   // mix_out -> sm->adj_b (kD-strided)
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            a->h1[s][j] = hin[s * mb::kD + j] + mo[s * mb::kD + j];   // h1 = x + mixer_out
        }
        __syncthreads();
        // --- SwiGLU sub-block: h2 = h1 + mlp(RMSNorm_mlp(h1)) ---
        // xhat throwaway (adj_c); backward recomputes from the raw h1.
        mb_rmsnorm_fwd(&a->h1[0][0], L.mlpn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mlpn_r[0],
                       sm->red, mb::kD, mb::kD);   // h1n -> sm->dr
        mb_swiglu_fwd(L, &sm->dr[0][0], a, mo, sm);  // mlp_out -> sm->adj_b (kD-strided)
        float* dst = (li + 1 < mb::kLayers) ? &sm->layer_in[li + 1][0][0] : &sm->final_in[0][0];
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            dst[s * mb::kD + j] = a->h1[s][j] + mo[s * mb::kD + j];   // h2 = h1 + mlp_out
        }
        __syncthreads();
    }
```

NEW (only `a`/`hin`/`dst` selection + a trailing spill change; the math is verbatim):
```
    for (int li = 0; li < mb::kLayers; ++li) {
        const MambaWeights::Layer& L = w.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[mb_smem_la(li)];   // streamed: slot 0
        float* hin = &sm->layer_in[mb_ring(li)][0][0];            // streamed: ring slot
        // --- mixer sub-block: h1 = hin + mixer(RMSNorm_mix(hin)) --- (UNCHANGED MATH)
        mb_rmsnorm_fwd(hin, L.mixn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mixn_r[0],
                       sm->red, mb::kD, mb::kD);
        float* mo = &sm->adj_b[0][0];
        mb_mixer_fwd(L, &sm->dr[0][0], a, mo, sm);
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            a->h1[s][j] = hin[s * mb::kD + j] + mo[s * mb::kD + j];
        }
        __syncthreads();
        mb_rmsnorm_fwd(&a->h1[0][0], L.mlpn_w, &sm->dr[0][0], &sm->adj_c[0][0], &a->mlpn_r[0],
                       sm->red, mb::kD, mb::kD);
        mb_swiglu_fwd(L, &sm->dr[0][0], a, mo, sm);
        // STREAMED: the next block input must go to the ring slot for li+1; on SMALL
        // it goes to layer_in[li+1] exactly as before.
        float* dst = (li + 1 < mb::kLayers) ? &sm->layer_in[mb_ring(li + 1)][0][0]
                                            : &sm->final_in[0][0];
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            dst[s * mb::kD + j] = a->h1[s][j] + mo[s * mb::kD + j];
        }
        __syncthreads();
        // STREAMED ONLY: spill this layer's block input + LayerAct to HBM so the
        // backward can replay them (the smem slot is reused by li+1). On SMALL the
        // whole struct already holds every layer → no spill (if constexpr folds out).
        if constexpr (kMbStreamSmem) {
            float* hin_hbm = mb_acts_layer_in(acts, li);
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x)
                hin_hbm[idx] = hin[idx];
            float* a_hbm = reinterpret_cast<float*>(mb_acts_act(acts, li));
            const float* a_src = reinterpret_cast<const float*>(a);
            for (int64_t idx = threadIdx.x; idx < kMbOneLayerActFloats; idx += blockDim.x)
                a_hbm[idx] = a_src[idx];
            __syncthreads();
        }
    }
```

> CRITICAL ring caveat: with `kMbActsRing=2`, `mb_ring(li)` and `mb_ring(li+1)` differ for adjacent layers, so writing `dst` for `li+1` does NOT clobber `hin` for `li` (still being spilled). The spill of `hin` must complete (the trailing `__syncthreads()`) before slot `mb_ring(li)` is reused at `li+2`. Ring depth 2 is sufficient because only the current layer's input is live at spill time. (If a future change reads `layer_in[li-1]` after producing `li`, bump `kMbActsRing`.) Actually the simplest correctness-safe choice is `kMbActsRing = 2` AND spilling `hin` BEFORE producing `dst`; the order above (produce `dst` into ring slot `li+1`, then spill `hin` from ring slot `li`) is safe only because the two slots differ — KEEP ring ≥ 2.

### 7c. backward layer loop (mb_backward_sample:1273-1316)

VERBATIM OLD (1273-1277):
```
    for (int li = mb::kLayers - 1; li >= 0; --li) {
        const MambaWeights::Layer& L = w.layer[li];
        const MambaGrad::Layer& G = g.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        float* hin = &sm->layer_in[li][0][0];   // block input (raw x for mixer_norm bwd)
```
NEW (refill from HBM into the single smem slot, then run the SHIPPED bwd verbatim):
```
    for (int li = mb::kLayers - 1; li >= 0; --li) {
        const MambaWeights::Layer& L = w.layer[li];
        const MambaGrad::Layer& G = g.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[mb_smem_la(li)];   // streamed: slot 0
        float* hin = &sm->layer_in[mb_ring(li)][0][0];
        // STREAMED ONLY: refill this layer's block input + LayerAct from HBM into the
        // single smem slot (the fwd spilled them). On SMALL they are already resident.
        if constexpr (kMbStreamSmem) {
            const float* hin_hbm = mb_acts_layer_in(acts, li);
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x)
                hin[idx] = hin_hbm[idx];
            float* a_dst = reinterpret_cast<float*>(a);
            const float* a_hbm = reinterpret_cast<const float*>(mb_acts_act(acts, li));
            for (int64_t idx = threadIdx.x; idx < kMbOneLayerActFloats; idx += blockDim.x)
                a_dst[idx] = a_hbm[idx];
            __syncthreads();
        }
        // ... (1278-1316 backward body UNCHANGED — it reads `a` and `hin`) ...
```

The rest of the backward body (1278-1316) is VERBATIM — it only reads `a->*` and `hin`, both now correctly populated. The running `sm->dh` adjoint chain is layer-independent already (it lives in the layer-independent part of the struct), so no spill is needed for it.

> One subtlety: the SMALL path's `hin = &sm->layer_in[li]` is the SAME storage the forward wrote and is still resident (all layers cached) — unchanged. The streamed path overwrites the single ring slot per layer; correctness holds because the backward processes layers in strict reverse order and each refill precedes that layer's reads.

---

## 8. EDIT 5 — the launch `dyn_smem` request (now config-dependent)

FILE: `fused_mamba_megakernel.cuh`. Three sites request `kMambaSmemBytes`; all become the gated value. Define a single helper near the launcher:

```
// Per-CTA dynamic smem the TC kernel needs: the streamed (one-layer) size on the
// flagship path, the all-layers size on SMALL/bench (byte-identical request there).
__host__ __device__ __forceinline__ int mb_tc_dyn_smem_bytes() {
    return (int)(kMbStreamSmem ? kMbStreamSmemBytes : kMambaSmemBytes);
}
```

VERBATIM OLD (1311) — in `launch_fused_mamba_megakernel_tc`:
```
    const int dyn_smem = (int)kMambaSmemBytes;
```
NEW:
```
    const int dyn_smem = mb_tc_dyn_smem_bytes();   // streamed on flagship, all-layers on SMALL
```
Same single-line change at **1283** (`mb_tc_launched_nctas`). (The legacy scalar `launch_fused_mamba_megakernel` at line 317 is `SG_MB_SCALAR_MEGAKERNEL`-gated and not the TC path; it may keep `kMambaSmemBytes` — it is never built at flagship — but for consistency apply the same helper.)

> SMALL byte-identity: `kMbStreamSmem==false` ⇒ `mb_tc_dyn_smem_bytes()==kMambaSmemBytes` (210.79 KB) → the `cudaFuncSetAttribute`, the occupancy query, and `<<<>>>` are bit-identical to today; `test_mamba_tc.py` is unaffected. Flagship requests 1784.85 KB (Level A) — see §9.

---

## 9. HONEST feasibility verdict + Level B (what actually clears 227 KB)

### Level A (this spec — EXACT, apply-ready)
- Makes `MambaSampleSmem` layer-independent (decoder `DecTcSmem` mirror): one `LayerAct` + a `kMbActsRing`-deep `layer_in` ring.
- Moves the cross-layer acts (`layer_in[li]`, `LayerAct[li]`) to the HBM `MbActsHbm` region (decoder `DecActs` mirror), bound by `mb_acts_bind`, carved from `tok.workspace` front.
- Per-CTA workspace grows by `nCTA * mb_acts_stride_floats()` floats (flagship: `nCTA * 24*(8*2048 + 187440) = nCTA * 4,891,776` floats ≈ `nCTA · 18.66 MB`; at ~132 CTAs ≈ 2.4 GB — acceptable HBM, parallel to the decoder Fork-B acts which are ~161 MB but per-T not per-CTA-per-layer; the Mamba per-CTA-per-layer cost is higher because the scan bwd replays the full LayerAct, not bf16 tiles).
- **Byte-identical at SMALL: yes** (every new path `if constexpr (kMbStreamSmem)` / zero-width term folds out; struct extents fold to `mb::kLayers`).
- **Flagship launchable: NO, not yet** — `dyn_smem` drops 19.56 MB → 1.74 MB but 1.74 MB > 227 KB, so `cudaFuncSetAttribute` still rejects it. Level A removes the `×L` blowup (the prompt's named cause) but the d-scaled per-sample working set remains over budget.

### Level B (structure + exact formulas; the deeper rewrite to actually launch)
To clear 227 KB at flagship, additionally stream the **big `SEQ×{DINNER,DFF,D}` working buffers** out of smem. Two viable structures (both keep SMALL byte-identical via the same `if constexpr (kMbStreamSmem)` gate):

1. **Scratch-in-HBM (DecActs-extended):** move `x_in,z,y_scan,g_pre,u_mlp,adj_a,adj_b,adj_c,wff_a,wff_b,h1,dh,dr,final_in` to HBM (they are already POD `SEQ×W` slabs). The smem-resident remainder is the small scan/Nc/head caches + `xproj` + `fn_*` + `red` ≈ **120.85 KB** (computed in §1) → fits 227 KB. Cost: every `mb_linear`/`mb_rmsnorm`/scan now reads/writes HBM slabs instead of smem — heavy traffic, but the scan is the wall anyway (model_stage_mamba_tc.cuh:6 "scan-dominated"). Workspace adds `nCTA * (14 * SEQ * max(DINNER,DFF,D))` floats.
   - Honest extra win available: shrink `fn_xhat` from `[SEQ][D]` to `[1][D]` (only the last position is used — see `mb_forward_sample:949` `hlast`, and the head/final-norm bwd only touch `kSeq-1`). Saves ~56 KB at flagship. Requires editing the 3 `fn_xhat[mb::kSeq-1]` sites + the bwd `fn_xhat` reuse (it is reused as a d-wide scratch in `mb_swiglu_bwd`/`mb_mixer_bwd` — that reuse must move to a separate `SEQ×D` HBM slab). This is why it is Level B, not the minimal change.

2. **Tile over d_inner/d_ff (no HBM scratch):** process the SSM channels and the SwiGLU width in tiles of `TILE` (e.g. 512) so `x_in/z/y_scan/adj_*` are `SEQ×TILE` in smem. The scan is already per-channel (one thread owns channel `j`), so a channel-tile loop is natural; the projections (`mb_linear`) tile cleanly. This keeps everything in smem (no added HBM traffic) at the cost of re-reading weights per tile. Smem ≈ `kMbActsRing*SEQ*D + one-LayerAct-with-DINNER→TILE + scratch-with-DINNER→TILE` — sized to fit 227 KB by choosing `TILE`. This is the higher-perf but larger rewrite (touches every `mb_*` device fn's loop bounds).

Either Level-B structure, once applied, lets you add the `static_assert(kMbStreamSmemBytes <= 227*1024)` guard and the flagship TC kernel both **compiles AND launches** (`dyn_smem < 227 KB`, occupancy ≥ 1).

### Gate after apply
- `test_mamba_tc.py` green: guaranteed by `kMbStreamSmem==false` at d=128 ⇒ every edit folds to the shipped code (struct extents, workspace offsets, dyn_smem request, fwd/bwd bodies all bit-identical). Verify with `-Xptxas -v` that the d=128 TU's `.shared` size and reg count are unchanged.
- Flagship compiles: Level A yes. Flagship LAUNCHES (`dyn_smem < 227 KB`): requires Level B.

---

## 10. Apply checklist (file → edit)

1. `mamba3_layout.cuh` + `mamba_flagship_layout.cuh`: add `kMbOneLayerActFloats`, `kMbAllLayersSmemFloats`, `kMbStreamSmem`, `kMbActsRing`, `kMbStreamSmemFloats`, `kMbStreamSmemBytes` (§3, §6a). Do NOT add the `<=227KB` assert until Level B.
2. `model_stage_mamba3.cuh`: gated struct extents + `mb_smem_la`/`mb_ring`/`kMbSmemLayers`/`kMbLayerInRing` + conditional static_assert (§4); `LayerAct` POD-size static_assert (§5).
3. `model_stage_mamba_tc.cuh` (namespace `mbtc`): `MbActsHbm`, `mb_acts_stride_floats`, `mb_acts_bind`, `mb_acts_layer_in`, `mb_acts_act` (§5).
4. `model_stage_mamba3.cuh`: add `MbActsHbm acts = {}` param to `mb_forward_sample`/`mb_backward_sample`; gated spill/refill + ring indexing in both layer loops (§7).
5. `fused_mamba_megakernel.cuh`: `mb_tc_workspace_floats` acts term (§6b); kernel workspace partition `acts_base`/`part_base`/`mb_acts_bind` (§6b); pass `acts` to the two `mb_*_sample` calls (§7a); `mb_tc_dyn_smem_bytes()` helper + the 2–3 `dyn_smem` sites (§8).
6. `model_stage_mamba_tc.cuh`: forward `acts` through the `_tp` SingleGPU wrappers if the flagship-TP path is in scope (§7a).

All new code is `if constexpr (kMbStreamSmem)` / zero-width gated ⇒ the d=128 production/test TU is byte-identical.
