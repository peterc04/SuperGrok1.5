# flagship_dw — L-generalize the decoder TC backward dW-spec / LN-vec / Muon-2D enumeration

AREA: `csrc/fused/sm_90/model_stage_decoder_tc.cuh` (+ pinned call sites in
`csrc/fused/sm_90/fused_decoder_megakernel.cuh`).

GOAL: generalize every HARDCODED 2-layer enumeration in the decoder TC backward
path to a general `dec::kLayers` form, so the flagship TC megakernel
(d=1600, L=48; layout `decoder_flagship_layout.cuh`, `kDecNumTensors=582`)
FUNCTIONALLY processes all 48 layers. Every edit is BYTE-IDENTICAL at L=2
(`dec::kLayers==2`) so `tests/hw/test_decoder_tc.py` (d=128/L=2) is unaffected:
the new constant `kDecNumDwSpecs` equals 9 at L=2, the spec/bias/loop bounds
collapse to the current `9`/`8`, and the new index formulas reproduce the exact
current literal tables verbatim at L=2 (proven below).

## Verification of the index formulas at L=2 (must reproduce the literals)

Per-layer 12-tensor stride (codegen `_decoder_param_sizes`, megakernel_codegen.py
596-617): for layer `li`, tensor index `2+12*li+r` with r∈{0..11} =
{in_w,in_b, out_w,out_b, n1_w,n1_b,n2_w,n2_b, ff0_w,ff0_b, ff2_w,ff2_b}.
Tail: `2+12*L+{0,1,2,3}` = {norm_w, norm_b, out_w, out_b}.

dW weights `tidx(li,kind) = 2 + 12*li + {0,2,8,10}[kind]`, bias `+1`; head
weight `2+12*L+2`, bias `+1`:
- L=2: s0..s7 → wi {2,4,10,12, 14,16,22,24}, bi {3,5,11,13, 15,17,23,25};
  head wi=2+24+2=28, bi=29. EXACTLY the current `wi[9]/bi[9]` + `spec[8]`.

LN-vec slots: v∈[0,4L): `li=v/4, kind=v%4`, tidx `6+12*li+kind`;
v=4L → norm_w `2+12*L`; v=4L+1 → norm_b `2+12*L+1`.
- L=2: {6,7,8,9, 18,19,20,21, 26,27}. EXACTLY the current `kLnVecTensorIdx[10]`.

Muon-2D: mi=0 tok{0}, mi=1 pos{1}, mi∈[2,2+4L): `li=(mi-2)/4, kind=(mi-2)%4` →
{in_w 2+12li, out_w 4+12li, ff0_w 10+12li, ff2_w 12+12li}; mi=2+4L → out_w `2+12L+2`.
- L=2: {0,1, 2,4,10,12, 14,16,22,24, 28}. EXACTLY the current `kDecMuon2D[11]`.

`dec_is_muon_2d(t)`: t∈{0,1} OR t==2+12L+2 OR (t∈[2,2+12L) AND (t-2)%12∈{0,2,8,10}).
- L=2: returns true for {0,1,2,4,10,12,14,16,22,24,28}. Same membership set.

`kDecNumDwSpecs = 4*dec::kLayers + 1` (= 9 at L=2, 193 at L=48).
`kNumLnVec = 4*dec::kLayers + 2` (= 10 at L=2, 194 at L=48).
`kDecNumMuon2D = 4*dec::kLayers + 3` (= 11 at L=2, 195 at L=48). [tok+pos+head + 4/layer]

---

# FILE 1 — csrc/fused/sm_90/model_stage_decoder_tc.cuh

## Edit 1.1 — LN-vec count + the `kLnVecTensorIdx` constant table → formula

The `__device__ __constant__ int kLnVecTensorIdx[...]` brace-list cannot be filled
by a runtime loop at L=48 (194 entries). Replace it with a `constexpr` index
formula `dec_lnvec_tensor_idx(v)` consumed by `dectc_lnvec_reduce`. `kNumLnVec`
and `kLnVecElems` become L-general (drive the HBM `lnvec` workspace carve, which
is already sized from `kLnVecElems` everywhere — no other change needed).

OLD:
```
// ── LN vector-grad partials layout (the 10 tile-local γ/β grads). Order MUST
//    match dec_layout tensor indices {6,7,8,9,18,19,20,21,26,27}. We store them
//    densely [10 x kD] per CTA; the P2 reduce maps them back by tensor index. ──
constexpr int kNumLnVec = 10;                 // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * dec::kD;   // 10 * 128 = 1280
// The dec_layout tensor index of each LN-vector slot, in our dense order.
__device__ __constant__ int kLnVecTensorIdx[kNumLnVec] = {
    6, 7, 8, 9,        // L0 n1.w, n1.b, n2.w, n2.b
    18, 19, 20, 21,    // L1 n1.w, n1.b, n2.w, n2.b
    26, 27             // norm.w, norm.b
};
```

NEW:
```
// ── LN vector-grad partials layout (the tile-local γ/β grads). Dense order:
//    4 slots/layer (n1.w,n1.b,n2.w,n2.b) for li∈[0,L), then norm.w,norm.b. At L=2
//    this is {6,7,8,9,18,19,20,21,26,27} (the original 10-slot table). We store
//    them densely [kNumLnVec x kD] per CTA; the P2 reduce maps them back by tensor
//    index via dec_lnvec_tensor_idx (a formula — a __constant__ array can't be
//    filled by a loop at L=48). L-GENERAL: kNumLnVec = 4*L+2. ──
constexpr int kNumLnVec = 4 * dec::kLayers + 2;   // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * dec::kD;  // (4*L+2)*kD  (1280 at L=2)
// dec_layout tensor index of LN-vector dense slot v. v∈[0,4L): li=v/4, kind=v%4 →
// 6+12*li+kind (n1.w/b,n2.w/b are tensor 6..9 of each 12-tensor layer block);
// v=4L → norm.w (2+12*L); v=4L+1 → norm.b (2+12*L+1). At L=2 reproduces the old
// kLnVecTensorIdx[10] EXACTLY ({6,7,8,9,18,19,20,21,26,27}).
__host__ __device__ __forceinline__ int dec_lnvec_tensor_idx(int v) {
    const int Lx4 = 4 * dec::kLayers;
    if (v < Lx4) return 6 + 12 * (v / 4) + (v % 4);
    return 2 + 12 * dec::kLayers + (v - Lx4);     // 2+12L (norm.w), 2+12L+1 (norm.b)
}
```

## Edit 1.2 — Muon 2D-weight table (constant array) → formula

The `__device__ __constant__ DecMuon2D kDecMuon2D[...]` brace-list cannot be a
loop-filled 195-entry table at L=48. Replace it with a `constexpr`
`dec_muon_2d(mi)` accessor returning the same `DecMuon2D` by value, and a
formula-based `dec_is_muon_2d(t)`. `kDecNumMuon2D = 4*L+3`. The Muon driver loop
in fused_decoder_megakernel.cuh is updated in Edit 2.6 to call `dec_muon_2d(mi)`
instead of indexing the (now-removed) array.

OLD:
```
constexpr int kDecNumMuon2D = 11;
struct DecMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ DecMuon2D kDecMuon2D[kDecNumMuon2D] = {
    { 0, dec::kVocab,  dec::kD     },   // tok.weight          [99,128]
    { 1, dec::kSeq,    dec::kD     },   // pos.weight          [4,128]
    { 2, 3*dec::kD,    dec::kD     },   // L0 in_proj_weight   [384,128]
    { 4, dec::kD,      dec::kD     },   // L0 out_proj.weight  [128,128]
    {10, dec::kDff,    dec::kD     },   // L0 ff.0.weight      [512,128]
    {12, dec::kD,      dec::kDff   },   // L0 ff.2.weight      [128,512]
    {14, 3*dec::kD,    dec::kD     },   // L1 in_proj_weight   [384,128]
    {16, dec::kD,      dec::kD     },   // L1 out_proj.weight  [128,128]
    {22, dec::kDff,    dec::kD     },   // L1 ff.0.weight      [512,128]
    {24, dec::kD,      dec::kDff   },   // L1 ff.2.weight      [128,512]
    {28, dec::kVocab,  dec::kD     },   // out.weight          [99,128]
};
// Is tensor index `t` one of the Muon 2D matrices (orthogonalized in P2.7)? P3
// uses this to route ONLY the 1D / non-2D weights to the AdamW aux tail for Muon.
__device__ __forceinline__ bool dec_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kDecNumMuon2D; ++mi) if (kDecMuon2D[mi].tidx == t) return true;
    return false;
}
```

NEW:
```
// kDecNumMuon2D = tok + pos + 4 weights/layer (in_proj,out_proj,ff0,ff2) + head.out
//   = 2 + 4*L + 1  (= 11 at L=2). The table is now a FORMULA (dec_muon_2d) — a
// __device__ __constant__ array can't be loop-filled to 195 entries at L=48.
constexpr int kDecNumMuon2D = 2 + 4 * dec::kLayers + 1;
struct DecMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Dense order:
//   mi=0 tok[V,d]; mi=1 pos[seq,d];
//   mi∈[2,2+4L): li=(mi-2)/4, kind=(mi-2)%4 →
//     kind0 in_proj  tidx 2 +12li  [3d,d]
//     kind1 out_proj tidx 4 +12li  [d, d]
//     kind2 ff0      tidx 10+12li  [dff,d]
//     kind3 ff2      tidx 12+12li  [d, dff]
//   mi=2+4L head out.weight tidx 2+12L+2 [V,d].
// At L=2 reproduces the old kDecMuon2D[11] EXACTLY (tidx {0,1,2,4,10,12,14,16,22,24,28}).
__host__ __device__ __forceinline__ DecMuon2D dec_muon_2d(int mi) {
    if (mi == 0)                         return { 0, dec::kVocab, dec::kD };   // tok
    if (mi == 1)                         return { 1, dec::kSeq,   dec::kD };   // pos
    if (mi == 2 + 4 * dec::kLayers)      return { 2 + 12 * dec::kLayers + 2, dec::kVocab, dec::kD }; // head.out
    const int li   = (mi - 2) / 4;
    const int kind = (mi - 2) % 4;
    if (kind == 0) return { 2  + 12 * li, 3 * dec::kD, dec::kD   };  // in_proj
    if (kind == 1) return { 4  + 12 * li, dec::kD,     dec::kD   };  // out_proj
    if (kind == 2) return { 10 + 12 * li, dec::kDff,   dec::kD   };  // ff0
    return            { 12 + 12 * li, dec::kD,     dec::kDff };      // ff2
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only
// the 1D / non-2D weights to the AdamW aux tail. Closed-form (no table scan):
//   t∈{0,1} (tok/pos), OR t==head.out (2+12L+2), OR a per-layer 2D weight
//   (t∈[2,2+12L) and (t-2)%12 ∈ {0,2,8,10} = in_w/out_w/ff0_w/ff2_w).
__device__ __host__ __forceinline__ bool dec_is_muon_2d(int t) {
    if (t == 0 || t == 1) return true;
    if (t == 2 + 12 * dec::kLayers + 2) return true;       // head out.weight
    if (t >= 2 && t < 2 + 12 * dec::kLayers) {
        const int r = (t - 2) % 12;
        return (r == 0 || r == 2 || r == 8 || r == 10);
    }
    return false;
}
```

## Edit 1.3 — `kDecNumDwSpecs` constant (NEW) next to the DecDwSpec struct

Add a compile-time spec count right after the `DecDwSpec` struct closes (line
1967 `};`). It is the array bound everywhere a dW spec array is declared/passed.

OLD:
```
    __nv_bfloat16* dYt;        // [Mpad, K] (transposed dY) — null unless active
    __nv_bfloat16* Xt;         // [Kin,  K] (transposed X)  — null unless active
    int64_t t_off;             // element offset of (dYt|Xt) base in the dW-T scratch
#endif
};
```

NEW:
```
    __nv_bfloat16* dYt;        // [Mpad, K] (transposed dY) — null unless active
    __nv_bfloat16* Xt;         // [Kin,  K] (transposed X)  — null unless active
    int64_t t_off;             // element offset of (dYt|Xt) base in the dW-T scratch
#endif
};

// Number of dW specs: 4 per layer (in_proj, out_proj, ff0, ff2) + 1 head.out.
//   = 4*L + 1  (= 9 at L=2, the original spec[9]; = 193 at the flagship L=48).
// This is the compile-time bound for EVERY DecDwSpec array (the DecTcSmem member,
// every spec[] signature/local, the dW phase loops, the bias prefix). At L=2 it
// is exactly 9 → the spec arrays + their stack/smem footprint are byte-identical.
constexpr int kDecNumDwSpecs = 4 * dec::kLayers + 1;
```

## Edit 1.4 — `dectc_build_dw_specs` (fwd decl, back-compat overload, body)

Generalize all three `spec[9]` to `spec[kDecNumDwSpecs]`, the `wi[9]/bi[9]` literal
tables + the `s<8` loop + `spec[8]` head to formulas, and the two `s<9` transpose
bind loops. The `8` head index becomes `4*dec::kLayers`.

OLD:
```
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[9],
        __nv_bfloat16* dwt_base);

// Back-compat overload (scalar path / pre-existing call sites): no transpose
// scratch. Forwards with dwt_base=nullptr so dYt/Xt are null and stage==0 runs
// the proven lambda gather unchanged.
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[9]) {
    dectc_build_dw_specs(acts, B, T, spec, /*dwt_base=*/nullptr);
}

__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[9],
        __nv_bfloat16* dwt_base) {
    // dec_layout offsets: see kDecOffsets. Weight tensor indices (and bias idx):
    //   L0: in_w=2 (in_b=3), out_w=4 (out_b=5), ff0_w=10 (ff0_b=11), ff2_w=12 (ff2_b=13)
    //   L1: in_w=14(15), out_w=16(17), ff0_w=22(23), ff2_w=24(25)
    //   head: out_w=28 (out_b=29)
    const int wi[9]  = {2,4,10,12, 14,16,22,24, 28};
    const int bi[9]  = {3,5,11,13, 15,17,23,25, 29};
    for (int s = 0; s < 8; ++s) {
        const int li = s / 4, kind = s % 4;
        DecDwSpec& sp = spec[s];
        sp.K = T; sp.grad_off = kDecOffsets[wi[s]]; sp.bias_off = kDecOffsets[bi[s]];
        if (kind == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * dec::kD; sp.Kin = dec::kD;   }
        else if (kind == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = dec::kD;     sp.Kin = dec::kD;   }
        else if (kind == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = dec::kDff;   sp.Kin = dec::kD;   }
        else                { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = dec::kD;     sp.Kin = dec::kDff; }
        sp.dY_bias = sp.dY;
    }
    DecDwSpec& hd = spec[8];
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = dec::kVocab; hd.Kin = dec::kD; hd.K = B;
    hd.grad_off = kDecOffsets[28]; hd.bias_off = kDecOffsets[29]; hd.dY_bias = hd.dY;

    // CONTIGUOUS-TRANSPOSE bind (stage==1): pack each weight's dYt[Mpad,K] then
    // Xt[Kin,K] into dwt_base via the SAME running-offset walk dec_dw_transpose_elems
    // uses (so the kernel offsets == the host carve). On the scalar path (dwt_base
    // null / stage==0) leave dYt/Xt null → dectc_dw_run_tile takes the lambda gather.
    // #if-guarded (not if-constexpr): the dYt/Xt/t_off fields only EXIST when the
    // macro is set, so the scalar default's struct + smem are byte-identical.
#if SG_TUNED_DEC_DW_STAGE
    if (kDecDwTransposeActive && dwt_base != nullptr) {
        int64_t e = 0;
        for (int s = 0; s < 9; ++s) {
            DecDwSpec& sp = spec[s];
            sp.t_off = e;
            sp.dYt   = dwt_base + e;                                  // [Mpad, K]
            sp.Xt    = dwt_base + e + dec_dw_mpad(sp.Nout) * sp.K;    // [Kin,  K]
            e += dec_dw_weight_t_elems(sp.Nout, sp.Kin, sp.K);
        }
    } else {
        for (int s = 0; s < 9; ++s) { spec[s].dYt = nullptr; spec[s].Xt = nullptr; spec[s].t_off = 0; }
    }
#else
    (void)dwt_base;
#endif
}
```

NEW:
```
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs],
        __nv_bfloat16* dwt_base);

// Back-compat overload (scalar path / pre-existing call sites): no transpose
// scratch. Forwards with dwt_base=nullptr so dYt/Xt are null and stage==0 runs
// the proven lambda gather unchanged.
__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs]) {
    dectc_build_dw_specs(acts, B, T, spec, /*dwt_base=*/nullptr);
}

__device__ __forceinline__ void dectc_build_dw_specs(
        const DecActs& acts, int B, int T, DecDwSpec spec[kDecNumDwSpecs],
        __nv_bfloat16* dwt_base) {
    // dec_layout offsets: see kDecOffsets. Per-layer 12-tensor block (li):
    //   weight tidx = 2 + 12*li + {0,2,8,10}[kind] (in_w,out_w,ff0_w,ff2_w), bias +1.
    //   head out.weight tidx = 2 + 12*L + 2, bias +1.
    // At L=2 these reproduce the old wi {2,4,10,12,14,16,22,24} / bi {3,5,11,13,
    // 15,17,23,25} and head {28,29} EXACTLY. 4*L layer specs + 1 head = kDecNumDwSpecs.
    for (int s = 0; s < 4 * dec::kLayers; ++s) {
        const int li = s / 4, kind = s % 4;
        const int woff = (kind == 0) ? 0 : (kind == 1) ? 2 : (kind == 2) ? 8 : 10;
        const int wi = 2 + 12 * li + woff;   // weight tensor index
        const int bi = wi + 1;               // bias tensor index
        DecDwSpec& sp = spec[s];
        sp.K = T; sp.grad_off = kDecOffsets[wi]; sp.bias_off = kDecOffsets[bi];
        if (kind == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * dec::kD; sp.Kin = dec::kD;   }
        else if (kind == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = dec::kD;     sp.Kin = dec::kD;   }
        else if (kind == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = dec::kDff;   sp.Kin = dec::kD;   }
        else                { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = dec::kD;     sp.Kin = dec::kDff; }
        sp.dY_bias = sp.dY;
    }
    DecDwSpec& hd = spec[4 * dec::kLayers];   // head spec (was spec[8] at L=2)
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = dec::kVocab; hd.Kin = dec::kD; hd.K = B;
    hd.grad_off = kDecOffsets[2 + 12 * dec::kLayers + 2];           // out.weight (28 at L=2)
    hd.bias_off = kDecOffsets[2 + 12 * dec::kLayers + 3];           // out.bias   (29 at L=2)
    hd.dY_bias = hd.dY;

    // CONTIGUOUS-TRANSPOSE bind (stage==1): pack each weight's dYt[Mpad,K] then
    // Xt[Kin,K] into dwt_base via the SAME running-offset walk dec_dw_transpose_elems
    // uses (so the kernel offsets == the host carve). On the scalar path (dwt_base
    // null / stage==0) leave dYt/Xt null → dectc_dw_run_tile takes the lambda gather.
    // #if-guarded (not if-constexpr): the dYt/Xt/t_off fields only EXIST when the
    // macro is set, so the scalar default's struct + smem are byte-identical.
#if SG_TUNED_DEC_DW_STAGE
    if (kDecDwTransposeActive && dwt_base != nullptr) {
        int64_t e = 0;
        for (int s = 0; s < kDecNumDwSpecs; ++s) {
            DecDwSpec& sp = spec[s];
            sp.t_off = e;
            sp.dYt   = dwt_base + e;                                  // [Mpad, K]
            sp.Xt    = dwt_base + e + dec_dw_mpad(sp.Nout) * sp.K;    // [Kin,  K]
            e += dec_dw_weight_t_elems(sp.Nout, sp.Kin, sp.K);
        }
    } else {
        for (int s = 0; s < kDecNumDwSpecs; ++s) { spec[s].dYt = nullptr; spec[s].Xt = nullptr; spec[s].t_off = 0; }
    }
#else
    (void)dwt_base;
#endif
}
```

## Edit 1.5 — `dectc_dw_transpose_operands` (signature + `s<9` loop)

OLD:
```
__device__ __forceinline__ void dectc_dw_transpose_operands(
        const DecDwSpec spec[9], int cta, int nCTA) {
#if SG_TUNED_DEC_DW_STAGE
    if constexpr (!kDecDwTransposeActive) { (void)spec; (void)cta; (void)nCTA; return; }
    const int tpb = blockDim.x;
    const int64_t stride = (int64_t)nCTA * tpb;
    const int64_t lane0  = (int64_t)cta * tpb + threadIdx.x;
    for (int s = 0; s < 9; ++s) {
```

NEW:
```
__device__ __forceinline__ void dectc_dw_transpose_operands(
        const DecDwSpec spec[kDecNumDwSpecs], int cta, int nCTA) {
#if SG_TUNED_DEC_DW_STAGE
    if constexpr (!kDecDwTransposeActive) { (void)spec; (void)cta; (void)nCTA; return; }
    const int tpb = blockDim.x;
    const int64_t stride = (int64_t)nCTA * tpb;
    const int64_t lane0  = (int64_t)cta * tpb + threadIdx.x;
    for (int s = 0; s < kDecNumDwSpecs; ++s) {
```

## Edit 1.6 — `dectc_dw_total_tiles` (signature + `s<9` loop)

OLD:
```
template <int N>
__device__ __forceinline__ int dectc_dw_total_tiles(const DecDwSpec spec[9]) {
    int n = 0;
    for (int s = 0; s < 9; ++s)
        n += dec_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}
```

NEW:
```
template <int N>
__device__ __forceinline__ int dectc_dw_total_tiles(const DecDwSpec spec[kDecNumDwSpecs]) {
    int n = 0;
    for (int s = 0; s < kDecNumDwSpecs; ++s)
        n += dec_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}
```

## Edit 1.7 — `dectc_dw_run_tile` (signature + `s<9` decode loop)

OLD:
```
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile(
        const DecDwSpec spec[9], int gt, float* __restrict__ grad,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    // Decode gt → (s, m_group, n_tile).
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < 9; ++s) {
```

NEW:
```
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, float* __restrict__ grad,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
    // Decode gt → (s, m_group, n_tile).
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < kDecNumDwSpecs; ++s) {
```

## Edit 1.8 — `dectc_dw_decode` (signature + `s<9` loop + fallback head index)

The unreachable fallback `s = 8;` is the head index — generalize to `4*dec::kLayers`.

OLD:
```
template <int N>
__device__ __forceinline__ void dectc_dw_decode(
        const DecDwSpec spec[9], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < 9; ++s) {
        const int ng = dec_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 8; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined
}
```

NEW:
```
template <int N>
__device__ __forceinline__ void dectc_dw_decode(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < kDecNumDwSpecs; ++s) {
        const int ng = dec_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 4 * dec::kLayers; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined (head idx)
}
```

## Edit 1.9 — `dectc_dw_run_tile_splitk` (signature)

OLD:
```
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile_splitk(
        const DecDwSpec spec[9], int gt, int kc, int G, float* __restrict__ dw_part,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
```

NEW:
```
template <int N>
__device__ __forceinline__ void dectc_dw_run_tile_splitk(
        const DecDwSpec spec[kDecNumDwSpecs], int gt, int kc, int G, float* __restrict__ dw_part,
        __nv_bfloat16* sA, __nv_bfloat16* sB) {
```

## Edit 1.10 — `dectc_dw_reduce_splitk` (signature only; loop is over n_dw)

OLD:
```
template <int N>
__device__ __forceinline__ void dectc_dw_reduce_splitk(
        const DecDwSpec spec[9], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
```

NEW:
```
template <int N>
__device__ __forceinline__ void dectc_dw_reduce_splitk(
        const DecDwSpec spec[kDecNumDwSpecs], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
```

## Edit 1.11 — `dectc_dw_biases` (signature + `pre[10]` + the two `s<9`/`t<9` + `pre[9]`)

`pre[10]` is a `[#specs + 1]` exclusive-prefix array → `pre[kDecNumDwSpecs + 1]`.
`pre[9]` (= total) → `pre[kDecNumDwSpecs]`. The `#pragma unroll` directives are
KEPT (byte-identical at L=2; they fully unroll a tiny 1-line prefix body — valid,
just larger code at L=48; the bodies are trivial).

OLD:
```
__device__ __forceinline__ void dectc_dw_biases(
        const DecDwSpec spec[9], float* __restrict__ grad, int cta, int nCTA) {
    // exclusive prefix of Nout across the 9 specs → total bias-output count.
    int pre[10];
    pre[0] = 0;
    #pragma unroll
    for (int s = 0; s < 9; ++s) pre[s + 1] = pre[s] + spec[s].Nout;
    const int total = pre[9];
    const int stride = nCTA * blockDim.x;
    for (int go = cta * blockDim.x + threadIdx.x; go < total; go += stride) {
        // decode global output index → (spec s, local row o). 9 specs → linear scan.
        int s = 0;
        #pragma unroll
        for (int t = 0; t < 9; ++t) if (go >= pre[t + 1]) s = t + 1;
        const DecDwSpec& sp = spec[s];
```

NEW:
```
__device__ __forceinline__ void dectc_dw_biases(
        const DecDwSpec spec[kDecNumDwSpecs], float* __restrict__ grad, int cta, int nCTA) {
    // exclusive prefix of Nout across the specs → total bias-output count.
    int pre[kDecNumDwSpecs + 1];
    pre[0] = 0;
    #pragma unroll
    for (int s = 0; s < kDecNumDwSpecs; ++s) pre[s + 1] = pre[s] + spec[s].Nout;
    const int total = pre[kDecNumDwSpecs];
    const int stride = nCTA * blockDim.x;
    for (int go = cta * blockDim.x + threadIdx.x; go < total; go += stride) {
        // decode global output index → (spec s, local row o). #specs → linear scan.
        int s = 0;
        #pragma unroll
        for (int t = 0; t < kDecNumDwSpecs; ++t) if (go >= pre[t + 1]) s = t + 1;
        const DecDwSpec& sp = spec[s];
```

## Edit 1.12 — `dectc_backward_tile` LN-vec slot pointers (`8`/`9` head, formula base)

The `gn_normw/gn_normb` are at hardcoded `8*kD`/`9*kD` — valid ONLY at L=2 (8 =
4*L). Generalize to `(4*L)*kD`/`(4*L+1)*kD`. The per-layer slot offsets
(`li*4 + {0,1,2,3}`) are ALREADY L-general (loop over `dec::kLayers`) — left
verbatim. Only the two norm-slot literals change.

OLD:
```
    float* gn_normw = lnvec + (int64_t)8 * dec::kD;
    float* gn_normb = lnvec + (int64_t)9 * dec::kD;
```

NEW:
```
    float* gn_normw = lnvec + (int64_t)(4 * dec::kLayers + 0) * dec::kD;  // 8*kD at L=2
    float* gn_normb = lnvec + (int64_t)(4 * dec::kLayers + 1) * dec::kD;  // 9*kD at L=2
```

## Edit 1.13 — `dectc_lnvec_reduce` (use `dec_lnvec_tensor_idx` instead of the array)

OLD:
```
    // Each CTA reduces a subset of the 10 LN tensors (round-robin by tensor).
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = kLnVecTensorIdx[v];
        const int64_t gbase = kDecOffsets[goff];
```

NEW:
```
    // Each CTA reduces a subset of the LN tensors (round-robin by tensor).
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = dec_lnvec_tensor_idx(v);   // was kLnVecTensorIdx[v]
        const int64_t gbase = kDecOffsets[goff];
```

---

# FILE 2 — csrc/fused/sm_90/fused_decoder_megakernel.cuh

## Edit 2.1 — `DecTcSmem::spec[9]` member → `spec[kDecNumDwSpecs]`

This is the ONLY smem-footprint change. At L=2, `kDecNumDwSpecs==9` → byte-identical
DecTcSmem. At L=48 the array grows by (193-9)*sizeof(DecDwSpec). With
SG_TUNED_DEC_DW_STAGE=1, sizeof(DecDwSpec)=72 B → +13,248 B. See RISKS for the
budget proof (fits the 48 KB STATIC cap at S=2 and the 228 KB DYNAMIC cap at the
default flagship S=4).

OLD:
```
    float red[256];
    dectc::DecDwSpec spec[9];
```

NEW:
```
    float red[256];
    dectc::DecDwSpec spec[dectc::kDecNumDwSpecs];
```

## Edit 2.2 — Muon driver loop: index `kDecMuon2D[mi]` → `dec_muon_2d(mi)`

The loop bound `kDecNumMuon2D` is unchanged (it is now L-general from Edit 1.2).
Only the table read changes from array-index to the formula accessor.

OLD:
```
        for (int mi = 0; mi < dectc::kDecNumMuon2D; ++mi) {
            const dectc::DecMuon2D M = dectc::kDecMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
```

NEW:
```
        for (int mi = 0; mi < dectc::kDecNumMuon2D; ++mi) {
            const dectc::DecMuon2D M = dectc::dec_muon_2d(mi);   // was kDecMuon2D[mi]
            const int rows = M.rows, cols = M.cols;
```

## Edit 2.3 — comment fix (non-functional, keeps the doc honest about the table)

OLD:
```
    //    kDecMuon2D, 11 matrices including tok[99,128]/pos[4,128]) and offset array
```

NEW:
```
    //    dec_muon_2d, 2+4L+1 matrices including tok/pos) and offset array
```

NOTE: every OTHER call site in fused_decoder_megakernel.cuh — `sm.spec` (lines
892/894/898/1068/1070), `dectc_dw_total_tiles<...>(spec)` (899/1071),
`dectc_dw_transpose_operands(spec,...)` (912), the split-K loops (922-927/
1073-1078), `dectc_dw_run_tile(spec,...)` (929-930/1080-1081),
`dectc_dw_biases(spec,...)` (937/1083), `dectc_lnvec_reduce(lnvec_base,...)`
(939/1087), and ALL lnvec/acts/dw-part workspace carves (`kLnVecElems`,
`dec_tc_acts_floats`, `dec_dw_transpose_elems`, `dec_dw_part_floats`,
`kDecDwMaxTiles`) — are ALREADY L-general (they pass the `sm.spec` POINTER /
loop over `dec::kLayers` / size from `kLnVecElems`/`kDecNumDwSpecs`-derived
constants). They need NO edit; they correctly pick up the generalized count
once Edits 1.x + 2.1 land.

---

# FILE 3 — csrc/fused/sm_90/pp_stage_decoder_tc.cuh (REQUIRED to keep the repo compiling)

Edit 1.1 REMOVES the `kLnVecTensorIdx` __constant__ array. `pp_stage_decoder_tc.cuh`
reads it at line 314 and is compiled by `tests/hw/pp_stage_binding.cu` (at the
DEFAULT L=2 layout — it defines no flagship layout). So this one-line read must
switch to the new `dec_lnvec_tensor_idx(v)` accessor or that TU fails to compile.
At L=2 the function returns the identical value the array held, so pp_stage is
behaviorally unchanged. (pp_stage's OTHER L=2 literals — the `spec[9]`/`s<9`/
`pre[10]` bias-prefix at lines 287-296 and `owns_*` ownership — are LEFT AS-IS:
they are correct + byte-identical at L=2, pp_stage's only compile target. pp_stage
is out of scope for L=48; see RISKS.)

## Edit 3.1 — pp_stage LN-vec reduce: array read → formula

OLD:
```
        const int goff = dectc::kLnVecTensorIdx[v];
```

NEW:
```
        const int goff = dectc::dec_lnvec_tensor_idx(v);   // was kLnVecTensorIdx[v]
```

## Edit 3.2 — pp_stage comment (non-functional; keep the doc reference valid)

OLD:
```
    // LN-vec slot ownership (dectc::kLnVecTensorIdx order: 4 slots/layer + 8,9).
```

NEW:
```
    // LN-vec slot ownership (dec_lnvec_tensor_idx order: 4 slots/layer + norm.w/b).
```

---

# WHAT IS ALREADY L-GENERAL (verified, NO edit needed)

- `DecActs` / `dec_acts_bind` / `dec_tc_acts_floats` — loop `dec::kLayers`.
- `DecWBf` / `dec_wbf_bind` / `dectc_wbf_convert` — the bf16 weight cache walks
  `li < dec::kLayers` with stride `kDecOffsets[wi + li*12]` (the same 12-stride),
  so item (4) of the task is ALREADY general. (`kWbfTotalElems = kLayers *
  kWbfLayerElems`.) Confirmed: `dectc_wbf_convert`'s `li = ii / kWbfLayerElems`
  and `kDecOffsets[wi + li * 12]` are layer-general for any L.
- `dec_dw_transpose_elems` — loops `dec::kLayers` + head (item (2) workspace carve
  is already L-general). `dec_dw_part_floats` → `kDecDwMaxTiles = dec::kLayers *
  kDecDwTilesPerLayer + kDecDwHeadTiles` (L-general).
- `DecTileScratch` / `dec_bind` (dec_weights.cuh) — per-layer arrays + loops over
  `dec::kLayers`. Forward path already runs L=48 per the project facts.
- `kDecMuonMaxNumel = dec::kDff*dec::kD` / `kDecMuonMaxRows = dec::kDff` — d-derived
  (ff0 [dff,d] is the largest 2D weight at every L), so the Muon NS scratch carve
  is already correct at L=48.
- `kLnVecElems` drives EVERY lnvec workspace carve (megakernel 641/710/711/800/
  810/1055; pp_stage 191/203/207/319) — they all scale automatically once Edit 1.1
  makes `kNumLnVec` L-general.

# RISKS / NOTES

1. **dW WORKSPACE (HBM) carves are L-general already** — `dec_dw_transpose_elems`
   (transpose scratch) and `dec_dw_part_floats`/`kDecDwMaxTiles` (split-K partials)
   loop `dec::kLayers`, and `dec_tc_acts_floats` loops `L`. The launcher's per-step
   dW work count is `n_dw = dectc_dw_total_tiles(spec)` (RUNTIME, counted from the
   spec array), NOT a 2-pinned literal — so it auto-scales to 193 specs at L=48.
   No L-pinned workspace literal remains.

2. **DecTcSmem static-smem budget** — `spec[]` is the ONLY layer-dependent smem
   member (Edit 2.1). With SG_TUNED_DEC_DW_STAGE=1, sizeof(DecDwSpec)=72 B, so the
   array grows 9*72=648 B (L=2) → 193*72=13,896 B (L=48), a +13.25 KB delta.
   Budget at the flagship (d=1600, N=128): sA/sB/red depend on TILE_N + ring depth,
   NOT on d or L. At the DEFAULT flagship build (PIPE=1, STAGES=4 → SG_DEC_TC_DYNAMIC
   _SMEM=1) DecTcSmem lives in DYNAMIC smem (228 KB cap, certified at launch via
   cudaOccupancyMaxActiveBlocks) so the +13.25 KB is absorbed (est. total ~47 KB ≪
   228 KB → still 1 CTA/SM). At the static path (PIPE=0 or STAGES=2) the est. total
   at S=2 is ~31 KB < the 48 KB `static_assert` — so even a static flagship build
   passes the compile-time cap. At L=2 the member is byte-identical (spec[9]) → the
   17.64 KB measured default and the static_assert are unchanged. NOTE: I did NOT
   re-measure sizeof at L=48 (read-only; no build) — the lead's 2nd gate_command
   (compile_to_object on the flagship layout) is the real check; if the static path
   ever trips the 48 KB static_assert at L=48, the fix is to keep `spec` in smem but
   build it once (it already is) — or route to dynamic smem via the existing gate.

3. **`#pragma unroll` over `kDecNumDwSpecs` in `dectc_dw_biases`** (Edit 1.11): at
   L=2 this fully unrolls a 9-trip prefix loop (byte-identical). At L=48 it fully
   unrolls a 193-trip loop and declares `pre[194]` (a per-thread local array,
   array-indexed → spills to local memory). This is FUNCTIONALLY correct and
   compiles, but at L=48 it is larger code + local-mem traffic in the bias pass.
   Left as `#pragma unroll` to GUARANTEE L=2 byte-identity (the gate). If L=48
   compile time / code size becomes an issue, change BOTH `#pragma unroll`s in this
   function to `#pragma unroll 1` — but ONLY behind `if constexpr (dec::kLayers>2)`
   or it perturbs the L=2 PTX. (Not needed for the functional flagship goal.)

4. **Muon table → formula PTX is NOT byte-identical** (Edits 1.2 + 2.2): the
   `kDecMuon2D` `.const` array becomes the inlined `dec_muon_2d()` accessor, and the
   table-scan `dec_is_muon_2d` becomes a closed form. This is the Muon optimizer
   cell, which is a DIFFERENT TU/cell from the gated AdamW path
   (`mega_decoder_real_adamw_tc.cu` → OptId::AdamW). The L=2 gate
   (test_decoder_tc.py) does NOT exercise Muon, and the formula returns
   VALUE-IDENTICAL (tidx,rows,cols) + membership at L=2 (proven above). So Muon
   numerics/determinism are preserved at L=2; only the (untested-by-this-gate) PTX
   shape changes. If a Muon-cell byte-identity gate exists elsewhere, treat this as
   a value-identical (not PTX-identical) change.

5. **pp_stage_decoder_tc.cuh is NOT made L-general** — only its `kLnVecTensorIdx`
   read is migrated to the new accessor (Edit 3.1, required because the symbol is
   removed). Its internal `spec[9]`/`s<9`/`pre[10]` bias-prefix + `owns_*`
   ownership are LEFT at the L=2 form. pp_stage compiles ONLY at L=2 (via
   tests/hw/pp_stage_binding.cu, no flagship layout), where those are correct and
   byte-identical. If pp_stage is ever run at the flagship, it needs the SAME
   `kDecNumDwSpecs` treatment — out of scope here.

6. **gfx942 / tpu_v6e untouched** — all edits are under `csrc/fused/sm_90/` decoder
   TC headers; no AMD/TPU tree is referenced.

7. **The Muon `dec_muon_2d` ordering must match `dec_is_muon_2d` membership** — both
   derive from the same per-layer {0,2,8,10} weight offsets + tok/pos/head, so P2.7
   (orthogonalize the table) and P3 (skip the 2D ones in the AdamW tail) stay
   consistent at any L. Verified identical sets at L=2.
