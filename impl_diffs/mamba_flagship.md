# mamba_flagship — parameterize the Mamba-3 layout emitter (+ `--mamba-layout-flagship`) and L-generalize the Mamba TC backward Muon-2D enumeration

AREA:
- `grokking_optimizers/megakernel_codegen.py` (the Mamba layout emitters — the single source of truth)
- `csrc/fused/sm_90/model_stage_mamba_tc.cuh` (the `kMbMuon2D` per-layer 2D-weight table + the Muon-scratch size constants)
- `csrc/fused/sm_90/fused_mamba_megakernel.cuh` (the Muon driver loop that reads `kMbMuon2D[mi]`, the dormant `MbTcSmem::spec[8]`, comments)
- NEW file: `csrc/fused/sm_90/mamba_flagship_layout.cuh` (GENERATED — emitted by `--mamba-layout-flagship`)

GOAL: mirror the DECODER flagship work (templates `/workspace/impl_diffs/flagship.md` for the layout emitter and `/workspace/impl_diffs/flagship_dw.md` for the L-generalized backward) for **Mamba-3**:

1. Parameterize the Mamba layout emitter (`_mamba_param_sizes` / `_mamba_layout_body`) by `(d, layers)` with **defaults == the current module globals**, add `mamba_flagship_layout_header()`, and add the `--mamba-layout-flagship` CLI flag. The committed `mamba3_layout.cuh` (prod d=128 + bench d=1024) is **byte-identical** after these edits.
2. L-generalize the ONLY 2/L-layer-pinned enumeration in the Mamba TC backward — the `kMbMuon2D[17]` `__constant__` table in `model_stage_mamba_tc.cuh` — to a `mb::kLayers`-general FORMULA (`mb_muon_2d(mi)` + closed-form `mb_is_muon_2d(t)`), **value-identical at L=2**. Fix the now-wrong `kMbMuonMaxNumel` (it under-sizes the Muon NS scratch 7× at flagship). Generalize the dormant `MbTcSmem::spec[8]` count to `mbtc::kMbNumDwSpecs` (byte-identical at L=2).

Flagship Mamba dims come from `grokking_race_v2.py:253` `MODEL_SCALES_BY_MODEL['flagship']['mamba']` = `{dim_model:2048, num_heads:32, num_layers:24, mamba_state_dim:128, mamba_head_dim:64, mamba_expand:2, mamba_mlp_ratio:2}`.

> **CRITICAL eager fact (verified):** the eager `Mamba3Layer.__init__` (`grokking_optimizers/mamba3_block.py:289-313`) DERIVES `n_heads = d_inner // head_dim` and **ignores** any `num_heads`. `_raw_model` (`grokking_race_v2.py:504-515`) constructs the flagship Mamba with `head_dim=64, expand_factor=2` and does **NOT** pass `num_heads`. So at `d=2048`: `d_inner = 2*2048 = 4096`, `n_heads = 4096//64 = 64`. The config's `num_heads:32` is a DISPLAY value the Mamba path never consumes (it is `c["num_heads"]`, used by the decoder/vit only). Therefore the flagship Mamba layout uses **`n_heads=64`**, which `_mamba_dims(2048)` already produces. **The emitter needs NO `num_heads` / `head_dim` / `state` override — only `(d=2048, layers=24)`.** Everything else (`state=128, head_dim=64, expand=2, mlp_ratio=2`) is the module default and matches the flagship config exactly.

## Verified flagship layout numbers (computed; emitter must reproduce)

`_mamba_dims(2048)` → `d_inner=4096, Nc=64, head_dim=64, n_heads=64, dt_rank=128, d_ff=4096, x_proj_out=576`.

- `kMambaNumTensors = 2 + 20*L + 3 = 2 + 20*24 + 3 = 485`.
- `kMambaTotalElems = 1,265,411,169` (`int64_t`; holds it). Largest offset = `1,265,411,072 < INT32_MAX (2,147,483,647)` → the int32 `kMambaOffsets`/`kMambaSizes` tables and `kMambaMaxTensorNumel` (int) are EXACT (no overflow).
- Largest per-tensor numel = **`in_proj 2*d_inner*d = 16,777,216`** (tensor index 9 of layer 0). NOTE: at flagship the LARGEST 2D weight is **in_proj**, not x_proj (x_proj = `xpo*di = 576*4096 = 2,359,296`). At d=128 the largest is x_proj (`336*256 = 86016`). See the **Muon-scratch fix** below — the current `kMbMuonMaxNumel = mb::kXProj*mb::kDInner` is a d=128 coincidence that breaks at flagship.

## Verification of the Muon-2D index formula at L=2 (must reproduce `kMbMuon2D[17]` literals)

Per-layer 20-tensor block (codegen `_mamba_param_sizes`, megakernel_codegen.py 1231-1253; mirror in `mb_wbind`, model_stage_mamba3.cuh 247-267). For layer `li` the block starts at flat tensor index `2 + 20*li`; block-offsets r∈{0..19} are
`{mixn_w, A_log, D, B_bias, Bhat_bias, C_bias, Chat_bias, in_proj, x_proj, dt_proj_w, dt_proj_b, B_norm_w, Bhat_norm_w, C_norm_w, Chat_norm_w, out_proj, mlpn_w, gate, up, down}`.
Tail (after `2 + 20*L`): `{norm_w, out_weight, out_bias}`.

The **2D (Muon) weights** are the 7 per-layer block-offsets `{7,8,9,15,17,18,19}` (in_proj, x_proj, dt_proj_w, out_proj, gate, up, down) + global `tok` (0) + `pos` (1) + head `out.weight` (`2+20*L+1`).
`mb_muon_2d(mi)` dense order: `mi=0` tok; `mi=1` pos; `mi∈[2, 2+7L)`: `li=(mi-2)/7, kind=(mi-2)%7` → block-offset `{7,8,9,15,17,18,19}[kind]`, tidx `2+20*li+off`; `mi=2+7L` head `out.weight` tidx `2+20*L+1`.

- L=2 → tidx `{0,1, 9,10,11,17,19,20,21, 29,30,31,37,39,40,41, 43}` — **EXACTLY** the current `kMbMuon2D[17]` tidx list. (Verified by replay: matches element-for-element.)
- `kMbNumMuon2D = 2 + 7*mb::kLayers + 1 = 7*L + 3` (= **17** at L=2, **171** at the flagship L=24).
- `mb_is_muon_2d(t)`: `t∈{0,1}` (tok/pos) OR `t == 2+20*L+1` (head.out) OR (`t∈[2, 2+20*L)` AND `(t-2)%20 ∈ {7,8,9,15,17,18,19}`). At L=2 returns true for exactly the 17-tidx set above (same membership).

The per-weight `(rows, cols)` (unchanged shape formulas, just emitted per `kind`):
`in_proj [2*kDInner, kD]`, `x_proj [kXProj, kDInner]`, `dt_proj [kNHeads, kDtRank]`, `out_proj [kD, kDInner]`, `gate/up [kDff, kD]`, `down [kD, kDff]`; `tok [kVocab, kD]`, `pos [kSeq, kD]`, head `[kPHead, kD]`. At L=2 these reproduce the current `kMbMuon2D` rows/cols verbatim.

---

# FILE 1 — grokking_optimizers/megakernel_codegen.py

## Edit 1.1 — add the flagship dim-mirror constant (after `_MAMBA_BENCH_D = 1024`)

Mirrors `flagship.md` EDIT 1. Only `(d, layers)` are needed — everything else is the module default (matches the eager flagship config; see the CRITICAL eager fact above).

### OLD (verbatim)
```python
# SG_DEC_BENCH_LAYOUT dual-branch (commit 79d3840). At d=1024: d_inner=2*1024=2048,
# dt_rank=max(1024//16,1)=64, n_heads=2048//64=32, d_ff=2*1024=2048 (state_dim N=128
# and head_dim=64 are width-invariant, so Nc=64 is width-invariant too).
_MAMBA_BENCH_D = 1024
```

### NEW
```python
# SG_DEC_BENCH_LAYOUT dual-branch (commit 79d3840). At d=1024: d_inner=2*1024=2048,
# dt_rank=max(1024//16,1)=64, n_heads=2048//64=32, d_ff=2*1024=2048 (state_dim N=128
# and head_dim=64 are width-invariant, so Nc=64 is width-invariant too).
_MAMBA_BENCH_D = 1024

# ── FLAGSHIP Mamba-3 tier (canonical-published 1.5 B SSM) ────────────────────
# SINGLE SOURCE OF TRUTH for these dims: grokking_race_v2.py
# MODEL_SCALES_BY_MODEL['flagship']['mamba'] = {dim_model:2048, num_heads:32,
# num_layers:24, mamba_state_dim:128, mamba_head_dim:64, mamba_expand:2,
# mamba_mlp_ratio:2}. Mirrored here as a build-time constant (this generator imports
# NO torch / grokking_race_v2 — it is a pure codegen tool with no runtime call
# sites), exactly as _MAMBA_* mirror the tiny-tier eager model. ONLY (d, layers) are
# needed: the eager Mamba3Layer DERIVES n_heads = d_inner // head_dim and IGNORES
# the config's num_heads (mamba3_block.py:310-313; _raw_model passes head_dim=64,
# expand=2, NOT num_heads). At d=2048 that gives d_inner=4096, n_heads=4096//64=64
# (the config's num_heads:32 is a DISPLAY value the Mamba path never consumes). All
# other dims (state=128, head_dim=64, expand=2, mlp_ratio=2) are the module defaults
# == the flagship config, so _mamba_dims(2048) reproduces the eager flagship shapes
# exactly. The flagship layout is emitted into its OWN standalone header
# (mamba_flagship_layout_header / --mamba-layout-flagship); it does NOT touch the
# committed d=128 production or d=1024 bench layouts. At d=2048,L=24 the table is 485
# tensors, total 1,265,411,169 elems (every offset < INT32_MAX, so the int32
# kMambaOffsets/kMambaSizes tables are exact).
_MAMBA_FLAGSHIP_D, _MAMBA_FLAGSHIP_LAYERS = 2048, 24
```

## Edit 1.2 — parameterize `_mamba_param_sizes` by `layers`

Mirrors `flagship.md` EDIT 2. The body reads `_MAMBA_LAYERS` from the module global in the loop bound — replace that one read with a `layers` kwarg defaulting to it. Defaults == the historical value → callers that pass only `d` are byte-identical.

### OLD (verbatim)
```python
def _mamba_param_sizes(d: int = _MAMBA_D) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of the eager
    Mamba3Model, grokking_optimizers/mamba3_block.py). 45 tensors at nl=2; at
    d=128 total 593713. Per Mamba3Block (20 tensors) the order is: mixer_norm.w,
    then the mixer's OWN params A_log/D/B_bias/Bhat_bias/C_bias/Chat_bias, then the
    mixer submodules in_proj/x_proj/dt_proj.w/dt_proj.b/B_norm/Bhat_norm/C_norm/
    Chat_norm/out_proj, then mlp_norm.w, then SwiGLU gate/up/down. `d` is parametric
    so the d-scaled bench layout (SG_MB_BENCH_LAYOUT) reuses the SAME shape formula
    — every per-tensor shape is a function of the derived dims (d_inner=expand*d,
    Nc=state//2, n_heads, dt_rank=max(d//16,1), d_ff=mlp_ratio*d, x_proj_out) plus
    (vocab, phead, seq), so a single d controls the whole table."""
    v, ph, seq = _MAMBA_VOCAB, _MAMBA_PHEAD, _MAMBA_SEQ
    di, Nc, hd, nh, dtr, dff, xpo = _mamba_dims(d)
    sizes = [v * d, seq * d]                          # tok.weight, pos.weight
    for _ in range(_MAMBA_LAYERS):
```

### NEW
```python
def _mamba_param_sizes(d: int = _MAMBA_D, *, layers: int = _MAMBA_LAYERS) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of the eager
    Mamba3Model, grokking_optimizers/mamba3_block.py). 45 tensors at nl=2; at
    d=128 total 593713. Per Mamba3Block (20 tensors) the order is: mixer_norm.w,
    then the mixer's OWN params A_log/D/B_bias/Bhat_bias/C_bias/Chat_bias, then the
    mixer submodules in_proj/x_proj/dt_proj.w/dt_proj.b/B_norm/Bhat_norm/C_norm/
    Chat_norm/out_proj, then mlp_norm.w, then SwiGLU gate/up/down. Parametric in
    (d, layers): every per-tensor shape is a function of the derived dims
    (d_inner=expand*d, Nc=state//2, n_heads, dt_rank=max(d//16,1), d_ff=mlp_ratio*d,
    x_proj_out) plus (vocab, phead, seq), so the SAME formula drives the d-scaled
    bench layout (SG_MB_BENCH_LAYOUT, d=1024) AND the flagship layout (d=2048,
    layers=24). Layer-count L emits 2 + 20*L + 3 tensors (tok/pos head, 20 per
    Mamba3Block, norm+out.weight+out.bias tail). Default layers=_MAMBA_LAYERS →
    callers that pass only `d` are byte-identical."""
    v, ph, seq = _MAMBA_VOCAB, _MAMBA_PHEAD, _MAMBA_SEQ
    di, Nc, hd, nh, dtr, dff, xpo = _mamba_dims(d)
    sizes = [v * d, seq * d]                          # tok.weight, pos.weight
    for _ in range(layers):
```

> The rest of `_mamba_param_sizes` (the per-block `sizes += [...]` and the `sizes += [d, ph * d, ph]` tail + `return sizes`) is UNCHANGED — only the loop bound `_MAMBA_LAYERS` → `layers`. At d=128 `layers` binds `_MAMBA_LAYERS=2` → identical element-for-element.

## Edit 1.3 — parameterize `_mamba_layout_body` by `layers`

Mirrors `flagship.md` EDIT 3. Two `_MAMBA_LAYERS` reads inside the body must become `layers`: the `_mamba_param_sizes(d)` call (line 1277) and the `smem_floats` computation (lines 1302-1303, `_MAMBA_LAYERS * ...`). The emitted `SG_MB_LAYERS = {_MAMBA_LAYERS}` literal (line 1371) must become `{layers}`. Everything else (the `__constant__` tables, the static_asserts, `mamba_layout_check`, the smem block) already derives from the locals `sizes`/`total`/`n_tensors`/`smem_floats`, which now follow `layers`.

### OLD (verbatim)
```python
def _mamba_layout_body(d: int) -> str:
    """The constants + __constant__ tables + compile-time cross-check + dynamic-smem
    budget for ONE Mamba width `d`. Emitted into ONE of the SG_MB_BENCH_LAYOUT
    branches; the branches are mutually exclusive at preprocess time, so reusing the
    SAME symbol names (kMambaOffsets/kMambaSizes/mamba_layout_check) across both is
    safe.
```

### NEW
```python
def _mamba_layout_body(d: int, *, layers: int = _MAMBA_LAYERS) -> str:
    """The constants + __constant__ tables + compile-time cross-check + dynamic-smem
    budget for ONE Mamba config (d, layers). Emitted into ONE of the SG_MB_BENCH_LAYOUT
    branches (or the standalone flagship header); the branches are mutually exclusive
    at preprocess time, so reusing the SAME symbol names (kMambaOffsets/kMambaSizes/
    mamba_layout_check) across them is safe. Default layers=_MAMBA_LAYERS → callers
    that pass only `d` are byte-identical.
```

### OLD (verbatim)
```python
    sizes = _mamba_param_sizes(d)
    offsets, acc = [], 0
```

### NEW
```python
    sizes = _mamba_param_sizes(d, layers=layers)
    offsets, acc = [], 0
```

### OLD (verbatim)
```python
    smem_floats = (_MAMBA_LAYERS * seq * d + seq * d      # layer_in, final_in
                   + _MAMBA_LAYERS * la                   # act[layers]
                   + seq * d + seq + ph                   # fn_xhat, fn_r, logits
```

### NEW
```python
    smem_floats = (layers * seq * d + seq * d      # layer_in, final_in
                   + layers * la                   # act[layers]
                   + seq * d + seq + ph                   # fn_xhat, fn_r, logits
```

### OLD (verbatim)
```python
constexpr int SG_MB_LAYERS  = {_MAMBA_LAYERS};     // nl (Mamba3Block count)
```

### NEW
```python
constexpr int SG_MB_LAYERS  = {layers};     // nl (Mamba3Block count)
```

> Byte-identity note: at d=128 / d=1024 the body now substitutes `layers=_MAMBA_LAYERS=2` — identical to the old global read in all three sites — so the size list, the offset table, `n_tensors`, `total`, `smem_floats`, and the `SG_MB_LAYERS` literal are unchanged. The `smem_block` `if d == _MAMBA_D:` branch selector (line 1335) is LEFT keyed on `d` (NOT on layers) — prod d=128 keeps its production smem comment + `<228KB` cap assert; bench d=1024 keeps the bench comment with NO cap assert; the flagship (d=2048 ≠ `_MAMBA_D`) takes the bench-style branch (no `<228KB` cap assert), which is correct — the flagship TC build gates the scalar megakernel OFF (`SG_MB_SCALAR_MEGAKERNEL=0`), the only `MambaSampleSmem` consumer, exactly like the bench. **Do not key the smem-head selector on `layers`.**

## Edit 1.4 — add `mamba_flagship_layout_header()`

Mirrors `flagship.md` EDIT 4. Insert **immediately after** the end of `mamba_layout_header()` (after its closing `"""` return, before the next top-level `def dispatch_table()`). Anchor on the final lines of `mamba_layout_header`.

### OLD (verbatim)
```python
}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
"""


def dispatch_table() -> str:
```

### NEW
```python
}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
"""


def mamba_flagship_layout_header() -> str:
    """Emit a STANDALONE FLAGSHIP Mamba-3 weight-layout header (1.5 B SSM tier,
    d=2048, layers=24, n_heads=64, state=128, head_dim=64, expand=2, mlp_ratio=2 —
    485 tensors, total 1,265,411,169 elems). Separate include guard
    (SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_) and a SINGLE config (no
    SG_MB_BENCH_LAYOUT #if branch): a TU that wants the flagship layout includes
    THIS header instead of mamba3_layout.cuh. Symbol names are IDENTICAL
    (kMambaOffsets/kMambaSizes/kMambaNumTensors/kMambaTotalElems/SG_MB_* /
    kMambaMaxTensorNumel/kMambaSmemFloats/kMambaSmemBytes, namespace sg::fused::sm90),
    so the SAME kernel template binds against it unchanged.

    SOURCE OF TRUTH for the dims: grokking_race_v2.py
    MODEL_SCALES_BY_MODEL['flagship']['mamba'] (mirrored in _MAMBA_FLAGSHIP_*). Only
    (d, layers) are passed — the eager Mamba3Layer DERIVES n_heads = d_inner//head_dim
    (= 4096//64 = 64 at d=2048) and ignores the config's num_heads; all other dims are
    the module defaults == the flagship config.
    Generated by: python -m grokking_optimizers.megakernel_codegen
    --mamba-layout-flagship > csrc/fused/sm_90/mamba_flagship_layout.cuh"""
    di, _Nc, _hd, nh, _dtr, _dff, _xpo = _mamba_dims(_MAMBA_FLAGSHIP_D)
    body = _mamba_layout_body(_MAMBA_FLAGSHIP_D, layers=_MAMBA_FLAGSHIP_LAYERS)
    return f"""#ifndef SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_
#define SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/mamba_flagship_layout.cuh — GENERATED weight-layout mirror
// for the FLAGSHIP L3-REAL Mamba-3 megakernel (1.5 B SSM tier).
//
// AUTO-GENERATED by: python -m grokking_optimizers.megakernel_codegen \\
//     --mamba-layout-flagship > csrc/fused/sm_90/mamba_flagship_layout.cuh
// Do NOT hand-edit the numbers. SINGLE SOURCE OF TRUTH: megakernel_codegen.py
// _mamba_param_sizes() (parameterized by (d, layers)); the flagship dims
// (d={_MAMBA_FLAGSHIP_D}, layers={_MAMBA_FLAGSHIP_LAYERS}, n_heads={nh}, d_inner={di})
// mirror grokking_race_v2.py MODEL_SCALES_BY_MODEL['flagship']['mamba']. The eager
// Mamba3Layer derives n_heads = d_inner//head_dim (head_dim=64) and ignores the
// config's nominal num_heads:32, so n_heads={nh}. The flat blob is
// torch.cat([p.reshape(-1) for _, p in model.named_parameters()]); the kernel
// addresses tensor i at params + kMambaOffsets[i] for kMambaSizes[i] elems.
//
// A count/total mismatch fails the BUILD loudly (the static_asserts below).
//
// This is a STANDALONE single-config header (NO SG_MB_BENCH_LAYOUT #if branch):
// a TU that wants the flagship layout includes THIS file instead of
// mamba3_layout.cuh. Symbol names are byte-identical to mamba3_layout.cuh
// (kMambaOffsets/kMambaSizes/kMambaNumTensors/kMambaTotalElems/SG_MB_*), so the
// SAME kernel template binds against it unchanged. The committed d=128 production /
// d=1024 bench header mamba3_layout.cuh is NOT affected.
// ============================================================================

#include <cstdint>

namespace sg {{ namespace fused {{ namespace sm90 {{

// ── FLAGSHIP (d={_MAMBA_FLAGSHIP_D}, layers={_MAMBA_FLAGSHIP_LAYERS}): 1.5 B SSM tier (n_heads={nh}). ──
{body}

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_
"""


def dispatch_table() -> str:
```

> CRITICAL — brace escaping in the f-string (same rule as `decoder_flagship_layout_header`): the namespace OPEN is written `namespace sg {{ namespace fused {{ namespace sm90 {{` (doubled `{{` → `{`); the namespace CLOSE is the six-brace token `}}}}}}` (each `}}` → `}`, rendering the three literal C++ braces `}}}`). Keep all brace-doubling intact. After applying, the emitted header must end with the line `}}} // namespace sg::fused::sm90` immediately followed by `#endif  // SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_` — if you see `}} }`, the close token was mistyped.

## Edit 1.5 — add the `--mamba-layout-flagship` CLI flag (registration + handler)

Mirrors `flagship.md` EDIT 5.

### OLD (verbatim)
```python
    ap.add_argument("--mamba-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL Mamba weight-layout header "
                         "(csrc/fused/sm_90/mamba3_layout.cuh)")
    args = ap.parse_args(argv)
```

### NEW
```python
    ap.add_argument("--mamba-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL Mamba weight-layout header "
                         "(csrc/fused/sm_90/mamba3_layout.cuh)")
    ap.add_argument("--mamba-layout-flagship", action="store_true",
                    help="emit the FLAGSHIP (d=2048, layers=24) L3-REAL Mamba "
                         "weight-layout header "
                         "(csrc/fused/sm_90/mamba_flagship_layout.cuh)")
    args = ap.parse_args(argv)
```

### OLD (verbatim)
```python
    if args.mamba_layout:
        sys.stdout.write(mamba_layout_header())
        return 0

    ap.print_help()
```

### NEW
```python
    if args.mamba_layout:
        sys.stdout.write(mamba_layout_header())
        return 0

    if args.mamba_layout_flagship:
        sys.stdout.write(mamba_flagship_layout_header())
        return 0

    ap.print_help()
```

> `argparse` maps `--mamba-layout-flagship` → `args.mamba_layout_flagship` (dashes → underscores).

---

# FILE 2 — csrc/fused/sm_90/model_stage_mamba_tc.cuh

## Edit 2.1 — `kMbNumMuon2D` + the `kMbMuon2D` constant table → formula (`mb_muon_2d`), `mb_is_muon_2d` closed-form

Mirrors `flagship_dw.md` Edit 1.2. The `__device__ __constant__ MbMuon2D kMbMuon2D[...]` brace-list cannot be a loop-filled 171-entry table at L=24. Replace it with a `constexpr mb_muon_2d(mi)` accessor (returns the same `MbMuon2D` by value) and a formula `mb_is_muon_2d(t)`. `kMbNumMuon2D = 7*L+3`. The Muon driver loop in `fused_mamba_megakernel.cuh` is updated in Edit 3.1 to call `mb_muon_2d(mi)` instead of indexing the (now-removed) array.

### OLD (verbatim)
```cpp
constexpr int kMbNumMuon2D = 17;
struct MbMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ MbMuon2D kMbMuon2D[kMbNumMuon2D] = {
    {  0, mb::kVocab,      mb::kD       },   // tok.weight            [99,128]
    {  1, mb::kSeq,        mb::kD       },   // pos.weight            [8,128]
    {  9, 2 * mb::kDInner, mb::kD       },   // L0 in_proj.weight     [512,128]
    { 10, mb::kXProj,      mb::kDInner  },   // L0 x_proj.weight      [336,256]
    { 11, mb::kNHeads,     mb::kDtRank  },   // L0 dt_proj.weight     [4,8]
    { 17, mb::kD,          mb::kDInner  },   // L0 out_proj.weight    [128,256]
    { 19, mb::kDff,        mb::kD       },   // L0 gate_proj.weight   [256,128]
    { 20, mb::kDff,        mb::kD       },   // L0 up_proj.weight     [256,128]
    { 21, mb::kD,          mb::kDff     },   // L0 down_proj.weight   [128,256]
    { 29, 2 * mb::kDInner, mb::kD       },   // L1 in_proj.weight
    { 30, mb::kXProj,      mb::kDInner  },   // L1 x_proj.weight
    { 31, mb::kNHeads,     mb::kDtRank  },   // L1 dt_proj.weight
    { 37, mb::kD,          mb::kDInner  },   // L1 out_proj.weight
    { 39, mb::kDff,        mb::kD       },   // L1 gate_proj.weight
    { 40, mb::kDff,        mb::kD       },   // L1 up_proj.weight
    { 41, mb::kD,          mb::kDff     },   // L1 down_proj.weight
    { 43, mb::kPHead,      mb::kD       },   // out.weight            [97,128]
};
__device__ __forceinline__ bool mb_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kMbNumMuon2D; ++mi) if (kMbMuon2D[mi].tidx == t) return true;
    return false;
}
```

### NEW
```cpp
// kMbNumMuon2D = tok + pos + 7 weights/layer (in_proj,x_proj,dt_proj,out_proj,
//   gate,up,down) + head.out = 2 + 7*L + 1 (= 17 at L=2). The table is now a
// FORMULA (mb_muon_2d) — a __device__ __constant__ array can't be loop-filled to
// 171 entries at the flagship L=24.
constexpr int kMbNumMuon2D = 2 + 7 * mb::kLayers + 1;
struct MbMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Per-layer
// 20-tensor block (li) starts at flat tidx 2+20*li; the 7 2D weights are at
// block-offsets {7,8,9,15,17,18,19} = in_proj,x_proj,dt_proj_w,out_proj,gate,up,down.
// Dense order:
//   mi=0 tok[V,d]; mi=1 pos[seq,d];
//   mi∈[2,2+7L): li=(mi-2)/7, kind=(mi-2)%7 →
//     kind0 in_proj  tidx 2 +20li+7  [2*d_inner, d]
//     kind1 x_proj   tidx 2 +20li+8  [x_proj_out, d_inner]
//     kind2 dt_proj  tidx 2 +20li+9  [n_heads, dt_rank]
//     kind3 out_proj tidx 2 +20li+15 [d, d_inner]
//     kind4 gate     tidx 2 +20li+17 [d_ff, d]
//     kind5 up       tidx 2 +20li+18 [d_ff, d]
//     kind6 down     tidx 2 +20li+19 [d, d_ff]
//   mi=2+7L head out.weight tidx 2+20*L+1 [phead, d].
// At L=2 reproduces the old kMbMuon2D[17] EXACTLY (tidx {0,1,9,10,11,17,19,20,21,
// 29,30,31,37,39,40,41,43}).
__host__ __device__ __forceinline__ MbMuon2D mb_muon_2d(int mi) {
    if (mi == 0)                       return { 0, mb::kVocab, mb::kD };   // tok
    if (mi == 1)                       return { 1, mb::kSeq,   mb::kD };   // pos
    if (mi == 2 + 7 * mb::kLayers)     return { 2 + 20 * mb::kLayers + 1, mb::kPHead, mb::kD }; // head.out
    const int li   = (mi - 2) / 7;
    const int kind = (mi - 2) % 7;
    const int base = 2 + 20 * li;
    if (kind == 0) return { base + 7,  2 * mb::kDInner, mb::kD       };  // in_proj
    if (kind == 1) return { base + 8,  mb::kXProj,      mb::kDInner  };  // x_proj
    if (kind == 2) return { base + 9,  mb::kNHeads,     mb::kDtRank  };  // dt_proj
    if (kind == 3) return { base + 15, mb::kD,          mb::kDInner  };  // out_proj
    if (kind == 4) return { base + 17, mb::kDff,        mb::kD       };  // gate
    if (kind == 5) return { base + 18, mb::kDff,        mb::kD       };  // up
    return            { base + 19, mb::kD,          mb::kDff     };      // down
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only the
// 1D / non-2D weights to the AdamW aux tail. Closed-form (no table scan):
//   t∈{0,1} (tok/pos), OR t==head.out (2+20L+1), OR a per-layer 2D weight
//   (t∈[2,2+20L) and (t-2)%20 ∈ {7,8,9,15,17,18,19}).
__device__ __forceinline__ bool mb_is_muon_2d(int t) {
    if (t == 0 || t == 1) return true;
    if (t == 2 + 20 * mb::kLayers + 1) return true;       // head out.weight
    if (t >= 2 && t < 2 + 20 * mb::kLayers) {
        const int r = (t - 2) % 20;
        return (r == 7 || r == 8 || r == 9 || r == 15 || r == 17 || r == 18 || r == 19);
    }
    return false;
}
```

## Edit 2.2 — fix `kMbMuonMaxNumel` (d=128-coincidence → layout-derived, L/width-correct)

The current `kMbMuonMaxNumel = mb::kXProj * mb::kDInner` assumes **x_proj is the largest 2D weight**. That holds ONLY at d=128 (x_proj 86016 > in_proj 65536) and is **WRONG at the flagship**, where **in_proj `2*d_inner*d = 16,777,216`** is the largest 2D weight (x_proj is only `576*4096 = 2,359,296`). Under-sizing the Muon NS scratch (`mb_tc_muon_floats` carves `4*kMbMuonMaxNumel` for X/AX/AAX/orth) would OVERRUN the Muon workspace by ~7× at flagship → silent corruption / OOB in the Muon optimizer cell.

Fix: derive `kMbMuonMaxNumel` from the layout's `kMambaMaxTensorNumel` (= `mamba_layout_check::max_size()`, the largest of ALL tensors). VERIFIED (all three configs) the global-max tensor is ALWAYS one of the 2D Muon weights (prod d=128 → x_proj 86016; bench d=1024 → in_proj 4194304; flagship d=2048 → in_proj 16777216), so `kMambaMaxTensorNumel` is an EXACT, width-safe bound for the Muon scratch. At d=128 `kMambaMaxTensorNumel == 86016 == mb::kXProj*mb::kDInner` → **byte-identical** prod/bench (the workspace carve numbers are unchanged at every committed config). `kMbMuonMaxRows = 2*mb::kDInner` (in_proj rows) is already the largest-rows over the 2D set at every width — LEFT verbatim (the stale 65536 in the comment is corrected; the value was already 86016).

### OLD (verbatim)
```cpp
// Largest 2D weight (numel) + largest #rows over the table — sizes the per-matrix
// NS scratch. x_proj [336,256]=86016 is the largest numel; in_proj rows=512 is the
// largest #rows (A=XXᵀ is rows×rows).
constexpr int kMbMuonMaxNumel = mb::kXProj * mb::kDInner;   // 336*256 = 86016
constexpr int kMbMuonMaxRows  = 2 * mb::kDInner;            // 512 (in_proj rows)
```

### NEW
```cpp
// Largest 2D weight (numel) + largest #rows over the Muon-2D set — sizes the per-
// matrix NS scratch (mb_tc_muon_floats carves 4*kMbMuonMaxNumel + kMbMuonMaxRows²).
// kMbMuonMaxNumel is derived from the LAYOUT's max_size() (the largest of ALL
// tensors). VERIFIED at every config the global-max tensor is a 2D Muon weight, so
// max_size() is an exact, width-safe bound: prod d=128 → x_proj 86016; bench
// d=1024 → in_proj 4194304; flagship d=2048 → in_proj 16777216. (The old literal
// mb::kXProj*mb::kDInner was a d=128 COINCIDENCE — x_proj is NOT the largest 2D
// weight at the flagship, where in_proj 2*d_inner*d dominates; that literal would
// under-size the NS scratch ~7× → Muon-cell OOB.) At d=128 this equals 86016
// (== the old literal) → byte-identical workspace carve on prod/bench. kMbMuonMaxRows
// = 2*d_inner (in_proj rows) is the largest #rows over the 2D set at every width;
// A = X Xᵀ is rows×rows.
constexpr int kMbMuonMaxNumel = kMambaMaxTensorNumel;      // == mamba_layout_check::max_size()
constexpr int kMbMuonMaxRows  = 2 * mb::kDInner;           // 512 at d=128 (in_proj rows)
```

> `kMambaMaxTensorNumel` is in the SAME namespace (`sg::fused::sm90`, from the included `mamba3_layout.cuh` — or the force-included flagship header, whose `kMambaMaxTensorNumel` is the L=24 max), and is visible at this point (the include is at the top of `model_stage_mamba_tc.cuh`). `mbtc::kMbMuonMaxNumel` therefore tracks whichever layout the TU binds.

## Edit 2.3 — comment fix in the header banner (non-functional; keep the doc honest)

The banner says the table is "VERIFIED ... (17 matrices at the toy config)". Note it is now a formula.

### OLD (verbatim)
```cpp
// ── Per-model Muon 2D-weight table (the ndim==2 parameters Muon's P2.7
//    Newton-Schulz orthogonalizes; ndim==1/other weights take the AdamW aux tail).
//    Flat tensor index (named_parameters() order) + rows[dim0] + cols[dim1],
//    VERIFIED against the live Mamba3Model (17 matrices at the toy config).
//    A_log is now [n_heads] (ndim==1) → NOT here, unlike Mamba-1. ──────────────
```

### NEW
```cpp
// ── Per-model Muon 2D-weight set (the ndim==2 parameters Muon's P2.7 Newton-Schulz
//    orthogonalizes; ndim==1/other weights take the AdamW aux tail). Now a FORMULA
//    (mb_muon_2d / mb_is_muon_2d), L-general: 2 + 7*L + 1 matrices (tok, pos, 7
//    2D weights/layer, head.out). Flat tensor index (named_parameters() order) +
//    rows[dim0] + cols[dim1]; VERIFIED value-identical at L=2 against the live
//    Mamba3Model (the original 17-matrix toy table). A_log is [n_heads] (ndim==1) →
//    NOT here, unlike Mamba-1. ──────────────────────────────────────────────────
```

---

# FILE 3 — csrc/fused/sm_90/fused_mamba_megakernel.cuh

## Edit 3.1 — Muon driver loop: `kMbMuon2D[mi]` → `mb_muon_2d(mi)`

Mirrors `flagship_dw.md` Edit 2.2. The loop bound `kMbNumMuon2D` is unchanged (now L-general from Edit 2.1). Only the table read changes from array-index to the formula accessor.

### OLD (verbatim)
```cpp
        for (int mi = 0; mi < mbtc::kMbNumMuon2D; ++mi) {
            const mbtc::MbMuon2D M = mbtc::kMbMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
```

### NEW
```cpp
        for (int mi = 0; mi < mbtc::kMbNumMuon2D; ++mi) {
            const mbtc::MbMuon2D M = mbtc::mb_muon_2d(mi);   // was kMbMuon2D[mi]
            const int rows = M.rows, cols = M.cols;
```

## Edit 3.2 — dormant `MbTcSmem::spec[8]` → `mbtc::kMbNumDwSpecs` (byte-identical at L=2)

Mirrors `flagship_dw.md` Edit 2.1, but the array is the DORMANT dummy `MbDwSpec` (the scalar Mamba TC path does NOT use dW specs — `model_stage_mamba_tc.cuh:62-69` documents it as a vestige kept only so `MbTcSmem` compiles). The `8` is a magic literal that is NOT a function of layers, so it does not "break" at L=24 — but the task asks to generalize any layer-pinned spec count, and the decoder twin's `kDecNumDwSpecs` is the established pattern. Add a named compile-time count and use it.

> Why `8` and not `4*L+1` or `7*L`: this struct member is never populated/read on the scalar path; `8` was carried from the old Mamba-1 Fork-B (4 GEMMs/layer × 2 layers). To stay BYTE-IDENTICAL at L=2 and avoid growing dormant smem at L=24, define `kMbNumDwSpecs = 8` as a fixed dormant constant (NOT layer-scaled). This documents intent and removes the bare literal without changing any footprint at any L. (If a future Mamba TC dW path is revived, this is the single knob to re-derive — analogous to the decoder's `kDecNumDwSpecs`.)

First add the constant in `model_stage_mamba_tc.cuh` next to the `MbDwSpec` struct:

### OLD (verbatim)
```cpp
// Dummy dW spec (the dormant MbTcSmem holds an array of these; unused on the
// scalar path). Kept minimal so the struct + any reference compiles.
struct MbDwSpec {
    const __nv_bfloat16* dY; const __nv_bfloat16* X;
    int Nout; int Kin; int T; int grad_off;
    bool has_bias; const __nv_bfloat16* dY_bias; int bias_off;
};
```

### NEW
```cpp
// Dummy dW spec (the dormant MbTcSmem holds an array of these; unused on the
// scalar path). Kept minimal so the struct + any reference compiles.
struct MbDwSpec {
    const __nv_bfloat16* dY; const __nv_bfloat16* X;
    int Nout; int Kin; int T; int grad_off;
    bool has_bias; const __nv_bfloat16* dY_bias; int bias_off;
};
// Dormant dW-spec count for the (unused-on-the-scalar-path) MbTcSmem::spec[] array.
// FIXED at 8 (the old Mamba-1 Fork-B 4-GEMM×2-layer count) — NOT layer-scaled: the
// scalar Mamba TC path never populates/reads these specs, so keeping it constant
// holds MbTcSmem BYTE-IDENTICAL at every L (no dormant-smem growth at the flagship
// L=24). Named (vs a bare literal) to mirror the decoder's kDecNumDwSpecs and to be
// the single knob if a Mamba TC dW path is ever revived.
constexpr int kMbNumDwSpecs = 8;
```

Then update the `MbTcSmem` member in `fused_mamba_megakernel.cuh`:

### OLD (verbatim)
```cpp
    float dBmat[mb::kSeq * mb::kState];
    float dCmat[mb::kSeq * mb::kState];
    mbtc::MbDwSpec spec[8];
};
```

### NEW
```cpp
    float dBmat[mb::kSeq * mb::kState];
    float dCmat[mb::kSeq * mb::kState];
    mbtc::MbDwSpec spec[mbtc::kMbNumDwSpecs];
};
```

## Edit 3.3 — comments referencing the old 2D table (non-functional; keep doc accurate)

Two comments cite stale matrix counts / the array. Fix both (the P2.7 header at ~957 says "13 matrices" — already wrong, it's 17 at L=2; the Muon-scratch header at ~471-472 says "in_proj = 512×128 = 65536" which is wrong since the carve is x_proj 86016 at d=128).

### OLD (verbatim)
```cpp
    //    sequence; only the per-model 2D table (mbtc::kMbMuon2D, 13 matrices incl.
    //    tok[99,128]/pos[8,128]/A_log[256,16]/dt_proj[256,8]/x_proj[40,256]) and offset
    //    array (kMambaOffsets) differ. For EACH 2D matrix all CTAs cooperate: buf=μ·buf+g
```

### NEW
```cpp
    //    sequence; only the per-model 2D set (mbtc::mb_muon_2d, 2+7*L+1 matrices incl.
    //    tok[V,d]/pos[seq,d]/in_proj/x_proj/dt_proj per layer + head.out) and offset
    //    array (kMambaOffsets) differ. For EACH 2D matrix all CTAs cooperate: buf=μ·buf+g
```

### OLD (verbatim)
```cpp
//    is NOT here — it PERSISTS across steps as optimizer state, bound to the m slice
//    (st.exp_avg). Largest mamba 2D weight: in_proj = 512×128 = 65536 numel; largest rows =
//    512 ⇒ A = 512×512 (mbtc::kMbMuonMaxNumel/kMbMuonMaxRows). Carved UNCONDITIONALLY (≈
//    4·65536 + 512² + nCTA + 1 floats ≈ 2 MB) so the opt-agnostic cached launcher workspace
```

### NEW
```cpp
//    is NOT here — it PERSISTS across steps as optimizer state, bound to the m slice
//    (st.exp_avg). Largest mamba 2D weight (numel) is layout-derived (kMambaMaxTensorNumel:
//    x_proj 86016 at d=128, in_proj at the flagship); largest rows = 2*d_inner (512 at
//    d=128) ⇒ A = rows×rows (mbtc::kMbMuonMaxNumel/kMbMuonMaxRows). Carved UNCONDITIONALLY
//    (≈ 4·maxNumel + rows² + nCTA + 1 floats) so the opt-agnostic cached launcher workspace
```

---

# WHAT IS ALREADY L-GENERAL (verified, NO edit needed)

- **Forward / backward weight + grad binds** (`mb_wbind` / the grad-bind twin, `model_stage_mamba3.cuh:245-314`) walk `for (int li = 0; li < mb::kLayers; ++li)` with `i++` over the 20-tensor block → L-general for any layer count once `SG_MB_LAYERS` scales.
- **The forward layer loop** (`model_stage_mamba3.cuh:891`) and **backward layer loop** (`:1246`, `for li = mb::kLayers-1 .. 0`) are `mb::kLayers`-bounded.
- **Per-CTA smem arrays** `layer_in[mb::kLayers]`, `act[mb::kLayers]`, the grad-weight `layer[mb::kLayers]` arrays — all sized by `mb::kLayers` (= `SG_MB_LAYERS`), so they auto-scale with the layout.
- **P3 AdamW tail** (`fused_mamba_megakernel.cuh:1124`) loops `for t in [0, kMambaNumTensors)` and skips 2D via `mb_is_muon_2d(t)` (Edit 2.1 formula) → auto-scales.
- **GrokAdamW per-tensor β1** (`:1144`, `powf(1-γ, t)` with `t` = flat tensor index) — uses the work-steal task id directly, no per-layer literal.
- **Every workspace carve** in `fused_mamba_megakernel.cuh`: `mb_tc_workspace_floats` (`nCTA*kMambaTotalElems`), `mb_tc_looksam_floats` (`2*kMambaTotalElems`), `mb_tc_sg2_floats` / `mb_sg2_ws_stride_floats` (`2*kMambaNumTensors + sg2_ws_stride(kMbSG2Nmax)`, `kMbSG2Nmax = kMambaMaxTensorNumel`), and the SG2 `row_off64` stage (`for t in kMambaNumTensors`) — all derive from `kMambaTotalElems`/`kMambaNumTensors`/`kMambaMaxTensorNumel`, which the layout header sets. They auto-scale at the flagship.
- **`kMbMuonMaxRows = 2*mb::kDInner`** — in_proj rows is the largest #rows over the 2D set at every width (verified prod/bench/flagship). LEFT verbatim (only the numel was wrong).
- **`mamba3_layout.cuh` static_asserts** (`offsets_consistent` / `sum_sizes == kMambaTotalElems`) fold at compile time over the table; at L=24 that is 485 entries — well within nvcc constexpr step limits (the existing bench path folds 45; the decoder flagship already folds 582).
- **The selective-SSM per-layer state (A/B/C/D, conv):** there is NO conv in Mamba-3 (RMSNorm, no conv1d — `mamba3_layout.cuh:27`). A_log/D/B_bias/Bhat_bias/C_bias/Chat_bias and the 4 B/C-norm weights are ndim==1 (NOT Muon-2D) and are bound positionally by the `i++` walk (already L-general) and optimized by the P3 AdamW tail (`mb_is_muon_2d` returns false for them). So there is **no per-layer SSM-state enumeration table** to generalize beyond the Muon-2D set — the scalar fwd/bwd recompute the scan from the bound per-layer weights. This is the Mamba analogue of the decoder's LN-vec table being absent here (Mamba's RMSNorm γ has no separate reduce table; it is a 1D weight handled by the AdamW tail).

---

# GATE COMMANDS (corrected — the task's 3rd command has a typo)

```bash
cd /workspace/SuperGrok1.5

# (1) byte-identity regression — production d=128 / bench d=1024 header UNCHANGED:
python -m grokking_optimizers.megakernel_codegen --mamba-layout > /tmp/m.cuh
diff /tmp/m.cuh csrc/fused/sm_90/mamba3_layout.cuh   # MUST be empty (byte-identical)

# (2) emit the flagship layout header:
python -m grokking_optimizers.megakernel_codegen --mamba-layout-flagship \
    > csrc/fused/sm_90/mamba_flagship_layout.cuh

# (3) compile the Mamba TC megakernel TU AGAINST the flagship layout. The TU
#     transitively #includes mamba3_layout.cuh (via model_stage_mamba_tc.cuh); to
#     bind the flagship table instead, force-include the flagship header FIRST (it
#     defines all kMamba* symbols) AND pre-define the COMMITTED header's include
#     guard so its body is skipped. THE REAL GUARD IS SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
#     (with the "3") — the task's gate wrote SG_FUSED_SM90_MAMBA_LAYOUT_CUH_ (no "3"),
#     which would NOT match the real guard and would double-define the symbols. Use:
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu \
    -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_DEC_SCALAR_MEGAKERNEL=0 \
    -DSG_FUSED_SM90_MAMBA3_LAYOUT_CUH_=1 \
    -include csrc/fused/sm_90/mamba_flagship_layout.cuh

# (4) ptxas regs/smem + occupancy on the flagship build:
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu \
    -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_DEC_SCALAR_MEGAKERNEL=0 \
    -DSG_FUSED_SM90_MAMBA3_LAYOUT_CUH_=1 \
    -include csrc/fused/sm_90/mamba_flagship_layout.cuh \
    -Xptxas -v --resource-usage 2>&1 | grep -E 'registers|smem|shared|spill'
```

> **Gate typo (loud flag for the lead):** the task's 3rd gate_command uses `-DSG_FUSED_SM90_MAMBA_LAYOUT_CUH_=1` (no "3"). The committed layout's actual include guard is `SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_` (with the "3" — `mamba3_layout.cuh:1`). With the no-"3" macro, the committed body is NOT skipped → `kMambaOffsets`/`kMambaSizes`/`SG_MB_*` are defined TWICE (flagship + committed) → redefinition error. The corrected `-D` (with the "3") is what makes the `-include` swap work, exactly as the decoder gate predefines `SG_FUSED_SM90_DECODER_LAYOUT_CUH_` (flagship.md EDIT 6). Also added `-DSG_DEC_SCALAR_MEGAKERNEL=0` per the task's command (harmless to the Mamba TU; the Mamba scalar gate is `SG_MB_SCALAR_MEGAKERNEL`).
>
> The flagship header's OWN guard is `SG_FUSED_SM90_MAMBA_FLAGSHIP_LAYOUT_CUH_` (distinct from both), so force-including it never collides with itself.

---

# RISKS / NOTES

1. **`n_heads=64`, NOT 32 (the config's nominal num_heads).** The flagship layout's `SG_MB_NHEADS=64` (derived `d_inner//head_dim`). This is CORRECT — the eager `Mamba3Layer` ignores `num_heads` and derives it (verified `mamba3_block.py:310-313` + `_raw_model` passes `head_dim`, not `num_heads`). If a reviewer expects 32 from the config row, the layout will look "wrong" but matches the eager model the parity test would build. (If the project ever wants 32 heads it must set `mamba_head_dim=128`; that is an eager-model change, out of scope.)

2. **Muon-scratch under-size was a LATENT bug at any large d (now fixed).** Edit 2.2 — the old `kMbMuonMaxNumel = mb::kXProj*mb::kDInner` is correct ONLY where x_proj is the largest 2D weight (d=128). At the bench d=1024 it is ALREADY wrong (in_proj 4194304 > x_proj 1048576) — so the Muon×Mamba bench cell, if ever run, would overrun. The fix (derive from `kMambaMaxTensorNumel`) is byte-identical at d=128 and correct everywhere. This is a value-identical-at-prod, behavior-FIX-at-scale change; it touches ONLY the Muon optimizer cell's workspace size (the L=2 AdamW gate `test_l3tc_tail_gate.py` does not exercise Muon scratch sizing, and the carve is the SAME 86016 at d=128).

3. **`mb_muon_2d` formula PTX is NOT byte-identical to the `.const` table (Edits 2.1+3.1).** The `kMbMuon2D` `.const` array becomes the inlined `mb_muon_2d()` accessor and the table-scan `mb_is_muon_2d` becomes a closed form. This is the Muon optimizer cell — a DIFFERENT cell from the gated AdamW path (`mega_mamba_real_adamw_tc.cu` → OptId::AdamW). The formula returns VALUE-IDENTICAL `(tidx,rows,cols)` + membership at L=2 (proven above), so Muon numerics/determinism are preserved at L=2; only the (untested-by-the-AdamW-gate) PTX shape changes. The `mb_muon_2d` ordering and `mb_is_muon_2d` membership both derive from the SAME per-layer `{7,8,9,15,17,18,19}` offsets + tok/pos/head, so P2.7 (orthogonalize) and P3 (skip-2D in the AdamW tail) stay consistent at any L.

4. **Production AdamW TU (`mega_mamba_real_adamw_tc.cu`) at L=2 is unaffected by the formula edits.** It binds OptId::AdamW; the Muon `if constexpr (Opt == OptId::Muon)` block (driver loop) and the 2D-skip are if-constexpr'd/dead for AdamW. The dormant `MbTcSmem::spec[8]→spec[kMbNumDwSpecs]` is byte-identical (`kMbNumDwSpecs==8`). So the L=2 AdamW STATE/parity gate (`tests/hw/test_mamba_megakernel.py`, `test_l3tc_tail_gate.py`) is bit-unchanged.

5. **smem / 1-CTA-per-SM at flagship — the same caveat as the decoder (LIKELY does NOT fit as-is).** The scalar Mamba megakernel's `MambaSampleSmem` is sized `∝ kLayers · (per-layer act)` and `∝ d`; at d=2048,L=24 the emitted `kMambaSmemFloats` placeholder is ~5.1M floats (~20 MB) — FAR over the sm_90 ~228 KB/SM cap. BUT the flagship header (like the bench branch) DROPS the `<228KB` cap assert, because the flagship TC build gates the scalar megakernel OFF (`SG_MB_SCALAR_MEGAKERNEL=0`) — the TC engine uses the small d-independent static `MbTcSmem`, NOT `MambaSampleSmem`. So the gate compiles the TC kernel + layout-table/asserts; it does NOT make the SCALAR megakernel fit. Whether the TC kernel's persistent per-CTA scan smem fits at d=2048 is the real occupancy question (the per-sample scalar fwd+bwd holds per-layer activations in HBM partials, not all in smem — but the per-CTA scan working set still grows with d/d_inner). Treat the flagship compile as "layout header + table correctness + smem-budget diagnosis", NOT "occupancy≥1 must pass". Mirrors `flagship.md` RISK 2 for the decoder.

6. **HBM workspace at flagship is enormous (scalar design).** `mb_tc_workspace_floats` carves `nCTA*kMambaTotalElems` (per-CTA FULL grad partial) + `2*kMambaTotalElems` (LookSAM). At 1.265e9 elems × ~132 SMs × 4 B ≈ 668 GB for the grad partial alone — far beyond HBM. This is the Mamba scalar TC design's per-CTA-full-grad (the honest deviation noted in `model_stage_mamba_tc.cuh`); it is unusable at flagship scale single-GPU and is the motivation for the TP8·ZeRO-3 mesh (the grad partial shards across the 8-way TP group). NOT a blocker for the compile gate (which proves table/symbol correctness), but flagged so a green compile is not misread as single-GPU flagship-trainable. (The decoder TC Fork-B eliminated per-CTA grad partials; the Mamba scalar path has not — a separate kernel redesign, out of scope here.)

7. **gfx942 / tpu_v6e untouched.** All edits are under `csrc/fused/sm_90/` Mamba TC headers + the sm_90 Mamba layout emitter + a new sm_90-only header. The gfx942 dispatch and Pallas/TPU paths are not referenced. Preserved.

8. **No committed-file regen required.** This spec does NOT rewrite the committed `mamba3_layout.cuh`; the flagship header is emitted on demand to `csrc/fused/sm_90/mamba_flagship_layout.cuh`. Gate (1) guarantees the production/bench header is byte-identical after Edits 1.1-1.3.
