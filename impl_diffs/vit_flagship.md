# vit_flagship — mirror the DECODER flagship work for ViT

AREA: `grokking_optimizers/megakernel_codegen.py` (the ViT emitters `_vit_*`) +
`csrc/fused/sm_90/model_stage_vit_tc.cuh` (+ one pinned member in
`csrc/fused/sm_90/fused_vit_megakernel.cuh`) + a NEW generated header
`csrc/fused/sm_90/vit_flagship_layout.cuh`.

GOAL (two parts, exact mirror of the decoder flagship + flagship_dw specs):

1. **Layout emitter** — parameterize the ViT layout emitter by `(d, layers, vocab,
   patch, npatch, heads)` like the decoder, add a `decoder_flagship`-style
   `vit_flagship_layout_header()` + a `--vit-layout-flagship` CLI flag. The existing
   `--vit-layout` output stays BYTE-IDENTICAL (every new keyword arg defaults to the
   historical `_VIT_*` constant).

2. **TC backward L-generalization** — every HARDCODED 2-layer / 10-spec / 10-LN-slot
   / 11-Muon enumeration in `model_stage_vit_tc.cuh` (the ViT analogues of the
   decoder's `dectc_build_dw_specs` / `kLnVecTensorIdx` / `kDecMuon2D`) is generalized
   to `vit::kLayers` via formulas PROVEN byte-identical at the current L=2. The new
   counts are `kVitNumDwSpecs = 4*L+2` (= 10 at L=2), `kNumLnVec = 4*L+2` (= 10),
   `kVitNumMuon2D = 2+4*L+1` (= 11). At L=2 they collapse to 10/10/11 and every index
   formula reproduces the exact current literal tables verbatim (proven below).

This is the SAME structure the applied decoder spec `/workspace/impl_diffs/flagship_dw.md`
delivered (read it as the template). The ViT deltas from the decoder are: ViT has
**4 lead tensors** (cls/patch.w/patch.b/pos) not 2 (tok/pos), so the per-layer block
base is `4 + 12*li` (decoder's is `2 + 12*li`) and the tail is `4 + 12*L + {0,1,2,3}`
= {norm.w, norm.b, out.w, out.b}; the dW spec list LEADS with patch_proj (spec[0],
`kind==1`) so `kVitNumDwSpecs = 4*L + 2` (patch + 4·L + head), not `4*L + 1`; and the
Muon table leads with patch_proj + pos (mi 0/1) so `kVitNumMuon2D = 2 + 4*L + 1`.

Flagship ViT dims = `MODEL_SCALES_BY_MODEL['flagship']['vit']` (grokking_race_v2.py:252
→ `{"dim_model": 1664, "num_heads": 16, "num_layers": 48}`); `vocab=97, patch=49,
npatch=16` are the ViT's fixed tokenizer/patch geometry (== `_VIT_VOCAB / _VIT_PATCH /
_VIT_NPATCH`). NOTE flagship `heads=16` is NOT what the `_vit_heads(d)=max(d//64,4)`
rule yields (1664//64 = 26) — so heads is mirrored EXPLICITLY as a flagship constant
(exactly as the decoder mirrored `_DEC_FLAGSHIP_HEADS=25` rather than computing it).

## Verification of the index formulas at L=2 (must reproduce the literals)

Per-layer 12-tensor stride (codegen `_vit_param_sizes`, megakernel_codegen.py
876-894): 4 lead tensors {cls=0, patch.w=1, patch.b=2, pos=3}, then for layer `li`
the 12-tensor block starts at `4 + 12*li` =
{in_w,in_b, out_w,out_b, n1_w,n1_b,n2_w,n2_b, ff0_w,ff0_b, ff2_w,ff2_b}.
Tail: `4 + 12*L + {0,1,2,3}` = {norm.w, norm.b, out.weight, out.bias}.

**dW specs** (`vittc_build_dw_specs`): spec[0]=patch_proj (tidx 1, bias 2, kind 1);
spec[1+s] for s∈[0,4L): li=s/4, kk=s%4, base=4+12·li; head spec[1+4L]:
out.weight `4+12L+2`, bias `+1`.
- L=2: spec[1..8] base+{0/2/8/10}: L0 {4,6,12,14}/bias{5,7,13,15}, L1 {16,18,24,26}/
  bias{17,19,25,27}; head out.weight `4+24+2=30`, bias 31. EXACTLY the current
  `spec[1..8]` loop + `spec[9]`.

**LN-vec slots** (`kLnVecTensorIdx`): v∈[0,4L): li=v/4, kind=v%4, tidx `8+12·li+kind`
(the n1.w/b,n2.w/b are tensors base+4..base+7 = (4+12li)+4..+7); v=4L → norm.w `4+12L`;
v=4L+1 → norm.b `4+12L+1`.
- L=2: {8,9,10,11, 20,21,22,23, 28,29}. EXACTLY the current `kLnVecTensorIdx[10]`.

**Muon-2D** (`kVitMuon2D`): mi=0 patch_proj{1}; mi=1 pos{3}; mi∈[2,2+4L):
li=(mi-2)/4, kind=(mi-2)%4 → {in_w 4+12li, out_w 6+12li, ff0_w 12+12li, ff2_w 14+12li};
mi=2+4L → out.weight `4+12L+2`.
- L=2: {1,3, 4,6,12,14, 16,18,24,26, 30}. EXACTLY the current `kVitMuon2D[11]`.

**`vit_is_muon_2d(t)`**: t∈{1,3} OR t==4+12L+2 OR (t∈[4,4+12L) AND (t-4)%12∈{0,2,8,10}).
- L=2: returns true for {1,3,4,6,12,14,16,18,24,26,30}. Same membership set.

**Counts at L=2 / L=48:**
`kVitNumDwSpecs = 4*L + 2` (= 10 at L=2, 194 at L=48). [patch + 4/layer + head]
`kNumLnVec      = 4*L + 2` (= 10 at L=2, 194 at L=48). [4/layer + norm.w/b]
`kVitNumMuon2D  = 2 + 4*L + 1` (= 11 at L=2, 195 at L=48). [patch+pos + 4/layer + head]

**Flagship layout table** (d=1664, L=48, heads=16, vocab=97, patch=49, npatch=16):
`kVitNumTensors = 4 + 12*48 + 4 = 584`; `kVitTotalElems = 1,596,200,417`; largest
per-tensor numel = `dff*d = 6656*1664 = 11,075,584` (ff.0 / ff.2); largest offset =
`1,596,200,320 < INT32_MAX` → the int32 `kVitOffsets`/`kVitSizes` tables + the int
`kVitMaxTensorNumel` are exact, no overflow. `kVitTotalElems` is `int64_t`, holds 1.6e9.
The per-sample `VitSampleSmem` is ≈ 2,304,784 B (2250.77 KB) — far over the 227 KB
dynamic cap, so the flagship layout takes the SCALED smem block (the `else` branch:
no `< 227 KB` assert), exactly like the d=2048 bench branch — the SCALAR ViT
megakernel is gated OFF (`SG_VIT_SCALAR_MEGAKERNEL=0`) and the TC engine the flagship
drives uses the small d-independent static `VitTcSmem`.

---

# FILE 1 — grokking_optimizers/megakernel_codegen.py

## Edit 1.1 — add the flagship ViT dim-mirror constant

Add the flagship ViT dims right after `_VIT_BENCH_D = 2048` (line 861), mirroring
the decoder's `_DEC_FLAGSHIP_*` block.

OLD:
```python
# mirrors the decoder's SG_DEC_BENCH_LAYOUT dual-branch (commit 79d3840).
_VIT_BENCH_D = 2048
```

NEW:
```python
# mirrors the decoder's SG_DEC_BENCH_LAYOUT dual-branch (commit 79d3840).
_VIT_BENCH_D = 2048

# ── FLAGSHIP ViT tier (~1.5 B params, the decoder-twin Vision-Transformer) ───
# SINGLE SOURCE OF TRUTH for these dims: grokking_race_v2.py
# MODEL_SCALES_BY_MODEL['flagship']['vit'] = {dim_model:1664, num_heads:16,
# num_layers:48}. Mirrored here as build-time constants (this generator imports
# NO torch / grokking_race_v2 — it is a pure codegen tool with no runtime call
# sites), exactly as _VIT_* mirror the tiny-tier eager model and _DEC_FLAGSHIP_*
# mirror the flagship decoder. vocab/patch/npatch are the ViT's fixed tokenizer/
# patch geometry (== _VIT_VOCAB / _VIT_PATCH / _VIT_NPATCH). HEADS is mirrored
# EXPLICITLY (not via _vit_heads(d)=max(d//64,4), which would give 26 at d=1664):
# the flagship model class uses 16 heads (head_dim=104). The flagship layout is
# emitted into its OWN standalone header (vit_flagship_layout_header /
# --vit-layout-flagship); it does NOT touch the committed d=128 production or
# d=2048 bench layouts. At d=1664,L=48 the table is 584 tensors, total
# 1,596,200,417 elems (every offset < INT32_MAX, so the int32 kVitOffsets/
# kVitSizes tables are exact).
_VIT_FLAGSHIP_D, _VIT_FLAGSHIP_HEADS, _VIT_FLAGSHIP_LAYERS = 1664, 16, 48
```

## Edit 1.2 — parameterize `_vit_param_sizes` by `(layers, vocab, patch, npatch)`

OLD:
```python
def _vit_param_sizes(d: int = _VIT_D) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of vit_oracle.py
    vit_param_layout()). 32 tensors; at d=128 total 418017. cls_token leads (leaf
    before the patch_proj submodule). `d` is parametric so the d-scaled bench
    layout (SG_VIT_BENCH_LAYOUT) reuses the SAME shape formula — every per-tensor
    shape is a function of (d, dff=4d, vocab, patch, npatch), so a single d
    controls the whole table."""
    dff, v, patch, npatch = 4 * d, _VIT_VOCAB, _VIT_PATCH, _VIT_NPATCH
    sizes = [1 * 1 * d, d * patch, d, (npatch + 1) * d]   # cls, patch.w/b, pos
    for _ in range(_VIT_LAYERS):
        sizes += [
            3 * d * d, 3 * d,                     # attn.in_proj_weight/bias
            d * d, d,                             # attn.out_proj.weight/bias
            d, d, d, d,                           # n1.w/b, n2.w/b
            dff * d, dff,                         # ff.0.weight/bias
            d * dff, d,                           # ff.2.weight/bias
        ]
    sizes += [d, d, v * d, v]                     # norm.w/b, out.weight/bias
    return sizes
```

NEW:
```python
def _vit_param_sizes(d: int = _VIT_D, *, layers: int = _VIT_LAYERS,
                     vocab: int = _VIT_VOCAB, patch: int = _VIT_PATCH,
                     npatch: int = _VIT_NPATCH) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of vit_oracle.py
    vit_param_layout()). At d=128,layers=2 there are 32 tensors, total 418017.
    cls_token leads (leaf before the patch_proj submodule). Parametric in
    (d, layers, vocab, patch, npatch) — every per-tensor shape is a function of
    those (dff=4d), so the SAME formula drives the d-scaled bench layout
    (SG_VIT_BENCH_LAYOUT, d=2048) AND the flagship layout (d=1664, layers=48).
    Layer-count L emits 4 + 12*L + 4 tensors (cls/patch.w/patch.b/pos lead, 12
    per layer, norm+out tail). Defaults == the historical (d=128, L=2) constants →
    callers that pass only `d` are byte-identical."""
    dff, v = 4 * d, vocab
    sizes = [1 * 1 * d, d * patch, d, (npatch + 1) * d]   # cls, patch.w/b, pos
    for _ in range(layers):
        sizes += [
            3 * d * d, 3 * d,                     # attn.in_proj_weight/bias
            d * d, d,                             # attn.out_proj.weight/bias
            d, d, d, d,                           # n1.w/b, n2.w/b
            dff * d, dff,                         # ff.0.weight/bias
            d * dff, d,                           # ff.2.weight/bias
        ]
    sizes += [d, d, v * d, v]                     # norm.w/b, out.weight/bias
    return sizes
```

> Byte-identity note: the default call `_vit_param_sizes(d)` now binds
> `layers=_VIT_LAYERS, vocab=_VIT_VOCAB, patch=_VIT_PATCH, npatch=_VIT_NPATCH` —
> the SAME values the old body read from globals — so the size list is identical
> element-for-element at d=128 and d=2048.

## Edit 1.3 — parameterize `_vit_layout_body` by `(layers, vocab, patch, npatch, heads)`

The body f-string header lines interpolate `{_VIT_VOCAB}/{_VIT_LAYERS}/{_VIT_PATCH}/
{_VIT_NPATCH}` directly; those become the parameters. The `heads` keyword lets the
flagship override the `_vit_heads(d)` rule (which would give the wrong 26). Defaults
== historical constants, so `--vit-layout` is byte-identical. The smem-block `if d ==
_VIT_D:` branch is UNCHANGED — flagship d=1664 ≠ _VIT_D=128 takes the `else` (scaled)
block (no `< 227 KB` cap assert), exactly as intended.

### Edit 1.3a — the `def` signature + the two locals that read globals

OLD:
```python
def _vit_layout_body(d: int) -> str:
    """The constants + __constant__ tables + compile-time cross-check + dynamic-smem
    budget for ONE ViT width `d`. Emitted into ONE of the SG_VIT_BENCH_LAYOUT
    branches; the branches are mutually exclusive at preprocess time, so reusing the
    SAME symbol names (kVitOffsets/kVitSizes/vit_layout_check) across both is safe.

    The smem-budget block mirrors sizeof(VitSampleSmem) (model_stage_vit.cuh)
    field-by-field; every field is a function of (seq, d, dff=4d, heads, npatch,
    patch, vocab) so a single d (+ its head count) scales it. The `< 227 KB`
    per-block dynamic-smem cap assert guards the SCALAR ViT megakernel (the only
    consumer of VitSampleSmem). It is emitted ONLY in the production branch: the
    bench branch compiles with the scalar megakernel gated OFF
    (SG_VIT_SCALAR_MEGAKERNEL=0, mirroring the decoder's SG_DEC_SCALAR_MEGAKERNEL),
    so the cap does not apply — the TC engine that the bench drives uses a small,
    d-independent static VitTcSmem, not VitSampleSmem."""
    sizes = _vit_param_sizes(d)
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)
    heads, dff, seq = _vit_heads(d), 4 * d, _VIT_NPATCH + 1
    npatch, patch, vocab = _VIT_NPATCH, _VIT_PATCH, _VIT_VOCAB
```

NEW:
```python
def _vit_layout_body(d: int, *, layers: int = _VIT_LAYERS,
                     vocab: int = _VIT_VOCAB, patch: int = _VIT_PATCH,
                     npatch: int = _VIT_NPATCH, heads: int | None = None) -> str:
    """The constants + __constant__ tables + compile-time cross-check + dynamic-smem
    budget for ONE ViT config (d, layers, vocab, patch, npatch, heads). Emitted into
    ONE of the SG_VIT_BENCH_LAYOUT branches (or the standalone flagship header); the
    branches are mutually exclusive at preprocess time, so reusing the SAME symbol
    names (kVitOffsets/kVitSizes/vit_layout_check) across them is safe. Defaults ==
    the historical (vocab=97, layers=2, patch=49, npatch=16) constants, and heads
    defaults to the _vit_heads(d) rule → callers that pass only `d` are byte-identical.
    Pass `heads=` to override the rule (the flagship uses 16, not the rule's d//64=26).

    The smem-budget block mirrors sizeof(VitSampleSmem) (model_stage_vit.cuh)
    field-by-field; every field is a function of (seq, d, dff=4d, heads, npatch,
    patch, vocab) so a single d (+ its head count) scales it. The `< 227 KB`
    per-block dynamic-smem cap assert guards the SCALAR ViT megakernel (the only
    consumer of VitSampleSmem). It is emitted ONLY in the production (d==_VIT_D)
    branch: the bench AND flagship widths compile with the scalar megakernel gated
    OFF (SG_VIT_SCALAR_MEGAKERNEL=0, mirroring the decoder's SG_DEC_SCALAR_MEGAKERNEL),
    so the cap does not apply — the TC engine that they drive uses a small,
    d-independent static VitTcSmem, not VitSampleSmem."""
    sizes = _vit_param_sizes(d, layers=layers, vocab=vocab, patch=patch, npatch=npatch)
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)
    if heads is None:
        heads = _vit_heads(d)
    dff, seq = 4 * d, npatch + 1
```

> Byte-identity note: at d=128 the body now binds `layers=2, vocab=97, patch=49,
> npatch=16, heads=_vit_heads(128)=4` — identical to the old globals — and the
> removed line `npatch, patch, vocab = _VIT_NPATCH, _VIT_PATCH, _VIT_VOCAB` is folded
> into the new parameters (`npatch/patch/vocab` are now the keyword args with those
> exact defaults). `seq = npatch + 1` is identical to the old `_VIT_NPATCH + 1`.

### Edit 1.3b — the f-string header lines: globals → parameters

The body f-string's first six `constexpr` lines hardcode `{_VIT_VOCAB}/{_VIT_LAYERS}/
{_VIT_PATCH}/{_VIT_NPATCH}`. Switch them to the parameters so the flagship emits its
own L/vocab/patch/npatch. (`{d}`, `{heads}`, `{seq}`, `{dff}` already use locals.)

OLD:
```python
    return f"""constexpr int SG_VIT_VOCAB  = {_VIT_VOCAB};          // p (head Linear(d, p))
constexpr int SG_VIT_D      = {d};
constexpr int SG_VIT_HEADS  = {heads};
constexpr int SG_VIT_LAYERS = {_VIT_LAYERS};
constexpr int SG_VIT_PATCH  = {_VIT_PATCH};          // patch pixel count (7×7)
constexpr int SG_VIT_NPATCH = {_VIT_NPATCH};          // image patches
constexpr int SG_VIT_SEQ    = SG_VIT_NPATCH + 1;  // {seq} (CLS + {npatch} patches)
constexpr int SG_VIT_DFF    = 4 * SG_VIT_D;       // {dff}
```

NEW:
```python
    return f"""constexpr int SG_VIT_VOCAB  = {vocab};          // p (head Linear(d, p))
constexpr int SG_VIT_D      = {d};
constexpr int SG_VIT_HEADS  = {heads};
constexpr int SG_VIT_LAYERS = {layers};
constexpr int SG_VIT_PATCH  = {patch};          // patch pixel count (7×7)
constexpr int SG_VIT_NPATCH = {npatch};          // image patches
constexpr int SG_VIT_SEQ    = SG_VIT_NPATCH + 1;  // {seq} (CLS + {npatch} patches)
constexpr int SG_VIT_DFF    = 4 * SG_VIT_D;       // {dff}
```

> Byte-identity note: at d=128 these substitute `vocab=97, layers=2, patch=49,
> npatch=16` — identical literals to the old `{_VIT_VOCAB}/{_VIT_LAYERS}/{_VIT_PATCH}/
> {_VIT_NPATCH}`. The other interpolations (`{n_tensors}`, `{total}`, `{offsets_block}`,
> `{sizes_block}`, `{smem_block}`) already derive from the now-parameterized `sizes`.
> Nothing below `SG_VIT_DFF` in the f-string changes.

## Edit 1.4 — add `vit_flagship_layout_header()`

Insert a new function immediately after the end of `vit_layout_header()` (after its
closing `"""` return, before the next top-level comment block `# ── PHASE 2: the real
Mamba-3 weight layout`). Anchor on the final lines of `vit_layout_header`.

OLD:
```python
}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_VIT_LAYOUT_CUH_
"""


# ── PHASE 2: the real Mamba-3 weight layout (single C++ source). ─────────────
```

NEW:
```python
}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_VIT_LAYOUT_CUH_
"""


def vit_flagship_layout_header() -> str:
    """Emit a STANDALONE FLAGSHIP ViT weight-layout header (the decoder-twin
    Vision-Transformer, d=1664, layers=48, heads=16, vocab=97, patch=49, npatch=16 —
    ~1.5 B params, 584 tensors, total 1,596,200,417 elems). Separate include guard
    (SG_FUSED_SM90_VIT_FLAGSHIP_LAYOUT_CUH_) and a SINGLE config (no
    SG_VIT_BENCH_LAYOUT #if branch): a TU that wants the flagship layout includes
    THIS header instead of vit_layout.cuh. Symbol names are IDENTICAL
    (kVitOffsets/kVitSizes/kVitNumTensors/kVitTotalElems/SG_VIT_* /
    kVitMaxTensorNumel, namespace sg::fused::sm90), so the SAME kernel template binds
    against it unchanged. The body takes the SCALED smem block (d≠_VIT_D ⇒ the `else`
    branch: no `< 227 KB` cap assert) — the flagship VitSampleSmem (≈2.25 MB) cannot
    host the SCALAR megakernel (gated OFF, SG_VIT_SCALAR_MEGAKERNEL=0); the TC engine
    the flagship drives uses the small d-independent static VitTcSmem.

    SOURCE OF TRUTH for the dims: grokking_race_v2.py
    MODEL_SCALES_BY_MODEL['flagship']['vit'] (mirrored in _VIT_FLAGSHIP_*).
    Generated by: python -m grokking_optimizers.megakernel_codegen
    --vit-layout-flagship > csrc/fused/sm_90/vit_flagship_layout.cuh"""
    body = _vit_layout_body(
        _VIT_FLAGSHIP_D, layers=_VIT_FLAGSHIP_LAYERS, vocab=_VIT_VOCAB,
        patch=_VIT_PATCH, npatch=_VIT_NPATCH, heads=_VIT_FLAGSHIP_HEADS)
    return f"""#ifndef SG_FUSED_SM90_VIT_FLAGSHIP_LAYOUT_CUH_
#define SG_FUSED_SM90_VIT_FLAGSHIP_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/vit_flagship_layout.cuh — GENERATED weight-layout mirror for
// the FLAGSHIP L3-REAL Vision-Transformer megakernel (~1.5 B params).
//
// AUTO-GENERATED by: python -m grokking_optimizers.megakernel_codegen \\
//     --vit-layout-flagship > csrc/fused/sm_90/vit_flagship_layout.cuh
// Do NOT hand-edit the numbers. SINGLE SOURCE OF TRUTH: megakernel_codegen.py
// _vit_param_sizes() (parameterized by (d, layers, vocab, patch, npatch)); the
// flagship dims (d={_VIT_FLAGSHIP_D}, layers={_VIT_FLAGSHIP_LAYERS}, heads={_VIT_FLAGSHIP_HEADS})
// mirror grokking_race_v2.py MODEL_SCALES_BY_MODEL['flagship']['vit']. The flat
// blob is torch.cat([p.reshape(-1) for _, p in model.named_parameters()]); the
// kernel addresses tensor i at params + kVitOffsets[i] for kVitSizes[i] elems.
//
// A count/total mismatch fails the BUILD loudly (a static_assert below).
//
// This is a STANDALONE single-config header (NO SG_VIT_BENCH_LAYOUT #if branch):
// a TU that wants the flagship layout includes THIS file instead of vit_layout.cuh.
// Symbol names are byte-identical to vit_layout.cuh
// (kVitOffsets/kVitSizes/kVitNumTensors/kVitTotalElems/SG_VIT_*), so the SAME
// kernel template binds against it unchanged. The committed d=128 production /
// d=2048 bench header vit_layout.cuh is NOT affected.
// ============================================================================

#include <cstdint>

namespace sg {{ namespace fused {{ namespace sm90 {{

// ── FLAGSHIP (d={_VIT_FLAGSHIP_D}, layers={_VIT_FLAGSHIP_LAYERS}): the decoder-twin ViT, ~1.5 B params. ──
{body}

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_VIT_FLAGSHIP_LAYOUT_CUH_
"""


# ── PHASE 2: the real Mamba-3 weight layout (single C++ source). ─────────────
```

> CRITICAL — brace escaping in the f-string: the namespace close must be written as
> the six-brace token `}}}}}}`, which an f-string renders to the three literal C++
> braces `}}}` (each `}}` → `}`). This matches `vit_layout_header()`'s own close (it
> uses `}}}}}}` → `}}}`). The `namespace sg {{ namespace fused {{ namespace sm90 {{`
> open uses doubled `{{` → `{`. Keep all brace-doubling intact. The emitted header
> must end with `}}} // namespace sg::fused::sm90` immediately followed by
> `#endif  // SG_FUSED_SM90_VIT_FLAGSHIP_LAYOUT_CUH_`.

## Edit 1.5 — add the `--vit-layout-flagship` CLI flag

### Edit 1.5a — the argparse arg (after `--vit-layout`)

OLD:
```python
    ap.add_argument("--vit-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL ViT weight-layout header "
                         "(csrc/fused/sm_90/vit_layout.cuh)")
    ap.add_argument("--mamba-layout", action="store_true",
```

NEW:
```python
    ap.add_argument("--vit-layout", action="store_true",
                    help="emit the PHASE-2 L3-REAL ViT weight-layout header "
                         "(csrc/fused/sm_90/vit_layout.cuh)")
    ap.add_argument("--vit-layout-flagship", action="store_true",
                    help="emit the FLAGSHIP (d=1664, layers=48) L3-REAL ViT "
                         "weight-layout header "
                         "(csrc/fused/sm_90/vit_flagship_layout.cuh)")
    ap.add_argument("--mamba-layout", action="store_true",
```

### Edit 1.5b — the handler (after the `args.vit_layout` block)

OLD:
```python
    if args.vit_layout:
        sys.stdout.write(vit_layout_header())
        return 0

    if args.mamba_layout:
```

NEW:
```python
    if args.vit_layout:
        sys.stdout.write(vit_layout_header())
        return 0

    if args.vit_layout_flagship:
        sys.stdout.write(vit_flagship_layout_header())
        return 0

    if args.mamba_layout:
```

> `argparse` maps `--vit-layout-flagship` to `args.vit_layout_flagship` (dashes →
> underscores), so the handler reads cleanly.

---

# FILE 2 — csrc/fused/sm_90/model_stage_vit_tc.cuh

## Edit 2.1 — Muon 2D-weight table (constant array) → formula

The `__device__ __constant__ VitMuon2D kVitMuon2D[...]` brace-list cannot be a
loop-filled 195-entry table at L=48. Replace it with a `constexpr` `vit_muon_2d(mi)`
accessor returning the same `VitMuon2D` by value, and a formula-based
`vit_is_muon_2d(t)`. `kVitNumMuon2D = 2 + 4*L + 1`. The Muon driver loop in
fused_vit_megakernel.cuh is updated in Edit 3.2 to call `vit_muon_2d(mi)`.

OLD:
```
constexpr int kVitNumMuon2D = 11;
struct VitMuon2D { int tidx; int rows; int cols; };
__device__ __constant__ VitMuon2D kVitMuon2D[kVitNumMuon2D] = {
    { 1, vit::kD,      vit::kPatch },   // patch_proj.weight  [128,49]
    { 3, vit::kSeq,    vit::kD     },   // pos.weight         [17,128]
    { 4, 3*vit::kD,    vit::kD     },   // L0 in_proj_weight  [384,128]
    { 6, vit::kD,      vit::kD     },   // L0 out_proj.weight [128,128]
    {12, vit::kDff,    vit::kD     },   // L0 ff.0.weight     [512,128]
    {14, vit::kD,      vit::kDff   },   // L0 ff.2.weight     [128,512]
    {16, 3*vit::kD,    vit::kD     },   // L1 in_proj_weight  [384,128]
    {18, vit::kD,      vit::kD     },   // L1 out_proj.weight [128,128]
    {24, vit::kDff,    vit::kD     },   // L1 ff.0.weight     [512,128]
    {26, vit::kD,      vit::kDff   },   // L1 ff.2.weight     [128,512]
    {30, vit::kVocab,  vit::kD     },   // out.weight         [97,128]
};
// Is tensor index `t` one of the Muon 2D matrices (orthogonalized in P2.7)? P3 uses
// this to route ONLY the 1D / non-2D weights to the AdamW tail for Muon.
__device__ __forceinline__ bool vit_is_muon_2d(int t) {
    #pragma unroll
    for (int mi = 0; mi < kVitNumMuon2D; ++mi) if (kVitMuon2D[mi].tidx == t) return true;
    return false;
}
```

NEW:
```
// kVitNumMuon2D = patch_proj + pos + 4 weights/layer (in_proj,out_proj,ff0,ff2)
//   + head.out = 2 + 4*L + 1  (= 11 at L=2). The table is now a FORMULA
// (vit_muon_2d) — a __device__ __constant__ array can't be loop-filled to 195
// entries at L=48.
constexpr int kVitNumMuon2D = 2 + 4 * vit::kLayers + 1;
struct VitMuon2D { int tidx; int rows; int cols; };
// The mi-th Muon 2D matrix (tensor index + rows/cols), L-general. Dense order:
//   mi=0 patch_proj.weight tidx 1  [d, patch];
//   mi=1 pos.weight        tidx 3  [seq, d];
//   mi∈[2,2+4L): li=(mi-2)/4, kind=(mi-2)%4, base=4+12*li →
//     kind0 in_proj  tidx base+0 [3d,d]
//     kind1 out_proj tidx base+2 [d, d]
//     kind2 ff0      tidx base+8 [dff,d]
//     kind3 ff2      tidx base+10 [d, dff]
//   mi=2+4L head out.weight tidx 4+12*L+2 [V,d].
// At L=2 reproduces the old kVitMuon2D[11] EXACTLY (tidx {1,3,4,6,12,14,16,18,24,26,30}).
__host__ __device__ __forceinline__ VitMuon2D vit_muon_2d(int mi) {
    if (mi == 0)                     return { 1, vit::kD,    vit::kPatch }; // patch_proj
    if (mi == 1)                     return { 3, vit::kSeq,  vit::kD     }; // pos
    if (mi == 2 + 4 * vit::kLayers)  return { 4 + 12 * vit::kLayers + 2, vit::kVocab, vit::kD }; // head.out
    const int li   = (mi - 2) / 4;
    const int kind = (mi - 2) % 4;
    const int base = 4 + 12 * li;
    if (kind == 0) return { base + 0,  3 * vit::kD, vit::kD   };  // in_proj
    if (kind == 1) return { base + 2,  vit::kD,     vit::kD   };  // out_proj
    if (kind == 2) return { base + 8,  vit::kDff,   vit::kD   };  // ff0
    return            { base + 10, vit::kD,     vit::kDff };      // ff2
}
// Is tensor index `t` a Muon 2D matrix (orthogonalized in P2.7)? P3 routes only
// the 1D / non-2D weights to the AdamW tail. Closed-form (no table scan):
//   t∈{1,3} (patch_proj/pos), OR t==head.out (4+12L+2), OR a per-layer 2D weight
//   (t∈[4,4+12L) and (t-4)%12 ∈ {0,2,8,10} = in_w/out_w/ff0_w/ff2_w).
__device__ __forceinline__ bool vit_is_muon_2d(int t) {
    if (t == 1 || t == 3) return true;
    if (t == 4 + 12 * vit::kLayers + 2) return true;       // head out.weight
    if (t >= 4 && t < 4 + 12 * vit::kLayers) {
        const int r = (t - 4) % 12;
        return (r == 0 || r == 2 || r == 8 || r == 10);
    }
    return false;
}
```

## Edit 2.2 — LN-vec count + the `kLnVecTensorIdx` constant table → formula

The `__device__ __constant__ int kLnVecTensorIdx[...]` brace-list cannot be filled by
a runtime loop at L=48 (194 entries). Replace it with a `constexpr` index formula
`vit_lnvec_tensor_idx(v)` consumed by `vittc_lnvec_reduce`. `kNumLnVec` and
`kLnVecElems` become L-general (they drive the HBM `lnvec` workspace carve, already
sized from `kLnVecElems` everywhere — no other change needed).

OLD:
```
// ── LN vector-grad partials layout (the 10 tile-local γ/β grads). Order MUST
//    match the vit_layout tensor indices of {n1.w, n1.b, n2.w, n2.b}×L plus
//    {norm.w, norm.b}. We store them densely [10 × kD] per CTA; the P2 reduce
//    maps them back by tensor index. (vit_layout order: per-layer block starts
//    at 4 + li*12, with n1.w/b at +4/+5, n2.w/b at +6/+7; norm.w/b at 28/29.) ─
constexpr int kNumLnVec = 10;                  // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * vit::kD;   // 10 * 128 = 1280
// The vit_layout tensor index of each LN-vector slot, in our dense order.
__device__ __constant__ int kLnVecTensorIdx[kNumLnVec] = {
    8, 9, 10, 11,      // L0 n1.w, n1.b, n2.w, n2.b  (4 + 0*12 + {4,5,6,7})
    20, 21, 22, 23,    // L1 n1.w, n1.b, n2.w, n2.b  (4 + 1*12 + {4,5,6,7})
    28, 29             // norm.w, norm.b
};
```

NEW:
```
// ── LN vector-grad partials layout (the tile-local γ/β grads). Dense order:
//    4 slots/layer (n1.w,n1.b,n2.w,n2.b) for li∈[0,L), then norm.w,norm.b. At L=2
//    this is {8,9,10,11,20,21,22,23,28,29} (the original 10-slot table). We store
//    them densely [kNumLnVec × kD] per CTA; the P2 reduce maps them back by tensor
//    index via vit_lnvec_tensor_idx (a formula — a __constant__ array can't be
//    filled by a loop at L=48). L-GENERAL: kNumLnVec = 4*L+2. (vit_layout order:
//    per-layer block starts at 4 + li*12, with n1.w/b at +4/+5, n2.w/b at +6/+7;
//    norm.w/b at 4+12L / 4+12L+1.) ─
constexpr int kNumLnVec = 4 * vit::kLayers + 2;    // n1_w,n1_b,n2_w,n2_b ×L + norm_w,norm_b
constexpr int kLnVecElems = kNumLnVec * vit::kD;   // (4*L+2)*kD  (1280 at L=2)
// vit_layout tensor index of LN-vector dense slot v. v∈[0,4L): li=v/4, kind=v%4 →
// 8+12*li+kind (n1.w/b,n2.w/b are tensors 4..7 of each 12-tensor layer block, base
// 4+12*li); v=4L → norm.w (4+12*L); v=4L+1 → norm.b (4+12*L+1). At L=2 reproduces
// the old kLnVecTensorIdx[10] EXACTLY ({8,9,10,11,20,21,22,23,28,29}).
__host__ __device__ __forceinline__ int vit_lnvec_tensor_idx(int v) {
    const int Lx4 = 4 * vit::kLayers;
    if (v < Lx4) return 8 + 12 * (v / 4) + (v % 4);
    return 4 + 12 * vit::kLayers + (v - Lx4);     // 4+12L (norm.w), 4+12L+1 (norm.b)
}
```

## Edit 2.3 — `kVitNumDwSpecs` constant (NEW) next to the `VitDwSpec` struct

Add a compile-time spec count right after the `VitDwSpec` struct closes (line 1279
`};`). It is the array bound everywhere a dW spec array is declared/passed.

OLD:
```
struct VitDwSpec {
    const __nv_bfloat16* dY;   // [K, Nout]  (for patch_proj: nullptr — uses dh0)
    const __nv_bfloat16* X;    // [K, Kin]
    int Nout; int Kin; int K;  // Kin = REAL in-dim; K = contraction length
    int Kpad;                  // Kin padded to a multiple of 16 (=64 for patch_proj's Kin=49)... see note
    int grad_off;              // element offset of this weight in `grad`
    int bias_off;              // element offset of the bias in `grad`
    int kind;                  // 0=transformer (dY/X both token rows), 1=patch_proj
};
```

NEW:
```
struct VitDwSpec {
    const __nv_bfloat16* dY;   // [K, Nout]  (for patch_proj: nullptr — uses dh0)
    const __nv_bfloat16* X;    // [K, Kin]
    int Nout; int Kin; int K;  // Kin = REAL in-dim; K = contraction length
    int Kpad;                  // Kin padded to a multiple of 16 (=64 for patch_proj's Kin=49)... see note
    int grad_off;              // element offset of this weight in `grad`
    int bias_off;              // element offset of the bias in `grad`
    int kind;                  // 0=transformer (dY/X both token rows), 1=patch_proj
};

// Number of dW specs: 1 patch_proj + 4 per layer (in_proj,out_proj,ff0,ff2) + 1
//   head.out = 2 + 4*L  (= 10 at L=2, the original spec[10]; = 194 at flagship L=48).
// This is the compile-time bound for EVERY VitDwSpec array (the VitTcSmem member,
// every spec[] signature/local, the dW phase loops). At L=2 it is exactly 10 → the
// spec arrays + their smem footprint are byte-identical.
constexpr int kVitNumDwSpecs = 2 + 4 * vit::kLayers;
```

## Edit 2.4 — `vittc_build_dw_specs` (signature + `s<8` loop + `spec[9]` head)

Generalize `spec[10]` to `spec[kVitNumDwSpecs]`, the `s<8` transformer loop to
`s < 4*vit::kLayers`, and the `spec[9]` head index to `spec[1 + 4*vit::kLayers]`.
The per-layer `base = 4 + li*12` is ALREADY L-general (left verbatim); the head
tensor index `30/31` becomes `4+12*L+2 / 4+12*L+3`.

OLD:
```
// Build the 10 specs (called by all CTAs; cheap). T = B*kSeq; Tp = B*kNPatch.
__device__ __forceinline__ void vittc_build_dw_specs(
        const VitActs& acts, int B, int T, int Tp, VitDwSpec spec[10]) {
    // vit_layout weight tensor indices (and bias idx). Per-layer block base = 4 + li*12:
    //   in_w = base+0 (in_b base+1), out_w base+2 (out_b base+3),
    //   ff0_w base+8 (ff0_b base+9), ff2_w base+10 (ff2_b base+11).
    //   patch_proj.weight=1 (bias=2); head out.weight=30 (out.bias=31).
    // spec[0] = patch_proj; spec[1..8] = transformer (li,kind); spec[9] = head.
    {
        VitDwSpec& sp = spec[0];
        sp.dY = nullptr; sp.X = acts.X_patch; sp.Nout = vit::kD; sp.Kin = vit::kPatch;
        sp.K = Tp; sp.Kpad = vit::kPatch;  // K (contraction over patch rows) IS Tp; Kin padding handled in run_tile
        sp.grad_off = kVitOffsets[1]; sp.bias_off = kVitOffsets[2]; sp.kind = 1;
    }
    for (int s = 0; s < 8; ++s) {
        const int li = s / 4, kk = s % 4;
        const int base = 4 + li * 12;
        VitDwSpec& sp = spec[1 + s];
        sp.K = T; sp.kind = 0; sp.Kpad = 0;
        if (kk == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * vit::kD; sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 0]; sp.bias_off = kVitOffsets[base + 1]; }
        else if (kk == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = vit::kD;     sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 2]; sp.bias_off = kVitOffsets[base + 3]; }
        else if (kk == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = vit::kDff;   sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 8]; sp.bias_off = kVitOffsets[base + 9]; }
        else              { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = vit::kD;     sp.Kin = vit::kDff; sp.grad_off = kVitOffsets[base + 10]; sp.bias_off = kVitOffsets[base + 11]; }
    }
    VitDwSpec& hd = spec[9];
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = vit::kVocab; hd.Kin = vit::kD; hd.K = B; hd.kind = 0; hd.Kpad = 0;
    hd.grad_off = kVitOffsets[30]; hd.bias_off = kVitOffsets[31];
}
```

NEW:
```
// Build the dW specs (called by all CTAs; cheap). T = B*kSeq; Tp = B*kNPatch.
// kVitNumDwSpecs = 2 + 4*L (patch + 4/layer + head); 10 at L=2.
__device__ __forceinline__ void vittc_build_dw_specs(
        const VitActs& acts, int B, int T, int Tp, VitDwSpec spec[kVitNumDwSpecs]) {
    // vit_layout weight tensor indices (and bias idx). Per-layer block base = 4 + li*12:
    //   in_w = base+0 (in_b base+1), out_w base+2 (out_b base+3),
    //   ff0_w base+8 (ff0_b base+9), ff2_w base+10 (ff2_b base+11).
    //   patch_proj.weight=1 (bias=2); head out.weight = 4+12*L+2 (out.bias +1; 30/31 at L=2).
    // spec[0] = patch_proj; spec[1..4L] = transformer (li,kind); spec[1+4L] = head.
    {
        VitDwSpec& sp = spec[0];
        sp.dY = nullptr; sp.X = acts.X_patch; sp.Nout = vit::kD; sp.Kin = vit::kPatch;
        sp.K = Tp; sp.Kpad = vit::kPatch;  // K (contraction over patch rows) IS Tp; Kin padding handled in run_tile
        sp.grad_off = kVitOffsets[1]; sp.bias_off = kVitOffsets[2]; sp.kind = 1;
    }
    for (int s = 0; s < 4 * vit::kLayers; ++s) {
        const int li = s / 4, kk = s % 4;
        const int base = 4 + li * 12;
        VitDwSpec& sp = spec[1 + s];
        sp.K = T; sp.kind = 0; sp.Kpad = 0;
        if (kk == 0)      { sp.dY = acts.dY_qkv[li]; sp.X = acts.X_in[li];  sp.Nout = 3 * vit::kD; sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 0]; sp.bias_off = kVitOffsets[base + 1]; }
        else if (kk == 1) { sp.dY = acts.dY_a[li];   sp.X = acts.X_ctx[li]; sp.Nout = vit::kD;     sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 2]; sp.bias_off = kVitOffsets[base + 3]; }
        else if (kk == 2) { sp.dY = acts.dY_ff0[li]; sp.X = acts.X_x1[li];  sp.Nout = vit::kDff;   sp.Kin = vit::kD;   sp.grad_off = kVitOffsets[base + 8]; sp.bias_off = kVitOffsets[base + 9]; }
        else              { sp.dY = acts.dY_ff2[li]; sp.X = acts.X_gact[li];sp.Nout = vit::kD;     sp.Kin = vit::kDff; sp.grad_off = kVitOffsets[base + 10]; sp.bias_off = kVitOffsets[base + 11]; }
    }
    VitDwSpec& hd = spec[1 + 4 * vit::kLayers];   // head spec (was spec[9] at L=2)
    hd.dY = acts.dY_logits; hd.X = acts.X_hn; hd.Nout = vit::kVocab; hd.Kin = vit::kD; hd.K = B; hd.kind = 0; hd.Kpad = 0;
    hd.grad_off = kVitOffsets[4 + 12 * vit::kLayers + 2];   // out.weight (30 at L=2)
    hd.bias_off = kVitOffsets[4 + 12 * vit::kLayers + 3];   // out.bias   (31 at L=2)
}
```

## Edit 2.5 — `vittc_dw_total_tiles` (signature + `s<10` loop)

OLD:
```
template <int N>
__device__ __forceinline__ int vittc_dw_total_tiles(const VitDwSpec spec[10]) {
    int n = 0;
    for (int s = 0; s < 10; ++s)
        n += vit_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}
```

NEW:
```
template <int N>
__device__ __forceinline__ int vittc_dw_total_tiles(const VitDwSpec spec[kVitNumDwSpecs]) {
    int n = 0;
    for (int s = 0; s < kVitNumDwSpecs; ++s)
        n += vit_dw_groups(spec[s].Nout) * ((spec[s].Kin + N - 1) / N);
    return n;
}
```

## Edit 2.6 — `vittc_dw_run_tile` (signature + `s<10` decode loop)

OLD:
```
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile(
        const VitDwSpec spec[10], int gt, const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < 10; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; break; }
        acc += ng * nt;
    }
```

NEW:
```
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile(
        const VitDwSpec spec[kVitNumDwSpecs], int gt, const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, __nv_bfloat16* sA, __nv_bfloat16* sB) {
    int acc = 0, s = 0, m_group = 0, n_tile = 0;
    for (s = 0; s < kVitNumDwSpecs; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; break; }
        acc += ng * nt;
    }
```

## Edit 2.7 — `vittc_dw_decode` (signature + `s<10` loop + fallback head index)

The unreachable fallback `s = 9;` is the head index — generalize to `1 + 4*vit::kLayers`.

OLD:
```
template <int N>
__device__ __forceinline__ void vittc_dw_decode(
        const VitDwSpec spec[10], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < 10; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 9; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined
}
```

NEW:
```
template <int N>
__device__ __forceinline__ void vittc_dw_decode(
        const VitDwSpec spec[kVitNumDwSpecs], int gt, int& s, int& m_group, int& n_tile) {
    int acc = 0;
    for (s = 0; s < kVitNumDwSpecs; ++s) {
        const int ng = vit_dw_groups(spec[s].Nout);
        const int nt = (spec[s].Kin + N - 1) / N;
        if (gt < acc + ng * nt) { int loc = gt - acc; m_group = loc / nt; n_tile = loc % nt; return; }
        acc += ng * nt;
    }
    s = 1 + 4 * vit::kLayers; m_group = 0; n_tile = 0;   // unreachable (gt < n_dw); keep defined (head idx)
}
```

## Edit 2.8 — `vittc_dw_run_tile_splitk` (signature)

OLD:
```
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile_splitk(
        const VitDwSpec spec[10], int gt, int kc, int G,
        const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ dw_part, __nv_bfloat16* sA, __nv_bfloat16* sB) {
```

NEW:
```
template <int N>
__device__ __forceinline__ void vittc_dw_run_tile_splitk(
        const VitDwSpec spec[kVitNumDwSpecs], int gt, int kc, int G,
        const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ dw_part, __nv_bfloat16* sA, __nv_bfloat16* sB) {
```

## Edit 2.9 — `vittc_dw_reduce_splitk` (signature only; loop is over n_dw)

OLD:
```
template <int N>
__device__ __forceinline__ void vittc_dw_reduce_splitk(
        const VitDwSpec spec[10], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
```

NEW:
```
template <int N>
__device__ __forceinline__ void vittc_dw_reduce_splitk(
        const VitDwSpec spec[kVitNumDwSpecs], int n_dw, int G, const float* __restrict__ dw_part,
        float* __restrict__ grad, int cta, int nCTA) {
```

## Edit 2.10 — `vittc_dw_biases` (signature + `s<10` loop)

OLD:
```
__device__ __forceinline__ void vittc_dw_biases(
        const VitDwSpec spec[10], const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, int cta, int nCTA) {
    const int warp = threadIdx.x >> 5;            // 0..(blockDim/32 - 1)
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    int gcol_base = 0;
    for (int s = 0; s < 10; ++s) {
```

NEW:
```
__device__ __forceinline__ void vittc_dw_biases(
        const VitDwSpec spec[kVitNumDwSpecs], const __nv_bfloat16* __restrict__ dh0,
        float* __restrict__ grad, int cta, int nCTA) {
    const int warp = threadIdx.x >> 5;            // 0..(blockDim/32 - 1)
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    int gcol_base = 0;
    for (int s = 0; s < kVitNumDwSpecs; ++s) {
```

## Edit 2.11 — `vittc_backward_tile` LN-vec norm-slot pointers (`8`/`9` head, formula base)

The `gn_normw/gn_normb` are at hardcoded `8*kD`/`9*kD` — valid ONLY at L=2 (8 = 4*L).
Generalize to `(4*L)*kD`/`(4*L+1)*kD`. The per-layer slot offsets (`li*4 + {0,1,2,3}`)
are ALREADY L-general (loop over `vit::kLayers`) — left verbatim. Only the two
norm-slot literals change.

OLD:
```
    float* gn_normw = lnvec + (int64_t)8 * vit::kD;
    float* gn_normb = lnvec + (int64_t)9 * vit::kD;
```

NEW:
```
    float* gn_normw = lnvec + (int64_t)(4 * vit::kLayers + 0) * vit::kD;  // 8*kD at L=2
    float* gn_normb = lnvec + (int64_t)(4 * vit::kLayers + 1) * vit::kD;  // 9*kD at L=2
```

## Edit 2.12 — `vittc_lnvec_reduce` (use `vit_lnvec_tensor_idx` instead of the array)

OLD:
```
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = kLnVecTensorIdx[v];
        const int64_t gbase = kVitOffsets[goff];
```

NEW:
```
    for (int v = cta; v < kNumLnVec; v += nCTA) {
        const int goff = vit_lnvec_tensor_idx(v);   // was kLnVecTensorIdx[v]
        const int64_t gbase = kVitOffsets[goff];
```

---

# FILE 3 — csrc/fused/sm_90/fused_vit_megakernel.cuh

## Edit 3.1 — `VitTcSmem::spec[10]` member → `spec[kVitNumDwSpecs]`

This is the ONLY smem-footprint change. At L=2, `kVitNumDwSpecs==10` → byte-identical
VitTcSmem. At L=48 the array grows by `(194-10)*sizeof(VitDwSpec)`. `VitDwSpec` =
2 ptrs (16 B) + 6 ints (24 B) = 40 B → +184*40 = +7,360 B. See RISKS for the budget
proof (VitTcSmem is the small static TC smem, NOT VitSampleSmem).

OLD:
```
    float red[256];
    vittc::VitDwSpec spec[10];
};
```

NEW:
```
    float red[256];
    vittc::VitDwSpec spec[vittc::kVitNumDwSpecs];
};
```

## Edit 3.2 — Muon driver loop: index `kVitMuon2D[mi]` → `vit_muon_2d(mi)`

The loop bound `kVitNumMuon2D` is unchanged (it is now L-general from Edit 2.1). Only
the table read changes from array-index to the formula accessor.

OLD:
```
        for (int mi = 0; mi < vittc::kVitNumMuon2D; ++mi) {
            const vittc::VitMuon2D M = vittc::kVitMuon2D[mi];
            const int rows = M.rows, cols = M.cols;
```

NEW:
```
        for (int mi = 0; mi < vittc::kVitNumMuon2D; ++mi) {
            const vittc::VitMuon2D M = vittc::vit_muon_2d(mi);   // was kVitMuon2D[mi]
            const int rows = M.rows, cols = M.cols;
```

## Edit 3.3 — comment fix (non-functional; keep the doc honest about the table)

OLD:
```
    //    2D weights (INTEGRATION-OPTSTAGES §3). For EACH 2D matrix (kVitMuon2D) all
```

NEW:
```
    //    2D weights (INTEGRATION-OPTSTAGES §3). For EACH 2D matrix (vit_muon_2d) all
```

NOTE: every OTHER call site in fused_vit_megakernel.cuh — `sm.spec` (lines 630/632/
770/772), `vittc_dw_total_tiles<...>(spec)` (633/773), the split-K loops (642/645/
777/780), `vittc_dw_run_tile(spec,...)` (648/783), `vittc_dw_biases(spec,...)`
(655/785), `vittc_lnvec_reduce(lnvec_base,...)` (657/787), and ALL lnvec/acts/dw-part
workspace carves (`kLnVecElems`, `vit_acts_bf16_count`, `vit_dw_part_floats`,
`kVitDwMaxTiles`) — are ALREADY L-general (they pass the `sm.spec` POINTER / loop over
`vit::kLayers` / size from `kLnVecElems`-derived constants). They need NO edit; they
correctly pick up the generalized count once Edits 2.x + 3.1 land.

---

# WHAT IS ALREADY L-GENERAL (verified, NO edit needed)

- `VitActs` / `vit_acts_bind` / `vit_acts_bf16_count` — loop `vit::kLayers` (lines
  309/331). The per-layer X/dY pointer arrays are `[vit::kLayers]`.
- `VitTileScratch` / `vit_bind` (the per-tile qkv/ff0pre/attn/n1x/n2x arrays) —
  `[vit::kLayers]` + loops over `vit::kLayers` (lines 638-682). The forward/backward
  layer loops `for (li=0; li<vit::kLayers; ++li)` / `for (li=vit::kLayers-1; li>=0;
  --li)` are already L-general.
- `vittc_backward_tile` per-layer LN slot pointers `gn_n1w/b[li], gn_n2w/b[li]` at
  `(li*4 + {0,1,2,3})*kD` (lines 1119-1126) — ALREADY a loop over `vit::kLayers`.
  Only the two NORM-slot literals (`8`/`9`) needed Edit 2.11.
- `kVitDwMaxTiles = kVitDwPatchTiles + vit::kLayers * kVitDwTilesPerLayer +
  kVitDwHeadTiles` (line 1444-1445) — L-general (patch + per-layer × L + head). The
  split-K partial-scratch carve `vit_dw_part_floats(G)` derives from it. No edit.
- `kVitMuonMaxNumel = vit::kDff * vit::kD` / `kVitMuonMaxRows = vit::kDff`
  (fused_vit_megakernel.cuh 434-435) — d-derived (ff0 [dff,d] is the largest 2D
  weight at every L), so the Muon NS scratch carve is already correct at L=48.
- `kLnVecElems` drives EVERY lnvec workspace carve (megakernel 480/522/554/560/757)
  — they all scale automatically once Edit 2.2 makes `kNumLnVec` L-general.
- The runtime dW work count `n_dw = vittc_dw_total_tiles(spec)` is COUNTED from the
  spec array (line 633/773), NOT a 10-pinned literal — auto-scales to 194 specs at L=48.

# RISKS / NOTES

1. **dW WORKSPACE (HBM) carves are L-general already** — `vit_acts_bf16_count` (acts),
   `vit_dw_part_floats`/`kVitDwMaxTiles` (split-K partials), and `kLnVecElems` (LN
   partials) all loop / derive from `vit::kLayers`. The launcher's per-step dW work
   count is `n_dw = vittc_dw_total_tiles(spec)` (RUNTIME) — so it auto-scales to 194
   specs at L=48. No L-pinned workspace literal remains.

2. **VitTcSmem static-smem budget** — `spec[]` is the ONLY layer-dependent smem
   member (Edit 3.1). `sizeof(VitDwSpec)` = 2 ptrs (16 B) + 6 ints (24 B) = 40 B
   (no padding — 8-byte aligned, total 40 is a multiple of 8). The array grows
   10*40=400 B (L=2) → 194*40=7,760 B (L=48), a +7.36 KB delta. The OTHER VitTcSmem
   members (sA = kVitAtomsPerSlot·64·16 bf16, sB = TILE_N·16 bf16, red[256]) depend on
   TILE_N + the interleave cap, NOT on d or L. VitTcSmem is the SMALL static TC smem
   (NOT the 2.25 MB VitSampleSmem, which is the SCALAR path gated OFF at flagship).
   I did NOT re-measure sizeof at L=48 (read-only; no build) — the lead's 3rd
   gate_command (compile_to_object on the flagship layout) is the real check. If a
   static-smem cap is tripped at L=48, the fix is the SAME as the decoder's: keep
   `spec` in smem (it is built once by thread 0) or route to dynamic smem via the
   existing TC-smem gate. At L=2 the member is byte-identical (spec[10]) → VitTcSmem
   and any sizeof assert are unchanged.

3. **Muon table → formula PTX is NOT byte-identical** (Edits 2.1 + 3.2): the
   `kVitMuon2D` `.const` array becomes the inlined `vit_muon_2d()` accessor, and the
   table-scan `vit_is_muon_2d` becomes a closed form. This is the Muon optimizer cell,
   a DIFFERENT TU/cell from the gated AdamW path (`mega_vit_real_adamw_tc.cu` →
   OptId::AdamW). The L=2 AdamW gate (test_vit_megakernel.py / test_l3tc_tail_gate.py)
   does NOT exercise Muon NS, and the formula returns VALUE-IDENTICAL (tidx,rows,cols)
   + membership at L=2 (proven above). So Muon numerics/determinism are preserved at
   L=2; only the (untested-by-the-AdamW-gate) PTX shape changes. If a Muon-cell
   byte-identity gate exists elsewhere, treat this as a value-identical (not
   PTX-identical) change. NOTE `test_l3tc_tail_gate.py:120` references the
   `kVitMuon2D` ROUTING (which tensors are NS vs AdamW) — that routing is preserved
   exactly by `vit_is_muon_2d` at L=2 (same membership set), so the gate's
   m.parameters() partition is unchanged.

4. **`vit_muon_2d` ordering must match `vit_is_muon_2d` membership** — both derive
   from the same per-layer {0,2,8,10} weight offsets (base=4+12li) + patch/pos/head,
   so P2.7 (orthogonalize the table) and P3 (skip the 2D ones in the AdamW tail) stay
   consistent at any L. Verified identical sets at L=2.

5. **The Muon NS rows/cols pull the right shapes at every L** — `vit_muon_2d` returns
   the SAME (rows,cols) per kind (in_proj [3d,d], out_proj [d,d], ff0 [dff,d], ff2
   [d,dff], patch [d,patch], pos [seq,d], head [V,d]); `kVitMuonMaxNumel = dff·d` and
   `kVitMuonMaxRows = dff` bound the NS scratch for ALL of them at every L (ff0/ff2 are
   the largest), so the per-matrix NS scratch carve (fused_vit_megakernel.cuh 936-942)
   is already big enough at L=48.

6. **Consumer-kernel functional correctness (the goal)** — with Edits 2.x + 3.x the
   ViT TC backward enumerations (`vittc_build_dw_specs`, the dW decode/total/run/
   reduce/biases loops, the LN-vec reduce, and the Muon table) walk `vit::kLayers`
   layers / `kVitNumDwSpecs` specs / `kNumLnVec` LN slots / `kVitNumMuon2D` Muon
   matrices — so the SAME kernel BODY functionally processes all 48 layers when bound
   against `vit_flagship_layout.cuh` (SG_VIT_LAYERS=48), exactly as the decoder
   flagship_dw spec made the decoder process 48 layers. The forward path
   (`vittc_forward_tile`, the `li<vit::kLayers` loop) was ALREADY L-general.

7. **smem / 1-CTA-per-SM fit at flagship dims** — the per-tile scratch `VitTileScratch`
   holds per-layer caches `[vit::kLayers]` (qkv/ff0pre/attn/n1x/n2x), so the per-tile
   HBM scratch (`vit_tc_per_tile_floats`-style carve, lines 654-682) grows ∝ kLayers;
   at d=1664 × L=48 this per-tile region is large but lives in HBM workspace (the host
   sizes it from `vit_acts_bf16_count` + the per-CTA scratch), NOT in smem. The static
   VitTcSmem (sA/sB/red/spec) is the only smem and grows only by the +7.36 KB spec[]
   delta (note 2). This matches the decoder flagship: the layout + enumeration are
   correct and L-general; whether 1-CTA/SM holds at flagship dims is the lead's
   gate-command compile/occupancy probe, a property of the kernel's smem design, not
   of this enumeration change.

8. **No pp_stage equivalent to migrate** — unlike the decoder (whose
   `pp_stage_decoder_tc.cuh` reads `kLnVecTensorIdx` and needed Edit 3.x in
   flagship_dw.md), the ViT has NO `pp_stage_vit_tc.cuh` consuming `kLnVecTensorIdx` /
   `kVitMuon2D` (grep over csrc/tests finds the symbols ONLY in model_stage_vit_tc.cuh
   + fused_vit_megakernel.cuh). So removing the two `__constant__` arrays breaks no
   other TU — both removed symbols (`kLnVecTensorIdx`, `kVitMuon2D`) have their only
   readers migrated by Edits 2.12 + 3.2.

9. **gfx942 / tpu_v6e untouched** — all C++ edits are under `csrc/fused/sm_90/` ViT TC
   headers; the codegen edits add a new sm_90-only emitter + flag. No AMD/TPU tree is
   referenced. The `--dispatch-table-gfx942` path and Pallas/tpu paths are untouched.

10. **Existing `--vit-layout` byte-identity (the 1st gate command)** — verified in
    this session: `python -m grokking_optimizers.megakernel_codegen --vit-layout`
    reproduces the committed `csrc/fused/sm_90/vit_layout.cuh` byte-for-byte on the
    UNMODIFIED tree. Edits 1.2-1.3 add keyword args whose defaults equal the values
    the old code read from globals (and `seq = npatch + 1` with npatch defaulting to
    `_VIT_NPATCH`), so the production d=128 + bench d=2048 bodies stay byte-identical.

# Flagship layout table (computed, for the lead to sanity-check the emitted header)

- `kVitNumTensors = 4 + 12*48 + 4 = 584`.
- `kVitTotalElems = 1,596,200,417` (int64_t — holds 1.6e9).
- Largest per-tensor numel = `dff*d = 6656*1664 = 11,075,584` (ff.0 / ff.2) →
  `kVitMaxTensorNumel` (int) is exact.
- Largest offset = `1,596,200,320 < INT32_MAX (2,147,483,647)` → int32 kVitOffsets /
  kVitSizes are exact, no overflow.
- VitSampleSmem (SCALED smem block) ≈ 2,304,784 B (2250.77 KB) at seq=17, D=1664,
  HEADS=16 — over the 227 KB cap, so the flagship body emits the `else` (scaled) smem
  block WITHOUT the `< 227 KB` assert (the SCALAR megakernel is gated off; the TC
  engine uses the small static VitTcSmem). This is the SAME treatment the d=2048 bench
  branch already gets.

# Gate commands (from the task; for the lead to run)

```bash
cd /workspace/SuperGrok1.5

# (1) byte-identity regression — production d=128 / bench d=2048 header UNCHANGED:
python -m grokking_optimizers.megakernel_codegen --vit-layout > /tmp/v.cuh
diff /tmp/v.cuh csrc/fused/sm_90/vit_layout.cuh          # MUST be empty (byte-identical)

# (2) emit the flagship layout header:
python -m grokking_optimizers.megakernel_codegen --vit-layout-flagship \
    > csrc/fused/sm_90/vit_flagship_layout.cuh

# (3) compile the TC megakernel TU AGAINST the flagship layout (force-include the
#     flagship header + pre-define vit_layout.cuh's guard so its body is skipped;
#     symbol names are identical):
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_vit_real_adamw_tc.cu \
    -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 -DSG_DEC_SCALAR_MEGAKERNEL=0 \
    -DSG_FUSED_SM90_VIT_LAYOUT_CUH_=1 \
    -include csrc/fused/sm_90/vit_flagship_layout.cuh
```

> The `-DSG_FUSED_SM90_VIT_LAYOUT_CUH_=1 -include vit_flagship_layout.cuh` pair is the
> portable, generic way to swap the layout table without editing the TU: it
> force-includes the flagship header first (defining all the kVit* / SG_VIT_* symbols,
> incl. SG_VIT_LAYERS=48) and pre-defines vit_layout.cuh's include guard so its body
> is skipped. The compile-script invocation is GENERIC (no SuperGrok-hardcoding). A
> green compile proves: the flagship layout header + table correctness + the
> L-generalized TC backward (Edits 2.x/3.x) all compile at SG_VIT_LAYERS=48 with the
> new spec[]/Muon/LN counts; combined with the L=2 byte-identity (gate 1) it
> establishes the flagship path functionally walks 48 layers.
>
> GATE NOTE — the task's gate-3 passes `-DSG_DEC_SCALAR_MEGAKERNEL=0`, which is the
> DECODER scalar gate and a no-op for the ViT TU (whose scalar gate is
> `SG_VIT_SCALAR_MEGAKERNEL`, defaulting to 1). So gate-3 as written ALSO compiles the
> LEGACY scalar ViT megakernel at flagship dims. That is harmless for a green compile:
> the flagship body emits the SCALED smem block WITHOUT the `< 227 KB` cap assert, and
> the only scalar-path static_assert (`sizeof(VitSampleSmem) ==
> vit_layout_check::kVitSampleSmemBytes`, fused_vit_megakernel.cuh:136) holds because
> BOTH sides recompute from the same flagship (d,seq,heads) formula (the ~2.25 MB cap
> is a RUNTIME cudaFuncSetAttribute refusal, not a compile error). If the lead prefers
> the decoder-flagship convention (scalar path gated OFF, only the TC path compiled),
> ADD `-DSG_VIT_SCALAR_MEGAKERNEL=0` to gate-3 — it is byte-identical for the TC path
> and matches what `mega_vit_real_adamw_tc.cu` ships at the d=2048 bench width.
