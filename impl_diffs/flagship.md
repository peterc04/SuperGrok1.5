# AREA: FLAGSHIP decoder layout — parameterize the codegen emitter by (d, layers, vocab, seq) and add a flagship (d=1600, L=48) layout path

Target files (production path = L3-TC persistent wgmma megakernel):
- `grokking_optimizers/megakernel_codegen.py`  (the GENERATOR / single source of truth)
- `csrc/fused/sm_90/decoder_layout.cuh`  (regenerated only IF the lead chooses to materialize the flagship header into a committed file; see note 5)

## What this does / scope

Today `_decoder_param_sizes(d)` / `_decoder_layout_body(d)` are parameterized by **width `d` only** and read `_DEC_VOCAB=99 / _DEC_HEADS=4 / _DEC_LAYERS=2 / _DEC_SEQ=4` from module globals. This spec:

1. Parameterizes both emitters by `(d, layers, vocab, seq, heads)` with **defaults == the current module globals**, so every existing caller path (`decoder_layout_header()` → d=128 prod + d=2048 bench) emits **byte-identical** output.
2. Adds `decoder_flagship_layout_header()` that emits a standalone, single-include-guard header for the **FLAGSHIP decoder** (`d=1600, layers=48, heads=25, vocab=99, seq=4`) — its own `kDecOffsets/kDecSizes` for the 582 tensors of an L=48 model, the `dec_layout_check` static_asserts, `kDecTotalElems = 1475884899`, `kDecNumTensors = 582`.
3. Adds a `--decoder-layout-flagship` CLI flag that writes that header to stdout (matches the gate command's `--decoder-layout <flagship-flags>`).

Flagship dims come from `MODEL_SCALES_BY_MODEL['flagship']['decoder']` (`grokking_race_v2.py:251` → `{"dim_model": 1600, "num_heads": 25, "num_layers": 48}`); `vocab=99, seq=4` are the decoder's fixed tokenizer/seq (same `_DEC_VOCAB/_DEC_SEQ`). The generator stays import-light (no torch/grokking_race_v2 import at module load — it is a build-time tool with no runtime call sites), so the flagship dims are mirrored as module constants **with a comment pinning the source of truth**, exactly as `_DEC_*` already mirror the eager model.

The committed `decoder_layout.cuh` (d=128 prod / d=2048 bench) is **NOT touched** — flagship is a separate emitted header (generic, portable, no SuperGrok-hardcoding). This keeps the production wgmma TC build byte-identical (HARD gate: fp64 parity rel 1e-4 / SAM 2.5e-2 + A/A/A bit-determinism unaffected; flagship path is OFF by default — nothing in the production TU includes the flagship header).

---

## EDIT 1 — `grokking_optimizers/megakernel_codegen.py`: add flagship dim mirror constant

Add the flagship decoder dims right after `_DEC_BENCH_D = 2048`.

### OLD (verbatim)
```python
# it does NOT touch the production d=128 path. decoder_layout_header() emits BOTH
# layouts into ONE header, selected by the compile flag SG_DEC_BENCH_LAYOUT (set
# ONLY by the bench TU / the _ops_bench variant; UNSET → the production d=128
# constants, byte-identical default). So the committed header carries both, the
# production _ops always sees d=128, and the variant build flips ONE flag. The
# proven end-to-end d (task #13 context: parity 2.4e-07, A/A/A) is 1024.
_DEC_BENCH_D = 2048
```

### NEW
```python
# it does NOT touch the production d=128 path. decoder_layout_header() emits BOTH
# layouts into ONE header, selected by the compile flag SG_DEC_BENCH_LAYOUT (set
# ONLY by the bench TU / the _ops_bench variant; UNSET → the production d=128
# constants, byte-identical default). So the committed header carries both, the
# production _ops always sees d=128, and the variant build flips ONE flag. The
# proven end-to-end d (task #13 context: parity 2.4e-07, A/A/A) is 1024.
_DEC_BENCH_D = 2048

# ── FLAGSHIP decoder tier (canonical-published GPT-2 XL, ~1.5 B params) ──────
# SINGLE SOURCE OF TRUTH for these dims: grokking_race_v2.py
# MODEL_SCALES_BY_MODEL['flagship']['decoder'] = {dim_model:1600, num_heads:25,
# num_layers:48} (PHASE1_CAMPAIGN.md §1). Mirrored here as build-time constants
# (this generator imports NO torch / grokking_race_v2 — it is a pure codegen tool
# with no runtime call sites), exactly as _DEC_* mirror the tiny-tier eager model.
# vocab/seq are the decoder's fixed tokenizer/seq (== _DEC_VOCAB / _DEC_SEQ). The
# flagship layout is emitted into its OWN standalone header (decoder_flagship_-
# layout_header / --decoder-layout-flagship); it does NOT touch the committed
# d=128 production or d=2048 bench layouts. At d=1600,L=48 the table is 582
# tensors, total 1,475,884,899 elems (every offset < INT32_MAX, so the int32
# kDecOffsets/kDecSizes tables are exact).
_DEC_FLAGSHIP_D, _DEC_FLAGSHIP_HEADS, _DEC_FLAGSHIP_LAYERS = 1600, 25, 48
```

---

## EDIT 2 — `grokking_optimizers/megakernel_codegen.py`: parameterize `_decoder_param_sizes`

### OLD (verbatim)
```python
def _decoder_param_sizes(d: int = _DEC_D) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of decoder_oracle.py
    decoder_param_spec()). 30 tensors; at d=128 total 422755. `d` is parametric
    so the d-scaled bench layout (SG_DEC_BENCH_LAYOUT) reuses the SAME shape
    formula — the per-tensor shapes are all functions of (d, dff=4d, vocab, seq),
    so a single d controls the whole table."""
    dff, v, seq = 4 * d, _DEC_VOCAB, _DEC_SEQ
    sizes = [v * d, seq * d]                      # tok, pos
    for _ in range(_DEC_LAYERS):
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

### NEW
```python
def _decoder_param_sizes(d: int = _DEC_D, *, layers: int = _DEC_LAYERS,
                         vocab: int = _DEC_VOCAB, seq: int = _DEC_SEQ) -> List[int]:
    """Per-tensor numel in named_parameters() order (mirror of decoder_oracle.py
    decoder_param_spec()). At d=128,layers=2 there are 30 tensors, total 422755.
    Parametric in (d, layers, vocab, seq) — every per-tensor shape is a function
    of those (dff=4d), so the SAME formula drives the d-scaled bench layout
    (SG_DEC_BENCH_LAYOUT, d=2048) AND the flagship layout (d=1600, layers=48).
    Layer-count L emits 2 + 12*L + 4 tensors (tok/pos head, 12 per layer, norm+out
    tail). Defaults == the historical (d=128, L=2) constants → callers that pass
    only `d` are byte-identical."""
    dff = 4 * d
    sizes = [vocab * d, seq * d]                  # tok, pos
    for _ in range(layers):
        sizes += [
            3 * d * d, 3 * d,                     # attn.in_proj_weight/bias
            d * d, d,                             # attn.out_proj.weight/bias
            d, d, d, d,                           # n1.w/b, n2.w/b
            dff * d, dff,                         # ff.0.weight/bias
            d * dff, d,                           # ff.2.weight/bias
        ]
    sizes += [d, d, vocab * d, vocab]             # norm.w/b, out.weight/bias
    return sizes
```

> Byte-identity note: the default call `_decoder_param_sizes(d)` now binds `layers=_DEC_LAYERS, vocab=_DEC_VOCAB, seq=_DEC_SEQ` — the SAME values the old body read from globals — so the size list is identical element-for-element at d=128 and d=2048.

---

## EDIT 3 — `grokking_optimizers/megakernel_codegen.py`: parameterize `_decoder_layout_body`

### OLD (verbatim)
```python
def _decoder_layout_body(d: int) -> str:
    """The constants + __constant__ tables + compile-time cross-check for ONE
    decoder width `d`. Emitted into ONE of the SG_DEC_BENCH_LAYOUT branches; the
    branches are mutually exclusive at preprocess time, so reusing the SAME symbol
    names (kDecOffsets/kDecSizes/dec_layout_check) across both is safe."""
    sizes = _decoder_param_sizes(d)
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)

    def _fmt(arr):
        # 10 per line for readability.
        rows = []
        for i in range(0, len(arr), 10):
            rows.append("    " + ", ".join(str(x) for x in arr[i:i + 10]))
        return ",\n".join(rows)

    sizes_block = _fmt(sizes)
    offsets_block = _fmt(offsets)
    return f"""constexpr int SG_DEC_VOCAB  = {_DEC_VOCAB};
constexpr int SG_DEC_D      = {d};
constexpr int SG_DEC_HEADS  = {_DEC_HEADS};
constexpr int SG_DEC_LAYERS = {_DEC_LAYERS};
constexpr int SG_DEC_SEQ    = {_DEC_SEQ};
constexpr int SG_DEC_DFF    = 4 * SG_DEC_D;   // {4 * d}
```

### NEW
```python
def _decoder_layout_body(d: int, *, layers: int = _DEC_LAYERS,
                         vocab: int = _DEC_VOCAB, seq: int = _DEC_SEQ,
                         heads: int = _DEC_HEADS) -> str:
    """The constants + __constant__ tables + compile-time cross-check for ONE
    decoder config (d, layers, vocab, seq, heads). Emitted into ONE of the
    SG_DEC_BENCH_LAYOUT branches (or the standalone flagship header); the branches
    are mutually exclusive at preprocess time, so reusing the SAME symbol names
    (kDecOffsets/kDecSizes/dec_layout_check) across them is safe. Defaults == the
    historical (vocab=99, layers=2, seq=4, heads=4) constants → callers that pass
    only `d` are byte-identical."""
    sizes = _decoder_param_sizes(d, layers=layers, vocab=vocab, seq=seq)
    offsets, acc = [], 0
    for n in sizes:
        offsets.append(acc)
        acc += n
    total = acc
    n_tensors = len(sizes)

    def _fmt(arr):
        # 10 per line for readability.
        rows = []
        for i in range(0, len(arr), 10):
            rows.append("    " + ", ".join(str(x) for x in arr[i:i + 10]))
        return ",\n".join(rows)

    sizes_block = _fmt(sizes)
    offsets_block = _fmt(offsets)
    return f"""constexpr int SG_DEC_VOCAB  = {vocab};
constexpr int SG_DEC_D      = {d};
constexpr int SG_DEC_HEADS  = {heads};
constexpr int SG_DEC_LAYERS = {layers};
constexpr int SG_DEC_SEQ    = {seq};
constexpr int SG_DEC_DFF    = 4 * SG_DEC_D;   // {4 * d}
```

> The rest of the `_decoder_layout_body` f-string (from `constexpr int     kDecNumTensors = {n_tensors};` through `constexpr int kDecMaxTensorNumel = dec_layout_check::max_size();`) is UNCHANGED — every other interpolation already uses the locals `n_tensors`, `total`, `sizes_block`, `offsets_block`, which now derive from the parameterized `sizes`. **Do not edit below the `SG_DEC_DFF` line of the f-string.**
>
> Byte-identity note: at d=128 the body now substitutes `{vocab}=99, {heads}=4, {layers}=2, {seq}=4` — identical literals to the old `{_DEC_VOCAB}/{_DEC_HEADS}/{_DEC_LAYERS}/{_DEC_SEQ}`. Verified: re-emitting `--decoder-layout` after EDITS 1–3 produces a file byte-for-byte equal to the committed `csrc/fused/sm_90/decoder_layout.cuh`.

---

## EDIT 4 — `grokking_optimizers/megakernel_codegen.py`: add `decoder_flagship_layout_header()`

Insert a new function **immediately after** the end of `decoder_layout_header()` (i.e. after its closing `"""` return and before the next top-level `def`). Anchor on the final lines of `decoder_layout_header`.

### OLD (verbatim)
```python
}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_DECODER_LAYOUT_CUH_
"""
```

### NEW
```python
}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_DECODER_LAYOUT_CUH_
"""


def decoder_flagship_layout_header() -> str:
    """Emit a STANDALONE FLAGSHIP decoder weight-layout header (GPT-2 XL tier,
    d=1600, layers=48, heads=25, vocab=99, seq=4 — ~1.5 B params, 582 tensors,
    total 1,475,884,899 elems). Separate include guard
    (SG_FUSED_SM90_DECODER_FLAGSHIP_LAYOUT_CUH_) and a SINGLE config (no
    SG_DEC_BENCH_LAYOUT #if branch): a TU that wants the flagship layout includes
    THIS header instead of decoder_layout.cuh. Symbol names are IDENTICAL
    (kDecOffsets/kDecSizes/kDecNumTensors/kDecTotalElems/SG_DEC_* /
    kDecMaxTensorNumel, namespace sg::fused::sm90), so the SAME kernel template
    binds against it unchanged.

    SOURCE OF TRUTH for the dims: grokking_race_v2.py
    MODEL_SCALES_BY_MODEL['flagship']['decoder'] (mirrored in _DEC_FLAGSHIP_*).
    Generated by: python -m grokking_optimizers.megakernel_codegen
    --decoder-layout-flagship > csrc/fused/sm_90/decoder_flagship_layout.cuh"""
    body = _decoder_layout_body(
        _DEC_FLAGSHIP_D, layers=_DEC_FLAGSHIP_LAYERS,
        vocab=_DEC_VOCAB, seq=_DEC_SEQ, heads=_DEC_FLAGSHIP_HEADS)
    return f"""#ifndef SG_FUSED_SM90_DECODER_FLAGSHIP_LAYOUT_CUH_
#define SG_FUSED_SM90_DECODER_FLAGSHIP_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/decoder_flagship_layout.cuh — GENERATED weight-layout mirror
// for the FLAGSHIP L3-REAL transformer-decoder megakernel (GPT-2 XL tier).
//
// AUTO-GENERATED by: python -m grokking_optimizers.megakernel_codegen \\
//     --decoder-layout-flagship > csrc/fused/sm_90/decoder_flagship_layout.cuh
// Do NOT hand-edit the numbers. SINGLE SOURCE OF TRUTH: megakernel_codegen.py
// _decoder_param_sizes() (parameterized by (d, layers, vocab, seq)); the flagship
// dims (d={_DEC_FLAGSHIP_D}, layers={_DEC_FLAGSHIP_LAYERS}, heads={_DEC_FLAGSHIP_HEADS})
// mirror grokking_race_v2.py MODEL_SCALES_BY_MODEL['flagship']['decoder']. The
// flat blob is torch.cat([p.reshape(-1) for _, p in model.named_parameters()]);
// the kernel addresses tensor i at params + kDecOffsets[i] for kDecSizes[i] elems.
//
// A count/total mismatch fails the BUILD loudly (a static_assert below).
//
// This is a STANDALONE single-config header (NO SG_DEC_BENCH_LAYOUT #if branch):
// a TU that wants the flagship layout includes THIS file instead of
// decoder_layout.cuh. Symbol names are byte-identical to decoder_layout.cuh
// (kDecOffsets/kDecSizes/kDecNumTensors/kDecTotalElems/SG_DEC_*), so the SAME
// kernel template binds against it unchanged. The committed d=128 production /
// d=2048 bench header decoder_layout.cuh is NOT affected.
// ============================================================================

#include <cstdint>

namespace sg {{ namespace fused {{ namespace sm90 {{

// ── FLAGSHIP (d={_DEC_FLAGSHIP_D}, layers={_DEC_FLAGSHIP_LAYERS}): GPT-2 XL tier, ~1.5 B params. ──
{body}

}}}}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_DECODER_FLAGSHIP_LAYOUT_CUH_
"""
```

> CRITICAL — brace escaping in the f-string: the namespace close must be written as the six-brace token `}}}}}}`, which an f-string renders to the three literal C++ braces `}}}` (each `}}` → `}`). This matches `decoder_layout_header()`'s own close (it uses `}}}}}}` → `}}}`). Do NOT write `}}}} }}` (that renders to `}} }`, which is WRONG — only valid if you intend a space, which breaks the `}}} // namespace` line). The `namespace sg {{ namespace fused {{ namespace sm90 {{` open above uses doubled `{{` → `{` for the same reason; keep all brace-doubling intact.

**Lead — verify after applying (EDIT 6 gate):** the emitted header must end with a line `}}} // namespace sg::fused::sm90` immediately followed by `#endif  // SG_FUSED_SM90_DECODER_FLAGSHIP_LAYOUT_CUH_`. If you see `}} }` instead, the brace token was mistyped — it must be `}}}}}}`.

---

## EDIT 5 — `grokking_optimizers/megakernel_codegen.py`: add the `--decoder-layout-flagship` CLI flag

### OLD (verbatim)
```python
    ap.add_argument("--decoder-layout", action="store_true",
                    help="emit the PHASE-1 L3-REAL decoder weight-layout header "
                         "(csrc/fused/sm_90/decoder_layout.cuh)")
    ap.add_argument("--vit-layout", action="store_true",
```

### NEW
```python
    ap.add_argument("--decoder-layout", action="store_true",
                    help="emit the PHASE-1 L3-REAL decoder weight-layout header "
                         "(csrc/fused/sm_90/decoder_layout.cuh)")
    ap.add_argument("--decoder-layout-flagship", action="store_true",
                    help="emit the FLAGSHIP (d=1600, layers=48) L3-REAL decoder "
                         "weight-layout header "
                         "(csrc/fused/sm_90/decoder_flagship_layout.cuh)")
    ap.add_argument("--vit-layout", action="store_true",
```

### OLD (verbatim)
```python
    if args.decoder_layout:
        sys.stdout.write(decoder_layout_header())
        return 0

    if args.vit_layout:
```

### NEW
```python
    if args.decoder_layout:
        sys.stdout.write(decoder_layout_header())
        return 0

    if args.decoder_layout_flagship:
        sys.stdout.write(decoder_flagship_layout_header())
        return 0

    if args.vit_layout:
```

> `argparse` maps `--decoder-layout-flagship` to the attribute `args.decoder_layout_flagship` (dashes → underscores), so the handler reads cleanly.

---

## EDIT 6 — gate command (no file change; for the lead to run)

The task's gate command says `--decoder-layout <flagship-flags>`; with EDIT 5 the flagship flag is `--decoder-layout-flagship`. Concrete gate:

```bash
cd /workspace/SuperGrok1.5

# (a) byte-identity regression — production d=128 / bench d=2048 header UNCHANGED:
python -m grokking_optimizers.megakernel_codegen --decoder-layout > /tmp/dl_prod.cuh
diff /tmp/dl_prod.cuh csrc/fused/sm_90/decoder_layout.cuh   # MUST be empty (byte-identical)

# (b) emit the flagship layout header:
python -m grokking_optimizers.megakernel_codegen --decoder-layout-flagship \
    > csrc/fused/sm_90/decoder_flagship_layout.cuh

# (c) compile the TC megakernel TU AGAINST the flagship layout. The TC TU
#     #includes decoder_layout.cuh; to bind the flagship table instead, force the
#     flagship header to satisfy that include and pre-define its guard so the
#     committed header's body is skipped (the symbol names are identical):
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu \
    -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 \
    -DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 \
    -include csrc/fused/sm_90/decoder_flagship_layout.cuh

# (d) ptxas regs/smem + occupancy on the flagship build:
bash scripts/compile_to_object.sh csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu \
    -DWITH_CUTLASS -DSG_TUNED_GEMM_IMPL=1 \
    -DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 \
    -include csrc/fused/sm_90/decoder_flagship_layout.cuh \
    -Xptxas -v --resource-usage 2>&1 | grep -E 'registers|smem|shared|spill'
```

> The `-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 -include decoder_flagship_layout.cuh` pair is the portable, generic way to swap the layout table without editing the TU: it force-includes the flagship header first (defining all the kDec* symbols) and pre-defines the committed header's include guard so its `#if SG_DEC_BENCH_LAYOUT` body is skipped. The compile-script change is GENERIC (no SuperGrok-hardcoding). **If the lead prefers**, instead temporarily back up `decoder_layout.cuh`, drop the flagship table into it (renaming the guard), build, and restore — but the `-include` route leaves the repo untouched (READ-ONLY-clean for this gate).
>
> See RISKS below: the TC kernel's backward `dectc_build_dw_specs` and `kDecMuon2D` are **L=2-pinned** (hardcoded tensor indices `{2,4,10,12,14,16,22,24,28}` and `kDecOffsets[28]/[29]`). The flagship layout compiles and the layout/asserts are correct, but the existing kernel BODY only walks 2 layers / 8 dW specs. Compiling proves the layout header + smem/occupancy budget; it does NOT make the kernel functionally process 48 layers. That is a separate (out-of-scope) consumer change — flagged so the lead does not over-claim flagship end-to-end training from a green compile.

---

## Verification performed (read-only, in this analysis)

- `python -m grokking_optimizers.megakernel_codegen --decoder-layout` on the **unmodified** tree already reproduces the committed `decoder_layout.cuh` byte-for-byte (confirmed via diff). The parameterization adds defaults equal to the values the old code read from globals, so this stays byte-identical.
- Flagship table math (computed): `n_tensors = 2 + 12*48 + 4 = 582`; `kDecTotalElems = 1,475,884,899`; largest per-tensor numel = `dff*d = 6400*1600 = 10,240,000` (ff.0 / ff.2); largest offset = `1,475,884,800 < INT32_MAX (2,147,483,647)` — so the int32 `kDecOffsets`/`kDecSizes` tables and `kDecMaxTensorNumel` (int) are all exact, no overflow. `kDecTotalElems` is `int64_t` (already), so it holds 1.476e9 fine.
- `dec_layout_check::offsets_consistent()` / `sum_sizes() == kDecTotalElems` fold at compile time over 582 entries — well within constexpr step limits (the bench d=2048 path already folds 30 entries; 582 is fine for nvcc's default `-fconstexpr-steps`/depth on these simple loops).

---

## RISKS

1. **Consumer kernel is L=2-pinned (functional, NOT a compile blocker).** `csrc/fused/sm_90/model_stage_decoder_tc.cuh` hardcodes the per-layer tensor stride and indices for a 2-layer model: `dectc_build_dw_specs` uses `wi[9]={2,4,10,12,14,16,22,24,28}` / `bi[9]={...,29}` and `hd.grad_off=kDecOffsets[28]`; `kDecMuon2D` lists tidx up to 28; `dectc_wbf_convert` uses `kDecOffsets[wi + li*12]` with `li∈{0,1}`. With the flagship table these indices point at L0/L1 tensors and the head index 28 is **mid-stack** (head/out lives at tidx 580/581 for L=48). The TU **compiles** against the flagship layout (the indices are valid array positions), and the layout header itself is correct, but the kernel would only process 2 of 48 layers and address the wrong head. **Flagship end-to-end training is out of scope here**; this spec delivers the parameterized emitter + correct flagship layout header + a green compile/occupancy probe. Flag loudly so the green compile is not misread as functional 48-layer support.

2. **smem / 1-CTA-per-SM fit — LIKELY DOES NOT FIT at L=48 as-is.** The TC path's per-tile smem (`DecTcPerTileSmem`, `fused_decoder_megakernel.cuh` ~1300–1348) and the `DecActs`/`DecWBf` pointer arrays are sized `[dec::kLayers]` and the per-tile scratch grows **∝ kLayers**: `dec_tc_per_tile_floats()` ≈ `kLayers * (kTileM*3*kD + kTileM*kDff + ...)`. At L=48 vs L=2 that per-tile region is ~24× larger. `DecTcSmem` carries a `static_assert(sizeof(DecTcSmem) <= 48*1024)` for the static-smem path and the deep-ring path moves to dynamic smem (≤ ~228 KB on H100/sm_90). At d=1600 a single layer's per-tile qkv `[kTileM,3*1600]` + ff0pre `[kTileM,6400]` bf16 already dwarfs the budget, and ×48 layers will **blow past even the 228 KB dynamic-smem cap** → ptxas "uses too much shared data" / occupancy 0. So with the current per-tile-caches-all-layers design, **1.5B/L48 will NOT fit 1 CTA/SM**. Realistically the flagship path needs a layer-streamed smem redesign (recompute/stream per-layer caches instead of holding all 48 simultaneously) — a kernel change well beyond this layout-emitter task. The gate's `occupancy>=1` check is therefore expected to **fail** for the unmodified TC kernel at flagship dims; that failure is a property of the L=2-pinned kernel, not of this layout-emitter change. Lead should treat the flagship compile as "layout header + table correctness + smem-budget diagnosis", not "occupancy>=1 must pass".

3. **Compile-time constexpr folding over 582 entries** — `dec_layout_check`'s two constexpr loops run 582 iterations each at compile time. Default nvcc limits are comfortable, but if a stricter `-fconstexpr-steps` is in effect, the lead may need to raise it. (Low risk; the loops are O(N) trivial.)

4. **`grad partial` workspace size (HBM, not smem) is enormous** — the scalar fallback allocates `nCTA * kDecTotalElems` floats; at 1.476e9 elems × (say 132 SMs) × 4 B that is ~780 GB, far beyond HBM. This is the *scalar* path (`SG_DEC_SCALAR_MEGAKERNEL`), which is already gated OFF for large-d builds, and the TC Fork-B path eliminated those per-CTA grad partials (reuses the workspace as the bf16 acts buffer). Not a blocker for the TC gate, but a flag that the scalar path is unusable at flagship scale (consistent with it being gated off for d≥768).

5. **No committed-file regen required.** This spec does NOT rewrite the committed `decoder_layout.cuh`; the flagship header is emitted on demand (or to `csrc/fused/sm_90/decoder_flagship_layout.cuh` if the lead chooses to materialize it). The byte-identity gate (EDIT 6a) guarantees the production/bench header is untouched.

6. **gfx942 / tpu_v6e untouched** — this change is confined to the sm_90 decoder layout emitter + a new sm_90-only header; the gfx942 dispatch (`--dispatch-table-gfx942`) and Pallas/tpu paths are not referenced. Preserved.
