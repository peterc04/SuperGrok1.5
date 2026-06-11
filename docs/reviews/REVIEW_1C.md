# Stage-1C Correctness Review — decoder/ViT tensor-core GEMMs

Reviewer: Opus 4.8 (max effort). Branch `claude/custom-optimizer-analysis-HFYhg`.
Scope: 5 decoder + 6 ViT matmuls routed to Sm90 tensor cores via the new
`mma::sm90_run_gemm_bt` helper, plus the `#ifdef WITH_CUTLASS` gating and the
`from_float` build fix. Read-mostly; fix only clear bugs.

**Verdict: ALL 11 matmuls CORRECT. No bugs found. No source changes made.**

---

## 1. The generic helper `mma::sm90_run_gemm_bt` (csrc/backends/cuda/sm_90/mma.cuh:193-284)

**CORRECT.** Compared line-by-line against the proven `fmha_sm90_gemm`
(attention_sm90.cuh:204-280):

| Aspect | fmha_sm90_gemm | sm90_run_gemm_bt | Match |
|---|---|---|---|
| TileShape | `_128,_128,_64` | `_128,_128,_64` | ✓ |
| ClusterShape | `_1,_1,_1` | `_1,_1,_1` | ✓ |
| LayoutA / LayoutC | RowMajor / RowMajor | RowMajor / RowMajor | ✓ |
| LayoutB | `LayoutBT` (param) | `LayoutBT` (param) | ✓ |
| Elem A/B / Acc / C | In / float / float | In / float / float | ✓ |
| Align A/B/C | 128/bits | 128/bits | ✓ |
| CollectiveBuilder args (epilogue+mainloop) | identical | identical | ✓ |
| StageCountAutoCarveout / KernelScheduleAuto | yes | yes | ✓ |
| stride_a/b/c dims | `{M,K,1}/{N,K,1}/{M,N,1}` | `{M,K,1}/{N,K,1}/{M,N,1}` | ✓ |
| Args (mode, {M,N,K,1}, alpha=1/beta=0, C aliased as src+dst) | identical | identical | ✓ |
| can_implement→NotSupported, ws query, initialize→Unknown, run | identical | identical | ✓ |

Only differences are cosmetic: the cached-workspace function name
(`sm90_get_workspace` vs `fmha_get_workspace`, mma.cuh:66) and single- vs
multi-line error-check style. No divergence that affects results or runtime
behaviour. `cudaErrorNotSupported` is returned on un-implementable shapes,
correctly triggering the caller's fallback.

**Layout semantics validated by the reference**: the FMHA path is the known-good
oracle. It computes `S[i,j]=Σ_d Q[i,d]·K[j,d]` (i.e. `A·Bᵀ`) using
`LayoutBT=ColumnMajor` with K stored row-major `[N,D]` (attention_sm90.cuh:384-388),
and `O=P·V` (`A·B`) using `RowMajor` with V row-major `[K,N]` (L400-403). The
new helper inherits this convention verbatim, so:
- `ColumnMajor` + B physically row-major `[N,K]` ⇒ `C = A·Bᵀ` (the `x·Wᵀ` Linear convention). ✓
- `RowMajor` + B physically row-major `[K,N]` ⇒ `C = A·B`. ✓ (unused by 1C; all 11 use ColumnMajor.)

---

## 2. Decoder matmuls (transformer_decoder_sm90.cuh) — all `decoder_linear_gemm<WeightT, ColumnMajor>` ⇒ `C = x·Wᵀ`

`decoder_linear_gemm` (L150-165): GEMM into FP32 scratch, then
`gemm_cast_f32_kernel` casts back to T (no bias — bias added separately by
`bias_add_kernel`). Scratch is `thread_local`, reused; GEMM→cast→next-GEMM are
all enqueued in order on one stream, so reuse is safe.

| # | Matmul | Line | M | N | K | Weight shape | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | QKV proj | 911 | B·S | 3D | D | qkv_W `[3D,D]`=[N,K] | **CORRECT** |
| 2 | Output proj | 946 | B·S | D | D | out_W `[D,D]` | **CORRECT** |
| 3 | FFN up | 970 | B·S | FH | D | ff1_W `[FH,D]` | **CORRECT** |
| 4 | FFN down | 996 | B·S | D | FH | ff2_W `[D,FH]` | **CORRECT** |
| 5 | Vocab head | 1030 | B·S | V | D | vocab_W `[V,D]` | **CORRECT** |

Each call's M/N/K exactly matches the cuBLAS `#else` fallback directly below it
(L918, 953, 977, 1003, 1037), which uses `cublas_gemm_rm(OP_N, OP_T, M, N, K, A[lda=K], W[ldb=K], C[ldc=N])`
= `x·Wᵀ`. **Both branches compute identical math** (FP32 accumulate via
`CUBLAS_COMPUTE_32F` / CUTLASS `ElementAcc=float`; output cast to T). Bias is
added after the GEMM in both paths via `bias_add_kernel` (correct `[N]`-indexed
column add). No epsilon/dim issues: subsequent LN/GELU consume the bias-added
buffer of matching shape.

---

## 3. ViT matmuls (vit_sm90.cuh) — all via `launch_linear_bias` → `vit_linear_gemm_bias<T>` (ColumnMajor) ⇒ `out = x·Wᵀ + b`

`vit_linear_gemm_bias` (L388-403): GEMM into FP32 scratch, then
`gemm_cast_bias_kernel` casts back **and fuses the bias** in the same pass
(`b[i % N]` = correct column index for row-major `[M,N]`, L361). Dispatch
`launch_linear_bias` (L407-419) gates on `cutlass_gemm_supported<ActT> &&
is_same<ActT,WeightT>`; on `cudaErrorNotSupported` it falls through to the
scalar `launch_gemm_bias`.

| # | Matmul | Line | M | N | K | Weight shape | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Patch embed | 946 | B·nP | d_model | patch_dim | patch_W `[d_model,patch_dim]` | **CORRECT** |
| 2 | QKV proj | 973 | B·S | 3·HD | d_model | qkv_W `[3HD,d_model]` | **CORRECT** |
| 3 | Output proj | 1024 | B·S | d_model | HD | out_W `[d_model,HD]` | **CORRECT** |
| 4 | MLP up | 1047 | B·S | ffn_hidden | d_model | ff1_W `[ffn_hidden,d_model]` | **CORRECT** |
| 5 | MLP down | 1063 | B·S | d_model | ffn_hidden | ff2_W `[d_model,ffn_hidden]` | **CORRECT** |
| 6 | Head | 1112 | batch | n_classes | d_model | head_W `[n_classes,d_model]` | **CORRECT** |

Scalar oracle `gemm_bias_kernel` (L319-337) computes
`out[m,n]=Σ_k W[n,k]·in[m,k] + b[n]` = `x·Wᵀ + b` with FP32 accumulate — exactly
what the CUTLASS path (GEMM `A·Wᵀ` + fused-bias cast) computes. **Both branches
identical math.** No GEMM aliases its own input/output; the q/k/v and attn
scratch reuse (proj_out/ln1_in/ln1_out/ffn_h_pre) is consumed before being
overwritten and is unrelated to the GEMM conversion (pre-existing).

---

## 4. `#ifdef WITH_CUTLASS` gating

**CORRECT.**
- Tensor-core path selected only when `WITH_CUTLASS` **and** element is half/bf16:
  decoder gates `if constexpr (cutlass_gemm_supported<WeightT>::value)` (L910 etc.);
  ViT gates `cutlass_gemm_supported<ActT> && is_same<ActT,WeightT>` (L412). FP32
  (`cutlass_gemm_elem<float>` is undefined — primary template only) never
  instantiates a float CUTLASS GEMM. ✓
- `ActT == WeightT` for every instantiation (decoder.cu / vit.cu dtype matrix:
  {float,bf16,half} tied), so the `(WeightT*)` casts of activation pointers in
  the decoder are type-safe. ✓
- `#else` fallback correct and builds **without** `WITH_CUTLASS`: verified by
  the no-cutlass compile gate (both TUs `COMPILE_OK`). Decoder falls to cuBLAS;
  ViT falls to scalar `launch_gemm_bias` (the entire `launch_linear_bias`
  dispatch wrapper is itself inside `#ifdef WITH_CUTLASS`, so the forward's
  `#else` correctly calls `launch_gemm_bias` directly). ✓

---

## 5. The `from_float<T>` forward-declaration build fix (transformer_decoder_sm90.cuh:113)

**CORRECT — fixes a genuine ordering issue, masks nothing.** The primary
template is forward-declared at L113; `gemm_cast_f32_kernel` (first user, L118)
needs it visible. Full definition + `__half`/`__nv_bfloat16` specializations are
at L202-207. All instantiations of `gemm_cast_f32_kernel<T>` occur at the
`decoder_linear_gemm` call sites (L911+), which are after L207, so the
specializations are visible at every point of instantiation (no hidden
`static_cast<T>(float)` fallback for bf16/half). Standard, legal C++; no ODR
hazard (`__forceinline__` template specializations). Confirmed by clean compile
of all three decoder dtype instantiations under `-DWITH_CUTLASS`.

---

## 6. Numerical hazards

No issues. Bias fused/added correctly (column index `n % N`); FP32-accumulate on
both paths; output cast `from_float`/`from_f32` round-to-nearest; FP32 scratch
sized `M·N` and grid covers all elements; no dim mismatch between GEMM output and
the downstream bias_add / LayerNorm / GELU (all consume `[M,N]` of matching
extent). LayerNorm eps (1e-5) untouched by the GEMM conversion.

---

## Gate results (run as-is; no fixes needed)

- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/decoder.cu -DWITH_CUTLASS` → **COMPILE_OK**
- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/vit.cu -DWITH_CUTLASS` → **COMPILE_OK**
- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/decoder.cu` (no-cutlass) → **COMPILE_OK**
- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/models/vit.cu` (no-cutlass) → **COMPILE_OK**
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` → **137 passed, 1 failed (137/1)**
- `ruff check grokking_optimizers/` → **All checks passed!**

No SUSPECT/unprovable items requiring a 🟡 entry in HARDWARE_VALIDATION.md.
