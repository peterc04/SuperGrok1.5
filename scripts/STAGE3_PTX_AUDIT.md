# STAGE 3.0 PTX Audit: Redundancy Analysis

**Date:** 2026-06-01  
**Build Mode:** `--use_fast_math` enabled (setup.py:276, compile.py verified)  
**Scope:** All hand-written inline-PTX (`asm volatile`/`asm(`) blocks in the codebase

---

## Summary

With `--use_fast_math`, nvcc **automatically** lowers transcendental/approx operations to PTX:
- `expf(x)` → `ex2.approx.f32` (after log2(e) scale) in a single instruction
- `sqrtf(x)`, `rsqrtf(x)` → `sqrt.approx.f32`, `rsqrt.approx.f32`
- `logf(x)` → `lg2.approx.f32` (after bit tricks or trig)
- `tanhf(x)`, `__sincosf(x)` → approximate forms

Any hand-written PTX block that merely re-derives this is **REDUNDANT** (dead duplication).

**Key Finding:** Several functions in `ptx_intrinsics.cuh` are **defined but never called** (dead code).

---

## Detailed Audit

### 1. **softplus_ptx** — REDUNDANT (never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh:70–87`

**PTX Opcodes (7):**
```ptx
mul.f32 t, %1, 0f3FB8AA3B;         # x * log2(e)
ex2.approx.f32 ex, t;               # exp(x)
add.f32 ep1, ex, 0f3F800000;        # 1 + exp(x)
lg2.approx.f32 lg, ep1;             # log2(1+exp(x))
mul.f32 lg, lg, 0f3F317218;         # * ln(2)
setp.gt.f32 p, %1, 0f41A00000;      # x > 20?
selp.f32 %0, %1, lg, p;             # branchless select
```

**Equivalent libm:** `logf(1.0f + expf(x))` with saturation at x > 20

**Classification:** **REDUNDANT**  
With `--use_fast_math`, the C++ sequence `logf(1.0f + expf(x))` already compiles to:
- `expf(x)` → `ex2.approx.f32 ex, t` (after log2(e) scaling)
- `logf(...)` → `lg2.approx.f32 lg, ...` (after base conversion)
- The compiler fuses these into the same PTX pipeline without the manual wrapper.
- The saturation (selp for x > 20) is the only non-standard part, but it's small and the caller rarely needs it.

**Call Sites:** **0 (never called)**
- Grep shows only definition and HIP fallback in `ptx_intrinsics.cuh:172–174`.
- No references elsewhere in codebase.

**Action:** Delete function and its HIP fallback. Calling code (if any) falls back to `logf(1.0f + expf(x))`.

---

### 2. **fast_exp_ptx** — REDUNDANT (never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh:94–105`

**PTX Opcodes (2):**
```ptx
mul.f32 t, %1, 0f3FB8AA3B;      # x * log2(e)
ex2.approx.f32 %0, t;            # exp(x) via 2^t
```

**Equivalent libm:** `expf(x)` or `__expf(x)`

**Classification:** **REDUNDANT**  
With `--use_fast_math`, the C++ `expf(x)` or `__expf(x)` already compiles to exactly this PTX sequence.
This is a pure wrapper with zero performance difference. The comment claims "1 cycle vs ~8", but `__expf` is already an intrinsic that uses the approx form under `--use_fast_math`.

**Call Sites:** **0 (never called)**
- Only definition and HIP fallback (`expf(x)`) in `ptx_intrinsics.cuh`.
- Actual exp calls in the codebase use `ptx_expf` (see next), not `fast_exp_ptx`.

**Action:** Delete function and its HIP fallback.

---

### 3. **stochastic_round_ptx** — REDUNDANT (never called, plus design problem)
**Location:** `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh:112–131`

**PTX Opcodes (8):**
```ptx
cvt.rmi.f32.f32 fl, %1;          # floor(x)
sub.f32 frac, %1, fl;            # fractional part
cvt.rn.f32.u32 r, %2;            # random bits → float
mul.f32 r, r, 0f2F800000;        # normalize to [0,1)
setp.lt.f32 p, r, frac;          # r < frac?
cvt.rzi.s32.f32 ifl, fl;         # floor as int
selp.s32 up, 1, 0, p;            # branchless round-up
add.s32 %0, ifl, up;             # result
```

**Equivalent libm:** Stochastic rounding with branching:
```cpp
float fl = floorf(x);
float frac = x - fl;
float r = (float)rand_bits * (1.0f / 4294967296.0f);
return (int)fl + (r < frac ? 1 : 0);
```

**Classification:** **REDUNDANT**  
The branchless `selp` construction is clever, but it's **never used** in the codebase. The HIP fallback (line 180–185) is the only reference to stochastic rounding, and it uses the branching version.

**Call Sites:** **0 (never called)**
- No callsites outside of the definition block.

**Design Issue:** The PTX version claims to be branchless, but it's dead code. If it *were* used, the branchless selp would be LOAD-BEARING. But since it isn't, deleting it is safe.

**Action:** Delete function and its HIP fallback.

---

### 4. **gru_gates_ptx** — REDUNDANT (never called, dual-pipeline is clever but unused)
**Location:** `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh:139–162`

**PTX Opcodes (12):**
```ptx
add.f32 nz, %2, %3;              # wxz + bz
add.f32 nr, %4, %5;              # wxr + br
neg.f32 nz, nz;                  # -(wxz + bz)
neg.f32 nr, nr;                  # -(wxr + br)
mul.f32 tz, nz, 0f3FB8AA3B;      # * log2(e)
mul.f32 tr, nr, 0f3FB8AA3B;      # * log2(e)
ex2.approx.f32 ez, tz;           # exp(-(wxz+bz))
ex2.approx.f32 er, tr;           # exp(-(wxr+br))
add.f32 dz, ez, 0f3F800000;      # 1 + exp(-)
add.f32 dr, er, 0f3F800000;      # 1 + exp(-)
rcp.approx.f32 %0, dz;           # 1 / (1 + exp(-)) = sigmoid
rcp.approx.f32 %1, dr;           # (second sigmoid)
```

**Equivalent libm:**
```cpp
z_out = 1.0f / (1.0f + expf(-(wx_z + bz)));
r_out = 1.0f / (1.0f + expf(-(wx_r + br)));
```

**Classification:** **REDUNDANT**  
The hand-written PTX interleaves two sigmoid computations to fill both FMA pipelines (the comment on line 136 explicitly claims this). However:
1. **Never called:** No callsites outside the definition.
2. **Compiler already does this:** With `--use_fast_math` and `-O3`, the compiler can schedule two independent `sigmoid` calls into the same dual-pipeline structure without the manual wrapper.
3. **The refactoring win is nil:** If a caller wants two sigmoids, they write `z = 1/(1+exp(-x)); r = 1/(1+exp(-y));` and the compiler does the ILP scheduling.

If this *were* called, the dual-pipe fill would be LOAD-BEARING (the point of the function). But it's dead code.

**Call Sites:** **0 (never called)**
- No references outside of `ptx_intrinsics.cuh` definition and HIP fallback.

**Action:** Delete function and its HIP fallback.

---

### 5. **affine_combine_ptx** — REDUNDANT (only definition, see affine_combine below)
**Location:** `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh:28–62`

**PTX Opcodes (12):**
12 `fma.rn.f32` instructions arranged in three "waves" for ILP (lines 40–53).

**Classification:** **REDUNDANT** (superseded by `affine_combine` in `affine2x2.h`)

**Details:**
- This function is **identical** to the inline PTX block in `/home/user/SuperGrok1.5/csrc/scan/affine2x2.h:43–62`.
- Both are only wrappers; neither is called directly.
- The actual function used is `affine_combine` (defined in `affine2x2.h:36`), which is a **C++ wrapper** that selects the PTX version or HIP fallback at compile time.
- The `ptx_intrinsics.cuh` version is **dead duplication** of `affine2x2.h` (copied from canonical location).

**Call Sites:** **0 (affine_combine_ptx itself never called)**
- The real function is `affine_combine` (used in Mamba scan, SG2 optimizer, and parallel prefix scan).
- `affine_combine_ptx` is only defined, not called.

**Clarification:** The **load-bearing** part is the 12-FMA ILP structure *within* affine_combine, which exists in `affine2x2.h`. The `ptx_intrinsics.cuh` version is a deprecated duplicate of that.

**Action:** Delete `affine_combine_ptx` from `ptx_intrinsics.cuh` (keep `affine_combine` in `affine2x2.h`).

---

### 6. **fast_rsqrt_nr** — LOAD-BEARING (1 callsite, Newton-Raphson refinement)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:96–102`

**PTX Opcodes (1):**
```ptx
rsqrt.approx.f32 %0, %1;         # 1/(sqrt(x)), approx
```

**C++ post-processing:**
```cpp
r = r * (1.5f - 0.5f * x * r * r);  // Newton-Raphson iteration
```

**Equivalent libm:** `rsqrtf(x)` (without refinement) or `1.0f / sqrtf(x)`

**Classification:** **LOAD-BEARING**  
The **refinement iteration** (`r * (1.5f - 0.5f * x * r * r)`) is NOT implicit in `--use_fast_math`. This is a Newton-Raphson step that improves accuracy beyond the approx.
- `rsqrtf(x)` under `--use_fast_math` emits `rsqrt.approx.f32` (1 cycle, low accuracy).
- `fast_rsqrt_nr` adds one explicit FMA iteration (the refinement), improving accuracy at a small cost.
- Deleting it and replacing with `rsqrtf(x)` loses the refinement.

**Call Sites:** **1**
- `/home/user/SuperGrok1.5/csrc/algorithms/supergrok2.h:161` — used in "indexer_rank" denominator for a normalized dot product.

**Action:** KEEP. The Newton-Raphson step is the point; it's not redundant.

---

### 7. **ptx_fma** — LOAD-BEARING (14 callsites, explicit FMA fusion guarantee)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:106–110`

**PTX Opcodes (1):**
```ptx
fma.rn.f32 %0, %1, %2, %3;       # a*b + c in one instruction
```

**Equivalent libm:** `fmaf(a, b, c)` or plain `a*b+c`

**Classification:** **LOAD-BEARING (architectural guarantee)**  
The hand-written PTX **guarantees** that the three operations (`a*b+c`) fuse into a single `fma.rn.f32` instruction. This is critical because:
1. **Precision:** FMA rounds only once; `a*b` then `+c` rounds twice, losing one bit of precision.
2. **Performance:** Single instruction vs. two.
3. **Portability:** `fmaf(a,b,c)` *should* fuse, but the compiler might not—especially in loop contexts where register pressure forces spilling.
4. **Used extensively:** This is the core of `ptx_affine_combine` and `ptx_expert_mlp_forward`, which are called O(log N) times in parallel prefix scans.

**Call Sites:** **14**
- `utils.cuh:155–161` — internal use in `ptx_affine_combine` (6 calls)
- `utils.cuh:179,181` — internal use in `ptx_expert_mlp_forward` (2 calls)
- `supergrok2.h:158,199,387–389,404` — 6 direct calls

**Action:** KEEP. The explicit PTX wrapper guarantees fusion, which is NOT guaranteed by `fmaf` alone (especially in tight loops).

---

### 8. **ptx_exp2** — REDUNDANT (only internal use, 2 calls within utils.cuh)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:114–118`

**PTX Opcodes (1):**
```ptx
ex2.approx.f32 %0, %1;           # 2^x, approx
```

**Equivalent libm:** `exp2f(x)` (or `powf(2.0f, x)`)

**Classification:** **REDUNDANT**  
With `--use_fast_math`, the C++ `exp2f(x)` already compiles to exactly `ex2.approx.f32 %0, %1`.
This is a pure wrapper with zero performance difference.

**Internal Use:**
- Called only within the same file: `ptx_expf` (line 129), `ptx_tanhf` (line 135), `ptx_sigmoidf` (line 142).
- These are **themselves never called** (see below).

**Call Sites (external):** **0**
- `ptx_exp2` is not called anywhere outside `utils.cuh`.

**Action:** Delete `ptx_exp2` and inline `exp2f(...)` into `ptx_expf`, `ptx_tanhf`, `ptx_sigmoidf`. But first, check if those functions are called.

---

### 9. **ptx_log2** — REDUNDANT (never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:121–125`

**PTX Opcodes (1):**
```ptx
lg2.approx.f32 %0, %1;           # log2(x), approx
```

**Equivalent libm:** `log2f(x)`

**Classification:** **REDUNDANT**  
With `--use_fast_math`, the C++ `log2f(x)` already compiles to exactly this.

**Call Sites:** **0**
- No references anywhere in the codebase.

**Action:** Delete function.

---

### 10. **ptx_expf** — REDUNDANT (15 callsites, but wrapper is unnecessary)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:128–130`

**Implementation:**
```cpp
return ptx_exp2(x * 1.4426950408889634f);  // log2(e)
```

**Equivalent libm:** `expf(x)` or `__expf(x)`

**Classification:** **REDUNDANT**  
The C++ `expf(x)` with `--use_fast_math` already compiles to:
1. Scale: `x * log2(e)` (one FMA)
2. Approx: `ex2.approx.f32 ...` (one instruction)

This is identical to what `ptx_expf` does, wrapped in a function call. The function call overhead is negligible (inlined), but the hand-written wrapper adds no value over the library version.

**Call Sites:** **15**
- `mamba_scan_adapter.cuh:58,71,121,142,143,209,210,237,311,328,351` (11 calls)
- `supergrok2.h:128,207,208,290` (4 calls)

**Replacement:** Replace `ptx_expf(x)` with `expf(x)` or `__expf(x)`.

**HIP Fallback:** Already correct (`expf(x)` on line 220).

**Action:** Delete function. Replace callsites with `expf(x)`.

---

### 11. **ptx_tanhf** — REDUNDANT (never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:134–137`

**Implementation:**
```cpp
float e2x = ptx_exp2(2.0f * x * 1.4426950408889634f);
return (e2x - 1.0f) / (e2x + 1.0f);
```

**Equivalent libm:** `tanhf(x)`

**Classification:** **REDUNDANT**  
With `--use_fast_math`, the C++ `tanhf(x)` already compiles to an approx form. The manual expansion via `exp2` is equivalent but more verbose.

**Call Sites:** **0**
- No references anywhere in the codebase.

**Action:** Delete function and its HIP fallback.

---

### 12. **ptx_sigmoidf** — REDUNDANT (never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:141–144`

**Implementation:**
```cpp
float en = ptx_exp2(-x * 1.4426950408889634f);
return 1.0f / (1.0f + en);
```

**Equivalent libm:** Standard sigmoid: `1.0f / (1.0f + expf(-x))`

**Classification:** **REDUNDANT**  
This is a manual expansion of sigmoid using `ptx_exp2`. With `--use_fast_math`, the C++ `1.0f / (1.0f + expf(-x))` compiles to the same form.

**Call Sites:** **0**
- No references anywhere in the codebase.

**Action:** Delete function and its HIP fallback.

---

### 13. **ptx_affine_combine** — UNCERTAIN (structure suggests LOAD-BEARING, but no direct calls)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:150–163`

**Implementation (C++ wrapper using ptx_fma):**
```cpp
out.m00 = ptx_fma(left.m00, right.m00, left.m01 * right.m10);
out.m01 = ptx_fma(left.m00, right.m01, left.m01 * right.m11);
// ... (6 more ptx_fma calls)
```

**Equivalent libm:** Plain C++:
```cpp
out.m00 = left.m00 * right.m00 + left.m01 * right.m10;
out.m01 = left.m00 * right.m01 + left.m01 * right.m11;
// ...
```

**Classification:** **UNCERTAIN** → **likely REDUNDANT, but see note below**

**Details:**
- This C++ function **uses `ptx_fma` internally** (which is LOAD-BEARING).
- But `ptx_affine_combine` itself is **never called directly**.
- The actual workhorse is `affine_combine` in `affine2x2.h:36`, which has the 12-FMA inline PTX block.
- If the caller needs explicit FMA fusion (which they do, for precision), they would use `affine_combine`.

**Call Sites:** **0**
- `ptx_affine_combine` is not referenced anywhere in the codebase.
- The real function is `affine_combine` (in `affine2x2.h`).

**HIP Fallback:** Line 224–228 correctly delegates to `affine_combine(left, right)`.

**Action:** Delete `ptx_affine_combine`. Keep `affine_combine` in `affine2x2.h` (which has the true 12-FMA PTX).

**Note:** The `ptx_fma` calls within `ptx_affine_combine` are LOAD-BEARING on their own (FMA fusion guarantee), but the function that wraps them is dead code.

---

### 14. **ptx_expert_mlp_forward** — UNCERTAIN (template, never instantiated)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:169–184`

**Implementation (template<int EXPERT_HIDDEN>):**
```cpp
float result = b2;
#pragma unroll
for (int h = 0; h < EXPERT_HIDDEN; h++) {
    float hidden = ptx_fma(W1[h], input, b1[h]);
    hidden = fmaxf(hidden, 0.0f);  // ReLU
    result = ptx_fma(W2[h], hidden, result);
}
return result;
```

**Equivalent libm:** Plain C++ with compiler FMA fusion (assuming `-O3`):
```cpp
float result = b2;
#pragma unroll
for (int h = 0; h < EXPERT_HIDDEN; h++) {
    float hidden = W1[h] * input + b1[h];  // compiler fuses to FMA
    hidden = fmaxf(hidden, 0.0f);
    result = W2[h] * hidden + result;      // compiler fuses to FMA
}
return result;
```

**Classification:** **UNCERTAIN → likely REDUNDANT**

**Details:**
- This is a template function, so any instantiation would be visible as a link-time symbol.
- Grep shows **no callsites** anywhere in the codebase.
- The `ptx_fma` calls within guarantee FMA fusion, but the template is never instantiated.
- If the caller wanted an expert MLP forward pass, they'd write the loop inline and let the compiler fuse (with `--use_fast_math` and `-O3`, fusion is reliable).

**Call Sites:** **0**
- No references.

**HIP Fallback:** Line 219 correctly defaults to `fmaf(a, b, c)`.

**Action:** Delete template function. If an expert MLP is needed in the future, write it inline or use standard libm.

---

### 15. **ptx_int8_stochastic_round** — UNCERTAIN (uses `prmt` bit trick, but never called)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:189–201`

**PTX Opcodes (1):**
```ptx
prmt.b32 %0, %1, 0, 0x4140;      # extract lower 16 bits
```

**Implementation:**
```cpp
// Extract lower 16 bits via prmt, then normalize to [0,1)
unsigned lo16;
asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(lo16) : "r"(rand_bits));
float threshold = (float)lo16 / 65536.0f;
if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
```

**Equivalent libm:**
```cpp
float frac = fabsf(scaled - tr);
float threshold = (float)(rand_bits & 0xFFFF) / 65536.0f;
if (frac > threshold) tr += (scaled > 0) ? 1.0f : -1.0f;
return (int8_t)fmaxf(-127.0f, fminf(127.0f, tr));
```

**Classification:** **UNCERTAIN → LOAD-BEARING (if called), but NEVER CALLED**

**Details:**
- The `prmt.b32 %0, %1, 0, 0x4140` instruction is a **bit-permutation trick** that extracts the lower 16 bits in a single instruction (faster than branching/shifting).
- This is **load-bearing IF used** (the `prmt` is the whole point).
- But it's **never called** in the codebase.
- The HIP fallback (`float_to_int8_stochastic` in lines 72–83) uses the standard bit-mask approach (`rand_bits & 0xFFFF`), which is perfectly fine.

**Call Sites:** **0**
- No references anywhere.

**Action:** Delete function. The HIP fallback (`float_to_int8_stochastic`) already handles int8 stochastic rounding and is never called either.

---

### 16. **cluster_dsmem_reduce_sum** — OUT OF SCOPE (Stage 4.2, DSMEM reduction)
**Location:** `/home/user/SuperGrok1.5/csrc/common/utils.cuh:206–212`

**Classification:** **OUT OF SCOPE for Stage 3.0**

**Details:**
- This is a **Hopper DSMEM cluster reduction**, not a transcendental/approx function.
- It's part of Stage 4.2 (advanced Hopper optimizations), not Stage 3.0 (transcendental PTX).
- Current implementation is a fallback to `warp_reduce_sum`, which is standard.

**Call Sites:** **1**
- `/home/user/SuperGrok1.5/csrc/backends/cuda/sm_90/primitives.cuh:99`

**Action:** SKIP for Stage 3.0. Handle in Stage 4.2 audit.

---

## Summary Table: Classification & Action

| Function | File:Line | Classification | Call Sites | Action |
|----------|-----------|-----------------|-----------|--------|
| **softplus_ptx** | ptx_intrinsics.cuh:70 | REDUNDANT | 0 | **DELETE** |
| **fast_exp_ptx** | ptx_intrinsics.cuh:94 | REDUNDANT | 0 | **DELETE** |
| **stochastic_round_ptx** | ptx_intrinsics.cuh:112 | REDUNDANT | 0 | **DELETE** |
| **gru_gates_ptx** | ptx_intrinsics.cuh:139 | REDUNDANT | 0 | **DELETE** |
| **affine_combine_ptx** | ptx_intrinsics.cuh:28 | REDUNDANT (duplicate) | 0 | **DELETE** |
| **fast_rsqrt_nr** | utils.cuh:96 | LOAD-BEARING (Newton-Raphson) | 1 | **KEEP** |
| **ptx_fma** | utils.cuh:106 | LOAD-BEARING (FMA fusion guarantee) | 14 | **KEEP** |
| **ptx_exp2** | utils.cuh:114 | REDUNDANT | 0 (internal only) | **DELETE** |
| **ptx_log2** | utils.cuh:121 | REDUNDANT | 0 | **DELETE** |
| **ptx_expf** | utils.cuh:128 | REDUNDANT | 15 | **REPLACE with `expf()`** |
| **ptx_tanhf** | utils.cuh:134 | REDUNDANT | 0 | **DELETE** |
| **ptx_sigmoidf** | utils.cuh:141 | REDUNDANT | 0 | **DELETE** |
| **ptx_affine_combine** | utils.cuh:150 | REDUNDANT (dead wrapper) | 0 | **DELETE** |
| **ptx_expert_mlp_forward** | utils.cuh:169 | REDUNDANT (never instantiated) | 0 | **DELETE** |
| **ptx_int8_stochastic_round** | utils.cuh:189 | LOAD-BEARING (if called, but never called) | 0 | **DELETE** |
| **cluster_dsmem_reduce_sum** | utils.cuh:206 | OUT OF SCOPE (Stage 4.2) | 1 | **SKIP** |

---

## Redundant PTX Opcode Count

### For Deletion (no calls):

| Function | Opcodes | Notes |
|----------|---------|-------|
| softplus_ptx | 7 | 7 transcendental/comparison opcodes |
| fast_exp_ptx | 2 | wrapper around ex2.approx |
| stochastic_round_ptx | 8 | branchless rounding (never used) |
| gru_gates_ptx | 12 | dual sigmoid (never used) |
| affine_combine_ptx | 12 | 12-FMA (duplicate of affine_combine in affine2x2.h) |
| ptx_exp2 | 1 | (called only internally by ptx_expf/ptx_tanhf/ptx_sigmoidf) |
| ptx_log2 | 1 | (no calls) |
| ptx_tanhf | ~3–4 | (pure C++, no inline PTX) |
| ptx_sigmoidf | ~2–3 | (pure C++, no inline PTX) |
| ptx_affine_combine | 0 | (C++ wrapper, no inline PTX) |
| ptx_expert_mlp_forward | 0 | (template, never instantiated) |
| ptx_int8_stochastic_round | 1 | prmt.b32 for bit extraction |

**TOTAL (dead inline PTX opcodes):** ~49 opcodes

**PLUS ptx_expf replacements (15 callsites):**
- If each callsite is replaced `ptx_expf(x)` → `expf(x)`, the compiler still generates the same PTX (ex2.approx.f32).
- No net opcode change in generated SASS, but **no redundant wrapper function**.

---

## Recommended Cleanup (Stage 3.1)

### DELETE from `ptx_intrinsics.cuh`:
1. Lines 65–87: `softplus_ptx` (CUDA) + HIP fallback (172–174)
2. Lines 90–105: `fast_exp_ptx` (CUDA) + HIP fallback (176–178)
3. Lines 108–131: `stochastic_round_ptx` (CUDA) + HIP fallback (180–185)
4. Lines 134–162: `gru_gates_ptx` (CUDA) + HIP fallback (187–193)
5. Lines 28–62: `affine_combine_ptx` (CUDA) + HIP fallback (166–170)

### DELETE from `utils.cuh`:
1. Lines 121–125: `ptx_log2`
2. Lines 114–118: `ptx_exp2` (after inlining into ptx_expf, ptx_tanhf, ptx_sigmoidf if those were used)
3. Lines 134–137: `ptx_tanhf`
4. Lines 141–144: `ptx_sigmoidf`
5. Lines 150–163: `ptx_affine_combine`
6. Lines 169–184: `ptx_expert_mlp_forward`
7. Lines 189–201: `ptx_int8_stochastic_round`

### REPLACE in `utils.cuh`:
1. Lines 128–130: Inline `ptx_expf` into `expf` (or delete, since `expf` is already available from libm)

### REPLACE callsites (15 total):
- `mamba_scan_adapter.cuh`: Replace 11 `ptx_expf(...)` with `expf(...)`
- `supergrok2.h`: Replace 4 `ptx_expf(...)` with `expf(...)`

### KEEP in `utils.cuh`:
1. Lines 96–102: `fast_rsqrt_nr` (Newton-Raphson refinement is load-bearing)
2. Lines 106–110: `ptx_fma` (FMA fusion guarantee is load-bearing, 14 callsites)

### KEEP in `affine2x2.h`:
1. Lines 36–73: `affine_combine` (the real 12-FMA parallel prefix scan function with true inline PTX)

---

## Total Estimated Savings

- **Dead inline PTX:** ~49 opcodes (deleted)
- **Dead C++ wrapper functions:** 6 (softplus_ptx, fast_exp_ptx, stochastic_round_ptx, gru_gates_ptx, affine_combine_ptx, ptx_affine_combine, ptx_expert_mlp_forward, ptx_int8_stochastic_round)
- **Redundant function wrapper:** ptx_expf (15 callsites) — replace with libm `expf` (no opcode difference, but cleaner)

**Note:** These are **not** the ≈180 opcodes mentioned in the task spec (that likely refers to a different codebase version or hidden dead code). The audit finds ≈49 opcodes of explicit inline PTX duplication, plus 8 functions worth of dead wrapper code.

---

## Validation

All classifications have been cross-checked against:
1. Grep of the entire `/home/user/SuperGrok1.5/csrc` tree
2. HIP fallback implementations (which serve as the "reference" for what plain libm provides)
3. Build flags: `--use_fast_math` confirmed in setup.py:276
4. No undeclared dependencies on these functions (all callsites identified)

