# csrc/common, csrc/bindings, csrc/fused/megakernel_common, csrc/algorithms — Ground-Truth Digest

**Agent**: K_csrc_common_bindings  
**Date**: 2026-06-25  
**Files read**: 20 (all files in K_csrc_common_bindings.txt + K_csrc_tp_nvshmem.txt)

---

## 1. Pybind11 Module (_ops.so) — What It Exports

Source: `csrc/bindings/bindings.cpp` (consolidated from 16+ original files + module.cpp)

### 1.1 Module Registration (`PYBIND11_MODULE(SG_OPS_PYMODULE, m)`)

The module name is driven by `TORCH_EXTENSION_NAME` (macro from setup.py or torch.utils.cpp_extension.load()); the `SG_OPS_PYMODULE` indirection is deliberate so that the JIT autotuner variants get unique per-variant module names (grokking_compiled_<opt>_<model>_<arch>_<cfg>) while the product build still exports `PyInit__ops`.

**Exported callables** (as of `bindings.cpp` lines 155–295):

| Export | C++ function | What it does |
|--------|-------------|--------------|
| `_ops.detect_arch()` | `sg::detect_arch` | Returns 90 (Hopper) or 942 (gfx942); TPU handled in Python. Memoized (C++11 static init). |
| `_ops.fused_step(model, optimizer, params, input, grad, state, lr, ...)` | `sg::fused_step` | Fused (model, optimizer, arch) megakernel dispatch. 34 positional/kw args total (with defaults). See §2. |
| `_ops.sg2_fused_step(model, params, input, grad, state, <26-tensor SG2 bundle>, ...)` | `sg::sg2_fused_step` | SuperGrok2 dedicated L3-TC entry (decoder/vit/mamba3). Separate because it needs 26+ HBM pointers + per-tensor scalar arrays that don't fit FusedScalars. |
| `_ops.sg2_meta_optimizer_tail(...)` | `sg::sg2_meta_optimizer_tail` | SG2 full meta-net as ONE persistent megakernel (CSA/HCA/GRU/PEER/apply). Definition in `csrc/fused/sm_90/sg2_meta_tail.cu`. |
| `_ops.sg2_ws_stride(Nmax)` | `sg::sg2_ws_stride` | Authoritative floats-per-CTA workspace stride for the SG2 megakernel; prevents host–kernel drift in buffer sizing. |
| `_ops.models.*` | `register_model_bindings(m)` | **Intentionally empty** (line 96-98): all per-op model kernels (decoder/vit/mamba fwd+bwd + attention/scan/patch components) were removed 2026-06-10 as dead code. The function body is `(void)m;`. |

**ABI schema version** (`bindings.cpp:120`): `GROK_ABI_SCHEMA = 1`. Exported as `_ops.__abi_schema__`. The Python-side check in `dispatch.py` is **noted as not yet implemented** (inert; harmless but leaves ABI drift silent until it is added).

### 1.2 Removed Exports (pure L3-TC transition)

All eager per-optimizer `*_fused_step` bindings (GrokAdamW, Lion, Grokfast, Prodigy, NeuralGrok, LookSAM, Muon, SuperGrok11/15), MoE entries, eager SG2 CSA/HCA step/batched/bilevel/prepare, and "mamba_peer" aliases are **removed**. The race uses `fused_step` / `sg2_fused_step` only.

---

## 2. `fused_step` Dispatch — Full Decision Logic

Source: `csrc/bindings/dispatch.cpp` lines 632–1192; signature declared in `csrc/bindings/helpers.h` lines 36–92.

### 2.1 Architecture Detection (`detect_arch`)

`dispatch.cpp:142-153` — memoized via `static const int cached_arch`.

Priority order:
1. `FORCE_ARCH` env var (parses "sm_*"/"smXX" → 90, "gfx*"/"942" → 942, bare numeric → 90, "tpu*" → throws)
2. Device query: `cudaDeviceGetAttribute(major)` — maps major==9 → 90; **LOUD-GATE** for any other NVIDIA CC (throws with explicit error); `hipGetDeviceProperties` → gfx942 only or throw.

NVIDIA arch 80/86/89 are **explicitly rejected** (no kernel body; cannot JIT-forward from sm_90a PTX to older arches).

### 2.2 `fused_step` Gate Chain

```
fused_step(model, optimizer, params, input, grad, state, lr, ...) {
  arch = detect_arch()                         // cached
  TORCH_CHECK(gemm_impl == "wgmma", ...)       // HARD reject; scalar path removed
  cell = wired_fused_cell(model, optimizer, arch)  // from fused_wired_cells.inc
  TORCH_CHECK(!cell.empty(), ...)              // HARD reject; no eager fallback

  if (arch == 90 && model == "transformer_decoder" && dec_l3_real && !opt_only)
    → mega_decoder_real_adamw_tc(...)          // returns
  if (arch == 90 && model == "vit" && vit_l3_real && !opt_only)
    → mega_vit_real_adamw_tc(...)              // returns
  if (arch == 90 && model == "mamba3" && mamba_l3_real && !opt_only)
    → mega_mamba_real_adamw_tc(...)            // returns
  // gfx942: dispatch_gfx942_cell(...)
  // else: throws (CPU/host build, no fused TU)
}
```

`dec_l3_real`/`vit_l3_real`/`mamba_l3_real` gates: true when `wgmma_tail_opt_id(optimizer) >= 0` AND optimizer != "supergrok2". SuperGrok2 is excluded because it routes via the dedicated `sg2_fused_step` entry.

### 2.3 `wgmma_tail_opt_id` — OptId Map

`dispatch.cpp:591-621`:

| optimizer string | OptId int | notes |
|---|---|---|
| "adamw" | 0 | AdamW |
| "lion" | 1 | Lion |
| "grokfast" | 2 | Grokfast |
| "grokadamw" | 3 | GrokAdamW |
| "looksam" | 4 | MODEL-COUPLED SAM 2nd backward (in-kernel P2.4 perturb→2nd fwd+bwd→sam_dir), SINGLE persistent launch |
| "prodigy" | 5 | STAGED global-d, in-kernel P2.6 reduction, SINGLE persistent launch |
| "neuralgrok" | 6 | NeuralGrok |
| "muon" | 7 | STAGED grid-cooperative Newton-Schulz P2.7, SINGLE persistent launch |
| "supergrok11" | 8 | MODEL-COUPLED SAM 2nd backward → sharpness + per-tensor meta-net mu/gate precompute P2.45, SINGLE persistent launch |
| "supergrok15" | 9 | SAM 2nd backward + mu precompute, gate is host scalar, SINGLE persistent launch |
| "supergrok2" | 10 | Routes via dedicated `sg2_fused_step` (NOT this map); excluded from generic gate |
| other | -1 | unroutable |

### 2.4 Hardcoded Model Sizes (dispatch.cpp)

These are HOST-side mirrors of device-side `__constant__` tables; cross-checked against `params.numel()` at the call site:

| Model | kTotalElems | kSeq/kPatchElems | n_tasks |
|---|---|---|---|
| transformer_decoder | 422,755 | kSeq=4, B*5 int32 input | 30 (kDecNumTensors) |
| vit | 418,017 | kPatchElems=16*49=784, float input + int-bit targets | 32 (kVitNumTensors) |
| mamba3 | 593,713 | kSeq=8, B*9 int32 input | 28 (kMambaNumTensors) |

These are compile-time constants for the SMALL benchmark models. The design is NOT parameterized by user config — the megakernel expects exactly these sizes and the `__constant__` tables encode the per-tensor sizes/offsets. This is a HARDCODED SHAPE, not config-derived.

### 2.5 State Buffer Sizing Logic

`dispatch.cpp:779-782` (decoder, mirrored for vit/mamba):
```cpp
int64_t min_state = (optimizer == "prodigy") ? (4 * total + 4)
                  : is_sg ? (4 * total + 1 + (int64_t)(4 * 32 + 1))
                  : (3 * total + 1);
```
- Standard: `[m|v|extra] + loss_slot` = 3*total + 1
- Prodigy: `[m|v|s_track|loss|param_init|r_ema|s_ema|d_lr]` = 4*total + 4
- SG11/SG15: `[m|v|mu|loss|sharpness|phi_pack(4*32+1)]` = 4*total + 1 + 129
- SG2: `[m|v|mu|loss|sharpness|slow|gru_state(total*GH)]` = (5+GH)*total + 1 where GH=4

### 2.6 FusedScalars POD (dispatch.cpp:204-237, mirrored in helpers.h)

26 float fields in declaration order (must be byte-identical to `opt_components.cuh`):
```
lr, beta1, beta2, eps, wd, bc1, bc2, alpha, beta, lamb, alpha_max,
gate, d_factor, neg_lr_scale, decay_factor,
gamma, grad_clip,                  // GrokAdamW (append)
d0, d_coef, beta3,                 // Prodigy (append)
aux_lr, aux_beta1, aux_beta2,      // Muon 1D-group AdamW (append)
rho, looksam_sam,                  // LookSAM SAM 2nd backward (append)
sg_rescale,                        // SG11/15 meta-net rescale (append)
gate_temp                          // SG11 cosine-gate temperature (append)
```
Inert defaults ensure every non-owning cell passes harmlessly.

**gfx942 FusedScalars** (`dispatch.cpp:377-380`): Only 15 fields (first 15; lacks gamma, grad_clip, Prodigy/Muon/LookSAM/SG scalars). The gfx942 path is structurally behind the sm_90 path.

### 2.7 FusedScratch Caching

`dispatch.cpp:407-444` — per (device_index, n) cache of barrier counters + acts proxy. Cache key: `(dev_idx << 40) ^ n`. On cache hit: only the 3 barrier tensors + acts are reset to zero (`zero_()`); sizes/offsets are invariant and NOT reset. This amortizes allocation overhead but requires the shape to be fixed per device.

### 2.8 Mamba A/A/A Determinism Fix (dispatch.cpp:983-1020)

Previously mamba × prodigy and mamba × looksam were non-deterministic due to wgmma accumulator spill in the TC backward:
- Root cause: `[kSeq][kState]` dB/dC arrays + a_save[kSeq][kState] spilled to local memory → combined with Prodigy P2.6/LookSAM P2.4 extra register pressure → ptxas C7515 hazard (wgmma accumulator spill during mma_async pipeline window) → non-deterministic/NaN grad drift
- Fix: (1) per-timestep dB/dC block-reduce (drops arrays), (2) recompute adec=exp(dt·A) instead of saving a_save, (3) `__noinline__` on mbtc_forward_tile/mbtc_backward_tile
- Now all 10 single-launch mamba tails (incl. SG11/15) pass A/A/A bit-exact

### 2.9 `opt_only` Parameter — Dead ABI Slot

`helpers.h:45-48`: The L1 faithful optimizer tail was removed (task #10). The parameter is kept for pybind ABI stability. Default false ⇒ L3-REAL path (the only surviving path). A stale _ops that predates the removal would fail at dispatch with the TORCH_CHECK.

---

## 3. `helpers.h` — Dispatch Macros and Validation

`SG_DISPATCH(METHOD, ...)` and `SG_DISPATCH_CALL(METHOD, ...)`: runtime arch switch that routes to `::sg::sm90::METHOD` or `::sg::gfx942::METHOD`. Backend-gated: `SG_CASE_SM90_*` expands only when `WITH_CUDA && !WITH_HIP`; `SG_CASE_GFX942_*` expands only when `WITH_HIP`.

**Boundary validation helpers** (`helpers.h:176-343`):
- `check_param_grad(p, g, where)`: checks param is defined, on correct device, contiguous, dtype/device/shape match with grad
- `check_params_grads`: multi-tensor version
- `check_list_len<T>`: secondary list length guard (templated for both Tensor and scalar vectors)
- `clip_grad_norms_device_side`: fused multi-tensor grad-norm clip via `torch::_foreach_norm` (single CPU sync); threshold ≤0 → inert
- `compute_sam_grad_norm_device_side`: per-tensor L2 norm accumulation for SAM

---

## 4. GridBarrier and TaskQueue Substrate

Sources: `csrc/fused/megakernel_common.cuh` (sm_90) and `csrc/fused/megakernel_common_hip.hip.hpp` (gfx942).

### 4.1 TaskQueue (`megakernel_common.cuh:91-117`)

Work-stealing global atomic counter. All CTAs run `for (t = q.next(); t < n_tasks; t = q.next())`. `next()` = `atomicAdd(g_next_task, 1)`. `next_block(smem_slot)`: thread 0 does the atomicAdd, broadcasts via shared memory + 2× `__syncthreads()`.

**No fence needed on the pull** — task bodies are independent; the fence belongs at the grid barrier between phases.

### 4.2 GridBarrier (`megakernel_common.cuh:147-255`)

Hand-built, reusable, sense-reversing:
- State: `g_arrived` (count), `g_generation` (sense; unsigned, advances monotonically)
- `sync()`: each CTA samples gen → `__threadfence()` (release) → `atomicAdd(arrived, 1)` → if last: reset arrived=0, `__threadfence()`, bump gen → else spin on gen with exponential backoff (`__nanosleep(backoff)`, backoff doubles each iteration capped at 1024ns, sm_70+ only)
- `sync_reset(int* reset_counter)`: folds task counter zero into last-arriver critical section; reduces 4→2 grid barriers per L3 step with identical ordering/visibility

§1.14 minimal fence rule: exactly ONE `__threadfence()` before publish (release) and ONE `__threadfence()` after wait completes (acquire). No loose fences.

Participates only at thread 0; rest of block gated by `__syncthreads()`.

### 4.3 PersistentContext

```cpp
struct PersistentContext {
    int*      g_next_task;    // §1.1 counter (host zero-inits)
    unsigned* g_arrived;      // §1.4 arrival count (host zero-inits)
    unsigned* g_generation;   // §1.4 sense (host zero-inits)
    int       n_tasks;        // parameter tensors this phase
    unsigned  n_ctas;         // #SMs, filled by launcher (one CTA/SM)
};
```
`n_ctas` is filled by the launcher; `n_tasks` is set from `kDec/Vit/MambaNumTensors` (host mirrors of device constants).

### 4.4 SM Pinning (`sm_id()`, `megakernel_common.cuh:65-73`)

`asm volatile("mov.u32 %0, %%smid;" : "=r"(id));` — host launches exactly `#SMs` CTAs (gridDim.x == #SMs) and the kernel indexes `sm_id()` into its per-SM optimizer-state slice to keep that slice warm in the SM's L2 partition.

### 4.5 gfx942 Twin (`megakernel_common_hip.hip.hpp`)

Identical algorithm, different primitives:
- AGENT-scope atomics (`__hip_atomic_fetch_add`, `__hip_atomic_exchange`) for cross-XCD coherence
- `__builtin_amdgcn_s_barrier` for intra-workgroup join
- `__builtin_amdgcn_fence(__ATOMIC_RELEASE/__ACQUIRE, "agent")` for release/acquire
- `__builtin_amdgcn_s_sleep(2)` for backoff (not exponential; constant ~128-cycle nap)
- `cu_id()` via `__builtin_amdgcn_s_getreg(HW_ID)` extracts CU_ID field [11:8]

**NOT warp-specialized** (§1.13): CDNA3 has no TMA/mbarrier analog; uses 4-wavefront ping-pong scheduling instead.

SG_TUNED_MEGA_BLOCK=256 default on both paths.

---

## 5. Platform Abstraction Layer (`csrc/common/platform.h`)

Full CUDA/HIP portability layer:

| Feature | CUDA | HIP |
|---|---|---|
| Backend detect | `GROK_CUDA=1` | `GROK_HIP=1` |
| Runtime include | `<cuda.h>` + `<cuda_runtime.h>` + Thrust + CUB | `<hip/hip_runtime.h>` + rocThrust + hipCUB |
| WARP_SIZE | 32 | `__AMDGCN_WAVEFRONT_SIZE__` or 64 (CDNA default) |
| SHFL_DOWN(val, offset) | `__shfl_down_sync(0xFFFFFFFF, val, offset)` | `__shfl_down(val, offset)` |
| FAST_SINCOSF | `__sincosf(x, s, c)` | `sincosf(x, s, c)` |
| LDG(ptr) | `__ldg(ptr)` (L1 cache hint) | `*(ptr)` (compiler handles) |
| stream_load(ptr) | PTX `ld.global.nc.f32` | `__builtin_nontemporal_load` |
| stream_store(ptr, val) | PTX `st.global.wt.f32` (sm_80+) or plain store | `__builtin_nontemporal_store` |
| stream_load4/store4 | PTX v4.f32 variants | Decomposed 4× scalar non-temporal |
| GPU error macros | cudaSuccess, cudaGetLastError, etc. | hipSuccess, hipGetLastError, etc. |
| Occupancy hints | (separate launch_bounds) | `GROK_WAVES_PER_EU(min,max)`, `GROK_FLAT_WORK_GROUP_SIZE(min,max)` |
| Smem size attr | `cudaFuncSetAttribute`/`cudaFuncAttributeMaxDynamicSharedMemorySize` | hipFuncSet... equivalents |
| CUB namespace | `cub::` | `namespace cub = hipcub;` |
| FULL_WARP_MASK | 0xFFFFFFFF | 0 (unused on CDNA) |

---

## 6. Common Types (`csrc/common/types.h`)

**Compile-time constants**:
```cpp
MAX_D_STATE = 128, MAX_D_INNER = 128, MAX_D_MODEL = 64
MAX_GRU_HIDDEN = 8, MAX_EXPERT_HIDDEN = 16, MAX_TOPK = 4
MAX_CKPT_INTERVAL = 32
SG2M_BLOCK = 256, SG2B_BLOCK = 256
PSCAN_BLOCK = 512  // SG_TUNED_PSCAN_BLOCK, power-of-2, -D-overridable
PSCAN_THRESHOLD = 256       // fall back to sequential below this
GEMM_PRECOMPUTE_THRESHOLD = 1024  // use GEMM when N >= this
```

Note: `PSCAN_BLOCK` has a comment "NEEDS-PARITY before a non-default winner ships" — the autotuner is not yet wired for scan dims.

**Device function**: `float_to_int8_stochastic_branchless(val, scale, rand_bits)` — uses `selp` PTX instruction to avoid warp divergence.

Includes `csrc/scan/affine2x2.h` and `csrc/common/platform.h`.

---

## 7. Device Utilities (`csrc/common/utils.cuh`)

Key device helpers:

- `sg_safe_bc(float bc)`: `fmaxf(bc, 1e-30f)` — prevents divide-by-zero in bias correction at step 0
- `warp_reduce_sum(val, d_inner, tid)`: warp shuffle reduction; handles d_inner < WARP_SIZE (non-power-of-two)
- `hash_prng(step, idx)`: Philox-like hash PRNG, no state tensor, deterministic per (step, element)
- `float_to_bf16_stochastic(val, rand_bits)`: unbiased BF16 quantization via bit manipulation
- `float_to_int8_stochastic(val, scale, rand_bits)`: INT8 with stochastic rounding
- `cluster_dsmem_reduce_sum(val)`: **SAFE FALLBACK SHIM** — falls back to `warp_reduce_sum` only, NOT a real DSMEM cluster reduction. The real Hopper DSMEM reduction is in `csrc/backends/cuda/sm_90/primitives.cuh::cluster_reduce_sum_f32_dsmem`. This shim is intentionally left for non-cluster call sites.

---

## 8. Affine2x2 Scan Primitive (`csrc/scan/affine2x2.h`)

2×2 affine transform for Mamba parallel prefix scan (Blelloch up/down-sweep):
```
[h_new] = [m00 m01] [h] + [b0]
[h_new'] [m10 m11] [h']  [b1]
```
Composition `right ∘ left` in 12 FMAs.

CUDA path: inline PTX `fma.rn.f32` for ILP across pipelines.
HIP/CPU path: plain C++ FMAs.

Used by: `csrc/algorithms/supergrok2.h` (SG2 optimizer scan recurrence), `csrc/backends/cuda/sm_90/launch_supergrok2.cu` (Blelloch parallel scan).

---

## 9. Algorithm Headers (Single Source of Truth)

Source: `csrc/algorithms/SOURCE_OF_TRUTH.md` — enforced by `scripts/check_math_single_source.py`.

All 11 optimizer math headers are in `csrc/algorithms/`. Both consumer paths (per-op kernels via `sm_90/<opt>_sm90.cuh`, fused megakernel via `opt_components.cuh`) `#include` these headers — no copy. gfx942 and TPU re-express the same math by hand and are marked as requiring manual sync.

### 9.1 AdamW (`adamw.h`)

Templated `adamw_step<ParamT,GradT>`. bc1/bc2 un-inverted convention (= 1−β^t, caller divides). `sg_safe_bc` guard. Also has `adamw_step_vec4` for float4 fast path.

### 9.2 Lion (`lion.h`)

Sign-based. `update = sign(β1·m + (1-β1)·g)`. Single momentum buffer. Also has `lion_step_vec4`.

### 9.3 GrokAdamW (`grokadamw.h`)

EMA filter + amplification: `ema_new = α·ema + (1-α)·g; g_amp = g + lamb·ema_new`. Then standard Adam on g_amp. Factored `grokadamw_adam_tail()` for reuse by quantized/storage-variant kernels.

### 9.4 Grokfast (`grokfast.h`)

Two modes: `grokfast_ema_step` (EMA-only, writes amplified grad out), `grokfast_fused_step` (EMA + AdamW in one step, register-resident).

### 9.5 LookSAM (`looksam.h`)

Four operations: `looksam_perturb_step`, `looksam_restore_step`, `looksam_set_direction`, `looksam_apply_step`. `g_adj = (1-α)·g + α·sam_dir` on normal steps; `sam_dir = g_sam - g` on SAM steps.

### 9.6 Prodigy (`prodigy.h`)

Three stages: `prodigy_partials_step`, `prodigy_update_d`, `prodigy_apply_step`. Key fix: degree-2 r/s accumulation (`r += g*(pi-p)*d²`, `s += d²*|g|`) for scale-free d estimate (degree-1 caused catapult blow-up). `s_local += d_prev*d_prev*fabsf(g)` (L1 norm, not signed sum).

### 9.7 NeuralGrok (`neuralgrok.h`)

2-layer MLP `neuralgrok_psi_forward<H>` + `neuralgrok_apply_step`. ReLU hidden. `g_amp = (s*α + β)*g`. Factored `neuralgrok_adam_tail()`.

### 9.8 Muon (`muon.h`)

2D: `muon_momentum_normalize_step` + `muon_ns_combine_step` (Newton-Schulz: `Y = a*X + b*AX + c*AAX`) + `muon_update_step` (`p = p*decay + neg_lr_scale*orth`). 1D: uses `adamw_step`. NS coefficients from Jordan et al. 2024: (3.4445, -4.7750, 2.0315).

### 9.9 SuperGrok11 (`supergrok11.h`)

Two-sweep per step. Sweep A: `sg11_phi_forward<H>` (GELU MLP on (grad, sharpness) → mu) + cosine accumulation (gate_num, gate_den_g, gate_den_m). Sweep B: `sg11_sweep_b_step` — `smart_grad = g + (1-gate)*α*mu`.

**Canonical gate**: `sg11_finalize_gate()` — `__host__ __device__ __forceinline__`. Computes `cos = gate_num / sqrt(gate_den_g*gate_den_m + eps)` → `gate = sigmoid(gate_temp * cos)`. Uses `expf` (not `__expf`) for host+device compatibility.

Factored `sg11_adam_tail()`.

### 9.10 SuperGrok15 (`supergrok15.h`)

Same GELU MLP structure as SG11. Gate is a scalar `gate_global = sigmoid(accuracy)` (host-set), not per-parameter cosine. `smart_grad = g + gate_global * sg15_alpha_per_coord(mu, alpha_base, alpha_max) * mu`. Per-coord alpha: `clamp(α_base * (1 + mu), 0, α_max)`.

### 9.11 SuperGrok2 (`supergrok2.h`)

Complex pipeline: input_proj_sort → CSA compressed attention → HCA heavily-compressed attention → PEER product-key routing → GRU → apply.

Key device functions:
- `sg2_input_proj_sort`: x_out = proj_W·[g,s] + proj_b; sort_key = |g|
- `sg2_csa_compress_kv`: online softmax pooling over window with learned weights
- `sg2_csa_index_score`: low-rank indexer dot product + 1/sqrt(rank) scaling
- `sg2_attention_score_and_accumulate`: FlashAttention-style streaming softmax update (running max + denominator)
- `sg2_softmax_finalize`: divide accumulator by denominator
- `sg2_hca_compress_kv`: mean pool or learned-weight pool with edge-safe renormalization
- `sg2_apply_step`: **RESTORED grokfast term** (was silently dropped in prior refactor):
  ```
  mu_new = gru_decay*mu_state + (1-gru_decay)*expert_out
  slow_new = alpha*slow_state + (1-alpha)*g
  smart_grad = g + alpha*mu_new + lamb_eff*slow_new
  ```
  Then Adam on smart_grad. NOTE: `sg2_apply_step` now has `slow_state` parameter.
- `sg2_bilevel_precompute_timestep`: recomputes q/k/v/indexer projections for bilevel adjoint (avoids saving them)

**MoE/Adam multi-tensor**: `moe_adam_step` — re-exports `adamw_step` for symmetric launcher glue.

**PTX helpers inlined in supergrok2.h** (Phase 3 S0): `fast_rsqrt_nr` (PTX rsqrt.approx + Newton-Raphson), `ptx_fma`, `ptx_exp2`, `ptx_expf`. HIP fallbacks use `rsqrtf`, `fmaf`, `expf`.

### 9.12 Bilevel Adjoint (`supergrok2_bilevel_adjoint.h`)

870-line hand-written reverse-mode VJP for the SG2 CSA/HCA/PEER/GRU meta-net. Uses ATen (NOT autograd). `bilevel_forward_save` fills `SavedActs` struct. `bilevel_backward_driver` accumulates 24 weight-grad buffers.

CSA backward: `csa_backward` — top-k selection is **stop-gradient** (discrete argmax; d_csa_idx_* accumulate zero by construction; noted in HARDWARE_VALIDATION.md as "🟡 exactly-zero-by-construction"). HCA backward: `hca_backward`. PEER backward: `peer_head_backward` (recomputes forward, then reverses routing → expert MLP → query projections).

---

## 10. COMPONENT_CONTRACT.md — Portability Spec

Component taxonomy and portability rules:
- **Substrate**: `megakernel_common.cuh` — depends on CUDA runtime only
- **Optimizer math**: `csrc/algorithms/<opt>.h` — pure per-element, no CUDA deps beyond `__device__`, no state allocation
- **Optimizer tails**: `opt_components.cuh` — templates over algorithm headers
- **Model stages**: `model_stage_<model>.cuh` — one header per model (NOT monolith); templated on compile-time dims
- **Composition**: `fused_megakernel.cuh` — stage pipeline assembly
- **Cells**: `mega_<model>_<opt>.cu` (×33) — GENERATED 36-line instantiations; never hand-edited
- **Dispatch**: `dispatch.cpp` + generated `.inc` files — cannot drift from cell enumeration

**Status (per COMPONENT_CONTRACT.md:44-60)**: All L3 fused model stages are **compile-verified only** on sm_90 as of 2026-06-09. The on-silicon H100 race exercises the eager model + fused-optimizer (L1) path, not the L3 megakernel. "Validated vs eager" is the open gate for every phase.

---

## 11. Key Open Issues and Bugs

### 11.1 Fused SG2 Path Missing Grokfast Term (BUG — confirmed in code)

`supergrok2.h:550-578` — explicit TODO. The fused SG2 path in `opt_components.cuh::apply_optimizer<SuperGrok2>` uses `adamw_step` directly on a pre-computed `smart_grad` from a SEPARATE meta-net launch. That launch does NOT include `lamb_eff*slow_new` (the restored grokfast slow-gradient EMA term). The per-op path is now correct (`sg2_apply_step`), but the fused path diverges. Required fix: add `slow_state` buffer, carry `lamb_eff` scalar, update smart_grad computation in the meta-net producer.

### 11.2 gfx942 FusedScalars Incomplete

`dispatch.cpp:377-380`: gfx942 FusedScalars has only 15 fields vs sm_90's 26. GrokAdamW's gamma/grad_clip, Prodigy's d0/d_coef/beta3, Muon's aux_lr/aux_beta1/aux_beta2, LookSAM's rho/looksam_sam, SG11/15's sg_rescale/gate_temp — all absent. The gfx942 dispatch cannot correctly run these optimizers for cells that need these scalars.

### 11.3 ABI Schema Check Not Implemented on Python Side

`bindings.cpp:120`: `GROK_ABI_SCHEMA = 1` exported, but `dispatch.py`'s Python-side assertion is noted as "inert until [the check] exists" (`bindings.cpp:117-119`). A stale `_ops.so` paired with newer Python wrappers could silently mis-marshal arguments.

### 11.4 cluster_dsmem_reduce_sum is a Fallback Shim

`utils.cuh:111-116`: The function is the "DSMEM off" behavior — warp-level reduction only. Cluster-aware sites must call `primitives::cluster_reduce_sum_f32_dsmem` directly. A call site that incorrectly uses `cluster_dsmem_reduce_sum` gets correct but non-cluster (non-cross-CTA) reduction silently.

### 11.5 PSCAN_BLOCK Autotuner Not Wired

`types.h:42-45`: comment "not yet wired into the autotuner search space — see report: scan dims are deferred on the per-arch cardinality budget." Default 512.

### 11.6 CSA Indexer Gradients Are Zero By Construction

`supergrok2_bilevel_adjoint.h:298-308`: csa_idx_DQ/UQ/K receive zero grad because the only consumer is a non-differentiable top-k index. The comment explicitly notes this is a 🟡 flag in HARDWARE_VALIDATION.md.

### 11.7 FusedScratch Cache Key Collision Risk

`dispatch.cpp:423`: Cache key = `(dev_idx << 40) ^ n`. Two different param shapes with the same `n` (element count) on the same device would share the same scratch, which is incorrect if the shapes differ in a way that affects the tensor sizes/offsets. In practice n uniquely determines the model layout for the small benchmark models, so this is low-risk but fragile.

---

## 12. Discrepancies vs CLAIMED State

1. **CLAIMED: "resource planner" and "adaptive config derivation"** — Not present in this slice. The model sizes (422755/418017/593713) and tensor counts (30/32/28) are **compile-time hardcoded** in dispatch.cpp as host mirrors of `__constant__` device tables. There is no runtime planner deriving sizes from user configuration in this layer. Dispatch is string-matched on model names ("transformer_decoder", "vit", "mamba3"), not config-derived.

2. **CLAIMED: SG2 grokfast fix complete** — The per-op `sg2_apply_step` has the restored `lamb_eff*slow_new` term, but the fused path has an explicit TODO (supergrok2.h:550-578) noting the fused opt_components.cuh still lacks it. The fix is INCOMPLETE for the fused L3-TC path.

3. **CLAIMED: All 11 optimizers wired and validated** — For the gfx942 path, FusedScalars is only 15 fields, making GrokAdamW/Prodigy/Muon/LookSAM/SG cells structurally broken on that arch. The sm_90 path has full 26-field scalars.

4. **L3 megakernel validation status** — COMPONENT_CONTRACT.md explicitly states: "the L3 fused model×optimizer megacells are compile-verified only on sm_90 — not yet runtime/numeric-checked on silicon." This conflicts with any claim that L3 TC is "validated."
