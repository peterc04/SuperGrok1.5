# csrc/fused/sm_90 Infrastructure Digest
## Assigned Slice: SM_90 Infra (TP machinery, ZeRO-3 sharded optimizer, if-constexpr self-specialization)

**Agent:** K_csrc_sm90_other  
**Files read:** 15 (all in `csrc/fused/sm_90/` and `csrc/backends/cuda/sm_90/`)  
**Date:** 2026-06-25

---

## 1. PARALLEL_CONFIG.CUH — The Compile-Time Parallelism Point

**File:** `csrc/fused/sm_90/parallel_config.cuh` (279 lines)

### ParConfig — the adaptive 3D→5D template

```cpp
template <int DP, int TP, int PP, int SP, ZeROStage Z, int EP = 1>
struct ParConfig { ... };
```

All fields are `static constexpr`. The derived boolean gates that every megakernel consumer branches on with `if constexpr`:

| Gate | Expression | Effect |
|------|-----------|--------|
| `kIsSingleGPU` | `DP==1 && TP==1 && PP==1 && SP==1 && EP==1` | the degenerate byte-identical point |
| `kEmitComm` | `!kIsSingleGPU` | master gate for ALL NVSHMEM/NCCL symbols |
| `kShardParams` | `Z == ZeROStage::Z3` | ZeRO-3 param residency shard |
| `kShardOptGrad` | `Z >= ZeROStage::Z2` | grad+opt-state shard; kernel early-exits at B2 |
| `kTPComm` | `TP > 1` | in-kernel TP all-reduce enabled |
| `kPPStage` | `PP > 1` | pipeline stage cut |
| `kEPComm` | `EP > 1` | expert all-to-all (MoE); folds to zero when EP==1 (dense) |

**Adaptive axis semantics (actual mechanism):**
- Base 3D = DP × TP × PP for every model
- SP (4th) is EXPRESSIBLE via the template but enforced to 1 via `static_assert(SP == 1, ...)` at line 99 — any SP!=1 instantiation is a BUILD ERROR, not a runtime branch. At seq 4-17 a sequence split is moot (cited design §1.1/PARALLELISM-FINAL).
- EP (5th) defaults to 1, folds away on dense models. A future MoE model instantiates EP>1 cleanly; current dense roster never does.
- ZeRO is an ORTHOGONAL sharding strategy, not an axis in the DP×TP×PP count.

**Key invariant (line 88):** `kIsSingleGPU` folds EVERY NVSHMEM/NCCL branch away → the `SingleGPU` instantiation is byte-for-byte the pre-Par legacy kernel (the "PTX-diff gate," design §1.2).

```cpp
using SingleGPU = ParConfig<1, 1, 1, 1, ZeROStage::Z0>;   // line 111
```

### CommCtx — the runtime communication context (line 220)

Empty-by-default POD that carries:
- Rank/size info for DP, TP, PP axes
- `tp_comm_handle`: NVSHMEM TP team id stored as `reinterpret_cast<void*>((intptr_t)nvshmem_team_t)`
- `tp_sym_heap`: pointer to `nvshmem_malloc`'d symmetric TP-slot base (NOT plain cudaMalloc)
- `tp_heap_stride_floats`: per-PE symmetric stride
- `tp_team_local_pe` / `tp_team_n_pes`: the PE-in-TP-team index and team size
- EP fields (ep_comm_handle, ep_sym_heap, etc.) — mirror of TP fields for expert-parallel

All null/0 on the `SingleGPU` path — the kernel never reads them.

### SizeConfig — adaptive CTA-tiling (line 153)

```cpp
template <bool CtaTile, int CtasPerTile, int ClusterDim, int TileN>
struct SizeConfig { ... };
using SizeSmall = SizeConfig<false, 1, 1, SG_TUNED_TILE_N>;    // degenerate, default, byte-identical
using SizeLarge = SizeConfig<true,  2, 2, SG_TUNED_TILE_N>;    // CTA-tiled, 2 CTAs/tile, Hopper cluster
```

The Python-side selector (`megakernel_codegen.py::decoder_knobs_for_size`) picks the config from (d, layers, T, n_sms) and passes it as a template argument. The `SizeLarge` alias is defined for the dispatch allow-list, but the CTA-tiled body (`if constexpr (Sz::kCtaTile)`) is a NO-OP stub until §7 CTA-tiling lands.

---

## 2. MEM_CONFIG.CUH — Memory Strategy Compile-Time Descriptor

**File:** `csrc/fused/sm_90/mem_config.cuh` (55 lines)

```cpp
template <bool OffloadOpt, bool RecomputeActs, bool StreamLayers, int StreamDepth>
struct MemConfig { ... };
using InHbm = MemConfig<false, false, false, 0>;   // byte-identical default
```

Three independent memory strategies, all gated by `if constexpr` in the megakernel:
- **kOffloadOpt**: optimizer state in pinned host RAM, staged async
- **kRecomputeActs**: checkpoint layer boundaries; recompute fwd during bwd
- **kStreamLayers**: weight ring (kStreamDepth layers resident at once, prefetch via cudaMemcpyAsync)

`MemRuntime` POD carries: `host_state_base`, `stage_stream`, `stage_tile_floats`, `host_param_base`, `ring_slots`, `prefetch_stream`. All null on the `InHbm` path.

**This is the mechanism enabling single-GPU 10B+ training** by trading compute/bandwidth for capacity. The planner (Python-side `mem_strategy.plan_memory_strategy`) decides which gates to enable based on workload fit; the megakernel's code is the same regardless. Capacity does NOT dictate GPU count — the strategy-selection does.

---

## 3. TP_TRANSPORT.CUH — The Transport Swap Point

**File:** `csrc/fused/sm_90/tp_transport.cuh` (304 lines)

### The concept TpTransport exposes

```
int     my_pe() const
int     n_pes() const
float*  local(int64_t off)           // THIS pe's buffer at symmetric offset
const float* peer(int64_t off, pe)   // pe p's buffer at the SAME offset
void    rendezvous(GridBarrier& bar)  // cross-PE barrier
```

### LoopbackTransport (lines 118-158) — the single-process honest simulation

- Heap: ONE device allocation of `n_pes * stride_floats` floats
- `local(off)` = `heap_base + my_pe_ * stride_floats + off`
- `peer(off, pe)` = `heap_base + pe * stride_floats + off`
- `rendezvous(bar)` = `bar.sync()` (whole-grid GridBarrier IS the cross-PE barrier — all virtual PEs live in one grid)
- CTA→PE partition via `pe_of_cta(cta, nCTA, P)` = `cta / (nCTA/P)` (contiguous CTA groups, must have `nCTA % P == 0`)

This is NOT a fake stub — the TP math (shard GEMMs, publish, fixed-order ascending-pe fp32 reduce) executes identically to NVSHMEM; only the address translation and rendezvous scope differ.

### NvshmemTransport (lines 191-222) — the real multi-GPU device-NVSHMEM transport

Compiled ONLY under `-DSG_HAS_NVSHMEM=1`. Uses `nvshmem_ptr(heap_base, global_pe)` for NVLink direct load/store.

**rendezvous protocol (line 211):**
```cpp
bar.sync();                         // all CTAs of THIS GPU arrived
if (blockIdx.x == 0) {              // ONE CTA crosses the GPU boundary
    nvshmem_quiet();                 // drain THIS PE's published partials
    nvshmemx_barrier_block(tp_team_); // TEAM-scoped block barrier (TP group only)
}
bar.sync();                         // release every CTA after cross-GPU join
```

Key: `nvshmemx_barrier_block` (not `nvshmemx_barrier_all`) — DP/PP replicas are NOT dragged into the TP barrier. This is NVSHMEM 3.7.0's `nvshmemx_barrier_block` from `nvshmemx_coll_api.h`.

**make_transport_from_comm<Par>** (line 288): the ONE call-site seam. Selects NvshmemTransport or LoopbackTransport at compile time (#if SG_HAS_NVSHMEM). Both bind from the same CommCtx fields.

### The fixed-order all-reduce (lines 243-268)

```cpp
for (int64_t i = tid; i < n; i += nthreads) {
    float acc = 0.0f;
    #pragma unroll 1
    for (int pe = 0; pe < P; ++pe) {   // ASCENDING pe — fixed order
        acc += tr.peer(slot_off, pe)[i];
    }
    dst[i] = acc;
}
```

This is deliberately NOT `nvshmem_float_sum_reduce` (whose reduction order is unspecified → ULP drift → A/A/A failure). The hand-rolled ascending-pe loop IS the deliverable.

---

## 4. TP_LAYER.CUH — TP Math (Megatron col/row split)

**File:** `csrc/fused/sm_90/tp_layer.cuh` (291 lines)

### Megatron split layout for 30-tensor decoder

```cpp
__device__ __constant__ TpTensorShard kDecTpShard[30] = {
    { 0, Replicated},  // tok.weight
    { 2, ColQKV    },  // L0 in_proj_weight  (3-block q|k|v split by heads)
    { 4, Row       },  // L0 out_proj.weight (split along Kin)
    { 5, Replicated},  // L0 out_proj.bias   (post-reduce add)
    {10, Col       },  // L0 ff.0.weight
    {12, Row       },  // L0 ff.2.weight
    ...
};
```

QKV 3-block shard: rank r owns heads [r·H/P, (r+1)·H/P) for q, k, v independently; dense shard is [q_own | k_own | v_own], shape 3·d/P rows × d cols.

### Four all-reduce points (design §5.1)

1. **① fwd out_proj**: after `Ypart = Xown @ Wshard^T`, publish to sym slot, rendezvous, fixed-order reduce into sc.work, rendezvous
2. **② fwd ff2**: same after ff2 GEMM partial
3. **①' bwd in_proj dX**: after `dXpart = dYown @ Wshard`, publish + reduce → dh
4. **②' bwd ff0 dX**: same for ff0 backward

On `SingleGPU` path: all four `if constexpr (Par::kTPComm)` blocks fold away → byte-identical.

### Symmetric slot budget (line 283)

```cpp
tp_tile_slot_floats() = kTileM * dec::kD          // [kTileM · d] per slot
tp_heap_stride_floats(ctas_per_pe) = ctas_per_pe * 2 * kTileM * dec::kD  // 2 slots per CTA
```

Two slots: slot 0 (publish target), slot 1 (reduce output). Per CTA-in-flight. Stride = n_ctas_per_pe × 2 slots.

---

## 5. SHARDED_OPTIMIZER_KERNEL.CUH — ZeRO-2/3 Sharded Optimizer

**File:** `csrc/fused/sm_90/sharded_optimizer_kernel.cuh` (167 lines)

### What it does

After `[fwd+bwd megakernel] → reduce-scatter(grad)`, each rank holds the reduced grad for ONLY its owned shard. This kernel applies the optimizer over that shard:

```cpp
template <OptId Opt, class Par>
__global__ void sharded_optimizer_kernel(
    float* params_shard, const float* grad_shard, int64_t shard_numel,
    float lr, int step, FusedOptState st_shard) {
    // Flat grid-stride over [0, shard_numel)
    for (; i < shard_numel; i += stride) {
        apply_optimizer<Opt>(params_shard, grad_shard, i, step, st_shard);
    }
}
```

**Zero new math** (design §2.3): reuses `apply_optimizer<Opt>` from `opt_components.cuh` verbatim. No GridBarrier — single phase.

### Per-tensor-boundary taxonomy (lines 38-73)

The kernel COMPILES for all 11 OptIds but is CORRECT from this kernel alone only for:

- **Elementwise-drivable (flat kernel sufficient):** AdamW, Lion, Grokfast, NeuralGrok, and the elementwise cores of GrokAdamW/LookSAM/Prodigy (given their precomputed scalars already in `st_shard`)
- **Per-tensor / per-matrix (need upstream stage):** Muon (needs Newton-Schulz over whole 2D weight → tensor-granular ZeRO shard), SG11/SG15 (needs per-tensor meta-net mu), SG2 (needs full CSA/HCA/PEER/GRU)

For per-tensor optimizers, the sharded path uses the full persistent megakernel restricted to the tensors this rank owns (not this flat kernel). Host shard-mode selection enforces the correct path.

---

## 6. OPT_COMPONENTS.CUH — 11-Optimizer Device-Function Library

**File:** `csrc/fused/sm_90/opt_components.cuh` (524 lines)

### OptId enum

```cpp
enum class OptId : int {
    AdamW=0, Lion=1, Grokfast=2, GrokAdamW=3, LookSAM=4,
    Prodigy=5, NeuralGrok=6, Muon=7, SuperGrok11=8, SuperGrok15=9, SuperGrok2=10
};
```

### FusedOptState (105-243) — the unified state ABI

Carries ALL optimizer state via `float*` pointers and `float` scalars. Key fields:
- Adam moments: `exp_avg` (m), `exp_avg_sq` (v)
- Grokfast/GrokAdamW: `ema`
- LookSAM: `sam_dir`
- Prodigy: `s_track`, `d_factor`, `param_init`, `prodigy_persist[3]` (r_ema|s_ema|d_lr)
- SG11/SG15: `mu`, `gate`, `sharpness`, phi-net weights (`sg_phi_W1/b1/W2`, `sg_phi_b2`)
- Muon: `orth`, `neg_lr_scale`, `decay_factor`
- NeuralGrok: psi-net weights (`psi_W1/b1/W2`, `psi_b2`)
- SG2: `sg2_slow`, `sg2_gru_state`, full weight bundle pointers (`sg2_csa_q_W`, etc.), per-tensor scalars

**C2-GAP FIX (lines 249-272):** Previously bc1/bc2 were frozen at 1.0 (no Adam bias correction), gate=1.0 (SG gating inert), d_factor=1.0 (Prodigy inert). The `FusedScalars` POD + `apply_scalars()` function now correctly bridge runtime scalars → FusedOptState.

### apply_optimizer<Opt> (lines 381-520)

Pure `if constexpr` dispatch — each branch calls its canonical `csrc/algorithms/<opt>.h` function. No AdamW fallback. Notable specifics:

- **Grokfast (line 403):** COLD-START: `if (step == 1) st.ema[idx] = grad[idx]` (seeds EMA = first gradient to match eager behavior)
- **GrokAdamW (line 408-467):** Three faithful mechanisms: (i) per-tensor layer-wise β1 = β1·(1-γ)^layer (pre-rebased in P3), (ii) global grad-norm clip via `st.clip_coef`, (iii) adaptive α (faithfully a no-op in-context: no loss signal reaches GrokAdamW.step)
- **NeuralGrok (line 477-495):** Reads `psi_b2` ON-DEVICE from `st.psi_W2[kPsiHidden]` (host cannot deref device pointer)
- **SuperGrok2 (line 511-518):** Adam-on-smart_grad stub — falls back to `grad` if `st.smart_grad == nullptr`

**NeuralGrok psi-net packing** (line 75): `extra` buffer layout: `[psi_W1(16) | psi_b1(16) | psi_W2(16) | psi_b2(1)]`  
**SG11/SG15 phi-net packing** (line 88): `[phi_W1(64 = H×2) | phi_b1(32) | phi_W2(32) | phi_b2(1)]`

---

## 7. OPT_STAGES_PRECOMPUTE.CUH — Per-Step Precompute Stages

**File:** `csrc/fused/sm_90/opt_stages_precompute.cuh` (673 lines)

### Per-optimizer verdict

| Optimizer | Stage Type | Mechanism |
|-----------|-----------|-----------|
| Grokfast | NOTHING-NEEDED | EMA fused in apply (grokfast.h:63-65) |
| GrokAdamW | NOTHING-NEEDED | EMA fused in apply |
| NeuralGrok | NOTHING-NEEDED | psi MLP inline in apply |
| LookSAM | MODEL-COUPLED | sam_dir from 2nd backward; model stage supplies it |
| Prodigy | STAGED (phaseA+B) | cross-ALL-tensors r/s reduction → d |
| Muon | STAGED (sequential) | Newton-Schulz per matrix (5 matmul iterations) |
| SuperGrok11 | STAGED (per-tensor) | per-element mu + per-tensor cosine gate (single-drain) |
| SuperGrok15 | STAGED (per-tensor) | per-element mu only (gate = host scalar FusedScalars.gate) |
| SuperGrok2 | SKIP | sibling-agent owned |

### Determinism discipline

All cross-CTA reductions use **owner-computes per-CTA slot publish → grid barrier → ascending-index fixed-order sum** (owner_sum_slots at line 131). This REPLACES atomicAdd (which the live per-op kernels use; the replacement is math-equivalent in exact arithmetic but now has fixed, reproducible fp32 summation order).

### Prodigy (phaseA/B, lines 229-278)

- Phase A: each CTA drains task queue, accumulates (r,s) partial via `algo::prodigy_partials_step`, publishes per-CTA slot
- [caller grid barrier]
- Phase B: owner-sum slots in ascending index → `algo::prodigy_update_d` → writes `ws.prodigy_d[0]` (= st.d_factor)

### Muon (lines 336-433)

NS constants: kMuonNS_A=3.4445, kMuonNS_B=-4.7750, kMuonNS_C=2.0315 (from bindings.cpp:938-940).  
Phase sequence: `muon_momentum_norm_phaseA` → [barrier] → `muon_norm_reduce_phaseB` (writes inv_norm) → [barrier] → `muon_scale_X` → (5× { `muon_matmul(XXᵀ)` → [barrier] → `muon_matmul(AX)` → [barrier] → `muon_matmul(AAX)` → [barrier] → `muon_ns_combine_phase` → [barrier] → swap orth↔X }).  
The matmul (`muon_matmul` at line 391) is owner-computes: one thread per output element, K-dimension summed in ascending order.

### SG11 per-tensor (lines 523-562)

`sg11_precompute_mu_and_gate_for_tensor<H=32>`: single CTA, single drain. Writes mu[i] = rescale·sg11_phi_forward(g, sharpness), accumulates (num/den_g/den_m) for cos(grad, momentum), then calls `sg11_finalize_gate` → `sigmoid(gate_temp · cos)`. CRITICAL: this uses the START-OF-STEP momentum (exp_avg), not mu, as the 2nd cosine operand (per sg11.h:22-29).

**ABI gap**: FusedOptState has no `sharpness` field; it is passed as an explicit pointer to this stage. Documented in INTEGRATION-OPTSTAGES.md, not bridged in the dispatch ABI yet.

---

## 8. OPT_STAGE_SUPERGROK2.CUH — SG2 CSA/HCA/PEER/GRU Meta-Net

**File:** `csrc/fused/sm_90/opt_stage_supergrok2.cuh` (1343 lines)

### Architecture overview

Collapses ~15-20 per-tensor-per-step kernel launches into ONE persistent kernel. Each CTA processes WHOLE tensors from a work-steal task queue; stages within one CTA are `__syncthreads`-separated (not grid-barrier separated, because SG2 attention couples only a tensor's OWN rows — no cross-CTA coupling between stages).

### SG2Dims<> compile-time template (line 178)

Defaults: D_MODEL=8, NUM_HEADS=2, GRU_HIDDEN=4, NUM_EXPERTS=144, EXPERT_HIDDEN=16, NUM_PEER_HEADS=4, PEER_TOPK=4, CSA_COMPRESS=4, CSA_WINDOW=8, CSA_TOPK=16, HCA_COMPRESS=128, INDEXER_RANK=4.

Derived: pk_dim=12 (sqrt(144)), gru_in=2+2·D_MODEL=18, peer_in=GRU_HIDDEN+2·D_MODEL+2=22.

### SG2SmemWeights — 35.3 KB weight bundle staged once per CTA

```
input_proj: 8×2+8=24 floats
csa q/k/v/out: 4×64=256 floats
compress_w: 8; idx_DQ/K: 2×(8×4)=64 floats
hca q/k/v/out: 4×64=256 floats
gru Wz/Wr/Wh + biases: 3×(4×26)+3×4=324 floats (gru_in+gru_hidden=22+4=26)
peer Wq: 4×(8×22)=704 floats; prod A/B: 2×4×(12×4)=384 floats
experts W1/b1/W2 + b2: 3×(144×16)+144=7056 floats
Total: ~9028 floats = 36.1 KB (< 48 KB static smem cap)
```

NOTE: the code claims "35.3 KB" but the formula sums to ~36.1 KB. Both are below the 48 KB cap, so there is no functional issue, but the exact figure is slightly different.

### Stage pipeline (per tensor, CTA-local)

```
S0: input proj + gather sorted features (sg2_stage_input_proj_sorted)
    → x_sorted[N, d_model] using pre-computed perm[N]
S1: CSA context: q→compress(c_k,c_v)→window(win_k,win_v)→indexer(qI,kI)→top-k(sel)→attention→out_proj
    → csa_ctx[N, d_model]
S2: HCA context: same shape, dense (no indexer), stride hca_compress
    → hca_ctx[N, d_model]
S3: matrix-GRU: z,r,h̃,h_new per element from gru_input=[g,s,csa_ctx,hca_ctx]⊕h_old
    → new_gru[N, gru_hidden]; written back to gru_state in ORIGINAL order via unsort
S4: PEER routing + experts: per head, product-key top-k × k experts, softmax(×10), expert MLP, head avg
    → expert_out_sorted[N]; *= rescale
S5: unsort + sg2_apply_step (canonical: smart_grad = g + alpha·mu_new + lamb_eff·slow_new → Adam tail)
```

### BuildSort template parameter (line 1113)

- `BuildSort=false`: uses pre-computed `st.perm` / `st.unsort` (standalone, one batched torch.argsort by Python driver)
- `BuildSort=true`: runs STAGE -1 in-kernel segmented bitonic sort (`sg2_stage_segmented_sort`) — composed megakernel path

### Per-CTA workspace (sg2_ws_stride / sg2_carve_ws, lines 440-511)

Conservatively sized for Nmax (largest tensor). Includes sort scratch: `2 * sg2_next_pow2(Nmax)` floats for keys+idx (padded to next power of 2 to avoid overflow when N is non-power-of-2). **Important fix**: the carve logic at line 505 uses `sg2_next_pow2(N)` not `N` for sort scratch — silently corrupting perm/unsort when N is non-power-of-2 (e.g., N=384→Npow2=512) would have been a bug, now correctly sized.

### Persistent megakernel (lines 1266-1297)

`sg2_meta_optimizer_megakernel<Dims, ParamT, GradT><<<n_sms, 256>>>`. Stages smem weights once, then work-steals tensors, running `sg2_meta_stages` per tensor. ONE grid barrier at the end (fence for composing model stage queue reset).

**Launch guard (line 1321-1325):** `cudaOccupancyMaxActiveBlocksPerMultiprocessor` refuses if `occ < 1` → `cudaErrorLaunchOutOfResources`. Hang-freedom contract.

---

## 9. SG2_META_TAIL.CU — Host Launcher TU

**File:** `csrc/fused/sm_90/sg2_meta_tail.cu` (164 lines)

Two non-template host entry points:
- `sg::sg2_meta_optimizer_tail(...)` — marshals 30+ torch::Tensor args into SG2Weights/SG2State/SG2Scalars/PersistentContext → launches `launch_sg2_meta_optimizer_tail<SG2DimsDefault>`
- `sg::sg2_ws_stride(int64_t Nmax)` — exposes the authoritative per-CTA workspace stride (prevents host/kernel drift)

Uses `SG2DimsDefault = SG2Dims<>` (default dims). Separate TU (`.cu`) so `<<<>>>` launch syntax compiles; `bindings.cpp` extern-declares and pybind-registers the entry points without including the `.cuh`.

---

## 10. NVSHMEM_BRINGUP_PYBIND.CPP — Host TP-Team Bring-Up

**File:** `csrc/fused/sm_90/nvshmem_bringup_pybind.cpp` (357 lines)

### Module: sg_nvshmem_bringup

Exposes a pybind11 module for NVSHMEM init/team/heap/smoke operations:

| Function | Description |
|----------|-----------|
| `get_uniqueid()` | Returns 128-byte `nvshmemx_uniqueid_t` blob (rank-0 only) |
| `uniqueid_size()` | Returns `sizeof(nvshmemx_uniqueid_t)` = 128 |
| `init_with_uniqueid(rank, nranks, uid_bytes, device)` | `nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID)` — UID bootstrap path |
| `team_split_strided(pe_start, pe_stride, pe_size)` | `nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD)` → returns int team handle |
| `malloc_symmetric_heap(need_floats)` | Collective `nvshmem_malloc` → returns device ptr as int64 |
| `tp_allreduce_smoke(n)` | Fill sym heap with (world_pe+1), host-callable `nvshmem_float_sum_reduce` → return reduced tensor |
| `barrier_world()` | `nvshmem_barrier_all()` |
| `finalize()` | `nvshmemx_hostlib_finalize()` (the EXPORTED teardown) |

**Critical implementation detail (line 145-146):** Uses `nvshmemx_hostlib_init_attr` (the EXPORTED symbol), NOT the header-inline `nvshmemx_init_attr` (which calls the UNEXPORTED `nvshmemi_init_thread`). This correctly links against `libnvshmem_host.so.3`.

**BringupState** module-local singleton: tracks initialized/world_pe/world_npes/tp_team/tp_local_pe/tp_n_pes/sym_heap/sym_floats. Idempotent finalize.

**World PE consistency assert (line 157-163):** verifies `world_pe == my_rank && world_npes == n_ranks` after init — loud failure if NVSHMEM assigns unexpected PE IDs.

---

## 11. FUSED_DISPATCH_TABLE.INC — 33-Cell sm_90 Router

**File:** `csrc/fused/sm_90/fused_dispatch_table.inc` (119 lines)

Auto-generated (marker: "AUTO-GENERATED by megakernel_codegen.py"). Declares 33 functions (11 opts × 3 models: transformer_decoder, vit, mamba3), then provides `dispatch_sm90_cell()` as a string-keyed router:

```cpp
inline cudaError_t dispatch_sm90_cell(
    const std::string& model, const std::string& optimizer,
    PersistentContext ctx, float* params, const float* input,
    float* acts, float* grad, float* m, float* v, float* extra,
    const int* sizes, const int* offsets, float lr, int step,
    const FusedScalars& scalars, bool opt_only,
    cudaStream_t stream, bool* found);
```

All 33 cells follow the IDENTICAL signature — enabling the Python front-end to dispatch a (model, optimizer) pair to the correct megakernel TU without #ifdef clutter.

---

## 12. BACKEND SUPPORT FILES (sm_90/)

**Files:** `mma.cuh` (877 lines), `primitives.cuh` (679 lines), `tile_pipeline.cuh` (366 lines), `warp_specialize.cuh` (173 lines)

### mma.cuh — CUTLASS 3.6 Hopper WGMMA/TMA

Gated by `-DWITH_CUTLASS`. Uses `GemmUniversalAdapter<GemmUniversal>` from `CollectiveBuilder<arch::Sm90, OpClassTensorOp>` — explicitly NOT the default `device::Gemm` (no arch tag) which silently defaults to SIMT/Sm70 (no tensor cores).  
Per-thread lazy workspace: `sm90_get_workspace(bytes)` with thread-local cache.  
TMA descriptor reuse: initializes operator once per shape, caches keyed by (M,N,K,A_ptr,B_ptr).

### primitives.cuh — Block/Warp Reductions and Grid Helpers

Key: `block_reduce_sum_f32` (deterministic within-block, fixed thread count → fixed fp32 order), `block_reduce_sum2_f32` (two-value variant for Prodigy (r,s) reductions), `SG_LAUNCH_CHECK(stream)` macro (post-launch error check without sync in release).

### tile_pipeline.cuh — Double/Triple-Buffer Producer-Consumer Pipeline

`TilePipeline<N, Depth>`: mbarrier ring of `2*Depth` barriers (full + empty per stage), driving the wgmma mainloop with producer WG0 (cp.async stages K-tiles, arrives_expect_tx) and consumer WG1 (wait, issue wgmma in ascending-k order, arrive empty). `tc_pipelined_gemm_m64nNk16` is the turnkey mainloop.  
setmaxnreg register rebalance: producer deallocates (SG_TUNED_PROD_REGS=40), consumer allocates up (SG_TUNED_CONS_REGS=232).

### warp_specialize.cuh — Hopper Warp-Specialization Primitives

`elect_one_sync`: `elect.sync` PTX (single leader lane), sm_90+ guard.  
`Mbarrier`: `mbarrier.{init, arrive_expect_tx, try_wait, invalidate}` PTX wrappers.  
`warpgroup_reg_alloc/dealloc`: `setmaxnreg.{inc,dec}` PTX.

---

## 13. DEC_WEIGHTS.CUH — Decoder Weight Substrate

**File:** `csrc/fused/sm_90/dec_weights.cuh` (1023 lines)

Real transformer-decoder weight binding + fp32 forward/backward stages. Architecture: Transformer(nl=2, d=128, h=4, ntok=99, seq=4). Salvaged from the eager/surrogate removal — NOT dead. Key consumers: `model_stage_decoder_tc.cuh`, `fused_decoder_megakernel.cuh`, `tp_layer.cuh`. Full fwd+bwd stages with real GELU (exact erf, NOT tanh approx), LayerNorm (eps=1e-5), attention (1/√dh causal masking), cross-entropy (mean over batch). Verified bit-identical to the PyTorch oracle in `tests/hw/decoder_oracle.py`.

---

## 14. THE TP DATA-PATH FIX WIP PATCH

**File:** `/workspace/phase6/tp_datapath_fix_WIP.patch` (358 lines)

### Status: UNGATED (not applied to working tree)

This patch addresses 3 bugs found during the live one-model-across-8 training run:

**Bug A (line 84, launcher fix):** OOM-safe workspace `cudaMalloc` in `mega_decoder_real_adamw_tc_launcher.cu`. The flagship production-layout TC workspace is "hundreds of GB" → fails on 80 GB GPU → old code silently kept a stale/null `s.workspace` → IMA. Fix: honor the `merr != cudaSuccess` case, set `workspace=nullptr`, return `cudaErrorMemoryAllocation` before launch.

**Bug B (model_stage_decoder_tc.cuh):** Removes the Megatron weight-shard on the `kTPComm` path. Instead, ALL projections are computed FULL-WIDTH REPLICATED (same math as SingleGPU, including attention at full `kHeads=25` — eliminating the `kHeads%TP != 0` head-split invariant violation).

**Mathematical identity approach (patch line ~2145):**
```cpp
const float kTpInvP = Par::kTPComm ? (1.0f / (float)tr.n_pes()) : 1.0f;
// Each rank publishes full_result/P to its sym slot
// Fixed-order ascending-pe sum of P identical copies → full_result
```

This is mathematically correct (sum of P copies of x/P = x), genuinely exercises the NVLink NvshmemTransport peer()/nvshmem_ptr path + team barrier, and writes only full-width buffers (matching dec_acts_bind) eliminating out-of-bounds shard-width writes (Bug C).

**Bug C (IMA):** Claimed resolved by Bug A (OOM guard prevents null-workspace launch) + Bug B (full-width writes match dec_acts_bind). Status: UNCONFIRMED under compute-sanitizer.

---

## 15. DISCREPANCIES AND ISSUES

### D1: NVSHMEM Installation State Stale

`tp_transport.cuh:44` states "NVSHMEM IS NOT INSTALLED ON THIS BOX (verified 2026-06-12)". This is STALE — NVSHMEM 3.7.0 was pip-installed during the session (`pip install nvidia-nvshmem-cu12`), confirmed in `.session_memory/nvshmem-installed.md`. However, **NVSHMEM is currently NOT installed** on the current box because pip installs outside `/workspace` are deleted on closure (RESUME.md §1: "reinstall deps that lived outside /workspace"). The comment in `nvshmem_bringup_pybind.cpp` saying "NVSHMEM 3.7.0 IS installed" reflects the mid-session state, not the current state.

The NVSHMEM install is **a prerequisite to validate** that must be restored: `pip install nvidia-nvshmem-cu12`.

### D2: TP Data-Path Fix WIP Not Applied

RESUME.md §4 says the patch is "UNGATED — apply, then finish bug C". The current working tree does NOT have the patch applied. The `kTPComm` path in `model_stage_decoder_tc.cuh` still has the pre-fix Megatron weight-shard code with the 25-heads invariant violation.

### D3: "Cross-GPU in-kernel device-NVSHMEM TP all-reduce VALIDATED on 8 GPUs" Claim

RESUME.md §3 claims this is DONE+VALIDATED. However, what was validated was the NvshmemTransport transport-layer smoke (the `tp_allreduce_smoke` in `nvshmem_bringup_pybind.cpp`) and basic correctness, NOT the full training path with the data-path bugs fixed. The training TP data-path (kTPComm megakernel) has known bugs A/B still in the working tree, confirmed by the remaining work in RESUME.md §4.

### D4: SG2 Weight Bundle Size Claim (Minor)

`opt_stage_supergrok2.cuh:219` says "35.3 KB" but the formula sums to ~36.1 KB (9028 floats × 4 bytes / 1024 ≈ 35.3 KB — actually 35.3 KB is correct: 9028 × 4 = 36,112 bytes = 35.3 KB). No issue; the float count is correct.

### D5: SG11 ABI Gap

`FusedOptState` in `opt_components.cuh` has no `sharpness` field. `opt_stages_precompute.cuh` takes `sharpness` as an explicit pointer parameter. This gap is documented in INTEGRATION-OPTSTAGES.md but not bridged in the dispatch ABI yet — the integration wiring is outstanding work.

### D6: SP Axis Frame in Comments

`parallel_config.cuh:57` describes SP as "+SP (4th) iff the model is a SEQUENCE model (decoder / ViT-patches / Mamba) — EXPRESSIBLE but pinned 1 this campaign". The code is honest (static_assert at line 99 is the gate), but the comment frame suggesting SP would naturally be 4th for sequence models is aspirational design language, not current behavior.

---

## 16. CONFIG-DERIVATION MECHANISM ANALYSIS

The "self-adapting" claim is IMPLEMENTED in the following concrete mechanisms:

### How config is derived from workload × hardware

1. **Python-side planner** (`grokking_optimizers/parallel/auto_config`, `grokking_optimizers/megakernel_codegen.py::decoder_knobs_for_size`) takes (d_model, layers, T, n_sms, n_gpus) and outputs a `ParConfig<DP,TP,PP,SP,Z,EP>` + `SizeConfig<...>` + `MemConfig<...>` instantiation.

2. The derived config is passed as TEMPLATE ARGUMENTS to the megakernel. All branches are `if constexpr` keyed on compile-time constants → zero-overhead.

3. **ParConfig derivation rules:**
   - `kIsSingleGPU = (DP==1 && TP==1 && PP==1 && SP==1 && EP==1)` → SingleGPU path
   - `kEmitComm = !kIsSingleGPU` → ALL comm code folds away on single-GPU
   - `kTPComm = (TP > 1)` → TP all-reduce enabled
   - `kShardOptGrad = (Z >= Z2)` → megakernel early-exits at B2 for ZeRO>=2
   - `kEPComm = (EP > 1)` → expert dispatch enabled (dense EP==1 always folds)

4. **SizeConfig derivation:** `SizeSmall` (degenerate, byte-identical default) vs `SizeLarge` (CTA-tiled occupancy>1 + Hopper cluster). Selector picks based on workload size vs SM count.

5. **MemConfig derivation:** `InHbm` (default) vs enabled combinations. Single GPU can train 10B+ with OffloadOpt + RecomputeActs + StreamLayers — no GPU-count constraint.

**No hardcoded "10M params → 1 GPU" or "if num_gpus==1" branching:** confirmed. The if-constexpr gates are all on compile-time ParConfig/SizeConfig/MemConfig fields.

**Robustness:** SP=1 is enforced by static_assert (not a runtime fallback). EP=1 truly folds by the constexpr bool. TP>1 properly enables all 4 reduce points. ZeRO stage controls the B2 early-exit gate.

**Gap:** The SizeLarge `if constexpr (Sz::kCtaTile)` body is a NO-OP stub (§7 CTA-tiling not yet implemented). The alias is defined and the gate exists, but the CTA-tiled execution body is not authored.

---

## 17. OPEN ITEMS / BUGS / BLOCKERS

1. **TP data-path fix WIP NOT applied** — `tp_datapath_fix_WIP.patch` is ungated. Apply it, then confirm bug C IMA cleared under compute-sanitizer.
2. **NVSHMEM not currently installed** — requires `pip install nvidia-nvshmem-cu12` to restore the session-installed state.
3. **Bug C (IMA) unconfirmed** — the patch claims it's resolved by the OOM guard + full-width math, but no compute-sanitizer run confirms this.
4. **Real-data benchmark (Layer-B) not done** — 11×3 ranking is currently OVERFIT placeholder (`/workspace/phase6/flagship_11opt_ranking.{json,txt}`).
5. **Full 33-cell roofline incomplete** — currently 10 cells (decoder only); Mamba now launches but roofline not re-run.
6. **SizeLarge kCtaTile body not implemented** — the `if constexpr (Sz::kCtaTile)` arm in the megakernel is a NO-OP stub. SizeLarge compiles but degenerates to SizeSmall behavior.
7. **SG11/SG15 sharpness ABI gap** — `FusedOptState` has no `sharpness` field; integration wiring from model-stage SAM 2nd-backward to precompute stage is outstanding.
8. **LookSAM model-stage wiring** — `sam_dir` = g_sam − g needs the megakernel's SAM 2nd-backward wiring into the optimizer precompute seam; not yet authored.
9. **`tp_transport.cuh:44` comment stale** — says NVSHMEM "NOT INSTALLED" (true pre-session, false mid-session, true again post-closure). Should be updated to reflect the conditional state.
10. **Mamba 2 test failures** — RESUME.md §3: "Mamba 3/5 (2 PRE-EXISTING fails: B_bias-tol + obsolete proj_dw)".
