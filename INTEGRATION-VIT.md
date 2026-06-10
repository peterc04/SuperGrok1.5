# INTEGRATION-VIT.md — wiring the L3-REAL ViT megakernel (PHASE 2)

This is the integrator's cell/dispatch spec for the **real Vision-Transformer
forward+backward as persistent-megakernel stages**. The model stages, the
composition+launcher, the weight-layout header, the oracle/mirror, and the
hardware-gated tests are **already built and CPU-validated** (see "What ships"
below). What remains is the **integration seam** — the generated cell `.cu`, the
`dispatch.cpp` routing, the codegen layout emitter, and the Python `fused_train_step`
wiring — which the directive reserves for the integrator / sibling agents who own
`megakernel_codegen.py`, `dispatch.cpp`, and the `csrc/bindings/*` files.

This file is the exact contract for that seam. It mirrors PHASE 1's decoder
integration (BUILD_AND_VALIDATE.md "PHASE 1") with the **four ViT-specific
deltas** called out loudly. Source-complete only: no builds, no GPU runs were
performed.

---

## 0. What ships in PHASE 2 (already built + validated, do NOT re-derive)

| file | role | status |
|---|---|---|
| `tests/hw/vit_oracle.py` | manual fwd+bwd ORACLE + `vit_param_layout()` (THE single source of truth for the flat layout) | ✅ matches autograd, loss rel 0, worst grad rel **1.0e-15** (fp64, 32/32 tensors) |
| `tests/hw/vit_kernel_mirror.py` | single-threaded STRUCTURAL mirror of the kernel (buffer aliasing / recompute / FULL-attn 3-pass bwd / cls+patch scatter) | ✅ matches oracle, loss rel **2.0e-16**, worst grad rel **9.9e-16** |
| `csrc/fused/sm_90/model_stage_vit.cuh` | REAL ViT fwd/bwd stages (`vit_forward_sample` / `vit_backward_sample`), self-contained per COMPONENT_CONTRACT.md | source-complete (not built) |
| `csrc/fused/sm_90/fused_vit_megakernel.cuh` | composition + host launcher `launch_fused_vit_megakernel<Opt>` (incl. the **dynamic-smem opt-in**) | source-complete (not built) |
| `csrc/fused/sm_90/vit_layout.cuh` | hand-written weight-layout mirror, **static_asserts** on count(32)/total(418017)/smem-budget; **marked for codegen adoption** | source-complete |
| `tests/hw/test_vit_megakernel.py` | the parity ladder (no-GPU A/A2/B/B2 + GPU-gated C/D/E) | ✅ no-GPU gates PASS; GPU gates SKIP until this seam lands |

The kernel's MATH and STRUCTURE are therefore already proven on CPU; the seam
below only routes data to it. **Do not modify** `model_stage_vit.cuh` /
`fused_vit_megakernel.cuh` / `vit_layout.cuh` / the oracle / the mirror to make
integration easier — adapt the seam to them.

---

## 1. The architecture, pinned (cites grokking_race_v2.py)

`_raw_model` → `ViT(p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2)` (lines
389-404; `MODEL_SCALES["small"]` → d=128/h=4/nl=2, confirmed against
`DEFAULT_CONFIG`):

| component | def | shape |
|---|---|---|
| `cls_token` | `nn.Parameter[1,1,128]*.02` (:393) | [1,1,128] — **position 0 of every sample** |
| `patch_proj` | `nn.Linear(49,128)` (:392) | [128,49]+[128] — per-patch embed (16 patches) |
| `pos` | `nn.Embedding(16+1,128)` (:394) | [17,128] |
| 2× `EncoderBlock` | (:379) | |
| `attn` | `nn.MultiheadAttention(128,4,batch_first)` (:382), **NO mask — FULL attention** | in_proj [384,128]+[384], out_proj [128,128]+[128] |
| `n1,n2` | `nn.LayerNorm(128)` (:383), eps **1e-5** | [128]×2 each |
| `ff` | `Linear(128,512)→GELU→Linear(512,128)` (:384) | [512,128]+[512], [128,512]+[128] |
| post-LN | `x=n1(x+attn(x)); h=n2(x+ff(x))` (:386-387) | |
| `norm` | `nn.LayerNorm(128)` (:396) | [128]×2 |
| `out` | `nn.Linear(128,97)` (:396) | [97,128]+[97] |
| forward | `h=patch_proj(x); h=cat([cls,h]); h+=pos; for l: h=l(h); return out(norm(h[:,0,:]))` (:400-404) | **CLS token (pos 0)** |
| loss | `F.cross_entropy(m(tx), ty)` (:967…) | mean over B |

**32 parameter tensors, 418,017 total params.** Numerics (each verified against
autograd in `vit_oracle.py`): **GELU = exact erf**; LayerNorm **eps=1e-5**;
attention scale **1/√32**, softmax with row-max subtraction; CE **mean over B** →
`dlogits=(softmax−onehot)/B`. `nn.MultiheadAttention` transcribed in the
already-verified qkv-split form.

### The four ViT-specific deltas vs the decoder seam

1. **INPUT is FLOAT image patches `[B,16,49]`, not int tokens.** (Decoder: int32
   `input`.) targets are int `[B]`. ABI packing in §4.
2. **NO causal mask** — FULL attention. (Already baked into the stages/oracle/
   mirror; nothing for the integrator to do, but the parity test must use the
   no-mask eager model — `test_vit_megakernel.py::_build_eager_vit` already does.)
3. **Head reads CLS position 0**, not the last position. (Baked in.)
4. **32 tensors / 418017 elems** (decoder: 30 / 422755), and the first three are
   `cls_token, patch_proj.weight, patch_proj.bias` (cls_token is yielded BEFORE
   the patch_proj submodule — a leaf-before-children ordering; verified against
   the live `named_parameters()`). The layout header encodes this.

---

## 2. THE one hard requirement: dynamic shared memory (don't drop it)

ViT's per-sample working set `VitSampleSmem` is **188,080 bytes (≈183.67 KB)** at
seq=17 — **far over the 48 KB STATIC `__shared__` cap** (a static
`__shared__ VitSampleSmem sm;` would **not compile**). So the kernel uses
`extern __shared__` DYNAMIC smem, and the launcher
(`launch_fused_vit_megakernel`, already written) does **all three**:

```cpp
const int dyn_smem = (int)sizeof(VitSampleSmem);            // 188080
cudaFuncSetAttribute(&fused_vit_megakernel<Opt>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem); // (1) opt-in >48KB
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ..., dyn_smem); // (2) honest occ
fused_vit_megakernel<Opt><<<grid, 256, dyn_smem, stream>>>(...);     // (3) launch
```

**188080 B ≪ the sm_90 227 KB per-block dynamic cap** (44 KB headroom;
static_asserted in `vit_layout.cuh`), and the persistent megakernel is
**one-CTA-per-SM by design** (gridDim = #SMs, grid barrier over #SMs), so
**occupancy = 1 IS the design point** — a 184 KB block at occ=1 is intended, not a
regression. **Missing any of the three steps makes the kernel silently fail to
launch.** The launcher already encapsulates this; the integrator's only job is to
**not** pass `dynamicSMemBytes=0` anywhere on this path, and to ensure whatever
calls the launcher does so on a device whose `MaxDynamicSharedMemoryPerBlockOptin`
≥ 188080 (every sm_90 satisfies this).

---

## 3. Stage / barrier layout (one persistent kernel — identical to the decoder)

`fused_vit_megakernel.cuh`, gridDim = #SMs (one CTA/SM), 256 threads/CTA,
**batch-parallel** (NOT the param-tensor work-steal queue — that would vary the
batch→CTA grouping and fp32 sums aren't associative):

```
P0  each CTA zeroes its OWN grad-partial slice + loss slot
--- grid barrier B0 ---
P1  each CTA owns a FIXED contiguous batch slice (blockIdx.x); processes its
    samples ONE AT A TIME (CTA-cooperative): loads the sample's 16×49 float
    patches into smem, fwd+bwd, accumulating each sample's weight-grad into the
    CTA's partial with a SINGLE-OWNER-THREAD-PER-ELEMENT rule (no atomics →
    deterministic), and sums its slice's NLL (fp32)
--- grid barrier B1 ---
P2  deterministic cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA index
    into the global grad (no float atomics; order fixed → reuses the work-steal
    queue to pick WHO reduces which tensor). Loss: fp64 ordered sum → loss/B → a
    device float the host reads back.
--- grid barrier B2 (sync_reset: also resets the queue for P3) ---
P3  the REAL apply_optimizer<Opt> tail consumes the reduced grad in place
```

`n_tasks = 32` (kVitNumTensors) for the reduce + optimizer phases; the kernel
reads per-tensor numel/offset from the generated `__constant__` tables
`kVitSizes/kVitOffsets` (vit_layout.cuh). Determinism: fixed batch→CTA grouping +
fixed ascending-CTA reduction order; **no float atomics** in the weight-grad
reduce. The cls/pos embedding scatter maps **thread→embedding column** and the
patch_proj weight-grad maps **thread→(out,in)**, both looping the
sample's/patch's positions sequentially → deterministic.

**SMEM budget per CTA: `VitSampleSmem` = 188080 B dynamic** (one sample, one layer
live; the backward recomputes each layer's forward from the cached layer input).

---

## 4. ABI seam (no `bindings.cpp` / `setup.py` edit — the owned extension points)

The pybind `m.def("fused_step", ...)` arity is pinned in `bindings.cpp` (NOT
owned), so `fused_step`'s arity is unchanged. Carry the ViT input through the
EXISTING `input`/`state`/`grad` tensors and add the **behavior** in `dispatch.cpp`
(owned), exactly as the decoder did.

**Tensor contract (what Python passes; what dispatch.cpp reads):**

* `input` = **float32** `[B*16*49 + B]` contiguous — the `B*16*49` patch pixels
  (row-major `[B][16][49]`) followed by **B targets bit-cast to float** (each
  target is an `int32` reinterpreted into the float slot:
  `input[B*784 + b] = reinterpret as float of (int32)target[b]`). dispatch.cpp
  reads patches as `input.data_ptr<float>()` and targets as
  `reinterpret_cast<const int*>(input.data_ptr<float>() + (int64_t)B*784)`.
  `B = (input.numel()) / (16*49 + 1) = numel / 785`.
  *(Rationale: `fused_step`'s arity has one input tensor; the decoder packed
  int tokens++int targets into one int32 tensor. ViT's patches are float, so we
  keep `input` float and bit-pack the int targets into trailing float slots —
  the symmetric move. This is a BIT REINTERPRET both ways (store: copy the int32
  bit pattern into the float slot; read: `reinterpret_cast<const int*>`), NOT a
  numeric value cast — so it is lossless for ALL int32, independent of magnitude.
  An equivalent alternative is to carry targets in `state`'s trailing slots; the
  bit-pack keeps `input` a single contiguous tensor, matching the decoder.)*
* `params` = flat `[418017]` fp32 (the `torch.cat` of `named_parameters()` in
  order — see `vit_param_layout()` / `vit_layout.cuh`).
* `state` = `[m|v|extra]` (`3*418017`) **+ 1 trailing loss slot** the kernel
  writes the mean CE into (read back in Python). AdamW uses only `m|v`.
* `grad` = the **reduced weight-grad OUTPUT** `[418017]` — the kernel writes the
  deterministically-reduced grad here (P2) and the optimizer tail consumes it in
  place (P3, no overwrite), so after the call this buffer holds exactly the grad
  AdamW used. **Routing it through the ABI `grad` tensor exposes the kernel's
  grads to the parity test** (`return_grad=True`) — the keystone check (C.2).
* the **workspace** (`nCTA*418017` grad partials + `nCTA` loss) is device scratch
  allocated in `dispatch.cpp` (keyed by device, like `decoder_scratch_for`) — it
  never crosses the ABI. At 132 SMs this is `132 × 418017 × 4 B ≈ 221 MB`,
  allocated once.

**routing** (add to `dispatch.cpp`, the `#if defined(WITH_CUDA)&&!defined(WITH_HIP)`
block, mirroring the decoder branch at dispatch.cpp:455):

```cpp
if (arch == 90 && model == "vit" && optimizer == "adamw" && !opt_only) {
    const int64_t total = kVitTotalElems;          // 418017 (add a mirror const)
    TORCH_CHECK(params.numel() == total && params.scalar_type()==kFloat32
                && params.is_contiguous(), "...");
    TORCH_CHECK(input.scalar_type()==kFloat32 && input.is_contiguous(), "...");
    const int64_t in_n = input.numel();
    TORCH_CHECK(in_n % (16*49 + 1) == 0 && in_n > 0, "...");   // 785
    const int B = (int)(in_n / (16*49 + 1));
    TORCH_CHECK(state.numel() >= 3*total + 1 && ... , "...");
    TORCH_CHECK(grad.numel() == total && ... , "...");
    ViTScratch& vsc = vit_scratch_for(params);       // new device-keyed scratch
    fused::PersistentContext ctx{ vsc.g_next..., vsc.g_arrived..., vsc.g_gen...,
                                  32 /*kVitNumTensors*/, 0u };
    float* m = state.data_ptr<float>();
    float* loss_slot = m + 3*total;
    fused::sm90::FusedScalars scalars{ lr, beta1, beta2, eps, weight_decay,
        bc1, bc2, alpha, beta, lamb, alpha_max, gate, d_factor, neg_lr_scale,
        decay_factor };
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = fused::sm90::mega_vit_real_adamw(
        ctx, params.data_ptr<float>(),
        input.data_ptr<float>(),                                    // patches
        reinterpret_cast<const int*>(input.data_ptr<float>()
                                     + (int64_t)B*16*49),           // targets
        B, m, grad.data_ptr<float>(), vsc.workspace.data_ptr<float>(),
        loss_slot, lr, (int)step, scalars, stream);
    if (err != cudaSuccess) throw std::runtime_error(...);
    return;
}
```

Add near `kDecoderTotalElems` (dispatch.cpp:333): `constexpr int64_t
kVitTotalElems = 418017;` and a `ViTScratch` + `vit_scratch_for(params)` modeled
on `DecoderScratch` / `decoder_scratch_for` (workspace sized
`(int64_t)n_sms * 418017 + n_sms + 1`).

**The cell TU** — `csrc/fused/sm_90/mega_vit_real_adamw.cu` (new; picked up by
setup.py's `csrc/fused/sm_90/*.cu` glob, like `mega_decoder_real_adamw.cu`).
dispatch.cpp is HOST-compiled, so ALL `<<<>>>`/`__global__`/device code lives in
this nvcc TU. It exposes ONE non-template host launcher whose boundary signature
is plain pointers/ints + the `FusedScalars` POD (NO header-only types cross the
boundary), so dispatch.cpp `extern`-declares it with the mirror structs it already
has. Body (mirror `mega_decoder_real_adamw.cu` exactly):

```cpp
#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"
namespace sg { namespace fused { namespace sm90 {
cudaError_t mega_vit_real_adamw(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* workspace, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream) {
    const int64_t total = kVitTotalElems;
    FusedOptState st;
    st.exp_avg = state; st.exp_avg_sq = state + total;
    apply_scalars(st, scalars); st.lr = lr;
    ViTInputCtx in; in.patches = patches; in.targets = targets; in.B = B;
    in.workspace = workspace; in.loss_out = loss_out;
    return launch_fused_vit_megakernel<OptId::AdamW>(
        ctx, params, in, grad, lr, step, st, stream);
}
}}}
```

dispatch.cpp `extern` decl (host side, mirror structs only):
```cpp
namespace sg { namespace fused { namespace sm90 {
struct FusedScalars; // (the existing mirror)
cudaError_t mega_vit_real_adamw(fused::PersistentContext, float*, const float*,
    const int*, int, float*, float*, float*, float*, float, int,
    const FusedScalars&, cudaStream_t);
}}}
```

> The 33 generated **surrogate** cells (`mega_vit_*.cu` etc.) stay UNTOUCHED, so
> no cell/table regeneration is forced and `git` stays generator-consistent — the
> decoder did exactly this.

---

## 5. Codegen adoption of the layout header (so it can't drift)

`vit_layout.cuh` is **hand-written but generated FROM** the single source of truth
(`vit_oracle.py::vit_param_layout()`) and guarded by static_asserts. To make it
**generator-owned** like the decoder's, add to `megakernel_codegen.py`:

* `_VIT_VOCAB, _VIT_D, _VIT_HEADS, _VIT_LAYERS, _VIT_PATCH, _VIT_NPATCH = 97,128,4,2,49,16`
  and `_VIT_DFF = 4*_VIT_D`.
* `_vit_param_sizes() -> List[int]` returning numel in named_parameters() order:
  `[1*1*d, d*patch, d, (npatch+1)*d]` then per layer `[3d*d,3d, d*d,d, d,d,d,d,
  dff*d,dff, d*dff,d]` ×2, then `[d,d, v*d,v]` — **identical numbers to
  vit_param_layout()** (32 tensors, total 418017).
* `vit_layout_header()` modeled on `decoder_layout_header()` emitting
  `csrc/fused/sm_90/vit_layout.cuh` (the SG_VIT_* constants, `kVitOffsets`,
  `kVitSizes`, the count/total static_asserts, the host-constexpr consistency
  fold, AND the smem-budget block — keep `kVitSampleSmemBytes == 188080` and the
  `< 227*1024` assert).
* a `--vit-layout` CLI flag (mirror `--decoder-layout`).

Consistency check (must be a no-op diff once adopted):
```bash
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --vit-layout \
    > csrc/fused/sm_90/vit_layout.cuh && git diff --stat   # EXPECT: no changes
```

(Until adopted, the static_asserts + the no-GPU layout test
`test_vit_layout_matches_named_parameters` are the drift guard.)

---

## 6. Python wiring (`grokking_optimizers/dispatch.py`)

Mirror the decoder's PHASE-1 hooks for `"vit"`:

* `_VIT_TOTAL_ELEMS = 418017` (the no-GPU test cross-checks this against
  `vit_param_layout()["total"]` if present).
* `has_l3_real("vit", "adamw")` → True on sm_90 with the built extension exposing
  `fused_step` AND the dispatch.cpp branch above (the readiness gate; loud-None,
  never a silent stub).
* `fused_train_step("vit", opt, module, optimizer, patches, targets, ...)` — packs
  the float `input` (patch pixels ++ int-target-bits per §4), owns the persistent
  flat-param + `[m|v|extra]+loss` state buffers (keyed by model on the optimizer
  instance, allocated once — reallocating resets momentum → never groks), calls
  `ops.fused_step(..., opt_only=False)`, scatters updated params back, reads the
  loss slot, and (if `return_grad=True`) returns the reduced `grad` tensor.
  **Signature must match** `test_vit_megakernel.py::_run_vit_l3_real_step` (it
  calls `fused_train_step("vit","adamw", m, _Opt(), patches, targets, lr=...,
  betas=..., weight_decay=..., eps=..., state_cache=state, step=step,
  return_grad=...)`).
* The race's ViT `train_adamw` loop calls a `_try_fused_train_step(...)` FIRST for
  the vit cell (mirror the decoder hook): if the L3-REAL kernel is available it
  runs the whole step (fwd+bwd+adamw) and the loop skips its own eager
  fwd/bwd/`optimizer.step()`; otherwise falls back to eager. `eval_every` stays
  eager.
* (Optional, for test E) a `tests/hw/_vit_grok_smoke.py::vit_grok_smoke_impl(seed)`
  that trains the ViT cell through the race budget and returns best test acc — the
  ViT analogue of the decoder's `_grok_smoke_impl`. The test skips cleanly if
  absent.

---

## 7. Validation (operator, on a real sm_90 device after the GPU job releases)

```bash
# no-GPU correctness gates (run anywhere; the rigor substitute for the un-runnable .cu):
PYTHONPATH=. python -m pytest tests/hw/test_vit_megakernel.py \
    -k "oracle or mirror or layout or smem" -q          # EXPECT: 4 passed

# (after the seam lands) generator-consistency of the layout header:
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --vit-layout \
    > csrc/fused/sm_90/vit_layout.cuh && git diff --stat # EXPECT: no diff

# clean rebuild (the dispatch.cpp behavior changed + the new .cu must compile):
PYTHONPATH=. pip install -e . --no-build-isolation        # or rm -rf build/ first

# GPU parity + grok (sm_90):
PYTHONPATH=. python -m pytest tests/hw/test_vit_megakernel.py -m hw -q

# end-to-end: the ViT race now takes the L3-REAL fused train step:
PYTHONPATH=. python grokking_race_v2.py --tasks vit --num-seeds 3 \
    --eval-every 50 --output results/l3_real_vit
```

Expected GPU results (the gates encoded in `test_vit_megakernel.py`):
* C single-step: kernel loss within **1e-5 rel** of eager; **every** weight grad
  within **1e-4 rel** of the oracle (per-tensor dump); params after 1 step
  **1e-5 rel**.
* D trajectory: 200-step loss curve within **1e-3 rel** of eager AdamW; final
  params **1e-3 rel** (fp32 drift).
* E grok: 3 seeds, best test acc **≥ 0.95** through the race's ViT train path.

---

## 8. Known scope / caveats (PHASE 2)

* **fp32 compute is the correctness baseline.** bf16 compute is a flagged
  follow-up (`TODO[bf16]` in `model_stage_vit.cuh`): it would roughly halve the
  ~184 KB smem and default to THIS fp32 path. The race's bf16-autocast default
  applies to the EAGER fallback, not the fused kernel (which raises/falls back
  under AMP — a loud None, never a silent stub).
* **vit × adamw on sm_90 only.** Other optimizers / models / gfx942 keep the
  eager (+ L1) path; only this one cell gets the real fwd+bwd+opt kernel (the
  same scope the decoder PHASE 1 took).
* **The surrogate `mega_vit_*.cu` cells remain compiled but unreachable on the
  race path** (the vit real path is routed before them in `fused_step`).
  Documented, not hidden.
* **No CUDA graphs** (owner directive) — the persistent megakernel + in-kernel
  grid barriers are the chosen "zero intermediate launches" mechanism.
* **Persistent flat-param + optimizer state are driver-owned**, allocated once
  and never reallocated (reallocating resets momentum → never groks). NOT
  checkpointed by `optimizer.state_dict()`.
* **The model must be eager** (not `torch.compile`-wrapped) on the fused path —
  the flat layout assumes the eager `named_parameters()` order. The race builds
  the ViT eager by default (`compile_model=False`).
* **Per-step workspace** is `nCTA × 418017` fp32 (~221 MB at 132 SMs), allocated
  once. A warp-per-sample concurrent variant + bf16 tensor-core matmuls are the
  perf follow-ups; this is a correctness/wiring deliverable.
