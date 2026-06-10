# INTEGRATION-MAMBA.md — wiring the L3-REAL Mamba megakernel (PHASE 2)

This is the integrator's cell/dispatch/binding spec for the **real Mamba
(selective-SSM) forward+backward as persistent-megakernel stages**. The model
stages, the weight-layout header, the oracle/mirror, and the hardware-gated test
are **already built and CPU-validated** (see "What ships"). What remains is the
**composition+launcher** and the **integration seam** — the megakernel pipeline
header, the generated cell `.cu`, the `dispatch.cpp` routing, the codegen layout
emitter, and the Python `fused_train_step` wiring — which the PHASE-2 directive
reserves for the integrator / sibling agents who own `fused_megakernel`,
`megakernel_codegen.py`, `dispatch.cpp`, and `csrc/bindings/*`.

This file is the exact contract for that seam. It mirrors PHASE 1's decoder
integration (BUILD_AND_VALIDATE.md "PHASE 1") and the PHASE-2 ViT seam
(INTEGRATION-VIT.md), with the **Mamba-specific deltas** called out loudly.
Source-complete only: no builds, no GPU runs were performed.

---

## 0. What ships in this phase (already built + validated, do NOT re-derive)

| file | role | status |
|---|---|---|
| `tests/hw/mamba_oracle.py` | manual fwd+bwd ORACLE + `mamba_param_layout()` (THE single source of truth for the flat layout) **incl. the selective-scan backward derivation** | ✅ matches autograd, loss diff **8.9e-16**, worst grad rel **1.3e-15** (fp64, 28/28 tensors) |
| `tests/hw/mamba_kernel_mirror.py` | single-threaded STRUCTURAL mirror of the kernel (per-channel register scan, scan-bwd recompute+reverse, NON-causal conv transpose, 3-path dx_main, owner-thread accumulation, token collision) | ✅ matches oracle, loss + worst grad rel **< 1e-6** (run with B=2 + forced token collision) |
| `csrc/fused/sm_90/model_stage_mamba3.cuh` | REAL Mamba fwd/bwd stages (`mb_forward_sample` / `mb_backward_sample`), self-contained per COMPONENT_CONTRACT.md | source-complete (front-end + `-ptx` codegen clean; not built into the project) |
| `csrc/fused/sm_90/mamba3_layout.cuh` | hand-written weight-layout mirror, **static_asserts** on count(28)/total(259425)/smem-budget(`kMambaSmemBytes`); **marked for codegen adoption** | source-complete (front-end clean) |
| `tests/hw/test_mamba_megakernel.py` | the parity ladder (no-GPU oracle/mirror/layout + GPU-gated single-step/trajectory/grok) | ✅ no-GPU gates PASS; GPU gates SKIP until this seam lands |

The kernel's MATH and STRUCTURE are therefore already proven on CPU; the seam
below only composes the stages into a kernel and routes data to it. **Do not
modify** `model_stage_mamba3.cuh` / `mamba3_layout.cuh` / the oracle / the mirror
to make integration easier — adapt the seam to them.

---

## 1. The architecture, pinned (cites grokking_race_v2.py)

`_raw_model` → `MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2)` (lines
451-462, 477; `MODEL_SCALES["small"]` → d=128/nl=2):

| component | def | shape |
|---|---|---|
| `tok` | `nn.Embedding(99,128)` (:454) | [99,128] — **99 token rows** |
| `pos` | `nn.Embedding(8,128)`, `arange(8)` (:454,:457) | [8,128] |
| 2× `SelectiveSSMLayer(d=128)` (:455) | d_inner=256(=d·2), state_dim=16, dt_rank=max(d/16,1)=8 (:408-411) | |
| `A_log` | `log(arange(1,17)).expand(256,16)`; `A=-exp(A_log)` (:417-418,:423) | [256,16] |
| `D` | `ones(256)` (:419) | [256] |
| `in_proj` | `nn.Linear(128, 2·256=512, bias=False)` (:412) | [512,128] |
| `conv1d` | `nn.Conv1d(256,256,k=3,pad=1,groups=256,bias=True)` **NON-CAUSAL** (:413-414) | [256,1,3]+[256] |
| `x_proj` | `nn.Linear(256, 8+2·16=40, bias=False)` (:415) | [40,256] |
| `dt_proj` | `nn.Linear(8, 256, bias=True)` (:416) | [256,8]+[256] |
| `out_proj` | `nn.Linear(256, 128, bias=False)` (:420) | [128,256] |
| `norm` (per-layer) | `nn.LayerNorm(128)`, eps **1e-5** (:421) | [128]×2 |
| layer fwd | `residual=x; x_main,z=in_proj(x).chunk(2); x_main=SiLU(conv1d(x_main)); dt,B,C=x_proj(x_main).split([8,16,16]); y=scan(x_main,dt_proj(dt),B,C); y=out_proj((y+x_main·D)·SiLU(z)); return norm(y+residual)` (:442-449) | |
| `norm` (final) | `nn.LayerNorm(128)` (:456) | [128]×2 |
| `out` | `nn.Linear(128, 97)` (:456) | [97,128]+[97] — **head width p=97** |
| forward | `h=tok(x)+pos(pos_ids); for l: h=l(h); return out(norm(h[:,-1,:]))` (:460-462) | **LAST token** |
| loss | `F.cross_entropy(m(tx), ty)` (:745…) | mean over B |

`selective_scan(x,dt,B,C)` (:422-441 — the PYTHON FALLBACK the race runs, since
`mamba_scan_kernel.cu` is absent and CPU forces `is_cuda=False`; this fallback +
torch.autograd is the ground truth):
`A=-exp(A_log); dt=softplus(dt); h=0; for t: h=exp(dt_t·A)·h + (dt_t·B_t)·x_t; y_t=Σ_s C_t·h_t`.

**28 parameter tensors, 259,425 total params.** Numerics (each verified against
autograd in `mamba_oracle.py`): **SiLU = x·σ(x)**; **softplus = log1p(exp)**
(applied INSIDE the scan; dt_proj's bias is OUTSIDE); **conv1d NON-causal** (k=3
pad=1: `y[t]=b+W0·x[t-1]+W1·x[t]+W2·x[t+1]`, zero-pad); **A=-exp(A_log)** →
`dA_log=dA·A`; LayerNorm **eps=1e-5**; CE **mean over B** →
`dlogits=(softmax−onehot)/B`.

### The Mamba-specific deltas vs the decoder seam

1. **seq = 8** (decoder seq=4); **head reads the LAST position** (like the
   decoder, UNLIKE ViT's CLS-pos-0). INPUT is **int32 tokens** `[B,8]` + int32
   targets `[B]` — the SAME int32 packing the decoder uses (§4), NOT ViT's float
   patches.
2. **28 tensors / 259425 elems** (decoder 30/422755). **CRITICAL ORDERING:**
   within each `SelectiveSSMLayer`, `A_log` and `D` are yielded BEFORE `in_proj`
   (a module yields its OWN `nn.Parameter`s before its submodules — NOT the
   `__init__` visual order). `mamba3_layout.cuh` + the oracle encode this; it is
   asserted against the live `named_parameters()` in
   `test_mamba_layout_matches_named_parameters`.
3. **Head width p=97 ≠ the 99-token embedding** (`out=Linear(128,97)`; `tok` has
   99 rows). The kernel uses `kPHead=97` for logits/CE and `kVocab=99` only for
   the tok-embedding scatter.
4. **The selective scan is the new hard primitive** (no attention). It is the
   per-channel register recurrence + reverse-time backward (§3, §0). There is NO
   causal-attention machinery in this model.

---

## 2. THE one hard requirement: dynamic shared memory (don't drop it)

Mamba's per-sample working set `MambaSampleSmem` is **145,124 bytes (≈141.72 KB)**
— **far over the 48 KB STATIC `__shared__` cap** (a static
`__shared__ MambaSampleSmem sm;` would **not compile**). This is an HONEST
deviation from the decoder's <48 KB static footprint: the decoder's
recompute-in-backward existed solely to clear 48 KB, but Mamba's **d_inner=256**
makes even one layer's activation set bump the cliff and the full live set is
~74 KB regardless — so recompute buys nothing, and the stages instead **cache both
layers' forward activations** (the clean, bug-free choice; the scan state stays in
per-thread registers, so smem does NOT explode to the ~128 KB an h-in-smem scan
would need).

The composition launcher (`launch_fused_mamba_megakernel<Opt>`, to be written —
§3) MUST therefore declare the CTA smem **dynamic** and do **all three**:

```cpp
const int dyn_smem = (int)sg::fused::sm90::kMambaSmemBytes;     // 145124
cudaFuncSetAttribute(&fused_mamba_megakernel<Opt>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);     // (1) opt-in >48KB
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ..., dyn_smem); // (2) honest occ
fused_mamba_megakernel<Opt><<<grid, 256, dyn_smem, stream>>>(...);  // (3) launch
```

and the kernel allocates the struct from dynamic smem:
`extern __shared__ char smem_raw[]; auto& sm = *reinterpret_cast<MambaSampleSmem*>(smem_raw);`

**145124 B ≪ the sm_90 ~227 KB per-block dynamic-optin cap** (≈82 KB headroom;
`kMambaSmemBytes < 224*1024` is static_asserted in `mamba3_layout.cuh`), and the
persistent megakernel is **one-CTA-per-SM by design** (gridDim = #SMs, grid
barrier over #SMs), so **occupancy = 1 IS the design point** — a 142 KB block at
occ=1 is intended, not a regression. **Missing any of the three steps makes the
kernel silently fail to launch.** The launcher's only job beyond the decoder's is
to pass `dyn_smem` (NOT 0) in all three places, on a device whose
`MaxDynamicSharedMemoryPerBlockOptin ≥ 145124` (every sm_90 satisfies this; the
launcher should `assert(occ>=1)` and REFUSE — return
`cudaErrorLaunchOutOfResources` — rather than hang the grid barrier, exactly like
`launch_fused_decoder_megakernel`).

---

## 3. Composition + stage / barrier layout (one persistent kernel)

`fused_mamba_megakernel.cuh` (to be written, modeled BYTE-FOR-BYTE on
`fused_decoder_megakernel.cuh` with the dynamic-smem changes from §2 and the
Mamba constants). gridDim = #SMs (one CTA/SM), 256 threads/CTA, **batch-parallel**
(NOT the param-tensor work-steal queue — that would vary the batch→CTA grouping
and fp32 sums aren't associative):

```
P0  each CTA zeroes its OWN grad-partial slice + loss slot
--- grid barrier B0 ---
P1  each CTA owns a FIXED contiguous batch slice (blockIdx.x); processes its
    samples ONE AT A TIME (CTA-cooperative): broadcasts the sample's 8 int tokens
    + target into smem, runs mb_forward_sample + mb_backward_sample, accumulating
    each sample's weight-grad into the CTA's partial with a SINGLE-OWNER-THREAD-
    PER-ELEMENT rule (no atomics → deterministic), and sums its slice's NLL (fp32)
--- grid barrier B1 ---
P2  deterministic cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA index
    into the global grad (no float atomics; order fixed → reuses the work-steal
    queue to pick WHO reduces which tensor). Loss: fp64 ordered sum → loss/B → a
    device float the host reads back.
--- grid barrier B2 (sync_reset: also resets the queue for P3) ---
P3  the REAL apply_optimizer<Opt> tail consumes the reduced grad in place
```

`n_tasks = 28` (`kMambaNumTensors`) for the reduce + optimizer phases; the kernel
reads per-tensor numel/offset from the generated `__constant__` tables
`kMambaSizes/kMambaOffsets` (mamba3_layout.cuh). Determinism: fixed batch→CTA
grouping + fixed ascending-CTA reduction order; **no float atomics** in the
weight-grad reduce.

**Mamba-specific determinism notes (already handled in the stages — the integrator
does NOT re-derive, but should understand the contract):**
* the `tok.weight` scatter maps **thread→embedding column** and loops
  `(position)` sequentially, so colliding token ids (vocab 99) accumulate
  deterministically (the mirror's forced collision proves it).
* the selective-scan backward's `dB`/`dC` are SHARED across the d_inner channels,
  so they are reduced across threads via `mb_block_sum` in a **fixed ascending
  lane/warp order** (deterministic), NOT atomically. `dA_log` is per-channel
  (owner thread j) — a plain partial `+=`. `dx_main` (3 paths) and `ddt_pre` are
  per `(t,channel)` — owner thread j, plain writes.

The `fused_mamba_megakernel` ALSO needs the `MambaTokenCtx` POD (mirror
`DecoderTokenCtx`: `const int* tokens; const int* targets; int B; float*
workspace; float* loss_out;`) and the `rebase_state<Opt>` helper — both already
exist in `fused_decoder_megakernel.cuh`; lift the pattern verbatim. (This header
does NOT exist yet because the directive scoped this phase to the stages+layout+
oracle/mirror+doc; it is straightforward to write from the decoder template.)

**SMEM budget per CTA: `MambaSampleSmem` = 145124 B dynamic** (BOTH layers cached;
the scan state is per-thread registers).

---

## 4. ABI seam (no `bindings.cpp` / `setup.py` edit — the owned extension points)

The pybind `m.def("fused_step", ...)` arity is pinned in `bindings.cpp` (NOT
owned), so `fused_step`'s arity is unchanged. Carry the Mamba input through the
EXISTING `input`/`state`/`grad` tensors and add the **behavior** in `dispatch.cpp`
(owned), **identically to the decoder** (Mamba's input is int tokens, like the
decoder — NOT ViT's float pack).

**Tensor contract (what Python passes; what dispatch.cpp reads):**

* `input` = **int32** `[B*(8+1)]` contiguous — tokens `[B*8]` (row-major `[B][8]`)
  then targets `[B]`. `B = input.numel() / (kSeq+1) = numel / 9`. dispatch.cpp
  reads `tokens = input.data_ptr<int>()`, `targets = input.data_ptr<int>() + B*8`.
  (Seq/vocab/d/p are compile-time in `mamba3_layout.cuh`.)
* `params` = flat `[259425]` fp32 (the `torch.cat` of `named_parameters()` in
  order — see `mamba_param_layout()` / `mamba3_layout.cuh`).
* `state` = `[m|v|extra]` (`3*259425`) **+ 1 trailing loss slot** the kernel
  writes the mean CE into (read back in Python). AdamW uses only `m|v`.
* `grad` = the **reduced weight-grad OUTPUT** `[259425]` — the kernel writes the
  deterministically-reduced grad here (P2) and the optimizer tail consumes it in
  place (P3, no overwrite), so after the call this buffer holds exactly the grad
  AdamW used. **Routing it through the ABI `grad` tensor exposes the kernel's
  grads to the parity test** (`return_grad=True`) — the keystone check (§7 C.2),
  the only thing that exercises the hand-written selective-scan backward's
  magnitudes (loss is fwd-only; params-after-step is sign-dominated at step 1).
* the **workspace** (`nCTA*259425` grad partials + `nCTA` loss) is device scratch
  allocated in `dispatch.cpp` (keyed by device, like `decoder_scratch_for`) — it
  never crosses the ABI. At 132 SMs this is `132 × 259425 × 4 B ≈ 137 MB`,
  allocated once.

**routing** (add to `dispatch.cpp`, the `#if defined(WITH_CUDA)&&!defined(WITH_HIP)`
block, mirroring the decoder branch — the decoder branch is the exact template):

```cpp
if (arch == 90 && model == "mamba" && optimizer == "adamw" && !opt_only) {
    const int64_t total = kMambaTotalElems;        // 259425 (add a mirror const)
    TORCH_CHECK(params.numel() == total && params.scalar_type()==at::kFloat
                && params.is_contiguous(), "...");
    TORCH_CHECK(input.scalar_type()==at::kInt && input.is_contiguous(), "...");
    const int64_t in_n = input.numel();
    TORCH_CHECK(in_n % (8 + 1) == 0 && in_n > 0, "...");          // kSeq+1 = 9
    const int B = (int)(in_n / (8 + 1));
    TORCH_CHECK(state.numel() >= 3*total + 1, "...");
    TORCH_CHECK(grad.numel() == total, "...");
    MambaScratch& msc = mamba_scratch_for(params);   // new device-keyed scratch
    fused::PersistentContext ctx{ msc.g_next..., msc.g_arrived..., msc.g_gen...,
                                  28 /*kMambaNumTensors*/, 0u };
    float* m = state.data_ptr<float>();
    float* loss_slot = m + 3*total;
    fused::sm90::FusedScalars scalars{ /* full set, as the decoder branch */ };
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = fused::sm90::mega_mamba_real_adamw(
        ctx, params.data_ptr<float>(),
        input.data_ptr<int>(),                                   // tokens
        input.data_ptr<int>() + (int64_t)B*8,                    // targets
        B, m, grad.data_ptr<float>(), msc.workspace.data_ptr<float>(),
        loss_slot, lr, (int)step, scalars, stream);
    if (err != cudaSuccess) throw std::runtime_error(...);
    return;
}
```

Add near `kDecoderTotalElems` (dispatch.cpp:333): `constexpr int64_t
kMambaTotalElems = 259425;` and a `MambaScratch` + `mamba_scratch_for(params)`
modeled on `DecoderScratch`/`decoder_scratch_for` (workspace sized
`(int64_t)n_sms * 259425 + n_sms + 1`).

**The cell TU** — `csrc/fused/sm_90/mega_mamba_real_adamw.cu` (new; picked up by
setup.py's `csrc/fused/sm_90/*.cu` glob, like `mega_decoder_real_adamw.cu`).
dispatch.cpp is HOST-compiled, so ALL `<<<>>>`/`__global__`/device code lives in
this nvcc TU. It exposes ONE non-template host launcher whose boundary signature
is plain pointers/ints + the `FusedScalars` POD (NO header-only types cross the
boundary), so dispatch.cpp `extern`-declares it with the mirror structs it already
has. Body (mirror `mega_decoder_real_adamw.cu` exactly):

```cpp
#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"
namespace sg { namespace fused { namespace sm90 {
cudaError_t mega_mamba_real_adamw(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* workspace, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream) {
    const int64_t total = kMambaTotalElems;
    FusedOptState st;
    st.exp_avg = state; st.exp_avg_sq = state + total;
    apply_scalars(st, scalars); st.lr = lr;
    MambaTokenCtx tok; tok.tokens = tokens; tok.targets = targets; tok.B = B;
    tok.workspace = workspace; tok.loss_out = loss_out;
    return launch_fused_mamba_megakernel<OptId::AdamW>(
        ctx, params, tok, grad, lr, step, st, stream);
}
}}}
```

dispatch.cpp `extern` decl (host side, mirror structs only):
```cpp
namespace sg { namespace fused { namespace sm90 {
struct FusedScalars; // (the existing mirror)
cudaError_t mega_mamba_real_adamw(fused::PersistentContext, float*, const int*,
    const int*, int, float*, float*, float*, float*, float, int,
    const FusedScalars&, cudaStream_t);
}}}
```

> The generated **surrogate** cells (`mega_mamba_*.cu` etc.) stay UNTOUCHED, so no
> cell/table regeneration is forced and `git` stays generator-consistent — the
> decoder + ViT seams did exactly this.

---

## 5. Codegen adoption of the layout header (so it can't drift)

`mamba3_layout.cuh` is **hand-written but derived FROM** the single source of truth
(`mamba_oracle.py::mamba_param_layout()`) and guarded by static_asserts. To make it
**generator-owned** like the decoder's, add to `megakernel_codegen.py` (OFF-LIMITS
this phase — this is the integrator's task):

* `_MAMBA_VOCAB,_MAMBA_PHEAD,_MAMBA_D,_MAMBA_LAYERS,_MAMBA_SEQ = 99,97,128,2,8`
  and `_MAMBA_DINNER=256, _MAMBA_STATE=16, _MAMBA_DTRANK=8, _MAMBA_CONVK=3`.
* `_mamba_param_sizes() -> List[int]` returning numel in named_parameters() order
  — `[vocab*d, seq*d]` then per layer **in the leaf-before-submodule order**
  `[d_inner*state (A_log), d_inner (D), 2*d_inner*d (in_proj), d_inner*1*3
  (conv_w), d_inner (conv_b), (dt_rank+2*state)*d_inner (x_proj),
  d_inner*dt_rank (dt_proj_w), d_inner (dt_proj_b), d*d_inner (out_proj),
  d (n_w), d (n_b)]` ×2, then `[d (norm_w), d (norm_b), phead*d (out_w),
  phead (out_b)]` — **identical numbers to `mamba_param_layout()`** (28 tensors,
  total 259425).
* `mamba_layout_header()` modeled on `decoder_layout_header()` emitting
  `csrc/fused/sm_90/mamba3_layout.cuh` (the SG_MB_* constants, `kMambaOffsets`,
  `kMambaSizes`, the count/total static_asserts, the host-constexpr consistency
  fold, AND the smem-budget block — keep `kMambaSmemFloats == 36281` /
  `kMambaSmemBytes == 145124` and the `>48*1024` + `<224*1024` asserts).
* a `--mamba-layout` CLI flag (mirror `--decoder-layout`).

Consistency check (must be a no-op diff once adopted):
```bash
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --mamba-layout \
    > csrc/fused/sm_90/mamba3_layout.cuh && git diff --stat   # EXPECT: no changes
```

(Until adopted, the static_asserts + the no-GPU layout test
`test_mamba_layout_matches_named_parameters` are the drift guard.)

> **If you re-emit `kMambaSmemFloats` from the generator**, compute it field-by-
> field from `MambaSampleSmem` (the comment block in `mamba3_layout.cuh` lists
> every field's element count) — do NOT read `sizeof()` at codegen time (the
> generator is host Python, not nvcc). The 36281 number is: layer_in 2048 +
> final_in 1024 + 2×(LayerAct 11528) + fn_xhat 1024 + fn_inv 8 + logits 97 + dh
> 1024 + dr 1024 + adj_a/adj_b/adj_c 3×2048 + dbc 320 + dBmat/dCmat 2×128 + red 256.

---

## 6. Python wiring (`grokking_optimizers/dispatch.py`)

Mirror the decoder's PHASE-1 hooks for `"mamba"`:

* `_MAMBA_TOTAL_ELEMS = 259425` (the no-GPU test can cross-check this against
  `mamba_param_layout()["total"]`).
* `has_l3_real("mamba", "adamw")` → True on sm_90 with the built extension exposing
  `fused_step` AND the dispatch.cpp branch above (the readiness gate; loud-None,
  never a silent stub).
* `fused_train_step("mamba", opt, module, optimizer, tokens, targets, ...)` — packs
  the int32 `input` (tokens `[B*8]` ++ targets `[B]`, §4), owns the persistent
  flat-param + `[m|v|extra]+loss` state buffers (keyed by model on the optimizer
  instance, allocated once — reallocating resets momentum → never groks), calls
  `ops.fused_step(..., opt_only=False)`, scatters updated params back, reads the
  loss slot, and (if `return_grad=True`) returns the reduced `grad` tensor.
  **Signature must match** `test_mamba_megakernel.py`'s GPU-gated helpers when
  they are un-skipped (they currently `pytest.skip` pending this seam).
* The race's Mamba `train_adamw` loop calls a `_try_fused_train_step(...)` FIRST
  for the mamba cell (mirror the decoder hook): if the L3-REAL kernel is available
  it runs the whole step (fwd+bwd+adamw) and the loop skips its own eager
  fwd/bwd/`optimizer.step()`; otherwise falls back to eager. `eval_every` stays
  eager. **The model must be eager** (the race builds Mamba with
  `compile_model=False`; `grad_checkpoint` is irrelevant to the fused path since
  the kernel owns the whole fwd+bwd).

---

## 7. Validation (operator, on a real sm_90 device after the GPU job releases)

```bash
# no-GPU correctness gates (run anywhere; the rigor substitute for the un-runnable .cu):
PYTHONPATH=. python -m pytest tests/hw/test_mamba_megakernel.py \
    -k "oracle or mirror or layout" -q                 # EXPECT: 3 passed (~60s; the
                                                       # mirror is single-threaded fp64)

# (after the seam lands) generator-consistency of the layout header:
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --mamba-layout \
    > csrc/fused/sm_90/mamba3_layout.cuh && git diff --stat # EXPECT: no diff

# clean rebuild (the dispatch.cpp behavior changed + the new .cu must compile):
PYTHONPATH=. pip install -e . --no-build-isolation        # or rm -rf build/ first

# GPU parity + grok (sm_90):
PYTHONPATH=. python -m pytest tests/hw/test_mamba_megakernel.py -m hw -q

# end-to-end: the Mamba race now takes the L3-REAL fused train step:
PYTHONPATH=. python grokking_race_v2.py --tasks mamba --num-seeds 3 \
    --eval-every 50 --output results/l3_real_mamba
```

Expected GPU results (the gates encoded in `test_mamba_megakernel.py`, currently
`pytest.skip` until the seam lands):
* C single-step: kernel loss within **1e-5 rel** of eager; **every** weight grad
  within **1e-4 rel** of the oracle (per-tensor dump) — the keystone that
  exercises the selective-scan backward; params after 1 step **1e-5 rel**.
* D trajectory: 200-step loss curve within **1e-3 rel** of eager AdamW; final
  params **1e-3 rel** (fp32 drift).
* E grok: 3 seeds, best test acc **≥ 0.95** through the race's Mamba train path
  (the `(a÷b₁÷b₂÷b₃) mod 97` sequential-division task).

---

## 8. Known scope / caveats (PHASE 2)

* **fp32 compute is the correctness baseline.** bf16 compute is a flagged
  follow-up (`TODO(bf16)` in `model_stage_mamba3.cuh`): it would roughly halve the
  ~142 KB smem and default to THIS fp32 path. The scan recurrence in particular is
  exp/accumulation-heavy and would keep fp32 accumulators even under bf16 storage.
  The race's auto-precision for mamba is already **fp32** (`_AUTO_PRECISION`
  line 606), so there is no AMP mismatch to guard on the eager fallback.
* **mamba × adamw on sm_90 only.** Other optimizers / models / gfx942 keep the
  eager (+ L1) path; only this one cell gets the real fwd+bwd+opt kernel (the same
  scope the decoder PHASE 1 + ViT PHASE 2 took).
* **The surrogate `mega_mamba_*.cu` cells remain compiled but unreachable on the
  race path** (the mamba real path is routed before them in `fused_step`).
  Documented, not hidden.
* **No CUDA graphs** (owner directive) — the persistent megakernel + in-kernel
  grid barriers are the chosen "zero intermediate launches" mechanism.
* **The selective scan runs per-channel in registers** (the seq=8 exploit): one
  thread owns channel j∈[0,256), holds h[16] in registers, unrolls t=0..7; the
  backward recomputes that forward (h_hist[9][16] in registers — seq=8, so NO
  checkpoint, unlike the long-sequence `scan_backward_kernel` in
  `csrc/scan/mamba_scan_adapter.cuh`) then reverse-scans. This is what keeps the
  smem at ~142 KB instead of the ~128 KB/sample an h-in-smem scan would add. The
  derivation is written out in `mamba_oracle.py::selective_scan_backward` and
  cross-checked against the existing `scan_backward_kernel`.
* **Persistent flat-param + optimizer state are driver-owned**, allocated once and
  never reallocated (reallocating resets momentum → never groks). NOT checkpointed
  by `optimizer.state_dict()`.
* **Per-step workspace** is `nCTA × 259425` fp32 (~137 MB at 132 SMs), allocated
  once. A warp-per-sample concurrent variant + bf16 tensor-core matmuls are the
  perf follow-ups; this is a correctness/wiring deliverable.
* **`csrc/scan` is READ-ONLY for this phase** and was used only as a cross-check
  reference for the scan backward; the stages do NOT depend on it (they are
  self-contained per COMPONENT_CONTRACT.md: `megakernel_common.cuh` +
  `mamba3_layout.cuh` + CUDA only).
```
