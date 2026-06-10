# INTEGRATION-NOTES.md — SuperGrok2 meta-net in the persistent megakernel

Owner handoff for **`csrc/fused/sm_90/opt_stage_supergrok2.cuh`** (the full SG2
CSA/HCA/PEER/GRU meta-net as in-kernel stages of ONE persistent kernel). This
file tells the **integrator** (bindings.cpp / dispatch / combined-cell owner)
exactly what to wire. The header author may NOT touch `bindings.cpp`,
`dispatch.py`, or the generated cells — those are sibling-agent territory — so
the wiring is specified here, not done here.

The component is self-contained per `csrc/fused/COMPONENT_CONTRACT.md`: it
includes only the substrate (`megakernel_common.cuh`) + the algorithm header
(`csrc/algorithms/supergrok2.h`) + CUDA. No globals; all state flows through the
POD structs `SG2Weights` / `SG2State` / `SG2Scalars` (defined in the header).

---

## 0. What it is (one line)

`launch_sg2_meta_optimizer_tail<Dims, ParamT, GradT>(...)` runs the ENTIRE SG2
optimizer step (input-proj+sort-gather → CSA → HCA → matrix-GRU → PEER+experts →
sg2_apply_step) for **all** parameter tensors in **ONE persistent kernel** —
replacing the per-op path's ~15-20 launches/tensor (`csa_hca_step_one`).

**Launch count, per parameter tensor, per optimizer step:**
- before: **~15-20** kernel launches (+ host ATen mm/topk/softmax dispatches)
- after: **1 batched argsort prep** (one torch call for ALL tensors) **+ 1
  persistent kernel** (processes all tensors via the task queue).

The argsort is the ONE explicitly-retained pre-kernel step (honesty rail #5 —
see §5). Everything else is launch-free and HBM-round-trip-free for the meta
weights (the bundle is staged into shared memory once per CTA).

---

## 1. Binding to add to `csrc/bindings/bindings.cpp`

Add a host wrapper `sg2_meta_optimizer_tail(...)` next to `supergrok2_batched_step`
(~L1926) and register it in `PYBIND11_MODULE`. It mirrors the existing
`supergrok2_batched_step` weight/state argument list, plus the persistent-kernel
scratch the megakernel needs. **It does NOT touch the C++ in
`opt_stage_supergrok2.cuh`** — it marshals torch tensors into the POD structs and
calls the templated launcher.

### 1a. The `m.def` signature spec

```cpp
m.def("sg2_meta_optimizer_tail", &sg2_meta_optimizer_tail,
      "SuperGrok2 full meta-net as ONE persistent megakernel (launch-elimination "
      "of csa_hca_step_one). Consumes the per-tensor PACKED flat buffers + the "
      "pre-computed |grad|-ascending sort perms; runs CSA/HCA/GRU/PEER/apply "
      "in-kernel.",
      py::arg("params_packed"),       // [total]   float32, all tensors back-to-back
      py::arg("grads_packed"),        // [total]   float32
      py::arg("sharpness_packed"),    // [total]   float32
      py::arg("exp_avg_packed"),      // [total]   float32  (Adam m)
      py::arg("exp_avg_sq_packed"),   // [total]   float32  (Adam v)
      py::arg("mu_packed"),           // [total]   float32  (expert-output EMA)
      py::arg("slow_packed"),         // [total]   float32  (grokfast slow EMA)
      py::arg("gru_state_packed"),    // [total*gru_hidden] float32, row-major [total,gh]
      py::arg("perm_packed"),         // [total]   int32  sorted-row -> original-row (per tensor)
      py::arg("unsort_packed"),       // [total]   int32  original-row -> sorted-row (per tensor)
      py::arg("n_per_tensor"),        // [P]       int32  element count of each tensor
      py::arg("row_off"),             // [P]       int64  start element of each tensor
      py::arg("workspace"),           // [n_ctas * ws_stride] float32 (kernel scratch)
      py::arg("ws_stride"),           // int64  floats per CTA (== sg2_ws_stride<Dims>(Nmax))
      // ── meta weight bundle (fp32; upcast once by the driver) ──
      py::arg("input_proj_W"), py::arg("input_proj_b"),
      py::arg("csa_q_W"), py::arg("csa_k_W"), py::arg("csa_v_W"), py::arg("csa_out_W"),
      py::arg("csa_compress_w"), py::arg("csa_idx_DQ"), py::arg("csa_idx_K"),
      py::arg("hca_q_W"), py::arg("hca_k_W"), py::arg("hca_v_W"), py::arg("hca_out_W"),
      py::arg("gru_Wz"), py::arg("gru_bz"), py::arg("gru_Wr"), py::arg("gru_br"),
      py::arg("gru_Wh"), py::arg("gru_bh"),
      py::arg("peer_query_Ws"), py::arg("prod_keys_A"), py::arg("prod_keys_B"),
      py::arg("expert_W1"), py::arg("expert_b1"), py::arg("expert_W2"), py::arg("expert_b2"),
      // ── per-tensor scalars (length P) + shared scalars ──
      py::arg("alpha"),       // [P] float  mu/slow mixing + slow-EMA decay (alpha_i)
      py::arg("gru_decay"),   // [P] float  expert-EMA (mu) decay (== beta1_i)
      py::arg("lamb_eff"),    // [P] float  grokfast amplification (lamb·ramp·gate)
      py::arg("beta1"),       // [P] float  Adam beta1 (layer-scaled)
      py::arg("bc1"),         // [P] float  1 - beta1_i^t
      py::arg("bc2"),         // [P] float  1 - beta2^t
      py::arg("rescale"),     // float  expert-output scale (shared)
      py::arg("beta2"),       // float
      py::arg("lr"),          // float
      py::arg("wd"),          // float  decoupled weight decay (wd_eff)
      py::arg("eps"),         // float
      // ── persistent-context scratch (host-allocated, zero-init each launch) ──
      py::arg("g_next_task"), // int32   [1]  TaskQueue counter
      py::arg("g_arrived"),   // uint32  [1]  GridBarrier arrival count
      py::arg("g_generation"));// uint32 [1]  GridBarrier generation
```

### 1b. The host wrapper body (what to write in bindings.cpp)

```cpp
void sg2_meta_optimizer_tail(/* args above */) {
  using namespace sg::fused::sm90;
  if (params_packed.numel() == 0) return;
  TORCH_CHECK(params_packed.scalar_type() == at::kFloat, "fp32 in this pass");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int P = (int)n_per_tensor.numel();

  SG2Weights w {
    input_proj_W.data_ptr<float>(), input_proj_b.data_ptr<float>(),
    csa_q_W.data_ptr<float>(), csa_k_W.data_ptr<float>(), csa_v_W.data_ptr<float>(),
    csa_out_W.data_ptr<float>(), csa_compress_w.data_ptr<float>(),
    csa_idx_DQ.data_ptr<float>(), csa_idx_K.data_ptr<float>(),
    hca_q_W.data_ptr<float>(), hca_k_W.data_ptr<float>(), hca_v_W.data_ptr<float>(),
    hca_out_W.data_ptr<float>(),
    gru_Wz.data_ptr<float>(), gru_bz.data_ptr<float>(),
    gru_Wr.data_ptr<float>(), gru_br.data_ptr<float>(),
    gru_Wh.data_ptr<float>(), gru_bh.data_ptr<float>(),
    peer_query_Ws.data_ptr<float>(), prod_keys_A.data_ptr<float>(),
    prod_keys_B.data_ptr<float>(),
    expert_W1.data_ptr<float>(), expert_b1.data_ptr<float>(),
    expert_W2.data_ptr<float>(), expert_b2.data_ptr<float>() };

  SG2State st {
    exp_avg_packed.data_ptr<float>(), exp_avg_sq_packed.data_ptr<float>(),
    mu_packed.data_ptr<float>(), slow_packed.data_ptr<float>(),
    gru_state_packed.data_ptr<float>(),
    perm_packed.data_ptr<int>(), unsort_packed.data_ptr<int>(),
    workspace.data_ptr<float>(), (int64_t)ws_stride, P,
    n_per_tensor.data_ptr<int>(), row_off.data_ptr<int64_t>() };

  SG2Scalars sc {
    alpha.data_ptr<float>(), gru_decay.data_ptr<float>(), lamb_eff.data_ptr<float>(),
    beta1.data_ptr<float>(), bc1.data_ptr<float>(), bc2.data_ptr<float>(),
    rescale, beta2, lr, wd, eps };

  PersistentContext ctx {};
  ctx.g_next_task  = g_next_task.data_ptr<int>();
  ctx.g_arrived    = (unsigned*)g_arrived.data_ptr<int>();      // int32 buffer reused
  ctx.g_generation = (unsigned*)g_generation.data_ptr<int>();
  ctx.n_tasks      = P;
  // n_ctas is set inside the launcher (== #SMs).

  // Dims = the supergrok2.py defaults (d_model=8, ...). If a config differs,
  // template-specialize the launcher for that shape (compile-time dims).
  using D = SG2Dims<>;  // defaults match OPTIMIZER_CONFIGS["supergrok2"]
  C10_CUDA_CHECK(launch_sg2_meta_optimizer_tail<D, float, float>(
      ctx, w, st, sc,
      params_packed.data_ptr<float>(), grads_packed.data_ptr<float>(),
      sharpness_packed.data_ptr<float>(), stream));
}
```

Notes:
- `g_arrived`/`g_generation` are `unsigned` in the substrate; pass int32 buffers
  and reinterpret (the substrate only does atomicAdd/atomicExch — sign-agnostic).
  Or declare them as int32 tensors and the cast above is exact.
- **Dispatch**: route only on a real sm_90 device. This is a NEW op, not a
  drop-in replacement — it does not alter the existing `supergrok2_batched_step`
  path (which stays the parity oracle).

---

## 2. Python driver call-shape

This is the driver the optimizer would call instead of the per-op
`ops.supergrok2_batched_step`. It is documented here (not added to the shared
`grokking_optimizers/dispatch.py`, which is sibling territory). Drop it into a
`@torch.no_grad()` helper on the SuperGrok2 optimizer once the binding exists.

```python
import torch
from grokking_optimizers.dispatch import get_ops
from csrc... import SG2Dims  # dims are compile-time in C++; mirror them in py for packing

def sg2_step_megakernel(self, params, grads, sharpness_list, states, meta, scalars):
    """ONE persistent-kernel SG2 optimizer step over ALL param tensors.
    states: dict of lists per tensor (exp_avg, exp_avg_sq, mu, slow, gru_state).
    meta:   the meta weight bundle (fp32). scalars: per-tensor + shared."""
    ops = get_ops()
    GH = 4  # gru_hidden (== meta net default)

    # 1) PACK all tensors back-to-back (one contiguous fp32 buffer each). The
    #    kernel addresses tensor t at row_off[t] .. row_off[t]+n[t].
    P = len(params)
    n = torch.tensor([p.numel() for p in params], dtype=torch.int32, device=dev)
    row_off = torch.zeros(P, dtype=torch.int64, device=dev)
    row_off[1:] = torch.cumsum(n.to(torch.int64), 0)[:-1]
    total = int(n.sum())

    def pack(lst, width=1):
        out = torch.empty(total * width, dtype=torch.float32, device=dev)
        for t, x in enumerate(lst):
            o = int(row_off[t]) * width
            out[o:o + x.numel()] = x.reshape(-1).float()
        return out

    params_packed = pack([p.data for p in params])
    grads_packed  = pack(grads)
    sharp_packed  = pack(sharpness_list)
    exp_avg_packed    = pack(states['exp_avg'])
    exp_avg_sq_packed = pack(states['exp_avg_sq'])
    mu_packed   = pack(states['mu'])
    slow_packed = pack(states['slow'])
    gru_state_packed = pack(states['gru_state'], width=GH)  # [total*GH] row-major

    # 2) THE ONE EXPLICIT PRE-KERNEL SORT (honesty rail #5). Per tensor, argsort
    #    |grad| ASCENDING, STABLE — reproducing csa_hca_step_one's
    #    `sort_keys.sort(0, descending=false)`. Build perm/unsort PACKED, with
    #    indices LOCAL to each tensor (in [0, n[t])).
    perm_packed   = torch.empty(total, dtype=torch.int32, device=dev)
    unsort_packed = torch.empty(total, dtype=torch.int32, device=dev)
    for t, gt in enumerate(grads):
        o, m = int(row_off[t]), int(n[t])
        # MATCH THE REFERENCE SORT EXACTLY. csa_hca_step_one does
        #   sort_keys.sort(0, descending=false)  — plain torch.sort, which is
        # NOT stable on CUDA. To stay bit-aligned with the parity oracle on tied
        # |grad| values, reproduce THAT call (do NOT force stable=True — that
        # would break ties differently and drift the GPU parity over 200 steps).
        _, perm = gt.reshape(-1).abs().sort(dim=0, descending=False)    # [m]
        unsort = perm.argsort()                                         # inverse perm
        perm_packed[o:o + m]   = perm.to(torch.int32)
        unsort_packed[o:o + m] = unsort.to(torch.int32)

    # 3) Per-CTA workspace. ws_stride = sg2_ws_stride<Dims>(Nmax) — mirror the C++
    #    formula in py (see csrc header) or expose it via a tiny binding. Allocate
    #    n_ctas (== #SMs) slices; n_ctas can be queried once and cached.
    Nmax = int(n.max())
    ws_stride = sg2_ws_stride_py(Nmax, GH=GH)   # mirror of the C++ sg2_ws_stride
    n_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    workspace = torch.empty(n_ctas * ws_stride, dtype=torch.float32, device=dev)

    # 4) Persistent-context scratch (zero-init; the launcher re-zeros each call).
    g_next_task  = torch.zeros(1, dtype=torch.int32, device=dev)
    g_arrived    = torch.zeros(1, dtype=torch.int32, device=dev)
    g_generation = torch.zeros(1, dtype=torch.int32, device=dev)

    # 5) Meta bundle as fp32 (upcast ONCE here — the kernel is fp32-compute).
    meta_f32 = {k: v.float().contiguous() for k, v in meta.items()}

    # 6) Per-tensor scalar arrays (length P) — the layer-scaled betas/alphas the
    #    per-op path already computes per tensor.
    alpha     = torch.tensor(scalars['alpha'],     dtype=torch.float32, device=dev)
    gru_decay = torch.tensor(scalars['gru_decay'], dtype=torch.float32, device=dev)
    lamb_eff  = torch.tensor(scalars['lamb_eff'],  dtype=torch.float32, device=dev)
    beta1     = torch.tensor(scalars['beta1'],     dtype=torch.float32, device=dev)
    bc1       = torch.tensor(scalars['bc1'],       dtype=torch.float32, device=dev)
    bc2       = torch.tensor(scalars['bc2'],       dtype=torch.float32, device=dev)

    ops.sg2_meta_optimizer_tail(
        params_packed, grads_packed, sharp_packed,
        exp_avg_packed, exp_avg_sq_packed, mu_packed, slow_packed,
        gru_state_packed, perm_packed, unsort_packed, n, row_off,
        workspace, ws_stride,
        meta_f32['input_proj_W'], meta_f32['input_proj_b'],
        meta_f32['csa_q_W'], meta_f32['csa_k_W'], meta_f32['csa_v_W'], meta_f32['csa_out_W'],
        meta_f32['csa_compress_w'], meta_f32['csa_idx_DQ'], meta_f32['csa_idx_K'],
        meta_f32['hca_q_W'], meta_f32['hca_k_W'], meta_f32['hca_v_W'], meta_f32['hca_out_W'],
        meta_f32['gru_Wz'], meta_f32['gru_bz'], meta_f32['gru_Wr'], meta_f32['gru_br'],
        meta_f32['gru_Wh'], meta_f32['gru_bh'],
        meta_f32['peer_query_Ws'], meta_f32['prod_keys_A'], meta_f32['prod_keys_B'],
        meta_f32['expert_W1'], meta_f32['expert_b1'], meta_f32['expert_W2'], meta_f32['expert_b2'],
        alpha, gru_decay, lamb_eff, beta1, bc1, bc2,
        scalars['rescale'], scalars['beta2'], scalars['lr'], scalars['wd'], scalars['eps'],
        g_next_task, g_arrived, g_generation)

    # 7) UNPACK results back into the per-tensor buffers (params + carried state).
    for t, p in enumerate(params):
        o, m = int(row_off[t]), int(n[t])
        p.data.copy_(params_packed[o:o + m].reshape(p.shape))
        states['exp_avg'][t].copy_(exp_avg_packed[o:o + m].reshape(-1))
        states['exp_avg_sq'][t].copy_(exp_avg_sq_packed[o:o + m].reshape(-1))
        states['mu'][t].copy_(mu_packed[o:o + m].reshape(-1))
        states['slow'][t].copy_(slow_packed[o:o + m].reshape(-1))
        states['gru_state'][t].copy_(
            gru_state_packed[o*GH:(o+m)*GH].reshape(m, GH))
```

The packing/unpacking is overhead the L3 combined cell can elide by keeping the
optimizer state PERMANENTLY packed (see §3). For an isolated optimizer tail the
copies are cheap relative to the eliminated launches.

### `sg2_ws_stride_py` (mirror of the C++ formula — keep in sync)

```python
def sg2_ws_stride_py(Nmax, d=8, rk=4, GH=4, csa_compress=4, csa_topk=16):
    Ncmax = (Nmax + csa_compress - 1) // csa_compress
    topk = max(csa_topk, 1)
    return (Nmax*d*7      # x_sorted, csa_ctx, hca_ctx, q, win_k, win_v, concat
            + Ncmax*d*2   # c_k, c_v
            + Nmax*rk     # qI
            + Ncmax*rk    # kI
            + Nmax*topk   # sel (int, 1 float each)
            + Nmax*GH     # new_gru
            + Nmax)       # expert_out
    # NOTE: the authoritative count is C++ sg2_ws_stride<Dims>(Nmax) — prefer
    # exposing THAT via a 1-line binding over duplicating it. The slot list is:
    # x_sorted, csa_ctx, hca_ctx, q, c_k, c_v, win_k, win_v, qI, kI, sel, concat,
    # new_gru, expert_out (14 slots).
```

> Strongly prefer binding `sg2_ws_stride<SG2Dims<>>(Nmax)` directly (one
> `m.def("sg2_ws_stride", ...)`) so the host stride and the kernel's carve can
> never drift. The py mirror above is a fallback for a quick bring-up only.

---

## 3. What the COMBINED (model-fused L3) cell must wire

This header is the **optimizer half** — the L1-style contract agent A
established, now with SG2's FULL meta-net inside one persistent kernel instead of
~20 launches. Composing it with a model fwd/bwd stage (decoder/vit/mamba3) into
ONE cell (`mega_<model>_supergrok2.cu`) is the integrator's job. The combined
cell must:

1. **Run the model fwd+bwd FIRST**, producing the REAL reduced gradient into the
   packed `grads_packed` buffer (same packing as §2 — the model's weight-grad
   reduction writes the flat grad the SG2 tail consumes). This is the L3
   structure of `fused_decoder_megakernel.cuh`: P0 zero → P1 fwd+bwd → P2
   deterministic reduce → **then** the optimizer tail.

2. **Share the PersistentContext**: the model stages and the SG2 stages use the
   SAME `ctx` (task queue + grid barrier). After the model's reduce phase, call
   `bar.sync_reset(ctx.g_next_task)` (as the decoder cell does at B2) to reset
   the queue for the SG2 tensor loop, THEN inline `sg2_meta_stages<Dims>` per
   tensor instead of `apply_optimizer<Opt>`. The ONE grid barrier the SG2
   launcher does at the end is then redundant inside the combined cell — drop it
   (the combined kernel exits after the apply, like the decoder cell's P3).

3. **Keep the optimizer state PERMANENTLY packed** so no per-step pack/unpack is
   needed — the model's grad reduction already writes into the packed layout, and
   the SG2 tail reads/writes the same packed state in place. The perm/unsort sort
   (§5) still runs as the one pre-kernel step (it needs the freshly-reduced grad).

4. **Workspace sizing**: the combined cell's workspace must hold BOTH the model's
   grad-partial workspace AND the SG2 per-CTA meta scratch (`ws_stride` slices).
   They are used in disjoint phases (model reduce, then SG2), so they MAY alias
   the same allocation if sized for the max — but simplest is two regions.

5. **`perm`/`unsort` must be recomputed AFTER the model bwd** (the grad is only
   known then). For a pure optimizer-only cell the driver computes them up front
   (§2); for the combined cell they are computed between the reduce phase and the
   SG2 phase — either a tiny argsort launch (the one retained launch) or, as the
   path-to-zero in §5, an in-kernel segmented-sort stage.

---

## 4. GPU test wiring (tests/hw/test_sg2_megakernel.py part B)

`tests/hw/test_sg2_megakernel.py` part **(A)** (the fp64 structural mirror vs the
clean oracle) runs TODAY with no GPU and is the primary correctness evidence:
machine-epsilon agreement (~1e-16) across N∈{5,17,64,200} and a 200-step
trajectory. Part **(B)** is hw-gated and currently `pytest.skip`s with the wiring
recipe; once `ops.sg2_meta_optimizer_tail` exists, replace the two skip bodies:

- **(B1) single-step**: build a small SuperGrok2 optimizer over a few random
  param tensors; snapshot state; run ONE `ops.supergrok2_batched_step` (the
  ORACLE) on a copy and ONE `sg2_step_megakernel` (§2) on another copy from the
  SAME init; assert `max|Δparam|, max|Δexp_avg|, max|Δexp_avg_sq|, max|Δmu|,
  max|Δslow|, max|Δgru_state| < 1e-5`.
- **(B2) trajectory**: 200 steps of a FIXED random-grad sequence through both
  paths; assert the per-step `mean|param|` proxy and the final `|Δparam|` stay
  `< 1e-5`.

The `1e-5` (not `1e-12`) tol is the documented parity-hotspot allowance: the
megakernel's GRU/PEER reductions are hand-written sequential dot products vs the
reference's cuBLAS/ATen `mm`/`matmul` (fp32 round-off only — the transcendentals
already match: accurate `expf`/`tanhf` in GRU/PEER, `ptx_expf` in attention to
match the reference kernels). See the `opt_stage_supergrok2.cuh` header
"PARITY (NOT bit-identity)" section.

---

## 5. The sort (honesty rail #5) — residual + path to zero

`csa_hca_step_one` sorts each tensor's sequence by `|grad|` ASCENDING via a HOST
`torch::sort`. **There is no device sort kernel in the repo** (verified). A
single device sort over the PACKED buffer would be WRONG: SG2 attention is
per-tensor (each flat grad vector is its own length-N sequence), so a global sort
would cross tensor boundaries. The sort is therefore left as **ONE explicit
pre-kernel step** (a per-tensor `torch.argsort`, §2 step 2) producing the
`perm`/`unsort` index buffers the kernel consumes.

> **Residual: 1 argsort prep + 1 persistent kernel, vs the previous ~15-20
> launches per tensor.** The argsort is ONE torch op for all tensors; everything
> else (CSA/HCA/GRU/PEER/apply) is launch-free.

**Path to zero extra launches**: add a device **segmented bitonic sort** as
STAGE -1 of the kernel — per tensor, sort its `[N]` keys (= `|grad|`) in shared
memory with barrier-separated bitonic passes, writing `perm`/`unsort`. N here is
a single tensor's element count (the largest race param is ~1e5), so an in-CTA
segmented sort is feasible. The header is written so this drops in cleanly: STAGE
-1 only needs to populate `perm`/`unsort` before STAGE 0; no other stage changes.
This was deliberately NOT done in this pass (it is a separate, larger piece of
work) to keep the pass at **maximal HONEST launch-elimination of everything
else** with bit-faithful math — per honesty rail #5.

**Sort tie-handling (subtle — get this right)**: reproduce the reference's
EXACT sort call `sort_keys.sort(0, descending=false)` (plain `torch.sort`, which
is NOT stable on CUDA) — NOT `argsort(stable=True)`. The megakernel and the
parity oracle must consume the SAME `perm`; on tied `|grad|` an unstable-vs-stable
mismatch breaks ties differently → different sorted order → CSA/HCA diverge past
1e-5 over the 200-step trajectory. The §2 driver uses the exact `.sort(...)` call.
(Lower-risk sibling: the PEER top-k uses descending insertion sort vs the
reference's `torch.topk`; identical for distinct scores, tie-order unspecified in
both — list it among the parity hotspots but it rarely bites.)

---

## 6. bf16-compute TODO (flagged, consistent with the decoder cell)

This pass is **fp32-compute** throughout (matches the parity-validated apply and
`csa_hca_step_one`, whose `detail::` helpers upcast bf16 weights to fp32 on load
and accumulate in fp32). A bf16-compute follow-up would be a flag defaulting to
THIS fp32 path — exactly like `fused_decoder_megakernel.cuh`'s bf16 TODO. It
would: keep the smem weight bundle in bf16 (halving its ~9KB footprint), do the
projection/attention dot products in bf16 with fp32 accumulation, and keep the
GRU/PEER/apply in fp32 (the precision-sensitive parts). Not wired here.

---

## 7. Files in this deliverable

| file | role |
|---|---|
| `csrc/fused/sm_90/opt_stage_supergrok2.cuh` | the persistent SG2 meta-net kernel + `launch_sg2_meta_optimizer_tail` + POD structs + `sg2_ws_stride` |
| `tests/hw/sg2_kernel_mirror.py` | fp64 STRUCTURAL mirror of the kernel + INDEPENDENT clean oracle of `csa_hca_step_one` |
| `tests/hw/test_sg2_megakernel.py` | (A) mirror-vs-oracle parity gate (CPU, runs now) + (B) hw-gated megakernel-vs-per-op gate (skip-bodied, integrator fills) |
| `INTEGRATION-NOTES.md` | this file |

The integrator touches: `csrc/bindings/bindings.cpp` (§1), the dispatch/cell
codegen (§3), and the optimizer driver (§2). The header author touched none of
those.
