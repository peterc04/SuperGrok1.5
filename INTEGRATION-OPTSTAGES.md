# INTEGRATION-OPTSTAGES.md — wiring the in-kernel optimizer precompute stages

**Scope.** This spec tells the binding/cell/codegen layer how to drive the
per-step PRECOMPUTE stages in `csrc/fused/sm_90/opt_stages_precompute.cuh` so
that every optimizer TAIL (`csrc/fused/sm_90/opt_components.cuh::apply_optimizer
<OptId>`) can run fully inside one persistent megakernel. It is a wiring
contract, not code: it does not touch `opt_components.cuh`,
`megakernel_codegen.py`, `dispatch.cpp`, `bindings.cpp`, or any
`supergrok2*`/`neuralgrok.py`/`grokadamw.py`/model-stage file (sibling-owned).

Today only AdamW / Lion have a complete in-kernel tail (no precompute). The 9
non-trivial optimizers are classified below; the 4 STAGED ones get an upstream
stage that fills the `FusedOptState` fields their apply reads.

---

## 0. Verdict table (per-optimizer)

| optimizer    | verdict        | what the apply reads | who produces it |
|--------------|----------------|----------------------|-----------------|
| grokfast     | NOTHING-NEEDED | (EMA fused in apply) | — `grokfast.h:63-65` |
| grokadamw    | NOTHING-NEEDED | (EMA fused in apply) | — `grokadamw.h:48-49` |
| neuralgrok   | NOTHING-NEEDED | (psi MLP inline)     | — `opt_components.cuh:222-229` → `neuralgrok.h:31-46` |
| looksam      | MODEL-COUPLED  | `st.sam_dir`         | **model stage** (2nd backward), `looksam.h:52-59` |
| prodigy      | STAGED         | `st.d_factor`        | `prodigy_precompute_*` |
| muon         | STAGED         | `st.orth`, `st.neg_lr_scale`, `st.decay_factor` | `muon_*` stage chain |
| supergrok11  | STAGED         | `st.mu`, `st.gate`   | `sg11_precompute_mu_and_gate_for_tensor` (per-tensor) |
| supergrok15  | STAGED         | `st.mu` (`st.gate` = host scalar) | `sg15_precompute_mu_for_tensor` (per-tensor) |
| supergrok2   | SKIP           | (sibling-owned)      | — |

For the 3 NOTHING-NEEDED optimizers the codegen maps OptId → "no precompute
stage": their apply already runs a complete tail. `optimizer_needs_precompute
<Opt>::value` is `false` for them and a `static_assert` in
`optimizer_precompute_stage<Opt>` rejects an accidental instantiation.

---

## 1. The `PrecomputeWorkspace` POD (host allocation)

The cross-CTA stages reduce through global scratch grouped in
`PrecomputeWorkspace` (mirrors `PersistentContext`: one POD so the codegen kernel
signature stays uniform). The host allocates ONLY the pointers the chosen
optimizer uses and zero-fills the rest; a stage dereferences only its own
optimizer's buffers. All FP32.

| field | size (floats) | used by | meaning |
|-------|---------------|---------|---------|
| `prodigy_partials` | `2 * n_ctas` | prodigy | per-CTA (r,s) slots: `[0..n_ctas)`=r, `[n_ctas..2n_ctas)`=s |
| `prodigy_d` | `1` | prodigy | reduced d; bind `st.d_factor = prodigy_d[0]` |
| `muon_buf` | `rows*cols` | muon | momentum buffer (persists across steps) |
| `muon_X` | `rows*cols` | muon | current NS iterate |
| `muon_A` | `rows*rows` | muon | `A = X Xᵀ` |
| `muon_AX` | `rows*cols` | muon | `A X` |
| `muon_AAX` | `rows*cols` | muon | `A (A X)` |
| `muon_orth` | `rows*cols` | muon | NS output; bind `st.orth = muon_orth` |
| `muon_nrm_partials` | `n_ctas` | muon | per-CTA `‖buf‖_F²` slots |
| `muon_inv_norm` | `1` | muon | `1/(‖buf‖_F + 1e-7)` |

`n_ctas == #SMs` (one persistent CTA per SM, `fused_megakernel.cuh:357`). The
muon matrix buffers are sized for the **largest** 2D weight processed (the stage
runs one matrix at a time; reuse the buffers across matrices).

`prodigy_partials`, `muon_nrm_partials`, `prodigy_d`, `muon_inv_norm` must be
**zero-initialized before launch** (like the barrier counters,
`fused_megakernel.cuh:363-374`) — the owner-computes reducer reads every slot.

---

## 2. STAGED — prodigy (cross-ALL-tensors d, owner-computes tree)

**Shape:** GLOBAL. One scalar `d` for the whole model; `r,s` sum over every
element of every parameter tensor.

**Determinism deviation (contract-mandated, documented — not a transcription):**
the live per-op path reduces `r,s` with `atomicAdd` (`prodigy_sm90.cuh:105-108`,
`:135-138`). The COMPONENT_CONTRACT requires deterministic / no-float-atomic
reductions, so the stage REPLACES that with an owner-computes tree: each CTA
publishes its own `(r,s)` slot (no contention, no atomic) → grid barrier → one
owner thread sums the slots in ascending index order. Math-equivalent in exact
arithmetic; fp32 summation order is now fixed/reproducible.

**Host/cell vs in-kernel split (faithful to the live multi-tensor driver):** the
PERSISTENT beta3-EMA decay of `(r,s)` across steps and the `d_coef` scaling live
in the host launcher (`prodigy_sm90.cuh:488-521`, `bindings.cpp:1246-1264`),
OUTSIDE `prodigy.h`. They are NOT in-kernel. The stage computes the INSTANTANEOUS
per-step `Σ r / Σ s` + `prodigy_update_d` (the cross-tensor reduction Audit-C
flagged). A cell that wants the grokking-race EMA estimator must, BETWEEN
phaseA-publish and phaseB-reduce: decay the prior `prodigy_partials` by `beta3`
and scale `r` by `d_coef` (host-side, on the workspace buffers) — exactly as the
live driver does around the same `prodigy_update_d`. The device math
(`prodigy.h`) stays byte-unchanged either way.

**Codegen barrier sequence (per L3/L1 step):**

```
// param_init = the Prodigy trajectory anchor (flat blob parallel to params).
optimizer_precompute_stage<Prodigy>(ctx, params, grad, sizes, offsets, st, ws,
                                    /*param_init=*/param_init);   // phaseA
bar.sync_reset(ctx.g_next_task);            // grid barrier; reset queue for reuse
prodigy_precompute_reduce_phaseB(ctx, st.d_factor, ws);          // owner-sum → d
bar.sync();                                  // d visible to all before apply
st.d_factor = ws.prodigy_d[0];               // bind the reduced d for the tail
// re-drain the queue for the apply (fused_optimizer_stage / the L3 bwd+opt):
apply_optimizer<Prodigy> over all tensors (opt_components.cuh:211-215).
```

`st.d_factor` enters phaseA as `d_prev` (the comparison floor) and is OVERWRITTEN
post-reduce with the new `d`. Two grid barriers (reduce-publish, d-visible) plus
the queue resets — consistent with `fused_megakernel.cuh`'s phase pipeline.

---

## 3. STAGED — muon (Newton-Schulz orthogonalization, grid-cooperative)

**Shape:** per-MATRIX, GRID-COOPERATIVE. A matmul is cross-element and a real 2D
weight does not fit one CTA's SMEM, so all CTAs cooperate on ONE matrix at a
time. This does NOT fit the flat one-CTA-per-tensor `(params, sizes, offsets)`
signature; the codegen drives the explicit phase fns (the
`optimizer_precompute_stage<Muon>` branch is a documented no-op that points
here).

**New device code, cited:** the live path delegates `X Xᵀ`/`A X`/`A (A X)` to
`torch::mm`/cuBLAS (`bindings.cpp:992-994`), NOT a per-element canonical fn — so
the stage's tiled owner-computes matmul (`muon_matmul`) is legitimate new device
code, cited to that mm sequence. The drift guard requires only that the
ELEMENT-WISE pieces call `muon.h`: the momentum body and the NS polynomial
combine do (`muon_momentum_normalize_step` `muon.h:33-46`, `muon_ns_combine_step`
`muon.h:49-60`). The matmul contracts the k-dimension in ascending order (one
thread owns one output element) — deterministic fp32, no atomic.

**1D params are NOT handled here** — they take the AdamW tail upstream
(`muon.h:75-76`, `bindings.cpp:970-977`). The codegen routes only 2D weights to
this stage.

**Codegen sequence (per 2D matrix, `numel = rows*cols`, `ns_steps` default 5):**

```
muon_momentum_norm_phaseA(grad_mtx, numel, momentum, ws);  // buf=μ·buf+g; ‖buf‖² slots
bar.sync();
muon_norm_reduce_phaseB(ctx, ws);                          // owner-sum → inv_norm
bar.sync();
muon_scale_X(numel, ws);                                   // X = buf*inv_norm (bindings.cpp:964)
bar.sync();
for (s = 0; s < ns_steps; ++s) {
    muon_matmul(ws.muon_X,  ws.muon_X, ws.muon_A,  rows, rows, cols, cols, cols, /*bT=*/true);  // A=XXᵀ
    bar.sync();
    muon_matmul(ws.muon_A,  ws.muon_X, ws.muon_AX, rows, cols, rows, rows, cols, /*bT=*/false); // AX
    bar.sync();
    muon_matmul(ws.muon_A,  ws.muon_AX,ws.muon_AAX,rows, cols, rows, rows, cols, /*bT=*/false); // A·AX
    bar.sync();
    muon_ns_combine_phase(numel, ws);                      // orth = a·X + b·AX + c·AAX
    bar.sync();
    std::swap(ws.muon_X, ws.muon_orth);  // next iter reads X; last iter result is in X
}
// After the loop the final orthogonalized matrix is in ws.muon_X (an odd # of
// swaps) — bind st.orth to whichever buffer holds the last combine output. With
// ns_steps swaps, the final result is in ws.muon_X iff ns_steps is even, else
// ws.muon_orth; the codegen tracks the parity (or copies into muon_orth once).
scale = 0.2f * sqrtf((float)max(rows, cols));   // bindings.cpp:981
st.orth         = <final NS buffer>;
st.neg_lr_scale = -lr * scale;                  // bindings.cpp:982
st.decay_factor = 1.0f - lr * wd;               // bindings.cpp:983
apply_optimizer<Muon> over this matrix (opt_components.cuh:230-234 → muon.h:63-73).
```

`+1e-7` Frobenius-norm epsilon is in `muon_norm_reduce_phaseB`
(`opt_stages_precompute.cuh::kMuonNormEps`, `bindings.cpp:962`, `ref_muon_step`).
NS coefficients `(3.4445, -4.7750, 2.0315)` are `kMuonNS_{A,B,C}`
(`bindings.cpp:938-940`).

> Implementation note for the codegen author: per-matrix grid-cooperative
> orthogonalization inside ONE persistent megakernel is the heaviest stage; if a
> cell is register/SMEM-bound it may keep muon's NS as a SEPARATE launch (the L1
> tail still runs `apply_optimizer<Muon>` in-kernel reading `st.orth` produced by
> that launch) — exactly the split `opt_components.cuh:19-26` documents. The
> stage here makes the fully-in-kernel path POSSIBLE; choosing it per cell is a
> codegen/tier decision, not forced.

---

## 4. STAGED — supergrok11 (meta-MLP mu + per-TENSOR cosine gate)

**Shape: PER-TENSOR, SINGLE-DRAIN, NO grid barrier** — and the "no barrier" is
correct for a SUBTLE reason, NOT by analogy to anything. The live per-op path
(`bindings.cpp:1407-1416`) loops per parameter doing metanet→gate→adam, with each
tensor SELF-CONTAINED: `gate(T)` and `apply(T)` depend ONLY on `mu(T)`, never on
any other tensor's mu (`bindings.cpp:1411` computes the gate INSIDE the per-param
loop; `compute_cosine_gate_fused` takes ONE tensor's smart_grad/mu). So mu/gate/
apply for one tensor fuse into ONE body owned end-to-end by the SAME CTA inside
the codegen's existing single task-queue drain (`fused_optimizer_stage`,
`fused_megakernel.cuh:101-142`).

> **Why NOT a separate full-queue mu pre-pass (the rejected Design A):** a mu
> stage that drains the queue to `n_tasks` and a SEPARATE apply drain would (a)
> need a grid barrier for cross-CTA mu visibility AND (b) **silently no-op** — the
> apply drain's `next_block()` returns `t >= n_tasks` immediately (the counter was
> consumed) and never applies, unless a `sync_reset` re-zeros it. The single-drain
> per-tensor form has NO such hazard. (Prodigy DOES use a barrier because its `d`
> is a genuine cross-ALL-tensors reduction — SG11's gate is per-tensor, so the
> cases are NOT analogous.)

- **mu(T) — per ELEMENT**: `mu[i] = rescale · phi(grad[i], sharpness[i])` over the
  tensor slice (the meta-net is a per-TENSOR weight set).
- **gate(T) — per TENSOR**: block-level cosine tree over the SAME tensor's
  elements via the deterministic `block_reduce_sum_f32` (`primitives.cuh:125`) —
  no cross-CTA, no atomic.

`sg11_precompute_mu_and_gate_for_tensor<32>` does BOTH (mu→`__syncthreads`→gate)
in one per-tensor call and RETURNS the gate; `sg_stage_phi_weights<32>` stages the
phi weights to SMEM once per block before the loop.

**Gate formula — mirror the CODE not the stale comment.**
`compute_cosine_gate_fused` (`supergrok11_sm90.cuh:280-285`) computes
`clamp(<sg,mu> / sqrt(‖sg‖²·‖mu‖² + 1e-12), 0, 1)` with `sg == grad` (because
`bindings.cpp:1408` calls `mu_metanet` with `alpha=0`, so `smart_grad == grad`).
It **IGNORES** `gate_temp` despite the `bindings.cpp:1276` comment claiming
`sigmoid(t·cos)` — the function does a bare clamp, and `ref_sg11_step` agrees.

**Codegen sequence (ONE task-queue drain; mu+gate+apply fused per tensor):**

```
__shared__ float sW1[32*2], sb1[32], sW2[32];
sg_stage_phi_weights<32>(W1, b1, W2, sW1, sb1, sW2);   // once per block (syncs)
TaskQueue q = ctx.queue();
for (int t = q.next_block(&slot); t < ctx.n_tasks; t = q.next_block(&slot)) {
    int   n   = sizes[t];
    int64 off = offsets[t];
    // mu(T)→global, sync, gate(T) — SAME CTA owns T; reads the mu it just wrote.
    float gate = sg11_precompute_mu_and_gate_for_tensor<32>(
        st.mu, grad, sharpness, sW1, sb1, sW2, b2, rescale, off, n);
    FusedOptState ts = st;  ts.gate = gate;            // bind the per-tensor gate
    // apply over [off, off+n): the tail reads ts.mu + ts.gate.
    apply_optimizer<SuperGrok11> elementwise over the slice (opt_components.cuh:235-239).
}
```
No grid barrier, no `sync_reset` — one drain, race-free.

> The apply (`sg11_sweep_b_step`, `supergrok11.h:79-111`) uses `smart_grad = g +
> (1-gate)*alpha*mu`. The live driver folds `ramp*lamb` into a `lamb_eff`
> multiplier (`bindings.cpp:1400-1416`); if a cell needs that, it sets
> `st.alpha = ramp*lamb*alpha` (the apply's `alpha` slot) so the single live
> correction matches — a scalar binding, no math change.

### ⚠ ABI GAP — `sharpness` is not in `FusedOptState` (flag loudly)

`sharpness` is **model-coupled**, exactly like looksam's `sam_dir`: it is
`(sam_grad − normal_grad)²` from a SAM SECOND backward
(`sg11_sharpness_restore_kernel`, `supergrok11_sm90.cuh:236-248`).
`FusedOptState` (`opt_components.cuh:84-113`, which the stage header MUST NOT
edit) has a `mu` field but **NO `sharpness` field and NO phi-weight fields**.
Therefore the stage takes `sharpness` + `(W1,b1,W2,b2,rescale)` as **explicit
pointers/scalars**, and the cell must supply a sharpness buffer the current ABI
does not carry. Two integration options (a sibling agent owning `opt_components
.cuh` chooses):

1. **Extend the ABI** (preferred long-term): add `const float* sharpness` and the
   phi-weight pointers to `FusedOptState`, populated by the cell like the other
   model-coupled inputs. Requires editing `opt_components.cuh` (out of THIS
   phase's scope).
2. **Side-channel pointers** (no ABI change): the cell passes `sharpness` + phi
   weights as extra kernel arguments straight into
   `sg11_precompute_mu_and_gate_for_tensor<32>` / `sg15_precompute_mu_for_tensor
   <32>` (their signatures take them explicitly). The codegen threads them
   alongside `FusedOptState`.

Until one is wired, an SG11/SG15 cell CANNOT run fully in-kernel — the
sharpness pipeline (SAM perturb → 2nd backward → restore) is a model-stage
responsibility documented here, not faked in the optimizer stage.

---

## 5. STAGED — supergrok15 (meta-MLP mu; gate is a host scalar)

Same PER-TENSOR single-drain shape as SG11 — but SIMPLER: NO gate stage at all
(the phi-net shares the SG11/SG15 form, `supergrok15.h:34-52`). DIFFERENCES:

- **gate is GLOBAL = sigmoid(training accuracy)**, set HOST-side and passed as a
  scalar (`supergrok15.h:85`), already carried by `FusedScalars.gate`
  (`opt_components.cuh:153`). **No gate reduction stage, no per-tensor gate.**
- the per-coord alpha clip (`sg15_alpha_per_coord`, `supergrok15.h:68-75`) and
  `smart_grad = g + gate*a*mu` happen INSIDE the apply tail
  (`apply_optimizer<SuperGrok15>` → `sg15_sweep_b_step`,
  `opt_components.cuh:240-244`). So the ONLY precompute is `mu`.
- `sg15_sweep_a` also accumulates a `sharpness²` partial
  (`supergrok15_sm90.cuh:99-102`) that updates the NEXT step's `sharpness` input
  host-side; it is NOT consumed by THIS step's apply (which reads `mu` + the host
  gate). It is part of the model-coupled sharpness pipeline, documented here,
  not computed in this step's optimizer precompute.

**Codegen sequence (ONE drain; mu+apply fused per tensor, no gate, no barrier):**

```
__shared__ float sW1[32*2], sb1[32], sW2[32];
sg_stage_phi_weights<32>(W1, b1, W2, sW1, sb1, sW2);   // once per block (syncs)
st.gate = sigmoid(accuracy);     // host scalar (FusedScalars.gate) — NO stage
TaskQueue q = ctx.queue();
for (int t = q.next_block(&slot); t < ctx.n_tasks; t = q.next_block(&slot)) {
    int n = sizes[t];  int64 off = offsets[t];
    sg15_precompute_mu_for_tensor<32>(st.mu, grad, sharpness,
                                      sW1, sb1, sW2, b2, rescale, off, n);
    __syncthreads();   // mu(T) visible before the apply reads it (block-scope)
    apply_optimizer<SuperGrok15> over [off, off+n) (opt_components.cuh:240-244).
}
```
One drain, no grid barrier, no `sync_reset`.
```

Same `sharpness` ABI gap as §4 (the FusedOptState lacks a sharpness field).

---

## 6. MODEL-COUPLED — looksam (`st.sam_dir` from the model stage)

`looksam_apply_step` (`looksam.h:63-91`) reads `st.sam_dir`. On a SAM step
`sam_dir = g_sam − g` (`looksam.h:52-59`) where `g_sam` is the gradient at the
PERTURBED weights `p + rho·g/‖g‖` (`looksam.h:27-38`) — the output of a SECOND
model backward. A pure optimizer stage cannot produce it. There is therefore NO
looksam specialization of `optimizer_precompute_stage` by design.

**Contract:** the MODEL STAGE fills `st.sam_dir` on SAM steps (every k steps),
mirroring the per-op LookSAM pipeline (perturb → 2nd backward → `set_direction`).
On intervening steps the cached `sam_dir` is reused verbatim (`looksam.h:1-19`).
`FusedOptState.sam_dir` (`opt_components.cuh:91`) already exists as the bound
input; the cell points it at the model-produced buffer. The wiring of the SAM
second backward into the megakernel's model phase is the model-stage owner's
responsibility (out of THIS phase's scope), tracked here as the explicit
contract so it is not silently dropped.

---

## 7. Determinism summary (no float atomics anywhere)

| stage | reduction | determinism mechanism |
|-------|-----------|------------------------|
| prodigy d | cross-all-tensors `(r,s)` | per-CTA slot publish → barrier → ascending-index owner-sum (`owner_sum_slots`) |
| muon ‖buf‖_F | cross-CTA over one matrix | per-CTA slot publish → barrier → ascending owner-sum |
| muon matmul | k-dim contraction | one thread per output, ascending-k inner product |
| sg11 cosine gate | per-tensor `(num, den_g, den_m)` | in-CTA `block_reduce_sum_f32` tree (fixed thread count) |

No `atomicAdd` on floats anywhere in `opt_stages_precompute.cuh`. The only
atomics are the substrate's integer task-counter and barrier-generation
(`megakernel_common.cuh`), which are order-independent.

---

## 8. Validation status (this phase)

- **CPU fp64 mirrors:** `tests/hw/test_opt_stages.py` — prodigy d-reduction, SG11
  cosine gate, SG11/SG15 mu, and the muon 5-iteration Newton-Schulz chain, each
  vs the canonical fp64 oracle (reusing `ref_*` from
  `tests/hw/test_reference_parity.py` where correct). **9/9 pass.** fp32-vs-fp64
  tolerances reflect NS accumulation (muon rel-err `< 2e-3`), not `1e-9`.
- **Math single-source drift guard:** `scripts/check_math_single_source.py` →
  OK (the stage CALLS the canonical `csrc/algorithms/*.h` fns, never re-inlines).
- **CUDA compile / GPU run:** DEFERRED per the NO-builds / NO-GPU-runs directive.
  The header is self-contained per `csrc/fused/COMPONENT_CONTRACT.md` (substrate
  + canonical algorithm headers + CUDA only); compiling it and wiring the cells
  is the integration step this spec describes.

> ⚠ NOTE for the prodigy oracle: `tests/hw/test_reference_parity.py::
> ref_prodigy_partials` is STALE vs canonical `prodigy.h` (it uses `d_prev^1` and
> SIGNED `g`; the header uses `d_prev^2` and `fabsf(g)` — the L1 norm). The mirror
> follows the header (the single source of truth) via a local
> `ref_prodigy_partials_canonical`; `ref_prodigy_update_d` is correct and reused.
> Fixing the sibling oracle is out of THIS phase's scope (sibling-owned file).
