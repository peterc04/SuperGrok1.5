# BUILD_AND_VALIDATE — lever (a): L1 fused megakernel in the grokking race

This document is the operator runbook for the change that wires the L3-substrate
fused model×optimizer megakernels into the grokking race **as the optimizer
step**, with the FULL optimizer scalar set (no placeholder scalars). Run the
sequence below on a real **sm_90 (H100/H200)** device after the concurrent GPU
job releases the device and the live `_ops.so`.

> Source-complete only. The implementing agent did NOT build, did NOT run any GPU
> validation, and did NOT touch `setup.py` / `compile.py` / `build.sh` / `README`
> (a sibling agent owns those). Everything below is for the operator to run.

---

## 0. The one deviation you must know (read first)

The directive's items 4–5 assume the L3 fused megakernel computes the **real**
model forward/backward (so a "megakernel-vs-eager loss-trajectory" test is
meaningful). **It does not.** The L3 model stages are an element-local
**surrogate** over the flat parameter blob:

* `csrc/fused/sm_90/model_stages.cuh`:
  `acts[gi] = GELU(params[gi] + input[gi])`, `grad[gi] = acts[gi]·GELU'(params[gi])`
  — no token embedding, no attention, no matmul, no cross-entropy.
* The race's model `input` is **int token indices** `[B, seq]`; the kernel wants
  `float[numel(params)]`. They are not even shape-compatible.
* On the L3 path the kernel does `(void)grad;` and recomputes its own surrogate
  grad, ignoring any real gradient handed in.

Therefore an L3 cell's loss **cannot** match an eager PyTorch run, and "skip the
race's fwd/bwd and let the megakernel own fwd+bwd+opt" would train on gradients
unrelated to the task. (Three independent confirmations; see the cited file.)

**What was built instead — the faithful unit.** The **L1 optimizer tail**
(`fused_optimizer_stage<Opt>`) reads the **real** gradient from global memory
(computed by the framework's own `loss.backward()`) and applies the canonical
per-element `apply_optimizer<Opt>` update. That is exactly the eager optimizer
step. So lever (a) is wired as: **keep the real PyTorch forward + backward, and
replace `optimizer.step()` with the L1 fused megakernel** (`opt_only=True`). This
is validated by **optimizer-equivalence** + a **grok smoke**, not by an (ill-
posed) L3-vs-eager loss test. The L3 path stays compiled (perf-placement
coverage) and reachable via `opt_only=False`, but is **not** the race path.

---

## 1. What changed (files)

ABI / C++ (the C2-gap fix — carry the FULL scalar set, not just `lr`):
* `csrc/fused/sm_90/opt_components.cuh` — new `FusedScalars` POD (lr, beta1,
  beta2, eps, wd, bc1, bc2, alpha, beta, lamb, alpha_max, gate, d_factor,
  neg_lr_scale, decay_factor) + `apply_scalars()` (folds it into `FusedOptState`,
  which `apply_optimizer<>` already reads — this is what un-freezes bc1/bc2/gate/
  d_factor from their inert `1.0` defaults). HIP twin in
  `csrc/fused/gfx942/opt_components.hip.hpp`.
* `csrc/bindings/dispatch.cpp` — `fused_step` widened to carry the scalar set +
  an `int64 step` + `opt_only` (tier selector); builds `FusedScalars` and passes
  it through `dispatch_{sm90,gfx942}_cell`. Layout-identical `FusedScalars`
  mirrors added (sm90 / gfx942_mega namespaces) so the call-site mangling matches
  the cell `.cu`/`.hip` definitions (same pattern the existing `PersistentContext`
  mirror uses).
* `csrc/bindings/helpers.h` — `fused_step` declaration widened (default values
  live HERE only — the definition in dispatch.cpp omits them, per C++ rules).
* `csrc/bindings/bindings.cpp` — `m.def("fused_step", …)` gains `py::arg`
  defaults so the short 7-arg call still works.

Cells + dispatch tables (regenerated from the single-source codegen — do NOT
hand-edit; see §3):
* `grokking_optimizers/megakernel_codegen.py` — `_emit_cuda` / `_emit_hip` now
  emit the widened host entry that binds the full scalar set and instantiates
  **both** `FuseTier::L1` and `FuseTier::L3`, selecting at runtime on `opt_only`.
  `dispatch_table_sm90` / `dispatch_table_gfx942` + the launcher-signature
  constants updated to match.
* `csrc/fused/sm_90/mega_*.cu` (33), `csrc/fused/gfx942/mega_*.hip` (33),
  `csrc/fused/sm_90/fused_dispatch_table.inc`,
  `csrc/fused/gfx942/fused_dispatch_table.inc` — **regenerated** (the diff is
  purely the ABI widening; tpu cells unchanged).

Python wiring:
* `grokking_optimizers/dispatch.py` — readiness whitelist `_FUSED_READY`
  (`{decoder,vit,mamba} × {adamw,lion}`), `has_fused`/`register_fused`/
  `dispatch_fused` made real for whitelisted cells, `fused_optimizer_step()` (the
  per-tensor L1 driver that owns persistent `[m|v|extra]` state + pulls live
  scalars + computes `bc1=1-beta1^step`, `bc2=1-beta2^step`), and a one-time loud
  run-start banner `announce_fused_readiness()`.
* `grokking_optimizers/megakernel_engine.py` — `dispatch_fused_megakernel` passes
  `opt_only` from the solver tier.
* `grokking_race_v2.py` — `_try_fused_step` is now a **post-backward optimizer
  step** (grads already populated); the **adamw** and **lion** loops call it
  AFTER `loss.backward()` and skip `scaler.step(opt)` on the fused path. **Bug
  fixed:** the AdamW loop passed `"grokadamw"` (which would route plain-AdamW
  state into the grokfast-EMA cell — wrong math); now `"adamw"`.

Validation:
* `tests/hw/test_megakernel_vs_eager.py` — the parity gate (see §5).

The READINESS whitelist shipped: **`decoder:adamw, vit:adamw, mamba:adamw,
decoder:lion, vit:lion, mamba:lion`** (6 cells). L1 is model-agnostic (the tail
never touches the model stages), so all 3 models × {adamw, lion} are equivalent
on L1. The other 9 optimizers are honestly staged behind the readiness gate (a
loud one-time TODO at run start) because their L1 tail needs a per-step
precomputed quantity it does not itself produce — prodigy's adaptive `d`,
grokfast/grokadamw's slow-grad EMA, looksam's SAM direction, muon's NS-orth
direction, SG11/15's reduced gate + `mu`, SG2's meta-net smart-grad. Wiring those
with placeholder scalars would silently degrade the math (the suppression the
owner forbids); `register_fused` is the seam to add them once their precompute is
plumbed and validated.

---

## 2. Rebuild

```bash
# sm_90 (H100/H200). --no-build-isolation reuses the environment's torch.
PYTHONPATH=. pip install -e . --no-build-isolation
# (AMD MI300X twin, if applicable:)
# WITH_HIP=1 PYTHONPATH=. pip install -e . --no-build-isolation
```

The widened `fused_step` ABI is binary-incompatible with a stale `_ops.so`, so a
**clean rebuild of the extension is required** (a partial rebuild that keeps an
old `dispatch.o` will mismatch the cell symbols). If the build caches objects,
force the dispatch + fused TUs to recompile (or `rm -rf build/` first).

Smoke that the binding took the new signature:

```bash
PYTHONPATH=. python -c "import inspect, grokking_optimizers as g; \
from grokking_optimizers.dispatch import get_ops; \
print('fused_step' in dir(get_ops()))"   # -> True
PYTHONPATH=. python -c "from grokking_optimizers.dispatch import announce_fused_readiness as a; a()"
# -> [fused] L1 megakernel path ENABLED for 6 cell(s): decoder:adamw, ...
```

---

## 3. No-build consistency check (cells == codegen)

The 66 cell files + 2 dispatch tables are GENERATED. Confirm they match the
codegen exactly (this is the source-of-truth guard the directive requires — if
this diff is non-empty, the cells drifted from the generator):

```bash
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --write-all
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --dispatch-table-sm90 \
    > csrc/fused/sm_90/fused_dispatch_table.inc
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --dispatch-table-gfx942 \
    > csrc/fused/gfx942/fused_dispatch_table.inc
git diff --stat        # EXPECT: no changes (already regenerated & committed)
```

---

## 4. Parity gate (existing per-op kernels — must still pass)

This is the repo's existing single-step fused-vs-reference gate for the per-op
kernels. The ABI change is additive (the per-op `*_fused_step` bindings are
untouched), so it must remain green:

```bash
PYTHONPATH=. python3 tests/hw/parity_gate_h100.py     # expect 11/0
```

---

## 5. NEW validation: L1 megakernel vs eager reference

The lever-(a) gate. (A) optimizer-equivalence per whitelisted cell (the L1 tail
vs a pure-PyTorch reference transcribed from `csrc/algorithms/<opt>.h`, at
~1e-4 rel over K=200 steps — tight enough to catch a frozen `bc1/bc2`); (B) a
3-seed `(decoder, adamw)` grok smoke (real decoder + real fwd/bwd + the L1 fused
AdamW step).

```bash
# The two example cells from the directive:
PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager --cells decoder:adamw,decoder:lion
# All 6 whitelisted cells:
PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager --all
# Add the grok smoke (slower — trains 3 short runs):
PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager --all --grok-smoke

# Or via pytest (hardware-marked):
PYTHONPATH=. python -m pytest tests/hw/test_megakernel_vs_eager.py -m hw -q
```

Expected: `ALL PASS` (each cell `rel_err < 1e-4`, `traj_dev < 1e-4`; each grok
seed `best test acc ≥ 0.95`). A non-zero `rel_err` almost always means a scalar
did not reach the tail — i.e. the C2 gap regressed (frozen `bc1/bc2`, or wrong
betas/eps/wd binding).

> Without a GPU / built extension the script prints a clear `SKIP` and exits 0
> (honest skip, not a pass).

---

## 6. End-to-end race + roofline re-run

Run the race with the fused path live (default `use_fused=True`); the run-start
banner names the 6 cells taking the L1 megakernel. The AdamW and Lion entries now
take the fused optimizer step; everything else is the untouched eager/per-op path.

```bash
# A short decoder race (adamw + lion now fused):
PYTHONPATH=. python grokking_race_v2.py --tasks decoder --num-seeds 3 \
    --eval-every 50 --output results/fused_l1_check
# Disable to A/B against the eager path:
PYTHONPATH=. python grokking_race_v2.py --tasks decoder --num-seeds 3 --no-fused \
    --output results/eager_baseline

# Roofline re-run (perf placement; compares fused vs eager throughput):
PYTHONPATH=. python -m tuning.roofline --models decoder,vit,mamba --steps 100
```

Grok parity is the bar: the fused-AdamW / fused-Lion runs must reach the grokking
threshold in step-counts consistent with the eager baseline (the L1 tail is the
same math, so the trajectories should match modulo fp32 accumulation).

---

## 7. Known risks / caveats

* **fp32 only.** The L1 tail indexes raw `float` memory; `fused_optimizer_step`
  raises on non-fp32 params and `.contiguous()`-es param/grad. The race runs AMP
  off by default, so this holds. With AMP on, the fused path raises and the run
  degrades to eager (no silent corruption).
* **`fused_step` does NOT run `check_param_grad`** (that guard is on the per-op
  path). The Python driver enforces dtype/contiguity instead; do not call the raw
  `ops.fused_step` with a non-contiguous/half tensor.
* **Single `(bc1, bc2)` per call** is correct only because every race parameter
  steps together (one shared step counter). A caller that steps tensors at
  different counts must call `fused_step` per-tensor with that tensor's bc. This
  is asserted-by-construction in the race and documented loudly in
  `opt_components.cuh` / `dispatch.cpp`.
* **Persistent optimizer state is owned by the driver**, keyed by `id(param)` on
  the optimizer instance (`_fused_state_cache`). It is allocated once per param
  and never reallocated (reallocating would reset momentum and the run would not
  grok). It is NOT checkpointed by `optimizer.state_dict()` — a mid-run
  save/restore would lose the fused momentum. For a from-scratch race this is a
  non-issue; flagged for any resume-from-checkpoint use.
* **L3 remains compiled but unused by the race.** If a future change wants real
  fused fwd+bwd+opt, the model stages must be replaced with the real model graph
  (out of scope here) — `opt_only=False` currently runs the surrogate.
* **gfx942 host launch is 🟡** (no hipcc/MI300X in the implementing environment);
  the AMD cells were regenerated for ABI consistency but only the sm_90 path is
  exercised by this runbook.
* **Per-parameter launch cost (perf, not correctness).** The `fused_step` ABI is
  one parameter tensor per call (`n=numel`, `n_tasks=1`, `state≥3n`), so
  `fused_optimizer_step` launches ONE persistent megakernel per parameter (each a
  `#SMs`-CTA grid where a single CTA does the one-task work). For a model with
  dozens of small params this is many tiny launches — expected to be SLOWER per
  step than a multi-tensor fused optimizer, and the roofline number for the fused
  AdamW/Lion path should be read with that in mind (it is a correctness/wiring
  deliverable at this ABI, not a throughput win). A batched multi-tensor ABI
  (pack all params into one call with real per-tensor `sizes`/`offsets`) is the
  natural follow-up but is out of scope for lever (a).

---

## PHASE 1 — the TRUE L3 fused megakernel for `decoder × adamw` (real fwd+bwd+opt)

> Supersedes §0's "L3 cannot match eager" **for the one cell
> `(transformer_decoder, adamw)` on sm_90 only**. Every OTHER cell's L3 path is
> still the element-local surrogate (§0 holds for them); they remain compiled and
> unused by the race. This section is what changed in PHASE 1 and how to validate
> it. Source-complete only — the implementing agent did NOT build and did NOT run
> any GPU work (tuning owns the GPU).

### What PHASE 1 does

Replaces the SURROGATE model stage with the **REAL transformer-decoder
forward+backward** inside the existing persistent megakernel, so that for the
`decoder × adamw` cell **`(decoder × adamw)` runs as ONE persistent kernel per
training step** — real model math, real optimizer math, **zero intermediate
kernel launches** (the owner rejected CUDA graphs; this is the chosen path).
Weights/state stay resident; the stages are separated only by in-kernel grid
barriers.

The race's `train_adamw` loop now, for the decoder cell, calls
`_try_fused_train_step(...)` FIRST: if the L3-REAL kernel is available it runs the
whole step (fwd+bwd+adamw) and returns the loss, and the loop **skips its own
eager forward/backward/`optimizer.step()`**. Otherwise it falls back to the eager
fwd+bwd + the L1 fused optimizer tail (the pre-PHASE-1 path). `eval_every`
evaluation stays eager/unchanged.

### Exact architecture transcribed (cites grokking_race_v2.py)

`_raw_model` → `Transformer(nl=2, d=128, h=4, ntok=99, seq=4)` (seq hardcoded 4):

| component | def | shape |
|---|---|---|
| `tok` | `nn.Embedding(99,128)` (:360) | [99,128] |
| `pos` | `nn.Embedding(4,128)` (:360) | [4,128] |
| 2× `DecoderBlock` | (:346) | |
| `attn` | `nn.MultiheadAttention(128,4,batch_first)` (:349), **causal** `triu(diag=1)` (:352) | in_proj [384,128]+[384], out_proj [128,128]+[128] |
| `n1,n2` | `nn.LayerNorm(128)` (:350), eps **1e-5** | [128]×2 each |
| `ff` | `Linear(128,512)→GELU→Linear(512,128)` (:351) | [512,128]+[512], [128,512]+[128] |
| post-LN | `x=n1(x+attn(x)); h=n2(x+ff(x))` (:354-355) | |
| `norm` | `nn.LayerNorm(128)` (:362) | [128]×2 |
| `out` | `nn.Linear(128,99)` (:362) | [99,128]+[99] |
| forward | `h=tok+pos; for l: h=l(h); return out(norm(h)[:,-1,:])` (:366-368) | **LAST token** |
| loss | `F.cross_entropy(logits, targets)` (:745) | mean over B |

**30 parameter tensors, 422,755 total params.** Numerics pinned (each verified
against autograd in `tests/hw/decoder_oracle.py`): **GELU = exact erf** (the
surrogate's tanh approx is ~4e-4 off — wrong here); LayerNorm **eps=1e-5**;
attention scale **1/√32**, softmax with row-max subtraction; CE **mean over B** →
`dlogits=(softmax−onehot)/B`. `nn.MultiheadAttention` is transcribed bit-identical
(in_proj packs `[Wq;Wk;Wv]` row-blocks; verified `max|diff|==0`).

### Stage / barrier layout (one persistent kernel)

`csrc/fused/sm_90/fused_decoder_megakernel.cuh`, gridDim = #SMs (one CTA/SM),
256 threads/CTA. **Batch-parallel** (NOT the param-tensor work-steal queue — that
would vary the batch→CTA grouping and fp32 sums aren't associative):

```
P0  each CTA zeroes its OWN grad-partial slice + loss slot
--- grid barrier B0 ---
P1  each CTA owns a FIXED contiguous batch slice (blockIdx.x); processes its
    samples ONE AT A TIME (CTA-cooperative), accumulating each sample's weight-
    grad into the CTA's partial with a SINGLE-OWNER-THREAD-PER-ELEMENT rule
    (no atomics → deterministic), and sums its slice's NLL (fp32)
--- grid barrier B1 ---
P2  deterministic cross-CTA reduce: sum partial[0..nCTA) in ASCENDING CTA index
    into the global grad (no float atomics; order fixed → reuses the work-steal
    queue to pick WHO reduces which tensor). Loss: fp64 ordered sum → loss/B → a
    device float the host reads back.
--- grid barrier B2 (sync_reset: also resets the queue for P3) ---
P3  the REAL apply_optimizer<AdamW> tail consumes the reduced grad in place
```

Determinism: fixed batch→CTA grouping + fixed ascending-CTA reduction order; **no
float atomics** in the weight-grad reduce. The embedding `tok.weight` scatter maps
**thread→embedding column** and loops `(sample,position)` sequentially, so
colliding token ids (vocab 99) accumulate deterministically.

### SMEM budget per CTA

One sample, one layer live (the backward **recomputes** each layer's forward from
the cached layer input rather than caching every intermediate, so only one layer's
activations are live): `DecSampleSmem` ≈ **41.96 KB** — UNDER the 48 KB static-smem
cliff, so NO dynamic-smem opt-in and the occupancy≥1 launch guard
(`dynamicSMemBytes=0`) is unchanged.

### Weight layout (ONE source of truth)

`tests/hw/decoder_oracle.py::decoder_param_layout()` (asserted == the eager
`named_parameters()` order in the parity test) is the single source. The C++
device header `csrc/fused/sm_90/decoder_layout.cuh` is **GENERATED** from the same
table by `megakernel_codegen.py --decoder-layout` and **static-asserts the count
(30) and total (422755)** + an offset/size consistency fold — a mismatch fails the
**build**, never corrupts at dispatch. The Python wrapper
(`dispatch.py::_DECODER_TOTAL_ELEMS`) and dispatch.cpp (`kDecoderTotalElems`)
mirror the literal and cross-check `params.numel()` at the call site.

### ABI (no `bindings.cpp` / `setup.py` edit; the owned-seam extension)

The pybind `m.def("fused_step", ...)` arity is pinned in `bindings.cpp` (NOT
owned), so `fused_step`'s arity is unchanged. The token path is carried through the
EXISTING tensors, and the **behavior** is extended in `dispatch.cpp` (owned):

* `input` = **int32** `[B*(kSeq+1)]` — tokens `[B*kSeq]` then targets `[B]`
  (`B = numel/(kSeq+1)`; S/vocab/d are compile-time).
* `params` = flat `[422755]` fp32; `state` = `[m|v|extra]` (`3*total`) **+ a
  trailing loss slot** the kernel writes the mean CE into (read back in Python).
* the **workspace** (`nCTA*total` grad partials + `nCTA` loss) is device scratch
  allocated in `dispatch.cpp` (keyed by device) — it never crosses the ABI.
* routing: `dispatch.cpp::fused_step` branches on
  `(model=="transformer_decoder" && optimizer=="adamw" && !opt_only)` →
  `mega_decoder_real_adamw` (an `extern` nvcc launcher; **the 33 generated
  surrogate cells are untouched, so no cell/table regeneration is forced and git
  stays generator-consistent**). dispatch.cpp is HOST-compiled, so all
  `<<<>>>`/device code lives in the new nvcc TU
  `csrc/fused/sm_90/mega_decoder_real_adamw.cu` (picked up by setup.py's
  `csrc/fused/sm_90/*.cu` glob), reached via an `extern` decl using the existing
  mirror structs (same FQN/layout → mangling matches).

Python: `dispatch.py::fused_train_step(model, opt, module, optimizer, tokens,
targets, ...)` packs the int32 input, owns the persistent flat-param + state
buffers, calls `ops.fused_step(..., opt_only=False)`, scatters the updated params
back, and returns the loss. The L3-REAL tier marker is `_FUSED_L3_REAL` +
`has_l3_real()` (sm_90-only).

### What is validated by which test (`tests/hw/test_megakernel_vs_eager.py`)

**No-GPU (run anywhere, the rigor substitute for an un-runnable .cu):**
* `test_decoder_oracle_matches_autograd` — the manual fwd+bwd oracle == autograd
  (loss + every grad, ~1e-12 fp64). This is the math the CUDA transcribes.
* `test_decoder_kernel_mirror_matches_oracle` — a single-threaded **structural
  mirror** of the kernel (same buffer aliasing, recompute, head-split index math,
  owner-thread accumulation, the 3-pass attention backward, a token collision) ==
  the oracle (~1e-12). Catches the missing-term/index/alias bug class the
  un-runnable .cu hides. (It does NOT cover `__syncthreads`/grid-barrier races —
  those are verified by reading each barrier against the buffer it guards.)
* `test_decoder_layout_matches_named_parameters` — flat layout == `named_parameters`
  order (30 tensors, 422755), and `_DECODER_TOTAL_ELEMS` agrees.

**GPU-gated (sm_90 + built extension; skip cleanly otherwise):**
* `test_decoder_l3_real_single_step_parity` — (a) kernel loss within **1e-5 rel**
  of eager; **the kernel's reduced weight grad** (routed out through the ABI
  `grad` tensor, returned by `fused_train_step(return_grad=True)`) compared
  **PER-TENSOR against the oracle** within **1e-4 rel** (dumps max-rel per tensor)
  — the keystone check that actually exercises the hand-written backward's grad
  magnitudes (loss is fwd-only; params-after-step is sign-dominated at step 1);
  params after 1 step within **1e-5 rel**.
* `test_decoder_l3_real_trajectory` — (b) 200-step loss curve tracks eager AdamW
  (**1e-3 rel**); final params within **1e-3 rel** (fp32 accumulation drift).
* `test_decoder_l3_real_groks` — (c) 3-seed grok smoke through `train_adamw` (now
  routing the decoder cell through the L3-REAL megakernel).

### Run (operator, on a real sm_90 device after the GPU job releases)

```bash
# clean rebuild (the dispatch.cpp behavior changed + the new .cu must compile):
PYTHONPATH=. pip install -e . --no-build-isolation   # or rm -rf build/ first

# no-GPU correctness gates (also run in CI):
PYTHONPATH=. python -m pytest tests/hw/test_megakernel_vs_eager.py \
    -k "oracle or mirror or layout" -q

# generator-consistency of the layout header (EXPECT: no diff):
PYTHONPATH=. python -m grokking_optimizers.megakernel_codegen --decoder-layout \
    > csrc/fused/sm_90/decoder_layout.cuh && git diff --stat

# GPU parity + grok (sm_90):
PYTHONPATH=. python -m pytest tests/hw/test_megakernel_vs_eager.py -m hw -q

# end-to-end: the decoder race now takes the L3-REAL fused train step:
PYTHONPATH=. python grokking_race_v2.py --tasks decoder --num-seeds 3 \
    --eval-every 50 --output results/l3_real_decoder
```

### Known scope / caveats (PHASE 1)

* **fp32 compute is the correctness baseline.** bf16 compute is a deliberate
  follow-up: it would be a flag defaulting to THIS fp32 path (the L3-REAL path
  raises/falls back to eager under AMP today — a loud None, never a silent stub).
  The race's bf16-autocast default applies to the EAGER fallback path, not the
  fused kernel.
* **decoder × adamw on sm_90 only.** Other optimizers / models / gfx942 keep the
  eager (+ L1) path; only this one cell has the real fwd+bwd+opt kernel.
* **The surrogate L3 cells are now unreachable dead code on the race path** (the
  decoder real path is routed before them in `fused_step`; the other cells'
  `opt_only=False` still runs the surrogate but the race never sets it). Documented,
  not hidden.
* **Persistent flat-param + optimizer state are driver-owned** (keyed by model on
  the optimizer instance), allocated once and never reallocated (reallocating
  resets momentum → never groks). NOT checkpointed by `optimizer.state_dict()`.
* **The model must be eager** (not `torch.compile`-wrapped) on the fused path — the
  flat layout assumes the eager `named_parameters()` order. The race builds the
  decoder eager by default (`compile_model=False`).
* **Per-step cost:** the workspace is `nCTA × 422755` fp32 (~223 MB at 132 SMs),
  allocated once. This is a correctness/wiring deliverable; a warp-per-sample
  concurrent variant + bf16 tensor-core matmuls are the perf follow-ups.
