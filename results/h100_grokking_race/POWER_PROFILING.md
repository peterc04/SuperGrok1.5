# H100 Power / Profiling Investigation — Grokking Race

**Question (user):** H100 has a 700 W limit; SG2 looked like only 200–300 W.
Is the race register-bound? Should `compile.py` change? Is SG2 a "glorified AdamW"?

## TL;DR
1. **~235 W is a workload ceiling, not an SG2 defect.** *Every* optimizer —
   including plain AdamW (a single 44-reg kernel) — tops out at ~196–235 W mean /
   39–44 % GPU util on this task. The p=97 / ~420K-param / full-batch problem is
   far too small to fill an H100; the SMs sit ~60 % idle every step. 700 W is not
   reachable without changing the *workload* (bigger model/batch) or the
   *execution path* (see #3) — both change the race itself.
2. **Nothing on the race path is register-bound.** The fused *optimizer* kernels
   that actually run are 18–48 regs (AdamW/Lion/GrokAdamW) and SG2's are 27–40
   regs with ~1 KB smem — all high-occupancy. The "168 regs" figure is a **CUTLASS
   Sm90 WGMMA GEMM**, where ~168 regs is *optimal by design* for Hopper
   warp-specialized tensor cores (low occupancy is intentional, not a leak).
3. **`compile.py` / the 168-reg megakernel is NOT on the race path.** Verified:
   `dispatch._FUSED_REGISTRY` is empty → `has_fused()→False` → the race always
   falls back to **eager model + fused optimizer (the L1 path)**. The L3
   model×optimizer megacell is "compile-verified only" and the race "does not
   exercise" it (this is documented in HARDWARE_VALIDATION.md §"sm_90 update").
   So **editing `compile.py` would not change race power** — it tunes a kernel
   the race never launches.
4. **SG2 is not a glorified AdamW — it's a sophisticated optimizer with broken
   parts.** Its machinery is wired and the meta-net trains modestly/stably
   (|Δw|≈0.1, finite), but `sam_step` fails **100%** (grad bug → no SAM signal)
   and `bilevel_step` fails **~50%** (grads cleared by the failed SAM → meta
   trained at half cadence), and the **grokfast `lamb`** term then drives the
   update into a runaway collapse (8.7× anti-aligned by step 900). So "fully
   utilize SG2" is a *correctness/stability* problem (fix SAM, bilevel, `lamb`),
   not a power-tuning one. SG2 is **not** a low-power outlier — it runs at
   225 W / 99% util like the pack, just 18× slower because its step is a long
   chain of light memory-bound ops (its own machinery, §B).

## A. Cross-optimizer sustained power (real race path, decoder, p=97)
Idle baseline ≈ 72 W.

| optimizer | steps/s | mean W | p50 | max | util | J/step |
|-----------|--------:|-------:|----:|----:|-----:|-------:|
| AdamW     | 104.2 | 235 | 222 | 378 | 44% | 2.26 |
| Lion      |  81.3 | 218 | 216 | 261 | 42% | 2.69 |
| Muon      |  58.0 | 196 | 196 | 201 | 39% | 3.38 |
| GrokAdamW |  74.5 | 211 | 211 | 218 | 40% | 2.83 |
| Prodigy   |  79.2 | 218 | 218 | 229 | 40% | 2.75 |
| SuperGrok11 | 45.0 | 183 | 181 | 212 | 31% | 4.06 |
| SuperGrok15 | 54.3 | 196 | 195 | 204 | 36% | 3.61 |
| SuperGrok2  |  5.8 | 225 | 216 | 277 | **99%** | 38.60 |

Read: the *fastest* optimizer draws the *most* power (AdamW packs more identical
work/sec). None breaks ~235 W. **SG2 is NOT a low-power outlier** — at 225 W it
sits with the pack, and its util is actually the *highest* (99%) because it runs
a long unbroken chain of ~20 light kernels per step; it's just 18× slower
(5.8 vs 104 steps/s) and burns 17× the energy/step. The H100 is "busy" but the
work per kernel is too small to draw real power. 700 W is a function of *work
size*, not optimizer choice.

## A2. SG2's SAM + bilevel are BROKEN — meta-net never trains (likely)
During every SG2 run the loop logs (swallowed by try/except):
- `sam_step failed: element 0 of tensors does not require grad and does not have a grad_fn`
- `bilevel_step failed: SuperGrok2.bilevel_step() requires CUDA tensors` (really
  "no params have grads" — `any(p.is_cuda … if p.grad is not None)` over an empty
  set; grads were cleared by the failed sam_step that runs just before it).

`bilevel_step` is the **only** path that trains the meta-net (`mopt =
Adam(meta_net.params)`; the C++ kernel only *reads* cached meta weights, it can't
autograd).

**Direct verification (200–300 real SG2 steps, decoder):**
- `sam_step`:    **0/15 succeed** — raises `element 0 of tensors does not require
  grad and does not have a grad_fn` *every* time. Root cause (supergrok2.py:1751):
  `perturbed_params = p.detach() + rho·grad` are non-leaf, requires_grad=False, so
  `functional_call(...).backward()` has nothing to differentiate. This is a
  **pre-existing bug** (git-blamed to 6941457, 2026-06-02 — *not* introduced this
  session; this session's SG2 commits never touched sam_step), silently swallowed
  by the race loop's try/except, which is why the prior 8/11 race + green parity
  gates never surfaced it (parity tests the kernel, not the Python SAM). SG2 thus
  runs with **no SAM/sharpness signal at all**. Fix = the same one already applied
  to SuperGrok11's sam_step this session (`requires_grad_(True)` + `autograd.grad`,
  leave `p.grad` intact).
- `bilevel_step`: **5/10 succeed** — fails on exactly the steps where it fires
  right after the broken `sam_step`, which left the params grad-less. So the
  meta-net is trained at only ~half its intended cadence.
- meta-net **trainable weights train modestly and stably** — max |Δw| ≈ **0.10**
  over 200 steps (expert_W1/W2, product_keys), all finite. (An earlier 7e7
  "delta" was the `expert_counts` *int buffer*, a MoE activation counter, not a
  weight — corrected.) So the meta-net is **not** frozen and **not** diverging;
  it learns, just starved of the SAM input and half its bilevel updates.

So SG2's machinery is wired and the meta-net trains, but **two components are
broken** (SAM 100% dead, bilevel half-dead) so it never runs at full capability,
and — independently — the **grokfast `lamb` term is the destabilizer** that drives
collapse (Part C: at step 900 the update is 8.7× Adam's and *anti-aligned*,
cos −0.74, with train_acc already collapsed to 0.014; and the prior isolation:
`lamb=0` → stable, test→0.948). This — **not** "glorified AdamW" — is why SG2
DNFs. "Fully utilize what it can do" = (1) fix the SAM grad bug, (2) stop bilevel
double-firing after SAM so it gets its full bilevel signal, (3) tame `lamb` so it
doesn't blow up — i.e. the SG2 fix + tuning (tasks #15/#17), with the Part-C
cosine/magnitude metric as the guardrail that the meta stays active but bounded.

## B. Kernel trace — where the step goes
**AdamW** (top self-CUDA): `aten::mm` 18.9%, `layer_norm_backward` 12.9%,
`addmm` 12.8%, `GammaBetaBackward` 11.2%, `bmm` 10.2%, `gemmSN` 7.3%, `copy_`
7.0%, `xmma_gemm`/`cutlass_80_sgemm` ~12%. → the step is **dominated by the eager
model's GEMMs/layernorm** (sm80/cuBLAS/cutlass kernels, 40–68 µs each), not the
optimizer. These small GEMMs can't saturate Hopper tensor cores, so the SMs idle
between them (44% util). The fused AdamW kernel is a negligible slice.

**SG2** (top self-CUDA): `Optimizer.step#SuperGrok2.step` **65.8%** (167 ms/step!),
`csa_hca::csa_indexer_topk_kernel` **20.7%** (3.5 ms each × ~15/step), then a soup
of tiny eager ops — `index_select` ×306/step, `mul` ×530/step, `topk`,
`scatter_add`, `mm` ×260/step. → SG2's cost is its **own machinery**, not the
model, and most of it is hundreds of light memory-bound ops + one slow indexer
kernel per step. That's the 18× slowdown and the 99%-util/225 W signature:
launch/latency-bound chains of small work, not register- or compute-bound.

## C. SG2 meta-activity (full vs Adam-core alpha=0/lamb=0, matched state)
| step | train_acc | cos(full,adam) | \|full\|/\|adam\| | \|full−adam\|/\|adam\| |
|-----:|----------:|---------------:|------------------:|-----------------------:|
| 50   | 0.127 | **+1.0000** | 1.000 | **0.000** |
| 300  | 0.934 | +0.308 | 2.913 | 2.773 |

**Caveat:** this Part-C probe never calls `bilevel_step`, so the meta-net sits at
zero-init *by construction* — §C isolates the **grokfast `lamb`** contribution, it
does **not** characterize the real-race meta contribution (in the real race
bilevel does train the meta, modestly — Δw≈0.1, §A2). Read that way: at step 50
the update is byte-identical to AdamW (lamb gate off, meta at zero-init); by
step 300 it diverges 2.9× exactly when the `lamb` gate switches on (train_acc>0.8);
by step 900 it is 8.7× and anti-aligned (cos −0.74) as `lamb` runs away → the
**`lamb` term is the destabilizer**, independent of the (modest, real-race) meta.

## D. nsight compute — environment-blocked; substituted
`ncu` is installed but its **hardware counters are blocked**: `ERR_NVGPUCTRPERM`
even as root, because this is a Docker container with `RmProfilingAdminOnly=1`
and no way to reload the driver with `NVreg_RestrictProfilingToAdminUsers=0`.
`nsys` is not installed. (To enable on the host: set that module param + reload
the nvidia driver, then `ncu` gives SM/mem SoL + warp stall reasons.)

Substituted with three host-side sources that answer the same questions:
- **cuobjdump --dump-resource-usage** (static, the real `_ops.so`): every
  on-race-path kernel's regs/smem → occupancy. AdamW/Lion/GrokAdamW 18–48 regs;
  SG2's csa/hca/moe/gru 27–40 regs, ~1 KB smem. **None register-bound.** The only
  168-reg kernels are CUTLASS Sm90 WGMMA GEMMs (optimal-by-design; off the race
  path anyway).
- **torch.profiler** (live kernel time, §B): names the bottleneck kernels and
  their per-step counts — the actual "where do the cycles go" that ncu's timeline
  would show.
- **NVML** (live power/util/clock, §A): the power ceiling + the 99%-util-yet-225W
  signature that says "busy on light work," not "stalled on a hot kernel."

Verdict from the substitutes: SG2's slowness/power is **launch-count + light
memory-bound work**, not register pressure or compute saturation — so a
`maxrregcount`/occupancy change (what ncu would tune) is not the lever.

## Utilization ≠ wattage — what each lever actually does
Key distinction: **higher util does not mean higher power.** SG2 already runs at
99% util / 225 W — busy, but on light work. The only way to approach 700 W is to
add *heavy* work (bigger GEMMs), which means a bigger problem.
- **Wire the L3 megakernel into the race** (one fused model×optimizer launch
  instead of eager-model kernels + ~20 optimizer micro-kernels): cuts launch
  overhead → **faster, higher util**, but **same ~235 W ceiling** (no new work).
  Also changes execution semantics + FLOPs accounting and risks numeric
  divergence from the validated eager+L1 path. Major task.
- **CUDA-graph the optimizer step** (collapse SG2's ~20 launches into one replay):
  **faster**, but again no new work → not more power; and SG2's dynamic control
  flow (expert recycling, alpha schedule) complicates capture.
- **Bigger model/batch** → more work → **the only lever that raises the power
  ceiling** — but that *is* changing the science/FLOPs of the race, so it's the
  user's call, not something to do silently.
