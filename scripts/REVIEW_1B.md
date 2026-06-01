# REVIEW 1B — SuperGrok2 MoE-compaction kernels (correctness review)

Reviewer: Opus 4.8, max effort. Branch `claude/custom-optimizer-analysis-HFYhg`.
Scope: the 9 MoE kernels in both backends:
- sm_90 CUDA: `grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh`
- gfx942 ATen: `grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp`

Ground-truth caller: `grokking_optimizers/optimizers/supergrok2.py::MoEAwareSuperGrok2._moe_step`
(lines 2207-2308) and class config (lines 2159-2175).

## Caller-contract note (important context, not a bug)
`_moe_step` currently `raise NotImplementedError` on its first line (2221-2225); the
kernel-calling body below it (2226-2308) is dead at runtime. The review still treats
that body as the binding contract (it documents exactly how each kernel is invoked and
what shapes/dtypes are passed), since the kernels are real and exported. All findings
below are about whether the kernels would be correct *if* that path were enabled.

## Verdict summary
| # | Kernel | sm_90 | gfx942 | Cross-backend math |
|---|--------|-------|--------|--------------------|
| 1 | moe_count_expert_activations | CORRECT | CORRECT | EQUIVALENT |
| 2 | moe_compute_load_balance_loss | CORRECT | CORRECT | EQUIVALENT |
| 3 | moe_apply_frequency_scaling | CORRECT | CORRECT | EQUIVALENT |
| 4 | moe_filter_active_params | CORRECT | CORRECT | EQUIVALENT (orderings differ, contract-OK) |
| 5 | moe_scatter_results | CORRECT | CORRECT | EQUIVALENT |
| 6 | moe_dynamic_expert_load | CORRECT | CORRECT | EQUIVALENT |
| 7 | moe_dynamic_expert_fwd | CORRECT | CORRECT | EQUIVALENT |
| 8 | moe_dynamic_expert_bwd | CORRECT | CORRECT | EQUIVALENT |
| 9 | moe_scan_compacted (vestigial) | CORRECT | CORRECT | EQUIVALENT |

No bugs found. No code changed. All gates hold (see bottom).

---

## (1) count_expert_activations — CORRECT
sm_90 lines 1847-1874; gfx942 1055-1062.
- Histogram over [N, num_experts]: `gate_logits[row,e] > threshold` → `expert_counts[e]`.
- sm_90: flattened grid-stride loop over `N*num_experts` (`long total`), `e = idx % num_experts`,
  `atomicAdd(&expert_counts[e],1)`. Bounds correct (`e ∈ [0,num_experts)` by modulo). No OOB on E.
- **Pre-zeroing**: the Python caller zeroes `self._expert_counts.zero_()` (line 2248) BEFORE the
  call, and sm_90 accumulates onto it — correct. gfx942 computes the full `(gl>threshold).sum(0)`
  and `copy_` overwrites — also correct, and independent of pre-zeroing. Both agree.
- Strict `>` threshold matches the doc and the gfx942 `(gl > threshold)`. Consistent.

## (2) compute_load_balance_loss — CORRECT
sm_90 1882-1895; gfx942 1066-1076. Switch-Transformer aux loss verified EXACTLY:
- `P_e = softmax(gate_logits, dim=1).mean(dim=0)` — softmax over experts (axis 1), mean over
  tokens (axis 0). Correct axes. [E].
- `f_e = expert_counts / N` — uses the token *fraction* (not raw counts). Correct (divides by N,
  not by num_experts).
- `loss = num_experts * Σ_e f_e·P_e` — the E multiplier is present. Correct.
- Both backends use identical ATen expression; sm_90 does NOT hand-roll this (avoids the common
  softmax-axis / counts-vs-fraction bugs). EQUIVALENT.

## (3) apply_frequency_scaling — CORRECT
sm_90 1900-1930; gfx942 1080-1091.
- `freq_e = (counts[e]+smoothing)/(total+smoothing·E)`; `scale_e = clamp((1/E)/freq_e, min, max)`.
- **Inverse-frequency direction is correct**: rare expert ⇒ small `freq_e` ⇒ large `(1/E)/freq_e`
  ⇒ HIGHER lr scale (clamped to max). Frequent expert ⇒ lower scale. Matches spec.
- Clamp order `clamp(x, min, max)` correct in both. sm_90 `fminf(fmaxf(scale,min),max)`.
- sm_90 has a `freq>0 ? : max_scale` guard; with default `smoothing=0.9` freq is always >0, so the
  guard is dead but harmless and matches gfx942 (which needs no guard since denom>0 always).
- sm_90 uses float accumulation, gfx942 double — numerically negligible, same math. EQUIVALENT.
- Caller passes `total_act = int(expert_counts.sum())` (line 2261); consistent with the histogram.

## (4) filter_active_params — CORRECT
sm_90 1943-1987; gfx942 1098-1122.
- Keep param i iff `expert_active[param_to_expert[i]] != 0`; emit to compact_{params,grads,m,v}
  at the SAME out-position and set `scatter_indices[out]=i`; `compact_count[0]=#kept`.
- sm_90: out-position via single `atomicAdd(&compact_count[0],1)` — each kept i gets a unique slot,
  and ALL FIVE writes (params/grads/m/v + scatter_indices) use that one `out`, so the 4 compact
  arrays + the index map stay in lock-step at the same position. No race on compact_count (atomic).
  `compact_count` is freshly zeroed by the caller (line 2281) and the kernel accumulates from 0.
- gfx942: boolean-mask `nonzero` gather → deterministic ascending order; `compact_count.fill_(K)`;
  all five arrays gathered with the SAME `idx`. Self-consistent.
- Ordering among kept elements DIFFERS between backends (atomic = nondeterministic vs ascending),
  but the only contract is the (out→i) scatter map being self-consistent — satisfied by both.
  scatter_results writes back by stored index, so ordering is irrelevant. CORRECT.

## (5) scatter_results — CORRECT, exact inverse of (4)
sm_90 1992-2028; gfx942 1126-1138.
- `params[scatter_indices[j]] = compact_params[j]` for params/m/v. (grads are not scattered — only
  params/m/v are updated by the Adam tail; matches the caller, which passes only those three.)
- sm_90: one thread per j, reads `i=scatter_indices[j]`, writes the three arrays. Indices were set
  by (4) at the same position, so they align. No write collisions: each kept i appears once.
- gfx942: `index_copy_(0, idx, compact[...])` with `idx = scatter_indices[:N]`. Exact inverse.
- Caller slices `compact_*[:N_active]` and `scatter_indices[:N_active]` (lines 2296-2305) so the
  `compact_N` passed matches the populated prefix. CORRECT. EQUIVALENT.

## (6) dynamic_expert_load — CORRECT (not on reachable path)
sm_90 2037-2055; gfx942 1143-1156.
- Masked gather: pack `expert_{w1,b1,w2,b2}[e]` for active e in ascending expert order into smem_*.
- Both use `nonzero(active_mask!=0)` → `index_select`/`narrow` copy. Identical packing semantics.
  Empty-active early-out in both. EQUIVALENT.

## (7) dynamic_expert_fwd — CORRECT (not on reachable path)
sm_90 2070-2140; gfx942 1162-1183.
- `output[t] = rw[t]·(W2_e · relu(W1_e·x[t]+b1_e) + b2_e)`.
- Shapes: input [N,d_in], w1 [E,hidden,d_in], w2 [E,d_out,hidden]; `hidden=w1.size(1)`,
  `d_out=w2.size(1)`. Indexing `expert_indices[t]` selects the e-th slice; pointer arithmetic
  `W1 + e·hidden·d_in`, `W2 + e·d_out·hidden`, `b1 + e·hidden`, `b2 + e·d_out` is dimensionally
  correct. Matmul dims: h=W1·x ([hidden,d_in]·[d_in]), y=W2·h ([d_out,hidden]·[hidden]). Correct.
- sm_90: one block/token, shared `h[hidden]`, grid-stride over hidden then d_out; `__syncthreads`
  between. Block `min(256,max(d_out,hidden))` with hidden-stride loops covers hidden>block. Correct.
- gfx942: per-token `index_select` weights + batched `bmm`; relu then second bmm + bias, ×rw.
  Dimensionally identical. EQUIVALENT.

## (8) dynamic_expert_bwd — CORRECT (not on reachable path)
sm_90 2150-2260; gfx942 1189-1228.
- VJP of (7): `dy=rw·d_output`; `db2+=dy`; `dW2+=dy⊗h`; `dh=W2ᵀdy`; `dz1=dh⊙[z1>0]`;
  `dW1+=dz1⊗x`; `db1+=dz1`; `d_input=W1ᵀdz1`.
- Relu mask: sm_90 gates dz1 by `h[j]>0`; since `h=relu(z1)`, `h>0 ⟺ z1>0` (z1=0 → both off),
  identical to gfx942 `(z1>0)`. Chain rule correct.
- Atomics: multiple tokens share expert e, so per-expert grad accumulation uses `atomicAdd` (sm_90)
  / `index_add_` (gfx942). sm_90 also uses shared-mem `atomicAdd(&dz1[j],…)` across the d_out loop
  (different lanes, same j) — legitimate; `dz1` is zero-initialized then synced before/after. Correct.
- `d_input[t]=Σ_j W1[j,k]·dz1[j]` matches gfx942 `bmm(W1g.transpose(1,2), dz1)`. EQUIVALENT.

## (9) scan_compacted — CORRECT (VESTIGIAL, never called)
sm_90 2281-2352; gfx942 1238-1273. Low-priority sanity only.
- Standard discretized SSM: `A=-exp(A_log)`, `A_bar=exp(dt·A)` (stable: A<0 ⇒ A_bar∈(0,1]),
  `h_t=A_bar⊙h_{t-1}+(dt·B_t)·x_t`, `y_t=Σ_s C_t[s]·h_t[d,s]+D[d]·x_t[d]`.
- sm_90: one thread/inner-channel, sequential over t, `h[256]` register array guarded by
  `TORCH_CHECK(d_state<=256)`. gfx942: vectorized over (d_inner,d_state), sequential over t.
  Same recurrence, same A/A_bar/y. Numerically sound. EQUIVALENT. `rope_freq` unused in both.

---

## Cross-cutting checks
- Pre-zeroing assumptions verified against the Python caller (counts zeroed at 2248,
  compact_count zeroed at 2281). Consistent with sm_90's accumulate-from-zero design.
- No OOB on E in any kernel (modulo / bounds-checked loops).
- All 4 compact arrays kept in sync at the same out-position in (4); (5) is its exact inverse.
- sm_90 and gfx942 compute identical math for all 9 kernels (verified expression-by-expression).

## Gates (no code changed — baseline == final)
- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/launch_supergrok2.cu -DWITH_CUTLASS`
  → `COMPILE_OK tu=csrc/backends/cuda/sm_90/launch_supergrok2.cu`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` → `137 passed, 1 failed` (137/1)
- `ruff check grokking_optimizers/` → `All checks passed!`
