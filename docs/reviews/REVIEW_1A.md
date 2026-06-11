# REVIEW_1A — Correctness review of the SG2 CSA/HCA bilevel backward adjoint

**Reviewer:** Opus 4.8 (max effort), 2026-06-01.
**Scope:** `csrc/algorithms/supergrok2_bilevel_adjoint.h` (the ~44KB shared VJP
header), plus the launchers in
`grokking_optimizers/kernels/sm_90/supergrok2_sm90.cuh` and
`grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp`.
**Oracle:** `CSAHCAMetaNet.forward_for_bilevel` (`grokking_optimizers/optimizers/supergrok2.py:734`).

## Method

The header math is pure ATen and device-agnostic, so I validated it on CPU
without a GPU. I wrote a **line-by-line faithful Python transcription** of the
header (`csa_forward/csa_backward`, `hca_forward/hca_backward`,
`peer_head_backward`, `bilevel_backward_driver`) and compared every weight-grad
buffer it produces against `torch.autograd.grad` taken through the **real**
`forward_for_bilevel`. Tested configs:

| N | csa_compress | csa_window | csa_topk | hca_compress | notes |
|---|---|---|---|---|---|
| 20 | 2 | 4 | 3 | 4 | exercises nc>topk selection, multi-Nh |
| 30 | 4 | 8 | 16 | 128 | **production defaults** |
| 5 | 2 | 4 | 3 | 4 | small-N edge (Nh=1, window≥N) |
| 20 | 4 | 8 | 16 | 4 | multi-Nh HCA |

Each config run **twice**: once with `gru_state = 0` and once with
`gru_state = randn*0.5` to exercise the reset-gate `r` path (the subtle GRU
path the brief flagged). **All 24 weight-grad buffers (36 incl. per-head
expansions) match autograd to fp32 precision** (max abs error ≤ 6e-9, rel ≤
4e-7) in every config. Several buffers (`peer_Wq`, `prod_A/B`) match
**bit-for-bit** (diff = 0.0) because the adjoint replays the identical op order.

## Per-stage verdict

### 1. `smart_grad = grad + rescale * expert_out`  — **CORRECT**
`bilevel_backward_driver:748-753`. `d_total = rescale*d_smart_grad`,
`d_head_out = d_total / num_peer_heads` correctly inverts
`total_expert_out = (Σ_h head_out) / num_peer_heads` (oracle line 804). The `+g`
term contributes no meta-param grad. Split over PEER heads is correct (each head
receives the same `d_head_out`, matching the sum-then-mean forward).

### 2. PEER expert MLP + product-key routing — **CORRECT**
`peer_head_backward:610-707`.
- **relu 2-layer MLP VJP**: `d_pre_z = d_z * (pre_z>0)` (line 671) is the exact
  relu mask; `d_z = d_out * W2` (668), `d_W2 = d_out*z` (665), `d_W1 = d_pre_z*g`
  (675). Verified vs autograd.
- **softmax routing (×10 temp)**: `d_top_vals = soft*(d_soft - Σ d_soft·soft)*T`
  with `T=10` (686-688) — correct `(diag(p)-ppᵀ)` Jacobian times the temperature
  chain factor. Verified.
- **top-k gather→scatter**: `d_scores.scatter_(1, top_idx, d_top_vals)` (692-694)
  is the correct adjoint of `top_vals = gather(scores, top_idx)`.
- **query projection**: `d_A = d_scores_a.t()@q_a`, `d_q_a = d_scores_a@A`,
  `d_Wq = d_query.t()@peer_input`, `d_peer_input = d_query@Wq` — all correct
  transposes.
- **atomics**: `d_expert_{W1,b1,W2,b2}` use `index_add_(0, eidx_flat, …)`
  (664,666,673,676). `index_add_` is the correct accumulating scatter for the
  many-tokens→same-expert case (atomic on device). Verified numerically:
  `expert_*` grads match autograd even with expert-index collisions across the
  N×16 active set.

### 3. GRU z/r/h̃ — **CORRECT**
`bilevel_backward_driver:777-813`. `h_new=(1-z)*h_old+z*h̃` →
`d_z=dnew*(h̃-h_old)`, `d_h̃=dnew*z`, `d_h_old=dnew*(1-z)` (780-782). The subtle
reset-gate path — `h̃=tanh(Wh·[gru_input, r*h_old])` — is handled correctly:
`d_xrh` is split, the `r*h_old` block gives `d_r = d_rh*h_old` and
`d_h_old += d_rh*r` (793-795). sigmoid VJPs `z(1-z)`, `r(1-r)` and tanh `(1-h̃²)`
all correct. Confirmed against autograd **with a nonzero gru_state** so the `W_r`
path is non-degenerate (`d_gru_Wr/br` match, max abs 7e-15). `d_h_old` is
correctly discarded (it flows into the carried `gru_state`, not a per-step
meta-param).

### 4. HCA dense attention — **CORRECT**
`hca_forward/hca_backward:361-501`. Softmax VJP
`dS = P⊙(dP - Σ(dP⊙P))` (447-448), masked-window positions zeroed (451-452),
mean-pool compression scatter via `index_add_` on `gather_c` with `w_eff`
(488-492), window scatter via `win_idx_c` (476-483), `d_hca_out_W = d_out.t()@ctx`
(848). Multi-head split/merge (`split_heads` permute) correctly inverted. Matches
autograd.

### 5. CSA sparse attention — **CORRECT**
`csa_forward/csa_backward:119-336`.
- joint `(selected-compressed ∪ window)` softmax VJP identical structure to HCA,
  correct (230-236).
- selected-entry value/key grads scatter to `d_c_k/d_c_v` via `index_add_` on
  `sel` (265-274) — correct adjoint of `c_k[sel]` gather (gradient flows through
  the **gathered values**, not the discrete indices).
- **learned softmax-pool compression VJP** for `d_csa_compress_w`: the
  normalized weighted pool `w_eff = (pool_w·valid)/Σ(pool_w·valid)` backward
  (310-322) correctly composes the normalization Jacobian
  `d_num=(d_w_eff - Σ(d_w_eff·w_eff))/denom` with the softmax Jacobian
  `pool_w*(d_pool_w - Σ d_pool_w·pool_w)`. Verified vs autograd (`csa_compress_w`
  matches, rel 3e-7). This was a 🟡 ledger item — **now numerically confirmed
  correct on CPU** (still 🟡 for on-device confirm, but the math is right).

### 5b. Lightning-indexer top-k stop-gradient — **CORRECT**
`csa_backward:296-308` accumulates **zero** into `d_csa_idx_{DQ,UQ,K}`. I
verified against autograd that the oracle's indexer params feed **only** the
`idx_scores.topk(...).indices` path — a non-differentiable discrete index. A
minimal autograd repro confirms PyTorch reports those params as **not in the
graph at all** (`does not require grad and does not have a grad_fn`), i.e. exactly
zero/None grad. **The stop-gradient is the mathematically correct adjoint** of
the oracle. The header's reasoning (lines 275-279, 296-308) is sound.

### 6. input_proj + sort — **CORRECT**
`bilevel_backward_driver:824-864`. The sort is a permutation (VJP = inverse
permutation, not differentiated): `csa_ctx = csa_out[unsort_idx]` →
`d_csa_out = d_csa_ctx.index_select(0, sort_idx)` (828) is correct since
`sort_idx = unsort_idx.argsort()`. `input_proj` accumulation
`d_W[:,0]=Σ d_x·g_sorted`, `d_W[:,1]=Σ d_x·s_sorted`, `d_b=Σ d_x` (861-864) is
the correct transpose of `x = stack(g_sorted,s_sorted)@W.t()+b`. Verified
(`input_proj_W/b` match).

## Cross-cutting checks

- **(a) all 24 buffers written**: every buffer receives a contribution; none
  silently dropped. The three `d_csa_idx_*` are intentionally zero (correct, see
  5b). Verified all present and matching.
- **(b) zero-init**: the driver uses `.add_()`/`index_add_()` accumulate
  semantics; the **caller** must zero the 24 output buffers (documented contract,
  header lines 37-38). The bindings (`supergrok2_bilevel_backward`,
  bindings.cpp:1890) are pure pass-through and do **not** zero them. The
  launchers correctly zero their internal per-head temporaries
  (`dpeer_Wq/dprod_A/dprod_B`, sm90:1700-1702 / gfx942:890-892) before
  accumulating, then `add_` into the (caller-zeroed) outputs. **Documented
  contract, not a bug** — flagged 🟡 below for an on-device guard.
- **(c) fwd_save save-set**: `x_sorted`, `sort_indices`, `csa_ctx`, `hca_ctx`,
  `csa_sel_idx`, probs are saved and round-trip correctly. The backward
  RECOMPUTES q/k/v + indexer projections and the compressed pools from
  `x_sorted` (layer-boundary checkpointing) — verified this recompute reproduces
  the forward exactly (the parity test recomputes the same way and matches). One
  gap flagged 🟡 below (GRU gates).
- **(d) numerical hazards**: softmax denom uses `clamp_min(1e-12)` on the pool
  normalizer (149,314,385); masked attention uses `-inf` fill then softmax (no
  NaN because at least the diagonal/self window position is always valid);
  relu/tanh/sigmoid domains are safe. No div-by-zero or log/exp domain issues
  found. All accumulators are `zeros_like`-initialized before use; `d_h_old` is
  assigned (782) before its first `+=` (795). No uninitialized reads.

- **gfx942 parity**: the gfx942 launcher
  (`supergrok2_gfx942.hip.hpp:802-933`) calls the **identical** shared
  `sg2adj::bilevel_backward_driver` / `bilevel_forward_save` with a byte-for-byte
  equivalent SavedActs reconstruction to sm_90. Math-identical by construction —
  no independent verification needed beyond the shared header.

## Bugs fixed

**None.** No clear bug (wrong VJP / dropped gradient / sign error) was found. The
adjoint is mathematically correct end-to-end and matches autograd to fp32
precision in every tested regime.

## SUSPECT / hardware-validation items (documented, not changed)

1. **GRU-gate recompute fallback drops biases** (🟡). When the caller passes
   *empty* `gru_z_gate/gru_r_gate/gru_h_tilde`, both launchers recompute the
   gates **without the gate biases** (`linear_fwd(xh, gru_Wz)` with no bias —
   sm90:1684,1687,1691 / gfx942:875,878,882). The oracle's gates DO include
   `gru_{bz,br,bh}`, so the bias-free recompute would be inexact. The documented
   save-set (HARDWARE_VALIDATION.md) lists the gates as saved, so the canonical
   path passes them and is exact (this is the path my parity test validates).
   This is a latent plumbing hazard only if a caller violates the save-set
   contract. The backward ABI does not currently receive the gate biases, so a
   correct in-place fix would require an ABI change; deferred as a 🟡 item rather
   than risk an invasive signature change. Added to HARDWARE_VALIDATION.md.

2. **Output-buffer zero-init is a caller contract with no guard** (🟡). The
   bindings do not zero the 24 `d_*` buffers; correctness relies on the caller
   zeroing them. Recommend the on-device numerics harness explicitly zero (or the
   binding `zero_()` them) before the accumulate. Added to HARDWARE_VALIDATION.md.

3. **General on-device bit-parity** (🟡, pre-existing). The CPU parity here is
   strong evidence, but the ledger's on-silicon `rtol=1e-3/atol=1e-5` check vs
   autograd should still be run on H100/MI300X per the existing Stage-1A oracle.

## Gate tokens

- `bash scripts/compile_to_object.sh csrc/backends/cuda/sm_90/launch_supergrok2.cu -DWITH_CUTLASS` → `COMPILE_OK`
- `ruff check grokking_optimizers/` → `All checks passed!`
- `PYTHONPATH=. python grokking_optimizers/compile.py --self-test` → (see report)
