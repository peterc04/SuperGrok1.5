# csrc/algorithms/ — Optimizer Math Single Source of Truth
## Digest for SuperGrok2 / SuperGrok1.5 parallel analysis

Generated: 2026-06-25. Agent assignment: every .h in csrc/algorithms/ + SOURCE_OF_TRUTH.md.

---

## 1. Architecture Overview (SOURCE_OF_TRUTH.md)

The 11 optimizer implementations in `csrc/algorithms/<opt>.h` are the **canonical single source** for all CUDA paths. Both CUDA consumers — the per-op kernel path (`grokking_optimizers/kernels/sm_90/<opt>_sm90.cuh`) and the fused L3-TC megakernel (`csrc/fused/sm_90/opt_components.cuh`) — derive from these headers via `#include`, making math drift impossible for the CUDA paths.

**Three tiers of math re-expression:**
1. **CUDA (sm_90, per-op + fused)**: single source via `#include` — enforced by `scripts/check_math_single_source.py`
2. **gfx942 HIP**: byte-faithful transcription in `csrc/fused/gfx942/opt_components.hip.hpp` + per-op `kernels/gfx942/<opt>_gfx942.hip.hpp`; cross-referenced but **manually synced** (numeric parity is yellow / unverified on MI300X)
3. **TPU/Pallas**: `csrc/backends/pallas/launch_<opt>.py` is the JAX/XLA canonical — single TPU source; the former `kernels/tpu/` re-export shims were removed in Phase 7

**Enforcement status (SOURCE_OF_TRUTH.md:41-43):**
- CUDA single-source: `enforced + verified` (check_math_single_source.py → OK)
- gfx942/TPU re-expressions: cross-referenced, manual-sync (parity yellow)

**Edit protocol**: change canonical header → (auto) updates both CUDA consumers; manually mirror to gfx942 transcription and TPU JAX path; run `check_math_single_source.py`.

**Math-drift guards (3 teeth in check_math_single_source.py):**
1. Structural single-source: each `<opt>_sm90.cuh` must `#include` its `csrc/algorithms/<opt>.h`
2. Re-inline detection: if a CUDA consumer locally re-types the Adam moment-update/apply (instead of calling the canonical step), the check fails
3. Content-hash manifest: `scripts/optimizer_math_manifest.json` records normalized hashes of each canonical header; editing the canonical math forces `--update-manifest` — silent drift is impossible

---

## 2. Shared Infrastructure

### utils.cuh — sg_safe_bc

```
__device__ __forceinline__ float sg_safe_bc(float bc) {
    return fmaxf(bc, 1e-30f);
}
```

Used by **every** Adam-family optimizer. Guards the bias-correction denominator `bc = 1 - beta^t` against divide-by-zero at step 0 (one `fmaxf`, effectively free). The host contract is `step >= 1`, so this never fires in normal operation.

**Calling convention** (shared across all Adam-family headers): `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t` are passed **un-inverted**; the step functions divide by them using `sg_safe_bc()`. This is noted consistently in every header comment.

---

## 3. Per-Optimizer Catalog

### 3.1 AdamW (`adamw.h`, 107 lines)

**Reference**: Loshchilov & Hutter 2017, arXiv 1711.05101, Algorithm 2 (decoupled WD variant).

**Math:**
```
m_t = beta1*m_{t-1} + (1-beta1)*g
v_t = beta2*v_{t-1} + (1-beta2)*g²
m_hat = m / (1-beta1^t)
v_hat = v / (1-beta2^t)
p -= lr * (m_hat/(sqrt(v_hat)+eps) + wd*p)
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v) — FP32.

**Hyperparameters**: `lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `adamw_step<ParamT,GradT>`: scalar per-element (file:adamw.h:30-62)
- `adamw_step_vec4`: FP32 float4 fast path, processes 4 elements per call (file:adamw.h:65-105)

**Notes**: The vec4 path uses `sg_safe_bc(bc2)` inside `sqrtf(v/bc2)` rather than `v_hat`, avoiding a temporary — slightly different form but mathematically identical.

---

### 3.2 GrokAdamW (`grokadamw.h`, 103 lines)

**Reference**: Lee et al. 2024, "GrokAdamW: A Faster Optimizer for Late Generalization."

**Math:**
```
ema_t = alpha*ema_{t-1} + (1-alpha)*g       [EMA filter]
g_amp = g + lamb*ema_t                       [amplified gradient]
m_t   = beta1*m_{t-1} + (1-beta1)*g_amp
v_t   = beta2*v_{t-1} + (1-beta2)*g_amp²
update = (m/bc1) / (sqrt(v/bc2) + eps)
p    -= lr*(update + wd*p)
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), `ema` — all FP32.

**Hyperparameters**: `alpha` (EMA decay, e.g. 0.98), `lamb` (amplification factor, e.g. 5.0), `lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `grokadamw_step<ParamT,GradT>`: full fused step (file:grokadamw.h:27-64)
- `grokadamw_adam_tail`: factored Adam tail on a caller-supplied effective gradient `g_eff` — re-used by quantized Config-3 path without re-inlining the moment-update math (file:grokadamw.h:79-101)

---

### 3.3 Grokfast (`grokfast.h`, 77 lines)

**Reference**: Lee et al. 2024, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients," arXiv 2405.20233.

**Math (two modes):**
```
Mode A (EMA-only):
  ema = alpha*ema + (1-alpha)*g
  grad_out = g + lamb*ema   [written to grad buffer; downstream Adam consumes it]

Mode B (fused):
  ema = alpha*ema + (1-alpha)*g
  g_amp = g + lamb*ema
  [then full AdamW on g_amp]
```

**State tensors**: `ema`, `exp_avg`, `exp_avg_sq` (Mode B only).

**Hyperparameters**: `grokfast_alpha, grokfast_lamb, lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `grokfast_ema_step<GradT>`: EMA-only path, writes amplified gradient to `grad_out` (file:grokfast.h:26-38)
- `grokfast_fused_step<ParamT,GradT>`: EMA + Adam in one call (file:grokfast.h:43-75)

**Notes**: Grokfast is the simplest grokking-aware optimizer; the two modes correspond to using it as a gradient preprocessor vs. a fully fused optimizer.

---

### 3.4 Lion (`lion.h`, 77 lines)

**Reference**: "EvoLved Sign Momentum" — sign-based optimizer with interpolated momentum.

**Math:**
```
update  = sign(beta1*exp_avg + (1-beta1)*grad)
param  -= lr*(update + wd*param)
exp_avg = beta2*exp_avg + (1-beta2)*grad
```

**State tensors**: `exp_avg` (momentum buffer) — FP32. **No second moment** (v).

**Hyperparameters**: `lr, beta1, beta2, wd`. No `eps` needed (no division).

**Implementations**:
- `lion_step<ParamT,GradT>`: scalar (file:lion.h:18-37)
- `lion_step_vec4`: float4 fast path (file:lion.h:39-75)

**Notes**: Sign function implemented as `copysignf(1.0f, interp)` with explicit zero-guard `(interp != 0.0f) ? ... : 0.0f` (file:lion.h:33). The exp_avg update uses beta2 (NOT beta1), which matches the Lion formulation where the two betas have independent roles.

---

### 3.5 LookSAM (`looksam.h`, 93 lines)

**Reference**: Liu et al. 2022, "Towards Efficient and Scalable Sharpness-Aware Minimization," arXiv 2203.02714, Algorithm 1 (periodic-perturbation variant).

**Math (four operations, split across calls):**
```
(1) perturb:      backup = param; param += rho*(g/||g||)     [scale = rho/(||g||+eps), precomputed]
(2) restore:      param = backup
(3a) SAM step:    sam_dir = g_sam - g_orig
(3b) normal step: g_adj = (1-alpha)*g + alpha*sam_dir
(4) apply:        AdamW on g_adj
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), `backup` (param copy), `sam_dir` (cached gradient difference) — all FP32.

**Hyperparameters**: `rho, alpha` (SAM blend), `k` (step period — managed host-side), `lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `looksam_perturb_step<ParamT,GradT>` (file:looksam.h:28-38)
- `looksam_restore_step<ParamT>` (file:looksam.h:42-48)
- `looksam_set_direction<GradT>` (file:looksam.h:51-59)
- `looksam_apply_step<ParamT,GradT>` (file:looksam.h:63-91): blends cached SAM direction with current grad, then full AdamW

**Notes**: The `scale` parameter in perturb is `rho/(||g||+eps)` and is **precomputed by the host**; the device kernel does not compute the gradient norm. The SAM gradient recomputation cadence (every k steps) is managed entirely host-side.

---

### 3.6 Muon (`muon.h`, 78 lines)

**Reference**: Jordan et al. 2024, "Muon: An optimizer for the orthogonal manifold."

**Math (dual-strategy):**

For **2D parameters** (matrices): Newton-Schulz orthogonalized momentum
```
buf = momentum*buf + grad        [plain SGD momentum — NO (1-momentum) factor]
X   = buf/||buf||_F              [Frobenius-normalized]
[NS iteration, 5 steps]:
  Y = a*X + b*A_X + c*AA_X      [polynomial: (a,b,c) = (3.4445, -4.7750, 2.0315)]
param = param*decay_factor + neg_lr_scale*orth
```

For **1D parameters**: standard AdamW (re-exported via `using sg::algorithms::adamw_step`)

**State tensors**: `buf` (momentum buffer), `X` (normalized), intermediate GEMM outputs for NS iteration — the heavy matrix-multiply work lives in `primitives.cuh`/`mma.cuh`.

**Hyperparameters**: `momentum` (e.g. 0.95), `neg_lr_scale, decay_factor`.

**Implementations**:
- `muon_momentum_normalize_step<GradT>`: updates buf + computes X (file:muon.h:34-46)
- `muon_ns_combine_step`: polynomial combine Y = a*X + b*AX + c*AAX (file:muon.h:49-59)
- `muon_update_step<ParamT>`: trust-ratio scaled param update (file:muon.h:63-73)
- 1D path: `using sg::algorithms::adamw_step` (file:muon.h:76)

**Notes**: The momentum update is **plain SGD-momentum** (no `(1-momentum)` factor on the gradient), matching the bindings.cpp `muon_fused_step` (`bufs[i].mul_(momentum).add_(grads[i])`). The NS polynomial coefficients `(3.4445, -4.7750, 2.0315)` match the Jordan 2024 source.

---

### 3.7 NeuralGrok (`neuralgrok.h`, 119 lines)

**Reference**: Wang et al. 2024, "NeuralGrok: Accelerating Grokking via Learned Gradient Amplification."

**Math (two-stage):**
```
Stage 1 — psi_net forward [2-layer MLP, per-element]:
  h[j] = ReLU(W1[j]*|g| + b1[j])
  s = Σ_j W2[j]*h[j] + b2           [psi_scale output]

Stage 2 — amplify + AdamW:
  g_amp = (s*alpha + beta) * g       [clip_coef applied to g before this]
  [then full AdamW on g_amp]
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), MLP weight tensors (`W1,b1,W2,b2`) in constant/LDS memory.

**Hyperparameters**: `alpha, beta` (affine scaling of psi output), `lr, beta1, beta2, eps, wd, bc1, bc2`, `clip_coef` (global grad-norm clip, default=1.0 = inert).

**Template parameter**: `H` (hidden size of the 2-layer MLP, compile-time).

**Implementations**:
- `neuralgrok_psi_forward<H>`: scalar MLP forward (file:neuralgrok.h:31-46); ReLU hidden, linear output; takes `|grad|` as input
- `neuralgrok_apply_step<ParamT,GradT>`: amplify + full AdamW (file:neuralgrok.h:50-81)
- `neuralgrok_adam_tail<ParamT>`: factored Adam tail on caller-supplied `g_eff` — same DRY motivation as `grokadamw_adam_tail` (file:neuralgrok.h:96-117)

**Notes**: `clip_coef` is applied to g BEFORE the MLP amplification. The MLP weights live in constant memory (CUDA) / LDS (HIP) — not passed via global memory per element.

---

### 3.8 Prodigy (`prodigy.h`, 99 lines)

**Reference**: Mishchenko & Defazio 2023, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner," arXiv 2306.06101, Algorithm 1.

**Math (self-tuning Adam, three-phase):**
```
Phase 1 — per-element partial reductions:
  r_local += g*(param_init - param)*d_prev²   [degree-2 in d, scale-free]
  s_local += d_prev²*|g|                      [L1 norm accumulation, coord-wise]

Phase 2 — d update (single thread, after block/global reduce):
  d_new = max(d_prev, r_sum/|s_sum|)

Phase 3 — apply:
  g_scaled = d*g
  m = beta1*m + (1-beta1)*g_scaled
  v = beta2*v + (1-beta2)*g_scaled²
  s_track += d*g
  update = (m/bc1)/(sqrt(v/bc2)+eps)
  param -= d*(update + wd*param)               [note: uses d, not lr]
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), `param_init` (initial params, FP32 copy), `s_track` (trajectory accumulator).

**Hyperparameters**: `beta1, beta2, eps, wd, bc1, bc2`. No explicit `lr` — d is self-tuned.

**Implementations**:
- `prodigy_partials_step<ParamT,GradT>`: per-element r,s contributions (file:prodigy.h:29-53)
- `prodigy_update_d`: single-thread d update (file:prodigy.h:56-65)
- `prodigy_apply_step<ParamT,GradT>`: Adam apply with d (file:prodigy.h:69-97)

**Critical bug fix documented in code (file:prodigy.h:43-52)**:
- The numerator must carry **d²** (not d¹) to make d_hat = r/s scale-free (degree 0 in d). A prior degree-1 numerator made `d_hat ∝ 1/d`, which at the `d0 = 1e-6` init caused a ~1e6x catapult in the first step, destroying training.
- The denominator accumulates `|g|` per coordinate (not signed g): `abs-of-sum ≠ L1 norm`. A prior version summed signed g and took |Σ| at reduce time — this is now fixed to proper coordinate-wise L1.

---

### 3.9 SuperGrok v1.1 (`supergrok11.h`, 178 lines)

**Reference**: Internal algorithm. Python source of truth: `grokking_optimizers/optimizers/supergrok11.py`.

**Math (2-sweep pipeline per step):**

Sweep A — meta-net forward + cosine accumulation:
```
mu[i] = phi(grad[i], sharpness[i])    [2-layer GELU-MLP, file:supergrok11.h:43-59]
gate_num   += grad * momentum
gate_den_g += grad²
gate_den_m += momentum²
```

Sweep B — cosine gate + smart_grad + AdamW:
```
cos_sim = gate_num / sqrt(gate_den_g*gate_den_m + eps)    [pre-reduced]
gate    = sigmoid(gate_temperature * cos_sim)              [file:sg11_finalize_gate]
smart_grad = grad + (1-gate)*alpha*mu
[then full AdamW on smart_grad]
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), `mu` (meta-net output buffer), momentum buffer (for cosine accumulation), `sharpness` (external input).

**Hyperparameters**: `alpha` (meta-net strength), `gate_temperature`, MLP weights (`W1[H×2], b1[H], W2[H], b2`), `lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `sg11_phi_forward<H>`: 2-layer GELU-MLP forward (file:supergrok11.h:43-59); inputs are `grad_val` and `sharp_val`; activation is **exact erf GELU** (`0.5f*h*(1+erff(h*0.70710...))`) matching `nn.GELU()`
- `sg11_finalize_gate`: CANONICAL gate computation (host+device; file:supergrok11.h:72-84); turns cosine accumulators into sigmoid gate; uses `expf` (not `__expf`) for host compatibility
- `sg11_sweep_a_step<GradT>`: Sweep A per-element (file:supergrok11.h:88-105)
- `sg11_sweep_b_step<ParamT,GradT>`: Sweep B per-element (file:supergrok11.h:109-140)
- `sg11_adam_tail<ParamT>`: factored Adam tail on caller-supplied `g_eff` (file:supergrok11.h:154-176)

**Key distinction from SG1.5**: SG1.1's gate signal is the **per-parameter cosine similarity between grad and momentum** (computed per-element via reduction). SG1.5 uses a **scalar sigmoid of training accuracy** (set host-side). Both use a sigmoid as the final squashing function.

**Historical bug (fixed)**: An earlier implementation clamped the raw cosine and ignored `gate_temperature`. The canonical `sg11_finalize_gate` now correctly applies `sigmoid(gate_temperature * cos_sim)`.

---

### 3.10 SuperGrok v1.5 (`supergrok15.h`, 114 lines)

**Reference**: Internal algorithm. Python source of truth: `grokking_optimizers/optimizers/supergrok15.py`.

**Math (2-sweep pipeline per step):**

Sweep A — meta-net forward + sharpness accumulation:
```
mu[i] = phi(grad[i], sharpness[i])    [same 2-layer GELU-MLP as SG1.1]
sharp_local += grad²                   [sharpness reduction for host-side update]
```

Sweep B — gate (scalar, host-side) + smart_grad + AdamW:
```
alpha_per_coord = clip(alpha_base*(1+mu[i]), 0, alpha_max)    [per-coord alpha gate]
smart_grad = grad + gate_global * alpha_per_coord * mu[i]     [gate_global = sigmoid(accuracy)]
[then full AdamW on smart_grad]
```

**State tensors**: `exp_avg` (m), `exp_avg_sq` (v), `mu` (meta-net output buffer).

**Hyperparameters**: `alpha_base, alpha_max` (per-coord alpha gate bounds), `gate_global` (scalar sigmoid of accuracy, host-computed), MLP weights, `lr, beta1, beta2, eps, wd, bc1, bc2`.

**Implementations**:
- `sg15_phi_forward<H>`: identical MLP structure to SG1.1's `sg11_phi_forward` (file:supergrok15.h:35-52)
- `sg15_sweep_a_step<GradT>` (file:supergrok15.h:56-65)
- `sg15_alpha_per_coord`: per-coord alpha = `clip(alpha_base*(1+mu), 0, alpha_max)` (file:supergrok15.h:68-74)
- `sg15_sweep_b_step<ParamT,GradT>` (file:supergrok15.h:79-112)

**Key distinction from SG1.1**: gate signal is a **scalar** (sigmoid of training accuracy from host), not per-element cosine. No cosine accumulation needed; Sweep A only accumulates sharpness. Per-coord alpha is a function of the meta-net output, providing element-wise adaptivity.

---

### 3.11 SuperGrok v2 (`supergrok2.h`, 578 lines) — the flagship optimizer

**Reference**: Internal algorithm. Python reference: `grokking_optimizers/optimizers/supergrok2.py`.

**Architecture**: CSA/HCA compressed-attention + 4-Head PEER product-key routing + per-element GRU + Adam.

**Per-step pipeline:**
```
(1) input_proj_sort:   [grad, sharpness] → [N, d_model], sort keys = |grad|
(2) CSA attention:     compressed-sparse (m=4, top-k, +window) → csa_ctx
    HCA attention:     heavily-compressed (m'=128, dense, +window) → hca_ctx
(3) peer_route:        product-key expert routing, top-4 of (pk_dim)² experts
(4) gru_step:          per-element GRU integrates expert output with temporal state
(5) apply:             smart_grad + Adam + decoupled WD
```

**Key per-element apply math (sg2_apply_step, file:supergrok2.h:394-438):**
```
mu_new   = gru_decay*mu_state + (1-gru_decay)*expert_out   [expert-output EMA]
slow_new = alpha*slow_state + (1-alpha)*g                  [slow-gradient EMA]
smart_grad = g + alpha*mu_new + lamb_eff*slow_new          [= grad + GRU term + grokfast term]
m = beta1*m + (1-beta1)*smart_grad
v = beta2*v + (1-beta2)*smart_grad²
update = (m/bc1)/(sqrt(v/bc2)+eps)
param -= lr*(update + wd*param)
```

**State tensors per element**: `exp_avg` (m), `exp_avg_sq` (v), `mu_state` (expert-EMA), `slow_state` (slow-gradient EMA / grokfast accumulator).

**Hyperparameters**: `alpha` (BOTH mu/slow mixing AND slow-EMA decay), `gru_decay`, `lamb_eff` (grokfast amplification = lamb·ramp·gate), `lr, beta1, beta2, eps, wd, bc1, bc2`.

**SG2-specific constants (file:supergrok2.h:99-106)**:
```cpp
constexpr int SG2_MAX_D_MODEL = 64;       // max feature width (default d_model)
constexpr int SG2_CSA_WINDOW_MAX  = 16;   // default CSA_WINDOW = 8
constexpr int SG2_CSA_TOPK_MAX    = 64;   // default CSA_TOPK = 16
constexpr int SG2_INDEXER_RANK_MAX = 8;   // default INDEXER_RANK = 4
constexpr int SG2_HCA_COMPRESS    = 128;  // HCA stride m'
```

**Implementations**:
- `sg2_input_proj_sort<scalar_t,wt_t>`: input projection + sort key computation (file:supergrok2.h:117-146)
- `sg2_csa_compress_kv<feat_t>`: CSA KV compression with learned softmax pooling, online softmax for numerical stability (file:supergrok2.h:167-200)
- `sg2_csa_index_score`: lightning indexer dot-product score `qI·kI/sqrt(rank)` via `ptx_fma` + `fast_rsqrt_nr` (file:supergrok2.h:217-229)
- `sg2_attention_score_and_accumulate`: FlashAttention-style online softmax streaming update (file:supergrok2.h:252-283)
- `sg2_softmax_finalize`: divide accumulator by denominator (file:supergrok2.h:293-303)
- `sg2_hca_compress_kv<feat_t>`: HCA (stride=window=128), mean or learned weighted pool (file:supergrok2.h:321-362)
- `sg2_apply_step<ParamT,GradT>`: GRU + slow EMA + smart_grad + Adam (file:supergrok2.h:394-438)
- `sg2_bilevel_precompute_timestep`: recomputes q/k/v + indexer projections for bilevel adjoint (file:supergrok2.h:461-513)
- `moe_adam_step<ParamT,GradT>`: thin wrapper around `adamw_step`, re-exported for MoE-parameter-group symmetry (file:supergrok2.h:529-545) — merged in from former `csrc/algorithms/moe_adam.h`

**PTX helpers inlined at top of supergrok2.h** (CUDA path only, file:supergrok2.h:36-79):
- `fast_rsqrt_nr`: PTX rsqrt.approx.f32 + one Newton-Raphson iteration (2-3x faster than `sqrtf+fdividef` for Adam denominator)
- `ptx_fma`: PTX fma.rn.f32 (ensures single FMA in affine_combine inner loop)
- `ptx_exp2`: PTX ex2.approx.f32
- `ptx_expf`: fast exp via exp2 (exp(x) = exp2(x * log2(e)))
HIP fallbacks use `rsqrtf`, `fmaf`, `expf`.

**CRITICAL BUG FIX documented (file:supergrok2.h:383-391)**:
The `lamb_eff*slow_new` grokfast term was **silently dropped** in a prior refactor. `sg2_apply_step` took no `lamb` param; launchers `(void)`-ed `lamb_eff`/`lamb_effs`; the host-computed amplification never reached the update. This is **RESTORED** in the current code: `sg2_apply_step` now takes `slow_state` and `lamb_eff` parameters and computes the full `smart_grad = g + alpha*mu_new + lamb_eff*slow_new`.

**Open TODO (file:supergrok2.h:549-578)**:
> The fused path `csrc/fused/**` shares the SAME grokfast drop fixed above, but in a DIFFERENT place than the per-op path.

However, checking `csrc/fused/sm_90/opt_stage_supergrok2.cuh`, the fused path **does** call `sg2_apply_step` (via `sg2alg::sg2_apply_step`) with `mu_state, slow_state, grad, expert_out, alpha, gru_decay, lamb_eff` — the fused fix is already applied. The TODO comment in supergrok2.h appears to be a stale artifact from when the fix was made to the per-op path before the fused path was updated; the fused path has since been converged.

---

## 4. SuperGrok v2 Bilevel Adjoint (`supergrok2_bilevel_adjoint.h`, 869 lines)

**Purpose**: Real reverse-mode VJP (no torch::autograd, no grad-tracking tensors) through the SG2 CSA/HCA/GRU/PEER meta-net forward. Used for bilevel meta-learning: the meta-net weights are trained to minimize a validation loss using the adjoint of the optimizer's update rule.

**Uses ATen directly** (`torch::mm`, `torch::einsum`, etc.) — header-only, shared between sm_90 and gfx942.

**Forward pipeline covered** (matches `CSAHCAMetaNet.forward_for_bilevel`, supergrok2.py:734):
```
input_proj+sort → CSA → HCA → GRU → PEER routing + expert MLP → smart_grad
```

**Backward (adjoint) pipeline** (implemented in reverse order):
```
smart_grad → PEER → GRU → HCA → CSA (incl. compression + indexer) → input_proj (scatter via sort_indices)
```

**Key structs:**
- `SavedActs` (file:sg2_bilevel_adjoint.h:58-83): activation bundle persisted from forward save; includes x_sorted, sort_idx, csa_ctx/hca_ctx, attention probs, GRU gates, PEER tensors
- `CsaFwd` (file:sg2_bilevel_adjoint.h:101-117): all intermediates needed for CSA backward
- `HcaFwd` (file:sg2_bilevel_adjoint.h:342-354): HCA intermediates

**Key functions:**
- `bilevel_forward_save(...)` (file:sg2_bilevel_adjoint.h:523-589): full meta-net forward; fills SavedActs
- `csa_forward(...)` (file:sg2_bilevel_adjoint.h:119-193): CSA with indexer, compression, top-k sparse + window attention
- `csa_backward(...)` (file:sg2_bilevel_adjoint.h:198-336): CSA VJP
- `hca_forward(...)` (file:sg2_bilevel_adjoint.h:361-419): HCA dense attention
- `hca_backward(...)` (file:sg2_bilevel_adjoint.h:421-501): HCA VJP
- `peer_head_backward(...)` (file:sg2_bilevel_adjoint.h:610-706): PEER product-key routing + expert MLP VJP; RECOMPUTES forward from saved inputs (avoids storing per-expert intermediates)
- `bilevel_backward_driver(...)` (file:sg2_bilevel_adjoint.h:713-867): top-level driver; accumulates all 24+ weight-grad buffers

**PEER specifics (from bilevel_adjoint)**:
- Product-key routing: two independent top-k selections (`scores_a.topk(topk)` and `scores_b.topk(topk)`), creating `topk²` active experts per sample (file:sg2_bilevel_adjoint.h:623)
- Soft temperature: `softmax(vals * T, T=10)` for differentiable routing
- Expert MLP: 2-layer `[1, expert_hidden, 1]` (scalar input g → H → scalar output); expanded to `[N, num_active, expert_hidden]` for batched matmul
- Top-k is treated as a stop-gradient routing decision (discrete; gradients flow through gathered values only)

**Checkpointing** (file:sg2_bilevel_adjoint.h:25-35):
- Saves heavy contexts (csa_ctx, hca_ctx, attention probs, GRU gates, sort permutation)
- **Recomputes** cheap per-row q/k/v and indexer projections from x_sorted (activation checkpointing at layer boundary)
- checkpoint_interval accepted and threaded through; for this "first correct cut" recompute granularity is "every layer" (≤ MAX_CKPT_INTERVAL=32)

**Lightning indexer gradient note (file:sg2_bilevel_adjoint.h:299-308)**:
- `d_csa_idx_*` (idx_DQ, idx_UQ, idx_K) accumulate **exactly zero by construction** — the indexer's only consumer is the non-differentiable top-k index; gradients flow through the values at selected indices, not through the scoring/selection
- HARDWARE_VALIDATION.md marks d_csa_idx_* as exactly-zero-by-construction

---

## 5. The 509 GB Per-CTA Workspace Claim

**Source**: `.session_memory/flagship-distributed-config.md` and `PERF_ANALYSIS.md:40`.

**Context**: The SG2 optimizer's CSA/HCA/PEER meta-net produces a per-CTA workspace (`dec_tc_sg2_floats = nCTA * ~91 * Nmax`) that is **linear in Nmax** — the largest tensor's number of elements. For a single-GPU decoder at 1.5B params (d=1600, ff weight = `d_ff * d` = 10.24M elements), this totals ~509 GB — does not fit in a single 80 GB H100.

**Resolution**: TP=8 shards that weight → per-rank Nmax = 10.24M/8 = 1.28M → SG2 scratch shrinks to ~58 GB/rank. At TP=8: 10 of 11 optimizers run at 1-CTA/SM (nCTA=132, 66–68 GiB/rank); SuperGrok2 auto-caps to nCTA=64 (~40.9 GiB). The flagship GENUINELY REQUIRES 4D sharding (TP≥2) to fit SG2.

The "~509 GB" figure is for the staged-opt workspace (the SG2 meta-net activations + scratch), not for the parameter count itself. It confirms the central design claim that the memory strategy is config-derived and multi-GPU is required not by policy but by fit.

---

## 6. Math-Drift Guard: Known Risks

1. **gfx942/TPU re-expressions**: manually synced, parity yellow (SOURCE_OF_TRUTH.md:42-43). Any edit to csrc/algorithms/*.h must be mirrored manually to `csrc/fused/gfx942/opt_components.hip.hpp` and `csrc/backends/pallas/launch_<opt>.py`. The content-hash manifest detects if the canonical changed without a manifest update (--update-manifest ack required).

2. **Stale TODO in supergrok2.h** (lines 549-578): References the fused path needing `slow_state` + `lamb_eff`. This TODO is **stale/already resolved** — `csrc/fused/sm_90/opt_stage_supergrok2.cuh` already calls `sg2_apply_step` with both `slow_state` and `lamb_eff` threaded through. The TODO should be removed or updated.

3. **Prodigy degree-2 fix**: the prior degree-1 numerator (catapult bug) is fixed in the canonical header. The fix must be mirrored to gfx942/TPU paths per the edit protocol.

4. **Prodigy L1 norm fix**: the prior `abs-of-sum ≠ L1 norm` bug is fixed (per-coordinate `fabsf(g)` accumulation). Same mirror requirement.

5. **SG1.1 gate formula fix**: the prior clamp-of-raw-cosine (ignoring gate_temperature) is fixed in `sg11_finalize_gate`. Same mirror requirement.

---

## 7. Implementation Status Summary

| Optimizer | State | State tensors | Notable |
|-----------|-------|---------------|---------|
| AdamW | Fully implemented, vec4 fast path | m, v | Reference baseline |
| GrokAdamW | Fully implemented | m, v, ema | Factored adam_tail for Config-3 quant path |
| Grokfast | Fully implemented (2 modes) | ema [, m, v] | EMA-only + fused modes |
| Lion | Fully implemented, vec4 fast path | exp_avg only | No second moment |
| LookSAM | Fully implemented (4 sub-ops) | m, v, backup, sam_dir | Scale precomputed host-side |
| Muon | Per-element pieces only; GEMMs in primitives.cuh | buf, X, [m,v for 1D] | NS polynomial (3.4445, -4.7750, 2.0315) |
| NeuralGrok | Fully implemented | m, v [, W1,b1,W2,b2] | clip_coef before MLP |
| Prodigy | Fully implemented (2 bugs fixed) | m, v, param_init, s_track | degree-2 + L1 norm fixed |
| SG1.1 | Fully implemented (2-sweep) | m, v, mu [, momentum] | Per-element cosine gate; gate_temp bug fixed |
| SG1.5 | Fully implemented (2-sweep) | m, v, mu | Scalar accuracy gate from host |
| SG2 | Fully implemented per-element; fused path uses sg2_apply_step | m, v, mu_state, slow_state | grokfast term restored; PEER d²² experts; 509 GB scratch at flagship scale |
| SG2 bilevel | Fully implemented, hand-written VJP | SavedActs bundle | 869 lines; indexer grads zero by construction; TODO stale |

---

## 8. Files Read (13 total)

- `/workspace/SuperGrok1.5/csrc/algorithms/SOURCE_OF_TRUTH.md`
- `/workspace/SuperGrok1.5/csrc/algorithms/adamw.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/grokadamw.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/grokfast.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/lion.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/looksam.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/muon.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/neuralgrok.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/prodigy.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/supergrok11.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/supergrok15.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/supergrok2.h`
- `/workspace/SuperGrok1.5/csrc/algorithms/supergrok2_bilevel_adjoint.h`

Plus supporting reads:
- `/workspace/SuperGrok1.5/csrc/common/utils.cuh` (for sg_safe_bc)
- `/workspace/SuperGrok1.5/.session_memory/flagship-distributed-config.md` (for 509 GB claim)
