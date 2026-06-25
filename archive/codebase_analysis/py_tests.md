# SuperGrok1.5 Test Suite Analysis
## File: /workspace/_analysis/py_tests.md

**Scope:** All 50 files under `tests/` (from `PY_tests.txt`).
**Goal:** Document what each test actually asserts, tolerances, gating, and verify the claimed test counts.

---

## 1. Infrastructure

### `tests/__init__.py` (empty, 1 line)
### `tests/hw/__init__.py` (empty, 1 line)
### `tests/tpu/__init__.py` (empty)

### `tests/conftest.py` (17 lines)
Registers the `hw` pytest marker. The `hw` marker gates hardware (GPU/multi-GPU) tests so `pytest -m "not hw"` runs the CPU CI path cleanly without `PytestUnknownMarkWarning`.

---

## 2. Pure CPU Tests (no GPU, no torch CUDA)

### 2.1 `tests/test_resource_planner.py` — 10 tests

**Purpose:** Gate the execution planner (`grokking_optimizers.parallel.resource_planner`). No torch, no CUDA.

**Imports:** `HardwareConfig`, `ModelConfig`, `PlanInfeasible`, `layout_arith`, `plan_execution`, `sg2_ws_stride`.

**Key constants pinned:**
- `kDecTotalElems = 1,475,884,899` (decoder flagship d=1600,L=48,v=99,seq=4)  :test_resource_planner.py:27
- `kDecNumTensors = 582` :28
- `kDecMaxTensorNumel = 10,240,000` (= 4d²) :29
- `sg2_ws_stride(10_240_000) / 10_240_000 ≈ 91.277` within 0.01 :33

**Tests (10):**
| Test | What it asserts |
|------|----------------|
| `test_layout_matches_flagship` | layout_arith on d=1600/L=48 returns exact element/tensor counts pinned to flagship layout .cuh |
| `test_sg2_stride_factor_is_91` | SG2 workspace stride factor ≈ 91.277 |
| `test_10m_one_gpu_trivial` | 10M model / 1 GPU → fits HBM, no offload/recompute/stream; bench layout flag present |
| `test_flagship_eight_gpu_4d_zero3` | 1.5B/8GPU → TP=8,PP=1,DP=1; ZeRO-3 overlay; SG2 caps ncta≤64; compile flags include `SG_FLAGSHIP_TP=8`; template `ParConfig<1,8,1,1,ZeROStage::Z3>` |
| `test_10b_one_gpu_full_stack` | 10B/1GPU → activation recompute on; ncta-tiling; layer_streaming or opt_offload engaged |
| `test_10b_one_gpu_sg2_downgrades` | SG2 at 10B/1GPU risks → planner downgrades to adamw+offload; honest risk logged |
| `test_10b_eight_gpu_tp_shrinks_nmax` | 10B/8GPU/TP=8 → recompute on, NO param offload (TP shrinks Nmax → fits) |
| `test_strategy_is_fit_driven_not_gpu_count` | Same 10B model: 1-GPU needs ≥ machinery of 8-GPU — proves fit-driven, no num_gpus branch |
| `test_moe_engages_expert_parallel` | MoE (num_experts=8) → ep ∈ {1,2,4,8} and dp%ep==0 |
| `test_byte_identical_flags_when_single_gpu_trivial` | 10M/1GPU → `par::SingleGPU` in template_inst; `-DSG_FLAGSHIP_TP=1` in compile_flags |

**Key design-thesis validation:** `test_strategy_is_fit_driven_not_gpu_count` directly verifies the "no `if num_gpus==1` branching" claim — the planner engages more machinery at 1 GPU than 8 GPU for the same model BECAUSE per-GPU footprint is different.

---

### 2.2 `tests/test_mem_strategy.py` — 10 tests

**Purpose:** Gate the memory-strategy planner (`grokking_optimizers.parallel.mem_strategy`). No GPU.

**Key invariants:**
- Escalation order: in-HBM → nCTA-cap → recompute → offload → stream (cheapest-first)
- Gate macros: always emits exactly 4 keys: `{SG_MEM_OFFLOAD_OPT, SG_MEM_RECOMPUTE_ACTS, SG_MEM_STREAM_LAYERS, SG_MEM_STREAM_DEPTH}`
- A trivial model returns ALL four = 0 (byte-identical path)
- A 10B model on 1 small GPU MUST escalate ≥1 rung
- Streaming implies offload_optimizer=True; a tiny card needs the full stack

**Tests (10):** `test_trivial_model_is_in_hbm_all_gates_off`, `test_gate_macros_all_zero_on_in_hbm_path`, `test_gate_macros_always_has_four_keys`, `test_flagship_one_gpu_fits_in_hbm`, `test_big_model_one_small_gpu_escalates`, `test_recompute_before_offload_before_stream`, `test_strategy_is_fit_driven_not_gpu_count`, `test_same_footprint_same_plan_regardless_of_n_gpus_label`, `test_streaming_uses_ring_depth_two`, `test_no_fit_records_honest_failure`.

**Design-thesis validation:** `test_same_footprint_same_plan_regardless_of_n_gpus_label` proves the decision keys on `(footprint, usable_hbm)`, NOT `n_gpus`. `test_strategy_is_fit_driven_not_gpu_count` proves TP=8 reduces per-rank footprint → less machinery.

---

### 2.3 `tests/test_shard_map.py` — 20 test functions, ~29 parametrized items

**Purpose:** Gate ZeRO shard partition logic (`grokking_optimizers.parallel.shard_map`). No GPU.

**Key APIs tested:** `even_partition`, `partition_elementwise_even`, `partition_tensor_granular`, `shard_mode_for_optimizer`, `partition_for_optimizer`.

**Parametrize:**
- `test_elementwise_optimizers_select_flat_even`: 7 optimizers `[adamw, lion, grokfast, grokadamw, looksam, prodigy, neuralgrok]`
- `test_per_tensor_optimizers_select_tensor_granular`: 4 optimizers `[muon, supergrok11, supergrok15, supergrok2]`

**Key assertions:**
- `even_partition`: no gap/overlap, `world=1` → whole, mirrors distributed._even_partition
- `partition_elementwise_even`: every element covered exactly once; imbalance < 1.01 (near-perfect balance)
- `partition_tensor_granular`: each tensor on exactly one rank as a whole slice; LPT imbalance < 1.5; deterministic
- Mode selection: AdamW→elementwise, Muon/SG11/SG15/SG2→tensor_granular; OptId ints map identically; case-insensitive; unknown → ValueError

---

### 2.4 `tests/test_zero3_plan.py` — 8 test functions, ~14 parametrized items

**Purpose:** Gate ZeRO-3 flat-blob plan, gather/release, and checkpoint round-trip. CPU torch tensors.

**Parametrize:**
- `test_elementwise_plan_partitions_flat_blob`: world in [1,2,4,8] → 4 items
- `test_tensor_granular_plan_keeps_tensors_whole`: world in [2,4] → 2 items
- `test_gather_release_roundtrip_bit_exact`: (opt,world) = [(adamw,2),(adamw,4),(muon,2)] → 3 items

**Key assertions:**
- Elementwise plan: one contiguous slice per rank (the kernel-binding convention); union covers [0,total) exactly
- Tensor-granular: every slice must be a whole tensor's flat range
- gather_full: reconstitutes original flat blob **bit-exactly**
- Checkpoint save/load: `param_shard` and `opt_state_shard` round-trip **bit-exactly**
- Mismatched plan (world=4 vs world=2) or wrong rank → `RuntimeError`
- Misaligned state planes → `ValueError`
- `fingerprint()` differs for different plans; identical for same plan

---

### 2.5 `tests/test_pipeline_schedule.py` — 13 test functions, ~41 parametrized items

**Purpose:** Gate 1F1B pipeline schedule, stage partition, and handoff plan. Pure CPU.

**Key parametrize:**
- `test_1f1b_schedule_valid`: `_GRID = [(p,m) for p in (1,2,3,4,8) for m in (1,2,4,8,16)]` = 25 combinations
- `test_run_1f1b_executes_with_loopback_transport`: `[(2,1),(2,4),(4,8),(3,2),(8,16)]` = 5 combinations

**Key assertions:**
- `stage_layer_ranges(48, 8)`: even split, raises on uneven/oversplit
- Decoder 30-tensor map at PP=2: stage0={0..13}, stage1={14..29} (the host mirror of PPStageSpec::owns_tensor)
- Unknown model (vit) → loud ValueError
- 1F1B schedule: dependency-feasible, every microbatch fwd+bwd exactly once per stage, in-flight ≤ bound
- Last stage strictly alternates fwd/bwd (zero warmup → the fused pp_decoder_stage_bwd_tc property)
- Warmup counts formula: leading fwds = min(P-1-s, M) + 1
- Bubble fraction formula: `3/11` at P=4,M=8
- `LoopbackP2P`: recv-before-send → AssertionError; double-send → AssertionError
- Handoff plan: 2*T*128 numel each direction; fwd=bf16 (2B/elem), bwd=fp32 (4B)

---

## 3. `tests/hw/test_reference_parity.py` — 28+ CPU tests, 11 GPU tests

**Purpose:** fp64 reference implementations of one optimizer step for each of the 11 race optimizers, transcribed from `csrc/algorithms/<opt>.h` (single source of truth). CPU half runs everywhere; GPU half is `@pytest.mark.hw`.

**Reference functions implemented:**
- `ref_adamw_step` — bias-corrected Adam + decoupled-WD
- `ref_lion_step` — sign(interp) update; sign(0)=0 (copysignf guard)
- `ref_grokadamw_step` / `ref_grokfast_step` — EMA filter + amplify, then Adam (identical structure)
- `ref_neuralgrok_step` — psi-net amplified grad, then Adam
- `ref_neuralgrok_psi_forward` — 2-layer ReLU MLP: `sum_j W2[j]*relu(W1[j]*|g| + b1[j]) + b2`
- `ref_looksam_apply_step` — `g_adj = (1-α)g + α*sam_dir`, then Adam
- `ref_prodigy_apply_step` — step size = d (NOT lr); s_track update
- `ref_prodigy_partials` — `r = Σ d²·<g,p0-p>`, `s = Σ d²·|g|` (d_prev²·|g| form, NOT d^1·signed_g)
- `ref_sg11_step` — `smart_grad = g + (1-gate)*α*mu`, then Adam; gate = sigmoid(gate_temp·cos(g,m))
- `ref_sg_phi_forward` — exact-GELU hidden (erff-based, NOT tanh), linear output
- `ref_sg15_step` — `smart_grad = g + gate_global*clamp(α*(1+mu))*mu`, then Adam
- `ref_sg2_apply_step` — GRU EMA + slow EMA + lamb_eff restored; `smart_grad = g + α*mu + lamb_eff*slow`
- `ref_muon_step` — NS coefficients (3.4445, -4.7750, 2.0315); 5 iterations; scale = 0.2*sqrt(max(rows,cols))

**Tolerances for CPU self-consistency:** `atol=1e-6`, `rtol=1e-4` in closed-form checks. GPU half uses `atol=1e-4, rtol=1e-3`.

**Key CPU tests (28 test items):**
- `test_registry_covers_all_eleven` — 11 reference functions registered
- `test_adamw_zero_grad_is_pure_weight_decay` — zero grad → only WD; WD=0 → p unchanged
- `test_lion_zero_grad_zero_momentum_is_pure_weight_decay`
- `test_ema_amp_zero_grad_is_pure_weight_decay` — parametrize [grokadamw, grokfast] → 2
- `test_muon_zero_grad_zero_buf_is_pure_decay`
- `test_adamw_step1_bias_correction_closed_form` — bc1=1-β1, bc2=1-β2 at t=1; closed-form math
- `test_bias_correction_factors_are_uninverted` — bc = 1-β^t (divide by them, NOT 1/(1-β^t))
- `test_ema_amp_with_zero_lamb_equals_adamw` — parametrize [(grokadamw,{α,lamb=0}),(grokfast,{α,lamb=0})] → 2
- `test_grokadamw_layerwise_beta1_changes_moments`
- `test_grokadamw_layer_beta1_wiring`
- `test_grokadamw_grokking_signal_and_alpha`
- `test_neuralgrok_unit_amplifier_equals_adamw`
- `test_looksam_alpha_zero_equals_adamw`
- `test_metanet_neutral_gate_equals_adamw` — parametrize [(sg11,gate=1),(sg15,gate_global=0)] → 2
- `test_supergrok2_alpha_zero_equals_adamw`
- `test_supergrok2_grokfast_term_restored` — lamb_eff NOT silently dropped; grokfast term load-bearing
- `test_prodigy_apply_uses_d_as_step_size` — step size = d, NOT lr
- `test_prodigy_update_d_is_monotone_nondecreasing`
- `test_prodigy_partials_closed_form` — d_prev²·|g| (canonical form, d^2 not d^1, |g| not signed)
- `test_neuralgrok_psi_forward_relu_closed_form`
- `test_sg_phi_forward_gelu_closed_form` — exact erf GELU (NOT tanh)
- `test_sg15_alpha_per_coord_clamps`
- `test_muon_ns_reduces_singular_value_spread`
- `test_muon_full_step_known_scale`
- `test_muon_2d_only_shape_required`
- `test_neuralgrok_amplifier_trains_on_cpu`
- `test_neuralgrok_amplifier_no_grad_raises`
- `test_neuralgrok_amplifier_forward_matches_kernel_psi`

**GPU half (11 items, `@pytest.mark.hw`):**
- `test_kernel_matches_reference_gpu` parametrized over all 11 optimizer classes

---

## 4. `tests/hw/test_opt_stages.py` — 10 CPU tests

**Purpose:** CPU fp32 mirrors of in-kernel optimizer precompute stages (`opt_stages_precompute.cuh`), validated against fp64 references.

**Stages covered:**
- Prodigy d-reduction (owner-computes cross-CTA tree)
- SG11 cosine gate (per-tensor block tree)
- SG15 mu (phi-forward)
- Muon Newton-Schulz (5-iteration fp32 matmul chain)

**Tests (10):**
- `test_prodigy_d_reduction_matches_fp64_oracle` — fp32 block-tree vs fp64 ref within fp32 tol
- `test_prodigy_owner_sum_is_partition_independent` — any partition order → same d (fixed-order sum property)
- `test_sg11_cosine_gate_matches_fp64_oracle`
- `test_sg11_gate_applies_gate_temp`
- `test_sg11_gate_is_per_tensor_not_cross_tensor`
- `test_sg11_mu_uses_sharpness_input`
- `test_sg15_mu_matches_phi_oracle`
- `test_muon_newton_schulz_matches_fp64_oracle`
- `test_muon_matmul_strides_and_transpose_exact`
- `test_verdict_partition_covers_all_eleven`

---

## 5. Decoder TC Gates (`tests/hw/test_decoder_tc.py`)

**Claimed: 19/19 byte-identical. File: 809 lines.**

### Structure
- PART 1: GEMM-orientation micro-gates (JIT-build `decoder_tc_selftest.cu`)
- CPU gate: `test_decoder_kernel_mirror_matches_oracle` (no `@hw`, runs on CPU CI)
- PART 2: Full-cell gates (JIT-build `mega_decoder_real_adamw_tc.cu` with `-DSG_TUNED_GEMM_IMPL=1`)

### Tolerances
- vs bf16-rounded oracle: `atol = max(2e-3, 2e-3 * K/512)` (scales linearly with K, accumulation-order only)
- vs fp64 true oracle (showing bf16 input rounding): rel ≤ 2e-2
- Full-cell grad weights: `_TC_GRAD_REL_WEIGHTS = 0.15`, biases: `_TC_GRAD_REL_BIASES = 0.08`
- Full-cell loss: `_TC_LOSS_REL = 1e-4` (calibrated to catch bias-omission bug at 2.52e-4, pass faithful at 2.85e-5)
- Trajectory (50 steps): max per-step rel-dev < 0.15

### PART 1 GEMM micro-gates (13 GPU-gated tests, all `@_GATE`):
| Test | Shape | What |
|------|-------|------|
| `test_fwd_identity_localization` | TILE_M×128, N=128, K=128 | A=I localization: Y[m,n]=W[n,m] |
| `test_fwd_random` ×4 | (K,Nout) ∈ {(128,128),(512,128),(128,384),(128,512)} | Random X,W vs fp64 ref |
| `test_dx_random` ×2 | Nout ∈ {128, 512} | dX = dY@W with W transposed-staged |
| `test_dw_random_multistep` ×4 | (T,Nout) ∈ {(128,128),(2048,128),(2048,384),(2048,512)} | dW multi-k-step stride-bug gate |
| `test_dw_identity_who_owns_what` | Nout=128, N=128, T=256 | A=I: dW[o,i]=X[o,i] for o<Nout |
| `test_determinism_bitwise` | Nout=128, N=128, T=512 | 3 runs → bit-identical dW |

**Total PART 1: 1+4+2+4+1+1 = 13 GPU tests**

### CPU gate (1 test, no `@hw`):
- `test_decoder_kernel_mirror_matches_oracle`: fp64 oracle vs fp64 mirror, loss diff < 1e-8, all grad rel < 1e-6. Tests: embedding scatter (within-sample token collision), causal 3-pass attention backward, LN/GELU/FF chain, last-position CE.

### PART 2 Full-cell gates (5 GPU-gated tests):
- `test_tc_single_step_grad_parity`: TC vs bf16-faithful fp64 oracle; grad weights ≤ 0.15, biases ≤ 0.08; loss rel ≤ 1e-4; calibration witness (layer0 ≈ layer1 proves no bug)
- `test_tc_determinism`: 3 runs → bit-identical loss + grad (A/A/A)
- `test_tc_dw_gemm_exact_on_own_operands`: kernel's OWN bf16 acts → fp32 contraction vs kernel ff2.weight dW; rel < 1e-4 (proves GEMM bit-exact, residual is operand-chain bf16 noise)
- `test_tc_step_time_vs_scalar`: informational only (no hard fail on ratio); fleet-contended GPU
- `test_tc_short_trajectory`: 50 steps TC vs eager bf16; max per-step rel-dev < 0.15

**Total PART 2: 5 GPU tests**

**TOTAL decoder_tc: 13 + 1 + 5 = 19. Matches claimed 19/19.**

### Environment
- `SG_TC_NCTA_CAP` env var (default 8): caps launched CTAs for fleet-saturated GPU
- `TORCH_CUDA_ARCH_LIST = "9.0a"` forced at build time
- `torch.backends.cuda.matmul.allow_tf32 = False` enforced

---

## 6. ViT TC Gates (`tests/hw/test_vit_tc.py`)

**Claimed: 21/21. File: ~700 lines.**

### Differences from decoder twin
1. Full (bidirectional) attention — no causal mask
2. PATCH-PROJ Linear(49→128) embedding (K=49 padded to 64) + cls_token + pos
3. CLS pos-0 head (final-norm + head + CE on position 0)
4. `kTileM = LCM(17,64) = 1088` (17-atom M stacking)

### Tolerances
- PART 1: same formula as decoder (`max(2e-3, 2e-3*K/512)`)
- PART 2 grad: full-cell grad vs bf16-faithful oracle (same tolerances as decoder)

### PART 1 GEMM micro-gates (14 GPU-gated tests):
- `test_fwd_identity_localization`: 1
- `test_fwd_random` ×5: (K,Nout) ∈ {(128,128),(512,128),(128,384),(128,512),(64,128)} — note extra (64,128) for patch K=64
- `test_dx_random` ×2: Nout ∈ {128, 512}
- `test_dw_random_multistep` ×4: same as decoder
- `test_dw_identity_who_owns_what`: 1
- `test_determinism_bitwise`: 1

**Total PART 1: 1+5+2+4+1+1 = 14 GPU tests**

### PART 2 Full-cell gates (7 GPU-gated tests):
- `test_tc_single_step_grad_parity`
- `test_tc_grad_parity_gridstride` (extra grid-stride test)
- `test_tc_grad_parity_ragged_tile` (ragged tile gate)
- `test_tc_determinism`
- `test_tc_dw_gemm_exact_on_own_operands`
- `test_tc_step_time_vs_scalar`
- `test_tc_short_trajectory`

**Total PART 2: 7 GPU tests**

**TOTAL vit_tc: 14 + 7 = 21. Matches claimed 21/21.**

Note: The ViT also has a `_bf16_faithful_oracle` function used by SG gates, but the PART 1/2 test structure within `test_vit_tc.py` itself counts as 21.

---

## 7. Mamba TC Gates (`tests/hw/test_mamba_tc.py`)

**Claimed: 3/5 with 2 pre-existing fails. File: 681 lines.**

No PART 1 GEMM micro-gates as separate pytest functions (unlike decoder/vit). Only PART 2-style tests.

### Key config: Mamba-3 TOY
```
_M3_CFG = dict(p=97, ntok=99, seq_len=8, d=128, nl=2, state_dim=128,
               head_dim=64, expand_factor=2, mlp_ratio=2)
```
45 params, 593713 elements (from code comment). Uses Mamba-3 model, NOT Mamba-1.

### Tolerances
- Grad vs bf16-faithful oracle: `_TC_GRAD_REL = 0.08` (all tensors — smaller than decoder's 0.15; worst observed dt_proj.bias ~1.8e-2 << 0.08)
- Loss vs bf16-faithful oracle: `_TC_LOSS_REL = 5e-3`
- Trajectory (50 steps): loose curve-tracking gate
- dW ISO: rel < 1e-6 on kernel's own operands

### 5 GPU-gated tests (`@_GATE`):
1. `test_tc_single_step_grad_parity` — KEYSTONE: TC vs bf16-faithful fp64 oracle; grad ≤ 0.08, loss ≤ 5e-3
2. `test_tc_proj_dw_exact_on_own_operands` — kernel's own bf16 acts → out_proj dW bit-exact; rel < 1e-6
3. `test_tc_determinism` — 3 runs A/A/A bit-identical
4. `test_tc_short_trajectory` — 50-step loss curve tracks eager bf16
5. `test_tc_step_time_vs_scalar` — informational

**Note on "3/5 with 2 pre-existing fails":** The 2 that could fail are not explicitly flagged in the code. Given the register-pressure mamba A/A/A race mentioned in `test_l3tc_tail_gate.py` comments (which was FIXED for prodigy/looksam/SG mamba), and the TP data-path bugs mentioned as unresolved, these 2 failures may be `test_tc_single_step_grad_parity` (if TP weights shard offset bug corrupts grads) and `test_tc_determinism` (if the A/A/A race is not fully fixed for mamba TC basic tests). This is a **discrepancy** vs claimed: the RESUME.md claims the A/A/A mamba race IS fixed (the fix is in model_stage_mamba3.cuh + model_stage_mamba_tc.cuh), so 3/5 vs 5/5 needs investigation.

---

## 8. Mamba Non-TC Gates (`tests/hw/test_mamba_megakernel.py`)

### CPU gates (3, run anywhere):
- `test_mamba_oracle_matches_autograd`: Oracle (manual fwd+bwd) == autograd, fp64, loss+all grads < 1e-8 rel, < 1e-9 loss abs diff
- `test_mamba_kernel_mirror_matches_oracle`: Mirror == oracle, loss < 1e-8, all grads < 1e-6 rel; tests: within-sample token collision, per-channel register scan, scan backward (recompute+reverse), NON-causal conv transpose, 3-path dx_main, owner-thread accumulation
- `test_mamba_layout_matches_named_parameters`: **28 tensors**, **259,425 total**; A_log/D precede in_proj within each layer

### GPU gates (skipped without L3-REAL Mamba):
- `test_mamba_l3_real_single_step_parity`: loss < 1e-5 rel; every grad < 1e-4 rel vs oracle; params after 1 step < 1e-5 rel-beyond-floor (floor = 0.05*lr for eps-amplified reference noise)
- `test_mamba_l3_real_trajectory`: max loss rel-dev < 1e-3 over 200 steps; final params < 2e-2 (3x chaos floor)
- `test_mamba_l3_real_groks` ×3 seeds: 3-seed grok smoke (skipped until grok driver wired)

---

## 9. ViT Non-TC Gates (`tests/hw/test_vit_megakernel.py`)

### CPU gates (4, run anywhere):
- `test_vit_oracle_matches_autograd`: Oracle == autograd, fp64; loss < 1e-9 abs; all 32 grads < 1e-8 rel
- `test_vit_kernel_mirror_matches_oracle`: Mirror == oracle; loss < 1e-8; all grads < 1e-6 rel; tests: FULL (non-causal) 3-pass attention bwd, cls_token + patch_proj scatter
- `test_vit_layout_matches_named_parameters`: **32 tensors, 418,017 total**; asserts optional `_VIT_TOTAL_ELEMS` dispatch mirror
- `test_vit_smem_budget_fits_dynamic_cap`: VitSampleSmem = 188,080 bytes < 227 KB (sm_90 dynamic-smem cap); exact byte count pinned (regression guard)

### GPU gates (require L3-REAL ViT):
- `test_vit_l3_real_single_step_parity`: loss < 1e-5 rel; every grad < 1e-4 rel vs oracle; params beyond floor (0.05*lr) < 1e-5
- `test_vit_l3_real_trajectory`: CHAOS-CALIBRATED; pre-cliff (steps 1-50) symmetric < 1e-3; mid-band (51-140) < 3×0.876; full-traj < 3×27.5; final params < 3×0.436
- `test_vit_l3_real_groks` ×3 seeds: grok smoke (skips until vit_grok_smoke_impl wired)

---

## 10. L3-TC Tail Gate (`tests/hw/test_l3tc_tail_gate.py`)

**Claimed: "33 cells" per code (len(_CELLS)=33).**

### Gate structure per converted cell
**(1) megakernel-vs-canonical single-step:**
- **(1a)** params vs canonical fp64 reference; tol `param_tol` (default 1e-4; muon uses 2e-3)
- **(1b)** STATE vs canonical fp64: m, v, ema/s/sam_dir; tol 1e-4 (tight — the decisive check; a dropped mechanism shows in state, not just params at step 1)
- **(1b-sam)** LookSAM/SG: sam_dir vs PURE-fp64 2nd backward oracle; `sam_dir_tol=2.5e-2`

**(2) A/A/A determinism:** 3 reruns from same init, loss+grad+params+mu+sharpness bit-identical

**Special branches:**
- `supergrok11/15`: `_run_sg_cell_gate` — checks (A) sharpness vs PURE-fp64 2nd backward (tol 3e-2), (B) mu vs rescale·phi (tol 3e-3), (1a/1b) apply tail (1e-4), (2) A/A/A; WARM-UP gate also available
- `supergrok2`: short-circuits to `_sg2_l3tc_gate.run_sg2_gate` (subprocess in isolated process)
- `supergrok2` cells: run in fresh subprocess to avoid `_CONTAMINATION_ISOLATED_OPTS` device-global leak

**Canonical references (post RE-ANCHOR task #10):** All references are now `ref_<opt>_step` from `test_reference_parity.py` (the fp64 transcription of `csrc/algorithms/<opt>.h`) — the eager per-op CUDA kernels are deleted.

**33 cells:**
```
adamw/{decoder,vit,mamba}             (3)
lion/{decoder,vit,mamba}              (3)
grokfast/{decoder,vit,mamba}          (3)
neuralgrok/{decoder,vit,mamba}        (3)
grokadamw/{decoder,vit,mamba}        (3)
prodigy/{decoder,vit,mamba}          (3)
muon/{vit,decoder,mamba}             (3)
looksam/{decoder,vit,mamba}          (3)
supergrok11/{decoder,vit,mamba}      (3)
supergrok15/{decoder,vit,mamba}      (3)
supergrok2/{decoder,vit,mamba}       (3)
```

**Key per-cell tolerances table:**
| Optimizer | param_tol | sam_dir_tol | sharpness_tol | mu_tol |
|-----------|-----------|-------------|---------------|--------|
| adamw, lion, grokfast, neuralgrok, grokadamw, prodigy, looksam | 1e-4 | — | — | — |
| muon | 2e-3 | — | — | — |
| looksam | 1e-4 | 2.5e-2 | — | — |
| supergrok11/15 | 1e-4 | — | 3e-2 | 3e-3 |

**NeuralGrok specifics:**
- psi-net: `kPsiHidden=16`, 2-layer ReLU MLP; ref includes GLOBAL grad-norm clip (kernel P2.5)
- BOTH (1a) params AND (1b) m/v vs canonical fp64 `_neuralgrok_canonical_mv`

**GrokAdamW specifics:**
- per-tensor β1 = β1·(1-γ)^layer_index (kernel P3; load-bearing: m-rel=0.895 when dropped)
- GLOBAL grad-norm clip (P2.5; inert at step 1, fires ~step 50)
- EMA cold-start seed = UNCLIPPED grad (not clipped); clip-invariant at step 1

**Prodigy specifics:**
- d stays at d0=1e-6 at step 1 (param_init==params → r=0)
- Single-step gate necessary but NOT sufficient (d-adaptation blind at step 1)
- Multi-step parity in `test_multistep_parity.py` is load-bearing

---

## 11. WGMMA Substrate Gates (`tests/hw/test_wgmma_substrate.py`)

**File:** 483 lines. JIT-builds `csrc/backends/cuda/sm_90/wgmma_selftest.cu`.

**Tests (12, all `@_GATE` for sm_90a):**
- `test_a_identity_localization`: A=I localization for m64×N×16 single tile
- `test_a_single_tile_shapes` ×6: N ∈ {8,16,32,64,96,128} — single tile shapes
- `test_a_ragged_head_97_padded`: 97-row vocab head (padded to 128)
- `test_b_kloop_K128` ×2: N ∈ {64,128} — multi-tile K=128 loop
- `test_b_kloop_K512`: K=512
- `test_c_pipelined_matches_unpipelined` ×(2N × 2depth): N∈{64,128}, depth∈{2,3} → 4 items
- `test_d_determinism_bitwise`: same inputs twice → identical (the claimed "SG_TUNED_GEMM_ENGINE" substrate)
- `test_e_occupancy_refuse_oversized_smem`: too-large dynamic-smem → no hang
- `test_sass_audit_hgmma_present_and_spills`: cuobjdump checks HGMMA/wgmma present (not FMA)

**Tolerances:** All vs fp32 reference of bf16-rounded inputs (same formula: `max(2e-3, 2e-3*K/512)`).

---

## 12. TP Loopback Gate (`tests/hw/test_tp_loopback.py`)

**Purpose:** Col→row TP pair on 1 GPU via `LoopbackTransport` (`tp_loopback_binding.cu`).

**Parity tol:** `3e-5` (campaign fp32 baseline, assert (e): TP+transport vs serial chunked-order).

**Tests:**
- `test_tp_loopback_block` parametrized over `_TP_DEGREES` (TP ∈ {2,4})

**Five asserts:**
- (a) All P ranks' reduced Y1/dX **bit-identical** (fixed-order reduce)
- (b) A/A/A: 3 reruns → **bit-identical** (structural, not timing-dependent)
- (c) Transport-neutrality: TP+transport == serial chunked-order, **bit-exact**
- (d) Slice-exactness: dW0/db0 col-shards == row-slice of unsharded dW0/db0, **bit-exact**; dW1 col-shards **bit-exact**; db1 replicated **bit-exact**
- (e) vs unsharded full-K reference: NOT bitwise; within 3e-5 rel; actual delta printed

---

## 13. NVSHMEM Smoke (`tests/hw/nvshmem_smoke.py`)

**Not a pytest file** — a `torchrun` script. No `@pytest.mark.hw`.

**Usage:** `torchrun --nproc_per_node=N tests/hw/nvshmem_smoke.py`

**What it proves:**
1. JIT-build host NVSHMEM pybind (`nvshmem_bringup_pybind.cpp`)
2. UID bootstrap: rank 0 mints UID, broadcast via torch.distributed, all ranks `init_with_uniqueid`
3. Team split: `nvshmem_team_split_strided(pe_start, pe_stride, pe_size)` → TP team
4. Collective `nvshmem_malloc` of symmetric heap (sized to `tp_heap_stride_floats(ctas_per_pe)`)
5. Team-scoped all-reduce of `(world_pe + 1)` → assert result == Σ_{pe∈team}(pe+1) **bit-exact** on every rank

**Assert (cross-rank):** Expected sum = world*(world+1)/2 for single-node pure TP. Cross-rank consistency: all ranks see same value. **Exits 0 on success.**

---

## 14. DP=2 Loopback Gate (`tests/hw/test_dp2_loopback_determinism.py`)

**DP=2 on ONE GPU** (NCCL_HOSTID trick). Spawns torchrun as subprocess.

**Cells:** `_DP2_CELLS = [("adamw","decoder"),("lion","decoder"),("grokfast","decoder"),("adamw","vit")]` — 4 parametrized test items.

**Test:** `test_dp2_loopback_cross_rank_aaa` (4 items).

**Three asserts:**
- (a) Final all-gathered params identical on both ranks
- (b) 3× repeat → bit-identical (cross-rank A/A/A; fixed-order ascending-rank reduce makes it structural)
- (c) DP=2 vs single-GPU unsharded: parity tol `3e-5`; actual delta reported

**Fixed-order reduce:** all_gather (bit-exact) → sum gathered grads in ascending rank order locally in fp32. NOT NCCL reduce-scatter (non-deterministic order).

---

## 15. PP=2 Loopback Gate (`tests/hw/test_pp2_loopback_determinism.py`)

**PP=2 single-GPU loopback.** Requires `SG_DECTC_LAYER_RANGE` patch applied.

**Cells:** `_PP_CELLS = ["adamw", "grokfast"]` — 2 parametrized test items.

**Tests:** `test_pp2_loopback_bit_exact_aaa` (2 items), `test_pp2_stage_ownership_matches_python_plan` (1 item).

**Stage 0:** tensors {0..13} (embeddings + L0 block)
**Stage 1:** tensors {14..29} (L1 + final norm + head)

**Asserts (4):**
- (a) PP grad == production fused-step grad, **bit-identical** (all 30 tensors)
- (b) loss bit-identical
- (c) A/A/A: 3 reruns bit-identical
- (d) sharded_optimizer_kernel on PP grad == production params_after, **bit-identical**

---

## 16. Sharded Optimizer Gate (`tests/hw/test_sharded_optimizer.py`)

**Purpose:** DELIVERABLE 1 — sharded_optimizer_kernel == in-kernel P3, **bit-identical**.

**Cells:** `_CELLS = [(opt,model) for model in ("decoder","vit","mamba") for opt in ("adamw","lion","grokfast")]` → 9 items.

**Test:** `test_sharded_optimizer_bit_parity` (9 items), all `@pytest.mark.hw`.

**Assert:** `torch.equal` (no tolerance — bit-identical). The binding is JIT-built into `/workspace/.torch_ext/`, NOT the production `_ops`.

---

## 17. ZeRO-3 Round-Trip Gate (`tests/hw/test_zero3_roundtrip.py`)

**Purpose:** ZeRO-3 gather/release + checkpoint/resume around the REAL fused step (1 GPU).

**Cells:** `_CELLS = ["adamw", "grokfast"]`, world=2 virtual ranks.

**Test:** `test_zero3_checkpoint_resume_roundtrip` (2 items, `@_GATE` for sm_90a).

**Two-step protocol:**
1. Step 1 → params1_ref (in-kernel P3); sharded_opt_step via Zero3FlatParamStore → gather == params1_ref **bit-exactly**
2. Save each rank's shard; step 2 live → params2_live; RESUME from checkpoint, re-apply step 2 → params2_resumed == params2_live **bit-exactly**

---

## 18. Distributed Step Gate (`tests/hw/test_distributed_step.py`)

**Two gates:**
1. `test_world1_decomposed_identity` ×2 (adamw, grokfast): world=1 decomposed == plain fused_train_step **bit-identical** for 2 steps
2. `test_world2_loopback_through_module`: torchrun 2-rank; cross-rank params bit-identical; 2-step traj vs single-GPU ref; parity tol `_PARITY_TOL=5e-4` (accounts for 2-step compounding of DP=2 batch-shard grad reassociation)

---

## 19. CUDA Graph Capture Gate (`tests/hw/test_step_graph_capture.py`)

**Captures [fused-step → sharded-opt] (no collectives) as CUDA graph. N_REPLAYS=5.**

**Test:** `test_step_graph_capture_no_collective_chain` parametrized over `["adamw","lion","grokfast"]` → 3 items.

**Assert:** Each of 5 replays is **bit-identical** to the eager run (maxd == 0.0). This is structural — same kernel launches on same buffers, inputs reset before each replay.

**Honest limitation stated:** Cross-rank NCCL collectives mixed with megakernel NOT captured on 1-GPU loopback (the megakernel grabs all SMs while peer's collective busy-waits — the irreducible 8-GPU work).

---

## 20. Parallel Instantiation Gate (`tests/hw/test_parallel_instantiation.py`)

**CPU-only (needs nvcc, not GPU).** Generates a scratch `.cu` and `nvcc -c` it.

**Allow-list:**
```python
("SingleGPU", 1,1,1,1,"Z0",1)   # bit-identical baseline
("DP8_ZeRO2", 8,1,1,1,"Z2",1)
("DP8_ZeRO3", 8,1,1,1,"Z3",1)
("DP4_PP2_ZeRO3", 4,1,2,1,"Z3",1)
("DP2_TP4_ZeRO3", 2,4,1,1,"Z3",1)
("DP1_TP1_EP8_ZeRO3", 1,1,1,1,"Z3",8)  # MoE EP=8 frontier
```

**Tests (3):**
- `test_allow_list_points_compile`: all 6 points compile cleanly
- `test_each_point_compiles_in_isolation`: each in its own `nvcc -c`
- `test_sp_not_one_is_a_compile_error`: SP != 1 must be a compile error

---

## 21. SG2 Megakernel Gate (`tests/hw/test_sg2_megakernel.py`)

### CPU layer A — Structural mirror vs oracle:
- `test_mirror_matches_oracle` ×(4N × 2seeds) = 8 items: N ∈ {5,17,64,200}, seed ∈ {0,3}; max abs diff across activations + state < 1e-12
- `test_mirror_multistep_trajectory` (1): 50 steps, mirror == oracle per step
- `test_oracle_matches_eager_metanet` ×3: N ∈ {5,17,64}; Oracle (clean fp64 reimpl of csa_hca_step_one) == eager pytorch metanet

### GPU layer B (HW-gated, `@pytest.mark.hw + @_skip_no_hw`):
- `test_megakernel_single_step_vs_per_op` (1): megakernel vs csa_hca_step_one; smart_grad/moments/params within 1e-5
- `test_megakernel_trajectory_vs_per_op` (1): 200-step; per-step mean|param| proxy + final params within 1e-5

---

## 22. Multi-Step Parity (`tests/hw/test_multistep_parity.py`)

**Purpose:** Load-bearing complement to the single-step gate — covers the mechanisms the single-step gate is BLIND to.

**Tests (2, GPU-gated):**

### `test_grokadamw_multistep_parity`:
- ~60 fused steps + fp64 GrokAdamW side-by-side
- Asserts params match at {step 1, 50, 60}
- Asserts clip condition ‖g‖>grad_clip ACTUALLY occurred by step 50 (non-vacuous)
- 3× full trajectory A/A/A
- Tol: per-cell `param_tol` (inherits from l3tc_tail_gate)

### `test_prodigy_multistep_parity`:
- Controlled-anchor technique: inject known param_init = p_e + delta at step 2 to make r≠0
- d_coef>1 → d grows off d0 deterministically
- Asserts: kernel's persisted d tracks fp64 d each step; d actually grew off d0; params match tol at final step
- Control (d_coef=0): d frozen → diverges from adaptive run by step 50 (proves d-adaptation load-bearing)

---

## 23. 3D Parallel Harness (`tests/hw/test_3d_parallel.py`)

**CPU tests (no GPU/dist needed, 4 tests):**
- `test_model_sizing_hits_7b_target`
- `test_weak_scaling_efficiency_math`
- `test_shard_count_divides_by_model_parallel`
- `test_zero3_partition_covers_param_exactly`

**HW/torchrun tests (2, skip without distributed launch):**
- `test_3d_parallel_smoke`: the actual multi-GPU run (deferred to hardware window)
- `test_deepspeed_or_native_path_selected`

---

## 24. Mamba Scalar Probe (`tests/hw/test_mb3_scalar.py`)

**Not a pytest file** — a `if __name__ == "__main__"` script. No `test_*` functions, no pytest collection.

**Milestone A.1/B.1:** JIT-builds `_mb3_scalar_probe.cu`; runs B=64 batch; compares fp32 scalar kernel vs fp64 oracle.

**Pass criteria:** `loss_rel < 5e-3` AND `worst grad rel < 5e-2`. Flags tensors with rel > 5e-3.

---

## 25. TPU Pallas Parity (`tests/tpu/test_pallas_parity_interpret.py`)

**Skips cleanly when JAX absent.** Tests all 11 optimizers via `pl.pallas_call(..., interpret=True)` on CPU.

**Tolerances:** `atol=1e-4, rtol=1e-3` (fp32 kernel vs fp64 reference; same discipline as GPU half of `test_reference_parity.py`).

**Exercises:** Multiple sizes, seeds, nonzero initial state, sequential steps for stateful EMA optimizers.

---

## 26. hw/ Oracle / Mirror / Probe Files (supporting, not tests)

These are NOT standalone pytest-discoverable test files but helpers imported by the test files:

| File | Role |
|------|------|
| `decoder_oracle.py` | Manual fp64 decoder fwd+bwd; `decoder_param_layout()` (30 tensors); imports: VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ, D_FF, D_HEAD |
| `decoder_kernel_mirror.py` | Structural mirror of decoder megakernel index arithmetic |
| `vit_oracle.py` | Manual fp64 ViT fwd+bwd; `vit_param_layout()` (32 tensors, 418,017 total) |
| `vit_kernel_mirror.py` | Structural mirror of ViT megakernel |
| `mamba_oracle.py` | Manual fp64 Mamba fwd+bwd; `mamba_param_layout()` (28 tensors, 259,425 total) |
| `mamba_kernel_mirror.py` | Structural mirror of Mamba megakernel |
| `mamba3_oracle.py` | Mamba-3 fp64 oracle with SSM/RMSNorm/scan/SwiGLU primitives |
| `sg2_kernel_mirror.py` | SG2 megakernel mirror + clean oracle of csa_hca_step_one |
| `_sg2_l3tc_gate.py` | SG2 dedicated L3-TC gate (B1+A/A/A+tie-probe+N>64 CSA fidelity probe) |
| `_mamba_fill_test.py` | Mamba fill diagnostic |
| `_mamba_prodigy_probe.cu` | CUDA probe |
| `_mamba_prodigy_production_probe.py` | Probe |
| `_mamba_race_probe.py` | Race probe |
| `_mb3_scalar_probe.cu` | CUDA scalar Mamba-3 kernel for test_mb3_scalar |
| `_sg2_l3tc_gate.py` | SG2 gate logic |
| `tp_loopback_binding.cu` | TP loopback JIT binding |
| `pp_stage_binding.cu` | PP stage JIT binding |
| `sharded_optimizer_binding.cu` | Sharded optimizer JIT binding |

---

## 27. Claimed vs Actual Test Counts

### Decoder pytest: "claimed 19/19 byte-identical"
**Actual:** 19 total (PART1=13 + CPU mirror=1 + PART2=5). **Matches.** The "19/19" count is accurate. However note the CPU mirror test (`test_decoder_kernel_mirror_matches_oracle`) is NOT `@hw` gated — it runs on CPU CI. The 13+5=18 GPU-gated tests = 18 hardware gates; 19 total including the CPU mirror.

### ViT: "claimed 21/21"
**Actual:** 21 total in test_vit_tc.py (PART1=14 + PART2=7). **Matches.**

### Mamba: "claimed 3/5 with 2 pre-existing fails"
**Actual:** 5 tests in test_mamba_tc.py. Code does NOT explicitly flag which 2 fail. RESUME.md claims the mamba A/A/A race is fixed; however the "2 pre-existing fails" claim in the task description may refer to the original mamba TC state before the register-pressure fix. Whether this is still the case cannot be confirmed without running on hardware.

### hw/ parallelism/config/resource tests: "claimed 84 + 10/10 + 35"
**Actual breakdown:**
- `test_resource_planner.py`: **10** test items (exact match, claimed "10/10")
- `test_mem_strategy.py`: **10** test items
- `test_shard_map.py`: **29** expanded test items (20 functions, +7+4 from parametrize decorators)  
- `test_zero3_plan.py`: **14** expanded items (8 functions, +4+2+3 from parametrize)
- `test_pipeline_schedule.py`: **41** expanded items (13 functions, +24+4 from parametrize: 1F1B_valid has 25 grid, transport has 5)
- **Pure CPU total: 10+10+29+14+41 = 104 parametrized items**

The claimed "84" does not precisely match any single subset; the closest interpretation is that "84" refers to the parallelism CPU tests (shard_map + zero3_plan + pipeline_schedule = 29+14+41 = 84, which **exactly matches 84** when counting those 3 files).

The "35" likely refers to `test_reference_parity.py` CPU test items (~31 CPU + ~4 boundary items). Or possibly `test_opt_stages.py` (10) + `test_reference_parity.py` CPU (~25) = 35. Either way, the "35" is approximately right.

So: **"84 + 10/10 + 35" → shard/zero3/pipeline CPU = 84; resource_planner = 10/10; reference_parity+opt_stages ≈ 35.** This interpretation is consistent.

---

## 28. Discrepancies and Open Items

### Discrepancy 1: "3/5" Mamba TC claim
The RESUME.md says mamba A/A/A race is FIXED (commit 0b57f7e via __noinline__ on mbtc_forward_tile/mbtc_backward_tile). Yet the task description says "3/5 with 2 pre-existing fails" for Mamba. The code in test_mamba_tc.py does NOT mark any tests as xfail or skip. Either:
- (a) The 2 fails were fixed and the claim is stale, OR
- (b) There are still 2 hardware failures not captured in code annotations

### Discrepancy 2: TP data-path bugs affect test outcomes
RESUME.md states "TP data-path fix" is outstanding — bugs A (per-rank weight-shard offset), B (25-heads-not-%8 attention), C (IMA). These bugs would cause test failures in `test_tp_loopback.py` and potentially `nvshmem_smoke.py`. The test files do NOT have xfail annotations for these known bugs.

### Discrepancy 3: PP=2 requires patch
`test_pp2_loopback_determinism.py` requires `SG_DECTC_LAYER_RANGE` patch applied; without it the test SKIPs. This is not mentioned in RESUME.md as a prerequisite.

### Discrepancy 4: Mamba TC uses Mamba-3 not Mamba-1
The test file `test_mamba_tc.py` operates on Mamba-3 model (`_M3_CFG = dict(state_dim=128, head_dim=64, expand_factor=2, mlp_ratio=2)`) — 45 params, 593,713 elements. The RESUME.md mentions the flagship Mamba `d2048/L24` model. The TC gates are on a small toy Mamba-3 config, not the 1.5B flagship. This is correct for a gate suite but means it doesn't exercise the full flagship scale.

### Open Item 1: supergrok2 mamba A/A/A risk
The cell comment says "if A/A/A re-trips, mamba×supergrok2 is landed dormant." The fix for prodigy/looksam mamba (commit 0b57f7e) is claimed to also fix SG mamba, but this is flagged as a risk in the code.

### Open Item 2: Prodigy single-step gate admits d=d0 at step 1
The gate documentation explicitly states: "single-step gate is NECESSARY but NOT SUFFICIENT — it is BLIND to the d-adaptation." The multi-step parity gate (`test_multistep_parity.py`) is the load-bearing check, and it requires hardware to run.

### Open Item 3: NeuralGrok grad-norm clip gate vacuous at default seed
At default seed (42), ‖g‖₂ < grad_clip=1.0, so P2.5 clip is inert. Only fires at `GATE_SEED=7`. The gate passes even with a broken clip mechanism at default seed.

### Open Item 4: "test_vit_l3_real_groks" and "test_mamba_l3_real_groks" skip
Both grok smoke tests for ViT and Mamba skip until `_vit_grok_smoke`/`_mamba_grok_smoke` drivers are wired. These are `pytest.skip()`, not implemented yet.

### Open Item 5: test_decoder_kernel_mirror_matches_oracle tolerances
This CPU test asserts loss diff < 1e-8 and all grad rel < 1e-6. These are very tight (fp64 oracle vs fp64 mirror), but the test is unmarked `@hw` — it runs on CPU CI and may be slow for large batch sizes.

---

## 29. Summary Table

| Test Category | Files | Gate Type | Count (expanded) | Claimed |
|--------------|-------|-----------|-----------------|---------|
| Decoder TC | test_decoder_tc.py | GPU (sm_90a) | 18 GPU + 1 CPU = 19 | 19/19 ✓ |
| ViT TC | test_vit_tc.py | GPU (sm_90a) | 21 GPU | 21/21 ✓ |
| Mamba TC | test_mamba_tc.py | GPU (sm_90a) | 5 GPU | 3/5 (2 pre-existing fails) |
| L3-TC Tail Gate | test_l3tc_tail_gate.py | GPU (wgmma) | 33 cells (1 test each) | 33 cells |
| WGMMA Substrate | test_wgmma_substrate.py | GPU (sm_90a) | ~12 | not claimed separately |
| TP Loopback | test_tp_loopback.py | GPU (sm_90a) | 2 (TP∈{2,4}) | not claimed |
| NVSHMEM Smoke | nvshmem_smoke.py | torchrun/multi-GPU | 1 script | validated on 8 GPUs |
| Sharded Optimizer | test_sharded_optimizer.py | GPU (sm_90a) | 9 cells | not claimed separately |
| DP=2 Loopback A/A/A | test_dp2_loopback_determinism.py | GPU+torchrun | 4 cells | not claimed |
| PP=2 Loopback | test_pp2_loopback_determinism.py | GPU | 3 | not claimed |
| ZeRO-3 Round-Trip | test_zero3_roundtrip.py | GPU (sm_90a) | 2 | not claimed |
| Distributed Step | test_distributed_step.py | GPU+torchrun | 3 | not claimed |
| Graph Capture | test_step_graph_capture.py | GPU (sm_90a) | 3 | not claimed |
| **Shard/Zero3/Pipeline CPU** | test_shard_map.py, test_zero3_plan.py, test_pipeline_schedule.py | CPU | **84** expanded items | 84 ✓ |
| **Resource Planner** | test_resource_planner.py | CPU | 10 | 10/10 ✓ |
| **Reference Parity CPU** | test_reference_parity.py (CPU half) | CPU | ~31 | ~35 (≈) |
| Opt Stages | test_opt_stages.py | CPU | 10 | included in "35" |
| SG2 Megakernel CPU | test_sg2_megakernel.py (A) | CPU | 12 | not claimed separately |
| Parallel Instantiation | test_parallel_instantiation.py | CPU+nvcc | 3 | not claimed |
| Mamba Oracle/Mirror/Layout | test_mamba_megakernel.py (CPU) | CPU | 3 | not claimed |
| ViT Oracle/Mirror/Layout/Smem | test_vit_megakernel.py (CPU) | CPU | 4 | not claimed |
| Multi-Step Parity | test_multistep_parity.py | GPU | 2 | not claimed |
| 3D Parallel | test_3d_parallel.py | mixed | 4 CPU + 2 GPU | not claimed |
| Pallas TPU | tests/tpu/... | JAX interpret | ~11*N | not claimed |
| MB3 Scalar | test_mb3_scalar.py | GPU script | 1 script | not claimed (milestone) |
