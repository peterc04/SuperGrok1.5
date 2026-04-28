# REFRESH.md — SuperGrok1.5 Reference

A compact, granular catch-up. Plain language. Each kernel, file, and optimizer gets its own entry. No fluff, no withholding.

---

## Contents

1. Repo layout
2. Project state
3. Optimizers
4. Python infrastructure
5. csrc/common — shared headers
6. csrc/cuda/generic — generic kernels
7. csrc/cuda/sm_80 — Ampere
8. csrc/cuda/sm_90 — Hopper
9. csrc/cuda/sm_100 — Blackwell
10. csrc/hip — AMD ROCm
11. csrc/quantization — quantization kernels
12. Algorithms
13. JAX/TPU
14. Tests
15. Benchmarks
16. Codegen
17. Build
18. Recent commits
19. Known gaps
20. Quick reference

---

## 1. Repo layout

- `grokking_optimizers/` — Python package, eleven optimizers plus infra
- `supergrok2_jax_tpu/` — JAX/TPU port of the suite
- `csrc/common/` — shared C++/CUDA/HIP headers and dispatch
- `csrc/cuda/generic/` — kernels that compile under both CUDA and HIP
- `csrc/cuda/sm_80/` `sm_90/` `sm_100/` — NVIDIA tier-specific kernels
- `csrc/hip/cdna2/` `cdna3/` `cdna4/` — AMD tier-specific kernels
- `csrc/quantization/` — quantization kernels (FP8, INT8, INT4, MXFP4)
- `csrc/cpu/` — CPU fallback with AVX-512 / NEON SIMD
- `tests/` — eight test files
- `benchmarks/` — three benchmark scripts
- `codegen/` — kernel generator scripts plus YAML spec
- `setup.py` — build entry, detects backend
- `README.md` — user docs
- `ANALYSIS.md` — internal review with bug findings and optimization opportunities
- `REFRESH.md` — this file

## 2. Project state

- Branch: `claude/custom-optimizer-analysis-HFYhg`
- Working tree: clean
- Size: ~60k LOC of C++/CUDA/HIP across 98 files
- Backends supported: NVIDIA sm_70 → sm_100, AMD gfx908 → gfx950, CPU x86_64/ARM64, TPU v3 → v6e
- Status: production-ready, no known correctness blockers
- Recent focus: bug fixing, hot-path optimization, architecture coverage
- Architecture: SuperGrok v2 design settled long ago; no structural changes recently
- Last 5 commits, newest first:
  - `ea968b6` — sweeping bug-fix and optimization pass
  - `a6323c9` — fix `_single_param_step` in muon, prodigy, grokadamw
  - `6c48166` — fix AMD gcnArchName parsing for 3-digit codes
  - `dbe3ef4` — 9-bug fix pass with FP32 skip optimization
  - `1d930db` — wire dead fused kernels, kill Python AdamW bottleneck

## 3. Optimizers

Eleven total. Each entry: purpose, state per param, hyperparameters with defaults, fused kernel name, Python fallback availability.

### 3.1 SuperGrok v2 (`supergrok2.py`)

- Purpose: flagship. Mamba-3 + 4-head PEER + per-element GRU + 144-expert pool, per-element learned gradient correction, on top of Adam with SAM and bilevel meta-learning.
- State per param: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`, `gru_states[N, gru_hidden]`, `mamba_fwd_states[d_inner, d_state]`, `mamba_bwd_states[d_inner, d_state]`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1.0, alpha_init=0.98, lamb=2.0, kappa=0.1, gradient_clipping=1.0, d_model=8, d_state=16, mamba_expand=2, num_peer_heads=4, num_experts=144, expert_hidden=16, gru_hidden=4, meta_rescale=0.1, recycle_interval=100, recycle_threshold=0.001, sam_rho=0.05, projection_precision='auto', state_precision='fp32' or 'config3'
- Fused kernel: `_ops.supergrok2_prepare_and_batched_step`
- Bilevel kernel: `_ops.supergrok2_bilevel_fwd_save_batched` + `_ops.supergrok2_bilevel_backward`
- Python fallback: yes, full
- Distributed: meta-grad allreduce, expert count allreduce, mamba state broadcast from rank 0
- FSDP: meta-net excluded from sharding via `exclude_meta_net_from_fsdp`
- Compilable: `CompiledSuperGrok2` wrapper for CUDA graph capture

### 3.2 SuperGrok v1.5 (`supergrok15.py`)

- Purpose: simpler v2. Replaces Mamba+PEER+GRU with a 2-input 2-layer MLP.
- State per param: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=0.98, lamb=2.0, kappa=0.1, sam_rho=0.05, hidden_dim=32 (also 16/64/128 specialized)
- Fused kernel: `supergrok15_fused_step`
- Python fallback: no
- Special: register-resident smart_grad in fused full-step kernel

### 3.3 SuperGrok v1.1 (`supergrok11.py`)

- Purpose: v1.5 with cosine-similarity gating instead of sigmoid-on-accuracy.
- State per param: same as v1.5
- Hyperparameters: same as v1.5, plus gate_temperature=5.0, meta_update_freq=5
- Fused kernel: `supergrok11_fused_step`
- Reduction kernel: `cosine_gate_reduce_kernel` — fused dot/norm/norm reduction
- Python fallback: no

### 3.4 GrokAdamW (`grokadamw.py`)

- Purpose: AdamW with EMA gradient filter and persistent-direction amplification.
- State per param: `exp_avg`, `exp_avg_sq`, `ema`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=0.98, lamb=5.0, grad_clip=1.0
- Fused kernel: `grokadamw_fused_step`
- Quantized variant: `_q3` kernel — INT8 per-block exp_avg + BF16 stochastic-rounded exp_avg_sq + ema
- Python fallback: no (CPU build has C++ implementation)

### 3.5 NeuralGrok (`neuralgrok.py`)

- Purpose: AdamW with learned MLP amplifier on |grad|.
- State per param: `exp_avg`, `exp_avg_sq`
- Amplifier: 2- or 3-layer MLP, input |grad|, output multiplicative scale
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, alpha=10.0, beta=4.0, num_layers=3, hidden_dim=128, inner_steps=1
- Fused kernel: `neuralgrok_fused_step`
- Python fallback: no

### 3.6 Prodigy (`prodigy.py`)

- Purpose: self-tuning Adam. Estimates `d_lr` from cumulative parameter-space distance. Set lr=1.0 and let it auto-tune.
- State per param: `exp_avg`, `exp_avg_sq`, `s`, `param_init`
- Hyperparameters: lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1.0
- Fused kernel: `prodigy_fused_step`
- Reduction kernel: `prodigy_dlr_reduce_kernel` — computes new `d_lr` via global reduction
- Python fallback: no

### 3.7 Grokfast (`grokfast.py`)

- Purpose: simplest grokking-aware AdamW. EMA + amplification.
- State per param: `ema`, `exp_avg`, `exp_avg_sq`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, grokfast_alpha=0.98, grokfast_lamb=2.0
- Fused kernel: `grokfast_fused_ema_adam_step`
- Python fallback: no

### 3.8 Lion (`lion.py`)

- Purpose: sign-based Adam alternative (EvoLved Sign Momentum).
- State per param: `exp_avg` (momentum buffer)
- Hyperparameters: lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0
- Fused kernel: `lion_fused_step`
- Multi-tensor variant: yes — fuses many small params into one launch
- Python fallback: yes, in CPU C++

### 3.9 LookSAM (`looksam.py`)

- Purpose: AdamW with periodic SAM (every k steps) instead of every-step SAM.
- State per param: `exp_avg`, `exp_avg_sq`, `sam_direction`
- Hyperparameters: lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0, rho=0.05, k=5, alpha=0.7
- Fused kernel: `fused_adamw_simple_step` (regular), manual SAM step
- Python fallback: yes, in CPU C++

### 3.10 Muon (`muon.py`)

- Purpose: dual optimizer. Newton-Schulz orthogonalization for 2D weights, AdamW for 1D.
- State (2D): `momentum_buffer`
- State (1D): `exp_avg`, `exp_avg_sq`
- Hyperparameters (2D): lr=0.02, momentum=0.95, weight_decay=1.0, ns_steps=5
- Hyperparameters (1D): adamw_lr=1e-3, adamw_betas=(0.9, 0.98), adamw_eps=1e-8
- Fused kernels: `muon_fused_step` (2D), `fused_adamw_simple_step` (1D)
- Python fallback: yes, in CPU C++

### 3.11 Mamba3PEERMetaNet (`mamba3_peer_metanet.py`)

- Purpose: meta-net used internally by SuperGrok v2; not a standalone optimizer.
- Submodules: `Mamba3ScanBlock`, `MiniGRU`, PEER router (in same file), expert MLP pool
- Trained by SuperGrok v2's bilevel update
- Has full pure-PyTorch CPU fallback path

## 4. Python infrastructure

### `dispatch.py`
- Detects backend at runtime, no GPU import required.
- `get_gpu_vendor()` → 'nvidia' | 'amd' | 'none'
- `get_gpu_arch()` → SM number (NVIDIA) or CDNA arch (AMD)
- `get_backend()` → 'cuda' | 'hip' | 'cpu'
- `get_warp_size()` → 32 (NVIDIA) or 64 (AMD CDNA)
- `get_arch_tier()` → 'blackwell'|'hopper'|'ampere'|'generic'
- `get_amd_tier()` → 'cdna4'|'cdna3'|'cdna2'|'generic'
- `supports_bf16/fp8/tf32/tma/block_clusters/matrix_cores/nvfp4` predicates
- Env override: `FORCE_ARCH=N`

### `quantization.py`
- `PrecisionConfig` class with three knobs:
  - projection precision: `'fp32'|'tf32'|'bf16'|'fp8'|'mxfp4'|'nvfp4'|'auto'`
  - expert precision: `'fp32'|'int8'|'int4'|'auto'`
  - state precision: always FP32
- `convert_projection_weights(w)` → (quantized, scale)
- `convert_expert_weights(w1, b1, w2, b2)` → dict with mode + tensors
- Auto chain: nvfp4 → mxfp4 → fp8 → bf16 → fp32
- Optional dynamic-precision mode that lowers precision as training stabilizes

### `cuda_graph_optimizer.py`
- `CUDAGraphOptimizer(opt, warmup_steps=3)` — wraps any optimizer
- `CompiledSuperGrok2` — SuperGrok v2 specialization
- Records first non-warmup step as graph, replays after
- Auto-invalidates when kwargs passed to step()
- `invalidate()` method for manual reset
- ~2-3× speedup small models, ~1.5× large

### `distributed.py`
- `setup_distributed(backend='nccl')` — torchrun-style init
- `cleanup_distributed()`, `get_rank()`, `get_world_size()`, `is_main_process()`
- `broadcast_optimizer_state(opt, src=0)` — align ranks
- `wrap_model_ddp(model)` — DDP wrapper
- SuperGrok v2 private helpers:
  - `_is_distributed()`
  - `_allreduce_meta_grads()` — sum + divide by world size
  - `_allreduce_expert_counts()` — for recycling consistency
  - `_sync_mamba_states()` — broadcast from rank 0
  - `_gather_full_grad_fsdp()` — context manager for FSDP
  - `exclude_meta_net_from_fsdp(meta_net)` — keep meta-net replicated

### `jit/` directory
- Optional runtime kernel specialization, cached in `~/.cache/supergrok2/`
- `specializer.py` — base class + `ModelConfig`
- `cuda_specializer.py`, `hip_specializer.py`, `tpu_specializer.py`, `cpu_specializer.py`
- `smem_layout.py` — shared memory layout optimization
- `block_size_optimizer.py` — tile size selection
- `gcn_scheduler.py` — AMD GCN wavefront scheduling
- `ptx_scheduler.py` — NVIDIA PTX instruction scheduling
- Falls back to pre-compiled `_ops` if anything fails

### `__init__.py`
- Exports all eleven optimizers
- Meta-net classes: Mamba3PEERMetaNet, Mamba3ScanBlock, MiniGRU, SharpnessMetaNet
- Wrappers: CompiledSuperGrok2, CUDAGraphOptimizer, OverlappedOptimizer, PipelinedOptimizer, GradientHookOptimizer, AsyncSuperGrok2, MoEAwareSuperGrok2
- Distributed helpers, dispatch helpers, PrecisionConfig
- Flags: `_HAS_OPS`, `_HAS_CUDA`, `_HAS_CPU_OPS`

## 5. csrc/common — shared headers

### `platform.h`
- Single-source CUDA/HIP abstraction.
- Sets `GROK_CUDA=1` or `GROK_HIP=1`.
- Warp size: 32 (CUDA), 64 on CDNA via `__AMDGCN_WAVEFRONT_SIZE__`.
- Macros remap intrinsics:
  - `SHFL_DOWN` → `__shfl_down_sync` (CUDA, with masks) / `__shfl_down` (HIP)
  - `FAST_SINCOSF` → `__sincosf` (CUDA) / `sincosf` (HIP)
  - `LDG` → `__ldg` (CUDA, read-only cache) / `*ptr` (HIP)
  - `FULL_WARP_MASK` → `0xFFFFFFFF` (CUDA) / `0` (HIP)
- Stream type alias: `GpuStream_t`
- Error checking: `gpuGetLastError`, `gpuDeviceSynchronize`, `gpuGetDeviceProperties`
- CUB/hipcub namespace alias for portable CUB calls
- Non-temporal I/O (bypass L2):
  - CUDA sm_80+: inline PTX `ld.global.nc` and `st.global.wt`
  - HIP: `__builtin_nontemporal_load/store`
  - float4 vectorized variants
- AMD occupancy attributes: `GROK_WAVES_PER_EU(min, max)`, `GROK_FLAT_WORK_GROUP_SIZE(min, max)` (no-op on CUDA)

### `types.h`
- Compile-time constants:
  - `MAX_D_STATE = 32`
  - `MAX_D_INNER = 32`
  - `MAX_D_MODEL = 16`
  - `MAX_GRU_HIDDEN = 8`
  - `MAX_EXPERT_HIDDEN = 16`
  - `MAX_TOPK = 4`
  - `PSCAN_BLOCK = 512` (Blelloch threads/block)
  - `PSCAN_THRESHOLD = 256` (use sequential below this N)
  - `GEMM_PRECOMPUTE_THRESHOLD = 1024` (use cuBLAS GEMM above this N)
- `Affine2x2` struct: 4 floats matrix + 2 floats bias = 6 floats
- `affine_combine()` — portable C++ composition

### `ptx_intrinsics.cuh`
- `affine_combine_ptx(left, right)` — 12 FMAs in 3 waves, ~10 cycles
- `softplus_ptx(x)` — `ex2.approx` + `lg2.approx` + branchless saturation, ~2 cycles
- `fast_exp_ptx(x)` — `ex2.approx`, 1 cycle
- `stochastic_round_ptx(x, rand_bits)` — `cvt.rmi` + `selp`, branchless
- `gru_gates_ptx(...)` — interleaved sigmoid pair using `ex2.approx` + `rcp.approx`
- HIP fallbacks use standard math library

### `utils.cuh`
- `warp_reduce_sum(val, d_inner, tid)` — warp-shuffle reduction adapting to warp size
- `hash_prng(step, idx)` — Philox-like deterministic PRNG, no state buffer
- BF16 and INT8 stochastic rounding helpers
- `fast_rsqrt_nr(x)` — `rsqrt.approx.f32` + Newton-Raphson refinement
- `ptx_fma`, `ptx_exp2`, `ptx_log2`, `ptx_expf`, `ptx_tanhf`, `ptx_sigmoidf`
- `ptx_expert_mlp_forward<H>` — templated, fully unrollable
- `ptx_int8_stochastic_round` — uses `prmt.b32` byte permutation

### `ops.h` / `ops.cpp`
- C++ binding layer; ~79 kernel launchers
- `ops.h` declares all launchers
- `ops.cpp` is high-level glue per optimizer step
- Decides parallel vs sequential scan, GEMM vs custom precompute
- CPU fallback via PyTorch ATen ops (correct, slow), guarded by `WITH_CUDA`/`WITH_HIP`

### `quantization.h`
- `PrecisionMode` enum: FP32, TF32, BF16, FP8_E4M3, INT8_SYM, INT4_GPTQ, MXFP4
- Device-side dequant helpers:
  - `dequant_int8(q, scale)` — symmetric, per-tensor
  - `dequant_int4(packed, which, scale, zero)` — group_size=32, asymmetric
  - `dequant_mxfp4(packed, which, shared_exp)` — block_size=32 shared exponent
  - `fp4_e2m1_to_float` — lookup table {0, 0.5, 1, 1.5, 2, 3, 4, 6}

### `dispatch.h`
- C++ side of arch detection
- NVIDIA tiers: GENERIC (sm_70/75), AMPERE (sm_80–89), HOPPER (sm_90), BLACKWELL (sm_100)
- AMD tiers: GENERIC (gfx908/90a/942), CDNA4 (gfx950)
- `get_sm_arch()` via `cudaGetDeviceProperties`, respects `FORCE_ARCH` env var
- `StatePrecision` enum: FP32, CONFIG4 (INT8 state), FP6 (CDNA4)
- `ExpertPrecision` enum: FP32, INT8, INT4, MXFP4, FP4 (CDNA4)

## 6. csrc/cuda/generic — generic kernels

### `supergrok2_mamba_peer_kernels.cu` (forward path)

- **`input_proj_sort_kernel`** — projects `[grad, sharpness]` to `[N, d_model]`, emits `|grad|` as sort key plus identity index permutation. Clips NaN/Inf to zero. 256 threads/block, one element per thread, `#pragma unroll 4` on d_model loop.
- **`mamba3_scan_kernel`** — sequential selective scan, used when N < 256. 16 threads per param (one per d_inner). Per timestep: x-branch and z-gate via shared in_proj_W, dt via softplus_ptx, B and C projections via shared x_branch, trapezoidal state recurrence with paired RoPE rotation via FAST_SINCOSF, gated output `y * silu(z) + D * x`. Reverse flag drives backward bidirectional pass.
- **`mamba3_parallel_precompute_kernel`** — precomputes `pre_x_val`, `pre_z_val`, `pre_dt_val`, `pre_B_val`, `pre_C_val` for all timesteps in parallel, no inter-timestep dependencies. Used when 256 ≤ N < 1024. 256 threads/block, one timestep per thread.
- **`mamba3_parallel_scan_kernel`** — Blelloch parallel prefix scan over Affine2x2 transforms. PSCAN_BLOCK=512 threads/block. Three phases:
  1. Each thread sequentially scans a chunk to produce one Affine2x2 summary
  2. Up-sweep + down-sweep on summaries in shared memory (12KB for 6 floats × 512 threads)
  3. Each thread re-scans its chunk applying its prefix, accumulates into output
  Skips `__syncthreads()` for stride < WARP_SIZE.
- **`fused_elem_step_kernel`** — the per-element step. One thread per element. Sequence: load fwd/bwd Mamba scan outputs (float4 vectorized for d_inner ≤ 4), project to d_model contexts, non-temporal load of GRU state, GRU update with `gru_gates_ptx` for sigmoid pair, non-temporal store of new GRU state, build query per PEER head, score against 12 product keys per half via `LDG` cached loads, hard-route to one expert per head, evaluate 2-layer expert MLP from shared memory, accumulate weighted output, atomic-add expert counter, smart_grad = grad + rescale × meta_out, slow EMA update, effective grad = smart + λ × mu, Adam moment updates, parameter update with decoupled weight decay. Shared memory ~8.5 KB per block.

### `supergrok2_mamba_peer_backward_kernels.cu` (backward path)

- **`bilevel_precompute_kernel`** — same as forward parallel precompute, used for backward replay
- **`softplus_bias_kernel`** — applies `softplus(x + bias)` element-wise, used after cuBLAS dt projection
- **`bilevel_precompute_gemm`** — wraps `torch::mm_out` calls (cuBLAS path) when N ≥ 1024. Splits in_proj into x and z halves; runs in_proj_x, in_proj_z, dt_proj, B_proj, C_proj as 5 GEMMs. Calls `softplus_bias_kernel` after dt.
- **`mamba3_parallel_scan_fwd_save_kernel`** — same as forward parallel scan but saves selected hidden states to `saved_states` for backward. Checkpoint policy: save every state if `checkpoint_interval ≤ 1`, else save every Cth state.
- **`mamba3_scan_fwd_save_kernel`** — sequential variant for small N
- **`mamba3_scan_backward_kernel`** — backward scan. Walks timesteps in reverse. Per step: backprop through SiLU gating, through C projection (two-pass with warp reductions to a `d_C_vals_buf`, then backward GEMM for d_C_proj_W), through trapezoidal-discretized affine recurrence and RoPE, through dt projection, through B projection, through input projection. Block-local shared-memory accumulators for weight gradients, atomicAdd flush at block end.
- **`input_proj_backward_kernel`** — outer-product accumulation of `d_x` against `[grad, sharpness]` into proj_W and proj_b. Block-local accumulator + atomic flush.
- **`gru_backward_kernel`** — gradients w.r.t. Wz, Wr, Wh and bz, br, bh and gru_input. Unrolls gate logic carefully, accumulates via shared memory.
- **`expert_peer_backward_kernel`** — two-pass for softmax-backward coupling:
  - Pass 1: accumulate softmax dot products
  - Pass 2: full softmax-backward + gradient accumulation into expert weights, product keys, query weights
- **`out_proj_backward_kernel`** — outer-product accumulation for d_out_proj_W

### `supergrok15_kernels.cu`

- **`fused_mu_metanet_kernel`** — updates mu EMA, evaluates 2-input MLP per element with weights in shared memory, fast-GELU activation, output is smart_grad = grad + rescale × mlp_out
- **`fused_adam_decay_kernel`** — final blend with mu, Adam moments update with `fast_rsqrt_nr`, progressive decoupled weight decay. Non-temporal stores. Float4 fast path.
- **`sam_perturb_kernel`** — `param[i] += rho_over_norm × grad[i]`. Float4 fast path.
- **`sharpness_restore_kernel`** — `sharpness[i] = |sam_grad - normal_grad|`, restore param from backup
- **`fused_supergrok15_full_step_kernel`** — fuses mu_metanet + adam_decay. Smart_grad is register-resident, never hits global memory. ~50% bandwidth reduction.
- Templated specializations for H=16/32/64/128 with full unrolling. Runtime variant uses `#pragma unroll 4`.

### `supergrok11_kernels.cu`

- **`launch_sg11_mu_metanet`** — same as v1.5 mu_metanet but with cosine gating
- **`compute_cosine_gate`** — ATen helper that computes cos_sim between smart_grad and mu, passes through temperature sigmoid
- **`cosine_gate_reduce_kernel`** — fused 3-quantity reduction (dot, |sg|², |mu|²). Warp shuffle within warp, atomicAdd per warp into globals.
- **`compute_cosine_gate_fused`** — ATen wrapper around the reduce kernel
- **`launch_sg11_adam_decay`** — same shape as v1.5, takes lamb_eff = ramp × cos_gate × base_lamb
- **`fused_sg11_full_step_kernel`** — fused full step with cosine gate input

### `grokadamw_kernels.cu`

- **`fused_grokadamw_step_kernel`** — EMA update, gradient amplification, Adam moments, decoupled WD, parameter update. Non-temporal I/O for state. Float4 fast path.
- **`fused_grokadamw_step_q3_kernel`** — quantized variant. INT8 per-block exp_avg with FP32 per-block scales (block_size=8). BF16 stochastic-rounded exp_avg_sq and ema using `hash_prng`. ~50% optimizer state memory reduction.

### `neuralgrok_kernels.cu`

- **`fused_neuralgrok_amplifier_kernel`** — amplifier MLP per element. Linear(1→H), ReLU, Linear(H→1). Weights cooperatively loaded into shared memory. `amplified_grad = grad × (alpha × mlp_out + beta)`.
- **`fused_neuralgrok_full_step_kernel`** — fuses amplifier + Adam, amplified_grad register-resident
- Templated H=16/32/64/128

### `prodigy_kernels.cu`

- **`prodigy_dlr_reduce_kernel`** — global reduction, computes numerator (Σ grad × distance) and denominator (Σ s) via warp shuffles and per-warp atomicAdd. New `d_lr = sqrt(num / denom + eps)`.
- **`fused_prodigy_step_kernel`** — moment updates scaled by `d_lr`, s update, parameter update with `lr × d_lr × wd`.

### `grokfast_kernels.cu`

- **`fused_grokfast_ema_kernel`** — standalone EMA update + amplification (used in non-fused paths)
- **`fused_grokfast_adam_kernel`** — fused full step. Amplified grad register-resident.

### `lion_kernels.cu`

- **`fused_lion_step_kernel`** — interpolated direction `β1 × m + (1-β1) × grad`, sign extraction via `copysignf`, parameter update with decoupled WD, momentum EMA update with β2. Non-temporal I/O. Float4 fast path.

### `looksam_kernels.cu`

- **`looksam_norm_reduce_kernel`** — fused two-norm reduction: `|sam_grad - normal_grad|²` and `|grad|²` in one pass. Warp shuffles + per-warp atomic.
- **`looksam_direction_kernel`** — `v_dir[i] = (sam_grad - normal_grad) × inv_norm`
- **`looksam_direction_adjust_fused_kernel`** — fused direction + gradient adjustment. v_dir register-resident.

### `muon_kernels.cu`

- **`muon_momentum_normalize_kernel`** — momentum EMA update + division by Frobenius norm
- **`muon_ns_combine_kernel`** — Newton-Schulz inner: `out = a × X + b × AX + c × AAX` with hand-tuned coefficients (a=3, b=-3, c=1 default). AX and AAX are computed by separate cuBLAS matmul calls outside this kernel.
- **`muon_ns_combine_update_fused_kernel`** — final NS iteration combine + parameter update fused. Orthogonalized direction register-resident.

### `moe_deep_kernels.cu`

- **`moe_dynamic_expert_load_kernel`** — load only active experts' weights into shared memory based on gate logits
- **`moe_dynamic_expert_fwd_kernel`** — forward through dynamically loaded subset
- **`moe_dynamic_expert_bwd_kernel`** — backward through dynamic loading
- **`moe_filter_active_params_kernel`** — compact parameter index list to active experts only
- **`moe_scan_compacted_kernel`** — Mamba scan on compacted subset
- **`moe_scatter_results_kernel`** — scatter results back to full positions
- **`moe_count_expert_activations_kernel`** — atomicAdd per expert
- **`moe_compute_load_balance_loss_kernel`** — auxiliary loss for uniform expert utilization
- **`moe_apply_frequency_scaling_kernel`** — per-expert lr scaling by activation frequency

### `multi_tensor_optimizer_kernels.cu`

- Single kernel launch for many small parameter tensors. 2D grid: blockIdx.y selects param, threads in row iterate via grid-stride.
- Supports: GrokAdamW, Lion, Grokfast EMA, Prodigy step
- Pointer-packing: param pointers packed once on host, transferred to device, indexed by blockIdx.y
- Saves 100-500 ms/step on transformers with many small params

### `multi_tensor_prepare.cu`

- Fuses per-step preparation into one kernel: gradient norm, clipping, NaN/Inf replace, bias correction, per-param scalars
- One block per parameter, parallel reduction within block via warp shuffles + shared memory

### `distributed_scan_kernels.cu`

- **`mamba3_scan_local_with_summary_kernel`** — each GPU runs local Blelloch scan on its chunk, produces one Affine2x2 summary
- **`scan_summary_prefix_kernel`** — gathers summaries on rank 0, computes per-GPU prefix corrections via small prefix scan
- **`mamba3_apply_scan_prefix_kernel`** — each GPU applies its prefix to local output
- Communication: ~6 floats per GPU per scan
- Backward variants for gradient computation

## 7. csrc/cuda/sm_80 — Ampere

Headline optimization: `cp.async` double-buffered prefetch. Overlaps multi-hundred-cycle global memory latency with scan compute, hides ~50% of memory stalls.

- **`supergrok2_scan_sm80.cu`** — sequential scan with cp.async prefetch
  - **`mamba3_scan_batched_cpasync_kernel`** — batched sequential scan, double-buffered shared memory
  - **`mamba3_scan_combined_cpasync_kernel`** — forward + backward scan fused
- **`supergrok2_backward_sm80.cu`** — backward scan with cp.async
- **`supergrok2_fused_elem_sm80.cu`**
  - **`fused_elem_step_cpasync_kernel`** — per-element step with cp.async-prefetched expert weights
- **`metanet_optimizers_sm80.cu`** — Ampere-tuned optimizer kernel templates
- **`metanet_cpasync_variants_sm80.cu`** — cp.async variants for the meta-net optimizers
- **`muon_sm80.cu`** — Muon for Ampere with TF32 GEMMs via cuBLAS

Precision: TF32 for projection matmuls (transparent via cuBLAS). 192KB shared memory per SM.

## 8. csrc/cuda/sm_90 — Hopper

Headline optimization: FP8 E4M3 projection precompute via cuBLAS for N ≥ 4096. ~2× speedup vs BF16 (905 vs 452 TFLOPS on H100). Device-side absmax computation avoids host-device sync.

- **`supergrok2_scan_sm90.cu`** — FP8 precompute integrated with scan
  - **`hopper_fp8_gemm`** — cuBLAS GEMM with FP8 inputs and FP32 accumulation, scale = absmax / 448.0 (FP8 E4M3 max)
- **`supergrok2_backward_sm90.cu`** — backward with FP8 projection backward GEMMs
- **`supergrok2_warp_specialized_sm90.cu`** — uses Hopper distributed shared memory; producer/consumer warp specialization (one warp loads, another computes)
- **`metanet_optimizers_sm90.cu`** — Hopper-tuned optimizer kernels
- **`muon_sm90.cu`** — Muon for Hopper with FP8 GEMMs

Note: TMA is **not** used for the scan because per-timestep scattered reads (sort permutation) are poorly suited to TMA's bulk-copy descriptor model.

228KB shared memory. Thread block clusters supported.

## 9. csrc/cuda/sm_100 — Blackwell

Conservative tier. Most heavy features (TMEM, MMA.2SM, native NVFP4) deferred to Hopper FP8 fallback with documented delegation.

- **`supergrok2_sm100.cu`** — TMA bulk-copy kernels for expert weights
  - **`fused_elem_step_tma_kernel`** — per-element step with TMA-prefetched expert weights (single-thread initiation, hardware-managed transfer)
- **`supergrok2_precompute_sm100.cu`** — FP4 precompute scaffolding
- **`supergrok2_scan_sm100.cu`** — scan with TMA

Hardware features available:
- TMA (Tensor Memory Accelerator): hardware-managed asynchronous bulk copy via descriptors
- FP4 E2M1 native matrix multiply: `mfma_f32_32x32x8_fp4`, 8× elements per instruction
- TMEM: on-chip tensor memory (currently unused)

Fallback chain: Blackwell → Hopper FP8 → Ampere → Generic.

## 10. csrc/hip — AMD ROCm

Wavefront 64 throughout (CDNA). All kernels use `WARP_SIZE` from `platform.h` for portability.

### `cdna2/` (gfx90a, MI250)

- **`supergrok2_scan_cdna2.hip.cpp`** — baseline CDNA scan, MFMA `mfma_f32_16x16x4` for matrix ops
- 8MB L2, 220 CUs

### `cdna3/` (gfx942, MI300X)

- **`supergrok2_cdna3.hip.cpp`** — BF16 MFMA projection precompute
  - **`cdna3_precompute_bf16`** — runs in_proj_x, in_proj_z, dt_proj, B_proj, C_proj as BF16 matmuls via rocBLAS, output cast back to FP32. Dispatches to `MFMA_F32_32x32x8_BF16`, ~2× FP32 MFMA throughput.
- 304 CUs, 256MB L2 (meta-net always resident)

### `cdna4/` (gfx950, MI350X)

- **`cdna4_kernels.hip.cpp`** — native FP4/FP6/2:4 sparsity scaffolding

FP4 expert kernels:
- **`cdna4_fp4_expert_load`** — dequantize FP4 weights to FP32
- **`cdna4_fp4_expert_fwd`** — forward with FP4 expert weights via `mfma_f32_32x32x8_fp4`
- **`cdna4_fp4_expert_bwd`** — backward with gradient accumulation
- **`cdna4_fp4_quantize_experts`** — re-quantize expert gradients

FP6 state kernels (E3M2 native):
- **`cdna4_fp6_state_pack`** — FP32 → FP6 + per-block scale
- **`cdna4_fp6_state_unpack`** — FP6 → FP32
- **`cdna4_fp6_adam_step`** — Adam directly on FP6 state
- **`cdna4_fp6_lamb_step`** — LAMB on FP6 state

2:4 sparsity:
- **`cdna4_sparse24_select`** — select 2 non-zeros from each group of 4
- **`cdna4_sparse24_apply_mask`** — mask gradients to 2:4 pattern
- **`cdna4_sparse24_project`** — project momentum to sparse pattern
- **`cdna4_sparse24_densify`** — convert sparse → dense

Fused combos:
- **`cdna4_fp4_sparse24_fused_expert`** — expert MLP with FP4 weights + 2:4 sparsity
- **`cdna4_supergrok15_full_step`** — full v1.5 step on FP6 state + FP4 experts

512 CUs, 288MB L2.

### `README_HIP.md`

- Notes on wavefront-64 specific tuning, sync-skip behavior, MFMA dispatch

## 11. csrc/quantization — quantization kernels

### `quantization_kernels.cu`

FP8 E4M3:
- Two-phase kernel.
- Phase 1: warp-shuffle reduction within warp + atomicAdd to global accumulator → absmax
- Phase 2: rescale by `absmax / 448.0` (FP8 E4M3 max), quantize element by element with float4 vectorization, write uint8 + FP32 scale

INT8 symmetric:
- Same reduction pattern, limit 127, signed output
- `q = clamp(round(x / scale), -127, 127)`, `scale = absmax / 127`

INT4 GPTQ-style:
- Group-wise (group_size=32)
- Per-group min/max → scale and zero-point
- Asymmetric: `[min, max] → [0, 15]`
- Two values packed per byte (low/high nibble)

MXFP4:
- Per-block (block_size=32) shared 8-bit exponent
- 4-bit FP4 magnitudes per element + separate sign bit
- Magnitude lookup: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`
- Block scale: `2^(shared_exp - 127)`

Dequant helpers in `csrc/common/quantization.h` are inlined into optimizer kernels (e.g., expert weight dequant in shared memory).

## 12. Algorithms

### Affine2x2 Mamba encoding
- Mamba-3 recurrence: `h[t] = A_bar[t] × RoPE_rotate(h[t-1]) + B_bar[t] × x[t]`
- Split state into RoPE-paired even/odd dims → recurrence is an affine map on a 2D vector
- 2×2 matrix folds in: state-transition `A_bar`, RoPE rotation
- 2-vector bias holds: input contribution `B_bar × x`
- Composable: `(M_r, b_r) ∘ (M_l, b_l) = (M_r × M_l, M_r × b_l + b_r)`
- Associativity → eligible for parallel prefix scan
- Storage: 6 floats per element

### 12-FMA composition
- Affine2x2 composition: 8 FMAs for matrix product + 4 FMAs for matrix-vector + bias = 12 FMAs
- Inline PTX `affine_combine_ptx` arranges in 3 waves of 4 FMAs each:
  - Wave 0: 4 independent matrix products (no dependencies, fills both FMA pipelines)
  - Wave 1: 4 dependent accumulations + 2 bias starts
  - Wave 2: 2 final bias accumulations
- ~10 cycles total vs ~40+ for naive C++

### Blelloch parallel prefix scan
- Two-phase associative-operator scan
- Up-sweep: combine pairs at strides 1, 2, 4, …, leaves a root holding full composition
- Down-sweep: distribute exclusive prefixes back down with identity at root
- Work: O(N), depth: O(log N)
- Each leaf is one thread's chunk summary; each combine is one `affine_combine_ptx`
- Skip `__syncthreads()` for stride < WARP_SIZE (lanes implicitly synced)

### Bilevel checkpointing
- Naive backward stores all N timesteps × d_inner × d_state floats per param (e.g., 20 GB for N=100K)
- Checkpoint every C-th state (C = `checkpoint_interval`), recompute intermediates from nearest checkpoint
- Memory savings: ~(C-1)/C
- Backward compute: ~2× with C=256
- Default C=1 (full save), tunable via `bilevel_checkpoint_interval`

### Register-resident smart_grad
- Pattern shared by v1.5, v1.1, NeuralGrok, Grokfast, Muon final iter
- Smart_grad / amplified_grad / orthogonalized direction held in CUDA register
- Immediately consumed by Adam update in same kernel
- Avoids N writes + N reads of intermediate
- ~50% bandwidth reduction → 20-30% throughput gain

### Non-temporal I/O
- Optimizer state read-modify-write per step, not reused within step
- `stream_load`/`stream_store` use PTX `ld.global.nc` / `st.global.wt` on Ampere+
- HIP: `__builtin_nontemporal_load/store`
- Bypasses L2, leaves cache for hot data (weights, scan output)
- float and float4 variants

### PTX hot-path intrinsics
- `ex2.approx`: 1-cycle approximation of 2^x
- `lg2.approx`: 1-cycle approximation of log2(x)
- `rcp.approx`: 1-cycle reciprocal
- `softplus_ptx`: 2 cycles via ex2 + lg2 + selp
- `fast_exp_ptx`: 1 cycle via ex2.approx after multiply by log2(e)
- `gru_gates_ptx`: interleaved sigmoid pair (both pipelines busy)
- `stochastic_round_ptx`: branchless via cvt.rmi + selp
- 1-2 ULP error, acceptable for averaged optimizer state

### Warp-shuffle reductions
- Used in: cosine gate reduce, prodigy d_lr reduce, looksam norm reduce, expert backward
- Within warp: `__shfl_down_sync` butterfly at strides 16, 8, 4, 2, 1 (CUDA) or 32, 16, … (HIP-64)
- Cross-warp: lane 0 atomicAdds to global accumulator
- Avoids shared memory bottlenecks for small reductions

### Product-key PEER routing
- Score N elements against E experts in O(√E) work, not O(E)
- Split query into two halves, score against √E sub-keys per half
- Top-K each half → outer product → top-K(K²) candidates
- For E=144: 12-key sub-scoring × 2 = 24 dot products, vs 144 naive

### Cooperative shared-memory weight loading
- Used in: meta-net MLP weights, GRU weights, expert weights, prod keys
- All threads in block load disjoint slices of weights into shared memory
- One `__syncthreads()` then per-thread access at ~5 cycle latency vs 100+ for global

## 13. JAX/TPU

Functional rewrite of the suite. ~300 lines of core logic vs ~2000 lines of CUDA.

### Modules in `supergrok2_jax_tpu/`

- **`supergrok2_jax.py`** — main optimizer loop, `PerParamState`, `SuperGrok2State`, `OptimizerConfig`, `supergrok2_step`
- **`mamba3_peer_metanet_jax.py`** — meta-net architecture, `MetaNetWeights`, `init_meta_weights`, `meta_net_forward`
- **`scan.py`** — Mamba-3 scan via `jax.lax.associative_scan` with Affine2x2 combine operator (~40 lines)
- **`gru.py`** — `mini_gru` cell
- **`peer.py`** — `peer_expert_forward` (soft routing for bilevel), `peer_expert_forward_hard` (argmax for forward step)
- **`bilevel.py`** — `bilevel_step` using `jax.grad`, no custom backward needed
- **`pallas_kernels.py`** — optional Pallas kernels with try/except fallback
- **`sharding.py`** — `create_mesh`, `shard_params`, multi-host helpers
- **`simple_optimizers_jax.py`** — GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM
- **`metanet_optimizers_jax.py`** — SuperGrok v1.5, v1.1, NeuralGrok
- **`quantization_jax.py`** — INT8 symmetric quantization round-trip
- **`bridge.py`** — PyTorch ↔ JAX weight conversion + test vector export

### Key differences vs CUDA
- Functional: state is immutable pytrees, threaded through each step
- No custom backward: `jax.grad` autodiffs through `lax.associative_scan`
- Sharding declarative: `jax.sharding.NamedSharding`, `PartitionSpec`
- Multi-host: `jax.distributed.initialize`, `lax.pmean`, `lax.all_gather`

### Pallas kernels
- **`pallas_affine_scan`** — tiles scan into 128-element (v4/v5) or 256-element (v6e) blocks; intra-tile sequential scan + cross-tile prefix
- **`pallas_fused_gru_peer`** — fuses GRU + PEER routing + expert MLP, intermediates in VMEM
- **`vmem_persistent_expert_mlp`** — `eviction_policy="none"` keeps expert weights resident across tiles on v5p/v6e
- All wrapped in try/except, fall back to pure JAX if Pallas API changes

### TPU version detection
- `detect_tpu_version()` reads `jax.devices()[0].device_kind`
- Returns 'v4', 'v5e', 'v5p', 'v6e'
- Drives tile size and VMEM policy

### Feature gaps vs CUDA
- SAM perturbation not fully integrated (sharpness field exists, perturbation not wired)
- No explicit bilevel checkpointing (XLA's automatic rematerialization handles activations)
- INT8 only (no FP8/INT4/MXFP4/FP6)
- Expert load balancing minimal (counts tracked, no auxiliary loss)

## 14. Tests

Eight files, ~2,964 LOC. README still says six (stale). Total test points ~82.

### `test_supergrok2.py` (27 sections, 12A–12AA)
- 12A — import and build
- 12B — sequential vs parallel scan equivalence (N from 1 to 1024)
- 12C — forward step correctness (param changes, state populated)
- 12D — bilevel meta-learning correctness
- 12E — two-pass backward equivalence for scan weight gradients
- 12F — expert recycling stability (50 steps)
- 12G — gradient checkpointing equivalence (interval=1 vs 8)
- 12H — edge cases (N=0, N=1, zero grad, large grad, FP16 params)
- 12I — all 11 optimizers construct + step
- 12J — memory leak check (200 steps, <10% growth)
- 12K — two-pass GEMM backward reproducibility (max diff <1e-4)
- 12L — batched parallel scan single-launch with bitwise reproducibility
- 12M — dispatch detection (Python/C++ agreement)
- 12N — precision config auto-selection
- 12O — projection precision FP32 vs auto equivalence
- 12P — dispatch convergence (10 steps)
- 12Q — platform/vendor detection
- 12R — INT8 symmetric quantization round-trip
- 12S — INT4 GPTQ packing correctness
- 12T — MXFP4 quantization
- 12U — dynamic precision selection
- 12V — expert FP32 passthrough
- 12W — distributed helpers (DDP hooks, no-op without dist)
- 12X — CompiledSuperGrok2 wrapper (warmup/capture/replay)
- 12Y — `step_compiled` method
- 12Z — FSDP exclusion helper
- 12AA — distributed module imports

### `test_matrix.py`
- Cross-platform correctness matrix
- Runs 10 optimizers (excludes Mamba3PEERMetaNet which is internal)
- 5 steps per config, validates no NaN, measures step time
- Honors `FORCE_ARCH` env var

### `test_all_tiers.py`
- Validates dispatch correctness across NVIDIA tiers (generic, Ampere, Hopper)
- Sets `SUPERGROK_FORCE_ARCH` and runs `test_matrix.py` for each

### `test_cpu_fallback.py` (12 sections)
- `_HAS_*` flag sanity
- Python fallback module existence (13 variants)
- Strict `_ops` import in optimizer files
- Numerical correctness for Lion, GrokAdamW, Grokfast EMA, LookSAM, Muon Newton-Schulz
- CPU C++ extension completeness
- Importability of all optimizers
- Prodigy `d_lr` return value
- `setup.py` CPU sources listing

### `test_jax_matrix.py`
- Same matrix as PyTorch but for JAX optimizers
- 10 JAX optimizers, validates param changes and no NaN

### `test_amd_hip.py` (6 sections)
- `platform.h` adherence (no raw CUDA in generic)
- AMD tier detection via FORCE_ARCH
- PrecisionConfig auto for CDNA2/3 (BF16)
- `get_amd_label()` GPU labels
- GCN arch parsing (MI100/MI250/MI300X, three-digit codes)
- Wavefront-64 sync skip behavior

### `test_new_features.py` (7 sections)
- float4 vectorized GrokAdamW with alignment fallback
- OverlappedOptimizer distributed wrapper
- INT8 / PowerSGD gradient compression
- Pallas scan fallback
- Interleaved states layout
- Sparse gradient mask inference
- Partial CUDA graph optimizer

### `test_training_aware.py` (7 sections)
- Non-temporal stream_load/store correctness
- Q3 quantized states valid (no NaN, loss decreases)
- Q3 matches FP32 direction (cosine similarity > 0.99)
- Stochastic rounding unbiasedness
- No `.item()` calls in hot path
- PipelinedOptimizer equivalence
- training_benchmark script error-free

### Notable gap
- No explicit fused-CUDA-vs-Python-fallback bitwise/numerical agreement test (called out in `ANALYSIS.md`)

## 15. Benchmarks

### `benchmark_supergrok2.py`
- Models: tiny (h=32), small (h=64), medium (h=128), large (h=256), xlarge (h=512)
- Optimizers: all 11 + AdamW baseline
- Metrics: step time (ms), peak GPU memory (MB) by category, throughput (params/sec)
- Phases: 10 warmup + 100 timed
- Same init, same data (batch=32), same seed across optimizers
- Flags: `--optimizer`, `--model-size`, `--include-bilevel`, `--per-tier`, `--verbose`

### `autotune.py`
- Per-GPU profiling, results cached at `~/.cache/supergrok/autotune_{gpu_key}.json`
- GPU key: hash of device name + SM + total memory
- Profiles: scan block-size throughput, projection precision (FP32/TF32/BF16/FP8/MXFP4), memory
- Note: `PSCAN_BLOCK` is constexpr; changing requires rebuild
- Flags: `--dry-run`, `--force`, `--verbose`

### `training_benchmark.py`
- End-to-end grokking-style training run
- Reports loss/accuracy curves over time
- For comparing convergence speed and memory efficiency

### Fairness notes (from `ANALYSIS.md` §3)
- Same init, multi-GPU round-robin, multi-seed bands ✓
- SuperGrok optimizers do extra per-step work (meta-net forward, SAM, bilevel) — wall-clock not directly comparable
- SuperGrok bilevel uses validation data → information advantage vs other optimizers
- Missing baseline: standalone SAM/GSAM

## 16. Codegen

Development-time scripts. Generated outputs are checked in; not run at build.

### `generate_kernels.py`
- Generates GrokAdamW Q3 kernels (INT8 per-block exp_avg + BF16 stochastic-rounded)
- Generates `compute_absmax_scale_kernel.cu`
- Generates `muon_update_generated.cu` with non-temporal I/O
- Scalar (S) and float4 (V) variants

### `generate_sg2_kernels.py`
- Template-based from `kernel_specs.yaml`
- Generates Ampere (sm_80) and Hopper (sm_90) optimizer kernel variants
- Features: cp.async (Ampere), FP8 E4M3 (Hopper)
- Output goes to `csrc/cuda/sm_80/` and `csrc/cuda/sm_90/`

### `kernel_specs.yaml`
- Lists 12+ optimizer specs
- Each spec: block_size, launch_bounds, state vars + quantization formats, scalars, update math
- Variant axes: GPU (S_F_D, V_F_D, S_Q_D, V_Q_D, S_F_M, V_F_M, S_Q_M, V_Q_M), CPU (cpu_F, cpu_Q)
- Templates use placeholders: STATE_LOAD/STORE, PARAM_LOAD/STORE, GRAD_STORE, EXTRA_LOAD
- GrokAdamW has 16 GPU + 2 CPU variants

### `common_macros.j2`
- Jinja2 macros: synchronization, warp reductions, memory access patterns

## 17. Build

`setup.py` at repo root.

### Backend detection
- HIP: `torch.version.hip is not None`
- CUDA: `torch.cuda.is_available()` (or `FORCE_CUDA=1` for build-only)
- Falls back to CPU otherwise

### CUDA path (WITH_CUDA)
- Generic sources: 18 files (optimizer kernels, distributed scan, MoE, quantization)
- sm_80: 6 files (cp.async scan, backward, fused_elem, optimizers, cpasync variants, muon)
- sm_90: 5 files (FP8 scan, backward, warp-specialized, optimizers, muon)
- sm_100: 3 files (TMA kernels)
- Auto-detects generated sources in `csrc/cuda/generated/`
- Flags: `nvcc -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr`
- Arches: `-gencode arch=compute_{70,75,80,86,89,90,100},code=sm_*`
- Override: `TORCH_CUDA_ARCH_LIST` env var

### HIP path (WITH_HIP)
- Generic sources: same 18
- CDNA-specific: gfx90a (CDNA2), gfx942 (CDNA3, 3 files), gfx950 (CDNA4, 1 file)
- Flags: `hipcc -O3 -std=c++17 --offload-arch=gfx908,gfx90a,gfx942,gfx950`

### CPU path
- 7 core sources + generated
- SIMD: AVX-512 (x86_64) or NEON (ARM64) detected via `-march=native`
- Flags: `g++ -O3 -std=c++17 -fopenmp -ffast-math -funroll-loops`
- OpenMP parallelism at parameter level

### Total
- ~67 source files on CUDA path
- Clean CUDA build for full arch matrix: several minutes
- Editable install: `pip install -e .`

## 18. Recent commits

Newest first.

- **`ea968b6`** — Fix critical bugs and apply optimizations across all optimizer components
  - NeuralGrok: fix `_single_param_step` crash (wrong function name, missing `step_list`)
  - JAX/TPU Pallas: fix tile corruption, infinite recursion in persistent scan
  - SuperGrok v2: replace `except Exception: pass` with `RuntimeError` + warning
  - C++ `dispatch.h`: fix AMD GCN arch parsing for 3-digit codes
  - C++ `ops.cpp`: batch CPU syncs, vectorize Mamba-3 inner loops, cache `g × d_lr` in Prodigy
  - Python fallback: eliminate `.item()` calls in hot path

- **`a6323c9`** — Fix `_single_param_step` bugs in muon, prodigy, grokadamw

- **`6c48166`** — Fix potential out-of-bounds read in AMD `gcnArchName` parsing (3-digit codes)

- **`dbe3ef4`** — Fix 9 bugs and apply FP32 skip optimization across optimizer suite (skip `.to(kFloat32)` when already FP32)

- **`1d930db`** — Wire dead fused kernels and eliminate Python `adamw_step` bottleneck

### Trajectory
- Architecture settled, focus on hot-path performance and correctness
- Recent: kernel fusions, register-resident intermediates, non-temporal I/O, reduction kernel improvements
- Architecture coverage: Hopper FP8 added, CDNA3 BF16 MFMA added, Blackwell + CDNA4 scaffolded
- Bug fix backlog draining: silent exception swallowing, redundant forward passes, meta-net device placement, `id`-based caching fragility, single-param-step bugs

## 19. Known gaps

### Optimization opportunities (`ANALYSIS.md` §8)

| # | Optimization | Impact | Difficulty |
|---|---|---|---|
| 1 | Fuse v1.1 cosine gate into full_step kernel | Medium | Easy |
| 2 | Fuse NeuralGrok amplifier + Adam | Low | Easy |
| 3 | Cache meta-net weights across steps | Low-Medium | Easy |
| 4 | Pre-allocate scan workspace buffers | Low-Medium | Easy |
| 5 | Persistent CUDA streams | Very Low | Easy |
| 6 | Skip `.to(kFloat32)` when already FP32 | Very Low | Easy (partial done) |
| 7 | Switch meta-net GELU from tanh to sigmoid form | Low | Easy |
| 8 | Custom cosine-gate reduction kernel | Low | Medium |
| 9 | Batch Muon Newton-Schulz across 2D params | Low | Medium |
| 10 | CUB segmented sort for batched gradient sort | Low | Medium |

### Design concerns (`ANALYSIS.md` §2)
- Peak weight decay aggressive: sigmoid scheduler can multiply base WD by 20 → effective 5.0 → ~99.3% shrinkage over 1000 steps
- Memorization detection binary: sharp threshold at training_acc=0.995, transition not smooth
- SuperGrok bilevel uses validation data → information advantage in benchmark comparisons

### Test gap
- No explicit fused-CUDA-vs-Python-fallback bitwise/numerical agreement test

### Architecture-specific gaps
- Blackwell: TMEM, MMA.2SM, NVFP4 native — scaffolded, delegates to Hopper FP8
- CDNA4: native FP4 expert, FP6 state — scaffolded, delegates to next-lower tier

### JAX/TPU gaps
- SAM not fully integrated (sharpness field exists, perturbation not wired)
- No explicit bilevel checkpointing (XLA rematerialization handles)
- Quantization INT8 only (no FP8/INT4/MXFP4/FP6)
- Expert load balancing minimal

### Documentation staleness
- README test count: says 6 files, actual 8
- ANALYSIS.md test point count: says 67, actual ~82
- Codegen relationship to setup.py not visible to readers

## 20. Quick reference

### Optimizer feature matrix

| Optimizer | Meta-net | State tensors | Decoupled WD | SAM | Bilevel | Grokking | Fused kernel | Python fallback |
|-----------|----------|---------------|--------------|-----|---------|----------|--------------|-----------------|
| SuperGrok2 | Mamba3+PEER+GRU | 7 | ✓ | ✓ functional | ✓ | ✓ | ✓ | ✓ full |
| SuperGrok15 | MLP 2-layer | 4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| SuperGrok11 | MLP + cosine gate | 4 | ✓ | ✓ | ✓ meta_step | ✓ | ✓ | ✗ |
| GrokAdamW | EMA filter | 3 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ (CPU C++) |
| NeuralGrok | Learned MLP | 2 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Prodigy | distance-aware | 4+init | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Grokfast | EMA amplify | 3 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Lion | momentum | 1 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ CPU |
| LookSAM | AdamW + periodic SAM | 3 | ✓ | ✓ periodic | ✗ | ✗ | ✓ | ✓ CPU |
| Muon | NS ortho 2D / Adam 1D | 1 or 3 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ CPU |

### Compile-time constants

| Constant | Value | Used where |
|----------|-------|------------|
| `MAX_D_STATE` | 32 | scan state dim cap |
| `MAX_D_INNER` | 32 | Mamba inner dim cap |
| `MAX_D_MODEL` | 16 | projection dim cap |
| `MAX_GRU_HIDDEN` | 8 | GRU hidden cap |
| `MAX_EXPERT_HIDDEN` | 16 | expert MLP cap |
| `MAX_TOPK` | 4 | PEER top-k |
| `PSCAN_BLOCK` | 512 | Blelloch threads/block |
| `PSCAN_THRESHOLD` | 256 | seq vs parallel scan switch |
| `GEMM_PRECOMPUTE_THRESHOLD` | 1024 | custom vs cuBLAS precompute switch |

### Decision tree for SuperGrok v2 forward
- N < 256 → sequential `mamba3_scan_kernel`
- 256 ≤ N < 1024 → `mamba3_parallel_precompute_kernel` + `mamba3_parallel_scan_kernel`
- N ≥ 1024 → `bilevel_precompute_gemm` (cuBLAS) + `mamba3_parallel_scan_kernel`

### Architecture tier fallback chains
- NVIDIA: Blackwell → Hopper → Ampere → Generic
- AMD: CDNA4 → CDNA3 → Generic

### Precision auto-selection chain
- nvfp4 → mxfp4 → fp8 → bf16 → fp32

### Where to find
- Optimizer Python: `grokking_optimizers/<name>.py`
- Optimizer JAX: `supergrok2_jax_tpu/<name>_jax.py` or in `simple_optimizers_jax.py`/`metanet_optimizers_jax.py`
- Optimizer kernel: `csrc/cuda/generic/<name>_kernels.cu`
- Arch-specific kernel: `csrc/cuda/sm_{80,90,100}/<name>_sm{80,90,100}.cu`
- HIP kernel: `csrc/hip/cdna{2,3,4}/<name>_cdna{2,3,4}.hip.cpp`
- Common headers: `csrc/common/`
- Quantization kernels: `csrc/quantization/`
- C++ binding: `csrc/common/ops.h`, `csrc/common/ops.cpp`
- Tests: `tests/test_<topic>.py`
- Benchmarks: `benchmarks/<name>.py`
- Codegen: `codegen/<name>.py` + `kernel_specs.yaml`
- Build: `setup.py`
- User docs: `README.md`
- Internal review: `ANALYSIS.md`
