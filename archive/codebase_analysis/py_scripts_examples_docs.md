# PY_SCRIPTS_EXAMPLES_DOCS — SuperGrok1.5 Analysis

**Scope:** `scripts/`, `examples/`, `docs/reviews/`
**Date:** 2026-06-25

---

## 1. scripts/check_math_single_source.py

**Purpose:** Enforced drift guard ensuring the canonical optimizer math (in `csrc/algorithms/<opt>.h`) is never silently reimplemented elsewhere. Exits 1 on any violation — this fails the build, not just warns.

**Three enforcement tiers:**

1. **Structural single-source (WS-F5):** `csrc/fused/sm_90/opt_components.cuh` (the only surviving includable CUDA consumer after Task #10 removed per-op kernels) must `#include` every canonical header AND call its apply symbol (via comment-stripped text search to avoid false passes from comments). The 11 non-SG2 apply symbols are hardcoded in `_FUSED_APPLY_SYMBOL` (lines 290-302). SuperGrok2 is exempted (WARN, not FAIL) because its fused tail uses a separate `launch_csa_hca_batched_step` path, not `opt_components.cuh`.

2. **Re-inline detection (WS3):** Scans `opt_components.cuh` for Adam moment-update expressions (`beta1 * exp_avg`, `beta2 * exp_avg_sq`, bias-corrected apply pattern). Any match in the consumer FAILS — math must come from a call into the canonical header.

3. **Cosine re-inline detection:** Scans `csrc/fused/sm_90/opt_stages_precompute.cuh` for re-inlined `den_m += x*x` / `den_g += x*x` SG11 cosine-gate accumulators. This guards against the exact pattern that caused the historical `cos(grad,mu)` bug.

4. **Content-hash manifest (WS2):** `scripts/optimizer_math_manifest.json` records SHA-256 of normalized (comments+whitespace stripped) canonical headers for all 11 optimizers. Hash drift fails unless `--update-manifest` is explicitly passed.

5. **Binding-region hashes (WS-F4):** `BINDING_FUNCS = ()` — empty because production routes through the unified fused ABI, not per-function pybind entrypoints. `binding_regions` in the manifest is `{}`.

**Optimizer list (line 51-54):** adamw, lion, grokfast, grokadamw, looksam, prodigy, neuralgrok, muon, supergrok11, supergrok15, supergrok2 (11 total).

**Usage:** `python3 scripts/check_math_single_source.py` (verify) or `--update-manifest` (re-record).

---

## 2. scripts/optimizer_math_manifest.json

Committed SHA-256 hashes for all 11 `csrc/algorithms/<opt>.h` canonical headers. `binding_regions` is empty `{}` (Task #10 consequence). Hashes are present and non-empty for all 11 optimizers. This is the source of truth for the WS2 drift guard.

---

## 3. scripts/roofline_bench.py

**Purpose:** 3-seed roofline timer for the d-scaled bench variants. Wraps `tuning/{decoder,vit,mamba}_bench.py`'s `build_variant()` for PROTOCOL-COMPLIANT 3-seed median step-ms.

**Key details:**
- Uses H100 989 TF/s as the roofline denominator (line 169: `/ 989.0`)
- Reports `pct_roofline` (% of H100 peak TF/s achieved)
- Supports `-D KEY=VAL` overrides (hill-climb macro injection)
- Default: `--seeds 42,7,123 --reps 3 --warmup 8 --iters 15`
- Calls `mod.tc_train_step(params, a, b, state, lr, beta1, beta2, eps, wd, bc1, bc2, 1, ncta)`
- FLOPS computed per-model: decoder uses `_decoder_gemm_flops_per_step`; vit/mamba compute manually from architecture dims (lines 87-107)
- NEVER touches the production `_ops.so` — builds a coexisting bench variant

---

## 4. scripts/fast_triage.py (1198 lines total, partial read of 856)

**Purpose:** Fast (~1-3 min) directional screen for kernel changes, avoiding the ~20-30 min full cycle. The full cycle (fp64 parity + A/A/A determinism + 3-seed timing) remains the ONLY arbiter for a KEEP.

**Architecture:**
- **Per-model wiring** (`_MODELS` dict, lines 210-248): decoder (d_default=2048, 8-phase profiler), vit (d_default=1024, 6-phase profiler), mamba (WALL-ONLY — no per-phase profiler yet)
- **Phase auto-targeting** (`_PHASE_HINTS`, lines 255-265): infers targeted phase from macro key substrings (DEC_DW→P2_dW_GEMM, GEMM_STAGES→P1_fwd, ADAMW/OPT→P3_opt_tail, etc.)
- **Calibration suite** (`_CALIBRATION`, lines 347-380): one registered ground-truth — the dW contiguous-layout staging KEEP (+2.05×, 1889.8→920.7ms @ d=2048/B=16384). ViT/mamba/compile.py/level-2 contexts are marked UNCALIBRATED.
- **Representativeness assertions** (`_assert_representative`): checks scale (d>=d_default, B>=16384) and phase-share drift vs calibration. Flags drift loudly.
- **Screen-vs-gate agreement ledger** (`.perf/screen_gate_agreement.jsonl`): records screen verdict vs full gate outcome for fidelity tracking. `--fidelity-report` summarizes per (model,context).
- **Relevance gate (Amdahl):** phase_share >= `--relevance-floor` (default 5%) required, else verdict = IRRELEVANT. Projected step Δ = share × phase_delta.
- **Isolated subprocess protocol:** each variant is built+timed in its own Python subprocess (`-c` script) for clean CUDA contexts.

**Profiling doctrine:** MEASURE BROAD always (all 8 phases), FOCUS FIX NARROW (attack one bottleneck phase), ZOOM DEEP only where digging (temporary sub-phase counters).

**Usage examples:**
```bash
python scripts/fast_triage.py --baseline-D SG_TUNED_DEC_DW_STAGE=0 --cand-D SG_TUNED_DEC_DW_STAGE=1
python scripts/fast_triage.py --validate  # runs calibration suite
python scripts/fast_triage.py --fidelity-report
```

---

## 5. scripts/compile_to_object.sh

**Purpose:** Minimal compile-to-/dev/null gate for a single CUDA TU. Used by CI/review gates.

```bash
nvcc -c -std=c++17 -DWITH_CUDA -gencode arch=compute_90a,code=sm_90a \
  -I. -I$TORCH/include ... -Ithird_party/cutlass/include ... \
  --expt-relaxed-constexpr --expt-extended-lambda [EXTRA] TU -o /dev/null
```

Prints `COMPILE_OK tu=<path>` or `COMPILE_FAIL rc=N tu=<path>`.

---

## 6. scripts/verify_stage0.sh

**Purpose:** Stage-0 compilation gate. Compiles 10 per-optimizer launchers + supergrok2 (with -DWITH_CUTLASS) + 3 model TUs (mamba, decoder, vit, each with -DWITH_CUTLASS). Prints PASS/FAIL per TU. Does NOT run self-test or lint (those are separate).

**TUs checked:** launch_{adamw,lion,grokfast,grokadamw,looksam,muon,neuralgrok,prodigy,supergrok11,supergrok15}.cu, launch_supergrok2.cu, models/{mamba,decoder,vit}.cu.

---

## 7. scripts/bootstrap_env.sh

**Purpose:** One-shot ephemeral environment restoration on a fresh RunPod container restart. The `/workspace` NFS volume is persistent; `/usr`, pip site-packages are ephemeral.

**Steps:**
1. Creates/activates venv at `/workspace/venv` with `--system-site-packages` (inherits torch 2.4.1+cu124)
2. Asserts `torch.__version__.startswith("2.4.1+cu124")` and `numpy.__version__ == "1.26.3"`
3. Runs `scripts/install_deps.sh`
4. Fetches sccache v0.8.2 to `/workspace/.local/bin` if absent
5. Sources `.regpressure/env.sh` (env vars, SCCACHE_DIR on /dev/shm)
6. Rebuilds `_ops.so` in-place only if missing (`pip install -e .`)
7. Writes `~/.supergrok_env` for auto-activation in new shells

---

## 8. scripts/nvcc_baseline.py

**Purpose:** Task #11 three-point comparison (A regular nvcc / B compile.py default / C compile.py JIT-tuned) isolating what the compile.py flag pipeline + autotuner buy over vanilla -O3.

**Key design:**
- Resolves the same 6 production TUs as setup.py (bindings.cpp + dispatch.cpp + sm_90/*.cu + sm_90/models/*.cu + fused/sm_90/*.cu)
- Variant A: vanilla flags (`-O3 --use_fast_math -arch sm_90a` + structural defines), NO ptxas micro-tuning
- Variants B/C: uses `grokking_optimizers.compile.NVCC_DEVICE_BASE + _newer_compiler_flags("sm_90")` (the exact production ladder: `--register-usage-level`, `--def-load-cache=ca`, `--def-store-cache=wb`, `--extra-device-vectorization`, `--maxrregcount`, etc.)
- Variant C additionally applies split-K=2 (the tuned knob)
- A→B isolates compilation pipeline; B→C isolates autotuner; A→C is total
- `--bench-d2048`: runs the d=2048 roofline 3-point comparison on the bench TU alone
- `--gate-cell`: runs fp64 parity gate against the vanilla binary

**TORCH_CUDA_ARCH_LIST forced to "9.0a"** to prevent torch from injecting non-`a` PTX targets which make ptxas reject wgmma.* (line 121).

---

## 9. scripts/time_cell.py

**Purpose:** Times `dispatch.fused_train_step()` end-to-end with CUDA events for the register-pressure patch ratchet.

- Forces SAM path on (looksam_sam=1 via step=1) for SAM-coupled cells (looksam, sg11, sg15, sg2)
- Uses `G._build_cell(model)` (exact same cell as the fp64 gate) for consistency
- Reports median/min/walls_ms per (opt, model, seed)

---

## 10. scripts/diag_sg_sharpness.py

**Purpose:** Diagnostic for SG11/SG15 sharpness correctness — classifies whether kernel sharpness error is on the bf16 floor (benign) or a real gap.

**Method:**
1. Runs `fused_train_step` with `return_grad=True`; extracts kernel sharpness from state at `kstate[3*total+1: 4*total+1]`
2. Computes bf16-faithful reference sharpness via `G._sg_bf16_sharpness_oracle`
3. Computes pure fp64 oracle via `_pure_fp64_oracle`
4. Reports `(kernel-vs-fp64)/(bf16-vs-fp64)` ratio: ~1 means on bf16 floor (benign); >>1 means real gap
5. Also shows per-layer worst-case (layer0 >> layer1 = compounding bug signature)

Runs for `supergrok11/mamba` and `supergrok15/mamba`.

---

## 11. scripts/diag_looksam_samdir.py, scripts/diag_neuralgrok_seed123.py

(Not read in detail — names suggest targeted diagnostics for LookSAM SAM direction and NeuralGrok on seed 123.)

---

## 12. scripts/amdgcn_check.sh, scripts/install_deps.sh

(Not read — amdgcn_check presumably validates HIP/AMD GPU kernel compilation; install_deps installs pip packages.)

---

## 13. scripts/_vit_baseline.py, _vit_ncu_driver.py, _vit_phase_profile.py

(Not read — ViT-specific profiling and NCU driver scripts, likely for ViT kernel benchmarking.)

---

## 14. scripts/STAGE3_PTX_AUDIT.md

**Purpose:** Audit of hand-written inline-PTX blocks to identify redundant vs load-bearing ones (Stage 3.0).

**Key findings:**

| Function | Location | Classification | Action |
|---|---|---|---|
| softplus_ptx | ptx_intrinsics.cuh:70 | REDUNDANT, 0 callsites | DELETE |
| fast_exp_ptx | ptx_intrinsics.cuh:94 | REDUNDANT, 0 callsites | DELETE |
| stochastic_round_ptx | ptx_intrinsics.cuh:112 | REDUNDANT, 0 callsites | DELETE |
| gru_gates_ptx | ptx_intrinsics.cuh:139 | REDUNDANT, 0 callsites | DELETE |
| affine_combine_ptx | ptx_intrinsics.cuh:28 | REDUNDANT (dup of affine2x2.h), 0 callsites | DELETE |
| **fast_rsqrt_nr** | utils.cuh:96 | **LOAD-BEARING** (Newton-Raphson refinement), 1 callsite (supergrok2.h:161) | **KEEP** |
| **ptx_fma** | utils.cuh:106 | **LOAD-BEARING** (FMA fusion guarantee), 14 callsites | **KEEP** |
| ptx_exp2 | utils.cuh:114 | REDUNDANT (internal only) | DELETE |
| ptx_log2 | utils.cuh:121 | REDUNDANT, 0 callsites | DELETE |
| ptx_expf | utils.cuh:128 | REDUNDANT (=expf under --use_fast_math), 15 callsites | REPLACE with expf() |
| ptx_tanhf | utils.cuh:134 | REDUNDANT, 0 callsites | DELETE |
| ptx_sigmoidf | utils.cuh:141 | REDUNDANT, 0 callsites | DELETE |
| ptx_affine_combine | utils.cuh:150 | REDUNDANT (dead wrapper), 0 callsites | DELETE |
| ptx_expert_mlp_forward | utils.cuh:169 | REDUNDANT (never instantiated) | DELETE |
| ptx_int8_stochastic_round | utils.cuh:189 | Load-bearing IF called, but 0 callsites | DELETE |
| cluster_dsmem_reduce_sum | utils.cuh:206 | OUT OF SCOPE (Stage 4.2), 1 callsite | SKIP |

Total dead inline PTX opcodes: ~49. Total dead functions: 11 (beyond expf replacement).

**IMPORTANT:** The ptx_intrinsics.cuh file was noted as located at `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh` in this audit — the REVIEW_0_2.md confirms `csrc/common/ptx_intrinsics.cuh` was DELETED (Stage 3.0, commit bc91f05). So this audit documents the pre-deletion state.

---

## 15. docs/reviews/REVIEW_0_2.md

**Verdict: NO BUGS FOUND. Stage-0 header de-inline + Stage-2 L2 persistence — CORRECT.**

**Key verified facts:**
- `csrc/common/types.h:24-36`: constants byte-identical to pre-de-inline inlined copies (MAX_D_STATE=128, MAX_D_INNER=128, MAX_D_MODEL=64, etc.)
- Include topology: platform (leaf) ← affine2x2 ← types ← utils ← primitives/mma/adapter — acyclic
- `prim::L2PersistScope` RAII helper: constructor has 6 safety gates in order; destructor resets stream window + calls `cudaCtxResetPersistingL2Cache()` + `cudaDeviceSetLimit(..., 0)` — no leak
- All 11 optimizer launchers wire L2PersistScope over their hot momentum state buffers
- Gate results: COMPILE_OK for launch_adamw.cu; `137 passed, 1 failed` self-test (the 1 failure is pre-existing `flag_base_superset_regression`, 0 net regressions); ruff clean

---

## 16. docs/reviews/REVIEW_1A.md

**Verdict: SG2 CSA/HCA bilevel backward adjoint — NO BUGS FOUND. All 24 weight-grad buffers match autograd to fp32 precision (max abs ≤ 6e-9).**

**Method:** Line-by-line Python transcription of the header, compared against `torch.autograd.grad` through `forward_for_bilevel`, 4 configs × 2 GRU states = 8 runs.

**Flagged items (not bugs, documented 🟡):**
1. GRU-gate recompute fallback drops biases when called with empty gate tensors (sm90:1684,1687,1691 / gfx942:875,878,882) — latent hazard if caller violates save-set contract; canonical path saves gates so it's exact
2. Output-buffer zero-init is a caller contract with no guard — bindings do not zero the 24 d_* buffers; correctness relies on caller
3. On-device bit-parity pending (🟡, pre-existing)

**Confirmed:** gfx942 launcher calls identical `sg2adj::bilevel_backward_driver` / `bilevel_forward_save` — math-identical by construction.

---

## 17. docs/reviews/REVIEW_1B.md

**Verdict: All 9 MoE compaction kernels — NO BUGS FOUND. sm_90 ↔ gfx942 math EQUIVALENT for all 9.**

**Important note:** `_moe_step` at `supergrok2.py:2221-2225` currently raises `NotImplementedError` on its first line — the MoE kernel-calling body (2226-2308) is dead at runtime. The review treats the dead body as the binding contract (documents intended invocation). MoE compaction is STUBBED (not runnable).

**Key verified kernels:** count_expert_activations (atomicAdd histogram), compute_load_balance_loss (Switch-Transformer aux loss = E * Σ f_e·P_e), apply_frequency_scaling (inverse-frequency lr scaling), filter_active_params (atomic out-position, 5 arrays kept in sync), scatter_results (exact inverse), dynamic_expert_{fwd,bwd} (2-layer ReLU MLP + VJP), scan_compacted (vestigial SSM, never called).

---

## 18. docs/reviews/REVIEW_1C.md

**Verdict: All 11 decoder+ViT tensor-core GEMMs — NO BUGS FOUND. All `x·Wᵀ` convention verified.**

- Generic helper `mma::sm90_run_gemm_bt` (mma.cuh:193-284): byte-identical tile config to proven `fmha_sm90_gemm`
- 5 decoder matmuls all `ColumnMajor` + B physically row-major = `x·Wᵀ`; cuBLAS `#else` fallback computes identical math
- 6 ViT matmuls via `vit_linear_gemm_bias` — same convention with fused bias
- `from_float<T>` forward-declaration at transformer_decoder_sm90.cuh:113 correctly fixes a genuine ordering issue
- `#ifdef WITH_CUTLASS` gating: FP32 never instantiates CUTLASS GEMM; `ActT == WeightT` enforced; no-CUTLASS fallback correct
- Gate: `137 passed, 1 failed` (same 1 pre-existing failure)

---

## 19. docs/reviews/SG2_BACKWARD_SPEC.md

**Purpose:** Implementation spec for the SG2 CSA/HCA bilevel backward adjoint.

**Forward pipeline:** input_proj+sort → CSA → HCA → GRU → PEER → smart_grad

**24 weight-grad output buffers:** d_input_proj_{W,b}, d_csa_{q,k,v,out}_W + d_csa_compress_w + d_csa_idx_{DQ,UQ,K}, d_hca_{q,k,v,out}_W, d_gru_{Wz,bz,Wr,br,Wh,bh}, d_peer_query_Ws, d_prod_keys_{A,B}, d_expert_{W1,b1,W2,b2}

**Key notes:**
- Reusable primitive: `sg2_bilevel_precompute_timestep` (supergrok2.h:365) recomputes per-row Q/K/V/QI projections avoiding save
- checkpoint_interval ≤ MAX_CKPT_INTERVAL=32 (types.h constant)
- Numerics oracle: compare d_*_W against torch.autograd.grad for N=20 rows, rtol=1e-3/atol=1e-5 (hardware-deferred)

---

## 20. examples/toy_tune_project/

**Purpose:** Portability proof harness — a non-grokking SAXPY project to demonstrate compile.py can tune a foreign kernel with zero edits.

**Files:**
- `toy_kernel.cu`: SAXPY `y = a*x + b` as `torch.ops.toy.saxpy`, tunable block size via `#ifndef TOY_TUNED_BLOCK` (default 256, static_assert ≤ 1024)
- `tune_hook.py`: implements portable hook contract `run(*, so_path, model, optimizer, arch, regime, seed) -> {"output": np.ndarray, "elapsed_ms": float}`
- `compile_config.toml`: `macro_prefix = "TOY_"`, `tune_hook = "examples.toy_tune_project.tune_hook:run"`
- `toy_search_space.yaml`: one dim `block` → macro `TOY_TUNED_BLOCK` over `[128, 256, 512, 1024]`

**Status:** All individual components verified. End-to-end `python -m grokking_optimizers.compile` is BLOCKED by three portability gaps in compile.py:
- **Gap 1** (decisive): `_make_variant_timer` + `_validate_against_regimes` don't consult `spec.tune_hook` — hardwired to grokking optimizer class
- **Gap 2**: `_resolve_sources` unconditionally globs `csrc/fused/<arch>/*.cu` (hardcoded), preventing Tier-1 auto-glob from finding `toy_kernel.cu`
- **Gap 3**: `_validate(spec)` hard-checks optimizer against `OPTIMIZERS` profile constants, rejects foreign names → `ValueError: optimizer='toy_saxpy' not in ['adamw', ...]`

The **hook timing+validation seam (Gap 1) is documented as CLOSED** in the RESULT.md but NOT in the README — `_make_variant_timer` routes through `_hook_capture + _compare_outputs when spec.tune_hook is set` (compile.py:15307-15397). Gap 3 fires FIRST (before hook routing is reached) per `run_autotune.py`'s docstring confirming Gap 3 still blocks end-to-end.

---

## 21. examples/autotune_demo/

**Purpose:** Demo of compile.py's autotuner machinery over a naive tiled fp32 GEMM with 4 knobs (`SG_TUNED_TILE_M/N/K`, `SG_TUNED_BLOCK`).

**Key files:**
- `gemm_kernel.cu`: parameterized tiled GEMM, default tile 16×16×16/block=256 (textbook slow). 4 `#ifndef SG_TUNED_*` guards. Full boundary guards + static_asserts (BLOCK divides TILE_M*TILE_N; smem ≤ 48KB).
- `run_autotune.py`: uses compile.py's internal functions directly (Gap 3 workaround): `C.load_config`, `C.get_search_space`, `C.cartesian`, `C.resolve_macros`, `C._hook_capture`, `C._compare_outputs`, `C.config_key`
- `build_variant.py`: builds one variant .so per subprocess (TORCH_LIBRARY namespace collision avoidance)
- `tune_hook.py`: portable hook returning output + elapsed_ms
- `compile_config.toml` + `gemm_search_space.yaml`: 24-config search space (4×3×2×1)
- `RESULT.md`: documented run results

**Actual results (from RESULT.md):**
- 24/24 configs tried, all pass correctness gate (status="deterministic", max_rel=0.0 — bit-identical to strict reference)
- Winner: `tile_m=32, tile_n=16, tile_k=16, block=256` at 4.786 ms vs naive 4.970 ms = **1.04x speedup**
- Worst config: `tile_m=128, tile_n=64, tile_k=32, block=256` at 14.703 ms (3x slower)
- cuBLAS baseline: 0.347 ms (14.3x faster than any config)
- Sweep wall time: 1090s (dominated by nvcc compiles, H100 shared with other process)

**Honest interpretation:** Small gain (1.04x) because the "naive" default (16×16) is already near optimal for this kernel's expressive space. Larger tiles blow register pressure and spill. The demo proves correctness validation works (24/24 gate-PASS) but does not demonstrate large leverage since the knob space lacks vectorized loads, double-buffering, or tensor cores.

---

## Summary: config-derivation / adaptivity observations

The scripts reveal the OUTER layer of the self-adapting design:

1. **check_math_single_source.py** enforces the single canonical implementation — every optimizer variant (fused megakernel, per-op, gfx942) MUST derive from `csrc/algorithms/<opt>.h`. This is a CODE-LEVEL enforcement of the "one math source" design principle.

2. **fast_triage.py** reveals the ACTUAL phase table for the decoder: `[P1_fwd, P1_bwd, B1_barrier, P2_dW_GEMM, P2_grad_asm, P3_opt_tail, B2_barrier, B0_barrier]` (8 slots, `g_dec_prof_max`). ViT has 6 slots. Mamba has NO per-phase profiler yet. The relevance gate (Amdahl floor 5%) is the adaptive machinery that decides which kernel change matters at the current roofline bottleneck.

3. **roofline_bench.py** reveals the roofline constant: **H100 989 TF/s** (line 169). `pct_roofline = achieved_tf_s / 989.0 * 100`.

4. **nvcc_baseline.py** documents that compile.py's ptxas micro-tuning flags include: `--register-usage-level`, `--def-load-cache=ca`, `--def-store-cache=wb`, `--extra-device-vectorization`, `--maxrregcount`, `-Xfatbin -compress-all`. These are the CUDA-version-gated augmentations.

5. The **toy_tune_project README** documents that compile.py is NOT yet a general-purpose portable tuner for foreign projects — 3 concrete gaps remain. Gap 1 (hook timing seam) is claimed closed in RESULT.md, contradicting README which says "the decisive blocker."

---

## Discrepancies and open items

1. **Gap 1 closure inconsistency:** README (toy_tune_project) says "the tune-hook seam is only HALF-wired (the decisive blocker)" but run_autotune.py's docstring says "Gap 1 is closed: `_make_variant_timer` routes through `_hook_capture + _compare_outputs` when `spec.tune_hook` is set (compile.py:15307-15397)." Gap 3 (`_validate` rejects foreign optimizer names) fires FIRST anyway and blocks the end-to-end path regardless.

2. **MoE path is dead (NotImplementedError):** REVIEW_1B confirms `_moe_step` raises `NotImplementedError` at line 2221-2225. MoE kernels exist and are reviewed as correct, but the path is unreachable at runtime.

3. **ptx_intrinsics.cuh deletion:** STAGE3_PTX_AUDIT.md recommends deleting the file. REVIEW_0_2.md notes it was deleted in Stage 3.0 (commit bc91f05). The audit's path `/home/user/SuperGrok1.5/csrc/common/ptx_intrinsics.cuh` suggests it was on a different host (old session path), consistent with deletion having happened.

4. **SG2 GRU backward biases:** REVIEW_1A flagged that the recompute fallback drops GRU gate biases (sm90:1684,1687,1691). This is an ABI issue that requires an invasive fix — remains 🟡 open.

5. **Mamba profiler not wired:** `fast_triage.py` `_MODELS["mamba"]` is `wall_only=True` — no per-phase clock64 profiler exists for Mamba yet. Any Mamba triage is DIRECTION-ONLY.
