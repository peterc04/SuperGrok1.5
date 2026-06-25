# Python Root Scripts Analysis — SuperGrok1.5
## Slice: PY_repo_root_scripts (excluding compile-brain files)
## Agent: py_root_scripts digest | Date: 2026-06-25

---

## Files Analyzed

| File | Lines | Role |
|------|-------|------|
| grokking_race_v2.py | 2584 | Central 11-optimizer benchmark race harness |
| bench_backends.py | 729 | Build/profile/benchmark backend tool |
| _probe1.py | 42 | Single mamba optimizer probe |
| _sg_realsg_probe.py | 112 | REAL-SG fidelity verification probe |
| compile_config.toml | 183 | Autotune/build config (mirrors compile.py defaults) |
| pyproject.toml | 20 | Package metadata (grokking-optimizers v3.0.0) |
| ruff.toml | 82 | Ruff lint config |
| _build_snapshot.sh | 104 | Repo snapshot creation script |
| build.sh | 195 | Top-level build wrapper (debug/profile/package modes) |
| .smoke12_driver.py | 85 | Compile.py #12 defect acceptance test |
| _fg_runner.py.disabled | 89 | DISABLED foreground tuner runner |
| .env_requirements.txt | 180 | pip freeze (env reference) |
| ENV_SNAPSHOT.txt | 180 | pip freeze (same content, env snapshot) |
| .fast_build_env.sh | 46 | Build acceleration env (sccache+ccache+nvcc threading) |
| MANIFEST.in | 36 | sdist manifest (csrc/ kernel tree) |
| .github/dependabot.yml | 18 | Dependabot config (weekly pip + actions updates) |
| .github/workflows/ci.yml | 1604 | Full CI pipeline (35+ jobs) |
| .github/workflows/codeql.yml | 25 | CodeQL security scan |
| .task11_perf_authoritative.log | 20 | Task #11 GEMM perf results (AUTHORITATIVE) |
| _build_snapshot.log | 22 | Build snapshot log |
| build.log | 524 | Last build log (editable meta-path install) |
| .opt_candidates.json | 531 | Compile.py optimization candidates (catalog) |
| .opt_candidates_broad.json | 768 | Broader compile.py opt candidates |

---

## 1. grokking_race_v2.py — The Central Race Harness

### 1.1 Architecture

Three models × 11 optimizers × 4 splits (10/25/50/80). All training is PURE L3-TC — every train_* loop calls `_try_fused_train_step()` exclusively, which RAISES on any degradation condition.

**Model specs (grokking toy tasks):**
- Decoder Transformer: `(a ÷ b) mod 97`, 4-token seq
- ViT: MNIST `(a + b) mod 97`, 16 patches + CLS
- Mamba SSM: `(a÷b₁÷b₂÷b₃) mod 97`, 8-token chain

**Flagship (roofline/benchmark) model dims** — `MODEL_SCALES_BY_MODEL["flagship"]` (line 249-257):
- decoder: d=1600, h=25, L=48 (GPT-2 XL, ~1.5B)
- vit: d=1664, h=16, L=48 (ViT-G/14, ~1.8B)
- mamba: d=2048, h=32, L=24, state=128, head_dim=64, expand=2, mlp_ratio=2 (~1.5B)

**Toy (race) model dims** — `MODEL_SCALES`:
- small: d=128, h=4, L=2 (~420K params)
- medium: d=256, h=8, L=4 (~3.5M)
- large: d=512, h=8, L=6 (~20M)

### 1.2 PURE L3-TC Design (Line 862-881, 984-1060)

`_try_fused_train_step()` (line 984) is the SOLE execution path. It RAISES RuntimeError with "L3-TC unavailable; refusing to silently downgrade" in ALL fallback scenarios:
- `use_fused=False` → raises
- `has_l3_real(model_name, opt_name)` False → raises
- `use_amp=True` → raises (AMP incompatible with fp32-param megakernel)
- `gemm_impl_for_cell()` returns None → raises
- stale ABI (pybind TypeError) → raises

`LAST_L3_ENGINE` global (line 970): set to `dict(engine=gemm_impl, model=..., precision=..., path=...)` on successful L3-TC step; reset to None at run start. `_record_train_path()` (line 852) stamps `r.train_path` from this; HARD RAISES if None after a run.

**L3 path labels** (line 973-982): "L3-TC bf16" (production), "L3-scalar" and "L1+eager" are legacy inert labels only.

### 1.3 Precision Handling

`_AUTO_PRECISION` (line 725):
```python
{"decoder": "bf16", "vit": "fp32", "mamba": "fp32"}
```
ViT defaults to fp32 (precision analysis showed bf16 lost 2/3 seeds). Decoder defaults bf16. Mamba fp32 until measured.

`_autocast()` (line 733): bf16 precision triggers `torch.amp.autocast('cuda', dtype=torch.bfloat16)`; tf32/fp32 is `contextlib.nullcontext()`.

### 1.4 EarlyStopper (Lines 585-639)

Two modes:
- `"acc"` (default): threshold 0.95, patience 50 evals (= 500 steps at eval_every=10)
- `"loss"`: plateau stop for LM/forecasting; `plateau_patience` evals with no improvement

DEFAULT_CONFIG `eval_every=10` (not 100). The argparse default was fixed from 100→10 (note: `[A4-H1]`).
EarlyStopper patience counts EVALS, not steps: 50 evals × 10 = 500-step post-grok hold.

### 1.5 Data Pipeline

`make_data_for_task()` dispatches:
- `data_source="modular"` → in-memory mod-p grokking (legacy, byte-identical)
- Any other value → `grokking_optimizers.dataset_sources.make_source_for_task()` (streaming)

**Known data sources config**: `"modular" | "fineweb_edu" | "imagenet1k" | "gifteval" | "synthetic"`

### 1.6 Model Loading — Mamba3

Line 540: Mamba uses `Mamba3Model` from `grokking_optimizers.mamba3_block` (NOT the local `MambaModel` class at line 507-518 which is the Mamba-1 fallback).

### 1.7 Optimizer Config Details

Key per-optimizer config decisions at OPTIMIZER_CONFIGS (line 2277):

**NeuralGrok** (lines 2282-2308): `neural_layers=2, neural_hidden=16` — CRITICAL: forced for kernel parity. The NeuralGrok class default is 128. The CUDA kernel evaluates a 2-layer MLP with kPsiHidden=16 (opt_components.cuh:61/225). Old config used layers=3, hidden=128, creating train/deploy divergence (the "A3 defect").

**SuperGrok2** (line 2341): `sg2_d_model=8, sg2_num_experts=144` — audit fixed silently ignored keys that made SG2 always run constructor defaults. These are now the correct keys.

**SuperGrok1.1/1.5**: `lamb=1.0` (SG11) / `lamb=2.0` (SG15) identity/multiplier defaults.

### 1.8 Multi-GPU Support

`run_pipeline()` (line 1887): `use_multi_gpu=True` when `len(gpu_ids) > 1 AND n_gpus >= 2`. Workers pool tasks from a shared `MPQueue`; each GPU gets its own process via `mp.Process`. Crash stubs (`_crash_stub`) ensure failed runs are counted not dropped.

### 1.9 SuperGrok Bilevel Side-Steps on L3 Path

Lines 1268-1297 (SG11), 1364-1393 (SG15): On the L3 path, bilevel meta-net training (autograd-dependent) still runs HOST-SIDE. The kernel cannot run bilevel autograd, so:
1. Eager fwd/bwd seeds `p.grad` (meta-featurization only)
2. `sam_step` + `meta_step`/`bilevel_step` train the meta-net on the host
3. L3 kernel re-extracts fresh phi pack from meta-net before launch

### 1.10 TrainResult Fields

`train_path` (line 688): the ACTUAL executed path recorded for wiring guard.
`grokking_step_test_confirmed` (line 710): confirms grok transferred to TEST split, guards against val-only meta-net circular grok.
`best_metric_acc` (line 712): best criterion-accuracy seen by stopper.
`component_failures` (line 709): per-component failure counts (sam/meta/bilevel).

---

## 2. bench_backends.py — Backend Build+Profile+Benchmark

### 2.1 Purpose

Two modes (`--mode bench|profile`):
- `bench` (default): measures fused vs ATen-baseline latency on realistic shapes
- `profile`: runs ncu/rocprof/jax profiler dumps per launcher

### 2.2 Backend Detection (line 114)

Auto-detects: cuda (H100 sm_90), hip (gfx942), pallas (TPU). Falls back to cuda.

### 2.3 Benchmark Workloads (line 339)

Realistic transformer-ish shapes (NOT 64x64):
```python
("tiny",   [(256, 256), (256,)]),
("small",  [(1024, 1024), (1024,), (4096, 1024), (4096,)]),
("medium", [(4096, 4096), (4096,), (4096, 11008), (11008,)]),
("wide",   [(8192, 2048), (2048, 8192), (8192,), (2048,)]),
```

### 2.4 Timing Methodology (line 356)

CUDA Events (`torch.cuda.Event(enable_timing=True)`) for GPU, `time.perf_counter` for CPU. Reports median_ms, p90_ms, min_ms, throughput (Melem/s). Speedup vs ATen baseline for adamw and lion.

### 2.5 NCU Flags (line 91)

```
--set full --target-processes all --import-source yes
--section ComputeWorkloadAnalysis LaunchStats MemoryWorkloadAnalysis
         SchedulerStats WarpStateStats InstructionStats Occupancy
```

### 2.6 Build Integration

`build_extension()` (line 149): `pip install -e . --no-deps --force-reinstall -v` with `MAX_JOBS`, `TORCH_CUDA_VERBOSE_BUILD=1`, `NVCC_APPEND_FLAGS` for diagnostics.

---

## 3. _probe1.py — Mamba Optimizer Probe

Simple script: runs `T.run_trial_config(opt, "mamba", {}, TUNING_SEED)` against `tuning/tune_optimizers.py`. Writes results to `/workspace/mamba_probe_results.json`. Used to probe individual optimizers under the step-cap budget.

---

## 4. _sg_realsg_probe.py — REAL-SG Fidelity Probe

**Purpose**: Verifies SG11/SG15 run as REAL SuperGrok (not degenerate AdamW) on the L3 path after the meta-net training fix.

**Instruments** (line 43-50):
- `meta_net.rescale` — must move off 0
- `||mu||` (meta correction norm) — must be nonzero
- `LAST_L3_ENGINE` — must show wgmma kernel
- Accuracy trajectory — must show memorize-then-collapse dynamics (NOT grokking)

**Honest statement** (line 10): "the goal is FIDELITY (real algorithm), not grokking. We REPORT the trajectory."

**Mechanism**: Monkey-patches `SharpnessMetaNet.forward` to record correction norms, then runs the real `train_supergrok`/`train_supergrok15` with eval callbacks.

---

## 5. compile_config.toml — Build Configuration

**Version**: project = "supergrok" v2.0.0 (note: pyproject says 3.0.0)

**Tuned macro floor** (lines 171-182):
```toml
mega_block    = ["SG_TUNED_MEGA_BLOCK", 256]
tile_m        = ["SG_TUNED_TILE_M", 128]
tile_n        = ["SG_TUNED_TILE_N", 128]
dec_dw_splitk = ["SG_TUNED_DEC_DW_SPLITK", 4]
vit_dw_splitk = ["SG_TUNED_VIT_DW_SPLITK", 4]
mb_dw_splitk  = ["SG_TUNED_MB_DW_SPLITK", 4]
prod_regs     = ["SG_TUNED_PROD_REGS", 40]
cons_regs     = ["SG_TUNED_CONS_REGS", 232]
maxrregcount  = ["--maxrregcount", 0]   # 0 == unset
```

Note: The task11 perf log shows split-K=2 (variant C) is FASTER than split-K=4 (compile.py default). The config still defaults split-K=4 for all three models.

**Features OFF by default**: `synth_codegen.enable=false`, `runtime_specialization.enable=false`, `device_pgo.enable=false`, `polyhedral.enable=false`, `cost_model.enable=false`

**NDEBUG**: `-DNDEBUG` is an explicit project decision recorded here, NOT compile.py's base device flags.

---

## 6. .task11_perf_authoritative.log — CRITICAL BENCHMARK

Task #11 decoder GEMM perf benchmark (2026-06-16 13:50):
- Model: d=2048, B=4096, total_params=101,134,435 (~101M, NOT the 1.5B flagship)
- 7 reps per seed, seeds {42, 7, 123}

| Variant | Desc | split-K | Median ms | TF/s | % Roofline |
|---------|------|---------|-----------|------|------------|
| A | regular-nvcc | 4 | 500.5ms | 19.785 | **2.001%** |
| B | compile.py-default | 4 | 502.6ms | 19.701 | **1.992%** |
| C | compile.py-tuned | **2** | 492.1ms | 20.121 | **2.034%** |

**CRITICAL**: ~2% of roofline TF/s is EXTREMELY LOW. The H100 theoretical peak for bf16 is ~989 TF/s; achieving 20 TF/s on a 101M-param model suggests fundamental underutilization. The wgmma megakernel is NOT approaching roofline efficiency.

Autotuner gain A→C: -1.667% faster (from 500ms → 492ms). Very marginal.

---

## 7. build.log — Last Build State

The latest `build.log` (524 lines) shows a **pure editable meta-path install** with NO CUDA compilation:
- `pip install -e .` → setuptools editable install
- Output: `grokking_optimizers-3.0.0-0.editable-cp311-cp311-linux_x86_64.whl`
- Uses "meta path finder" strategy (no actual extension build)
- Python 3.11, torch 2.4.1+cu124

**IMPLICATION**: The CUDA extension (_ops*.so) was NOT rebuilt in the last build visible in build.log. This editable install alone does not compile the C++/CUDA code.

---

## 8. build.sh — Build Wrapper

Features:
- `--debug`: `CUDA_DEBUG=1` → disables fast-math, enables asserts
- `--profile`: runs ncu after build
- `--package`: stages dist/ tree (Python sources + compiled .so + pyproject.toml)
- `--package-tarball`: adds `supergrok2-VERSION-SHA.tar.gz`
- `--autotune` / `--no-autotune`: REMOVED (old two-pass flow called nonexistent `autotune/tune.py`)
- tqdm progress filter for ninja "[N/M]" lines
- Sources `.fast_build_env.sh` for sccache/ccache/nvcc threading

---

## 9. .fast_build_env.sh — Build Acceleration

**Mechanisms**:
1. `PYTORCH_NVCC="nvcc-cached"` → routes nvcc through sccache (device compile cache)
2. `CXX="g++-cached"` → routes g++ through ccache (host compile cache)
3. `nvcc --threads 0` auto-detects CPU count for parallel gencode phases
4. `MAX_JOBS=$(nproc)` for ninja parallelism
5. `TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1` unblocks sccache
6. Symlinks wrapper to `/workspace/.local/bin/g++-cached` (path fix for torch's ABI probe)
7. Cache dirs are VOLUME-BACKED (survives session teleport): `$ROOT/.build_cache/sccache`, `$ROOT/.build_cache/ccache`

---

## 10. CI Pipeline (.github/workflows/ci.yml — 1604 lines)

35+ jobs covering: lint (ruff), py_compile, import_smoke, self_test, dry_run_all_archs, single_arch_dry_run, cli_surface, package_metadata, macos_smoke, lint_full, cpp_structural, functional_smoke, codegen_matrix, exhaustive_mode, import_matrix, optimizer_construct, functional_metanets, determinism, docs_consistency, coverage, arch_dispatch, compile_to_object, amdgcn_check, drift_guard, verify_all, profile_maximal, pytest_cpu, pytest_hw_gate, typecheck, security_bandit, deps_audit, codespell, yamllint.

Uses CPU-only torch wheels (`PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu`) for most jobs. No GPU hardware in CI.

---

## 11. .opt_candidates.json — Compile.py Optimization Catalog

531-line catalog of compile.py optimization candidates (NOT benchmark results). Key findings:
- High-value candidates: JSON cache compact serialization, per-trial build-sig hash hoisting, per-arch nvcc version probe caching
- **CRITICAL FINDING**: "An entire cluster of tile_pipeline.cuh candidates target `tc_pipelined_gemm_m64nNk16 / pipeline_produce_ktile / TilePipeline`, which grep confirms are SELF-TEST-ONLY and never called by any production megakernel" — dead code
- The real production GEMM staging is `tc_gemm_block_unpipelined` (ring-based cp.async)
- Most SuperGrok2 candidates excluded as parity risks

---

## Key Discrepancies vs. CLAIMED State

### D1. PURE L3-TC vs. Build State
CLAIMED: L3-TC megakernel runs everything. ACTUAL: `build.log` shows latest build was an editable meta-path Python-only install — NO CUDA compilation. If the _ops*.so extension is stale or missing, EVERY training call raises "L3-TC unavailable; refusing to silently downgrade". The race would crash on all 11 optimizers.

### D2. 2% Roofline (Critical)
The .task11_perf_authoritative.log shows 2.001-2.034% of roofline TF/s for the decoder megakernel on a 101M-param model (d=2048, B=4096). The CLAIMED state describes a CuTe-atom GEMM engine validated as "SG_TUNED_GEMM_ENGINE" bit-identical. Even if correct numerically, 2% roofline efficiency means the kernel is not close to compute-bound. The 33-cell roofline deliverable (#1 remaining) would show mostly very low % numbers.

### D3. Split-K Configuration
Variant C (split-K=2) is 2.1% faster than variant B (split-K=4, the compile.py default). But `compile_config.toml` defaults all three models to `*_dw_splitk=4`. This means the autotuned winner (split-K=2) is NOT the default in the config file — the config file shows the pre-tuning default.

### D4. NeuralGrok hidden_dim Discrepancy
`OPTIMIZER_CONFIGS["neuralgrok"]["neural_hidden"] = 16` (for kernel parity) but `NeuralGrok.__init__` default is `hidden_dim=128`. Any direct construction of NeuralGrok outside this config uses a different architecture than deployed in the race. This is a known/documented issue (the A3 defect fix), but remains a footgun.

### D5. SuperGrok2 CSA Lightning-Indexer Gap
Lines 1443-1448 explicitly document that SG2's in-kernel CSA indexer diverges from the eagerly-trained net for N>64 (idx_UQ drop + scores /sqrt(rank) not /sqrt(d)). SG2 achieves "L3-TC + single-step parity" but "WON'T GROK". This is a known fidelity gap, not a bug per se, but it's documented in the code as an honest caveat.

### D6. ViT Precision Default
DEFAULT_CONFIG has `"matmul_precision": "bf16"` globally, but `_AUTO_PRECISION["vit"] = "fp32"`. ViT runs fp32 in auto mode, contradicting the "bf16 mixed precision everywhere" claim in the config comments.

### D7. SG11 train_path Claim and Removal of Eager Paths
Comments in train_adamw (line 1076) say "Falls back to the eager fwd/bwd + L1 fused tail when the L3-REAL kernel is unavailable" — but the actual code below RAISES instead. The comment is a STALE DESCRIPTION from before the eager paths were removed.

---

## State Assessment

| Component | State |
|-----------|-------|
| grokking_race_v2.py | In-flight; PURE L3-TC enforced; many commented-out older paths removed |
| bench_backends.py | Done/validated; functional benchmark harness |
| _probe1.py | Validated (mamba tuner probe) |
| _sg_realsg_probe.py | Done; verifies REAL-SG fidelity (meta-net moves off 0) |
| compile_config.toml | Done; documents tuned floor but split-K default may be stale (task11 says 2 is faster) |
| build.sh | Done; autotuning decoupled; package mode works |
| .fast_build_env.sh | Done; volume-backed caching |
| .task11_perf_authoritative.log | Done (2026-06-16 13:50); shows ~2% roofline |
| CI (ci.yml) | In-flight; 35+ jobs, no GPU runners |
| build.log | Shows editable meta-path install only; CUDA extension build state unclear |

---

## Open Items / Bugs / TODOs

1. **CUDA extension build status**: build.log shows meta-path editable install only. Need to confirm _ops*.so exists and is current. If missing, PURE L3-TC mode crashes every run.

2. **~2% roofline efficiency**: The flagship claim is 100% tensor-core utilization; actual task11 shows 2% for d=2048. The 33-cell roofline deliverable (RESUME.md #1 remaining) hasn't been run yet for the flagship sizes.

3. **split-K default vs. tuned winner**: Config defaults split-K=4; task11 shows split-K=2 is 2% faster. Should update compile_config.toml to reflect the actual tuned winner.

4. **TP data-path fix (RESUME.md #1)**: Bugs A (per-rank weight-shard offset) and B (25-heads-not-%8 attention) fixed in patch but unmerged. Bug C (IMA) unconfirmed.

5. **SG2 CSA fidelity gap**: SG2 won't grok due to in-kernel CSA divergence. Documented as honest caveat but may affect reported results.

6. **Stale comment in train_adamw**: Line 1076 says "Falls back to eager" but code raises. Comment is stale.

7. **VERSION mismatch**: compile_config.toml says project version "2.0.0"; pyproject.toml says "3.0.0".

8. **NeuralGrok direct-construction footgun**: Class default hidden_dim=128 != race config 16. Easy misconfiguration if used outside OPTIMIZER_CONFIGS.

9. **ViT precision auto-fp32**: DEFAULT_CONFIG says bf16 but ViT auto-resolves to fp32. May confuse performance analysis expecting bf16 everywhere.

10. **SG2 mamba blocked**: Comments say "mamba×SG2 is BLOCKED: the SAM double-forward + segmented sort re-trip the shared mamba-forward A/A/A race". SG2 cannot run with mamba model.
