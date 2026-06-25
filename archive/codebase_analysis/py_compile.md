# compile.py Deep-Read Digest

**File:** `/workspace/SuperGrok1.5/grokking_optimizers/compile.py`
**Lines:** 32,900
**Role:** Autotuner + JIT/AOT build brain for the SuperGrok2 L3-TC megakernel stack.

---

## 1. Module Overview (lines 1–123)

Two-phase pipeline: **AOT** (CPU-only, no GPU required) compiles a baseline `.so` from the C++ source tree; **JIT** (GPU required) runs the Bayesian search sweep over the per-arch search space to find the optimal hyperparameter combo, writes the winning config to `csrc/algorithms/tuned_configs.h`, and records the result in the JSON cache. Public entrypoints: `build()`, `build_aot()`, `build_jit()`, importable as shown in the module docstring.

---

## 2. Imports & Module Constants (lines 124–300)

- **optuna / TPESampler** imported at module level; yaml, fcntl for locking.
- Imports from `grokking_optimizers.profile` and `grokking_optimizers.dispatch`.
- `_COMPILE_LOG_LEVEL = 0` (line 219): 0=silent, 1=trace on debug, 2=always. Bumped by `--debug`/`--debug-flags` CLI.
- `_debug_swallow()` (lines 222–238): broad-except swallowing only visible at log level 2.
- `_PINNED_VERSIONS` (lines 291–296): scikit-learn>=1.5,<2; jinja2>=3.1,<4; libclang>=16,<19; cuda-python>=12.2,<14.
- `_AUTO_INSTALL_OPTIONAL_DEPS = False` (line 388): opt-in; env `GROK_AUTO_INSTALL` overrides.

---

## 3. Architecture Table (lines 433–1130)

### ArchEntry Dataclass (lines 433–486)
Fields: `vendor`, `display_name`, `subdir`, `launcher_glob`, `model_glob`, `macro`, `host_define`, `min_toolchain_version`, `arch_suffix`, `nvcc_gencode`, `hipcc_offload_arch`, `cutlass_arch`, `max_smem_per_block`, `warp_size`, `max_regs_per_thread`, `max_threads_per_block`, `features`, `l2_bytes`, `peak_tf`, `peak_bw`, `search_space_builder`, `has_kernel_body`.

### _ARCH_TABLE_PRIMARY (lines 552–1116)
25 canonical entries covering sm_70..sm_120a (CUDA), gfx906..gfx1201 (HIP/AMD), tpu_v4..tpu_v7 (Pallas).

**H100 (sm_90a) entry:** min_toolchain=(12,0), max_smem=228KB, l2_bytes=50MB, peak_tf=989.4e12, peak_bw=3.35e12, features include `{wgmma, tma, cluster, fp8, fp4}`.

### _ARCH_TABLE_ALIASES (lines 1125–1130)
sm_90→sm_90a, sm_100→sm_100a, sm_103→sm_103a, sm_120→sm_120a.

- `_SASS_ONLY_GENCODE = False` (line 505): toggled by `--sass-only` CLI.
- `_nvcc_gencode_pair()` (lines 508–527): emits SASS + PTX fallback gencodes.

---

## 4. Tuning Dimension System (lines 1300–1715)

### Live Tuning Dimensions
`_LIVE_TUNING_DIMS` (lines 1300–1303): frozenset = `{block, vec, unroll, async_depth, cluster_shape, maxrregcount, tile_m, tile_n, dec_dw_splitk, vit_dw_splitk}`.

`_ASYNC_DEPTH_MAX = 4` (line 1309): was 16, reduced after profile analysis showed [1,16] wasted budget.

`_PTXAS_FLAG_DIMS` (lines 1329–1333): `{maxrregcount, opt_level, allow_expensive_opts, def_load_cache, register_usage_level}`.

`_MAXRREGCOUNT_UNCAPPED = -1` (line 1713): sentinel meaning "omit `--maxrregcount` flag entirely" (genuinely uncapped); distinct from =255.

**Removed dimension:** `mb_dw_splitk` removed 2026-06-17 (Mamba-3 TC rewrite).

### Kernel Source Scan (lines 1348–1451)
`_kernel_source_macros()` (lines 1348–1378): scans `csrc/` + `grokking_optimizers/kernels/` for `SG_TUNED_*` tokens; memoized.

`_scan_kernel_ifndef()` (lines 1385–1451): scans `#ifndef SG_TUNED_X` guards and their `#define` defaults.

### Dead Dim Detection
`_auto_derived_dims()` (lines 1490–1536): adds pinned dim for each kernel knob lacking explicit spec.

`_is_dead_dim()` (lines 1546–1577): pallas kwargs always live; bare compiler flags always live; macro dim live iff kernel reads it.

`_DEAD_KEY_DIMS_CACHE` (line 1591): module-level cache dict.

`_dead_key_dims()` (lines 1626–1658): derives dims safe to drop from config_key (memoized).

`_pin_dead_dims()` (lines 1661–1689): collapses dead dims to single value; labels live=True/False.

---

## 5. Search Space Construction (lines 2122–2771)

`_sm90_full_space()` (lines 2122–2327): Hopper space: tile_m=[128,64,256], tile_n=[128,64,256], dec_dw_splitk=[1,2,4,8], vit_dw_splitk=[4,1,2,8], prod_regs=[40,24,32,48,56], cons_regs=[232,168,184,200,216,240], mega_block=[256,512], grad_tile=[1024,2048].

`_gfx942_full_space()` (lines 2373–2428): CDNA3 MI300X search space.

`megakernel_cell_search_space()` (lines 2684–2707): per-cell register-cap sweep seeded from megakernel solver estimate.

`build_full_search_space()` (lines 2735–2771): returns complete per-arch space dict covering all ARCH_TABLE entries.

---

## 6. Source Hashing & Config Key (lines 2804–3483)

`_INCLUDE_WALK_CACHE` (line 2804): mtime-keyed transitive include closure cache.

`_hash_sources()` (lines 2965–2977): hashes TUs + transitive includes (P0.1 fix — was TUs-only → stale cache hits).

`config_key()` (lines 3468–3483): stable key excluding dead dims (dead dim exclusion is now derived, not a static frozenset).

---

## 7. Flag Resolution (lines 3276–3455)

`resolve_macros()` (lines 3276–3309): `-DFOO=VAL` emission; tuple dims → per-component scalars.

`resolve_extra_nvcc_flags()` (lines 3312–3409): maxrregcount→`--maxrregcount=N`, opt_level, allow_expensive_opts, def_load_cache, register_usage_level; arch feature macros.

`resolve_extra_hipcc_flags()` (lines 3412–3455): HIP analog, `-mllvm -amdgpu-max-num-vgprs=N`.

---

## 8. GPU Clock Locking (lines 3757–3896)

`_GpuClockLock`: nvidia-smi `-lgc`/`-lmc` / rocm-smi `--setperflevel`; restores on exit via `__exit__`. Mandate #1.

---

## 9. Timing Infrastructure (lines 3925–5205)

### cuda_graph_median_ms() (lines 3925–3981)
CUDA graph replay timing with L2 flush (Mandate #2 — cold-cache timing).

### _WORKER_BODY (lines 4041–4307)
Embedded Python subprocess worker script; protocol is JSON over stdin/stdout.

### TimingWorker (lines 4525–4871)
Persistent subprocess holding warm CUDA context. Watchdog: 30s interval, 60s grace.

**MED.12 heartbeat stamp** (line 4765): stamps pong BEFORE acquiring io_lock to prevent false watchdog trigger on slow-but-alive worker.

Methods: `calibrate()`, `ping()`, `time()`.

### MultiGPUTimingPool (lines 4874–5205)
Fan-out timing pool across all visible GPUs:
- `_calib_factors` per worker (Mandate #5 — per-GPU calibration via reference matmul normalization).
- Work-stealing via `queue.Queue` + dispatcher threads per device.
- `_normalize_result()` for cross-GPU ranking.

---

## 10. Cost Model (lines 5333–5975)

### Feature Dimensionality (lines 5333–5338)
`FEATURE_DIM`: one-hots(canonical dims) + 11 numeric + 13 arch-features + 14 stall-reasons = total FEATURE_DIM.

### CostModel (lines 5550–5864)
Backend priority: XGBoost → sklearn → `_LinearRidgeRegressor`. Bootstrap K=5 mini-models; quantile heads on xgboost. Joblib/pickle persistence.

`_COST_MODEL_COLD_START_FLOOR = 100` (line 5547): pruning only activates after 100 trials.

`_cost_model_train_from_trials()` (lines 5914–5975): training helper with P1-#23 ptxas spill features (spill_stores/stack_frame) from ptxas-v analysis.

---

## 11. Bayesian Autotune Loop (lines 6142–6781)

### BayesianEarlyStopper (lines 6142–6316)
Multi-criterion stopper:
- `_DEFAULT_MIN_DELTA_REL = 0.005`
- `_DEFAULT_STOPPER_SAFETY_CAP = 1,000,000` (hard ceiling — Mandate #13 stopper defaults as documented single source of truth)
- Criteria: plateau, EI exhaustion, coverage saturation, wall-clock cap, hard ceiling, cost-model rejection.

### run_bayesian() (lines 6488–6704)
Main TPE loop with `multivariate=True`. SQLite persistence for cross-run resume. `bias_trial_queue` for stall-info biasing. Top-K refinement phase via `topk_refine()`.

### topk_refine() (lines 6721–6781)
±radius=2 neighbor exploration around top-K seeds; K auto-detected via `_detect_topk_elbow()` (second discrete-difference curvature method).

---

## 12. Winner Selection (lines 6791–7132)

### ORIGIN_* Constants (lines 6791–6796)
`ORIGIN_TEMPLATE`, `ORIGIN_SYNTH`, `ORIGIN_POLYHEDRAL`, `ORIGIN_CUTLASS`, `ORIGIN_CK`, `ORIGIN_FASTMATH`.

### pick_winner() (lines 6824–6974)
- Filters: numerical failures, non-deterministic trials (rejected unconditionally — P0.2), unvalidated generated variants (Mandate #16 — origin validation).
- Tiered-spill ranking (Mandate #23: ptxas-v spill_stores/stack_frame features).
- fp64 ground-truth + A/A/A determinism gate via `tests/hw/test_l3tc_tail_gate`.

### roofline_for_winner() (lines 7094–7132)
Computes `achieved_tf`, `pct_roofline`, `sub_ceiling` (threshold 10%) — Mandate #24.

`_analytic_kernel_flops()` (lines 7054–7091): conservative GEMM lower bound: 3 × 2 × 16 × d × d.

---

## 13. Compile Cache (lines 7685–8000+)

### CompileCache
- JSON cache with POSIX `fcntl.flock` inter-process locking + Windows sentinel fallback.
- `CACHE_VERSION = 4`.
- v2→v3→v4 migration chain.
- `.jsonl` trial sidecars for v4.
- `_merge_disk_entries()` for AOT+JIT subprocess safety.

---

## 14. BuildSpec Dataclass (lines 8488–8637)

Key fields:
```
optimizer, model, arch, out_dir
autotune: bool = True
autotune_mode: str = "bayesian"   # "exhaustive" | "bayesian"
runtime: str = "both"             # "aot" | "jit" | "both"
pgo: bool = False
bayesian_trials: Optional[int] = None   # None → multi-criterion auto early-stop
top_k: Optional[int] = None
max_tune_seconds: Optional[float] = None
min_improvement: float = 0.005
patience: Optional[int] = None
ei_floor: float = 1e-6
coverage_growth_floor: float = 0.001
seed: int = 0
debug_symbols: bool = False
pruner: str = "none"   # "none" | "median" | "hyperband"
transfer_learning: bool = False
enable_runtime_specialization: bool = False
enable_emitter: bool = False
enable_synth_codegen: bool = False
enable_device_pgo: bool = False
prune_after_autotune: bool = True
allow_nondeterministic: bool = False
cross_host: bool = False
enable_fastmath_variants: bool = True
macro_prefix: str = "SG_BUILD_"
fused_op_template: str = "torch.ops.grokking_optimizers.fused_{opt_lower}_simple_step"
tune_hook: Optional[str] = None
python_package: str = "grokking_optimizers"
tuned_header_path: str = "csrc/algorithms/tuned_configs.h"
enable_polyhedral: bool = False
enable_cost_model: bool = True
cost_model_retrain_every: int = 20
cost_model_rejection_threshold_x: float = 3.0
cost_model_rejection_max_pct: float = 0.8
config: Dict[str, Any] = {}
```

---

## 15. Flag Resolution Infrastructure (lines 10000–12900)

### _device_cflags() and _host_cflags()
CUDA path builds: NVCC_DEVICE_BASE + SASS gencodes from ARCH_TABLE + debug/profile conditionals + feature-gated macros (TMA, wgmma, cluster, fp8, fp4, tcgen05) + CUTLASS macros.

HIP path: HIPCC_DEVICE_BASE + offload-arch + `-mcumode` (CDNA/wave64) or `-mtgsplit -mwavefrontsize32` (RDNA/wave32) + AMD feature macros.

Version-gated flags (`_newer_compiler_flags()`):
- CUDA ≥11.0: `-Xptxas --def-load-cache=ca`
- CUDA ≥11.2: `--threads N`, `--diag-suppress=20012,20013`
- CUDA ≥11.5: `-Xptxas --def-store-cache=wb`
- CUDA ≥12.3: `-Xptxas --register-usage-level=10`
- CUDA ≥12.5: `-Xptxas --maxrregcount-list=64,128,192,<cap>`, `-Xptxas --allow-expensive-optimizations=true`
- CUDA ≥12.6: `--split-compile=<NCPUS>`
- CUDA ≥13.0: `--minimal`

**NOTE:** `-dlto` / `-rdc=true` (device LTO) explicitly NOT emitted (lines 12783–12806): incompatible with per-TU `code=sm_90a` gencodes + torch's cpp_extension build driver has no `nvcc -dlink` step. `--device-link-options=-dlto` lives in `_device_ldflags()` for future dlink callers only.

### MAX_JOBS wiring (lines 13158–13176)
Honours `$MAX_JOBS` env or defaults to `NCPUS`. `$NVCC_THREADS` defaults to "8". Sets `CMAKE_BUILD_PARALLEL_LEVEL` in the ninja overlay. `TORCH_CUDA_ARCH_LIST` is pinned to the build's own target arch in the overlay (not leaked into `os.environ`) to prevent torch from injecting its own gencode-without-"a"-suffix.

### ccache / sccache wiring (lines 11530–11708)

`_sccache_env()` (lines 11608–11656):
- **ccache** (if on PATH): wraps host CC/CXX via masquerade symlink dir (`/dev/shm/ccache-shim`). Sets `CCACHE_DIR` preferring `/dev/shm/ccache`.
- **sccache** (if on PATH): wraps NVCC via `PYTORCH_NVCC = "<sccache> <real_nvcc_binary>"` (NOT path-spelled "nvcc" — avoids the 2026-06-12 bug where a sccache-shim script was presented as the compiler). Optionally uses `SCCACHE_REDIS_ENDPOINT` for cross-host distributed cache.
- Priority: ccache for host TUs (3–4.5× faster local C/C++ hit rate), sccache for NVCC (CUDA hash bugs fixed post-vllm#13697).

`_warn_if_no_compiler_cache()`: warns loudly if neither is on PATH (sweep is compile-bound otherwise).

`_compiler_cache_stats()` and `_report_compiler_cache_stats()`: parse sccache `--show-stats` for hit-rate reporting.

`_resolve_real_nvcc()` (lines 11590–11605): resolves the actual `$CUDA_HOME/bin/nvcc` binary for wrapping, not a shim script.

`_writable_cache_dir()` (lines 11534–11545): prefers `/dev/shm/<name>`, falls back to `~/.cache/<name>`.

### Flag Probe (Agent-F2) (lines 11977–12350)
`_validate_flag_set()`: dry-run validates suspect flags via tiny kernel compilation (`-ptx` for CUDA, `--cuda-host-only -c` for HIP). Pair flags like `-Xptxas --opt-level=3` probed together. Results cached per `(compiler-path, version-stamp, flag)`. Disabled by `--no-flag-probe` CLI (`_FLAG_PROBE_DISABLED = False`).

`_probe_flag_support()`: handles pair-form flags, special-cases `-dlto`/`-rdc=true` using `-c` mode (not `-ptx`, since `-ptx -dlto` is incompatible). Conservative on tempdir failure: defaults to accepting the flag.

---

## 16. XLA / Pallas Environment (lines 11408–11478)

`_XLA_FLAGS_GPU` and `_XLA_FLAGS_TPU` tuples: XLA backend env vars for GPU/TPU JIT workers.

`_xla_env()`: returns env dict (empty for non-Pallas). Pins `JAX_PLATFORMS=tpu` for tpu_* archs on multi-accelerator boxes. Passes through `LIBTPU_INIT_ARGS` from the user's shell.

---

## 17. Include Path Resolution (lines 11481–11527)

`_include_paths()`: resolves `-I` list. `spec.source_roots["bindings"]` overrides `REPO_ROOT/csrc/bindings`. Appends CUTLASS include dirs when `third_party/cutlass/include` exists (matching `setup.py`'s install build).

`_cutlass_include_paths()`: returns `[third_party/cutlass/include, third_party/cutlass/tools/util/include]` when vendored.

---

## 18. Build Entry Points

### build_aot() (line 17409)
1. Pallas short-circuit: returns `Path("pallas-noop")` sentinel (no nvcc needed for TPU).
2. Resolves sources, host/device/ldflags.
3. Folds version-gated flag hashes into `host_hash`/`device_hash` so compiler upgrades invalidate cache.
4. Cache hit check: returns cached .so path if fresh.
5. On PGO: calls `_build_aot_pgo()` (3-pass: instrument → collect → use).
6. On cache miss: calls `_torch_load()` then `_publish_aot_artifact()`.
7. Calls `cache.record_aot()` and `cache.save()`.

### _build_aot_pgo() (line 17519)
Three-pass PGO loop:
- Pass 1: instrumented build (`-DSG_PGO_INSTRUMENT=1`).
- Pass 2: collect workload via `collect_workload()`.
- Pass 3: profile-use build.
- Fallback on any pass failure: non-PGO AOT build (A.15 fix).

### device_profiling import (line 17652) — CONDITIONAL / DEAD PATH
```python
if spec.enable_device_pgo:
    try:
        from grokking_optimizers.device_profiling import run_device_pgo_round
        ...
    except ImportError:
        pass
```
**Status: INERT by default.** `enable_device_pgo=False` in `BuildSpec` → the import is never reached in normal operation. The import is inside a try/except ImportError so it is also gracefully degrading. This is NOT a hard dependency.

### build_jit() (line 17693)
1. Pallas: routes to `_pallas_autotune()`, writes JSON manifest.
2. Checks AOT cache entry; runs `build_aot()` if missing.
3. Resolves sources, flags, runs autotune sweep.
4. Writes winning config to `_write_tuned_configs_header()`.

### build() (line 17899)
Top-level orchestrator (in-process, no fork):
- All proven layers ON by default (`pgo=True`, `transfer_learning=True`, `enable_runtime_specialization=True`, `enable_emitter=True`, `enable_device_pgo=True`, `pruner="hyperband"`).
- `enable_synth_codegen=False`, `enable_polyhedral=False` (opt-in, scaffold not yet proven).
- Auto-detects arch when `arch in (None, "auto", "")` via `_resolve_default_arch()` (torch.cuda → rocm-smi → jax → config → "sm_90a" fallback).
- Bootstraps toolchain if missing and `bootstrap_cuda/bootstrap_rocm/bootstrap_jax=True`.
- Hard preflight: fails loudly if nvcc/hipcc missing after bootstrap.

---

## 19. _write_tuned_configs_header() (line 17311)

Materializes the JIT-winner combo to a C++ `#define` header at `spec.tuned_header_path` (default `csrc/algorithms/tuned_configs.h`). On space-load failure: emits the SAFE per-TU macros from `_DERIVED_HEADER_BACKCOMPAT` and warns loudly (audit fix #3 — previously silent degradation).

---

## 20. Bootstrap Toolchain (lines 9296–9405)

`bootstrap_cuda_toolkit()`: 10-method probe chain: conda → nvidia-apt → apt → dnf → yum → zypper → pacman → apk → brew → winget → pip-wheels. NVIDIA apt repo installs correct version per arch.

Analogous `bootstrap_rocm_toolkit()` and `bootstrap_jax_tpu()`.

---

## 21. _DEFAULT_PROJECT_CONFIG (line 32330) — Portability Layer

The single source of truth for all project-specific opinions:

```python
_DEFAULT_PROJECT_CONFIG = {
    "project": {
        "name": "supergrok",
        "version": "2.0.0",
        "macro_prefix": "SG_BUILD_",
        "fused_op_template": "torch.ops.grokking_optimizers.fused_{opt_lower}_simple_step",
        "tune_hook": "grokking_optimizers.tune_hook:run",
        "python_package": "grokking_optimizers",
        "namespace": "",
    },
    "sources": {
        "cuda_root": "csrc/backends/cuda",
        "hip_root": "csrc/backends/hip",
        "pallas_root": "csrc/backends/pallas",
        "algorithms_dir": "csrc/algorithms",
        "bindings_dir": "csrc/bindings",
        "tuned_header_path": "csrc/algorithms/tuned_configs.h",
    },
    "device_cflags": {"extra": ["-DNDEBUG"]},   # project's -DNDEBUG; NOT in compile.py base flags
    "optimizers": {"enabled": ["adamw","lion","muon","prodigy","grokadamw",
                               "grokfast","looksam","neuralgrok",
                               "supergrok11","supergrok15","supergrok2"]},
    "models": {"enabled": ["mamba","decoder","vit"]},
    "archs": {"default": "sm_90a", "allowed": []},
    "pgo": {"workload_script": "", "steps": 1000},
    "autotune": {"min_improvement": 0.005, "patience": 0, "max_seconds": None},
    "codegen": {"enable_emitter": False, "template_dir": "grokking_optimizers/templates", "template_overrides": {}},
    "synth_codegen": {"enable": False, "allowed_patterns": [...], "max_fusion_depth": 3, "prefer_synth_over_template": False},
    "runtime_specialization": {"enable": False, "cache_dir": ""},
    "device_pgo": {"enable": False},
    "cache": {"auto_prune_after_jit": True, "max_age_days": 30, "keep_top_n": 100},
    "numerics": {"strict": False},
    "polyhedral": {"enable": False, "max_schedules_per_template": 16, "allowed_transforms": [...], "tile_size_candidates": [16,32,64,128]},
    "cost_model": {"enable": False, "retrain_every": 20, "rejection_threshold_x": 3.0, "rejection_max_pct": 0.8, "uncertainty_method": "bootstrap"},
}
```

`load_config()` (line 32558): merges in priority order — caller path > CWD `compile_config.toml` > inlined defaults.

`DEFAULT_CONFIG_PATH = Path(__file__).parent / "compile_config.toml"` (line 32514).

The `-DNDEBUG` flag (assert-stripping, deserializes the WGMMA mainloop per comment "C7509 6→0") is a project decision tracked in this config, NOT baked into compile.py's base flag lists. A config-less generic build gets no `-DNDEBUG`.

---

## 22. STALL_DIM_HINTS (line 32009) — Device-Side PGO Stall Mapping

```python
STALL_DIM_HINTS = {
    "long_scoreboard":    ["swizzle", "lds_padding", "vec"],
    "not_selected":       ["block", "waves_per_eu", "maxrregcount"],
    "math_pipe_throttle": ["unroll", "num_stages"],
    "memory_throttle":    ["block", "vec", "async_depth", "num_stages"],
    "tex_throttle":       ["block", "vec"],
    "barrier":            ["block", "warp_specialization"],
    "wait":               ["num_stages", "async_depth"],
    "imc_miss":           ["block", "unroll"],
    "lg_throttle":        ["lds_padding", "swizzle"],
    "dispatch_stall":     ["block", "maxrregcount"],
    "vmem_lat":           ["waves_per_eu", "vec"],
    "lds_bank_conflict":  ["lds_padding"],
    "valu_dep":           ["unroll", "num_stages"],
    "inst_fetch":         ["maxrregcount", "block"],
}
```

Used by `bias_trial_queue()` to enqueue stall-directed trial points. Vendor-specific collectors: `collect_nvidia_stalls()` via nsys, `collect_amd_stalls()` via rocprof, `collect_pallas_stalls()` via XLA HLO dump. `_COST_MODEL_STALL_REASON_COUNT = 14` (line 5332).

---

## 23. _MMA_NATIVE_LOADS_WIRED = False (line 30179) — Dead/Inert Path

```python
_MMA_NATIVE_LOADS_WIRED: bool = False
```

Controls `_gemm_native_mma_mainloop()` (line 30183). When False (current state), the native tensor-core MMA bodies for wgmma/tcgen05/mfma/wmma are ZERO-FRAGMENT STUBS — the function returns None and the caller emits a numerically-correct scalar triple-loop instead. The stub bodies are present in the code as wiring scaffold but are unreachable in practice.

Self-tests at lines 26333 and 26506 explicitly assert `not _MMA_NATIVE_LOADS_WIRED` and will fail if this is ever set to True without the native fragment loads being implemented.

**Impact:** Any `enable_synth_codegen=True` GEMM synthesis does NOT use tensor cores; it falls back to scalar math. This is a hard functional gap in the OpGraph layer.

---

## 24. device_profiling Import (line 17652) — Conditional / Inert

Located inside `_build_aot_pgo()`, gated by `spec.enable_device_pgo`:
```python
if spec.enable_device_pgo:
    try:
        from grokking_optimizers.device_profiling import run_device_pgo_round
        ...
    except ImportError:
        pass
```

`enable_device_pgo=False` is the default in both `BuildSpec` and `_DEFAULT_PROJECT_CONFIG["device_pgo"]["enable"] = False`. The import is doubly inert: (1) the condition is False by default; (2) even if True, the ImportError is silently swallowed.

The `device_profiling` module's existence and interface is tested via `_self_test_device_profiling()` at line 20299, which tests `device_profiling._stall_to_bias_hints()`, `bias_trial_queue()`, `run_device_pgo_round()`, `write_stall_sidecar()`, and `read_stall_sidecar()`. So the module exists but is not activated by default.

---

## 25. main() CLI (line 18690)

Early intercepts (in order): `--self-test`, `--so` (PGO workload mode), `--debug-flags`/`--verbose-flags`, `--flag-audit`, `--e2e-smoke`, `--list-archs`, `--dry-run-all-archs`. Then full argparse with required: `--optimizer`, `--model`, (`--arch` optional — auto-detected if omitted).

---

## 26. PGO

### Host-side PGO (`_build_aot_pgo()`, line 17519)
3-pass: instrument (-DSG_PGO_INSTRUMENT=1) → collect workload → profile-use rebuild. Fallback to non-PGO on any pass failure (A.15).

### Device-side PGO (`spec.enable_device_pgo`)
Gated, inert by default. When enabled: nsys (NVIDIA), rocprof (AMD), XLA HLO dump (TPU). Output feeds `bias_trial_queue` in the Bayesian sweep via stall-to-dim hint mapping.

---

## 27. Key Fixes and P0 Items

- **P0.1**: Transitive `#include` walk for source hashing (was TUs-only → stale cache hits on header changes).
- **P0.2**: `non_deterministic` trials rejected by pick_winner unconditionally; `allow_nondeterministic` field in BuildSpec is the narrow opt-out.
- **P0.3**: Dead dim exclusion from config_key is now derived dynamically, not a static frozenset.
- **MED.12**: Heartbeat stamp before io_lock in TimingWorker to prevent false watchdog.
- **2026-06-12 sccache bug**: `PYTORCH_NVCC` set to `<sccache> <real_nvcc_binary>`, NOT `<sccache> nvcc` (to avoid sccache being asked to treat a bash shim script as the compiler).
- **Mandate #7/#23 cross-host**: `cross_host=True` → portable ISA baseline (`cross_host_march`) for AOT ship.

---

## 28. Self-Tests (lines ~20000–26900)

Self-test suite invoked via `--self-test`. Includes:
- `_self_test_device_profiling()` (line 20299): tests the device_profiling module interface.
- `test_pallas_build_aot_noop()` (line 25777): confirms build_aot returns `Path("pallas-noop")` for tpu_v5p.
- Tests for flag probe, flag trace, cache migration, config_key, dead dims, synth codegen, MMA stub state.

---

## State Assessment

**DONE / VALIDATED (per code):**
- ARCH_TABLE with 25+ entries, aliases, H100/CDNA3/TPU coverage.
- Bayesian autotune (TPE + multi-criterion stopper + XGBoost cost model + top-K refine).
- MultiGPUTimingPool with per-GPU calibration and work-stealing.
- CompileCache v4 with POSIX locking, migration chain, trial sidecars.
- ccache/sccache wiring including the 2026-06-12 PYTORCH_NVCC bug fix.
- MAX_JOBS wiring with NCPUS default and $MAX_JOBS override.
- _DEFAULT_PROJECT_CONFIG portability layer (config-derived naming/layout).
- Version-gated nvcc/hipcc flags (11.0..13.0 gates fully documented).
- Flag probe (dry-run compiler validation, pair-flag handling, -dlto special case).
- PGO 3-pass host-side with fallback.
- pick_winner tiered-spill ranking, origin validation, determinism gate.
- roofline metrics (achieved_tf, pct_roofline, sub_ceiling).
- Dead dim detection and exclusion from config_key.

**INERT / DEAD:**
- `_MMA_NATIVE_LOADS_WIRED = False` → native wgmma/tcgen05/mfma/wmma MMA mainloops unreachable; OpGraph GEMM synthesis falls back to scalar triple-loop.
- `device_profiling` import: doubly inert (enable_device_pgo=False default + ImportError silenced).
- `enable_synth_codegen=False` (default) → OpGraph synthesis never runs without explicit opt-in.
- `enable_polyhedral=False` (default) → libclang/islpy schedule search never runs.
- `enable_emitter=False` in `_DEFAULT_PROJECT_CONFIG.codegen` (Jinja2 per-variant source emission off by default; `build()` passes `enable_emitter=True` as its default, but TOML config overrides to False unless changed).

**OPEN ITEMS:**
- `_MMA_NATIVE_LOADS_WIRED` needs to flip True with real fragment load implementation + silicon validation before OpGraph GEMM can use tensor cores.
- `enable_synth_codegen` / `enable_polyhedral` are Level-2 scaffold (task #27) not yet proven for production.
- Device-side PGO (CUPTI/rocprof/XLA HLO) is wired but off by default — nsys/rocprof must be on PATH.
- `mb_dw_splitk` dimension was removed 2026-06-17 but may still appear in cached configs from before that date (migration path not explicitly documented).
