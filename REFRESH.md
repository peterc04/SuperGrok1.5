# REFRESH.md — SuperGrok1.5 Reference

A compact, granular catch-up. Plain language. Each kernel, file, and optimizer gets its own entry. No fluff, no withholding.

> Read §0 for the current state. §24 is the per-arch per-optimizer hot-path map; §25 is the live engineering-remaining list.

---

## 0. Post-refactor state (NEW)

### Supported arches (no fallback)

After the §10 expansion, eight GPU arches plus two TPU versions are supported:

- **NVIDIA**:
  - `sm_80` — Ampere family. A100, A30, A10. RTX 30-series and sm_86 route here.
  - `sm_89` — Ada Lovelace. RTX 40-series, L40, L40S.
  - `sm_90` — Hopper. H100, H200.
  - `sm_100` — Datacenter Blackwell. B100, B200, GB200.
  - `sm_103` — Blackwell Ultra. B300, GB300 NVL72. NVFP4 hot path.
  - `sm_120` — Consumer Blackwell. RTX 50-series, RTX PRO 6000 Blackwell. 128KB shared memory per SM.
- **AMD**:
  - `gfx942` — CDNA3. MI300X, MI300A.
  - `gfx950` — CDNA4. MI350X, MI355X. Native FP4 expert MFMA, FP6 state, 2:4 sparsity.
- **TPU** (JAX path): `v5p` (128-wide MXU), `v6e` (256-wide MXU).
- **CPU**: testing only, not a runtime fallback.

Anything else raises `UnsupportedArchError` or fails the build. Permanently unsupported: V100, T4, RTX 20-series, MI100 (gfx908), MI200 (gfx90a), AMD RDNA, TPU v3/v4/v5e.

### Filesystem layout

```
csrc/
├── common/          (shared headers; tuned_configs.h with 8-row tables)
├── bindings/        (per-optimizer dispatchers + pybind11 module)
│                     _helpers.h with clip-grad / SAM-norm host helpers
│                     _dispatch_macro.h with SG_DISPATCH + SG_DISPATCH_CALL
└── kernels/
    ├── cuda/
    │   ├── sm_80/   (17 wrapped baselines + 5 arch-tuned files in
    │   │            sg::sm80 namespace: muon TF32 inlined into canonical;
    │   │            metanet_optimizers, metanet_cpasync_variants,
    │   │            supergrok2_{backward,fused_elem,scan} folded in)
    │   ├── sm_89/   (17 wrapped baselines, ported from sm_90)
    │   ├── sm_90/   (17 wrapped baselines + 3 arch-tuned files in
    │   │            sg::sm90 namespace: muon FP8 inlined into canonical;
    │   │            supergrok2_{backward,scan,warp_specialized} folded in)
    │   ├── sm_100/  (17 wrapped baselines + 2 arch-tuned files in
    │   │            sg::sm100 namespace: supergrok2_{precompute,scan}
    │   │            folded in. Blackwell delegating overlay deleted —
    │   │            per-arch baselines are sufficient)
    │   ├── sm_103/  (17 wrapped baselines, ported from sm_100)
    │   └── sm_120/  (17 wrapped baselines, ported from sm_100)
    ├── hip/
    │   ├── gfx942/  (17 wrapped baselines in sg::gfx942 namespace;
    │   │            muon BF16 MFMA + supergrok2 CDNA3 BF16 MFMA paths
    │   │            inlined into canonical batched_step)
    │   └── gfx950/  (17 wrapped baselines + four per-feature CDNA4
    │                 files in sg::gfx950 namespace: fp4_expert_kernels,
    │                 fp6_state_kernels, sparse24_kernels, fused_combos.
    │                 Shared FP4/FP6 helpers in csrc/common/fp4_helpers.hip.h)
    ├── tpu/
    │   ├── _pallas_kernels.py  (shared Pallas implementation)
    │   ├── v5p/                 (tile-128 re-export for v4/v5e/v5p)
    │   ├── v6e/                 (tile-256 re-export for v6e)
    │   └── __init__.py          (detect_tpu_version + get_kernels)
    └── cpu/         (avx512/, neon/, scalar; testing only)
autotune/            (tune.py, grids.py with NVFP4/FP4/FP6 entries,
                     cutlass_profile.py for sm_89/103a/120a, runner.py)
```

Removed entirely: `csrc/cuda/generic/`, `csrc/cuda/generated/`,
`csrc/cuda/sm_75/86/89/`, `csrc/hip/cdna2/cdna3/cdna4/`, `csrc/cpu/`
(moved to `csrc/kernels/cpu/`), `csrc/common/{ops.h,ops.cpp,dispatch.h}`,
`grokking_optimizers/jit/`, `grokking_optimizers/jit_kernels.py`,
`codegen/`. The `*_overlay.*` naming convention is gone — all overlay
files have been folded into per-arch namespaced canonical kernels.

### Per-arch kernel pattern

Each per-arch source file is wrapped in `namespace sg::<arch> { ... }`
so the eight translation units do not collide on launcher / kernel
symbols at link time. The 8 × 17 = 136 wrapped baselines start out
with identical math; per-arch divergence (cp.async, TMA, MFMA, FP8
paths, warp specialization) is layered on top by direct edits to the
canonical launchers within each `sg::<arch>` namespace (e.g. the
Muon Ampere TF32 / Hopper FP8 / CDNA3 BF16 MFMA paths, and the SG2
sm_90 Hopper FP8 batched-step path). Cross-arch numerical agreement
is guarded by `tests/test_cross_arch_agreement.py`, with filled-in
test bodies for every optimizer including a Muon 2D-param harness.

### Bindings layer

- `csrc/bindings/bindings.h` — shared declarations and arch namespace anchors
- `csrc/bindings/dispatch.cpp` — `detect_arch()`: returns one of
  `{80, 89, 90, 100, 103, 120, 942, 950}` or raises (no fallback chain)
- `csrc/bindings/_dispatch_macro.h` — two macros:
  - `SG_DISPATCH(method, args...)` — early-returns from the enclosing
    function. Use at the tail of a per-tensor wrapper.
  - `SG_DISPATCH_CALL(method, args...)` — same dispatch with `break`
    instead of `return`. Use inside loops.
- `csrc/bindings/_helpers.h` — host-side helpers shared by per-optimizer
  binding files: `clip_grad_norms_device_side`,
  `compute_sam_grad_norm_device_side`. Both do single-CPU-sync norm
  reductions over a `std::vector<torch::Tensor>`.
- `csrc/bindings/<optimizer>.cpp` — per-optimizer dispatcher. Each file
  exposes two API surfaces:
    - **High-level vector-signature** — primary contract. Takes
      `std::vector<torch::Tensor>&`, runs host-side bookkeeping
      (per-step `bc1/bc2`, gradient clipping, batched norm reductions
      with single CPU sync), then dispatches into the per-arch launcher
      via `SG_DISPATCH_CALL`. The Python optimizers in
      `grokking_optimizers/*.py` call these (e.g.
      `_ops.muon_fused_step(params, grads, bufs, ...)`).
    - **Per-tensor wrappers** — escape hatches for tests. Take individual
      `torch::Tensor` args and use the early-returning `SG_DISPATCH`.
- `csrc/bindings/module.cpp` — pybind11 aggregator. Registers every
  vector-signature entry point + every per-tensor wrapper. All seven
  SG v2 entry points are wired (see commit `fd875b8`).

Registered vector-signature entry points (callable from Python via `_ops.<name>`):

| Optimizer | Entry points |
|-----------|--------------|
| GrokAdamW | `grokadamw_fused_step`, `fused_adamw_simple_step` |
| Lion | `lion_fused_step` |
| Grokfast | `grokfast_fused_step` (EMA only), `grokfast_fused_ema_adam_step` |
| Prodigy | `prodigy_fused_step` (returns updated `d_lr`) |
| NeuralGrok | `neuralgrok_fused_step` |
| LookSAM | `looksam_perturb_all`, `looksam_restore_all`, `looksam_compute_directions_and_adjust` |
| Muon | `muon_fused_step` |
| SG v1.5 | `supergrok15_fused_step`, `supergrok15_sam_perturb_all`, `supergrok15_sharpness_restore_all` |
| SG v1.1 | `supergrok11_fused_step`, `supergrok11_sam_perturb_all`, `supergrok11_sharpness_restore_all` |
| SG v2 | `supergrok2_mamba_peer_step`, `supergrok2_mamba_peer_batched_step`, `supergrok2_bilevel_fwd_save{_batched}`, `supergrok2_bilevel_backward{_batched}`, `supergrok2_prepare_and_batched_step` |
| MoE | `moe_dynamic_expert_{load,fwd,bwd}`, `moe_filter_active_params`, `moe_scan_compacted`, `moe_scatter_results`, `moe_count_expert_activations`, `moe_compute_load_balance_loss`, `moe_apply_frequency_scaling` |
| Distributed scan | `distributed_scan_phase{1,2,3}` |
| Quantization | `fp8_e4m3_quantize`, `int8_symmetric_quantize`, `int4_gptq_quantize`, `mxfp4_quantize` |

### dispatch.py / __init__.py changes

- `dispatch.get_gpu_arch()` returns one of `{80, 90, 100, 942}` or raises `UnsupportedArchError`
- Tier helpers (`get_arch_tier`, `get_amd_tier`, `get_amd_label`) are gone
- `assert_supported_arch()` and `SUPPORTED_ARCHES` are new public surface
- `__version__` bumped to `3.0.0` (breaking)
- `_HAS_OPS` simplified: extension or error

### Autotune

`autotune/tune.py` is the offline tuner: runs each kernel × arch × shape × config grid, picks the median-fastest, writes `csrc/common/tuned_configs.h`. The header is committed. CUTLASS GEMMs (SG2 projections, Muon Newton-Schulz) go through `cutlass_profile.py`. Currently scaffolding — `runner.py:bench()` and `cutlass_profile.py:profile_gemm()` raise `NotImplementedError`; they need a hardware-equipped session to wire to `torch.utils.cpp_extension` and the CUTLASS profiler binary.

### TPU dispatch

Pallas kernels moved from `supergrok2_jax_tpu/pallas_kernels.py` to `csrc/kernels/tpu/`. The shared implementation is `_pallas_kernels.py`; v5p re-exports the tile-128 variants, v6e re-exports the tile-256 variants. `csrc/kernels/tpu/__init__.py:get_kernels()` selects based on `detect_tpu_version()`. `supergrok2_jax_tpu/pallas_kernels.py` is now a backwards-compat shim that re-exports from the new path.

### Code statistics (post-refactor)

180,845 lines of code across 267 tracked files. Breakdown by language:

| Language | Files | Lines | Notes |
|----------|------:|------:|-------|
| CUDA C++ (`.cu` / `.cuh`) | 119 | 112,822 | Per-arch wrapped baselines (17 optimizers × 6 NVIDIA arches = 102 files) + arch-tuned per-namespace divergence files (sm_80 cp.async metanet variants, sm_90 warp-specialized scan, sm_100 TMA precompute scaffold) + 2 shared `.cuh` headers (`ptx_intrinsics.cuh`, `utils.cuh`) + `_cutlass_gemm.cuh` + 1 quantization kernel. |
| HIP C++ (`.hip.cpp`) | 38 | 38,050 | Per-arch wrapped baselines for gfx942 + gfx950 (17 optimizers × 2 = 34) + four gfx950 per-feature files (`fp4_expert`, `fp6_state`, `sparse24`, `fused_combos`). |
| Python (`.py`) | 71 | 21,342 | `grokking_optimizers/` package (eleven optimizers + dispatch + bindings + distributed + quantization + cuda_graph wrappers), `supergrok2_jax_tpu/` JAX implementation, `csrc/kernels/tpu/_pallas_kernels.py` (1190 lines), `autotune/` scripts, tests. |
| C++ host code (`.cpp` / `.h`) | 31 | 5,523 | `csrc/bindings/` per-optimizer dispatchers + module aggregator (~16 files), shared headers (`platform.h`, `types.h`, `arch_tier.h`, `quantization.h`, `tuned_configs.h`, `bindings.h`, `fp4_helpers.hip.h`), CPU testing-only sources (`csrc/kernels/cpu/`). |
| Markdown docs (`.md`) | 6 | 2,213 | `README.md`, `REFRESH.md` (this file), `ANALYSIS.md`, plus per-tree READMEs (`csrc/kernels/README.md`, `csrc/kernels/hip/README_HIP.md`, `autotune/README.md`). |
| Config (TOML) | 1 | 19 | `pyproject.toml`. |
| **Total** | **267** | **180,845** | |

The CUDA + HIP totals include large amounts of structurally-identical baseline content. Real divergence — cp.async vs TMA vs MFMA, FP8 vs NVFP4 vs FP4 paths, warp specialization — is added per-arch under hardware-validated tuning passes; the cross-arch numerical agreement test (`tests/test_cross_arch_agreement.py`) catches drift.

### 0.5 What is NOT yet done

The all-specialized refactor, arch-matrix expansion, overlay merges,
SG v2 binding wiring, ninja build wrapper, autotune execution layer,
and CUTLASS scaffolding are now all complete. Remaining work is
engineering-focused, deferred until hardware. See §25 for a structured
list and §24 for the per-arch per-optimizer plan.

Items that have been **completed since the previous REFRESH.md edit**:

- **Fix 1 — Binding signature re-audit**: SG11, SG15, and Muon
  binding signatures verified against all 8 per-arch launchers and
  the pre-refactor `682eab4^:csrc/common/ops.cpp`. Zero drift found.
- **Fix 2 — Muon neg_lr_scale anomaly**: the binding wrapper in
  `muon.cpp` had an erroneous `/ sqrt(max_dim)` that cancelled the
  spectral scaling, producing `-lr * 0.2` instead of the correct
  `-lr * 0.2 * sqrt(max_dim)`. Bug was present since the original
  `ops.cpp` and faithfully copied during refactor. Fixed in binding.
- **Fix 3 — MAX_D caps raised**: `MAX_D_MODEL` 16→64,
  `MAX_D_STATE` 32→128, `MAX_D_INNER` 32→128. Activates the Hopper
  FP8 path (gated on `d_inner/d_state/d_model >= 64`) and supports
  real-world `d_state=128` workloads. No shared memory overflow —
  all MAX_D-sized arrays are thread-private. Four CPU kernel files
  and one sm_90 warp-specialized file had local shadow constants
  synced.
- **Race driver — 3-way train/val/test split**: `grokking_race_v2.py`
  refactored to carve a val set out of the train portion (controlled
  by `--val-ratio`, default 0.10, auto-overrides to 0.05 on 10/90).
  All `make_data*` functions return six tensors instead of four. All
  11 training functions take `(c, init, tx, ty, vax, vay, tex, tey,
  dev, bp)`. SG2 / SG1.5 / SG1.1 bilevel and meta updates now consume
  the val set; previously they consumed test data — that test leak
  is now closed. Other 8 optimizers stay train-only.
- **Race driver — fixed early-stopping rule**: `EarlyStopper` now
  triggers on either `test_acc >= 0.95` or `step >= 20,000`, whichever
  fires first. Both thresholds are CLI-configurable
  (`--early-stop-test-acc`, `--early-stop-max-steps`). Eval frequency
  controlled by `--eval-every` (default 100). `stopping_reason` is
  one of `test_acc_threshold` or `max_steps`. The "test" referred to
  here is the outer held-out portion (1 − `frac_train`); the inner
  val carve-out is a separate set consumed only by SG variants.
- **Race driver — per-step val + test eval, held-out test eval at
  end**: `_eval_log` evaluates train, val, and test every
  `eval_every` steps (default 100); the stopper reads test_acc.
  `_fin()` then runs `model.eval()` + `torch.no_grad()` for the
  final test-set evaluation. `TrainResult` and JSON output track
  `train_losses`, `train_accs`, `val_losses`, `val_accs`,
  `test_losses`, `test_accs` arrays plus the final scalars
  `final_test_acc`, `final_test_loss`, `final_val_acc`,
  `final_val_loss`, `val_test_gap` (the meta-learning vs
  masked-overfitting diagnostic), `stopping_reason`, `stopping_step`,
  and `val_ratio`. See §3.12 for the fairness framing and §15 for
  the full CLI surface.
- **Race driver — CLI surface expanded**: new flags `--optimizers`,
  `--seeds`/`--num-seeds`, `--tasks`, `--train-test-ratios`,
  `--val-ratio`, `--early-stop-test-acc`, `--early-stop-max-steps`,
  `--eval-every`, `--output`. All pre-existing flags (`--setup`,
  `--ntfy`, `--gpus`, `--grad-hooks`, `--port`, `--no-status-server`)
  preserved.
- **Race driver — split sanity tests**: new `tests/test_race_split.py`
  with 10 sections covering split arithmetic for both 80/20+10% and
  10/90+5%, disjointness via index sets, deterministic split,
  val_ratio auto-override, EarlyStopper stopping_reason for both
  triggers, and TrainResult output schema completeness. Skip-marks
  gracefully when `_HAS_OPS` is false.
- **SG v2 bindings**: all seven SG2 entry points
  (`supergrok2_mamba_peer_step`, `supergrok2_mamba_peer_batched_step`,
  `supergrok2_bilevel_fwd_save{_batched}`,
  `supergrok2_bilevel_backward{_batched}`,
  `supergrok2_prepare_and_batched_step`) are wired in
  `csrc/bindings/supergrok2.cpp` and registered in
  `csrc/bindings/module.cpp`. Each is a thin SG_DISPATCH wrapper
  around the per-arch launcher.
- **Inlining of arch-suffixed launchers**: complete across all arches.
  sm_80 (Ampere TF32 wrap inlined into supergrok15 / supergrok11 /
  neuralgrok / sg2 backward / sg2 scan canonicals), sm_100 (Blackwell
  prefix dropped from supergrok2_precompute / supergrok2_scan symbol
  names), gfx942 (CDNA3 BF16 MFMA inlined into mamba_peer_batched_step
  canonical), and **sm_90 Hopper FP8 fast path restored and inlined
  into `launch_mamba3_peer_batched_step`** (commit `15928da`). FP8
  helpers (`hopper_fp8_gemm`, `hopper_precompute_fp8`) live inside
  `sg::sm90`, gated on `CUDA_VERSION >= 11080`, activated when
  `total_N >= 1024` and `d_inner / d_state / d_model >= 64`.
- **gfx950 CDNA4 split**: the 2491-line `cdna4_kernels_gfx950.hip.cpp`
  monolith has been split into four per-feature files
  (`fp4_expert_kernels`, `fp6_state_kernels`, `sparse24_kernels`,
  `fused_combos`) with shared FP4/FP6 helpers extracted into
  `csrc/common/fp4_helpers.hip.h`. All helpers are
  `__device__ static __forceinline__` so each TU gets internal-linkage
  copies and avoids ODR errors.
- **Ninja-backed build with multi-arch AOT fatbin**: `setup.py` now
  uses `BuildExtension.with_options(use_ninja=True)`, emits
  `-gencode arch=compute_X,code=sm_X` for every supported arch plus
  PTX embedding of `compute_120` for forward-compat driver JIT, and
  detects ccache/sccache via `CMAKE_*_COMPILER_LAUNCHER`. `build.sh`
  wraps `pip install -e . --no-build-isolation -v` with a tqdm
  progress bar that parses ninja `[N/M]` lines, plus `--autotune`
  (two-pass), `--debug` (cuda-gdb), and `--profile` (NCU) modes.
  `pyproject.toml` adds `ninja` to build-system requires; bumped
  to version 3.0.0.
- **Autotune execution layer**: `autotune/runner.py:bench()` records
  one `(start, stop)` CUDA event pair per iteration with a single
  post-loop `torch.cuda.synchronize()`, returns median microseconds.
  `autotune/cutlass_profile.py:profile_gemm()` invokes the
  `cutlass_profiler` binary, parses CSV output (column-name drift
  tolerant), returns sorted-by-latency results. `autotune/tune.py`
  orchestrates per-arch grids and writes winners between
  `// AUTOTUNE_BEGIN` / `// AUTOTUNE_END` markers in
  `tuned_configs.h`. The C++ `LaunchConfig` table itself is left
  intact for hand-tuned fallback values.
- **CUTLASS migration scaffold**: `third_party/cutlass` is registered
  as a v3.6.0 submodule (not cloned in this session — the user runs
  `git submodule update --init` later). `setup.py` adds CUTLASS
  include dirs + `-DCUTLASS_NVCC_ARCHS=90a/100a/103a/120a` + a new
  `WITH_CUTLASS=1` opt-in flag. `csrc/kernels/cuda/_cutlass_gemm.cuh`
  exposes `cutlass_gemm_fp16/bf16` and `cutlass_dt_proj_fused`
  helpers (the latter currently runs the unfused linear-combo GEMM
  and relies on the pre-existing `softplus_bias_kernel` post-pass —
  fused softplus epilogue is deferred per §25). Muon Newton-Schulz
  GEMMs (sm_90/100/103/120) and the SG2 five-projection set
  (`in_proj_x`, `in_proj_z`, `dt_proj`, `B_proj`, `C_proj` — actually
  in `supergrok2_bwd_*.cu`'s `bilevel_precompute_gemm` helper, not
  the fwd files as initially expected) are wrapped in
  `#ifdef WITH_CUTLASS` so the existing torch::mm path stays the
  default until the user opts in. sm_80/sm_89 keep cuBLAS, gfx942/950
  keep rocBLAS. `tests/test_cutlass_parity.py` asserts CUTLASS output
  matches cuBLAS within 1e-3 (FP16) or 1e-4 (BF16).
- **Build-error post-merge audit**: 41 stale `#include "ops.h"` and
  `#include "dispatch.h"` references stripped from per-arch kernel
  TUs (both headers were deleted in `682eab4` but the includes
  survived). `BatchedScanCtx` struct (used by 9 SG2 forward TUs)
  recovered into `csrc/common/types.h`. `ArchTier` /
  `StatePrecision` / `ExpertPrecision` enums recovered into a new
  `csrc/common/arch_tier.h` shim with constexpr `kArchTier` picked
  per-TU via `SG_ARCH_<X>` preprocessor switches; legacy `ArchTier::X`
  call sites in distributed_pipeline TUs continue to work without
  body edits.
- **Cross-arch agreement test bodies**: filled in for every
  optimizer including a Muon 2D-param harness (`tests/test_cross_arch_agreement.py`).

Items that **remain deferred** (engineering work — see §25 for detail):

- Real per-arch kernel divergence beyond Muon and SG2-on-sm_90 / sm_80 / gfx942
  (most of the 8 × 17 wrapped baselines are still byte-identical modulo namespace).
- Raising `MAX_D_MODEL/MAX_D_STATE/MAX_D_INNER` caps in `csrc/common/types.h`
  above 64 to actually activate the Hopper FP8 path (currently capped at 16/32/32).
- Hopper warp-specialized scan activation
  beyond the renamed `_warp_specialized` declarators.
- Fused softplus epilogue in CUTLASS for SG2 `dt_proj`.
- Real autotune output (placeholders remain in `tuned_configs.h`).
- CI updates for the eight-row arch matrix.

### Migration commit series

For navigation, the structural-refactor commits:

- `895c32e` — `REFACTOR_PLAN.md`
- `866d42b` — create new directory tree
- `864499c` — `git mv` 27 existing arch-specific files into new tree
- `e180964` — rename pre-existing arch-tuned files to `*_overlay`
- `5d8085a` — wrap 17 generic kernels into 68 per-arch baselines
- `4f71d02` — bindings layer + worked-example GrokAdamW dispatcher
- `9eb21d4` — per-optimizer bindings + pybind11 module aggregator
- `c8022cc` — refactor dispatch.py / loader / __init__.py
- `e9d22cf` — move Pallas TPU kernels to `csrc/kernels/tpu/`
- `6a50d3e` — autotune scaffolding + `tuned_configs.h`
- `f422ada` — rewrite `setup.py` for new tree
- `3683945` — `tests/test_cross_arch_agreement.py`
- `6307566` — refactor `tests/test_amd_hip.py` and `tests/test_all_arches.py`
- `8c2280d` — delete unsupported arches (sm_75/86/89, cdna2/cdna4)
- `8725e3a` — delete `csrc/cuda/generic/` (17 files) + `csrc/cuda/generated/` (30 files)
- `95b77e0` — delete `csrc/cpu/` tree
- `104f3ff` — delete `grokking_optimizers/jit/` and `codegen/`
- `682eab4` — delete `csrc/common/{ops.h,ops.cpp,dispatch.h}`

§10 arch matrix expansion commits:

- `dd1e0a6` — `REFACTOR_PLAN.md §10` (arch expansion plan)
- `8da3d80` — create directory tree for sm_89, sm_103, sm_120, gfx950
- `bf157b4` — port 17 sm_90 baselines → sm_89 (Ada Lovelace)
- `e2545a8` — port 17 sm_100 baselines → sm_103 (Blackwell Ultra)
- `50925ae` — port 17 sm_100 baselines → sm_120 (consumer Blackwell)
- `98f8190` — port 17 gfx942 baselines → gfx950 (CDNA4)
- `02348cc` — recover CDNA4 FP4/FP6/2:4 sparsity overlay (2491 lines) into gfx950
- `813ffdb` — drop now-redundant `.gitkeep` placeholders
- `5b4218b` — extend bindings layer (`bindings.h`, `_dispatch_macro.h`, `dispatch.cpp`, all per-optimizer dispatchers) to 8 arches
- `0fe9cc4` — extend `tuned_configs.h` (`ArchId` enum + table widths 4→8) and `setup.py` (`-gencode` for sm_89/103/120, `--offload-arch=gfx950`)
- `40954ae` — extend `dispatch.py` (SUPPORTED_ARCHES, detection, label table) and `autotune/` (NVFP4/FP4/FP6 grids, sm_89/103a/120a CUTLASS targets)
- `58b9e54` — extend tests (`test_cross_arch_agreement.py`, `test_all_arches.py`, `test_amd_hip.py` covering both gfx942 + gfx950)

---

## Contents

1. Repo layout
2. Project state
3. Optimizers
4. Python infrastructure
5. csrc/common — shared headers
6. csrc/bindings — pybind11 layer
7. csrc/kernels/cuda — NVIDIA per-arch kernels
8. csrc/kernels/hip — AMD per-arch kernels
9. csrc/kernels/tpu — TPU Pallas kernels
10. csrc/quantization — quantization kernels
11. Algorithms
12. JAX/TPU
13. Tests
14. Benchmarks
15. Autotune
16. Build
17. Recent commits
18. Known gaps
19. Quick reference

§24 is the per-arch per-optimizer rundown; §25 is the engineering-remaining list.

---

## 1. Repo layout

- `grokking_optimizers/` — Python package, eleven optimizers plus infra
- `supergrok2_jax_tpu/` — JAX/TPU port of the suite (Pallas kernels live in `csrc/kernels/tpu/`)
- `csrc/common/` — shared headers (`platform.h`, `types.h`, `arch_tier.h`, `ptx_intrinsics.cuh`, `utils.cuh`, `quantization.h`, `fp4_helpers.hip.h`) plus `tuned_configs.h` (autotune output)
- `csrc/bindings/` — per-optimizer dispatchers (`grokadamw.cpp`, `lion.cpp`, `supergrok2.cpp`, …) + arch detection (`dispatch.cpp`) + pybind11 module aggregator (`module.cpp`) + helpers (`_helpers.h`, `_dispatch_macro.h`, `bindings.h`)
- `csrc/kernels/cuda/sm_80/` `sm_89/` `sm_90/` `sm_100/` `sm_103/` `sm_120/` — NVIDIA per-arch kernels (six arches), each in `namespace sg::sm<N>`
- `csrc/kernels/hip/gfx942/` `gfx950/` — AMD per-arch kernels (two arches), each in `namespace sg::gfx<N>`
- `csrc/kernels/tpu/v5p/` `v6e/` — TPU Pallas kernels per version (re-export tile-128 / tile-256 from shared `_pallas_kernels.py`)
- `csrc/kernels/cpu/` — CPU implementations with AVX-512 / NEON SIMD (testing only, not a runtime fallback)
- `csrc/quantization/` — quantization kernels (FP8, INT8, INT4, MXFP4)
- `autotune/` — offline tuning: `tune.py`, `grids.py`, `runner.py`, `cutlass_profile.py`, `_wrap_kernel.py`
- `tests/` — eleven test files including `test_cross_arch_agreement.py` (renamed from `test_all_tiers.py` → `test_all_arches.py`)
- `benchmarks/` — `benchmark_supergrok2.py`, `autotune.py`, `training_benchmark.py`, `profile_smoke.py`
- `setup.py` — build entry, ninja-backed, supports the eight GPU arches with multi-arch fatbin
- `pyproject.toml` — build-system requires (ninja, torch); version 3.0.0
- `build.sh` — tqdm-progress wrapper around `pip install -e . --no-build-isolation -v`
- `third_party/cutlass` — CUTLASS v3.6.0 submodule for sm_90a / sm_100a / sm_103a / sm_120a GEMMs (opt-in via `WITH_CUTLASS=1`)
- `README.md` — user docs
- `REFRESH.md` — this file
- `ANALYSIS.md` — internal review with bug findings and optimization opportunities

## 2. Project state

- Branch: `claude/custom-optimizer-analysis-HFYhg`
- Working tree: clean
- Size: 180,845 LOC across 267 tracked files (see §0 Code statistics for the per-language breakdown)
- Package version: `3.0.0` (breaking refactor — bumped in commit `c8022cc`)
- Backends supported: NVIDIA `sm_80, sm_89, sm_90, sm_100, sm_103, sm_120`; AMD `gfx942, gfx950`; TPU `v5p, v6e`. CPU build for testing only.
- Permanently unsupported: V100, T4, RTX 20-series, MI100 (gfx908), MI200 (gfx90a), AMD RDNA, TPU v3/v4/v5e.
- Status: structural refactor complete; per-arch hand-tuning + CUTLASS migration + autotune execution deferred to hardware-validated sessions (see §0 "What is NOT yet done").
- Recent focus: arch matrix expansion (6 NVIDIA + 2 AMD = 8 GPU arches, up from 4); per-arch wrapped baselines; bindings layer split; autotune scaffolding.
- Architecture: SuperGrok v2 design settled long ago; the refactor changes how kernels are organized (single per-arch source-of-truth instead of generic + overlay), not what they compute.
- Last 5 commits, newest first:
  - `8a100ac` — REFRESH.md §0 with 8-arch supported set + expansion commits
  - `58b9e54` — extend tests for sm_89/103/120 + gfx950
  - `40954ae` — extend dispatch.py + autotune for new arches
  - `0fe9cc4` — extend tuned_configs.h + setup.py for 8-arch matrix
  - `5b4218b` — extend bindings layer to recognize 4 new arches

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

### 3.12 Race fairness model

The grokking race uses four outer train/test splits (10/90, 25/75, 50/50, 80/20) with an inner val carve-out controlled by `val_ratio` (default 0.10; auto-overrides to 0.05 on 10/90 to avoid near-empty val sets). A fixed early-stopping rule ends each run at whichever comes first: test accuracy reaching 95% or step count reaching 20,000 — both thresholds are CLI-configurable and identical across all 11 optimizers, so test is "selection-free" for stopping (no hyperparameter is being chosen by it). Three SG variants (v2, v1.5, v1.1) consume the inner val natively for bilevel and meta updates; the other eight train on train only and never see val during optimization. The val/test gap (`final_val_acc - final_test_acc`) in the output is the key diagnostic for distinguishing meta-learning from masked overfitting on the val signal.

## 4. Python infrastructure

### `dispatch.py`
- Detects backend at runtime, no GPU import required.
- `get_gpu_vendor()` → 'nvidia' | 'amd' | 'none'
- `get_gpu_arch()` → one of `{80, 89, 90, 100, 103, 120, 942, 950}` or raises `UnsupportedArchError`
- `get_backend()` → 'cuda' | 'hip' | 'cpu'
- `get_warp_size()` → 32 (NVIDIA) or 64 (AMD CDNA)
- `assert_supported_arch()` and `SUPPORTED_ARCHES` constants — public surface for the no-fallback policy
- `supports_bf16/fp8/tf32/tma/block_clusters/matrix_cores/nvfp4` predicates
- Env override: `FORCE_ARCH=N`
- Tier helpers (`get_arch_tier`, `get_amd_tier`, `get_amd_label`) are gone — the eight arches each compile their own per-arch kernel TUs, no tier fallback

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

### `_ops_loader.py` and `_python_fallback.py`
- `_ops_loader.py` imports the compiled `_ops` extension or raises a clear error if missing — there is no runtime JIT and no graceful "skip the kernel" path under the no-fallback policy.
- `_python_fallback.py` provides pure-PyTorch correctness implementations for SG2's meta-net (used by `tests/` and as a development fallback when `_HAS_OPS` is false).

### `__init__.py`
- Exports all eleven optimizers
- Meta-net classes: Mamba3PEERMetaNet, Mamba3ScanBlock, MiniGRU, SharpnessMetaNet
- Wrappers: CompiledSuperGrok2, CUDAGraphOptimizer, OverlappedOptimizer, PipelinedOptimizer, GradientHookOptimizer, AsyncSuperGrok2, MoEAwareSuperGrok2
- Distributed helpers, dispatch helpers, PrecisionConfig
- Flag: `_HAS_OPS` — extension or error (no `_HAS_CUDA` / `_HAS_CPU_OPS` split under the simplified loader)
- `__version__ = "3.0.0"`

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

### `quantization.h`
- `PrecisionMode` enum: FP32, TF32, BF16, FP8_E4M3, INT8_SYM, INT4_GPTQ, MXFP4
- Device-side dequant helpers:
  - `dequant_int8(q, scale)` — symmetric, per-tensor
  - `dequant_int4(packed, which, scale, zero)` — group_size=32, asymmetric
  - `dequant_mxfp4(packed, which, shared_exp)` — block_size=32 shared exponent
  - `fp4_e2m1_to_float` — lookup table {0, 0.5, 1, 1.5, 2, 3, 4, 6}

### `arch_tier.h`
- Lightweight shim providing `ArchTier` / `StatePrecision` / `ExpertPrecision` enums and a per-TU constexpr `kArchTier` selected by `SG_ARCH_<X>` preprocessor switches.
- Recovered for distributed-pipeline TUs that referenced the deleted `dispatch.h`. Lets legacy `ArchTier::HOPPER`-style call sites keep compiling without body edits.
- `StatePrecision`: FP32, CONFIG4 (INT8 state), FP6 (CDNA4)
- `ExpertPrecision`: FP32, INT8, INT4, MXFP4, FP4 (CDNA4)

### `tuned_configs.h`
- Auto-generated launch-config table indexed by `(ArchId, KernelId)`.
- `ArchId` enum spans 8 arches: `{kSm80, kSm89, kSm90, kSm100, kSm103, kSm120, kGfx942, kGfx950}`.
- Each entry: `LaunchConfig { int block; int grid; int warps_per_block; int smem_bytes; }`.
- The autotune sweep writes winners between `// AUTOTUNE_BEGIN` / `// AUTOTUNE_END` markers; hand-tuned defaults remain outside the markers.

### `fp4_helpers.hip.h`
- Shared FP4/FP6 dequant + Philox stochastic-rounding helpers used by all four post-split gfx950 per-feature files (`fp4_expert`, `fp6_state`, `sparse24`, `fused_combos`).
- All helpers `__device__ static __forceinline__` so each TU gets internal-linkage copies (avoids ODR errors). The `__constant__` LUT is wrapped in an anonymous namespace.

(The deleted `csrc/common/{ops.h,ops.cpp,dispatch.h}` headers are replaced by the bindings layer described in §6.)

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

Nine files, ~3,120 LOC. Total test points ~92.

> **Post-refactor update.** `tests/test_cross_arch_agreement.py` — the
> safety net for the all-specialized refactor — is no longer a stub.
> Every per-optimizer test (`test_<optimizer>_cross_arch`) runs the
> optimizer for N steps under each available `FORCE_ARCH` and asserts
> elementwise `allclose()` across arches with tolerance `1e-4`. Arches
> not compiled into the local extension are skipped. The Muon harness
> uses 2D params (64×64 matrix) since its Newton-Schulz orthogonalization
> requires `dim ≥ 2`. The whole class skips if no GPU is available.
> See §0.5 + §21 for what arches are expected to agree.

> The legacy CPU-fallback test `tests/test_cpu_fallback.py` had a
> `test_setup_cpu_sources` checker that referenced `csrc/cpu/generic/`
> paths deleted in `95b77e0`. It is now `test_setup_kernel_sources`,
> which checks `setup.py` walks the post-refactor `csrc/kernels/` tree.

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

### `test_race_split.py` (10 sections)
- Split arithmetic for 80/20+10% and 10/90+5% configurations
- Disjointness of train/val/test index sets via set comparison
- Deterministic split (same seed = same split)
- val_ratio auto-override to 0.05 on 10/90
- EarlyStopper stopping_reason for max_steps and test_acc_threshold
- TrainResult output schema completeness (all JSON columns present)
- Pure arithmetic tests (no C++ extension needed)
- Skips gracefully when `_HAS_OPS` is false

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

### `grokking_race_v2.py` (race driver)
- 11 optimizers × 3 architectures × 4 train/test splits × multi-seed
- 3-way train/val/test split with val carved from train (see §3.12)
- Early stopping triggers on test_acc ≥ 95% or 20k steps (whichever
  first); test is the outer held-out portion. Val is the inner carve-
  out consumed only by SG variants.
- CLI: `--optimizers`, `--seeds`/`--num-seeds`, `--tasks`, `--train-test-ratios`, `--val-ratio`, `--early-stop-test-acc`, `--early-stop-max-steps`, `--eval-every`, `--output`
- Per-step train/val/test eval at `eval_every` intervals + final
  `model.eval()` + `no_grad` test eval in `_fin()`
- JSON output includes: optimizer, seed, task, train_test_ratio, val_ratio, stopping_reason, stopping_step, final_val_acc, final_val_loss, final_test_acc, final_test_loss, val_test_gap, wall_clock_seconds, plus full train/val/test curves
- Multi-GPU support via `--gpus`; ntfy.sh notifications via `--ntfy`

### Fairness notes (from `ANALYSIS.md` §3)
- Same init, multi-GPU round-robin, multi-seed bands ✓
- SuperGrok optimizers do extra per-step work (meta-net forward, SAM, bilevel) — wall-clock not directly comparable
- SuperGrok bilevel uses validation data → information advantage vs other optimizers (now properly separated from test set)
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

The migration is complete: eight GPU arches plus two TPU versions
under all-specialized per-arch kernels with no fallback chain; ninja
AOT build with multi-arch fatbin and a build wrapper that supports
autotune, debug, profile, and redistributable-tarball modes; CUTLASS
submodule scaffolding with an opt-in build flag; and Hopper FP8 plus
CDNA3 BF16 plus Ampere TF32 fast paths inlined into the canonical
launchers they belong to. For the chronological commit list, see
`git log --oneline` (or `git log --oneline --first-parent` for major
phases). For the live engineering-remaining list, see §25.

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


## 24. Per-arch per-optimizer rundown

For each of the eleven optimizers, what each arch does. One short
paragraph per cell. Hot-path file paths live in §0; this section is
pure English. Abbreviations on first use per optimizer section.

### 24.1 SuperGrok v2 — Mamba-3 + PEER + GRU meta-net

The largest optimizer. Per step it runs an affine prefix scan over
packed segments, an in-place gated recurrent unit, a product-key
expert router into a small mixture-of-experts MLP, then an Adam plus
LAMB update. Five projection matmuls dominate (input, gating,
discrete-time projection with a softplus epilogue, and the two Mamba
state projections). PEER means product-key expert routing. PTX
helpers used across all NVIDIA arches: the inline-assembly affine
combine for the parallel scan, branchless float-to-int8 stochastic
rounding, fast inverse square root with one Newton-Raphson step,
fast sin-cos pair, and gate-pair sigmoid. AMD arches use the same
math via portable C++ fallbacks for those helpers.

**sm_80 (Ampere — A100, A30, A10).** Canonical math. Async copies
double-buffer the projection weights and expert tiles into shared
memory. The cuBLAS handle is set to TF32 tensor-core mode for the
projection matmuls and restored on scope exit. Affine-combine PTX
helper drives the inner scan; non-temporal global stores on the
update writeback. Shared memory budget around 100 KB per streaming
multiprocessor on RTX 30 and 164 KB on A100; both fit the chosen
tile sizes. Register pressure roughly 96 per thread; occupancy
target two blocks per multiprocessor. cuBLAS handles all five
projection matmuls. Reductions go through warp shuffles plus an
atomic add per warp into a small scratch. Hand-tuning remaining:
async-copy pipeline depth (three versus four stages), per-segment
block size, and expert-tile shape for varying expert count and
hidden size. A10 has 24 GB so very large state-dimension and
expert-hidden combinations spill workspace; fall back to a smaller
state dimension on A10.

**sm_89 (Ada Lovelace — RTX 40, L40, L40S).** Today the wrapped
baseline ported from sm_90. Same math as sm_80 with TF32-mode
projection matmuls. Ada has FP8 four-bit-mantissa tensor cores but
the bindings still pass FP16 to the projections, so FP8 is not yet
on this path. Shared memory 100 KB per multiprocessor. cuBLAS for
matmuls; reductions identical to sm_80. CUTLASS is not engaged here
even with the build flag — sm_89 explicitly stays on cuBLAS to keep
parity with the Ampere baseline. Hand-tuning: small-batch occupancy
on the consumer Ada cards, and a future FP8 projection path that
would give the largest absolute win on this arch. L40S has no
NVLink, so the distributed pipeline files fall back to PCIe ring
all-gather.

**sm_90 (Hopper — H100, H200).** Canonical math plus an inlined
Hopper FP8 four-bit-mantissa fast path for the five projection
matmuls. The FP8 helpers live in the per-arch namespace as static
inline functions, gated behind a CUDA-version preprocessor check.
The path activates only when the total sequence length is at least
the matmul-precompute threshold and all of state-dim, inner-dim,
and model-dim are at least 64. The warp-specialized scan kernels
(generic-state and unrolled state-dim 16) are declared in this
namespace but are not wired into the canonical batched-step yet.
Tensor-memory-accelerator descriptors are scaffolded only on
sm_100; sm_90 does not use them. Affine-combine PTX helper is
shared. Distributed shared memory across cooperative-thread-array
clusters could collapse the per-warp norm reductions but is
unwired. Shared memory budget 228 KB per multiprocessor; register
pressure tighter at roughly 80 per thread to keep two blocks per
multiprocessor. With the CUTLASS build flag the projection matmuls
can alternately go through CUTLASS sm_90a's FP16 or BF16 paths
instead of cuBLAS FP8; the choice is per-build today, not
per-shape. Reductions: warp shuffles plus per-warp atomic.
Hand-tuning: producer-consumer warp-specialization ratio,
cooperative-thread-array cluster size (Hopper supports 16-CTA
clusters), and tail-effect on small batches. The FP8 path is wired
but unreachable until the small dim caps (model-dim 16, state-dim
32, inner-dim 32) in the shared types header are raised above 64;
see §25.

**sm_100 (Datacenter Blackwell — B100, B200, GB200).** Canonical
math; tensor-memory-accelerator-style precompute scaffolding lives
in a dedicated source file but is not yet activated. NVFP4 (the
native four-bit Blackwell format with shared-exponent blocks of 16)
is not active here — that lives on sm_103. Shared memory budget 228
KB per multiprocessor; register pressure very high at roughly 112
per thread for the fused full-step launcher, occupancy one block per
multiprocessor at full feature set. cuBLAS for matmuls or CUTLASS
sm_100a with the build flag. Hand-tuning: tensor-memory-accelerator
descriptor reuse across scan segments, fourth-generation tensor-core
utilization for the discrete-time-projection softplus epilogue
(currently unfused — the post-pass softplus runs separately).
Cooperative-thread-array cluster topology favors 16-CTA clusters
on B100 and GB200 but the scan kernel does not yet cross-CTA-share
its reductions through distributed shared memory.
**sm_103 (Blackwell Ultra — B300, GB300 NVL72).** Canonical math
with the NVFP4 hot path scaffolded for the projection matmuls
through CUTLASS sm_103a. Shared memory matches sm_100 at 228 KB.
With the CUTLASS build flag the matmuls go through CUTLASS sm_103a;
without it they fall back to cuBLAS FP16, since cuBLAS lacks NVFP4
on Blackwell Ultra at the time of writing. The autotune grid lists
sm_103a NVFP4 entries; the profiler binary needs running on
hardware to populate them. Hand-tuning: NVFP4 block-scaling factor
calibration. Gotcha: NVFP4 requires careful per-block scale handling
to avoid scale-overflow at the tails.

**sm_120 (Consumer Blackwell — RTX 50, RTX PRO 6000).** Canonical
math. Uses CUTLASS sm_120a target with the build flag. Shared
memory is 128 KB per multiprocessor here, significantly less than
sm_100 and sm_103's 228 KB. The current placeholder launch configs
in the shared tuned-configs header mirror sm_100 and will under-
occupy on sm_120; the autotune sweep will detect this. Hand-tuning:
shared-memory tile reduction (likely halve the segment block size).
NVFP4 and FP4 are both present on RTX PRO 6000 but consumer RTX 50
cards have varying tensor-core mixes. RTX 5090 has 32 GB versus RTX
PRO 6000's 96 GB — respect the workspace ceiling.

**gfx942 (CDNA3 — MI300X, MI300A).** Canonical math plus an inlined
CDNA3 BF16 matrix-fused-multiply-add fast path for the batched
step. The pipeline runs setup-and-sort, then a BF16 precompute pass
that converts to BF16 and uses rocBLAS (which dispatches to the
32-by-32-by-8 BF16 matrix-fused-multiply-add instruction), then the
scan and fused-element pass. Local data store budget per compute
unit is 64 KB; register pressure is moderate at roughly 64 vector
registers. MI300X has 192 GB high-bandwidth memory so workspace is
effectively unlimited. rocBLAS handles matmuls; CUTLASS is not
engaged on AMD arches. Reductions: wave-reduction primitives
analogous to NVIDIA warp shuffles. Hand-tuning: BF16 matrix-fused-
multiply-add tile shape, local-data-store double-buffer depth, async
copy queue depth. Gotcha: MI300A's unified memory between CPU and
GPU means parameter buffers can live on CPU pages; verify the kernel
sees device pointers.

**gfx950 (CDNA4 — MI350X, MI355X).** Canonical math; the FP4 expert
matrix-fused-multiply-add path lives in the FP4-expert per-feature
file (split this session from the original CDNA4 monolith). Expert
weights are stored as packed FP4 (eight values per 32-bit word) in
high-bandwidth memory, dequantized to FP32 in local data store via
the shared FP4 helpers header, and the matrix multiply itself uses
the 16-by-16-by-128 FP4 matrix-fused-multiply-add instruction. FP6
shared-exponent state packing lives in a sibling file; structured
2:4 sparsity in another. Local data store 64 KB per compute unit.
rocBLAS for the FP32 fallback; native FP4 matrix-fused-multiply-add
for expert weights; no CUTLASS. Hand-tuning: FP4 quantization-scale
calibration, local-data-store bank-conflict avoidance, FP6 unpack
throughput versus the affine-scan recurrence rate. Gotcha:
stochastic rounding for FP4 quantization uses a Philox hash; the
seed must be deterministic across distributed ranks.

**TPU v5p (128-wide matrix-multiply unit).** The JAX-Pallas
implementation tiles the prefix scan and projections for the
128-lane matrix-multiply unit. The discrete-time-projection
softplus epilogue is naturally fused via the JAX accelerated linear
algebra fusion pass. BF16 throughout; no FP4, FP6, or FP8 paths.
High-bandwidth memory is 32 GB per chip; v5p pods scale to 8960
chips. Reductions go through accelerated linear algebra all-reduce.
Hand-tuning: the pjit sharding spec for state dimension and expert
count; warm the host accelerated linear algebra cache. Gotcha: TPU
v5p does not run custom CUDA kernels — the entire SG2 path is Pallas
on TPU and the optimizer ops Python module is unused.

**TPU v6e (256-wide matrix-multiply unit).** Same math as v5p,
re-tiled for the 256-lane matrix-multiply unit. Roughly 2x v5p
throughput on the projection matmuls; the prefix scan is bound by
high-bandwidth memory bandwidth in both cases and gains less.
Hand-tuning: lane-256 tile shape for the expert MLP — wider than
typical expert hidden sizes (8 to 32), so most tiles will be padded
unless the Pallas kernel slices explicitly to avoid wasted matrix-
multiply-unit cycles.
### 24.2 SuperGrok v1.5

A grokking-aware optimizer with a small two-layer multi-layer
perceptron meta-net (typical hidden size 16 to 128), Lamb-style
trust-ratio update, fused sharpness-aware-minimization perturb and
sharpness restore, and per-step gradient clipping. The hot path is
the fused full-step launcher with twenty-one arguments. SAM stands
for sharpness-aware minimization; its two phases (perturb and
restore) are exposed as separate launchers. PTX helpers used: the
inline gate-pair sigmoid for the meta-net activations, fast inverse
square root with one Newton-Raphson step for the Adam denominator,
and float-4 vectorized loads on aligned tails.

**sm_80.** Canonical math. The async-copy pipelined weight-load
variant is now the canonical body — it sets cuBLAS to TF32 mode via
a scope helper, then dispatches to one of four fully-unrolled
hidden-size templates (16/32/64/128) or the runtime-size async-copy
fallback. Shared memory budget around 16 KB per block (four weight
tiles plus a small scratch). Register pressure low (~48); occupancy
target four blocks per multiprocessor. cuBLAS handles the
projection step; the meta-net itself fits in registers per block.
Hand-tuning: per-hidden-size launch bounds, async-copy stage count.

**sm_89.** Canonical math; baseline ported from sm_90. Same as
sm_80 today. FP8 path on the meta-net is not active — the meta-net
is small (~64 to 256 parameters) so FP8 buys little versus the
async-copy pipelined weight-load win. cuBLAS TF32 for the trust-
ratio matmul. At 100 KB shared memory per multiprocessor, lower
occupancy is acceptable because the per-block footprint is small.

**sm_90.** Canonical math; baseline. FP8 deferred — the meta-net
weights are FP32 in the current Python optimizer and don't benefit
from FP8 conversion at this scale. The warp-specialized scan does
not apply here (the full-step is element-wise in the parameter
dimension, not a recurrence). Hand-tuning: hidden-size 128
specialization for Hopper's larger register file.

**sm_100.** Canonical math; baseline. Tensor-memory-accelerator
does not apply (no large matmul in the hot path). Shared memory
ample at 228 KB. Hand-tuning: launch bounds for B100 versus B200
multiprocessor count differences; not much to gain from Blackwell-
specific features for this optimizer.

**sm_103.** Canonical math; baseline. NVFP4 inapplicable to a
small meta-net. Same notes as sm_100.

**sm_120.** Canonical math; baseline. Shared memory 128 KB but the
per-block footprint is around 16 KB so the constraint doesn't bite.
High-occupancy targeting on RTX 50 and RTX PRO 6000.

**gfx942.** Canonical math; baseline. BF16 matrix-fused-multiply-add
is marginal here — the matmul is too small to amortize MFMA setup
cost. Local data store 64 KB per compute unit; well under.

**gfx950.** Canonical math; baseline. FP4 expert matrix-fused-
multiply-add doesn't apply (no expert mixture in v1.5). Inherits
the gfx942 "matrix-fused-multiply-add marginal" gotcha.

**TPU v5p.** The JAX-Pallas tiles the meta-net's two-layer multi-
layer perceptron across the 128-wide matrix-multiply unit; bias
corrections and Lamb trust-ratio run in parallel via the
accelerated linear algebra fusion pass. BF16. SAM perturb and
restore are fused into pjit graph regions.

**TPU v6e.** Identical to v5p; tiled for the 256-lane matrix-
multiply unit. Meta-net is small enough that v6e's 2x throughput
shows up only in the projection step. Hand-tuning: avoid padding
the matrix-multiply unit below 256 lanes when the meta-net hidden
size is smaller — pack multiple parameters' meta-nets into a single
matrix-multiply-unit launch.

### 24.3 SuperGrok v1.1

Predecessor of v1.5. Same meta-net-based grokking-aware structure
but a simpler 2-phase pipeline: meta-net update and computation,
then a runtime-computed cosine gate, then an Adam plus weight-
decay step. SAM perturb and sharpness restore exposed as separate
launchers.

**sm_80.** Canonical math; baseline meta-net kernel wraps a TF32
scope for the matmul. Shared memory budget similar to v1.5 (~16
KB per block). Register pressure low (~48). cuBLAS TF32. Hand-
tuning: cosine-gate fusion into the meta-net kernel (currently a
separate kernel call).

**sm_89.** Baseline. Same notes as v1.5 sm_89.

**sm_90.** Baseline. Hopper FP8 deferred for the same meta-net-too-
small reason as v1.5.

**sm_100.** Baseline. Tensor-memory-accelerator inapplicable.

**sm_103.** Baseline. NVFP4 inapplicable.

**sm_120.** Baseline. Shared-memory constraint doesn't bite.

**gfx942.** Baseline. BF16 matrix-fused-multiply-add marginal.

**gfx950.** Baseline.

**TPU v5p.** Pallas tiling for 128-wide matrix-multiply unit. The
meta-net update, cosine gate, and Adam-decay pipeline run as a
single accelerated linear algebra graph, so the intermediate
cosine-gate value never leaves staticRAM.

**TPU v6e.** Tile-256. Same shape as v5p; 2x throughput on the
meta-net matmul.
### 24.4 GrokAdamW

Plain Adam with decoupled weight decay plus grokking-detection
scheduling and a slow-fast parameter ramp. The hot path is the
fused step, which merges parameter update, decay, and bias-
correction across all parameters in one launch. Multi-tensor variant
batches small parameter tensors per block. PTX helpers used: fast
inverse square root with one Newton-Raphson step (NVIDIA), float-4
vectorized loads on aligned tails. AMD uses portable C++ for the
inverse-square-root fallback.

**sm_80.** Canonical math; baseline. Async-copy pipelining for the
multi-tensor parameter list (each block handles one parameter).
Cuda-graph capture friendly. Shared-memory budget ~4 KB per block.
Register pressure low. No matmul in the hot path. Hand-tuning:
per-block parameter batching to coalesce small tensors into larger
work units.

**sm_89.** Baseline. Identical to sm_80 today.

**sm_90.** Baseline. Distributed shared memory across cooperative-
thread-array clusters could cross-CTA-share the global gradient
norm; not yet wired.

**sm_100.** Baseline. Tensor-memory-accelerator inapplicable.

**sm_103.** Baseline. The Lamb trust-ratio matmul could use NVFP4
but is very low priority — small relative cost.

**sm_120.** Baseline. Shared memory ample at 128 KB.

**gfx942.** Baseline. Local data store under-used.

**gfx950.** Baseline. FP4 expert matrix-fused-multiply-add
inapplicable.

**TPU v5p.** The Adam step is naturally vectorizable; runs at
matrix-multiply-unit peak throughput when shape is divisible by 128.

**TPU v6e.** Tile-256.

### 24.5 NeuralGrok

Grokking optimizer with a two-layer multi-layer perceptron
"amplifier" that scales the effective gradient direction. Smaller
than v1.5's meta-net but similar shape. Has a per-step amplifier-
net pass.

**sm_80.** Canonical math; baseline amplifier kernel inherits the
TF32 scope from sm_80's meta-net variant inlining. Shared memory
~8 KB per block. Hand-tuning: amplifier hidden-size specialization
analogous to v1.5.

**sm_89.** Baseline.

**sm_90.** Baseline. Same FP8-deferred reasoning as v1.5.

**sm_100.** Baseline.

**sm_103.** Baseline. NVFP4 inapplicable.

**sm_120.** Baseline. Shared memory easy.

**gfx942.** Baseline.

**gfx950.** Baseline.

**TPU v5p.** Pallas tile-128 for the amplifier's two-layer multi-
layer perceptron.

**TPU v6e.** Tile-256.

### 24.6 Prodigy

Adaptive learning-rate optimizer that learns its own d_lr scaling
factor. Returns the updated d_lr from the C++ binding. Hot path is
the fused step plus a multi-tensor fused-reduce-step variant for
batched parameter lists.

**sm_80.** Canonical math; baseline. Element-wise update; no
matmul. Shared memory trivial. Async-copy unhelpful (no shared-
memory weight reuse). Hand-tuning: warps-per-block for maximum
occupancy on the fused update plus reduction.

**sm_89.** Baseline.

**sm_90.** Baseline. Distributed-shared-memory cross-CTA
reductions could improve the adaptive-learning-rate aggregation;
deferred.

**sm_100.** Baseline.

**sm_103.** Baseline.

**sm_120.** Baseline. Shared memory inapplicable (no usage).

**gfx942.** Baseline. CDNA3 BF16 matrix-fused-multiply-add
inapplicable (no matmul).

**gfx950.** Baseline.

**TPU v5p.** Pallas; the d_lr accumulation is a scalar reduction
across all parameter tensors — implicitly a parallel reduction in
the accelerated linear algebra fusion pass.

**TPU v6e.** Tile-256.

### 24.7 Grokfast

Two modes: an exponential-moving-average-only mode and a Grokfast-
EMA-plus-Adam variant. Both run a single fused element-wise pass
per parameter.

**sm_80.** Canonical math; baseline. Element-wise; no matmul.
Shared memory trivial. Hand-tuning: combined parameter batching for
many small tensors (currently one block per parameter).

**sm_89.** Baseline.

**sm_90.** Baseline.

**sm_100.** Baseline.

**sm_103.** Baseline.

**sm_120.** Baseline.

**gfx942.** Baseline.

**gfx950.** Baseline.

**TPU v5p.** Trivially Pallas-vectorizable.

**TPU v6e.** Tile-256.
### 24.8 Lion

Sign-momentum optimizer. The simplest hot path: one fused element-
wise step that takes the sign of the interpolated momentum, applies
decoupled decay, and updates the momentum.

**sm_80.** Canonical math; baseline. Pure element-wise; no matmul.
The sign function compiles to a select-predicated PTX instruction
(branchless, warp-uniform). Float-4 fast path on aligned tails.
Non-temporal stores on the parameter writeback.

**sm_89.** Baseline.

**sm_90.** Baseline.

**sm_100.** Baseline.

**sm_103.** Baseline.

**sm_120.** Baseline.

**gfx942.** Baseline. AMD's equivalent of the select-predicated
instruction is a wave-conditional move.

**gfx950.** Baseline.

**TPU v5p.** Pallas; trivially vectorizable. The sign function is
a single accelerated linear algebra primitive.

**TPU v6e.** Tile-256.

### 24.9 LookSAM

Sharpness-Aware Minimization variant with periodic direction
caching. Three entry points: SAM-style perturb across all
parameters, restore from a saved backup, and a fused compute-
directions-and-adjust step that batches the two norm reductions
into a single CPU sync. The norm reductions are the bottleneck.

**sm_80.** Canonical math; baseline. The two-sync reduction
(versus N syncs in the naive form) is the key optimization: stacks
all parameter norms into one device tensor and does a single CPU
sync to read them. Shared memory trivial. Hand-tuning: warp-
shuffle reduction over the per-parameter chunks.

**sm_89.** Baseline.

**sm_90.** Baseline. Distributed shared memory could combine all
cooperative-thread-arrays' reductions without going through global
memory; deferred.

**sm_100.** Baseline.

**sm_103.** Baseline.

**sm_120.** Baseline.

**gfx942.** Baseline. Wave-reduction primitives on CDNA3 are
similar to NVIDIA warp shuffles.

**gfx950.** Baseline.

**TPU v5p.** Pallas; the global-norm reduction is a single
accelerated linear algebra all-reduce at the host side.

**TPU v6e.** Tile-256.

### 24.10 Muon

Newton-Schulz orthogonalization optimizer for 2D parameters. The hot
path is a 5-step iteration applied to a momentum-normalized
direction with three quintic-coefficient mixes, two matrix multiplies
per step. Has been the canonical worked-example for arch divergence:
TF32 on Ampere, FP8 on Hopper, BF16 matrix-fused-multiply-add on
CDNA3, all inlined into the canonical fused-step body.

**sm_80.** Canonical math plus an Ampere TF32 fast path. Opens a
TF32 scope helper that sets the cuBLAS handle to TF32 tensor-core
mode for the Newton-Schulz matmul chain and restores on scope exit.
Roughly 2x speedup over plain FP32 on A100. cuBLAS handles the
matmul; CUTLASS not engaged on sm_80 (kept on cuBLAS per the build-
flag policy). Hand-tuning: Newton-Schulz step granularity (current
launcher unrolls all 5 steps in one call).

**sm_89.** Canonical math plus TF32 (same as sm_80). FP8 matmuls not
yet wired here — would buy roughly 4x over FP32 like Hopper, but
consumer Ada's small multiprocessor count limits the absolute win.

**sm_90.** Canonical math plus a Hopper FP8 fast path. A small
helper computes a per-tensor absolute-maximum, casts to FP8 four-
bit-mantissa, runs cuBLAS extended-GEMM with FP32 accumulation; used
for the leading transposed-product matmul when both dimensions are
at least 64. Smaller matmuls stay FP32. Roughly 4x speedup on H100.
With the CUTLASS build flag the matmul can alternatively go through
CUTLASS sm_90a's FP16/BF16 paths instead of cuBLAS FP8 — choice is
per-build today, not per-shape; revisit on hardware. Hand-tuning:
when to cross over from cuBLAS-FP8 to CUTLASS-FP16; FP8 quantization-
scale calibration.

**sm_100.** Canonical math; baseline (no FP8 wrap inlined). CUTLASS
sm_100a routes the matmul with the build flag. Fourth-generation
tensor core could use FP4, but the Newton-Schulz chain is small
(typical 2D parameter is 64x64 to 1024x1024) and FP4 quantization
overhead is non-trivial. Hand-tuning: Newton-Schulz iteration count
for very small matrices (could drop from 5 to 3 with a tighter
spectral-norm bound).

**sm_103.** Canonical math. NVFP4 hot path possible via CUTLASS
sm_103a but not yet activated for Muon (deferred — the spectral-norm
bounds for NVFP4 versus FP16 Newton-Schulz are not yet validated).

**sm_120.** Canonical math; CUTLASS sm_120a path for the matmul.
Shared memory 128 KB — adequate.

**gfx942.** Canonical math plus a CDNA3 BF16 matrix-fused-multiply-
add fast path. When the leading dimension is at least 128 and the
parameter is 2D, converts to BF16 and runs the full Newton-Schulz
chain through rocBLAS (which dispatches to the 32-by-32-by-8 BF16
matrix-fused-multiply-add). Roughly 2x speedup. Below leading
dimension 128 falls back to FP32. CUTLASS not engaged on AMD.

**gfx950.** Canonical math; baseline. CDNA4 has FP4 matrix-fused-
multiply-add but Muon is not (yet) routed through it — same reason
as sm_103: spectral-norm validation needed. rocBLAS for the matmul.

**TPU v5p.** Pallas; the Newton-Schulz chain is 5 matmuls per step,
all naturally matrix-multiply-unit-accelerated. BF16. Tile-128.

**TPU v6e.** Tile-256. The Newton-Schulz matmuls are usually too
small (64 to 1024) to fully fill the 256-wide matrix-multiply unit;
expect modest speedup over v5p.

### 24.11 MoE / Mamba3PEER auxiliary entries

Beyond v2's own mixture-of-experts-routed expert MLP, a separate
binding surface exposes nine entry points for use outside v2:
expert load, forward, backward, active-parameter filtering,
compacted scan, scatter-results, expert-activation counting, load-
balance loss, and frequency scaling. These power the Mamba3PEER
block in `grokking_optimizers/mamba3_peer_metanet.py` which can run
independently of any optimizer.

**sm_80.** Canonical math; baseline. Token routing uses a top-k
selection kernel; expert weights loaded with async-copy pipelining.
Hand-tuning: top-k threshold for L1-versus-shared-memory expert
weight caching.

**sm_89.** Baseline.

**sm_90.** Baseline. Hopper warp specialization is the highest-
value wire-up here (each expert's MLP becomes a producer-consumer
pipeline).

**sm_100.** Baseline. Tensor-memory-accelerator descriptor reuse
across expert loads is the highest-value Blackwell win.

**sm_103.** Baseline. NVFP4 expert weights on sm_103a would be a
direct mixture-of-experts quantization win — needs profiling
against accuracy.

**sm_120.** Baseline. Shared memory 128 KB constrains how many
experts fit in cache simultaneously.

**gfx942.** Baseline. BF16 matrix-fused-multiply-add expert MLP is
the natural fit.

**gfx950.** The FP4 expert matrix-fused-multiply-add hot path —
uses the FP4-expert per-feature file directly via the mixture-of-
experts bindings; eight FP4 weights packed per 32-bit word in high-
bandwidth memory; dequantized to FP32 in local data store via the
shared FP4 helpers header. Stochastic rounding with Philox. This
is the canonical "real divergence" implementation across the arch
matrix.

**TPU v5p.** Pallas; expert routing in a single accelerated linear
algebra shard map; v5p's 8-way pod can hold the full expert table
in high-bandwidth memory.

**TPU v6e.** Tile-256. The 256-wide matrix-multiply unit pairs
naturally with expert hidden sizes of 256+; smaller experts pad and
waste throughput.

## 25. Engineering work remaining

The all-specialized refactor, eight-arch expansion, overlay merges,
SG v2 binding wiring, ninja build wrapper, autotune execution layer,
and CUTLASS migration scaffolding are now complete. What remains is
real per-arch hand-tuning and a small set of items that need
hardware to validate. Rough order of expected payoff:

~~**1. (DONE) Raise type-cap constants.**~~ Completed: MAX_D_MODEL
16→64, MAX_D_STATE 32→128, MAX_D_INNER 32→128. No shared memory
overflow. Hopper FP8 path now reachable.

**1. Hopper warp-specialized scan activation.** The
`launch_scan_warp_specialized` and `launch_scan_warp_specialized_d16`
declarations in `supergrok2_warp_specialized_sm90.cu` are unwired
from the canonical scan launcher. To activate, the canonical
batched-step needs a code-path that picks the warp-specialized
variant when `d_state` is uniform across all parameters in the
batch — typically true for SG2. Expected ~1.5× on H100/H200 for
long-segment workloads.

**2. Real autotune output for tuned_configs.h.** All 17 optimizers
× 8 GPU arches = 136 entries currently use placeholder
`LaunchConfig` values that match hand-coded `__launch_bounds__` in
the per-arch baselines. Run `bash build.sh --autotune` on hardware:
this does a stub-config build, runs `python autotune/tune.py` to
sweep grids, writes winners between the
`// AUTOTUNE_BEGIN`/`// AUTOTUNE_END` markers in
`csrc/common/tuned_configs.h`, then rebuilds. Expected 5–30%
launch-config wins per arch.

**3. Fused softplus epilogue in CUTLASS for SG2 dt_proj.** The
current `cutlass_dt_proj_fused` runs the unfused linear-combo GEMM
plus a separate `softplus_bias_kernel` post-pass. CUTLASS 3.x's
`EpilogueOp` template can fuse `softplus(x + bias)` into the GEMM
tail, saving one elementwise pass over the dt activation. Math is
identical; the API surface change is internal to CUTLASS.

**4. NVFP4 path for Blackwell Ultra (sm_103) projections.** The
CUTLASS sm_103a target is wired in `setup.py` but the SG2 Python
optimizer still passes FP16 / BF16 for the projection inputs. To
activate, the projection precompute (in the Python pre-step) needs
an NVFP4 quantization pass with proper block-scaling factors;
`autotune/grids.py` already lists the sm_103a NVFP4 entries for
the autotune sweep.

**5. sm_120 retuned tile sizes.** Consumer Blackwell has 128 KB
shared memory per SM versus sm_100 / sm_103's 228 KB. Current
placeholder `tuned_configs.h` values for sm_120 mirror sm_100 and
will under-occupy. The autotune sweep above will detect this; the
specific kernels affected are SG2's batched scan and the metanet
cp.async variants.

**6. CDNA4 FP4 / FP6 / 2:4 sparsity engagement beyond MoE.**
Currently only the MoE expert path uses gfx950's native FP4 MFMA.
Wiring NVFP4-equivalent FP4 into the SG2 projections, FP6 state
into the scan recurrence, and 2:4 sparsity into the dt_proj weights
are all open per-experiment opportunities. Profiling required.

**7. DSMEM for cross-CTA reductions on Hopper / Blackwell.** Norm
reductions (LookSAM, GrokAdamW, Prodigy, the SAM step in SG1.5/1.1)
all currently round-trip through global memory. DSMEM (distributed
shared memory across CTA clusters, available on sm_90+) can do
cross-CTA reductions without that round-trip. Expected ~5–10% on
the global-norm step.

**8. Per-feature gfx950 file split refinement.** The post-split
gfx950 files (`fp4_expert`, `fp6_state`, `sparse24`, `fused_combos`)
currently use `__device__ static __forceinline__` helpers in
`fp4_helpers.hip.h` to avoid ODR. Each TU gets its own internal
copy of every helper; the `__constant__` LUT is wrapped in an
anonymous namespace for the same reason. If a future refactor wants
shared helpers (single copy in the binary), they need to move to a
non-template `.cpp` file with explicit `extern` declarations from
each TU.

**9. CI matrix for the eight-row arch sweep.** Tests
(`test_amd_hip.py`, `test_all_arches.py`,
`test_cross_arch_agreement.py` with the Muon 2D harness,
`test_cutlass_parity.py`) exist but the CI runner needs configuring
to exercise the full {sm_80, sm_89, sm_90, sm_100, sm_103, sm_120,
gfx942, gfx950} × {test_*.py} matrix. The cross-arch agreement
test honors `FORCE_ARCH=<n>` so a single multi-build CI image can
run the full matrix.

**10. CPU SIMD test paths.** The `csrc/kernels/cpu/{avx512,neon}/`
files exist but are testing-only and not exercised under any
public test. A small `tests/test_cpu_simd.py` that runs each
optimizer for a few steps on CPU would catch SIMD regressions
without needing a GPU. Low priority.

**11. PyPI-distributable wheel.** `bash build.sh --package-tarball`
already produces a redistributable `dist/` tree plus a
`supergrok2-3.0.0-<sha>.tar.gz` for direct GitHub release upload,
with three documented install paths in `dist/INSTALL.md`. Going from
that tarball to an `auditwheel`-compatible PyPI wheel is a smaller
delta: rerun `python -m build --wheel --no-isolation` inside the
staged tree under a manylinux container. Not urgent while the
project is under active iteration.

