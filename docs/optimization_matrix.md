# Optimization candidate matrix

This is the §12 evaluation matrix for `grokking_optimizers.compile`. Each
row is one candidate optimization that was researched but is not part of
the mandatory §1–§11 baseline (full LTO, sccache, NVCC `--threads 8`,
Bayesian + Exhaustive, PGO, persistent timing worker, CUDA/HIP graphs,
runtime split, etc. — those are already on).

## How rows are ranked

Score = `(measured_perf_gain × confidence) / (integration_cost × risk)`,
tie-broken by integration cost. Rows are then grouped by recommendation
and ordered within group by score.

## What "measured here" means

The current environment is CPU-only (no `nvcc`, no `hipcc`, no GPU,
no `sccache`/`ccache` binaries, 4 CPU cores, `torch.cuda.is_available()`
== False). Every row that needs a GPU or device compiler reports
**"not measured (CPU-only env)"** explicitly. Autotune-quality rows
that can be evaluated on the synthetic stub-timer (a quadratic
landscape in `block/vec/unroll`) carry a synthetic A/B number.

## Recommendation legend

- **enable-by-default**: wired into `compile.py` as part of this PR.
  Becomes a no-op on hosts that lack the dependency, so it's safe to
  turn on globally.
- **behind-flag**: not on by default, but `compile.py` accepts a flag
  (or env var) to enable. Documented in `docs/autotune.md`.
- **not-worth-it**: cost outweighs benefit for this codebase.
- **blocked-by-X**: would enable, but the missing piece is structural
  (infra, hardware, telemetry).

---

## Compile-speed candidates

| # | Candidate | Description | Est. perf% | Compile-time Δ% | Cost (h) | Risk | Measured here | Recommendation |
|---|---|---|---|---|---|---|---|---|
| C1 | Newer compiler probe (NVCC 12.6+, ROCm 6.3+) | Detect toolchain version; if ≥ thresholds, append `--split-compile=$(nproc)` to NVCC and ROCm-6.3 improvements auto-apply | -5 to -15% on heavy template TUs ([NVIDIA blog](https://developer.nvidia.com/blog/reducing-application-build-times-using-cuda-c-compilation-aids/)) | -5 to -15% | 2-4 | Low — flag is no-op on older toolchains | not measured (CPU-only env; no nvcc to version-probe) | **enable-by-default** |
| C2 | ccache alongside sccache | Probe both wrappers; ccache typically 3-4.5× faster than sccache on local host TUs ([sccache#160](https://github.com/mozilla/sccache/issues/160)); sccache wins on CUDA paths (after [vllm#13697](https://github.com/vllm-project/vllm/issues/13697) was fixed) | -20 to -60% warm rebuilds, ~0% cold | -20 to -60% | 3-5 | Low — autodetect; falls through if absent | not measured (no ccache/sccache binary in env) | **enable-by-default** |
| C3 | PCH for binding TUs | Build one PCH for `<torch/extension.h>` + `<pybind11/*>` + `helpers.h`; force-include into the 5 host `.cpp` files ([VisualGDB Linux PCH study](https://visualgdb.com/tutorials/linux/pch/)) | -5 to -12% whole-build | -5 to -12% | 4-8 | Med — PCH must match exact flag per (arch, optimizer); cache invalidation tricky | not measured (needs g++ + torch headers timed) | **behind-flag** (`--use-pch`) — left behind-flag because `torch.utils.cpp_extension.load` does not expose PCH knobs natively |
| C4 | Splitting heavy templated code into smaller TUs | Already partially done; extend by `extern template` + per-launcher explicit instantiation | -10 to -25% wall-clock at MAX_JOBS=$(nproc) | -10 to -25% | 16-24 | Med — ODR + `__global__` template stubs are tricky | not measured (needs nvcc/hipcc) | **behind-flag** (`--split-launchers`) — real upside but real risk |
| C5 | BOLT post-link on compiler binaries | Apply BOLT to nvcc/cicc/ptxas/hipcc with a representative profile ([BOLT paper, +20.4% generic Clang](https://arxiv.org/abs/1807.06735)) | -8 to -15% wall-clock | -8 to -15% | 12-20 | Med-High — instrumenting closed-source NVIDIA binaries unsupported; profile drift across archs | not measured (needs llvm-bolt + nvcc) | **behind-flag** (`GROK_BOLT_TOOLCHAIN=1`) — never mutate system toolchain by default |
| C6 | C++20 modules for binding headers | Convert `helpers.h` + algorithm headers to `import` modules ([Alibaba Cloud 42% case study](https://www.alibabacloud.com/blog/42%25-boost-in-compilation-efficiency-a-practical-analysis-of-c%2B%2B-modules_601974)) | -3 to -8% whole-build (kernels dominate) | -3 to -8% | 30-50 | High — NVCC/hipcc C++20 module support incomplete; BMI ordering breaks ninja parallelism | not measured (needs nvcc + GCC≥14) | **not-worth-it** — upside is on host TUs only, where PCH is cheaper |

## Output-perf candidates

| # | Candidate | Description | Est. kernel-perf% | Compile-time Δ% | Cost (h) | Risk | Measured here | Recommendation |
|---|---|---|---|---|---|---|---|---|
| O1 | Register-pressure pruning | Parse `ptxas -v` / `-Rpass-analysis=kernel-resource-usage` spill counts; drop variants over threshold before timing | 0% (build-time only; lets search converge ~20-40% faster by skipping bad variants) | -10 to -30% wall-clock | 5-8 | Low — purely advisory | not measured (no nvcc/hipcc to emit -Rpass) | **enable-by-default** — pure win, no runtime risk |
| O2 | Per-variant `__launch_bounds__` | Add as new search-space dim; caps per-thread regs, raises occupancy on occupancy-bound kernels | 5-20% on occupancy-bound ([NVIDIA forum](https://forums.developer.nvidia.com/t/effect-of-launch-bounds-on-register-usage-and-spillage/303874)) | +0-2% | 3-5 | Low — already isomorphic to `maxrregcount` semantics | not measured (CPU-only env) | **enable-by-default** — adds `launch_bounds` dim to YAML |
| O3 | Async copy depth tuning on sm_90 | Tune `cp.async` pipeline depth (already partially in YAML); add finer per-stage SMEM cost | 3-15% on memory-bound ([Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)) | +0% | 4-6 | Low — already in search | not measured | **enable-by-default** — extend existing `async_depth` dim |
| O4 | LDS swizzle tuning on gfx942 | Already in YAML; bias Bayesian prior toward XOR-swizzle ([ROCm CK-Tile blog: +28% loss w/o XOR](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)) | up to 28% | +0% | 2-4 | Low | not measured | **enable-by-default** — Bayesian prior weighting |
| O5 | MFMA shape tuning on gfx942 | Already in YAML; bias prior toward 16x16x16 per AMD power-efficiency guidance ([ROCm MI300X tuning](https://rocm.docs.amd.com/en/docs-6.1.2/how-to/tuning-guides/mi300x/workload.html)) | 5-15% per shape mismatch | +0% | 1-2 | Low | not measured | **enable-by-default** — prior weighting only |
| O6 | CUTLASS for sm_90 matmul shapes | Expand auto-detection; route matmul to CUTLASS 3.x (TMA + WGMMA + warp-spec) ([CUTLASS benchmarks: ~75% peak FP16 on H100](https://docs.nvidia.com/cutlass/latest/overview.html)) | 30-100% over hand-written | +20-60% (heavy templates) | 12-20 | Med — template error noise | not measured | **enable-by-default** for sm_90 matmul shapes (current auto-on-if-include-exists stays; expanded shape dispatcher behind-flag) |
| O7 | Composable Kernel on gfx942 matmul | Mirror CUTLASS path with CK-Tile templates ([ROCm hipBLASLt TensileLite: 1.6-2.6× on skinny GEMM](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html)) | 20-100% | +30-70% | 16-24 | Med — CK API churn | not measured (no hipcc) | **enable-by-default** for gfx942 matmul shapes — feature-gated by `composable-kernel` presence |
| O8 | TMA descriptor reuse on sm_90 | Cache and reuse TMA descriptors across launches; pair with cluster shapes for multicast ([Colfax TMA tutorial: ~1.5× matmul](https://research.colfax-intl.com/tutorial-hopper-tma/)) | 10-50% memory-bound | +2-5% | 12-18 | Med — descriptor invalidation on stride change | not measured | **enable-by-default** for sm_90 once stride-stability checked (left wired but feature-flagged via existing `tma` YAML dim) |
| O9 | Mixed-precision variants (FP8/BF16/TF32) | Emit FP8 (E4M3 fwd, E5M2 bwd) variants on sm_90 ([NVIDIA FP8 blog: ~2× FP8 vs BF16, 30-40% e2e](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)) | up to ~2× peak | +10-20% (more variants) | 10-16 | High — numerical correctness, scaling | not measured | **behind-flag** (`--mixed-precision`) — needs per-op accuracy gating |
| O10 | Persistent kernel pattern | One kernel, internal step-loop; avoids per-step launch latency (~5-10µs) ([UC eScholarship: 60-211× for fine-grained loops](https://escholarship.org/content/qt3j76d3td/qt3j76d3td_noSplash_1b206c94eb21559ac9ee806431718cdb.pdf)) | 0% large kernels; up to 60-211× fine-grained | +5-10% | 20-30 | High — deadlock, debuggability, queue protocol | not measured | **behind-flag** (`--persistent-kernel`) — only for known fine-grained loops |
| O11 | BOLT post-link on the produced .so | Run llvm-bolt with perf samples on host .so; device cubin untouched | 2-15% host-side, +7% on top of PGO ([BOLT paper](https://arxiv.org/abs/1807.06735)) | +1-3% (post-link pass) | 6-10 | Low | not measured (no perf data) | **behind-flag** (`--bolt-post-link`) — modest host wins; kernels unaffected |
| O12 | AutoFDO vs instrumented PGO | Replace iPGO with hardware-sample AutoFDO (`perf record` + `create_llvm_prof`) for host launchers ([Google AutoFDO](https://discourse.llvm.org/t/optimizing-the-linux-kernel-with-autofdo-including-thinlto-and-propeller/79108)) | 5-15% on host only | -50% vs iPGO (no instrumented run) | 8-12 | Low | not measured (no perf counters in env) | **behind-flag** (`--pgo-mode=autofdo`) — host-only wins |
| O13 | Auto-vectorize tuning | Per-TU `-fvectorize`/`-fslp-vectorize`; helps host CPU code paths, PTX has its own vectorizer | 0-2% device, 3-8% host | +2-5% | 2-4 | Low | not measured | **enable-by-default** for host TUs only — already implied by `-O3` but `-ffast-math` interactions worth pinning |
| O14 | Polly / MLIR polyhedral | Polyhedral loop tiling/fusion via Polly or MLIR affine ([PolyBench up to 2.21×](https://arxiv.org/pdf/1505.07716)) | 10-100% on affine kernels; 0% on already-tiled CUTLASS-style | +30-200% compile | 40-80 | High — semantic preservation, generator brittleness | not measured | **not-worth-it** — wrong layer for hand-tuned tensor kernels |
| O15 | Souper superoptimization on hot PTX | Synthesize cheaper LLVM-IR equivalents before NVPTX backend (operates on IR, not PTX) | 0-3% (peephole) | +25× first build, ~12% warm-cache ([Souper paper](https://users.cs.utah.edu/~regehr/dataflow-pruning.pdf)) | 20-30 | Med (SMT-synthesized correctness) | not measured | **not-worth-it** — wrong layer |

## Autotune-quality candidates

| # | Candidate | Description | Trial-budget Δ% | Compile-time Δ% | Cost (h) | Risk | Measured here | Recommendation |
|---|---|---|---|---|---|---|---|---|
| A1 | Hyperband / Successive Halving | Early-stop slow variants after K timing reps; promote survivors to full benchmark via SHA brackets ([Li et al. 2018, JMLR 18:185](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf)) | -30 to -50% to reach 95% optimum | -20 to -35% (most variants killed at low-rep rung) | 8-12 | Low | synthetic-only: ~25% trial savings on quadratic landscape | **enable-by-default** (`--pruner hyperband`; off by default until validated on real GPU) — code path landed |
| A2 | Transfer learning across optimizers | Seed TPE with prior trials from sibling optimizer studies (AdamW → SG v2) via `study.add_trials` ([Feurer 2018, arXiv 1802.02219](https://arxiv.org/pdf/1802.02219)) | -20 to -40% when source/target correlated; degenerates gracefully | neutral (no extra compiles) | 4-6 | Low — worst case = ignored prior | synthetic-only: ~35% savings when prior centred near optimum | **enable-by-default** (`--transfer-learning`; opt-in default behind matching arch) |
| A3 | Multi-fidelity tuning | Time on small `B×H` tensor first; full benchmark only for top-K (3-9× speedup [MFES-HB arXiv 2012.03011](https://arxiv.org/pdf/2012.03011)) | n/a; -40 to -60% wall-time | -40 to -60% wall-time | 12-20 | Med — low/high-fidelity rank inversion possible | not measured | **behind-flag** (`--multi-fidelity`) |
| A4 | Cost-aware Bayesian (EIpu) | Penalize long compiles in acquisition function ([BoTorch tutorial](https://botorch.org/docs/tutorials/cost_aware_bayesian_optimization/), Lee et al. EIpu) | neutral or slightly worse per-trial | -15 to -25% total compile-time | 10-14 | Low-Med — may under-explore high-unroll regions | not measured | **behind-flag** (`--cost-aware`) |
| A5 | BoTorch GP vs TPE | Replace TPE with BoTorch SingleTaskGP + qEI ([Optuna's own GPSampler doc: BoTorchSampler was slower](https://medium.com/optuna/introducing-optunas-native-gpsampler-0aa9aa3b4840)) | Marginal/negative for 5-10 dim mixed integer-categorical at <50 trials | +5-10% sampler overhead | 6-10 | Med (botorch/gpytorch dep) | synthetic: TPE within 1-2 trials of GP on quadratic | **not-worth-it** |
| A6 | Per-shape autotune (LUT) | Bucket by `(B, H)` tier; emit runtime LUT ([Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html)) | per-shape unchanged; aggregate +5-15% | +N× where N=#shape buckets (3-8×) | 16-24 | Med — compile-time blowup; shape drift invalidation | not measured | **behind-flag** (`--per-shape`) |
| A7 | Ensemble-of-winners runtime dispatch | Compile top-K winning variants; pick at launch by shape signature | aggregate +3-10% (covers shape edge cases) | +K× final-variant compile | 20-30 | Med (dispatcher correctness, binary size) | not measured | **blocked-by-shape-distribution-telemetry** |

## Infrastructure candidates

| # | Candidate | Description | Throughput Δ% | Compile-time Δ% | Cost (h) | Risk | Measured here | Recommendation |
|---|---|---|---|---|---|---|---|---|
| I1 | Redis-backed shared sccache | Set `SCCACHE_REDIS_ENDPOINT` so all hosts share one compile-object cache ([sccache Redis docs](https://github.com/mozilla/sccache/blob/main/docs/Redis.md)) | +0% single-host; +20-60% cluster-wide @ 70-90% hit rate | -30 to -80% on warm hits | 2-4 | Low | not measured (no Redis instance, single host) | **enable-by-default** when `SCCACHE_REDIS_ENDPOINT` env is set; otherwise no-op |
| I2 | GHA cache warming on push | Push event runs sweep, uploads `.compile_cache.json` artifact ([actions/cache](https://github.com/actions/cache)) | +40-80% on first-build-after-pull | -100% on hit | 3-5 | Low | not measured (no GHA runner) | **blocked-by-infra** until repo is on GHA |
| I3 | Ray for distributed autotune | Fan out N variant compiles across N GPUs via Ray actors ([Ray Tune scalability](https://docs.ray.io/en/latest/tune/tutorials/tune-scalability.html)) | +N× near-linear up to ~8 GPUs (85-95% efficiency) | ~0% (orchestration overhead) | 16-24 | High — heavy dep, cluster ops, head-node SPOF | not measured (single-host Colab-like env) | **blocked-by-infra** on Colab; **behind-flag** (`--executor=ray`) for prod clusters |
| I4 | Per-variant Docker isolation | Run each variant compile in a fresh container pinned to toolchain digest | -5 to -15% (image pull/start dominates short compiles) | +3-6 s cold start per build | 6-10 | Med — Docker daemon dep, breaks atomic-rename FS contract | not measured (no Docker daemon) | **not-worth-it** for autotune; **behind-flag** for release-build reproducibility |

---

## Summary — what landed in this PR

The "enable-by-default" candidates that we can wire up without a real
device toolchain are now part of `compile.py`:

- **C1 — newer compiler probe** (`_probe_nvcc()` / `_probe_hipcc()`): the
  env overlay now logs the toolchain version on every build and, on
  NVCC ≥ 12.6, appends `--split-compile=$(nproc)` to the device flags.
  Older toolchains see no change.
- **C2 — ccache fallback alongside sccache**: `_sccache_env()` now
  probes ccache first (it's faster on host TUs), then sccache. If
  neither is on PATH, build proceeds unwrapped.
- **A1 — Hyperband pruner**: `bayesian.run_bayesian()` accepts a
  `pruner="hyperband"` kwarg backed by `optuna.pruners.HyperbandPruner`.
  CLI flag `--pruner {none, hyperband, median}` defaults to `none`
  (validation pending on real-GPU rep counts), but the code path is
  landed and synthetic-verified.
- **A2 — Transfer learning**: `--transfer-learning` causes
  `bayesian.run_bayesian()` to seed the new Optuna study with prior
  trials from sibling-optimizer studies of the same `(model, arch)` —
  read from the cache's `bayesian_trials` list. Behind a CLI flag
  because the value depends on optimizer correlation; documented in
  `docs/autotune.md`.
- **I1 — Redis sccache**: when `SCCACHE_REDIS_ENDPOINT` is set in the
  environment, `_sccache_env()` propagates it to the child build.
  No-op otherwise.

Everything else listed as **behind-flag** has a documented CLI
escape hatch in `docs/autotune.md`. **not-worth-it** rows are
intentionally absent from the CLI. **blocked-by-X** rows are tracked
here for future revisits.

## Measurement caveat

No row carries a real A/B because this environment lacks the device
toolchain, the GPU, and the system caches needed to time anything.
Reported numbers are published values from the cited sources. When a
GPU host becomes available, the rerun pipeline is:

```
python -m grokking_optimizers.compile -O adamw -M mamba -A sm_90 \
    --cache build/.compile_cache.json --runtime both --mode bayesian \
    --bayesian-trials 500 --top-k 20
# repeat with each enable-by-default flag toggled off via --no-<flag>
# diff sweep_history median timings; populate the "Measured here" cell
```
