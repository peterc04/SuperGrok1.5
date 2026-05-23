# INTERFACES.md — module contracts for the grokking_optimizers.compile rewrite

This file is the coordination contract between the modules added by the
mandatory §1–§11 changes. Every sub-agent and follow-up edit must respect
these interfaces. Any change to a contract here requires the consumers to
move in lockstep.

## High-level layering

```
                ┌───────────────────────────────────────────────┐
                │       grokking_optimizers.compile (orchestrator)
                │   build_aot()  build_jit()  build()  main()
                └───────────────┬───────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────────┐
        │                       │                            │
┌───────▼──────────┐    ┌───────▼─────────┐         ┌────────▼─────────┐
│ search_space.py  │    │  timing_worker  │         │   pgo.py         │
│   load_yaml()    │    │   .py           │         │  instrument_     │
│   prefilter()    │    │   spawn_worker  │         │  flags()         │
│   resolve_dim    │    │  TimingWorker   │         │  use_flags()     │
│   _macros()      │    │  .time(cfg)     │         │  collect_workload│
└──────────────────┘    └─────────────────┘         └──────────────────┘

┌───────────────────┐    ┌──────────────────┐
│ bayesian.py       │    │ bench_graph.py   │
│  run_optuna_      │    │  cuda_graph_     │
│  tpe()            │    │  median_ms()     │
│  topk_refine()    │    │  hip_graph_      │
│                   │    │  median_ms()     │
└───────────────────┘    └──────────────────┘
                │
        ┌───────▼────────┐
        │ profile.py     │
        │  (existing —   │
        │  do not break) │
        └────────────────┘
```

## 1. Cache schema v3 (compile.py owns this)

`CACHE_VERSION = 3`. Forward-compatible with v2 (migration in
`CompileCache._load`). Per-entry shape:

```python
{
    # — Identity (v2) —
    "source_hash":         str | None,
    "host_cflags_hash":    str | None,
    "device_cflags_hash":  str | None,

    # — Artefacts (v2) —
    "primary_artifact":    {path, size, mtime, sha256} | None,
    "variant_artifacts":   {config_key: {path, size, mtime}},

    # — Phases (v2 + v3 extensions) —
    "aot_completed_at":    str | None,
    "jit_completed_at":    str | None,
    "aot_host":            host_dict | None,
    "jit_host":            host_dict | None,

    # — v3 additions —
    "mode":                "exhaustive" | "bayesian" | None,
    "pgo_enabled":         bool,
    "pgo_profile_dir":     str | None,        # path to .gcda dir
    "pgo_workload_hash":   str | None,        # SHA-256 of workload + step count
    "pgo_completed_at":    str | None,
    "pgo_host":            host_dict | None,
    "search_space_hash":   str | None,        # SHA-256 of resolved YAML space
    "bayesian_trials":     [trial_record, …], # Optuna trial list, stage-tagged

    # — Tuning result (v2; payload expanded by v3) —
    "tuned_config":        dim_dict | None,   # macros resolved (e.g. block, vec, …)
    "sweep_history":       [sweep_record, …], # union over all stages
}
```

`trial_record`:
```python
{
    "trial_num":  int,
    "stage":      "tpe" | "refine" | "exhaustive",
    "config":     {dim_name: value, …},
    "timing_ms":  float | None,  # None = build/timing failed
    "min_ms":     float | None,
    "max_ms":     float | None,
    "n":          int | None,
    "host":       host_dict,
    "recorded_at": str,           # ISO timestamp
}
```

`sweep_record` — same shape as `trial_record` (the two lists are kept
separate so a v3 cache can be inspected by stage; `sweep_history`
remains the historical/global log).

`is_aot_fresh(opt, model, arch, *, source_hash, host_flags_hash,
device_flags_hash, pgo_enabled)` — adds `pgo_enabled` to the freshness
hash inputs so a PGO build is not mistaken for a non-PGO build.

### v2 → v3 migration

`_load` detects `version == 2`, adds the new keys with defaults
(`mode=None`, `pgo_enabled=False`, `pgo_profile_dir=None`,
`pgo_workload_hash=None`, `pgo_completed_at=None`, `pgo_host=None`,
`search_space_hash=None`, `bayesian_trials=[]`) and rewrites
`version=3`. No data is dropped; the original is backed up as
`<cache>.v2.bak`.

## 2. configs/search_space.yaml

Single top-level dict keyed by arch. Each arch maps to:
```yaml
sm_90:
  dims:
    - name: block
      type: int
      values: [64, 128, 256, 512, 1024]
      macro: SG_TUNED_BLOCK_SIZE
      applies_to: [host, device]
    - name: vec
      …
  prefilter:
    register_pressure_max: 255         # variant rejected if regs > this
    smem_budget_bytes: 102400          # 100 KB
    rules:
      - name: warps_per_block
        expr: "(block // 32) <= 32"    # max 32 warps per block on sm_90
      - name: vec_block_alignment
        expr: "block % (vec * 4) == 0"
gfx942:
  dims:
    …
  prefilter:
    waves_per_eu_max: 10
    rules:
      - name: wave_alignment
        expr: "block % 64 == 0"
```

Dim entry fields:
- `name` (str, required)
- `type` (`int`, `bool`, `enum`, `tuple`)
- `values` (list of literals; tuples become `[a, b, c]`)
- `macro` (str — the `-D` name to set; `None` to skip)
- `applies_to` (list ⊆ `{"host", "device"}`)

A dim with `macro: None` is filtered before reaching the compiler (e.g.
internal metadata). A dim with `applies_to: [device]` only goes into
NVCC/HIPCC cflags.

## 3. search_space.py — public API

```python
def load_yaml(path: Path) -> dict
def cartesian(space: dict, arch: str) -> List[dict]
def prefilter(configs: List[dict], rules: dict) -> Tuple[List[dict], int]
    # returns (survivors, eliminated_count)
def resolve_macros(config: dict, dim_specs: List[dict],
                   target: Literal["host", "device"]) -> List[str]
    # returns ["-DSG_TUNED_BLOCK_SIZE=128", …] filtered by applies_to
def hash_space(space: dict, arch: str) -> str
```

## 4. timing_worker.py — public API

```python
class TimingWorker:
    """Persistent subprocess that holds a warm CUDA/HIP context.
    JSON-RPC line protocol over stdin/stdout."""

    def __init__(self, opt_class: str, *, size: int = 4096,
                 warmup: int = 5, iters: int = 21,
                 use_cuda_graph: bool = True,
                 use_hip_graph: bool = True,
                 timeout_per_variant: int = 180): ...

    def start(self) -> None:                              # spawn process
    def time(self, variant_so: Path) -> dict | None:      # JSON {timing_ms, min_ms, max_ms, n}
    def alive(self) -> bool: ...                          # heartbeat
    def restart(self) -> None: ...                        # on crash
    def stop(self) -> None:                               # graceful close
```

On worker crash, the public `.time()` returns `None`, logs the error
text, and `compile.py` falls back to a per-subprocess `_time_variant`
for that one call (existing behaviour); the worker is restarted before
the next variant.

Wire protocol (line-delimited JSON over stdin):
```
{"op": "time", "so_path": "...", "opt_class": "Lion"}
```
→ stdout:
```
{"timing_ms": 0.123, "min_ms": 0.118, "max_ms": 0.135, "n": 21}
{"error": "<msg>", "tb": "<traceback>"}
```

## 5. bench_graph.py — public API

```python
def cuda_graph_median_ms(opt_class: str, *, size: int = 4096,
                         warmup: int = 5, iters: int = 21) -> dict
def hip_graph_median_ms(opt_class: str, *, size: int = 4096,
                        warmup: int = 5, iters: int = 21) -> dict
```

Each captures a single `opt.step()` into a graph then replays `iters`
times under a single `cudaEventRecord` pair. Returns the same dict
shape as `_time_variant` so the worker can swap between graph and
event timing without callers caring.

The implementation is in this module so `timing_worker.py` (which runs
inside the subprocess) can import it without pulling in compile.py.

## 6. bayesian.py — public API

```python
def run_bayesian(
    arch: str,
    space: dict,                  # parsed YAML space (post-prefilter dims)
    *,
    n_trials: int = 500,
    seed: int = 0,
    storage: Path | None = None,  # Optuna SQLite for resume
    study_name: str = "sg_tune",
    timer: Callable[[dict], dict | None],   # builds+times one config
    progress: Callable[[int, int, dict], None] | None = None,
) -> List[trial_record]
```

Sampler: `optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(10, n_trials // 10))`.
Random fraction throughout: ~3% (`random_sample=True` with probability
0.03 in the TPE config).

```python
def topk_refine(
    bayes_trials: List[trial_record],
    space: dict,
    *,
    top_k: int = 20,
    timer: Callable,
    progress: Callable | None = None,
) -> List[trial_record]
```

For each of the top-K trials, generates ±2-step neighbours per dim
(dedup against seen configs, dedup against each other) and times each.
Returns the refine-stage trial list. The caller's overall winner is
`min(all_trials, key=lambda t: t["timing_ms"] if t["timing_ms"] is not None else ∞)`.

## 7. pgo.py — public API

```python
def instrument_flags(arch: str, profile_dir: Path,
                     host_cflags: List[str],
                     device_cflags: List[str],
                     ldflags: List[str]) -> Tuple[List[str], List[str], List[str]]
    # returns (host_cflags, device_cflags, ldflags) with -fprofile-generate
    #   appended; profile_dir is the output directory for .gcda

def use_flags(arch: str, profile_dir: Path,
              host_cflags: List[str],
              device_cflags: List[str],
              ldflags: List[str]) -> Tuple[List[str], List[str], List[str]]
    # returns (host, device, ldflags) with -fprofile-use appended

def collect_workload(workload_script: Path, *,
                     so_path: Path,
                     opt_class: str,
                     model: str,
                     arch: str,
                     steps: int,
                     env: dict | None = None,
                     timeout: int = 600,
                     report=None) -> bool
    # runs <python> <workload_script> --so <so> --opt <opt_class> --steps N
    # validates that .gcda files appear under profile_dir; returns bool

def hash_workload(workload_script: Path, steps: int) -> str
    # sha256(file_contents) + sha256(str(steps))
```

A default workload script is shipped at `scripts/pgo_workload.py`. The
PGO driver imports the produced .so, runs N training steps on a small
model, and discards outputs — its only purpose is to exercise the
optimizer's hot path.

## 8. compile.py orchestrator extensions

```python
@dataclass
class BuildSpec:
    optimizer: str
    model: str
    arch: str
    out_dir: Path
    autotune: bool = True
    autotune_mode: str = "bayesian"     # "exhaustive" | "bayesian"
    profile: bool = True
    verbose: bool = False
    extra_macros: List[str] = field(default_factory=list)
    runtime: str = "both"               # "aot" | "jit" | "both"
    aot_only: bool = False              # alias for runtime="aot"
    jit_only: bool = False              # alias for runtime="jit"
    # — v3 additions —
    search_space_path: Optional[Path] = None
    aot_artifact_dir: Optional[Path] = None
    pgo: bool = False
    pgo_workload: Optional[Path] = None
    pgo_steps: int = 1000
    bayesian_trials: int = 500
    top_k: int = 20
    seed: int = 0
    debug_symbols: bool = False
```

Public functions:
- `build(...)` — the orchestrator. Same kwargs as `BuildSpec` (minus
  `out_dir`-derived defaults). Returns final .so path.
- `build_aot(spec, cache, report)` — runs AOT phase in this process,
  no GPU init required. Returns the AOT artefact path.
- `build_jit(spec, cache, report)` — runs JIT autotune + final link in
  this process; requires GPU.
- `main(argv)` — CLI; spawns subprocesses for AOT/JIT based on
  `--runtime`.

Runtime split: `main` is the only place that spawns subprocesses. When
`--runtime both`, main spawns an AOT subprocess (`python -m
grokking_optimizers.compile --runtime aot …`), waits, then spawns a
JIT subprocess. Each subprocess re-enters `main()` and ends in
`build_aot` or `build_jit` exclusively.

## 9. Output flags (compile.py owns these)

```python
HOST_CFLAGS_BASE = [
    "-O3", "-std=c++17", "-fPIC",
    "-flto=full", "-march=native", "-mtune=native",
    "-fno-semantic-interposition", "-fvisibility=hidden",
    "-fdata-sections", "-ffunction-sections",
    "-fno-math-errno", "-fno-trapping-math",
    "-fomit-frame-pointer",
    "-ffast-math", "-funroll-loops",
]

NVCC_DEVICE_BASE = [
    "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
    "--expt-relaxed-constexpr",
    "--threads", "8",
    "-Xfatbin", "-compress-all",
    "-Xptxas", "-O3", "-Xptxas", "-v", "-Xptxas", "--warn-on-spills",
    "-Xptxas", "--allow-expensive-optimizations=true",
    "-Xptxas", "--def-load-cache=ca",
    "-Xptxas", "--def-store-cache=wb",
    "--extra-device-vectorization",
    "-Xcompiler", "-fPIC", "-Xcompiler", "-flto=full",
    "--resource-usage",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
    "-dlto",
]

HIPCC_DEVICE_BASE = [
    "-O3", "-std=c++17", "-DWITH_HIP",
    "-ffast-math", "-fPIC",
    "--offload-arch=gfx942",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-amdgpu-internalize-symbols",
    "-fgpu-flush-denormals-to-zero",
    "-Rpass-analysis=kernel-resource-usage",
    "-flto",
]

LDFLAGS_BASE = [
    "-flto=full", "-Wl,--as-needed",
    "-Wl,--gc-sections", "-Wl,-O3",
    "-Wl,--icf=all",
]
```

Debug symbols (off by default; on when `--debug-symbols` or `--profile`):
host adds `-ggdb`; NVCC adds `-lineinfo --generate-line-info`.

No `--maxrregcount` is hardcoded — it is now a search-space dim.

sccache / NVCC threads: see `env_overlay` in `compile.py`. When
`sccache` is on PATH, `CC` and `CXX` are prepended; `CUDA_NVCC_EXECUTABLE`
is set to `"sccache nvcc"`. `SCCACHE_DIR=/dev/shm/sccache` if writable
else `~/.cache/sccache`. `NVCC_THREADS=8` is exported (NVCC honours it
when seen at link time).

Variant builds split base TUs (bindings + models) from the variant TU
(launcher with macro changes). The base is compiled once and the
launcher TU links N times. Implemented inside `_torch_load` via
`build_directory` reuse for base + per-variant `build_directory` for
the launcher.

## 10. CLI surface (compile.py main)

```
python -m grokking_optimizers.compile \
  --optimizer supergrok2 --model mamba --arch sm_90 \
  --mode {exhaustive,bayesian} \
  --bayesian-trials 500 --top-k 20 \
  --pgo --pgo-workload scripts/pgo_workload.py --pgo-steps 1000 \
  --search-space configs/search_space.yaml \
  --cache build/.compile_cache.json \
  --aot-artifact-dir build/compiled/aot_artifacts \
  --runtime {aot,jit,both} \
  --debug-symbols \
  --seed 0 \
  [--aot-only | --jit-only] \
  [--quick]                 # alias: bayesian + low trial budget
  [--no-autotune]
  [--no-profile]
  [-D MACRO[=VALUE]]
  [-v / --verbose]
```

Defaults: `--mode bayesian --bayesian-trials 500 --top-k 20 --pgo`
**off** (opt-in because it doubles compile time), `--runtime both`,
`--search-space configs/search_space.yaml`, `--seed 0`,
`--debug-symbols` off (auto-on if `--profile`).

## 11. Tests

```
tests/test_compile_search_space.py    # YAML loader, prefilter
tests/test_compile_cache_migration.py # v2 → v3 forward migration
tests/test_compile_bayesian.py        # Optuna stub-timer end-to-end
tests/test_compile_pgo_hashing.py     # workload hash determinism
tests/test_compile_dry_run.py         # full pipeline on a no-GPU host
                                      #   (pallas backend; AOT cache hit logic)
```

All tests run on CPU and use stub timers / pallas backend / dry-runs
where a GPU would be needed.

## 12. Out of scope (do not touch)

- `profile.py`: imports are stable; do not add new exports.
- `grokking_optimizers/optimizers/`: optimizer math is frozen.
- `csrc/`: kernel sources are frozen; only `tuned_configs.h` is
  regenerated.
- `setup.py`: untouched.
- `grokking_race_v2.py`: untouched.

## 13. Branch + commit hygiene

- Branch: `claude/custom-optimizer-analysis-HFYhg` (must end in the
  active session id; pushes to anything else 403 by policy).
- Commit between logical phases: (1) coordination scaffolding,
  (2) module additions, (3) compile.py rewrite, (4) tests + docs,
  (5) §12 matrix + README.
- Use `git mv` if any file moves.
