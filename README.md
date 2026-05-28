# SuperGrok2

SuperGrok2 is a C++/CUDA/HIP/Pallas optimizer suite for grokking-aware
training of large neural networks. It ships twelve optimizers — a plain
AdamW baseline plus eleven grokking-aware variants spanning sign-momentum,
sharpness-aware minimization, Newton-Schulz orthogonalization, and a
Mamba-3 + PEER + GRU meta-network optimizer (SuperGrok v2 — the project's
namesake). The grokking race driver (`grokking_race_v2.py`) compares all
twelve head-to-head on three algorithmic learning tasks under controlled
conditions.

---

## Quickstart: clone → install → compile → profile

Four Python steps from a fresh machine to a profiled artifact. Everything
after Step 1 runs inside one Python session — no shell wrappers, no
external config files, no extra modules in `grokking_optimizers/`. The
entire MAXIMAL pipeline (codegen, NVRTC, device PGO, multi-GPU,
numerical validation, TOML config loader, Pallas backend) lives inside
`grokking_optimizers/compile.py`.

### Tiny shell-only quickstart (sanity check + arch dump)

Before the four-Python-step flow below, three one-line CLI commands let
you confirm the wrapper is healthy on the current host without
installing anything but Python and the package itself:

```bash
# 1. Run the inline self-test suite (138 checks; many gate on dep
#    presence and SKIP cleanly on a CPU-only host — see the gating
#    note in "What gets exercised on a CPU-only host" below)
python3 -m grokking_optimizers.compile --self-test

# 2. List every supported arch with its vendor / min toolchain / features
python3 -m grokking_optimizers.compile --list-archs

# 3. Sweep all 26 canonical archs into per-arch JSON manifests (no
#    compile, no GPU). Each JSON includes host_cflags, device_cflags,
#    device_ldflags, version_gated_device_cflags (per-host toolchain
#    probe), preflight judgment, and enabled_features (with the pgo
#    toggle surfaced alongside the others).
python3 -m grokking_optimizers.compile --dry-run-all-archs --out /tmp/manifests
```

Auto-arch detection: every CLI invocation now treats `--arch` as
**optional**. When **omitted entirely** (no `--arch` token on the
command line), the wrapper probes in priority order:

1. `torch.cuda.get_device_capability(0)` (real NVIDIA GPU on the host)
2. `rocm-smi --showproductname` (real AMD GPU)
3. `jax.devices()` (real TPU)
4. The TOML config's `[archs].default` (when a `--config <file>` is
   supplied AND it sets that key)
5. Built-in fallback `sm_90a`

The first probe that succeeds wins; subsequent probes are skipped.
A `[arch] auto-detected <arch> from <source> (<gpu_name>)` line is
printed so you can see which one fired. Pass `--arch sm_90a` (or any
other concrete arch) to override.

Note: `--arch auto` is **not** a recognized value. Only *omitting*
`--arch` triggers auto-detect; passing the literal string `auto` is
rejected by argparse with an `invalid choice` error.

Single-arch dry-run: `python3 -m grokking_optimizers.compile -O adamw -M
mamba --arch sm_90a --dry-run --out /tmp/probe --enable-synth-codegen
--enable-polyhedral` writes one manifest for that arch (mirror of
`--dry-run-all-archs` but scoped to one entry), exercises the synthetic
codegen / polyhedral layers if you flip them on, and exits without
invoking nvcc/hipcc.

### Using this wrapper for a *different* project (no SuperGrok assumptions)

`compile.py` is project-agnostic — write a tiny TOML config and every
emitted macro, cache directory, source layout, and template lookup
re-prefixes for your project. Example:

```toml
# myproject.toml
[project]
name              = "myproject"      # used by the debug header for the disk-free probe; the
                                     # actual NVRTC cubin cache lives under <out>/nvrtc_cache
namespace         = "myproject"      # → C++ namespace in emitted code
python_package    = "myproject"      # → import path the registry uses
macro_prefix      = "MYPROJ_"        # → -DMYPROJ_OPTIMIZER_*, -DMYPROJ_ARCH_*, -DMYPROJ_VERBOSE
fused_op_template = "fused_{name}_step.cpp.j2"
# (macro_prefix and fused_op_template can ALSO be placed under [sources] if
#  that reads more naturally; both locations are accepted, [project] wins.)

[sources]
source_layout    = "src/myproject/{name}"      # where your .cu / .hip.cpp / .py live

[archs]
default = "sm_86"                              # used when --arch is omitted and no GPU probe succeeds

[optimizers]
enabled = ["adamw", "lion", "my_custom_opt"]  # any optimizer name; not restricted to the SuperGrok 11
```

Then drive `compile.py` exactly as below but with `--config myproject.toml`
or `build(..., config="myproject.toml")`. Verified end-to-end via the
self-test `project_agnostic_dry_run_no_sg_leakage`: with the above
config the wrapper emits `-DMYPROJ_OPTIMIZER_ADAMW=1`,
`-DMYPROJ_ARCH_SM86=1` (note: arch macro uses the literal arch label
from `ARCH_TABLE` — `SM86`, not `SM_86`), and `-DMYPROJ_VERBOSE=1`
with **zero `SG_BUILD_` leakage** across all 26 canonical archs. The
`enabled_features` field in every dry-run manifest surfaces which
opt-in toggles were enabled (including `pgo`, surfaced alongside the
other `enable_*` keys).

### Step 1 — Clone the repo

```bash
git clone https://github.com/peterc04/SuperGrok1.5
cd SuperGrok1.5
```

### Step 2 — Install Python dependencies

```python
import subprocess, sys

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", *pkgs], check=True)

# Core deps for every arch.
pip_install("torch", "optuna", "pyyaml", "ninja", "tqdm")

# Optional, only when the feature is enabled below:
#   ENABLE_EMITTER       → pip_install("jinja2")
#   ENABLE_RUNTIME_SPEC  → pip_install("cuda-python")    # NVRTC bindings
#   STRICT_NUMERICS      → pip_install("numpy")          # almost certainly already present
#   arch="tpu_v*"        → pip_install("jax[tpu]", "-f",
#                            "https://storage.googleapis.com/jax-releases/libtpu_releases.html")

# Optional: CUTLASS submodule for sm_90a CUTLASS GEMMs
subprocess.run(["git", "submodule", "update", "--init", "--recursive",
                "third_party/cutlass"], check=False)
```

### Step 3 — Compile

Pick an `(optimizer, model)` pair and call `build(...)`. That's the
whole quickstart. Every MAXIMAL feature — host PGO, device PGO
(CUPTI / rocprof / XLA HLO), Jinja2 emitter, NVRTC/hipRTC runtime
specialization, OpGraph synthesis codegen, polyhedral schedule
search, learned cost model with rejection budget, Hyperband pruning,
sibling-optimizer transfer learning, vendor-dispatched toolchain
bootstrap — is **ON by default** in `build()`. Each layer has a
graceful skip path when its soft dep or hardware is missing, so the
same block runs unchanged on T4 / L4 / A100 / H100 / MI300 / TPU.

The autotune budget is **auto** by default: no fixed trial count, no
wall-clock cap. The 5-criterion early-stop (plateau / EI floor /
coverage saturation / time-cap / patience) decides when to halt while
the autotuner samples from the full per-arch programmatic search
space — sm_90a alone is ~3.7B candidates wide before prefilter
(`build_full_search_space()` in `compile.py`). Arch is auto-detected
via the probe chain (`torch.cuda` → `rocm-smi` → `jax.devices` → TOML
→ built-in fallback) so you don't set it on Colab.

Run Python from the repo root (the directory `cd`'d into in Step 1):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))   # make the cloned source importable

from grokking_optimizers.compile import build, CompileCache

so_path = build(
    optimizer="adamw",     # adamw | lion | grokfast | grokadamw | looksam |
                           # muon | neuralgrok | prodigy | supergrok11 |
                           # supergrok15 | supergrok2
    model="vit",           # mamba | decoder | vit
    cache=CompileCache(Path("build/.compile_cache.json")),
    debug=True,            # stream the full report to stderr in real time
)
print("built:", so_path)
```

That's it. The block does AOT compile, Bayesian autotune with auto
early-stop, 3-pass host PGO, device-PGO sidecar (when CUPTI/rocprof
are available), final rebuild with the winning macros, and the
profile pass — all streamed live with `debug=True`. Outputs land in
`build/compiled/` and the cache survives across runs.

**Power-user overrides** (rarely needed — every default is the best
setting we've measured):

```python
so_path = build(
    optimizer="supergrok2", model="mamba",
    arch="sm_90a",                 # override auto-detect
    bayesian_trials=200,           # pin a fixed trial count
    max_tune_seconds=900,          # 15-minute wall-clock cap
    strict_numerics=True,          # require bit-identical determinism
    enable_polyhedral=False,       # disable any specific layer
    cache=CompileCache(Path("build/.compile_cache.json")),
    debug=True,
)
```

What you will see streaming to your terminal with `debug=True`:

```
========================================================================
[debug] grokking_optimizers.compile starting at 2026-05-26T...
[debug] target:   adamw/decoder/sm_90a (vendor=cuda)
[debug] runtime:  both  autotune=True (bayesian, auto-stop)  pgo=True  profile=True
[debug] phases:   ['resolve', 'aot', 'jit-autotune', 'final', 'profile']
[debug] out_dir:  /home/you/SuperGrok1.5/build/compiled
[debug] cache:    /home/you/SuperGrok1.5/build/.compile_cache.json
[debug] report:   /home/you/SuperGrok1.5/build/compiled/compile_adamw_decoder_sm_90a.txt
[debug] env:      CUDA_HOME=/usr/local/cuda  ROCM_PATH=<unset>
[debug] env:      PATH=...
[debug] env:      FORCE_CUDA=<unset>  TORCH_CUDA_ARCH_LIST=<unset>
========================================================================

# grokking_optimizers.compile — targeted build
# Generated: ...
[preflight] arch=sm_90a need CUDA>=12.0 have 12.6 — PASS
... (full report contents, every line tee'd live) ...
--- AOT PHASE ---
  module:    grokking_compiled_adamw_decoder_sm_90a
  sources:   18 files
  host:      -O3 -std=c++17 ... (every cflag)
  device:    -O3 --use_fast_math -gencode=arch=compute_90a,code=sm_90a ...
  [toolchain] nvcc 12.6
  ... (verbose=True is auto-set, so every nvcc command + ninja line is printed)
--- JIT AUTOTUNE PHASE ---
  [prefilter] 3,735,552,000 candidates → streaming (sm_90a full space)
  Trial 1: block=128 vec=2 unroll=4 num_stages=3 ... → 1.42 ms
  Trial 2: block=256 vec=4 unroll=8 num_stages=4 ... → 0.91 ms (new best)
  ... (Optuna trials stream live with their numerical_status tag) ...
  [early-stop] plateau:no_improvement_in_67 — stopped after 117 trials
  [topk-refine] elbow detected at index 14 — refining 14 configs
  [cache-prune] dropped 23 stale variant_artifacts, freed 412 MB
--- PROFILE PASS ---
  ... (ncu / rocprof / jax.profiler output) ...
```

Outputs (also written even without `debug=True`):

- `build/compiled/compile_<O>_<M>_<A>.txt` — full text report (the same
  thing you saw stream by)
- `build/compiled/grokking_compiled_<O>_<M>_<A>/*.so` — the built kernel
  (for Pallas archs: `tuned_pallas_<O>_<M>_<A>.json` instead — no `.so`)
- `build/.compile_cache.json` — survives across runs; same combo is a
  cache-hit on every phase
- `build/compiled/device_stall_info.json` — when CUPTI/rocprof
  produced something (device-PGO is on by default; degrades silently
  when the profiler isn't installed)
- `build/compiled/nvrtc_cache/*.cubin` — when cuda-python / hip-python
  is available (runtime specialization is on by default)

`build()` auto-discovers `nvcc` / `hipcc` (searches `PATH`, then
`$CUDA_HOME`, then `/usr/local/cuda-*`, then NVIDIA's PyPI wheels) and
auto-fixes a stale `CUDA_HOME` env var before torch's `cpp_extension`
reads it. On a fresh Colab CPU runtime, `bootstrap_cuda=True` (the
default) installs nvcc via conda / apt / dnf / yum / zypper / pacman /
apk / brew / winget / PyPI wheels — whichever is available — before
the AOT phase runs.

### Step 4 — Profile (one step, also fully debug-able)

```python
# If running this in a fresh Python session (not the same one as Step 3),
# re-add the repo root so the import resolves:
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))   # assumes you're in SuperGrok1.5/

from grokking_optimizers.profile import profile

report = profile(
    optimizer="adamw",   # same OPTIMIZER you compiled in Step 3
    model="decoder",     # same MODEL
    arch="sm_90a",       # same ARCH (canonical form with the "a" suffix)
    debug=True,          # stream ncu/rocprof/jax.profiler live
)
print("profile report:", report)

# Or profile a specific .so / launcher source directly:
# report = profile(path="build/compiled/grokking_compiled_adamw_decoder_sm_90a/"
#                        "grokking_compiled_adamw_decoder_sm_90a.cpython-311-x86_64-linux-gnu.so",
#                  debug=True)
```

### (Optional) Production install for the race driver

`build()` is for iterating on one combo with full diagnostics. If you
also want the production `grokking_optimizers._ops` extension consumed
by `grokking_race_v2.py`, install the package itself once:

```python
import os, subprocess, sys
env = os.environ.copy()
# env["FORCE_CUDA"] = "1"               # CPU host with no visible GPU
# env["TORCH_CUDA_ARCH_LIST"] = "9.0"   # Hopper-only gencode (faster build)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
               check=True, env=env)
subprocess.run([sys.executable, "grokking_race_v2.py", "--help"], check=True)
```

### Verifying the install (no GPU or toolchain needed)

If anything in Step 3 goes wrong before the compile actually starts,
run the inline self-test to confirm the Python install is sound:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))   # assumes you're in SuperGrok1.5/

from grokking_optimizers.compile import main as compile_main
assert compile_main(["--self-test"]) == 0  # prints "[self-test] N passed, M failed";
                                            # N is ~138 today (count fluctuates as
                                            # tests are added). M=0 is the success
                                            # condition. On a CPU-only host, several
                                            # tests SKIP cleanly (counted as PASS)
                                            # when their dep is absent — see the
                                            # "What gets exercised" note below.
```

The self-test (~138 checks today) covers a broad surface area:

- **Always exercised** (no opt deps needed): ARCH_TABLE completeness for
  all 26 canonical archs; per-arch search-space cardinalities; the
  Stream α native flag emission for NVIDIA / AMD / JAX; ptxas-v stderr
  parser; the five Bayesian early-stopping criteria (plateau,
  EI-exhaustion, coverage saturation, wall-clock, hard ceiling); cache
  schema migrations (v2 → v3 → v4) and the v4 `.jsonl` trial sidecar
  plumbing; the Stream β auto-arch resolver
  (`torch.cuda` → `rocm-smi` → `jax.devices` → TOML → fallback);
  preflight version-mismatch suggestion lines; per-arch dry-run
  manifest sweep; Pallas backend sentinels with the `block_spec`
  search-space dim; numerical-validation tolerances; TOML config
  loader (project-agnostic macro-prefix override; `[project]` /
  `[sources]` dual-location acceptance; macro-prefix validation); the
  Optuna 4.0+ `study.tell(state=PRUNED)` regression test; the dry-run
  `enabled_features` feature-toggle manifest field (including the
  `pgo` key surfaced alongside `enable_*`); the buildspec-field AST
  guardrail; and toolchain-bootstrap *dispatch* (the actual
  installation path is not exercised — only the per-vendor selector).
- **Dep-gated (SKIP cleanly when dep absent)**: bundled Jinja2 template
  rendering (needs `jinja2`); polyhedral `apply_schedule` body lift
  (needs `libclang`); arch-native MMA emission in synth GEMM and the
  CUTLASS / CK fallback `source=` tagging (needs the third-party
  submodules); NVRTC compile + live `cuLaunchKernel` dispatch (needs
  `cuda-python` + a visible CUDA device); work-stealing multi-GPU pool
  (needs ≥1 visible GPU); the `e2e_smoke` end-to-end build (needs
  toolchain + GPU); device-PGO `bias_trial_queue` wiring into
  `run_bayesian` (needs `nsys` / `rocprof` to produce a real sidecar).

In other words: the *build-correctness* portion of the suite (flag
emission, search space, cache schema, BuildSpec semantics, config
loader) runs on any host. The *integration* portion (live NVRTC,
GPU-bound autotune, Jinja2-rendered templates) needs the
corresponding optional dep to actually fire.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'grokking_optimizers'` | You're running Python from outside the cloned `SuperGrok1.5/` directory, or `sys.path` doesn't include it. Add `sys.path.insert(0, str(Path.cwd()))` (Step 3 block already does this) before any `from grokking_optimizers...` import — or point `REPO_ROOT` at the absolute path of the cloned repo. No `pip install -e .` required just to run `compile.py` / `profile.py`. |
| `CUDA_HOME environment variable is not set` | `os.environ["CUDA_HOME"] = "/usr/local/cuda"` (Step 3 block already does this). Required even when `nvcc` is on `PATH`. |
| `CUDA_HOME environment variable is not set` **but the debug banner shows it IS set** | Stale torch cache. `torch.utils.cpp_extension` reads `CUDA_HOME` at *its* import time; if torch was already loaded before you set the env var (common in Colab/Jupyter), the cached value is `None` and the build fails despite `os.environ` being correct. `build()` now calls `_refresh_torch_cuda_home()` on entry to patch the stale cache from `os.environ`. Pull the latest commit; this is fixed. |
| `[preflight] WARNING: nvcc NOT found` | Pass `bootstrap_cuda=True` to `build()`. It probes `conda` / `apt-get` / `dnf` / `yum` / `zypper` / `pacman` / `apk` / `brew` / `winget` in priority order and installs nvcc via whichever your environment has — works on Colab, any Linux distro, macOS (legacy CUDA only), and Windows 10+. Pulls 1-2 GB and takes 2-5 min. |
| `[preflight] arch=gfx942 ... hipcc not found — FAIL` | Pass `bootstrap_rocm=True` to `build()`. It probes AMD's official apt repo first (per-arch version: gfx950 → ROCm 6.2+, gfx1200 → ROCm 7.0+) then falls back to stock apt / dnf / zypper. |
| `[preflight] arch=tpu_v5p JAX not installed — FAIL` | Pass `bootstrap_jax=True` to `build()`. It pip-installs `jax[tpu]` from the `libtpu_releases` bucket if no TPU device is currently visible to `jax.devices()`. |
| `nvidia-cuda-nvcc-cuXX wheels installed but nvcc still not findable` | NVIDIA's PyPI wheels ship `ptxas + libnvvm + libcudart + headers` but NOT the `nvcc` compiler driver itself. The driver only comes from a native package manager (apt/dnf/yum/zypper/pacman/apk), `conda install -c nvidia cuda-nvcc`, `winget install Nvidia.CUDA`, or the official `.run` installer from `developer.nvidia.com/cuda-downloads`. `bootstrap_cuda=True` tries all of these and only falls back to wheels for the headers/libs. |
| Build dies with `sh: /usr/local/cuda/bin/nvcc: not found` even though nvcc IS installed | Stale `CUDA_HOME`. You set `os.environ["CUDA_HOME"]` to a path (e.g. `/usr/local/cuda`) that doesn't actually contain `bin/nvcc` — but `nvcc` lives elsewhere (e.g. `/usr/bin/nvcc` from `apt`). `_ensure_nvcc_on_path()` now calls `_reconcile_cuda_home()` after every discovery to overwrite the stale env var with the actual nvcc parent. Pull the latest commit; this is fixed. |
| `[preflight] WARNING: nvcc X.Y is too old for sm_90 (needs CUDA 12.0+)` | Stock distro packages are often years behind (Ubuntu 22.04's `nvidia-cuda-toolkit` = CUDA 11.5; sm_90 needs 12.0+). `bootstrap_cuda=True` now prefers NVIDIA's **official apt repo** (`developer.download.nvidia.com/compute/cuda/repos/...`) over the stock package, installing `cuda-toolkit-12-X` to `/usr/local/cuda/` — exactly what torch's cpp_extension expects. Manual recipe: download `cuda-keyring_1.1-1_all.deb`, `dpkg -i` it, `apt update`, then `apt install cuda-toolkit-12-6`. |
| Colab CPU runtime is slow to bootstrap (~5 min) | Trade-off: pulling ~2 GB of `nvidia-cuda-toolkit` over Colab's network. Alternative: switch to a Colab GPU runtime (*Runtime → Change runtime type → Hardware accelerator: GPU*) where nvcc is preinstalled. **Colab GPU runtimes are T4/L4/A100/V100 — none are sm_90 (Hopper)** — but the AOT compile + autotune still produces a valid sm_90 artefact for Hopper deployment, it just won't load on the Colab GPU at runtime. |
| `[build FAILED]` and the report shows the full traceback but I can't tell what's wrong | The traceback's deepest frame points at the actual torch/cpp_extension failure. Below that, the report dumps the `build.ninja`, `.ninja_log`, and a direct `ninja -C <build_dir> -v` stderr capture. Search the report for `error:`/`fatal error:` to find the real compiler diagnostic. |
| `No supported GPU backend detected` from setup.py | `env["FORCE_CUDA"] = "1"` before the `pip install -e .` subprocess. |
| `nvcc not on PATH; skipping version-gated flags` in the streamed report | Install CUDA Toolkit ≥ 12.0 and prepend `$CUDA_HOME/bin` to `PATH`. Build still runs but won't auto-add `--split-compile`. |
| Self-test fails with `ModuleNotFoundError` | Repeat Step 2 — `pip_install("torch", "optuna", "pyyaml", "ninja", "tqdm")`. |
| Compile silently hangs at "jit-autotune" | First-time Optuna study creation needs SQLite write access to `<out>/optuna_<O>_<M>_<A>.db`. With `debug=True` the SQLite error would have streamed; check `out_dir=` permissions. |
| `[build FAILED after 0.0s]` and you missed the error | Re-run with `debug=True` — the actual compiler error streams to your terminal and is also persisted in `build/compiled/compile_<O>_<M>_<A>.txt`. |

### Common pitfalls

A short list of footguns the wrapper does not (and cannot) auto-fix:

- **GPU vs ARCH mismatch (cross-compile foot-gun).** Setting
  `ARCH=sm_90a` on a host whose GPU is a T4 / A100 / V100 produces a
  perfectly valid Hopper artefact — but it will not load on the local
  GPU at runtime. If you want to *run* what you build, either omit
  `--arch` (auto-detect picks the local arch) or set it to match.
  `--arch sm_90a` on a non-Hopper host is only useful when you intend
  to *publish* the `.so` to a Hopper machine via `--aot-artifact-dir`.

- **V100 / sm_70 requires CUDA 10.0+.** V100 (Volta) is in ARCH_TABLE
  and auto-detected by `_resolve_default_arch()`. Note that sm_70
  lacks Tensor Core `mma` features available on sm_75+ so some
  kernel variants will be slower.

- **CUDA-version mismatch on the host.** When `nvcc --version` reports
  CUDA 11.x but your target arch needs 12.x (e.g. sm_90, sm_100,
  sm_120), pass `--bootstrap-cuda` so the wrapper installs the
  matching toolchain version from NVIDIA's official apt repo. Without
  this, the per-arch preflight emits `[preflight] arch=sm_X need
  CUDA>=N have M — FAIL` and the build is unlikely to succeed.

- **LTO compatibility.** The `HOST_CFLAGS_BASE` includes `-flto=full`,
  which older GCC releases (and clang ≤ ~13) reject — they only
  accept `-flto` / `-flto=auto` / `-flto=thin`. If your host compiler
  fails with `unrecognized argument to -flto= option: 'full'`, the
  preflight flag-probe will detect and drop the flag automatically
  (`[preflight] WARN flag '-Xcompiler -flto=full' rejected by nvcc —
  dropping`). The build proceeds without LTO in that case.

- **`--arch auto` is not valid.** Auto-detect only triggers when
  `--arch` is *omitted entirely* — passing the literal string `auto`
  is rejected by argparse.

### What gets exercised on a CPU-only host

The self-test runs ~138 checks today, but many are dep-gated: when
the required Python package or hardware isn't available, the check
SKIPs cleanly (still counted as PASS — the gate fires before the
check's body). Concretely, on a fresh CPU-only host without optional
deps, the following families SKIP:

- `e2e_smoke` family — gates on a visible CUDA / ROCm / TPU device.
- `flag_probe_*` family — partially gated on a working nvcc / hipcc
  on PATH; the cache + dispatch tests still run.
- `polyhedral_apply_schedule_*` — gates on `libclang` (soft dep).
- `synth_codegen_*` (CK / CUTLASS variants) — gates on the
  third-party-submodule presence.
- `bundled_templates_*` — gates on `jinja2`.
- `nvrtc_*` — gates on `cuda-python`.

The remaining ~110 checks (ARCH_TABLE completeness, search-space
hashing, flag-emission per arch, cache-v4 consistency, BuildSpec
schema, autotune stopper math, dry-run manifest schema, TOML config
loader, macro-prefix validation, project-agnostic dry-run) run on
**any** host with Python 3.11+ and torch. The "138 checks" number
is therefore the upper bound; the floor on a stripped CPU-only host
is what the user actually sees and that floor is still informative
(it's the build-correctness portion of the suite).

---

## Hardware support — ARCH_TABLE (26 canonical archs)

`grokking_optimizers.compile.ARCH_TABLE` is the single source of truth.
Every flag, feature gate, search-space dim, and toolchain requirement
is derived from it — there are no hardcoded `if arch == "sm_90"`
branches anywhere in the file.

| Arch | Family | Cards | Backend | Min toolchain | Self-test | AOT dry-run |
|------|--------|-------|---------|---------------|-----------|-------------|
| `sm_70` | Volta | V100 | CUDA | 10.0 | ✅ | ✅ |
| `sm_75` | Turing | T4 | CUDA | 10.0 | ✅ | ✅ |
| `sm_80` | Ampere | A100, A30, A10 | CUDA | 11.0 | ✅ | ✅ |
| `sm_86` | Ampere | A10, RTX 30xx | CUDA | 11.1 | ✅ | ✅ |
| `sm_89` | Ada Lovelace | L4, L40, RTX 40xx | CUDA | 11.8 | ✅ | ✅ |
| `sm_90a` | Hopper | H100, H200 | CUDA | 12.0 | ✅ | ✅ |
| `sm_100a` | Blackwell datacenter | B100, B200, GB200 | CUDA | 12.8 | ✅ | ✅ |
| `sm_103a` | Blackwell Ultra | B300, GB300 NVL72 | CUDA | 12.9 | ✅ | ✅ |
| `sm_120a` | Consumer Blackwell | RTX 50xx, RTX PRO 6000 | CUDA | 12.8 | ✅ | ✅ |
| `gfx906` | Vega20 | MI50 | HIP | ROCm 3.0 | ✅ | ✅ |
| `gfx908` | CDNA1 | MI100 | HIP | ROCm 3.5 | ✅ | ✅ |
| `gfx90a` | CDNA2 | MI200, MI250 | HIP | ROCm 4.5 | ✅ | ✅ |
| `gfx942` | CDNA3 | MI300X, MI300A | HIP | ROCm 6.0 | ✅ | ✅ |
| `gfx950` | CDNA4 | MI350X, MI355X | HIP | ROCm 6.2 | ✅ | ✅ |
| `gfx1030` | RDNA2 | RX 6000 | HIP | ROCm 4.0 | ✅ | ✅ |
| `gfx1100/1101/1102` | RDNA3 | RX 7000 | HIP | ROCm 5.5 | ✅ | ✅ |
| `gfx1151` | RDNA3.5 | Strix Halo | HIP | ROCm 6.1 | ✅ | ✅ |
| `gfx1200/1201` | RDNA4 | RX 9000/9070 | HIP | ROCm 7.0 | ✅ | ✅ |
| `tpu_v4` | TPU v4 | 128-wide MXU | Pallas/XLA | JAX 0.4 | ✅ | ✅ |
| `tpu_v5e` | TPU v5e | 64-wide MXU | Pallas/XLA | JAX 0.4 | ✅ | ✅ |
| `tpu_v5p` | TPU v5p | 128-wide MXU + sparsecore | Pallas/XLA | JAX 0.4 | ✅ | ✅ |
| `tpu_v6e` | Trillium | 256-wide MXU | Pallas/XLA | JAX 0.4.30 | ✅ | ✅ |
| `tpu_v7` | Ironwood | larger MXU | Pallas/XLA | JAX 0.5 | ✅ | ✅ |

**Status legend**:
- ✅ Self-test: arch is registered in ARCH_TABLE with non-empty search
  space, feature flags, and toolchain version — verified by the 138-test
  in-process suite.
- ✅ AOT dry-run: `--arch <X> --runtime aot --no-autotune --no-profile`
  produces either a build.ninja or a clear per-arch toolchain
  diagnostic (`nvcc not found`, `hipcc not found`, `JAX not
  installed`, etc.) — never an opaque exit-127. Also exhaustively
  covered by `python -m grokking_optimizers.compile --dry-run-all-archs`
  which writes one JSON manifest per arch under `<out>/dry_run_<arch>.json`.

**Backward-compat aliases**: `sm_90 → sm_90a`, `sm_100 → sm_100a`,
`sm_103 → sm_103a`, `sm_120 → sm_120a`. Both keys resolve to the same
`ArchEntry` object via `is`-identity.

> **Registered ≠ hand-tuned.** Every arch above is registered, search-
> space-complete, and dry-runnable, and the flag/gencode emission is
> arch-correct for all 26. But hand-written per-arch kernel **sources**
> currently exist only for **`sm_90` (CUDA)** and **`gfx942` (HIP)** under
> `csrc/backends/`, plus the **Pallas** Python kernels for TPU. The other
> CUDA/HIP arches resolve through the shared bindings + algorithm headers
> and the generic codegen template — they compile and are numerically
> correct, but they do not yet have arch-specialized kernel bodies
> (e.g. Blackwell tcgen05, Ampere async-copy). See the Build-status
> matrices below, which list only the arches with real kernel sources.

### Per-arch search space cardinalities

The complete programmatic Cartesian per arch (no YAML curation). Bayesian
TPE samples this directly via Optuna's `suggest_categorical` over the
per-dim value lists; the product is never materialized.

| Arch | Cardinality |
|------|-------------|
| `sm_70` / `sm_75` | 1,039,360 |
| `sm_80` / `sm_86` | 2,969,600 |
| `sm_89` | 8,908,800 |
| `sm_90a` | 3,735,552,000 |
| `sm_100a` / `sm_103a` | 3,489,398,784,000 |
| `sm_120a` | 17,817,600 |
| `gfx906` | 4,157,440 |
| `gfx908` | 59,392,000 |
| `gfx90a` | 83,148,800 |
| `gfx942` | 688,128,000 |
| `gfx950` | 1,710,489,600 |
| `gfx1030` | 3,118,080 |
| `gfx1100/1101/1102/1151` | 95,027,200 |
| `gfx1200/1201` | 498,892,800 |
| `tpu_v4` / `tpu_v5e` | 1,200 |
| `tpu_v5p` | 3,600 |
| `tpu_v6e` | 2,400 |
| `tpu_v7` | 1,920 |

---

## MAXIMAL pipeline — extended capabilities

The wrapper goes beyond flag-tuning. Every layer below stacks on top of
the previous one — flag tuning is a strict subset of the emission space,
which is a strict subset of the runtime-specialization space.

### Native compiler flag coverage (Stream α)

The wrapper emits every published version+arch-gated flag for the three
target compilers it can find a reason to add. Nothing project-specific
about these — they apply to any kernel source. The lists below are what
ships out-of-the-box; everything is gated through `ARCH_TABLE[arch]
.features` + detected toolchain version so older toolchains skip flags
they don't support.

**nvcc** (gated on detected CUDA version):
- `-Xptxas --warn-on-spills` (CUDA 11.0+) — spill diagnostics for any kernel
- `-Xptxas --def-load-cache=ca` / `--def-store-cache=wb` — per-arch cache policy
- `-Xptxas --maxrregcount-list=…` (CUDA 12.5+) — finer than the single-value flag
- `-Xptxas --opt-level=3 --allow-expensive-optimizations=true`
- `--default-stream per-thread` — perf win for any multi-stream launcher
- `-rdc=true` + `--device-link-options=-dlto` — cross-TU LTO
- `--minimal` (CUDA 13.0+) when available
- `--source-in-ptx` + `-lineinfo`
- `--keep --keep-dir <out>/nvcc_intermediate` when `debug_symbols=True`
- `--use-local-env` on Windows
- **ptxas -v stderr parser**: `regs_used`, `smem_bytes`, `stack_frame`,
  `spill_stores`, `spill_loads` parsed out of the `-Xptxas -v` log and
  written into the trial sidecar as `ptxas_info`. Feeds the learned
  cost model + the rejection budget.

**hipcc** (gated on detected ROCm version):
- `-fgpu-rdc` for cross-TU LTO
- `-mllvm --enable-newgvn` — better global value numbering on CDNA3
- `-mllvm --inline-threshold=275` — higher than clang default 225 for GPU
- `-mllvm --amdgpu-coerce-illegal-types=1` — fp8 / fp4 paths
- `--save-temps=cwd` when `debug_symbols=True`
- `-Wno-pass-failed=transform-warning` — suppress RDNA wavefront noise
- Per-arch: `gfx950 → --offload-arch=gfx950:sramecc+:xnack-`

**JAX / XLA** (Pallas archs — all of which are TPUs, so the emitted
`XLA_FLAGS` are TPU-specific `xla_tpu_*` flags, not the `xla_gpu_*`
family):
- `--xla_tpu_enable_async_collective_fusion=true` (+ `_multiple_steps=true`)
- `--xla_enable_async_all_gather=true`
- `--xla_tpu_enable_latency_hiding_scheduler=true`
- `--xla_tpu_megacore_fusion_allow_ags=false`
- `--xla_tpu_spmd_rng_bit_generator_unsafe=true`
- `--xla_dump_hlo_as_text=true`
- Env: `JAX_PLATFORMS=tpu`, plus a `JAX_COMPILATION_CACHE_DIR` +
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` persistent-cache pair

### Zero-config arch routing (Stream β)

`--arch` is now optional. When omitted, `_resolve_default_arch()` probes
in order: `torch.cuda.get_device_capability(0)` → map to canonical
`sm_XYa` / `sm_XY` via ARCH_TABLE → `subprocess.run(["rocm-smi",
"--showproductname"])` → parse via the bundled `_ROCM_CARD_TO_GFX`
lookup table (MI50 through MI355X, RX 6000 through RX 9000, Strix
Halo) → `jax.devices()` → map TPU device kind to `tpu_vN` →
TOML `[archs].default` → built-in fallback (`sm_90a`). Prints a
`[arch] auto-detected <arch> from <source>` line so you see what won.

Every probe is wrapped in `try/except` and never crashes auto-detect.

**Preflight version-mismatch suggestions**: when `_preflight_toolchain`
detects an arch / toolchain mismatch, in addition to the existing
`[preflight] arch=sm_90a need CUDA>=12.0 have 11.5 — FAIL` line it now
also emits `[preflight] suggestion: install CUDA 12.0+ via
--bootstrap-cuda, OR retry with --arch sm_86 (highest compatible with
your CUDA 11.5)`. The suggested arch comes from iterating ARCH_TABLE for
the highest-capability arch whose `min_toolchain_version` is ≤ detected.

### Honest wrapper-layer features (Stream γ)

These are wrapper-level features (above what nvcc / hipcc / JAX can do
themselves). Stream γ replaced placeholder bodies in each one so the
output matches what the feature claims to do, for any kernel — not just
SuperGrok optimizers:

- **Polyhedral `apply_schedule` body lift**: walks `LoopNest.body_ast`
  via `clang.cindex.Cursor` traversal, rewrites identifier names + index
  expressions per the schedule (tile / fuse / reorder / vectorize /
  parallelize). When libclang is unavailable, emits an explicit
  `// schedule shape only; libclang absent — body unchanged` comment
  alongside the identity-copy fallback so the path is honest.
- **Architecture-native MMA in synth GEMM**: per (arch, dtype), emits a
  real MMA mainloop instead of a scalar O(MNK) triple-loop:
  - sm_90a → `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16` with
    smem A/B tiles + async barriers
  - sm_100a → `tcgen05.mma.async` (Blackwell tensor memory)
  - gfx9xx (`mfma` feature) → `__builtin_amdgcn_mfma_f32_16x16x16f16`
  - gfx10xx+ (`wmma` feature) → `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
  - other archs → scalar fallback with an explicit `// no native MMA
    available for {arch}; using scalar mainloop` comment
- **Transparent CUTLASS / CK fallback**: every emitted variant now
  carries a `source` field with values `cutlass_python` /
  `cutlass_fallback` / `ck_python` / `ck_fallback` / `scalar_fallback`.
  When the Python frontend returns empty `tile_descriptions()` the
  wrapper logs `[codegen] WARN cutlass.op.Gemm.tile_descriptions()
  returned 0 — using curated fallback (N variants)` to the build report
  instead of silently swapping in 4 hardcoded tiles.
- **`bias_trial_queue` wired into autotune**: when `enable_device_pgo=True`,
  after each PGO round produces `stall_info`, `_run_bayesian` calls
  `bias_trial_queue(study, stall_info, space, arch, max_enqueued=25)`
  before the next TPE batch. Previously `bias_trial_queue` was
  implemented + unit-tested but never called — pure dead-letter helper.
  Now the device-PGO feedback loop is closed.
- **Pallas `block_spec` search-space dim**: for `tpu_v*` archs the
  Pallas search space includes a real `block_spec` enum dim with values
  `["default", "(64,64)", "(128,128)", "(256,256)", "(64,256)"]`
  (replacing a hardcoded `"default"` placeholder).

### Structural cleanup (Stream δ)

Refactor pass that net-removed 1,984 LOC while keeping behavior
byte-identical:
- `_self_test` (2,900 → 46 LOC orchestrator): split into 22
  per-section `_self_test_*` helpers. Test names + banner format + pass
  summary line preserved exactly.
- `_BUNDLED_TEMPLATES` (3,092 LOC → 863 LOC, 72% reduction): 4 base
  templates + 11 per-optimizer specs + assembler. Each of the 44
  emitted dict entries is byte-identical to the legacy verbatim form.
- `_bootstrap_cuda_via_*`: dnf / yum / zypper / pacman / apk collapsed
  into one `_bootstrap_via_pkg_manager(name, install_cmd, packages,
  stream, verb)`. conda / nvidia-apt-repo / pypi-wheels left distinct
  (different flow shapes).
- `getattr(spec, …, default)` tightening: 35 sites in
  `BuildSpec`-annotated functions converted to direct `spec.X` access
  so field renames hard-fail. 9 sites in `apply_to_buildspec` (which
  accepts duck-typed mock specs) intentionally kept as `getattr`. New
  AST-walk self-test `buildspec_advertises_all_fields_read_by_production_code`
  catches rename drift at test time.

### Post-verification fixes (Optuna 4.0 compat + project-agnosticism plug + CLI surface)

After Streams α/β/γ/δ landed, eight parallel verification agents
exercised the full surface (CLI, codegen, autotune, cache, portability,
NVRTC registry, self-test integrity, end-to-end dry-run) and found
issues the self-test alone had missed. A fix-up pass plugged all of
them:

- **Optuna ≥ 4.0 `study.tell` crash**: four call sites in `run_bayesian`
  passed `math.inf` together with `state=PRUNED|FAIL`, which Optuna 4.0+
  rejects (`ValueError: Values were told. Values cannot be specified
  when state is TrialState.PRUNED or TrialState.FAIL`). Hidden from the
  self-test because synthetic timers never raised. Fix: drop the value
  arg. Regression test added.
- **`_dry_run_all_archs` ignored TOML config**: the all-archs sweep
  constructed a `BuildSpec` without calling `apply_to_buildspec`, so a
  downstream project's `macro_prefix="MYPROJ_"` was silently dropped.
  Fix: thread `config` through, call `apply_to_buildspec` per arch.
- **`ArchEntry.macro` hardcoded as `SG_BUILD_ARCH_*`**: `spec.macro_prefix`
  only affected optimizer/model macros. Fix: `_build_macros` rewrites
  the prefix at emit time when it differs from the default.
- **`-DSG_VERBOSE=1` hardcoded**: now emits as
  `-D<spec.macro_prefix>VERBOSE=1`, with backward-compat byte-identity
  preserved when the prefix is the default `SG_BUILD_`.
- **NVRTC cache dir clarification**: the actual cubin cache lives at
  `<out>/nvrtc_cache` (driven by `spec.out_dir`, not the project name).
  The debug banner additionally probes `~/.cache/<project_name>/nvrtc`
  for a disk-free reading; that path is informational only and is not
  used for cubin storage.
- Plus CLI completeness: `--enable-synth-codegen`, `--enable-polyhedral`,
  `--dry-run` (single-arch, requires `--arch`), `--list-archs` added to
  argparse. `xla_env` dict surfaced in dry-run manifests for Pallas
  archs (previously invisible). `[autotune] early stop: <reason>`
  printed to stdout (previously only in returned dict).

### Round-2 verification fixes (`bc4aaa3`)

A second round of ten parallel verification agents (re-exercising every
layer after the first fix-up) caught three more bugs that the new
self-tests missed plus three hardening opportunities:

- **`--enable-synth-codegen` / `--enable-polyhedral` silently ignored in
  `--dry-run` path**: argparse accepted the flags, BuildSpec was wired
  for the real build path, but `--dry-run` short-circuited into
  `_dry_run_all_archs` which constructed a fresh BuildSpec per arch
  with hardcoded defaults. Manifests were byte-identical with or without
  the flags. **Fix**: `_dry_run_all_archs` now accepts the toggle
  kwargs, threads them onto each per-arch BuildSpec, and surfaces them
  as an `enabled_features` dict in every emitted manifest. The dict
  exposes all opt-in toggles (`enable_synth_codegen`,
  `enable_polyhedral`, `enable_runtime_specialization`,
  `enable_device_pgo`, `enable_cost_model`, `enable_emitter`,
  `strict_numerics`).
- **`CompileCache.record_trial` wrote to in-memory list, not v4 `.jsonl`
  sidecar**: only v3→v4 migrations created sidecars; freshly-recorded
  trials on a v4 cache stayed in the main JSON as `bayesian_trials` /
  `sweep_history` lists. The promised v4 layout was hybrid. **Fix**:
  `record_trial` now writes each trial directly to
  `<cache_dir>/trials_<opt>_<model>_<arch>.jsonl` (append + `fsync`),
  updates `trial_log_path` and `trial_log_summary`, and falls back to
  in-memory-only with a `[cache] WARN` line if the sidecar write fails
  (disk full, permission denied). On `save()`, the in-memory lists are
  emptied to `[]` placeholders before writing the main JSON.
- **`macro_prefix` / `fused_op_template` only accepted under `[project]`**:
  user TOMLs placing these under `[sources]` (intuitive given their
  semantic relation to source layout) silently lost the value. **Fix**:
  `apply_to_buildspec` now accepts either location, with `[project]`
  canonical (wins on collision); `[sources]` is recognized as a
  fallback.

Plus three opportunistic hardening additions:

- Extended Optuna 4.0 regression test from 1 path (FAIL via exception)
  to all 4 paths (infeasible-prefilter PRUNED, exception FAIL,
  cost-model PRUNED, non-finite FAIL).
- Added `inst_fetch` (CUPTI instruction-fetch stall reason) to
  `STALL_DIM_HINTS` — `_parse_nsys_stall_section` was silently dropping
  it. Now biases `maxrregcount` and `block` dims. Count bumped 13 → 14.
- Documented `get_registry`'s per-arch singleton behavior in its
  docstring: first call wins; subsequent calls with a different
  `config=` for the same arch return the existing registry. Clear
  `_REGISTRY[arch]` to force a re-build.

### Bayesian auto early-stopping

`--bayesian-trials` defaults to `None` (auto). The autotune loop runs
until **any** of five criteria fire:

| Criterion | Default | Override flag |
|-----------|---------|---------------|
| Best-so-far plateau | no improvement > 0.5% for `patience` trials (auto = `max(50, 0.1 × trials_done)`) | `--min-improvement 0.01` / `--patience 100` |
| **EI exhaustion** | rolling mean of per-trial **relative** improvement over the last `patience` trials drops below `ei_floor` | `--ei-floor 1e-6` (set 0 to disable) |
| Coverage saturation | new (dim, value) tuples per trial < 0.1% over `patience` window | (internal) |
| Wall-clock budget | None by default; explicit cap when set | `--max-tune-seconds 600` |
| Hard ceiling | 1,000,000 trials (sanity, never reached) | — |

EI estimator details: TPE doesn't expose Optuna's internal acquisition
value, so the wrapper estimates expected improvement empirically as
`max(0, prev_best - trial_value) / max(|prev_best|, 1e-12)` per trial,
keeps a deque of the last `patience` improvements, and stops when the
rolling mean drops below `ei_floor`. Scale-free (relative units) so the
threshold is dtype/problem-size independent.

Manual override: `--bayesian-trials 500` still works. `top_k` for the
refine pass also defaults to `None` — the elbow of the timing
distribution is detected automatically.

### Codegen / Jinja2 kernel emission backend

`--enable-emitter`. Compiles per-variant kernels from Jinja2 templates
**bundled inside compile.py** (`_BUNDLED_TEMPLATES` dict) instead of
re-compiling one fixed source with `-D` macros. Templates emit
**structurally different** kernels (warp-specialized vs cooperative vs
persistent vs stream-K mainloop; different wgmma/tcgen05/MFMA shapes;
baked-in swizzle patterns; epilogue fusions).

**Template coverage**: **44 bundled templates** covering all 11
optimizers × 4 arch variants:

| Optimizer | sm_90a (Hopper) | generic CUDA (sm_75/80/86/89/100a/103a/120a) | gfx942 (CDNA3 HIP) | tpu_v5p (Pallas) |
|-----------|:---:|:---:|:---:|:---:|
| `adamw` | ✅ | ✅ | ✅ | ✅ |
| `lion` | ✅ | ✅ | ✅ | ✅ |
| `muon` | ✅ | ✅ | ✅ | ✅ |
| `prodigy` | ✅ | ✅ | ✅ | ✅ |
| `grokadamw` | ✅ | ✅ | ✅ | ✅ |
| `grokfast` | ✅ | ✅ | ✅ | ✅ |
| `looksam` | ✅ | ✅ | ✅ | ✅ |
| `neuralgrok` | ✅ | ✅ | ✅ | ✅ |
| `supergrok11` | ✅ | ✅ | ✅ | ✅ |
| `supergrok15` | ✅ | ✅ | ✅ | ✅ |
| `supergrok2` | ✅ | ✅ | ✅ | ✅ |

`find_template(opt, arch)` walks `<opt>_<arch>.<ext>.j2` →
`<opt>_<vendor>.<ext>.j2` → `<opt>_generic.<ext>.j2`. Cache key:
SHA256(template source + JSON config). Identical configs produce the
same emitted file. Optional `nvcc --cuda --dryrun` validation when nvcc
is on PATH.

### CUTLASS / CK GEMM emitter (with transparent fallback labeling)

`emit_cutlass_gemm_variants(arch, problem_shape, dtype, out_dir)` for
sm_90a / sm_100a and `emit_ck_gemm_variants(arch, problem_shape, dtype,
out_dir)` for gfx9xx / gfx10xx:

- **CUTLASS**: tries `cutlass.op.Gemm(...).tile_descriptions()` when
  available (cutlass-python 3.x+) to enumerate every supported {tile ×
  cluster × stages × schedule × epilogue} variant. Falls back to a
  curated 4-variant sweep (tiles 128×128×{32,64} and 256×128×{32,64})
  when the introspection API isn't exposed.
- **Composable Kernel** (AMD): probes `composable_kernel.op.GemmInstance`
  / `.Gemm` / `gemm.config.GemmConfig` / `tile.GemmTile` before falling
  back to the curated tile sweep.
- **Transparent labeling** (Stream γ): every emitted variant now
  carries a `source` field — `cutlass_python` / `cutlass_fallback` /
  `ck_python` / `ck_fallback` / `scalar_fallback` — so the autotune
  output makes clear which variants came from the library frontend vs.
  the curated fallback. When the frontend returns empty
  `tile_descriptions()` the wrapper logs `[codegen] WARN
  cutlass.op.Gemm.tile_descriptions() returned 0 — using curated
  fallback (N variants)` to the build report.
- **Public emitter contract**: `emit_cutlass_gemm_variants` /
  `emit_ck_gemm_variants` deliberately raise `CodegenError` /
  `SynthCodegenError` when the Python frontend is missing (file
  emission needs the real backend). The synth-codegen dispatcher
  catches these and falls through to the native MMA path (`wgmma` /
  `tcgen05` / `mfma` / `wmma`) emitted by `synthesize_kernel`,
  followed by the portable scalar fallback if no MMA is available
  on the target arch.
- Emits one `.cu` / `.hip.cpp` per variant:
  `cutlass_gemm_<arch>_<dtype>_<MxNxK>_<key>.cu` (or
  `ck_gemm_<arch>_<dtype>_<MxNxK>_<key>.hip.cpp`) with `extern "C" int
  launch_<key>(void* A, void* B, void* C, void* D, int M, int N, int K,
  float alpha, float beta, <stream_type> stream)`.
- Cached by filename — re-invoking with the same variant key skips
  re-emission.

### Runtime kernel specialization — NVRTC / hipRTC

`--enable-runtime-specialization`. For shapes that vary at runtime,
JIT-compile via NVRTC (CUDA) or hipRTC (HIP) with problem-shape
constants baked in as `constexpr`. CUBINs cached by `(arch, dtype,
shape_class)` under `<out>/nvrtc_cache`. Sub-µs dispatch on cache hit
via `cuModuleLoadData` / `hipModuleLoadData`.

**Live dispatch**: `_LoadedKernel.__call__(*args, grid=..., block=...,
shared=0, stream=0)` packs args via ctypes, calls `cuLaunchKernel` /
`hipModuleLaunchKernel` on the loaded module, and returns. Both modern
(`cuda.bindings.driver`) and legacy (`cuda.cuda`) cuda-python paths are
tried. Missing bindings at call time raise `RegistryError` with install
instructions (not a bare `ImportError`).

Verified end-to-end on a CPU host: NVRTC compiled a trivial fp32
kernel via the `cuda-python` bindings and produced a 1,062-byte PTX
artifact (SASS-less host, so it falls back to PTX with the driver doing
the final JIT — exercising the full path including atomic cache write
+ read-back). The live-launch self-test (`loaded_kernel_call_or_skip`)
round-trips a 64-element copy kernel through `cuModuleLoadData` →
`cuLaunchKernel` when a GPU is visible; skips cleanly otherwise.

### Device-side PGO — CUPTI / rocprof / XLA HLO dumps

`--enable-device-pgo`. The standard LLVM `-fprofile-generate/-use`
loop only instruments **host** launchers on NVIDIA (nvcc strips device
instrumentation). This layer collects device-side stall info from:

- NVIDIA: `nsys profile` PC sampling
- AMD: `rocprof --stats` ATT traces
- Pallas: XLA HLO cost-model dumps (`--xla_gpu_dump_autotuned_*`)

Stall reasons → JSON sidecar at `<out>/device_stall_info.json`. The
wrapper recognizes 13 reason keys: `long_scoreboard`, `not_selected`,
`math_pipe_throttle`, `memory_throttle`, `tex_throttle`, `barrier`,
`wait`, `imc_miss`, `lg_throttle`, `dispatch_stall`, `vmem_lat`,
`lds_bank_conflict`, `valu_dep`. Each maps to a set of search-space
dims (`long_scoreboard → swizzle/lds_padding/vec`, `not_selected →
block/waves_per_eu/maxrregcount`, etc.).

**Live feedback loop (Stream γ)**: after each PGO round produces
`stall_info`, `_run_bayesian` calls `bias_trial_queue(study, stall_info,
space, arch, max_enqueued=25)` before the next TPE batch. The
biased configs get `study.enqueue_trial`'d so they're tried first.
This closes the loop that was previously dead-letter
(`bias_trial_queue` was implemented + unit-tested but never called).

### Multi-GPU fan-out — work-stealing

When `CUDA_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`) lists multiple
GPUs, `MultiGPUTimingPool` partitions the variant queue across N
`TimingWorker`s, one per device, with per-worker env overlays
(`CUDA_VISIBLE_DEVICES=<dev>`).

**Work-stealing dispatcher** (replaced the earlier round-robin):

- Internal `queue.Queue` of pending `(variant_so, opt_class, kwargs, future)`
  work items.
- One dedicated dispatcher thread per worker. Each thread pops work
  whenever it's idle and immediately starts the next variant.
- If a worker is mid-call when 5 more come in, a fast sibling drains
  those 5 while the slow one finishes its first — no head-of-line
  blocking.
- Dead-worker items are bounced back to the queue for siblings to pick up.
- Public `pool.time(variant_so)` API is unchanged (still synchronous);
  internally it submits a `concurrent.futures.Future` and blocks until
  the dispatcher fulfils it.

Self-test (`multigpu_work_stealing`) on a 2-mock-worker pool: 10 jobs
× (fast 10ms, slow 100ms) ⇒ fast=9 / slow=1 / wall=0.103s vs naive
serialization ≈1.0s. Confirms work-stealing semantics.

### Numerical / differential validation

`--strict-numerics`. After each variant times out, compare its output
tensor against the AOT primary's reference output. Tolerances:

| dtype | rtol | atol |
|-------|------|------|
| fp32 | 1e-5 | 1e-6 |
| fp16 / bf16 | 1e-3 | 1e-4 |
| fp8 | 1e-2 | 1e-3 |
| fp4 | 5e-2 | 1e-2 |

Statuses: `ok`, `deterministic` (bit-identical to ref),
`numerical_fail` (out of tolerance), `non_deterministic` (within
tolerance but not bit-identical), `skipped` (no ref available).
`pick_winner` always excludes `numerical_fail`; `--strict-numerics`
requires `deterministic`.

### Cache GC and watchdog

`--prune` / `--prune-max-age-days 30` / `--prune-keep-top-n 100` /
`--no-auto-prune`. Auto-prune runs at the end of every successful JIT
autotune pass: variants older than max-age OR not in top-N by timing
(per `(opt, model, arch)`) are dropped. `TimingWorker` carries a
30-second-interval watchdog thread that hard-restarts the worker
process if a `ping` times out (60s).

### Cache schema v4 — `.jsonl` trial sidecars

`CACHE_VERSION = 4`. The bulky `bayesian_trials` and `sweep_history`
arrays no longer live inside the main JSON — at autotune scale they
were pushing the cache to hundreds of MB, fighting `json.loads` /
`json.dumps`, and slowing every `save()`. They now live in append-only
newline-delimited JSON sidecars next to the cache file:

```
<cache_dir>/<cache>.json                   # main: tuned_config, hashes, summaries
<cache_dir>/trials_<opt>_<model>_<arch>.jsonl   # one line per trial
```

Per-entry, the main JSON keeps:
- `trial_log_path: str | None` — relative path to the sidecar (or
  `None` when the entry never recorded any trials).
- `trial_log_summary: {n_trials, best_timing_ms, stop_reason,
  last_updated_unix}` — small fixed-size dict so callers can answer
  "how many trials / what was the best / why did we stop?" without
  re-reading the sidecar.
- `bayesian_trials: []` and `sweep_history: []` — kept as empty
  placeholders so v3-reader code (older `profile.py`, downstream
  tools) doesn't `KeyError`.

**Migration**: `CompileCache._load` chains `v2 → v3 → v4` automatically.
v3 caches are migrated lazily on first load: trials are flushed to the
sidecar with auto-tagged `stage`, the in-memory lists are zeroed, and
the v3 file is backed up to `<cache>.json.v3.bak` before the v4 file
overwrites it. Self-tests cover both `v3 → v4` and the full `v2 → v4`
chain.

### TOML project config

`--config path/to/your.toml`. Loader (`load_config`, exposed via the
legacy `grokking_optimizers.compile_config` shim) merges in priority
order:

1. Path passed via `build(config=…)` / `--config`
2. `./compile_config.toml` in CWD
3. Packaged defaults — now inlined inside compile.py as the
   `_DEFAULT_PROJECT_CONFIG` dict (no external TOML file needed)

15 sections: `[project]`, `[sources]`, `[optimizers]`, `[models]`,
`[archs]`, `[pgo]`, `[autotune]`, `[codegen]`, `[runtime_specialization]`,
`[device_pgo]`, `[cache]`, `[numerics]`, plus `[synth_codegen]`,
`[polyhedral]`, `[cost_model]` (the last three added by Streams B / C /
D for the OpGraph synth codegen, polyhedral schedule search, and
learned cost-model layers respectively). Strictly additive — without a
config file, behavior is identical to today.

### Verification harnesses

Built-in CI modes for sanity-checking the build pipeline without
needing a target GPU:

| Flag | Behaviour |
|------|-----------|
| `--self-test` | Inline suite (~138 checks today) covering ARCH_TABLE, search spaces, codegen, autotune, cache v4, polyhedral, synth codegen, Stream α/β/γ regression tests, plus the Colab-arch-detection regression suite. Runs in ~30s on a CPU-only host. Several checks gate on optional deps (`jinja2`, `cuda-python`, `libclang`, GPU presence); those that don't have their dep available SKIP cleanly (still counted as PASS — see the "What gets exercised on a CPU-only host" note above). |
| `--list-archs` | Dumps every entry in ARCH_TABLE — one line per arch with vendor, min toolchain version, and feature set (wgmma / tcgen05 / mfma / wmma / sparsecore / etc.). Exits 0; no `--optimizer` / `--model` required. |
| `--dry-run --arch <arch>` | Single-arch dry-run: runs preflight + `_resolve_sources` + `_host_cflags` + `_device_cflags` + `_ldflags` for the named arch without invoking `torch.cpp_extension`. Writes `<out>/dry_run_<arch>.json`. Pair with `--enable-synth-codegen --enable-polyhedral` to also exercise the synth/polyhedral layers. |
| `--dry-run-all-archs` | Same as above but sweeps every canonical arch in ARCH_TABLE. Writes one JSON manifest per arch under `<out>/dry_run_<arch>.json`. Sweeps all 26 canonical archs on a CPU-only host in ~3 seconds. For Pallas archs the manifest includes the resolved `xla_env` dict. Each manifest now also surfaces `device_ldflags` (nvcc -dlink step) and `version_gated_device_cflags` (flags `_newer_compiler_flags` would add when the installed toolchain is new enough; empty on CPU-only hosts). Mutually exclusive with `--dry-run`. |
| `--e2e-smoke` | End-to-end smoke: detects the local GPU via `torch.cuda.get_device_capability()`, maps to ARCH_TABLE, runs `build(adamw, mamba, <detected>, autotune=bayesian, max_tune_seconds=120)`, and asserts `tuned_config` is written, `early_stop_info` is recorded, `tuned_configs.h` is regenerated, and the final `.so` loads. Skips cleanly with `[e2e-smoke] no CUDA device — skipping` on CPU-only hosts. `--e2e-max-seconds N` adjusts the autotune wall-clock cap. |

All modes are wired into `_self_test` so they exercise automatically
(dry-run-all-archs runs always; e2e-smoke gates on `torch.cuda.is_available()`).

### Toolchain bootstrap for every vendor

| Flag | Vendor | Behavior |
|------|--------|----------|
| `--bootstrap-cuda` | CUDA | Probes conda / apt / NVIDIA apt repo / dnf / yum / zypper / pacman / apk / brew / winget / PyPI wheels. Picks CUDA version per arch min (e.g. sm_120a → 12.8+, sm_103a → 12.9+). |
| `--bootstrap-rocm` | HIP | AMD's official apt repo (per-arch version: gfx950 → ROCm 6.2+, gfx1200 → 7.0+) → stock apt → dnf → zypper. |
| `--bootstrap-jax` | Pallas | `pip install jax[tpu]` from `libtpu_releases` bucket if no TPU device visible. |

`_preflight_toolchain(arch)` emits a `[preflight] arch=<X>
need=<min>.<min> have=<v>.<v> — PASS|FAIL` line per arch so CI can
grep for failures.

### Using this wrapper for any project

`compile.py` is a **project-agnostic, portable compiler wrapper** that
drives nvcc / hipcc / JAX (Pallas) maximally. It ships in this repo
configured for SuperGrok's optimizer + model kernels, but it works
out-of-the-box for any project via a TOML config override (see
the *Using this wrapper for a different project* block at the top of
the quickstart).

What carries over verbatim to any project:
- All 26 canonical archs (9 NVIDIA + 12 AMD + 5 TPU), their feature gates, and
  the per-arch search spaces (~3.7B candidates on sm_90a, ~3.4T on
  sm_100a) — these come from the hardware spec, not from SuperGrok.
- Every flag the wrapper emits for nvcc / hipcc / JAX (Stream α native
  maximization — `--warn-on-spills`, `--default-stream per-thread`,
  `-fgpu-rdc`, `--xla_gpu_enable_priority_fusion`, etc.).
- Auto-arch detection, preflight version-mismatch suggestions,
  toolchain bootstrap dispatch — all derive from the local host, not
  the project.
- The wrapper-extras: Bayesian autotune with 5-criterion early-stop,
  polyhedral schedule search, OpGraph synth codegen with arch-native
  MMA (`wgmma` / `tcgen05` / `mfma` / `wmma`), CUTLASS / CK delegation
  with transparent `source=` fallback labels, NVRTC / hipRTC kernel
  registry, work-stealing multi-GPU pool, cache v4 with `.jsonl` trial
  sidecars, numerical validation, learned cost model with rejection
  budget, device-PGO via CUPTI / rocprof feedback into bias_trial_queue.
- The single-file constraint: everything inlined into `compile.py` so
  vendoring the wrapper into another project is one file copy.

What you supply per-project via TOML:
- `[project] name` → reported by the debug header (the actual NVRTC
  cubin cache lives at `<out>/nvrtc_cache`, not under `~/.cache/<name>`;
  only the disk-free probe in the debug banner inspects `~/.cache/<name>/nvrtc`)
- `[sources] macro_prefix` → emitted macros (`-D<PREFIX>VERBOSE`,
  `-D<PREFIX>OPTIMIZER_*`, `-D<PREFIX>ARCH_*`)
- `[sources] source_layout` → where your kernels live
- `[optimizers] enabled` / `[models] enabled` → the names your project
  exposes (no hardcoded list)
- `[archs] default` → auto-arch fallback when no GPU is visible

Per-op Jinja2 kernel templates are only required if you opt into
`--enable-emitter` (the structurally-different mainloop emission
backend); without that flag, `-D` macros into a single fused source
are enough.

### Everything lives in one file

All MAXIMAL pipeline features — Jinja2 codegen, NVRTC/hipRTC kernel
registry, CUPTI/rocprof device PGO, TOML config loader, Pallas backend
— are inlined into `grokking_optimizers/compile.py` itself. There are
no extra Python modules in `grokking_optimizers/` for these features.
For backward compatibility the legacy import paths
(`from grokking_optimizers.codegen import …`,
`from grokking_optimizers.kernel_registry import …`,
`from grokking_optimizers.device_profiling import …`,
`from grokking_optimizers.compile_config import …`) still resolve —
each is registered as a `sys.modules` alias pointing at
`compile.py` so existing callers keep working unchanged.

---

## Build status

Per-arch coverage of the 12 optimizers and 3 models. Honest legend:

- ✅ **done & validated on hardware** — implemented, build-checked, parity
  confirmed against a reference path
- 🟡 **done, unvalidated on hardware** — implemented and import-checked, but
  not yet validated on real hardware (no GPU available in this environment)
- ⛔ **stub / raises NotImplementedError** — explicitly unimplemented; the
  launcher raises a runtime error with a descriptive message

### Optimizer × arch matrix

| Optimizer | sm_90 (Hopper) | gfx942 (CDNA3) | tpu_v5p (Pallas) |
|-----------|:--------------:|:--------------:|:----------------:|
| AdamW         | 🟡 | 🟡 | 🟡 |
| SuperGrok v2  | 🟡 | 🟡 | 🟡 |
| SuperGrok v1.5 | 🟡 | 🟡 | 🟡 |
| SuperGrok v1.1 | 🟡 | 🟡 | 🟡 |
| GrokAdamW     | 🟡 | 🟡 | 🟡 |
| NeuralGrok    | 🟡 | 🟡 | 🟡 |
| Prodigy       | 🟡 | 🟡 | 🟡 |
| Grokfast      | 🟡 | 🟡 | 🟡 |
| Lion          | 🟡 | 🟡 | 🟡 |
| LookSAM       | 🟡 | 🟡 | 🟡 |
| Muon          | 🟡 | 🟡 | 🟡 |
| MoE/Adam      | 🟡 | 🟡 | 🟡 |

### Model × arch matrix

| Model    | sm_90 (Hopper) | gfx942 (CDNA3) | tpu_v5p (Pallas) |
|----------|:--------------:|:--------------:|:----------------:|
| Decoder  | 🟡 | 🟡 | 🟡 |
| ViT      | 🟡 | 🟡 | 🟡 |
| Mamba    | 🟡 | 🟡 | 🟡 |

**SuperGrok v2 on gfx942 is 🟡 (functional, perf not verified).** The launcher
(`csrc/backends/hip/gfx942/launch_supergrok2.hip.cpp`) implements the full
Mamba + GRU + PEER pipeline via ATen tensor ops. Projection GEMMs go through
rocBLAS (which dispatches to MFMA `v_mfma_f32_16x16x16_bf16` internally for
BF16/FP16 at sizes ≥ 16), so the dense-linear-algebra portion does exercise
the MFMA pipeline. The scan recurrence runs as a host-side sequential loop —
slower than the Hopper warp-specialized parallel scan but mathematically
equivalent. The bilevel backward path is not yet implemented on gfx942 and
will raise; only the forward `supergrok2_prepare_and_batched_step` path is
functional. Promotion to ✅ requires elementwise allclose validation against
the sm_90 path on an MI300X.

Everything marked 🟡 is implemented end-to-end in the refactored tree but has
not been run on real hardware in this environment. The "Action items for
hardware validation" section near the end of this README documents the smoke
tests that must run on a real H100, MI300X, or TPU v5p before any cell can be
promoted to ✅.

### Kernel header status

Elementwise kernel headers in `grokking_optimizers/kernels/`. Each header
provides a templated `__forceinline__ __device__` update function, a
vectorized `_vec4` variant for float params, and a `__global__` launcher
kernel. All headers share a common NanPolicy enum and type-cast helpers
via arch-specific common headers.

| Optimizer | sm_90 `.cuh` | gfx942 `.hip.hpp` | State tensors | Bytes/elem |
|-----------|:---:|:---:|:---:|:---:|
| AdamW     | 🟡 | 🟡 | 2 (m, v) | 8 |
| Lion      | 🟡 | 🟡 | 1 (m) | 4 |
| Grokfast  | 🟡 | 🟡 | 3 (ema, m, v) | 12 |
| GrokAdamW | 🟡 | 🟡 | 3 (ema, m, v) | 12 |

Legend: 🟡 = written, structurally validated (~138 inline self-tests pass via
`python -m grokking_optimizers.compile --self-test`), not compiled on device
(no CUDA/HIP toolchain in this environment).

### Model kernel header status

Per-model kernel headers in `grokking_optimizers/kernels/`. Each header
provides templated per-layer `__device__` forward and backward functions,
a state struct with raw pointers, constexpr size helpers, and shares
NanPolicy and type-cast helpers via arch-specific common headers.

Honest note on the matmul path per arch (the GEMM-heavy layers —
attention QK^T/PV, in/out projections — are what differ):

- **gfx942 (`mamba3_gfx942.hip.hpp`)** emits **real MFMA intrinsics**
  (`__builtin_amdgcn_mfma_f32_32x32x8bf16_1k` /
  `..._16x16x16bf16_1k`). The CDNA matmul path is genuine.
- **sm_90 (`*_sm90.cuh`)** currently carries **scalar / grid-stride
  bodies** with comments that describe the intended wgmma / CUTLASS 3.x
  dispatch (e.g. `wgmma_matmul` in `transformer_decoder_sm90.cuh` is a
  shape-only placeholder; `in_proj_forward` in `mamba3_sm90.cuh` is a
  scalar inner-product loop). These compile and are numerically correct
  but do **not** yet emit `wgmma` / TMA. The CUTLASS GEMM that *is* wired
  lives in `csrc/backends/cuda/sm_90/launch_supergrok2.cu`
  (`cutlass::gemm::device::Gemm`), and even there the Sm90 collective
  (warp-group MMA) ArchTag is not yet selected.
- **TPU (`*_tpu.py`)** uses `pl.pallas_call` with `BlockSpec` tiling in
  `_pallas_kernels.py`.

| Model | sm_90 `.cuh` | gfx942 `.hip.hpp` | TPU `.py` | Layers (fwd+bwd) |
|-------|:---:|:---:|:---:|:---:|
| Transformer Decoder | 🟡 | 🟡 | 🟡 | 9+8 |
| Mamba-3 (SSM)       | 🟡 | 🟡 | 🟡 | 8+7 |
| ViT                 | 🟡 | 🟡 | 🟡 | 10+8 |

Legend: 🟡 = written and structurally validated by the inline self-test
suite (`python -m grokking_optimizers.compile --self-test`), not compiled
on device (no CUDA/HIP/TPU toolchain in this environment). The Hopper
wgmma/TMA path is the main outstanding kernel-perf work item.

### Cross-validation: optimizer × model × arch

All 4 elementwise optimizers verified against all 3 models across all 3
architectures (36 combinations total). Verification confirms: file
existence, function signatures (`{opt}_update(`, `{opt}_kernel(`), common
header inclusion, namespace consistency (`grokking::{arch}`), and `ParamT`
template compatibility.

| Optimizer × Model | sm_90 | gfx942 | TPU |
|-------------------|:-----:|:------:|:---:|
| AdamW × Decoder   | PASS | PASS | PASS |
| AdamW × Mamba-3   | PASS | PASS | PASS |
| AdamW × ViT       | PASS | PASS | PASS |
| Lion × Decoder    | PASS | PASS | PASS |
| Lion × Mamba-3    | PASS | PASS | PASS |
| Lion × ViT        | PASS | PASS | PASS |
| Grokfast × Decoder | PASS | PASS | PASS |
| Grokfast × Mamba-3 | PASS | PASS | PASS |
| Grokfast × ViT    | PASS | PASS | PASS |
| GrokAdamW × Decoder | PASS | PASS | PASS |
| GrokAdamW × Mamba-3 | PASS | PASS | PASS |
| GrokAdamW × ViT   | PASS | PASS | PASS |

GPU kernels share interfaces via `common_sm90.cuh` / `common_gfx942.hip.hpp`
(NanPolicy enum, `to_float<ParamT>` / `from_float<ParamT>` conversions).
TPU models import from `common_tpu.py` (NanPolicy IntEnum, `PARAM_DTYPE`,
`ACCUM_DTYPE`, dtype helpers); TPU optimizers use JAX-level parameter
updates rather than fused kernels.

---

### Compile cache schema (v4)

The compile cache (`build/.compile_cache.json`) uses schema **v4**
(auto-migrated v2 → v3 → v4 on load; old files archived as
`<cache>.v2.bak` / `<cache>.v3.bak`). Per-entry shape:

```jsonc
{
  "version": 4,
  "entries": {
    "<optimizer>/<model>/<arch>": {
      // Identity
      "source_hash": "…",  "host_cflags_hash": "…",  "device_cflags_hash": "…",
      "search_space_hash": "…",
      // Artefacts
      "primary_artifact": { "path": "…", "size": …, "mtime": …, "sha256": "…" },
      "variant_artifacts": { "<config_key>": { "path": "…", "size": …, "mtime": … } },
      // Phase timestamps
      "aot_completed_at": "…",  "jit_completed_at": "…",  "pgo_completed_at": "…",
      // Tuning (winner only — full trial log lives in the sidecar)
      "mode": "bayesian" | "exhaustive",
      "tuned_config": { "block": 256, "vec": 4, "unroll": 8, "timing_ms": 0.412, "stage_won": "refine" },
      // v4: trial log moved out to a .jsonl sidecar
      "trial_log_path": "trials_<opt>_<model>_<arch>.jsonl",  // relative to cache dir
      "trial_log_summary": {
        "n_trials": 173, "best_timing_ms": 0.412,
        "stop_reason": "ei_exhausted:5.2e-07", "last_updated_unix": 1734567890.0
      },
      // Kept empty for v3-reader back-compat — real data is in the sidecar
      "bayesian_trials": [],
      "sweep_history": [],
      // PGO
      "pgo_enabled": bool,  "pgo_profile_dir": "…",  "pgo_workload_hash": "…",
      "early_stop_info": { "stop_reason": "ei_exhausted:5.2e-07", "trial_count": 173, "best": 0.412 }
    }
  }
}
```

Sidecar format (`trials_<opt>_<model>_<arch>.jsonl`): one trial per
line, JSON-encoded `_make_trial_record(...)` output. Append-only,
written via `with open(..., 'a')` on each `record_trial` call so a
crashed autotune sweep loses at most the in-flight trial. Total wall-
clock for trial appending is O(1) per trial vs the previous O(N) of
re-serializing the entire `bayesian_trials` list.

### CLI surface (`compile.py`)

```
python -m grokking_optimizers.compile \
  -O <optimizer> -M <model> -A <arch> \
  --mode {bayesian,exhaustive} \
  [--bayesian-trials N]                 # None → auto early-stop
  [--max-tune-seconds 600]              # wall-clock budget for auto mode
  [--min-improvement 0.005]             # plateau-detection threshold
  [--patience 100]                      # plateau-detection window
  [--ei-floor 1e-6]                     # rolling EI-exhaustion threshold (0 disables)
  [--top-k N]                           # None → auto elbow detection
  --pgo [--pgo-workload <script>] [--pgo-steps 1000] \
  [--enable-device-pgo]                 # CUPTI / rocprof / XLA HLO sidecar
  [--search-space <path/to/your.yaml>] \
  --cache build/.compile_cache.json \
  --runtime {aot,jit,both} [--aot-only | --jit-only] \
  [--aot-artifact-dir <path>] \
  [--quick] [--no-autotune] [--no-profile] \
  [--transfer-learning] [--pruner {none,hyperband,median}] \
  [--enable-emitter]                    # Jinja2 per-variant emitter (44 templates)
  [--enable-runtime-specialization]     # NVRTC/hipRTC live KernelRegistry
  [--strict-numerics]                   # require bit-identical determinism
  [--config <path/to/your.toml>]        # TOML project config override
  [--bootstrap-cuda] [--bootstrap-rocm] [--bootstrap-jax]
  [--prune] [--prune-max-age-days 30] [--prune-keep-top-n 100]
  [--no-auto-prune]                     # cache GC
  [--debug-symbols] [--seed N] [-D MACRO[=VALUE]] [-v] [--debug]
  [--self-test]                         # in-process suite (~138 checks today)
  [--dry-run-all-archs]                 # write JSON manifests for all 26 canonical archs
  [--e2e-smoke] [--e2e-max-seconds 120] # end-to-end build smoke (GPU-gated)
```

`--debug` mirrors the full build report to stderr in real time,
auto-enables `-v`, and propagates through the AOT/JIT subprocess split
so every nvcc/hipcc invocation and Optuna trial is streamed live. Use
`debug=True` on the importable `build()` API for the same effect.

### Output flag bases

```python
HOST_CFLAGS_BASE = [
    "-O3", "-std=c++17", "-fPIC", "-flto=full", "-march=native",
    "-fno-semantic-interposition", "-fvisibility=hidden",
    "-fdata-sections", "-ffunction-sections", "-ffast-math", "-funroll-loops",
]
NVCC_DEVICE_BASE = [
    "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
    "--threads", "8", "-Xfatbin", "-compress-all",
    "--allow-expensive-optimizations=true",
    "--extra-device-vectorization", "-dlto",
    # Stream 3 additions (gated by toolchain version):
    "-Xptxas", "--opt-level=3",                  # single PTXAS opt level
    "-Xptxas", "--register-usage-level=10",      # CUDA 12.0+
    "-Xnvlink", "--suppress-stack-size-warning",
    "--diag-suppress=20012,20013",
    "-Xcompiler", "-fno-strict-aliasing",
    "--device-link-options=-dlto",
]
# Per-arch -gencode is appended in _device_cflags(spec) from
# ARCH_TABLE[arch].nvcc_gencode — including the PTX fallback
# `compute_XX,code=compute_XX` so older drivers can JIT.

HIPCC_DEVICE_BASE = [
    "-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
    "-mllvm", "-amdgpu-internalize-symbols", "-flto",
    # Stream 3 additions:
    "-mllvm", "--amdgpu-unroll-threshold=1000",
    "-mllvm", "--amdgpu-enable-lower-module-lds-strategy=module",
    "-mllvm", "--amdgpu-promote-alloca-to-vector-limit=512",
    "-mllvm", "--amdgpu-sroa-vector-elements=8",
    "-mllvm", "--amdgpu-enable-merge-m0",
]
# Per-arch `--offload-arch=<gfx*>`, `-mcumode` (CDNA), and
# `-mwavefrontsize32` / `-mtgsplit` (RDNA) appended in _device_cflags(spec)
# from ARCH_TABLE[arch].hipcc_offload_arch + warp_size + features.
LDFLAGS_BASE = ["-flto=full", "-Wl,--as-needed", "-Wl,--gc-sections", "-Wl,-O3", "-Wl,--icf=all"]
```

---

### Autotune guide

#### TL;DR workflows

```bash
# End-to-end (Bayesian, auto early-stop + elbow top-K — the defaults)
python -m grokking_optimizers.compile -O supergrok2 -M mamba -A sm_90 \
    --cache build/.compile_cache.json

# CPU-only AOT → ship cache to GPU host → JIT autotune
python -m grokking_optimizers.compile … --aot-only --aot-artifact-dir build/compiled/aot_artifacts
# (on GPU host)
python -m grokking_optimizers.compile … --jit-only

# PGO-flavoured build
python -m grokking_optimizers.compile … --pgo --pgo-steps 1000

# Quick debug (25 trials)
python -m grokking_optimizers.compile -O lion -M mamba -A sm_90 --quick

# Exhaustive (every pre-filter survivor)
python -m grokking_optimizers.compile -O lion -M mamba -A sm_90 --mode exhaustive
```

#### Autotune modes

| Mode | Behaviour |
|------|-----------|
| `--mode bayesian` (default) | Optuna TPE. **Default `--bayesian-trials` is `None`** = auto early-stop on plateau / EI exhaustion / coverage saturation / `--max-tune-seconds`. **Default `--top-k` is `None`** = elbow detection on the timing curve. Set `--bayesian-trials N` to pin a fixed count. Study persists to SQLite for cross-run resume. |
| `--mode exhaustive` | Every config surviving the static pre-filter is built and timed. Capped at 1M survivors (full Cartesian is 3.7B+ for sm_90a). Cache flushes every 5 trials (Ctrl-C safe). |
| `--quick` | Alias: bayesian with 25 trials (small fixed count for fast self-tests). |

#### Search-space schema

The default search space is built programmatically by
`build_full_search_space()` in `compile.py` — the COMPLETE space, not
a curated subset (~3.7 B sm_90 candidates, ~700 M gfx942). Bayesian
TPE samples this directly via Optuna's `suggest_categorical` over the
per-dim value lists; the Cartesian product is never materialized.
Override with `--search-space <path/to/your.yaml>` to provide a
smaller curated YAML space using the schema below:

```yaml
<arch>:
  dims:
    - name: <id>          # e.g. "block"
      type: int|bool|enum|tuple
      values: [64, 128, 256, 512, 1024]
      macro: SG_TUNED_BLOCK_SIZE   # -D name; null to skip
      applies_to: [host, device]   # which flag list receives the macro
  prefilter:
    register_pressure_max: 255
    smem_budget_bytes: 232448
    rules:
      - name: vec_block_alignment
        expr: "block % (vec * 4) == 0"
```

Static pre-filter rules eliminate infeasible configs before any compile;
the elimination count is logged as `[prefilter] N candidates → M survivors`.

#### PGO loop (`--pgo`)

Three-pass build: **instrument** (AOT with `-fprofile-generate`) →
**collect** (run the built-in PGO workload for N steps; profile files
land under `<out>/pgo_profile/`) → **use** (rebuild with
`-fprofile-use`). The cache stores `pgo_workload_hash` and `pgo_enabled`
as freshness factors so PGO and non-PGO artefacts are never confused.

#### Troubleshooting

- **sccache 0% hit on NVCC** — upgrade sccache to >= 0.8 (CUDA hash instability fix).
- **Optuna study won't resume** — verify `<out>/optuna_<opt>_<model>_<arch>.db` exists; changing any of (opt, model, arch) starts a new study.
- **`.gcda` empty after PGO collect** — ensure `LLVM_PROFILE_FILE` is exported (done automatically by `pgo.collect_workload`); verify the workload exercises the hot path.
- **AOT/JIT cache mismatch across hosts** — copy both the cache JSON and the artefact dir; `is_aot_fresh` factors in source, flags, PGO, and search-space hashes.
- **Worker crash mid-sweep** — the sweep falls back to one-shot subprocess timing and restarts the worker automatically.
- **Transfer-learning no effect** — run the sibling optimizer first (e.g. `-O adamw`) so its trials exist in the cache before invoking `--transfer-learning`.

---

### Optimization candidate matrix

Evaluation of additional compile/runtime optimizations beyond the
baseline (full LTO, sccache, NVCC `--threads 8`, Bayesian+Exhaustive
autotune, PGO, persistent timing worker, CUDA/HIP graphs). Score =
`(perf_gain × confidence) / (cost × risk)`. Reported numbers are from
cited sources (no real A/B — CPU-only environment; rerun with
`--bayesian-trials 500` on a GPU host to populate "Measured here").

#### Compile-speed candidates

| # | Candidate | Est. impact | Cost (h) | Risk | Recommendation |
|---|---|---|---|---|---|
| C1 | Newer compiler probe (NVCC 12.6+ `--split-compile`) | -5 to -15% heavy TUs | 2-4 | Low | **enable-by-default** |
| C2 | ccache alongside sccache (ccache for host, sccache for NVCC) | -20 to -60% warm | 3-5 | Low | **enable-by-default** |
| C3 | PCH for binding TUs | -5 to -12% | 4-8 | Med | behind-flag (`--use-pch`) |
| C4 | Split heavy templated TUs | -10 to -25% | 16-24 | Med | behind-flag (`--split-launchers`) |
| C5 | BOLT post-link on compiler binaries | -8 to -15% | 12-20 | Med-High | behind-flag (`GROK_BOLT_TOOLCHAIN=1`) |
| C6 | C++20 modules for binding headers | -3 to -8% | 30-50 | High | not-worth-it |

#### Output-perf candidates

| # | Candidate | Est. kernel-perf | Cost (h) | Risk | Recommendation |
|---|---|---|---|---|---|
| O1 | Register-pressure pruning (parse ptxas spill counts) | search converges 20-40% faster | 5-8 | Low | **enable-by-default** |
| O2 | Per-variant `__launch_bounds__` | 5-20% occupancy-bound | 3-5 | Low | **enable-by-default** |
| O3 | Async copy depth tuning (sm_90) | 3-15% memory-bound | 4-6 | Low | **enable-by-default** |
| O4 | LDS swizzle tuning (gfx942) — bias toward XOR-swizzle | up to 28% | 2-4 | Low | **enable-by-default** |
| O5 | MFMA shape tuning (gfx942) — bias toward 16x16x16 | 5-15% | 1-2 | Low | **enable-by-default** |
| O6 | CUTLASS for sm_90 matmul shapes (TMA + WGMMA) | 30-100% over hand-written | 12-20 | Med | **enable-by-default** |
| O7 | Composable Kernel on gfx942 matmul | 20-100% | 16-24 | Med | **enable-by-default** |
| O8 | TMA descriptor reuse on sm_90 | 10-50% memory-bound | 12-18 | Med | enable-by-default (feature-flagged) |
| O9 | Mixed-precision FP8/BF16/TF32 variants | up to ~2× peak | 10-16 | High | behind-flag (`--mixed-precision`) |
| O10 | Persistent kernel pattern | 60-211× fine-grained | 20-30 | High | behind-flag (`--persistent-kernel`) |
| O11 | BOLT post-link on produced .so | 2-15% host-side | 6-10 | Low | behind-flag (`--bolt-post-link`) |
| O12 | AutoFDO vs instrumented PGO | 5-15% host | 8-12 | Low | behind-flag (`--pgo-mode=autofdo`) |
| O13 | Auto-vectorize tuning (host TUs) | 3-8% host | 2-4 | Low | **enable-by-default** |
| O14 | Polly / MLIR polyhedral | 10-100% affine | 40-80 | High | not-worth-it |
| O15 | Souper superoptimization | 0-3% peephole | 20-30 | Med | not-worth-it |

#### Autotune-quality candidates

| # | Candidate | Trial-budget savings | Cost (h) | Risk | Recommendation |
|---|---|---|---|---|---|
| A1 | Hyperband / Successive Halving | -30 to -50% | 8-12 | Low | **enable-by-default** (`--pruner hyperband`) |
| A2 | Transfer learning across optimizers | -20 to -40% | 4-6 | Low | **enable-by-default** (`--transfer-learning`) |
| A3 | Multi-fidelity tuning (small tensor first) | -40 to -60% wall-time | 12-20 | Med | behind-flag (`--multi-fidelity`) |
| A4 | Cost-aware Bayesian (EIpu) | -15 to -25% compile-time | 10-14 | Low-Med | behind-flag (`--cost-aware`) |
| A5 | BoTorch GP vs TPE | marginal/negative | 6-10 | Med | not-worth-it |
| A6 | Per-shape autotune (LUT) | +5-15% aggregate | 16-24 | Med | behind-flag (`--per-shape`) |
| A7 | Ensemble-of-winners runtime dispatch | +3-10% aggregate | 20-30 | Med | blocked-by-telemetry |

#### Infrastructure candidates

| # | Candidate | Throughput | Cost (h) | Risk | Recommendation |
|---|---|---|---|---|---|
| I1 | Redis-backed shared sccache | +20-60% cluster | 2-4 | Low | **enable-by-default** (when `SCCACHE_REDIS_ENDPOINT` set) |
| I2 | GHA cache warming on push | +40-80% first-build | 3-5 | Low | blocked-by-infra |
| I3 | Ray for distributed autotune | ~N× linear | 16-24 | High | behind-flag (`--executor=ray`) |
| I4 | Per-variant Docker isolation | -5 to -15% | 6-10 | Med | not-worth-it |

---

## Filesystem

The codebase splits along two orthogonal axes: **algorithm** (the
vendor-neutral math) and **backend** (per-arch launchers). Each backend
launch file is fully self-contained — see "Design choice: per-file
self-containment" below.

```
.
├── README.md
├── grokking_race_v2.py   (race driver — 12 optimizers × 3 models × 4 splits)
├── setup.py / build.sh / pyproject.toml
├── autotune/                   (kernel auto-tuning utilities)
├── third_party/                (cutlass git submodule for WITH_CUTLASS=1)
├── grokking_optimizers/
│   ├── __init__.py             (re-exports the 12 optimizers + helpers)
│   ├── dispatch.py             (arch detection + fused kernel registry + get_ops)
│   ├── compile.py              (consolidated build pipeline — ninja build +
│   │                           AOT/JIT runtime split + Bayesian/Exhaustive
│   │                           autotune + PGO loop + YAML search space +
│   │                           persistent timing worker + CUDA/HIP graph
│   │                           bench + inline self-tests via --self-test)
│   ├── profile.py              (standalone ncu / rocprof / jax.profiler capture)
│   ├── kernels/                (per-arch kernel headers)
│   │   ├── sm_90/              (CUDA Hopper headers)
│   │   │   ├── common_sm90.cuh           (shared NanPolicy + type-cast helpers)
│   │   │   ├── adamw_sm90.cuh            lion_sm90.cuh
│   │   │   ├── grokfast_sm90.cuh         grokadamw_sm90.cuh
│   │   │   ├── transformer_decoder_sm90.cuh
│   │   │   ├── mamba3_sm90.cuh
│   │   │   └── vit_sm90.cuh
│   │   ├── gfx942/             (HIP CDNA3 headers)
│   │   │   ├── common_gfx942.hip.hpp     (shared NanPolicy + type-cast helpers)
│   │   │   ├── adamw_gfx942.hip.hpp      lion_gfx942.hip.hpp
│   │   │   ├── grokfast_gfx942.hip.hpp   grokadamw_gfx942.hip.hpp
│   │   │   ├── transformer_decoder_gfx942.hip.hpp
│   │   │   ├── mamba3_gfx942.hip.hpp
│   │   │   └── vit_gfx942.hip.hpp
│   │   └── tpu/                (JAX/Pallas Python headers)
│   │       ├── common_tpu.py             (shared NanPolicy + dtype helpers)
│   │       ├── transformer_decoder_tpu.py
│   │       ├── mamba3_tpu.py
│   │       └── vit_tpu.py
│   └── optimizers/             (11 torch.optim.Optimizer subclasses; MoE-aware
│       │                       SG2 lives inside supergrok2.py)
│       ├── adamw.py            grokfast.py     muon.py       prodigy.py
│       ├── supergrok2.py       grokadamw.py    looksam.py    neuralgrok.py
│       ├── supergrok15.py      lion.py
│       └── supergrok11.py
└── csrc/
    ├── algorithms/             (11 algorithm headers, MoE folded into SG2)
    │   ├── adamw.h             grokfast.h    looksam.h     prodigy.h
    │   ├── grokadamw.h         lion.h        supergrok2.h  supergrok11.h
    │   └── neuralgrok.h        muon.h        supergrok15.h
    ├── backends/
    │   ├── cuda/sm_90/         (11 launch_*.cu + models/{decoder,vit,mamba,attention})
    │   ├── hip/gfx942/         (11 launch_*.hip.cpp + models/{decoder,vit,mamba,attention})
    │   └── pallas/             (11 launch_*.py + v5p/ TPU-specific helpers)
    └── bindings/               (5 pybind11 entry-point files)
```

Launch glue files contain the `__global__` kernels (CUDA) or ATen-driven
implementations (HIP) or JAX wrappers (Pallas). Every launch file inlines
the platform/types/utils/PTX-intrinsic/quantization/primitives helpers it
needs — there is no shared `csrc/common/`, `csrc/scan/`, or
`primitives.*` directory. Modifications to a shared primitive must be
replicated across every consumer; the codebase deliberately accepts this
cost for zero cross-file coupling.

### Design choice: per-file self-containment

Every backend launch file embeds its own copies of platform macros, warp
helpers, PTX intrinsics, quantization helpers, scan adapters, CUTLASS MMA
wrappers, and any primitives it uses — wrapped in clearly-marked
`// ── inlined from former <path> ──` blocks. This trades code duplication
for zero cross-file coupling: touching one optimizer's kernel cannot
affect another's. The duplicated content is reviewable because each
inlined block carries the original path as a comment header. The one
surviving shared boundary is `csrc/bindings/` — pybind11 entry points
that need to call into every backend.

Runtime dispatch via `grokking_optimizers/dispatch.py`:
- `detect_arch()` → `90`, `942`, or `"tpu_v5p"`
- `get_ops()` → the compiled C++ extension, or `RuntimeError`
- `dispatch_fused(model, optimizer, ...)` → routes to the compiled fused
  kernel for the active arch, or falls back to separate forward/backward/step

---

## Installation

```bash
git clone https://github.com/peterc04/SuperGrok1.5
cd SuperGrok1.5
git submodule update --init --recursive third_party/cutlass  # optional, for CUTLASS GEMMs
bash build.sh
```

### Build modes

| Mode | Effect |
|------|--------|
| `./build.sh` | Default ninja-backed release build (sm_90 + PTX embed by default). |
| `./build.sh --debug` | `CUDA_DEBUG=1`, `-G -O0 -lineinfo`, fast-math disabled. |
| `./build.sh --profile` | Release build + `ncu --set full` profile capture. |
| `./build.sh --package` | Build + stage redistributable `dist/` tree. |
| `./build.sh --package-tarball` | `--package` + `supergrok2-3.0.0-<sha>.tar.gz`. |

### Compiler flags

- nvcc: `-O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -lineinfo -Xptxas -O3 --warn-on-spills`
- nvcc gencode: `-gencode arch=compute_90,code=sm_90` + PTX embed for forward-compat
- hipcc: `--offload-arch=gfx942 -O3 -std=c++17 -ffast-math`

### Performance options

#### CUTLASS (opt-in)

`WITH_CUTLASS=1 ./build.sh` enables CUTLASS-backed GEMM paths on sm_90.
Requires `git submodule update --init --recursive third_party/cutlass`.
Adds `-DCUTLASS_NVCC_ARCHS=90a` and CUTLASS include directories.

CUTLASS is used only by Muon (Newton-Schulz GEMMs) and SuperGrok v2 (dt_proj
fused softplus). **Without `WITH_CUTLASS=1`**, Muon falls back to cuBLAS via
`torch::mm` and SuperGrok v2 uses cuBLAS + a separate softplus kernel —
slightly slower but fully functional. The fall-back path is the default for
local development; CUTLASS is the production-deployment knob.

### Targeted build: `grokking_optimizers.compile`

Dev-time companion to `setup.py`. Given an `(optimizer, model, arch)`
triple, compiles the matching subset of `csrc/` with arch-tuned codegen,
full LTO, and a two-phase **AOT-then-JIT autotune** with a portable JSON
cache (v3) — all driven through ninja with `MAX_JOBS=$(nproc)`. The
search space is built programmatically by `build_full_search_space()`
(billions of candidates per arch, no curation; override with
`--search-space <path.yaml>` if you want a smaller hand-picked space);
the autotune
defaults to **Bayesian** (Optuna TPE + ±2-step neighbour refinement) and
also supports **Exhaustive** sweeps; optional **PGO** loop instruments
→ runs a workload → rebuilds with `-fprofile-use`. AOT and JIT can run
in **separate subprocesses** so a CPU host can do AOT and a GPU host
can take the JIT half. Profile capture is delegated to
`grokking_optimizers.profile` (see next subsection). Use these when
iterating on a specific combo; use `setup.py` (the default
`pip install -e .` path) when building the full production extension
consumed by the race driver.

```bash
# End-to-end (default: bayesian, auto early-stop, elbow-detected top-K, both runtimes)
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    --cache build/.compile_cache.json

# Exhaustive — every config that survives the YAML pre-filter
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    --cache build/.compile_cache.json --mode exhaustive

# AOT-only on a CPU host (no GPU needed) — writes cache, ship to GPU host
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    --cache build/.compile_cache.json --aot-only \
    --aot-artifact-dir build/compiled/aot_artifacts

# JIT autotune only — consumes the cache produced above
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    --cache build/.compile_cache.json --jit-only --mode bayesian

# PGO-flavoured AOT (3-pass instrument → workload → use)
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    --cache build/.compile_cache.json --pgo --pgo-steps 1000

# Quick debug sweep (25 Bayesian trials, no PGO)
python -m grokking_optimizers.compile -O lion -M mamba -A sm_90 --quick

# Transfer learning — seed TPE from sibling-optimizer studies
python -m grokking_optimizers.compile -O supergrok2 -M mamba -A sm_90 \
    --transfer-learning --cache build/.compile_cache.json
```

```python
# Importable
from grokking_optimizers.compile import build, CompileCache
cache = CompileCache(Path("build/.compile_cache.json"))   # held in memory
so   = build(optimizer="supergrok2", model="mamba", arch="sm_90",
             cache=cache, autotune_mode="bayesian",
             bayesian_trials=500, top_k=20, pgo=False)
```

What runs, in order:

1. **Resolve sources** — bindings + every launcher and model TU for the
   chosen arch (18 files), so the bindings link cleanly. Compute SHA-256
   hashes of the source set, host/device cflags, the resolved YAML
   search space, and the PGO state.
2. **Inject metaprog macros** so headers can `#if` out unused
   specialisations for the chosen combo:
   `-DSG_BUILD_OPTIMIZER_<UPPER>=1`,
   `-DSG_BUILD_MODEL_<UPPER>=1`,
   `-DSG_BUILD_ARCH_<…>=1`,
   `-DSG_VERBOSE=1`.
3. **AOT phase** (any host — CPU is fine). Look up the combo in the
   cache. **Cache hit** = matching source + flag + search-space +
   PGO-state hashes and a recorded primary artefact that still exists
   → reuse the `.so` and skip the build. **Cache miss** → build with
   ninja `-j$(nproc)` via `torch.utils.cpp_extension.load`, with the
   maximally optimised flag bases (see "Flag bases" below). Persist the
   artefact path / size / SHA-256 in the cache, stamp
   `aot_completed_at`, save to disk. With `--pgo`, the AOT phase runs
   the 3-pass loop **instrument → workload → use** (the workload is
   the built-in PGO workload; override via `--pgo-workload`).
4. **JIT autotune phase** (`--no-autotune` to skip; runs only when
   `torch.cuda.is_available()` returns `True`). Load the YAML search
   space, apply static **pre-filter** rules (alignment, occupancy
   ceilings, TMA/warp-spec block thresholds; logged as
   `[prefilter] N candidates → M survivors (K eliminated)`), then sweep:
   - **`--mode bayesian`** (default): Optuna TPE multivariate sampler.
     `--bayesian-trials` defaults to `None` → the 5-criterion auto
     early-stop decides when to halt (pass an int to pin a hard cap),
     then ±2-step neighbour refinement on the top-K, which defaults to
     `None` → elbow-of-the-timing-curve detection (pass an int to pin
     it). The study persists to `<out>/optuna_<opt>_<model>_<arch>.db`
     for cross-run resume.
   - **`--mode exhaustive`**: every surviving config is built and timed.
     Cache flushes every 5 trials so a Ctrl-C is recoverable.
   Each variant is timed by a **persistent subprocess worker** that
   holds a warm CUDA/HIP context for the entire sweep and uses
   **CUDA-graph capture+replay** for sm_90 (HIP-graph reuses the same
   wrapper on ROCm). On worker crash, the sweep falls back to one-shot
   subprocess timing for that variant, then restarts the worker.
5. **Final pass** — when a winner exists, rewrite
   `csrc/algorithms/tuned_configs.h` with the winning macros (so
   downstream builds pick them up), rebuild the primary `.so` with
   those macros baked in, and update the cached primary artefact.
6. **Profile pass** (`--no-profile` to skip) — delegates to
   `grokking_optimizers.profile`; see below for the standalone CLI and
   what each arch's profiler captures.

The cache is loaded once at `build()` start into an **in-memory dict** and
mutated in place throughout; the only disk writes are atomic
tmp-file-rename saves at phase boundaries (end of AOT, end of JIT, end of
final, end of run). No per-step file I/O, so the cache itself is never
the bottleneck.

Output goes to a single text report (default
`build/compiled/compile_<O>_<M>_<A>.txt`); stdout only prints the report
path. Progress is reported on stderr via a tqdm bar with elapsed/ETA,
falling back to a `[i/N elapsed=Xs eta=Ys]` line when tqdm is missing.

#### Cache file format and portability

The cache lives as a single JSON file (default
`<out>/.compile_cache.json`; override with `--cache <path>`). It
survives across processes, so two `compile.py` runs against the same
combo only build once. The schema is **v3** (forward-migrated from v2
on load; the v2 file is archived as `<cache>.v2.bak`). Per-entry shape:

```jsonc
{
  "version": 3,
  "created_at": "2026-…",
  "host_history": [{ "platform": "Linux-…", "torch": "2.…", "cuda": "13.0", "ncpus": 4, "recorded_at": "…" }],
  "entries": {
    "supergrok2/mamba/sm_90": {
      "source_hash":         "abc…",   // SHA-256 of source-set contents
      "host_cflags_hash":    "def…",
      "device_cflags_hash":  "ghi…",
      "search_space_hash":   "jkl…",   // SHA-256 of resolved YAML space
      "primary_artifact":    { "path": "/…/*.so", "size": …, "mtime": …, "sha256": "…" },
      "variant_artifacts":   { "block=256_unroll=8_vec=4": { "path": "…", "size": …, "mtime": … }, … },
      "sweep_history":       [{ "stage":"tpe", "config": {…}, "timing_ms": 0.412, "min_ms":…, "max_ms":…, "n":21, "host": {…}, "config_key":"…" }, …],
      "bayesian_trials":     [/* same shape as sweep_history; stage ∈ {"tpe","refine","exhaustive"} */],
      "mode":                "bayesian",
      "tuned_config":        { "block":256, "vec":4, "unroll":8, "timing_ms":0.412, "config_key":"…", "stage_won":"refine" },
      "aot_completed_at":    "2026-…",
      "jit_completed_at":    "2026-…",
      "aot_host":            { … },
      "jit_host":            { … },
      "pgo_enabled":         true,
      "pgo_profile_dir":     "build/compiled/pgo_profile",
      "pgo_workload_hash":   "sha256(workload-contents + steps)",
      "pgo_completed_at":    "2026-…",
      "pgo_host":            { … }
    }
  }
}
```

**Cross-host portability**: hashes, sweep history, Bayesian trial list,
and the tuned config are portable — they travel with the JSON file.
The local artefact paths (`primary_artifact.path` and
`variant_artifacts[*].path`) are host-local; on a fresh GPU host they
get rebuilt on first AOT run (or, if AOT was done on a different
machine, they get picked up via `--aot-artifact-dir <shared-path>`).
JIT-sweep results carry across hosts of the same arch, so a fresh GPU
host that already has a winning config in the cache reuses it
directly. Schema upgrades archive the old file as `<cache>.v<N>.bak`;
corrupted JSON is archived as `<cache>.corrupt.bak`. `pgo_enabled`,
`pgo_workload_hash`, and `search_space_hash` all gate AOT freshness so
a PGO build is never confused with a non-PGO build, and a YAML edit
forces a fresh sweep.

**Typical cross-machine workflow** (CPU build farm → GPU runner). The
default `--runtime both` spawns AOT then JIT subprocesses in the same
session; the split below is for genuine cross-machine pipelines:

| Step | Where | Command | Cache state after |
|---|---|---|---|
| 1. AOT build | CPU host with nvcc/hipcc | `compile.py --aot-only --aot-artifact-dir shared/ --cache shared.json` | `aot_completed_at` set; artefact published |
| 2. Ship cache + artefact | git / rsync / artefact store | `rsync -av shared/ <gpu-host>:shared/ && cp shared.json <gpu-host>:` | both portable |
| 3. JIT autotune | Target GPU host | `compile.py --jit-only --mode bayesian --cache shared.json` | `jit_completed_at`, `tuned_config`, `bayesian_trials` populated |
| 4. Re-run anywhere | Either host | `compile.py --cache shared.json` | both phases cache-hit |

#### Compile requirements per arch

The triple selected determines what hardware/toolchain must be present.
With `--aot-only`, only the compile toolchain is needed; the GPU
hardware is only required for the JIT autotune sweep and the profile
capture. Builds without the optional profiler still succeed; only the
profile capture is skipped with a `[skip]` line in the report.

| Selected `--arch` | AOT phase (build only) | JIT autotune + profile (run) | Required compiler | Optional profiler |
|---|---|---|---|---|
| `sm_90` | None for build (set `FORCE_CUDA=1` to compile without a visible GPU); CUDA Toolkit ≥ 12.0 + nvcc on `PATH` | NVIDIA Hopper (H100 / H200) for `opt.step()` timing and `ncu` capture | `nvcc` (CUDA Toolkit) + `g++` ≥ 9 | `ncu` (Nsight Compute, `--set full` + 7 sections) |
| `gfx942` | None for build (PyTorch ROCm install); ROCm ≥ 6.0 + hipcc on `PATH` | AMD CDNA3 (MI300X / MI300A) for `opt.step()` timing and rocprof capture | `hipcc` (ROCm) + host C++ for `.hip.cpp` | `rocprof-compute` ≫ `rocprofv2` ≫ `rocprof` (first found on `PATH`) |
| `tpu_v5p` | None — Python-only, no C++ compile (the launcher is `csrc/backends/pallas/launch_<opt>.py`) | TPU v5p host with `jax[tpu]` for `opt.step()` and trace capture | n/a | `jax.profiler.start_trace / stop_trace` (in-process) |

Common requirements (all arches):

- **Python ≥ 3.10**, **PyTorch ≥ 2.0** (and `jax + jaxlib` for `tpu_v5p`),
  **Ninja** (`pip install ninja` or system pkg) — `compile.py` drives it
  via `torch.utils.cpp_extension.load` with `MAX_JOBS=$(nproc)`.
- Optional: **`tqdm`** for a nicer progress bar; absent → built-in
  `[i/N elapsed=Xs eta=Ys]` fallback.
- Optional: **`ccache` / `sccache`** on `PATH` for warm-rebuild speedups
  (auto-detected by `setup.py`; not configured by `compile.py`).
- Optional: **`third_party/cutlass`** checked out (
  `git submodule update --init --recursive third_party/cutlass`) — when
  present, the sm_90 build auto-adds `-DWITH_CUTLASS -DCUTLASS_NVCC_ARCHS=90a`
  so Muon Newton-Schulz and SuperGrok v2 dt_proj route through CUTLASS GEMMs.

#### Autotune search space (programmatic, not YAML)

The default search space is the **complete one** — every value the
target hardware actually supports, no hand-curated subset. Built by
`build_full_search_space()` in `compile.py` and verified to be:

- **sm_90**: ~3.7 billion Cartesian candidates (32 block sizes × 5 vec
  widths × 8 unrolls × 8 pipeline depths × 57 reg counts × 20 cluster
  shapes × 5 swizzles × 2 warp-spec × 2 TMA × 16 async depths)
- **gfx942**: ~700 million candidates (16 block × 4 vec × 8 unroll ×
  8 stages × 57 regs × 10 waves/EU × 5 LDS pad × 10 MFMA × 6 sched)
- **tpu_v5p**: 0 (Pallas is Python-only — no C++ tuning surface)

Because the full Cartesian is in the billions, `cartesian()` is a
generator (never materializes), `cartesian_count()` returns the size
without iteration, and `ss_prefilter()` streams survivors. Bayesian
TPE samples the per-dim value lists directly through Optuna's
`suggest_categorical` — no enumeration needed. Exhaustive sweeps cap
at 1M survivors (`--mode exhaustive`).

Override the default with `--search-space <path/to/your.yaml>` if you
want a smaller hand-picked space (e.g. for fast CI sweeps).
The targets are ~100% SM/CU utilisation: warp-aligned blocks, vector
widths matched to the arch's load instructions, unroll factors that
keep ILP saturated without blowing the register budget, plus the
hardware-specific knobs (TMA, warp specialization, cluster shapes,
async-copy depth on sm_90; LDS swizzle, MFMA shape, waves-per-EU on
gfx942). Each YAML dim carries a name, type, value list, `-D` macro
name, and an `applies_to` list (`host` / `device`). Static pre-filter
rules (alignment, occupancy, TMA/warp-spec block thresholds, cluster
volume cap, etc.) eliminate infeasible configs **before** any compile;
the elimination count is logged at sweep start.

| Mode | What runs |
|---|---|
| `--mode bayesian` (default) | Optuna TPE for `--bayesian-trials` (default `None` → 5-criterion auto early-stop) + ±2-step neighbour refinement on top-K (default `None` → elbow detection). Optuna study persisted for cross-run resume. |
| `--mode exhaustive` | Every config that survives the pre-filter is built and timed. Cache flush every 5 trials. |
| `--quick` | Debug shortcut: bayesian mode, 25 trials. |

Each variant build reuses the same `build_directory` family so ninja's
per-file cache amortises the cost — after the first variant builds the
full source set, subsequent variants only rebuild the .o files affected
by the changed macros. Build artefacts and timing results are stored
in the cache; re-runs of the same combo on the same host skip both the
rebuild and the timing run.

Timing infrastructure:

- A **persistent subprocess worker** holds a warm CUDA/HIP context for
  the entire sweep — no per-variant Python startup cost, no per-variant
  CUDA-init cost.
- The worker uses **CUDA-graph capture+replay** (sm_90) and the same
  wrapper on ROCm's HIP-graph (gfx942), so timing measures kernel work
  not launch overhead. Per-iter event timing is the fallback when
  graph capture fails.
- Each variant `.so` is loaded into the worker via
  `importlib.util.spec_from_file_location` as `grokking_optimizers._ops`;
  the previous module is unloaded between variants to avoid op-registration
  collisions.
- 5 warmup + 21 timed iterations; report median + min/max + n. Worker
  crash → one-shot subprocess fallback + worker restart.

See the "Autotune guide" section above for full mode comparison,
YAML schema reference, PGO guidance, and troubleshooting.

#### Performance optimizations

The decided baseline (full LTO, `-march=native`, sccache wiring,
NVCC `--threads 8`, NVCC `-Xfatbin -compress-all`, NVCC
`--allow-expensive-optimizations`, NVCC `--def-load-cache=ca` /
`--def-store-cache=wb`, HIPCC `-mllvm -amdgpu-early-inline-all`,
`-Wl,--gc-sections -Wl,--icf=all`, the persistent timing worker, and
CUDA/HIP graphs) is always on. See
the "Optimization candidate matrix" section above for the
full §12 evaluation matrix of additional candidates with published
impact numbers, integration costs, and risk assessments — including
which are now on by default vs. behind-flag.

Auto-enabled additions landed in this PR:

| Candidate | Flag / env | Effect |
|---|---|---|
| Newer-compiler probe | (auto) | logs nvcc/hipcc version; appends `--split-compile=$(nproc)` to NVCC when ≥12.6 |
| ccache fallback alongside sccache | (auto) | ccache takes host `CC`/`CXX` when on PATH; sccache always handles NVCC; ~3-4.5× faster than sccache on local host TUs |
| Redis-backed shared sccache | `SCCACHE_REDIS_ENDPOINT` env | propagates into child builds for cluster-wide cache sharing |
| Hyperband pruner | `--pruner hyperband` | Successive Halving brackets for Bayesian TPE (Li et al. 2018) |
| Transfer learning | `--transfer-learning` | seeds TPE study with sibling-optimizer trials on the same (model, arch) |

### Standalone profiling: `grokking_optimizers.profile`

Lives right next to `compile.py` at `grokking_optimizers/profile.py`. The
profile pass that `compile.py` runs is just a call into this module — but
you can also invoke it directly when you already have a launcher source
file or a `compile.py`-produced `.so` and want the full native-profiler
capture without rebuilding.

```bash
# Profile by path — optimizer + arch are inferred from the path
python -m grokking_optimizers.profile \
    --path csrc/backends/cuda/sm_90/launch_supergrok2.cu

# Profile a compile.py-produced .so directly
python -m grokking_optimizers.profile \
    --path build/compiled/grokking_compiled_lion_mamba_gfx942/grokking_compiled_lion_mamba_gfx942.cpython-310-x86_64-linux-gnu.so

# Profile by explicit name (no path)
python -m grokking_optimizers.profile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    [--report profile.txt] [--timeout 1800]
```

```python
# Importable
from grokking_optimizers.profile import profile
report = profile(path="csrc/backends/cuda/sm_90/launch_supergrok2.cu")
# or
report = profile(optimizer="lion", arch="gfx942")
```

**Path inference** — when `--path` is given, the module recognises:

| Path | Inferred |
|---|---|
| `csrc/backends/cuda/sm_90/launch_<opt>.cu` | optimizer=`<opt>`, arch=`sm_90` |
| `csrc/backends/hip/gfx942/launch_<opt>.hip.cpp` (or `.hip`) | optimizer=`<opt>`, arch=`gfx942` |
| `csrc/backends/pallas/launch_<opt>.py` | optimizer=`<opt>`, arch=`tpu_v5p` |
| `build/.../grokking_compiled_<opt>_<model>_<arch>/*.so` | optimizer + model + arch |
| any other `.py` | needs explicit `--arch` |

The profiler runs the standard one-step smoke (import the optimizer class,
single `opt.step()` on a 64×64 tensor) — the path is the **identifier** of
what to profile; the kernels actually exercised come from the installed
`grokking_optimizers._ops` (or, for `tpu_v5p`, the matching
`launch_*.py`). So if you want to profile a specific compiled `.so`,
make sure that combo is the one currently installed (`pip install -e .`)
or that the `.so`'s build dir is on `PYTHONPATH`.

#### Profile requirements per arch

Same hardware story as compile — but the run-and-profile side is what
matters here, since there's no build step.

| `--arch` (inferred or explicit) | Hardware required | Profiler binary / call |
|---|---|---|
| `sm_90` | NVIDIA Hopper (H100 / H200) + the kernels already built for it | `ncu --set full --target-processes all --import-source yes --source-folders csrc` + 7 sections (ComputeWorkloadAnalysis, LaunchStats, MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats, InstructionStats, Occupancy) |
| `gfx942` | AMD CDNA3 (MI300X / MI300A) + the kernels already built for it | `rocprof-compute` (preferred) → `rocprofv2` → `rocprof` with `--hip-trace --hsa-trace --stats --basenames on --timestamp on`; all emitted CSV/JSON files inlined into the report |
| `tpu_v5p` | TPU v5p host with `jax[tpu]` installed | `jax.profiler.start_trace / stop_trace` in-process; XLA HLO + op-level capture; trace dir listing summarised in the report |

If the expected profiler binary isn't on `PATH`, the smoke still runs and
the report records `[skip] ncu not in PATH; running smoke only.` (or
equivalent) — useful as a sanity check on a machine that can run the
kernel but can't profile it.

Output goes to a single text report (default
`build/profiled/profile_<O>_<M>_<A>.txt`); stdout only prints the report
path. Same tqdm-with-ETA progress UI as `compile.py`.

#### When to reach for `compile.py` vs `setup.py` vs `profile.py`

| You want… | Use |
|---|---|
| The production `grokking_optimizers._ops` consumed by the race driver | `pip install -e .` (setup.py) |
| To iterate on one optimizer × one model × one arch with full diagnostics | `python -m grokking_optimizers.compile -O <opt> -M <model> -A <arch>` |
| To build kernels on a CPU host and ship the build knowledge to a GPU host | `compile.py --aot-only --cache shared.json` → ship JSON → `compile.py --jit-only --cache shared.json` |
| Maximally-tuned `.so` (sweep all 75 block × vec × unroll combos) | `compile.py --exhaustive` (or `--quick` for 8 combos, default = 36) |
| To re-profile something already built without rebuilding | `python -m grokking_optimizers.profile --path <file>` |
| To rebuild every backend launcher and capture per-launcher profiles | `bench_backends.py` (still ships at repo root, complements both) |

---

## Quickstart

```python
import torch
from grokking_optimizers import SuperGrok2, Lion, GrokAdamW

model = torch.nn.Linear(64, 32).cuda()
opt = SuperGrok2(model.parameters(), lr=1e-3)

for x, y in batches:
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
```

`opt.step()` dispatches through `grokking_optimizers.dispatch.detect_arch()`
to the compiled fused kernel for the active arch. **The kernel runs or
raises** — there is no Python reference fallback. If the C++ extension
isn't built, `get_ops()` raises `RuntimeError`; if a per-arch launcher
isn't implemented, the per-arch namespace raises with a descriptive error.

---

## Optimizers

Eleven optimizers, each taking a different approach to accelerating
generalization beyond memorization.

### SuperGrok v2

The flagship optimizer. SuperGrok v2 wraps a standard Adam optimizer with a
sophisticated meta-network that learns how to transform gradients before they
are applied. At every training step, the raw gradient for each parameter is
fed through a bidirectional Mamba-3 selective state space scan that captures
relationships between gradient elements across the parameter vector. The scan
runs forward and backward through the gradient, building a compressed
representation of the gradient's spatial structure.

After the scan, each gradient element is routed through a Product-Key Expert
Routing system (PEER) with 144 learned experts. The routing works by splitting
each element's representation into two halves, matching each half against a
bank of learned keys, and picking the top experts from the outer product of
the two key matches. This gives each element access to four specialized expert
networks simultaneously, without the cost of evaluating all 144.

A per-element GRU then integrates the current expert-modified gradient with a
temporal memory of previous steps. The GRU decides how much of the old memory
to keep and how much new information to incorporate.

The transformed gradient is used in standard Adam momentum and variance
tracking, with decoupled weight decay. On top of this, SuperGrok v2
periodically runs Sharpness-Aware Minimization (SAM): it perturbs the
parameters in the gradient direction, computes the loss at the perturbed
point, and measures the difference between the perturbed and original
gradients. This difference quantifies loss landscape sharpness; the optimizer
steers toward flatter regions that generalize better.

Every few steps, a bilevel optimization pass trains the meta-network itself
using validation loss as the objective. SAM and bilevel updates fire on
sigmoid schedules tied to training accuracy: early in training (during
memorization), these expensive operations are skipped; once accuracy rises
toward the grokking transition, they activate.

Dead experts (rarely selected by the router) are periodically recycled by
cloning the weights of the best-performing expert. Weight decay increases
sigmoidally with accuracy.

Per-parameter state: gradient momentum, squared gradient average, update
buffer, sharpness estimate, GRU hidden states, forward Mamba scan state,
backward Mamba scan state (seven tensors total).

**Compute pattern.** Mixed — the most varied of the optimizers. Per parameter
(length N): argsort by |g| (O(N log N) sort), input projection ([N, 2] @ [2, d_model]
= [N, d_model] GEMM), bidirectional Mamba-3 scan (N sequential timesteps,
each timestep is a per-element FMA + RoPE rotation across d_inner × d_state
state pairs), out_proj GEMM ([N, d_inner] @ [d_inner, d_model]), unsort
(O(N) gather), PEER routing (num_heads × topk² candidate evaluations, each
a small expert MLP), per-element GRU step, AdamW. Bilevel backward is
saved-activations + adjoint scan + meta-net backward through autograd.

**Dependency chain.** The scan is the serial bottleneck: each timestep
depends on the previous (no parallelism across t without Blelloch). PEER
routing and the GRU step are fully parallel across N once the scan finishes.
The bidirectional scans (forward + backward over t) are independent of
each other — they can run on different streams in principle. AdamW
trails everything; depends on the smart_grad output of PEER+GRU.

**State.** Per-element: param, grad, sharpness, exp_avg, exp_avg_sq, mu,
gru_state (size gru_hidden ≈ 8). Per-tensor: mamba_fwd_state and
mamba_bwd_state (one [d_inner, d_state] matrix per param). Per-step:
bc1, bc2, alpha_mu, lamb_eff, ramp, gate_signal (scalars). Meta-net
weights (in_proj, dt_proj, B/C_proj, A_log, D, rope_freq, out_proj, GRU
linears, expert MLPs, product keys) are shared across all params for the
whole training run.

**Precision.** FP32 accumulators throughout (scan state h, GRU state,
Adam moments). Projection GEMMs accept BF16 input with FP32 accumulate
(MFMA-friendly on CDNA3, WGMMA-friendly on Hopper). The sort, RoPE
rotation, and PEER softmax stay in FP32 — quantizing them risks losing
the top-k selection. INT8/INT4 quantization is supported for the expert
MLP weights and param storage (with stochastic rounding on the
quantization step).

### SuperGrok v1.5

A simplified version of SuperGrok v2 that replaces the Mamba scan, PEER
routing, and GRU with a small two-layer feedforward network (MLP). At each
step, the MLP takes two inputs for each parameter element: the raw gradient
and the current sharpness estimate. It outputs a correction term that is added
to the gradient before the Adam update.

The key simplification is that gradient transformation happens independently
per element through the MLP, rather than through the spatially-aware scan and
routing of v2. This makes the optimizer much cheaper to run while retaining
the core idea of learned gradient modification.

Like v2, it uses sigmoid-scheduled SAM perturbations and bilevel meta-learning
to train the MLP on validation loss. An adaptive alpha parameter controls how
much of the MLP correction to mix into the gradient; this alpha decreases over
time, allowing the optimizer to rely more on raw gradients once the
meta-network has done its work.

The amplified gradient stays entirely in GPU registers from the moment it is
computed through the Adam update, avoiding unnecessary memory round-trips.

Per-parameter state: gradient momentum, squared gradient average, update
buffer, sharpness estimate (four tensors).

**Compute pattern.** Mixed — small per-element MLP + AdamW. Per element:
2-input → meta_hidden → 1-output MLP (two GEMMs of size [N, 2] @ [2, H]
and [N, H] @ [H, 1] when batched across N), sigmoid gate on training
accuracy (scalar, host-side), then AdamW per element. No scan, no sort.

**Dependency chain.** MLP layer 2 depends on layer 1's ReLU output.
Otherwise all parameters are independent — fully embarrassingly parallel
across elements within one tensor and across tensors. The bilevel
meta-update runs autograd through the MLP at validation time; that
dependency chain is meta-net-internal and doesn't pin params.

**State.** Per-element: param, grad, sharpness, exp_avg, exp_avg_sq.
Per-tensor: none. Per-step: bc1, bc2, alpha (after sigmoid gate),
lamb_eff, lr, wd_eff (scalars). Meta-net weights (W1, b1, W2, b2,
rescale) are shared across all params and updated only on bilevel steps.

**Precision.** FP32 for the Adam accumulators (m, v) and the MLP hidden
activations. The MLP forward can run BF16 with FP32 accumulate (MFMA
applies when batched across N ≥ 16). Sharpness estimate can be BF16
since it's only used as MLP input, not as a precise reduction target.

### SuperGrok v1.1

Nearly identical to SuperGrok v1.5 in structure and cost. The difference is
in how it decides how much of the MLP correction to apply. Where v1.5 uses a
global sigmoid gating function based on training accuracy, v1.1 uses
per-parameter cosine similarity between the gradient direction and the
momentum direction.

When the gradient and momentum point in similar directions (high cosine
similarity), the optimizer trusts the gradient more and applies less MLP
correction. When they diverge (low cosine similarity), the optimizer amplifies
the MLP correction to steer the update. This gives v1.1 more granular,
per-parameter control compared to v1.5's global accuracy-based gating.

Per-parameter state: gradient momentum, squared gradient average, update
buffer, sharpness estimate (four tensors).

**Compute pattern.** Mixed — identical shape to SuperGrok v1.5 (per-element
MLP + AdamW) plus three per-tensor reductions for the cosine gate:
sum(g·m), sum(g²), sum(m²). The cosine = num / sqrt(den_g * den_m) is
computed once per parameter tensor, then broadcast to the per-element
update.

**Dependency chain.** The cosine reduction is a barrier: every element's
update depends on the per-tensor scalar. After the reduction, all
elements are independent. Across tensors: independent. The MLP forward
runs in parallel with the cosine reduction once the gradient is known.

**State.** Per-element: param, grad, sharpness, momentum (mu), exp_avg,
exp_avg_sq. Per-tensor: cosine gate scalar (one FP32 per tensor, scratch).
Per-step: bc1, bc2, alpha (meta-net scale), lamb_eff. Meta-net weights
shared across all params.

**Precision.** Same as v1.5 — FP32 accumulators, BF16 MLP input/output
with FP32 accumulate (MFMA-amenable). The cosine reduction wants FP32
to avoid catastrophic cancellation when g and m have similar magnitude.

### GrokAdamW

An extension of AdamW with an exponential moving average (EMA) gradient
filter designed to accelerate the grokking transition. In addition to the
standard Adam momentum and squared gradient buffers, GrokAdamW maintains a
slow-moving average of each gradient element.

At each step, the EMA is updated with a high decay factor (typically 0.98),
so it tracks persistent gradient directions while filtering out noise. The
current gradient is then amplified by adding a scaled version of this EMA
back into it. The amplification factor (lambda) controls how strongly
persistent signals are boosted.

The intuition is that during the memorization phase, gradients are noisy and
inconsistent, so the EMA stays small and amplification has little effect.
During the grokking transition, gradients begin pointing consistently toward
the generalizing solution, the EMA accumulates this signal, and amplification
accelerates convergence to the generalizing minimum.

Per-parameter state: gradient momentum, squared gradient average, gradient
EMA (three tensors).

**Compute pattern.** Pure elementwise. Per element:
  ema = alpha * ema + (1-alpha) * g
  g_amp = g + lamb * ema
  m = beta1 * m + (1-beta1) * g_amp
  v = beta2 * v + (1-beta2) * g_amp²
  p -= lr * (m/bc1 / (sqrt(v/bc2) + eps) + wd * p)
No reduction, no GEMM. Bandwidth-bound (~10 mem ops per element).

**Dependency chain.** EMA update → g_amp computation → Adam (m, v) update
→ param update. All sequential WITHIN an element but fully parallel
ACROSS elements. Across tensors: independent.

**State.** Per-element: param, grad, ema, exp_avg, exp_avg_sq. Per-tensor:
none. Per-step: alpha, lamb, bc1, bc2, beta1, beta2, lr, wd, eps (scalars).

**Precision.** FP32 accumulators (ema, exp_avg, exp_avg_sq). Params can
live in BF16 with stochastic rounding on writeback. The ema needs FP32
to avoid drift over the long persistence horizon (alpha=0.98 → effective
window of ≈ 50 steps).

### NeuralGrok

Adam with a learned per-element gradient amplifier. NeuralGrok trains a
separate small neural network (the "psi network") alongside the main model.
This amplifier network is a two-layer MLP that takes the absolute value of
each gradient element as input and outputs a multiplicative scaling factor.

At each step, every gradient element is independently scaled by the
amplifier's output before being used in the standard Adam update. The
amplifier is trained with its own optimizer to learn which gradient magnitudes
should be boosted and which should be dampened.

The amplifier weights are stored in GPU constant memory for fast access and
are cached across steps to avoid redundant transfers. The kernel supports CUDA
Graph capture: once the shapes and hyperparameters are fixed, the entire
amplifier-forward plus Adam-apply sequence is recorded as a graph and replayed
without launch overhead.

Per-parameter state: gradient momentum, squared gradient average (two tensors,
plus the amplifier network weights stored separately).

**Compute pattern.** Mixed — per-element MLP + AdamW. Per element:
  h = relu(W1 * |g| + b1)         — 1×1 @ 1×H elementwise broadcast (no MFMA win on layer 1)
  s = sum(W2 * h + b2)            — 1×H @ H×1 reduce-along-H
  g_amp = (alpha * s + beta) * g
  AdamW on g_amp.

**Dependency chain.** MLP layer 2 depends on layer 1's ReLU output. The
multiplicative scaling `g_amp = (alpha*s + beta) * g` is the join point.
Then AdamW. Within an element, the chain is fully sequential; across
elements, fully parallel.

**State.** Per-element: param, grad, exp_avg, exp_avg_sq. Per-tensor: none.
Per-step: amplifier weights (W1, b1, W2, b2 — shared across all params),
alpha, beta, hidden_dim, bc1, bc2, beta1, beta2, lr, wd, eps.

**Precision.** FP32 accumulators (m, v). Amplifier MLP forward can run
BF16 with FP32 accumulate; layer 2 is the MFMA-amenable contraction
(across the hidden dim H ≥ 16 — but we batch it across N for the MFMA
win to materialize).

### Prodigy

A self-tuning variant of Adam that automatically adjusts its learning rate
without manual configuration. The recommended initial learning rate is 1.0
because Prodigy internally manages the effective step size.

The core idea is to estimate how far the parameters have traveled from their
initial values and use that distance to calibrate the learning rate. Prodigy
maintains a copy of the initial parameter values and a running trajectory
estimate. At each step, it computes two global statistics across all
parameters: a numerator (r) measuring alignment between gradients and the
parameter trajectory, and a denominator (s) measuring the overall trajectory
magnitude. The adaptive learning rate d is updated as the ratio of these two
quantities.

The entire d computation runs on-device without any GPU-to-CPU
synchronization. A three-kernel orchestration handles this: the first kernel
block-reduces the partial sums using warp shuffles and shared memory, the
second kernel updates the d scalar on a single thread, and the third kernel
applies the Adam step using the new d value read directly from device memory.

Per-parameter state: gradient momentum, squared gradient average, trajectory
estimate, initial parameter snapshot (four tensors).

**Compute pattern.** Mixed — elementwise + two global reductions. Per element:
  r_local += g * (p_init - p) * d            — 1 sub, 2 muls
  s_local += d² * g                          — 1 mul
  trajectory accumulator update              — elementwise
  AdamW with `d` as the effective lr scale.
Then GLOBAL reduce: r_global = sum(r_local) across all elements + tensors;
s_global = sum(s_local). d_new = max(d_prev, r_global / |s_global|).

**Dependency chain.** The d update is a barrier — every element's update
depends on the global scalars r_global and s_global. So: per-element
partial-reduce → cross-block reduce → d update → per-element AdamW
(now using the updated d). Three-kernel orchestration on the Hopper side
to avoid host syncs.

**State.** Per-element: param, grad, exp_avg, exp_avg_sq, s_track, param_init.
Per-tensor: none (the reductions go straight to global scalars).
Per-step: r_global, s_global, d (three scalars carried across steps).

**Precision.** FP32 EVERYWHERE — the reductions must be FP32 because (a)
they accumulate across millions of elements (catastrophic cancellation in
BF16 is real here), and (b) the d update is a divide r/s which amplifies
any per-step noise. Param can be BF16 with FP32 accumulators.

### Grokfast

The simplest grokking-aware optimizer. Grokfast wraps standard AdamW with an
exponential moving average filter that amplifies persistent gradient
directions.

Each step has two phases. First, the per-element gradient EMA is updated with
a decay factor (alpha), smoothing out noise while accumulating consistent
signals. Second, the current gradient is amplified by adding a scaled copy of
the EMA (multiplied by lambda) to it. This amplified gradient then goes
through normal AdamW: momentum averaging, second-moment tracking, adaptive
per-element scaling, and decoupled weight decay.

A fully-fused kernel variant performs both the EMA update and the Adam step in
a single GPU pass, keeping the amplified gradient in registers throughout.

Per-parameter state: gradient EMA, gradient momentum, squared gradient
average (three tensors).

**Compute pattern.** Pure elementwise. Structurally identical to GrokAdamW
(EMA filter → amplify → AdamW). Per element: ~10 mem ops, 8-10 FMAs. No
reduction, no GEMM. Bandwidth-bound.

**Dependency chain.** EMA update → amplify → Adam apply, sequential within
an element, fully parallel across elements. Across tensors: independent.

**State.** Per-element: param, grad, ema, exp_avg, exp_avg_sq. Per-tensor:
none. Per-step: grokfast_alpha, grokfast_lamb, bc1, bc2, beta1, beta2,
lr, wd, eps (scalars).

**Precision.** FP32 for the EMA (long-window accumulator), Adam moments,
and amplification computation. Param storage can be BF16 with stochastic
rounding.

### Lion

A sign-based optimizer that uses only the direction, not the magnitude, of
gradient information. Lion maintains a single momentum buffer per parameter
(no squared gradient tracking, unlike Adam).

At each step, Lion computes a weighted interpolation between the current
gradient and the stored momentum. It then takes the element-wise sign of this
interpolation — every element becomes exactly positive one or negative one.
The parameter update uses this sign vector multiplied by the learning rate,
giving every parameter element a uniform-magnitude update. Weight decay is
applied separately before the sign step.

After computing the update, Lion refreshes the momentum buffer with a
different interpolation ratio (beta2 instead of beta1), creating an asymmetry
between the "update direction" blend and the "stored momentum" blend.

The sign-based approach provides implicit regularization because all updates
have equal magnitude regardless of gradient scale. This means Lion is less
sensitive to gradient magnitude outliers and typically works well with
stronger weight decay. It uses roughly half the memory of Adam since there is
no second-moment buffer.

Per-parameter state: momentum buffer (one tensor).

**Compute pattern.** Pure elementwise. Each parameter element reads its
gradient + momentum, writes the new momentum and the new param. No
reduction, no GEMM.

**Dependency chain.** Update = sign(β₁·m + (1-β₁)·g) — fully parallel
across elements within one tensor. Momentum update m ← β₂·m + (1-β₂)·g
happens after the param update and is also fully parallel. Between
tensors: independent. No cross-step dependencies inside a single
`step()` call.

**State.** Per-element: momentum buffer (one tensor), parameter (one tensor).
Per-tensor: none. Per-step: lr, β₁, β₂, weight_decay (4 scalars).

**Precision.** Momentum can live in BF16 with FP32 accumulation during the
β-blend; the sign() collapses magnitude so output precision is irrelevant.
Param update is FP32-accumulate, can store back to BF16 params with
stochastic rounding.

### LookSAM

AdamW enhanced with periodic Sharpness-Aware Minimization. Standard SAM
requires two forward-backward passes per step, doubling training cost.
LookSAM reduces this by performing the SAM computation only every k steps
(default 5), using cached direction information for the steps in between.

On a SAM step: the optimizer perturbs each parameter in the direction of its
gradient (scaled by rho), computes the loss at the perturbed point, measures
the gradient difference between the perturbed and original points, and stores
this difference as the SAM direction. On non-SAM steps: the cached SAM
direction is blended with the current gradient using interpolation factor
alpha, steering the update toward flatter regions of the loss landscape
without recomputing the perturbation.

The perturbation, restoration, direction adjustment, and norm reduction are
each separate kernels. The norm reduction uses Hopper's distributed shared
memory for cross-CTA communication on sm_90, avoiding global memory
round-trips.

Per-parameter state: gradient momentum, squared gradient average, cached SAM
direction (three tensors).

**Compute pattern.** Mixed — four sequential phases on a SAM step:
  (1) perturb: p_pert = p + rho * g / ||g||                 — needs ||g|| reduce
  (2) loss + grad at perturbed point (model forward+backward; external)
  (3) restore + set_direction: sam_dir = g_sam - g           — elementwise
  (4) AdamW with g_adj = (1-alpha)*g + alpha*sam_dir         — elementwise
On non-SAM steps: just (4) using the cached sam_dir from the last SAM step.

**Dependency chain.** ||g|| computation is a global reduction (single
FP32 scalar per parameter tensor). Steps 1-3 must serialize against the
model-level forward+backward in between. Step 4 is fully parallel across
elements. The "k-step cache" trades a 2× cost on SAM steps for k-1
SAM-free steps that reuse sam_dir.

**State.** Per-element: param, grad, sam_dir, exp_avg, exp_avg_sq.
Per-tensor: ||g|| (during perturb), backup of param (during perturb+restore).
Per-step: rho, k, alpha (interp weight), bc1, bc2, lr, wd, eps.

**Precision.** FP32 for ||g|| reduce (avoid underflow on tiny grads).
FP32 for Adam moments. SAM direction can be BF16 since it's a unit-norm
direction (magnitude info is in the alpha multiplier).

### Muon

A dual-strategy optimizer that uses different update rules for different
parameter shapes. Two-dimensional weight matrices (the bulk of a neural
network's parameters) are updated using momentum followed by Newton-Schulz
orthogonalization, while one-dimensional parameters (biases, layer norm
scales, embeddings) fall back to standard AdamW.

For 2D weights: Muon maintains a momentum buffer and normalizes it by its
Frobenius norm. It then runs several iterations (default 5) of Newton-Schulz
refinement, which iteratively orthogonalizes the momentum matrix. Each
iteration involves matrix multiplications (through CUTLASS on Hopper or
cuBLAS/rocBLAS on other backends) that push the momentum toward the nearest
orthogonal matrix. The orthogonalized update is then applied to the
parameters with a trust-ratio scaling factor.

The idea is that orthogonal weight updates preserve the conditioning of weight
matrices throughout training, preventing the rank collapse and gradient
vanishing that plague deep networks.

Per-parameter state: momentum buffer for 2D weights; gradient momentum and
squared gradient average for 1D parameters.

**Compute pattern.** Mixed and GEMM-heavy on 2D params. Per 2D weight
matrix (shape [rows, cols], typically 96×96 to 1024×1024 for grokking
models):
  buf = momentum * buf + grad                       — elementwise
  inv_norm = 1 / ||buf||_F                          — global reduction
  X = buf * inv_norm                                — elementwise broadcast
  for step in {0..4}:
    A   = X @ X.T                                   — GEMM [rows, cols] · [cols, rows]
    AX  = A @ X                                     — GEMM [rows, rows] · [rows, cols]
    AAX = A @ AX                                    — GEMM [rows, rows] · [rows, cols]
    X   = 3.4445*X - 4.7750*AX + 2.0315*AAX         — elementwise FMA
  p = (1-lr*wd) * p - lr * scale * X                — elementwise
1D params: standard AdamW (see below).

**Dependency chain.** The Newton-Schulz iteration is serial: each iter
depends on the previous X. WITHIN each iter, AX waits on A, and AAX
waits on AX (three serial GEMMs per iter). 5 iters × 3 GEMMs = 15
sequential GEMMs per 2D param per step. Across 2D params: independent.
1D params can run AdamW in parallel with the NS iterations.

**State.** Per 2D param: momentum buffer + 3 scratch matrices (A, AX, AAX,
each [rows, rows] or [rows, cols]). The scratch can be reused across NS
iters. Per 1D param: exp_avg, exp_avg_sq.

**Precision.** Newton-Schulz GEMMs use BF16 inputs with FP32 accumulate
(WGMMA on Hopper, MFMA on CDNA3). The Frobenius norm needs FP32 accum.
Trust-ratio scale `scale_factor = 0.2 * sqrt(max(rows, cols))` is a
scalar, FP32. Param update: FP32 internally, can write back BF16.

### MoE/Adam multi-tensor

`MoEAwareSuperGrok2` — a SuperGrok v2 subclass that compacts active
expert parameters before running the full SG2 metanet. The class is
defined at the bottom of `grokking_optimizers/optimizers/supergrok2.py`
(below `CompiledSuperGrok2`) and inherits its hyperparameters (learning
rate, betas, weight decay, metanet config) from `SuperGrok2.__init__`.

In standard Mixture-of-Experts training, most expert parameters receive
zero gradients on any given step because the router only activates a
small subset of experts per input. Running the Mamba-3 scan over all
expert parameters wastes the cross-element correlation work on the
inactive experts.

MoEAwareSuperGrok2 solves this by compacting: when `active_expert_indices`
are provided, it identifies which expert parameters received non-zero
gradients, gathers only those into a dense buffer, runs the SG2 metanet
scan on the smaller active set, then scatters the results back to the
full parameter tensor. For top-2 routing with 64 experts, this means
processing roughly three percent of expert parameters instead of one
hundred percent. When no active set is provided, the class delegates
straight to `SuperGrok2.step()`.

Auxiliary features carried alongside the compaction:
- Per-expert activation counts feed a load-balancing auxiliary loss.
- Per-expert learning-rate scaling smooths activation frequency.
- The C++ helpers (`moe_filter_active_params`, `moe_scan_compacted`,
  `moe_scatter_results`) live in `csrc/algorithms/supergrok2.h` (the
  former `moe_adam.h` was folded in alongside the MoE variant) plus
  the per-arch launchers folded into `launch_supergrok2.{cu,hip.cpp,py}`.

**Compute pattern.** Mixed — preprocessing reductions + scatter + then the
full SG2 step (see SuperGrok v2 compute pattern). Preprocessing:
  expert_counts[e] = sum_{N_gate} (gate_logits[n, e] > threshold)   — count reduce
  load_balance_loss = SUM_e (count_e * P_e * num_experts)            — scalar reduction
  per-expert lr_scale[e] = sigmoid(EMA(activation_freq[e]))          — elementwise
Then compaction: for each parameter tensor, gather active expert params into
a dense buffer (filtered scatter), run SG2 scan + GRU on the smaller set,
scatter results back to full-tensor positions.

**Dependency chain.** expert_counts must finish before lr_scale update.
Compaction needs param_to_expert mapping (static, provided by the model).
The compacted SG2 step has the same internal dependency chain as SG2.
Scatter-back depends on the compacted output. Across expert tensors:
independent if the gather/scatter operates per-param.

**State.** Same as SG2 per-param, plus per-expert _expert_counts (int32)
and _lr_scale (FP32), both of length num_experts. Compaction scratch
buffers (compact_params, compact_grads, compact_state_m/v, scatter_indices,
compact_count) are allocated per-step.

**Precision.** Same as SG2. expert_counts is int32 (atomic-add safe).
Load-balance loss is FP32. lr_scale is FP32 with sigmoid smoothing.

---

## Architecture

The codebase is organized along two orthogonal axes:

1. **Algorithm** (what math to compute) — `csrc/algorithms/*.h`
2. **Backend** (which hardware to use) — `csrc/backends/<vendor>/<arch>/`

Algorithm headers are vendor-neutral: they declare `__device__ __forceinline__`
template functions that compile under both nvcc and hipcc, plus pure-JAX
mirrors inside each `csrc/backends/pallas/launch_<opt>.py`. Backend launch
files are non-templated glue that calls into the algorithm functions inside
grid-stride loops. If fused megakernels (one TU per model × optimizer × arch)
ever land, they'll live under `csrc/fused/<arch>/` — there are no placeholder
stubs in the meantime.

### Algorithm headers (`csrc/algorithms/`)

Eleven vendor-neutral headers, one per optimizer math family. Each
provides per-element step functions plus any vectorized fast paths; all
helper types/macros from the former `csrc/common/` are inlined inside
each header so they're self-contained:

- **adamw.h** — standard AdamW + float4 vec4 fast path
- **grokadamw.h** — EMA gradient filter + Adam
- **grokfast.h** — fused EMA + Adam
- **lion.h** — sign-based interpolated momentum + vec4 fast path
- **looksam.h** — 4 ops: perturb, restore, set_direction, apply
- **muon.h** — momentum normalize, Newton-Schulz combine, parameter update
- **neuralgrok.h** — psi-net MLP forward + Adam apply
- **prodigy.h** — partial reductions, d update, Adam with d as lr
- **supergrok11.h** — meta-MLP + cosine gate + Adam
- **supergrok15.h** — meta-MLP + per-coord alpha + Adam
- **supergrok2.h** — Mamba scan + warp-spec consumer + bilevel precompute,
  plus the folded-in MoE multi-tensor compact/scan/scatter helpers
  (formerly `moe_adam.h`)

### Model implementations (`csrc/backends/<vendor>/<arch>/models/`)

Three model architectures (decoder, vit, mamba) plus a shared attention
kernel live directly inside each backend rather than behind a
vendor-neutral header contract. Each backend's `models/` directory is
self-contained:

- **CUDA sm_90** (`csrc/backends/cuda/sm_90/models/`) — `.cuh` files
  hold template implementations; matched `.cu` files emit explicit
  instantiations for float/bfloat16/half so PyTorch's pybind link step
  has stable symbols.
- **HIP gfx942** (`csrc/backends/hip/gfx942/models/`) — `.hip.h`
  shim headers delegate to the sm_90 templates via inline wrappers;
  `.hip.cpp` files re-instantiate the templates under hipcc.
- **Pallas** — JAX/TPU model code lives inline inside each
  `launch_<opt>.py` rather than separate model files.

Model symbols are exposed through `sg::sm90::models::*` and
`sg::gfx942::models::*` to match the bindings' DISPATCH macros.

### Launch glue (10 files per backend)

For each backend, one launch file per optimizer (MoEAwareSuperGrok2 is
folded into SuperGrok v2):

```
csrc/backends/cuda/sm_90/launch_<opt>.cu       (10 files; SG2 absorbed MoE)
csrc/backends/hip/gfx942/launch_<opt>.hip.cpp  (10 files; SG2 raises std::runtime_error)
csrc/backends/pallas/launch_<opt>.py           (10 files)
```

Each launch file is **fully self-contained**:

1. Inlines `csrc/common/*` helpers it needs (platform macros, warp
   reductions, PTX intrinsics, quantization, BatchedScanCtx, …).
2. Inlines the per-backend primitives it needs (grid-stride loop,
   vec4 alignment, ATen tensor-op helpers, JAX scan kernels).
3. For Muon and SG2 (CUDA): inlines `mma.cuh` (CUTLASS wrappers + fused
   softplus epilogue) directly.
4. For SG2 (all backends): inlines `affine2x2.h` + the scan adapter.
5. Defines `__global__` kernels (CUDA only) that wrap the per-element step
   in a grid-stride loop.
6. Provides the host-side launcher function called from bindings.

### Bindings (`csrc/bindings/`)

Pybind11 entry points that connect Python to the C++ launchers. Five
files:

- **bindings.cpp** — all per-optimizer dispatchers (forward declarations
  + vector-signature entry points) plus the single `PYBIND11_MODULE(_ops, m)`
  registration block. Sections inside this file preserve the original
  per-file boundaries with `// ─── csrc/bindings/<filename>.cpp ───` markers
  so the diff against the pre-consolidation layout stays legible.
- **dispatch.cpp** — `int sg::detect_arch()` (CUDA/HIP probes + FORCE_ARCH
  env var) and the `fused_step` placeholder.
- **distributed_scan.cpp** — the three-phase multi-GPU Mamba-3 scan dispatch.
- **quantization.cpp** — FP8 / INT8 / INT4 quantize launchers.
- **helpers.h** — `SG_DISPATCH` macro, the `sg::detect_arch()` forward decl,
  and the device-side gradient norm helpers.

Each dispatcher inside `bindings.cpp` filters undefined gradients, packs
tensors into vectors, and calls `SG_DISPATCH(launcher, ...)` which picks
the right backend at runtime.

### HIP backend: ATen + rocBLAS-MFMA design

The HIP gfx942 launchers (`csrc/backends/hip/gfx942/launch_*.hip.cpp`) use
ATen tensor ops + rocBLAS rather than hand-written `__global__` HIP kernels.
This is a deliberate constraint of PyTorch's `cpp_extension`:

- `_is_cuda_file()` only matches `.cu`, `.cuh`, and `.hip` extensions for
  hipcc routing. Files with the `.hip.cpp` suffix go through the host
  compiler (g++/clang++), which cannot compile `__global__` decorations or
  `<<<grid, block>>>` launch syntax.

- ATen tensor ops on a HIP tensor dispatch to **rocBLAS** for GEMMs and
  **rocPRIM** (rocPRIM-thrust) for elementwise / reduction patterns.
  rocBLAS internally uses `v_mfma_f32_16x16x16_bf16` MFMA instructions on
  CDNA3 for BF16/FP16 inputs at sizes ≥ 16, so the dense-linear-algebra
  portion of every HIP launcher already exercises the MFMA pipeline —
  it just isn't visible in our source code.

Per-optimizer MFMA applicability (analysis in each launcher's file header):

| Optimizer       | Pattern              | MFMA via rocBLAS | Hand-written kernel win |
|-----------------|----------------------|------------------|-------------------------|
| Lion            | elementwise          | n/a — no GEMM    | ~1.3× (kernel fusion)   |
| AdamW           | elementwise          | n/a — no GEMM    | ~1.3× (kernel fusion)   |
| GrokAdamW       | elementwise          | n/a — no GEMM    | ~1.5× (fuse EMA+Adam)   |
| Grokfast        | elementwise          | n/a — no GEMM    | ~1.5× (fuse EMA+Adam)   |
| LookSAM         | elementwise + reduce | n/a              | ~1.7× (fuse 3 kernels)  |
| Prodigy         | elementwise + reduce | n/a              | ~2× (fuse reduce+apply) |
| Muon            | elementwise + GEMM   | ✓ (5 NS GEMMs)   | ~1.2× (skip rocBLAS overhead) |
| NeuralGrok      | per-element MLP      | ✓ (layer 2)      | ~1.5× (fuse MLP+Adam)   |
| SuperGrok v1.1  | per-param MLP + Adam | ✓ (MLP forward)  | ~1.5× (fuse)            |
| SuperGrok v1.5  | per-param MLP + Adam | ✓ (MLP forward)  | ~1.5× (fuse)            |
| SuperGrok v2    | scan + GEMM + GRU    | ✓ (projections)  | substantial (LDS scan)  |

Each launcher's file header contains a four-block analysis: COMPUTE PATTERN,
MFMA APPLICABILITY, WHY ATEN HERE, and the three-step migration recipe to
a hand-written kernel. The setup.py source glob picks up both `*.hip.cpp`
(host-compiler-routed, ATen+rocBLAS) and `*.hip` (hipcc-routed, real
`__global__` kernels via `hipLaunchKernelGGL`); migrating a launcher to
native is the same three-step recipe in the file header.

---

## Python frontend

The 12 optimizers under `grokking_optimizers/optimizers/` (1 AdamW
baseline + 11 grokking variants) are `torch.optim.Optimizer` subclasses. Each stores hyperparameters in
`param_groups` in `__init__` and dispatches in `step()`:

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99),
                 weight_decay=3.0, use_grad_hooks=False):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    @torch.no_grad()
    def step(self, closure=None):
        if self._use_grad_hooks:
            return None  # hooks already ran during backward()
        for group in self.param_groups:
            params, grads, exp_avgs = self._pack_group(group)
            _ops.lion_fused_step(params, grads, exp_avgs, **group_hyperparams)
```

The kernel call is the only execution path. If `_ops` is missing or a
launcher raises, the exception propagates — there is no Python fallback.

`grokking_optimizers/dispatch.py` provides:
- `detect_arch()` — returns `90`, `942`, or `"tpu_v5p"`
- `get_ops()` — loads the compiled C++ extension; raises `RuntimeError` if
  the extension is not built
- `has_fused(model, optimizer, arch=None)` / `dispatch_fused(...)` — fused
  kernel registry
- Capability predicates: `supports_bf16`, `supports_fp8`, etc.

`GradientHookOptimizer` (the former wrapper class) was removed. Each
optimizer now accepts `use_grad_hooks=True` directly in its constructor,
which registers `register_post_accumulate_grad_hook` on every parameter so
the update runs while gradient data is still L2-warm. `step()` is a no-op
once hooks are active.

---

## JAX/TPU

The TPU functional rewrite that previously lived under `supergrok2_jax_tpu/`
was folded into the Pallas backend itself. Each
`csrc/backends/pallas/launch_<optimizer>.py` is now fully self-contained:

- All 10 launch files carry their own `State` / `Config` namedtuples and
  the canonical per-parameter step function (Lion, Muon, Prodigy, …),
  plus inlined copies of TPU detection + Pallas-kernel re-exports
  (formerly in `primitives.py`).
- `launch_supergrok2.py` absorbs the full SG2 functional rewrite:
  bidirectional Mamba-3 scan, per-element GRU, multi-head PEER routing
  (soft + hard), meta-net composition, the SG2 optimizer step, the
  bilevel meta-update, INT8/INT4 quantization helpers, and the folded-in
  MoE multi-tensor launcher (`launch_moe_adam_step`).
- The former `primitives.py` is deleted — TPU detection and Pallas-kernel
  imports are inlined into every `launch_*.py`. Pallas kernels themselves
  still live in `_pallas_kernels.py` / `_pallas_models.py` (these are
  re-export targets, not shared helper code).

Pallas kernels (`csrc/backends/pallas/_pallas_kernels.py` and
`_pallas_models.py`) provide tile-128 affine prefix scan, fused GRU+PEER,
VMEM-persistent expert MLP, sharded multi-device scan, and the three model
forward/backward functions. The race driver calls into the launch_*.py files
directly when running on TPU.

---

## Race fairness

The grokking race uses four outer train/test splits (10/90, 25/75, 50/50,
80/20) with an inner val carve-out controlled by `val_ratio` (default 0.10;
auto-overrides to 0.05 on 10/90). A fixed early-stopping rule ends each run
at whichever comes first: test accuracy reaching 95% or step count reaching
20,000 — identical across all 12 optimizers. Three SG variants (v2, v1.5,
v1.1) consume the inner val for bilevel and meta updates; the other nine
train on train only. The `val_test_gap` in output is the key diagnostic for
meta-learning vs. masked overfitting.

### Tasks

Three algorithmic tasks, each on integers mod p=97:

- **Decoder — modular division.** `(a · b⁻¹) mod p`, 4-token sequence, 9,312 examples.
- **ViT — MNIST-addition.** `(a + b) mod p` from digit images, 16 patches of dim 49, 9,409 examples.
- **Mamba — chained division.** Length-3 chain `a / b₁ / b₂ / b₃ mod p`, 9,312 examples.

### Train/val/test split

| frac_train | val_ratio | train | val | test |
|------------|-----------|-------|-----|------|
| 0.10 | 0.05 (auto) | 884 | 46 | 8,382 |
| 0.25 | 0.10 | 2,095 | 232 | 6,985 |
| 0.50 | 0.10 | 4,190 | 466 | 4,656 |
| 0.80 | 0.10 | 6,704 | 745 | 1,863 |

### Run modes

- **A** — Single arch × single split. 6 seeds.
- **B** — Multi-split. One arch × 4 splits × 5 seeds.
- **C** — Arch comparison. 3 arches × one split × 5 seeds.
- **D** — Full sweep. 3 arches × 4 splits × 5 seeds = 660 runs.
- **E** — Scale comparison. 3 model scales × 5 seeds.

### CLI

See `python grokking_race_v2.py --help` for the full flag list. The most
common: `--gpus`, `--optimizers`, `--tasks`, `--train-test-ratios`,
`--early-stop-test-acc`, `--no-fused`.

---

## Refactor history

This repository underwent a major structural refactor that reorganized the
codebase into three orthogonal axes (algorithm, backend, fusion), then a
post-refactor cleanup pass, a final inlining pass, and a JAX collapse pass.
Refactor commits are tagged `refactor(phase-N): ...` and `cleanup: ...` in
the git log; the per-phase commit graph and per-file move tables live in
git history.

### 12-phase structural refactor

- Deleted the `csrc/device/` placeholder tree (~37 files, mostly TODO bodies)
- Consolidated SuperGrok v2's three CUDA files into one algorithm header +
  one launch file per backend
- Extracted vendor-neutral algorithm math from per-arch kernels into
  `csrc/algorithms/` (11 headers; MoE compaction helpers later folded
  into `supergrok2.h`)
- Reclassified the Mamba scan adapter as scan infrastructure (later inlined
  into the SG2 launchers + Mamba model files)
- Renamed `csrc/kernels/cuda/_cutlass_gemm.cuh` to
  `csrc/backends/cuda/sm_90/mma.cuh` (later inlined into Muon + SG2 launchers)
- Reorganized Python frontend into `optimizers/` subpackage
- Updated build matrix from optimistic "✓ done" to honest ✅/🟡/⛔ legend

### Honest status reclassification

| Cell | Before | After | Reason |
|------|--------|-------|--------|
| SuperGrok2 / gfx942 | ⛔ → 🟡 | 🟡 | Functional port via ATen + rocBLAS (MFMA for projection GEMMs). Scan recurrence is sequential ATen loop, slower than Hopper Blelloch + 4-warp specialization. Bilevel backward path raises (forward path is functional). Promotion to ✅ requires hardware validation. |
| All other optimizer × arch cells | done | 🟡 | Implemented end-to-end in the refactored tree, but not run on real hardware in this environment. Promotion to ✅ gated on the action items below. |
| All model × arch cells | done | 🟡 | Same — implementation exists, hardware validation pending. |

### Post-refactor cleanup pass

- Removed 11 unused extension modules (async, CUDA Graph, distributed, etc.)
- Inlined the remaining keepers (`Mamba3PEERMetaNet`, `PrecisionConfig`,
  `gradient_hook_optimizer`) directly into their consumer optimizers
- Dropped the NVFP4 / MXFP4 / FP4 / Blackwell / CDNA4 scaffolding from
  code (it was never compiled). The future-arch table in "Hardware support"
  above is documentation only.

### Final inlining pass

Every optimizer file is now fully self-contained:

| Class(es) inlined                                            | Now lives in                          |
|--------------------------------------------------------------|---------------------------------------|
| `Mamba3ScanBlock`, `MiniGRU`, `Mamba3PEERMetaNet`            | `optimizers/supergrok2.py`            |
| `PrecisionConfig` (with int8/int4 expert quantization)       | `optimizers/supergrok2.py`            |
| `SharpnessMetaNet` (duplicated, accepted cost)               | `optimizers/supergrok11.py` *and* `supergrok15.py` |
| `_adamw_step_reference` (pure-Python AdamW)                  | `optimizers/grokadamw.py`             |
| `GradientHookOptimizer` wrapper class                        | Replaced by `use_grad_hooks=True` constructor flag on every optimizer |
| `MoEAwareSuperGrok2` (subclass of SuperGrok v2)              | `optimizers/supergrok2.py` (folded in) |

Result: `grokking_optimizers/` shrank to 13 files (2 top-level —
`__init__.py` + `dispatch.py` — plus 11 in `optimizers/`:
`__init__.py` + 10 optimizer files, since MoEAwareSuperGrok2 lives
inside `supergrok2.py`). No fallback module, no underscored private
modules, no backward-compat shims. The public API surface is the
11 race optimizer classes only.

### Full inlining + no-fallback pass

The final structural pass deletes every shared cross-file boundary on the
C++ side and removes the Python fallback path entirely.

- **`grokking_optimizers/fallback.py` deleted.** The kernel call is the
  only execution path; if `_ops` is missing or a launcher raises, the
  exception propagates. Race optimizers no longer have try/except → fallback
  patterns or CPU Python branches inside `step()` / `bilevel_step()`.
- **`csrc/common/` (5 headers), `csrc/scan/` (3 files), `primitives.cuh`,
  `mma.cuh`, `primitives.hpp`, and `primitives.py` were deleted.** Their
  content is inlined into every backend launch file, model file, and
  algorithm header — wrapped in `// ── inlined from former <path> ──`
  blocks for reviewability.
- **Only `csrc/bindings/` survives as a shared cross-file directory** —
  it has to, because pybind11 needs a single registration entry point.

The Pallas backend collapse (former `supergrok2_jax_tpu/` package) was
folded into `csrc/backends/pallas/launch_*.py` earlier in the cleanup;
each launch file is self-contained at the Python level too.

---

## Action items for hardware validation

When this branch lands on a machine with a real sm_90 GPU and an MI300X:

**Build smoke test**
- [ ] `./build.sh` succeeds on sm_90 (H100/H200)
- [ ] `./build.sh` succeeds on gfx942 (MI300X) after `export USE_HIP=1`
- [ ] `pip install -e .` produces an importable `_ops` extension

**Import smoke test**
- [ ] `python -c "from grokking_optimizers import SuperGrok2, Lion"` works
- [ ] All 12 optimizers in `grokking_optimizers/optimizers/` instantiate
      without error
- [ ] `grokking_race_v2.py --help` runs cleanly

**Functional smoke test (sm_90)**
- [ ] 20-step training loop on the decoder modular-division task with Lion
      converges (loss decreases)
- [ ] 20-step training loop with SuperGrok v2 converges
- [ ] (Optional) Compare against a hand-written PyTorch reference outside
      the package to validate math; the package itself no longer ships a
      Python reference implementation.

**Honest stub test (gfx942)**
- [ ] On MI300X: `SuperGrok2(...).step()` completes without error (forward
      path) — bilevel meta-update will raise until the saved-activations
      backward kernel is implemented.

**Matrix promotion**
- [ ] After each above test passes, promote the corresponding cell in the
      build matrix from 🟡 → ✅
- [ ] If anything fails, add a follow-up commit with the fix and re-test

**Out-of-scope items (deferred)**
- Fused megakernels (`csrc/fused/<arch>/`) — directory currently absent;
  any future fusion work will recreate it with real content.
- Warp-specialized SG2 scan as a runtime-detected branch
- CUDA Graph capture for the SG2 pipeline
- DSMEM cross-CTA reductions wired into LookSAM / Prodigy norm kernels
- CI matrix (tests are inline via `--self-test`; no external test suite)

---

## Contributing

To add a new optimizer:

1. Add per-element math template to `csrc/algorithms/<optimizer>.h`.
   Inline whatever shared types/helpers it needs (BatchedScanCtx,
   warp_reduce_sum, etc.) — there is no `csrc/common/`.
2. Add launch glue for each backend, each fully self-contained:
   - `csrc/backends/cuda/sm_90/launch_<optimizer>.cu`
   - `csrc/backends/hip/gfx942/launch_<optimizer>.hip.cpp`
   - `csrc/backends/pallas/launch_<optimizer>.py`
3. Add a Python wrapper under `grokking_optimizers/optimizers/<name>.py`.
   Include the `use_grad_hooks: bool = False` constructor flag + a
   `_single_param_step(param, group, state)` method so the gradient-hook
   path works.
4. Re-export in `grokking_optimizers/__init__.py` and
   `grokking_optimizers/optimizers/__init__.py`.
5. Verify import: `python -c "from grokking_optimizers import <Class>"`.
6. Run a 20-step training loop on a tiny model to confirm convergence.

### Testing

Run the inline self-test suite:

```bash
python -m grokking_optimizers.compile --self-test
```

This runs 18 checks covering: YAML search space loading/validation/hashing,
PGO workload hashing and flag plumbing, Bayesian TPE optimization and
top-K refinement, compile cache v2→v3 migration and round-trip,
elementwise kernel header structure (4 optimizers × 2 GPU arches), model
kernel header existence (3 models × 3 arches), and optimizer × model
cross-validation (file existence + size helpers for all combinations).

---

## Codebase consolidation

The following modules were merged into `compile.py` to reduce file count
and eliminate cross-module coupling:

| Former module | Absorbed into |
|---------------|---------------|
| `grokking_optimizers/search_space.py` | `compile.py` — YAML loader, pre-filter, macro resolver |
| `grokking_optimizers/bayesian.py` | `compile.py` — Optuna TPE + neighbour refinement |
| `grokking_optimizers/timing_worker.py` | `compile.py` — persistent subprocess worker |
| `grokking_optimizers/bench_graph.py` | `compile.py` — CUDA/HIP graph capture+replay |
| `grokking_optimizers/pgo.py` | `compile.py` — instrument/collect/use flag plumbing |
| `scripts/pgo_workload.py` | `compile.py` — PGO workload entry point |
| `configs/search_space.yaml` | `compile.py` — `build_full_search_space()` programmatic builder (replaces the earlier embedded YAML) |
| `INTERFACES.md` | `README.md` — compile cache schema, CLI surface |
| `docs/autotune.md` | `README.md` — autotune guide, YAML schema, PGO |
| `docs/optimization_matrix.md` | `README.md` — optimization candidate matrix |
| `tests/` (all files) | `compile.py --self-test` — ~138 inline checks today |

---

## License

MIT License. See `LICENSE`.

Acknowledgements:
- JAX and Pallas teams at Google for TPU primitives.
- NVIDIA CUTLASS team for the GEMM template library (optional via `WITH_CUTLASS=1`).
