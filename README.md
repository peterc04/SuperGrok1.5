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

## Hardware support

The build targets a **3-arch active set**:

| Arch | Family | Cards | Backend |
|------|--------|-------|---------|
| `sm_90` | Hopper | H100, H200 | CUDA |
| `gfx942` | CDNA3 | MI300X, MI300A | HIP |
| `tpu_v5p` | TPU v5p | 128-wide MXU | JAX/Pallas |

Anything else raises `UnsupportedArchError`. There is no tier fallback chain
for production runs — the active set is the only compiled set.

### Future arches (scaffolded in dispatch, not compiled)

These arches are recognized by `grokking_optimizers.dispatch.detect_arch`
and will route to a `NotImplementedError` with a descriptive message
until the corresponding kernels are added. They are not part of the
current build matrix.

| Arch    | Family               | Cards                          | Status   |
|---------|----------------------|--------------------------------|----------|
| sm_80   | Ampere               | A100, A30, A10                 | future   |
| sm_89   | Ada Lovelace         | RTX 40, L40, L40S              | future   |
| sm_100  | Datacenter Blackwell | B100, B200, GB200              | future   |
| sm_103  | Blackwell Ultra      | B300, GB300 NVL72              | future   |
| sm_120  | Consumer Blackwell   | RTX 50, RTX PRO 6000           | future   |
| gfx950  | CDNA4                | MI350X, MI355X                 | future   |
| tpu_v6e | TPU v6e              | 256-wide MXU                   | future   |

The active set narrowed to sm_90 + gfx942 + tpu_v5p during refactor to
focus on what's runnable now; future arches return when kernels are added.

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
not been run on real hardware in this environment. Phase 12 of the refactor
(see `REFACTOR_NOTES.md`) documents the smoke tests that must run on a real
H100, MI300X, or TPU v5p before any cell can be promoted to ✅.

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
├── scripts/                    (build / dev helpers)
├── tests/                      (correctness tests + JAX/Triton reference impls)
│   └── reference/
│       └── models/
├── third_party/                (cutlass git submodule for WITH_CUTLASS=1)
├── grokking_optimizers/
│   ├── __init__.py             (re-exports the 12 optimizers + helpers)
│   ├── dispatch.py             (arch detection + fused kernel registry + get_ops)
│   ├── compile.py              (targeted (opt, model, arch) ninja build + autotune)
│   ├── profile.py              (standalone ncu / rocprof / jax.profiler capture)
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

Dev-time companion to `setup.py`. Given an `(optimizer, model, arch)` triple,
compiles the matching subset of `csrc/` with arch-tuned codegen, LTO, and an
optional autotune pre-pass — all driven through ninja with
`MAX_JOBS=$(nproc)`. The profile capture is delegated to the sibling module
`grokking_optimizers.profile`, which can also be invoked standalone against
any pre-built artefact (see next subsection). Use these when iterating on a
specific combo; use `setup.py` (the default `pip install -e .` path) when
building the full production extension consumed by the race driver.

```bash
# CLI
python -m grokking_optimizers.compile \
    --optimizer supergrok2 --model mamba --arch sm_90 \
    [--no-autotune] [--no-profile] [--out build/compiled] \
    [--report build_report.txt] [-D MY_FLAG=1]

# Importable
from grokking_optimizers.compile import build
so_path = build(optimizer="supergrok2", model="mamba", arch="sm_90")
```

What runs, in order:

1. **Resolve sources** — bindings + every launcher and model TU for the
   chosen arch (18 files), so the bindings link cleanly.
2. **Inject metaprog macros** so headers can `#if` out unused
   specialisations for the chosen combo:
   `-DSG_BUILD_OPTIMIZER_<UPPER>=1`,
   `-DSG_BUILD_MODEL_<UPPER>=1`,
   `-DSG_BUILD_ARCH_<…>=1`,
   `-DSG_VERBOSE=1`.
3. **Build with ninja `-j$(nproc)`** via
   `torch.utils.cpp_extension.load`, with arch-tuned flags + LTO
   (sm_90: `--maxrregcount=255 -gencode arch=compute_90,code=sm_90 -dlto
   -Xcompiler -flto`, CUTLASS auto-enabled if `third_party/cutlass/` is
   present; gfx942: `--offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage
   -flto`).
4. **Autotune pass** (`--no-autotune` to skip): pre-build with
   `AUTOTUNE_PASS=1` stub configs, then rewrite
   `csrc/algorithms/tuned_configs.h` with per-arch
   `SG_TUNED_BLOCK_SIZE / VEC_WIDTH / UNROLL` constants so the final
   build picks them up.
5. **Profile pass** (`--no-profile` to skip) — delegates to
   `grokking_optimizers.profile`; see below for the standalone CLI and
   what each arch's profiler captures.

Output goes to a single text report (default
`build/compiled/compile_<O>_<M>_<A>.txt`); stdout only prints the report
path. Progress is reported on stderr via a tqdm bar with elapsed/ETA,
falling back to a `[i/N elapsed=Xs eta=Ys]` line when tqdm is missing.

#### Compile requirements per arch

The triple selected determines what hardware/toolchain must be present.
Builds without the optional profiler still succeed; only the profile
capture is skipped with a `[skip]` line in the report.

| Selected `--arch` | Hardware required to build | Hardware required to run + profile | Required compiler | Optional profiler |
|---|---|---|---|---|
| `sm_90` | None for build (set `FORCE_CUDA=1` to compile without a visible GPU); CUDA Toolkit ≥ 12.0 + nvcc on `PATH` | NVIDIA Hopper (H100 / H200) for `opt.step()` and `ncu` capture | `nvcc` (CUDA Toolkit) + `g++` ≥ 9 | `ncu` (Nsight Compute, `--set full` + 7 sections) |
| `gfx942` | None for build (PyTorch ROCm install); ROCm ≥ 6.0 + hipcc on `PATH` | AMD CDNA3 (MI300X / MI300A) for `opt.step()` and rocprof capture | `hipcc` (ROCm) + host C++ for `.hip.cpp` | `rocprof-compute` ≫ `rocprofv2` ≫ `rocprof` (first found on `PATH`) |
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

Notes on the autotune pass:

- The pre-pass build sets `AUTOTUNE_PASS=1`, so any header guarded by
  `#ifdef AUTOTUNE_PASS` can swap in stub configs to make the build cheap
  even on a hot kernel.
- The rewritten `csrc/algorithms/tuned_configs.h` is the only mutating
  side-effect — re-run the command (or pass `--no-autotune`) to leave it
  alone. The defaults are 256-block on sm_90 and 512-block on gfx942
  (matching warp width × occupancy targets); a real micro-bench sweep
  would refine these per-`(opt, model)` combo.

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
- CI matrix (no `tests/` directory at the moment)

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

---

## License

MIT License. See `LICENSE`.

Acknowledgements:
- JAX and Pallas teams at Google for TPU primitives.
- NVIDIA CUTLASS team for the GEMM template library (optional via `WITH_CUTLASS=1`).
