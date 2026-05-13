# SuperGrok2

SuperGrok2 is a C++/CUDA/HIP/Pallas optimizer suite for grokking-aware
training of large neural networks. It ships eleven optimizers spanning AdamW
variants, sign-momentum, sharpness-aware minimization, Newton-Schulz
orthogonalization, and a Mamba-3 + PEER + GRU meta-network optimizer
(SuperGrok v2 — the project's namesake). The grokking race driver
(`grokking_race_v2.py`) compares all eleven head-to-head on three algorithmic
learning tasks under controlled conditions.

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

Per-arch coverage of the 11 optimizers and 3 models. Honest legend:

- ✅ **done & validated on hardware** — implemented, build-checked, parity
  confirmed against a reference path
- 🟡 **done, unvalidated on hardware** — implemented and import-checked, but
  not yet validated on real hardware (no GPU available in this environment)
- ⛔ **stub / raises NotImplementedError** — explicitly unimplemented; the
  launcher raises a runtime error with a descriptive message

### Optimizer × arch matrix

| Optimizer | sm_90 (Hopper) | gfx942 (CDNA3) | tpu_v5p (Pallas) |
|-----------|:--------------:|:--------------:|:----------------:|
| SuperGrok v2  | 🟡 | ⛔ | 🟡 |
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

**SuperGrok v2 on gfx942 is honestly marked ⛔.** The launcher
(`csrc/backends/hip/gfx942/launch_supergrok2.hip.cpp`) raises
`std::runtime_error` with a clear message: the full Mamba + GRU + PEER
pipeline relies on Hopper-specific features (DSMEM cluster reductions, WGMMA,
4-warp-scheduler specialization) that have no direct CDNA3 equivalent. A
complete CDNA3 port using MFMA + LDS-resident scan + manual producer/consumer
synchronization would be weeks of additional work.

Everything marked 🟡 is implemented end-to-end in the refactored tree but has
not been run on real hardware in this environment. Phase 12 of the refactor
(see `REFACTOR_NOTES.md`) documents the smoke tests that must run on a real
H100, MI300X, or TPU v5p before any cell can be promoted to ✅.

---

## Filesystem

The codebase splits along three orthogonal axes: **algorithm** (the
vendor-neutral math), **backend** (per-arch launchers + primitives), and
**model** (decoder / ViT / mamba). Each axis owns its own directory.

```
.
├── README.md
├── grokking_race_v2.py   (race driver — 11 optimizers × 3 models × 4 splits)
├── setup.py / build.sh / pyproject.toml
├── grokking_optimizers/
│   ├── __init__.py             (re-exports the 11 optimizers + helpers)
│   ├── dispatch.py             (arch detection + fused kernel registry + get_ops)
│   ├── fallback.py             (pure-Python reference implementations)
│   ├── optimizers/             (11 torch.optim.Optimizer subclasses)
│   │   ├── supergrok2.py       grokadamw.py    looksam.py    prodigy.py
│   │   ├── supergrok15.py      grokfast.py     muon.py       moe_adam.py
│   │   └── supergrok11.py      lion.py         neuralgrok.py
└── csrc/
    ├── algorithms/             (12 vendor-neutral algorithm headers)
    │   ├── adamw.h             grokfast.h    looksam.h     prodigy.h
    │   ├── grokadamw.h         lion.h        moe_adam.h    supergrok11.h
    │   └── neuralgrok.h        muon.h        supergrok2.h  supergrok15.h
    ├── models/                 (3 vendor-neutral model definitions)
    │   ├── decoder.h
    │   ├── vit.h
    │   └── mamba.h
    ├── common/                 (platform.h, types.h, utils.cuh, ...)
    ├── scan/                   (Affine2x2 + mamba_scan_adapter)
    │   ├── affine2x2.h
    │   ├── mamba_scan_adapter.cuh
    │   └── mamba_scan_adapter.hip.h
    ├── backends/
    │   ├── cuda/sm_90/         (primitives.cuh + mma.cuh + 11 launch_*.cu + models/)
    │   ├── hip/gfx942/         (primitives.hpp + 11 launch_*.hip.cpp + models/)
    │   └── pallas/             (primitives.py + 12 launch_*.py + models/ + v5p/)
    └── bindings/               (pybind11 dispatchers + dispatch macro + helpers)
```

Launch glue files contain the `__global__` kernels (CUDA) or ATen-driven
implementations (HIP) or JAX wrappers (Pallas). When fused megakernels
ever get written, they'll live under `csrc/fused/<arch>/` with real
content — the prior placeholder stubs were removed.

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
to the compiled fused kernel for the active arch. On `NotImplementedError` or
a missing kernel, the optimizer falls back to the pure-Python reference in
`grokking_optimizers/fallback.py`.

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

### MoE/Adam multi-tensor

A multi-tensor batched Adam variant optimized for Mixture-of-Experts models.
In standard MoE training, most expert parameters receive zero gradients on
any given step because the router only activates a small subset of experts
per input. Running a full optimizer pass over all expert parameters wastes
computation on the inactive experts.

MoE/Adam solves this by compacting: it identifies which expert parameters
received non-zero gradients (the active set), gathers only those into a dense
buffer, runs the Adam step on this smaller active set, then scatters the
results back to the full parameter tensor. For top-2 routing with 64 experts,
this means processing roughly three percent of expert parameters instead of
one hundred percent.

The same launcher also serves as the multi-tensor variant of plain AdamW for
non-MoE models — when all parameters receive gradients, the compaction is a
no-op and the math degenerates to standard AdamW.

Per-parameter state: gradient momentum, squared gradient average (two
tensors), plus per-expert activation counts when used in MoE mode.

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

Twelve vendor-neutral headers, one per optimizer math family. Each provides
per-element step functions plus any vectorized fast paths:

- **adamw.h** — standard AdamW + float4 vec4 fast path
- **grokadamw.h** — EMA gradient filter + Adam
- **grokfast.h** — fused EMA + Adam
- **lion.h** — sign-based interpolated momentum + vec4 fast path
- **looksam.h** — 4 ops: perturb, restore, set_direction, apply
- **moe_adam.h** — multi-tensor AdamW wrapper
- **muon.h** — momentum normalize, Newton-Schulz combine, parameter update
- **neuralgrok.h** — psi-net MLP forward + Adam apply
- **prodigy.h** — partial reductions, d update, Adam with d as lr
- **supergrok11.h** — meta-MLP + cosine gate + Adam
- **supergrok15.h** — meta-MLP + per-coord alpha + Adam
- **supergrok2.h** — Mamba scan + warp-spec consumer + bilevel precompute
  (consolidated; one file holds fwd, bwd, and warp-spec math)

### Model headers (`csrc/models/`)

Three vendor-neutral model contracts:

- **decoder.h** — autoregressive transformer (causal self-attention + FFN)
- **vit.h** — vision transformer (patch projection + non-causal attention)
- **mamba.h** — selective state-space model (depthwise conv + scan + gate)

Each defines a `<Model>Config` struct and a `<Model>LayerWeights` pointer
layout. Per-backend forward/backward implementations live under
`csrc/backends/<vendor>/<arch>/models/`.

### Scan infrastructure (`csrc/scan/`)

Shared between the Mamba model and the SuperGrok v2 optimizer:

- **affine2x2.h** — `Affine2x2` struct + 12-FMA composition (PTX on CUDA,
  C++ fallback on HIP). Used as the associative operator for parallel prefix
  scan over Mamba's selective recurrence.
- **mamba_scan_adapter.cuh** — CUDA scan adapter (packs model-level
  parameters into `Affine2x2` maps, dispatches to sequential or parallel
  scan).
- **mamba_scan_adapter.hip.h** — HIP equivalent.

### Common headers (`csrc/common/`)

Shared infrastructure used by every backend:

- **platform.h** — CUDA/HIP portability macros (warp size, shuffle, non-temporal)
- **types.h** — `BatchedScanCtx`, dimension caps, branchless stochastic rounding
- **utils.cuh** — warp/cluster reductions, hash PRNG, PTX wrappers
- **ptx_intrinsics.cuh** — hot-path PTX (softplus, exp, stochastic round, GRU gates)
- **quantization.h** — FP8/INT8/INT4 quantization helpers

### Backend primitives

Per-vendor helpers that the launch files share:

- **cuda/sm_90/primitives.cuh** — grid-stride helpers, warp/block/cluster
  reductions, RoPE pair rotation, vec4 alignment check, non-temporal load/
  store, stochastic rounding, last-block-finished pattern.
- **cuda/sm_90/mma.cuh** — CUTLASS GEMM wrappers (FP16/BF16) + fused softplus
  epilogue. Gated on `-DWITH_CUTLASS`.
- **hip/gfx942/primitives.hpp** — host-side ATen tensor-op helpers
  (`ema_update_inplace`, `adam_apply_inplace`, `pack_valid`). HIP launchers
  cannot define `__global__` kernels because `.hip.cpp` files route through
  the host compiler, not hipcc.
- **pallas/primitives.py** — pure JAX `adamw_step` / `lion_step` /
  `ema_update`, TPU version detection, and re-exports of working Pallas
  kernels (`mamba3_scan_pallas_tile128`, `pallas_fused_gru_peer`, etc.).

### Launch glue (33 files per backend)

For each backend, one launch file per optimizer:

```
csrc/backends/cuda/sm_90/launch_<opt>.cu       (11 files + SG2 fwd+bwd+warp-spec consolidated)
csrc/backends/hip/gfx942/launch_<opt>.hip.cpp  (11 files; SG2 raises NotImplementedError)
csrc/backends/pallas/launch_<opt>.py           (11 files + grokadamw)
```

Each launch file:
1. Includes the algorithm header + backend primitives
2. Defines `__global__` kernels (CUDA only) that wrap the per-element step in
   a grid-stride loop
3. Provides the host-side launcher function called from bindings

### Bindings (`csrc/bindings/`)

Pybind11 entry points that connect Python to the C++ launchers:

- **bindings.cpp** (formerly `module.cpp`) — pybind11 module entry
- **dispatch.cpp** — arch detection + `fused_step` router
- **helpers.h** (formerly `_helpers.h`) — gradient clipping + SAM norm

Per-optimizer dispatcher .cpp files in `csrc/bindings/` filter undefined
gradients, pack tensors into vectors, and call `SG_DISPATCH(launcher, ...)`
which picks the right backend at runtime.

---

## Python frontend

The 11 optimizers under `grokking_optimizers/optimizers/` are
`torch.optim.Optimizer` subclasses. Each stores hyperparameters in
`param_groups` in `__init__` and dispatches in `step()`:

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params, grads, exp_avgs = self._pack_group(group)
            try:
                _ops.lion_fused_step(params, grads, exp_avgs, **group_hyperparams)
            except NotImplementedError:
                from grokking_optimizers.fallback import lion_fallback
                lion_fallback(params, grads, exp_avgs, **group_hyperparams)
```

`grokking_optimizers/dispatch.py` provides:
- `detect_arch()` — returns `90`, `942`, or `"tpu_v5p"`
- `get_ops()` — loads the compiled C++ extension; raises `RuntimeError` if
  the extension is not built
- `has_fused(model, optimizer, arch=None)` / `dispatch_fused(...)` — fused
  kernel registry
- Capability predicates: `supports_bf16`, `supports_fp8`, etc.

Pure-Python reference implementations live in
`grokking_optimizers/fallback.py` and are used both as the dispatch fallback
and as ground-truth for parity tests.

`GradientHookOptimizer` (a thin wrapper that runs per-parameter steps via
`register_post_accumulate_grad_hook`) lives at
`grokking_optimizers/optimizers/gradient_hook.py` and is the only "extension"
that survived the post-refactor cleanup.

---

## JAX/TPU

The TPU functional rewrite that previously lived under `supergrok2_jax_tpu/`
was folded into the Pallas backend itself. Each
`csrc/backends/pallas/launch_<optimizer>.py` is now fully self-contained:

- All 11 launch files carry their own `State` / `Config` namedtuples and
  the canonical per-parameter step function (Lion, Muon, Prodigy, …).
- `launch_supergrok2.py` (≈875 lines) absorbs the full SG2 functional
  rewrite: bidirectional Mamba-3 scan, per-element GRU, multi-head PEER
  routing (soft + hard), meta-net composition, the SG2 optimizer step,
  the bilevel meta-update, and INT8/INT4 quantization helpers.
- `primitives.py` is now slim — just TPU version detection and re-exports
  of the Pallas kernels in `_pallas_kernels.py` / `_pallas_models.py`.

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
20,000 — identical across all 11 optimizers. Three SG variants (v2, v1.5,
v1.1) consume the inner val for bilevel and meta updates; the other eight
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
  `csrc/algorithms/` (12 headers — `moe_adam.h` is a thin AdamW wrapper kept
  separate for naming symmetry)
- Reclassified the Mamba scan adapter as scan infrastructure (shared by both
  the model and the SG2 optimizer) under `csrc/scan/`
- Renamed `csrc/kernels/cuda/_cutlass_gemm.cuh` to
  `csrc/backends/cuda/sm_90/mma.cuh`
- Reorganized Python frontend into `optimizers/` subpackage
- Updated build matrix from optimistic "✓ done" to honest ✅/🟡/⛔ legend

### Honest status reclassification

| Cell | Before | After | Reason |
|------|--------|-------|--------|
| SuperGrok2 / gfx942 | done | ⛔ | `launch_supergrok2.hip.cpp` raises `std::runtime_error`. The full Mamba+GRU+PEER pipeline needs Hopper-specific features (DSMEM cluster reductions, WGMMA, 4-warp specialization) with no direct CDNA3 equivalent. |
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
| `GradientHookOptimizer`                                      | `optimizers/gradient_hook.py`         |

Result: `grokking_optimizers/` shrank from 30 → 16 files. No underscored
private modules, no backward-compat shims. The public API surface trimmed
down to the 11 optimizers plus `GradientHookOptimizer`.

### JAX collapse pass

The `supergrok2_jax_tpu/` package was folded into the Pallas backend.
Each `csrc/backends/pallas/launch_<opt>.py` now carries its own JAX
implementation (State, Config, step function). `launch_supergrok2.py`
absorbed seven JAX files (≈875 lines): bidirectional Mamba-3 scan, GRU
cell, PEER routing (soft + hard), meta-net composition, optimizer step,
bilevel meta-update, INT8/INT4 quantization. `primitives.py` is now slim
(TPU detection + Pallas kernel re-exports).

---

## Action items for hardware validation

When this branch lands on a machine with a real sm_90 GPU and an MI300X:

**Build smoke test**
- [ ] `./build.sh` succeeds on sm_90 (H100/H200)
- [ ] `./build.sh` succeeds on gfx942 (MI300X) after `export USE_HIP=1`
- [ ] `pip install -e .` produces an importable `_ops` extension

**Import smoke test**
- [ ] `python -c "from grokking_optimizers import SuperGrok2, Lion"` works
- [ ] All 11 optimizers in `grokking_optimizers/optimizers/` instantiate
      without error
- [ ] `grokking_race_v2.py --help` runs cleanly

**Functional smoke test (sm_90)**
- [ ] 20-step training loop on the decoder modular-division task with Lion
      converges (loss decreases)
- [ ] 20-step training loop with SuperGrok v2 converges
- [ ] Both above tests pass elementwise allclose vs the Python fallback to
      within 1e-3

**Honest stub test (gfx942)**
- [ ] On MI300X: `SuperGrok2(...).step()` raises `NotImplementedError` with
      the message from `launch_supergrok2.hip.cpp`

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

1. Add per-element math template to `csrc/algorithms/<optimizer>.h`
2. Add launch glue for each backend:
   - `csrc/backends/cuda/sm_90/launch_<optimizer>.cu`
   - `csrc/backends/hip/gfx942/launch_<optimizer>.hip.cpp`
   - `csrc/backends/pallas/launch_<optimizer>.py`
3. Add a Python wrapper under `grokking_optimizers/optimizers/<name>.py`
4. Add a pure-Python reference in `grokking_optimizers/fallback.py`
5. Re-export in `grokking_optimizers/__init__.py`
6. Verify import: `python -c "from grokking_optimizers import <Class>"`
7. Run a 20-step training loop on a tiny model to confirm convergence

---

## License

MIT License. See `LICENSE`.

Acknowledgements:
- JAX and Pallas teams at Google for TPU primitives.
- NVIDIA CUTLASS team for the GEMM template library (optional via `WITH_CUTLASS=1`).
