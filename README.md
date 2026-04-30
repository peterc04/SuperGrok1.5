# SuperGrok2

SuperGrok2 is a C++/CUDA/HIP optimizer suite for grokking-aware training of
large neural networks. It ships eleven optimizers spanning AdamW variants,
sign-momentum, sharpness-aware minimization, Newton-Schulz orthogonalization,
and a Mamba-3 + PEER + GRU meta-network optimizer (SuperGrok v2 — the
project's namesake).

Every kernel is specialized per GPU architecture. There is no generic
fallback. Each supported architecture has its own translation unit wrapped in
its own C++ namespace, and dispatch is decided at runtime against the
detected device.

A JAX/Pallas path provides equivalent semantics on Google TPU. A CPU path
exists for reference and testing only — it is not a runtime fallback.

For deep architecture and engineering notes see `REFRESH.md` (this README
points to the relevant sections throughout).

## Hardware support

The build is AOT (ahead-of-time): kernels are compiled into a single fat
binary that embeds machine code for every supported architecture. Driver
JIT from embedded `sm_120` PTX provides forward-compatibility on newer
NVIDIA hardware.

### NVIDIA (CUDA)

| Arch     | Family                  | Cards                                  |
|----------|-------------------------|----------------------------------------|
| `sm_80`  | Ampere                  | A100, A30, A10                         |
| `sm_89`  | Ada Lovelace            | RTX 40-series, L40, L40S               |
| `sm_90`  | Hopper                  | H100, H200                             |
| `sm_100` | Datacenter Blackwell    | B100, B200, GB200                      |
| `sm_103` | Blackwell Ultra         | B300, GB300 NVL72                      |
| `sm_120` | Consumer Blackwell      | RTX 50-series, RTX PRO 6000 Blackwell  |

### AMD (ROCm/HIP)

| Arch     | Family | Cards                |
|----------|--------|----------------------|
| `gfx942` | CDNA3  | MI300X, MI300A       |
| `gfx950` | CDNA4  | MI350X, MI355X       |

### TPU (JAX / Pallas)

| Version | MXU width |
|---------|-----------|
| `v5p`   | 128       |
| `v6e`   | 256       |

### Not supported

The following are intentionally not supported and the build or runtime
dispatch will refuse them: V100, T4, RTX 20-series, `sm_86` (RTX 30-series
silently routes to `sm_80`), MI100 (`gfx908`), MI200 (`gfx90a`), all AMD
RDNA consumer GPUs, and TPU v3 / v4 / v5e. The CPU build path is for
testing only.

## Installation

```
git clone https://github.com/peterc04/SuperGrok1.5
cd SuperGrok1.5
git submodule update --init --recursive third_party/cutlass
bash build.sh
```

`build.sh` accepts:

```
bash build.sh                # default ninja-backed build
bash build.sh --autotune     # two-pass build with autotune sweep
bash build.sh --debug        # cuda-gdb friendly build (-G -O0)
bash build.sh --profile      # NCU-friendly build + profile_smoke run
```

Notes:

- Build is AOT only. There is no NVRTC and no runtime kernel compilation.
- Driver JIT from embedded `sm_120` PTX provides forward-compatibility on
  newer hardware.
- `ccache` and `sccache` are auto-detected via
  `CMAKE_*_COMPILER_LAUNCHER` if present.
- `WITH_CUTLASS=1` opts into CUTLASS GEMMs on Hopper and Blackwell. The
  default keeps cuBLAS / rocBLAS until you opt in.

## Quickstart

A minimal training loop with Lion (the simplest optimizer in the suite):

```python
import torch
from grokking_optimizers import Lion

model = torch.nn.Linear(64, 32).cuda()
opt = Lion(model.parameters(), lr=3e-4)

for x, y in batches:
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
```

## Optimizers

| Optimizer | Summary |
|-----------|---------|
| **Lion** | Sign-momentum optimizer; minimal memory footprint. |
| **GrokAdamW** | AdamW with grokking-detection scheduling on top. |
| **NeuralGrok** | Grokking optimizer with a learned 2-layer MLP gradient amplifier. |
| **Prodigy** | Adaptive learning-rate optimizer that learns its own `d_lr` online. |
| **Grokfast** | Slow-gradient EMA filter to accelerate the grokking transition. |
| **LookSAM** | Sharpness-Aware Minimization with periodic perturbation-direction caching. |
| **Muon** | Newton-Schulz orthogonalization step for 2D parameters. |
| **SuperGrok v1.1** | Grokking-aware MLP meta-net optimizer (the v1.5 predecessor). |
| **SuperGrok v1.5** | Grokking-aware MLP meta-net + Lamb trust-ratio + SAM. |
| **SuperGrok v2** | Mamba-3 + PEER + GRU meta-net optimizer — the project's namesake. |
| **MoE / Mamba3PEER bindings** | Auxiliary FP4 / FP6 / 2:4-sparsity kernels supporting SG v2 and the MoE expert path. |

For full per-optimizer descriptions see `REFRESH.md §3`.

## Build modes

| Mode         | Effect |
|--------------|--------|
| (default)    | AOT fatbin build through ninja. Embeds machine code for every supported arch and `sm_120` PTX for forward-compat. |
| `--autotune` | Two-pass build. Stub-config build first, then `python autotune/tune.py` sweeps grids and writes winners between the `// AUTOTUNE_BEGIN` / `// AUTOTUNE_END` markers in `csrc/common/tuned_configs.h`, then a final rebuild. |
| `--debug`    | Compiles with `-G -O0`, suitable for `cuda-gdb` stepping through device code. |
| `--profile`  | Compiles with `-lineinfo` and runs `ncu --set full` against `benchmarks/profile_smoke.py` (5 steps × all 11 optimizers) so the profile is dropped next to the build. |

## Architecture overview

The codebase is split into three layers:

- `csrc/bindings/` — per-optimizer dispatchers and the pybind11 module
  aggregator (`module.cpp`). Each per-optimizer file exposes both a
  high-level vector-signature entry point (the primary contract used by
  the Python optimizers) and per-tensor escape hatches used by tests.
- `csrc/kernels/{cuda,hip,tpu,cpu}/<arch>/` — per-architecture
  translation units. Each file is wrapped in `namespace sg::<arch>` so
  the eight GPU arches' kernel and launcher symbols can coexist in one
  binary without link-time collisions.
- `csrc/common/` — shared headers (`platform.h`, `types.h`,
  `bindings.h`, `tuned_configs.h`, `quantization.h`, FP4 helpers).

Runtime architecture detection is centralized in
`csrc/bindings/dispatch.cpp` in `detect_arch()`, which returns one of
`{80, 89, 90, 100, 103, 120, 942, 950}` or raises an
`UnsupportedArchError`. There is no fallback chain. The Python helper
`grokking_optimizers.dispatch.get_gpu_arch()` mirrors the same contract
on the Python side.

For the filesystem layout in detail see `REFRESH.md §0`. For the
per-arch per-optimizer rundown (which kernel lives where, what is
arch-specific) see `REFRESH.md §24`.

## Testing

Run the full test suite with:

```
pytest tests/
```

Notable test files:

- `tests/test_cross_arch_agreement.py` — verifies math equivalence
  across every compiled-in architecture by cycling through
  `FORCE_ARCH=<n>`. This is the regression net against per-arch
  divergence.
- `tests/test_all_arches.py` — basic dispatch sanity per arch.
- `tests/test_amd_hip.py` — `gfx942` / `gfx950` specific paths.
- `tests/test_cutlass_parity.py` — CUTLASS GEMM output matches cuBLAS
  within FP tolerance. Skipped automatically when the build was made
  without `WITH_CUTLASS=1`.
- Per-optimizer suites such as `tests/test_supergrok2.py` cover
  optimizer-specific correctness (forward / backward equivalence,
  bilevel meta-learning, expert recycling, gradient checkpointing,
  edge cases, memory-leak checks).

## Contributing

To add a new kernel for an existing optimizer on an existing
architecture:

1. Write the per-arch source under
   `csrc/kernels/<lang>/<arch>/<your_kernel>.{cu,hip.cpp}` and wrap
   the file in `namespace sg::<arch>`.
2. Declare your launcher inside the `DECLARE_*(NS)` macro at the top
   of the matching `csrc/bindings/<optimizer>.cpp`. The macro is
   expanded once per supported architecture so every per-arch
   namespace's launcher is visible to the dispatcher.
3. Register the public Python-facing entry point in
   `csrc/bindings/module.cpp` if the optimizer does not already
   expose it.
4. Run `pytest tests/test_cross_arch_agreement.py` to make sure your
   new kernel agrees numerically with the other arches.

The bindings macro pattern (`SG_DISPATCH`, `SG_DISPATCH_CALL`,
`DECLARE_*`) is described in `REFRESH.md §22`. The list of currently
registered Python entry points is in `REFRESH.md §22.1`. Engineering
work that is still open (real per-arch divergence, Hopper FP8 fast
path, NVFP4 on Blackwell Ultra, autotune sweeps on hardware, CI
matrix) is tracked in `REFRESH.md §25`.

## License

This project is released under the MIT License. See the `LICENSE`
file for the full text.

Acknowledgements:

- The JAX and Pallas teams at Google for the TPU primitives that the
  `csrc/kernels/tpu/` path is built on.
- The NVIDIA CUTLASS team for the GEMM template library used in the
  Hopper- and Blackwell-class projection and Newton-Schulz paths
  (enabled with `WITH_CUTLASS=1`).
