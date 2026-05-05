# SuperGrok2

SuperGrok2 is a C++/CUDA/HIP optimizer suite for grokking-aware training of
large neural networks. It ships eleven optimizers spanning AdamW variants,
sign-momentum, sharpness-aware minimization, Newton-Schulz orthogonalization,
and a Mamba-3 + PEER + GRU meta-network optimizer (SuperGrok v2 — the
project's namesake). The grokking race driver (`grokking_race_v2.py`) compares
all eleven head-to-head on algorithmic learning tasks under controlled
conditions.

---

## Hardware support

The build targets a **3-arch active set** with fused (model, optimizer, arch)
instantiations. The "build for 3, expand winners" strategy means only arches
with real hardware validation are compiled; winners from the race get expanded
to future arches after the empirical comparison phase.

### Active arches (compiled, tested)

| Arch | Family | Cards | Backend |
|------|--------|-------|---------|
| `sm_90` | Hopper | H100, H200 | CUDA |
| `gfx942` | CDNA3 | MI300X, MI300A | HIP |
| `tpu_v5p` | TPU v5p | 128-wide MXU | JAX/Pallas |

### Future arches (scaffolded in dispatch, no kernels yet)

| Arch | Family | Notes |
|------|--------|-------|
| `sm_80` | Ampere | A100, A30, A10 |
| `sm_89` | Ada Lovelace | RTX 40, L40, L40S |
| `sm_100` | Datacenter Blackwell | B100, B200, GB200 |
| `sm_103` | Blackwell Ultra | B300, GB300 NVL72 |
| `sm_120` | Consumer Blackwell | RTX 50, RTX PRO 6000 |
| `gfx950` | CDNA4 | MI350X, MI355X |
| `tpu_v6e` | TPU v6e | 256-wide MXU |
| `x86_64` | CPU AVX-512 | Testing only |
| `ARM64` | CPU NEON | Testing only |

Anything else raises `UnsupportedArchError`. There is no tier fallback chain.

---

## Build status

Per-arch coverage of the 11 optimizers and 3 models. A cell is **done** when
the per-arch device template is wired through to a real implementation
(no `NotImplementedError`, no `TODO` body). The table reflects the state
after Phases 1–6 of the all-specialized rebuild.

### Optimizer × arch matrix

| Optimizer       | sm_90 (Hopper) | gfx942 (CDNA3) | tpu_v5p (JAX/Pallas) |
|-----------------|:--------------:|:--------------:|:--------------------:|
| SuperGrok2      | done           | done           | done                 |
| SuperGrok15     | done           | done           | done                 |
| SuperGrok11     | done           | done           | done                 |
| GrokAdamW       | done           | done           | done                 |
| NeuralGrok      | done           | done           | done                 |
| Prodigy         | done           | done           | done                 |
| Grokfast        | done           | done           | done                 |
| Lion            | done           | done           | done                 |
| LookSAM         | done           | done           | done                 |
| Muon            | done           | done           | done                 |
| MoE / Adam mt   | done           | done           | done                 |

### Model × arch matrix

The three race models — Decoder Transformer (causal LM), Vision Transformer
(ViT), and Mamba (selective SSM) — each have a forward + backward
specialization per arch. Phase 3 closed the sm_90 model row; Phase 5
closed the tpu_v5p model row.

| Model       | sm_90 (Hopper) | gfx942 (CDNA3) | tpu_v5p (JAX/Pallas) |
|-------------|:--------------:|:--------------:|:--------------------:|
| Decoder     | done (Phase 3) | done           | done (Phase 5)       |
| ViT         | done (Phase 3) | done           | done (Phase 5)       |
| Mamba       | done (Phase 3) | done           | done (Phase 5)       |

**sm_90 model implementation details**
- `csrc/kernels/cuda/sm_90/models/decoder.cuh`  (~1,180 LOC)
- `csrc/kernels/cuda/sm_90/models/vit.cuh`      (~1,345 LOC)
- `csrc/kernels/cuda/sm_90/models/mamba.cuh`    (~679 LOC)
- shared `attention.cuh` (~269 LOC) used by Decoder + ViT
- `mamba_scan_adapter.cuh` (~407 LOC) bridges Mamba to the Affine2x2 scan kernels

**tpu_v5p model implementation details**
- `csrc/kernels/tpu/_pallas_models.py` (~666 LOC) — shared Pallas/JAX
  `decoder_*`, `vit_*`, `mamba_*` forward/backward
- `csrc/kernels/tpu/_pallas_kernels.py` (~1,190 LOC) — tile-128
  affine prefix scan, fused GRU+PEER, VMEM-persistent expert MLP
- `csrc/kernels/tpu/v5p/__init__.py` re-exports the tile-128 variants and
  exposes `get_kernels(kind='optimizers'|'models')` per-version surface
- the `csrc/device/models/tpu_v5p/{transformer,vit,mamba}_tpu_v5p.py`
  device-template files re-export from `_pallas_models.py` so the
  per-arch dispatch path stays uniform across CUDA / HIP / TPU

### Bindings (`_ops.models.*` Python API)

`csrc/bindings/models_module.cpp` registers a `models` submodule on the
compiled extension so model entry points appear as `_ops.models.<name>`.
The shape mirrors the per-arch device templates and gives the race
driver a single CUDA/HIP launch site per (model, arch).

```python
from grokking_optimizers import _get_ops
_ops = _get_ops()

# Decoder Transformer (causal LM head, last-token logits)
_ops.models.decoder_forward(...)
_ops.models.decoder_backward(...)
_ops.models.decoder_attention_forward(...)   # component test surface
_ops.models.decoder_attention_backward(...)

# Vision Transformer (full attention, [CLS] classify head)
_ops.models.vit_forward(...)
_ops.models.vit_backward(...)
_ops.models.vit_attention_forward(...)
_ops.models.vit_attention_backward(...)
_ops.models.vit_patch_project(...)            # component test surface

# Mamba (selective SSM, last-token logits)
_ops.models.mamba_forward(...)
_ops.models.mamba_backward(...)
_ops.models.mamba_layer_forward(...)              # component test surface
_ops.models.mamba_selective_scan_forward(...)
_ops.models.mamba_selective_scan_backward(...)
```

Each function dispatches at runtime to the arch reported by
`grokking_optimizers.dispatch.get_arch_label()`. The `*_attention_*`,
`*_patch_project`, `*_layer_forward`, and `*_selective_scan_*` entries
are component-level test surfaces that the parity tests use to isolate
sub-kernels without running the full forward.

---

## Architecture

99 fused (model, optimizer, arch) compile-time instantiations from 42 device
headers (9 model + 33 optimizer). Three models × eleven optimizers × three
arches.

```
csrc/
├── device/
│   ├── optimizers/
│   │   ├── sm_90/      (11 .cuh headers)
│   │   ├── gfx942/     (11 .hip.cuh headers)
│   │   └── tpu_v5p/    (11 .py headers)
│   └── models/
│       ├── sm_90/      (3 .cuh headers)
│       ├── gfx942/     (3 .hip.cuh headers)
│       └── tpu_v5p/    (3 .py headers)
├── kernels/
│   ├── cuda/sm_90/models/   (decoder.cuh, vit.cuh, mamba.cuh, attention.cuh — sm_90 real impls)
│   ├── hip/gfx942/models/   (gfx942 model headers)
│   └── tpu/                 (_pallas_kernels.py, _pallas_models.py + v5p/__init__.py)
├── fused/
│   ├── sm_90/          (33 .cu TUs)
│   ├── gfx942/         (33 .hip.cpp TUs)
│   └── tpu_v5p/        (33 .py TUs)
├── bindings/           (pybind11 dispatchers + fused_step + models submodule)
└── common/             (types.h, platform.h, ptx_intrinsics.cuh, tuned_configs.h)
```

Each device header provides `__device__ __forceinline__` template functions.
Each fused TU includes one model header and one optimizer header, producing a
single compile-time instantiation. The namespace pattern is
`sg::device::<arch>` for templates and `sg::fused::<arch>` for instantiations.

Runtime dispatch via `grokking_optimizers/fused_dispatch.py`:
- `detect_arch()` → 90, 942, or "tpu_v5p"
- `dispatch_fused(model, optimizer, ...)` → routes to the compiled kernel
- Fallback to separate forward/backward/step when fused kernel unavailable

---

## Header file architecture

Every header in this project follows a layered design: platform abstractions
at the bottom, shared math primitives in the middle, per-arch kernel
implementations at the top, and binding/dispatch glue connecting everything to
Python. Below is an English description of what each header does and how it
fits into the overall system.

### Common layer (`csrc/common/`)

These headers are included by nearly every compilation unit across all three
backends. They define the portable abstractions that allow one codebase to
target NVIDIA, AMD, and Google hardware.

#### `platform.h`

Unified portability layer for CUDA and HIP. Detects the active backend at
compile time and provides conditional macros for warp size (32 on NVIDIA, 64
on CDNA), warp-shuffle intrinsics, fast sincos, streaming (non-temporal)
load/store via inline PTX or GCC builtins, error-checking macros, and
CUB/hipCUB namespace aliasing. Every GPU source file includes this as its
first dependency so that no downstream code needs backend-specific ifdefs.

#### `types.h`

Shared data structures and compile-time constants consumed by all forward and
backward kernels. Defines the `BatchedScanCtx` struct (14 members describing
sorted input segments and scan state layout for parallel Mamba), the
`Affine2x2` struct (2x2 matrix plus 2-vector for associative prefix scan
composition), and the inline `affine_combine` composition function used by
Blelloch scan. Also defines dimension caps (`MAX_D_STATE=128`,
`MAX_D_INNER=128`, `MAX_D_MODEL=64`), GRU and expert size limits, block size
constants (`PSCAN_BLOCK=512`), and branchless stochastic rounding helpers.

#### `utils.cuh`

Device-side utility functions shared across optimizer and model kernels.
Provides `warp_reduce_sum` (flexible warp-level butterfly reduction supporting
non-power-of-2 counts), `cluster_dsmem_reduce_sum` (Hopper DSMEM cooperative
cluster reduction with sm_80 fallback), `hash_prng` (deterministic Philox-like
PRNG keyed on step and element index), quantization helpers
(`float_to_bf16_stochastic`, `float_to_int8_stochastic`), and wrapped PTX
hot-path intrinsics (`fast_rsqrt_nr`, `ptx_fma`, `ptx_exp2`, `ptx_log2`,
`ptx_tanhf`, `ptx_sigmoidf`, `ptx_affine_combine`, `ptx_expert_mlp_forward`).
HIP builds receive fallback implementations using standard math functions.

#### `ptx_intrinsics.cuh`

Dedicated inline PTX intrinsics for the innermost hot loops of the forward
scan and GRU computation. Replaces multi-cycle standard-library calls with
single or few-cycle GPU instructions: `affine_combine_ptx` (12-FMA
composition for Blelloch scan), `softplus_ptx` (2-cycle approximation via
`ex2.approx` and `lg2.approx`), `fast_exp_ptx`, `stochastic_round_ptx`
(branchless FP32-to-INT rounding), and `gru_gates_ptx` (interleaved sigmoid
pair). Each has an HIP fallback using standard math so the same calling code
compiles on both backends.

#### `tuned_configs.h`

Architecture-specific launch configurations (block size, minimum blocks per
SM, GEMM tile dimensions, pipeline stages) auto-generated by the autotune
framework. Defines an `ArchId` enum covering all 8 supported architectures, a
`LaunchConfig` struct with 6 tuning fields, and per-optimizer lookup tables
indexed by architecture and problem-size bucket. When autotune has not been
run, all entries fall back to `DEFAULT_CONFIG`. The `autotune/tune.py` script
overwrites this file with measured winners.

#### `quantization.h`

Quantization utilities for projection GEMMs and expert weights. Supports six
precision modes (FP32, TF32, BF16, FP8_E4M3, INT8 symmetric, INT4 GPTQ, and
MXFP4 with shared exponents). CPU-side functions handle quantization and
packing; device-side functions handle dequantization during kernel execution.
Architecture-aware default selection picks the best precision based on
hardware capability.

#### `arch_tier.h`

Compile-time architecture tier classification. Maps preprocessor symbols
(`SG_ARCH_SM80`, `SG_ARCH_SM90`, etc.) to an `ArchTier` enum (GENERIC,
AMPERE, HOPPER, BLACKWELL, CDNA3, CDNA4). Kernels that need to pick between
`cp.async` (Hopper+), TMA (Blackwell+), or generic shared-memory loads use
this at compile time without runtime branching.

#### `fp4_helpers.hip.h`

HIP-specific FP4 (E2M1) and FP6 (E3M2) quantization helpers for CDNA4
(gfx950). Provides encode/decode functions for both formats, a 16-entry
dequantization lookup table in `__constant__` memory, and a simplified Philox
PRNG for stochastic rounding. Marked with `static __forceinline__` linkage to
avoid ODR violations across multiple HIP translation units.

### Bindings layer (`csrc/bindings/`)

These headers connect per-arch kernel launchers to the Python extension module.

#### `bindings.h`

Top-level arch detection interface. Declares `detect_arch()` (returns 90 or
942, or throws `UnsupportedArchError`) and forward-declares the `sg::sm90` and
`sg::gfx942` namespaces so per-optimizer dispatcher files can reference
specific launcher signatures without pulling in the full kernel headers.

#### `_dispatch_macro.h`

Arch-dispatch macros. `SG_DISPATCH(METHOD, ...)` calls `detect_arch()` and
switches to the appropriate `sg::<arch>::METHOD()` launcher. Eliminates the
8-way switch boilerplate that would otherwise be duplicated in every optimizer
dispatcher file. Also provides `SG_DISPATCH_CALL` for void-returning paths.

#### `_helpers.h`

CPU-side helper functions shared by multiple optimizer dispatchers: gradient
L2-norm clipping (accumulate, sync, scale) and SAM gradient-norm computation.
These operate on `std::vector<torch::Tensor>` and run on the host CPU
(PyTorch operations), called before or after per-arch kernel launches.

### CUTLASS integration (`csrc/kernels/cuda/`)

#### `_cutlass_gemm.cuh`

CUTLASS-based GEMM wrappers for Hopper+ architectures, gated behind
`-DWITH_CUTLASS`. Provides `cutlass_gemm_fp16` and `cutlass_gemm_bf16` (row-
major A*B with FP32 accumulation and output), a `LinearCombinationSoftplusBias`
epilogue functor for fusing softplus activation into the GEMM tail, a
lightweight `cutlass_softplus_bias_kernel` post-pass, and convenience wrappers
(`cutlass_dt_proj_fused_with_bias`) for the Mamba dt_proj operation. Produces
a hard compile error if included without the CUTLASS flag, catching misuse
early.

### SM90 optimizer kernels (`csrc/kernels/cuda/sm_90/`)

Full-featured Hopper implementations. Each header defines `__global__` kernels
templated on parameter/state/gradient dtype, plus host-side launcher functions
that select block size from `tuned_configs.h` and dispatch the kernel.

#### `adamw.cuh`

Standard AdamW with decoupled weight decay. Provides a scalar grid-stride
kernel and a float4-vectorized fast path for all-FP32 instantiations. Uses
non-temporal loads for read-once state, `fast_rsqrt_nr` for the adaptive
learning rate denominator, and `hash_prng` for BF16 stochastic rounding.
Launcher caps grid at 8192 blocks.

#### `grokadamw.cuh`

Fused Grokfast + AdamW in a single pass. Implements gradient clamping, slow-
EMA filter, learnable amplification, and Adam update without intermediate
writes. Includes three kernel variants: scalar, float4-vectorized, and a Q3
quantized path (INT8 `exp_avg` with per-block scale maintained via warp
shuffles). Multi-tensor overloads support batched launches across all model
parameters in one dispatch.

#### `grokfast.cuh`

Two operational modes: EMA-only (updates EMA state and writes amplified
gradient back for downstream Adam) and fully-fused (EMA + Adam in one kernel
with the amplified gradient kept register-resident). Both have scalar and
vec4 variants. Stochastic rounding prevents small EMA delta collapse on BF16
state.

#### `lion.cuh`

Sign-based momentum optimizer (no variance buffer). Computes interpolation of
momentum and gradient, extracts sign, applies weight decay. Provides per-
tensor and multi-tensor kernel variants; the multi-tensor path uses a constant-
memory offset table with linear search (faster than binary search for typical
parameter counts under 96 tensors).

#### `looksam.cuh`

Sharpness-Aware Minimization with periodic perturbation (every k steps
instead of every step). Four distinct kernels: parameter perturbation by
rho-scaled gradient, restoration from backup, direction adjustment via SAM-
vs-normal gradient difference, and norm reduction. The norm reduction uses
`cluster_dsmem_reduce_sum` on Hopper for cross-CTA communication without
global memory round-trips.

#### `muon.cuh`

Matrix-manifold optimizer using Frobenius-normalized momentum and Newton-
Schulz orthogonalization. Orchestrates five kernels (Frobenius reduction,
momentum normalize, NS combine, NS combine-apply, parameter update) plus
CUTLASS BF16 GEMMs for the Newton-Schulz iterations. Includes a 32x32
shared-memory transpose kernel and an FP8 fast-path stub for large matrices.

#### `neuralgrok.cuh`

Learned per-element amplifier via 2-layer MLP. The meta-psi kernel is
templated on hidden width H and fully unrolled; weights live in `__constant__`
memory (budget 2048 floats for H<=256). The apply kernel fuses amplified-
gradient Adam in one pass. Supports CUDA Graph caching keyed by a shape/
dtype/hyperparameter signature to avoid re-launch overhead.

#### `prodigy.cuh`

Self-tuning Adam that estimates its own learning rate from cumulative
parameter-space distance. Three-kernel orchestration: (1) block-reduce the
two partial sums (r, s) using warp shuffles + SMEM tree + Hopper cluster
reduction, (2) single-thread device-side scalar update of d_t, (3) fused Adam
step consuming d_t from device memory (no host sync). Fully asynchronous—the
adaptive learning rate never leaves the GPU.

#### `supergrok11.cuh`

SuperGrok v1.1 with cosine-similarity gating. Two cooperative grid-wide
sweeps: sweep A computes the meta-MLP forward, cosine gate, and sharpness
reduction (using a last-block-finished atomic pattern), then sweep B performs
gated gradient mixing, Adam, and trust-ratio scaling via cooperative grid
sync. Meta-net weights reside in `__constant__` memory. Uses PTX FMAs and
`fast_exp_ptx` for the hot path.

#### `supergrok15.cuh`

SuperGrok v1.5 with per-coordinate alpha gate (clipped affine of MLP output,
no sigmoid). Similar cooperative two-sweep structure to v1.1 but with a
simpler single-scalar reduction. Emphasizes register locality: the amplified
gradient (smart_grad) stays in registers from computation through the Adam
update without spilling to shared or global memory.

#### `supergrok2_fwd.cuh`

SuperGrok v2 forward pass: the full Mamba-3 + PEER + GRU + apply pipeline.
Routes by problem size: sequential scan for N<256, Blelloch parallel scan for
256-1024, bilevel-checkpointed scan for N>=1024. Uses CUTLASS WGMMA + TMA for
projection GEMMs, Affine2x2 parallel prefix scan with shared-memory state,
the fused softplus CUTLASS epilogue for dt_proj, and CUDA Graph capture for
the entire multi-kernel pipeline. Hopper FP8 fast-path activates when all
dimensions exceed 64.

#### `supergrok2_bwd.cuh`

SuperGrok v2 backward pass. Reverse-time Mamba-3 scan with checkpoint
recomputation (saves state every C steps, recomputes intermediates from
nearest checkpoint). Gradient accumulation for scan weights uses per-element
shared-memory atomics or warp-reduced buffers depending on dimension.
CUTLASS GEMM-T handles B/C projection weight gradients. The hot loop is
macro-encoded so one code path handles both checkpoint and no-checkpoint
modes.

#### `supergrok2_warp_specialized.cuh`

Warp-specialized SSM scan kernels exploiting Hopper's 4 hardware warp
schedulers. Producer warp (warp 0) issues global memory loads into double-
buffered shared memory; consumer warps (warp 1+) run the selective scan
recurrence. Atomic phase flags synchronize handoff. Two variants: generic
d_state, and d_state=16 with fully unrolled RoPE pairs in consumer registers.
Host launchers activate this path when uniform d_state is detected.

### SM90 model kernels (`csrc/kernels/cuda/sm_90/models/`)

Complete forward and backward implementations for the three race models.

#### `decoder.cuh`

Autoregressive Decoder Transformer (post-norm, causal self-attention). Defines
per-layer weight/activation pointer layouts, and specialized kernels for
embedding lookup, fused residual + layer normalization, GELU activation, bias
addition, QKV reshaping, and last-token extraction. Linear projections route
through cuBLAS GEMMs. Multi-layer forward pass reuses a fixed scratch buffer;
backward carefully orders operations to minimize peak memory.

#### `vit.cuh`

Vision Transformer for image classification. Handles patch projection (unfold
+ linear), CLS token prepend, learned positional embedding addition, multi-
head non-causal attention (delegates to `attention.cuh` with kCausal=false),
and a classification head. Same fused residual + layernorm and cuBLAS GEMM
patterns as the decoder, adapted for the fixed 17-token sequence length (16
patches + CLS).

#### `mamba.cuh`

Selective State-Space Model (Mamba) with depthwise convolution and gating.
Orchestrates embedding, per-layer SSM blocks (input projection, 1D depthwise
conv + SiLU, state-space scan via adapter, gating, output projection), and
residual + layer normalization. Delegates the selective scan to
`mamba_scan_adapter.cuh`. Provides component-test wrappers
(`selective_scan_fwd`/`bwd`) for isolated sub-kernel debugging.

#### `attention.cuh`

Shared attention kernel used by both Decoder and ViT. Computes QK^T scores in
shared memory, applies row-wise softmax (with optional causal mask), and
multiplies by V. Designed for the tiny sequence lengths in the grokking tasks
(4 tokens for decoder, 17 for ViT) where full-matrix SMEM computation
outperforms block-tiled FlashAttention. Stores softmax log-sum-exp for the
backward pass. Template parameter switches between causal and non-causal
modes.

#### `mamba_scan_adapter.cuh`

Thin adapter bridging model-level Mamba parameters (x, dt, A_log, B, C) to
the Affine2x2 representation used by the shared scan infrastructure. Packs
discretized dynamics (A_bar, B_bar from ZOH discretization), dispatches to
sequential scan (register-based, for N<256) or parallel Blelloch scan (for
larger N), and computes adjoint gradients for the backward pass. Reuses
Affine2x2 primitives from the SuperGrok v2 device templates.

### GFX942 optimizer kernels (`csrc/kernels/hip/gfx942/`)

HIP ports for MI300X. These files route through the host compiler (not
hipcc), so they contain no `__global__` kernels or `<<<>>>` launch syntax.
Instead they declare launcher function signatures that the corresponding
`.hip.cpp` translation units implement using ATen tensor operations (which
route through rocBLAS for GEMMs and HIP kernels internally via PyTorch).

#### `_common.hip.h`

Common includes for all gfx942 headers (PyTorch extension headers, standard
library utilities). Serves as a build-environment marker ensuring consistent
compilation flags across all HIP files.

#### `adamw.hip.h`

Declares `launch_fused_adamw_simple` in `sg::gfx942` namespace. The
implementation uses ATen element-wise operations (add, mul, addcmul, lerp)
that PyTorch's HIP backend maps to rocBLAS/hipBLAS as appropriate.

#### `grokadamw.hip.h`

Declares `launch_fused_grokadamw_step` and multi-tensor overload. Mirrors the
sm_90 math (EMA filter + amplification + Adam) using ATen operations.

#### `grokfast.hip.h`

Declares EMA-only and fused EMA+Adam launchers. Implementation applies the
same amplification math via PyTorch tensor arithmetic.

#### `lion.hip.h`

Declares per-tensor and multi-tensor Lion launchers. Both by-reference and
by-value overloads provided for binding compatibility.

#### `looksam.hip.h`

Declares perturb, restore, direction adjust, and norm reduce launchers.
Direction adjustment uses PyTorch operations for the SAM gradient difference.

#### `muon.hip.h`

Declares momentum normalize, NS combine, NS combine-apply, and parameter
update launchers. Newton-Schulz iterations are orchestrated host-side using
`torch::mm` (which routes to rocBLAS on HIP). No CUTLASS dependency.

#### `neuralgrok.hip.h`

Declares meta-psi (MLP forward) and apply (amplified Adam) launchers.
Weights passed as tensors rather than constant memory on HIP.

#### `prodigy.hip.h`

Declares reduce, d-update, and apply launchers. The d_t scalar stays on-
device via a 1-element tensor.

#### `supergrok11.hip.h` / `supergrok15.hip.h`

Declare sweep-A and sweep-B launchers matching the sm_90 cooperative pattern,
implemented via sequential ATen calls (cooperative launch not available on
HIP without explicit stream synchronization).

#### `supergrok2_fwd.hip.h` / `supergrok2_bwd.hip.h`

Runtime-throwing stubs. The full Mamba+GRU+PEER pipeline requires sm_90-
specific cluster and WGMMA features; a complete gfx942 port would be a multi-
week effort. These stubs link cleanly so non-SG2 workflows run on MI300X; SG2
calls raise a descriptive runtime error.

#### `supergrok2_warp_specialized.hip.h`

Runtime-throwing stub. Warp specialization relies on Hopper's 4 independent
warp schedulers which have no CDNA3 equivalent.

### GFX942 model kernels (`csrc/kernels/hip/gfx942/models/`)

Thin delegation wrappers. Since the model kernels use cuBLAS GEMMs and
element-wise kernels (no sm_90-specific intrinsics), the HIP ports alias the
sm_90 implementations. PyTorch's hipify pipeline handles the cuBLAS-to-
rocBLAS and dtype remapping automatically.

#### `decoder.hip.h` / `vit.hip.h` / `mamba.hip.h`

Each declares a `ModelConfig` struct and inline `forward`/`backward` functions
that delegate directly to the sm_90 model implementations. No reimplementation
needed — the automatic hipification of cuBLAS calls, `__nv_bfloat16` to
`__hip_bfloat16`, and stream types handles the platform gap.

#### `attention.hip.h` / `mamba_scan_adapter.hip.h`

Matching delegation headers for shared sub-components used by the model
wrappers above.

### Device-function templates (`csrc/device/`)

Per-element `__device__ __forceinline__` templates that fused TUs inline into
their kernels. These contain no `__global__` functions — they are pure
computation designed to be called from within a grid-stride loop.

#### `csrc/device/optimizers/sm_90/*.cuh`

One file per optimizer (adam, grokadamw, grokfast, lion, looksam, moe, muon,
neuralgrok, prodigy, supergrok11, supergrok15, supergrok2). Each provides a
per-element step function (`*_element`) that computes the complete optimizer
math for one parameter coordinate. The `supergrok2_sm90.cuh` template also
includes device-level scan recurrence steps, RoPE rotation, and bilevel
precomputation used by `mamba_scan_adapter.cuh`.

#### `csrc/device/optimizers/gfx942/*.hip.cuh`

Mirror of the sm_90 device templates for CDNA3. Same function signatures and
math, using HIP-compatible intrinsics (wave reduction instead of warp
shuffle, `__hip_bfloat16` instead of `__nv_bfloat16`).

#### `csrc/device/models/sm_90/*.cuh`

Stub device-function templates for future warp-specialized model operations
(transformer, vit, mamba forward/backward). Currently placeholder — the real
implementations live in `csrc/kernels/cuda/sm_90/models/`.

#### `csrc/device/models/gfx942/*.hip.cuh`

Thin delegation to sm_90 model device templates via namespace aliasing.

### TPU/Pallas layer (`csrc/kernels/tpu/`)

JAX/Pallas implementations of the same algorithms, targeting TPU v5p's 128-
wide MXU.

#### `_pallas_kernels.py`

Core Pallas kernel implementations: tile-128 affine prefix scan (Mamba-3),
tile-256 variant for v6e, fused scan + output projection + Adam, VMEM-
persistent expert MLP, fused GRU + PEER routing, sharded multi-device scan
via `shard_map`, and runtime dispatch helpers that select the right tile width
based on detected TPU version.

#### `_pallas_models.py`

Model forward/backward implementations in JAX: decoder (causal transformer),
ViT (patch project + non-causal attention), and Mamba (selective scan). Uses
splash-attention with BF16 fallback. All paths wrapped in try/except so the
module stays importable when the Pallas API drifts.

#### `v5p/__init__.py`

Per-version surface that re-exports tile-128 variants from the shared
kernel/model files and exposes `get_kernels(kind='optimizers'|'models')` for
uniform dispatch.

---

## Installation

```bash
git clone https://github.com/peterc04/SuperGrok1.5
cd SuperGrok1.5
git submodule update --init --recursive third_party/cutlass
bash build.sh
```

### Build modes

| Mode | Effect |
|------|--------|
| `./build.sh` | Default ninja-backed release build. Single-arch: `-gencode arch=compute_90,code=sm_90` + PTX embed. |
| `./build.sh --autotune` | Two-pass: stub-config build, sweep grids via `autotune/tune.py`, write winners to `tuned_configs.h`, rebuild. |
| `./build.sh --debug` | `CUDA_DEBUG=1`, `-G -O0 -lineinfo`, fast-math disabled. |
| `./build.sh --profile` | Release build + `ncu --set full` profile capture. |
| `./build.sh --package` | Build + stage redistributable `dist/` tree. |
| `./build.sh --package-tarball` | `--package` + `supergrok2-3.0.0-<sha>.tar.gz`. |

### Compiler flags

- nvcc: `-O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -lineinfo -Xptxas -O3 --warn-on-spills`
- nvcc gencode: `-gencode arch=compute_90,code=sm_90` + PTX embed for forward-compat
- hipcc: `--offload-arch=gfx942 -O3 -std=c++17 -ffast-math`

### ccache / sccache

Auto-detected via `CMAKE_*_COMPILER_LAUNCHER`. Ninja routes compilation
through the launcher transparently.

### CUTLASS (opt-in)

`WITH_CUTLASS=1 ./build.sh` enables CUTLASS-backed GEMM paths on sm_90.
Requires `git submodule update --init --recursive third_party/cutlass`.
Adds `-DCUTLASS_NVCC_ARCHS=90a` and CUTLASS include dirs.

### Pre-built distribution

`bash build.sh --package-tarball` produces a redistributable tarball with
three documented install paths in `dist/INSTALL.md` (drop into PYTHONPATH,
copy to site-packages, or build a wheel). No source compilation needed on the
consumer side.

---

## Quickstart

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

---

## Optimizers

Eleven total. Each entry: purpose, state per param, hyperparameters, fused kernel name.

### SuperGrok v2 (`supergrok2.py`)

Flagship. Mamba-3 + 4-head PEER + per-element GRU + 144-expert pool, per-element learned gradient correction, on top of Adam with SAM and bilevel meta-learning.
- State: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`, `gru_states`, `mamba_fwd_states`, `mamba_bwd_states`
- Key hyperparams: lr=1e-3, betas=(0.9, 0.999), d_model=8, d_state=16, num_experts=144, sam_rho=0.05
- Fused kernel: `_ops.supergrok2_prepare_and_batched_step`
- Python fallback: full

### SuperGrok v1.5 (`supergrok15.py`)

Simpler v2. Replaces Mamba+PEER+GRU with a 2-input 2-layer MLP.
- State: `exp_avg`, `exp_avg_sq`, `mus`, `sharpness`
- Key hyperparams: lr=1e-3, betas=(0.9, 0.98), hidden_dim=32, sam_rho=0.05
- Fused kernel: `supergrok15_fused_step`

### SuperGrok v1.1 (`supergrok11.py`)

v1.5 with cosine-similarity gating instead of sigmoid-on-accuracy.
- Fused kernel: `supergrok11_fused_step`
- Reduction: `cosine_gate_reduce_kernel`

### GrokAdamW (`grokadamw.py`)

AdamW with EMA gradient filter and persistent-direction amplification.
- State: `exp_avg`, `exp_avg_sq`, `ema`
- Key hyperparams: lr=1e-3, alpha=0.98, lamb=5.0
- Fused kernel: `grokadamw_fused_step`
- Quantized variant: `_q3` kernel (INT8 + BF16 stochastic-rounded)

### NeuralGrok (`neuralgrok.py`)

AdamW with learned MLP amplifier on |grad|.
- Key hyperparams: alpha=10.0, beta=4.0, num_layers=3, hidden_dim=128
- Fused kernel: `neuralgrok_fused_step`

### Prodigy (`prodigy.py`)

Self-tuning Adam. Estimates `d_lr` from cumulative parameter-space distance. Set lr=1.0.
- State: `exp_avg`, `exp_avg_sq`, `s`, `param_init`
- Fused kernel: `prodigy_fused_step`
- Reduction: `prodigy_dlr_reduce_kernel`

### Grokfast (`grokfast.py`)

Simplest grokking-aware AdamW. EMA + amplification.
- State: `ema`, `exp_avg`, `exp_avg_sq`
- Key hyperparams: grokfast_alpha=0.98, grokfast_lamb=2.0
- Fused kernel: `grokfast_fused_ema_adam_step`

### Lion (`lion.py`)

Sign-based Adam alternative (EvoLved Sign Momentum).
- State: `exp_avg`
- Key hyperparams: lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0
- Fused kernel: `lion_fused_step`
- Python fallback: yes

### LookSAM (`looksam.py`)

AdamW with periodic SAM (every k steps) instead of every-step SAM.
- State: `exp_avg`, `exp_avg_sq`, `sam_direction`
- Key hyperparams: rho=0.05, k=5, alpha=0.7
- Python fallback: yes

### Muon (`muon.py`)

Dual optimizer. Newton-Schulz orthogonalization for 2D weights, AdamW for 1D.
- State (2D): `momentum_buffer`; State (1D): `exp_avg`, `exp_avg_sq`
- Key hyperparams: lr=0.02, momentum=0.95, ns_steps=5
- Fused kernels: `muon_fused_step` (2D), `fused_adamw_simple_step` (1D)
- Python fallback: yes

### Mamba3PEERMetaNet (`mamba3_peer_metanet.py`)

Meta-net used internally by SuperGrok v2; not a standalone optimizer.
Submodules: `Mamba3ScanBlock`, `MiniGRU`, PEER router, expert MLP pool.

---

## Race fairness model

The grokking race uses four outer train/test splits (10/90, 25/75, 50/50,
80/20) with an inner val carve-out controlled by `val_ratio` (default 0.10;
auto-overrides to 0.05 on 10/90). A fixed early-stopping rule ends each run
at whichever comes first: test accuracy reaching 95% or step count reaching
20,000 — identical across all 11 optimizers. Three SG variants (v2, v1.5,
v1.1) consume the inner val for bilevel and meta updates; the other eight
train on train only. The `val_test_gap` in output is the key diagnostic for
meta-learning vs masked overfitting.

---

## Grokking race driver

The 11-optimizer, 3-model, 4-split grokking race lives in
`grokking_race_v2.py` (1700+ lines). Full sweep: 11 × 3 × 4 × 5 seeds =
660 runs.

### Purpose

Compare 11 optimizers head-to-head on three algorithmic grokking tasks across
four train/test splits, holding initialization, model architecture, eval
cadence, early-stopping rule, and seed bands fixed.

### Tasks

Three tasks, each on integers mod p=97:

- **Decoder — modular division.** `(a · b⁻¹) mod p`, 4-token sequence, 9312 examples.
- **ViT — MNIST-addition.** `(a + b) mod p` from digit images, 16 patches of dim 49, 9409 examples.
- **Mamba — sequential chained division.** Length-3 chain `a / b₁ / b₂ / b₃ mod p`, 9312 examples.

### Models

- **Transformer** (decoder task) — 2-layer causal attention + FFN. ~430K params (small).
- **ViT** (ViT task) — Patch projection, [CLS] token, full-attention encoder.
- **MambaModel** (mamba task) — Stacked SelectiveSSMLayer blocks with 1D depthwise conv.

Three sizes: small (128-dim), medium (256-dim), large (512-dim).

### Train/val/test split

| frac_train | val_ratio | train | val | test |
|------------|-----------|-------|-----|------|
| 0.10 | 0.05 (auto) | 884 | 46 | 8382 |
| 0.25 | 0.10 | 2095 | 232 | 6985 |
| 0.50 | 0.10 | 4190 | 466 | 4656 |
| 0.80 | 0.10 | 6704 | 745 | 1863 |

### Eval cadence and early stopping

`_eval_log` every `eval_every` steps (default 100). `EarlyStopper` triggers on:
- `test_acc >= 0.95` for `patience` consecutive checks, OR
- `step >= 20,000`

Records `stopping_reason` ("test_acc_threshold" or "max_steps"), `stopping_step`.

### Run modes

- **A** — Single arch × single split. 6 seeds.
- **B** — Multi-split. One arch × 4 splits × 5 seeds.
- **C** — Arch comparison. 3 arches × one split × 5 seeds.
- **D** — Full sweep. 3 arches × 4 splits × 5 seeds = 660 runs.
- **E** — Scale comparison. 3 model scales × 5 seeds.

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `--setup` | off | Install deps and exit |
| `--gpus` | None | Multi-GPU IDs ("0,1,2,3" or "auto") |
| `--ntfy` | None | ntfy.sh topic for push notifications |
| `--port` | 8080 | HTTP status server port |
| `--no-status-server` | off | Disable status server |
| `--val-ratio` | None (0.10) | Val carve-out fraction |
| `--early-stop-test-acc` | 0.95 | Test accuracy threshold |
| `--early-stop-max-steps` | 20000 | Hard step cap |
| `--eval-every` | 100 | Eval frequency |
| `--optimizers` | all | Comma-separated subset |
| `--seeds` | default | Comma-separated seed list |
| `--num-seeds` | 5 | First N from default seeds |
| `--tasks` | all | Comma-separated subset |
| `--train-test-ratios` | all | "10/90,25/75,50/50,80/20" |
| `--output` | "results" | Output directory |
| `--no-fused` | off | Disable fused kernel dispatch |

### Output schema

```json
{
  "_meta": {"total_wall": float, "model_type": str, "frac_train": float},
  "<optimizer>": [
    {
      "seed": int,
      "steps": [int], "train_losses": [float], "train_accs": [float],
      "val_losses": [float], "val_accs": [float],
      "test_losses": [float], "test_accs": [float],
      "wall_time": float, "total_steps": int,
      "grokking_step": int|null, "grokking_wall": float|null,
      "final_train_acc": float, "final_val_acc": float,
      "final_test_acc": float, "val_test_gap": float,
      "stopping_reason": "test_acc_threshold"|"max_steps",
      "stopping_step": int, "val_ratio": float
    }
  ]
}
```

Plus PNG plots: curves, race bars, split comparison, architecture heatmaps.

### Multi-GPU mode

`--gpus` ≥ 2 devices → `mp.set_start_method("spawn")`, one worker per GPU,
tasks distributed via `MPQueue` with poison-pill termination. Fast GPUs
naturally pick up more work.

### Status server and notifications

Daemon HTTP on `--port`, serves JSON progress at `GET /`. ntfy.sh integration
for push notifications on start, grok events, errors, and completion.

### Sanity tests

`tests/test_race_split.py` (10 sections): split arithmetic, disjointness,
determinism, auto-override, EarlyStopper, TrainResult schema.

---

## Algorithms

### Affine2x2 Mamba encoding
Mamba-3 recurrence encoded as 2×2 affine maps (6 floats per element).
Associative → eligible for parallel prefix scan.

### 12-FMA composition
Inline PTX `affine_combine_ptx`: 8 FMAs for matrix product + 4 for bias = 12
total, arranged in 3 waves of 4 for pipeline utilization. ~10 cycles.

### Blelloch parallel prefix scan
Two-phase: up-sweep combines pairs at doubling strides, down-sweep distributes
exclusive prefixes. O(N) work, O(log N) depth.

### Bilevel checkpointing
Checkpoint every C-th scan state, recompute intermediates from nearest
checkpoint. Memory savings ~(C-1)/C. Default C=1 (full save), tunable.

### Register-resident smart_grad
Amplified/orthogonalized gradient held in CUDA register, immediately consumed
by Adam update in same kernel. ~50% bandwidth reduction.

### Non-temporal I/O
PTX `ld.global.nc` / `st.global.wt` (CUDA) or `__builtin_nontemporal_*` (HIP).
Bypasses L2 for read-once optimizer state.

### PTX hot-path intrinsics
`ex2.approx` (1 cycle), `lg2.approx` (1 cycle), `rcp.approx` (1 cycle),
`softplus_ptx` (2 cycles), `gru_gates_ptx` (interleaved sigmoid pair),
`stochastic_round_ptx` (branchless). 1-2 ULP error.

### Warp-shuffle reductions
`__shfl_down_sync` butterfly (CUDA) or wave-reduction (HIP). Per-warp atomic
to global accumulator. Avoids shared memory bottlenecks.

### Product-key PEER routing
Score N elements against E experts in O(√E). Split query into halves, top-K
each against √E sub-keys, outer product candidates.

### Cooperative shared-memory weight loading
All threads load disjoint weight slices into shared memory, one
`__syncthreads()`, then per-thread access at ~5 cycle latency.

---

## JAX/TPU

Functional rewrite of the suite (~300 lines core logic vs ~2000 CUDA).

### Modules in `supergrok2_jax_tpu/`

- `supergrok2_jax.py` — main optimizer loop
- `mamba3_peer_metanet_jax.py` — meta-net architecture
- `scan.py` — `jax.lax.associative_scan` with Affine2x2 operator
- `gru.py`, `peer.py`, `bilevel.py`
- `simple_optimizers_jax.py` — GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM
- `metanet_optimizers_jax.py` — SuperGrok v1.5, v1.1, NeuralGrok
- `quantization_jax.py`, `sharding.py`, `bridge.py`

### Pallas kernels (`csrc/kernels/tpu/_pallas_kernels.py`)
- `mamba3_scan_pallas_tile128` — affine prefix scan, 128-wide MXU tiles (v4/v5e/v5p)
- `mamba3_scan_pallas_tile256` — same algorithm, 256-wide tiles for v6e
- `pallas_persistent_scan_fused_elem_tile{128,256}` — fused
  scan + output projection + Adam, scan output stays in VMEM
- `vmem_persistent_expert_mlp` — expert weights pinned in VMEM with
  `eviction_policy="none"` on v5p / v6e
- `pallas_fused_gru_peer` — GRU + PEER routing + expert MLP fused in one kernel
- `sharded_mamba3_scan` — multi-device 3-phase prefix scan via
  `shard_map` (local scan → summary all-gather → correction)
- `tpu_auto_select_scan` / `tpu_auto_select_fused_scan_elem` — runtime
  dispatch keyed on `detect_tpu_version()`

### Pallas models (`csrc/kernels/tpu/_pallas_models.py`)
- `decoder_forward` / `decoder_backward` (causal Decoder Transformer)
- `vit_forward` / `vit_backward` / `vit_patch_project`
- `mamba_forward` / `mamba_backward` / `mamba_layer_forward` /
  `mamba_selective_scan`
- `attention_forward` / `attention_backward` — splash-attention path
  with hand-tiled BF16 fallback

All Pallas paths are wrapped in `try`/`except` with a pure-JAX fallback
so the module remains importable when the Pallas API drifts.

### Feature gaps vs CUDA
- SAM perturbation not fully integrated into the JAX optimizer harness
- Quantization INT8 only (no FP8/INT4/MXFP4)
- Expert load balancing minimal

---

## Tests

Eleven files, ~3,400 LOC, ~100 test points.

| File | Focus |
|------|-------|
| `test_supergrok2.py` | 27 sections covering scan, forward, bilevel, recycling, checkpointing, edge cases, memory, dispatch, quantization, distributed |
| `test_matrix.py` | Cross-platform correctness (10 optimizers × 5 steps) |
| `test_all_arches.py` | Dispatch sanity for sm_90, gfx942 |
| `test_cross_arch_agreement.py` | Elementwise allclose across FORCE_ARCH (tolerance 1e-4) |
| `test_cutlass_parity.py` | CUTLASS GEMM vs reference (sm_90 only, skipped without WITH_CUTLASS=1) |
| `test_cpu_fallback.py` | Python fallback validation, import sanity |
| `test_jax_matrix.py` | JAX optimizer correctness |
| `test_amd_hip.py` | gfx942 paths, precision config, arch detection |
| `test_new_features.py` | float4 vectorized, OverlappedOptimizer, compression, Pallas |
| `test_training_aware.py` | Non-temporal I/O, Q3 states, stochastic rounding, pipelining |
| `test_race_split.py` | Split arithmetic, disjointness, determinism, EarlyStopper |
| `test_models_sm_90.py` | Smoke test that `_ops.models.{decoder,vit,mamba}_forward/backward` are registered (no kernel run) |

Run: `pytest tests/`

---

## Per-arch per-optimizer rundown

For each optimizer, what each active arch does. Only sm_90, gfx942, and
tpu_v5p are currently compiled; other arches will be added post-expansion.
After Phases 1–6 every cell in the optimizer × arch and model × arch
matrices above has a real implementation — see the
[Build status](#build-status) tables.

### SuperGrok v2

**sm_90 (Hopper):** Canonical math + FP8 fast path for projection matmuls
(gated behind dim ≥ 64 check). Warp-specialized scan kernels declared but
unwired from canonical batched-step. CUTLASS sm_90a optional. 228 KB smem,
~80 regs/thread.

**gfx942 (CDNA3):** Canonical math + BF16 MFMA fast path. rocBLAS for
matmuls. 64 KB LDS, ~64 vGPRs. MI300X 192 GB HBM.

**tpu_v5p:** JAX-Pallas tiled scan for 128-lane MXU. BF16 throughout. XLA
fusion handles softplus epilogue.

### SuperGrok v1.5 / v1.1

**sm_90:** Register-resident smart_grad in fused full-step kernel. Hopper FP8
projection path available.

**gfx942:** BF16 MFMA for the MLP meta-net forward. Wave-reduction for
cosine gate (v1.1).

**tpu_v5p:** Pure JAX functional implementation.

### GrokAdamW

**sm_90:** Fused step + vec4 variant + Q3 quantized variant (INT8/BF16).

**gfx942:** Same math, BF16 accumulation for EMA filter.

**tpu_v5p:** JAX implementation in `simple_optimizers_jax.py`.

### Muon

**sm_90:** Newton-Schulz with CUTLASS mm (optional) or cuBLAS. FP8 scale
helper for spectral normalization. `neg_lr_scale = -lr * 0.2 * sqrt(max_dim)`.

**gfx942:** rocBLAS for matmuls. BF16 MFMA available for the NS combine step.

**tpu_v5p:** JAX orthogonalization via `jnp.linalg` + custom NS iteration.

### Lion / Grokfast / LookSAM / Prodigy / NeuralGrok

All follow the same pattern: fused element-wise kernel on sm_90 and gfx942,
JAX functional on tpu_v5p. Arch-specific divergence is minimal (vec4
vectorization on CUDA, wave-width adaptation on HIP).

---

## Quick reference

### Optimizer feature matrix

| Optimizer | Meta-net | State tensors | SAM | Bilevel | Fused kernel | Fallback |
|-----------|----------|---------------|-----|---------|--------------|----------|
| SuperGrok2 | Mamba3+PEER+GRU | 7 | ✓ | ✓ | ✓ | ✓ full |
| SuperGrok15 | MLP 2-layer | 4 | ✓ | ✓ | ✓ | ✗ |
| SuperGrok11 | MLP + cosine gate | 4 | ✓ | ✓ | ✓ | ✗ |
| GrokAdamW | EMA filter | 3 | ✗ | ✗ | ✓ | ✗ |
| NeuralGrok | Learned MLP | 2 | ✗ | ✗ | ✓ | ✗ |
| Prodigy | distance-aware | 4+init | ✗ | ✗ | ✓ | ✗ |
| Grokfast | EMA amplify | 3 | ✗ | ✗ | ✓ | ✗ |
| Lion | momentum | 1 | ✗ | ✗ | ✓ | ✓ |
| LookSAM | periodic SAM | 3 | ✓ | ✗ | ✓ | ✓ |
| Muon | NS ortho 2D | 1-3 | ✗ | ✗ | ✓ | ✓ |

### Compile-time constants (csrc/common/types.h)

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_D_STATE` | 128 | Scan state dim cap |
| `MAX_D_INNER` | 128 | Mamba inner dim cap |
| `MAX_D_MODEL` | 64 | Projection dim cap |
| `MAX_GRU_HIDDEN` | 8 | GRU hidden cap |
| `MAX_EXPERT_HIDDEN` | 16 | Expert MLP cap |
| `PSCAN_BLOCK` | 512 | Blelloch threads/block |
| `PSCAN_THRESHOLD` | 256 | Sequential vs parallel scan switch |

### Architecture policy

No-fallback. `detect_arch()` returns 90, 942, or "tpu_v5p" or raises
`UnsupportedArchError`. No tier walking. Missing arch = build failure or
runtime error.

### Precision auto-selection chain

nvfp4 → mxfp4 → fp8 → bf16 → fp32

---

## Engineering work remaining

### Active (3-arch set)

**1. Optimizer device headers — DONE for the 3-arch active set.** All 11
optimizers have real per-arch implementations under
`csrc/device/optimizers/{sm_90,gfx942,tpu_v5p}/`. Phase 4 closed the
gfx942 row; Phase 5 closed tpu_v5p.

**2. Model device headers — DONE for the 3-arch active set.** Decoder
Transformer, ViT, and Mamba forward + backward templates exist for all
three arches. Phase 3 closed sm_90; Phase 5 closed tpu_v5p (the v5p
device-template files now re-export from
`csrc/kernels/tpu/_pallas_models.py`).

**3. Compile-time instantiation of 99 fused TUs.** Verify build, link, and
numerical parity against the separate-kernel path.

**4. Hopper warp-specialized scan activation.** The
`launch_scan_warp_specialized` and `_d16` declarations in sm_90 are unwired.
Expected ~1.5× on H100/H200 for long-segment workloads.

**5. Real autotune output for tuned_configs.h.** Run `build.sh --autotune` on
hardware to populate launch-config winners.

**6. Fused softplus epilogue in CUTLASS for SG2 dt_proj.** CUTLASS 3.x
EpilogueOp can fuse `softplus(x + bias)` into the GEMM tail.

**7. DSMEM for cross-CTA reductions on Hopper.** Norm reductions (LookSAM,
GrokAdamW, Prodigy, SG1.5/1.1) currently round-trip through global memory.

**8. CI matrix.** Configure runner for {sm_90, gfx942} × {test_*.py} matrix
via FORCE_ARCH.

**9. PyPI wheel.** From `--package-tarball` to `auditwheel`-compatible wheel.

### Future arch expansion (deferred to post-winner)

**A. NVFP4 path for sm_103.** Block-scaling factor calibration for Blackwell
Ultra projections.

**B. sm_120 retuned tile sizes.** Consumer Blackwell has 128 KB smem vs 228 KB;
placeholder configs will under-occupy.

**C. CDNA4 FP4/FP6/2:4 sparsity.** Native FP4 MFMA, FP6 state packing,
structured sparsity on gfx950.

**D. Per-feature gfx950 file split refinement.** ODR-safe helpers via
non-template `.cpp` with explicit extern declarations.

**E. Quantization device-template variants.** Will multiply kernel count when
added to the fused instantiation matrix.

---

## Contributing

To add a new fused kernel:

1. Write the device-function template under
   `csrc/device/optimizers/<arch>/<optimizer>_<arch>.cuh` (or model equivalent).
2. Create the fused TU under `csrc/fused/<arch>/fused_<model>_<optimizer>_<arch>.cu`
   that includes both model and optimizer headers.
3. Register the fused kernel in `grokking_optimizers/fused_dispatch.py` via
   `@register_fused(model, optimizer, arch)`.
4. Run `pytest tests/test_cross_arch_agreement.py` to verify numerical parity.

---

## License

MIT License. See `LICENSE`.

Acknowledgements:
- JAX and Pallas teams at Google for TPU primitives.
- NVIDIA CUTLASS team for the GEMM template library (enabled with `WITH_CUTLASS=1`).
