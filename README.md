# Grokking Race v2 — 11-Optimizer Benchmark with Custom C++/CUDA Kernels

A comprehensive benchmark suite for studying **grokking** (delayed generalization after memorization) across 11 optimizers, 3 model architectures, and multiple data splits. Includes **SuperGrok v2**, a novel optimizer with a Mamba-3 + PEER + GRU meta-network implemented in custom CUDA kernels.

## Quick Start

```bash
# 1. Build the C++/CUDA optimizer extension
pip install -e .

# 2. Run the benchmark (single GPU)
python grokking_race_v2.py

# 3. Run (multi-GPU — fast, still fair)
python grokking_race_v2.py --gpus 0,1,2,3
```

## Optimizers (11)

| Optimizer | Description |
|-----------|-------------|
| **SuperGrok v2** | Mamba-3 bidirectional scan + 4-head PEER routing + GRU + 144 experts (custom CUDA) |
| **SuperGrok v1.5** | 2D sharpness meta-net + SAM + progressive WD (custom CUDA) |
| **SuperGrok v1.1** | Meta-net with cosine similarity gating (custom CUDA) |
| **GrokAdamW** | EMA gradient filter + amplification + AdamW (custom CUDA) |
| **NeuralGrok** | MLP gradient amplifier + AdamW (custom CUDA) |
| **Prodigy** | Distance-aware self-tuning AdamW (custom CUDA) |
| **Grokfast** | EMA + gradient amplification (custom CUDA) |
| **Lion** | Sign-based momentum optimizer (custom CUDA) |
| **LookSAM** | Sharpness-Aware Minimization with direction caching (custom CUDA) |
| **Muon** | Momentum + Newton-Schulz orthogonalization (custom CUDA) |
| **AdamW** | Standard PyTorch AdamW (baseline) |

## SuperGrok v2 Architecture

The meta-network processes per-parameter gradient vectors through:

1. **Sort** by gradient magnitude → creates meaningful sequence for scan
2. **Bidirectional Mamba-3 Scan** (forward + backward) — trapezoidal discretization, paired RoPE, selective B/C/dt gating — captures cross-element gradient correlations
3. **Per-element GRU** (4-dim hidden) — temporal memory across optimizer steps
4. **4-Head PEER Routing** — product-key expert selection (top-4 per sub-key per head, 144 total experts)
5. **Expert MLP** — per-expert gradient transformation (1 → 16 → 1)
6. **Skip connection** — `smart_grad = grad + 0.1 * expert_output`
7. **Adaptive Adam** — bias-corrected first/second moments + weight decay

Additional features: dynamic expert recycling, sigmoid-driven SAM/bilevel/WD scheduling, functional_call SAM (no parameter modification), CUDA batched scan, AMP support.

### Performance Optimizations

- **Blelloch Parallel Prefix Scan**: For parameters with N >= 256 elements, the sequential O(N) scan is replaced with a work-efficient parallel scan achieving O(N/P + log N) time. Uses affine transform composition over paired RoPE state dimensions. Automatically falls back to sequential scan for small parameters.
- **Expert Weights in Shared Memory**: All 144 expert MLP weights (W1, b1, W2, b2) are loaded into CUDA shared memory at block start, eliminating repeated global memory reads during per-element PEER evaluation.
- **Expert Backward Shared-Memory Reduction**: Per-block accumulators in shared memory for expert weight gradients, with a single block-level `atomicAdd` at the end — reduces atomic contention by 256x.
- **Two-Pass GEMM Backward for Projection Weights**: Backward scan weight gradients (d_C_proj_W, d_B_proj_W) use a two-pass approach: Pass 1 writes per-timestep warp-reduced derivative scalars to a global buffer via `__shfl_down_sync`; Pass 2 accumulates via cuBLAS GEMM (`torch::mm_out`). Eliminates N×d_state×d_inner shared-memory atomicAdds per scan direction, replacing them with a single matrix multiply.
- **Batched Parallel Scan Single-Launch**: For parameters with N >= 256 elements, the per-parameter for-loop of parallel scan kernel launches is replaced with a single `mamba3_parallel_scan_batched_kernel` using a 2D grid `dim3(d_inner, num_params)`. Eliminates num_params kernel launch overhead and enables cross-parameter SM scheduling.
- **Gradient Checkpointing for Bilevel**: Optional `bilevel_checkpoint_interval` parameter saves Mamba scan states every C steps instead of every step during bilevel forward-save. During backward, intermediate states are recomputed from the nearest checkpoint. With C=32, reduces bilevel saved-state memory by ~82% (1.2 GB → 224 MB for 50 parameters).
- **Pre-Allocated Bilevel Workspace**: A `thread_local` `BilevelWorkspace` struct reuses temporary buffers (precompute outputs, reversed sort arrays, gradient accumulators) across optimizer steps, eliminating ~100 MB of per-step `torch::empty` allocations for 50 parameters.
- **ATen GEMM for Projection Precompute**: For parameters with N >= 1024 elements, bilevel precompute projections (input, dt, B, C) use cuBLAS via `torch::mm_out` instead of a custom CUDA kernel. Automatically falls back to the custom kernel for small N where cuBLAS launch overhead dominates.
- **Dimension Safety Guards**: Runtime `TORCH_CHECK` assertions validate that d_model, d_inner, and d_state do not exceed compile-time maximums (MAX_D_MODEL=16, MAX_D_INNER=32, MAX_D_STATE=32) in all forward and backward launchers.

## Testing

```bash
python tests/test_supergrok2.py
```

The PyTorch test suite (`test_supergrok2.py`) covers 27 areas (plus 12 JAX tests in `supergrok2_jax_tpu/tests/`):

| Test | Description |
|------|-------------|
| 12A | Import and build verification |
| 12B | Sequential vs parallel scan numerical equivalence |
| 12C | Forward step correctness (params changed, no NaN, state populated) |
| 12D | Bilevel meta-learning correctness |
| 12E | Two-pass backward equivalence |
| 12F | Expert recycling stability (50 steps without crash) |
| 12G | Gradient checkpointing equivalence (checkpoint_interval=1 vs 8) |
| 12H | Edge cases (N=0, N=1, zero grads, large grads, FP16 params) |
| 12I | All 11 optimizers construct + step |
| 12J | Memory leak check (200 steps, <10% growth) |
| 12K | Two-pass GEMM backward reproducibility (max diff < 1e-4 across seeded runs) |
| 12L | Batched parallel scan single-launch (finite params, bitwise reproducibility) |
| 12M | Dispatch detection (Python/C++ GPU architecture agreement) |
| 12N | Precision config auto-selection |
| 12O | Projection precision equivalence (FP32 vs auto) |
| 12P | Dispatch convergence (10 steps) |
| 12Q | Platform/vendor detection (Python/C++ agreement) |
| 12R | INT8 symmetric quantization round-trip |
| 12S | INT4 GPTQ-style packing correctness |
| 12T | MXFP4 microscaling FP4 quantization |
| 12U | Dynamic precision selection (stability-aware) |
| 12V | Expert FP32 passthrough |
| 12W | Distributed helper methods (DDP hooks, no-op without dist) |
| 12X | CompiledSuperGrok2 wrapper (warmup/capture/replay) |
| 12Y | step_compiled method (_prepare_for_compile + step) |
| 12Z | FSDP exclusion helper (meta-net module marking) |
| 12AA | Distributed module import and utilities |

Each test reports PASS/FAIL. Exit code 0 = all pass, 1 = any failure.

### Cross-Platform Test Matrix

```bash
# Test all optimizers on current GPU
python tests/test_matrix.py

# Test all NVIDIA architecture tiers via FORCE_ARCH
python tests/test_all_tiers.py

# Test all JAX optimizers
python tests/test_jax_matrix.py
```

### Benchmarks

```bash
# Benchmark all optimizers (step time, memory, throughput)
python benchmarks/benchmark_supergrok2.py

# Benchmark a single optimizer with larger model
python benchmarks/benchmark_supergrok2.py --optimizer SuperGrok2 --model-size 512

# Auto-tune kernel configs for current GPU
python benchmarks/autotune.py
```

## Model Architectures

- **Decoder Transformer** — causal attention, standard for modular arithmetic grokking
- **Vision Transformer (ViT)** — patch embeddings for image classification
- **Mamba SSM** — selective state space model (linear-time sequence processing)

## Directory Structure

```
./
├── csrc/
│   ├── common/                                 # Shared headers and pybind dispatch
│   │   ├── platform.h                          # CUDA/HIP abstraction layer
│   │   ├── types.h                             # Affine2x2, constants, common structs
│   │   ├── utils.cuh                           # warp_reduce_sum, device helpers
│   │   ├── dispatch.h                          # Runtime GPU arch detection (C++)
│   │   ├── quantization.h                      # Quantization structs and device helpers
│   │   ├── ops.h                               # Master C++ declarations
│   │   └── ops.cpp                             # Pybind11 dispatch
│   │
│   ├── cuda/
│   │   ├── generic/                            # Architecture-independent GPU kernels
│   │   │   ├── supergrok2_mamba_peer_kernels.cu
│   │   │   ├── supergrok2_mamba_peer_backward_kernels.cu
│   │   │   ├── supergrok15_kernels.cu
│   │   │   ├── supergrok11_kernels.cu
│   │   │   ├── grokadamw_kernels.cu
│   │   │   ├── neuralgrok_kernels.cu
│   │   │   ├── prodigy_kernels.cu
│   │   │   ├── grokfast_kernels.cu
│   │   │   ├── lion_kernels.cu
│   │   │   ├── looksam_kernels.cu
│   │   │   └── muon_kernels.cu
│   │   ├── sm_80/                              # Ampere-optimized kernels (TF32)
│   │   └── sm_90/                              # Hopper-optimized kernels
│   │
│   ├── hip/                                    # AMD ROCm/HIP notes and optimizations
│   │   └── README_HIP.md                       # Wavefront-64 architecture notes
│   ├── cpu/                                    # CPU fallback kernels (future)
│   └── quantization/                           # Quantization kernels (future)
│
├── supergrok2_jax_tpu/                          # TPU/JAX implementation
│   ├── __init__.py                              # Package exports
│   ├── scan.py                                  # Mamba-3 scan via lax.associative_scan
│   ├── gru.py                                   # Per-element GRU cell
│   ├── peer.py                                  # Multi-head PEER routing (soft + hard)
│   ├── mamba3_peer_metanet_jax.py               # Full meta-net forward pass
│   ├── supergrok2_jax.py                        # Optimizer step (functional)
│   ├── bilevel.py                               # Bilevel optimization via jax.grad
│   ├── sharding.py                              # TPU mesh + data-parallel sharding
│   ├── quantization_jax.py                      # INT8 quantization utilities
│   ├── bridge.py                                # PyTorch <-> JAX weight conversion
│   ├── simple_optimizers_jax.py                  # JAX: GrokAdamW, Lion, Grokfast, Prodigy, Muon, LookSAM
│   ├── metanet_optimizers_jax.py                 # JAX: SuperGrok v1.5, v1.1, NeuralGrok
│   ├── pallas_kernels.py                         # Pallas custom kernels (with fallback stub)
│   ├── distributed_example.py                    # Multi-TPU pod training example
│   └── tests/
│       └── test_supergrok2_jax.py                # 17-test JAX test suite
│
├── grokking_optimizers/                        # Python package
│   ├── __init__.py                             # Package exports
│   ├── dispatch.py                             # Runtime hardware + vendor detection
│   ├── quantization.py                         # PrecisionConfig, INT8/INT4/MXFP4
│   ├── supergrok2.py                           # SuperGrok v2 optimizer
│   ├── mamba3_peer_metanet.py                  # Mamba-3+PEER+GRU meta-net
│   ├── supergrok15.py                          # SuperGrok v1.5 optimizer
│   ├── supergrok11.py                          # SuperGrok v1.1 optimizer
│   ├── grokadamw.py                            # GrokAdamW
│   ├── neuralgrok.py                           # NeuralGrok
│   ├── prodigy.py                              # Prodigy
│   ├── grokfast.py                             # Grokfast
│   ├── lion.py                                 # Lion
│   ├── looksam.py                              # LookSAM
│   ├── muon.py                                 # Muon
│   ├── cuda_graph_optimizer.py                 # CUDA graph wrapper
│   └── distributed.py                          # DDP/FSDP training utilities
│
├── tests/                                      # Test suite
│   ├── test_supergrok2.py                      # 27-area PyTorch test suite
│   ├── test_matrix.py                          # Cross-platform optimizer matrix
│   ├── test_all_tiers.py                       # Multi-tier FORCE_ARCH validation
│   ├── test_jax_matrix.py                      # JAX optimizer test matrix (10 tests)
│   ├── test_amd_hip.py                         # AMD ROCm/HIP-specific tests
│   └── test_cpu_fallback.py                    # CPU fallback path tests
│
├── benchmarks/                                 # Performance benchmarks
│   ├── benchmark_supergrok2.py                 # Step time, memory, throughput
│   └── autotune.py                             # Per-GPU kernel auto-tuning
│
├── setup.py                                    # Build script (CUDA sm_70–sm_90 + ROCm)
├── pyproject.toml
├── grokking_race_v2.py                         # Benchmark harness
├── README.md
└── ANALYSIS.md
```

## Configuration

Key SuperGrok v2 hyperparameters (defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 8 | Meta-net internal dimension |
| `d_state` | 16 | Mamba state dimension (paired RoPE) |
| `mamba_expand` | 2 | Expansion factor (d_inner = d_model × expand) |
| `num_experts` | 144 | Total experts in PEER pool |
| `expert_hidden` | 16 | Expert MLP hidden dimension |
| `gru_hidden` | 4 | GRU temporal memory dimension |
| `num_peer_heads` | 4 | PEER routing heads |
| `meta_rescale` | 0.1 | Skip connection scale factor |
| `sam_rho` | 0.05 | SAM perturbation radius |
| `recycle_interval` | 100 | Steps between dead expert recycling |
| `bilevel_checkpoint_interval` | 1 | Checkpoint interval for bilevel gradient checkpointing (1 = save every step, 32 = save every 32 steps) |
| `projection_precision` | `'auto'` | Precision for projection GEMMs: `'fp32'`, `'tf32'`, `'bf16'`, `'fp8'`, `'mxfp4'`, or `'auto'` |
| `expert_precision` | `'fp32'` | Expert weight quantization: `'fp32'`, `'int8'`, `'int4'`, or `'auto'` |
| `dynamic` | `False` | Enable dynamic precision selection (progressively lowers precision as training stabilizes) |
| `bilevel_allreduce_meta_grads` | `True` | All-reduce meta-net gradients across ranks in distributed training |
| `expert_allreduce_before_recycle` | `True` | All-reduce expert counts before recycling in distributed training |
| `mamba_state_sync_interval` | 1000 | Steps between mamba state broadcasts (0 = disable) |

## Hardware Support

Kernels are tiered by GPU architecture for automatic hardware-specific optimization:

### NVIDIA (CUDA)

| Tier | Architectures | Key Features |
|------|--------------|--------------|
| **Generic** | sm_70, sm_75 (V100, T4) | FP32 only, basic smem |
| **Ampere** | sm_80, sm_86, sm_89 (A100, RTX 3090, L4, RTX 4090) | TF32 Tensor Cores, cp.async, 192KB smem, BF16 |
| **Hopper** | sm_90 (H100) | All Ampere features + FP8 E4M3 cuBLAS GEMMs for projections (CUDA 11.8+), 228KB smem |
| **Blackwell** | sm_100 (B200) | Hopper path (FP8 + cp.async). TMEM/MMA.2SM/NVFP4 deferred pending hardware access |

### AMD (ROCm/HIP)

| Architecture | GPU | Key Features |
|-------------|-----|--------------|
| gfx908 | MI100 | Matrix Cores, FP32/FP16 |
| gfx90a (CDNA2) | MI200 (MI210, MI250, MI250X) | BF16 Matrix Cores, wavefront-64 sync skip in Blelloch scan (via platform.h WARP_SIZE=64) |
| gfx942 (CDNA3) | MI300X | BF16 MFMA projections, 256MB L2 cache (meta-net weights L2-resident), wavefront-64 scan |

All generic kernels compile for both CUDA and HIP via the `platform.h` abstraction layer. AMD uses wavefront-64 (vs CUDA warp-32) — handled automatically via `WARP_SIZE` in `platform.h`. CDNA2 gets intra-wavefront sync skip in the Blelloch scan (strides 1-32 skip `__syncthreads()`). CDNA3 adds BF16 MFMA projections for ~2x throughput. Ampere/Hopper tier kernels are NVIDIA-only.

Runtime dispatch is automatic — the optimal kernel tier is selected based on the detected GPU.

## Quantization

### Projection Precision

Projection GEMMs support multi-precision via `projection_precision`:

| Format | Where | Benefit |
|--------|-------|---------|
| TF32 | Projections on sm_80+ | 2x FP32 throughput, transparent via cuBLAS |
| BF16 | Projections on sm_80+ / gfx90a+ | Same range as FP32, 2x bandwidth |
| FP8 (E4M3) | Projections on sm_89+/sm_90+ | 4x throughput vs FP32 on Tensor Cores |
| MXFP4 | Projection weights (all GPUs) | 8x compression, Microscaling FP4 with shared exponents |

### Expert Weight Quantization

Expert MLP weights support weight-only quantization via `expert_precision`:

| Format | Compression | Description |
|--------|------------|-------------|
| FP32 | 1x | Default, full precision |
| INT8 | 4x | Symmetric per-tensor quantization (scale = max(\|w\|)/127) |
| INT4 | 8x | GPTQ-style packing with group scales and zero-points |

### Dynamic Precision Selection

When `dynamic=True`, the optimizer monitors gradient norm stability (coefficient of variation) and progressively lowers precision as training stabilizes. If training becomes unstable, precision is raised back. This is inspired by Unsloth's progressive precision approach.

Scan state accumulation always stays FP32 (numerical necessity for long recurrences).

## Multi-GPU

`--gpus 0,1,2,3` spawns one process per GPU. Tasks are distributed round-robin with exclusive GPU access for fair wall-clock measurements.

## Distributed Training (DDP / FSDP)

SuperGrok v2 supports PyTorch DDP and FSDP for multi-node training.

### DDP

```python
from grokking_optimizers import SuperGrok2, setup_distributed, wrap_model_ddp

setup_distributed()
model = wrap_model_ddp(model.cuda())
opt = SuperGrok2(model.parameters(), lr=1e-3)
# Training loop works as normal — meta-grad all-reduce is automatic
```

Key features:
- **Meta-gradient all-reduce**: Bilevel meta-net gradients are averaged across ranks before stepping (controlled by `bilevel_allreduce_meta_grads=True`).
- **Expert count sync**: Expert activation counts are all-reduced across ranks before recycling dead experts (`expert_allreduce_before_recycle=True`).
- **Mamba state broadcast**: Periodic broadcast of Mamba scan states from rank 0 to prevent drift (`mamba_state_sync_interval=1000`).

### FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

opt = SuperGrok2(model.parameters(), lr=1e-3)
SuperGrok2.exclude_meta_net_from_fsdp(opt.meta_net)  # Keep meta-net replicated
model = FSDP(model, auto_wrap_policy=...)
```

Launch with `torchrun`:
```bash
torchrun --nproc_per_node=4 train.py
```

## torch.compile / CUDA Graph Support

SuperGrok v2 provides a graph-capturable optimizer step via `CompiledSuperGrok2`:

```python
from grokking_optimizers import SuperGrok2, CompiledSuperGrok2

opt = SuperGrok2(model.parameters(), lr=1e-3)
compiled_opt = CompiledSuperGrok2(opt, warmup_steps=3)

for step in range(n_steps):
    loss.backward()
    compiled_opt.step()  # Warmup → capture → replay automatically
```

Features:
- **Warmup phase**: First N steps run in eager mode to initialize optimizer state.
- **CUDA graph capture**: After warmup, the step is captured as a CUDA graph.
- **Automatic replay**: Subsequent steps replay the graph with zero CPU overhead.
- **Expert recycling**: Periodically drops to eager mode for dead expert recycling.
- **Graceful fallback**: Falls back to eager mode if graph capture fails.
- **torch.compile support**: Optional `enable_compile=True` for `torch.compile` integration.

The low-level `step_compiled()` method on `SuperGrok2` is also available for custom graph capture pipelines.

## JAX / TPU Support

A complete JAX rewrite of SuperGrok v2 for TPU (and JAX-on-GPU). Uses JAX native primitives instead of CUDA kernels.

### Key Differences from PyTorch/CUDA

| Feature | PyTorch/CUDA | JAX/TPU |
|---------|-------------|---------|
| Mamba scan | Sequential CUDA kernel (Blelloch parallel for N>=256) | `lax.associative_scan` (O(log N) depth) |
| Bilevel backward | 1000+ lines of custom backward CUDA kernels | `jax.grad` (automatic differentiation) |
| State management | In-place mutation (`tensor.mul_()`) | Functional (explicit state in, state out) |
| Compilation | `torch.compile` / CUDA graphs | `jax.jit` (XLA compilation) |
| Multi-device | DDP/FSDP | `jax.sharding.Mesh` with data-parallel axis |

### Usage

```python
import jax
import jax.numpy as jnp
from supergrok2_jax_tpu import (
    OptimizerConfig, MetaNetConfig,
    init_state, init_meta_weights, supergrok2_step,
)

# Initialize
config = OptimizerConfig(lr=1e-3)
meta_config = MetaNetConfig()
key = jax.random.PRNGKey(0)
meta_weights = init_meta_weights(meta_config, key)
opt_state = init_state(params, config, meta_config)

# Training step (JIT-compatible)
@jax.jit
def train_step(params, grads, opt_state, meta_weights):
    return supergrok2_step(params, grads, opt_state, meta_weights, config, meta_config)

new_params, new_opt_state = train_step(params, grads, opt_state, meta_weights)
```

### PyTorch <-> JAX Bridge

```python
from supergrok2_jax_tpu import pytorch_weights_to_jax, jax_weights_to_pytorch

# Convert trained PyTorch meta-net to JAX
jax_weights = pytorch_weights_to_jax(pytorch_meta_net)

# Convert back
jax_weights_to_pytorch(jax_weights, pytorch_meta_net)
```

### JAX Tests

```bash
python supergrok2_jax_tpu/tests/test_supergrok2_jax.py
```

17 tests covering: imports, associative scan operator, Mamba scan, GRU cell, PEER routing, full forward, optimizer step, bilevel gradients, JIT compilation, INT8 quantization, sharding, pytree compatibility, cross-framework test vectors, all simple optimizers, meta-net optimizers, sharding utilities, and JIT no-retrace verification.

### Requirements (JAX)

- JAX 0.4+ (`pip install jax[tpu]` for TPU, `pip install jax[cuda12]` for GPU)
- NumPy

## Requirements

- PyTorch 2.0+ with CUDA or ROCm support
- **NVIDIA**: CUDA 11.8 or 12.x — GPU architectures: V100 (sm_70), T4 (sm_75), A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89), H100 (sm_90)
- **AMD**: ROCm 5.4+ (recommended 6.0+) — GPU architectures: MI100 (gfx908), MI200 (gfx90a), MI300X (gfx942)
