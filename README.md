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

The test suite (`test_supergrok2.py`) covers 12 areas:

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

Each test reports PASS/FAIL. Exit code 0 = all pass, 1 = any failure.

## Model Architectures

- **Decoder Transformer** — causal attention, standard for modular arithmetic grokking
- **Vision Transformer (ViT)** — patch embeddings for image classification
- **Mamba SSM** — selective state space model (linear-time sequence processing)

## Directory Structure

```
./
├── csrc/
│   ├── common/                                 # Shared headers and pybind dispatch
│   │   ├── types.h                             # Affine2x2, constants, common structs
│   │   ├── utils.cuh                           # warp_reduce_sum, device helpers
│   │   ├── ops.h                               # Master C++ declarations
│   │   └── ops.cpp                             # Pybind11 dispatch
│   │
│   ├── cuda/
│   │   └── generic/                            # Architecture-independent CUDA kernels
│   │       ├── supergrok2_mamba_peer_kernels.cu
│   │       ├── supergrok2_mamba_peer_backward_kernels.cu
│   │       ├── supergrok15_kernels.cu
│   │       ├── supergrok11_kernels.cu
│   │       ├── grokadamw_kernels.cu
│   │       ├── neuralgrok_kernels.cu
│   │       ├── prodigy_kernels.cu
│   │       ├── grokfast_kernels.cu
│   │       ├── lion_kernels.cu
│   │       ├── looksam_kernels.cu
│   │       └── muon_kernels.cu
│   │   ├── sm_75/ .. sm_100/                   # Per-architecture specialized kernels (future)
│   │
│   ├── hip/                                    # AMD ROCm/HIP kernels (future)
│   ├── cpu/                                    # CPU fallback kernels (future)
│   └── quantization/                           # Quantization kernels (future)
│
├── jax/                                        # TPU/JAX implementation (future)
│
├── grokking_optimizers/                        # Python package
│   ├── __init__.py                             # Package exports
│   ├── dispatch.py                             # Runtime hardware detection
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
│   └── cuda_graph_optimizer.py                 # CUDA graph wrapper
│
├── tests/                                      # Test suite
│   └── test_supergrok2.py
│
├── setup.py                                    # Build script (sm_70–sm_90)
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

## Multi-GPU

`--gpus 0,1,2,3` spawns one process per GPU. Tasks are distributed round-robin with exclusive GPU access for fair wall-clock measurements.

## Requirements

- PyTorch 2.0+ with CUDA support
- CUDA 11.8 or 12.x
- GPU architectures: V100 (sm_70), T4 (sm_75), A100 (sm_80), RTX 3090 (sm_86), RTX 4090 (sm_89), H100 (sm_90)
