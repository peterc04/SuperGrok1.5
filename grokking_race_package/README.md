# Grokking Race v2 — 11-Optimizer Benchmark with Custom C++/CUDA Kernels

A comprehensive benchmark suite for studying **grokking** (delayed generalization after memorization) across 11 optimizers, 3 model architectures, and multiple data splits. Includes **SuperGrok v2**, a novel optimizer with a Mamba-3 + PEER + GRU meta-network implemented in custom CUDA kernels.

## Quick Start

```bash
# 1. Build the C++/CUDA optimizer extension
cd grokking_race_package/grokking_optimizers
pip install -e .

# 2. Run the benchmark (single GPU)
cd ..
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
- **Gradient Checkpointing for Bilevel**: Optional `bilevel_checkpoint_interval` parameter saves Mamba scan states every C steps instead of every step during bilevel forward-save. During backward, intermediate states are recomputed from the nearest checkpoint. With C=32, reduces bilevel saved-state memory by ~82% (1.2 GB → 224 MB for 50 parameters).

## Model Architectures

- **Decoder Transformer** — causal attention, standard for modular arithmetic grokking
- **Vision Transformer (ViT)** — patch embeddings for image classification
- **Mamba SSM** — selective state space model (linear-time sequence processing)

## File Structure

```
grokking_race_package/
├── grokking_race_v2.py                         # Benchmark harness (1,687 lines)
├── README.md
├── ANALYSIS.md
├── grokking_optimizers/                        # C++/CUDA optimizer package
│   ├── setup.py                                # Build script (sm_70–sm_90)
│   └── grokking_optimizers/
│       ├── __init__.py                         # Package exports
│       ├── supergrok2.py                       # SuperGrok v2 optimizer (1,049 lines)
│       ├── mamba3_peer_metanet.py              # Mamba-3+PEER+GRU meta-net (574 lines)
│       ├── supergrok15.py                      # SuperGrok v1.5 optimizer (478 lines)
│       ├── supergrok11.py                      # SuperGrok v1.1 optimizer (296 lines)
│       ├── grokadamw.py                        # GrokAdamW (144 lines)
│       ├── neuralgrok.py                       # NeuralGrok (228 lines)
│       ├── prodigy.py                          # Prodigy (136 lines)
│       ├── grokfast.py                         # Grokfast (145 lines)
│       ├── lion.py                             # Lion (101 lines)
│       ├── looksam.py                          # LookSAM (247 lines)
│       ├── muon.py                             # Muon (210 lines)
│       └── cuda_graph_optimizer.py             # CUDA graph wrapper (168 lines)
│   └── csrc/
│       ├── ops.h                               # C++ declarations (554 lines)
│       ├── ops.cpp                             # Pybind11 dispatch (1,076 lines)
│       ├── supergrok2_mamba_peer_kernels.cu    # v2 forward kernels (1,218 lines)
│       ├── supergrok2_mamba_peer_backward_kernels.cu  # v2 backward kernels (1,963 lines)
│       ├── supergrok15_kernels.cu              # v1.5 kernels (464 lines)
│       ├── supergrok11_kernels.cu              # v1.1 kernels (349 lines)
│       ├── grokadamw_kernels.cu                # GrokAdamW kernels (140 lines)
│       ├── neuralgrok_kernels.cu               # NeuralGrok kernels (233 lines)
│       ├── prodigy_kernels.cu                  # Prodigy kernels (254 lines)
│       ├── grokfast_kernels.cu                 # Grokfast kernels (60 lines)
│       ├── lion_kernels.cu                     # Lion kernels (79 lines)
│       ├── looksam_kernels.cu                  # LookSAM kernels (155 lines)
│       └── muon_kernels.cu                     # Muon kernels (146 lines)
```

Total: ~12,400 lines (5,500 Python, 4,500 CUDA, 1,600 C++)

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
