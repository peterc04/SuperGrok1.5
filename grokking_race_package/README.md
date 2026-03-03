# Grokking Race v2 — 11-Optimizer Multi-GPU Benchmark

## Quick Start

```bash
# 1. Upload this entire directory to your GCP VM

# 2. Install dependencies + build C++ extension
python grokking_race_v2.py --setup

# 3. Run (single GPU, sequential — fair benchmark)
python grokking_race_v2.py

# 4. Run (multi-GPU — fast, still fair)
python grokking_race_v2.py --gpus 0,1,2,3

# 5. With phone notifications
python grokking_race_v2.py --gpus auto --ntfy peter-grok-2025
```

## Files

```
grokking_race_v2.py              # Main benchmark (1,497 lines)
supergrok15_cpp/                 # C++/CUDA optimizer (build with pip install -e .)
  csrc/kernels.cu                #   4 fused CUDA kernels
  csrc/ops.cpp                   #   C++ dispatch + CPU fallback + pybind11
  csrc/ops.h                     #   Declarations
  supergrok15_cpp/optim.py       #   Python wrapper
  supergrok15_cpp/__init__.py
  setup.py                       #   Build script (auto-detects CUDA)
  tests.py                       #   Test suite
  pyproject.toml
supergrok_v1_5/                  # Pure Python fallback (no build needed)
  supergrok15/supergrok15.py
  supergrok15/__init__.py
  tests.py
  pyproject.toml
```

## Modes

Change `MODE` at bottom of `grokking_race_v2.py`:

- **A** — Single run (1 model, 1 split, 6 seeds)
- **B** — Multi-split (1 model, 4 splits × 5 seeds)
- **C** — Architecture comparison (3 models × 5 seeds)
- **D** — Full sweep (3 models × 4 splits × 5 seeds = 660 runs)

## Optimizers

AdamW, NeuralGrok, GrokAdamW, SuperGrok (v1.1), **SuperGrok 1.5 (C++/CUDA)**,
Grokfast, Muon, Lion, LookSAM, Prodigy

## SuperGrok 1.5 — What's New

Three modifications targeting low-data grokking (ft10/ft25):

1. **2D SharpnessMetaNet**: Meta-net sees `(gradient, sharpness)` per element.
   Sharpness = `|SAM_grad − normal_grad|` — direct signal for memorization
   (sharp basin) vs generalization (flat basin).

2. **LookSAM integration**: Every 5 steps, compute SAM perturbation to get
   sharpness direction, cache for intermediate steps. Synced with bilevel.

3. **Progressive weight decay**: `wd_eff = wd × (1 + 4 × σ(20 × (acc − 0.9)))`.
   Gentle during feature learning, 5× base after full memorization.

## Multi-GPU

`--gpus 0,1,2,3` spawns one process per GPU. Tasks distributed round-robin.
Each GPU runs its tasks sequentially with exclusive access — wall-clock
measurements are uncontested and fair. Without `--gpus`, runs sequentially
on default device (also fair, just slower).

## Building the C++ Extension

```bash
cd supergrok15_cpp
pip install -e .
python tests.py
```

Requires PyTorch with CUDA. Falls back to CPU-only ATen ops if CUDA
unavailable. Falls back to pure Python if extension fails to build entirely.

Set `SUPERGROK_NO_CUDA=1` to force CPU-only build.
