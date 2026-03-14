"""
Grokking Optimizers — C++/CUDA Extension Build Script

Builds custom CUDA kernels for all optimizers in the grokking race.
Requires CUDA toolkit + PyTorch with CUDA support.

Build:
    pip install -e .

Or for development:
    python setup.py build_ext --inplace
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA is REQUIRED — no CPU-only fallback
if not torch.cuda.is_available():
    raise RuntimeError(
        "Grokking Optimizers requires CUDA. "
        "Install PyTorch with CUDA support: https://pytorch.org/get-started/"
    )

print(f"Building Grokking Optimizers C++/CUDA extension")
print(f"  CUDA version: {torch.version.cuda}")

ext = CUDAExtension(
    name="grokking_optimizers._ops",
    sources=[
        "csrc/ops.cpp",
        # SuperGrok family
        "csrc/supergrok15_kernels.cu",
        "csrc/supergrok11_kernels.cu",
        "csrc/supergrok2_mamba_peer_kernels.cu",
        "csrc/supergrok2_mamba_peer_backward_kernels.cu",
        # Other optimizers
        "csrc/grokadamw_kernels.cu",
        "csrc/neuralgrok_kernels.cu",
        "csrc/prodigy_kernels.cu",
        "csrc/grokfast_kernels.cu",
        "csrc/lion_kernels.cu",
        "csrc/looksam_kernels.cu",
        "csrc/muon_kernels.cu",
    ],
    define_macros=[("WITH_CUDA", None)],
    extra_compile_args={
        "cxx": [
            "-O3",
            "-std=c++17",
            "-DWITH_CUDA",
            "-ffast-math",
            "-funroll-loops",
        ],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DWITH_CUDA",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            # Target common GPU architectures
            "-gencode=arch=compute_70,code=sm_70",   # V100
            "-gencode=arch=compute_75,code=sm_75",   # T4
            "-gencode=arch=compute_80,code=sm_80",   # A100
            "-gencode=arch=compute_86,code=sm_86",   # A10, RTX 3090
            "-gencode=arch=compute_89,code=sm_89",   # L4, RTX 4090
            "-gencode=arch=compute_90,code=sm_90",   # H100
        ],
    },
)

setup(
    name="grokking-optimizers",
    version="2.0.0",
    description=(
        "C++/CUDA fused optimizer kernels for grokking experiments. "
        "Includes SuperGrok v1.1/v1.5/v2, GrokAdamW, NeuralGrok, Prodigy, "
        "Grokfast, Lion, LookSAM, Muon."
    ),
    author="Peter C.",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
