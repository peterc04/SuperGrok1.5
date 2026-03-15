"""
Grokking Optimizers — C++/CUDA/HIP Extension Build Script

Builds custom GPU kernels for all optimizers in the grokking race.
Requires CUDA toolkit or ROCm + PyTorch with GPU support.

Build:
    pip install -e .

Or for development:
    python setup.py build_ext --inplace
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect backend: CUDA or ROCm/HIP
_is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None

if not torch.cuda.is_available():
    raise RuntimeError(
        "Grokking Optimizers requires a GPU (CUDA or ROCm). "
        "Install PyTorch with GPU support: https://pytorch.org/get-started/"
    )

if _is_hip:
    print(f"Building Grokking Optimizers C++/HIP extension")
    print(f"  ROCm version: {torch.version.hip}")
else:
    print(f"Building Grokking Optimizers C++/CUDA extension")
    print(f"  CUDA version: {torch.version.cuda}")

# Generic kernels — shared between CUDA and HIP
generic_sources = [
    "csrc/common/ops.cpp",
    "csrc/cuda/generic/supergrok15_kernels.cu",
    "csrc/cuda/generic/supergrok11_kernels.cu",
    "csrc/cuda/generic/supergrok2_mamba_peer_kernels.cu",
    "csrc/cuda/generic/supergrok2_mamba_peer_backward_kernels.cu",
    "csrc/cuda/generic/grokadamw_kernels.cu",
    "csrc/cuda/generic/neuralgrok_kernels.cu",
    "csrc/cuda/generic/prodigy_kernels.cu",
    "csrc/cuda/generic/grokfast_kernels.cu",
    "csrc/cuda/generic/lion_kernels.cu",
    "csrc/cuda/generic/looksam_kernels.cu",
    "csrc/cuda/generic/muon_kernels.cu",
]

# NVIDIA-specific arch-specialized kernels (not compiled under HIP)
nvidia_sources = [
    "csrc/cuda/sm_80/supergrok2_scan_sm80.cu",
    "csrc/cuda/sm_80/supergrok2_backward_sm80.cu",
    "csrc/cuda/sm_90/supergrok2_scan_sm90.cu",
    "csrc/cuda/sm_90/supergrok2_backward_sm90.cu",
]

sources = generic_sources
define_macros = []

if _is_hip:
    define_macros.append(("WITH_HIP", None))
    extra_cxx = [
        "-O3",
        "-std=c++17",
        "-DWITH_HIP",
        "-ffast-math",
        "-funroll-loops",
    ]
    # hipcc flags — ROCm targets
    extra_gpu = [
        "-O3",
        "-std=c++17",
        "-DWITH_HIP",
        # CDNA architectures (data center)
        "--offload-arch=gfx908",    # MI100
        "--offload-arch=gfx90a",    # MI200 (MI210, MI250, MI250X)
        "--offload-arch=gfx942",    # MI300X
    ]
else:
    define_macros.append(("WITH_CUDA", None))
    sources += nvidia_sources
    extra_cxx = [
        "-O3",
        "-std=c++17",
        "-DWITH_CUDA",
        "-ffast-math",
        "-funroll-loops",
    ]
    extra_gpu = [
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
    ]

ext = CUDAExtension(
    name="grokking_optimizers._ops",
    sources=sources,
    include_dirs=["csrc/common", "csrc"],
    define_macros=define_macros,
    extra_compile_args={
        "cxx": extra_cxx,
        "nvcc": extra_gpu,
    },
)

setup(
    name="grokking-optimizers",
    version="2.0.0",
    description=(
        "C++/CUDA/HIP fused optimizer kernels for grokking experiments. "
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
