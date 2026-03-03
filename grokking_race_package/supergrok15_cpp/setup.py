"""
SuperGrok v1.5 C++ Extension — Build Script

Detects CUDA availability and builds with fused CUDA kernels if available,
otherwise builds CPU-only with ATen operations.

Build:
    pip install -e .

Or for development:
    python setup.py build_ext --inplace
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# ── Detect CUDA ───────────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available() and os.environ.get("SUPERGROK_NO_CUDA") != "1"

print(f"Building SuperGrok v1.5 C++ extension")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Building with CUDA: {USE_CUDA}")
if USE_CUDA:
    print(f"  CUDA version: {torch.version.cuda}")

# ── Extension module ──────────────────────────────────────────────────
if USE_CUDA:
    ext = CUDAExtension(
        name="supergrok15_cpp._ops",
        sources=[
            "csrc/ops.cpp",
            "csrc/kernels.cu",
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
                "-lineinfo",  # for profiling
                # Targeting common GPU architectures
                "-gencode=arch=compute_70,code=sm_70",   # V100
                "-gencode=arch=compute_75,code=sm_75",   # T4
                "-gencode=arch=compute_80,code=sm_80",   # A100
                "-gencode=arch=compute_86,code=sm_86",   # A10, RTX 3090
                "-gencode=arch=compute_89,code=sm_89",   # L4, RTX 4090
                "-gencode=arch=compute_90,code=sm_90",   # H100
            ],
        },
    )
else:
    ext = CppExtension(
        name="supergrok15_cpp._ops",
        sources=["csrc/ops.cpp"],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
                "-ffast-math",
                "-funroll-loops",
                "-march=native",
            ],
        },
    )

# ── Setup ─────────────────────────────────────────────────────────────
setup(
    name="supergrok15_cpp",
    version="1.5.0",
    description=(
        "SuperGrok v1.5 C++/CUDA — Sharpness-aware grokking optimizer "
        "with fused CUDA kernels for meta-net inference and Adam updates"
    ),
    author="Peter C.",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=1.13.0"],
)
