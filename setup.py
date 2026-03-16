"""
Grokking Optimizers — C++/CUDA/HIP/CPU Extension Build Script

Builds custom optimizer kernels. Supports:
  - NVIDIA CUDA (sm_70-sm_90)
  - AMD ROCm/HIP (gfx908, gfx90a, gfx942)
  - CPU-only (OpenMP, reference implementation for debugging/testing)

Build:
    pip install -e .

Or for development:
    python setup.py build_ext --inplace
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

# Detect backend: CUDA, ROCm/HIP, or CPU-only
# FORCE_CUDA=1 allows building CUDA extension without a physical GPU
_is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
_force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
_has_gpu = torch.cuda.is_available() or (_force_cuda and torch.version.cuda is not None)

if _has_gpu and _is_hip:
    from torch.utils.cpp_extension import CUDAExtension
    import glob as _glob
    print(f"Building Grokking Optimizers C++/HIP extension")
    print(f"  ROCm version: {torch.version.hip}")

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
        "csrc/quantization/quantization_kernels.cu",
    ]

    # Auto-detect CDNA specialization sources
    cdna_sources = sorted(
        _glob.glob("csrc/hip/cdna2/*.hip.cpp") +
        _glob.glob("csrc/hip/cdna3/*.hip.cpp")
    )
    if cdna_sources:
        print(f"  CDNA sources: {', '.join(os.path.basename(s) for s in cdna_sources)}")

    # ROCm arch flags from TORCH_CUDA_ARCH_LIST or defaults
    rocm_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if rocm_archs:
        hipcc_arch_flags = []
        for arch in rocm_archs.replace(",", ";").split(";"):
            arch = arch.strip()
            if arch:
                hipcc_arch_flags.append(f"--offload-arch={arch}")
        print(f"  ROCm archs (from TORCH_CUDA_ARCH_LIST): {rocm_archs}")
    else:
        hipcc_arch_flags = [
            "--offload-arch=gfx908",    # MI100
            "--offload-arch=gfx90a",    # MI250
            "--offload-arch=gfx942",    # MI300X
        ]

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=generic_sources + cdna_sources,
        include_dirs=["csrc/common", "csrc"],
        define_macros=[("WITH_HIP", None)],
        extra_compile_args={
            "cxx": [
                "-O3", "-std=c++17", "-DWITH_HIP",
                "-ffast-math", "-funroll-loops",
            ],
            "nvcc": [
                "-O3", "-std=c++17", "-DWITH_HIP",
            ] + hipcc_arch_flags,
        },
    )

elif _has_gpu:
    from torch.utils.cpp_extension import CUDAExtension
    print(f"Building Grokking Optimizers C++/CUDA extension")
    print(f"  CUDA version: {torch.version.cuda}")

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
    nvidia_sources = [
        "csrc/cuda/sm_80/supergrok2_scan_sm80.cu",
        "csrc/cuda/sm_80/supergrok2_backward_sm80.cu",
        "csrc/cuda/sm_80/supergrok2_fused_elem_sm80.cu",
        "csrc/cuda/sm_80/metanet_optimizers_sm80.cu",
        "csrc/cuda/sm_80/metanet_cpasync_variants_sm80.cu",
        "csrc/cuda/sm_80/muon_sm80.cu",
        "csrc/cuda/sm_90/supergrok2_scan_sm90.cu",
        "csrc/cuda/sm_90/supergrok2_backward_sm90.cu",
        "csrc/cuda/sm_90/muon_sm90.cu",
        "csrc/cuda/sm_90/metanet_optimizers_sm90.cu",
        "csrc/cuda/sm_100/supergrok2_sm100.cu",
        "csrc/quantization/quantization_kernels.cu",
    ]
    # Auto-detect generated kernel sources
    import glob as _glob_nv
    generated_sources = sorted(_glob_nv.glob("csrc/cuda/generated/*.cu"))
    if generated_sources:
        print(f"  Generated sources: {', '.join(os.path.basename(s) for s in generated_sources)}")
        nvidia_sources += generated_sources
    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=generic_sources + nvidia_sources,
        include_dirs=["csrc/common", "csrc"],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [
                "-O3", "-std=c++17", "-DWITH_CUDA",
                "-ffast-math", "-funroll-loops",
            ],
            "nvcc": [
                "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
                "--expt-relaxed-constexpr", "-lineinfo",
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
    from torch.utils.cpp_extension import CppExtension
    import platform as _platform
    print("Building Grokking Optimizers C++ CPU-only extension")
    print("  No GPU detected — building reference CPU kernels with OpenMP + SIMD.")

    _cpu_sources = [
        "csrc/cpu/cpu_ops.cpp",
        "csrc/cpu/cpu_kernels.cpp",
        "csrc/cpu/generic/all_optimizers_cpu.cpp",
        "csrc/cpu/generic/supergrok2_scan_cpu.cpp",
    ]

    _cpu_cxx_flags = [
        "-O3", "-std=c++17", "-DWITH_CPU",
        "-ffast-math", "-funroll-loops",
        "-fopenmp",
    ]

    _cpu_arch = _platform.machine().lower()
    if _cpu_arch in ("x86_64", "amd64"):
        _cpu_sources.append("csrc/cpu/avx512/simd_kernels.cpp")
        _cpu_cxx_flags.append("-march=native")
        print("  SIMD: x86_64 detected, AVX-512 auto-detected via -march=native")
    elif _cpu_arch in ("aarch64", "arm64"):
        _cpu_sources.append("csrc/cpu/neon/simd_kernels.cpp")
        print("  SIMD: ARM detected, NEON intrinsics enabled")
    else:
        print(f"  SIMD: unknown arch '{_cpu_arch}', scalar fallback only")

    ext = CppExtension(
        name="grokking_optimizers._ops",
        sources=_cpu_sources,
        include_dirs=["csrc/common", "csrc", "csrc/cpu"],
        define_macros=[("WITH_CPU", None)],
        extra_compile_args={
            "cxx": _cpu_cxx_flags,
        },
        extra_link_args=["-fopenmp"],
    )

setup(
    name="grokking-optimizers",
    version="2.1.0",
    description=(
        "C++/CUDA/HIP/JAX optimizer suite with SuperGrok v2 "
        "(Mamba-3 + PEER + GRU meta-net). Supports NVIDIA (sm_70-100), "
        "AMD (MI250/MI300X), TPU (v4/v5), and CPU. "
        "Multi-precision: FP32/TF32/BF16/FP8/INT8/INT4/MXFP4/NVFP4."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Peter C.",
    url="https://github.com/peterc04/SuperGrok1.5",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
    extras_require={
        "jax": ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "test": ["pytest", "numpy"],
    },
)
