"""
Grokking Optimizers — All-Specialized Per-Arch Kernel Build

Supported arches:
  - NVIDIA: sm_80 (Ampere family: A100/A30/A10/RTX 30 routes here),
            sm_89 (Ada: RTX 40, L40, L40S),
            sm_90 (Hopper: H100/H200),
            sm_100 (Datacenter Blackwell: B100/B200/GB200),
            sm_103 (Blackwell Ultra: B300, GB300 NVL72),
            sm_120 (Consumer Blackwell: RTX 50, RTX PRO 6000 Blackwell)
  - AMD:    gfx942 (CDNA3: MI300X, MI300A),
            gfx950 (CDNA4: MI350X, MI355X)
  - CPU:    x86_64 (AVX-512), aarch64 (NEON) -- testing only

Build fails on unsupported arches. There is no generic-kernel fallback
and no tier fallback chain. See REFACTOR_PLAN.md (esp. §10) and
csrc/kernels/README.md for the underlying policy.

Build:
    pip install -e .

For build-only without a physical GPU:
    FORCE_CUDA=1 pip install -e .

To build for a specific arch subset:
    TORCH_CUDA_ARCH_LIST="9.0" pip install -e .            # Hopper only
    TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0" pip install -e .   # all NVIDIA
"""

import glob
import os
import platform as _platform
import shutil

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension


# ----------------------------------------------------------------------
# Build-mode env vars (read by build.sh wrapper):
#   CUDA_DEBUG=1     -> add -G -O0 -lineinfo, drop --use_fast_math
#   AUTOTUNE_PASS=1  -> first-pass autotune build (informational only;
#                       autotune writes tuned_configs.h between passes)
#   FORCE_CUDA=1     -> permit configuring without a visible GPU
# ----------------------------------------------------------------------

_cuda_debug = os.environ.get("CUDA_DEBUG", "0") == "1"
_autotune_pass = os.environ.get("AUTOTUNE_PASS", "0") == "1"


# ----------------------------------------------------------------------
# Compiler launcher detection: ccache/sccache wrap nvcc/hipcc cleanly.
# torch's BuildExtension respects CXX/NVCC envs and PATH; we expose the
# launcher via env vars that nvcc and hipcc honor through their wrapper
# behavior, and additionally surface CMAKE_CUDA_COMPILER_LAUNCHER for
# any cmake-driven sub-builds.
# ----------------------------------------------------------------------

def _detect_launcher():
    for cand in ("ccache", "sccache"):
        path = shutil.which(cand)
        if path:
            return cand, path
    return None, None


_launcher_name, _launcher_path = _detect_launcher()
if _launcher_path:
    print(f"  Compiler launcher: {_launcher_name} -> {_launcher_path}")
    os.environ.setdefault("CMAKE_CUDA_COMPILER_LAUNCHER", _launcher_path)
    os.environ.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", _launcher_path)
    # Torch's BuildExtension reads CUDA_NVCC_EXECUTABLE if set; pointing
    # it at "<launcher> nvcc" is not portable across torch versions, so
    # we instead rely on launcher-on-PATH masquerading (the standard
    # ccache/sccache install style) and let torch invoke nvcc normally.
else:
    print("  Compiler launcher: none (ccache/sccache not found)")


# ----------------------------------------------------------------------
# Backend detection. Same conventions as before; the supported set is
# narrowed.
# ----------------------------------------------------------------------

_is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
_force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
_has_gpu = torch.cuda.is_available() or (_force_cuda and torch.version.cuda is not None)

# ----------------------------------------------------------------------
# Source lists. Walk the new csrc/kernels/ tree; never csrc/cuda/generic/.
# ----------------------------------------------------------------------

def _collect(globs):
    out = []
    for g in globs:
        out.extend(sorted(glob.glob(g)))
    # All *_overlay.* files have been merged into per-arch canonical
    # kernels in csrc/kernels/{cuda,hip}/<arch>/. The filter is
    # retained as a no-op safety net in case a stray overlay file is
    # checked in by accident.
    return [s for s in out if "_overlay" not in os.path.basename(s)]


COMMON_BINDINGS = [
    "csrc/bindings/dispatch.cpp",
    "csrc/bindings/grokadamw.cpp",
    "csrc/bindings/grokfast.cpp",
    "csrc/bindings/lion.cpp",
    "csrc/bindings/looksam.cpp",
    "csrc/bindings/moe.cpp",
    "csrc/bindings/multi_tensor.cpp",
    "csrc/bindings/muon.cpp",
    "csrc/bindings/neuralgrok.cpp",
    "csrc/bindings/prodigy.cpp",
    "csrc/bindings/quantization.cpp",
    "csrc/bindings/supergrok11.cpp",
    "csrc/bindings/supergrok15.cpp",
    "csrc/bindings/supergrok2.cpp",
    "csrc/bindings/distributed_scan.cpp",
    "csrc/bindings/module.cpp",
]

if _has_gpu and _is_hip:
    from torch.utils.cpp_extension import CUDAExtension

    print("Building Grokking Optimizers C++/HIP extension")
    print(f"  ROCm version: {torch.version.hip}")

    sources = COMMON_BINDINGS + _collect([
        "csrc/kernels/hip/gfx942/*.hip.cpp",
        "csrc/kernels/hip/gfx950/*.hip.cpp",
    ])

    rocm_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if rocm_archs:
        offload = []
        for a in rocm_archs.replace(",", ";").split(";"):
            a = a.strip()
            if a:
                offload.append(f"--offload-arch={a}")
        print(f"  ROCm archs (from TORCH_CUDA_ARCH_LIST): {rocm_archs}")
    else:
        offload = [
            "--offload-arch=gfx942",   # MI300X / MI300A
            "--offload-arch=gfx950",   # MI350X / MI355X
        ]

    hip_cxx = ["-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math", "-funroll-loops", "-fPIC"]
    hip_nvcc = ["-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math", "-fPIC"] + offload
    if _cuda_debug:
        # CUDA_DEBUG also affects HIP (hipcc) — drop fast-math, add -g -O0.
        hip_cxx = [f for f in hip_cxx if f != "-ffast-math"] + ["-g", "-O0"]
        hip_nvcc = [f for f in hip_nvcc if f != "-ffast-math"] + ["-g", "-O0"]
        print("  HIP build mode: DEBUG (-g -O0, fast-math disabled)")

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=sources,
        include_dirs=["csrc/common", "csrc/bindings", "csrc"],
        define_macros=[("WITH_HIP", None)],
        extra_compile_args={"cxx": hip_cxx, "nvcc": hip_nvcc},
    )

elif _has_gpu:
    from torch.utils.cpp_extension import CUDAExtension

    print("Building Grokking Optimizers C++/CUDA extension")
    print(f"  CUDA version: {torch.version.cuda}")

    sources = COMMON_BINDINGS + _collect([
        "csrc/kernels/cuda/sm_80/*.cu",
        "csrc/kernels/cuda/sm_89/*.cu",
        "csrc/kernels/cuda/sm_90/*.cu",
        "csrc/kernels/cuda/sm_100/*.cu",
        "csrc/kernels/cuda/sm_103/*.cu",
        "csrc/kernels/cuda/sm_120/*.cu",
        "csrc/quantization/*.cu",  # split per-arch in a follow-up
    ])

    nvcc_archs_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if nvcc_archs_env:
        # Honor user override; convert "8.0;9.0" to gencode flags.
        gencode = []
        for a in nvcc_archs_env.replace(",", ";").split(";"):
            a = a.strip().replace(".", "")
            if not a:
                continue
            gencode.append(f"-gencode=arch=compute_{a},code=sm_{a}")
        print(f"  CUDA archs (from TORCH_CUDA_ARCH_LIST): {nvcc_archs_env}")
    else:
        # Supported set: sm_80, sm_89, sm_90, sm_100, sm_103, sm_120.
        # sm_86 (Ampere RTX 30) routes to the sm_80 binding at runtime via
        # grokking_optimizers/dispatch.py. Pre-Ampere (sm_70/75) is unsupported.
        # AOT-only model: no NVRTC, no runtime compilation. Driver JIT
        # provides forward-compat from the embedded PTX on the highest
        # compute target (sm_120) for any future arch >= 12.0.
        gencode = [
            "-gencode=arch=compute_80,code=sm_80",         # A100, A30, A10 (Ampere family)
            "-gencode=arch=compute_89,code=sm_89",         # RTX 40-series, L40, L40S (Ada)
            "-gencode=arch=compute_90,code=sm_90",         # H100, H200 (Hopper)
            "-gencode=arch=compute_100,code=sm_100",       # B100, B200, GB200 (datacenter Blackwell)
            "-gencode=arch=compute_103,code=sm_103",       # B300, GB300 NVL72 (Blackwell Ultra)
            "-gencode=arch=compute_120,code=sm_120",       # RTX 50-series, RTX PRO 6000 (consumer Blackwell)
            "-gencode=arch=compute_120,code=compute_120",  # PTX for forward-compat JIT on > sm_120
        ]

    cuda_cxx = ["-O3", "-std=c++17", "-DWITH_CUDA", "-ffast-math", "-funroll-loops", "-fPIC"]
    cuda_nvcc = [
        "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
        "--expt-relaxed-constexpr", "-lineinfo",
        "-Xptxas", "-O3",
        "-Xptxas", "--warn-on-spills",
        "-Xcompiler", "-fPIC",
    ] + gencode

    if _cuda_debug:
        # Debug build: device debug info (-G), no opt, no fast-math.
        cuda_cxx = [f for f in cuda_cxx if f != "-ffast-math"]
        cuda_cxx = [f for f in cuda_cxx if f != "-O3"] + ["-O0", "-g"]
        cuda_nvcc = [
            "-O0", "-g", "-G", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas", "-O0",
            "-Xptxas", "--warn-on-spills",
            "-Xcompiler", "-fPIC",
        ] + gencode
        print("  CUDA build mode: DEBUG (-G -O0 -lineinfo, fast-math disabled)")
    if _autotune_pass:
        print("  CUDA build mode: AUTOTUNE_PASS=1 (first pass, stub configs)")

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=sources,
        include_dirs=["csrc/common", "csrc/bindings", "csrc"],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={"cxx": cuda_cxx, "nvcc": cuda_nvcc},
    )

else:
    # CPU build is testing-only. Not a runtime fallback.
    from torch.utils.cpp_extension import CppExtension

    print("Building Grokking Optimizers C++ CPU extension (testing only)")
    print("  CPU build is for unit tests; not a runtime fallback path.")

    cpu_sources = _collect([
        "csrc/kernels/cpu/*.cpp",
    ])
    cpu_cxx_flags = [
        "-O3", "-std=c++17", "-DWITH_CPU",
        "-ffast-math", "-funroll-loops", "-fopenmp",
    ]

    cpu_arch = _platform.machine().lower()
    if cpu_arch in ("x86_64", "amd64"):
        cpu_sources += _collect(["csrc/kernels/cpu/avx512/*.cpp"])
        cpu_cxx_flags.append("-march=native")
        print("  SIMD: x86_64 detected, AVX-512 via -march=native")
    elif cpu_arch in ("aarch64", "arm64"):
        cpu_sources += _collect(["csrc/kernels/cpu/neon/*.cpp"])
        print("  SIMD: ARM detected, NEON intrinsics enabled")
    else:
        print(f"  SIMD: unknown arch '{cpu_arch}', scalar fallback only")

    ext = CppExtension(
        name="grokking_optimizers._ops",
        sources=COMMON_BINDINGS + cpu_sources,
        include_dirs=["csrc/common", "csrc/bindings", "csrc/kernels/cpu", "csrc"],
        define_macros=[("WITH_CPU", None)],
        extra_compile_args={"cxx": cpu_cxx_flags},
        extra_link_args=["-fopenmp"],
    )


setup(
    name="grokking-optimizers",
    version="3.0.0",
    description=(
        "All-specialized per-arch optimizer kernels. "
        "Supported: NVIDIA sm_80/sm_89/sm_90/sm_100/sm_103/sm_120 (CUDA), "
        "AMD gfx942/gfx950 (HIP), TPU v5p/v6e (Pallas via JAX). "
        "CPU build for testing only. No generic-kernel fallback, no tier "
        "fallback chain."
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
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0"],
    extras_require={
        "jax":  ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "test": ["pytest", "numpy"],
    },
)
