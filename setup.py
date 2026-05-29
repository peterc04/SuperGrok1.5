"""
Grokking Optimizers — All-Specialized Per-Arch Kernel Build

Supported arches (3-arch active set):
  - NVIDIA: sm_90 (Hopper: H100/H200)
  - AMD:    gfx942 (CDNA3: MI300X, MI300A)
  - TPU:    v5p (Pallas via JAX)

Build fails on unsupported arches. There is no generic-kernel fallback
and no tier fallback chain. See README.md (Architecture section)
and csrc/kernels/README.md for the underlying policy.

Build:
    pip install -e .

For build-only without a physical GPU:
    FORCE_CUDA=1 pip install -e .

To build for a specific arch subset:
    TORCH_CUDA_ARCH_LIST="9.0" pip install -e .            # Hopper only
"""

import glob
import os
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
#   WITH_CUTLASS=1   -> route Hopper GEMMs (SG2 projections +
#                       Muon Newton-Schulz) through CUTLASS instead of
#                       cuBLAS. Requires third_party/cutlass cloned via
#                       `git submodule update --init`. Only emits CUTLASS
#                       arch flags for sm_90; gfx942 stays on rocBLAS
#                       unconditionally.
# ----------------------------------------------------------------------

_cuda_debug = os.environ.get("CUDA_DEBUG", "0") == "1"
_autotune_pass = os.environ.get("AUTOTUNE_PASS", "0") == "1"
_with_cutlass = os.environ.get("WITH_CUTLASS", "0") == "1"


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
    "csrc/bindings/bindings.cpp",
    "csrc/bindings/dispatch.cpp",
]

if _has_gpu and _is_hip:
    from torch.utils.cpp_extension import CUDAExtension

    print("Building Grokking Optimizers C++/HIP extension")
    print(f"  ROCm version: {torch.version.hip}")

    sources = COMMON_BINDINGS + _collect([
        # `.hip.cpp` files go through the host compiler (PyTorch's
        # cpp_extension._is_cuda_file() doesn't match this suffix). They use
        # ATen tensor ops + rocBLAS — see each launcher's "WHY ATEN HERE"
        # block.
        "csrc/backends/hip/gfx942/*.hip.cpp",
        "csrc/backends/hip/gfx942/models/*.hip.cpp",
        # `.hip` files go through hipcc (PyTorch routes the extension to its
        # HIP/CUDA pipeline). Use this extension for hand-written
        # `__global__` kernels with `hipLaunchKernelGGL` launch syntax.
        "csrc/backends/hip/gfx942/*.hip",
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
        # csrc/common/ and csrc/scan/ were deleted; their content is
        # inlined into every backend file. Only csrc/bindings/ (the one
        # surviving cross-file boundary) and csrc/ (for algorithm-header
        # absolute-path includes from launch files) remain.
        include_dirs=["csrc/bindings", "csrc"],
        define_macros=[("WITH_HIP", None)],
        extra_compile_args={"cxx": hip_cxx, "nvcc": hip_nvcc},
    )

elif _has_gpu:
    from torch.utils.cpp_extension import CUDAExtension

    print("Building Grokking Optimizers C++/CUDA extension")
    print(f"  CUDA version: {torch.version.cuda}")

    sources = COMMON_BINDINGS + _collect([
        "csrc/backends/cuda/sm_90/*.cu",
        "csrc/backends/cuda/sm_90/models/*.cu",
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
        # Supported set: sm_90 only.
        # AOT-only model: no NVRTC, no runtime compilation. Driver JIT
        # provides forward-compat from the embedded PTX.
        gencode = [
            "-gencode=arch=compute_90,code=sm_90",         # H100, H200 (Hopper)
            "-gencode=arch=compute_90,code=compute_90",    # PTX for forward-compat
        ]

    cuda_cxx = ["-O3", "-std=c++17", "-DWITH_CUDA", "-ffast-math", "-funroll-loops", "-fPIC"]
    cuda_nvcc = [
        "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
        "--expt-relaxed-constexpr", "-lineinfo",
        "-Xptxas", "-O3",
        "-Xptxas", "--warn-on-spills",
        "-Xcompiler", "-fPIC",
    ] + gencode

    # ------------------------------------------------------------------
    # CUTLASS opt-in. Hopper passes -DCUTLASS_NVCC_ARCHS=90a, which emits
    # the "advanced" SM target (TMA + WGMMA). Without -DWITH_CUTLASS=1
    # Muon and SG2 stay on cuBLAS via torch::mm.
    #
    # Arches present in TORCH_CUDA_ARCH_LIST (or the default supported
    # set) are inspected: any of {9.0,10.0,10.3,12.0} triggers a
    # corresponding CUTLASS_NVCC_ARCHS define. Multiple targets are
    # encoded as a semicolon-joined list, matching CUTLASS conventions.
    # ------------------------------------------------------------------
    # csrc/common/ and csrc/scan/ were deleted; their content is inlined
    # into every backend file. Only csrc/bindings/ (the one surviving
    # cross-file boundary) and csrc/ remain.
    cuda_include_dirs = ["csrc/bindings", "csrc"]
    cuda_define_macros = [("WITH_CUDA", None)]
    if _with_cutlass:
        cutlass_archs = []
        # Detect requested archs: same source as `gencode` selection.
        archs_src = nvcc_archs_env if nvcc_archs_env else "9.0"
        for a in archs_src.replace(",", ";").split(";"):
            a = a.strip().replace(".", "")
            if a == "90":
                cutlass_archs.append("90a")
        if cutlass_archs:
            cuda_nvcc.append("-DWITH_CUTLASS")
            cuda_cxx.append("-DWITH_CUTLASS")
            cuda_nvcc.append(f"-DCUTLASS_NVCC_ARCHS={';'.join(cutlass_archs)}")
            cuda_include_dirs += [
                "third_party/cutlass/include",
                "third_party/cutlass/tools/util/include",
            ]
            cuda_define_macros.append(("WITH_CUTLASS", None))
            print(f"  CUTLASS enabled for archs: {cutlass_archs}")
        else:
            print("  WITH_CUTLASS=1 set but no sm_90 arch in TORCH_CUDA_ARCH_LIST; ignoring")

    if _cuda_debug:
        # Debug build: device debug info (-G), no opt, no fast-math.
        cuda_cxx = [f for f in cuda_cxx if f != "-ffast-math"]
        cuda_cxx = [f for f in cuda_cxx if f != "-O3"] + ["-O0", "-g"]
        # Preserve CUTLASS flags across the debug-mode rebuild.
        _cutlass_dbg_flags = [f for f in cuda_nvcc if f.startswith("-DWITH_CUTLASS") or f.startswith("-DCUTLASS_NVCC_ARCHS")]
        cuda_nvcc = [
            "-O0", "-g", "-G", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas", "-O0",
            "-Xptxas", "--warn-on-spills",
            "-Xcompiler", "-fPIC",
        ] + gencode + _cutlass_dbg_flags
        print("  CUDA build mode: DEBUG (-G -O0 -lineinfo, fast-math disabled)")
    if _autotune_pass:
        print("  CUDA build mode: AUTOTUNE_PASS=1 (first pass, stub configs)")

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=sources,
        include_dirs=cuda_include_dirs,
        define_macros=cuda_define_macros,
        extra_compile_args={"cxx": cuda_cxx, "nvcc": cuda_nvcc},
    )

else:
    raise RuntimeError(
        "No supported GPU backend detected. "
        "Grokking Optimizers requires CUDA (sm_90) or ROCm/HIP (gfx942). "
        "CPU builds are no longer supported. Set FORCE_CUDA=1 to build "
        "without a visible GPU."
    )


setup(
    name="grokking-optimizers",
    version="3.0.0",
    description=(
        "All-specialized per-arch optimizer kernels. "
        "Supported: NVIDIA sm_90 (CUDA), AMD gfx942 (HIP), TPU v5p (Pallas via JAX)."
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
