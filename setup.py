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
import re
import shutil
import subprocess
import tempfile

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

# ----------------------------------------------------------------------
# Stage-2 unification (Phase 8): source the CUDA device flags from the
# autotuner's own flag tables so the install build compiles against the
# SAME base flags the autotuner tunes against. compile.py exposes a
# GPU-less module-level NVCC base list and the gencode-pair helper; we
# import those low-risk shared constants here. The full version-gated
# ladder (_newer_compiler_flags) and the per-arch _device_cflags() entry
# require a BuildSpec / live nvcc probe, so we do NOT pull those into the
# install path yet.
#   TODO(stage-2 follow-up): once compile.py exposes a GPU-less
#   _device_cflags entry (BuildSpec-free, arch-string in / flag-list out),
#   replace the hand-maintained cuda_nvcc base below with a direct call so
#   the version-gated ladder (--threads, -dlto, --allow-expensive-
#   optimizations, etc.) is shared too. Tracked with the compile.py agent.
# ----------------------------------------------------------------------
try:
    from grokking_optimizers.compile import (
        NVCC_DEVICE_BASE as _COMPILE_NVCC_BASE,
        _nvcc_gencode_pair as _compile_gencode_pair,
    )
except Exception as _compile_import_err:  # pragma: no cover - defensive
    _COMPILE_NVCC_BASE = None
    _compile_gencode_pair = None
    print(f"  (compile.py flag tables unavailable: {_compile_import_err}; "
          "using setup.py-local fallback flags)")


# ----------------------------------------------------------------------
# Build-mode env vars (read by build.sh wrapper):
#   CUDA_DEBUG=1     -> add -G -O0 -lineinfo, drop --use_fast_math
#   AUTOTUNE_PASS=1  -> first-pass autotune build (informational only;
#                       autotune writes tuned_configs.h between passes)
#   FORCE_CUDA=1     -> permit configuring without a visible GPU
#   WITH_CUTLASS     -> route Hopper GEMMs (SG2 projections +
#                       Muon Newton-Schulz) through CUTLASS instead of
#                       cuBLAS. Requires third_party/cutlass cloned via
#                       `git submodule update --init`. Only emits CUTLASS
#                       arch flags for sm_90 and restricts the multi-arch
#                       gencode set to Hopper+ (the Sm90 collective will not
#                       build for older targets); gfx942 stays on rocBLAS.
#                       POLICY (Phase 8 Stage 2): WITH_CUTLASS now DEFAULTS ON
#                       when (a) a CUDA >= 12 toolkit is the active backend AND
#                       (b) the cutlass headers are present at
#                       third_party/cutlass/include. This makes the install
#                       default consistent with the tuned path: the shipped
#                       sm_90 GEMM #ifdef CUTLASS branch is the one the
#                       autotuner targets, so an install that silently fell
#                       back to cuBLAS would ship un-tuned kernels. Set
#                       WITH_CUTLASS=0 to force the cuBLAS fallback (explicit
#                       escape hatch); WITH_CUTLASS=1 forces it on even if the
#                       auto-detect would have left it off.
#   TORCH_CUDA_ARCH_LIST -> override the default multi-arch build. Unset, the
#                       build targets every NVIDIA CC / AMD gfx the installed
#                       toolchain accepts (probe-filtered fat binary).
# ----------------------------------------------------------------------

_cuda_debug = os.environ.get("CUDA_DEBUG", "0") == "1"
_autotune_pass = os.environ.get("AUTOTUNE_PASS", "0") == "1"


def _cutlass_headers_present():
    """True if the cutlass submodule headers are checked out."""
    return os.path.isfile(
        os.path.join("third_party", "cutlass", "include", "cutlass", "cutlass.h")
    ) or os.path.isdir(os.path.join("third_party", "cutlass", "include", "cutlass"))


def _cuda_major_ge_12():
    """True if the active CUDA toolkit (per torch) is CUDA >= 12."""
    cuda_ver = getattr(torch.version, "cuda", None)
    if not cuda_ver:
        return False
    try:
        return int(cuda_ver.split(".")[0]) >= 12
    except (ValueError, IndexError):
        return False


# CUTLASS opt-in policy (Phase 8 Stage 2): default ON when the tuned path is
# actually buildable — a CUDA >= 12 toolkit AND the cutlass headers present —
# so the shipped sm_90 GEMM does not silently #ifdef out to cuBLAS. The env
# var, when set, always wins: WITH_CUTLASS=0 is the explicit escape hatch.
_cutlass_env = os.environ.get("WITH_CUTLASS")
if _cutlass_env is not None:
    _with_cutlass = _cutlass_env == "1"
    print(f"  CUTLASS: WITH_CUTLASS={_cutlass_env} (explicit env override)")
else:
    _auto_cutlass = _cuda_major_ge_12() and _cutlass_headers_present()
    _with_cutlass = _auto_cutlass
    if _auto_cutlass:
        print("  CUTLASS: auto-enabled (CUDA>=12 toolkit + cutlass headers "
              "present); set WITH_CUTLASS=0 to force cuBLAS fallback")
    else:
        _why = []
        if not _cuda_major_ge_12():
            _why.append("CUDA<12 or non-CUDA backend")
        if not _cutlass_headers_present():
            _why.append("third_party/cutlass/include missing "
                        "(run: git submodule update --init)")
        print(f"  CUTLASS: auto-disabled ({'; '.join(_why)}); "
              "set WITH_CUTLASS=1 to force on")


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
# Multi-arch targets. The kernel source is single-per-vendor (one CUDA
# tree, one HIP tree) and portable: the arch-specific paths (CUTLASS Sm90,
# wgmma, MFMA) are #ifdef / __CUDA_ARCH__-gated, and the default path is
# ATen/cuBLAS/rocBLAS. So we compile that one source for EVERY arch the
# installed toolchain accepts — a fat binary covering the full picture —
# rather than only sm_90 / gfx942. Runtime dispatch (csrc/bindings/dispatch.cpp)
# vendor-routes any NVIDIA device to the sm90 impl and any AMD device to the
# gfx942 impl. The candidate lists mirror grokking_optimizers/compile.py's
# ARCH_TABLE; each candidate is probe-compiled and silently dropped if the
# toolchain rejects it, so adding new (or very old) arches never hard-fails
# the build — it degrades to whatever nvcc/hipcc supports (min: sm_90 / gfx942).
# ----------------------------------------------------------------------

_NVIDIA_CCS = ["70", "75", "80", "86", "89", "90", "100", "103", "120"]
_AMD_GFXS = [
    "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1030", "gfx1100", "gfx1101", "gfx1102",
    "gfx1151", "gfx1200", "gfx1201",
]


def _toolchain_accepts(compiler, probe_flags, suffix):
    """True if ``<compiler> <probe_flags> -c <trivial-kernel>`` succeeds.

    Used to filter arch targets down to what the installed toolchain actually
    supports. Missing compiler or any failure -> False (target dropped).
    """
    exe = shutil.which(compiler)
    if not exe:
        return False
    try:
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "probe" + suffix)
            with open(src, "w") as fh:
                fh.write("__global__ void _probe_kernel() {}\n")
            obj = os.path.join(d, "probe.o")
            res = subprocess.run(
                [exe, *probe_flags, "-c", src, "-o", obj],
                capture_output=True, timeout=120,
            )
            return res.returncode == 0
    except Exception:
        return False


# ----------------------------------------------------------------------
# Per-arch -gencode emission. Hopper+ (cc >= 90) MUST use the "a"
# (architecture-specific) target: the Sm90 CUTLASS WGMMA/TMA collective and
# the setmaxnreg / cp.async.bulk PTX the kernels emit are rejected by the
# plain sm_90 target. compile.py:_nvcc_gencode_pair already does this (it is
# the autotuner's emitter); we reuse it when importable and otherwise mirror
# its behavior locally so the install build and the tuned build agree.
# ----------------------------------------------------------------------
def _arch_suffix(cc):
    """``"a"`` for architecture-specific Hopper+ targets, else ``""``."""
    try:
        return "a" if int(cc) >= 90 else ""
    except ValueError:
        return ""


def _gencode_sass_for(cc):
    """The SASS-target gencode flag for a CC (with the Hopper+ ``a`` suffix)."""
    suf = _arch_suffix(cc)
    return f"-gencode=arch=compute_{cc}{suf},code=sm_{cc}{suf}"


def _gencode_pair_for(cc):
    """SASS + PTX-fallback gencode flags for a CC.

    Prefers compile.py's _nvcc_gencode_pair (the autotuner's own emitter) so
    the suffix/PTX policy stays identical; falls back to a local mirror.
    """
    if _compile_gencode_pair is not None:
        try:
            return list(_compile_gencode_pair(int(cc), _arch_suffix(cc)))
        except Exception:
            pass
    # Local mirror: SASS for the target SM + non-"a" PTX fallback so older
    # drivers can JIT-forward (PTX fallback is always the non-"a" compute_NN).
    return [_gencode_sass_for(cc),
            f"-gencode=arch=compute_{cc},code=compute_{cc}"]


def _supported_gencode(min_cc=0):
    """`-gencode` flags for every NVIDIA CC the installed nvcc accepts (+PTX).

    ``min_cc`` lets the CUTLASS path restrict to Hopper+ (the Sm90 collective
    will not compile for older device targets). Falls back to sm_90a-only.
    cc >= 90 emits the architecture-specific ``sm_90a`` target (see
    _arch_suffix); the probe itself uses the ``a`` form so acceptance matches
    what we actually emit.
    """
    candidates = [cc for cc in _NVIDIA_CCS if int(cc) >= min_cc]
    accepted = [
        cc for cc in candidates
        if _toolchain_accepts("nvcc", [_gencode_sass_for(cc)], ".cu")
    ]
    if not accepted:
        accepted = ["90"]  # safe fallback: Hopper only (emitted as sm_90a)
    flags = [_gencode_sass_for(cc) for cc in accepted]
    # Embed PTX of the newest accepted CC for driver-JIT forward-compat.
    # PTX fallback is always the non-"a" compute_NN so any driver can JIT it.
    newest = max(accepted, key=int)
    flags.append(f"-gencode=arch=compute_{newest},code=compute_{newest}")
    print(f"  CUDA gencode archs (toolchain-probed): "
          f"{[cc + _arch_suffix(cc) for cc in accepted]} (+PTX {newest})")
    return flags


def _supported_offload():
    """``--offload-arch`` for every AMD gfx the installed hipcc accepts."""
    accepted = [
        g for g in _AMD_GFXS
        if _toolchain_accepts("hipcc", [f"--offload-arch={g}"], ".hip")
    ]
    if not accepted:
        accepted = ["gfx942"]  # safe fallback: MI300X/MI300A only
    print(f"  ROCm offload archs (toolchain-probed): {accepted}")
    return [f"--offload-arch={g}" for g in accepted]


# ----------------------------------------------------------------------
# Backend detection. Same conventions as before; the supported set is
# narrowed.
# ----------------------------------------------------------------------

_is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
_force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
_has_gpu = torch.cuda.is_available() or (_force_cuda and torch.version.cuda is not None)


# ----------------------------------------------------------------------
# Preflight: nvcc major == torch.version.cuda major.
#
# torch's BuildExtension links the new .so against the PyTorch CUDA runtime
# (torch.version.cuda). If the nvcc on PATH is a DIFFERENT major version, two
# silent failures result: (a) a newer GPU arch (e.g. Blackwell sm_100, which
# needs nvcc 12.8+) is dropped from the fatbin even though torch supports it,
# and (b) device/host code is compiled against a CUDA major that mismatches
# the runtime torch loads, an ABI hazard. We surface this loudly. Set
# NVCC_CUDA_MAJOR_OK=1 to downgrade the raise to a warning (deliberate
# cross-major builds, e.g. CI matrices).
# ----------------------------------------------------------------------
def _nvcc_major():
    exe = shutil.which("nvcc")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=30).stdout
        m = re.search(r"release (\d+)\.", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _preflight_nvcc_torch_cuda():
    if _is_hip:
        return  # HIP build: nvcc/torch.version.cuda guard does not apply.
    torch_cuda = getattr(torch.version, "cuda", None)
    if not torch_cuda:
        return
    nvcc_maj = _nvcc_major()
    if nvcc_maj is None:
        return  # no nvcc on PATH (e.g. torch ships its own); nothing to check.
    try:
        torch_maj = int(torch_cuda.split(".")[0])
    except (ValueError, IndexError):
        return
    if nvcc_maj != torch_maj:
        msg = (
            f"CUDA major mismatch: nvcc is CUDA {nvcc_maj}.x but "
            f"torch.version.cuda is {torch_cuda} (major {torch_maj}). "
            "The extension would be compiled against a different CUDA major "
            "than the PyTorch runtime it loads into — newer arches (e.g. "
            "Blackwell sm_100, nvcc 12.8+) may be silently dropped and the "
            "ABI may mismatch. Align nvcc with PyTorch's CUDA toolkit, or set "
            "NVCC_CUDA_MAJOR_OK=1 to proceed anyway."
        )
        if os.environ.get("NVCC_CUDA_MAJOR_OK", "0") == "1":
            print(f"  WARNING: {msg}")
        else:
            raise RuntimeError(msg)
    else:
        print(f"  Preflight: nvcc CUDA {nvcc_maj}.x matches "
              f"torch.version.cuda {torch_cuda}")


if _has_gpu:
    _preflight_nvcc_torch_cuda()

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
        # Stage 6: generated L3 persistent megakernels (gfx942). The demo lives
        # as a gate-verified .hip.hpp header; the generator emits per-cell .hip
        # TUs here. Glob is guarded — the build works if the dir is sparse.
        "csrc/fused/gfx942/*.hip",
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
        # Default: every AMD gfx the installed hipcc accepts (probe-filtered).
        offload = _supported_offload()

    hip_cxx = ["-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math", "-funroll-loops", "-fPIC"]
    hip_nvcc = ["-O3", "-std=c++17", "-DWITH_HIP", "-ffast-math", "-fPIC"] + offload

    # Optional fast-attention library gates (Phase 8 Stage 2). The gfx942
    # attention header (kernels/gfx942/attention_gfx942.hip.hpp) #ifdef-routes
    # to Composable Kernel FMHA (WITH_CK) or AITER (WITH_AITER), falling back
    # to the hand-written MFMA path when neither is set. These were previously
    # phantom macros — referenced by the header but settable by no build
    # plumbing, so the header branches were dead. They are now real env gates;
    # OFF by default (the MFMA fallback is the committed, tested path).
    for _macro, _env in (("WITH_CK", "WITH_CK"), ("WITH_AITER", "WITH_AITER")):
        if os.environ.get(_env, "0") == "1":
            hip_cxx.append(f"-D{_macro}")
            hip_nvcc.append(f"-D{_macro}")
            print(f"  HIP attention: {_macro}=1 (opt-in library path)")
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
        include_dirs=[".", "csrc/bindings", "csrc"],
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
        # Stage 6: generated L3 persistent megakernels (one .cu per solver-
        # chosen cell, emitted by grokking_optimizers/megakernel_codegen.py).
        # Glob is guarded — the build works if the dir is sparse / absent.
        "csrc/fused/sm_90/*.cu",
    ])

    nvcc_archs_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if nvcc_archs_env:
        # Honor user override; convert "8.0;9.0" to gencode flags. cc >= 90
        # emits the architecture-specific sm_90a/sm_100a/... target (required
        # by the Sm90 WGMMA/TMA collective + setmaxnreg/cp.async.bulk), and
        # each arch keeps a non-"a" PTX fallback for driver-JIT forward-compat
        # (matching the probe-path policy in _supported_gencode). A trailing
        # "+PTX"/"a"/"f" suffix on a TORCH_CUDA_ARCH_LIST entry (e.g. "9.0a",
        # "12.0+PTX") is normalized away — we re-derive the suffix ourselves.
        gencode = []
        for a in nvcc_archs_env.replace(",", ";").split(";"):
            a = a.strip()
            if not a:
                continue
            # strip torch-style "+PTX" and any trailing arch-letter suffix,
            # keeping just the numeric CC.
            cc = re.sub(r"[^0-9].*$", "", a.replace(".", ""))
            if not cc:
                continue
            gencode.extend(_gencode_pair_for(cc))
        if not gencode:
            gencode = _supported_gencode(min_cc=90 if _with_cutlass else 0)
        print(f"  CUDA archs (from TORCH_CUDA_ARCH_LIST): {nvcc_archs_env} "
              f"-> {[g for g in gencode if ',code=sm_' in g]}")
    else:
        # Default: every NVIDIA CC the installed nvcc accepts (probe-filtered),
        # plus PTX of the newest for driver-JIT forward-compat. When CUTLASS is
        # on, restrict to Hopper+ (the Sm90 collective will not build for older
        # device targets); the portable cuBLAS path covers the rest.
        gencode = _supported_gencode(min_cc=90 if _with_cutlass else 0)

    cuda_cxx = ["-O3", "-std=c++17", "-DWITH_CUDA", "-ffast-math", "-funroll-loops", "-fPIC"]
    # NVCC base flags. Prefer compile.py's NVCC_DEVICE_BASE (the autotuner's
    # own base list) so the install build compiles against the SAME base flags
    # the autotuner tunes against (Phase 8 Stage 2 unification). NVCC_DEVICE_BASE
    # already carries -O3, --use_fast_math, -std=c++17, -DWITH_CUDA,
    # --expt-relaxed-constexpr, the ptxas opt/warn-on-spills/vectorization
    # knobs, -Xcompiler -fPIC, fatbin compression, etc. The version-gated
    # ladder (--threads, -dlto, --allow-expensive-optimizations) is NOT here —
    # see the module-top TODO. We fall back to the historical hand-maintained
    # list if compile.py could not be imported.
    if _COMPILE_NVCC_BASE is not None:
        cuda_nvcc = list(_COMPILE_NVCC_BASE) + ["-lineinfo"] + gencode
        print("  NVCC base flags: sourced from compile.NVCC_DEVICE_BASE "
              "(autotuner-shared)")
    else:
        cuda_nvcc = [
            "-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA",
            "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas", "-O3",
            "-Xptxas", "--warn-on-spills",
            "-Xcompiler", "-fPIC",
        ] + gencode
        print("  NVCC base flags: setup.py-local fallback "
              "(compile.py unavailable)")

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
    cuda_include_dirs = [".", "csrc/bindings", "csrc"]
    cuda_define_macros = [("WITH_CUDA", None)]
    if _with_cutlass:
        cutlass_archs = []
        # Detect requested archs: same source as `gencode` selection. When no
        # override is given the default supported set always includes Hopper,
        # so the CUTLASS Sm90 collective target (90a) applies.
        archs_src = nvcc_archs_env if nvcc_archs_env else "9.0"
        for a in archs_src.replace(",", ";").split(";"):
            cc = re.sub(r"[^0-9].*$", "", a.strip().replace(".", ""))
            if cc == "90":
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

    # Optional FlashAttention-3 gate (Phase 8 Stage 2). The sm_90 attention
    # header (kernels/sm_90/attention_sm90.cuh) #ifdef-routes to FA3 when
    # WITH_FLASH_ATTN_3 is set, else uses the in-tree Hopper attention kernel.
    # Previously a phantom macro (header-referenced, settable by no build
    # plumbing); now a real env gate, OFF by default.
    if os.environ.get("WITH_FLASH_ATTN_3", "0") == "1":
        cuda_nvcc.append("-DWITH_FLASH_ATTN_3")
        cuda_cxx.append("-DWITH_FLASH_ATTN_3")
        print("  CUDA attention: WITH_FLASH_ATTN_3=1 (opt-in FA3 path)")

    if _cuda_debug:
        # Debug build: device debug info (-G), no opt, no fast-math.
        cuda_cxx = [f for f in cuda_cxx if f != "-ffast-math"]
        cuda_cxx = [f for f in cuda_cxx if f != "-O3"] + ["-O0", "-g"]
        # Preserve CUTLASS + opt-in attention macros across the debug rebuild.
        _cutlass_dbg_flags = [
            f for f in cuda_nvcc
            if f.startswith("-DWITH_CUTLASS")
            or f.startswith("-DCUTLASS_NVCC_ARCHS")
            or f == "-DWITH_FLASH_ATTN_3"
        ]
        # Rebuild from the shared base (compile.NVCC_DEVICE_BASE when present)
        # with optimization/fast-math swapped out for device-debug, so the
        # debug build differs from release only in the debug knobs — not in
        # which base flags are present. gencode (sm_90a + PTX) is preserved.
        if _COMPILE_NVCC_BASE is not None:
            _dbg = [f for f in _COMPILE_NVCC_BASE
                    if f not in ("-O3", "--use_fast_math")]
            # swap any ptxas opt-level token to -O0 for the debug rebuild.
            _dbg = [("--opt-level=0" if f == "--opt-level=3"
                     else ("-O0" if f == "-O3" else f)) for f in _dbg]
            cuda_nvcc = ["-O0", "-g", "-G"] + _dbg + ["-lineinfo"] \
                + gencode + _cutlass_dbg_flags
        else:
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
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    # Ship the per-arch kernel headers and the math manifest inside the wheel
    # so a source/sdist consumer can re-tune / re-emit kernels. MANIFEST.in
    # covers the sdist; package_data covers the bdist/wheel for files that
    # live under an importable package.
    package_data={
        "grokking_optimizers": [
            "kernels/sm_90/*.cuh",
            "kernels/gfx942/*.hip.hpp",
            "kernels/gfx942/*.cuh",
        ],
    },
    include_package_data=True,
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0"],
    extras_require={
        "jax":  ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "test": ["pytest", "numpy"],
    },
)
