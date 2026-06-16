"""
Grokking Optimizers — All-Specialized Per-Arch Kernel Build

Supported arches (3-arch active set):
  - NVIDIA: sm_90 (Hopper: H100/H200)
  - AMD:    gfx942 (CDNA3: MI300X, MI300A)
  - TPU:    v6e (Pallas via JAX)

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
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
import torch.utils.cpp_extension as _torch_cpp_ext


# ----------------------------------------------------------------------
# Lever (b): autotuner → product-build linkage.
#
# Load the pure-stdlib injector module (grokking_optimizers/_tuned_inject.py)
# BY FILE PATH so importing it never re-enters grokking_optimizers/__init__.py
# (which imports torch + probes CUDA). It is the single source of truth for
# the canonical _kernel_tuned.json schema, the TU->optimizer mapping, and the
# per-TU -DSG_TUNED_* / --maxrregcount flag computation. If it cannot be
# loaded, the build degrades to identical-to-today (in-header defaults) with a
# loud notice — never a hard failure.
# ----------------------------------------------------------------------
def _load_tuned_inject():
    try:
        mod_path = os.path.join(_REPO_ROOT, "grokking_optimizers",
                                "_tuned_inject.py")
        spec = importlib.util.spec_from_file_location(
            "grokking_optimizers_tuned_inject", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:  # noqa: BLE001 — optional; degrade gracefully.
        print(f"  [tuned-inject] could not load _tuned_inject.py ({exc}); "
              "build will use in-header kernel defaults (no per-TU tuning).")
        return None

# ----------------------------------------------------------------------
# Absolute repo root (directory containing this setup.py). Include dirs MUST
# be absolute: torch's BuildExtension runs ninja with cwd=build_temp, so a
# relative "-I." resolves to the build-temp dir, NOT the project root — and the
# project-root-relative  #include "csrc/fused/sm_90/..."  in dispatch.cpp would
# not be found (fatal error: No such file or directory). Anchoring every
# include dir to abspath(__file__) makes the build work regardless of the
# invoking cwd (pip editable PEP 660, `setup.py build_ext`, or wheel build),
# matching what scripts/compile_to_object.sh achieves by cd-ing to the root.
# ----------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abs_incs(dirs):
    """Anchor repo-relative include dirs at the repo root; pass absolutes through."""
    return [d if os.path.isabs(d) else os.path.join(_REPO_ROOT, d) for d in dirs]

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
#   CUDA_DEBUG=1     -> add -G -O0 -lineinfo, drop --use_fast_math + -DNDEBUG
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


def _resolve_real_nvcc():
    """The nvcc BINARY torch's cpp_extension would invoke ($CUDA_HOME/bin/
    nvcc), for wrapping in a compiler cache. Mirrors compile.py::
    _resolve_real_nvcc. Falls back to PATH-spelled "nvcc" only when torch/
    CUDA_HOME cannot name a real file."""
    try:
        cuda_home = getattr(_torch_cpp_ext, "CUDA_HOME", None)
        if cuda_home:
            cand = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.isfile(cand):
                return cand
    except Exception:  # noqa: BLE001 — fall back to PATH form.
        pass
    return "nvcc"


def _sccache_nvcc_env():
    """Fold sccache into the _ops build by wrapping nvcc via torch's
    SUPPORTED PYTORCH_NVCC knob — mirrors compile.py::_sccache_env so that
    repeated `pip install -e .` / `setup.py build_ext` runs hit the same
    on-disk CUDA object cache the autotuner populates. Returns a dict of env
    vars to set (empty when sccache is not on PATH).

    PYTORCH_NVCC (not CUDA_NVCC_EXECUTABLE, a CMake var cpp_extension ignores)
    is the knob torch writes verbatim into build.ninja. We wrap the REAL nvcc
    binary ($CUDA_HOME/bin/nvcc), NOT PATH-spelled "nvcc": on pods where the
    first nvcc on PATH is itself a caching wrapper SCRIPT, sccache would be
    asked to treat a bash script as the compiler and die with "Compiler not
    supported" (observed 2026-06-12). On hosts whose PATH nvcc IS the real
    binary this resolves to the same compiler. Host CC/CXX are left to the
    launcher-on-PATH masquerade above; this targets the .cu TUs only."""
    out = {}
    sccache = shutil.which("sccache")
    if not sccache:
        return out
    sc_dir = os.environ.get("SCCACHE_DIR")
    if sc_dir:
        out.setdefault("SCCACHE_DIR", sc_dir)
    out["PYTORCH_NVCC"] = f"{sccache} {_resolve_real_nvcc()}"
    for k in ("SCCACHE_REDIS_ENDPOINT", "SCCACHE_REDIS", "SCCACHE_S3_BUCKET"):
        if k in os.environ:
            out[k] = os.environ[k]
    return out


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
    # checked in by accident. *_selftest.cu TUs (e.g. wgmma_selftest.cu)
    # define their OWN pybind module and are JIT-loaded by their tests —
    # linking them into _ops would collide on PyInit__ops (multiple
    # definition; observed at the wgmma substrate landing).
    #
    # The same collision is produced by any standalone-JIT cell driver that
    # owns a PYBIND11_MODULE(TORCH_EXTENSION_NAME, …) and is loaded directly
    # via torch.utils.cpp_extension.load (e.g. csrc/fused/sm_90/
    # mega_decoder_real_adamw_tc.cu — the R2.3 Fork-B TC cell, JIT-loaded by
    # tests/hw/test_decoder_tc.py PART 2; its own header states "No setup.py
    # glob change"). These slip past the *_selftest.cu name filter, so we also
    # drop, content-wise, any .cu that declares its own
    # PYBIND11_MODULE(TORCH_EXTENSION_NAME …): inside _ops that macro re-emits
    # PyInit__ops and (via its #included megakernel header) the decoder dec_*
    # helpers, both already defined by bindings.cpp / the scalar cell TU. The
    # check is content-based so it auto-tracks new standalone-JIT drivers
    # regardless of naming. (.cu only — the standalone-JIT pattern is CUDA;
    # HIP cell TUs use the SG_GFX942_DEVICE_TU guard, not their own module.)
    def _owns_extension_module(path: str) -> bool:
        if not path.endswith(".cu"):
            return False
        try:
            with open(path, "r", errors="ignore") as fh:
                return "PYBIND11_MODULE(TORCH_EXTENSION_NAME" in fh.read()
        except OSError:
            return False

    return [s for s in out
            if "_overlay" not in os.path.basename(s)
            and not os.path.basename(s).endswith("_selftest.cu")
            and not _owns_extension_module(s)]


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
        # HIP/CUDA pipeline, running a DEVICE pass). Use this extension for
        # hand-written `__global__` kernels with `hipLaunchKernelGGL` syntax.
        #
        # Stage 7 (hipcc device routing): the per-optimizer / per-model kernel
        # headers (grokking_optimizers/kernels/gfx942/*_gfx942.hip.hpp) carry a
        # DEVICE section gated `#if __AMDGCN__||__HIPCC__`, but their only
        # includers were `.hip.cpp` HOST TUs — so hipcc's device pass never
        # compiled the MFMA/DPP/FP8 kernels. The thin `device_*.hip` TUs here
        # (one per optimizer/model) `#define SG_GFX942_DEVICE_TU` then #include
        # the canonical header, forcing the device pass to emit the kernels
        # (the header force-instantiates them) while the host orchestration
        # stays owned by the `.hip.cpp` launchers (the guard suppresses a
        # duplicate-symbol re-emit). This glob already matches them.
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

    hip_cxx = ["-O3", "-std=c++17", "-DWITH_HIP", "-DNDEBUG", "-ffast-math", "-funroll-loops", "-fPIC"]
    hip_nvcc = ["-O3", "-std=c++17", "-DWITH_HIP", "-DNDEBUG", "-ffast-math", "-fPIC"] + offload

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
        # CUDA_DEBUG also affects HIP (hipcc) — drop fast-math + NDEBUG (restore
        # asserts), add -g -O0.
        hip_cxx = [f for f in hip_cxx if f not in ("-ffast-math", "-DNDEBUG")] + ["-g", "-O0"]
        hip_nvcc = [f for f in hip_nvcc if f not in ("-ffast-math", "-DNDEBUG")] + ["-g", "-O0"]
        print("  HIP build mode: DEBUG (-g -O0, fast-math disabled)")

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=sources,
        # csrc/common/ and csrc/scan/ were deleted; their content is
        # inlined into every backend file. Only csrc/bindings/ (the one
        # surviving cross-file boundary) and csrc/ (for algorithm-header
        # absolute-path includes from launch files) remain.
        include_dirs=_abs_incs([".", "csrc/bindings", "csrc"]),
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

    cuda_cxx = ["-O3", "-std=c++17", "-DWITH_CUDA", "-DNDEBUG", "-ffast-math", "-funroll-loops", "-fPIC"]
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
        # compile.NVCC_DEVICE_BASE is now STRICT-MATH (mandate #6) and carries
        # NO project opinion (NDEBUG migrated out). setup.py IS the grokking
        # project, so it re-adds grokking's OWN flag decisions here — exactly
        # the [device_cflags].extra = ["-DNDEBUG"] the autotuner sources from
        # compile_config.toml, plus grokking's chosen --use_fast_math. This
        # keeps the install build byte-aligned with the historical fast path
        # (WGMMA de-serialization, fused-math) while the universal tool's base
        # stays opinion-free. CUDA_DEBUG strips both below.
        _grok_proj_nvcc = ["--use_fast_math", "-DNDEBUG"]
        cuda_nvcc = (list(_COMPILE_NVCC_BASE) + _grok_proj_nvcc
                     + ["-lineinfo"] + gencode)
        print("  NVCC base flags: compile.NVCC_DEVICE_BASE (strict) "
              "+ grokking project flags (--use_fast_math, -DNDEBUG)")
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
        # Debug build: device debug info (-G), no opt, no fast-math, asserts on.
        cuda_cxx = [f for f in cuda_cxx if f not in ("-ffast-math", "-DNDEBUG")]
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
            # Debug build restores asserts: drop -DNDEBUG along with the
            # release opt/fast-math flags so device + CUTLASS assert() fire.
            _dbg = [f for f in _COMPILE_NVCC_BASE
                    if f not in ("-O3", "--use_fast_math", "-DNDEBUG")]
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

    # Strip host-side LTO (-Xcompiler -flto / -flto=auto) from the INSTALL build.
    # nvcc emits one `fatbinData` symbol per CUDA TU; when distutils links the
    # final .so it drives gcc's LTO over the thin-LTO objects, and lto-wrapper
    # merges the per-TU fatbinData into one assembly → "symbol fatbinData is
    # already defined" (a hard link failure). LTO of the thin host glue is a
    # negligible win for a GPU-kernel extension, and the autotuner's own flag
    # tables (compile.py) keep -flto for its standalone single-TU probes — this
    # only de-LTOs the multi-TU install link so `pip install -e .` / build_ext
    # link reliably.
    def _strip_flto(flags):
        out = []
        for f in flags:
            if "flto" in f:
                if out and out[-1] == "-Xcompiler":
                    out.pop()           # drop the paired `-Xcompiler` token too
                continue
            out.append(f)
        return out
    cuda_nvcc = _strip_flto(cuda_nvcc)
    cuda_cxx = _strip_flto(cuda_cxx)

    ext = CUDAExtension(
        name="grokking_optimizers._ops",
        sources=sources,
        include_dirs=_abs_incs(cuda_include_dirs),
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


# ----------------------------------------------------------------------
# Lever (b): per-TU tuned-flag injection at build time.
#
# PROBLEM torch's BuildExtension cannot express directly: extra_compile_args
# is PER-LANGUAGE (one 'nvcc' list for every .cu), not per-source. The
# autotuner picks DIFFERENT launch params (block/vec/unroll/async_depth +
# maxrregcount) per optimizer, so each optimizer's TU needs its OWN nvcc
# flags. torch 2.4.1's ninja path (unix_wrap_ninja_compile ->
# _write_ninja_file_and_compile_objects -> _write_ninja_file) computes ONE
# shared `cuda_post_cflags` ninja variable applied to every .cu via a single
# `rule cuda_compile` (see /usr/local/lib/python3.11/dist-packages/torch/
# utils/cpp_extension.py:2260-2368). It emits one `build <obj>: cuda_compile
# <src>` statement per source but no per-statement flag override.
#
# HOOK CHOSEN: monkeypatch the MODULE-LEVEL torch.utils.cpp_extension.
# _write_ninja_file (the function that emits those per-source build
# statements and receives `sources`, `objects`, `cuda_post_cflags`) for the
# duration of super().build_extensions(). We call the ORIGINAL writer
# unchanged (so build.ninja is byte-correct), then APPEND a per-build-
# statement `cuda_post_cflags = <FULL list>` override line under each TU that
# maps to a tuned optimizer. We do NOT touch self.compiler.compile (torch
# rebinds it inside build_extensions, so the timing of that binding is
# irrelevant — whatever compile fn torch ends up calling routes through OUR
# _write_ninja_file). All per-TU flags land in the SAME build.ninja, so the
# runbook's `grep build.ninja for SG_TUNED + maxrregcount` succeeds.
#
# NINJA SCOPING (load-bearing): each override emits the FULL flag list
# (global cuda_post_cflags + that optimizer's extras), NOT a self-referential
# `cuda_post_cflags = $cuda_post_cflags ...` — a wrong eval order would
# silently drop every base nvcc flag (arch/-O3/fast-math) and ship a
# wrong-but-compiling kernel. We have the full list at write time; we emit it.
# Extras are shlex.quoted to match torch's own quoting of the base flags.
#
# DEGRADES GRACEFULLY: no _kernel_tuned.json -> identical-to-today build
# (one-line notice). _write_ninja_file absent in a future torch -> loud
# warning + stock build, never a hard failure (hasattr-guarded).
# ----------------------------------------------------------------------
_TUNED_INJECT = _load_tuned_inject()
# Arch key for the JSON lookup: the JSON uses the short, user-facing key.
# NOTE: this phase only emits per-TU flags onto CUDA (.cu) TUs (sm_90). A
# "gfx942" key here is wired but currently a NO-OP — optimizer_for_source()
# rejects non-.cu sources, and the HIP VGPR-cap equivalent
# (-amdgpu-max-num-vgprs) is intentionally out of scope (see _tuned_inject.py
# docstring + AUTOTUNE_LINKAGE.md §5). So on a HIP build nothing is injected.
_BUILD_ARCH_KEY = "gfx942" if _is_hip else "sm_90"


class TunedBuildExtension(BuildExtension):
    """BuildExtension that injects per-TU autotuned nvcc flags (lever (b))."""

    def build_extensions(self):
        import shlex as _shlex

        # sccache fold-in (guarded): wrap nvcc via PYTORCH_NVCC so _ops
        # rebuilds hit the same CUDA object cache compile.py's autotuner
        # populates. Set into the live env here (BEFORE super() writes
        # build.ninja, which reads PYTORCH_NVCC). No-op when sccache is not
        # on PATH; never raised — diagnostic only. Runs on every build path,
        # ahead of the tuned-inject early-returns below.
        try:
            _sc_env = _sccache_nvcc_env()
            if _sc_env:
                for _k, _v in _sc_env.items():
                    os.environ.setdefault(_k, _v)
                print(f"  [sccache] nvcc wrapped via PYTORCH_NVCC = "
                      f"{os.environ.get('PYTORCH_NVCC')!r} (cached _ops "
                      "rebuilds).")
        except Exception as _scexc:  # noqa: BLE001 — never break the build.
            print(f"  [sccache] fold-in skipped ({_scexc}); building with "
                  "the unwrapped toolchain.")

        inject = _TUNED_INJECT
        tuned = inject.load_tuned() if inject is not None else None

        # No injector module OR no JSON -> stock build (identical to today).
        if inject is None:
            print("  [tuned-inject] injector module unavailable; building "
                  "with in-header kernel defaults (no per-TU tuning).")
            return super().build_extensions()
        if not tuned:
            print(f"  [tuned-inject] no {os.path.basename(inject.KERNEL_TUNED_JSON)} "
                  "found; building with in-header kernel defaults. Run "
                  "`python -m grokking_optimizers.compile ... --jit-only` to "
                  "produce winners (see AUTOTUNE_LINKAGE.md).")
            return super().build_extensions()

        # Drift guard (design 1c): the MACROS table must agree with the
        # kernel header #ifndef defaults. We do not have a GPU here, but we
        # can confirm the macro NAMES we are about to emit are the ones the
        # header guards. Best-effort, never fatal.
        _verify_macro_names(inject)

        orig_writer = getattr(_torch_cpp_ext, "_write_ninja_file", None)
        if orig_writer is None:
            # Future/older torch without this internal -> cannot inject.
            print("  [tuned-inject] WARNING: torch.utils.cpp_extension."
                  "_write_ninja_file is unavailable in this torch "
                  f"({getattr(torch, '__version__', '?')}); per-TU tuned "
                  "flags will NOT be injected — building with in-header "
                  "defaults. (The tuned JSON exists but cannot be applied.)")
            return super().build_extensions()

        arch_key = _BUILD_ARCH_KEY

        def _patched_writer(*args, **kwargs):
            # torch calls _write_ninja_file entirely by keyword
            # (cpp_extension.py:1771); read what we need from kwargs.
            orig_writer(*args, **kwargs)
            path = kwargs.get("path")
            sources = kwargs.get("sources")
            objects = kwargs.get("objects")
            cuda_post_cflags = kwargs.get("cuda_post_cflags")
            if not (path and sources and objects):
                return
            try:
                n = inject.inject_overrides_into_ninja(
                    path, list(sources), list(objects),
                    list(cuda_post_cflags or []), tuned, arch_key,
                    quote=_shlex.quote)
                if n:
                    print(f"  [tuned-inject] injected per-TU tuned flags into "
                          f"{n} CUDA TU(s) in {os.path.basename(path)}.")
            except Exception as exc:  # noqa: BLE001 — never break the build.
                print(f"  [tuned-inject] WARNING: per-TU override append "
                      f"failed ({exc}); build.ninja left as torch wrote it "
                      "(in-header defaults apply). Build continues.")

        _torch_cpp_ext._write_ninja_file = _patched_writer
        try:
            return super().build_extensions()
        finally:
            _torch_cpp_ext._write_ninja_file = orig_writer


def _verify_macro_names(inject):
    """Best-effort drift check: the macro names we emit are guarded by the
    kernel header. Loud warning on mismatch; never fatal (design 1c)."""
    try:
        header = os.path.join(_REPO_ROOT, "grokking_optimizers", "kernels",
                              "sm_90", "adamw_sm90.cuh")
        if not os.path.isfile(header):
            return
        with open(header, "r", encoding="utf-8") as fh:
            text = fh.read()
        missing = [name for name in inject.header_default_macros()
                   if name not in text]
        if missing:
            print("  [tuned-inject] WARNING: macro name(s) in the injector "
                  f"source-of-truth are NOT guarded by {os.path.basename(header)}: "
                  f"{missing}. Emitting them anyway, but they may be silent "
                  "no-ops — reconcile _tuned_inject.MACROS with the header "
                  "(#ifndef SG_TUNED_*).")
    except Exception:  # noqa: BLE001 — diagnostic only.
        pass


setup(
    name="grokking-optimizers",
    version="3.0.0",
    description=(
        "All-specialized per-arch optimizer kernels. "
        "Supported: NVIDIA sm_90 (CUDA), AMD gfx942 (HIP), TPU v6e (Pallas via JAX)."
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
            # NOTE: kernels/sm_90/*.cuh removed — the eager per-op/model sm_90
            # device headers (adamw_sm90.cuh … transformer_decoder_sm90.cuh,
            # common_sm90.cuh) were deleted in the L3-TC-only cut; the sm_90
            # device path now lives under csrc/fused/sm_90/ (megakernel +
            # _tc model/opt stages), which is not an importable package dir
            # and is shipped via MANIFEST.in (sdist). kernels/sm_90/ retains
            # only __init__.py, so this glob matched nothing post-cut.
            "kernels/gfx942/*.hip.hpp",
            "kernels/gfx942/*.cuh",
            # Lever (b): ship the canonical autotuner winners when a wheel is
            # built on a tuned host (gitignored in source — see .gitignore).
            # Absent, the build uses in-header defaults.
            "_kernel_tuned.json",
        ],
    },
    include_package_data=True,
    ext_modules=[ext],
    # Lever (b): TunedBuildExtension injects per-TU autotuned nvcc flags from
    # grokking_optimizers/_kernel_tuned.json (degrades to stock BuildExtension
    # behavior when the JSON is absent). use_ninja=True is REQUIRED — the
    # injection hook patches the ninja-path _write_ninja_file.
    cmdclass={"build_ext": TunedBuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0"],
    extras_require={
        "jax":  ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "test": ["pytest", "numpy"],
    },
)
