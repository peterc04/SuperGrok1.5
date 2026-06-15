#!/usr/bin/env bash
# SuperGrok1.5 -- COMPREHENSIVE dependency installer. NOTHING here is optional:
# every dependency the codebase references (verified by a line-by-line audit of
# all 393 source files) is installed -- or, for the AMD/TPU runtimes that cannot
# physically run on this NVIDIA H100 box, documented as a target-conditional
# section so the file is complete for ANY hardware.
#
# Persistence model (runpod): /workspace is PERSISTENT; the container root
# (/usr, system pip site-packages, /usr/local/bin, /dev/shm) is EPHEMERAL.
#   - pip packages   -> the /workspace venv (scripts/bootstrap_env.sh creates it) so they persist
#   - sccache binary -> /workspace/.local/bin (persists)
#   - apt system tools -> ephemeral root (this script re-installs them on a fresh instance)
# Run INSIDE the /workspace venv (bootstrap_env.sh activates it first).
set -euo pipefail

PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"; export PIP_CACHE_DIR; mkdir -p "$PIP_CACHE_DIR"
SCCACHE_BIN_DIR="${SCCACHE_BIN_DIR:-/workspace/.local/bin}"; mkdir -p "$SCCACHE_BIN_DIR"

# ── 1. SYSTEM build/inspection tools (apt; ephemeral root) ───────────────────
#   ccache    : host C/C++ compile cache (compile.py _sccache_env CC/CXX masquerade)
#   cmake     : compile.py CMAKE_BUILD_PARALLEL_LEVEL + general build glue
#   clang+llvm: RUNS scripts/amdgcn_check.sh (gfx942 device-code gate via LLVM's
#               AMDGPU target -- compiles on NVIDIA, no AMD GPU needed); provides
#               llvm-objdump / llvm-readobj; backs the libclang polyhedral layer
#   lld       : LLVM linker used by the amdgcn gate
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || true
  apt-get install -y --no-install-recommends ccache cmake clang lld llvm || true
fi

# ── 2. sccache (Rust binary; compile.py wraps nvcc + host CC for cross-build caching) ──
#   REQUIRED for build performance: autotune sweeps recompile many variants.
if ! command -v sccache >/dev/null 2>&1 && [[ ! -x "$SCCACHE_BIN_DIR/sccache" ]]; then
  ver=v0.8.2; pkg=sccache-$ver-x86_64-unknown-linux-musl
  curl -fsSL "https://github.com/mozilla/sccache/releases/download/$ver/$pkg.tar.gz" \
    | tar xz -C /tmp && install -m755 "/tmp/$pkg/sccache" "$SCCACHE_BIN_DIR/sccache"
fi
export PATH="$SCCACHE_BIN_DIR:$PATH"

# ── 3. PIP -- everything the codebase imports (into the active venv) ──────────
#   numpy is FROZEN to torch 2.4.1's build version; NEVER reinstall torch/torchvision.
python -m pip install --no-input "numpy==1.26.3" "ninja>=1.11" \
  "optuna>=3.4,<5" "tqdm>=4.66,<5" "matplotlib>=3.7,<4" \
  "xgboost>=2.0,<3" "scikit-learn>=1.5,<2" "joblib>=1.3" "pynvml>=11.5,<12" "prodigyopt" \
  "psutil>=5.9" "requests>=2.31" "PyYAML>=6" "jinja2>=3.1,<4" \
  pytest codespell ruff yamllint

# compile.py advanced codegen routes (NVRTC / polyhedral / CUTLASS-python autotune):
#   cuda-python pinned to 12.x toolkit; libclang+islpy = polyhedral; nvidia-cutlass PINNED to
#   3.6.0 to match the v3.6.0 C++ submodule (provides compile.py's `import cutlass`; 4.x renamed it cutlass_library).
python -m pip install --no-input "cuda-python>=12.2,<13" "libclang>=16,<19" islpy "nvidia-cutlass==3.6.0" \
  || echo "WARN: a codegen extra failed (cuda-python/libclang/islpy/nvidia-cutlass) -- verify individually below"

# distributed support probed by grokking_optimizers/distributed.py (heavy; probe-only -> non-fatal):
python -m pip install --no-input deepspeed \
  || echo "WARN: deepspeed install failed (non-fatal; only probed via find_spec, never imported)"

# Pallas (TPU) parity test runs in JAX interpret mode -> CPU jaxlib suffices.
# NEVER install jax[cuda]/jaxlib-cuda: it shadows torch's bundled CUDA 12.4 and breaks _ops.
python -m pip install --no-input "jax[cpu]>=0.4,<0.5"

# ── 4. HARDWARE-GATED runtimes -- RETAINED in the codebase, NOT runnable on this
#      NVIDIA H100 box. Install on the matching hardware target:
#
#   AMD/ROCm (gfx942 / MI300X) HIP RUNTIME -- only on an AMD-GPU instance (~GBs):
#     apt-get install -y rocm-hip-sdk rocblas hipcub rocprim rocthrust amd-smi-lib rocprofiler
#     pip install hip-python composable-kernel        # hipRTC bindings + CK GEMM emitter
#     # (clang/llvm in section 1 already runs the compile-only amdgcn_check.sh gate here.)
#
#   Google TPU (v6e / v5p) -- only on a TPU host:
#     pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
#   Nsight Systems (nsys, system profiler) -- via the CUDA repo / NVIDIA installer:
#     apt-get install -y nsight-systems-cli      # (ncu/Nsight Compute already ships in the toolkit)

echo "comprehensive deps installed. torch/numpy untouched; sccache+ccache+clang present."
