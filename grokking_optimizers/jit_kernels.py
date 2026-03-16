"""
SuperGrok v2 — JIT Kernel Compilation Fallback (Phase 4B)

For exotic configurations (d_inner != 16, unusual d_state, etc.) where
pre-compiled template instantiations don't exist, this module compiles
specialized CUDA kernels at runtime using torch.utils.cpp_extension.load().

Compiled kernels are cached at ~/.cache/supergrok2_jit/ to avoid
recompilation across runs.

Usage:
    from grokking_optimizers.jit_kernels import get_scan_kernel
    kernel = get_scan_kernel(d_inner=24, d_state=8)
    kernel.mamba3_scan_jit(...)
"""

import os
import hashlib
import torch
from pathlib import Path
from typing import Optional

_JIT_CACHE_DIR = Path.home() / ".cache" / "supergrok2_jit"
_loaded_kernels = {}

# Template for d_inner-specialized scan kernel
_SCAN_KERNEL_TEMPLATE = r"""
#include <torch/extension.h>
#include "platform.h"
#include "types.h"
#include "utils.cuh"

constexpr int D_INNER = {d_inner};
constexpr int D_STATE = {d_state};

__launch_bounds__({d_inner}, 8)
__global__ void mamba3_scan_jit_kernel(
    const float* __restrict__ x_sorted,
    const float* __restrict__ in_proj_W,
    const float* __restrict__ dt_proj_W,
    const float* __restrict__ dt_proj_b,
    const float* __restrict__ B_proj_W,
    const float* __restrict__ C_proj_W,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ rope_freq,
    float* __restrict__ scan_output,
    float* __restrict__ final_state,
    const float* __restrict__ initial_state,
    const int N,
    const int d_model,
    const int reverse
) {{
    const int tid = threadIdx.x;
    if (tid >= D_INNER) return;

    extern __shared__ float smem[];
    float* s_x_branch = smem;
    float* s_in_proj_W = s_x_branch + D_INNER;
    float* s_dt_proj_W = s_in_proj_W + 2 * D_INNER * d_model;
    float* s_dt_proj_b = s_dt_proj_W + D_INNER * D_INNER;
    float* s_B_proj_W = s_dt_proj_b + D_INNER;
    float* s_C_proj_W = s_B_proj_W + D_STATE * D_INNER;

    for (int i = tid; i < 2 * D_INNER * d_model; i += D_INNER)
        s_in_proj_W[i] = in_proj_W[i];
    for (int i = tid; i < D_INNER * D_INNER; i += D_INNER)
        s_dt_proj_W[i] = dt_proj_W[i];
    if (tid < D_INNER) s_dt_proj_b[tid] = dt_proj_b[tid];
    for (int i = tid; i < D_STATE * D_INNER; i += D_INNER)
        s_B_proj_W[i] = B_proj_W[i];
    for (int i = tid; i < D_STATE * D_INNER; i += D_INNER)
        s_C_proj_W[i] = C_proj_W[i];
    __syncthreads();

    float h[D_STATE];
    if (initial_state != nullptr) {{
        #pragma unroll
        for (int s = 0; s < D_STATE; s++) h[s] = initial_state[tid * D_STATE + s];
    }} else {{
        #pragma unroll
        for (int s = 0; s < D_STATE; s++) h[s] = 0.0f;
    }}

    float A[D_STATE], freq[D_STATE / 2];
    #pragma unroll
    for (int s = 0; s < D_STATE; s++) A[s] = -expf(A_log[tid * D_STATE + s]);
    #pragma unroll
    for (int p = 0; p < D_STATE / 2; p++) freq[p] = rope_freq[tid * (D_STATE / 2) + p];
    float D_val = D_param[tid];

    for (int step = 0; step < N; step++) {{
        const int i = reverse ? (N - 1 - step) : step;

        float x_val = 0.0f, z_val = 0.0f;
        for (int d = 0; d < d_model; d++) {{
            float inp = x_sorted[i * d_model + d];
            x_val += s_in_proj_W[tid * d_model + d] * inp;
            z_val += s_in_proj_W[(tid + D_INNER) * d_model + d] * inp;
        }}

        s_x_branch[tid] = x_val;
        __syncthreads();

        float dt_raw = s_dt_proj_b[tid];
        #pragma unroll
        for (int j = 0; j < D_INNER; j++) {{
            dt_raw += s_dt_proj_W[tid * D_INNER + j] * s_x_branch[j];
        }}
        float dt_val = (dt_raw > 20.0f) ? dt_raw : logf(1.0f + expf(dt_raw));

        #pragma unroll
        for (int s = 0; s < D_STATE; s++) {{
            float A_bar = (1.0f + dt_val * A[s] / 2.0f) / (1.0f - dt_val * A[s] / 2.0f + 1e-8f);
            float B_val = 0.0f;
            #pragma unroll
            for (int j = 0; j < D_INNER; j++) {{
                B_val += s_B_proj_W[s * D_INNER + j] * s_x_branch[j];
            }}
            h[s] = A_bar * h[s] + dt_val * B_val * x_val;
        }}

        float y_val = D_val * x_val;
        #pragma unroll
        for (int s = 0; s < D_STATE; s++) {{
            float C_val = 0.0f;
            #pragma unroll
            for (int j = 0; j < D_INNER; j++) {{
                C_val += s_C_proj_W[s * D_INNER + j] * s_x_branch[j];
            }}
            y_val += C_val * h[s];
        }}

        float z_sig = z_val / (1.0f + expf(-z_val));
        scan_output[i * D_INNER + tid] = y_val * z_sig;

        __syncthreads();
    }}

    #pragma unroll
    for (int s = 0; s < D_STATE; s++)
        final_state[tid * D_STATE + s] = h[s];
}}

void launch_mamba3_scan_jit(
    torch::Tensor x_sorted, torch::Tensor in_proj_W,
    torch::Tensor dt_proj_W, torch::Tensor dt_proj_b,
    torch::Tensor B_proj_W, torch::Tensor C_proj_W,
    torch::Tensor A_log, torch::Tensor D_param, torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int N, int d_model, int reverse
) {{
    const int smem_size =
        D_INNER + 2 * D_INNER * d_model + D_INNER * D_INNER +
        D_INNER + D_STATE * D_INNER + D_STATE * D_INNER;
    mamba3_scan_jit_kernel<<<1, D_INNER, smem_size * sizeof(float)>>>(
        x_sorted.data_ptr<float>(), in_proj_W.data_ptr<float>(),
        dt_proj_W.data_ptr<float>(), dt_proj_b.data_ptr<float>(),
        B_proj_W.data_ptr<float>(), C_proj_W.data_ptr<float>(),
        A_log.data_ptr<float>(), D_param.data_ptr<float>(),
        rope_freq.data_ptr<float>(),
        scan_output.data_ptr<float>(), final_state.data_ptr<float>(),
        initial_state.defined() ? initial_state.data_ptr<float>() : nullptr,
        N, d_model, reverse);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("launch_mamba3_scan_jit", &launch_mamba3_scan_jit);
}}
"""


def _kernel_key(d_inner: int, d_state: int) -> str:
    return f"scan_d{d_inner}_s{d_state}"


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def get_scan_kernel(d_inner: int, d_state: int, verbose: bool = False):
    """Get a JIT-compiled scan kernel specialized for the given dimensions.

    First checks the in-memory cache, then the on-disk cache at
    ~/.cache/supergrok2_jit/. If neither exists, compiles from template.

    Args:
        d_inner: Inner dimension of the Mamba scan.
        d_state: State dimension of the Mamba scan.
        verbose: Print compilation progress.

    Returns:
        Module with launch_mamba3_scan_jit() function.
    """
    key = _kernel_key(d_inner, d_state)

    if key in _loaded_kernels:
        return _loaded_kernels[key]

    if not torch.cuda.is_available():
        raise RuntimeError("JIT kernel compilation requires CUDA")

    source = _SCAN_KERNEL_TEMPLATE.format(d_inner=d_inner, d_state=d_state)
    src_hash = _source_hash(source)
    build_dir = _JIT_CACHE_DIR / f"{key}_{src_hash}"

    # Write source to temp file
    build_dir.mkdir(parents=True, exist_ok=True)
    src_file = build_dir / "scan_jit.cu"
    if not src_file.exists():
        src_file.write_text(source)

    # Get include dirs from the project
    project_root = Path(__file__).parent.parent
    include_dirs = [
        str(project_root / "csrc" / "common"),
        str(project_root / "csrc"),
    ]

    from torch.utils.cpp_extension import load

    module = load(
        name=f"sg2_jit_{key}_{src_hash}",
        sources=[str(src_file)],
        extra_include_paths=include_dirs,
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17",
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build_dir),
        verbose=verbose,
    )

    _loaded_kernels[key] = module
    return module


def precompile_common_configs(verbose: bool = False):
    """Pre-compile kernels for common d_inner/d_state combinations.

    Call this during model initialization to avoid compilation during training.
    """
    configs = [
        (16, 16), (16, 32),  # Default configs
        (8, 16), (8, 8),     # Small configs
        (32, 16), (32, 32),  # Large configs
    ]
    for d_inner, d_state in configs:
        try:
            get_scan_kernel(d_inner, d_state, verbose=verbose)
        except Exception:
            pass  # Skip configs that fail to compile


def clear_cache():
    """Remove all JIT-compiled kernel caches."""
    import shutil
    if _JIT_CACHE_DIR.exists():
        shutil.rmtree(_JIT_CACHE_DIR)
    _loaded_kernels.clear()
