#!/usr/bin/env bash
# Manual build of the TP8 scratch pybind WITH the device-link (-dlink) step torch's
# JIT load() omits for -rdc=true objects. Produces sg_tp8_scratch.so with the RDC
# registration stub resolved + libnvshmem_device.a device symbols pulled.
set -euo pipefail

NVSHMEM_HOME=${NVSHMEM_HOME:-/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem}
# WORKTREE build: distinct build dir (sg_tp8_scratch_wt) so the worktree's edited
# megakernel .so does NOT clobber the main repo's cached sg_tp8_scratch.so, and a
# worktree ROOT so the -I/-include resolve the EDITED csrc/ headers.
BD=${SG_TP8_BD:-${SG_TORCH_EXT_DIR:-/workspace/.torch_ext}/sg_tp8_scratch_wt}
ROOT=${SG_TP8_ROOT:-/workspace/SuperGrok1.5}
SRC=$ROOT/tuning/_tp8_scratch_pybind.cu
NVCC=/usr/local/cuda/bin/nvcc
CXX=c++
TORCH=/usr/local/lib/python3.11/dist-packages/torch
PYINC=/usr/include/python3.11

mkdir -p "$BD"
cd "$BD"

CUDA_CFLAGS="-DTORCH_EXTENSION_NAME=sg_tp8_scratch -DTORCH_API_INCLUDE_EXTENSION_H \
-DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" \
-I$ROOT -isystem $TORCH/include -isystem $TORCH/include/torch/csrc/api/include \
-isystem $TORCH/include/TH -isystem $TORCH/include/THC -isystem /usr/local/cuda/include \
-isystem $PYINC -D_GLIBCXX_USE_CXX11_ABI=0 \
--expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a --compiler-options -fPIC \
-O3 -std=c++17 -DSG_TUNED_GEMM_IMPL=1 -DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1 \
-DSG_DEC_BENCH_LAYOUT=1 \
-include csrc/fused/sm_90/decoder_flagship_layout.cuh -DSG_HAS_NVSHMEM=1 -rdc=true \
-I$NVSHMEM_HOME/include \
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
-U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_BFLOAT162_OPERATORS__"

echo "[build] 1/3 compile (-rdc=true -dc) ..."
# -dc == --relocatable-device-code=true -c ; we pass -rdc=true above + -c here.
cd "$ROOT"   # so -include relative path resolves
$NVCC $CUDA_CFLAGS -c "$SRC" -o "$BD/_tp8_scratch_pybind.cuda.o"

echo "[build] 2/3 device-link (-dlink against libnvshmem_device) ..."
# Use the NAMED -lnvshmem_device (NOT -l:libnvshmem_device.a): nvlink only resolves
# the archive's device members transitively with the named -l form. The -l: exact
# form leaves nvshmemi_transfer_quiet / nvshmemi_device_state_d undefined.
# -Xcompiler -fPIC is REQUIRED: the dlink object carries a __fatbinwrap relocation
# that ld rejects in a shared object unless PIC.
$NVCC -dlink -gencode=arch=compute_90a,code=sm_90a -Xcompiler -fPIC \
    "$BD/_tp8_scratch_pybind.cuda.o" \
    -L"$NVSHMEM_HOME/lib" -lnvshmem_device \
    -L/usr/local/cuda/lib64 -lcudart \
    -o "$BD/_tp8_scratch_dlink.o"

echo "[build] 3/3 host link -> .so ..."
# -lcuda: init_device_state() uses the DRIVER API cuFuncGetModule to recover this
# .so's CUmodule for nvshmemx_cumodule_init (the in-kernel device-state registration).
LDFLAGS="-shared -L$NVSHMEM_HOME/lib -lnvshmem_device -l:libnvshmem_host.so.3 \
-Wl,-rpath,$NVSHMEM_HOME/lib -L$TORCH/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda \
-ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -lcudadevrt \
-L/usr/lib/x86_64-linux-gnu -lcuda"
$CXX "$BD/_tp8_scratch_pybind.cuda.o" "$BD/_tp8_scratch_dlink.o" $LDFLAGS -o "$BD/sg_tp8_scratch.so"

echo "[build] DONE: $BD/sg_tp8_scratch.so"
