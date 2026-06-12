#!/usr/bin/env bash
# compile_one.sh <src-root> <tu-rel-path> <out-tag> [extra flags...]
# Reproduces the PRODUCTION _ops nvcc device flag set for one TU and captures
# ptxas -v / --resource-usage to .regpressure/logs/<out-tag>.log (CPU-only).
#
# Production flag provenance (HEAD 642e360):
#   compile.py NVCC_DEVICE_BASE (lines 9334-9402)
#   + setup.py grokking project flags: --use_fast_math -DNDEBUG -lineinfo
#   + gencode sm_90a (PTX-fallback compute_90 omitted: no ptxas run for it,
#     so it cannot change the sm_90a resource numbers)
#   + -DWITH_CUDA define (cuda_define_macros) [already in base]
#   + include dirs: ROOT, ROOT/csrc/bindings, ROOT/csrc  (+ torch includes for
#     the JIT bench TUs that include <torch/extension.h>)
set -uo pipefail
ROOT="$1"; TU="$2"; TAG="$3"; shift 3
OUT=/workspace/SuperGrok1.5/.regpressure/logs
mkdir -p "$OUT" /dev/shm/tmp/objs
TORCH_INC1=/usr/local/lib/python3.11/dist-packages/torch/include
TORCH_INC2=/usr/local/lib/python3.11/dist-packages/torch/include/torch/csrc/api/include
PYTHON_INC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
nvcc -c "$ROOT/$TU" -o "/dev/shm/tmp/objs/$TAG.o" \
  -O3 -std=c++17 -DWITH_CUDA \
  --expt-relaxed-constexpr \
  -Xfatbin -compress-all \
  -Xptxas --opt-level=3 \
  -Xptxas -v -Xptxas --warn-on-spills \
  --extra-device-vectorization \
  -Xcompiler -fPIC -Xcompiler -fno-strict-aliasing \
  --resource-usage \
  --default-stream per-thread \
  --use_fast_math -DNDEBUG -lineinfo \
  -gencode=arch=compute_90a,code=sm_90a \
  -I"$ROOT" -I"$ROOT/csrc/bindings" -I"$ROOT/csrc" \
  -I"$TORCH_INC1" -I"$TORCH_INC2" -I"$PYTHON_INC" \
  "$@" > "$OUT/$TAG.log" 2>&1
RC=$?
echo "rc=$RC" >> "$OUT/$TAG.log"
exit $RC
