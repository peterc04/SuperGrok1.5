#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
TU="${1:?usage: compile_to_object.sh <tu.cu> [extra flags]}"; shift || true
EXTRA=("$@")
TORCH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
PYINC="$(python -c "import sysconfig;print(sysconfig.get_path('include'))" 2>/dev/null)"
LOG="$(mktemp)"
nvcc -c -std=c++17 -DWITH_CUDA -gencode arch=compute_90a,code=sm_90a \
  -I. -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC" \
  -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include \
  --expt-relaxed-constexpr --expt-extended-lambda "${EXTRA[@]}" "$TU" -o /dev/null 2> "$LOG"
RC=$?
if [ "$RC" -ne 0 ]; then echo "COMPILE_FAIL rc=$RC tu=$TU"; head -60 "$LOG"; else echo "COMPILE_OK tu=$TU"; fi
rm -f "$LOG"; exit "$RC"
