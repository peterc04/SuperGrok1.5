#!/usr/bin/env bash
# Stage-5 AMDGCN device-code compile gate (no hipcc / no ROCm device-libs).
#
# clang 18's AMDGPU backend compiles free-standing gfx942 device code, which
# catches real builtin-signature / DPP-control / MFMA-register-type bugs that
# structural review misses. Caught during Stage 5 bring-up:
#   * the bf16 MFMA `_1k` builtins take bf16x4 (short4), NOT u32x4 (the former
#     in-repo reference was wrong and would not have compiled on hipcc);
#   * cvt_f32_fp8 / mov_dpp / ds_swizzle / sched_group_barrier byte-select /
#     dpp-ctrl / mask args must be COMPILE-TIME CONSTANTS.
#
# It gates the DEVICE intrinsics, not the ATen/torch host glue (the `.hip.cpp`
# thin TUs go through the host compiler; full link is hardware-gated on MI300X).
#
# Usage:
#   scripts/amdgcn_check.sh --header <device_header.hip.hpp>   # compile a header
#   scripts/amdgcn_check.sh <free_standing_snippet.cpp>        # compile a snippet
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

# Minimal HIP stubs so device headers that #include <hip/...> resolve under the
# free-standing amdgcn target (these only provide __device__/__forceinline__
# and uint32_t; the real types come from ROCm on a hipcc build).
STUB="$(mktemp -d)/hip"; mkdir -p "$STUB"
cat > "$STUB/hip_runtime.h" <<'EOF'
#pragma once
typedef unsigned int uint32_t;
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
EOF
: > "$STUB/hip_bf16.h"; : > "$STUB/hip_fp16.h"
printf '#pragma once\n' > "$STUB/hip_bf16.h"
printf '#pragma once\n' > "$STUB/hip_fp16.h"

FLAGS=(--target=amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -std=c++17
       -Wno-unused-function -I"$(dirname "$STUB")" -I.
       "-D__hip_atomic_fetch_add(a,b,c,d)=__atomic_fetch_add(a,b,c)"
       "-D__hip_atomic_load(a,b,c)=__atomic_load_n(a,b)"
       -D__HIP_MEMORY_SCOPE_AGENT=0 -D__HIP_MEMORY_SCOPE_WORKGROUP=0)

LOG="$(mktemp)"
if [ "${1:-}" = "--header" ]; then
  HDR="${2:?usage: amdgcn_check.sh --header <header>}"
  clang "${FLAGS[@]}" -c "$HDR" -o /dev/null 2> "$LOG"
else
  SRC="${1:?usage: amdgcn_check.sh <snippet.cpp> | --header <header>}"
  clang "${FLAGS[@]}" -c "$SRC" -o /dev/null 2> "$LOG"
fi
RC=$?
if [ "$RC" -ne 0 ]; then echo "AMDGCN_FAIL rc=$RC ${2:-$1}"; head -40 "$LOG";
else echo "AMDGCN_OK ${2:-$1}"; fi
rm -f "$LOG"; exit "$RC"
