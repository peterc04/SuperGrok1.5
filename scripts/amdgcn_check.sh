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
printf '#pragma once\n' > "$STUB/hip_bf16.h"
printf '#pragma once\n' > "$STUB/hip_fp16.h"
# Minimal hip_fp16/bf16 type + cstdint/cmath shims so the pre-existing
# common_gfx942.hip.hpp (which uses real ROCm __half/__hip_bfloat16 types) can
# also be parsed by the free-standing gate. Under real hipcc these come from
# ROCm; here we provide just enough surface for the device code to type-check.
cat >> "$STUB/hip_fp16.h" <<'EOF'
struct __half { unsigned short x; __half()=default; __device__ __half(float f){ x=(unsigned short)(__builtin_bit_cast(unsigned,f)>>16); } __device__ operator float() const { return __builtin_bit_cast(float, (unsigned)x << 16); } };
static inline __attribute__((always_inline)) float __half2float(__half h){ return __builtin_bit_cast(float, (unsigned)h.x << 16); }
static inline __attribute__((always_inline)) __half __float2half(float f){ __half h; h.x = (unsigned short)(__builtin_bit_cast(unsigned, f) >> 16); return h; }
static inline __attribute__((always_inline)) __half __float2half_rn(float f){ return __float2half(f); }
EOF
cat >> "$STUB/hip_bf16.h" <<'EOF'
struct __hip_bfloat16 { unsigned short x; __hip_bfloat16()=default; __device__ __hip_bfloat16(float f){ x=(unsigned short)(__builtin_bit_cast(unsigned,f)>>16); } __device__ operator float() const { return __builtin_bit_cast(float, (unsigned)x << 16); } };
typedef __hip_bfloat16 hip_bfloat16;  // older ROCm spelling used by some headers
static inline __attribute__((always_inline)) float __bfloat162float(__hip_bfloat16 h){ return __builtin_bit_cast(float, (unsigned)h.x << 16); }
static inline __attribute__((always_inline)) __hip_bfloat16 __float2bfloat16(float f){ __hip_bfloat16 h; h.x = (unsigned short)(__builtin_bit_cast(unsigned, f) >> 16); return h; }
EOF
# common_gfx942.hip.hpp includes the longer spelling <hip/hip_bfloat16.h>.
cp "$STUB/hip_bf16.h" "$STUB/hip_bfloat16.h"
# cstdint/cmath are absent on the free-standing target; provide a tiny subset.
mkdir -p "$(dirname "$STUB")/_cxx"
cat > "$(dirname "$STUB")/_cxx/cstdint" <<'EOF'
#pragma once
typedef unsigned char uint8_t; typedef signed char int8_t;
typedef unsigned short uint16_t; typedef short int16_t;
typedef unsigned int uint32_t; typedef int int32_t;
typedef unsigned long long uint64_t; typedef long long int64_t;
EOF
cat > "$(dirname "$STUB")/_cxx/cmath" <<'EOF'
#pragma once
static inline float sqrtf(float x){ return __builtin_sqrtf(x); }
static inline float expf(float x){ return __builtin_expf(x); }
static inline float fabsf(float x){ return __builtin_fabsf(x); }
static inline float fmaxf(float a,float b){ return __builtin_fmaxf(a,b); }
static inline float fminf(float a,float b){ return __builtin_fminf(a,b); }
static inline float tanhf(float x){ return __builtin_tanhf(x); }
static inline float copysignf(float a,float b){ return __builtin_copysignf(a,b); }
static inline float truncf(float x){ return __builtin_truncf(x); }
static inline float floorf(float x){ return __builtin_floorf(x); }
EOF

FLAGS=(--target=amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -std=c++17
       -Wno-unused-function -I"$(dirname "$STUB")" -I"$(dirname "$STUB")/_cxx" -I.
       "-D__hip_atomic_fetch_add(a,b,c,d)=__atomic_fetch_add(a,b,c)"
       "-D__hip_atomic_load(a,b,c)=__atomic_load_n(a,b)"
       -D__HIP_MEMORY_SCOPE_AGENT=0 -D__HIP_MEMORY_SCOPE_WORKGROUP=0)

LOG="$(mktemp)"
if [ "${1:-}" = "--header" ]; then
  HDR="${2:?usage: amdgcn_check.sh --header <header>}"
  clang "${FLAGS[@]}" -c "$HDR" -o /dev/null 2> "$LOG"
elif [ "${1:-}" = "--cell" ]; then
  # Full fused .hip cell: clang infers HIP language from the .hip extension and
  # then errors ("unsupported architecture amdgcn for host compilation"). Force
  # C++ language mode so the amdgcn device pass type-checks the forced template
  # instantiation block (the host hipLaunchKernelGGL block is #if'd out under
  # the bare amdgcn target — __HIPCC__ is undefined here).
  CELL="${2:?usage: amdgcn_check.sh --cell <cell.hip>}"
  clang "${FLAGS[@]}" -x c++ -c "$CELL" -o /dev/null 2> "$LOG"
elif [ "${1:-}" = "--emit-obj" ]; then
  # Emit a REAL gfx942 ELF object (NOT a PCH) for emitted-ISA profiling:
  # disassemble with llvm-objdump for v_mfma / DPP, read VGPR/SGPR/LDS from the
  # kernel descriptor with llvm-readobj. -x c++ forces a translation unit so a
  # .hip.hpp header emits its __global__ kernels as code, not a precompiled
  # header. -O3 so the matrix builtins lower to real v_mfma.
  SRC="${2:?usage: amdgcn_check.sh --emit-obj <src> <out.o>}"
  OUT="${3:?usage: amdgcn_check.sh --emit-obj <src> <out.o>}"
  clang "${FLAGS[@]}" -x c++ -O3 \
    -Wno-pragma-once-outside-header -Wno-unknown-attributes \
    -c "$SRC" -o "$OUT" 2> "$LOG"
else
  SRC="${1:?usage: amdgcn_check.sh <snippet.cpp> | --header <h> | --cell <c.hip>}"
  clang "${FLAGS[@]}" -c "$SRC" -o /dev/null 2> "$LOG"
fi
RC=$?
if [ "$RC" -ne 0 ]; then echo "AMDGCN_FAIL rc=$RC ${2:-$1}"; head -40 "$LOG";
else echo "AMDGCN_OK ${2:-$1}"; fi
rm -f "$LOG"; exit "$RC"
