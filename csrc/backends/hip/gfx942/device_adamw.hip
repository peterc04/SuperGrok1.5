// csrc/backends/hip/gfx942/device_adamw.hip — Stage-7 device translation unit.
//
// WHY THIS FILE EXISTS (the core of WS Stage 7 — hipcc device routing):
//   setup.py routes `csrc/backends/hip/gfx942/*.hip.cpp` through the HOST
//   compiler (ATen + rocBLAS path). It routes `csrc/backends/hip/gfx942/*.hip`
//   through hipcc (device pass). Before this TU there were ZERO `.hip` device
//   TUs in this directory, so the hand-written AMDGCN grid-stride kernel in
//   adamw_gfx942.hip.hpp — guarded `#if defined(__AMDGCN__)||defined(__HIPCC__)`
//   — was NEVER reached by hipcc: the only includer (launch_adamw.hip.cpp) is a
//   `.hip.cpp` host TU. This thin `.hip` TU #include's the canonical header so
//   hipcc's DEVICE pass actually compiles + emits the kernel (the header already
//   force-instantiates adamw_gfx942_kernel<float,float> under the device gate).
//
//   SG_GFX942_DEVICE_TU is defined BEFORE the include so the header's section
//   (A) host launcher (launch_adamw_step) is NOT re-emitted in this TU's host
//   pass — it stays owned by launch_adamw.hip.cpp, avoiding a duplicate symbol.
//
// STATUS 🟡: clang-gate-verified (scripts/amdgcn_check.sh), hipcc/MI300X-
//   deferred. There is no hipcc/ROCm in this build environment, so the device
//   compile + link is hardware-gated (see HARDWARE_VALIDATION.md). The routing
//   here is what makes a real ROCm build compile the device path.
#define SG_GFX942_DEVICE_TU 1
#include "grokking_optimizers/kernels/gfx942/adamw_gfx942.hip.hpp"
