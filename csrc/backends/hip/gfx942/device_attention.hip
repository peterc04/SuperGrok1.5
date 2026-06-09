// csrc/backends/hip/gfx942/device_attention.hip — Stage-7 device translation unit.
//
// WHY (the core of WS Stage 7 — hipcc device routing): setup.py routes
// `csrc/backends/hip/gfx942/*.hip` through hipcc (device pass) and `*.hip.cpp`
// through the host compiler. The hand-written AMDGCN LDS + MFMA attention kernels in
// attention_gfx942.hip.hpp (guarded `#if __AMDGCN__||__HIPCC__`) were only #included
// by a `.hip.cpp` HOST TU (models/*.hip.cpp), so hipcc's device pass never
// compiled them. This thin `.hip` TU #include's the canonical header so the
// DEVICE pass emits the force-instantiated __global__ kernels. SG_GFX942_DEVICE_TU
// suppresses the host model orchestration (it stays owned by models/*.hip.cpp).
//
// STATUS 🟡: clang-gate-verified (scripts/amdgcn_check.sh); hipcc/MI300X
// deferred (no ROCm in this build env — see HARDWARE_VALIDATION.md).
#define SG_GFX942_DEVICE_TU 1
#include "grokking_optimizers/kernels/gfx942/attention_gfx942.hip.hpp"
