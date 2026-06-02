// csrc/backends/hip/gfx942/device_muon.hip — Stage-7 device translation unit.
//
// WHY (the core of WS Stage 7 — hipcc device routing): setup.py routes
// `csrc/backends/hip/gfx942/*.hip` through hipcc (device pass) and `*.hip.cpp`
// through the host compiler. The hand-written AMDGCN Frobenius-reduce /
// Newton-Schulz MFMA / apply kernels in muon_gfx942.hip.hpp (guarded
// `#if __AMDGCN__||__HIPCC__`) had no `.hip` device includer, so hipcc never
// compiled them. This thin `.hip` TU #include's the canonical header so hipcc's
// DEVICE pass emits the kernels. SG_GFX942_DEVICE_TU suppresses re-emitting the
// host launchers here — they stay owned by launch_muon.hip.cpp.
//
// STATUS 🟡: clang-gate-verified (scripts/amdgcn_check.sh); hipcc/MI300X
// deferred (no ROCm in this build env — see HARDWARE_VALIDATION.md).
#define SG_GFX942_DEVICE_TU 1
#include "grokking_optimizers/kernels/gfx942/muon_gfx942.hip.hpp"
