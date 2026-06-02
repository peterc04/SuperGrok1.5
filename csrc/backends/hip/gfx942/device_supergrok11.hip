// csrc/backends/hip/gfx942/device_supergrok11.hip — Stage-7 device translation unit.
//
// WHY (the core of WS Stage 7 — hipcc device routing): setup.py routes
// `csrc/backends/hip/gfx942/*.hip` through hipcc (device pass) and `*.hip.cpp`
// through the host compiler. The hand-written AMDGCN reduction/apply kernels in
// supergrok11_gfx942.hip.hpp (guarded `#if __AMDGCN__||__HIPCC__`) had no `.hip`
// device includer, so hipcc never compiled them. This thin `.hip` TU #include's
// the canonical header so hipcc's DEVICE pass emits the kernels (the header
// force-instantiates them under the device gate). SG_GFX942_DEVICE_TU suppresses
// re-emitting the host launchers here — they stay owned by launch_supergrok11.hip.cpp.
//
// STATUS 🟡: clang-gate-verified (scripts/amdgcn_check.sh); hipcc/MI300X
// deferred (no ROCm in this build env — see HARDWARE_VALIDATION.md). The
// routing here is what makes a real ROCm build compile the device path.
#define SG_GFX942_DEVICE_TU 1
#include "grokking_optimizers/kernels/gfx942/supergrok11_gfx942.hip.hpp"
