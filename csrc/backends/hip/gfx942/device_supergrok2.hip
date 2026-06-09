// csrc/backends/hip/gfx942/device_supergrok2.hip — Stage-7 device TU.
//
// WHY (the core of WS Stage 7 — hipcc device routing): setup.py routes
// `csrc/backends/hip/gfx942/*.hip` through hipcc (device pass) and `*.hip.cpp`
// through the host compiler. The hand-written AMDGCN CSA/HCA attention MFMA,
// peer-route/GRU, MoE, and adam-apply kernels in supergrok2_gfx942.hip.hpp
// (guarded `#if __AMDGCN__||__HIPCC__`) had no `.hip` device includer, so hipcc
// never compiled them. This thin `.hip` TU #include's the canonical header so
// hipcc's DEVICE pass emits the force-instantiated kernels. SG_GFX942_DEVICE_TU
// suppresses re-emitting the host launchers — they stay owned by the
// launch_supergrok2.hip.cpp host TU.
//
// STATUS 🟡: clang-gate-verified (scripts/amdgcn_check.sh); hipcc/MI300X
// deferred (no ROCm in this build env — see HARDWARE_VALIDATION.md).
#define SG_GFX942_DEVICE_TU 1
#include "grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp"
