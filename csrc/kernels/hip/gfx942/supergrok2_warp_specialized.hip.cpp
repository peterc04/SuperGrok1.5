// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok2_warp_specialized.hip.cpp
//
//  See supergrok2_warp_specialized.hip.h. This TU is intentionally
//  empty — no internal SG2 helper is currently routed through a warp-
//  specialized path on gfx942. Kept to mirror sm_90 file pairing and
//  as a hook for a future MFMA-based scan implementation.
// =====================================================================

#include "csrc/kernels/hip/gfx942/supergrok2_warp_specialized.hip.h"

namespace sg { namespace gfx942 { namespace supergrok2 {

// No symbols. The sm_90 warp_specialized scan is internal to that
// arch and has no gfx942 callers.

}}} // namespace sg::gfx942::supergrok2
