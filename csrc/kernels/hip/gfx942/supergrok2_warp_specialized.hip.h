// =====================================================================
//  csrc/kernels/hip/gfx942/supergrok2_warp_specialized.hip.h
//
//  Placeholder for the warp-specialized scan path used by SuperGrok v2
//  on sm_90 (Hopper TMA + cluster). The gfx942 (CDNA3) port does NOT
//  use this path — there is no TMA equivalent, and inter-block scan is
//  handled by a different LDS-resident strategy (see notes in
//  supergrok2_fwd.hip.h).
//
//  Kept as a TU pair so that:
//    1) the file naming mirrors the sm_90 layout
//    2) any future gfx942 specialization has a home
//    3) setup.py's *.hip.cpp glob picks up the stub TU
// =====================================================================

#pragma once

namespace sg { namespace gfx942 { namespace supergrok2 {

// No public binding entry points are forward-declared here; the
// warp-specialized scan is purely an internal sm_90 helper. This file
// exists only to keep the file pairing consistent.

}}} // namespace sg::gfx942::supergrok2
