// =====================================================================
//  bindings.h — declarations for per-arch launchers
//
//  Each per-arch translation unit lives in
//  csrc/kernels/{cuda/<sm>,hip/gfx942}/<optimizer>_<arch>.cu and is wrapped in
//  'namespace sg::<arch> { ... }'. This header forward-declares the launcher
//  signatures inside each arch namespace so per-optimizer dispatchers in
//  csrc/bindings/<optimizer>.cpp can pick the right one at runtime.
//
//  Arch detection is centralized in csrc/bindings/dispatch.cpp; if the host
//  is not one of {sm_80, sm_90, sm_100, gfx942}, dispatch raises.
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg {

// ---------------------------------------------------------------------
// Arch detection
// ---------------------------------------------------------------------

// Returns one of the supported arches: 80, 89, 90, 100, 103, 120, 942, 950.
// Throws std::runtime_error on anything else (sm_70/75/86, gfx908/gfx90a, RDNA).
//
// Honors FORCE_ARCH env var for cross-arch testing on a host that has
// the target binding compiled in.
int detect_arch();

// Convenience predicates used by dispatch switches.
inline bool is_cuda_arch(int a) {
    return a == 80 || a == 89 || a == 90 || a == 100 || a == 103 || a == 120;
}
inline bool is_hip_arch(int a)  { return a == 942 || a == 950; }

// ---------------------------------------------------------------------
// Per-arch launcher declarations
//
// The full launcher set is too large to forward-declare here; each per-
// optimizer dispatcher (csrc/bindings/<optimizer>.cpp) declares the
// launchers it needs from the appropriate namespace before calling them.
// This header just brings in the namespaces.
// ---------------------------------------------------------------------

namespace sm80   {}
namespace sm89   {}
namespace sm90   {}
namespace sm100  {}
namespace sm103  {}
namespace sm120  {}
namespace gfx942 {}
namespace gfx950 {}

} // namespace sg
