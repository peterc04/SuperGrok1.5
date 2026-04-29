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

// Returns one of: 80, 90, 100, 942. Throws std::runtime_error on
// any other detected arch (e.g. sm_70/75/86/89, gfx908/90a/950, RDNA).
//
// Honors FORCE_ARCH env var for cross-arch testing on a host that has
// the target binding compiled in.
int detect_arch();

// Convenience predicates used by dispatch switches.
inline bool is_cuda_arch(int a) { return a == 80 || a == 90 || a == 100; }
inline bool is_hip_arch(int a)  { return a == 942; }

// ---------------------------------------------------------------------
// Per-arch launcher declarations
//
// The full launcher set is too large to forward-declare here; each per-
// optimizer dispatcher (csrc/bindings/<optimizer>.cpp) declares the
// launchers it needs from the appropriate namespace before calling them.
// This header just brings in the namespaces.
// ---------------------------------------------------------------------

namespace sm80   {}
namespace sm90   {}
namespace sm100  {}
namespace gfx942 {}

} // namespace sg
