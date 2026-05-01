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
//  is not one of {sm_90, gfx942}, dispatch raises.
//  3-arch active set: sm_90, gfx942, tpu_v5p.
// =====================================================================

#pragma once

#include <torch/extension.h>

namespace sg {

// ---------------------------------------------------------------------
// Arch detection
// ---------------------------------------------------------------------

// Returns one of the supported arches: 90, 942.
// Throws std::runtime_error on anything else.
// 3-arch active set: sm_90, gfx942, tpu_v5p (TPU handled in Python).
//
// Honors FORCE_ARCH env var for cross-arch testing on a host that has
// the target binding compiled in.
int detect_arch();

// Convenience predicates used by dispatch switches.
inline bool is_cuda_arch(int a) { return a == 90; }
inline bool is_hip_arch(int a)  { return a == 942; }

// ---------------------------------------------------------------------
// Fused (model, optimizer, arch) dispatch — thin wrapper around the 99
// fused TUs (3 models x 11 optimizers x 3 arches).
// ---------------------------------------------------------------------

void fused_step(const std::string& model, const std::string& optimizer,
                torch::Tensor params, torch::Tensor input,
                torch::Tensor grad, torch::Tensor state, float lr);

// ---------------------------------------------------------------------
// Per-arch launcher declarations
//
// The full launcher set is too large to forward-declare here; each per-
// optimizer dispatcher (csrc/bindings/<optimizer>.cpp) declares the
// launchers it needs from the appropriate namespace before calling them.
// This header just brings in the namespaces.
// ---------------------------------------------------------------------

namespace sm90   {}
namespace gfx942 {}

} // namespace sg
