// Fused (vit + muon) instantiation for gfx942 (HIP / CDNA3).
//
// This TU is the compile-time fusion point that wires the model forward/
// backward kernel with the optimizer launch glue. The actual kernel
// implementations live in:
//   csrc/algorithms/muon.h                            — optimizer math spec
//   csrc/models/vit.h                                 — model contract
//   csrc/backends/hip/gfx942/launch_muon.hip.cpp      — optimizer launchers
//   csrc/backends/hip/gfx942/models/vit.hip.cpp       — model kernels
//   csrc/backends/hip/gfx942/primitives.hpp             — vendor primitives
//
// Note: .hip.cpp files route through the host compiler, NOT hipcc. Any
// __global__ kernel must live in a launch_*.hip.cpp file that uses ATen
// tensor ops instead of <<<...>>> launch syntax.

#include "csrc/models/vit.h"
#include "csrc/algorithms/muon.h"
#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace fused { namespace gfx942 {

// TODO: Instantiate fused forward-backward-update kernel
// combining vit forward/backward with muon per-element step.

}}} // namespace sg::fused::gfx942
