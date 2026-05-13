// Fused (mamba + prodigy) instantiation for gfx942 (HIP / CDNA3).
//
// This TU is the compile-time fusion point that wires the model forward/
// backward kernel with the optimizer launch glue. The actual kernel
// implementations live in:
//   csrc/algorithms/prodigy.h                            — optimizer math spec
//   csrc/models/mamba.h                                 — model contract
//   csrc/backends/hip/gfx942/launch_prodigy.hip.cpp      — optimizer launchers
//   csrc/backends/hip/gfx942/models/mamba.hip.cpp       — model kernels
//   csrc/backends/hip/gfx942/primitives.hpp             — vendor primitives
//
// Note: .hip.cpp files route through the host compiler, NOT hipcc. Any
// __global__ kernel must live in a launch_*.hip.cpp file that uses ATen
// tensor ops instead of <<<...>>> launch syntax.

#include "csrc/models/mamba.h"
#include "csrc/algorithms/prodigy.h"
#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace fused { namespace gfx942 {

// TODO: Instantiate fused forward-backward-update kernel
// combining mamba forward/backward with prodigy per-element step.

}}} // namespace sg::fused::gfx942
