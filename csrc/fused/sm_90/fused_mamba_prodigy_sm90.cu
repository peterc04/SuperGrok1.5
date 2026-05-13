// Fused (mamba + prodigy) instantiation for sm_90 (CUDA / Hopper).
//
// This TU is the compile-time fusion point that wires the model forward/
// backward kernel with the optimizer launch glue. The actual kernel
// implementations live in:
//   csrc/algorithms/prodigy.h                       — optimizer math
//   csrc/models/mamba.h                            — model contract
//   csrc/backends/cuda/sm_90/launch_prodigy.cu      — optimizer launchers
//   csrc/backends/cuda/sm_90/models/mamba.cu       — model kernels
//   csrc/backends/cuda/sm_90/primitives.cuh        — vendor primitives
//
// At this stage the fused TU is a placeholder; Phase 8 of the refactor
// only updates includes to the new architecture. The fused megakernel
// instantiation itself will be added when we wire model fwd/bwd into
// the optimizer step body.

#include "csrc/models/mamba.h"
#include "csrc/algorithms/prodigy.h"
#include "csrc/backends/cuda/sm_90/primitives.cuh"

namespace sg { namespace fused { namespace sm90 {

// TODO: Instantiate fused forward-backward-update kernel
// combining mamba forward/backward with prodigy per-element step.

}}} // namespace sg::fused::sm90
