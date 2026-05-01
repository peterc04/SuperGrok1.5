// Fused (model, optimizer, arch) instantiation: transformer + supergrok2 on gfx942.

#include "csrc/device/models/gfx942/transformer_gfx942.hip.cuh"
#include "csrc/device/optimizers/gfx942/supergrok2_gfx942.hip.cuh"

namespace sg { namespace fused { namespace gfx942 {

// TODO: Instantiate fused forward-backward-update kernel

}}} // namespace sg::fused::gfx942
