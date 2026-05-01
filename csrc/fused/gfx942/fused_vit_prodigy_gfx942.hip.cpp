// Fused (model, optimizer, arch) instantiation: vit + prodigy on gfx942.

#include "csrc/device/models/gfx942/vit_gfx942.hip.cuh"
#include "csrc/device/optimizers/gfx942/prodigy_gfx942.hip.cuh"

namespace sg { namespace fused { namespace gfx942 {

// TODO: Instantiate fused forward-backward-update kernel

}}} // namespace sg::fused::gfx942
