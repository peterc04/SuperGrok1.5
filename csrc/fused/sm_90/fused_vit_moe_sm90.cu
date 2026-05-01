// Fused (model, optimizer, arch) instantiation: vit + moe on sm_90.
// Compile-time instantiation combining device model and optimizer templates.

#include "csrc/device/models/sm_90/vit_sm90.cuh"
#include "csrc/device/optimizers/sm_90/moe_sm90.cuh"

namespace sg { namespace fused { namespace sm90 {

// TODO: Instantiate fused forward-backward-update kernel
// combining vit_forward/backward with moe_step

}}} // namespace sg::fused::sm90
