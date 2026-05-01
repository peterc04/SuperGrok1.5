// Fused (model, optimizer, arch) instantiation: transformer + looksam on sm_90.
// Compile-time instantiation combining device model and optimizer templates.

#include "csrc/device/models/sm_90/transformer_sm90.cuh"
#include "csrc/device/optimizers/sm_90/looksam_sm90.cuh"

namespace sg { namespace fused { namespace sm90 {

// TODO: Instantiate fused forward-backward-update kernel
// combining transformer_forward/backward with looksam_step

}}} // namespace sg::fused::sm90
