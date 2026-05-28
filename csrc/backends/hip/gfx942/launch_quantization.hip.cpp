// HIP gfx942 quantization kernel stubs.
// These resolve the symbols declared in csrc/bindings/quantization.cpp.
// Real implementations require FP8/INT8/INT4 kernels — see Tier 5 in the
// optimization roadmap.

#include <torch/extension.h>
#include <stdexcept>

namespace sg { namespace gfx942 {

void launch_fp8_e4m3_quantize(
    torch::Tensor input, torch::Tensor q_out, torch::Tensor scale
) {
    throw std::runtime_error(
        "fp8_e4m3_quantize: gfx942 kernel not yet implemented. "
        "See optimization roadmap Tier 5.");
}

void launch_int8_symmetric_quantize(
    torch::Tensor input, torch::Tensor q_out, torch::Tensor scale
) {
    throw std::runtime_error(
        "int8_symmetric_quantize: gfx942 kernel not yet implemented. "
        "See optimization roadmap Tier 5.");
}

void launch_int4_gptq_quantize(
    torch::Tensor input, torch::Tensor packed,
    torch::Tensor scales, torch::Tensor zeros, int group_size
) {
    throw std::runtime_error(
        "int4_gptq_quantize: gfx942 kernel not yet implemented. "
        "See optimization roadmap Tier 5.");
}

}} // namespace sg::gfx942
