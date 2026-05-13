// bindings/quantization.cpp — runtime dispatch to per-arch quantization launchers.
//
// Supports FP8 E4M3, INT8 symmetric, INT4 GPTQ-style. The MXFP4/NVFP4
// formats were removed in the post-refactor cleanup (3-arch active set
// has no Blackwell / CDNA4 hardware).

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_Q(NS) \
 namespace NS { \
 void launch_fp8_e4m3_quantize( \
 torch::Tensor input, torch::Tensor q_out, torch::Tensor scale); \
 void launch_int8_symmetric_quantize( \
 torch::Tensor input, torch::Tensor q_out, torch::Tensor scale); \
 void launch_int4_gptq_quantize( \
 torch::Tensor input, torch::Tensor packed, \
 torch::Tensor scales, torch::Tensor zeros, int group_size); \
 }

 DECLARE_Q(sm90) DECLARE_Q(gfx942)

#undef DECLARE_Q

void fp8_e4m3_quantize(torch::Tensor input, torch::Tensor q_out, torch::Tensor scale) {
 SG_DISPATCH(launch_fp8_e4m3_quantize, input, q_out, scale);
}

void int8_symmetric_quantize(torch::Tensor input, torch::Tensor q_out, torch::Tensor scale) {
 SG_DISPATCH(launch_int8_symmetric_quantize, input, q_out, scale);
}

void int4_gptq_quantize(
 torch::Tensor input, torch::Tensor packed,
 torch::Tensor scales, torch::Tensor zeros, int group_size)
{
 SG_DISPATCH(launch_int4_gptq_quantize, input, packed, scales, zeros, group_size);
}

} // namespace sg
