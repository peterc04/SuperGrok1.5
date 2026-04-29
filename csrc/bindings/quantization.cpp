// bindings/quantization.cpp — runtime dispatch to per-arch quantization launchers.
//
// Supports FP8 E4M3, INT8 symmetric, INT4 GPTQ-style, MXFP4 microscaling.
// Per-arch launchers live in csrc/kernels/{cuda/<sm>,hip/gfx942}/quantization_<arch>.<ext>
// once csrc/quantization/quantization_kernels.cu is split. Until that
// split, the bindings call into the shared csrc/quantization/ source.

#include "_dispatch_macro.h"

namespace sg {

#define DECLARE_Q(NS)                                                         \
    namespace NS {                                                            \
        void launch_fp8_e4m3_quantize(                                        \
            torch::Tensor input, torch::Tensor q_out, torch::Tensor scale);   \
        void launch_int8_symmetric_quantize(                                  \
            torch::Tensor input, torch::Tensor q_out, torch::Tensor scale);   \
        void launch_int4_gptq_quantize(                                       \
            torch::Tensor input, torch::Tensor packed,                        \
            torch::Tensor scales, torch::Tensor zeros, int group_size);       \
        void launch_mxfp4_quantize(                                           \
            torch::Tensor input, torch::Tensor packed,                        \
            torch::Tensor shared_exps, int block_size);                       \
    }

DECLARE_Q(sm80) DECLARE_Q(sm90) DECLARE_Q(sm100) DECLARE_Q(gfx942)
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

void mxfp4_quantize(
    torch::Tensor input, torch::Tensor packed,
    torch::Tensor shared_exps, int block_size)
{
    SG_DISPATCH(launch_mxfp4_quantize, input, packed, shared_exps, block_size);
}

} // namespace sg
