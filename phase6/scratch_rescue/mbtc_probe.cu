#define SG_TUNED_GEMM_IMPL 1
#include "csrc/fused/sm_90/model_stage_mamba_tc.cuh"
// Force-instantiate the TP templates so the bodies are type-checked.
namespace sg { namespace fused { namespace sm90 {
using P8 = ::sg::fused::par::ParConfig<8,8,1,1,::sg::fused::par::ZeROStage::Z3>;
__global__ void _probe(MambaWeights w, MambaGrad g, int* tok, int tgt, MambaSampleSmem* sm,
                       ::sg::fused::sm90::tp::LoopbackTransport tr, ::sg::fused::GridBarrier bar) {
    (void)mb_forward_sample_tp<P8>(w, tok, tgt, true, sm, tr, bar, 0, 0);
    mb_backward_sample_tp<P8>(w, g, tok, tgt, 16, true, sm, tr, bar, 0, 0);
    (void)mb_forward_sample_tp<::sg::fused::par::SingleGPU>(w, tok, tgt, true, sm, tr, bar, 0, 0);
}
}}}
