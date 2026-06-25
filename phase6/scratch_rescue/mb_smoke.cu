#define SG_TUNED_GEMM_IMPL 1
#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"
// Force-instantiate the SMALL (d=128) launcher path to type-check the edits.
namespace sg { namespace fused { namespace sm90 {
template int mb_tc_launched_nctas<OptId::AdamW>(int, int);
}}}
int main(){ return (int)sizeof(::sg::fused::sm90::MambaSampleSmem); }
