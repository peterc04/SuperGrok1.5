#define SG_TUNED_GEMM_IMPL 1
#define SG_MB_SCALAR_MEGAKERNEL 0
#define SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_ 1
#include "csrc/fused/sm_90/mamba_flagship_layout.cuh"
#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"
namespace sg { namespace fused { namespace sm90 {
template int mb_tc_launched_nctas<OptId::AdamW>(int, int);
}}}
int main(){ return (int)sizeof(::sg::fused::sm90::MambaSampleSmem); }
