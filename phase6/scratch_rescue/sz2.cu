#include <cstdio>
#include <cstdint>
#define SG_TUNED_GEMM_IMPL 1
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"
using namespace sg::fused::sm90;
int main(){
  int B=512, T=B*dec::kSeq, nCTA=64;
  printf("dec::kD=%d kDff=%d kHeads=%d kLayers=%d kSeq=%d kVocab=%d\n",dec::kD,dec::kDff,dec::kHeads,dec::kLayers,dec::kSeq,dec::kVocab);
  printf("kLnVecElems = %d (0x%x)\n", dectc::kLnVecElems, dectc::kLnVecElems);
  printf("scratch_per = %ld (0x%lx)\n", (long)dectc::dec_tile_scratch_total_f32(), (long)dectc::dec_tile_scratch_total_f32());
  printf("acts_floats = %ld\n", (long)dec_tc_acts_floats(T,B));
  printf("workspace   = %ld floats = %.2f GB\n", (long)dec_tc_workspace_floats(T,B,nCTA), dec_tc_workspace_floats(T,B,nCTA)*4.0/1e9);
  printf("0x4bc80 = %d ; kLnVecElems? %s\n", 0x4bc80, (dectc::kLnVecElems==0x4bc80)?"YES":"no");
  return 0;
}
