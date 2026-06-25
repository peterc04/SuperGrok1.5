#include <cstdio>
#include <cstdint>
#define SG_TUNED_GEMM_IMPL 1
#include "csrc/fused/sm_90/decoder_flagship_layout.cuh"
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"
using namespace sg::fused::sm90;
int main(){
  int B=512, T=B*dec::kSeq, nCTA=64;
  printf("kLnVecElems = %d (0x%x)\n", dectc::kLnVecElems, dectc::kLnVecElems);
  printf("scratch_per = %ld (0x%lx)\n", (long)dectc::dec_tile_scratch_total_f32(), (long)dectc::dec_tile_scratch_total_f32());
  printf("acts_floats = %ld\n", (long)dec_tc_acts_floats(T,B));
  printf("workspace   = %ld floats = %.2f GB\n", (long)dec_tc_workspace_floats(T,B,nCTA), dec_tc_workspace_floats(T,B,nCTA)*4.0/1e9);
  printf("dw_part     = %ld\n", (long)dec_tc_dw_part_floats());
  printf("dw_transpose= %ld\n", (long)dec_tc_dw_transpose_floats(B,T));
  printf("kDecTotalElems=%ld\n", (long)kDecTotalElems);
  printf("0x4bc80 = %d\n", 0x4bc80);
  printf("kAttnPerTile=%d kLogitsPerTile=%d\n", dectc::kAttnPerTile, dectc::kLogitsPerTile);
  return 0;
}
