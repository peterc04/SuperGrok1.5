#define SG_TUNED_GEMM_IMPL 1
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"
#include <cstdio>
int main(){
  using namespace sg::fused::sm90;
  printf("kMbStreamSmem=%d\n", (int)kMbStreamSmem);
  printf("sizeof(MambaSampleSmem)=%zu ; kMambaSmemBytes=%lld ; equal=%d\n",
         sizeof(MambaSampleSmem), (long long)kMambaSmemBytes,
         (int)(sizeof(MambaSampleSmem)==(size_t)kMambaSmemBytes));
  return 0;
}
