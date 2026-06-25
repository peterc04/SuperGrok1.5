#define SG_TUNED_GEMM_IMPL 1
#define SG_MB_SCALAR_MEGAKERNEL 0
#define SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_ 1
#include "csrc/fused/sm_90/mamba_flagship_layout.cuh"
#include "csrc/fused/sm_90/model_stage_mamba3.cuh"
#include <cstdio>
int main(){
  using namespace sg::fused::sm90;
  printf("kMbStreamSmem=%d\n", (int)kMbStreamSmem);
  printf("sizeof(MambaSampleSmem)=%zu bytes = %.2f KB\n", sizeof(MambaSampleSmem), sizeof(MambaSampleSmem)/1024.0);
  printf("kMbStreamSmemBytes(formula)=%lld = %.2f KB\n", (long long)kMbStreamSmemBytes, kMbStreamSmemBytes/1024.0);
  printf("kMambaSmemBytes(all-layers)=%lld = %.2f KB\n", (long long)kMambaSmemBytes, kMambaSmemBytes/1024.0);
  printf("227KB cap=%d ; fits=%d\n", 227*1024, (int)(sizeof(MambaSampleSmem)<=227*1024));
  printf("kMbLayerActSmallFloatsExact=%lld ; offsetof(LayerAct,x_in)/4=%zu\n",
         (long long)kMbLayerActSmallFloatsExact, offsetof(MambaSampleSmem::LayerAct,x_in)/4);
  return 0;
}
