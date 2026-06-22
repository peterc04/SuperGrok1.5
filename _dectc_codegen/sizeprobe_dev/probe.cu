#define SG_TUNED_GEMM_IMPL 1
#define SG_TUNED_TILE_M 256
#define SG_TUNED_TILE_N 128
#define SG_TUNED_DEC_GEMM_INTERLEAVE 4
#define SG_TUNED_DEC_FWD_PIPE 1
#define SG_TUNED_DEC_FWD_STAGES 4
#include <cstdint>
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"
using ::sg::fused::sm90::DecTcSmem;
static_assert(sizeof(DecTcSmem) == 50832, "size mismatch");
