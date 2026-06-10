#ifndef SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
#define SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/mamba3_layout.cuh — weight-layout mirror for the L3-REAL
// Mamba megakernel (PHASE 2).
//
// HAND-WRITTEN (not generated): megakernel_codegen.py is OFF-LIMITS for this
// phase per the owner directive, so this header is hand-derived from the SINGLE
// SOURCE OF TRUTH tests/hw/mamba_oracle.py::mamba_param_layout(), which is
// asserted == the eager model's named_parameters() ORDER + count + total in the
// parity test (tests/hw/test_mamba_megakernel.py). When megakernel_codegen.py is
// later extended with a --mamba-layout emitter (the decoder pattern), this file
// becomes its generated output; until then the numbers below are pinned by the
// static_asserts and the Python parity test. The flat blob is
// torch.cat([p.reshape(-1) for _, p in model.named_parameters()]); the kernel
// addresses tensor i at params + kMambaOffsets[i] for kMambaSizes[i] elements.
//
// CRITICAL ORDERING NOTE: PyTorch yields a module's OWN nn.Parameters before its
// submodules, so within each SelectiveSSMLayer **A_log and D come BEFORE in_proj**
// (NOT the __init__ visual order). The order below matches the verified dump.
//
// A count/total mismatch fails the BUILD loudly (the static_asserts below), never
// corrupts at dispatch.
// ============================================================================

#include <cstdint>

namespace sg { namespace fused { namespace sm90 {

constexpr int SG_MB_VOCAB  = 99;    // ntok (tok embedding rows)
constexpr int SG_MB_PHEAD  = 97;    // p (head width = out Linear cols)
constexpr int SG_MB_D      = 128;   // d_model
constexpr int SG_MB_LAYERS = 2;     // nl
constexpr int SG_MB_SEQ    = 8;     // seq_len
constexpr int SG_MB_DINNER = 256;   // d_inner (= d * expand_factor=2)
constexpr int SG_MB_STATE  = 16;    // state_dim
constexpr int SG_MB_DTRANK = 8;     // dt_rank = max(d/16,1)
constexpr int SG_MB_CONVK  = 3;     // conv1d kernel size

constexpr int     kMambaNumTensors = 28;
constexpr int64_t kMambaTotalElems = 259425;

// Per-tensor element offsets into the flat param blob, named_parameters() order.
__device__ __constant__ int kMambaOffsets[kMambaNumTensors] = {
    0, 12672, 13696, 17792, 18048, 83584, 84352, 84608, 94848, 96896,
    97152, 129920, 130048, 130176, 134272, 134528, 200064, 200832, 201088, 211328,
    213376, 213632, 246400, 246528, 246656, 246784, 246912, 259328
};

// Per-tensor element sizes (numel), same order.
__device__ __constant__ int kMambaSizes[kMambaNumTensors] = {
    12672, 1024, 4096, 256, 65536, 768, 256, 10240, 2048, 256,
    32768, 128, 128, 4096, 256, 65536, 768, 256, 10240, 2048,
    256, 32768, 128, 128, 128, 128, 12416, 97
};

static_assert(kMambaNumTensors == 28,
              "mamba3_layout: tensor count drifted from the oracle layout (28).");
static_assert(kMambaTotalElems == 259425,
              "mamba3_layout: total param count drifted (259425).");

// Host-constexpr mirrors so a sum/offset cross-check folds at compile time (a
// __constant__ array can't be folded in a constexpr).
namespace mamba_layout_check {
constexpr int kSizes[kMambaNumTensors] = {
    12672, 1024, 4096, 256, 65536, 768, 256, 10240, 2048, 256,
    32768, 128, 128, 4096, 256, 65536, 768, 256, 10240, 2048,
    256, 32768, 128, 128, 128, 128, 12416, 97
};
constexpr int kOffsets[kMambaNumTensors] = {
    0, 12672, 13696, 17792, 18048, 83584, 84352, 84608, 94848, 96896,
    97152, 129920, 130048, 130176, 134272, 134528, 200064, 200832, 201088, 211328,
    213376, 213632, 246400, 246528, 246656, 246784, 246912, 259328
};
constexpr int64_t sum_sizes() {
    int64_t s = 0;
    for (int i = 0; i < kMambaNumTensors; ++i) s += kSizes[i];
    return s;
}
constexpr bool offsets_consistent() {
    int64_t acc = 0;
    for (int i = 0; i < kMambaNumTensors; ++i) {
        if (kOffsets[i] != (int)acc) return false;
        acc += kSizes[i];
    }
    return true;
}
static_assert(sum_sizes() == kMambaTotalElems,
              "mamba3_layout: sum(kMambaSizes) != kMambaTotalElems. Re-derive "
              "from tests/hw/mamba_oracle.py::mamba_param_layout().");
static_assert(offsets_consistent(),
              "mamba3_layout: kMambaOffsets[i] != sum(kMambaSizes[0..i)).");
}  // namespace mamba_layout_check

// ── SMEM footprint of MambaSampleSmem (model_stage_mamba3.cuh). The Mamba CTA
//    smem EXCEEDS the 48 KB static cap (d_inner=256 + both-layer caching), so the
//    launcher MUST declare it DYNAMIC and opt in via
//    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
//    kMambaSmemBytes) before launch (sm_90 has ~228 KB smem/SM; one persistent
//    block/SM still places, occupancy>=1 holds, NO CUDA graphs). This is the
//    HONEST deviation from the decoder's <48 KB static footprint — see the
//    SMEM BUDGET note in model_stage_mamba3.cuh and INTEGRATION-MAMBA.md.
//
//    The number below is a static_assert-guarded mirror of sizeof(MambaSampleSmem)
//    computed field-by-field (all fields are float, 4-byte aligned -> no padding):
//      layer_in   [2][8][128]              = 2048
//      final_in   [8][128]                 = 1024
//      act[2].{ x_main_raw,z,conv,dt_pre,y_scan [8][256] (5*2048),
//               Bmat,Cmat [8][16] (2*128), ln_xhat [8][128], ln_inv [8] }
//                 per layer = 11528 ; x2  = 23056
//      fn_xhat[8][128]=1024; fn_inv[8]=8; logits[97]=97
//      dh,dr [8][128] (2*1024); adj_a,adj_b,adj_c [8][256] (3*2048)
//      dbc [8][40]=320; dBmat,dCmat [8][16] (2*128); red[256]
//      ------------------------------------------------------- = 36281 floats
constexpr int64_t kMambaSmemFloats = 36281;
constexpr int64_t kMambaSmemBytes  = kMambaSmemFloats * (int64_t)sizeof(float); // 145124
static_assert(kMambaSmemBytes > 48 * 1024,
              "mamba3_layout: if the smem ever drops below 48KB, switch the "
              "launcher back to STATIC smem (no opt-in needed).");
static_assert(kMambaSmemBytes < 224 * 1024,
              "mamba3_layout: CTA smem exceeds the sm_90 ~228KB/SM budget; "
              "one-block-per-SM would fail to place (GridBarrier hang). Shrink "
              "the live set (e.g. drop a cached LayerAct buffer).");

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MAMBA3_LAYOUT_CUH_
