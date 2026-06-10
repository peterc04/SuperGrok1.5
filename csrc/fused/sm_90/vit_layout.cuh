#ifndef SG_FUSED_SM90_VIT_LAYOUT_CUH_
#define SG_FUSED_SM90_VIT_LAYOUT_CUH_
// ============================================================================
// csrc/fused/sm_90/vit_layout.cuh — weight-layout mirror for the L3-REAL
// Vision-Transformer megakernel (PHASE 2).
//
// HAND-WRITTEN (PHASE 2 deliverable) — MARKED FOR CODEGEN ADOPTION. The decoder
// twin (decoder_layout.cuh) is emitted by megakernel_codegen.py --decoder-layout;
// the integrator should add a `--vit-layout` emitter (mirroring
// _decoder_param_sizes / decoder_layout_header) whose output is byte-identical to
// THIS file, then regenerate so it cannot drift. Until then, the numbers below are
// generated FROM the single source of truth — tests/hw/vit_oracle.py
// ::vit_param_layout() (asserted == the live model's named_parameters() order in
// the parity test, tests/hw/test_vit_megakernel.py) — and guarded by the
// static_asserts below. A count/total mismatch fails the BUILD loudly, never
// corrupts at dispatch.
//
// The flat blob is torch.cat([p.reshape(-1) for _, p in model.named_parameters()]);
// the kernel addresses tensor i at params + kVitOffsets[i] for kVitSizes[i] elems.
//
// named_parameters() ORDER (32 tensors) — note cls_token (a leaf nn.Parameter on
// the ViT module) is yielded BEFORE the patch_proj submodule's params:
//   0  cls_token              [1,1,128]
//   1  patch_proj.weight      [128,49]
//   2  patch_proj.bias        [128]
//   3  pos.weight             [17,128]
//   4..15   layers.0.{attn.in_proj_w/b, attn.out_proj.w/b, n1.w/b, n2.w/b,
//                     ff.0.w/b, ff.2.w/b}
//   16..27  layers.1.{...same...}
//   28 norm.weight [128]  29 norm.bias [128]  30 out.weight [97,128]  31 out.bias [97]
// ============================================================================

#include <cstdint>

namespace sg { namespace fused { namespace sm90 {

constexpr int SG_VIT_VOCAB  = 97;          // p (head Linear(d, p))
constexpr int SG_VIT_D      = 128;
constexpr int SG_VIT_HEADS  = 4;
constexpr int SG_VIT_LAYERS = 2;
constexpr int SG_VIT_PATCH  = 49;          // patch pixel count (7×7)
constexpr int SG_VIT_NPATCH = 16;          // image patches
constexpr int SG_VIT_SEQ    = SG_VIT_NPATCH + 1;  // 17 (CLS + 16 patches)
constexpr int SG_VIT_DFF    = 4 * SG_VIT_D;       // 512

constexpr int     kVitNumTensors = 32;
constexpr int64_t kVitTotalElems = 418017;

// Per-tensor element offsets into the flat param blob, named_parameters() order.
__device__ __constant__ int kVitOffsets[kVitNumTensors] = {
    0, 128, 6400, 6528, 8704, 57856, 58240, 74624, 74752, 74880,
    75008, 75136, 75264, 140800, 141312, 206848, 206976, 256128, 256512, 272896,
    273024, 273152, 273280, 273408, 273536, 339072, 339584, 405120, 405248, 405376,
    405504, 417920
};

// Per-tensor element sizes (numel), same order.
__device__ __constant__ int kVitSizes[kVitNumTensors] = {
    128, 6272, 128, 2176, 49152, 384, 16384, 128, 128, 128,
    128, 128, 65536, 512, 65536, 128, 49152, 384, 16384, 128,
    128, 128, 128, 128, 65536, 512, 65536, 128, 128, 128,
    12416, 97
};

static_assert(kVitNumTensors == 32,
              "vit_layout: tensor count drifted. Regenerate from "
              "vit_oracle.py::vit_param_layout() (codegen adoption: --vit-layout).");
static_assert(kVitTotalElems == 418017,
              "vit_layout: total param count drifted. Regenerate.");

// Host-constexpr mirrors so a sum/offset cross-check folds at compile time (a
// __constant__ array can't be folded in a constexpr). These guarantee
// offsets/sizes/total agree.
namespace vit_layout_check {
constexpr int kSizes[kVitNumTensors] = {
    128, 6272, 128, 2176, 49152, 384, 16384, 128, 128, 128,
    128, 128, 65536, 512, 65536, 128, 49152, 384, 16384, 128,
    128, 128, 128, 128, 65536, 512, 65536, 128, 128, 128,
    12416, 97
};
constexpr int kOffsets[kVitNumTensors] = {
    0, 128, 6400, 6528, 8704, 57856, 58240, 74624, 74752, 74880,
    75008, 75136, 75264, 140800, 141312, 206848, 206976, 256128, 256512, 272896,
    273024, 273152, 273280, 273408, 273536, 339072, 339584, 405120, 405248, 405376,
    405504, 417920
};
constexpr int64_t sum_sizes() {
    int64_t s = 0;
    for (int i = 0; i < kVitNumTensors; ++i) s += kSizes[i];
    return s;
}
constexpr bool offsets_consistent() {
    int64_t acc = 0;
    for (int i = 0; i < kVitNumTensors; ++i) {
        if (kOffsets[i] != (int)acc) return false;
        acc += kSizes[i];
    }
    return true;
}
static_assert(sum_sizes() == kVitTotalElems,
              "vit_layout: sum(kVitSizes) != kVitTotalElems. Regenerate.");
static_assert(offsets_consistent(),
              "vit_layout: kVitOffsets[i] != sum(kVitSizes[0..i)). Regenerate.");

// ── Per-CTA dynamic-smem budget guard. The ViT per-sample working set
//    (VitSampleSmem, model_stage_vit.cuh) is ≈ 188,080 bytes (≈ 183.67 KB) at
//    seq=17 — FAR over the 48 KB STATIC __shared__ cap (so it MUST be dynamic
//    smem), but comfortably UNDER the sm_90 per-block dynamic cap of 227 KB
//    (232448 B). This bound is the size the launcher passes to
//    cudaFuncSetAttribute(MaxDynamicSharedMemorySize) + <<<dynamicSMemBytes>>>.
//    Computed from the field list (all float, 4 B, no padding):
//      patch 16*49 + layer_in 2*17*128 + final_in 17*128 + qkv 17*384
//      + ctx 17*128 + x1 17*128 + ff0 17*512 + gact 17*512 + attn 4*17*17
//      + n1_xhat 17*128 + n1_inv 17 + n2_xhat 17*128 + n2_inv 17
//      + fn_xhat 17*128 + fn_inv 17 + dh 17*128 + logits 97 + dsc 4*17*17
//      + red 256  = 47020 floats = 188080 bytes.
//    (If VitSampleSmem changes, update BOTH this literal and the sum below;
//    fused_vit_megakernel.cuh static_asserts sizeof(VitSampleSmem) against it.) ─
constexpr int kVitSampleSmemFloats =
    16 * 49            // patch[NPATCH][PATCH]
  + 2 * 17 * 128       // layer_in[LAYERS][SEQ][D]
  + 17 * 128           // final_in[SEQ][D]
  + 17 * 384           // qkv[SEQ][3D]
  + 17 * 128           // ctx[SEQ][D]
  + 17 * 128           // x1[SEQ][D]
  + 17 * 512           // ff0[SEQ][DFF]
  + 17 * 512           // gact[SEQ][DFF]
  + 4 * 17 * 17        // attn[HEADS][SEQ][SEQ]
  + 17 * 128 + 17      // n1_xhat[SEQ][D] + n1_inv[SEQ]
  + 17 * 128 + 17      // n2_xhat[SEQ][D] + n2_inv[SEQ]
  + 17 * 128 + 17      // fn_xhat[SEQ][D] + fn_inv[SEQ]
  + 17 * 128           // dh[SEQ][D]
  + 97                 // logits[VOCAB]
  + 4 * 17 * 17        // dsc[HEADS][SEQ][SEQ]
  + 256;               // red[256]
constexpr int kVitSampleSmemBytes = kVitSampleSmemFloats * (int)sizeof(float);
static_assert(kVitSampleSmemBytes == 188080,
              "vit_layout: VitSampleSmem byte budget drifted from 188080.");
static_assert(kVitSampleSmemBytes < 227 * 1024,
              "vit_layout: VitSampleSmem exceeds the sm_90 227 KB dynamic-smem "
              "per-block cap — the megakernel could not launch.");
}  // namespace vit_layout_check

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_VIT_LAYOUT_CUH_
