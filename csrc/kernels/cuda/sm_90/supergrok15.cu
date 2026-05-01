// =====================================================================
//  csrc/kernels/cuda/sm_90/supergrok15.cu
//
//  sm_90 (Hopper) SuperGrok v1.5 — explicit instantiation TU.
//
//  All kernel + launcher logic lives in supergrok15.cuh as templates.
//  This TU forces emission of the dtype matrix below into a single
//  object file so callers outside this TU (e.g. the per-arch shim that
//  the bindings dispatch through) can link against the per-tensor entry
//  points without dragging the cooperative-groups + fp8 headers into a
//  non-CUDA source.
//
//  Dtype matrix (per build spec):
//    ParamT in {float, __nv_bfloat16, __half}                     (3)
//    StateT in {float, __nv_bfloat16}                             (2)
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}                     (5)
//
//  Coherence rule (mirrors adamw.cu and supergrok11.cu):
//    FP8 grad with FP32 param is REJECTED via static_assert and is
//    therefore absent from the instantiation list below. All other
//    cells of the 3·2·5 = 30 cube are valid for fused_step (26 active);
//    SAM perturb / restore are only (ParamT, GradT) pairs (13 active
//    each). Total instantiations: 26 + 13 + 13 = 52.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/supergrok15.cuh"

namespace sg { namespace sm90 { namespace supergrok15 {

// =====================================================================
// fused_step instantiations (26 total)
// =====================================================================

#define INST_FUSED(P, S, G)                                                   \
    template cudaError_t launch_supergrok15_fused_step<P, S, G>(              \
        P*, S*, S*, S*,                                                       \
        const G*, const G*,                                                   \
        const float*,                                                         \
        float, float, float, float, float,                                    \
        float, float, float, float,                                           \
        float, float,                                                         \
        int, int,                                                             \
        int64_t, int64_t,                                                     \
        cudaStream_t)

// ---------------------------------------------------------------------
// FP32 param family — FP8 grads excluded by is_coherent_combo.
// State ∈ {FP32, BF16} × Grad ∈ {FP32, BF16, FP16}  → 6 cells.
// ---------------------------------------------------------------------
INST_FUSED(float, float,         float);
INST_FUSED(float, float,         __nv_bfloat16);
INST_FUSED(float, float,         __half);
INST_FUSED(float, __nv_bfloat16, float);
INST_FUSED(float, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(float, __nv_bfloat16, __half);

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section incl. FP8.
// 2 states × 5 grads = 10 cells.
// ---------------------------------------------------------------------
INST_FUSED(__nv_bfloat16, float,         float);
INST_FUSED(__nv_bfloat16, float,         __nv_bfloat16);
INST_FUSED(__nv_bfloat16, float,         __half);
INST_FUSED(__nv_bfloat16, float,         __nv_fp8_e4m3);
INST_FUSED(__nv_bfloat16, float,         __nv_fp8_e5m2);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, float);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __half);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
INST_FUSED(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section incl. FP8.
// 2 states × 5 grads = 10 cells.
// ---------------------------------------------------------------------
INST_FUSED(__half, float,         float);
INST_FUSED(__half, float,         __nv_bfloat16);
INST_FUSED(__half, float,         __half);
INST_FUSED(__half, float,         __nv_fp8_e4m3);
INST_FUSED(__half, float,         __nv_fp8_e5m2);
INST_FUSED(__half, __nv_bfloat16, float);
INST_FUSED(__half, __nv_bfloat16, __nv_bfloat16);
INST_FUSED(__half, __nv_bfloat16, __half);
INST_FUSED(__half, __nv_bfloat16, __nv_fp8_e4m3);
INST_FUSED(__half, __nv_bfloat16, __nv_fp8_e5m2);

#undef INST_FUSED

// =====================================================================
// sam_perturb_all instantiations (13 total)
// =====================================================================

#define INST_SAM(P, G)                                                        \
    template cudaError_t launch_supergrok15_sam_perturb_all<P, G>(            \
        P*, const G*, float, float, int64_t, cudaStream_t)

// FP32 param — FP8 grads excluded.
INST_SAM(float, float);
INST_SAM(float, __nv_bfloat16);
INST_SAM(float, __half);

// BF16 param — full grad cross-section.
INST_SAM(__nv_bfloat16, float);
INST_SAM(__nv_bfloat16, __nv_bfloat16);
INST_SAM(__nv_bfloat16, __half);
INST_SAM(__nv_bfloat16, __nv_fp8_e4m3);
INST_SAM(__nv_bfloat16, __nv_fp8_e5m2);

// FP16 param — full grad cross-section.
INST_SAM(__half, float);
INST_SAM(__half, __nv_bfloat16);
INST_SAM(__half, __half);
INST_SAM(__half, __nv_fp8_e4m3);
INST_SAM(__half, __nv_fp8_e5m2);

#undef INST_SAM

// =====================================================================
// sharpness_restore_all instantiations (13 total — mirror of SAM)
// =====================================================================

#define INST_RESTORE(P, G)                                                    \
    template cudaError_t launch_supergrok15_sharpness_restore_all<P, G>(      \
        P*, const G*, float, float, int64_t, cudaStream_t)

// FP32 param — FP8 grads excluded.
INST_RESTORE(float, float);
INST_RESTORE(float, __nv_bfloat16);
INST_RESTORE(float, __half);

// BF16 param — full grad cross-section.
INST_RESTORE(__nv_bfloat16, float);
INST_RESTORE(__nv_bfloat16, __nv_bfloat16);
INST_RESTORE(__nv_bfloat16, __half);
INST_RESTORE(__nv_bfloat16, __nv_fp8_e4m3);
INST_RESTORE(__nv_bfloat16, __nv_fp8_e5m2);

// FP16 param — full grad cross-section.
INST_RESTORE(__half, float);
INST_RESTORE(__half, __nv_bfloat16);
INST_RESTORE(__half, __half);
INST_RESTORE(__half, __nv_fp8_e4m3);
INST_RESTORE(__half, __nv_fp8_e5m2);

#undef INST_RESTORE

}}} // namespace sg::sm90::supergrok15
