// =====================================================================
//  csrc/kernels/cuda/sm_90/prodigy.cu
//
//  sm_90 (Hopper) Prodigy optimizer kernels — explicit instantiation TU.
//
//  All kernel + launcher logic lives in prodigy.cuh as templates. This
//  TU forces emission of the dtype matrix below into a single object
//  file so callers outside this TU can link against the per-tensor
//  entry point without dragging the whole header into a non-CUDA TU.
//
//  Dtype matrix (per main spec):
//    ParamT in {float, __nv_bfloat16, __half}
//    StateT in {float, __nv_bfloat16}
//    GradT  in {float, __nv_bfloat16, __half,
//               __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Incoherent combos (FP8 grad with FP32 param) are statically rejected
//  by the templates — those rows are simply absent from the list below.
//  Total: 3 (FP32 param x 3 grads) * 2 states
//       + 5 (BF16 param x 5 grads) * 2 states
//       + 5 (FP16 param x 5 grads) * 2 states
//       = 6 + 10 + 10 = 26 instantiations.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/prodigy.cuh"

namespace sg { namespace sm90 { namespace prodigy {

// Convenience macro for the launcher signature (verbose enough that the
// compiler can pick the right overload, but short enough to keep the
// instantiation list scannable).
#define SG_PRODIGY_INST(P, S, G)                                              \
    template cudaError_t launch_prodigy_fused_step<P, S, G>(                  \
        P*, P*, S*, S*, S*, S*, const G*,                                     \
        float*, float*, float*,                                               \
        float, float, float, float, float,                                    \
        int64_t, int64_t, cudaStream_t)

// ---------------------------------------------------------------------
// FP32 param family — FP8 grads excluded (is_coherent_combo).
// ---------------------------------------------------------------------
SG_PRODIGY_INST(float, float, float);
SG_PRODIGY_INST(float, float, __nv_bfloat16);
SG_PRODIGY_INST(float, float, __half);
SG_PRODIGY_INST(float, __nv_bfloat16, float);
SG_PRODIGY_INST(float, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(float, __nv_bfloat16, __half);

// ---------------------------------------------------------------------
// BF16 param family — full grad cross-section (incl. FP8 grads).
// ---------------------------------------------------------------------
SG_PRODIGY_INST(__nv_bfloat16, float, float);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_bfloat16);
SG_PRODIGY_INST(__nv_bfloat16, float, __half);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_fp8_e4m3);
SG_PRODIGY_INST(__nv_bfloat16, float, __nv_fp8_e5m2);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, float);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __half);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3);
SG_PRODIGY_INST(__nv_bfloat16, __nv_bfloat16, __nv_fp8_e5m2);

// ---------------------------------------------------------------------
// FP16 param family — full grad cross-section.
// ---------------------------------------------------------------------
SG_PRODIGY_INST(__half, float, float);
SG_PRODIGY_INST(__half, float, __nv_bfloat16);
SG_PRODIGY_INST(__half, float, __half);
SG_PRODIGY_INST(__half, float, __nv_fp8_e4m3);
SG_PRODIGY_INST(__half, float, __nv_fp8_e5m2);
SG_PRODIGY_INST(__half, __nv_bfloat16, float);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_bfloat16);
SG_PRODIGY_INST(__half, __nv_bfloat16, __half);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_fp8_e4m3);
SG_PRODIGY_INST(__half, __nv_bfloat16, __nv_fp8_e5m2);

#undef SG_PRODIGY_INST

}}} // namespace sg::sm90::prodigy
