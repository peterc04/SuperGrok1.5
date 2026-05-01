// =====================================================================
//  csrc/kernels/cuda/sm_90/grokfast.cu
//
//  Thin instantiation TU for the sm_90 Grokfast launchers declared in
//  csrc/kernels/cuda/sm_90/grokfast.cuh. All kernel logic lives in the
//  header; this TU only forces the template instantiations the linker
//  needs for the dtype matrix:
//
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Counts:
//     - launch_grokfast_fused_ema_adam_step : 3 × 2 × 5 = 30 combos
//     - launch_grokfast_fused_step          : 1 × 2 × 5 = 10 combos
//                                             (ParamT is unused on this
//                                             path; pin to float so the
//                                             SFINAE guard is satisfied
//                                             without expanding the matrix)
//     - Total                               : 40 instantiations
//
//  Incoherent dtype combos are rejected at compile time inside the
//  header via static_assert (see is_param_dtype / is_state_dtype /
//  is_grad_dtype) so any nonsense from a future caller fails the build
//  rather than miscompiling silently.
// =====================================================================

#include "csrc/kernels/cuda/sm_90/grokfast.cuh"

namespace sg { namespace sm90 { namespace grokfast {

// X-macro over the GradT axis. Used by both launchers.
#define SG_GROKFAST_FOREACH_GRAD(_)                  \
    _(float)                                         \
    _(__nv_bfloat16)                                 \
    _(__half)                                        \
    _(__nv_fp8_e4m3)                                 \
    _(__nv_fp8_e5m2)

// X-macro over (ParamT, StateT) for the fused EMA+Adam path.
#define SG_GROKFAST_FOREACH_PARAM_STATE(_)           \
    _(float,         float)                          \
    _(float,         __nv_bfloat16)                  \
    _(__nv_bfloat16, float)                          \
    _(__nv_bfloat16, __nv_bfloat16)                  \
    _(__half,        float)                          \
    _(__half,        __nv_bfloat16)

// ---------------------------------------------------------------------
// launch_grokfast_fused_ema_adam_step — 30 instantiations
// ---------------------------------------------------------------------

#define SG_GF_INST_ADAM(PARAM_T, STATE_T, GRAD_T)                              \
    template cudaError_t                                                       \
    launch_grokfast_fused_ema_adam_step<PARAM_T, STATE_T, GRAD_T>(             \
        PARAM_T*, STATE_T*, STATE_T*, STATE_T*, const GRAD_T*,                 \
        float, float, float, float, float,                                     \
        float, float, float, float,                                            \
        int64_t, int64_t, cudaStream_t);

#define SG_GF_INST_ADAM_BIND_PARAM_STATE(PARAM_T, STATE_T)                     \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, float)                                   \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_bfloat16)                           \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __half)                                  \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_fp8_e4m3)                           \
    SG_GF_INST_ADAM(PARAM_T, STATE_T, __nv_fp8_e5m2)

SG_GROKFAST_FOREACH_PARAM_STATE(SG_GF_INST_ADAM_BIND_PARAM_STATE)

#undef SG_GF_INST_ADAM_BIND_PARAM_STATE
#undef SG_GF_INST_ADAM

// ---------------------------------------------------------------------
// launch_grokfast_fused_step — 10 instantiations (ParamT pinned to float)
// ---------------------------------------------------------------------

#define SG_GF_INST_EMA(STATE_T, GRAD_T)                                        \
    template cudaError_t                                                       \
    launch_grokfast_fused_step<float, STATE_T, GRAD_T>(                        \
        STATE_T*, GRAD_T*, float, float, int64_t, cudaStream_t);

#define SG_GF_INST_EMA_BIND_STATE(STATE_T)                                     \
    SG_GF_INST_EMA(STATE_T, float)                                             \
    SG_GF_INST_EMA(STATE_T, __nv_bfloat16)                                     \
    SG_GF_INST_EMA(STATE_T, __half)                                            \
    SG_GF_INST_EMA(STATE_T, __nv_fp8_e4m3)                                     \
    SG_GF_INST_EMA(STATE_T, __nv_fp8_e5m2)

SG_GF_INST_EMA_BIND_STATE(float)
SG_GF_INST_EMA_BIND_STATE(__nv_bfloat16)

#undef SG_GF_INST_EMA_BIND_STATE
#undef SG_GF_INST_EMA

#undef SG_GROKFAST_FOREACH_PARAM_STATE
#undef SG_GROKFAST_FOREACH_GRAD

}}} // namespace sg::sm90::grokfast
