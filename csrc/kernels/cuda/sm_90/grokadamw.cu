// =====================================================================
//  csrc/kernels/cuda/sm_90/grokadamw.cu
//
//  sm_90 (Hopper) GrokAdamW kernel — explicit instantiation TU.
//
//  All kernel + launcher logic lives in grokadamw.cuh as templates;
//  this TU forces emission of the dtype matrix below into a single
//  object file so callers outside this TU (specifically the four
//  torch::Tensor-form launchers in csrc/bindings/grokadamw.cpp) can
//  link against the per-tensor entry points without dragging the
//  whole header into a non-CUDA source.
//
//  Non-q3 dtype matrix (per main spec):
//     ParamT in {float, __nv_bfloat16, __half}
//     StateT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half,
//                __nv_fp8_e4m3, __nv_fp8_e5m2}
//
//  Coherent rows: FP32 param drops FP8 grads (rejected by
//  is_coherent_combo<float, FP8>). 3 ParamT × 2 StateT × 5 GradT = 30
//  combos, minus 2 (FP32-param × {FP8_E4M3, FP8_E5M2}) per StateT
//  branch = 30 - 4 = 26 instantiations.
//
//  Q3 dtype matrix:
//     ParamT in {float, __nv_bfloat16}
//     GradT  in {float, __nv_bfloat16, __half}
//  → 2 × 3 = 6 instantiations. FP8 grad rejected unconditionally
//  on q3 (is_q3_grad_dtype) — documented in static_assert.
//
//  The four torch::Tensor-form binding shims are defined inline (in
//  the SG_SM90_GROKADAMW_DEFINE_LAUNCHER block of the header) so this
//  TU ALSO emits their bodies. ODR-safe because this is the single TU
//  that #defines the macro before including the header.
// =====================================================================

#define SG_SM90_GROKADAMW_DEFINE_LAUNCHER 1

#include "csrc/kernels/cuda/sm_90/grokadamw.cuh"

namespace sg { namespace sm90 { namespace grokadamw {

// ---------------------------------------------------------------------
// X-macro tables for the non-q3 instantiation matrix.
// ---------------------------------------------------------------------

// Coherent (ParamT, GradT) cross-section excluding (FP32 param, FP8 grad).
#define SG_GAW_FOREACH_PARAM_GRAD(_)                  \
    _(float,         float)                           \
    _(float,         __nv_bfloat16)                   \
    _(float,         __half)                          \
    _(__nv_bfloat16, float)                           \
    _(__nv_bfloat16, __nv_bfloat16)                   \
    _(__nv_bfloat16, __half)                          \
    _(__nv_bfloat16, __nv_fp8_e4m3)                   \
    _(__nv_bfloat16, __nv_fp8_e5m2)                   \
    _(__half,        float)                           \
    _(__half,        __nv_bfloat16)                   \
    _(__half,        __half)                          \
    _(__half,        __nv_fp8_e4m3)                   \
    _(__half,        __nv_fp8_e5m2)

// ---------------------------------------------------------------------
// Non-q3: launch_grokadamw_fused_step<ParamT, StateT, GradT>
// 13 (ParamT,GradT) rows × 2 (StateT) = 26 instantiations.
// ---------------------------------------------------------------------

#define SG_GAW_INST_STEP(PARAM_T, STATE_T, GRAD_T)                          \
    template cudaError_t                                                    \
    launch_grokadamw_fused_step<PARAM_T, STATE_T, GRAD_T>(                  \
        PARAM_T*, STATE_T*, STATE_T*, STATE_T*, const GRAD_T*,              \
        float, float, float, float, float,                                  \
        float, float, float,                                                \
        float, float,                                                       \
        int64_t, int64_t, cudaStream_t);

#define SG_GAW_INST_STEP_FP32_STATE(PARAM_T, GRAD_T)                        \
    SG_GAW_INST_STEP(PARAM_T, float, GRAD_T)
#define SG_GAW_INST_STEP_BF16_STATE(PARAM_T, GRAD_T)                        \
    SG_GAW_INST_STEP(PARAM_T, __nv_bfloat16, GRAD_T)

SG_GAW_FOREACH_PARAM_GRAD(SG_GAW_INST_STEP_FP32_STATE)
SG_GAW_FOREACH_PARAM_GRAD(SG_GAW_INST_STEP_BF16_STATE)

#undef SG_GAW_INST_STEP_FP32_STATE
#undef SG_GAW_INST_STEP_BF16_STATE
#undef SG_GAW_INST_STEP

// ---------------------------------------------------------------------
// Q3: launch_grokadamw_fused_step_q3<ParamT, GradT>
// 2 ParamT × 3 GradT = 6 instantiations.
// FP8 grad rejected by static_assert is_q3_grad_dtype<GradT> at
// compile time — those rows are simply absent from the table below.
// ---------------------------------------------------------------------

#define SG_GAW_INST_STEP_Q3(PARAM_T, GRAD_T)                                \
    template cudaError_t                                                    \
    launch_grokadamw_fused_step_q3<PARAM_T, GRAD_T>(                        \
        PARAM_T*,                                                           \
        int8_t*, float*,                                                    \
        __nv_bfloat16*,                                                     \
        __nv_bfloat16*,                                                     \
        const GRAD_T*,                                                      \
        float, float, float, float, float,                                  \
        float, float, float,                                                \
        float, float,                                                       \
        int64_t, int64_t, cudaStream_t);

SG_GAW_INST_STEP_Q3(float,         float)
SG_GAW_INST_STEP_Q3(float,         __nv_bfloat16)
SG_GAW_INST_STEP_Q3(float,         __half)
SG_GAW_INST_STEP_Q3(__nv_bfloat16, float)
SG_GAW_INST_STEP_Q3(__nv_bfloat16, __nv_bfloat16)
SG_GAW_INST_STEP_Q3(__nv_bfloat16, __half)

#undef SG_GAW_INST_STEP_Q3
#undef SG_GAW_FOREACH_PARAM_GRAD

}}} // namespace sg::sm90::grokadamw
