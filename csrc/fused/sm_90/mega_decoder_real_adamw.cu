// csrc/fused/sm_90/mega_decoder_real_adamw.cu — PHASE 1 host-launcher TU for the
// TRUE L3 fused (transformer_decoder × adamw) megakernel.
//
// WHY A SEPARATE .cu (the 33-cell pattern): dispatch.cpp is HOST-compiled (it is
// a .cpp in the cpp_extension source list), so it cannot host <<<>>> launches,
// __global__ kernels, or device intrinsics. This nvcc-compiled TU (setup.py
// globs csrc/fused/sm_90/*.cu) owns ALL of that. It exposes ONE non-template host
// launcher whose boundary signature is decomposed to plain pointers/ints + the
// FusedScalars POD — NO header-only types (DecoderTokenCtx/FusedOptState) cross
// the boundary — so dispatch.cpp can `extern`-declare it using only the mirror
// structs it already has (sg::fused::PersistentContext, sg::fused::sm90::
// FusedScalars). The FQN + layout match → the mangling matches → it links.
//
// Inside (where the device header IS visible) it builds FusedOptState +
// DecoderTokenCtx and calls launch_fused_decoder_megakernel<OptId::AdamW>, which
// runs the REAL decoder fwd+bwd + AdamW as one persistent kernel (no surrogate,
// no intermediate launches). See fused_decoder_megakernel.cuh +
// model_stages_decoder.cuh (transcribed from the verified oracle).

#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"

// GEMM-engine selector (DESIGN-TC-PIPELINE.md §9). This cell TU is one of the
// per-TU sites the kernel-autotuner targets with -DSG_TUNED_GEMM_IMPL=<token>
// (the other being fused_decoder_megakernel.cuh, included above, which owns the
// token definitions + the scalar-default + the wgmma-pending #error). The guard
// is repeated here at the CELL boundary so a wgmma-requested decoder×adamw cell
// refuses LOUDLY at its own launcher TU — it never links a scalar body under the
// wgmma name (the owner no-suppression directive). When the Fork-B driver lands
// (DESIGN R2.3) the wgmma branch will select the tensor-core launcher here;
// until then the only buildable token is the scalar default (the shipped path).
#if (SG_TUNED_GEMM_IMPL == SG_GEMM_IMPL_WGMMA)
#error "mega_decoder_real_adamw: SG_TUNED_GEMM_IMPL=wgmma but the decoder Fork-B \
tensor-core cell driver is pending (DESIGN R2.3); see fused_decoder_megakernel.cuh."
#endif

namespace sg { namespace fused { namespace sm90 {

cudaError_t mega_decoder_real_adamw(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* workspace, float* loss_out,
        const int* sizes, const int* offsets,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream) {
    // sizes/offsets are unused on this path (the kernel reads the generated
    // __constant__ kDecSizes/kDecOffsets directly); kept in the signature for ABI
    // symmetry with the surrogate cells and forward-compat. Silence the warning.
    (void)sizes; (void)offsets;

#if !SG_DEC_SCALAR_MEGAKERNEL
    // The legacy fp32 SCALAR decoder megakernel is compile-gated OFF (it does not
    // fit smem at scaled SG_DEC_D; see the flag note in fused_decoder_megakernel
    // .cuh). The symbol is retained so dispatch.cpp links, but the fp32 fallback is
    // unavailable in this build — the production bf16 wgmma TC path carries every
    // routed cell, so this branch is dead in practice. Refuse LOUDLY (no surrogate).
    (void)ctx; (void)params; (void)tokens; (void)targets; (void)B; (void)state;
    (void)grad; (void)workspace; (void)loss_out; (void)lr; (void)step;
    (void)scalars; (void)stream;
    return cudaErrorNotSupported;
#else
    const int64_t total = kDecTotalElems;
    // AdamW state binding: state = [m | v | extra] (3*total) + 1 trailing loss
    // slot. extra is unused by AdamW (rebase_state<AdamW> guards it out).
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    DecoderTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = workspace;
    tok.loss_out  = loss_out;

    return launch_fused_decoder_megakernel<OptId::AdamW>(
        ctx, params, tok, grad, lr, step, st, stream);
#endif  // SG_DEC_SCALAR_MEGAKERNEL
}

}}}  // namespace sg::fused::sm90
