// csrc/fused/sm_90/mega_mamba_real_adamw.cu — PHASE 2 host-launcher TU for the
// TRUE L3 fused (mamba × adamw) megakernel. The Mamba counterpart of PHASE 1's
// mega_decoder_real_adamw.cu.
//
// WHY A SEPARATE .cu (the 33-cell pattern): dispatch.cpp is HOST-compiled (it is
// a .cpp in the cpp_extension source list), so it cannot host <<<>>> launches,
// __global__ kernels, or device intrinsics. This nvcc-compiled TU (setup.py
// globs csrc/fused/sm_90/*.cu) owns ALL of that. It exposes ONE non-template host
// launcher whose boundary signature is decomposed to plain pointers/ints + the
// FusedScalars POD — NO header-only types (MambaTokenCtx/FusedOptState) cross the
// boundary — so dispatch.cpp can `extern`-declare it using only the mirror
// structs it already has (sg::fused::PersistentContext, sg::fused::sm90::
// FusedScalars). The FQN + layout match → the mangling matches → it links.
//
// Inside (where the device header IS visible) it builds FusedOptState +
// MambaTokenCtx and calls launch_fused_mamba_megakernel<OptId::AdamW>, which runs
// the REAL Mamba (selective-SSM) fwd+bwd + AdamW as one persistent kernel (no
// surrogate, no intermediate launches; the launcher does the dynamic-smem opt-in
// the ~142 KB MambaSampleSmem requires). See fused_mamba_megakernel.cuh +
// model_stage_mamba3.cuh (transcribed from the verified oracle, incl. the
// selective-scan reverse-time backward).

#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

cudaError_t mega_mamba_real_adamw(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* workspace, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream) {
    const int64_t total = kMambaTotalElems;
    // AdamW state binding: state = [m | v | extra] (3*total) + 1 trailing loss
    // slot. extra is unused by AdamW (rebase_state<AdamW> guards it out).
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    MambaTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = workspace;
    tok.loss_out  = loss_out;

    return launch_fused_mamba_megakernel<OptId::AdamW>(
        ctx, params, tok, grad, lr, step, st, stream);
}

}}}  // namespace sg::fused::sm90
