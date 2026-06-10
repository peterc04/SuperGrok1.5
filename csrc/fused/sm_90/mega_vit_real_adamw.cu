// csrc/fused/sm_90/mega_vit_real_adamw.cu — PHASE 2 host-launcher TU for the
// TRUE L3 fused (vit × adamw) megakernel. The ViT counterpart of PHASE 1's
// mega_decoder_real_adamw.cu.
//
// WHY A SEPARATE .cu (the 33-cell pattern): dispatch.cpp is HOST-compiled (it is
// a .cpp in the cpp_extension source list), so it cannot host <<<>>> launches,
// __global__ kernels, or device intrinsics. This nvcc-compiled TU (setup.py
// globs csrc/fused/sm_90/*.cu) owns ALL of that. It exposes ONE non-template host
// launcher whose boundary signature is decomposed to plain pointers/ints + the
// FusedScalars POD — NO header-only types (ViTInputCtx/FusedOptState) cross the
// boundary — so dispatch.cpp can `extern`-declare it using only the mirror
// structs it already has (sg::fused::PersistentContext, sg::fused::sm90::
// FusedScalars). The FQN + layout match → the mangling matches → it links.
//
// Inside (where the device header IS visible) it builds FusedOptState +
// ViTInputCtx and calls launch_fused_vit_megakernel<OptId::AdamW>, which runs the
// REAL ViT fwd+bwd + AdamW as one persistent kernel (no surrogate, no
// intermediate launches; the launcher does the dynamic-smem opt-in the ~184 KB
// VitSampleSmem requires). See fused_vit_megakernel.cuh + model_stage_vit.cuh
// (transcribed from the verified oracle).

#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

cudaError_t mega_vit_real_adamw(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* workspace, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream) {
    const int64_t total = kVitTotalElems;
    // AdamW state binding: state = [m | v | extra] (3*total) + 1 trailing loss
    // slot. extra is unused by AdamW (rebase_state<AdamW> guards it out).
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    ViTInputCtx in;
    in.patches   = patches;
    in.targets   = targets;
    in.B         = B;
    in.workspace = workspace;
    in.loss_out  = loss_out;

    return launch_fused_vit_megakernel<OptId::AdamW>(
        ctx, params, in, grad, lr, step, st, stream);
}

}}}  // namespace sg::fused::sm90
