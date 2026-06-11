// csrc/fused/sm_90/mega_vit_real_adamw_tc_launcher.cu — R2.4 WIRED tensor-core
// host-launcher TU for the TRUE L3 fused (vit × adamw) megakernel.
//
// Twin of mega_decoder_real_adamw_tc_launcher.cu for ViT (see that file's header
// for the full rationale). The shipped scalar launcher mega_vit_real_adamw.cu
// compiles the SCALAR cell only; the Fork-B TC driver mega_vit_real_adamw_tc.cu
// owns its own PYBIND11_MODULE and is dropped from _ops. This TU is the missing
// seam: globbed INTO _ops (no own pybind module), compiled -DSG_TUNED_GEMM_IMPL=1
// so it links the wgmma branch of fused_vit_megakernel.cuh, exposing ONE
// non-template host launcher with a plain pointers/ints + FusedScalars boundary
// (dispatch.cpp extern-decls it with the mirror structs). Calls ONLY
// launch_fused_vit_megakernel_tc → never instantiates the scalar launcher
// template → CANNOT collide with the scalar TU in _ops. Owns its own TC-sized
// activations/grad workspace (vit_tc_workspace_floats != the scalar nCTA*total
// partials), so dispatch.cpp passes no workspace for the TC path.
//
// ViT input is FLOAT image patches with int targets (dispatch.cpp bit-reinterprets
// the targets out of the trailing float slots and passes a real const int* here).
//
// HONESTY: the REAL bf16 wgmma path (HGMMA in SASS, validated by
// tests/hw/test_vit_tc.py — 21/21 gates). The scalar TU is UNTOUCHED.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver (Fork-B TC branch)

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

namespace {
struct VitTcLauncherScratch {
    int*      g_next = nullptr;
    unsigned* g_arrived = nullptr;
    unsigned* g_generation = nullptr;
    float*    workspace = nullptr;
    int64_t   ws_floats = 0;
    int       dev = -1;
};

VitTcLauncherScratch& vit_tc_launcher_scratch(int dev, int64_t need_floats) {
    static VitTcLauncherScratch s;
    if (s.dev != dev) {
        s.dev = dev;
        if (!s.g_next)       cudaMalloc(&s.g_next, sizeof(int));
        if (!s.g_arrived)    cudaMalloc(&s.g_arrived, sizeof(unsigned));
        if (!s.g_generation) cudaMalloc(&s.g_generation, sizeof(unsigned));
    }
    if (s.ws_floats < need_floats) {
        if (s.workspace) cudaFree(s.workspace);
        cudaMalloc(&s.workspace, (size_t)need_floats * sizeof(float));
        s.ws_floats = need_floats;
    }
    return s;
}
}  // anonymous namespace

// mega_vit_real_adamw_tc — the WIRED ViT TC launcher. Boundary mirrors the scalar
// mega_vit_real_adamw EXCEPT it takes NO workspace pointer (TC workspace is a
// different size; this TU owns it). `loss_out` points into the caller's
// state[3*total] slot.
//
// OPTID-GENERIC (owner baseline directive — all 33 cells on L3-TC): identical to the
// decoder TC launcher — the wgmma fwd+bwd is optimizer-independent, only the per-
// element tail (apply_optimizer<Opt>) differs. `opt_id` selects the kernel
// instantiation over the SAME [m|v|extra]+loss state buffer (ema = extra slice for
// grokfast/grokadamw; the neuralgrok psi-net pack lives at the head of extra). The
// STAGED/model-coupled/SG2 optimizers are NOT cases (no single-launch TC path);
// dispatch.cpp gates them out and an unsupported id returns cudaErrorInvalidValue.
cudaError_t mega_vit_real_adamw_tc(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap, int opt_id) {
    const int64_t total = kVitTotalElems;
    const int T = B * vit::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int nCTA = n_sms;
    if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap;

    const int64_t need = vit_tc_workspace_floats(T, B, nCTA);
    VitTcLauncherScratch& sc = vit_tc_launcher_scratch(dev, need);

    ctx.g_next_task  = sc.g_next;
    ctx.g_arrived    = sc.g_arrived;
    ctx.g_generation = sc.g_generation;
    ctx.n_tasks      = kVitNumTensors;

    float* const extra_slice = state + 2 * total;
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    st.ema        = extra_slice;          // grokfast/grokadamw slow-grad EMA
    st.psi_W1     = extra_slice + kPsiW1Off;   // neuralgrok psi-net pack
    st.psi_b1     = extra_slice + kPsiB1Off;
    st.psi_W2     = extra_slice + kPsiW2Off;
    // PRODIGY (STAGED global-d) state bindings — the ViT twin of the decoder TC
    // launcher's. The cell extends the state buffer to
    //   [m | v | extra/s_track | loss | param_init(total) | r_ema | s_ema | d_lr]
    //   = 4*total + 4 (host sizes it in dispatch.fused_train_step). The `extra` slice
    // doubles as Prodigy's `s` trajectory accumulator (the apply reads st.s_track,
    // not st.ema — they alias here, harmless: Prodigy has no slow-grad EMA). loss_out
    // points at state+3*total (dispatch's loss_slot), so param_init = loss_out + 1
    // and the 3 persisted estimator scalars [r_ema | s_ema | d_lr] follow it.
    st.s_track         = extra_slice;            // Prodigy `s` (per-element accumulator)
    st.param_init      = loss_out + 1;           // state + 3*total + 1
    st.prodigy_persist = loss_out + 1 + total;   // [r_ema | s_ema | d_lr]
    // LookSAM (MODEL-COUPLED SAM 2nd backward) — the ViT twin of the decoder TC
    // launcher's sam_dir binding. sam_dir is the PERSISTENT cached SAM direction
    // (g_sam − g), carried across steps in the `extra` slice (aliases the grokfast EMA /
    // Prodigy s_track slot — harmless: LookSAM has none of those). The in-kernel P2.4
    // phase WRITES it on SAM steps (st.looksam_sam!=0); the apply tail READS it every step.
    st.sam_dir         = extra_slice;            // [total] cached SAM direction
    // SuperGrok11/15 (MODEL-COUPLED SAM 2nd backward + per-tensor meta-net mu). State
    // layout: [m | v | mu | loss | sharpness(total) | phi_pack(4H+1)] — the ViT twin of
    // the decoder binding. st.mu = the `extra` slice (P2.45-filled, apply reads it);
    // st.sharpness = loss_out+1 (PERSISTED (g_sam−g)² from P2.4); st.sg_phi_W1/b1/W2 =
    // the host-scattered phi pack after sharpness (sg_phi_b2 read on-device from [H]).
    st.mu              = extra_slice;            // [total] meta-net mu (P2.45-filled)
    {
        float* sharp_base = loss_out + 1;        // state + 3*total + 1
        float* phi_base   = sharp_base + total;  // phi pack after sharpness
        st.sharpness = sharp_base;
        st.sg_phi_W1 = phi_base + kSgPhiW1Off;
        st.sg_phi_b1 = phi_base + kSgPhiB1Off;
        st.sg_phi_W2 = phi_base + kSgPhiW2Off;   // sg_phi_b2 read on-device from [H]
    }
    apply_scalars(st, scalars);   // FULL scalar set (+ d0/d_coef/beta3 + rho/looksam_sam/sg_rescale)
    st.lr = lr;

    ViTInputCtx in;
    in.patches   = patches;
    in.targets   = targets;
    in.B         = B;
    in.workspace = sc.workspace;
    in.loss_out  = loss_out;

    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return launch_fused_vit_megakernel_tc<OptId::AdamW>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return launch_fused_vit_megakernel_tc<OptId::Lion>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return launch_fused_vit_megakernel_tc<OptId::Grokfast>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            return launch_fused_vit_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            return launch_fused_vit_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Prodigy:
            // STAGED global-d: the SAME single persistent launch — the d-estimate
            // cross-tensor reduction is an IN-KERNEL phase (P2.6, between B2 and P3),
            // not a separate launch. st carries param_init/prodigy_persist/d0/d_coef/
            // beta3; the kernel decays+accumulates the EMA, updates d, applies.
            return launch_fused_vit_megakernel_tc<OptId::Prodigy>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Muon:
            // STAGED grid-cooperative Newton-Schulz: the SAME single persistent launch —
            // the NS orthogonalization of the 2D weights is an IN-KERNEL phase (P2.7,
            // between B2 and P3), looping the matrices with grid barriers. The momentum
            // buffer is the PERSISTENT m-slice (st.exp_avg, bound above); st.beta1 is the
            // Muon momentum; the 1D/non-2D weights take the AdamW tail in P3 (reading
            // st.exp_avg/exp_avg_sq). The per-matrix NS scratch lives in the workspace.
            return launch_fused_vit_megakernel_tc<OptId::Muon>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::LookSAM:
            // MODEL-COUPLED SAM 2nd backward (vit SAM-tier lane): the SAME single
            // persistent launch — the perturb→2nd in-kernel fwd+bwd→sam_dir=g_sam−g is
            // an IN-KERNEL phase (P2.4, between B2 and P2.5/P3), gated to every-k SAM
            // steps by st.looksam_sam. The transient backup + g_sam buffers live in the
            // workspace (vit_tc_looksam_floats, carved after the Muon scratch); sam_dir
            // persists in the extra slice (st.sam_dir, bound above). The apply tail (P3)
            // blends g_adj=(1−α)g+α·sam_dir and runs AdamW. NOT a separate launch. The
            // decoder twin (mega_decoder_real_adamw_tc_launcher.cu) proved this routing.
            return launch_fused_vit_megakernel_tc<OptId::LookSAM>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok11:
            // MODEL-COUPLED SAM 2nd backward + per-tensor meta-net mu/gate (wave-3 vit
            // lane): the SAME single persistent launch — the SAM 2nd backward (P2.4,
            // sharpness=(g_sam−g)²) + per-tensor mu precompute (P2.45) + per-tensor cosine
            // gate are in-kernel phases; the apply tail (sg11_sweep_b_step) runs in P3.
            // SAM scratch reuses the looksam buffers; sharpness + phi pack ride the
            // extended state (bound above). The decoder twin proved this routing.
            return launch_fused_vit_megakernel_tc<OptId::SuperGrok11>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok15:
            // Same SAM 2nd backward + mu precompute; SIMPLER tail (host-scalar gate, no
            // cosine stage). sg15_sweep_b_step does the per-coord alpha clip + AdamW.
            return launch_fused_vit_megakernel_tc<OptId::SuperGrok15>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;
    }
}

}}}  // namespace sg::fused::sm90
