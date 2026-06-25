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
#if defined(SG_HAS_NVSHMEM)
#include <nvshmem.h>   // nvshmem_malloc/_free + nvshmem_team_my_pe + NVSHMEM_TEAM_WORLD (EDIT E)
#endif

namespace sg { namespace fused { namespace sm90 {

namespace {
struct VitTcLauncherScratch {
    int*      g_next = nullptr;
    unsigned* g_arrived = nullptr;
    unsigned* g_generation = nullptr;
    float*    workspace = nullptr;     // cudaMalloc — acts/grad/state
    int64_t   ws_floats = 0;
#if defined(SG_HAS_NVSHMEM)
    // SYMMETRIC TP-slot heap (nvshmem_malloc — the ONLY operands that need cross-PE
    // addressing; tp_nvshmem.md §2 Option A). Sized to the WORLD-UNIFORM per-PE
    // stride so every PE's collective nvshmem_malloc agrees. nullptr on the TP==1 /
    // no-NVSHMEM path. The ViT twin of the decoder launcher's tp_sym_heap (EDIT E).
    float*    tp_sym_heap = nullptr;   // nvshmem_malloc'd [tp_sym_floats]
    int64_t   tp_sym_floats = 0;
#endif
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

#if defined(SG_HAS_NVSHMEM)
// Ensure the symmetric TP-slot heap is sized >= need_sym_floats. COLLECTIVE:
// every PE must call nvshmem_malloc with the SAME size in the SAME order, so the
// caller passes the WORLD-UNIFORM stride (computed from the global shapes, not a
// per-rank size). nvshmem_malloc is a collective barrier internally; call it from
// the host TP-group bootstrap BEFORE the kernel launch. ViT twin of the decoder
// launcher's dec_tc_ensure_tp_sym_heap (EDIT E).
void vit_tc_ensure_tp_sym_heap(VitTcLauncherScratch& s, int64_t need_sym_floats) {
    if (s.tp_sym_floats >= need_sym_floats) return;
    if (s.tp_sym_heap) nvshmem_free(s.tp_sym_heap);
    s.tp_sym_heap   = static_cast<float*>(
        nvshmem_malloc((size_t)need_sym_floats * sizeof(float)));
    s.tp_sym_floats = need_sym_floats;
}
#endif
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
        int ncta_cap, int opt_id, int tp_size) {
    (void)tp_size;               // unread unless SG_HAS_NVSHMEM (TP dispatch arm below).
    // NOTE: this is the TP-AWARE definition (trailing int tp_size). The original
    // 15-arg boundary dispatch.cpp links is provided as a thin BYTE-IDENTICAL
    // forwarder OVERLOAD just below this function (tp_size=1), so the shipped _ops
    // symbol is unchanged and dispatch.cpp needs no edit — the new tp_size>1 path
    // is reached only by a (future) dispatch.cpp caller of this 16-arg overload.
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
#if defined(SG_HAS_NVSHMEM)
            // TP allow-list (the §1.3/§7.2 explicit-instantiation gate): {1, 8}.
            // DP rides in CommCtx at runtime (no Par::kDP read in the kernel), so a
            // fixed DP sentinel avoids a DP×TP instantiation matrix (dist_step.md §6.C.4).
            if (tp_size == 8) {
                using ParTP8 = ::sg::fused::par::ParConfig<
                    /*DP=*/8, /*TP=*/8, /*PP=*/1, /*SP=*/1,
                    ::sg::fused::par::ZeROStage::Z3>;
                // Symmetric TP-slot heap: one publish+reduced slot per CTA-in-PE.
                // WORLD-UNIFORM stride (every PE agrees) = ctas_per_pe·2·kTileM·d.
                const int P = tp_size;
                const int ctas_per_pe = nCTA / P;   // launcher asserts nCTA % P == 0
                const int64_t sym_floats =
                    ::sg::fused::sm90::vittc::vit_tp_heap_stride_floats(ctas_per_pe);
                vit_tc_ensure_tp_sym_heap(sc, sym_floats);
                ::sg::fused::par::CommCtx comm{};
                comm.world_size = 8; comm.tp_size = 8; comm.dp_size = 8;
                comm.tp_rank = nvshmem_team_my_pe(/*TP team*/NVSHMEM_TEAM_WORLD);
                comm.tp_sym_heap = sc.tp_sym_heap;
                comm.tp_heap_stride_floats = sym_floats;
                comm.tp_team_n_pes  = 8;
                comm.tp_team_local_pe = comm.tp_rank;
                // Store the TP team id as void* (int32 team → intptr → void*); the
                // host bootstrap that nvshmem_team_split_strided's the TP group sets
                // the real team — NVSHMEM_TEAM_WORLD for the single-node pure-TP run.
                comm.tp_comm_handle = reinterpret_cast<void*>(
                    static_cast<intptr_t>(NVSHMEM_TEAM_WORLD));
                return launch_fused_vit_megakernel_tc<OptId::AdamW, ParTP8>(
                    ctx, params, in, grad, lr, step, st, stream, nCTA, comm);
            }
#endif
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

// BACK-COMPAT 15-arg forwarder — the EXACT boundary dispatch.cpp extern-decls and
// links today (no tp_size). Forwards to the 16-arg TP-aware definition with
// tp_size=1 ⇒ the single-GPU path, byte-identical to the pre-TP launcher. The
// shipped _ops symbol is therefore unchanged; the TP path (tp_size>1) is reached
// only by a future dispatch.cpp caller of the 16-arg overload (EDIT E.2).
cudaError_t mega_vit_real_adamw_tc(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* workspace_unused, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap, int opt_id) {
    return mega_vit_real_adamw_tc(
        ctx, params, patches, targets, B, state, grad, workspace_unused, loss_out,
        lr, step, scalars, stream, ncta_cap, opt_id, /*tp_size=*/1);
}

// mega_vit_sg2_tc — the DEDICATED SuperGrok2 ViT TC launcher (twin of
// mega_decoder_sg2_tc). Binds the SG2 meta-net state slices + the HBM weight bundle
// + per-tensor scalar arrays, dispatches launch_fused_vit_megakernel_tc<SuperGrok2>.
// State layout: [m | v | mu | loss | sharpness | slow | gru_state(total*GH)].
cudaError_t mega_vit_sg2_tc(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* loss_out,
        const float* input_proj_W, const float* input_proj_b,
        const float* csa_q_W, const float* csa_k_W, const float* csa_v_W, const float* csa_out_W,
        const float* csa_compress_w, const float* csa_idx_DQ, const float* csa_idx_K,
        const float* hca_q_W, const float* hca_k_W, const float* hca_v_W, const float* hca_out_W,
        const float* gru_Wz, const float* gru_bz, const float* gru_Wr, const float* gru_br,
        const float* gru_Wh, const float* gru_bh,
        const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
        const float* expert_W1, const float* expert_b1, const float* expert_W2, const float* expert_b2,
        const float* sc_alpha, const float* sc_gru_decay, const float* sc_lamb_eff,
        const float* sc_beta1, const float* sc_bc1, const float* sc_bc2,
        float rescale, float beta2, float lr, float wd, float eps,
        float rho, float sam_on,
        int step, cudaStream_t stream, int ncta_cap) {
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

    FusedOptState st;
    st.exp_avg     = state;
    st.exp_avg_sq  = state + total;
    st.mu          = state + 2 * total;
    float* sharp_base = loss_out + 1;             // state + 3*total + 1
    st.sharpness   = sharp_base;
    st.sg2_slow    = sharp_base + total;
    st.sg2_gru_state = st.sg2_slow + total;
    st.sg2_input_proj_W = input_proj_W; st.sg2_input_proj_b = input_proj_b;
    st.sg2_csa_q_W = csa_q_W; st.sg2_csa_k_W = csa_k_W; st.sg2_csa_v_W = csa_v_W;
    st.sg2_csa_out_W = csa_out_W; st.sg2_csa_compress_w = csa_compress_w;
    st.sg2_csa_idx_DQ = csa_idx_DQ; st.sg2_csa_idx_K = csa_idx_K;
    st.sg2_hca_q_W = hca_q_W; st.sg2_hca_k_W = hca_k_W; st.sg2_hca_v_W = hca_v_W;
    st.sg2_hca_out_W = hca_out_W;
    st.sg2_gru_Wz = gru_Wz; st.sg2_gru_bz = gru_bz; st.sg2_gru_Wr = gru_Wr;
    st.sg2_gru_br = gru_br; st.sg2_gru_Wh = gru_Wh; st.sg2_gru_bh = gru_bh;
    st.sg2_peer_query_Ws = peer_query_Ws; st.sg2_prod_keys_A = prod_keys_A;
    st.sg2_prod_keys_B = prod_keys_B;
    st.sg2_expert_W1 = expert_W1; st.sg2_expert_b1 = expert_b1;
    st.sg2_expert_W2 = expert_W2; st.sg2_expert_b2 = expert_b2;
    st.sg2_alpha = sc_alpha; st.sg2_gru_decay = sc_gru_decay; st.sg2_lamb_eff = sc_lamb_eff;
    st.sg2_beta1 = sc_beta1; st.sg2_bc1 = sc_bc1; st.sg2_bc2 = sc_bc2;
    st.sg2_rescale = rescale;
    st.beta2 = beta2; st.lr = lr; st.wd = wd; st.eps = eps;
    st.rho = rho;
    st.looksam_sam = sam_on;

    ViTInputCtx in;
    in.patches   = patches;
    in.targets   = targets;
    in.B         = B;
    in.workspace = sc.workspace;
    in.loss_out  = loss_out;

    return launch_fused_vit_megakernel_tc<OptId::SuperGrok2>(
        ctx, params, in, grad, lr, step, st, stream, nCTA);
}

}}}  // namespace sg::fused::sm90
