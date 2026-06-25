// csrc/fused/sm_90/mega_mamba_real_adamw_tc_launcher.cu — cycle-2 WIRED tensor-core
// host-launcher TU for the TRUE L3 fused (mamba3 × {adamw,lion,grokfast}) megakernel.
//
// Twin of mega_{decoder,vit}_real_adamw_tc_launcher.cu for Mamba (see the decoder
// file's header for the full rationale). The standalone TC driver
// mega_mamba_real_adamw_tc.cu owns its own PYBIND11_MODULE and is dropped from _ops
// by setup.py's content-based glob filter, so the race path (dispatch.cpp →
// _try_fused_train_step) had NO way to reach the mamba TC kernel — bf16 mamba race
// configs ran the scalar L3 megakernel (the measured 905a4bb carve-out, TC 0.46×).
//
// This TU is the missing seam: it is globbed INTO _ops (no own pybind module), is
// compiled with -DSG_TUNED_GEMM_IMPL=1 (the in-source #define below) so it links the
// wgmma branch of fused_mamba_megakernel.cuh, and exposes ONE non-template host
// launcher whose boundary is plain pointers/ints + the FusedScalars POD — NO
// header-only types (MambaTokenCtx / FusedOptState) cross the boundary — so
// dispatch.cpp can extern-declare it with only the mirror structs it already has. It
// calls ONLY launch_fused_mamba_megakernel_tc (the TC launcher); it never
// instantiates the scalar launcher template, so it CANNOT collide (ODR/duplicate-
// symbol) with the scalar mega_mamba_real_adamw.cu TU even though both are in _ops.
// The TC activations/grad workspace is sized DIFFERENTLY from the scalar
// MambaScratch partials (mb_tc_workspace_floats), so this TU owns its own cached
// scratch internally — dispatch.cpp passes no workspace for the TC path.
//
// WHY MAMBA IS NOW ON THE TC PATH (owner cycle-2 directive (c)): mamba×adamw was
// excluded from _L3_WGMMA_CELLS as a measured scalar-WINS carve-out (0.46×, the
// selective-scan/conv1d dominate, not the projection GEMMs). That is a PERFORMANCE
// fact the roofline surfaces — NOT a correctness reason: the mamba TC kernel is
// validated 5/5 by tests/hw/test_mamba_tc.py. The directive lists mamba conversion
// explicitly, and no-suppression cuts toward exposing the path + letting the roofline
// report the (honest) 0.46×, rather than hiding a correct kernel. The scalar TU
// (mega_mamba_real_adamw.cu) is UNTOUCHED and remains the fp32 path; mamba×adamw at
// fp32 still routes there (gemm_impl_for_cell returns "scalar" for fp32).
//
// OPTID-GENERIC: identical to the decoder/vit TC launchers — the wgmma fwd+bwd is
// optimizer-independent, only the per-element tail (apply_optimizer<Opt>) differs.
// `opt_id` selects the kernel instantiation over the SAME [m|v|extra]+loss state
// buffer (ema = extra slice for grokfast; the neuralgrok psi-net pack would live at
// the head of extra). The STAGED/model-coupled/SG2 optimizers are NOT cases (no
// single-launch TC path); dispatch.cpp gates them out and an unsupported id returns
// cudaErrorInvalidValue → dispatch throws LOUD (no silent scalar/adamw fallback).
//
// HONESTY: this is the REAL bf16 wgmma path (HGMMA in SASS for the 4 projection
// GEMMs; the scan/conv stay scalar by design — REUSED verbatim), validated by
// tests/hw/test_mamba_tc.py.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver (Fork-B TC branch)

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"
#if defined(SG_HAS_NVSHMEM)
// Real NVSHMEM host API — present ONLY when the toolkit is on the path and the
// build opts in (-DSG_HAS_NVSHMEM=1 -rdc=true). Used to carve the symmetric
// TP-slot heap (nvshmem_malloc) + read the TP team pe (decoder EDIT E mirror).
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace sg { namespace fused { namespace sm90 {

namespace {
// Persistent barrier/queue counters + the TC activations/grad workspace, cached
// per device. Mirrors mega_{decoder,vit}_real_adamw_tc_launcher.cu's scratch but
// for mamba's workspace shape. Raw cudaMalloc (process-lived scratch, recreated
// only when a larger B needs a bigger workspace; never freed — one per device,
// leaked at process exit like every cuBLAS/cuDNN handle).
struct MbTcLauncherScratch {
    int*      g_next = nullptr;        // int [1]
    unsigned* g_arrived = nullptr;     // unsigned [1]
    unsigned* g_generation = nullptr;  // unsigned [1]
    float*    workspace = nullptr;     // float [mb_tc_workspace_floats(T, nCTA)] (cudaMalloc — grad/state)
    int64_t   ws_floats = 0;
#if defined(SG_HAS_NVSHMEM)
    // SYMMETRIC TP-slot heap (nvshmem_malloc — the ONLY operands that need cross-PE
    // addressing; decoder tp_kernel.md §2 / EDIT E mirror). Sized to the WORLD-UNIFORM
    // per-PE stride so every PE's collective nvshmem_malloc agrees. nullptr on the
    // TP==1 / no-NVSHMEM path.
    float*   tp_sym_heap = nullptr;   // nvshmem_malloc'd [tp_sym_floats]
    int64_t  tp_sym_floats = 0;
#endif
    int       dev = -1;
};

MbTcLauncherScratch& mb_tc_launcher_scratch(int dev, int64_t need_floats) {
    static MbTcLauncherScratch s;
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
// Ensure the symmetric TP-slot heap is sized >= need_sym_floats. COLLECTIVE: every
// PE must call nvshmem_malloc with the SAME size in the SAME order, so the caller
// passes the WORLD-UNIFORM stride (computed from the global shapes, not a per-rank
// size). nvshmem_malloc is a collective barrier internally; call it from the host
// TP-group bootstrap BEFORE the kernel launch. (Decoder dec_tc_ensure_tp_sym_heap mirror.)
void mb_tc_ensure_tp_sym_heap(MbTcLauncherScratch& s, int64_t need_sym_floats) {
    if (s.tp_sym_floats >= need_sym_floats) return;
    if (s.tp_sym_heap) nvshmem_free(s.tp_sym_heap);
    s.tp_sym_heap   = static_cast<float*>(
        nvshmem_malloc((size_t)need_sym_floats * sizeof(float)));
    s.tp_sym_floats = need_sym_floats;
}
#endif
}  // anonymous namespace

// mega_mamba_real_adamw_tc — the WIRED Mamba TC launcher. Boundary mirrors the scalar
// mega_mamba_real_adamw (dispatch.cpp extern-decls it with the same mirror types)
// EXCEPT it takes NO workspace pointer: the TC workspace is a different size from the
// scalar partials, so this TU allocates+owns it. `loss_out` still points into the
// caller's state[3*total] slot (dispatch reads it back). `opt_id` selects the tail.
//
// ncta_cap: forwarded to the TC launcher (0 = one CTA/SM = full saturation, the
// shipped config). The race always passes 0.
cudaError_t mega_mamba_real_adamw_tc(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap, int opt_id, int tp_size = 1) {
    const int64_t total = kMambaTotalElems;
    const int T = B * mb::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    // Size the workspace for the EXACT count the kernel launcher runs (occ·n_sms with
    // the occupancy-fill default; reg-capped uniform across OptIds, so AdamW's count
    // == every tail's). Falls back to n_sms if the occ query fails.
    int nCTA = mb_tc_launched_nctas<OptId::AdamW>(dev, ncta_cap);
    if (nCTA <= 0) { nCTA = n_sms; if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap; }

    const int64_t need = mb_tc_workspace_floats(T, nCTA);
    MbTcLauncherScratch& sc = mb_tc_launcher_scratch(dev, need);

    // Re-point the ctx counters at THIS TU's cached counters (the TC launcher zeroes
    // them per-launch, but they must be the ones this scratch owns so the grid
    // barrier is consistent with the workspace).
    ctx.g_next_task  = sc.g_next;
    ctx.g_arrived    = sc.g_arrived;
    ctx.g_generation = sc.g_generation;
    ctx.n_tasks      = kMambaNumTensors;   // parameter tensors for reduce+opt phases

    // Bind the optimizer state slices. [m | v | extra] over the SAME state buffer;
    // each tail reads only its own buffers (opt_components.cuh guarantees unused
    // pointers are never dereferenced). ema/psi-net live in the `extra` slice.
    float* const extra_slice = state + 2 * total;
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    st.ema        = extra_slice;          // grokfast/grokadamw slow-grad EMA
    st.psi_W1     = extra_slice + kPsiW1Off;   // neuralgrok psi-net pack (when wired)
    st.psi_b1     = extra_slice + kPsiB1Off;
    st.psi_W2     = extra_slice + kPsiW2Off;
    // PRODIGY (STAGED global-d) state bindings (mirror the decoder/vit TC launchers).
    // The cell extends the state buffer to [m | v | extra/s_track | loss | param_init |
    // r_ema | s_ema | d_lr] (host sizes it >= 4*total+4; dispatch.fused_train_step). The
    // `extra` slice doubles as Prodigy's `s` trajectory accumulator (the apply reads
    // st.s_track, not st.ema — they alias here, harmless: Prodigy has no slow-grad EMA).
    // loss_out == dispatch's loss_slot (state+3*total), so param_init = loss_out+1 and
    // the 3 persisted estimator scalars [r_ema | s_ema | d_lr] follow param_init.
    st.s_track        = extra_slice;            // Prodigy `s` (per-element accumulator)
    st.param_init     = loss_out + 1;           // state + 3*total + 1
    st.prodigy_persist = loss_out + 1 + total;  // [r_ema | s_ema | d_lr]
    // LookSAM (MODEL-COUPLED SAM 2nd backward) — the Mamba twin of the decoder/vit TC
    // launchers' sam_dir binding. sam_dir is the PERSISTENT cached SAM direction
    // (g_sam − g), carried across steps in the `extra` slice (aliases the grokfast EMA /
    // Prodigy s_track slot — harmless: LookSAM has none of those). The in-kernel P2.4
    // phase WRITES it on SAM steps (st.looksam_sam!=0); the apply tail READS it every step.
    st.sam_dir        = extra_slice;            // [total] cached SAM direction
    // SuperGrok11/15 (MODEL-COUPLED SAM 2nd backward + per-tensor meta-net mu) — the
    // mamba twin of the decoder/vit TC launchers. State layout:
    //   [m | v | mu | loss | sharpness(total) | phi_pack(4H+1)].
    //   * st.mu = the `extra` slice (recomputed each step by the P2.45/P3 precompute; the
    //     apply reads it — it aliases sam_dir/ema/s_track, harmless: SG has none).
    //   * st.sharpness = loss_slot+1 (PERSISTED (g_sam−g)²; the P2.4 SAM 2nd backward
    //     writes it on SAM steps, the cached value is reused on intervening steps).
    //   * st.sg_phi_W1/b1/W2 = the phi pack at sharpness+total (host-scattered each step,
    //     like NeuralGrok's psi pack); sg_phi_b2 read on-device from sg_phi_W2[H].
    st.mu             = extra_slice;            // [total] meta-net mu (P2.45/P3-filled)
    {
        float* sharp_base = loss_out + 1;       // state + 3*total + 1
        float* phi_base   = sharp_base + total; // phi pack after sharpness
        st.sharpness = sharp_base;
        st.sg_phi_W1 = phi_base + kSgPhiW1Off;
        st.sg_phi_b1 = phi_base + kSgPhiB1Off;
        st.sg_phi_W2 = phi_base + kSgPhiW2Off;  // sg_phi_b2 read on-device from [H]
    }
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/... + d0/d_coef/beta3 + rho/looksam_sam/sg_rescale)
    st.lr = lr;

    MambaTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = sc.workspace;
    tok.loss_out  = loss_out;

    // Dispatch to the matching kernel instantiation. The fwd+bwd is identical across
    // these OptIds; the template differs ONLY in the P3 apply_optimizer<Opt> tail.
    // Unsupported opt_ids return cudaErrorInvalidValue → dispatch.cpp throws LOUD.
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
#if defined(SG_HAS_NVSHMEM)
            // TP allow-list (the decoder §1.3/§7.2 explicit-instantiation gate): {1, 8}.
            // DP rides in CommCtx at runtime (no Par::kDP read in the kernel), so a fixed
            // DP sentinel avoids a DP×TP instantiation matrix (dist_step.md §6.C.4).
            if (tp_size == 8) {
                using ParTP8 = ::sg::fused::par::ParConfig<
                    /*DP=*/8, /*TP=*/8, /*PP=*/1, /*SP=*/1,
                    ::sg::fused::par::ZeROStage::Z3>;
                // Symmetric TP-slot heap: one publish+reduced slot per CTA-in-PE. The
                // WORLD-UNIFORM stride (every PE agrees) = ctas_per_pe·2·kSeq·d.
                const int P = tp_size;
                const int ctas_per_pe = nCTA / P;   // launcher asserts nCTA % P == 0
                const int64_t sym_floats =
                    ::sg::fused::sm90::mbtc::mb_tp_heap_stride_floats(ctas_per_pe);
                mb_tc_ensure_tp_sym_heap(sc, sym_floats);
                ::sg::fused::par::CommCtx comm{};
                comm.world_size = 8; comm.tp_size = 8; comm.dp_size = 8;
                comm.tp_rank = nvshmem_team_my_pe(/*TP team*/NVSHMEM_TEAM_WORLD);
                comm.tp_sym_heap = sc.tp_sym_heap;
                comm.tp_heap_stride_floats = sym_floats;
                comm.tp_team_n_pes  = 8;
                comm.tp_team_local_pe = comm.tp_rank;
                // Store the TP team id as void* (int32 team → intptr → void*); the host
                // bootstrap that nvshmem_team_split_strided's the TP group sets the real
                // team — NVSHMEM_TEAM_WORLD for the single-node pure-TP run.
                comm.tp_comm_handle = reinterpret_cast<void*>(
                    static_cast<intptr_t>(NVSHMEM_TEAM_WORLD));
                return launch_fused_mamba_megakernel_tc<OptId::AdamW, ParTP8>(
                    ctx, params, tok, grad, lr, step, st, stream, nCTA, comm);
            }
#endif
            (void)tp_size;
            return launch_fused_mamba_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return launch_fused_mamba_megakernel_tc<OptId::Lion>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return launch_fused_mamba_megakernel_tc<OptId::Grokfast>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            // 3-mechanism GrokAdamW (wave-2 mamba): the kernel's P2.5 global grad-norm
            // clip + P3 per-tensor layer-wise β1 land in fused_mamba_megakernel_tc<Opt>
            // (mirrors the decoder/vit TC kernels). γ/grad_clip thread through
            // FusedScalars (apply_scalars). ema = the `extra` slice (cold-start seed).
            return launch_fused_mamba_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            // psi-net MLP tail (wave-2 mamba): apply_optimizer<NeuralGrok> reads the
            // psi pack the host scattered into the `extra` slice (st.psi_W1/b1/W2 bound
            // above) + alpha/beta from FusedScalars. Pure elementwise — no precompute,
            // no cross-CTA reduction, so it inherits the deterministic grad reduce.
            return launch_fused_mamba_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Prodigy:
            // STAGED global-d (wave-2 mamba): the SAME single persistent launch — the
            // cross-ALL-tensors d-estimate is an IN-KERNEL phase (P2.6, between the grad
            // reduce B2 and the apply P3), not a separate launch. st carries param_init/
            // prodigy_persist/d0/d_coef/beta3 (bound above); the mamba TC kernel decays+
            // accumulates the EMA, updates d, applies. The reduce slots are carved in
            // mb_tc_workspace_floats (mb_tc_opt_reduce_floats), so this scratch already fits.
            return launch_fused_mamba_megakernel_tc<OptId::Prodigy>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::LookSAM:
            // MODEL-COUPLED SAM 2nd backward (mamba SAM-tier lane): the SAME single
            // persistent launch — the perturb→2nd in-kernel fwd+bwd→sam_dir=g_sam−g is
            // an IN-KERNEL phase (P2.4, between B2 and P2.5/P3), gated to every-k SAM
            // steps by st.looksam_sam. The transient backup + g_sam buffers live in the
            // workspace (mb_tc_looksam_floats, carved after the Prodigy reduce slots);
            // sam_dir persists in the extra slice (st.sam_dir, bound above). The apply
            // tail (P3) blends g_adj=(1−α)g+α·sam_dir and runs AdamW. NOT a separate
            // launch. The decoder/vit twins proved this routing.
            return launch_fused_mamba_megakernel_tc<OptId::LookSAM>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Muon:
            // STAGED grid-cooperative Newton-Schulz (mamba muon lane): the SAME single
            // persistent launch — the NS orthogonalization of the 13 2D weights (tok/pos/
            // A_log/in_proj/x_proj/dt_proj/out_proj × 2 layers + out.weight) is an IN-KERNEL
            // phase (P2.7, between B2/P2.6 and P3), grid-barrier-looped per matrix, NOT a
            // separate launch. The per-matrix NS scratch lives in the workspace
            // (mb_tc_muon_floats, carved after the LookSAM scratch). The persistent momentum
            // buffer is the m-slice (st.exp_avg, bound above); the 1D/non-2D weights take the
            // AdamW tail in P3 reading the INDEPENDENT aux_lr/aux_betas (eager Muon adamw_*
            // group, carried via apply_scalars). The decoder/vit twins proved this routing.
            // muon/mamba is a SINGLE forward + NS precompute (NOT a 2nd forward), so it does
            // NOT hit the shared mamba-forward A/A/A race that blocks the SAM-2nd-pass mamba
            // cells — VERIFIED A/A/A bit-exact by test_l3tc_tail_gate (muon/mamba).
            return launch_fused_mamba_megakernel_tc<OptId::Muon>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok11:
            // MODEL-COUPLED SAM 2nd backward + per-tensor meta-net mu/gate (mamba SAM-tier
            // lane): the SAME single persistent launch. The SAM 2nd backward (P2.4,
            // sharpness=(g_sam−g)²) + the per-tensor mu precompute (P3, mu=rescale·
            // phi(g,sharpness)) + the per-tensor cosine gate are ALL in-kernel phases; the
            // apply tail (sg11_sweep_b_step: smart_grad=g+(1−gate)·alpha·mu, AdamW) runs in
            // P3. The SAM scratch reuses the looksam buffers (mb_tc_looksam_floats);
            // sharpness + the phi pack ride the extended state buffer (bound above). The SAM
            // double-forward is now A/A/A bit-exact on mamba (the register-pressure wgmma-
            // accumulator-spill race is fixed; looksam/mamba proved it). NOT a separate launch.
            return launch_fused_mamba_megakernel_tc<OptId::SuperGrok11>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok15:
            // Same SAM 2nd backward + mu precompute as SG11, but SIMPLER tail: the gate is a
            // host scalar (st.gate = sigmoid(accuracy)), NO per-tensor cosine stage; the apply
            // tail (sg15_sweep_b_step) does the per-coord alpha clip + smart_grad = g +
            // gate·a·mu + AdamW. Single persistent launch.
            return launch_fused_mamba_megakernel_tc<OptId::SuperGrok15>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;  // STAGED/coupled opt not single-launch TC
    }
}

// mega_mamba_sg2_tc — the DEDICATED SuperGrok2 mamba TC launcher (the mamba twin of
// mega_{decoder,vit}_sg2_tc). SG2 needs the FULL CSA/HCA/PEER/GRU meta-net weight bundle
// (26 HBM pointers, model-independent) + per-tensor scalar ARRAYS (length P) + the
// meta-net state (m/v/mu/slow/gru_state/sharpness), none of which fit the FusedScalars
// POD or the generic mega_mamba_real_adamw_tc boundary. So this is a PARALLEL entry (the
// generic launcher + the byte-identical cells are UNTOUCHED). It binds the SG2 state
// slices, threads the weight bundle + scalar arrays through FusedOptState's sg2_* fields,
// and dispatches the OptId::SuperGrok2 kernel instantiation (its P3-SG2 phase runs
// sg2_meta_stages per tensor: STAGE -1 in-kernel segmented sort → CSA/HCA/GRU/PEER/apply,
// reading st.sharpness from the SAM 2nd backward (P2.4, shared with SG11/15)).
//
// STATE LAYOUT (host sizes it; this carves it):
//   [m(total) | v(total) | mu(total) | loss(1) | sharpness(total) | slow(total)
//    | gru_state(total*GH)]   ⇒ min_state = (4+1+GH)*total + 1   (GH=gru_hidden=4)
// perm/unsort are built IN-KERNEL into the per-CTA workspace (NOT state).
cudaError_t mega_mamba_sg2_tc(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* loss_out,
        // SG2 meta-net weight bundle (HBM, fp32, model-independent), in SG2Weights order.
        const float* input_proj_W, const float* input_proj_b,
        const float* csa_q_W, const float* csa_k_W, const float* csa_v_W, const float* csa_out_W,
        const float* csa_compress_w, const float* csa_idx_DQ, const float* csa_idx_K,
        const float* hca_q_W, const float* hca_k_W, const float* hca_v_W, const float* hca_out_W,
        const float* gru_Wz, const float* gru_bz, const float* gru_Wr, const float* gru_br,
        const float* gru_Wh, const float* gru_bh,
        const float* peer_query_Ws, const float* prod_keys_A, const float* prod_keys_B,
        const float* expert_W1, const float* expert_b1, const float* expert_W2, const float* expert_b2,
        // per-tensor scalar arrays (device, length kMambaNumTensors).
        const float* sc_alpha, const float* sc_gru_decay, const float* sc_lamb_eff,
        const float* sc_beta1, const float* sc_bc1, const float* sc_bc2,
        // shared scalars + the SAM 2nd-backward gate (SG2 reuses the LookSAM P2.4 machinery).
        float rescale, float beta2, float lr, float wd, float eps,
        float rho, float sam_on,
        int step, cudaStream_t stream, int ncta_cap) {
    const int64_t total = kMambaTotalElems;
    const int T = B * mb::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    // Size the workspace for the EXACT count the kernel launcher runs (occ·n_sms with the
    // occupancy-fill default; reg-capped uniform across OptIds). Falls back to n_sms.
    int nCTA = mb_tc_launched_nctas<OptId::SuperGrok2>(dev, ncta_cap);
    if (nCTA <= 0) { nCTA = n_sms; if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap; }

    const int64_t need = mb_tc_workspace_floats(T, nCTA);
    MbTcLauncherScratch& sc = mb_tc_launcher_scratch(dev, need);
    ctx.g_next_task  = sc.g_next;
    ctx.g_arrived    = sc.g_arrived;
    ctx.g_generation = sc.g_generation;
    ctx.n_tasks      = kMambaNumTensors;

    // Bind the SG2 state slices: [m | v | mu | loss | sharpness | slow | gru_state].
    FusedOptState st;
    st.exp_avg     = state;                       // m
    st.exp_avg_sq  = state + total;               // v
    st.mu          = state + 2 * total;           // expert-output EMA
    // loss_out points at state + 3*total (dispatch's loss slot).
    float* sharp_base = loss_out + 1;             // state + 3*total + 1
    st.sharpness   = sharp_base;                  // (g_sam − g)² (SAM 2nd backward, P2.4)
    st.sg2_slow    = sharp_base + total;          // grokfast slow-grad EMA
    st.sg2_gru_state = st.sg2_slow + total;       // [total*GH] per-element GRU state
    // meta-net weight bundle (HBM).
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
    // per-tensor scalar arrays.
    st.sg2_alpha = sc_alpha; st.sg2_gru_decay = sc_gru_decay; st.sg2_lamb_eff = sc_lamb_eff;
    st.sg2_beta1 = sc_beta1; st.sg2_bc1 = sc_bc1; st.sg2_bc2 = sc_bc2;
    st.sg2_rescale = rescale;
    // shared scalars the SG2 phase reads (beta2/lr/wd/eps via FusedScalars fields).
    st.beta2 = beta2; st.lr = lr; st.wd = wd; st.eps = eps;
    // SAM 2nd-backward gate: SG2 reuses the LookSAM P2.4 machinery. st.rho is the
    // perturbation radius (SG2's sam_rho); st.looksam_sam is the every-k SAM-step gate
    // (1.0 ⇒ run the in-kernel perturb→2nd fwd+bwd→sharpness=(g_sam−g)² this step; 0.0 ⇒
    // reuse the cached sharpness). On a non-SAM step the cached sharpness feeds verbatim.
    st.rho = rho;
    st.looksam_sam = sam_on;

    MambaTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = sc.workspace;
    tok.loss_out  = loss_out;

    return launch_fused_mamba_megakernel_tc<OptId::SuperGrok2>(
        ctx, params, tok, grad, lr, step, st, stream, nCTA);
}

}}}  // namespace sg::fused::sm90
