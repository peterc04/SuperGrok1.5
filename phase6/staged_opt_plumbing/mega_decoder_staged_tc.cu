// SCRATCHPAD JIT DRIVER (NOT a committed-source edit): extends
// mega_decoder_multiopt_tc.cu's OptId-generic TC step to the STAGED optimizers
// Prodigy(5)/Muon(7)/LookSAM(4)/SuperGrok11(8)/SuperGrok15(9) AND SuperGrok2(10),
// mirroring mega_decoder_real_adamw_tc_launcher.cu's opt_id dispatch + state-binding
// block (csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu:167-451) EXACTLY.
//
// Build-via-include ONLY (NO committed-source edit): compiled against the flagship
// layout with the SAME -D flags + -include decoder_flagship_layout.cuh as
// flagship_train.py, EXCEPT it does NOT set -DSG_DEC_BENCH_LAYOUT=1 — so the live
// kDecStagedOptScratch gate (fused_decoder_megakernel.cuh:541-545) is TRUE and the
// four staged-opt scratch regions (Prodigy reduce | Muon NS | LookSAM 2nd-bwd |
// SuperGrok2 meta-net) are carved into the workspace, exactly as the production
// opt-agnostic launcher needs. The 5 elementwise opts also run through this TU
// (their P3 never touches the staged regions ⇒ byte-identical math).
//
// STATE LAYOUT (host sizes it; this binds it), mirroring the launcher:
//   generic opts:  [m | v | extra | loss | param_init(total) | prodigy_persist(3) ...]
//                  + SG11/15 [sharpness(total) | phi_pack(kSgPhiPackFloats)]
//   SG2:           [m | v | mu | loss | sharpness(total) | slow(total) | gru_state(total*GH)]
// The host (flagship_staged_run.py) allocates state >= the largest layout (SG2's
// (4+1+GH)*total + 1 for SG2; (4+1)*total + 1 + kSgPhiPackFloats for SG11/15) and
// zero-inits it; param_init is seeded = params at step 1 by the host (Prodigy anchor).
//
// SG2 device packs: the 26 model-INDEPENDENT meta-net weight buffers + the 6
// per-tensor scalar arrays (length kDecNumTensors) are allocated + filled host-side
// (flagship_staged_run.py builds them from CSAHCAMetaNet.get_weights) and bound here
// through FusedOptState's sg2_* fields, exactly as mega_decoder_sg2_tc does.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"

namespace dec = ::sg::fused::sm90::dec;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::DecoderTokenCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::FusedScalars;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kDecTotalElems;
using ::sg::fused::sm90::kDecNumTensors;
using ::sg::fused::sm90::apply_scalars;
using ::sg::fused::sm90::kPsiW1Off;
using ::sg::fused::sm90::kPsiB1Off;
using ::sg::fused::sm90::kPsiW2Off;
using ::sg::fused::sm90::kSgPhiW1Off;
using ::sg::fused::sm90::kSgPhiB1Off;
using ::sg::fused::sm90::kSgPhiW2Off;
using ::sg::fused::sm90::kSgPhiPackFloats;

struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace, grad;
    int n_ctas = 0; int64_t ws_floats = 0;
};

static int tc_effective_nctas(const torch::Tensor& params, int ncta_cap) {
    int dev = params.get_device(); int n_sms = 1;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    int n = n_sms;
    if (ncta_cap > 0 && ncta_cap < n) n = ncta_cap;
    return n;
}

static TcScratch& tc_scratch_for(const torch::Tensor& params, int B, int nCTA, bool need_grad) {
    static TcScratch s;
    const int T = B * dec::kSeq;
    const int64_t need = ::sg::fused::sm90::dec_tc_workspace_floats(T, B, nCTA);
    if (!s.g_next.defined()) {
        auto oi = torch::dtype(torch::kInt32).device(params.device());
        s.g_next = torch::zeros({1}, oi);
        s.g_arrived = torch::zeros({1}, oi);
        s.g_generation = torch::zeros({1}, oi);
    }
    s.n_ctas = nCTA;
    if (s.ws_floats < need) {
        s.workspace = torch::empty({need}, torch::dtype(torch::kFloat32).device(params.device()));
        s.ws_floats = need;
    }
    // PERSISTENT grad buffer (one per process, zeroed in-place each step) — avoids a
    // per-step 5.5 GB alloc/free churn that fragments + OOMs at flagship width. The
    // SG2 path passes need_grad=false: it aliases grad onto the dead LookSAM workspace
    // region (see sg2_train_step) to reclaim the 5.5 GiB and FIT at ncta=1.
    if (need_grad) {
        if (!s.grad.defined()) {
            s.grad = torch::zeros({kDecTotalElems}, torch::dtype(torch::kFloat32).device(params.device()));
        } else {
            s.grad.zero_();
        }
    }
    return s;
}

// Dispatch one TC step for the requested OptId. Mirrors the launcher's switch
// (mega_decoder_real_adamw_tc_launcher.cu:231-327) for the generic (NON-SG2) arms:
// the 5 elementwise + Prodigy/Muon/LookSAM/SG11/SG15. SG2 has its own entry below.
static cudaError_t launch_opt(int opt_id, PersistentContext ctx, float* params,
                              DecoderTokenCtx tok, float* grad, float lr, int step,
                              FusedOptState st, cudaStream_t stream, int nCTA) {
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::Lion>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::Grokfast>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Prodigy:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::Prodigy>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Muon:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::Muon>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::LookSAM:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::LookSAM>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok11:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::SuperGrok11>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::SuperGrok15:
            return ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::SuperGrok15>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;   // SG2 has its own entry (sg2_train_step)
    }
}

// ── Generic OptId TC step (opts 0-9). Binds the FULL state layout the launcher's
//    state-binding block binds (mega_decoder_real_adamw_tc_launcher.cu:167-215):
//    [m | v | extra] + loss(state+3*total) + Prodigy param_init/persist + SG11/15
//    sharpness/phi pack. loss rides state[3*total] (the launcher's loss_slot).
static std::vector<torch::Tensor> tc_train_step_opt(
        int64_t opt_id,
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        double alpha, double lamb, double gamma,
        double rho, double looksam_sam, double sg_rescale, double gate, double gate_temp,
        double d0, double d_coef, double beta3, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems, "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B % 16 == 0");
    const int64_t total = kDecTotalElems;
    // SG11/15 need [m|v|extra|loss|sharpness(total)|phi]; Prodigy [m|v|extra|loss|param_init(total)|persist(3)].
    // The host sizes state >= 4*total + 1 + max(total+1?, kSgPhiPackFloats, 3). We assert the SG11/15 worst case.
    const int64_t min_state = 4 * total + 1 + total + (int64_t)kSgPhiPackFloats;
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= min_state,
                "state must be contiguous fp32 [>=", min_state, "] (got ", state.numel(), ")");

    const int nCTA = tc_effective_nctas(params, (int)ncta_cap);
    TcScratch& sc = tc_scratch_for(params, B, nCTA, /*need_grad=*/true);
    auto& grad = sc.grad;

    PersistentContext ctx{
        sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
        kDecNumTensors, 0u};

    // Bind [m | v | extra] over the SAME state buffer (launcher :170-176). loss_out
    // points at state + 3*total (the launcher's loss_slot); the extended slots follow.
    float* m = state.data_ptr<float>();
    float* const extra_slice = m + 2 * total;
    float* const loss_out    = m + 3 * total;            // state + 3*total (loss slot)
    FusedOptState st;
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.ema = extra_slice;                       // grokfast/grokadamw slow-grad EMA
    st.psi_W1 = extra_slice + kPsiW1Off;        // NeuralGrok psi-net pack head
    st.psi_b1 = extra_slice + kPsiB1Off;
    st.psi_W2 = extra_slice + kPsiW2Off;
    // Prodigy state bindings (launcher :183-193): s_track aliases extra; param_init
    // follows loss; the 3 persisted estimator scalars [r_ema|s_ema|d_lr] follow it.
    st.s_track        = extra_slice;
    st.param_init     = loss_out + 1;                    // state + 3*total + 1
    st.prodigy_persist = loss_out + 1 + total;           // [r_ema | s_ema | d_lr]
    // LookSAM persistent SAM direction (launcher :194-198) aliases the extra slice.
    st.sam_dir        = extra_slice;
    // SG11/15 (launcher :199-215): mu aliases extra; sharpness = loss+1; phi pack follows.
    st.mu             = extra_slice;
    {
        float* sharp_base = loss_out + 1;                // state + 3*total + 1
        float* phi_base   = sharp_base + total;          // phi pack after sharpness
        st.sharpness = sharp_base;
        st.sg_phi_W1 = phi_base + kSgPhiW1Off;
        st.sg_phi_b1 = phi_base + kSgPhiB1Off;
        st.sg_phi_W2 = phi_base + kSgPhiW2Off;           // sg_phi_b2 read on-device from [H]
    }

    // LIVE scalars (launcher's full set via apply_scalars).
    FusedScalars scal;
    scal.lr = (float)lr; scal.beta1 = (float)beta1; scal.beta2 = (float)beta2;
    scal.eps = (float)eps; scal.wd = (float)weight_decay;
    scal.bc1 = (float)bc1; scal.bc2 = (float)bc2;
    scal.alpha = (float)alpha; scal.lamb = (float)lamb; scal.gamma = (float)gamma;
    scal.rho = (float)rho; scal.looksam_sam = (float)looksam_sam;
    scal.sg_rescale = (float)sg_rescale; scal.gate = (float)gate; scal.gate_temp = (float)gate_temp;
    scal.d0 = (float)d0; scal.d_coef = (float)d_coef; scal.beta3 = (float)beta3;
    apply_scalars(st, scal);
    st.lr = (float)lr;

    DecoderTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = launch_opt((int)opt_id, ctx, params.data_ptr<float>(), tok,
                                 grad.data_ptr<float>(), (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC megakernel launch failed (opt_id=", opt_id, "): ",
                cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    // loss rides state[3*total]; return a host copy of it.
    auto loss_dev = state.narrow(0, 3 * total, 1);
    return {loss_dev.to(torch::kCPU), grad};
}

// ── SuperGrok2 dedicated TC step. Mirrors mega_decoder_sg2_tc
//    (mega_decoder_real_adamw_tc_launcher.cu:363-451) EXACTLY: state layout
//    [m | v | mu | loss | sharpness(total) | slow(total) | gru_state(total*GH)],
//    the 26 HBM meta-net weight packs + the 6 per-tensor scalar arrays, dispatched
//    to launch_fused_decoder_megakernel_tc<OptId::SuperGrok2>. ncta_cap is forwarded;
//    SG2's per-CTA meta-net workspace is O(Nmax) ⇒ run at ncta=1 (the resource
//    planner's verdict for the flagship width — full occupancy OOMs the H100).
static std::vector<torch::Tensor> sg2_train_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state,
        // 26 meta-net weight packs (HBM, fp32, model-independent), SG2Weights order.
        torch::Tensor input_proj_W, torch::Tensor input_proj_b,
        torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
        torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
        torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
        torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
        torch::Tensor gru_Wh, torch::Tensor gru_bh,
        torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
        torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
        // 6 per-tensor scalar arrays (device, length kDecNumTensors).
        torch::Tensor sc_alpha, torch::Tensor sc_gru_decay, torch::Tensor sc_lamb_eff,
        torch::Tensor sc_beta1, torch::Tensor sc_bc1, torch::Tensor sc_bc2,
        // shared scalars + SAM gate.
        double rescale, double beta2, double lr, double wd, double eps,
        double rho, double sam_on, int64_t step, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems, "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B % 16 == 0");
    const int64_t total = kDecTotalElems;
    const int GH = ::sg::fused::sm90::DecSG2Dims::gru_hidden;
    // SG2 state = [m | v | mu | loss(1) | sharpness(total) | gru_state(total*GH)] — the
    // `slow` plane is aliased onto the dead LookSAM sam_grad workspace region (below),
    // so state is (3 + 1 + GH)*total + 1 (ONE plane smaller than the 9-plane layout).
    const int64_t min_state = (int64_t)(3 + 1 + GH) * total + 1;
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= min_state,
                "SG2 state must be contiguous fp32 [>=", min_state, "] (got ", state.numel(), ")");
    TORCH_CHECK(sc_alpha.numel() >= kDecNumTensors, "sc_alpha must be length >= kDecNumTensors");

    const int nCTA = tc_effective_nctas(params, (int)ncta_cap);
    const int T = B * dec::kSeq;
    TcScratch& sc = tc_scratch_for(params, B, nCTA, /*need_grad=*/false);
    // MEMORY: SG2's 9-plane state (49.5 GiB at flagship) + params + the opt-agnostic
    // workspace (which carries the 11 GiB LookSAM 2*total scratch that SG2 NEVER
    // touches — its phase is if-constexpr'd to OptId::LookSAM) marginally exceed the
    // 80 GiB H100 when a SEPARATE 5.5 GiB grad is also allocated. Since the workspace's
    // LookSAM region (dec_tc_looksam_floats == 2*total) is DEAD for SG2, we point the
    // SG2 grad at the HEAD of that region (offset computed from the SAME public header
    // carve-order helpers dec_tc_workspace_floats uses). The kernel's grad-reduce
    // writes there; the SG2 phase reads it; the LookSAM phase that would write it is
    // never instantiated for OptId::SuperGrok2 → no aliasing hazard. This reclaims the
    // 5.5 GiB and makes SG2 FIT at ncta=1.
    float* ws = sc.workspace.data_ptr<float>();
    const int64_t looksam_off =
          ::sg::fused::sm90::dec_tc_acts_floats(T, B)
        + (int64_t)nCTA * ::sg::fused::sm90::dectc::dec_tile_scratch_total_f32()
        + (int64_t)nCTA * ::sg::fused::sm90::dectc::kLnVecElems
        + nCTA + 1
        + ::sg::fused::sm90::dec_tc_dw_part_floats()
        + ::sg::fused::sm90::dec_tc_opt_reduce_floats(nCTA)
        + ::sg::fused::sm90::dec_tc_muon_floats(nCTA);   // LookSAM region starts here
    TORCH_CHECK(::sg::fused::sm90::dec_tc_looksam_floats() >= total,
                "SG2 grad-alias: LookSAM region (", ::sg::fused::sm90::dec_tc_looksam_floats(),
                ") smaller than total (", total, ")");
    float* grad_ptr = ws + looksam_off;          // alias grad onto the dead LookSAM scratch

    PersistentContext ctx{
        sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
        kDecNumTensors, 0u};

    float* m = state.data_ptr<float>();
    float* const loss_out = m + 3 * total;       // state + 3*total (loss slot)
    FusedOptState st;
    st.exp_avg     = m;                            // m
    st.exp_avg_sq  = m + total;                    // v
    st.mu          = m + 2 * total;                // expert-output EMA
    float* sharp_base = loss_out + 1;             // state + 3*total + 1
    st.sharpness   = sharp_base;                  // (g_sam − g)² (SAM 2nd backward)
    // MEMORY: alias the `slow` (grokfast EMA) plane onto the DEAD LookSAM `sam_grad`
    // region (looksam_off + total). It persists across steps (the workspace scratch is
    // process-lived) and is NEVER touched by the SG2 phase otherwise, so this reclaims
    // a 5.5 GiB state plane — the last GiB SG2 needs to FIT single-GPU at ncta=1. The
    // state buffer is correspondingly one `total` plane SMALLER ([m|v|mu|loss|sharpness|
    // gru_state]); gru_state follows sharpness directly.
    st.sg2_slow    = ws + looksam_off + total;    // dead sam_grad region (persistent)
    st.sg2_gru_state = sharp_base + total;        // [total*GH] per-element GRU state
    // meta-net weight bundle (HBM).
    st.sg2_input_proj_W = input_proj_W.data_ptr<float>(); st.sg2_input_proj_b = input_proj_b.data_ptr<float>();
    st.sg2_csa_q_W = csa_q_W.data_ptr<float>(); st.sg2_csa_k_W = csa_k_W.data_ptr<float>();
    st.sg2_csa_v_W = csa_v_W.data_ptr<float>(); st.sg2_csa_out_W = csa_out_W.data_ptr<float>();
    st.sg2_csa_compress_w = csa_compress_w.data_ptr<float>();
    st.sg2_csa_idx_DQ = csa_idx_DQ.data_ptr<float>(); st.sg2_csa_idx_K = csa_idx_K.data_ptr<float>();
    st.sg2_hca_q_W = hca_q_W.data_ptr<float>(); st.sg2_hca_k_W = hca_k_W.data_ptr<float>();
    st.sg2_hca_v_W = hca_v_W.data_ptr<float>(); st.sg2_hca_out_W = hca_out_W.data_ptr<float>();
    st.sg2_gru_Wz = gru_Wz.data_ptr<float>(); st.sg2_gru_bz = gru_bz.data_ptr<float>();
    st.sg2_gru_Wr = gru_Wr.data_ptr<float>(); st.sg2_gru_br = gru_br.data_ptr<float>();
    st.sg2_gru_Wh = gru_Wh.data_ptr<float>(); st.sg2_gru_bh = gru_bh.data_ptr<float>();
    st.sg2_peer_query_Ws = peer_query_Ws.data_ptr<float>();
    st.sg2_prod_keys_A = prod_keys_A.data_ptr<float>(); st.sg2_prod_keys_B = prod_keys_B.data_ptr<float>();
    st.sg2_expert_W1 = expert_W1.data_ptr<float>(); st.sg2_expert_b1 = expert_b1.data_ptr<float>();
    st.sg2_expert_W2 = expert_W2.data_ptr<float>(); st.sg2_expert_b2 = expert_b2.data_ptr<float>();
    // per-tensor scalar arrays.
    st.sg2_alpha = sc_alpha.data_ptr<float>(); st.sg2_gru_decay = sc_gru_decay.data_ptr<float>();
    st.sg2_lamb_eff = sc_lamb_eff.data_ptr<float>(); st.sg2_beta1 = sc_beta1.data_ptr<float>();
    st.sg2_bc1 = sc_bc1.data_ptr<float>(); st.sg2_bc2 = sc_bc2.data_ptr<float>();
    st.sg2_rescale = (float)rescale;
    // shared scalars the SG2 phase reads.
    st.beta2 = (float)beta2; st.lr = (float)lr; st.wd = (float)wd; st.eps = (float)eps;
    // SAM 2nd-backward gate (SG2 reuses LookSAM P2.4 machinery).
    st.rho = (float)rho; st.looksam_sam = (float)sam_on;

    DecoderTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // Zero the aliased grad region (the kernel accumulates grad partials into it).
    C10_CUDA_CHECK(cudaMemsetAsync(grad_ptr, 0, (size_t)total * sizeof(float), stream));
    // On step 1, cold-init the aliased `slow` plane (sam_grad region) to 0 (the
    // persistent grokfast EMA seed). Steps >1 carry it forward (the workspace is
    // process-lived, so the EMA persists across calls).
    if (step == 1) {
        C10_CUDA_CHECK(cudaMemsetAsync(st.sg2_slow, 0, (size_t)total * sizeof(float), stream));
    }
    cudaError_t err = ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::SuperGrok2>(
        ctx, params.data_ptr<float>(), tok, grad_ptr,
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "SG2 TC megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto loss_dev = state.narrow(0, 3 * total, 1);
    return {loss_dev.to(torch::kCPU)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step_opt", &tc_train_step_opt,
           "OptId-generic TC decoder fwd+bwd+opt step (opts 0-9, incl. staged); returns (loss, grad)",
           pybind11::arg("opt_id"),
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("alpha") = 0.98, pybind11::arg("lamb") = 2.0,
           pybind11::arg("gamma") = 0.0,
           pybind11::arg("rho") = 0.0, pybind11::arg("looksam_sam") = 0.0,
           pybind11::arg("sg_rescale") = 0.0, pybind11::arg("gate") = 1.0,
           pybind11::arg("gate_temp") = 1.0,
           pybind11::arg("d0") = 1e-6, pybind11::arg("d_coef") = 1.0, pybind11::arg("beta3") = 0.0,
           pybind11::arg("ncta_cap") = 0);
    mm.def("sg2_train_step", &sg2_train_step,
           "SuperGrok2 dedicated TC decoder fwd+bwd+meta-net step; returns (loss, grad)",
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"),
           pybind11::arg("input_proj_W"), pybind11::arg("input_proj_b"),
           pybind11::arg("csa_q_W"), pybind11::arg("csa_k_W"), pybind11::arg("csa_v_W"), pybind11::arg("csa_out_W"),
           pybind11::arg("csa_compress_w"), pybind11::arg("csa_idx_DQ"), pybind11::arg("csa_idx_K"),
           pybind11::arg("hca_q_W"), pybind11::arg("hca_k_W"), pybind11::arg("hca_v_W"), pybind11::arg("hca_out_W"),
           pybind11::arg("gru_Wz"), pybind11::arg("gru_bz"), pybind11::arg("gru_Wr"), pybind11::arg("gru_br"),
           pybind11::arg("gru_Wh"), pybind11::arg("gru_bh"),
           pybind11::arg("peer_query_Ws"), pybind11::arg("prod_keys_A"), pybind11::arg("prod_keys_B"),
           pybind11::arg("expert_W1"), pybind11::arg("expert_b1"), pybind11::arg("expert_W2"), pybind11::arg("expert_b2"),
           pybind11::arg("sc_alpha"), pybind11::arg("sc_gru_decay"), pybind11::arg("sc_lamb_eff"),
           pybind11::arg("sc_beta1"), pybind11::arg("sc_bc1"), pybind11::arg("sc_bc2"),
           pybind11::arg("rescale"), pybind11::arg("beta2"), pybind11::arg("lr"),
           pybind11::arg("wd"), pybind11::arg("eps"),
           pybind11::arg("rho"), pybind11::arg("sam_on"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 1);
    mm.attr("TOTAL") = (int64_t)kDecTotalElems;
    mm.attr("NUM_TENSORS") = (int)kDecNumTensors;
    mm.attr("D") = (int)::sg::fused::sm90::SG_DEC_D;
    mm.attr("LAYERS") = (int)::sg::fused::sm90::SG_DEC_LAYERS;
    mm.attr("VOCAB") = (int)::sg::fused::sm90::SG_DEC_VOCAB;
    mm.attr("SEQ") = (int)::sg::fused::sm90::SG_DEC_SEQ;
    mm.attr("SG2_GRU_HIDDEN") = (int)::sg::fused::sm90::DecSG2Dims::gru_hidden;
    mm.attr("SG_PHI_PACK_FLOATS") = (int)kSgPhiPackFloats;
}
