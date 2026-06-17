// csrc/fused/sm_90/mega_decoder_real_adamw_tc.cu — R2.3 TENSOR-CORE cell driver
// TU (DESIGN-TC-PIPELINE.md Fork B). The bf16 wgmma fwd+bwd+AdamW persistent
// megakernel, compiled with -DSG_TUNED_GEMM_IMPL=1 so the wgmma branch of
// fused_decoder_megakernel.cuh is selected (the scalar default TU
// mega_decoder_real_adamw.cu is UNTOUCHED and ships as the live path).
//
// BUILD-ONLY extension of the build path (the R2.3 deliverable (2)): this is a
// SECOND compiled TU for the SAME (decoder × adamw) cell, selected by the GEMM
// impl token. It is NOT wired into compile.py / dispatch — the parity gate
// (tests/hw/test_decoder_tc.py PART 2) JIT-loads it via cpp_extension.load and
// drives it directly through the pybind entry below (mirroring the scalar
// dispatch.cpp:630-651 launch). No setup.py glob change, no dispatch routing.
//
// The pybind entry `tc_train_step` runs ONE Fork-B TC step over a flat fp32
// param blob in place, returning (mean loss, reduced weight grad [total]) so the
// gate can compare every grad per-tensor vs the bf16-rounded fp64 oracle (the
// keystone validation — the CPU mirror does NOT cover tile-batched P1).
//
// HONESTY: this exercises the REAL tensor-core path (HGMMA in SASS, verified by
// the SASS audit in test_decoder_tc.py). No scalar fallback, no surrogate.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"

namespace dec = ::sg::fused::sm90::dec;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::DecoderTokenCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kDecTotalElems;

// One persistent set of barrier/queue counters + the TC workspace, cached per
// device (the kernel zeroes the barrier per launch; the workspace is sized for
// the TC layout, NOT the scalar 223 MB partials). Re-created if a larger B
// requires a larger workspace.
struct TcScratch {
    torch::Tensor g_next;       // int [1]
    torch::Tensor g_arrived;    // int [1]
    torch::Tensor g_generation; // int [1]
    torch::Tensor workspace;    // float [dec_tc_workspace_floats(T,B,nCTA)]
    int n_ctas = 0;
    int64_t ws_floats = 0;
};

// Effective nCTA the launcher will use (must match for workspace + ownership).
static int tc_effective_nctas(const torch::Tensor& params, int ncta_cap) {
    int dev = params.get_device();
    int n_sms = 1;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    int n = n_sms;
    if (ncta_cap > 0 && ncta_cap < n) n = ncta_cap;
    return n;
}

static TcScratch& tc_scratch_for(const torch::Tensor& params, int B, int nCTA) {
    static TcScratch s;
    const int T = B * dec::kSeq;
    const int64_t need = ::sg::fused::sm90::dec_tc_workspace_floats(T, B, nCTA);
    if (!s.g_next.defined()) {
        auto opt_i = torch::dtype(torch::kInt32).device(params.device());
        s.g_next = torch::zeros({1}, opt_i);
        s.g_arrived = torch::zeros({1}, opt_i);
        s.g_generation = torch::zeros({1}, opt_i);
    }
    s.n_ctas = nCTA;
    if (s.ws_floats < need) {
        auto opt_f = torch::dtype(torch::kFloat32).device(params.device());
        s.workspace = torch::empty({need}, opt_f);
        s.ws_floats = need;
    }
    return s;
}

// tc_train_step: one Fork-B TC step. params [total] fp32 (updated in place);
// tokens [B,kSeq] int32; targets [B] int32; state [3*total] fp32 ([m|v|extra]).
// Returns {loss (cpu float scalar tensor), grad [total] fp32 (the reduced grad)}.
static std::vector<torch::Tensor> tc_train_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems, "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B %% 16 == 0 (dW K-loop is 16-step atoms)");
    const int64_t total = kDecTotalElems;
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= 3 * total, "state must be contiguous fp32 [>=3*total]");

    const int nCTA = tc_effective_nctas(params, (int)ncta_cap);
    TcScratch& sc = tc_scratch_for(params, B, nCTA);
    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));
    auto loss_dev = torch::zeros({1}, torch::dtype(torch::kFloat32).device(params.device()));

    PersistentContext ctx{
        sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
        ::sg::fused::sm90::kDecNumTensors,
        0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    DecoderTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_decoder_megakernel_tc<OptId::AdamW>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return {loss_dev.to(torch::kCPU), grad};
}

// CALIBRATION HOOK (test_decoder_tc.py::test_tc_dw_gemm_exact_on_own_operands):
// slice the kernel's OWN stored bf16 acts dY_ff2[L1] [T,d] and X_gact[L1] [T,dff]
// from the workspace, returned as fp32 CPU tensors. The gate contracts them in
// fp32 ascending-t and compares to the kernel's ff2.weight grad slice — isolating
// the dW GEMM (incl. the ONLY Kin=dff multi-N-tile path, untested by the 13/13
// engine micro-gates) from the operand-chain bf16 divergence. A ~1e-6 match
// proves the dW GEMM is bit-exact on its own operands, calibrating the per-tensor
// bf16 tolerance as headroom over a GEMM-exact floor. Re-uses the cached
// workspace from the last tc_train_step (call AFTER it). Read-only; ships in the
// test TU only (never globbed into the .so).
static std::vector<torch::Tensor> tc_dump_ff2_operands(torch::Tensor params, int64_t B) {
    const int nCTA = tc_effective_nctas(params, 0);  // unused for offsets; acts layout is nCTA-independent
    (void)nCTA;
    TcScratch& sc = tc_scratch_for(params, (int)B, tc_effective_nctas(params, 4));
    const int T = (int)B * dec::kSeq;
    const int64_t d = dec::kD, dff = dec::kDff;
    auto acts_bf16 = sc.workspace.slice(0, 0, ::sg::fused::sm90::dec_tc_acts_floats(T, (int)B))
                         .view(torch::kBFloat16);
    // mirror dec_acts_bind: walk to dY_ff2[1] and X_gact[1].
    const int64_t Td = (int64_t)T * d, T3d = (int64_t)T * 3 * d, Tff = (int64_t)T * dff;
    int64_t off = 0; int64_t off_Xgact1 = -1, off_dYff2_1 = -1;
    for (int li = 0; li < dec::kLayers; ++li) {
        off += Td;          // X_in
        off += Td;          // X_ctx
        off += Td;          // X_x1
        if (li == 1) off_Xgact1 = off;
        off += Tff;         // X_gact
        off += T3d;         // dY_qkv
        off += Td;          // dY_a
        off += Tff;         // dY_ff0
        if (li == 1) off_dYff2_1 = off;
        off += Td;          // dY_ff2
    }
    auto Xgact1 = acts_bf16.slice(0, off_Xgact1, off_Xgact1 + Tff).to(torch::kFloat32).to(torch::kCPU);
    auto dYff2_1 = acts_bf16.slice(0, off_dYff2_1, off_dYff2_1 + Td).to(torch::kFloat32).to(torch::kCPU);
    return {dYff2_1, Xgact1};  // [T*d], [T*dff]
}

// scalar_train_step: the SAME (decoder × AdamW) cell run through the SCALAR
// default launcher (launch_fused_decoder_megakernel<AdamW>, which lives OUTSIDE
// the wgmma guard at fused_decoder_megakernel.cuh — so this wgmma-token TU
// can still call it). This exists ONLY so the step-time gate can time the live
// scalar path and the TC path BACK-TO-BACK in one process / one contention
// regime → an honest scalar:TC ratio. The scalar path uses its OWN workspace
// layout (nCTA*total grad partials + nCTA loss slots + 1 reduced loss), sized
// for nCTA = #SMs (the scalar launcher pins one CTA per SM, no cap). It is NOT
// the shipped invocation (that is dispatch.cpp via mega_decoder_real_adamw.cu);
// it is a measurement mirror. Returns {loss, reduced grad} like tc_train_step.
//
// GATED by SG_DEC_SCALAR_MEGAKERNEL: the legacy fp32 scalar megakernel does not
// fit smem at scaled SG_DEC_D (see the flag note in fused_decoder_megakernel.cuh).
// When OFF, this back-to-back timing mirror is compiled out (its only purpose is
// the scalar:TC ratio, which is moot when the scalar path is unavailable); the TC
// gates (tc_train_step) are unaffected.
#if SG_DEC_SCALAR_MEGAKERNEL
static std::vector<torch::Tensor> scalar_train_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems, "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    const int64_t total = kDecTotalElems;
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= 3 * total, "state must be contiguous fp32 [>=3*total]");

    int dev = params.get_device();
    int n_sms = 1;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);

    // Separate cached scalar scratch (DO NOT reuse the TC workspace — different
    // layout/size; the TC one under-allocates the scalar nCTA*total partials).
    static torch::Tensor s_gn, s_ga, s_gg, s_ws;
    static int64_t s_ws_floats = 0;
    const int64_t need = (int64_t)n_sms * total + n_sms + 1;
    if (!s_gn.defined()) {
        auto oi = torch::dtype(torch::kInt32).device(params.device());
        s_gn = torch::zeros({1}, oi); s_ga = torch::zeros({1}, oi); s_gg = torch::zeros({1}, oi);
    }
    if (s_ws_floats < need) {
        s_ws = torch::empty({need}, torch::dtype(torch::kFloat32).device(params.device()));
        s_ws_floats = need;
    }
    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));
    float* loss_out = s_ws.data_ptr<float>() + (int64_t)n_sms * total + n_sms;

    PersistentContext ctx{
        s_gn.data_ptr<int>(),
        reinterpret_cast<unsigned*>(s_ga.data_ptr<int>()),
        reinterpret_cast<unsigned*>(s_gg.data_ptr<int>()),
        ::sg::fused::sm90::kDecNumTensors, 0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m; st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    DecoderTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = s_ws.data_ptr<float>();
    tok.loss_out = loss_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_decoder_megakernel<OptId::AdamW>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream);
    TORCH_CHECK(err == cudaSuccess, "scalar megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto loss_cpu = s_ws.slice(0, (int64_t)n_sms * total + n_sms, (int64_t)n_sms * total + n_sms + 1)
                        .to(torch::kCPU);
    return {loss_cpu, grad};
}
#endif  // SG_DEC_SCALAR_MEGAKERNEL

#ifdef SG_DEC_PROFILE
// Diagnostic-only (SG_DEC_PROFILE; never shipped — the production _ops never sets
// this flag). Read + reset the per-phase clock64 maxima (cycles), 8 slots:
// [0]=P1 fwd, [1]=P1 bwd, [2]=B1 wait, [3]=P2 dW-GEMM, [4]=P2 grad-assembly,
// [5]=P3 opt tail, [6]=B2 wait, [7]=B0 wait. Call AFTER one tc_train_step; divide
// by the SM clock (~1.99 GHz boost on H100) to get ms, or read the RATIOS.
static std::vector<int64_t> tc_profile_read() {
    unsigned long long h[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cudaMemcpyFromSymbol(h, ::sg::fused::sm90::g_dec_prof_max, sizeof(h));
    unsigned long long z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(::sg::fused::sm90::g_dec_prof_max, z, sizeof(z));
    std::vector<int64_t> out(8);
    for (int i = 0; i < 8; ++i) out[i] = (int64_t)h[i];
    return out;
}

#if SG_DEC_PROFILE_FWD_FINE
// Parallel reader for the FINE fwd/dX sub-phase counters (SG_DEC_PROFILE_FWD_FINE;
// only compiled when BOTH SG_DEC_PROFILE and SG_DEC_PROFILE_FWD_FINE are set).
// Read + reset g_dec_prof_fwd_fine [kDecFwdFineSlots] (cycles). Layout is
// [phase*kDecFwdFineSub + sub]: phase 0 = fwd ring, 1 = dX ring; sub ∈
// {0 ISSUE (cp.async LDGSTS issue), 1 WAIT (cp.async drain — the DRAIN/latency
// cost), 2 WGMMA (mma issue+commit+wait), 3 EPI (epilogue store), 4 BARRIER
// (fence+__syncthreads publish)}. Call AFTER one tc_train_step; divide by the SM
// clock (~1.98 GHz on H100) for ms, or read the RATIOS to localize whether P1_fwd
// / P1_bwd is DRAIN-bound (WAIT dominates → the deeper-ring lever helps) vs
// COMPUTE/EPILOGUE-bound (WGMMA/EPI dominate → it won't). Returns the flat slots
// + the (phases, sub) shape so the main loop can label them.
static std::vector<int64_t> tc_profile_read_fwd_fine() {
    constexpr int kSlots = ::sg::fused::sm90::dectc::kDecFwdFineSlots;
    unsigned long long h[kSlots];
    for (int i = 0; i < kSlots; ++i) h[i] = 0;
    cudaMemcpyFromSymbol(h, ::sg::fused::sm90::dectc::g_dec_prof_fwd_fine, sizeof(h));
    unsigned long long z[kSlots];
    for (int i = 0; i < kSlots; ++i) z[i] = 0;
    cudaMemcpyToSymbol(::sg::fused::sm90::dectc::g_dec_prof_fwd_fine, z, sizeof(z));
    std::vector<int64_t> out(kSlots);
    for (int i = 0; i < kSlots; ++i) out[i] = (int64_t)h[i];
    return out;
}
#endif
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step", &tc_train_step,
           "Fork-B tensor-core decoder fwd+bwd+AdamW step (in place); returns (loss, reduced grad)",
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 0);
    mm.def("tc_dump_ff2_operands", &tc_dump_ff2_operands,
           "gate-only: dump the kernel's stored dY_ff2[L1], X_gact[L1] (fp32 CPU)");
#if SG_DEC_SCALAR_MEGAKERNEL
    mm.def("scalar_train_step", &scalar_train_step,
           "gate-only: SAME cell via the SCALAR launcher (back-to-back step-time mirror)",
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"));
#endif
    mm.attr("TILE_M") = (int)::sg::fused::sm90::dectc::kTileM;
    mm.attr("TILE_N") = (int)SG_TUNED_TILE_N;
    mm.attr("TOTAL") = (int)kDecTotalElems;
    mm.attr("D") = (int)::sg::fused::sm90::SG_DEC_D;          // model width (128 prod / 1024 bench)
    mm.attr("DFF") = (int)::sg::fused::sm90::SG_DEC_DFF;
    mm.attr("LAYERS") = (int)::sg::fused::sm90::SG_DEC_LAYERS;
    mm.attr("SEQ") = (int)::sg::fused::sm90::SG_DEC_SEQ;
    mm.attr("VOCAB") = (int)::sg::fused::sm90::SG_DEC_VOCAB;
#ifdef SG_DEC_PROFILE
    mm.def("tc_profile_read", &tc_profile_read,
           "diagnostic-only: per-phase clock64 maxima "
           "[P1fwd,P1bwd,B1wait,P2dW,P2asm,P3opt,B2wait,B0wait] (cycles), resets after read");
    mm.attr("HAS_PROFILE") = true;
#if SG_DEC_PROFILE_FWD_FINE
    mm.def("tc_profile_read_fwd_fine", &tc_profile_read_fwd_fine,
           "diagnostic-only: FINE fwd/dX sub-phase clock64 maxima, flat "
           "[phase*FWD_FINE_SUB + sub]; phase 0=fwd ring, 1=dX ring; "
           "sub {0 ISSUE,1 WAIT(drain),2 WGMMA,3 EPI,4 BARRIER} (cycles), resets after read");
    mm.attr("HAS_FWD_FINE") = true;
    mm.attr("FWD_FINE_SUB") = (int)::sg::fused::sm90::dectc::kDecFwdFineSub;       // 5
    mm.attr("FWD_FINE_PHASES") = (int)::sg::fused::sm90::dectc::kDecFwdFinePhases; // 2
#else
    mm.attr("HAS_FWD_FINE") = false;
#endif
#else
    mm.attr("HAS_PROFILE") = false;
    mm.attr("HAS_FWD_FINE") = false;
#endif
    // Expose the fwd/dX deeper-ring knobs so the bench/main-loop can report them.
    mm.attr("FWD_PIPE") = (int)::sg::fused::sm90::dectc::kDecFwdPipe;
    mm.attr("FWD_STAGES") = (int)::sg::fused::sm90::dectc::kDecFwdStages;
}
