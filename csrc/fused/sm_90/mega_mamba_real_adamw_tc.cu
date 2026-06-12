// csrc/fused/sm_90/mega_mamba_real_adamw_tc.cu — R2 Mamba TENSOR-CORE cell driver
// TU (DESIGN-TC-PIPELINE.md Fork B, the Mamba counterpart of the decoder's
// mega_decoder_real_adamw_tc.cu). The bf16 wgmma fwd+bwd+AdamW persistent
// megakernel, compiled with -DSG_TUNED_GEMM_IMPL=1 so the wgmma branch of
// fused_mamba_megakernel.cuh is selected (the scalar default TU
// mega_mamba_real_adamw.cu is UNTOUCHED and ships as the live path).
//
// BUILD-ONLY second TU for the SAME (mamba × adamw) cell, selected by the GEMM
// impl token. It is NOT wired into compile.py / dispatch — the parity gate
// (tests/hw/test_mamba_tc.py) JIT-loads it via cpp_extension.load and drives it
// directly through the pybind entry below. setup.py AUTO-EXCLUDES it (it owns a
// PYBIND11_MODULE(TORCH_EXTENSION_NAME) → the content-based glob filter drops it).
//
// The pybind entry `tc_train_step` runs ONE Fork-B TC step over a flat fp32 param
// blob in place, returning (mean loss, reduced weight grad [total]) so the gate
// compares every grad per-tensor vs the bf16-rounded fp64 Mamba oracle (the
// keystone — the CPU mirror does NOT cover the tile-batched P1 + the TC dW).
//
// HONESTY: exercises the REAL tensor-core path (HGMMA in SASS). No scalar
// fallback for the 4 projections; the selective scan + conv1d stay scalar (the
// contract — REUSED verbatim from model_stage_mamba3.cuh, not re-implemented).

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"

namespace mb = ::sg::fused::sm90::mb;
namespace mbtc = ::sg::fused::sm90::mbtc;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::MambaTokenCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kMambaTotalElems;

// Persistent barrier/queue counters + the TC workspace, cached per device.
struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace;
    int n_ctas = 0;
    int64_t ws_floats = 0;
};

static int tc_effective_nctas(const torch::Tensor& params, int ncta_cap) {
    int dev = params.get_device();
    // Use the EXACT count the launcher will run (occ·n_sms with occupancy-fill) so
    // the workspace (per-CTA scratch + partials = nCTA·slab) is sized for what runs.
    // occ is register-capped uniform across OptIds by __launch_bounds__(256,2), so
    // AdamW's count == every tail's count. Fall back to n_sms if the attr query fails.
    int n = ::sg::fused::sm90::mb_tc_launched_nctas<::sg::fused::sm90::OptId::AdamW>(dev, ncta_cap);
    if (n <= 0) {
        int n_sms = 1;
        cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
        n = n_sms;
        if (ncta_cap > 0 && ncta_cap < n) n = ncta_cap;
    }
    return n;
}

static TcScratch& tc_scratch_for(const torch::Tensor& params, int B, int nCTA) {
    static TcScratch s;
    const int T = B * mb::kSeq;
    const int64_t need = ::sg::fused::sm90::mb_tc_workspace_floats(T, nCTA);
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
// Returns {loss (cpu float scalar), grad [total] fp32 (the reduced grad)}.
static std::vector<torch::Tensor> tc_train_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kMambaTotalElems, "params must be the flat mamba blob (", kMambaTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * mb::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B %% 16 == 0 (dW K-loop is 16-step atoms)");
    const int64_t total = kMambaTotalElems;
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
        ::sg::fused::sm90::kMambaNumTensors,
        0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::AdamW>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC mamba megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return {loss_dev.to(torch::kCPU), grad};
}

// CALIBRATION HOOK (test_mamba_tc.py::test_tc_proj_dw_exact_on_own_operands):
// slice the kernel's OWN stored bf16 acts dY_dyout[L1] [T,d] and X_ygated[L1]
// [T,d_inner] from the workspace, returned as fp32 CPU tensors. The gate
// contracts them in fp32 ascending-t and compares to the kernel's
// out_proj.weight grad slice — isolating the output-stationary dW GEMM (K=T)
// from the operand-chain bf16 divergence. A ~1e-6 match proves the dW GEMM is
// bit-exact on its own operands, calibrating the per-tensor bf16 tol as headroom
// over a GEMM-exact floor. Reuses the cached workspace (call AFTER tc_train_step).
static std::vector<torch::Tensor> tc_dump_outproj_operands(torch::Tensor params, int64_t B) {
    TcScratch& sc = tc_scratch_for(params, (int)B, tc_effective_nctas(params, 4));
    const int T = (int)B * mb::kSeq;
    auto acts_bf16 = sc.workspace.slice(0, 0, mbtc::mb_acts_floats(T)).view(torch::kBFloat16);
    // walk mb_acts_bind to dY_dyout[1] and X_ygated[1].
    const int64_t d = mb::kD, di = mb::kDInner, dr = mb::kDtRank, dbc = mb::kDbc;
    const int64_t Td = (int64_t)T * d, Tdi = (int64_t)T * di, Tdr = (int64_t)T * dr,
                  Tdbc = (int64_t)T * dbc, T2di = (int64_t)T * 2 * di;
    int64_t off = 0, off_Xyg1 = -1, off_dYout1 = -1;
    for (int li = 0; li < mb::kLayers; ++li) {
        off += Td;            // X_in
        off += Tdi;           // X_xmain
        off += Tdr;           // X_dtraw
        if (li == 1) off_Xyg1 = off;
        off += Tdi;           // X_ygated
        off += T2di;          // dY_dxz
        off += Tdbc;          // dY_dxdbc
        off += Tdi;           // dY_ddtpre
        if (li == 1) off_dYout1 = off;
        off += Td;            // dY_dyout
    }
    auto dYout1 = acts_bf16.slice(0, off_dYout1, off_dYout1 + Td).to(torch::kFloat32).to(torch::kCPU);
    auto Xyg1 = acts_bf16.slice(0, off_Xyg1, off_Xyg1 + Tdi).to(torch::kFloat32).to(torch::kCPU);
    return {dYout1, Xyg1};   // [T*d], [T*d_inner]
}

#if SG_MB_SCALAR_MEGAKERNEL
// scalar_train_step: the SAME (mamba × AdamW) cell run through the SCALAR default
// launcher (launch_fused_mamba_megakernel<AdamW>, OUTSIDE the wgmma guard — this
// wgmma-token TU can still call it). For the step-time gate to time scalar and TC
// back-to-back in one process / one contention regime → an honest scalar:TC ratio.
// The scalar path uses its OWN workspace (nCTA*total partials + nCTA loss + 1).
// NOT the shipped invocation (that is dispatch.cpp via mega_mamba_real_adamw.cu);
// a measurement mirror. Returns {loss, reduced grad}. NOTE: the scalar mamba
// kernel needs ~142KB DYNAMIC smem (its launcher does the opt-in internally). GATED
// by SG_MB_SCALAR_MEGAKERNEL: at the d-scaled bench width the scalar megakernel's
// MambaSampleSmem overflows the smem budget and is compiled out (the TC path is what
// the bench measures), exactly as the decoder bench gates its scalar step.
static std::vector<torch::Tensor> scalar_train_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kMambaTotalElems, "params must be the flat mamba blob (", kMambaTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * mb::kSeq, "tokens must be [B,kSeq]");
    const int64_t total = kMambaTotalElems;
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= 3 * total, "state must be contiguous fp32 [>=3*total]");

    int dev = params.get_device();
    int n_sms = 1;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);

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
        ::sg::fused::sm90::kMambaNumTensors, 0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m; st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = s_ws.data_ptr<float>();
    tok.loss_out = loss_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_mamba_megakernel<OptId::AdamW>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream);
    TORCH_CHECK(err == cudaSuccess, "scalar mamba megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto loss_cpu = s_ws.slice(0, (int64_t)n_sms * total + n_sms, (int64_t)n_sms * total + n_sms + 1)
                        .to(torch::kCPU);
    return {loss_cpu, grad};
}
#endif  // SG_MB_SCALAR_MEGAKERNEL

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step", &tc_train_step,
           "Fork-B tensor-core mamba fwd+bwd+AdamW step (in place); returns (loss, reduced grad)",
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 0);
    mm.def("tc_dump_outproj_operands", &tc_dump_outproj_operands,
           "gate-only: dump the kernel's stored dY_dyout[L1], X_ygated[L1] (fp32 CPU)");
#if SG_MB_SCALAR_MEGAKERNEL
    mm.def("scalar_train_step", &scalar_train_step,
           "gate-only: SAME cell via the SCALAR launcher (back-to-back step-time mirror)",
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"));
#endif  // SG_MB_SCALAR_MEGAKERNEL
    mm.attr("TILE_M") = (int)mbtc::kTileM;
    mm.attr("TILE_N") = (int)SG_TUNED_TILE_N;
    mm.attr("TOTAL") = (int)kMambaTotalElems;
    // Compiled-width introspection (the d-scaled bench asserts/reports these).
    mm.attr("D") = (int)mb::kD;
    mm.attr("DINNER") = (int)mb::kDInner;
    mm.attr("DTRANK") = (int)mb::kDtRank;
    mm.attr("LAYERS") = (int)mb::kLayers;
    mm.attr("SEQ") = (int)mb::kSeq;
    mm.attr("STATE") = (int)mb::kState;
    mm.attr("VOCAB") = (int)mb::kVocab;
    mm.attr("PHEAD") = (int)mb::kPHead;
}
