// SCRATCHPAD JIT DRIVER (NOT a committed-source edit): OptId-generic Mamba-3 flagship
// TC driver. Mirrors mega_mamba_real_adamw_tc.cu's tc_train_step but dispatches over
// OptId via a runtime opt_id. Own pybind module. Built against mamba_flagship_layout.cuh.
//
// Scope: single-launch elementwise opts over [m|v|extra] (3*total):
//   AdamW(0), Lion(1), Grokfast(2), GrokAdamW(3), NeuralGrok(6).

#define SG_TUNED_GEMM_IMPL 1

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"

namespace mb = ::sg::fused::sm90::mb;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::MambaTokenCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::FusedScalars;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kMambaTotalElems;
using ::sg::fused::sm90::kMambaNumTensors;
using ::sg::fused::sm90::apply_scalars;
using ::sg::fused::sm90::kPsiW1Off;
using ::sg::fused::sm90::kPsiB1Off;
using ::sg::fused::sm90::kPsiW2Off;

struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace;
    int n_ctas = 0; int64_t ws_floats = 0;
};

static int tc_effective_nctas(const torch::Tensor& params, int ncta_cap) {
    int dev = params.get_device();
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
    return s;
}

static cudaError_t launch_opt(int opt_id, PersistentContext ctx, float* params,
                              MambaTokenCtx tok, float* grad, float lr, int step,
                              FusedOptState st, cudaStream_t stream, int nCTA) {
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::Lion>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::Grokfast>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            return ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            return ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;
    }
}

static std::vector<torch::Tensor> tc_train_step_opt(
        int64_t opt_id,
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        double alpha, double lamb, double gamma, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kMambaTotalElems, "params must be the flat mamba blob (", kMambaTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * mb::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B % 16 == 0");
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
        kMambaNumTensors, 0u};

    float* m = state.data_ptr<float>();
    float* const extra_slice = m + 2 * total;
    FusedOptState st;
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.ema = extra_slice;
    st.psi_W1 = extra_slice + kPsiW1Off;
    st.psi_b1 = extra_slice + kPsiB1Off;
    st.psi_W2 = extra_slice + kPsiW2Off;

    FusedScalars scal;
    scal.lr = (float)lr; scal.beta1 = (float)beta1; scal.beta2 = (float)beta2;
    scal.eps = (float)eps; scal.wd = (float)weight_decay;
    scal.bc1 = (float)bc1; scal.bc2 = (float)bc2;
    scal.alpha = (float)alpha; scal.lamb = (float)lamb; scal.gamma = (float)gamma;
    apply_scalars(st, scal);
    st.lr = (float)lr;

    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = launch_opt((int)opt_id, ctx, params.data_ptr<float>(), tok,
                                 grad.data_ptr<float>(), (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC mamba megakernel launch failed (opt_id=", opt_id, "): ",
                cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step_opt", &tc_train_step_opt,
           "OptId-generic TC Mamba fwd+bwd+opt step (elementwise opts); returns (loss, grad)",
           pybind11::arg("opt_id"),
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("alpha") = 0.98, pybind11::arg("lamb") = 2.0,
           pybind11::arg("gamma") = 0.0, pybind11::arg("ncta_cap") = 0);
    mm.attr("TOTAL") = (int)kMambaTotalElems;
    mm.attr("D") = (int)mb::kD;
    mm.attr("DINNER") = (int)mb::kDInner;
    mm.attr("DTRANK") = (int)mb::kDtRank;
    mm.attr("LAYERS") = (int)mb::kLayers;
    mm.attr("SEQ") = (int)mb::kSeq;
    mm.attr("STATE") = (int)mb::kState;
    mm.attr("VOCAB") = (int)mb::kVocab;
    mm.attr("PHEAD") = (int)mb::kPHead;
}
