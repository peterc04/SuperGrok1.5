// SCRATCHPAD JIT DRIVER (NOT a committed-source edit): OptId-generic ViT flagship
// TC driver. Mirrors mega_vit_real_adamw_tc.cu's tc_train_step but dispatches over
// OptId via a runtime opt_id, exactly as the decoder multiopt driver does. Own pybind
// module so cpp_extension.load drives it directly. Built against vit_flagship_layout.cuh.
//
// Scope: the SINGLE-LAUNCH ELEMENTWISE opts over the plain [m|v|extra] (3*total)
// buffer: AdamW(0), Lion(1), Grokfast(2), GrokAdamW(3), NeuralGrok(6).

#define SG_TUNED_GEMM_IMPL 1

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"

namespace vit = ::sg::fused::sm90::vit;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::ViTInputCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::FusedScalars;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kVitTotalElems;
using ::sg::fused::sm90::kVitNumTensors;
using ::sg::fused::sm90::apply_scalars;
using ::sg::fused::sm90::kPsiW1Off;
using ::sg::fused::sm90::kPsiB1Off;
using ::sg::fused::sm90::kPsiW2Off;

struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace;
    int n_ctas = 0; int64_t ws_floats = 0;
};

static int tc_effective_nctas(const torch::Tensor& params, int ncta_cap) {
    int dev = params.get_device(); int n_sms = 1;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    int n = n_sms;
    if (ncta_cap > 0 && ncta_cap < n) n = ncta_cap;
    return n;
}

static TcScratch& tc_scratch_for(const torch::Tensor& params, int B, int nCTA) {
    static TcScratch s;
    const int T = B * vit::kSeq;
    const int64_t need = ::sg::fused::sm90::vit_tc_workspace_floats(T, B, nCTA);
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
                              ViTInputCtx in, float* grad, float lr, int step,
                              FusedOptState st, cudaStream_t stream, int nCTA) {
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::AdamW>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::Lion>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::Grokfast>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            return ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            return ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, in, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;
    }
}

static std::vector<torch::Tensor> tc_train_step_opt(
        int64_t opt_id,
        torch::Tensor params, torch::Tensor patches, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        double alpha, double lamb, double gamma, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kVitTotalElems, "params must be the flat ViT blob (", kVitTotalElems, ")");
    TORCH_CHECK(patches.is_cuda() && patches.scalar_type() == torch::kFloat32 && patches.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(patches.numel() == (int64_t)B * vit::kNPatch * vit::kPatch, "patches must be [B, kNPatch, kPatch]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B % 16 == 0");
    const int64_t total = kVitTotalElems;
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
        kVitNumTensors, 0u};

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

    ViTInputCtx in;
    in.patches = patches.data_ptr<float>();
    in.targets = targets.data_ptr<int>();
    in.B = B;
    in.workspace = sc.workspace.data_ptr<float>();
    in.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = launch_opt((int)opt_id, ctx, params.data_ptr<float>(), in,
                                 grad.data_ptr<float>(), (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC vit megakernel launch failed (opt_id=", opt_id, "): ",
                cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step_opt", &tc_train_step_opt,
           "OptId-generic TC ViT fwd+bwd+opt step (elementwise opts); returns (loss, grad)",
           pybind11::arg("opt_id"),
           pybind11::arg("params"), pybind11::arg("patches"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("alpha") = 0.98, pybind11::arg("lamb") = 2.0,
           pybind11::arg("gamma") = 0.0, pybind11::arg("ncta_cap") = 0);
    mm.attr("TOTAL") = (int)kVitTotalElems;
    mm.attr("D") = (int)vit::kD;
    mm.attr("HEADS") = (int)vit::kHeads;
    mm.attr("LAYERS") = (int)vit::kLayers;
    mm.attr("SEQ") = (int)vit::kSeq;
    mm.attr("NPATCH") = (int)vit::kNPatch;
    mm.attr("PATCH") = (int)vit::kPatch;
    mm.attr("VOCAB") = (int)vit::kVocab;
}
