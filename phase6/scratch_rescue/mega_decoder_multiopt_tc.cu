// SCRATCHPAD JIT DRIVER (NOT a committed-source edit): generalizes
// mega_decoder_real_adamw_tc.cu's `tc_train_step` to dispatch over OptId via a
// runtime `opt_id`, exactly as the OptId-generic launcher
// mega_decoder_real_adamw_tc_launcher.cu does — but with its OWN pybind module so
// cpp_extension.load can drive it directly (the launcher TU has no pybind; it is
// globbed into _ops). Build-via-include only: compiled against the flagship layout
// with the SAME -D flags + -include decoder_flagship_layout.cuh as flagship_train.py.
//
// Scope: the SINGLE-LAUNCH ELEMENTWISE optimizers the launcher routes WITHOUT extra
// state plumbing over the plain [m|v|extra] (3*total) buffer:
//   AdamW(0), Lion(1), Grokfast(2), GrokAdamW(3), NeuralGrok(6)
// (Prodigy/Muon/LookSAM/SG11/SG15 need an EXTENDED state buffer + extra device packs;
// SG2 needs ncta=1 + a meta-net weight bundle. Those are intentionally out of scope
// here — see the launcher's state-binding block.)
//
// State layout mirrors the launcher: state = [m | v | extra] (3*total); the
// NeuralGrok psi-net pack rides the head of the `extra` slice (kPsiW1Off/...).
// Scalars are passed through FusedScalars so the optimizer math is LIVE (bc1/bc2,
// alpha/lamb/gamma — not the inert struct defaults).

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
using ::sg::fused::sm90::FusedScalars;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kDecTotalElems;
using ::sg::fused::sm90::kDecNumTensors;
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
    return s;
}

// Dispatch one TC step for the requested elementwise OptId. Mirrors
// mega_decoder_real_adamw_tc_launcher.cu's switch (the 5 elementwise arms only).
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
        default:
            return cudaErrorInvalidValue;   // out-of-scope opt for this elementwise driver
    }
}

static std::vector<torch::Tensor> tc_train_step_opt(
        int64_t opt_id,
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        double alpha, double lamb, double gamma, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems, "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B % 16 == 0");
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
        kDecNumTensors, 0u};

    // Bind [m | v | extra] over the SAME state buffer (launcher lines 170-176).
    float* m = state.data_ptr<float>();
    float* const extra_slice = m + 2 * total;
    FusedOptState st;
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.ema = extra_slice;                       // grokfast/grokadamw slow-grad EMA
    st.psi_W1 = extra_slice + kPsiW1Off;        // NeuralGrok psi-net pack head
    st.psi_b1 = extra_slice + kPsiB1Off;
    st.psi_W2 = extra_slice + kPsiW2Off;

    // LIVE scalars (not inert defaults): bc1/bc2 drive Adam bias-correction; alpha/
    // lamb/gamma drive grokfast/grokadamw/neuralgrok. Pass via FusedScalars then
    // apply_scalars (the launcher's apply_scalars(st, scalars) path).
    FusedScalars scal;
    scal.lr = (float)lr; scal.beta1 = (float)beta1; scal.beta2 = (float)beta2;
    scal.eps = (float)eps; scal.wd = (float)weight_decay;
    scal.bc1 = (float)bc1; scal.bc2 = (float)bc2;
    scal.alpha = (float)alpha; scal.lamb = (float)lamb; scal.gamma = (float)gamma;
    apply_scalars(st, scal);
    st.lr = (float)lr;

    DecoderTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = launch_opt((int)opt_id, ctx, params.data_ptr<float>(), tok,
                                 grad.data_ptr<float>(), (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC megakernel launch failed (opt_id=", opt_id, "): ",
                cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("tc_train_step_opt", &tc_train_step_opt,
           "OptId-generic TC decoder fwd+bwd+opt step (elementwise opts); returns (loss, grad)",
           pybind11::arg("opt_id"),
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("alpha") = 0.98, pybind11::arg("lamb") = 2.0,
           pybind11::arg("gamma") = 0.0, pybind11::arg("ncta_cap") = 0);
    mm.attr("TOTAL") = (int)kDecTotalElems;
    mm.attr("D") = (int)::sg::fused::sm90::SG_DEC_D;
    mm.attr("LAYERS") = (int)::sg::fused::sm90::SG_DEC_LAYERS;
    mm.attr("VOCAB") = (int)::sg::fused::sm90::SG_DEC_VOCAB;
    mm.attr("SEQ") = (int)::sg::fused::sm90::SG_DEC_SEQ;
}
