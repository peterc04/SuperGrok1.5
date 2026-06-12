// _mamba_prodigy_probe.cu — DIAGNOSTIC probe TU for the shared mamba-forward
// A/A/A race that blocks mamba×prodigy + mamba×looksam. Instantiates the Prodigy
// (and LookSAM) mamba TC kernel and runs the SAME step N times from a bit-identical
// init, returning (loss, reduced grad) so a Python driver can diff A/A/A and so
// compute-sanitizer (racecheck/initcheck) can wrap a single step. NOT shipped /
// not in setup.py glob — JIT-loaded by _mamba_race_probe.py.
//
// Mirrors mega_mamba_real_adamw_tc.cu::tc_train_step but templated on the Opt so
// we can drive the Prodigy/LookSAM register/occupancy profile that exposes the race.
#define SG_TUNED_GEMM_IMPL 1

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <map>
#include <string>
#include <algorithm>

#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"

namespace mb = ::sg::fused::sm90::mb;
namespace mbtc = ::sg::fused::sm90::mbtc;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::MambaTokenCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kMambaTotalElems;
using ::sg::fused::sm90::kMambaNumTensors;

struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace;
    int64_t ws_floats = 0;
};

// DIAGNOSTIC: sentinel-fill the cached workspace before each launch. If the kernel
// output depends on the fill value, a region is READ-BEFORE-WRITTEN. -1 = no fill
// (production behavior, cached residue). Set via env SG_PROBE_WS_FILL (float) or the
// per-call arg. A region-restricted fill (only [a,b)) localizes WHICH buffer.
static float g_ws_fill = 0.0f;
static bool  g_ws_fill_on = false;
static int64_t g_ws_fill_a = -1, g_ws_fill_b = -1;   // [a,b) in floats; -1,-1 = whole ws

static int eff_nctas(const torch::Tensor& params, int cap) {
    int dev = params.get_device();
    int n = ::sg::fused::sm90::mb_tc_launched_nctas<OptId::Prodigy>(dev, cap);
    if (n <= 0) {
        int n_sms = 1;
        cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
        n = n_sms;
        if (cap > 0 && cap < n) n = cap;
    }
    return n;
}

static TcScratch g_scratch_singleton;
TcScratch& _last_scratch() { return g_scratch_singleton; }
static TcScratch& scratch_for(const torch::Tensor& params, int B, int nCTA) {
    TcScratch& s = g_scratch_singleton;
    const int T = B * mb::kSeq;
    const int64_t need = ::sg::fused::sm90::mb_tc_workspace_floats(T, nCTA);
    if (!s.g_next.defined()) {
        auto oi = torch::dtype(torch::kInt32).device(params.device());
        s.g_next = torch::zeros({1}, oi);
        s.g_arrived = torch::zeros({1}, oi);
        s.g_generation = torch::zeros({1}, oi);
    }
    if (s.ws_floats < need) {
        auto of = torch::dtype(torch::kFloat32).device(params.device());
        s.workspace = torch::empty({need}, of);
        s.ws_floats = need;
    }
    // DIAGNOSTIC sentinel fill (read-before-write probe). Region mode ALWAYS zeroes
    // the whole workspace first, then sets [a,b) to the sentinel — so a clean (0)
    // baseline isolates exactly the region under test (no cached-residue contamination).
    if (g_ws_fill_on) {
        if (g_ws_fill_a < 0) {
            s.workspace.fill_(g_ws_fill);
        } else {
            s.workspace.fill_(0.0f);
            s.workspace.slice(0, g_ws_fill_a, std::min<int64_t>(g_ws_fill_b, s.ws_floats)).fill_(g_ws_fill);
        }
    }
    return s;
}

static void set_ws_fill(bool on, double v, int64_t a, int64_t b) {
    g_ws_fill_on = on; g_ws_fill = (float)v; g_ws_fill_a = a; g_ws_fill_b = b;
}

// One mamba TC step under OptId::Prodigy. state layout: a flat fp32 buffer holding
// [m | v | extra] with extra big enough; param_init is a separate [total] tensor
// seeded == params at step 1; prodigy_persist a [3] device scalar buffer.
// Returns {loss(cpu), grad[total], params_after(clone)}.
static std::vector<torch::Tensor> prodigy_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, torch::Tensor param_init, torch::Tensor persist,
        double lr, double beta1, double beta2, double eps, double weight_decay,
        double bc1, double bc2, double d0, double d_coef, double beta3,
        int64_t step, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kMambaTotalElems);
    const int B = (int)targets.numel();
    TORCH_CHECK((B % 16) == 0);
    const int64_t total = kMambaTotalElems;

    const int nCTA = eff_nctas(params, (int)ncta_cap);
    TcScratch& sc = scratch_for(params, B, nCTA);
    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));
    auto loss_dev = torch::zeros({1}, torch::dtype(torch::kFloat32).device(params.device()));

    PersistentContext ctx{
        sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
        kMambaNumTensors, 0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.s_track = m + 2 * total;   // Prodigy `s` per-element accumulator (extra slice)
    st.ema     = m + 2 * total;   // alias (harmless; prodigy has no slow-grad EMA)
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;
    st.param_init = param_init.data_ptr<float>();
    st.prodigy_persist = persist.data_ptr<float>();
    st.d0 = (float)d0; st.d_coef = (float)d_coef; st.beta3 = (float)beta3;

    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::Prodigy>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "prodigy TC launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad, params.clone()};
}

// One mamba TC step under OptId::LookSAM. state = [m | v | extra/sam_dir(total)].
// looksam_sam (1.0 on SAM steps) gates the in-kernel P2.4 2nd backward; rho/alpha
// the perturbation + blend. Returns {loss(cpu), grad[total], sam_dir(clone)}.
static std::vector<torch::Tensor> looksam_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, double rho, double alpha,
        double looksam_sam, int64_t step, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kMambaTotalElems);
    const int B = (int)targets.numel();
    TORCH_CHECK((B % 16) == 0);
    const int64_t total = kMambaTotalElems;

    const int nCTA = eff_nctas(params, (int)ncta_cap);
    TcScratch& sc = scratch_for(params, B, nCTA);
    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));
    auto loss_dev = torch::zeros({1}, torch::dtype(torch::kFloat32).device(params.device()));

    PersistentContext ctx{
        sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()),
        kMambaNumTensors, 0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.sam_dir = m + 2 * total;   // persistent cached SAM direction (extra slice)
    st.ema     = m + 2 * total;   // alias (harmless)
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;
    st.alpha = (float)alpha; st.rho = (float)rho; st.looksam_sam = (float)looksam_sam;

    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>();
    tok.targets = targets.data_ptr<int>();
    tok.B = B;
    tok.workspace = sc.workspace.data_ptr<float>();
    tok.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::LookSAM>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "looksam TC launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad, state.slice(0, 2 * total, 3 * total).clone()};
}

// DIAG: clone the cached workspace (acts + scratch + partials) after the last
// launch, so Python can diff the FORWARD acts vs BACKWARD acts across runs to
// localize where the non-determinism first appears.
static torch::Tensor dump_ws(int64_t B, int64_t ncta_cap) {
    static TcScratch* last = nullptr;
    (void)B; (void)ncta_cap;
    // scratch_for keeps a single static instance; re-fetch it via a dummy params on CUDA.
    extern TcScratch& _last_scratch();
    TcScratch& sc = _last_scratch();
    if (!sc.workspace.defined()) return torch::zeros({1});
    return sc.workspace.clone();
}

// CONTROL: same harness as looksam_step/prodigy_step (eff_nctas templated on
// Prodigy, same workspace sizing) but launches OptId::AdamW. Isolates harness
// artifact from Opt-specific kernel behavior.
static std::vector<torch::Tensor> adamw_step(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.numel() == kMambaTotalElems);
    const int B = (int)targets.numel();
    TORCH_CHECK((B % 16) == 0);
    const int64_t total = kMambaTotalElems;
    const int nCTA = eff_nctas(params, (int)ncta_cap);
    TcScratch& sc = scratch_for(params, B, nCTA);
    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));
    auto loss_dev = torch::zeros({1}, torch::dtype(torch::kFloat32).device(params.device()));
    PersistentContext ctx{sc.g_next.data_ptr<int>(),
        reinterpret_cast<unsigned*>(sc.g_arrived.data_ptr<int>()),
        reinterpret_cast<unsigned*>(sc.g_generation.data_ptr<int>()), kMambaNumTensors, 0u};
    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m; st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay; st.bc1 = (float)bc1; st.bc2 = (float)bc2;
    MambaTokenCtx tok;
    tok.tokens = tokens.data_ptr<int>(); tok.targets = targets.data_ptr<int>();
    tok.B = B; tok.workspace = sc.workspace.data_ptr<float>(); tok.loss_out = loss_dev.data_ptr<float>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_mamba_megakernel_tc<OptId::AdamW>(
        ctx, params.data_ptr<float>(), tok, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "adamw TC launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {loss_dev.to(torch::kCPU), grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    mm.def("adamw_step", &adamw_step,
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 0);
    mm.def("set_ws_fill", &set_ws_fill,
           pybind11::arg("on"), pybind11::arg("v"),
           pybind11::arg("a") = (int64_t)-1, pybind11::arg("b") = (int64_t)-1);
    // expose workspace-region offsets (floats) so Python can localize the unwritten
    // buffer. Computed for a given (B, nCTA).
    mm.def("ws_layout", [](int64_t B, int64_t nCTA) {
        const int T = (int)B * mb::kSeq;
        std::map<std::string, int64_t> m;
        const int64_t acts_f = mbtc::mb_acts_floats(T);
        const int64_t scratch_per = mbtc::mb_tile_scratch_floats();
        m["acts_begin"] = 0; m["acts_end"] = acts_f;
        int64_t o = acts_f;
        m["scratch_begin"] = o; o += nCTA * scratch_per; m["scratch_end"] = o;
        m["part_begin"] = o; o += nCTA * mbtc::kPartElems; m["part_end"] = o;
        m["loss_begin"] = o; o += nCTA; m["loss_end"] = o;
        m["lossout_begin"] = o; o += 1; m["lossout_end"] = o;
        o += ::sg::fused::sm90::mb_tc_dw_part_floats();
        m["dwpart_end"] = o;
        o += ::sg::fused::sm90::mb_tc_opt_reduce_floats((int)nCTA);
        m["optreduce_end"] = o;
        m["sam_backup_begin"] = o; o += kMambaTotalElems; m["sam_backup_end"] = o;
        m["sam_grad_begin"] = o; o += kMambaTotalElems; m["sam_grad_end"] = o;
        return m;
    }, pybind11::arg("B"), pybind11::arg("nCTA"));
    mm.def("looksam_step", &looksam_step,
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("rho"),
           pybind11::arg("alpha"), pybind11::arg("looksam_sam"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 0);
    mm.def("prodigy_step", &prodigy_step,
           pybind11::arg("params"), pybind11::arg("tokens"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("param_init"), pybind11::arg("persist"),
           pybind11::arg("lr"), pybind11::arg("beta1"), pybind11::arg("beta2"),
           pybind11::arg("eps"), pybind11::arg("weight_decay"), pybind11::arg("bc1"),
           pybind11::arg("bc2"), pybind11::arg("d0"), pybind11::arg("d_coef"),
           pybind11::arg("beta3"), pybind11::arg("step"), pybind11::arg("ncta_cap") = 0);
    mm.def("dump_ws", &dump_ws, pybind11::arg("B") = (int64_t)0, pybind11::arg("ncta_cap") = (int64_t)0);
    mm.attr("TOTAL") = (int)kMambaTotalElems;
    mm.attr("NCTA_DEFAULT") = 0;
}
