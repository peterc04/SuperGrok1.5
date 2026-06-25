// tuning/_tp8_scratch_pybind.cu — NON-COMMITTED scratch wiring (the multiopt-runner
// style) for the TP8 + in-kernel device-NVSHMEM all-reduce flagship run.
//
// WHY: the committed mega_decoder_real_adamw_tc.cu pybind `tc_train_step` calls
// launch_fused_decoder_megakernel_tc<OptId::AdamW>() — the SingleGPU template,
// tp_size=1, NO CommCtx — so it NEVER fires the in-kernel TP all-reduce. The TP
// dispatch (ParTP8 + dec_tc_ensure_tp_sym_heap + CommCtx + the device
// NvshmemTransport) lives in the launcher's 18-arg mega_decoder_real_adamw_tc(...,
// int tp_size) (mega_decoder_real_adamw_tc_launcher.cu:128, the `tp_size==8` arm).
// That arm is reached only via a caller passing tp_size=8.
//
// THIS scratch TU #includes the committed launcher .cu (so it gets the 18-arg
// TP-aware function verbatim — NO committed source edited) and exposes ONE pybind
// entry `tc_train_step_tp8(params, tokens, targets, state, scalars..., step, ncta)`
// that calls that function with tp_size=8. Built with -DSG_HAS_NVSHMEM=1 -rdc=true
// + device-linked against libnvshmem_device.a, so:
//   * the launcher arm's nvshmem_malloc / nvshmem_team_my_pe resolve to the SAME
//     process-global NVSHMEM world the sg_nvshmem_bringup pybind already nvshmem_init'd
//     (same physical libnvshmem_host.so.3 in the process);
//   * the megakernel's device-side make_transport_from_comm<ParTP8>() returns the
//     REAL NvshmemTransport (tp_transport.cuh:291), whose peer() uses nvshmem_ptr
//     over NVLink — the in-kernel device all-reduce, NOT loopback.
//
// The launcher's anonymous-namespace DecTcLauncherScratch + dec_tc_ensure_tp_sym_heap
// come in via the #include; we DO NOT redefine them.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda.h>            // CUmodule / cuFuncGetModule (device-state module init)
#include <nvshmemx.h>        // nvshmemx_cumodule_init (register device state in THIS .so)
#include <cstdint>
#include <vector>

// Pull in the committed TP-aware launcher verbatim (gives us the 18-arg
// ::sg::fused::sm90::mega_decoder_real_adamw_tc + DecTcLauncherScratch +
// dec_tc_ensure_tp_sym_heap + the whole megakernel header). Compiled as ONE TU.
#include "csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu"

using ::sg::fused::sm90::kDecTotalElems;
using ::sg::fused::sm90::kDecNumTensors;
namespace dec = ::sg::fused::sm90::dec;

// The ParTP8 megakernel instantiation this TU launches (OptId::AdamW, ParTP8).
// We need its SYMBOL to recover the CUmodule of THIS .so for nvshmemx_cumodule_init.
using ScratchParTP8 = ::sg::fused::par::ParConfig<
    /*DP=*/8, /*TP=*/8, /*PP=*/1, /*SP=*/1, ::sg::fused::par::ZeROStage::Z3>;

// init_device_state — register THIS module's NVSHMEM DEVICE STATE
// (nvshmemi_device_state_d) so the megakernel's in-kernel nvshmemx_barrier_block /
// nvshmem_ptr peer reads see a non-null state. REQUIRED because this .so is
// device-linked against its OWN copy of libnvshmem_device and is dlopen'd as a
// SEPARATE CUDA module: nvshmem_init (run from the bringup .so) populates device
// state ONLY in the modules it tracks, NOT in a separately-loaded RDC .so — so the
// in-kernel team barrier read a NULL state pointer (the +0x34610 IMA). The supported
// fix for a separately-loaded module is nvshmemx_cumodule_init(CUmodule). We recover
// this .so's CUmodule from the megakernel kernel symbol (cudaGetFuncBySymbol →
// cuFuncGetModule). MUST be called AFTER nvshmem_init_with_uniqueid (the host world
// is up) and BEFORE the first tc_train_step_tp8.
static void init_device_state() {
    const void* kfn = reinterpret_cast<const void*>(
        &::sg::fused::sm90::fused_decoder_megakernel_tc<
            ::sg::fused::sm90::OptId::AdamW, ScratchParTP8>);
    cudaFunction_t cf = nullptr;
    C10_CUDA_CHECK(cudaGetFuncBySymbol(&cf, kfn));
    CUmodule mod = nullptr;
    CUresult r = cuFuncGetModule(&mod, reinterpret_cast<CUfunction>(cf));
    TORCH_CHECK(r == CUDA_SUCCESS, "cuFuncGetModule failed (rc=", (int)r, ")");
    int rc = nvshmemx_cumodule_init(mod);
    TORCH_CHECK(rc == 0, "nvshmemx_cumodule_init failed (rc=", rc, ") — the in-kernel "
                "device-NVSHMEM state was not registered into this module");
}

// Expose the built-in flagship dims so the harness can assert the layout swap took.
static int64_t cfg_D()      { return (int64_t)dec::kD; }
static int64_t cfg_LAYERS() { return (int64_t)dec::kLayers; }
static int64_t cfg_TOTAL()  { return (int64_t)kDecTotalElems; }
static int64_t cfg_NUMT()   { return (int64_t)kDecNumTensors; }

// tc_train_step_tp8 — one Fork-B TC step through the TP8 in-kernel-allreduce path.
//   params  [total] fp32  (the FULL flat decoder blob, replicated per rank — the
//           kernel reads its TP-owned column/row shards via the layout; the host
//           passes the whole blob and the kTPComm GEMM tiles index the rank's
//           kTP slice. params updated in place by the P3 AdamW tail.)
//   tokens  [B,kSeq] int32 ; targets [B] int32  (TP-replicated batch)
//   state   [>= 4*total + 8] fp32  ([m | v | extra | loss | ...]) — AdamW uses
//           m,v; loss_out points at state + 3*total.
// Returns {loss (cpu f32 scalar), grad [total] fp32 (the rank's reduced grad)}.
static std::vector<torch::Tensor> tc_train_step_tp8(
        torch::Tensor params, torch::Tensor tokens, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kDecTotalElems,
                "params must be the flat decoder blob (", kDecTotalElems, ")");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(tokens.numel() == (int64_t)B * dec::kSeq, "tokens must be [B,kSeq]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B %% 16 == 0");
    const int64_t total = kDecTotalElems;
    // The launcher binds loss_out = state + 3*total and writes [param_init|...]
    // AFTER it for staged opts; AdamW only needs m,v,extra + the loss slot, so
    // require >= 3*total + 1. Give a margin (4*total + 8) to be safe.
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32 && state.is_contiguous()
                && state.numel() >= 3 * total + 1,
                "state must be contiguous fp32 [>= 3*total + 1]");

    auto grad = torch::zeros({total}, torch::dtype(torch::kFloat32).device(params.device()));

    // Persistent grid-barrier counters (cached per device inside the launcher's
    // DecTcLauncherScratch; ctx fields get re-pointed there by the launcher, so the
    // values we pass for g_* are placeholders the launcher overwrites). We still
    // must pass a valid PersistentContext POD.
    static torch::Tensor s_gn, s_ga, s_gg;
    if (!s_gn.defined()) {
        auto oi = torch::dtype(torch::kInt32).device(params.device());
        s_gn = torch::zeros({1}, oi); s_ga = torch::zeros({1}, oi); s_gg = torch::zeros({1}, oi);
    }
    ::sg::fused::PersistentContext ctx{
        s_gn.data_ptr<int>(),
        reinterpret_cast<unsigned*>(s_ga.data_ptr<int>()),
        reinterpret_cast<unsigned*>(s_gg.data_ptr<int>()),
        kDecNumTensors, 0u};

    // loss_out lives in the caller's state buffer at state + 3*total (the launcher's
    // dispatch loss_slot convention).
    float* state_p   = state.data_ptr<float>();
    float* loss_out  = state_p + 3 * total;

    // FusedScalars POD (AdamW reads lr/beta1/beta2/eps/wd/bc1/bc2; the rest inert).
    ::sg::fused::sm90::FusedScalars scalars;
    scalars.lr = (float)lr; scalars.beta1 = (float)beta1; scalars.beta2 = (float)beta2;
    scalars.eps = (float)eps; scalars.wd = (float)weight_decay;
    scalars.bc1 = (float)bc1; scalars.bc2 = (float)bc2;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // THE TP8 call: tp_size=8 → the launcher's ParTP8 arm (dec_tc_ensure_tp_sym_heap
    // + CommCtx + the device NvshmemTransport in-kernel all-reduce).
    cudaError_t err = ::sg::fused::sm90::mega_decoder_real_adamw_tc(
        ctx, params.data_ptr<float>(),
        tokens.data_ptr<int>(), targets.data_ptr<int>(), B,
        state_p, grad.data_ptr<float>(), /*workspace_unused=*/nullptr, loss_out,
        /*sizes=*/nullptr, /*offsets=*/nullptr,
        (float)lr, (int)step, scalars, stream, (int)ncta_cap,
        /*opt_id=*/(int)::sg::fused::sm90::OptId::AdamW, /*tp_size=*/8);
    TORCH_CHECK(err == cudaSuccess,
                "TP8 TC megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    // loss_out holds the reduced NLL the kernel wrote (state + 3*total).
    auto loss_dev = state.narrow(0, 3 * total, 1).clone();
    return {loss_dev.to(torch::kCPU), grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TP8 scratch pybind: in-kernel device-NVSHMEM all-reduce flagship step.";
    m.def("tc_train_step_tp8", &tc_train_step_tp8,
          py::arg("params"), py::arg("tokens"), py::arg("targets"), py::arg("state"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("bc1"), py::arg("bc2"), py::arg("step"),
          py::arg("ncta_cap"),
          "one Fork-B TC step via the ParTP8 in-kernel NVSHMEM all-reduce path");
    m.def("init_device_state", &init_device_state,
          "register THIS module's NVSHMEM device state (nvshmemx_cumodule_init) so "
          "the in-kernel team barrier / nvshmem_ptr peer reads are non-null; call "
          "AFTER nvshmem_init and BEFORE the first tc_train_step_tp8");
    m.attr("D")      = cfg_D();
    m.attr("LAYERS") = cfg_LAYERS();
    m.attr("TOTAL")  = cfg_TOTAL();
    m.attr("NUMT")   = cfg_NUMT();
}
