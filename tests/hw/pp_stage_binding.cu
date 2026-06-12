// ============================================================================
// tests/hw/pp_stage_binding.cu — JIT binding for the PIPELINE-PARALLEL 2-stage
// loopback gate (design §4, task #25 phase-2, Lane C).
//
// Drives csrc/fused/sm_90/pp_stage_decoder_tc.cuh on ONE GPU: both stages of a
// PP=2 split of the decoder TC megakernel run as sequential launches on one
// device, sharing one workspace (the acts handoff is zero-copy — exactly the
// virtual-boundary discipline ef433ac used for DP=2). The 1F1B mb=1 order:
//
//     [stage0 fwd]  →  X_in[1] boundary acts (bf16 [T,d], in the shared ws)
//     [stage1 fwd+bwd] (the fused last-stage unit) → owned grads {14..29}
//                      + loss + dh_boundary (fp32 [T,d])
//     [stage0 fwd-recompute+bwd] (dh_in = dh_boundary) → owned grads {0..13}
//
// The test (test_pp2_loopback_determinism.py) asserts the CONCATENATED grad +
// loss are BIT-IDENTICAL to the single-launch production fused step's
// (return_grad) output, and A/A/A across 3 reruns — the PP analogue of
// ef433ac's DP=2 cross-rank A/A/A.
//
// REQUIRES the layer-range patch on model_stage_decoder_tc.cuh (the include
// chain #errors loudly if absent — see pp_stage_decoder_tc.cuh). JIT-built into
// /workspace/.torch_ext; owns its PYBIND11_MODULE (excluded from _ops by
// setup.py::_owns_extension_module).
// ============================================================================

#define SG_TUNED_GEMM_IMPL 1   // the wgmma (production L3-TC) path

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "csrc/fused/sm_90/pp_stage_decoder_tc.cuh"

namespace {

using namespace sg::fused;          // PersistentContext
using namespace sg::fused::sm90;    // dec::, dectc::, DecoderTokenCtx
namespace pp = sg::fused::sm90::pp;

constexpr int kNumStages = 2;       // the race decoder: 2 layers → 1 layer/stage

// One PersistentContext worth of device slots carved from an int32 tensor:
// [ g_next_task | g_arrived | g_generation ].
PersistentContext make_ctx(at::Tensor& slots) {
    PersistentContext ctx{};
    int* base = slots.data_ptr<int>();
    ctx.g_next_task  = base;
    ctx.g_arrived    = reinterpret_cast<unsigned*>(base + 1);
    ctx.g_generation = reinterpret_cast<unsigned*>(base + 2);
    ctx.n_tasks = kDecNumTensors;   // unused by the PP stages (no P3) — kept uniform
    ctx.n_ctas = 0;                 // launcher fills
    return ctx;
}

// ─────────────────────────────────────────────────────────────────────────
//  pp2_step — ONE PP=2 loopback step. params [total] fp32 (NOT mutated — the
//  optimizer is the separate sharded-opt kernel, design §2.2); tokens [T] /
//  targets [B] int32 CUDA. Returns (loss[1], grad[total], dh[T*d]).
//  ncta_cap mirrors the production launcher arg (0 = one CTA per SM).
// ─────────────────────────────────────────────────────────────────────────
std::vector<at::Tensor> pp2_step(at::Tensor params, at::Tensor tokens,
                                 at::Tensor targets, int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == at::kFloat &&
                params.is_contiguous() && params.numel() == kDecTotalElems,
                "params must be CUDA fp32 contiguous [", kDecTotalElems, "]");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == at::kInt && tokens.is_contiguous(),
                "tokens must be CUDA int32 contiguous [B*kSeq]");
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == at::kInt && targets.is_contiguous(),
                "targets must be CUDA int32 contiguous [B]");
    const int B = (int)targets.numel();
    const int T = B * dec::kSeq;
    TORCH_CHECK(tokens.numel() == (int64_t)T, "tokens numel != B*kSeq");
    TORCH_CHECK(B % 16 == 0, "B must be a multiple of 16 (wgmma K-step), got ", B);

    int dev = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    int n_sms = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    unsigned nCTA = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < nCTA) nCTA = (unsigned)ncta_cap;

    auto opts_f = params.options();
    auto opts_i = tokens.options();
    // SAME workspace carve as the production fused step (shared by all three
    // launches — the zero-copy acts handoff + identical dw_part slot bases).
    const int64_t ws_floats = dec_tc_workspace_floats(T, B, (int)nCTA);
    at::Tensor ws    = at::zeros({ws_floats}, opts_f);
    at::Tensor grad  = at::zeros({(int64_t)kDecTotalElems}, opts_f);
    at::Tensor dh    = at::zeros({(int64_t)T * dec::kD}, opts_f);   // fp32 boundary adjoint
    at::Tensor loss  = at::zeros({1}, opts_f);
    at::Tensor slots = at::zeros({4}, opts_i);                      // ctx slots

    DecoderTokenCtx tok{};
    tok.tokens   = tokens.data_ptr<int>();
    tok.targets  = targets.data_ptr<int>();
    tok.B        = B;
    tok.workspace = ws.data_ptr<float>();
    tok.loss_out = loss.data_ptr<float>();

    PersistentContext ctx = make_ctx(slots);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* p = params.data_ptr<float>();
    float* g = grad.data_ptr<float>();
    float* dhp = dh.data_ptr<float>();

    // [1] stage0 FWD — boundary acts X_in[1] into the shared workspace.
    cudaError_t err = pp::launch_pp_decoder_stage_fwd_tc<0, kNumStages>(
        ctx, p, tok, stream, (int)ncta_cap);
    TORCH_CHECK(err == cudaSuccess, "stage0 fwd launch: ", cudaGetErrorString(err));

    // [2] stage1 FWD+BWD (the fused last-stage unit): consumes X_in[1], produces
    //     owned grads {14..29} + loss + the fp32 dh boundary handoff.
    err = pp::launch_pp_decoder_stage_bwd_tc<1, kNumStages>(
        ctx, p, tok, g, /*dh_in=*/nullptr, /*dh_out=*/dhp, stream, (int)ncta_cap);
    TORCH_CHECK(err == cudaSuccess, "stage1 bwd launch: ", cudaGetErrorString(err));

    // [3] stage0 FWD-RECOMPUTE+BWD: consumes dh, produces owned grads {0..13}.
    err = pp::launch_pp_decoder_stage_bwd_tc<0, kNumStages>(
        ctx, p, tok, g, /*dh_in=*/dhp, /*dh_out=*/nullptr, stream, (int)ncta_cap);
    TORCH_CHECK(err == cudaSuccess, "stage0 bwd launch: ", cudaGetErrorString(err));

    return {loss, grad, dh};
}

// Host-side ownership introspection (the Python pipeline plan cross-checks
// against THIS — the kernel-side single source).
std::vector<int64_t> pp2_owned_tensors(int64_t stage) {
    std::vector<int64_t> out;
    for (int t = 0; t < kDecNumTensors; ++t) {
        const bool own = (stage == 0)
            ? pp::PPStageSpec<0, kNumStages>::owns_tensor(t)
            : pp::PPStageSpec<1, kNumStages>::owns_tensor(t);
        if (own) out.push_back(t);
    }
    return out;
}

// Per-tensor (offset, size) of the dec layout — lets the test slice the grad
// per owned tensor without re-deriving the layout in Python.
std::vector<int64_t> pp2_layout() {
    std::vector<int64_t> out;
    for (int t = 0; t < kDecNumTensors; ++t) {
        out.push_back(sm90::dec_layout_check::kOffsets[t]);
        out.push_back(sm90::dec_layout_check::kSizes[t]);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PP=2 loopback stage binding (design §4; layer-range patch required).";
    m.def("pp2_step", &pp2_step,
          "Run one PP=2 loopback step (s0 fwd → s1 fwd+bwd → s0 fwd+bwd); "
          "returns (loss[1], grad[total], dh_boundary[T*d]).",
          py::arg("params"), py::arg("tokens"), py::arg("targets"),
          py::arg("ncta_cap") = 0);
    m.def("pp2_owned_tensors", &pp2_owned_tensors,
          "Tensor indices owned by a stage (kernel-side single source).");
    m.def("pp2_layout", &pp2_layout, "Flat [off,size]*30 of the decoder layout.");
    m.attr("TOTAL") = (int64_t)kDecTotalElems;
    m.attr("SEQ") = (int64_t)dec::kSeq;
    m.attr("NUM_STAGES") = (int64_t)kNumStages;
}
