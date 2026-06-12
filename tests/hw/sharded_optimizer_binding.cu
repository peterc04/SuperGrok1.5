// ============================================================================
// tests/hw/sharded_optimizer_binding.cu — THIN JIT binding for the ZeRO-2/3
// sharded-optimizer kernel (csrc/fused/sm_90/sharded_optimizer_kernel.cuh).
//
// PURPOSE (/workspace/.parallelism_design.md §7.3, task #25 DELIVERABLE 1): give
// tests/hw/test_sharded_optimizer.py + test_dp2_loopback_determinism.py a real
// device entry point to `sharded_optimizer_kernel<Opt>` for the ELEMENTWISE
// OptIds {adamw, lion, grokfast} so the DP=1 bit-parity check (sharded kernel ==
// in-kernel P3 fused tail on the SAME (params, grad, state)) can run on one GPU.
//
// IT IS NOT THE PRODUCTION BUILD. It is JIT-loaded via torch.utils.cpp_extension
// .load into its OWN build dir under /workspace/.torch_ext — it does NOT touch
// the production `_ops` extension (which the setup.py glob excludes anyway: this
// TU owns its own PYBIND11_MODULE(TORCH_EXTENSION_NAME …), and
// setup.py::_owns_extension_module drops exactly such standalone-JIT drivers).
//
// ZERO NEW MATH: it includes the header verbatim and reuses
// sg::fused::sm90::sharded_optimizer_kernel<Opt> + the canonical FusedOptState /
// apply_scalars from opt_components.cuh. The FusedOptState construction here is a
// 1:1 mirror of mega_decoder_real_adamw_tc_launcher.cu's binding (state slices
// [m|v|extra] = [state | state+total | state+2*total]; apply_scalars then
// st.lr=lr), so the per-element apply at index i is byte-identical to the P3
// tail's apply at the same flat element — which is the whole point of §7.3.
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

// Pulls in sharded_optimizer_kernel<Opt,Par> + (transitively) OptId /
// FusedOptState / FusedScalars / apply_scalars / apply_optimizer<Opt>.
#include "csrc/fused/sm_90/sharded_optimizer_kernel.cuh"
#include "csrc/fused/sm_90/parallel_config.cuh"

namespace {

using sg::fused::sm90::FusedOptState;
using sg::fused::sm90::FusedScalars;
using sg::fused::sm90::OptId;
using sg::fused::sm90::apply_scalars;
using sg::fused::sm90::launch_sharded_optimizer_kernel;
using SingleGPU = sg::fused::par::SingleGPU;

// Build the FusedScalars POD the same way dispatch.cpp does for the decoder TC
// path (csrc/bindings/dispatch.cpp:893). Only the fields the elementwise tails
// read are non-default; everything else keeps the inert ABI default (the struct
// declares in-class defaults). bc1/bc2 are the UN-INVERTED bias corrections the
// host computes (1 - beta^step) — passed in so they match _opt_scalars_from
// byte-for-byte (the single source the production path uses).
FusedScalars make_scalars(double lr, double beta1, double beta2, double eps,
                          double wd, double bc1, double bc2,
                          double alpha, double lamb) {
    FusedScalars s;          // in-class defaults for every field
    s.lr    = static_cast<float>(lr);
    s.beta1 = static_cast<float>(beta1);
    s.beta2 = static_cast<float>(beta2);
    s.eps   = static_cast<float>(eps);
    s.wd    = static_cast<float>(wd);
    s.bc1   = static_cast<float>(bc1);
    s.bc2   = static_cast<float>(bc2);
    s.alpha = static_cast<float>(alpha);   // grokfast EMA decay (inert for adamw/lion)
    s.lamb  = static_cast<float>(lamb);    // grokfast amplification (inert for adamw/lion)
    return s;
}

// Mirror of the TC launcher's FusedOptState binding (lines 138-185): the m/v/ema
// slices over the SAME flat [m|v|extra] state buffer, then apply_scalars + st.lr.
// `total` is the FULL param count (DP=1: the owned shard IS the whole param set).
FusedOptState make_state(float* state, int64_t total,
                         const FusedScalars& scalars, double lr) {
    FusedOptState st;
    st.exp_avg    = state;                 // m   = state[0      : total)
    st.exp_avg_sq = state + total;         // v   = state[total  : 2*total)
    st.ema        = state + 2 * total;     // ema = state[2*total: 3*total)  (grokfast)
    apply_scalars(st, scalars);            // un-frozen bc1/bc2 + alpha/lamb (+ inert rest)
    st.lr = static_cast<float>(lr);        // single source for the rate (matches P3)
    return st;
}

// Dispatch the flat sharded kernel for an elementwise OptId. The kernel is a flat
// grid-stride over [0, shard_numel) calling apply_optimizer<Opt> — the EXACT P3
// math, no GridBarrier (design §2.3). DP=1: shard_numel == total.
void launch_for_opt(int opt_id, float* params, const float* grad, int64_t total,
                    const FusedOptState& st, double lr, int step,
                    cudaStream_t stream) {
    cudaError_t err = cudaSuccess;
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            err = launch_sharded_optimizer_kernel<OptId::AdamW, SingleGPU>(
                params, grad, total, static_cast<float>(lr), step, st, stream);
            break;
        case OptId::Lion:
            err = launch_sharded_optimizer_kernel<OptId::Lion, SingleGPU>(
                params, grad, total, static_cast<float>(lr), step, st, stream);
            break;
        case OptId::Grokfast:
            err = launch_sharded_optimizer_kernel<OptId::Grokfast, SingleGPU>(
                params, grad, total, static_cast<float>(lr), step, st, stream);
            break;
        default:
            TORCH_CHECK(false,
                "sharded_optimizer_binding: opt_id ", opt_id, " is not one of the "
                "elementwise ids {adamw=0, lion=1, grokfast=2} this JIT binding "
                "instantiates (design §7.3 — the per-tensor cells need the full "
                "persistent kernel, not this flat shard).");
    }
    TORCH_CHECK(err == cudaSuccess,
        "sharded_optimizer_kernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────
//  sharded_opt_step — run ONE sharded-optimizer step IN PLACE on `params`.
//
//  params : [total] fp32 owned param slice (DP=1: the whole flat param set),
//           MUTATED in place (the test compares it to the P3 fused tail's output).
//  grad   : [total] fp32 reduce-scattered grad (DP=1: the captured fused grad).
//  state  : [>= 3*total] fp32 [m|v|extra] state buffer (zero-init at step 1, as
//           the production state cache is). MUTATED in place (m/v/ema updated).
//  The scalars are passed positionally so the Python side computes them via the
//  SAME _opt_scalars_from the production path uses — guaranteeing bit-identity.
// ─────────────────────────────────────────────────────────────────────────
void sharded_opt_step(int64_t opt_id,
                      at::Tensor params, at::Tensor grad, at::Tensor state,
                      double lr, double beta1, double beta2, double eps,
                      double weight_decay, double bc1, double bc2,
                      double alpha, double lamb, int64_t step) {
    TORCH_CHECK(params.is_cuda() && grad.is_cuda() && state.is_cuda(),
                "sharded_opt_step: params/grad/state must be CUDA tensors");
    TORCH_CHECK(params.scalar_type() == at::kFloat &&
                grad.scalar_type() == at::kFloat &&
                state.scalar_type() == at::kFloat,
                "sharded_opt_step: params/grad/state must be float32");
    TORCH_CHECK(params.is_contiguous() && grad.is_contiguous() &&
                state.is_contiguous(),
                "sharded_opt_step: params/grad/state must be contiguous");
    const int64_t total = params.numel();
    TORCH_CHECK(grad.numel() == total,
                "sharded_opt_step: grad numel ", grad.numel(),
                " != params numel ", total);
    TORCH_CHECK(state.numel() >= 3 * total,
                "sharded_opt_step: state needs >= 3*total (", 3 * total,
                ") floats, got ", state.numel());

    FusedScalars scalars = make_scalars(lr, beta1, beta2, eps, weight_decay,
                                        bc1, bc2, alpha, lamb);
    FusedOptState st = make_state(state.data_ptr<float>(), total, scalars, lr);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_for_opt(static_cast<int>(opt_id), params.data_ptr<float>(),
                   grad.data_ptr<float>(), total, st, lr,
                   static_cast<int>(step), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "JIT binding for the ZeRO-2/3 sharded-optimizer kernel "
              "(design §7.3 bit-parity gate; elementwise OptIds only).";
    m.def("sharded_opt_step", &sharded_opt_step,
          "Run one sharded_optimizer_kernel<Opt> step in place over the owned "
          "flat shard (opt_id in {0=adamw,1=lion,2=grokfast}).",
          py::arg("opt_id"), py::arg("params"), py::arg("grad"), py::arg("state"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("bc1"), py::arg("bc2"),
          py::arg("alpha"), py::arg("lamb"), py::arg("step"));
}
