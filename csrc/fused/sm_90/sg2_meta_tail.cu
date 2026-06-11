// csrc/fused/sm_90/sg2_meta_tail.cu — host-launcher TU for the SuperGrok2 FULL
// CSA/HCA/PEER/GRU meta-net persistent megakernel (csrc/fused/sm_90/
// opt_stage_supergrok2.cuh). This is the INTEGRATION seam specified in
// INTEGRATION-NOTES.md §1: the §1b host wrapper body lives HERE (a .cu TU that
// nvcc compiles, so the `kernel<<<...>>>` launch + the device-templated
// `launch_sg2_meta_optimizer_tail<Dims,...>` are legal), while bindings.cpp
// (host-only .cpp) extern-declares + PYBIND11-registers the two entry points
// (it cannot #include the .cuh: host gcc cannot parse the launch syntax).
//
// TWO non-template host entry points, both with a plain torch::Tensor / scalar
// boundary (the same boundary every existing SG2 launcher uses):
//   sg::sg2_meta_optimizer_tail(...)  — marshals the packed buffers + meta
//       bundle + scalars into SG2Weights/SG2State/SG2Scalars/PersistentContext
//       and launches the persistent megakernel (the launch-elimination of
//       csa_hca_step_one's ~15-20 per-tensor launches).
//   sg::sg2_ws_stride(int64_t Nmax) -> int64_t  — exposes the AUTHORITATIVE
//       per-CTA workspace stride sg::fused::sm90::sg2_ws_stride<SG2Dims<>>(Nmax)
//       so the Python driver's host stride can never drift from the kernel's
//       carve (INTEGRATION-NOTES.md §2 "Strongly prefer binding ... directly").
//
// SETUP.PY: this TU is globbed into _ops by `csrc/fused/sm_90/*.cu` and owns NO
// PYBIND11_MODULE — so it is NOT dropped by setup.py's content-based filter
// (which removes only standalone-JIT cell drivers that define their own module)
// and does NOT collide on PyInit__ops. No glob edit needed.
//
// PRECISION / PARITY: fp32 throughout (matches csa_hca_step_one's fp32 compute
// and the parity-validated apply). Bit-faithful to the per-op math up to fp32
// reduction-order round-off in the hand-written GRU/PEER dots (the documented
// 1e-5 parity hotspots — see the .cuh header). NOT bit-identity.

#include <cuda_runtime.h>
#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "csrc/fused/sm_90/opt_stage_supergrok2.cuh"

namespace sg {

// Compile-time dims = supergrok2.py constructor defaults + OPTIMIZER_CONFIGS
// ["supergrok2"] (d_model=8, num_heads=2, gru_hidden=4, num_experts=144, ...).
// The race uses these defaults; a config with different shapes would template-
// specialize the launcher (compile-time dims). Mirrored in the Python driver +
// sg2_ws_stride below so the host packing/stride and the kernel carve agree.
using SG2DimsDefault = ::sg::fused::sm90::SG2Dims<>;

// ── sg2_ws_stride: the AUTHORITATIVE floats-per-CTA workspace stride. The
//    Python driver calls this (not a duplicated py formula) so the host
//    allocation and the kernel's sg2_carve_ws can never drift. ───────────────
int64_t sg2_ws_stride(int64_t Nmax) {
    return ::sg::fused::sm90::sg2_ws_stride<SG2DimsDefault>(Nmax);
}

// ── sg2_meta_optimizer_tail: the §1b host wrapper. Marshals the packed flat
//    buffers, the (fp32) meta weight bundle, the per-tensor + shared scalars,
//    and the persistent-context scratch into the POD structs, then launches the
//    ONE persistent kernel that runs CSA/HCA/GRU/PEER/apply for ALL tensors.
//    The argsort (perm/unsort) is PRE-COMPUTED by the driver (the one explicit
//    pre-kernel step — honesty rail #5). ──────────────────────────────────────
void sg2_meta_optimizer_tail(
    torch::Tensor params_packed,       // [total]   float32, all tensors back-to-back
    torch::Tensor grads_packed,        // [total]   float32
    torch::Tensor sharpness_packed,    // [total]   float32
    torch::Tensor exp_avg_packed,      // [total]   float32  (Adam m)
    torch::Tensor exp_avg_sq_packed,   // [total]   float32  (Adam v)
    torch::Tensor mu_packed,           // [total]   float32  (expert-output EMA)
    torch::Tensor slow_packed,         // [total]   float32  (grokfast slow EMA)
    torch::Tensor gru_state_packed,    // [total*gru_hidden] float32, row-major
    torch::Tensor perm_packed,         // [total]   int32  sorted-row -> original-row
    torch::Tensor unsort_packed,       // [total]   int32  original-row -> sorted-row
    torch::Tensor n_per_tensor,        // [P]       int32  element count of each tensor
    torch::Tensor row_off,             // [P]       int64  start element of each tensor
    torch::Tensor workspace,           // [n_ctas * ws_stride] float32 (kernel scratch)
    int64_t ws_stride,                 // floats per CTA (== sg2_ws_stride(Nmax))
    // ── meta weight bundle (fp32; upcast once by the driver) ──
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W, torch::Tensor csa_out_W,
    torch::Tensor csa_compress_w, torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_K,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W, torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr, torch::Tensor gru_br,
    torch::Tensor gru_Wh, torch::Tensor gru_bh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1, torch::Tensor expert_W2, torch::Tensor expert_b2,
    // ── per-tensor scalars (length P) + shared scalars ──
    torch::Tensor alpha,        // [P] float  mu/slow mixing + slow-EMA decay (alpha_i)
    torch::Tensor gru_decay,    // [P] float  expert-EMA (mu) decay (== beta1_i)
    torch::Tensor lamb_eff,     // [P] float  grokfast amplification (lamb·ramp·gate)
    torch::Tensor beta1,        // [P] float  Adam beta1 (layer-scaled)
    torch::Tensor bc1,          // [P] float  1 - beta1_i^t
    torch::Tensor bc2,          // [P] float  1 - beta2^t
    double rescale,             // expert-output scale (shared)
    double beta2,               // shared
    double lr,                  // shared
    double wd,                  // shared  decoupled weight decay (wd_eff)
    double eps,                 // shared
    // ── persistent-context scratch (host-allocated, zero-init each launch) ──
    torch::Tensor g_next_task,  // int32   [1]  TaskQueue counter
    torch::Tensor g_arrived,    // int32   [1]  GridBarrier arrival count (reused as unsigned)
    torch::Tensor g_generation) // int32   [1]  GridBarrier generation (reused as unsigned)
{
    using namespace ::sg::fused::sm90;
    if (params_packed.numel() == 0) return;
    TORCH_CHECK(params_packed.scalar_type() == at::kFloat,
                "sg2_meta_optimizer_tail: fp32 packed buffers in this pass");
    TORCH_CHECK(perm_packed.scalar_type() == at::kInt &&
                unsort_packed.scalar_type() == at::kInt,
                "sg2_meta_optimizer_tail: perm/unsort must be int32");
    TORCH_CHECK(row_off.scalar_type() == at::kLong,
                "sg2_meta_optimizer_tail: row_off must be int64");
    TORCH_CHECK(n_per_tensor.scalar_type() == at::kInt,
                "sg2_meta_optimizer_tail: n_per_tensor must be int32");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int P = (int)n_per_tensor.numel();

    SG2Weights w {
        input_proj_W.data_ptr<float>(), input_proj_b.data_ptr<float>(),
        csa_q_W.data_ptr<float>(), csa_k_W.data_ptr<float>(), csa_v_W.data_ptr<float>(),
        csa_out_W.data_ptr<float>(), csa_compress_w.data_ptr<float>(),
        csa_idx_DQ.data_ptr<float>(), csa_idx_K.data_ptr<float>(),
        hca_q_W.data_ptr<float>(), hca_k_W.data_ptr<float>(), hca_v_W.data_ptr<float>(),
        hca_out_W.data_ptr<float>(),
        gru_Wz.data_ptr<float>(), gru_bz.data_ptr<float>(),
        gru_Wr.data_ptr<float>(), gru_br.data_ptr<float>(),
        gru_Wh.data_ptr<float>(), gru_bh.data_ptr<float>(),
        peer_query_Ws.data_ptr<float>(), prod_keys_A.data_ptr<float>(),
        prod_keys_B.data_ptr<float>(),
        expert_W1.data_ptr<float>(), expert_b1.data_ptr<float>(),
        expert_W2.data_ptr<float>(), expert_b2.data_ptr<float>() };

    SG2State st {
        exp_avg_packed.data_ptr<float>(), exp_avg_sq_packed.data_ptr<float>(),
        mu_packed.data_ptr<float>(), slow_packed.data_ptr<float>(),
        gru_state_packed.data_ptr<float>(),
        perm_packed.data_ptr<int>(), unsort_packed.data_ptr<int>(),
        workspace.data_ptr<float>(), (int64_t)ws_stride, P,
        n_per_tensor.data_ptr<int>(), row_off.data_ptr<int64_t>() };

    SG2Scalars sc {
        alpha.data_ptr<float>(), gru_decay.data_ptr<float>(), lamb_eff.data_ptr<float>(),
        beta1.data_ptr<float>(), bc1.data_ptr<float>(), bc2.data_ptr<float>(),
        (float)rescale, (float)beta2, (float)lr, (float)wd, (float)eps };

    ::sg::fused::PersistentContext ctx {};  // PersistentContext lives in sg::fused (parent of sm90)
    ctx.g_next_task  = g_next_task.data_ptr<int>();
    ctx.g_arrived    = reinterpret_cast<unsigned*>(g_arrived.data_ptr<int>());     // int32 buffer reused
    ctx.g_generation = reinterpret_cast<unsigned*>(g_generation.data_ptr<int>());
    ctx.n_tasks      = P;
    // n_ctas is set inside the launcher (== #SMs).

    // NOTE: bind to a temporary FIRST. The template arg list's commas would
    // otherwise be parsed by the preprocessor as extra C10_CUDA_CHECK macro
    // arguments ("passed 3 arguments, but takes just 1").
    cudaError_t launch_err =
        launch_sg2_meta_optimizer_tail<SG2DimsDefault, float, float>(
            ctx, w, st, sc,
            params_packed.data_ptr<float>(), grads_packed.data_ptr<float>(),
            sharpness_packed.data_ptr<float>(), stream);
    C10_CUDA_CHECK(launch_err);
}

}  // namespace sg
