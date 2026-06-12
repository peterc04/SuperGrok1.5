// csrc/fused/sm_90/mega_vit_real_adamw_tc.cu — R2 TENSOR-CORE ViT cell driver
// TU (DESIGN-TC-PIPELINE.md Fork B, the ViT twin of mega_decoder_real_adamw_tc
// .cu). The bf16 wgmma fwd+bwd+AdamW persistent megakernel, compiled with
// -DSG_TUNED_GEMM_IMPL=1 so the wgmma branch of fused_vit_megakernel.cuh is
// selected (the scalar default TU mega_vit_real_adamw.cu is UNTOUCHED and ships
// as the live path).
//
// BUILD-ONLY extension of the build path: this is a SECOND compiled TU for the
// SAME (vit × adamw) cell, selected by the GEMM impl token. It is NOT wired into
// compile.py / dispatch — the parity gate (tests/hw/test_vit_tc.py PART 2)
// JIT-loads it via cpp_extension.load and drives it directly through the pybind
// entries below. It owns its OWN PYBIND11_MODULE(TORCH_EXTENSION_NAME, …), so
// setup.py's CONTENT filter (_owns_extension_module) AUTO-EXCLUDES it from the
// shipped _ops glob (no setup.py change needed — exactly the decoder TC TU
// pattern).
//
// HONESTY: this exercises the REAL tensor-core path (HGMMA in SASS). No scalar
// fallback, no surrogate.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"

namespace vit = ::sg::fused::sm90::vit;
namespace vittc = ::sg::fused::sm90::vittc;
namespace wgs = ::sg::sm90::wgs;
using ::sg::fused::PersistentContext;
using ::sg::fused::sm90::ViTInputCtx;
using ::sg::fused::sm90::FusedOptState;
using ::sg::fused::sm90::OptId;
using ::sg::fused::sm90::kVitTotalElems;
using ::sg::fused::sm90::kVitNumTensors;

// ════════════════════════════════════════════════════════════════════════════
//  PART 1 — GEMM-orientation micro-gates (the SAME engine the decoder selftest
//  silicon-gated 13/13; the engine is model-agnostic, but exercising it through
//  a ViT TU at the ViT TILE_M=1088 (17 stacked m64 atoms) is the honest witness
//  for the ViT path's M-atom stacking). Three orientations: fwd / dX / dW, each
//  one tiny GEMM on a single CTA (256 thr) returning the fp32 result. ──────────
// ════════════════════════════════════════════════════════════════════════════

// Ring smem: kVitAtomsPerSlot A(64×16) tiles [the M-atom interleave group] + one
// B(N=128×16) tile. Without the kVitAtomsPerSlot factor the interleaved engine
// stages the group's 2nd A-tile out of bounds (illegal access).
static size_t arena_bytes() {
    return (size_t)(vittc::kVitAtomsPerSlot * 64 * 16 + 128 * 16) * sizeof(__nv_bfloat16) + 256;
}

// (fwd) Y[M,Nout] = X[M,Kin] @ W[Nout,Kin]^T. M = TILE_M (17 atoms). N-tiled.
template <int N>
__global__ void __launch_bounds__(256)
gemm_fwd_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                float* __restrict__ D, int Kin, int Nout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem; __nv_bfloat16* sB = sA + vittc::kVitAtomsPerSlot * 64 * 16;
    const int M = vittc::kTileM, m_atoms = M / 64, k_steps = Kin / 16;
    for (int n0 = 0; n0 < Nout; n0 += N) {
        const int n_real = (Nout - n0) < N ? (Nout - n0) : N;
        auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return X[(int64_t)m * Kin + k]; };
        auto srcB = [&] (int n, int k) -> __nv_bfloat16 {
            int nn = n0 + n; return nn < Nout ? W[(int64_t)nn * Kin + k] : __float2bfloat16(0.f); };
        auto out  = [&] (int m, int n, float v) { if (m < M) D[(int64_t)m * Nout + n0 + n] = v; };
        vittc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/vittc::kAtomsM>(
            0, m_atoms, n_real, k_steps, srcA, srcB, out, sA, sB);
    }
#else
    (void)X; (void)W; (void)D; (void)Kin; (void)Nout;
#endif
}

// (dX) dX[m,kin] = Σ_out dY[m,out]·W[out,kin]. N(wgmma)=Kin, K=Nout, W transposed.
template <int N>
__global__ void __launch_bounds__(256)
gemm_dx_kernel(const __nv_bfloat16* __restrict__ dY, const __nv_bfloat16* __restrict__ W,
               float* __restrict__ dX, int Nout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem; __nv_bfloat16* sB = sA + vittc::kVitAtomsPerSlot * 64 * 16;
    const int M = vittc::kTileM, m_atoms = M / 64, k_steps = Nout / 16;
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 { return dY[(int64_t)m * Nout + k]; };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 { return W[(int64_t)k * N + n]; };
    auto out  = [&] (int m, int n, float v) { if (m < M) dX[(int64_t)m * N + n] = v; };
    vittc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/vittc::kAtomsM>(
        0, m_atoms, /*n_real=*/N, k_steps, srcA, srcB, out, sA, sB);
#else
    (void)dY; (void)W; (void)dX; (void)Nout;
#endif
}

// (dW) dW[out,in] = Σ_t dY[t,out]·X[t,in]. BOTH transposed, K=T multi-k-step.
template <int N>
__global__ void __launch_bounds__(256)
gemm_dw_kernel(const __nv_bfloat16* __restrict__ dY, const __nv_bfloat16* __restrict__ X,
               float* __restrict__ dW, int Nout, int T) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sA = smem; __nv_bfloat16* sB = sA + vittc::kVitAtomsPerSlot * 64 * 16;
    const int k_steps = T / 16, m_atoms = (Nout + 63) / 64;   // up to 8 for Nout=512
    auto srcA = [&] (int m, int k) -> __nv_bfloat16 {
        return m < Nout ? dY[(int64_t)k * Nout + m] : __float2bfloat16(0.f); };
    auto srcB = [&] (int n, int k) -> __nv_bfloat16 { return X[(int64_t)k * N + n]; };
    auto out  = [&] (int m, int n, float v) { if (m < Nout) dW[(int64_t)m * N + n] = v; };
    vittc::tc_gemm_block_unpipelined<N, /*MaxAtomsM=*/8>(
        0, m_atoms, /*n_real=*/N, k_steps, srcA, srcB, out, sA, sB);
#else
    (void)dY; (void)X; (void)dW; (void)Nout; (void)T;
#endif
}

template <int N>
static torch::Tensor run_fwd(torch::Tensor X, torch::Tensor W, int Kin, int Nout) {
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && W.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.is_contiguous() && W.is_contiguous());
    auto D = torch::zeros({vittc::kTileM, Nout}, torch::dtype(torch::kFloat32).device(X.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_fwd_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_fwd_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), D.data_ptr<float>(), Kin, Nout);
    C10_CUDA_CHECK(cudaGetLastError());
    return D;
}
template <int N>
static torch::Tensor run_dx(torch::Tensor dY, torch::Tensor W, int Nout) {
    auto dX = torch::zeros({vittc::kTileM, N}, torch::dtype(torch::kFloat32).device(dY.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_dx_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_dx_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), dX.data_ptr<float>(), Nout);
    C10_CUDA_CHECK(cudaGetLastError());
    return dX;
}
template <int N>
static torch::Tensor run_dw(torch::Tensor dY, torch::Tensor X, int Nout, int T) {
    auto dW = torch::zeros({Nout, N}, torch::dtype(torch::kFloat32).device(dY.device()));
    size_t smem = arena_bytes();
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(gemm_dw_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_dw_kernel<N><<<1, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()), dW.data_ptr<float>(), Nout, T);
    C10_CUDA_CHECK(cudaGetLastError());
    return dW;
}

static torch::Tensor gemm_fwd(torch::Tensor X, torch::Tensor W, int N, int Kin, int Nout) {
    switch (N) { case 96: return run_fwd<96>(X, W, Kin, Nout); case 128: return run_fwd<128>(X, W, Kin, Nout);
                 default: TORCH_CHECK(false, "fwd N ", N); }
}
static torch::Tensor gemm_dx(torch::Tensor dY, torch::Tensor W, int N, int Nout) {
    switch (N) { case 96: return run_dx<96>(dY, W, Nout); case 128: return run_dx<128>(dY, W, Nout);
                 default: TORCH_CHECK(false, "dx N ", N); }
}
static torch::Tensor gemm_dw(torch::Tensor dY, torch::Tensor X, int N, int Nout, int T) {
    switch (N) { case 96: return run_dw<96>(dY, X, Nout, T); case 128: return run_dw<128>(dY, X, Nout, T);
                 default: TORCH_CHECK(false, "dw N ", N); }
}

// ════════════════════════════════════════════════════════════════════════════
//  PART 2 — full-cell driver. One persistent set of barrier/queue counters + the
//  TC workspace, cached per device. Re-created if a larger B requires a larger
//  workspace.
// ════════════════════════════════════════════════════════════════════════════
struct TcScratch {
    torch::Tensor g_next, g_arrived, g_generation, workspace;
    int n_ctas = 0;
    int64_t ws_floats = 0;
};

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
    const int T = B * vit::kSeq;
    const int64_t need = ::sg::fused::sm90::vit_tc_workspace_floats(T, B, nCTA);
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
// patches [B, kNPatch, kPatch] fp32; targets [B] int32; state [3*total] fp32.
// Returns {loss (cpu float scalar), grad [total] fp32 (the reduced grad)}.
static std::vector<torch::Tensor> tc_train_step(
        torch::Tensor params, torch::Tensor patches, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step,
        int64_t ncta_cap) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kVitTotalElems, "params must be the flat ViT blob (", kVitTotalElems, ")");
    TORCH_CHECK(patches.is_cuda() && patches.scalar_type() == torch::kFloat32 && patches.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(patches.numel() == (int64_t)B * vit::kNPatch * vit::kPatch,
                "patches must be [B, kNPatch, kPatch]");
    TORCH_CHECK((B % 16) == 0, "TC path requires B %% 16 == 0 (dW K-loop is 16-step atoms)");
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
        kVitNumTensors,
        0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m;
    st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    ViTInputCtx in;
    in.patches = patches.data_ptr<float>();
    in.targets = targets.data_ptr<int>();
    in.B = B;
    in.workspace = sc.workspace.data_ptr<float>();
    in.loss_out = loss_dev.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_vit_megakernel_tc<OptId::AdamW>(
        ctx, params.data_ptr<float>(), in, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream, nCTA);
    TORCH_CHECK(err == cudaSuccess, "TC megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return {loss_dev.to(torch::kCPU), grad};
}

// CALIBRATION HOOK (test_vit_tc.py::test_tc_dw_gemm_exact_on_own_operands): slice
// the kernel's OWN stored bf16 acts dY_ff2[L1] [T,d] and X_gact[L1] [T,dff] from
// the workspace, returned as fp32 CPU tensors. The gate contracts them in fp32
// ascending-t and compares to the kernel's ff2.weight grad slice — isolating the
// dW GEMM (incl. the ONLY Kin=dff multi-N-tile path) from operand-chain bf16
// divergence. Re-uses the cached workspace from the last tc_train_step.
static std::vector<torch::Tensor> tc_dump_ff2_operands(torch::Tensor params, int64_t B) {
    TcScratch& sc = tc_scratch_for(params, (int)B, tc_effective_nctas(params, 4));
    const int T = (int)B * vit::kSeq;
    const int64_t d = vit::kD, dff = vit::kDff;
    const int Tp = (int)B * vit::kNPatch;
    auto acts_bf16 = sc.workspace.slice(0, 0, ::sg::fused::sm90::vit_tc_acts_floats(T, (int)B))
                         .view(torch::kBFloat16);
    // mirror vit_acts_bind: walk to dY_ff2[1] and X_gact[1].
    const int64_t Td = (int64_t)T * d, T3d = (int64_t)T * 3 * d, Tff = (int64_t)T * dff;
    int64_t off = (int64_t)Tp * vit::kPatch;    // skip X_patch
    int64_t off_Xgact1 = -1, off_dYff2_1 = -1;
    for (int li = 0; li < vit::kLayers; ++li) {
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

#if SG_VIT_SCALAR_MEGAKERNEL
// scalar_train_step: the SAME (vit × AdamW) cell run through the SCALAR default
// launcher (launch_fused_vit_megakernel<AdamW>, which lives OUTSIDE the wgmma
// guard — so this wgmma-token TU can still call it). ONLY for the step-time gate
// (times the live scalar path and the TC path BACK-TO-BACK in one process → an
// honest scalar:TC ratio). NOT the shipped invocation. Returns {loss, reduced
// grad} like tc_train_step. GATED by SG_VIT_SCALAR_MEGAKERNEL: at the d-scaled
// bench width the scalar megakernel's VitSampleSmem overflows the smem cap and is
// compiled out (the TC path is what the bench measures), exactly as the decoder
// bench gates its scalar step.
static std::vector<torch::Tensor> scalar_train_step(
        torch::Tensor params, torch::Tensor patches, torch::Tensor targets,
        torch::Tensor state, double lr, double beta1, double beta2, double eps,
        double weight_decay, double bc1, double bc2, int64_t step) {
    TORCH_CHECK(params.is_cuda() && params.scalar_type() == torch::kFloat32 && params.is_contiguous());
    TORCH_CHECK(params.numel() == kVitTotalElems, "params must be the flat ViT blob (", kVitTotalElems, ")");
    TORCH_CHECK(patches.is_cuda() && patches.scalar_type() == torch::kFloat32 && patches.is_contiguous());
    TORCH_CHECK(targets.is_cuda() && targets.scalar_type() == torch::kInt32 && targets.is_contiguous());
    const int B = (int)targets.numel();
    TORCH_CHECK(patches.numel() == (int64_t)B * vit::kNPatch * vit::kPatch, "patches must be [B, kNPatch, kPatch]");
    const int64_t total = kVitTotalElems;
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
        kVitNumTensors, 0u};

    FusedOptState st;
    float* m = state.data_ptr<float>();
    st.exp_avg = m; st.exp_avg_sq = m + total;
    st.lr = (float)lr; st.beta1 = (float)beta1; st.beta2 = (float)beta2;
    st.eps = (float)eps; st.wd = (float)weight_decay;
    st.bc1 = (float)bc1; st.bc2 = (float)bc2;

    ViTInputCtx in;
    in.patches = patches.data_ptr<float>();
    in.targets = targets.data_ptr<int>();
    in.B = B;
    in.workspace = s_ws.data_ptr<float>();
    in.loss_out = loss_out;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = ::sg::fused::sm90::launch_fused_vit_megakernel<OptId::AdamW>(
        ctx, params.data_ptr<float>(), in, grad.data_ptr<float>(),
        (float)lr, (int)step, st, stream);
    TORCH_CHECK(err == cudaSuccess, "scalar megakernel launch failed: ", cudaGetErrorString(err));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto loss_cpu = s_ws.slice(0, (int64_t)n_sms * total + n_sms, (int64_t)n_sms * total + n_sms + 1)
                        .to(torch::kCPU);
    return {loss_cpu, grad};
}
#endif  // SG_VIT_SCALAR_MEGAKERNEL

#ifdef SG_VIT_PROFILE
// Diagnostic-only: read + reset the per-phase clock64 maxima (cycles). Returns
// [P1_fwd, P1_bwd, P2_dW, P3_opt]. Call AFTER one tc_train_step; divide by the
// SM clock to convert to ms (or just read the RATIOS — phase attribution).
static std::vector<int64_t> tc_profile_read() {
    unsigned long long h[6] = {0, 0, 0, 0, 0, 0};
    cudaMemcpyFromSymbol(h, ::sg::fused::sm90::g_vit_prof_max, sizeof(h));
    unsigned long long z[6] = {0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(::sg::fused::sm90::g_vit_prof_max, z, sizeof(z));
    return {(int64_t)h[0], (int64_t)h[1], (int64_t)h[2],
            (int64_t)h[3], (int64_t)h[4], (int64_t)h[5]};
}
// per-sample head/CE loop cost (fwd[0], bwd[1]) — a SUBSET of P1_fwd/P1_bwd.
static std::vector<int64_t> tc_profile_head_read() {
    unsigned long long h[2] = {0, 0};
    cudaMemcpyFromSymbol(h, ::sg::fused::sm90::vittc::g_vit_prof_head, sizeof(h));
    unsigned long long z[2] = {0, 0};
    cudaMemcpyToSymbol(::sg::fused::sm90::vittc::g_vit_prof_head, z, sizeof(z));
    return {(int64_t)h[0], (int64_t)h[1]};
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
    // PART 1 GEMM-orientation micro-gates (engine, ViT TILE_M).
    mm.def("gemm_fwd", &gemm_fwd, "TC fwd Y=X@W^T (TILE_M x N), bf16/fp32acc");
    mm.def("gemm_dx",  &gemm_dx,  "TC dX=dY@W (W transposed-staged), bf16/fp32acc");
    mm.def("gemm_dw",  &gemm_dw,  "TC dW=dY^T@X (both transposed-staged, K=T), bf16/fp32acc");
    // PART 2 full-cell driver.
    mm.def("tc_train_step", &tc_train_step,
           "Fork-B tensor-core ViT fwd+bwd+AdamW step (in place); returns (loss, reduced grad)",
           pybind11::arg("params"), pybind11::arg("patches"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"),
           pybind11::arg("ncta_cap") = 0);
    mm.def("tc_dump_ff2_operands", &tc_dump_ff2_operands,
           "gate-only: dump the kernel's stored dY_ff2[L1], X_gact[L1] (fp32 CPU)");
#if SG_VIT_SCALAR_MEGAKERNEL
    mm.def("scalar_train_step", &scalar_train_step,
           "gate-only: SAME cell via the SCALAR launcher (back-to-back step-time mirror)",
           pybind11::arg("params"), pybind11::arg("patches"), pybind11::arg("targets"),
           pybind11::arg("state"), pybind11::arg("lr"), pybind11::arg("beta1"),
           pybind11::arg("beta2"), pybind11::arg("eps"), pybind11::arg("weight_decay"),
           pybind11::arg("bc1"), pybind11::arg("bc2"), pybind11::arg("step"));
#endif  // SG_VIT_SCALAR_MEGAKERNEL
    mm.attr("TILE_M") = (int)vittc::kTileM;
    mm.attr("TILE_N") = (int)SG_TUNED_TILE_N;
    mm.attr("TOTAL") = (int)kVitTotalElems;
    // Compiled-width introspection (the d-scaled bench asserts/reports these).
    mm.attr("D") = (int)vit::kD;
    mm.attr("HEADS") = (int)vit::kHeads;
    mm.attr("LAYERS") = (int)vit::kLayers;
    mm.attr("SEQ") = (int)vit::kSeq;
    mm.attr("NPATCH") = (int)vit::kNPatch;
    mm.attr("PATCH") = (int)vit::kPatch;
    mm.attr("VOCAB") = (int)vit::kVocab;
#ifdef SG_VIT_PROFILE
    mm.def("tc_profile_read", &tc_profile_read,
           "diagnostic-only: per-phase clock64 maxima [P1fwd,P1bwd,P2dW,P3opt] (cycles), resets after read");
    mm.def("tc_profile_head_read", &tc_profile_head_read,
           "diagnostic-only: per-sample head/CE loop cycles [fwd,bwd] (subset of P1), resets after read");
    mm.attr("HAS_PROFILE") = true;
#endif
}
