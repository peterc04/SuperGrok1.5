// csrc/fused/sm_90/mega_vit_real_adamw_tc_launcher.cu — R2.4 WIRED tensor-core
// host-launcher TU for the TRUE L3 fused (vit × adamw) megakernel.
//
// Twin of mega_decoder_real_adamw_tc_launcher.cu for ViT (see that file's header
// for the full rationale). The shipped scalar launcher mega_vit_real_adamw.cu
// compiles the SCALAR cell only; the Fork-B TC driver mega_vit_real_adamw_tc.cu
// owns its own PYBIND11_MODULE and is dropped from _ops. This TU is the missing
// seam: globbed INTO _ops (no own pybind module), compiled -DSG_TUNED_GEMM_IMPL=1
// so it links the wgmma branch of fused_vit_megakernel.cuh, exposing ONE
// non-template host launcher with a plain pointers/ints + FusedScalars boundary
// (dispatch.cpp extern-decls it with the mirror structs). Calls ONLY
// launch_fused_vit_megakernel_tc → never instantiates the scalar launcher
// template → CANNOT collide with the scalar TU in _ops. Owns its own TC-sized
// activations/grad workspace (vit_tc_workspace_floats != the scalar nCTA*total
// partials), so dispatch.cpp passes no workspace for the TC path.
//
// ViT input is FLOAT image patches with int targets (dispatch.cpp bit-reinterprets
// the targets out of the trailing float slots and passes a real const int* here).
//
// HONESTY: the REAL bf16 wgmma path (HGMMA in SASS, validated by
// tests/hw/test_vit_tc.py — 21/21 gates). The scalar TU is UNTOUCHED.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver (Fork-B TC branch)

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/fused/sm_90/fused_vit_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

namespace {
struct VitTcLauncherScratch {
    int*      g_next = nullptr;
    unsigned* g_arrived = nullptr;
    unsigned* g_generation = nullptr;
    float*    workspace = nullptr;
    int64_t   ws_floats = 0;
    int       dev = -1;
};

VitTcLauncherScratch& vit_tc_launcher_scratch(int dev, int64_t need_floats) {
    static VitTcLauncherScratch s;
    if (s.dev != dev) {
        s.dev = dev;
        if (!s.g_next)       cudaMalloc(&s.g_next, sizeof(int));
        if (!s.g_arrived)    cudaMalloc(&s.g_arrived, sizeof(unsigned));
        if (!s.g_generation) cudaMalloc(&s.g_generation, sizeof(unsigned));
    }
    if (s.ws_floats < need_floats) {
        if (s.workspace) cudaFree(s.workspace);
        cudaMalloc(&s.workspace, (size_t)need_floats * sizeof(float));
        s.ws_floats = need_floats;
    }
    return s;
}
}  // anonymous namespace

// mega_vit_real_adamw_tc — the WIRED ViT TC launcher. Boundary mirrors the scalar
// mega_vit_real_adamw EXCEPT it takes NO workspace pointer (TC workspace is a
// different size; this TU owns it). `loss_out` points into the caller's
// state[3*total] slot.
cudaError_t mega_vit_real_adamw_tc(
        PersistentContext ctx, float* params,
        const float* patches, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap) {
    const int64_t total = kVitTotalElems;
    const int T = B * vit::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int nCTA = n_sms;
    if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap;

    const int64_t need = vit_tc_workspace_floats(T, B, nCTA);
    VitTcLauncherScratch& sc = vit_tc_launcher_scratch(dev, need);

    ctx.g_next_task  = sc.g_next;
    ctx.g_arrived    = sc.g_arrived;
    ctx.g_generation = sc.g_generation;
    ctx.n_tasks      = kVitNumTensors;

    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    apply_scalars(st, scalars);
    st.lr = lr;

    ViTInputCtx in;
    in.patches   = patches;
    in.targets   = targets;
    in.B         = B;
    in.workspace = sc.workspace;
    in.loss_out  = loss_out;

    return launch_fused_vit_megakernel_tc<OptId::AdamW>(
        ctx, params, in, grad, lr, step, st, stream, nCTA);
}

}}}  // namespace sg::fused::sm90
