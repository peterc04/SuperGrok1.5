// csrc/fused/sm_90/mega_mamba_real_adamw_tc_launcher.cu — cycle-2 WIRED tensor-core
// host-launcher TU for the TRUE L3 fused (mamba3 × {adamw,lion,grokfast}) megakernel.
//
// Twin of mega_{decoder,vit}_real_adamw_tc_launcher.cu for Mamba (see the decoder
// file's header for the full rationale). The standalone TC driver
// mega_mamba_real_adamw_tc.cu owns its own PYBIND11_MODULE and is dropped from _ops
// by setup.py's content-based glob filter, so the race path (dispatch.cpp →
// _try_fused_train_step) had NO way to reach the mamba TC kernel — bf16 mamba race
// configs ran the scalar L3 megakernel (the measured 905a4bb carve-out, TC 0.46×).
//
// This TU is the missing seam: it is globbed INTO _ops (no own pybind module), is
// compiled with -DSG_TUNED_GEMM_IMPL=1 (the in-source #define below) so it links the
// wgmma branch of fused_mamba_megakernel.cuh, and exposes ONE non-template host
// launcher whose boundary is plain pointers/ints + the FusedScalars POD — NO
// header-only types (MambaTokenCtx / FusedOptState) cross the boundary — so
// dispatch.cpp can extern-declare it with only the mirror structs it already has. It
// calls ONLY launch_fused_mamba_megakernel_tc (the TC launcher); it never
// instantiates the scalar launcher template, so it CANNOT collide (ODR/duplicate-
// symbol) with the scalar mega_mamba_real_adamw.cu TU even though both are in _ops.
// The TC activations/grad workspace is sized DIFFERENTLY from the scalar
// MambaScratch partials (mb_tc_workspace_floats), so this TU owns its own cached
// scratch internally — dispatch.cpp passes no workspace for the TC path.
//
// WHY MAMBA IS NOW ON THE TC PATH (owner cycle-2 directive (c)): mamba×adamw was
// excluded from _L3_WGMMA_CELLS as a measured scalar-WINS carve-out (0.46×, the
// selective-scan/conv1d dominate, not the projection GEMMs). That is a PERFORMANCE
// fact the roofline surfaces — NOT a correctness reason: the mamba TC kernel is
// validated 5/5 by tests/hw/test_mamba_tc.py. The directive lists mamba conversion
// explicitly, and no-suppression cuts toward exposing the path + letting the roofline
// report the (honest) 0.46×, rather than hiding a correct kernel. The scalar TU
// (mega_mamba_real_adamw.cu) is UNTOUCHED and remains the fp32 path; mamba×adamw at
// fp32 still routes there (gemm_impl_for_cell returns "scalar" for fp32).
//
// OPTID-GENERIC: identical to the decoder/vit TC launchers — the wgmma fwd+bwd is
// optimizer-independent, only the per-element tail (apply_optimizer<Opt>) differs.
// `opt_id` selects the kernel instantiation over the SAME [m|v|extra]+loss state
// buffer (ema = extra slice for grokfast; the neuralgrok psi-net pack would live at
// the head of extra). The STAGED/model-coupled/SG2 optimizers are NOT cases (no
// single-launch TC path); dispatch.cpp gates them out and an unsupported id returns
// cudaErrorInvalidValue → dispatch throws LOUD (no silent scalar/adamw fallback).
//
// HONESTY: this is the REAL bf16 wgmma path (HGMMA in SASS for the 4 projection
// GEMMs; the scan/conv stay scalar by design — REUSED verbatim), validated by
// tests/hw/test_mamba_tc.py.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver (Fork-B TC branch)

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/fused/sm_90/fused_mamba_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

namespace {
// Persistent barrier/queue counters + the TC activations/grad workspace, cached
// per device. Mirrors mega_{decoder,vit}_real_adamw_tc_launcher.cu's scratch but
// for mamba's workspace shape. Raw cudaMalloc (process-lived scratch, recreated
// only when a larger B needs a bigger workspace; never freed — one per device,
// leaked at process exit like every cuBLAS/cuDNN handle).
struct MbTcLauncherScratch {
    int*      g_next = nullptr;        // int [1]
    unsigned* g_arrived = nullptr;     // unsigned [1]
    unsigned* g_generation = nullptr;  // unsigned [1]
    float*    workspace = nullptr;     // float [mb_tc_workspace_floats(T, nCTA)]
    int64_t   ws_floats = 0;
    int       dev = -1;
};

MbTcLauncherScratch& mb_tc_launcher_scratch(int dev, int64_t need_floats) {
    static MbTcLauncherScratch s;
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

// mega_mamba_real_adamw_tc — the WIRED Mamba TC launcher. Boundary mirrors the scalar
// mega_mamba_real_adamw (dispatch.cpp extern-decls it with the same mirror types)
// EXCEPT it takes NO workspace pointer: the TC workspace is a different size from the
// scalar partials, so this TU allocates+owns it. `loss_out` still points into the
// caller's state[3*total] slot (dispatch reads it back). `opt_id` selects the tail.
//
// ncta_cap: forwarded to the TC launcher (0 = one CTA/SM = full saturation, the
// shipped config). The race always passes 0.
cudaError_t mega_mamba_real_adamw_tc(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap, int opt_id) {
    const int64_t total = kMambaTotalElems;
    const int T = B * mb::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int nCTA = n_sms;
    if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap;

    const int64_t need = mb_tc_workspace_floats(T, nCTA);
    MbTcLauncherScratch& sc = mb_tc_launcher_scratch(dev, need);

    // Re-point the ctx counters at THIS TU's cached counters (the TC launcher zeroes
    // them per-launch, but they must be the ones this scratch owns so the grid
    // barrier is consistent with the workspace).
    ctx.g_next_task  = sc.g_next;
    ctx.g_arrived    = sc.g_arrived;
    ctx.g_generation = sc.g_generation;
    ctx.n_tasks      = kMambaNumTensors;   // parameter tensors for reduce+opt phases

    // Bind the optimizer state slices. [m | v | extra] over the SAME state buffer;
    // each tail reads only its own buffers (opt_components.cuh guarantees unused
    // pointers are never dereferenced). ema/psi-net live in the `extra` slice.
    float* const extra_slice = state + 2 * total;
    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    st.ema        = extra_slice;          // grokfast/grokadamw slow-grad EMA
    st.psi_W1     = extra_slice + kPsiW1Off;   // neuralgrok psi-net pack (when wired)
    st.psi_b1     = extra_slice + kPsiB1Off;
    st.psi_W2     = extra_slice + kPsiW2Off;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    MambaTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = sc.workspace;
    tok.loss_out  = loss_out;

    // Dispatch to the matching kernel instantiation. The fwd+bwd is identical across
    // these OptIds; the template differs ONLY in the P3 apply_optimizer<Opt> tail.
    // Unsupported opt_ids return cudaErrorInvalidValue → dispatch.cpp throws LOUD.
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return launch_fused_mamba_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return launch_fused_mamba_megakernel_tc<OptId::Lion>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return launch_fused_mamba_megakernel_tc<OptId::Grokfast>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;  // STAGED/coupled opt not single-launch TC
    }
}

}}}  // namespace sg::fused::sm90
