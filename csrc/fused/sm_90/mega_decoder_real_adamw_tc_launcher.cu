// csrc/fused/sm_90/mega_decoder_real_adamw_tc_launcher.cu — R2.4 WIRED tensor-core
// host-launcher TU for the TRUE L3 fused (transformer_decoder × adamw) megakernel.
//
// WHY THIS TU EXISTS (the GEMM_IMPL=wgmma wiring, owner directive task 1): the
// shipped scalar launcher mega_decoder_real_adamw.cu compiles the SCALAR cell only
// (its own #error refuses SG_TUNED_GEMM_IMPL=wgmma — no-suppression guard, never
// links a scalar body under a wgmma name). The Fork-B tensor-core cell driver
// (mega_decoder_real_adamw_tc.cu) DOES build the wgmma branch but owns its own
// PYBIND11_MODULE, so setup.py's _collect() drops it from _ops (PyInit__ops
// collision). The race path (dispatch.cpp → _try_fused_train_step) therefore had
// NO way to reach the TC kernel — bf16 race configs silently ran eager.
//
// This TU is the missing seam: it is globbed INTO _ops (no own pybind module), is
// compiled with -DSG_TUNED_GEMM_IMPL=1 (per-TU; see setup.py's per-source nvcc
// flag rewrite) so it links the wgmma branch of fused_decoder_megakernel.cuh, and
// exposes ONE non-template host launcher whose boundary is plain pointers/ints +
// the FusedScalars POD — NO header-only types (DecoderTokenCtx/FusedOptState)
// cross the boundary — so dispatch.cpp can `extern`-declare it with only the
// mirror structs it already has. It calls ONLY launch_fused_decoder_megakernel_tc
// (the TC launcher); it never instantiates the scalar launcher template, so it
// CANNOT collide (ODR/duplicate-symbol) with the scalar TU even though both are in
// _ops. The TC activations/grad workspace is sized DIFFERENTLY from the scalar
// nCTA*total partials (dec_tc_workspace_floats), so this TU owns its own cached
// scratch internally — dispatch.cpp passes no workspace for the TC path.
//
// HONESTY: this is the REAL bf16 wgmma path (HGMMA in SASS, validated by
// tests/hw/test_decoder_tc.py). No scalar fallback, no surrogate. The scalar TU
// (mega_decoder_real_adamw.cu) is UNTOUCHED and remains the fp32 path.

#define SG_TUNED_GEMM_IMPL 1   // select the wgmma cell driver (Fork-B TC branch)

#include <cuda_runtime.h>
#include <cstdint>
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"

namespace sg { namespace fused { namespace sm90 {

namespace {
// Persistent barrier/queue counters + the TC activations/grad workspace, cached
// per device. Mirrors mega_decoder_real_adamw_tc.cu::TcScratch but uses raw
// cudaMalloc (this TU is inside _ops; it has c10 available, but raw allocation
// keeps the launcher free of any ATen-tensor lifetime questions on the hot path —
// the buffers are process-lived scratch, recreated only when a larger B needs a
// bigger workspace). Zeroed in-kernel (P0) each launch; never freed (one per
// device, leaked at process exit like every other cuBLAS/cuDNN handle).
struct DecTcLauncherScratch {
    int*     g_next = nullptr;        // int [1]
    unsigned* g_arrived = nullptr;    // unsigned [1]
    unsigned* g_generation = nullptr; // unsigned [1]
    float*   workspace = nullptr;     // float [dec_tc_workspace_floats(T,B,nCTA)]
    int64_t  ws_floats = 0;
    int      dev = -1;
};

DecTcLauncherScratch& dec_tc_launcher_scratch(int dev, int64_t need_floats) {
    static DecTcLauncherScratch s;
    if (s.dev != dev) {
        // First use on this device (or a device switch — the race is single-GPU,
        // but be honest if it ever moves): (re)allocate the counters.
        s.dev = dev;
        if (!s.g_next)      cudaMalloc(&s.g_next, sizeof(int));
        if (!s.g_arrived)   cudaMalloc(&s.g_arrived, sizeof(unsigned));
        if (!s.g_generation)cudaMalloc(&s.g_generation, sizeof(unsigned));
    }
    if (s.ws_floats < need_floats) {
        if (s.workspace) cudaFree(s.workspace);
        cudaMalloc(&s.workspace, (size_t)need_floats * sizeof(float));
        s.ws_floats = need_floats;
    }
    return s;
}
}  // anonymous namespace

// mega_decoder_real_adamw_tc — the WIRED TC launcher. Boundary mirrors the scalar
// mega_decoder_real_adamw (dispatch.cpp extern-decls it with the same mirror
// types) EXCEPT it takes NO workspace pointer: the TC workspace is a different
// size from the scalar nCTA*total partials, so this TU allocates+owns it. `loss_out`
// still points into the caller's state[3*total] slot (dispatch reads it back).
//
// ncta_cap: forwarded to the TC launcher (0 = one CTA/SM = full saturation, the
// shipped config). The race always passes 0.
cudaError_t mega_decoder_real_adamw_tc(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        const int* sizes, const int* offsets,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap) {
    (void)sizes; (void)offsets;  // TC kernel reads kDecSizes/kDecOffsets directly.

    const int64_t total = kDecTotalElems;
    const int T = B * dec::kSeq;

    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 1;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int nCTA = n_sms;
    if (ncta_cap > 0 && ncta_cap < nCTA) nCTA = ncta_cap;

    const int64_t need = dec_tc_workspace_floats(T, B, nCTA);
    DecTcLauncherScratch& sc = dec_tc_launcher_scratch(dev, need);

    // Re-point the ctx counters at THIS TU's cached counters (dispatch.cpp built
    // ctx pointing at its scalar DecoderScratch counters; the TC launcher zeroes
    // them per-launch, but they must be the ones this scratch owns so the grid
    // barrier is consistent with the workspace).
    ctx.g_next_task   = sc.g_next;
    ctx.g_arrived     = sc.g_arrived;
    ctx.g_generation  = sc.g_generation;
    ctx.n_tasks       = kDecNumTensors;  // parameter tensors for reduce+opt phases

    FusedOptState st;
    st.exp_avg    = state;
    st.exp_avg_sq = state + total;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    DecoderTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = sc.workspace;
    tok.loss_out  = loss_out;

    return launch_fused_decoder_megakernel_tc<OptId::AdamW>(
        ctx, params, tok, grad, lr, step, st, stream, nCTA);
}

}}}  // namespace sg::fused::sm90
