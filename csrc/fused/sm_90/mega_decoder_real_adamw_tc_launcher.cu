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
// OPTID-GENERIC (owner baseline directive — all 33 cells on L3-TC): the fwd+bwd+
// grad-reduction is OPTIMIZER-INDEPENDENT (the same validated wgmma decoder kernel);
// ONLY the per-element optimizer TAIL (apply_optimizer<Opt>) differs, and that tail
// is the 14/0-apply-parity math already in opt_components.cuh. So this launcher
// takes `opt_id` (the OptId int) and dispatches to the matching kernel instantiation
// over the SAME state buffer. The state layout is [m | v | extra] (3*total) + loss:
//   * lion              uses m only (v/extra unread; bound harmlessly)
//   * adamw             uses m, v
//   * grokfast/grokadamw use m, v, ema  (ema == the `extra` slice, state + 2*total)
//   * neuralgrok        uses m, v, AND a psi-net weight pack the cell supplies in the
//                       `extra` slice (kPsiPackFloats floats); st.psi_W1/b1/W2 bind
//                       to those offsets, exactly as opt_components.cuh documents.
// The STAGED optimizers (prodigy/muon/SG11/SG15) and the model-coupled (looksam) and
// SG2 are NOT routed here — they need precompute stages / a 2nd backward / a sharpness
// ABI field that this single-launch path does not carry; dispatch.cpp gates them out
// (and wiring_check fails them loud with the cited reason). NeuralGrok is wired but
// its host-trained amplifier is a separate concern (the race fn gates it).
//
// ncta_cap: forwarded to the TC launcher (0 = one CTA/SM = full saturation, the
// shipped config). The race always passes 0.
cudaError_t mega_decoder_real_adamw_tc(
        PersistentContext ctx, float* params,
        const int* tokens, const int* targets, int B,
        float* state, float* grad, float* /*workspace_unused*/, float* loss_out,
        const int* sizes, const int* offsets,
        float lr, int step, const FusedScalars& scalars, cudaStream_t stream,
        int ncta_cap, int opt_id) {
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

    // Bind the optimizer state slices. [m | v | extra] over the SAME state buffer;
    // each tail reads only its own buffers (opt_components.cuh guarantees unused
    // pointers are never dereferenced). ema/psi-net live in the `extra` slice.
    float* const m_slice     = state;
    float* const v_slice     = state + total;
    float* const extra_slice = state + 2 * total;
    FusedOptState st;
    st.exp_avg    = m_slice;
    st.exp_avg_sq = v_slice;
    st.ema        = extra_slice;          // grokfast/grokadamw slow-grad EMA
    // NeuralGrok psi-net pack (kPsiPackFloats floats at the head of `extra`): the
    // cell places [psi_W1 | psi_b1 | psi_W2 | psi_b2] there; bind the device pointers
    // so the apply reads real weights (psi_b2 is read on-device from psi_W2[kPsiHidden]).
    st.psi_W1     = extra_slice + kPsiW1Off;
    st.psi_b1     = extra_slice + kPsiB1Off;
    st.psi_W2     = extra_slice + kPsiW2Off;
    apply_scalars(st, scalars);   // FULL scalar set (un-frozen bc1/bc2/...)
    st.lr = lr;

    DecoderTokenCtx tok;
    tok.tokens    = tokens;
    tok.targets   = targets;
    tok.B         = B;
    tok.workspace = sc.workspace;
    tok.loss_out  = loss_out;

    // Dispatch to the matching kernel instantiation. The fwd+bwd is identical across
    // these OptIds; the template differs ONLY in the P3 apply_optimizer<Opt> tail.
    // Unsupported opt_ids return cudaErrorInvalidValue → dispatch.cpp throws LOUD (no
    // silent scalar/adamw fallback — the no-suppression rule). The STAGED/model-coupled
    // optimizers are intentionally NOT cases here (they cannot run as a single launch).
    switch (static_cast<OptId>(opt_id)) {
        case OptId::AdamW:
            return launch_fused_decoder_megakernel_tc<OptId::AdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Lion:
            return launch_fused_decoder_megakernel_tc<OptId::Lion>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::Grokfast:
            return launch_fused_decoder_megakernel_tc<OptId::Grokfast>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::GrokAdamW:
            return launch_fused_decoder_megakernel_tc<OptId::GrokAdamW>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        case OptId::NeuralGrok:
            return launch_fused_decoder_megakernel_tc<OptId::NeuralGrok>(
                ctx, params, tok, grad, lr, step, st, stream, nCTA);
        default:
            return cudaErrorInvalidValue;  // STAGED/coupled opt not single-launch TC
    }
}

}}}  // namespace sg::fused::sm90
