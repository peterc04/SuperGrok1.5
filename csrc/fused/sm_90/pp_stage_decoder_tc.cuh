#ifndef SG_FUSED_SM90_PP_STAGE_DECODER_TC_CUH_
#define SG_FUSED_SM90_PP_STAGE_DECODER_TC_CUH_
// ============================================================================
// csrc/fused/sm_90/pp_stage_decoder_tc.cuh — PIPELINE-PARALLEL stage kernels
// for the L3-TC decoder megakernel (design §4, task #25 phase-2, Lane C).
//
// THE CUT (design §4.1): PP splits the LAYER STACK across stages; each stage is
// its OWN persistent megakernel launch over its layer subset — the megakernel
// remains the unit of execution (P0→P1→B1→P2 fused inside each stage; we shard
// AROUND the fused step, never decompose it into ops). The acts-to-HBM DecActs
// design makes the boundary natural:
//   * FWD handoff  = acts.X_in[Lhi]  (bf16 [T,d]) — the boundary layer's n2
//     output, written by the EXISTING li+1<kLayers branch. Same-device loopback:
//     zero-copy (shared acts). Real PP: a P2P copy of exactly this region.
//   * BWD handoff  = dh_boundary (fp32 [T,d]) — the downstream stage's running
//     adjoint at its Llo input. fp32 ON PURPOSE: in the fused kernel this value
//     never rounds to bf16, so the fp32 carrier keeps the stage composition
//     BIT-IDENTICAL to the single-launch kernel (the loopback gate's assert).
//
// 1F1B / fwd-bwd split (design §4.1-4.2): a pipeline interleaves microbatches'
// fwd and bwd, so a stage exposes
//   * pp_decoder_stage_fwd_tc   — FWD-ONLY launch (single phase, no barriers):
//     P1-fwd tiles → boundary acts (+ nothing else; loss only exists on the
//     last stage's bwd pass).
//   * pp_decoder_stage_bwd_tc   — the FUSED fwd-RECOMPUTE + bwd + grad-assembly
//     launch: P0 → B0 → P1 (fwd<Llo,Lhi> re-runs to repopulate the per-CTA tile
//     scratch — per-tile recompute is the standard PP bwd pattern and is
//     BIT-identical here because the tile math is deterministic — then
//     bwd<Llo,Lhi> with dh_in/dh_out) → B1 → P2 over the OWNED tensors only
//     (dW split-K path identical to production, owned biases, embedding scan on
//     stage 0, owned LN-vec slots, loss on the last stage).
//   The last stage's 1F1B "fwd then immediately bwd" is exactly ONE
//   pp_decoder_stage_bwd_tc launch — the fused fwd+bwd unit is preserved there.
//
// OPTIMIZER: NOT here. PP composes with the established B2-seam cut (design
// §2.1/§2.2): each stage produces the reduced grad for its OWNED tensors; the
// DP/ZeRO collective dance + sharded_optimizer_kernel (bit-parity-proven by
// ef433ac DELIVERABLE 1) applies the update. A PP stage kernel therefore ends
// at its B2-equivalent ("owned grad assembled").
//
// TENSOR OWNERSHIP (dec_layout named_parameters() order):
//   stage owning layers [Llo,Lhi) owns tensors [2+12·Llo, 2+12·Lhi)
//   (12 tensors per layer: in_w,in_b,out_w,out_b,n1w,n1b,n2w,n2b,ff0w,ff0b,
//    ff2w,ff2b); stage 0 additionally owns {0,1} (tok,pos); the LAST stage
//   additionally owns {26,27,28,29} (final norm γ/β + head w/b).
//   dW specs (4 per layer + head): spec s ∈ [4·Llo, 4·Lhi) (+ spec 8 on last).
//   LN-vec slots: v ∈ [4·Llo, 4·Lhi) (+ {8,9} on last).
//
// REQUIRES the layer-range patch (.phase2/patches/0001-dectc-layer-range-pp
// .patch) on model_stage_decoder_tc.cuh — gated LOUDLY below, no silent
// degradation. This header is NEW (not in the production build glob); it is
// JIT-compiled by tests/hw/pp_stage_binding.cu.
// ============================================================================

// The wgmma stage substrate + (transitively) the patched tile functions.
#ifndef SG_TUNED_GEMM_IMPL
#define SG_TUNED_GEMM_IMPL 1   // PP stages exist for the TC (wgmma) path
#endif
#include "csrc/fused/sm_90/fused_decoder_megakernel.cuh"

#ifndef SG_DECTC_LAYER_RANGE
#error "pp_stage_decoder_tc.cuh requires the dectc layer-range patch. Apply \
.phase2/patches/0001-dectc-layer-range-pp.patch to \
csrc/fused/sm_90/model_stage_decoder_tc.cuh first (see .phase2/RUNBOOK.md). \
This is a loud gate, not an optional feature — building PP stages against the \
unpatched single-stage tile functions would silently run the wrong layers."
#endif

namespace sg { namespace fused { namespace sm90 { namespace pp {

// ─────────────────────────────────────────────────────────────────────────
//  Stage spec: contiguous layer split. NumStages must divide kLayers (the
//  2-layer race decoder ⇒ NumStages=2 ⇒ 1 layer/stage; ladder depths divide
//  evenly by construction — the host plan asserts before instantiating).
// ─────────────────────────────────────────────────────────────────────────
template <int Stage, int NumStages>
struct PPStageSpec {
    static_assert(NumStages >= 1 && NumStages <= dec::kLayers,
                  "PP: NumStages must be in [1, kLayers]");
    static_assert(dec::kLayers % NumStages == 0,
                  "PP: contiguous even split requires NumStages | kLayers");
    static_assert(Stage >= 0 && Stage < NumStages, "PP: Stage out of range");

    static constexpr int kLayersPerStage = dec::kLayers / NumStages;
    static constexpr int kLlo = Stage * kLayersPerStage;
    static constexpr int kLhi = kLlo + kLayersPerStage;
    static constexpr bool kIsFirst = (Stage == 0);
    static constexpr bool kIsLast  = (Stage == NumStages - 1);

    // dW spec ownership (dectc_build_dw_specs order: 4 specs/layer + head=8).
    static constexpr int kSpecLo = 4 * kLlo;
    static constexpr int kSpecHi = 4 * kLhi;
    __device__ __forceinline__ static bool owns_spec(int s) {
        return (s >= kSpecLo && s < kSpecHi) || (kIsLast && s == 8);
    }
    // LN-vec slot ownership (dectc::kLnVecTensorIdx order: 4 slots/layer + 8,9).
    __device__ __forceinline__ static bool owns_lnvec(int v) {
        return (v >= 4 * kLlo && v < 4 * kLhi) || (kIsLast && v >= 8);
    }
    // Flat tensor ownership (dec_layout order) — host-side mirror lives in
    // grokking_optimizers/parallel/pipeline.py::stage_tensor_ownership (the two
    // are cross-checked by tests/test_pipeline_schedule.py via this header's
    // constants printed by the binding).
    __host__ __device__ __forceinline__ static bool owns_tensor(int t) {
        if (t >= 2 + 12 * kLlo && t < 2 + 12 * kLhi) return true;   // layer block
        if (kIsFirst && (t == 0 || t == 1)) return true;            // tok, pos
        if (kIsLast && t >= 26 && t <= 29) return true;             // norm + head
        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────
//  FWD-ONLY stage kernel (single phase — no grid barriers, no grad writes).
//  Writes this stage's acts X-regions; the boundary X_in[kLhi] is the handoff.
//  Last stage included for completeness (1F1B never schedules it: its fwd+bwd
//  run fused in pp_decoder_stage_bwd_tc).
// ─────────────────────────────────────────────────────────────────────────
template <int Stage, int NumStages>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
pp_decoder_stage_fwd_tc(PersistentContext ctx,
                        const float* __restrict__ params,
                        DecoderTokenCtx tok) {
    using S = PPStageSpec<Stage, NumStages>;
    __shared__ DecTcSmem sm;
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int B = tok.B;
    const int T = B * dec::kSeq;

    // Workspace partition — IDENTICAL carve to the fused TC kernel (shared
    // workspace across the stage launches of one step ⇒ identical offsets).
    float* ws = tok.workspace;
    const int64_t acts_f = dec_tc_acts_floats(T, B);
    __nv_bfloat16* acts_base = reinterpret_cast<__nv_bfloat16*>(ws);
    float* scratch_base = ws + acts_f;
    const int64_t scratch_per = dectc::dec_tile_scratch_total_f32();

    dectc::DecActs acts = dectc::dec_acts_bind(acts_base, T, B);
    dectc::DecTileScratch sc = dectc::dec_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    DecWeights w = dec_bind(const_cast<float*>(params));

    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
        (void)dectc::dectc_forward_tile<S::kLlo, S::kLhi>(
            w, g0, nrows, acts, sc, tok.tokens, tok.targets, sm.sA, sm.sB, sm.red);
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  FWD-RECOMPUTE + BWD + OWNED-GRAD-ASSEMBLY stage kernel. Mirrors the fused
//  TC kernel's P0→B0→P1→B1→P2 with the layer-range tiles and owned-tensor
//  filters; ends at the B2-equivalent (owned grad assembled in `grad`).
//
//  dh_in : fp32 [T,d] boundary adjoint from the DOWNSTREAM stage
//          (required unless last stage — last stage starts from CE).
//  dh_out: fp32 [T,d] boundary adjoint for the UPSTREAM stage
//          (required unless first stage — first stage writes dh0/embeds).
//  Preconditions (loopback shares one workspace; real PP copies regions in):
//    * this stage's input acts X_in[kLlo] are populated (upstream fwd handoff);
//    * for the last stage, nothing else; for others, dh_in is populated.
// ─────────────────────────────────────────────────────────────────────────
template <int Stage, int NumStages>
__global__ void __launch_bounds__(SG_TC_MEGA_BLOCK)
pp_decoder_stage_bwd_tc(PersistentContext ctx,
                        const float* __restrict__ params,
                        DecoderTokenCtx tok,
                        float* __restrict__ grad,
                        const float* __restrict__ dh_in,
                        float* __restrict__ dh_out) {
    using S = PPStageSpec<Stage, NumStages>;
    __shared__ DecTcSmem sm;
    GridBarrier bar = ctx.barrier();
    const int cta = blockIdx.x;
    const int nCTA = (int)ctx.n_ctas;
    const int B = tok.B;
    const int T = B * dec::kSeq;

    // Workspace partition — byte-identical carve to fused_decoder_megakernel_tc
    // (same helpers, same order) so the loopback can share one workspace with
    // identical region addresses (dw_part slot bases included).
    float* ws = tok.workspace;
    const int64_t acts_f = dec_tc_acts_floats(T, B);
    __nv_bfloat16* acts_base = reinterpret_cast<__nv_bfloat16*>(ws);
    float* scratch_base = ws + acts_f;
    const int64_t scratch_per = dectc::dec_tile_scratch_total_f32();
    float* lnvec_base = scratch_base + (int64_t)nCTA * scratch_per;
    float* loss_part  = lnvec_base + (int64_t)nCTA * dectc::kLnVecElems;
    float* loss_out   = loss_part + nCTA;
    float* dw_part    = loss_out + 1;
    const int kDwG    = dectc::kDecDwSplitK;
    float* embed_ws_base = dw_part + dec_tc_dw_part_floats()
                         + dec_tc_staged_opt_floats(nCTA);
    if (((uintptr_t)embed_ws_base & 0x7) != 0) embed_ws_base += 1;  // 8-byte align
    int* embed_row_start = reinterpret_cast<int*>(embed_ws_base);   // [V+1]
    int* embed_perm      = embed_row_start + (dec::kVocab + 1);     // [T]

    dectc::DecActs acts = dectc::dec_acts_bind(acts_base, T, B);
    dectc::DecTileScratch sc = dectc::dec_tile_scratch_bind(scratch_base + (int64_t)cta * scratch_per);
    float* my_lnvec = lnvec_base + (int64_t)cta * dectc::kLnVecElems;
    DecWeights w = dec_bind(const_cast<float*>(params));

    // ── P0: zero this CTA's LN-vec partials + loss slot (mirrors fused P0). ──
    for (int i = threadIdx.x; i < dectc::kLnVecElems; i += blockDim.x) my_lnvec[i] = 0.0f;
    if (threadIdx.x == 0) loss_part[cta] = 0.0f;
    bar.sync();   // B0

    // Embedding token lists: ONLY the stage that owns the embeddings consumes
    // them; built once on cta 0 in the P1 window (fused-kernel pattern).
    if (S::kIsFirst && cta == 0)
        dectc::dectc_embed_build_lists(tok.tokens, T, embed_row_start, embed_perm);

    // ── P1: fwd<Llo,Lhi> RECOMPUTE (repopulates this CTA's tile scratch +
    //    this stage's acts — bit-identical to the fwd launch's values) then
    //    bwd<Llo,Lhi> with the boundary adjoints. ──
    const int nrows_tile = dectc::kTileM;
    const int n_tiles = (T + nrows_tile - 1) / nrows_tile;
    float nll_acc = 0.0f;
    for (int ti = cta; ti < n_tiles; ti += nCTA) {
        const int g0 = ti * nrows_tile;
        const int nrows = (T - g0) < nrows_tile ? (T - g0) : nrows_tile;
        float nll = dectc::dectc_forward_tile<S::kLlo, S::kLhi>(
            w, g0, nrows, acts, sc, tok.tokens, tok.targets, sm.sA, sm.sB, sm.red);
        dectc::dectc_backward_tile<S::kLlo, S::kLhi>(
            w, g0, nrows, B, acts, sc, tok.targets, my_lnvec, sc.work2,
            sm.sA, sm.sB, sm.red, dh_in, dh_out);
        if (threadIdx.x == 0) nll_acc += nll;
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_part[cta] = nll_acc;
    bar.sync();   // B1: this stage's acts (X + dY) + LN-vec partials complete

    // ── P2: assemble the OWNED grads. Same global work-item space + slot bases
    //    as the fused kernel (so the loopback composition is offset-identical);
    //    non-owned items are SKIPPED (the owning stage writes them). ──
    if (threadIdx.x == 0) dectc::dectc_build_dw_specs(acts, B, T, sm.spec);
    __syncthreads();
    dectc::DecDwSpec* spec = sm.spec;
    const int n_dw = dectc::dectc_dw_total_tiles<SG_TUNED_TILE_N>(spec);

    if (kDwG > 1) {
        // Split-K partials for OWNED tiles only (identical (gt,kc) slot bases).
        for (int item = cta; item < n_dw * kDwG; item += nCTA) {
            const int gt = item / kDwG, kc = item % kDwG;
            int s, mg, nt;
            dectc::dectc_dw_decode<SG_TUNED_TILE_N>(spec, gt, s, mg, nt);
            if (!S::owns_spec(s)) continue;
            dectc::dectc_dw_run_tile_splitk<SG_TUNED_TILE_N>(spec, gt, kc, kDwG, dw_part, sm.sA, sm.sB);
        }
        bar.sync();   // all owned (gt,kc) partials complete
        // Deterministic ascending-kc reduce, filtered to owned tiles.
        for (int gt = cta; gt < n_dw; gt += nCTA) {
            int s, m_group, n_tile;
            dectc::dectc_dw_decode<SG_TUNED_TILE_N>(spec, gt, s, m_group, n_tile);
            if (!S::owns_spec(s)) continue;
            const dectc::DecDwSpec& sp = spec[s];
            const int mbase = m_group * dectc::kDecDwIL * 64;
            const int n0 = n_tile * SG_TUNED_TILE_N;
            const int n_real = (sp.Kin - n0) < SG_TUNED_TILE_N ? (sp.Kin - n0) : SG_TUNED_TILE_N;
            const int grp_rows = dectc::kDecDwIL * 64;
            const int Nrow = (sp.Nout - mbase) < grp_rows ? (sp.Nout - mbase) : grp_rows;
            const int64_t base = (int64_t)gt * kDwG * dectc::kDecDwTileFloats;
            for (int idx = threadIdx.x; idx < Nrow * n_real; idx += blockDim.x) {
                const int row = idx / n_real, col = idx % n_real;
                float accv = 0.0f;
                for (int kc = 0; kc < kDwG; ++kc)
                    accv += dw_part[base + (int64_t)kc * dectc::kDecDwTileFloats
                                    + (int64_t)row * SG_TUNED_TILE_N + col];
                grad[sp.grad_off + (int64_t)(mbase + row) * sp.Kin + n0 + col] = accv;
            }
        }
    } else {
        for (int gt = cta; gt < n_dw; gt += nCTA) {
            int s, mg, nt;
            dectc::dectc_dw_decode<SG_TUNED_TILE_N>(spec, gt, s, mg, nt);
            if (!S::owns_spec(s)) continue;
            dectc::dectc_dw_run_tile<SG_TUNED_TILE_N>(spec, gt, grad, sm.sA, sm.sB);
        }
    }

    // Owned biases (single-owner grid-stride column-sum over the GLOBAL bias
    // index space — same owner map as fused; non-owned outputs skipped).
    {
        int pre[10];
        pre[0] = 0;
        #pragma unroll
        for (int s = 0; s < 9; ++s) pre[s + 1] = pre[s] + spec[s].Nout;
        const int total_b = pre[9];
        const int stride = nCTA * blockDim.x;
        for (int go = cta * blockDim.x + threadIdx.x; go < total_b; go += stride) {
            int s = 0;
            #pragma unroll
            for (int t = 0; t < 9; ++t) if (go >= pre[t + 1]) s = t + 1;
            if (!S::owns_spec(s)) continue;
            const dectc::DecDwSpec& sp = spec[s];
            const int o = go - pre[s];
            float accv = 0.0f;
            for (int k = 0; k < sp.K; ++k)
                accv += __bfloat162float(sp.dY_bias[(int64_t)k * sp.Nout + o]);
            grad[sp.bias_off + o] = accv;
        }
    }

    // Embedding grads (tok/pos) — first stage only (it owns dh0 + the lists).
    if (S::kIsFirst)
        dectc::dectc_embed_owner_scan(acts, embed_row_start, embed_perm, T, grad, cta, nCTA);

    // Owned LN-vec slots (ascending-CTA reduce, fused pattern, filtered).
    for (int v = cta; v < dectc::kNumLnVec; v += nCTA) {
        if (!S::owns_lnvec(v)) continue;
        const int goff = dectc::kLnVecTensorIdx[v];
        const int64_t gbase = kDecOffsets[goff];
        for (int j = threadIdx.x; j < dec::kD; j += blockDim.x) {
            float accv = 0.0f;
            for (int c = 0; c < nCTA; ++c)
                accv += lnvec_base[(int64_t)c * dectc::kLnVecElems + (int64_t)v * dec::kD + j];
            grad[gbase + j] = accv;
        }
    }

    // Loss reduce (fp64, CTA 0) — last stage only (it ran CE).
    if (S::kIsLast && cta == 0 && threadIdx.x == 0) {
        double s = 0.0;
        for (int c = 0; c < nCTA; ++c) s += (double)loss_part[c];
        *tok.loss_out = (float)(s / (double)B);
    }
    // B2-equivalent: owned grad assembled; kernel exits (the optimizer is the
    // separate sharded-opt kernel per the design §2.2 cut — already gated).
}

// ─────────────────────────────────────────────────────────────────────────
//  Launchers — mirror launch_fused_decoder_megakernel_tc's discipline exactly:
//  one CTA/SM (ncta_cap-able), occupancy>=1 refuse-or-hang guard, B%16 gate,
//  barrier/queue slots zeroed before launch.
// ─────────────────────────────────────────────────────────────────────────
template <int Stage, int NumStages>
cudaError_t launch_pp_decoder_stage_fwd_tc(
        PersistentContext ctx, const float* params, DecoderTokenCtx tok,
        cudaStream_t stream, int ncta_cap = 0) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&pp_decoder_stage_fwd_tc<Stage, NumStages>,
        SG_TC_MEGA_BLOCK, 0);
    if (err != cudaSuccess) return err;
    if (occ < 1) return cudaErrorLaunchOutOfResources;
    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;
    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    pp_decoder_stage_fwd_tc<Stage, NumStages><<<grid, block, 0, stream>>>(ctx, params, tok);
    return cudaGetLastError();
}

template <int Stage, int NumStages>
cudaError_t launch_pp_decoder_stage_bwd_tc(
        PersistentContext ctx, const float* params, DecoderTokenCtx tok,
        float* grad, const float* dh_in, float* dh_out,
        cudaStream_t stream, int ncta_cap = 0) {
    using S = PPStageSpec<Stage, NumStages>;
    // Loud preconditions: a non-last stage NEEDS dh_in; a non-first NEEDS dh_out.
    if (!S::kIsLast && dh_in == nullptr) return cudaErrorInvalidValue;
    if (!S::kIsFirst && dh_out == nullptr) return cudaErrorInvalidValue;
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return err;
    int n_sms = 0;
    err = cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) return err;
    int occ = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void*)&pp_decoder_stage_bwd_tc<Stage, NumStages>,
        SG_TC_MEGA_BLOCK, 0);
    if (err != cudaSuccess) return err;
    if (occ < 1) return cudaErrorLaunchOutOfResources;
    unsigned launch_ctas = (unsigned)n_sms;
    if (ncta_cap > 0 && (unsigned)ncta_cap < launch_ctas) launch_ctas = (unsigned)ncta_cap;
    ctx.n_ctas = launch_ctas;
    if ((tok.B % 16) != 0) return cudaErrorInvalidValue;
    if (ctx.g_arrived) { err = cudaMemsetAsync(ctx.g_arrived, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_generation) { err = cudaMemsetAsync(ctx.g_generation, 0, sizeof(unsigned), stream); if (err) return err; }
    if (ctx.g_next_task) { err = cudaMemsetAsync(ctx.g_next_task, 0, sizeof(int), stream); if (err) return err; }
    dim3 grid(launch_ctas), block(SG_TC_MEGA_BLOCK);
    pp_decoder_stage_bwd_tc<Stage, NumStages><<<grid, block, 0, stream>>>(
        ctx, params, tok, grad, dh_in, dh_out);
    return cudaGetLastError();
}

}}}}  // namespace sg::fused::sm90::pp

#endif  // SG_FUSED_SM90_PP_STAGE_DECODER_TC_CUH_
