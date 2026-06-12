// ============================================================================
// tests/hw/tp_loopback_binding.cu — JIT binding for the TENSOR-PARALLEL
// LOOPBACK gate (design §5.1/§5.2 via §7; task #25 phase-2, Lane C).
//
// WHAT IT PROVES (driven by tests/hw/test_tp_loopback.py, single GPU):
// the full Megatron column→row TP pair — col-parallel GEMM, local gelu,
// row-parallel partial GEMM, PUBLISH → rendezvous → FIXED-ORDER ascending-pe
// fp32 all-reduce (fwd point ②), and the conjugate backward (row-parallel dX
// local, col-parallel dX partial → all-reduce, bwd point ②') plus the
// exact-slice dW/db shards — running on TP=P VIRTUAL ranks in ONE grid over
// the LoopbackTransport symmetric heap, using the PRODUCTION wgmma tile GEMMs
// (dectc_gemm_fwd_f32 / dectc_gemm_dx_f32 — the same engine, staging, fp32
// accumulate as the L3-TC megakernel; ffn shapes d=128, dff=512, kTileM=128).
//
// THE TRANSPORT SWAP POINT: the kernel below is written against the
// tp_transport.cuh concept. Re-pointing `LoopbackTransport` → `NvshmemTransport`
// (8×H100 window, -DSG_HAS_NVSHMEM) changes NO math in this file — that is the
// design §5.3/§5.4 seam, and the residual GPU-window task.
//
// HONESTY: the loopback is NOT fake success — the host asserts
//   (a) all P virtual ranks produced BIT-IDENTICAL reduced outputs,
//   (b) 3 reruns are BIT-IDENTICAL (A/A/A),
//   (c) the TP+transport result == a SERIAL chunked-order reference BIT-EXACT
//       (proves the transport contributes exactly zero numerical effect),
//   (d) every dW/db shard == the corresponding slice of the UNSHARDED dW/db
//       BIT-EXACT (the slice-exactness property tp_layer.cuh documents),
//   (e) vs the unsharded full-K reference within the campaign parity tol
//       (the chunk-reassociation delta, reported, not asserted bitwise).
//
// JIT-built via torch.utils.cpp_extension.load into /workspace/.torch_ext —
// owns its PYBIND11_MODULE; setup.py's _owns_extension_module excludes it from
// the production _ops build (same discipline as sharded_optimizer_binding.cu).
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "csrc/fused/sm_90/tp_layer.cuh"        // TP math + transport + dectc GEMMs

namespace {

using namespace sg::fused;          // GridBarrier
using namespace sg::fused::sm90;    // dec::, dectc::
namespace tp = sg::fused::sm90::tp;

constexpr int kM   = dectc::kTileM;     // 128 token rows (one production tile)
constexpr int kD   = dec::kD;           // 128
constexpr int kDff = dec::kDff;         // 512
constexpr int kBlk = 256;               // the megakernel CTA shape (2 warpgroups)

// Shared-memory mirror of DecTcSmem's GEMM ring (sA/sB/red) — same sizes the
// production kernel hands the dectc GEMMs.
struct TpSmem {
    static constexpr int kAtomsPerSlot =
        (dectc::kAtomsM < dectc::kDecMaxIL) ? dectc::kAtomsM : dectc::kDecMaxIL;
    __nv_bfloat16 sA[dectc::kDecTcStages * kAtomsPerSlot * 64 * 16];
    __nv_bfloat16 sB[dectc::kDecTcStages * SG_TUNED_TILE_N * 16];
};

// ─────────────────────────────────────────────────────────────────────────
//  Device buffers bundle (raw pointers; all fp32 unless noted).
//  Shard buffers are DENSE concatenations over ranks:
//    W0s [P][dff/P, d]   b0s [P][dff/P]   W1s [P][d, dff/P]
//  Per-PE scratch/outputs are [P][...] concatenations.
// ─────────────────────────────────────────────────────────────────────────
struct TpArgs {
    const __nv_bfloat16* X;      // [M, d]   tile input (bf16, like X_x1)
    const __nv_bfloat16* dY1;    // [M, d]   output adjoint (bf16, like dY_ff2)
    const float* W0s; const float* b0s; const float* W1s;   // dense shards
    const float* b1;             // [d] replicated
    float* heap;                 // [P * 2 * M*d] loopback symmetric heap
    __nv_bfloat16* pre;          // [P][M, dff/P] bf16 (ff0pre mirror)
    __nv_bfloat16* G;            // [P][M, dff/P] bf16 (X_gact mirror)
    float* dG;                   // [P][M, dff/P] fp32
    __nv_bfloat16* dY0;          // [P][M, dff/P] bf16 (dY_ff0 mirror)
    float* Y1;                   // [P][M, d] reduced fwd output, per-PE copy
    float* dX;                   // [P][M, d] reduced bwd dX, per-PE copy
    float* dW0;                  // [P][dff/P, d]  (== concat: [dff, d])
    float* db0;                  // [P][dff/P]     (== concat: [dff])
    float* dW1;                  // [P][d, dff/P]
    float* db1;                  // [P][d] (replicated — every PE computes it)
    int P;
};

// ─────────────────────────────────────────────────────────────────────────
//  THE TP=P LOOPBACK KERNEL. grid = P CTAs (1 CTA per virtual PE — one tile in
//  flight per PE, the persistent-kernel shape); block = 256. The GridBarrier
//  over the whole grid is the cross-PE rendezvous (LoopbackTransport contract).
// ─────────────────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(kBlk)
tp_loopback_block_kernel(TpArgs a, unsigned* g_arrived, unsigned* g_generation) {
    __shared__ TpSmem sm;
    const int P  = a.P;
    const int F  = kDff / P;                 // local feature width
    GridBarrier bar(g_arrived, g_generation, (unsigned)gridDim.x);
    tp::LoopbackTransport tr{a.heap, tp::tp_heap_stride_floats(1), P,
                             tp::LoopbackTransport::pe_of_cta(blockIdx.x, gridDim.x, P)};
    const int pe = tr.my_pe();
    const int64_t slot_part = 0;                          // publish slot
    const int64_t slot_n    = tp::tp_tile_slot_floats();  // M*d floats

    // Per-PE views.
    __nv_bfloat16* pre  = a.pre + (int64_t)pe * kM * F;
    __nv_bfloat16* G    = a.G   + (int64_t)pe * kM * F;
    float*         dG   = a.dG  + (int64_t)pe * kM * F;
    __nv_bfloat16* dY0  = a.dY0 + (int64_t)pe * kM * F;
    float*         Y1   = a.Y1  + (int64_t)pe * kM * kD;
    float*         dX   = a.dX  + (int64_t)pe * kM * kD;
    const float*   W0o  = a.W0s + (int64_t)pe * F * kD;   // [F, d]
    const float*   b0o  = a.b0s + (int64_t)pe * F;        // [F]
    const float*   W1o  = a.W1s + (int64_t)pe * kD * F;   // [d, F]
    float*         dW0o = a.dW0 + (int64_t)pe * F * kD;
    float*         db0o = a.db0 + (int64_t)pe * F;
    float*         dW1o = a.dW1 + (int64_t)pe * kD * F;
    float*         db1o = a.db1 + (int64_t)pe * kD;

    // ── FWD: col-parallel ff0 (comm-free): Y0_own = X @ W0_own^T → fp32 via the
    //    production wgmma tile; bias-fold + gelu mirror the production rounding
    //    (pre kept bf16 like ff0pre; gact bf16 like X_gact). Reuse the publish
    //    slot as the [M,F] fp32 GEMM scratch (slot is M*d ≥ M*F... F ≤ d only
    //    when P ≥ dff/d = 4) — NOT guaranteed; use Y1 as scratch instead (M*d
    //    floats is too small for F>d). dG is [M,F] — use it as the fwd scratch
    //    (free until the backward). ──
    tp::tp_colparallel_fwd_tile<SG_TUNED_TILE_N>(a.X, W0o, dG, kD, F, sm.sA, sm.sB);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kM * F; idx += blockDim.x) {
        const int j = idx % F;
        const float p = dG[idx] + b0o[j];                  // fp32 bias fold
        pre[idx] = __float2bfloat16(p);                    // ff0pre mirror
        G[idx]   = __float2bfloat16(dec_gelu(p));          // X_gact mirror
    }
    __syncthreads();

    // ── FWD: row-parallel ff2 PARTIAL → publish to the symmetric slot (the
    //    all-reduce point ② of design §5.1). ──
    tp::tp_rowparallel_fwd_partial_tile<SG_TUNED_TILE_N>(
        tr, slot_part, G, W1o, F, kD, sm.sA, sm.sB);
    tr.rendezvous(bar);                                    // partials published
    // Fixed-order ascending-pe fp32 reduce → this PE's Y1 copy; + replicated b1.
    tp::tp_allreduce_sum_fixed_order(tr, slot_part, Y1, (int64_t)kM * kD,
                                     threadIdx.x, blockDim.x);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x)
        Y1[idx] += a.b1[idx % kD];
    tr.rendezvous(bar);                                    // slot reusable

    // ── BWD: row-parallel dX (comm-free): dG_own = dY1 @ W1_own. ──
    tp::tp_rowparallel_dx_tile<SG_TUNED_TILE_N>(a.dY1, W1o, dG, F, kD, sm.sA, sm.sB);
    __syncthreads();
    // gelu' on the ROUNDED pre (production dY_ff0 pattern), bf16 like dY_ff0.
    for (int idx = threadIdx.x; idx < kM * F; idx += blockDim.x)
        dY0[idx] = __float2bfloat16(dG[idx] * dec_gelu_grad(__bfloat162float(pre[idx])));
    __syncthreads();

    // ── BWD: col-parallel dX PARTIAL → publish to the SECOND slot (the
    //    conjugate point ②'; distinct slot exercises multi-slot addressing). ──
    tp::tp_colparallel_dx_partial_tile<SG_TUNED_TILE_N>(
        tr, slot_n, dY0, W0o, kD, F, sm.sA, sm.sB);
    tr.rendezvous(bar);
    tp::tp_allreduce_sum_fixed_order(tr, slot_n, dX, (int64_t)kM * kD,
                                     threadIdx.x, blockDim.x);
    tr.rendezvous(bar);

    // ── dW/db shards (exact-slice property; ascending-m fp32, single owner per
    //    element — the P2 determinism contract). dW0_own[i,k] = Σ_m dY0[m,i]·X[m,k];
    //    dW1_own[j,c] = Σ_m dY1[m,j]·G_own[m,c]; db0_own[i] = Σ_m dY0[m,i];
    //    db1[j] = Σ_m dY1[m,j] (replicated: every PE computes the identical value). ──
    for (int idx = threadIdx.x; idx < F * kD; idx += blockDim.x) {
        const int i = idx / kD, k = idx % kD;
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m)
            acc += __bfloat162float(dY0[(int64_t)m * F + i]) * __bfloat162float(a.X[(int64_t)m * kD + k]);
        dW0o[idx] = acc;
    }
    for (int idx = threadIdx.x; idx < kD * F; idx += blockDim.x) {
        const int j = idx / F, c = idx % F;
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m)
            acc += __bfloat162float(a.dY1[(int64_t)m * kD + j]) * __bfloat162float(G[(int64_t)m * F + c]);
        dW1o[idx] = acc;
    }
    for (int i = threadIdx.x; i < F; i += blockDim.x) {
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m) acc += __bfloat162float(dY0[(int64_t)m * F + i]);
        db0o[i] = acc;
    }
    for (int j = threadIdx.x; j < kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m) acc += __bfloat162float(a.dY1[(int64_t)m * kD + j]);
        db1o[j] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  REFERENCE kernel (1 CTA): (i) the UNSHARDED pair with the same wgmma tiles
//  (full-K contraction — the tol-comparison baseline), and (ii) the SERIAL
//  CHUNKED-ORDER reference — per-shard partial GEMMs (identical inputs/code as
//  the TP kernel) summed ascending-pe in fp32 with the same element loop as
//  tp_allreduce_sum_fixed_order ⇒ must equal the TP+transport result BIT-EXACT.
// ─────────────────────────────────────────────────────────────────────────
struct TpRefArgs {
    const __nv_bfloat16* X; const __nv_bfloat16* dY1;
    const float* W0; const float* b0; const float* W1; const float* b1;  // FULL
    const float* W0s; const float* b0s; const float* W1s;                // shards
    __nv_bfloat16* pre_full;  // [M, dff]
    __nv_bfloat16* G_full;    // [M, dff]
    __nv_bfloat16* dY0_full;  // [M, dff]
    float* scratch_f;         // [M, dff] fp32 general
    __nv_bfloat16* pre_o;     // [M, dff/P] per-shard recompute scratch
    __nv_bfloat16* G_o;       // [M, dff/P]
    __nv_bfloat16* dY0_o;     // [M, dff/P]
    float* part;              // [M, d] one shard's partial
    float* Y1_ref; float* dX_ref;          // [M, d] unsharded
    float* dW0_ref; float* db0_ref;        // [dff, d], [dff]
    float* dW1_ref; float* db1_ref;        // [d, dff], [d]
    float* Y1_chunked; float* dX_chunked;  // [M, d] serial chunked-order
    int P;
};

__global__ void __launch_bounds__(kBlk)
tp_reference_kernel(TpRefArgs a) {
    __shared__ TpSmem sm;
    const int P = a.P;
    const int F = kDff / P;

    // ── (i) UNSHARDED: ff0 → gelu → ff2 (+b1); bwd dG → dY0 → dX; dW/db. ──
    dectc::dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(a.X, a.W0, a.scratch_f, kD, kDff, sm.sA, sm.sB);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kM * kDff; idx += blockDim.x) {
        const int j = idx % kDff;
        const float p = a.scratch_f[idx] + a.b0[j];
        a.pre_full[idx] = __float2bfloat16(p);
        a.G_full[idx]   = __float2bfloat16(dec_gelu(p));
    }
    __syncthreads();
    dectc::dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(a.G_full, a.W1, a.Y1_ref, kDff, kD, sm.sA, sm.sB);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x)
        a.Y1_ref[idx] += a.b1[idx % kD];
    __syncthreads();
    dectc::dectc_gemm_dx_f32<SG_TUNED_TILE_N>(a.dY1, a.W1, a.scratch_f, kDff, kD, sm.sA, sm.sB);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kM * kDff; idx += blockDim.x)
        a.dY0_full[idx] = __float2bfloat16(
            a.scratch_f[idx] * dec_gelu_grad(__bfloat162float(a.pre_full[idx])));
    __syncthreads();
    dectc::dectc_gemm_dx_f32<SG_TUNED_TILE_N>(a.dY0_full, a.W0, a.dX_ref, kD, kDff, sm.sA, sm.sB);
    __syncthreads();
    for (int idx = threadIdx.x; idx < kDff * kD; idx += blockDim.x) {
        const int i = idx / kD, k = idx % kD;
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m)
            acc += __bfloat162float(a.dY0_full[(int64_t)m * kDff + i]) * __bfloat162float(a.X[(int64_t)m * kD + k]);
        a.dW0_ref[idx] = acc;
    }
    for (int idx = threadIdx.x; idx < kD * kDff; idx += blockDim.x) {
        const int j = idx / kDff, c = idx % kDff;
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m)
            acc += __bfloat162float(a.dY1[(int64_t)m * kD + j]) * __bfloat162float(a.G_full[(int64_t)m * kDff + c]);
        a.dW1_ref[idx] = acc;
    }
    for (int i = threadIdx.x; i < kDff; i += blockDim.x) {
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m) acc += __bfloat162float(a.dY0_full[(int64_t)m * kDff + i]);
        a.db0_ref[i] = acc;
    }
    for (int j = threadIdx.x; j < kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int m = 0; m < kM; ++m) acc += __bfloat162float(a.dY1[(int64_t)m * kD + j]);
        a.db1_ref[j] = acc;
    }
    __syncthreads();

    // ── (ii) SERIAL CHUNKED-ORDER: ascending-pe accumulation of per-shard
    //    partials computed with the IDENTICAL shard GEMM calls the TP kernel
    //    makes. acc starts at 0 and adds partials in pe order — the same
    //    arithmetic as tp_allreduce_sum_fixed_order ⇒ bit-exact vs TP. ──
    for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x) {
        a.Y1_chunked[idx] = 0.0f; a.dX_chunked[idx] = 0.0f;
    }
    __syncthreads();
    for (int pe = 0; pe < P; ++pe) {                      // ASCENDING pe
        const float* W0o = a.W0s + (int64_t)pe * F * kD;
        const float* b0o = a.b0s + (int64_t)pe * F;
        const float* W1o = a.W1s + (int64_t)pe * kD * F;
        // recompute this shard's pre/G/dY0 (same code path as the TP kernel).
        tp::tp_colparallel_fwd_tile<SG_TUNED_TILE_N>(a.X, W0o, a.scratch_f, kD, F, sm.sA, sm.sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < kM * F; idx += blockDim.x) {
            const int j = idx % F;
            const float p = a.scratch_f[idx] + b0o[j];
            a.pre_o[idx] = __float2bfloat16(p);
            a.G_o[idx]   = __float2bfloat16(dec_gelu(p));
        }
        __syncthreads();
        dectc::dectc_gemm_fwd_f32<SG_TUNED_TILE_N>(a.G_o, W1o, a.part, F, kD, sm.sA, sm.sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x)
            a.Y1_chunked[idx] += a.part[idx];             // ascending-pe fp32 sum
        __syncthreads();
        dectc::dectc_gemm_dx_f32<SG_TUNED_TILE_N>(a.dY1, W1o, a.scratch_f, F, kD, sm.sA, sm.sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < kM * F; idx += blockDim.x)
            a.dY0_o[idx] = __float2bfloat16(
                a.scratch_f[idx] * dec_gelu_grad(__bfloat162float(a.pre_o[idx])));
        __syncthreads();
        dectc::dectc_gemm_dx_f32<SG_TUNED_TILE_N>(a.dY0_o, W0o, a.part, kD, F, sm.sA, sm.sB);
        __syncthreads();
        for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x)
            a.dX_chunked[idx] += a.part[idx];
        __syncthreads();
    }
    // + b1 on the chunked fwd (after the sum, matching the TP kernel's order).
    for (int idx = threadIdx.x; idx < kM * kD; idx += blockDim.x)
        a.Y1_chunked[idx] += a.b1[idx % kD];
}

// ─────────────────────────────────────────────────────────────────────────
//  Host plumbing.
// ─────────────────────────────────────────────────────────────────────────
#define CHK(t, n, len) \
    TORCH_CHECK((t).is_cuda() && (t).scalar_type() == at::kFloat && (t).is_contiguous() \
                && (t).numel() == (len), n " must be CUDA fp32 contiguous of numel ", (int64_t)(len))

void tp_loopback_run(int64_t P_,
                     at::Tensor X, at::Tensor dY1,
                     at::Tensor W0s, at::Tensor b0s, at::Tensor W1s, at::Tensor b1,
                     at::Tensor Y1, at::Tensor dX,
                     at::Tensor dW0, at::Tensor db0, at::Tensor dW1, at::Tensor db1) {
    const int P = (int)P_;
    TORCH_CHECK(P >= 2 && kDff % P == 0 && (kDff / P) % 16 == 0,
                "TP degree must divide dff with 16 | dff/P (wgmma K-step); got P=", P);
    const int F = kDff / P;
    CHK(X, "X", (int64_t)kM * kD); CHK(dY1, "dY1", (int64_t)kM * kD);
    CHK(W0s, "W0s", (int64_t)kDff * kD); CHK(b0s, "b0s", (int64_t)kDff);
    CHK(W1s, "W1s", (int64_t)kD * kDff); CHK(b1, "b1", (int64_t)kD);
    CHK(Y1, "Y1", (int64_t)P * kM * kD); CHK(dX, "dX", (int64_t)P * kM * kD);
    CHK(dW0, "dW0", (int64_t)kDff * kD); CHK(db0, "db0", (int64_t)kDff);
    CHK(dW1, "dW1", (int64_t)kD * kDff); CHK(db1, "db1", (int64_t)P * kD);

    auto opts_f = X.options();
    auto opts_b = X.options().dtype(at::kBFloat16);
    // bf16 inputs (round once host-side via torch, exact production round).
    at::Tensor Xb  = X.to(at::kBFloat16).contiguous();
    at::Tensor dYb = dY1.to(at::kBFloat16).contiguous();
    // Scratch + the loopback symmetric heap (zero-init: determinism of the heap
    // content before first publish; the publish overwrites every read element).
    at::Tensor heap = at::zeros({(int64_t)P * 2 * kM * kD}, opts_f);
    at::Tensor pre  = at::zeros({(int64_t)P * kM * F}, opts_b);
    at::Tensor G    = at::zeros({(int64_t)P * kM * F}, opts_b);
    at::Tensor dG   = at::zeros({(int64_t)P * kM * F}, opts_f);
    at::Tensor dY0  = at::zeros({(int64_t)P * kM * F}, opts_b);
    at::Tensor barb = at::zeros({2}, X.options().dtype(at::kInt));  // arrived|generation

    TpArgs a{};
    a.X   = reinterpret_cast<const __nv_bfloat16*>(Xb.data_ptr());
    a.dY1 = reinterpret_cast<const __nv_bfloat16*>(dYb.data_ptr());
    a.W0s = W0s.data_ptr<float>(); a.b0s = b0s.data_ptr<float>();
    a.W1s = W1s.data_ptr<float>(); a.b1 = b1.data_ptr<float>();
    a.heap = heap.data_ptr<float>();
    a.pre = reinterpret_cast<__nv_bfloat16*>(pre.data_ptr());
    a.G   = reinterpret_cast<__nv_bfloat16*>(G.data_ptr());
    a.dG  = dG.data_ptr<float>();
    a.dY0 = reinterpret_cast<__nv_bfloat16*>(dY0.data_ptr());
    a.Y1 = Y1.data_ptr<float>(); a.dX = dX.data_ptr<float>();
    a.dW0 = dW0.data_ptr<float>(); a.db0 = db0.data_ptr<float>();
    a.dW1 = dW1.data_ptr<float>(); a.db1 = db1.data_ptr<float>();
    a.P = P;

    unsigned* barp = reinterpret_cast<unsigned*>(barb.data_ptr<int>());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tp_loopback_block_kernel<<<P, kBlk, 0, stream>>>(a, barp, barp + 1);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tp_loopback_block_kernel launch failed");
}

void tp_reference_run(int64_t P_,
                      at::Tensor X, at::Tensor dY1,
                      at::Tensor W0, at::Tensor b0, at::Tensor W1, at::Tensor b1,
                      at::Tensor W0s, at::Tensor b0s, at::Tensor W1s,
                      at::Tensor Y1_ref, at::Tensor dX_ref,
                      at::Tensor dW0_ref, at::Tensor db0_ref,
                      at::Tensor dW1_ref, at::Tensor db1_ref,
                      at::Tensor Y1_chunked, at::Tensor dX_chunked) {
    const int P = (int)P_;
    const int F = kDff / P;
    CHK(X, "X", (int64_t)kM * kD); CHK(dY1, "dY1", (int64_t)kM * kD);
    CHK(W0, "W0", (int64_t)kDff * kD); CHK(b0, "b0", (int64_t)kDff);
    CHK(W1, "W1", (int64_t)kD * kDff); CHK(b1, "b1", (int64_t)kD);
    CHK(W0s, "W0s", (int64_t)kDff * kD); CHK(b0s, "b0s", (int64_t)kDff);
    CHK(W1s, "W1s", (int64_t)kD * kDff);
    CHK(Y1_ref, "Y1_ref", (int64_t)kM * kD); CHK(dX_ref, "dX_ref", (int64_t)kM * kD);
    CHK(dW0_ref, "dW0_ref", (int64_t)kDff * kD); CHK(db0_ref, "db0_ref", (int64_t)kDff);
    CHK(dW1_ref, "dW1_ref", (int64_t)kD * kDff); CHK(db1_ref, "db1_ref", (int64_t)kD);
    CHK(Y1_chunked, "Y1_chunked", (int64_t)kM * kD); CHK(dX_chunked, "dX_chunked", (int64_t)kM * kD);

    auto opts_f = X.options();
    auto opts_b = X.options().dtype(at::kBFloat16);
    at::Tensor Xb  = X.to(at::kBFloat16).contiguous();
    at::Tensor dYb = dY1.to(at::kBFloat16).contiguous();
    at::Tensor pre_full = at::zeros({(int64_t)kM * kDff}, opts_b);
    at::Tensor G_full   = at::zeros({(int64_t)kM * kDff}, opts_b);
    at::Tensor dY0_full = at::zeros({(int64_t)kM * kDff}, opts_b);
    at::Tensor scratch  = at::zeros({(int64_t)kM * kDff}, opts_f);
    at::Tensor pre_o    = at::zeros({(int64_t)kM * F}, opts_b);
    at::Tensor G_o      = at::zeros({(int64_t)kM * F}, opts_b);
    at::Tensor dY0_o    = at::zeros({(int64_t)kM * F}, opts_b);
    at::Tensor part     = at::zeros({(int64_t)kM * kD}, opts_f);

    TpRefArgs a{};
    a.X = reinterpret_cast<const __nv_bfloat16*>(Xb.data_ptr());
    a.dY1 = reinterpret_cast<const __nv_bfloat16*>(dYb.data_ptr());
    a.W0 = W0.data_ptr<float>(); a.b0 = b0.data_ptr<float>();
    a.W1 = W1.data_ptr<float>(); a.b1 = b1.data_ptr<float>();
    a.W0s = W0s.data_ptr<float>(); a.b0s = b0s.data_ptr<float>();
    a.W1s = W1s.data_ptr<float>();
    a.pre_full = reinterpret_cast<__nv_bfloat16*>(pre_full.data_ptr());
    a.G_full   = reinterpret_cast<__nv_bfloat16*>(G_full.data_ptr());
    a.dY0_full = reinterpret_cast<__nv_bfloat16*>(dY0_full.data_ptr());
    a.scratch_f = scratch.data_ptr<float>();
    a.pre_o = reinterpret_cast<__nv_bfloat16*>(pre_o.data_ptr());
    a.G_o   = reinterpret_cast<__nv_bfloat16*>(G_o.data_ptr());
    a.dY0_o = reinterpret_cast<__nv_bfloat16*>(dY0_o.data_ptr());
    a.part = part.data_ptr<float>();
    a.Y1_ref = Y1_ref.data_ptr<float>(); a.dX_ref = dX_ref.data_ptr<float>();
    a.dW0_ref = dW0_ref.data_ptr<float>(); a.db0_ref = db0_ref.data_ptr<float>();
    a.dW1_ref = dW1_ref.data_ptr<float>(); a.db1_ref = db1_ref.data_ptr<float>();
    a.Y1_chunked = Y1_chunked.data_ptr<float>(); a.dX_chunked = dX_chunked.data_ptr<float>();
    a.P = P;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tp_reference_kernel<<<1, kBlk, 0, stream>>>(a);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tp_reference_kernel launch failed");
}

// Pack the dense per-rank shards on CPU using the tp_layer.cuh index maps (the
// SINGLE source of the pack arithmetic — no Python re-derivation). Returns
// (W0s [dff,d], b0s [dff], W1s [d,dff]) as the per-rank dense concatenations.
std::vector<at::Tensor> tp_pack_pair_shards(int64_t P_, at::Tensor W0, at::Tensor b0,
                                            at::Tensor W1) {
    const int P = (int)P_;
    TORCH_CHECK(P >= 2 && kDff % P == 0, "P must divide dff=", kDff);
    TORCH_CHECK(!W0.is_cuda() && !b0.is_cuda() && !W1.is_cuda(),
                "pack expects CPU fp32 tensors (move to CUDA after packing)");
    TORCH_CHECK(W0.scalar_type() == at::kFloat && W0.is_contiguous() && W0.numel() == (int64_t)kDff * kD);
    TORCH_CHECK(b0.scalar_type() == at::kFloat && b0.is_contiguous() && b0.numel() == (int64_t)kDff);
    TORCH_CHECK(W1.scalar_type() == at::kFloat && W1.is_contiguous() && W1.numel() == (int64_t)kD * kDff);
    const int F = kDff / P;
    at::Tensor W0s = at::empty_like(W0), b0s = at::empty_like(b0), W1s = at::empty_like(W1);
    const float* w0 = W0.data_ptr<float>(); float* w0s = W0s.data_ptr<float>();
    const float* b0p = b0.data_ptr<float>(); float* b0sp = b0s.data_ptr<float>();
    const float* w1 = W1.data_ptr<float>(); float* w1s = W1s.data_ptr<float>();
    for (int r = 0; r < P; ++r) {
        int lo, hi;
        tp::tp_own_range(kDff, P, r, &lo, &hi);            // Col split rows of W0
        for (int i = 0; i < F; ++i) {
            const int fi = tp::tp_col_full_row(lo, i);
            for (int k = 0; k < kD; ++k)
                w0s[((int64_t)r * F + i) * kD + k] = w0[(int64_t)fi * kD + k];
            b0sp[(int64_t)r * F + i] = b0p[fi];
        }
        for (int j = 0; j < kD; ++j)                       // Row split cols of W1
            for (int c = 0; c < F; ++c) {
                const int fc = tp::tp_row_full_col(lo, c);
                w1s[((int64_t)r * kD + j) * F + c] = w1[(int64_t)j * kDff + fc];
            }
    }
    return {W0s, b0s, W1s};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TP loopback gate binding (design §5.1/§5.2; LoopbackTransport — "
              "the NVSHMEM swap point lives in tp_transport.cuh).";
    m.def("tp_loopback_run", &tp_loopback_run,
          "Run the TP=P virtual-rank col→row pair fwd+bwd over the loopback "
          "symmetric heap (fixed-order all-reduce at the two comm points).");
    m.def("tp_reference_run", &tp_reference_run,
          "Run the 1-CTA unsharded + serial-chunked-order references.");
    m.def("tp_pack_pair_shards", &tp_pack_pair_shards,
          "Pack dense per-rank (W0 col / W1 row) shards via the tp_layer maps.");
    m.attr("M") = kM; m.attr("D") = kD; m.attr("DFF") = kDff;
}
