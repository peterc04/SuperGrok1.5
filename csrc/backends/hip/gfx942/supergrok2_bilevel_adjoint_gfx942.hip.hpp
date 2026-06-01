#pragma once
#ifndef CSRC_BACKENDS_HIP_GFX942_SUPERGROK2_BILEVEL_ADJOINT_GFX942_HIP_HPP_
#define CSRC_BACKENDS_HIP_GFX942_SUPERGROK2_BILEVEL_ADJOINT_GFX942_HIP_HPP_
// ============================================================================
// supergrok2_bilevel_adjoint_gfx942.hip.hpp — FIRST device-side AMDGCN cut of
// the SuperGrok2 (SG2) CSA/HCA bilevel-adjoint (reverse-mode VJP) for gfx942
// (CDNA3 / MI300X).
//
// WHY THIS FILE EXISTS
// --------------------
// The SG2 backward on BOTH backends (sm_90 + gfx942) currently routes the full
// reverse-mode VJP through the vendor-neutral ATen oracle
//   csrc/algorithms/supergrok2_bilevel_adjoint.h
// (host orchestration: bilevel_forward_save + bilevel_backward_driver). That
// header is the authoritative math (input_proj+sort → CSA → HCA → GRU → PEER →
// smart_grad and its adjoint) and remains the CPU/host fallback.
//
// This header adds, FOR THE FIRST TIME on gfx942, a set of hand-written AMDGCN
// __global__ adjoint kernels for the reverse-pipeline stages that are MFMA- /
// DPP-shaped, so the gfx942 SG2 backward is no longer purely ATen-by-default at
// the device level. The kernels mirror the AMDGCN idioms of the SG2 FORWARD
// device kernels in
//   grokking_optimizers/kernels/gfx942/supergrok2_gfx942.hip.hpp  (§5)
// and are built on the verified primitives in
//   csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp.
//
// DEVICE ADJOINT STAGES PROVIDED (device pass, real AMDGCN intrinsics):
//   §A1  PEER expert-MLP backward       — DPP-reduced expert grads + d_routing
//        (mirrors sg2_bilevel::peer_head_backward, oracle lines 610-707).
//   §A2  GRU-gate backward              — sigmoid/tanh adjoints, weight grads
//        (mirrors bilevel_backward_driver GRU block, oracle lines 777-813).
//   §A3  Attention-context backward     — the CSA/HCA shared adjoint core:
//        d_ctx → d_P (via Vᵀ, MFMA) → softmax-backward (DPP) → d_S → d_Q (via
//        K, MFMA) / d_Kᵀ (via Q, MFMA) / d_V (via Pᵀ, MFMA). Reuses the §5
//        forward MFMA tile helper (transposed operands for the adjoint matmuls).
//        (mirrors hca_backward / csa_backward attention core, oracle 198-501.)
//   §A4  softmax-backward (element-local, grid-stride) — d_S = P⊙(d_P − Σ d_P·P)
//        the joint-axis softmax VJP; also used standalone for the routing-softmax.
//
// HONESTY / SCOPE (read me)
// -------------------------
//   * This is a BLIND first device cut: there is NO hipcc and NO MI300X in this
//     environment, so NONE of it is numerically validated. Every device kernel
//     here is 🟡 (compile-verified for gfx942 only via scripts/amdgcn_check.sh)
//     until it runs on MI300X with parity against the ATen oracle.
//   * It is COMPILE-VERIFIED for gfx942: it passes the clang __AMDGCN__ device
//     gate (scripts/amdgcn_check.sh), which catches the real builtin-signature
//     bugs (bf16 MFMA operands must be short[4]/bf16x4, not u32x4; DPP-ctrl /
//     cvt byte-index / sched_group args must be compile-time constants).
//   * The element-local scatter pieces of the adjoint (the index_add_ scatters
//     into d_k_tok/d_v_tok/d_c_k/d_c_v over gather/sel/window indices, the
//     input_proj scatter via sort_indices, and the projection-weight d_W =
//     d_yᵀ·x accumulations) are NOT reimplemented as device kernels here — on a
//     real hipcc build they are rocPRIM-shaped scatter / small rocBLAS GEMMs and
//     stay in the ATen oracle, exactly as the forward's MoE-compaction tail does.
//     §A3 produces the d_Q / d_Kᵀ / d_V context-attention adjoint tiles (the
//     MFMA-justified work); the scatter-to-token-grads + weight-grad reductions
//     are the documented host tail.
//   * The discrete top-k index sets (PEER routing indices, CSA indexer top-k)
//     are stop-gradient routing decisions — identical to the oracle's treatment
//     (torch.topk reports no grad to the indices). Grads flow only through the
//     gathered VALUES. See the oracle notes (lines 275-279, 296-308).
//
// COMPILE ROUTING (two passes, one header) — mirrors the §5 forward pattern:
//   * HOST pass  (`!__AMDGCN__`): exposes the launch entry points and documents
//     that the vendor-neutral ATen adjoint remains the fallback. Pulls in the
//     oracle header so the host launchers can delegate. Does NOT define
//     __global__ kernels (the host `.hip.cpp` TU routes through g++/clang++).
//   * DEVICE pass (`__AMDGCN__` — the gate — AND `__HIPCC__` — the hipcc device
//     pass): the hand-written AMDGCN adjoint kernels. The host TU launches them
//     via hipLaunchKernelGGL on a migrated hardware build (see §A.LAUNCH).
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics +
// rocprof MFMA-utilization are deferred (see HARDWARE_VALIDATION.md, Stage 5).
// ============================================================================

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST pass — launch entry points + the vendor-neutral ATen fallback note.
// The free-standing AMDGCN device gate (__AMDGCN__) does NOT see this block
// (torch/extension.h pulls in ATen, which the free-standing device target cannot
// resolve). On a real hipcc build the host pass compiles this and launches the
// device §A kernels via hipLaunchKernelGGL (see §A.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if !defined(__AMDGCN__)
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cstdint>

// The authoritative reverse-mode VJP math. On a CPU/host build (and until the
// device §A kernels below are wired + MI300X-validated) THIS is the live SG2
// backward path on gfx942 — the launchers delegate here. Bit-for-bit shared
// with sm_90.
#include "csrc/algorithms/supergrok2_bilevel_adjoint.h"

namespace sg { namespace gfx942 { namespace bilevel_adjoint {

namespace sg2adj = ::sg::algorithms::sg2_bilevel;

// ─────────────────────────────────────────────────────────────────────────
//  Live-path selector. The device §A AMDGCN adjoint kernels are compile-
//  verified for gfx942 but NOT yet numerically validated on MI300X, so the
//  default live backward path is the ATen oracle. Flip SG2_ADJOINT_GFX942_LIVE
//  to 1 (and wire §A.LAUNCH on a hipcc/hardware build) to drive the device
//  kernels instead. Kept here so the choice is explicit and greppable.
// ─────────────────────────────────────────────────────────────────────────
#ifndef SG2_ADJOINT_GFX942_LIVE
#define SG2_ADJOINT_GFX942_LIVE 0
#endif

// Host fallback entry: run the full vendor-neutral reverse-mode VJP. This is a
// thin re-export of the oracle driver so a gfx942 host TU has a single, clearly
// named symbol to call. The signature intentionally matches the oracle driver
// 1:1 (see csrc/algorithms/supergrok2_bilevel_adjoint.h:713).
static inline void bilevel_backward_aten_fallback(
    const torch::Tensor& d_smart_grad, float rescale,
    const sg2adj::SavedActs& S,
    const torch::Tensor& input_proj_W,
    const torch::Tensor& csa_q_W, const torch::Tensor& csa_k_W,
    const torch::Tensor& csa_v_W, const torch::Tensor& csa_compress_w,
    const torch::Tensor& csa_idx_DQ, const torch::Tensor& csa_idx_UQ,
    const torch::Tensor& csa_idx_K, const torch::Tensor& csa_out_W,
    const torch::Tensor& hca_q_W, const torch::Tensor& hca_k_W,
    const torch::Tensor& hca_v_W, const torch::Tensor& hca_out_W,
    const torch::Tensor& gru_Wz, const torch::Tensor& gru_Wr,
    const torch::Tensor& gru_Wh,
    const std::vector<torch::Tensor>& peer_Wq,
    const std::vector<torch::Tensor>& prod_A,
    const std::vector<torch::Tensor>& prod_B,
    const torch::Tensor& expert_W1, const torch::Tensor& expert_b1,
    const torch::Tensor& expert_W2, const torch::Tensor& expert_b2,
    int64_t d_model, int64_t num_heads, int64_t gru_hidden,
    int64_t pk_dim, int64_t topk, int64_t expert_hidden,
    int64_t csa_compress, int64_t csa_window, int64_t csa_topk,
    int64_t hca_compress, int64_t indexer_rank,
    torch::Tensor& d_input_proj_W, torch::Tensor& d_input_proj_b,
    torch::Tensor& d_csa_q_W, torch::Tensor& d_csa_k_W, torch::Tensor& d_csa_v_W,
    torch::Tensor& d_csa_compress_w, torch::Tensor& d_csa_idx_DQ,
    torch::Tensor& d_csa_idx_UQ, torch::Tensor& d_csa_idx_K,
    torch::Tensor& d_csa_out_W,
    torch::Tensor& d_hca_q_W, torch::Tensor& d_hca_k_W, torch::Tensor& d_hca_v_W,
    torch::Tensor& d_hca_out_W,
    torch::Tensor& d_gru_Wz, torch::Tensor& d_gru_bz,
    torch::Tensor& d_gru_Wr, torch::Tensor& d_gru_br,
    torch::Tensor& d_gru_Wh, torch::Tensor& d_gru_bh,
    std::vector<torch::Tensor>& d_peer_Wq,
    std::vector<torch::Tensor>& d_prod_A, std::vector<torch::Tensor>& d_prod_B,
    torch::Tensor& d_expert_W1, torch::Tensor& d_expert_b1,
    torch::Tensor& d_expert_W2, torch::Tensor& d_expert_b2)
{
    sg2adj::bilevel_backward_driver(
        d_smart_grad, rescale, S,
        input_proj_W, csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
        csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
        hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        gru_Wz, gru_Wr, gru_Wh, peer_Wq, prod_A, prod_B,
        expert_W1, expert_b1, expert_W2, expert_b2,
        d_model, num_heads, gru_hidden, pk_dim, topk, expert_hidden,
        csa_compress, csa_window, csa_topk, hca_compress, indexer_rank,
        d_input_proj_W, d_input_proj_b,
        d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
        d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_csa_out_W,
        d_hca_q_W, d_hca_k_W, d_hca_v_W, d_hca_out_W,
        d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
        d_peer_Wq, d_prod_A, d_prod_B,
        d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2);
}

// ── §A.LAUNCH (host-side wiring note) ────────────────────────────────────────
// On a real hipcc / MI300X build, the host TU launches the device §A kernels
// instead of routing the MFMA/DPP-shaped adjoint stages through ATen. bf16
// activations flow as raw `short` bit-patterns; one wavefront owns one tile/row:
//
//   // PEER expert-MLP backward (one wavefront per token row, per head):
//   dim3 grid(N), block(64);
//   hipLaunchKernelGGL((native::sg2_peer_route_bwd_kernel<4>), grid, block, 0, st,
//       peer_input_bf16, g_col, Wq_bf16, A_bf16, B_bf16,
//       eW1_bf16, eW2_bf16, d_head_out, /*out grads*/ d_eW1, d_eW2, ...);
//
//   // GRU-gate backward (one wavefront per element row):
//   hipLaunchKernelGGL(native::sg2_gru_gate_bwd_kernel, dim3(N), block, lds, st,
//       x_in, h_old, z, r, h_tilde, Wz_bf16, Wr_bf16, Wh_bf16,
//       d_h_new, /*out*/ d_x, d_h_old, dWz, dWr, dWh, dbz, dbr, dbh,
//       in_dim, hidden);
//
//   // Attention-context backward (one wavefront per head):
//   hipLaunchKernelGGL((native::sg2_attn_ctx_bwd_kernel<4>), dim3(H), block, lds,
//       st, q_bf16, cK_bf16, cV_bf16, attn_probs, d_ctx_bf16,
//       /*out*/ d_q, d_cK, d_cV, N, Lc, scale);
//
// 🟡 DEFERRED: the live launch + hipcc link is MI300X-gated (no hipcc / no
// device here). Until then SG2_ADJOINT_GFX942_LIVE==0 and the host launchers
// delegate to bilevel_backward_aten_fallback() above (the proven oracle).

}}} // namespace sg::gfx942::bilevel_adjoint
#endif  // !defined(__AMDGCN__)  — end host pass (A)

// ════════════════════════════════════════════════════════════════════════════
// (B) DEVICE pass — real hand-written AMDGCN adjoint kernels (§A1..§A4).
// Compiled by the AMDGCN device pass only: the Stage-5 gate (__AMDGCN__, no
// hipcc) AND the hipcc device pass (__HIPCC__). The host `.hip.cpp` TU never
// sees it — that pass keeps the ATen orchestration above (which LAUNCHES these
// kernels via hipLaunchKernelGGL on a migrated build, see §A.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)
#include "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp"

// ── gate-only launch-builtin shim (verbatim from the §5 forward exemplar) ─────
// The free-standing AMDGCN device gate stubs out <hip/hip_runtime.h>, so the
// launch builtins (threadIdx/blockIdx/blockDim/gridDim, __global__, __shared__)
// HIP normally provides are absent. Model them with the AMDGCN workitem ISA
// builtins so the device bodies type-check. Active ONLY on the bare gate.
#if defined(__AMDGCN__) && !defined(__HIPCC__)
#ifndef GROK_GFX942_LAUNCH_SHIM_
#define GROK_GFX942_LAUNCH_SHIM_
struct GrokTidX { __device__ operator unsigned() const { return __builtin_amdgcn_workitem_id_x(); } };
struct GrokBidX { __device__ operator unsigned() const { return __builtin_amdgcn_workgroup_id_x(); } };
struct GrokBdimX{ __device__ operator unsigned() const { return __builtin_amdgcn_workgroup_size_x(); } };
struct GrokGdimX{ __device__ operator unsigned() const { return __builtin_amdgcn_grid_size_x()
                                                              / __builtin_amdgcn_workgroup_size_x(); } };
struct GrokThreadIdx { GrokTidX  x; };
struct GrokBlockIdx  { GrokBidX  x; };
struct GrokBlockDim  { GrokBdimX x; };
struct GrokGridDim   { GrokGdimX x; };
static GrokThreadIdx threadIdx;
static GrokBlockIdx  blockIdx;
static GrokBlockDim  blockDim;
static GrokGridDim   gridDim;
#ifndef __global__
#define __global__ __attribute__((amdgpu_kernel))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#endif  // GROK_GFX942_LAUNCH_SHIM_
#endif  // bare gate

// ============================================================================
// §A  AMD-NATIVE device adjoint kernels (Stage 5 — SG2 bilevel backward, first
// device cut). Built on the shared, compiler-verified primitives in
//   csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp   (namespace amd = …),
// mirroring the §5 forward AMDGCN idioms (MFMA tile, DPP butterflies, streaming
// loads). Self-contained: under the free-standing gate the only reachable
// headers are the amdgcn_primitives stub set (no <cmath>/<cstdint>/bfloat16), so
// the math uses clang __builtin_* (valid under hipcc too) and a local bf16<->f32
// bit codec.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics
// are deferred (HARDWARE_VALIDATION.md, Stage 5).
// ============================================================================
namespace sg { namespace gfx942 { namespace models { namespace supergrok2 {
namespace native_adjoint {

namespace amd = ::sg::gfx942::amdgcn;

// CDNA3 wavefront width (== amd::kWave).
static constexpr int kWave = 64;

// ── device math shims (clang builtins; resolve under the bare gate AND hipcc) ─
__device__ __forceinline__ float dexpf(float x)  { return __builtin_expf(x); }
__device__ __forceinline__ float dtanhf(float x) { return __builtin_tanhf(x); }
__device__ __forceinline__ float dfmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
__device__ __forceinline__ float dsigmoidf(float x) { return 1.f / (1.f + dexpf(-x)); }

// ── bf16 <-> f32 bit codec (self-contained: the gate has no hip_bfloat16) ─────
// bf16 is the top 16 bits of an f32; operands flow to the MFMA wrappers as the
// raw `short` bit-pattern (matches amd::mfma_bf16_* which take const short[4]).
__device__ __forceinline__ float bf16_to_f32(short h) {
    unsigned u = static_cast<unsigned>(static_cast<unsigned short>(h)) << 16;
    return __builtin_bit_cast(float, u);
}
__device__ __forceinline__ short f32_to_bf16(float f) {
    unsigned u = __builtin_bit_cast(unsigned, f);
    unsigned lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<short>(static_cast<unsigned short>(u >> 16));
}

// ════════════════════════════════════════════════════════════════════════════
// §A.0  Shared MFMA tile + driver for the adjoint matmuls.
//
// IDENTICAL operand contract to the §5 forward mfma_tile_16x16 /
// mfma_matmul_bf16: C[M,N] = A[M,K] · Wᵀ where W is [N,K] row-major (= Bᵀ
// already), contracting over K. The reverse pass simply feeds TRANSPOSED
// operands to realise the adjoint GEMMs:
//
//   forward  S[N,Lc]   = Q[N,D]  · Kᵀ           → A=Q[N,D],   W=K[Lc,D]
//   forward  O[N,D]    = P[N,Lc] · V            → A=P[N,Lc],  W=Vᵀ[D,Lc]
//   adjoint  dP[N,Lc]  = dO[N,D] · Vᵀ           → A=dO[N,D],  W=V[Lc,D]
//   adjoint  dV[Lc,D]  = Pᵀ[Lc,N] · dO          → A=Pᵀ[Lc,N], W=dOᵀ[D,N]
//   adjoint  dQ[N,D]   = dS[N,Lc] · K           → A=dS[N,Lc], W=Kᵀ[D,Lc]
//   adjoint  dKᵀ[Lc,D] = dSᵀ[Lc,N] · Q          → A=dSᵀ[Lc,N],W=Qᵀ[D,N]
//
// where in each case the second operand is supplied in the [out_cols, K]
// row-major layout the helper expects. Sub-16 K (grokking head_dim=4) is
// zero-padded inside the K loop, exactly like the forward.
// ════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void mfma_tile_16x16(
    const short* __restrict__ A,   // [M, K] row-major bf16 bits
    const short* __restrict__ W,   // [N, K] row-major bf16 bits  (= Bᵀ)
    float*       __restrict__ C,   // [M, N] row-major f32
    int M, int N, int K,
    int tile_row, int tile_col)    // 16-aligned output-tile origin
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const int half = lane / 16;        // which 16-K group this lane feeds (0..3)
    const int idx  = lane % 16;        // row (A) / col (W) within the tile

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    const int aRow = tile_row + idx;   // A row this lane streams
    const int bCol = tile_col + idx;   // W row (= output col) this lane streams

    for (int k0 = 0; k0 < K; k0 += 16) {
        short af[4], bf[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = k0 + half * 4 + j;   // 64 lanes cover the 16 K-values ×4
            af[j] = (aRow < M && k < K) ? amd::streaming_load(&A[aRow * K + k]) : (short)0;
            bf[j] = (bCol < N && k < K) ? amd::streaming_load(&W[bCol * K + k]) : (short)0;
        }
        amd::mfma_bf16_16x16x16(acc, af, bf);
        amd::sched_group_barrier<0x008, 1>();   // 1 MFMA …
        amd::sched_group_barrier<0x100, 2>();   // … then 2 VMEM reads
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int outRow = tile_row + 4 * half + r;
        int outCol = tile_col + idx;
        if (outRow < M && outCol < N) C[outRow * N + outCol] = acc[r];
    }
}

// Whole-matrix driver: one wavefront strides the 16×16 output-tile lattice.
__device__ __forceinline__ void mfma_matmul_bf16(
    const short* __restrict__ A, const short* __restrict__ W,
    float* __restrict__ C, int M, int N, int K)
{
    const int tilesM = (M + 15) / 16;
    const int tilesN = (N + 15) / 16;
    const int nTiles = tilesM * tilesN;
    for (int t = 0; t < nTiles; ++t) {
        int tr = (t / tilesN) * 16;
        int tc = (t % tilesN) * 16;
        mfma_tile_16x16(A, W, C, M, N, K, tr, tc);
    }
}

// Transpose helper: pack src[R,Cc] (bf16 bits) → dst[Cc,R] (bf16 bits) in LDS.
__device__ __forceinline__ void transpose_bf16(
    const short* __restrict__ src, short* __restrict__ dst, int R, int Cc)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    for (int idx = lane; idx < R * Cc; idx += kWave) {
        int r = idx / Cc, c = idx % Cc;
        dst[c * R + r] = src[idx];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §A4  softmax-backward over a length-L row held by this wavefront (the joint
// attention axis [topk+wsz], or the PEER candidate axis). Given the forward
// softmax probs P[0,L) and the upstream grad dP[0,L) (both lane-strided in LDS),
// writes dS = P ⊙ (dP − Σ_j dP_j·P_j) back into dS_row. This is the exact
// softmax VJP the oracle uses (supergrok2_bilevel_adjoint.h:229-231):
//   dot = (dP*P).sum(-1);  dS = P*(dP - dot).
// The reduction is a DPP sum butterfly (§2.6); no LDS round-trip.
// ════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void softmax_row_backward(
    const float* __restrict__ P,    // [L] forward softmax probs
    const float* __restrict__ dP,   // [L] upstream grad wrt probs
    float* __restrict__ dS,         // [L] out: grad wrt pre-softmax scores
    int L)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    float dot = 0.f;
    for (int j = lane; j < L; j += kWave) dot += dP[j] * P[j];
    dot = amd::wave_reduce_add_dpp(dot);          // §2.6 DPP sum butterfly
    for (int j = lane; j < L; j += kWave)
        dS[j] = P[j] * (dP[j] - dot);
}

// ════════════════════════════════════════════════════════════════════════════
// §A3  Attention-context backward — the CSA/HCA shared adjoint core.
//
// Mirrors hca_backward / csa_backward attention math (oracle 198-501) for ONE
// head/wavefront, single-head dense attention over the compressed entries
// (CSA: top-k selected, presented pre-gathered as cK/cV[Lc,D]; HCA: all
// compressed). Inputs: q[N,D], cK/cV[Lc,D] (bf16 bits), the saved forward probs
// P[N,Lc] (f32), and the upstream context grad d_ctx[N,D] (bf16 bits).
//
// Reverse pipeline (per head):
//   O = P·cV          ⇒  dP = d_ctx·cVᵀ          (MFMA: A=d_ctx[N,D], W=cV[Lc,D])
//                        dcV = Pᵀ·d_ctx           (MFMA: A=Pᵀ[Lc,N], W=d_ctxᵀ[D,N])
//   P = softmax(S)    ⇒  dS = P⊙(dP − Σ dP·P)     (§A4 DPP softmax-backward)
//   S = Q·cKᵀ·scale   ⇒  dQ  = (dS·scale)·cK      (MFMA: A=dS[N,Lc], W=cKᵀ? no →
//                                                   W=cK[Lc,D] is already [out,K])
//                        dcK = (dSᵀ·scale)·Q       (MFMA: A=dSᵀ[Lc,N], W=Qᵀ[D,N])
//
// Writes d_q[N,D], d_cK[Lc,D], d_cV[Lc,D] (bf16 bits). The downstream scatter of
// d_cK/d_cV back into token grads (compression / window / sel scatters) and the
// q/k/v-projection weight grads are the documented host (ATen oracle) tail.
//
// LDS scratch layout (caller-provided): dPf[N*Lc] f32, dSf[N*Lc] f32,
// d_ctxf[N*D] f32, accf[max(N*D, Lc*D)] f32, and a bf16 pack region
// pk[N*Lc + N*D + Lc*N + D*N + ...] — sized by sg2_attn_bwd_lds_bytes().
// ════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void attention_ctx_backward_core(
    const short* __restrict__ q,      // [N, D] bf16 bits
    const short* __restrict__ cK,     // [Lc, D] bf16 bits (compressed keys)
    const short* __restrict__ cV,     // [Lc, D] bf16 bits (compressed values)
    const float* __restrict__ P,      // [N, Lc] forward softmax probs (f32)
    const short* __restrict__ d_ctx,  // [N, D] bf16 bits (upstream ctx grad)
    short* __restrict__ d_q,          // [N, D] bf16 bits (out)
    short* __restrict__ d_cK,         // [Lc, D] bf16 bits (out)
    short* __restrict__ d_cV,         // [Lc, D] bf16 bits (out)
    float* __restrict__ dPf,          // LDS: [N*Lc] f32
    float* __restrict__ dSf,          // LDS: [N*Lc] f32
    float* __restrict__ accf,         // LDS: [max(N*D, Lc*D)] f32
    short* __restrict__ Pbf,          // LDS: [N*Lc] bf16  (P / Pᵀ)
    short* __restrict__ Tbf,          // LDS: [max(Lc*N, D*N, Lc*D)] bf16 scratch
    int N, int Lc, int D, float scale)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;

    // ── (1) dP = d_ctx · cVᵀ.  O[N,D]=P[N,Lc]·cV[Lc,D]; adjoint dP = dO·cVᵀ.
    //     mfma helper computes A·Wᵀ with W=[out_cols,K] row-major. Here out_cols
    //     = Lc, K = D, and W = cV[Lc,D] already (cV[j][d] contracts with d_ctx[i][d]).
    mfma_matmul_bf16(d_ctx, cV, dPf, N, Lc, D);
    amd::wait_vmcnt0();

    // ── (2) softmax backward per row: dS = P ⊙ (dP − Σ dP·P).
    for (int i = 0; i < N; ++i)
        softmax_row_backward(&P[i * Lc], &dPf[i * Lc], &dSf[i * Lc], Lc);
    // fold the attention scale (S = Q·cKᵀ·scale ⇒ dS *= scale) into dS.
    for (int idx = lane; idx < N * Lc; idx += kWave) dSf[idx] *= scale;
    amd::workgroup_barrier_release();

    // ── (3) dQ = dS · cK.  S[N,Lc]=Q[N,D]·cKᵀ; adjoint dQ = dS·cK (contract Lc).
    //     mfma A·Wᵀ with A=dS[N,Lc], W=cKᵀ?  We need dQ[N,D]=Σ_j dS[i][j]·cK[j][d];
    //     that is A=dS[N,Lc] (K=Lc) and W=[out_cols=D, K=Lc] = cKᵀ[D,Lc]. Pack
    //     dS→bf16 and cK→cKᵀ in LDS, then MFMA.
    short* dSbf = Pbf;                          // reuse Pbf region: N*Lc shorts
    for (int idx = lane; idx < N * Lc; idx += kWave) dSbf[idx] = f32_to_bf16(dSf[idx]);
    short* cKt = Tbf;                            // D*Lc shorts (cKᵀ)
    for (int idx = lane; idx < Lc * D; idx += kWave) {
        int j = idx / D, d = idx % D;            // cK[j][d] → cKᵀ[d][j]
        cKt[d * Lc + j] = cK[idx];
    }
    amd::workgroup_barrier_release();
    mfma_matmul_bf16(dSbf, cKt, accf, N, D, Lc); // dQ[N,D]
    amd::wait_vmcnt0();
    for (int idx = lane; idx < N * D; idx += kWave) d_q[idx] = f32_to_bf16(accf[idx]);
    amd::workgroup_barrier_release();

    // ── (4) dcK = dSᵀ · Q.  adjoint dcK[Lc,D]=Σ_i dS[i][j]·Q[i][d]; A=dSᵀ[Lc,N]
    //     (K=N), W=[out_cols=D, K=N]=Qᵀ[D,N]. Pack dSᵀ and Qᵀ to bf16 in LDS.
    short* dSt = Pbf;                            // Lc*N shorts (dSᵀ)
    for (int idx = lane; idx < N * Lc; idx += kWave) {
        int i = idx / Lc, j = idx % Lc;          // dS[i][j] → dSᵀ[j][i]
        dSt[j * N + i] = f32_to_bf16(dSf[idx]);
    }
    short* Qt = Tbf;                             // D*N shorts (Qᵀ)
    for (int idx = lane; idx < N * D; idx += kWave) {
        int i = idx / D, d = idx % D;            // Q[i][d] → Qᵀ[d][i]
        Qt[d * N + i] = q[idx];
    }
    amd::workgroup_barrier_release();
    mfma_matmul_bf16(dSt, Qt, accf, Lc, D, N);   // dcK[Lc,D]
    amd::wait_vmcnt0();
    for (int idx = lane; idx < Lc * D; idx += kWave) d_cK[idx] = f32_to_bf16(accf[idx]);
    amd::workgroup_barrier_release();

    // ── (5) dcV = Pᵀ · d_ctx.  O=P·cV; adjoint dcV[Lc,D]=Σ_i P[i][j]·d_ctx[i][d];
    //     A=Pᵀ[Lc,N] (K=N), W=[out_cols=D, K=N]=d_ctxᵀ[D,N]. Pack both to bf16.
    short* Pt = Pbf;                             // Lc*N shorts (Pᵀ)
    for (int idx = lane; idx < N * Lc; idx += kWave) {
        int i = idx / Lc, j = idx % Lc;          // P[i][j] → Pᵀ[j][i]
        Pt[j * N + i] = f32_to_bf16(P[idx]);
    }
    short* dCt = Tbf;                            // D*N shorts (d_ctxᵀ)
    for (int idx = lane; idx < N * D; idx += kWave) {
        int i = idx / D, d = idx % D;            // d_ctx[i][d] → d_ctxᵀ[d][i]
        dCt[d * N + i] = d_ctx[idx];
    }
    amd::workgroup_barrier_release();
    mfma_matmul_bf16(Pt, dCt, accf, Lc, D, N);   // dcV[Lc,D]
    amd::wait_vmcnt0();
    for (int idx = lane; idx < Lc * D; idx += kWave) d_cV[idx] = f32_to_bf16(accf[idx]);
}

// ════════════════════════════════════════════════════════════════════════════
// §A2  GRU-gate backward (per element row, one wavefront).
//
// Mirrors bilevel_backward_driver GRU block (oracle 777-813). Forward:
//   z = σ(Wz·[x;h] + bz) ; r = σ(Wr·[x;h] + br) ;
//   h̃ = tanh(Wh·[x; r⊙h] + bh) ; h_new = (1−z)⊙h + z⊙h̃.
// Given d_h_new[hidden], the saved gates z,r,h̃[hidden], the input x[in_dim] and
// the carried h[hidden], accumulate the weight/bias grads (via AGENT atomics, as
// many element-rows share the weights) and write d_x[in_dim] / d_h_old[hidden].
//
// Adjoints (per hidden unit u):
//   d_z = d_h_new ⊙ (h̃ − h) ;  d_h̃ = d_h_new ⊙ z ;  d_h += d_h_new ⊙ (1−z)
//   d_preh = d_h̃ ⊙ (1 − h̃²) ;  d_pre_z = d_z ⊙ z ⊙ (1−z)
//   (Wh row dotted [x ; r⊙h]) ⇒ d_x += d_prehᵀ·Wh[:, :in] ; d_(r⊙h) = d_prehᵀ·Wh[:, in:]
//   d_r = d_(r⊙h) ⊙ h ;  d_h += d_(r⊙h) ⊙ r ;  d_pre_r = d_r ⊙ r ⊙ (1−r)
//   (Wz/Wr rows dotted [x ; h]) ⇒ d_x += d_pre_{z,r}ᵀ·W{z,r}[:, :in] ;
//                                  d_h += d_pre_{z,r}ᵀ·W{z,r}[:, in:]
//   dW{z,r,h}[u, j] += d_pre_*[u]·(row input)[j] ;  db{z,r,h}[u] += d_pre_*[u].
//
// Each lane owns a strided subset of hidden units. The d_x / d_h contributions
// require a cross-unit reduction over the [in_dim+hidden] concat columns, done
// with per-column AGENT atomics into the (zero-init) out buffers — element-local
// but correct. Weight/bias grads likewise go through AGENT atomics (§2.13).
// ════════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void gru_gate_backward_row(
    const float* __restrict__ x,        // [in_dim]
    const float* __restrict__ h,        // [hidden]  (carried h_old)
    const float* __restrict__ z,        // [hidden]  saved gate
    const float* __restrict__ r,        // [hidden]  saved gate
    const float* __restrict__ h_tilde,  // [hidden]  saved candidate
    const short* __restrict__ Wz,       // [hidden, in_dim+hidden] bf16 bits
    const short* __restrict__ Wr,
    const short* __restrict__ Wh,
    const float* __restrict__ d_h_new,  // [hidden]  upstream grad
    float* __restrict__ d_x,            // [in_dim]   out (atomic-accumulated)
    float* __restrict__ d_h_old,        // [hidden]   out (atomic-accumulated)
    float* __restrict__ d_Wz, float* __restrict__ d_Wr, float* __restrict__ d_Wh,
    float* __restrict__ d_bz, float* __restrict__ d_br, float* __restrict__ d_bh,
    int in_dim, int hidden)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const int cat = in_dim + hidden;

    for (int u = lane; u < hidden; u += kWave) {
        const float zu = z[u], ru = r[u], htu = h_tilde[u], hu = h[u];
        const float dnew = d_h_new[u];

        // convex update adjoints
        float d_z      = dnew * (htu - hu);
        float d_htilde = dnew * zu;
        // d_h via (1-z)*h_old ; remaining d_h paths (r⊙h, [x;h] gate inputs)
        // accumulate below into d_h_old[u] atomically.
        amd::atomic_add_agent_f32(&d_h_old[u], dnew * (1.f - zu));

        // tanh / sigmoid pre-activation grads
        float d_preh = d_htilde * (1.f - htu * htu);
        float d_prez = d_z * zu * (1.f - zu);

        // bias grads
        amd::atomic_add_agent_f32(&d_bh[u], d_preh);
        amd::atomic_add_agent_f32(&d_bz[u], d_prez);

        // h̃ row dotted [x ; r⊙h]: scatter dW and propagate to x / (r⊙h).
        float d_rh = 0.f;  // grad wrt (r⊙h)[u]'s contribution gathered per-col below
        // Wh column loop — input part (x) and hidden part (r⊙h).
        for (int j = 0; j < in_dim; ++j) {
            float wij = bf16_to_f32(Wh[u * cat + j]);
            amd::atomic_add_agent_f32(&d_Wh[u * cat + j], d_preh * x[j]);
            amd::atomic_add_agent_f32(&d_x[j], d_preh * wij);
        }
        for (int j = 0; j < hidden; ++j) {
            float wij = bf16_to_f32(Wh[u * cat + in_dim + j]);
            float rh_j = r[j] * h[j];               // (r⊙h)[j]
            amd::atomic_add_agent_f32(&d_Wh[u * cat + in_dim + j], d_preh * rh_j);
            // d(r⊙h)[j] += d_preh * wij ; split into d_r[j], d_h[j] at unit j.
            float drh = d_preh * wij;
            // d_r[j] = drh*h[j] → through r=σ(...) → d_pre_r[j]; fold into Wr/x/h
            // here using j's own r,h. (Each (u,j) pair contributes; correctness
            // relies on AGENT-atomic accumulation across u.)
            float d_r_j   = drh * h[j];
            float d_pre_r = d_r_j * r[j] * (1.f - r[j]);
            amd::atomic_add_agent_f32(&d_h_old[j], drh * r[j]);  // d_h via r⊙h
            amd::atomic_add_agent_f32(&d_br[j], d_pre_r);
            // Wr row j dotted [x ; h]
            for (int q2 = 0; q2 < in_dim; ++q2) {
                float wrj = bf16_to_f32(Wr[j * cat + q2]);
                amd::atomic_add_agent_f32(&d_Wr[j * cat + q2], d_pre_r * x[q2]);
                amd::atomic_add_agent_f32(&d_x[q2], d_pre_r * wrj);
            }
            for (int q2 = 0; q2 < hidden; ++q2) {
                float wrj = bf16_to_f32(Wr[j * cat + in_dim + q2]);
                amd::atomic_add_agent_f32(&d_Wr[j * cat + in_dim + q2], d_pre_r * h[q2]);
                amd::atomic_add_agent_f32(&d_h_old[q2], d_pre_r * wrj);
            }
        }
        (void)d_rh;

        // z row dotted [x ; h]: scatter dWz and propagate to x / h.
        for (int j = 0; j < in_dim; ++j) {
            float wij = bf16_to_f32(Wz[u * cat + j]);
            amd::atomic_add_agent_f32(&d_Wz[u * cat + j], d_prez * x[j]);
            amd::atomic_add_agent_f32(&d_x[j], d_prez * wij);
        }
        for (int j = 0; j < hidden; ++j) {
            float wij = bf16_to_f32(Wz[u * cat + in_dim + j]);
            amd::atomic_add_agent_f32(&d_Wz[u * cat + in_dim + j], d_prez * h[j]);
            amd::atomic_add_agent_f32(&d_h_old[j], d_prez * wij);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §A1  PEER expert-MLP backward (one wavefront per token row, single head).
//
// Mirrors sg2_bilevel::peer_head_backward (oracle 610-707) for the inline-expert
// path the §5 forward peer_route_row uses (scalar expert input = a_val+b_val,
// hidden = relu(W1·inp), out = Σ W2·hidden). Forward (recomputed here):
//   per-half top-k over num_keys → soft = softmax(top_vals*T) (T=10) →
//   candidate grid (expert = a_idx*num_keys+b_idx, routing_w = soft_a⊗soft_b) →
//   out_c = Σ_h W2[e,h]·relu(W1[e,h]·inp_c) ; head_out = Σ_c routing_w_c·out_c.
//
// Given d_head_out (scalar for this row/head), accumulate the expert weight
// grads (d_W1/d_W2 via AGENT atomics on the shared expert tables, §2.13) and the
// product-key grads d_A/d_B. The grad wrt peer_input flows through the query
// projection and is the documented host tail (d_Wq, d_peer_input = d_query·Wq);
// here we emit the expert + product-key adjoints (the MFMA/DPP-justified work).
//
// NOTE (stop-grad top-k): the discrete top-k INDEX sets are routing decisions
// with no grad (oracle 275-279). Grads flow through the gathered values: the
// softmax-of-top-vals path (→ d_top_vals → scatter into d_scores → d_q_a/d_q_b →
// d_A/d_B) and the expert-value path (→ d_W1/d_W2). Identical to the oracle.
// ════════════════════════════════════════════════════════════════════════════
template <int kTopK>
__device__ __forceinline__ void peer_route_backward_row(
    const float* __restrict__ q_a,    // [half_qd]
    const float* __restrict__ q_b,    // [half_qd]
    const short* __restrict__ keys_a, // [num_keys, half_qd] bf16 bits
    const short* __restrict__ keys_b, // [num_keys, half_qd] bf16 bits
    const short* __restrict__ expert_W1,  // [num_experts, expert_hidden] bf16 bits
    const short* __restrict__ expert_W2,  // [num_experts, expert_hidden] bf16 bits
    float d_head_out,
    float* __restrict__ d_A,          // [num_keys, half_qd]  out (atomic)
    float* __restrict__ d_B,          // [num_keys, half_qd]  out (atomic)
    float* __restrict__ d_expert_W1,  // [num_experts, expert_hidden] out (atomic)
    float* __restrict__ d_expert_W2,  // [num_experts, expert_hidden] out (atomic)
    int num_keys, int half_qd, int expert_hidden)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const float T = 10.f;

    // ── recompute per-half top-k (lane-0 owns the small scan; broadcast) ──
    float a_val[kTopK]; int a_idx[kTopK];
    float b_val[kTopK]; int b_idx[kTopK];
    #pragma unroll
    for (int t = 0; t < kTopK; ++t) { a_val[t]=-1e30f; a_idx[t]=0; b_val[t]=-1e30f; b_idx[t]=0; }
    for (int t = 0; t < kTopK; ++t) {
        float best=-1e30f; int besti=0;
        for (int kk = 0; kk < num_keys; ++kk) {
            bool taken=false;
            for (int u=0;u<t;++u) if (a_idx[u]==kk) taken=true;
            if (taken) continue;
            float s=0.f;
            for (int d=0; d<half_qd; ++d) s += q_a[d]*bf16_to_f32(keys_a[kk*half_qd+d]);
            if (s>best){best=s;besti=kk;}
        }
        a_val[t]=best; a_idx[t]=besti;
        best=-1e30f; besti=0;
        for (int kk = 0; kk < num_keys; ++kk) {
            bool taken=false;
            for (int u=0;u<t;++u) if (b_idx[u]==kk) taken=true;
            if (taken) continue;
            float s=0.f;
            for (int d=0; d<half_qd; ++d) s += q_b[d]*bf16_to_f32(keys_b[kk*half_qd+d]);
            if (s>best){best=s;besti=kk;}
        }
        b_val[t]=best; b_idx[t]=besti;
    }

    // ── soft = softmax(top_vals*T) ──
    const int kk2 = kTopK * kTopK;
    float pmax=-1e30f;
    for (int t=0;t<kTopK;++t){ pmax=dfmaxf(pmax,a_val[t]); }   // for soft_a stability
    float amax=-1e30f, bmax=-1e30f;
    for (int t=0;t<kTopK;++t){ amax=dfmaxf(amax,a_val[t]); bmax=dfmaxf(bmax,b_val[t]); }
    float asum=0.f,bsum=0.f, soft_a[kTopK], soft_b[kTopK];
    for (int t=0;t<kTopK;++t){ soft_a[t]=dexpf((a_val[t]-amax)*T); asum+=soft_a[t]; }
    for (int t=0;t<kTopK;++t){ soft_b[t]=dexpf((b_val[t]-bmax)*T); bsum+=soft_b[t]; }
    float ainv=1.f/dfmaxf(asum,1e-12f), binv=1.f/dfmaxf(bsum,1e-12f);
    for (int t=0;t<kTopK;++t){ soft_a[t]*=ainv; soft_b[t]*=binv; }
    (void)pmax;

    // ── recompute out_c and the expert-value adjoint; accumulate d_routing ──
    // out_c = Σ_h W2[e,h]·relu(W1[e,h]·inp_c), inp_c = a_val[i]+b_val[j].
    // head_out = Σ_c routing_w_c·out_c ;  d_routing_w_c = d_head_out·out_c ;
    // d_out_c = d_head_out·routing_w_c.
    float d_soft_a[kTopK], d_soft_b[kTopK];
    #pragma unroll
    for (int t=0;t<kTopK;++t){ d_soft_a[t]=0.f; d_soft_b[t]=0.f; }

    for (int c=0;c<kk2;++c) {
        int i=c/kTopK, j=c%kTopK;
        int e = a_idx[i]*num_keys + b_idx[j];
        float inp = a_val[i] + b_val[j];
        float rw  = soft_a[i]*soft_b[j];

        // out_c via lane-cooperative reduction over expert_hidden.
        float out=0.f;
        for (int h=lane; h<expert_hidden; h+=kWave) {
            float w1=bf16_to_f32(expert_W1[e*expert_hidden+h]);
            float w2=bf16_to_f32(expert_W2[e*expert_hidden+h]);
            out += w2 * (w1*inp > 0.f ? w1*inp : 0.f);
        }
        out = amd::wave_reduce_add_dpp(out);     // §2.6 every lane gets out_c

        float d_out = d_head_out * rw;           // d wrt out_c
        d_soft_a[i] += d_head_out * out * soft_b[j];   // d_routing → d_soft_a
        d_soft_b[j] += d_head_out * out * soft_a[i];

        // expert-weight adjoints (lane-strided; AGENT-atomic into shared tables):
        //   z_h = relu(W1[e,h]·inp) ; out = Σ_h W2[e,h]·z_h.
        //   d_W2[e,h] += d_out·z_h ; d_z_h = d_out·W2[e,h] ;
        //   d_pre_h = d_z_h·[W1·inp>0] ; d_W1[e,h] += d_pre_h·inp.
        for (int h=lane; h<expert_hidden; h+=kWave) {
            float w1=bf16_to_f32(expert_W1[e*expert_hidden+h]);
            float w2=bf16_to_f32(expert_W2[e*expert_hidden+h]);
            float pre = w1*inp;
            float z_h = pre > 0.f ? pre : 0.f;
            amd::atomic_add_agent_f32(&d_expert_W2[e*expert_hidden+h], d_out * z_h);
            float d_z   = d_out * w2;
            float d_pre = pre > 0.f ? d_z : 0.f;
            amd::atomic_add_agent_f32(&d_expert_W1[e*expert_hidden+h], d_pre * inp);
        }
    }

    // ── soft = softmax(vals*T) backward → d_top_vals ──
    float sa_dot=0.f, sb_dot=0.f;
    for (int t=0;t<kTopK;++t){ sa_dot+=d_soft_a[t]*soft_a[t]; sb_dot+=d_soft_b[t]*soft_b[t]; }
    float d_top_a[kTopK], d_top_b[kTopK];
    for (int t=0;t<kTopK;++t){
        d_top_a[t] = soft_a[t]*(d_soft_a[t]-sa_dot)*T;
        d_top_b[t] = soft_b[t]*(d_soft_b[t]-sb_dot)*T;
    }

    // ── top_vals = q_half · keys[idx] → d_keys (and d_q_half, host tail). ──
    // scores_a[idx] = Σ_d q_a[d]·keys_a[idx][d]; only selected idx get grad.
    // d_A[a_idx[t], d] += d_top_a[t]·q_a[d]   (lane-strided over half_qd).
    for (int t=0;t<kTopK;++t) {
        int ia=a_idx[t], ib=b_idx[t];
        for (int d=lane; d<half_qd; d+=kWave) {
            amd::atomic_add_agent_f32(&d_A[ia*half_qd+d], d_top_a[t]*q_a[d]);
            amd::atomic_add_agent_f32(&d_B[ib*half_qd+d], d_top_b[t]*q_b[d]);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §A.LAUNCH  __global__ entry kernels (host launches via hipLaunchKernelGGL from
// the ATen orchestration in the host TU; see the §A.LAUNCH note in pass A). bf16
// activations flow as raw `short` bit-patterns.
// ════════════════════════════════════════════════════════════════════════════

// LDS byte budget for one attention-context backward head.
//   dPf[N*Lc]+dSf[N*Lc]+accf[max(N*D,Lc*D)] f32  +  Pbf[max(N*Lc,Lc*N)]+
//   Tbf[max(Lc*N,D*N,Lc*D)] bf16.
__device__ __forceinline__ int sg2_attn_bwd_lds_bytes(int N, int Lc, int D) {
    int accf = (N * D > Lc * D) ? N * D : Lc * D;
    int f32b = (N * Lc + N * Lc + accf) * (int)sizeof(float);
    int pbf  = (N * Lc);                       // Pbf region (also holds Lc*N == N*Lc)
    int tmax = Lc * N; if (D * N > tmax) tmax = D * N; if (Lc * D > tmax) tmax = Lc * D;
    int bf16b = (pbf + tmax) * (int)sizeof(short);
    return f32b + bf16b;
}

// Attention-context backward: one wavefront per head. cK/cV are the already-
// gathered compressed entries [Lc,D] for this head (CSA top-k∪window or HCA all-
// compressed); P[N,Lc] are the saved forward probs for this head.
template <int kHeadDimT>
__global__ void sg2_attn_ctx_bwd_kernel(
    const short* __restrict__ q,      // [N, D]
    const short* __restrict__ cK,     // [Lc, D]
    const short* __restrict__ cV,     // [Lc, D]
    const float* __restrict__ P,      // [N, Lc]
    const short* __restrict__ d_ctx,  // [N, D]
    short* __restrict__ d_q,          // [N, D]
    short* __restrict__ d_cK,         // [Lc, D]
    short* __restrict__ d_cV,         // [Lc, D]
    int N, int Lc, float scale)
{
    extern __shared__ float lds[];
    const int D = kHeadDimT;
    int accf_n = (N * D > Lc * D) ? N * D : Lc * D;
    float* dPf = lds;                          // N*Lc f32
    float* dSf = dPf + N * Lc;                 // N*Lc f32
    float* accf = dSf + N * Lc;                // accf_n f32
    short* Pbf  = reinterpret_cast<short*>(accf + accf_n);  // N*Lc shorts
    int tmax = Lc * N; if (D * N > tmax) tmax = D * N; if (Lc * D > tmax) tmax = Lc * D;
    short* Tbf  = Pbf + N * Lc;                 // tmax shorts
    (void)tmax;
    attention_ctx_backward_core(q, cK, cV, P, d_ctx, d_q, d_cK, d_cV,
                                dPf, dSf, accf, Pbf, Tbf, N, Lc, D, scale);
}

// GRU-gate backward: one wavefront per element row. Out grads are AGENT-atomic
// accumulated (zero-init by the caller).
__global__ void sg2_gru_gate_bwd_kernel(
    const float* __restrict__ x,        // [N, in_dim]
    const float* __restrict__ h,        // [N, hidden]
    const float* __restrict__ z,        // [N, hidden]
    const float* __restrict__ r,        // [N, hidden]
    const float* __restrict__ h_tilde,  // [N, hidden]
    const short* __restrict__ Wz, const short* __restrict__ Wr,
    const short* __restrict__ Wh,
    const float* __restrict__ d_h_new,  // [N, hidden]
    float* __restrict__ d_x,            // [N, in_dim]   (or shared; atomic)
    float* __restrict__ d_h_old,        // [N, hidden]   (or shared; atomic)
    float* __restrict__ d_Wz, float* __restrict__ d_Wr, float* __restrict__ d_Wh,
    float* __restrict__ d_bz, float* __restrict__ d_br, float* __restrict__ d_bh,
    int in_dim, int hidden)
{
    const int row = static_cast<int>(blockIdx.x);
    gru_gate_backward_row(
        x + row * in_dim, h + row * hidden, z + row * hidden, r + row * hidden,
        h_tilde + row * hidden, Wz, Wr, Wh, d_h_new + row * hidden,
        d_x + row * in_dim, d_h_old + row * hidden,
        d_Wz, d_Wr, d_Wh, d_bz, d_br, d_bh, in_dim, hidden);
}

// PEER expert-MLP backward: one wavefront per token row, single head (host loops
// heads). q_a/q_b are the pre-split projected query halves for this row.
__global__ void sg2_peer_route_bwd_kernel(
    const float* __restrict__ q_a, const float* __restrict__ q_b,
    const short* __restrict__ keys_a, const short* __restrict__ keys_b,
    const short* __restrict__ expert_W1, const short* __restrict__ expert_W2,
    const float* __restrict__ d_head_out,  // [N]
    float* __restrict__ d_A, float* __restrict__ d_B,
    float* __restrict__ d_expert_W1, float* __restrict__ d_expert_W2,
    int num_keys, int half_qd, int expert_hidden)
{
    const int row = static_cast<int>(blockIdx.x);
    peer_route_backward_row<4>(
        q_a + row * half_qd, q_b + row * half_qd,
        keys_a, keys_b, expert_W1, expert_W2,
        d_head_out[row], d_A, d_B, d_expert_W1, d_expert_W2,
        num_keys, half_qd, expert_hidden);
}

// Force-instantiate the device kernels at the grokking shapes (head_dim=4). The
// host TU dispatches to these on the migrated .hip build.
template __global__ void sg2_attn_ctx_bwd_kernel<4>(
    const short*, const short*, const short*, const float*, const short*,
    short*, short*, short*, int, int, float);

}  // namespace native_adjoint
}}}}  // namespace sg::gfx942::models::supergrok2
#endif  // (B) device pass

#endif  // CSRC_BACKENDS_HIP_GFX942_SUPERGROK2_BILEVEL_ADJOINT_GFX942_HIP_HPP_
