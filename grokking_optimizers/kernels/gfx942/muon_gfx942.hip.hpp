#ifndef GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
// ============================================================================
// muon_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'muon'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen + rocBLAS host orchestration (the public
//       sg::gfx942::launch_muon_* entry points the bindings call — UNCHANGED,
//       byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN reduction kernel (§5 below) built on the
//       shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — the momentum-buffer
//       Frobenius norm ‖M‖_F = sqrt(Σ_ij M_ij²) that normalizes the
//       Newton-Schulz iterate, expressed as the canonical
//       wave→block→AGENT-atomic sum-of-squares reduction (DPP wavefront reduce,
//       LDS block tree, then a single AGENT-scope global atomic), replacing the
//       ATen `.norm()`.
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A). It pulls in
//     torch/extension.h + primitives.hpp and exposes the launchers; the thin
//     host launch_muon.hip.cpp TU resolves exactly as before.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device reduction kernel.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics +
// the wave→block→AGENT bit-parity check are deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_muon.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Muon.
// Algorithm: csrc/algorithms/muon.h
//
// COMPUTE PATTERN
// Mixed: GEMM-heavy.
//   1. momentum buffer:    buf = momentum * buf + g           — elementwise
//   2. Frobenius norm:     inv_norm = 1 / ||buf||_F           — global reduction
//   3. normalize:          X = buf * inv_norm                  — elementwise
//   4. Newton-Schulz × 5:  for step in {0..4}:
//                             A   = X @ X.T          — GEMM (rows × cols)
//                             AX  = A @ X            — GEMM
//                             AAX = A @ AX           — GEMM
//                             X   = 3.4445*X - 4.7750*AX + 2.0315*AAX
//   5. update:             p -= lr * X * scale + p * decay     — elementwise
//
// MFMA APPLICABILITY: significant.
// The 3 GEMMs per Newton-Schulz step are exactly what MFMA accelerates.
// Typical Muon shapes for grokking models (e.g. 96×96 weight matrices):
// MFMA `v_mfma_f32_16x16x16_bf16` runs 6×6 = 36 MFMA tiles per GEMM.
// At MI300X's 1100 TFLOPS BF16, the 3 GEMMs × 5 steps complete in ~5 µs.
//
// WHY ATEN HERE (for the GEMMs)
// `torch::mm` on a HIP tensor dispatches to rocBLAS's GEMM, which
// internally uses `v_mfma_f32_16x16x16_bf16` for the BF16 path (or
// `v_mfma_f32_16x16x4_f32` for FP32). The MFMA acceleration is already
// being exercised through rocBLAS. The Frobenius-norm REDUCTION, however,
// is the high-value AMDGCN piece (§5) — a global sum-of-squares.

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST orchestration — ATen public entry points. Compiled by the HOST pass
// only. The free-standing AMDGCN device gate (__AMDGCN__) does NOT see this
// block (torch/extension.h pulls in <cuda.h>/ATen, which the free-standing
// device target cannot resolve); the §5 device kernel below is the device-pass
// content. On a real hipcc build the host pass compiles this and launches the
// §5 kernel via hipLaunchKernelGGL (see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
// SG_GFX942_DEVICE_TU (Stage 7): set by the thin `.hip` device TU so this host
// launcher is NOT re-emitted in that TU's host pass (it stays owned by the
// `.hip.cpp` host TU) — avoids duplicate launch_muon_* symbols at link.
#if !defined(__AMDGCN__) && !defined(SG_GFX942_DEVICE_TU)
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "csrc/backends/hip/gfx942/primitives.hpp"

#if defined(__HIPCC__)
// LIVE (hipcc) host launch glue: the §5 device kernels (section B) and the HIP
// runtime launch/stream API. On the bare AMDGCN gate (no hipcc) this whole host
// block is skipped, so amdgcn_check.sh never parses these HIP-runtime includes.
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#endif

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

#if defined(__HIPCC__)
// Forward declarations of the §5 / §5.2 device kernels (defined in section B
// below) so the host launchers can hipLaunchKernelGGL them. Signatures match
// the extern "C" __global__ definitions exactly.
namespace native {
extern "C" __global__ void muon_gfx942_frobenius_reduce(
    const float* __restrict__ m, float* __restrict__ acc, int64_t n);
extern "C" __global__ void muon_gfx942_newton_schulz(
    float* __restrict__ X_inout, int M, int N, int steps,
    float a, float b, float c);
}  // namespace native
#endif

static inline torch::Tensor newton_schulz_iterate(
    torch::Tensor X, int ns_steps, float a, float b, float c
) {
    for (int it = 0; it < ns_steps; it++) {
        auto AX  = torch::mm(X.transpose(-2, -1), X);
        auto AAX = torch::mm(AX, AX);
        X = a * X + b * torch::mm(X, AX) + c * torch::mm(X, AAX);
    }
    return X;
}

void launch_muon_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& bufs,
    std::vector<torch::Tensor>& grads,
    float lr, float momentum, float wd, int ns_steps,
    float ns_a, float ns_b, float ns_c
) {
#if defined(__HIPCC__)
    // LIVE (hipcc): dispatch the §5 device kernels per stage; §5 / §5.2 are the
    // live AMD-native path on real hardware. Stage order mirrors the COMPUTE
    // PATTERN above:
    //   (1) momentum EMA + (5) p-update stay elementwise on the ATen host path;
    //   (2) Frobenius ‖buf‖_F  → §5  muon_gfx942_frobenius_reduce  (DPP reduce);
    //   (3) normalize X = buf / ‖buf‖_F                            (host scalar);
    //   (4) Newton-Schulz × ns_steps → §5.2 muon_gfx942_newton_schulz (bf16 MFMA).
    // 1D params have no 2-D orthogonalization → AdamW-like device fallthrough.
    hipStream_t stream = at::hip::getCurrentHIPStream();
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& buf = bufs[i];
        auto& g = grads[i];

        // (1) momentum EMA (elementwise — no MFMA/DPP value).
        buf.mul_(momentum).add_(g.to(buf.scalar_type()), 1.0f - momentum);

        if (p.dim() >= 2) {
            auto buf_f = buf.to(torch::kFloat32).contiguous();
            const int64_t n = static_cast<int64_t>(buf_f.numel());  // Stage 1: 64-bit

            // (2) Frobenius ‖buf‖_F = sqrt(Σ buf²): §5 DPP wave→block→AGENT reduce.
            auto acc = torch::zeros({1}, buf_f.options());
            dim3 fgrid(static_cast<unsigned>(
                std::min<int64_t>(1024, (n + 255) / 256))), fblock(256);
            hipLaunchKernelGGL(native::muon_gfx942_frobenius_reduce, fgrid, fblock,
                               0, stream, buf_f.data_ptr<float>(),
                               acc.data_ptr<float>(), n);
            SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK
            float frob = sqrtf(acc.item<float>()) + 1e-8f;

            // (3) normalize the iterate in place: X = buf / ‖buf‖_F.
            int M = static_cast<int>(p.size(-2));
            int N = static_cast<int>(p.size(-1));
            auto X = (buf_f / frob).contiguous();

            // (4) Newton-Schulz × ns_steps: §5.2 one wavefront per 2-D param,
            //     16×16×16 bf16 MFMA orthogonalization (device matrix-core path).
            int sq = N * N, rect = M * N, mx = sq > rect ? sq : rect;
            int lds = (rect + sq + sq + rect) * (int)sizeof(float)
                      + (2 * mx) * (int)sizeof(short);   // host mirror of muon_ns_lds_bytes
            hipLaunchKernelGGL(native::muon_gfx942_newton_schulz, dim3(1), dim3(64),
                               lds, stream, X.data_ptr<float>(), M, N, ns_steps,
                               ns_a, ns_b, ns_c);
            SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK

            // (5) p -= lr·X·scale + p·decay (elementwise — host path).
            float neg_lr_scale = -lr * 0.2f * sqrtf((float)std::max<int64_t>(p.size(-1), p.size(-2)));
            p.mul_(1.0f - lr * wd).add_(X.to(p.scalar_type()), neg_lr_scale);
        } else {
            // 1D fall back: Adam-like (no 2-D orthogonalization).
            p.add_(buf.to(p.scalar_type()), -lr);
        }
    }
#else
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& buf = bufs[i];
        auto& g = grads[i];

        buf.mul_(momentum).add_(g.to(buf.scalar_type()), 1.0f - momentum);

        if (p.dim() >= 2) {
            auto frob = buf.norm() + 1e-8f;
            auto X = buf / frob;
            X = newton_schulz_iterate(X, ns_steps, ns_a, ns_b, ns_c);
            float neg_lr_scale = -lr * 0.2f * sqrtf((float)std::max<int64_t>(p.size(-1), p.size(-2)));
            p.mul_(1.0f - lr * wd).add_(X.to(p.scalar_type()), neg_lr_scale);
        } else {
            // 1D fall back: Adam-like
            p.add_(buf.to(p.scalar_type()), -lr);
        }
    }
#endif
}


void launch_muon_ns_combine_update_fused(
    torch::Tensor param, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX, float a, float b, float c, float neg_lr_scale, float decay_factor
) {
    // Y = a*X + b*AX + c*AAX
    auto Y = a * X + b * AX + c * AAX;
    // param = param + neg_lr_scale*Y - decay_factor*param
    //       = (1 - decay_factor)*param + neg_lr_scale*Y
    param.mul_(1.0f - decay_factor).add_(Y.to(param.scalar_type()), neg_lr_scale);
}

void launch_muon_momentum_normalize(
    torch::Tensor buf, torch::Tensor X, torch::Tensor grad, float momentum, float inv_norm
) {
    // buf = momentum*buf + inv_norm*grad.float()
    buf.mul_(momentum).add_(grad.to(torch::kFloat32), inv_norm);
    // X = buf (copy)
    X.copy_(buf);
}

void launch_muon_ns_combine(
    torch::Tensor X_out, torch::Tensor X, torch::Tensor AX, torch::Tensor AAX, float a, float b, float c
) {
    // X_out = a*X + b*AX + c*AAX
    X_out.copy_(a * X + b * AX + c * AAX);
}

void launch_muon_update(
    torch::Tensor param, torch::Tensor orth, float neg_lr_scale, float decay_factor
) {
    // param = param + neg_lr_scale*orth - decay_factor*param
    //       = (1 - decay_factor)*param + neg_lr_scale*orth
    param.mul_(1.0f - decay_factor).add_(orth.to(param.scalar_type()), neg_lr_scale);
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring — NOW DISPATCHED) ────────────────────────────
// launch_muon_step() DISPATCHES the §5 device kernels on hipcc (`#if __HIPCC__`);
// ATen `.norm()` + torch::mm Newton-Schulz is the `#else` CPU-host fallback.
// Live (hipcc) stage order for each 2-D param (see launch_muon_step):
//   dim3 grid(min(1024, (n+255)/256)), block(256);   // 4 wavefronts/block
//   hipLaunchKernelGGL(native::muon_gfx942_frobenius_reduce, grid, block,
//                      0, stream, buf_ptr, d_acc, n);   // §5 DPP reduce
//   // host then: frob = sqrtf(*d_acc) + 1e-8f;  X = buf / frob
//   int lds = (M*N + N*N + N*N + M*N)*sizeof(float)
//             + 2*max(M*N,N*N)*sizeof(short);          // host mirror of muon_ns_lds_bytes
//   hipLaunchKernelGGL(native::muon_gfx942_newton_schulz, dim3(1), dim3(64),
//                      lds, stream, X_ptr, M, N, ns_steps, ns_a, ns_b, ns_c);
// The §5.2 path is the device 16×16×16 bf16 MFMA Newton-Schulz; rocBLAS/torch::mm
// is ONLY the `#else` CPU fallback. The §5 / §5.2 kernels are COMPILER-VERIFIED
// for gfx942 via scripts/amdgcn_check.sh.
// 🟡 host-launch glue UNEXERCISED here (no hipcc / no MI300X in this CI); the
//    live launch + hipcc link is hardware-gated — see HARDWARE_VALIDATION.md.
#endif  // !defined(__AMDGCN__)  — end host orchestration (A)

// ════════════════════════════════════════════════════════════════════════════
// (B) DEVICE pass — real hand-written AMDGCN reduction (§5).
// Compiled by the AMDGCN device pass only: the Stage-5 gate (__AMDGCN__, no
// hipcc) AND the hipcc device pass (__HIPCC__). The host `.hip.cpp` TU never
// sees it — that pass keeps the ATen orchestration above (which LAUNCHES this
// kernel via hipLaunchKernelGGL, see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)
#include "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp"
// ── gate-only launch-builtin shim (verbatim from the mamba3/attention exemplar)
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
// §5  AMD-NATIVE device kernel (Stage 5 hand-written AMDGCN).
//
// The momentum-buffer Frobenius norm ‖M‖_F = sqrt(Σ_ij M_ij²). The high-value
// AMDGCN piece is the global sum-of-squares reduction; the host does the final
// sqrt (and the 1/‖M‖_F normalize scale for Newton-Schulz). This is the
// canonical 2-level (wave→block→grid) tree from the task spec:
//   1. each thread grid-strides M², accumulating a per-thread partial;
//   2. amd::wave_reduce_add_dpp gives every lane the per-wavefront sum (§2.6);
//   3. lane 0 of each wavefront writes its sum to an LDS slot; workgroup_barrier;
//   4. the first wavefront reduces the LDS slots via a second DPP reduce;
//   5. one thread does amd::atomic_add_agent_f32 to the global accumulator
//      (§2.13: AMD has no DSMEM, so cross-workgroup uses an AGENT-scope atomic
//      visible across all 8 XCDs of MI300X).
// Two AMDGCN device pieces are provided:
//   §5.1  Frobenius-norm reduction (DPP wave→block→grid sum-of-squares), and
//   §5.2  Newton-Schulz orthogonalization via REAL 16×16×16 bf16 MFMA (the
//         3-GEMMs/step inner loop — the matmul-bearing work the task calls for).
// APPLY: the momentum EMA + the final p -= lr·X·scale + decay update stay on the
// ATen host path (launch_muon_*); they are pure elementwise (no MFMA/DPP value).
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;
static constexpr int kWave = 64;   // == amd::kWave (CDNA3 wavefront width)

// Block (workgroup) sum-of-squares of `m[0..n)`, atomically added to *acc.
// blockDim.x must be a multiple of kWave and <= kWave*kWave (<= 4096); LDS holds
// one float per wavefront (<= 64 floats). The momentum buffer is laid out
// contiguously [rows*cols], so Σ_ij M_ij² is a flat sum over n = numel.
__device__ __forceinline__ void muon_frobenius_block(
    const float* __restrict__ m, float* __restrict__ acc, int64_t n)
{
    // LDS/lane bookkeeping is LOCAL to the block (<= 4096 threads) → int. The
    // GLOBAL element index/stride is int64 so the Frobenius reduction over a
    // >2^31-element momentum buffer cannot wrap (Stage 1).
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid % kWave;
    const int waveId = tid / kWave;
    const int wpb    = static_cast<int>(blockDim.x) / kWave;
    const int64_t gtid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local = 0.f;
    for (int64_t i = gtid; i < n; i += stride) {
        float v = amd::streaming_load(&m[i]);
        local += v * v;
    }
    // wavefront DPP reduce: every lane holds the wavefront sum.
    local = amd::wave_reduce_add_dpp(local);

    __shared__ float lds[kWave];
    if (lane == 0) lds[waveId] = local;
    amd::workgroup_barrier_release();

    // first wavefront reduces the per-wavefront slots, then one AGENT atomic.
    if (waveId == 0) {
        float slot = (lane < wpb) ? lds[lane] : 0.f;
        slot = amd::wave_reduce_add_dpp(slot);
        if (lane == 0) amd::atomic_add_agent_f32(acc, slot);
    }
}

// Bandwidth-bound reduction → high occupancy (256-thread block = 4 wavefronts;
// 8 waves/EU) to hide global-memory latency. (WS5 occupancy wire.)
extern "C" SG_KERNEL_BOUNDS(256, 8) void muon_gfx942_frobenius_reduce(
    const float* __restrict__ m, float* __restrict__ acc, int64_t n)
{
    muon_frobenius_block(m, acc, n);
}

// ════════════════════════════════════════════════════════════════════════════
// §5.2  Newton-Schulz iteration via REAL 16×16×16 bf16 MFMA (the §2.4 matrix-
// core path). The orthogonalization inner loop is three GEMMs per step over the
// normalized weight X[M,N] (M=rows, N=cols):
//   AX  = Xᵀ·X            [N,N]   (A=Xᵀ[N,M], W=Xᵀ[N,M] → contract M)
//   AAX = AX·AX           [N,N]   (A=AX[N,N],  W=AXᵀ[N,N] → contract N)
//   X'  = a·X + b·X·AX + c·X·AAX  (X·AX,X·AAX: A=X[M,N], W=AXᵀ/AAXᵀ[N,N])
// where the §2.4 mfma helper computes C[M,N']=A[M,K]·W[N',K]ᵀ (W in [out,K]
// row-major = the transpose the contraction wants). bf16 operands flow as raw
// `short` bit-patterns to amd::mfma_bf16_16x16x16 (short[4]); f32 accumulate.
// One wavefront strides the 16×16 output-tile lattice; MFMA vs VMEM are pinned
// via sched_group_barrier (§2.11) so the matrix unit is not starved.
//
// This is the genuine MFMA-justified Muon work (vs the rocBLAS-via-ATen host
// fallback). 🟡 device-compile-verified only; MI300X parity vs newton_schulz_
// iterate() deferred (HARDWARE_VALIDATION.md, Stage 5).
// ════════════════════════════════════════════════════════════════════════════

// bf16 <-> f32 bit codec (the gate has no hip_bfloat16; matches the SG2 codec).
__device__ __forceinline__ float ns_bf16_to_f32(short h) {
    unsigned u = static_cast<unsigned>(static_cast<unsigned short>(h)) << 16;
    return __builtin_bit_cast(float, u);
}
__device__ __forceinline__ short ns_f32_to_bf16(float f) {
    unsigned u = __builtin_bit_cast(unsigned, f);
    unsigned lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<short>(static_cast<unsigned short>(u >> 16));
}

// One 16×16 output tile of C[M,N] = A[M,K] · W[N,K]ᵀ (K-contraction), MFMA path.
__device__ __forceinline__ void ns_mfma_tile_16x16(
    const short* __restrict__ A,   // [M, K] row-major bf16 bits
    const short* __restrict__ W,   // [N, K] row-major bf16 bits (= Bᵀ)
    float*       __restrict__ C,   // [M, N] row-major f32
    int M, int N, int K, int tile_row, int tile_col)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    const int half = lane / 16;        // which 16-K group this lane feeds (0..3)
    const int idx  = lane % 16;        // row (A) / col (W) within the tile
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int aRow = tile_row + idx;
    const int bCol = tile_col + idx;
    for (int k0 = 0; k0 < K; k0 += 16) {
        short af[4], bf[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = k0 + half * 4 + j;
            af[j] = (aRow < M && k < K) ? amd::streaming_load(&A[aRow * K + k]) : (short)0;
            bf[j] = (bCol < N && k < K) ? amd::streaming_load(&W[bCol * K + k]) : (short)0;
        }
        amd::mfma_bf16_16x16x16(acc, af, bf);
        amd::sched_group_barrier<0x008, 1>();   // 1 MFMA …
        amd::sched_group_barrier<0x100, 2>();   // … then 2 VMEM reads (§2.11)
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int outRow = tile_row + 4 * half + r;
        int outCol = tile_col + idx;
        if (outRow < M && outCol < N) C[outRow * N + outCol] = acc[r];
    }
}

// Whole-matrix MFMA GEMM: one wavefront strides the 16×16 output-tile lattice.
__device__ __forceinline__ void ns_mfma_matmul_bf16(
    const short* __restrict__ A, const short* __restrict__ W,
    float* __restrict__ C, int M, int N, int K)
{
    const int tilesM = (M + 15) / 16;
    const int tilesN = (N + 15) / 16;
    const int nTiles = tilesM * tilesN;
    for (int t = 0; t < nTiles; ++t)
        ns_mfma_tile_16x16(A, W, C, M, N, K, (t / tilesN) * 16, (t % tilesN) * 16);
}

// Pack f32 src[R,Cc] → bf16-bits dst[R,Cc] (lane-strided).
__device__ __forceinline__ void ns_pack_bf16(
    const float* __restrict__ src, short* __restrict__ dst, int n)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    for (int i = lane; i < n; i += kWave) dst[i] = ns_f32_to_bf16(src[i]);
}
// Transpose-pack f32 src[R,Cc] → bf16-bits dst[Cc,R] (lane-strided).
__device__ __forceinline__ void ns_transpose_pack_bf16(
    const float* __restrict__ src, short* __restrict__ dst, int R, int Cc)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    for (int i = lane; i < R * Cc; i += kWave) {
        int r = i / Cc, c = i % Cc;
        dst[c * R + r] = ns_f32_to_bf16(src[i]);
    }
}

// One Newton-Schulz step over X[M,N] (single wavefront, X already 1/‖·‖-scaled).
// Matches newton_schulz_iterate(): AX=XᵀX[N,N]; AAX=AX·AX[N,N];
// X' = a·X + b·(X·AX) + c·(X·AAX). AX and AAX stay resident through both final
// rectangular GEMMs, so no operand is recomputed.
// LDS scratch (caller-sized via muon_ns_lds_bytes): Xf[M*N], AXf[N*N], AAXf[N*N],
// XAXf[M*N] f32 + a bf16 pack region pkA/pkW each [max(M*N,N*N)] shorts.
__device__ __forceinline__ void muon_ns_step(
    float* __restrict__ Xf,     // LDS [M,N] in/out (f32)
    float* __restrict__ AXf,    // LDS [N,N]   AX = XᵀX
    float* __restrict__ AAXf,   // LDS [N,N]   AAX = AX·AX
    float* __restrict__ XAXf,   // LDS [M,N]   X·AX (then X·AAX)
    short* __restrict__ pkA,    // LDS bf16 scratch A operand
    short* __restrict__ pkW,    // LDS bf16 scratch W operand
    int M, int N, float a, float b, float c)
{
    const int lane = static_cast<int>(threadIdx.x) % kWave;

    // (1) AX = Xᵀ·X  [N,N]: A=Xᵀ[N,M], W=Xᵀ[N,M] (helper computes A·Wᵀ; both
    //     operands = Xᵀ[N,M] contract over M, giving Xᵀ·(Xᵀ)ᵀ = Xᵀ·X).
    ns_transpose_pack_bf16(Xf, pkA, M, N);          // pkA = Xᵀ[N,M] bf16
    amd::workgroup_barrier_release();
    ns_mfma_matmul_bf16(pkA, pkA, AXf, N, N, M);    // AXf = XᵀX [N,N]
    amd::wait_vmcnt0();
    amd::workgroup_barrier_release();

    // (2) AAX = AX·AX [N,N]: A=AX[N,N], W=AXᵀ[N,N] (AX symmetric, but transpose
    //     explicitly to keep the A·Wᵀ contract general).
    ns_pack_bf16(AXf, pkA, N * N);                  // pkA = AX[N,N] bf16
    ns_transpose_pack_bf16(AXf, pkW, N, N);         // pkW = AXᵀ[N,N] bf16
    amd::workgroup_barrier_release();
    ns_mfma_matmul_bf16(pkA, pkW, AAXf, N, N, N);   // AAXf = AX·AX [N,N]
    amd::wait_vmcnt0();
    amd::workgroup_barrier_release();

    // (3) X·AX [M,N]: A=X[M,N], W=AXᵀ[N,N]  (→ XAXf).
    ns_pack_bf16(Xf, pkA, M * N);                   // pkA = X[M,N] bf16
    ns_transpose_pack_bf16(AXf, pkW, N, N);         // pkW = AXᵀ[N,N] bf16
    amd::workgroup_barrier_release();
    ns_mfma_matmul_bf16(pkA, pkW, XAXf, M, N, N);   // XAXf = X·AX [M,N]
    amd::wait_vmcnt0();
    amd::workgroup_barrier_release();

    // (4) X·AAX [M,N]: A=X[M,N] (pkA still valid), W=AAXᵀ[N,N]  (→ AAXf reused).
    ns_transpose_pack_bf16(AAXf, pkW, N, N);        // pkW = AAXᵀ[N,N] bf16
    amd::workgroup_barrier_release();
    ns_mfma_matmul_bf16(pkA, pkW, AAXf, M, N, N);   // AAXf = X·AAX [M,N]
    amd::wait_vmcnt0();
    amd::workgroup_barrier_release();

    // (5) X' = a·X + b·(X·AX) + c·(X·AAX).
    for (int i = lane; i < M * N; i += kWave)
        Xf[i] = a * Xf[i] + b * XAXf[i] + c * AAXf[i];
    amd::workgroup_barrier_release();
}

// LDS byte budget for one Newton-Schulz wavefront over X[M,N].
__device__ __forceinline__ int muon_ns_lds_bytes(int M, int N) {
    int sq = N * N, rect = M * N;
    int mx = sq > rect ? sq : rect;
    int f32b = (rect + sq + sq + rect) * (int)sizeof(float);  // Xf,AXf,AAXf,XAXf
    int bf16b = (2 * mx) * (int)sizeof(short);
    return f32b + bf16b;
}

// Newton-Schulz orthogonalization kernel: one wavefront per 2-D parameter.
// X_inout[M,N] is the 1/‖·‖_F-normalized momentum (host does the normalize); the
// kernel applies `steps` Newton-Schulz iterations in place via the MFMA GEMMs.
// MFMA matrix-core kernel: launched single-wavefront (dim3(64)); the inner GEMM
// loop is VGPR-heavy (f32 accumulators), so request a modest occupancy that does
// NOT over-constrain the register allocator (flat WG = 64 = 1 wavefront, 4
// waves/EU). (WS5 occupancy wire — replaces the bare __global__.)
extern "C" SG_KERNEL_BOUNDS(64, 4) void muon_gfx942_newton_schulz(
    float* __restrict__ X_inout, int M, int N, int steps,
    float a, float b, float c)
{
    extern __shared__ float lds[];
    const int rect = M * N, sq = N * N;
    int mx = sq > rect ? sq : rect;
    float* Xf   = lds;
    float* AXf  = Xf   + rect;
    float* AAXf = AXf  + sq;
    float* XAXf = AAXf + sq;
    short* pkA  = reinterpret_cast<short*>(XAXf + rect);
    short* pkW  = pkA + mx;
    const int lane = static_cast<int>(threadIdx.x) % kWave;
    for (int i = lane; i < rect; i += kWave) Xf[i] = amd::streaming_load(&X_inout[i]);
    amd::workgroup_barrier_release();
    for (int s = 0; s < steps; ++s)
        muon_ns_step(Xf, AXf, AAXf, XAXf, pkA, pkW, M, N, a, b, c);
    for (int i = lane; i < rect; i += kWave) amd::streaming_store(&X_inout[i], Xf[i]);
}

// ════════════════════════════════════════════════════════════════════════════
// §5.3  Muon APPLY — per-element param update (elementwise, vectorized only; NO
// DPP — the cross-lane ‖buf‖_F reduction is §5.1 and the orthogonalization is the
// §5.2 MFMA Newton-Schulz; both stay as-is).
//
// Once the orthogonalized iterate `orth` (= X, post-Newton-Schulz) and the two
// host scalars are known, the apply is the decoupled-weight-decay step from
// launch_muon_update() / launch_muon_step() stage (5):
//   param = param·decay + neg_lr_scale·orth
// where decay = 1 - lr·wd and neg_lr_scale = -lr·0.2·sqrt(max(rows,cols)) are
// uniform host scalars. Math is verbatim from the ATen body
// (param.mul_(1 - decay_factor).add_(orth, neg_lr_scale), here folded as
// param·decay + neg_lr_scale·orth with decay = 1 - decay_factor).
//
// VECTORIZATION (WS5/B — scalar→f32x4): the fp32 path is bandwidth-bound, so the
// bulk loop is widened to 128-bit (dwordx4) memory access. orth is vector-loaded
// via amd::streaming_load<f32x4> (read-once); param is vector-loaded and the
// write-once result goes through amd::streaming_store<f32x4>. The 4 lanes each
// evaluate the IDENTICAL scalar expression (uniform decay/neg_lr_scale on all
// lanes), so the result is BIT-IDENTICAL to the scalar path — only the access
// width changes. A scalar TAIL handles the final N%4 (and the whole array on the
// fp32-only fallback).
// ════════════════════════════════════════════════════════════════════════════

// Per-element Muon apply — canonical scalar body, shared by the scalar tail and
// (replicated lane-by-lane) the f32x4 fast-path so both are bit-identical.
template <typename ParamT, typename OrthT>
__device__ __forceinline__ void muon_apply_elem(
    ParamT* __restrict__ param, const OrthT* __restrict__ orth, int64_t i,
    float decay, float neg_lr_scale)
{
    const float p = static_cast<float>(param[i]);
    const float x = static_cast<float>(amd::streaming_load(&orth[i]));
    amd::streaming_store(&param[i],
                         static_cast<ParamT>(p * decay + neg_lr_scale * x));
}

template <typename ParamT, typename OrthT>
SG_KERNEL_BOUNDS(256, 8) void muon_gfx942_apply_kernel(
    ParamT* __restrict__ param, const OrthT* __restrict__ orth,
    float decay, float neg_lr_scale, int64_t N)
{
    // 64-bit grid-stride indexing (Stage 1): cast BEFORE the multiply.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;

    constexpr bool kVecOk =
        sizeof(ParamT) == sizeof(float) && sizeof(OrthT) == sizeof(float);
    const int64_t n4 = kVecOk ? (N >> 2) : 0;   // number of full float4 groups

    using f32x4 = amd::f32x4;
    auto* p4 = reinterpret_cast<f32x4*>(param);
    const auto* x4 = reinterpret_cast<const f32x4*>(orth);

    for (int64_t q = tid; q < n4; q += stride) {
        const f32x4 p = p4[q];
        const f32x4 x = amd::streaming_load(&x4[q]);   // 128-bit dwordx4 load (orth)
        f32x4 po;
        for (int l = 0; l < 4; ++l) {
            po[l] = p[l] * decay + neg_lr_scale * x[l];
        }
        amd::streaming_store(&p4[q], po);              // 128-bit dwordx4 store
    }

    // SCALAR TAIL: the final N%4 elements (and the whole array on fp32 fallback).
    for (int64_t i = (n4 << 2) + tid; i < N; i += stride) {
        muon_apply_elem(param, orth, i, decay, neg_lr_scale);
    }
}

// Force-instantiate the grokking dtype combo (fp32 param + fp32 orth iterate).
template __global__ void muon_gfx942_apply_kernel<float, float>(
    float*, const float*, float, float, int64_t);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_MUON_GFX942_HIP_HPP_
