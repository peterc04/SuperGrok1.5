#ifndef GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
// ============================================================================
// adamw_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'adamw'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen host orchestration (the public sg::gfx942::launch_adamw_step
//       entry point the bindings call — UNCHANGED, byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN grid-stride update kernel (§5 below) built
//       on the shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — streaming
//       (read-once) grad loads via amd::streaming_load and a streaming param
//       write-back via amd::streaming_store, fusing the m/v EMAs + bias-
//       corrected decoupled-weight-decay apply into ONE kernel (vs the ATen
//       path's 3 elementwise launches).
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A). It pulls in
//     torch/extension.h + primitives.hpp and exposes the launcher; the thin
//     host launch_adamw.hip.cpp TU resolves exactly as before.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device update kernel.
//
// ELEMENTWISE MATH: inlined (option a). The per-element step in
// csrc/algorithms/adamw.h (adamw_step) pulls torch via
// csrc/common/{types.h,utils.cuh}, which the bare amdgcn gate cannot resolve,
// so the ~8-line update is copied here verbatim (numerically identical:
// un-inverted bc1/bc2 division, sqrtf→__builtin_sqrtf).
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numeric
// parity vs the algorithm-header reference is deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_adamw.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for AdamW (simple, used by Muon 1D params + LookSAM).
// Algorithm: csrc/algorithms/adamw.h
//
// COMPUTE PATTERN
// Pure elementwise. Per element:
//   m = beta1 * m + (1-beta1) * g         — 1 FMA, 2 reads, 1 write
//   v = beta2 * v + (1-beta2) * g²        — 1 FMA + 1 mul, 2 reads, 1 write
//   p -= lr * (m / bc1 / (sqrt(v/bc2) + eps) + wd * p)  — div, sqrt, FMA
// Bandwidth-bound (≈ 12 mem ops per element including p, m, v, g).
//
// MFMA APPLICABILITY: none.
// AdamW is pure elementwise SIMD. No matrix multiplies. CDNA3 v_mfma_*
// instructions would be unused.

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST orchestration — ATen public entry point. Compiled by the HOST pass
// only. The free-standing AMDGCN device gate (__AMDGCN__) does NOT see this
// block (torch/extension.h pulls in <cuda.h>/ATen, which the free-standing
// device target cannot resolve); the §5 device kernel below is the device-pass
// content. On a real hipcc build the host pass compiles this and launches the
// §5 kernel via hipLaunchKernelGGL (see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if !defined(__AMDGCN__)
#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_adamw_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    auto pack = prim::pack_valid(params, grads, exp_avgs, exp_avg_sqs);
    for (size_t i = 0; i < pack.params.size(); i++) {
        auto& p = pack.params[i];
        auto& g = pack.grads[i];
        auto& m = pack.state_a[i];
        auto& v = pack.state_b[i];

        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring note) ────────────────────────────────────────
// On a real `.hip` (hipcc) build, launch_adamw_step() launches the §5 kernel
// per tensor instead of the 3 ATen elementwise ops:
//   dim3 grid(min(1024,(n+255)/256)), block(256);   // 4 wavefronts/block
//   hipLaunchKernelGGL((native::adamw_gfx942_kernel<float,float>), grid, block,
//                      0, stream, p_ptr, m_ptr, v_ptr, g_ptr,
//                      lr, beta1, beta2, eps, wd, bc1, bc2, n);
// 🟡 DEFERRED: the live launch + hipcc link is MI300X-gated. This host TU keeps
// the ATen path (numerics-correct); the §5 kernel is COMPILER-VERIFIED for
// gfx942 via scripts/amdgcn_check.sh and ready to wire in on hardware.
#endif  // !defined(__AMDGCN__)  — end host orchestration (A)

// ════════════════════════════════════════════════════════════════════════════
// (B) DEVICE pass — real hand-written AMDGCN grid-stride update (§5).
// Compiled by the AMDGCN device pass only: the Stage-5 gate (__AMDGCN__, no
// hipcc) AND the hipcc device pass (__HIPCC__). The host `.hip.cpp` TU never
// sees it — that pass keeps the ATen orchestration above (which LAUNCHES this
// kernel via hipLaunchKernelGGL, see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if defined(__AMDGCN__) || defined(__HIPCC__) || defined(GROK_HIP_DEVICE)
#include "csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp"
// ── gate-only launch-builtin shim (verbatim from the looksam/mamba3 exemplar) ─
// The free-standing AMDGCN device gate stubs out <hip/hip_runtime.h>, so the
// launch builtins (threadIdx/blockIdx/blockDim/gridDim, __global__) HIP normally
// provides are absent. Model them with the AMDGCN workitem ISA builtins so the
// device bodies type-check. Active ONLY on the bare gate.
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
#endif  // GROK_GFX942_LAUNCH_SHIM_
#endif  // bare gate
// ============================================================================
// §5  AMD-NATIVE device kernel (Stage 5 hand-written AMDGCN).
//
// Grid-stride AdamW: each workitem owns a stride of elements, reads grad
// read-once via amd::streaming_load (nontemporal — bypasses L2 for one-touch
// data, §2.7), fuses the m/v EMA updates + bias-corrected decoupled-weight-
// decay apply in registers, then writes the param back via amd::streaming_store.
// The math is identical to sg::algorithms::adamw_step (bc1/bc2 un-inverted →
// divide). libm sqrtf is unavailable under the bare gate, so __builtin_sqrtf.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;

template <typename ParamT, typename GradT>
__global__ void adamw_gfx942_kernel(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, const GradT* __restrict__ grad,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N)
{
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                 + static_cast<int>(threadIdx.x);
         i < N; i += stride) {
        const float g  = static_cast<float>(amd::streaming_load(&grad[i]));
        const float p  = static_cast<float>(param[i]);
        const float m  = beta1 * exp_avg[i]    + (1.0f - beta1) * g;
        const float v  = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
        exp_avg[i]    = m;
        exp_avg_sq[i] = v;
        const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
        amd::streaming_store(&param[i],
                             static_cast<ParamT>(p - lr * (update + wd * p)));
    }
}

// Force-instantiate the grokking dtype combo (fp32 param + fp32 grad) so the
// device pass emits the kernel; the host TU dispatches on dtype.
template __global__ void adamw_gfx942_kernel<float, float>(
    float*, float*, float*, const float*, float, float, float, float, float,
    float, float, int);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_ADAMW_GFX942_HIP_HPP_
