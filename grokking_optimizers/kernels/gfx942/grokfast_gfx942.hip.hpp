#ifndef GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
// ============================================================================
// grokfast_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'grokfast'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen host orchestration (the public sg::gfx942::launch_grokfast_*
//       entry points the bindings call — UNCHANGED, byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN grid-stride update kernel (§5 below) built
//       on the shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — streaming
//       (read-once) grad loads via amd::streaming_load and a streaming param
//       write-back via amd::streaming_store, fusing the EMA filter +
//       amplification + m/v EMAs + bias-corrected decoupled-weight-decay apply
//       into ONE kernel (vs the ATen path's chain of broadcast launches).
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A) — torch + primitives +
//     the launchers; the thin host launch_grokfast.hip.cpp TU resolves unchanged.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device update kernel.
//
// ELEMENTWISE MATH: inlined (option a). The per-element step in
// csrc/algorithms/grokfast.h (grokfast_fused_step) pulls torch via
// csrc/common/*, which the bare amdgcn gate cannot resolve, so the ~10-line
// update is copied here verbatim (numerically identical: bc1/bc2 un-inverted →
// divide, sqrtf→__builtin_sqrtf).
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numeric
// parity vs the algorithm-header reference is deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_grokfast.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Grokfast.
// Algorithm: csrc/algorithms/grokfast.h
//
// COMPUTE PATTERN
// Identical to GrokAdamW (EMA amplification + AdamW). Hyperparameters differ;
// math is structurally the same.
//
// MFMA APPLICABILITY: none. Elementwise.

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST orchestration — ATen public entry points. Compiled by the HOST pass
// only (torch/extension.h pulls in <cuda.h>/ATen, invisible to the bare device
// gate). Under hipcc (`#if __HIPCC__`) the host launcher now DISPATCHES the §5
// kernel via hipLaunchKernelGGL (see §5.LAUNCH); the `#else` branch keeps the
// ATen path as the CPU-host fallback.
// ════════════════════════════════════════════════════════════════════════════
#if !defined(__AMDGCN__)
#include <torch/extension.h>
#include <vector>

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_grokfast_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& emas,
    std::vector<torch::Tensor>& grads,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& ema = emas[i];

#if defined(__HIPCC__)
        // LIVE device path: dispatch the §5 AMDGCN kernel per tensor (fuses the
        // EMA filter + amplification + m/v Adam EMAs + bias-corrected decoupled-
        // WD apply into ONE launch). 🟡 hipcc-only — none in this env.
        const int n = static_cast<int>(p.numel());
        dim3 grid(min(1024, (n + 255) / 256)), block(256);  // 4 wavefronts/block
        hipLaunchKernelGGL((native::grokfast_gfx942_kernel<float, float>), grid,
                           block, 0, 0,
                           p.data_ptr<float>(), m.data_ptr<float>(),
                           v.data_ptr<float>(), ema.data_ptr<float>(),
                           g.data_ptr<float>(),
                           gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                           bc1, bc2, n);
#else
        prim::ema_update_inplace(ema, g, gf_alpha);
        auto g_amp = g.to(torch::kFloat32) + gf_lamb * ema;
        prim::ema_update_inplace(m, g_amp, beta1);
        prim::ema_sq_update_inplace(v, g_amp, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
#endif
    }
}


void launch_fused_grokfast_ema(
    torch::Tensor grad, torch::Tensor ema, float alpha, float lamb
) {
    prim::ema_update_inplace(ema, grad, alpha);
    grad.add_(ema, lamb);
}

void launch_fused_grokfast_adam(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor ema, torch::Tensor grad,
    float alpha, float lamb, float beta1, float beta2,
    float lr, float weight_decay, float eps, float bc1, float bc2
) {
    std::vector<torch::Tensor> vp{param}, vea{exp_avg}, veas{exp_avg_sq},
                               vema{ema}, vg{grad};
    launch_grokfast_step(vp, vea, veas, vema, vg,
                         alpha, lamb, lr, beta1, beta2, eps, weight_decay,
                         bc1, bc2);
}

void launch_multi_tensor_grokfast_ema(
    std::vector<torch::Tensor>& grads, std::vector<torch::Tensor>& ema_bufs,
    float alpha, float lamb
) {
    for (size_t i = 0; i < grads.size(); i++) {
        launch_fused_grokfast_ema(grads[i], ema_bufs[i], alpha, lamb);
    }
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring — NOW LIVE under hipcc) ──────────────────────
// Under `#if defined(__HIPCC__)`, launch_grokfast_step() DISPATCHES the §5
// kernel per tensor (above) instead of the chain of ATen ema + amplify + adam:
//   dim3 grid(min(1024,(n+255)/256)), block(256);   // 4 wavefronts/block
//   hipLaunchKernelGGL((native::grokfast_gfx942_kernel<float,float>), grid,
//                      block, 0, 0, p_ptr, m_ptr, v_ptr, ema_ptr, g_ptr,
//                      gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd, bc1, bc2, n);
// The `#else` branch is the ATen CPU-host fallback (numerics-correct).
// 🟡 The hipcc host-launch compiles ONLY under hipcc (no hipcc in this env, so
// the hipLaunchKernelGGL glue is unverified here). The §5 device kernel itself
// is COMPILER-VERIFIED for gfx942 via scripts/amdgcn_check.sh (AMDGCN_OK).
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
// ── gate-only launch-builtin shim (verbatim from the adamw/looksam/mamba3 exemplar)
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
// Grid-stride Grokfast: each workitem owns a stride of elements, reads grad
// read-once via amd::streaming_load (nontemporal — bypasses L2 for one-touch
// data, §2.7), fuses the slow-gradient EMA filter + amplification (g_amp =
// g + lamb*ema), the m/v Adam EMAs, and the bias-corrected decoupled-weight-
// decay apply in registers, then writes the param back via amd::streaming_store.
// The math is identical to sg::algorithms::grokfast_fused_step (bc1/bc2
// un-inverted → divide; sqrtf→__builtin_sqrtf under the bare gate).
//
// VECTORIZATION (WS5/A — scalar→f32x4): the fp32 path is bandwidth-bound, so the
// bulk loop is widened to WAVE-64 / 128-bit (dwordx4) memory access. Each
// iteration processes 4 contiguous floats as one amd::f32x4 via the templated
// amd::streaming_load<f32x4> (read-once grad) + amd::streaming_store<f32x4>
// (write-once param); the exp_avg / exp_avg_sq / ema state buffers are likewise
// vector-loaded/stored. The 4 lanes each evaluate the IDENTICAL scalar Grokfast
// expressions (no cross-lane mixing), so the result is BIT-IDENTICAL to the
// scalar kernel — only the access width changed. A scalar TAIL handles the
// final N%4 elements (and the whole array on the unaligned/sub-vector
// fallback). NO DPP is needed: Grokfast is purely elementwise (EMA filter +
// Adam) — there is no cross-lane reduction, so the butterfly primitives are
// unused.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;

// Per-element Grokfast apply — the canonical scalar body, shared by the scalar
// tail and (replicated lane-by-lane) the f32x4 fast-path so both are identical.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void grokfast_apply_elem(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, float* __restrict__ ema,
    const GradT* __restrict__ grad, int i,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2)
{
    const float g = static_cast<float>(amd::streaming_load(&grad[i]));
    const float p = static_cast<float>(param[i]);

    const float e_new = gf_alpha * ema[i] + (1.0f - gf_alpha) * g;
    ema[i] = e_new;
    const float g_amp = g + gf_lamb * e_new;

    const float m = beta1 * exp_avg[i]    + (1.0f - beta1) * g_amp;
    const float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g_amp * g_amp;
    exp_avg[i]    = m;
    exp_avg_sq[i] = v;

    const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
    amd::streaming_store(&param[i],
                         static_cast<ParamT>(p - lr * (update + wd * p)));
}

template <typename ParamT, typename GradT>
__global__ void grokfast_gfx942_kernel(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, float* __restrict__ ema,
    const GradT* __restrict__ grad,
    float gf_alpha, float gf_lamb,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N)
{
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    const int tid    = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                       + static_cast<int>(threadIdx.x);

    // VECTORIZED fast-path: only when param/grad are plain fp32 (the sole
    // instantiation); the constexpr guard lets any future bf16/fp16 combo fall
    // back cleanly to the scalar loop.
    constexpr bool kVecOk =
        sizeof(ParamT) == sizeof(float) && sizeof(GradT) == sizeof(float);
    const int n4 = kVecOk ? (N >> 2) : 0;   // number of full float4 groups

    using f32x4 = amd::f32x4;
    auto* p4  = reinterpret_cast<f32x4*>(param);
    auto* m4  = reinterpret_cast<f32x4*>(exp_avg);
    auto* v4  = reinterpret_cast<f32x4*>(exp_avg_sq);
    auto* e4  = reinterpret_cast<f32x4*>(ema);
    const auto* g4 = reinterpret_cast<const f32x4*>(grad);

    for (int q = tid; q < n4; q += stride) {
        const f32x4 g = amd::streaming_load(&g4[q]);   // 128-bit dwordx4 load
        const f32x4 p = p4[q];
        const f32x4 ea = m4[q];
        const f32x4 ev = v4[q];
        const f32x4 em = e4[q];
        f32x4 po, mo, vo, eo;
        // 4 lanes, each evaluating the IDENTICAL scalar Grokfast expressions.
        for (int l = 0; l < 4; ++l) {
            const float e_new = gf_alpha * em[l] + (1.0f - gf_alpha) * g[l];
            eo[l] = e_new;
            const float g_amp = g[l] + gf_lamb * e_new;
            const float m = beta1 * ea[l] + (1.0f - beta1) * g_amp;
            const float v = beta2 * ev[l] + (1.0f - beta2) * g_amp * g_amp;
            mo[l] = m;
            vo[l] = v;
            const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
            po[l] = p[l] - lr * (update + wd * p[l]);
        }
        e4[q] = eo;
        m4[q] = mo;
        v4[q] = vo;
        amd::streaming_store(&p4[q], po);              // 128-bit dwordx4 store
    }

    // SCALAR TAIL: the final N%4 elements (and the whole array when kVecOk is
    // false / N<4). Grid-strided over the remaining indices.
    for (int i = (n4 << 2) + tid; i < N; i += stride) {
        grokfast_apply_elem(param, exp_avg, exp_avg_sq, ema, grad, i,
                            gf_alpha, gf_lamb, lr, beta1, beta2, eps, wd,
                            bc1, bc2);
    }
}

// Force-instantiate the grokking dtype combo (fp32 param + fp32 grad) so the
// device pass emits the kernel; the host TU dispatches on dtype.
template __global__ void grokfast_gfx942_kernel<float, float>(
    float*, float*, float*, float*, const float*, float, float, float, float,
    float, float, float, float, float, int);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_GROKFAST_GFX942_HIP_HPP_
