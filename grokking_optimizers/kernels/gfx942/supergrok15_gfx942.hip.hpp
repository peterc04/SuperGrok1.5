#ifndef GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
// ============================================================================
// supergrok15_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'supergrok15'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen host orchestration (the public sg::gfx942::launch_supergrok15_*
//       entry points the bindings call — UNCHANGED, byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN reduction kernel (§5 below) built on the
//       shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — the sharpness
//       reduction (the global gate signal Σ_i sharpness_i, used to form the
//       global gate scale) as the canonical wave→block→AGENT-atomic global sum
//       (DPP wavefront reduce, LDS block tree, then a single AGENT-scope global
//       atomic), replacing the ATen `.sum()`.
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A). It pulls in
//     torch/extension.h + primitives.hpp and exposes the launchers; the thin
//     host launch_supergrok15.hip.cpp TU resolves exactly as before.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device reduction kernel.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics +
// the wave→block→AGENT bit-parity check are deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_supergrok15.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for SuperGrok v1.5.
// Algorithm: csrc/algorithms/supergrok15.h
//
// COMPUTE PATTERN
// Mixed: meta-MLP + per-coord alpha gate + sharpness backward + AdamW.
//   Per element:
//     mu = phi_mlp(grad, sharpness)        — 2-input × H × 1 MLP
//     alpha = clamp(alpha_base * (1 + mu), 0, alpha_max)
//     smart_grad = g + gate_signal * alpha * mu
//     AdamW(smart_grad)
//   Plus: sharpness EMA update (separate kernel).
//
// MFMA APPLICABILITY: same as NeuralGrok / SG11 — partial via rocBLAS dispatch
// for the MLP. The sharpness/gate-signal reduction is the high-value AMDGCN
// piece (§5).

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
// `.hip.cpp` host TU) — avoids duplicate launch_supergrok15_* symbols at link.
#if !defined(__AMDGCN__) && !defined(SG_GFX942_DEVICE_TU)
#include <torch/extension.h>
#include <vector>
#include <cmath>

#include "csrc/backends/hip/gfx942/primitives.hpp"

#if defined(__HIPCC__)
// LIVE (hipcc) host launch glue: the §5 sharpness-reduce device kernel
// (section B) and the HIP runtime launch/stream API. On the bare AMDGCN gate
// (no hipcc) this block is skipped, so amdgcn_check.sh never parses these HIP
// includes.
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#endif

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

#if defined(__HIPCC__)
// Forward declaration of the §5 device kernel (defined in section B below) so
// the host launcher can hipLaunchKernelGGL it; signature matches exactly.
namespace native {
extern "C" __global__ void supergrok15_gfx942_sharpness_reduce(
    const float* __restrict__ sharpness, float* __restrict__ acc, int64_t n);
// Per-element smart_grad + Adam apply (§5.APPLY; defined in section B below).
// smart = g + gate_global*clamp(alpha_base*(1+mu),0,alpha_max)*mu, then the m/v
// EMA + bias-corrected decoupled-WD apply. Vectorized to 128-bit (f32x4) memory
// access with a scalar tail.
template <typename ParamT, typename GradT>
__global__ void supergrok15_gfx942_apply(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, const float* __restrict__ mu,
    const GradT* __restrict__ grad, float gate, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N);
}  // namespace native
#endif

void launch_supergrok15_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mu_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sharpnesses,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    float gate_global,
    float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& mu = mu_bufs[i];

        // Sweep A: meta-net forward
        auto x = torch::stack({g.to(torch::kFloat32).view({-1}),
                               sharpnesses[i].view({-1})}, /*dim=*/1);
        auto h = torch::gelu(torch::matmul(x, phi_W1.t()) + phi_b1);  // exact GELU (canonical PyTorch meta-net)
        auto mu_flat = (torch::matmul(h, phi_W2.unsqueeze(1)) + phi_b2).view_as(g);
        mu.copy_(mu_flat);

        // Per-coord alpha gate signal: the sharpness-driven scale.
        float gate = gate_global;
#if defined(__HIPCC__)
        // LIVE (hipcc): §5 sharpness reduce — one pass yields acc = Σ_i
        // sharpness_i via DPP wave→block→AGENT atomics (replacing an ATen
        // .sum()). The host forms the sharpness-driven gate scale from the mean
        // and folds it into the externally-supplied gate_global.
        {
            auto sh = sharpnesses[i].to(torch::kFloat32).contiguous();
            const int64_t n = static_cast<int64_t>(sh.numel());  // Stage 1: 64-bit
            if (n > 0) {
                auto acc = torch::zeros({1}, sh.options());
                hipStream_t stream = at::hip::getCurrentHIPStream();
                dim3 grid(static_cast<unsigned>(
                    std::min<int64_t>(1024, (n + 255) / 256))), block(256);
                hipLaunchKernelGGL(native::supergrok15_gfx942_sharpness_reduce,
                                   grid, block, 0, stream,
                                   sh.data_ptr<float>(), acc.data_ptr<float>(), n);
                SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK
                float sharp_mean = acc.item<float>() / static_cast<float>(n);
                gate = gate_global * (1.0f / (1.0f + sharp_mean));
            }
        }
#endif

        // Sweep B: per-coord alpha gate + smart_grad + Adam apply.
#if defined(__HIPCC__)
        // LIVE device path: the §5.APPLY f32x4-vectorized kernel fuses
        // a_per_coord=clamp(alpha_base*(1+mu),0,alpha_max), smart=g+gate*a*mu,
        // the m/v EMAs, and the bias-corrected decoupled-WD apply into ONE launch
        // (128-bit dwordx4 access). Dispatched exactly like sg11 (the apply was
        // previously DEFINED but never launched — WS5/item 4). gate already folds
        // the on-device sharpness mean (above). The `#else` keeps the ATen path.
        {
            auto gfc = g.to(torch::kFloat32).contiguous();
            auto muc = mu.to(torch::kFloat32).contiguous();
            const int64_t n = static_cast<int64_t>(gfc.numel());  // Stage 1: 64-bit
            if (n > 0) {
                hipStream_t stream = at::hip::getCurrentHIPStream();
                dim3 grid(static_cast<unsigned>(
                    std::min<int64_t>(1024, (n + 255) / 256))), block(256);
                hipLaunchKernelGGL((native::supergrok15_gfx942_apply<float, float>),
                                   grid, block, 0, stream,
                                   p.data_ptr<float>(), m.data_ptr<float>(),
                                   v.data_ptr<float>(), muc.data_ptr<float>(),
                                   gfc.data_ptr<float>(), gate, alpha_base, alpha_max,
                                   lr, beta1, beta2, eps, wd, bc1, bc2, n);
                SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK
            }
        }
#else
        // Per-coord alpha, then smart_grad
        auto a_per_coord = torch::clamp(alpha_base * (1.0f + mu), 0.0f, alpha_max);
        auto smart = g.to(torch::kFloat32) + gate * a_per_coord * mu;

        prim::ema_update_inplace(m, smart, beta1);
        prim::ema_sq_update_inplace(v, smart, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
#endif
    }
}


void launch_fused_supergrok15_full_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness, float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2, float rescale, float lamb_eff, float beta1, float beta2, float lr, float wd_eff, float eps, float bc1, float bc2, int hidden_dim
) {
    // Delegate to the working launch_supergrok15_step, wrapping single tensors in vectors
    float b2_val = b2.item<float>();
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> vm{exp_avg};
    std::vector<torch::Tensor> vv{exp_avg_sq};
    std::vector<torch::Tensor> vmu{mu};
    std::vector<torch::Tensor> vg{grad};
    std::vector<torch::Tensor> vs{sharpness};
    float gate_global = lamb_eff;
    float alpha_base = alpha;
    float alpha_max = alpha;
    launch_supergrok15_step(vp, vm, vv, vmu, vg, vs,
                            W1, b1, W2, b2_val,
                            gate_global, alpha_base, alpha_max,
                            lr, beta1, beta2, eps, wd_eff, bc1, bc2);
}

void launch_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    // param += rho_over_norm * grad
    param.add_(grad.to(param.scalar_type()), rho_over_norm);
}

void launch_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup, torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    // param = backup, sharpness = (sam_grad - normal_grad)^2
    param.copy_(backup);
    auto diff = sam_grad.to(torch::kFloat32) - normal_grad.to(torch::kFloat32);
    sharpness.copy_(diff * diff);
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring — NOW DISPATCHED) ────────────────────────────
// launch_supergrok15_step() DISPATCHES the §5 sharpness reduce on hipcc
// (`#if __HIPCC__`); the ATen-scalar `gate_global` path is the `#else` CPU-host
// fallback. The global gate scale (the mean of the per-coordinate sharpness
// across the parameter) is computed on-device per param:
//   auto acc = torch::zeros({1}, sh.options());
//   dim3 grid(min(1024, (n+255)/256)), block(256);   // 4 wavefronts/block
//   hipLaunchKernelGGL(native::supergrok15_gfx942_sharpness_reduce, grid, block,
//                      0, stream, sh.data_ptr<float>(), acc.data_ptr<float>(), n);
//   // host then: sharp_mean = acc/ n;  gate = gate_global / (1 + sharp_mean)
// The per-coord alpha gate + smart_grad + Adam apply is NOW DISPATCHED on hipcc
// to the §5.APPLY f32x4 kernel (native::supergrok15_gfx942_apply<float,float>),
// matching sg11 (WS5/item 4 — previously defined but never launched). The meta-
// net MLP stays on the ATen host path (rocBLAS). The §5 kernels are
// COMPILER-VERIFIED for gfx942 via scripts/amdgcn_check.sh.
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
// The sharpness reduction: the global gate signal Σ_i sharpness_i over the
// per-coordinate sharpness map (sharpness_i = (g_sam_i − g_i)², the SAM
// curvature proxy). The high-value AMDGCN piece is the global sum; the host
// forms the sharpness-driven gate scale from it (e.g. mean = Σ/n). This is the
// canonical 2-level (wave→block→grid) tree from the task spec:
//   1. each thread grid-strides sharpness_i, accumulating a per-thread partial;
//   2. amd::wave_reduce_add_dpp gives every lane the per-wavefront sum (§2.6);
//   3. lane 0 of each wavefront writes its sum to an LDS slot; workgroup_barrier;
//   4. the first wavefront reduces the LDS slots via a second DPP reduce;
//   5. one thread does amd::atomic_add_agent_f32 to the global accumulator
//      (§2.13: AMD has no DSMEM, so cross-workgroup uses an AGENT-scope atomic
//      visible across all 8 XCDs of MI300X).
// APPLY: the per-element meta-net MLP + per-coord alpha gate + smart_grad + Adam
// apply stays on the ATen host path (launch_supergrok15_step) — only the
// sharpness/gate-signal reduction is migrated to AMDGCN.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;
static constexpr int kWave = 64;   // == amd::kWave (CDNA3 wavefront width)

// Stage-1 NaN/Inf gradient sanitization (mirror of sm_90 sg_sanitize_grad).
// Identity unless built with -DSG_SANITIZE_NONFINITE=1; applied on the APPLY
// path's grad read only (the sharpness reduce is unchanged). Default byte-identical.
#ifndef SG_SANITIZE_NONFINITE
#define SG_SANITIZE_NONFINITE 0
#endif
__device__ __forceinline__ float sg_sanitize_grad(float g) {
#if SG_SANITIZE_NONFINITE
    return __builtin_isfinite(g) ? g : 0.0f;
#else
    return g;
#endif
}

// Block (workgroup) sum of `sharpness[0..n)`, atomically added to *acc.
// blockDim.x must be a multiple of kWave and <= kWave*kWave (<= 4096); LDS holds
// one float per wavefront (<= 64 floats).
__device__ __forceinline__ void supergrok15_sharpness_block(
    const float* __restrict__ sharpness, float* __restrict__ acc, int64_t n)
{
    // LDS/lane bookkeeping is LOCAL to the block (<= 4096 threads) → int. The
    // GLOBAL element index/stride is int64 so the reduction cannot wrap (Stage 1).
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid % kWave;
    const int waveId = tid / kWave;
    const int wpb    = static_cast<int>(blockDim.x) / kWave;
    const int64_t gtid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    float local = 0.f;
    for (int64_t i = gtid; i < n; i += stride) {
        local += amd::streaming_load(&sharpness[i]);
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

// Bandwidth-bound reduction → request high occupancy (256-thread block = 4
// wavefronts; 8 waves/EU) to hide global-memory latency. (WS5 occupancy wire.)
extern "C" SG_KERNEL_BOUNDS(256, 8) void supergrok15_gfx942_sharpness_reduce(
    const float* __restrict__ sharpness, float* __restrict__ acc, int64_t n)
{
    supergrok15_sharpness_block(sharpness, acc, n);
}

// ── §5.APPLY  per-element smart_grad + Adam apply (128-bit / f32x4 vectorized) ─
// Per-coord alpha gate + smart_grad + Adam, BIT-IDENTICAL to the ATen host path
// (launch_supergrok15_step + primitives.hpp):
//   a_per_coord = clamp(alpha_base*(1+mu), 0, alpha_max)
//   smart       = g + gate*a_per_coord*mu
//   m   = beta1*m + (1-beta1)*smart
//   v   = beta2*v + (1-beta2)*smart*smart
//   m_hat = m/bc1 ; v_hat = v/bc2 ; denom = sqrt(v_hat)+eps
//   p  -= lr*(m_hat/denom + wd*p)
// `gate` (the sharpness-folded gate_global) is uniform; alpha_base/alpha_max are
// scalars; the per-coordinate alpha is recomputed per element. Memory access
// widens to 128-bit dwordx4 (f32x4 streaming_load/streaming_store on
// param/exp_avg/exp_avg_sq/mu/grad); the scalar tail runs the identical math on
// the n%4 remainder. Per-lane math, order, constants and __builtin_sqrtf are
// unchanged from the scalar form — only the access width changes.
using f32x4 = ::sg::gfx942::amdgcn::f32x4;

__device__ __forceinline__ float sg15_clampf(float x, float lo, float hi) {
    // Matches torch::clamp(x, lo, hi) element semantics.
    return __builtin_fminf(__builtin_fmaxf(x, lo), hi);
}

// Identical per-element apply (used by both the f32x4 lanes and the scalar tail).
__device__ __forceinline__ float sg15_apply_elem(
    float* __restrict__ pp, float* __restrict__ pm, float* __restrict__ pv,
    float mu_i, float g_i, float gate, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2)
{
    float a_per_coord = sg15_clampf(alpha_base * (1.0f + mu_i), 0.0f, alpha_max);
    float smart = g_i + gate * a_per_coord * mu_i;
    float m = beta1 * (*pm) + (1.0f - beta1) * smart;
    float v = beta2 * (*pv) + (1.0f - beta2) * smart * smart;
    *pm = m;
    *pv = v;
    float m_hat = m / bc1;
    float v_hat = v / bc2;
    float denom = __builtin_sqrtf(v_hat) + eps;
    float update = m_hat / denom + wd * (*pp);
    return (*pp) - lr * update;
}

template <typename ParamT, typename GradT>
SG_KERNEL_BOUNDS(256, 8) void supergrok15_gfx942_apply(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, const float* __restrict__ mu,
    const GradT* __restrict__ grad, float gate, float alpha_base, float alpha_max,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N)
{
    // 64-bit grid-stride indexing (Stage 1): cast BEFORE the multiply.
    const int64_t gtid   = static_cast<int64_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t n4     = N & ~static_cast<int64_t>(3);   // largest mult of 4 <= N

    // Vectorized body: 4 contiguous floats / iter via f32x4 (128-bit access).
    // NONTEMPORAL POLICY (matches adamw_gfx942): grad + mu are read-once this
    // step → streaming (nontemporal, L2-bypass); exp_avg / exp_avg_sq are
    // recurring STATE (read+written every step, reused next step) → CACHED;
    // param is read once then written once → cached load + streaming store.
    for (int64_t base = gtid * 4; base < n4; base += stride * 4) {
        f32x4 pv4 = amd::cached_load(reinterpret_cast<const f32x4*>(param + base));
        f32x4 mv4 = amd::cached_load(reinterpret_cast<const f32x4*>(exp_avg + base));
        f32x4 vv4 = amd::cached_load(reinterpret_cast<const f32x4*>(exp_avg_sq + base));
        f32x4 uv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(mu + base));
        f32x4 gv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(grad + base));
        f32x4 ov4;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float pj = pv4[j], mj = mv4[j], vj = vv4[j];
            ov4[j] = sg15_apply_elem(&pj, &mj, &vj, uv4[j],
                                     sg_sanitize_grad(gv4[j]), gate,
                                     alpha_base, alpha_max,
                                     lr, beta1, beta2, eps, wd, bc1, bc2);
            mv4[j] = mj;
            vv4[j] = vj;
        }
        amd::cached_store(reinterpret_cast<f32x4*>(exp_avg + base), mv4);
        amd::cached_store(reinterpret_cast<f32x4*>(exp_avg_sq + base), vv4);
        amd::streaming_store(reinterpret_cast<f32x4*>(param + base), ov4);
    }

    // Scalar tail: the n%4 remainder, identical per-element function.
    for (int64_t i = n4 + gtid; i < N; i += stride) {
        float pi = param[i], mi = exp_avg[i], vi = exp_avg_sq[i];
        float out = sg15_apply_elem(&pi, &mi, &vi, mu[i],
                                    sg_sanitize_grad(grad[i]), gate,
                                    alpha_base, alpha_max,
                                    lr, beta1, beta2, eps, wd, bc1, bc2);
        exp_avg[i]    = mi;
        exp_avg_sq[i] = vi;
        param[i]      = out;
    }
}

// Force-instantiate the <float,float> apply the host launcher dispatches.
template __global__ void supergrok15_gfx942_apply<float, float>(
    float*, float*, float*, const float*, const float*, float, float, float,
    float, float, float, float, float, float, float, int64_t);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_SUPERGROK15_GFX942_HIP_HPP_
