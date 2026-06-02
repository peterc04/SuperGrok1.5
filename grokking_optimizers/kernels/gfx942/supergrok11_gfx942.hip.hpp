#ifndef GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
// ============================================================================
// supergrok11_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'supergrok11'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen host orchestration (the public sg::gfx942::launch_supergrok11_*
//       / compute_cosine_gate_fused entry points the bindings call —
//       UNCHANGED, byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN reduction kernel (§5 below) built on the
//       shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — the cosine-gate
//       3-quantity reduction (num = Σ g·m, den_g = Σ g², den_m = Σ m²) computed
//       TOGETHER in one pass: three DPP wavefront reduces, an LDS block tree per
//       quantity, then three AGENT-scope global atomics. The host forms the
//       gate = clamp(num / sqrt(den_g·den_m), 0, 1), replacing the ATen
//       `.sum()`/`.norm()` triple.
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A). It pulls in
//     torch/extension.h + primitives.hpp and exposes the launchers; the thin
//     host launch_supergrok11.hip.cpp TU resolves exactly as before.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device reduction kernel.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics +
// the wave→block→AGENT bit-parity check are deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_supergrok11.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for SuperGrok v1.1.
// Algorithm: csrc/algorithms/supergrok11.h
//
// COMPUTE PATTERN
// Mixed: per-parameter meta-MLP + cosine-similarity gating + AdamW.
//   Per element:
//     mu = phi_mlp(grad, sharpness)        — 2-input × H × 1 MLP
//     cosine = sum(grad * momentum) / (||grad|| * ||momentum||)
//     gate   = clamp(cosine, 0, 1)
//     smart_grad = g + (1-gate) * alpha * mu
//     AdamW(smart_grad)
// The cosine numerator + two denominators are global reductions.
//
// MFMA APPLICABILITY: partial (same as NeuralGrok).
// The MLP forward could route through MFMA if we batch across N; in
// practice ATen + rocBLAS already does this via the rocBLAS dispatch. The
// cosine-gate 3-quantity reduction is the high-value AMDGCN piece (§5).

// ════════════════════════════════════════════════════════════════════════════
// (A) HOST orchestration — ATen public entry points. Compiled by the HOST pass
// only. The free-standing AMDGCN device gate (__AMDGCN__) does NOT see this
// block (torch/extension.h pulls in <cuda.h>/ATen, which the free-standing
// device target cannot resolve); the §5 device kernel below is the device-pass
// content. On a real hipcc build the host pass compiles this and launches the
// §5 kernel via hipLaunchKernelGGL (see §5.LAUNCH).
// ════════════════════════════════════════════════════════════════════════════
#if !defined(__AMDGCN__)
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "csrc/backends/hip/gfx942/primitives.hpp"

#if defined(__HIPCC__)
// LIVE (hipcc) host launch glue: the §5 cosine-gate device kernel (section B)
// and the HIP runtime launch/stream API. On the bare AMDGCN gate (no hipcc)
// this block is skipped, so amdgcn_check.sh never parses these HIP includes.
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#endif

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

#if defined(__HIPCC__)
// Forward declaration of the §5 device kernel (defined in section B below) so
// the host launchers can hipLaunchKernelGGL it; signature matches exactly.
namespace native {
extern "C" __global__ void supergrok11_gfx942_cosine_reduce(
    const float* __restrict__ g, const float* __restrict__ m,
    float* __restrict__ num_acc, float* __restrict__ den_g_acc,
    float* __restrict__ den_m_acc, int n);
// Per-element smart_grad + Adam apply (§5.APPLY; defined in section B below).
// smart = g + (1-gate)*alpha*mu, then m/v EMA + bias-corrected decoupled-WD
// apply. Vectorized to 128-bit (f32x4) memory access with a scalar tail.
template <typename ParamT, typename GradT>
__global__ void supergrok11_gfx942_apply(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, const float* __restrict__ mu,
    const GradT* __restrict__ grad, float gate, float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N);
}  // namespace native
#endif

void launch_supergrok11_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& mu_bufs,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& sharpnesses,
    std::vector<torch::Tensor>& momenta,
    const torch::Tensor& phi_W1,
    const torch::Tensor& phi_b1,
    const torch::Tensor& phi_W2,
    float phi_b2,
    float alpha,
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
        auto h = torch::tanh(torch::matmul(x, phi_W1.t()) + phi_b1);
        auto mu_flat = (torch::matmul(h, phi_W2.unsqueeze(1)) + phi_b2).view_as(g);
        mu.copy_(mu_flat);

        // Cosine gate
        auto gf = g.to(torch::kFloat32);
        auto mom = momenta[i];
#if defined(__HIPCC__)
        // LIVE (hipcc): §5 cosine-gate reduce — one pass yields num=Σg·m,
        // den_g=Σg², den_m=Σm² via DPP wave→block→AGENT atomics (replacing the
        // ATen .sum()/.norm() triple). gate = clamp(num / sqrt(den_g·den_m),0,1).
        float gate;
        {
            auto gfc  = gf.contiguous();
            auto momc = mom.to(torch::kFloat32).contiguous();
            int n = static_cast<int>(gfc.numel());
            auto acc = torch::zeros({3}, gfc.options());
            hipStream_t stream = at::hip::getCurrentHIPStream();
            dim3 grid(std::min(1024, (n + 255) / 256)), block(256);  // 4 wavefronts/block
            hipLaunchKernelGGL(native::supergrok11_gfx942_cosine_reduce, grid, block,
                               0, stream, gfc.data_ptr<float>(), momc.data_ptr<float>(),
                               acc.data_ptr<float>(), acc.data_ptr<float>() + 1,
                               acc.data_ptr<float>() + 2, n);
            auto h_acc = acc.cpu();
            float num   = h_acc[0].item<float>();
            float den_g = h_acc[1].item<float>();
            float den_m = h_acc[2].item<float>();
            float denom = sqrtf(den_g * den_m);
            gate = (denom > 1e-12f) ? (num / denom) : 0.0f;
            gate = std::min(std::max(gate, 0.0f), 1.0f);
        }
#else
        float dot = (gf * mom).sum().item<float>();
        float ng = gf.norm().item<float>();
        float nm = mom.norm().item<float>();
        float gate = (ng * nm > 1e-12f) ? (dot / (ng * nm)) : 0.0f;
        gate = std::min(std::max(gate, 0.0f), 1.0f);
#endif

        // Sweep B: smart_grad + Adam
#if defined(__HIPCC__)
        // LIVE device path: the §5.APPLY f32x4-vectorized smart_grad + Adam
        // kernel fuses smart=g+(1-gate)*alpha*mu, the m/v EMAs, and the bias-
        // corrected decoupled-WD apply into ONE launch (128-bit dwordx4 access).
        {
            auto gfc = gf.contiguous();
            auto muc = mu.to(torch::kFloat32).contiguous();
            int n = static_cast<int>(gfc.numel());
            if (n > 0) {
                hipStream_t stream = at::hip::getCurrentHIPStream();
                dim3 grid(std::min(1024, (n + 255) / 256)), block(256);
                hipLaunchKernelGGL((native::supergrok11_gfx942_apply<float, float>),
                                   grid, block, 0, stream,
                                   p.data_ptr<float>(), m.data_ptr<float>(),
                                   v.data_ptr<float>(), muc.data_ptr<float>(),
                                   gfc.data_ptr<float>(), gate, alpha,
                                   lr, beta1, beta2, eps, wd, bc1, bc2, n);
            }
        }
#else
        auto smart = gf + (1.0f - gate) * alpha * mu;
        prim::ema_update_inplace(m, smart, beta1);
        prim::ema_sq_update_inplace(v, smart, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
#endif
    }
}


void launch_sg11_mu_metanet(
    torch::Tensor mu, torch::Tensor grad, torch::Tensor sharpness, torch::Tensor smart_grad, float alpha, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, torch::Tensor b2, float rescale, int hidden_dim
) {
    // phi network forward: 2-input MLP with tanh activation
    auto gf = grad.to(torch::kFloat32).view({-1});
    auto sf = sharpness.view({-1});
    auto x = torch::stack({gf, sf}, /*dim=*/1);  // [N, 2]
    auto h = torch::tanh(torch::matmul(x, W1.t()) + b1);
    float b2_val = b2.item<float>();
    auto mu_flat = (torch::matmul(h, W2.unsqueeze(1)) + b2_val).view_as(grad) * rescale;
    mu.copy_(mu_flat);
    // smart_grad = grad + alpha * mu
    smart_grad.copy_(gf.view_as(grad) + alpha * mu_flat);
}

void launch_sg11_adam_decay(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor smart_grad, torch::Tensor mu, float lamb_eff, float beta1, float beta2, float lr, float wd_eff, float eps, float bc1, float bc2
) {
    // g = smart_grad + lamb_eff * mu, then Adam update
    auto g = smart_grad + lamb_eff * mu;
    prim::ema_update_inplace(exp_avg, g, beta1);
    prim::ema_sq_update_inplace(exp_avg_sq, g, beta2);
    prim::adam_apply_inplace(param, exp_avg, exp_avg_sq, lr, bc1, bc2, eps, wd_eff);
}

void launch_sg11_sam_perturb(
    torch::Tensor param, torch::Tensor grad, float rho_over_norm
) {
    // param += rho_over_norm * grad
    param.add_(grad.to(param.scalar_type()), rho_over_norm);
}

void launch_sg11_sharpness_restore(
    torch::Tensor param, torch::Tensor sharpness, torch::Tensor backup, torch::Tensor sam_grad, torch::Tensor normal_grad
) {
    // param = backup, sharpness = (sam_grad - normal_grad)^2
    param.copy_(backup);
    auto diff = sam_grad.to(torch::kFloat32) - normal_grad.to(torch::kFloat32);
    sharpness.copy_(diff * diff);
}

float compute_cosine_gate_fused(
    torch::Tensor smart_grad, torch::Tensor mu, float gate_temp
) {
    // cos_sim(smart_grad, mu) clamped to [0, 1]
    auto sg_f = smart_grad.to(torch::kFloat32).flatten().contiguous();
    auto mu_f = mu.to(torch::kFloat32).flatten().contiguous();
#if defined(__HIPCC__)
    // LIVE (hipcc): §5 cosine-gate reduce (num=Σsg·mu, den_g=Σsg², den_m=Σmu²)
    // via DPP wave→block→AGENT atomics; ATen .sum() triple is the #else fallback.
    int n = static_cast<int>(sg_f.numel());
    auto acc = torch::zeros({3}, sg_f.options());
    hipStream_t stream = at::hip::getCurrentHIPStream();
    dim3 grid(std::min(1024, (n + 255) / 256)), block(256);  // 4 wavefronts/block
    hipLaunchKernelGGL(native::supergrok11_gfx942_cosine_reduce, grid, block,
                       0, stream, sg_f.data_ptr<float>(), mu_f.data_ptr<float>(),
                       acc.data_ptr<float>(), acc.data_ptr<float>() + 1,
                       acc.data_ptr<float>() + 2, n);
    auto h_acc = acc.cpu();
    float num = h_acc[0].item<float>(), den_g = h_acc[1].item<float>(), den_m = h_acc[2].item<float>();
    float denom = sqrtf(den_g * den_m + 1e-12f);
    float gate = (denom > 0.0f) ? (num / denom) : 0.0f;
    return std::min(std::max(gate, 0.0f), 1.0f);
#else
    float num = (sg_f * mu_f).sum().item<float>();
    float den_g = (sg_f * sg_f).sum().item<float>();
    float den_m = (mu_f * mu_f).sum().item<float>();
    float denom = sqrtf(den_g * den_m + 1e-12f);
    float gate = (denom > 0.0f) ? (num / denom) : 0.0f;
    return std::min(std::max(gate, 0.0f), 1.0f);
#endif
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring — NOW DISPATCHED) ────────────────────────────
// launch_supergrok11_step() / compute_cosine_gate_fused() DISPATCH the §5
// cosine-gate kernel on hipcc (`#if __HIPCC__`); the ATen `.sum()`/`.norm()`
// triple is the `#else` CPU-host fallback. Live (hipcc) stage:
//   auto acc = torch::zeros({3}, gfc.options());
//   dim3 grid(min(1024, (n+255)/256)), block(256);   // 4 wavefronts/block
//   hipLaunchKernelGGL(native::supergrok11_gfx942_cosine_reduce, grid, block, 0,
//                      stream, g_ptr, m_ptr, acc, acc+1, acc+2, n);
//   // host then: num=acc[0]; den_g=acc[1]; den_m=acc[2];
//   //            denom = sqrtf(den_g*den_m + 1e-12f);
//   //            gate  = clamp(num / denom, 0, 1);
// The per-element meta-net MLP + smart_grad + Adam apply stay on the ATen host
// path (pure elementwise — no MFMA/DPP value). The §5 kernel is
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
// The cosine-gate 3-quantity reduction: THREE parallel global sums in one pass,
//   num   = Σ_i g_i · m_i        (g·m dot product)
//   den_g = Σ_i g_i²             (‖g‖² )
//   den_m = Σ_i m_i²             (‖m‖² )
// then on the host: gate = clamp(num / sqrt(den_g·den_m + 1e-12), 0, 1).
// Each thread accumulates all THREE partials, then we run THREE independent
// wave→block→AGENT-atomic trees (the standard 2-level tree from the spec):
//   1. each thread grid-strides, accumulating num/den_g/den_m;
//   2. amd::wave_reduce_add_dpp on each → per-wavefront sums (§2.6);
//   3. lane 0 of each wavefront writes the three quantities to LDS slots;
//      workgroup_barrier;
//   4. the first wavefront reduces each slot array via a second DPP reduce;
//   5. one thread does three amd::atomic_add_agent_f32 (num/den_g/den_m accs)
//      (§2.13: AGENT-scope atomics, visible across all 8 XCDs of MI300X).
// APPLY: the per-element meta-net MLP + smart_grad + Adam apply stays on the
// ATen host path (launch_supergrok11_step) — only the cosine-gate reduction is
// migrated to AMDGCN.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;
static constexpr int kWave = 64;   // == amd::kWave (CDNA3 wavefront width)

// Block (workgroup) cosine-gate 3-quantity reduction over `[0..n)`, three
// AGENT atomics. blockDim.x must be a multiple of kWave and <= kWave*kWave
// (<= 4096); LDS holds one float per wavefront per quantity (<= 3*64 floats).
__device__ __forceinline__ void supergrok11_cosine_block(
    const float* __restrict__ g, const float* __restrict__ m,
    float* __restrict__ num_acc, float* __restrict__ den_g_acc,
    float* __restrict__ den_m_acc, int n)
{
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid % kWave;
    const int waveId = tid / kWave;
    const int wpb    = static_cast<int>(blockDim.x) / kWave;
    const int gtid   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);

    float num = 0.f, den_g = 0.f, den_m = 0.f;
    for (int i = gtid; i < n; i += stride) {
        float gi = amd::streaming_load(&g[i]);
        float mi = amd::streaming_load(&m[i]);
        num   += gi * mi;
        den_g += gi * gi;
        den_m += mi * mi;
    }
    // three wavefront DPP reduces: every lane holds the wavefront sums.
    num   = amd::wave_reduce_add_dpp(num);
    den_g = amd::wave_reduce_add_dpp(den_g);
    den_m = amd::wave_reduce_add_dpp(den_m);

    __shared__ float lds_num[kWave];
    __shared__ float lds_dg[kWave];
    __shared__ float lds_dm[kWave];
    if (lane == 0) {
        lds_num[waveId] = num;
        lds_dg[waveId]  = den_g;
        lds_dm[waveId]  = den_m;
    }
    amd::workgroup_barrier_release();

    // first wavefront reduces each per-wavefront slot array, then three atomics.
    if (waveId == 0) {
        float sn = (lane < wpb) ? lds_num[lane] : 0.f;
        float sg = (lane < wpb) ? lds_dg[lane]  : 0.f;
        float sm = (lane < wpb) ? lds_dm[lane]  : 0.f;
        sn = amd::wave_reduce_add_dpp(sn);
        sg = amd::wave_reduce_add_dpp(sg);
        sm = amd::wave_reduce_add_dpp(sm);
        if (lane == 0) {
            amd::atomic_add_agent_f32(num_acc,   sn);
            amd::atomic_add_agent_f32(den_g_acc, sg);
            amd::atomic_add_agent_f32(den_m_acc, sm);
        }
    }
}

extern "C" __global__ void supergrok11_gfx942_cosine_reduce(
    const float* __restrict__ g, const float* __restrict__ m,
    float* __restrict__ num_acc, float* __restrict__ den_g_acc,
    float* __restrict__ den_m_acc, int n)
{
    supergrok11_cosine_block(g, m, num_acc, den_g_acc, den_m_acc, n);
}

// ── §5.APPLY  per-element smart_grad + Adam apply (128-bit / f32x4 vectorized) ─
// smart = g + (1-gate)*alpha*mu, then the m/v EMA + bias-corrected decoupled-WD
// apply — BIT-IDENTICAL to the ATen host path (primitives.hpp):
//   m   = beta1*m + (1-beta1)*smart
//   v   = beta2*v + (1-beta2)*smart*smart
//   m_hat = m/bc1 ; v_hat = v/bc2 ; denom = sqrt(v_hat)+eps
//   p  -= lr*(m_hat/denom + wd*p)
// The reduced `gate` is uniform (host-computed); alpha is a scalar. Memory access
// widens to 128-bit dwordx4 (f32x4 streaming_load / streaming_store on
// param/exp_avg/exp_avg_sq/mu/grad); the scalar tail runs the identical math on
// the n%4 remainder. Per-lane math, order, constants and __builtin_sqrtf are
// unchanged from the scalar form — only the access width changes.
using f32x4 = ::sg::gfx942::amdgcn::f32x4;

// Identical per-element apply (used by both the f32x4 lanes and the scalar tail).
__device__ __forceinline__ float sg11_apply_elem(
    float* __restrict__ pp, float* __restrict__ pm, float* __restrict__ pv,
    float mu_i, float g_i, float gate, float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2)
{
    float smart = g_i + (1.0f - gate) * alpha * mu_i;
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
__global__ void supergrok11_gfx942_apply(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, const float* __restrict__ mu,
    const GradT* __restrict__ grad, float gate, float alpha,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int N)
{
    const int gtid   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                       + static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
    const int n4     = N & ~3;   // largest multiple of 4 <= N

    // Vectorized body: 4 contiguous floats / iter via f32x4 (128-bit access).
    for (int base = gtid * 4; base < n4; base += stride * 4) {
        f32x4 pv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(param + base));
        f32x4 mv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(exp_avg + base));
        f32x4 vv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(exp_avg_sq + base));
        f32x4 uv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(mu + base));
        f32x4 gv4 = amd::streaming_load(reinterpret_cast<const f32x4*>(grad + base));
        f32x4 ov4;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float pj = pv4[j], mj = mv4[j], vj = vv4[j];
            ov4[j] = sg11_apply_elem(&pj, &mj, &vj, uv4[j], gv4[j], gate, alpha,
                                     lr, beta1, beta2, eps, wd, bc1, bc2);
            mv4[j] = mj;
            vv4[j] = vj;
        }
        amd::streaming_store(reinterpret_cast<f32x4*>(exp_avg + base), mv4);
        amd::streaming_store(reinterpret_cast<f32x4*>(exp_avg_sq + base), vv4);
        amd::streaming_store(reinterpret_cast<f32x4*>(param + base), ov4);
    }

    // Scalar tail: the n%4 remainder, identical per-element function.
    for (int i = n4 + gtid; i < N; i += stride) {
        float pi = param[i], mi = exp_avg[i], vi = exp_avg_sq[i];
        float out = sg11_apply_elem(&pi, &mi, &vi, mu[i], grad[i], gate, alpha,
                                    lr, beta1, beta2, eps, wd, bc1, bc2);
        exp_avg[i]    = mi;
        exp_avg_sq[i] = vi;
        param[i]      = out;
    }
}

// Force-instantiate the <float,float> apply the host launcher dispatches.
template __global__ void supergrok11_gfx942_apply<float, float>(
    float*, float*, float*, const float*, const float*, float, float,
    float, float, float, float, float, float, float, int);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_SUPERGROK11_GFX942_HIP_HPP_
