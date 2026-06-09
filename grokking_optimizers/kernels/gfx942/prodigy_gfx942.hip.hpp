#ifndef GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
// ============================================================================
// prodigy_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'prodigy'.
//
// AMDGCN-asm status (Stage 5 — AMD-native): this file now carries BOTH
//   (A) the ATen host orchestration (the public sg::gfx942::launch_prodigy_*
//       entry points the bindings call — UNCHANGED, byte-for-byte), AND
//   (B) a REAL hand-written AMDGCN reduction kernel (§5 below) built on the
//       shared, compiler-verified primitives
//       csrc/backends/hip/gfx942/amdgcn_primitives.hip.hpp — the prodigy
//       r-sum and s-sum global reductions (the prodigy_reduce step) computed
//       TOGETHER in one pass: each thread accumulates r_local += g·(p_init−p)·d²
//       and s_local += |s| (or d²·|g|), two DPP wavefront reduces, an LDS block
//       tree for each, then two AGENT-scope global atomics. Mirrors the sm_90
//       prodigy_reduce_kernel, replacing the ATen `.sum()` pair.
//
// COMPILE ROUTING (two passes, one header):
//   * HOST pass  (`!__AMDGCN__`): sees ONLY section (A). It pulls in
//     torch/extension.h + primitives.hpp and exposes the launchers; the thin
//     host launch_prodigy.hip.cpp TU resolves exactly as before.
//   * DEVICE pass (`__AMDGCN__` — scripts/amdgcn_check.sh — or `__HIPCC__`):
//     sees ONLY section (B), the device reduction kernel.
//
// HARDWARE-GATED 🟡: device-compile-verified for gfx942 only; MI300X numerics +
// the wave→block→AGENT bit-parity check are deferred — see
// HARDWARE_VALIDATION.md, Stage 5.
//
// The production TU csrc/backends/hip/gfx942/launch_prodigy.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for Prodigy.
// Algorithm: csrc/algorithms/prodigy.h
//
// COMPUTE PATTERN
// Mixed: per-element + reduction.
//   Per element: r_local += g * (p_init - p) * d²
//                s_local += d² * |g|
//                AdamW apply with d as the lr scale
//   Reduction:   r_global = sum(r_local) across all elements (single FP32 scalar)
//                s_global = sum(s_local) across all elements
//                d_new = max(d_prev, r_global / |s_global|)
// The reduction is the bottleneck: needs wavefront reduce → LDS tree reduce
// → cross-block (cooperative or atomic) final reduce.
//
// MFMA APPLICABILITY: none.
// The reduction needs wave-reduce (DPP on CDNA3), then LDS-tree across waves
// in a block, then a single AGENT-scope atomic to a global counter. No GEMM,
// no MFMA. The §5 AMDGCN kernel implements exactly this 2-level tree.
//
// WHY ATEN HERE (for the apply)
// ATen's `.sum()` dispatches to rocPRIM's segmented reduction, which on
// MI300X already uses wave-reduce + LDS-tree internally; the §5 kernel
// hand-writes the r/s reduction with the DPP→LDS→AGENT-atomic tree (the
// high-value AMDGCN piece). The AdamW apply stays on the ATen host path.

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
// `.hip.cpp` host TU) — avoids duplicate launch_prodigy_* symbols at link.
#if !defined(__AMDGCN__) && !defined(SG_GFX942_DEVICE_TU)
#include <torch/extension.h>
#include <vector>
#include <algorithm>

#include "csrc/backends/hip/gfx942/primitives.hpp"

#if defined(__HIPCC__)
// LIVE host-launch path: HIP runtime (hipLaunchKernelGGL, dim3) + ATen↔HIP
// stream accessor, and a forward decl of the §5 r/s reduction kernel so the
// host reduce loops below can dispatch it. The body is in section (B), compiled
// by the hipcc DEVICE pass from this same header (__HIPCC__ gate).
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
namespace sg { namespace gfx942 { namespace native {
extern "C" __global__ void prodigy_gfx942_rs_reduce(
    const float* __restrict__ g, const float* __restrict__ p,
    const float* __restrict__ p_init, float* __restrict__ r_acc,
    float* __restrict__ s_acc, float d_prev, int64_t n);
}}}
#endif

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

void launch_prodigy_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& s_tracks,
    std::vector<torch::Tensor>& param_inits,
    std::vector<torch::Tensor>& grads,
    torch::Tensor& d_t,
    float d_prev,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    // Reduce r, s across all parameters.
    auto r_sum = torch::zeros({}, d_t.options());
    auto s_sum = torch::zeros({}, d_t.options());
#if defined(__HIPCC__)
    // LIVE (hipcc): the §5 DPP r/s reduction. The kernel folds d_prev² (r) and
    // d_prev² (s) in directly: r = Σ g·(p_init−p)·d_prev², s = Σ d_prev²·|g| —
    // numerically identical to the ATen body's two scaled .sum()s. One launch
    // per tensor accumulates into a 2-float AGENT accumulator [r,s].
    // §5.LAUNCH grid/block: block(256) = 4 wavefronts, grid = min(1024,(n+255)/256).
    auto rs_acc = torch::zeros({2}, torch::TensorOptions()
                                        .device(d_t.device()).dtype(torch::kFloat32));
    auto stream = at::hip::getCurrentHIPStream();
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = grads[i].to(torch::kFloat32).contiguous();
        auto pf = params[i].to(torch::kFloat32).contiguous();
        auto pif = param_inits[i].to(torch::kFloat32).contiguous();
        // 64-bit safe element count + grid sizing (Stage 1).
        const int64_t n = static_cast<int64_t>(gf.numel());
        dim3 block(256), grid(static_cast<unsigned>(
            std::min<int64_t>(1024, (n + 255) / 256)));
        hipLaunchKernelGGL(native::prodigy_gfx942_rs_reduce, grid, block, 0, stream,
                           gf.data_ptr<float>(), pf.data_ptr<float>(),
                           pif.data_ptr<float>(),
                           rs_acc.data_ptr<float>(), rs_acc.data_ptr<float>() + 1,
                           d_prev, n);
        SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK
    }
    r_sum += rs_acc[0];
    s_sum += rs_acc[1];
#else
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& pi = param_inits[i];
        auto& g = grads[i];

        auto delta = (pi - p).to(torch::kFloat32);
        r_sum += (g.to(torch::kFloat32) * delta).sum() * (d_prev * d_prev);
        s_sum += (g.to(torch::kFloat32).abs().sum()) * (d_prev * d_prev);
    }
#endif

    // Update d (on-device scalar).
    auto candidate = r_sum / (s_sum.abs() + 1e-12f);
    d_t.copy_(torch::maximum(d_t.new_full({}, d_prev), candidate));

    float d_val = d_t.item<float>();

    // Apply Adam with d as effective lr.
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];
        auto& st = s_tracks[i];

        auto g_scaled = d_val * g.to(torch::kFloat32);
        prim::ema_update_inplace(m, g_scaled, beta1);
        prim::ema_sq_update_inplace(v, g_scaled, beta2);
        st.add_(g.to(torch::kFloat32), d_val);
        prim::adam_apply_inplace(p, m, v, d_val, bc1, bc2, eps, wd);
    }
}


void launch_fused_prodigy_step(
    torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor s, torch::Tensor param_init, torch::Tensor grad, float lr, float d_lr, float beta1, float beta2, float weight_decay, float eps, float bc1, float bc2
) {
    auto d_t = torch::tensor({d_lr},
        torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    std::vector<torch::Tensor> vp{param};
    std::vector<torch::Tensor> vm{exp_avg};
    std::vector<torch::Tensor> vv{exp_avg_sq};
    std::vector<torch::Tensor> vs{s};
    std::vector<torch::Tensor> vpi{param_init};
    std::vector<torch::Tensor> vg{grad};
    launch_prodigy_step(vp, vm, vv, vs, vpi, vg,
                        d_t, d_lr,
                        beta1, beta2, eps, weight_decay, bc1, bc2);
}

void launch_prodigy_dlr_reduce(
    torch::Tensor grad, torch::Tensor param, torch::Tensor param_init, torch::Tensor s, torch::Tensor numerator, torch::Tensor denominator, float eps
) {
    // numerator += sum(grad * (param - param_init))
    auto gf = grad.to(torch::kFloat32);
    auto delta = (param - param_init).to(torch::kFloat32);
    numerator.add_((gf * delta).sum());
    // denominator += sum(|s|)
    denominator.add_(s.to(torch::kFloat32).abs().sum());
}

// Canonical persistent-EMA Prodigy estimator (mirrors the sm_90 launcher):
// r_num_buf / s_den_buf are running EMAs that PERSIST across steps, decayed by
// beta3 = sqrt(beta2) here (NOT zeroed) before this step's r/s reduction is
// added on top, so d = max(d_prev, d_coef·r_ema/|s_ema|) PLATEAUS once params
// stop drifting from init (the instantaneous form ratcheted d up without bound
// and collapsed training). prodigy.h device math is UNCHANGED — the EMA/beta3/
// d_coef live in this host launcher.
void launch_multi_tensor_prodigy_fused_reduce_step(
    std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& grads, std::vector<torch::Tensor>& param_inits, std::vector<torch::Tensor>& exp_avgs, std::vector<torch::Tensor>& exp_avg_sqs, std::vector<torch::Tensor>& s_bufs, std::vector<float>& bc1s, std::vector<float>& bc2s, torch::Tensor d_lr_buf, torch::Tensor r_num_buf, torch::Tensor s_den_buf, float beta1, float beta2, float beta3, float lr, float wd, float eps, float d_coef
) {
    if (params.empty()) return;
    auto dev = params[0].device();
    float d_prev = d_lr_buf.item<float>();

    // Phase 1: PERSISTENT-EMA accumulate. Seed r/s with the beta3-decayed EMAs
    // carried in from the host, then add this step's per-tensor reduction.
    auto r_sum = (r_num_buf.to(torch::kFloat32) * beta3).reshape({});
    auto s_sum = (s_den_buf.to(torch::kFloat32) * beta3).reshape({});
#if defined(__HIPCC__)
    // LIVE (hipcc): §5 DPP r/s reduction, one launch per tensor → 2-float AGENT
    // accumulator [r,s]. d_prev² / d_prev² folded in the kernel; identical to ATen.
    auto rs_acc = torch::zeros({2}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto stream = at::hip::getCurrentHIPStream();
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = grads[i].to(torch::kFloat32).contiguous();
        auto pf = params[i].to(torch::kFloat32).contiguous();
        auto pif = param_inits[i].to(torch::kFloat32).contiguous();
        // 64-bit safe element count + grid sizing (Stage 1).
        const int64_t n = static_cast<int64_t>(gf.numel());
        dim3 block(256), grid(static_cast<unsigned>(
            std::min<int64_t>(1024, (n + 255) / 256)));
        hipLaunchKernelGGL(native::prodigy_gfx942_rs_reduce, grid, block, 0, stream,
                           gf.data_ptr<float>(), pf.data_ptr<float>(),
                           pif.data_ptr<float>(),
                           rs_acc.data_ptr<float>(), rs_acc.data_ptr<float>() + 1,
                           d_prev, n);
        SG_HIP_LAUNCH_CHECK(stream);  // mirror sm_90 SG_LAUNCH_CHECK
    }
    r_sum += rs_acc[0];
    s_sum += rs_acc[1];
#else
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = grads[i].to(torch::kFloat32);
        auto delta = (param_inits[i] - params[i]).to(torch::kFloat32);
        r_sum += (gf * delta).sum() * (d_prev * d_prev);
        s_sum += gf.abs().sum() * (d_prev * d_prev);
    }
#endif

    // Persist the UNSCALED EMA numerator/denominator back to the host scalars
    // (d_coef is applied only to the candidate, not the stored EMA).
    r_num_buf.copy_(r_sum.reshape({1}));
    s_den_buf.copy_(s_sum.reshape({1}));

    // Phase 2: update d = max(d_prev, d_coef·r_ema/|s_ema|).
    auto candidate = (d_coef * r_sum) / (s_sum.abs() + 1e-12f);
    d_lr_buf.copy_(torch::maximum(d_lr_buf.new_full({}, d_prev), candidate));
    float d_val = d_lr_buf.item<float>();

    // Phase 3: apply Adam with d as effective lr
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto gf = d_val * grads[i].to(torch::kFloat32);
        prim::ema_update_inplace(exp_avgs[i], gf, beta1);
        prim::ema_sq_update_inplace(exp_avg_sqs[i], gf, beta2);
        s_bufs[i].add_(grads[i].to(torch::kFloat32), d_val);
        prim::adam_apply_inplace(params[i], exp_avgs[i], exp_avg_sqs[i],
                                 d_val, bc1s[i], bc2s[i], eps, wd);
    }
}

}} // namespace sg::gfx942

// ── §5.LAUNCH (host-side wiring — NOW LIVE on hipcc) ─────────────────────────
// launch_prodigy_step() and launch_multi_tensor_prodigy_fused_reduce_step()
// above DISPATCH the §5 r/s reduction under `#if defined(__HIPCC__)` instead of
// the ATen `.sum()` pair; the ATen `.sum()` loop is the `#else` CPU-host
// fallback. Stages match the ATen body: reduce r/s → host d-update → ATen apply.
//   dim3 block(256), grid(min(1024, (n+255)/256));   // 4 wavefronts/block
//   hipLaunchKernelGGL(native::prodigy_gfx942_rs_reduce, grid, block, 0, stream,
//                      g_ptr, p_ptr, pinit_ptr, rs+0, rs+1, d_prev, n);
//   // host then: d_new = max(d_prev, rs[0] / (|rs[1]| + 1e-12f))
// (launch_prodigy_dlr_reduce keeps ATen on both paths: its denominator is
// sum(|s_track|), for which no §5 device kernel is defined — honest no-op.)
// 🟡 host-launch DEFERRED: the live hipLaunchKernelGGL glue COMPILES only on a
// real hipcc/ROCm toolchain (no hipcc here) and the MI300X numeric bit-parity
// check is still gated. The §5 device kernel is COMPILER-VERIFIED for gfx942 via
// scripts/amdgcn_check.sh; the host glue links on hardware.
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
// The prodigy_reduce step: TWO parallel global sums computed in one pass,
// mirroring the sm_90 prodigy_reduce_kernel —
//   r_global = Σ_i g_i · (p_init_i − p_i) · d_prev²    (numerator)
//   s_global = Σ_i d_prev² · |g_i|                       (|denominator|)
// then on the host: d_new = max(d_prev, r_global / (|s_global| + 1e-12)).
// Each thread accumulates BOTH partials, then we run TWO independent
// wave→block→AGENT-atomic trees (the standard 2-level tree from the spec):
//   1. each thread grid-strides, accumulating r_local and s_local;
//   2. amd::wave_reduce_add_dpp on each → per-wavefront r/s sums (§2.6);
//   3. lane 0 of each wavefront writes r/s to LDS slots; workgroup_barrier;
//   4. the first wavefront reduces both slot arrays via a second DPP reduce;
//   5. one thread does two amd::atomic_add_agent_f32 (to r_acc and s_acc)
//      (§2.13: AGENT-scope atomics, visible across all 8 XCDs of MI300X).
// APPLY: the per-element Adam apply (with d as the effective lr) stays on the
// ATen host path (launch_prodigy_step) — only the r/s reduction is AMDGCN.
// ============================================================================
namespace sg { namespace gfx942 { namespace native {

namespace amd = ::sg::gfx942::amdgcn;

// Stage-1 NaN/Inf gradient sanitization (mirror of sm_90 sg_sanitize_grad).
// Identity unless built with -DSG_SANITIZE_NONFINITE=1; default byte-identical.
// Applied on the APPLY path's grad read only (the r/s reduction is unchanged).
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
static constexpr int kWave = 64;   // == amd::kWave (CDNA3 wavefront width)

__device__ __forceinline__ float dabsf(float x) { return __builtin_fabsf(x); }

// Block (workgroup) prodigy r/s reduction over `[0..n)`, two AGENT atomics.
// r_i = g_i·(p_init_i − p_i)·d_prev² ;  s_i = d_prev²·|g_i|.
// blockDim.x must be a multiple of kWave and <= kWave*kWave (<= 4096); LDS holds
// one float per wavefront per quantity (<= 2*64 floats).
__device__ __forceinline__ void prodigy_rs_block(
    const float* __restrict__ g, const float* __restrict__ p,
    const float* __restrict__ p_init, float* __restrict__ r_acc,
    float* __restrict__ s_acc, float d_prev, int64_t n)
{
    // LDS/lane bookkeeping is LOCAL to the block (<= 4096 threads) → int. The
    // GLOBAL element index/stride is int64 so the reduction cannot wrap (Stage 1).
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid % kWave;
    const int waveId = tid / kWave;
    const int wpb    = static_cast<int>(blockDim.x) / kWave;
    const int64_t gtid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const float d2   = d_prev * d_prev;

    // VECTORIZED reduction body (WS5/item 6): widen the 3-buffer read to 128-bit
    // (f32x4 / dwordx4) memory access — was three scalar loads/iter. The 4 lanes
    // accumulate the IDENTICAL scalar partials (r_i = g_i·(p_init_i−p_i)·d_prev²,
    // s_i = d²·|g_i|), so the per-thread partials are bit-identical to the scalar
    // form — only the access width changes. A scalar TAIL covers the n%4 remainder.
    // NONTEMPORAL POLICY: g is the read-once grad → streaming (nontemporal); p is
    // the current param (read here, written by the apply → recurring) and p_init
    // is re-read every step → both CACHED (keep L2-resident).
    using f32x4 = amd::f32x4;
    const int64_t n4v = n & ~static_cast<int64_t>(3);   // largest mult of 4 <= n
    const auto* g4  = reinterpret_cast<const f32x4*>(g);
    const auto* p4  = reinterpret_cast<const f32x4*>(p);
    const auto* pi4 = reinterpret_cast<const f32x4*>(p_init);

    float r_local = 0.f, s_local = 0.f;
    for (int64_t q = gtid; q < (n4v >> 2); q += stride) {
        const f32x4 gv  = amd::streaming_load(&g4[q]);   // read-once grad
        const f32x4 pv  = amd::cached_load(&p4[q]);      // recurring param (cached)
        const f32x4 piv = amd::cached_load(&pi4[q]);     // p_init re-read (cached)
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            float delta = piv[l] - pv[l];
            r_local += gv[l] * delta * d2;
            s_local += d2 * dabsf(gv[l]);
        }
    }
    // SCALAR TAIL: the final n%4 elements, identical per-element partials.
    for (int64_t i = n4v + gtid; i < n; i += stride) {
        float gi = amd::streaming_load(&g[i]);
        float delta = amd::cached_load(&p_init[i]) - amd::cached_load(&p[i]);
        r_local += gi * delta * d2;
        s_local += d2 * dabsf(gi);
    }
    // two wavefront DPP reduces: every lane holds the wavefront r/s sums.
    r_local = amd::wave_reduce_add_dpp(r_local);
    s_local = amd::wave_reduce_add_dpp(s_local);

    __shared__ float lds_r[kWave];
    __shared__ float lds_s[kWave];
    if (lane == 0) { lds_r[waveId] = r_local; lds_s[waveId] = s_local; }
    amd::workgroup_barrier_release();

    // first wavefront reduces both per-wavefront slot arrays, then two atomics.
    if (waveId == 0) {
        float rs = (lane < wpb) ? lds_r[lane] : 0.f;
        float ss = (lane < wpb) ? lds_s[lane] : 0.f;
        rs = amd::wave_reduce_add_dpp(rs);
        ss = amd::wave_reduce_add_dpp(ss);
        if (lane == 0) {
            amd::atomic_add_agent_f32(r_acc, rs);
            amd::atomic_add_agent_f32(s_acc, ss);
        }
    }
}

// Bandwidth-bound reduction → high occupancy (256-thread block = 4 wavefronts;
// 8 waves/EU) to hide global-memory latency. (WS5 occupancy wire.)
extern "C" SG_KERNEL_BOUNDS(256, 8) void prodigy_gfx942_rs_reduce(
    const float* __restrict__ g, const float* __restrict__ p,
    const float* __restrict__ p_init, float* __restrict__ r_acc,
    float* __restrict__ s_acc, float d_prev, int64_t n)
{
    prodigy_rs_block(g, p, p_init, r_acc, s_acc, d_prev, n);
}

// ════════════════════════════════════════════════════════════════════════════
// §5.2  Prodigy APPLY — per-element param/state update (elementwise, vectorized
// only; NO DPP — the cross-lane r/s reduction is §5 above and stays as-is).
//
// Once the d-factor d_val = max(d_prev, r/(|s|+1e-12)) is known (host-reduced
// from the §5 r/s sums), the apply scales the gradient by d_val and runs the
// Adam EMAs + bias-corrected decoupled-weight-decay apply with d_val as the
// effective lr, plus the s-track accumulator:
//   g_s = d_val·g
//   m   = beta1·m + (1-beta1)·g_s ;  v = beta2·v + (1-beta2)·g_s²
//   st  = st + d_val·g
//   update = (m/bc1)/(sqrt(v/bc2)+eps) ;  p = p - d_val·(update + wd·p)
// Math is verbatim from launch_prodigy_step()'s ATen body (ema_update_inplace,
// ema_sq_update_inplace, s_track add_, adam_apply_inplace with lr=d_val;
// bc1/bc2 un-inverted → divide; sqrtf→__builtin_sqrtf).
//
// VECTORIZATION (WS5/B — scalar→f32x4): the fp32 path is bandwidth-bound, so the
// bulk loop is widened to 128-bit (dwordx4) memory access. grad is read-once via
// amd::streaming_load<f32x4>; exp_avg / exp_avg_sq / s_track are vector
// load/stored; the param write-once goes through amd::streaming_store<f32x4>.
// The 4 lanes each evaluate the IDENTICAL scalar expressions (uniform scalars
// d_val/beta1/beta2/bc1/bc2/eps/wd applied to all lanes), so the result is
// BIT-IDENTICAL to the scalar path — only the access width changes. A scalar
// TAIL handles the final N%4 (and the whole array on the fp32-only fallback).
// ════════════════════════════════════════════════════════════════════════════

// Per-element Prodigy apply — canonical scalar body, shared by the scalar tail
// and (replicated lane-by-lane) the f32x4 fast-path so both are bit-identical.
template <typename ParamT, typename GradT>
__device__ __forceinline__ void prodigy_apply_elem(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, float* __restrict__ s_track,
    const GradT* __restrict__ grad, int64_t i, float d_val,
    float beta1, float beta2, float eps, float wd, float bc1, float bc2)
{
    const float g   = sg_sanitize_grad(static_cast<float>(amd::streaming_load(&grad[i])));
    const float p   = static_cast<float>(param[i]);
    const float g_s = d_val * g;
    const float m   = beta1 * exp_avg[i]    + (1.0f - beta1) * g_s;
    const float v   = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g_s * g_s;
    exp_avg[i]    = m;
    exp_avg_sq[i] = v;
    s_track[i]    = s_track[i] + d_val * g;
    const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
    amd::streaming_store(&param[i],
                         static_cast<ParamT>(p - d_val * (update + wd * p)));
}

template <typename ParamT, typename GradT>
SG_KERNEL_BOUNDS(256, 8) void prodigy_gfx942_apply_kernel(
    ParamT* __restrict__ param, float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq, float* __restrict__ s_track,
    const GradT* __restrict__ grad, float d_val,
    float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int64_t N)
{
    // 64-bit grid-stride indexing (Stage 1): cast BEFORE the multiply.
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t tid    = static_cast<int64_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;

    constexpr bool kVecOk =
        sizeof(ParamT) == sizeof(float) && sizeof(GradT) == sizeof(float);
    const int64_t n4 = kVecOk ? (N >> 2) : 0;   // number of full float4 groups

    using f32x4 = amd::f32x4;
    auto* p4 = reinterpret_cast<f32x4*>(param);
    auto* m4 = reinterpret_cast<f32x4*>(exp_avg);
    auto* v4 = reinterpret_cast<f32x4*>(exp_avg_sq);
    auto* s4 = reinterpret_cast<f32x4*>(s_track);
    const auto* g4 = reinterpret_cast<const f32x4*>(grad);

    for (int64_t q = tid; q < n4; q += stride) {
        const f32x4 g  = amd::streaming_load(&g4[q]);   // 128-bit dwordx4 load
        const f32x4 p  = p4[q];
        const f32x4 ea = m4[q];
        const f32x4 ev = v4[q];
        const f32x4 st = s4[q];
        f32x4 mo, vo, so, po;
        for (int l = 0; l < 4; ++l) {
            const float gl = sg_sanitize_grad(g[l]);  // identity unless sanitize on
            const float g_s = d_val * gl;
            const float m = beta1 * ea[l] + (1.0f - beta1) * g_s;
            const float v = beta2 * ev[l] + (1.0f - beta2) * g_s * g_s;
            mo[l] = m;
            vo[l] = v;
            so[l] = st[l] + d_val * gl;
            const float update = (m / bc1) / (__builtin_sqrtf(v / bc2) + eps);
            po[l] = p[l] - d_val * (update + wd * p[l]);
        }
        m4[q] = mo;
        v4[q] = vo;
        s4[q] = so;
        amd::streaming_store(&p4[q], po);               // 128-bit dwordx4 store
    }

    // SCALAR TAIL: the final N%4 elements (and the whole array on fp32 fallback).
    for (int64_t i = (n4 << 2) + tid; i < N; i += stride) {
        prodigy_apply_elem(param, exp_avg, exp_avg_sq, s_track, grad, i, d_val,
                           beta1, beta2, eps, wd, bc1, bc2);
    }
}

// Force-instantiate the grokking dtype combo (fp32 param + fp32 grad).
template __global__ void prodigy_gfx942_apply_kernel<float, float>(
    float*, float*, float*, float*, const float*, float,
    float, float, float, float, float, float, int64_t);

}}} // namespace sg::gfx942::native
#endif  // (B) device pass

#endif  // GROKKING_KERNELS_GFX942_PRODIGY_GFX942_HIP_HPP_
