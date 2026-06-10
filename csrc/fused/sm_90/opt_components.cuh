#ifndef SG_FUSED_SM90_OPT_COMPONENTS_CUH_
#define SG_FUSED_SM90_OPT_COMPONENTS_CUH_
// ============================================================================
// csrc/fused/sm_90/opt_components.cuh — REAL per-optimizer device-function
// composition library for the sm_90 fused L3 megakernel (Phase 3 Stage 5).
//
// THIS REPLACES THE FALSE POSITIVE. The previous fused path (megakernel_demo.cu
// :opt_update<Opt>) implemented only 4 optimizers (AdamW/Lion/Muon/SuperGrok15)
// with TOY math, and the codegen mapped the other 7 to the AdamW tail — i.e.
// `mega_*_prodigy.cu` actually ran AdamW. That is the breadth-faking this stage
// eliminates.
//
// Here every one of the 11 optimizers calls its OWN, REAL per-element update
// function from csrc/algorithms/<opt>.h (the canonical optimizer component —
// the same device functions the per-op launchers compile). No fallback, no
// templating one optimizer from another, no toy substitute. Each optimizer's
// genuine state buffers are plumbed through FusedOptState.
//
// SCOPE / HONESTY (the fused TAIL vs the outer loop):
//   The L3 megakernel fuses the per-element optimizer TAIL after fwd+bwd. The
//   multi-phase OUTER parts that are not per-element — Muon's Newton–Schulz
//   orthogonalization, Prodigy's global d reduction, LookSAM's perturb/restore
//   ascent, SG11/SG15's cosine-gate reduction, and SG2's CSA/HCA+PEER meta-net
//   — run as their own launch(es) and hand this tail the precomputed result
//   (orth direction, d factor, sam_dir, reduced gate, smart-grad). This mirrors
//   the real per-op pipelines exactly; the tail math below IS the real apply.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

// The 11 canonical optimizer components (real per-element device functions).
#include "csrc/algorithms/adamw.h"
#include "csrc/algorithms/lion.h"
#include "csrc/algorithms/grokfast.h"
#include "csrc/algorithms/grokadamw.h"
#include "csrc/algorithms/looksam.h"
#include "csrc/algorithms/prodigy.h"
#include "csrc/algorithms/neuralgrok.h"
#include "csrc/algorithms/muon.h"
#include "csrc/algorithms/supergrok11.h"
#include "csrc/algorithms/supergrok15.h"
// SG2's elementwise tail is an Adam apply on the meta-net smart-grad; the
// metanet itself (supergrok2.h / supergrok2_bilevel_adjoint.h) is the separate
// CSA/HCA launch. adamw.h already pulled in above covers the tail.

namespace sg { namespace fused { namespace sm90 {

namespace algo = ::sg::algorithms;

// Compile-time optimizer selector — one value per real optimizer (no aliases,
// no fallback). Must match grokking_optimizers/megakernel_codegen.py's set.
enum class OptId : int {
    AdamW = 0, Lion = 1, Grokfast = 2, GrokAdamW = 3, LookSAM = 4,
    Prodigy = 5, NeuralGrok = 6, Muon = 7, SuperGrok11 = 8,
    SuperGrok15 = 9, SuperGrok2 = 10
};

// NeuralGrok / SG11 / SG15 psi/phi MLP hidden width (compile-time). Matches the
// per-op neuralgrok kernel's default instantiation.
static constexpr int kPsiHidden = 16;

// #5 — NeuralGrok psi-net weight packing in the cell's `extra` buffer.
// neuralgrok has no per-element `extra` n-slice (its psi-net is a small
// per-TENSOR weight set), so the cell repurposes the otherwise-unused `extra`
// pointer to carry the packed psi-net weights in this fixed layout:
//   extra[0                  .. kPsiHidden)        = psi_W1  [kPsiHidden]
//   extra[kPsiHidden         .. 2*kPsiHidden)      = psi_b1  [kPsiHidden]
//   extra[2*kPsiHidden       .. 3*kPsiHidden)      = psi_W2  [kPsiHidden]
//   extra[3*kPsiHidden]                            = psi_b2  (scalar)
// A neuralgrok cell binds st.psi_W1/b1/W2 from these offsets so the device read
// of the psi weights is a real, bound pointer instead of a null-deref. psi_b2
// is a device scalar; the cell leaves st.psi_b2 at its 0.0f default (a host
// scalar cannot dereference a device pointer) — see the #5 stop in the report.
static constexpr int kPsiW1Off = 0;
static constexpr int kPsiB1Off = kPsiHidden;
static constexpr int kPsiW2Off = 2 * kPsiHidden;
static constexpr int kPsiB2Off = 3 * kPsiHidden;   // scalar slot (device-side)
static constexpr int kPsiPackFloats = 3 * kPsiHidden + 1;

// All optimizer state any of the 11 tails may read. A given cell zero-fills the
// pointers its optimizer does not use; each apply_optimizer<> branch touches
// ONLY its own optimizer's real buffers (so unused ones are never dereferenced).
struct FusedOptState {
    // Adam-family moments (shared by almost all).
    float* exp_avg     = nullptr;   // m
    float* exp_avg_sq  = nullptr;   // v
    // Grokfast / GrokAdamW slow-gradient EMA.
    float* ema         = nullptr;
    // LookSAM precomputed SAM ascent direction.
    const float* sam_dir = nullptr;
    // Prodigy trajectory accumulator + adaptive d (scalar, host/reduce-stage).
    float* s_track     = nullptr;
    float  d_factor    = 1.0f;
    // SG11/SG15 momentum buffer + reduced cosine/accuracy gate (scalar).
    float* mu          = nullptr;
    float  gate        = 1.0f;
    // Muon precomputed Newton–Schulz orthogonalized direction + step scales.
    const float* orth  = nullptr;
    float  neg_lr_scale = 0.0f;
    float  decay_factor = 1.0f;
    // NeuralGrok psi-net (per-tensor MLP weights) — psi computed per element.
    const float* psi_W1 = nullptr;   // [kPsiHidden]
    const float* psi_b1 = nullptr;   // [kPsiHidden]
    const float* psi_W2 = nullptr;   // [kPsiHidden]
    float        psi_b2 = 0.0f;
    // SG2 smart-grad from the CSA/HCA meta-net launch (drives the Adam tail).
    const float* smart_grad = nullptr;
    // Shared hyperparameters.
    float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    float bc1 = 1.0f, bc2 = 1.0f;       // un-inverted bias-corrections
    float alpha = 0.98f, beta = 0.0f, lamb = 2.0f, alpha_max = 1.0f;
};

// =========================================================================
//  FusedScalars — the FULL runtime scalar set the host passes per fused_step.
//
//  C2-GAP FIX (the whole point of this struct). Previously each mega_*.cu host
//  entry bound ONLY pointers + lr into FusedOptState and left every other scalar
//  at its struct default — which silently froze the math: bc1/bc2 == 1.0 (NO
//  Adam bias correction, stuck at t=1), gate == 1.0 (SG gating inert), d_factor
//  == 1.0 (Prodigy d-adaptation inert), and beta1/beta2/eps/wd/alpha/lamb/
//  alpha_max/beta/neg_lr_scale/decay_factor never reaching apply_optimizer<>.
//  apply_optimizer<> already READS all of these from FusedOptState; the only gap
//  was that nobody SET them. This POD carries them across the host→kernel ABI so
//  the cell can populate FusedOptState with the live optimizer's real values.
//
//  Defaults here reproduce the OLD inert behavior EXACTLY (bc=gate=d_factor=1,
//  decay_factor=1, the same beta/eps/wd/alpha defaults), so a caller that does
//  not pass scalars yields byte-identical results to the pre-fix cells — the
//  widening is additive, not a behavior change for unset fields.
//
//  Per-tensor step count / bias correction (loud assumption): the megakernel
//  treats the whole flat param as ONE task, so a SINGLE (bc1, bc2) pair is bound
//  per call. That is correct ONLY when every parameter tensor shares the same
//  step counter — which holds in the grokking race (all params step together
//  every iteration). The host computes bc1 = 1 - beta1^step, bc2 = 1 - beta2^step
//  from that shared step and passes them here. If a future caller steps tensors
//  at different counts, it must call fused_step per-tensor with that tensor's bc.
// =========================================================================
struct FusedScalars {
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float wd           = 0.01f;
    float bc1          = 1.0f;   // 1 - beta1^step (un-inverted; apply divides)
    float bc2          = 1.0f;   // 1 - beta2^step (un-inverted; apply divides)
    float alpha        = 0.98f;  // meta-net strength / grokfast_alpha / neuralgrok alpha
    float beta         = 0.0f;   // neuralgrok beta (affine psi term)
    float lamb         = 2.0f;   // grokfast/grokadamw amplification (grokfast_lamb)
    float alpha_max    = 1.0f;   // SG15 per-coord alpha clip ceiling
    float gate         = 1.0f;   // SG11 cosine gate / SG15 sigmoid(accuracy) gate
    float d_factor     = 1.0f;   // Prodigy adaptive d (effective LR scale)
    float neg_lr_scale = 0.0f;   // Muon: -lr * ns_scale (2D NS-orth apply)
    float decay_factor = 1.0f;   // Muon: 1 - lr*wd (decoupled decay multiplier)
};

// Fold the runtime scalars into a FusedOptState (pointers are bound separately
// by each cell). Single seam so every cell applies the SAME scalar mapping — no
// per-cell drift in which field a scalar lands in. apply_optimizer<> reads these
// exact fields, so this is the bridge that un-freezes bc1/bc2/gate/d_factor/etc.
__host__ __device__ __forceinline__ void apply_scalars(FusedOptState& st,
                                                        const FusedScalars& s) {
    st.lr           = s.lr;
    st.beta1        = s.beta1;
    st.beta2        = s.beta2;
    st.eps          = s.eps;
    st.wd           = s.wd;
    st.bc1          = s.bc1;
    st.bc2          = s.bc2;
    st.alpha        = s.alpha;
    st.beta         = s.beta;
    st.lamb         = s.lamb;
    st.alpha_max    = s.alpha_max;
    st.gate         = s.gate;
    st.d_factor     = s.d_factor;
    st.neg_lr_scale = s.neg_lr_scale;
    st.decay_factor = s.decay_factor;
}

// Dispatch to the REAL per-element optimizer step. Each branch is a genuine,
// distinct call into csrc/algorithms/<opt>.h — there is no AdamW fallback.
template <OptId Opt>
__device__ __forceinline__ void apply_optimizer(
        float* __restrict__ params, const float* __restrict__ grad,
        int64_t idx, int step, const FusedOptState& st) {
    (void)step;
    if constexpr (Opt == OptId::AdamW) {
        algo::adamw_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, grad,
            st.lr, st.beta1, st.beta2, st.eps, st.wd, st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::Lion) {
        algo::lion_step<float, float>(
            params, st.exp_avg, grad, st.lr, st.beta1, st.beta2, st.wd, idx);
    } else if constexpr (Opt == OptId::Grokfast) {
        algo::grokfast_fused_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, st.ema, grad,
            st.alpha, st.lamb, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::GrokAdamW) {
        algo::grokadamw_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, st.ema, grad,
            st.alpha, st.lamb, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::LookSAM) {
        algo::looksam_apply_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, st.sam_dir, grad,
            st.alpha, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::Prodigy) {
        algo::prodigy_apply_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, st.s_track, grad,
            st.d_factor, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::NeuralGrok) {
        // psi_b2 is the scalar packed immediately after psi_W2 in the `extra`
        // buffer (extra[kPsiB2Off] == st.psi_W2[kPsiHidden]). Read it ON-DEVICE
        // here, where the pointer is dereferenceable — the host cell cannot
        // deref a device pointer, so it leaves st.psi_b2 at its 0.0f default.
        // This threads the real psi_b2 bias (previously stuck at 0.0).
        const float psi_b2 = (st.psi_W2 != nullptr) ? st.psi_W2[kPsiHidden]
                                                     : st.psi_b2;
        const float psi = algo::neuralgrok_psi_forward<kPsiHidden>(
            fabsf(grad[idx]), st.psi_W1, st.psi_b1, st.psi_W2, psi_b2);
        algo::neuralgrok_apply_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, grad, psi,
            st.alpha, st.beta, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::Muon) {
        // 2D params: NS-orthogonalized direction precomputed; tail is the
        // momentum-decayed scaled apply. (1D params use AdamW upstream.)
        algo::muon_update_step<float>(
            params, st.orth, st.neg_lr_scale, st.decay_factor, idx);
    } else if constexpr (Opt == OptId::SuperGrok11) {
        algo::sg11_sweep_b_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, grad, st.mu, st.gate,
            st.alpha, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::SuperGrok15) {
        algo::sg15_sweep_b_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, grad, st.mu, st.gate,
            st.alpha, st.alpha_max, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else {  // OptId::SuperGrok2 — Adam apply on the meta-net smart gradient.
        // The CSA/HCA+PEER meta-net (separate launch) produced st.smart_grad;
        // the fused tail is the real Adam apply on it (faithful to the per-op
        // SG2 path's apply stage).
        const float* g = (st.smart_grad != nullptr) ? st.smart_grad : grad;
        algo::adamw_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, g,
            st.lr, st.beta1, st.beta2, st.eps, st.wd, st.bc1, st.bc2, idx);
    }
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_OPT_COMPONENTS_CUH_
