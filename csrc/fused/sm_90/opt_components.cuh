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

// SuperGrok11/15 SharpnessMetaNet (phi-net) hidden width (compile-time). Matches
// the canonical per-op kernels' SG11_H / SG15_H == 32 (supergrok11_sm90.cuh:63,
// supergrok15_sm90.cuh:51) and opt_stages_precompute.cuh::kSGMetaHidden — NOT the
// 16 of NeuralGrok's psi width. The phi-net is W1[H,2] (row-major), b1[H], W2[H],
// b2 scalar — IDENTICAL shape to the psi-net but 2 inputs/hidden unit (grad +
// sharpness) and H=32. An SG11/SG15 cell carries this weight pack in the head of a
// dedicated state slice (host-scattered each step, like NeuralGrok's psi pack):
//   phi[0           .. 2*H)            = phi_W1 [H,2]  (row-major: [j*2], [j*2+1])
//   phi[2*H         .. 3*H)            = phi_b1 [H]
//   phi[3*H         .. 4*H)            = phi_W2 [H]
//   phi[4*H]                          = phi_b2 (scalar; read on-device)
// The cell binds st.sg_phi_W1/b1/W2 from these offsets (a real bound pointer, not a
// null-deref) and reads phi_b2 on-device from sg_phi_W2[H] (a host scalar cannot
// deref a device pointer — same pattern as psi_b2).
static constexpr int kSgPhiHidden = 32;
static constexpr int kSgPhiW1Off  = 0;
static constexpr int kSgPhiB1Off  = 2 * kSgPhiHidden;
static constexpr int kSgPhiW2Off  = 3 * kSgPhiHidden;
static constexpr int kSgPhiB2Off  = 4 * kSgPhiHidden;   // scalar slot (device-side)
static constexpr int kSgPhiPackFloats = 4 * kSgPhiHidden + 1;

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
    // GrokAdamW (decoder L3-TC conversion): layer-wise β1 decay rate γ + global
    // grad-norm clip threshold. clip_coef is DEVICE-computed once per launch (the
    // global grad-norm reduction in P2.5) and stashed here; it passes through
    // rebase_state unchanged (a scalar). Defaults are the inert no-op (γ=0 ⇒ a
    // single global β1; grad_clip<=0 ⇒ no clip; clip_coef=1 ⇒ grad unscaled).
    float gamma = 0.0f, grad_clip = 0.0f, clip_coef = 1.0f;
    // Prodigy (decoder L3-TC conversion, STAGED global-d). The d-estimate's
    // INPUTS that the in-kernel reduction stage (fused_decoder_megakernel_tc's
    // Prodigy branch, between B2 and P3) consumes/produces:
    //   * param_init — the trajectory anchor p0 (flat blob parallel to params),
    //     seeded = params at step 1 by the host; r = Σ d²·<g, p0−p> needs it.
    //     rebase_state offsets it per tensor (added below).
    //   * d0 / d_coef / beta3 — the estimator scalars (host-bound via FusedScalars):
    //     d0 seeds d_prev at step 1 (the persisted d_lr slot zero-inits to 0, but
    //     eager inits _d_lr=d0=1e-6 — the cold-start the single-step gate needs);
    //     beta3=sqrt(β2) decays the persistent (r_ema,s_ema); d_coef scales the
    //     candidate d. Defaults are the inert/eager values (d_coef=1 ⇒ no scale).
    //   * prodigy_persist — device pointer to the 3 PERSISTED estimator scalars
    //     [r_ema | s_ema | d_lr] that carry across steps (host owns the buffer; the
    //     stage decays + accumulates + writes d back each step). NOT rebased.
    // st.s_track (Prodigy's `s` buffer) + st.d_factor (the reduced d the apply
    // reads) already exist above. Defaults leave every non-Prodigy cell untouched.
    const float* param_init = nullptr;
    float* prodigy_persist  = nullptr;   // [r_ema | s_ema | d_lr]  (persisted)
    float d0 = 1e-6f, d_coef = 1.0f, beta3 = 0.0f;
    // Muon (ViT L3-TC, STAGED NS): the 1D-weight AdamW group's INDEPENDENT
    // hyperparameters (eager adamw_lr/adamw_betas, muon.py:115-125). The 2D group's
    // lr/momentum ride the main lr/beta1 fields (P2.7 NS + apply); the 1D AdamW
    // tail (P3) reads aux_lr/aux_beta1/aux_beta2 instead (bc1/bc2 device-computed
    // = 1-aux_beta^step). Defaults = eager Muon adamw_* defaults; non-Muon cells
    // never read these (byte-identical).
    float aux_lr = 1e-3f, aux_beta1 = 0.9f, aux_beta2 = 0.98f;
    // LookSAM (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM 2nd backward). rho is the
    // perturbation radius; the model phase computes rho_over_norm = rho/‖g‖_global
    // ON-DEVICE (same deterministic ascending-CTA reduction as GrokAdamW's P2.5),
    // perturbs p'=p+rho_over_norm·g, runs a SECOND in-kernel fwd+bwd at p' → g_sam,
    // and writes st.sam_dir = g_sam − g (persisted in the `extra` state slice) BEFORE
    // the apply tail blends g_adj=(1−α)g+α·sam_dir. looksam_sam is the SAM-step gate
    // (1.0 on every-k SAM steps, 0.0 on intervening steps — the host computes it from
    // the optimizer's _global_step%%k cadence); when 0 the cached sam_dir is reused
    // verbatim and NO 2nd backward runs. α rides the shared st.alpha field. Defaults
    // are inert (rho=0 ⇒ no perturb; looksam_sam=0 ⇒ no 2nd pass) so every non-LookSAM
    // cell is byte-identical.
    float rho = 0.0f, looksam_sam = 0.0f;
    // SuperGrok11/15 (decoder/vit L3-TC, MODEL-COUPLED SAM 2nd backward + per-tensor
    // meta-net mu precompute). The SAME in-kernel SAM machinery as LookSAM produces
    // `sharpness = (g_sam − g)²` (NOT looksam's sam_dir = g_sam − g; the elementwise
    // write differs — supergrok11_sm90.cuh:246 / supergrok15_sm90.cuh:315), persisted
    // across steps in a state slice the cell binds. The per-tensor meta-net precompute
    // phase (P2.45) then fills st.mu = sg_rescale·phi(g, sharpness) over each tensor via
    // sg11_precompute_mu_and_gate_for_tensor<32> / sg15_precompute_mu_for_tensor<32>
    // (opt_stages_precompute.cuh), and the apply tail reads st.mu (+ st.gate). The phi
    // weights (W1[H,2]/b1[H]/W2[H]/b2) are a per-TENSOR weight set carried as device
    // pointers (the cell scatters the host meta-net pack into a state slice and binds
    // these, exactly like NeuralGrok's psi pointers). sg_phi_b2 is read on-device from
    // sg_phi_W2[kSgPhiHidden] (a host scalar cannot deref a device pointer — the cell
    // leaves it at 0.0f). sg_rescale rides FusedScalars. Defaults are inert (null phi /
    // sharpness ⇒ the precompute phase is if-constexpr'd to SG11/SG15 only, never run
    // for other cells), so every non-SG cell is byte-identical.
    const float* sharpness = nullptr;   // [total] (g_sam − g)², persisted, SAM 2nd pass
    const float* sg_phi_W1 = nullptr;   // [kSgPhiHidden*2]  (row-major [H,2])
    const float* sg_phi_b1 = nullptr;   // [kSgPhiHidden]
    const float* sg_phi_W2 = nullptr;   // [kSgPhiHidden]
    float        sg_phi_b2 = 0.0f;      // read on-device from sg_phi_W2[kSgPhiHidden]
    float        sg_rescale = 0.0f;     // SharpnessMetaNet rescale (mu = rescale·phi)
    // SG11 gate temperature: gate = sigmoid(gate_temp · cos(grad, mu)) (the per-
    // tensor cosine finalizer sg11_finalize_gate, algorithms/supergrok11.h). Inert
    // default 1.0f for every non-SG11 cell (it never reads gate_temp); SG15's gate
    // is a host scalar (st.gate) and ignores this field.
    float        gate_temp = 1.0f;
    // ── SuperGrok2 (decoder/vit L3-TC, FULL CSA/HCA/PEER/GRU meta-net as the
    //    optimizer phase). The composed megakernel runs sg2_meta_stages per tensor
    //    INSTEAD of apply_optimizer<SuperGrok2> (which is only the Adam-on-smart_grad
    //    stub). These plumb the meta-net state + HBM weight bundle + per-tensor scalar
    //    arrays the kernel reconstructs SG2Weights/SG2Scalars/SG2State from (kept as
    //    raw pointers here so opt_components.cuh need NOT include the heavy
    //    opt_stage_supergrok2.cuh / supergrok2.h). m/v reuse exp_avg/exp_avg_sq, the
    //    expert-EMA mu reuses st.mu, and the SAM side-channel (g_sam−g)² reuses
    //    st.sharpness (SG2 shares the SAM 2nd-backward machinery). perm/unsort are
    //    built IN-KERNEL into the per-CTA workspace (STAGE -1), NOT here. Inert null
    //    defaults ⇒ every non-SG2 cell is byte-identical (the SG2 phase is
    //    if-constexpr'd to OptId::SuperGrok2 only).
    float*       sg2_slow = nullptr;      // [total] grokfast slow-grad EMA (persisted)
    float*       sg2_gru_state = nullptr; // [total*gru_hidden] per-element GRU state, row-major
    // meta-net weight bundle (HBM; model-independent; staged NOT to smem — the SG2
    // bundle does not fit alongside DecTcSmem/VitTcSmem under the 48KB cap).
    const float* sg2_input_proj_W = nullptr;  const float* sg2_input_proj_b = nullptr;
    const float* sg2_csa_q_W = nullptr;  const float* sg2_csa_k_W = nullptr;
    const float* sg2_csa_v_W = nullptr;  const float* sg2_csa_out_W = nullptr;
    const float* sg2_csa_compress_w = nullptr;
    const float* sg2_csa_idx_DQ = nullptr;  const float* sg2_csa_idx_K = nullptr;
    const float* sg2_hca_q_W = nullptr;  const float* sg2_hca_k_W = nullptr;
    const float* sg2_hca_v_W = nullptr;  const float* sg2_hca_out_W = nullptr;
    const float* sg2_gru_Wz = nullptr;  const float* sg2_gru_bz = nullptr;
    const float* sg2_gru_Wr = nullptr;  const float* sg2_gru_br = nullptr;
    const float* sg2_gru_Wh = nullptr;  const float* sg2_gru_bh = nullptr;
    const float* sg2_peer_query_Ws = nullptr;
    const float* sg2_prod_keys_A = nullptr;  const float* sg2_prod_keys_B = nullptr;
    const float* sg2_expert_W1 = nullptr;  const float* sg2_expert_b1 = nullptr;
    const float* sg2_expert_W2 = nullptr;  const float* sg2_expert_b2 = nullptr;
    // per-tensor scalar arrays (device, length #tensors). These do NOT fit the
    // global FusedScalars POD (it carries only scalars), so they ride here as
    // device buffers the host fills (alpha_i/gru_decay_i/lamb_eff_i/beta1_i + the
    // per-tensor bias corrections). Shared scalars (rescale/beta2/lr/wd/eps) ride
    // the existing FusedScalars fields (lr/beta2/eps/wd) + sg2_rescale below.
    const float* sg2_alpha = nullptr;      const float* sg2_gru_decay = nullptr;
    const float* sg2_lamb_eff = nullptr;   const float* sg2_beta1 = nullptr;
    const float* sg2_bc1 = nullptr;        const float* sg2_bc2 = nullptr;
    float        sg2_rescale = 0.0f;       // expert-output scale (shared)
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
    // ── GrokAdamW append-only widening (decoder L3-TC conversion). Both default
    //    to the INERT value (0.0 ⇒ disabled), so every other cell is byte-identical
    //    (additive ABI, same contract as the original C2-gap widening above).
    float gamma     = 0.0f;   // GrokAdamW layer-wise β1 decay rate: β1_i=β1*(1-γ)^i
    float grad_clip = 0.0f;   // GrokAdamW global grad-norm clip (<=0 ⇒ no clip)
    // ── Prodigy append-only widening (decoder L3-TC conversion, STAGED global-d).
    //    The estimator scalars; defaults are eager/inert so every other cell is
    //    byte-identical. d0 seeds d_prev at step 1; beta3=sqrt(β2) decays the
    //    persisted (r_ema,s_ema); d_coef scales the candidate d (1 ⇒ no scale).
    float d0     = 1e-6f;     // Prodigy d0 (step-1 d_prev cold-start)
    float d_coef = 1.0f;      // Prodigy d_coef (candidate-d scale)
    float beta3  = 0.0f;      // Prodigy EMA decay = sqrt(beta2) (0 ⇒ no persistence)
    // ── Muon append-only widening (ViT L3-TC, STAGED NS). The eager Muon manages
    //    TWO param groups with INDEPENDENT hyperparameters: a 2D-matrix group
    //    (lr/momentum carried by the main lr/beta1 fields, consumed by the P2.7 NS
    //    + apply) and a 1D-weight AdamW group (muon.py:115-125: adamw_lr,
    //    adamw_betas, shared weight_decay). A single global scalar set cannot map
    //    the main lr/beta1 onto BOTH groups, so the 1D AdamW tail (P3) reads these
    //    aux_* fields instead. aux_bc1/bc2 are NOT carried — the muon branch
    //    computes them on-device (1-aux_beta^step), shrinking the mirror surface.
    //    Defaults mirror the eager Muon adamw_* defaults; every non-Muon caller is
    //    byte-identical (only the OptId::Muon P3 branch reads aux_*).
    float aux_lr    = 1e-3f;  // Muon 1D-group AdamW lr (eager adamw_lr)
    float aux_beta1 = 0.9f;   // Muon 1D-group AdamW betas[0] (eager adamw_betas[0])
    float aux_beta2 = 0.98f;  // Muon 1D-group AdamW betas[1] (eager adamw_betas[1])
    // ── LookSAM append-only widening (decoder/vit/mamba L3-TC, MODEL-COUPLED SAM
    //    2nd backward). rho = SAM perturbation radius; looksam_sam = the every-k
    //    SAM-step gate (1.0 ⇒ run the in-kernel perturb→2nd fwd+bwd→sam_dir=g_sam−g;
    //    0.0 ⇒ reuse the cached sam_dir, no 2nd pass). α rides the existing `alpha`
    //    field. Defaults are inert (rho=0 / looksam_sam=0), so every non-LookSAM
    //    caller is byte-identical (only OptId::LookSAM's model phase reads them).
    float rho         = 0.0f;
    float looksam_sam = 0.0f;
    // ── SuperGrok11/15 append-only widening (decoder/vit L3-TC, meta-net mu). The
    //    ONLY SG scalar that flows through here: the SharpnessMetaNet rescale (mu =
    //    rescale·phi(g, sharpness), supergrok11_sm90.cuh:130 / supergrok15_sm90.cuh:98).
    //    The phi WEIGHTS + sharpness are DEVICE pointers the cell binds directly (not
    //    scalars). rho is SHARED with LookSAM (SG's SAM perturbation radius also rides
    //    st.rho); st.alpha carries the SG meta-net strength (the apply tail reads it).
    //    Inert default (0.0 ⇒ mu=0 ⇒ the SG tail degenerates to AdamW on g) for every
    //    non-SG caller; layout stays byte-identical (only the SG P2.45 phase + tail read it).
    float sg_rescale  = 0.0f;
    // ── SuperGrok11 append-only widening (decoder/vit/mamba L3-TC, per-tensor cosine
    //    gate). gate_temp = the SharpnessMetaNet gate_temperature; the P2.45 cosine
    //    finalizer computes gate = sigmoid(gate_temp · cos(grad, mu)) (sg11_finalize_
    //    gate, algorithms/supergrok11.h). KEEP IN LOCK-STEP with the dispatch.cpp +
    //    gfx942 FusedScalars mirrors (one trailing field). Inert default 1.0f for
    //    every non-SG11 caller (SG15's gate is the host st.gate scalar; this is unread).
    float gate_temp   = 1.0f;
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
    st.gamma        = s.gamma;       // GrokAdamW layer-wise β1 decay rate
    st.grad_clip    = s.grad_clip;   // GrokAdamW global grad-norm clip threshold
    // st.clip_coef is NOT host-bound: the kernel computes it on-device (P2.5) from
    // the reduced grad's global L2 norm and st.grad_clip, then applies it in P3.
    st.d0           = s.d0;          // Prodigy d0 (step-1 d_prev cold-start)
    st.d_coef       = s.d_coef;      // Prodigy candidate-d scale
    st.beta3        = s.beta3;       // Prodigy EMA decay (sqrt(beta2))
    st.aux_lr       = s.aux_lr;      // Muon 1D-group AdamW lr (eager adamw_lr)
    st.aux_beta1    = s.aux_beta1;   // Muon 1D-group AdamW betas[0]
    st.aux_beta2    = s.aux_beta2;   // Muon 1D-group AdamW betas[1]
    st.rho          = s.rho;         // LookSAM / SuperGrok11/15 SAM perturbation radius
    st.looksam_sam  = s.looksam_sam; // LookSAM every-k SAM-step gate (1=run 2nd bwd)
    st.sg_rescale   = s.sg_rescale;  // SuperGrok11/15 SharpnessMetaNet rescale (mu=rescale·phi)
    st.gate_temp    = s.gate_temp;   // SuperGrok11 cosine-gate temperature (sigmoid(gate_temp·cos))
    // st.d_factor is NOT host-bound for the STAGED-d cell: the kernel's Prodigy
    // reduction stage computes it on-device (between B2 and P3) and stashes it.
    // st.param_init / st.prodigy_persist are device pointers bound by the cell.
}

// Dispatch to the REAL per-element optimizer step. Each branch is a genuine,
// distinct call into csrc/algorithms/<opt>.h — there is no AdamW fallback.
template <OptId Opt>
__device__ __forceinline__ void apply_optimizer(
        float* __restrict__ params, const float* __restrict__ grad,
        int64_t idx, int step, const FusedOptState& st) {
    (void)step;  // referenced only by the cold-start branches (grokfast/grokadamw)
    if constexpr (Opt == OptId::AdamW) {
        algo::adamw_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, grad,
            st.lr, st.beta1, st.beta2, st.eps, st.wd, st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::Lion) {
        algo::lion_step<float, float>(
            params, st.exp_avg, grad, st.lr, st.beta1, st.beta2, st.wd, idx);
    } else if constexpr (Opt == OptId::Grokfast) {
        // COLD-START (state-aware tail; owner baseline blocker (a)): the eager
        // Grokfast seeds the slow-grad EMA with the FIRST gradient, not zeros
        // (grokfast.py _group_cache: state["ema"] = grad0.clone()), because a
        // zero seed under-amplifies the early grokking phase and the kernel
        // applies no EMA bias correction. The persistent [m|v|extra] state cache
        // zero-inits ema=0, so at step 1 we must seed ema=grad HERE — then
        // grokfast_fused_step's e_new = alpha*g + (1-alpha)*g = g matches the
        // eager ema=grad0 exactly. Per-element on this thread's own idx (the P3
        // tail owns each element once), so no race / no barrier needed. For
        // step>1 the cache carries the real EMA forward (no reseed).
        if (step == 1) st.ema[idx] = grad[idx];
        algo::grokfast_fused_step<float, float>(
            params, st.exp_avg, st.exp_avg_sq, st.ema, grad,
            st.alpha, st.lamb, st.lr, st.beta1, st.beta2, st.eps, st.wd,
            st.bc1, st.bc2, idx);
    } else if constexpr (Opt == OptId::GrokAdamW) {
        // FAITHFUL GrokAdamW (decoder L3-TC conversion) — all THREE eager
        // mechanisms now land, so the cell is no longer a hollow pass:
        //
        //  (i)  PER-TENSOR LAYER-WISE β1 = β1·(1-γ)^layer (grokadamw.py
        //       _layer_beta1_by_id). The caller (the P3 work-steal loop, which
        //       owns the tensor index t == the flat named_parameters() layer
        //       index) has ALREADY rebased st.beta1 to β1_i AND st.bc1 to
        //       1-β1_i^step for THIS tensor before calling us (see the
        //       fused_decoder_megakernel P3 block, guarded by if constexpr
        //       GrokAdamW). So st.beta1/st.bc1 here are the per-tensor values;
        //       this branch just consumes them. bc2 stays global (β2 is not
        //       layer-wise). This is the mechanism that fails the STATE gate at
        //       step 1 if dropped (m-rel 0.895), so it is load-bearing.
        //
        //  (ii) GLOBAL GRAD-NORM CLIP to grad_clip (bindings.cpp
        //       clip_grad_norms_device_side → a GLOBAL norm over ALL tensors,
        //       one scalar clip_coef = grad_clip/(‖g‖₂+1e-6) applied to every
        //       grad when ‖g‖₂ > grad_clip). The kernel computes ‖g‖₂ + clip_coef
        //       ON-DEVICE in P2.5 (deterministic ascending-CTA reduction over the
        //       reduced grad) and stashes st.clip_coef; we scale the grad by it
        //       HERE. Eager clips IN-PLACE before the ema-seed AND the step, so we
        //       use the SAME clipped grad for both. At step 1 ‖g‖₂≈0.72<1 ⇒
        //       clip_coef=1 (inert), but it FIRES multi-step — the missing-clip
        //       2e-4 divergence is what the multi-step parity check catches.
        //
        //  (iii) ADAPTIVE α = alpha_init·exp(-κ·signal): in-context this is a
        //       genuine NO-OP. No race/gate path feeds (train_loss, val_loss) to
        //       GrokAdamW.step(), so eager α stays at alpha_init for ALL steps =
        //       the static st.alpha bound here. Faithful, not dropped (verified:
        //       fused_train_step/fused_optimizer_step/race never pass losses).
        //
        // COLD-START: eager seeds ema = the UNCLIPPED grad0, NOT the clipped grad.
        // grokadamw.py _group_cache clones state["ema"] = p.grad.clone() (the raw
        // gradient) BEFORE grokadamw_fused_step's in-place clip_grad_norms_device_
        // side (bindings.cpp:215) runs — so the cold-start seed is the UNCLIPPED
        // grad while the EMA filter input + amplification use the CLIPPED grad. The
        // prior `st.ema[idx]=gc` seeded the CLIPPED grad, which is byte-identical
        // when the clip is INERT (clip_coef==1 ⇒ gc==grad) — every currently-green
        // config — but diverges the m/ema state when the clip FIRES (the forced-fire
        // multi-step parity surfaced this: kernel ema/clipped==1.0 vs eager 1/coef).
        // Fix: seed the UNCLIPPED grad; keep the filter + g_amp on gc.
        const float gc = grad[idx] * st.clip_coef;   // (ii) global-clipped grad
        if (step == 1) st.ema[idx] = grad[idx];       // cold-start on UNCLIPPED grad
        // EMA filter + amplification (csrc/algorithms/grokadamw.h math), on gc.
        const float ema_new = st.alpha * st.ema[idx] + (1.0f - st.alpha) * gc;
        st.ema[idx] = ema_new;
        const float g_amp = gc + st.lamb * ema_new;
        // Adam moments + decoupled-WD apply: the canonical grokadamw_adam_tail
        // (bit-identical to grokadamw_step's tail), driven by the per-tensor
        // st.beta1/st.bc1 (i) and the un-inverted global st.bc2.
        float m_out, v_out, p_out;
        algo::grokadamw_adam_tail(
            g_amp, params[idx], st.exp_avg[idx], st.exp_avg_sq[idx],
            st.lr, st.beta1, st.beta2, st.eps, st.wd, st.bc1, st.bc2,
            m_out, v_out, p_out);
        st.exp_avg[idx]    = m_out;
        st.exp_avg_sq[idx] = v_out;
        params[idx]        = p_out;
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
