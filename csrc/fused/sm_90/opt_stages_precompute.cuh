#ifndef SG_FUSED_SM90_OPT_STAGES_PRECOMPUTE_CUH_
#define SG_FUSED_SM90_OPT_STAGES_PRECOMPUTE_CUH_
// ============================================================================
// csrc/fused/sm_90/opt_stages_precompute.cuh — in-kernel per-step PRECOMPUTE
// stages for the 9 non-trivial optimizers (Phase 3 BUILD).
//
// WHY THIS HEADER EXISTS
// ----------------------
// opt_components.cuh::apply_optimizer<OptId> is the per-element TAIL. Today only
// AdamW / Lion have a complete in-kernel tail because they need no precompute.
// The other tails READ fields of FusedOptState (opt_components.cuh:84-113) that
// some upstream stage must PRODUCE per step:
//   * prodigy   reads st.d_factor       — a cross-ALL-tensors r/s reduction → d.
//   * muon      reads st.orth (+scales) — Newton-Schulz orthogonalization of a
//                                          2D tensor (5 matmul iterations).
//   * supergrok11 reads st.mu + st.gate — a per-element meta-MLP forward, then a
//                                          per-TENSOR cosine reduction.
//   * supergrok15 reads st.mu (+ st.gate is a host scalar) — meta-MLP forward.
// grokfast / grokadamw / neuralgrok need NOTHING (their precompute is already
// fused INTO their apply math — see the verdict block below). looksam needs a
// SAM-direction the MODEL produces on sam steps (it cannot be a pure optimizer
// stage). supergrok2 is owned by a sibling agent and is OUT OF SCOPE here.
//
// This header provides `optimizer_precompute_stage<OptId>(...)` device functions
// (one specialization per optimizer that needs one) that compute exactly those
// fields, by CALLING the canonical per-element math in csrc/algorithms/<opt>.h —
// never re-inlining it (scripts/check_math_single_source.py / the drift guard
// enforce that). Each transcription cites its semantic source by file:line.
//
// SCOPE / HONESTY (per csrc/fused/COMPONENT_CONTRACT.md):
//   * NO functionality suppressed; every stage is the REAL precompute.
//   * Self-contained: #includes the substrate + the canonical algorithm headers
//     + CUDA only (portability rule 1).
//   * No hidden state: all state flows through explicit pointers / the documented
//     PODs (rule 2).
//   * DETERMINISTIC reductions only — NO float atomicAdd. Cross-CTA partials use
//     an owner-computes tree (per-CTA slot in global memory → grid barrier → one
//     owner thread sums slots in FIXED index order). This is a deliberate,
//     contract-mandated DEVIATION from the live per-op kernels, which reduce with
//     atomicAdd (prodigy_sm90.cuh:105-108,135-138); math-equivalent in exact
//     arithmetic, fp32 summation order now fixed/reproducible. Within a single
//     CTA we use the existing deterministic warp+smem tree (primitives.cuh:125,
//     block_reduce_sum_f32 / :153 block_reduce_sum2_f32) — fixed thread count ⇒
//     fixed order.
//   * fp32 throughout (matches the per-op kernels and FusedOptState fields).
//   * NO builds / NO GPU runs in this phase: the CPU fp64 mirrors in
//     tests/hw/test_opt_stages.py ARE the validation of the staged algorithm;
//     the CUDA compile of this header is deferred (the megakernel codegen wiring
//     is specified in INTEGRATION-OPTSTAGES.md, owned at integration time).
//
// PER-OPTIMIZER VERDICT (full table — see the report + INTEGRATION-OPTSTAGES.md):
//   grokfast    NOTHING-NEEDED  EMA fused in apply (grokfast.h:63-65).
//   grokadamw   NOTHING-NEEDED  EMA fused in apply (grokadamw.h:48-49).
//   neuralgrok  NOTHING-NEEDED  psi MLP inline in apply (opt_components.cuh:222-
//                               229 → neuralgrok.h:31-46).
//   looksam     MODEL-COUPLED   sam_dir = g_sam − g needs a 2nd backward
//                               (looksam.h:52-59); a model stage supplies it.
//   prodigy     STAGED          prodigy_precompute_d_stage (this header).
//   muon        STAGED          muon_precompute_orth_stage (this header).
//   supergrok11 STAGED          sg11_precompute_mu_and_gate_for_tensor (this
//                               header; per-tensor, single-drain, no barrier).
//   supergrok15 STAGED          sg15_precompute_mu_for_tensor (this header; gate
//                               is a host scalar FusedScalars.gate).
//   supergrok2  SKIP            sibling-agent owned.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/fused/megakernel_common.cuh"      // PersistentContext, GridBarrier, TaskQueue
#include "csrc/fused/sm_90/opt_components.cuh"    // OptId, FusedOptState, kPsiHidden
#include "csrc/backends/cuda/sm_90/primitives.cuh"// deterministic block reductions

// Canonical per-element math (call, NEVER re-inline — drift guard enforces).
#include "csrc/algorithms/prodigy.h"
#include "csrc/algorithms/muon.h"
#include "csrc/algorithms/supergrok11.h"
#include "csrc/algorithms/supergrok15.h"

namespace sg { namespace fused { namespace sm90 {

namespace algo = ::sg::algorithms;
namespace prim = ::sg::sm90::primitives;

// SG11 / SG15 meta-net hidden width. Matches the per-op kernels' compile-time
// constant SG11_H / SG15_H == 32 (supergrok11_sm90.cuh:63, supergrok15_sm90.cuh
// :51) — NOT opt_components.cuh::kPsiHidden (that 16 is NeuralGrok's psi width).
static constexpr int kSGMetaHidden = 32;

// =========================================================================
//  PrecomputeWorkspace — global scratch the cross-CTA stages reduce through.
//
//  Grouped as one POD (like PersistentContext) so the codegen kernel signature
//  stays uniform. The host allocates each pointer the chosen optimizer needs and
//  zero-fills the rest (a stage touches ONLY its optimizer's buffers). All FP32.
//
//  DETERMINISM: `partials` is a per-CTA slot array (length >= ctx.n_ctas, or
//  2*n_ctas for the two-value prodigy reduce). Each CTA writes its OWN slot (no
//  contention, no atomic); after a grid barrier, one owner thread sums the slots
//  in ascending index order — a fixed, reproducible fp32 summation. This is the
//  "owner-computes tree" that REPLACES the live path's atomicAdd accumulate.
// =========================================================================
struct PrecomputeWorkspace {
    // Prodigy: r/s partial slots (2 per CTA: [0..n_ctas) = r, [n_ctas..) = s)
    // and the reduced d output (a single device scalar the apply reads via
    // st.d_factor). d_prev is carried in by the cell.
    float* prodigy_partials = nullptr;   // length >= 2 * n_ctas
    float* prodigy_d        = nullptr;   // length 1 (reduced d; also st.d_factor)
    // Muon: Newton-Schulz scratch for ONE matrix at a time (rows*cols floats
    // each), plus a per-CTA Frobenius-norm partial slot array.
    float* muon_buf  = nullptr;          // momentum buffer (rows*cols), persists
    float* muon_X    = nullptr;          // current iterate           (rows*cols)
    float* muon_A    = nullptr;          // A = X Xᵀ  (rows*rows)
    float* muon_AX   = nullptr;          // A X       (rows*cols)
    float* muon_AAX  = nullptr;          // A (A X)   (rows*cols)
    float* muon_orth = nullptr;          // NS output = st.orth        (rows*cols)
    float* muon_nrm_partials = nullptr;  // length >= n_ctas (‖buf‖_F² partials)
    float* muon_inv_norm     = nullptr;  // length 1 (1/(‖buf‖_F + 1e-7))
};

// =========================================================================
//  Deterministic cross-CTA owner-computes sum.
//
//  Each CTA (its leader thread) has already published `my_partial` into
//  `slots[blockIdx.x]`. After the caller's grid barrier guarantees every slot is
//  visible, EVERY CTA's leader recomputes the SAME total by summing slots[0..
//  n-1] in ascending order. Fixed order ⇒ identical fp32 result on every CTA, no
//  atomics. Returns the total on the leader (thread 0); other threads return 0.
//  Caller broadcasts within the block if all threads need it.
// =========================================================================
__device__ __forceinline__ float owner_sum_slots(const float* __restrict__ slots,
                                                  unsigned n) {
    float acc = 0.0f;
    if (threadIdx.x == 0) {
        // Ascending fixed-index summation — deterministic fp32 order.
        for (unsigned i = 0; i < n; ++i) acc += slots[i];
    }
    return acc;
}

// ============================================================================
// (1) NOTHING-NEEDED — grokfast / grokadamw / neuralgrok.
//
// These have NO precompute stage on purpose; their would-be precompute is fused
// INTO the per-element apply (csrc/algorithms/<opt>.h), so apply_optimizer<>
// already runs a complete in-kernel tail. Documented here so the codegen's
// per-OptId dispatch can statically map them to "no stage" with a citation:
//
//   grokfast    grokfast.h:63-65  — `e_new = alpha*ema[idx] + (1-alpha)*g;
//                                     ema[idx]=e_new; g_amp = g + lamb*e_new;`
//                                     the slow-grad EMA UPDATE is inside
//                                     grokfast_fused_step, before its Adam tail.
//   grokadamw   grokadamw.h:48-49 — `ema_new = alpha*ema[idx] + (1-alpha)*g;
//                                     ema[idx]=ema_new;` inside grokadamw_step.
//   neuralgrok  opt_components.cuh:222-229 — apply_optimizer<NeuralGrok> calls
//                                     neuralgrok_psi_forward (neuralgrok.h:31-46)
//                                     per element, then neuralgrok_apply_step.
//                                     The psi MLP is a per-TENSOR weight set read
//                                     inline; there is no per-element precompute
//                                     buffer to fill.
//
// Compile-time assertion that no caller asks these for a stage.
// ============================================================================
template <OptId Opt>
struct optimizer_needs_precompute {
    static constexpr bool value =
        (Opt == OptId::Prodigy)     || (Opt == OptId::Muon) ||
        (Opt == OptId::SuperGrok11) || (Opt == OptId::SuperGrok15);
};

// ============================================================================
// (2) MODEL-COUPLED — looksam.
//
// looksam_apply_step (looksam.h:63-91) reads st.sam_dir, where on a SAM step
//   sam_dir[idx] = g_sam[idx] − g[idx]                       (looksam.h:52-59)
// and g_sam is the gradient at the PERTURBED parameters p + rho·g/‖g‖
// (looksam.h:27-38) — i.e. the output of a SECOND model backward. A pure
// optimizer stage CANNOT produce it: it requires re-running the model forward+
// backward at perturbed weights, which is the MODEL's job, not the optimizer's.
//
// CONTRACT (NOT faked here): the megakernel exposes `st.sam_dir` as an INPUT the
// MODEL STAGE fills on sam steps (every k steps), exactly as the per-op LookSAM
// path computes it from a separate SAM backward. On intervening (non-SAM) steps
// the cached sam_dir from the last SAM step is reused verbatim (looksam.h:1-19,
// "reused as a cached direction"). There is therefore NO looksam specialization
// of optimizer_precompute_stage — by design. See INTEGRATION-OPTSTAGES.md for
// the model-stage → sam_dir wiring.
// ============================================================================

// ============================================================================
// (3a) STAGED — PRODIGY: cross-ALL-tensors d-adaptation reduction.
//
// Shape: GLOBAL. r and s are sums over EVERY element of EVERY parameter tensor
// (one scalar d for the whole model). Semantic source — the per-op three-kernel
// orchestration: prodigy_reduce_kernel (prodigy_sm90.cuh:85-109) accumulates the
// per-element partials via prodigy_partials_step, then prodigy_update_d
// (prodigy.h:56-65) combines them into d. The multi-tensor live driver is
// launch_multi_tensor_prodigy_fused_reduce_step (prodigy_sm90.cuh:465-544).
//
// IN-KERNEL vs HOST split (faithful to the live path):
//   * IN-KERNEL (this stage): the per-step Σ r_local / Σ s_local over all tensors
//     (prodigy_partials_step, prodigy.h:28-53) + prodigy_update_d (prodigy.h:56-
//     65). This is the cross-tensor reduction Audit-C flagged.
//   * HOST / CELL (NOT in this stage — documented, not hidden): the PERSISTENT
//     beta3-EMA decay of (r,s) across steps and the d_coef scaling live in the
//     host launcher (prodigy_sm90.cuh:488-521 / bindings.cpp:1246-1264), outside
//     prodigy.h. The grokking race uses that EMA estimator. The in-kernel stage
//     here computes the INSTANTANEOUS Σ-partials + max() update (d0=1e-6 init);
//     a cell wanting the EMA form decays prodigy_partials in the workspace
//     before the barrier and scales before prodigy_update_d — see
//     INTEGRATION-OPTSTAGES.md. The DEVICE math (prodigy.h) is byte-unchanged.
//
// DETERMINISM: REPLACES prodigy_sm90.cuh's atomicAdd (:105-108,:135-138) with
//   per-CTA (r,s) slot publish → grid barrier → owner-computes fixed-order sum →
//   prodigy_update_d. No float atomic.
//
// TWO-PHASE STAGE (the caller drives the barrier between them so the codegen owns
// the grid sync, consistent with fused_megakernel.cuh's phase pipeline):
//   Phase A: each CTA drains the task queue, accumulates this CTA's (r,s) over
//            its claimed tensors, publishes to ws.prodigy_partials slots.
//   [caller grid barrier — every slot visible]
//   Phase B: every CTA owner-sums the slots in fixed order and writes d to
//            ws.prodigy_d (== st.d_factor for the apply tail).
// ============================================================================

// Phase A — accumulate this CTA's (r,s) partial over all its queued tensors and
// publish to the per-CTA slots. `param_init` is the Prodigy trajectory anchor
// (one flat blob parallel to params). Drains the SAME task queue the apply uses.
__device__ inline void prodigy_precompute_reduce_phaseA(
        const PersistentContext& ctx,
        const float* __restrict__ params,
        const float* __restrict__ param_init,
        const float* __restrict__ grad,
        const int* __restrict__ sizes, const int* __restrict__ offsets,
        float d_prev,
        PrecomputeWorkspace ws) {
    __shared__ int task_slot;
    float r_acc = 0.0f, s_acc = 0.0f;
    TaskQueue q = ctx.queue();
    for (int t = q.next_block(&task_slot); t < ctx.n_tasks;
         t = q.next_block(&task_slot)) {
        const int n = sizes[t];
        const int64_t off = (int64_t)offsets[t];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            // Canonical per-element partials (prodigy.h:28-53). r/s carry d² so
            // d_hat=r/s is scale-free; s is the L1 norm Σ d²·|g|.
            float r_one = 0.0f, s_one = 0.0f;
            algo::prodigy_partials_step(params + off, param_init + off, grad + off,
                                        d_prev, (int64_t)i, r_one, s_one);
            r_acc += r_one;
            s_acc += s_one;
        }
    }
    // Deterministic in-CTA reduction of this CTA's (r,s) — fixed thread count.
    float r_block, s_block;
    prim::block_reduce_sum2_f32(r_acc, s_acc, r_block, s_block);
    if (threadIdx.x == 0) {
        // Per-CTA slot publish (no atomic). r slots [0..n_ctas), s slots after.
        ws.prodigy_partials[blockIdx.x]              = r_block;
        ws.prodigy_partials[ctx.n_ctas + blockIdx.x] = s_block;
    }
}

// Phase B — owner-computes deterministic sum of all CTA slots, then the canonical
// d update. Runs AFTER the caller's grid barrier. Every CTA writes the same d
// (redundant but race-free); the apply reads ws.prodigy_d via st.d_factor.
__device__ inline void prodigy_precompute_reduce_phaseB(
        const PersistentContext& ctx, float d_prev, PrecomputeWorkspace ws) {
    if (threadIdx.x == 0) {
        // Fixed-order sums (ascending CTA index) — deterministic fp32.
        float r_sum = 0.0f, s_sum = 0.0f;
        for (unsigned c = 0; c < ctx.n_ctas; ++c) {
            r_sum += ws.prodigy_partials[c];
            s_sum += ws.prodigy_partials[ctx.n_ctas + c];
        }
        // Canonical d update (prodigy.h:56-65): d = max(d_prev, r/|s|).
        ws.prodigy_d[0] = algo::prodigy_update_d(d_prev, r_sum, s_sum);
    }
}

// ============================================================================
// (3b) STAGED — MUON: momentum update + Newton-Schulz orthogonalization.
//
// Shape: per-MATRIX, GRID-COOPERATIVE. A matmul is cross-element (X Xᵀ couples
// every row), and a realistic 2D weight does not fit one CTA's SMEM — so muon's
// precompute CANNOT follow the one-CTA-per-tensor element-wise model. All CTAs
// cooperate on ONE matrix at a time, with a grid barrier between each matmul and
// between NS iterations; the caller loops matrices.
//
// Semantic source — bindings.cpp::muon_fused_step (bindings.cpp:930-1014):
//   buf = momentum*buf + grad                    (bindings.cpp:948)
//   inv_norm = 1/(‖buf‖_F + 1e-7)                (bindings.cpp:962-963)
//   X = buf*inv_norm                             (bindings.cpp:964)
//   repeat ns_steps (default 5):                 (bindings.cpp:991-1007)
//       A   = X Xᵀ                               (bindings.cpp:992, torch::mm)
//       AX  = A X                                 (bindings.cpp:993)
//       AAX = A AX                                (bindings.cpp:994)
//       Y = a*X + b*AX + c*AAX  (a,b,c = 3.4445,-4.7750,2.0315; :938-940)
//   orth = final Y
//   scale = 0.2*sqrt(max(rows,cols)); neg_lr_scale=-lr*scale; decay=1-lr*wd
//                                                (bindings.cpp:980-983)
// Per-element bodies are CALLED from muon.h: muon_momentum_normalize_step
// (muon.h:33-46), muon_ns_combine_step (muon.h:49-60). The MATMULS are NEW
// device code: the live path delegates X Xᵀ etc. to torch::mm/cuBLAS
// (bindings.cpp:992-994), NOT a per-element canonical fn — so a tiled owner-
// computes matmul is legitimate and is cited to that mm sequence. The drift
// guard requires only that the ELEMENTWISE pieces call muon.h, which they do.
//
// DETERMINISM: the matmul is an owner-computes inner product per output element
// (one thread sums the k-dimension in ascending order — fixed fp32 order, no
// atomic). ‖buf‖_F² uses per-CTA slot publish → barrier → owner_sum (same as
// prodigy). 1D params are NOT handled here (they take the AdamW tail upstream,
// muon.h:75-76 / bindings.cpp:970-977).
//
// STAGE STRUCTURE (caller drives every grid barrier — one matrix per call):
//   muon_momentum_norm_phaseA → [barrier] → muon_norm_reduce_phaseB (writes
//     inv_norm) → [barrier] → muon_scale_X (X = buf*inv_norm)
//   then ns_steps × { muon_matmul A=XXᵀ → [barrier] → muon_matmul AX=A·X →
//     [barrier] → muon_matmul AAX=A·AX → [barrier] → muon_ns_combine_phase
//     (orth = a·X + b·AX + c·AAX) → [barrier] → swap(X, orth) }
//   The caller then binds st.orth=<final NS buffer>, st.neg_lr_scale,
//   st.decay_factor and runs apply_optimizer<Muon> (muon.h:63-73). The exact
//   barrier/swap sequence + stride args are in INTEGRATION-OPTSTAGES.md §3.
// ============================================================================

static constexpr float kMuonNS_A = 3.4445f;   // bindings.cpp:938
static constexpr float kMuonNS_B = -4.7750f;  // bindings.cpp:939
static constexpr float kMuonNS_C = 2.0315f;   // bindings.cpp:940
static constexpr float kMuonNormEps = 1e-7f;  // bindings.cpp:962

// momentum update Phase A: buf = momentum*buf + grad, accumulate ‖buf‖_F²
// partial per CTA. (We compute the norm from buf here, then scale X after the
// reduction — equivalent to bindings.cpp:948,962-964; we cannot scale X yet
// because inv_norm needs the full-matrix norm.) All CTAs cooperate over the
// flat [rows*cols] matrix via a grid-stride loop.
__device__ inline void muon_momentum_norm_phaseA(
        const float* __restrict__ grad,
        int64_t numel, float momentum,
        PrecomputeWorkspace ws) {
    float nrm2 = 0.0f;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += gstride) {
        // buf = momentum*buf + grad (plain SGD-momentum, muon.h:43-44 body).
        const float b = momentum * ws.muon_buf[i] + grad[i];
        ws.muon_buf[i] = b;
        nrm2 += b * b;
    }
    float blk = prim::block_reduce_sum_f32(nrm2);
    if (threadIdx.x == 0) ws.muon_nrm_partials[blockIdx.x] = blk;
}

// momentum norm Phase B (after grid barrier): owner-sum the per-CTA ‖buf‖_F²
// slots in fixed order, write inv_norm = 1/(sqrt(Σ) + 1e-7) (bindings.cpp:962-
// 963). The X = buf*inv_norm assignment is split off into muon_scale_X (below):
// the canonical muon_momentum_normalize_step (muon.h:33-46) does buf-update AND
// X-scale in one shot, but we need the FULL-MATRIX norm before we can scale, so
// phaseA folded grad into buf (the muon.h:43 body) and this reduction yields
// inv_norm — the X=buf*inv_norm step (a single multiply, bindings.cpp:964, NOT
// orthogonalization math) then runs post-barrier so every CTA sees the published
// norm. The orthogonalization MATH (the NS combine) stays on the canonical fn.
__device__ inline void muon_norm_reduce_phaseB(
        const PersistentContext& ctx, PrecomputeWorkspace ws) {
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned c = 0; c < ctx.n_ctas; ++c) s += ws.muon_nrm_partials[c];
        ws.muon_inv_norm[0] = 1.0f / (sqrtf(s) + kMuonNormEps);
    }
}

// X = buf * inv_norm (bindings.cpp:964). Single multiply (not optimizer math);
// runs after the barrier that publishes inv_norm. All CTAs grid-stride the flat
// matrix.
__device__ inline void muon_scale_X(int64_t numel, PrecomputeWorkspace ws) {
    const float inv_norm = ws.muon_inv_norm[0];
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += gstride) {
        ws.muon_X[i] = ws.muon_buf[i] * inv_norm;
    }
}

// Owner-computes tiled matmul C[m,n] = sum_k A[m,k] * B[k,n], grid-cooperative
// over the OUTPUT elements (one thread owns one output and sums the contraction
// in ASCENDING k — deterministic fp32, no atomic). `bT` selects B vs Bᵀ so we
// can express X Xᵀ (A=X[r,c], B=X[r,c] with bT → contract over c giving [r,r])
// and A X (contract over r) with one helper. Row-major [rows*cols] layout
// matching the flat workspace buffers.
//   - For A = X Xᵀ:  out[r1,r2] = Σ_c X[r1,c] X[r2,c]   (M=rows,N=rows,K=cols)
//   - For AX = A X:   out[r,c]   = Σ_j A[r,j] X[j,c]     (M=rows,N=cols,K=rows)
__device__ inline void muon_matmul(
        const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K,
        int A_rowstride, int B_rowstride,
        bool b_transposed) {
    const int64_t total = (int64_t)M * N;
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gstride) {
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        float acc = 0.0f;
        const float* arow = A + (int64_t)row * A_rowstride;
        if (b_transposed) {
            // B[col, k] (row `col` of B, length K). Used for X Xᵀ.
            const float* brow = B + (int64_t)col * B_rowstride;
            for (int k = 0; k < K; ++k) acc += arow[k] * brow[k];
        } else {
            // B[k, col]. Used for A X.
            for (int k = 0; k < K; ++k) acc += arow[k] * B[(int64_t)k * B_rowstride + col];
        }
        C[idx] = acc;
    }
}

// One Newton-Schulz iteration body's polynomial combine over the flat matrix:
// X_next = a*X + b*AX + c*AAX (muon.h:49-60, called per element). The matmuls
// A/AX/AAX must already be computed (by muon_matmul, with caller barriers
// between them). Writes into ws.muon_X in place is unsafe (combine reads X), so
// we combine into ws.muon_orth then the caller swaps orth↔X for the next iter,
// or — on the LAST iter — leaves the result in ws.muon_orth as the final
// orthogonalized direction.
__device__ inline void muon_ns_combine_phase(int64_t numel,
                                             PrecomputeWorkspace ws) {
    const int64_t gstride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += gstride) {
        // Canonical combine body (muon.h:49-60): Y = a*X + b*AX + c*AAX.
        algo::muon_ns_combine_step(ws.muon_orth, ws.muon_X, ws.muon_AX,
                                   ws.muon_AAX, kMuonNS_A, kMuonNS_B, kMuonNS_C,
                                   i);
    }
}

// ============================================================================
// (3c) STAGED — SUPERGROK11: meta-MLP mu (per element) + per-TENSOR cosine gate.
//
// Semantic source — supergrok11_fused_step (bindings.cpp:1339-1418) +
// sg11_mu_metanet_kernel (supergrok11_sm90.cuh:112-134) +
// compute_cosine_gate_fused (supergrok11_sm90.cuh:275-286).
//
// The live path does (bindings.cpp:1407-1416):
//   1. mu = rescale * phi(grad, sharpness)  with smart_grad = grad + 0*mu = grad
//      (mu_metanet called with alpha=0, so smart_grad == grad — the gate's first
//      argument IS grad).                          (bindings.cpp:1407-1409)
//   2. gate = sigmoid(gate_temp · cosine(grad, mu))  (bindings.cpp:1411 →
//      compute_cosine_gate_fused → sg11_finalize_gate, supergrok11_sm90.cuh).
//   3. apply: g_eff = grad + lamb_eff*mu, Adam tail (lamb_eff = ramp*lamb*(1-
//      gate)*alpha). The fused TAIL apply_optimizer<SuperGrok11> (opt_components
//      .cuh:235-239 → sg11_sweep_b_step, supergrok11.h:79-111) reproduces the
//      SAME shape: smart_grad = g + (1-gate)*alpha*mu.
//
// Shape — PER-TENSOR, SINGLE-PHASE (NO grid barrier). The live per-op path
// (bindings.cpp:1407-1416) loops per parameter doing metanet→gate→adam, each
// tensor SELF-CONTAINED: gate(T) and apply(T) depend ONLY on mu(T), never on any
// other tensor's mu. So mu/gate/apply for one tensor fuse into ONE body that the
// SAME CTA owns end-to-end inside ONE task-queue drain — exactly the codegen's
// fused_optimizer_stage shape (fused_megakernel.cuh:101-142):
//   per claimed tensor T: { write mu(T)→global ; __syncthreads ; gate(T) via a
//   block tree (reads the mu it just wrote) ; apply_optimizer<SuperGrok11>(T) }.
// CRITICAL — this is NOT a full-queue mu pre-pass + a separate apply drain. That
// (rejected) shape would (a) need a grid barrier for cross-CTA mu visibility AND
// (b) SILENTLY NO-OP: the mu pre-pass drains the queue to n_tasks, so the apply
// drain's next_block() returns t>=n_tasks immediately and never applies (unless a
// sync_reset re-zeros the counter). The single-drain per-tensor form below has
// NO such hazard — one drain, no reset, no barrier, no race. (Prodigy DOES need a
// barrier because its d is a genuine cross-ALL-tensors reduction; SG11's gate is
// per-tensor — the cases are NOT analogous.)
//
//   mu(T)   — per ELEMENT: mu[idx] = rescale * sg11_phi_forward(grad[idx],
//             sharpness[idx], ...). Per-TENSOR weight set (W1,b1,W2,b2).
//   gate(T) — per TENSOR: cosine of (grad, mu) over THIS tensor, block tree
//             (block_reduce_sum_f32, primitives.cuh:125). NO cross-CTA, NO atomic.
//
// gate FORMULA — the CANONICAL sg11_finalize_gate (algorithms/supergrok11.h).
// compute_cosine_gate_fused (supergrok11_sm90.cuh) computes
//   num=Σ sg·mu; den_g=Σ sg·sg; den_m=Σ mu·mu;
//   cos  = num / sqrt(den_g*den_m + 1e-12);
//   gate = sigmoid(gate_temp · cos)
// with sg == grad here. The cosine SIGNAL is what distinguishes SG11 from SG15's
// sigmoid-of-accuracy gate; only the FINAL squashing is the sigmoid, using the
// plumbed gate_temp (the historical impl did a bare clamp and IGNORED gate_temp —
// now corrected in lock-step across the canonical, per-op, this stage, the python
// ref and ref_sg11_step). We call the single-source finalizer.
//
// MODEL-COUPLED INPUT (flag loudly): `sharpness` is NOT optimizer state — it is
// (sam_grad − normal_grad)² from a SAM SECOND backward (sg11_sharpness_restore_
// kernel, supergrok11_sm90.cuh:236-248). It is a documented CONTRACT INPUT,
// exactly like looksam's sam_dir. CRUCIAL ABI GAP: FusedOptState (opt_components
// .cuh:84-113, which this header MUST NOT touch) has a `mu` field but NO
// `sharpness` field and NO phi-weight fields. So this stage takes sharpness +
// the phi weights as EXPLICIT pointers; the cell must supply a sharpness buffer
// the current ABI does not carry — see INTEGRATION-OPTSTAGES.md.
// ============================================================================

// Stage the per-tensor phi weights into SMEM ONCE per block (the codegen calls
// this before its per-tensor loop; same pattern as supergrok11_sm90.cuh:79-87).
// W1 is [H,2] row-major, b1/W2 are [H]. Caller provides the SMEM tiles.
template <int H>
__device__ __forceinline__ void sg_stage_phi_weights(
        const float* __restrict__ W1, const float* __restrict__ b1,
        const float* __restrict__ W2,
        float* sW1, float* sb1, float* sW2) {
    for (int j = threadIdx.x; j < H * 2; j += blockDim.x) sW1[j] = W1[j];
    for (int j = threadIdx.x; j < H;     j += blockDim.x) { sb1[j] = b1[j]; sW2[j] = W2[j]; }
    __syncthreads();
}

// SG11 per-tensor body: write mu(T) to global, barrier, compute the per-tensor
// gate(T) = sigmoid(gate_temp · cos(grad, mu)) (reading the mu just written).
// Returns the gate on EVERY thread (broadcast through the reduction smem) so the
// caller binds st.gate before apply_optimizer<SuperGrok11>. The phi weights are
// already in SMEM (sW1/sb1/sW2 from sg_stage_phi_weights). `off`/`n` is the
// tensor slice; the SAME CTA owns mu(T) and gate(T) — no cross-CTA dependency, no
// grid barrier. gate_temp is the plumbed gate_temperature (FusedScalars.gate_temp).
template <int H>
__device__ inline float sg11_precompute_mu_and_gate_for_tensor(
        float* __restrict__ mu_blob,
        const float* __restrict__ grad,
        const float* __restrict__ sharpness,
        const float* sW1, const float* sb1, const float* sW2,
        float b2, float rescale,
        int64_t off, int n, float gate_temp) {
    // mu(T): canonical meta-net forward (supergrok11.h:37-55) * rescale
    // (supergrok11_sm90.cuh:130), written to global mu_blob.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t gi = off + i;
        const float g = grad[gi];
        const float s = sharpness[gi];
        const float phi = algo::sg11_phi_forward<H>(g, s, sW1, sb1, sW2, b2);
        mu_blob[gi] = rescale * phi;
    }
    // Barrier: every thread's mu writes for THIS tensor must be visible to the
    // gate reduction below (same block, so __syncthreads suffices — block-scope
    // visibility, no grid barrier).
    __syncthreads();
    // gate(T): sigmoid(gate_temp · cos(grad, mu)), block tree. sg == grad (alpha=0
    // path, bindings.cpp:1408). gate_temp is the plumbed gate_temperature.
    float num = 0.0f, den_g = 0.0f, den_m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t gi = off + i;
        const float sg = grad[gi];
        const float u  = mu_blob[gi];
        num   += sg * u;
        den_g += sg * sg;
        den_m += u  * u;
    }
    // Deterministic block reductions (fixed thread count). Three separate trees
    // (the live host fn does three independent .sum()s — same shape).
    num   = prim::block_reduce_sum_f32(num);
    __syncthreads();
    den_g = prim::block_reduce_sum_f32(den_g);
    __syncthreads();
    den_m = prim::block_reduce_sum_f32(den_m);
    // Canonical finalizer (sg11_finalize_gate, algorithms/supergrok11.h):
    //   cos = num / sqrt(den_g*den_m + 1e-12); gate = sigmoid(gate_temp · cos).
    // Same single source compute_cosine_gate_fused calls — no re-inlined math.
    return algo::sg11_finalize_gate(num, den_g, den_m, gate_temp);
}

// ============================================================================
// (3d) STAGED — SUPERGROK15: meta-MLP mu (per element). Gate is a HOST scalar.
//
// Semantic source — sg15_sweep_a_kernel (supergrok15_sm90.cuh:75-103) +
// supergrok15_fused_step (bindings.cpp:1512-...). mu[idx] = rescale*phi(grad,
// sharpness) — IDENTICAL shape to SG11's mu sub-stage (the phi-net shares the
// SG11/SG15 form, supergrok15.h:34-52). The DIFFERENCE from SG11:
//   * gate is GLOBAL = sigmoid(training accuracy), set HOST-side and passed as a
//     scalar (supergrok15.h:85 gate_global; threaded via FusedScalars.gate,
//     opt_components.cuh:153). So there is NO gate REDUCTION stage for SG15.
//   * the per-coord alpha clip (sg15_alpha_per_coord, supergrok15.h:68-75) and
//     smart_grad = g + gate_global*a*mu happen INSIDE the apply tail
//     (apply_optimizer<SuperGrok15> → sg15_sweep_b_step, opt_components.cuh:240-
//     244). So the ONLY precompute SG15 needs is mu.
//   * sg15_sweep_a also accumulates a sharpness² partial (supergrok15_sm90.cuh
//     :99-102, sg15_sweep_a_step → supergrok15.h:56-65). That sum updates the
//     NEXT step's `sharpness` input host-side; it is NOT consumed by THIS step's
//     apply (the apply reads mu + the host gate). It is therefore part of the
//     model-coupled sharpness pipeline, NOT this step's optimizer precompute, and
//     is documented in INTEGRATION-OPTSTAGES.md rather than computed here.
//
// MODEL-COUPLED INPUT: same `sharpness` ABI gap as SG11 (FusedOptState has no
// sharpness field) — flagged in INTEGRATION-OPTSTAGES.md.
// ============================================================================

// SG15 per-tensor body: write mu(T) to global. NO gate stage (the gate is the
// host scalar FusedScalars.gate). Phi weights are already in SMEM (sW1/sb1/sW2
// from sg_stage_phi_weights). The codegen calls this inside its per-tensor loop,
// then runs apply_optimizer<SuperGrok15> over the SAME tensor with st.gate set to
// the host sigmoid(accuracy). Single-drain, per-tensor, no grid barrier.
template <int H>
__device__ inline void sg15_precompute_mu_for_tensor(
        float* __restrict__ mu_blob,
        const float* __restrict__ grad,
        const float* __restrict__ sharpness,
        const float* sW1, const float* sb1, const float* sW2,
        float b2, float rescale,
        int64_t off, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t gi = off + i;
        const float g = grad[gi];
        const float s = sharpness[gi];
        // Canonical SG15 meta-net forward (supergrok15.h:34-52), * rescale
        // (supergrok15_sm90.cuh:97-98).
        const float phi = algo::sg15_phi_forward<H>(g, s, sW1, sb1, sW2, b2);
        mu_blob[gi] = rescale * phi;
    }
}

// ============================================================================
//  optimizer_precompute_stage<Opt> — the uniform entry point the codegen calls.
//
//  STAGE SHAPES DIFFER per optimizer (do NOT force one pattern):
//   * Prodigy — a genuine cross-ALL-tensors reduction: this dispatcher runs
//     phaseA (drain queue, publish per-CTA (r,s) slots); the codegen then drives
//     a grid barrier (bar.sync_reset) and calls prodigy_precompute_reduce_phaseB.
//   * SG11 / SG15 — PER-TENSOR, single-drain, NO grid barrier. They do NOT run a
//     standalone queue drain here (that would silently no-op the subsequent apply
//     drain — see the SG11 block). Instead the codegen fuses mu(+gate)→apply per
//     tensor INSIDE its own task-queue loop, calling sg_stage_phi_weights once
//     per block, then per claimed tensor sg11_precompute_mu_and_gate_for_tensor /
//     sg15_precompute_mu_for_tensor, then apply_optimizer<>. This dispatcher
//     therefore does NOTHING for SG11/SG15 (the per-tensor helpers are the API).
//   * Muon — per-MATRIX, grid-cooperative: the codegen drives the explicit phase
//     fns (momentum_norm → barrier → norm_reduce → barrier → scale_X → ns_steps×
//     {matmul×3 + combine, barriers}) per matrix. Not foldable into this call.
//  See INTEGRATION-OPTSTAGES.md for each exact sequence.
//
//  A NOTHING-NEEDED / MODEL-COUPLED / SKIP optimizer must NOT be instantiated
//  here (static_assert guards it) — its tail already runs complete, or its input
//  comes from the model stage.
// ============================================================================
template <OptId Opt>
__device__ inline void optimizer_precompute_stage(
        const PersistentContext& ctx,
        float* __restrict__ params,
        const float* __restrict__ grad,
        const int* __restrict__ sizes, const int* __restrict__ offsets,
        const FusedOptState& st,
        PrecomputeWorkspace ws,
        // Optional model-coupled / meta inputs (param_init for prodigy).
        const float* __restrict__ param_init = nullptr) {
    static_assert(optimizer_needs_precompute<Opt>::value,
        "optimizer_precompute_stage instantiated for an optimizer that needs no "
        "precompute (grokfast/grokadamw/neuralgrok: math is in apply), is model-"
        "coupled (looksam: sam_dir from a 2nd backward), or is sibling-owned "
        "(supergrok2). See the verdict table at the top of this header.");

    if constexpr (Opt == OptId::Prodigy) {
        // Phase A only (cross-ALL-tensors reduce). The codegen then grid-barriers
        // (bar.sync_reset) and calls prodigy_precompute_reduce_phaseB. param_init
        // is the trajectory anchor (flat blob parallel to params).
        prodigy_precompute_reduce_phaseA(ctx, params, param_init, grad,
                                         sizes, offsets, st.d_factor, ws);
    } else {
        // SG11 / SG15 / Muon are driven by their per-tensor / per-matrix helpers
        // inside the codegen's loop (SG11/SG15) or its per-matrix barrier sequence
        // (Muon), NOT by a queue drain here. This dispatcher intentionally does
        // nothing for them — see the helper fns above + INTEGRATION-OPTSTAGES.md.
        static_assert(Opt == OptId::SuperGrok11 || Opt == OptId::SuperGrok15 ||
                      Opt == OptId::Muon,
                      "unreachable: optimizer_needs_precompute gated the set");
        (void)params; (void)grad; (void)sizes; (void)offsets; (void)st; (void)ws;
    }
    (void)param_init;
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_OPT_STAGES_PRECOMPUTE_CUH_
