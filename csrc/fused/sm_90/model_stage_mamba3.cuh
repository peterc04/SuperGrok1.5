#ifndef SG_FUSED_SM90_MODEL_STAGE_MAMBA3_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_MAMBA3_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_mamba3.cuh — PHASE 2 of the TRUE L3 fused
// megakernel: the REAL **Mamba-3** (SISO) forward + backward as in-kernel
// stages of the persistent megakernel. ONE HEADER PER MODEL (COMPONENT_CONTRACT).
//
// HONESTY: this is the genuine Mamba-3 architecture (arXiv 2603.15569, ICLR
// 2026), transcribed LINE-FOR-LINE from the verified PyTorch oracle in
// tests/hw/mamba3_oracle.py + grokking_optimizers/mamba3_block.py, which is
// asserted bit-identical (fp64, ~4e-15 rel) to torch.autograd for the loss and
// EVERY one of the 45 parameter gradients — INCLUDING the complex
// exponential-trapezoidal selective-scan backward (the reverse-time recurrence
// with the width-2 beta-coupling, the per-head coefficient Jacobian fold, and
// the per-step 2x2 rotations). There is NO placeholder math on this path.
//
// ARCHITECTURE (exact; cites MAMBA3_REFERENCE.md sections):
//   Full Llama-style stack (Sec 3.4): tok+pos embed -> nl x Mamba3Block ->
//   final RMSNorm -> Linear head on the LAST token; CE loss.
//   Each Mamba3Block (one Llama layer):
//       h = h + Mamba3Mixer( RMSNorm_mix(h) )      # mixer residual OFF
//       h = h + SwiGLU_MLP ( RMSNorm_mlp(h) )      # down(SiLU(gate)*up), no bias
//   Mamba3Layer (mixer), forward:
//       xz = in_proj(x); x_main,z = chunk(2)       # NO conv1d, NO SSM-input SiLU
//       (dt_lr,A_mod,theta,u_lam,Br,Bi,Cr,Ci) = x_proj(x_main).split
//       dt   = softplus(dt_proj(dt_lr)+b)          # PER HEAD
//       A    = -softplus(A_mod + exp(A_log))       # PER HEAD scalar real part
//       lam  = sigmoid(u_lam)                      # PER HEAD trapezoid gate
//       Bbar = (BCNorm(Br)+B_bias, BCNorm(Bi)+Bhat_bias)   # head-shared, per Nc
//       Cbar = (BCNorm(Cr)+C_bias, -(BCNorm(Ci)+Chat_bias))
//       y    = mamba3_scan(x_main, dt, A, phi=dt*theta, lam, Bbar, Cbar)  (Eq 25)
//       y    = (y + x_main*D) * SiLU(z)            # D per-channel skip, gated out
//       out  = out_proj(y)
//   Layout (MAMBA3_REFERENCE.md sec 1, RESOLVED):
//     PER HEAD     : dt, A_real, lambda                 [n_heads]
//     HEAD-SHARED  : theta, B, Bhat, C, Chat            [N_c]
//     PER CHANNEL  : x (SSM input), D skip, gate z       [d_inner]
//   State h_t[b,h,p,c,:] is a 2-vector per complex coord c; rotation angle
//   phi[h,c] = dt[h]*theta[c] (per-head per-coord). y = Cbar^T h (real part).
//
// THE SCAN RECURRENCE (Eq 25, the heart of the kernel):
//     v_t = Bbar_t (.) x_t                              # 2-vector (Br*x, Bi*x)
//     h_t = alpha_t*(R_t@h_{t-1}) + beta_t*(R_t@v_{t-1}) + gamma_t*v_t
//     y_t = sum_{c,d} Cbar_t[c,d] * h_t[b,h,p,c,d]
//   alpha=exp(dt*A), gamma=lam*dt, beta=(1-lam)*dt*alpha (per head).
//   R(phi)@(w0,w1) = (cos*w0 - sin*w1, sin*w0 + cos*w1).
//
// THE SCAN BACKWARD (reverse time, MAMBA3_REFERENCE.md sec 6.2). Carry TWO
// adjoints per (b,h,p,c): gh=dL/dh_t and gv=dL/dv_t (the WIDTH-2 coupling —
// v_{t-1} feeds step t-1's gamma-term AND step t's beta-term). See mb_scan_bwd.
//
// PARALLELIZATION (Option B, batch-parallel, deterministic — IDENTICAL to the
// decoder PHASE-1 design): each CTA owns a FIXED contiguous batch slice; the
// whole CTA cooperates on ONE sample at a time. THE SCAN EXPLOIT (seq=8): one
// thread owns SSM channel j in [0,d_inner); within its head (h=j/head_dim) it
// holds the complex state h[Nc][2] in REGISTERS and unrolls t=0..7. The backward
// recomputes the forward keeping h_hist[seq+1][Nc][2] AND v_hist[seq+1][Nc][2]
// in registers (seq=8 trivial — NO checkpoint). The per-head coefficient grads
// (dalpha/dbeta/dgamma/dphi) are reduced over the head_dim channels in a head;
// the head-shared dBbar/dCbar/dtheta are reduced over ALL channels (ascending
// lane/warp -> deterministic). One grid barrier, deterministic cross-CTA reduce,
// another barrier, apply_optimizer<Opt> (composition in fused_mamba_megakernel.cuh).
//
// SMEM BUDGET PER CTA: the per-sample MambaSampleSmem caches BOTH layers'
// forward activations + the SwiGLU/RMSNorm caches + the complex B/C streams; the
// scan complex STATE is per-thread registers (the seq=8 exploit), NOT smem. The
// CTA smem is DYNAMIC (cudaFuncSetAttribute opt-in) — kMambaSmemBytes
// (mamba3_layout.cuh) is the static_assert-pinned sizeof(MambaSampleSmem).
//
// fp32 compute is the correctness baseline. The TC variant
// (model_stage_mamba_tc.cuh) reuses mb:: dims, MambaWeights/Grad, the
// MambaSampleSmem::LayerAct cache type, and the scalar scan/RMSNorm device fns
// VERBATIM; only the 7 projection GEMMs run on wgmma there.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/mamba3_layout.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

// ── Mamba-3 compile-time shape constants (mirror mamba3_block.py and
//    csrc/fused/sm_90/mamba3_layout.cuh; static_asserts there guard the count +
//    total against named_parameters()). ───────────────────────────────────────
namespace mb {
constexpr int kVocab  = SG_MB_VOCAB;    // 99  (tok embedding rows)
constexpr int kPHead  = SG_MB_PHEAD;    // 97  (head width; DISTINCT from vocab)
constexpr int kD      = SG_MB_D;        // 128 (d_model)
constexpr int kLayers = SG_MB_LAYERS;   // 2
constexpr int kSeq    = SG_MB_SEQ;      // 8
constexpr int kDInner = SG_MB_DINNER;   // 256 (d*2)
constexpr int kState  = SG_MB_STATE;    // 128 (REAL state N)
constexpr int kStateC = SG_MB_STATEC;   // 64  (complex dim Nc = N/2)
constexpr int kHeadDim = SG_MB_HEADDIM; // 64  (P)
constexpr int kNHeads = SG_MB_NHEADS;   // 4   (d_inner/head_dim)
constexpr int kDtRank = SG_MB_DTRANK;   // 8
constexpr int kXProj  = SG_MB_XPROJ;    // 336 (x_proj out width)
constexpr int kDff    = SG_MB_DFF;      // 256 (SwiGLU inner)
constexpr float kRmsEps = 1e-5f;        // RMSNorm eps (mamba3_block default)
// x_proj output split offsets: [dt_lr | A_mod | theta | u_lam | Br | Bi | Cr | Ci]
constexpr int kOffDtLr = 0;
constexpr int kOffAmod = kOffDtLr + kDtRank;        // + dt_rank
constexpr int kOffThet = kOffAmod + kNHeads;        // + n_heads
constexpr int kOffULam = kOffThet + kStateC;        // + Nc
constexpr int kOffBr   = kOffULam + kNHeads;        // + n_heads
constexpr int kOffBi   = kOffBr   + kStateC;        // + Nc
constexpr int kOffCr   = kOffBi   + kStateC;        // + Nc
constexpr int kOffCi   = kOffCr   + kStateC;        // + Nc
static_assert(kOffCi + kStateC == kXProj, "mamba3: x_proj split must sum to kXProj");
static_assert(kDInner % kHeadDim == 0 || kHeadDim == kDInner,
              "mamba3: d_inner must be a multiple of head_dim (or single-head fallback)");
}  // namespace mb

// ── Elementwise activations + derivatives (the real F.silu / F.softplus). ─────
__device__ __forceinline__ float mb_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float mb_sigmoid_grad(float x) {  // s*(1-s)
    const float s = mb_sigmoid(x);
    return s * (1.0f - s);
}
__device__ __forceinline__ float mb_silu(float x) {
    return x * mb_sigmoid(x);
}
__device__ __forceinline__ float mb_silu_grad(float x) {
    const float s = mb_sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}
__device__ __forceinline__ float mb_softplus(float x) {
    // log1p(exp(x)), guarded for large x (matches torch threshold=20).
    return (x > 20.0f) ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float mb_softplus_grad(float x) {  // sigmoid(x)
    return mb_sigmoid(x);
}

// ── Per-CTA scratch for one sample. POD held in smem (one instance per CTA).
//    BOTH layers' forward activations are cached. The scan complex STATE is NOT
//    here (per-thread registers — the seq=8 exploit). ──────────────────────────
struct MambaSampleSmem {
    // Cross-block residual stream: the INPUT to each block (= residual x), and
    // the final-block output feeding the head norm.
    float layer_in[mb::kLayers][mb::kSeq][mb::kD];   // block input (residual)
    float final_in[mb::kSeq][mb::kD];                // final-block output -> head
    // Per-block cached forward activations (both blocks): the values the backward
    // reads. The block-level "h1" (= x + mixer_out) is cached so mlp_norm's input
    // and the mixer-residual reconstruction are available in the backward.
    struct LayerAct {
        // --- mixer pre-norm (RMSNorm_mix) cache ---
        float mixn_xhat[mb::kSeq][mb::kD];     // xhat = x*r
        float mixn_r[mb::kSeq];                // rsqrt(mean(x^2)+eps)
        // --- mixer internals ---
        float x_in[mb::kSeq][mb::kDInner];     // in_proj first half (= x_main, scan/x_proj/D input)
        float z[mb::kSeq][mb::kDInner];        // in_proj second half (gate)
        float dt_lr[mb::kSeq][mb::kDtRank];    // dt_proj input (x_proj slice)
        float dt_pre[mb::kSeq][mb::kNHeads];   // dt_proj out + bias (PRE-softplus, per head)
        float A_mod[mb::kSeq][mb::kNHeads];    // x_proj A_mod slice (per head)
        float u_lam[mb::kSeq][mb::kNHeads];    // x_proj lambda logit (per head)
        float theta[mb::kSeq][mb::kStateC];    // x_proj theta slice (head-shared)
        // BCNorm caches: pre-norm raw streams + their rms recip. Streams: Br,Bi,Cr,Ci.
        float Br[mb::kSeq][mb::kStateC];       // pre-BCNorm B real
        float Bi[mb::kSeq][mb::kStateC];       // pre-BCNorm B imag (Bhat)
        float Cr[mb::kSeq][mb::kStateC];       // pre-BCNorm C real
        float Ci[mb::kSeq][mb::kStateC];       // pre-BCNorm C imag (Chat)
        float Br_r[mb::kSeq]; float Bi_r[mb::kSeq];   // BCNorm rms recips
        float Cr_r[mb::kSeq]; float Ci_r[mb::kSeq];
        // post-norm+bias 2-vector streams (the scan reads these):
        float Bbar[mb::kSeq][mb::kStateC][2];  // (Br2, Bi2)
        float Cbar[mb::kSeq][mb::kStateC][2];  // (Cr2, -Ci2)
        float y_scan[mb::kSeq][mb::kDInner];   // selective-scan output (per channel)
        // --- block inter-residual + mlp pre-norm (RMSNorm_mlp) cache ---
        float h1[mb::kSeq][mb::kD];            // x + mixer_out (mlp_norm input)
        float mlpn_xhat[mb::kSeq][mb::kD];     // mlp_norm xhat
        float mlpn_r[mb::kSeq];                // mlp_norm rms recip
        // --- SwiGLU internals ---
        float g_pre[mb::kSeq][mb::kDff];       // gate_proj out (pre-SiLU)
        float u_mlp[mb::kSeq][mb::kDff];       // up_proj out
    } act[mb::kLayers];
    // Final-norm caches (last position used) + head logits.
    float fn_xhat[mb::kSeq][mb::kD];
    float fn_r[mb::kSeq];
    float logits[mb::kPHead];
    // Backward adjoint scratch (reused across the per-block chain).
    float dh[mb::kSeq][mb::kD];                // running grad wrt block output
    float dr[mb::kSeq][mb::kD];                // RMSNorm-bwd output / residual scratch
    float adj_a[mb::kSeq][mb::kDInner];        // d_inner-wide scratch
    float adj_b[mb::kSeq][mb::kDInner];        // d_inner-wide scratch
    float adj_c[mb::kSeq][mb::kDInner];        // d_inner-wide scratch
    float wff_a[mb::kSeq][mb::kDff];           // d_ff-wide scratch (MLP)
    float wff_b[mb::kSeq][mb::kDff];           // d_ff-wide scratch (MLP)
    float xproj[mb::kSeq][mb::kXProj];         // x_proj fwd out / dx_proj staging
    // scan-bwd cross-channel reduce targets (head-shared), per timestep:
    float dBbar[mb::kSeq][mb::kStateC][2];
    float dCbar[mb::kSeq][mb::kStateC][2];
    float dtheta[mb::kSeq][mb::kStateC];
    // Reductions across threads. 256 threads = 8 warps; the two-level block
    // reduce uses red[warp] (warp<=7) then red[0]. 64 slots = ample headroom
    // (kept small so the CTA dynamic smem stays under the H100 per-block opt-in
    // cap of 227 KB — the full 256-slot red pushed sizeof past it by ~0.5 KB).
    float red[64];
};
// SAFETY: the launcher opts into kMambaSmemBytes of DYNAMIC smem (mamba3_layout.cuh).
// PIN the layout constant to the actual struct here so a field added without
// updating kMambaSmemFloats fails the BUILD (vs. silently under-allocating).
static_assert((int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "model_stage_mamba3: sizeof(MambaSampleSmem) drifted from "
              "kMambaSmemBytes (mamba3_layout.cuh). Update kMambaSmemFloats.");

// ── Typed views over the flat weight blob using the generated offsets. The
//    ORDER MUST match _mamba_param_sizes() / named_parameters() EXACTLY. ────────
struct MambaWeights {
    const float* tok;     // [kVocab, kD]
    const float* pos;     // [kSeq, kD]
    struct Layer {
        const float* mixn_w;    // mixer_norm.weight [d]
        const float* A_log;     // [n_heads]
        const float* D;         // [d_inner]
        const float* B_bias;    // [Nc]
        const float* Bhat_bias; // [Nc]
        const float* C_bias;    // [Nc]
        const float* Chat_bias; // [Nc]
        const float* in_w;      // [2*d_inner, d]
        const float* x_proj_w;  // [x_proj_out, d_inner]
        const float* dt_proj_w; // [n_heads, dt_rank]
        const float* dt_proj_b; // [n_heads]
        const float* B_norm_w;    // [Nc]
        const float* Bhat_norm_w; // [Nc]
        const float* C_norm_w;    // [Nc]
        const float* Chat_norm_w; // [Nc]
        const float* out_w;     // [d, d_inner]
        const float* mlpn_w;    // mlp_norm.weight [d]
        const float* gate_w;    // [d_ff, d]
        const float* up_w;      // [d_ff, d]
        const float* down_w;    // [d, d_ff]
    } layer[mb::kLayers];
    const float* norm_w;  // final norm [d]
    const float* out_w;   // [kPHead, d]
    const float* out_b;   // [kPHead]
};

__device__ __forceinline__ MambaWeights mb_bind(const float* p) {
    MambaWeights w;
    int i = 0;
    w.tok = p + kMambaOffsets[i++];
    w.pos = p + kMambaOffsets[i++];
    for (int li = 0; li < mb::kLayers; ++li) {
        w.layer[li].mixn_w      = p + kMambaOffsets[i++];
        w.layer[li].A_log       = p + kMambaOffsets[i++];
        w.layer[li].D           = p + kMambaOffsets[i++];
        w.layer[li].B_bias      = p + kMambaOffsets[i++];
        w.layer[li].Bhat_bias   = p + kMambaOffsets[i++];
        w.layer[li].C_bias      = p + kMambaOffsets[i++];
        w.layer[li].Chat_bias   = p + kMambaOffsets[i++];
        w.layer[li].in_w        = p + kMambaOffsets[i++];
        w.layer[li].x_proj_w    = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_w   = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_b   = p + kMambaOffsets[i++];
        w.layer[li].B_norm_w    = p + kMambaOffsets[i++];
        w.layer[li].Bhat_norm_w = p + kMambaOffsets[i++];
        w.layer[li].C_norm_w    = p + kMambaOffsets[i++];
        w.layer[li].Chat_norm_w = p + kMambaOffsets[i++];
        w.layer[li].out_w       = p + kMambaOffsets[i++];
        w.layer[li].mlpn_w      = p + kMambaOffsets[i++];
        w.layer[li].gate_w      = p + kMambaOffsets[i++];
        w.layer[li].up_w        = p + kMambaOffsets[i++];
        w.layer[li].down_w      = p + kMambaOffsets[i++];
    }
    w.norm_w = p + kMambaOffsets[i++];
    w.out_w  = p + kMambaOffsets[i++];
    w.out_b  = p + kMambaOffsets[i++];
    return w;
}

struct MambaGrad {
    float* tok; float* pos;
    struct Layer {
        float* mixn_w; float* A_log; float* D;
        float* B_bias; float* Bhat_bias; float* C_bias; float* Chat_bias;
        float* in_w; float* x_proj_w; float* dt_proj_w; float* dt_proj_b;
        float* B_norm_w; float* Bhat_norm_w; float* C_norm_w; float* Chat_norm_w;
        float* out_w; float* mlpn_w; float* gate_w; float* up_w; float* down_w;
    } layer[mb::kLayers];
    float* norm_w; float* out_w; float* out_b;
};
__device__ __forceinline__ MambaGrad mb_bind_grad(float* p) {
    MambaGrad w; int i = 0;
    w.tok = p + kMambaOffsets[i++];
    w.pos = p + kMambaOffsets[i++];
    for (int li = 0; li < mb::kLayers; ++li) {
        w.layer[li].mixn_w      = p + kMambaOffsets[i++];
        w.layer[li].A_log       = p + kMambaOffsets[i++];
        w.layer[li].D           = p + kMambaOffsets[i++];
        w.layer[li].B_bias      = p + kMambaOffsets[i++];
        w.layer[li].Bhat_bias   = p + kMambaOffsets[i++];
        w.layer[li].C_bias      = p + kMambaOffsets[i++];
        w.layer[li].Chat_bias   = p + kMambaOffsets[i++];
        w.layer[li].in_w        = p + kMambaOffsets[i++];
        w.layer[li].x_proj_w    = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_w   = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_b   = p + kMambaOffsets[i++];
        w.layer[li].B_norm_w    = p + kMambaOffsets[i++];
        w.layer[li].Bhat_norm_w = p + kMambaOffsets[i++];
        w.layer[li].C_norm_w    = p + kMambaOffsets[i++];
        w.layer[li].Chat_norm_w = p + kMambaOffsets[i++];
        w.layer[li].out_w       = p + kMambaOffsets[i++];
        w.layer[li].mlpn_w      = p + kMambaOffsets[i++];
        w.layer[li].gate_w      = p + kMambaOffsets[i++];
        w.layer[li].up_w        = p + kMambaOffsets[i++];
        w.layer[li].down_w      = p + kMambaOffsets[i++];
    }
    w.norm_w = p + kMambaOffsets[i++];
    w.out_w  = p + kMambaOffsets[i++];
    w.out_b  = p + kMambaOffsets[i++];
    return w;
}

// ── Block-wide reductions (256 threads). ──────────────────────────────────────
__device__ __forceinline__ float mb_block_sum(float v, float* red) {
    const unsigned full = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(full, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float p = (lane < nwarp) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(full, p, o);
        if (lane == 0) red[0] = p;
    }
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float mb_block_max(float v, float* red) {
    const unsigned full = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(full, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float p = (lane < nwarp) ? red[lane] : -CUDART_INF_F;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p = fmaxf(p, __shfl_down_sync(full, p, o));
        if (lane == 0) red[0] = p;
    }
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// ── Primitive linear op (CTA-cooperative). y[s,:out] = x[s,:in] @ W[out,in]^T
//    (+ b). One thread per (s,o), strided. `ldX`/`ldY` are the row strides (in
//    elements) of x/y (so a packed view into a wider buffer works). ───────────
__device__ __forceinline__ void mb_linear(
        const float* __restrict__ x, int in_dim, int ldX,
        const float* __restrict__ W, const float* __restrict__ b, int out_dim,
        float* __restrict__ y, int ldY) {
    const int total = mb::kSeq * out_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int s = idx / out_dim, o = idx % out_dim;
        const float* xr = x + (int64_t)s * ldX;
        const float* Wr = W + (int64_t)o * in_dim;
        float acc = (b != nullptr) ? b[o] : 0.0f;
        #pragma unroll 4
        for (int k = 0; k < in_dim; ++k) acc += xr[k] * Wr[k];
        y[(int64_t)s * ldY + o] = acc;
    }
    __syncthreads();
}

// ── RMSNorm forward over the last dim (width=D). y = x*rsqrt(mean(x^2)+eps)*w.
//    Caches xhat = x*r and r (per row). NO mean-subtract, NO bias (RMS, not LN).
//    ldX = explicit row stride of x. xhat_out/y packed D-wide. (mamba3_oracle
//    rmsnorm_forward.) `width` so it serves both kD (norms) and kStateC (BCNorm).
__device__ __forceinline__ void mb_rmsnorm_fwd(
        const float* __restrict__ x, const float* __restrict__ w,
        float* __restrict__ y, float* __restrict__ xhat_out,
        float* __restrict__ r_out, float* red, int width, int ldX) {
    for (int s = 0; s < mb::kSeq; ++s) {
        const float* xr = x + (int64_t)s * ldX;
        float ss = 0.0f;
        for (int j = threadIdx.x; j < width; j += blockDim.x) { float v = xr[j]; ss += v * v; }
        float ms = mb_block_sum(ss, red) / (float)width;
        float r = rsqrtf(ms + mb::kRmsEps);
        if (threadIdx.x == 0) r_out[s] = r;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float xh = xr[j] * r;
            xhat_out[(int64_t)s * width + j] = xh;
            y[(int64_t)s * width + j] = xh * w[j];
        }
        __syncthreads();
    }
}

// ── RMSNorm backward (mamba3_oracle rmsnorm_backward):
//     dw   = sum_rows (dy * xhat)
//     dxhat= dy * w
//     dx   = r * (dxhat - xhat * (sum_lastdim(dxhat*xhat) / D))
//    dy packed width-wide; xhat packed width-wide; dx_out packed width-wide.
//    gw is ACCUMULATED (+=). ─────────────────────────────────────────────────
__device__ inline void mb_rmsnorm_bwd(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ r, const float* __restrict__ w,
        float* __restrict__ dx_out, float* __restrict__ gw, float* red, int width) {
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float dgw = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s)
            dgw += dy[(int64_t)s * width + j] * xhat[(int64_t)s * width + j];
        gw[j] += dgw;
    }
    for (int s = 0; s < mb::kSeq; ++s) {
        const float* dyr = dy + (int64_t)s * width;
        const float* xhr = xhat + (int64_t)s * width;
        float sdax = 0.0f;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float dxhat = dyr[j] * w[j];
            sdax += dxhat * xhr[j];
        }
        sdax = mb_block_sum(sdax, red);
        float corr = sdax / (float)width;
        float rs = r[s];
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float dxhat = dyr[j] * w[j];
            dx_out[(int64_t)s * width + j] = rs * (dxhat - xhr[j] * corr);
        }
        __syncthreads();
    }
}

// ── dW += dY^T @ X for a linear Y = X @ W^T (+ b). dX = dY @ W (set or add).
//    Owner-thread per (o,i) for dW; db[o]+=Σ_s dY. ldX/ldY are x/dY row strides.─
__device__ inline void mb_linear_bwd(
        const float* __restrict__ dY, int ldY, const float* __restrict__ X, int ldX,
        const float* __restrict__ W, int in_dim, int out_dim,
        float* __restrict__ dW, float* __restrict__ db,
        float* __restrict__ dx_out, int ldDx, bool set_dx) {
    const int wtot = out_dim * in_dim;
    for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
        const int o = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s) acc += dY[(int64_t)s * ldY + o] * X[(int64_t)s * ldX + i];
        dW[(int64_t)o * in_dim + i] += acc;
    }
    if (db != nullptr) {
        for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < mb::kSeq; ++s) acc += dY[(int64_t)s * ldY + o];
            db[o] += acc;
        }
    }
    const int xtot = mb::kSeq * in_dim;
    for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
        const int s = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        for (int o = 0; o < out_dim; ++o) acc += dY[(int64_t)s * ldY + o] * W[(int64_t)o * in_dim + i];
        if (set_dx) dx_out[(int64_t)s * ldDx + i] = acc;
        else dx_out[(int64_t)s * ldDx + i] += acc;
    }
    __syncthreads();
}

// ── RMSNorm backward variant that recomputes xhat = x*r from the RAW pre-norm
//    input x (so callers that did NOT cache xhat — BCNorm, which caches only the
//    raw stream + recip — can reuse it). dx_out is written with row stride ldDx
//    (so it can land in a packed slice of a wider buffer). gw ACCUMULATED. ─────
__device__ inline void mb_rmsnorm_bwd_rawx(
        const float* __restrict__ dy, const float* __restrict__ x,
        const float* __restrict__ r, const float* __restrict__ w,
        float* __restrict__ dx_out, float* __restrict__ gw,
        float* red, int width, int ldDx) {
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float dgw = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s)
            dgw += dy[(int64_t)s * width + j] * (x[(int64_t)s * width + j] * r[s]);
        gw[j] += dgw;
    }
    for (int s = 0; s < mb::kSeq; ++s) {
        const float* dyr = dy + (int64_t)s * width;
        const float* xr  = x  + (int64_t)s * width;
        float rs = r[s];
        float sdax = 0.0f;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float xhat = xr[j] * rs;
            float dxhat = dyr[j] * w[j];
            sdax += dxhat * xhat;
        }
        sdax = mb_block_sum(sdax, red);
        float corr = sdax / (float)width;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float xhat = xr[j] * rs;
            float dxhat = dyr[j] * w[j];
            dx_out[(int64_t)s * ldDx + j] = rs * (dxhat - xhat * corr);
        }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════════════
//  COMPLEX EXPONENTIAL-TRAPEZOIDAL SELECTIVE SCAN — forward (Eq 25). One thread
//  owns SSM channel j in [0,d_inner). Its head is h = j/head_dim. Within the
//  head it holds the complex state h[Nc][2] in REGISTERS and unrolls t=0..kSeq-1.
//  Reads (from the LayerAct cache `a`): x_in[t][j], dt_pre[t][h] (per head),
//  A_mod[t][h], u_lam[t][h], theta[t][c], Bbar[t][c][:], Cbar[t][c][:], and
//  A_log[h]. Writes y_scan[t][j]. Mirrors mamba3_oracle.scan_forward.
//    alpha=exp(dt*A); A=-softplus(A_mod+exp(A_log)); dt=softplus(dt_pre);
//    lam=sigmoid(u_lam); gamma=lam*dt; beta=(1-lam)*dt*alpha; phi=dt*theta.
//    v_t = (Br2*x, Bi2*x);  h_t = alpha*R h_{t-1} + beta*R v_{t-1} + gamma*v_t;
//    y_t = sum_c (Cbar0*h0 + Cbar1*h1).
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_scan_fwd(const float* __restrict__ A_log,
                           const MambaSampleSmem::LayerAct* a,
                           float* __restrict__ y_scan_out) {
    for (int j = threadIdx.x; j < mb::kDInner; j += blockDim.x) {
        const int h = j / mb::kHeadDim;
        const float base_rate = __expf(A_log[h]);       // exp(A_log[h])
        // complex state (real,imag) per coord + previous v_t. NOTE: these per-coord
        // arrays (kStateC=64 each × 4) are RUNTIME-INDEXED local memory — the c-loop
        // is intentionally NOT #pragma unroll'd (full unroll would force 256 regs >
        // the 255 cap and the carried state across the t-loop would not persist; that
        // was the "t=0 right, t>0 wrong" bug). The t-loop (only kSeq=8) stays rolled.
        float hr[mb::kStateC], hi[mb::kStateC];
        float vpr[mb::kStateC], vpi[mb::kStateC];        // v_{t-1}
        for (int c = 0; c < mb::kStateC; ++c) { hr[c]=0.f; hi[c]=0.f; vpr[c]=0.f; vpi[c]=0.f; }
        for (int t = 0; t < mb::kSeq; ++t) {
            const float xv = a->x_in[t][j];
            const float dt_t = mb_softplus(a->dt_pre[t][h]);
            const float A_t = -mb_softplus(a->A_mod[t][h] + base_rate);
            const float lam = mb_sigmoid(a->u_lam[t][h]);
            const float alpha = __expf(dt_t * A_t);
            const float gamma = lam * dt_t;
            const float beta = (1.0f - lam) * dt_t * alpha;
            float yacc = 0.0f;
            for (int c = 0; c < mb::kStateC; ++c) {
                const float phi = dt_t * a->theta[t][c];
                float cs, sn; __sincosf(phi, &sn, &cs);
                // v_t = Bbar_t (.) x_t
                const float vr = a->Bbar[t][c][0] * xv;
                const float vi = a->Bbar[t][c][1] * xv;
                // R @ h_{t-1}
                const float Rhr = cs * hr[c] - sn * hi[c];
                const float Rhi = sn * hr[c] + cs * hi[c];
                // R @ v_{t-1}
                const float Rvr = cs * vpr[c] - sn * vpi[c];
                const float Rvi = sn * vpr[c] + cs * vpi[c];
                // h_t = alpha*Rh + beta*Rv + gamma*v_t
                const float nhr = alpha * Rhr + beta * Rvr + gamma * vr;
                const float nhi = alpha * Rhi + beta * Rvi + gamma * vi;
                hr[c] = nhr; hi[c] = nhi;
                vpr[c] = vr; vpi[c] = vi;
                // y_t += Cbar . h_t
                yacc += a->Cbar[t][c][0] * nhr + a->Cbar[t][c][1] * nhi;
            }
            y_scan_out[t * mb::kDInner + j] = yacc;
        }
    }
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  COMPLEX SCAN backward (reverse time). Per channel j (one thread, head
//  h=j/head_dim): RECOMPUTE the forward keeping h_hist[kSeq+1] AND v_hist[kSeq+1]
//  (complex) in registers (seq=8 — NO checkpoint), then reverse-scan the
//  two-adjoint (gh,gv) recurrence (MAMBA3_REFERENCE.md sec 6.2).
//
//  PER-CHANNEL outputs (this thread owns channel j):
//    dx_in[t,j]  (the scan's contribution to x_in)
//    ddt_pre[t,h] (per head — accumulated across the head_dim channels via block
//                  reduce BY HEAD), dA_mod, du_lam (per head, same)
//  HEAD-SHARED outputs (reduced over ALL channels into smem):
//    dBbar[t,c,:], dCbar[t,c,:], dtheta[t,c]
//
//  The per-head dalpha/dbeta/dgamma/dphi are folded into ddt/dA_real/dlam/dtheta
//  via the sec 6.3 coefficient Jacobian. Because alpha/beta/gamma/phi are shared
//  by the head_dim channels of a head, the head-scalar grads are summed over the
//  channels in the head (block reduce restricted to the head's lanes), then the
//  owner of that head writes ddt_pre/dA_mod/du_lam.
//
//  dA_log[h] = sum_t dA_real[t,h] * (-sigmoid(A_arg)) * exp(A_log[h]); the
//  -softplus and exp(A_log) Jacobians fold here. (Accumulated into g_A_log.)
//
//  Determinism: the head-shared dBbar/dCbar/dtheta block-reduce per (t,c) in the
//  reverse loop (ascending lane/warp); the per-head coeff grads block-reduce per
//  (t,h). Same addends, fixed order -> A/A/A bit-identical.
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_scan_bwd(const float* __restrict__ A_log,
                           const MambaSampleSmem::LayerAct* a,
                           const float* __restrict__ dy_scan,     // [kSeq,d_inner]
                           float* __restrict__ dx_in,             // [kSeq,d_inner] SET
                           float* __restrict__ ddt_pre,           // [kSeq,n_heads] SET (per head)
                           float* __restrict__ dA_mod,            // [kSeq,n_heads] SET
                           float* __restrict__ du_lam,            // [kSeq,n_heads] SET
                           float* __restrict__ dBbar,             // [kSeq,Nc,2]  smem (+=, zeroed by caller)
                           float* __restrict__ dCbar,             // [kSeq,Nc,2]  smem
                           float* __restrict__ dtheta,            // [kSeq,Nc]    smem
                           float* __restrict__ g_A_log,           // [n_heads] partial (+=)
                           float* __restrict__ red) {
    const bool active = (threadIdx.x < mb::kDInner);
    const int j = threadIdx.x;
    const int h = active ? (j / mb::kHeadDim) : 0;
    const float base_rate = active ? __expf(A_log[h]) : 0.0f;
    // Per-(t) cached scalars (recomputed), per-channel forward state history.
    float dt_save[mb::kSeq], alpha_s[mb::kSeq], beta_s[mb::kSeq], gamma_s[mb::kSeq];
    float A_s[mb::kSeq], lam_s[mb::kSeq];
    // complex state history: hh[t] = h_{t-1} (hh[0]=0), v stored as vh[t]=v_t.
    float hhr[mb::kSeq + 1][mb::kStateC], hhi[mb::kSeq + 1][mb::kStateC];
    float vhr[mb::kSeq][mb::kStateC], vhi[mb::kSeq][mb::kStateC];
    if (active) {
        for (int c = 0; c < mb::kStateC; ++c) { hhr[0][c]=0.f; hhi[0][c]=0.f; }
        for (int t = 0; t < mb::kSeq; ++t) {
            const float xv = a->x_in[t][j];
            const float dt_t = mb_softplus(a->dt_pre[t][h]);
            const float A_t = -mb_softplus(a->A_mod[t][h] + base_rate);
            const float lam = mb_sigmoid(a->u_lam[t][h]);
            const float alpha = __expf(dt_t * A_t);
            const float gamma = lam * dt_t;
            const float beta = (1.0f - lam) * dt_t * alpha;
            dt_save[t]=dt_t; A_s[t]=A_t; lam_s[t]=lam; alpha_s[t]=alpha; beta_s[t]=beta; gamma_s[t]=gamma;
            for (int c = 0; c < mb::kStateC; ++c) {
                const float phi = dt_t * a->theta[t][c];
                float cs, sn; __sincosf(phi, &sn, &cs);
                const float vr = a->Bbar[t][c][0] * xv;
                const float vi = a->Bbar[t][c][1] * xv;
                const float prev_vr = (t > 0) ? vhr[t-1][c] : 0.0f;
                const float prev_vi = (t > 0) ? vhi[t-1][c] : 0.0f;
                const float Rhr = cs * hhr[t][c] - sn * hhi[t][c];
                const float Rhi = sn * hhr[t][c] + cs * hhi[t][c];
                const float Rvr = cs * prev_vr - sn * prev_vi;
                const float Rvi = sn * prev_vr + cs * prev_vi;
                hhr[t+1][c] = alpha_s[t] * Rhr + beta_s[t] * Rvr + gamma_s[t] * vr;
                hhi[t+1][c] = alpha_s[t] * Rhi + beta_s[t] * Rvi + gamma_s[t] * vi;
                vhr[t][c] = vr; vhi[t][c] = vi;
            }
        }
    }
    // reverse-time adjoints (per channel): gh = dL/dh_t, gv = dL/dv_t (from t+1).
    float ghr[mb::kStateC], ghi[mb::kStateC], gvr[mb::kStateC], gvi[mb::kStateC];
    if (active) {
        #pragma unroll
        for (int c = 0; c < mb::kStateC; ++c) { ghr[c]=0.f; ghi[c]=0.f; gvr[c]=0.f; gvi[c]=0.f; }
    }
    // Per-head dA_log accumulation across t (this channel contributes via its head;
    // reduced by head). dA_log[h] += sum_t dA_real[t,h]*(-sigmoid(A_arg))*exp(A_log[h]).
    float dAlog_acc = 0.0f;   // this channel's running sum (folded per head at end)
    for (int t = mb::kSeq - 1; t >= 0; --t) {
        // This channel's per-(t,head) coefficient-grad contributions (summed over c).
        float dalpha = 0.f, dbeta = 0.f, dgamma = 0.f, dt_phi = 0.f;
        float dx_acc = 0.0f;
        const float xv = active ? a->x_in[t][j] : 0.0f;
        const float dt_t = active ? dt_save[t] : 0.0f;
        const float a_t = active ? alpha_s[t] : 0.0f;
        const float b_t = active ? beta_s[t] : 0.0f;
        const float g_t = active ? gamma_s[t] : 0.0f;
        const float dyv = active ? dy_scan[t * mb::kDInner + j] : 0.0f;
        for (int c = 0; c < mb::kStateC; ++c) {
            float dBbar0 = 0.f, dBbar1 = 0.f, dCbar0 = 0.f, dCbar1 = 0.f, dphi = 0.f;
            if (active) {
                const float phi = dt_t * a->theta[t][c];
                float cs, sn; __sincosf(phi, &sn, &cs);
                // (1) y_t = sum Cbar . h_t -> gh += Cbar * dy ; dCbar += h_t * dy
                const float h0 = hhr[t+1][c], h1 = hhi[t+1][c];
                dCbar0 = h0 * dyv; dCbar1 = h1 * dyv;
                ghr[c] += a->Cbar[t][c][0] * dyv;
                ghi[c] += a->Cbar[t][c][1] * dyv;          // gh now = full dL/dh_t
                // (2) v_t total grad = gv + gamma*gh ; v_t = Bbar*x
                const float dv0 = gvr[c] + g_t * ghr[c];
                const float dv1 = gvi[c] + g_t * ghi[c];
                dx_acc += dv0 * a->Bbar[t][c][0] + dv1 * a->Bbar[t][c][1];
                dBbar0 = dv0 * xv; dBbar1 = dv1 * xv;
                // (3) rotated terms Rh = R h_{t-1}, Rv = R v_{t-1}; coeff + phi grads
                const float hr_p = hhr[t][c], hi_p = hhi[t][c];
                const float vr_p = (t>0)? vhr[t-1][c]:0.0f, vi_p = (t>0)? vhi[t-1][c]:0.0f;
                const float Rh0 = cs * hr_p - sn * hi_p, Rh1 = sn * hr_p + cs * hi_p;
                const float Rv0 = cs * vr_p - sn * vi_p, Rv1 = sn * vr_p + cs * vi_p;
                dalpha += Rh0 * ghr[c] + Rh1 * ghi[c];
                dbeta  += Rv0 * ghr[c] + Rv1 * ghi[c];
                dgamma += vhr[t][c] * ghr[c] + vhi[t][c] * ghi[c];
                // dphi via R'(phi)=[[-sin,-cos],[cos,-sin]] on h_{t-1}(coef alpha) & v_{t-1}(coef beta)
                const float dRh = ghr[c]*(-sn*hr_p - cs*hi_p) + ghi[c]*(cs*hr_p - sn*hi_p);
                const float dRv = ghr[c]*(-sn*vr_p - cs*vi_p) + ghi[c]*(cs*vr_p - sn*vi_p);
                dphi = a_t * dRh + b_t * dRv;
                dt_phi += dphi * a->theta[t][c];           // dt-fold term (sum_c dphi*theta)
                // (4) propagate to prev step (R^T): R^T g = (cs*g0+sn*g1, -sn*g0+cs*g1)
                const float g0 = ghr[c], g1 = ghi[c];
                const float RTg0 = cs * g0 + sn * g1;
                const float RTg1 = -sn * g0 + cs * g1;
                gvr[c] = b_t * RTg0; gvi[c] = b_t * RTg1;   // dL/dv_{t-1}
                ghr[c] = a_t * RTg0; ghi[c] = a_t * RTg1;   // dL/dh_{t-1}
            }
            // head-shared reduces (per t,c): block-sum across ALL channels (ascending).
            float sB0 = mb_block_sum(dBbar0, red);
            float sB1 = mb_block_sum(dBbar1, red);
            float sC0 = mb_block_sum(dCbar0, red);
            float sC1 = mb_block_sum(dCbar1, red);
            float sTh = mb_block_sum(active ? (dphi * dt_t) : 0.0f, red);  // dtheta[t,c] += dphi*dt
            if (threadIdx.x == 0) {
                dBbar[(t * mb::kStateC + c) * 2 + 0] += sB0;
                dBbar[(t * mb::kStateC + c) * 2 + 1] += sB1;
                dCbar[(t * mb::kStateC + c) * 2 + 0] += sC0;
                dCbar[(t * mb::kStateC + c) * 2 + 1] += sC1;
                dtheta[t * mb::kStateC + c] += sTh;
            }
        }
        if (active) dx_in[t * mb::kDInner + j] = dx_acc;
        // ── Fold the per-head coefficient Jacobians (sec 6.3) for step t. The
        //    head_dim channels of a head share alpha/beta/gamma/dt/A/lam, so sum
        //    each channel's dalpha/dbeta/dgamma/dt_phi over the head, then the owner
        //    writes the per-head ddt_pre/dA_mod/du_lam. Reduce-by-head via masked
        //    block sums (one per head, ascending) keeps it deterministic.
        //    ddt(post-softplus) = dalpha*(A*alpha) + dbeta*((1-lam)*alpha*(1+dt*A))
        //                       + dgamma*lam + dt_phi
        //    dlam = dbeta*(-dt*alpha) + dgamma*dt
        //    dA_real = dalpha*(dt*alpha) + dbeta*((1-lam)*dt^2*alpha)
        //    then ddt_pre = ddt*softplus'(dt_pre); dA_mod = dA_real*(-sigmoid(A_arg));
        //         du_lam = dlam*lam*(1-lam); dA_log fold uses dA_real*(-sig)*exp(A_log).
        const float A_t = active ? A_s[t] : 0.0f;
        const float lam = active ? lam_s[t] : 0.0f;
        float ddt_c = 0.f, dlam_c = 0.f, dAreal_c = 0.f;
        if (active) {
            ddt_c   = dalpha*(A_t*a_t) + dbeta*((1.0f-lam)*a_t*(1.0f+dt_t*A_t)) + dgamma*lam + dt_phi;
            dlam_c  = dbeta*(-dt_t*a_t) + dgamma*dt_t;
            dAreal_c= dalpha*(dt_t*a_t) + dbeta*((1.0f-lam)*dt_t*dt_t*a_t);
        }
        // Reduce the 3 head scalars over the channels of each head. Loop heads;
        // each head's lanes [h*head_dim, (h+1)*head_dim) contribute, owner = first lane.
        #pragma unroll
        for (int hh = 0; hh < mb::kNHeads; ++hh) {
            const bool mine = active && (h == hh);
            float sddt   = mb_block_sum(mine ? ddt_c   : 0.0f, red);
            float sdlam  = mb_block_sum(mine ? dlam_c  : 0.0f, red);
            float sdAr   = mb_block_sum(mine ? dAreal_c: 0.0f, red);
            if (threadIdx.x == 0) {
                // A_arg = A_mod + exp(A_log); A_real = -softplus(A_arg)
                // (read A_mod/dt_pre/u_lam at any lane — use the cached per-(t,h) values).
                const float dt_pre_h = a->dt_pre[t][hh];
                const float u_lam_h  = a->u_lam[t][hh];
                const float A_arg_h  = a->A_mod[t][hh] + __expf(A_log[hh]);
                ddt_pre[t * mb::kNHeads + hh] = sddt * mb_softplus_grad(dt_pre_h);
                du_lam [t * mb::kNHeads + hh] = sdlam * mb_sigmoid_grad(u_lam_h);
                dA_mod [t * mb::kNHeads + hh] = sdAr * (-mb_softplus_grad(A_arg_h));
            }
            __syncthreads();
        }
        // dA_log[h] += sum_t dA_real[t,h]*(-sigmoid(A_arg))*exp(A_log[h]). Fold the
        // per-channel dAreal here (same channel-in-head contribution).
        if (active) {
            const float A_arg = a->A_mod[t][h] + base_rate;
            dAlog_acc += dAreal_c * (-mb_sigmoid(A_arg)) * base_rate;
        }
    }
    // reduce dA_log over the head's channels (owner writes the per-head partial).
    #pragma unroll
    for (int hh = 0; hh < mb::kNHeads; ++hh) {
        const bool mine = active && (h == hh);
        float sAl = mb_block_sum(mine ? dAlog_acc : 0.0f, red);
        if (threadIdx.x == 0) g_A_log[hh] += sAl;
        __syncthreads();
    }
    __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════
//  MIXER forward for one sample (CTA-cooperative). Input = xn (= RMSNorm_mix(x),
//  [kSeq,d]) staged by the block. Caches mixer internals into `a`. Writes the
//  bare mixer output (residual=False) into `mix_out` [kSeq,d]. Uses sm scratch.
//  Mirrors mamba3_oracle.layer_forward (residual OFF).
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_mixer_fwd(const MambaWeights::Layer& L,
                            const float* __restrict__ xn,   // [kSeq,d] (mixer input)
                            MambaSampleSmem::LayerAct* a,
                            float* __restrict__ mix_out,     // [kSeq,d] OUT
                            MambaSampleSmem* sm) {
    // xz = in_proj(xn); split rows [0,d_inner)->x_in, [d_inner,2di)->z.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
        const int s = idx / mb::kDInner, o = idx % mb::kDInner;
        const float* xr = xn + (int64_t)s * mb::kD;
        const float* Wr0 = L.in_w + (int64_t)o * mb::kD;
        const float* Wr1 = L.in_w + (int64_t)(mb::kDInner + o) * mb::kD;
        float a0 = 0.f, a1 = 0.f;
        #pragma unroll 4
        for (int k = 0; k < mb::kD; ++k) { float xv = xr[k]; a0 += xv * Wr0[k]; a1 += xv * Wr1[k]; }
        a->x_in[s][o] = a0;       // = x_main (NO conv, NO SiLU)
        a->z[s][o] = a1;
    }
    __syncthreads();
    // x_proj(x_in) -> sm->xproj [kSeq,xproj]; split into the SSM params.
    mb_linear(&a->x_in[0][0], mb::kDInner, mb::kDInner, L.x_proj_w, nullptr,
              mb::kXProj, &sm->xproj[0][0], mb::kXProj);
    // route the splits into the LayerAct caches.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kXProj; idx += blockDim.x) {
        const int s = idx / mb::kXProj, o = idx % mb::kXProj;
        const float v = sm->xproj[s][o];
        if (o < mb::kOffAmod) a->dt_lr[s][o - mb::kOffDtLr] = v;
        else if (o < mb::kOffThet) a->A_mod[s][o - mb::kOffAmod] = v;
        else if (o < mb::kOffULam) a->theta[s][o - mb::kOffThet] = v;
        else if (o < mb::kOffBr) a->u_lam[s][o - mb::kOffULam] = v;
        else if (o < mb::kOffBi) a->Br[s][o - mb::kOffBr] = v;
        else if (o < mb::kOffCr) a->Bi[s][o - mb::kOffBi] = v;
        else if (o < mb::kOffCi) a->Cr[s][o - mb::kOffCr] = v;
        else a->Ci[s][o - mb::kOffCi] = v;
    }
    __syncthreads();
    // dt_pre = dt_proj(dt_lr) + bias  [kSeq, n_heads] (PER HEAD).
    mb_linear(&a->dt_lr[0][0], mb::kDtRank, mb::kDtRank, L.dt_proj_w, L.dt_proj_b,
              mb::kNHeads, &a->dt_pre[0][0], mb::kNHeads);
    // BCNorm (RMSNorm over Nc) + biases -> Bbar=(Br2,Bi2), Cbar=(Cr2,-Ci2). The
    //   normed output lands in sm->adj_a at WIDTH-kStateC stride (s*Nc+c), NOT the
    //   [kSeq][kDInner] field stride — read it back with the same Nc stride.
    float* an = &sm->adj_a[0][0];
    mb_rmsnorm_fwd(&a->Br[0][0], L.B_norm_w, an, &sm->adj_b[0][0], &a->Br_r[0], sm->red, mb::kStateC, mb::kStateC);
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        a->Bbar[s][c][0] = an[s * mb::kStateC + c] + L.B_bias[c];
    }
    __syncthreads();
    mb_rmsnorm_fwd(&a->Bi[0][0], L.Bhat_norm_w, an, &sm->adj_b[0][0], &a->Bi_r[0], sm->red, mb::kStateC, mb::kStateC);
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        a->Bbar[s][c][1] = an[s * mb::kStateC + c] + L.Bhat_bias[c];
    }
    __syncthreads();
    mb_rmsnorm_fwd(&a->Cr[0][0], L.C_norm_w, an, &sm->adj_b[0][0], &a->Cr_r[0], sm->red, mb::kStateC, mb::kStateC);
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        a->Cbar[s][c][0] = an[s * mb::kStateC + c] + L.C_bias[c];
    }
    __syncthreads();
    mb_rmsnorm_fwd(&a->Ci[0][0], L.Chat_norm_w, an, &sm->adj_b[0][0], &a->Ci_r[0], sm->red, mb::kStateC, mb::kStateC);
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        a->Cbar[s][c][1] = -(an[s * mb::kStateC + c] + L.Chat_bias[c]);   // Cbar imag = -(Ci2)
    }
    __syncthreads();
    // selective scan -> a->y_scan.
    mb_scan_fwd(L.A_log, a, &a->y_scan[0][0]);
    // y_gated = (y_scan + x_in*D) * silu(z) -> sm->adj_a ; out = out_proj(y_gated).
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
        const int s = idx / mb::kDInner, c = idx % mb::kDInner;
        float yv = a->y_scan[s][c] + a->x_in[s][c] * L.D[c];
        sm->adj_a[s][c] = yv * mb_silu(a->z[s][c]);
    }
    __syncthreads();
    mb_linear(&sm->adj_a[0][0], mb::kDInner, mb::kDInner, L.out_w, nullptr,
              mb::kD, mix_out, mb::kD);
}

// ════════════════════════════════════════════════════════════════════════
//  SwiGLU forward for one sample. Input h1n = RMSNorm_mlp(h1). Caches g_pre,u_mlp.
//  out = down(silu(gate(x)) * up(x)). Mirrors mamba3_oracle.swiglu_forward.
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_swiglu_fwd(const MambaWeights::Layer& L,
                             const float* __restrict__ h1n,   // [kSeq,d]
                             MambaSampleSmem::LayerAct* a,
                             float* __restrict__ mlp_out,      // [kSeq,d] OUT
                             MambaSampleSmem* sm) {
    mb_linear(h1n, mb::kD, mb::kD, L.gate_w, nullptr, mb::kDff, &a->g_pre[0][0], mb::kDff);
    mb_linear(h1n, mb::kD, mb::kD, L.up_w,   nullptr, mb::kDff, &a->u_mlp[0][0], mb::kDff);
    // prod = silu(g_pre) * u -> sm->wff_a.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDff; idx += blockDim.x) {
        const int s = idx / mb::kDff, o = idx % mb::kDff;
        sm->wff_a[s][o] = mb_silu(a->g_pre[s][o]) * a->u_mlp[s][o];
    }
    __syncthreads();
    mb_linear(&sm->wff_a[0][0], mb::kDff, mb::kDff, L.down_w, nullptr, mb::kD, mlp_out, mb::kD);
}

// ════════════════════════════════════════════════════════════════════════
//  FORWARD for one sample (CTA-cooperative). Mirrors mamba3_oracle.model_forward
//  + block_forward. Caches BOTH blocks' activations into sm->act[li]. Returns NLL.
// ════════════════════════════════════════════════════════════════════════
__device__ inline float mb_forward_sample(const MambaWeights& w, const int* tokens_s,
                                    int target, MambaSampleSmem* sm) {
    // Embedding + positional -> layer_in[0].
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
        const int s = idx / mb::kD, j = idx % mb::kD;
        sm->layer_in[0][s][j] = w.tok[(int64_t)tokens_s[s] * mb::kD + j]
                              + w.pos[(int64_t)s * mb::kD + j];
    }
    __syncthreads();

    for (int li = 0; li < mb::kLayers; ++li) {
        const MambaWeights::Layer& L = w.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        float* hin = &sm->layer_in[li][0][0];   // block input (residual)
        // --- mixer sub-block: h1 = hin + mixer(RMSNorm_mix(hin)) ---
        mb_rmsnorm_fwd(hin, L.mixn_w, &sm->dr[0][0], &a->mixn_xhat[0][0], &a->mixn_r[0],
                       sm->red, mb::kD, mb::kD);   // xn -> sm->dr
        // mix_out written into adj_b at WIDTH-kD stride (s*kD+j); read it the same way.
        float* mo = &sm->adj_b[0][0];
        mb_mixer_fwd(L, &sm->dr[0][0], a, mo, sm);   // mix_out -> sm->adj_b (kD-strided)
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            a->h1[s][j] = hin[s * mb::kD + j] + mo[s * mb::kD + j];   // h1 = x + mixer_out
        }
        __syncthreads();
        // --- SwiGLU sub-block: h2 = h1 + mlp(RMSNorm_mlp(h1)) ---
        mb_rmsnorm_fwd(&a->h1[0][0], L.mlpn_w, &sm->dr[0][0], &a->mlpn_xhat[0][0], &a->mlpn_r[0],
                       sm->red, mb::kD, mb::kD);   // h1n -> sm->dr
        mb_swiglu_fwd(L, &sm->dr[0][0], a, mo, sm);  // mlp_out -> sm->adj_b (kD-strided)
        float* dst = (li + 1 < mb::kLayers) ? &sm->layer_in[li + 1][0][0] : &sm->final_in[0][0];
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            dst[s * mb::kD + j] = a->h1[s][j] + mo[s * mb::kD + j];   // h2 = h1 + mlp_out
        }
        __syncthreads();
    }

    // Final RMSNorm (LAST position) + head -> logits -> NLL.
    const float* hlast = &sm->final_in[mb::kSeq - 1][0];
    float ss = 0.0f;
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) { float v = hlast[j]; ss += v * v; }
    float ms = mb_block_sum(ss, sm->red) / (float)mb::kD;
    float r = rsqrtf(ms + mb::kRmsEps);
    if (threadIdx.x == 0) sm->fn_r[mb::kSeq - 1] = r;
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float xh = hlast[j] * r;
        sm->fn_xhat[mb::kSeq - 1][j] = xh;
        sm->adj_a[0][j] = xh * w.norm_w[j];   // hn (RMSNorm, NO bias) -> adj_a[0]
    }
    __syncthreads();
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
        const float* Wr = w.out_w + (int64_t)o * mb::kD;
        float acc = w.out_b[o];
        #pragma unroll 4
        for (int k = 0; k < mb::kD; ++k) acc += sm->adj_a[0][k] * Wr[k];
        sm->logits[o] = acc;
    }
    __syncthreads();
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, sm->logits[o]);
    lmax = mb_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(sm->logits[o] - lmax);
    es = mb_block_sum(es, sm->red);
    float logz = lmax + __logf(es);
    return logz - sm->logits[target];
}

// ════════════════════════════════════════════════════════════════════════
//  MIXER backward for one sample (CTA-cooperative, residual OFF). Given dmix_out
//  [kSeq,d] (= dL/d(mixer output)), accumulates the mixer's weight grads into G
//  and writes dxn [kSeq,d] (grad wrt the mixer INPUT = RMSNorm_mix(x)). Mirrors
//  mamba3_oracle.layer_backward (residual path handled by the block).
//  Scratch contract on entry: sm->dr holds the mixer's recomputed input xn
//  (re-staged by the caller). Uses adj_a/b/c, xproj, dBbar/dCbar/dtheta.
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_mixer_bwd(const MambaWeights::Layer& L, const MambaGrad::Layer& G,
                            const float* __restrict__ xn,    // [kSeq,d] mixer input (recomputed)
                            MambaSampleSmem::LayerAct* a,
                            const float* __restrict__ dmix_out,  // [kSeq,d]
                            float* __restrict__ dxn,         // [kSeq,d] OUT
                            MambaSampleSmem* sm) {
    // Recompute y_gated -> sm->adj_a (out_proj input). y_gated = (y_scan + x_in*D)*silu(z).
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
        const int s = idx / mb::kDInner, c = idx % mb::kDInner;
        float yv = a->y_scan[s][c] + a->x_in[s][c] * L.D[c];
        sm->adj_a[s][c] = yv * mb_silu(a->z[s][c]);
    }
    __syncthreads();
    // out_proj bwd: dW_out += dmix_out^T @ y_gated ; dy_gated = dmix_out @ out_w -> adj_c.
    mb_linear_bwd(dmix_out, mb::kD, &sm->adj_a[0][0], mb::kDInner, L.out_w,
                  mb::kDInner, mb::kD, G.out_w, nullptr, &sm->adj_c[0][0], mb::kDInner, true);
    // gate+skip bwd. y_gated=y_skip*silu(z); y_skip=y_scan + x_in*D.
    //   dy_skip = dy_gated*sz ; dsz = dy_gated*y_skip ; dz = dsz*silu'(z).
    //   dy_scan = dy_skip ; dx_in(D-path) = dy_skip*D ; dD += Σ dy_skip*x_in.
    //   dz -> sm->wff_b (NOT adj_a — the BCNorm bwd below clobbers adj_a; wff_b is
    //   free in the mixer bwd and kDff>=kDInner wide). adj_b := dy_scan (scan dy).
    for (int c = threadIdx.x; c < mb::kDInner; c += blockDim.x) {
        float dDc = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s) {
            float sz = mb_silu(a->z[s][c]);
            float xm = a->x_in[s][c];
            float yskip = a->y_scan[s][c] + xm * L.D[c];
            float dyg = sm->adj_c[s][c];
            float dyskip = dyg * sz;
            float dsz = dyg * yskip;
            sm->wff_b[s][c] = dsz * mb_silu_grad(a->z[s][c]);   // dz (survives BCNorm)
            dDc += dyskip * xm;
            sm->adj_b[s][c] = dyskip;          // dy_scan (D-path added back later)
        }
        G.D[c] += dDc;
    }
    __syncthreads();
    // selective scan bwd. dy_scan = adj_b. Zero the head-shared reduce buffers first.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        sm->dBbar[s][c][0] = 0.f; sm->dBbar[s][c][1] = 0.f;
        sm->dCbar[s][c][0] = 0.f; sm->dCbar[s][c][1] = 0.f;
        sm->dtheta[s][c] = 0.f;
    }
    __syncthreads();
    // scan bwd outputs: dx_in (scan path) -> sm->wff_b rows [0,d_inner) of row 0..? No:
    //   route dx_in (scan path) -> sm->adj_c[s][c] (dy_gated consumed). per-head grads
    //   -> sm->wff_a (3 blocks of kSeq*n_heads); dBbar/dCbar/dtheta -> smem reduce bufs;
    //   dA_log -> G.A_log (per head).
    float* ddt_pre_s = &sm->wff_a[0][0];                          // [kSeq,n_heads]
    float* dA_mod_s  = &sm->wff_a[0][0] + mb::kSeq * mb::kNHeads;  // [kSeq,n_heads]
    float* du_lam_s  = dA_mod_s + mb::kSeq * mb::kNHeads;          // [kSeq,n_heads]
    mb_scan_bwd(L.A_log, a, &sm->adj_b[0][0],
                &sm->adj_c[0][0],      // dx_in (scan path) SET (dy_gated consumed)
                ddt_pre_s, dA_mod_s, du_lam_s,
                &sm->dBbar[0][0][0], &sm->dCbar[0][0][0], &sm->dtheta[0][0],
                G.A_log, sm->red);
    // dx_in total = scan-path (adj_c) + D-path (dy_skip*D, dy_skip in adj_b) -> adj_c.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
        const int s = idx / mb::kDInner, c = idx % mb::kDInner;
        sm->adj_c[s][c] += sm->adj_b[s][c] * L.D[c];   // dx_in = scan + D path
    }
    __syncthreads();
    // dt_proj bwd: dW_dt += ddt_pre^T @ dt_lr ; db_dt += Σ ddt_pre ;
    //   ddt_lr = ddt_pre @ dt_proj_w -> staged into sm->xproj[:, kOffDtLr..dt_rank).
    mb_linear_bwd(ddt_pre_s, mb::kNHeads, &a->dt_lr[0][0], mb::kDtRank, L.dt_proj_w,
                  mb::kDtRank, mb::kNHeads, G.dt_proj_w, G.dt_proj_b,
                  &sm->xproj[0][mb::kOffDtLr], mb::kXProj, true);
    // dA_mod -> xproj[:, kOffAmod..) ; dtheta (smem) -> xproj[:, kOffThet..) ;
    // du_lam -> xproj[:, kOffULam..).  (theta is a direct slice of x_proj output.)
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kNHeads; idx += blockDim.x) {
        const int s = idx / mb::kNHeads, hh = idx % mb::kNHeads;
        sm->xproj[s][mb::kOffAmod + hh] = dA_mod_s[s * mb::kNHeads + hh];
        sm->xproj[s][mb::kOffULam + hh] = du_lam_s[s * mb::kNHeads + hh];
    }
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        sm->xproj[s][mb::kOffThet + c] = sm->dtheta[s][c];
    }
    __syncthreads();
    // BCNorm backward for each stream. Cbar=(Cr2,-Ci2): dCr2=dCbar0, dCi2=-dCbar1.
    //   Bbar=(Br2,Bi2): dBr2=dBbar0, dBi2=dBbar1. Each stream: bias grad = Σ_s d*2;
    //   then RMSNorm bwd (raw pre-norm input x = a->Br/Bi/Cr/Ci, recip = *_r).
    //   The staged d*2 lands in `an` at WIDTH-kStateC stride (s*Nc+c) so the
    //   rmsnorm-bwd (which reads dy at that stride) + the bias reduce agree.
    //   dBr -> xproj[:, kOffBr..); dBi -> kOffBi; dCr -> kOffCr; dCi -> kOffCi.
    float* anb = &sm->adj_a[0][0];
    // --- B real (dBr2 = dBbar0) ---
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        anb[s * mb::kStateC + c] = sm->dBbar[s][c][0];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < mb::kStateC; c += blockDim.x) {
        float acc = 0.f; for (int s = 0; s < mb::kSeq; ++s) acc += anb[s * mb::kStateC + c];
        G.B_bias[c] += acc;
    }
    __syncthreads();
    mb_rmsnorm_bwd_rawx(anb, &a->Br[0][0], &a->Br_r[0], L.B_norm_w,
                        &sm->xproj[0][mb::kOffBr], G.B_norm_w, sm->red, mb::kStateC, mb::kXProj);
    __syncthreads();
    // --- B imag (dBi2 = dBbar1) ---
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        anb[s * mb::kStateC + c] = sm->dBbar[s][c][1];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < mb::kStateC; c += blockDim.x) {
        float acc = 0.f; for (int s = 0; s < mb::kSeq; ++s) acc += anb[s * mb::kStateC + c];
        G.Bhat_bias[c] += acc;
    }
    __syncthreads();
    mb_rmsnorm_bwd_rawx(anb, &a->Bi[0][0], &a->Bi_r[0], L.Bhat_norm_w,
                        &sm->xproj[0][mb::kOffBi], G.Bhat_norm_w, sm->red, mb::kStateC, mb::kXProj);
    __syncthreads();
    // --- C real (dCr2 = dCbar0) ---
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        anb[s * mb::kStateC + c] = sm->dCbar[s][c][0];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < mb::kStateC; c += blockDim.x) {
        float acc = 0.f; for (int s = 0; s < mb::kSeq; ++s) acc += anb[s * mb::kStateC + c];
        G.C_bias[c] += acc;
    }
    __syncthreads();
    mb_rmsnorm_bwd_rawx(anb, &a->Cr[0][0], &a->Cr_r[0], L.C_norm_w,
                        &sm->xproj[0][mb::kOffCr], G.C_norm_w, sm->red, mb::kStateC, mb::kXProj);
    __syncthreads();
    // --- C imag (dCi2 = -dCbar1, since Cbar imag = -Ci2) ---
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kStateC; idx += blockDim.x) {
        const int s = idx / mb::kStateC, c = idx % mb::kStateC;
        anb[s * mb::kStateC + c] = -sm->dCbar[s][c][1];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < mb::kStateC; c += blockDim.x) {
        float acc = 0.f; for (int s = 0; s < mb::kSeq; ++s) acc += anb[s * mb::kStateC + c];
        G.Chat_bias[c] += acc;
    }
    __syncthreads();
    mb_rmsnorm_bwd_rawx(anb, &a->Ci[0][0], &a->Ci_r[0], L.Chat_norm_w,
                        &sm->xproj[0][mb::kOffCi], G.Chat_norm_w, sm->red, mb::kStateC, mb::kXProj);
    __syncthreads();
    // x_proj is now fully assembled in sm->xproj (dt_lr|A_mod|theta|u_lam|Br|Bi|Cr|Ci).
    //   x_proj bwd: dW_xproj += xproj^T @ x_in ; dx_in(x_proj path) = xproj @ x_proj_w
    //   -> ADD to dx_in accumulator (sm->adj_c).
    mb_linear_bwd(&sm->xproj[0][0], mb::kXProj, &a->x_in[0][0], mb::kDInner, L.x_proj_w,
                  mb::kDInner, mb::kXProj, G.x_proj_w, nullptr, &sm->adj_c[0][0], mb::kDInner, false);
    // x_in fans into THREE: scan (in adj_c via dx_acc), x_proj (just added), D-skip
    //   (the D-path: dy_skip*D was added to adj_c above). All folded into adj_c now.
    //   dx_in = adj_c (complete). x_in = x_main (no SiLU) -> dx_main = dx_in.
    // in_proj bwd: dxz = [dx_main | dz]. dx_main = adj_c (stride kDInner) ;
    //   dz = sm->wff_b (stride kDff). Build the in_proj dW + dX directly:
    //   dW_in += dxz^T @ xn ; dx = dxz @ in_w -> dxn.
    {
        const int half = mb::kDInner;
        for (int o = threadIdx.x; o < 2 * mb::kDInner; o += blockDim.x) {
            const bool lo = (o < half);
            const float* dyo = lo ? &sm->adj_c[0][0] : &sm->wff_b[0][0];
            const int dystride = lo ? mb::kDInner : mb::kDff;
            const int oo = lo ? o : (o - half);
            float* gw = G.in_w + (int64_t)o * mb::kD;
            for (int i = 0; i < mb::kD; ++i) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s)
                    acc += dyo[s * dystride + oo] * xn[s * mb::kD + i];
                gw[i] += acc;
            }
        }
        __syncthreads();
        // dxn[s,i] = Σ_{o<di} dx_main[s,o]*in_w[o,i] + Σ_{o<di} dz[s,o]*in_w[di+o,i].
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, i = idx % mb::kD;
            float acc = 0.0f;
            for (int o = 0; o < mb::kDInner; ++o) {
                acc += sm->adj_c[s][o] * L.in_w[(int64_t)o * mb::kD + i];
                acc += sm->wff_b[s][o] * L.in_w[(int64_t)(mb::kDInner + o) * mb::kD + i];
            }
            dxn[s * mb::kD + i] = acc;
        }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════════════
//  SwiGLU backward for one sample. Given dmlp_out [kSeq,d], accumulates gate/up/
//  down weight grads into G and writes dh1n [kSeq,d] (grad wrt mlp_norm output).
//  Mirrors mamba3_oracle.swiglu_backward. Scratch: wff_a/wff_b, adj_a (d-wide).
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_swiglu_bwd(const MambaWeights::Layer& L, const MambaGrad::Layer& G,
                             const float* __restrict__ h1n,  // [kSeq,d] (mlp_norm out, recomputed)
                             MambaSampleSmem::LayerAct* a,
                             const float* __restrict__ dmlp_out,  // [kSeq,d]
                             float* __restrict__ dh1n,        // [kSeq,d] OUT
                             MambaSampleSmem* sm) {
    // recompute prod = silu(g_pre)*u -> sm->wff_a (down_proj input).
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDff; idx += blockDim.x) {
        const int s = idx / mb::kDff, o = idx % mb::kDff;
        sm->wff_a[s][o] = mb_silu(a->g_pre[s][o]) * a->u_mlp[s][o];
    }
    __syncthreads();
    // down_proj bwd: dW_down += dmlp_out^T @ prod ; dprod = dmlp_out @ down_w -> wff_b.
    mb_linear_bwd(dmlp_out, mb::kD, &sm->wff_a[0][0], mb::kDff, L.down_w,
                  mb::kDff, mb::kD, G.down_w, nullptr, &sm->wff_b[0][0], mb::kDff, true);
    // ds = dprod*u ; du = dprod*s ; dg_pre = ds*silu'(g_pre). Stage dg_pre -> wff_a,
    //   du -> wff_b (dprod consumed in place).
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDff; idx += blockDim.x) {
        const int s = idx / mb::kDff, o = idx % mb::kDff;
        float dprod = sm->wff_b[s][o];
        float sgp = mb_silu(a->g_pre[s][o]);
        float ds = dprod * a->u_mlp[s][o];
        float du = dprod * sgp;
        sm->wff_a[s][o] = ds * mb_silu_grad(a->g_pre[s][o]);   // dg_pre
        sm->wff_b[s][o] = du;                                  // du
    }
    __syncthreads();
    // gate_proj bwd: dW_gate += dg_pre^T @ h1n ; dx_gate = dg_pre @ gate_w -> dh1n (SET).
    mb_linear_bwd(&sm->wff_a[0][0], mb::kDff, h1n, mb::kD, L.gate_w,
                  mb::kD, mb::kDff, G.gate_w, nullptr, dh1n, mb::kD, true);
    // up_proj bwd: dW_up += du^T @ h1n ; dx_up = du @ up_w -> ADD to dh1n.
    mb_linear_bwd(&sm->wff_b[0][0], mb::kDff, h1n, mb::kD, L.up_w,
                  mb::kD, mb::kDff, G.up_w, nullptr, dh1n, mb::kD, false);
}

// ════════════════════════════════════════════════════════════════════════
//  BACKWARD for one sample (CTA-cooperative). Assumes mb_forward_sample ran for
//  THIS sample. Accumulates every weight grad into the CTA partial `g`. Mirrors
//  mamba3_oracle.model_backward + block_backward.
// ════════════════════════════════════════════════════════════════════════
__device__ inline void mb_backward_sample(const MambaWeights& w, const MambaGrad& g,
                                    const int* tokens_s, int target, int B,
                                    MambaSampleSmem* sm) {
    // ── CE bwd: dlogits = (softmax - onehot)/B. ──
    float lmax = -CUDART_INF_F;
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) lmax = fmaxf(lmax, sm->logits[o]);
    lmax = mb_block_max(lmax, sm->red);
    float es = 0.0f;
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) es += __expf(sm->logits[o] - lmax);
    es = mb_block_sum(es, sm->red);
    float inv_es = 1.0f / es;
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
        float smo = __expf(sm->logits[o] - lmax) * inv_es;
        sm->logits[o] = (smo - ((o == target) ? 1.0f : 0.0f)) / (float)B;
    }
    __syncthreads();
    // ── head: logits = hn @ out_w^T + out_b ; hn = fn_xhat[last]*norm_w (NO bias). ──
    const float* xh = &sm->fn_xhat[mb::kSeq - 1][0];
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
        float dl = sm->logits[o];
        g.out_b[o] += dl;
        float* gwrow = g.out_w + (int64_t)o * mb::kD;
        #pragma unroll 4
        for (int j = 0; j < mb::kD; ++j) gwrow[j] += dl * (xh[j] * w.norm_w[j]);
    }
    __syncthreads();
    // dhn[j] = Σ_o dl[o]*out_w[o,j] -> sm->dr row 0.
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < mb::kPHead; ++o) acc += sm->logits[o] * w.out_w[(int64_t)o * mb::kD + j];
        sm->dr[0][j] = acc;
    }
    __syncthreads();
    // ── final RMSNorm bwd (single row last): dw += dy*xhat ; dx -> dh[last]. ──
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        g.norm_w[j] += sm->dr[0][j] * xh[j];
    }
    __syncthreads();
    {
        float sdax = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = sm->dr[0][j] * w.norm_w[j]; sdax += dxhat * xh[j];
        }
        sdax = mb_block_sum(sdax, sm->red);
        float corr = sdax / (float)mb::kD;
        float rs = sm->fn_r[mb::kSeq - 1];
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = sm->dr[0][j] * w.norm_w[j];
            for (int s = 0; s < mb::kSeq; ++s) sm->dh[s][j] = 0.0f;
            sm->dh[mb::kSeq - 1][j] = rs * (dxhat - xh[j] * corr);
        }
        __syncthreads();
    }
    // sm->dh = dL/d(final-block output) [kSeq,d], only last pos nonzero.

    for (int li = mb::kLayers - 1; li >= 0; --li) {
        const MambaWeights::Layer& L = w.layer[li];
        const MambaGrad::Layer& G = g.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        // (block input layer_in[li] is read implicitly via the mixer_norm xhat cache;
        //  the residual flows through sm->dh.)
        // h2 = h1 + mlp(mlp_norm(h1)). dh2 = sm->dh.
        //   dh1 = dh2 (residual) + rmsnorm_bwd(swiglu_bwd(dh2)).
        // recompute h1n = mlp_norm(h1) -> sm->dr (mlp_norm fwd, scratch xhat in wff... use cache).
        //   We cached mlpn_xhat + mlpn_r, so h1n = mlpn_xhat * mlp_w.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            sm->dr[s][j] = a->mlpn_xhat[s][j] * L.mlpn_w[j];   // h1n
        }
        __syncthreads();
        // swiglu bwd: dmlp_out = dh2 (sm->dh) -> dh1n (sm->adj_a row d-wide... use sm->wff? need [kSeq,d]).
        //   route dh1n -> sm->fn_xhat (free now: head done) [kSeq,d].
        mb_swiglu_bwd(L, G, &sm->dr[0][0], a, &sm->dh[0][0], &sm->fn_xhat[0][0], sm);
        // rmsnorm_mlp bwd: dh1_mlpnorm = rmsnorm_bwd(dh1n) ; dh1 = dh2 + dh1_mlpnorm.
        //   dh1n in sm->fn_xhat; xhat = mlpn_xhat; out dh1_mlpnorm -> sm->dr.
        mb_rmsnorm_bwd(&sm->fn_xhat[0][0], &a->mlpn_xhat[0][0], &a->mlpn_r[0], L.mlpn_w,
                       &sm->dr[0][0], G.mlpn_w, sm->red, mb::kD);
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            sm->dh[s][j] += sm->dr[s][j];   // dh1 = dh2(residual) + mlp_norm path
        }
        __syncthreads();
        // h1 = x + mixer(mixer_norm(x)). dh1 = sm->dh.
        //   dx = dh1 (residual) + rmsnorm_bwd(mixer_bwd(dh1)).
        // recompute xn = mixer_norm(hin) -> sm->dr (xhat * mix_w).
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            sm->dr[s][j] = a->mixn_xhat[s][j] * L.mixn_w[j];   // xn
        }
        __syncthreads();
        // mixer bwd: dmix_out = dh1 (sm->dh) ; xn in sm->dr ; out dxn -> sm->fn_xhat.
        mb_mixer_bwd(L, G, &sm->dr[0][0], a, &sm->dh[0][0], &sm->fn_xhat[0][0], sm);
        // rmsnorm_mix bwd: dx_mixnorm = rmsnorm_bwd(dxn) ; dx = dh1 + dx_mixnorm.
        mb_rmsnorm_bwd(&sm->fn_xhat[0][0], &a->mixn_xhat[0][0], &a->mixn_r[0], L.mixn_w,
                       &sm->dr[0][0], G.mixn_w, sm->red, mb::kD);
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, j = idx % mb::kD;
            sm->dh[s][j] += sm->dr[s][j];   // dx = dh1(residual) + mixer_norm path
        }
        __syncthreads();
        // sm->dh now = grad wrt the previous block's output (or the embedding for li=0).
    }

    // ── embedding backward: dh = grad wrt h0 [kSeq,d]. ──
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s) {
            float d = sm->dh[s][j];
            g.tok[(int64_t)tokens_s[s] * mb::kD + j] += d;
            g.pos[(int64_t)s * mb::kD + j] += d;
        }
    }
    __syncthreads();
}

}}} // namespace sg::fused::sm90

#endif  // SG_FUSED_SM90_MODEL_STAGE_MAMBA3_CUH_
