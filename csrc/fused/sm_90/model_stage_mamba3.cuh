#ifndef SG_FUSED_SM90_MODEL_STAGE_MAMBA3_CUH_
#define SG_FUSED_SM90_MODEL_STAGE_MAMBA3_CUH_
// ============================================================================
// csrc/fused/sm_90/model_stage_mamba3.cuh — PHASE 2 of the TRUE L3 fused
// megakernel: the REAL Mamba (selective-SSM) forward + backward as in-kernel
// stages of the persistent megakernel. ONE HEADER PER MODEL (COMPONENT_CONTRACT).
//
// HONESTY: this is the genuine eager architecture (grokking_race_v2.py
// _raw_model -> MambaModel / SelectiveSSMLayer), transcribed LINE-FOR-LINE from
// the verified PyTorch oracle in tests/hw/mamba_oracle.py, which is asserted
// bit-identical (fp64, ~1e-15 rel) to torch.autograd for the loss and EVERY one
// of the 28 parameter gradients — INCLUDING the selective-scan backward (the
// reverse-time recurrence for dA_log,dB,dC,ddt,dx, derived in the oracle with the
// derivation written out). There is NO placeholder math on this path.
//
// ARCHITECTURE (exact; cites grokking_race_v2.py lines):
//   MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2)         (:451-462, :477)
//     tok = Embedding(99,128) (:454); pos = Embedding(8,128) (:454), arange(8)
//     2 x SelectiveSSMLayer(d=128) (:455):
//       d_inner=256(=d*2), state_dim=16, dt_rank=max(d/16,1)=8           (:408-411)
//       in_proj  = Linear(128, 2*256=512, bias=False)                    (:412)
//       conv1d   = Conv1d(256,256,k=3,pad=1,groups=256,bias=True) NON-CAUSAL (:413)
//       x_proj   = Linear(256, 8+2*16=40, bias=False)                    (:415)
//       dt_proj  = Linear(8, 256, bias=True)                             (:416)
//       A_log    = log(arange(1,17)).expand(256,16); A = -exp(A_log)     (:417-418,:423)
//       D        = ones(256)                                             (:419)
//       out_proj = Linear(256, 128, bias=False)                          (:420)
//       norm     = LayerNorm(128) (POST, per layer)                      (:421)
//       forward (:442-449):
//         residual=x; xz=in_proj(x); x_main,z = xz.chunk(2,-1)
//         x_main = SiLU(conv1d(x_main))      (depthwise NON-causal k=3 pad=1)
//         dt,B,C = x_proj(x_main).split([8,16,16],-1)
//         y = selective_scan(x_main, dt_proj(dt), B, C)   (softplus INSIDE scan)
//         y = out_proj((y + x_main*D) * SiLU(z))
//         return norm(y + residual)
//     norm = LayerNorm(128) (:456 final); out = Linear(128,97) (:456 head, p=97!)
//     forward: h=tok+pos; for l: h=l(h); return out(norm(h[:,-1,:])) (:460-462)
//   loss = cross_entropy(logits, targets) (mean over batch; :745).
//
//   selective_scan(x,dt,B,C)  (:422-441, the PYTHON FALLBACK the race runs):
//     A=-exp(A_log); dt=softplus(dt); h=0;
//     for t: h = exp(dt_t*A)*h + (dt_t*B_t)*x_t ; y_t = sum_s C_t*h_t.
//
// NUMERICS (verified in the oracle; see mamba_oracle.py):
//   * SiLU(x)=x*sigmoid(x); SiLU'(x)=sig(x)*(1+x*(1-sig(x))).
//   * softplus(x)=log1p(exp(x)); softplus'(x)=sigmoid(x).
//   * Conv1d NON-causal: y[t]=b + W0*x[t-1] + W1*x[t] + W2*x[t+1] (zero-pad).
//   * A=-exp(A_log) -> dA_log = dA*A.
//   * LayerNorm eps=1e-5; dx uses the two mean-correction terms.
//   * CE MEAN over B -> dlogits=(softmax-onehot)/B; head width p=97.
//
// PARALLELIZATION (Option B, batch-parallel, deterministic — IDENTICAL to the
// decoder PHASE-1 design):
//   * Each CTA owns a FIXED contiguous batch slice (by blockIdx.x — NOT the
//     work-steal queue; fp32 sums aren't associative so a varying batch->CTA
//     grouping is non-deterministic). The whole CTA cooperates on ONE sample at a
//     time and iterates its slice SEQUENTIALLY.
//   * Per-sample, each dW element has a SINGLE owner thread that does
//     partial[e] += contrib across the slice's samples in fixed order
//     (__syncthreads between samples) — single-writer, no atomics, deterministic.
//   * THE SCAN EXPLOIT (seq=8, trivially short): the selective scan is a
//     SEQUENTIAL recurrence over seq=8 per (batch, channel j in [0,256)). One
//     thread owns channel j, holds the state h[STATE=16] in REGISTERS, and
//     UNROLLS t=0..7. The full h[seq][d_inner][state] (~128 KB/sample) does NOT
//     fit smem — keeping h per-channel in registers is the justification for the
//     thread-per-channel scan. The BACKWARD recomputes the per-channel forward
//     scan keeping h_hist[seq+1][state] in REGISTERS (seq=8 trivial — NO
//     checkpoint, unlike the long-sequence scan_backward_kernel in csrc/scan),
//     then reverse-scans the dA,dB,dC,ddt,dx recurrence.
//   * One grid barrier, then a deterministic cross-CTA reduce (ascending CTA
//     index), another barrier, then apply_optimizer<Opt>. (Composition lives in
//     fused_mamba_megakernel.cuh — see INTEGRATION-MAMBA.md.)
//
// SMEM BUDGET PER CTA — HONEST DEVIATION FROM THE DECODER (documented):
//   The decoder's <48 KB static footprint relied on recompute-in-backward; that
//   trick exists ONLY to clear the 48 KB static-smem cliff. Mamba's d_inner=256
//   makes EVEN ONE layer's activation set (5x [kSeq,d_inner] + caches ≈ 45 KB)
//   bump the cliff, and the full live set is ~74 KB regardless — so recompute
//   buys nothing here. We therefore CACHE BOTH LAYERS' forward activations (the
//   clean, bug-free choice) and declare the CTA smem DYNAMIC with a
//   cudaFuncSetAttribute(MaxDynamicSharedMemorySize) opt-in (sm_90 has ~228 KB
//   smem/SM; one persistent block/SM still places, occupancy>=1 holds, NO CUDA
//   graphs). sizeof(MambaSampleSmem) is computed + reported by mamba3_layout.cuh
//   (kMambaSmemBytes) and is the value the launcher opts into. The scan state is
//   NOT in smem — it is per-thread registers (the seq=8 exploit), which is what
//   keeps the smem from exploding to the 128 KB the naive h-in-smem would need.
//   See INTEGRATION-MAMBA.md for the dynamic-smem launch spec (built in a later
//   composition step; this header is launcher-agnostic).
//
// fp32 compute is the correctness baseline. bf16 compute is a flagged TODO (it
// would default to THIS fp32 path) — see the TODO(bf16) markers.
// ============================================================================

#include "csrc/fused/megakernel_common.cuh"
#include "csrc/fused/sm_90/mamba3_layout.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

namespace sg { namespace fused { namespace sm90 {

// ── Mamba compile-time shape constants (mirror tests/hw/mamba_oracle.py and
//    csrc/fused/sm_90/mamba3_layout.cuh; static_asserts there guard the count +
//    total against named_parameters()). ───────────────────────────────────────
namespace mb {
constexpr int kVocab  = SG_MB_VOCAB;   // 99  (tok embedding rows)
constexpr int kPHead  = SG_MB_PHEAD;   // 97  (head width; DISTINCT from vocab)
constexpr int kD      = SG_MB_D;       // 128 (d_model)
constexpr int kLayers = SG_MB_LAYERS;  // 2
constexpr int kSeq    = SG_MB_SEQ;     // 8
constexpr int kDInner = SG_MB_DINNER;  // 256 (d*2)
constexpr int kState  = SG_MB_STATE;   // 16
constexpr int kDtRank = SG_MB_DTRANK;  // 8
constexpr int kDbc    = kDtRank + 2 * kState;  // 40 (x_proj out width)
constexpr int kConvK  = SG_MB_CONVK;   // 3
constexpr float kLnEps = 1e-5f;        // torch LayerNorm default
}  // namespace mb

// ── Elementwise activations + derivatives (the real F.silu / F.softplus). ─────
__device__ __forceinline__ float mb_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
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
//    BOTH layers' forward activations are cached (see SMEM BUDGET note). The scan
//    state h[kState] is NOT here (per-thread registers). x_main = silu(conv) is
//    derived on the fly (one fewer cached buffer). ─────────────────────────────
struct MambaSampleSmem {
    // Cross-layer chain: input to each layer (= residual x for that layer), and
    // the final-layer output feeding the head norm.
    float layer_in[mb::kLayers][mb::kSeq][mb::kD];   // 2*8*128
    float final_in[mb::kSeq][mb::kD];                // 8*128
    // Per-layer cached forward activations (both layers): the values the backward
    // reads. x_main is recomputed as silu(conv) where needed.
    struct LayerAct {
        float x_main_raw[mb::kSeq][mb::kDInner];     // pre-conv (in_proj first half)
        float z[mb::kSeq][mb::kDInner];              // gate branch (in_proj 2nd half)
        float conv[mb::kSeq][mb::kDInner];           // PRE-silu conv output
        float dt_pre[mb::kSeq][mb::kDInner];         // dt_proj out (PRE-softplus)
        float y_scan[mb::kSeq][mb::kDInner];         // selective-scan output
        float Bmat[mb::kSeq][mb::kState];            // scan B
        float Cmat[mb::kSeq][mb::kState];            // scan C
        float ln_xhat[mb::kSeq][mb::kD];             // per-layer norm xhat
        float ln_inv[mb::kSeq];                      // per-layer norm inv_std
    } act[mb::kLayers];
    // Final-norm caches (last position used) + head logits.
    float fn_xhat[mb::kSeq][mb::kD];
    float fn_inv[mb::kSeq];
    float logits[mb::kPHead];
    // Backward adjoint scratch (reused across the per-layer chain). Named by their
    // FIRST use; the comments at each backward step note the reuse.
    float dh[mb::kSeq][mb::kD];                      // running grad wrt block output
    float dr[mb::kSeq][mb::kD];                      // LN-bwd output (dy_out / dx_residual)
    float adj_a[mb::kSeq][mb::kDInner];              // dy_gated -> dx_main accumulator
    float adj_b[mb::kSeq][mb::kDInner];              // dy_scan / dz / dconv scratch
    float adj_c[mb::kSeq][mb::kDInner];              // ddt_pre / dx_main_raw scratch
    float dbc[mb::kSeq][mb::kDbc];                   // dx_dbc = cat[ddt_raw,dB,dC]
    float dBmat[mb::kSeq][mb::kState];               // scan-bwd dB (reduced over channels)
    float dCmat[mb::kSeq][mb::kState];               // scan-bwd dC (reduced over channels)
    // Reductions across threads (row stats: mean/max/sum). 256 threads.
    float red[256];
};
// SAFETY: the launcher opts into kMambaSmemBytes of DYNAMIC smem (mamba3_layout.cuh)
// via cudaFuncSetAttribute. Static __shared__ would have the compiler enforce the
// size; the dynamic path does not — so PIN the layout constant to the actual struct
// here (both symbols are in scope). A field added to MambaSampleSmem without
// updating kMambaSmemFloats would otherwise SILENTLY under-allocate smem at launch.
static_assert((int64_t)sizeof(MambaSampleSmem) == kMambaSmemBytes,
              "model_stage_mamba3: sizeof(MambaSampleSmem) drifted from "
              "kMambaSmemBytes (mamba3_layout.cuh). Update kMambaSmemFloats.");

// ── Typed views over the flat weight blob using the generated offsets. ─────────
struct MambaWeights {
    const float* tok;     // [kVocab, kD]
    const float* pos;     // [kSeq, kD]
    struct Layer {
        const float* A_log;     // [d_inner, state]
        const float* D;         // [d_inner]
        const float* in_w;      // [2*d_inner, d]
        const float* conv_w;    // [d_inner, 1, 3]
        const float* conv_b;    // [d_inner]
        const float* x_proj_w;  // [dbc, d_inner]
        const float* dt_proj_w; // [d_inner, dt_rank]
        const float* dt_proj_b; // [d_inner]
        const float* out_w;     // [d, d_inner]
        const float* n_w;       // [d]
        const float* n_b;       // [d]
    } layer[mb::kLayers];
    const float* norm_w;  // [d]
    const float* norm_b;  // [d]
    const float* out_w;   // [kPHead, d]
    const float* out_b;   // [kPHead]
};

__device__ __forceinline__ MambaWeights mb_bind(const float* p) {
    MambaWeights w;
    int i = 0;
    w.tok = p + kMambaOffsets[i++];
    w.pos = p + kMambaOffsets[i++];
    for (int li = 0; li < mb::kLayers; ++li) {
        w.layer[li].A_log     = p + kMambaOffsets[i++];
        w.layer[li].D         = p + kMambaOffsets[i++];
        w.layer[li].in_w      = p + kMambaOffsets[i++];
        w.layer[li].conv_w    = p + kMambaOffsets[i++];
        w.layer[li].conv_b    = p + kMambaOffsets[i++];
        w.layer[li].x_proj_w  = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_w = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_b = p + kMambaOffsets[i++];
        w.layer[li].out_w     = p + kMambaOffsets[i++];
        w.layer[li].n_w       = p + kMambaOffsets[i++];
        w.layer[li].n_b       = p + kMambaOffsets[i++];
    }
    w.norm_w = p + kMambaOffsets[i++];
    w.norm_b = p + kMambaOffsets[i++];
    w.out_w  = p + kMambaOffsets[i++];
    w.out_b  = p + kMambaOffsets[i++];
    return w;
}

struct MambaGrad {
    float* tok; float* pos;
    struct Layer { float* A_log; float* D; float* in_w; float* conv_w;
        float* conv_b; float* x_proj_w; float* dt_proj_w; float* dt_proj_b;
        float* out_w; float* n_w; float* n_b; } layer[mb::kLayers];
    float* norm_w; float* norm_b; float* out_w; float* out_b;
};
__device__ __forceinline__ MambaGrad mb_bind_grad(float* p) {
    MambaGrad w; int i = 0;
    w.tok = p + kMambaOffsets[i++];
    w.pos = p + kMambaOffsets[i++];
    for (int li = 0; li < mb::kLayers; ++li) {
        w.layer[li].A_log     = p + kMambaOffsets[i++];
        w.layer[li].D         = p + kMambaOffsets[i++];
        w.layer[li].in_w      = p + kMambaOffsets[i++];
        w.layer[li].conv_w    = p + kMambaOffsets[i++];
        w.layer[li].conv_b    = p + kMambaOffsets[i++];
        w.layer[li].x_proj_w  = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_w = p + kMambaOffsets[i++];
        w.layer[li].dt_proj_b = p + kMambaOffsets[i++];
        w.layer[li].out_w     = p + kMambaOffsets[i++];
        w.layer[li].n_w       = p + kMambaOffsets[i++];
        w.layer[li].n_b       = p + kMambaOffsets[i++];
    }
    w.norm_w = p + kMambaOffsets[i++];
    w.norm_b = p + kMambaOffsets[i++];
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
//    (+ b). One thread per (s,o), strided. ──────────────────────────────────────
__device__ __forceinline__ void mb_linear(
        const float* __restrict__ x, int in_dim,
        const float* __restrict__ W, const float* __restrict__ b, int out_dim,
        float* __restrict__ y) {
    const int total = mb::kSeq * out_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int s = idx / out_dim, o = idx % out_dim;
        const float* xr = x + s * in_dim;
        const float* Wr = W + (int64_t)o * in_dim;
        float acc = (b != nullptr) ? b[o] : 0.0f;
        #pragma unroll 4
        for (int k = 0; k < in_dim; ++k) acc += xr[k] * Wr[k];
        y[s * out_dim + o] = acc;
    }
    __syncthreads();
}

// ── LayerNorm forward (last dim width=d). Caches xhat + inv_std. ──────────────
__device__ __forceinline__ void mb_layernorm_fwd(
        const float* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, float* __restrict__ y,
        float* __restrict__ xhat_out, float* __restrict__ inv_out, float* red) {
    for (int s = 0; s < mb::kSeq; ++s) {
        const float* xr = x + s * mb::kD;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) sum += xr[j];
        float mean = mb_block_sum(sum, red) / (float)mb::kD;
        float vs = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) { float c = xr[j] - mean; vs += c * c; }
        float var = mb_block_sum(vs, red) / (float)mb::kD;
        float inv = rsqrtf(var + mb::kLnEps);
        if (threadIdx.x == 0) inv_out[s] = inv;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float xh = (xr[j] - mean) * inv;
            xhat_out[s * mb::kD + j] = xh;
            y[s * mb::kD + j] = xh * gamma[j] + beta[j];
        }
        __syncthreads();
    }
}

// ── LayerNorm backward (mirrors decoder dec_layernorm_bwd; accumulates affine). ─
__device__ void mb_layernorm_bwd(
        const float* __restrict__ dy, const float* __restrict__ xhat,
        const float* __restrict__ inv, const float* __restrict__ gamma,
        float* __restrict__ dx_out, float* __restrict__ gw, float* __restrict__ gb,
        float* red) {
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float dgw = 0.0f, dgb = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s) {
            float d = dy[s * mb::kD + j];
            dgb += d; dgw += d * xhat[s * mb::kD + j];
        }
        gw[j] += dgw; gb[j] += dgb;
    }
    for (int s = 0; s < mb::kSeq; ++s) {
        const float* dyr = dy + s * mb::kD;
        const float* xhr = xhat + s * mb::kD;
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = mb_block_sum(sda, red);
        sdax = mb_block_sum(sdax, red);
        float inv_s = inv[s];
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = dyr[j] * gamma[j];
            dx_out[s * mb::kD + j] =
                inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)mb::kD);
        }
        __syncthreads();
    }
}

// ── dW += dY^T @ X for a linear Y = X @ W^T (+ b). dX = dY @ W (set or add).
//    Owner-thread per (o,i) for dW; db[o]+=Σ_s dY. Mirrors decoder dec_linear_bwd.─
__device__ void mb_linear_bwd(
        const float* __restrict__ dY, const float* __restrict__ X,
        const float* __restrict__ W, int in_dim, int out_dim,
        float* __restrict__ dW, float* __restrict__ db,
        float* __restrict__ dx_out, bool set_dx) {
    const int wtot = out_dim * in_dim;
    for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
        const int o = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * out_dim + o] * X[s * in_dim + i];
        dW[(int64_t)o * in_dim + i] += acc;
    }
    if (db != nullptr) {
        for (int o = threadIdx.x; o < out_dim; o += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * out_dim + o];
            db[o] += acc;
        }
    }
    const int xtot = mb::kSeq * in_dim;
    for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
        const int s = idx / in_dim, i = idx % in_dim;
        float acc = 0.0f;
        for (int o = 0; o < out_dim; ++o) acc += dY[s * out_dim + o] * W[(int64_t)o * in_dim + i];
        if (set_dx) dx_out[s * in_dim + i] = acc;
        else dx_out[s * in_dim + i] += acc;
    }
    __syncthreads();
}

// ── CONV1D (depthwise, NON-causal, k=3, pad=1) forward. ───────────────────────
//   out[t,c] = b[c] + W[c,0,0]*in[t-1,c] + W[c,0,1]*in[t,c] + W[c,0,2]*in[t+1,c].
//   Mirrors mamba_oracle.conv1d_forward.
__device__ void mb_conv1d_fwd(const float* __restrict__ in,
                              const float* __restrict__ W,
                              const float* __restrict__ b,
                              float* __restrict__ out) {
    const int total = mb::kSeq * mb::kDInner;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx / mb::kDInner, c = idx % mb::kDInner;
        const float* Wc = W + (int64_t)c * mb::kConvK;
        float acc = b[c];
        #pragma unroll
        for (int k = 0; k < mb::kConvK; ++k) {
            const int tt = t + k - 1;   // k=0->t-1, k=1->t, k=2->t+1
            if (tt >= 0 && tt < mb::kSeq) acc += Wc[k] * in[tt * mb::kDInner + c];
        }
        out[t * mb::kDInner + c] = acc;
    }
    __syncthreads();
}

// ── CONV1D backward. dW[c,0,k]+=Σ_t dy[t,c]*in[t+k-1,c]; db[c]+=Σ_t dy[t,c];
//   dx[tt,c]=Σ_k W[c,0,k]*dy[tt-k+1,c] (in-range). Mirrors conv1d_backward.
__device__ void mb_conv1d_bwd(const float* __restrict__ dy,
                             const float* __restrict__ in,
                             const float* __restrict__ W,
                             float* __restrict__ dW, float* __restrict__ db,
                             float* __restrict__ dx_out) {
    for (int c = threadIdx.x; c < mb::kDInner; c += blockDim.x) {
        float dwk[mb::kConvK]; float dbc = 0.0f;
        #pragma unroll
        for (int k = 0; k < mb::kConvK; ++k) dwk[k] = 0.0f;
        #pragma unroll
        for (int t = 0; t < mb::kSeq; ++t) {
            float dyv = dy[t * mb::kDInner + c];
            dbc += dyv;
            #pragma unroll
            for (int k = 0; k < mb::kConvK; ++k) {
                const int tt = t + k - 1;
                if (tt >= 0 && tt < mb::kSeq) dwk[k] += dyv * in[tt * mb::kDInner + c];
            }
        }
        #pragma unroll
        for (int k = 0; k < mb::kConvK; ++k) dW[(int64_t)c * mb::kConvK + k] += dwk[k];
        db[c] += dbc;
    }
    const int total = mb::kSeq * mb::kDInner;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int tt = idx / mb::kDInner, c = idx % mb::kDInner;
        const float* Wc = W + (int64_t)c * mb::kConvK;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < mb::kConvK; ++k) {
            const int t = tt - k + 1;   // tt = t+k-1 -> t = tt-k+1
            if (t >= 0 && t < mb::kSeq) acc += Wc[k] * dy[t * mb::kDInner + c];
        }
        dx_out[tt * mb::kDInner + c] = acc;
    }
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  SELECTIVE SCAN forward — THE CORE EXPLOIT. One thread owns channel j; holds
//  the state h[kState] in REGISTERS and unrolls t=0..kSeq-1. Reads x_main (=silu
//  (conv)) [t][j], dt_pre [t][j] (PRE-softplus), Bmat[t][s], Cmat[t][s],
//  A_log[j][s]. Writes y_scan[t][j]. Mirrors selective_scan_forward.
//    A[s]=-exp(A_log[j,s]); dt_t=softplus(dt_pre[t,j]);
//    h[s] = exp(dt_t*A[s])*h[s] + dt_t*Bmat[t,s]*x[t,j]; y[t,j]=Σ_s Cmat[t,s]*h[s].
//  x_main is passed in (caller materialises silu(conv) into a buffer). ──────────
__device__ void mb_scan_fwd(const float* __restrict__ A_log,
                           const float* __restrict__ x_main,  // [kSeq,d_inner]
                           const MambaSampleSmem::LayerAct* a,
                           float* __restrict__ y_scan_out) {
    for (int j = threadIdx.x; j < mb::kDInner; j += blockDim.x) {
        float A[mb::kState];
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s)
            A[s] = -__expf(A_log[(int64_t)j * mb::kState + s]);
        float h[mb::kState];
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s) h[s] = 0.0f;
        #pragma unroll
        for (int t = 0; t < mb::kSeq; ++t) {
            const float xv = x_main[t * mb::kDInner + j];
            const float dt_t = mb_softplus(a->dt_pre[t][j]);
            float yacc = 0.0f;
            #pragma unroll
            for (int s = 0; s < mb::kState; ++s) {
                const float adec = __expf(dt_t * A[s]);
                h[s] = adec * h[s] + dt_t * a->Bmat[t][s] * xv;
                yacc += a->Cmat[t][s] * h[s];
            }
            y_scan_out[t * mb::kDInner + j] = yacc;
        }
    }
    __syncthreads();
}

// SELECTIVE SCAN backward. Per channel j (one thread): RECOMPUTE the forward
// keeping h_hist[kSeq+1][kState] in registers (seq=8 — NO checkpoint), then
// reverse-scan (derivation transcribed from mamba_oracle.selective_scan_backward):
//   gh[s] += Cmat[t,s]*dy[t,j]   (full dL/dh_t after adding y_t's direct term)
//   dC_js[t,s] = h_t[s]*dy[t,j];  dB_js[t,s] = dt_t*x*gh[s]
//   dx_main_scan[t,j] = Σ_s Bmat[t,s]*dt_t*gh[s]
//   ddt_post = Σ_s (A[s]*a_t*h_{t-1}+Bmat[t,s]*x)*gh[s]
//   dA[s] += dt_t*a_t*h_{t-1}*gh[s];  gh[s] <- a_t*gh[s]
//   ddt_pre[t,j] = ddt_post * softplus'(dt_pre[t,j])
//   dA_log[j,s]  += dA[s]*A[s]   (A=-exp(A_log))   (owner j -> plain write)
// dBmat/dCmat are shared across channels -> reduced over threads in a FIXED order
// (mb_block_sum, ascending lane/warp = deterministic) into smem dBmat/dCmat.
__device__ void mb_scan_bwd(const float* __restrict__ A_log,
                           const float* __restrict__ x_main,    // [kSeq,d_inner]
                           const MambaSampleSmem::LayerAct* a,
                           const float* __restrict__ dy_scan,   // [kSeq,d_inner]
                           float* __restrict__ dx_main_scan,    // [kSeq,d_inner] SET
                           float* __restrict__ ddt_pre,         // [kSeq,d_inner] SET
                           float* __restrict__ dBmat,           // [kSeq,kState] += (smem; zeroed by caller)
                           float* __restrict__ dCmat,           // [kSeq,kState] += (smem; zeroed by caller)
                           float* __restrict__ g_A_log,         // [d_inner,state] partial (+=)
                           float* __restrict__ red) {
    float dB_loc[mb::kSeq][mb::kState];
    float dC_loc[mb::kSeq][mb::kState];
    const bool active = (threadIdx.x < mb::kDInner);
    if (active) {
        const int j = threadIdx.x;
        float A[mb::kState], hh[mb::kSeq + 1][mb::kState];
        float dt_save[mb::kSeq], a_save[mb::kSeq][mb::kState];
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s) {
            A[s] = -__expf(A_log[(int64_t)j * mb::kState + s]);
            hh[0][s] = 0.0f;
        }
        // forward recompute -> hh[t+1] = h_t (hh[0] = h_{-1} = 0).
        #pragma unroll
        for (int t = 0; t < mb::kSeq; ++t) {
            const float xv = x_main[t * mb::kDInner + j];
            const float dt_t = mb_softplus(a->dt_pre[t][j]);
            dt_save[t] = dt_t;
            #pragma unroll
            for (int s = 0; s < mb::kState; ++s) {
                const float adec = __expf(dt_t * A[s]);
                a_save[t][s] = adec;
                hh[t + 1][s] = adec * hh[t][s] + dt_t * a->Bmat[t][s] * xv;
            }
        }
        // reverse scan.
        float gh[mb::kState], dA[mb::kState];
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s) { gh[s] = 0.0f; dA[s] = 0.0f; }
        #pragma unroll
        for (int t = mb::kSeq - 1; t >= 0; --t) {
            const float xv = x_main[t * mb::kDInner + j];
            const float dt_t = dt_save[t];
            const float dyv = dy_scan[t * mb::kDInner + j];
            float dx_acc = 0.0f, ddt_acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < mb::kState; ++s) {
                const float h_t = hh[t + 1][s];
                const float h_tm1 = hh[t][s];
                const float adec = a_save[t][s];
                const float cc = a->Cmat[t][s];
                const float bb = a->Bmat[t][s];
                dC_loc[t][s] = h_t * dyv;
                gh[s] += cc * dyv;
                dB_loc[t][s] = dt_t * xv * gh[s];
                dx_acc += bb * dt_t * gh[s];
                ddt_acc += (A[s] * adec * h_tm1 + bb * xv) * gh[s];
                dA[s] += dt_t * adec * h_tm1 * gh[s];
                gh[s] = adec * gh[s];
            }
            dx_main_scan[t * mb::kDInner + j] = dx_acc;
            ddt_pre[t * mb::kDInner + j] = ddt_acc * mb_softplus_grad(a->dt_pre[t][j]);
        }
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s)
            g_A_log[(int64_t)j * mb::kState + s] += dA[s] * A[s];
    } else {
        #pragma unroll
        for (int t = 0; t < mb::kSeq; ++t)
            #pragma unroll
            for (int s = 0; s < mb::kState; ++s) { dB_loc[t][s] = 0.0f; dC_loc[t][s] = 0.0f; }
    }
    // Reduce dB_loc/dC_loc across channels (threads) in fixed order -> dBmat/dCmat.
    #pragma unroll
    for (int t = 0; t < mb::kSeq; ++t) {
        #pragma unroll
        for (int s = 0; s < mb::kState; ++s) {
            float sb = mb_block_sum(dB_loc[t][s], red);
            float sc = mb_block_sum(dC_loc[t][s], red);
            if (threadIdx.x == 0) { dBmat[t * mb::kState + s] += sb; dCmat[t * mb::kState + s] += sc; }
        }
    }
    __syncthreads();
}

// ── Materialise x_main = silu(conv) for layer li into `out` [kSeq,d_inner]. ────
__device__ __forceinline__ void mb_make_x_main(const MambaSampleSmem::LayerAct* a,
                                               float* __restrict__ out) {
    const int total = mb::kSeq * mb::kDInner;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
        out[idx] = mb_silu(a->conv[idx / mb::kDInner][idx % mb::kDInner]);
    __syncthreads();
}

// ────────────────────────────────────────────────────────────────────────────
//  FORWARD for one sample (CTA-cooperative). Mirrors mamba_oracle.mamba_forward.
//  Caches BOTH layers' forward activations into sm->act[li]; returns sample NLL.
// ────────────────────────────────────────────────────────────────────────────
__device__ float mb_forward_sample(const MambaWeights& w, const int* tokens_s,
                                    int target, MambaSampleSmem* sm) {
    // Embedding + positional.
    for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
        const int s = idx / mb::kD, j = idx % mb::kD;
        sm->layer_in[0][s][j] = w.tok[(int64_t)tokens_s[s] * mb::kD + j]
                              + w.pos[(int64_t)s * mb::kD + j];
    }
    __syncthreads();

    for (int li = 0; li < mb::kLayers; ++li) {
        const MambaWeights::Layer& L = w.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        float* hin = &sm->layer_in[li][0][0];   // residual x
        // xz = in_proj(x); split in_w rows [0,d_inner)->x_main_raw, [d_inner,2di)->z.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, o = idx % mb::kDInner;
            const float* xr = hin + s * mb::kD;
            const float* Wr0 = L.in_w + (int64_t)o * mb::kD;
            const float* Wr1 = L.in_w + (int64_t)(mb::kDInner + o) * mb::kD;
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < mb::kD; ++k) { float xv = xr[k]; a0 += xv * Wr0[k]; a1 += xv * Wr1[k]; }
            a->x_main_raw[s][o] = a0;
            a->z[s][o] = a1;
        }
        __syncthreads();
        // conv1d (NON-causal) -> a->conv (PRE-silu, cached for bwd).
        mb_conv1d_fwd(&a->x_main_raw[0][0], L.conv_w, L.conv_b, &a->conv[0][0]);
        // x_main = silu(conv) materialised into adj_a scratch (transient this layer).
        mb_make_x_main(a, &sm->adj_a[0][0]);
        // x_proj(x_main) -> dt_raw (stage in dbc scratch), Bmat, Cmat.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDbc; idx += blockDim.x) {
            const int s = idx / mb::kDbc, o = idx % mb::kDbc;
            const float* xr = &sm->adj_a[s][0];   // x_main
            const float* Wr = L.x_proj_w + (int64_t)o * mb::kDInner;
            float acc = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < mb::kDInner; ++k) acc += xr[k] * Wr[k];
            if (o < mb::kDtRank) sm->dbc[s][o] = acc;   // dt_raw staged in dbc[:, :dt_rank]
            else if (o < mb::kDtRank + mb::kState) a->Bmat[s][o - mb::kDtRank] = acc;
            else a->Cmat[s][o - mb::kDtRank - mb::kState] = acc;
        }
        __syncthreads();
        // dt_pre = dt_proj(dt_raw) + bias  -> a->dt_pre (cached).
        // mb_linear over the dt_raw staged in dbc[:, :dt_rank] (row stride kDbc!).
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, o = idx % mb::kDInner;
            const float* xr = &sm->dbc[s][0];   // dt_raw occupies [0,dt_rank)
            const float* Wr = L.dt_proj_w + (int64_t)o * mb::kDtRank;
            float acc = L.dt_proj_b[o];
            #pragma unroll
            for (int k = 0; k < mb::kDtRank; ++k) acc += xr[k] * Wr[k];
            a->dt_pre[s][o] = acc;
        }
        __syncthreads();
        // selective scan -> a->y_scan (cached).
        mb_scan_fwd(L.A_log, &sm->adj_a[0][0], a, &a->y_scan[0][0]);
        // gate+skip+out_proj: y_gated=(y_scan + x_main*D)*silu(z); stage in adj_b.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, c = idx % mb::kDInner;
            float yv = a->y_scan[s][c] + sm->adj_a[s][c] * L.D[c];   // x_main = adj_a
            sm->adj_b[s][c] = yv * mb_silu(a->z[s][c]);              // adj_b := y_gated
        }
        __syncthreads();
        // r = out_proj(y_gated) + residual; staged into adj_c[:, :d] then LN -> dst.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
            const int s = idx / mb::kD, o = idx % mb::kD;
            const float* Wr = L.out_w + (int64_t)o * mb::kDInner;
            const float* yg = &sm->adj_b[s][0];
            float acc = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < mb::kDInner; ++c) acc += yg[c] * Wr[c];
            sm->adj_c[s][o] = acc + hin[s * mb::kD + o];   // adj_c := r (uses [.,:d])
        }
        __syncthreads();
        float* dst = (li + 1 < mb::kLayers) ? &sm->layer_in[li + 1][0][0]
                                            : &sm->final_in[0][0];
        mb_layernorm_fwd(&sm->adj_c[0][0], L.n_w, L.n_b, dst,
                         &a->ln_xhat[0][0], &a->ln_inv[0], sm->red);
    }

    // Final norm (LAST position) + head -> logits -> NLL.
    const float* hlast = &sm->final_in[mb::kSeq - 1][0];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) sum += hlast[j];
    float mean = mb_block_sum(sum, sm->red) / (float)mb::kD;
    float vs = 0.0f;
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) { float c = hlast[j] - mean; vs += c * c; }
    float var = mb_block_sum(vs, sm->red) / (float)mb::kD;
    float inv = rsqrtf(var + mb::kLnEps);
    if (threadIdx.x == 0) sm->fn_inv[0] = inv;
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float xh = (hlast[j] - mean) * inv;
        sm->fn_xhat[mb::kSeq - 1][j] = xh;
        sm->adj_a[0][j] = xh * w.norm_w[j] + w.norm_b[j];   // hn staged in adj_a[0]
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

// ────────────────────────────────────────────────────────────────────────────
//  BACKWARD for one sample (CTA-cooperative). Assumes mb_forward_sample ran for
//  THIS sample (act[*] caches + final/fn caches + logits populated). Accumulates
//  every weight grad into the CTA partial `g`. Mirrors mamba_oracle.mamba_backward
//  + ssm_layer_backward, in the documented order 1..13.
// ────────────────────────────────────────────────────────────────────────────
__device__ void mb_backward_sample(const MambaWeights& w, const MambaGrad& g,
                                    const int* tokens_s, int target, int B,
                                    MambaSampleSmem* sm) {
    // ── CE backward: dlogits = (softmax - onehot)/B. ──
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
    // ── head: logits = hn @ out_w^T + out_b; hn = fn_xhat[last]*norm_w + norm_b. ──
    for (int o = threadIdx.x; o < mb::kPHead; o += blockDim.x) {
        float dl = sm->logits[o];
        g.out_b[o] += dl;
        float* gwrow = g.out_w + (int64_t)o * mb::kD;
        const float* xh = &sm->fn_xhat[mb::kSeq - 1][0];
        #pragma unroll 4
        for (int j = 0; j < mb::kD; ++j) gwrow[j] += dl * (xh[j] * w.norm_w[j] + w.norm_b[j]);
    }
    __syncthreads();
    // dhn[j] -> dr row 0 (scratch).
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < mb::kPHead; ++o) acc += sm->logits[o] * w.out_w[(int64_t)o * mb::kD + j];
        sm->dr[0][j] = acc;
    }
    __syncthreads();
    // ── final-norm bwd (single row last): accumulate norm g/b, dx -> dh. ──
    for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
        float d = sm->dr[0][j], xh = sm->fn_xhat[mb::kSeq - 1][j];
        g.norm_b[j] += d; g.norm_w[j] += d * xh;
    }
    __syncthreads();
    {
        const float* dyr = &sm->dr[0][0];
        const float* xhr = &sm->fn_xhat[mb::kSeq - 1][0];
        float sda = 0.0f, sdax = 0.0f;
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            sda += dxhat; sdax += dxhat * xhr[j];
        }
        sda = mb_block_sum(sda, sm->red);
        sdax = mb_block_sum(sdax, sm->red);
        float inv_s = sm->fn_inv[0];
        for (int j = threadIdx.x; j < mb::kD; j += blockDim.x) {
            float dxhat = dyr[j] * w.norm_w[j];
            float dlast = inv_s * (dxhat - (sda + xhr[j] * sdax) / (float)mb::kD);
            for (int s = 0; s < mb::kSeq; ++s) sm->dh[s][j] = 0.0f;
            sm->dh[mb::kSeq - 1][j] = dlast;
        }
        __syncthreads();
    }
    // dh = grad wrt last layer's OUTPUT [kSeq,d] (only last pos nonzero).

    for (int li = mb::kLayers - 1; li >= 0; --li) {
        const MambaWeights::Layer& L = w.layer[li];
        const MambaGrad::Layer& G = g.layer[li];
        MambaSampleSmem::LayerAct* a = &sm->act[li];
        float* hin = &sm->layer_in[li][0][0];   // residual x for this layer
        // (1) out = LN(r) -> dr; accumulate per-layer norm g/b. dr lives in sm->dr.
        mb_layernorm_bwd(&sm->dh[0][0], &a->ln_xhat[0][0], &a->ln_inv[0],
                         L.n_w, &sm->dr[0][0], G.n_w, G.n_b, sm->red);
        // (2) r = y_out + residual(=hin): dy_out = dr ; dx_residual = dr (kept in dr).
        // (3) out_proj bwd: need y_gated = (y_scan + x_main*D)*silu(z). Recompute
        //     y_gated into adj_b (x_main = silu(conv) into adj_a first).
        mb_make_x_main(a, &sm->adj_a[0][0]);   // adj_a := x_main
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, c = idx % mb::kDInner;
            float yv = a->y_scan[s][c] + sm->adj_a[s][c] * L.D[c];
            sm->adj_b[s][c] = yv * mb_silu(a->z[s][c]);   // adj_b := y_gated
        }
        __syncthreads();
        //   dW_out += dy_out^T @ y_gated ; dy_gated = dy_out @ out_w -> adj_a
        //   (x_main no longer needed from adj_a until the dx_main paths below;
        //   we re-materialise x_main when needed — but D-skip path needs x_main
        //   NOW. So compute the gate bwd that consumes x_main BEFORE clobbering
        //   adj_a. Reorder: do out_proj dW first (reads adj_b only), then gate
        //   bwd (reads adj_a=x_main, adj_b=y_gated, z), producing dy_scan + dz +
        //   D-path + dD, THEN overwrite adj_a with dy_gated? No — gate bwd needs
        //   dy_gated. So compute dy_gated into adj_c (free), keep adj_a=x_main.)
        {
            const float* dY = &sm->dr[0][0];        // dy_out [kSeq,d]
            const float* yg = &sm->adj_b[0][0];     // y_gated [kSeq,d_inner]
            const int wtot = mb::kD * mb::kDInner;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / mb::kDInner, c = idx % mb::kDInner;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * mb::kD + o] * yg[s * mb::kDInner + c];
                G.out_w[(int64_t)o * mb::kDInner + c] += acc;
            }
            __syncthreads();
            // dy_gated[s,c] = Σ_o dy_out[s,o]*out_w[o,c] -> adj_c.
            const int xtot = mb::kSeq * mb::kDInner;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / mb::kDInner, c = idx % mb::kDInner;
                float acc = 0.0f;
                for (int o = 0; o < mb::kD; ++o) acc += dY[s * mb::kD + o] * L.out_w[(int64_t)o * mb::kDInner + c];
                sm->adj_c[s][c] = acc;   // adj_c := dy_gated
            }
            __syncthreads();
        }
        // (4-6) gate+skip bwd. y_gated = y_skip*silu(z); y_skip = y_scan + x_main*D.
        //   sz = silu(z); dy_skip = dy_gated*sz; dsz = dy_gated*y_skip; dz = dsz*silu'(z).
        //   dy_scan = dy_skip ; dx_main(D-path) = dy_skip*D ; dD += Σ dy_skip*x_main.
        //   Layout: adj_a = x_main, adj_b = y_gated, adj_c = dy_gated.
        //   Write dz into adj_b (y_gated consumed here), dy_scan into... we need
        //   dy_scan for the scan bwd AND we still need x_main (adj_a) for the scan
        //   bwd's dx and for dD. Put dy_scan into dr-region? dr is [kSeq,d] too
        //   small. Reuse a->y_scan as dy_scan (forward y_scan consumed: scan bwd
        //   recomputes its own forward). So dy_scan -> a->y_scan (in place).
        //   dD accumulation: owner = channel c (loops s).
        for (int c = threadIdx.x; c < mb::kDInner; c += blockDim.x) {
            float dDc = 0.0f;
            #pragma unroll
            for (int s = 0; s < mb::kSeq; ++s) {
                float sz = mb_silu(a->z[s][c]);
                float xm = sm->adj_a[s][c];               // x_main
                float yskip = a->y_scan[s][c] + xm * L.D[c];
                float dyg = sm->adj_c[s][c];              // dy_gated
                float dyskip = dyg * sz;
                float dsz = dyg * yskip;
                sm->adj_b[s][c] = dsz * mb_silu_grad(a->z[s][c]);   // adj_b := dz
                dDc += dyskip * xm;
                // dx_main (D-path) accumulated into adj_a IN PLACE? adj_a currently
                // holds x_main, still needed (this same loop reads it per s). After
                // the s-loop for this c we can overwrite. Stash D-path into a->y_scan
                // alongside dy_scan? dy_scan needs the value dyskip too. We need BOTH
                // dy_scan (for scan bwd) and dx_main_D. Put dy_scan -> a->y_scan,
                // and accumulate dx_main_D into dt_pre-region? dt_pre needed by scan
                // bwd. Cleanest: scan bwd's dx output is SET into adj_a; the D-path
                // must be ADDED after. So save dx_main_D now into adj_c (dy_gated
                // consumed): adj_c := dx_main_D.
                a->y_scan[s][c] = dyskip;                  // a->y_scan := dy_scan
                sm->adj_c[s][c] = dyskip * L.D[c];         // adj_c := dx_main_D
            }
            g.layer[li].D[c] += dDc;
        }
        __syncthreads();
        // (7) selective scan bwd. Inputs: x_main must be re-materialised (adj_a was
        //   x_main but we just overwrote it? NO — we wrote adj_b=dz and adj_c=
        //   dx_main_D, adj_a is UNTOUCHED = x_main). dy_scan = a->y_scan.
        //   Zero dBmat/dCmat first. Outputs: dx_main_scan SET into a fresh buffer;
        //   we route it to dt_pre-region? dt_pre still needed (scan bwd reads it).
        //   Use adj_a? adj_a = x_main needed by scan bwd. So dx_main_scan -> a
        //   temporary: reuse the LayerAct x_main_raw? x_main_raw needed by conv bwd
        //   (step 11). Reuse z? z needed? after dz computed, z's forward value still
        //   read by scan bwd? NO — scan bwd does not read z. But mb_make_x_main /
        //   silu_grad(conv) at step 11 needs conv, not z. z forward is consumed
        //   (dz done). So dx_main_scan -> a->z (in place, z forward free).
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kState; idx += blockDim.x) {
            sm->dBmat[idx / mb::kState][idx % mb::kState] = 0.0f;
            sm->dCmat[idx / mb::kState][idx % mb::kState] = 0.0f;
        }
        __syncthreads();
        mb_scan_bwd(L.A_log, &sm->adj_a[0][0], a, &a->y_scan[0][0],
                    &a->z[0][0],          // dx_main_scan SET -> a->z
                    &a->dt_pre[0][0],     // ddt_pre SET -> a->dt_pre (in place; scan
                                          //   bwd reads dt_pre via mb_softplus BEFORE
                                          //   writing ddt_pre per (t,j)? It reads then
                                          //   writes the SAME (t,j) — read happens in
                                          //   the forward-recompute pass + the
                                          //   softplus' uses dt_pre; the WRITE is the
                                          //   last op per (t,j). But the recompute
                                          //   pass for OTHER t reads dt_pre too. Since
                                          //   one thread owns channel j and processes
                                          //   ALL t in its recompute BEFORE the reverse
                                          //   pass writes any ddt_pre[*,j], the in-place
                                          //   write is safe per channel.)
                    &sm->dBmat[0][0], &sm->dCmat[0][0], G.A_log, sm->red);
        // dx_main accumulator: combine D-path (adj_c) + scan-path (a->z). Put the
        // running dx_main into adj_a (x_main no longer needed after scan bwd).
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, c = idx % mb::kDInner;
            sm->adj_a[s][c] = sm->adj_c[s][c] + a->z[s][c];   // dx_main = D-path + scan-path
        }
        __syncthreads();
        // (8) dt_proj bwd: dt_pre = dt_proj(dt_raw)+bias. ddt_pre = a->dt_pre.
        //   dW_dt += ddt_pre^T @ dt_raw ; db_dt += Σ ddt_pre ; ddt_raw = ddt_pre @ dt_proj_w.
        //   dt_raw must be recomputed: dt_raw = (x_proj(x_main))[:, :dt_rank]. But
        //   x_main = silu(conv) (we lost adj_a=x_main -> now dx_main). Re-materialise
        //   x_main into adj_b? adj_b currently = dz (needed at step 12 for dxz). So
        //   recompute dt_raw into dbc[:, :dt_rank] using x_main re-made into a temp.
        //   We have no free [kSeq,d_inner] buffer (adj_a=dx_main, adj_b=dz, adj_c
        //   =dx_main_D done -> FREE!). adj_c is free now. Re-make x_main -> adj_c.
        mb_make_x_main(a, &sm->adj_c[0][0]);   // adj_c := x_main (recomputed)
        // dt_raw = (x_main @ x_proj_w[:dt_rank]^T) -> dbc[:, :dt_rank].
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDtRank; idx += blockDim.x) {
            const int s = idx / mb::kDtRank, o = idx % mb::kDtRank;
            const float* xr = &sm->adj_c[s][0];   // x_main
            const float* Wr = L.x_proj_w + (int64_t)o * mb::kDInner;
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < mb::kDInner; ++k) acc += xr[k] * Wr[k];
            sm->dbc[s][o] = acc;   // dt_raw in dbc[:, :dt_rank] (scratch)
        }
        __syncthreads();
        // dt_proj bwd. ddt_raw -> a fresh [kSeq,dt_rank]; stash in dr[:, :dt_rank]
        // (dr=[kSeq,d], dt_rank=8<=d). dr's dx_residual is [kSeq,d] full — STILL
        // NEEDED at step 13. dt_rank=8 occupies dr[:, :8], colliding with the
        // residual. So put ddt_raw into the TAIL of dbc (dbc width=40, dt_raw uses
        // [0,8); B/C grads go elsewhere). Actually dx_dbc = cat[ddt_raw,dB,dC] is
        // assembled into dbc next — so write ddt_raw straight into dbc[:, :dt_rank]
        // AFTER consuming dt_raw. But dW_dt needs dt_raw while computing — order:
        // compute dW_dt + db_dt (reads ddt_pre=a->dt_pre, dt_raw=dbc[:,:8]), then
        // ddt_raw (reads ddt_pre + dt_proj_w) overwriting dbc[:,:8].
        {
            const float* dY = &a->dt_pre[0][0];   // ddt_pre [kSeq,d_inner]
            // dW_dt [d_inner, dt_rank]; db_dt [d_inner].
            const int wtot = mb::kDInner * mb::kDtRank;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / mb::kDtRank, i = idx % mb::kDtRank;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * mb::kDInner + o] * sm->dbc[s][i];
                G.dt_proj_w[(int64_t)o * mb::kDtRank + i] += acc;
            }
            for (int o = threadIdx.x; o < mb::kDInner; o += blockDim.x) {
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * mb::kDInner + o];
                G.dt_proj_b[o] += acc;
            }
            __syncthreads();
            // ddt_raw[s,i] = Σ_o ddt_pre[s,o]*dt_proj_w[o,i] -> dbc[:, :dt_rank].
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDtRank; idx += blockDim.x) {
                const int s = idx / mb::kDtRank, i = idx % mb::kDtRank;
                float acc = 0.0f;
                for (int o = 0; o < mb::kDInner; ++o) acc += dY[s * mb::kDInner + o] * L.dt_proj_w[(int64_t)o * mb::kDtRank + i];
                sm->dbc[s][i] = acc;   // ddt_raw -> dbc[:, :dt_rank]
            }
            __syncthreads();
        }
        // (9) assemble dx_dbc = cat[ddt_raw, dBmat, dCmat] into dbc (ddt_raw already
        //   in dbc[:, :dt_rank]; place dB,dC after).
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kState; idx += blockDim.x) {
            const int s = idx / mb::kState, k = idx % mb::kState;
            sm->dbc[s][mb::kDtRank + k] = sm->dBmat[s][k];
            sm->dbc[s][mb::kDtRank + mb::kState + k] = sm->dCmat[s][k];
        }
        __syncthreads();
        //   x_proj bwd: dW_xproj += dx_dbc^T @ x_main ; dx_main(x_proj path) =
        //   dx_dbc @ x_proj_w  -> ADD into the dx_main accumulator (adj_a). x_main
        //   = adj_c (still valid).
        {
            const float* dY = &sm->dbc[0][0];     // dx_dbc [kSeq,40]
            const float* X = &sm->adj_c[0][0];    // x_main [kSeq,d_inner]
            const int wtot = mb::kDbc * mb::kDInner;
            for (int idx = threadIdx.x; idx < wtot; idx += blockDim.x) {
                const int o = idx / mb::kDInner, i = idx % mb::kDInner;
                float acc = 0.0f;
                #pragma unroll
                for (int s = 0; s < mb::kSeq; ++s) acc += dY[s * mb::kDbc + o] * X[s * mb::kDInner + i];
                G.x_proj_w[(int64_t)o * mb::kDInner + i] += acc;
            }
            __syncthreads();
            // dx_main += dx_dbc @ x_proj_w (ADD into adj_a).
            const int xtot = mb::kSeq * mb::kDInner;
            for (int idx = threadIdx.x; idx < xtot; idx += blockDim.x) {
                const int s = idx / mb::kDInner, i = idx % mb::kDInner;
                float acc = 0.0f;
                for (int o = 0; o < mb::kDbc; ++o) acc += dY[s * mb::kDbc + o] * L.x_proj_w[(int64_t)o * mb::kDInner + i];
                sm->adj_a[s][i] += acc;   // dx_main += x_proj path
            }
            __syncthreads();
        }
        // (10-11) dx_main complete (adj_a). dconv = dx_main * silu'(conv) -> adj_c
        //   (x_main no longer needed). conv1d bwd: dx_main_raw + dW_conv + db_conv.
        for (int idx = threadIdx.x; idx < mb::kSeq * mb::kDInner; idx += blockDim.x) {
            const int s = idx / mb::kDInner, c = idx % mb::kDInner;
            sm->adj_c[s][c] = sm->adj_a[s][c] * mb_silu_grad(a->conv[s][c]);   // dconv
        }
        __syncthreads();
        //   conv bwd: dx -> adj_a (dx_main_raw); reads dconv=adj_c, in=x_main_raw.
        mb_conv1d_bwd(&sm->adj_c[0][0], &a->x_main_raw[0][0], L.conv_w,
                      G.conv_w, G.conv_b, &sm->adj_a[0][0]);   // adj_a := dx_main_raw
        // (12) dxz = cat[dx_main_raw, dz] (adj_a | adj_b). in_proj bwd: dW_in +=
        //   dxz^T @ hin ; dx_inproj = dxz @ in_w.  in_w is [2*d_inner, d]; rows
        //   [0,d_inner) pair with x_main_raw, [d_inner,2di) with z.
        //   dx (=dx_inproj + dx_residual) goes to the previous layer's dh.
        {
            // dW_in: rows 0..d_inner-1 from dx_main_raw(adj_a), d_inner..2di-1 from dz(adj_b).
            const int half = mb::kDInner;
            for (int o = threadIdx.x; o < 2 * mb::kDInner; o += blockDim.x) {
                const float* dyo = (o < half) ? &sm->adj_a[0][0] : &sm->adj_b[0][0];
                const int oo = (o < half) ? o : (o - half);
                for (int i = 0; i < mb::kD; ++i) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int s = 0; s < mb::kSeq; ++s)
                        acc += dyo[s * mb::kDInner + oo] * hin[s * mb::kD + i];
                    G.in_w[(int64_t)o * mb::kD + i] += acc;
                }
            }
            __syncthreads();
            // dx_inproj[s,i] = Σ_{o<d_inner} dx_main_raw[s,o]*in_w[o,i]
            //               + Σ_{o<d_inner} dz[s,o]*in_w[d_inner+o,i]
            //   then + dx_residual (sm->dr). Write into dh for the previous layer.
            for (int idx = threadIdx.x; idx < mb::kSeq * mb::kD; idx += blockDim.x) {
                const int s = idx / mb::kD, i = idx % mb::kD;
                float acc = 0.0f;
                for (int o = 0; o < mb::kDInner; ++o) {
                    acc += sm->adj_a[s][o] * L.in_w[(int64_t)o * mb::kD + i];
                    acc += sm->adj_b[s][o] * L.in_w[(int64_t)(mb::kDInner + o) * mb::kD + i];
                }
                sm->dh[s][i] = acc + sm->dr[s][i];   // (13) dx = in_proj path + residual
            }
            __syncthreads();
        }
        // dh now = grad wrt the previous layer's output (or the embedding for li=0).
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
