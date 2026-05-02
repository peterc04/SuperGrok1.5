// csrc/kernels/cuda/sm_90/models/vit.cuh
// Vision Transformer model header for sm_90 (Hopper).
//
// Implements the post-norm ViT from grokking_race_v2.py:ViT/EncoderBlock:
//
//     h = patch_proj(x_patches) + b_patch
//     h = concat([cls_token, h], dim=seq) + pos_embed
//     for layer in layers:
//         a = attn(h, h, h)              (kCausal=false)
//         h = LN1(h + a)
//         h = LN2(h + FFN(h))           FFN: Linear -> GELU -> Linear
//     y = head(LN_final(h[:, 0, :]))   (CLS read, then norm, then class head)
//
// Defaults: d_model=128, n_heads=4, d_head=32, num_patches+1 = seq_len = 17.
// Attention uses sg::sm90::models::attention::attention_forward<ActT, d_head=32, kCausal=false>.
//
// Activations buffer (saved for backward):
//   For convenience, this file uses a contiguous activation buffer whose
//   per-section offsets are computed at runtime by ActLayout below.
//
// Weights buffer layout (single contiguous WeightT* span, in this order):
//   patch_W           [d_model, patch_dim]
//   patch_b           [d_model]
//   cls_token         [d_model]
//   pos_embed         [seq_len, d_model]                seq_len = num_patches + 1
//   for L in layers:
//     qkv_W           [3*H*Dh, d_model]                  H=n_heads, Dh=d_head
//     qkv_b           [3*H*Dh]
//     out_W           [d_model, H*Dh]
//     out_b           [d_model]
//     ln1_gamma       [d_model]
//     ln1_beta        [d_model]
//     ff1_W           [ffn_hidden, d_model]              ffn_hidden = ffn_expansion * d_model
//     ff1_b           [ffn_hidden]
//     ff2_W           [d_model, ffn_hidden]
//     ff2_b           [d_model]
//     ln2_gamma       [d_model]
//     ln2_beta        [d_model]
//   ln_final_gamma    [d_model]
//   ln_final_beta     [d_model]
//   head_W            [n_classes, d_model]
//   head_b            [n_classes]
//
// Activation layout (ActT* contiguous buffer; mirrors all required tensors
// the backward needs to recompute or reuse). Float-typed sections (means,
// invstds, softmax_lse) live in the same buffer reinterpreted as float.

#pragma once
#include "csrc/common/platform.h"
#include "csrc/common/types.h"
#include "csrc/common/utils.cuh"
#include "csrc/kernels/cuda/sm_90/models/attention.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace sg { namespace sm90 { namespace models { namespace vit {

// ─── dtype helpers ─────────────────────────────────────────────────────
template <typename T>
__device__ __forceinline__ float to_f32(T v) {
    return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_f32(float v) {
    return static_cast<T>(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
    return __float2half_rn(v);
}

// ─── weight slicer ─────────────────────────────────────────────────────
template <typename WeightT>
struct WeightLayout {
    const WeightT* base;
    int d_model, n_heads, d_head, ffn_hidden, n_layers, n_classes, num_patches, patch_dim;
    int seq_len;  // num_patches + 1

    __host__ __device__ const WeightT* patch_W() const { return base; }
    __host__ __device__ const WeightT* patch_b() const { return patch_W() + d_model * patch_dim; }
    __host__ __device__ const WeightT* cls_token() const { return patch_b() + d_model; }
    __host__ __device__ const WeightT* pos_embed() const { return cls_token() + d_model; }

    __host__ __device__ size_t per_layer_count() const {
        size_t qkv_W = (size_t)3 * n_heads * d_head * d_model;
        size_t qkv_b = (size_t)3 * n_heads * d_head;
        size_t out_W = (size_t)d_model * n_heads * d_head;
        size_t out_b = d_model;
        size_t ln1   = (size_t)2 * d_model;
        size_t ff1_W = (size_t)ffn_hidden * d_model;
        size_t ff1_b = ffn_hidden;
        size_t ff2_W = (size_t)d_model * ffn_hidden;
        size_t ff2_b = d_model;
        size_t ln2   = (size_t)2 * d_model;
        return qkv_W + qkv_b + out_W + out_b + ln1 + ff1_W + ff1_b + ff2_W + ff2_b + ln2;
    }

    __host__ __device__ const WeightT* layer_base(int L) const {
        return pos_embed() + (size_t)seq_len * d_model + (size_t)L * per_layer_count();
    }
    __host__ __device__ const WeightT* qkv_W(int L)   const { return layer_base(L); }
    __host__ __device__ const WeightT* qkv_b(int L)   const { return qkv_W(L)   + (size_t)3 * n_heads * d_head * d_model; }
    __host__ __device__ const WeightT* out_W(int L)   const { return qkv_b(L)   + (size_t)3 * n_heads * d_head; }
    __host__ __device__ const WeightT* out_b(int L)   const { return out_W(L)   + (size_t)d_model * n_heads * d_head; }
    __host__ __device__ const WeightT* ln1_gamma(int L) const { return out_b(L) + d_model; }
    __host__ __device__ const WeightT* ln1_beta(int L)  const { return ln1_gamma(L) + d_model; }
    __host__ __device__ const WeightT* ff1_W(int L)   const { return ln1_beta(L)+ d_model; }
    __host__ __device__ const WeightT* ff1_b(int L)   const { return ff1_W(L)   + (size_t)ffn_hidden * d_model; }
    __host__ __device__ const WeightT* ff2_W(int L)   const { return ff1_b(L)   + ffn_hidden; }
    __host__ __device__ const WeightT* ff2_b(int L)   const { return ff2_W(L)   + (size_t)d_model * ffn_hidden; }
    __host__ __device__ const WeightT* ln2_gamma(int L) const { return ff2_b(L) + d_model; }
    __host__ __device__ const WeightT* ln2_beta(int L)  const { return ln2_gamma(L) + d_model; }

    __host__ __device__ const WeightT* ln_final_gamma() const { return layer_base(n_layers); }
    __host__ __device__ const WeightT* ln_final_beta()  const { return ln_final_gamma() + d_model; }
    __host__ __device__ const WeightT* head_W()         const { return ln_final_beta() + d_model; }
    __host__ __device__ const WeightT* head_b()         const { return head_W() + (size_t)n_classes * d_model; }

    __host__ __device__ size_t total_count() const {
        return (size_t)(head_b() + n_classes - base);
    }
};

// ─── activation slicer ─────────────────────────────────────────────────
// All sections are densely packed in the activation buffer. ActT for everything
// except a few FP32 stat sections; we reinterpret_cast to float for those.
template <typename ActT>
struct ActLayout {
    ActT* base;
    int batch, seq_len, d_model, n_heads, d_head, ffn_hidden, n_layers, n_classes;

    __host__ __device__ size_t bsd() const { return (size_t)batch * seq_len * d_model; }
    __host__ __device__ size_t bsh() const { return (size_t)batch * seq_len * 3 * n_heads * d_head; }
    __host__ __device__ size_t bsf() const { return (size_t)batch * seq_len * ffn_hidden; }
    __host__ __device__ size_t bs()  const { return (size_t)batch * seq_len; }
    __host__ __device__ size_t bhs() const { return (size_t)batch * n_heads * seq_len; }

    // Sized in floats; we reserve enough ActT slots based on relative size.
    __host__ __device__ size_t f_to_act(size_t n_floats) const {
        size_t bytes = n_floats * sizeof(float);
        size_t a = sizeof(ActT);
        return (bytes + a - 1) / a;
    }

    // Layout (per layer L block):
    //   pre_attn_in   [B, S, D]    (the input to attn for layer L; needed for backward)
    //   qkv           [B, S, 3*H*Dh]
    //   attn_out      [B, S, H*Dh]   (output of attention before output projection)
    //   softmax_lse   [B, H, S]      (FP32 reinterpreted)
    //   proj_out      [B, S, D]      (= attn output projection result)
    //   ln1_in        [B, S, D]      (= pre_attn_in + proj_out, input to LN1)
    //   ln1_mean      [B, S] FP32
    //   ln1_invstd    [B, S] FP32
    //   ln1_out       [B, S, D]      (post-LN1; also serves as ffn input + residual)
    //   ffn_h_pre     [B, S, F]      (ff1_W·ln1_out + ff1_b)
    //   ffn_h         [B, S, F]      (GELU(ffn_h_pre))
    //   ffn_out       [B, S, D]      (ff2_W·ffn_h + ff2_b)
    //   ln2_in        [B, S, D]      (= ln1_out + ffn_out, input to LN2)
    //   ln2_mean      [B, S] FP32
    //   ln2_invstd    [B, S] FP32
    //   ln2_out       [B, S, D]      (post-LN2; layer output)
    //
    // After all layers:
    //   final_in      [B, D]         (= ln2_out_last[:, 0, :], the CLS slice)
    //   final_mean    [B] FP32
    //   final_invstd  [B] FP32
    //   final_out     [B, D]
    //
    // The "tokens" buffer (after patch+CLS+pos) is the layer-0 pre_attn_in.

    __host__ __device__ size_t per_layer_count_in_act() const {
        size_t s = 0;
        s += bsd();              // pre_attn_in
        s += bsh();              // qkv
        s += bsd();              // attn_out (= H*Dh = D, since H*Dh=d_model in default)
                                 // (use H*Dh strictly though; we conservatively allocate bsd
                                 //  because H*Dh == d_model in the default config; if this
                                 //  invariant is broken at the call-site, attn_out span uses
                                 //  H*Dh, not d_model.)
        s += f_to_act(bhs());    // softmax_lse (FP32)
        s += bsd();              // proj_out
        s += bsd();              // ln1_in
        s += f_to_act(bs());     // ln1_mean
        s += f_to_act(bs());     // ln1_invstd
        s += bsd();              // ln1_out
        s += bsf();              // ffn_h_pre
        s += bsf();              // ffn_h
        s += bsd();              // ffn_out
        s += bsd();              // ln2_in
        s += f_to_act(bs());     // ln2_mean
        s += f_to_act(bs());     // ln2_invstd
        s += bsd();              // ln2_out
        return s;
    }

    struct LayerPtrs {
        ActT*  pre_attn_in;
        ActT*  qkv;
        ActT*  attn_out;
        float* softmax_lse;
        ActT*  proj_out;
        ActT*  ln1_in;
        float* ln1_mean;
        float* ln1_invstd;
        ActT*  ln1_out;
        ActT*  ffn_h_pre;
        ActT*  ffn_h;
        ActT*  ffn_out;
        ActT*  ln2_in;
        float* ln2_mean;
        float* ln2_invstd;
        ActT*  ln2_out;
    };

    __host__ __device__ LayerPtrs layer(int L) {
        LayerPtrs p;
        ActT* cur = base + (size_t)L * per_layer_count_in_act();
        p.pre_attn_in = cur; cur += bsd();
        p.qkv         = cur; cur += bsh();
        p.attn_out    = cur; cur += bsd();
        p.softmax_lse = reinterpret_cast<float*>(cur); cur += f_to_act(bhs());
        p.proj_out    = cur; cur += bsd();
        p.ln1_in      = cur; cur += bsd();
        p.ln1_mean    = reinterpret_cast<float*>(cur); cur += f_to_act(bs());
        p.ln1_invstd  = reinterpret_cast<float*>(cur); cur += f_to_act(bs());
        p.ln1_out     = cur; cur += bsd();
        p.ffn_h_pre   = cur; cur += bsf();
        p.ffn_h       = cur; cur += bsf();
        p.ffn_out     = cur; cur += bsd();
        p.ln2_in      = cur; cur += bsd();
        p.ln2_mean    = reinterpret_cast<float*>(cur); cur += f_to_act(bs());
        p.ln2_invstd  = reinterpret_cast<float*>(cur); cur += f_to_act(bs());
        p.ln2_out     = cur; cur += bsd();
        return p;
    }

    struct FinalPtrs {
        ActT*  final_in;
        float* final_mean;
        float* final_invstd;
        ActT*  final_out;
    };

    __host__ __device__ FinalPtrs final_ptrs() {
        FinalPtrs p;
        ActT* cur = base + (size_t)n_layers * per_layer_count_in_act();
        p.final_in     = cur; cur += (size_t)batch * d_model;
        p.final_mean   = reinterpret_cast<float*>(cur); cur += f_to_act(batch);
        p.final_invstd = reinterpret_cast<float*>(cur); cur += f_to_act(batch);
        p.final_out    = cur; cur += (size_t)batch * d_model;
        return p;
    }
};

// ─── kernels: GEMM (small/general, FP32 accum) ──────────────────────────
// out[m, n] = sum_k W[n, k] * in[m, k] + b[n]    (b optional)
// Used for patch projection ([B*S, patch_dim] -> [B*S, d_model]),
// QKV ([B*S, D] -> [B*S, 3*H*Dh]), out_proj, ff1, ff2, head.
//
// Block: 1 token at a time per block, each thread handles one or more
// output columns. For our small d_model (<= 512) this is bandwidth-bound
// and clean to verify correctness.

template <typename ActT, typename WeightT>
__global__ void gemm_bias_kernel(
    const ActT* __restrict__ in,    // [M, K]
    const WeightT* __restrict__ W,  // [N, K]
    const WeightT* __restrict__ b,  // [N] or nullptr
    ActT* __restrict__ out,         // [M, N]
    int M, int N, int K
) {
    int m = blockIdx.x;
    if (m >= M) return;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float acc = (b != nullptr) ? to_f32(b[n]) : 0.0f;
        const WeightT* w_row = W + (size_t)n * K;
        const ActT*    a_row = in + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            acc += to_f32(w_row[k]) * to_f32(a_row[k]);
        }
        out[(size_t)m * N + n] = from_f32<ActT>(acc);
    }
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_gemm_bias(
    const ActT* in, const WeightT* W, const WeightT* b, ActT* out,
    int M, int N, int K, cudaStream_t stream
) {
    int block = 128;
    if (N < 128) block = ((N + 31) / 32) * 32;
    if (block < 32) block = 32;
    gemm_bias_kernel<ActT, WeightT><<<M, block, 0, stream>>>(in, W, b, out, M, N, K);
    return cudaGetLastError();
}

// dX = dY · W   where Y[m,n] = sum_k W[n,k] X[m,k]; dX[m,k] = sum_n dY[m,n] W[n,k]
template <typename ActT, typename WeightT>
__global__ void gemm_grad_input_kernel(
    const ActT* __restrict__ dY,     // [M, N]
    const WeightT* __restrict__ W,   // [N, K]
    ActT* __restrict__ dX,           // [M, K]
    int M, int N, int K
) {
    int m = blockIdx.x;
    if (m >= M) return;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; n++) {
            acc += to_f32(dY[(size_t)m * N + n]) * to_f32(W[(size_t)n * K + k]);
        }
        dX[(size_t)m * K + k] = from_f32<ActT>(acc);
    }
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_gemm_grad_input(
    const ActT* dY, const WeightT* W, ActT* dX,
    int M, int N, int K, cudaStream_t stream
) {
    int block = 128;
    if (K < 128) block = ((K + 31) / 32) * 32;
    if (block < 32) block = 32;
    gemm_grad_input_kernel<ActT, WeightT><<<M, block, 0, stream>>>(dY, W, dX, M, N, K);
    return cudaGetLastError();
}

// dW[n,k] = sum_m dY[m,n] X[m,k]; db[n] = sum_m dY[m,n]
template <typename ActT, typename WeightT>
__global__ void gemm_grad_weight_kernel(
    const ActT* __restrict__ dY,     // [M, N]
    const ActT* __restrict__ X,      // [M, K]
    WeightT* __restrict__ dW,        // [N, K]
    WeightT* __restrict__ db,        // [N] or nullptr
    int M, int N, int K
) {
    int n = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float acc = 0.0f;
    for (int m = 0; m < M; m++) {
        acc += to_f32(dY[(size_t)m * N + n]) * to_f32(X[(size_t)m * K + k]);
    }
    // accumulate (overwrite) — caller is responsible for zeroing dW upfront
    // if multiple sources contribute (we instead always overwrite per-call).
    dW[(size_t)n * K + k] = from_f32<WeightT>(acc);
    if (db != nullptr && k == 0) {
        float sb = 0.0f;
        for (int m = 0; m < M; m++) sb += to_f32(dY[(size_t)m * N + n]);
        db[n] = from_f32<WeightT>(sb);
    }
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_gemm_grad_weight(
    const ActT* dY, const ActT* X, WeightT* dW, WeightT* db,
    int M, int N, int K, cudaStream_t stream
) {
    int block = 128;
    int grid_x = (K + block - 1) / block;
    dim3 grid(grid_x, N);
    gemm_grad_weight_kernel<ActT, WeightT><<<grid, block, 0, stream>>>(dY, X, dW, db, M, N, K);
    return cudaGetLastError();
}

// ─── patch projection ──────────────────────────────────────────────────
// Treats `image` as [B, num_patches, patch_dim]. We accept the binding's
// (channels, height, width, patch_size) convention by computing
//   num_patches = (height / patch_size) * (width / patch_size)
//   patch_dim   = patch_size * patch_size * channels
// at the call site, then reshaping the input to [B*num_patches, patch_dim].
// The forward calls launch_gemm_bias to produce [B*num_patches, d_model].

// ─── CLS token + positional embedding fusion ──────────────────────────
// Input  patches_proj : [B, num_patches, d_model]
// Output tokens       : [B, num_patches+1, d_model]
//   tokens[b, 0, :]      = cls_token + pos_embed[0]
//   tokens[b, 1+p, :]    = patches_proj[b, p, :] + pos_embed[1+p]

template <typename ActT, typename WeightT>
__global__ void cls_pos_fwd_kernel(
    const ActT* __restrict__ patches_proj,   // [B, num_patches, D]
    const WeightT* __restrict__ cls_token,   // [D]
    const WeightT* __restrict__ pos_embed,   // [seq_len, D]
    ActT* __restrict__ tokens,               // [B, seq_len, D]
    int batch, int num_patches, int seq_len, int d_model
) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    if (b >= batch || s >= seq_len) return;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float pe = to_f32(pos_embed[(size_t)s * d_model + d]);
        float v;
        if (s == 0) {
            v = to_f32(cls_token[d]) + pe;
        } else {
            v = to_f32(patches_proj[((size_t)b * num_patches + (s - 1)) * d_model + d]) + pe;
        }
        tokens[((size_t)b * seq_len + s) * d_model + d] = from_f32<ActT>(v);
    }
}

template <typename ActT, typename WeightT>
__global__ void cls_pos_bwd_kernel(
    const ActT* __restrict__ d_tokens,        // [B, seq_len, D]
    ActT* __restrict__ d_patches_proj,        // [B, num_patches, D]
    WeightT* __restrict__ d_cls_token,        // [D]   (accumulator)
    WeightT* __restrict__ d_pos_embed,        // [seq_len, D] (accumulator)
    int batch, int num_patches, int seq_len, int d_model
) {
    // One block per (s, d-tile); we accumulate via atomics for simplicity.
    int s = blockIdx.x;
    if (s >= seq_len) return;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float pe_acc = 0.0f;
        float cls_acc = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dv = to_f32(d_tokens[((size_t)b * seq_len + s) * d_model + d]);
            pe_acc += dv;
            if (s == 0) {
                cls_acc += dv;
            } else {
                d_patches_proj[((size_t)b * num_patches + (s - 1)) * d_model + d] = from_f32<ActT>(dv);
            }
        }
        d_pos_embed[(size_t)s * d_model + d] = from_f32<WeightT>(pe_acc);
        if (s == 0) d_cls_token[d] = from_f32<WeightT>(cls_acc);
    }
}

// ─── QKV reshape: split [B, S, 3*H*Dh] into Q/K/V each [B, H, S, Dh] ────
template <typename ActT>
__global__ void split_qkv_kernel(
    const ActT* __restrict__ qkv,    // [B, S, 3*H*Dh]
    ActT* __restrict__ q,            // [B, H, S, Dh]
    ActT* __restrict__ k,            // [B, H, S, Dh]
    ActT* __restrict__ v,            // [B, H, S, Dh]
    int batch, int seq, int n_heads, int d_head
) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    int Dh = d_head;
    int H = n_heads;
    int HD = H * Dh;
    for (int idx = threadIdx.x; idx < 3 * HD; idx += blockDim.x) {
        int part = idx / HD;            // 0=Q, 1=K, 2=V
        int hd   = idx % HD;
        int h    = hd / Dh;
        int d    = hd % Dh;
        ActT val = qkv[((size_t)b * seq + s) * 3 * HD + idx];
        size_t out_idx = (((size_t)b * H + h) * seq + s) * Dh + d;
        if (part == 0) q[out_idx] = val;
        else if (part == 1) k[out_idx] = val;
        else v[out_idx] = val;
    }
}

template <typename ActT>
__global__ void merge_attn_out_kernel(
    const ActT* __restrict__ attn_out_bhsd,   // [B, H, S, Dh]
    ActT* __restrict__ attn_out_bsd,          // [B, S, H*Dh]
    int batch, int seq, int n_heads, int d_head
) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    int H = n_heads, Dh = d_head, HD = H * Dh;
    for (int hd = threadIdx.x; hd < HD; hd += blockDim.x) {
        int h = hd / Dh;
        int d = hd % Dh;
        ActT v = attn_out_bhsd[(((size_t)b * H + h) * seq + s) * Dh + d];
        attn_out_bsd[((size_t)b * seq + s) * HD + hd] = v;
    }
}

template <typename ActT>
__global__ void unmerge_attn_out_kernel(
    const ActT* __restrict__ d_attn_out_bsd,  // [B, S, H*Dh]
    ActT* __restrict__ d_attn_out_bhsd,       // [B, H, S, Dh]
    int batch, int seq, int n_heads, int d_head
) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    int H = n_heads, Dh = d_head, HD = H * Dh;
    for (int hd = threadIdx.x; hd < HD; hd += blockDim.x) {
        int h = hd / Dh;
        int d = hd % Dh;
        d_attn_out_bhsd[(((size_t)b * H + h) * seq + s) * Dh + d] =
            d_attn_out_bsd[((size_t)b * seq + s) * HD + hd];
    }
}

template <typename ActT>
__global__ void merge_qkv_grads_kernel(
    const ActT* __restrict__ dq,     // [B, H, S, Dh]
    const ActT* __restrict__ dk,
    const ActT* __restrict__ dv,
    ActT* __restrict__ dqkv,         // [B, S, 3*H*Dh]
    int batch, int seq, int n_heads, int d_head
) {
    int b = blockIdx.y;
    int s = blockIdx.x;
    int H = n_heads, Dh = d_head, HD = H * Dh;
    for (int hd = threadIdx.x; hd < HD; hd += blockDim.x) {
        int h = hd / Dh;
        int d = hd % Dh;
        size_t in_idx = (((size_t)b * H + h) * seq + s) * Dh + d;
        size_t base = ((size_t)b * seq + s) * 3 * HD;
        dqkv[base + 0 * HD + hd] = dq[in_idx];
        dqkv[base + 1 * HD + hd] = dk[in_idx];
        dqkv[base + 2 * HD + hd] = dv[in_idx];
    }
}

// ─── Residual add: out[i] = a[i] + b[i] ────────────────────────────────
template <typename ActT>
__global__ void add_kernel(const ActT* a, const ActT* b, ActT* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = from_f32<ActT>(to_f32(a[i]) + to_f32(b[i]));
}

template <typename ActT>
inline cudaError_t launch_add(const ActT* a, const ActT* b, ActT* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    add_kernel<ActT><<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

// ─── LayerNorm forward (one row = one block) ──────────────────────────
template <typename ActT, typename WeightT>
__global__ void layernorm_fwd_kernel(
    const ActT* __restrict__ x,           // [M, D]
    const WeightT* __restrict__ gamma,    // [D]
    const WeightT* __restrict__ beta,     // [D]
    ActT* __restrict__ y,                 // [M, D]
    float* __restrict__ mean,             // [M]
    float* __restrict__ invstd,           // [M]
    int M, int D, float eps
) {
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;

    extern __shared__ float smem[];   // [blockDim.x]
    // 1) sum + sumsq
    float s = 0.0f, ss = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = to_f32(x[(size_t)m * D + d]);
        s  += v;
        ss += v * v;
    }
    // block reduce
    smem[tid] = s; __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    float total_s = smem[0];
    __syncthreads();
    smem[tid] = ss; __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    float total_ss = smem[0];
    __syncthreads();

    float mu = total_s / (float)D;
    float var = total_ss / (float)D - mu * mu;
    float inv = fast_rsqrt_nr(var + eps);
    if (tid == 0) {
        if (mean   != nullptr) mean[m] = mu;
        if (invstd != nullptr) invstd[m] = inv;
    }

    for (int d = tid; d < D; d += blockDim.x) {
        float v = to_f32(x[(size_t)m * D + d]);
        float xh = (v - mu) * inv;
        float g = (gamma != nullptr) ? to_f32(gamma[d]) : 1.0f;
        float bt = (beta != nullptr) ? to_f32(beta[d]) : 0.0f;
        y[(size_t)m * D + d] = from_f32<ActT>(xh * g + bt);
    }
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_layernorm_fwd(
    const ActT* x, const WeightT* gamma, const WeightT* beta,
    ActT* y, float* mean, float* invstd,
    int M, int D, float eps, cudaStream_t stream
) {
    int block = 128;
    while (block > D && block > 32) block >>= 1;
    if (block < 32) block = 32;
    int smem = block * sizeof(float);
    layernorm_fwd_kernel<ActT, WeightT><<<M, block, smem, stream>>>(
        x, gamma, beta, y, mean, invstd, M, D, eps);
    return cudaGetLastError();
}

// LayerNorm backward
//   dx_i = (1/D) * gamma_i * inv * ( D*dy_i - sum(dy*g) - x_hat_i * sum(dy*g*x_hat) )
//   dgamma_i = sum_m dy_i * x_hat_i
//   dbeta_i  = sum_m dy_i

template <typename ActT, typename WeightT>
__global__ void layernorm_bwd_kernel(
    const ActT* __restrict__ x,           // [M, D] (input to LN)
    const ActT* __restrict__ dy,          // [M, D]
    const WeightT* __restrict__ gamma,    // [D]
    const float* __restrict__ mean,       // [M]
    const float* __restrict__ invstd,     // [M]
    ActT* __restrict__ dx,                // [M, D]
    int M, int D
) {
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    float mu = mean[m];
    float inv = invstd[m];

    extern __shared__ float smem[];
    float sum_a = 0.0f;  // sum(dy * gamma)
    float sum_b = 0.0f;  // sum(dy * gamma * x_hat)
    for (int d = tid; d < D; d += blockDim.x) {
        float dyv = to_f32(dy[(size_t)m * D + d]);
        float g = to_f32(gamma[d]);
        float xh = (to_f32(x[(size_t)m * D + d]) - mu) * inv;
        sum_a += dyv * g;
        sum_b += dyv * g * xh;
    }
    smem[tid] = sum_a; __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    float sa = smem[0]; __syncthreads();
    smem[tid] = sum_b; __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    float sb = smem[0]; __syncthreads();

    float invD = 1.0f / (float)D;
    for (int d = tid; d < D; d += blockDim.x) {
        float dyv = to_f32(dy[(size_t)m * D + d]);
        float g = to_f32(gamma[d]);
        float xh = (to_f32(x[(size_t)m * D + d]) - mu) * inv;
        float dxv = inv * (dyv * g - invD * sa - xh * invD * sb);
        dx[(size_t)m * D + d] = from_f32<ActT>(dxv);
    }
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_layernorm_bwd(
    const ActT* x, const ActT* dy, const WeightT* gamma,
    const float* mean, const float* invstd,
    ActT* dx, int M, int D, cudaStream_t stream
) {
    int block = 128;
    while (block > D && block > 32) block >>= 1;
    if (block < 32) block = 32;
    int smem = block * sizeof(float);
    layernorm_bwd_kernel<ActT, WeightT><<<M, block, smem, stream>>>(
        x, dy, gamma, mean, invstd, dx, M, D);
    return cudaGetLastError();
}

// dgamma = sum_m dy_m * xhat_m, dbeta = sum_m dy_m
template <typename ActT, typename WeightT>
__global__ void layernorm_grad_param_kernel(
    const ActT* __restrict__ x,
    const ActT* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    WeightT* __restrict__ dgamma,
    WeightT* __restrict__ dbeta,
    int M, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float dg = 0.0f, db = 0.0f;
    for (int m = 0; m < M; m++) {
        float dyv = to_f32(dy[(size_t)m * D + d]);
        float xh = (to_f32(x[(size_t)m * D + d]) - mean[m]) * invstd[m];
        dg += dyv * xh;
        db += dyv;
    }
    if (dgamma != nullptr) dgamma[d] = from_f32<WeightT>(dg);
    if (dbeta  != nullptr) dbeta[d]  = from_f32<WeightT>(db);
}

template <typename ActT, typename WeightT>
inline cudaError_t launch_layernorm_grad_param(
    const ActT* x, const ActT* dy, const float* mean, const float* invstd,
    WeightT* dgamma, WeightT* dbeta, int M, int D, cudaStream_t stream
) {
    int block = 128;
    int grid = (D + block - 1) / block;
    layernorm_grad_param_kernel<ActT, WeightT><<<grid, block, 0, stream>>>(
        x, dy, mean, invstd, dgamma, dbeta, M, D);
    return cudaGetLastError();
}

// ─── GELU (tanh approx, matches PyTorch nn.GELU default) ───────────────
// gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
// Backward: dx = dy * d/dx gelu(x)

template <typename ActT>
__global__ void gelu_fwd_kernel(const ActT* in, ActT* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = to_f32(in[i]);
    const float k = 0.7978845608f;  // sqrt(2/pi)
    float t = k * (x + 0.044715f * x * x * x);
    out[i] = from_f32<ActT>(0.5f * x * (1.0f + tanhf(t)));
}

template <typename ActT>
__global__ void gelu_bwd_kernel(const ActT* in, const ActT* dy, ActT* dx, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = to_f32(in[i]);
    const float k = 0.7978845608f;
    float u = k * (x + 0.044715f * x * x * x);
    float th = tanhf(u);
    float dudx = k * (1.0f + 3.0f * 0.044715f * x * x);
    float dgelu = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * dudx;
    dx[i] = from_f32<ActT>(to_f32(dy[i]) * dgelu);
}

template <typename ActT>
inline cudaError_t launch_gelu_fwd(const ActT* in, ActT* out, size_t n, cudaStream_t stream) {
    int block = 256; int grid = (int)((n + block - 1) / block);
    gelu_fwd_kernel<ActT><<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

template <typename ActT>
inline cudaError_t launch_gelu_bwd(const ActT* in, const ActT* dy, ActT* dx, size_t n, cudaStream_t stream) {
    int block = 256; int grid = (int)((n + block - 1) / block);
    gelu_bwd_kernel<ActT><<<grid, block, 0, stream>>>(in, dy, dx, n);
    return cudaGetLastError();
}

// ─── CLS extract: read x[:, 0, :] -> y[:, :] ────────────────────────────
template <typename ActT>
__global__ void cls_extract_fwd_kernel(
    const ActT* __restrict__ tokens,    // [B, S, D]
    ActT* __restrict__ cls,             // [B, D]
    int batch, int seq, int d_model
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        cls[(size_t)b * d_model + d] = tokens[((size_t)b * seq + 0) * d_model + d];
    }
}

template <typename ActT>
__global__ void cls_scatter_bwd_kernel(
    const ActT* __restrict__ d_cls,     // [B, D]
    ActT* __restrict__ d_tokens,        // [B, S, D]   (zeroed for s != 0)
    int batch, int seq, int d_model
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    if (b >= batch || s >= seq) return;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        if (s == 0) {
            d_tokens[((size_t)b * seq + 0) * d_model + d] = d_cls[(size_t)b * d_model + d];
        } else {
            d_tokens[((size_t)b * seq + s) * d_model + d] = from_f32<ActT>(0.0f);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Forward pass
// ════════════════════════════════════════════════════════════════════════
template <typename ActT, typename WeightT>
cudaError_t forward(
    const ActT* input,
    const WeightT* weights,
    ActT* output,
    ActT* activations,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    if (patch_size <= 0 || height % patch_size != 0 || width % patch_size != 0) {
        return cudaErrorInvalidValue;
    }
    int num_patches = (height / patch_size) * (width / patch_size);
    int patch_dim   = patch_size * patch_size * channels;
    int seq_len     = num_patches + 1;
    int ffn_hidden  = (int)(ffn_expansion * d_model);
    int HD          = n_heads * d_head;

    WeightLayout<WeightT> w{};
    w.base = weights; w.d_model = d_model; w.n_heads = n_heads; w.d_head = d_head;
    w.ffn_hidden = ffn_hidden; w.n_layers = n_layers; w.n_classes = n_classes;
    w.num_patches = num_patches; w.patch_dim = patch_dim; w.seq_len = seq_len;

    ActLayout<ActT> A{};
    A.base = activations; A.batch = batch; A.seq_len = seq_len; A.d_model = d_model;
    A.n_heads = n_heads; A.d_head = d_head; A.ffn_hidden = ffn_hidden;
    A.n_layers = n_layers; A.n_classes = n_classes;

    cudaError_t err;

    // Step 1: patch projection — input is [B, num_patches, patch_dim] flat.
    //   patches_proj = input @ patch_W^T + patch_b
    // We project directly into the layer-0 "pre_attn_in" buffer's "patches"
    // region by routing through a temporary ActT scratch (layer-0 attn_out).
    auto L0 = A.layer(0);
    ActT* patches_proj = L0.proj_out;  // reuse: temporary scratch (overwritten before use)
    err = launch_gemm_bias<ActT, WeightT>(
        input, w.patch_W(), w.patch_b(), patches_proj,
        /*M=*/batch * num_patches, /*N=*/d_model, /*K=*/patch_dim, stream);
    if (err != cudaSuccess) return err;

    // Step 2: prepend CLS + add pos embed -> tokens (= layer-0 pre_attn_in)
    {
        dim3 grid(seq_len, batch);
        int block = 128; if (block > d_model) block = d_model;
        cls_pos_fwd_kernel<ActT, WeightT><<<grid, block, 0, stream>>>(
            patches_proj, w.cls_token(), w.pos_embed(),
            L0.pre_attn_in, batch, num_patches, seq_len, d_model);
    }
    err = cudaGetLastError(); if (err != cudaSuccess) return err;

    // Step 3: encoder layers
    int M = batch * seq_len;
    for (int L = 0; L < n_layers; L++) {
        auto P = A.layer(L);

        // QKV projection: [M, D] -> [M, 3*HD]
        err = launch_gemm_bias<ActT, WeightT>(
            P.pre_attn_in, w.qkv_W(L), w.qkv_b(L), P.qkv,
            M, 3 * HD, d_model, stream);
        if (err != cudaSuccess) return err;

        // Split into Q/K/V [B, H, S, Dh] each via temporary scratch.
        // We reuse "ffn_h_pre" as scratch for q,k,v concatenated.
        // Each occupies B*H*S*Dh = bsd() (since H*Dh=d_model in default).
        // To be safe even when H*Dh != d_model, we require ffn_hidden >= 3*HD.
        // Otherwise we allocate via the attn_out + ln1_in + ln1_out triple.
        // For simplicity here, use proj_out (size B*S*D) for each of q,k,v
        // when H*Dh <= d_model.
        ActT* q_buf = P.proj_out;          // size B*S*D
        ActT* k_buf = P.ln1_in;            // size B*S*D
        ActT* v_buf = P.ln1_out;           // size B*S*D
        // These are reused before they're needed for their final purpose
        // (proj_out, ln1_in, ln1_out get rewritten later in this iteration).
        {
            dim3 grid(seq_len, batch);
            int block = 128;
            split_qkv_kernel<ActT><<<grid, block, 0, stream>>>(
                P.qkv, q_buf, k_buf, v_buf, batch, seq_len, n_heads, d_head);
        }
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // Attention: out is [B, H, S, Dh], stored into attn_out_bhsd scratch.
        ActT* attn_bhsd = P.ffn_h_pre;     // size B*S*F; we need B*H*S*Dh = B*S*HD
                                            // F = ffn_hidden >= d_model typically; ok if HD<=F.
        float scale = 1.0f / sqrtf((float)d_head);
        err = attention::attention_forward<ActT, /*kHeadDim*/32, /*kCausal*/false>(
            q_buf, k_buf, v_buf, attn_bhsd,
            reinterpret_cast<ActT*>(P.softmax_lse),
            batch, n_heads, seq_len, scale, stream);
        if (err != cudaSuccess) return err;

        // Merge heads back to [B, S, H*Dh] into attn_out
        {
            dim3 grid(seq_len, batch);
            int block = 128;
            merge_attn_out_kernel<ActT><<<grid, block, 0, stream>>>(
                attn_bhsd, P.attn_out, batch, seq_len, n_heads, d_head);
        }
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // Output projection: attn_out [M, HD] -> proj_out [M, D]
        err = launch_gemm_bias<ActT, WeightT>(
            P.attn_out, w.out_W(L), w.out_b(L), P.proj_out,
            M, d_model, HD, stream);
        if (err != cudaSuccess) return err;

        // Residual: ln1_in = pre_attn_in + proj_out
        err = launch_add<ActT>(P.pre_attn_in, P.proj_out, P.ln1_in, (size_t)M * d_model, stream);
        if (err != cudaSuccess) return err;

        // LayerNorm 1: ln1_out = LN(ln1_in)
        err = launch_layernorm_fwd<ActT, WeightT>(
            P.ln1_in, w.ln1_gamma(L), w.ln1_beta(L),
            P.ln1_out, P.ln1_mean, P.ln1_invstd,
            M, d_model, 1e-5f, stream);
        if (err != cudaSuccess) return err;

        // FFN up: ffn_h_pre = ff1_W @ ln1_out + ff1_b
        err = launch_gemm_bias<ActT, WeightT>(
            P.ln1_out, w.ff1_W(L), w.ff1_b(L), P.ffn_h_pre,
            M, ffn_hidden, d_model, stream);
        if (err != cudaSuccess) return err;

        // GELU
        err = launch_gelu_fwd<ActT>(P.ffn_h_pre, P.ffn_h, (size_t)M * ffn_hidden, stream);
        if (err != cudaSuccess) return err;

        // FFN down: ffn_out = ff2_W @ ffn_h + ff2_b
        err = launch_gemm_bias<ActT, WeightT>(
            P.ffn_h, w.ff2_W(L), w.ff2_b(L), P.ffn_out,
            M, d_model, ffn_hidden, stream);
        if (err != cudaSuccess) return err;

        // Residual: ln2_in = ln1_out + ffn_out
        err = launch_add<ActT>(P.ln1_out, P.ffn_out, P.ln2_in, (size_t)M * d_model, stream);
        if (err != cudaSuccess) return err;

        // LayerNorm 2: ln2_out = LN(ln2_in)
        err = launch_layernorm_fwd<ActT, WeightT>(
            P.ln2_in, w.ln2_gamma(L), w.ln2_beta(L),
            P.ln2_out, P.ln2_mean, P.ln2_invstd,
            M, d_model, 1e-5f, stream);
        if (err != cudaSuccess) return err;

        // Plumb to next layer's pre_attn_in
        if (L + 1 < n_layers) {
            auto Pn = A.layer(L + 1);
            cudaMemcpyAsync(Pn.pre_attn_in, P.ln2_out,
                            sizeof(ActT) * (size_t)M * d_model,
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Step 4: extract CLS (last layer ln2_out[:, 0, :])
    auto F = A.final_ptrs();
    ActT* last_ln2_out = A.layer(n_layers - 1).ln2_out;
    {
        int block = 128; if (block > d_model) block = d_model;
        cls_extract_fwd_kernel<ActT><<<batch, block, 0, stream>>>(
            last_ln2_out, F.final_in, batch, seq_len, d_model);
    }
    err = cudaGetLastError(); if (err != cudaSuccess) return err;

    // Step 5: final LayerNorm: final_out = LN(final_in)
    err = launch_layernorm_fwd<ActT, WeightT>(
        F.final_in, w.ln_final_gamma(), w.ln_final_beta(),
        F.final_out, F.final_mean, F.final_invstd,
        batch, d_model, 1e-5f, stream);
    if (err != cudaSuccess) return err;

    // Step 6: classification head: output = head_W @ final_out + head_b
    err = launch_gemm_bias<ActT, WeightT>(
        F.final_out, w.head_W(), w.head_b(), output,
        batch, n_classes, d_model, stream);
    return err;
}

// ════════════════════════════════════════════════════════════════════════
// Backward pass
// ════════════════════════════════════════════════════════════════════════
template <typename ActT, typename WeightT>
cudaError_t backward(
    const ActT* grad_output,         // [B, n_classes]
    const ActT* activations_saved,
    const WeightT* weights,
    ActT* grad_input,                // [B, num_patches, patch_dim] (or whatever input was)
    WeightT* grad_weights,
    int batch, int channels, int height, int width,
    int patch_size, int d_model, int n_heads, int d_head,
    int n_layers, int n_classes, int ffn_expansion,
    cudaStream_t stream
) {
    if (patch_size <= 0 || height % patch_size != 0 || width % patch_size != 0) {
        return cudaErrorInvalidValue;
    }
    int num_patches = (height / patch_size) * (width / patch_size);
    int patch_dim   = patch_size * patch_size * channels;
    int seq_len     = num_patches + 1;
    int ffn_hidden  = (int)(ffn_expansion * d_model);
    int HD          = n_heads * d_head;
    int M           = batch * seq_len;

    WeightLayout<WeightT> w{};
    w.base = weights; w.d_model = d_model; w.n_heads = n_heads; w.d_head = d_head;
    w.ffn_hidden = ffn_hidden; w.n_layers = n_layers; w.n_classes = n_classes;
    w.num_patches = num_patches; w.patch_dim = patch_dim; w.seq_len = seq_len;

    WeightLayout<WeightT> dw{};
    dw.base = grad_weights; dw.d_model = d_model; dw.n_heads = n_heads; dw.d_head = d_head;
    dw.ffn_hidden = ffn_hidden; dw.n_layers = n_layers; dw.n_classes = n_classes;
    dw.num_patches = num_patches; dw.patch_dim = patch_dim; dw.seq_len = seq_len;

    ActLayout<ActT> A{};
    A.base = const_cast<ActT*>(activations_saved); A.batch = batch; A.seq_len = seq_len;
    A.d_model = d_model; A.n_heads = n_heads; A.d_head = d_head; A.ffn_hidden = ffn_hidden;
    A.n_layers = n_layers; A.n_classes = n_classes;

    auto F = A.final_ptrs();

    // Allocate scratch for backward: grad arrays of various sizes. We
    // request them from a single device-side scratch via cudaMallocAsync.
    // Size budget (worst case):
    //   d_tokens  : B * S * D
    //   d_residual: B * S * D
    //   d_attn_out: B * S * HD
    //   d_attn_bhsd: B * H * S * Dh
    //   d_q,d_k,d_v: 3 * B * H * S * Dh
    //   d_ffn_h    : B * S * F
    //   d_cls      : B * D
    //   d_final_in : B * D
    // Total ~ B*S*(D + D + HD + ffn_hidden) + B*D.
    size_t bsd  = (size_t)batch * seq_len * d_model;
    size_t bsh  = (size_t)batch * seq_len * 3 * HD;
    size_t bsf  = (size_t)batch * seq_len * ffn_hidden;
    size_t bhsd = (size_t)batch * n_heads * seq_len * d_head;
    size_t bd   = (size_t)batch * d_model;

    size_t scratch_n = bsd*4 + bsh + bsf + bhsd*3 + bd*2;
    ActT* scratch = nullptr;
    cudaError_t err = cudaMallocAsync((void**)&scratch, scratch_n * sizeof(ActT), stream);
    if (err != cudaSuccess) return err;

    ActT* d_final_in = scratch;                            // [B, D]
    ActT* d_cls      = d_final_in  + bd;                   // [B, D]
    ActT* d_tokens   = d_cls       + bd;                   // [B, S, D]
    ActT* d_resid    = d_tokens    + bsd;                  // [B, S, D]
    ActT* d_attn_o   = d_resid     + bsd;                  // [B, S, HD] — bsd slots (HD<=D ok)
    ActT* d_attn_bhsd= d_attn_o    + bsd;                  // [B, H, S, Dh]
    ActT* d_q        = d_attn_bhsd + bhsd;                 // [B, H, S, Dh]
    ActT* d_k        = d_q         + bhsd;                 // [B, H, S, Dh]
    ActT* d_v        = d_k         + bhsd;                 // [B, H, S, Dh]
    ActT* d_ffn_h    = d_v         + bhsd;                 // [B, S, F]
    // d_q,d_k,d_v contiguous (3*bhsd = bsh) doubles as merged d_qkv buffer.
    (void)bsh;

    // Body returns cudaError_t; cleanup happens after the lambda.
    auto body = [&]() -> cudaError_t {

    // Step 1: head_W backward
    //   grad_final_out = grad_output @ head_W
    //   d_head_W = grad_output^T @ final_out
    //   d_head_b = sum_b grad_output
    err = launch_gemm_grad_input<ActT, WeightT>(
        grad_output, w.head_W(), d_final_in, batch, n_classes, d_model, stream);
    if (err != cudaSuccess) return err;
    err = launch_gemm_grad_weight<ActT, WeightT>(
        grad_output, F.final_out,
        const_cast<WeightT*>(dw.head_W()), const_cast<WeightT*>(dw.head_b()),
        batch, n_classes, d_model, stream);
    if (err != cudaSuccess) return err;

    // Step 2: final LN backward
    //   d_cls = LN_bwd(d_final_in)
    //   d_ln_final_gamma, d_ln_final_beta accumulate
    err = launch_layernorm_bwd<ActT, WeightT>(
        F.final_in, d_final_in, w.ln_final_gamma(),
        F.final_mean, F.final_invstd,
        d_cls, batch, d_model, stream);
    if (err != cudaSuccess) return err;
    err = launch_layernorm_grad_param<ActT, WeightT>(
        F.final_in, d_final_in, F.final_mean, F.final_invstd,
        const_cast<WeightT*>(dw.ln_final_gamma()),
        const_cast<WeightT*>(dw.ln_final_beta()),
        batch, d_model, stream);
    if (err != cudaSuccess) return err;

    // Step 3: scatter d_cls to d_tokens at last layer (s=0); zeros elsewhere
    {
        dim3 grid(batch, seq_len);
        int block = 128; if (block > d_model) block = d_model;
        cls_scatter_bwd_kernel<ActT><<<grid, block, 0, stream>>>(
            d_cls, d_tokens, batch, seq_len, d_model);
    }
    err = cudaGetLastError(); if (err != cudaSuccess) return err;

    // Step 4: backward through layers in reverse
    for (int L = n_layers - 1; L >= 0; L--) {
        auto P = A.layer(L);

        // d_tokens currently holds d(ln2_out_L). LN2 backward:
        //   d_ln2_in <- LN_bwd(d_tokens, ln2_in, ln2_mean, ln2_invstd, ln2_gamma)
        //   d_ln2_gamma, d_ln2_beta accumulated
        ActT* d_ln2_in_buf = d_resid;  // [B, S, D]
        err = launch_layernorm_bwd<ActT, WeightT>(
            P.ln2_in, d_tokens, w.ln2_gamma(L),
            P.ln2_mean, P.ln2_invstd,
            d_ln2_in_buf, M, d_model, stream);
        if (err != cudaSuccess) return err;
        err = launch_layernorm_grad_param<ActT, WeightT>(
            P.ln2_in, d_tokens, P.ln2_mean, P.ln2_invstd,
            const_cast<WeightT*>(dw.ln2_gamma(L)),
            const_cast<WeightT*>(dw.ln2_beta(L)),
            M, d_model, stream);
        if (err != cudaSuccess) return err;

        // Residual split: ln2_in = ln1_out + ffn_out
        //   d_ln1_out_residual = d_ln2_in_buf  (one path)
        //   d_ffn_out          = d_ln2_in_buf  (other path)
        // We need both. d_ln1_out_residual feeds back into the LN1 path,
        // d_ffn_out goes into the FFN backward chain.
        // ffn down: ffn_out = ff2_W @ ffn_h + ff2_b; backprop:
        //   d_ffn_h = d_ffn_out @ ff2_W
        //   d_ff2_W = d_ffn_out^T @ ffn_h
        //   d_ff2_b = sum d_ffn_out
        err = launch_gemm_grad_input<ActT, WeightT>(
            d_ln2_in_buf, w.ff2_W(L), d_ffn_h, M, d_model, ffn_hidden, stream);
        if (err != cudaSuccess) return err;
        err = launch_gemm_grad_weight<ActT, WeightT>(
            d_ln2_in_buf, P.ffn_h,
            const_cast<WeightT*>(dw.ff2_W(L)),
            const_cast<WeightT*>(dw.ff2_b(L)),
            M, d_model, ffn_hidden, stream);
        if (err != cudaSuccess) return err;

        // GELU backward: d_ffn_h_pre = d_ffn_h * gelu'(ffn_h_pre)
        ActT* d_ffn_h_pre = d_ffn_h;  // in-place
        err = launch_gelu_bwd<ActT>(P.ffn_h_pre, d_ffn_h, d_ffn_h_pre, (size_t)M * ffn_hidden, stream);
        if (err != cudaSuccess) return err;

        // FFN up: ffn_h_pre = ff1_W @ ln1_out + ff1_b; backprop:
        //   d_ln1_out (FFN path) = d_ffn_h_pre @ ff1_W
        //   d_ff1_W = d_ffn_h_pre^T @ ln1_out
        //   d_ff1_b = sum d_ffn_h_pre
        ActT* d_ln1_out_ffn = d_attn_o;  // [B, S, HD] alias [B, S, D]; reuse since HD<=D ok or use a separate slot.
        // safety: we need a buffer of size bsd; d_attn_o has bsd slots.
        err = launch_gemm_grad_input<ActT, WeightT>(
            d_ffn_h_pre, w.ff1_W(L), d_ln1_out_ffn, M, ffn_hidden, d_model, stream);
        if (err != cudaSuccess) return err;
        err = launch_gemm_grad_weight<ActT, WeightT>(
            d_ffn_h_pre, P.ln1_out,
            const_cast<WeightT*>(dw.ff1_W(L)),
            const_cast<WeightT*>(dw.ff1_b(L)),
            M, ffn_hidden, d_model, stream);
        if (err != cudaSuccess) return err;

        // Combine: d_ln1_out = d_ln1_out_ffn + d_ln2_in_buf (residual path of ln2_in)
        err = launch_add<ActT>(d_ln1_out_ffn, d_ln2_in_buf, d_ln1_out_ffn, (size_t)M * d_model, stream);
        if (err != cudaSuccess) return err;
        // d_ln1_out_ffn now holds d_ln1_out (post-LN1 grad)

        // LN1 backward:  ln1_out = LN(ln1_in)
        //   d_ln1_in <- LN_bwd(d_ln1_out, ln1_in, ln1_mean, ln1_invstd, ln1_gamma)
        ActT* d_ln1_in_buf = d_ln2_in_buf;  // reuse
        err = launch_layernorm_bwd<ActT, WeightT>(
            P.ln1_in, d_ln1_out_ffn, w.ln1_gamma(L),
            P.ln1_mean, P.ln1_invstd,
            d_ln1_in_buf, M, d_model, stream);
        if (err != cudaSuccess) return err;
        err = launch_layernorm_grad_param<ActT, WeightT>(
            P.ln1_in, d_ln1_out_ffn, P.ln1_mean, P.ln1_invstd,
            const_cast<WeightT*>(dw.ln1_gamma(L)),
            const_cast<WeightT*>(dw.ln1_beta(L)),
            M, d_model, stream);
        if (err != cudaSuccess) return err;

        // Residual: ln1_in = pre_attn_in + proj_out
        //   d_pre_attn_in (res) = d_ln1_in_buf
        //   d_proj_out         = d_ln1_in_buf
        // Output projection: proj_out = out_W @ attn_out + out_b
        //   d_attn_out = d_proj_out @ out_W
        //   d_out_W = d_proj_out^T @ attn_out
        //   d_out_b = sum d_proj_out
        ActT* d_attn_out_bsd = d_attn_o;  // reuse: write d_attn_out [B,S,HD]
        err = launch_gemm_grad_input<ActT, WeightT>(
            d_ln1_in_buf, w.out_W(L), d_attn_out_bsd, M, d_model, HD, stream);
        if (err != cudaSuccess) return err;
        err = launch_gemm_grad_weight<ActT, WeightT>(
            d_ln1_in_buf, P.attn_out,
            const_cast<WeightT*>(dw.out_W(L)),
            const_cast<WeightT*>(dw.out_b(L)),
            M, d_model, HD, stream);
        if (err != cudaSuccess) return err;

        // Convert d_attn_out [B,S,HD] -> [B,H,S,Dh] in d_attn_bhsd
        {
            dim3 grid(seq_len, batch);
            int block = 128;
            unmerge_attn_out_kernel<ActT><<<grid, block, 0, stream>>>(
                d_attn_out_bsd, d_attn_bhsd, batch, seq_len, n_heads, d_head);
        }
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // Reconstitute Q/K/V from saved qkv (split again)
        // We use d_v slots only briefly here. Since attention_backward needs
        // q,k,v as inputs in [B,H,S,Dh] form, we recompute the split.
        ActT* q_bhsd = d_q;       // [B, H, S, Dh]
        ActT* k_bhsd = d_k;
        ActT* v_bhsd = d_v;
        {
            dim3 grid(seq_len, batch);
            int block = 128;
            split_qkv_kernel<ActT><<<grid, block, 0, stream>>>(
                P.qkv, q_bhsd, k_bhsd, v_bhsd, batch, seq_len, n_heads, d_head);
        }
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // attn_backward writes dq, dk, dv into the same scratch (in place).
        // We need separate output buffers for dq,dk,dv. Use d_ffn_h scratch
        // (ffn_hidden-sized) — we have bsf >= 3*bhsd if F >= 3*HD. For
        // safety, we use three separate aliases by partitioning d_ffn_h.
        ActT* dq = d_ffn_h;
        ActT* dk = d_ffn_h + bhsd;
        ActT* dv = d_ffn_h + 2 * bhsd;
        // Validate: 3*bhsd <= bsf? bsf = M*F = M*ffn_hidden; 3*bhsd = 3*B*H*S*Dh = 3*M*HD.
        // So we need 3*HD <= ffn_hidden. With ffn_expansion>=3 and HD==d_model, ok.
        // For ffn_expansion==4 (PyTorch default), ffn_hidden=4*d_model >= 3*HD when HD<=d_model.
        // (HD==d_model in standard ViT.) OK.

        float scale = 1.0f / sqrtf((float)d_head);
        err = attention::attention_backward<ActT, /*kHeadDim*/32, /*kCausal*/false>(
            d_attn_bhsd, q_bhsd, k_bhsd, v_bhsd,
            /*out=*/(const ActT*)nullptr,    // unused by smem kernel
            (const ActT*)P.softmax_lse,
            dq, dk, dv,
            batch, n_heads, seq_len, scale, stream);
        if (err != cudaSuccess) return err;

        // Merge dQ/dK/dV [B,H,S,Dh] -> dqkv [B,S,3*HD]
        ActT* d_qkv_buf = d_attn_bhsd;  // bhsd*3 = bsh ; we need bsh slots
        // d_attn_bhsd has bhsd slots, not enough for bsh = 3*bhsd. Use a
        // dedicated aliased region of scratch instead. We have d_q,d_k,d_v
        // contiguous (3*bhsd = bsh slots). Reuse them as the output region.
        d_qkv_buf = d_q;  // contiguous 3*bhsd
        {
            dim3 grid(seq_len, batch);
            int block = 128;
            merge_qkv_grads_kernel<ActT><<<grid, block, 0, stream>>>(
                dq, dk, dv, d_qkv_buf, batch, seq_len, n_heads, d_head);
        }
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // QKV projection backward:
        //   d_pre_attn_in (attn path) = d_qkv @ qkv_W
        //   d_qkv_W = d_qkv^T @ pre_attn_in
        //   d_qkv_b = sum d_qkv
        ActT* d_pre_in_attn = d_attn_bhsd;  // reuse [B,S,D]; but bhsd<bsd in general.
        // Need bsd; use d_attn_o region (bsd slots, currently unused after this point).
        d_pre_in_attn = d_attn_o;
        err = launch_gemm_grad_input<ActT, WeightT>(
            d_qkv_buf, w.qkv_W(L), d_pre_in_attn, M, 3 * HD, d_model, stream);
        if (err != cudaSuccess) return err;
        err = launch_gemm_grad_weight<ActT, WeightT>(
            d_qkv_buf, P.pre_attn_in,
            const_cast<WeightT*>(dw.qkv_W(L)),
            const_cast<WeightT*>(dw.qkv_b(L)),
            M, 3 * HD, d_model, stream);
        if (err != cudaSuccess) return err;

        // Total d_pre_attn_in = d_pre_in_attn (attn path) + d_ln1_in_buf (residual path of ln1_in)
        err = launch_add<ActT>(d_pre_in_attn, d_ln1_in_buf, d_tokens, (size_t)M * d_model, stream);
        if (err != cudaSuccess) return err;
        // d_tokens now holds gradient w.r.t. layer L's pre_attn_in
        // (= layer L-1's ln2_out, or for L==0, the post-CLS-pos tokens)
    }

    // Step 5: backward through CLS+pos: d_patches_proj, d_cls_token, d_pos_embed
    {
        int block = 128; if (block > d_model) block = d_model;
        // We use proj_out scratch from layer 0 as d_patches_proj? No — that
        // scratch was activations memory; we need a fresh buffer of size
        // [B, num_patches, D]. Reuse d_resid (bsd slots ≥ batch*num_patches*D).
        ActT* d_patches_proj = d_resid;
        cls_pos_bwd_kernel<ActT, WeightT><<<seq_len, block, 0, stream>>>(
            d_tokens, d_patches_proj,
            const_cast<WeightT*>(dw.cls_token()),
            const_cast<WeightT*>(dw.pos_embed()),
            batch, num_patches, seq_len, d_model);
        err = cudaGetLastError(); if (err != cudaSuccess) return err;

        // Step 6: patch projection backward
        //   patches_proj = patch_W @ input + patch_b
        //   d_input    = d_patches_proj @ patch_W
        //   d_patch_W  = d_patches_proj^T @ input
        //   d_patch_b  = sum d_patches_proj
        // The activations buffer doesn't save the input verbatim; but for
        // grad_input + grad_patch_W computation we need it. The forward
        // writes patches_proj into layer-0 proj_out scratch (overwritten).
        // To avoid requiring the saved input, we reload it from grad_input's
        // companion: assume the binding passes the original `input` again
        // through the activations region (caller responsibility). Here we
        // accept that the caller passes `input` as the first batch*num_patches*patch_dim
        // ActT slots of activations_saved. Rather than require that, we
        // expect the binding to provide both `activations_saved` and a
        // reconstructed input via grad_input being pre-zeroed and the
        // caller having stashed the input separately. To keep this header
        // self-contained, we reconstruct `input` from a reserved tail of
        // the activation buffer (caller MUST place input there).
        //
        // Implementation choice: we treat the FINAL section of the
        // activations buffer as a saved `input` region, written by the
        // caller. The size is batch*num_patches*patch_dim ActT elements,
        // appended by the caller after the standard activations.
        const ActT* saved_input = reinterpret_cast<const ActT*>(F.final_out + bd);

        err = launch_gemm_grad_input<ActT, WeightT>(
            d_patches_proj, w.patch_W(), grad_input,
            batch * num_patches, d_model, patch_dim, stream);
        if (err != cudaSuccess) return err;
        err = launch_gemm_grad_weight<ActT, WeightT>(
            d_patches_proj, saved_input,
            const_cast<WeightT*>(dw.patch_W()),
            const_cast<WeightT*>(dw.patch_b()),
            batch * num_patches, d_model, patch_dim, stream);
        if (err != cudaSuccess) return err;
    }
        return cudaSuccess;
    };

    err = body();
    cudaFreeAsync(scratch, stream);
    return err;
}

// ─── Entry-point: standalone patch projection (used by binding test) ───
template <typename ActT, typename WeightT>
inline cudaError_t patch_project(
    const ActT* input,         // [B*num_patches, patch_dim]
    const WeightT* weight,     // [d_model, patch_dim]
    const WeightT* bias,       // [d_model] or nullptr
    ActT* output,              // [B*num_patches, d_model]
    int batch, int num_patches, int patch_dim, int d_model,
    cudaStream_t stream
) {
    return launch_gemm_bias<ActT, WeightT>(
        input, weight, bias, output,
        batch * num_patches, d_model, patch_dim, stream);
}

}}}}  // namespace sg::sm90::models::vit
